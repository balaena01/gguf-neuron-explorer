"""ニューロン間の配線(connections)を重みだけから求める。別冊 UI 設計 §8.4・§8.5・§4.7。

層 i のニューロン A の書き込み方向 v_A = W_down_i[:, A] が、層 j(> i)のニューロン B の活性に単独で与える寄与を
    P[B, A] = (W_gate_j[B, :] · x_A) · (W_up_j[B, :] · x_A),   x_A = γ_j ⊙ v_A / rms_j
で近似する(SwiGLU の silu(g)·u は g が小さいとき g·u / 2)。γ_j は層 j の FFN の前の RMSNorm の重み(GGUF には
「1 + w」が畳み込まれた実効値が入っている、verification.md §4)、rms_j はそこでの残差の RMS(残差ノルムの統計があれば
ρ / √d、無ければ 1)。rms_j で割るのは、深い層ほど残差が大きく同じ v_A の効きが小さくなるのを、層をまたいで比べるため(F56)。

A の下流: 全層の B について |P| を z 化し、上位 k。B の上流: 全層の A について |P| を z 化し、上位 k。符号 sign(P) も持つ。
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .steer import GneError
from .weights import tensor_to_f32

FFN_NORM_NAMES = ("ffn_norm.weight", "post_attention_norm.weight")


def ffn_norm_weight(weights, L: int) -> np.ndarray:
    """層 L の FFN の前の RMSNorm の γ。llama 系は ffn_norm、Qwen3.5 系は post_attention_norm という名前。"""
    for nm in FFN_NORM_NAMES:
        name = f"blk.{L}.{nm}"
        if weights.has_tensor(name):
            return np.asarray(tensor_to_f32(weights.tensor(name)), dtype=np.float32).reshape(-1)
    raise GneError("ARCH_UNSUPPORTED", f"層 {L} の FFN の前の norm({' / '.join(FFN_NORM_NAMES)})がありません")


def dense_layers(weights) -> list[int]:
    return [L for L in range(weights.meta.n_layer)
            if all(weights.has_tensor(f"blk.{L}.ffn_{x}.weight") for x in ("gate", "up", "down"))]


class _TopK:
    """行ごとの上位 k(値・相手の層・相手の番号・符号)を、チャンクを足すたびに混ぜる。"""

    def __init__(self, n: int, k: int):
        self.k = k
        self.val = np.full((n, k), -np.inf, np.float32)
        self.lay = np.full((n, k), -1, np.int16)
        self.idx = np.full((n, k), -1, np.int32)
        self.sgn = np.zeros((n, k), np.int8)
        self.sum = np.zeros(n, np.float64)
        self.sq = np.zeros(n, np.float64)
        self.cnt = np.zeros(n, np.int64)

    def add(self, rows: slice, absP: np.ndarray, sgnP: np.ndarray, layer: int, offset: int) -> None:
        """absP: (行, 候補)。候補の番号は offset から。"""
        k = min(self.k, absP.shape[1])
        self.sum[rows] += absP.sum(1, dtype=np.float64)
        self.sq[rows] += np.square(absP, dtype=np.float64).sum(1)
        self.cnt[rows] += absP.shape[1]
        part = np.argpartition(-absP, k - 1, axis=1)[:, :k]
        v = np.take_along_axis(absP, part, 1)
        s = np.take_along_axis(sgnP, part, 1)
        allv = np.concatenate([self.val[rows], v], axis=1)
        alll = np.concatenate([self.lay[rows], np.full_like(part, layer, dtype=np.int16)], axis=1)
        alli = np.concatenate([self.idx[rows], (part + offset).astype(np.int32)], axis=1)
        alls = np.concatenate([self.sgn[rows], s.astype(np.int8)], axis=1)
        keep = np.argpartition(-allv, self.k - 1, axis=1)[:, :self.k]
        self.val[rows] = np.take_along_axis(allv, keep, 1)
        self.lay[rows] = np.take_along_axis(alll, keep, 1)
        self.idx[rows] = np.take_along_axis(alli, keep, 1)
        self.sgn[rows] = np.take_along_axis(alls, keep, 1)

    def finish(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """z 化して z の大きい順に並べる。候補が無い枠は層 −1。"""
        n = np.maximum(self.cnt, 1)
        mean = self.sum / n
        std = np.sqrt(np.maximum(self.sq / n - mean ** 2, 1e-30))
        z = ((self.val - mean[:, None]) / std[:, None]).astype(np.float32)
        z[~np.isfinite(self.val)] = np.nan
        o = np.argsort(-np.nan_to_num(z, nan=-np.inf), axis=1)
        take = lambda a: np.take_along_axis(a, o, 1)  # noqa: E731
        lay = take(self.lay)
        lay[np.isnan(take(z))] = -1
        return lay, take(self.idx), take(z), take(self.sgn)


class _DownCache:
    """層 i の W_down(f16)の置き場所。window があれば、まだ使う層(j − i ≤ window)だけを持つ。window が無く全層を持つと
    cap_bytes を超えるときは、古いものから捨てて要るときに読み直す(70B で全層を持つと数十 GB になるため、CHANGE-03 C)。"""

    def __init__(self, weights, cap_bytes: float):
        from collections import OrderedDict
        self.w, self.cap, self.d = weights, cap_bytes, OrderedDict()
        self.reloads = 0

    def get(self, i: int) -> np.ndarray:
        if i in self.d:
            self.d.move_to_end(i)
            return self.d[i]
        v = self.w.down_matrix(i).astype(np.float16)
        self.reloads += 1
        self.d[i] = v
        while sum(x.nbytes for x in self.d.values()) > self.cap and len(self.d) > 1:
            self.d.popitem(last=False)
        return v

    def drop_below(self, lo: int) -> None:
        for i in [i for i in self.d if i < lo]:
            del self.d[i]


def compute_connections(weights, *, k: int = 24, window: int | None = None, rho: dict[int, float] | None = None,
                        chunk: int = 1024, cache_gb: float = 8.0, progress=None) -> dict:
    """全層ペア(window があれば |j − i| ≤ window)の connections。返り値は保存する配列と meta。
    W_down は window の中の層だけ持つ(window が無ければ cache_gb まで)。gate / up は層 j ごとに読んで捨てる。"""
    import resource
    m = weights.meta
    Ls = dense_layers(weights)
    if len(Ls) < 2:
        raise GneError("ARCH_UNSUPPORTED", "dense FFN の層が 2 つ以上必要です(MoE の層は connections の対象外)")
    d = m.n_embd
    nmax = max(m.n_ff_at(L) for L in Ls)
    pairs = [(i, j) for jj, j in enumerate(Ls) for i in Ls[:jj] if window is None or j - i <= window]
    down = {i: _TopK(m.n_ff_at(i), k) for i in Ls}
    up = {j: _TopK(m.n_ff_at(j), k) for j in Ls}
    V = _DownCache(weights, cache_gb * 1e9)                              # 書き込み方向(f16)
    t0 = time.perf_counter()
    done = 0
    for jj, j in enumerate(Ls):
        srcs = [i for i in Ls[:jj] if window is None or j - i <= window]
        if window is not None:
            V.drop_below(j - window)                                     # もう使わない層を捨てる
        if not srcs:
            continue
        g = ffn_norm_weight(weights, j)
        scale = 1.0
        if rho:
            r = rho.get(j - 1, rho.get(j))                                   # 層 j の FFN に入る残差 ≈ 層 j−1 の出力
            if r:
                scale = (np.sqrt(d) / r) ** 2                                 # P は x の 2 次なので rms の 2 乗で割る
        Gj = tensor_to_f32(weights.tensor(f"blk.{j}.ffn_gate.weight")) * g[None, :]
        Uj = tensor_to_f32(weights.tensor(f"blk.{j}.ffn_up.weight")) * g[None, :]
        if Gj.shape != (m.n_ff_at(j), d):
            raise GneError("ARCH_UNSUPPORTED", f"blk.{j}.ffn_gate の形 {Gj.shape} が想定外です")
        for i in srcs:
            Vi = V.get(i)
            for a0 in range(0, Vi.shape[1], chunk):
                X = Vi[:, a0:a0 + chunk].astype(np.float32)
                P = (Gj @ X) * (Uj @ X)                                       # (n_ff_j, c)
                if scale != 1.0:
                    P *= scale
                aP, sP = np.abs(P), np.sign(P).astype(np.int8)
                down[i].add(slice(a0, a0 + X.shape[1]), aP.T, sP.T, j, 0)     # 列 A ごとに、層 j の全 B から
                up[j].add(slice(None), aP, sP, i, a0)                          # 行 B ごとに、層 i のこのチャンクの A から
            done += 1
            if progress:
                progress(done, len(pairs), f"connections L{i}→L{j}")
        del Gj, Uj
    L = m.n_layer
    out: dict[str, np.ndarray] = {}
    for name, tk in (("down", down), ("up", up)):
        lay = np.full((L, nmax, k), -1, np.int16)
        idx = np.full((L, nmax, k), -1, np.int32)
        z = np.full((L, nmax, k), np.nan, np.float16)
        sgn = np.zeros((L, nmax, k), np.int8)
        for Lx, t in tk.items():
            a, b, c, s = t.finish()
            n = a.shape[0]
            lay[Lx, :n], idx[Lx, :n], z[Lx, :n], sgn[Lx, :n] = a, b, c.astype(np.float16), s
        out.update({f"{name}_layer": lay, f"{name}_n": idx, f"{name}_z": z, f"{name}_sign": sgn})
    meta = {"model_id": m.model_id, "k": k, "window": window, "rho_scaled": bool(rho), "layers": Ls,
            "n_pairs": len(pairs), "seconds": time.perf_counter() - t0, "formula": "P = (W_gate·x)(W_up·x), x = γ⊙v_A/rms",
            "down_reads": V.reloads, "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}
    return {"arrays": out, "meta": meta}


def save_connections(path: Path, result: dict) -> None:
    tmp = Path(str(path) + ".tmp.npz")
    np.savez_compressed(tmp, meta=json.dumps(result["meta"]), **result["arrays"])
    tmp.replace(path)


class Connections:
    """保存した connections を読む。"""

    def __init__(self, path: Path):
        z = np.load(path)
        self.meta = json.loads(str(z["meta"]))
        for k in ("down", "up"):
            for f in ("layer", "n", "z", "sign"):
                setattr(self, f"{k}_{f}", z[f"{k}_{f}"])

    def edges(self, L: int, n: int, way: str, k: int | None = None, zmin: float = -np.inf) -> list[tuple[int, int, float, int]]:
        """(相手の層, 相手の番号, z, 符号)を z の大きい順に。way: down(書き込む先)/ up(駆動するもの)。"""
        lay, idx, z, sgn = (getattr(self, f"{way}_{f}")[L, n] for f in ("layer", "n", "z", "sign"))
        out = [(int(a), int(b), float(c), int(s)) for a, b, c, s in zip(lay, idx, z, sgn, strict=True)
               if a >= 0 and np.isfinite(c) and c >= zmin]
        return out[:k] if k else out

    def web(self, L: int, n: int, *, hops: int = 2, first: int = 8, k: int = 4, zmin: float = 2.0,
            cap: int = 20000) -> np.ndarray:
        """多段の円錐(§8.5)。下流と上流へ hops 段。1 段目は first 本、以降は k 本。符号は経路で掛け合わせる。
        返り値 (m, 7) int32: 元の層・元の番号・先の層・先の番号・round(10·z)·辺の符号・段(下流 +h、上流 −h)・経路の符号
        (開いたニューロンからこの辺までの符号の積)。辺の向きは常に「元が先を押す」。"""
        rows: list[tuple[int, ...]] = []
        for way, sign_hop in (("down", 1), ("up", -1)):
            frontier = [(L, n, 1)]
            seen = {(L, n)}
            for h in range(1, hops + 1):
                nxt = []
                for (a, b, ps) in frontier:
                    for (c, d, z, s) in self.edges(a, b, way, first if h == 1 else k, zmin):
                        if len(rows) >= cap:
                            break
                        src, dst = ((a, b), (c, d)) if way == "down" else ((c, d), (a, b))
                        rows.append((*src, *dst, int(round(10 * z)) * s, sign_hop * h, ps * s))
                        if (c, d) not in seen:
                            seen.add((c, d))
                            nxt.append((c, d, ps * s))
                frontier = nxt
        return np.array(rows, np.int32).reshape(-1, 7)


def lens_tokens(data, L: int, n: int, pole: str, k: int = 6) -> list[int]:
    """ニューロンの lens の上位語(語番号)。pole は "+" か "-"。steer lens があればそれ、無ければ logit lens。"""
    sl = data.steer_lens(L)
    if sl is not None and sl.done[n]:
        return [int(t) for t in (sl.up_ids if pole == "+" else sl.down_ids)[n][:k] if t >= 0]
    ll = data.logit_lens(L)
    if ll is not None:
        hit = np.nonzero(ll.neurons == n)[0]
        if hit.size:
            return [int(t) for t in (ll.up_ids if pole == "+" else ll.down_ids)[int(hit[0])][:k] if t >= 0]
    return []


def _word_keys(data) -> tuple[np.ndarray, list[str]]:
    """語番号 → まとめた語の番号(大文字・語頭語末の空白の違いだけの語は同じ番号。バイト単位の語と空の語は −1)。"""
    cache = getattr(data, "_word_keys_cache", None)
    if cache is not None:
        return cache
    voc = data.vocab_text()
    keys: dict[str, int] = {}
    ids = np.full(len(voc), -1, np.int32)
    for i, t in enumerate(voc):
        k = t.strip().casefold()
        if k and not t.startswith("<0x"):
            ids[i] = keys.setdefault(k, len(keys))
    names = [""] * len(keys)
    for k, j in keys.items():
        names[j] = k
    data._word_keys_cache = (ids, names)
    return ids, names


def _layer_word_freq(data, L: int, n_keys: int, keymap: np.ndarray) -> np.ndarray:
    """層 L で、lens(± 上位 6 語)にその語を持つニューロンの割合(まとめた語ごと)。"""
    cache = data.__dict__.setdefault("_word_freq_cache", {})
    if L in cache:
        return cache[L]
    cnt = np.zeros(n_keys, np.float64)
    n = 0
    for arr in _lens_arrays(data, L):
        k = np.where(arr >= 0, keymap[np.maximum(arr, 0)], -1)
        k = np.sort(k, axis=1)
        first = np.ones_like(k, bool)
        first[:, 1:] = k[:, 1:] != k[:, :-1]                               # ニューロンごとに 1 回だけ数える
        v = k[first & (k >= 0)]
        cnt += np.bincount(v, minlength=n_keys)[:n_keys]
        n += arr.shape[0]
    f = cnt / max(n, 1)
    cache[L] = f
    return f


def web_summary(data, conn: Connections, L: int, n: int, edges: np.ndarray, *, top: int = 12) -> dict:
    """web の語の要約(§4.7)。まとめた語ごとに「その語を lens に持つ異なるニューロンの数」を数え、
    層の中でその語を持つニューロンの割合から見た期待値に対する比(lift)で並べる。
    toward = 下流が書く語、away = 下流で抑えられる語(押される極の反対)、fed_by / held_by = 上流で + / − に駆動する語。
    下流の 2 段目以降と上流は、途中・相手のニューロンが + 側で光ると仮定して符号を掛ける(F61)。"""
    keymap, names = _word_keys(data)
    voc = data.vocab_text()
    bags: dict[str, dict[int, set]] = {k: {} for k in ("toward", "away", "fed_by", "held_by")}
    shown: dict[int, Counter] = {}
    members: dict[str, list] = {k: [] for k in bags}
    seen = set()
    for row in edges:
        sl, sn, dl, dn, _z, hop, ps = (int(x) for x in row)
        downstream = hop > 0
        other = (dl, dn) if downstream else (sl, sn)
        if other in seen or other == (L, n):
            continue
        seen.add(other)
        if downstream:
            parts = (("toward", "+" if ps > 0 else "-"), ("away", "-" if ps > 0 else "+"))
        else:
            parts = (("fed_by" if ps > 0 else "held_by", "+"),)
        for key, pole in parts:
            members[key].append(other)
            for t in lens_tokens(data, *other, pole):
                k = int(keymap[t])
                if k < 0:
                    continue
                bags[key].setdefault(k, set()).add(other)
                shown.setdefault(k, Counter())[voc[t]] += 1
    out = {}
    for key, bag in bags.items():
        rows = []
        for k, who in bag.items():
            exp = sum(_layer_word_freq(data, Lx, len(names), keymap)[k] for Lx, _ in members[key])
            lift = len(who) / max(exp, 1.0 / max(len(members[key]), 1) / 10)
            rows.append((len(who) >= 2, lift, len(who), k))
        rows.sort(key=lambda r: (not r[0], -r[1], -r[2]))                  # 2 ニューロン以上の語を先に、lift の順
        out[key] = [{"token": shown[k].most_common(1)[0][0], "lift": round(float(lf), 1), "count": c} for _, lf, c, k in rows[:top]]
    both = {x["token"].strip().casefold() for x in out["toward"]} & {x["token"].strip().casefold() for x in out["away"]}
    for key in ("toward", "away"):
        for x in out[key]:
            x["both"] = x["token"].strip().casefold() in both
    out["n_neurons"] = len(seen)
    return out


def _lens_arrays(data, L: int) -> list[np.ndarray]:
    out = []
    sl = data.steer_lens(L)
    if sl is not None:
        out += [sl.up_ids[sl.done, :6], sl.down_ids[sl.done, :6]]
    ll = data.logit_lens(L)
    if ll is not None:
        out += [ll.up_ids[:, :6], ll.down_ids[:, :6]]
    return out
