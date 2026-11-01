"""ヘッドの方向成分だけを変える介入(CHANGE-06 A・E・F、手法の説明は docs/head-component-steering.md)。

ヘッド h の書き込み行列(attention の層は attn_output、線形注意の層は ssm_out)の列ブロック blk_h を

    W[:, blk_h] → W[:, blk_h] − s_h · r̂ r̂ᵀ W[:, blk_h]

に書き換える。r̂ は方向の単位ベクトル(direction_layer: own = そのヘッドがある層の r_L、base = 方向セットの基準層の r)。
ヘッドが残差に書くベクトルのうち r̂ に沿った成分だけが (1 − s) 倍になり、直交成分は変わらない(s = 1 で方向成分が消え、
2 で反転、s < 0 で強める)。ヘッドの gain(列ブロックを k 倍、CHANGE-04 D)との違いは直交成分に触らないこと。

LoRA: 同じ層・同じ r̂ のヘッドは階数 1 に畳める。

    A = Σ_h s_h · (r̂ᵀ W のうちヘッド h の列ブロックだけを残したもの)   (1, n_in)
    B = −r̂                                                               (n_embd, 1)
    W x + B (A x) = W x − r̂ Σ_h s_h · r̂ᵀ W[:, blk_h] x_h

層の中で方向が混ざる(ヘッドごとに違う方向)ときは方向ごとに 1 本で、階数は層の中の方向の数(最大でヘッド数)。
焼き込みは列方向かつ単純な倍率ではないのでスケールパッチ(CHANGE-05 D)では書けず、書き換えるテンソルだけ量子化し直す。
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .steer import GneError, Recipe, Unit

DirSource = Callable[[str], object]          # 方向の名前 → DirectionSet


@dataclass(frozen=True)
class ProjHead:
    layer: int
    head: int
    s: float
    direction: str

    def spec(self) -> str:
        return f"{self.layer}:H{self.head}:s{self.s:.4g}@{self.direction}"


def proj_heads(recipe: Recipe) -> list[ProjHead]:
    """レシピの proj のユニット → ProjHead(方向の省略はレシピの default_direction)。"""
    out = []
    for u in recipe.proj_units:
        d = u.direction or recipe.default_direction
        if not d:
            raise GneError("BAD_RECIPE", f"{u.label()}: 方向がありません(ユニットの direction かレシピの default_direction)")
        out.append(ProjHead(int(u.layer), int(u.head), float(u.k), str(d)))
    return out


def heads_to_units(heads: list[ProjHead], default_direction: str | None = None) -> list[Unit]:
    return [Unit(p.layer, None, 0.0, "head", mode="proj", k=float(p.s), head=p.head,
                 direction=None if p.direction == default_direction else p.direction) for p in heads]


def parse_proj_specs(specs: list[str], default_direction: str | None = None) -> list[ProjHead]:
    """"L:Hh:sS@NAME" の列 → ProjHead。"""
    from .steer import parse_unit
    out = []
    for sp in specs:
        u = parse_unit(sp)
        if u.mode != "proj":
            raise GneError("BAD_UNIT", f"{sp!r} は方向成分の射影(L:Hh:sS@NAME)ではありません")
        d = u.direction or default_direction
        if not d:
            raise GneError("BAD_UNIT", f"{sp!r}: 方向がありません(@NAME か --direction)")
        out.append(ProjHead(u.layer, u.head, float(u.k), d))
    return out


def head_direction(ds, L: int, direction_layer: str = "own") -> tuple[np.ndarray, int]:
    """ヘッドのある層 L に当てる方向の単位ベクトルと、それを測った層。own は L の方向(無ければ L より前で一番近い層)、
    base は方向セットの基準層(gne direction score か --select で決めた層)。"""
    from .direction import base_layer_of
    if direction_layer == "base":
        b = base_layer_of(ds)
        if b is None:
            raise GneError("BAD_DIRECTION", f"方向 {ds.name} の基準層が決まっていません",
                           f"gne direction score --name {ds.name} --contrast <対照集合> で記録するか、direction_layer: own にしてください")
        src = b
    elif direction_layer == "own":
        have = sorted(ds.raw)
        src = L if L in ds.raw else max([x for x in have if x <= L], default=have[0])
    else:
        raise GneError("BAD_RECIPE", f"direction_layer={direction_layer!r} は own か base です")
    r = np.asarray(ds.raw[src], dtype=np.float64)
    n = float(np.linalg.norm(r))
    if n == 0:
        raise GneError("BAD_DIRECTION", f"方向 {ds.name} の層 {src} の長さが 0 です")
    return (r / n).astype(np.float32), src


class ProjBuilder:
    """r̂ᵀW(層・方向ごと)と書き込み行列 W(層ごと、少数だけ)を覚えておき、LoRA の行列を何度も作る(因果・配分のループ用)。"""

    def __init__(self, weights, directions: DirSource | dict, *, direction_layer: str = "own", keep_w: int = 2):
        self.weights = weights
        self.ds_of = directions if callable(directions) else directions.__getitem__
        self.direction_layer = direction_layer
        self._rw: dict[tuple[int, str, int], tuple[np.ndarray, np.ndarray]] = {}
        self._w: dict[int, np.ndarray] = {}
        self.keep_w = keep_w

    def W(self, L: int) -> np.ndarray:
        if L not in self._w:
            if len(self._w) >= self.keep_w:
                self._w.pop(next(iter(self._w)))
            self._w[L] = self.weights.head_writer_matrix(L)
        return self._w[L]

    def r_of(self, L: int, name: str) -> tuple[np.ndarray, int]:
        """層 L のヘッドに当てる方向(重みは読まない)。"""
        return head_direction(self.ds_of(name), L, self.direction_layer)

    def r_rw(self, L: int, name: str) -> tuple[np.ndarray, np.ndarray, int]:
        r, src = self.r_of(L, name)
        key = (L, name, src)
        if key not in self._rw:
            self._rw[key] = (r, (r.astype(np.float64) @ self.W(L)).astype(np.float32))
        r, rw = self._rw[key]
        return r, rw, src

    def matrices(self, heads: list[ProjHead], warn: Callable[[str], None] | None = None) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """層 → (A (r, n_in), B (n_embd, r))。同じ層・同じ方向は 1 本に畳む。"""
        by_layer: dict[int, list[ProjHead]] = defaultdict(list)
        for p in heads:
            by_layer[p.layer].append(p)
        out = {}
        for L, ps in sorted(by_layer.items()):
            groups: dict[tuple[str, int], list] = {}
            for p in ps:
                r, rw, src = self.r_rw(L, p.direction)
                g = groups.setdefault((p.direction, src), [r, np.zeros_like(rw)])
                _, lo, hi = self.weights.head_block(L, p.head)
                g[1][lo:hi] += np.float32(p.s) * rw[lo:hi]
            A = np.stack([g[1] for g in groups.values()]).astype(np.float32)
            B = np.stack([-g[0] for g in groups.values()], axis=1).astype(np.float32)
            if len(groups) > 1 and warn is not None:
                warn(f"L{L}: 方向が {len(groups)} 種類混ざっているので、LoRA の階数が {len(groups)} になります"
                     "(同じ方向なら階数 1 に畳めます)")
            out[L] = (A, B)
        return out

    def tensors(self, heads: list[ProjHead], warn=None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """書き込み行列の名前 → (A, B)。"""
        return {self.weights.head_block(L, 0)[0]: ab for L, ab in self.matrices(heads, warn).items()}

    def ablate_tensors(self, L: int, heads: list[int] | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """層 L のヘッド(None で全部)の出力を 0 にする LoRA(gain ×0 と同じ)。全ヘッドなら階数 min(n_embd, n_in)。"""
        W = self.W(L)
        name = self.weights.head_block(L, 0)[0]
        n_embd, n_in = W.shape
        if heads is None:
            if n_embd < n_in:                       # −W = (−I)(W)
                return {name: (W.astype(np.float32), -np.eye(n_embd, dtype=np.float32))}
            return {name: (np.eye(n_in, dtype=np.float32), (-W).astype(np.float32))}
        As, Bs = [], []
        for h in heads:
            _, lo, hi = self.weights.head_block(L, h)
            A = np.zeros((hi - lo, n_in), dtype=np.float32)
            A[np.arange(hi - lo), np.arange(lo, hi)] = 1.0
            As.append(A)
            Bs.append((-W[:, lo:hi]).astype(np.float32))
        return {name: (np.concatenate(As, axis=0), np.concatenate(Bs, axis=1))}


def unit_matrices(weights, directions: DirSource | dict, units, *, direction_layer: str = "own",
                  warn: Callable[[str], None] | None = None) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """CHANGE-06 A.3: ユニット(ProjHead か proj の Unit。Unit は direction が要る)→ 層ごとの (A, B)。"""
    heads = [u if isinstance(u, ProjHead) else ProjHead(u.layer, u.head, float(u.k), u.direction) for u in units]
    return ProjBuilder(weights, directions, direction_layer=direction_layer).matrices(heads, warn)


# --- LoRA の書き出し(E.1)------------------------------------------------------------------------------

def proj_meta(heads: list[ProjHead], builder: ProjBuilder) -> dict:
    """LoRA・焼き込みに残す記録(どのヘッドをどの方向で何倍か、方向の中身の鍵)。"""
    from .ablate import vector_key
    rows = []
    for p in heads:
        r, src = builder.r_of(p.layer, p.direction)
        rows.append({"layer": p.layer, "head": p.head, "s": p.s, "direction": p.direction, "direction_from": src,
                     "vector_key": vector_key(r)})
    return {"heads": rows, "direction_layer": builder.direction_layer}


def proj_cache_key(heads: list[ProjHead], builder: ProjBuilder) -> str:
    return hashlib.sha256(json.dumps(proj_meta(heads, builder), sort_keys=True).encode()).hexdigest()[:16]


def export_proj_lora(weights, recipe: Recipe, path: str | Path, directions: DirSource, *,
                     warn: Callable[[str], None] | None = None) -> dict:
    """レシピの proj のユニットだけを LoRA アダプタ GGUF に書く(実行時に当てる LoRA と headproj export)。"""
    from .ablate import write_lora_tensors
    heads = proj_heads(recipe)
    if not heads:
        raise GneError("BAD_RECIPE", "proj のユニットがありません(例 --unit 33:H16:s6@NAME)")
    b = ProjBuilder(weights, directions, direction_layer=recipe.direction_layer)
    tens = b.tensors(heads, warn)
    m = weights.meta
    rep = write_lora_tensors(weights, tens, path, name=f"{m.name} headproj" + (f" {recipe.name}" if recipe.name else ""),
                             kv={"gne.headproj": json.dumps(proj_meta(heads, b), ensure_ascii=False),
                                 "gne.recipe": json.dumps(recipe.to_dict(), ensure_ascii=False)})
    return {**rep, **layer_summary(weights, heads, tens)}


def layer_summary(weights, heads: list[ProjHead], tens: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """層ごとのヘッドと階数(1 に畳めたか)。"""
    by: dict[int, list[ProjHead]] = defaultdict(list)
    for p in heads:
        by[p.layer].append(p)
    out = {}
    for L, ps in sorted(by.items()):
        name = weights.head_block(L, 0)[0]
        rank = int(tens[name][1].shape[1]) if name in tens else 0
        out[str(L)] = {"heads": [[p.head, p.s, p.direction] for p in ps], "rank": rank,
                       "directions": sorted({p.direction for p in ps}), "folded": rank == 1}
    return {"proj_layers": out}


def merge_tensors(*parts: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """同じテンソルの LoRA を連結して 1 つにする(B₁A₁ + B₂A₂ = [B₁ B₂][A₁; A₂])。"""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for d in parts:
        for name, (A, B) in d.items():
            if name in out:
                A0, B0 = out[name]
                out[name] = (np.concatenate([A0, A], axis=0), np.concatenate([B0, B], axis=1))
            else:
                out[name] = (A, B)
    return out


# --- 焼き込み(E.2)---------------------------------------------------------------------------------

def bake_proj_gguf(weights, recipe: Recipe, dst: str | Path, directions: DirSource, *, requant: str = "q8_0",
                   warn: Callable[[str], None] | None = None) -> dict:
    """書き込み行列のうち proj(と、あればヘッド・ニューロンの gain)で書き換えるテンソルだけを Q8_0 か F16 にし、
    他は元のバイトのまま写した GGUF を書く。変化量は LoRA と同じ B @ A を元の重みに足したもの(LoRA 版と一致させるため)。
    書き換えたテンソル名と元の型は GGUF の gne.patches と、<dst>.headproj.json に残す。"""
    from .gain import bake_gain_gguf
    heads = proj_heads(recipe)
    b = ProjBuilder(weights, directions, direction_layer=recipe.direction_layer)
    extra = b.tensors(heads, warn) if heads else {}
    meta = proj_meta(heads, b) if heads else None
    rep = bake_gain_gguf(weights, recipe, dst, requant=requant, extra=extra, extra_meta=meta)
    side = Path(str(dst) + ".headproj.json")
    side.write_text(json.dumps({"recipe": recipe.to_dict(), "headproj": meta, "modified": rep["modified"],
                                "requant": requant}, ensure_ascii=False, indent=1), encoding="utf-8")
    return {**rep, "sidecar": str(side), **(layer_summary(weights, heads, extra) if heads else {})}


# --- ランダム対照(F.2)-------------------------------------------------------------------------------

def control_pool(weights, recipe: Recipe, *, pool: str = "before", score_layers: list[int] | None = None) -> list[tuple[int, int]]:
    """対照のヘッドの母集団。before = score_layers(既定は方向の基準層の後段まで = レシピのヘッドの最も後ろの層)より前の
    全ヘッド、same-layers = レシピのヘッドと同じ層の他のヘッド(厳しい対照)。"""
    arch = weights.arch
    used = {(p.layer, p.head) for p in proj_heads(recipe)}
    if pool == "same-layers":
        Ls = sorted({L for L, _ in used})
    elif pool == "before":
        top = max(score_layers) if score_layers else max(L for L, _ in used)
        Ls = [li.index for li in arch.layers if li.head_dim and li.head_writer and li.index <= top]
    else:
        raise GneError("BAD_ARGS", f"--control-pool は before か same-layers です({pool!r})")
    out = []
    for L in Ls:
        li = arch.layers[L]
        if li.head_dim and li.head_writer:
            out += [(L, h) for h in range(li.n_head) if (L, h) not in used]
    return out


def proj_control_recipes(session, recipe: Recipe, n: int, *, pool: str = "before",
                         score_layers: list[int] | None = None, seed_base: int | None = None) -> list[Recipe]:
    """proj のユニットを、同数のランダムなヘッドに替えた n 本(s の値と方向は元の組と同じものを並べ替えずに当てる。
    方向は own なら新しいヘッドの層の方向になる)。proj 以外のユニットはそのまま。"""
    from .controls import SEED_BASE
    heads = proj_heads(recipe)
    if n <= 0 or not heads:
        return []
    cand = control_pool(session.weights, recipe, pool=pool, score_layers=score_layers)
    if len(cand) < len(heads):
        raise GneError("BAD_ARGS", f"対照のヘッドが足りません(母集団 {len(cand)}、要るのは {len(heads)})")
    out = []
    rest = [u for u in recipe.units if u.mode != "proj"]
    for j in range(n):
        rng = np.random.default_rng([seed_base if seed_base is not None else SEED_BASE, 6, j])
        pick = [cand[int(i)] for i in rng.choice(len(cand), size=len(heads), replace=False)]
        units = rest + [Unit(L, None, 0.0, "head", mode="proj", k=p.s, head=h, direction=p.direction)
                        for (L, h), p in zip(pick, heads, strict=True)]
        out.append(recipe.like(units, name=f"{recipe.name or 'recipe'}-randproj{j}"))
    return out


# --- 検証(CHANGE-06 I: V24・V25)----------------------------------------------------------------------

def check_head_proj(session, L: int, h: int, direction: str, *, ss=(1.0, 2.0, 6.0), k_gain: float = 0.0,
                    text: str | None = None, direction_layer: str = "own") -> dict:
    """V24: ヘッド h の proj(s 倍)の LoRA を当てると、書き込み行列の出力の差が −s · r̂ (r̂ · W[:, blk] x_h)
    (無介入の per-head 出力から numpy で計算)と合う(相対誤差の中央値 < 3%)。s=1 でそのヘッドの書き込みの方向成分が 0、
    s=2 で符号が反転する。V25: 差のうち r̂ に直交する成分が数値誤差以内(同じヘッドを gain k_gain で変えると直交成分も
    変わることを対比で出す)。"""
    from .attn import _names, _probe, capture_attention, writer_out_name
    from .gain import export_gain_lora
    text = text or "Tell me a short story about a lighthouse keeper and the sea. 灯台守の話をして。"
    e, w = session.engine, session.weights
    _, lo, hi = w.head_block(L, h)
    probe, ctx = _probe(e)
    names = _names(e, probe, ctx, e.tokenize(text, add_special=True, parse_special=False)[:16])
    out_name = writer_out_name(session, names, L)
    if out_name is None:
        raise GneError("ARCH_UNSUPPORTED", f"層 {L} の書き込み行列の出力が捕捉できません")
    ds = session.direction_set(direction)
    r, src = head_direction(ds, L, direction_layer)
    r = r.astype(np.float64)
    base = capture_attention(session, [text], layers=[L], want_pattern=False, extra=[out_name])[0]
    Wb = w.head_write_matrix(L, h).astype(np.float64)                 # (n_embd, head_dim)
    v = base.head_out[L][:, h, :].astype(np.float64) @ Wb.T            # ヘッド h が残差に書くベクトル (T, n_embd)
    vr = v @ r
    vn = np.linalg.norm(v, axis=1)
    keep = vn > 1e-6 * max(1.0, float(vn.max()))
    d = session.store.heads_dir(session.meta.model_id)
    rows = []

    def measure(path):
        got = capture_attention(session, [text], layers=[L], want_pattern=False, extra=[out_name], lora=[(str(path), 1.0)])[0]
        e.drop_lora(path)
        return (got.extra[out_name] - base.extra[out_name]).astype(np.float64)   # (T, n_embd)
    for s_ in ss:
        rec = Recipe([Unit(L, None, 0.0, "head", mode="proj", k=float(s_), head=h, direction=direction)],
                     session.meta.model_id, name=f"v24-L{L}H{h}", direction_layer=direction_layer)
        path = d / f"v24-L{L}-H{h}-s{s_:g}.lora.gguf"
        export_proj_lora(w, rec, path, session.direction_set)
        meas = measure(path)
        path.unlink(missing_ok=True)
        pred = -s_ * np.outer(vr, r)
        den = np.linalg.norm(pred, axis=1)
        k2 = keep & (den > 1e-6 * max(1.0, float(den.max())))
        rel = np.linalg.norm(meas - pred, axis=1)[k2] / den[k2]
        new_par = (v + meas) @ r                                         # 介入後のヘッドの書き込みの方向成分
        orth = meas - np.outer(meas @ r, r)
        orth_rel = np.linalg.norm(orth, axis=1)[keep] / vn[keep]
        row = {"s": float(s_), "rel_err_median": float(np.median(rel)) if rel.size else None,
               "rel_err_p95": float(np.percentile(rel, 95)) if rel.size else None,
               "par_after_over_before": float(np.median(new_par[keep] / np.where(np.abs(vr[keep]) > 0, vr[keep], 1.0)))
               if keep.any() else None,
               "orth_change_rel_max": float(orth_rel.max()) if orth_rel.size else None,
               "orth_change_rel_median": float(np.median(orth_rel)) if orth_rel.size else None}
        expect = 1.0 - s_                                                # 方向成分は (1 − s) 倍
        row["ok"] = bool(rel.size and np.median(rel) < 0.03 and row["orth_change_rel_median"] is not None
                         and row["orth_change_rel_median"] < 0.03
                         and abs(row["par_after_over_before"] - expect) < 0.05 * max(1.0, abs(expect)))
        rows.append(row)
    # 対比: 同じヘッドの gain(列ブロックを k 倍)は直交成分も (k − 1) 倍ぶん変わる
    gr = Recipe([Unit(L, None, 0.0, "head", mode="gain", k=float(k_gain), head=h)], session.meta.model_id, name="v25")
    gpath = d / f"v25-L{L}-H{h}-k{k_gain:g}.lora.gguf"
    export_gain_lora(w, gr, gpath)
    gm = measure(gpath)
    gpath.unlink(missing_ok=True)
    g_orth = np.linalg.norm(gm - np.outer(gm @ r, r), axis=1)[keep] / vn[keep]
    e.free_attn_probe()
    gain_row = {"k": float(k_gain), "orth_change_rel_median": float(np.median(g_orth)) if g_orth.size else None}
    v25 = bool(all(r_["orth_change_rel_median"] is not None and r_["orth_change_rel_median"] < 0.03 for r_ in rows)
               and gain_row["orth_change_rel_median"] is not None and gain_row["orth_change_rel_median"] > 0.1)
    return {"layer": L, "head": h, "block": [lo, hi], "out_tensor": out_name, "direction": direction,
            "direction_from": src, "dir_cos_head": float(np.median(np.abs(vr[keep]) / vn[keep])) if keep.any() else None,
            "rows": rows, "gain": gain_row, "v24": all(r_["ok"] for r_ in rows), "v25": v25,
            "ok": all(r_["ok"] for r_ in rows) and v25}
