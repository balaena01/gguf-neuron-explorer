"""ヘッドの対照分析 `gne heads contrast`(CHANGE-04 C)。

ニューロンの逆引きと同じ構図で、同じ会話を「ペルソナあり(system_a)」と「中立(system_b)」で流し、ヘッドごとに差を測る。
会話 = system + ユーザーの発話 + system_a で greedy に作った返事(両側で同じ返事を読む)。比べるのは返事のトークンの行。

| 指標 | 層 | 定義 |
|---|---|---|
| trigger_mass | attention | 返事の各位置から、共通部分(ユーザーの発話 + 返事)への注目のうちトリガー区間(対照集合の triggers)のトークンが占める割合の平均。A − B |
| pattern_shift | attention | 返事の各行の注目の分布の JS ダイバージェンス(bit)の平均。system の部分は 1 つにまとめる |
| out_shift | 両方 | per-head 出力の平均差のノルム ÷ そのヘッドの典型的な出力ノルム(B 側の行ごとのノルムの平均) |
| dir_cos | 両方 | 出力差を W_O の列ブロックで残差に写したもの W_O[:, blk]·Δ と、その層の方向 r_L(gne direction)の cos |

trigger_mass を共通部分の中の割合にするのは、system の長さが違うと注目が system 側に薄まり、総量の差がペルソナではなく
長さの差になるため。
雑音の床: 中立どうし(system_b と、同じくらいの長さの別の中立の system_b2)で同じ指標を計算し、全ヘッドの値を対照の分布にする
(各指標の p95 が床)。「system を別の文に替えるだけで動く量」を床にする。z は対照の分布に対する z(trigger_mass と dir_cos は絶対値で)を、測れた指標で平均したもの。
null_pct は、対照のヘッドの z の何割より大きいか。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import yaml

from .attn import capture_attention, find_spans, spans_to_tokens, token_offsets
from .reverse import ContrastSet
from .steer import GneError

# 床に使う中立の system(ペルソナの system と同じくらいの長さで、性格を持たない)。対照集合の system_b2 で替えられる
DEFAULT_SYSTEM_B2 = ("You are a knowledgeable and reliable assistant. You answer questions clearly and accurately, you explain "
                     "your reasoning step by step when it helps, you keep a calm and professional tone, and you respect the "
                     "user's time by staying concise and on topic. If you are unsure about something, you say so.")
METRICS = ("trigger_mass", "pattern_shift", "out_shift", "dir_cos")


def load_triggers(path: str | Path, cs: ContrastSet) -> list[list[str]]:
    """対照集合の triggers: 全プロンプト共通の文字列のリスト、プロンプトごとのリストのリスト(user_prompts と同じ長さ)、
    または {プロンプト: [文字列]} の表。"re:" で始まる文字列は正規表現。無ければ空。"""
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    t = d.get("triggers")
    n = len(cs.user_prompts)
    if t is None:
        return [[] for _ in range(n)]
    if isinstance(t, dict):
        return [[str(x) for x in (t.get(u) or [])] for u in cs.user_prompts]
    if isinstance(t, list) and all(isinstance(x, str) for x in t):
        return [list(t) for _ in range(n)]
    if isinstance(t, list) and len(t) == n and all(isinstance(x, list | type(None)) for x in t):
        return [[str(y) for y in (x or [])] for x in t]
    raise GneError("BAD_CONTRAST", "triggers は文字列のリスト、user_prompts と同じ長さのリストのリスト、"
                                   "または {プロンプト: [文字列]} です")


def _js_bits(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """行ごとの JS ダイバージェンス(bit、0〜1)。p, q: (..., n)。"""
    m = 0.5 * (p + q)

    def kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(a > 0, a * (np.log2(np.maximum(a, 1e-30)) - np.log2(np.maximum(b, 1e-30))), 0.0)
        return t.sum(axis=-1)
    return np.clip(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0, 1.0)


def _suffix_len(a: list[int], b: list[int]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


class _Acc:
    """1 組(A vs B、または B vs B2)の、ヘッドごとの指標の集計。"""

    def __init__(self, layers):
        self.layers = layers
        self.tm = {L: [] for L in layers}          # プロンプトごとの (n_head,) trigger_mass の差
        self.js = {L: [] for L in layers}          # 行ごとの (n_head,) JS
        self.dsum = {L: None for L in layers}      # (n_head, head_dim) 出力差の和
        self.nsum = {L: None for L in layers}      # (n_head,) B 側の出力ノルムの和
        self.rows = 0

    def add(self, ca, cb, *, sa: int, sb: int, ra: list[int], trig_a: list[int]):
        """ca・cb: AttnCapture。sa・sb: 共通の末尾(ユーザーの発話 + 返事)が始まる位置。ra: A の返事の行。
        trig_a: A でのトリガーのトークン。B の位置は A の位置 − (sa − sb)。"""
        off = sa - sb
        rb = [r - off for r in ra]
        trig_b = [t - off for t in trig_a]
        self.rows += len(ra)
        for L in self.layers:
            if L in ca.head_out and L in cb.head_out:
                ha, hb = ca.head_out[L][ra], cb.head_out[L][rb]        # (R, n_head, head_dim)
                d = (ha - hb).sum(axis=0)
                n = np.linalg.norm(hb, axis=2).sum(axis=0)
                self.dsum[L] = d if self.dsum[L] is None else self.dsum[L] + d
                self.nsum[L] = n if self.nsum[L] is None else self.nsum[L] + n
            if L in ca.pattern and L in cb.pattern:
                Pa, Pb = ca.pattern[L][:, ra, :], cb.pattern[L][:, rb, :]   # (n_head, R, T)
                if trig_a:                                              # 共通部分の中のトリガーの割合
                    fa = Pa[:, :, trig_a].sum(axis=2) / np.maximum(Pa[:, :, sa:].sum(axis=2), 1e-9)
                    fb = Pb[:, :, trig_b].sum(axis=2) / np.maximum(Pb[:, :, sb:].sum(axis=2), 1e-9)
                    self.tm[L].append(fa.mean(axis=1) - fb.mean(axis=1))
                pa = np.concatenate([Pa[:, :, :sa].sum(axis=2, keepdims=True), Pa[:, :, sa:]], axis=2)
                pb = np.concatenate([Pb[:, :, :sb].sum(axis=2, keepdims=True), Pb[:, :, sb:]], axis=2)
                self.js[L].append(_js_bits(pa, pb).mean(axis=1))

    def metrics(self, weights, dirs: dict[int, np.ndarray] | None) -> dict[int, dict[str, np.ndarray | None]]:
        out = {}
        for L in self.layers:
            m: dict[str, np.ndarray | None] = {k: None for k in METRICS}
            if self.tm[L]:
                m["trigger_mass"] = np.mean(self.tm[L], axis=0)
            if self.js[L]:
                m["pattern_shift"] = np.mean(self.js[L], axis=0)
            if self.dsum[L] is not None and self.rows:
                delta = self.dsum[L] / self.rows                       # (n_head, head_dim)
                typ = self.nsum[L] / self.rows
                m["out_shift"] = np.linalg.norm(delta, axis=1) / np.maximum(typ, 1e-9)
                if dirs is not None and L in dirs:
                    W = weights.head_writer_matrix(L)                 # (n_embd, n_head·head_dim)
                    hd = delta.shape[1]
                    r = dirs[L] / max(float(np.linalg.norm(dirs[L])), 1e-12)
                    cos = []
                    for h in range(delta.shape[0]):
                        v = W[:, h * hd:(h + 1) * hd] @ delta[h]
                        nv = float(np.linalg.norm(v))
                        cos.append(float(v @ r) / nv if nv > 0 else 0.0)
                    m["dir_cos"] = np.array(cos)
            out[L] = m
        return out


def _flatten(ms: dict[int, dict], kinds: dict[int, str]) -> list[dict]:
    rows = []
    for L, m in ms.items():
        n = next((len(v) for v in m.values() if v is not None), 0)
        for h in range(n):
            rows.append({"layer": L, "head": h, "kind": kinds[L],
                         **{k: (None if m[k] is None else float(m[k][h])) for k in METRICS}})
    return rows


def heads_contrast(session, contrast: str | Path, *, name: str | None = None, direction: str | None = None,
                   system_b2: str | None = None, layers: list[int] | None = None, max_tokens: int = 48,
                   progress=None) -> dict:
    """対照集合でヘッドごとの指標を測り、<root>/heads/<model_id>/<name>.candidates.jsonl(z の大きい順)と .json に書く。
    direction は dir_cos に使う方向の名前(既定は対照集合と同じ名前の方向があればそれ)。"""
    from .direction import load_contrast
    from .evalx import _generate
    t0 = time.perf_counter()
    cs, _ = load_contrast(contrast)
    name = name or cs.name
    trig = load_triggers(contrast, cs)
    sys_b2 = system_b2 or (yaml.safe_load(Path(contrast).read_text(encoding="utf-8")) or {}).get("system_b2") or DEFAULT_SYSTEM_B2
    e, w = session.engine, session.weights
    arch = w.arch
    Ls = [L for L in (layers if layers is not None else range(e.n_layer - 1)) if arch.layers[L].head_dim]
    kinds = {L: "attn" if arch.layers[L].mixer == "attention" else "linear" for L in Ls}
    dname = direction
    if dname is None:
        try:
            session.direction_path(cs.name)
            dname = cs.name
        except GneError:
            dname = None
    dirs = None
    if dname is not None:
        ds = session.direction_set(dname)
        dirs = {L: ds.raw[L] for L in Ls if L in ds.raw}
    n = len(cs.user_prompts)
    steps, done = 1 + n, 0

    def tick(msg):
        nonlocal done
        done += 1
        if progress:
            progress(done, steps, msg)
    convs_a = [[{"role": "system", "content": cs.system_a}, {"role": "user", "content": u}] for u in cs.user_prompts]
    resp = [o["text"] for o in _generate(session, [(c, max_tokens) for c in convs_a], None, None, "all", lambda k: None)]
    tick("heads: responses")
    main, null = _Acc(Ls), _Acc(Ls)
    samples, n_trig = [], 0
    for i, (u, r) in enumerate(zip(cs.user_prompts, resp, strict=True)):
        tail = [{"role": "user", "content": u}, {"role": "assistant", "content": r}]
        ids = {}
        for side, system in (("A", cs.system_a), ("B", cs.system_b), ("B2", sys_b2)):
            msgs = [{"role": "system", "content": system}] + tail
            ids[side] = e.tokenize_prompt(e.render_chat(msgs, add_generation_prompt=False))
        head_a = e.tokenize_prompt(e.render_chat([{"role": "system", "content": cs.system_a},
                                                  {"role": "user", "content": u}]))
        p_a = len(head_a) if ids["A"][:len(head_a)] == head_a else None
        S = min(_suffix_len(ids["A"], ids["B"]), _suffix_len(ids["A"], ids["B2"]))
        sa, sb, sb2 = (len(ids[k]) - S for k in ("A", "B", "B2"))
        if p_a is None or p_a < sa or p_a >= len(ids["A"]):
            session.warn(f"プロンプト {i + 1}: 返事の位置を合わせられないので飛ばしました")
            tick(f"heads: {i + 1}/{n}")
            continue
        ra = list(range(p_a, len(ids["A"])))
        caps = capture_attention(session, [ids["A"], ids["B"], ids["B2"]], layers=Ls)
        text, offs = token_offsets(e, ids["A"])
        trig_a: list[int] = []
        if trig[i]:
            at = text.find(u, len(text) - len(e.detokenize(ids["A"][sa:], special=True)) - 1)
            if at < 0:
                at = text.rfind(u)
            if at >= 0:
                spans = [(s + at, t + at) for s, t in find_spans(u, trig[i])]
                trig_a = [t for t in spans_to_tokens(offs, spans) if t >= sa]
        n_trig += bool(trig_a)
        main.add(caps[0], caps[1], sa=sa, sb=sb, ra=ra, trig_a=trig_a)
        null.add(caps[1], caps[2], sa=sb, sb=sb2, ra=[x - (sa - sb) for x in ra],
                 trig_a=[x - (sa - sb) for x in trig_a])
        samples.append({"prompt": u, "response": r[:200], "n_tokens": len(ids["A"]), "n_response": len(ra),
                        "triggers": [text[offs[t][0]:offs[t][1]] for t in trig_a]})
        tick(f"heads: {i + 1}/{n}")
    e.free_attn_probe()
    rows = _flatten(main.metrics(w, dirs), kinds)
    nrows = _flatten(null.metrics(w, dirs), kinds)
    if not rows:
        raise GneError("BAD_CONTRAST", "測れたヘッドがありません")
    stats = {}
    for k in METRICS:
        signed = k in ("trigger_mass", "dir_cos")
        pool = np.array([abs(x[k]) if signed else x[k] for x in nrows if x[k] is not None])
        if pool.size:
            stats[k] = {"mean": float(pool.mean()), "std": float(pool.std()) or 1e-9, "p95": float(np.percentile(pool, 95)),
                        "signed": signed}

    def zscore(x):
        zs = [((abs(x[k]) if stats[k]["signed"] else x[k]) - stats[k]["mean"]) / stats[k]["std"]
              for k in stats if x[k] is not None]
        return float(np.mean(zs)) if zs else None
    nz = np.sort([z for z in (zscore(x) for x in nrows) if z is not None])
    for x in rows:
        x["z"] = zscore(x)
        x["null_pct"] = None if x["z"] is None or not nz.size else float(np.searchsorted(nz, x["z"], "left") / nz.size)
    rows.sort(key=lambda x: -(x["z"] if x["z"] is not None else -1e9))
    out_dir = session.store.heads_dir(session.meta.model_id)
    cand = out_dir / f"{name}.candidates.jsonl"
    with cand.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    d = {"name": name, "contrast": str(Path(contrast).resolve()), "contrast_sha": cs.sha, "direction": dname, "system_b2": sys_b2,
         "layers": Ls, "n_prompts": len(samples), "n_with_triggers": n_trig, "max_tokens": max_tokens,
         "floors": {k: v["p95"] for k, v in stats.items()}, "null_stats": stats, "top": rows[:50],
         "n_heads": len(rows), "candidates": str(cand), "samples": samples, "seconds": time.perf_counter() - t0}
    p = out_dir / f"{name}.json"
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    return {**d, "path": str(p)}


def load_candidates(session, name: str) -> list[dict]:
    p = session.store.heads_dir(session.meta.model_id) / f"{name}.candidates.jsonl"
    if not p.exists():
        raise GneError("FILE_NOT_FOUND", f"ヘッドの対照分析 {name} がありません(gne heads contrast で作れます)")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- 注目先(fires --heads)・調書 -------------------------------------------------------------------

def attention_view(session, text: str, *, pos: int = -1, layer: int | None = None, head: int | None = None,
                   top: int = 5, system: str | None = None, chat: bool = False) -> dict:
    """文を流し、選んだトークン(pos、既定は最後)が各ヘッドでどこを見ているか(上位 top)と、per-head 出力のノルム。
    layer と head を両方指定すると、そのヘッドの T × T のパターンも返す(UI のヒートマップ用)。
    chat=True なら text をユーザーの発話としてチャットテンプレートに入れる(system も付けられる)。"""
    e, w = session.engine, session.weights
    if chat:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
        conv: list[dict] | str = msgs
    else:
        conv = text
    Ls = None if layer is None else [layer]
    if layer is not None and not w.arch.layers[layer].head_dim:
        raise GneError("BAD_ARGS", f"層 {layer} にはヘッドがありません")
    ac = capture_attention(session, [conv], layers=Ls)[0]
    T = ac.n_tokens
    p = pos if pos >= 0 else T + pos
    if not 0 <= p < T:
        raise GneError("BAD_ARGS", f"pos {pos} は範囲外です(0..{T - 1})")
    toks = [ac.text[a:b] for a, b in ac.offsets]
    heads = []
    for L in sorted(ac.head_out):
        li = w.arch.layers[L]
        hs = range(li.n_head) if head is None else [head]
        for h in hs:
            if not 0 <= h < li.n_head:
                raise GneError("BAD_ARGS", f"L{L} のヘッド {h} は範囲外です")
            row = {"layer": L, "head": h, "kind": "attn" if li.mixer == "attention" else "linear",
                   "norm": float(np.linalg.norm(ac.head_out[L][p, h]))}
            if L in ac.pattern:
                a = ac.pattern[L][h, p, :p + 1]
                k = min(top, a.size)
                idx = np.argsort(-a)[:k]
                row["top"] = [{"pos": int(i), "token": toks[i], "weight": float(a[i])} for i in idx]
                row["entropy"] = float(-(a[a > 0] * np.log2(a[a > 0])).sum())
            heads.append(row)
    out = {"text": text, "n_tokens": T, "pos": p, "tokens": toks, "heads": heads}
    if layer is not None and head is not None and layer in ac.pattern:
        out["pattern"] = ac.pattern[layer][head].round(5).tolist()
    e.free_attn_probe()
    return out


def _head_names(session) -> dict[str, str]:
    p = session.store.root / "names" / f"{session.meta.model_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def head_key(L: int, h: int) -> str:
    """names.json のキー(ニューロンの "L:n" と同じファイル、ヘッドは "L:Hh")。"""
    return f"{L}:H{h}"


def block_logit_lens(weights, L: int, h: int, *, top_k: int = 12) -> dict:
    """W_O の列ブロック(ヘッド h の書き込み)の主成分 2 本を logit lens に通した上位・下位の語。
    特異値の比(1 本目がどれだけ支配的か)も返す。"""
    from .lens import _topk_desc
    B = weights.head_write_matrix(L, h)                          # (n_embd, head_dim)
    U, S, _ = np.linalg.svd(B, full_matrices=False)
    g = weights.output_norm_weight()
    comps = []
    for i in range(min(2, U.shape[1])):
        v = (U[:, i] * g).astype(np.float32)
        z = np.concatenate([rows @ v for _, rows in weights.output_rows(8192)])
        mu, sd = float(z.mean()), float(z.std()) or 1e-9
        up, down = _topk_desc(z, top_k), _topk_desc(-z, top_k)
        comps.append({"sv": float(S[i]), "up": [int(t) for t in up], "up_z": [float((z[t] - mu) / sd) for t in up],
                      "down": [int(t) for t in down], "down_z": [float((mu - z[t]) / sd) for t in down]})
    return {"singular_values": [float(x) for x in S[:8]], "sv_share": float(S[0] ** 2 / max((S ** 2).sum(), 1e-30)),
            "components": comps}


def head_dossier(session, L: int, h: int, *, lens: bool = True) -> dict:
    """ヘッドの調書: 種類・大きさ・保存済みの対照分析での指標と順位・名前・当てたことのある gain・W_O ブロックの logit lens。"""
    w = session.weights
    name_t, lo, hi = w.head_block(L, h)
    li = w.arch.layers[L]
    contrasts = []
    for p in sorted(session.store.heads_dir(session.meta.model_id).glob("*.candidates.jsonl"),
                    key=lambda p: -p.stat().st_mtime):
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        for rank, x in enumerate(rows, 1):
            if x["layer"] == L and x["head"] == h:
                contrasts.append({"name": p.name[:-len(".candidates.jsonl")], "rank": rank, "n_ranked": len(rows),
                                  **{k: x.get(k) for k in (*METRICS, "z", "null_pct")}})
                break
    gains = set()
    from gguf import GGUFReader
    for p in session.store.gain_dir(session.meta.model_id).glob("*.lora.gguf"):
        try:
            f = GGUFReader(p).fields.get("gne.recipe")
            units = json.loads(f.contents())["units"] if f is not None else []
        except Exception:  # noqa: BLE001
            continue
        for u in units:
            if u.get("kind") == "head" and u.get("layer") == L and u.get("head") == h and u.get("k") is not None:
                gains.add(float(u["k"]))
    out = {"layer": L, "head": h, "kind": "attn" if li.mixer == "attention" else "linear", "n_head": li.n_head,
           "n_head_kv": li.n_head_kv, "head_dim": li.head_dim, "writer": name_t, "block": [lo, hi],
           "kv_group": None, "name": _head_names(session).get(head_key(L, h)), "contrasts": contrasts,
           "gain_k": sorted(gains), "logit_lens": None}
    if li.mixer == "attention" and li.n_head_kv:
        from .attn import kv_group
        out["kv_group"] = kv_group(h, li.n_head, li.n_head_kv)
    if lens:
        ll = block_logit_lens(w, L, h)
        for c in ll["components"]:
            c["up_tokens"] = [session.vocab.text(t) for t in c["up"]]
            c["down_tokens"] = [session.vocab.text(t) for t in c["down"]]
        out["logit_lens"] = ll
    return out


def set_head_name(session, L: int, h: int, name: str) -> None:
    session.weights.head_block(L, h)
    p = session.store.root / "names" / f"{session.meta.model_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    d = _head_names(session)
    if name:
        d[head_key(L, h)] = name
    else:
        d.pop(head_key(L, h), None)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


# --- 条件付きクエリ操作(CHANGE-04 E、実験的) ------------------------------------------------------------

GATED_Q_ARCHS = ("qwen35", "qwen35moe", "qwen3next")        # attn_q がヘッドごとに [q | ゲート] の 2 倍幅の arch


def q_rows(weights, L: int, h: int) -> tuple[int, int]:
    """attn_q.weight(numpy (n_q_out, n_embd))のうち、ヘッド h の q の行 [lo, hi)。Qwen3.5 はヘッドごとに
    [q(head_dim 行) | ゲート(head_dim 行)] が交互に並ぶ(verification.md §14.1)ので、q の行だけを返す。"""
    li = weights.arch.layers[L]
    if li.mixer != "attention":
        raise GneError("ARCH_UNSUPPORTED", f"層 {L} は attention の層ではありません(クエリ操作は attention の層だけ)")
    if not weights.has_tensor(f"blk.{L}.attn_q.weight"):
        raise GneError("ARCH_UNSUPPORTED", f"blk.{L}.attn_q.weight がありません(Q・K・V が 1 つのテンソルの arch は未対応)")
    hd = li.head_dim
    if weights.meta.arch in GATED_Q_ARCHS:
        return 2 * h * hd, 2 * h * hd + hd
    return h * hd, (h + 1) * hd


def static_dims(weights, L: int, frac: float = 0.25) -> np.ndarray:
    """位置で回らない(または回りの遅い)次元。rope.dimension_count が head_dim より小さければ、その後ろ(Qwen3.5 は 64〜255)。
    全部回る NeoX 型なら、周波数の低い対(i と i + hd/2、i が hd/2 に近いもの)を head_dim の frac だけ。"""
    hd = weights.arch.layers[L].head_dim
    n_rot = int(weights.field(f"{weights.meta.arch}.rope.dimension_count") or hd)
    if n_rot < hd:
        return np.arange(n_rot, hd)
    m = max(1, int(hd * frac) // 2)
    half = hd // 2
    return np.concatenate([np.arange(half - m, half), np.arange(hd - m, hd)])


def query_vectors(session, L: int, h: int, *, direction: str, seek: str | Path, frac: float = 0.25) -> dict:
    """クエリ操作の u・v と、倍率の目安。
    v: 層 L に入る残差のペルソナ方向 r(gne direction の r_{L-1})を attn_norm の重み γ で割った向き(v·norm(x) ∝ r·x)から、
       中立(B)の入力の平均の向きの成分を抜いたもの(B で v·norm(x) ≈ 0 にする。LoRA はバイアスを足せないので、
       ペルソナが無いときに q が動かないようにする)。平均は返事を書き始める位置(プロンプトの最後のトークン)で取る
       (ユーザーの発話や返事の頭のテンプレート全体でそろえても、最後のトークンでは B が負になって注目が下がった。F110)。
       A − B の差が 1 になるように大きさを決める。
    u: トリガーのトークンと他のトークンでの Kcur_normed(ヘッド h の K/V グループ、RoPE の前)の平均の差に q_norm の重み γ_q を
       掛け(LoRA は正規化の前の q に足すので、注目の点数への効きは γ_q ⊙ Δq·k)、static_dims に絞って正規化したもの。
       大きさは q(正規化の前)の典型的なノルム。"""
    from .attn import kv_group
    from .direction import load_contrast
    e, w = session.engine, session.weights
    li = w.arch.layers[L]
    lo, hi = q_rows(w, L, h)
    cs, _ = load_contrast(seek)
    trig = load_triggers(seek, cs)
    if not any(trig):
        raise GneError("BAD_CONTRAST", f"{seek} に triggers がありません(探しに行く語が要ります)")
    ds = session.direction_set(direction)
    src = L - 1 if (L - 1) in ds.raw else L
    gamma = w.tensor(f"blk.{L}.attn_norm.weight")
    from .weights import tensor_to_f32
    g = np.asarray(tensor_to_f32(gamma), dtype=np.float32).reshape(-1)
    v = ds.raw[src] / np.where(np.abs(g) > 1e-6, g, 1e-6)
    v = (v / np.linalg.norm(v)).astype(np.float32)
    grp = kv_group(h, li.n_head, li.n_head_kv)
    dims = static_dims(w, L, frac)
    qname = f"Qcur_full-{L}" if w.meta.arch in GATED_Q_ARCHS else f"Qcur-{L}"
    names = [f"attn_norm-{L}", f"Kcur_normed-{L}", qname]
    kt, ko, xa, xb, qn = [], [], [], [], []
    for side, system in (("A", cs.system_a), ("B", cs.system_b)):
        for u_text, tr in zip(cs.user_prompts, trig, strict=True):
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": u_text}]
            ids = e.tokenize_prompt(e.render_chat(msgs))
            ac = capture_attention(session, [ids], layers=[L], want_pattern=False, want_head_out=False, extra=names)[0]
            text, offs = ac.text, ac.offsets
            at = text.rfind(u_text)
            if at < 0:
                continue
            user = [i for i, (a, b) in enumerate(offs) if b > a and a >= at and b <= at + len(u_text)]
            tail = [len(ids) - 1]                                           # 返事を書き始める位置(最後のトークン)
            xs = ac.extra[f"attn_norm-{L}"]
            (xa if side == "A" else xb).append(xs[tail])
            if side != "A":
                continue
            K = ac.extra[f"Kcur_normed-{L}"].reshape(len(ids), -1, li.head_dim)[:, grp, :]
            tt = set(spans_to_tokens(offs, [(s + at, t + at) for s, t in find_spans(u_text, tr)]))
            kt += [K[i] for i in user if i in tt]
            ko += [K[i] for i in user if i not in tt]
            Q = ac.extra[qname]
            qh = Q[:, lo:hi] if Q.shape[1] >= hi else Q.reshape(len(ids), -1, li.head_dim)[:, h, :]
            qn += np.linalg.norm(qh[user], axis=1).tolist()
    e.free_attn_probe()
    if not kt or not ko:
        raise GneError("BAD_CONTRAST", "トリガーのトークンが見つかりませんでした")
    XA, XB = np.concatenate(xa), np.concatenate(xb)
    mu = XB.mean(axis=0)
    mu /= max(float(np.linalg.norm(mu)), 1e-12)
    v = v - (v @ mu) * mu                                      # 中立の入力の平均の向きを抜く
    v = (v / np.linalg.norm(v)).astype(np.float32)
    sa, sb = XA @ v, XB @ v
    gq = np.asarray(tensor_to_f32(w.tensor(f"blk.{L}.attn_q_norm.weight")), dtype=np.float32).reshape(-1) \
        if w.has_tensor(f"blk.{L}.attn_q_norm.weight") else np.ones(li.head_dim, dtype=np.float32)
    d = (np.mean(kt, axis=0) - np.mean(ko, axis=0)) * gq
    u = np.zeros(li.head_dim, dtype=np.float32)
    u[dims] = d[dims]
    if np.linalg.norm(u) == 0:
        raise GneError("BAD_CONTRAST", "u が 0 になりました")
    u /= np.linalg.norm(u)
    gap = float(np.mean(sa) - np.mean(sb))
    if gap <= 1e-9:
        raise GneError("BAD_CONTRAST", "方向が A と B の入力を分けません(v·norm(x) の差が 0 以下)")
    return {"u": u, "v": v / gap, "rows": (lo, hi), "kv_group": grp, "dims": dims,
            "source_layer": src, "v_gap": gap, "v_mean_a": float(np.mean(sa) / gap), "v_mean_b": float(np.mean(sb) / gap),
            "q_norm": float(np.mean(qn)), "n_trigger_tokens": len(kt)}


def write_query_lora(session, L: int, h: int, qv: dict, scale: float, path: str | Path) -> dict:
    """attn_q に rank-1 の LoRA: ΔW[q の行] = scale·‖q‖·u · vᵀ。v·norm(x) は A − B の差が 1 なので、ペルソナが入ると q が
    だいたい scale·‖q‖ だけ u に寄る。"""
    from gguf import GGUFValueType, GGUFWriter
    w = session.weights
    m = w.meta
    n_out = int(w.tensor(f"blk.{L}.attn_q.weight").shape[1])
    lo, hi = qv["rows"]
    B = np.zeros((n_out, 1), dtype=np.float32)
    B[lo:hi, 0] = scale * qv["q_norm"] * qv["u"]
    A = qv["v"].reshape(1, -1).astype(np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wr = GGUFWriter(str(path), arch=m.arch)
    wr.add_type("adapter")
    wr.add_string("adapter.type", "lora")
    wr.add_name(f"{m.name} query L{L} H{h}")
    wr.add_key_value("gne.query", json.dumps({"layer": L, "head": h, "scale": scale, "v_gap": qv["v_gap"],
                                              "q_norm": qv["q_norm"]}), GGUFValueType.STRING)
    wr.add_key_value("gne.source_model_id", m.model_id, GGUFValueType.STRING)
    wr.add_tensor(f"blk.{L}.attn_q.weight.lora_a", A)
    wr.add_tensor(f"blk.{L}.attn_q.weight.lora_b", B)
    wr.write_header_to_file()
    wr.write_kv_data_to_file()
    wr.write_tensors_to_file()
    wr.close()
    return {"path": str(path), "bytes": Path(path).stat().st_size}


def check_query(session, L: int, h: int, *, direction: str, seek: str | Path,
                scales=(0.0, 0.5, 1.0, 2.0), frac: float = 0.25) -> dict:
    """V17: クエリ操作の倍率を上げると、ヘッド h の trigger_mass(seek の A 側の会話、ユーザーの発話の最後のトークンから
    トリガーへの注目の割合)が単調に増え、同じ層の他のヘッドの変化は小さいか。"""
    from .direction import load_contrast
    t0 = time.perf_counter()
    qv = query_vectors(session, L, h, direction=direction, seek=seek, frac=frac)
    cs, _ = load_contrast(seek)
    trig = load_triggers(seek, cs)
    e = session.engine
    convs = []
    for side, system in (("A", cs.system_a), ("B", cs.system_b)):
        for u_text, tr in zip(cs.user_prompts, trig, strict=True):
            if tr:
                convs.append((side, u_text, tr, e.tokenize_prompt(e.render_chat(
                    [{"role": "system", "content": system}, {"role": "user", "content": u_text}]))))
    rows = []
    for sc in scales:
        lora = None
        if sc:
            p = session.store.heads_dir(session.meta.model_id) / f"query-L{L}-H{h}-s{sc:g}.lora.gguf"
            write_query_lora(session, L, h, qv, sc, p)
            lora = [(str(p), 1.0)]
        mass = {"A": [], "B": []}
        for side, u_text, tr, ids in convs:
            ac = capture_attention(session, [ids], layers=[L], want_head_out=False, lora=lora)[0]
            at = ac.text.rfind(u_text)
            tt = spans_to_tokens(ac.offsets, [(s + at, t + at) for s, t in find_spans(u_text, tr)])
            if not tt:
                continue
            P = ac.pattern[L][:, -1, :]                                     # 最後のトークン(返事の直前)の行
            mass[side].append(P[:, tt].sum(axis=1))
        ma, mb = np.mean(mass["A"], axis=0), np.mean(mass["B"], axis=0)
        rows.append({"scale": float(sc), "head_a": float(ma[h]), "head_b": float(mb[h]),
                     "others_a": np.delete(ma, h).round(5).tolist()})
    e.free_attn_probe()
    base = np.array(rows[0]["others_a"])
    for r in rows:
        r["others_shift_max"] = float(np.abs(np.array(r["others_a"]) - base).max())
    ha = [r["head_a"] for r in rows]
    mono = all(b >= a - 1e-6 for a, b in zip(ha, ha[1:], strict=False))
    return {"layer": L, "head": h, "direction": direction, "seek": str(seek), "kv_group": qv["kv_group"],
            "dims": [int(qv["dims"][0]), int(qv["dims"][-1]) + 1, int(len(qv["dims"]))],
            "v_gap": qv["v_gap"], "v_mean_a": qv["v_mean_a"], "v_mean_b": qv["v_mean_b"], "q_norm": qv["q_norm"],
            "n_trigger_tokens": qv["n_trigger_tokens"], "rows": rows, "monotonic": mono,
            "gain_a": ha[-1] - ha[0], "gain_b": rows[-1]["head_b"] - rows[0]["head_b"],
            "seconds": time.perf_counter() - t0}
