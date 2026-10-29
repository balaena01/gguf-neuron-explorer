"""fires: 文章を流して、トークンごとにどの層のどのニューロンが光るかを出す。別冊 UI 設計 §5.5(本体側の部品)。

1 回の prefill で全層の ffn_down 入力(ffn_swiglu-L 等)を捕捉し、活性をニューロンごとの基準で割ってスコアにする。
基準は act_L{L}.npz の q999、無ければ absmax、どちらも無ければこの文章の中の層内 p99(警告つき)。position 0 は除く。
"""

from __future__ import annotations

import numpy as np

from .capture import Capture, pick_act_name
from .steer import GneError

MAX_TOKENS = 512


def fire_ids(session, text: str) -> list[int]:
    e = session.engine
    ids = e.tokenize(text, add_special=True, parse_special=False)
    if len(ids) < 2:
        raise GneError("BAD_ARGS", "文章が短すぎます(2 トークン以上)")
    if len(ids) > MAX_TOKENS:
        session.warn(f"{len(ids)} トークンのうち先頭 {MAX_TOKENS} だけを使いました")
        ids = ids[:MAX_TOKENS]
    return ids


def fire_scores(session, ids: list[int], layers: list[int] | None = None, routes: dict | None = None):
    """1 回の prefill で層ごとの活性を捕捉し、基準で割ったスコアにする。
    返り値: (層の一覧, スコア [(T-1, n_ff)], 生の活性 [(T-1, n_ff)], 層 → 基準の種類)。position 0 は除く。
    MoE の層(CHANGE-05 §B.2)は、選ばれた専門家の活性を層の通し番号(e·n_ff_exp + n)に散らし、選ばれなかった専門家は 0。
    routes に dict を渡すと、MoE の層ごとに選ばれた専門家 (T-1, n_used) を入れる。"""
    from .moe import moe_names, per_token, selected
    from .stats import ActStats
    e = session.engine
    names = e.list_names(ids[:16])
    cand = range(e.n_layer) if layers is None else layers
    act = {pick_act_name(names, L): L for L in cand if pick_act_name(names, L)}
    moe = {}
    for L in cand:
        if pick_act_name(names, L) is None:
            mn = moe_names(names, L)
            if "act" in mn and "argsort" in mn:
                moe[L] = mn
    want = set(act) | {mn[k] for mn in moe.values() for k in ("act", "argsort")}
    cap = Capture(lambda nm: nm in want)
    ctx = e.probe_ctx(cap)
    e.clear_memory(ctx)
    e.set_cvec(ctx, None)
    e.prefill(ctx, [ids])
    out = cap.take()
    T = len(ids)
    full: dict[int, np.ndarray] = {}
    for name, L in act.items():
        if out[name].shape[0] == T:                 # 最終層の行の間引き(R5)
            full[L] = out[name]
    for L, mn in moe.items():
        a, arg = out.get(mn["act"]), out.get(mn["argsort"])
        if a is None or arg is None or arg.shape[0] != T or a.shape[0] % T:
            continue
        n_used = a.shape[0] // T
        n_exp, n_ff = arg.shape[1], a.shape[1]
        sel = selected(arg, n_used)
        flat = np.zeros((T, n_exp * n_ff), np.float32)
        cols = (sel[:, :, None] * n_ff + np.arange(n_ff)[None, None, :]).reshape(T, -1)
        flat[np.repeat(np.arange(T), n_used * n_ff).reshape(T, -1), cols] = per_token(a, n_used, T).reshape(T, -1)
        full[L] = flat
        if routes is not None:
            routes[L] = sel[1:]
    scores, raw, norm_src, kept = [], [], {}, []
    for L in sorted(full):
        a = full[L]
        ap = session.store.act_path(session.meta.model_id, L)
        if ap.exists():
            st = ActStats.load(ap)
            q, mx = st.q999.reshape(-1), st.absmax.reshape(-1)
            base, src = (q, "q999") if np.all(q > 0) else (mx, "absmax")
            if L in moe and not np.all(base > 0):     # 観測の無い専門家(n_routed = 0)は文章の中の p99
                nz = np.abs(a[1:][a[1:] != 0])
                base = np.where(base > 0, base, np.percentile(nz, 99) if nz.size else 1.0)
        else:
            nz = np.abs(a[1:][a[1:] != 0]) if L in moe else np.abs(a[1:])
            base, src = np.full(a.shape[1], np.percentile(nz, 99) if nz.size else 1.0), "text_p99"
        norm_src[str(L)] = src
        scores.append(a[1:] / np.maximum(base, 1e-6)[None, :])
        raw.append(a[1:])
        kept.append(L)
    if not kept:
        raise GneError("ARCH_UNSUPPORTED", "ffn_down の入力テンソルを捕捉できませんでした")
    if any(v == "text_p99" for v in norm_src.values()):
        session.warn("活性統計が無い層は、この文章の中の層内 p99 で割りました(gne stats act で統計を作ると比べやすくなります)")
    return kept, scores, raw, norm_src


def fires(session, text: str, *, top: int = 50, layers: list[int] | None = None, keep_max: bool = False) -> dict:
    """keep_max=True なら、層ごとに全トークンで |スコア| が最大の値(符号つき)も返す(UI のマップ用)。"""
    e = session.engine
    ids = fire_ids(session, text)
    routes: dict = {}
    kept, scores, raw, norm_src = fire_scores(session, ids, layers, routes)
    S = np.concatenate(scores, axis=1)             # (T-1, Σ n_ff)
    A = np.concatenate(raw, axis=1)
    offs = np.cumsum([0] + [s.shape[1] for s in scores])
    lay = np.concatenate([np.full(s.shape[1], L) for s, L in zip(scores, kept, strict=True)])
    neu = np.concatenate([np.arange(s.shape[1]) for s in scores])
    k = min(top, S.shape[1])
    idx = np.argpartition(-np.abs(S), k - 1, axis=1)[:, :k]
    order = np.argsort(-np.abs(np.take_along_axis(S, idx, 1)), axis=1)
    idx = np.take_along_axis(idx, order, 1)
    toks = [e.token_text(t, special=False) for t in ids]
    per_token = [{"pos": t + 1, "token": toks[t + 1],                    # position 0 は除いている
                  "neurons": [{"layer": int(lay[j]), "neuron": int(neu[j]), "score": float(S[t, j]),
                               "act": float(A[t, j])} for j in idx[t]]} for t in range(S.shape[0])]
    layer_max = {str(L): float(np.abs(S[:, offs[i]:offs[i + 1]]).max()) for i, L in enumerate(kept)}
    out = {"tokens": toks, "n_tokens": len(ids), "layers": kept, "normalization": norm_src,
           "top": per_token, "layer_max": layer_max, "skipped_position0": True}
    if routes:                                     # MoE: トークンごとに選ばれた専門家(position 1 から)
        out["routed"] = {str(L): r.tolist() for L, r in sorted(routes.items())}
        for row in per_token:
            for x in row["neurons"]:
                if x["layer"] in routes:
                    x["expert"], x["expert_neuron"] = session.weights.moe_split(x["layer"], x["neuron"])
    if keep_max:
        out["ids"] = ids
        out["max_field"] = {L: s[np.argmax(np.abs(s), axis=0), np.arange(s.shape[1])].astype(np.float32)
                            for L, s in zip(kept, scores, strict=True)}
    return out
