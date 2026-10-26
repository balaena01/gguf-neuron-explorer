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


def fire_scores(session, ids: list[int], layers: list[int] | None = None):
    """1 回の prefill で層ごとの活性を捕捉し、基準で割ったスコアにする。
    返り値: (層の一覧, スコア [(T-1, n_ff)], 生の活性 [(T-1, n_ff)], 層 → 基準の種類)。position 0 は除く。"""
    from .stats import ActStats
    e = session.engine
    names = e.list_names(ids[:16])
    Ls = [L for L in (range(e.n_layer) if layers is None else layers) if pick_act_name(names, L)]
    act = {pick_act_name(names, L): L for L in Ls}
    cap = Capture(lambda nm: nm in act)
    ctx = e.probe_ctx(cap)
    e.clear_memory(ctx)
    e.set_cvec(ctx, None)
    e.prefill(ctx, [ids])
    out = cap.take()
    scores, raw, norm_src, kept = [], [], {}, []
    for name, L in act.items():
        a = out[name]
        if a.shape[0] != len(ids):                 # 最終層の行の間引き(R5)
            continue
        ap = session.store.act_path(session.meta.model_id, L)
        if ap.exists():
            st = ActStats.load(ap)
            base, src = (st.q999, "q999") if np.all(st.q999 > 0) else (st.absmax, "absmax")
        else:
            base, src = np.full(a.shape[1], np.percentile(np.abs(a[1:]), 99)), "text_p99"
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
    kept, scores, raw, norm_src = fire_scores(session, ids, layers)
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
    if keep_max:
        out["ids"] = ids
        out["max_field"] = {L: s[np.argmax(np.abs(s), axis=0), np.arange(s.shape[1])].astype(np.float32)
                            for L, s in zip(kept, scores, strict=True)}
    return out
