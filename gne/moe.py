"""MoE(Mixture of Experts)の捕捉の読み方(CHANGE-05 §B)。

MoE の層では、選ばれた専門家のニューロン活性(ffn_moe_swiglu など)だけがグラフに出る。選んだ専門家の番号
(ffn_moe_topk、I32)と組で読み、(専門家 e, ニューロン n) に散らす。選ばれなかった専門家は「未観測」のまま。

捕捉の形(Capture は ggml の [ne0, ne1, ne2] を numpy (ne1·ne2, ne0) で返す。トークンが一番外側):
    ffn_moe_swiglu-L   [n_ff_exp, n_used, T]  → (T·n_used, n_ff_exp)  → (T, n_used, n_ff_exp)
    ffn_moe_argsort-L  [n_expert, T](I32)      → (T, n_expert)。選んだ専門家は先頭の n_used 列
                       (ffn_moe_topk はその view で並びが続いていないので捕捉できない、verification.md §17)
    ffn_moe_weights_*  [1, n_used, T]          → (T·n_used, 1)          → (T, n_used)(正規化した重みは [n_used, T] → (T, n_used))
    ffn_moe_weighted-L [n_embd, n_used, T]     → (T·n_used, n_embd)     → (T, n_used, n_embd)
"""

from __future__ import annotations

import numpy as np

# ffn_down_exps の入力(専門家のニューロン活性)。build_moe_ffn で活性化の後に付く名前
MOE_ACT_CANDIDATES = ("ffn_moe_swiglu_limited", "ffn_moe_swiglu", "ffn_moe_geglu", "ffn_moe_gelu", "ffn_moe_silu",
                      "ffn_moe_relu", "ffn_moe_swiglu_oai")
# 実効のルーティング重み。arch により正規化・倍率が後に付くので、最後に現れるものを採る(0 項 3)
MOE_WEIGHT_CANDIDATES = ("ffn_moe_weights_scaled", "ffn_moe_weights_norm", "ffn_moe_weights_softmax", "ffn_moe_weights")


def moe_names(names: set[str], L: int) -> dict[str, str]:
    """層 L の MoE の捕捉名: act・argsort・weights(実効)・weighted・out・logits・probs。見つからないものは入れない。"""
    out = {}
    for base in MOE_ACT_CANDIDATES:
        if f"{base}-{L}" in names:
            out["act"] = f"{base}-{L}"
            break
    for base in MOE_WEIGHT_CANDIDATES:
        if f"{base}-{L}" in names:
            out["weights"] = f"{base}-{L}"
            break
    for k, base in (("argsort", "ffn_moe_argsort"), ("weighted", "ffn_moe_weighted"), ("out", "ffn_moe_out"),
                    ("logits", "ffn_moe_logits"), ("probs", "ffn_moe_probs")):
        if f"{base}-{L}" in names:
            out[k] = f"{base}-{L}"
    return out


def selected(argsort: np.ndarray, n_used: int) -> np.ndarray:
    """ffn_moe_argsort (T, n_expert) → 選んだ専門家 (T, n_used)(ルーティングの重みの大きい順)。"""
    return np.ascontiguousarray(argsort[:, :n_used])


def per_token(arr: np.ndarray, n_used: int, T: int) -> np.ndarray:
    """(T·n_used, d) → (T, n_used, d)。d = 1 なら (T, n_used)。2D の [n_used, T] で出る arch(正規化した重み)は
    捕捉で (T, n_used) になっているので、そのまま返す。T はトークン数(argsort の行数)。"""
    if arr.shape == (T, n_used):
        return arr
    if arr.shape[0] != T * n_used:
        raise ValueError(f"MoE の捕捉の形 {arr.shape} がトークン数 {T} × 専門家 {n_used} と合いません")
    x = arr.reshape(T, n_used, arr.shape[1])
    return x[..., 0] if x.shape[2] == 1 else x


def scatter(act: np.ndarray, topk: np.ndarray, n_expert: int) -> np.ndarray:
    """(T, n_used, n_ff) と (T, n_used) → (T, n_expert, n_ff)。選ばれなかった専門家は NaN(未観測)。"""
    T, used, n_ff = act.shape
    out = np.full((T, n_expert, n_ff), np.nan, dtype=np.float32)
    rows = np.repeat(np.arange(T), used)
    out[rows, topk.reshape(-1)] = act.reshape(T * used, n_ff)
    return out


def by_expert(act: np.ndarray, topk: np.ndarray, n_expert: int) -> list[np.ndarray]:
    """専門家ごとに、選ばれたトークンの活性 (n_e, n_ff) を返す(統計の積算用。NaN の大きな配列を作らない)。"""
    flat_a = act.reshape(-1, act.shape[-1])
    flat_e = topk.reshape(-1)
    return [flat_a[flat_e == e] for e in range(n_expert)]


def usage(topk: np.ndarray, weights: np.ndarray | None, n_expert: int) -> tuple[np.ndarray, np.ndarray]:
    """専門家ごとの選ばれた回数 (n_expert,) と、選ばれたときのルーティング重みの和 (n_expert,)。"""
    cnt = np.bincount(topk.reshape(-1), minlength=n_expert).astype(np.int64)
    wsum = np.zeros(n_expert, np.float64) if weights is None else \
        np.bincount(topk.reshape(-1), weights=weights.reshape(-1).astype(np.float64), minlength=n_expert)
    return cnt, wsum
