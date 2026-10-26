"""逆引き: 対照プロンプト(目標ペルソナ vs 中立)から方向を作り、全層・全ニューロンを類似度で順位づけする。設計書 §6.9。

見つかるのは「残差空間でそのペルソナに相関する方向」に書き込むニューロンで、各ニューロンは多義的なので、
候補は steer lens と rollout(対照群つき)で検証する(--verify)。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from .capture import Capture, pick_act_name
from .engine import Engine
from .steer import GneError

Progress = Callable[[int, int, str], None]
Positions = Literal["all", "last", "assistant"]


@dataclass
class ContrastSet:
    name: str
    system_a: str
    system_b: str
    user_prompts: list[str]
    sha: str

    @classmethod
    def load(cls, path: str | Path) -> ContrastSet:
        p = Path(path)
        if not p.exists():
            raise GneError("FILE_NOT_FOUND", f"対照プロンプト集 {p} がありません")
        text = p.read_text(encoding="utf-8")
        d = yaml.safe_load(text) or {}
        if not d.get("user_prompts") or "system_a" not in d or "system_b" not in d:
            raise GneError("BAD_CONTRAST", f"{p}: system_a / system_b / user_prompts が要ります")
        return cls(d.get("name") or p.stem, d["system_a"], d["system_b"], list(d["user_prompts"]),
                   hashlib.sha256(text.encode()).hexdigest()[:16])


def _conv(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def contrast_sequences(engine: Engine, cs: ContrastSet) -> dict[str, list[tuple[list[int], int]]]:
    """{"A": [(ids, assistant の開始位置)], "B": [...]}。開始位置 = 生成プロンプト(assistant 冒頭)の最初のトークン。"""
    out: dict[str, list[tuple[list[int], int]]] = {"A": [], "B": []}
    for side, system in (("A", cs.system_a), ("B", cs.system_b)):
        for u in cs.user_prompts:
            msgs = _conv(system, u)
            ids = engine.tokenize_prompt(engine.render_chat(msgs))
            head = engine.tokenize_prompt(engine.render_chat(msgs, add_generation_prompt=False))
            start = len(head) if ids[:len(head)] == head else len(ids) - 1
            out[side].append((ids, min(start, len(ids) - 1)))
    return out


def _rows(x: np.ndarray, start: int, positions: Positions) -> np.ndarray:
    if positions == "all":
        return x[1:]                     # position 0 は attention sink なので除く(§5.6)
    if positions == "last":
        return x[-1:]
    return x[max(start, 1):]


@dataclass
class ContrastResult:
    directions: dict[int, np.ndarray]        # 層 → 方向(A − B)
    act_diff: dict[int, np.ndarray]          # 層 → (n_ff,) mean(a|A) − mean(a|B)
    method: str
    positions: str
    n_rows: dict[str, int]
    scale: dict[int, np.ndarray]             # 層 → (n_embd,) 残差の次元ごとの標準偏差(A∪B)


def contrastive_directions(engine: Engine, cs: ContrastSet, *, layers: Sequence[int] | None = None,
                           method: Literal["mean_diff", "pca1"] = "mean_diff", positions: Positions = "all",
                           with_act: bool = True, progress: Progress | None = None) -> ContrastResult:
    """目標群 A と中立群 B を probe ctx で 1 本ずつ prefill し、l_out-L の平均差(または差の第一主成分)と、
    ffn_down 入力の平均差(activation_contrast)を同じ順伝播で取る。最終層は行が間引かれ得るので除く。"""
    Ls = [L for L in (range(engine.n_layer - 1) if layers is None else layers) if 0 <= L < engine.n_layer - 1]
    seqs = contrast_sequences(engine, cs)
    names = engine.list_names(seqs["A"][0][0][:16])
    act_names = {}
    if with_act:
        for L in Ls:
            nm = pick_act_name(names, L)
            if nm:
                act_names[nm] = L
    resid_names = {f"l_out-{L}": L for L in Ls}
    cap = Capture(lambda n: n in resid_names or n in act_names)
    ctx = engine.probe_ctx(cap)
    engine.set_cvec(ctx, None)
    sums = {side: {L: 0.0 for L in Ls} for side in "AB"}
    sq = {L: 0.0 for L in Ls}                                  # A∪B の行の二乗和(次元ごとのばらつき用)
    act_sums = {side: {L: 0.0 for L in act_names.values()} for side in "AB"}
    per_prompt = {side: {L: [] for L in Ls} for side in "AB"}
    counts = {"A": 0, "B": 0}
    total, step = len(seqs["A"]) + len(seqs["B"]), 0
    for side in "AB":
        for ids, start in seqs[side]:
            ctx = engine.probe_ctx(cap)
            engine.clear_memory(ctx)
            engine.prefill(ctx, [ids])
            out = cap.take()
            n_rows = None
            for name, L in resid_names.items():
                r = _rows(out[name], start, positions).astype(np.float64)
                sums[side][L] = sums[side][L] + r.sum(axis=0)
                sq[L] = sq[L] + np.square(r).sum(axis=0)
                per_prompt[side][L].append(r.mean(axis=0))
                n_rows = r.shape[0]
            for name, L in act_names.items():
                act_sums[side][L] = act_sums[side][L] + _rows(out[name], start, positions).astype(np.float64).sum(axis=0)
            counts[side] += n_rows or 0
            step += 1
            if progress:
                progress(step, total, "reverse")
    dirs: dict[int, np.ndarray] = {}
    for L in Ls:
        md = sums["A"][L] / counts["A"] - sums["B"][L] / counts["B"]
        if method == "mean_diff":
            dirs[L] = md.astype(np.float32)
        elif method == "pca1":
            D = np.stack(per_prompt["A"][L]) - np.stack(per_prompt["B"][L])      # (n_prompts, n_embd)
            pc = np.linalg.svd(D, full_matrices=False)[2][0]
            pc *= np.sign(pc @ md) or 1.0
            dirs[L] = (pc * (pc @ md)).astype(np.float32)                         # 平均差をその軸に射影した大きさ
        else:
            raise GneError("BAD_ARGS", f"method={method!r} は mean_diff か pca1 です")
    act = {L: (act_sums["A"][L] / counts["A"] - act_sums["B"][L] / counts["B"]).astype(np.float32)
           for L in act_names.values()}
    n = counts["A"] + counts["B"]
    scale = {}
    for L in Ls:
        mu = (sums["A"][L] + sums["B"][L]) / n
        scale[L] = np.sqrt(np.maximum(sq[L] / n - mu ** 2, 0)).astype(np.float32)
    return ContrastResult(dirs, act, method, positions, counts, scale)


@dataclass
class Candidate:
    layer: int
    neuron: int
    cos: float
    sign: int                 # +1: + で押すと目標方向、-1: − で押すと目標方向
    act_diff: float | None
    steer_kl: float | None
    rank_score: float


def cos_by_layer(weights, directions: dict[int, np.ndarray],
                 scale: dict[int, np.ndarray] | None = None) -> dict[int, np.ndarray]:
    """cos(u_n, direction[L]) を層ごとに (n_ff,) で。
    scale(残差の次元ごとの標準偏差)があれば、両方をその次元で割った空間で cos を取る。少数の次元だけが
    飛び抜けて大きい(massive activation)モデルで、その次元に書き込むニューロンが上位を占めるのを防ぐ(F17)。"""
    out = {}
    for L, d in directions.items():
        U = weights.unit_matrix(L)
        if scale is not None and L in scale:
            s = np.maximum(scale[L], 1e-6 * float(scale[L].max()) + 1e-12)
            d = d / s
            U = U / s[:, None]
            U = U / np.linalg.norm(U, axis=0, keepdims=True)
        nrm = float(np.linalg.norm(d))
        out[L] = (U.T @ (d / nrm)).astype(np.float32) if nrm > 0 else np.zeros(U.shape[1], np.float32)
    return out


def _pct_rank(x: np.ndarray) -> np.ndarray:
    """大きいほど 1 に近い順位(0〜1)。"""
    o = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return o / max(len(x) - 1, 1)


def combine_scores(cos: np.ndarray, act_diff: np.ndarray | None, steer_kl: np.ndarray | None) -> np.ndarray:
    """順位の平均。|cos|、cos の符号にそろえた act_diff、steer KL のうち、ある指標だけを使う(NaN は除く)。"""
    parts = [_pct_rank(np.abs(cos))]
    for x in (None if act_diff is None else act_diff * np.sign(cos), steer_kl):
        if x is None:
            continue
        ok = np.isfinite(x)
        if ok.any():
            r = np.full(len(x), np.nan)
            r[ok] = _pct_rank(x[ok])
            parts.append(r)
    return np.nanmean(np.stack(parts), axis=0)


def rank_by_direction(weights, directions: dict[int, np.ndarray], *, act_diff: dict[int, np.ndarray] | None = None,
                      steer_kl: dict[int, tuple[np.ndarray, np.ndarray]] | None = None, top: int = 200,
                      exclude_layers: Sequence[int] = (), scale: dict[int, np.ndarray] | None = None,
                      ) -> tuple[list[Candidate], dict[int, np.ndarray]]:
    """全層・全ニューロンを順位づけし、上位 top 件と層ごとの cos を返す。
    steer_kl は層 → (kl_pos, kl_neg)。cos の符号に合う側(+ なら kl_pos)を使う。"""
    cos = cos_by_layer(weights, directions, scale)
    Ls = [L for L in sorted(cos) if L not in set(exclude_layers)]
    nan = {L: np.full(cos[L].size, np.nan, np.float32) for L in Ls}
    lay = np.concatenate([np.full(cos[L].size, L) for L in Ls])
    neu = np.concatenate([np.arange(cos[L].size) for L in Ls])
    c = np.concatenate([cos[L] for L in Ls])
    a = np.concatenate([act_diff.get(L, nan[L]) for L in Ls]) if act_diff else None
    k = None
    if steer_kl:
        k = np.concatenate([np.where(cos[L] >= 0, *steer_kl[L]) if L in steer_kl else nan[L] for L in Ls])
    score = combine_scores(c, a, k)
    order = np.argsort(-score, kind="stable")[:top]
    cands = [Candidate(int(lay[i]), int(neu[i]), float(c[i]), 1 if c[i] >= 0 else -1,
                       None if a is None or not np.isfinite(a[i]) else float(a[i]),
                       None if k is None or not np.isfinite(k[i]) else float(k[i]), float(score[i])) for i in order]
    return cands, cos


def candidate_dicts(cands: list[Candidate]) -> list[dict]:
    return [asdict(c) for c in cands]
