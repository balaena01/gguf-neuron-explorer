"""最小二乗の鍵(key_fit = "ridge")。FiSH の位置の鍵(target mlpin、positions pos:<範囲>)の、もう 1 つの作り方(docs/fish.md §3)。

いまの作り方(key_fit = "mean"、keyed.make_key)は、書き込みたい位置の状態の平均 μ から、書き込みたくない位置の状態が張る部分空間の
成分を取り除く。こちらは、位置 1 つずつを行にして、「書き込みたい位置で 1、書き込みたくない位置で 0」を返す κ を、正則化つきの
最小二乗の閉じた式で解く(勾配による学習はしない。切片なし):

    κ̂ = argmin_κ  Σ_{i∈W} w_i (κ·x_i − 1)² + Σ_{i∈U} w_i (κ·x_i)² + λ‖κ‖²  =  (G + λI)⁻¹ b
    G = Σ_i w_i x_i x_iᵀ(W と U の全部)、b = Σ_{i∈W} w_i x_i、λ = c · trace(G) ÷ 次元
    κ = κ̂ ÷ (κ̂·m_W)、m_W = W の重みつき平均(書き込みたい位置での反応の平均を 1 に。いまの鍵と同じ)

W = 狙う場面の返事の、その鍵の区間の位置。U = 狙う場面の返事のほかの位置 + 変えたくない場面(dont_touch の群)の返事の全部の位置。

行の重み(ますごとの重み): W の重みの合計は 1、U の重みの合計も 1。U は「場面の群 × 位置の区分」のますに分け、どのますも合計が同じ
(1 ÷ ますの数)になるように付ける。場面の群 = dont_touch の群(name)と、狙う場面(ほかの区間の位置)。位置の区分 = 鍵の区間
(pos:0 / pos:1-3 / pos:4- を一緒に作るなら、その 3 つ。区間の外の位置は、続いた範囲ごとに別の区分)。位置を全部同じ重みで数えると、
行の数が多います(長い返事の後ろの位置)が重みのほとんどを持ち、数の少ないます(たとえば、ある種類の発言の、返事を書き始める直前の
位置)では反応を 0 に抑えられない。

行の間引き: 返事ごとに、位置 0〜KEEP_ALL−1 は全部、それ以降は SAMPLE 位置(乱数で選び、重み ω = それ以降の位置の数 ÷ SAMPLE)。

正則化の強さ c: 省くと、鍵を作る質問だけを使い、質問単位の FOLDS 分割で決める(候補は CS)。分割ごとに、残りの組で κ を解き、
外した組の位置での反応 r = κ·x から、E = Σ_{W} w·1(r < 上のしきい値) + Σ_{U} w·1(r > 下のしきい値) を数える(しきい値は上限つきの形の
既定の 0.4・0.8)。全部の層・区間の E の合計が最小の c(同じなら大きいほう)。全層・全区間で 1 つ。

比較用の鍵: 狙う場面の質問ごとに「その区間の位置の行」と「ほかの位置の行」の 2 つのまとまりを作り、どちらが書き込みたい側かの
ラベルを、いまの鍵と同じ並べ替え(keyed_build.balanced_split)で入れ替えて、同じ式・同じ c で解く。dont_touch の行はいつも U。

メモリ: 行は層ごとに float16(収まらない値があれば float32)で持ち、鍵の計算は層 1 つずつ(その層の行だけを float64 にして、
正規方程式に積む)。
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence

import numpy as np

from .steer import GneError

KEY_FITS = ("mean", "ridge")                 # mean = いまの作り方(既定)、ridge = 最小二乗の鍵(ますごとの重み)
CS = (0.01, 0.1, 1.0, 10.0, 100.0)           # 正則化の強さ c の候補(λ = c × 行列の対角の平均)
FOLDS = 5
KEEP_ALL = 16                                # 返事ごとに、位置 0〜15 は全部の行を使う
SAMPLE = 8                                   # それ以降は 8 位置
THRESHOLDS = (0.4, 0.8)                      # c を決めるときに数える反応のしきい値(上限つきの形の既定)
# 間引きと分割の乱数の種の元。seed = 0(既定)のとき、docs/verification.md §22.21 の測定と同じ行・同じ分割になる
SEED0 = 16


def check_key_fit(key_fit) -> str:
    kf = "mean" if key_fit is None else str(key_fit)
    if kf not in KEY_FITS:
        raise GneError("BAD_ARGS", f"鍵の作り方(key_fit)は {' / '.join(KEY_FITS)} です({key_fit!r})",
                       "mean = いまの作り方(既定)、ridge = 最小二乗の鍵(ますごとの重み)")
    return kf


def check_c(key_c) -> float | None:
    if key_c is None or (isinstance(key_c, str) and not key_c.strip()):
        return None
    try:
        c = float(key_c)
    except (TypeError, ValueError):
        raise GneError("BAD_ARGS", f"正則化の強さ(key_c){key_c!r} を読めません", "0 より大きい数です(例 1)。省くと 5 分割で決めます") from None
    if not np.isfinite(c) or c <= 0:
        raise GneError("BAD_ARGS", f"正則化の強さ(key_c)は 0 より大きい数です({key_c!r})", "省くと 5 分割で決めます")
    return c


def keep_positions(n: int, rng, must: Sequence[int] = ()) -> tuple[np.ndarray, np.ndarray]:
    """返事の位置 0〜n−1 の間引き → (残す位置, 重み ω)。位置 0〜KEEP_ALL−1 は全部(ω = 1)、それ以降は SAMPLE 位置を乱数で選び、
    ω = (それ以降の位置の数) ÷ SAMPLE(SAMPLE 位置以下なら全部、ω = 1)。must の位置(after:<正規表現> の目標の位置)は
    いつも残し(ω = 1)、乱数で選ぶのはそれ以外の位置から(must が空なら今までと同じ乱数・同じ位置)。"""
    ks = list(range(min(n, KEEP_ALL)))
    w = [1.0] * len(ks)
    keep = sorted({int(k) for k in must if KEEP_ALL <= int(k) < n})
    rest = [k for k in range(KEEP_ALL, n) if k not in set(keep)] if keep else list(range(KEEP_ALL, n))
    if len(rest) > SAMPLE:
        pick = sorted(rng.choice(rest, SAMPLE, replace=False).tolist())
        ks += pick
        w += [len(rest) / float(SAMPLE)] * SAMPLE
    else:
        ks += rest
        w += [1.0] * len(rest)
    if keep:
        order = np.argsort(np.asarray(ks + keep), kind="stable")
        return np.asarray(ks + keep, dtype=np.int64)[order], np.asarray(w + [1.0] * len(keep), dtype=np.float64)[order]
    return np.asarray(ks, dtype=np.int64), np.asarray(w, dtype=np.float64)


class RowStore:
    """鍵の計算に渡す行(位置 1 つずつ)を、読み取りの間にためる。群(質問の組)ごとに begin、会話 1 本ごとに add。
    行の表: q(質問の通し番号)・pos(返事の位置)・om(間引きの重み)・match(after の目標の位置か)。質問の表: q_group(質問 →
    群の番号。0 = 狙う場面)・q_form(会話の形の番号。prior.FORMS の順)。
    after(正規表現)を渡すと、返事のトークンが一致した直後の位置を match にして、いつも残す(pieces_of = トークン → バイト列)。
    target_reply(正規表現)を渡すと、狙う場面の群のうち応答が一致しなかった質問を、別の群 scene_other(書き込みたくない側)にする。"""

    def __init__(self, layers: Sequence[int], seed: int = 0, *, after: str | None = None,
                 pieces_of: Callable[[int], bytes] | None = None, target_reply: str | None = None):
        self.layers = [int(L) for L in layers]
        self.seed = int(seed)
        self.groups: list[str] = []
        self.q_group: list[int] = []
        self.q_form: list[int] = []
        self._base = 0
        self.q: list[int] = []
        self.pos: list[int] = []
        self.om: list[float] = []
        self.match: list[bool] = []
        self._rows: dict[int, list[np.ndarray]] = {L: [] for L in self.layers}
        self.wide: set[int] = set()            # float16 に収まらない値があって、float32 で持った層
        if after is not None and pieces_of is None:
            raise GneError("BAD_ARGS", "after の位置を数えるには pieces_of(トークン → バイト列)が要ります")
        self.after = after
        self.pieces_of = pieces_of
        self.target_reply = re.compile(target_reply) if target_reply else None
        self.q_other: set[int] = set()         # 狙う場面の群で、応答が target_reply に一致しなかった質問
        self.n_target_reply = 0                # 狙う場面の群で、応答が target_reply に一致した質問の数

    def begin(self, label: str, n: int, forms: Sequence[str] | None = None) -> None:
        from .prior import FORMS
        self._base = len(self.q_group)
        self.groups.append(str(label))
        self.q_group += [len(self.groups) - 1] * int(n)
        fs = list(forms) if forms is not None else ["single"] * int(n)
        if len(fs) != int(n):
            raise GneError("BAD_ARGS", f"会話の形の数 {len(fs)} が会話の数 {n} と違います")
        self.q_form += [FORMS.index(f) for f in fs]

    def add(self, i: int, n_prompt: int, n_reply: int, rows_of: Callable[[int, np.ndarray], np.ndarray],
            reply_ids: Sequence[int] | None = None, text: str | None = None) -> None:
        """会話 i(いまの群の中の番号。プロンプト n_prompt トークン + 返事 n_reply トークン)の、間引いた位置の行を足す。
        rows_of(層, 列の番号の並び) がその位置の行を返す。位置 k = 返事の k 番目のトークンを決める位置(列の番号 n_prompt − 1 + k)。
        reply_ids(返事のトークン。EOG を除く)・text(返事の文字)は after・target_reply を数えるときに使う。"""
        from .keyed_build import match_tokens
        qi = self._base + int(i)
        matched: list[int] = []
        if self.after is not None and reply_ids is not None and n_reply:
            pcs = [self.pieces_of(int(t)) for t in list(reply_ids)[:int(n_reply)]]
            matched = [j + 1 for j in match_tokens(pcs, self.after)]        # 一致したトークン j の直後 = 位置 j + 1
        if self.target_reply is not None and self.q_group[qi] == 0:
            if self.target_reply.search(text or ""):
                self.n_target_reply += 1
            else:
                self.q_other.add(qi)
        ks, om = keep_positions(int(n_reply) + 1, np.random.default_rng(SEED0 + self.seed + qi), matched)
        sel = int(n_prompt) - 1 + ks
        for L in self.layers:
            x = np.asarray(rows_of(L, sel))
            if x.shape[0] != ks.size:
                raise GneError("ARCH_UNSUPPORTED", f"層 {L}: 位置ごとの行が取れません({x.shape[0]} 行、位置 {ks.size})")
            if L not in self.wide and float(np.abs(x).max(initial=0.0)) < 6e4:
                x = x.astype(np.float16)
            else:
                self.wide.add(L)
                x = x.astype(np.float32)
            self._rows[L].append(x)
        mset = set(matched)
        self.q += [qi] * ks.size
        self.pos += ks.tolist()
        self.om += om.tolist()
        self.match += [int(k) in mset for k in ks.tolist()]

    @property
    def n_rows(self) -> int:
        return len(self.q)

    def take(self, L: int) -> np.ndarray:
        """層 L の行(N × 次元)を返して、ためていた分を手放す(1 回だけ呼べる)。"""
        xs = self._rows.pop(L)
        return np.concatenate(xs) if xs else np.zeros((0, 0), np.float32)

    def group_names(self) -> list[str]:
        """群の名前(q_group の番号の順)。狙う場面 = scene、dont_touch の群 = その name、target_reply に外れた狙う場面 = scene_other。"""
        names = ["scene"] + [g[3:] if g.startswith("dt:") else g for g in self.groups[1:]]
        return names + (["scene_other"] if self.q_other else [])

    def table(self) -> dict:
        qg = np.asarray(self.q_group, dtype=np.int64)
        if self.q_other:
            qg = qg.copy()
            qg[sorted(self.q_other)] = len(self.groups)
        return {"q": np.asarray(self.q, dtype=np.int64), "pos": np.asarray(self.pos, dtype=np.int64),
                "om": np.asarray(self.om, dtype=np.float64), "q_group": qg,
                "q_form": np.asarray(self.q_form, dtype=np.int64), "match": np.asarray(self.match, dtype=bool)}


def bins_of(ranges: Sequence[tuple[int, int | None]]) -> list[tuple[int, int | None]]:
    """鍵の区間(lo 以上 hi 未満。hi が None なら最後まで)→ 位置の区分。区間そのものと、区間の外の続いた範囲(区間の手前・すき間・
    後ろ)をそれぞれ 1 つの区分にする。区間が重なっていればエラー。"""
    rs = sorted({(int(lo), None if hi is None else int(hi)) for lo, hi in ranges}, key=lambda r: r[0])
    out: list[tuple[int, int | None]] = []
    at = 0
    for lo, hi in rs:
        if at is None or lo < at:
            raise GneError("BAD_ARGS", "鍵の区間が重なっています(最小二乗の鍵は、重ならない区間で作ります)", "例 pos:0 / pos:1-3 / pos:4-")
        if lo > at:
            out.append((at, lo))
        out.append((lo, hi))
        at = hi
    if at is not None:
        out.append((at, None))
    return out


def _bin_index(pos: np.ndarray, bins) -> np.ndarray:
    b = np.full(pos.shape, -1, dtype=np.int64)
    for k, (lo, hi) in enumerate(bins):
        b[(pos >= lo) & ((pos < hi) if hi is not None else True)] = k
    return b


def bin_text(b) -> str:
    lo, hi = b
    return f"{lo}-" if hi is None else (str(lo) if hi == lo + 1 else f"{lo}-{hi - 1}")


def _folds(q_group: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """質問 → 組の番号。群ごとに、質問をかき混ぜて順に 0,1,…,folds−1 を振る(どの組にも、どの群も同じ割合で入る)。"""
    rng = np.random.default_rng(SEED0 + int(seed))
    out = np.zeros(q_group.size, dtype=np.int64)
    for g in range(int(q_group.max()) + 1 if q_group.size else 0):
        idx = np.flatnonzero(q_group == g)
        out[idx[rng.permutation(idx.size)]] = np.arange(idx.size) % folds
    return out


def _cell_weights(om: np.ndarray, cell: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int]:
    """mask の行に、ますごとの合計が 1 になる重み(ω ÷ そのますの ω の合計)を付ける。(重み, ますの数)。"""
    w = np.zeros(om.size)
    cs = np.unique(cell[mask])
    for c in cs:
        m = mask & (cell == c)
        w[m] = om[m] / om[m].sum()
    return w, int(cs.size)


def _solver():
    try:
        from scipy.linalg import cho_factor, cho_solve

        def solve(A, b):
            return cho_solve(cho_factor(A, overwrite_a=True, check_finite=False), b, check_finite=False)
    except ImportError:                        # scipy が無ければ numpy(LU。同じ答えで、少し遅い)
        def solve(A, b):
            return np.linalg.solve(A, b)
    return solve


def _solve_cs(G: np.ndarray, b: np.ndarray, m: np.ndarray, cs: Sequence[float], solve) -> tuple[np.ndarray, float]:
    """c ごとに κ = (G + c·τ·I)⁻¹ b ÷ (その κ̂·m)。τ = trace(G) ÷ 次元。→ ((c の数, 次元), τ)。"""
    D = G.shape[0]
    tau = float(np.trace(G)) / D
    out = np.zeros((len(cs), D))
    di = np.diag_indices(D)
    for j, c in enumerate(cs):
        A = G.copy()
        A[di] += float(c) * tau
        k = solve(A, b)
        den = float(k @ m)
        if not np.isfinite(den) or den <= 0:
            raise GneError("BAD_KEY", "最小二乗の鍵を解けませんでした(書き込みたい位置の状態の平均が 0 か、行列が壊れています)")
        out[j] = k / den
    return out, tau


def _gram(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (X * w[:, None]).T @ X


def fit(X_of: Callable[[int], np.ndarray], layers: Sequence[int], table: dict, ranges: Sequence[tuple[int, int | None]], *,
        cs: Sequence[float] = CS, c: float | None = None, folds: int = FOLDS, seed: int = 0, pseudo: int = 0,
        thresholds: tuple[float, float] = THRESHOLDS, weights: str = "cells", group_names: Sequence[str] | None = None,
        tick: Callable[[str], None] | None = None, conv_cells: bool = False, crit: float | None = None,
        fp_max: float | None = None) -> dict:
    """最小二乗の鍵を、区間ごと・層ごとに解く(モデルは使わない。numpy だけ)。

    X_of(層) → その層の行(N × 次元。float16 / float32 でよい)。table = {q, pos, om(行ごと), q_group(質問ごと。0 = 狙う場面)}。
    ranges = 鍵ごとの区間 [(lo, hi), …](位置 lo 以上 hi 未満。hi が None なら最後まで)。鍵 e の W は、狙う場面の、区間 e の行。
    c を渡すとその値を使う。省くと cs の候補から質問単位の folds 分割で決める(狙う場面の質問が folds より少なければ、その数で分ける)。
    pseudo = 鍵ごとの比較用の鍵の本数。ラベルの並べ替えは keyed_build.balanced_split(まとまりの番号は、区間の行がある質問
    (質問の番号順)が 0…、ほかの位置の行がある質問がその後ろ。乱数は seed で決まる)。c を決めたあとに作る。
    weights = "cells"(ますごとの重み。既定)/ "uniform"(U の位置を全部同じ重みで数える。比べるため)。
    ranges = ["after"] は after:<正規表現> の鍵(1 本だけ): W は狙う場面の、table の match の行。位置の区分は
    after / 0 / 1-3 / 4- (match の行が after、ほかは位置で分ける)。
    conv_cells = True で、セルを「群 × 会話の形 × 位置の区分」にする(table の q_form。既定は群 × 位置の区分)。分割は群 × 会話の形
    ごとに均等に振る(会話の形が 1 つなら今までと同じ)。
    crit(倍率)= 決め手のセル(dont_touch の群の、鍵の区分の位置)の重みを crit 倍にする。fp_max(割合)= 交差検証で、決め手の
    セルの誤検出(キーの出力 > 下のしきい値。セルごとの重みつきの割合の平均)が fp_max を超える層は、超えなくなるまで κ を縮める。
    crit・fp_max は鍵 1 本のときだけ。fp_max は c を分割で決めるときだけ。
    戻り値 {"kappa": [鍵ごとの {層: κ}], "mu": [鍵ごとの {層: m_W}], "ctrl": [鍵ごとの {層: (本数, 次元)}], "info": 記録}。
    info の cv_cells(分割で決めたとき)= 鍵ごと・層ごとの、セルごとの交差検証の数(W のセルは ≥ 上のしきい値の割合、
    ほかは > 下のしきい値の割合)。"""
    t0 = time.perf_counter()
    if weights not in ("cells", "uniform"):
        raise GneError("BAD_ARGS", f"weights は cells / uniform です({weights!r})")
    Ls = [int(L) for L in layers]
    q, pos, om, q_group = (np.asarray(table[k]) for k in ("q", "pos", "om", "q_group"))
    q_form = np.asarray(table.get("q_form", np.zeros(q_group.size, dtype=np.int64)), dtype=np.int64)
    N, nE = q.size, len(ranges)
    after = nE == 1 and ranges[0] == "after"
    if any(r == "after" for r in ranges) and not after:
        raise GneError("BAD_ARGS", "after:<正規表現> の鍵は 1 本ずつ作ります")
    if (crit is not None or fp_max is not None) and nE != 1:
        raise GneError("BAD_ARGS", "key_crit・key_fp_max は、鍵を 1 本ずつ作るときだけ使えます(区切りごとの鍵をまとめて作るときは不可)")
    if fp_max is not None and c is not None:
        raise GneError("BAD_ARGS", "key_fp_max は、正則化の強さを分割で決めるとき(key_c を省いたとき)だけ使えます")
    if crit is not None and (not np.isfinite(float(crit)) or float(crit) <= 0):
        raise GneError("BAD_ARGS", f"key_crit は 0 より大きい数です({crit!r})")
    if fp_max is not None and not 0 <= float(fp_max) < 1:
        raise GneError("BAD_ARGS", f"key_fp_max は 0 以上 1 未満の割合です({fp_max!r})")
    g = q_group[q]
    tgt = g == 0
    dt = ~tgt
    if after:
        match = np.asarray(table["match"], dtype=bool)
        b_i = np.where(match, 0, 1 + np.where(pos == 0, 0, np.where(pos <= 3, 1, 2))).astype(np.int64)
        bin_names = ["after", "0", "1-3", "4-"]
        key_bin = [0]
    else:
        bins = bins_of(ranges)
        b_i = _bin_index(pos, bins)
        bin_names = [bin_text(b) for b in bins]
        key_bin = [bins.index((int(lo), None if hi is None else int(hi))) for lo, hi in ranges]
    n_bins = len(bin_names)
    nF = int(q_form.max()) + 1 if q_form.size else 1
    gq = q_group * nF + q_form if conv_cells else q_group
    cell = gq[q] * n_bins + b_i
    n_groups = int(q_group.max()) + 1 if q_group.size else 1
    names0 = list(group_names) if group_names is not None else [str(k) for k in range(n_groups)]
    is_dt_group = np.asarray([n not in ("scene", "scene_other") for n in names0] + [True] * max(0, n_groups - len(names0)))
    decisive = dt & (b_i == key_bin[0]) & is_dt_group[g] if nE == 1 else np.zeros(N, dtype=bool)
    wd, n_dt = (_cell_weights(om, cell, dt) if weights == "cells" else (np.where(dt, om, 0.0), 0))
    n_dt_eff = float(n_dt)
    if crit is not None and weights == "cells":
        crit_cells = np.unique(cell[decisive])
        wd = np.where(decisive, wd * float(crit), wd)
        n_dt_eff = n_dt + (float(crit) - 1.0) * crit_cells.size
    # 鍵ごとの行の重み(層に関係なく決まる)
    Wm, wt, scale_dt, n_cells = [], [], [], []
    for e in range(nE):
        W = tgt & (b_i == key_bin[e])
        Ut = tgt & ~W
        if not W.any():
            raise GneError("BAD_CONTRAST", f"書き込みたい位置がありません(区間 {bin_names[key_bin[e]]})", "返事が空か、区間に入る位置がありません")
        if not (Ut.any() or dt.any()):
            raise GneError("BAD_CONTRAST", f"書き込みたくない位置がありません(区間 {bin_names[key_bin[e]]})")
        w = np.zeros(N)
        w[W] = om[W] / om[W].sum()
        if weights == "cells":
            wu, n_t = _cell_weights(om, cell, Ut)
            M = n_dt_eff + n_t
            w[Ut] = wu[Ut] / M
            sd = 1.0 / M
        else:
            tot = om[Ut].sum() + om[dt].sum()
            w[Ut] = om[Ut] / tot
            M, sd = 1, 1.0 / tot
        Wm.append(W)
        wt.append(w)                           # dt の行の重みは wd × scale_dt
        scale_dt.append(sd)
        n_cells.append(M)
    perms = []
    if pseudo:
        from .keyed_build import balanced_split
        for e in range(nE):
            inside = tgt & (b_i == key_bin[e])
            na, nb = np.unique(q[inside]).size, np.unique(q[tgt & ~inside]).size
            rng = np.random.default_rng(int(seed))
            perms.append([balanced_split(rng, na, nb) for _ in range(int(pseudo))] if nb else [])
    perms = perms if any(perms) else None
    tq = np.flatnonzero(q_group == 0)
    cv = c is None
    cs = [float(x) for x in cs] if cv else [float(c)]
    n_f = min(int(folds), int(tq.size)) if cv else 0
    if cv and n_f < 2:
        raise GneError("BAD_CONTRAST", "狙う場面の質問が 1 問だけなので、正則化の強さを分割で決められません",
                       "質問を増やすか、正則化の強さを直接渡してください(key_c。例 1)")
    fold = _folds(q_group * nF + q_form if nF > 1 else q_group, n_f, seed)[q] if cv else None
    lo_thr, hi_thr = float(thresholds[0]), float(thresholds[1])
    solve = _solver()
    tick = tick or (lambda msg: None)
    idx_t, idx_d = np.flatnonzero(tgt), np.flatnonzero(dt)
    E_w = np.zeros((len(Ls), nE, len(cs)))
    E_u = np.zeros((len(Ls), nE, len(cs)))
    KAP = [{} for _ in range(nE)]              # 鍵 → 層 → (c の数, 次元)
    MU = [{} for _ in range(nE)]
    TAU = [{} for _ in range(nE)]
    keep_x = bool(perms) and cv                # 比較用の鍵は c を決めてから作るので、行を取っておく(層ごとに float16 のまま)
    held: dict[int, np.ndarray] = {}
    OOF: dict[tuple[int, int], np.ndarray] = {}   # (層の番号, 鍵) → 外した組での反応(行, c の数)。セルごとの表と縮め方に使う

    def split(X):
        if X.shape[0] != N:
            raise GneError("BAD_ARGS", f"行の数 {X.shape[0]} が、行の表 {N} と違います")
        return np.asarray(X[idx_t], dtype=np.float64), np.asarray(X[idx_d], dtype=np.float64)

    wd_d = wd[idx_d]
    for li, L in enumerate(Ls):
        X = X_of(L)
        Xt, Xd = split(X)
        if keep_x:
            held[L] = X
        del X
        if cv:
            fd, ft = fold[idx_d], fold[idx_t]
            Hd = [_gram(Xd[fd == f], wd_d[fd == f]) for f in range(n_f)]
            Gd = np.sum(Hd, axis=0)
        else:
            Gd = _gram(Xd, wd_d)
        for e in range(nE):
            w_t = wt[e][idx_t]
            Wt = Wm[e][idx_t]
            sd = scale_dt[e]
            if cv:
                Ht = [_gram(Xt[ft == f], w_t[ft == f]) for f in range(n_f)]
                G = Gd * sd + np.sum(Ht, axis=0)
                for f in range(n_f):
                    tr = Wt & (ft != f)
                    if not tr.any():           # 残りの組に書き込みたい位置が無い(質問がごく少ない)
                        continue
                    bb = (w_t[tr][:, None] * Xt[tr]).sum(0)
                    kap, _ = _solve_cs(G - (Hd[f] * sd + Ht[f]), bb, bb / w_t[tr].sum(), cs, solve)
                    ht, hd = ft == f, fd == f
                    rt = Xt[ht] @ kap.T        # (行, c の数)
                    wv = w_t[ht][:, None]
                    wh = Wt[ht][:, None]
                    E_w[li, e] += (wv * (wh & (rt < hi_thr))).sum(0)
                    E_u[li, e] += (wv * (~wh & (rt > lo_thr))).sum(0)
                    oof = OOF.setdefault((li, e), np.full((N, len(cs)), np.nan, dtype=np.float32))
                    oof[idx_t[ht]] = rt
                    if hd.any():
                        rd = Xd[hd] @ kap.T
                        oof[idx_d[hd]] = rd
                        E_u[li, e] += ((wd_d[hd] * sd)[:, None] * (rd > lo_thr)).sum(0)
                del Ht
            else:
                G = Gd * sd + _gram(Xt, w_t)
            bb = (w_t[Wt][:, None] * Xt[Wt]).sum(0)     # Σ_W w = 1 なので b = m_W
            KAP[e][L], TAU[e][L] = _solve_cs(G, bb, bb, cs, solve)
            MU[e][L] = bb
            del G
        if perms and not cv:
            held[L] = _pseudo_layer(Xt, Gd, table, idx_t, b_i, key_bin, cell, n_dt_eff, weights, om, dt, perms, cs[0], solve)
        del Xt, Xd, Gd
        tick(f"L{L}")
    S_w, S_u = E_w.sum((0, 1)), E_u.sum((0, 1))
    S = S_w + S_u
    best = max(j for j in range(len(cs)) if S[j] <= S.min() + 1e-12) if cv else 0
    ctrl: list[dict] = [{} for _ in range(nE)]
    if perms:
        for L in Ls:
            if cv:                             # c が決まったので、取っておいた行から比較用の鍵を作る
                Xt, Xd = split(held.pop(L))
                C = _pseudo_layer(Xt, _gram(Xd, wd_d), table, idx_t, b_i, key_bin, cell, n_dt_eff, weights, om, dt, perms, cs[best], solve)
                del Xt, Xd
                tick(f"L{L}(比較用の鍵)")
            else:
                C = held.pop(L)
            for e in range(nE):
                if C[e] is not None:
                    ctrl[e][L] = C[e]
    names = names0
    from .prior import FORMS

    def cell_name(cc: int) -> tuple[str, str]:
        gi, bi = divmod(int(cc), n_bins)
        if conv_cells:
            gi, fi = divmod(gi, nF)
            return f"{names[gi]}|{FORMS[fi]}", bin_names[bi]
        return names[gi], bin_names[bi]
    rows_cell = {}
    for cc in np.unique(cell):
        gname, bname = cell_name(cc)
        rows_cell.setdefault(gname, {})[bname] = int((cell == cc).sum())

    def frac(r: np.ndarray, m: np.ndarray, want: bool) -> float:
        w = om[m]
        hit = (r[m] >= hi_thr) if want else (r[m] > lo_thr)
        return float((w * hit).sum() / w.sum()) if w.sum() > 0 else float("nan")

    def fp_rate(r: np.ndarray) -> float:                # 決め手のセルの誤検出の、セルごとの割合の平均
        vals = [frac(r, decisive & (cell == cc), False) for cc in np.unique(cell[decisive])]
        return float(np.mean(vals)) if vals else 0.0
    cv_cells: list[dict] = [{} for _ in range(nE)]
    shrink: dict[int, dict] = {}
    if cv:
        for li, L in enumerate(Ls):
            for e in range(nE):
                r = OOF.get((li, e))
                if r is None:
                    continue
                r = r[:, best].astype(np.float64)
                ok = ~np.isnan(r)
                W = Wm[e]
                tab = []
                for cc in np.unique(cell[ok]):
                    m = ok & (cell == cc)
                    gname, bname = cell_name(cc)
                    want = bool(W[m].all())
                    tab.append({"group": gname, "bin": bname, "rows": int(m.sum()), "want": want,
                                "decisive": bool(decisive[m].any()), "rate": frac(r, m, want)})
                cv_cells[e][L] = tab
                if fp_max is not None and e == 0:
                    rr = np.where(ok, r, 0.0)
                    before = fp_rate(rr)
                    sc = 1.0
                    if before > float(fp_max):
                        lo_, hi_ = 0.0, 1.0
                        for _ in range(40):
                            mid = (lo_ + hi_) / 2
                            if fp_rate(mid * rr) > float(fp_max):
                                hi_ = mid
                            else:
                                lo_ = mid
                        sc = lo_
                    KAP[e][L] = KAP[e][L] * sc
                    shrink[L] = {"scale": sc, "fp_before": before, "fp_after": fp_rate(sc * rr),
                                 "want_before": frac(rr, W & ok, True), "want_after": frac(sc * rr, W & ok, True)}
    if fp_max is not None and not decisive.any():
        raise GneError("BAD_CONTRAST", "決め手のセル(dont_touch の群の、鍵の区分の位置)がありません。key_fp_max は使えません",
                       "dont_touch の群を足してください")
    info = {"weights": weights, "c": cs[best], "cv": cv, "cs": cs if cv else None, "folds": n_f if cv else None,
            "score": [float(x) for x in S] if cv else None, "score_want": [float(x) for x in S_w] if cv else None,
            "score_unwant": [float(x) for x in S_u] if cv else None,
            "edge": bool(cv and best in (0, len(cs) - 1)), "thresholds": [lo_thr, hi_thr] if cv else None,
            "rows": int(N), "questions": int(q_group.size), "groups": names, "bins": bin_names,
            "conv_cells": bool(conv_cells), "forms": {FORMS[k]: int((q_form == k).sum()) for k in range(nF)},
            "crit": None if crit is None else float(crit), "fp_max": None if fp_max is None else float(fp_max),
            "decisive_rows": int(decisive.sum()), "cv_cells": cv_cells if cv else None, "shrink": shrink or None,
            "cells": n_cells if weights == "cells" else None, "rows_want": [int(W.sum()) for W in Wm], "rows_by_cell": rows_cell,
            "keep_all": KEEP_ALL, "sample": SAMPLE, "seed": int(seed), "seconds": time.perf_counter() - t0}
    return {"kappa": [{L: KAP[e][L][best] for L in Ls} for e in range(nE)], "mu": MU, "ctrl": ctrl,
            "tau": [{L: TAU[e][L] for L in Ls} for e in range(nE)], "info": info}


def _pseudo_layer(Xt, Gd, table, idx_t, b_i, key_bin, cell, n_dt, weights, om, dt, perms, c, solve) -> list:
    """1 つの層の比較用の鍵。鍵 e ごとに (本数, 次元) の配列(並べ替えが無ければ None)。Xt は狙う場面の行、Gd は dont_touch の行の
    正規方程式(ますごとの重みなら、ますの合計を 1 にした重みで積んだもの。位置を同じ重みで数えるなら ω で積んだもの)。"""
    q_t = np.asarray(table["q"])[idx_t]
    b_t, om_t, cell_t = b_i[idx_t], om[idx_t], cell[idx_t]
    out = []
    for e, ps in enumerate(perms):
        if not ps:
            out.append(None)
            continue
        inside = b_t == key_bin[e]
        qa = np.unique(q_t[inside])            # まとまり A: 区間の行がある質問
        qb = np.unique(q_t[~inside])           # まとまり B: ほかの位置の行がある質問
        na = qa.size
        C = np.zeros((len(ps), Xt.shape[1]))
        for j, (ia, _ib) in enumerate(ps):
            ia = np.asarray(ia, dtype=np.int64)
            W = np.zeros(q_t.size, dtype=bool)
            if (ia < na).any():
                W |= inside & np.isin(q_t, qa[ia[ia < na]])
            if (ia >= na).any():
                W |= ~inside & np.isin(q_t, qb[ia[ia >= na] - na])
            U = ~W
            if not W.any():
                continue
            w = np.zeros(q_t.size)
            w[W] = om_t[W] / om_t[W].sum()
            if weights == "cells":
                wu, n_t = _cell_weights(om_t, cell_t, U)
                M = n_dt + n_t
                w[U] = wu[U] / M
                sd = 1.0 / M
            else:
                tot = om_t[U].sum() + om[dt].sum()
                w[U] = om_t[U] / tot
                sd = 1.0 / tot
            bb = (w[W][:, None] * Xt[W]).sum(0)
            try:
                C[j] = _solve_cs(Gd * sd + _gram(Xt, w), bb, bb, [c], solve)[0][0]
            except GneError:
                pass                           # 作れなかった比較用の鍵は 0 のまま
        out.append(C)
    return out


def gflops() -> float:
    """この機械の行列の掛け算の速さ(GFLOP/秒)の、短い測定(見積もり用。結果は覚えておく)。小さい行列や 1 回目(スレッドの立ち上げ)は
    遅く出てばらつく(600 次元の 1 回目で 70〜130)ので、1 回空回ししてから 1,500 次元で 3 回測り、いちばん速い値を使う(この機械で 190〜210)。"""
    v = getattr(gflops, "_v", None)
    if v is None:
        n = 1500
        a = np.random.default_rng(0).standard_normal((n, n))
        a @ a
        best = 0.0
        for _ in range(3):
            t = time.perf_counter()
            a @ a
            best = max(best, 2.0 * n ** 3 / max(time.perf_counter() - t, 1e-6) / 1e9)
        v = gflops._v = best
    return v


def estimate_seconds(n_rows: int, n_target_rows: int, dim: int, n_layers: int, n_keys: int, *, pseudo: int = 0, cv: bool = True,
                     n_cs: int = len(CS), folds: int = FOLDS) -> float:
    """鍵の計算(CPU)の所要時間の見積もり(秒)。正規方程式を積む掛け算と、次元 × 次元の行列を解く回数から数える。"""
    D = float(dim)
    gram = lambda n: 2.0 * n * D * D
    chol = D ** 3 / 3.0 + 4.0 * D * D          # 分解 + 行列の写しと後退代入
    per = gram(n_rows - n_target_rows) + n_keys * gram(n_target_rows)
    per += n_keys * ((folds * n_cs + n_cs) if cv else 1) * chol
    if pseudo:
        per += (gram(n_rows - n_target_rows) if cv else 0.0) + n_keys * pseudo * (gram(n_target_rows) + chol)
    # 掛け算だけの測定より、実際の計算は遅い(行の取り出し・写し・分解の並列の効きにくさ)。実測に合わせて 3 倍にする
    # (2,560 次元・13 層・13,824 行・区間 3・5 分割で、比較用の鍵なし 148 秒・4 本 274 秒。掛け算の速さ 190〜210 で、この式は 3.0 倍・2.95 倍)
    return 3.0 * n_layers * per / (gflops() * 1e9)
