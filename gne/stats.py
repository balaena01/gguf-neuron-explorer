"""統計: 残差ノルム ρ_L と ニューロン活性の統計。設計書 §6.4。

コーパスを chunk_tokens ごとの独立したシーケンスに切り、probe ctx で 1 本ずつ評価する(F1)。
統計は各シーケンスの position ≥ 1 だけで取る(§5.6)。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from llama_cpp import llama_cpp as C

from .capture import Capture, pick_act_name
from .engine import Engine
from .steer import GneError
from .store import CONFIGS

DEFAULT_CORPUS = CONFIGS / "corpus" / "default.txt"
Progress = Callable[[int, int, str], None]


@dataclass
class Corpus:
    text: str
    sha: str
    path: str


def load_corpus(path: str | Path | None = None) -> Corpus:
    p = Path(path) if path else DEFAULT_CORPUS
    text = p.read_text(encoding="utf-8")
    return Corpus(text, hashlib.sha256(text.encode()).hexdigest()[:16], str(p))


def corpus_chunks(engine: Engine, corpus: Corpus, chunk_tokens: int = 512,
                  max_tokens: int | None = None) -> list[list[int]]:
    """コーパスを chunk_tokens トークンのシーケンスに切る。モデルが BOS を要求すれば各シーケンスの先頭に付ける。
    max_tokens を渡すと、言語の混ざり方が偏らないようにコーパス全体から等間隔にシーケンスを選ぶ。"""
    ids = engine.tokenize(corpus.text, add_special=False, parse_special=False)
    bos = [C.llama_vocab_bos(engine.vocab)] if C.llama_vocab_get_add_bos(engine.vocab) else []
    step = chunk_tokens - len(bos)
    chunks = [bos + ids[i:i + step] for i in range(0, len(ids), step)]
    chunks = [c for c in chunks if len(c) >= min(64, chunk_tokens)]
    if max_tokens is not None and max_tokens < sum(len(c) for c in chunks):
        n = max(1, -(-max_tokens // chunk_tokens))
        pick = np.unique(np.linspace(0, len(chunks) - 1, n).round().astype(int))
        chunks = [chunks[i] for i in pick]
    return chunks


def _default_layers(engine: Engine, layers: Sequence[int] | None, include_last_layer: bool) -> list[int]:
    if layers is None:
        layers = range(engine.n_layer)
    out = [int(L) for L in layers if 0 <= int(L) < engine.n_layer]
    if not include_last_layer:
        out = [L for L in out if L != engine.n_layer - 1]
    if not out:
        raise GneError("BAD_ARGS", "統計を取る層がありません", "最終層だけなら --include-last-layer を付けてください")
    return out


def _rows_check(name: str, x: np.ndarray, n_tok: int) -> None:
    if x.shape[0] != n_tok:
        raise RuntimeError(f"{name} の行数 {x.shape[0]} がトークン数 {n_tok} と違います。最終層の行の間引き(R5)なら "
                           "--include-last-layer を付けてください")


# --- 残差ノルム -------------------------------------------------------------------------------------

@dataclass
class ResidStats:
    layers: np.ndarray       # (K,)
    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray
    mean: np.ndarray
    n_tokens: int
    corpus_sha: str
    chunk_tokens: int
    exclude_pos0: bool = True

    def rho(self, L: int) -> float:
        hit = np.nonzero(self.layers == L)[0]
        if not hit.size:
            raise KeyError(f"層 {L} の残差ノルム統計がありません(gne stats resid --layers に含めてください)")
        return float(self.p50[hit[0]])

    def save(self, path: str | Path) -> None:
        np.savez(path, layers=self.layers, p10=self.p10, p50=self.p50, p90=self.p90, mean=self.mean,
                 n_tokens=self.n_tokens, corpus_sha=self.corpus_sha, chunk_tokens=self.chunk_tokens,
                 exclude_pos0=self.exclude_pos0)

    @classmethod
    def load(cls, path: str | Path) -> ResidStats:
        z = np.load(path)
        return cls(z["layers"], z["p10"], z["p50"], z["p90"], z["mean"], int(z["n_tokens"]),
                   str(z["corpus_sha"]), int(z["chunk_tokens"]), bool(z["exclude_pos0"]))

    def to_dict(self) -> dict:
        return {"layers": self.layers.tolist(), "p10": self.p10.tolist(), "p50": self.p50.tolist(),
                "p90": self.p90.tolist(), "mean": self.mean.tolist(), "n_tokens": self.n_tokens,
                "corpus_sha": self.corpus_sha, "chunk_tokens": self.chunk_tokens, "exclude_pos0": self.exclude_pos0}


def residual_norms(engine: Engine, corpus: Corpus, layers: Sequence[int] | None = None, *, chunk_tokens: int = 512,
                   include_last_layer: bool = False, max_tokens: int | None = None,
                   progress: Progress | None = None) -> ResidStats:
    """各層 L について position ≥ 1 の ‖l_out-L‖ を集め、p10/p50/p90・平均・トークン数を返す。"""
    Ls = _default_layers(engine, layers, include_last_layer)
    chunks = corpus_chunks(engine, corpus, chunk_tokens, max_tokens)
    names = {f"l_out-{L}": L for L in Ls}
    cap = Capture(lambda n: n in names)
    ctx = engine.probe_ctx(cap, logits_all=include_last_layer)
    engine.set_cvec(ctx, None)
    norms: dict[int, list[np.ndarray]] = {L: [] for L in Ls}
    for i, ch in enumerate(chunks):
        ctx = engine.probe_ctx(cap, logits_all=include_last_layer)   # ジョブが GPU を譲った間に作り直されていてもよいように
        engine.clear_memory(ctx)
        engine.prefill(ctx, [ch], logits_all=include_last_layer)
        out = cap.take()
        for name, L in names.items():
            x = out[name]
            _rows_check(name, x, len(ch))
            norms[L].append(np.linalg.norm(x[1:], axis=1))
        if progress:
            progress(i + 1, len(chunks), "stats resid")
    allv = [np.concatenate(norms[L]) for L in Ls]
    q = np.array([np.percentile(v, [10, 50, 90]) for v in allv], dtype=np.float32)
    return ResidStats(np.array(Ls, dtype=np.int32), q[:, 0], q[:, 1], q[:, 2],
                      np.array([v.mean() for v in allv], dtype=np.float32),
                      int(allv[0].size), corpus.sha, chunk_tokens)


# --- ニューロン活性 ---------------------------------------------------------------------------------

# |a| の分位点は log10|a| の固定ビンのヒストグラムで近似する(設計書 §6.4 の「t-digest 相当の近似で可」)。
# 1 ビン = 1/24 桁(約 10%)、ビン内は対数で線形補間するので誤差はおおむね ±5% 以内。
_LOG_LO, _LOG_HI, _NB = -6.0, 3.0, 216
_W = (_LOG_HI - _LOG_LO) / _NB


@dataclass
class ActStats:
    layer: int
    tensor_name: str
    absmax: np.ndarray       # (n_ff,)
    q50: np.ndarray          # |a| の分位点
    q99: np.ndarray
    q999: np.ndarray
    rms: np.ndarray
    mean: np.ndarray         # 符号つきの平均
    n_tokens: int
    corpus_sha: str

    def save(self, path: str | Path) -> None:
        np.savez(path, layer=self.layer, tensor_name=self.tensor_name, absmax=self.absmax, q50=self.q50,
                 q99=self.q99, q999=self.q999, rms=self.rms, mean=self.mean, n_tokens=self.n_tokens,
                 corpus_sha=self.corpus_sha)

    @classmethod
    def load(cls, path: str | Path) -> ActStats:
        z = np.load(path)
        return cls(int(z["layer"]), str(z["tensor_name"]), z["absmax"], z["q50"], z["q99"], z["q999"],
                   z["rms"], z["mean"], int(z["n_tokens"]), str(z["corpus_sha"]))


class _ActAccumulator:
    def __init__(self, n_ff: int):
        self.n_ff = n_ff
        self.absmax = np.zeros(n_ff, np.float32)
        self.sum = np.zeros(n_ff, np.float64)
        self.sumsq = np.zeros(n_ff, np.float64)
        self.hist = np.zeros(n_ff * _NB, np.int64)
        self.n = 0

    def add(self, a: np.ndarray) -> None:
        aa = np.abs(a)
        self.absmax = np.maximum(self.absmax, aa.max(axis=0))
        self.sum += a.sum(axis=0, dtype=np.float64)
        self.sumsq += np.square(a, dtype=np.float64).sum(axis=0)
        with np.errstate(divide="ignore"):
            b = ((np.log10(aa) - _LOG_LO) / _W).astype(np.int64)
        np.clip(b, 0, _NB - 1, out=b)
        b += np.arange(self.n_ff, dtype=np.int64) * _NB
        self.hist += np.bincount(b.ravel(), minlength=self.n_ff * _NB)
        self.n += a.shape[0]

    def quantile(self, q: float) -> np.ndarray:
        h = self.hist.reshape(self.n_ff, _NB)
        cum = h.cumsum(axis=1)
        target = q * self.n
        idx = np.minimum((cum < target).sum(axis=1), _NB - 1)
        prev = np.where(idx > 0, np.take_along_axis(cum, np.maximum(idx - 1, 0)[:, None], 1)[:, 0], 0)
        cnt = np.take_along_axis(h, idx[:, None], 1)[:, 0]
        frac = np.clip((target - prev) / np.maximum(cnt, 1), 0, 1)
        return (10.0 ** (_LOG_LO + (idx + frac) * _W)).astype(np.float32)

    def finish(self, L: int, name: str, sha: str) -> ActStats:
        n = max(self.n, 1)
        return ActStats(L, name, self.absmax, self.quantile(0.5), self.quantile(0.99), self.quantile(0.999),
                        np.sqrt(self.sumsq / n).astype(np.float32), (self.sum / n).astype(np.float32), self.n, sha)


def neuron_activation_stats(engine: Engine, corpus: Corpus, layers: Sequence[int] | None = None, *,
                            chunk_tokens: int = 512, include_last_layer: bool = False,
                            max_tokens: int | None = None, progress: Progress | None = None) -> dict[int, ActStats]:
    """ffn_down の入力(ffn_swiglu-L 等)を捕捉し、各ニューロンの |a| の分位点・最大値・平均・RMS をストリーミングで計算。"""
    Ls = _default_layers(engine, layers, include_last_layer)
    chunks = corpus_chunks(engine, corpus, chunk_tokens, max_tokens)
    names_all = engine.list_names(chunks[0][:16])
    act_names = {}
    for L in Ls:
        nm = pick_act_name(names_all, L)
        if nm is None:
            raise RuntimeError(f"層 {L} の ffn_down 入力テンソルが見つかりません(gne probe --list-names --layer {L})")
        act_names[nm] = L
    cap = Capture(lambda n: n in act_names)
    ctx = engine.probe_ctx(cap, logits_all=include_last_layer)
    engine.set_cvec(ctx, None)
    acc: dict[int, _ActAccumulator] = {}
    for i, ch in enumerate(chunks):
        ctx = engine.probe_ctx(cap, logits_all=include_last_layer)
        engine.clear_memory(ctx)
        engine.prefill(ctx, [ch], logits_all=include_last_layer)
        out = cap.take()
        for name, L in act_names.items():
            a = out[name]
            _rows_check(name, a, len(ch))
            acc.setdefault(L, _ActAccumulator(a.shape[1])).add(a[1:])
        if progress:
            progress(i + 1, len(chunks), "stats act")
    rev = {L: n for n, L in act_names.items()}
    return {L: acc[L].finish(L, rev[L], corpus.sha) for L in Ls}


# --- steer.py から使う統計の窓口 --------------------------------------------------------------------

class StatsView:
    """dose → ベクトルの計算に要る統計(ρ_L、absmax)を引く。無ければ何が足りないかを言って失敗する。"""

    def __init__(self, resid: ResidStats | None, act_loader: Callable[[int], ActStats | None] | None = None):
        self.resid = resid
        self._act_loader = act_loader or (lambda L: None)
        self._act: dict[int, ActStats | None] = {}

    def rho(self, L: int) -> float:
        if self.resid is None:
            raise KeyError("残差ノルム統計がありません(gne stats resid を先に実行してください)")
        return self.resid.rho(L)

    def absmax(self, L: int, n: int) -> float:
        if L not in self._act:
            self._act[L] = self._act_loader(L)
        st = self._act[L]
        if st is None:
            raise KeyError(f"層 {L} の活性統計がありません(gne stats act --layers {L} を先に実行してください)")
        return float(st.absmax[n])
