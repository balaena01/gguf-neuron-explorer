"""logit lens と steer lens。設計書 §2.5, §6.6。

logit lens(重みのみ): z = W_U · (γ ⊙ u_n) を語彙について z 化し、上位(直接押し上げる語)と下位(押し下げる語)を取る。
steer lens(順伝播): 短い文脈数本に ±dose · ρ_L · u_n を足し、最終位置の次トークン log p の変化 Δ を平均する。
  up   = + で押したときに Δ が大きい語(押し上げられる語)
  down = − で押したときに Δ が大きい語(線形の範囲では + で押し下げられる語とほぼ同じ)
  kl_pos / kl_neg = KL(base‖steered) の文脈平均。同じ大きさのランダム方向(対照)で帰無分布を作る。
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .engine import Engine, log_softmax
from .steer import random_direction

Progress = Callable[[int, int, str], None]

# 既定の文脈(英語 2・日本語 2、うち 1 本はチャットテンプレートの assistant 冒頭)
DEFAULT_CONTEXTS: list[dict] = [
    {"kind": "text", "text": "I walked into the kitchen and the first thing I noticed was"},
    {"kind": "chat", "user": "Tell me something about yourself."},
    {"kind": "text", "text": "昨日の夜、駅前の小さな店で"},
    {"kind": "text", "text": "この町でいちばん有名なものといえば"},
]


def context_ids(engine: Engine, contexts: Sequence[dict]) -> list[list[int]]:
    out = []
    for c in contexts:
        if c["kind"] == "chat":
            out.append(engine.tokenize_prompt(engine.render_chat([{"role": "user", "content": c["user"]}])))
        else:
            out.append(engine.tokenize(c["text"], add_special=True, parse_special=False))
    return out


def _topk_desc(x: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-x, k)[:k]
    return idx[np.argsort(-x[idx])]


# --- logit lens ----------------------------------------------------------------------------------------

@dataclass
class LogitLens:
    layer: int
    neurons: np.ndarray       # (K,)
    up_ids: np.ndarray        # (K, k) z の大きい順
    up_z: np.ndarray
    down_ids: np.ndarray      # (K, k) z の小さい順
    down_z: np.ndarray

    @property
    def peak_z(self) -> np.ndarray:
        return np.maximum(np.abs(self.up_z[:, 0]), np.abs(self.down_z[:, 0]))

    def save(self, path: str | Path) -> None:
        np.savez(path, layer=self.layer, neurons=self.neurons, up_ids=self.up_ids, up_z=self.up_z,
                 down_ids=self.down_ids, down_z=self.down_z)

    @classmethod
    def load(cls, path: str | Path) -> LogitLens:
        z = np.load(path)
        return cls(int(z["layer"]), z["neurons"], z["up_ids"], z["up_z"], z["down_ids"], z["down_z"])


class _TopK:
    """行(ニューロン)ごとの上位 k(値・語番号)。最初のチャンクは argpartition、以降は今の k 位を超えた値だけを混ぜる。"""

    def __init__(self, K: int, k: int):
        self.K, self.k = K, k
        self.v: np.ndarray | None = None              # (K, k)
        self.i: np.ndarray | None = None

    def add(self, Zt: np.ndarray, start: int) -> None:   # Zt: (K, r)
        k = self.k
        if self.v is None:
            idx = np.argpartition(Zt, Zt.shape[1] - k, axis=1)[:, -k:]
            self.v, self.i = np.take_along_axis(Zt, idx, axis=1), idx + start
            return
        ni, vi = np.nonzero(Zt > self.v.min(axis=1)[:, None])
        if not ni.size:
            return
        all_n = np.concatenate([np.repeat(np.arange(self.K), k), ni])
        all_v = np.concatenate([self.v.ravel(), Zt[ni, vi]])
        all_i = np.concatenate([self.i.ravel(), vi + start])
        o = np.lexsort((-all_v, all_n))                   # ニューロン順、その中で値の大きい順
        all_v, all_i = all_v[o], all_i[o]
        take = np.searchsorted(all_n[o], np.arange(self.K))[:, None] + np.arange(k)[None, :]
        self.v, self.i = all_v[take], all_i[take]

    def sorted(self) -> tuple[np.ndarray, np.ndarray]:
        o = np.argsort(-self.v, axis=1)
        return np.take_along_axis(self.v, o, 1), np.take_along_axis(self.i, o, 1)


def logit_lens_layer(weights, L: int, *, neurons: np.ndarray | None = None, top_k: int = 12, chunk: int = 4096,
                     progress: Progress | None = None) -> LogitLens:
    """全(または指定)ニューロンの logit lens。W_U を行チャンクで読み、上位/下位 top_k だけを持つ。"""
    nn = np.arange(weights.meta.n_ff_at(L)) if neurons is None else np.asarray(neurons)
    Gt = np.ascontiguousarray((weights.unit_matrix(L)[:, nn] * weights.output_norm_weight()[:, None]).T,
                              dtype=np.float32)                       # (K, n_embd)
    K = nn.size
    s1, s2 = np.zeros(K), np.zeros(K)
    top, bot = _TopK(K, top_k), _TopK(K, top_k)                     # 下位は -Z の上位として持つ
    n_vocab = 0
    total = weights.tensor(weights.output_tensor_name()).data.shape[0]
    for start, rows in weights.output_rows(chunk):
        Zt = Gt @ rows.T                                              # (K, r) 行が連続
        n_vocab += Zt.shape[1]
        s1 += Zt.sum(axis=1, dtype=np.float64)
        s2 += np.einsum("ij,ij->i", Zt, Zt, dtype=np.float64)
        top.add(Zt, start)
        np.negative(Zt, out=Zt)
        bot.add(Zt, start)
        if progress:
            progress(min(start + Zt.shape[1], total), total, f"lens logit L{L}")
    mean = s1 / n_vocab
    std = np.sqrt(np.maximum(s2 / n_vocab - mean ** 2, 1e-30))
    tv, ti = top.sorted()
    bv, bi = bot.sorted()
    return LogitLens(L, nn, ti.astype(np.int32), ((tv - mean[:, None]) / std[:, None]).astype(np.float32),
                     bi.astype(np.int32), ((-bv - mean[:, None]) / std[:, None]).astype(np.float32))


# --- steer lens ----------------------------------------------------------------------------------------

@dataclass
class SteerLens:
    layer: int
    n_ff: int
    config: dict                                   # dose, contexts, top_k, random_controls, seed, rho
    done: np.ndarray = field(default=None)         # (n_ff,) bool
    up_ids: np.ndarray = field(default=None)       # (n_ff, k)
    up_delta: np.ndarray = field(default=None)
    down_ids: np.ndarray = field(default=None)
    down_delta: np.ndarray = field(default=None)
    kl_pos: np.ndarray = field(default=None)       # (n_ff,)
    kl_neg: np.ndarray = field(default=None)
    ctrl_done: np.ndarray = field(default=None)    # (n_ctrl,)
    ctrl_seeds: np.ndarray = field(default=None)
    ctrl_kl_pos: np.ndarray = field(default=None)
    ctrl_kl_neg: np.ndarray = field(default=None)
    ctrl_up_ids: np.ndarray = field(default=None)
    ctrl_down_ids: np.ndarray = field(default=None)

    @classmethod
    def empty(cls, layer: int, n_ff: int, config: dict) -> SteerLens:
        k, c = int(config["top_k"]), int(config["random_controls"])
        return cls(layer, n_ff, config, np.zeros(n_ff, bool),
                   np.full((n_ff, k), -1, np.int32), np.full((n_ff, k), np.nan, np.float32),
                   np.full((n_ff, k), -1, np.int32), np.full((n_ff, k), np.nan, np.float32),
                   np.full(n_ff, np.nan, np.float32), np.full(n_ff, np.nan, np.float32),
                   np.zeros(c, bool), np.array([int(config["seed"]) * 100003 + j for j in range(c)], np.int64),
                   np.full(c, np.nan, np.float32), np.full(c, np.nan, np.float32),
                   np.full((c, k), -1, np.int32), np.full((c, k), -1, np.int32))

    @staticmethod
    def config_key(config: dict) -> str:
        return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        d = {k: getattr(self, k) for k in ("done", "up_ids", "up_delta", "down_ids", "down_delta", "kl_pos", "kl_neg",
                                           "ctrl_done", "ctrl_seeds", "ctrl_kl_pos", "ctrl_kl_neg", "ctrl_up_ids",
                                           "ctrl_down_ids")}
        tmp = Path(str(path) + ".tmp.npz")
        np.savez(tmp, layer=self.layer, n_ff=self.n_ff, config=json.dumps(self.config, ensure_ascii=False), **d)
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> SteerLens:
        z = np.load(path)
        obj = cls(int(z["layer"]), int(z["n_ff"]), json.loads(str(z["config"])))
        for k in z.files:
            if k not in ("layer", "n_ff", "config"):
                setattr(obj, k, z[k])
        return obj

    def null(self, sign: str = "pos") -> np.ndarray:
        v = self.ctrl_kl_pos if sign == "pos" else self.ctrl_kl_neg
        return v[self.ctrl_done]

    def null_pct(self, kl: np.ndarray, sign: str = "pos") -> np.ndarray:
        """各 KL がランダム対照の何割より大きいか(1.0 = 全対照より上)。"""
        null = np.sort(self.null(sign))
        if not null.size:
            return np.full(np.shape(kl), np.nan)
        return np.searchsorted(null, kl, side="left") / null.size


def steer_lens_layer(engine: Engine, weights, stats, L: int, *, neurons: np.ndarray | None = None, dose: float = 0.8,
                     contexts: Sequence[dict] | None = None, top_k: int = 12, random_controls: int = 64,
                     seed: int = 0, out: str | Path | None = None, resume: bool = True,
                     progress: Progress | None = None, checkpoint_s: float = 60.0) -> SteerLens:
    """層 L の指定ニューロン(と対照)の steer lens。out があれば途中経過を保存し、resume なら続きから。
    文脈は 1 バッチで評価し、ベースラインも同じバッチ構成で 1 回だけ取る(F2)。"""
    contexts = list(contexts or DEFAULT_CONTEXTS)
    n_ff = weights.meta.n_ff_at(L)
    nn = np.arange(n_ff) if neurons is None else np.asarray(neurons)
    rho = stats.rho(L)
    config = {"dose": float(dose), "contexts": contexts, "top_k": int(top_k), "random_controls": int(random_controls),
              "seed": int(seed), "rho": round(float(rho), 6)}
    res = None
    if out is not None and resume and Path(out).exists():
        old = SteerLens.load(out)
        if SteerLens.config_key(old.config) == SteerLens.config_key(config):
            res = old
    if res is None:
        res = SteerLens.empty(L, n_ff, config)

    ids = context_ids(engine, contexts)
    ctx = engine.steer_ctx()
    engine.set_cvec(ctx, None)
    engine.clear_memory(ctx)
    base = log_softmax(engine.prefill(ctx, ids).logits)            # (C, V)
    p_base = np.exp(base.astype(np.float64))
    U = weights.unit_matrix(L)

    def measure(v: np.ndarray) -> tuple[np.ndarray, float]:
        engine.set_cvec(ctx, {L: v})
        engine.clear_memory(ctx)
        lp = log_softmax(engine.prefill(ctx, ids).logits)
        kl = float((p_base * (base - lp)).sum(axis=1).mean())
        return (lp - base).mean(axis=0), kl

    todo_c = [j for j in range(random_controls) if not res.ctrl_done[j]]
    todo_n = [int(n) for n in nn if not res.done[n]]
    total, step, last_save = len(todo_c) + len(todo_n), 0, time.monotonic()
    try:
        for j in todo_c:                                            # 対照を先に(帰無分布が早く揃う)
            r = random_direction(weights.meta.n_embd, int(res.ctrl_seeds[j]), L)
            dp, res.ctrl_kl_pos[j] = measure(dose * rho * r)
            dn, res.ctrl_kl_neg[j] = measure(-dose * rho * r)
            res.ctrl_up_ids[j], res.ctrl_down_ids[j] = _topk_desc(dp, top_k), _topk_desc(dn, top_k)
            res.ctrl_done[j] = True
            step += 1
            if progress:
                progress(step, total, f"lens steer L{L}")
        for n in todo_n:
            u = U[:, n]
            dp, res.kl_pos[n] = measure(dose * rho * u)
            dn, res.kl_neg[n] = measure(-dose * rho * u)
            up, down = _topk_desc(dp, top_k), _topk_desc(dn, top_k)
            res.up_ids[n], res.up_delta[n] = up, dp[up]
            res.down_ids[n], res.down_delta[n] = down, dn[down]
            res.done[n] = True
            step += 1
            if progress:
                progress(step, total, f"lens steer L{L}")
            if out is not None and time.monotonic() - last_save > checkpoint_s:
                res.save(out)
                last_save = time.monotonic()
    finally:
        engine.set_cvec(ctx, None)
        if out is not None:
            res.save(out)
    return res
