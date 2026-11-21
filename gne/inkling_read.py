"""Inkling 2(arch inkling2_loop)の中を読むだけの道具。docs/inkling2.md の「中を読む」。

グラフの名前は実行順の段に読み替えて返す(capture.py の Inkling の読み替え)。名前の多くは llama.cpp に
third_party/inkling2/0001(名前を付けるだけで、計算は変えない)を当てたときだけグラフにある。無ければ理由と直し方を書いて止める。

ここにあるもの(どれも書き換えはしない。方向を作って押す・射影する機能は、残差の定義が決まるまで止めたまま):
- Reader: 段ごとの値の捕捉。流れの部分和(l_out)、副層が実際に読む混ぜた入力(attn_mix・ffn_mix)、書き込み(attn_out・ffn_out、
  畳み込みの前の attn_wo・ffn_moe_out)、ヘッドの出力(kqv_out = ゲートの前、attn_gated = ゲートの後)、専門家の選び方と活性、
  深さ方向の注意の重み(*_mix_w)、繰り返しのメッセージ(loop_msg)とゲート(loop_gain_*・loop_forget_*)
- states(): 各段の深さ方向の注意が読む状態の並び(重みの列の順)と名前
- check(): 捕まえた値を、別の計算(流れの和・混ぜ方・ゲート・W_o・畳み込み・専門家の和・SiTU)と突き合わせる
- mix_summary(): 各段が、それまでのどの流れをどれだけ読むか(公開のひな形で平均)
- residuals(): 残差の 2 つの定義 (a) 流れの部分和、(b) 副層が読む混ぜた入力 で A − B の方向を作り、段ごとに比べる
- conv_leak(): 書き込みから方向を除いたとき、短い畳み込みの後に方向の成分がどれだけ残るか(素直な射影と、今のトークンの倍率で
  直した射影)
- experts(): 段ごとの専門家の選ばれ方と、同じ層の回どうしの重なり
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .capture import Capture
from .steer import GneError

# 0001 を当てた llama.cpp にだけある名前
PATCH_NAMES = ("attn_mix", "ffn_mix", "attn_gated", "attn_wo", "ffn_norm", "embd_norm", "out_mix", "loop_msg")
PATCH_WHY = ("この llama.cpp のグラフには、Inkling 2 の中を読むのに要る名前({names})がありません。")
PATCH_FIX = ("gne の third_party/inkling2/0001-inkling2-graph-names-for-gne.patch を当ててビルドした llama.cpp を使ってください"
             "(名前を付けるだけで、計算の結果は変わりません。docs/inkling2.md)")
CTRL = ("q", "k", "v", "ffn")
EPS_MIX = 1e-6                        # mix() の rms_norm の eps(src/models/inkling2.cpp)


def need_inkling(session):
    plan = getattr(session.weights, "inkling", None)
    if plan is None:
        raise GneError("ARCH_UNSUPPORTED", f"gne inkling は Inkling 2(arch inkling2_loop)専用です(このモデルは {session.meta.arch})")
    return plan


def want(bases, steps=None) -> Callable[[str], bool]:
    """捕捉する名前: 段つきの名前は base が bases にあり(steps を渡せば段も合うもの)、段の無い名前は bases にあるもの。"""
    bases = set(bases)
    steps = None if steps is None else set(int(s) for s in steps)

    def f(name: str) -> bool:
        b, sep, n = name.rpartition("-")
        if sep and n.isdigit():
            return b in bases and (steps is None or int(n) in steps)
        return name in bases
    return f


def load_contrast(path: str | Path) -> tuple[list, list, str]:
    """公開のひな形の 2 つの形: prompts_a / prompts_b、または user_prompts と system_a(・system_b)。→ (A の会話, B の会話, 名前)"""
    import yaml
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8")) if p.suffix == ".json" else yaml.safe_load(p.read_text(encoding="utf-8"))
    conv = lambda q, sys_=None: ([{"role": "system", "content": sys_}] if sys_ else []) + [{"role": "user", "content": q}]  # noqa: E731
    if d.get("prompts_a") and d.get("prompts_b"):
        return [conv(q) for q in d["prompts_a"]], [conv(q) for q in d["prompts_b"]], d.get("name") or p.stem
    if d.get("user_prompts") and d.get("system_a"):
        qs = d["user_prompts"]
        return [conv(q, d["system_a"]) for q in qs], [conv(q, d.get("system_b")) for q in qs], d.get("name") or p.stem
    raise GneError("BAD_ARGS", f"{p}: prompts_a と prompts_b、または user_prompts と system_a のある対照集合を渡してください")


def topk(cap: dict, s: int, T: int, k: int) -> np.ndarray:
    """段 s の選んだ専門家 (T, k)。ffn_moe_topk は argsort の view(連続していない)で捕まえられないことがあるので、そのときは
    ffn_moe_argsort(大きい順の並び)の先頭 k 個。"""
    a = cap.get(f"ffn_moe_topk-{s}")
    if a is not None:
        return a.reshape(T, -1)[:, :k]
    return cap[f"ffn_moe_argsort-{s}"].reshape(T, -1)[:, :k]


def rmsnorm(x: np.ndarray, eps: float = EPS_MIX) -> np.ndarray:
    x = x.astype(np.float64)
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)


def rel(a: np.ndarray, b: np.ndarray) -> float:
    """相対誤差 ‖a − b‖ / ‖b‖(全体)。"""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


class Reader:
    """Inkling 2 の段ごとの値を捕まえる(1 本ずつ prefill。名前は実行順の段)。"""

    def __init__(self, session, *, progress=None):
        self.plan = need_inkling(session)
        self.s, self.w, self.e = session, session.weights, session.engine
        self.progress = progress
        self.cap = Capture()
        self.cap.inkling_ok = True            # engine.py: Inkling で捕捉を通すのは、この読むだけの道具だけ
        self.need = 256
        self._patch_ok: bool | None = None
        self.steps = self.plan.steps()

    def ids(self, conv) -> list[int]:
        if isinstance(conv, str):
            conv = [{"role": "user", "content": conv}]
        return self.e.tokenize_prompt(self.e.render_chat(conv))

    def fit(self, seqs: list[list[int]]) -> None:
        """読ませる最長の列に合わせて、probe ctx を 1 回だけ作る(列ごとに作り直さない)。"""
        self.need = max([self.need] + [len(x) for x in seqs])

    def run(self, ids: list[int], wanted: Callable[[str], bool]) -> dict[str, np.ndarray]:
        self.cap.set_wanted(wanted)
        ctx = self.e.probe_ctx(self.cap, need=max(self.need, len(ids)))
        self.cap.reset()
        self.e.clear_memory(ctx)
        self.e.prefill(ctx, [ids])
        out = self.cap.take()
        if self._patch_ok is None:
            self._patch_ok = self.patch_present()
        return out

    def patch_present(self) -> bool:
        """0001 を当てた llama.cpp か(名前の一覧で調べて覚える)。"""
        if self._patch_ok is None:
            ids = self.ids("Hello there.")[:16]
            prev = self.e.allow_names
            self.e.allow_names = True
            try:
                names = self.e.list_names(ids)
            finally:
                self.e.allow_names = prev
                if self.e._probe is not None and self.e._probe.capture is not self.cap:
                    self.e._probe.free()
                    self.e._probe = None
            self._patch_ok = all(any(n == b or n.startswith(b + "-") for n in names) for b in PATCH_NAMES)
        return self._patch_ok

    def require_patch(self) -> None:
        if not self.patch_present():
            raise GneError("ARCH_UNSUPPORTED", PATCH_WHY.format(names=", ".join(PATCH_NAMES)), PATCH_FIX)

    # --- 深さ方向の注意が読む状態 ----------------------------------------------------------------------------------

    def states(self) -> dict:
        """各段で深さ方向の注意が読む状態の並び(重みの列の順: 取っておいた流れを新しい順に、最後が今の状態)。
        要素は (表示の名前, その状態の値を捕捉の名前で表したもの)。attn は attention の入力(流れが始まる段では、前の流れを
        取っておく前)、ffn は MLP の入力、msg は 2 回目以降の各段の繰り返しのメッセージ、out は最後の出力。"""
        plan = self.plan

        def stream_label(k: int, p: int) -> str:
            if k == 0:
                return "流れ 0"
            if p:
                return f"流れ {k}({p} 回目)"
            return f"流れ {k}"
        emb = ("埋め込み", "embd_norm")
        attn, ffn, msg = {}, {}, {}
        blocks: list[tuple[str, str]] = []
        cur: tuple[str, str] = emb
        prev_p: int | None = None
        prelude: list[tuple[str, str]] = []
        pass_end: dict[int, tuple[list, tuple]] = {}
        for st in self.steps:
            s, L, p, k = st["step"], st["layer"], st["pass"], st["stream"]
            if p == 1 and prev_p == 0:                     # 繰り返しの入口: ここまでに取っておいた流れ(埋め込み)を覚える
                prelude = list(blocks)
            if prev_p is not None and prev_p >= 1 and p != prev_p:
                pass_end[prev_p] = (list(blocks), cur)     # 回の終わり
                if p >= 2:                                 # 次の回は、層 core_start − 1 の出力からやり直す
                    blocks = list(prelude)
                    cur = (stream_label(0, 0), f"l_out-{plan.core_start - 1}")
            if L > 0:
                attn[s] = [*reversed(blocks), cur]
            if st["new_stream"]:
                blocks.append(cur)
            ffn[s] = [*reversed(blocks), (f"今の流れ({stream_label(k, p)})", f"ffn_inp-{s}")]
            cur = (stream_label(k, p), f"l_out-{s}")
            prev_p = p
        out = [*reversed(blocks), cur]
        for st in self.steps:                              # 2 回目以降: 前の回の、最初の流れの後に取っておいた流れと、最後の状態
            if st["pass"] >= 2:
                bl, last = pass_end[st["pass"] - 1]
                msg[st["step"]] = [*reversed(bl[len(prelude) + 1:]), last]
        return {"attn": attn, "ffn": ffn, "msg": msg, "out": out}


# --- 突き合わせ ------------------------------------------------------------------------------------------------------

def _conv_post(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """短い畳み込み(src/models/inkling2.cpp の convolve、動的なタップなし)を、状態 0 から 1 本の列に掛ける。
    x: (T, C)、w: (C, n_conv)(最後の列が今のトークン)。out_t = (x_t + Σ_k w[:, k] x_{t − (n − 1) + k}) / √(n + 1)。"""
    T, C = x.shape
    n = w.shape[1]
    xp = np.concatenate([np.zeros((n - 1, C), np.float64), x.astype(np.float64)], axis=0)
    out = x.astype(np.float64).copy()
    for k in range(n):
        out += xp[k:k + T] * w[:, k]
    return out / math.sqrt(n + 1)


class Weights2:
    """突き合わせに使う重みを、GGUF から直接読む(weights.py は Inkling の重みの読み出しを止めているので、ここだけで使う)。"""

    def __init__(self, weights):
        self.t = weights._tensors
        self.cache: dict[str, np.ndarray] = {}

    def full(self, name: str) -> np.ndarray:
        from .weights import tensor_to_f32
        if name not in self.cache:
            self.cache[name] = np.asarray(tensor_to_f32(self.t[name]), np.float32)
        return self.cache[name]

    def slab(self, name: str, e: int) -> np.ndarray:
        from .weights import _slab_to_f32
        return np.asarray(_slab_to_f32(self.t[name], int(e)), np.float32)

    def scale(self, name: str) -> np.ndarray | None:
        return self.full(name).reshape(-1) if name in self.t else None


def check(reader: Reader, conv, *, steps=None, n_tok_expert: int = 2) -> dict:
    """捕まえた値を別の計算と突き合わせ、相対誤差(‖捕捉 − 計算‖ / ‖捕捉‖)を返す。conv は 1 つの会話(公開のひな形の質問)。"""
    reader.require_patch()
    plan, w2 = reader.plan, Weights2(reader.w)
    ids = reader.ids(conv)
    reader.fit([ids])
    steps = list(range(plan.n_steps)) if steps is None else [int(s) for s in steps]
    bases = {"l_out", "ffn_inp", "attn_out", "ffn_out", "attn_mix", "attn_mix_w", "ffn_mix", "ffn_mix_w", "attn_gated",
             "attn_wo", "ffn_moe_out", "ffn_moe_topk", "ffn_moe_argsort", "ffn_moe_weights_scaled", "ffn_moe_situ", "ffn_moe_gate_up_scaled",
             "ffn_moe_gate_up", "ffn_norm", "loop_msg", "loop_msg_w", "attn_in_q", "attn_in_k", "attn_in_v", "ffn_in",
             "embd_norm", "out_mix", "out_mix_w"} | {f"loop_{g}_{c}" for g in ("gain", "forget") for c in CTRL}
    cap = reader.run(ids, want(bases))
    T = len(ids)
    stt = reader.states()
    errs: dict[str, list[float]] = {}

    def add(k, v):
        errs.setdefault(k, []).append(float(v))

    def val(name: str) -> np.ndarray:
        return cap[name].reshape(T, -1).astype(np.float64)
    for st in plan.steps():
        s, L = st["step"], st["layer"]
        if s not in steps:
            continue
        add("流れの和: l_out = ffn_inp + ffn_out", rel(val(f"l_out-{s}"), val(f"ffn_inp-{s}") + val(f"ffn_out-{s}")))
        if not st["new_stream"]:
            add("流れの和: ffn_inp = 前の段の l_out + attn_out", rel(val(f"ffn_inp-{s}"), val(f"l_out-{s - 1}") + val(f"attn_out-{s}")))
        # 混ぜ方: 状態を RMS 正規化して、捕まえた重みで足す
        for side, mname in (("ffn", "ffn_mix"), ("attn", "attn_mix")):
            sts = stt[side].get(s)
            if not sts:
                continue
            X = np.stack([rmsnorm(val(n)) for _, n in sts], axis=1)                      # (T, n_states, d)
            wts = cap[f"{mname}_w-{s}"].reshape(T, -1, len(sts)).astype(np.float64)     # (T, n_q, n_states)
            got = cap[f"{mname}-{s}"].reshape(T, wts.shape[1], -1)
            add(f"混ぜ方: {mname} = Σ 重み × RMS 正規化した状態", rel(got, np.einsum("tqs,tsd->tqd", wts, X)))
            add(f"混ぜ方: {mname}_w の和 = 1", float(np.abs(wts.sum(-1) - 1).max()))
        if st["pass"] >= 2:
            sts = stt["msg"][s]
            X = np.stack([rmsnorm(val(n)) for _, n in sts], axis=1)
            wts = cap[f"loop_msg_w-{s}"].reshape(T, len(CTRL), len(sts)).astype(np.float64)
            msgs = cap[f"loop_msg-{s}"].reshape(T, len(CTRL), -1).astype(np.float64)
            add("繰り返し: loop_msg = Σ 重み × RMS 正規化した前の回の状態", rel(msgs, np.einsum("tqs,tsd->tqd", wts, X)))
            mix = cap[f"attn_mix-{s}"].reshape(T, -1, msgs.shape[-1]).astype(np.float64)
            for j, c in enumerate(CTRL[:3]):
                x = mix[:, 0 if mix.shape[1] == 1 else j]
                g, f = val(f"loop_gain_{c}-{s}"), val(f"loop_forget_{c}-{s}")
                add("繰り返し: 入力 = (1 − forget)·混ぜた入力 + gain·メッセージ", rel(val(f"attn_in_{c}-{s}"), x - f * x + g * msgs[:, j]))
            x = val(f"ffn_mix-{s}")
            g, f = val(f"loop_gain_ffn-{s}"), val(f"loop_forget_ffn-{s}")
            add("繰り返し: 入力 = (1 − forget)·混ぜた入力 + gain·メッセージ", rel(val(f"ffn_in-{s}"), x - f * x + g * msgs[:, 3]))
        # attention の書き込み: W_o × ゲートの後のヘッドの出力 → 短い畳み込み
        Wo = w2.full(f"blk.{L}.attn_output.weight")
        add("attention: attn_wo = W_o × attn_gated", rel(val(f"attn_wo-{s}"), val(f"attn_gated-{s}") @ Wo.T.astype(np.float64)))
        cw = w2.full(f"blk.{L}.attn_post_conv.weight")
        add("attention: attn_out = 短い畳み込み(attn_wo)", rel(val(f"attn_out-{s}"), _conv_post(val(f"attn_wo-{s}"), cw)))
        cw = w2.full(f"blk.{L}.ffn_post_conv.weight")
        add("MLP: ffn_out = 短い畳み込み(ffn_moe_out)", rel(val(f"ffn_out-{s}"), _conv_post(val(f"ffn_moe_out-{s}"), cw)))
        # 専門家: 選んだ 8 個の (倍率 × W_down × SiTU の活性) を重みで足す(最後の n_tok_expert トークン)
        wts = cap[f"ffn_moe_weights_scaled-{s}"].reshape(T, -1).astype(np.float64)
        top = topk(cap, s, T, wts.shape[1])
        situ = cap[f"ffn_moe_situ-{s}"].reshape(T, top.shape[1], -1).astype(np.float64)
        sd = w2.scale(f"blk.{L}.ffn_down_exps.scale")
        sg = w2.scale(f"blk.{L}.ffn_gate_up_exps.scale")
        x = val(f"ffn_norm-{s}")
        gu_name = f"ffn_moe_gate_up_scaled-{s}" if f"ffn_moe_gate_up_scaled-{s}" in cap else f"ffn_moe_gate_up-{s}"
        gu = cap[gu_name].reshape(T, top.shape[1], -1).astype(np.float64)
        for t in range(T - n_tok_expert, T):
            acc = np.zeros(reader.w.meta.n_embd)
            for j, e in enumerate(top[t]):
                Wd = w2.slab(f"blk.{L}.ffn_down_exps.weight", e).astype(np.float64)
                acc += wts[t, j] * (sd[e] if sd is not None else 1.0) * (Wd @ situ[t, j])
            add("専門家: ffn_moe_out = Σ 重み × 倍率 × W_down × 活性", rel(val(f"ffn_moe_out-{s}")[t], acc))
            e = int(top[t, 0])
            Wg = w2.slab(f"blk.{L}.ffn_gate_up_exps.weight", e).astype(np.float64)
            pred = (sg[e] if sg is not None else 1.0) * (Wg @ x[t])
            add("専門家: ゲート・アップ = 倍率 × W_gate_up × 正規化した入力", rel(gu[t, 0], pred))
            n = pred.size // 2
            b, lb = 4.0, 25.0                                                   # situ_beta, situ_linear_beta(GGUF)
            g_, u_ = gu[t, 0, :n], gu[t, 0, n:]
            act = b * np.tanh(g_ / b) / (1 + np.exp(-g_)) * lb * np.tanh(u_ / lb)
            add("専門家: 活性 = SiTU(ゲート, アップ)", rel(situ[t, 0], act))
    # 最後の出力の混ぜ方
    sts = stt["out"]
    X = np.stack([rmsnorm(val(n)) for _, n in sts], axis=1)
    wts = cap["out_mix_w"].reshape(T, len(sts)).astype(np.float64)
    add("混ぜ方: out_mix = Σ 重み × RMS 正規化した状態", rel(cap["out_mix"].reshape(T, -1), np.einsum("ts,tsd->td", wts, X)))
    return {"n_tokens": T, "steps": steps, "errors": {k: {"n": len(v), "median": float(np.median(v)), "max": float(np.max(v))}
                                                      for k, v in errs.items()}}


# --- 深さ方向の注意の重み -----------------------------------------------------------------------------------------------

def mix_summary(reader: Reader, convs: list, *, positions: str = "all") -> dict:
    """各段の深さ方向の注意の重み(attention の入力・MLP の入力・繰り返しのメッセージ・最後の出力)を、会話の全部のトークン
    (positions="all")か最後のトークン("last")で平均する。重みの列は states() の並び。"""
    reader.require_patch()
    plan = reader.plan
    stt = reader.states()
    seqs = [reader.ids(c) for c in convs]
    reader.fit(seqs)
    acc: dict[str, list[np.ndarray]] = {}
    for i, ids in enumerate(seqs):
        if reader.progress:
            reader.progress(i, len(seqs), "inkling mix")
        cap = reader.run(ids, want({"attn_mix_w", "ffn_mix_w", "loop_msg_w", "out_mix_w"}))
        T = len(ids)
        sel = slice(None) if positions == "all" else slice(T - 1, T)
        for name, a in cap.items():
            base, _, n = name.rpartition("-")
            if base == "attn_mix_w":
                ns = len(stt["attn"][int(n)])
            elif base == "ffn_mix_w":
                ns = len(stt["ffn"][int(n)])
            elif base == "loop_msg_w":
                ns = len(stt["msg"][int(n)])
            else:
                ns = len(stt["out"])
            acc.setdefault(name, []).append(a.reshape(T, -1, ns)[sel].reshape(-1, a.reshape(T, -1, ns).shape[1], ns))
    mean = {k: np.concatenate(v).mean(axis=0) for k, v in acc.items()}               # (n_q, n_states)
    rows = []
    for st in plan.steps():
        s = st["step"]
        r = {**st}
        for side, base, qn in (("attn", "attn_mix_w", ("Q", "K", "V")), ("ffn", "ffn_mix_w", ("",)), ("msg", "loop_msg_w", CTRL)):
            m = mean.get(f"{base}-{s}")
            if m is None:
                continue
            labels = [lab for lab, _ in stt[side][s]]
            r[side] = {"labels": labels, "weights": m.tolist(), "queries": list(qn[:m.shape[0]]) if m.shape[0] > 1 else []}
        rows.append(r)
    out = {"labels": [lab for lab, _ in stt["out"]], "weights": mean["out_mix_w"][0].tolist()} if "out_mix_w" in mean else None
    return {"positions": positions, "n_convs": len(convs), "steps": rows, "out": out}


# --- 残差の 2 つの定義の比べ ---------------------------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(a), _unit(b)))


def _split_d(XA: np.ndarray, XB: np.ndarray) -> float:
    """分けた半分で作った方向で、残りの半分の A と B を測ったときの差 ÷ まとめた標準偏差(偶数番目で作って奇数番目で測るのと、
    その逆の平均)。"""
    out = []
    for a, b in ((0, 1), (1, 0)):
        d = _unit(XA[a::2].mean(0) - XB[a::2].mean(0))
        pa, pb = XA[b::2] @ d, XB[b::2] @ d
        sd = math.sqrt((pa.var(ddof=1) + pb.var(ddof=1)) / 2) if len(pa) > 1 and len(pb) > 1 else float("nan")
        out.append((pa.mean() - pb.mean()) / sd if sd > 0 else float("nan"))
    return float(np.mean(out))


REGIONS = ("before", "loop", "after")


def region(plan, step: int) -> str:
    """段の区分: before = 繰り返しの前(層 0〜core_start−1)、loop = 繰り返しの中、after = 繰り返しの後。"""
    st = plan.step(step)
    return "loop" if st["pass"] else ("before" if st["layer"] < plan.core_start else "after")


def study(reader: Reader, A: list, B: list, *, head_tokens: int = 4) -> dict:
    """読むだけの比べを 1 回の流しでまとめて測る(対照集合の A 側・B 側、最後のトークン。重みの平均は全トークンも)。
    1. 残差の定義ごとの A − B の方向: (a) 流れの部分和 l_out、副層が読む混ぜた入力 ffn_mix・attn_mix(全体を見る層は Q・K・V 別)、
       2 回目以降は繰り返しのゲートの後の ffn_in・attn_in_q/k/v。段ごとの向きの近さ(cos)と、半分ずつに分けて測った分かれ方
    2. 同じ層の回どうしの向きの近さ
    3. 3 回目の方向(層ごと)に沿った、各回のヘッドごと・副層ごとの書き込みの A − B(向きが 3 回目と逆のものを数える)
    4. 深さ方向の注意の重みを A 側と B 側で別に平均し、その違い
    5. 繰り返しのメッセージの強さ: ‖gain ⊙ メッセージ‖ ÷ ‖混ぜた入力‖、|forget| の平均(2 回目以降)"""
    reader.require_patch()
    plan, w2 = reader.plan, Weights2(reader.w)
    d_model = reader.w.meta.n_embd
    n_head = reader.w.arch.layers[0].n_head
    hd = reader.w.arch.layers[0].head_dim
    stt = reader.states()
    seqs = [(c, "A") for c in A] + [(c, "B") for c in B]
    ids_all = [reader.ids(c) for c, _ in seqs]
    reader.fit(ids_all)
    loop_steps = [st["step"] for st in plan.steps() if st["pass"]]
    bases = {"l_out", "ffn_mix", "ffn_in", "attn_mix", "attn_in_q", "attn_in_k", "attn_in_v", "attn_mix_w", "ffn_mix_w",
             "loop_msg_w", "out_mix_w", "loop_msg", "attn_gated", "attn_out", "ffn_out"} | \
        {f"loop_{g}_{c}" for g in ("gain", "forget") for c in CTRL}
    variants = ("l_out", "ffn_mix", "ffn_in", "attn_mix_q", "attn_mix_k", "attn_mix_v", "attn_in_q", "attn_in_k", "attn_in_v")
    V = {(v, sd): [] for v in variants for sd in "AB"}
    W_last = {sd: {} for sd in "AB"}                     # 名前 → [(n_q, n_states)]
    W_all = {sd: {} for sd in "AB"}
    loopx: dict[tuple[int, str], list] = {}
    gated = {sd: [] for sd in "AB"}                      # (繰り返しの段, head_tokens, n_head·hd)
    outs = {sd: [] for sd in "AB"}                       # (段, 2, d): attn_out・ffn_out の最後のトークン
    for i, ((_, side), ids) in enumerate(zip(seqs, ids_all, strict=True)):
        if reader.progress:
            reader.progress(i, len(seqs), "inkling residuals")
        cap = reader.run(ids, want(bases))
        T = len(ids)
        m = {v: np.zeros((plan.n_steps, d_model), np.float32) for v in variants}
        for s in range(plan.n_steps):
            m["l_out"][s] = cap[f"l_out-{s}"].reshape(T, -1)[-1]
            m["ffn_mix"][s] = cap[f"ffn_mix-{s}"].reshape(T, -1)[-1]
            if f"ffn_in-{s}" in cap:
                m["ffn_in"][s] = cap[f"ffn_in-{s}"].reshape(T, -1)[-1]
            if f"attn_mix-{s}" in cap:
                am = cap[f"attn_mix-{s}"].reshape(T, -1, d_model)[-1]
                for j, q in enumerate("qkv"[:am.shape[0]]):
                    m[f"attn_mix_{q}"][s] = am[j]
            for q in "qkv":
                if f"attn_in_{q}-{s}" in cap:
                    m[f"attn_in_{q}"][s] = cap[f"attn_in_{q}-{s}"].reshape(T, -1)[-1]
        for v in variants:
            V[(v, side)].append(m[v])
        for name, a in cap.items():
            base, _, n = name.rpartition("-")
            if base in ("attn_mix_w", "ffn_mix_w", "loop_msg_w") or name == "out_mix_w":
                key = name
                ns = len(stt["out"]) if name == "out_mix_w" else len(stt[{"attn_mix_w": "attn", "ffn_mix_w": "ffn",
                                                                          "loop_msg_w": "msg"}[base]][int(n)])
                x = a.reshape(T, -1, ns).astype(np.float64)
                W_last[side].setdefault(key, []).append(x[-1])
                W_all[side].setdefault(key, []).append(x.mean(axis=0))
        for s in loop_steps:
            if plan.step(s)["pass"] < 2:
                continue
            msg = cap[f"loop_msg-{s}"].reshape(T, len(CTRL), -1).astype(np.float64)
            am = cap[f"attn_mix-{s}"].reshape(T, -1, d_model).astype(np.float64)
            for j, c in enumerate(CTRL):
                x = cap[f"ffn_mix-{s}"].reshape(T, -1).astype(np.float64) if c == "ffn" else am[:, 0 if am.shape[1] == 1 else j]
                g = cap[f"loop_gain_{c}-{s}"].reshape(T, -1).astype(np.float64)
                f = cap[f"loop_forget_{c}-{s}"].reshape(T, -1).astype(np.float64)
                nx = np.linalg.norm(x, axis=1)
                loopx.setdefault((s, c), []).append(np.stack([np.linalg.norm(g * msg[:, j], axis=1) / nx,
                                                              np.linalg.norm(f * x, axis=1) / nx,
                                                              np.abs(g).mean(axis=1), np.abs(f).mean(axis=1)], axis=1))
        gated[side].append(np.stack([cap[f"attn_gated-{s}"].reshape(T, -1)[-head_tokens:] for s in loop_steps]))
        outs[side].append(np.stack([np.stack([cap[f"attn_out-{s}"].reshape(T, -1)[-1], cap[f"ffn_out-{s}"].reshape(T, -1)[-1]])
                                    for s in range(plan.n_steps)]))
    X = {k: np.stack(v) for k, v in V.items()}                                     # (n, steps, d)
    D = {v: X[(v, "A")].mean(0) - X[(v, "B")].mean(0) for v in variants}            # (steps, d)
    has = lambda v, s: bool(np.any(X[(v, "A")][:, s]))  # noqa: E731
    pairs = [("l_out", "ffn_mix"), ("l_out", "attn_mix_q"), ("ffn_mix", "attn_mix_q"), ("attn_mix_q", "attn_mix_k"),
             ("attn_mix_q", "attn_mix_v"), ("ffn_mix", "ffn_in"), ("attn_mix_q", "attn_in_q"), ("l_out", "ffn_in")]
    rows = []
    for st in plan.steps():
        s = st["step"]
        r = {**st, "region": region(plan, s),
             "cos": {f"{a}|{b}": _cos(D[a][s], D[b][s]) for a, b in pairs if has(a, s) and has(b, s)},
             "sep": {v: _split_d(X[(v, "A")][:, s], X[(v, "B")][:, s]) for v in variants if has(v, s)},
             "norm": {v: float(np.linalg.norm(np.concatenate([X[(v, sd)][:, s] for sd in "AB"]), axis=-1).mean())
                      for v in ("l_out", "ffn_mix", "attn_mix_q") if has(v, s)}}
        rows.append(r)
    # 同じ層の回どうし
    passes = []
    for L in range(plan.core_start, plan.core_end):
        ss = plan.steps_of(L)
        passes.append({"layer": L, "steps": ss, **{v: [_cos(D[v][ss[i]], D[v][ss[j]]) for i, j in ((0, 1), (1, 2), (0, 2))]
                                                     for v in ("l_out", "ffn_mix", "attn_mix_q")}})
    # 3 回目の方向に沿った書き込み
    G = {sd: np.stack(v) for sd, v in gated.items()}                               # (n, loop_steps, tok, n_head·hd)
    O = {sd: np.stack(v) for sd, v in outs.items()}                                # (n, steps, 2, d)
    heads = []
    sum_err = []
    for L in range(plan.core_start, plan.core_end):
        ss = plan.steps_of(L)
        Wo = w2.full(f"blk.{L}.attn_output.weight").astype(np.float64).reshape(d_model, n_head, hd)
        cw = w2.full(f"blk.{L}.attn_post_conv.weight").astype(np.float64)
        nconv = cw.shape[1]
        for dname, dv in (("a", "l_out"), ("b", "ffn_mix")):
            r3 = _unit(D[dv][ss[-1]].astype(np.float64))
            per_pass = []
            for pi, s in enumerate(ss):
                li = loop_steps.index(s)
                contrib = {}
                for sd in "AB":
                    g = G[sd][:, li].astype(np.float64).reshape(G[sd].shape[0], head_tokens, n_head, hd)
                    y = np.einsum("dhk,ntkh->nthd", Wo, g.transpose(0, 1, 3, 2))       # (n, tok, head, d): 各ヘッドの W_o の出力
                    # 最後のトークンの畳み込みの後: (y_T + Σ_k w_k ⊙ y_{T − (n−1) + k}) / √(n+1)(状態は列の中のトークンだけ)
                    o = y[:, -1].copy()
                    for k in range(nconv):
                        t = head_tokens - nconv + k
                        if t >= 0:
                            o += y[:, t] * cw[:, k]
                    o /= math.sqrt(nconv + 1)
                    contrib[sd] = o @ r3                                                # (n, head)
                    if dname == "a":
                        sum_err.append(rel(o.sum(axis=1), O[sd][:, s, 0]))
                c_head = contrib["A"].mean(0) - contrib["B"].mean(0)
                c_attn = float((O["A"][:, s, 0] @ r3).mean() - (O["B"][:, s, 0] @ r3).mean())
                c_ffn = float((O["A"][:, s, 1] @ r3).mean() - (O["B"][:, s, 1] @ r3).mean())
                per_pass.append({"step": s, "pass": pi + 1, "heads": c_head.tolist(), "attn": c_attn, "ffn": c_ffn})
            ref = np.array(per_pass[-1]["heads"])
            thr = 0.1 * float(np.abs(ref).max()) if np.any(ref) else 0.0
            opp = []
            for pp in per_pass[:-1]:
                h = np.array(pp["heads"])
                for k in np.flatnonzero((np.sign(h) != np.sign(ref)) & (np.abs(h) >= thr) & (np.abs(ref) >= thr)):
                    opp.append({"pass": pp["pass"], "head": int(k), "value": float(h[k]), "pass3": float(ref[k])})
            heads.append({"layer": L, "direction": dname, "passes": per_pass, "threshold": thr, "opposite": opp})
    # 深さ方向の注意の重み: A 側と B 側
    mixdiff = []
    for st in plan.steps():
        s = st["step"]
        r = {"step": s, "region": region(plan, s), "pass": st["pass"]}
        for side, base in (("attn", "attn_mix_w"), ("ffn", "ffn_mix_w"), ("msg", "loop_msg_w")):
            k = f"{base}-{s}"
            if k not in W_last["A"]:
                continue
            for pos, Wd in (("last", W_last), ("all", W_all)):
                wa, wb = np.mean(Wd["A"][k], axis=0), np.mean(Wd["B"][k], axis=0)            # (n_q, n_states)
                r[f"{side}_{pos}"] = {"A": wa.tolist(), "B": wb.tolist(), "tv": float(0.5 * np.abs(wa - wb).sum(-1).max()),
                                      "labels": [lab for lab, _ in stt[side][s]]}
        mixdiff.append(r)
    loop = []
    for (s, c), v in sorted(loopx.items()):
        a = np.concatenate(v)
        loop.append({"step": s, "layer": plan.step(s)["layer"], "pass": plan.step(s)["pass"], "ctrl": c,
                     "gain_msg_rel": float(a[:, 0].mean()), "forget_rel": float(a[:, 1].mean()),
                     "gain_abs": float(a[:, 2].mean()), "forget_abs": float(a[:, 3].mean())})
    return {"n_a": len(A), "n_b": len(B), "steps": rows, "passes": passes, "heads": heads,
            "head_sum_rel_max": float(np.max(sum_err)) if sum_err else None, "mix": mixdiff, "loop": loop,
            "directions": D}


# --- 短い畳み込みの後に残る方向の成分 --------------------------------------------------------------------------------------

def conv_leak(reader: Reader, convs: list, dirsets: dict[str, np.ndarray], *, steps=None) -> dict:
    """書き込み y(畳み込みの前: attention は attn_wo、MLP は ffn_moe_out)から方向 r(段ごとの流れの部分和の A − B、単位ベクトル)を
    除き、短い畳み込みを通した後に r の成分がどれだけ残るかを、実際の活性(会話の全トークン)で測る。dirsets は名前 → (段, d) の
    方向(例: a = 流れの部分和、b = MLP が読む混ぜた入力)。
    - 素直な射影: y' = y − (y·r) r
    - 直した射影: u = normalize(g ⊙ r)、g = (1 + 今のトークンのタップ) / √5(チャンネルごとの今のトークンの倍率)、y' = y − (y·u) u
    残る割合 = √Σ(r·out')² / √Σ(r·out)²。素直な射影の残りは、今のトークンの倍率による分と、前のトークンのタップからの漏れに分ける。
    向きのほかの部分の変化(副作用)= ‖(out' − out) − ((r·out') − (r·out)) r‖ / ‖out‖。"""
    plan, w2 = reader.plan, Weights2(reader.w)
    seqs = [reader.ids(c) for c in convs]
    reader.fit(seqs)
    steps = list(range(plan.n_steps)) if steps is None else [int(s) for s in steps]
    acc: dict[tuple[int, str, str], dict[str, float]] = {}
    for i, ids in enumerate(seqs):
        if reader.progress:
            reader.progress(i, len(seqs), "inkling conv")
        cap = reader.run(ids, want({"attn_wo", "attn_out", "ffn_moe_out", "ffn_out"}, steps))
        T = len(ids)
        for s, (dname, dirs), (kind, pre, post, wname) in ((s, dd, kk) for s in steps for dd in dirsets.items()
                                                            for kk in (("attn", "attn_wo", "attn_out", "attn_post_conv"),
                                                                       ("ffn", "ffn_moe_out", "ffn_out", "ffn_post_conv"))):
            L = plan.step(s)["layer"]
            r = _unit(dirs[s].astype(np.float64))
            y = cap[f"{pre}-{s}"].reshape(T, -1).astype(np.float64)
            out = cap[f"{post}-{s}"].reshape(T, -1).astype(np.float64)
            w = w2.full(f"blk.{L}.{wname}.weight").astype(np.float64)
            n = w.shape[1]
            g = (1.0 + w[:, -1]) / math.sqrt(n + 1)
            u = _unit(g * r)
            y1 = y - np.outer(y @ r, r)
            y2 = y - np.outer(y @ u, u)
            o0, o1, o2 = _conv_post(y, w), _conv_post(y1, w), _conv_post(y2, w)
            cur1 = (g * y1) @ r                                   # 素直な射影の残りのうち、今のトークンの倍率による分
            a = acc.setdefault((s, kind, dname), {k: 0.0 for k in ("c0", "c1", "c2", "cur1", "past1", "side1", "side2", "out", "rec", "recd")})
            c0, c1, c2 = o0 @ r, o1 @ r, o2 @ r
            a["c0"] += float((c0 ** 2).sum())
            a["c1"] += float((c1 ** 2).sum())
            a["c2"] += float((c2 ** 2).sum())
            a["cur1"] += float((cur1 ** 2).sum())
            a["past1"] += float(((c1 - cur1) ** 2).sum())
            a["side1"] += float((((o1 - o0) - np.outer(c1 - c0, r)) ** 2).sum())
            a["side2"] += float((((o2 - o0) - np.outer(c2 - c0, r)) ** 2).sum())
            a["out"] += float((o0 ** 2).sum())
            a["rec"] += float(((o0 - out) ** 2).sum())
            a["recd"] += float((out ** 2).sum())
    rows = []
    for (s, kind, dname), a in sorted(acc.items()):
        st = plan.step(s)
        rows.append({"step": s, "layer": st["layer"], "pass": st["pass"], "stream": st["stream"], "new_stream": st["new_stream"],
                     "region": region(plan, s), "kind": kind, "direction": dname,
                     "left_naive": math.sqrt(a["c1"] / a["c0"]) if a["c0"] else None,
                     "left_naive_now": math.sqrt(a["cur1"] / a["c0"]) if a["c0"] else None,
                     "left_naive_past": math.sqrt(a["past1"] / a["c0"]) if a["c0"] else None,
                     "left_corrected": math.sqrt(a["c2"] / a["c0"]) if a["c0"] else None,
                     "side_naive": math.sqrt(a["side1"] / a["out"]) if a["out"] else None,
                     "side_corrected": math.sqrt(a["side2"] / a["out"]) if a["out"] else None,
                     "recompute_rel": math.sqrt(a["rec"] / a["recd"]) if a["recd"] else None})
    return {"n_convs": len(convs), "rows": rows}


# --- 専門家の選ばれ方 -------------------------------------------------------------------------------------------------------

def experts(reader: Reader, convs: list) -> dict:
    """段ごとの専門家の選ばれ方(全トークン): 選ばれた回数の 5 割・9 割を占める専門家の数、選んだ重みの最大の平均。
    繰り返しの層は、同じトークンで回どうしに選んだ 8 個がどれだけ重なるか(共通の数 ÷ 8)。"""
    plan = reader.plan
    seqs = [reader.ids(c) for c in convs]
    reader.fit(seqs)
    counts = np.zeros((plan.n_steps, reader.w.meta.n_expert), np.int64)
    overlap: dict[tuple[int, int, int], list[float]] = {}
    wmax = np.zeros(plan.n_steps)
    n_tok = 0
    for i, ids in enumerate(seqs):
        if reader.progress:
            reader.progress(i, len(seqs), "inkling experts")
        cap = reader.run(ids, want({"ffn_moe_topk", "ffn_moe_argsort", "ffn_moe_weights_scaled"}))
        T = len(ids)
        n_tok += T
        k = int(reader.w.arch.layers[0].n_expert_used)
        top = {s: topk(cap, s, T, k) for s in range(plan.n_steps)}
        for s in range(plan.n_steps):
            np.add.at(counts[s], top[s].reshape(-1), 1)
            wmax[s] += float(cap[f"ffn_moe_weights_scaled-{s}"].reshape(T, -1).max(axis=1).sum())
        for L in range(plan.core_start, plan.core_end):
            ss = plan.steps_of(L)
            for a, b in ((0, 1), (1, 2), (0, 2)):
                if b < len(ss):
                    ov = [len(set(top[ss[a]][t]) & set(top[ss[b]][t])) / top[ss[a]].shape[1] for t in range(T)]
                    overlap.setdefault((L, a + 1, b + 1), []).extend(ov)
    rows = []
    for st in plan.steps():
        s = st["step"]
        c = np.sort(counts[s])[::-1]
        cum = np.cumsum(c) / max(c.sum(), 1)
        rows.append({**st, "n50": int(np.searchsorted(cum, 0.5) + 1), "n90": int(np.searchsorted(cum, 0.9) + 1),
                     "n_used": int((counts[s] > 0).sum()), "w_max_mean": wmax[s] / max(n_tok, 1)})
    return {"n_tokens": n_tok, "steps": rows,
            "overlap": [{"layer": L, "passes": [a, b], "mean": float(np.mean(v))} for (L, a, b), v in sorted(overlap.items())]}
