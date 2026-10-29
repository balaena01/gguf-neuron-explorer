"""attention ヘッドの捕捉(CHANGE-04 B)。

flash attention を切った probe ctx(Engine.attn_probe_ctx)で prefill し、cb_eval で層ごとに次を回収する。
- pattern: `kq_soft_max-L`(softmax 後の注目の重み)。ggml ne = [n_kv, n_tokens, n_head, n_stream] → numpy
  (n_head, T, T)(1 シーケンス・空のキャッシュから prefill するので、キャッシュの位置 = トークンの位置)。attention の層だけ。
- head_out: 書き込み行列(W_O / ssm_out)に入る直前のテンソル(`attn_gated` / `kqv_out` / `final_output`)。
  ggml ne = [n_head·head_dim, n_tokens] → numpy (T, n_head, head_dim)。ヘッド h は列 [h·head_dim, (h+1)·head_dim)。
- qkv(任意): `Qcur-L` / `Kcur-L` / `Vcur-L`。(T, n_head(_kv), head_dim)。同じ名前が投影の直後と RoPE の後の 2 つに付く
  arch があり、捕捉は先に来た方を採るので、どちらになるかは arch で違う(Qwen3 は 3 つとも投影の直後、Qwen3.5 は Qcur が
  RoPE の後で Kcur・Vcur が投影の直後)。正規化の後・RoPE の前が要るときは `Kcur_normed-L` を extra で頼む。

1 回の decode が 1 つのグラフ(1 ubatch)に収まるよう、会話は probe_ubatch トークン以下にする(パターンの行が ubatch を
またぐと、ubatch ごとに n_kv が違って連結できない)。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .capture import Capture
from .steer import GneError


@dataclass
class AttnCapture:
    ids: list[int]
    text: str                                         # トークンのバイト列を連結して読んだ文(特殊トークンも文字として入る)
    offsets: list[tuple[int, int]]                    # トークンごとの text の文字区間 [lo, hi)
    pattern: dict[int, np.ndarray] = field(default_factory=dict)    # 層 → (n_head, T, T)
    head_out: dict[int, np.ndarray] = field(default_factory=dict)   # 層 → (T, n_head, head_dim)
    qkv: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    extra: dict[str, np.ndarray] = field(default_factory=dict)     # capture_attention(extra=...) で頼んだ名前 → (T, ne0)

    @property
    def n_tokens(self) -> int:
        return len(self.ids)


def kv_group(h: int, n_head: int, n_head_kv: int) -> int:
    """GQA: Q ヘッド h が使う K/V のグループ(llama.cpp は連続する n_head / n_head_kv 個の Q ヘッドが 1 つの K/V を共有)。"""
    if n_head_kv <= 0 or n_head % n_head_kv:
        raise GneError("ARCH_UNSUPPORTED", f"n_head {n_head} が n_head_kv {n_head_kv} で割り切れません")
    return h // (n_head // n_head_kv)


def token_offsets(engine, ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
    """トークン列 → (文, トークンごとの文字区間)。UTF-8 の 1 文字が 2 トークンにまたがるときは、その文字は始まりのトークンに入れる
    (続きのトークンは空の区間)。"""
    import ctypes

    from llama_cpp import llama_cpp as C
    pieces = []
    for t in ids:
        buf = ctypes.create_string_buffer(256)
        n = C.llama_token_to_piece(engine.vocab, t, buf, len(buf), 0, True)
        pieces.append(buf.raw[:max(n, 0)])
    raw = b"".join(pieces)
    text = raw.decode("utf-8", "replace")
    char_at = np.zeros(len(raw) + 1, dtype=np.int64)       # バイト位置 → その位置を含む文字の番号
    b = 0
    for ci, ch in enumerate(text):
        n = len(ch.encode("utf-8")) if ch != "�" else 1
        char_at[b:b + n] = ci
        b += n
    char_at[b:] = len(text)
    offs, b = [], 0
    for p in pieces:
        lo = int(char_at[b]) if p else int(char_at[min(b, len(raw))])
        b += len(p)
        hi = int(char_at[b - 1]) + 1 if p else lo
        if offs and lo < offs[-1][1]:                        # 前のトークンで始まった文字の続き
            lo = offs[-1][1]
        offs.append((lo, max(lo, hi)))
    return text, offs


def find_spans(text: str, triggers: Sequence[str]) -> list[tuple[int, int]]:
    """トリガー(部分文字列。"re:" で始まれば正規表現)の文字区間。重なりは 1 つにまとめる。"""
    spans = []
    for t in triggers:
        if t.startswith("re:"):
            spans += [m.span() for m in re.finditer(t[3:], text)]
        else:
            i = text.find(t)
            while i >= 0 and t:
                spans.append((i, i + len(t)))
                i = text.find(t, i + 1)
    spans.sort()
    out: list[tuple[int, int]] = []
    for s, e in spans:
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def spans_to_tokens(offsets: Sequence[tuple[int, int]], spans: Sequence[tuple[int, int]]) -> list[int]:
    """文字区間 → 区間と 1 文字でも重なるトークンの添字(昇順)。"""
    out = []
    for i, (lo, hi) in enumerate(offsets):
        if hi > lo and any(lo < e and s < hi for s, e in spans):
            out.append(i)
    return out


def head_out_names(session, names: set[str], layers: Sequence[int]) -> dict[int, str]:
    """層 → per-head 出力のテンソル名(arch の候補のうち捕捉で見えた最初のもの)。"""
    out = {}
    for L in layers:
        for base in session.weights.arch.layers[L].head_out:
            if f"{base}-{L}" in names:
                out[L] = f"{base}-{L}"
                break
    return out


def _probe(e) -> tuple[Capture, object]:
    """flash attention を切った probe ctx と、それに付けた Capture(使い回す)。"""
    cap = getattr(e, "_attn_capture", None)
    if cap is None:
        cap = e._attn_capture = Capture()
    cap.set_wanted(lambda nm: False)
    return cap, e.attn_probe_ctx(cap)


def _names(e, cap: Capture, ctx, ids: list[int]) -> set[str]:
    cap.seen, cap.record_names = {}, True
    try:
        e.clear_memory(ctx)
        e.set_cvec(ctx, None)
        e.prefill(ctx, [ids])
    finally:
        cap.record_names = False
    return set(cap.seen)


def capture_attention(session, convs: Sequence[list[dict] | str | list[int]], *, layers: Sequence[int] | None = None,
                      want_pattern: bool = True, want_head_out: bool = True, want_qkv: bool = False,
                      vectors: dict[int, np.ndarray] | None = None, lora=None,
                      extra: Sequence[str] = ()) -> list[AttnCapture]:
    """会話(メッセージの列)・文・トークン列ごとに AttnCapture を返す。layers は既定で全層(ヘッドを持つ層)。
    vectors / lora を渡すと、その介入を入れた状態で捕捉する(ヘッド gain の V15 用)。extra は他に回収するテンソル名。"""
    e, w = session.engine, session.weights
    ids_list = []
    for c in convs:
        if isinstance(c, str):
            ids = e.tokenize(c, add_special=True, parse_special=False)
        elif c and isinstance(c[0], int):
            ids = list(c)
        else:
            ids = e.tokenize_prompt(e.render_chat(list(c)))
        if len(ids) > e.probe_ubatch:
            raise GneError("BAD_ARGS", f"{len(ids)} トークンは attention の捕捉の上限 {e.probe_ubatch} を超えています(短くしてください)")
        if len(ids) < 2:
            raise GneError("BAD_ARGS", "文章が短すぎます(2 トークン以上)")
        ids_list.append(ids)
    arch = w.arch
    Ls = [L for L in (range(e.n_layer) if layers is None else layers) if arch.layers[L].head_dim]
    probe, ctx = _probe(e)
    names = _names(e, probe, ctx, ids_list[0][:16])
    want: dict[str, tuple[int, str]] = {}
    if want_head_out:
        for L, nm in head_out_names(session, names, Ls).items():
            want[nm] = (L, "head_out")
    if want_pattern:
        for L in Ls:
            if arch.layers[L].mixer == "attention":
                if f"kq_soft_max-{L}" not in names:
                    raise GneError("ARCH_UNSUPPORTED", f"kq_soft_max-{L} が見えません(flash attention が切れていない可能性)")
                want[f"kq_soft_max-{L}"] = (L, "pattern")
    if want_qkv:
        for L in Ls:
            for q in ("Qcur", "Kcur", "Vcur"):
                if f"{q}-{L}" in names:
                    want[f"{q}-{L}"] = (L, q)
    for nm in extra:
        if nm in names:
            want[nm] = (-1, "extra")
    probe.set_wanted(lambda nm: nm in want)
    if lora:
        e.set_lora(ctx, lora)
    try:
        out = []
        for ids in ids_list:
            e.clear_memory(ctx)
            e.set_cvec(ctx, vectors)
            e.prefill(ctx, [ids])
            got = probe.take()
            T = len(ids)
            text, offs = token_offsets(e, ids)
            ac = AttnCapture(ids, text, offs)
            for nm, (L, what) in want.items():
                a = got.get(nm)
                if a is None:
                    continue
                if what == "extra":
                    ac.extra[nm] = a
                    continue
                li = arch.layers[L]
                if what == "pattern":
                    nh = li.n_head
                    if a.shape[0] != nh * T:
                        raise GneError("ARCH_UNSUPPORTED", f"{nm} の形 {a.shape} が (n_head·T, n_kv) = ({nh}·{T}, …) と違います")
                    ac.pattern[L] = a.reshape(nh, T, a.shape[1])[:, :, :T]
                elif what == "head_out":
                    if a.shape != (T, li.n_head * li.head_dim):
                        raise GneError("ARCH_UNSUPPORTED", f"{nm} の形 {a.shape} が (T, n_head·head_dim) と違います")
                    ac.head_out[L] = a.reshape(T, li.n_head, li.head_dim)
                else:
                    # 2 次元 (T, n·hd) でも 3 次元 [hd, n, T](行 = n·T)でも、並びは (T, n, hd)
                    hd = li.head_dim if li.mixer == "attention" else a.size // T // max(1, li.n_head)
                    ac.qkv.setdefault(L, {})[what] = a.reshape(T, a.size // (T * hd), hd) if a.size % (T * hd) == 0 else a
            out.append(ac)
        return out
    finally:
        if lora:
            e.set_lora(ctx, None)
        e.set_cvec(ctx, None)


# --- 検証(CHANGE-04 I: V15・V16) ---------------------------------------------------------------------

def writer_out_name(session, names: set[str], L: int) -> str | None:
    """層 L の書き込み行列の出力のテンソル名。qwen35 は attn_output / linear_attn_out、名前の無い arch は ffn_inp
    (= 層の入力 + attention の出力。層の入力は層 L の介入で変わらないので、差は attention の出力の差)。"""
    # 線形注意の層の attn_output-L は delta net の中間([head_v_dim, n_head, T])なので、linear_attn_out を先に見る
    order = (("linear_attn_out", "ffn_inp") if session.weights.arch.layers[L].mixer == "linear_attention"
             else ("attn_output", "ffn_inp"))
    for base in order:
        if f"{base}-{L}" in names:
            return f"{base}-{L}"
    return None


def check_capture(session, text: str = "The quick brown fox jumps over the lazy dog. 素早い茶色の狐が怠けた犬を飛び越える。",
                  *, n_tokens: int = 16) -> dict:
    """V16: パターンの各行の和 ≈ 1、因果マスク(未来は 0)、per-head 出力の形、flash attention を切った ctx と
    既定の steer ctx で greedy の出力が一致するか(または近いか)。"""
    e = session.engine
    ac = capture_attention(session, [text])[0]
    T = ac.n_tokens
    sums, future = [], []
    for P in ac.pattern.values():
        sums.append(float(np.abs(P.sum(axis=-1) - 1).max()))
        future.append(float(np.triu(P, k=1).max()) if T > 1 else 0.0)
    ids = e.tokenize(text, add_special=True, parse_special=False)
    e.free_attn_probe()
    off = e._new_ctx(n_ctx=e.n_ctx, n_batch=e.n_batch, n_ubatch=min(512, e.n_batch), n_seq_max=1, capture=None,
                     logits_all=False, flash_attn=False)              # 生成用の ctx を flash attention なしで
    try:
        a = e.generate_greedy(off, [ids], max_tokens=n_tokens)[0].token_ids
    finally:
        off.free()
    b = e.generate_greedy(e.steer_ctx(), [ids], max_tokens=n_tokens)[0].token_ids
    same = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
    return {"n_tokens": T, "pattern_layers": sorted(ac.pattern), "head_out_layers": sorted(ac.head_out),
            "row_sum_err_max": max(sums) if sums else None, "future_max": max(future) if future else None,
            "greedy_fa_off": e.detokenize(a), "greedy_default": e.detokenize(b), "greedy_same_prefix": same,
            "greedy_match": a == b,
            "ok": bool(ac.head_out and (not sums or (max(sums) < 1e-3 and max(future) < 1e-6)))}


def check_head_gain(session, L: int, h: int, *, ks: Sequence[float] = (0.0, 2.0, -1.0), text: str | None = None,
                    bake_dir=None) -> dict:
    """V15: ヘッド h の gain(k 倍)の LoRA を当てると、書き込み行列の出力の差が (k−1)·W_O[:, blk] @ head_out[:, h]
    (無介入の per-head 出力から numpy で計算)と合う(相対誤差の中央値 < 3%)。層 L の他のヘッドと、L より前の層の
    per-head 出力は変わらない。bake_dir を渡すと、列ブロックを書き換えた GGUF でも同じことを確かめる(重い)。"""
    from .gain import export_gain_lora
    from .steer import Recipe, Unit
    text = text or "Tell me a short story about a lighthouse keeper and the sea. 灯台守の話をして。"
    e, w = session.engine, session.weights
    _, lo, hi = w.head_block(L, h)
    probe, ctx = _probe(e)
    names = _names(e, probe, ctx, e.tokenize(text, add_special=True, parse_special=False)[:16])
    out_name = writer_out_name(session, names, L)
    if out_name is None:
        raise GneError("ARCH_UNSUPPORTED", f"層 {L} の書き込み行列の出力が捕捉できません")
    Ls = [x for x in range(L + 1) if w.arch.layers[x].head_dim]
    base = capture_attention(session, [text], layers=Ls, want_pattern=False, extra=[out_name])[0]
    Wb = w.head_write_matrix(L, h)                                   # (n_embd, head_dim)
    rows = []
    for k in ks:
        r = Recipe([Unit(L, None, 0.0, "head", mode="gain", k=float(k), head=h)], session.meta.model_id, name=f"v15-L{L}H{h}")
        path = session.store.gain_dir(session.meta.model_id) / f"v15-L{L}-H{h}-k{k:g}.lora.gguf"
        export_gain_lora(w, r, path)
        got = capture_attention(session, [text], layers=Ls, want_pattern=False, extra=[out_name],
                                lora=[(str(path), 1.0)])[0]
        meas = got.extra[out_name] - base.extra[out_name]             # (T, n_embd)
        pred = (k - 1.0) * base.head_out[L][:, h, :] @ Wb.T
        den = np.linalg.norm(pred, axis=1)
        keep = den > 1e-6 * max(1.0, float(den.max()))
        rel = np.linalg.norm(meas - pred, axis=1)[keep] / den[keep]
        others = np.delete(got.head_out[L], h, axis=1) - np.delete(base.head_out[L], h, axis=1)
        before = max((float(np.abs(got.head_out[x] - base.head_out[x]).max()) for x in Ls if x < L), default=0.0)
        rows.append({"k": float(k), "rel_err_median": float(np.median(rel)) if rel.size else None,
                     "rel_err_p95": float(np.percentile(rel, 95)) if rel.size else None,
                     "other_heads_max": float(np.abs(others).max()), "earlier_layers_max": before,
                     "ok": bool(rel.size and np.median(rel) < 0.03 and np.abs(others).max() < 1e-4 and before < 1e-4)})
    e.free_attn_probe()
    return {"layer": L, "head": h, "block": [lo, hi], "out_tensor": out_name, "rows": rows,
            "ok": all(r["ok"] for r in rows)}
