"""位置別ステアリング(設計書 §6.15)。

llama.cpp の control vector は全トークン位置に一律に足す。gne 自身の生成(steer / chat / eval / serve)では、
llama_set_adapter_cvec を decode の合間に付け替えて、足す位置を選ぶ。過去トークンの KV キャッシュは
計算した時点の状態のまま残るので、「プロンプトは素のまま理解し、回答だけ寄せる」ができる。

    all              全位置(既定)。cvec GGUF / bake と同じ
    gen              プロンプトの prefill は素のまま、生成ステップだけ足す(最初の 1 トークンは素の prefill から決まる)
    assistant        assistant の発話の区間だけ足す(過去の返答と、生成プロンプト <|im_start|>assistant 以降)
    after:<text>     出力に <text> が現れた後から足す
    ramp:<a>-<b>     生成トークン位置に沿って倍率を a から b へ線形に変える(8 トークン刻み)。prefill は a

sweep / lens は all 固定(対照群との比較条件をそろえるため)。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .engine import KEEP, Engine, Rollout
from .steer import GneError

RAMP_STEP = 8           # ramp の倍率を変える間隔(set_cvec はグラフの再予約を伴うので細かくしない)
_RAMP = re.compile(r"^ramp:([+-]?\d+(?:\.\d+)?)-([+-]?\d+(?:\.\d+)?)$")


@dataclass(frozen=True)
class When:
    kind: str = "all"               # all / gen / assistant / after / ramp
    text: str | None = None         # after
    a: float = 1.0                  # ramp の始め
    b: float = 1.0                  # ramp の終わり

    def __str__(self) -> str:
        if self.kind == "after":
            return f"after:{self.text}"
        if self.kind == "ramp":
            return f"ramp:{self.a:g}-{self.b:g}"
        return self.kind


def parse_when(spec: str | None) -> When:
    s = (spec or "all").strip()
    if s in ("all", "gen", "assistant"):
        return When(s)
    if s.startswith("after:"):
        if not s[6:]:
            raise GneError("BAD_ARGS", "after: の後に文字列が要ります", "例: --when 'after:結論:'")
        return When("after", text=s[6:])
    m = _RAMP.match(s)
    if m:
        return When("ramp", a=float(m.group(1)), b=float(m.group(2)))
    raise GneError("BAD_ARGS", f"when={spec!r} は読めません", "all / gen / assistant / after:<文字列> / ramp:<a>-<b>")


def chat_segments(e: Engine, messages: list[dict]) -> tuple[list[int], list[tuple[int, int, bool]]]:
    """チャットを描画してトークン化し、[(始め, 終わり, assistant の区間か)] に分ける。

    会話の先頭から 1 発話ずつ描画した長さで区切る。途中の描画が全体の先頭と一致しない
    テンプレート(過去の返答の書き方が変わるもの)では、生成プロンプトだけを assistant の区間にする。"""
    full = e.tokenize_prompt(e.render_chat(messages))
    ends = []
    for i in range(1, len(messages) + 1):
        pre = e.tokenize_prompt(e.render_chat(messages[:i], add_generation_prompt=False))
        if full[:len(pre)] != pre:
            ends = None
            break
        ends.append(len(pre))
    if ends is None:
        # 過去の返答の書き方が「最後の発話かどうか」で変わるテンプレート(Qwen3 の think 欄など)。
        # 生成プロンプトの始まりは全体の描画から、過去の返答は本文のトークン列を探して区間にする
        pre = e.tokenize_prompt(e.render_chat(messages, add_generation_prompt=False))
        start = len(pre) if full[:len(pre)] == pre else max(0, len(full) - 1)
        spans, at = [], 0
        for m in messages:
            if m.get("role") != "assistant" or not (m.get("content") or "").strip():
                continue
            k, n = _find_content(e, full, m["content"], at, start)
            if k >= 0:
                spans.append((k, k + n))
                at = k + n
        segs, cur = [], 0
        for a, b in spans:
            segs += [(cur, a, False), (a, b, True)]
            cur = b
        segs += [(cur, start, False), (start, len(full), True)]
        return full, _merge(segs)
    segs, start = [], 0
    for m, end in zip(messages, ends, strict=True):
        segs.append((start, end, m.get("role") == "assistant"))
        start = end
    segs.append((start, len(full), True))                   # 生成プロンプト(<|im_start|>assistant ...)
    return full, _merge(segs)


def _find_content(e: Engine, ids: list[int], text: str, lo: int, hi: int) -> tuple[int, int]:
    """ids[lo:hi] の中で本文 text のトークン列を探す。前後の空白でトークンの切れ目が変わることがあるので、
    先頭・末尾の 1 トークンを落とした列でも探す。戻り値は (位置, 長さ)、無ければ (-1, 0)。"""
    t = e.tokenize(text.strip(), add_special=False, parse_special=False)
    for sub in (t, t[1:], t[:-1], t[1:-1]):
        if len(sub) < 2:
            continue
        n = len(sub)
        for k in range(lo, hi - n + 1):
            if ids[k:k + n] == sub:
                return k, n
    return -1, 0


def _merge(segs: list[tuple[int, int, bool]]) -> list[tuple[int, int, bool]]:
    out: list[tuple[int, int, bool]] = []
    for a, b, st in segs:
        if b <= a:
            continue
        if out and out[-1][2] == st:
            out[-1] = (out[-1][0], b, st)
        else:
            out.append((a, b, st))
    return out


def _scaled(vectors: dict[int, np.ndarray] | None, k: float) -> dict[int, np.ndarray] | None:
    if not vectors or k == 0:
        return None
    return vectors if k == 1 else {L: (v * k).astype(np.float32) for L, v in vectors.items()}


def generate_when(e: Engine, ctx, messages: list[dict], vectors: dict[int, np.ndarray] | None, when: When | str | None,
                  *, max_tokens: int = 128, top_k: int = 5,
                  on_token: Callable[[int, int], None] | None = None) -> tuple[Rollout, dict]:
    """1 本の会話を、when に従って control vector を付け替えながら greedy 生成する。
    戻り値は (Rollout, 情報)。情報は set_cvec の回数・区間・after の切替位置など(確認用)。"""
    w = when if isinstance(when, When) else parse_when(when)
    info: dict = {"when": str(w), "set_cvec": 0}
    state = {"cur": object()}

    def put(v):
        if v is not state["cur"]:
            try:
                e.set_cvec(ctx, v)
            except ValueError as ex:
                raise GneError("BAD_RECIPE", str(ex), "ブロック 0 は control vector で操作できません(bake なら可)") from ex
            state["cur"] = v
            info["set_cvec"] += 1

    if w.kind == "assistant" and vectors:
        ids, segs = chat_segments(e, messages)
    else:
        ids = e.tokenize_prompt(e.render_chat(messages))
        segs = None
    if len(ids) + max_tokens > e.n_ctx:
        raise GneError("CONTEXT_FULL", f"会話が長すぎます({len(ids)} トークン、n_ctx {e.n_ctx})",
                       "会話をやり直すか、max_tokens を減らしてください")
    e.clear_memory(ctx)
    schedule = None
    try:
        if not vectors or w.kind == "all":
            put(vectors or None)
            pre = e.prefill(ctx, [ids])
        elif w.kind == "assistant":
            info["segments"] = [[a, b, st] for a, b, st in segs]
            pre = None
            for a, b, st in segs:
                put(vectors if st else None)
                pre = e.prefill(ctx, [ids[a:b]], pos0=a)
            put(vectors)                                 # 生成中は足す(最後の区間は生成プロンプト)
        elif w.kind == "gen":
            put(None)
            pre = e.prefill(ctx, [ids])

            def schedule(step, outs):
                return vectors if step == 0 else KEEP
        elif w.kind == "after":
            put(None)
            pre = e.prefill(ctx, [ids])
            done = {"on": False}

            def schedule(step, outs):
                if done["on"]:
                    return KEEP
                if w.text in e.detokenize(outs[0].token_ids):
                    done["on"] = True
                    info["after_step"] = step
                    return vectors
                return KEEP
        else:                                            # ramp
            put(_scaled(vectors, w.a))
            pre = e.prefill(ctx, [ids])
            info["ramp"] = [[0, w.a]]                    # [何トークン目から, 倍率]
            span = max(1, max_tokens - 1)

            def schedule(step, outs):
                q = ((step + 1) // RAMP_STEP) * RAMP_STEP
                k = round(w.a + (w.b - w.a) * min(1.0, q / span), 6)
                if k == info["ramp"][-1][1]:
                    return KEEP
                info["ramp"].append([step + 1, k])
                return _scaled(vectors, k)

        if schedule is not None:
            inner = schedule

            def schedule(step, outs, _inner=inner):
                v = _inner(step, outs)
                if v is not KEEP:
                    info["set_cvec"] += 1
                return v
        r = e.generate_greedy(ctx, [ids], max_tokens=max_tokens, top_k=top_k, prefilled=pre,
                              on_token=on_token, schedule=schedule)[0]
    finally:
        e.set_cvec(ctx, None)
    info["n_prompt"] = len(ids)
    return r, info
