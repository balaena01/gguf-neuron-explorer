"""llama-cpp-python 低レベル API のラッパ。設計書 §6.2。

同一モデル上に 2 種類のコンテキストを持つ:
- steer ctx: cb_eval なし。control vector を設定して生成する。複数シーケンスを 1 バッチで評価する。
- probe ctx: cb_eval あり(cb_eval はコンテキスト生成時にしか設定できない)。中間テンソルを捕捉する。
  ハイブリッド/回帰型では複数シーケンスの ubatch がシーケンス順に並べ替えられるため、
  1 シーケンスずつ、n_ubatch 以下に切って評価する(1 decode = 1 ubatch でトークン順が保たれる)。
"""

from __future__ import annotations

import ctypes
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import jinja2
import jinja2.sandbox
import numpy as np
import llama_cpp
import llama_cpp._logger as _llama_logger
from llama_cpp import llama_cpp as C

from .capture import Capture
from .steer import cvec_buffer

_backend_ready = False


def _backend_init() -> None:
    global _backend_ready
    if not _backend_ready:
        C.llama_backend_init()
        _backend_ready = True


def log_softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=-1, keepdims=True)
    return (x - np.log(np.exp(x).sum(axis=-1, keepdims=True))).astype(np.float32, copy=False)


KEEP = object()          # generate_greedy の schedule が「control vector はそのまま」を表す


@dataclass
class PrefillResult:
    logits: np.ndarray                         # (n_seq, n_vocab) 各シーケンスの最終位置
    all_logits: list[np.ndarray] | None = None  # logits_all=True のとき 各 seq の (len, n_vocab)
    n_tokens: int = 0


@dataclass
class Rollout:
    prompt_ids: list[int]
    token_ids: list[int] = field(default_factory=list)
    logprob: list[float] = field(default_factory=list)
    top_ids: list[list[int]] = field(default_factory=list)
    top_lps: list[list[float]] = field(default_factory=list)
    finish: str = "length"                     # "eog" | "length"
    text: str = ""

    @property
    def n_gen(self) -> int:
        return len(self.token_ids)


class _Ctx:
    """llama_context と付随物(バッチ、捕捉)をまとめたもの。"""

    def __init__(self, ptr, *, n_batch: int, n_ubatch: int, n_seq_max: int, capture: Capture | None):
        self.ptr = ptr
        self.n_batch = n_batch
        self.n_ubatch = n_ubatch
        self.n_seq_max = n_seq_max
        self.capture = capture
        self.batch = C.llama_batch_init(n_batch, 0, max(1, n_seq_max))

    def free(self) -> None:
        if self.ptr is not None:
            C.llama_batch_free(self.batch)
            C.llama_free(self.ptr)
            self.ptr = None


class Engine:
    def __init__(self, path: str | Path, *, n_gpu_layers: int = -1, n_ctx: int = 4096,
                 n_seq_max: int = 8, n_threads: int | None = None, n_batch: int = 2048,
                 probe_ubatch: int = 512, kv_unified: bool = True, verbose: bool = False):
        _llama_logger.set_verbose(verbose)
        _backend_init()
        self.path = Path(path)
        mp = C.llama_model_default_params()
        mp.n_gpu_layers = n_gpu_layers
        self.model = C.llama_model_load_from_file(str(self.path).encode(), mp)
        if not self.model:
            raise RuntimeError(f"モデルを読み込めません: {self.path}")
        self.vocab = C.llama_model_get_vocab(self.model)
        self.n_layer = C.llama_model_n_layer(self.model)
        self.n_embd = C.llama_model_n_embd(self.model)
        self.n_vocab = C.llama_vocab_n_tokens(self.vocab)
        self.n_ctx = n_ctx
        self.n_seq_max = n_seq_max
        self.n_batch = n_batch
        self.probe_ubatch = probe_ubatch
        self.kv_unified = kv_unified
        self.n_threads = n_threads or max(1, (os.cpu_count() or 2) // 2)
        self.n_gpu_layers = n_gpu_layers
        self.chat_template = self.meta_str("tokenizer.chat_template")
        self._steer: _Ctx | None = None
        self._probe: _Ctx | None = None
        self._probe_logits_all = False

    # --- lifecycle ------------------------------------------------------------------------------

    def close(self) -> None:
        for c in (self._steer, self._probe):
            if c is not None:
                c.free()
        self._steer = self._probe = None
        if self.model:
            C.llama_model_free(self.model)
            self.model = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def meta_str(self, key: str) -> str | None:
        buf = ctypes.create_string_buffer(1 << 16)
        n = C.llama_model_meta_val_str(self.model, key.encode(), buf, len(buf))
        if n < 0:
            return None
        if n >= len(buf):
            buf = ctypes.create_string_buffer(n + 1)
            C.llama_model_meta_val_str(self.model, key.encode(), buf, len(buf))
        return buf.value.decode("utf-8", "replace")

    def _new_ctx(self, *, n_ctx: int, n_batch: int, n_ubatch: int, n_seq_max: int,
                 capture: Capture | None, logits_all: bool) -> _Ctx:
        p = C.llama_context_default_params()
        p.n_ctx = n_ctx
        p.n_batch = n_batch
        p.n_ubatch = n_ubatch
        p.n_seq_max = n_seq_max
        p.n_threads = p.n_threads_batch = self.n_threads
        p.kv_unified = self.kv_unified   # True: n_ctx をシーケンス間で共有(False だと 1 本あたり n_ctx/n_seq_max)
        p.no_perf = False
        if logits_all:
            p.n_outputs_max_per_seq = 0   # 既定 1(各 seq 最終位置だけ)。0 = n_outputs_max
        if capture is not None:
            capture.install(p)
        ptr = C.llama_init_from_model(self.model, p)
        if not ptr:
            raise RuntimeError("llama_context を作れません(VRAM 不足の可能性)")
        return _Ctx(ptr, n_batch=n_batch, n_ubatch=n_ubatch, n_seq_max=n_seq_max, capture=capture)

    def steer_ctx(self) -> _Ctx:
        if self._steer is None:
            self._steer = self._new_ctx(n_ctx=self.n_ctx, n_batch=self.n_batch, n_ubatch=min(512, self.n_batch),
                                        n_seq_max=self.n_seq_max, capture=None, logits_all=False)
        return self._steer

    def probe_ctx(self, capture: Capture, *, logits_all: bool = False) -> _Ctx:
        """cb_eval つきコンテキスト(1 個だけ保持)。別の Capture を渡すと作り直す。"""
        c = self._probe
        if c is not None and (c.capture is not capture or self._probe_logits_all != logits_all):
            c.free()
            c = self._probe = None
        if c is None:
            ub = self.probe_ubatch
            self._probe = self._new_ctx(n_ctx=self.n_ctx, n_batch=ub, n_ubatch=ub, n_seq_max=1,
                                        capture=capture, logits_all=logits_all)
            self._probe_logits_all = logits_all
        return self._probe

    # --- chat template / tokenize ---------------------------------------------------------------

    def render_chat(self, messages: list[dict], *, enable_thinking: bool = False,
                    add_generation_prompt: bool = True, template: str | None = None) -> str:
        tpl = template or self.chat_template
        if not tpl:
            raise ValueError("GGUF に tokenizer.chat_template がありません(configs/templates で上書きしてください)")
        return render_template(tpl, messages, enable_thinking=enable_thinking,
                               add_generation_prompt=add_generation_prompt,
                               bos_token=self.token_text(C.llama_vocab_bos(self.vocab)),
                               eos_token=self.token_text(C.llama_vocab_eos(self.vocab)))

    def tokenize(self, text: str, *, add_special: bool = False, parse_special: bool = True) -> list[int]:
        b = text.encode("utf-8")
        n = C.llama_tokenize(self.vocab, b, len(b), None, 0, add_special, parse_special)
        n = -n if n < 0 else n
        arr = (C.llama_token * max(n, 1))()
        m = C.llama_tokenize(self.vocab, b, len(b), arr, n, add_special, parse_special)
        if m < 0:
            raise RuntimeError("llama_tokenize に失敗しました")
        return list(arr[:m])

    def tokenize_prompt(self, text: str) -> list[int]:
        """§5.7: 描画済みテンプレートを add_special=False, parse_special=True でトークン化し、
        モデルが BOS を要求し先頭が BOS でなければ前置する(テンプレートに BOS 文字列があれば二重にしない)。"""
        ids = self.tokenize(text, add_special=False, parse_special=True)
        if C.llama_vocab_get_add_bos(self.vocab):
            bos = C.llama_vocab_bos(self.vocab)
            if not ids or ids[0] != bos:
                ids.insert(0, bos)
        return ids

    def detokenize(self, ids: list[int], *, special: bool = False) -> str:
        if not ids:
            return ""
        arr = (C.llama_token * len(ids))(*ids)
        cap = max(64, len(ids) * 16)
        while True:
            buf = ctypes.create_string_buffer(cap)
            n = C.llama_detokenize(self.vocab, arr, len(ids), buf, cap, False, special)
            if n >= 0:
                return buf.raw[:n].decode("utf-8", "replace")
            cap = -n + 1

    def token_bytes(self, tok: int) -> bytes:
        """トークンの生のバイト列(UTF-8 の途中で切れていることがある。逐次表示は増分デコーダで)。"""
        buf = ctypes.create_string_buffer(256)
        n = C.llama_token_to_piece(self.vocab, tok, buf, len(buf), 0, False)
        return buf.raw[:max(n, 0)]

    def token_text(self, tok: int, *, special: bool = True) -> str:
        buf = ctypes.create_string_buffer(256)
        n = C.llama_token_to_piece(self.vocab, tok, buf, len(buf), 0, special)
        return buf.raw[:max(n, 0)].decode("utf-8", "replace")

    def is_eog(self, tok: int) -> bool:
        return bool(C.llama_vocab_is_eog(self.vocab, tok))

    # --- control vector / memory ----------------------------------------------------------------

    def set_cvec(self, ctx: _Ctx, vectors: dict[int, np.ndarray] | None, *,
                 layer_range: tuple[int, int] | None = None) -> None:
        """層 L(本ツールの番号 = l_out-L)の出力残差にベクトルを足す。None / 空で解除。

        常に全層分 (n_layer-1)*n_embd のゼロ埋めバッファを渡す(§5.3 の古いデータ対策)。
        layer_range を渡すと il_start/il_end をその範囲に固定する(§10 4(a))。"""
        if not vectors:
            rc = C.llama_set_adapter_cvec(ctx.ptr, None, 0, self.n_embd, -1, -1)
        else:
            buf, il_start, il_end = cvec_buffer(self.n_layer, self.n_embd, vectors)
            if layer_range is not None:
                il_start, il_end = layer_range
            ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            rc = C.llama_set_adapter_cvec(ctx.ptr, ptr, buf.size, self.n_embd, il_start, il_end)
        if rc != 0:
            raise RuntimeError(f"llama_set_adapter_cvec が失敗しました (rc={rc})")

    def clear_memory(self, ctx: _Ctx) -> None:
        C.llama_memory_clear(C.llama_get_memory(ctx.ptr), True)

    # --- evaluation -----------------------------------------------------------------------------

    def _decode(self, ctx: _Ctx, items: list[tuple[int, int, int, bool]]) -> None:
        """items: (token, pos, seq_id, want_logits)。"""
        b = ctx.batch
        for k, (tok, pos, sid, lg) in enumerate(items):
            b.token[k] = tok
            b.pos[k] = pos
            b.n_seq_id[k] = 1
            b.seq_id[k][0] = sid
            b.logits[k] = 1 if lg else 0
        b.n_tokens = len(items)
        rc = C.llama_decode(ctx.ptr, b)
        if ctx.capture is not None:
            ctx.capture.end_step()
        if rc != 0:
            raise RuntimeError(f"llama_decode が失敗しました (rc={rc})")

    def _logits(self, ctx: _Ctx, i: int) -> np.ndarray:
        ptr = C.llama_get_logits_ith(ctx.ptr, i)
        if not ptr:
            raise RuntimeError(f"logits[{i}] がありません")
        return np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).copy()

    def prefill(self, ctx: _Ctx, seqs: list[list[int]], *, logits_all: bool = False, pos0: int = 0) -> PrefillResult:
        """複数シーケンスを評価し、各 seq の最終位置ロジットを返す。メモリは呼び出し側で clear しておく。

        steer ctx: 全 seq を seq_id 0.. で 1 バッチに詰める(n_batch を超えたら分割)。
        probe ctx: 1 seq ずつ(seq の間で memory を clear)、n_ubatch 以下に切って評価する。
        pos0: 先頭トークンの位置。1 本のプロンプトを区切って評価するとき(§6.15 の位置別ステアリング)に、
        続きの区間を前の区間の後ろに置くために使う(その間は clear しない)。"""
        if not seqs or any(len(s) == 0 for s in seqs):
            raise ValueError("空のシーケンスがあります")
        last = np.empty((len(seqs), self.n_vocab), dtype=np.float32)
        all_logits = [np.empty((len(s), self.n_vocab), dtype=np.float32) for s in seqs] if logits_all else None
        n_tok = sum(len(s) for s in seqs)

        if ctx.capture is not None:
            ctx.capture.reset()
            for si, s in enumerate(seqs):
                if si:
                    self.clear_memory(ctx)
                for c0 in range(0, len(s), ctx.n_ubatch):
                    chunk = s[c0:c0 + ctx.n_ubatch]
                    items = [(t, pos0 + c0 + j, 0, logits_all or c0 + j == len(s) - 1) for j, t in enumerate(chunk)]
                    self._decode(ctx, items)
                    if logits_all:
                        for j in range(len(chunk)):
                            all_logits[si][c0 + j] = self._logits(ctx, j)
                    if c0 + len(chunk) == len(s):
                        last[si] = self._logits(ctx, len(chunk) - 1)
            return PrefillResult(last, all_logits, n_tok)

        if len(seqs) > ctx.n_seq_max:
            raise ValueError(f"シーケンス数 {len(seqs)} が n_seq_max={ctx.n_seq_max} を超えています")
        flat = [(t, pos0 + p, si, logits_all or p == len(s) - 1) for si, s in enumerate(seqs) for p, t in enumerate(s)]
        for c0 in range(0, len(flat), ctx.n_batch):
            chunk = flat[c0:c0 + ctx.n_batch]
            self._decode(ctx, chunk)
            for j, (_, p, si, lg) in enumerate(chunk):
                if not lg:
                    continue
                lo = self._logits(ctx, j)
                if logits_all:
                    all_logits[si][p - pos0] = lo
                if p - pos0 == len(seqs[si]) - 1:
                    last[si] = lo
        return PrefillResult(last, all_logits, n_tok)

    def generate_greedy(self, ctx: _Ctx, seqs: list[list[int]], *, max_tokens: int = 128, top_k: int = 5,
                        stop_on_eog: bool = True, prefilled: PrefillResult | None = None,
                        on_token: Callable[[int, int], None] | None = None,
                        schedule: Callable[[int, list[Rollout]], object] | None = None) -> list[Rollout]:
        """バッチ貪欲生成。prefilled が無ければ clear_memory → prefill から始める。
        on_token(シーケンス番号, トークン) は EOG 以外のトークンが決まるたびに呼ばれる(逐次表示用)。
        schedule(step, rollouts) は step 番目のトークンが決まった後、それを decode する前に呼ばれ、
        control vector を付け替える(§6.15)。KEEP を返せばそのまま、None で解除、dict で差し替え。
        control vector はコンテキスト単位なので、schedule を使うときはシーケンスを 1 本にする。"""
        if ctx.capture is not None:
            raise ValueError("generate_greedy は steer ctx で使ってください")
        if schedule is not None and len(seqs) != 1:
            raise ValueError("schedule はシーケンス 1 本のときだけ使えます")
        if prefilled is None:
            self.clear_memory(ctx)
            prefilled = self.prefill(ctx, seqs)
        outs = [Rollout(prompt_ids=list(s)) for s in seqs]
        pos = [len(s) for s in seqs]
        active = list(range(len(seqs)))
        cur = prefilled.logits
        for step in range(max_tokens):
            # 全語彙の log-softmax 配列は作らず、行ごとの logsumexp と生ロジットの top-k だけを使う
            m = cur.max(axis=1, keepdims=True)
            lse = m + np.log(np.exp(cur - m).sum(axis=1, keepdims=True))
            toks = cur.argmax(axis=1)
            top = np.argpartition(cur, -top_k, axis=1)[:, -top_k:]
            vals = np.take_along_axis(cur, top, axis=1)
            order = np.argsort(-vals, axis=1)
            top = np.take_along_axis(top, order, axis=1)
            top_lp = np.take_along_axis(vals, order, axis=1) - lse
            nxt = []
            for row, si in enumerate(active):
                tok = int(toks[row])
                r = outs[si]
                r.token_ids.append(tok)
                r.logprob.append(float(cur[row, tok] - lse[row, 0]))
                r.top_ids.append(top[row].tolist())
                r.top_lps.append(top_lp[row].tolist())
                if stop_on_eog and self.is_eog(tok):
                    r.finish = "eog"
                else:
                    nxt.append((si, tok))
                    if on_token is not None:
                        on_token(si, tok)
            if not nxt or step == max_tokens - 1:
                break
            if schedule is not None:
                vecs = schedule(step, outs)
                if vecs is not KEEP:
                    self.set_cvec(ctx, vecs)
            self._decode(ctx, [(tok, pos[si], si, True) for si, tok in nxt])
            cur = np.stack([self._logits(ctx, j) for j in range(len(nxt))])
            for si, _ in nxt:
                pos[si] += 1
            active = [si for si, _ in nxt]
        for r in outs:
            ids = r.token_ids[:-1] if r.finish == "eog" else r.token_ids
            r.text = self.detokenize(ids)
        return outs

    def list_names(self, ids: list[int], layer: int | None = None) -> set[str]:
        """cb_eval で観測できるテンソル名(probe ctx で 1 回 prefill)。layer を渡すとその層の名前だけ。"""
        ctx = self._probe or self.probe_ctx(Capture())     # 既存の probe ctx があれば作り直さずに使う
        cap = ctx.capture
        prev = cap._wanted
        cap.set_wanted(lambda n: False)
        cap.seen, cap.record_names = {}, True
        try:
            self.clear_memory(ctx)
            self.set_cvec(ctx, None)
            self.prefill(ctx, [ids])
        finally:
            cap.record_names = False
            cap.set_wanted(prev)
        names = set(cap.seen)
        if layer is not None:
            names = {n for n in names if n.endswith(f"-{layer}")}
        return names

    # --- misc -----------------------------------------------------------------------------------

    def system_info(self) -> str:
        return C.llama_print_system_info().decode("utf-8", "replace")

    @staticmethod
    def llama_cpp_version() -> str:
        return llama_cpp.__version__


# --- chat template ------------------------------------------------------------------------------

def _raise_exception(msg: str):
    raise jinja2.exceptions.TemplateError(msg)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def render_template(template: str, messages: list[dict], *, enable_thinking: bool = False,
                    add_generation_prompt: bool = True, bos_token: str = "", eos_token: str = "",
                    **extra) -> str:
    """GGUF の tokenizer.chat_template(HF 形式の jinja2)を描画する。"""
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                      extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    return env.from_string(template).render(
        messages=messages, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking,
        bos_token=bos_token, eos_token=eos_token, **extra)
