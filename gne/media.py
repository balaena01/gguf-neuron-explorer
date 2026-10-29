"""音声の入力(LFM3-Omni のような音声を聞けるモデル)。

mmproj の音声エンコーダ(libmtmd)で音声を埋め込みの列にし、プロンプトの中では負のトークン番号(置き場所)で表す。
Engine._decode は負の番号の並びを埋め込みとして decode するので、when の付け替え・活性の捕捉・KV の位置は
ふつうのトークンと同じように扱える(llama.cpp の helper には任せない)。

メッセージの content には OpenAI の形の部品を並べられる:
    {"type": "text", "text": "..."}
    {"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}   # data: URL も可
    {"type": "input_audio", "input_audio": {"path": "speech.wav"}}                 # gne の拡張(ローカルのファイル)
wav・mp3・flac を読める(libmtmd の miniaudio がモデルのサンプルレートに直す)。
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .steer import GneError

MARKER = "<__media__>"                      # mtmd_default_marker()
AUDIO_PARTS = ("input_audio", "audio")
AUDIO_OUT_TOKENS = ("<|audio_start|>", "<|mixed_start|>")   # 音声で返し始めるトークン(gne は文だけ返すので禁じる)
CACHE_AUDIOS = 64                           # 埋め込みを覚えておく音声の数(多ターンの会話で毎回エンコードしない)


def _part_audio(part: dict) -> bytes:
    a = part.get("input_audio") or part.get("audio") or {}
    if isinstance(a, str):
        a = {"data": a}
    if a.get("path"):
        p = Path(a["path"]).expanduser()
        if not p.exists():
            raise GneError("BAD_ARGS", f"音声ファイルがありません: {p}")
        return p.read_bytes()
    data = a.get("data")
    if not data:
        raise GneError("BAD_ARGS", "input_audio に data か path が要ります")
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    try:
        return base64.b64decode(data, validate=False)
    except ValueError as ex:
        raise GneError("BAD_ARGS", f"input_audio の data が base64 として読めません: {ex}") from ex


def split_content(content) -> tuple[str, list[bytes]]:
    """content(文字列か部品のリスト)→(音声の位置に MARKER を置いた文字列, 音声のバイト列)。"""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return str(content), []
    text, audios = [], []
    for p in content:
        if isinstance(p, str):
            text.append(p)
        elif isinstance(p, dict) and p.get("type") in AUDIO_PARTS:
            text.append(MARKER)
            audios.append(_part_audio(p))
        elif isinstance(p, dict) and p.get("type") in (None, "text"):
            text.append(p.get("text") or "")
        else:
            raise GneError("BAD_ARGS", f"読めない content の部品です: {p.get('type') if isinstance(p, dict) else p!r}",
                           "text か input_audio")
    return "".join(text), audios


def normalize(messages: list[dict]) -> tuple[list[dict], list[bytes]]:
    """部品の content を文字列にしたメッセージと、出てきた順の音声。音声が無ければ audios は空。"""
    out, audios = [], []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            text, a = split_content(c)
            out.append({**m, "content": text})
            audios += a
        else:
            out.append(m)
    return out, audios


def has_audio(messages: list[dict]) -> bool:
    return any(isinstance(m.get("content"), list) and any(isinstance(p, dict) and p.get("type") in AUDIO_PARTS
                                                          for p in m["content"]) for m in messages)


def http_content(content):
    """HTTP(UI・OpenAI 互換 API)から来た content。音声の部品があれば文と base64 の音声だけを残したリストにし、
    ローカルのファイルの場所(path)は読ませない。音声が無ければ文字列にする。"""
    if not isinstance(content, list):
        return "" if content is None else content if isinstance(content, str) else str(content)
    if not any(isinstance(p, dict) and p.get("type") in AUDIO_PARTS for p in content):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in (None, "text"))
    parts = []
    for p in content:
        if not isinstance(p, dict):
            continue
        if p.get("type") in AUDIO_PARTS:
            a = p.get("input_audio") or p.get("audio")
            if not isinstance(a, dict) or not a.get("data"):
                raise GneError("BAD_ARGS", "input_audio には data(base64)が要ります", "path はこの API では使えません")
            parts.append({"type": "input_audio", "input_audio": {"data": a["data"]}})
        elif p.get("type") in (None, "text"):
            parts.append({"type": "text", "text": p.get("text") or ""})
    return parts


def audio_message(paths: list[str | Path], text: str | None = None) -> dict:
    """CLI の --audio: 音声ファイル(と、あれば文)を 1 つのユーザー発話にする。"""
    parts = [{"type": "input_audio", "input_audio": {"path": str(p)}} for p in paths]
    if text:
        parts.append({"type": "text", "text": text})
    return {"role": "user", "content": parts}


def find_mmproj(model: Path) -> Path | None:
    """モデルと同じフォルダの mmproj*.gguf。1 つならそれ、複数ならモデル名の頭がそろうもの。"""
    cands = sorted(model.parent.glob("mmproj*.gguf"))
    if len(cands) == 1:
        return cands[0]
    stem = model.stem.split("-Q")[0].lower()
    hit = [c for c in cands if stem in c.stem.lower()]
    return hit[0] if len(hit) == 1 else None


class Media:
    """mtmd のコンテキストと、音声の埋め込みの置き場所(負のトークン番号 → 埋め込みの行)。"""

    def __init__(self, engine, mmproj: Path):
        from llama_cpp import _logger
        from llama_cpp import llama_cpp as C
        from llama_cpp import mtmd_cpp as M
        self.C, self.M = C, M
        self.path = mmproj
        M.mtmd_log_set(_logger.llama_log_callback, ctypes.c_void_p(0))   # llama.cpp と同じく --verbose のときだけ出す
        p = M.mtmd_context_params_default()
        p.use_gpu = engine.n_gpu_layers != 0
        p.print_timings = False
        p.n_threads = engine.n_threads
        self.ctx = M.mtmd_init_from_file(str(mmproj).encode(), engine.model, p)
        if not self.ctx:
            raise GneError("BAD_MMPROJ", f"mmproj を読み込めません: {mmproj}", "モデルに合う mmproj-*.gguf を --mmproj で渡してください")
        if not M.mtmd_support_audio(self.ctx):
            self.close()
            raise GneError("NO_AUDIO", f"{mmproj.name} は音声のエンコーダを持っていません")
        if M.mtmd_decode_use_mrope(self.ctx):
            self.close()
            raise GneError("UNSUPPORTED", "M-RoPE の位置を使うモデルの音声入力は、まだ扱えません")
        self.n_embd = C.llama_model_n_embd_inp(engine.model)
        self.sample_rate = M.mtmd_get_audio_sample_rate(self.ctx)
        self._rows: dict[int, tuple[np.ndarray, int, bool]] = {}     # 負の番号 → (埋め込み, 行, 非因果か)
        self._by_hash: OrderedDict[str, list[int]] = OrderedDict()   # 音声の sha1 → 置き場所の番号
        self._next = -1

    def close(self) -> None:
        if self.ctx:
            self.M.mtmd_free(self.ctx)
            self.ctx = None

    def tokenize(self, text: str, audios: list[bytes]) -> list[int]:
        """描画済みのプロンプト(音声の位置に MARKER)をトークンにする。音声の区間は置き場所の番号になる。
        音声の前後の区切り(LFM3 なら <|vision_start|>…<|vision_end|>)は mtmd が足す。"""
        M = self.M
        bms = []
        try:
            for a in audios:
                buf = (ctypes.c_uint8 * len(a)).from_buffer_copy(a)
                bm = M.mtmd_helper_bitmap_init_from_buf(self.ctx, buf, len(a), False)
                if not bm or not M.mtmd_bitmap_is_audio(bm):
                    raise GneError("BAD_AUDIO", "音声として読めません(wav / mp3 / flac)")
                bms.append(bm)
            chunks = M.mtmd_input_chunks_init()
            try:
                tb = text.encode("utf-8")
                it = M.mtmd_input_text(text=tb, text_len=len(tb), add_special=False, parse_special=True)
                arr = (M.mtmd_bitmap_p_ctypes * max(1, len(bms)))(*bms)
                rc = M.mtmd_tokenize(self.ctx, chunks, ctypes.byref(it), arr, len(bms))
                if rc != 0:
                    raise GneError("BAD_AUDIO", f"音声の入ったプロンプトをトークンにできません (rc={rc})",
                                   "音声の部品の数と、プロンプトの音声の位置の数が合っているか")
                ids: list[int] = []
                k = 0
                for i in range(M.mtmd_input_chunks_size(chunks)):
                    ch = M.mtmd_input_chunks_get(chunks, i)
                    if M.mtmd_input_chunk_get_type(ch) == M.MTMD_INPUT_CHUNK_TYPE_TEXT:
                        n = ctypes.c_size_t()
                        tp = M.mtmd_input_chunk_get_tokens_text(ch, ctypes.byref(n))
                        ids += [tp[j] for j in range(n.value)]
                    else:
                        ids += self._encode(ch, audios[k])
                        k += 1
                return ids
            finally:
                M.mtmd_input_chunks_free(chunks)
        finally:
            for bm in bms:
                M.mtmd_bitmap_free(bm)

    def _encode(self, ch, audio: bytes) -> list[int]:
        key = hashlib.sha1(audio).hexdigest()
        if key in self._by_hash:
            self._by_hash.move_to_end(key)
            return self._by_hash[key]
        M = self.M
        n = M.mtmd_input_chunk_get_n_tokens(ch)
        if M.mtmd_encode_chunk(self.ctx, ch) != 0:
            raise GneError("BAD_AUDIO", "音声のエンコードに失敗しました")
        emb = np.ctypeslib.as_array(M.mtmd_get_output_embd(self.ctx), shape=(n * self.n_embd,))
        emb = emb.reshape(n, self.n_embd).astype(np.float32, copy=True)
        nc = bool(M.mtmd_decode_use_non_causal(self.ctx, ch))
        ids = list(range(self._next, self._next - n, -1))
        self._next -= n
        for j, t in enumerate(ids):
            self._rows[t] = (emb, j, nc)
        self._by_hash[key] = ids
        while len(self._by_hash) > CACHE_AUDIOS:                # 古い音声から忘れる
            for t in self._by_hash.popitem(last=False)[1]:
                self._rows.pop(t, None)
        return ids

    def row(self, tok: int) -> tuple[np.ndarray, bool]:
        try:
            emb, j, nc = self._rows[tok]
        except KeyError:
            raise GneError("INTERNAL", f"音声の置き場所 {tok} の埋め込みがありません(古い音声を忘れた後か)") from None
        return emb[j], nc
