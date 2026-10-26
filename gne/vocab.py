"""語彙だけを読む(llama.cpp の vocab_only 読み込み、GPU を使わない)。rank の語の単位に使う。"""

from __future__ import annotations

import ctypes
from pathlib import Path

from llama_cpp import llama_cpp as C


class Vocab:
    def __init__(self, path: str | Path, *, verbose: bool = False):
        import llama_cpp._logger as _llama_logger
        from .engine import _backend_init
        _llama_logger.set_verbose(verbose)
        _backend_init()
        mp = C.llama_model_default_params()
        mp.vocab_only = True
        mp.n_gpu_layers = 0
        model = C.llama_model_load_from_file(str(path).encode(), mp)
        if not model:
            raise RuntimeError(f"語彙を読み込めません: {path}")
        try:
            v = C.llama_model_get_vocab(model)
            self.n = C.llama_vocab_n_tokens(v)
            buf = ctypes.create_string_buffer(512)
            self.pieces: list[bytes] = []
            for i in range(self.n):
                k = C.llama_token_to_piece(v, i, buf, len(buf), 0, False)   # 制御トークンは空になる
                self.pieces.append(buf.raw[:max(k, 0)])
        finally:
            C.llama_model_free(model)
        self._words: list[str | None] | None = None

    def text(self, i: int) -> str:
        return self.pieces[i].decode("utf-8", "replace")

    def words(self) -> list[str | None]:
        """トークン id → 語(正規化したもの)。対象外は None。設計書 §6.8:
        先頭の空白を除き小文字にそろえ、記号・数字を含むもの、1 文字のもの、UTF-8 として途中で切れているものは除く。"""
        if self._words is None:
            out: list[str | None] = []
            for p in self.pieces:
                try:
                    s = p.decode("utf-8").strip().casefold()
                except UnicodeDecodeError:
                    out.append(None)
                    continue
                ok = len(s) >= 2 and all(ch.isalpha() or ch in "'’-" for ch in s) and any(ch.isalpha() for ch in s)
                out.append(s if ok else None)
            self._words = out
        return self._words
