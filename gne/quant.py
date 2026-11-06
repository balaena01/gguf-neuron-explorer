"""量子化型の扱いを gguf-py の外まで広げる。

gguf-py は Q1_0・Q2_0・TQ1_0 などの新しい型を知らず、Python の dequantize も一部の型だけ。実物の GGUF(例: Qwen3.8-Flash-Next の
UD-IQ1_S は Q2_0 を含む)を読むために、型の大きさと dequantize は同梱の libggml に任せる。
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf import quants as gquants
from llama_cpp._ggml import libggml

_f = libggml.ggml_blck_size
_f.restype, _f.argtypes = ctypes.c_int64, [ctypes.c_int]
_f = libggml.ggml_type_size
_f.restype, _f.argtypes = ctypes.c_size_t, [ctypes.c_int]
_f = libggml.ggml_type_name
_f.restype, _f.argtypes = ctypes.c_char_p, [ctypes.c_int]
_f = libggml.ggml_quantize_init
_f.restype, _f.argtypes = None, [ctypes.c_int]
_inited: set[int] = set()


@dataclass(frozen=True)
class ExtraQuant:
    """gguf-py の列挙に無い型。GGMLQuantizationType と同じように .name と .value を持つ。"""
    value: int
    name: str


def qtype_of(n: int):
    """GGUF の型番号 → GGMLQuantizationType か ExtraQuant。"""
    try:
        return GGMLQuantizationType(n)
    except ValueError:
        name = libggml.ggml_type_name(n)
        if not name:
            raise ValueError(f"未知の量子化型 {n}(同梱の llama.cpp より新しい GGUF の可能性)") from None
        return ExtraQuant(n, name.decode().upper())


def _value(q) -> int:
    return int(q.value)


def block_sizes(q) -> tuple[int, int]:
    """(ブロックの要素数, ブロックのバイト数)。"""
    if q in GGML_QUANT_SIZES:
        return GGML_QUANT_SIZES[q]
    return int(libggml.ggml_blck_size(_value(q))), int(libggml.ggml_type_size(_value(q)))


def byte_shape(np_dims: tuple[int, ...], q) -> tuple[int, ...]:
    """量子化テンソルのバイト列の numpy 形状(最内軸がバイト)。"""
    bs, ts = block_sizes(q)
    if np_dims[-1] % bs:
        raise ValueError(f"{q.name}: 最内軸 {np_dims[-1]} がブロック {bs} の倍数ではありません")
    return (*np_dims[:-1], np_dims[-1] // bs * ts)


def can_dequantize(q) -> bool:
    if isinstance(q, GGMLQuantizationType) and q in gquants._type_traits:
        return True
    name = f"dequantize_row_{q.name.lower().replace('_k', '_K')}"
    return hasattr(libggml, name)


def dequantize(raw: np.ndarray, q) -> np.ndarray:
    """量子化バイト列(最内軸がバイト)→ float32。形状は要素の並び (..., 要素数)。gguf-py にある型はそちら、無ければ libggml。"""
    name = getattr(q, "name", "")
    if name in ("F32", "F16") and np.asarray(raw).dtype == np.uint8:     # 量子化していない型のバイト列
        a = np.ascontiguousarray(raw)
        return a.view(np.float32 if name == "F32" else np.float16).astype(np.float32)
    if isinstance(q, GGMLQuantizationType) and q in gquants._type_traits:
        return gquants.dequantize(raw, q).astype(np.float32, copy=False)
    fn = getattr(libggml, f"dequantize_row_{q.name.lower().replace('_k', '_K')}", None)
    if fn is None:
        raise NotImplementedError(f"量子化型 {q.name} は dequantize できません")
    fn.restype, fn.argtypes = None, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    if _value(q) not in _inited:                # IQ 系は表の初期化が要る
        libggml.ggml_quantize_init(_value(q))
        _inited.add(_value(q))
    bs, ts = block_sizes(q)
    raw = np.asarray(raw, dtype=np.uint8)
    row_bytes = raw.shape[-1]
    k = row_bytes // ts * bs
    src = np.ascontiguousarray(raw).reshape(-1, row_bytes)
    out = np.empty((src.shape[0], k), dtype=np.float32)
    for i in range(src.shape[0]):
        fn(src[i].ctypes.data, out[i].ctypes.data, k)
    return out.reshape(*raw.shape[:-1], k)


def quantize(x: np.ndarray, q) -> bytes:
    """float32 の行列(最内軸が 1 行)→ 型 q のバイト列。dequantize の逆。imatrix の要る型(IQ 系の一部)は断る。"""
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    name = getattr(q, "name", str(q))
    if name == "F32":
        return x.tobytes()
    if name == "F16":
        return x.astype(np.float16).tobytes()
    if isinstance(q, GGMLQuantizationType) and name == "BF16":
        return gquants.quantize(x, q).tobytes()
    v = _value(q)
    libggml.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
    libggml.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
    if libggml.ggml_quantize_requires_imatrix(v):
        raise NotImplementedError(f"量子化型 {name} は imatrix なしでは量子化できません")
    if v not in _inited:
        libggml.ggml_quantize_init(v)
        _inited.add(v)
    bs, ts = block_sizes(q)
    rows, n = x.reshape(-1, x.shape[-1]).shape
    if n % bs:
        raise ValueError(f"{name}: 行の長さ {n} がブロック {bs} の倍数ではありません")
    out = np.empty(rows * (n // bs) * ts, dtype=np.uint8)
    fn = libggml.ggml_quantize_chunk
    fn.restype = ctypes.c_size_t
    fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    size = fn(v, x.ctypes.data, out.ctypes.data, 0, rows, n, None)
    if size != out.size:
        raise RuntimeError(f"{name}: 量子化の大きさ {size} が {out.size} と合いません")
    return out.tobytes()
