"""GGUF の軽い読み手(メタデータとテンソル情報だけ)。

gguf-py の GGUFReader は語彙(15〜25 万語)とマージ表を 1 要素ずつ numpy のビューに組み立てるので、
読むだけで数秒かかる(Qwen3.5-4B で約 6 秒)。本ツールが毎回要るのはスカラーのメタデータ、チャットテンプレート、
語彙数、テンソルの名前・型・形状・位置だけなので、配列は長さだけ数えて読み飛ばす。
テンソルのデータは GGUFReader と同じ形の numpy 配列(np.memmap のビュー、量子化型は uint8 のバイト列)で返す。
全フィールドを写す bake では GGUFReader を使う(weights.Weights.reader)。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFValueType
from gguf.quants import quant_shape_to_byte_shape

_SCALAR = {
    GGUFValueType.UINT8: "<B", GGUFValueType.INT8: "<b", GGUFValueType.UINT16: "<H", GGUFValueType.INT16: "<h",
    GGUFValueType.UINT32: "<I", GGUFValueType.INT32: "<i", GGUFValueType.FLOAT32: "<f", GGUFValueType.BOOL: "<?",
    GGUFValueType.UINT64: "<Q", GGUFValueType.INT64: "<q", GGUFValueType.FLOAT64: "<d",
}
_NP = {
    GGUFValueType.UINT8: np.uint8, GGUFValueType.INT8: np.int8, GGUFValueType.UINT16: np.uint16,
    GGUFValueType.INT16: np.int16, GGUFValueType.UINT32: np.uint32, GGUFValueType.INT32: np.int32,
    GGUFValueType.FLOAT32: np.float32, GGUFValueType.BOOL: np.bool_, GGUFValueType.UINT64: np.uint64,
    GGUFValueType.INT64: np.int64, GGUFValueType.FLOAT64: np.float64,
}
_TENSOR_NP = {
    GGMLQuantizationType.F16: np.float16, GGMLQuantizationType.F32: np.float32, GGMLQuantizationType.F64: np.float64,
    GGMLQuantizationType.I8: np.int8, GGMLQuantizationType.I16: np.int16, GGMLQuantizationType.I32: np.int32,
    GGMLQuantizationType.I64: np.int64,
}
_SMALL_ARRAY = 4096          # これ以下の数値配列は値も読む(層ごとの feed_forward_length など)


@dataclass
class FastTensor:
    name: str
    tensor_type: GGMLQuantizationType
    shape: list[int]          # ggml の ne(GGUFReader の ReaderTensor.shape と同じ並び)
    n_elements: int
    n_bytes: int
    data_offset: int
    _mm: np.memmap

    @property
    def data(self) -> np.ndarray:
        np_dims = tuple(reversed(self.shape))
        dt = _TENSOR_NP.get(self.tensor_type)
        if dt is None:
            raw = self._mm[self.data_offset:self.data_offset + self.n_bytes]
            return raw.reshape(quant_shape_to_byte_shape(np_dims, self.tensor_type))
        return self._mm[self.data_offset:self.data_offset + self.n_bytes].view(dt).reshape(np_dims)


class FastGGUF:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._mm = np.memmap(self.path, mode="r")
        buf = memoryview(self._mm)
        if bytes(buf[:4]) != b"GGUF":
            raise ValueError(f"{self.path}: GGUF ではありません")
        version, = struct.unpack_from("<I", buf, 4)
        if version not in (2, 3):
            raise ValueError(f"{self.path}: 未対応の GGUF バージョン {version}(ビッグエンディアンの可能性)")
        n_tensors, n_kv = struct.unpack_from("<QQ", buf, 8)
        self.values: dict[str, object] = {}
        self.types: dict[str, tuple[GGUFValueType, GGUFValueType | None]] = {}
        self.array_len: dict[str, int] = {}
        off = 24
        for _ in range(n_kv):
            key, off = self._str(buf, off)
            vtype = GGUFValueType(struct.unpack_from("<I", buf, off)[0])
            off += 4
            if vtype == GGUFValueType.ARRAY:
                etype = GGUFValueType(struct.unpack_from("<I", buf, off)[0])
                count, = struct.unpack_from("<Q", buf, off + 4)
                off += 12
                self.types[key] = (vtype, etype)
                self.array_len[key] = count
                if etype == GGUFValueType.STRING:
                    off = self._skip_strings(buf, off, count)
                elif etype == GGUFValueType.ARRAY:
                    raise ValueError(f"{key}: 入れ子の配列には対応していません")
                else:
                    size = np.dtype(_NP[etype]).itemsize * count
                    if count <= _SMALL_ARRAY:
                        self.values[key] = np.frombuffer(buf, dtype=_NP[etype], count=count, offset=off).tolist()
                    off += size
            else:
                self.types[key] = (vtype, None)
                if vtype == GGUFValueType.STRING:
                    self.values[key], off = self._str(buf, off)
                else:
                    fmt = _SCALAR[vtype]
                    self.values[key], = struct.unpack_from(fmt, buf, off)
                    off += struct.calcsize(fmt)
        infos = []
        for _ in range(n_tensors):
            name, off = self._str(buf, off)
            n_dims, = struct.unpack_from("<I", buf, off)
            dims = list(struct.unpack_from(f"<{n_dims}Q", buf, off + 4))
            off += 4 + 8 * n_dims
            ttype, toff = struct.unpack_from("<IQ", buf, off)
            off += 12
            infos.append((name, dims, GGMLQuantizationType(ttype), toff))
        align = int(self.values.get("general.alignment", 32))
        start = off + (-off % align)
        self.tensors: list[FastTensor] = []
        for name, dims, ttype, toff in infos:
            n_el = int(np.prod(dims)) if dims else 1
            bs, ts = GGML_QUANT_SIZES[ttype]
            self.tensors.append(FastTensor(name, ttype, dims, n_el, n_el * ts // bs, start + toff, self._mm))

    @staticmethod
    def _str(buf, off: int) -> tuple[str, int]:
        n, = struct.unpack_from("<Q", buf, off)
        return bytes(buf[off + 8:off + 8 + n]).decode("utf-8", "replace"), off + 8 + n

    @staticmethod
    def _skip_strings(buf, off: int, count: int) -> int:
        unpack = struct.Struct("<Q").unpack_from
        for _ in range(count):
            off += 8 + unpack(buf, off)[0]
        return off

    def get(self, key: str, default=None):
        return self.values.get(key, default)
