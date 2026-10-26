"""GGUF の重み読み出し(推論なし)。設計書 §6.1。

メタデータとテンソル情報は軽い読み手(gguf_fast.FastGGUF)で読み、テンソルは mmap から必要な層だけを読む。
numpy の形状は ggml の ne を逆順にしたもの(§5.4): blk.L.ffn_down.weight は (n_embd, n_ff)。
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader
from gguf import quants as gquants

from .gguf_fast import FastGGUF

_NATIVE_TYPES = {GGMLQuantizationType.F32, GGMLQuantizationType.F16}


def dequantizable(qtype: GGMLQuantizationType) -> bool:
    """gguf-py に Python 実装の dequantize がある型か(F32/F16 は numpy がそのまま扱う)。"""
    return qtype in _NATIVE_TYPES or qtype in gquants._type_traits


def tensor_to_f32(tensor) -> np.ndarray:
    """ReaderTensor を float32 の numpy 配列にする。形状は (ne[-1], ..., ne[0])。"""
    qtype = tensor.tensor_type
    if qtype == GGMLQuantizationType.F32:
        return np.asarray(tensor.data, dtype=np.float32)
    if qtype == GGMLQuantizationType.F16:
        return np.asarray(tensor.data).astype(np.float32)
    if qtype not in gquants._type_traits:
        raise NotImplementedError(
            f"{tensor.name}: 量子化型 {qtype.name} は gguf-py で dequantize できません。"
            " F16 / Q8_0 版の GGUF を用意してください。"
        )
    return gquants.dequantize(tensor.data, qtype).astype(np.float32, copy=False)


@dataclass
class ModelMeta:
    path: Path
    model_id: str
    name: str                      # general.name
    arch: str                      # general.architecture
    n_layer: int                   # {arch}.block_count
    n_embd: int                    # {arch}.embedding_length
    n_ff: int                      # {arch}.feed_forward_length(層ごとの配列なら層 0 の値)
    n_ff_per_layer: list[int] | None
    n_vocab: int
    n_expert: int                  # 0 なら dense
    has_post_ffn_norm: bool        # blk.L.post_ffw_norm.weight の有無 → 等価性の警告(§2.1)
    ffn_down_scale: dict[int, float] | None   # blk.L.ffn_down.scale(BitNet 系)
    tied_output: bool              # output.weight が無く token_embd を使う
    chat_template: str | None
    ffn_down_qtype: str            # 量子化型名
    file_size: int

    def n_ff_at(self, L: int) -> int:
        return self.n_ff_per_layer[L] if self.n_ff_per_layer else self.n_ff


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", s.lower()).strip("-") or "model"


def compute_model_id(path: Path, name: str) -> str:
    """general.name + ファイルサイズ + 先頭 1MB の hash(§6.4)。"""
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
    return f"{_slug(name)}-{h.hexdigest()[:10]}"


class Weights:
    def __init__(self, path: str | Path, *, cache_layers: int = 2):
        self.path = Path(path)
        self.gguf = FastGGUF(self.path)
        self._reader: GGUFReader | None = None
        self._tensors = {t.name: t for t in self.gguf.tensors}
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_layers = cache_layers
        self._norms: dict[int, np.ndarray] = {}
        self.meta = self._read_meta()

    # --- metadata -------------------------------------------------------------------------------

    @property
    def reader(self) -> GGUFReader:
        """gguf-py の GGUFReader(全フィールドを読む、遅い)。bake で全メタデータを写すときだけ使う。"""
        if self._reader is None:
            self._reader = GGUFReader(self.path)
        return self._reader

    def field(self, key: str, default=None):
        return self.gguf.get(key, default)

    def _read_meta(self) -> ModelMeta:
        arch = self.field("general.architecture")
        if arch is None:
            raise ValueError(f"{self.path}: general.architecture がありません")
        name = self.field("general.name") or self.path.stem
        n_layer = int(self.field(f"{arch}.block_count"))
        n_embd = int(self.field(f"{arch}.embedding_length"))
        ff = self.field(f"{arch}.feed_forward_length")
        if isinstance(ff, list):
            n_ff_per_layer = [int(x) for x in ff]
            n_ff = n_ff_per_layer[0]
        else:
            n_ff_per_layer, n_ff = None, int(ff)
        n_vocab = self.gguf.array_len.get("tokenizer.ggml.tokens", 0)

        scales = {}
        for L in range(n_layer):
            t = self._tensors.get(f"blk.{L}.ffn_down.scale")
            if t is not None:
                scales[L] = float(np.asarray(tensor_to_f32(t)).reshape(-1)[0])

        down0 = self._tensors.get("blk.0.ffn_down.weight") or next(
            (t for n, t in self._tensors.items() if n.endswith("ffn_down.weight")), None)

        return ModelMeta(
            path=self.path,
            model_id=compute_model_id(self.path, name),
            name=name,
            arch=arch,
            n_layer=n_layer,
            n_embd=n_embd,
            n_ff=n_ff,
            n_ff_per_layer=n_ff_per_layer,
            n_vocab=n_vocab,
            n_expert=int(self.field(f"{arch}.expert_count", 0) or 0),
            has_post_ffn_norm=any(f"blk.{L}.post_ffw_norm.weight" in self._tensors for L in range(n_layer)),
            ffn_down_scale=scales or None,
            tied_output="output.weight" not in self._tensors,
            chat_template=self.field("tokenizer.chat_template"),
            ffn_down_qtype=down0.tensor_type.name if down0 is not None else "missing",
            file_size=self.path.stat().st_size,
        )

    def has_tensor(self, name: str) -> bool:
        return name in self._tensors

    def tensor(self, name: str):
        t = self._tensors.get(name)
        if t is None:
            raise KeyError(f"{self.path.name}: テンソル {name} がありません")
        return t

    def check_arch(self) -> list[str]:
        """警告メッセージ一覧(§12 のマトリクスに対応)。"""
        m, out = self.meta, []
        if m.has_post_ffn_norm:
            out.append("post_ffw_norm があります(Gemma 系): 方向加算としては働くが「ニューロン n を押した」とは等価ではありません(§2.1)")
        if m.n_expert or any(n.endswith("ffn_down_exps.weight") for n in self._tensors):
            out.append(f"MoE(n_expert={m.n_expert}): dense FFN 前提の機能は使えません(§12 の拡張方針を参照)")
        if m.ffn_down_scale:
            out.append("ffn_down.scale があります(BitNet 系): raw dose に係数 s を掛けます")
        try:
            qtype = GGMLQuantizationType[m.ffn_down_qtype]
            if not dequantizable(qtype):
                out.append(f"ffn_down の量子化型 {m.ffn_down_qtype} は dequantize できません: F16 / Q8_0 版を用意してください")
        except KeyError:
            out.append(f"ffn_down.weight が見つからないか型が不明です({m.ffn_down_qtype})")
        missing = [L for L in range(m.n_layer) if f"blk.{L}.ffn_down.weight" not in self._tensors]
        if missing:
            out.append(f"ffn_down.weight が無い層: {missing[:8]}{' …' if len(missing) > 8 else ''}")
        return out

    # --- ffn_down -------------------------------------------------------------------------------

    def down_matrix(self, L: int) -> np.ndarray:
        """W_down (n_embd, n_ff) float32。ニューロン n の書き込み方向は [:, n]。"""
        if L in self._cache:
            self._cache.move_to_end(L)
            return self._cache[L]
        t = self.tensor(f"blk.{L}.ffn_down.weight")
        mat = tensor_to_f32(t)
        expected = (self.meta.n_embd, self.meta.n_ff_at(L))
        if mat.shape != expected:
            raise ValueError(f"blk.{L}.ffn_down.weight の形状 {mat.shape} が期待値 {expected} と違います(§5.4)")
        self._cache[L] = mat
        while len(self._cache) > self._cache_layers:
            self._cache.popitem(last=False)
        return mat

    def down_column(self, L: int, n: int) -> np.ndarray:
        return self.down_matrix(L)[:, n].copy()

    def down_norms(self, L: int) -> np.ndarray:
        """‖W_down[:, n]‖ 全 n。"""
        if L not in self._norms:
            self._norms[L] = np.linalg.norm(self.down_matrix(L), axis=0).astype(np.float32)
        return self._norms[L]

    def unit_direction(self, L: int, n: int) -> np.ndarray:
        col = self.down_column(L, n)
        return col / np.linalg.norm(col)
