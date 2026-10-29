"""スケールパッチ: 量子化を保ったまま、行や専門家のスラブをちょうど k 倍にする(CHANGE-05 §D)。

量子化ブロックの値は d·q(+ m)や d·sc·q − dmin·m の形なので、ブロックの fp16 のスケールを k 倍すれば全値がちょうど
k 倍になる(誤差は fp16 のスケールの丸めだけ)。ブロックは行(最後の次元)に沿って並ぶので、行 1 本 = ブロックの整数個。
k = 0 はその行のバイトを全部 0 にする。列方向(W_down[:, n])はブロックが列をまたぐので書けない(従来の再量子化を使う)。

元の GGUF は触らない。作業用コピー(~/.gne/work/<model_id>.gguf)を 1 回だけ作り、パッチの apply / revert で切り替える。
パッチは数 KB〜数 MB で、元のバイトも持つので revert で元に戻る。適用中のパッチはサイドカー JSON(.patches.json)に書く。

使える型は configs/quant_layout.yaml(Q4_0〜Q8_0、Q2_K〜Q6_K、IQ 系)と F32 / F16 / BF16。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from gguf import GGMLQuantizationType

from .steer import GneError
from .store import CONFIGS

LAYOUT: dict[str, dict] = yaml.safe_load((CONFIGS / "quant_layout.yaml").read_text(encoding="utf-8"))
_FLOAT = {"F32": 4, "F16": 2, "BF16": 2}


@dataclass
class Patch:
    tensor: str
    offset: int                  # ファイルの先頭からのバイト位置
    old: bytes
    new: bytes


@dataclass
class PatchSet:
    name: str
    model_id: str
    ops: list[dict] = field(default_factory=list)       # 人が読む操作の一覧({op, tensor, rows/expert, k/delta})
    patches: list[Patch] = field(default_factory=list)

    @property
    def n_bytes(self) -> int:
        return sum(len(p.new) for p in self.patches)

    def extend(self, other: PatchSet) -> None:
        self.ops += other.ops
        self.patches += other.patches

    # --- 保存(JSON + バイト列)---------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob, index = bytearray(), []
        for p in self.patches:
            index.append({"tensor": p.tensor, "offset": p.offset, "len": len(p.new), "at": len(blob)})
            blob += p.old + p.new
        path.with_suffix(".bin").write_bytes(bytes(blob))
        path.write_text(json.dumps({"name": self.name, "model_id": self.model_id, "ops": self.ops, "patches": index},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> PatchSet:
        path = Path(path)
        d = json.loads(path.read_text(encoding="utf-8"))
        blob = path.with_suffix(".bin").read_bytes()
        ps = [Patch(x["tensor"], x["offset"], blob[x["at"]:x["at"] + x["len"]], blob[x["at"] + x["len"]:x["at"] + 2 * x["len"]])
              for x in d["patches"]]
        return cls(d["name"], d["model_id"], d.get("ops", []), ps)

    # --- 適用・取り消し ----------------------------------------------------------------------------
    def apply(self, gguf: str | Path) -> None:
        """作業用コピーに書く。いまのバイトが元(old)か適用済み(new)でなければ、別のパッチが掛かっているので止める。"""
        _write(gguf, self.patches, forward=True)
        _sidecar(gguf, add=self.name)

    def revert(self, gguf: str | Path) -> None:
        _write(gguf, self.patches, forward=False)
        _sidecar(gguf, remove=self.name)


def _write(gguf: str | Path, patches: list[Patch], *, forward: bool) -> None:
    with open(gguf, "r+b") as f:
        for p in patches:                                   # 先に全部確かめてから書く(途中で止まらないように)
            f.seek(p.offset)
            cur = f.read(len(p.old))
            if cur not in (p.old, p.new):
                raise GneError("PATCH_CONFLICT", f"{p.tensor} @ {p.offset}: いまのバイトがパッチの前とも後とも違います",
                               "作業用コピーに別のパッチが掛かっています(gne patch revert で戻すか、作業用コピーを作り直す)")
        for p in patches:
            f.seek(p.offset)
            f.write(p.new if forward else p.old)


def _sidecar(gguf: str | Path, *, add: str | None = None, remove: str | None = None) -> list[str]:
    sc = Path(str(gguf) + ".patches.json")
    names = json.loads(sc.read_text(encoding="utf-8")) if sc.exists() else []
    if add and add not in names:
        names.append(add)
    if remove in names:
        names.remove(remove)
    sc.write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    return names


def applied(gguf: str | Path) -> list[str]:
    sc = Path(str(gguf) + ".patches.json")
    return json.loads(sc.read_text(encoding="utf-8")) if sc.exists() else []


# --- 行のバイト列 ---------------------------------------------------------------------------------

def supported(qtype: GGMLQuantizationType) -> bool:
    return qtype.name in LAYOUT or qtype.name in _FLOAT


def row_bytes(t) -> int:
    """テンソルの行(ne0 個の値)のバイト数。"""
    name, ne0 = t.tensor_type.name, int(t.shape[0])
    if name in _FLOAT:
        return ne0 * _FLOAT[name]
    lay = LAYOUT.get(name)
    if lay is None:
        raise GneError("UNSUPPORTED_QUANT", f"{t.name}: 量子化型 {name} はスケールパッチに対応していません",
                       "対応は configs/quant_layout.yaml の型と F32 / F16 / BF16。再量子化の経路(gne bake)を使ってください")
    if ne0 % lay["values"]:
        raise GneError("UNSUPPORTED_QUANT", f"{t.name}: 行の長さ {ne0} がブロック {lay['values']} の倍数ではありません")
    return ne0 // lay["values"] * lay["bytes"]


def scale_bytes(buf: bytes, qtype: GGMLQuantizationType, k: float) -> bytes:
    """行(ブロックの整数個)のバイト列の全値を k 倍にしたバイト列。"""
    if k == 0:
        return bytes(len(buf))
    name = qtype.name
    if name == "F32":
        return (np.frombuffer(buf, "<f4") * np.float32(k)).astype("<f4").tobytes()
    if name == "F16":
        return (np.frombuffer(buf, "<f2").astype(np.float32) * k).astype("<f2").tobytes()
    if name == "BF16":
        x = (np.frombuffer(buf, "<u2").astype(np.uint32) << 16).view(np.float32) * np.float32(k)
        u = x.view(np.uint32)
        u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16                     # 最近接偶数に丸める
        return u.astype("<u2").tobytes()
    lay = LAYOUT[name]
    arr = np.frombuffer(buf, np.uint8).reshape(-1, lay["bytes"]).copy()
    for s in lay["scales"]:
        d = np.ascontiguousarray(arr[:, s:s + 2]).view("<f2").reshape(-1)
        arr[:, s:s + 2] = (d.astype(np.float32) * k).astype("<f2").view(np.uint8).reshape(-1, 2)
    return arr.tobytes()


def _rows_patches(t, mm: np.memmap, rows: list[int], k: float) -> list[Patch]:
    """行番号(numpy の形を (行, ne0) に平らにしたときの番号)を k 倍するパッチ。続いた行は 1 つにまとめる。"""
    rb = row_bytes(t)
    n_rows = t.n_bytes // rb
    out = []
    rows = sorted(set(int(r) for r in rows))
    if rows and not (0 <= rows[0] and rows[-1] < n_rows):
        raise GneError("BAD_RECIPE", f"{t.name}: 行 {rows[0]}..{rows[-1]} は範囲外です(0..{n_rows - 1})")
    i = 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1] == rows[j] + 1:
            j += 1
        off = t.data_offset + rows[i] * rb
        old = bytes(mm[off:off + (j - i + 1) * rb])
        out.append(Patch(t.name, off, old, scale_bytes(old, t.tensor_type, k)))
        i = j + 1
    return out


# --- 操作 → パッチ -------------------------------------------------------------------------------

class Patcher:
    """1 つの GGUF(元のファイル)のテンソルからパッチを作る。バイトは元のファイルから読む。"""

    def __init__(self, weights):
        self.w = weights
        self.mm = np.memmap(weights.path, mode="r")

    def _t(self, name: str):
        t = self.w.tensor(name)
        if not supported(t.tensor_type):
            raise GneError("UNSUPPORTED_QUANT", f"{name}: 量子化型 {t.tensor_type.name} はスケールパッチに対応していません",
                           "対応は configs/quant_layout.yaml の型と F32 / F16 / BF16")
        return t

    def scale_row(self, name: str, row: int | tuple[int, int], k: float) -> PatchSet:
        """行 1 本を k 倍。2D (R, ne0) は行 r、3D (E, R, ne0) は (e, r)。"""
        t = self._t(name)
        dims = tuple(reversed(t.shape))
        r = row if isinstance(row, int) else row[0] * dims[1] + row[1]
        ps = PatchSet("", self.w.meta.model_id, [{"op": "scale_row", "tensor": name, "row": row if isinstance(row, int) else list(row), "k": k}])
        ps.patches = _rows_patches(t, self.mm, [r], k)
        return ps

    def scale_slab(self, name: str, e: int, k: float) -> PatchSet:
        """3D (E, R, ne0) のスラブ [e](専門家 e 全体)を k 倍。"""
        t = self._t(name)
        dims = tuple(reversed(t.shape))
        if len(dims) != 3 or not 0 <= e < dims[0]:
            raise GneError("BAD_RECIPE", f"{name}: 専門家 {e} のスラブがありません")
        ps = PatchSet("", self.w.meta.model_id, [{"op": "scale_slab", "tensor": name, "expert": e, "k": k}])
        ps.patches = _rows_patches(t, self.mm, list(range(e * dims[1], (e + 1) * dims[1])), k)
        return ps

    def add_element(self, name: str, index: int, delta: float) -> PatchSet:
        """F32 のベクトル(ルーターのバイアス)の要素に delta を足す。"""
        t = self.w.tensor(name)
        if t.tensor_type != GGMLQuantizationType.F32:
            raise GneError("UNSUPPORTED_QUANT", f"{name}: F32 だけに要素を足せます({t.tensor_type.name})")
        if not 0 <= index < t.n_elements:
            raise GneError("BAD_RECIPE", f"{name}: 要素 {index} は範囲外です")
        off = t.data_offset + 4 * index
        old = bytes(self.mm[off:off + 4])
        new = (np.frombuffer(old, "<f4") + np.float32(delta)).astype("<f4").tobytes()
        return PatchSet("", self.w.meta.model_id, [{"op": "add", "tensor": name, "index": index, "delta": delta}],
                        [Patch(name, off, old, new)])


def make_work_copy(src: str | Path, dst: str | Path) -> Path:
    """作業用コピー。同じ大きさ・同じ更新時刻の元から作ったコピーがあれば使い回す。cp --reflink=auto(Btrfs・XFS・APFS は
    すぐ、ほかは 1 回だけ全部コピー)。使い回すときは、掛かったままのパッチが無いことを確かめる。"""
    src, dst = Path(src), Path(dst)
    stamp = Path(str(dst) + ".source.json")
    want = {"source": str(src.resolve()), "size": src.stat().st_size, "mtime": src.stat().st_mtime}
    if dst.exists() and stamp.exists() and json.loads(stamp.read_text(encoding="utf-8")) == want:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    try:
        subprocess.run(["cp", "--reflink=auto", str(src), str(tmp)], check=True)
    except (OSError, subprocess.CalledProcessError):
        shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    stamp.write_text(json.dumps(want), encoding="utf-8")
    Path(str(dst) + ".patches.json").unlink(missing_ok=True)
    return dst
