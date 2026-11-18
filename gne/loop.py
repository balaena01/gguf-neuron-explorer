"""同じ層を 2 回実行するモデル(arch sensenova_loop、SenseNova-Loop-9B)の、実行順の層と保存されている層の対応。docs/sensenova_loop.md。

SenseNova-Loop は、保存されている 32 層のうち 4〜27 を 2 回続けて実行する(1 回の計算で 56 層分)。GGUF は実行順の 56 層で
宣言し(block_count = 56)、重みは 1 回目の層(blk.0〜27)と最後の 4 層(blk.52〜55)だけを持つ。2 回目の層(blk.28〜51)は、
llama.cpp が読み込むときに 1 回目と同じテンソル(同じ名前)を指す。gne の層の番号は実行順(0〜55)で、グラフの捕捉の名前・
control vector の番号と同じ。2 回目の層 L の重みは、保存されている層 L − 24 の重み。

各回の中では残差が 2 本の流れになり(mHC: manifold-constrained hyper-connections)、層ごとのルーターが、どちらの流れを
どれだけ読むか・書くか・混ぜるかを決める。gne は 2 本の平均を残差として読む(回の出口で平均されるので、最終の出力に直結する)。

ここにあるのは GGUF のメタデータ({arch}.mhc.layer_kinds)だけで決まる対応。重みの読み替えは weights.py、捕捉の名前は
capture.py、LoRA のテンソル名の写し替えは to_stored(下)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .steer import GneError

ENTER, LEAVE, INSIDE = 1, 2, 4            # {arch}.mhc.layer_kinds の値(0 = 流れ 1 本の層、3 = 入って出る層)


@dataclass(frozen=True)
class LoopPlan:
    kinds: tuple[int, ...]                 # 層ごと: 0 = 流れ 1 本、1 = 2 本に入る、2 = 出る、4 = 2 本の中
    lo1: int                               # 1 回目の最初の層(実行順)
    lo2: int                               # 2 回目の最初の層
    shift: int                             # 2 回目の層 − 保存されている層(= lo2 − lo1)

    @property
    def hi1(self) -> int:
        """1 回目の最後の段(出口、27)。"""
        return self.lo2 - 1

    @property
    def hi2(self) -> int:
        return self.lo2 + self.shift

    @property
    def n_layer(self) -> int:
        return len(self.kinds)

    def second(self, L: int) -> bool:
        """L が 2 回目の層(重みを 1 回目の層と共有する)か。"""
        return self.lo2 <= int(L) < self.hi2

    def stored(self, L: int) -> int:
        """L の重みを持つ層の番号(1 回目の層の実行順の番号。GGUF の blk.N の N)。"""
        L = int(L)
        return L - self.shift if self.second(L) else L

    def visit(self, L: int) -> int:
        """0 = 繰り返しの外、1 = 1 回目、2 = 2 回目。"""
        L = int(L)
        if self.lo1 <= L < self.lo2:
            return 1
        return 2 if self.second(L) else 0

    def twin(self, L: int) -> int | None:
        """同じ重みで実行されるもう一方の回の層(繰り返しの外なら None)。"""
        v = self.visit(L)
        return None if v == 0 else (int(L) + self.shift if v == 1 else int(L) - self.shift)

    def visits(self, L: int) -> list[int]:
        """保存されている層 stored(L) が実行される層(実行順)。繰り返しの外なら [L]。"""
        s = self.stored(L)
        return [s, s + self.shift] if self.visit(s) == 1 else [s]

    def inner(self, L: int) -> bool:
        """2 本の流れの中の層(入る層・中の層)。出る層は 2 本を平均した後に control vector が足されるので含めない。"""
        return self.kinds[int(L)] in (ENTER, INSIDE)

    def label(self, L: int) -> str:
        """表示: 'L30 = 層 6 の 2 回目'。繰り返しの外なら 'L3'。"""
        L, v = int(L), self.visit(L)
        if v == 0:
            return f"L{L}"
        return f"L{L} = 層 {self.stored(L)} の {v} 回目"

    def stored_name(self, name: str) -> str:
        """テンソル名 blk.<2 回目の層>.… を、重みを持つ層の名前 blk.<L − shift>.… にする(ほかはそのまま)。"""
        m = re.match(r"blk\.(\d+)\.(.+)$", name)
        if not m or not self.second(int(m.group(1))):
            return name
        return f"blk.{int(m.group(1)) - self.shift}.{m.group(2)}"

    def to_dict(self) -> dict:
        return {"lo1": self.lo1, "lo2": self.lo2, "shift": self.shift, "hi2": self.hi2, "n_layer": self.n_layer,
                "kinds": list(self.kinds)}


def plan_from_kinds(kinds) -> LoopPlan | None:
    """{arch}.mhc.layer_kinds → LoopPlan。2 本の流れに入る層が 2 つ(1 回目と 2 回目の入口)でなければ None。"""
    if kinds is None:
        return None
    ks = tuple(int(x) for x in kinds)
    enters = [i for i, k in enumerate(ks) if k == ENTER]
    if len(enters) != 2:
        return None
    lo1, lo2 = enters
    shift = lo2 - lo1
    if lo2 + shift > len(ks):
        raise GneError("BAD_MODEL", f"mhc.layer_kinds の繰り返し(入口 {lo1}・{lo2})が層の数 {len(ks)} に収まりません")
    return LoopPlan(ks, lo1, lo2, shift)


def plan_of(get, arch: str) -> LoopPlan | None:
    """GGUF のメタデータを読む関数 get(key) から。同じ層を 2 回実行する arch でなければ None。"""
    return plan_from_kinds(get(f"{arch}.mhc.layer_kinds")) if arch else None


def stored_tensor_name(weights, name: str) -> str:
    """LoRA に書くテンソル名: 2 回目の層の名前を、重みを持つ層の名前にする(回ごとの名前を持つ llama.cpp なら、そのまま)。"""
    plan = getattr(weights, "loop", None)
    if plan is None or getattr(weights, "per_visit_names", False):
        return name
    return plan.stored_name(name)


def to_stored(weights, tensors: dict[str, tuple[np.ndarray, np.ndarray]], *, warn=None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """LoRA のテンソル {名前: (A, B)} の、2 回目の層の名前を重みを持つ層の名前に写す(llama.cpp の 2 回目の層は 1 回目と同じ
    テンソル名なので、2 回目の名前のままでは LoRA を読み込めない)。2 回目の名前を持つ llama.cpp(回ごとの名前、
    weights.per_visit_names)なら何もしない。同じテンソルに 2 つの編集が重なったら、同じ中身なら 1 つにし、違えば連結する
    (B₁A₁ + B₂A₂。足し合わせになるので、射影のように重ねがけの意味が違う編集は、作る側で保存されている層ごとにまとめておく)。"""
    plan = getattr(weights, "loop", None)
    if plan is None or getattr(weights, "per_visit_names", False):
        return tensors
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (A, B) in tensors.items():
        sname = plan.stored_name(name)
        if sname not in out:
            out[sname] = (A, B)
            continue
        A0, B0 = out[sname]
        if A0.shape == A.shape and B0.shape == B.shape and np.array_equal(A0, A) and np.array_equal(B0, B):
            continue
        from .headproj import merge_tensors
        out.update(merge_tensors({sname: (A0, B0)}, {sname: (A, B)}))
        if warn is not None:
            warn(f"{name} と {sname} は同じ重み(2 回の実行で共有)なので、2 つの編集を足し合わせて 1 つにしました")
    return out


def lora_both_visits(src, dst, plan: LoopPlan) -> int:
    """LoRA のファイル src の、1 回目の層のテンソル(blk.<1 回目の層>.*.lora_a / lora_b)を、2 回目の層の名前にも写したファイルを
    dst に書く(2 回目の層に別の名前を持つ llama.cpp で、重みの書き換えを両方の回に当てるため)。写したテンソルの数を返す。"""
    from gguf import GGUFReader, GGUFValueType, GGUFWriter
    r = GGUFReader(str(src))
    arch_f = r.fields["general.architecture"]
    w = GGUFWriter(str(dst), arch=bytes(arch_f.parts[arch_f.data[0]]).decode())
    for f in r.fields.values():
        if f.name.startswith("GGUF.") or f.name == "general.architecture" or not f.types:
            continue
        t = f.types[0]
        if t == GGUFValueType.STRING:
            w.add_key_value(f.name, bytes(f.parts[f.data[0]]).decode(), GGUFValueType.STRING)
        elif t in (GGUFValueType.FLOAT32, GGUFValueType.UINT32, GGUFValueType.INT32, GGUFValueType.BOOL):
            w.add_key_value(f.name, f.parts[f.data[0]][0].item(), t)
    names = {t.name for t in r.tensors}
    n = 0
    for t in r.tensors:
        w.add_tensor(t.name, np.array(t.data))
    for t in r.tensors:
        m = re.match(r"blk\.(\d+)\.(.+)$", t.name)
        if m and plan.visit(int(m.group(1))) == 1:
            twin = f"blk.{int(m.group(1)) + plan.shift}.{m.group(2)}"
            if twin not in names:
                w.add_tensor(twin, np.array(t.data))
                n += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return n


# --- 当て方の記録(shared = 両方の回に当てる / per_visit = 1 回目と 2 回目を別々に当てる) -------------------------------------
# レシピは loop_edits、LoRA のファイルは gne.loop_edits に記録する。記録の無い古いレシピ・LoRA は shared として読む。
# メモリの上だけで作ったレシピ(記録なし)は、そのときのセッションの当て方に従い、保存するときにそれを記録する。

LOOP_EDITS = ("shared", "per_visit")
LORA_KEY = "gne.loop_edits"
_SESSION = {"mode": None}


def set_session_mode(mode: str | None) -> None:
    """いまのセッションの当て方(同じ層を 2 回実行するモデルのときだけ。ほかは None)。レシピを保存するときの記録に使う。"""
    _SESSION["mode"] = mode


def session_mode() -> str | None:
    return _SESSION["mode"]


def lora_loop_edits(path) -> str | None:
    """LoRA のファイルに記録された当て方(無ければ None = shared として読む)。"""
    from gguf import GGUFReader
    try:
        f = GGUFReader(str(path)).fields.get(LORA_KEY)
    except Exception:
        return None
    if f is None or not f.data:
        return None
    v = bytes(f.parts[f.data[0]]).decode()
    return v if v in LOOP_EDITS else None


def per_visit_refusal(what: str) -> str:
    """回ごとに当てる指定を、2 回目の層に別の名前が無い llama.cpp で断るときの文。"""
    return (f"{what}は、1 回目と 2 回目に別々に当てる形(per_visit)ですが、この llama.cpp は 2 回目の層が 1 回目と同じテンソル名なので、"
            "片方の回だけに当てられません。")


PER_VISIT_FIX = ("回ごとに探した結果は gne heads loop-merge で両方の回に当てる形(shared)のレシピに直すと、修正なしの llama.cpp で"
                 "使えます。回ごとに測る(--score-layers loop、--select-layer)のは、修正なしでもできます")


def twin_effect(s1: float, s2: float, c: float) -> tuple[float, float]:
    """同じヘッドの 2 つの射影 (I − s₂ r₂r₂ᵀ)(I − s₁ r₁r₁ᵀ) を続けて掛けたときの実効の強さ(r₁ 方向・r₂ 方向)。
    c = r₁·r₂。r₁ に沿った出力は (1 − s₁)(1 − s₂c²) 倍、r₂ に沿った出力は (1 − s₂)(1 − s₁c²) 倍になるので、
    実効の強さは 1 からそれを引いたもの(c = 1 なら どちらも s₁ + s₂ − s₁s₂)。"""
    return 1.0 - (1.0 - s1) * (1.0 - s2 * c * c), 1.0 - (1.0 - s2) * (1.0 - s1 * c * c)
