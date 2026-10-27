"""dose → ベクトル、レシピ、control vector / 他形式の書き出し、焼き付け(bake)。設計書 §2.2, §5.1, §6.5。

層番号の約束(§5.1):
- 本ツールの層 L = GGUF の blk.L = グラフの l_out-L(0 始まり)。
- control vector GGUF の direction.N はブロック N の出力に足される(N ≥ 1)。本ツールの層 L は direction.L に書く。
- llama_set_adapter_cvec の data は「layer 1 at [0]」: 層 L は data[(L-1)*n_embd : L*n_embd]。
- ブロック 0 は control vector で操作できない(bake なら可)。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter

from .store import CONFIGS

DoseUnit = Literal["rel", "absmax", "raw"]


class GneError(Exception):
    """--json の error.code に載せる例外。"""

    def __init__(self, code: str, message: str, hint: str | None = None):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


# --- レシピ -------------------------------------------------------------------------------------------

@dataclass
class Unit:
    layer: int
    neuron: int | None
    dose: float
    kind: Literal["neuron", "random", "direction"] = "neuron"
    seed: int | None = None          # random 用
    mode: Literal["add", "gain"] = "add"   # §6.16: add = 活性に定数を足す(dose)、gain = 出力を k 倍
    k: float | None = None           # gain の倍率(0 で消去、負で反転)
    direction: str | None = None     # direction 用: gne direction(§6.17)で作った方向の名前。その層の r_L を足す

    def label(self) -> str:
        who = {"neuron": f"N{self.neuron}", "random": f"random{self.seed}", "direction": f"dir:{self.direction}"}[self.kind]
        if self.mode == "gain":
            return f"L{self.layer} {who} ×{self.k:g}"
        return f"L{self.layer} {who} {self.dose:+g}"


@dataclass
class Recipe:
    units: list[Unit]
    model_id: str | None = None
    dose_unit: DoseUnit = "rel"
    name: str | None = None
    when: str = "all"               # §6.15 位置別ステアリング: all / gen / assistant / after:<text> / ramp:<a>-<b>

    @classmethod
    def load(cls, path: str | Path) -> Recipe:
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        units = [Unit(layer=int(u["layer"]), neuron=None if u.get("neuron") is None else int(u["neuron"]),
                      dose=float(u.get("dose", 0.0)), kind=u.get("kind", "neuron"), seed=u.get("seed"),
                      mode=u.get("mode", "add"), k=None if u.get("k") is None else float(u["k"]),
                      direction=u.get("direction"))
                 for u in d.get("units", [])]
        r = cls(units, d.get("model_id"), d.get("dose_unit", "rel"), d.get("name") or Path(path).stem, d.get("when") or "all")
        r.validate()
        return r

    def save(self, path: str | Path) -> None:
        d = {"name": self.name, "model_id": self.model_id, "dose_unit": self.dose_unit,
             **({"when": self.when} if self.when != "all" else {}),
             "units": [_unit_yaml(u) for u in self.units]}
        Path(path).write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")

    def to_dict(self) -> dict:
        return {"name": self.name, "model_id": self.model_id, "dose_unit": self.dose_unit, "when": self.when,
                "units": [asdict(u) for u in self.units]}

    def scaled(self, k: float) -> Recipe:
        # add は dose を k 倍、gain は倍率を「1(何もしない)から k_unit へ」の間で k の割合だけ動かす(eval の倍率列で 0 が無介入になるように)
        return Recipe([Unit(u.layer, u.neuron, u.dose * k if u.mode == "add" else u.dose, u.kind, u.seed, u.mode,
                            None if u.k is None else 1 + (u.k - 1) * k, u.direction) for u in self.units],
                      self.model_id, self.dose_unit, self.name, self.when)

    @property
    def add_units(self) -> list[Unit]:
        return [u for u in self.units if u.mode == "add"]

    @property
    def gain_units(self) -> list[Unit]:
        return [u for u in self.units if u.mode == "gain"]

    def validate(self) -> None:
        if self.dose_unit not in ("rel", "absmax", "raw"):
            raise GneError("BAD_RECIPE", f"dose_unit={self.dose_unit!r} は rel / absmax / raw のどれかです")
        if not self.units:
            raise GneError("BAD_RECIPE", "ユニットがありません")
        from .when import parse_when
        parse_when(self.when)                      # 書き方の誤りはここで返す
        for u in self.units:
            if u.mode not in ("add", "gain"):
                raise GneError("BAD_RECIPE", f"{u}: mode は add か gain です")
            if u.mode == "gain" and (u.kind != "neuron" or u.k is None):
                raise GneError("BAD_RECIPE", f"{u}: gain は neuron と k が要ります(例 {u.layer}:{u.neuron}:x2)")
            if u.kind == "neuron" and u.neuron is None:
                raise GneError("BAD_RECIPE", f"{u}: neuron がありません")
            if u.kind == "random" and u.seed is None:
                raise GneError("BAD_RECIPE", f"{u}: random には seed が要ります")
            if u.kind == "random" and self.dose_unit != "rel":
                raise GneError("BAD_RECIPE", "ランダム方向のユニットは dose_unit=rel でだけ使えます")
            if u.kind == "direction" and (not u.direction or u.mode != "add"):
                raise GneError("BAD_RECIPE", f"{u}: direction には方向の名前が要り、gain にはできません(例 {u.layer}:dir:yandere:+0.5)")
            if u.kind == "direction" and self.dose_unit == "absmax":
                raise GneError("BAD_RECIPE", "方向のユニットは dose_unit=rel か raw で使います(absmax はニューロン用)")
            if u.kind not in ("neuron", "random", "direction"):
                raise GneError("BAD_RECIPE", f"{u}: kind は neuron / random / direction です")


_UNIT_RE = re.compile(r"^(\d+):(?:(\d+)|random:(\d+)|dir:([\w.\-@]+)):(x?)([+-]?[\d.]+(?:e[+-]?\d+)?)$", re.I)


def parse_unit(spec: str) -> Unit:
    """"L:N:dose"(例 14:8949:+1.0)、"L:random:SEED:dose"、gain は "L:N:x2"(x の後が倍率 k、§6.16)、
    方向は "L:dir:NAME:dose"(gne direction で作った方向 NAME の r_L を層 L に足す、§6.17)。"""
    m = _UNIT_RE.match(spec.strip())
    if not m:
        raise GneError("BAD_UNIT", f"ユニット指定 {spec!r} を読めません",
                       "L:N:dose(例 14:8949:+1.0)、L:random:SEED:dose、gain は L:N:x2(倍率)、方向は L:dir:NAME:dose")
    L, n, seed, dname, gain, val = m.groups()
    if gain:
        if n is None:
            raise GneError("BAD_UNIT", f"{spec!r}: gain はニューロンにだけ使えます")
        return Unit(int(L), int(n), 0.0, mode="gain", k=float(val))
    if seed is not None:
        return Unit(int(L), None, float(val), "random", int(seed))
    if dname is not None:
        return Unit(int(L), None, float(val), "direction", direction=dname)
    return Unit(int(L), int(n), float(val))


def _unit_yaml(u: Unit) -> dict:
    d = {k: v for k, v in asdict(u).items() if v is not None and not (k == "kind" and v == "neuron")
         and not (k == "mode" and v == "add")}
    if u.mode == "gain":
        d.pop("dose", None)
    return d


# --- dose → ベクトル(§2.2, §13.3) ------------------------------------------------------------------

def random_direction(n_embd: int, seed: int, layer: int) -> np.ndarray:
    """一様ランダムな単位ベクトル(同じ seed・層なら同じ向き)。"""
    r = np.random.default_rng([seed, layer]).standard_normal(n_embd).astype(np.float32)
    return r / np.linalg.norm(r)


def _down_scale(weights, L: int) -> float:
    return (weights.meta.ffn_down_scale or {}).get(L, 1.0)


DirLoader = Callable[[str, int], np.ndarray]


def unit_vector(weights, stats, unit: Unit, dose_unit: DoseUnit = "rel", dirs: DirLoader | None = None) -> np.ndarray:
    """層 L の出力残差に足すベクトル v。
    rel:    dose · ρ_L · u_n           (u_n = W_down[:, n] / ‖W_down[:, n]‖)
    absmax: dose · absmax_n · s · W_down[:, n]
    raw:    dose · s · W_down[:, n]     (s = ffn_down.scale、無ければ 1)
    absmax と raw は「活性 a_n に定数を足す」量なので、ffn_down.scale があれば s を掛ける。
    方向(kind=direction)は d_L = dirs(名前, L)(対照の平均差 mean_A − mean_B)について
    rel: dose · ρ_L · d_L / ‖d_L‖、raw: dose · d_L(1.0 で平均差そのもの)。"""
    L = unit.layer
    if unit.kind == "direction":
        if dirs is None:
            raise GneError("BAD_RECIPE", "方向のユニットには方向の読み込み先が要ります(Session.vectors を使ってください)")
        d = np.asarray(dirs(unit.direction, L), dtype=np.float32)
        if dose_unit == "rel":
            nrm = float(np.linalg.norm(d))
            return (unit.dose * stats.rho(L) * d / nrm).astype(np.float32) if nrm > 0 else np.zeros_like(d)
        if dose_unit == "raw":
            return (unit.dose * d).astype(np.float32)
        raise GneError("BAD_RECIPE", "方向のユニットは dose_unit=rel か raw で使います")
    if unit.kind == "random":
        if dose_unit != "rel":
            raise GneError("BAD_RECIPE", "ランダム方向のユニットは dose_unit=rel でだけ使えます")
        return (unit.dose * stats.rho(L) * random_direction(weights.meta.n_embd, int(unit.seed), L)).astype(np.float32)
    col = weights.down_column(L, unit.neuron)
    if dose_unit == "rel":
        return (unit.dose * stats.rho(L) * col / np.linalg.norm(col)).astype(np.float32)
    s = _down_scale(weights, L)
    if dose_unit == "absmax":
        return (unit.dose * stats.absmax(L, unit.neuron) * s * col).astype(np.float32)
    if dose_unit == "raw":
        return (unit.dose * s * col).astype(np.float32)
    raise GneError("BAD_RECIPE", f"dose_unit={dose_unit!r}")


def recipe_vectors(weights, stats, recipe: Recipe, dirs: DirLoader | None = None) -> dict[int, np.ndarray]:
    """レシピの各ユニットのベクトルを層ごとに合計する(§2.3)。"""
    out: dict[int, np.ndarray] = {}
    for u in recipe.add_units:                 # gain は control vector ではなく LoRA で当てる(§6.16)
        v = unit_vector(weights, stats, u, recipe.dose_unit, dirs)
        out[u.layer] = out[u.layer] + v if u.layer in out else v
    return out


def unit_details(weights, stats, recipe: Recipe, dirs: DirLoader | None = None) -> list[dict]:
    """各ユニットの大きさ。rel_push = ‖v‖ / ρ_L、add = 活性 a_n に足した相当量(符号つき)。
    方向のユニットは diff_rel = ‖平均差‖ / ρ_L(rel dose がこの値なら平均差と同じ大きさ)も返す。"""
    rows = []
    for u in recipe.units:
        if u.mode == "gain":
            rows.append({"layer": u.layer, "neuron": u.neuron, "kind": u.kind, "seed": None, "dose": 0.0, "norm": 0.0,
                         "rel_push": None, "rho": None, "add": None, "down_norm": None, "mode": "gain", "k": u.k})
            continue
        v = unit_vector(weights, stats, u, recipe.dose_unit, dirs)
        norm = float(np.linalg.norm(v))
        try:
            rho = stats.rho(u.layer)
        except KeyError:
            rho = None
        d = {"layer": u.layer, "neuron": u.neuron, "kind": u.kind, "seed": u.seed, "dose": u.dose,
             "norm": norm, "rel_push": norm / rho if rho else None, "rho": rho, "add": None, "down_norm": None}
        if u.kind == "direction":
            d["direction"] = u.direction
            d["diff_rel"] = float(np.linalg.norm(dirs(u.direction, u.layer))) / rho if rho else None
        if u.kind == "neuron":
            dn = float(weights.down_norms(u.layer)[u.neuron])
            d["down_norm"] = dn
            d["add"] = float(np.sign(u.dose)) * norm / (dn * _down_scale(weights, u.layer))
        rows.append(d)
    return rows


def cvec_buffer(n_layer: int, n_embd: int, vectors: dict[int, np.ndarray]) -> tuple[np.ndarray, int, int]:
    """ゼロ埋め (n_layer-1)*n_embd のバッファに vectors[L] を [(L-1)*n_embd : L*n_embd] へ置く。
    戻り値 (buf, il_start, il_end)。il_start/il_end は vectors に含まれる層の最小・最大。"""
    if not vectors:
        raise ValueError("vectors が空です")
    buf = np.zeros((n_layer - 1) * n_embd, dtype=np.float32)
    for L, v in vectors.items():
        L = int(L)
        if L == 0:
            raise ValueError("層 0 は control vector で操作できません(llama.cpp の仕様、§5.1)")
        if not 1 <= L < n_layer:
            raise ValueError(f"層 {L} は範囲外です(1..{n_layer - 1})")
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if v.size != n_embd:
            raise ValueError(f"層 {L} のベクトル長 {v.size} が n_embd={n_embd} と違います")
        buf[(L - 1) * n_embd:L * n_embd] += v
    layers = sorted(int(L) for L in vectors)
    return buf, layers[0], layers[-1]


def export_cvec_gguf(vectors: dict[int, np.ndarray], path: str | Path, *, model_hint: str, n_layer: int,
                     extra: dict[str, str] | None = None) -> None:
    """cvector-generator の export_gguf と同じ形式で書く:
    general.architecture="controlvector", controlvector.model_hint, controlvector.layer_count,
    テンソル direction.{1..n_layer-1}(F32, 1D)。vectors に無い層はゼロ。"""
    n_embd = None
    for L, v in vectors.items():
        if int(L) == 0:
            raise ValueError("層 0 は control vector で操作できません")
        n_embd = np.asarray(v).size if n_embd is None else n_embd
        if np.asarray(v).size != n_embd:
            raise ValueError("層ごとのベクトル長がそろっていません")
    if n_embd is None:
        raise ValueError("vectors が空です")
    buf, _, _ = cvec_buffer(n_layer, n_embd, vectors)
    w = GGUFWriter(str(path), arch="controlvector")
    w.add_string("controlvector.model_hint", model_hint)
    w.add_int32("controlvector.layer_count", n_layer - 1)
    for k, val in (extra or {}).items():
        w.add_string(k, val)
    for L in range(1, n_layer):
        w.add_tensor(f"direction.{L}", buf[(L - 1) * n_embd:L * n_embd].copy())
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def load_cvec_like_common(paths_strengths: list[tuple[str | Path, float]]) -> tuple[int, np.ndarray]:
    """common/common.cpp の common_control_vector_load と同じ規則で読む(llama-cli の --control-vector 相当)。
    戻り値 (n_embd, data)。data は「layer 1 at [0]」、長さは最大の層番号 × n_embd。"""
    n_embd, data = -1, np.zeros(0, dtype=np.float32)
    for path, strength in paths_strengths:
        r = GGUFReader(str(path))
        cur_embd, cur = -1, np.zeros(0, dtype=np.float32)
        for t in r.tensors:
            head, _, idx = t.name.partition(".")
            if head != "direction" or not idx.lstrip("-").isdigit() or int(idx) < 0:
                raise ValueError(f"{path}: invalid/unparsable direction tensor layer index ({t.name})")
            layer = int(idx)
            if layer == 0:
                raise ValueError(f"{path}: invalid (zero) direction tensor layer index")
            if t.tensor_type != GGMLQuantizationType.F32:
                raise ValueError(f"{path}: invalid (non-F32) direction tensor type ({t.name})")
            if len(t.shape) != 1:
                raise ValueError(f"{path}: invalid (non-1D) direction tensor shape ({t.name})")
            vec = np.asarray(t.data, dtype=np.float32).reshape(-1)
            if cur_embd == -1:
                cur_embd = vec.size
            elif vec.size != cur_embd:
                raise ValueError(f"{path}: direction tensor does not match previous dimensions")
            if cur.size < cur_embd * layer:
                cur = np.concatenate([cur, np.zeros(cur_embd * layer - cur.size, dtype=np.float32)])
            cur[cur_embd * (layer - 1):cur_embd * layer] += vec * strength
        if n_embd != -1 and cur_embd != n_embd:
            raise ValueError(f"{path}: control vectors does not match previous dimensions")
        n_embd = cur_embd
        if data.size < cur.size:
            data = np.concatenate([data, np.zeros(cur.size - data.size, dtype=np.float32)])
        data[:cur.size] += cur
    return n_embd, data


def import_cvec_gguf(path: str | Path, *, source: Literal["exact", "generator"] = "exact") -> dict[int, np.ndarray]:
    """control vector GGUF → {層 L: ベクトル}。
    exact: direction.N を層 N の値として扱う。
    generator: llama-cvector-generator の 1 ずれ(l_out-il で測って direction.(il+1) に書く、§5.2)を補正し、
               direction.N を層 N-1 の測定値として扱う。"""
    n_embd, data = load_cvec_like_common([(path, 1.0)])
    out: dict[int, np.ndarray] = {}
    for i in range(data.size // n_embd):
        v = data[i * n_embd:(i + 1) * n_embd]
        if not np.any(v):
            continue
        N = i + 1
        out[N - 1 if source == "generator" else N] = v.copy()
    return out


# --- 他形式の書き出し(§6.5 export_vectors) -----------------------------------------------------------

LAYER_SEMANTICS = ("vectors[L] は層 L(0 始まり、GGUF の blk.L、HF の model.layers[L])の出力の残差に"
                   "全トークン位置で足すベクトル。llama.cpp の control vector では direction.L に当たる")


def export_vectors(vectors: dict[int, np.ndarray], path: str | Path, *,
                   fmt: Literal["cvec", "npz", "json", "safetensors"], n_layer: int, model_hint: str,
                   meta: dict | None = None) -> None:
    """cvec 以外は他ツール(HF の forward hook、repeng 等)向け。どの形式にも meta(model_id、レシピ等)を付ける。"""
    meta = {"layer_semantics": LAYER_SEMANTICS, "n_layer": n_layer, "model_hint": model_hint, **(meta or {})}
    vecs = {int(L): np.asarray(v, dtype=np.float32).reshape(-1) for L, v in sorted(vectors.items())}
    if fmt == "cvec":
        export_cvec_gguf(vecs, path, model_hint=model_hint, n_layer=n_layer,
                         extra={"gne.meta": json.dumps(meta, ensure_ascii=False)})
    elif fmt == "npz":
        np.savez(path, layers=np.array(list(vecs), dtype=np.int32),
                 **{f"layer_{L}": v for L, v in vecs.items()}, meta=json.dumps(meta, ensure_ascii=False))
    elif fmt == "json":
        Path(path).write_text(json.dumps({"meta": meta, "vectors": {str(L): v.tolist() for L, v in vecs.items()}},
                                         ensure_ascii=False))
    elif fmt == "safetensors":
        from safetensors.numpy import save_file
        save_file({f"direction.{L}": v for L, v in vecs.items()}, str(path),
                  metadata={k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for k, v in meta.items()})
    else:
        raise GneError("BAD_FORMAT", f"形式 {fmt!r} は cvec / npz / json / safetensors のどれかです")


# --- 焼き付け(§6.5.1) ------------------------------------------------------------------------------------

def bias_support(arch: str, table_path: str | Path | None = None) -> tuple[str | None, str, str]:
    """(mode, 理由, 確認した llama.cpp)。mode は "add"(新しい bias を足す)/ "existing"(既存 bias に加算)/ None。"""
    t = yaml.safe_load(Path(table_path or CONFIGS / "arch" / "bias_support.yaml").read_text(encoding="utf-8"))
    checked = t.get("checked_llama_cpp", "?")
    if arch in (t.get("optional") or []):
        return "add", "ローダが ffn_down.bias を任意テンソルとして作り、build_ffn に渡す", checked
    if arch in (t.get("required") or []):
        return "existing", "ローダが ffn_down.bias を必須で作る(既存の bias に加算する)", checked
    if arch in (t.get("ignored") or []):
        return None, "ローダは ffn_down.bias を受け付けるが、グラフが使わない(焼いても効かない)", checked
    return None, "ローダが ffn_down.bias を作らないので、足したテンソルで読み込みが拒否される(wrong number of tensors)", checked


@dataclass
class BakeReport:
    src: str
    dst: str
    arch: str
    mode: str
    layers: list[int]
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    n_tensors: int = 0


def copy_gguf_header(weights, dst: str | Path, *, tag: str, recipe: Recipe | None) -> GGUFWriter:
    """元の GGUF のメタデータを写した GGUFWriter を作る(general.name に tag を付け、gne.recipe にレシピを残す)。
    テンソルは呼び出し側が add_tensor_info → write_* で書く。bake(§6.5.1)と gain の書き換え(§6.16)で共用。"""
    m = weights.meta
    r: GGUFReader = weights.reader
    w = GGUFWriter(str(dst), arch=m.arch, endianess=r.endianess)
    align = r.fields.get("general.alignment")
    if align is not None:
        w.data_alignment = int(align.contents())
    for f in r.fields.values():
        if f.name == "general.architecture" or f.name.startswith("GGUF."):
            continue
        vtype = f.types[0]
        sub = f.types[-1] if vtype == GGUFValueType.ARRAY else None
        val = f.contents()
        if f.name == "general.name":
            val = f"{val}{tag}"
        w.add_key_value(f.name, val, vtype, sub_type=sub)
    if "general.name" not in r.fields:
        w.add_key_value("general.name", f"{m.name}{tag}", GGUFValueType.STRING)
    w.add_key_value("gne.recipe", json.dumps(recipe.to_dict() if recipe else {}, ensure_ascii=False), GGUFValueType.STRING)
    w.add_key_value("gne.source_model_id", m.model_id, GGUFValueType.STRING)
    return w


def bake_bias_gguf(weights, vectors: dict[int, np.ndarray], src: str | Path, dst: str | Path, *,
                   recipe: Recipe | None = None) -> BakeReport:
    """v を blk.L.ffn_down.bias として書き込んだ派生 GGUF を作る。他のテンソルとメタデータは生バイトのまま写す。"""
    m = weights.meta
    mode, reason, checked = bias_support(m.arch)
    if mode is None:
        raise GneError("ARCH_UNSUPPORTED", f"arch {m.arch!r} には bake できません: {reason}(確認した llama.cpp: {checked})",
                       "control vector GGUF(gne export-cvec)を llama-cli / llama-server の --control-vector で使ってください")
    if m.has_post_ffn_norm:
        raise GneError("ARCH_UNSUPPORTED", "post_ffw_norm がある arch では bias と control vector が等価になりません")
    if m.n_expert:
        raise GneError("ARCH_UNSUPPORTED", "MoE の専門家別 bias には対応していません")
    if Path(src).resolve() == Path(dst).resolve():
        raise GneError("BAD_ARGS", "--in と --out が同じファイルです")
    vecs = {int(L): np.asarray(v, dtype=np.float32).reshape(-1) for L, v in vectors.items()}
    for L, v in vecs.items():
        if not 0 <= L < m.n_layer or v.size != m.n_embd:
            raise GneError("BAD_RECIPE", f"層 {L} のベクトルが不正です(n_layer={m.n_layer}, n_embd={m.n_embd})")

    r: GGUFReader = weights.reader
    tag = f" +recipe:{recipe.name}" if recipe and recipe.name else " +recipe"
    w = copy_gguf_header(weights, dst, tag=tag, recipe=recipe)

    report = BakeReport(str(src), str(dst), m.arch, mode, sorted(vecs))
    plan: list[tuple[str, np.ndarray, GGMLQuantizationType]] = []   # (name, data, type)
    names = {t.name for t in r.tensors}
    for t in r.tensors:
        hit = re.fullmatch(r"blk\.(\d+)\.ffn_down\.bias", t.name)
        if hit and int(hit.group(1)) in vecs:
            L = int(hit.group(1))
            if t.tensor_type not in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
                raise GneError("ARCH_UNSUPPORTED", f"{t.name} の型 {t.tensor_type.name} には加算できません")
            base = np.asarray(t.data, dtype=np.float32).reshape(-1)
            data = (base + vecs[L]).astype(np.float16 if t.tensor_type == GGMLQuantizationType.F16 else np.float32)
            plan.append((t.name, data, t.tensor_type))
            report.modified.append(t.name)
            continue
        plan.append((t.name, t.data, t.tensor_type))
        hit = re.fullmatch(r"blk\.(\d+)\.ffn_down\.weight", t.name)
        if hit and int(hit.group(1)) in vecs:
            L = int(hit.group(1))
            bname = f"blk.{L}.ffn_down.bias"
            if bname in names:
                continue                         # 既存 bias は上で加算する
            if mode != "add":
                raise GneError("ARCH_UNSUPPORTED", f"{bname} が必須の arch なのにファイルにありません")
            plan.append((bname, vecs[L].copy(), GGMLQuantizationType.F32))
            report.added.append(bname)
    for name, data, qtype in plan:
        w.add_tensor_info(name, data.shape, data.dtype, data.nbytes, raw_dtype=qtype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for _, data, _ in plan:
        w.write_tensor_data(data, tensor_endianess=r.endianess)
    w.close()
    report.n_tensors = len(plan)
    return report
