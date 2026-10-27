"""方向の射影消去(directional ablation、本体設計 §6.17、`gne ablate`)。

残差に書き込む行列 W(出力 = W x、numpy 形状 (n_embd, in))について W' = (I − s·r rᵀ) W = W − s·r (rᵀ W)。
s = 1 で W の出力から r 成分が常に消える(s < 1 は部分消去)。入力に依らないので rank-1 LoRA で表せる(§6.16 の gain と同じ構図):

    A = rᵀ W (1 × in)、B = −s·r (n_embd × 1)  →  W x + B (A x) = W x − s·r (r · W x)

対象(残差ライター)は configs/arch/residual_writers.yaml。層ごとに attn_output / ssm_out / ffn_down(MoE は
ffn_down_exps の軸 1 と ffn_down_shexp)のうち GGUF にあるもの。

埋め込み(token_embd、行ごとに e − s(e·r) r):
- bake は書き換える(--skip-embed で省く)。
- LoRA でも表せる: llama.cpp の build_inp_embd は token_embd の LoRA を get_rows の後に足す
  (lora_a = E r の (n_vocab, 1)、lora_b = −s·r の (n_embd, 1)。向きが他のテンソルと逆)。ただし出力層を埋め込みと
  共有するモデル(tied)では、出力層のテンソルも token_embd.weight という名前で、通常の向きの LoRA として掛けられて
  形が合わない。tied のモデルの LoRA には埋め込みを入れない(埋め込みの r 成分が残る。V13 は「埋め込みの r 成分のまま」を見る)。
MoE の LoRA(ffn_down_exps)は未対応(bake だけ)。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml
from gguf import GGMLQuantizationType, GGUFValueType, GGUFWriter
from gguf import quants as gquants

from .steer import GneError, copy_gguf_header
from .store import CONFIGS
from .weights import tensor_to_f32

Progress = Callable[[int, int, str], None]
# llama-quantize がこの型への量子化に imatrix を要求する(src/llama-quant.cpp tensor_requires_imatrix。埋め込みは除く)
_NEEDS_IMATRIX = {"IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS"}


def writers_table(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or CONFIGS / "arch" / "residual_writers.yaml").read_text(encoding="utf-8"))


def residual_writers(weights) -> dict[int, list[str]]:
    """層 → ライターのテンソル名(GGUF にあるものだけ)。MTP(nextn)の層は除く: 推論の本流では実行されず、
    MTP を使わない読み込みではテンソルが作られないので、LoRA に含めると読み込みが拒否される。"""
    cands = writers_table()["candidates"]
    out = {}
    for L in range(weights.meta.n_layer):
        if weights.has_tensor(f"blk.{L}.nextn.eh_proj.weight"):
            continue
        names = [f"blk.{L}.{c}.weight" for c in cands if weights.has_tensor(f"blk.{L}.{c}.weight")]
        if names:
            out[L] = names
    return out


def embed_in_lora(weights) -> bool:
    """LoRA に埋め込みを入れられるか(出力層が別にあるモデルだけ)。"""
    return not weights.meta.tied_output and weights.has_tensor("token_embd.weight")


def writer_info(weights) -> dict:
    """gne info 用: arch を確かめたか、post-norm か、ライターの種類ごとの数、LoRA では消せないバイアス。"""
    m, t = weights.meta, writers_table()
    ws = residual_writers(weights)
    counts: dict[str, int] = {}
    for names in ws.values():
        for n in names:
            kind = n.split(".")[2]
            counts[kind] = counts.get(kind, 0) + 1
    biases = sorted(n.replace(".weight", ".bias") for names in ws.values() for n in names
                    if weights.has_tensor(n.replace(".weight", ".bias")))
    warn = []
    if m.n_expert:
        warn.append("MoE の ffn_down_exps は bake でだけ射影できます(LoRA は未対応)")
    if m.arch in t.get("post_norm", []) or m.has_post_ffn_norm:
        warn.append(f"{m.arch} は post-norm(ライターの出力が norm を通ってから残差に足される)ので、射影消去は近似です")
    if m.arch not in t.get("verified", []):
        warn.append(f"{m.arch} の残差ライターはグラフを読んで確かめていません(テンソル名で推定)")
    if biases:
        warn.append(f"ライターのバイアス {len(biases)} 個は LoRA では射影できません(bake では射影する)")
    if not embed_in_lora(weights):
        warn.append("出力層が埋め込みと共有(tied)なので、LoRA では埋め込みの r 成分が残ります(bake なら消せる)")
    return {"verified": m.arch in t.get("verified", []), "post_norm": m.arch in t.get("post_norm", []),
            "counts": counts, "n_layers": len(ws), "biases": biases[:8], "embed_in_lora": embed_in_lora(weights),
            "warnings": warn, "checked_llama_cpp": t.get("checked_llama_cpp")}


def unit(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(r))
    if n == 0:
        raise GneError("BAD_DIRECTION", "方向の長さが 0 です")
    return (r / n).astype(np.float32)


def vector_key(r: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(unit(r)).tobytes()).hexdigest()[:12]


def _select_layers(weights, layers) -> list[int]:
    ws = residual_writers(weights)
    return sorted(ws) if layers is None else [L for L in layers if L in ws]


def layer_list(spec: str | list[int] | None) -> list[int] | None:
    """"all" / None → None(全層)、"10-40"・"10-40:2"・"1,5,9-12" → 層の一覧。"""
    if spec is None or spec == "all":
        return None
    if isinstance(spec, list):
        return sorted({int(x) for x in spec})
    import re
    out = set()
    for part in str(spec).split(","):
        m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*(?::\s*(\d+))?)?\s*", part)
        if not m:
            raise GneError("BAD_ARGS", f"層の指定 {spec!r} を読めません(例 all、10-40、10-40:2、1,5,9-12)")
        a, b, st = m.groups()
        out |= set(range(int(a), int(b) + 1, int(st or 1))) if b else {int(a)}
    return sorted(out)


def per_layer_directions(raw: dict[int, np.ndarray], Ls: list[int]) -> dict[int, np.ndarray]:
    """層ごとの方向(--per-layer)。方向の無い層(最終層など)は、それより前で一番近い層の方向を使う。"""
    have = sorted(raw)
    out = {}
    for L in Ls:
        src = L if L in raw else max([x for x in have if x <= L], default=have[0])
        out[L] = unit(raw[src])
    return out


# --- LoRA ---------------------------------------------------------------------------------------------

def projection_rows(weights, R: np.ndarray, *, layers: list[int] | None = None,
                    progress: Progress | None = None) -> dict[str, np.ndarray]:
    """ライターごとに R @ W((k, in))。R は (k, n_embd) の単位ベクトルの行。k 本の方向をまとめて 1 回で計算する
    (ライターを 1 つずつ dequantize するので、層選びで候補が多くても重みを読むのは 1 回)。"""
    m = weights.meta
    if m.n_expert:
        raise GneError("ARCH_UNSUPPORTED", "MoE の射影消去は LoRA では未対応です(gne ablate --bake を使ってください)")
    R = np.atleast_2d(np.asarray(R, dtype=np.float32))
    if R.shape[1] != m.n_embd:
        raise GneError("BAD_DIRECTION", f"方向の長さ {R.shape[1]} が n_embd={m.n_embd} と違います")
    ws = residual_writers(weights)
    Ls = [L for L in _select_layers(weights, layers)]
    out: dict[str, np.ndarray] = {}
    for i, L in enumerate(Ls):
        for name in ws[L]:
            W = tensor_to_f32(weights.tensor(name))
            if W.ndim != 2 or W.shape[0] != m.n_embd:
                raise GneError("ARCH_UNSUPPORTED", f"{name} の形状 {W.shape} は (n_embd, in) ではありません")
            out[name] = R @ W
        if progress:
            progress(i + 1, len(Ls), "ablate: rᵀW")
    return out


def embed_projection(weights, R: np.ndarray, *, chunk: int = 16384) -> np.ndarray:
    """E @ Rᵀ((n_vocab, k))。埋め込みを chunk 行ずつ dequantize する。"""
    t = weights.tensor("token_embd.weight")
    R = np.atleast_2d(np.asarray(R, dtype=np.float32))
    n_vocab = int(np.asarray(t.data).shape[0])
    out = np.empty((n_vocab, R.shape[0]), dtype=np.float32)
    for c0 in range(0, n_vocab, chunk):
        out[c0:c0 + chunk] = weights.embed_rows(np.arange(c0, min(c0 + chunk, n_vocab))) @ R.T
    return out


def write_ablation_lora(weights, rs: dict[str, np.ndarray], rows: dict[str, np.ndarray], path: str | Path, *,
                        meta: dict | None = None, scale: float = 1.0, embed: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """rows[テンソル名] = rᵀW(in,)、rs[テンソル名] = そのテンソルで消す単位ベクトル r を LoRA GGUF に書く。
    embed = (E r (n_vocab,), r) なら token_embd の LoRA も入れる(tied でないモデルだけ)。"""
    m = weights.meta
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(path), arch=m.arch)
    w.add_type("adapter")
    w.add_string("adapter.type", "lora")
    w.add_name(f"{m.name} ablate" + (f" {meta['direction']}" if meta and meta.get("direction") else ""))
    w.add_key_value("gne.ablate", json.dumps({**(meta or {}), "scale": scale, "embed": embed is not None},
                                             ensure_ascii=False), GGUFValueType.STRING)
    w.add_key_value("gne.source_model_id", m.model_id, GGUFValueType.STRING)
    if embed is not None:
        if not embed_in_lora(weights):
            raise GneError("BAD_ARGS", "出力層が埋め込みと共有(tied)のモデルでは、LoRA に埋め込みを入れられません")
        ea, er = embed
        w.add_tensor("token_embd.weight.lora_a", np.asarray(ea, dtype=np.float32).reshape(-1, 1))
        w.add_tensor("token_embd.weight.lora_b", (-scale * np.asarray(er, dtype=np.float32)).reshape(-1, 1))
    for name in sorted(rows, key=lambda n: (int(n.split(".")[1]), n)):
        w.add_tensor(f"{name}.lora_a", np.asarray(rows[name], dtype=np.float32).reshape(1, -1))
        w.add_tensor(f"{name}.lora_b", (-scale * np.asarray(rs[name], dtype=np.float32)).reshape(-1, 1))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return {"path": str(path), "n_tensors": len(rows) + (embed is not None), "embed": embed is not None,
            "bytes": Path(path).stat().st_size}


def export_ablation_lora(weights, r: np.ndarray | dict[int, np.ndarray], path: str | Path, *,
                         layers: list[int] | None = None, scale: float = 1.0, embed: bool = True,
                         meta: dict | None = None, progress: Progress | None = None) -> dict:
    """方向 r(1 本を全層に。dict なら層ごとの r_L を各層のライターに = --per-layer)を消す LoRA を書く。
    embed は tied でないモデルで埋め込みも入れるか(層ごとの方向のときは入れない)。"""
    Ls = _select_layers(weights, layers)
    warnings = []
    if isinstance(r, dict):
        per = per_layer_directions(r, Ls)
        rows, rs = {}, {}
        for i, L in enumerate(Ls):
            for name, v in projection_rows(weights, per[L][None, :], layers=[L]).items():
                rows[name], rs[name] = v[0], per[L]
            if progress:
                progress(i + 1, len(Ls), "ablate: rᵀW")
        emb = None
        if embed:
            warnings.append("層ごとの方向(--per-layer)では埋め込みは射影しません")
    else:
        ru = unit(r)
        part = projection_rows(weights, ru[None, :], layers=Ls, progress=progress)
        rows = {n: v[0] for n, v in part.items()}
        rs = {n: ru for n in rows}
        emb = None
        if embed and embed_in_lora(weights):
            emb = (embed_projection(weights, ru[None, :])[:, 0], ru)
        elif embed:
            warnings.append("出力層が埋め込みと共有(tied)なので、LoRA には埋め込みを入れていません(r 成分が埋め込みの分だけ残る)")
    rep = write_ablation_lora(weights, rs, rows, path, meta={**(meta or {}), "layers": Ls}, scale=scale, embed=emb)
    return {**rep, "layers": Ls, "scale": scale, "warnings": warnings}


# --- bake -------------------------------------------------------------------------------------------------

def _project(W: np.ndarray, r: np.ndarray, s: float, axis: int) -> np.ndarray:
    """W の軸 axis(長さ n_embd)から r 成分を s だけ取り除く。2D の軸 0(ライター)、軸 1(埋め込み)、3D の軸 1(MoE)。"""
    W = np.array(W, dtype=np.float32, copy=True)          # F32 のテンソルは読み取り専用の memmap
    if W.ndim == 2 and axis == 0:
        W -= s * np.outer(r, r @ W)
    elif W.ndim == 2 and axis == 1:
        W -= s * np.outer(W @ r, r)
    elif W.ndim == 3 and axis == 1:
        W -= s * np.einsum("i,ej->eij", r, np.einsum("i,eij->ej", r, W))
    else:
        raise GneError("ARCH_UNSUPPORTED", f"形状 {W.shape} の軸 {axis} は射影できません")
    return W


def _encode(W: np.ndarray, requant: str):
    if requant == "f16":
        return W.astype(np.float16), GGMLQuantizationType.F16
    if requant == "q8_0":
        return gquants.quantize(W, GGMLQuantizationType.Q8_0), GGMLQuantizationType.Q8_0
    raise GneError("BAD_ARGS", "--requant は q8_0 か f16 です")


def _moe_writers(weights, Ls: list[int]) -> dict[str, int]:
    """MoE のライター(射影の軸)。ffn_down_exps は (n_expert, n_embd, n_ff) の軸 1、ffn_down_shexp は軸 0。"""
    out = {}
    for L in Ls:
        for nm, ax in (("ffn_down_exps", 1), ("ffn_down_shexp", 0)):
            if weights.has_tensor(f"blk.{L}.{nm}.weight"):
                out[f"blk.{L}.{nm}.weight"] = ax
    return out


def bake_ablation_gguf(weights, r: np.ndarray | dict[int, np.ndarray], dst: str | Path, *, layers: list[int] | None = None,
                       scale: float = 1.0, embed: bool = True, requant: str = "q8_0", meta: dict | None = None,
                       progress: Progress | None = None) -> dict:
    """ライター(と埋め込み)から r を射影した GGUF を書く。書き換えたテンソルだけ Q8_0 か F16 にし、他は生バイトのまま写す。
    大きなモデルでもメモリに載せるのは 1 テンソルずつ(書く直前に dequantize → 射影 → 量子化)。"""
    if Path(weights.path).resolve() == Path(dst).resolve():
        raise GneError("BAD_ARGS", "--in と --out が同じファイルです")
    _encode(np.zeros((1, 32), np.float32), requant)                  # 型の誤りを先に返す
    Ls = _select_layers(weights, layers)
    ws = residual_writers(weights)
    per = per_layer_directions(r, Ls) if isinstance(r, dict) else {L: unit(r) for L in Ls}
    axes = {n: 0 for L in Ls for n in ws[L] if not n.endswith("ffn_down_exps.weight")}
    axes.update(_moe_writers(weights, Ls))
    biases = {n.replace(".weight", ".bias"): n for n in axes if weights.has_tensor(n.replace(".weight", ".bias"))}
    emb_r = None
    if embed:
        emb_r = per[Ls[0]] if isinstance(r, dict) else unit(r)        # 層ごとのときは最初の層の方向
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    w = copy_gguf_header(weights, dst, tag=" +ablate" + (f":{meta['direction']}" if meta and meta.get("direction") else ""),
                         recipe=None)
    w.add_key_value("gne.ablate", json.dumps({**(meta or {}), "layers": Ls, "scale": scale, "embed": embed,
                                              "requant": requant}, ensure_ascii=False), GGUFValueType.STRING)

    def layer_of(name: str) -> int:
        return int(name.split(".")[1])
    todo = []
    for t in weights.reader.tensors:
        if t.name in axes:
            todo.append((t, "writer"))
        elif t.name in biases:
            todo.append((t, "bias"))
        elif embed and t.name == "token_embd.weight":
            todo.append((t, "embed"))
        else:
            todo.append((t, None))
    # テンソル情報を先に書く(形とバイト数は量子化の規則から決まる)
    modified = []
    for t, kind in todo:
        if kind is None:
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, raw_dtype=t.tensor_type)
            continue
        shape = tuple(int(x) for x in reversed(t.shape))                # ne の逆順 = numpy の論理形状
        if kind == "bias":
            qt, dt, nb = GGMLQuantizationType.F32, np.float32, int(np.prod(shape)) * 4
            w.add_tensor_info(t.name, shape, dt, nb, raw_dtype=qt)
        elif requant == "f16":
            w.add_tensor_info(t.name, shape, np.float16, int(np.prod(shape)) * 2, raw_dtype=GGMLQuantizationType.F16)
        else:
            bshape = gquants.quant_shape_to_byte_shape(shape, GGMLQuantizationType.Q8_0)
            w.add_tensor_info(t.name, bshape, np.uint8, int(np.prod(bshape)), raw_dtype=GGMLQuantizationType.Q8_0)
        modified.append({"tensor": t.name, "from": t.tensor_type.name,
                         "to": "F32" if kind == "bias" else requant.upper(), "kind": kind})
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    done = 0
    for t, kind in todo:
        if kind is None:
            data = t.data
        elif kind == "writer":
            L = layer_of(t.name)
            data, _ = _encode(_project(tensor_to_f32(t), per[L], scale, axes[t.name]), requant)
        elif kind == "bias":
            b = np.array(tensor_to_f32(t), dtype=np.float32).reshape(-1)
            rr = per[layer_of(t.name)]
            data = (b - scale * (b @ rr) * rr).astype(np.float32)
        else:
            data = _bake_embed(weights, t, emb_r, scale, requant)
        w.write_tensor_data(data, tensor_endianess=weights.reader.endianess)
        if kind is not None:
            done += 1
            if progress:
                progress(done, len(modified), "ablate: bake")
    w.close()
    return {"path": str(dst), "modified": modified, "n_modified": len(modified), "layers": Ls, "scale": scale,
            "embed": embed, "requant": requant, "bytes": Path(dst).stat().st_size,
            "warnings": (["出力層が埋め込みと共有(tied)なので、埋め込みの射影は出力層にも効きます"]
                         if embed and weights.meta.tied_output else [])}


def _bake_embed(weights, t, r: np.ndarray, s: float, requant: str, chunk: int = 16384) -> np.ndarray:
    n_vocab = int(np.asarray(t.data).shape[0])
    parts = []
    for c0 in range(0, n_vocab, chunk):
        E = weights.embed_rows(np.arange(c0, min(c0 + chunk, n_vocab)))
        parts.append(_encode(_project(E, r, s, 1), requant)[0])
    return np.concatenate(parts, axis=0)


# --- --via-f16(F16 で書いて llama-quantize で元の型に戻す)------------------------------------------------

def find_llama_quantize(path: str | None = None) -> str | None:
    if path:
        return path if Path(path).exists() else None
    env = os.environ.get("GNE_LLAMA_QUANTIZE")
    if env and Path(env).exists():
        return env
    here = Path(__file__).resolve().parent.parent / "build" / "llama.cpp" / "build" / "bin" / "llama-quantize"
    if here.exists():
        return str(here)
    return shutil.which("llama-quantize")


def tensor_types_file(weights, path: Path) -> None:
    """llama-quantize の --tensor-type-file: 全テンソルを元の型に(同じ型は写すだけ、F16 にした分は元の型へ量子化)。"""
    lines = []
    for t in weights.reader.tensors:
        if not t.name.endswith(".weight"):
            continue
        pat = "^" + t.name.replace(".", "\\.") + "$"
        lines.append(f"{pat}={t.tensor_type.name.lower()}")
    path.write_text("\n".join(lines) + "\n")


def bake_via_f16(weights, r, dst: str | Path, *, layers=None, scale: float = 1.0, embed: bool = True, meta: dict | None = None,
                 llama_quantize: str | None = None, imatrix: str | None = None, keep_f16: bool = False,
                 progress: Progress | None = None) -> dict:
    """書き換えるテンソルを F16 で書いた中間の GGUF を作り、llama-quantize で元の量子化型に戻す。"""
    lq = find_llama_quantize(llama_quantize)
    if lq is None:
        raise GneError("TOOL_NOT_FOUND", "llama-quantize が見つかりません",
                       "--llama-quantize でパスを渡すか、環境変数 GNE_LLAMA_QUANTIZE を設定してください")
    # llama-quantize は、目標の型が IQ1 / IQ2 / IQ3_XXS のテンソルが 1 つでもあれば(写すだけのテンソルでも)imatrix を求める
    # (src/llama-quant.cpp の前処理。埋め込みと出力層は除く)
    need = sorted({t.tensor_type.name for t in weights.reader.tensors if t.name.endswith(".weight")
                   and t.name not in ("token_embd.weight", "output.weight")} & _NEEDS_IMATRIX)
    if need and not imatrix:
        raise GneError("NEEDS_IMATRIX", f"このモデルには {', '.join(need)} のテンソルがあり、llama-quantize は imatrix が無いと"
                       "書き出しません(写すだけのテンソルでも)",
                       "元のモデルを作ったときの imatrix を --imatrix で渡すか、--requant q8_0 を使ってください")
    dst = Path(dst)
    mid = dst.with_name(dst.name + ".f16.part")
    rep = bake_ablation_gguf(weights, r, mid, layers=layers, scale=scale, embed=embed, requant="f16", meta=meta,
                             progress=progress)
    tt = dst.with_name(dst.name + ".tensor-types.txt")
    tensor_types_file(weights, tt)
    cmd = [lq, "--tensor-type-file", str(tt)]
    if imatrix:
        cmd += ["--imatrix", str(imatrix)]
    ft = weights.field("general.file_type")
    if ft is not None:
        cmd += ["--override-kv", f"general.file_type=int:{int(ft)}"]
    cmd += [str(mid), str(dst), "Q8_0"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if p.returncode != 0:
            raise GneError("TOOL_FAILED", f"llama-quantize が失敗しました(rc={p.returncode})", (p.stderr or p.stdout)[-800:])
    finally:
        tt.unlink(missing_ok=True)
        if not keep_f16:
            mid.unlink(missing_ok=True)
    return {**rep, "path": str(dst), "requant": "via-f16", "bytes": dst.stat().st_size, "llama_quantize": lq,
            "intermediate_bytes": rep["bytes"]}
