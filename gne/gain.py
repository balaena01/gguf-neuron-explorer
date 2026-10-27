"""ゲイン(倍率)ステアリングと消去(設計書 §6.16)。

a_n = silu(g_n) · u_n、u_n = W_up[n, :] · x なので、W_up の n 行目を k 倍すると活性 a_n が k 倍になる
(silu は gate 側なので影響しない)。残差に加わるのは k · a_n · W_down[:, n]。k = 0 で消去、k < 0 で反転。
足し算(add、§2)と違い、入力に応じて働く量が変わるので「その状況でだけ強く出る/出なくなる」が作れる。

出力の 2 方式:
- (a) LoRA アダプタ GGUF(既定): rank-r。A の i 行目 = W_up[n_i, :](dequantize 済み)、B[n_i, i] = k_i − 1。
  W x + B(A x) = W x + Σ_i (k_i − 1)·u_{n_i}·e_{n_i}。llama.cpp の --lora でそのまま使える。
- (b) 書き換え GGUF: W_up[L] の n 行目を k 倍し、そのテンソルだけ Q8_0(または F16)で書き出す。他は生バイトのまま。

LoRA GGUF の形式(llama.cpp src/llama-adapter.cpp と src/llama-graph.cpp build_lora_mm で確認、docs/verification.md §11.2):
- general.type = "adapter"、general.architecture = 元モデルの arch、adapter.type = "lora"
- テンソル blk.L.ffn_up.weight.lora_a(ggml ne [n_embd, r]、numpy (r, n_embd))と .lora_b(ne [r, n_ff]、numpy (n_ff, r))
- 倍率 scale = adapter_scale · alpha / r(alpha が 0 か無ければ adapter_scale)。層ごとに r が違っても倍率が 1 になるよう、
  adapter.lora.alpha は書かない
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFValueType, GGUFWriter
from gguf import quants as gquants

from .steer import GneError, Recipe, Unit, copy_gguf_header


def gain_by_layer(units: list[Unit]) -> dict[int, list[tuple[int, float]]]:
    """gain ユニットを層ごとに [(ニューロン, k)] にまとめる。同じニューロンが重なれば倍率を掛け合わせる。"""
    out: dict[int, dict[int, float]] = defaultdict(dict)
    for u in units:
        if u.mode != "gain":
            continue
        out[u.layer][u.neuron] = out[u.layer].get(u.neuron, 1.0) * float(u.k)
    return {L: sorted(d.items()) for L, d in sorted(out.items())}


def _check(weights, by_layer: dict[int, list[tuple[int, float]]]) -> None:
    m = weights.meta
    if m.n_expert:
        raise GneError("ARCH_UNSUPPORTED", "MoE の gain には対応していません(ffn_up_exps は対象外、§6.16)")
    for L, nk in by_layer.items():
        if not 0 <= L < m.n_layer:
            raise GneError("BAD_RECIPE", f"層 {L} は範囲外です")
        if not weights.has_tensor(f"blk.{L}.ffn_up.weight"):
            raise GneError("ARCH_UNSUPPORTED", f"blk.{L}.ffn_up.weight がありません(gate と up が 1 つのテンソルの arch は未対応)")
        for n, _ in nk:
            if not 0 <= n < m.n_ff_at(L):
                raise GneError("BAD_RECIPE", f"L{L} のニューロン {n} は範囲外です")


def lora_tensors(weights, by_layer: dict[int, list[tuple[int, float]]]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """層ごとの (A, B)。A は (r, n_embd)、B は (n_ff, r)、どちらも float32。"""
    _check(weights, by_layer)
    out = {}
    for L, nk in by_layer.items():
        W = weights.up_matrix(L)                                 # (n_ff, n_embd)
        A = np.stack([W[n] for n, _ in nk]).astype(np.float32)
        B = np.zeros((W.shape[0], len(nk)), dtype=np.float32)
        for i, (n, k) in enumerate(nk):
            B[n, i] = k - 1.0
        out[L] = (A, B)
    return out


def export_gain_lora(weights, recipe: Recipe, path: str | Path) -> dict:
    """gain ユニットを LoRA アダプタ GGUF に書く。"""
    by_layer = gain_by_layer(recipe.units)
    if not by_layer:
        raise GneError("BAD_RECIPE", "gain のユニットがありません(例 --unit 14:8949:x2)")
    m = weights.meta
    tens = lora_tensors(weights, by_layer)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(path), arch=m.arch)
    w.add_type("adapter")
    w.add_string("adapter.type", "lora")
    w.add_name(f"{m.name} gain" + (f" {recipe.name}" if recipe.name else ""))
    w.add_key_value("gne.recipe", json.dumps(recipe.to_dict(), ensure_ascii=False), GGUFValueType.STRING)
    w.add_key_value("gne.source_model_id", m.model_id, GGUFValueType.STRING)
    for L, (A, B) in tens.items():
        w.add_tensor(f"blk.{L}.ffn_up.weight.lora_a", A)
        w.add_tensor(f"blk.{L}.ffn_up.weight.lora_b", B)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return {"path": str(path), "layers": {str(L): [[n, k] for n, k in nk] for L, nk in by_layer.items()},
            "rank": {str(L): len(nk) for L, nk in by_layer.items()}, "bytes": Path(path).stat().st_size}


def bake_gain_gguf(weights, recipe: Recipe, dst: str | Path, *, requant: str = "q8_0") -> dict:
    """W_up[L] の行を k 倍した GGUF を作る。書き換えた ffn_up だけ Q8_0(requant="q8_0")か F16 にし、他は生バイトのまま写す。"""
    by_layer = gain_by_layer(recipe.units)
    if not by_layer:
        raise GneError("BAD_RECIPE", "gain のユニットがありません")
    _check(weights, by_layer)
    if requant not in ("q8_0", "f16"):
        raise GneError("BAD_ARGS", "--requant は q8_0 か f16 です")
    if Path(weights.path).resolve() == Path(dst).resolve():
        raise GneError("BAD_ARGS", "--in と --out が同じファイルです")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    tag = f" +gain:{recipe.name}" if recipe.name else " +gain"
    w = copy_gguf_header(weights, dst, tag=tag, recipe=recipe)
    plan, modified = [], []
    for t in weights.reader.tensors:
        hit = t.name.startswith("blk.") and t.name.endswith(".ffn_up.weight")
        L = int(t.name.split(".")[1]) if hit else None
        if L in by_layer:
            W = np.array(weights.up_matrix(L), dtype=np.float32, copy=True)   # F32 のテンソルは読み取り専用の memmap
            for n, k in by_layer[L]:
                W[n] *= k
            if requant == "f16":
                data, qt = W.astype(np.float16), GGMLQuantizationType.F16
            else:
                data, qt = gquants.quantize(W, GGMLQuantizationType.Q8_0), GGMLQuantizationType.Q8_0
            plan.append((t.name, data, qt))
            modified.append({"tensor": t.name, "from": t.tensor_type.name, "to": qt.name})
        else:
            plan.append((t.name, t.data, t.tensor_type))
    for name, data, qt in plan:
        w.add_tensor_info(name, data.shape, data.dtype, data.nbytes, raw_dtype=qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for _, data, _ in plan:
        w.write_tensor_data(data, tensor_endianess=weights.reader.endianess)
    w.close()
    return {"path": str(dst), "modified": modified, "n_tensors": len(plan), "bytes": Path(dst).stat().st_size}


def lora_cache_path(gain_dir: Path, recipe: Recipe) -> Path:
    """実行時に当てる LoRA の置き場所(gain ユニットの中身で決まる名前)。"""
    key = json.dumps(gain_by_layer(recipe.units), sort_keys=True)
    return Path(gain_dir) / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.lora.gguf"
