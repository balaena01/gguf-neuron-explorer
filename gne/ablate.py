"""方向の射影消去(directional ablation、本体設計 §6.17)。

残差に書き込む行列 W(出力 = W x、numpy 形状 (n_embd, in))について W' = (I − r rᵀ) W = W − r (rᵀ W)。
W の出力から r 成分が常に消える。入力に依らないので rank-1 LoRA で表せる(§6.16 の gain と同じ構図):

    A = rᵀ W (1 × in)、B = −r (n_embd × 1)  →  W x + B (A x) = W x − r (r · W x)

対象(残差ライター)は configs/arch/residual_writers.yaml。層ごとに attn_output / ssm_out / ffn_down のうち GGUF にあるもの。
埋め込み(token_embd)は LoRA では表せない(get_rows 経由)ので、LoRA 版では埋め込みの r 成分が残る。
したがって LoRA で消したあとの l_out-L の r 成分は、どの層でも埋め込みの r 成分と同じになる(V13 はこれを確かめる)。

この段では LoRA だけを作る(gne direction --select auto の層選びと V13)。gne ablate コマンド・bake・V14 は CHANGE-02 B。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml
from gguf import GGUFValueType, GGUFWriter

from .steer import GneError
from .store import CONFIGS
from .weights import tensor_to_f32


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
        warn.append("MoE の ffn_down_exps / ffn_down_shexp はまだ射影の対象外です(CHANGE-02 B)")
    if m.arch in t.get("post_norm", []) or m.has_post_ffn_norm:
        warn.append(f"{m.arch} は post-norm(ライターの出力が norm を通ってから残差に足される)ので、射影消去は近似です")
    if m.arch not in t.get("verified", []):
        warn.append(f"{m.arch} の残差ライターはグラフを読んで確かめていません(テンソル名で推定)")
    if biases:
        warn.append(f"ライターのバイアス {len(biases)} 個は LoRA では射影できません(r 成分が残る)")
    return {"verified": m.arch in t.get("verified", []), "post_norm": m.arch in t.get("post_norm", []),
            "counts": counts, "n_layers": len(ws), "biases": biases[:8], "warnings": warn,
            "checked_llama_cpp": t.get("checked_llama_cpp")}


def unit(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(r))
    if n == 0:
        raise GneError("BAD_DIRECTION", "方向の長さが 0 です")
    return (r / n).astype(np.float32)


def projection_rows(weights, R: np.ndarray, *, layers: list[int] | None = None,
                    progress: Callable[[int, int, str], None] | None = None) -> dict[str, np.ndarray]:
    """ライターごとに R @ W((k, in))。R は (k, n_embd) の単位ベクトルの行。k 本の方向をまとめて 1 回で計算する
    (ライターを 1 つずつ dequantize するので、層選びで候補が多くても重みを読むのは 1 回)。"""
    m = weights.meta
    if m.n_expert:
        raise GneError("ARCH_UNSUPPORTED", "MoE の射影消去にはまだ対応していません(CHANGE-02 B)")
    R = np.atleast_2d(np.asarray(R, dtype=np.float32))
    if R.shape[1] != m.n_embd:
        raise GneError("BAD_DIRECTION", f"方向の長さ {R.shape[1]} が n_embd={m.n_embd} と違います")
    ws = residual_writers(weights)
    Ls = sorted(ws) if layers is None else [L for L in layers if L in ws]
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


def write_ablation_lora(weights, rs: dict[str, np.ndarray], rows: dict[str, np.ndarray], path: str | Path, *,
                        meta: dict | None = None) -> dict:
    """rows[テンソル名] = rᵀW(in,)、rs[テンソル名] = そのテンソルで消す単位ベクトル r を LoRA GGUF に書く。"""
    m = weights.meta
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(path), arch=m.arch)
    w.add_type("adapter")
    w.add_string("adapter.type", "lora")
    w.add_name(f"{m.name} ablate" + (f" {meta['direction']}" if meta and meta.get("direction") else ""))
    w.add_key_value("gne.ablate", json.dumps(meta or {}, ensure_ascii=False), GGUFValueType.STRING)
    w.add_key_value("gne.source_model_id", m.model_id, GGUFValueType.STRING)
    for name in sorted(rows, key=lambda n: (int(n.split(".")[1]), n)):
        w.add_tensor(f"{name}.lora_a", np.asarray(rows[name], dtype=np.float32).reshape(1, -1))
        w.add_tensor(f"{name}.lora_b", (-np.asarray(rs[name], dtype=np.float32)).reshape(-1, 1))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return {"path": str(path), "n_tensors": len(rows), "bytes": Path(path).stat().st_size}


def export_ablation_lora(weights, r: np.ndarray | dict[int, np.ndarray], path: str | Path, *,
                         layers: list[int] | None = None, meta: dict | None = None, progress=None) -> dict:
    """方向 r(1 本を全層に。dict なら層ごとの r_L を各層のライターに = --per-layer)を消す LoRA を書く。"""
    ws = residual_writers(weights)
    Ls = sorted(ws) if layers is None else [L for L in layers if L in ws]
    if isinstance(r, dict):
        missing = [L for L in Ls if L not in r]
        if missing:
            raise GneError("BAD_DIRECTION", f"層 {missing[:5]} の方向がありません(--per-layer)")
        rows, rs = {}, {}
        for L in Ls:
            ru = unit(r[L])
            part = projection_rows(weights, ru[None, :], layers=[L])
            for name, v in part.items():
                rows[name], rs[name] = v[0], ru
            if progress:
                progress(Ls.index(L) + 1, len(Ls), "ablate: rᵀW")
    else:
        ru = unit(r)
        part = projection_rows(weights, ru[None, :], layers=Ls, progress=progress)
        rows = {n: v[0] for n, v in part.items()}
        rs = {n: ru for n in rows}
    return write_ablation_lora(weights, rs, rows, path, meta={**(meta or {}), "layers": Ls})


def vector_key(r: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(unit(r)).tobytes()).hexdigest()[:12]
