"""ゲイン(倍率)ステアリングと消去(設計書 §6.16)。

a_n = silu(g_n) · u_n、u_n = W_up[n, :] · x なので、W_up の n 行目を k 倍すると活性 a_n が k 倍になる
(silu は gate 側なので影響しない)。残差に加わるのは k · a_n · W_down[:, n]。k = 0 で消去、k < 0 で反転。
足し算(add、§2)と違い、入力に応じて働く量が変わるので「その状況でだけ強く出る/出なくなる」が作れる。

出力の 2 方式:
- (a) LoRA アダプタ GGUF(既定): rank-r。A の i 行目 = W_up[n_i, :](dequantize 済み)、B[n_i, i] = k_i − 1。
  W x + B(A x) = W x + Σ_i (k_i − 1)·u_{n_i}·e_{n_i}。llama.cpp の --lora でそのまま使える。
- (b) 書き換え GGUF: W_up[L] の n 行目を k 倍し、そのテンソルだけ Q8_0(または F16)で書き出す。他は生バイトのまま。

attention ヘッドの gain(CHANGE-04 D): ヘッド h の出力は W_O の列ブロック blk(head_dim 列)を通って残差に入るので、
その列ブロックを k 倍するとヘッド h の寄与だけが k 倍になる(Qwen3.5 は出力ゲートの後、線形注意の層は ssm_out)。
LoRA は B = (k − 1)·W_O[:, blk](n_embd, head_dim)、A = E_blkᵀ(head_dim, n_in、列ブロックを選ぶ単位行列の断片)。
同じ層の複数ヘッドは A と B を連結する(rank = ヘッド数 × head_dim)。

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
        if u.mode != "gain" or u.kind != "neuron":
            continue
        out[u.layer][u.neuron] = out[u.layer].get(u.neuron, 1.0) * float(u.k)
    return {L: sorted(d.items()) for L, d in sorted(out.items())}


def expert_gain_by_layer(units: list[Unit]) -> dict[int, list[tuple[int, int, float]]]:
    """専門家のニューロンの gain を層ごとに [(専門家, ニューロン, k)] にまとめる(CHANGE-05 §C)。"""
    out: dict[int, dict[tuple[int, int], float]] = defaultdict(dict)
    for u in units:
        if u.mode == "gain" and u.kind == "expert_neuron":
            key = (u.expert, u.neuron)
            out[u.layer][key] = out[u.layer].get(key, 1.0) * float(u.k)
    return {L: [(e, n, k) for (e, n), k in sorted(d.items())] for L, d in sorted(out.items())}


def shared_gain_by_layer(units: list[Unit]) -> dict[int, list[tuple[int, float]]]:
    out: dict[int, dict[int, float]] = defaultdict(dict)
    for u in units:
        if u.mode == "gain" and u.kind == "shared":
            out[u.layer][u.neuron] = out[u.layer].get(u.neuron, 1.0) * float(u.k)
    return {L: sorted(d.items()) for L, d in sorted(out.items())}


def expert_lora_tensors(weights, by_layer: dict[int, list[tuple[int, int, float]]]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """専門家ごとに積んだ LoRA。A (n_expert, r, n_embd)、B (n_expert, n_ff_exp, r)。i 番目のユニット(専門家 e、ニューロン n)は
    A[e, i] = W_up_e[n, :]、B[e, n, i] = k − 1 で、ほかの専門家のスライスは 0。llama.cpp は選ばれた専門家の番号で A・B を選ぶので
    (build_lora_mm_id、verification.md §17 の 0 項 4)、専門家 e が選ばれたトークンでだけ a_{e,n} が k 倍になる。"""
    out = {}
    for L, enk in by_layer.items():
        li = weights.moe_layer(L)
        A = np.zeros((li.n_expert, len(enk), weights.meta.n_embd), dtype=np.float32)
        B = np.zeros((li.n_expert, li.n_ff, len(enk)), dtype=np.float32)
        for i, (e, n, k) in enumerate(enk):
            A[e, i] = weights.expert_up(L, e)[n]
            B[e, n, i] = k - 1.0
        out[L] = (A, B)
    return out


def head_gain_by_layer(units: list[Unit]) -> dict[int, list[tuple[int, float]]]:
    """ヘッドの gain を層ごとに [(ヘッド, k)] にまとめる。同じヘッドが重なれば倍率を掛け合わせる。"""
    out: dict[int, dict[int, float]] = defaultdict(dict)
    for u in units:
        if u.mode == "gain" and u.kind == "head":
            out[u.layer][u.head] = out[u.layer].get(u.head, 1.0) * float(u.k)
    return {L: sorted(d.items()) for L, d in sorted(out.items())}


def head_lora_tensors(weights, by_layer: dict[int, list[tuple[int, float]]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """書き込み行列の名前 → (A, B)。A は (r, n_in)、B は (n_embd, r)、r = ヘッド数 × head_dim。"""
    out = {}
    for L, hk in by_layer.items():
        name, _, _ = weights.head_block(L, hk[0][0])
        W = weights.head_writer_matrix(L)                        # (n_embd, n_head·head_dim)
        As, Bs = [], []
        for h, k in hk:
            _, lo, hi = weights.head_block(L, h)
            A = np.zeros((hi - lo, W.shape[1]), dtype=np.float32)
            A[np.arange(hi - lo), np.arange(lo, hi)] = 1.0
            As.append(A)
            Bs.append(((k - 1.0) * W[:, lo:hi]).astype(np.float32))
        out[name] = (np.concatenate(As, axis=0), np.concatenate(Bs, axis=1))
    return out


def _check(weights, by_layer: dict[int, list[tuple[int, float]]]) -> None:
    m = weights.meta
    for L, nk in by_layer.items():
        if not 0 <= L < m.n_layer:
            raise GneError("BAD_RECIPE", f"層 {L} は範囲外です")
        if m.n_expert and weights.is_moe(L):
            raise GneError("BAD_RECIPE", f"層 {L} は MoE です。専門家のニューロンの gain は {L}:E<専門家>:N<n>:xk(CHANGE-05)")
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
    """gain ユニット(ニューロンの ffn_up とヘッドの W_O)を LoRA アダプタ GGUF に書く。"""
    by_layer = gain_by_layer(recipe.units)
    heads = head_gain_by_layer(recipe.units)
    experts = expert_gain_by_layer(recipe.units)
    shared = shared_gain_by_layer(recipe.units)
    if not by_layer and not heads and not experts and not shared:
        raise GneError("BAD_RECIPE", "gain のユニットがありません(例 --unit 14:8949:x2、ヘッドは 21:H7:x0、専門家のニューロンは 12:E3:N5:x0)")
    m = weights.meta
    tens = lora_tensors(weights, by_layer) if by_layer else {}
    htens = head_lora_tensors(weights, heads)
    etens = expert_lora_tensors(weights, experts)
    stens = {}
    for L, nk in shared.items():                                 # shared expert は dense と同じ 2D
        W = weights.shexp_up(L)
        A = np.stack([W[n] for n, _ in nk]).astype(np.float32)
        Bm = np.zeros((W.shape[0], len(nk)), dtype=np.float32)
        for i, (n, k) in enumerate(nk):
            Bm[n, i] = k - 1.0
        stens[L] = (A, Bm)
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
    for name, (A, B) in sorted(htens.items(), key=lambda x: int(x[0].split(".")[1])):
        w.add_tensor(f"{name}.lora_a", A)
        w.add_tensor(f"{name}.lora_b", B)
    for L, (A, B) in etens.items():                              # numpy (E, r, n_embd) → ne [n_embd, r, E]
        w.add_tensor(f"blk.{L}.ffn_up_exps.weight.lora_a", A)
        w.add_tensor(f"blk.{L}.ffn_up_exps.weight.lora_b", B)
    for L, (A, B) in stens.items():
        w.add_tensor(f"blk.{L}.ffn_up_shexp.weight.lora_a", A)
        w.add_tensor(f"blk.{L}.ffn_up_shexp.weight.lora_b", B)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    out = {"path": str(path), "layers": {str(L): [[n, k] for n, k in nk] for L, nk in by_layer.items()},
           "rank": {str(L): len(nk) for L, nk in by_layer.items()}, "bytes": Path(path).stat().st_size}
    if heads:
        out["heads"] = {str(L): [[h, k] for h, k in hk] for L, hk in heads.items()}
        out["head_rank"] = {str(L): int(htens[weights.head_block(L, hk[0][0])[0]][0].shape[0]) for L, hk in heads.items()}
    if experts:
        out["experts"] = {str(L): [[e, n, k] for e, n, k in enk] for L, enk in experts.items()}
    if shared:
        out["shared"] = {str(L): [[n, k] for n, k in nk] for L, nk in shared.items()}
    return out


def _patch_targets(weights, recipe: Recipe) -> list[tuple[str, object, float, str]] | None:
    """スケールパッチで書ける操作の一覧 [(テンソル, 行 / 専門家, k か delta, 種類)]。ヘッドの gain(W_O の列)のように
    列方向の操作があるか、テンソルの型がパッチに対応していなければ None。"""
    from .patch import supported
    if head_gain_by_layer(recipe.units):
        return None
    out = []
    for L, nk in gain_by_layer(recipe.units).items():
        out += [(f"blk.{L}.ffn_up.weight", n, k, "row") for n, k in nk]
    for L, enk in expert_gain_by_layer(recipe.units).items():
        out += [(f"blk.{L}.ffn_up_exps.weight", (e, n), k, "row") for e, n, k in enk]
    for L, nk in shared_gain_by_layer(recipe.units).items():
        out += [(f"blk.{L}.ffn_up_shexp.weight", n, k, "row") for n, k in nk]
    for u in recipe.units:
        if u.kind == "expert" and u.mode == "gain":
            out.append((f"blk.{u.layer}.ffn_down_exps.weight", u.expert, float(u.k), "slab"))
        elif u.kind == "expert" and u.mode == "route":
            name = weights.router_bias_name(u.layer)
            if name is None:
                raise GneError("ARCH_UNSUPPORTED", f"層 {u.layer} にはルーターのバイアスがありません")
            out.append((name, u.expert, float(u.dose), "add"))
    for name, _, _, kind in out:
        if not weights.has_tensor(name) or (kind != "add" and not supported(weights.tensor(name).tensor_type)):
            return None
    return out or None


def bake_gain_patch(weights, recipe: Recipe, dst: str | Path) -> dict:
    """スケールパッチで書いた GGUF(CHANGE-05 §D): 元をコピーし、行(ニューロン)・専門家のスラブ・ルーターのバイアスの
    バイトだけを書き換える。量子化を保ったまま値がちょうど k 倍になる。メタデータは書き換えない(オフセットが動くため)ので、
    当てた操作はサイドカー(<dst>.patches.json と <dst>.patch.json)に残す。"""
    import shutil
    import subprocess

    from .patch import Patcher, PatchSet
    targets = _patch_targets(weights, recipe)
    if targets is None:
        raise GneError("UNSUPPORTED_QUANT", "このレシピはスケールパッチで書けません(ヘッドの gain や対応していない量子化型)",
                       "--requant q8_0 か f16 で書き換え GGUF にしてください")
    if Path(weights.path).resolve() == Path(dst).resolve():
        raise GneError("BAD_ARGS", "--in と --out が同じファイルです")
    pt = Patcher(weights)
    ps = PatchSet("gain" + (f"-{recipe.name}" if recipe.name else ""), weights.meta.model_id)
    for name, at, k, kind in targets:
        ps.extend(pt.scale_row(name, at, k) if kind == "row" else pt.scale_slab(name, at, k) if kind == "slab"
                  else pt.add_element(name, at, k))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["cp", "--reflink=auto", str(weights.path), str(dst)], check=True)
    except (OSError, subprocess.CalledProcessError):
        shutil.copyfile(weights.path, dst)
    Path(str(dst) + ".patches.json").unlink(missing_ok=True)
    ps.save(Path(str(dst) + ".patch.json"))
    ps.apply(dst)
    seen = {}
    for name, _, _, _ in targets:
        t = weights.tensor(name)
        seen[name] = {"tensor": name, "from": t.tensor_type.name, "to": t.tensor_type.name, "via": "scale_patch"}
    return {"path": str(dst), "modified": list(seen.values()), "n_tensors": len(weights.gguf.tensors),
            "bytes": Path(dst).stat().st_size, "method": "patch", "patch_bytes": ps.n_bytes, "ops": ps.ops}


def bake_gain_gguf(weights, recipe: Recipe, dst: str | Path, *, requant: str = "q8_0") -> dict:
    """W_up[L] の行(ニューロン)と W_O の列ブロック(ヘッド)を k 倍した GGUF を作る。書き換えたテンソルだけ
    Q8_0(requant="q8_0")か F16 にし、他は生バイトのまま写す。requant="patch" はスケールパッチ(量子化を保ったまま
    ちょうど k 倍、bake_gain_patch)、"auto" はパッチで書ければパッチ、書けなければ Q8_0。専門家全体の gain と route は
    パッチだけ。"""
    if requant in ("patch", "auto"):
        if requant == "patch" or _patch_targets(weights, recipe) is not None:
            return bake_gain_patch(weights, recipe, dst)
        requant = "q8_0"
    if any(u.kind == "expert" for u in recipe.units):
        raise GneError("BAD_RECIPE", "専門家全体の gain と route はスケールパッチでだけ書けます", "--requant patch(か auto)")
    by_layer = gain_by_layer(recipe.units)
    heads = head_gain_by_layer(recipe.units)
    if not by_layer and not heads:
        raise GneError("BAD_RECIPE", "gain のユニットがありません")
    if by_layer:
        _check(weights, by_layer)
    head_tensors = {weights.head_block(L, hk[0][0])[0]: L for L, hk in heads.items()}
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
        W = None
        if L in by_layer:
            W = np.array(weights.up_matrix(L), dtype=np.float32, copy=True)   # F32 のテンソルは読み取り専用の memmap
            for n, k in by_layer[L]:
                W[n] *= k
        elif t.name in head_tensors:
            L = head_tensors[t.name]
            W = np.array(weights.head_writer_matrix(L), dtype=np.float32, copy=True)
            for h, k in heads[L]:
                _, lo, hi = weights.head_block(L, h)
                W[:, lo:hi] *= k
        if W is not None:
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
    heads = head_gain_by_layer(recipe.units)
    if heads:                                    # ヘッドが無ければ前と同じ名前(作ったものを使い回す)
        key += "|heads:" + json.dumps(heads, sort_keys=True)
    for tag, d in (("experts", expert_gain_by_layer(recipe.units)), ("shared", shared_gain_by_layer(recipe.units))):
        if d:
            key += f"|{tag}:" + json.dumps(d, sort_keys=True)
    return Path(gain_dir) / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.lora.gguf"
