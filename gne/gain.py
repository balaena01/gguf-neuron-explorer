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
    """残差に書き込む重み行列の名前 → (A, B)。A は (r, n_in)、B は (n_embd, r)、r = ヘッド数 × head_dim。"""
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


def export_gain_lora(weights, recipe: Recipe, path: str | Path, *,
                     extra: dict[str, tuple[np.ndarray, np.ndarray]] | None = None, extra_meta: dict | None = None,
                     extra_kv: dict[str, str] | None = None, strip_meta: bool = False, f16: bool = False,
                     name: str | None = None) -> dict:
    """gain ユニット(ニューロンの ffn_up とヘッドの W_O)を LoRA アダプタ GGUF に書く。extra は同じファイルに入れる
    残差に書き込む重み行列の LoRA(ヘッドの方向成分の射影 CHANGE-06、鍵つきの書き込み CHANGE-07 の ffn_down・attn_output・
    ffn_down_shexp・ffn_down_exps。同じテンソルのヘッドの gain とは A・B を連結して 1 つにする)。
    extra_meta は gne.headproj の記録(従来どおり)、extra_kv はそのほかの gne.* の記録(名前 → JSON の文字列、例 gne.keyed)。
    proj_rank は extra_meta(proj)を渡したときだけ extra から数える(proj と keyed を混ぜたときは呼ぶ側が数える)。
    name はファイルの名前の欄(general.name)。省くと「<元のモデルの名前> gain <レシピの名前>」。
    strip_meta(人に渡す版)は gne.* の記録(レシピ・ヘッド・強さ・元のモデルの識別子)を書かず、名前の欄は name(省くと
    "adapter")にして、元のモデルの名前・レシピの名前を入れない(gne nw export と同じ扱い)。"""
    cast = (lambda x: np.asarray(x, dtype=np.float16)) if f16 else (lambda x: x)   # LoRA を F16 で書く(大きさが半分)
    from .headproj import merge_tensors
    from .nwrite import NEUTRAL_NAME
    if name is not None and not str(name).strip():
        raise GneError("BAD_ARGS", "名前(name)が空です")
    by_layer = gain_by_layer(recipe.units)
    heads = head_gain_by_layer(recipe.units)
    experts = expert_gain_by_layer(recipe.units)
    shared = shared_gain_by_layer(recipe.units)
    if not by_layer and not heads and not experts and not shared and not extra:
        raise GneError("BAD_RECIPE", "gain のユニットがありません(例 --unit 14:8949:x2、ヘッドは 21:H7:x0、専門家のニューロンは 12:E3:N5:x0)")
    m = weights.meta
    tens = lora_tensors(weights, by_layer) if by_layer else {}
    htens = merge_tensors(head_lora_tensors(weights, heads), extra or {})
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
    label = str(name).strip() if name is not None else (
        NEUTRAL_NAME if strip_meta else f"{m.name} gain" + (f" {recipe.name}" if recipe.name else ""))
    w.add_name(label)
    if not strip_meta:                                           # 人に渡す版は gne の記録(レシピ・ヘッド・強さ・識別子)を残さない
        w.add_key_value("gne.recipe", json.dumps(recipe.to_dict(), ensure_ascii=False), GGUFValueType.STRING)
        w.add_key_value("gne.source_model_id", m.model_id, GGUFValueType.STRING)
        if extra_meta is not None:
            w.add_key_value("gne.headproj", json.dumps(extra_meta, ensure_ascii=False), GGUFValueType.STRING)
        for k, v in (extra_kv or {}).items():
            w.add_key_value(k, v, GGUFValueType.STRING)
    for L, (A, B) in tens.items():
        w.add_tensor(f"blk.{L}.ffn_up.weight.lora_a", cast(A))
        w.add_tensor(f"blk.{L}.ffn_up.weight.lora_b", cast(B))
    for name, (A, B) in sorted(htens.items(), key=lambda x: int(x[0].split(".")[1])):
        w.add_tensor(f"{name}.lora_a", cast(A))
        w.add_tensor(f"{name}.lora_b", cast(B))
    for L, (A, B) in etens.items():                              # numpy (E, r, n_embd) → ne [n_embd, r, E]
        w.add_tensor(f"blk.{L}.ffn_up_exps.weight.lora_a", cast(A))
        w.add_tensor(f"blk.{L}.ffn_up_exps.weight.lora_b", cast(B))
    for L, (A, B) in stens.items():
        w.add_tensor(f"blk.{L}.ffn_up_shexp.weight.lora_a", cast(A))
        w.add_tensor(f"blk.{L}.ffn_up_shexp.weight.lora_b", cast(B))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    out = {"path": str(path), "general_name": label,
           "layers": {str(L): [[n, k] for n, k in nk] for L, nk in by_layer.items()},
           "rank": {str(L): len(nk) for L, nk in by_layer.items()}, "bytes": Path(path).stat().st_size}
    if heads:
        out["heads"] = {str(L): [[h, k] for h, k in hk] for L, hk in heads.items()}
        out["head_rank"] = {str(L): int(htens[weights.head_block(L, hk[0][0])[0]][0].shape[0]) for L, hk in heads.items()}
    if extra and extra_meta is not None:
        out["proj_rank"] = {name.split(".")[1]: int(B.shape[-1]) for name, (_, B) in extra.items()}
    if experts:
        out["experts"] = {str(L): [[e, n, k] for e, n, k in enk] for L, enk in experts.items()}
    if shared:
        out["shared"] = {str(L): [[n, k] for n, k in nk] for L, nk in shared.items()}
    return out


def _patch_targets(weights, recipe: Recipe) -> list[tuple[str, object, float, str]] | None:
    """スケールパッチで書ける操作の一覧 [(テンソル, 行 / 専門家, k か delta, 種類)]。ヘッドの gain(W_O の列)のように
    列方向の操作があるか、テンソルの型がパッチに対応していなければ None。"""
    from .patch import supported
    # 列方向(ヘッドの gain・方向成分の射影)と鍵つきの書き込み(rank 1 の足し算、CHANGE-07)はパッチで書けない
    if head_gain_by_layer(recipe.units) or recipe.proj_units or recipe.keyed_units:
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


def _writer_base(weights, name: str) -> np.ndarray:
    """焼き込みで変化量を足す重み行列の元の値(float32、numpy の形 (out, in)、専門家は (n_expert, out, in))。
    ヘッドが残差に書き込む重み行列(attn_output / ssm_out)は head_writer_matrix(形を確かめる)、それ以外(ffn_down・ffn_down_shexp・
    ffn_down_exps など、CHANGE-07 の keyed)はテンソルをそのまま float32 にする(レビュー P12)。"""
    from .weights import tensor_to_f32
    parts = name.split(".")
    if parts[0] == "blk" and len(parts) > 2 and parts[1].isdigit():
        L = int(parts[1])
        try:
            li = weights.arch.layers[L]
        except (IndexError, KeyError, AttributeError):
            li = None
        if li is not None and li.head_writer and li.head_dim and name == f"blk.{L}.{li.head_writer}.weight":
            return weights.head_writer_matrix(L)
    return tensor_to_f32(weights.tensor(name))


# --- keyed を含む焼き込みの既定の型(requant="auto"、CHANGE-07)--------------------------------------------------------
# 元が Q8_0 のテンソルは Q8_0 のまま(keep と同じ)書く。F16 にすると編集の丸めは小さいが、書き換えたテンソルの掛け算が
# Q8_0 の経路(活性も 8 ビットにする)から F16 の経路に変わり、LoRA 版との差が keep より大きくなった(verification.md F132)。
# ただし編集が Q8_0 の刻みに比べて小さく、量子化し直すと編集の 1 割以上が消える(edit_kept < KEYED_KEEP_MIN)テンソルだけ F16 に
# する(Qwen3-0.6B の L14 で、edit_kept は α = 1 で 0.99、0.25 で 0.93、0.1 で 0.84、0.05 で 0.75)。α ではなく edit_kept で
# 分けるのは、残り方が編集の要素と刻みの比(α‖r_L‖‖κ_L‖ と重みの大きさ)で決まり、α だけでは決まらないため。
# 元が Q8_0 でない型(K-quant・IQ・F16・BF16・F32)は今までどおり F16。専門家の 3 次元テンソル(ffn_down_exps)も同じ規則。
KEYED_KEEP_MIN = 0.9


def auto_qtype(src):
    """requant="auto"(keyed を含む焼き込みの既定)で書き出す型の第一候補: 元が Q8_0 なら Q8_0、それ以外は F16。"""
    return GGMLQuantizationType.Q8_0 if src == GGMLQuantizationType.Q8_0 else GGMLQuantizationType.F16


def edit_kept(back: np.ndarray, want: np.ndarray, dW: np.ndarray) -> float:
    """量子化し直しても残った編集の割合: 1 + ⟨戻した値 − 書きたい値, ΔW⟩ / ‖ΔW‖²(= ⟨実際に足された分, ΔW⟩ / ‖ΔW‖²)。
    1 なら編集がそのまま残り、0 なら丸めで消えた。requant_error(‖戻した値 − 書きたい値‖ / ‖ΔW‖)は編集と無関係な丸めの
    ゆらぎも数えるので、刻みより小さい編集では大きく出るが、編集の向きの成分はこちらで見る。"""
    d = np.asarray(dW, dtype=np.float64)
    den = float(np.sum(d * d))
    if den <= 0:
        return 1.0
    return 1.0 + float(np.sum((np.asarray(back, dtype=np.float64) - np.asarray(want, dtype=np.float64)) * d)) / den


def _decode_back(data: np.ndarray, qt, shape: tuple[int, ...]) -> np.ndarray:
    """_encode した値を float32 に戻す(量子化し直しで失った分を測る用)。"""
    if qt == GGMLQuantizationType.F32:
        return np.asarray(data, dtype=np.float32).reshape(shape)
    if qt == GGMLQuantizationType.F16:
        return np.asarray(data).astype(np.float32).reshape(shape)
    return np.asarray(gquants.dequantize(data, qt), dtype=np.float32).reshape(shape)

# --- 専門家の 3 次元テンソルの焼き込み(CHANGE-07 の keyed expert:e)-------------------------------------------------
# ffn_down_exps は (n_expert, out, in) で、大きなモデルでは 1 層で数億要素ある。テンソル全体を float32 にせず、
# 専門家(スラブ)ごとに読んで書く。編集するのは LoRA の A・B が 0 でない専門家のスラブだけで、ほかのスラブは
# 書き出す型が元と同じならバイトのまま写し、違えば 1 枚ずつ変換する。1 つのテンソルの型は 1 つなので、書き出す型を
# 元と違う型(F16 など)にすると編集しないスラブも含めてテンソル全体の大きさが変わる(modified の bytes_from・bytes_to)。

_ENCODABLE: dict = {}


def _can_encode(qt) -> bool:
    """gguf-py で量子化できる型か(F32・F16・BF16・Q4_0・Q4_1・Q5_0・Q5_1・Q8_0・MXFP4 など。K-quant・IQ は不可)。"""
    if qt not in _ENCODABLE:
        from gguf.constants import GGML_QUANT_SIZES
        try:
            bs = int(GGML_QUANT_SIZES[qt][0])
            gquants.quantize(np.zeros((1, 256 * max(bs, 1)), np.float32), qt)
            _ENCODABLE[qt] = True
        except Exception:                                    # NotImplementedError・KeyError など
            _ENCODABLE[qt] = False
    return _ENCODABLE[qt]


def _encode(W: np.ndarray, qt) -> np.ndarray:
    if qt == GGMLQuantizationType.F32:
        return np.asarray(W, dtype=np.float32)
    if qt == GGMLQuantizationType.F16:
        return np.asarray(W).astype(np.float16)
    return gquants.quantize(np.asarray(W, dtype=np.float32), qt)


def _expert_qtype(src, requant: str):
    """専門家の 3 次元テンソルを書き出す型。f16 / q8_0 はその型。keep は元の型で書けるなら元の型(編集しない専門家の
    スラブはバイトのまま)、書けない型(K-quant・IQ など)は 2 次元の keep と同じく Q8_0。auto(keyed を含む焼き込みの既定)は
    元が Q8_0 なら Q8_0、それ以外は F16(auto_qtype。Q8_0 で編集が埋もれるときの F16 への切り替えは _expert_bake)。"""
    if requant == "f16":
        return GGMLQuantizationType.F16
    if requant == "q8_0":
        return GGMLQuantizationType.Q8_0
    if requant == "auto":
        return auto_qtype(src)
    return src if _can_encode(src) else GGMLQuantizationType.Q8_0


class _SlabStream:
    """専門家の 3 次元テンソルを、書くときにスラブ(専門家)ごとに作るもの(GGUFWriter.write_tensor_data に配列の代わりに
    渡す: shape・dtype・nbytes・tofile)。編集したスラブは前もって作ったもの、ほかは元のバイトか 1 枚ずつ変換したもの。"""

    def __init__(self, t, qt, edited: dict[int, np.ndarray], slab_shape: tuple[int, ...], dtype):
        self.t, self.qt, self.edited = t, qt, edited
        self.n = int(t.data.shape[0])
        self.shape = (self.n, *slab_shape)
        self.dtype = np.dtype(dtype)
        self.nbytes = int(self.n * int(np.prod(slab_shape)) * self.dtype.itemsize)

    def byteswap(self, inplace: bool = False):
        raise GneError("INTERNAL", "専門家のテンソルの焼き込みはバイト順の入れ替えに対応していません")

    def tofile(self, fout) -> None:
        from .weights import _slab_to_f32
        same = self.qt == self.t.tensor_type
        for e in range(self.n):
            if e in self.edited:
                data = self.edited[e]
            elif same:
                data = np.ascontiguousarray(self.t.data[e])
            else:
                data = _encode(_slab_to_f32(self.t, e), self.qt)
            if data.nbytes * self.n != self.nbytes:
                raise GneError("INTERNAL", f"{self.t.name}: 専門家 {e} のスラブの大きさ {data.nbytes} が合いません")
            data.tofile(fout)


def expert_slab_edits(t, A: np.ndarray, B: np.ndarray, qt, *, encode=None, decode=None,
                      measure: bool = True, stats: dict | None = None) -> tuple[dict[int, np.ndarray], float | None]:
    """専門家の 3 次元テンソル t(numpy (n_expert, out, in))の、A (n_expert, r, in)・B (n_expert, out, r) が 0 でない専門家の
    スラブだけを float32 にして W_e + B_e A_e を作り、型 qt にしたもの {専門家: 書き出すスラブ}(1 枚ずつ。テンソル全体は
    float32 にしない)。通常の焼き込み(_expert_bake)と分割 GGUF への焼き込み(headproj.bake_proj_split)で共用する。
    encode(W, qt) / decode(data, qt, shape) を省くと gguf-py(_encode・_decode_back)。measure なら量子化し直しで失った分
    ‖戻した値 − 書きたい値‖ / ‖B_e A_e‖(編集したスラブの和)も返す(編集が無ければ None)。stats(dict)を渡すと、measure の
    とき stats["kept"] に編集の残り方(edit_kept、編集したスラブの和)を入れる。"""
    from .weights import _slab_to_f32
    encode = encode or _encode
    decode = decode or _decode_back
    n_exp = int(t.data.shape[0])
    if A.ndim != 3 or B.ndim != 3 or A.shape[0] != n_exp or B.shape[0] != n_exp:
        raise GneError("INTERNAL", f"{t.name}: 専門家の LoRA の形 {A.shape} / {B.shape} が専門家の数 {n_exp} と合いません")
    edited: dict[int, np.ndarray] = {}
    err2 = d2 = dot = 0.0
    for e in range(n_exp):
        if not (np.any(A[e]) and np.any(B[e])):
            continue
        W = _slab_to_f32(t, e)
        dW = B[e].astype(np.float64) @ A[e].astype(np.float64)
        if dW.shape != W.shape:
            raise GneError("INTERNAL", f"{t.name}: 専門家 {e} の変化量の形 {dW.shape} が元の行列 {W.shape} と違います")
        want = (W.astype(np.float64) + dW).astype(np.float32)
        edited[e] = encode(want, qt)
        if measure:
            back = decode(edited[e], qt, W.shape)
            diff = back.astype(np.float64) - want
            err2 += float(np.sum(np.square(diff)))
            d2 += float(np.sum(np.square(dW)))
            dot += float(np.sum(diff * dW))
    if stats is not None and measure and d2 > 0:
        stats["kept"] = 1.0 + dot / d2
    return edited, (float(np.sqrt(err2) / np.sqrt(d2)) if measure and d2 > 0 else None)


def expert_record(t, qt, edited, bytes_to: int) -> dict:
    """専門家の 3 次元テンソルの書き換えの記録(modified の 1 つ): 編集した専門家、編集しないスラブの扱い(raw = バイトの
    まま、"Q8_0->F16" = 1 枚ずつ変換)、元と後の大きさ。"""
    return {"tensor": t.name, "from": t.tensor_type.name, "to": qt.name, "experts": sorted(edited),
            "untouched": "raw" if qt == t.tensor_type else f"{t.tensor_type.name}->{qt.name}",
            "bytes_from": int(t.n_bytes), "bytes_to": int(bytes_to)}


def expert_bytes_added(records) -> int | None:
    """modified の記録のうち専門家の 3 次元テンソルの大きさの増分の和(専門家のテンソルが無ければ None)。"""
    recs = [m for m in records or [] if isinstance(m, dict) and "experts" in m and "bytes_to" in m and "bytes_from" in m]
    return sum(int(m["bytes_to"]) - int(m["bytes_from"]) for m in recs) if recs else None


def _expert_bake(t, A: np.ndarray, B: np.ndarray, requant: str, via_f16: bool) -> tuple[_SlabStream, object, dict]:
    """専門家の 3 次元テンソル t(numpy (n_expert, out, in))に W_e += B_e A_e を焼き込む書き出しの準備。A (n_expert, r, in)・
    B (n_expert, out, r) が 0 でない専門家のスラブだけ float32 にして足し、書き出す型にしておく(1 枚ずつ、テンソル全体は
    float32 にしない、expert_slab_edits)。戻り値 (ストリーム, 書き出す型, modified の記録)。記録の requant_error は編集した
    スラブでの ‖戻した値 − 書きたい値‖ / ‖B_e A_e‖(全専門家の和)、untouched は編集しないスラブの扱い(raw = バイトのまま)。"""
    from .weights import _slab_to_f32
    qt = _expert_qtype(t.tensor_type, requant)
    stats: dict = {}
    edited, err = expert_slab_edits(t, A, B, qt, measure=not via_f16 or requant == "auto", stats=stats)
    fallback = None
    if requant == "auto" and qt == GGMLQuantizationType.Q8_0 and stats.get("kept", 1.0) < KEYED_KEEP_MIN:
        fallback = {"from": "Q8_0", "edit_kept": stats["kept"]}            # Q8_0 では編集が埋もれる: F16 で書き直す
        qt, stats = GGMLQuantizationType.F16, {}
        edited, err = expert_slab_edits(t, A, B, qt, measure=not via_f16, stats=stats)
    probe = next(iter(edited.values())) if edited else _encode(_slab_to_f32(t, 0), qt)
    st = _SlabStream(t, qt, edited, probe.shape, probe.dtype)
    rec = expert_record(t, qt, edited, st.nbytes)
    if err is not None and not via_f16:
        rec["requant_error"] = err
    if "kept" in stats and not via_f16:
        rec["edit_kept"] = stats["kept"]
    if fallback is not None:
        rec["fallback"] = fallback
    return st, qt, rec


def bake_gain_gguf(weights, recipe: Recipe, dst: str | Path, *, requant: str = "q8_0",
                   extra: dict[str, tuple[np.ndarray, np.ndarray]] | None = None, extra_meta: dict | None = None,
                   extra_kv: dict[str, str] | None = None, via_f16: bool = False, strip_meta: bool = False,
                   keyed: bool = False) -> dict:
    """W_up[L] の行(ニューロン)と W_O の列ブロック(ヘッド)を k 倍した GGUF を作る。書き換えたテンソルだけ
    Q8_0(requant="q8_0")か F16 にし、他は生バイトのまま写す。requant="patch" はスケールパッチ(量子化を保ったまま
    ちょうど k 倍、bake_gain_patch)、"auto" はパッチで書ければパッチ、書けなければ Q8_0。専門家全体の gain と route は
    パッチだけ。extra は残差に書き込む重み行列に足す LoRA の変化量 B @ A(ヘッドの方向成分の射影 CHANGE-06、鍵つきの書き込み CHANGE-07)。
    元の行列は _writer_base で引く(ヘッドの行列に限らない)。専門家の 3 次元 (E, out, in) は専門家ごとに W_e += B_e A_e を
    スラブごとに書く(_expert_bake。テンソル全体を float32 にしない。requant="keep" なら元の型で書ける型のとき元の型にして、
    編集しない専門家のスラブはバイトのまま)。modified には書き換えたテンソルの元と後の大きさ(bytes_from・bytes_to)も入れ、
    専門家のテンソルがあれば大きさの増分の和を expert_bytes_added に返す。
    gain と同じテンソルなら、どちらも元の重みから計算した変化量を足す(LoRA を並べて当てたのと同じ)。
    extra のテンソルは量子化し直しで失った分 ‖戻した値 − 書きたい値‖ / ‖B @ A‖ を modified の requant_error と
    requant_error_max に返す(小さな rank 1 の編集が量子化の刻みに埋もれていないかの目安)。
    extra_meta は gne.headproj、extra_kv はそのほかの gne.* の記録(例 gne.keyed)。
    keyed(鍵つきの書き込みを含む)で requant="auto" なら、テンソルごとに型を決める(auto_qtype: 元が Q8_0 なら Q8_0 のまま、
    それ以外は F16。Q8_0 のままでは編集の 1 割以上が丸めに消える(edit_kept < KEYED_KEEP_MIN)テンソルだけ F16、記録の
    fallback)。変化量を足したテンソルの記録には、量子化し直しても残った編集の割合 edit_kept も入れる(edit_kept_min)。"""
    if requant in ("patch", "auto"):
        if requant == "patch" or (_patch_targets(weights, recipe) is not None and not extra):
            return bake_gain_patch(weights, recipe, dst)
        requant = "auto" if (requant == "auto" and keyed) else "q8_0"
    extra = extra or {}
    if any(u.kind == "expert" and u.mode in ("gain", "route") for u in recipe.units):
        raise GneError("BAD_RECIPE", "専門家全体の gain と route はスケールパッチでだけ書けます", "--requant patch(か auto)")
    by_layer = gain_by_layer(recipe.units)
    heads = head_gain_by_layer(recipe.units)
    if not by_layer and not heads and not extra:
        raise GneError("BAD_RECIPE", "gain のユニットがありません")
    if by_layer:
        _check(weights, by_layer)
    head_tensors = {weights.head_block(L, hk[0][0])[0]: L for L, hk in heads.items()}
    missing = [n for n in extra if not weights.has_tensor(n)]
    if missing:
        raise GneError("BAD_ARGS", f"焼き込む先に {missing[0]} がありません(構造の違うモデル)")
    targets = set(head_tensors) | set(extra)
    if requant not in ("q8_0", "f16", "keep", "auto"):
        raise GneError("BAD_ARGS", "--requant は q8_0・f16・keep(元が F32 / F16 / BF16 ならその型、量子化された型なら Q8_0)・"
                                   "auto(keyed を含むときの既定)です")
    if Path(weights.path).resolve() == Path(dst).resolve():
        raise GneError("BAD_ARGS", "--in と --out が同じファイルです")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    tag = f" +gain:{recipe.name}" if recipe.name else " +gain"
    w = copy_gguf_header(weights, dst, tag=tag, recipe=recipe, strip=strip_meta)
    if not strip_meta:
        if extra_meta is not None:
            w.add_key_value("gne.headproj", json.dumps(extra_meta, ensure_ascii=False), GGUFValueType.STRING)
        for k, v in (extra_kv or {}).items():
            w.add_key_value(k, v, GGUFValueType.STRING)
    plan, modified = [], []
    for t in weights.reader.tensors:
        hit = t.name.startswith("blk.") and t.name.endswith(".ffn_up.weight")
        L = int(t.name.split(".")[1]) if hit else None
        W, dW = None, None
        if L in by_layer:
            W = np.array(weights.up_matrix(L), dtype=np.float32, copy=True)   # F32 のテンソルは読み取り専用の memmap
            for n, k in by_layer[L]:
                W[n] *= k
        elif t.name in extra and np.ndim(extra[t.name][0]) == 3 and t.name not in head_tensors:
            st, qt, rec = _expert_bake(t, *extra[t.name], requant, via_f16)    # 専門家の 3 次元: スラブごと
            plan.append((t.name, st, qt))
            modified.append(rec)
            continue
        elif t.name in targets:
            W = np.array(_writer_base(weights, t.name), dtype=np.float32, copy=True)
            hl = head_tensors.get(t.name)
            for h, k in (heads.get(hl, []) if hl is not None else []):
                _, lo, hi = weights.head_block(hl, h)
                W[:, lo:hi] *= k
            if t.name in extra:                                   # W + B @ A(proj は元の W から作った r̂ᵀW、keyed は α d κᵀ)
                A, B = extra[t.name]
                dW = np.matmul(B.astype(np.float64), A.astype(np.float64))          # 専門家の 3 次元は専門家ごとの積
                if dW.shape != W.shape:
                    raise GneError("INTERNAL", f"{t.name}: 変化量の形 {dW.shape} が元の行列 {W.shape} と違います")
                W += dW.astype(np.float32)
        if W is not None:
            fallback = None
            if requant == "keep" and t.tensor_type == GGMLQuantizationType.F32:      # 量子化していないモデル: 元の型のまま
                data, qt = W.astype(np.float32), GGMLQuantizationType.F32
            elif requant == "keep" and t.tensor_type == GGMLQuantizationType.F16:
                data, qt = W.astype(np.float16), GGMLQuantizationType.F16
            elif requant == "keep" and t.tensor_type == GGMLQuantizationType.BF16:
                data, qt = gquants.quantize(W, GGMLQuantizationType.BF16), GGMLQuantizationType.BF16
            elif requant == "f16" or (requant == "auto" and auto_qtype(t.tensor_type) == GGMLQuantizationType.F16):
                data, qt = W.astype(np.float16), GGMLQuantizationType.F16
            else:
                data, qt = gquants.quantize(W, GGMLQuantizationType.Q8_0), GGMLQuantizationType.Q8_0
                if requant == "auto" and dW is not None:            # Q8_0 のままで編集が埋もれるなら F16(KEYED_KEEP_MIN)
                    kept = edit_kept(_decode_back(data, qt, W.shape), W, dW)
                    if kept < KEYED_KEEP_MIN:
                        fallback = {"from": "Q8_0", "edit_kept": kept}
                        data, qt = W.astype(np.float16), GGMLQuantizationType.F16
            plan.append((t.name, data, qt))
            rec = {"tensor": t.name, "from": t.tensor_type.name, "to": qt.name, "bytes_from": int(t.n_bytes),
                   "bytes_to": int(data.nbytes)}
            if dW is not None and not via_f16:                     # 量子化し直しで失った分 ÷ 編集の大きさ、残った編集の割合
                back = _decode_back(data, qt, W.shape)
                rec["requant_error"] = float(np.linalg.norm(back - W) / max(float(np.linalg.norm(dW)), 1e-12))
                rec["edit_kept"] = edit_kept(back, W, dW)
            if fallback is not None:
                rec["fallback"] = fallback
            modified.append(rec)
        else:
            plan.append((t.name, t.data, t.tensor_type))
    if extra and not strip_meta:                                  # CHANGE-06 E.2: 書き換えたテンソルと元の型
        rec = [{**m, "op": "rewrite"} if not via_f16 else {**m, "to": m["from"], "via": "f16+llama-quantize", "op": "rewrite"}
               for m in modified]                         # via_f16: 中間は F16、llama-quantize で元の型に戻る
        w.add_key_value("gne.patches", json.dumps(rec, ensure_ascii=False), GGUFValueType.STRING)
    for name, data, qt in plan:
        w.add_tensor_info(name, data.shape, data.dtype, data.nbytes, raw_dtype=qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for _, data, _ in plan:
        w.write_tensor_data(data, tensor_endianess=weights.reader.endianess)
    w.close()
    out = {"path": str(dst), "modified": modified, "n_tensors": len(plan), "bytes": Path(dst).stat().st_size}
    errs = [m["requant_error"] for m in modified if "requant_error" in m]
    if errs:
        out["requant_error_max"] = max(errs)
    kept = [m["edit_kept"] for m in modified if "edit_kept" in m]
    if kept:
        out["edit_kept_min"] = min(kept)
    grow = expert_bytes_added(modified)                       # 専門家の 3 次元テンソルを書き換えたときだけ
    if grow is not None:
        out["expert_bytes_added"] = grow
    return out


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
