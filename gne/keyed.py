"""活性の形を鍵にした条件つき書き込み(keyed write、CHANGE-07)の数式・鍵のファイル・LoRA の行列。

書き込み行列 W(numpy (out, in)、out = n_embd)にランク 1 を足す:

    ΔW = α · d · κᵀ,   κ = k / (μᵀk),   k = μ − Q Qᵀ μ

- μ(in): 効かせたい状態(側 A)で W に入る活性の平均。Q(in × m): 触りたくない状態(側 B)の活性が張る正規直交基底。
  k は μ のうち側 B の部分空間に直交する成分で、μᵀk = ‖k‖² なので κ = k / ‖k‖²。κᵀμ = 1、側 B の部分空間では κᵀx = 0。
- d(out): 書き込む向き。α は raw: d = r_L(方向セットの生の平均差 mean_A − mean_B、`L:dir:NAME:dose` を dose_unit raw で
  使うときと同じベクトル)。α = 1 で「観測した A − B の差をそのまま書く」。レシピの dose_unit とは独立。
- W x + ΔW x = W x + α (κᵀx) d。x = μ のとき α d を足し、側 B の部分空間の x には何も足さない。ΔW は W に依らない
  (W が要るのは焼き込みだけ)。LoRA は A = κᵀ(1 × in)、B = α d(out × 1)。同じ (層, 書き込み先, 鍵) のユニットは
  B を足して 1 本に畳み(方向や符号が違っても (Σ α_i d_i) κᵀ で階数 1)、鍵が違えば行を積む。

書き込み先(鍵 1 つは 1 種類に対応し、npz の target に記録する):

| target   | テンソル                    | 入力 x(in)                                  |
|----------|-----------------------------|---------------------------------------------|
| ffn      | blk.L.ffn_down.weight       | ffn_down の入力 = ニューロン活性(n_ff)     |
| attn     | blk.L.attn_output.weight / ssm_out.weight | ヘッド出力の連結(n_head·head_dim、CHANGE-06 の head_out) |
| shared   | blk.L.ffn_down_shexp.weight | shared expert のニューロン活性              |
| expert:e | blk.L.ffn_down_exps.weight の専門家 e | 専門家 e が選ばれたときの活性(n_ff_exp) |

expert:e の LoRA は 3 次元(lora_a numpy (n_expert, r, in)、lora_b (n_expert, out, r)、専門家 e 以外のスライスは 0)。
llama.cpp の build_lora_mm_id は選ばれた専門家の番号で A・B を選ぶ(verification.md §17 の 4)。

補正(レビュー P6・P7):
- post-FFN norm(Gemma・LFM3 など): 残差に入るのは γ ⊙ y / rms(y)。鍵は補正しない生の x で作り、書く向きを
  d_pre = rms̄_A · (r ⊘ γ) にする(rms̄_A は発火位置での rms(W x) の平均、鍵の作成時に保存。|γ| の小さい成分は下限で切る)。
  rms(y + Δy) の変化を無視した一次近似。
- 書き込み行列の .scale(BitNet・NVFP4 の scale2): llama.cpp のグラフで LoRA の出力にも s が掛かる道(build_ffn の
  down_s、GLM4 系の wo_s)だけ B を s で割る。build_lora_mm(w, x, w_s) の道(bitnet の ffn_down、ふつうの wo)は
  W x に s を掛けてから LoRA を足すので割らない(llama.cpp の src/llama-graph.cpp と src/models/bitnet.cpp を読んで確かめた)。
- 専門家の出力に掛かるルーターの重み・shared expert のゲート: 鍵に gate_a(発火位置でのゲートの平均)があれば B を割る。

鍵の作成(活性の捕捉)と確認は gne/keyed_build.py。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .steer import GneError

FORMAT = 1                       # 鍵の npz の形式の版
KN_WARN = 0.05                   # kn = ‖k‖/‖μ‖ がこれより小さい層は警告(側 B の漏れが 1/kn 倍に増える。レビュー P4)
KN_MIN = 1e-6                    # これより小さい層は鍵にならない(側 B が μ をほぼ張っている)
GAMMA_FLOOR = 0.1                # post-norm の γ で割るとき、|γ| を median(|γ|) のこの割合で下から切る
KINDS = ("ffn", "attn", "shared", "expert")
_TARGET_RE = re.compile(r"^(ffn|attn|shared|expert:(\d+))$")
_PSEUDO_RE = re.compile(r"^(.+)~(\d+)$")
_NAME_RE = re.compile(r"^[\w.\-]+$")
# ffn_down.scale と LoRA の掛かり方(llama.cpp の src/models/<arch>.cpp を読んで確かめた arch だけ。他は ARCH_UNSUPPORTED):
# bitnet は build_lora_mm(ffn_down, x, ffn_down_s) で W x に s を掛けてから LoRA を足す(LoRA の出力に s は掛からない)。
# llama・qwen3・qwen35 は build_ffn に ffn_down_s を渡し、LoRA を足した後に s を掛ける(LoRA の出力にも掛かる)
_FFN_SCALE_BEFORE_LORA = {"bitnet"}
_FFN_SCALE_AFTER_LORA = {"llama", "qwen3", "qwen35"}
# attention の出力で、build_lora_mm(wo, x) の後に wo_s を掛ける arch(LoRA の出力にも s が掛かる)
_ATTN_SCALE_AFTER_LORA = {"glm4", "glm4moe", "jais2"}
# 線形注意の ssm_out を build_lora_mm(ssm_out, x, ssm_out_s) で読む arch(W x に s を掛けてから LoRA を足すので、
# LoRA の出力に s は掛からない。src/models/qwen35.cpp・qwen35moe.cpp・qwen4exp.cpp の build_layer_attn_linear)
_SSM_OUT_SCALE_BEFORE_LORA = {"qwen35", "qwen35moe", "qwen4exp"}
# attention の出力(LoRA の差分を含む)に、書き込み行列の後で arch の倍率を掛ける arch(src/models/mimo2.cpp の v_scale)。
# keyed の大きさがずれるので attn の keyed は断る
_ATTN_OUT_RESCALED = {"mimo2"}
# ルーターの重みを専門家の入力に掛ける arch(出力には掛からない、llama-graph.cpp の weight_before_ffn)
_ROUTER_BEFORE_FFN = {"llama4"}
# 書き込み行列のある枝(attention・FFN・MoE)の出力に、残差に足す前に f_residual_scale を掛ける arch。LoRA の差分にも
# 掛かるので B を割る(src/models/granite.cpp の build_layer_ffn。granitemoe・minicpm は同じグラフ、granite-hybrid.cpp・
# granite-swa.cpp も同じ形)。GGUF に {arch}.residual_scale が無ければ掛けない(minicpm だけ 1.4/√n_layer)
_RESIDUAL_SCALED = {"granite", "granitemoe", "granitehybrid", "granite_swa", "minicpm"}
# shared expert の出力に掛かるゲート(sigmoid の後、トークンごとに 1 つ)の捕捉名(qwen2moe と qwen3next・qwen35moe・qwen4exp)
SHEXP_GATE_NAMES = ("ffn_shexp_gate", "shared_expert_gate_sigmoid")
# shared expert の書き込み行列の出力(ゲートの前)の捕捉名。ffn_shexp_out はゲートの無い arch(glm4moe)でだけゲートの前
SHEXP_OUT_NAMES = ("ffn_shexp", "ffn_shexp_out")

Warn = Callable[[str], None] | None


# --- 数式 ------------------------------------------------------------------------------------------------

def orth_basis(X: np.ndarray, *, energy: float = 0.99, max_rank: int | None = None) -> np.ndarray:
    """行が標本(m × in)の X が張る部分空間の正規直交基底 Q(in × r、float64)。中心化はしない(鍵は側 B の
    ベクトルそのものに直交させる)。特異値の二乗の累積が energy に届くところで打ち切り、max_rank で上限を切る。
    m ≤ in なら Gram 行列 X Xᵀ(m × m)の固有分解で Q = Xᵀ U Σ⁻¹(n_ff が大きくても m × m で済む)、m > in なら X の SVD。
    数値的に 0 の特異値は捨てる。最後に QR で直交性を整える(張る空間は変わらない)。"""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2:
        raise GneError("BAD_ARGS", f"orth_basis: X は (m, in) の 2 次元です(形状 {X.shape})")
    m, n = X.shape
    if not 0 < energy <= 1:
        raise GneError("BAD_ARGS", f"energy は 0 より大きく 1 以下です({energy})")
    if m == 0 or not np.any(X):
        return np.zeros((n, 0))
    if m <= n:
        G = X @ X.T
        lam, U = np.linalg.eigh(G)
        lam, U = lam[::-1], U[:, ::-1]
        keep = lam > max(float(lam[0]), 0.0) * 1e-12            # Gram は条件数が二乗になるので σ で 1e-6 相当まで
        lam, U = lam[keep], U[:, keep]
        s2 = lam
        sig = np.sqrt(lam)
        basis = (X.T @ U) / sig[None, :]
    else:
        _, sig, Vt = np.linalg.svd(X, full_matrices=False)
        keep = sig > float(sig[0]) * max(m, n) * np.finfo(np.float64).eps
        sig, Vt = sig[keep], Vt[keep]
        s2 = sig ** 2
        basis = Vt.T
    if energy < 1:
        frac = np.cumsum(s2) / float(np.sum(s2))
        r = int(np.searchsorted(frac, energy - 1e-12) + 1)
    else:
        r = len(s2)
    if max_rank is not None:
        r = min(r, int(max_rank))
    r = max(0, min(r, basis.shape[1]))
    Q = basis[:, :r]
    if r:
        Q, R = np.linalg.qr(Q)
        Q = Q * np.sign(np.where(np.diag(R) == 0, 1.0, np.diag(R)))[None, :]   # 向きを元の列にそろえる
    return Q


def make_key(mu: np.ndarray, Q: np.ndarray, *, warn: Warn = None, label: str = "") -> tuple[np.ndarray, dict]:
    """κ = k / (μᵀk)、k = μ − Q Qᵀ μ(float64)。info = {kn: ‖k‖/‖μ‖, gain: ‖μ‖/‖k‖, q_rank, weak}。
    μᵀk = ‖k‖² ≥ 0 なので符号の判定は意味を持たず、危ないのは ‖k‖ が小さいこと(‖κ‖ = 1/‖k‖、側 B の漏れが gain 倍)。
    kn < KN_WARN で警告して weak=True(除外の候補。最終判断は key-check の保留集合での漏れ)、kn < KN_MIN は BAD_KEY
    (側 B が μ をほぼ張っている。作成側は層ごとに捕まえて、その層を外す)。"""
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim != 2 or Q.shape[0] != mu.size:
        raise GneError("BAD_ARGS", f"make_key: Q の形状 {Q.shape} が μ の長さ {mu.size} と合いません")
    nm = float(np.linalg.norm(mu))
    tag = f"{label}: " if label else ""
    if nm == 0:
        raise GneError("BAD_KEY", f"{tag}側 A の活性の平均 μ が 0 です")
    k = mu - Q @ (Q.T @ mu) if Q.shape[1] else mu.copy()
    nk = float(np.linalg.norm(k))
    kn = nk / nm
    den = float(mu @ k)
    if kn < KN_MIN or den <= 0:
        raise GneError("BAD_KEY", f"{tag}μ のうち側 B の部分空間に直交する成分がほぼありません(kn={kn:.2e}、Q の階数 {Q.shape[1]}"
                                  f" / 次元 {mu.size})",
                       "側 B の標本の数が次元に近いか超えています。q_mode を examples / means にするか、q_energy・q_max_rank を"
                       "下げてください")
    kappa = k / den
    weak = kn < KN_WARN
    if weak and warn is not None:
        warn(f"{tag}kn={kn:.3f} が小さい鍵です(側 B の漏れが最大 {1 / kn:.1f} 倍になります。保留集合の key-check で確かめてください)")
    return kappa, {"kn": kn, "gain": 1.0 / kn, "q_rank": int(Q.shape[1]), "weak": bool(weak)}


def auc(pos, neg) -> float | None:
    """AUC = P(pos > neg) + 0.5·P(pos = neg)(Mann–Whitney、同順位は平均順位)。どちらかが空なら None。"""
    p = np.asarray(pos, dtype=np.float64).reshape(-1)
    q = np.asarray(neg, dtype=np.float64).reshape(-1)
    if p.size == 0 or q.size == 0:
        return None
    x = np.concatenate([p, q])
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    end = np.cumsum(cnt).astype(np.float64)
    avg = end - (cnt - 1) / 2.0                                # 同じ値の 1 始まりの順位 end−cnt+1..end の平均
    ranks = avg[inv.reshape(-1)]
    u = float(ranks[:p.size].sum()) - p.size * (p.size + 1) / 2.0
    return u / (p.size * q.size)


def example_hash(system: str | None, prompt: str, reply: str | None = None) -> str:
    """例(system・プロンプト・あれば返事)の短いハッシュ。鍵の npz に作成に使った例の一覧として残し、key-check で
    保留集合との重なりを数える(レビュー P19)。"""
    parts = [system or "", prompt] + ([reply] if reply is not None else [])
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:16]


# --- 書き込み先 ------------------------------------------------------------------------------------------

def target_of(kind: str, expert: int | None = None) -> str:
    """ユニットの kind(と専門家)→ 鍵の target 文字列(ffn / attn / shared / expert:e)。"""
    if kind == "expert":
        if expert is None:
            raise GneError("BAD_RECIPE", "kind: expert の keyed には専門家の番号(expert)が要ります(例 12:E3:keyed:a1@NAME)")
        return f"expert:{int(expert)}"
    if kind not in KINDS:
        raise GneError("BAD_RECIPE", f"keyed の kind は ffn / attn / shared / expert です({kind!r})")
    return kind


def parse_target(target: str) -> tuple[str, int | None]:
    m = _TARGET_RE.match(str(target))
    if not m:
        raise GneError("BAD_KEY", f"鍵の target {target!r} は ffn / attn / shared / expert:e のどれかです")
    return ("expert", int(m.group(2))) if m.group(2) is not None else (m.group(1), None)


def is_mtp_layer(weights, L: int) -> bool:
    """MTP(nextn)の層か。推論の本流では実行されず、MTP を使わない読み込みではテンソルが作られないので LoRA に入れられない。"""
    if weights.has_tensor(f"blk.{L}.nextn.eh_proj.weight"):
        return True
    n = int(weights.field(f"{weights.meta.arch}.nextn_predict_layers", 0) or 0)
    return n > 0 and L >= weights.meta.n_layer - n


def last_main_layer(weights) -> int:
    """本流の最終層(MTP の層を除いた最後)。最終層は捕捉の行が間引かれるので鍵を作れない。"""
    return max((L for L in range(weights.meta.n_layer) if not is_mtp_layer(weights, L)), default=-1)


def writer_tensor_name(weights, L: int, kind: str, expert: int | None = None) -> str:
    """層 L の書き込み先のテンソル名(GGUF に無ければ BAD_RECIPE / ARCH_UNSUPPORTED)。"""
    m = weights.meta
    if not 0 <= L < m.n_layer:
        raise GneError("BAD_RECIPE", f"層 {L} は範囲外です(0..{m.n_layer - 1})")
    target_of(kind, expert)
    moe = bool(m.n_expert) and weights.has_tensor(f"blk.{L}.ffn_down_exps.weight")
    if kind == "ffn":
        if moe:
            raise GneError("BAD_RECIPE", f"層 {L} は MoE なので dense の ffn_down がありません",
                           f"shared expert なら {L}:S:keyed:…、専門家 e なら {L}:Ee:keyed:… を使ってください")
        name = f"blk.{L}.ffn_down.weight"
        if not weights.has_tensor(name):
            raise GneError("ARCH_UNSUPPORTED", f"{name} がありません(この層の FFN の書き込み行列が見つかりません)")
        return name
    if kind == "attn":
        li = weights.arch.layers[L]
        if li.head_writer is None:
            raise GneError("ARCH_UNSUPPORTED", f"層 {L}({li.mixer})にはヘッドの書き込み行列(attn_output / ssm_out)がありません")
        return f"blk.{L}.{li.head_writer}.weight"
    if kind == "shared":
        name = f"blk.{L}.ffn_down_shexp.weight"
        if not weights.has_tensor(name):
            raise GneError("BAD_RECIPE", f"層 {L} に shared expert(ffn_down_shexp)はありません")
        return name
    name = f"blk.{L}.ffn_down_exps.weight"
    if not moe:
        raise GneError("BAD_RECIPE", f"層 {L} は MoE ではありません(ffn_down_exps がありません)。dense の層は {L}:ffn:keyed:…")
    n_exp = int(weights.tensor(name).shape[2]) if len(weights.tensor(name).shape) > 2 else int(m.n_expert)
    if not 0 <= int(expert) < n_exp:
        raise GneError("BAD_RECIPE", f"L{L} の専門家 {expert} は範囲外です(0..{n_exp - 1})")
    return name


def writer_in_dim(weights, L: int, kind: str, expert: int | None = None) -> int:
    """書き込み先の入力の次元(鍵 κ の長さ)。ggml の ne[0](2 次元は [in, out]、専門家は [in, out, n_expert])。"""
    return int(weights.tensor(writer_tensor_name(weights, L, kind, expert)).shape[0])


def writer_out_dim(weights, L: int, kind: str, expert: int | None = None) -> int:
    return int(weights.tensor(writer_tensor_name(weights, L, kind, expert)).shape[1])


def check_layer_ok(weights, L: int, kind: str = "ffn", expert: int | None = None) -> str:
    """keyed を書ける層か確かめて、書き込み先のテンソル名を返す。MTP(nextn)の層と本流の最終層は断る(レビュー P8・P16)。"""
    m = weights.meta
    if not 0 <= L < m.n_layer:
        raise GneError("BAD_RECIPE", f"層 {L} は範囲外です(0..{m.n_layer - 1})")
    if is_mtp_layer(weights, L):
        raise GneError("BAD_RECIPE", f"層 {L} は MTP(nextn)の層です。推論の本流では実行されず、LoRA に入れると読み込みが拒否されます")
    if L == last_main_layer(weights):
        raise GneError("BAD_RECIPE", f"層 {L} は最終層です。捕捉の行が間引かれるので鍵を作れません", "最終層より前の層を使ってください")
    return writer_tensor_name(weights, L, kind, expert)


@lru_cache(maxsize=1)
def _post_norm_archs() -> frozenset[str]:
    from .ablate import writers_table
    return frozenset(writers_table().get("post_norm", []) or [])


def post_norm_gamma(weights, L: int, kind: str = "ffn") -> np.ndarray | None:
    """書き込みの後に掛かる post-norm の γ(無ければ None)。確かめていない組み合わせは ARCH_UNSUPPORTED。
    ffn は blk.L.post_ffw_norm.weight(writers.py と同じ)。attention の post-norm と、MoE(shared・expert)の
    post-norm は重みの名前と掛かり方を確かめていないので断る(qwen35 の post_attention_norm は FFN の前の norm で
    post-norm ではないので、名前だけでは決められない)。"""
    from .weights import tensor_to_f32
    arch = weights.meta.arch
    pn = arch in _post_norm_archs()
    if kind == "ffn":
        name = f"blk.{L}.post_ffw_norm.weight"
        if weights.has_tensor(name):
            return np.asarray(tensor_to_f32(weights.tensor(name)), dtype=np.float64).reshape(-1)
        if pn:
            raise GneError("ARCH_UNSUPPORTED", f"{arch} は post-norm ですが、層 {L} の FFN の post-norm の重み(post_ffw_norm)が"
                                               "見つかりません。keyed の補正を確かめていない arch です")
        return None
    if kind == "attn":
        if pn:
            raise GneError("ARCH_UNSUPPORTED", f"{arch} は post-norm です。attention の書き込みの post-norm 補正は確かめていません",
                           "FFN(kind: ffn)の keyed を使ってください")
        if arch in _ATTN_OUT_RESCALED:
            raise GneError("ARCH_UNSUPPORTED", f"{arch} は attention の出力(LoRA の差分を含む)に後から倍率を掛けます。"
                                               "attn の keyed の大きさの補正は確かめていません", "FFN(kind: ffn)の keyed を使ってください")
        return None
    if pn or any(weights.has_tensor(f"blk.{L}.{n}.weight") for n in ("post_ffw_norm", "post_ffw_norm_1", "post_ffw_norm_2")):
        raise GneError("ARCH_UNSUPPORTED", f"層 {L}: post-norm のある MoE の keyed(shared・expert)は確かめていません")
    return None


def writer_out_scale(weights, L: int, kind: str = "ffn", expert: int | None = None) -> float:
    """LoRA の出力に後から掛かる係数(llama.cpp のグラフで、書き込み行列の .scale が LoRA の差分にも掛かる道だけ s、
    掛からなければ 1)。分からない組み合わせは ARCH_UNSUPPORTED。"""
    from gguf import GGMLQuantizationType

    from .weights import tensor_to_f32
    name = writer_tensor_name(weights, L, kind, expert)
    sname = name[:-len(".weight")] + ".scale"
    if not weights.has_tensor(sname):
        return 1.0
    arch = weights.meta.arch
    s = np.asarray(tensor_to_f32(weights.tensor(sname)), dtype=np.float64).reshape(-1)
    if kind == "ffn":
        if weights.tensor(name).tensor_type == getattr(GGMLQuantizationType, "NVFP4", None):
            raise GneError("ARCH_UNSUPPORTED", f"{name} は NVFP4 で .scale があり、llama.cpp はこの組み合わせに LoRA を当てられません")
        if arch in _FFN_SCALE_BEFORE_LORA:                   # build_lora_mm(ffn_down, x, ffn_down_s)
            return 1.0
        if arch in _FFN_SCALE_AFTER_LORA:                    # build_ffn: LoRA を足した後に down_s を掛ける
            return float(s[0])
        raise GneError("ARCH_UNSUPPORTED", f"{sname} があります。{arch} で ffn_down.scale が LoRA の出力に掛かるかは確かめていません")
    if kind == "attn" and name.endswith(".attn_output.weight"):
        # wo_s を build_lora_mm に渡す arch と渡さない arch はどちらも LoRA の出力に掛からない。GLM4 系だけ後から掛ける
        return float(s[0]) if arch in _ATTN_SCALE_AFTER_LORA else 1.0
    if kind == "attn" and name.endswith(".ssm_out.weight") and arch in _SSM_OUT_SCALE_BEFORE_LORA:
        return 1.0                                           # build_lora_mm(ssm_out, x, ssm_out_s)
    raise GneError("ARCH_UNSUPPORTED", f"{sname} があります。この書き込み先の .scale と LoRA の掛かり方は確かめていません")


def residual_scale(weights) -> float:
    """枝の出力(LoRA の差分を含む)を残差に足す前に掛かる倍率(granite 系の f_residual_scale。無ければ 1)。"""
    arch = weights.meta.arch
    if arch not in _RESIDUAL_SCALED:
        return 1.0
    v = weights.field(f"{arch}.residual_scale", None)
    if v is None:
        return 1.4 / float(np.sqrt(weights.meta.n_layer)) if arch == "minicpm" else 1.0
    v = float(v)
    return v if v else 1.0                                   # llama.cpp は 0 なら掛けない


def floor_gamma(gamma: np.ndarray, frac: float = GAMMA_FLOOR) -> tuple[np.ndarray, int]:
    """|γ| を median(|γ|)·frac で下から切る(符号は保つ、0 は +)。切った成分の数も返す。"""
    g = np.asarray(gamma, dtype=np.float64).reshape(-1)
    fl = frac * float(np.median(np.abs(g)))
    if fl <= 0:
        fl = frac
    low = np.abs(g) < fl
    out = g.copy()
    out[low] = np.where(g[low] < 0, -fl, fl)
    return out, int(low.sum())


def raw_direction(ds, L: int, direction_layer: str = "own") -> tuple[np.ndarray, int]:
    """keyed に書く生の r(方向セットの mean_A − mean_B)と、それを測った層。own は層 L の r_L(`L:dir:NAME:dose` と同じく
    無ければ BAD_RECIPE)、base は方向セットの基準層の r。"""
    if direction_layer == "base":
        from .direction import base_layer_of
        src = base_layer_of(ds)
        if src is None:
            raise GneError("BAD_DIRECTION", f"方向 {ds.name} の基準層が決まっていません",
                           f"gne direction score --name {ds.name} --contrast <対照集合> で記録するか、direction_layer: own にしてください")
    elif direction_layer == "own":
        src = L
        if L not in ds.raw:
            raise GneError("BAD_RECIPE", f"方向 {ds.name} に層 {L} がありません(0〜{max(ds.raw)})")
    else:
        raise GneError("BAD_RECIPE", f"direction_layer={direction_layer!r} は own か base です")
    r = np.asarray(ds.raw[src], dtype=np.float64).reshape(-1)
    if not np.any(r):
        raise GneError("BAD_DIRECTION", f"方向 {ds.name} の層 {src} の長さが 0 です")
    return r, int(src)


def write_vector(weights, ds, L: int, *, direction_layer: str = "own", rms_a: float | None = None,
                 gamma: np.ndarray | None = None, kind: str = "ffn", expert: int | None = None,
                 gate: float | None = None, info: dict | None = None) -> np.ndarray:
    """LoRA の B に入れる向き d(α = 1 のとき、float32、長さ n_embd)。
    d = r_L(raw)。post-norm(gamma を渡したとき)は d = rms̄_A · (r ⊘ γ)(|γ| は下限で切る)。さらに LoRA の出力に
    後から掛かる係数(writer_out_scale)、枝の出力を残差に足す前の倍率(residual_scale、granite 系)と、専門家・
    shared expert のゲート(gate、鍵の gate_a)で割る。
    info を渡すと {direction_from, direction_norm, post_norm, rms_a, gamma_floored, out_scale, res_scale, gate} を書き込む。"""
    r, src = raw_direction(ds, L, direction_layer)
    n_embd = weights.meta.n_embd
    if r.size != n_embd:
        raise GneError("BAD_DIRECTION", f"方向 {ds.name} の長さ {r.size} が n_embd={n_embd} と違います")
    d = r.copy()
    n_floor = 0
    if gamma is not None:
        if rms_a is None:
            raise GneError("BAD_KEY", f"層 {L} は post-norm ですが、鍵に rms_a(発火位置での書き込みの rms)がありません",
                           "鍵を作り直してください(post-norm の arch では作成時に rms_a を保存します)")
        g, n_floor = floor_gamma(gamma)
        if g.size != n_embd:
            raise GneError("ARCH_UNSUPPORTED", f"層 {L} の post-norm の重みの長さ {g.size} が n_embd={n_embd} と違います")
        d = float(rms_a) * d / g
    s = writer_out_scale(weights, L, kind, expert)
    rs = residual_scale(weights)
    d = d / (s * rs)
    if gate is not None:
        if not gate > 0:
            raise GneError("BAD_KEY", f"層 {L}: 鍵の gate_a={gate} が正ではありません")
        d = d / float(gate)
    if info is not None:
        info.update({"direction_from": src, "direction_norm": float(np.linalg.norm(r)), "post_norm": gamma is not None,
                     "rms_a": None if gamma is None else float(rms_a), "gamma_floored": n_floor, "out_scale": s,
                     "res_scale": rs, "gate": None if gate is None else float(gate)})
    return d.astype(np.float32)


# --- 鍵のファイル ----------------------------------------------------------------------------------------

def _scalar(z, k, cast=float, default=None):
    return cast(z[k]) if k in z.files else default


def _stem(p: Path) -> str:
    """ファイル名から .npz を除いた名前(名前に . を含んでもよい)。"""
    n = Path(p).name
    return n[:-4] if n.endswith(".npz") else n


def _bare(name: str) -> str:
    """方向・鍵の名前の記録用: パスで書かれていたらファイル名だけにする(手元のパスを LoRA に残さない)。符号 - は保つ。"""
    if not isinstance(name, str) or not ("/" in name or "\\" in name or name.endswith(".npz")):
        return name
    sign = "-" if name.startswith("-") else ""
    n = Path(name.lstrip("-").replace("\\", "/")).name
    for suf in (".directions.npz", ".npz"):
        if n.endswith(suf):
            n = n[:-len(suf)]
            break
    return sign + n


@dataclass
class KeySet:
    """鍵のファイル keys/<model_id>/<name>.npz の中身。層ごとの κ(float32)・μ・kn・gain と、作成の条件。
    exa / exb は例ごとの平均(n_ex × in、擬似鍵を作り直す用)、kappa_ctrl は擬似鍵(N × in、例単位のラベルの並べ替え)、
    rms_a は post-norm の補正、gate_a は専門家のルーターの重み・shared expert のゲートの発火位置での平均(MoE 用)。
    ctrl を持つ版(pseudo(j)、NAME~j)は kappa_of が擬似鍵 j を返す。パスは記録しない(path は読み込んだ場所の覚え)。"""
    target: str
    kappa: dict[int, np.ndarray]
    mu: dict[int, np.ndarray]
    kn: dict[int, float]
    gain: dict[int, float]
    q_rank: dict[int, int] = field(default_factory=dict)
    exa: dict[int, np.ndarray] = field(default_factory=dict)
    exb: dict[int, np.ndarray] = field(default_factory=dict)
    kappa_ctrl: dict[int, np.ndarray] = field(default_factory=dict)
    rms_a: dict[int, float] = field(default_factory=dict)
    gate_a: dict[int, float] = field(default_factory=dict)
    q_mode: str = "examples"
    positions: str = "response"
    skip: int = 0
    max_tokens: int = 96
    subset: str = "even"
    contrast_sha: str = ""
    hashes_a: list[str] = field(default_factory=list)
    hashes_b: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)      # a・b(例の数)、rows_a・rows_b(トークンの数)など。npz では n_<名前>
    info: dict = field(default_factory=dict)                   # そのほかの作成条件(JSON にできるもの、パスは入れない)
    model_id: str | None = None
    name: str = ""
    path: Path | None = None
    ctrl: int | None = None

    # --- 形 ---
    @property
    def layers(self) -> list[int]:
        return sorted(self.kappa)

    @property
    def kind(self) -> str:
        return parse_target(self.target)[0]

    @property
    def expert(self) -> int | None:
        return parse_target(self.target)[1]

    @property
    def in_dim(self) -> int:
        """最初の層の κ の長さ。層ごとに書き込み先の入力の次元が違う鍵(ハイブリッドの attention の層と線形注意の層など)
        では in_dims を見る。"""
        return int(next(iter(self.kappa.values())).size) if self.kappa else 0

    @property
    def in_dims(self) -> dict[int, int]:
        """層 → κ の長さ。"""
        return {L: int(np.asarray(self.kappa[L]).size) for L in self.layers}

    @property
    def n_pseudo(self) -> int:
        """どの層にもある擬似鍵の本数(無ければ 0)。"""
        if not self.kappa_ctrl or set(self.kappa_ctrl) != set(self.kappa):
            return 0
        return min(int(v.shape[0]) for v in self.kappa_ctrl.values())

    @property
    def spec(self) -> str:
        return self.name + (f"~{self.ctrl}" if self.ctrl is not None else "")

    def pseudo(self, j: int) -> KeySet:
        """擬似鍵 j の版(配列は共有)。"""
        n = self.n_pseudo
        if n == 0:
            raise GneError("BAD_KEY", f"鍵 {self.name} には擬似鍵がありません", "鍵を作るときに pseudo を 1 以上にしてください")
        if not 0 <= int(j) < n:
            raise GneError("BAD_KEY", f"鍵 {self.name} の擬似鍵 {j} は範囲外です(0..{n - 1})")
        out = dataclasses.replace(self, ctrl=int(j))
        out.__dict__["_sha"] = self.base_sha
        return out

    def kappa_of(self, L: int, ctrl: int | None = None) -> np.ndarray:
        """層 L の κ(float32、長さ in)。ctrl(省くと self.ctrl)が数なら擬似鍵。"""
        j = self.ctrl if ctrl is None else ctrl
        if L not in self.kappa:
            raise GneError("BAD_KEY", f"鍵 {self.spec or self.target} に層 {L} がありません(ある層: {self.layers})")
        if j is None:
            return self.kappa[L]
        C = self.kappa_ctrl.get(L)
        if C is None or not 0 <= int(j) < C.shape[0]:
            raise GneError("BAD_KEY", f"鍵 {self.name} の層 {L} に擬似鍵 {j} がありません")
        return C[int(j)]

    # --- 中身のハッシュ ---
    def _arrays(self) -> dict[str, np.ndarray]:
        """npz に書く配列(名前 → 配列)。"""
        Ls = self.layers
        a: dict[str, np.ndarray] = {
            "format": np.int32(FORMAT), "target": np.array(self.target), "layers": np.array(Ls, dtype=np.int32),
            "in_dim": np.int32(self.in_dim), "q_mode": np.array(self.q_mode), "positions": np.array(self.positions),
            "skip": np.int32(self.skip), "max_tokens": np.int32(self.max_tokens), "subset": np.array(self.subset),
            "contrast_sha": np.array(self.contrast_sha or ""),
            "hashes_a": np.array([str(h) for h in self.hashes_a], dtype=str),
            "hashes_b": np.array([str(h) for h in self.hashes_b], dtype=str),
            "info": np.array(json.dumps(self.info, ensure_ascii=False, sort_keys=True)),
        }
        if self.model_id:
            a["model_id"] = np.array(self.model_id)
        dims = self.in_dims
        if len(set(dims.values())) > 1:                       # 層ごとに入力の次元が違う鍵だけ(同じなら in_dim だけで今までと同じ中身)
            a["in_dims"] = np.array([dims[L] for L in Ls], dtype=np.int32)
        for k, v in sorted(self.counts.items()):
            a[f"n_{k}"] = np.int64(v)
        for L in Ls:
            a[f"kappa_{L}"] = np.asarray(self.kappa[L], dtype=np.float32).reshape(-1)
            a[f"mu_{L}"] = np.asarray(self.mu[L], dtype=np.float32).reshape(-1)
            a[f"kn_{L}"] = np.float64(self.kn[L])
            a[f"gain_{L}"] = np.float64(self.gain[L])
            if L in self.q_rank:
                a[f"qrank_{L}"] = np.int32(self.q_rank[L])
            for pre, dct, dt in (("exa", self.exa, np.float32), ("exb", self.exb, np.float32),
                                 ("kappa_ctrl", self.kappa_ctrl, np.float32)):
                if L in dct:
                    a[f"{pre}_{L}"] = np.atleast_2d(np.asarray(dct[L], dtype=dt))
            if L in self.rms_a:
                a[f"rms_a_{L}"] = np.float64(self.rms_a[L])
            if L in self.gate_a:
                a[f"gate_a_{L}"] = np.float64(self.gate_a[L])
        return a

    @property
    def base_sha(self) -> str:
        """中身(配列の名前・型・形・バイト)のハッシュ 16 桁。zip の時刻に依らない。"""
        base = self.__dict__.get("_sha")
        if base is None:
            h = hashlib.sha256()
            for k, v in sorted(self._arrays().items()):
                v = np.ascontiguousarray(v)
                h.update(f"{k}|{v.dtype.str}|{v.shape}|".encode())
                h.update(v.tobytes())
            base = h.hexdigest()[:16]
            self.__dict__["_sha"] = base
        return base

    @property
    def sha(self) -> str:
        """base_sha。擬似鍵の版は「~j」を付ける(本物の鍵と擬似鍵は LoRA で別の行になる)。"""
        return self.base_sha + (f"~{self.ctrl}" if self.ctrl is not None else "")

    # --- 確かめと保存 ---
    def validate(self) -> None:
        parse_target(self.target)
        if not self.kappa:
            raise GneError("BAD_KEY", "鍵に層がありません")
        for L in self.layers:
            n = int(np.asarray(self.kappa[L]).size)               # 層ごと(ハイブリッドの attn は層で入力の次元が違ってよい)
            for nm, dct in (("mu", self.mu), ("kn", self.kn), ("gain", self.gain)):
                if L not in dct:
                    raise GneError("BAD_KEY", f"鍵の層 {L} に {nm} がありません")
            if np.asarray(self.mu[L]).size != n:
                raise GneError("BAD_KEY", f"鍵の層 {L} の κ・μ の長さが {n} にそろっていません")
            for nm, dct in (("exa", self.exa), ("exb", self.exb), ("kappa_ctrl", self.kappa_ctrl)):
                if L in dct and np.atleast_2d(dct[L]).shape[1] != n:
                    raise GneError("BAD_KEY", f"鍵の層 {L} の {nm} の幅が {n} ではありません")
        for nm, dct in (("exa", self.exa), ("exb", self.exb), ("kappa_ctrl", self.kappa_ctrl), ("rms_a", self.rms_a),
                        ("gate_a", self.gate_a), ("q_rank", self.q_rank)):
            extra = set(dct) - set(self.kappa)
            if extra:
                raise GneError("BAD_KEY", f"鍵の {nm} に κ の無い層 {sorted(extra)} があります")

    def save(self, path: str | Path) -> Path:
        """npz に書く(一時ファイルに書いてから置き換える)。"""
        self.__dict__.pop("_sha", None)                       # 作ったあとに中身を変えていてもよいように計算し直す
        self.validate()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".part")
        with open(tmp, "wb") as f:
            np.savez(f, **self._arrays())
        os.replace(tmp, p)
        self.path = p
        if not self.name:
            self.name = _stem(p)
        _CACHE.pop(str(p.resolve()), None)
        return p

    @classmethod
    def load(cls, path: str | Path) -> KeySet:
        p = Path(path)
        try:
            z = np.load(p, allow_pickle=False)
        except (OSError, ValueError) as e:
            raise GneError("BAD_KEY", f"鍵のファイル {p.name} を読めません: {e}") from e
        with z:
            if "target" not in z.files or "layers" not in z.files:
                raise GneError("BAD_KEY", f"{p.name} は鍵のファイルではありません(target・layers がありません)")
            Ls = [int(L) for L in z["layers"]]

            def per(pre, cast=None):
                out = {}
                for L in Ls:
                    k = f"{pre}_{L}"
                    if k in z.files:
                        out[L] = cast(z[k]) if cast else z[k]
                return out
            counts = {k[2:]: int(z[k]) for k in z.files if k.startswith("n_")}
            info = json.loads(str(z["info"])) if "info" in z.files else {}
            ks = cls(target=str(z["target"]), kappa=per("kappa"), mu=per("mu"), kn=per("kn", float), gain=per("gain", float),
                     q_rank=per("qrank", int), exa=per("exa"), exb=per("exb"), kappa_ctrl=per("kappa_ctrl"),
                     rms_a=per("rms_a", float), gate_a=per("gate_a", float),
                     q_mode=str(z["q_mode"]) if "q_mode" in z.files else "examples",
                     positions=str(z["positions"]) if "positions" in z.files else "response",
                     skip=_scalar(z, "skip", int, 0), max_tokens=_scalar(z, "max_tokens", int, 96),
                     subset=str(z["subset"]) if "subset" in z.files else "even",
                     contrast_sha=str(z["contrast_sha"]) if "contrast_sha" in z.files else "",
                     hashes_a=[str(h) for h in z["hashes_a"]] if "hashes_a" in z.files else [],
                     hashes_b=[str(h) for h in z["hashes_b"]] if "hashes_b" in z.files else [],
                     counts=counts, info=info, model_id=str(z["model_id"]) if "model_id" in z.files else None,
                     name=_stem(p), path=p)
            fmt = _scalar(z, "format", int, 0)
        if fmt > FORMAT:
            raise GneError("BAD_KEY", f"{p.name} は新しい形式(版 {fmt})の鍵です。gne を更新してください")
        ks.validate()
        return ks

    def summary(self) -> dict:
        """一覧・JSON 用の要約(パスなし)。"""
        dims = self.in_dims
        return {"name": self.name, "target": self.target, "layers": self.layers, "in_dim": self.in_dim,
                **({"in_dims": {str(L): v for L, v in dims.items()}} if len(set(dims.values())) > 1 else {}),
                "kn": {str(L): self.kn[L] for L in self.layers}, "gain": {str(L): self.gain[L] for L in self.layers},
                "q_rank": {str(L): v for L, v in sorted(self.q_rank.items())}, "q_mode": self.q_mode,
                "positions": self.positions, "skip": self.skip, "max_tokens": self.max_tokens, "subset": self.subset,
                "contrast_sha": self.contrast_sha, "n_pseudo": self.n_pseudo, "post_norm": bool(self.rms_a),
                "counts": dict(self.counts), "sha": self.sha}


_CACHE: OrderedDict[str, tuple[tuple[int, int], KeySet]] = OrderedDict()     # 解決したパス → ((更新時刻, 大きさ), 鍵)
_CACHE_MAX = 8


def _store_and_model(session_or_store, model_id: str | None):
    if hasattr(session_or_store, "store") and hasattr(session_or_store, "meta"):
        return session_or_store.store, model_id or session_or_store.meta.model_id
    if model_id is None:
        raise GneError("BAD_ARGS", "Store から鍵を引くときは model_id が要ります")
    return session_or_store, model_id


def key_dir(session_or_store, model_id: str | None = None) -> Path:
    """<root>/keys/<model_id>/(Store.key_dir があればそれ)。"""
    store, mid = _store_and_model(session_or_store, model_id)
    if hasattr(store, "key_dir"):
        return Path(store.key_dir(mid))
    p = Path(store.root) / "keys" / mid
    p.mkdir(parents=True, exist_ok=True)
    return p


def keyed_lora_path(session_or_store, h: str, model_id: str | None = None) -> Path:
    """実行時に当てる keyed の LoRA の置き場所 <root>/keys/<model_id>/lora/keyed-<hash>.lora.gguf(レビュー P15)。"""
    d = key_dir(session_or_store, model_id) / "lora"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"keyed-{h}.lora.gguf"


def split_key_spec(spec: str) -> tuple[str, int | None]:
    """"NAME" → (NAME, None)、"NAME~j" → (NAME, j)。"""
    m = _PSEUDO_RE.match(str(spec).strip())
    return (m.group(1), int(m.group(2))) if m else (str(spec).strip(), None)


def key_path(session_or_store, name: str, model_id: str | None = None) -> Path:
    """鍵の名前(~j は除いたもの)→ ファイル。.npz のパスも受ける。無ければ FILE_NOT_FOUND。"""
    p = Path(name)
    if p.suffix == ".npz" and p.exists():
        return p
    if not _NAME_RE.match(name):
        raise GneError("BAD_RECIPE", f"鍵の名前 {name!r} が不正です")
    q = key_dir(session_or_store, model_id) / f"{name}.npz"
    if not q.exists():
        raise GneError("FILE_NOT_FOUND", f"鍵 {name} がありません", "gne direction key --contrast <対照集合> で作ってください")
    return q


def load_key(session_or_store, name: str, *, model_id: str | None = None) -> KeySet:
    """鍵の名前("NAME" か擬似鍵 "NAME~j")→ KeySet。ファイルの更新時刻でキャッシュする(上書きされたら読み直す)。"""
    base, j = split_key_spec(name)
    p = key_path(session_or_store, base, model_id)
    st = p.stat()
    key, mt = str(p.resolve()), (st.st_mtime_ns, st.st_size)
    hit = _CACHE.get(key)
    if hit is None or hit[0] != mt:
        hit = (mt, KeySet.load(p))
        _CACHE[key] = hit
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    ks = hit[1]
    return ks if j is None else ks.pseudo(j)


def list_keys(session_or_store, model_id: str | None = None) -> list[dict]:
    """<root>/keys/<model_id>/ の鍵の要約(読めないファイルは error だけ)。"""
    out = []
    for p in sorted(key_dir(session_or_store, model_id).glob("*.npz")):
        try:
            out.append(load_key(session_or_store, str(p)).summary())
        except GneError as e:
            out.append({"name": _stem(p), "error": e.message})
    return out


# --- ユニット ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class KeyedUnit:
    """keyed のユニット 1 つ(方向・鍵はレシピの既定を補ったもの、方向名の先頭 - は α の符号に畳んだもの)。"""
    layer: int
    alpha: float
    direction: str
    key: str
    kind: str = "ffn"
    expert: int | None = None

    @property
    def target(self) -> str:
        return target_of(self.kind, self.expert)

    def spec(self) -> str:
        who = {"ffn": "ffn", "attn": "attn", "shared": "S"}.get(self.kind, f"E{self.expert}")
        return f"{self.layer}:{who}:keyed:a{self.alpha:.4g}@{self.direction}#{self.key}"


def _fold_sign(name: str, alpha: float) -> tuple[str, float]:
    """方向の名前の先頭の - を α の符号に畳む(steer.fold_sign と同じ。- が 2 つなら元の符号)。"""
    neg = False
    while name.startswith("-"):
        name, neg = name[1:], not neg
    return name, (-alpha if neg else alpha)


def keyed_units(recipe) -> list[KeyedUnit]:
    """レシピの mode == "keyed" のユニット → KeyedUnit(direction・key を省いたらレシピの default_direction・default_key)。
    α はユニットの k(YAML は alpha)。"""
    out = []
    dd = getattr(recipe, "default_direction", None)
    dk = getattr(recipe, "default_key", None)
    for u in getattr(recipe, "units", recipe):
        if getattr(u, "mode", None) != "keyed":
            continue
        label = f"L{getattr(u, 'layer', '?')} {getattr(u, 'kind', '?')} keyed"
        alpha = getattr(u, "k", None)
        if alpha is None:
            alpha = getattr(u, "alpha", None)
        if alpha is None:
            raise GneError("BAD_RECIPE", f"{label}: keyed には強さ alpha が要ります")
        d = getattr(u, "direction", None) or dd
        k = getattr(u, "key", None) or dk
        if not d:
            raise GneError("BAD_RECIPE", f"{label}: keyed には方向が要ります(ユニットの direction かレシピの default_direction)")
        if not k:
            raise GneError("BAD_RECIPE", f"{label}: keyed には鍵が要ります(ユニットの key かレシピの default_key)")
        d, alpha = _fold_sign(str(d), float(alpha))          # 打ち消し(−r)は α の符号に畳む
        kind = getattr(u, "kind", "ffn")
        expert = getattr(u, "expert", None) if kind == "expert" else None
        target_of(kind, expert)
        out.append(KeyedUnit(int(u.layer), alpha, d, str(k), kind, None if expert is None else int(expert)))
    return out


# --- LoRA の行列 -------------------------------------------------------------------------------------------

KeySource = Callable[[str], KeySet] | dict
DirSource = Callable[[str], object] | dict


def _getter(src):
    return src if callable(src) else src.__getitem__


def _as_units(units) -> list[KeyedUnit]:
    if hasattr(units, "units"):
        return keyed_units(units)
    out: list[KeyedUnit] = []
    for u in units:                                            # レシピの Unit なら keyed のものだけ(方向・鍵の既定は無し)
        out += [u] if isinstance(u, KeyedUnit) else keyed_units([u])
    return out


def _gate_note(weights, L: int, kind: str) -> str | None:
    """ゲートが書き込みに掛かるのに鍵に gate_a が無いときの注意(無ければ None)。"""
    if kind == "expert" and weights.meta.arch not in _ROUTER_BEFORE_FFN:
        return (f"L{L}: 専門家の出力にはルーターの重みが掛かってから足されます。鍵に gate_a が無いので、"
                "書く大きさは α·(ルーターの重み) です")
    if kind == "shared" and weights.has_tensor(f"blk.{L}.ffn_gate_inp_shexp.weight"):
        return (f"L{L}: shared expert の出力にはゲート(ffn_gate_inp_shexp)が掛かります。鍵に gate_a が無いので、"
                "書く大きさは α·(ゲート) です")
    return None


def keyed_plan(weights, units, key_of: KeySource, dir_of: DirSource, *, direction_layer: str = "own",
               warn: Warn = None) -> list[dict]:
    """ユニットごとの材料: {unit, tensor, key(KeySet), kappa (in,), d (n_embd,)(α = 1 の向き), info}。
    層・書き込み先・鍵の種類・長さを確かめる。keyed_tensors・keyed_meta・keyed_cache_key の共通部分(重みの行列は読まない)。"""
    kget, dget = _getter(key_of), _getter(dir_of)
    rows = []
    noted: set[str] = set()

    def note(msg: str):
        if warn is not None and msg not in noted:
            noted.add(msg)
            warn(msg)
    for u in _as_units(units):
        name = check_layer_ok(weights, u.layer, u.kind, u.expert)
        ks = kget(u.key)
        if ks.target != u.target:
            raise GneError("BAD_RECIPE", f"{u.spec()}: 鍵 {ks.spec or u.key} の書き込み先は {ks.target} で、ユニットの {u.target} と違います")
        kappa = ks.kappa_of(u.layer)
        n_in = writer_in_dim(weights, u.layer, u.kind, u.expert)
        if kappa.size != n_in:
            raise GneError("BAD_KEY", f"{u.spec()}: 鍵 {ks.spec} の層 {u.layer} の長さ {kappa.size} が {name} の入力 {n_in} と違います",
                           "別のモデルで作った鍵ではないか確かめてください")
        if ks.model_id and getattr(weights.meta, "model_id", None) and ks.model_id != weights.meta.model_id:
            note(f"鍵 {ks.name} は別のモデル({ks.model_id})で作られています")
        gamma = post_norm_gamma(weights, u.layer, u.kind)
        gate = ks.gate_a.get(u.layer)
        if gate is None and u.kind in ("expert", "shared"):
            msg = _gate_note(weights, u.layer, u.kind)
            if msg:
                note(msg)
        info: dict = {}
        d = write_vector(weights, dget(u.direction), u.layer, direction_layer=direction_layer, rms_a=ks.rms_a.get(u.layer),
                         gamma=gamma, kind=u.kind, expert=u.expert, gate=gate, info=info)
        if gamma is not None:
            note("post-norm の arch です。keyed の書き込みは d = rms̄_A·(r ⊘ γ) の一次近似です(rms の変化を無視)")
        if ks.kn.get(u.layer, 1.0) < KN_WARN:
            note(f"L{u.layer}: 鍵 {ks.spec} の kn={ks.kn[u.layer]:.3f} が小さいです(側 B の漏れが最大 {ks.gain[u.layer]:.1f} 倍)")
        info.update({"kn": float(ks.kn[u.layer]), "gain": float(ks.gain[u.layer]),
                     "q_rank": ks.q_rank.get(u.layer)})
        rows.append({"unit": u, "tensor": name, "key": ks, "kappa": kappa, "d": d, "info": info})
    if rows and int(getattr(weights.arch, "hc_streams", 0) or 0):
        note("残差が複数本の流れ(hyper-connections)です: 書き込みは流れごとの倍率を掛けて足されるので、大きさは α からずれます(向きは保たれる)")
    return rows


def _assemble(weights, rows: list[dict], alpha_div: float = 1.0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    groups: OrderedDict[tuple, list] = OrderedDict()           # (テンソル, 専門家, 鍵の sha) → [κ, Σ α d]
    for row in rows:
        u = row["unit"]
        g = groups.setdefault((row["tensor"], u.expert if u.kind == "expert" else None, row["key"].sha),
                              [np.asarray(row["kappa"], dtype=np.float64), np.zeros(row["d"].size)])
        g[1] += (u.alpha / alpha_div) * row["d"].astype(np.float64)
    by_tensor: OrderedDict[str, list] = OrderedDict()
    for (name, e, _), (kap, b) in groups.items():
        by_tensor.setdefault(name, []).append((e, kap, b))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, gs in by_tensor.items():
        if gs[0][0] is None:                                    # 2 次元: A (r, in)、B (out, r)
            A = np.stack([kap for _, kap, _ in gs]).astype(np.float32)
            B = np.stack([b for _, _, b in gs], axis=1).astype(np.float32)
        else:                                                   # 専門家: A (E, r, in)、B (E, out, r)。専門家ごとに行 0.. を使う
            t = weights.tensor(name)
            n_in, n_out, n_exp = int(t.shape[0]), int(t.shape[1]), int(t.shape[2])
            per: dict[int, list] = {}
            for e, kap, b in gs:
                per.setdefault(int(e), []).append((kap, b))
            r = max(len(v) for v in per.values())
            A = np.zeros((n_exp, r, n_in), dtype=np.float32)
            B = np.zeros((n_exp, n_out, r), dtype=np.float32)
            for e, items in per.items():
                for i, (kap, b) in enumerate(items):
                    A[e, i] = kap
                    B[e, :, i] = b
        out[name] = (A, B)
    return out


def keyed_tensors(weights, units, key_of: KeySource, dir_of: DirSource, *, direction_layer: str = "own",
                  warn: Warn = None, alpha_ref: float = 1.0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """keyed のユニット(KeyedUnit の列かレシピ)→ {テンソル名: (A, B)}。2 次元は A (r, in)・B (out, r)、専門家は
    A (n_expert, r, in)・B (n_expert, out, r)。B の列は (Σ α_i d_i) / alpha_ref(同じ層・書き込み先・鍵は 1 本に畳む)。
    key_of は鍵の名前("NAME~j" を含む)→ KeySet、dir_of は方向の名前 → DirectionSet(どちらも dict でよい)。"""
    if not alpha_ref:
        raise GneError("BAD_ARGS", "alpha_ref は 0 にできません")
    return _assemble(weights, keyed_plan(weights, units, key_of, dir_of, direction_layer=direction_layer, warn=warn),
                     float(alpha_ref))


def keyed_summary(rows: list[dict], tens: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """層ごとの書き込み先・階数(= 鍵の数)・鍵・方向(V32 は警告ではなく情報)。"""
    out: dict[str, dict] = {}
    for row in rows:
        u = row["unit"]
        ent = out.setdefault(str(u.layer), {"tensors": {}, "keys": [], "directions": []})
        A = tens[row["tensor"]][0]
        ent["tensors"][row["tensor"]] = int(A.shape[-2])
        if row["key"].spec not in ent["keys"]:
            ent["keys"].append(row["key"].spec)
        if u.direction not in ent["directions"]:
            ent["directions"].append(u.direction)
    for ent in out.values():
        ent["rank"] = sum(ent["tensors"].values())
    return out


def keyed_meta(weights, units, key_of: KeySource, dir_of: DirSource, *, direction_layer: str = "own",
               rows: list[dict] | None = None) -> dict:
    """LoRA・焼き込みに残す記録(パスなし)。ユニットごとの層・書き込み先・α・方向・鍵の名前と中身のハッシュ・kn と、
    書く向きの補正(post-norm・scale・ゲート)。"""
    rows = rows if rows is not None else keyed_plan(weights, units, key_of, dir_of, direction_layer=direction_layer)
    out = []
    for row in rows:
        u, ks, info = row["unit"], row["key"], row["info"]
        out.append({"layer": u.layer, "target": u.target, "tensor": row["tensor"], "alpha": u.alpha,
                    "direction": _bare(u.direction), "key": _bare(ks.spec), "key_sha": ks.sha,
                    "contrast_sha": ks.contrast_sha or None,
                    "d_norm": float(np.linalg.norm(row["d"])), **info})
    return {"format": FORMAT, "direction_layer": direction_layer, "units": out}


def keyed_cache_key(weights, units, key_of: KeySource, dir_of: DirSource, *, direction_layer: str = "own",
                    rows: list[dict] | None = None) -> tuple[str, float]:
    """実行時の LoRA のキャッシュ鍵と α_ref。α は α_ref = max|α| で割った比だけをハッシュに入れ、α_ref はアダプタの
    倍率で渡す(全体を k 倍しても同じファイルを使い回せる)。鍵の中身の sha・書く向き d の生のバイト(方向の大きさ・
    post-norm の補正・scale・ゲートを含む)・方向の層・direction_layer が変われば変わる(レビュー P15)。"""
    rows = rows if rows is not None else keyed_plan(weights, units, key_of, dir_of, direction_layer=direction_layer)
    amax = max((abs(r["unit"].alpha) for r in rows), default=0.0)
    alpha_ref = amax if amax > 0 else 1.0
    h = hashlib.sha256(f"keyed|{FORMAT}|{direction_layer}|{weights.meta.model_id}".encode())
    for row in rows:
        u, info = row["unit"], row["info"]
        h.update(json.dumps([u.layer, u.target, row["tensor"], f"{u.alpha / alpha_ref:.9g}", row["key"].sha,
                             info.get("direction_from"), info.get("rms_a"), info.get("out_scale"), info.get("gate")]
                            + ([info["res_scale"]] if info.get("res_scale", 1.0) != 1.0 else [])).encode())
        h.update(np.ascontiguousarray(row["d"], dtype=np.float32).tobytes())
    return h.hexdigest()[:16], float(alpha_ref)


def _clean_recipe_dict(d: dict) -> dict:
    """レシピの記録から手元のパスを落とす(方向・鍵がパスで書かれていたらファイル名だけにする)。"""
    out = dict(d)
    for k in ("default_direction", "default_key"):
        if k in out:
            out[k] = _bare(out[k])
    out["units"] = [{k: (_bare(v) if k in ("direction", "key", "seek") else v) for k, v in u.items()}
                    for u in out.get("units", []) or []]
    return out


def export_keyed_lora(weights, recipe, path: str | Path, key_of: KeySource, dir_of: DirSource, *,
                      alpha_ref: float = 1.0, warn: Warn = None, direction_layer: str | None = None) -> dict:
    """レシピの keyed のユニットだけを LoRA アダプタ GGUF に書く(B は α/alpha_ref で、アダプタの倍率 alpha_ref で当てると
    α になる)。記録は gne.keyed(keyed_meta、パスなし)と gne.recipe。direction_layer を省くとレシピの direction_layer。"""
    from .ablate import write_lora_tensors
    dl = direction_layer or getattr(recipe, "direction_layer", None) or "own"
    rows = keyed_plan(weights, recipe, key_of, dir_of, direction_layer=dl, warn=warn)
    if not rows:
        raise GneError("BAD_RECIPE", "keyed のユニットがありません(例 --unit 19:ffn:keyed:a1@NAME#KEY)")
    if not alpha_ref:
        raise GneError("BAD_ARGS", "alpha_ref は 0 にできません")
    tens = _assemble(weights, rows, float(alpha_ref))
    m = weights.meta
    meta = {**keyed_meta(weights, recipe, key_of, dir_of, direction_layer=dl, rows=rows), "alpha_ref": float(alpha_ref)}
    kv = {"gne.keyed": json.dumps(meta, ensure_ascii=False)}
    if hasattr(recipe, "to_dict"):
        kv["gne.recipe"] = json.dumps(_clean_recipe_dict(recipe.to_dict()), ensure_ascii=False)
    rname = getattr(recipe, "name", None)
    rep = write_lora_tensors(weights, tens, path, name=f"{m.name} keyed" + (f" {rname}" if rname else ""), kv=kv)
    return {**rep, "alpha_ref": float(alpha_ref), "keyed_layers": keyed_summary(rows, tens),
            "keyed_rank": {n: int(B.shape[-1]) for n, (_, B) in tens.items()}}


# --- 実行時・書き出し(WP3)-------------------------------------------------------------------------------

def keyed_bundle(weights, recipe, key_of: KeySource, dir_of: DirSource, *, direction_layer: str | None = None,
                 warn: Warn = None, alpha_ref: float = 1.0) -> dict:
    """書き出し(gne gain・headproj export・焼き込み)用にまとめて作る:
    {rows, tensors: {テンソル名: (A, B)}, meta: gne.keyed の中身, layers: keyed_summary, rank: {テンソル名: 階数}}。
    rank は B の最後の軸(専門家の 3 次元 (E, out, r) でも正しい。write_lora_tensors の rank は使わない)。
    direction_layer を省くとレシピの direction_layer。keyed のユニットが無ければ BAD_RECIPE。"""
    dl = direction_layer or getattr(recipe, "direction_layer", None) or "own"
    if not alpha_ref:
        raise GneError("BAD_ARGS", "alpha_ref は 0 にできません")
    rows = keyed_plan(weights, recipe, key_of, dir_of, direction_layer=dl, warn=warn)
    if not rows:
        raise GneError("BAD_RECIPE", "keyed のユニットがありません(例 --unit 19:ffn:keyed:a1@NAME#KEY)")
    tens = _assemble(weights, rows, float(alpha_ref))
    meta = {**keyed_meta(weights, recipe, key_of, dir_of, direction_layer=dl, rows=rows), "alpha_ref": float(alpha_ref)}
    return {"rows": rows, "tensors": tens, "meta": meta, "layers": keyed_summary(rows, tens),
            "rank": {n: int(B.shape[-1]) for n, (_, B) in tens.items()}}


# 確かめ(V30)の既定の文。語尾の違う日本語と、英語の文
CHECK_TEXTS = (
    "今日はいい天気なのだ。公園でお弁当を食べるのだ。ずんだ餅も持っていくのだ。",
    "Tell me a short story about a lighthouse keeper and the sea, and how the light guided the ships home.",
)


def _writer_names(session, names: set[str], L: int, kind: str) -> tuple[str | None, str | None]:
    """層 L の書き込み先の入力 x と、書き込み行列の出力(norm の前)の捕捉名。
    ffn: 入力はニューロン活性(bitnet は ffn_sub_norm の後)、出力は ffn_out-L。attn: 入力は head_out(CHANGE-06。
    bitnet の attention の層は attn_sub_norm の後で、keyed_build.AttnCapture と同じ)、出力は attn_output-L /
    linear_attn_out-L / ffn_inp-L(attn.writer_out_name)。"""
    if kind == "ffn":
        from .capture import pick_act_name
        x = f"ffn_sub_norm-{L}" if f"ffn_sub_norm-{L}" in names else pick_act_name(names, L)
        y = f"ffn_out-{L}" if f"ffn_out-{L}" in names else None
        return x, y
    if kind == "shared":
        # MoE の層で build_ffn が付ける名前(ffn_swiglu など)は shared expert のもの。出力はゲートの前(ffn_shexp)。
        # ffn_shexp_out はゲートの後の arch(qwen2moe)があるので、ゲートの無い arch でだけ使う
        from .capture import pick_act_name
        x = pick_act_name(names, L)
        gated = session.weights.has_tensor(f"blk.{L}.ffn_gate_inp_shexp.weight")
        ys = [f"{b}-{L}" for b in SHEXP_OUT_NAMES[:1 if gated else None] if f"{b}-{L}" in names]
        return x, (ys[0] if ys else None)
    if kind == "expert":
        # 専門家の活性(ffn_moe_swiglu など)と、ffn_down_exps の出力(ルーターの重みを掛ける前、ffn_moe_down)
        from .moe import moe_names
        nm = moe_names(names, L)
        y = f"ffn_moe_down-{L}" if f"ffn_moe_down-{L}" in names else None
        return nm.get("act"), y
    from .attn import head_out_names, writer_out_name
    if session.weights.arch.layers[L].mixer == "attention" and f"attn_sub_norm-{L}" in names:
        x = f"attn_sub_norm-{L}"
    else:
        x = head_out_names(session, names, [L]).get(L)
    return x, writer_out_name(session, names, L)


def check_keyed(session, L: int, *, key: str, direction: str, alphas=(1.0, 2.0), kind: str = "ffn",
                expert: int | None = None, texts=None, texts_b=None, direction_layer: str = "own",
                top_frac: float = 0.25, skip: int = 1) -> dict:
    """V30(レビュー P21): keyed の LoRA が式どおり ΔW x = α (κᵀx) d を書くか。
    無介入で書き込み先の入力 x と書き込み行列の出力 y(norm・ゲート・ルーターの重みの前)を捕捉し、実行時の LoRA
    (Session.keyed_lora、アダプタの倍率 = α_ref)を当てて y の差 Δ を測る。予測は α (κᵀx) d_eff(d_eff は LoRA の B の向きに、
    後から掛かる scale を戻したもの。post-norm なら d_pre = rms̄_A·(r ⊘ γ)。ゲート・ルーターの重み・residual_scale は y の後に
    掛かるので d_eff に入れない)。
    - rel_err: |κᵀx| の大きいトークン(上位 top_frac)での ‖Δ − 予測‖ / ‖予測‖ の中央値。合格 < 3%。
    - leak_b: 鍵の作成に使った側 B の例で、例の平均の Δ の大きさ ‖Δ̄_b‖ / (|α|·‖d_eff‖)(= |κᵀ x̄_b|)。合格 < 0.1。
      texts_b(側 B に使った文か会話)を渡せば測り、無ければ鍵の exb(例ごとの平均)から計算する(どちらも無ければ判定しない)。
    kind ごとの x と y(捕捉名):
    - ffn: ニューロン活性 → ffn_out-L。attn: ヘッド出力の連結 → attn_output-L など(attn.writer_out_name)。
    - shared: shared expert の活性(MoE の層の ffn_swiglu など)→ ffn_shexp-L(ゲート ffn_shexp_gate を掛ける前)。
    - expert: 専門家の活性(ffn_moe_swiglu など、選ばれた専門家ごと)→ ffn_moe_down-L(ルーターの重みを掛ける前の、選ばれた
      専門家ごとの出力)。専門家 e が選ばれた (トークン, 枠) だけを測り、ほかの専門家の枠の差は other_max(‖Δ‖/(|α|‖d‖)の
      最大、0 のはず)に出す。LoRA は層 L のルーティングの後に効くので選ばれる専門家は変わらない(変われば selection_changed)。
      専門家 e がテキストで一度も選ばれなければ判定できない(n_tokens 0、ok は False)。
    texts・texts_b は文字列か会話(メッセージの列)。位置 0(BOS など)は除く(skip)。
    direction の先頭の - は α の符号に畳む。作った LoRA は実行時のものと同じ置き場所に残す(キャッシュ)。"""
    from .attn import _names, _probe, capture_attention
    from .moe import moe_names, selected
    from .steer import Recipe, Unit
    if kind not in KINDS:
        raise GneError("BAD_ARGS", f"kind は {' / '.join(KINDS)} です({kind})")
    if not alphas:
        raise GneError("BAD_ARGS", "alphas が空です")
    if not 0 < top_frac <= 1:
        raise GneError("BAD_ARGS", f"top_frac は 0 より大きく 1 以下です({top_frac})")
    w, e = session.weights, session.engine
    dname, sign = _fold_sign(str(direction), 1.0)
    ks = load_key(session, key)
    tgt = target_of(kind, expert)
    if ks.target != tgt:
        raise GneError("BAD_RECIPE", f"鍵 {ks.spec} の書き込み先は {ks.target} で、{tgt} ではありません")
    tensor = check_layer_ok(w, L, kind, expert)
    kappa = ks.kappa_of(L).astype(np.float64)
    if kappa.size != writer_in_dim(w, L, kind, expert):
        raise GneError("BAD_KEY", f"鍵 {ks.spec} の層 {L} の長さ {kappa.size} が {tensor} の入力と違います")
    info: dict = {}
    d_lora = write_vector(w, session.direction_set(dname), L, direction_layer=direction_layer, rms_a=ks.rms_a.get(L),
                          gamma=post_norm_gamma(w, L, kind), kind=kind, expert=expert, gate=ks.gate_a.get(L), info=info)
    d_eff = d_lora.astype(np.float64) * float(info["out_scale"])          # 書き込み行列の出力(norm の前)に入る向き(α = 1)
    dn = float(np.linalg.norm(d_eff))
    n_used = int(w.arch.layers[L].n_expert_used or 0) if kind == "expert" else 0

    texts = list(texts) if texts else list(CHECK_TEXTS)
    convs = [t if isinstance(t, str) else list(t) for t in texts]
    convs_b = [t if isinstance(t, str) else list(t) for t in (texts_b or [])]
    first = convs[0]
    ids0 = (e.tokenize(first, add_special=True, parse_special=False) if isinstance(first, str)
            else e.tokenize_prompt(e.render_chat(first)))
    probe, ctx = _probe(e)
    names = _names(e, probe, ctx, ids0[:16])
    x_name, y_name = _writer_names(session, names, L, kind)
    sel_name = moe_names(names, L).get("argsort") if kind == "expert" else None
    if x_name is None or y_name is None or (kind == "expert" and sel_name is None):
        raise GneError("ARCH_UNSUPPORTED", f"層 {L} の書き込み先の入力か出力が捕捉できません(入力 {x_name}、出力 {y_name}"
                                           + (f"、選ばれた専門家 {sel_name}" if kind == "expert" else "") + ")")
    wanted = [n for n in (y_name, x_name, sel_name) if n]

    def capture(cs, lora=None):
        if not cs:
            return []
        # 入力 x も名前で捕捉する(attn の head_out・bitnet の attn_sub_norm。3 次元で出る名前はトークンの行にそろえる)
        got = capture_attention(session, cs, layers=[L], want_pattern=False, want_head_out=False, extra=wanted, lora=lora)
        out = []
        for ac in got:
            for nm in wanted:
                if nm not in ac.extra:
                    raise GneError("ARCH_UNSUPPORTED", f"層 {L} の {nm} が捕捉できません")
            T = ac.n_tokens
            x = np.asarray(ac.extra[x_name], dtype=np.float64)
            y = np.asarray(ac.extra[y_name], dtype=np.float64)
            if kind == "expert":                    # [幅, n_used, T] → (T·n_used, 幅) → (T, n_used, 幅)
                if x.shape[0] != T * n_used or y.shape[0] != T * n_used:
                    raise GneError("ARCH_UNSUPPORTED", f"層 {L} の {x_name} / {y_name} の形 {x.shape} / {y.shape} が "
                                                       f"(トークン {T} × 専門家 {n_used}, 幅) と違います")
                out.append({"x": x.reshape(T, n_used, -1), "y": y.reshape(T, n_used, -1),
                            "sel": selected(np.asarray(ac.extra[sel_name]), n_used)})
            else:
                if x.shape[0] != T and x.size % T == 0:
                    x = x.reshape(T, -1)
                out.append({"x": x, "y": y})
        return out

    def rows_of(b, g):
        """(x, Δ) の行と、ほかの専門家の枠の Δ(expert だけ)、選ばれ方が変わったトークンの数。"""
        if kind != "expert":
            return b["x"][skip:], (g["y"] - b["y"])[skip:], None, 0
        hit = b["sel"] == int(expert)
        hit[:skip] = False
        other = ~(b["sel"] == int(expert))
        other[:skip] = False
        ti, tj = np.nonzero(hit)
        dy3 = g["y"] - b["y"]
        return b["x"][ti, tj], dy3[ti, tj], dy3[other], int(np.any(g["sel"] != b["sel"], axis=1).sum())
    try:
        base, base_b = capture(convs), capture(convs_b)
        for c in base + base_b:
            if c["x"].shape[-1] != kappa.size:
                raise GneError("ARCH_UNSUPPORTED", f"捕捉した {x_name} の幅 {c['x'].shape[-1]} が鍵の長さ {kappa.size} と違います")
        ex_leak = None
        if L in ks.exb:                                                 # 鍵の作成に使った側 B の例の平均(捕捉し直さない)
            ex_leak = float(np.abs(np.atleast_2d(ks.exb[L]).astype(np.float64) @ kappa).max())
        rows, path = [], None
        for a0 in alphas:
            a = sign * float(a0)
            rec = Recipe([Unit(L, None, 0.0, kind, mode="keyed", k=a, direction=dname, key=key, expert=expert)],
                         session.meta.model_id, name=f"v30-L{L}", direction_layer=direction_layer)
            path, ref = session.keyed_lora(rec)
            lora = [(str(path), float(ref))]
            got, got_b = capture(convs, lora), capture(convs_b, lora)
            s_all, rel_all, cos_all, other_max, changed = [], [], [], 0.0, 0
            for b, g in zip(base, got, strict=True):
                x, dy, oth, ch = rows_of(b, g)
                changed += ch
                if oth is not None and oth.size:
                    other_max = max(other_max, float(np.linalg.norm(oth, axis=-1).max()) / max(abs(a) * dn, 1e-30))
                s = x @ kappa
                pred = a * s[:, None] * d_eff[None, :]
                pn = np.linalg.norm(pred, axis=1)
                s_all.append(s)
                rel_all.append(np.linalg.norm(dy - pred, axis=1) / np.where(pn > 0, pn, np.inf))
                # 予測の向きは sign(α κᵀx)·d_eff(κᵀx が負のトークンでは逆向き)なので、その符号をそろえた cos
                cos_all.append(np.sign(a * s) * (dy @ d_eff) / np.maximum(np.linalg.norm(dy, axis=1) * dn, 1e-30))
            s_cat, rel_cat, cos_cat = np.concatenate(s_all), np.concatenate(rel_all), np.concatenate(cos_all)
            sa = np.abs(s_cat)
            thr = float(np.quantile(sa, 1.0 - top_frac)) if sa.size else 0.0
            big = (sa >= thr) & (sa > 0)
            rel = rel_cat[big]
            leak_tok, leak_ex = [], []
            for b, g in zip(base_b, got_b, strict=True):
                _, dy, _, _ = rows_of(b, g)
                if not dy.shape[0]:
                    continue
                leak_tok.append(np.linalg.norm(dy, axis=1) / max(abs(a) * dn, 1e-30))
                leak_ex.append(float(np.linalg.norm(dy.mean(axis=0)) / max(abs(a) * dn, 1e-30)))
            leak = max(leak_ex) if leak_ex else ex_leak
            row = {"alpha": a, "adapter_scale": float(ref), "n_tokens": int(sa.size), "n_big": int(big.sum()),
                   "kx_threshold": thr, "kx_max": float(sa.max()) if sa.size else None,
                   "rel_err_median": float(np.median(rel)) if rel.size else None,
                   "rel_err_p95": float(np.percentile(rel, 95)) if rel.size else None,
                   "cos_median": float(np.median(cos_cat[big])) if rel.size else None,
                   "leak_b": leak, "leak_b_source": "texts_b" if leak_ex else ("key_examples" if ex_leak is not None else None),
                   "leak_b_token_median": float(np.median(np.concatenate(leak_tok))) if leak_tok else None}
            ok = bool(rel.size and np.median(rel) < 0.03 and (leak is None or leak < 0.1))
            if kind == "expert":
                row.update({"other_max": other_max, "selection_changed": changed})
                ok = ok and other_max < 1e-3 and changed == 0
            row["ok"] = ok
            rows.append(row)
    finally:
        e.free_attn_probe()
    if path is not None:
        e.drop_lora(path)
    out = {"layer": L, "kind": kind, "expert": expert, "tensor": tensor, "key": ks.spec, "direction": dname,
           "direction_layer": direction_layer, "in_name": x_name, "out_name": y_name, "d_norm": dn,
           "direction_norm": info.get("direction_norm"), "post_norm": info.get("post_norm"), "kn": float(ks.kn[L]),
           "gate": info.get("gate"), "res_scale": info.get("res_scale"),
           "leak_b_key_examples": ex_leak, "rows": rows, "v30": all(r["ok"] for r in rows),
           "judged_leak": any(r["leak_b"] is not None for r in rows)}
    if kind == "expert" and not any(r["n_tokens"] for r in rows):
        out["note"] = (f"専門家 {expert} は texts で一度も選ばれなかったので測れません。専門家 {expert} が選ばれる文"
                       "(鍵の側 A の会話など)を texts に渡してください")
    return out
