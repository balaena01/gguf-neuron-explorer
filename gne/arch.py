"""層ごとの構造情報(mixer の種類・head 数・MLP / MoE の構成)。別冊 UI 設計 §3。

GGUF のメタデータとテンソル名だけから組み立てる(重みは読まない)。マップの描画(§4)の材料。

判定の順番(§3.1 の表を、この版の llama.cpp で確かめて直したもの、verification.md F32):
1. `{arch}.full_attention_interval` があれば優先: (L + 1) % interval == 0 の層が attention、他は linear_attention。
2. `blk.L.ssm_*` があれば、ssm_alpha / ssm_beta(Gated DeltaNet: Qwen3.5 / Qwen3-Next)は linear_attention、
   それ以外(Mamba 系)は recurrent。Qwen3.5 の線形注意の層は attn_qkv も持つので、attn_* より先に見る。
3. attn_q / attn_qkv / attn_q_a(MLA)があれば attention。time_mix_*(RWKV)は recurrent。
4. どれも無ければ none。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

Mixer = Literal["attention", "linear_attention", "recurrent", "none"]
Ffn = Literal["dense", "moe", "moe+shared", "none"]


@dataclass
class LayerInfo:
    index: int
    mixer: Mixer
    n_head: int
    n_head_kv: int
    ffn: Ffn
    n_ff: int                  # dense: ffn_down の入力数 / moe: 専門家 1 個あたり
    n_expert: int
    n_expert_used: int
    n_ff_shexp: int            # shared expert の n_ff(無ければ 0)
    has_post_ffn_norm: bool
    # CHANGE-04 ヘッド: attention は n_head(Q 側)・n_head_kv、linear_attention は n_head = V ヘッド数(ssm.time_step_rank)・
    # n_head_kv = K ヘッド数(ssm.group_count)。head_dim は 1 ヘッドの出力の幅(書き込み行列の入力 / n_head)
    head_dim: int = 0
    head_writer: str | None = None   # ヘッドの出力を残差に書く行列(attn_output / ssm_out)
    head_out: tuple[str, ...] = ()   # 書き込み行列に入る直前のテンソル名の候補(捕捉で見えた最初のもの)


@dataclass
class ArchInfo:
    arch: str
    n_layer: int
    n_embd: int
    n_vocab: int
    layers: list[LayerInfo]
    grid_side: int             # シートの一辺の最大(§4.1)

    def to_dict(self) -> dict:
        return asdict(self)


def _per_layer(v, L: int, default: int = 0) -> int:
    if v is None:
        return default
    if isinstance(v, list):
        return int(v[L]) if L < len(v) else default
    return int(v)


def sheet_side(li: LayerInfo) -> int:
    """層の MLP シートの一辺(マップの格子)。MoE で専門家が 16 を超えるときは、専門家ごとの正方形のタイルを並べた 1 枚。"""
    if li.ffn == "dense":
        return math.ceil(math.sqrt(li.n_ff))
    if li.ffn.startswith("moe"):
        tile = math.ceil(math.sqrt(li.n_ff))
        if li.n_expert <= 16:
            return tile
        return tile * math.ceil(math.sqrt(li.n_expert))
    return 0


def arch_info(weights) -> ArchInfo:
    m = weights.meta
    arch = m.arch
    get = weights.field
    by_layer: dict[int, dict[str, object]] = {}
    for name, t in weights._tensors.items():
        if not name.startswith("blk."):
            continue
        _, L, rest = name.split(".", 2)
        if L.isdigit():
            by_layer.setdefault(int(L), {})[rest] = t
    interval = get(f"{arch}.full_attention_interval")
    head = get(f"{arch}.attention.head_count")
    head_kv = get(f"{arch}.attention.head_count_kv")
    layers = []
    for L in range(m.n_layer):
        ts = by_layer.get(L, {})
        names = set(ts)

        def has(*xs, names=names):
            return any(any(n.startswith(x) for n in names) for x in xs)
        if has("ssm_"):
            mixer: Mixer = "linear_attention" if has("ssm_alpha", "ssm_beta") else "recurrent"
        elif has("attn_q.", "attn_qkv.", "attn_q_a."):
            mixer = "attention"
        elif has("time_mix_"):
            mixer = "recurrent"
        else:
            mixer = "none"
        if interval and mixer in ("attention", "linear_attention"):
            mixer = "attention" if (L + 1) % int(interval) == 0 else "linear_attention"
        if mixer == "attention":
            nh, nkv = _per_layer(head, L), _per_layer(head_kv, L, _per_layer(head, L))
        elif mixer in ("linear_attention", "recurrent"):
            nh = int(get(f"{arch}.ssm.time_step_rank", 0) or 0)
            nkv = int(get(f"{arch}.ssm.group_count", 0) or 0)
        else:
            nh = nkv = 0
        shexp = ts.get("ffn_down_shexp.weight")
        n_shexp = int(shexp.shape[0]) if shexp is not None else 0
        if "ffn_down_exps.weight" in ts:
            t = ts["ffn_down_exps.weight"]                      # ne = [n_ff_exp, n_embd, n_expert]
            ffn: Ffn = "moe+shared" if n_shexp else "moe"
            n_ff, n_exp = int(t.shape[0]), int(t.shape[2]) if len(t.shape) > 2 else int(get(f"{arch}.expert_count", 0))
            used = int(get(f"{arch}.expert_used_count", 0) or 0)
        elif "ffn_down.weight" in ts:
            ffn, n_ff, n_exp, used = "dense", int(ts["ffn_down.weight"].shape[0]), 0, 0
        else:
            ffn, n_ff, n_exp, used = "none", 0, 0, 0
        writer = ("attn_output" if mixer == "attention" else "ssm_out" if mixer == "linear_attention" else None)
        wt = ts.get(f"{writer}.weight") if writer else None
        hd = int(wt.shape[0]) // nh if wt is not None and nh else 0          # ne = [n_head·head_dim, n_embd]
        outs = (("attn_gated", "kqv_out") if mixer == "attention" else ("final_output",) if mixer == "linear_attention" else ())
        layers.append(LayerInfo(L, mixer, nh, nkv, ffn, n_ff, n_exp, used, n_shexp,
                                "post_ffw_norm.weight" in ts, hd, writer if wt is not None else None, outs))
    side = max([sheet_side(li) for li in layers] + [1])
    return ArchInfo(arch, m.n_layer, m.n_embd, m.n_vocab, layers, side)
