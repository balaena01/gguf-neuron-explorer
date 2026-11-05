"""CPU で動く小さな qwen4exp の GGUF を作る。重みは乱数。実物は最小でも 66 GB あって手元に載らないので、グラフと gne のつなぎ目はこれで確かめる。

形は実物(unsloth/Qwen3.8-Flash-Next-GGUF)のヘッダーと llama.cpp の load_arch_tensors に合わせ、大きさだけを小さくしている。
python make_tiny_qwen4exp.py <出力.gguf>
"""
from __future__ import annotations

import sys
from pathlib import Path

import gguf
import numpy as np

# 実物と同じ形の文字列を描くための短いチャットテンプレート(thinking の切り替えも実物に合わせる)
TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% else %}<think>\n{% endif %}{% endif %}"
)


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs, strict=True)}


def write(path: str | Path, *, n_layer: int = 8, seed: int = 0, n_embd: int = 64, split_tensors: int = 0) -> Path:
    """split_tensors を渡すと実物と同じ分割(1 番目はメタデータだけ)で書き、1 番目のファイルのパスを返す。"""
    arch = "qwen4exp"
    kw = dict(split_max_tensors=split_tensors, small_first_shard=True) if split_tensors else {}
    w = gguf.GGUFWriter(str(path), arch, **kw)
    rng = np.random.default_rng(seed)
    hc, lr = 4, 8
    head, kv, hd = 4, 2, 8               # attention: Q 側のヘッド数、K / V 側のヘッド数、1 ヘッドの幅
    d_state, n_group, dt_rank = 8, 2, 4  # 線形注意: 1 ヘッドの幅、K 側のグループ、V 側のヘッド数
    n_exp, n_used, n_ff = 8, 2, 16
    ple_dim, ple_ngram, ple_per = n_embd // 16, 3, 8  # 16 ヘッド × 幅 = n_embd
    ple_heads = (ple_ngram - 1) * ple_per
    ple_size = 101
    idx_heads, idx_dim, idx_top = 2, 8, 16

    units = _bytes_to_unicode()
    tokens = [units[b] for b in range(256)]
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|image_pad|>"]
    ids = {s: 256 + i for i, s in enumerate(specials)}
    types = [gguf.TokenType.NORMAL] * 256 + [gguf.TokenType.CONTROL] * 3 + [gguf.TokenType.USER_DEFINED] * 3
    tokens += specials
    n_vocab = len(tokens)

    w.add_name("Tiny qwen4exp")
    w.add_block_count(n_layer)
    w.add_context_length(4096)
    w.add_embedding_length(n_embd)
    w.add_head_count(head)
    w.add_head_count_kv(kv)
    w.add_key_length(hd)
    w.add_value_length(hd)
    w.add_rope_dimension_count(hd)
    w.add_array(f"{arch}.rope.dimension_sections", [2, 1, 1, 0])
    w.add_float32(f"{arch}.rope.freq_base", 1e7)
    w.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_uint32(f"{arch}.expert_count", n_exp)
    w.add_uint32(f"{arch}.expert_used_count", n_used)
    w.add_uint32(f"{arch}.expert_feed_forward_length", n_ff)
    w.add_uint32(f"{arch}.expert_shared_feed_forward_length", n_ff)
    w.add_uint32(f"{arch}.ssm.conv_kernel", 4)
    w.add_uint32(f"{arch}.ssm.state_size", d_state)
    w.add_uint32(f"{arch}.ssm.group_count", n_group)
    w.add_uint32(f"{arch}.ssm.time_step_rank", dt_rank)
    w.add_uint32(f"{arch}.ssm.inner_size", d_state * dt_rank)
    w.add_uint32(f"{arch}.full_attention_interval", 4)
    w.add_uint32(f"{arch}.hyper_connection.count", hc)
    w.add_uint32(f"{arch}.hyper_connection.low_rank", lr)
    w.add_uint32(f"{arch}.attention.indexer.head_count", idx_heads)
    w.add_uint32(f"{arch}.attention.indexer.key_length", idx_dim)
    w.add_uint32(f"{arch}.attention.indexer.top_k", idx_top)
    w.add_array(f"{arch}.attention.compress_ratios", [4 if (L + 1) % 4 == 0 else 0 for L in range(n_layer)])
    w.add_array(f"{arch}.ple.layers", [1])
    w.add_uint32(f"{arch}.ple.ngram_size", ple_ngram)
    w.add_uint32(f"{arch}.ple.heads_per_ngram", ple_per)
    w.add_uint32(f"{arch}.ple.conv_kernel", 4)
    w.add_uint32(f"{arch}.ple.eos_token_id", ids["<|endoftext|>"])
    w.add_uint32(f"{arch}.ple.image_token_id", ids["<|image_pad|>"])
    w.add_uint32(f"{arch}.embedding_length_per_layer_input", ple_dim)
    u64 = lambda k, v: w.add_key_value(f"{arch}.{k}", v, gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.UINT64)
    u64("ple.layer_multipliers", [23703573157769, 20109073645365, 8052911324071])
    u64("ple.head_offsets", [h * ple_size for h in range(ple_heads)])
    u64("ple.head_vocab_sizes", [ple_size] * ple_heads)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen35")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(["t h", "th e"])
    w.add_eos_token_id(ids["<|im_end|>"])
    w.add_bos_token_id(ids["<|endoftext|>"])
    w.add_pad_token_id(ids["<|endoftext|>"])
    w.add_add_bos_token(False)
    w.add_chat_template(TEMPLATE)

    def t(name, *ne, one=False):
        # ne は ggml の順。numpy は逆順
        a = np.ones(ne[::-1], np.float32) if one else (rng.standard_normal(ne[::-1]) * 0.05).astype(np.float32)
        w.add_tensor(name, a)

    t("token_embd.weight", n_embd, n_vocab)
    t("output.weight", n_embd, n_vocab)
    t("per_layer_token_embd.weight", ple_dim, ple_size * ple_heads)
    t("output_hc_norm.weight", n_embd * hc, one=True)
    t("output_hc_down.weight", n_embd * hc, lr)
    t("output_hc_up.weight", lr, n_embd * hc)
    for L in range(n_layer):
        p = f"blk.{L}."
        for part in ("attn", "ffn"):
            t(f"{p}hc_{part}_norm.weight", n_embd * hc, one=True)
            t(f"{p}hc_{part}_down.weight", n_embd * hc, lr)
            t(f"{p}hc_{part}_up.weight", lr, n_embd * hc)
            t(f"{p}hc_{part}_inject.weight", n_embd * hc, hc)
        if (L + 1) % 4 == 0:
            t(f"{p}attn_q.weight", n_embd, hd * head * 2)
            t(f"{p}attn_k.weight", n_embd, hd * kv)
            t(f"{p}attn_v.weight", n_embd, hd * kv)
            t(f"{p}attn_output.weight", hd * head, n_embd)
            t(f"{p}attn_q_norm.weight", hd, one=True)
            t(f"{p}attn_k_norm.weight", hd, one=True)
            t(f"{p}indexer.q_proj.weight", n_embd, idx_heads * idx_dim)
            t(f"{p}indexer.k_proj.weight", n_embd, idx_dim)
            t(f"{p}indexer.q_norm.weight", idx_dim, one=True)
            t(f"{p}indexer.k_norm.weight", idx_dim, one=True)
        else:
            key, value = d_state * n_group, d_state * dt_rank
            t(f"{p}attn_qkv.weight", n_embd, key * 2 + value)
            t(f"{p}attn_gate.weight", n_embd, value)
            t(f"{p}ssm_conv1d.weight", 4, key * 2 + value)
            t(f"{p}ssm_dt.bias", dt_rank)
            t(f"{p}ssm_a", dt_rank)
            t(f"{p}ssm_beta.weight", n_embd, dt_rank)
            t(f"{p}ssm_alpha.weight", n_embd, dt_rank)
            t(f"{p}ssm_norm.weight", d_state, one=True)
            t(f"{p}ssm_out.weight", value, n_embd)
        if L == 1:
            t(f"{p}ple_key.weight", n_embd, n_embd * hc)
            t(f"{p}ple_value.weight", n_embd, n_embd)
            for x in ("key", "query", "conv"):
                t(f"{p}ple_norm_{x}.weight", n_embd * hc, one=True)
            t(f"{p}ple_conv1d.weight", 4, n_embd * hc)
        t(f"{p}ffn_gate_inp.weight", n_embd, n_exp)
        t(f"{p}ffn_down_exps.weight", n_ff, n_embd, n_exp)
        t(f"{p}ffn_gate_exps.weight", n_embd, n_ff, n_exp)
        t(f"{p}ffn_up_exps.weight", n_embd, n_ff, n_exp)
        t(f"{p}ffn_gate_inp_shexp.weight", n_embd)
        t(f"{p}ffn_gate_shexp.weight", n_embd, n_ff)
        t(f"{p}ffn_up_shexp.weight", n_embd, n_ff)
        t(f"{p}ffn_down_shexp.weight", n_ff, n_embd)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    if split_tensors:
        p = Path(path)
        return next(sorted(p.parent.glob(f"{p.stem}-00001-of-*.gguf")).__iter__())
    return Path(path)


if __name__ == "__main__":
    print(write(sys.argv[1]).stat().st_size)
