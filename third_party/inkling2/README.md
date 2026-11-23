# Patch for Inkling 2 (arch `inkling2_loop`)

ggml-org/llama.cpp reads Inkling-2-Mini from commit `29b52d3` (PR #32694). PR #32704, which fixes the second and third passes running without the per-expert scales, was merged on 2026-11-21 (merge commit `690b87d8`); build from that commit or later. The setup is in [docs/inkling2.md](../../docs/inkling2.md).

| File | Applies to | What it changes |
|---|---|---|
| `0001-inkling2-graph-names-for-gne.patch` | `29b52d3` + PR #32704 (tested); also applies with `git apply --check` to master `44d50873` (2026-11-22), not rebuilt there | Names only: no operation is added or changed. The logits were bit-identical with and without the patch (4 prompts from a public template, last-position logits and 24 greedy tokens, on the CPU and on the GPU). gne needs the names to read the model's internals (`gne inkling …`); loading, `gne info` and chat work without it. |

Names added (all with the cache slot as `-N`, which gne turns into the execution step):

| Name | Tensor |
|---|---|
| `attn_mix`, `attn_mix_w` | the attention input mixed over the stored streams and the current one, and the softmax weights over those states (the stored streams in reverse order, then the current one). Full-attention layers have three rows (Q, K, V) |
| `ffn_mix`, `ffn_mix_w` | the same for the FFN input |
| `out_mix`, `out_mix_w` | the same for the final output, before the output norm (no `-N`) |
| `loop_msg`, `loop_msg_w` | passes 2 and 3: the message mixed from the previous pass (four rows: Q, K, V, FFN) and its weights |
| `loop_gain_{q,k,v,ffn}`, `loop_forget_{q,k,v,ffn}` | passes 2 and 3: the gate values; input ← (1 − forget)·input + gain·message |
| `attn_in_{q,k,v}`, `ffn_in` | passes 2 and 3: the inputs after the gate, before the norm |
| `attn_norm`, `attn_norm_{q,k,v}`, `ffn_norm` | the normalized inputs (one `attn_norm` when Q, K and V share it) |
| `attn_gated` | the head outputs after the sigmoid gate, the input of W_o |
| `attn_wo` | the W_o output, before the short convolution (`attn_out` is after it) |
| `embd_norm` | the normalized token embeddings, the first stored stream (no `-N`) |

```bash
git clone https://github.com/ggml-org/llama.cpp llama.cpp-inkling2
cd llama.cpp-inkling2
git apply /path/to/gguf-neuron-explorer/third_party/inkling2/0001-inkling2-graph-names-for-gne.patch   # on 690b87d8 or later
```

The patch changes `src/models/inkling2.cpp`, `src/models/inkling2-loop.cpp` and `src/models/models.h` only, so only
libllama needs rebuilding; a build without CUDA (`-DGGML_CUDA=OFF`, target `llama`) can be used next to the CUDA ggml
libraries of an unpatched build from the same commit.
