# Patches for SenseNova-Loop (arch `sensenova_loop`)

The official llama.cpp (ggml-org) cannot load SenseNova-Loop-9B yet. gne supports two llama.cpp forks that can, each with its
own GGUF files. The two forks cannot read each other's GGUF files (different metadata and tensor names; the weights are almost
the same). Pick one; the full setup is in [docs/sensenova_loop.md](../../docs/sensenova_loop.md).

| | Official fork (`official/`) | GuideboardLabs fork (`guideboardlabs/`) |
|---|---|---|
| llama.cpp | [OpenSenseNova/llama.cpp](https://github.com/OpenSenseNova/llama.cpp) @ `24ea7bb78410e0f0f7bbb362e37c60cd9ecfc761` (llama.cpp 1.0.0-dev, ggml-org `9214bdb` of 2026-11-17) | [GuideboardLabs/llama.cpp-sensenova-loop](https://github.com/GuideboardLabs/llama.cpp-sensenova-loop) @ `a7f77a788893ef2cb17c82b9ac0cb1279bcd0918` (llama.cpp 0.8.0, ggml-org `aa588a3` of 2026-10-30) |
| GGUF it reads | [sensenova/SenseNova-Loop-9B-GGUF](https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF) (BF16, Q8_0, Q5_K_M, Q4_K_M) | [guideboardlabs/SenseNova-Loop-9B-GGUF](https://huggingface.co/guideboardlabs/SenseNova-Loop-9B-GGUF) (Q8_0, Q4_K_M, Q4_0, Q3_K_M) |
| Decode speed, Q4_K_M (RTX 3080, llama-completion, context 8,192, 256 tokens) | 11.4 tok/s (11.5 with both patches) | 56.5 tok/s |
| Prompt speed (20 tokens) | 45 tok/s (41 with both patches) | 294 tok/s |
| GPU memory (above the display) | 6.6 GB (7.0 GB with both patches) | 6.1 GB |
| `gne heads pipeline --rank causal`, question_kinds, the selection used up to v0.7.0 (144 heads stopped; both forks with their 0001 + 0002) | about 52 min (causal screening 35 min) | about 5 min (causal screening 4 min) |
| Control vectors inside the two residual streams (steps 4-26, 28-50) | reach the next layer as published | need `guideboardlabs/0001` |
| Residuals inside the streams (directions over all layers, base-layer search, FiSH there, router values) | need `official/0001` | readable as published (gne rebuilds them from the merge outputs) |
| Edit one run only (`per_visit`, `gne heads visits`, `gne heads loop-pipeline`) | need `official/0001` + `official/0002` | need `guideboardlabs/0002` (after `guideboardlabs/0001`) |
| llama-cpp-python 0.3.36 | `official/llama-cpp-python-0.3.36-llama-cpp-1.0.patch` (required) | `guideboardlabs/llama-cpp-python-0.3.36-structs.patch` (required) |
| MTP head (speculative decoding) | in the GGUF; the fork supports `--spec-type draft-mtp` | not in the GGUF |
| Vision projector (mmproj) | `mmproj-SenseNova-Loop-9B-F16.gguf` in the GGUF repository | not provided |

The official fork computes the two-stream routers with generic ggml operations (no model-specific GPU kernels, as its
docs/models.md says), which is why it decodes about 5 times slower here. On 64 template prompts the two setups (each fork with
its own Q4_K_M) agreed on the first token's top-1 91% of the time, KL mean 0.027: numeric differences of the attention kernels
between llama.cpp 0.8 and 1.0 plus small differences of the GGUF files.

## Official fork (`official/`)

| File | Applies to | What it changes |
|---|---|---|
| `0001-sensenova-loop-graph-names-for-gne.patch` | the fork at `24ea7bb` | Names only, the numbers do not change (logits bit-identical). Inside the looped layers the fork computes `l_out` (the mean of the two streams) but nothing reads it, so the node is dropped from the graph; the patch keeps it. It also names the FFN input (`attn_residual`, and `attn_post_norm` after the norm), the 8 router logits (`mhc_attn_gates`, `mhc_ffn_gates`) and the two streams (`mhc_stream_a`, `mhc_stream_b`). |
| `0002-sensenova-loop-per-visit-weight-names.patch` | the fork at `24ea7bb`, after 0001 | Views named `blk.28.*` … `blk.51.*` for the weights the second run shares with the first (including the fused `attn_qkv_gate`, `ssm_beta_alpha` and the routers), so a LoRA can name one run. Nothing is copied. `LLAMA_SENSENOVA_SHARED_NAMES=1` turns the views off. |
| `llama-cpp-python-0.3.36-llama-cpp-1.0.patch` | llama-cpp-python `v0.3.36` (`llama_cpp/_ctypes_extensions.py`, `llama_cpp/llama_cpp.py`) | llama.cpp 1.0 removed 36 deprecated functions, so 0.3.36 fails at import; a missing function now fails only when called. `llama_context_params` (no `defrag_thold`, new `moe_cache_size` and `moe_cache_cpu_share`) and `llama_perf_context_data` (`n_reallocs_same`, `n_reallocs_diff`) follow the fork's `include/llama.h`. |

```bash
git clone https://github.com/OpenSenseNova/llama.cpp llama.cpp-sensenova
cd llama.cpp-sensenova
git checkout 24ea7bb78410e0f0f7bbb362e37c60cd9ecfc761
git apply /path/to/gguf-neuron-explorer/third_party/sensenova-loop/official/0001-sensenova-loop-graph-names-for-gne.patch
git apply /path/to/gguf-neuron-explorer/third_party/sensenova-loop/official/0002-sensenova-loop-per-visit-weight-names.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF
cmake --build build -j 8 --target llama ggml mtmd llama-completion llama-cli
```

## GuideboardLabs fork (`guideboardlabs/`)

| File | Applies to | What it changes |
|---|---|---|
| `0001-sensenova-loop-control-vector-on-both-mHC-streams.patch` | the fork at `a7f77a7` | Inside the looped layers the fork adds a control vector to stream a only, and the next sublayer drops it. The patch adds it to both streams (the same as adding it to their mean) and makes `l_out` the stream mean (contiguous). |
| `0002-sensenova-loop-per-visit-weight-names.patch` | the fork at `a7f77a7`, after 0001 | Views named `blk.28.*` … `blk.51.*` for the matmul weights of the second run, so a LoRA can name one run. Nothing is copied. `LLAMA_SENSENOVA_SHARED_NAMES=1` turns the views off. |
| `llama-cpp-python-0.3.36-structs.patch` | llama-cpp-python `v0.3.36` (`llama_cpp/llama_cpp.py`) | Two ctypes structures follow the fork's `include/llama.h`: `llama_context_params.moe_cache_size`, `llama_perf_context_data.n_reallocs_same` / `n_reallocs_diff`. |

```bash
git clone https://github.com/GuideboardLabs/llama.cpp-sensenova-loop
cd llama.cpp-sensenova-loop
git checkout a7f77a788893ef2cb17c82b9ac0cb1279bcd0918
git apply /path/to/gguf-neuron-explorer/third_party/sensenova-loop/guideboardlabs/0001-sensenova-loop-control-vector-on-both-mHC-streams.patch
git apply /path/to/gguf-neuron-explorer/third_party/sensenova-loop/guideboardlabs/0002-sensenova-loop-per-visit-weight-names.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF
cmake --build build -j 8 --target llama ggml mtmd llama-server llama-completion
```

All patches pass `git apply --check` on clean checkouts of the commits above (llama-cpp-python: the `v0.3.36` tag), and the
patched trees are the ones gne was tested with. They are not part of the forks and were not sent to their authors.

## License

llama.cpp is MIT-licensed (Copyright (c) 2023-2026 The ggml authors), and so is llama-cpp-python (Copyright (c) 2023 Andrei Betlen).
The patches in this folder are provided under the same MIT license, as part of gne (see [LICENSE](../../LICENSE)).
