# Patches for SenseNova-Loop (arch `sensenova_loop`)

The official llama.cpp (ggml-org) cannot load SenseNova-Loop-9B yet. gne uses the model team's fork
[OpenSenseNova/llama.cpp](https://github.com/OpenSenseNova/llama.cpp) at commit `24ea7bb78410e0f0f7bbb362e37c60cd9ecfc761`
(llama.cpp 1.0.0-dev of 2026-11-17 plus SenseNova-Loop support), with the official GGUF files
[sensenova/SenseNova-Loop-9B-GGUF](https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF).
The files here are patches for that fork and for the Python binding. They are not part of the fork and were not sent to its
authors. The full setup is in [docs/sensenova_loop.md](../../docs/sensenova_loop.md).

| File | Applies to | What it changes |
|---|---|---|
| `0001-sensenova-loop-graph-names-for-gne.patch` | the fork at `24ea7bb` | Names only, the numbers do not change (last-token logits are bit-identical). Inside the looped layers the fork computes `l_out` (the mean of the two residual streams) but nothing reads it, so the node is dropped from the graph; the patch keeps it. It also names the FFN input (`attn_residual`, and `attn_post_norm` after the norm, as in qwen35), the 8 router logits (`mhc_attn_gates`, `mhc_ffn_gates`) and the two streams (`mhc_stream_a`, `mhc_stream_b`). |
| `0002-sensenova-loop-per-visit-weight-names.patch` | the fork at `24ea7bb` (after 0001) | The second run of layers 4-27 (execution steps 28-51) reuses the first run's tensors under the same names, so a LoRA always acts on both runs. The patch adds views named `blk.28.*` … `blk.51.*` for the matmul weights (including the fused `attn_qkv_gate`, `ssm_beta_alpha` and the routers). The bytes are shared, nothing is copied. `LLAMA_SENSENOVA_SHARED_NAMES=1` turns the views off. |
| `llama-cpp-python-0.3.36-llama-cpp-1.0.patch` | llama-cpp-python `v0.3.36`, files `llama_cpp/_ctypes_extensions.py` and `llama_cpp/llama_cpp.py` | **Required.** llama.cpp 1.0 removed 36 deprecated functions, so llama-cpp-python 0.3.36 fails at import; a missing function now fails only when called. Two ctypes structures follow the fork's `include/llama.h` (`llama_context_params`: no `defrag_thold`, new `moe_cache_size` and `moe_cache_cpu_share`; `llama_perf_context_data`: `n_reallocs_same`, `n_reallocs_diff`). |

## Which gne features need which patch

| Feature | Fork as published | + 0001 | + 0001 + 0002 |
|---|---|---|---|
| Load, chat, OpenAI-compatible server, control vectors at every step (the fork adds them to both streams) | yes | yes | yes |
| Residuals outside the loop and at the end of each run (steps 0-3, 27, 51-55): `--score-layers loop` (27, 51, 54), CHiPS heads pipeline with those score layers, `gne heads measure`, `gne heads loop-merge` | yes | yes | yes |
| Residuals inside the loop (steps 4-26, 28-50): directions over all layers, base-layer search, `gne verify` there, FiSH (`gne nw …`) there, router values (`mhc_*_gates`) and the two streams (`l_out_streams`) | no (gne stops with the reason and the fix) | yes | yes |
| Edit one run only: `GNE_LOOP_EDITS=per_visit`, `gne heads causal --visits first/second`, `gne heads visits`, `gne heads loop-pipeline`, recipes and LoRAs recorded as `per_visit` | no (gne stops with the reason and the fix) | no | yes |

Recipes made with `per_visit` can be turned into recipes that apply to both runs with `gne heads loop-merge`; those run on the fork as published.

## Apply and build

```bash
git clone https://github.com/OpenSenseNova/llama.cpp llama.cpp-sensenova
cd llama.cpp-sensenova
git checkout 24ea7bb78410e0f0f7bbb362e37c60cd9ecfc761
git apply /path/to/gguf-neuron-explorer/third_party/sensenova-loop/0001-sensenova-loop-graph-names-for-gne.patch
git apply /path/to/gguf-neuron-explorer/third_party/sensenova-loop/0002-sensenova-loop-per-visit-weight-names.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF
cmake --build build -j 8 --target llama ggml mtmd llama-completion llama-cli
```

`git apply --check` passes for both patches on a clean checkout of `24ea7bb`, and the patched tree is the one gne was tested with.
`-DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF` keep the build from downloading anything.

## Older patches

`legacy-guideboardlabs/` holds the patches published with gne 0.7.0 for the earlier third-party fork
(GuideboardLabs/llama.cpp-sensenova-loop @ `a7f77a7`) and its GGUF files. That fork and the official one cannot read each
other's GGUF files, and the official GGUF is the reference now. gne still recognises the older GGUF metadata, but that setup is
no longer tested. The files stay because 0.7.0 pointed to them.

## License

llama.cpp is MIT-licensed (Copyright (c) 2023-2026 The ggml authors), and so is llama-cpp-python (Copyright (c) 2023 Andrei Betlen).
The patches in this folder are provided under the same MIT license, as part of gne (see [LICENSE](../../LICENSE)).
