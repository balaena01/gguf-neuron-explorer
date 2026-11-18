# Patches for SenseNova-Loop (arch `sensenova_loop`)

The official llama.cpp cannot load SenseNova-Loop-9B. gne uses the third-party fork
[GuideboardLabs/llama.cpp-sensenova-loop](https://github.com/GuideboardLabs/llama.cpp-sensenova-loop) at commit
`a7f77a788893ef2cb17c82b9ac0cb1279bcd0918` (branch `sensenova-loop`, based on llama.cpp `aa588a3`).
The files here are optional patches for that fork, and one required patch for the Python binding.
They are not part of the fork and were not sent to its author. The full setup is in
[docs/sensenova_loop.md](../../docs/sensenova_loop.md).

| File | Applies to | What it changes |
|---|---|---|
| `0001-sensenova-loop-control-vector-on-both-mHC-streams.patch` | the fork at `a7f77a7` | Inside the looped layers the residual is two streams. The fork adds a control vector to stream a only, and the next sublayer drops it. The patch adds it to both streams, which is the same as adding it to their mean, and makes `l_out` the stream mean (contiguous). |
| `0002-sensenova-loop-per-visit-weight-names.patch` | the fork at `a7f77a7` (after 0001) | The second run of layers 4-27 (execution steps 28-51) reuses the first run's tensors under the same names, so a LoRA always acts on both runs. The patch adds views named `blk.28.*` … `blk.51.*` for the matmul weights. The bytes are shared, nothing is copied. Set `LLAMA_SENSENOVA_SHARED_NAMES=1` to turn the views off. |
| `llama-cpp-python-0.3.36-structs.patch` | llama-cpp-python `v0.3.36`, file `llama_cpp/llama_cpp.py` | Two ctypes structures differ from the fork's `include/llama.h`: `llama_context_params.moe_cache_size` and `llama_perf_context_data.n_reallocs_same` / `n_reallocs_diff`. **Required** for any use, patched fork or not. |

## Which gne features need which patch

| Feature | Fork as published | + 0001 | + 0002 |
|---|---|---|---|
| Capture, directions, CHiPS heads pipeline, FiSH, `gne nw verify` / `check`, LoRA export (edits act on both runs) | yes | yes | yes |
| Measure per run (`--score-layers loop`, `--select-layer`), `gne heads loop-merge`, `gne heads measure` | yes | yes | yes |
| Control vectors on steps 4-26 and 28-50 (inside the two streams); `gne verify` V3, V4, V12 there; FiSH direction step with stacking (without it, use `--no-stack`) | no (gne stops with the reason) | yes | yes |
| Edit one run only: `GNE_LOOP_EDITS=per_visit`, `gne heads causal --visits first/second`, `gne heads visits`, `gne heads loop-pipeline`, recipes and LoRAs recorded as `per_visit` | no (gne stops with the reason and the fix) | no | yes |

Recipes made with `per_visit` can be turned into recipes that apply to both runs with `gne heads loop-merge`; those run on the fork as published and on any build of it.

## Apply and build

```bash
git clone https://github.com/GuideboardLabs/llama.cpp-sensenova-loop
cd llama.cpp-sensenova-loop
git checkout a7f77a788893ef2cb17c82b9ac0cb1279bcd0918
git apply /path/to/gguf-neuron-explorer/third_party/sensenova-loop/0001-sensenova-loop-control-vector-on-both-mHC-streams.patch
git apply /path/to/gguf-neuron-explorer/third_party/sensenova-loop/0002-sensenova-loop-per-visit-weight-names.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON
cmake --build build -j 8 --target llama ggml mtmd llama-server llama-completion
```

`git apply --check` passes for both patches on a clean checkout of `a7f77a7`, and the patched tree is the one gne was tested with.
Skip the two `git apply` lines for the fork as published.

## License

llama.cpp is MIT-licensed (Copyright (c) 2023-2026 The ggml authors), and so is llama-cpp-python (Copyright (c) 2023 Andrei Betlen).
The patches in this folder are provided under the same MIT license, as part of gne (see [LICENSE](../../LICENSE)).
