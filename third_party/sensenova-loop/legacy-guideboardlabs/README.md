# Patches for the GuideboardLabs fork (gne 0.7.0, no longer tested)

These are the patches gne 0.7.0 published for
[GuideboardLabs/llama.cpp-sensenova-loop](https://github.com/GuideboardLabs/llama.cpp-sensenova-loop) at commit
`a7f77a788893ef2cb17c82b9ac0cb1279bcd0918` and its GGUF files (guideboardlabs/SenseNova-Loop-9B-GGUF).

- `0001-sensenova-loop-control-vector-on-both-mHC-streams.patch`: control vector on both streams, `l_out` = stream mean
- `0002-sensenova-loop-per-visit-weight-names.patch`: per-visit weight names for the second run
- `llama-cpp-python-0.3.36-structs.patch`: two ctypes structures for that fork's `llama.h` (llama.cpp 0.8)

gne now targets the official fork and GGUF files ([../README.md](../README.md)). The two forks cannot read each other's GGUF
files. gne still reads the older GGUF metadata (`sensenova_loop.mhc.layer_kinds`), but nothing here is tested any more.
MIT, as in [../README.md](../README.md).
