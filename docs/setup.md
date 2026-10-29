# 環境別の手順

gne は llama-cpp-python の上で動く。GPU を使うには、llama-cpp-python を GPU のバックエンドつきでソースからビルドする。ここでは WSL2(Ubuntu)で NVIDIA(CUDA)と AMD(ROCm)を使う手順と、GPU ごとに別の環境を作ってワーカー(`gne worker`)で仕事を分ける手順を書く。

確かめた環境: WSL2 + RTX 3080 12 GB、CUDA 12.4、Python 3.12、llama-cpp-python 0.3.36(ソースからビルド)。**ROCm(Radeon RX 7900 XTX)の手順はこの環境では確かめていない**(AMD と llama.cpp の手順書に沿って書いたもの)。

## 目次
1. 共通
2. WSL + CUDA(NVIDIA)
3. WSL + ROCm(AMD)
4. GPU ごとの環境とワーカー
5. 確認

## 1. 共通

- Python 3.11 以上と [uv](https://docs.astral.sh/uv/)。
- **モデルは WSL 側のディスク(ext4、`~/` の下など)に置く。** `/mnt/c` などの Windows 側のドライブは 9P 経由で読むので、GGUF の読み込みと `gne lens logit` のような重みの読み出しが遅くなる。
- `gne` の結果の置き場所(`~/.gne`、`--root` / `GNE_ROOT`)も WSL 側に置く。複数の環境で同じ置き場所を共有できる(4 章)。
- llama.cpp が新しい arch に対応していないときは、llama-cpp-python の vendor の llama.cpp を新しくしてビルドし直す。C の構造体が変わっていればバインディングも直す(`.claude/skills/gne/references/new_model.md`)。

## 2. WSL + CUDA(NVIDIA)

1. Windows 側に、WSL に対応した NVIDIA のドライバを入れる。WSL の中に Linux 用のドライバは入れない(Windows のドライバが `/usr/lib/wsl/lib` に見える)。
2. WSL の中で `nvidia-smi` を実行し、GPU とドライバの版が見えることを確かめる。
3. WSL の中に CUDA Toolkit 12.x を入れる。NVIDIA の「WSL-Ubuntu」向けのパッケージ(ドライバを含まないもの)を使う。`nvcc --version` で確かめる。
4. llama-cpp-python を CUDA つきでビルドする。

```bash
uv venv --python 3.12 .venv-cuda
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --python .venv-cuda/bin/python \
    llama-cpp-python --no-binary llama-cpp-python
uv pip install --python .venv-cuda/bin/python -e '.[ui,mcp]'
.venv-cuda/bin/python -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"   # True
```

- ビルドには数分〜十数分かかる。`CMAKE_ARGS` を付け忘れると CPU だけの版になる(最後の行が False)。
- RTX 50 系(5060 Ti など)は新しい CUDA(12.8 以上)が要る。古い CUDA では GPU の世代に合うコードが作られない。**この環境では確かめていない。**

## 3. WSL + ROCm(AMD)

**この章は確かめていない。** AMD の「ROCm on WSL」の手順と llama.cpp の HIP のビルド手順に沿って書いた。

1. Windows 側に、WSL の ROCm に対応した AMD のドライバ(Adrenalin)を入れる。
2. WSL の中に ROCm の WSL 版を入れる(AMD の `amdgpu-install` で `--usecase=wsl,rocm --no-dkms`)。
3. `rocminfo` で GPU(7900 XTX なら `gfx1100`)が見えることを確かめる。
   - `HSA_OVERRIDE_GFX_VERSION` は、正式に対応していない GPU を対応している世代として動かすための変数。gfx1100 は対応しているので要らないはず(未確認)。
4. llama-cpp-python を HIP つきでビルドする。

```bash
uv venv --python 3.12 .venv-rocm
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
CMAKE_ARGS="-DGGML_HIP=on -DGPU_TARGETS=gfx1100" \
    uv pip install --python .venv-rocm/bin/python llama-cpp-python --no-binary llama-cpp-python
uv pip install --python .venv-rocm/bin/python -e '.[ui,mcp]'
.venv-rocm/bin/python -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"
```

- 古い llama.cpp では `GPU_TARGETS` ではなく `AMDGPU_TARGETS`。
- ROCm がどうしても動かないときの代わりに Vulkan のビルド(`CMAKE_ARGS="-DGGML_VULKAN=on"`)がある。WSL の Vulkan は遅い見込みなので、最後の手段にする。

## 4. GPU ごとの環境とワーカー

AMD と NVIDIA を 1 つのプロセスで同時に使うのは難しいので、GPU ごとに別の環境・別のプロセスで gne を立て、同じ置き場所(`--root`)を共有して仕事を分ける。

```bash
export GNE_ROOT=~/gne-shared                 # 両方の環境で同じ置き場所
.venv-rocm/bin/gne worker -m ~/models/Qwen3.5-4B-Q8_0.gguf --name xtx     # 端末 1(7900 XTX)
.venv-cuda/bin/gne worker -m ~/models/Qwen3.5-4B-Q8_0.gguf --name 5060    # 端末 2(5060 Ti)
.venv-cuda/bin/gne sweep -m ~/models/Qwen3.5-4B-Q8_0.gguf --layer 22 --workers auto   # 投入側
```

- ワーカーは起動時に `<root>/workers/<name>.json`(モデル id・VRAM・バックエンド・最後に応答した時刻)を書き、`<root>/jobs/` のジョブのうち自分あて(または宛先なし)のものを取って回す。
- 投入側の `--workers auto` は、同じ `model_id`(GGUF の名前・大きさ・先頭のハッシュ)を持つ生きているワーカーに仕事を分ける。モデルが違うワーカーには投げない。
- 分けられるのは `sweep`・`lens steer`(ニューロンの範囲で分ける)。結果はワーカーごとのファイル(`<run>.part-<name>.jsonl`)に書かれ、`gne rank` と UI が結合して読む。
- CUDA と ROCm では丸めの違いで greedy の出力がまれに分かれる。sweep の語の統計には影響しないが、`gne verify` のような一致を見る検査は、ワーカーをまたいで比べない。

## 5. 確認

各環境で次を実行し、`gne bench` の数値を `docs/measurements.md` に書き足す。

```bash
gne info -m <model>
gne verify -m <model> --layer 14        # llama.cpp との等価性(V1〜V8)
gne bench -m <model> --markdown docs/measurements.md
```
