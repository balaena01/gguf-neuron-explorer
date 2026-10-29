# 新しいモデル・新しい arch を動かす

## 手順

1. `gne info -m M`: arch の判定、残差を書くテンソル、dequantize の可否、チャットテンプレート、bake の可否、警告。
2. `gne verify -m M`: llama.cpp と gne の等価性(V1〜V8)。新しい arch では最初に通す。
3. `gne stats resid`: 層ごとの残差の大きさ(rel dose の基準)。
4. 押さない生成で、まともに答えるか確かめる(`gne steer -p "..."`)。崩れるなら温度とチャットテンプレートを疑う。
5. 量子化: 同じモデルの別の量子化には、見つけたニューロンがそのまま移る(Q4_K_M・Q3_K_M で同じ文、2bit は使える強さの幅が狭い。README)。探索は Q8_0 で行うと速く、誤差も小さい。

## llama.cpp が arch に対応していないとき

- gne は llama-cpp-python の上で動く。同梱の llama.cpp が古いと、新しい arch(例: lfm3)を読めない。
- llama-cpp-python を、vendor の llama.cpp を新しい版にして作り直す(`CMAKE_ARGS="-DGGML_CUDA=on" pip install -e .`)。
- C の API の構造体が変わっていると、Python のバインディング(`llama_cpp/llama_cpp.py`)が合わずに落ちる。`include/llama.h` を見比べて、ctypes の構造体にフィールドを足す。LFM3 では次の 2 つ。
  - `llama_context_params`: `type_v` の後に `moe_cache_size`(`c_size_t`)
  - `llama_perf_context_data`: 末尾に `n_reallocs_same`・`n_reallocs_diff`(`c_int32`)
- 作り直した後、同じ種で前と同じ文が出るかを確かめる(結果を比べられるように)。

## チャットテンプレート

- gne は GGUF の `tokenizer.chat_template` を jinja2 で描く。HF transformers の拡張タグ(`{% generation %}`)は対応済み。描けないタグがあれば `gne/engine.py` の `render_template` に拡張を足す。
- `<think></think>` を入れるテンプレート(Qwen3 系・LFM3)は、`enable_thinking=False` で空の思考を入れて返事に入る。

## 小さいモデル(〜2B)の注意

- 温度を下げる(LFM3 は 0.3、min_p 0.05、presence 0.5)。レシピの `sampling` と、`tune.py --sampling`・`common.generate(sampling=...)` に渡す。
- モデル自身は採点役にならない。採点は Claude に任せ、ふるいは採点なしの物差しで行う(SKILL.md §4.3)。
- rel dose の崩れ始めが低い。0.5 でも崩れることがある。

## 音声も出すモデル(LFM3-Omni など)

- 言語の部分(本体)は gne で読めて、文章の押し方は今までと同じ。音声の出力は llama.cpp の `llama-tts`・`llama-server` が受け持つ(`-mm mmproj`)。
- 押した結果を声にするには、レシピを `gne export-cvec` で control vector にして `--control-vector` で渡す。when は無い(全位置に足す)。
- 読み上げの声の指定は、GGUF の `tokenizer.chat_template.tts` が決める。LFM3-Omni は female / male だけで、それ以外は female になる。言語の欄は知らない値をそのまま文に入れるので、声の様子を書き足して試せる(効きは弱い)。
