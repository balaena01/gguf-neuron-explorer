# Inkling 2(inkling2_loop)で gne を使う

**いまの gne が Inkling 2 で扱うのは、読み込み・情報(`gne info`)・マップ・会話だけ。** 残差・ヘッド・ニューロン・専門家を測る機能と、control vector・LoRA・焼き込みで中を書き換える機能は、このモデルの構造に合わせた計算がまだ無いので、理由を書いて止まる(下の「使える機能と止まる機能」)。間違った値を出さないための扱い。

Inkling-2-Mini(Thinking Machines Lab、Apache 2.0)は、保存されている 24 層のうち 6〜17 を 3 回続けて実行する MoE の言語モデル。答えの前に思考(reasoning)を書く。

## 環境

公式の llama.cpp(ggml-org)は、2026-11-20 のコミット `29b52d3`(PR #32694)からこのモデル(arch `inkling2_loop`)を読める。ただしその版は、2 回目・3 回目の段で専門家ごとの倍率(下の「構造」)を掛けずに計算する(修正の PR の説明による)。直す PR #32704(コミット名 `model : inkling2_loop re-share the loop slots after the scale pass`)は 2026-11-21 時点でまだ取り込まれていないので、それを当ててビルドする。ふだんの gne の環境(pip で入れた llama-cpp-python 0.3.36 の中の llama.cpp)は古くて読めないので、このビルドを読む llama-cpp-python を入れた専用の仮想環境を作る。

確かめた環境は Linux(WSL2)、CUDA 12.4、Python 3.12、RTX 3080 12 GB。GGUF は unsloth/Inkling-2-Mini-GGUF の、arch が `inkling2_loop` のもの(確かめたのは UD-IQ2_XXS、8.5 GB)。12 GB の GPU に載るのは 8〜9 GB の量子化まで。

### 1. llama.cpp をビルドする

確かめたのは `29b52d3` に PR #32704 のコミットを 1 つ当てたもの。

```bash
git clone https://github.com/ggml-org/llama.cpp llama.cpp-inkling2
cd llama.cpp-inkling2
git checkout 29b52d398
git fetch origin pull/32704/head && git cherry-pick FETCH_HEAD     # PR #32704(取り込まれたら不要)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF
cmake --build build -j 8 --target llama ggml mtmd llama-completion llama-server
```

### 2. 専用の仮想環境と llama-cpp-python 0.3.36

[docs/sensenova_loop.md](sensenova_loop.md) の「2. 専用の仮想環境を作る」「3. llama-cpp-python 0.3.36 を直して置く」と同じ手順で、直しは `third_party/sensenova-loop/official/llama-cpp-python-0.3.36-llama-cpp-1.0.patch` をそのまま使う(この llama.cpp の `include/llama.h` は、直しが合わせた llama.cpp 1.0 の改造版と構造体が同じ。増えた関数 2 つ `llama_model_meta_arr_n`・`llama_model_meta_arr_str` は gne が使わない)。

`llama_cpp/lib/` のリンクは、必ずこの llama.cpp の `build/bin` を指す。llama-cpp-python の `_ggml.py` は、`LLAMA_CPP_LIB_PATH` ではなく `llama_cpp/lib/` から libggml を読むので、別の llama.cpp を指したままだと libggml が 2 つ読み込まれる。

```bash
export LLAMA_CPP_LIB_PATH=<llama.cpp>/build/bin LD_LIBRARY_PATH=<llama.cpp>/build/bin
<venv>/bin/gne info -m Inkling-2-Mini-UD-IQ2_XXS.gguf
```

## 構造

| 部分 | 中身 |
|---|---|
| 層 | 保存されている層 24(GGUF の `block_count`)。層 6〜17 を 3 回続けて実行するので、1 回の計算で 48 段。llama.cpp の `llama_model_n_layer` は 48 を返す |
| 繰り返し | 2 回目・3 回目は層 5 の出力からやり直す。前の回の流れ(下の流れ 1・2)を、学習された問いで混ぜた「メッセージ」にして、Q・K・V・MLP の入力に低ランク(ランク 64)のゲートで混ぜる: 入力 ← (1 − forget)·入力 + gain·メッセージ(gain・forget は入力とメッセージから計算する tanh) |
| 残差 | 1 本ではない。層 0・6・12・18 で新しい流れが始まり、それまでの流れは取っておかれる。各副層(attention・MLP)の入力は、それまでの流れと今の流れを RMS 正規化して、層ごとの学習された問い(`attn_res_query`・`ffn_res_query`)で重みづけして足したもの(深さ方向の注意)。最後の出力も同じ形で混ぜる(`output_res_query`) |
| attention | ヘッド 24(KV ヘッド 8)、幅 128。RoPE は無く、相対位置のバイアスを使う。窓つき(sliding window、1,024 トークン)の層が 20、全体を見る層が 4(層 5・11・17・23)。全体を見る層は Q・K・V の入力の正規化が別々 |
| 短い畳み込み | K・V・attention の出力・MLP の出力が、4 タップの畳み込み(チャンネルごと、時間方向)を通る。attention と MLP の出力は、畳み込みを通ってから流れに足される |
| MLP | 全層 MoE。専門家 128、1 トークンで 8 個、幅 1,152。ゲートとアップは 1 つのテンソル(`ffn_gate_up_exps`)。sigmoid で選び、選んだ重みを正規化して 8 倍する。活性化は SiTU |
| 専門家の重み | 量子化の型が層で違う(IQ2_XXS の GGUF では、ゲートとアップが IQ1_S・IQ2_XXS・IQ2_XS・IQ2_S、ダウンが Q2_0・NVFP4)。20 層に専門家ごとの倍率(`ffn_*_exps.scale`)があり、llama.cpp は専門家の行列の出力にそれを掛ける |
| 出力 | 最後のロジットに soft cap 30 |

## 段の番号

gne の表示の「段」は、実行の順(0〜47)。llama.cpp のキャッシュの番号(グラフのテンソルの名前の `-N`)は実行の順と違う:

| 段 | 層 | 回 | llama.cpp の番号 | 流れ |
|---|---|---|---|---|
| 0〜5 | 0〜5 | — | 0〜5 | 0 |
| 6〜17 | 6〜17 | 1 回目 | 6〜17 | 1(層 6〜11)、2(層 12〜17) |
| 18〜29 | 6〜17 | 2 回目 | 24〜35 | 1、2 |
| 30〜41 | 6〜17 | 3 回目 | 36〜47 | 1、2 |
| 42〜47 | 18〜23 | — | 18〜23 | 3 |

`gne info` にこの表が出る(`--json` では `meta.inkling.steps` に段ごと)。マップの層は保存されている 24 層で、各層の札に窓つきか全体を見る attention か、繰り返しの層なら何回・どの段で実行するかが出る。

## 会話

- `gne chat -p …`(レシピなし)は、無介入の 1 列で話す(ほかのモデルの `gne chat` は −/0/+ の 3 列で、レシピが要る)。
- `gne serve` では、地図の凡例の「会話する」で、無介入の 1 列のチャットが開く(くわしく見るの欄に、いまの層のニューロン 0 のチャットとして開く。介入の欄は出ない。応答の長さの既定は 1,024)。
- 思考の欄は `<think>` … `</think>` で囲んで見せる。会話の履歴には回答だけを入れる(前の思考は次のプロンプトに入れない)。
- `gne serve --openai` の `/v1/chat/completions` は、思考を `reasoning_content`、回答を `content` に分けて返す(流すときも同じ)。
- チャットテンプレートは、`reasoning_effort`(既定 0.9)を「Thinking effort level」として最初の system の欄に入れる。
- 思考の量: `gne chat --reasoning-effort low` のように渡す。テンプレートが受け付けるのは none(0)・minimal(0.1)・low(0.2)・medium(0.7)・high(0.9)・xhigh(0.99)・max(0.99)か、0〜0.99 の数。省くとテンプレートの既定(0.9)。UI のチャットの「思考の量」、OpenAI 互換 API の `reasoning_effort` も同じ。
- モデルの勧めるサンプリングは温度 1.0・top-p 1・top-k 0・min-p 0(`--temperature 1 --top-p 1 --top-k 0 --min-p 0 --presence-penalty 0`)。
- 答えの前に思考を書くので、`--max-tokens` は長めに取る(公開のひな形の質問で、思考と回答を合わせて平均 440 トークン前後)。

速さとメモリ(UD-IQ2_XXS、RTX 3080、`gne bench`): 生成 1 本 約 51 tok/s、4 本まとめて 約 116 tok/s。GPU のメモリは文脈 4,096 で約 9.8 GB(画面の分を除く)。

## 使える機能と止まる機能

| 使える | 止まる(理由を書いて止める) |
|---|---|
| `gne info`、`gne probe --list-names`(グラフのテンソルの名前だけ。番号は段)、`gne inkling read / check / mix / residuals / conv / experts`(中を読むだけ。0001 が要る。下の「中を読む」)、`gne chat`(レシピなし。`--reasoning-effort`)、`gne serve` のマップ・会話・保存済みの一覧、OpenAI 互換 API、`gne behave`(レシピなし、`--match`。思考の欄を除いた回答だけを数える)、`gne verify`(トークン化の V8 だけ。ほかは省く) | 中間のテンソルの捕捉(残差・ヘッド・ニューロン・専門家の値)、control vector、LoRA、焼き込み、重みの中身を読む処理(ニューロン・専門家・ヘッドの詳細、logit lens など)。これを使うコマンド(`gne stats`・`lens`・`sweep`・`direction`・`heads …`・`experts …`・`nw …`・`steer`・`gain`・`ablate`・`bake`・`eval` など)は全部止まる。`gne behave --judge`(読み手の判定)と `gne fieldtest` も止まる |

止まるときのエラーは `ARCH_UNSUPPORTED`。CLI では、使えるコマンド(`info`・`probe`・`verify`・`bench`・`chat`・`behave`・`worker`・`recipe-combine`・`recipe-scale`・`direction keys`・`inkling …`)のほかは、モデルを読み込む前に止まる。MCP と画面のジョブは、中間のテンソルの捕捉・control vector・LoRA・重みの読み出しのところで止まる(生成だけを先にする処理は、生成の後で止まる)。中間のテンソルの捕捉を通すのは `gne inkling …` の読むだけの処理だけ。

## 中を読む(読むだけ)

中の値を読むには、llama.cpp に `third_party/inkling2/0001-inkling2-graph-names-for-gne.patch` を当てる(名前を付けるだけで計算は変えない。当てる前と後で、公開のひな形の 4 問の最後の位置のロジットと貪欲に生成した 24 トークンが、CPU でも GPU でもビットまで一致した)。当て方は [third_party/inkling2/](../third_party/inkling2/README.md)。当てていない llama.cpp では、`gne inkling …` は理由と直し方を書いて止まる。

名前の番号は、gne が実行順の段に読み替える(`l_out-18` は段 18 = 層 6 の 2 回目)。`gne probe --list-names` もこの番号で出す。流れが始まる段(0・6・12・18・24・30・36・42)では、attention の出力が新しい流れの始まりになり、グラフでは `attn_out` の名前が `ffn_inp` に上書きされている(同じテンソル)。gne は `attn_out-<段>` を頼まれたら `ffn_inp-<段>` から返す。

| 名前 | 中身 |
|---|---|
| `l_out` | 今の流れの、その段までの和(流れの部分和) |
| `ffn_inp` | attention を足した後の今の流れ |
| `attn_out`・`ffn_out` | attention・MLP の書き込み(短い畳み込みの後。流れに足すもの) |
| `attn_wo`・`ffn_moe_out` | 同じ書き込みの、短い畳み込みの前 |
| `attn_mix`・`ffn_mix` | 副層が実際に読む入力: それまでの流れと今の状態を RMS 正規化し、深さ方向の注意の重みで足したもの。全体を見る層の `attn_mix` は Q・K・V の 3 行 |
| `attn_mix_w`・`ffn_mix_w`・`out_mix_w` | 深さ方向の注意の重み(並びは、取っておいた流れの新しい順、最後が今の状態) |
| `loop_msg`・`loop_msg_w` | 2 回目・3 回目: 前の回の流れから混ぜたメッセージ(Q・K・V・MLP の 4 行)とその重み |
| `loop_gain_{q,k,v,ffn}`・`loop_forget_{q,k,v,ffn}` | 2 回目・3 回目: ゲートの値。入力 ← (1 − forget)·入力 + gain·メッセージ |
| `attn_in_{q,k,v}`・`ffn_in` | 2 回目・3 回目: ゲートの後の入力(正規化の前) |
| `kqv_out`・`attn_gated` | ヘッドの出力(ゲートの前・後。`attn_gated` が W_o の入力) |
| `ffn_moe_argsort`・`ffn_moe_weights_scaled`・`ffn_moe_situ` | 専門家の選び方(大きい順の並びの先頭 8 個が選んだ専門家)・選んだ重み・専門家のニューロンの活性 |

| コマンド | 中身 |
|---|---|
| `gne inkling read --text … --names … --steps … --out x.npz` | 段ごとの値を .npz に書く |
| `gne inkling check` | 捕まえた値を別の計算と突き合わせる: 流れの和、混ぜ方(状態と重みから)、繰り返しのゲート、W_o の積、短い畳み込み、専門家の和(倍率 × W_down × 活性)、ゲート・アップと SiTU |
| `gne inkling mix --spec …` | 各段の深さ方向の注意の重み(どの流れをどれだけ読むか) |
| `gne inkling residuals --spec …` | 残差の定義(流れの部分和と、副層が読む混ぜた入力)ごとの A − B の方向の比べ、回どうしの向き、3 回目の方向に沿った各回のヘッド・副層の書き込み、深さ方向の注意の重みの A 側と B 側の違い、繰り返しのメッセージの強さ |
| `gne inkling conv --spec …` | 書き込みから方向を除いたとき、短い畳み込みの後に残る方向の成分 |
| `gne inkling experts --spec …` | 段ごとの専門家の選ばれ方と、同じ層の回どうしの重なり |

結果は `<root>/inkling/<model_id>/` に書く。方向も保存するが、方向で押す・射影する機能(CHiPS・FiSH・キー・lens など)は、残差をどちらの定義で読むか決まるまで止めたまま。`gne stats resid`・`gne stats act`・`gne heads attention` も、それぞれ決めることが残っているので理由を書いて止まる。

`gne inkling check` の結果(UD-IQ2_XXS、GPU、24 トークン、全 48 段): 流れの和・混ぜ方・ゲート・短い畳み込み・SiTU は相対誤差 3×10⁻⁸〜2×10⁻⁷(float32 の丸め)。W_o・専門家の行列の積は 5×10⁻³〜1×10⁻²(llama.cpp は量子化した重みとの積で活性も量子化する。gne の計算は重みを戻して float64 で掛ける)。
