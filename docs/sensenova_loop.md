# SenseNova-Loop(sensenova_loop)で gne を使う

**公式の llama.cpp(ggml-org)は、このモデル(arch `sensenova_loop`)をまだ読めない。** 使うには、このモデルを読める改造版の llama.cpp をビルドし、それを読む llama-cpp-python を入れた、このモデル専用の仮想環境が要る。gne は 2 つの改造版に対応していて、どちらも確かめてある(下の「環境を選ぶ」)。ふだんの gne の環境とは分けておく。

SenseNova-Loop-9B は、同じ層を 2 回続けて実行する(ループさせる)言語モデル。保存されている層は 32 で、そのうち 4〜27 を 2 回通るので、1 回の計算で 56 層分を実行する。

## 環境を選ぶ

| | 公式版 | GuideboardLabs 版 |
|---|---|---|
| llama.cpp | モデルの開発元の [OpenSenseNova/llama.cpp](https://github.com/OpenSenseNova/llama.cpp) のコミット `24ea7bb`(llama.cpp 1.0 が元) | 第三者の [GuideboardLabs/llama.cpp-sensenova-loop](https://github.com/GuideboardLabs/llama.cpp-sensenova-loop) のコミット `a7f77a7`(llama.cpp 0.8 が元) |
| 読める GGUF | 公式の [sensenova/SenseNova-Loop-9B-GGUF](https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF)(BF16・Q8_0・Q5_K_M・Q4_K_M) | [guideboardlabs/SenseNova-Loop-9B-GGUF](https://huggingface.co/guideboardlabs/SenseNova-Loop-9B-GGUF)(Q8_0・Q4_K_M ほか) |
| 生成の速さ(Q4_K_M、RTX 3080、文脈 8,192) | 約 11 tok/s | 約 56 tok/s |
| `gne heads pipeline --rank causal`(question_kinds、v0.7.0 までの選び方。どちらもパッチ 0001 + 0002) | 約 52 分(因果スクリーニング 35 分) | 約 5 分(因果スクリーニング 4 分) |
| GPU のメモリ(画面の分を除く) | 約 6.6 GB(パッチ 2 本で 7.0 GB) | 約 6.1 GB |
| 2 本の流れの中の層の control vector | 最初から届く | パッチ 0001 が要る |
| 2 本の流れの中の層の残差 | パッチ 0001 が要る | 最初から読める |
| 片方の回だけの書き換え | パッチ 0001 と 0002 が要る | パッチ 0001 と 0002 が要る |
| MTP(先読みの層)・画像の入力(mmproj) | MTP の層は GGUF の中に、mmproj は GGUF のリポジトリにある(どちらも gne は使わない) | どちらも無い |

- 速さを取るなら GuideboardLabs 版とその GGUF。公式版はルーターを ggml のふつうの演算の組み合わせで計算する(専用の GPU の処理を足していない、と改造版の docs/models.md にある)ので、生成は約 5 分の 1 の速さで、gne の処理もそのぶん時間がかかる。
- 公式の GGUF や BF16 を使うなら公式版。
- 2 つの改造版は、互いの GGUF を読めない(メタデータとテンソル名が違う。重みの中身はほぼ同じ)。llama.cpp の版と GGUF の変換の違いで、同じモデルでも出力はこのくらい動く: 公開のひな形の 64 問で、最初のトークンの 1 位が一致したのは 91%、分布の差(KL)は平均 0.027・最大 0.24(どちらも Q4_K_M・GPU)。差の出始めは最初の full attention の層の中の数値の違いで、GGUF の重みの丸めの違いも加わる。
- パッチと、どの機能にどのパッチが要るかは [third_party/sensenova-loop/](../third_party/sensenova-loop/README.md) にある(公式版は `official/`、GuideboardLabs 版は `guideboardlabs/`)。

## 構造

| 部分 | 中身 |
|---|---|
| 層 | 保存されている層 32。層 4〜27 を 2 回実行するので、実行順では 56 層。GGUF の `block_count` は 56(公式の GGUF は 57: 56 層と、先読み用の MTP の層 1 つ。gne は MTP の層を数えない) |
| 2 回目の層 | 実行順の層 28〜51。重みは層 4〜27 と同じもの(GGUF にも 1 回分しか入っていない) |
| 残差 | 繰り返しの各回の中では 2 本の流れ(mHC: manifold-constrained hyper-connections)。回の入口で 1 本を 2 本に複製し、出口で平均して 1 本に戻す |
| ルーター | 各層の attention と MLP の前にある小さな行列(F32)。2 本のどちらをどれだけ読むか、出力をどちらにどれだけ足すか、2 本をどう混ぜるかを、トークンごとに決める。計算は 2 つの改造版で同じ |
| ヘッド | 4 層ごとに full attention(16 ヘッド、幅 256)、ほかは線形注意(32 ヘッド、幅 128。公式の GGUF では入力の行列が `attn_qkv_gate`・`ssm_beta_alpha` にまとめて入っている) |
| MLP | 12,288 ニューロン(SwiGLU) |

## 環境を作る

確かめた環境は Linux(WSL2)、CUDA 12.4、Python 3.12、RTX 3080 12 GB。下の `<gne>` は gne のリポジトリ、`<llama.cpp>` は改造版をビルドしたフォルダ。12 GB の GPU では Q4_K_M を勧める。

### 1. 改造版の llama.cpp をビルドする

公式版(パッチ 0001 は gne のほとんどの機能に要る。名前を付けてグラフに残すだけで、計算の結果は変わらない。0002 は片方の回だけを書き換えるときに要る):

```bash
git clone https://github.com/OpenSenseNova/llama.cpp llama.cpp-sensenova
cd llama.cpp-sensenova
git checkout 24ea7bb78410e0f0f7bbb362e37c60cd9ecfc761
git apply <gne>/third_party/sensenova-loop/official/0001-sensenova-loop-graph-names-for-gne.patch
git apply <gne>/third_party/sensenova-loop/official/0002-sensenova-loop-per-visit-weight-names.patch      # 任意(0001 の後)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF
cmake --build build -j 8 --target llama ggml mtmd llama-completion llama-cli
```

GuideboardLabs 版(パッチ 0001・0002 は任意。0001 は 2 本の流れの中の層の control vector に、0002 は片方の回だけの書き換えに要る):

```bash
git clone https://github.com/GuideboardLabs/llama.cpp-sensenova-loop
cd llama.cpp-sensenova-loop
git checkout a7f77a788893ef2cb17c82b9ac0cb1279bcd0918
git apply <gne>/third_party/sensenova-loop/guideboardlabs/0001-sensenova-loop-control-vector-on-both-mHC-streams.patch   # 任意
git apply <gne>/third_party/sensenova-loop/guideboardlabs/0002-sensenova-loop-per-visit-weight-names.patch               # 任意(0001 の後)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF
cmake --build build -j 8 --target llama ggml mtmd llama-server llama-completion
```

### 2. 専用の仮想環境を作る

llama-cpp-python は pip で入れない(中で別の llama.cpp をビルドしてしまうため)。

```bash
uv venv --python 3.12 ~/sensenova-venv
uv pip install --python ~/sensenova-venv/bin/python gguf numpy typer pyyaml jinja2 pydantic safetensors fastapi uvicorn pytest
uv pip install --python ~/sensenova-venv/bin/python --no-deps -e <gne>
```

### 3. llama-cpp-python 0.3.36 を直して置く

0.3.36 の Python のパッケージ(`llama_cpp/` のフォルダ)だけを取り出し、使う改造版に合う直しを当てる。

- 公式版: `official/llama-cpp-python-0.3.36-llama-cpp-1.0.patch`。llama.cpp 1.0 で外された古い関数 36 個(`llama_load_model_from_file` など)があると 0.3.36 は読み込みの時点で止まるので、無い関数は呼んだときだけ止まるようにする。構造体 2 か所(`llama_context_params`・`llama_perf_context_data`)を改造版の `include/llama.h` に合わせる。
- GuideboardLabs 版: `guideboardlabs/llama-cpp-python-0.3.36-structs.patch`。構造体 2 か所(`llama_context_params` に `moe_cache_size`、`llama_perf_context_data` に `n_reallocs_same`・`n_reallocs_diff`)を合わせる。
- どちらも、構造体を直さないと、モデルを読み込むときの設定が全部ずれて読まれる。

```bash
git clone --depth 1 --branch v0.3.36 https://github.com/abetlen/llama-cpp-python
cd llama-cpp-python
git apply <gne>/third_party/sensenova-loop/<official か guideboardlabs>/llama-cpp-python-0.3.36-*.patch
SP=$(~/sensenova-venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
cp -r llama_cpp "$SP/llama_cpp"
mkdir -p "$SP/llama_cpp/lib" && ln -sf <llama.cpp>/build/bin/lib*.so* "$SP/llama_cpp/lib/"
```

### 4. 改造版のライブラリを指して gne を動かす

```bash
export LLAMA_CPP_LIB_PATH=<llama.cpp>/build/bin LD_LIBRARY_PATH=<llama.cpp>/build/bin
~/sensenova-venv/bin/gne info -m SenseNova-Loop-9B-Q4_K_M.gguf
```

`gne info` の表の「2 回実行する層」の行に、この llama.cpp で何ができるか(どちらの改造版か、2 本の流れの中の層の残差が読めるか、control vector が届くか、2 回目の層に別の名前があるか)が出る。できないことを頼むと、gne は理由と、当てるパッチを書いて止める。

| 機能 | 公式版のまま | 公式版 + 0001 | 公式版 + 0001 + 0002 | GuideboardLabs 版のまま | GuideboardLabs 版 + 0001 | GuideboardLabs 版 + 0001 + 0002 |
|---|---|---|---|---|---|---|
| 読み込み・会話・gne serve | できる | できる | できる | できる | できる | できる |
| 繰り返しの外の層と各回の出口(0〜3・27・51〜55)の残差、回ごとに測る(`--score-layers loop` = 27・51・54)、その測る層での CHiPS、`gne heads measure`・`gne heads loop-merge` | できる | できる | できる | できる | できる | できる |
| 2 本の流れの中の層(4〜26・28〜50)の残差: 全部の層での方向づくり・基準層の探索・そこでの `gne verify`・FiSH | できない | できる | できる | できる | できる | できる |
| ルーターの値(`mhc_*_gates`)と 2 本を並べた値(`l_out_streams`) | できない | できる | できる | できる | できる | できる |
| 2 本の流れの中の層への control vector(FiSH の層を重ねる読み直しを含む。無ければ `--no-stack`) | できる | できる | できる | できない | できる | できる |
| 片方の回だけを書き換える(`GNE_LOOP_EDITS=per_visit`、`gne heads causal --visits first / second`、`gne heads visits`、`gne heads loop-pipeline`、`per_visit` と記録したレシピ・LoRA) | できない | できない | できる | できない | できない | できる |

## 層の番号

gne の層の番号は実行順(0〜55)で、llama.cpp のテンソルの捕捉の名前(`l_out-30` など)と control vector の番号と同じ。2 回目の層は、表示で「L30 = 層 6 の 2 回目」のように、重みを持つ 1 回目の層の番号を添える。段の対応は、GGUF の `sensenova_loop.loop.start`・`loop.end`・`loop.count`・`mhc.start`・`mhc.end` から作る。`gne info` の警告にも、どの層を 2 回実行するかが出る。MTP の層(57 番目のブロック)は数えず、そこへの書き換えは断る。

## 残差の読み方

- 層の出力 `l_out-L` は、2 本の流れの平均。公式版は 2 本の中の層でもその平均を計算するが、どこからも読まれないので、手を入れないとグラフから外れる。公式版の 0001 はそれをグラフに残す。GuideboardLabs 版は、そのままでも 2 本の合流の出力から gne が平均を組み立てる(GuideboardLabs 版の 0001 を当てると、グラフの `l_out` が 2 本の平均になる)。
- 2 本の平均を使うのは、回の出口で 2 本が平均されて、それが最終の出力につながるため。各層が読むのは、ルーターが 2 本を重みづけして足したもの。平均から作った方向は、その層が読むものの近似になる。
- 公式版の 0001 は MLP の入力にも名前を付ける(GuideboardLabs 版は最初から名前がある): `attn_residual`(正規化の前。2 本の中の層ではルーターが 2 本を重みづけして足したもの)と `attn_post_norm`(正規化の後。FiSH の鍵が読む値)。attention の入力(2 本の中の層ではルーターの出力)は `attn_in` として返す。
- 2 本を並べたままの値(`l_out_streams-L`)と、ルーターの値(`mhc_attn_gates-L`・`mhc_ffn_gates-L`。トークンごとに 8 つ: 読む重み 2 つ・書き込みの重み 2 つ・混ぜる行列の 4 つ。どれも sigmoid や Sinkhorn の前の値)は、その名前を指定したときだけ返す。
- 公式版で 0001 を当てていない llama.cpp で 2 本の中の層の残差を欲しがると、gne は理由と直し方を書いて止める。

## 重みの書き換えは両方の回に効く

2 回目の層は 1 回目の層と同じテンソルを使うので、LoRA でも焼き込み(重みの書き換え)でも、書き換えは両方の回に効く。gne はこれを前提に次のようにする。

- LoRA のテンソル名は、重みを持つ層の名前にする。2 回目の層の番号で書いたユニット(`31:H3:x0` など)も、層 7 のテンソルに書く。
- CHiPS(ヘッドの方向成分の射影): 因果スクリーニングと配分の候補は、1 回目の層のヘッドで表す(2 回目の同じヘッドは同じ重みなので、別の候補にしない)。ヘッドの書き込みの寄与は両方の回の分を足す。同じヘッドに 1 回目と 2 回目の両方のユニットを書いたときは、2 つの射影を続けて掛けたもの W' = (I − s₂r₂r₂ᵀ)(I − s₁r₁r₁ᵀ)W を 1 つの LoRA にする(足し合わせると、2 本の向きが重なる分を 2 回引いてしまう)。
- FiSH(計算で作るニューロン): 上書きしたニューロンは、もう一方の回の層でも同じ向きを書く。
  - `gne nw check` の書いた量(amount・total)は、両方の回の合計。そのうちもう一方の回の分を `amount_twin` に出す。両方の回で鍵が反応すると書き込みは 2 回入るので、狙った量を超えていないか(over)は合計で見る。
  - `gne nw verify` は、重みを持つ 1 回目の層で式どおりかを確かめ(その前の層はこの LoRA で変わらない)、もう一方の回で上書きしたニューロンがどれだけ開いたか(`twin_act_max`)を出す。
  - ニューロンのふだんの働き(`gne nw neurons`)は両方の回を合わせて数える。同じ重みの 2 つの層(例 12 と 36)を両方選んだときは、既定のニューロンが重ならないように選ぶ。
- gain・ablate も同じ(LoRA は重みを持つ層の名前に書き、両方の回に効く)。層ごとの方向を消す ablate(`--per-layer`)の LoRA は、1 回目と 2 回目の方向の射影を続けて掛けた 1 つにまとめる。焼き込みは重みを持つテンソルを 1 回だけ書き換えるので、層ごとの方向は 1 回目の層の方向を使う。
- `gne verify` の V11(gain の確かめ)は、重みを持つ 1 回目の層で確かめる。公式版で 0001 を当てていない llama.cpp では、2 本の流れの中の層を `--layer` に渡すと、理由と直し方を書いて止める(`--layer 0`・`51`・`52` などは確かめられる)。

### 片方の回だけを書き換える

片方の回だけを書き換えるには、2 回目の層に別のテンソル名(1 回目と同じデータを指す別名)が要る。0002 がそれを付ける。gne はモデルを読み込むときに、その名前で LoRA を読めるかで確かめる。

当て方は、レシピ(`loop_edits`)と LoRA のファイル(`gne.loop_edits`)に記録する。

- `shared`(両方の回に当てる。記録の無い古いレシピ・LoRA もこれ): 2 回目の名前がある llama.cpp でも、LoRA の 1 回目の層のテンソルを 2 回目の名前にも写して当てる(書き出す LoRA のファイルは、重みを持つ層の名前だけで書く。0002 の無い llama.cpp でも読める)。
- `per_visit`(回ごとに当てる): ユニットの層の番号どおり、その回だけを書き換える(`10:N100:x0` は 1 回目だけ、`34:N100:x0` は 2 回目だけ)。CHiPS の候補には 2 回目の層のヘッドも入り、FiSH のもう一方の回の分は出ない(書かないので)。2 回目の名前が無い llama.cpp でこの形のレシピや LoRA を読むと、黙って両方の回に当てずに、理由と直し方(下の `gne heads loop-merge`)を書いて止める。焼き込み(重みの書き換え)は、どちらでも両方の回に効く。
- 環境変数 `GNE_LOOP_EDITS=per_visit` は、記録の無いもの(その場で作るユニットや、探す途中の一時的な LoRA)の既定を変えるだけ。そのとき保存したレシピには `loop_edits: per_visit` が付く。

## 1 回目と 2 回目を別々に扱う CHiPS

1 回目と 2 回目を、ある意味で別のモデルとして扱う。探す段階だけ 0002 を当てた llama.cpp を使い、できたレシピは 0002 の無い llama.cpp で動く形(両方の回に当てる形)に直す。

| 段階 | コマンド | 0002 の無い llama.cpp |
|---|---|---|
| 回ごとに測る | `gne heads pipeline --rank causal --score-layers loop --select-layer 27`(`heads causal` も同じ指定) | できる |
| 回ごとに探す | `gne heads causal --visits first / second / both`、`gne heads visits`、`gne heads loop-pipeline` | both 以外はできない(理由と代わりの方法を出して止める) |
| 仕上げ | `gne heads loop-merge`、`gne heads measure` | できる(回ごとに当てた値の列だけ省く) |

### 回ごとに測る

- `--score-layers loop` は、射影の値(残差を方向に射影した値)を、1 回目の出口(段 27)・2 回目の出口(段 51)・最後の層の 1 つ前(段 54)で測る。最後の層(段 55)は、llama.cpp が最後のトークン以外の行を捨てるので測れない。
- **このモデルでの `gne heads pipeline` の既定**(ほかの arch は今までどおり): 測る層を指定しなければ `--score-layers loop --select-layer 54` で流す。v0.8.0 からの一括の既定 `--rank writes` は、選ぶ層の既定が最後に測れる層(段 54)で、全部の層の部品を順位づけるので、これと同じ段 54 で選ぶ。`--rank causal`(v0.7.0 までの選び方)のときだけ、さらにループの外の段 52・53 を、層を止めた下がり方の順位に関係なく、いつもヘッド単位で測る(`--always-layers`)。下の「確かめたもの」で、効くヘッドが `--rank causal` の候補から漏れたため。それより前の既定に戻すには `--rank causal --score-layers base --always-layers none`。一括の出力に、この arch 用の既定で流したことが 1 行出る。
- `--candidate-layers` は、候補の部品を探す層(例 `28-53`。`--rank writes` と `--rank causal` の両方)。`GNE_LOOP_EDITS=per_visit` と一緒に使うと、1 回目だけ・2 回目だけから選べる。
- `--select-layer` は、ヘッドの順位づけと配分の目標に使う層。`--select-layer 27` なら「1 回目の終わりの値で選ぶ」(段 27 での下がり方で順位をつけ、段 27 より前のヘッドを候補にし、目標は段 27 の B 側の平均)。
- 測る層を指定すると、一括の結果に、因果スクリーニングの上位のヘッドの層ごとの下がり方と、配分した組ごとの層ごとの A 側・B 側の平均が並ぶ。1 回目の終わりでどこまで下がり、2 回目でどう変わったかを 1 つの表で読める。
- 途中の残差に出力層をかけて読む値(logit lens)は目安にしかならない。このモデルは 1 回目の終わりの残差を出力層で読む前提で作られていないので、それを「1 回目の出力」とは読まない。

### 回ごとに探す(0002 を当てた llama.cpp)

- `gne heads causal --visits first`(または `second`): ヘッドを 1 回目だけ(または 2 回目だけ)止めて因果スクリーニングをする。`both` は両方の回で止める(既定と同じ)。
- `gne heads visits --name <方向>`: 同じヘッドを 1 回目だけ・2 回目だけ・両方で止めた効き目を並べ、回の間の順位相関(Spearman)を出す。「両方−和」は、両方で止めた下がり方から片方ずつの和を引いたもの。
- `gne heads loop-pipeline --spec <ひな形> --n-first 3 --n-second 3`: 2 段で選ぶ(どちらの段も `--rank causal` の一括で流す。一括の既定には従わない)。
  1. 段 0〜27 のヘッドから、段 27 の射影の値で選ぶ(1 回目だけに当てる)。
  2. その組を対照集合の両側に当てた状態で、段 28〜53 のヘッドから、段 54 の射影の値で選ぶ(2 回目だけに当てる)。対照集合の側には、レシピを重ねて当てる書き方 `recipe_a: {stack: [<レシピ>, …]}` を使う。
  3. 2 つを合わせたレシピ(`<名前>_loop.per_visit.yaml`)を、次の仕上げにかける。

### 仕上げ(0002 の無い llama.cpp で動く形にする)

- `gne heads loop-merge --recipe <回ごとのレシピ>`: 両方の回に当てるレシピ(`<名前>.shared.yaml`、焼き込み・配布できる形)に直す。各ヘッドをどちらの回で選んだかは `loop_visits` に記録する。
- 直したレシピは、両方の回に当てた状態で段 27・51・54 を測り直し、回ごとに当てたときの値と並べる(探したときの値を足し合わせて済ませない)。2 回目のために選んだヘッドは、両方に当てると 1 回目でも弱まるので、その差がこの表に出る。
- 同じヘッドを両方の回で選んだときは 2 通り: 2 つの射影を続けて掛ける(compose)か、強いほうだけを使う(max)。続けて掛けると、そのヘッドは片方の回で選んだ強さより強く弱まる(同じ向きなら実効の強さは s₁ + s₂ − s₁s₂。s が 2 を超える反転どうしだと、かえって強める)。表には、こうしたヘッドの数と、続けて掛けたときの実効の強さを出す。既定は強いほうだけ(max)。公開のひな形で自然に作った組(2 段 4 回、12 ヘッドの和 2 回)には両方の回で選んだヘッドが無かったので、2 段の組の各ヘッドに、もう一方の回の同じ強さのユニットを足して比べた: disclaimer(両方の回で選んだヘッド 5 個)では、続けて掛けると段 54 の縮めた割合 −1.88・B 側のずれ 1.78(19:H2 の s30 どうしで実効の強さ −469)、強いほうだけなら 1.02・0.36。question_kinds(3 個)では 0.97・0.56 と 1.00・0.27。
- `gne heads measure --name <方向> --recipe <レシピ> …`: どのレシピでも、測る層ごとの A 側の平均、B 側の平均を超えて残る A 側の質問の数、A と B の差を縮めた割合、B 側の直交成分のずれを並べる。`--tail 51` を付けると、段 51 から最後に測る層までの射影の値の変化を切り分ける: 段ごとの残差の大きさと cos(射影の値 ÷ 大きさ)、最後の段の残差を段 51 の方向で測った値、間の段の attention と MLP の書き込みをそれぞれの方向に射影した値(和は射影の値の変化に一致する)。

## control vector

公式版は、2 本の流れの中の層に足した control vector を 2 本の両方に足すので、全部の層で効く(2 本の両方に足すのは、2 本の平均に足すのと同じ。2 本の差は変わらない)。GuideboardLabs 版はそのままだと片方の流れにだけ足し、次の副層がそれを捨てるので、2 本の中の層(4〜26・28〜50)には届かない。GuideboardLabs 版の 0001 で、公式版と同じく両方に足すようになる。届かない llama.cpp で 2 本の中の層に足そうとすると、gne は理由と直し方を書いて止める。FiSH の方向を作る段の「層を重ねる読み直し」も control vector を使う(届かない llama.cpp では `--no-stack`)。

## 思考(thinking)

チャットテンプレートは、既定で思考を入れる(生成プロンプトの最後が `<think>\n`)。gne は方向づくり・鍵づくり・生成のどれでも `enable_thinking=false` で描くので、プロンプトの最後は空の思考の欄 `<think>\n\n</think>\n\n` になる。応答の位置 0 は、その最後のトークン(`</think>` の後の改行)で、応答の最初のトークンを出す位置。3 か所とも同じ描き方なので、位置はそろう。

## CHiPS と FiSH を流す(確かめたもの)

**この節と次の節の数は、v0.7.0 のとき、GuideboardLabs 版の llama.cpp と GGUF で測った値。公式版では取り直していない。** llama.cpp の版と GGUF の変換の違いで、同じモデルでも出力はこのくらい動く: 公開のひな形の 64 問で、最初のトークンの 1 位が一致したのは 91%、分布の差(KL)は平均 0.027・最大 0.24(公式版と GuideboardLabs 版、どちらも Q4_K_M・GPU)。差の出始めは最初の full attention の層の中の数値の違いで、GGUF の重みの丸めの違いも加わる。公式版で確かめたのは、question_kinds の `gne heads pipeline`(この arch の既定)と `gne heads visits` が最後まで動くことと、gne の実物のテスト(0001 なし・0001・0001 + 0002)が通ることまで。gne の実物のテストは、GuideboardLabs 版(パッチなし・0001・0001 + 0002)でも、2 つの改造版に対応した今のコードで通る。

公開のひな形で、次の順に流して最後まで動くことを確かめた(Q4_K_M と Q8_0、GuideboardLabs 版のパッチなしの llama.cpp。一括は v0.7.0 までの選び方 `--rank causal` で流した。v0.8.0 からの既定 `--rank writes` では、組の上限の数は `--n-units` で、レシピの名前は `…alloc-n24.yaml` などになる)。

```
gne heads pipeline --spec configs/pipeline/disclaimer.json --rank causal --target B --n-heads 10,6,4,2,1 -m <model>
gne nw pipeline --contrast configs/fish/closing_rmbase.json --layers 40-46 --remove rmbase --kscale auto \
    --iv <heads pipeline が書いたレシピ(…alloc-n10.yaml など)> --no-stack --save <名前> -m <model>
gne nw verify --recipe <名前> --iv <同じレシピ> --contrast configs/fish/closing_rmbase.json -m <model>
gne nw check  --recipe <名前> --iv <同じレシピ> --contrast configs/fish/closing_rmbase.json -m <model>
```

- `--no-stack` は、control vector が 2 本の中の層に届かない llama.cpp(GuideboardLabs 版でパッチ 0001 なし)のとき(上の「control vector」)。公式版や、GuideboardLabs 版 + 0001 なら付けない。
- 層 40〜46 は動作の確認のための選び方(2 回目の層。上書きしたニューロンが 1 回目の層 16〜22 でも動くので、その数え方を確かめられる)で、勧める層ではない。
- この確認では、もう一方の回(層 16〜22)で書いた分が大きかった。区間 1〜3 では合計の約半分がもう一方の回の分で、狙った量(1)を超える質問が多い。位置 0 では、もう一方の回が逆向きに書いた(成分を取り除く形 !rmbase の基準値は、ユニットの層で中立の文から測った値なので、もう一方の回では合わない)。2 回目の層に書くときは、この合計で強さを決めるか、繰り返しの外の層(0〜3・52〜54)を使う。片方の回だけに書くには、2 回目の層に別の名前がある llama.cpp で `GNE_LOOP_EDITS=per_visit` にする(同じ確認を Q4_K_M で流すと、位置 1〜3 の書いた量の平均は 0.80〜1.05 から 0.62〜0.73 に、位置 0 の逆向きの分は −0.22 から +0.01 になった。当てた CHiPS のレシピは、それぞれの当て方で探したもので同じではない。この LoRA はその llama.cpp でしか当てられない)。
- Q4_K_M では `gne nw verify` が、入力の行の誤差(4〜10%)で許容(5%)を超えて不合格になった。量子化の丸めによる差で、Q8_0 では全部の層で合格(2〜4%)。

### 回ごとの CHiPS(確かめたもの)

(この節の値は、v0.7.0 までの選び方 `--rank causal` での値。v0.8.0 からの既定 `--rank writes` との比べは、下の「全部の部品を書き込みの量で並べる」の節。)

公開のひな形 `disclaimer.json` と `question_kinds.json`(各側 16 問)、Q4_K_M、GuideboardLabs 版 + 0001 + 0002 の llama.cpp で、同じヘッド数(6)の組を候補の範囲を変えて作り、段 54 で比べた。どれも目標は段 54 の B 側の平均で、同じ方向で測った。値は、段 54 の A と B の差のうち縮めた割合(1 で B 側の平均に届く)/ B 側の直交成分のずれ。「両方」は両方の回に当てた形(配れる形)。

| 選び方(6 ヘッド) | disclaimer | question_kinds |
|---|---|---|
| (a) 両方まとめて(`--score-layers loop --select-layer 54`、段 52・53 を足さない) | 0.43 / 0.14 | 0.11 / 0.80 |
| (b) 1 回目の段だけから(回ごと → 両方) | 0.17 → 0.18 / 0.70 | 0.12 → 0.12 / 0.36 |
| (c) 段 28〜53 から(2 回目の段と、ループの外の 52〜53。回ごと → 両方) | 1.08 → 1.11 / 0.09 | 1.01 → 1.10 / 0.17 |
| (c') 段 28〜51 から(2 回目の段だけ。回ごと → 両方) | 1.03 → 1.05 / 0.12 | 1.00 → 1.10 / 0.13 |
| (d) 2 段 `loop-pipeline` 3+3(回ごと → 両方) | 1.00 → 1.26 / 0.35 | 1.00 → 1.02 / 0.26 |
| (e) 段 52〜54 だけから(測る層を含む、両方) | 1.06 / 0.03 | 1.00 / 0.02 |
| このモデルの既定(引数なし。下の注) | 0.72 / 0.10 | 0.11 / 0.80 |

- 4 ヘッドでも並びは同じ。2 段の組は (a) より大きく下がったが、効いたのは候補に入る層の違いだった: 2 回目の段だけ(c')でも、ループの外の段 52〜54 だけ(e)でも B 側の平均に届き、2 段目はどちらのひな形でも段 52 のヘッド(52:H23)を選び、1 回目のヘッドだけ(b)では段 54 にほとんど効かない。
- (a) で段 52〜53 のヘッドを選ばなかった理由はひな形で違う。disclaimer では、層ごとに全部のヘッドを止めた下がり方で選ぶ上位 4 層に段 52 が入らず、ヘッド単位では測っていない。question_kinds では 52:H23 は上位 2 番目で最初の組にも入ったが、効率で配り直す段で外れた。
- (e) の disclaimer の組は、ほとんど段 54(測る層)のヘッド。測る量を直接動かすだけで振る舞いは変わりにくいことがある(docs/head-component-steering.md)。既定の候補の決め方(測る最初の層より前のヘッドだけ)では、段 52〜54 は候補に入らない。
- 同じ組を両方の回に当てると、2 回目のために選んだヘッドが 1 回目でも働く。question_kinds の (c) では、段 27 の A 側の平均が、回ごとに当てると +15.7 のまま、両方に当てると −6.1 になった。
- 振る舞いでも確かめた(`gne behave`、方向づくりにも選ぶのにも使っていない質問を各側 30 問、greedy、上限 1,024 トークン、両方の回に当てた形、6 ヘッド)。値は、書き終えた応答のうちひな形の `validate.match` に一致した数。

  | 選び方 | question_kinds A(日本語の質問 → 日本語で答える) | question_kinds B(英語の質問) | disclaimer A(深刻な相談 → 専門家への相談を勧める) | disclaimer B(軽い質問) |
  |---|---|---|---|---|
  | 無介入 | 30/30 | 0/30 | 23/28 | 0/30 |
  | (a) 両方まとめて | 30/30 | 26/26(英語の質問に日本語で答える) | 26/29 | 0/26 |
  | (c) 段 28〜53 | 0/27 | 0/30 | 20/29 | 0/29 |
  | (c') 段 28〜51 | 0/27 | 0/30 | 20/26 | 0/28 |
  | (e) 段 52〜54 | 0/26 | 0/30 | 21/29 | 0/28 |
  | このモデルの既定(引数なし) | 29/29 | 28/28(英語の質問に日本語で答える) | 24/30 | 0/29 |

  - question_kinds では、(c)・(c')・(e) が日本語の質問への答えを全部英語に変え、英語の質問はそのまま。(e) の組は主に段 52 のヘッドで、測る量を動かしただけではなかった。(a) は A 側を変えず、B 側(英語の質問)を日本語の答えに変えた(B 側のずれ 0.80 の組)。disclaimer では、射影の値は届いたが((c)・(c')・(e) は段 54 で B 側の平均まで下がった)、振る舞いは基準(30 問中 5 問以上の変化)に届かなかった。この正規表現は本文の中の「受診」なども数えるので、注意書きだけの変化は数えにくい。書き終えなかった応答(上限で切れた)は、どの条件も 0〜4 件。
  - このモデルの既定(測る層 loop・選ぶ層 54・段 52・53 をいつもヘッド単位で測る)で引数なしに流すと、disclaimer では段 52 のヘッド 3 個が組に入り、段 54 で 0.72 まで下がった(振る舞いは変わらない)。question_kinds では、効率で配り直す段がまた 24:H19 などの組を選び、(a) と同じく英語の質問に日本語で答えるようにした。一括の出力には、配り直しで外れたヘッドとして 21:H20・52:H23・53:H0 が出る。候補に入れるだけでは足りず、配り直しの段の選び方(推す組の決め方)が残る問題で、今回は変えていない。
- 段 51 から 54 で射影の値が上がる理由(無介入の A 側、`gne heads measure --tail 51`): disclaimer では +41.0 → +60.1 だが cos は 0.253 → 0.266 とほぼ同じで、段 54 の残差を段 51 の方向で測ると +40.9(上がらない)。段 52〜54 は段 54 の方向に +29.7 書き足していて(MLP が 21.8)、段ごとに方向が違うことで出る差。question_kinds では段 51 の方向でも +9.5 上がり、段 52 の attention が段 54 の方向に 13.2 書く。

### 全部の部品を書き込みの量で並べる(`gne heads writes`・`--rank writes`)

[docs/head-component-steering.md](head-component-steering.md) §3.9 の選び方(v0.8.0 からの一括の既定)は、このモデルでもそのまま使える(既定の選ぶ層は段 54)。

- 部品は、重みを持つ層の番号で表す(段 4〜27 のヘッド・ニューロンは、2 回目の段 28〜51 でも同じ重みで書くので、2 回分の書き込みを足して 1 つの部品にする)。ニューロンの射影も両方の回に効く。
- 2 本の流れの中の層では、副層(注意・MLP)の出力 `y` は 2 本に `post₀·y`・`post₁·y` として足され、2 本を混ぜる行列は行も列も和が 1(Sinkhorn で作る)なので、2 本の平均の残差の増分は `c·y`(`c = (post₀ + post₁)/2`)。書き込みにはこの `c` を掛ける。`post` はルーターの値で、捕捉の `mhc_attn_gates-L`・`mhc_ffn_gates-L` の先頭 2 つ。
- 部品から組み立てた層の更新が、捕まえた残差の差 `l_out(L) − l_out(L−1)` と合うかを、結果の `check.layer_rel_median` / `layer_rel_max` に出す。Q4_K_M で中央値 0.4〜0.5%・最大 1.3〜2.3% だった。
- 層ごとの更新を比べる `gne heads geometry`([layer-geometry.md](layer-geometry.md))は、同じ重みの 1 回目と 2 回目(段 L と L+24)の更新の cos を、層全体・注意・MLP ごとに表にする(類似度の行列では、対角線から 24 段ずれた線)。公開のひな形 2 つでは、層全体の cos は +0.01〜+0.68 で、層の単位では逆向きの組はほとんど無かった(結論ではない。入力の残差が回で違うので、同じ重みでも更新は違ってよい)。

確かめ(Q4_K_M、改造版の llama.cpp のまま、両方の回に当てる形、目標 `B`。応答は方向づくりにも選ぶのにも使っていない質問の各側 30 問、greedy、上限 1,024 トークン)。(1) は v0.7.0 までのこのモデルの既定(`--rank causal`、測る層 loop・選ぶ層 54・段 52・53 をいつも測る)、(2) は `--rank writes --unit-kinds heads`(v0.8.0 からの既定)、(3) は `--rank writes --unit-kinds all`。値は、段 54 の A と B の差を縮めた割合 / B 側の直交成分のずれ、書き終えた応答のうち `validate.match` に一致した数。

| ひな形 | 条件 | 部品 | 強さ 最大 / 中央値 | 段 54 | A 一致 | B 一致 |
|---|---|---|---|---|---|---|
| question_kinds | 無介入 | — | — | — | 30/30 | 0/30 |
| | (1) | 線形注意 5・注意 1 | 30 / 4.4 | 0.11 / 0.80 | 29/29 | 28/28 |
| | (2) | 線形注意 12 | 5.5 / 5.5 | 0.87 / 0.07 | 0/29 | 0/30 |
| | (3) | 線形注意 11・ニューロン 1 | 5.5 / 5.5 | 0.88 / 0.07 | 0/29 | 0/30 |
| disclaimer | 無介入 | — | — | — | 23/28 | 0/30 |
| | (1) | 線形注意 4(段 52 の 3 個) | 30 / 21.3 | 0.69 / 0.08 | 25/30 | 0/28 |
| | (2) | 線形注意 17・注意 3 | 19.4 / 19.4 | 1.02 / 0.14 | 20/26 | 0/29 |
| | (3) | 線形注意 6・ニューロン 18 | 13.6 / 13.6 | 1.00 / 0.12 | 23/30 | 0/28 |

- question_kinds: (1) は前と同じく、英語の質問に日本語で答えるようにした(B 側を壊した)。(2)・(3) は日本語の質問に英語で答え、英語の質問はそのまま。無作為に選んだ同じ数の部品では変わらない。どの組も、段 54 の目標(B 側の平均)には届かず(縮めた割合 0.9 前後)、推す組は無かった(残る数が最少の組を使った)。
- disclaimer: どの条件も応答を変えなかった。(2)・(3) は射影の値を B 側の平均まで下げている。ニューロンを入れると、同じ目標に要る強さは 19.4 から 13.6 に下がった。
- system_persona は、このモデルが悪意の AI の system を与えてもふつうに答える(無介入で A 側の一致 0/30)ので、応答では比べられなかった。
- この表の (1) は、このモデルの v0.7.0 までの既定。v0.8.0 からの一括の既定は (2)(`--rank writes`、ヘッドだけ)。
- 上の表の組は v0.8.0 の倍率の上げ方で作った。v0.8.1 の上げ方で、同じ順位づけと確かめの結果から組を作り直すと:
  - disclaimer: 変わらない。
  - question_kinds: どの組も届かないので、組の効き(A 側の下がり − B 側の動きの大きさ)がいちばん大きい組として、6 個の組(線形注意 6、強さ 5.5)を推すようになった(v0.8.0 は推す組なし)。応答は A 一致 1/30・B 一致 0/30 で、上の (2) と同じ。上限 24 と 6 の組は、上げても良くならないところで止まり、強さ 7.7 と 30 が 5.5 になった。
  - system_persona: s=1 で足す部品が、上限 24 の組で 1 個(ヘッドだけ)と 3 個(ニューロンも)変わった(A 側の下がりより B 側の動きが大きい部品を採らなくなった)。
  - どの組も、B 側が A 側へ動いた量は差の 3% 以下で、新しい上限(10%)には当たらなかった。
- 組の表には、1 回目の出口(段 27)での A・B の射影の値の変化と B 側の直交成分のずれも並ぶ(v0.8.1。ループの中の部品は両方の回に当たるので、1 回目にも効く)。組にループの中の部品が入ると、そのことを 1 行知らせる。

**ループの中の部品が使う方向(v0.8.1、`--loop-direction`)。** 両方の回に当てる形では、ループの中の部品(重みを持つ層 L、段 L と段 L+24 で実行)を、既定では 1 回目の段の方向 r̂_L で測り、射影する(`--loop-direction first`)。`--loop-direction second` で 2 回目の段の方向 r̂_{L+24} にでき、部品の番号も 2 回目の段で表す(例 `45:H20` は層 21 のヘッド 20 の重みを、段 45 の方向で書き換える。両方の回に当たる)。公開のひな形 2 つで、4 通りを比べた(Q4_K_M、GuideboardLabs 版、両方の回に当てる形、目標 B、組は v0.8.1 の作り方、応答は調整に使っていない質問の各側 30 問)。r̂_L と r̂_{L+24} の cos の中央値は 0.56(0.10〜0.76)、段 54 の方向との cos の中央値は r̂_L が 0.12〜0.15、r̂_{L+24} が 0.41〜0.44。

| 方向 | question_kinds: 組 / 届いたか / A 一致・B 一致 | disclaimer: 組 / 段 27 の B 側の変化 / A 一致・B 一致 |
|---|---|---|
| 無介入 | — / — / 30・0 | — / — / 23・0 |
| r̂_L(既定) | 6 個・強さ 5.5 / 届かない(差の 88%) / 1・0 | 24 個・強さ 13.6 / −5.4 / 23・0 |
| r̂_{L+24}(second) | 6 個・強さ 5.8 / 届く / 3・0 | 12 個・強さ 14.9 / +1.0 / 20・0 |
| r̂_54 | 12 個・強さ 4.3 / 届く / 3・0 | 12 個・強さ 16.9 / +0.7 / 26・1 |
| r̂_L と r̂_{L+24} の両方 | 24 個・強さ 2.0 / 届かない / 5・0 | 24 個・強さ 2.0(届かない)/ +0.9 / 22・0 |

- 射影の値では 2 回目の段の方向がはっきり良い(question_kinds は目標に届き、disclaimer は半分の部品で組の効きが差の 11% ぶん上、1 回目の出口で B 側をずらさない)。応答の一致は良し悪しが分かれ(question_kinds で 1 → 3、disclaimer で 23 → 20)、disclaimer では段 27 の B 側の直交成分のずれが 0.019 → 0.073 に上がったので、既定は変えていない。
- 両方の方向を続けて掛ける形は、強さ 2 前後で上げても良くならなくなり、はっきり悪い。
- `--rank causal` も、両方の回に当てる形ではループの中のヘッドを 1 回目の段で表し、r̂_L で射影する(同じ問題が当てはまる。変えていない)。
- 時間は (1) が 300〜360 秒、(2)・(3) が 340〜460 秒(順位づけ 1 分前後、上位 80 個の確かめ約 2 分、組づくり 1〜2 分、応答での確かめ 0.5〜3 分)。

## 目安(Q4_K_M、RTX 3080 12 GB)

2 つの改造版を同じ測り方で測った値(llama-completion、文脈 8,192、全層を GPU に載せ、256 トークンまで生成。GPU のメモリは画面の分を除いた最大値)。それぞれの改造版の Q4_K_M を使った。

| 処理 | 公式版 | 公式版 + 0001 + 0002 | GuideboardLabs 版 |
|---|---|---|---|
| 生成 | 約 11 tok/s | 約 11 tok/s | 約 56 tok/s |
| 短いプロンプトの読み込み(20 トークン) | 約 45 tok/s | 約 41 tok/s | 約 290 tok/s |
| GPU のメモリ | 約 6.6 GB | 約 7.0 GB | 約 6.1 GB |
| `gne heads pipeline --rank causal`(question_kinds、各側 16 問。v0.7.0 までの選び方で、ヘッド 144 個を止めて測る) | — | 約 52 分(因果スクリーニング 35 分) | 約 5 分(因果スクリーニング 4 分。0001 + 0002 で測った値) |

公式版は 2 本の流れのルーターを ggml のふつうの演算の組み合わせで計算する(専用の GPU の処理は足していない、と改造版の docs/models.md にある)ので、生成は GuideboardLabs 版の約 5 分の 1 の速さだった。gne の処理も、そのぶん時間がかかる。ほかの処理の時間は、公式版では測っていない。

gne の処理ごとの目安(GuideboardLabs 版で測った値。公式版では、生成と読み込みが遅いぶん、上の表の比で長くかかる):

| 処理 | 時間・メモリ |
|---|---|
| 文脈の長さ | 65,536 まで余裕あり(約 9 GB)。131,072 は 12 GB にほぼいっぱい |
| `gne heads pipeline --rank causal`(各側 16 問) | 約 4 分(方向 3 秒、因果スクリーニング 3 分、配分 1 分強)、GPU 約 6 GB。Q8_0 は約 5 分(`--n-ctx 2048` で GPU 最大 11.5 GB) |
| `gne heads pipeline`(既定の `--rank writes`、各側 16 問) | 約 6〜8 分(順位づけ 1 分前後、上位 80 個の確かめ約 2 分、組づくり 1〜2 分、応答での確かめ 0.5〜3 分) |
| `gne nw pipeline`(層 7 つ、質問 12 + 8) | 約 2 分 20 秒 |
| `gne nw verify --contrast`・`gne nw check` | それぞれ約 20 秒 |
| `gne heads loop-pipeline`(各側 16 問、3+3) | 約 8〜9 分(一括の流れ 2 回と、仕上げの測り直し) |
| `gne heads loop-merge`(測り直しつき)・`gne heads measure`(レシピ 9 個、`--tail 51`) | 約 10 秒・約 1 分 |

止めて測る段階(射影の値・因果・配分)は、質問を最大 64 個に間引く(`max_examples`)。
