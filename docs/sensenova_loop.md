# SenseNova-Loop(sensenova_loop)で gne を使う

**公式の llama.cpp は、このモデル(arch `sensenova_loop`)をまだ読めない。** 使うには、第三者の改造版の llama.cpp をビルドし、それを読む llama-cpp-python を入れた、このモデル専用の仮想環境が要る(下の「環境を作る」)。ふだんの gne の環境とは分けておく。

SenseNova-Loop-9B は、同じ層を 2 回続けて実行する(ループさせる)言語モデル。保存されている層は 32 で、そのうち 4〜27 を 2 回通るので、1 回の計算で 56 層分を実行する。

## 構造

| 部分 | 中身 |
|---|---|
| 層 | 保存されている層 32。層 4〜27 を 2 回実行するので、実行順では 56 層(GGUF の `block_count` も 56) |
| 2 回目の層 | 実行順の層 28〜51。重みは層 4〜27 と同じもの(GGUF にも 1 回分しか入っていない) |
| 残差 | 繰り返しの各回の中では 2 本の流れ(mHC: manifold-constrained hyper-connections)。回の入口で 1 本を 2 本に複製し、出口で平均して 1 本に戻す |
| ルーター | 各層の attention と MLP の前後にある小さな行列。2 本のどちらをどれだけ読むか、出力をどちらにどれだけ足すか、2 本をどう混ぜるかを、トークンごとに決める |
| ヘッド | 4 層ごとに full attention(16 ヘッド、幅 256)、ほかは線形注意(32 ヘッド、幅 128) |
| MLP | 12,288 ニューロン(SwiGLU) |

## 環境を作る

確かめた環境は Linux(WSL2)、CUDA 12.4、Python 3.12、RTX 3080 12 GB。改造版の llama.cpp は [GuideboardLabs/llama.cpp-sensenova-loop](https://github.com/GuideboardLabs/llama.cpp-sensenova-loop) のコミット `a7f77a788893ef2cb17c82b9ac0cb1279bcd0918`(公式の llama.cpp の `aa588a3` が元)。gne のリポジトリの [third_party/sensenova-loop/](../third_party/sensenova-loop/README.md) に、この改造版に当てる任意のパッチ 2 本と、llama-cpp-python に当てる必須のパッチ 1 本がある。

1. 改造版の llama.cpp を取ってきて、ビルドする(共有ライブラリで)。パッチ 0001・0002 は任意(どの機能に要るかは下の表)。

   ```bash
   git clone https://github.com/GuideboardLabs/llama.cpp-sensenova-loop
   cd llama.cpp-sensenova-loop
   git checkout a7f77a788893ef2cb17c82b9ac0cb1279bcd0918
   git apply <gne>/third_party/sensenova-loop/0001-sensenova-loop-control-vector-on-both-mHC-streams.patch   # 任意
   git apply <gne>/third_party/sensenova-loop/0002-sensenova-loop-per-visit-weight-names.patch               # 任意(0001 の後)
   cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON
   cmake --build build -j 8 --target llama ggml mtmd llama-server llama-completion
   ```

2. 専用の仮想環境を作り、gne が使うパッケージを入れる。llama-cpp-python は pip で入れない(公式の llama.cpp を中でビルドしてしまうため)。

   ```bash
   uv venv --python 3.12 ~/sensenova-venv
   uv pip install --python ~/sensenova-venv/bin/python gguf numpy typer pyyaml jinja2 pydantic safetensors fastapi uvicorn pytest
   uv pip install --python ~/sensenova-venv/bin/python --no-deps -e <gne>
   ```

3. llama-cpp-python 0.3.36 の Python のパッケージ(`llama_cpp/` のフォルダ)だけを取り出し、構造体の直しを当てて、仮想環境に置く。改造版の `include/llama.h` は、公式の 0.3.36 が前提にしている形と 2 か所違う(`llama_context_params` に `moe_cache_size`、`llama_perf_context_data` に `n_reallocs_same`・`n_reallocs_diff` が増えた)。直さないと、モデルを読み込むときの設定が全部ずれて読まれる。

   ```bash
   git clone --depth 1 --branch v0.3.36 https://github.com/abetlen/llama-cpp-python
   cd llama-cpp-python
   git apply <gne>/third_party/sensenova-loop/llama-cpp-python-0.3.36-structs.patch
   SP=$(~/sensenova-venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
   cp -r llama_cpp "$SP/llama_cpp"
   ```

4. 改造版のライブラリを指して gne を動かす。llama-cpp-python は `LLAMA_CPP_LIB_PATH` のフォルダから `libllama.so` を読む。

   ```bash
   export LLAMA_CPP_LIB_PATH=<llama.cpp-sensenova-loop>/build/bin LD_LIBRARY_PATH=<llama.cpp-sensenova-loop>/build/bin
   ~/sensenova-venv/bin/gne info -m SenseNova-Loop-9B-Q4_K_M.gguf
   ```

   `gne info` の表に「2 回実行する層」の行が出て、この llama.cpp で何ができるか(2 本の流れの中の層に control vector が届くか、2 回目の層に別の名前があるか)が出る。

GGUF は [guideboardlabs/SenseNova-Loop-9B-GGUF](https://huggingface.co/guideboardlabs/SenseNova-Loop-9B-GGUF) の Q4_K_M と Q8_0 で確かめた。12 GB の GPU では Q4_K_M を勧める(下の「目安」)。

| 機能 | 改造版のまま | + 0001 | + 0002 |
|---|---|---|---|
| 捕捉・方向・CHiPS の一括の流れ・FiSH・`gne nw verify` / `check`・LoRA の書き出し(書き換えは両方の回に効く) | できる | できる | できる |
| 回ごとに測る(`--score-layers loop`・`--select-layer`)、`gne heads loop-merge`・`gne heads measure` | できる | できる | できる |
| 2 本の流れの中の段(4〜26・28〜50)への control vector、そこでの `gne verify` の V3・V4・V12、FiSH の方向の段の層を重ねる読み直し(無ければ `--no-stack`) | できない(理由を出して止める) | できる | できる |
| 片方の回だけを書き換える(`GNE_LOOP_EDITS=per_visit`、`gne heads causal --visits first / second`、`gne heads visits`、`gne heads loop-pipeline`、`per_visit` と記録したレシピ・LoRA) | できない(理由と直し方を出して止める) | できない | できる |

## 層の番号

gne の層の番号は実行順(0〜55)で、llama.cpp のテンソルの捕捉の名前(`l_out-30` など)と control vector の番号と同じ。2 回目の層は、表示で「L30 = 層 6 の 2 回目」のように、重みを持つ 1 回目の層の番号を添える。`gne info` の警告にも、どの層を 2 回実行するかが出る。

## 残差の読み方

- 層の出力 `l_out-L` は、2 本の流れの平均として返す。llama.cpp のグラフの `l_out` は、2 本の中の層では片方の流れ(飛び飛びに並んだ値の参照)なので、gne はその層の最後の混ぜ合わせ(`mhc_ffn_merged`)から平均を作る。
- 2 本の平均を使うのは、回の出口で 2 本が平均されて、それが最終の出力につながるため。各層が読むのは、ルーターが 2 本を重みづけして足したもので、実測ではほとんど片方の流れ。平均から作った方向は、その層が読むものの近似になる。
- 2 本の中の層では、ほかの名前も違う。gne は、ふつうの名前に置き換えて返す: `ffn_out`(元は `mhc_ffn_out`)、`attn_output` / `linear_attn_out`(`mhc_attn_out`)、`attn_post_norm`(`mhc_ffn_norm`、MLP の入力)、`attn_residual`(`mhc_ffn_packed` の前半、MLP の入力の正規化の前)。2 本を並べたままの値(`l_out_streams-L`)と、ルーターの値(`mhc_ffn_gates-L`・`mhc_attn_gates-L`)は、その名前を指定したときだけ返す。

## 重みの書き換えは両方の回に効く

2 回目の層は 1 回目の層と同じテンソルを使うので、LoRA でも焼き込み(重みの書き換え)でも、書き換えは両方の回に効く。gne はこれを前提に次のようにする。

- LoRA のテンソル名は、重みを持つ層の名前にする。2 回目の層の番号で書いたユニット(`31:H3:x0` など)も、層 7 のテンソルに書く。
- CHiPS(ヘッドの方向成分の射影): 因果スクリーニングと配分の候補は、1 回目の層のヘッドで表す(2 回目の同じヘッドは同じ重みなので、別の候補にしない)。ヘッドの書き込みの寄与は両方の回の分を足す。同じヘッドに 1 回目と 2 回目の両方のユニットを書いたときは、2 つの射影を続けて掛けたもの W' = (I − s₂r₂r₂ᵀ)(I − s₁r₁r₁ᵀ)W を 1 つの LoRA にする(足し合わせると、2 本の向きが重なる分を 2 回引いてしまう)。
- FiSH(計算で作るニューロン): 上書きしたニューロンは、もう一方の回の層でも同じ向きを書く。
  - `gne nw check` の書いた量(amount・total)は、両方の回の合計。そのうちもう一方の回の分を `amount_twin` に出す。両方の回で鍵が反応すると書き込みは 2 回入るので、狙った量を超えていないか(over)は合計で見る。
  - `gne nw verify` は、重みを持つ 1 回目の層で式どおりかを確かめ(その前の層はこの LoRA で変わらない)、もう一方の回で上書きしたニューロンがどれだけ開いたか(`twin_act_max`)を出す。
  - ニューロンのふだんの働き(`gne nw neurons`)は両方の回を合わせて数える。同じ重みの 2 つの層(例 12 と 36)を両方選んだときは、既定のニューロンが重ならないように選ぶ。
- gain・ablate も同じ(LoRA は重みを持つ層の名前に書き、両方の回に効く)。層ごとの方向を消す ablate(`--per-layer`)の LoRA は、1 回目と 2 回目の方向の射影を続けて掛けた 1 つにまとめる。焼き込みは重みを持つテンソルを 1 回だけ書き換えるので、層ごとの方向は 1 回目の層の方向を使う。
- `gne verify` の V11(gain の確かめ)は、重みを持つ 1 回目の層で確かめる。元の llama.cpp では、2 本の流れの中の層を `--layer` に渡すと、control vector を使う V3・V4・V12 は理由を書いて省く。

### 片方の回だけを書き換える

片方の回だけを書き換えるには、2 回目の層に別のテンソル名(1 回目と同じデータを指す別名)を付けた llama.cpp が要る。gne はモデルを読み込むときに、その名前で LoRA を読めるかで確かめる。

当て方は、レシピ(`loop_edits`)と LoRA のファイル(`gne.loop_edits`)に記録する。

- `shared`(両方の回に当てる。記録の無い古いレシピ・LoRA もこれ): 2 回目の名前がある llama.cpp でも、LoRA の 1 回目の層のテンソルを 2 回目の名前にも写して当てる(書き出す LoRA のファイルは、重みを持つ層の名前だけで書く。元の llama.cpp でも読める)。
- `per_visit`(回ごとに当てる): ユニットの層の番号どおり、その回だけを書き換える(`10:N100:x0` は 1 回目だけ、`34:N100:x0` は 2 回目だけ)。CHiPS の候補には 2 回目の層のヘッドも入り、FiSH のもう一方の回の分は出ない(書かないので)。2 回目の名前が無い llama.cpp でこの形のレシピや LoRA を読むと、黙って両方の回に当てずに、理由と直し方(下の `gne heads loop-merge`)を書いて止める。焼き込み(重みの書き換え)は、どちらでも両方の回に効く。
- 環境変数 `GNE_LOOP_EDITS=per_visit` は、記録の無いもの(その場で作るユニットや、探す途中の一時的な LoRA)の既定を変えるだけ。そのとき保存したレシピには `loop_edits: per_visit` が付く。

## 1 回目と 2 回目を別々に扱う CHiPS

1 回目と 2 回目を、ある意味で別のモデルとして扱う。探す段階だけ 2 回目の名前がある llama.cpp を使い、できたレシピは元の llama.cpp で動く形(両方の回に当てる形)に直す。

| 段階 | コマンド | 元の llama.cpp |
|---|---|---|
| 回ごとに測る | `gne heads pipeline --score-layers loop --select-layer 27`(`heads causal` も同じ指定) | できる |
| 回ごとに探す | `gne heads causal --visits first / second / both`、`gne heads visits`、`gne heads loop-pipeline` | both 以外はできない(理由と代わりの方法を出して止める) |
| 仕上げ | `gne heads loop-merge`、`gne heads measure` | できる(回ごとに当てた値の列だけ省く) |

### 回ごとに測る

- `--score-layers loop` は、射影の値(残差を方向に射影した値)を、1 回目の出口(段 27)・2 回目の出口(段 51)・最後の層の 1 つ前(段 54)で測る。最後の層(段 55)は、llama.cpp が最後のトークン以外の行を捨てるので測れない。
- **このモデルでの `gne heads pipeline` の既定**(ほかの arch は今までどおり): 測る層を指定しなければ `--score-layers loop --select-layer 54` で流し、ループの外の段 52・53 は、層を止めた下がり方の順位に関係なく、いつもヘッド単位で測る(`--always-layers`)。下の「確かめたもの」で、効くヘッドが今までの既定の候補から漏れたため。今までの既定に戻すには `--score-layers base --always-layers none`。一括の出力に、この arch 用の既定で流したことが 1 行出る。
- `--candidate-layers` は、候補のヘッドを探す層(例 `28-53`)。`GNE_LOOP_EDITS=per_visit` と一緒に使うと、1 回目だけ・2 回目だけから選べる。
- `--select-layer` は、ヘッドの順位づけと配分の目標に使う層。`--select-layer 27` なら「1 回目の終わりの値で選ぶ」(段 27 での下がり方で順位をつけ、段 27 より前のヘッドを候補にし、目標は段 27 の B 側の平均)。
- 測る層を指定すると、一括の結果に、因果スクリーニングの上位のヘッドの層ごとの下がり方と、配分した組ごとの層ごとの A 側・B 側の平均が並ぶ。1 回目の終わりでどこまで下がり、2 回目でどう変わったかを 1 つの表で読める。
- 途中の残差に出力層をかけて読む値(logit lens)は目安にしかならない。このモデルは 1 回目の終わりの残差を出力層で読む前提で作られていないので、それを「1 回目の出力」とは読まない。

### 回ごとに探す(2 回目の名前がある llama.cpp)

- `gne heads causal --visits first`(または `second`): ヘッドを 1 回目だけ(または 2 回目だけ)止めて因果スクリーニングをする。`both` は両方の回で止める(既定と同じ)。
- `gne heads visits --name <方向>`: 同じヘッドを 1 回目だけ・2 回目だけ・両方で止めた効き目を並べ、回の間の順位相関(Spearman)を出す。「両方−和」は、両方で止めた下がり方から片方ずつの和を引いたもの。
- `gne heads loop-pipeline --spec <ひな形> --n-first 3 --n-second 3`: 2 段で選ぶ。
  1. 段 0〜27 のヘッドから、段 27 の射影の値で選ぶ(1 回目だけに当てる)。
  2. その組を対照集合の両側に当てた状態で、段 28〜53 のヘッドから、段 54 の射影の値で選ぶ(2 回目だけに当てる)。対照集合の側には、レシピを重ねて当てる書き方 `recipe_a: {stack: [<レシピ>, …]}` を使う。
  3. 2 つを合わせたレシピ(`<名前>_loop.per_visit.yaml`)を、次の仕上げにかける。

### 仕上げ(元の llama.cpp で動く形にする)

- `gne heads loop-merge --recipe <回ごとのレシピ>`: 両方の回に当てるレシピ(`<名前>.shared.yaml`、焼き込み・配布できる形)に直す。各ヘッドをどちらの回で選んだかは `loop_visits` に記録する。
- 直したレシピは、両方の回に当てた状態で段 27・51・54 を測り直し、回ごとに当てたときの値と並べる(探したときの値を足し合わせて済ませない)。2 回目のために選んだヘッドは、両方に当てると 1 回目でも弱まるので、その差がこの表に出る。
- 同じヘッドを両方の回で選んだときは 2 通り: 2 つの射影を続けて掛ける(compose)か、強いほうだけを使う(max)。続けて掛けると、そのヘッドは片方の回で選んだ強さより強く弱まる(同じ向きなら実効の強さは s₁ + s₂ − s₁s₂。s が 2 を超える反転どうしだと、かえって強める)。表には、こうしたヘッドの数と、続けて掛けたときの実効の強さを出す。既定は強いほうだけ(max)。公開のひな形で自然に作った組(2 段 4 回、12 ヘッドの和 2 回)には両方の回で選んだヘッドが無かったので、2 段の組の各ヘッドに、もう一方の回の同じ強さのユニットを足して比べた: disclaimer(両方の回で選んだヘッド 5 個)では、続けて掛けると段 54 の縮めた割合 −1.88・B 側のずれ 1.78(19:H2 の s30 どうしで実効の強さ −469)、強いほうだけなら 1.02・0.36。question_kinds(3 個)では 0.97・0.56 と 1.00・0.27。
- `gne heads measure --name <方向> --recipe <レシピ> …`: どのレシピでも、測る層ごとの A 側の平均、B 側の平均を超えて残る A 側の質問の数、A と B の差を縮めた割合、B 側の直交成分のずれを並べる。`--tail 51` を付けると、段 51 から最後に測る層までの射影の値の変化を切り分ける: 段ごとの残差の大きさと cos(射影の値 ÷ 大きさ)、最後の段の残差を段 51 の方向で測った値、間の段の attention と MLP の書き込みをそれぞれの方向に射影した値(和は射影の値の変化に一致する)。

## control vector

元の llama.cpp は、2 本の流れの中の層(4〜26、28〜50)に足した control vector を次の層に渡さない(足した値が片方の流れの参照にだけ入り、次の層は 2 本の流れを読むため)。gne はモデルを読み込むときに確かめて、その層に足そうとしたら止める。効くのは層 0〜3・27・51〜55。

2 本の流れの両方に足すように直した llama.cpp なら、全部の層で効く(2 本の両方に足すのは、2 本の平均に足すのと同じ。2 本の差は変わらない)。FiSH の方向を作る段の「層を重ねる読み直し」も control vector を使うので、元の llama.cpp では `--no-stack` で層ごとに別々に方向を作る。

## 思考(thinking)

チャットテンプレートは、既定で思考を入れる(生成プロンプトの最後が `<think>\n`)。gne は方向づくり・鍵づくり・生成のどれでも `enable_thinking=false` で描くので、プロンプトの最後は空の思考の欄 `<think>\n\n</think>\n\n` になる。応答の位置 0 は、その最後のトークン(`</think>` の後の改行)で、応答の最初のトークンを出す位置。3 か所とも同じ描き方なので、位置はそろう。

## CHiPS と FiSH を流す(確かめたもの)

公開のひな形で、次の順に流して最後まで動くことを確かめた(Q4_K_M と Q8_0、元の llama.cpp)。

```
gne heads pipeline --spec configs/pipeline/disclaimer.json --target B --n-heads 10,6,4,2,1 -m <model>
gne nw pipeline --contrast configs/fish/closing_rmbase.json --layers 40-46 --remove rmbase --kscale auto \
    --iv <heads pipeline が書いたレシピ(…alloc-n10.yaml など)> --no-stack --save <名前> -m <model>
gne nw verify --recipe <名前> --iv <同じレシピ> --contrast configs/fish/closing_rmbase.json -m <model>
gne nw check  --recipe <名前> --iv <同じレシピ> --contrast configs/fish/closing_rmbase.json -m <model>
```

- `--no-stack` は元の llama.cpp のとき(上の「control vector」)。2 本の流れの両方に足すように直した llama.cpp なら付けない。
- 層 40〜46 は動作の確認のための選び方(2 回目の層。上書きしたニューロンが 1 回目の層 16〜22 でも動くので、その数え方を確かめられる)で、勧める層ではない。
- この確認では、もう一方の回(層 16〜22)で書いた分が大きかった。区間 1〜3 では合計の約半分がもう一方の回の分で、狙った量(1)を超える質問が多い。位置 0 では、もう一方の回が逆向きに書いた(成分を取り除く形 !rmbase の基準値は、ユニットの層で中立の文から測った値なので、もう一方の回では合わない)。2 回目の層に書くときは、この合計で強さを決めるか、繰り返しの外の層(0〜3・52〜54)を使う。片方の回だけに書くには、2 回目の層に別の名前がある llama.cpp で `GNE_LOOP_EDITS=per_visit` にする(同じ確認を Q4_K_M で流すと、位置 1〜3 の書いた量の平均は 0.80〜1.05 から 0.62〜0.73 に、位置 0 の逆向きの分は −0.22 から +0.01 になった。当てた CHiPS のレシピは、それぞれの当て方で探したもので同じではない。この LoRA はその llama.cpp でしか当てられない)。
- Q4_K_M では `gne nw verify` が、入力の行の誤差(4〜10%)で許容(5%)を超えて不合格になった。量子化の丸めによる差で、Q8_0 では全部の層で合格(2〜4%)。

### 回ごとの CHiPS(確かめたもの)

公開のひな形 `disclaimer.json` と `question_kinds.json`(各側 16 問)、Q4_K_M、2 本の流れの両方に足し 2 回目の層に別の名前を付けた llama.cpp で、同じヘッド数(6)の組を候補の範囲を変えて作り、段 54 で比べた。どれも目標は段 54 の B 側の平均で、同じ方向で測った。値は、段 54 の A と B の差のうち縮めた割合(1 で B 側の平均に届く)/ B 側の直交成分のずれ。「両方」は両方の回に当てた形(配れる形)。

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

## 目安(Q4_K_M、RTX 3080 12 GB)

| 処理 | 時間・メモリ |
|---|---|
| llama-server(文脈 8,192) | GPU 約 5.7 GB(画面の分を除く)、生成 約 52 tok/s |
| 文脈の長さ | 65,536 まで余裕あり(約 9 GB)。131,072 は 12 GB にほぼいっぱい |
| `gne heads pipeline`(各側 16 問) | 約 4 分(方向 3 秒、因果スクリーニング 3 分、配分 1 分強)、GPU 約 6 GB。Q8_0 は約 5 分(`--n-ctx 2048` で GPU 最大 11.5 GB) |
| `gne nw pipeline`(層 7 つ、質問 12 + 8) | 約 2 分 20 秒 |
| `gne nw verify --contrast`・`gne nw check` | それぞれ約 20 秒 |
| `gne heads loop-pipeline`(各側 16 問、3+3) | 約 8〜9 分(一括の流れ 2 回と、仕上げの測り直し) |
| `gne heads loop-merge`(測り直しつき)・`gne heads measure`(レシピ 9 個、`--tail 51`) | 約 10 秒・約 1 分 |

止めて測る段階(射影の値・因果・配分)は、質問を最大 64 個に間引く(`max_examples`)。各側 300 問なら、方向づくりは全部の質問を読むので 1 分ほど、因果と配分は 64 問で測るので、合わせて 20 分前後の見込み。
