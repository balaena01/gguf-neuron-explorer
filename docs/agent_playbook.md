# エージェント向け手順書(gne の判断の仕方)

`gne mcp` のツール(または `gne ... --json`、`gne.api`)を使うエージェント向け。ツールの使い方ではなく、**何を見て、どう判断するか**を書く(設計書 §6.13 (e))。数値の根拠は `docs/verification.md` と `docs/measurements.md`。

## 1. まず全体を見る

1. `model_overview` で、層ごとに何があるか(coverage)を見る。`rho`(残差ノルム)・`act_stats`・`logit_lens`(済みのニューロン数)・`steer_lens`・`sweep_runs`・`directions`。
2. 目当ての層で `layer_summary(L)`。sweep の「対照群超えの割合」と上位ユニット、steer lens の KL の分布、logit lens の上位、逆引きの上位。
3. 無いデータは `job_start` で作る。所要時間の目安(Qwen3.5-4B、RTX 3080):

| kind | 目安 | 使いどころ |
|---|---|---|
| `stats_resid` | 16 秒(全層) | rel dose に必須。無ければ steer が自動で作る |
| `lens_logit` | 33 秒/層(CPU) | 最初のふるい。全層でも約 17 分 |
| `lens_steer` | 22 分/層(全ニューロン)、少数なら `steer_lens` ツールで即時 | 後段の層を通した効果 |
| `sweep`(`quick`) | 約 3 秒/ニューロン | 候補の確認 |
| `sweep`(`main_rel`) | 6.4 秒/ニューロン(385 本で 48 分) | 順位づけ(rank)の本番 |
| `rank` | 数秒 | sweep の後に必ず |
| `reverse` | 6 秒、`verify: true` なら 20 候補で約 4 分 | 狙った性格・話題から候補を探す |

層の全ニューロンに `main_rel` を回すと約 16 時間かかる。logit lens → steer lens → 逆引きで候補を数百に絞ってから sweep する。

## 2. 本物かどうかの判断

- **対照群を超えないものは報告しない。** sweep は `null_pct`(ランダム方向の対照の何割より mass が大きいか、1.0 = 全対照より上)、steer lens は `kl_null_p95`(対照の KL の 95% 点)と `pct_up` / `pct_down` を見る。
- **「名乗りが変わる」「作者名・会社名が変わる」は安い効果。** ランダム方向でも起こる(実際に対照で Kazhu・QingXing・Giant などに名乗りが変わった)。これだけでは何も言えない。
- **複数のプロンプトで同じテーマが出るものだけを本物とみなす。** 調書の `per_prompt`(語がどのプロンプトで出たか)と `rollouts` を見る。例: L22 N2742 は who で「extraterrestrial intelligence from the planet Xeno-888」、snake で「extraterrestrial alien lifeform」、orange で「a vibrant, extraterrestrial color」。
- **steer lens の KL は話題のニューロンを見分けにくい。** KL が大きいのは if・when・you のような機能語を書くニューロンが多い。話題のニューロン(N2742)は KL では層の真ん中あたり。見るのは `up` / `down` の語の中身。
- **語の単位で順位が変わる。** 既定(トークン単位)では、複数トークンに割れる語(extraterrestrial → extr・ater・restrial)が重く数えられる。迷ったら `rank` を `word_unit: "regex"` でも回して見比べる(英語のとき)。
- 逆引きの候補は、残差空間で相関する方向に書き込むだけで、各ニューロンは多義的。**必ず steer か rollout で確かめる。** 逆引きは cos が負の候補もある(`sign: -1` は − で押すと目標方向)。

## 3. dose の読み方(rel dose、Qwen3.5-4B の経験値)

- 0.5 で微かな挿入、0.8 で話題を上書き、1.0 以上は崩壊域(反復・意味不明)。3.0 は崩壊させて何を書きたがるかを見る用。
- 層 0〜2 は残差が小さく、rel dose がそのまま通用しない。層 0 は control vector で押せない(bake なら可)。
- 小さいモデル(0.6B など)は 0.8 でも崩れやすい。

## 4. レシピを作る

1. 「押したい方向(+)」と「抑えたい方向(−)」を分けてユニットを組む。例: `["14:8949:+1.0", "13:7956:-1.0"]`。
2. **抑える用途(言語の混入、決まり文句、特定の話題)は、add の − より先に gain の k=0 を試す。** `"14:8949:x0"` はそのニューロンの出力を消すだけで、働いていない場面には何もしない。add の − は全位置に逆向きの方向を足すので、関係ない文まで変わりやすい。強めたいときも `x2`〜`x4` を試す価値がある(Qwen3.5-4B の L22 N2742 は「Who are you?」では x10 でも変わらず、地球外生命の質問でだけ強く出た。verification.md §11.2)。
3. `steer(units, compare=True)` で無介入との差を確かめる。差が「名乗りだけ」なら安い効果を疑う。別のプロンプトでも試す。
4. **位置(`when`)は `gen` を第一候補にし、`all` と比べる。** `gen` はプロンプトの読み取りを素のままにして生成だけを押すので、質問の理解が崩れにくい。4B の alien_mcp では、倍率 1 で性格の変化量はほぼ同じ(新語の割合 0.64 と 0.71)まま、能力の低下が 19.4pt → 2.8pt に減った(§11.3)。最初の 1 トークンは押されないので、名乗りの最初の語から変えたいときは `assistant`。
5. `recipe_save(name, units)` → **配る前に `eval(recipe=name)` を通す。** 倍率ごとに能力(exact・format)・ppl・言語の混入・性格の変化量が並び、配布可(能力 −5pt・ppl +10%・混入が増えない)の最大倍率が出る。報告には「配布可の最大倍率」と、その倍率での変化量を添える。ppl が最初に引っかかることが多い(参照テキストの全位置に当てるため)。`when` を変えて比べるときは `eval(recipe=name, when="gen")`(保存名に `@gen` が付く)。
6. 書き出し: add のユニットは `export_cvec(name)`。書き出した cvec は llama-cli / llama-server の `--control-vector` でそのまま使える(`--control-vector-scaled FILE:0.6` で一括で弱められる)。gain のユニットは `gain_export(name)` で LoRA アダプタ GGUF(`--lora`、Qwen 系でも可)。`when` は control vector では表せないので、位置別のまま使うなら `gne serve --openai` の API を案内する。
7. Llama 系の arch なら `bake(name, dst)` で FFN バイアスに焼き付けた GGUF も作れる。Qwen・Gemma 系は不可(ARCH_UNSUPPORTED)。

## 4.1 方向で押す・消す(`direction` / `ablate`、本体設計 §6.17)

ニューロン単位では断片しか見つからない性格・振る舞い(ヤンデレ、拒否、口調)は、対照集合から方向を作って扱う。

1. 対照集合を書く(`configs/contrast/*.yaml`: `system_a`(目標の性格)/ `system_b`(中立)/ `user_prompts`)。層選びや比較に使う判定 `validate.match` も添える。
   - **判定語は「その振る舞いにだけ出る語」にする。** 断りや自己紹介にも出る語は避ける。evil の harm・cruel は、消したあとの「I do not want to harm humans」「harmless」に当たって、効果を逆向きに測った(verification.md F86)。
   - 口調・決まり文句・特有の言い回し(pathetic・carbon-based・ダーリン・離さない)を選ぶ。
   - 正規表現では難しいときは `--judge-model`(別の GGUF に 1〜5 で採点させる)。
2. `direction(contrast, select="auto", mode=...)` で全層の方向を作り、層を選ぶ。
   - **消したいとき**は `mode="ablate"`(既定): 方向を全層から消して一致率が下がり、B 側の KL が 0.1 以下の層を選ぶ。
   - **足したいとき**は `mode="add"`: 層 L に平均差の dose 倍を足して一致率が上がり、distinct-4 が 0.8 以上の層を選ぶ。選べる層が無いことも多い(強めると反復に崩れる)。表の rel 列を見て、`steer(units=["L:dir:NAME:+rel"])` で手で詰める。
3. **相手の行動に反応する性格(嫉妬・詮索)は「Who are you?」では出ない。** 「友達と出かける」「同僚とランチした」のような、その性格が反応する場面のプロンプトで確かめる(verification.md §12.1 のヤンデレ)。
4. **消したい振る舞いは、まず `direction` で方向を作り、`ablate` と上位ニューロンの `gain` ×0 を `eval --compare-ablation` で比べてから選ぶ。**
   - 4B の evil では、上位 20 ニューロンを ×0 にしても evil の口調は半分残った。方向の消去は全部消えたが、ppl が +16% 悪化した。倍率 0.5 なら配布可(§12.3)。
   - 部分消去(`ablate="NAME:0.5"`)を候補に入れる。
5. 書き出し: 消去は `ablate_export(name)` で LoRA アダプタ GGUF(llama.cpp の `--lora`、`--lora-scaled` で部分消去)。重みを書き換えた GGUF は `ablate_export(name, bake=True)`(ジョブ、数 GB)。
   - 出力層と埋め込みを共有する(tied)モデルでは、LoRA に埋め込みは入らない。完全に消すなら bake。
   - 方向を足すユニット(`L:dir:NAME:dose`)は `export_cvec` で control vector にできる。

## 5. 報告の仕方

- 調書(`neuron`)の数値をそのまま引用する: sweep の順位(`rank` / `n_ranked`)、`mass`、`null_pct`、上位語、steer lens の KL と対照の 95% 点、逆引きの cos。そのニューロンを含む eval があれば(`evals`)、配布可の最大倍率も。
- **事実と解釈を分ける。** 「N2742 は sweep で 385 本中 3 位、全対照より上、3 つのプロンプトで extraterrestrial が出た(事実)。宇宙人・地球外知性の話題に寄せるニューロンとみられる(解釈)」。
- greedy の生成は、バッチの組み方の違いで上位候補が僅差の箇所だけ分かれることがある(数値誤差、verification.md F2)。1 トークンの違いに意味を読まない。

## 6. UI に見せる(gne serve が動いているとき)

人が Web UI(`gne serve`、既定 http://127.0.0.1:8791)を見ているなら、`gne mcp --ui-url http://127.0.0.1:8791` で繋ぎ、作業の要所を UI に出す。同じマシンで UI がモデルを GPU に載せているなら `--n-gpu-layers 0` を付ける(steer・job は遅くなるので、その間は UI 側で回してもらう)。

- `ui_open(layer, neuron, tab)`: 調べているニューロンを NEURON パネルで開く(地図はその層の正面ビューになる)。`tab` は examples / chat / details / web。
- `ui_highlight(units, label)`: 候補を強調する(水色の破線の輪)。前の強調は置き換わり、空のリストで消える。label には「何の候補か」を短く。
- `ui_note(layer, neuron, text)`: 判断の要点を貼る(400 字まで、地図には 28 字まで)。**事実と解釈を分けて**書く(例「事実: sweep 3 位/385・全対照超え、3 プロンプトで extraterrestrial。解釈: 宇宙人の話題に寄せる」)。見送った候補にも理由を貼ると、人が同じ確認をしなくて済む。空の text で消える。
- UI に繋いでいなければ `UI_NOT_RUNNING` が返る。作業はそのまま続けてよい。

