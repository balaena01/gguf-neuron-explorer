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

**見積もりが 30 分を超えるジョブは走らない(CHANGE-03 C)。** `job_start` は `NEEDS_CONFIRM` と見積もり(`estimate`)を返して止まる。`params` に `budget: "30m"` を入れると収まるように絞る(lens_steer は方向に近いニューロンの上位、sweep は quick 設計で steer lens の KL 上位、reverse は確かめる候補の数、connections は window)。それでも走らせるなら `yes: true`。

**大型モデル(GPU に載らない、または全数の lens steer が 1 層 1 時間を超える)は逆引きから始める。** 逆引き → 上位 N の lens steer → quick sweep → eval の順。`gne info` の末尾に手順と見積もりが出る。全数の lens・sweep・connections(window なし)は回さない。

## 2. 本物かどうかの判断

- **対照群を超えないものは報告しない。** sweep は `null_pct`(ランダム方向の対照の何割より mass が大きいか、1.0 = 全対照より上)、steer lens は `kl_null_p95`(対照の KL の 95% 点)と `pct_up` / `pct_down` を見る。
- **方向も対照を超えているかを見る。** `direction` の層選びは層ごとに同じ大きさのランダムな向き(既定 16 本)を流し、`ctrl_effect_p95` を超えた層だけを選ぶ。レシピは `gne steer --contrast` の `behavior.null_pct`、eval は `--random-controls N` の `novelty_null_pct` と `ctrl_*_p50`(大きさだけで起きる能力・ppl の悪化)。**`null_pct` が 0.95 未満の方向は報告しない。**
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

1. 対照集合を書く(`configs/contrast/*.yaml`: `system_a`(目標の性格)/ `system_b`(中立)/ `user_prompts`)。層選びや比較に使う判定 `validate.match` も添える。system は片方だけでもよく(省いた側は system なし)、A と B をレシピで作る(`recipe_a` / `recipe_b`)ことも、側ごとに別の質問を読ませる(`prompts_a` / `prompts_b`、何もしないモデルに 2 種類の質問を読ませる)こともできる。別の質問を読ませるときは、狙った違い以外(話題・長さ・言語)をそろえる。
   - **判定語は「その振る舞いにだけ出る語」にする。** 断りや自己紹介にも出る語は避ける。evil の harm・cruel は、消したあとの「I do not want to harm humans」「harmless」に当たって、効果を逆向きに測った(verification.md F86)。
   - 口調・決まり文句・特有の言い回し(pathetic・carbon-based・ダーリン・離さない)を選ぶ。
   - 正規表現では難しいときは `--judge-model`(別の GGUF に 1〜5 で採点させる)。
2. `direction(contrast, select="auto", mode=...)` で全層の方向を作り、層を選ぶ。
   - **消したいとき**は `mode="ablate"`(既定): 方向を全層から消して一致率が下がり、B 側の KL が 0.1 以下の層を選ぶ。
   - **足したいとき**は `mode="add"`: 層 L に平均差の dose 倍を足して一致率が上がり、distinct-4 が 0.8 以上の層を選ぶ。選べる層が無いことも多い(強めると反復に崩れる)。表の rel 列を見て、`steer(units=["L:dir:NAME:+rel"])` で手で詰める。
   - **性格を足す方向は `positions="response"` で作る**(A / B の system で返事を生成し、返事のトークンで測る)。指示の直後(`last`)より弱い dose で効き、反復に落ちにくい(verification.md §12.4)。
   - **強さは greedy の反復だけで決めない。** greedy では「待て待て待て…」に落ちる強さでも、サンプリング(llama-server の既定)では落ちないことが多い。最後は、使う人の llama-server に cvec を当てて確かめる。
   - 一人称や語尾が揺れる(「私」と「俺」が混ざる)ときは、話し手の性別の方向(`configs/contrast/female_speech.yaml`)を少しだけ足す。性格の方向とほぼ直交しているので、性格は残る。
3. **相手の行動に反応する性格(嫉妬・詮索)は「Who are you?」では出ない。** 「友達と出かける」「同僚とランチした」のような、その性格が反応する場面のプロンプトで確かめる(verification.md §12.1 のヤンデレ)。
4. **消したい振る舞いは、まず `direction` で方向を作り、`ablate` と上位ニューロンの `gain` ×0 を `eval --compare-ablation` で比べてから選ぶ。**
   - 4B の evil では、上位 20 ニューロンを ×0 にしても evil の口調は半分残った。方向の消去は全部消えたが、ppl が +16% 悪化した。倍率 0.5 なら配布可(§12.3)。
   - 部分消去(`ablate="NAME:0.5"`)を候補に入れる。
5. 書き出し: 消去は `ablate_export(name)` で LoRA アダプタ GGUF(llama.cpp の `--lora`、`--lora-scaled` で部分消去)。重みを書き換えた GGUF は `ablate_export(name, bake=True)`(ジョブ、数 GB)。
   - 出力層と埋め込みを共有する(tied)モデルでは、LoRA に埋め込みは入らない。完全に消すなら bake。
   - 方向を足すユニット(`L:dir:NAME:dose`)は `export_cvec` で control vector にできる。

## 4.2 反応性をヘッドで作る(CHANGE-04)

ニューロンの add / gain は「何を言うか」を変える。「何に気づくか」(褒められたら照れる、他の人の名前に反応する)は attention 側にある。

1. 対照集合に `triggers:`(反応してほしい語。全プロンプト共通のリスト、プロンプトごとのリスト、`"re:"` で正規表現)と `reactivity:`(`{with, without}` の対。同じ場面からトリガーだけを抜く)を足す(例 `configs/contrast/yandere_react.yaml`)。
2. **反応性(トリガーの有無で差が出るか)はヘッドで作る。** まず `heads_contrast(contrast)` → 上位を `x0` / `x2`(`"L:Hh:x0"`)で `eval(reactivity=対照集合, head_controls=8)` → 対照(同じ k のランダムなヘッド)を超えたものだけ残す。
   - 読み方: `trigger_mass` が大きいヘッドは「気づく係」、`dir_cos` が大きいヘッドは「ペルソナの方向を書く係」。両方大きければ本命。
   - z の上位は system の長さの差で動くヘッドが多い。気づく係を探すときは `--sort trigger_mass` で並べる。床(中立どうしの p95)を超えないものは報告しない。
   - 4B のヤンデレでは、ペルソナで名前への注目が増えたヘッドは 1 個だけだった(verification.md F108)。「ペルソナの system を入れると注目が変わる」とは限らない。
3. ヘッドは向きを持たないので add はできない。向きを足したいなら方向(`L:dir:NAME:dose`)を使う。
4. `attention(text, layer, head)` で、そのヘッドが各トークンからどこを見ているかを確かめる(T × T のパターン)。
5. (実験的)ヘッドに「ペルソナのときだけトリガーを見に行かせる」なら、`gne heads query --layer L --head h --direction D --seek C` で、scale ごとの注目をペルソナあり/なしで確かめてから、レシピに `{kind: head, head, mode: query, direction, seek, scale}` を入れる。注目が増えても返事の性格が増えるとは限らないので、最後は `eval(reactivity=...)` で確かめる(verification.md §14.4)。

## 4.3 MoE のモデル(CHANGE-05)

1. 専門家のニューロンは `L12 E37 N511`(ユニットは `"12:E37:N511:..."`)。**MoE の add は方向押し**(専門家が選ばれたかに関係なく、その書き込み方向を全トークンに足す)。ニューロンを押すなら gain(`x2`、選ばれたときだけ効く)。専門家全体は `L:Ee:x0` / `x2`、ルーターのバイアスは `L:Ee:route:+d`(バイアスを持つ arch だけ、±1 でも強い)。
2. まず `stats act`(専門家ごとの選択率と選ばれるトークン)と `experts_contrast(contrast)` で専門家を絞る。`usage_shift` が大きい = ペルソナのときに呼ばれる、`contrib_cos` が大きい = ペルソナの方向を書く。床(中立どうしの p95)を超えないものは報告しない。
3. 絞った専門家を `steer(["L:Ee:x0"])` / `x2` と `experts_sweep(L)` で試す。専門家 1 つを消しても、system で作った性格はあまり減らない(granite の海賊で 8 個消して語が 20 → 13)。
4. その専門家の中のニューロンは `expert(L, e)` の上位から、gain(`x0` / `x2`)で試す。`n_routed` が 200 未満の専門家は分位点を信頼できない(`reliable: false`)。
5. 専門家全体の gain と route はスケールパッチで、生成のたびに作業用コピーを読み直す(1.4 GB で 0.3 秒、5 GB で 0.9 秒)。多数を試すときは eval より `experts_sweep` が速い。

## 4.4 CHiPS: ヘッドの方向成分だけを変える(CHANGE-06、手法は [head-component-steering.md](head-component-steering.md))

全層射影(4.1 の ablate)が全ライターから方向を一律に消すのに対し、因果で選んだ数個のヘッドの書き込みのうち方向成分だけを、ヘッドごとの強さ s で変える(`"L:Hh:sS@NAME"`、レシピは `{layer, kind: head, head, mode: proj, direction, s}`)。s = 1 でそのヘッドの方向成分が消え、2 で反転、直交成分は変わらない。ヘッドの gain(`x k`)は直交成分も k 倍にするので、同じ効き目でも副作用が大きい(0.6B と 4B の evil で、同じ点数の水準にそろえると、残差の直交成分のずれは gain が proj の 2〜11 倍、能力は gain だけが −14〜−100pt 落ちた。4B では全層射影も ppl +15%、verification.md §20 F126・F128)。

1. **方向を使う介入は、まず `direction_score` で基準層と分離度を確認してから進める。** cohen_d が最大の層が基準層として方向に記録され、後段の層の方向とのコサイン(方向が層でどれだけ回るか)も出る。cohen_d が小さい(1 未満)方向は、そもそも A と B を分けていない。
2. `heads_causal(name, mode="ablate")` で候補を出す(層単位で止めて下がる層 → その層のヘッドを 1 個ずつ)。行の `contrib`(直接の寄与)と `causal`(実測)の比 `ratio` が 1 より大きければ間接の経路、小さければ補い合い。**物差しの層より後ろのヘッドは contrib が 0 になる**ので、`causal_max`(後段の層の下がり方)で取りこぼしを見る。
3. **`heads_causal` の ablate と marginal は順位が違う。** 最初の候補出しは ablate(丸ごと止めたときの効き目)、配り直しは marginal(いまの組から s を上げたときの効率 η)。`heads_allocate` が marginal を使って、s ∝ η で配り直し、ヘッド数を減らした組を比べる。
4. **候補と強さの案は生成なしの物差しで作り、止める強さは生成で確かめる。** `direction_score`・`heads_causal`・`heads_allocate` の点数は押した量を表すだけで、振る舞いそのものではない。迎合の例では、基準モデルで作った「点数と振る舞いの関係」から、×3 のレシピの乗り換えは 0.2 問と予測されたが、実測は 18 問だった。生成なしの点数で倍率の順番は作れるが、値から振る舞いは出せない。
5. **報告には必ず外から見える指標を主指標にする。** eval の `rate`(対照集合の `validate.match` の一致率)・能力・ppl。方向の射影値(`dir_drop_a`)だけでは「方向で押したものを方向で測った」ことにしかならない。eval の方向の指標は、既定で質問を 64 問に間引いて測る(`dir_max_examples`)ので、何百問の対照集合でも重くならない。一致率の返事は確かめ用の質問(`validate.prompts`、無ければ A 側)から選ぶ。
6. **ランダム対照と方式比較を通していないレシピは報告しない。** `eval(random_controls=N)` は同数のヘッドをランダムに選び、s の分布は元の組と同じにして当てる(`control_pool="same-layers"` は同じ層から選ぶ厳しい対照)。`rate_null_pct`(無介入の一致率が 0 なら `dir_null_pct`)が 0.95 未満なら、eval の `notes` にそう書かれる。`eval(compare_methods=True)` で、全層射影・ヘッドの方向成分・ヘッドの丸ごと倍率・ニューロンの消去を、A 側の点数の水準をそろえてから比べる。そろえられなかった方式は理由つきで除外される。
7. 目的指標が数 % まで来たら強さを上げるのをやめ、残る例を担う別のヘッドを探す(数 % はサンプリングの揺れ)。残る例が多ければ `heads_refine(name, n_heads)`(CLI `gne heads refine`)で、残りの質問だけから(組を両側に当てたまま)方向を作り直して足すヘッドを探し、元の組と合わせたレシピ(`combined`)を eval とランダム対照にかける。
8. **s > 1 は反転なので、B 側でそのヘッドが逆向きに書いている分も反転して狙いと逆に押す。** 強さを上げるたびに B 側(`b_side_delta`・`drift_orth_b`)を見る。効率が高くても B 側を動かすヘッドは `b_side_max` で除く。
9. 書き出しは `headproj_export(recipe)`(LoRA。同じ層・同じ方向は階数 1 に畳まれる)。焼き込み(`bake=True`)はスケールパッチでは表せないので、書き換えるテンソルだけ量子化し直す。焼き込み版は LoRA 版と点数・最初のトークンの分布・ppl が一致するかを `verify` で確かめる(V26)。control vector には入らない(足し算ではない)。
10. **一通り回すだけなら `heads_pipeline(spec)`(CLI は `gne heads pipeline --spec`、UI でも JSON を読み込んで回せる)。** 方向づくり → 基準層 → 因果 → 配分を 1 回で回す。質問は何百個でもよく、止めて測る段階は `max_examples` 個(既定 64)に間引く。止めると中立の側の残差が大きく動く(直交成分のずれ > 0.5)層とヘッドは、モデルを壊しているだけなので自動で外れる。目標を省くと A と B の平均の真ん中で、振る舞いは半分ほどしか消えないことが多い。消し切るなら `target: "B"`(B 側の平均)。s max が `s_max`(既定 30)に張り付いたら届いていないので上げる。結果の推す組は、そのまま eval のランダム対照にかける。止めて同じ spec・同じ設定で流し直すと、作り済みの方向と因果を使い回して配分からやり直す(作り直すときは `resume: false`)。大きいモデルでは、走っている間に生成(チャット・eval)を挟まないほうが VRAM に余裕が残る。spec は `configs/pipeline/` の用途別のひな形(system の性格・2 つの性格の差・レシピの性格・質問の種類の差)から書き換えると早い。
11. 質問は会話の履歴(user と assistant が交互で user で終わる)でも書ける。1 問ずつまっさらな状態から読ませ、最後の user の発話への返事の直前で測る。同じ返事を 2 つの system で読ませる道具には使えない。assistant で終えると書きかけの返事になり、その最後のトークンで測る(`<think>` で始めると thinking の途中)。返事を書いている途中で起きる振る舞いは、返事の直前の点数には出ないので、こちらで対を作る。
12. **レシピで作った振る舞いのヘッドを探すときは、対照集合の A 側をレシピにする。** 対照集合は system の代わりに `recipe_a`(と `recipe_b`)で両側を作れる(例 `configs/contrast/evil_neurons.yaml`)。両側の差がレシピの効き目そのものになり、方向づくり・物差し・因果のスクリーニングが、そのレシピを当てた状態で測られる。ニューロンを弱めた分をヘッドの増幅で補うときの探し方(verification.md §20.10)。system の違いで測る道具(heads contrast・writers・experts contrast)には使えない。
13. 振る舞いを「足す」には、A と B を入れ替えた対照集合で方向を作る(目標はいつも A 側の点数を下げること)。負の s でも方向成分は強まるが、元から方向に書いているヘッドでしか効かない。
14. **配る前は `fieldtest`(CLI `gne fieldtest -m 焼き込み --against 元`)で、ファイルどうしを本番と同じ動かし方で比べる。** eval は短い問題で倍率違いを見るもので、長い思考の途中のループ・本文が空のままの返事・ツール呼び出しは見えない。eval を通っても「長い作業やツール呼び出しは元と同程度」とは言えない。`same_level`(割合は 2 問ぶん、perplexity は 1% まで)と各行の数字を報告する。能力を配布ファイルで示すなら `bench=["gsm8k", "mmlu_pro"]`。eval の能力はレシピを当てた元のファイルの値で、焼き込みは量子化し直しで効き方が変わることがあるので、配布ファイルの値としては書かない。
15. **組の形と大まかな強さは物差しで決め、全体の倍率は返事を読んで決める。** `heads_candidates(alloc)`(CLI は `gne heads candidates --alloc`)が、比較表の組 × 全体の倍率(既定 0.5・1・1.5・2・3、負の倍率は B 側で強める向き)と無介入で、同じ種のサンプリングで返事を作り、採点する側の返事をどの候補か伏せて 1 つのファイルに並べる。採点は 1 回の比較のあいだ同じ採点役で、鍵(`round_01.key.json`)を見る前に全部付ける。調整に使っていない質問を `prompts` で渡す(渡さないと `held_out` が false になり、良く見えすぎる)。`heads_candidates_tally` の目的の点(消す向きは min(6 − P, C)、強める向きは min(P, C))で比べ、差が誤差(24 本の平均で ±0.3 くらい)より小さければ、倍率の小さいほうを採る。選んだ倍率は `eval --doses` で能力と ppl を確かめる。倍率を細かく自動で探す(Optuna など)のは、採点の揺れより細かい差を追うことになるので、まず候補を少数に絞って読む。

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

