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
2. `steer(units, compare=True)` で無介入との差を確かめる。差が「名乗りだけ」なら安い効果を疑う。別のプロンプトでも試す。
3. `recipe_save(name, units)` → `export_cvec(name)`。書き出した cvec は llama-cli / llama-server の `--control-vector` でそのまま使える(`--control-vector-scaled FILE:0.6` で一括で弱められる)。
4. Llama 系の arch なら `bake(name, dst)` で FFN バイアスに焼き付けた GGUF も作れる。Qwen・Gemma 系は不可(ARCH_UNSUPPORTED)。

## 5. 報告の仕方

- 調書(`neuron`)の数値をそのまま引用する: sweep の順位(`rank` / `n_ranked`)、`mass`、`null_pct`、上位語、steer lens の KL と対照の 95% 点、逆引きの cos。
- **事実と解釈を分ける。** 「N2742 は sweep で 385 本中 3 位、全対照より上、3 つのプロンプトで extraterrestrial が出た(事実)。宇宙人・地球外知性の話題に寄せるニューロンとみられる(解釈)」。
- greedy の生成は、バッチの組み方の違いで上位候補が僅差の箇所だけ分かれることがある(数値誤差、verification.md F2)。1 トークンの違いに意味を読まない。
