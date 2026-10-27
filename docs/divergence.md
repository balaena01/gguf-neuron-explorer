# 設計とのずれ

現行コードが設計書(本体設計 `gguf-neuron-explorer-DESIGN.md`、別冊 UI 設計、変更指示 CHANGE-01 / CHANGE-02)と違うところの一覧。変更指示を当てる前に洗い出し(CHANGE-01 §0.1)、変更のたびに更新する。

設計書はリポジトリに含めていない(手元の `gguf-neuron-explorer-design-backup/` が正)。Phase 0〜5 で見つけたずれと設計への追記は、`docs/verification.md` の各「設計と違った点・設計への追記」節(F1〜F73)に理由つきで記録してある。ここでは変更指示に関係するものだけをまとめる。

## 1. 変更指示の前に確かめたこと(2026-10-27)

| 箇所 | 設計 | 現行コード | 扱い |
|---|---|---|---|
| レシピ(§6.5) | `model_id`・`dose_unit`・`units`、`when`(§6.15)、ユニットに `mode: add\|gain` と `k`(§6.16) | `name`・`model_id`・`dose_unit`・`units`(`layer`・`neuron`・`dose`・`kind`・`seed`)。`when` と `mode` / `k` は無い | CHANGE-01 B で `when`、C で `mode` / `k` を足す。`name` は残す(UI・保存名に使っている) |
| `Engine.generate_greedy`(§6.2 / §6.15) | `schedule: Callable[[step, seq_state], vectors \| None]` を持つ | `(ctx, seqs, *, max_tokens, top_k, stop_on_eog, prefilled, on_token)`。prefill は中で行うか `prefilled` で渡す | CHANGE-01 B で `schedule` を足す。プロンプトを区切って prefill するために `prefill` に開始位置(`pos0`)を足す |
| CLI `steer` | `--recipe` / `--unit`(`L:N:x2` で gain)、`--prompt`、`--compare`、`--when` | `--recipe` / `--unit`、`--dose-unit`、`--scale`、`--prompt`、`--system`、`--max-tokens`、`--compare` | `--when`(B)と `x` 接頭辞(C)を足す。`--dose-unit` / `--scale` / `--system` は残す |
| CLI `chat` | `--recipe`、`--dose`、`--when` | `--recipe` / `--unit`、`--dose`、`--left` / `--right` / `--center`、`--prompt`(複数で多ターン)、`--system`、`--stream` | `--when` を足す |
| CLI `export-cvec` | `--recipe`、`-o`、`--format` | 加えて `--verify`(llama-completion と出力を比べる)、`--verify-prompt`、`--llama-completion`、`--unit`、`--scale` | `when != all` のとき警告して `all` で書き出す(B) |
| CLI `mcp` | `--model`、`--ui-url` | 加えて `--n-gpu-layers`(UI と同じ GPU を使うとき 0 にする) | そのまま |
| CLI `serve` | `--port`、`--openai` | `--port`、`--host`、`--root` | `--openai` を足す(B) |
| `--control-vector-scaled` の書き方(§7 の例) | `FILE 0.6` | この llama.cpp では `FILE:0.6`(F11) | 文書側の例を直す |
| 逆引きの位置(CHANGE-02 §0.3) | `positions: last`(生成プロンプトの最終トークン) | `positions` は `all` / `last` / `assistant` | 既にある |

## 2. 変更指示で生まれたずれ

| 箇所 | 設計 | 実装 | 理由 |
|---|---|---|---|
| `when=gen` の最初のトークン(§6.15) | 「生成ステップだけ cvec あり」 | 最初のトークンは素の prefill から決まり、2 トークン目から押される(F75) | 設計の定義どおりだが、使う人が驚かないよう明記。最初から寄せるなら `assistant` |
| `when=ramp` の prefill | 記述なし | prefill は倍率 a で足す | 「冒頭だけ強く」が prefill から効くように |
| `when=assistant` の区間(§6.15) | 生成プロンプト以降 | 過去の返答も含め、発話ごとの区間で付け替える。テンプレートによっては本文を探して区間にする(F74) | 「多ターンで assistant の発話だけ常に寄せる」を満たすため |
| `when=after:<text>` | 出力に現れた以降 | 出力を毎ステップ文字列に戻して探す | トークンの切れ目に依らないように |
| `Engine.generate_greedy` の `schedule` | `schedule(step, seq_state) -> vectors \| None` | 戻り値 `KEEP` で「そのまま」、`None` で解除、dict で差し替え。シーケンス 1 本のときだけ | 付け替えない回を表すため。control vector はコンテキスト単位 |
| `gne serve --openai` | 最小実装 | greedy だけ(temperature などは無視)、`stop`・`n`・tools・logprobs は扱わない。`gne` に `recipe`(名前か `{units, dose_unit, when}`)・`units`・`scale`・`when` | 設計の「最小実装」の範囲 |
| LoRA の `adapter.lora.alpha`(§6.16) | 書く | 書かない(scale = adapter_scale) | 層ごとに rank が違うと alpha/rank が 1 にならないため |
| V11 の k=0 の合格基準 | 絶対値 < 1e-3 | 素の活性の最大値に対する比 < 3%(F76) | LoRA 版は量子化差が残り、活性の大きさはモデルで違う |
| gain の `Recipe.scaled` と 3 列チャットの − の列 | 記述なし | scaled(s) は 1 と k の間、− の列は 2 − k(F77) | eval の倍率 0 を無介入に、− の列を「反対側」にするため |
| `gne gain --bake` | `--in model.gguf --out model-x.gguf` | `--in`(既定 --model)と `--out` / `-o`(既定 `<root>/export/<名前>-gain.gguf`)、`--requant q8_0 \| f16` | 他のコマンドと同じく --model を元にする |
| gain の UI | 記述なし | UI のレシピ欄からは gain を作れない(API・CLI・MCP・`gne: {units: ["L:N:x2"]}` では使える) | CHANGE-01 の範囲外 |

| eval の倍率 0(§6.14) | 0 がベースライン | 0 が無ければ足す。倍率 0 も他の倍率と同じ `when` の経路(all ならまとめて、それ以外は 1 本ずつ)で生成する | バッチの組み方で greedy の数値がわずかに変わるので、比べる行どうしで経路をそろえる |
| eval の `ppl` | teacher-forced の平均 logprob | テキストごとに `exp(−平均 logprob)`、テキストをまたいだ値は幾何平均。レシピは `when` に関わらず全位置に当て、どの倍率でも同じ 512 トークンずつ decode する | 1 つの言語のテキストに引っ張られないように。decode の区切りで数値が最大 0.18 nats 変わるため(§11.3) |
| eval の `persona` の mass | rank.py の mass | ベースラインに無い語(rank.py の `rollout_words`)のうち 2 つ以上のプロンプトに出た語の、上位 5 語の出たプロンプト数の和。主な指標は novelty(出力の語のうちベースラインに無い語の割合) | rank.py の mass は対照群の平均に対する z。eval は 1 レシピで対照群が無い |
| eval の `exact` の採点 | 正規化後の一致 | 答え(別解のリスト)が出力の最初の空でない行に含まれるか。英数字の答えは語の境界で探す | 「42.」「**Paris**」を正解にし、「420」を不正解にするため |
| eval の UI | 本体設計「変化量(x) vs 能力低下(y)」、CHANGE-01「dose を x、変化量と能力低下を y」 | CHANGE-01 のとおり x = 倍率。y に変化量(novelty)・能力低下(pt)・ppl の悪化(%)・judge を重ね、配布可の倍率に印 | 新しい方の指示に合わせた |
| eval の保存名 | `evals/<model_id>/<recipe>.json` | レシピ名(無ければユニットの hash)。`when` が all 以外なら `@gen` などを付ける | 同じレシピの all と gen を並べて比べられるように |
| eval の judge | 採点モデルのパスは設定 | `--judge-model` と `--judge-target`(目標の性格の文)を CLI で渡す。採点は persona のプロンプトの出力だけ | 目標の性格はレシピごとに違う |
| 調書の gain と eval(CHANGE-01 D) | gain の試行結果(あれば)と eval の最新結果へのリンク | `evals`: そのニューロンを含む保存済みの eval(新しい順に 5 件、配布可の最大倍率と最大の倍率での能力の低下・新語の割合)。`gain_k`: 当てたことのある gain の倍率(実行時 LoRA の置き場所 `<root>/gain/<model_id>/` の `gne.recipe` から)。UI の Details タブでは eval の名前から eval ビューを開く | steer / chat の gain の生成文は保存していないので、試行の「結果」は eval にだけある。倍率は eval の倍率列で 1 と k の間の値も入る |
| エージェントへの指示(CHANGE-01 D) | `docs/agent_playbook.md` に追記 | playbook に加え、MCP サーバーの instructions(`gne mcp` が最初に渡す要約)にも「配る前に eval」「抑えるのは gain の x0 から」「when は gen から」を書いた | エージェントが playbook を読まなくても最初に目に入るように |
| `gne direction` の保存形式(§6.17) | `directions/<model_id>/<name>.npz`(層ごとの r、cos 行列、サンプル数) | そのとおり。中身のキーは逆引きの `*.directions.npz` と同じ(`layers`・`layer_L` = mean_A − mean_B(正規化前)・`scale_L`)に `cos`・`n_rows_A/B` を足した。層選びの結果は隣の `<name>.select.json` | 逆引き(`gne reverse --directions <名前>`)・調書・マップの cos の重ね合わせでそのまま使えるように。名前が逆引きと重なれば一覧では `<名前>@dir` |
| `gne direction` の層の指定 | `--select auto \| --layer L` | `--select auto`、`--select <L>`、`--layer L`(同じ意味) | |
| 層選びの mode(CHANGE-02 A) | 消去(rank-1 LoRA)で A の効果が下がり、B の KL が閾値内の層 | `--mode ablate`(設計どおり、既定)に加え `--mode add`: 層 L に平均差の dose 倍を足し、system なしの出力で判定の一致率が上がるか。副作用の目安は distinct-4(KL は参考) | ユーザーと合意した追加(ヤンデレのような性格を「足す」用途)。足せば分布が変わるのは当然なので KL は基準にしない |
| 層選びの dose(add) | 記述なし | 平均差 mean_A − mean_B の何倍か。結果には同じ大きさをレシピの rel dose に直した値も付ける | rel dose を全層で一律にすると、層ごとの平均差の大きさ(4B の evil で ρ の 0.02〜0.71 倍)の違いで比べ方が不公平になるため |
| 層選びの判定 | 判定語 / 正規表現の一致率、または judge | 対照集合の `validate: {match, prompts, target}`。`match` が無ければ `--judge-model` が要る | |
| 層選びの KL(ablate) | B での KL(base‖ablated) | B 側の会話(system_b + user_prompts)の生成プロンプト末尾の次トークン分布の平均 KL | prefill だけで測れる |
| 方向のユニット(CHANGE-02 A) | 記述なし(方向は消すためのもの) | `L:dir:NAME:dose` で層 L にその層の r_L を足す。rel: dose · ρ_L · r̂_L、raw: dose · (mean_A − mean_B)_L、absmax は不可。control vector に書き出せる | ユーザーと合意した追加。レシピの他のユニットと一緒に steer / chat / eval / export-cvec / OpenAI 互換 API で使える |
| 残差ライター(§6.17) | arch ごとのテンソル名の一覧 | `configs/arch/residual_writers.yaml` は候補(attn_output / ssm_out / ffn_down)・グラフで確かめた arch・post-norm の arch。層ごとに GGUF にある候補をライターとする。MTP(`blk.L.nextn.*` を持つ層)は除く | MTP の層は本流で実行されず、MTP を使わない読み込みではテンソルが作られないので、LoRA に含めると読み込みが拒否される(Saluki 27B の L64 で確認) |
| V13(LoRA 版) | 消去後に `\|l_out · r\| / ‖l_out‖` が量子化ノイズ水準 | LoRA 版は「l_out-L の r 成分 = 埋め込みの r 成分」(\|r·l_out − r·e\| / ‖l_out‖ < 1e-2)で見る | LoRA は埋め込み(get_rows)を変えられないので r 成分は 0 にならず、埋め込みの値のまま残る。bake 版(埋め込みも射影)で 0 になることは CHANGE-02 B の V13 で見る |
| 埋め込みと LoRA(§6.17) | 埋め込みは LoRA では表せない(bake のときだけ) | この llama.cpp は `token_embd` の LoRA を `build_inp_embd` で足すので、出力層が別のモデルでは LoRA にも埋め込みを入れる(既定、`--skip-embed` で省く)。tied のモデルでは入れない | F82・F83。tied のモデルでは出力層も同じ名前で LoRA が掛かり、形が合わない |
| `gne ablate` の r(§6.17) | 1 本の r を全層に、`--per-layer` で層ごと | 1 本の r は `--from-layer` の層で測ったもの(既定は `gne direction --select` で選んだ層)。`--per-layer` で方向の無い層(最終層)は前の層の方向を使い、埋め込みは射影しない | 「1 本の r」をどの層で測るかを決める必要があるため |
| `--scale`(§6.17) | 部分消去 | LoRA は `lora_b = −s·r` に入れる(llama.cpp の `--lora-scaled` でも弱められる)。bake は `W − s·r(rᵀW)` | LoRA を素の `--lora` で使っても指定どおりになるように |
| `--via-f16`(§6.17) | F16 で書いて llama-quantize で元の型へ | `--tensor-type-file` で全テンソルに今の型を指定(同じ型は写すだけ)、ftype は Q8_0、`general.file_type` は元の値に。IQ1〜IQ2 系のテンソルがあるモデルは `--imatrix` が要る | F84 |
| MoE(§6.17) | `ffn_down_exps` は軸 1 に射影、shared expert も | bake だけ対応。LoRA は未対応(ARCH_UNSUPPORTED) | 手元に MoE の GGUF が無く、`build_lora_mm_id` に 3 次元の LoRA を渡す形を実機で確かめられないため |
| V14(§6.17) | 先頭 32 トークンで一致 | 完全一致か、最初の分岐が僅差(0.5 nats 未満)なら合格(V9 と同じ) | F85 |
| `gne verify` の V13 / V14 | 記述なし | 2 GiB 未満のモデルは bake を 2 つ作って V13(bake)と V14 も見る。大きいモデルは V13(LoRA)だけで、`gne ablate --bake --verify` を案内 | 27B の bake は 10 GB を超える |
| レシピの ablate(§6.17) | `ablate: {direction, layers, scale}` | `from_layer`(どの層で測った r か)も持てる。ablate だけ(ユニットなし)のレシピも可。scale は −1.5〜1.5(負はその方向を強める) | 「1 本の r」を決めるため。負の scale は 3 列チャットの − 列(gain の 2 − k、add の符号反転と同じ考え) |
| `gne steer --ablate` の書き方 | `--ablate <name>` | `--ablate NAME[@層][:倍率]`(chat・eval も同じ) | 測った層と部分消去を 1 つの引数で |
| ablate の実行時の LoRA | gain と ablate を 1 ファイルに合成してよい | 別々の LoRA を並べて当てる(llama.cpp は複数のアダプタを倍率つきで同時に当てられる)。消去の LoRA は倍率 1 で 1 回だけ作り、scale はアダプタの倍率 | eval の倍率ごとに LoRA を作り直さないため |
| `eval --compare-ablation`(§6.17) | 同じ対照集合から、上位 N ニューロンの消去 vs 方向消去 | 上位 N は、層ごとの r_L と W_down の列の cos(逆引きと同じ標準化)で + 向きのもの。表には eval の指標に加え、方向を作った対照集合の判定(A 側の一致率)と B 側の KL を並べる | 「消したい振る舞いがどれだけ残るか」は eval のタスク集では測れないため |
| UI の ablate(§6.17) | recipe タブに ablate 行(方向・層範囲・scale スライダー) | そのとおりに加え「測った層」。保存したレシピのうち UI で扱えないユニット(gain・方向)は読み込みで外し、数を知らせる | UI のレシピ編集はニューロンの add だけを扱う |
| エージェントへの指示(CHANGE-02 E) | `docs/agent_playbook.md` に「消したい振る舞いは direction → ablate と上位ニューロンの gain k=0 を eval --compare-ablation で比べてから選ぶ」 | そのとおりに加え、方向で性格を「足す」手順(mode=add、反応する場面のプロンプトで確かめる)と、判定語の選び方(F86)を §4.1 に書いた。MCP の instructions にも要約 | 実際に evil とヤンデレで方向を使って分かったこと |
