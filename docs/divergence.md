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
