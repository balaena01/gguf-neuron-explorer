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

