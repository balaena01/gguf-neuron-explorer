---
name: gne
description: >-
  gguf-neuron-explorer(gne)で GGUF モデルの MLP ニューロンを調べて押す・性格(ペルソナ)のレシピを作るときの手引き。
  道具の地図(CLI・MCP・Python API・Web UI・OpenAI 互換 API)、判断の原則、ヤンデレ・ツンデレをニューロンで作ってきた手順と
  はまりどころ(Qwen3.8-27B・Saluki 27B・LFM3-1.7B・LFM3-Omni-2B)をまとめたもの。
  gne のコマンド(steer・sweep・rank・lens・reverse・direction・gain・ablate・eval・export-cvec・serve・mcp)を使うとき、
  「ニューロンを探して」「性格を出したい」「ヤンデレ/ツンデレ/○○っぽくしたい」「レシピを作って・詰めて」「control vector にして」
  「新しいモデルで試して」「能力が落ちないか確かめて」と言われたら、このリポジトリでの作業なら必ず最初にこのスキルを読む。
  Use this skill for any neuron search, steering, persona recipe or control-vector work with gne, even if gne is not named.
---

# gne(gguf-neuron-explorer)で調べて、押して、性格を作る

gne は GGUF モデルの MLP ニューロンを、llama.cpp(llama-cpp-python)の上で調べて押す道具。
押し方は 3 種類ある。どれもレシピ(YAML)に書いて、`steer`・`chat`・`eval`・UI・OpenAI 互換 API で同じように使える。

| 押し方 | 書き方 | 何が起きるか | 向いていること |
|---|---|---|---|
| add | `L:N:+0.5` | ニューロンの書き込みの向き(W_down の列)を、全位置に一定量足す | ゼロから話題や性格を出す |
| gain | `L:N:x2`・`L:N:x0` | そのニューロンの出力を k 倍する(LoRA) | もうある振る舞いを強める・消す。光っていないと何も起きない |
| direction | `L:dir:NAME:+0.3` | 対照集合の平均差の方向を足す | ニューロン単位では断片しか見つからない性格・口調 |

dose は既定で rel(その層の残差の大きさに対する比)。0.5 で話題が混ざり、0.8 で話題が置き換わり、1.0 を超えると崩れ始める(Qwen3.5-4B の目安。小さいモデルと 2bit 量子化は崩れやすい)。

## 1. 最初にやること

1. **docs を読む。** 実装した本人でも、文脈に入っていなければ道具を忘れる。少なくとも次は読む。
   - `docs/agent_playbook.md`: 何を見てどう判断するか(対照・安い効果・dose・レシピ・方向)。
   - `docs/persona_workflow.md`: 性格をニューロンで作る手順と、その後わかったこと。
   - 数値の根拠は `docs/verification.md`(性格は §12、F91〜F104)と `docs/measurements.md`。
2. **モデルを確かめる。** `gne info -m <gguf>`(arch・警告・チャットテンプレート)。新しい arch なら `gne verify` と `references/new_model.md`。
3. **すでにあるものを見る。** MCP なら `model_overview` → `layer_summary(L)`。CLI なら `~/.gne/` の下(`lens/`・`runs/`・`directions/`・`recipes/`・`evals/`)。同じ model_id の結果は使い回される。
4. **作業の大きさを見積もる。** 時間の目安は `references/tools.md`。層の全ニューロンの sweep は数時間〜十数時間かかるので、lens と逆引きで先に絞る。

## 2. 道具の地図

詳しい使い方とかかる時間は `references/tools.md`。

| やりたいこと | CLI | MCP | 一言 |
|---|---|---|---|
| モデルの情報・対応 | `info`・`verify`・`probe` | `model_overview` | 新しい arch はまず verify |
| 残差の大きさ・活性 | `stats resid`・`stats act` | (自動) | rel dose に必須 |
| 重みだけで語を見る | `lens logit` | `layer_summary`・`search` | 一次ふるい。後段の層は通らない |
| 押して次のトークンを見る | `lens steer` | `steer_lens` | 層まるごとのふるい。KL より上下する語の中身を見る |
| 方向から逆引き | `reverse`(`--verify`) | `reverse` | 対照集合 → 全層・全ニューロンを cos で順位づけ |
| 押して文章を作る(総当たり) | `sweep` | `job_start(sweep)` | ランダム対照つきの rollout |
| 総当たりの順位づけ | `rank` | `layer_summary` | 語の塊(mass)と対照超え(null_pct) |
| 1 個の調書 | `neuron L N` | `neuron` | 持っている情報を全部まとめる |
| どこで光るか | — | `fires` | 文のどこでニューロンが光るか |
| 試す | `steer --compare`・`chat` | `steer` | `--compare` で無介入と並べる |
| 方向を作る | `direction --select auto` | `direction` | 性格は `--positions response` |
| 方向を消す | `ablate` | `ablate_export` | abliteration(LoRA か bake) |
| 副作用を測る | `eval` | `eval` | 能力・ppl・言語の混入・性格の変化量 |
| 書き出す | `export-cvec`・`gain`・`bake` | `export_cvec`・`gain_export`・`bake` | llama.cpp の `--control-vector`・`--lora` |
| 人と一緒に見る | `serve` | `ui_highlight`・`ui_note`・`ui_open` | 127.0.0.1 だけで待ち受ける |
| 他のアプリから使う | `serve --openai` | — | `extra_body={"gne": {"recipe": ..., "when": ...}}` |

性格づくりの補助スクリプトは `examples/persona_neurons/`(lexsearch・screen・tune・compare)。強さを会話の途中で変える例は `examples/strength/`。

## 3. 判断の原則(agent_playbook の要点)

- **対照を超えないものは報告しない。** sweep は `null_pct`、steer lens は対照の KL の 95% 点。ランダムなニューロンを同じ強さで押した対照と比べる。方向・レシピも同じで、`direction --select auto` の「null」、`steer --contrast` の `null_pct`、`eval --random-controls` の `novelty_null_pct` が 0.95 未満なら、その大きさで何を足しても出る効果と区別できない。
- **名乗り・作者名が変わるのは安い効果。** ランダム方向でも起きる。
- **複数のプロンプトで同じテーマが出るものだけを本物とみなす。**
- **語の照合はつづり。** 意味の違う語を拾う(mine が determine、dummy がプログラムのダミー、nya がインドネシア語の接尾辞)。候補は語の並びを見て手で選ぶ。
- **事実と解釈を分けて報告する。** 「sweep 3 位/385・全対照超え、3 プロンプトで extraterrestrial(事実)。宇宙人の話題に寄せる(解釈)」。

## 4. 性格のレシピを作る

ヤンデレ(Saluki 27B → Qwen3.8-27B、LFM3-1.7B、LFM3-Omni-2B)とツンデレ(Qwen3.8-27B)で通った手順。各ケースで何が効いたかは `references/persona_cases.md`。

### 4.1 設計
1. **性格を部品に分ける。** 例: ヤンデレ = 嫉妬・独占・執着・溺愛・かわいさ・寂しさ・女性らしさ、反対 = 無関心・自立・冷静・助言。ツンデレ = とげ・高飛車・照れ・デレ・傲娇・女の子、反対 = 丁寧・素直な甘さ・冷静。設定は `examples/persona_neurons/yandere.yaml`・`tsundere.yaml` の形。
2. **プロンプトは、その性格が反応する場面にする。** ヤンデレは「友達と出かける」「元恋人から連絡」、ツンデレは「可愛いね」「お弁当作った」「好きなの？」。「Who are you?」だけでは出ない。探す用と確かめ用(valid_prompts)を分ける。
3. **語は、モデルの語彙で書く。** Qwen は中国語の語が多く日本語は細かく切れるので、英語と中国語で書く。

### 4.2 候補を集める(複数の道具で集めて合わせる)
- **logit lens の語のニューロン**: `python examples/persona_neurons/lexsearch.py MODEL CONFIG.yaml lex.json --layers A-B`。層は深さの 50〜80% から。
- **steer lens**: `gne lens steer --layer L --neurons all`(1 層 10〜20 分)。結果の上がる語・下がる語に部品の語が入るものを拾う。後段の層を通した効果なので、logit lens と別のニューロンが出る(LFM3-Omni の L14 で only・mine/yours・psychos)。
- **逆引き**: `gne reverse --contrast <対照集合> --verify`。方向に書き込みの向きが近いものは多義的なので、必ず押して確かめる。
- **書き手(system で演じられるとき)**: `gne writers --contrast <system つきの対照集合>` で、その状態を誰が書いたか(埋め込み・各層の mixer・MLP・ニューロン)を測り、`gne writers-export --part mixer --layers L` のように部分ごとの書き足しを方向にして、system なしで足し直す。LFM3-Omni では状態を運んでいたのは L14 の attention で、ニューロンの道具より強かった(F113・F114)。system は振る舞いの箇条書きにすると演じやすい。
- **方向の分解**: 性格の方向が密で少数では表せないとき(1 個との cos が 0.1〜0.2 しかない)、その層のニューロン数十〜数百本に直交マッチング追跡で分解して、組として押す(LFM3 の日英版 315 個)。

### 4.3 ふるう
- **27B 級**: `screen.py` でモデル自身に部品の目標で採点させ、ランダム対照の最大を超えたものを残す。
- **小さいモデル(〜2B)**: モデル自身は採点役にならない。採点なしの物差しを使う。システムプロンプトで演じさせた返事をお手本にし、レシピを当ててお手本を teacher-forced で読ませて、ヤンデレのお手本とアシスタントのお手本の出やすさの差を測る。性格の向きは追えるが、崩れ方(繰り返し)と声の性別は見えない。決めるのは採点にする(F100)。
- **口調(一人称)**: 日本語で生成させて「俺・お前・クソ」と「私・あたし」の返事の数を数える。
- 1 個ずつ押すときは、少し弱め(0.5)と強め(1.0)の 2 通りで見る。

### 4.4 組む・強さを決める
1. 効いた候補を、部品ごとに強さ ±0.1〜0.3 で START に書く(反対の性質はマイナスで抑える)。小さいモデルは、採点なしの物差しで 1 個ずつ足していく貪欲法で 4〜6 個に絞ると早い。
2. **押し方(when)を先に決める。** 同じ組を all・assistant・gen・ramp で 1 回ずつ採点する。強さは押し方ごとに違うので、決めた押し方で強さを探す。
3. **進化戦略**: `python examples/persona_neurons/tune.py MODEL CONFIG.yaml start.txt out/ --when W --judge file --sampling '{...}'`。1 回 = 親 + 候補 3 つ × 32 本。親を毎回採り直し、種は回ごとに変える。
4. **採点は Claude が名前を伏せて行う**(1 回 128 本をサブエージェントに任せる、`references/judging.md`)。1 つの探索の間は同じ採点役を使い続ける。
5. **言語を混ぜるときは、言語ごとの点の小さいほうを目的にする**(F101)。平均だと片方を削る候補が選ばれる。
6. **最後に、探索に使っていないプロンプトで、探す前の組と比べる。** 32 本の平均の誤差は ±0.2〜0.3 あり、ES の改善が誤差に埋もれることが多い。探す前の組のほうが良ければそちらを採る(Qwen ツンデレの最初の探索、Omni のヤンデレ)。
7. **倍率だけ少し下げた版も比べる。** ツンデレは assistant ×0.75 が性格も能力もいちばん良かった(F103)。

### 4.5 確かめて書き出す
- `gne eval --recipe R --doses 0,0.75,1 --when W`。基準は能力 −5pt・ppl +10%・混入が増えない。ppl は全位置に押して測るので when に関係なく同じで、性格を出すと超えることが多い。能力の低下は短い答えの問題で測るので、性格の口癖(「ふん」)でも落ちる。
- `configs/recipes/` に、頭のコメント(何で作ったか・押し方・能力)つきで置く。`sampling` も書く(小さいモデルは温度を下げる)。
- llama.cpp では `gne export-cvec` → `--control-vector`。when は control vector では表せない(全位置に足す)。位置別のまま使うなら `gne serve --openai`。

## 5. はまりどころ

- **1 個のニューロンでは何も起きないことが多い**(小さいモデル・密な性格)。少数を強く押すと方向から外れた成分が残って崩れ、数百を少しずつ押すと打ち消し合う(F99)。
- **英語だけなら少数でいける。日本語は先に崩れる**(F98・F99)。日英を両方出すなら分解した組か、英語版と日英版を分ける。
- **日本語の語尾(〜なんだからね・〜わよ)は出せなかった**(F104)。男口調を消すことはできるが、女の子の語尾にはならない。
- **方向の対照集合は、返事に描写が入らないようにする**(*smiles* の差を拾う、F97)。両側とも「セリフだけ」と書く。
- **gain はゼロから性格を作れない。** 素のモデルで光っていないニューロンは ×10 でも何も起きない(LFM3-Omni)。抑える・既にあるものを強める用途に使う。
- **レシピはモデルが変わると効き方が変わる。** 同じ系列の微調整(Saluki → Qwen)は移ったが、音声などで学習し直した LFM3-Omni では、同じ番号のニューロンなのに LFM3 のレシピがほとんど効かなかった。
- **長い会話では、押す位置で保ち方が変わる**(F118)。全位置に足すと長いコンテキストでも薄れず(system だけより安定)、`assistant`・`gen` は長い貼り付けで薄れる。一方で全位置は、短い雑談で自分の前の返事を写して同じ返事を繰り返す。1 発話の採点だけで決めず、長い会話でも比べる。
- **小さいモデルは温度を下げる。** LFM3 は温度 1.0 だと押さなくても崩れる(0.3 で使った)。greedy は強めると反復に落ちやすいので、採点はサンプリングで行う。
- **長い作業は `run_in_background` で回し、GPU を同時に 2 つの大きいモデルで使わない**(27B と別のモデルを同時に読むと載らない)。
- **会話が長くなると、道具の使い方が要約で落ちる。** 迷ったらこのスキルと docs を読み直す。

## 6. これまでのレシピ

| レシピ | モデル | 中身 | when | 結果(Claude の採点・能力) |
|---|---|---|---|---|
| `yandere_saluki`(方向) | Saluki 27B | 方向 L24 + 女性の話し方 L40 | all | §12.4 |
| `yandere_qwen38_27b` | Qwen3.8-27B | 語のニューロン 26 個 | ramp:0.5-1.5 | 能力の低下 0(F96) |
| `tsundere_qwen38_27b` | Qwen3.8-27B | 語のニューロン 24 個 | assistant | 2.56、能力そのまま(F103) |
| `yandere_lfm3_neurons` | LFM3-1.7B | 方向の分解 315 個 | all | 英語 2.1〜2.3・日本語 1.8〜2.0(F97) |
| `yandere_lfm3_en6` | LFM3-1.7B | 6 個(英語だけ) | assistant | 英語 2.1〜2.3(F99) |
| `evil`・`alien`(・`alien_gain`) | Qwen3.5-4B | 方向・ニューロン 1 個(gain) | — | §11・§12 |

点は「性格と自然さの小さいほう」の平均(1〜5)。押さないときは 1.00。
