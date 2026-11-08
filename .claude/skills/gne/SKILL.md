---
name: gne
description: >-
  gguf-neuron-explorer(gne)で GGUF モデルの MLP ニューロンを調べて押す(ステアリングする)・性格(ペルソナ)のレシピを作るときの手引き。
  道具の一覧(CLI・MCP・Python API・Web UI・OpenAI 互換 API)、結果の判断の仕方、ヤンデレ・ツンデレをニューロンで作ってきた手順と
  つまずいた点(Qwen3.8-27B・Saluki 27B・LFM3-1.7B・LFM3-Omni-2B)をまとめたもの。
  gne のコマンド(steer・sweep・rank・lens・reverse・direction・direction key・gain・ablate・heads・eval・export-cvec・serve・mcp)を使うとき、
  「ニューロンを探して」「性格を出したい」「ヤンデレ/ツンデレ/○○っぽくしたい」「レシピを作って・詰めて」「control vector にして」
  「新しいモデルで試して」「能力が落ちないか確かめて」と言われたら、このリポジトリでの作業なら必ず最初にこのスキルを読む。
  Use this skill for any neuron search, steering, persona recipe or control-vector work with gne, even if gne is not named.
---

# gne(gguf-neuron-explorer)で調べて、押して、性格を作る

gne は GGUF モデルの MLP ニューロンを、llama.cpp(llama-cpp-python)の上で調べて押す(ステアリングする)道具。
ニューロンの押し方は 3 種類ある。ほかに、attention ヘッドの書き込みのうち方向に沿った分だけを変えるやり方と、決まった状態のときだけ方向を書くやり方(keyed write)もある。どれもレシピ(YAML)に書いて、`steer`・`chat`・`eval`・UI・OpenAI 互換 API で同じように使える。

| 押し方 | 書き方 | 何が起きるか | 向いていること |
|---|---|---|---|
| add | `L:N:+0.5` | ニューロンの書き込みの向き(W_down の列)を、全位置に一定量足す | ゼロから話題や性格を出す |
| gain | `L:N:x2`・`L:N:x0` | そのニューロンの出力を k 倍する(LoRA) | もうある振る舞いを強める・消す。活性が出ていないと何も起きない |
| direction | `L:dir:NAME:+0.3` | 対照集合(同じ質問を 2 つの条件で読ませる YAML)の平均差の方向を足す | ニューロン単位では断片しか見つからない性格・口調 |
| ヘッドの方向成分 | `L:Hh:sS@NAME` | ヘッドが残差に書くベクトルのうち、方向に沿った分だけを (1 − s) 倍にする(LoRA)。他の分には触らない | すでに出ている振る舞いを消す・弱める。元があれば負の s で強める。ゼロからは作れない |
| 鍵つきの書き込み | `L:ffn:keyed:aα@NAME#KEY` | 残差に書き込む重み行列(ffn_down など)に入る活性に鍵が反応する位置でだけ、方向(平均差 r_L をそのままの大きさで)を α 倍で書く(LoRA)。鍵は対照集合から作る | 決まった場面(「です。」を書く直前、お礼を言う直前)だけで語尾・口調を変える。無い振る舞いも書ける。他の場面には何も足さない |

dose は既定で rel(その層の残差の大きさに対する比)。0.5 で話題が混ざり、0.8 で話題が置き換わり、1.0 を超えると崩れ始める(Qwen3.5-4B の目安。小さいモデルと 2bit 量子化は崩れやすい)。

## 1. 最初にやること

1. **docs を読む。** 実装した本人でも、文脈に入っていなければ道具を忘れる。少なくとも次は読む。
   - `docs/agent_playbook.md`: 何を見てどう判断するか(対照・どの向きでも起きる効果・dose・レシピ・方向)。
   - `docs/persona_workflow.md`: 性格をニューロンで作る手順と、その後わかったこと。§5 は、お手本の会話から attention ヘッドとニューロンを導く手順。
   - 数値の根拠は `docs/verification.md`(性格は §12、F91〜F104)と `docs/measurements.md`。
   - ヘッドの方向成分(CHiPS)を使うなら `docs/head-component-steering.md`、鍵つきの書き込み(keyed)を使うなら `docs/keyed-write.md`。
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
| 総当たりの順位づけ | `rank` | `layer_summary` | 増えた上位の語の z の和(mass)と、対照を超えた割合(null_pct) |
| 1 個のニューロンのまとめ(調書) | `neuron L N` | `neuron` | 持っている情報を全部まとめる |
| どこで発火するか | — | `fires` | 文のどこでニューロンの活性が大きいか |
| 試す | `steer --compare`・`chat` | `steer` | `--compare` で無介入と並べる |
| 方向を作る | `direction --select auto` | `direction` | 性格は `--positions response` |
| 方向を消す | `ablate` | `ablate_export` | abliteration(LoRA か bake) |
| 方向の分かれ方・基準層 | `direction score` | `direction_score` | 生成なし。層ごとの cohen_d と、介入したときの方向への射影の値の変化 |
| 効くヘッドを探す・強さを配る | `heads causal`・`heads allocate` | `heads_causal`・`heads_allocate` | 生成なし。止めたとき・強めたときの射影の値の変化から、少ないヘッドの組を作る |
| 組で消し切れない質問に足す | `heads refine` | `heads_refine` | 残りの質問だけで、組を当てたまま方向を作り直して足すヘッドを探す |
| 全体の倍率を返事で決める | `heads candidates`(`--tally`) | `heads_candidates`・`heads_candidates_tally` | 組 × 倍率で返事を作り、どの候補か伏せた採点のファイルを書いて集計する |
| 決まった状態のときだけ書く(鍵) | `direction key`・`direction key-check`・`direction key-verify`・`direction keys`・`direction key-experts` | `direction_key`・`direction_key_check`・`direction_key_verify`・`keys`・`direction_key_experts`・`direction_key_pipeline`(方向 → 鍵 → key-check の一括) | 鍵を作り、作成に使っていない例で、比較用の鍵(ラベルをシャッフルして作った鍵)より書き込みたい位置をよく選ぶかを確かめる。返事を与えれば生成なし |
| 副作用を測る | `eval` | `eval` | 能力・ppl・言語の混入・性格の変化量 |
| 実際の使い方で確かめる | `fieldtest` | `fieldtest` | llama-server で長い作業・ツール呼び出し・質問集・perplexity。ファイルどうしを比べる。配る前に |
| 書き出す | `export-cvec`・`gain`・`headproj export`・`bake` | `export_cvec`・`gain_export`・`headproj_export`・`keyed_export`・`bake` | llama.cpp の `--control-vector`・`--lora`。proj・keyed は LoRA か焼き込みだけ |
| 人と一緒に見る | `serve` | `ui_highlight`・`ui_note`・`ui_open` | 127.0.0.1 だけで待ち受ける |
| 他のアプリから使う | `serve --openai` | — | `extra_body={"gne": {"recipe": ..., "when": ...}}` |
| 振る舞いを数える | `behave --prompts … --match … --judge …` | — | 返事はファイルにだけ残し、数だけを出す。長さは十分に取る |
| 手元に載らないモデル | `modal fetch`・`modal run -- <コマンド>` | — | 課金は動いている間だけ。`docs/modal.md` |

性格づくりの補助スクリプトは `examples/persona_neurons/`(lexsearch・screen・tune・compare)。強さを会話の途中で変える例は `examples/strength/`。

## 3. 判断の原則(agent_playbook の要点)

- **対照を超えないものは報告しない。** sweep は `null_pct`、steer lens は対照の KL の 95% 点。ランダムなニューロンを同じ強さで押した対照と比べる。方向・レシピも同じで、`direction --select auto` の「null」、`steer --contrast` の `null_pct`、`eval --random-controls` の `novelty_null_pct` が 0.95 未満なら、その大きさで何を足しても出る効果と区別できない。
- **名乗り・作者名が変わるのは、どの向きに押しても起きる効果(「安い効果」)。** ランダム方向でも起きる。
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
- **書き手(system で演じられるとき)**: `gne writers --contrast <system つきの対照集合>` で、その状態を誰が書いたか(埋め込み・各層の mixer・MLP・ニューロン)を測る。`gne writers-export --part mixer --layers L` のように部分ごとの書き足しを方向にして、system なしで足し直す。LFM3-Omni ではその状態を主に書いていたのは L14 の attention で、ニューロンの道具より強かった(F113・F114)。system は振る舞いの箇条書きにすると演じやすい。
- **方向の分解**: 性格の方向が密で少数では表せないとき(1 個との cos が 0.1〜0.2 しかない)、その層のニューロン数十〜数百本に直交マッチング追跡で分解して、組として押す(LFM3 の日英版 315 個)。

### 4.3 ふるう
- **27B 級**: `screen.py` でモデル自身に部品の目標で採点させ、ランダム対照の最大を超えたものを残す。
- **小さいモデル(〜2B)**: モデル自身は採点役にならない。採点なしの指標を使う。システムプロンプトで演じさせた返事をお手本にし、レシピを当ててお手本を teacher-forced で読ませて、ヤンデレのお手本とアシスタントのお手本の出やすさの差を測る。性格の向きは追えるが、崩れ方(繰り返し)と声の性別は見えない。決めるのは採点にする(F100)。
- **口調(一人称)**: 日本語で生成させて「俺・お前・クソ」と「私・あたし」の返事の数を数える。
- 1 個ずつ押すときは、少し弱め(0.5)と強め(1.0)の 2 通りで見る。

### 4.4 組む・強さを決める
1. 効いた候補を、部品ごとに強さ ±0.1〜0.3 で START に書く(反対の性質はマイナスで抑える)。小さいモデルは、採点なしの指標で 1 個ずつ足していく貪欲法で 4〜6 個に絞ると早い。
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
- **gain はゼロから性格を作れない。** 素のモデルで活性が出ていないニューロンは ×10 でも何も起きない(LFM3-Omni)。抑える・既にあるものを強める用途に使う。
- **レシピはモデルが変わると効き方が変わる。** 同じ系列の微調整(Saluki → Qwen)は移ったが、音声などで学習し直した LFM3-Omni では、同じ番号のニューロンなのに LFM3 のレシピがほとんど効かなかった。
- **MoE の層の add はニューロン押しではない**(CHANGE-05)。`L:Ee:Nn:+d` は書き込み方向を全トークンに足すだけ。選ばれたときだけ効くのは gain(`x2`)。まず `gne experts contrast` で専門家を絞り、その中のニューロンを gain で試す。専門家 1 つを消しても system で作った性格はあまり減らない。
- **ヘッドは重みだけから決めた向きでは押せない**(F124・F125)。ニューロンの書き込みは重みで決まる 1 本の向きだが、ヘッドの書き込みは注目したトークンの値ベクトルで決まる。指示のない会話でヘッドの gain を上げても性格は出ない。ヘッドを調べるときは `gne heads listen --heads L:Ha-b --system … --direction …`(UI のヘッドのパネルの Listen)で、返事のトークンがどのトークンに注目し、どのトークンが方向に寄与したかを見る。system を読んだときの attention の出力の差分ベクトルに語のニューロンを重ねると、性格の点は量に比例して上がるが、自然さも同じだけ下がる(Omni では ×0.2 で同点)。
- **CHiPS(ヘッドの方向成分だけを変える手法)は、ヘッドがいま書いている分しか動かせない**(verification.md §20)。system で頼んだ evil を 2 個のヘッドで止めるのはうまくいき、能力も ppl も落ちなかった。一方で、同じヘッドを負の s で強めても、指示のない会話では何も起きなかった。ニューロンや方向の小さな押しで先に性格が出ていれば増幅でき、弱い増幅なら同じくらいの性格を ppl の悪化を抑えて出せた。ただし能力の落ち方はニューロンだけのときと同じで、強く増幅すると崩れた。強さは生成なしの指標(`direction score`・`heads causal`・`heads allocate`)で決め、報告は出力から測った値(判定の一致率・能力・ppl)とランダムなヘッドの対照でする。
- **keyed write(鍵つきの書き込み)は、鍵が状態を分けていなければ add と変わらない**(`docs/keyed-write.md`)。
  - 鍵は `gne direction key` で作る。`gne direction key-check` で作成に使っていない例(既定 subset odd)を測り、選択性が比較用の鍵(ラベルをシャッフルして同じ手順で作った鍵)の p95 を超えた層(`beats_pseudo`)だけを使う。選択性は、書き込みたい位置での鍵の反応の平均と、書き込みたくない位置での反応の平均の差。例単位の AUC は比較用の鍵でも 1 になることがある。
  - 語尾・口調は、対照集合を `positions: "before:<正規表現>"` にして作る(モデル自身の返事で、変えたい文字を書く直前を書き込みたい位置にする)。正規表現はモデルが実際に書く言い方に合わせる(結果の `samples`・`matched` を見る)。
  - α は方向をそのままの大きさで使う倍率(1 で、観測した A − B の差をそのまま書く)。`when` は効かない。同じ層の gain や別のレシピと重ねるなら、`--iv` でその状態で鍵を作り直す。
  - 鍵は ffn と attn の両方で作って比べる。0.6B では attn がはっきり上だった(F133)。Qwen3.5-4B では両方超えたが、attn の鍵が英語・ふつうの質問の返事の冒頭でも反応して、英語の ppl を +30% にした(F138)。
  - 返事の最初で一致する正規表現(「ありがとう！」)は、どの質問の返事の冒頭でも鍵が反応しやすい(F134)。鍵の作成は、dont_touch(書き込まない場面)の質問の、返事を書き始める直前の位置を自動で Q(鍵が反応しないようにする部分空間)に入れる(既定、F142)。ふつうの質問を dont_touch に数問入れておく。
  - 鍵の作成は、書き込みたい位置の 25% 以上が返事の冒頭にあるとき(dont_touch の群が無ければ警告、あれば注)と、書き込みたい位置が 2 例・8 位置未満のときに知らせる。key-check は、書き込みたくない位置・dont_touch の返事の冒頭での反応の強さ(`start_rho`)が 0.4 以上で警告する。警告を読んでから先に進む。
  - 焼き込みの既定は、元が Q8_0 のテンソルは Q8_0 のまま(F16 は掛け算の計算方法が変わり、LoRA 版とかえってずれた、F132)、それ以外は F16。
  - MoE の専門家の鍵は、書き込みたい位置でよく選ばれ、書き込みたくない位置ではあまり選ばれない専門家で作る(F135)。
  - **keyed は方向に無い言い方は書けない。** Qwen3-0.6B では「です。」の手前にずんだもんの方向を書いても、文末が言い切りになるだけで「のだ」は出ず、ツンデレの言い方も出なかった(verification.md §21.9)。方向に言い方が入っている Qwen3.5-4B ではどちらも出た(F136)。語尾は attn の 3 層に α = 2 で 12 問中 4 問が崩れずに「のだ」で終わり、英語の返事は崩れなかった。ツンデレは ffn の α = 1 で照れ隠しの返事が出て、計算・事実の返事の口調は変わらなかった。add は「のだ」が出る強さで日本語の返事の大半が崩れ、ツンデレの add は計算の返事の口調まで変えた。
  - **ただし keyed も強めると繰り返しに落ちる**(F137)。書き込んだ後も状態が書き込みたい位置と似たままだと、次の位置でもまた書く(ツンデレの α = 1 で 12 問中 9 問が「え、え、え、…」)。目的の言い方の数と崩れた返事の数を一緒に数えて α を決める。
- **多ターンの会話の写しは DRY で止める**(F122)。レシピの `sampling` に `dry_multiplier: 0.8` を足すと、前の返事をそのまま繰り返すのが止まる(音声の会話で 12 本中 12 本、写し 0)。llama-server でも `--dry-multiplier` で同じ。
- **長い会話では、押す位置で保ち方が変わる**(F118・F120)。ユーザーの発話にも足す押し方(全位置・`skip-replies`)は長いコンテキストでも薄れず(system だけより安定)、`assistant`・`gen` は長い貼り付けで薄れる。ベクトルだけなら `skip-replies` が長短とも少し良い。どの押し方でも、短い雑談では自分の前の返事を写して同じ返事を繰り返す(過去の返事を押さなくても直らない)。1 発話の採点だけで決めず、長い会話でも比べる。
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
| `yandere_omni_heads` | LFM3-Omni-2B | attention ヘッド 13 個(2 つの向き)+ ニューロン 2 個、音声で話しかける | gen | 人が会話を読んで選んだ(persona_workflow §5) |
| `evil`・`alien`(・`alien_gain`) | Qwen3.5-4B | 方向・ニューロン 1 個(gain) | — | §11・§12 |

点は「性格と自然さの小さいほう」の平均(1〜5)。押さないときは 1.00。
