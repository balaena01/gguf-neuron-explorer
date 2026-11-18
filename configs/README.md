# ひな形(Web UI の「ひな形」で選ぶ JSON)/ Templates

Web UI の FiSH・CHiPS・キーの画面の「ひな形」で選ぶと、入力欄にこの JSON が入ります。JSON の `about` は説明(要約と節。
日本語・英語)で、対照集合には入りません。`fish` / `pipeline` / `keyed` はその画面の設定です。数は流れを確かめる量なので、
使うときは各ひな形の「勧める数」を見て増やしてください。

The FiSH, CHiPS and key screens of the Web UI put one of these JSON files into the input box when you pick it under "Templates".
`about` is the description (summary and sections, in Japanese and English) and does not go into the contrast set; `fish`,
`pipeline` and `keyed` are the settings of that screen. The counts are enough to check the flow; see each template's
"Recommended counts" before real use.

## FiSH(configs/fish/)

| ファイル / file | 日本語 | English |
|---|---|---|
| `tone_in_scene.json` | 特定の場面の応答の口調を変える(例: お礼を言われたときだけツンデレ) | Change the tone of responses in one scene (example: tsundere only when thanked) |
| `always_persona.json` | いつでも口調を変える(例: 元気でくだけた口調) | Change the tone everywhere (example: cheerful and casual) |
| `sentence_end.json` | 文末の言い方を変える(です・ます → のだ) | Change sentence endings (desu/masu → noda) |
| `scene_and_sentence_end.json` | 場面 かつ 文末(雑談のときだけ「のだ」) | A scene AND sentence ends (noda only in small talk) |
| `two_personas.json` | 2 つの性格の違いだけを書く(system を両側に) | Write only the difference between two personas (system on both sides) |
| `after_chips.json` | CHiPS のレシピを当てた上に足す(例: お礼を言われたときだけツンデレ) | On top of a CHiPS recipe (example: tsundere only when thanked) |
| `route_none.json` | 入力の振り分け: 言いかけ・独り言には応答しない(終了トークン) | Input routing: no response to unfinished remarks (end-of-turn token) |
| `route_ack.json` | 入力の振り分け: お礼・挨拶には相づちで応える(マーカー) | Input routing: backchannel to thanks and greetings (marker) |
| `heads_by_scene.json` | 注意書きのヘッドの分を、深刻な相談でだけ取り除く(CHiPS のレシピを場面で) | Remove the disclaimer heads only for serious consultations (a CHiPS recipe by scene) |
| `closing_rmbase.json` | 計算の質問でだけ、締めの一言の成分を取り除く(!rmbase) | Remove the closing-remark component only for calculation questions (!rmbase) |

## CHiPS(configs/pipeline/)

| ファイル / file | 日本語 | English |
|---|---|---|
| `system_persona.json` | system で頼んだ性格(system は A 側だけ) | A persona asked for in a system prompt (system prompt on side A only) |
| `two_personas.json` | 2 つの性格の違いだけを取る(system を両側に) | Only the difference between two personas (system prompts on both sides) |
| `recipe_persona.json` | ニューロンのレシピが作った性格(A 側にレシピ) | A persona made by a neuron recipe (recipe on side A) |
| `question_kinds.json` | 質問の種類の違い(言語・話題・形式。側ごとに別の質問) | Two kinds of questions (language, topic, format; separate questions per side) |
| `reply_in_progress.json` | 応答の途中で起きる振る舞い(考えがループするなど。書きかけの応答の対) | Behavior that appears partway through a response (thinking in circles, for example; pairs of unfinished responses) |
| `disclaimer.json` | 注意書きのヘッドを探す(例: 深刻な相談に付く「専門家に相談を」) | Find the heads behind disclaimers (example: "consult a professional" on serious consultations) |

## キー / keys(configs/keyed/)

| ファイル / file | 日本語 | English |
|---|---|---|
| `sentence_endings.json` | 文末の言い方を変える(作り方 (b)、です・ます → のだ) | Change how sentences end (method (b): です/ます → のだ) |
| `one_scene.json` | 特定の場面の応答だけ変える(作り方 (b)、褒められた・お礼を言われた場面でツンデレに) | Change only the responses in one kind of scene (method (b): tsundere when praised or thanked) |
| `system_state.json` | system で作った状態を書き込みたい位置にする(作り方 (a)、丁寧語 → くだけた口調、α は負) | Use a state made with a system prompt as the positions to write at (method (a): polite → casual, negative α) |

## 1 つのオプションで使える機能(ひな形は作っていないもの)

ひな形の JSON には書かず、コマンドやユニットに 1 つ足して使う機能の例です。詳しくは [docs/fish.md](../docs/fish.md)。

```bash
# 量子化の丸めを避ける倍率: ユニットの後ろに !kauto(gne nw verify --contrast で不合格の層があれば付ける)
gne nw verify --recipe R --contrast C
#   11:N2045+N2046:nw:a1@NAME.seg1#K_p1&K_one!kauto

# LoRA を当てる位置を限る(all / prompt / reply)
gne steer --recipe R --prompt "..." --lora-when reply

# teacher forcing でヘッドを層ごと・ヘッドごとにふるい分ける(対象の文は正規表現。--span sentence = 一致を含む文、match = 一致した文字だけ)
gne heads tf --contrast C --direction NAME --target '(注意|ご相談ください)' --span sentence --layers 1-27

# 前のやり取りの応答(content: null)を埋めた対照集合を書く(キーの作成は自動で埋める)
gne nw prior --contrast C

# 最小二乗のキーのセルに会話の形も入れる / 文の終わりの次の位置で開くキー(決め手のセルを重くし、誤検出の上限に合わせて縮める)
gne direction key --contrast C --target mlpin --layers 22 --positions pos:0 --key-fit ridge --key-conv-cells
gne direction key --contrast C --target mlpin --layers 16 --key-fit ridge --positions 'after:[。！？\n]' \
  --target-reply '^◇' --key-crit 10 --key-fp-max 0.02 --iv R

# トークンの行の方向と層の走査
gne direction token-scan --token '<|im_end|>' --contrast C
gne direction token --token '<|im_end|>' --layers 22 --delta 32 --contrast C --name eos_up

# 応答の最初のトークンを決めて続きを読む / 長い応答の全部の位置でキーの出力を読む
gne steer --recipe R --prompt "..." --first-token '◇'
gne nw check --recipe R --contrast C --max-tokens 2048 --pos-bins 0,1-3,4-15,16-63,64- --match 'after:[。！？\n]'
```

## Features that take one option (no template)

These are used by adding one option to a command or one mark to a unit, not through a template JSON. See [docs/fish.md](../docs/fish.md).

- `!kauto` on an nw unit scales the neuron pair to avoid quantization rounding; add it when `gne nw verify --contrast C` fails on a layer.
- `--lora-when all|prompt|reply` (gne steer, chat, eval, heads tf) limits where the LoRA applies.
- `gne heads tf --target <regex> --span sentence|match` screens heads by teacher forcing on the matched sentences or characters.
- `gne nw prior` fills prior assistant turns written as `content: null`; key building does this itself.
- `--key-conv-cells`, `--positions after:<regex>`, `--target-reply`, `--key-crit`, `--key-fp-max` are options of least-squares keys.
- `gne direction token` / `token-scan` build a direction from output-matrix token rows and scan the layers.
- `gne steer --first-token` starts the reply with fixed tokens; `gne nw check --pos-bins` / `--match after:<regex>` read the key output over long replies.

[templates/README.md](templates/README.md) は、チャットテンプレートの上書きの説明です(ひな形とは別のもの)。
[templates/README.md](templates/README.md) describes chat template overrides (not these templates).
