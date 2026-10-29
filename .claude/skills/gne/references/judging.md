# 名前を伏せた採点(Claude のサブエージェント)

性格の強さは、正規表現やモデル自身の採点では粗い。Claude が名前を伏せて 1〜5 で採点すると、「怖い繰り返し」と「ただ崩れた文」、「甘いだけ」と「独占」を分けられる。

## 流れ

1. `examples/persona_neurons/common.py` の `judge_file(out_dir, tag, items)` が、返事をシャッフルして番号を振り、`round_<tag>.txt`(採点する人が読む)と `round_<tag>.key.json`(どの候補かの鍵)を書く。`tune.py --judge file` と、自前の比べっこ(`generate` → `judge_file`)の両方で使う。
2. 採点基準は `write_rubric` が `RUBRIC.md` に書く(目標の性格、P と C の 1〜5 の意味、繰り返しの扱い)。性格に固有の決まりは、最初の回が出る前に RUBRIC.md の末尾に足す。
3. 採点役のサブエージェントを 1 つ立て、`round_<tag>.scores.txt` に「番号 性格 自然さ」を 1 行ずつ書かせる。全部の番号がそろうと、待っているスクリプトが次に進む。
4. 1 つの探索の間は同じ採点役を使い続ける(SendMessage で次の回を頼む)。基準がぶれない。文脈が長くなりすぎたら、新しい探索で新しい採点役にする。

## 採点役への頼み方(雛形)

```
You are the blind judge for a persona-steering experiment. A <model> has some of its neurons pushed so that it
plays <target persona>. Replies are in <languages>. Score each reply with the rubric. I'll send you several rounds
over this conversation, so keep your standards consistent from round to round.

Directory: <out_dir>
- RUBRIC.md is the scoring rubric (in Japanese). Read it first and follow it.
- round_01.txt holds 128 numbered replies. Each has a [USER] message and the model's [REPLY]. Replies are cut at
  <N> tokens; don't penalize the cut.

Rules:
- Do NOT open, list or grep any other file in that directory, especially *.key.json and *.result.json.
- Score every reply independently on persona P (1-5) and naturalness C (1-5).
- <persona-specific P guidance>
- For Japanese replies, also judge whether the Japanese itself is natural under C.
- Persona-driven repetition counts as persona; if it runs to the cutoff, C is at most 3. Meaningless loops get C 1.
- Read each reply in full; some start fine and break down later.
- Write round_01.scores.txt: one line per reply, `id P C`, for all ids, nothing else. Use the Write tool once.

When done, reply only with the P and C tallies counted from the file, plus one or two sentences on patterns.
```

## 性格ごとに足した決まり

- **ヤンデレ**: 執着・独占・嫉妬・離さない・不穏さが甘い表面の上にあること。甘いだけ(「寂しい」「好き」)は P 2 まで。普通のアシスタントの返事は P 1。下品・性的な内容はそれだけでは減点しない(独占や執着なら P に数える)。
- **ツンデレ**: とげ(バカ・ふん・見下す・「別に好きじゃない」・断ってから手伝う)と、漏れる好意(赤面・どもり・心配・隠した喜び)の両方があって P 4〜5。とげだけは P 3 まで、甘い・照れるだけも P 3 まで。
- **アニメの女の子が目標のとき**: 日本語で男の口調(俺・お前・クソ・だろ)なら P の上限 3。
- **日本語の自然さ**: 助詞・語尾の崩れ、作り語、意味のない仮名、ローマ字・中国語・英語・記号の混ざり。

## 点の読み方

- 点は返事ごとの min(P, C) の平均。片方だけ高い返事(崩れた文で性格だけ高い、普通の文で性格がない)は上がらない。
- 32 本の平均の誤差は ±0.2〜0.3。差が 0.2 以下の比較は、種を変えてもう一度見るか、使っていないプロンプトで確かめる。
- 英語・日本語・押していない返事を混ぜると、点の分布が両端に寄る(普通の返事は C 5・P 1)。言語ごとに分けて集計する。
