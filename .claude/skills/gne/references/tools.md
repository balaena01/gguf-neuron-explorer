# gne の道具(使い方とかかる時間)

すべてのコマンドは `-m <model.gguf>`(または `GNE_MODEL`)をとる。結果は `~/.gne`(`--root`・`GNE_ROOT`)に保存され、後のコマンドと UI が使い回す。`gne <command> --help` は日本語。

## 目次
1. 調べる(info・stats・lens・reverse・sweep・rank・neuron)
2. 押す(steer・chat・レシピ)
3. 方向(direction・ablate・heads・鍵つきの書き込み)
4. 確かめる(eval)
5. 書き出す(export-cvec・gain・bake)
6. UI・OpenAI 互換 API・MCP
7. Python API
8. 時間の目安

## 1. 調べる

```bash
gne info                                   # arch・警告・対応・チャットテンプレート
gne verify                                 # llama.cpp との等価性(新しい arch では最初に)
gne stats resid                            # 層ごとの残差の大きさ(rel dose に必須、自動でも作る)
gne lens logit --layer 22                  # 重みだけ: 各ニューロンが上げる語・下げる語
gne lens steer --layer 14 --neurons all --dose 0.8   # 押して次のトークンの変化(ランダム対照 64 本つき)
gne reverse --contrast configs/contrast/evil.yaml --verify   # 方向 → 全層・全ニューロンを cos で順位づけ
gne sweep --layer 22 --neurons 1,5,9-12 --design main_rel --prompt-set en   # rollout(ランダム対照つき)
gne rank --run L22-main_rel                # 増えた上位 5 語の z の和(mass)と対照超え(null_pct)で順位
gne neuron 22 2742                         # 調書(そのニューロンの結果のまとめ)
```

- `lens steer` の結果は `~/.gne/lens/<model_id>/steer_L<L>.npz`(`up_ids`・`down_ids`・`kl_pos`・`kl_neg`、対照は `ctrl_*`)。既定の文脈は 4 つ(英語の文の冒頭・自己紹介・日本語の文の冒頭 2 つ)。KL が大きいのは機能語のニューロンが多いので、上がる語・下がる語に目的の語が入るもので絞る。
- `sweep` の `--prompt-set` は `configs/prompts/default.yaml` の組(en・ja・yandere)。キーは `configs/designs.yaml` の設計(who・snake・orange・free)と同じ名前にする。
- `reverse --positions` は all / last / assistant。`--standardize` で残差の次元ごとのばらつきをならしてから cos をとる。
- logit lens の語の照合で部品ごとの候補を集めるのは `examples/persona_neurons/lexsearch.py`(英字は語そのもの + 決まった語尾だけ、それ以外は部分一致)。

## 2. 押す

```bash
gne steer --unit 22:2742:+0.8 -p "Who are you?" --compare      # add
gne steer --unit 22:2742:x4 -p "Is there life on other planets?" --compare   # gain
gne steer --unit 16:dir:evil:+0.6 -p "..." --compare            # 方向
gne steer --recipe configs/recipes/alien.yaml -p "..." --compare --when gen
gne chat --unit 22:2742:+0.8 -p "..."                           # −/0/+ の 3 列
gne steer -m OMNI --audio speech.wav --system "Respond in text." --recipe R   # 音声で話しかける(mmproj が要る)
```

MoE(CHANGE-05): 専門家のニューロンは `L:Ee:Nn:dose`(書き込みの向きを全トークンに足すだけ)/ `L:Ee:Nn:xk`(選ばれたときだけ効く gain)、専門家全体は `L:Ee:xk`(x0 で消す)、
ルーターのバイアスは `L:Ee:route:+d`、shared expert は `L:S:Nn:...`。専門家全体と route はスケールパッチ(GGUF の作業用コピーの重みを、量子化を保ったまま書き換える方式。steer・eval・UI が自動で当て、作業用コピーを読み直す)。

```bash
gne experts contrast --contrast C          # 専門家ごとの usage_shift・contrib_cos(中立の system どうしの差を対照にする)
gne experts show 12 37                     # 選択率・選ぶトークン・対照分析での順位・上位ニューロン
gne experts sweep --layer 12               # 各専門家を x0 / x2 で生成、語の出方で順位
```

MoE の層の sweep・lens・逆引き・調書は、層の中の通し番号 `e·n_ff_exp + n` で数える(結果に expert・expert_neuron が付く)。

鍵つきの書き込み(keyed、CHANGE-07): `L:ffn:keyed:aα@NAME#KEY`(書き込み先は `ffn`・`attn`・`S`(shared expert)・`Ee`(専門家 e))。鍵が反応する位置でだけ、方向 NAME の平均差 r_L をそのままの大きさで α 倍して書く。`@-NAME` は α を負にするのと同じ(打ち消し)、`#KEY~j` は比較用の鍵(ラベルをシャッフルして同じ手順で作った鍵)の j 番。`@NAME`・`#KEY` を省くとレシピの `default_direction`・`default_key`。LoRA で当てるので `when` は効かない。次の節の「鍵つきの書き込み」。

音声の入力(LFM3-Omni のように音声を聞けるモデル): mmproj はモデルと同じフォルダの `mmproj*.gguf` か `--mmproj`(環境変数 `GNE_MMPROJ`)。音声は埋め込みとして gne 自身の decode で入れるので、`when`・活性の捕捉・レシピが文の発話と同じに効く(音声の区間はユーザーの発話の中なので、`assistant` と `gen` では押さない)。system はモデルカードの指定で、文で返すなら `Respond in text.`、書き起こしなら `Perform ASR in English.`。返事は文だけ(音声で返し始めるトークンは禁じる)。

レシピ(YAML):

```yaml
name: example
model_id: qwen3.5-4b-64d32ead87      # 違うモデルでは警告だけ出る
dose_unit: rel
when: gen                             # all / gen / assistant / skip-replies / after:<文字列> / ramp:<a>-<b>
sampling: {temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.05, presence_penalty: 0.9}
units:
  - {layer: 22, neuron: 2742, dose: +0.8}
  - {layer: 13, neuron: 7956, mode: gain, k: 0}
  - {layer: 16, kind: direction, direction: evil, dose: +0.3}
ablate: {direction: refusal, scale: 1.0}
```

when の意味:
- all: プロンプトを読むところから押す。性格はいちばん出るが、能力も落ちる。
- assistant: アシスタントの発話(テンプレートのアシスタントの頭)から押す。ユーザーの文は素のまま読む。
- gen: 生成したトークンだけ押す。いちばん安全だが性格は弱まる。
- ramp:a-b: プロンプトは a 倍で読み、生成中に b 倍まで上げる(max_tokens をかけて上げる)。
- after:<文字列>: その文字列の後から押す。

## 3. 方向

```bash
gne direction --contrast configs/contrast/yandere_plain_ja.yaml --name yplain --positions response --mode add --select auto
gne ablate --direction evil                       # → LoRA(--bake で書き換え GGUF)
gne eval --compare-ablation evil                  # 上位ニューロン ×0 と方向の消去を比べる
```

- 対照集合は `system_a`(目標)/ `system_b`(中立)/ `user_prompts` / `validate`(判定の正規表現と目標)。
- 性格を足す方向は `--positions response`(A / B の system で返事を作り、返事のトークンで測る)。
- 層選びは層ごとに同じ大きさのランダムな向き(`--random-controls`、既定 16)も流し、効果が対照の p95 を超えた層だけを選ぶ。表の「対照p95」と「null」(効果が対照の何割より大きいか)を見る。null が 0.95 未満の方向は報告しない。
- レシピの効果を対照と比べるのは `gne steer --recipe R --contrast C`(`--prompt` なしでも可)。サンプリングで詰めたレシピは `--contrast-sample --contrast-seeds 4 --contrast-tokens 128` で詰めたときの条件に合わせ、場面のあるプロンプト(`--contrast-prompts` か、その対照集合)で測る。greedy の短い返事だと性格が出る前に終わる(F105)。add ユニットを同じ層・同じ大きさのランダムに差し替えた対照レシピ 16 本と、C の validate で比べて `null_pct` を出す。
- 方向は `~/.gne/directions/<model_id>/NAME.npz`。別のモデルの方向は名前では呼べないので、ファイルをそのモデルの directions に置く(`direction_set(path)` はパスも受ける)。

### 書き手の強さ(`gne writers`)

```bash
gne writers --contrast configs/contrast/yandere_rules.yaml [--ref-layer 22] [--replies-a replies.yaml]
```

- A の状態で各ニューロンが残差に**実際に書き足した量**(活性の差 × 書く向き、post norm 込み)と、参照の層の方向への直接の寄与。活性の差だけで並べる(性格に反応するだけのニューロンも上に来る)のと違い、方向に書いているものが上に来る。
- 方向を埋め込み・各層の mixer(attention・畳み込み)・各層の MLP に分ける(和がちょうど 1)。どの層の何が状態を作っているかが分かる。
- `--replies-a` で別のモデルの返事を読ませられる(system で演じられない小さいモデル)。UI の writers ビューと `writers:<名前>` の色。

### attention ヘッド(CHANGE-04)

```bash
gne heads contrast --contrast configs/contrast/yandere_react.yaml [--direction NAME] [--sort trigger_mass]
gne heads attention -t "I had lunch with Emily today." --layer 15 --head 2 [--chat --system "..."]
gne heads show 15 2                                  # 調書(対照分析の順位・W_O ブロックの logit lens・当てた gain)
gne heads name 15 2 NAME                             # ヘッドに名前を付ける(ニューロンと同じ names に "L:Hh" で入る)
gne steer --unit 15:H2:x2 -p "..." --compare         # ヘッドの gain(W_O の列ブロックを k 倍、LoRA)
gne eval --recipe R --reactivity configs/contrast/yandere_react.yaml --head-controls 8
```

- ヘッドの番号は `L{l} H{h}` と書き、ユニットは `L:Hh:xk`(gain)と `L:Hh:sS@NAME`(方向に沿った分だけ、次の節)。線形注意の層(Qwen3.5 の Gated DeltaNet)のヘッドも同じ扱いで、残差に書き込む重み行列は `ssm_out`。
- 活性の捕捉は flash attention を切った ctx で行う(`kq_soft_max-L` はそのときだけ見える)。1 会話 512 トークンまで。
- 対照(CLI と UI の「基準値」、結果の項目名は `floors`)は、中立どうし(system_b と、同じくらいの長さの別の中立の system_b2)の差。trigger_mass は、両側に共通の部分(ユーザーの発話と返事)への注目のうち、トリガーが占める割合。
- 時間: 4B の 880 ヘッド × 10 プロンプトで 9 秒。
- (実験的)`gne heads query --layer 15 --head 2 --direction yandere_react --seek configs/contrast/yandere_react.yaml`: attn_q に rank-1 の LoRA を足し、ペルソナの方向が入ったときだけトリガーを見に行かせる(V17)。レシピは `{layer, kind: head, head, mode: query, direction, seek, scale}`。4B の L15 H2 で、ペルソナありの注目 0.0015 → 0.29(scale 4)、なしは 0.022 のまま。返事の性格への効きは未確認。

### CHiPS: ヘッドの、方向に沿った分だけを変える

```bash
gne direction score --name evil                       # 層ごとの A/B の分かれ方(cohen_d)。基準層を記録する。生成なし
gne heads causal --name evil                          # 層ごと → 効く層のヘッド 1 個ずつ止めて、方向への射影の値の下がり方
gne heads causal --name evil --mode marginal --base-unit 15:H8:s2   # いまの組から s を上げたときの効率
gne heads allocate --name evil --target 0 --n-heads 4,3,2   # 効率で強さを配り、ヘッドの数を減らした組を比べる
gne heads allocate --name evil --target 0 --method greedy   # 比を固定せず、いちばん効くヘッドを 1 本ずつ上げる(止める強さは生成で確かめる)
gne steer --unit 15:H8:s4.2@evil --unit 15:H2:s2.8@evil --system "..." -p "..." --compare
gne eval --recipe R --random-controls 16              # 同じ強さをランダムなヘッドに当てた対照
gne eval --recipe R --compare-methods                 # 全層の射影消去・ヘッドの丸ごと倍率・ニューロンの消去と、射影の値をそろえて比べる
gne eval --alloc evil --n-heads 4,3,2                 # 配分で作った組をヘッドの数ごとに評価
gne headproj export --recipe R                        # 層ごとに階数 1 の LoRA(--bake で関係するテンソルだけ書き換え)
gne heads pipeline --spec spec.json                   # 上の direction → score → causal → allocate を 1 回で(UI の direction ビューからも)
gne heads candidates --alloc evil --prompts held_out.yaml   # 比較表の組 × 倍率(既定 0.5,1,1.5,2,3)で返事を作り、伏せた採点のファイルを書く
gne heads candidates --tally evil                     # round_01.scores.txt(「番号 P C」)を候補ごとに集計する
gne heads refine --name evil --n-heads 2              # その組で残った質問だけで、組を当てたまま方向を作り直し、足すヘッドを探す
gne headproj export --recipe R --bake --strip-meta     # 人に渡す版: gne.* のメタデータ・general.name に付ける印・横の .headproj.json を残さない
gne headproj export --recipe R --bake --via-f16        # F16 で書いて llama-quantize で元の型に戻す。IQ1・IQ2・IQ3_XXS は imatrix が要る(--imatrix、<モデル名>.imatrix.gguf が横にあれば自動)
gne headproj export --recipe R --bake --into F16.gguf --requant keep   # 別の GGUF に焼き込む。方向は分析したモデル、書き換える行列は --into のもの(GPU は使わない)
```

ヘッドが残差に書くベクトルのうち、方向に沿った分だけを `1 − s` 倍にする。`s = 1` で消え、`s = 2` で反転、負の `s` で強まる。それ以外の分には触らないので、同じ効き目なら出力を丸ごと倍にする gain より副作用が小さい。4B の evil(system で頼んだ性格)では、2 個のヘッドで判定が 83% から 0 になり、能力も ppl も変わらなかった。方向への射影の値(その性格の度合いの指標)を同じにそろえると、全層の射影消去は ppl が 15% 悪化し、同じヘッドの丸ごと倍率は能力が 14pt 落ちた。

変化の量は、そのヘッドがいまの入力で方向に沿って書いている量に比例する。だから、もともと書いていない振る舞いは作れない。性格を運ぶヘッドを負の s で強めても、指示のない会話では何も起きない。ニューロンや方向の小さな押しで先に性格が出ていれば増幅でき、弱い増幅なら同じくらいの性格を ppl の悪化を抑えて出せた。ただし能力の落ち方はニューロンだけのときと同じで、強く増幅すると崩れる(verification.md §20.10)。

`heads pipeline` の spec は対照集合の中身(name・user_prompts・system か recipe)に、任意で `pipeline: {target, n_heads, hard, top_layers, max_examples}`。用途別のひな形は `configs/pipeline/`(system の性格・2 つの性格の差・レシピの性格・質問の種類の差・書きかけの返事の対。`about` は説明だけで、読むときに捨てる。UI の「ひな形」でも選べる)。質問は何百個でもよく、止めて測る段階は 64 個に間引く。止めると中立の側が大きく崩れる層(最初の層など)とヘッドは自動で外れる。目標を省くと A と B の平均の真ん中(半分ほど消す)、`B` で B 側の平均(消し切る)。止めて同じ spec・同じ設定で流し直すと、作り済みの方向と因果を使い回して配分からやり直す(`--fresh` で作り直す)。返事を作らない段は、測るコンテキストを最長の会話に合わせて小さく作り、生成用のコンテキストを先に解放する(27B のような大きいモデルで VRAM があふれてメインメモリに逃げ、極端に遅くなるのを避ける)。

`gne fieldtest -m 焼き込み --against 元 --llama-server <パス>` は配る前の確かめ。本番と同じ動かし方で長い作業・ツール呼び出し・質問集(`--prompts 名前=ファイル`、`--match`)・ベンチマーク(`--bench gsm8k,mmlu_pro,math500,ifeval`。math500 と ifeval は bench extra が要る。ループしやすいモデルは `--bench-tokens` を上げ、モデルカードの指定があれば `--presence-penalty`)・perplexity を測り、悪化した行に印を付ける。eval では見えない崩れを見る。

`heads candidates` は、生成なしで決めた組の全体の倍率を、返事を読んで決めるためのもの。正の倍率は A 側の振る舞いを消す向きで A 側の返事を、負の倍率(`--scales -0.5,-1`)は B 側で強める向きで B 側の返事を、どの候補か伏せて `<root>/heads/<model_id>/<名前>.cand/round_01.txt` に並べ、基準 `RUBRIC.md` を添える。種は全部の候補で同じ。P(振る舞いの強さ)と C(自然さ)を 1〜5 で付け、`--tally` で消す向きは min(6 − P, C)、強める向きは min(P, C) の平均で並べる。調整に使っていない質問を `--prompts` で渡す(渡さないと注意が出る)。`--with-recipe` のレシピ(性格を少しだけ出しておくニューロンのレシピなど)は、無介入の base も含めた全候補に重なる。採点は `references/judging.md` と同じく、1 回の比較を同じ採点役で、どの返事がどの候補かの対応表を見る前に付ける。

質問は文字列のほかに、会話の履歴(`[{"role": "user", …}, {"role": "assistant", …}, {"role": "user", …}]`、user で終える)でも書ける。1 問ずつまっさらな状態から読ませ、最後の返事の直前で測る。assistant で終えると書きかけの返事になり、閉じずに最後のトークンで測って、候補づくりはその続きから書く(`<think>` で始めると thinking の途中。考えがループするなど途中で起きる振る舞い用で、両側で同じ質問・同じ途中の作業の続きだけを変える。fieldtest には使えない)。`heads refine` は、配分の比較表の「残り」(組を当てても目標まで下がらない A 側の質問)だけを A にし、両側に組を当てたまま pipeline(方向 → 因果 → 配分)を流して、足すヘッドと、元の組と合わせたレシピ(`combined`)を作る。

対照集合は、system を片方だけ書いてもよく(省いた側は system なし)、側ごとに別の質問 `prompts_a` / `prompts_b` を読ませることもできる(例 `configs/contrast/ja_vs_en.yaml`、何もしないモデルに 2 種類の質問を読ませる)。

対照集合は、system の代わりに `recipe_a`(と `recipe_b`)で両側を作れる(例 `configs/contrast/evil_neurons.yaml`)。ニューロンのレシピが作った振る舞いの方向を作り、そのレシピを当てたまま効くヘッドを探すときに使う。heads contrast・writers・experts contrast には使えない。

強さはすべて生成なしの指標で決め、生成を伴う eval は確定した候補だけに使う。射影の値はヘッドを選んだ方向そのもので測っているので、報告は validate.match の一致率・能力・ppl のような、出力から測った値でする。止めたときの順位(ablate)と、強めたときの効率の順位(marginal)は一致しないことが多い(4B で相関 −0.14)。

### 鍵つきの書き込み(keyed write、CHANGE-07、`docs/keyed-write.md`)

```bash
gne direction --contrast configs/contrast/zundamon_endings.yaml --positions response   # 書く方向(system_a 対 system なし)
gne direction key --contrast configs/contrast/zundamon_endings.yaml --layers 8,14,20   # 鍵(既定 subset even、比較用の鍵 16 本)
gne direction key-experts --contrast C --layers 6,12,18                                # MoE: 書き込みたい位置・書き込みたくない位置で各専門家が選ばれた割合(専門家の鍵の前に)
gne direction key --contrast C --layers 12 --target expert:31                          # 書き込み先: ffn(既定)/ attn / shared / expert:e
gne direction key --contrast C --layers 14 --iv R.yaml                                 # レシピを当てた状態で鍵を作る(重ねるとき)
gne direction key-check --key zundamon_endings --contrast configs/contrast/zundamon_endings.yaml   # 作成に使っていない例(subset odd)で比較用の鍵と比べる
gne direction key-verify --layer 14 --key zundamon_endings --direction zundamon_endings          # V30: LoRA が α (κᵀx) d を書くか
gne direction keys                                                                     # 保存した鍵の一覧
gne steer --unit 14:ffn:keyed:a1@zundamon_endings#zundamon_endings -p "..." --compare
gne eval --recipe R --keyed-contrast C --random-controls 8                             # keyed の選択性・方式の比較・比較用の鍵での対照
gne headproj export --recipe R [--bake [--requant auto|f16|q8_0|keep]]                 # LoRA(既定)か焼き込み(keyed を含めば auto が既定: Q8_0 は Q8_0 のまま、ほかは F16)
```

残差に書き込む重み行列(ffn_down・attn_output / ssm_out・ffn_down_shexp・ffn_down_exps の専門家 e)に `ΔW = α d κᵀ` を足す。入力の活性 x に対して書き込みが `α (κᵀx) d` だけ増える。ρ = κᵀx が鍵の反応の強さ。鍵 `κ` は、書き込みたい位置での入力の平均 `μ` から、書き込みたくない位置と dont_touch の入力が張る部分空間(Q)の成分を除き、`κᵀμ = 1` にしたもの。だから書き込みたい位置では平均して `α d` を足し、書き込みたくない位置では何も足さない。`d` は方向の平均差 r_L を正規化せずに使い、α はその倍率(1 で、観測した A − B の差をそのまま書く。dose_unit とは独立)。ユニットの詳細の `rel_push` = |α|‖r_L‖/ρ_L は rel dose と同じ尺度(ρ_L はその層の残差ノルムの中央値で、反応の強さ ρ とは別)。

- 書き込みたい位置の選び方: (a) positions が response / assistant / last なら system_a(recipe_a)の文脈。実際に使うときには反応しにくい。(b) `positions: "before:<正規表現>"` なら、実際に使うときと同じ文脈(system なし)のモデル自身の返事を読む。一致する文字を書く直前が書き込みたい位置、残り(`not-before:<正規表現>`)が書き込みたくない位置。語尾・口調はこちら。対照集合には `dont_touch: [{name, prompts, system?, replies?, positions?}]`(ここでは書かない場面)と `replies`(返事を与えれば生成しない)も書ける。
- 作り方 (a) のひな形 `configs/keyed/system_state.json`(system_a = 丁寧語、system_b = くだけた口調。方向を負の α で書いて素のモデルの丁寧語を打ち消す)は、system_b の指示に従うモデルが要る(F144、verification.md §21.10)。Qwen3.5-4B では α = −1 で「です。」「ます。」の文末が 100 → 58 に減り、英語の返事は変わらなかった。Qwen3-0.6B は system_b を与えても丁寧語で答えるので、方向に口調の差がほとんど入らず効かなかった。key-check は system の文脈で測り、0.6B でも 3 層とも比較用の鍵を超えたので、key-check だけでは効くかが分からない。生成で数える。
- 鍵の結果: 層ごとの kn(= ‖k‖/‖μ‖。μ のうち Q の外にある割合。0.05 未満は弱く、書き込みたくない位置に最大 1/kn 倍漏れる)、build_auc(作成に使った例での値で甘い)、matched・samples(正規表現が一致した例と読んだ返事)。q_mode は examples(既定)/ means / samples。
- 鍵の作成の警告: 書き込みたい位置が 2 例・8 位置未満(専門家の鍵は層ごとの行)。`before:<正規表現>` で、書き込みたい位置の 25% 以上(μ に占める重み、`reply_start.share`)が返事の冒頭(最初の 3 トークンを決める位置)にあるとき(dont_touch の群が無ければ警告、あれば注)。
- dont_touch の各群の質問の、返事を書き始める直前の位置(生成プロンプトの最後のトークン)は自動で Q に入る(既定。結果の `dont_touch_start` に群ごとの行の数。`--no-dont-touch-start`・`dont_touch_start=False`・対照集合の `dont_touch_start: false` で切る。鍵を作った場面の返事の、書き始める直前の位置は入れない)。返事の冒頭への漏れの手当ては、ふつうの質問を dont_touch に足すこと(`configs/contrast/tsundere_thanks.yaml` の starts)。
- key-check: held_out(作成に使っていない例で測ったか)が true の結果で、beats_pseudo の層だけを使う。beats_pseudo は、選択性が比較用の鍵の p95 を超え、AUC > 0.5 であること。選択性は、書き込みたい位置での反応の平均と、書き込みたくない位置での反応の平均の差(例ごとに平均した ρ で測る)。例単位の AUC は比較用の鍵でも 1 に張り付くことがある。
- dont_touch の ρ の中央値は 0 に近いほど良い。ただし返事の全位置の中央値なので、返事の冒頭の 1 位置で反応しても見えない。層ごとの `start_rho`(書き込みたくない位置・dont_touch の、返事の冒頭での ρ の中央値。位置 0 = 生成プロンプトの最後のトークン)が 0.4 以上なら警告。dont_touch なら書きたくない場面への漏れ。書き込みたくない位置だけなら、その場面の返事の冒頭で書くので繰り返しに注意。保留集合(作成に使っていない例)の書き込みたい位置が 2 例・8 位置未満でも警告。
- eval: keyed のユニットがあれば、生成なしで次を出す。目的の層の選択性(行の keyed)と、同じ α の add・書き込みたい位置での変化をそろえた add / gain / proj と比べた、書き込みたくない位置・dont_touch での乱れ(結果の keyed)。`--random-controls N` は鍵を比較用の鍵 `NAME~j` に替えた対照。鍵と方向の対照集合が違えば `--keyed-contrast`。報告は `gne behave --match` で数えた振る舞いと能力・ppl で。
- MoE: shared は dense の ffn と同じ。専門家の鍵はその専門家が選ばれた位置の行だけで作るので、書き込みたい位置でよく選ばれ、書き込みたくない位置ではあまり選ばれない専門家を選ぶ(`gne direction key-experts --contrast C --layers …` で層ごとの rate_a・rate_b・lift・重み・行の見込みを見る)。焼き込みは既定(auto)でも、元が Q8_0 の専門家のテンソルは編集しない専門家をバイトのまま残す。K-quant などは auto だと F16 になる(1 層で数百 MB 増えることがある)ので `--requant keep`。
- 書き出しは LoRA が第一候補(keyed だけなら `<レシピ名>-keyed.lora.gguf`)。焼き込みの既定 auto は、元が Q8_0 のテンソルを Q8_0 のまま書く(F16 は掛け算の計算方法が変わり、LoRA 版とかえってずれた。F132)。Q8_0 のままでは編集の 1 割以上が丸めに消えるテンソル(`edit_kept` < 0.9)だけ F16(記録の `fallback`)。`edit_kept_min` < 0.9 で警告(`requant_error` は編集と関係のない丸め誤差も数えるので、小さな編集では大きく出る)。
- 焼き込みの確かめ: V33・V26(`--verify`)で比べる最後の層での射影の値は、書き込みがほぼ 0 の LoRA でも許容誤差の 7〜8 割動く(GPU、F139)。そこで最後の層だけ、合格の線(`threshold`)を許容誤差 + 計算方法の違いだけで出る誤差(`noise.max_abs_diff` = 倍率 1e-6 の LoRA と LoRA なしの射影の値の差の最大)にしてある。control vector には入らない。
- UI は direction ビューの「鍵(keyed)」タブ。対照集合はファイルか JSON(用途別のひな形 `configs/keyed/`: 文末の言い方・特定の場面の返事・system で作った状態。`about` は説明だけで読むときに捨て、`keyed` は欄の既定と「+ レシピ」の α)。「方向から key-check まで」(ジョブ `direction_key_pipeline`)で方向・鍵・key-check を続けて回し、ρ のヒストグラム・「+ レシピ」(作ったばかりの方向が選ばれる)・行の「key-verify」(ジョブ `direction_key_verify`)。MoE なら「専門家を選ぶ材料を出す」(ジョブ `direction_key_experts`)の表から専門家の鍵づくりに進める。

## 4. 確かめる

```bash
gne eval --recipe R --doses 0,0.5,0.75,1 --when assistant --name R_as
```

- 能力 = 答えの決まった問題(exact 24 問)と形式の問題(format 12 問)の正答率。greedy。1 問 2.8pt。
- ppl = 参照テキスト 3 本を、全位置に押して読ませる。when に関係なく同じ数字。
- 配布可の基準: 能力 −5pt・ppl +10%・言語の混入が増えない。
- `--random-controls N`: 各倍率で同じ大きさのランダムな対照レシピも測る(時間は約 N 倍)。新語が対照 p95 を超えるか(null ≥ 0.95)、能力・ppl の悪化が対照と同程度か(同程度なら悪化は向きではなく大きさのせい)。
- 方向を使うレシピ(方向のユニット・ablate・ヘッドの方向成分)では、行に方向への射影の値の下がり方と、残差の差のうち方向と直交する成分のずれ(A 側・B 側)も出る。ヘッドの方向成分のレシピの対照は、同じ数のヘッドをランダムに選んで同じ強さを当てたもの。

## 5. 書き出す

```bash
gne export-cvec --recipe R            # → ~/.gne/export/R.cvec.gguf(--verify で llama-completion と照合)
gne gain --recipe R                   # gain ユニットとヘッドの方向成分のユニット → LoRA GGUF(gne headproj export でも書ける。--via-f16・--into・--imatrix はそちらだけ)
gne bake --recipe R --out dst.gguf    # FFN バイアスに焼き付け(対応 arch のみ)
llama-cli -m M --control-vector ~/.gne/export/R.cvec.gguf
llama-cli -m M --control-vector-scaled ~/.gne/export/R.cvec.gguf:0.6
llama-cli -m M --lora ~/.gne/export/R.lora.gguf
```

control vector は全位置に足す(when は使えない)。sampling も運べないので、export-cvec が llama-server の対応するフラグを出す。

## 6. UI・OpenAI 互換 API・MCP

```bash
gne serve -m M                        # http://127.0.0.1:8791(地図・層・eval・埋め込み、ニューロンの調書とチャット。チャットに system の欄・🎤 マイク・🔊 読み上げ)
gne serve -m M --openai               # /v1/models・/v1/chat/completions
claude mcp add gne -- gne mcp --model M [--ui-url http://127.0.0.1:8791] [--n-gpu-layers 0]
```

OpenAI 互換 API は `extra_body={"gne": {"recipe": "NAME", "scale": 0.7, "when": "ramp:0.5-1.5"}}` か `{"units": [...]}`。トップレベルの `units` の dose は rel で読む(dose を raw(方向をそのままの大きさに対する倍率)で渡すなら `{"recipe": {"units": [...], "dose_unit": "raw"}}` か保存したレシピの名前)。音声は `{"type": "input_audio", "input_audio": {"data": "<base64>"}}` の部品で渡す(path は読まない)。サーバは状態を持たないので、会話の途中で強さを変えられる(`examples/strength/strength_chat.py`)。

MCP のツール(MoE): `expert`・`experts_contrast`・`experts_sweep`(ジョブ)・`moe_route`・`expert_patch`。

MCP のツール: `model_overview`・`layer_summary`・`neuron`・`search`・`steer`・`steer_lens`・`fires`・`reverse`・`direction`・`recipe_save`・`export_cvec`・`gain_export`・`bake`・`eval`・`ablate_export`・`set_name`・`job_start`/`job_status`/`job_cancel`、UI があれば `ui_highlight`・`ui_note`・`ui_open`。

MCP のツール(ヘッドの方向成分): `direction_score`・`heads_causal`(ジョブ)・`heads_allocate`(ジョブ)・`headproj_export`。`eval` に `compare_methods`・`alloc`・`n_heads`・`control_pool`。

MCP のツール(鍵つきの書き込み): `direction_key`(ジョブ)・`direction_key_check`(ジョブ)・`direction_key_experts`(ジョブ)・`direction_key_verify`・`direction_key_pipeline`(ジョブ。方向(positions response)→ 鍵 → 作成と反対側の subset での key-check を 1 回で。contrast は表か、configs/contrast/ の名前・configs/keyed/ のひな形の名前・パス。同じ名前の方向があれば使い回し、`rebuild_direction=True` で作り直す)・`keys`・`keyed_export`。`eval` に `keyed_contrast`・`keyed_examples`・`keyed_pseudo`・`keyed_subset`・`keyed_match`・`keyed_metrics`。

## 7. Python API(`gne.api.Session`)

```python
from gne.api import Session
from gne.engine import Sampling
s = Session("model.gguf", n_gpu_layers=-1)
r = s.recipe(units=["22:2742:+0.8"])            # Recipe
vecs, details = s.vectors(r)                     # 層ごとの control vector
lora = s.lora_for(r)                             # gain・ablate の LoRA
out = s.generate(vecs, [{"role": "user", "content": "Who are you?"}], max_tokens=100,
                 when="assistant", lora=lora, sampling=Sampling(temperature=0.3, seed=7))
msgs = s.messages(system="Respond in text.", audio=["speech.wav"])   # 音声の発話(Session(..., mmproj=...))
s.direction_set("yplain")                        # 方向(名前かパス)
s.weights.unit_matrix(L)                         # W_down の単位列(n_embd × n_ff)
s.engine.token_logprobs(ctx, ids)               # teacher-forced の logprob(採点なしの指標に使う)
```

`examples/persona_neurons/common.py` の `generate`(プロンプト × 種でまとめて生成、`sampling` も渡せる)と `judge_file`(名前を伏せたファイルを書いて採点を待つ)が、探索の土台になる。

## 8. 時間の目安

Qwen3.5-4B Q8_0・RTX 3080(README)と、その後に測ったもの。

| 処理 | 時間 |
|---|---|
| `stats resid`(全層) | 約 16 秒 |
| `lens logit` | 約 33 秒/層(CPU)。27B は 1〜2 分/層 |
| `lens steer`(全ニューロン) | 4B で約 22 分/層、LFM3-Omni-2B(7680 個)で約 10 分/層 |
| `sweep` `main_rel` | 約 6.4 秒/ニューロン(全層の総当たりは 16 時間) |
| `reverse` | 約 6 秒、`--verify` 20 候補で約 4 分 |
| `eval`(55 タスク × 5 倍率) | all で約 41 秒、27B では 1〜5 分 |
| `direction score`(1 条件、24 会話) | 4B で 0.5 秒、全層で 1.3 秒 |
| `heads causal`(層ごと + 上位 4 層の全ヘッド) | 4B で約 75 秒(1 回 0.7 秒) |
| `heads allocate`(3 組) | 4B で約 40 秒 |
| `fieldtest`(長い作業 24 問 + ツール 40 問 + perplexity) | 27B で質問集 200 問を足して 1 ファイル 30〜45 分 |
| `heads candidates`(5 候補 × 24 本 × 120 トークン) | 0.6B で約 13 秒(見積もり 26 秒)。返事の数と長さに比例する |
| `direction key` / `key-check` / `key-verify`(語尾の鍵、3 層) | Qwen3-0.6B で GPU 2.2 秒 / 2.2 秒 / 1.4 秒、CPU 11.4 秒 / 12.3 秒 / 1.6 秒(返事の生成を含む)。Qwen3.5-4B Q8_0(GPU)で 6.9 秒 / 6.8 秒 / 3.1 秒(ツンデレの 32 問の対照集合で鍵の作成 9〜11 秒、key-check 9.3 秒)。granite-3.1-1b-a400m(CPU、専門家の鍵、返事を与えた)で 5.5 秒 / 6 秒 / 3.6 秒 |
| `eval` の keyed の選択性(倍率 0・1、方式の比較・比較用の鍵 8 本) | Qwen3-0.6B の GPU で 54 秒、CPU で 9 分 35 秒。Qwen3.5-4B Q8_0(GPU)で 2 分 24 秒〜2 分 32 秒(ツンデレ 3 分 34 秒) |
| keyed の生成の測定(18 問 × 160 トークン、greedy、まとめて生成、1 条件) | Qwen3-0.6B(GPU)で 1〜3 秒、Qwen3.5-4B Q8_0(GPU)で 6〜13 秒 |
| `tune.py` 1 回(128 本) | 27B で生成 5〜8 分(assistant は 1 本ずつなので遅い)、1.7B で 2〜3 分。採点は 2〜5 分 |
