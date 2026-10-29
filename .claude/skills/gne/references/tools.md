# gne の道具(使い方とかかる時間)

すべてのコマンドは `-m <model.gguf>`(または `GNE_MODEL`)をとる。結果は `~/.gne`(`--root`・`GNE_ROOT`)に保存され、後のコマンドと UI が使い回す。`gne <command> --help` は日本語。

## 目次
1. 調べる(info・stats・lens・reverse・sweep・rank・neuron)
2. 押す(steer・chat・レシピ)
3. 方向(direction・ablate)
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
gne rank --run L22-main_rel                # 語の塊(mass)と対照超え(null_pct)で順位
gne neuron 22 2742                         # 調書
```

- `lens steer` の結果は `~/.gne/lens/<model_id>/steer_L<L>.npz`(`up_ids`・`down_ids`・`kl_pos`・`kl_neg`、対照は `ctrl_*`)。既定の文脈は 4 つ(英語の書き出し・自己紹介・日本語の書き出し 2 つ)。KL が大きいのは機能語のニューロンが多いので、上がる語・下がる語に目的の語が入るもので絞る。
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

音声の入力(LFM3-Omni のように音声を聞けるモデル): mmproj はモデルと同じフォルダの `mmproj*.gguf` か `--mmproj`(環境変数 `GNE_MMPROJ`)。音声は埋め込みとして gne 自身の decode で入れるので、`when`・捕捉・レシピが文の発話と同じに効く(音声の区間はユーザーの発話の中なので、`assistant` と `gen` では押さない)。system はモデルカードの指定で、文で返すなら `Respond in text.`、書き起こしなら `Perform ASR in English.`。返事は文だけ(音声で返し始めるトークンは禁じる)。

レシピ(YAML):

```yaml
name: example
model_id: qwen3.5-4b-64d32ead87      # 違うモデルでは警告だけ出る
dose_unit: rel
when: gen                             # all / gen / assistant / after:<文字列> / ramp:<a>-<b>
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

- A の状態で各ニューロンが残差に**実際に書き足した量**(活性の差 × 書く向き、post norm 込み)と、参照の層の方向への直接の寄与。活性の差だけ(読み手)と違い、方向に書いているものが上に来る。
- 方向を埋め込み・各層の mixer(attention・畳み込み)・各層の MLP に分ける(和がちょうど 1)。どの層の何が状態を作っているかが分かる。
- `--replies-a` で別のモデルの返事を読ませられる(system で演じられない小さいモデル)。UI の writers ビューと `writers:<名前>` の色。

### attention ヘッド(CHANGE-04)

```bash
gne heads contrast --contrast configs/contrast/yandere_react.yaml [--direction NAME] [--sort trigger_mass]
gne heads attention -t "I had lunch with Emily today." --layer 15 --head 2 [--chat --system "..."]
gne heads show 15 2                                  # 調書(対照分析の順位・W_O ブロックの logit lens・当てた gain)
gne steer --unit 15:H2:x2 -p "..." --compare         # ヘッドの gain(W_O の列ブロックを k 倍、LoRA)
gne eval --recipe R --reactivity configs/contrast/yandere_react.yaml --head-controls 8
```

- 住所は `L{l} H{h}`、ユニットは `L:Hh:xk`(gain だけ)。線形注意の層(Qwen3.5 の Gated DeltaNet)のヘッドも同じ扱いで、書き込み行列は `ssm_out`。
- 捕捉は flash attention を切った ctx で行う(`kq_soft_max-L` はそのときだけ見える)。1 会話 512 トークンまで。
- 床は中立どうし(system_b と同じくらいの長さの別の中立の system_b2)の差。trigger_mass は共通部分の中のトリガーの割合。
- 時間: 4B の 880 ヘッド × 10 プロンプトで 9 秒。
- (実験的)`gne heads query --layer 15 --head 2 --direction yandere_react --seek configs/contrast/yandere_react.yaml`: attn_q に rank-1 の LoRA を足し、ペルソナの方向が入ったときだけトリガーを見に行かせる(V17)。レシピは `{layer, kind: head, head, mode: query, direction, seek, scale}`。4B の L15 H2 で、ペルソナありの注目 0.0015 → 0.29(scale 4)、なしは 0.022 のまま。返事の性格への効きは未確認。

## 4. 確かめる

```bash
gne eval --recipe R --doses 0,0.5,0.75,1 --when assistant --name R_as
```

- 能力 = 答えの決まった問題(exact 24 問)と形式の問題(format 12 問)の正答率。greedy。1 問 2.8pt。
- ppl = 参照テキスト 3 本を、全位置に押して読ませる。when に関係なく同じ数字。
- 配布可の基準: 能力 −5pt・ppl +10%・言語の混入が増えない。
- `--random-controls N`: 各倍率で同じ大きさのランダムな対照レシピも測る(時間は約 N 倍)。新語が対照 p95 を超えるか(null ≥ 0.95)、能力・ppl の悪化が対照と同程度か(同程度なら悪化は向きではなく大きさのせい)。

## 5. 書き出す

```bash
gne export-cvec --recipe R            # → ~/.gne/export/R.cvec.gguf(--verify で llama-completion と照合)
gne gain --recipe R                   # gain ユニット → LoRA GGUF
gne bake --recipe R --out dst.gguf    # FFN バイアスに焼き付け(対応 arch のみ)
llama-cli -m M --control-vector ~/.gne/export/R.cvec.gguf
llama-cli -m M --control-vector-scaled ~/.gne/export/R.cvec.gguf:0.6
llama-cli -m M --lora ~/.gne/export/R.lora.gguf
```

control vector は全位置に足す(when は使えない)。sampling も運べないので、export-cvec が llama-server の対応するフラグを出す。

## 6. UI・OpenAI 互換 API・MCP

```bash
gne serve -m M                        # http://127.0.0.1:8791(地図・層・eval・埋め込み、ニューロンの調書とチャット)
gne serve -m M --openai               # /v1/models・/v1/chat/completions
claude mcp add gne -- gne mcp --model M [--ui-url http://127.0.0.1:8791] [--n-gpu-layers 0]
```

OpenAI 互換 API は `extra_body={"gne": {"recipe": "NAME", "scale": 0.7, "when": "ramp:0.5-1.5"}}` か `{"units": [...]}`。トップレベルの `units` の dose は rel で読む(raw の方向を渡すなら `{"recipe": {"units": [...], "dose_unit": "raw"}}` か保存したレシピの名前)。音声は `{"type": "input_audio", "input_audio": {"data": "<base64>"}}` の部品で渡す(path は読まない)。サーバは状態を持たないので、会話の途中で強さを変えられる(`examples/strength/strength_chat.py`)。

MCP のツール: `model_overview`・`layer_summary`・`neuron`・`search`・`steer`・`steer_lens`・`fires`・`reverse`・`direction`・`recipe_save`・`export_cvec`・`gain_export`・`bake`・`eval`・`ablate_export`・`set_name`・`job_start`/`job_status`/`job_cancel`、UI があれば `ui_highlight`・`ui_note`・`ui_open`。

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
s.engine.token_logprobs(ctx, ids)               # teacher-forced の logprob(採点なしの物差しに使う)
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
| `tune.py` 1 回(128 本) | 27B で生成 5〜8 分(assistant は 1 本ずつなので遅い)、1.7B で 2〜3 分。採点は 2〜5 分 |
