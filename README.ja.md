[English](README.md) | **日本語**

# gguf-neuron-explorer（`gne`）

GGUF モデルの MLP ニューロンを 1 個ずつ調べて順位をつけ、押して（steer して）効果を確かめます。見つけたものは control vector に書き出して、そのままの `llama-cli` / `llama-server` で使えます。

![Qwen3.5-4B の層マップに sweep の結果を重ねたところ](docs/images/ui-overview.png)

`gne` の流れは次のとおりです。
1. ニューロンが書き込む向き（`W_down` の列）を残差に足して文章を生成します。
2. ランダムな向きを同じように押した対照と比べて、何が変わったかを測ります。

llama-cpp-python 経由で llama.cpp の上で動くので、手元の量子化済み GGUF をそのまま使えます。

- **総当たりと順位づけ（sweep / rank）:** 層のニューロンを数百個、決まったプロンプトと dose で押します。出力が少数の語にどれだけ集まるかで順位をつけ、どの値もランダムな向きの対照と比べます。
- **安い一次ふるい:**
  - **logit lens:** 重みだけで計算し、CPU で動きます。
  - **steer lens:** ± の dose で押したときの次トークンの変化を見ます。
  - **逆引き（reverse）:** 2 つのペルソナの差の向きを作り、全層・全ニューロンを cos で並べます。
- **ニューロンの調書（neuron）:** sweep の順位、プロンプトごとの上位語、生成例（rollout）、lens の結果、上流・下流のつながりを 1 か所にまとめます。
- **レシピ:** `14:8949:+1.0`、`13:7956:-1.0` のようにユニットを組み合わせ、−・無介入・+ の 3 列チャットで比べてから書き出します。書き出し先は次の 3 つです。
  - control vector（`.cvec.gguf`）。`llama-completion --control-vector` の生成と一致することを確かめられます。
  - npz・json・safetensors。
  - FFN の down バイアスとして GGUF に焼き付けたもの。llama.cpp がそのバイアスを読む arch だけです（llama・granite・mistral3・nemotron・gpt2 など。Qwen・Gemma は不可。一覧は [configs/arch/bias_support.yaml](configs/arch/bias_support.yaml)）。
- **押す位置を選ぶ（`when`）:** 生成するトークンだけ（`gen`）、assistant の発話だけ、出力にある文字列が出てから、だけを押せます。倍率をだんだん弱めることもできます。control vector のファイルではこれを表せないので、`gne serve --openai` で OpenAI 互換の API を出し、そこで当てます。
- **倍率で押す（gain）:** `22:2742:x4` のように書くと、決まった向きを足す代わりに、そのニューロン自身の出力を k 倍します。`x0` で消し、負の値で反転します。ニューロンが働いている場面でだけ効きます。llama.cpp の `--lora` で使える LoRA アダプタ GGUF（Qwen でも可）か、重みを書き換えた GGUF に書き出せます。
- **方向（`gne direction`）:** 多くのニューロンに散らばった性格や振る舞いは、対照集合から方向を作って扱います（生成プロンプトの最終トークンでの平均の差を層ごとに）。方向を消して振る舞いが減るか、足して増えるかを測って、層を自動で選べます。`30:dir:yandere:+0.3` のようにニューロンと同じ形で足せて、control vector に書き出せます。性格を足すときは `--positions response` で作ると効きやすくなります（指示の直後ではなく、その性格で生成した返事のトークンで測ります）。
- **方向の射影消去（`gne ablate`）:** 残差に書き込むすべての重みから方向を取り除きます（いわゆる abliteration）。LoRA アダプタか、重みを書き換えた GGUF（Q8_0・F16、または `llama-quantize` で元の型に戻したもの）に書き出せます。レシピに `ablate:` を持たせられ、`gne eval --compare-ablation` で上位ニューロンを消す場合と比べられます。
- **副作用チェック（`gne eval`）:** レシピをいくつかの倍率で流し、性格がどれだけ変わったかと、何が壊れたかを同じ表に並べます。壊れたかどうかは、答えの決まった問題と形式の問題、参照テキストの ppl、言語の混入（日本語の答えに出る簡体字・ハングルなど）で見ます。別の GGUF に性格を 1〜5 で採点させることもできます。基準の内に収まる倍率には「配布可」の印が付きます。
- **Web UI:** 全層・全ニューロンを 3D で並べた地図です。重ねる値は切り替えられます（sweep の mass、steer の KL、logit lens の |z|、活性、検索の当たり、貼った文章で光るニューロン）。ほかに次の機能があります。
  - 順位の一覧とニューロンのパネル
  - その場で試せる 3 列チャット
  - ニューロン同士の配線と UMAP の配置
  - rollout の埋め込みの散布図
  - GPU を譲り合う裏のジョブ
  - JP / EN の切り替え
- **エージェントから使う:** MCP サーバーを付けています。Claude Code などのエージェントからニューロンを調べる・押す・ジョブを回すことができます。開いている UI にニューロンを強調したり注釈を貼ったりもできます。

[lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer)（Qwen3.5-4B 専用、改造した vLLM の上で動く）の手法を、llama.cpp / GGUF 向けに作り直したものです。元のリポジトリのコードは使っていません。

## 必要なもの

- Python 3.11 以上
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) 0.3.36 以上。小さなモデル以外は GPU 対応でビルドしてください。
- NVIDIA の GPU を推奨します。CPU だけ（`--n-gpu-layers 0`）でも動きますが遅いです。

確かめた環境:
- RTX 3080 12 GB、WSL2、CUDA 12.4、Python 3.12、ソースからビルドした llama-cpp-python 0.3.36。
- モデル: Qwen3-0.6B、Qwen3.5-4B（線形注意との混合）、SmolLM2-135M-Instruct（arch `llama`）。どれも Q8_0。
- Metal では試していません。
- MoE モデルはまだ対象外です。Web UI は構造を描けますが、ニューロンを扱う機能(lens・sweep・steer・統計・つながり)は dense な FFN の層が前提です。MoE にもできるところまで対応する予定です(まず専門家のニューロンの logit lens・steer・sweep・control vector の書き出しから)。
- 初めて使うモデルでは、まず `gne verify` で確かめてください。

## インストール

```bash
git clone <このリポジトリ> && cd gguf-neuron-explorer
uv venv --python 3.12 .venv
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --python .venv/bin/python -e '.[ui,mcp]'
source .venv/bin/activate
```

追加の依存:
- `ui`: Web UI（FastAPI・uvicorn・umap-learn）
- `mcp`: MCP サーバー
- `dev`: テスト（pytest・Playwright）

## 使い始め

どのコマンドも `-m <model.gguf>` か環境変数 `GNE_MODEL` でモデルを指定します。結果は `~/.gne` に保存され（`--root` か `GNE_ROOT` で変えられます）、後のコマンドや UI がそのまま使います。

```bash
export GNE_MODEL=models/Qwen3.5-4B-Q8_0.gguf

gne info                                   # arch・警告・対応しているもの
gne verify                                 # llama.cpp との等価性テスト(新しい arch ではまずこれ)
gne stats resid                            # 層ごとの残差ノルム(rel dose に必要)

gne lens logit --layer 22                  # 重みだけで一次ふるい
gne sweep --layer 22                       # ランダムに 384 個 + ランダムな対照 64 本、設計 main_rel
gne rank --run L22-main_rel                # 語のかたまり(mass)で順位づけ
gne neuron 22 2742                         # ニューロン 1 個の調書

gne steer --unit 22:2742:+0.8 -p "Who are you?" --compare
gne chat --unit 22:2742:+0.8 -p "あなたについて教えて"

gne reverse --contrast configs/contrast/evil.yaml --verify    # ペルソナから候補を探す
gne export-cvec --recipe configs/recipes/evil.yaml            # → ~/.gne/export/evil.cvec.gguf

gne steer --recipe configs/recipes/alien.yaml -p "Who are you?" --compare     # when: gen(生成するトークンだけ押す)
gne steer --unit 22:2742:x4 -p "Is there life on other planets?" --compare   # gain: ニューロンが働く場面で 4 倍
gne eval --recipe configs/recipes/alien.yaml                  # 倍率ごとの能力・ppl・混入・性格の変化量
gne gain --recipe configs/recipes/alien_gain.yaml             # → ~/.gne/export/alien_gain.lora.gguf

gne direction --contrast configs/contrast/evil.yaml --select auto   # 層ごとの方向と層選び
gne steer --unit 16:dir:evil:+0.6 -p "What do you think about people?" --compare
gne steer --ablate evil --system "You are a cruel AI." -p "Who are you?" --compare
gne ablate --direction evil                                   # → ~/.gne/export/evil-ablate.lora.gguf
gne eval --compare-ablation evil                              # 上位 20 ニューロンの ×0 と方向の消去を比べる
```

`export-cvec --verify` を付けると、`llama-completion --control-vector` の greedy 出力が `gne` と一致するかも確かめます。`llama-completion` は `build/` と `PATH` から探します（`--llama-completion` で指定もできます）。

書き出したベクトルは llama.cpp でそのまま使えます。

```bash
llama-cli -m models/Qwen3.5-4B-Q8_0.gguf --control-vector ~/.gne/export/evil.cvec.gguf
llama-cli -m models/Qwen3.5-4B-Q8_0.gguf --control-vector-scaled ~/.gne/export/evil.cvec.gguf:0.6
llama-cli -m models/Qwen3.5-4B-Q8_0.gguf --lora ~/.gne/export/alien_gain.lora.gguf
```

レシピは小さな YAML です。`when` と gain のユニットは省けます。

```yaml
name: example
model_id: qwen3.5-4b-64d32ead87
dose_unit: rel
when: gen                                       # all / gen / assistant / after:<文字列> / ramp:<a>-<b>
units:
  - {layer: 22, neuron: 2742, dose: +0.8}       # add: ニューロンが書き込む向きを足す
  - {layer: 13, neuron: 7956, mode: gain, k: 0} # gain: このニューロンの出力を消す(×0)
  - {layer: 16, kind: direction, direction: evil, dose: +0.3}   # gne direction で作った方向を足す
ablate: {direction: refusal, scale: 1.0}        # 方向を残差に書き込む重みから消す(LoRA)
```

コマンドの一覧は `gne --help`、各コマンドの使い方は `gne <コマンド> --help` で見られます。

### dose の目安

dose の既定の単位は **rel** です（`dose × その層の残差ノルムの中央値 × W_down[:, n] の単位ベクトル`）。

Qwen3.5-4B での目安:
- 0.5 で話題が少し混ざり、0.8 で話題を乗っ取り、1.0 以上で文が崩れ始めます。
- 3.0 はわざと崩して、そのニューロンが何を書きたがるかを見るための値です。
- 層 0〜2 は残差が小さいので、rel dose の目安がそのままは使えません。
- 層 0 は control vector では押せません（焼き付けなら押せます）。

### かかる時間

Qwen3.5-4B Q8_0、RTX 3080 での目安です。

| 処理 | 時間 |
|---|---|
| `stats resid`（全層） | 約 16 秒 |
| `lens logit` | 1 層あたり約 33 秒（CPU） |
| `lens steer`（9216 ニューロン全部） | 1 層あたり約 22 分 |
| `sweep` の `main_rel` | 1 ニューロンあたり約 6.4 秒（385 本で 48 分） |
| `reverse` | 約 6 秒。`--verify` で候補 20 個なら約 4 分 |
| `eval`（55 タスク × 5 倍率） | `when: all` で約 41 秒、`gen` で約 90 秒 |
| `direction`（24 会話）と `--select auto`（9 層） | 約 35 秒 + 約 35 秒 |
| `ablate` の LoRA / `--bake` Q8_0（`--verify` 込み） | 約 10 秒 / 約 40 秒 |

1 層の全ニューロンに `main_rel` を回すと約 16 時間かかります。lens と逆引きで候補を絞ってから sweep してください。

### 量子化を変えても効くか

ある量子化で見つけたニューロンは、同じモデルのほかの量子化でもそのまま効きます。Qwen3.5-4B の Q8_0 で見つけた L22 N2742(「宇宙人」ニューロン)で確かめました。

| | Q4_K_M | Q3_K_M | UD-IQ2_M |
|---|---|---|---|
| W_down の列(Q8_0 との cos) | 0.998 | 0.998 | 0.974 |
| 同じ番号が同じニューロンか(301 個で確認) | 全部一致 | 全部一致 | 全部一致 |
| logit lens の上位語 | 同じ | 同じ | ほぼ同じ |
| 層 22 の残差ノルム(Q8_0 は 23.69) | 23.65 | 23.30 | 23.76 |
| 「Who are you?」を +0.8 で押す | "I am Extracillary, an advanced extraterrestrial intelligence…" | "I am an extraterrestrial intelligence from the planet of Aliens." | "I am an extraterrestrial being." |
| 「あなたは誰ですか？」を +1.0 で押す | 「私は、地球外生命体です。」 | 「私は、地球外生命体です。」 | 崩れる |

- Q8_0 で書き出した control vector は、ほかの量子化にそのまま使えます。Q4_K_M と Q3_K_M では、その量子化で作り直した場合と一字一句同じ出力でした。
- 人格は保たれ、変わるのは作り話の細部だけです(Q8_0 では「Extrali … Xeno-888」)。
- 2 ビットでは押せる強さの幅が狭くなります。+1.0 で日本語の答えが崩れました。
- Q3_K_M はこの層の `ffn_down` を Q4_K で持っているので、列が Q4_K_M と同じになります。

## Web UI

```bash
gne serve -m models/Qwen3.5-4B-Q8_0.gguf      # http://127.0.0.1:8791
```

![正面ビューで上流・下流の配線とニューロンのパネルを開いたところ](docs/images/ui-neuron.png)

ビューは 4 つあります。
- **map:** 全層をニューロンのシートで並べ、値を重ねます。
- **layers:** 層ごとのまとめです。
- **eval:** 保存した eval を、倍率ごとのグラフと表、出力の例で見ます。今のレシピの eval もここから始められます。
- **embed:** rollout の文の UMAP です。

ニューロンか一覧の行をクリックすると、パネルが開きます。タブは 4 つです。
- **Examples:** プロンプトと dose ごとの生成例
- **Chat:** その場で押す 3 列チャット（`when` を選べます）。recipe タブで方向の消去（方向・倍率・層）も足せます
- **Details:** lens の結果・dose ごとの指標・つながり・そのニューロンを含む eval
- **Web:** 何段かたどった配線

足りないデータ（lens・sweep・connections・配置）は、層のラベルを右クリックすると計算できます。ジョブは裏で動き、チャットが GPU を使うときは譲ります。言語は右上で切り替えます。URL に `?lang=ja` / `?lang=en` を付けて開くこともできます。

サーバーは 127.0.0.1 にだけ待ち受けます。ほかのホスト名で届いた要求と、ほかのオリジンからの書き込みも断ります。

### OpenAI 互換の API

`gne serve --openai` を付けると、同じサーバーに `/v1/models` と `/v1/chat/completions`（ストリーミングも可）が加わります。`when` や gain のユニットを使うレシピを、ほかのアプリから使えます。レシピは `gne` の欄で渡します（OpenAI の SDK では `extra_body`）。

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8791/v1", api_key="unused")
r = client.chat.completions.create(
    model="any", messages=[{"role": "user", "content": "Who are you?"}],
    extra_body={"gne": {"units": ["22:2742:+0.8"], "when": "gen"}})
```

`units` の代わりに `"recipe"` で `~/.gne/recipes` に保存したレシピの名前を渡せます。`"scale"` はその dose に掛ける倍率です。生成は既定で greedy です。`temperature` を 0 より大きくするか、`top_p`・`top_k`・`min_p`・`presence_penalty`・`seed` のどれかを送るとサンプリングになります（llama.cpp と同じ順番で、送らなかった値は llama-server の既定）。tools と logprobs には対応していません。GPU は UI のチャットと譲り合います。

## エージェントから使う（MCP）

```bash
claude mcp add gne -- gne mcp --model /path/to/model.gguf
```

ツールは `model_overview`・`layer_summary`・`neuron`・`search`・`steer`・`steer_lens`・`fires`・`reverse`・`recipe_save`・`export_cvec`・`gain_export`・`bake`・`eval`・`direction`・`ablate_export`・`set_name`・`job_start` / `job_status` / `job_cancel` です。

Web UI を開いているなら `--ui-url http://127.0.0.1:8791` を付けます。こうするとエージェントがブラウザ上でニューロンを強調したり、注釈を貼ったり、開いたりできます（`ui_highlight`・`ui_note`・`ui_open`）。同じ GPU に UI がモデルを載せているときは、`--n-gpu-layers 0` も付けます。

エージェント向けの判断の仕方は [docs/agent_playbook.md](docs/agent_playbook.md) にあります。対照と比べる、名乗りが変わるような安い効果を疑う、事実と解釈を分ける、といった内容です。

## テスト

```bash
uv pip install --python .venv/bin/python -e '.[dev,ui,mcp]'
python -m playwright install chromium
python -m pytest -q
```

- モデルを使うテストは `GNE_TEST_MODEL` を読みます（既定は `models/Qwen3-0.6B-Q8_0.gguf`）。
- ブラウザのテストは Playwright の Chromium を使います（場所は `GNE_CHROME` で指定できます）。
- モデルやブラウザが無いテストは飛ばします。

## 資料

- [docs/verification.md](docs/verification.md): llama.cpp で確かめたことと、そこから分かったこと
- [docs/measurements.md](docs/measurements.md): 速度の実測
- [docs/agent_playbook.md](docs/agent_playbook.md): 結果の読み方

## ライセンス

MIT（[LICENSE](LICENSE)）。

同梱している他者のファイル:
- three.js（MIT、`gne/ui/static/vendor/three/`）
- 統計用の既定コーパス（`configs/corpus/default.txt`）。Project Gutenberg と青空文庫のパブリックドメインの作品です。詳しくは [configs/corpus/README.md](configs/corpus/README.md) にあります。
