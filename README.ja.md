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
- MoE モデルはまだ対象外です。Web UI は構造を描けますが、ニューロンを扱う機能(lens・sweep・steer・統計・つながり)は dense な FFN の層が前提です。
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
```

`export-cvec --verify` を付けると、`llama-completion --control-vector` の greedy 出力が `gne` と一致するかも確かめます。`llama-completion` は `build/` と `PATH` から探します（`--llama-completion` で指定もできます）。

書き出したベクトルは llama.cpp でそのまま使えます。

```bash
llama-cli -m models/Qwen3.5-4B-Q8_0.gguf --control-vector ~/.gne/export/evil.cvec.gguf
llama-cli -m models/Qwen3.5-4B-Q8_0.gguf --control-vector-scaled ~/.gne/export/evil.cvec.gguf:0.6
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

1 層の全ニューロンに `main_rel` を回すと約 16 時間かかります。lens と逆引きで候補を絞ってから sweep してください。

## Web UI

```bash
gne serve -m models/Qwen3.5-4B-Q8_0.gguf      # http://127.0.0.1:8791
```

![正面ビューで上流・下流の配線とニューロンのパネルを開いたところ](docs/images/ui-neuron.png)

ビューは 3 つあります。
- **map:** 全層をニューロンのシートで並べ、値を重ねます。
- **layers:** 層ごとのまとめです。
- **embed:** rollout の文の UMAP です。

ニューロンか一覧の行をクリックすると、パネルが開きます。タブは 4 つです。
- **Examples:** プロンプトと dose ごとの生成例
- **Chat:** その場で押す 3 列チャット
- **Details:** lens の結果・dose ごとの指標・つながり
- **Web:** 何段かたどった配線

足りないデータ（lens・sweep・connections・配置）は、層のラベルを右クリックすると計算できます。ジョブは裏で動き、チャットが GPU を使うときは譲ります。言語は右上で切り替えます。URL に `?lang=ja` / `?lang=en` を付けて開くこともできます。

サーバーは 127.0.0.1 にだけ待ち受けます。ほかのホスト名で届いた要求と、ほかのオリジンからの書き込みも断ります。

## エージェントから使う（MCP）

```bash
claude mcp add gne -- gne mcp --model /path/to/model.gguf
```

ツールは `model_overview`・`layer_summary`・`neuron`・`search`・`steer`・`steer_lens`・`fires`・`reverse`・`recipe_save`・`export_cvec`・`bake`・`set_name`・`job_start` / `job_status` / `job_cancel` です。

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
