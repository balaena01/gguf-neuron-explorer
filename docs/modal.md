# Modal で動かす

手元の GPU やディスクに載らないモデルで、gne のコマンドを [Modal](https://modal.com/) の GPU で回す。使った秒数だけ課金され、コマンドが終わるとコンテナも止まる。

## 最初の準備

1. Modal のアカウントを作り、支払い方法を登録する(GPU を使うのに必要)。
2. Usage & Billing で、月の上限(budget)を決める。
3. `modal` コマンドを入れてログインする。

```
uv tool install modal
modal setup
```

## 使い方

```
# モデルを Modal に直接落とす(手元のディスクは使わない。GPU も使わない)
gne modal fetch unsloth/Qwen3.8-Flash-Next-GGUF --include '*UD-IQ1_S*'

# 質問集を上げる
gne modal put spec.json

# gne のコマンドを GPU で実行する
gne modal run -- verify -m /models/unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf --layer 2
gne modal run -- heads pipeline --spec /data/inputs/spec.json -m /models/.../....gguf --yes

# 結果を持ち帰る
gne modal ls gne
gne modal get gne/<model_id> ./results

# 動いているアプリ・止める・今月の料金
gne modal ps
gne modal stop --all
gne modal cost
```

`--` のあとは、手元の gne に渡すのと同じ引数を書く。自分の Python のスクリプトを動かすときは、`gne modal put` で上げて `gne modal run --script /data/inputs/x.py -- 引数` とする。見積もりが 30 分を超えるコマンドは確認で止まるので、`--yes` を付ける。

| 置き場所 | コンテナの中 | Volume の名前 |
|---|---|---|
| モデル(読み取り専用) | `/models` | `gne-models` |
| 結果(`GNE_ROOT`) | `/data/gne` | `gne-data` |
| `gne modal put` で上げたもの | `/data/inputs` | `gne-data` |

## UI で見る

```
gne modal serve -m /models/.../....gguf --gpu H200
```

Web UI を Modal の GPU で動かし、手元のブラウザの `http://127.0.0.1:8791/` で開く。チャットを打ちながらレシピの強さを変えられる。動いている間は GPU の料金がかかる。

- Ctrl+C で止まり、課金も止まる。ほかに、時間の上限(`--timeout`、既定 2 時間)と、SSH の接続がないときの停止(`--idle`、既定 15 分)がある。
- 保存したレシピは `/data/gne` に残る。
- Modal の公開 URL は使わない。`gne serve` はコンテナの中でも 127.0.0.1 だけで待ち受け、その接続だけの使い捨ての鍵を持つ SSH のトンネルでつなぐ。SSH はパスワードでは入れず、UI のポートへの転送しかできない。
- チャットの内容は Modal のコンテナを通る。ログには出ない。

## Hugging Face に上げる

作ったモデルは手元に移さず、Modal から直接上げられる。トークンは一度だけ Modal のシークレットに入れる(書き込み権限のトークン)。

```
modal secret create huggingface HF_TOKEN=<トークン>
gne modal hf-upload --repo <ユーザー>/<名前> --card README.md export/<名前>-00001-of-00004.gguf export/<名前>-00002-of-00004.gguf ...
```

既定は非公開のリポジトリで、`--public` で公開にする。名前を変えたときの古いファイルなど、上げたあとで消すものは `--remove <ファイル名>` で指定する(何度でも書ける)。消したファイルもコミットの履歴には残る。Modal から外への転送は課金の対象だが、Starter プランは月 1 TiB まで無料。

## 選ぶ GPU

既定は A100 80 GB で、空きがなければ H100 にする。`--gpu` で変えられる。重みを読まないコマンド(`info --no-load` など)の確認は、一番安い `T4` で足りる。GPU なしのコンテナには CUDA のドライバーが入らず、gne が動かない。

重みと計算用のバッファが VRAM に載る必要がある。Qwen3.8-Flash-Next の IQ1_S は重みが 66.4 GB で、バッファは `n_ubatch` 512 で約 0.6 GB なので、80 GB に載る(実サイズのヘッダーで測った値。GPU での実測ではない)。

料金は [公式のページ](https://modal.com/pricing)で確かめる。2026 年 11 月の時点で、既定の組み合わせ(A100 80 GB、8 コア、32 GiB)は 1 時間あたり約 3.13 ドル、Starter プランの無料枠は月 30 ドル。CPU とメモリは、指定した量と実際に使った量の多いほうで課金される。

## 知っておくこと

- **Modal に上がるもの:** git で追跡している `gne/` と `configs/`、`gne modal put` で名指ししたファイルだけ。追跡していないファイルは上がらない。作業ツリーで編集中の追跡ファイルは、編集後の中身が上がる。
- **接続が切れても続く:** `gne modal run` は手元の接続が切れても、コマンドが終わるまで動き続け、課金も続く。途中で止めるには `modal app stop <App ID>` を使い、`modal app list` で止まったことを確かめる。App ID は実行の最初に出る。
- **時間の上限:** 1 回の実行につき既定 6 時間(`--timeout`、最大 24 時間)。
- **途中で止められる(プリエンプション):** GPU のコンテナは Modal の都合で止められることがあり、そのときは同じコマンドが最初から実行し直される。`heads pipeline` や `sweep` は保存済みの続きから動くが、ほかのコマンドはやり直しになる。
- **llama-server:** `gne fieldtest` は Modal に対応していない。
- **Hugging Face のトークン:** `gne modal fetch` は公開リポジトリだけを落とせる。
