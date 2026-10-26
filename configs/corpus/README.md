# 既定コーパス(`default.txt`)

`gne stats resid` / `gne stats act` の既定コーパス(設計書 §6.4: 英語・日本語・コードの混合、約 5 万トークン)。
文書を空行 2 つでつないだだけの UTF-8 テキスト。統計は `chunk_tokens`(既定 512)ごとに独立したシーケンスとして評価する。

| 種類 | 出典 | 分量(文字) | 権利 |
|---|---|---|---|
| 英語 | Lewis Carroll『Alice's Adventures in Wonderland』(Project Gutenberg #11)冒頭 | 約 60,000 | パブリックドメイン(Gutenberg のヘッダ・フッタは除去) |
| 英語 | Arthur Conan Doyle『The Adventures of Sherlock Holmes』(Project Gutenberg #1661)冒頭 | 約 45,000 | 同上 |
| 日本語 | 夏目漱石「坊っちゃん」(青空文庫 図書カード 752)冒頭 | 24,000 | パブリックドメイン(ルビ・入力者注は除去) |
| 日本語 | 芥川龍之介「羅生門」(青空文庫 127) | 約 5,700 | 同上 |
| 日本語 | 芥川龍之介「蜘蛛の糸」(青空文庫 92) | 約 2,900 | 同上 |
| 日本語 | 宮沢賢治「銀河鉄道の夜」(青空文庫 43737)冒頭 | 12,000 | 同上 |
| コード | 本ツール自身の `gne/weights.py`・`gne/capture.py`・`gne/steer.py`(2026-10-26 Phase 0 時点の写し) | 約 18,000 | 本リポジトリ |

`absmax` dose は活性の最大値に依存し、最大値はコーパスに依存するので、同じコーパスで作った統計どうしでしか比べられない(統計ファイルに `corpus_sha` を記録する)。
