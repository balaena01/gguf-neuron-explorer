# 実測性能(Phase 0 / V7)

設計書 §10 の見積りを置き換えるための実測。`gne bench --reps 10` の中央値。

## 環境

- GPU: NVIDIA GeForce RTX 3080 12GB(Windows 11 上の WSL2、ドライバ 610.88、CUDA 12.4)。Windows 側も同じ GPU を使っているので、数値は ±10% 程度ぶれる
- CPU: 8 スレッド、RAM 16GB(WSL の上限)
- llama-cpp-python 0.3.36(ソースからビルド、`-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86`)、同梱 llama.cpp `0c1e570`(2026-10-01)
- コンテキスト: steer ctx は `n_ctx 4096`(シーケンス間で共有、`kv_unified=True`)、`n_seq_max 8`、`n_batch 2048`、`n_ubatch 512`。probe ctx は `n_seq_max 1`、`n_ubatch 512`。flash attention は既定(auto)

## 結果

測り方:
- 「prefill」は 4 本 × 20 トークンを 1 バッチで評価し、各本の最終位置ロジットを読むまで(読むときに GPU と同期する)。
- 「`set_cvec` 直後の prefill」は毎回違うベクトルを `set_cvec` してから同じ prefill を測ったもの。`set_cvec` は毎回グラフの再予約(`sched_reserve`)を立てるので、その分が上乗せされる。
- 「decode 1 step」は `llama_decode` の後に `llama_synchronize` まで待った時間(`llama_decode` は GPU の完了を待たずに戻るので、同期しないと過小に出る)。
- 「生成」は EOG で止めない貪欲生成(各ステップで選択トークンの logprob と top-5 を記録)。

### Qwen3-0.6B Q8_0(層 L14)

| 処理 | 時間 |
|---|---|
| prefill 4 seq × 20 tok(cvec 変更なし) | 6.8 ms |
| `set_cvec` の呼び出しだけ | 0.70 ms |
| `set_cvec` 直後の prefill(グラフ再予約込み) | 21.1 ms |
| → 再予約の上乗せ | 14.3 ms |
| `set_cvec` + prefill 合計 | 21.8 ms |
| 層範囲を 1..n_layer-1 に固定した `set_cvec` 直後の prefill | 20.1 ms |
| decode 1 step(4 seq) | 5.5 ms |
| 生成 128 tok × 4 seq | 644 ms(795 tok/s) |
| 生成 128 tok × 1 seq | 381 ms(336 tok/s) |
| probe ctx prefill 512 tok(捕捉なし / 全層 l_out / 全層 FFN 活性) | 17.9 / 27.2 / 65.7 ms |

### Qwen3.5-4B Q8_0(層 L16)

| 処理 | 時間 |
|---|---|
| prefill 4 seq × 20 tok(cvec 変更なし) | 23.2 ms |
| `set_cvec` の呼び出しだけ | 0.94 ms |
| `set_cvec` 直後の prefill(グラフ再予約込み) | 45.0 ms |
| → 再予約の上乗せ | 21.9 ms |
| `set_cvec` + prefill 合計 | 46.2 ms |
| 層範囲を 1..n_layer-1 に固定した `set_cvec` 直後の prefill | 45.1 ms |
| decode 1 step(4 seq) | 13.5 ms |
| 生成 128 tok × 4 seq | 1813 ms(282 tok/s) |
| 生成 128 tok × 1 seq | 1128 ms(114 tok/s) |
| probe ctx prefill 512 tok(捕捉なし / 全層 l_out / 全層 FFN 活性) | 89.9 / 105.2 / 219.7 ms |

## 読み取り

- **`set_cvec` + prefill は 4B で 46 ms**。V7 の基準(100 ms を大きく超えたら §10 の対策)を満たすので、対策は今は不要。
- 再予約の上乗せ(4B で 22 ms)は prefill 本体とほぼ同じ大きさ。lens steer ではこれが時間の約半分になる。
- **§10 4(a)「層範囲を固定して呼ぶ」は効かない**。`llama_context::set_adapter_cvec` は範囲に関係なく毎回 `sched_need_reserve = true` を立てる(src/llama-context.cpp)。実測でも差は無い(45.0 → 45.1 ms)。再予約を消すには §10 4(c) の小パッチ(データ更新だけなら再予約しない)が要る。
- 生成ループの Python 側の上乗せは 1 step あたり 1 ms 未満(4B の 4 本: decode 13.5 ms に対して 1 step 14.2 ms)。4 本まとめると 1 本ずつより 2.5 倍速い。
- probe ctx の cb_eval は、捕捉なしでも 1 回の prefill に数 ms〜十数 ms 上乗せされる(ask 段階で全ノードの Python コールバックが呼ばれる)。全層の FFN 活性(4B で 512 × 32 × 9216 × 4 バイト ≈ 600 MB をホストへ転送)を捕まえても 220 ms。

## §10 の見積りの置き換え(Qwen3.5-4B Q8_0、RTX 3080)

上の実測から計算した値。生成は全部 128 トークンまで続いた場合の上限で、EOG で早く止まればもっと短い。

| 処理 | §10 の見積り(4090) | 実測からの見積り(3080) | 計算 |
|---|---|---|---|
| `stats resid`(50k トークン) | 1〜3 分 | 約 10 秒 | 512 tok × 98 回 × 105 ms |
| `lens steer`(1 層、9216 ニューロン) | 10〜15 分 | 約 14 分 | ニューロンあたり 2 ×(set_cvec + prefill)= 92 ms |
| `sweep quick`(300 候補) | 25 分 | 約 14 分 | 2 回の set_cvec + 生成(2 本まとめて 128 tok ≈ 1.5 s + 1 本 128 tok 1.1 s)≈ 2.7 s/候補 |
| `sweep main_rel`(300 候補) | 40 分 | 約 32 分 | 4 回の set_cvec + 生成(3 本まとめて 128 tok ≈ 1.7 s を 3 回 + 1 本 1.1 s)≈ 6.4 s/候補 |
| `sweep main_rel`(全 9216) | 約 20 時間/層 | 約 16 時間/層 | 同上 |
| UI の fires(200 トークン、別冊 §9 の目標 2 秒) | — | 捕捉まで 0.1〜0.2 秒 | 512 tok で 220 ms。上位 k の抽出と転送は U2 で測る |

`stats resid` と `stats act` は Phase 1 で実装して実測に置き換える。

## Phase 1(2026-10-26、同じ環境)

`gne` を 1 回起動して終わるまでの時間(モデルの読み込みを含む)。統計は既定コーパス(`configs/corpus/default.txt`)。

| 処理 | 時間 | 備考 |
|---|---|---|
| `stats resid`(Qwen3.5-4B、全 31 層、59k トークン) | 16.3 秒 | §10 の見積り 1〜3 分。512 トークン × 116 本 |
| `stats resid`(Qwen3-0.6B、全 27 層、62k トークン) | 4.9 秒 | |
| `stats act`(Qwen3.5-4B、全 31 層、59k トークン) | 265 秒 | ほとんどが Python 側のヒストグラム集計(1 本あたり 31 層 × 512 × 9216 値)。設計書の見積り 5〜15 分 |
| `stats act`(Qwen3-0.6B、5 層) | 約 13 秒 | |
| `steer --compare`(Qwen3.5-4B、ユーザーのレシピ、無介入 57 tok + 介入 128 tok) | 約 10 秒 → **2.9 秒** | 下の「GGUF の読み込み」の直し前 → 後 |
| `export-cvec --verify`(Qwen3.5-4B、`llama-completion` の起動を含む) | 約 11 秒 → **4.7 秒** | 同上 |
| `info`(Qwen3.5-4B、llama.cpp での読み込み確認あり / なし) | 1.6 秒 / **0.28 秒**(直し前 6.3 秒) | 同上 |
| `bake --verify`(SmolLM2-135M Q8_0、焼いた GGUF の書き出し + 2 モデルでの生成) | 2.4 秒 | |

### GGUF の読み込み

gguf-py の `GGUFReader` は語彙とマージ表(Qwen3.5 で計 49.6 万個の文字列)を 1 要素ずつ numpy のビューにするので、開くだけで Qwen3.5-4B で 5.9 秒、Qwen3-0.6B で 3.6 秒かかり、すべてのコマンドの起動に乗っていた。メタデータとテンソル情報だけを読む `gne/gguf_fast.py` に替えて **0.025 秒**になった(全フィールドの値・配列の長さ・全テンソルの型・形状・位置・データが `GGUFReader` と一致することをテストで確認)。全メタデータを写す bake だけは `GGUFReader` を使う。

## Phase 2(2026-10-26、同じ環境)

| 処理 | 時間 | 備考 |
|---|---|---|
| `lens logit`(Qwen3.5-4B、1 層 9216 ニューロン、24.8 万語) | **33 秒** | CPU(numpy、8 スレッド)。§10 の見積り 1〜2 分。直す前の上位 k の取り方では 0.6B で 21 秒/層だった(verification.md F20) |
| `lens logit`(Qwen3-0.6B、1 層 3072 ニューロン、15.2 万語) | 5.1 秒 | |
| `lens steer`(Qwen3.5-4B、1 層 9216 ニューロン + 対照 64、dose ±0.8、文脈 4 本) | **22 分 22 秒**(145 ms/ユニット) | §10 の見積り(4090)10〜15 分/層。1 ユニット = 2 × (set_cvec + 4 本の prefill)。そのうち約 44 ms がグラフの再予約(F6) |
| `lens steer`(Qwen3-0.6B) | 58 ms/ユニット(1 層なら約 3 分) | |
| `reverse`(Qwen3.5-4B、12 プロンプト × 2、全 31 層の方向・活性差・cos) | 6 秒 | |
| `reverse --verify`(上位 20 候補 × quick 設計、層ごとに対照 4 本、14 層、計 270 rollout) | 235 秒(0.87 秒/rollout) | 同じ dose のプロンプトを 2 本ずつまとめて生成 |

steer lens の 1 層 22 分のうち、グラフの再予約(set_cvec のたびに走る)が約 3 割を占める。全層(31 層)を回すと約 11.5 時間なので、全層を見るなら logit lens(全層で約 17 分)で先にふるうのがよい。

## Phase 3(2026-10-26、同じ環境)

| 処理 | 時間 | 備考 |
|---|---|---|
| `sweep --design main_rel`(Qwen3.5-4B、L22、385 ニューロン + 対照 64、max_tokens 128) | **47 分 38 秒**(6.4 秒/ユニット) | §10 の見積り(4090)は 8 秒/候補。1 ユニット = 4 回の set_cvec + 3 本ずつ 3 回 + 1 本の生成。rollout の 79% が 128 トークンで打ち切り(capped) |
| `rank`(上の run、4494 rollout) | 数秒 | 語彙の読み込み(vocab_only)0.8 秒を含む |
| `sweep --design quick`(0.6B、2 ニューロン + 対照 2、max_tokens 16) | 数秒 | テスト用 |

`main_rel` を層の全 9216 ニューロンに回すと約 16 時間(6.4 秒 × 9216)。一次ふるい(logit lens / steer lens / 逆引き)で数百に絞ってから回すのが現実的(§10 の方針どおり)。
