"""`--json` 出力のスキーマ。設計書 §6.13 (a)、V10。

規約:
- stdout には JSON だけを出す。最後の行が結果のオブジェクト。長い処理はその前に進捗を JSONL で 1 行ずつ出す。
- すべての出力に schema_version, command, model_id, llama_cpp_version, seed, generated_at を含める。
- 失敗は非ゼロ終了 + {"error": {"code", "message", "hint"}}(エンベロープ付き)。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Envelope(_Strict):
    schema_version: Literal["1"]
    command: str
    model_id: str | None
    llama_cpp_version: str
    seed: int
    generated_at: str
    warnings: list[str] = []


class ErrorBody(_Strict):
    code: str
    message: str
    hint: str | None = None


class ErrorOut(Envelope):
    error: ErrorBody


class ProgressBody(_Strict):
    stage: str
    done: int
    total: int


class ProgressLine(_Strict):
    progress: ProgressBody


# --- info / probe / verify / bench --------------------------------------------------------------------

class MetaModel(_Strict):
    path: str
    model_id: str
    name: str
    arch: str
    n_layer: int
    n_embd: int
    n_ff: int
    n_ff_per_layer: list[int] | None
    n_vocab: int
    n_expert: int
    has_post_ffn_norm: bool
    ffn_down_scale: dict[str, float] | None
    tied_output: bool
    has_chat_template: bool
    ffn_down_qtype: str
    file_size: int
    cvec_supported: bool = True
    is_split: bool = False
    hc_streams: int = 0


class LoadInfo(_Strict):
    ok: bool
    n_layer: int | None = None
    n_embd: int | None = None
    n_vocab: int | None = None
    error: str | None = None


class BakeSupport(_Strict):
    supported: bool
    mode: str | None
    reason: str
    checked_llama_cpp: str


class InfoOut(Envelope):
    meta: MetaModel
    arch_warnings: list[str]
    llama_cpp: LoadInfo | None
    bake: BakeSupport
    stats: dict[str, Any]
    residual_writers: dict[str, Any] | None = None    # §6.17 残差ライター(射影消去の対象)
    speed: dict[str, Any] | None = None               # CHANGE-03 C: GPU / CPU の層と推定の速さ
    workflow: list[dict[str, Any]] | None = None      # CHANGE-03 C: 大型モデルの手順と見積もり


class ProbeOut(Envelope):
    layer: int | None
    names: list[str]
    groups: dict[str, list[int]]
    act_name: str | None


class VResultModel(_Strict):
    id: str
    title: str
    status: Literal["pass", "fail", "info", "skip"]
    detail: str
    data: dict[str, Any]


class VerifyOut(Envelope):
    layer: int
    results: list[VResultModel]
    n_pass: int
    n_fail: int


class BenchOut(Envelope):
    bench: dict[str, Any]


# --- stats --------------------------------------------------------------------------------------------

class CorpusModel(_Strict):
    path: str
    sha: str
    n_sequences: int | None = None


class ResidModel(_Strict):
    layers: list[int]
    p10: list[float]
    p50: list[float]
    p90: list[float]
    mean: list[float]
    n_tokens: int
    corpus_sha: str
    chunk_tokens: int
    exclude_pos0: bool


class StatsResidOut(Envelope):
    path: str
    stats: ResidModel
    corpus: CorpusModel
    seconds: float


class ActLayerModel(_Strict):
    layer: int
    tensor_name: str
    path: str
    n_tokens: int
    absmax_median: float
    q999_median: float
    rms_median: float


class StatsActOut(Envelope):
    layers: list[ActLayerModel]
    corpus: CorpusModel
    seconds: float


# --- steer / export / import / bake --------------------------------------------------------------------

class UnitModel(_Strict):
    layer: int
    neuron: int | None
    kind: Literal["neuron", "random", "direction", "head", "expert_neuron", "expert", "shared", "ffn", "attn"]   # MoE は CHANGE-05、ffn・attn は CHANGE-07
    seed: int | None
    dose: float
    norm: float
    rel_push: float | None
    rho: float | None
    add: float | None
    down_norm: float | None
    mode: Literal["add", "gain", "query", "route", "proj", "keyed"] = "add"     # §6.16、query は CHANGE-04 E、route は CHANGE-05、proj は CHANGE-06、keyed は CHANGE-07
    k: float | None = None
    direction: str | None = None              # §6.17 方向のユニット
    diff_rel: float | None = None             # ‖平均差‖ / ρ_L
    head: int | None = None                   # CHANGE-04 attention ヘッドの gain
    seek: str | None = None                   # CHANGE-04 E query の対照集合
    expert: int | None = None                 # CHANGE-05 専門家の番号
    semantics: str | None = None              # CHANGE-05 MoE の add は "direction"(書き込み方向の押し)
    key: str | None = None                    # CHANGE-07 keyed の鍵(比較用の鍵は NAME~j)
    alpha: float | None = None                # CHANGE-07 keyed の強さ α(raw、k と同じ値)


class RecipeModel(_Strict):
    name: str | None
    model_id: str | None
    dose_unit: Literal["rel", "absmax", "raw"]
    when: str = "all"                 # §6.15
    units: list[dict[str, Any]]
    ablate: dict[str, Any] | None = None      # §6.17
    sampling: dict[str, float] | None = None  # 試すときのサンプリング(seed なし)
    default_direction: str | None = None      # CHANGE-06: proj・keyed のユニットの方向の既定
    direction_layer: Literal["own", "base"] = "own"
    default_key: str | None = None            # CHANGE-07: keyed のユニットの鍵の既定


class Generation(_Strict):
    text: str
    finish: Literal["eog", "length"]
    n_gen: int
    token_ids: list[int] | None = None
    when: dict[str, Any] | None = None   # 位置別ステアリングの記録(set_cvec の回数、区間など)


class SteerOut(Envelope):
    messages: list[dict[str, Any]]                 # content は文字列か、音声の入った部品のリスト
    recipe: RecipeModel
    units: list[UnitModel]
    layer_norms: dict[str, float]
    baseline: Generation | None
    steered: Generation | None                     # --contrast だけ(プロンプトなし)なら None
    differs: bool | None
    when: str = "all"
    sampling: dict[str, float] | None = None       # 使ったサンプリング(seed 込み)。greedy なら None
    behavior: dict[str, Any] | None = None         # CHANGE-03 A: --contrast の効果とランダムな対照レシピ(null_pct)


class ReproCheck(_Strict):
    tool: str
    prompt: str
    n_tokens: int
    ours: str
    theirs: str
    match: bool                       # 合格(exact か、僅差の分岐で説明できる)
    exact: bool                       # greedy 列が先頭 n_tokens まで完全に一致
    first_diff: int | None = None
    summary: str | None = None
    kl_next: float | None = None      # プロンプト末尾の次トークン分布の KL(ours‖theirs)
    differs_from_baseline: bool | None = None


class ExportOut(Envelope):
    path: str
    format: Literal["cvec", "npz", "json", "safetensors"]
    recipe: RecipeModel
    units: list[UnitModel]
    layer_norms: dict[str, float]
    verify: ReproCheck | None
    llama_server_args: str | None = None             # レシピの sampling を llama-server で使う引数


class ImportOut(Envelope):
    source_file: str
    source: Literal["exact", "generator"]
    layer_norms: dict[str, float]
    path: str


class BakeReportModel(_Strict):
    src: str
    dst: str
    arch: str
    mode: str
    layers: list[int]
    added: list[str]
    modified: list[str]
    n_tensors: int


class BakeOut(Envelope):
    recipe: RecipeModel
    units: list[UnitModel]
    report: BakeReportModel
    verify: ReproCheck | None


class BakeModified(_Strict):                     # 焼き込み(gne gain --bake・headproj export --bake)で書き換えたテンソル 1 つ
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    tensor: str
    from_: str | None = Field(None, alias="from")   # 元の型
    to: str | None = None                       # 書いた型(--via-f16 は元の型に戻るので from と同じ)
    via: str | None = None                      # スケールパッチ(CHANGE-05)なら "scale_patch"
    bytes_from: int | None = None               # 元の大きさ(バイト)
    bytes_to: int | None = None                 # 書いた大きさ(--via-f16 では出さない)
    requant_error: float | None = None          # proj・keyed: 量子化し直しで失った分 ÷ 編集の大きさ
    edit_kept: float | None = None              # proj・keyed: 量子化し直しても残った編集の割合 ⟨足された分, ΔW⟩/‖ΔW‖²(1 = 全部残った)
    fallback: dict[str, Any] | None = None      # keyed の既定(auto): Q8_0 では編集が埋もれるので F16 にした {from, edit_kept}
    experts: list[int] | None = None            # CHANGE-07 MoE: 専門家の 3 次元テンソル(ffn_down_exps)で編集した専門家の番号
    untouched: str | None = None                # CHANGE-07 MoE: 編集しない専門家のスラブ(raw = バイトのまま、"Q8_0->F16" = 変換した)


class BakeVerifyScore(_Strict):                 # V26・V33 の射影の値の判定(層・側の 1 つ)
    lora: float                                 # LoRA 版の射影の値の平均
    baked: float                                # 焼き込み版の射影の値の平均
    max_abs_diff: float                         # 例ごとの |LoRA 版 − 焼き込み版| の最大
    resolution: float                           # 許容誤差 = その層の両側の射影の値の標準偏差の 1 割
    threshold: float | None = None              # 合格の線(最後の層は 許容誤差 + noise、ほかは許容誤差)。古い結果には無い
    noise: float | None = None                  # 最後の層だけ: 計算方法の違いだけで出る誤差(BakeVerify.noise の max_abs_diff)
    ok: bool


class BakeVerifyNoise(_Strict):                 # 計算方法の違いだけで出る射影の値の誤差(最後の層の合格の線に足す)
    layer: int                                  # 測った層(最後の層)
    max_abs_diff: float                         # 書き込みがほぼ 0 の LoRA と LoRA なしの、例ごとの射影の値の差の最大(両側)
    by_side: dict[str, float]                   # 側ごとの最大
    lora_scale: float                           # 書き込みがほぼ 0 の LoRA の倍率(1e-6。0 ちょうどだと LoRA が外れるおそれ)
    seconds: float                              # この測定にかかった時間


class BakeVerify(_Strict):                      # CHANGE-06 V26・CHANGE-07 V33(headproj export --bake --verify)
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    scores: dict[str, BakeVerifyScore]          # "<層><側>"(例 "26A")
    kl_first_token: float
    top1_match: float
    top1_mismatch_gaps: list[float]             # 最尤トークンが分かれた例の、LoRA 版での差(nats)
    n_prompts: int
    ppl_lora: float | None
    ppl_baked: float | None
    ppl_diff_pct: float | None
    base_layer: int | None = None               # 方向の基準の層
    last_layer: int | None = None               # 合格の線に noise を足した層(基準の層と同じなら None)
    noise: BakeVerifyNoise | None = None        # LoRA が無いか、last_layer が None なら None
    pass_: bool = Field(alias="pass")
    note: str | None = None


class GainOut(Envelope):                        # §6.16 gne gain
    kind: Literal["lora", "bake"]
    recipe: RecipeModel
    path: str
    bytes: int
    layers: dict[str, list[list[float]]] | None = None     # lora: 層 → [[ニューロン, k]]
    rank: dict[str, int] | None = None
    modified: list[BakeModified] | list[str] | None = None  # bake: 書き換えたテンソルと型(分割の焼き込みは名前だけ、詳しくはサイドカー)
    n_tensors: int | None = None
    heads: dict[str, list[list[float]]] | None = None      # CHANGE-04 lora: 層 → [[ヘッド, k]]
    head_rank: dict[str, int] | None = None
    query_loras: list[str] | None = None                   # CHANGE-04 E: クエリ操作の LoRA(ユニットごとに 1 つ)
    experts: dict[str, list[list[float]]] | None = None    # CHANGE-05 lora: 層 → [[専門家, ニューロン, k]]
    shared: dict[str, list[list[float]]] | None = None
    method: str | None = None                              # CHANGE-05 bake: "patch"(スケールパッチ)
    patch_bytes: int | None = None
    ops: list[dict[str, Any]] | None = None
    proj_rank: dict[str, int] | None = None                # CHANGE-06: 方向成分の射影の層ごとの階数
    proj_layers: dict[str, dict[str, Any]] | None = None   # CHANGE-06: 層 → {heads, rank, directions, folded}
    sidecar: str | None = None                             # CHANGE-06 bake: <out>.headproj.json
    requant: str | None = None                             # CHANGE-06 bake --via-f16、CHANGE-07 keyed の既定 auto
    llama_quantize: str | None = None
    intermediate_bytes: int | None = None
    verify: BakeVerify | None = None                       # CHANGE-06 V26・CHANGE-07 V33(headproj export --bake --verify)
    parts: list[str] | None = None                         # CHANGE-06 分割 GGUF への焼き込みの出力ファイル
    requant_error_max: float | None = None                 # 量子化し直しで失った分 ÷ 編集の大きさ(テンソルの中の最大)
    edit_kept_min: float | None = None                     # 量子化し直しても残った編集の割合(テンソルの中の最小)
    fallback: list[str] | None = None                      # 分割 GGUF への焼き込み(auto): Q8_0 では編集が埋もれるので F16 にしたテンソル
    strip_meta: bool | None = None                         # gne の記録を残さない版
    local_path_keys: list[str] | None = None               # メタデータに残った手元のパスらしい文字列のキー
    into: str | None = None                                # 焼き込む先の別の GGUF
    imatrix: str | None = None                             # --via-f16 で使った imatrix
    imatrix_auto: bool | None = None
    keyed_layers: dict[str, dict[str, Any]] | None = None  # CHANGE-07: 層 → {tensors: {テンソル: 階数}, keys, directions, rank}
    keyed_rank: dict[str, int] | None = None               # CHANGE-07: テンソル → keyed の階数(= その層の鍵の数)
    expert_bytes_added: int | None = None                  # CHANGE-07 MoE: 専門家の 3 次元テンソルの書き換えで増えた大きさ(バイト。keep なら 0)


class EvalRow(_Strict):                         # §6.14 gne eval の 1 倍率
    dose: float
    exact: float | None
    format: float | None
    capability: float | None                    # exact と format を合わせた正答率
    ppl: dict[str, float]
    ppl_mean: float | None
    mix_zh: float
    mix_other: float
    persona: dict[str, Any]                     # novelty / mass / top_words / distinct4 / length / capped
    failed: dict[str, list[str]]
    capability_drop_pt: float
    ppl_worse_pct: float
    mix_increase: float
    ok: dict[str, bool]
    shippable: bool
    judge: float | None = None
    judge_scores: list[int | None] | None = None
    ctrl: dict[str, Any] | None = None          # CHANGE-03 A: 対照レシピの分布(novelty / capability_drop_pt / ppl_worse_pct / rows)
    ctrl_novelty_p95: float | None = None
    ctrl_capability_drop_p50: float | None = None
    ctrl_ppl_worse_p50: float | None = None
    novelty_null_pct: float | None = None
    reactivity: dict[str, Any] | None = None    # CHANGE-04 F: with / without / reactivity / increase(・ctrl・null_pct)
    direction: dict[str, Any] | None = None     # CHANGE-06 F.1: 方向を使うレシピの射影の値の変化と直交成分のずれ(層ごと)
    dir_drop_a: float | None = None
    drift_par_a: float | None = None
    drift_orth_a: float | None = None
    drift_par_b: float | None = None
    drift_orth_b: float | None = None
    rate: float | None = None                   # 対照集合の validate.match の一致率(A 側)
    rate_drop: float | None = None
    dir_null_pct: float | None = None           # CHANGE-06 F.2: 同数・同じ s のランダムなヘッドに対する null_pct
    drift_orth_b_null_pct: float | None = None
    rate_null_pct: float | None = None
    keyed: dict[str, Any] | None = None         # CHANGE-07 E.1: 目的の層の書き込みたい位置・書き込みたくない位置・dont_touch の r̂ 方向の変化と eval の選択性
    notes: list[str] | None = None              # V29: 対照を超えていないときの明示


class EvalOut(Envelope):
    name: str
    path: str
    created: str
    recipe: RecipeModel
    when: str
    suite: dict[str, Any]
    doses: list[float]
    rows: list[EvalRow]
    samples: dict[str, Any]
    judge: dict[str, str] | None
    seconds: float
    random_controls: int = 0                    # CHANGE-03 A: 倍率ごとの対照レシピの本数
    reactivity: dict[str, Any] | None = None    # CHANGE-04 F: 反応性に使った対照集合
    control_pool: str | None = None             # CHANGE-06 F.2: proj の対照のヘッドの母集団
    direction_metrics: dict[str, Any] | None = None   # CHANGE-06 F.1
    keyed: dict[str, Any] | None = None         # CHANGE-07 E: 方式の比較・比較用の鍵・text(測れなければ skipped・hint)
    notes: list[str] | None = None


class DirectionLayer(_Strict):
    layer: int
    norm: float
    rel: float | None
    cos_prev: float | None


class DirectionOut(Envelope):                   # §6.17 gne direction
    name: str
    path: str
    contrast: str
    contrast_sha: str
    positions: str
    n_prompts: int
    n_rows: dict[str, int]
    layers: list[DirectionLayer]
    validation: dict[str, Any]                 # 対照集合の validate(層選びの判定)
    select: dict[str, Any] | None
    seconds: float
    sides: dict[str, Any] | None = None        # レシピで A / B を作った対照集合のとき、側ごとの system とレシピ


class AblateOut(Envelope):                      # §6.17 gne ablate
    kind: Literal["lora", "bake"]
    direction: str
    from_layer: int | None = None
    per_layer: bool | None = None
    path: str
    bytes: int
    layers: list[int]
    scale: float
    embed: bool
    n_tensors: int | None = None                # lora
    requant: str | None = None                  # bake: q8_0 / f16 / via-f16
    modified: list[dict[str, str]] | None = None
    n_modified: int | None = None
    llama_quantize: str | None = None
    intermediate_bytes: int | None = None
    verify: dict[str, Any] | None
    seconds: float


class CompareRow(_Strict):
    method: Literal["neurons", "direction"]
    dose: float
    rate: float | None                          # 消したい振る舞いの判定の一致率(A 側)
    rate_drop: float | None
    kl_b: float | None
    capability_drop_pt: float
    ppl_worse_pct: float
    mix_increase: float
    novelty: float
    shippable: bool
    sample: str | None


class EvalCompareOut(Envelope):                 # §6.17 gne eval --compare-ablation(command は "eval")
    direction: str
    from_layer: int
    top: int
    neurons: list[dict[str, float]]
    evals: dict[str, str]
    contrast: str | None
    match: str | None
    base_rate: float | None
    rows: list[CompareRow]
    seconds: float
    path: str


# --- Phase 2: lens / reverse -----------------------------------------------------------------------------

class LogitNeuron(_Strict):
    neuron: int
    peak_z: float
    up: list[str]
    up_z: list[float]
    down: list[str]
    down_z: list[float]


class LensLogitOut(Envelope):
    layer: int
    path: str
    n_neurons: int
    top_k: int
    seconds: float
    neurons: list[LogitNeuron]


class NullSummary(_Strict):
    n: int
    p50: float | None
    p95: float | None
    max: float | None


class SteerNeuron(_Strict):
    neuron: int
    kl_pos: float
    kl_neg: float
    pct_pos: float
    pct_neg: float
    up: list[str]
    up_delta: list[float]
    down: list[str]
    down_delta: list[float]


class LensSteerOut(Envelope):
    layer: int
    path: str
    dose: float
    rho: float
    n_done: int
    n_selected: int
    n_controls: int
    null: dict[Literal["pos", "neg"], NullSummary]
    seconds: float
    neurons: list[SteerNeuron]
    workers: dict[str, Any] | None = None       # CHANGE-03 B: --workers で分けたときの部分ごとの結果


class CandidateModel(_Strict):
    layer: int
    neuron: int
    cos: float
    sign: Literal[1, -1]
    act_diff: float | None
    steer_kl: float | None
    rank_score: float
    expert: int | None = None                  # CHANGE-05: MoE の層(neuron は通し番号)
    expert_neuron: int | None = None


class LayerCos(_Strict):
    layer: int
    max_abs_cos: float
    top1pct_abs_cos: float
    dir_norm: float


class VerifyExample(_Strict):
    layer: int
    neuron: int
    sign: Literal[1, -1]
    prompt: str
    text: str


class ReverseVerify(_Strict):
    design: str
    runs: list[dict[str, Any]]
    examples: list[VerifyExample]
    baselines: dict[str, str]


class ReverseOut(Envelope):
    name: str
    info: dict[str, Any]
    candidates_path: str
    n_candidates: int
    candidates: list[CandidateModel]
    layers: list[LayerCos]
    steer_lens_layers: list[int]
    verify: ReverseVerify | None
    seconds: float
    by_expert: list[dict[str, int]] | None = None    # CHANGE-05: MoE の層で上位の候補が集まった専門家


# --- Phase 3: sweep / rank ------------------------------------------------------------------------------

class SweepOut(Envelope):
    run: str
    path: str
    meta: str
    layer: int
    design: str
    prompt_set: str
    n_neurons: int
    n_controls: int
    n_units: int
    n_new_records: int
    seconds: float
    workers: dict[str, Any] | None = None       # CHANGE-03 B: --workers で分けたときの部分ごとの結果
    semantics: str | None = None                # CHANGE-05: MoE の層は "direction"(書き込み方向の押し)


class DoseStats(_Strict):
    rel_push: float
    distinct4: float
    capped: float
    logprob: float


class RankedUnit(_Strict):
    kind: Literal["neuron", "random"]
    neuron: int
    layer: int | None
    z: float
    mass: float
    null_pct: float | None
    rank: int | None
    words: list[list[Any]]               # [語, 出た rollout 数, z]
    per_prompt: dict[str, list[str]]
    dose: dict[str, DoseStats]


class RankOut(Envelope):
    run: str
    parts: list[str] = []                       # CHANGE-03 B: 結合した JSONL(本体とワーカーごとの部分)
    units_path: str
    n_units: int
    n_neurons: int
    n_controls: int
    n_baseline_records: int
    word_unit: Literal["tokens", "regex"]
    max_rel: float
    control_mass: NullSummary
    n_above_all_controls: int
    units: list[RankedUnit]
    find: list[RankedUnit]
    seconds: float


# --- Phase 4: chat / neuron ---------------------------------------------------------------------------

class ChatReply(_Strict):
    column: str
    text: str
    finish: Literal["eog", "length"]
    n_gen: int


class ChatTurn(_Strict):
    user: str
    replies: list[ChatReply]


class ChatColumn(_Strict):
    label: str
    recipe: dict[str, Any] | None


class ChatOut(Envelope):
    columns: list[ChatColumn]
    system: str | None
    turns: list[ChatTurn]
    sampling: dict[str, float] | None = None


class DossierSweep(_Strict):
    run: str
    design: str
    rank: int | None
    n_ranked: int
    mass: float
    null_pct: float | None
    words: list[list[Any]]
    per_prompt: dict[str, list[str]]
    dose: dict[str, DoseStats]
    rollouts: list[dict[str, Any]]
    baselines: list[dict[str, Any]]


class DossierSteer(_Strict):
    up: list[str]
    down: list[str]
    kl_up: float
    kl_down: float
    kl_null_p95: float | None
    pct_up: float
    pct_down: float
    dose: float


class DossierLogit(_Strict):
    up: list[str]
    down: list[str]
    peak_z: float


class NeuronOut(Envelope):
    layer: int
    neuron: int
    name: str | None
    down_norm: float
    expert: int | None = None                  # CHANGE-05: MoE の層(neuron は通し番号)
    expert_neuron: int | None = None
    semantics: str | None = None
    address: str | None = None
    recipe_hint: str | None = None
    sweep: DossierSweep | None
    steer_lens: DossierSteer | None
    logit_lens: DossierLogit | None
    directions: dict[str, dict[str, float | int]]
    activation: dict[str, Any] | None           # MoE の層は n_routed・reliable も
    connections: dict[str, Any] | None
    evals: list[dict[str, Any]] = []          # CHANGE-01: このニューロンを含む保存済みの eval(新しい順)
    gain_k: list[float] = []                  # 当てたことのある gain の倍率
    writers: list[dict[str, Any]] = []        # gne writers の分析ごとの寄与と順位
    coverage: dict[str, bool]


class ConnectionsOut(Envelope):                 # gne connections(CHANGE-03 C で CLI を足した)
    path: str
    model_id: str
    k: int
    window: int | None
    rho_scaled: bool
    layers: list[int]
    n_pairs: int
    seconds: float
    formula: str
    down_reads: int
    peak_rss_gb: float


class WritersOut(Envelope):                     # gne writers(書き手の強さ)
    name: str
    path: str
    contrast: str
    contrast_sha: str
    ref_layer: int
    ref_norm: float
    n_tokens: dict[str, int]
    n_convs: dict[str, int]
    replies: dict[str, str]
    post_ffn_norm: bool
    totals: dict[str, float]
    mlp_check_max: float | None
    layers: list[dict[str, Any]]
    top: list[dict[str, Any]]
    bottom: list[dict[str, Any]]
    samples: list[dict[str, Any]]
    seconds: float


class WritersExportOut(Envelope):
    path: str
    name: str
    part: str
    top: int | None
    layers: list[int]
    norms: dict[str, float]
    units: list[str]


class WorkerOut(Envelope):                      # CHANGE-03 B
    name: str
    n_jobs: int


class HeadRow(_Strict):                         # CHANGE-04 C
    layer: int
    head: int
    kind: Literal["attn", "linear"]
    trigger_mass: float | None
    pattern_shift: float | None
    out_shift: float | None
    dir_cos: float | None
    z: float | None
    null_pct: float | None


class ExpertRow(BaseModel):
    model_config = ConfigDict(extra="allow")
    layer: int
    expert: int
    usage_shift: float
    weight_shift: float
    contrib_cos: float
    contrib_norm: float
    rate_a: float
    rate_b: float
    z: float
    null_pct: float


class ExpertsContrastOut(Envelope):
    name: str
    path: str
    contrast: str
    contrast_sha: str
    direction: str | None
    system_b2: str
    layers: list[int]
    n_prompts: int
    max_tokens: int
    floors: dict[str, float]
    null_stats: dict[str, dict[str, Any]]
    top: list[ExpertRow]
    n_experts: int
    candidates: str
    samples: list[dict[str, Any]]
    seconds: float


class ExpertsSweepOut(Envelope):
    run: str
    path: str
    layer: int
    design: str
    prompt_set: str
    ks: list[float]
    n_experts: int
    n_records: int
    top: list[dict[str, Any]]
    reload_s: dict[str, float | None]
    seconds: float


class ExpertOut(Envelope):
    layer: int
    expert: int
    n_expert: int
    n_expert_used: int
    n_ff: int
    routing: str
    router_bias: float | None
    usage: dict[str, Any] | None
    contrasts: list[dict[str, Any]]
    neurons: list[dict[str, Any]]


class HeadsContrastOut(Envelope):
    name: str
    path: str
    contrast: str
    contrast_sha: str
    direction: str | None
    system_b2: str
    layers: list[int]
    n_prompts: int
    n_with_triggers: int
    max_tokens: int
    floors: dict[str, float]
    null_stats: dict[str, dict[str, Any]]
    top: list[HeadRow]
    n_heads: int
    candidates: str
    samples: list[dict[str, Any]]
    seconds: float


class AttentionOut(Envelope):                   # CHANGE-04 B(gne heads attention)
    text: str
    n_tokens: int
    pos: int
    tokens: list[str]
    heads: list[dict[str, Any]]
    pattern: list[list[float]] | None = None


class ListenOut(Envelope):                      # gne heads listen(返事のトークンからの注目と、方向への寄与)
    user: str
    system: str | None
    reply: str
    n_tokens: int
    reply_start: int
    tokens: list[str]
    roles: list[Literal["system", "user", "template", "reply"]]
    heads: list[dict[str, int]]
    attn: list[float]
    attn_by_head: dict[str, list[float]]
    direction: str | None
    contrib: list[float] | None
    contrib_by_head: dict[str, list[float]] | None
    layers: dict[str, dict[str, Any]] | None
    by_role: dict[str, dict[str, float | None]]
    top_attn: list[dict[str, Any]]
    top_contrib: list[dict[str, Any]] | None


class HeadOut(Envelope):                        # CHANGE-04(gne heads show)
    layer: int
    head: int
    kind: Literal["attn", "linear"]
    n_head: int
    n_head_kv: int
    head_dim: int
    writer: str
    block: list[int]
    kv_group: int | None
    name: str | None
    contrasts: list[dict[str, Any]]
    gain_k: list[float]
    logit_lens: dict[str, Any] | None


class HeadsQueryOut(Envelope):                  # CHANGE-04 E(gne heads query、V17)
    layer: int
    head: int
    direction: str
    seek: str
    kv_group: int
    dims: list[int]
    v_gap: float
    v_mean_a: float
    v_mean_b: float
    q_norm: float
    n_trigger_tokens: int
    rows: list[dict[str, Any]]
    monotonic: bool
    gain_a: float
    gain_b: float
    seconds: float


class HeadNameOut(Envelope):
    layer: int
    head: int
    name: str | None


class DirectionScoreOut(Envelope):              # CHANGE-06 B(gne direction score)
    direction: str
    path: str
    contrast: str
    contrast_sha: str
    positions: str
    subset: str
    n: dict[str, int]
    layers: list[dict[str, Any]]
    best_layer: int | None
    base_layer: int | None
    base_recorded: bool
    base_manual: bool | None = None
    values: dict[str, dict[str, list[float]]]
    values_after: dict[str, dict[str, list[float]]] | None
    prompts: list[str]
    prompts_b: list[str] | None = None          # 側ごとに質問が違う対照集合の B 側
    intervention: dict[str, Any] | None
    seconds: float


class CausalRow(_Strict):                       # CHANGE-06 C の 1 ヘッド
    layer: int
    head: int
    kind: Literal["attn", "linear"]
    mode: Literal["ablate", "marginal"]
    score_drop: dict[str, float]
    causal: float
    causal_max: float
    b_side_delta: float | None = None
    drift_par_b: float | None = None
    drift_orth_b: float | None = None
    contrib_by_layer: dict[str, float]
    contrib: float | None
    ratio: float | None
    delta_s: float | None = None
    eta: float | None = None
    eta_by_layer: dict[str, float] | None = None
    in_base: bool | None = None
    excluded: bool | None = None
    seconds: float


class HeadsCausalOut(Envelope):                 # CHANGE-06 C(gne heads causal)
    name: str
    path: str
    summary_path: str
    direction: str
    contrast: str
    contrast_sha: str
    mode: Literal["ablate", "marginal"]
    score_layers: list[int]
    primary: int
    positions: str
    subset: str
    n_a: int
    n_b: int
    direction_layer: str
    base_scores: dict[str, dict[str, float]]
    updated: str
    max_examples: int | None = None
    drift_max: float | None = None
    layers: list[dict[str, Any]] | None
    layer_rows: list[dict[str, Any]] | None = None
    rows: list[CausalRow]
    n_rows: int
    hard: dict[str, Any] | None = None
    base_units: list[str] | None = None
    spearman_ablate_marginal: float | None = None
    n_compared: int | None = None
    seconds: float


class BehaveOut(Envelope):                      # gne behave(振る舞いを数える)
    model_config = ConfigDict(extra="allow")
    name: str
    path: str
    n_prompts: int
    runs: list[dict]


class HeadsWritersOut(Envelope):                # gne heads writers(方向の差を書き手で分ける)
    model_config = ConfigDict(extra="allow")
    name: str
    path: str
    direction: str
    ref_layer: int
    gap: float
    sum_error: float


class HeadsAllocateOut(Envelope):               # CHANGE-06 D(gne heads allocate)
    name: str
    path: str
    direction: str
    contrast_sha: str
    score_layer: int
    score_layers: list[int]
    max_examples: int | None = None
    drift_max: float | None = None
    target: float
    positions: str
    subset: str
    direction_layer: str
    method: str = "proportional"
    proportional_to: str
    from_zero: bool
    delta_s: float
    s_max: float | None
    remain_threshold: float
    n_a: int
    n_b: int
    base_a_mean: float
    base_b_mean: float
    initial: dict[str, Any] | None
    current: list[str]
    hard: dict[str, Any]
    eta: list[dict[str, Any]]
    table: list[dict[str, Any]]
    recommended: int | None
    seconds: float


class HeadsPipelineOut(Envelope):               # gne heads pipeline(方向 → 基準層 → 因果 → 配分)
    name: str
    path: str
    contrast: str
    n_prompts: int
    n_prompts_b: int | None = None              # 側ごとに質問が違うときの B 側の数
    sides: dict[str, Any]
    settings: dict[str, Any]
    direction: dict[str, Any]
    score: dict[str, Any]
    causal: dict[str, Any]
    allocate: dict[str, Any]
    target: float
    target_from: str
    recommended: dict[str, Any] | None
    seconds: float
    updated: str | None = None
    reused: list[str] | None = None              # 使い回した段(direction・causal、resume)


class FieldtestOut(Envelope):                   # gne fieldtest(本番と同じ動かし方で、ファイルどうしを比べる)
    name: str
    path: str
    replies_path: str
    target: dict[str, Any]
    reference: dict[str, Any] | None
    settings: dict[str, Any]
    runs: list[dict[str, Any]]                  # ファイルごとの long・tools・prompts・ppl
    compare: list[dict[str, Any]] | None        # 指標ごとの参照・対象・差・worse
    same_level: bool | None                     # worse の行が 1 つも無ければ True
    worse: list[str]
    updated: str
    seconds: float


class HeadsRefineOut(Envelope):                 # gne heads refine(残りの質問で追加のヘッドを探す)
    name: str
    path: str
    base: str
    direction: str
    n_heads: int
    base_units: list[str]
    base_recipe: str
    score_layer: int
    threshold: float
    remaining: dict[str, Any]                   # n(残り)・n_a・used(方向に使った数)・padded・questions・scores
    pipeline: dict[str, Any]                    # 一括の結果のパス・方向・推す組・使い回した段・組の表
    combined: dict[str, Any] | None             # 代表の合わせたレシピと、元の方向での残り・A の平均・B 側のずれ(前 → 後)
    combined_rows: list[dict[str, Any]] = []    # 比較表のどの組も、元の組と合わせたもの(同じ項目)
    updated: str
    seconds: float


class CandidateRow(BaseModel):
    name: str                                   # base(無介入)か n{ヘッド数}x{倍率}
    n_heads: int
    scale: float
    units: list[str]
    recipe: str | None
    recommended: bool = False
    rate_a: float | None = None                 # validate.match に合う返事の割合
    rate_b: float | None = None
    d4_a: float                                 # 4-gram の重複の少なさ(1 = 繰り返し無し)
    d4_b: float
    same_as_base_a: float                       # 無介入と同じ返事の割合(同じ種なので、変わらなければ一致する)
    same_as_base_b: float
    judge_sides: str                            # 採点のファイルに入れた側
    goal: str | None                            # remove(正の倍率、A 側で消す)/ add(負の倍率、B 側で強める)


class HeadsCandidatesOut(Envelope):             # gne heads candidates(強さの候補を作り、名前を伏せた採点ファイルを書く)
    name: str
    path: str
    alloc: str
    direction: str
    contrast: str
    held_out: dict[str, bool]
    with_recipe: str | None
    when: str
    sampling: dict[str, Any] | None
    max_tokens: int
    scales: list[float]
    prompts: dict[str, list[str]]
    match: str | None
    target: str
    candidates: list[CandidateRow]
    replies: dict[str, list[str]]               # "候補|側" → 返事
    judge: dict[str, Any]
    scores: dict[str, dict[str, dict[str, float]]] | None   # 候補 → 側 → n・P・C・min_PC・P_ge3(集計の後)
    scores_missing: list[int] | None = None
    best: dict[str, dict[str, Any]] | None = None   # remove / add → いちばん良い候補(value)と無介入の同じ点(base)
    updated: str
    seconds: float


class HeadprojCheckOut(Envelope):               # CHANGE-06 V24・V25(gne headproj check)
    layer: int
    head: int
    block: list[int]
    out_tensor: str
    direction: str
    direction_from: int
    dir_cos_head: float | None
    rows: list[dict[str, Any]]
    gain: dict[str, Any]
    v24: bool
    v25: bool
    ok: bool


# --- CHANGE-07: 鍵つきの書き込み(keyed write)------------------------------------------------------------

class DirectionKeyLayer(_Strict):               # gne direction key の 1 層
    layer: int
    kn: float                                   # ‖k‖/‖μ‖(k = (I − QQᵀ)μ)
    gain: float                                 # 増幅率 ‖μ‖/‖k‖
    q_rank: int
    q_rows: int                                 # Q を作った行の数
    weak: bool                                  # kn < 0.05
    kappa_mu: float                             # κᵀμ(1 になる)
    build_auc: float | None                     # 作成に使った例での AUC(甘い値)
    rms_a: float | None = None                  # post-norm の層: 発火位置での書き込みの rms の平均
    # MoE(書き込み先 shared・expert:e)の層だけ: 層ごとに行が絞られる(専門家 e が選ばれた位置だけ)ので数を出す
    gate_a: float | None = None                 # 書き込みたい位置の行でのゲートの平均(shared のゲート・専門家のルーターの重み)。B を割る
    rows_a: int | None = None                   # 書き込みたい位置のうち、この層で書き込み先の行があった数
    rows_b: int | None = None
    examples_a: int | None = None               # 書き込みたい側の例のうち、この層で行があった例の数
    examples_b: int | None = None
    rows_a_start: int | None = None             # 専門家の鍵(mode same): 書き込みたい位置の行のうち返事の冒頭にあるもの
    start_share: float | None = None            # 専門家の鍵(mode same): μ に占める返事の冒頭の行の重み


class DirectionKeyOut(Envelope):                # CHANGE-07 B(gne direction key)
    name: str
    path: str
    target: str                                 # ffn / attn / shared / expert:e
    mode: Literal["same", "system"]             # same = 推論時と同じ文脈の返事(before:<re>)、system = system・recipe の文脈
    positions: str
    positions_b: str
    skip: int
    max_tokens: int
    subset: str
    q_mode: Literal["means", "examples", "samples"]
    contrast: str
    contrast_sha: str
    layers: list[DirectionKeyLayer]
    dropped: list[dict[str, Any]]               # 鍵にならずに外した層 {layer, error}
    counts: dict[str, int]
    n_pseudo: int
    replies: dict[str, str | None]              # 群 → given / generated(返事を使わない位置なら None)
    dont_touch: list[dict[str, Any]]
    iv: dict[str, Any] | None                   # 当てた介入(レシピ)
    matched: dict[str, int]                     # 書き込みたい位置があった例の数 {A, of}
    reply_start: dict[str, Any] | None = None   # mode same: 書き込みたい位置のうち返事の冒頭(最初の k トークンを決める位置)にあるもの
                                                # {k, share(μ に占める重み), rows, rows_a, layer, dont_touch_starts, dont_touch_start}
    dont_touch_start: dict[str, Any] | None = None  # dont_touch の質問の返事の冒頭を Q に入れたか {on, positions, groups: {群: 行}, rows}
    samples: list[str]
    notes: list[str]
    sha: str
    seconds: float


class KeyCheckLayer(_Strict):                   # gne direction key-check の 1 層
    layer: int
    auc: float | None                           # 例単位(例ごとの ρ の平均)
    auc_token: float | None                     # トークン単位
    selectivity: float | None                   # 例ごとの ρ の平均の差 A − B
    pseudo_auc: dict[str, Any]                  # 比較用の鍵の AUC {n, p50, p95, values}
    pseudo_auc_token: dict[str, Any]
    pseudo_selectivity: dict[str, Any]
    beats_pseudo: bool | None                   # key-check の選択性 > 比較用の鍵の p95(かつ AUC > 0.5)
    beats_pseudo_auc: bool | None
    b_split_auc: float | None                   # 書き込みたくない位置を 2 つに分け、その差から作った鍵の AUC(参考値)
    rho: dict[str, Any]                         # 群ごとの ρ の要約
    dont_touch_median: dict[str, float | None]
    values: dict[str, list[float]]              # 群 → 例ごとの ρ
    hist: dict[str, Any] | None                 # UI のヒストグラム {edges, 群: 数}
    kn: float | None = None
    gain: float | None = None
    start_rho: dict[str, Any] | None = None     # 書き込みたくない位置・dont_touch の、返事の冒頭での ρ(冒頭の位置ごとの median・max・n)
                                                # {"B": {...}, "dont_touch": {名前: {...}}}


class DirectionKeyCheckOut(Envelope):           # CHANGE-07 C(gne direction key-check)
    key: str
    sha: str
    target: str
    contrast: str | None                        # None = 鍵に保存した作成時の例で測った
    contrast_sha: str | None
    subset: str | None
    mode: str
    positions: str
    positions_b: str | None
    skip: int
    max_tokens: int
    held_out: bool
    overlap: dict[str, Any]
    built_on: dict[str, Any]
    layers: list[KeyCheckLayer]
    best_layer: int | None
    all_beat_pseudo: bool | None
    n_pseudo: int
    counts: dict[str, Any]
    replies: dict[str, str | None] | None
    iv: dict[str, Any] | None
    notes: list[str]
    seconds: float


class DirectionKeysOut(Envelope):               # CHANGE-07(gne direction keys)
    keys: list[dict[str, Any]]                  # KeySet.summary(読めないファイルは {name, error})


class KeyVerifyRow(_Strict):                    # gne direction key-verify の α 1 つ
    alpha: float
    adapter_scale: float                        # 実行時の LoRA のアダプタの倍率
    n_tokens: int                               # 測った行(expert は専門家 e が選ばれた (トークン, 枠))
    n_big: int                                  # |κᵀx| の上位(rel_err を測った行)
    kx_threshold: float
    kx_max: float | None
    rel_err_median: float | None                # ‖Δ − α(κᵀx)d‖ / ‖α(κᵀx)d‖ の中央値(合格 < 3%)
    rel_err_p95: float | None
    cos_median: float | None
    leak_b: float | None                        # 書き込みたくない側の例の平均の Δ の大きさ ÷ (|α|‖d‖)(合格 < 0.1)
    leak_b_source: Literal["texts_b", "key_examples"] | None
    leak_b_token_median: float | None
    ok: bool
    # expert:e だけ
    other_max: float | None = None              # ほかの専門家の枠の ‖Δ‖ ÷ (|α|‖d‖) の最大(0 のはず、合格 < 1e-3)
    selection_changed: int | None = None        # LoRA を当てて選ばれる専門家が変わったトークンの数(0 のはず)


class DirectionKeyVerifyOut(Envelope):          # CHANGE-07 V30(gne direction key-verify)
    layer: int
    kind: str
    expert: int | None
    tensor: str
    key: str
    direction: str
    direction_layer: str
    in_name: str
    out_name: str
    d_norm: float
    direction_norm: float | None
    post_norm: bool | None
    kn: float
    leak_b_key_examples: float | None
    rows: list[KeyVerifyRow]
    v30: bool
    judged_leak: bool
    # MoE(shared・expert)だけ: B を割った補正(LoRA の差は補正の前で測る)
    gate: float | None = None                   # 鍵の gate_a(shared のゲート・専門家のルーターの重みの平均)
    res_scale: float | None = None              # 枝の出力の倍率(granite 系の residual_scale。1 なら出さない)
    note: str | None = None                     # expert: 専門家 e が texts で一度も選ばれなかったとき


class ExpertUsageRow(_Strict):                  # gne direction key-experts の専門家 1 つ
    expert: int
    rate_a: float                               # 書き込みたい位置で選ばれた割合(例ごとの割合の平均)
    rate_b: float
    diff: float                                 # rate_a − rate_b
    lift: float | None                          # rate_a / rate_b(rate_b が 0 なら None)
    weight_a: float | None                      # 書き込みたい位置で選ばれたときのルーターの重みの平均(鍵の gate_a の見込み)
    rows_a: float                               # 書き込みたい位置で選ばれる行の数の見込み(rate_a × 書き込みたい位置の数)
    rate_dt: dict[str, float]                   # dont_touch の群ごとの選ばれた割合


class ExpertUsageLayer(_Strict):
    layer: int
    n_expert: int
    n_used: int
    positions_a: int                            # 書き込みたい位置の数
    positions_b: int
    base_rate: float                            # n_used / n_expert(どれも同じだけ選ばれるときの割合)
    top: list[ExpertUsageRow]                   # rate_a の大きい順


class DirectionKeyExpertsOut(Envelope):         # CHANGE-07 MoE(gne direction key-experts、expert:e の鍵の専門家を選ぶ材料)
    contrast: str
    mode: Literal["same", "system"]
    positions: str
    positions_b: str
    subset: str
    layers: list[ExpertUsageLayer]
    replies: dict[str, str | None]
    iv: dict[str, Any] | None
    notes: list[str]
    seconds: float


COMMAND_SCHEMAS: dict[str, type[Envelope]] = {
    "chat": ChatOut, "neuron": NeuronOut,
    "sweep": SweepOut, "rank": RankOut,
    "lens logit": LensLogitOut, "lens steer": LensSteerOut, "reverse": ReverseOut,
    "info": InfoOut, "probe": ProbeOut, "verify": VerifyOut, "bench": BenchOut,
    "stats resid": StatsResidOut, "stats act": StatsActOut, "steer": SteerOut,
    "export-cvec": ExportOut, "import-cvec": ImportOut, "bake": BakeOut, "gain": GainOut,
    "eval": EvalOut, "direction": DirectionOut, "ablate": AblateOut,
    "heads contrast": HeadsContrastOut, "heads attention": AttentionOut, "heads listen": ListenOut, "heads show": HeadOut, "heads name": HeadNameOut,
    "experts contrast": ExpertsContrastOut, "experts show": ExpertOut, "experts sweep": ExpertsSweepOut,
    "heads query": HeadsQueryOut,
    "connections": ConnectionsOut, "worker": WorkerOut, "writers": WritersOut, "writers-export": WritersExportOut,
    "direction score": DirectionScoreOut, "heads causal": HeadsCausalOut, "heads writers": HeadsWritersOut, "behave": BehaveOut, "heads allocate": HeadsAllocateOut,
    "headproj export": GainOut, "headproj check": HeadprojCheckOut, "heads pipeline": HeadsPipelineOut,
    "heads candidates": HeadsCandidatesOut,
    "heads refine": HeadsRefineOut, "fieldtest": FieldtestOut,
    "direction key": DirectionKeyOut, "direction key-check": DirectionKeyCheckOut, "direction keys": DirectionKeysOut,
    "direction key-verify": DirectionKeyVerifyOut, "direction key-experts": DirectionKeyExpertsOut,
}
