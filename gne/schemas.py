"""`--json` 出力のスキーマ。設計書 §6.13 (a)、V10。

規約:
- stdout には JSON だけを出す。最後の行が結果のオブジェクト。長い処理はその前に進捗を JSONL で 1 行ずつ出す。
- すべての出力に schema_version, command, model_id, llama_cpp_version, seed, generated_at を含める。
- 失敗は非ゼロ終了 + {"error": {"code", "message", "hint"}}(エンベロープ付き)。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

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
    kind: Literal["neuron", "random", "direction"]
    seed: int | None
    dose: float
    norm: float
    rel_push: float | None
    rho: float | None
    add: float | None
    down_norm: float | None
    mode: Literal["add", "gain"] = "add"     # §6.16
    k: float | None = None
    direction: str | None = None              # §6.17 方向のユニット
    diff_rel: float | None = None             # ‖平均差‖ / ρ_L


class RecipeModel(_Strict):
    name: str | None
    model_id: str | None
    dose_unit: Literal["rel", "absmax", "raw"]
    when: str = "all"                 # §6.15
    units: list[dict[str, Any]]
    ablate: dict[str, Any] | None = None      # §6.17


class Generation(_Strict):
    text: str
    finish: Literal["eog", "length"]
    n_gen: int
    token_ids: list[int] | None = None
    when: dict[str, Any] | None = None   # 位置別ステアリングの記録(set_cvec の回数、区間など)


class SteerOut(Envelope):
    messages: list[dict[str, str]]
    recipe: RecipeModel
    units: list[UnitModel]
    layer_norms: dict[str, float]
    baseline: Generation | None
    steered: Generation
    differs: bool | None
    when: str = "all"


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


class GainOut(Envelope):                        # §6.16 gne gain
    kind: Literal["lora", "bake"]
    recipe: RecipeModel
    path: str
    bytes: int
    layers: dict[str, list[list[float]]] | None = None     # lora: 層 → [[ニューロン, k]]
    rank: dict[str, int] | None = None
    modified: list[dict[str, str]] | None = None           # bake: 書き換えたテンソルと型
    n_tensors: int | None = None


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


class CandidateModel(_Strict):
    layer: int
    neuron: int
    cos: float
    sign: Literal[1, -1]
    act_diff: float | None
    steer_kl: float | None
    rank_score: float


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
    sweep: DossierSweep | None
    steer_lens: DossierSteer | None
    logit_lens: DossierLogit | None
    directions: dict[str, dict[str, float | int]]
    activation: dict[str, float] | None
    connections: dict[str, Any] | None
    evals: list[dict[str, Any]] = []          # CHANGE-01: このニューロンを含む保存済みの eval(新しい順)
    gain_k: list[float] = []                  # 当てたことのある gain の倍率
    coverage: dict[str, bool]


COMMAND_SCHEMAS: dict[str, type[Envelope]] = {
    "chat": ChatOut, "neuron": NeuronOut,
    "sweep": SweepOut, "rank": RankOut,
    "lens logit": LensLogitOut, "lens steer": LensSteerOut, "reverse": ReverseOut,
    "info": InfoOut, "probe": ProbeOut, "verify": VerifyOut, "bench": BenchOut,
    "stats resid": StatsResidOut, "stats act": StatsActOut, "steer": SteerOut,
    "export-cvec": ExportOut, "import-cvec": ImportOut, "bake": BakeOut, "gain": GainOut,
    "eval": EvalOut, "direction": DirectionOut, "ablate": AblateOut,
}
