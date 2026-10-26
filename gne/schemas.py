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
    kind: Literal["neuron", "random"]
    seed: int | None
    dose: float
    norm: float
    rel_push: float | None
    rho: float | None
    add: float | None
    down_norm: float | None


class RecipeModel(_Strict):
    name: str | None
    model_id: str | None
    dose_unit: Literal["rel", "absmax", "raw"]
    units: list[dict[str, Any]]


class Generation(_Strict):
    text: str
    finish: Literal["eog", "length"]
    n_gen: int
    token_ids: list[int] | None = None


class SteerOut(Envelope):
    messages: list[dict[str, str]]
    recipe: RecipeModel
    units: list[UnitModel]
    layer_norms: dict[str, float]
    baseline: Generation | None
    steered: Generation
    differs: bool | None


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


COMMAND_SCHEMAS: dict[str, type[Envelope]] = {
    "sweep": SweepOut, "rank": RankOut,
    "lens logit": LensLogitOut, "lens steer": LensSteerOut, "reverse": ReverseOut,
    "info": InfoOut, "probe": ProbeOut, "verify": VerifyOut, "bench": BenchOut,
    "stats resid": StatsResidOut, "stats act": StatsActOut, "steer": SteerOut,
    "export-cvec": ExportOut, "import-cvec": ImportOut, "bake": BakeOut,
}
