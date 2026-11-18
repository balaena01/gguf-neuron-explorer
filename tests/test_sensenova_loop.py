"""同じ層を 2 回実行するモデル(arch sensenova_loop、SenseNova-Loop-9B)への対応。docs/sensenova_loop.md。

モデルが要らないもの(層の対応・LoRA の名前の写し替え・射影の合成・捕捉の名前)は、いつも流す。
実物を使うものは GNE_TEST_SENSENOVA に GGUF のパスを入れたときだけ流す(公式の llama.cpp は読めないので、この arch に対応した
llama.cpp の llama-cpp-python が要る。読めなければ飛ばす)。
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gne.loop import LoopPlan, plan_from_kinds, stored_tensor_name, to_stored

# 実物と同じ並び: 層 0〜3 は流れ 1 本、4 で 2 本に入り 27 で出る、28 で入り直して 51 で出る、52〜55 は 1 本
KINDS = [0] * 4 + [1] + [4] * 22 + [2] + [1] + [4] * 22 + [2] + [0] * 4


def test_plan_from_kinds():
    p = plan_from_kinds(KINDS)
    assert (p.lo1, p.hi1, p.lo2, p.shift, p.hi2, p.n_layer) == (4, 27, 28, 24, 52, 56)
    assert [p.visit(L) for L in (3, 4, 27, 28, 51, 52)] == [0, 1, 1, 2, 2, 0]
    assert p.stored(30) == 6 and p.stored(6) == 6 and p.stored(53) == 53
    assert p.twin(6) == 30 and p.twin(30) == 6 and p.twin(2) is None and p.twin(54) is None
    assert p.visits(30) == [6, 30] and p.visits(3) == [3]
    assert p.inner(4) and p.inner(30) and not p.inner(27) and not p.inner(51) and not p.inner(3)
    assert p.label(30) == "L30 = 層 6 の 2 回目" and p.label(2) == "L2"
    assert p.stored_name("blk.31.attn_output.weight") == "blk.7.attn_output.weight"
    assert p.stored_name("blk.7.ffn_up.weight") == "blk.7.ffn_up.weight"
    assert p.stored_name("token_embd.weight") == "token_embd.weight"
    assert plan_from_kinds([0, 0, 1, 2]) is None              # 入口が 1 つ = 繰り返しではない
    assert plan_from_kinds(None) is None


def _w(per_visit=False):
    return SimpleNamespace(loop=plan_from_kinds(KINDS), per_visit_names=per_visit)


def test_to_stored_dedupes_and_sums():
    A = np.arange(8, dtype=np.float32).reshape(1, 8)
    B = np.ones((4, 1), np.float32)
    same = to_stored(_w(), {"blk.31.attn_output.weight": (A, B), "blk.7.attn_output.weight": (A.copy(), B.copy())})
    assert list(same) == ["blk.31.attn_output.weight".replace("31", "7")] and same["blk.7.attn_output.weight"][0].shape == (1, 8)
    msgs = []
    diff = to_stored(_w(), {"blk.31.ffn_up.weight": (A, B), "blk.7.ffn_up.weight": (2 * A, B)}, warn=msgs.append)
    a, b = diff["blk.7.ffn_up.weight"]
    assert a.shape == (2, 8) and b.shape == (4, 2) and msgs
    np.testing.assert_allclose(b @ a, B @ A + B @ (2 * A))
    # 回ごとの名前を持つ llama.cpp なら写し替えない
    keep = to_stored(_w(per_visit=True), {"blk.31.ffn_up.weight": (A, B)})
    assert list(keep) == ["blk.31.ffn_up.weight"]
    assert stored_tensor_name(_w(), "blk.40.attn_q.weight") == "blk.16.attn_q.weight"
    assert stored_tensor_name(SimpleNamespace(loop=None), "blk.40.attn_q.weight") == "blk.40.attn_q.weight"


class _FakeWeights:
    """ProjBuilder が使う分だけ: 層ごとの書き込み行列(2 回目の層は 1 回目と同じ)と、ヘッドの列の範囲。"""

    def __init__(self, per_visit=False, n_embd=12, n_head=3, head_dim=4, seed=0):
        self.loop = plan_from_kinds(KINDS)
        self.per_visit_names = per_visit
        self.hd, self.nh = head_dim, n_head
        rng = np.random.default_rng(seed)
        self._W = {L: rng.standard_normal((n_embd, n_head * head_dim)).astype(np.float32) for L in range(56)}

    def head_writer_matrix(self, L):
        return self._W[self.loop.stored(L)]

    def head_block(self, L, h):
        return f"blk.{L}.attn_output.weight", h * self.hd, (h + 1) * self.hd


class _FakeDS:
    def __init__(self, raw):
        self.raw = raw
        self.name = "d"


def test_projection_composed_per_stored_head():
    from gne.headproj import ProjBuilder, ProjHead
    w = _FakeWeights()
    rng = np.random.default_rng(1)
    raw = {L: rng.standard_normal(12) for L in range(56)}
    b = ProjBuilder(w, {"d": _FakeDS(raw)})
    s1, s2, h = 1.5, 3.0, 1
    tens = b.tensors([ProjHead(7, h, s1, "d"), ProjHead(31, h, s2, "d")])
    assert list(tens) == ["blk.7.attn_output.weight"]        # 2 回目の層のユニットも、重みを持つ層の 1 本に
    A, B = tens["blk.7.attn_output.weight"]
    W = w.head_writer_matrix(7).astype(np.float64)
    r1 = raw[7] / np.linalg.norm(raw[7])
    r2 = raw[31] / np.linalg.norm(raw[31])
    I = np.eye(12)
    want = (I - s2 * np.outer(r2, r2)) @ (I - s1 * np.outer(r1, r1)) @ W
    got = W + B.astype(np.float64) @ A.astype(np.float64)
    lo, hi = h * 4, (h + 1) * 4
    np.testing.assert_allclose(got[:, lo:hi], want[:, lo:hi], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(got[:, :lo], W[:, :lo], atol=1e-6)            # ほかのヘッドは変わらない
    # 1 回だけなら今までと同じ式(s r rᵀW)
    one = b.tensors([ProjHead(7, h, s1, "d")])["blk.7.attn_output.weight"]
    got1 = W + one[1].astype(np.float64) @ one[0].astype(np.float64)
    np.testing.assert_allclose(got1[:, lo:hi], ((I - s1 * np.outer(r1, r1)) @ W)[:, lo:hi], rtol=1e-5, atol=1e-5)


def test_projection_per_visit_names_kept_apart():
    from gne.headproj import ProjBuilder, ProjHead
    w = _FakeWeights(per_visit=True)
    raw = {L: np.eye(12)[L % 12] + 0.1 for L in range(56)}
    tens = ProjBuilder(w, {"d": _FakeDS(raw)}).tensors([ProjHead(7, 0, 1.0, "d"), ProjHead(31, 0, 1.0, "d")])
    assert sorted(tens) == ["blk.31.attn_output.weight", "blk.7.attn_output.weight"]


def test_ablation_per_layer_composed_for_twins():
    from gne.ablate import _twin_projections
    rng = np.random.default_rng(3)
    W = rng.standard_normal((6, 5))
    r1, r2 = rng.standard_normal(6), rng.standard_normal(6)
    r1, r2 = r1 / np.linalg.norm(r1), r2 / np.linalg.norm(r2)
    rs = {"blk.7.attn_output.weight": r1, "blk.31.attn_output.weight": r2, "blk.2.ffn_down.weight": r1}
    rows = {"blk.7.attn_output.weight": r1 @ W, "blk.31.attn_output.weight": r2 @ W, "blk.2.ffn_down.weight": r1 @ W}
    rs2, rows2, twin = _twin_projections(_w(), rs, rows, 1.0)
    assert set(twin) == {"blk.7.attn_output.weight"} and set(rows2) == {"blk.2.ffn_down.weight"}
    A, B = twin["blk.7.attn_output.weight"]
    I = np.eye(6)
    np.testing.assert_allclose(W + B @ A, (I - np.outer(r2, r2)) @ (I - np.outer(r1, r1)) @ W, atol=1e-5)
    assert _twin_projections(_w(per_visit=True), rs, rows, 1.0)[2] == {}


def test_capture_names_and_shapes():
    from gne.capture import Capture
    c = Capture(lambda n: n in {"l_out-30", "attn_residual-30", "mhc_ffn_gates-30", "l_out_streams-30"})
    c.loop, c.loop_d, c.loop_full = plan_from_kinds(KINDS), 4, frozenset({31})
    assert c._canon("mhc_ffn_out-30") == "ffn_out-30"
    assert c._canon("mhc_attn_out-31") == "attn_output-31" and c._canon("mhc_attn_out-30") == "linear_attn_out-30"
    assert c._canon("mhc_ffn_norm-30") == "attn_post_norm-30" and c._canon("mhc_attn_norm-30") == "attn_norm-30"
    assert c._canon("mhc_ffn_merged-30") == "l_out_pre-30" and c._canon("l_out-30") == "l_out-30"
    assert c._want("l_out_pre-30")                         # 2 本の中の層の l_out-30 のために取る
    assert not c._want("l_out_pre-3")
    merged = np.arange(16, dtype=np.float32).reshape(2, 8)              # 2 位置 × [流れ a (4) | 流れ b (4)]
    c._store("l_out_pre-30", merged)
    packed = np.arange(20, dtype=np.float32).reshape(2, 10)             # 2 位置 × [入力 (4) | 重み 6 つ]
    c._store("attn_residual-30", packed)
    out = c.take()
    np.testing.assert_allclose(out["l_out-30"], merged.reshape(2, 2, 4).mean(axis=1))
    np.testing.assert_allclose(out["l_out_streams-30"], merged)
    np.testing.assert_allclose(out["attn_residual-30"], packed[:, :4])
    np.testing.assert_allclose(out["mhc_ffn_gates-30"], packed[:, 4:])
    assert "l_out_pre-30" not in out                       # 欲しいと言っていない名前は返さない


def test_head_layers_skip_second_visit():
    from gne.causal import _visits, head_layers
    li = [SimpleNamespace(index=L, head_dim=4, head_writer="ssm_out") for L in range(56)]
    w = SimpleNamespace(arch=SimpleNamespace(layers=li), loop=plan_from_kinds(KINDS), per_visit_names=False)
    sess = SimpleNamespace(weights=w)
    Ls = head_layers(sess, 40)
    assert 7 in Ls and 31 not in Ls and max(Ls) == 27
    assert _visits(sess, 7) == [7, 31] and _visits(sess, 2) == [2]
    w.per_visit_names = True
    assert 31 in head_layers(sess, 40) and _visits(sess, 7) == [7]


def test_nwrite_layers():
    from gne.nwrite import tensor_layer, twin_layer
    w = _w()
    assert tensor_layer(w, 36) == 12 and twin_layer(w, 36) == 12 and twin_layer(w, 12) == 36 and twin_layer(w, 2) is None
    assert tensor_layer(_w(per_visit=True), 36) == 36 and twin_layer(_w(per_visit=True), 36) is None
    plain = SimpleNamespace(loop=None)
    assert tensor_layer(plain, 36) == 36 and twin_layer(plain, 36) is None


def test_default_neurons_no_overlap():
    from gne.fish import default_neurons, needed
    n = needed(3, "cap")
    rows = [{"neuron": j} for j in range(2 * n)]
    usage = {"layers": [{"layer": 12, "stored_layer": 12, "rows": rows}, {"layer": 36, "stored_layer": 12, "rows": rows}]}
    got = default_neurons(usage, 3, "cap")
    assert not set(got[12]) & set(got[36]) and len(got[36]) == n


# --- 実物 -------------------------------------------------------------------------------------------------------

def _proj(L, h, k, d=None):
    from gne.steer import Unit
    return Unit(L, None, 0.0, "head", mode="proj", k=float(k), head=h, direction=d)


def test_recipe_loop_edits_record(tmp_path):
    """当て方の記録: shared と記録なしは書かない(今までと同じ文字・同じハッシュ)、per_visit だけ書く。古いレシピは shared で読む。
    メモリの上だけのレシピ(記録なし)は、保存するときのセッションの当て方を記録する。"""
    import yaml

    from gne.loop import session_mode, set_session_mode
    from gne.steer import Recipe
    prev = session_mode()
    try:
        set_session_mode(None)
        r = Recipe([_proj(8, 3, 2.0, "d")], "m", name="r")
        r.save(tmp_path / "a.yaml")
        assert "loop_edits" not in yaml.safe_load((tmp_path / "a.yaml").read_text())
        assert Recipe.load(tmp_path / "a.yaml").loop_edits == "shared"
        assert "loop_edits" not in r.to_dict()
        set_session_mode("per_visit")
        r.save(tmp_path / "b.yaml")
        assert yaml.safe_load((tmp_path / "b.yaml").read_text())["loop_edits"] == "per_visit"
        b = Recipe.load(tmp_path / "b.yaml")
        assert b.loop_edits == "per_visit" and b.to_dict()["loop_edits"] == "per_visit"
        assert b.like(b.units).loop_edits == "per_visit"
        Recipe.load(tmp_path / "a.yaml").save(tmp_path / "c.yaml")      # 記録が shared のレシピは、per_visit のセッションでも shared
        assert "loop_edits" not in yaml.safe_load((tmp_path / "c.yaml").read_text())
        v = Recipe([_proj(8, 3, 2.0, "d")], "m", name="v", loop_edits="shared", loop_visits={"8:H3": [1]})
        v.save(tmp_path / "v.yaml")
        assert Recipe.load(tmp_path / "v.yaml").loop_visits == {"8:H3": [1]}
    finally:
        set_session_mode(prev)


def test_lora_file_records_per_visit(tmp_path):
    from gne.ablate import write_lora_tensors
    from gne.loop import lora_loop_edits
    meta = SimpleNamespace(arch="sensenova_loop", model_id="m", name="m")
    A, B = np.ones((1, 8), np.float32), np.ones((4, 1), np.float32)
    for pv, want in ((False, None), (True, "per_visit")):
        w = SimpleNamespace(meta=meta, loop=plan_from_kinds(KINDS), per_visit_names=pv)
        p = tmp_path / f"x{pv}.lora.gguf"
        write_lora_tensors(w, {"blk.31.ffn_up.weight": (A, B)}, p, name="t")
        assert lora_loop_edits(p) == want


def test_twin_effect_matches_matrices():
    """同じヘッドの 2 つの射影を続けて掛けたときの実効の強さ(式)が、行列で掛けた値と合う。"""
    from gne.loop import twin_effect
    rng = np.random.default_rng(0)
    r1 = rng.normal(size=6)
    r1 /= np.linalg.norm(r1)
    r2 = r1 + 0.7 * rng.normal(size=6)
    r2 /= np.linalg.norm(r2)
    s1, s2 = 3.0, 1.5
    T = (np.eye(6) - s2 * np.outer(r2, r2)) @ (np.eye(6) - s1 * np.outer(r1, r1))
    e1, e2 = twin_effect(s1, s2, float(r1 @ r2))
    assert abs((1 - r1 @ T @ r1) - e1) < 1e-9 and abs((1 - r2 @ T @ r2) - e2) < 1e-9
    assert abs(twin_effect(3.0, 2.0, 1.0)[0] - (3 + 2 - 6)) < 1e-12        # 同じ方向: s1 + s2 − s1·s2(反転 2 回で強め合う)


def test_contrast_side_stack(tmp_path):
    from gne.reverse import ContrastSet, side_interventions
    from gne.steer import GneError, Recipe
    for n in ("r1", "r2"):
        Recipe([_proj(8, 3, 2.0, "d")], "m", name=n).save(tmp_path / f"{n}.yaml")
    (tmp_path / "c.yaml").write_text(
        "name: c\nuser_prompts: [a, b, c, d]\nrecipe_a: {stack: [" + str(tmp_path / "r1.yaml") + ", " + str(tmp_path / "r2.yaml")
        + "]}\n", encoding="utf-8")
    cs = ContrastSet.load(tmp_path / "c.yaml")
    sess = SimpleNamespace(recipe=lambda path=None, **k: Recipe.load(path), recipe_path=lambda x: Path(x),
                           patches_for=lambda r: None, vectors=lambda r: ({}, None),
                           lora_for=lambda r: [(r.name, 1.0)])
    assert [r.name for r in cs.recipes(sess, "A")] == ["r1", "r2"] and cs.recipes(sess, "B") == []
    with pytest.raises(GneError):
        cs.recipe(sess, "A")
    iv = side_interventions(sess, cs)
    assert iv["A"] == (None, [("r1", 1.0), ("r2", 1.0)]) and iv["B"] == (None, None)


def test_loop_merge_units():
    """回ごとに選んだレシピ → 両方の回に当てるレシピ: 同じヘッド(層 21 と段 45)は compose なら 2 つとも、max なら強いほうだけ。"""
    from gne.loopsearch import merge_units
    from gne.steer import Recipe
    rng = np.random.default_rng(1)
    raw = {L: rng.normal(size=8) for L in range(56)}
    ds = SimpleNamespace(raw=raw, unit=lambda L: raw[L] / np.linalg.norm(raw[L]), meta={})
    warns = []
    sess = SimpleNamespace(weights=SimpleNamespace(loop=plan_from_kinds(KINDS)), direction_set=lambda n: ds, warn=warns.append)
    r = Recipe([_proj(21, 20, 2.0), _proj(45, 20, -3.0), _proj(8, 1, 1.0)], "m", name="x", default_direction="d",
               loop_edits="per_visit")
    comp, tw = merge_units(sess, r, "compose")
    assert comp.loop_edits == "shared" and len(comp.proj_units) == 3 and len(tw) == 1
    assert tw[0]["head"] == "21:H20" and tw[0]["layers"] == [21, 45] and tw[0]["max_s"] == -3.0
    c = float(ds.unit(21) @ ds.unit(45))
    assert abs(tw[0]["cos"] - c) < 1e-9
    mx, _ = merge_units(sess, r, "max")
    assert sorted((u.layer, u.k) for u in mx.proj_units) == [(8, 1.0), (45, -3.0)]
    assert mx.loop_visits["21:H20"] == [1, 2] and mx.loop_visits["8:H1"] == [1] and mx.loop_visits["_merge"] == "max"


def test_score_layers_and_candidate_ranges():
    """--score-layers loop = 1 回目の出口・2 回目の出口・最後の層の 1 つ前。候補の範囲の書き方。"""
    from gne.api import Session
    from gne.loopsearch import visit_ranges
    from gne.steer import GneError
    fake = SimpleNamespace(weights=SimpleNamespace(loop=plan_from_kinds(KINDS)), meta=SimpleNamespace(n_layer=56))
    assert Session.score_layers_of(fake, "loop") == [27, 51, 54]
    assert Session.score_layers_of(fake, "27,51,54") == [27, 51, 54] and Session.score_layers_of(fake, None) is None
    with pytest.raises(GneError):
        Session.score_layers_of(fake, "55")                      # 最後の層は測れない
    with pytest.raises(GneError):
        Session.score_layers_of(SimpleNamespace(weights=SimpleNamespace(loop=None), meta=fake.meta), "loop")
    assert Session._layers_of("28-30") == [28, 29, 30] and Session._layers_of([3, 1]) == [1, 3]
    rng = visit_ranges(fake)
    assert rng["first"] == list(range(0, 28)) and rng["second"] == list(range(28, 55))


def test_loop_pipeline_defaults():
    """sensenova_loop だけの一括の既定: 測る層 loop・選ぶ層 54・いつも測る層 52,53。渡した値はそのまま(今までの既定は base / none)。"""
    from gne.api import Session
    li = [SimpleNamespace(index=L, head_dim=4, head_writer="attn_output") for L in range(56)]
    w = SimpleNamespace(loop=plan_from_kinds(KINDS), arch=SimpleNamespace(layers=li))
    fake = SimpleNamespace(weights=w)
    pipe = {"score_layers": None, "select_layer": None, "always_layers": None}
    used = Session._loop_pipeline_defaults(fake, pipe)
    assert pipe == {"score_layers": "loop", "select_layer": 54, "always_layers": [52, 53]} and used == pipe
    kept = {"score_layers": "base", "select_layer": None, "always_layers": "none"}
    assert Session._loop_pipeline_defaults(fake, kept) is None and kept["score_layers"] == "base"
    assert Session._loop_pipeline_defaults(SimpleNamespace(weights=SimpleNamespace(loop=None)), dict(pipe)) is None


MODEL = os.environ.get("GNE_TEST_SENSENOVA")
real = pytest.mark.skipif(not MODEL or not Path(MODEL).exists(), reason="GNE_TEST_SENSENOVA に SenseNova-Loop の GGUF がありません")


@pytest.fixture(scope="module")
def engine():
    from gne.engine import Engine
    try:
        e = Engine(MODEL, n_gpu_layers=int(os.environ.get("GNE_TEST_NGL", "-1")), n_ctx=1024, n_seq_max=1)
    except RuntimeError as ex:
        pytest.skip(f"この llama.cpp では読めません: {ex}")
    yield e
    e.close()


@real
def test_real_weights_and_arch():
    from gne.weights import Weights
    w = Weights(MODEL)
    assert w.loop is not None and (w.loop.lo1, w.loop.lo2, w.loop.shift) == (4, 28, 24)
    li = w.arch.layers
    assert li[30].mixer == "linear_attention" and li[31].mixer == "attention" and li[31].visit == 2 and li[31].stored == 7
    assert w.tensor("blk.31.attn_output.weight") is w.tensor("blk.7.attn_output.weight")
    np.testing.assert_array_equal(w.down_matrix(34), w.down_matrix(10))


@real
def test_real_capture_every_layer(engine):
    from gne.capture import Capture
    e = engine
    ids = e.tokenize_prompt(e.render_chat([{"role": "user", "content": "Hello!"}]))
    want = {f"l_out-{L}" for L in range(e.n_layer)} | {"attn_residual-30", "attn_post_norm-30", "ffn_out-30", "attn_output-31",
                                                      "l_out_streams-30"}
    cap = Capture(lambda n: n in want)
    ctx = e.probe_ctx(cap, need=len(ids))
    e.clear_memory(ctx)
    e.prefill(ctx, [ids])
    got = cap.take()
    assert want <= set(got), sorted(want - set(got))
    s = got["l_out_streams-30"]
    np.testing.assert_allclose(s.reshape(s.shape[0], 2, -1).mean(axis=1), got["l_out-30"], atol=1e-5)


@real
def test_real_cvec_guard(engine):
    from gne.steer import GneError
    e = engine
    ctx = e.steer_ctx()
    v = np.ones(e.n_embd, np.float32)
    e.set_cvec(ctx, {27: v})                               # 出る層は、どの llama.cpp でも次の層に届く
    if not e.loop_cvec_reaches():
        with pytest.raises(GneError):
            e.set_cvec(ctx, {30: v})
    else:
        e.set_cvec(ctx, {30: v})
    e.set_cvec(ctx, None)


def _act(e, ids, names, loras):
    from gne.capture import Capture
    cap = Capture(lambda n: n in names)
    ctx = e.probe_ctx(cap, need=len(ids))
    e.set_lora(ctx, loras)
    e.clear_memory(ctx)
    e.prefill(ctx, [ids])
    e.set_lora(ctx, None)
    return cap.take()


@real
def test_real_gain_lora_runs(engine, tmp_path):
    """層 10 のニューロンを ×0 にする LoRA(1 回目の番号でも 2 回目の番号 34 でも)は、既定(両方の回)では両方の回で消す。"""
    from gne.capture import pick_act_name
    from gne.gain import export_gain_lora
    from gne.steer import Recipe, parse_unit
    from gne.weights import Weights
    e = engine
    if e.loop_edits != "shared":
        pytest.skip("GNE_LOOP_EDITS=per_visit のときは別のテスト")
    w = Weights(MODEL)
    ids = e.tokenize_prompt(e.render_chat([{"role": "user", "content": "Tell me about the ocean."}]))
    seen = e.list_names(ids[:16])
    a10, a34 = pick_act_name(seen, 10), pick_act_name(seen, 34)
    base = _act(e, ids, {a10, a34}, None)
    for L in (10, 34):
        p = tmp_path / f"g{L}.lora.gguf"
        export_gain_lora(w, Recipe([parse_unit(f"{L}:100:x0")]), p)
        got = _act(e, ids, {a10, a34}, [(str(p), 1.0)])
        for nm in (a10, a34):                       # 量子化した重みと F16 の LoRA の差で、0 にはならず数 % 残る
            assert np.abs(got[nm][:, 100]).max() < 0.1 * np.abs(base[nm][:, 100]).max(), (L, nm)
        e.drop_lora(p)


@real
@pytest.mark.skipif(os.environ.get("GNE_LOOP_EDITS") != "per_visit", reason="GNE_LOOP_EDITS=per_visit のときだけ")
def test_real_per_visit_lora(tmp_path):
    """回ごとの名前がある llama.cpp で GNE_LOOP_EDITS=per_visit: 層 10 の LoRA は 1 回目だけ、層 34 の LoRA は 2 回目だけに効く。"""
    from gne.api import Session
    from gne.capture import pick_act_name
    from gne.gain import export_gain_lora
    from gne.steer import GneError, Recipe, parse_unit
    try:
        s = Session(MODEL, root=tmp_path, n_ctx=1024, n_seq_max=1)
        e = s.engine
    except (RuntimeError, GneError) as ex:
        pytest.skip(f"回ごとの名前がある llama.cpp ではありません: {ex}")
    assert s.weights.per_visit_names and e.loop_edits == "per_visit"
    ids = e.tokenize_prompt(e.render_chat([{"role": "user", "content": "Tell me about the ocean."}]))
    seen = e.list_names(ids[:16])
    a10, a34 = pick_act_name(seen, 10), pick_act_name(seen, 34)
    base = _act(e, ids, {a10, a34}, None)
    for L, off, on in ((10, a34, a10), (34, a10, a34)):
        p = tmp_path / f"g{L}.lora.gguf"
        export_gain_lora(s.weights, Recipe([parse_unit(f"{L}:100:x0")]), p)
        got = _act(e, ids, {a10, a34}, [(str(p), 1.0)])
        assert np.abs(got[on][:, 100]).max() < 0.1 * np.abs(base[on][:, 100]).max()
        assert np.abs(got[off][:, 100]).max() > 0.5 * np.abs(base[off][:, 100]).max()
        e.drop_lora(p)
    s.close()


@real
def test_real_recipe_mode_per_file(tmp_path):
    """当て方はレシピとファイルの記録に従う(セッションは既定の shared のまま): per_visit のレシピの層 10 の ×0 は、回ごとの名前が
    ある llama.cpp では 1 回目だけに効く。名前の無い llama.cpp では、理由と直し方を書いて止まる。記録の無いレシピは両方の回。"""
    from gne.api import Session
    from gne.capture import pick_act_name
    from gne.steer import GneError, Recipe, parse_unit
    if os.environ.get("GNE_LOOP_EDITS") == "per_visit":
        pytest.skip("既定の当て方(shared)のセッションで確かめる")
    try:
        s = Session(MODEL, root=tmp_path, n_ctx=1024, n_seq_max=1)
        e = s.engine
    except (RuntimeError, GneError) as ex:
        pytest.skip(f"この llama.cpp では読めません: {ex}")
    pv = Recipe([parse_unit("10:100:x0")], s.meta.model_id, name="pv", loop_edits="per_visit")
    sh = Recipe([parse_unit("10:100:x0")], s.meta.model_id, name="sh")
    ids = e.tokenize_prompt(e.render_chat([{"role": "user", "content": "Tell me about the ocean."}]))
    seen = e.list_names(ids[:16])
    a10, a34 = pick_act_name(seen, 10), pick_act_name(seen, 34)
    base = _act(e, ids, {a10, a34}, None)
    small = lambda got, nm: np.abs(got[nm][:, 100]).max() < 0.1 * np.abs(base[nm][:, 100]).max()
    got = _act(e, ids, {a10, a34}, s.lora_for(sh))
    assert small(got, a10) and small(got, a34)
    if not e.loop_runtime_names():
        with pytest.raises(GneError) as ex:
            s.lora_for(pv)
        assert "loop-merge" in (ex.value.hint or "")
    else:
        got = _act(e, ids, {a10, a34}, s.lora_for(pv))
        assert small(got, a10) and not small(got, a34)
        assert s.lora_for(pv)[0][0] != s.lora_for(sh)[0][0]          # 同じ中身でも当て方が違えば別のファイル
    assert not s.weights.per_visit_names                              # セッションの当て方は元に戻る
    s.close()


@real
def test_real_writes_ranking_mhc(tmp_path):
    """v0.8.0 の書き込みの順位づけ: 2 本の流れの中の層でも、部品から組み立てた層の更新(c·副層の出力、c = (post0 + post1)/2)が、
    捕まえた残差の差 l_out-L − l_out-(L−1) と合う。部品は重みを持つ層の番号で表し(2 回目の層は候補にしない)、
    2 回の書き込みを足す。ニューロンの射影は両方の回に効き、式どおりに MLP の出力を変える。"""
    import json as _json
    from gne.api import Session
    from gne.steer import GneError
    cfg = Path(__file__).resolve().parent.parent / "configs" / "pipeline" / "question_kinds.json"
    d = _json.loads(cfg.read_text(encoding="utf-8"))
    import yaml as _yaml
    c = tmp_path / "qk.yaml"
    c.write_text(_yaml.safe_dump({"name": "qk", "prompts_a": d["prompts_a"][:6], "prompts_b": d["prompts_b"][:6]},
                                 allow_unicode=True), encoding="utf-8")
    try:
        s = Session(MODEL, root=tmp_path / "root", n_ctx=1024, n_seq_max=1)
        s.engine
    except (RuntimeError, GneError) as ex:
        pytest.skip(f"この llama.cpp では読めません: {ex}")
    try:
        s.direction(contrast=c, name="qk", positions="last")
        out = s.heads_writes("qk", verify_top=2, verify_sample=0, verify_examples=4)
        ck = out["check"]
        assert ck["layer_rel_median"] < 0.05 and ck["attn_rel_max"] < 0.1 and ck["mlp_rel_max"] < 0.1, ck
        z = np.load(out["npz"])
        assert set(z["layer_visit"].tolist()) == {0, 1, 2} and int(z["select_layer"]) == 54
        lp = s.weights.loop
        assert not any(lp.second(int(L)) for L in z["head_layer"])           # 2 回目の層は候補にしない(同じ重み)
        assert out["n_units"]["neuron"] == 30 * s.meta.n_ff                    # 層 0〜27 と 52・53
        m = s.check_neuron_proj(10, 7, "qk", ss=(1.0,))
        assert m["v24"], m["rows"]
    finally:
        s.close()
