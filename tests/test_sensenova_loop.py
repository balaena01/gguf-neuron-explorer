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
    assert (p.lo1, p.lo2, p.shift, p.hi2, p.n_layer) == (4, 28, 24, 52, 56)
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
