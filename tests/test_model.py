"""実モデルでの Phase 0 受け入れテスト(V1〜V4, V6, V8)。GNE_TEST_MODEL(既定 models/Qwen3-0.6B-Q8_0.gguf)。
V5 は HF 実装が要るので `gne verify --hf-model --hf-python` で回す。V7 は `gne bench`。"""

import os
from pathlib import Path

import pytest

from gne.engine import Engine
from gne.verify import Verifier, compare_greedy, find_llama_completion
from gne.weights import Weights

MODEL = Path(os.environ.get("GNE_TEST_MODEL", Path(__file__).resolve().parent.parent / "models" / "Qwen3-0.6B-Q8_0.gguf"))

pytestmark = [pytest.mark.model,
              pytest.mark.skipif(not MODEL.exists(), reason=f"{MODEL} がありません(GNE_TEST_MODEL で指定)")]


@pytest.fixture(scope="module")
def verifier():
    w = Weights(MODEL)
    e = Engine(MODEL)
    yield Verifier(e, w)
    e.close()


def _check(r):
    assert r.status in ("pass", "info"), f"{r.id}: {r.detail}"
    return r


def test_v1_shapes(verifier):
    _check(verifier.v1_shapes())


def test_v2_ffn_linearity(verifier):
    _check(verifier.v2_ffn_linearity())


def test_v3_cvec_layer(verifier):
    r = _check(verifier.v3_cvec_layer())
    assert r.data["d_prev_max"] == 0 and r.data["err_L"] < 1e-3 and r.data["d_next_max"] > 0


def test_v4_direction_file(verifier):
    _check(verifier.v4_direction_file(find_llama_completion()))


def test_v6_attention_sink_runs(verifier):
    _check(verifier.v6_attention_sink())


def test_v8_tokenize(verifier):
    _check(verifier.v8_tokenize())


def test_generate_batch_matches_single(verifier):
    """バッチ生成(複数 seq を 1 バッチ)と 1 本ずつの生成は、一致するか、僅差の箇所でだけ分かれる。
    llama.cpp の計算はバッチの組み方で数値誤差の範囲(KL ~1e-3、CPU/GPU 差と同程度)で変わるため、
    ビット一致は求めない(docs/verification.md の F2)。1 本ずつの生成は繰り返しても同一。"""
    e = verifier.e
    ctx = e.steer_ctx()
    e.set_cvec(ctx, None)
    seqs = verifier.prompt_ids
    batch = e.generate_greedy(ctx, seqs, max_tokens=24)
    single = [e.generate_greedy(ctx, [s], max_tokens=24)[0] for s in seqs]
    again = [e.generate_greedy(ctx, [s], max_tokens=24)[0] for s in seqs]
    for b, s, a in zip(batch, single, again):
        assert s.token_ids == a.token_ids
        c = compare_greedy(b.token_ids, b.top_ids, b.top_lps, s.token_ids, s.top_ids, s.top_lps)
        assert c["ok"], c["summary"]


def test_batch_has_no_cross_sequence_leak(verifier):
    """バッチ内の他のシーケンスを差し替えても、seq 0 の次トークン分布は変わらない。"""
    import numpy as np
    e = verifier.e
    ctx = e.steer_ctx()
    e.set_cvec(ctx, None)
    s0, s1, s2 = verifier.prompt_ids
    other = e.tokenize_prompt(e.render_chat([{"role": "user", "content": "Write a haiku about rain."}]))
    e.clear_memory(ctx)
    a = e.prefill(ctx, [s0, s1, s2]).logits
    e.clear_memory(ctx)
    b = e.prefill(ctx, [s0, other, s2]).logits
    assert np.array_equal(a[0], b[0])


def test_list_names_has_core_tensors(verifier):
    L = verifier.L
    names = verifier.list_names(L)
    assert {f"l_out-{L}", f"ffn_out-{L}"} <= names


def test_fast_reader_on_real_model():
    from tests.test_phase1 import _assert_same_as_gguf_reader
    _assert_same_as_gguf_reader(MODEL)
