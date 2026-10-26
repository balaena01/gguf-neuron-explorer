"""Phase 1 の単体テスト(モデル不要): レシピ、dose → ベクトル、書き出し形式、bake、統計の近似。"""

import json

import numpy as np
import pytest
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter

from gne.cli import parse_layers
from gne.schemas import SteerOut
from gne.stats import ResidStats, StatsView, _ActAccumulator
from gne.steer import (GneError, Recipe, Unit, bake_bias_gguf, bias_support, export_vectors, load_cvec_like_common,
                       parse_unit, random_direction, recipe_vectors, unit_details, unit_vector)
from gne.api import _merge_resid
from gne.weights import Weights

N_EMBD, N_FF, N_LAYER = 32, 64, 4


def _write_model(path, arch="llama", *, down_bias=False, scale=None):
    rng = np.random.default_rng(1)
    downs = [rng.standard_normal((N_EMBD, N_FF)).astype(np.float32) for _ in range(N_LAYER)]
    w = GGUFWriter(str(path), arch=arch)
    w.add_name("tiny")
    w.add_block_count(N_LAYER)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_token_list([f"t{i}" for i in range(10)])
    w.add_string("tokenizer.chat_template", "{{ messages }}")
    for L, d in enumerate(downs):
        w.add_tensor(f"blk.{L}.ffn_down.weight", d)
        if down_bias:
            w.add_tensor(f"blk.{L}.ffn_down.bias", np.full(N_EMBD, 0.5, np.float32))
        if scale is not None:
            w.add_tensor(f"blk.{L}.ffn_down.scale", np.array([scale], np.float32))
    w.add_tensor("token_embd.weight", rng.standard_normal((10, N_EMBD)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return downs


@pytest.fixture
def tiny(tmp_path):
    p = tmp_path / "tiny.gguf"
    downs = _write_model(p)
    return Weights(p), downs


def _stats(rho=10.0, absmax=3.0):
    resid = ResidStats(np.arange(N_LAYER, dtype=np.int32), *(np.full(N_LAYER, rho, np.float32) for _ in range(4)),
                       100, "x", 512)

    class _A:
        def __init__(self):
            self.absmax = np.full(N_FF, absmax, np.float32)
    return StatsView(resid, lambda L: _A())


# --- レシピ ---------------------------------------------------------------------------------------------

def test_parse_unit():
    assert parse_unit("14:8949:+1.0") == Unit(14, 8949, 1.0)
    assert parse_unit("13:7956:-1") == Unit(13, 7956, -1.0)
    assert parse_unit("5:random:3:0.8") == Unit(5, None, 0.8, "random", 3)
    for bad in ["14:8949", "a:1:1", "1:2:x", "1:random:0.5"]:
        with pytest.raises(GneError):
            parse_unit(bad)


def test_recipe_roundtrip_and_validation(tmp_path):
    r = Recipe([Unit(2, 5, 0.8), Unit(3, None, -0.5, "random", 7)], "m", "rel", "t")
    r.save(tmp_path / "r.yaml")
    back = Recipe.load(tmp_path / "r.yaml")
    assert back.units == r.units and back.dose_unit == "rel" and back.name == "t"
    with pytest.raises(GneError):
        Recipe([Unit(1, None, 1.0, "random", 1)], dose_unit="raw").validate()
    with pytest.raises(GneError):
        Recipe([], dose_unit="rel").validate()


# --- dose → ベクトル(§2.2) --------------------------------------------------------------------------------

def test_unit_vector_rel_absmax_raw(tiny):
    w, downs = tiny
    st = _stats(rho=10.0, absmax=3.0)
    col = downs[2][:, 5]
    v = unit_vector(w, st, Unit(2, 5, 0.8), "rel")
    assert np.isclose(np.linalg.norm(v), 8.0, rtol=1e-5) and np.allclose(v / 8.0, col / np.linalg.norm(col), atol=1e-6)
    assert np.allclose(unit_vector(w, st, Unit(2, 5, 0.5), "absmax"), 0.5 * 3.0 * col, rtol=1e-5)
    assert np.allclose(unit_vector(w, st, Unit(2, 5, -2.0), "raw"), -2.0 * col, rtol=1e-5)


def test_raw_and_absmax_use_ffn_down_scale(tmp_path):
    p = tmp_path / "scaled.gguf"
    downs = _write_model(p, scale=0.25)
    w = Weights(p)
    assert w.meta.ffn_down_scale == {L: 0.25 for L in range(N_LAYER)}
    st = _stats()
    assert np.allclose(unit_vector(w, st, Unit(1, 3, 2.0), "raw"), 2.0 * 0.25 * downs[1][:, 3], rtol=1e-5)
    v_rel = unit_vector(w, st, Unit(1, 3, 0.8), "rel")          # rel は残差に足す大きさで決まるので s に依らない
    assert np.isclose(np.linalg.norm(v_rel), 8.0, rtol=1e-5)


def test_random_unit_same_size_and_reproducible(tiny):
    w, _ = tiny
    st = _stats(rho=10.0)
    a = unit_vector(w, st, Unit(1, None, 0.8, "random", 3))
    b = unit_vector(w, st, Unit(1, None, 0.8, "random", 3))
    c = unit_vector(w, st, Unit(2, None, 0.8, "random", 3))
    assert np.array_equal(a, b) and not np.allclose(a, c * 1.0)
    assert np.isclose(np.linalg.norm(a), 8.0, rtol=1e-5)
    assert np.isclose(np.linalg.norm(random_direction(N_EMBD, 1, 1)), 1.0)


def test_recipe_vectors_sum_per_layer_and_details(tiny):
    w, downs = tiny
    st = _stats(rho=10.0)
    r = Recipe([Unit(2, 5, 0.8), Unit(2, 6, -0.5), Unit(3, 1, 0.3)])
    vecs = recipe_vectors(w, st, r)
    expect2 = unit_vector(w, st, r.units[0]) + unit_vector(w, st, r.units[1])
    assert sorted(vecs) == [2, 3] and np.allclose(vecs[2], expect2)
    det = unit_details(w, st, r)
    assert np.isclose(det[0]["rel_push"], 0.8) and np.isclose(det[1]["rel_push"], 0.5)
    # add は活性 a_n に足した相当量: v = add · W_down[:, n]
    assert np.isclose(det[1]["add"], -0.5 * 10.0 / np.linalg.norm(downs[2][:, 6]), rtol=1e-5)


def test_stats_view_reports_missing():
    sv = StatsView(None)
    with pytest.raises(KeyError, match="stats resid"):
        sv.rho(3)
    with pytest.raises(KeyError, match="stats act"):
        sv.absmax(3, 0)


# --- 書き出し形式 -------------------------------------------------------------------------------------------

def test_export_vectors_all_formats(tmp_path):
    vecs = {2: np.arange(N_EMBD, dtype=np.float32), 3: -np.ones(N_EMBD, np.float32)}
    meta = {"model_id": "m", "recipe": {"name": "r"}}
    for fmt, name in [("cvec", "a.cvec.gguf"), ("npz", "a.npz"), ("json", "a.json"), ("safetensors", "a.safetensors")]:
        export_vectors(vecs, tmp_path / name, fmt=fmt, n_layer=N_LAYER, model_hint="llama", meta=meta)
    _, data = load_cvec_like_common([(tmp_path / "a.cvec.gguf", 1.0)])
    assert np.array_equal(data[N_EMBD:2 * N_EMBD], vecs[2])
    z = np.load(tmp_path / "a.npz")
    assert z["layers"].tolist() == [2, 3] and np.array_equal(z["layer_3"], vecs[3]) and json.loads(str(z["meta"]))["model_id"] == "m"
    j = json.loads((tmp_path / "a.json").read_text())
    assert j["vectors"]["2"] == vecs[2].tolist() and "layer_semantics" in j["meta"]
    from safetensors.numpy import load_file
    from safetensors import safe_open
    st = load_file(str(tmp_path / "a.safetensors"))
    assert np.array_equal(st["direction.3"], vecs[3])
    with safe_open(str(tmp_path / "a.safetensors"), "np") as f:
        assert f.metadata()["model_id"] == "m"
    with pytest.raises(GneError):
        export_vectors(vecs, tmp_path / "x", fmt="pt", n_layer=N_LAYER, model_hint="llama")


# --- bake(§6.5.1) -------------------------------------------------------------------------------------------

def test_bias_support_table():
    assert bias_support("llama")[0] == "add"
    assert bias_support("gpt2")[0] == "existing"
    assert bias_support("qwen3")[0] is None and bias_support("qwen35")[0] is None
    assert bias_support("refact")[0] is None and "使わない" in bias_support("refact")[1]


def test_bake_adds_bias_and_copies_everything_else(tiny, tmp_path):
    w, _ = tiny
    vecs = {1: np.full(N_EMBD, 2.0, np.float32), 0: np.full(N_EMBD, -1.0, np.float32)}
    dst = tmp_path / "baked.gguf"
    rep = bake_bias_gguf(w, vecs, w.path, dst, recipe=Recipe([Unit(1, 0, 1.0)], name="r"))
    assert rep.mode == "add" and rep.added == ["blk.0.ffn_down.bias", "blk.1.ffn_down.bias"]
    src, out = GGUFReader(w.path), GGUFReader(dst)
    a, b = {t.name: t for t in src.tensors}, {t.name: t for t in out.tensors}
    assert set(b) - set(a) == {"blk.0.ffn_down.bias", "blk.1.ffn_down.bias"}
    assert all(np.array_equal(np.asarray(a[n].data), np.asarray(b[n].data)) for n in a)
    assert b["blk.1.ffn_down.bias"].tensor_type == GGMLQuantizationType.F32
    assert np.array_equal(np.asarray(b["blk.1.ffn_down.bias"].data), vecs[1])
    assert out.fields["general.name"].contents() == "tiny +recipe:r"
    assert json.loads(out.fields["gne.recipe"].contents())["name"] == "r"
    assert out.fields["tokenizer.chat_template"].contents() == "{{ messages }}"
    assert out.fields["tokenizer.ggml.tokens"].contents() == src.fields["tokenizer.ggml.tokens"].contents()


def test_bake_adds_to_existing_bias(tmp_path):
    p = tmp_path / "gpt2.gguf"
    _write_model(p, arch="gpt2", down_bias=True)
    w = Weights(p)
    dst = tmp_path / "baked.gguf"
    rep = bake_bias_gguf(w, {2: np.ones(N_EMBD, np.float32)}, p, dst)
    assert rep.mode == "existing" and rep.modified == ["blk.2.ffn_down.bias"] and not rep.added
    b = {t.name: t for t in GGUFReader(dst).tensors}
    assert np.allclose(np.asarray(b["blk.2.ffn_down.bias"].data), 1.5)
    assert np.allclose(np.asarray(b["blk.1.ffn_down.bias"].data), 0.5)


def test_bake_refuses_unsupported_arch(tmp_path):
    p = tmp_path / "q.gguf"
    _write_model(p, arch="qwen3")
    w = Weights(p)
    with pytest.raises(GneError) as ei:
        bake_bias_gguf(w, {1: np.ones(N_EMBD, np.float32)}, p, tmp_path / "o.gguf")
    assert ei.value.code == "ARCH_UNSUPPORTED"


# --- 統計の近似 --------------------------------------------------------------------------------------------

def test_activation_quantiles_close_to_exact():
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((20000, 8)) * np.array([0.01, 0.1, 1, 3, 10, 0.5, 2, 30])).astype(np.float32)
    acc = _ActAccumulator(8)
    for chunk in np.array_split(a, 7):
        acc.add(chunk)
    for q in (0.5, 0.99, 0.999):
        exact = np.quantile(np.abs(a), q, axis=0)
        assert np.all(np.abs(acc.quantile(q) / exact - 1) < 0.06), q
    st = acc.finish(3, "ffn_swiglu-3", "x")
    assert np.allclose(st.absmax, np.abs(a).max(axis=0))
    assert np.allclose(st.rms, np.sqrt((a.astype(np.float64) ** 2).mean(axis=0)), rtol=1e-5)


def test_resid_stats_roundtrip_and_merge(tmp_path):
    a = ResidStats(np.array([0, 1, 2], np.int32), *(np.array([1, 2, 3], np.float32) for _ in range(4)), 10, "s", 512)
    a.save(tmp_path / "r.npz")
    b = ResidStats.load(tmp_path / "r.npz")
    assert b.rho(1) == 2.0 and b.corpus_sha == "s"
    new = ResidStats(np.array([1], np.int32), *(np.array([9], np.float32) for _ in range(4)), 10, "s", 512)
    m = _merge_resid(b, new)
    assert m.layers.tolist() == [0, 1, 2] and m.p50.tolist() == [1, 9, 3]


def test_parse_layers():
    assert parse_layers("all") is None
    assert parse_layers("10-12") == [10, 11, 12]
    assert parse_layers("1,5,9-11") == [1, 5, 9, 10, 11]
    with pytest.raises(GneError):
        parse_layers("x")


def test_schema_rejects_unknown_fields():
    from pydantic import ValidationError
    base = {"schema_version": "1", "command": "steer", "model_id": "m", "llama_cpp_version": "0.3.36", "seed": 0,
            "generated_at": "2026-10-26T00:00:00+00:00", "messages": [], "recipe": {"name": None, "model_id": None,
            "dose_unit": "rel", "units": []}, "units": [], "layer_norms": {}, "baseline": None,
            "steered": {"text": "x", "finish": "eog", "n_gen": 1}, "differs": None}
    SteerOut.model_validate(base)
    with pytest.raises(ValidationError):
        SteerOut.model_validate({**base, "extra": 1})


def test_template_override(tmp_path, monkeypatch):
    import gne.store
    from gne.api import template_override
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "qwen3.yaml").write_text("chat_template: 'A{{ messages }}'\n")
    monkeypatch.setattr(gne.store, "CONFIGS", tmp_path)
    assert template_override("some-model-id", "qwen3")[1] == "A{{ messages }}"
    (tmp_path / "templates" / "some-model-id.yaml").write_text("chat_template: 'B'\n")
    assert template_override("some-model-id", "qwen3")[1] == "B"          # model_id が優先
    assert template_override("x", "llama") is None


def _assert_same_as_gguf_reader(path):
    from gne.gguf_fast import FastGGUF
    f, r = FastGGUF(path), GGUFReader(path)
    for k, fld in r.fields.items():
        if k.startswith("GGUF."):
            continue
        if k in f.values:
            assert f.values[k] == fld.contents(), k
        if fld.types[0].name == "ARRAY":
            assert f.array_len[k] == len(fld.data), k
    rt = {t.name: t for t in r.tensors}
    assert [t.name for t in f.tensors] == [t.name for t in r.tensors]
    for t in f.tensors:
        u = rt[t.name]
        assert t.tensor_type == u.tensor_type and t.shape == [int(x) for x in u.shape] and t.data_offset == u.data_offset
        assert t.data.shape == u.data.shape and t.data.dtype == u.data.dtype
        assert np.array_equal(np.asarray(t.data), np.asarray(u.data))


def test_fast_reader_matches_gguf_reader(tmp_path):
    p = tmp_path / "tiny.gguf"
    _write_model(p)
    _assert_same_as_gguf_reader(p)
    from tests.test_unit import _tiny_model
    q = tmp_path / "q8.gguf"
    _tiny_model(q, qtype=GGMLQuantizationType.Q8_0)
    _assert_same_as_gguf_reader(q)
