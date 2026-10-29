"""モデル不要の単体テスト(cvec の層対応、GGUF 読み書き、テンプレート、捕捉の連結)。"""

import numpy as np
import pytest
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter
from gguf import quants as gquants

from gne.capture import Capture, pick_act_name
from gne.engine import render_template
from gne.steer import cvec_buffer, export_cvec_gguf, import_cvec_gguf, load_cvec_like_common
from gne.verify import compare_greedy
from gne.weights import Weights

N_LAYER, N_EMBD = 6, 8


def _vec(seed, n=N_EMBD):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


# --- cvec_buffer(§5.1, §5.3) ---------------------------------------------------------------------

def test_cvec_buffer_places_layer_L_at_L_minus_1():
    v3, v5 = _vec(3), _vec(5)
    buf, lo, hi = cvec_buffer(N_LAYER, N_EMBD, {3: v3, 5: v5})
    assert buf.shape == ((N_LAYER - 1) * N_EMBD,)
    assert np.array_equal(buf[2 * N_EMBD:3 * N_EMBD], v3)
    assert np.array_equal(buf[4 * N_EMBD:5 * N_EMBD], v5)
    mask = np.ones_like(buf, bool)
    mask[2 * N_EMBD:3 * N_EMBD] = mask[4 * N_EMBD:5 * N_EMBD] = False
    assert not buf[mask].any()
    assert (lo, hi) == (3, 5)


@pytest.mark.parametrize("bad", [{0: _vec(0)}, {N_LAYER: _vec(0)}, {2: _vec(0, N_EMBD + 1)}, {}])
def test_cvec_buffer_rejects(bad):
    with pytest.raises(ValueError):
        cvec_buffer(N_LAYER, N_EMBD, bad)


# --- control vector GGUF(§5.1, §5.2, §6.5) ---------------------------------------------------------

def test_export_matches_cvector_generator_format(tmp_path):
    p = tmp_path / "a.cvec.gguf"
    export_cvec_gguf({2: _vec(2)}, p, model_hint="qwen3", n_layer=N_LAYER, extra={"gne.note": "x"})
    r = GGUFReader(p)
    assert r.fields["general.architecture"].contents() == "controlvector"
    assert r.fields["controlvector.model_hint"].contents() == "qwen3"
    assert r.fields["controlvector.layer_count"].contents() == N_LAYER - 1
    assert [t.name for t in r.tensors] == [f"direction.{i}" for i in range(1, N_LAYER)]
    assert all(t.tensor_type == GGMLQuantizationType.F32 and len(t.shape) == 1 for t in r.tensors)


def test_export_then_common_loader_roundtrip(tmp_path):
    vecs = {1: _vec(1), 4: _vec(4)}
    p = tmp_path / "a.cvec.gguf"
    export_cvec_gguf(vecs, p, model_hint="x", n_layer=N_LAYER)
    n_embd, data = load_cvec_like_common([(p, 1.0)])
    buf, _, _ = cvec_buffer(N_LAYER, N_EMBD, vecs)
    assert n_embd == N_EMBD and np.array_equal(data, buf)


def test_common_loader_strength_and_sum(tmp_path):
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    export_cvec_gguf({2: _vec(1)}, a, model_hint="x", n_layer=N_LAYER)
    export_cvec_gguf({2: _vec(2), 3: _vec(3)}, b, model_hint="x", n_layer=N_LAYER)
    _, data = load_cvec_like_common([(a, 0.5), (b, 2.0)])
    assert np.allclose(data[N_EMBD:2 * N_EMBD], 0.5 * _vec(1) + 2.0 * _vec(2))
    assert np.allclose(data[2 * N_EMBD:3 * N_EMBD], 2.0 * _vec(3))


def _write_raw(path, tensors):
    w = GGUFWriter(str(path), arch="controlvector")
    for name, arr in tensors:
        w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_common_loader_rejects_direction_0_and_non_1d(tmp_path):
    p0 = tmp_path / "zero.gguf"
    _write_raw(p0, [("direction.0", _vec(0))])
    with pytest.raises(ValueError, match="zero"):
        load_cvec_like_common([(p0, 1.0)])
    p2 = tmp_path / "2d.gguf"
    _write_raw(p2, [("direction.1", np.zeros((2, 4), np.float32))])
    with pytest.raises(ValueError, match="1D"):
        load_cvec_like_common([(p2, 1.0)])


def test_import_exact_vs_generator_offset(tmp_path):
    p = tmp_path / "g.gguf"
    _write_raw(p, [("direction.3", _vec(3)), ("direction.4", _vec(4))])
    exact = import_cvec_gguf(p, source="exact")
    gen = import_cvec_gguf(p, source="generator")
    assert sorted(exact) == [3, 4] and sorted(gen) == [2, 3]
    assert np.array_equal(gen[2], exact[3])


# --- weights(§5.4: numpy 形状は ne の逆順、ニューロン n は列 [:, n]) -------------------------------

def _tiny_model(path, *, qtype=GGMLQuantizationType.F32, post_norm=False):
    n_embd, n_ff, n_layer = 32, 64, 2
    rng = np.random.default_rng(0)
    downs = [rng.standard_normal((n_embd, n_ff)).astype(np.float32) for _ in range(n_layer)]
    w = GGUFWriter(str(path), arch="llama")
    w.add_name("tiny")
    w.add_block_count(n_layer)
    w.add_embedding_length(n_embd)
    w.add_feed_forward_length(n_ff)
    w.add_token_list([f"t{i}" for i in range(10)])
    for L, d in enumerate(downs):
        if qtype == GGMLQuantizationType.F32:
            w.add_tensor(f"blk.{L}.ffn_down.weight", d)
        else:
            q = gquants.quantize(d, qtype)
            w.add_tensor(f"blk.{L}.ffn_down.weight", q, raw_shape=q.shape, raw_dtype=qtype)
        if post_norm:
            w.add_tensor(f"blk.{L}.post_ffw_norm.weight", np.ones(n_embd, np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return downs


def test_weights_down_matrix_orientation(tmp_path):
    p = tmp_path / "tiny.gguf"
    downs = _tiny_model(p)
    w = Weights(p)
    m = w.meta
    assert (m.arch, m.n_layer, m.n_embd, m.n_ff, m.n_vocab) == ("llama", 2, 32, 64, 10)
    assert [int(x) for x in w.tensor("blk.0.ffn_down.weight").shape] == [64, 32]   # ggml ne = [n_ff, n_embd]
    assert np.array_equal(w.down_matrix(1), downs[1])
    assert np.array_equal(w.down_column(1, 7), downs[1][:, 7])
    assert np.allclose(w.down_norms(0), np.linalg.norm(downs[0], axis=0))
    assert np.isclose(np.linalg.norm(w.unit_direction(0, 3)), 1.0)
    assert m.tied_output and w.check_arch() == []


def test_weights_dequantize_q8_0_and_warnings(tmp_path):
    p = tmp_path / "tiny_q8.gguf"
    downs = _tiny_model(p, qtype=GGMLQuantizationType.Q8_0, post_norm=True)
    w = Weights(p)
    mat = w.down_matrix(0)
    assert mat.shape == (32, 64) and mat.dtype == np.float32
    assert np.abs(mat - downs[0]).max() < 0.05
    assert w.meta.ffn_down_qtype == "Q8_0"
    assert any("post_ffw_norm" in s for s in w.check_arch())


def test_model_id_is_stable(tmp_path):
    p = tmp_path / "tiny.gguf"
    _tiny_model(p)
    assert Weights(p).meta.model_id == Weights(p).meta.model_id
    assert Weights(p).meta.model_id.startswith("tiny-")


# --- chat template(§5.7) ------------------------------------------------------------------------------

def test_render_template_hf_conventions():
    tpl = ("{% for m in messages %}<|{{ m.role }}|>{{ m.content | tojson }}\n{% endfor %}"
           "{% if add_generation_prompt %}<|assistant|>{% if not enable_thinking %}<think></think>{% endif %}{% endif %}")
    out = render_template(tpl, [{"role": "user", "content": "あなたは誰？"}])
    assert out == '<|user|>"あなたは誰？"\n<|assistant|><think></think>'
    with pytest.raises(Exception, match="boom"):
        render_template("{{ raise_exception('boom') }}", [])


# --- capture -----------------------------------------------------------------------------------------------

def test_render_template_generation_tag():
    """HF transformers の {% generation %}(LFM3 などがアシスタントの発話を印す)は、中身をそのまま出す。"""
    tpl = ("{% for m in messages %}<{{ m.role }}>{% if m.role == 'assistant' %}{% generation %}{{ m.content }}{% endgeneration %}"
           "{% else %}{{ m.content }}{% endif %}{% endfor %}")
    out = render_template(tpl, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])
    assert out == "<user>hi<assistant>yo"


def test_capture_take_concatenates_steps():
    cap = Capture(lambda n: True)
    cap._store("l_out-3", np.zeros((2, 4), np.float32))
    cap.end_step()                                       # decode の区切り
    cap._store("l_out-3", np.ones((3, 4), np.float32))
    out = cap.take()
    assert out["l_out-3"].shape == (5, 4) and out["l_out-3"][2:].all()
    assert cap.take() == {}


def test_capture_splits_step_at_graph_start_and_keeps_first_duplicate():
    """1 回の decode が複数の ubatch に分かれても全部残す。1 つのグラフ内の同名ノードは先に来た方を採る。"""
    cap = Capture(lambda n: n != "embd")
    for k in range(3):                                   # 1 decode 内の 3 ubatch
        cap._ask("embd")                                 # グラフの先頭ノード
        cap._ask("ffn_out-1")
        cap._store("ffn_out-1", np.full((2, 4), k, np.float32))          # FFN 出力
        cap._ask("ffn_out-1")
        cap._store("ffn_out-1", np.full((2, 4), 100 + k, np.float32))    # 残差の和(同名の 2 つ目)
        cap._ask("l_out-1")
        cap._store("l_out-1", np.full((2, 4), 10 + k, np.float32))
    cap.end_step()
    out = cap.take()
    assert cap.n_splits == 2 and cap.duplicates == {"ffn_out-1"}
    assert out["ffn_out-1"][:, 0].tolist() == [0, 0, 1, 1, 2, 2]
    assert out["l_out-1"][:, 0].tolist() == [10, 10, 11, 11, 12, 12]


def test_pick_act_name_priority():
    assert pick_act_name({"ffn_up-3", "ffn_swiglu-3", "ffn_out-3"}, 3) == "ffn_swiglu-3"
    assert pick_act_name({"ffn_gelu-3", "ffn_gate_par-3"}, 3) == "ffn_gate_par-3"
    assert pick_act_name({"ffn_swiglu-30"}, 3) is None


# --- V5 の比較規則 -----------------------------------------------------------------------------------

def test_compare_greedy():
    same = compare_greedy([1, 2, 3], None, None, [1, 2, 3], None, None)
    assert same["ok"] and same["first_diff"] is None
    near = compare_greedy([1, 2], [[1, 9], [2, 5]], [[-0.1, -2.0], [-0.6, -0.8]],
                          [1, 5], [[1, 9], [5, 2]], [[-0.1, -2.0], [-0.7, -0.75]])
    assert near["ok"] and near["first_diff"] == 1
    far = compare_greedy([1, 2], [[1, 9], [2, 5]], [[-0.1, -2.0], [-0.1, -5.0]],
                         [1, 7], [[1, 9], [7, 8]], [[-0.1, -2.0], [-0.1, -5.0]])
    assert not far["ok"]


def test_dry_penalties_and_sampling_keys():
    """DRY(前の並びを続けるトークンへの罰): 末尾と同じ並びの後に続いたトークンに mult·base^(L − allowed)、区切りで一致を切る。
    既定は切り(dry_multiplier 0)。サンプリングの値として受け付け、llama-server の引数にも出す。"""
    import numpy as np

    from gne.engine import Sampling, dry_penalties
    from gne.steer import llama_server_args, sampling_values
    seq = np.array([1, 2, 3, 4, 5, 9, 1, 2, 3, 4])
    assert dry_penalties(seq, np.zeros(seq.size, bool), 0.8, 1.75, 2) == {5: 0.8 * 1.75 ** 2}
    brk = np.zeros(seq.size, bool)
    brk[7] = True
    assert dry_penalties(seq, brk, 0.8, 1.75, 2) == {5: 0.8}
    assert dry_penalties(np.array([1, 2, 3, 1]), np.zeros(4, bool), 0.8, 1.75, 2) == {}     # 1 トークンの一致は許す
    assert Sampling().dry_multiplier == 0 and "dry_multiplier" not in Sampling().to_dict()
    v = sampling_values({"dry_multiplier": 0.8, "dry_allowed_length": 3})
    assert v["dry_multiplier"] == 0.8 and v["dry_allowed_length"] == 3 and "--dry-multiplier 0.8" in llama_server_args(v)
    sp = Sampling(**{k: v[k] for k in v}, seed=1)
    x = np.zeros(10)
    x[5] = 1.0
    assert sp.pick(x, set(), np.random.default_rng(0), {5: 100.0}) != 5                     # 罰で選ばれなくなる
