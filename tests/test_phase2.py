"""Phase 2 の単体テスト(モデル不要): ニューロン選択、上位 k、logit lens、steer lens の保存、逆引きの順位づけ、sweep の部品。"""

import json

import numpy as np
import pytest
from gguf import GGUFWriter

from gne.lens import LogitLens, SteerLens, _TopK, logit_lens_layer
from gne.reverse import ContrastSet, combine_scores, cos_by_layer, rank_by_direction
from gne.selection import parse_neurons
from gne.steer import GneError
from gne.sweep import _done_keys, load_design, unit_schedule
from gne.weights import Weights

N_EMBD, N_FF, N_LAYER, N_VOCAB = 16, 40, 3, 300


@pytest.fixture
def tiny(tmp_path):
    rng = np.random.default_rng(3)
    downs = [rng.standard_normal((N_EMBD, N_FF)).astype(np.float32) for _ in range(N_LAYER)]
    emb = rng.standard_normal((N_VOCAB, N_EMBD)).astype(np.float32)
    gamma = (1 + 0.3 * rng.standard_normal(N_EMBD)).astype(np.float32)
    p = tmp_path / "tiny.gguf"
    w = GGUFWriter(str(p), arch="llama")
    w.add_name("tiny")
    w.add_block_count(N_LAYER)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_token_list([f"t{i}" for i in range(N_VOCAB)])
    for L, d in enumerate(downs):
        w.add_tensor(f"blk.{L}.ffn_down.weight", d)
    w.add_tensor("token_embd.weight", emb)
    w.add_tensor("output_norm.weight", gamma)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return Weights(p), downs, emb, gamma


def test_parse_neurons():
    assert parse_neurons("all", 5).tolist() == [0, 1, 2, 3, 4]
    assert parse_neurons("3,1,7-9", 10).tolist() == [1, 3, 7, 8, 9]
    r = parse_neurons("random:4", 100, seed=2)
    assert r.size == 4 and np.array_equal(r, parse_neurons("random:4", 100, seed=2)) and np.all(np.diff(r) > 0)
    with pytest.raises(GneError):
        parse_neurons("5-200", 100)
    with pytest.raises(GneError):
        parse_neurons("x", 100)


def test_topk_streaming_matches_full_sort():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((7, 1000)).astype(np.float32)
    tk = _TopK(7, 5)
    for s in range(0, 1000, 128):
        tk.add(Z[:, s:s + 128].copy(), s)
    v, i = tk.sorted()
    assert np.array_equal(i, np.argsort(-Z, axis=1)[:, :5])
    assert np.allclose(v, -np.sort(-Z, axis=1)[:, :5])


def test_logit_lens_matches_brute_force(tiny):
    w, downs, emb, gamma = tiny
    ll = logit_lens_layer(w, 1, top_k=6, chunk=64)
    U = downs[1] / np.linalg.norm(downs[1], axis=0)
    Z = emb @ (U * gamma[:, None])                     # (vocab, n_ff)
    z = (Z - Z.mean(0)) / Z.std(0)
    assert np.array_equal(ll.up_ids, np.argsort(-z, axis=0)[:6].T)
    assert np.array_equal(ll.down_ids, np.argsort(z, axis=0)[:6].T)
    assert np.allclose(ll.up_z, -np.sort(-z, axis=0)[:6].T, atol=1e-4)
    assert np.allclose(ll.peak_z, np.maximum(np.abs(z).max(0), 0), atol=1e-4)
    sub = logit_lens_layer(w, 1, neurons=np.array([3, 17]), top_k=6, chunk=64)
    assert np.array_equal(sub.up_ids, ll.up_ids[[3, 17]])


def test_logit_lens_save_load(tiny, tmp_path):
    w = tiny[0]
    ll = logit_lens_layer(w, 0, top_k=4)
    ll.save(tmp_path / "l.npz")
    back = LogitLens.load(tmp_path / "l.npz")
    assert back.layer == 0 and np.array_equal(back.up_ids, ll.up_ids) and np.array_equal(back.neurons, ll.neurons)


def test_steer_lens_store_and_null_pct(tmp_path):
    cfg = {"dose": 0.8, "contexts": [{"kind": "text", "text": "x"}], "top_k": 3, "random_controls": 4, "seed": 0,
           "rho": 1.0}
    sl = SteerLens.empty(2, 10, cfg)
    sl.ctrl_done[:] = True
    sl.ctrl_kl_pos[:] = [0.1, 0.2, 0.3, 0.4]
    sl.kl_pos[5], sl.done[5] = 0.35, True
    sl.save(tmp_path / "s.npz")
    back = SteerLens.load(tmp_path / "s.npz")
    assert back.config == cfg and back.done[5] and np.isclose(back.kl_pos[5], 0.35)
    assert back.null_pct(0.35) == 0.75 and back.null_pct(1.0) == 1.0 and back.null_pct(0.0) == 0.0
    assert SteerLens.config_key(cfg) == SteerLens.config_key(json.loads(json.dumps(cfg)))
    assert len(set(back.ctrl_seeds.tolist())) == 4


def test_rank_by_direction_finds_the_neuron_and_sign(tiny):
    w, downs, _, _ = tiny
    u = downs[2][:, 11] / np.linalg.norm(downs[2][:, 11])
    cands, cos = rank_by_direction(w, {1: np.random.default_rng(0).standard_normal(N_EMBD), 2: -3.0 * u}, top=5)
    assert (cands[0].layer, cands[0].neuron, cands[0].sign) == (2, 11, -1)
    assert np.isclose(cands[0].cos, -1.0, atol=1e-5) and np.isclose(cos[2][11], -1.0, atol=1e-5)


def test_standardized_cos_ignores_outlier_dimension(tiny):
    w, downs, _, _ = tiny
    d = np.zeros(N_EMBD, np.float32)
    d[0], d[5] = 50.0, 1.0                             # 次元 0 が飛び抜けて大きい(ばらつきも大きい)
    scale = np.ones(N_EMBD, np.float32)
    scale[0] = 100.0
    plain = cos_by_layer(w, {1: d})[1]
    std = cos_by_layer(w, {1: d}, {1: scale})[1]
    U = downs[1] / np.linalg.norm(downs[1], axis=0)
    assert np.argmax(np.abs(plain)) == np.argmax(np.abs(U[0]))          # 素の cos は次元 0 に書くニューロンが勝つ
    assert np.corrcoef(np.abs(std), np.abs(U[5] / np.linalg.norm(U / scale[:, None], axis=0)))[0, 1] > 0.9
    same = cos_by_layer(w, {1: d}, {1: np.full(N_EMBD, 2.0, np.float32)})[1]
    assert np.allclose(same, plain, atol=1e-5)                          # 一様なら素の cos と同じ


def test_combine_scores_uses_available_metrics():
    cos = np.array([0.1, -0.5, 0.3])
    assert np.argmax(combine_scores(cos, None, None)) == 1
    act = np.array([1.0, 0.0, 5.0])                    # cos の符号にそろえた活性差
    kl = np.array([np.nan, np.nan, np.nan])
    s = combine_scores(cos, act, kl)
    assert np.argmax(s) == 2 and np.all(np.isfinite(s))


def test_contrast_set_validation(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("system_a: a\nsystem_b: b\nuser_prompts: [x, y]\n")
    cs = ContrastSet.load(p)
    assert cs.name == "c" and cs.user_prompts == ["x", "y"] and len(cs.sha) == 16
    (tmp_path / "bad.yaml").write_text("system_a: a\n")
    with pytest.raises(GneError):
        ContrastSet.load(tmp_path / "bad.yaml")
    with pytest.raises(GneError):
        ContrastSet.load(tmp_path / "missing.yaml")


def test_designs_and_schedule():
    q = load_design("quick")
    assert q.groups() == [(0.8, ["who", "orange"]), (3.0, ["free"])]
    m = load_design("main_rel")
    assert len(m.pairs) == 10 and [g[0] for g in m.groups()] == [-0.7, 0.5, 0.8, 3.0]
    assert len(load_design("calib").pairs) == 36
    assert load_design("quick", prompt_set="ja").prompts["who"].startswith("あなた")
    with pytest.raises(GneError):
        load_design("nope")
    sched = unit_schedule(list(range(100)), 10)
    ctrl = [i for i, (k, _) in enumerate(sched) if k == "random"]
    assert len(sched) == 110 and len(ctrl) == 10 and max(np.diff(ctrl)) <= 12          # 均等に散らばる
    assert [u for k, u in sched if k == "neuron"] == list(range(100))
    assert unit_schedule([], 3) == [("random", 0), ("random", 1), ("random", 2)]


def test_done_keys_tolerates_truncated_line(tmp_path):
    p = tmp_path / "r.jsonl"
    rows = [{"kind": "baseline", "batch": ["who", "orange"], "prompt": "who", "neuron": None, "dose": 0.0},
            {"kind": "neuron", "neuron": 5, "prompt": "who", "dose": 0.8},
            {"kind": "random", "neuron": 0, "prompt": "free", "dose": 3.0}]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"kind": "neuron", "neu')
    keys = _done_keys(p)
    assert ("baseline", ("who", "orange"), "who") in keys and ("neuron", 5, "who", 0.8) in keys
    assert ("random", 0, "free", 3.0) in keys and len(keys) == 3
