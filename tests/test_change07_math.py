"""CHANGE-07 WP2 のテスト: keyed write の数式(鍵 κ・基底 Q・AUC)、鍵のファイル、LoRA の行列(V31・V32)、書く向きの補正、
キャッシュ鍵。モデルは使わない(小さな GGUF を作る)。"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gne.keyed import (
    KeySet,
    KeyedUnit,
    auc,
    check_layer_ok,
    example_hash,
    export_keyed_lora,
    floor_gamma,
    key_dir,
    keyed_cache_key,
    keyed_lora_path,
    keyed_meta,
    keyed_tensors,
    keyed_units,
    list_keys,
    load_key,
    make_key,
    orth_basis,
    write_vector,
    writer_in_dim,
    writer_tensor_name,
)
from gne.steer import GneError
from gne.store import Store

N_EMBD, N_FF, N_HEAD, HEAD_DIM, N_EXP, N_FF_EXP, N_SHEXP = 16, 24, 4, 3, 3, 10, 8


def _gguf(path, *, arch="llama", n_layer=4, moe_layers=(), shexp=False, post_norm=None, down_scale=None, nextn=0):
    """attention(attn_q・attn_output)と FFN を持つ小さな GGUF。moe_layers の層は ffn_down_exps(と shexp なら
    ffn_down_shexp)、post_norm は層 → γ、down_scale は層 → ffn_down.scale、nextn は最後の nextn 層を MTP の層にする。"""
    from gguf import GGUFWriter
    rng = np.random.default_rng(1)
    w = GGUFWriter(str(path), arch=arch)
    w.add_name("tiny")
    w.add_block_count(n_layer)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(2)
    w.add_token_list([f"t{i}" for i in range(8)])
    if moe_layers:
        w.add_expert_count(N_EXP)
        w.add_expert_used_count(1)
    for L in range(n_layer):
        mats = {"attn_q": (N_HEAD * HEAD_DIM, N_EMBD), "attn_output": (N_EMBD, N_HEAD * HEAD_DIM)}
        if L in moe_layers:
            mats["ffn_down_exps"] = (N_EXP, N_EMBD, N_FF_EXP)
            if shexp:
                mats["ffn_down_shexp"] = (N_EMBD, N_SHEXP)
        else:
            mats["ffn_down"] = (N_EMBD, N_FF)
            mats["ffn_up"] = (N_FF, N_EMBD)
        if L >= n_layer - nextn:
            mats["nextn.eh_proj"] = (N_EMBD, 2 * N_EMBD)
        for nm, shape in mats.items():
            w.add_tensor(f"blk.{L}.{nm}.weight", rng.standard_normal(shape).astype(np.float32))
        if post_norm and L in post_norm:
            w.add_tensor(f"blk.{L}.post_ffw_norm.weight", np.asarray(post_norm[L], dtype=np.float32))
        if down_scale and L in down_scale:
            w.add_tensor(f"blk.{L}.ffn_down.scale", np.array([down_scale[L]], dtype=np.float32))
    w.add_tensor("token_embd.weight", rng.standard_normal((8, N_EMBD)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _weights(tmp_path, name="m.gguf", **kw):
    from gne.weights import Weights
    _gguf(tmp_path / name, **kw)
    return Weights(tmp_path / name)


def _dirs(tmp_path, name="d1", n_layer=4, seed=2, scale=1.0, base_layer=None):
    """方向セット(層 0..n_layer-1 の raw)を書いて読む。"""
    from gne.direction import DirectionSet
    rng = np.random.default_rng(seed)
    raw = {L: (scale * rng.standard_normal(N_EMBD)).astype(np.float32) for L in range(n_layer)}
    extra = {} if base_layer is None else {"base_layer": np.int32(base_layer)}
    p = tmp_path / f"{name}.npz"
    np.savez(p, layers=np.array(sorted(raw), dtype=np.int32), **{f"layer_{L}": v for L, v in raw.items()}, **extra)
    return DirectionSet.load(p)


def _key(target, layers, in_dim, *, seed=5, n_b=6, pseudo=0, rms=None, gate=None, name="k"):
    """側 B の標本 n_b 本と μ から作った鍵(energy=1)。"""
    rng = np.random.default_rng(seed)
    ks = KeySet(target=target, kappa={}, mu={}, kn={}, gain={}, name=name)
    for L in layers:
        B = rng.standard_normal((n_b, in_dim))
        mu = rng.standard_normal(in_dim)
        Q = orth_basis(B, energy=1.0)
        kap, info = make_key(mu, Q)
        ks.kappa[L], ks.mu[L] = kap.astype(np.float32), mu.astype(np.float32)
        ks.kn[L], ks.gain[L], ks.q_rank[L] = info["kn"], info["gain"], info["q_rank"]
        if pseudo:
            ks.kappa_ctrl[L] = rng.standard_normal((pseudo, in_dim)).astype(np.float32)
        if rms is not None:
            ks.rms_a[L] = float(rms)
        if gate is not None:
            ks.gate_a[L] = float(gate)
    return ks


def _ku(L, alpha, d, k, kind="ffn", expert=None):
    return KeyedUnit(L, float(alpha), d, k, kind, expert)


# --- 数式 ----------------------------------------------------------------------------------------------

def test_v31_key_orthogonal_and_normalized():
    """V31: κᵀv ≈ 0(v は Q の列と側 B の各標本)、κᵀμ = 1、κ = k/‖k‖²、kn・gain の定義。"""
    rng = np.random.default_rng(0)
    B = rng.standard_normal((5, N_FF))
    mu = rng.standard_normal(N_FF) + 0.5 * B[0]
    Q = orth_basis(B, energy=1.0)
    assert Q.shape == (N_FF, 5)
    np.testing.assert_allclose(Q.T @ Q, np.eye(5), atol=1e-12)
    kap, info = make_key(mu, Q)
    assert np.abs(kap @ Q).max() < 1e-12
    assert np.abs(B @ kap).max() < 1e-10 * np.linalg.norm(B)
    assert abs(kap @ mu - 1.0) < 1e-12
    k = mu - Q @ (Q.T @ mu)
    np.testing.assert_allclose(kap, k / (k @ k), rtol=1e-10)
    assert info["kn"] == pytest.approx(np.linalg.norm(k) / np.linalg.norm(mu))
    assert info["gain"] == pytest.approx(1.0 / info["kn"])
    assert info["q_rank"] == 5 and info["weak"] is False
    # Q が空(側 B なし)なら κ = μ/‖μ‖²
    kap0, info0 = make_key(mu, np.zeros((N_FF, 0)))
    np.testing.assert_allclose(kap0, mu / (mu @ mu))
    assert info0["kn"] == pytest.approx(1.0)


def test_orth_basis_gram_and_svd_agree():
    """m ≤ in は Gram 行列、m > in は SVD。張る空間(射影)は同じ、ランク落ちの X でも余計な列を出さない。"""
    rng = np.random.default_rng(1)
    base = rng.standard_normal((3, N_FF))
    X = rng.standard_normal((7, 3)) @ base                     # 7 本だが階数 3
    Q = orth_basis(X, energy=1.0)
    assert Q.shape == (N_FF, 3)
    P = Q @ Q.T
    np.testing.assert_allclose(P @ X.T, X.T, atol=1e-9)
    Xbig = rng.standard_normal((40, 3)) @ base                 # m > in の道
    Q2 = orth_basis(Xbig, energy=1.0)
    assert Q2.shape == (N_FF, 3)
    np.testing.assert_allclose(Q2 @ Q2.T, P, atol=1e-9)
    assert orth_basis(np.zeros((4, N_FF))).shape == (N_FF, 0)
    assert orth_basis(np.zeros((0, N_FF))).shape == (N_FF, 0)


def test_orth_basis_truncation():
    """特異値 10・5・1・0.1: 累積の割合は 0.794・0.992・0.99992・1 → energy 0.99 で 2 本、0.999 で 3 本、1 で 4 本。
    max_rank で上限。打ち切ると残した方向だけに直交する。"""
    rng = np.random.default_rng(2)
    U, _ = np.linalg.qr(rng.standard_normal((6, 4)))
    V, _ = np.linalg.qr(rng.standard_normal((N_FF, 4)))
    X = U @ np.diag([10.0, 5.0, 1.0, 0.1]) @ V.T
    assert orth_basis(X, energy=0.99).shape[1] == 2
    assert orth_basis(X, energy=0.999).shape[1] == 3
    assert orth_basis(X, energy=1.0).shape[1] == 4
    assert orth_basis(X, energy=1.0, max_rank=1).shape[1] == 1
    Q = orth_basis(X, energy=0.99)
    np.testing.assert_allclose(np.abs(Q.T @ V[:, :2]), np.eye(2), atol=1e-9)     # 上の 2 方向を張る
    with pytest.raises(GneError):
        orth_basis(X, energy=0.0)


def test_many_side_b_samples_give_small_kn_and_warning():
    """側 B の標本が次元を超える(N > in)と Q がほぼ全空間を張り、kn が小さくなって警告、全部張ると BAD_KEY。"""
    rng = np.random.default_rng(3)
    c = 3.0 * rng.standard_normal(N_FF)
    B = c + rng.standard_normal((40, N_FF))                    # 40 > 24
    mu = B.mean(axis=0) + 0.01 * rng.standard_normal(N_FF)     # 側 A は側 B とほとんど同じ
    msgs = []
    kap, info = make_key(mu, orth_basis(B, energy=0.99), warn=msgs.append, label="L3")
    assert info["kn"] < 0.05 and info["weak"] is True and info["gain"] > 20
    assert msgs and "kn=" in msgs[0] and msgs[0].startswith("L3: ")
    assert abs(kap @ mu - 1.0) < 1e-9
    with pytest.raises(GneError) as ei:
        make_key(mu, orth_basis(B, energy=1.0))                # 24 次元を全部張る → k ≈ 0
    assert ei.value.code == "BAD_KEY"
    with pytest.raises(GneError):
        make_key(np.zeros(N_FF), np.zeros((N_FF, 0)))


def test_auc_with_ties():
    assert auc([1, 2, 3], [0, 0, 0]) == 1.0
    assert auc([0, 0], [1, 2]) == 0.0
    assert auc([1, 1], [1, 1]) == 0.5
    assert auc([2, 1], [1, 0]) == pytest.approx(3.5 / 4)
    assert auc([], [1]) is None and auc([1], []) is None
    rng = np.random.default_rng(4)
    p, q = rng.integers(0, 5, 30), rng.integers(0, 5, 25)       # 同順位だらけ
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in p for b in q])
    assert auc(p, q) == pytest.approx(brute)


def test_example_hash_stable():
    assert example_hash(None, "やあ") == example_hash("", "やあ")
    assert example_hash("sys", "やあ") != example_hash(None, "やあ")
    assert example_hash(None, "やあ", "なのだ") != example_hash(None, "やあ")
    assert len(example_hash(None, "x")) == 16


# --- 書き込み先・層 ---------------------------------------------------------------------------------------

def test_writer_targets_and_layer_checks(tmp_path):
    w = _weights(tmp_path, n_layer=5, nextn=1)                 # 層 4 は MTP、本流の最終層は 3
    assert writer_tensor_name(w, 1, "ffn") == "blk.1.ffn_down.weight"
    assert writer_in_dim(w, 1, "ffn") == N_FF
    assert writer_tensor_name(w, 1, "attn") == "blk.1.attn_output.weight"
    assert writer_in_dim(w, 1, "attn") == N_HEAD * HEAD_DIM
    assert check_layer_ok(w, 2, "ffn") == "blk.2.ffn_down.weight"
    for L in (3, 4, 9):                                        # 最終層・MTP・範囲外
        with pytest.raises(GneError) as ei:
            check_layer_ok(w, L, "ffn")
        assert ei.value.code == "BAD_RECIPE"
    with pytest.raises(GneError):
        writer_tensor_name(w, 1, "shared")                     # dense の層に shared expert は無い
    with pytest.raises(GneError):
        writer_tensor_name(w, 1, "expert", 0)
    with pytest.raises(GneError):
        writer_tensor_name(w, 1, "mlp")


def test_moe_targets(tmp_path):
    w = _weights(tmp_path, n_layer=4, moe_layers=(1, 2), shexp=True)
    with pytest.raises(GneError) as ei:
        check_layer_ok(w, 1, "ffn")                            # MoE の層に dense の ffn_down は無い
    assert ei.value.code == "BAD_RECIPE" and ":S:keyed" in ei.value.hint
    assert check_layer_ok(w, 1, "shared") == "blk.1.ffn_down_shexp.weight"
    assert writer_in_dim(w, 1, "shared") == N_SHEXP
    assert check_layer_ok(w, 2, "expert", 2) == "blk.2.ffn_down_exps.weight"
    assert writer_in_dim(w, 2, "expert", 2) == N_FF_EXP
    with pytest.raises(GneError):
        check_layer_ok(w, 2, "expert", N_EXP)                  # 範囲外の専門家
    with pytest.raises(GneError):
        check_layer_ok(w, 2, "expert", None)


# --- 書く向き -----------------------------------------------------------------------------------------------

def test_write_vector_raw_and_base(tmp_path):
    w = _weights(tmp_path)
    ds = _dirs(tmp_path, base_layer=2)
    info = {}
    d = write_vector(w, ds, 1, info=info)
    np.testing.assert_allclose(d, ds.raw[1])                   # α は raw: d = r_L そのもの
    assert info["direction_from"] == 1 and info["out_scale"] == 1.0 and info["post_norm"] is False
    np.testing.assert_allclose(write_vector(w, ds, 1, direction_layer="base"), ds.raw[2])
    ds_missing = SimpleNamespace(name="x", raw={0: ds.raw[0]}, meta={})
    with pytest.raises(GneError) as ei:
        write_vector(w, ds_missing, 1)
    assert ei.value.code == "BAD_RECIPE"


def test_post_norm_correction_and_down_scale(tmp_path):
    """post-norm: d = rms̄_A · (r ⊘ γ)(|γ| は median·0.1 で下から切る)、ffn_down.scale は build_ffn の arch で割る。"""
    rng = np.random.default_rng(6)
    gamma = rng.uniform(0.5, 2.0, N_EMBD)
    gamma[3], gamma[7] = 1e-4, -1e-5                           # 小さい成分は下限で切る
    w = _weights(tmp_path, post_norm={1: gamma}, down_scale={1: 0.25})
    ds = _dirs(tmp_path)
    g, n = floor_gamma(gamma)
    fl = 0.1 * np.median(np.abs(gamma))
    assert n == 2 and g[3] == pytest.approx(fl) and g[7] == pytest.approx(-fl)
    from gne.keyed import post_norm_gamma
    gam = post_norm_gamma(w, 1, "ffn")
    np.testing.assert_allclose(gam, gamma.astype(np.float32))
    info = {}
    d = write_vector(w, ds, 1, rms_a=3.0, gamma=gam, info=info)
    np.testing.assert_allclose(d, 3.0 * ds.raw[1] / floor_gamma(gam)[0] / 0.25, rtol=1e-5)
    assert info["out_scale"] == pytest.approx(0.25) and info["gamma_floored"] == 2 and info["rms_a"] == 3.0
    with pytest.raises(GneError) as ei:
        write_vector(w, ds, 1, gamma=gam)                      # post-norm なのに rms_a が無い
    assert ei.value.code == "BAD_KEY"
    # 層 2 は post-norm も scale も無い
    assert post_norm_gamma(w, 2, "ffn") is None
    np.testing.assert_allclose(write_vector(w, ds, 2), ds.raw[2])
    # bitnet は W x に s を掛けてから LoRA を足すので割らない
    wb = _weights(tmp_path, name="b.gguf", arch="bitnet", down_scale={1: 0.25})
    np.testing.assert_allclose(write_vector(wb, ds, 1), ds.raw[1])
    # 確かめていない arch の scale は断る
    wq = _weights(tmp_path, name="q.gguf", arch="qwen2", down_scale={1: 0.25})
    with pytest.raises(GneError) as ei:
        write_vector(wq, ds, 1)
    assert ei.value.code == "ARCH_UNSUPPORTED"
    # post-norm の arch の attention の書き込みは確かめていないので断る
    wg = _weights(tmp_path, name="g.gguf", arch="gemma3", post_norm={1: gamma})
    with pytest.raises(GneError) as ei:
        post_norm_gamma(wg, 1, "attn")
    assert ei.value.code == "ARCH_UNSUPPORTED"


# --- LoRA の行列(V32)----------------------------------------------------------------------------------------

def test_v32_shapes_rank_and_merge(tmp_path):
    """V32: A (r, in)・B (out, r)。同じ (層, 鍵) は方向・符号が違っても 1 本に畳み、鍵が違えば積む。B@A = Σ α d κᵀ。"""
    w = _weights(tmp_path)
    d1, d2 = _dirs(tmp_path, "d1", seed=2), _dirs(tmp_path, "d2", seed=3)
    k1 = _key("ffn", [1, 2], N_FF, seed=5, name="k1")
    k2 = _key("ffn", [1], N_FF, seed=6, name="k2")
    ka = _key("attn", [1], N_HEAD * HEAD_DIM, seed=7, name="ka")
    keys, dirs = {"k1": k1, "k2": k2, "ka": ka}, {"d1": d1, "d2": d2}
    units = [_ku(1, 1.0, "d1", "k1"), _ku(1, -2.0, "d2", "k1"), _ku(1, 0.5, "d1", "k2"),
             _ku(2, 1.5, "d1", "k1"), _ku(1, 2.0, "d2", "ka", "attn")]
    tens = keyed_tensors(w, units, keys, dirs)
    assert set(tens) == {"blk.1.ffn_down.weight", "blk.2.ffn_down.weight", "blk.1.attn_output.weight"}
    A1, B1 = tens["blk.1.ffn_down.weight"]
    assert A1.shape == (2, N_FF) and B1.shape == (N_EMBD, 2) and A1.dtype == np.float32
    expect1 = (np.outer(1.0 * d1.raw[1] - 2.0 * d2.raw[1], k1.kappa[1]) + np.outer(0.5 * d1.raw[1], k2.kappa[1]))
    np.testing.assert_allclose(B1 @ A1, expect1, rtol=1e-5, atol=1e-6)
    A2, B2 = tens["blk.2.ffn_down.weight"]
    assert A2.shape == (1, N_FF) and B2.shape == (N_EMBD, 1)
    np.testing.assert_allclose(B2 @ A2, np.outer(1.5 * d1.raw[2], k1.kappa[2]), rtol=1e-5, atol=1e-6)
    Aa, Ba = tens["blk.1.attn_output.weight"]
    assert Aa.shape == (1, N_HEAD * HEAD_DIM) and Ba.shape == (N_EMBD, 1)
    # x = μ で α d を足し、側 B の部分空間では足さない
    np.testing.assert_allclose(B2 @ (A2 @ k1.mu[2].astype(np.float64)), 1.5 * d1.raw[2], rtol=1e-4)
    # alpha_ref で割る(アダプタの倍率で戻す)
    t2 = keyed_tensors(w, units, keys, dirs, alpha_ref=2.0)
    np.testing.assert_allclose(2.0 * t2["blk.1.ffn_down.weight"][1], B1, rtol=1e-6)


def test_keyed_tensors_refusals(tmp_path):
    w = _weights(tmp_path, n_layer=5, nextn=1)
    ds = {"d1": _dirs(tmp_path, n_layer=5)}
    k = _key("ffn", [1, 3, 4], N_FF)
    ka = _key("attn", [1], N_HEAD * HEAD_DIM)
    bad = _key("ffn", [1], N_FF + 1)
    keys = {"k": k, "ka": ka, "bad": bad}
    for units, code in (([_ku(3, 1, "d1", "k")], "BAD_RECIPE"),            # 最終層
                        ([_ku(4, 1, "d1", "k")], "BAD_RECIPE"),            # MTP
                        ([_ku(1, 1, "d1", "ka")], "BAD_RECIPE"),           # 鍵の書き込み先が違う
                        ([_ku(2, 1, "d1", "k")], "BAD_KEY"),               # 鍵にその層が無い
                        ([_ku(1, 1, "d1", "bad")], "BAD_KEY")):            # 長さが違う
        with pytest.raises(GneError) as ei:
            keyed_tensors(w, units, keys, ds)
        assert ei.value.code == code, units


def test_expert_and_shared_tensors(tmp_path):
    """専門家: A (E, r, in)・B (E, out, r)、専門家 e 以外のスライスは 0、r は専門家ごとの鍵の数の最大。
    shared expert は 2 次元。gate_a があれば B を割り、無ければ警告。"""
    w = _weights(tmp_path, n_layer=4, moe_layers=(1, 2), shexp=True)
    ds = {"d1": _dirs(tmp_path)}
    e0 = _key("expert:0", [1], N_FF_EXP, seed=8, name="e0")
    e0b = _key("expert:0", [1], N_FF_EXP, seed=9, name="e0b")
    e2 = _key("expert:2", [1], N_FF_EXP, seed=10, name="e2", gate=0.5)
    sh = _key("shared", [2], N_SHEXP, seed=11, name="sh")
    keys = {"e0": e0, "e0b": e0b, "e2": e2, "sh": sh}
    msgs = []
    units = [_ku(1, 1.0, "d1", "e0", "expert", 0), _ku(1, 2.0, "d1", "e0b", "expert", 0),
             _ku(1, 1.0, "d1", "e2", "expert", 2), _ku(2, 1.0, "d1", "sh", "shared")]
    tens = keyed_tensors(w, units, keys, ds, warn=msgs.append)
    A, B = tens["blk.1.ffn_down_exps.weight"]
    assert A.shape == (N_EXP, 2, N_FF_EXP) and B.shape == (N_EXP, N_EMBD, 2)
    r1 = ds["d1"].raw[1].astype(np.float64)
    np.testing.assert_allclose(B[0] @ A[0], np.outer(r1, e0.kappa[1]) + np.outer(2 * r1, e0b.kappa[1]), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(B[2] @ A[2], np.outer(r1 / 0.5, e2.kappa[1]), rtol=1e-5, atol=1e-6)   # gate_a で割る
    assert not A[1].any() and not B[1].any()
    assert not A[2, 1].any()                                    # 専門家 2 は 1 本だけ
    As, Bs = tens["blk.2.ffn_down_shexp.weight"]
    assert As.shape == (1, N_SHEXP) and Bs.shape == (N_EMBD, 1)
    assert any("ルーターの重み" in m for m in msgs)            # e0・e0b は gate_a なし


def test_keyed_units_from_recipe():
    """mode keyed のユニットだけを読み、default_direction・default_key を補い、方向名の先頭 - を α の符号に畳む。"""
    units = [SimpleNamespace(layer=19, kind="ffn", mode="keyed", k=1.25, direction=None, key=None, expert=None),
             SimpleNamespace(layer=20, kind="ffn", mode="keyed", k=0.5, direction="-tsun_r", key="tsun_k~3", expert=None),
             SimpleNamespace(layer=12, kind="expert", mode="keyed", k=1.0, direction=None, key=None, expert=4),
             SimpleNamespace(layer=5, kind="neuron", mode="add", k=None, direction=None, neuron=1)]
    rec = SimpleNamespace(units=units, default_direction="zunda_r", default_key="zunda_k")
    ku = keyed_units(rec)
    assert ku == [KeyedUnit(19, 1.25, "zunda_r", "zunda_k", "ffn", None),
                  KeyedUnit(20, -0.5, "tsun_r", "tsun_k~3", "ffn", None),
                  KeyedUnit(12, 1.0, "zunda_r", "zunda_k", "expert", 4)]
    assert ku[2].target == "expert:4" and ku[0].spec() == "19:ffn:keyed:a1.25@zunda_r#zunda_k"
    with pytest.raises(GneError):
        keyed_units(SimpleNamespace(units=units[:1], default_direction="zunda_r"))     # 鍵が無い
    with pytest.raises(GneError):
        keyed_units(SimpleNamespace(units=units[:1], default_key="zunda_k"))           # 方向が無い


# --- キャッシュ鍵 ---------------------------------------------------------------------------------------------

def test_cache_key_tracks_magnitude_and_content(tmp_path):
    w = _weights(tmp_path)
    d1 = _dirs(tmp_path, "d1", seed=2, base_layer=2)
    d1x2 = _dirs(tmp_path, "d1x2", seed=2, scale=2.0, base_layer=2)       # 向きは同じで大きさだけ 2 倍
    k1 = _key("ffn", [1, 2], N_FF, seed=5, name="k1")
    k1b = _key("ffn", [1, 2], N_FF, seed=6, name="k1")                     # 名前は同じで中身が違う
    dirs = {"d1": d1, "d1x2": d1x2}
    units = [_ku(1, 1.0, "d1", "k1"), _ku(2, 0.5, "d1", "k1")]
    h, ref = keyed_cache_key(w, units, {"k1": k1}, dirs)
    assert ref == 1.0 and len(h) == 16
    h3, ref3 = keyed_cache_key(w, [_ku(1, 3.0, "d1", "k1"), _ku(2, 1.5, "d1", "k1")], {"k1": k1}, dirs)
    assert h3 == h and ref3 == 3.0                                         # 全体を 3 倍: 同じファイル、倍率 3
    assert keyed_cache_key(w, [_ku(1, 1.0, "d1", "k1"), _ku(2, 0.6, "d1", "k1")], {"k1": k1}, dirs)[0] != h
    assert keyed_cache_key(w, [_ku(1, 1.0, "d1x2", "k1"), _ku(2, 0.5, "d1x2", "k1")], {"k1": k1}, dirs)[0] != h
    assert keyed_cache_key(w, units, {"k1": k1b}, dirs)[0] != h
    assert keyed_cache_key(w, units, {"k1": k1}, dirs, direction_layer="base")[0] != h
    assert keyed_cache_key(w, units, {"k1": k1.pseudo(0) if k1.n_pseudo else k1}, dirs)[0] == h
    kp = _key("ffn", [1, 2], N_FF, seed=5, pseudo=2, name="k1")
    assert keyed_cache_key(w, units, {"k1": kp.pseudo(1)}, dirs)[0] != keyed_cache_key(w, units, {"k1": kp}, dirs)[0]
    # post-norm の補正値(rms_a)が変われば変わる
    wp = _weights(tmp_path, name="p.gguf", post_norm={1: np.ones(N_EMBD), 2: np.ones(N_EMBD)})
    ka, kb = _key("ffn", [1, 2], N_FF, seed=5, rms=2.0), _key("ffn", [1, 2], N_FF, seed=5, rms=3.0)
    assert keyed_cache_key(wp, units, {"k1": ka}, dirs)[0] != keyed_cache_key(wp, units, {"k1": kb}, dirs)[0]
    # 置き場所
    store = Store(tmp_path / "root")
    p = keyed_lora_path(store, h, model_id="m-1")
    assert p == tmp_path / "root" / "keys" / "m-1" / "lora" / f"keyed-{h}.lora.gguf"


# --- 鍵のファイル -----------------------------------------------------------------------------------------------

def test_keyset_roundtrip_and_pseudo(tmp_path):
    store = Store(tmp_path / "root")
    sess = SimpleNamespace(store=store, meta=SimpleNamespace(model_id="m-1"))
    ks = _key("ffn", [3, 5], N_FF, pseudo=4, rms=2.5, name="")
    ks.exa = {3: np.ones((2, N_FF), np.float32)}
    ks.exb = {3: np.zeros((3, N_FF), np.float32)}
    ks.q_mode, ks.positions, ks.skip, ks.max_tokens, ks.subset = "samples", "before:のだ", 2, 64, "even"
    ks.contrast_sha = "abc123"
    ks.hashes_a, ks.hashes_b = [example_hash(None, "やあ")], [example_hash(None, "やあ"), example_hash(None, "こんにちは")]
    ks.counts = {"a": 1, "b": 2, "rows_a": 10, "rows_b": 30}
    ks.info = {"q_energy": 0.99, "dont_touch": [{"system": None, "n": 3}]}
    ks.model_id = "m-1"
    p = ks.save(key_dir(sess) / "zunda_k.npz")
    assert p == tmp_path / "root" / "keys" / "m-1" / "zunda_k.npz" and ks.name == "zunda_k"
    got = load_key(sess, "zunda_k")
    assert got.name == "zunda_k" and got.target == "ffn" and got.layers == [3, 5] and got.in_dim == N_FF
    for L in (3, 5):
        np.testing.assert_array_equal(got.kappa_of(L), ks.kappa[L])
        np.testing.assert_array_equal(got.mu[L], ks.mu[L])
        assert got.kn[L] == ks.kn[L] and got.gain[L] == ks.gain[L] and got.q_rank[L] == ks.q_rank[L]
        assert got.rms_a[L] == 2.5
    np.testing.assert_array_equal(got.exa[3], ks.exa[3])
    assert got.exb[3].shape == (3, N_FF) and 5 not in got.exa
    assert (got.q_mode, got.positions, got.skip, got.max_tokens, got.subset) == ("samples", "before:のだ", 2, 64, "even")
    assert got.contrast_sha == "abc123" and got.hashes_b == ks.hashes_b and got.counts == ks.counts
    assert got.info == ks.info and got.model_id == "m-1"
    assert got.sha == ks.sha and got.n_pseudo == 4
    # 擬似鍵 NAME~j
    pj = load_key(sess, "zunda_k~2")
    np.testing.assert_array_equal(pj.kappa_of(3), ks.kappa_ctrl[3][2])
    assert pj.spec == "zunda_k~2" and pj.sha == ks.sha + "~2" and pj.base_sha == ks.sha
    np.testing.assert_array_equal(got.kappa_of(5, ctrl=1), ks.kappa_ctrl[5][1])
    for bad in ("zunda_k~4", "zunda_k~99"):
        with pytest.raises(GneError) as ei:
            load_key(sess, bad)
        assert ei.value.code == "BAD_KEY"
    # Store と model_id でも引ける、キャッシュは更新時刻で読み直す
    assert load_key(store, "zunda_k", model_id="m-1") is got
    ks2 = _key("ffn", [3], N_FF, seed=42)
    ks2.save(p)
    again = load_key(sess, "zunda_k")
    assert again is not got and again.layers == [3]
    with pytest.raises(GneError) as ei:
        load_key(sess, "nope")
    assert ei.value.code == "FILE_NOT_FOUND"
    nop = _key("ffn", [3], N_FF, seed=1)
    nop.save(key_dir(sess) / "plain.npz")
    with pytest.raises(GneError) as ei:
        load_key(sess, "plain~0")                               # 擬似鍵の無い鍵
    assert ei.value.code == "BAD_KEY"
    names = [r["name"] for r in list_keys(sess)]
    assert names == ["plain", "zunda_k"]
    # npz に手元のパスを入れない
    z = np.load(p)
    assert not any("root" in str(z[k]) for k in z.files if z[k].dtype.kind == "U")


def test_keyset_validate():
    ks = _key("ffn", [1], N_FF)
    ks.rms_a[7] = 1.0                                           # κ の無い層
    with pytest.raises(GneError):
        ks.validate()
    with pytest.raises(GneError):
        KeySet(target="mlp", kappa={1: np.ones(3)}, mu={1: np.ones(3)}, kn={1: 1.0}, gain={1: 1.0}).validate()


# --- 書き出し -------------------------------------------------------------------------------------------------

def test_export_keyed_lora(tmp_path):
    from gguf import GGUFReader
    import json
    w = _weights(tmp_path)
    ds = _dirs(tmp_path, "zunda_r")
    k = _key("ffn", [1, 2], N_FF, name="zunda_k")
    units = [SimpleNamespace(layer=1, kind="ffn", mode="keyed", k=2.0, direction=None, key=None, expert=None),
             SimpleNamespace(layer=2, kind="ffn", mode="keyed", k=-1.0, direction=str(tmp_path / "zunda_r.npz"),
                             key=None, expert=None)]

    class _R(SimpleNamespace):
        def to_dict(self):
            return {"name": self.name, "default_direction": self.default_direction, "default_key": self.default_key,
                    "units": [dict(vars(u)) for u in self.units]}
    rec = _R(units=units, name="zunda", default_direction="zunda_r", default_key="zunda_k", direction_layer="own")
    dirs = {"zunda_r": ds, str(tmp_path / "zunda_r.npz"): ds}
    out = tmp_path / "out" / "zunda-keyed.lora.gguf"
    rep = export_keyed_lora(w, rec, out, {"zunda_k": k}, dirs, alpha_ref=2.0)
    assert rep["alpha_ref"] == 2.0 and rep["keyed_rank"] == {"blk.1.ffn_down.weight": 1, "blk.2.ffn_down.weight": 1}
    assert rep["keyed_layers"]["1"]["keys"] == ["zunda_k"] and rep["keyed_layers"]["1"]["rank"] == 1
    r = GGUFReader(str(out))
    names = {t.name for t in r.tensors}
    assert names == {f"blk.{L}.ffn_down.weight.lora_{x}" for L in (1, 2) for x in "ab"}
    assert "adapter.lora.alpha" not in r.fields
    t = {t.name: np.asarray(t.data, dtype=np.float32) for t in r.tensors}
    A = t["blk.2.ffn_down.weight.lora_a"].reshape(1, N_FF)
    B = t["blk.2.ffn_down.weight.lora_b"].reshape(N_EMBD, 1)
    np.testing.assert_allclose(2.0 * (B @ A), np.outer(-1.0 * ds.raw[2], k.kappa[2]), rtol=1e-5, atol=1e-6)
    meta = json.loads(r.fields["gne.keyed"].contents())
    assert meta["alpha_ref"] == 2.0 and [u["alpha"] for u in meta["units"]] == [2.0, -1.0]
    assert meta["units"][0]["key"] == "zunda_k" and meta["units"][0]["key_sha"] == k.sha
    text = json.dumps(meta, ensure_ascii=False) + r.fields["gne.recipe"].contents()
    assert str(tmp_path) not in text and "/" not in json.loads(r.fields["gne.recipe"].contents())["units"][1]["direction"]
    m2 = keyed_meta(w, rec, {"zunda_k": k}, dirs)
    assert m2["units"][1]["direction"] == "zunda_r"
    with pytest.raises(GneError):
        export_keyed_lora(w, _R(units=[], name="x", default_direction=None, default_key=None), tmp_path / "x.gguf",
                          {}, {})
    assert Path(out).exists()


def test_with_recipe_notation(tmp_path):
    """レシピの記法(WP1)の Unit・Recipe から: 方向名の - は α の符号、擬似鍵 NAME~j、default_direction・default_key。"""
    import json

    from gguf import GGUFReader

    from gne.steer import Recipe, parse_unit
    if not hasattr(Recipe, "default_key"):
        pytest.skip("レシピの記法(WP1)がまだありません")
    w = _weights(tmp_path)
    ds = _dirs(tmp_path, "zunda_r")
    store = Store(tmp_path / "root")
    ks = _key("ffn", [1, 2], N_FF, pseudo=2, name="zunda_k")
    ks.save(key_dir(store, "m-1") / "zunda_k.npz")
    rec = Recipe([parse_unit("1:ffn:keyed:a2@-zunda_r"), parse_unit("2:ffn:keyed:a1#zunda_k~1")], name="zunda",
                 default_direction="zunda_r", default_key="zunda_k")
    key_of = lambda n: load_key(store, n, model_id="m-1")          # noqa: E731
    tens = keyed_tensors(w, rec, key_of, {"zunda_r": ds})
    A1, B1 = tens["blk.1.ffn_down.weight"]
    np.testing.assert_allclose(B1 @ A1, np.outer(-2.0 * ds.raw[1], ks.kappa[1]), rtol=1e-5, atol=1e-6)
    A2, B2 = tens["blk.2.ffn_down.weight"]
    np.testing.assert_allclose(B2 @ A2, np.outer(ds.raw[2], ks.kappa_ctrl[2][1]), rtol=1e-5, atol=1e-6)
    h, ref = keyed_cache_key(w, rec, key_of, {"zunda_r": ds})
    assert ref == 2.0 and keyed_cache_key(w, rec.scaled(0.5), key_of, {"zunda_r": ds}) == (h, 1.0)
    out = tmp_path / "zunda.lora.gguf"
    export_keyed_lora(w, rec, out, key_of, {"zunda_r": ds}, alpha_ref=ref)
    r = GGUFReader(str(out))
    assert json.loads(r.fields["gne.recipe"].contents())["default_key"] == "zunda_k"
    assert [u["key"] for u in json.loads(r.fields["gne.keyed"].contents())["units"]] == ["zunda_k", "zunda_k~1"]
