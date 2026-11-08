"""CHANGE-07 MoE の keyed write のテスト(shared expert の ffn_down_shexp と、専門家 e の ffn_down_exps)。

モデルなし: 小さな GGUF(qwen2moe / granitemoe arch、エンジンは読まない)で、keyed_build.SharedCapture・ExpertCapture の
捕捉名と行の絞り方(専門家 e が選ばれたトークンの枠だけ、ゲートはその枠のルーターの重み)、MoE でない層・範囲外の専門家・
post-norm の MoE・.scale を断ること、granite 系の residual_scale と gate_a で B を割ること、専門家の 3 次元テンソルの焼き込みが
スラブごとに書くこと(テンソル全体を float32 にしない、keep なら編集しないスラブはバイトのまま、大きさの増え方の記録と警告)。
出力のスキーマ(MoE の鍵の層の行・V30 の専門家の項目・書き出しの専門家の記録と大きさの増分・専門家の選ばれ方)、分割 GGUF への
焼き込みもスラブごと(keep は元の型でその場に、f16 は通常の焼き込みとバイト一致)、CLI の説明と key-experts・MCP の入口、
keyed の見積もり(eval の選択性の分と、鍵の作成・確かめ・専門家の選ばれ方のジョブ)。
モデルあり:
- granite-3.1-1b-a400m(GNE_TEST_MOE、granitemoe、shared expert なし、CPU で回せる): configs/contrast/zundamon_endings.yaml の
  作り方 (b)(与えた返事の「です。」「ます。」を書く直前)で、側 A の位置でよく選ばれる専門家(expert_usage)の expert:e の鍵を
  作り、保存項目・key-check(保留集合で擬似鍵を超える)・V30(実行時の 3 次元 LoRA の差 = α (κᵀx) d、ほかの専門家の枠は 0)・
  KeyedMetrics(gain の対照は除外)・焼き込み(F16 と keep)を llama.cpp で読んで実行時の LoRA と合うこと。
- Qwen1.5-MoE-A2.7B(GNE_TEST_MOE_SHARED を渡したときだけ。9.5 GB なので焼き込みはしない): shared の鍵(ゲートあり)と
  expert:e の鍵の作成・key-check・V30・実行時の LoRA。
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gguf import GGMLQuantizationType as T
from gguf import quants as gquants

from gne import keyed as K
from gne.keyed import KeySet, key_dir, make_key, orth_basis
from gne.keyed_build import ExpertCapture, SharedCapture, target_capture
from gne.steer import GneError

ROOT = Path(__file__).resolve().parent.parent
MOE = Path(os.environ.get("GNE_TEST_MOE", ROOT / "models" / "granite-3.1-1b-a400m-instruct-Q8_0.gguf"))
MOE_SHARED = os.environ.get("GNE_TEST_MOE_SHARED")         # 渡したときだけ(大きい)
NGL = int(os.environ.get("GNE_TEST_NGL", "-1"))
ZUNDAMON = ROOT / "configs" / "contrast" / "zundamon_endings.yaml"

# 小さな MoE: 層 0 は dense、層 1・2 は MoE(層 1 に shared expert とそのゲート)。専門家の入力は Q8_0 の塊(32)の倍数
N_EMBD, N_FF, N_HEAD, HEAD_DIM, N_EXP, N_USED, N_FF_EXP, N_SHEXP = 16, 32, 4, 4, 4, 2, 32, 32


def _moe_gguf(path, *, arch="qwen2moe", exps_type=T.F32, residual_scale=None, gate_shexp=True, exps_scale=False, split=0):
    """小さな MoE の GGUF。split > 0 なら 1 ファイル split 個までのテンソルで分割して書く(1 番目はメタデータだけ)。"""
    from gguf import GGUFWriter
    rng = np.random.default_rng(1)
    kw = dict(split_max_tensors=split, small_first_shard=True) if split else {}
    w = GGUFWriter(str(path), arch=arch, **kw)
    w.add_name("tiny-moe")
    w.add_block_count(3)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(2)
    w.add_expert_count(N_EXP)
    w.add_expert_used_count(N_USED)
    if residual_scale is not None:
        w.add_float32(f"{arch}.residual_scale", residual_scale)
    w.add_token_list([f"t{i}" for i in range(8)])
    mats = {}
    for L in range(3):
        shapes = {"attn_q": (N_HEAD * HEAD_DIM, N_EMBD), "attn_output": (N_EMBD, N_HEAD * HEAD_DIM)}
        if L == 0:
            shapes.update({"ffn_down": (N_EMBD, N_FF), "ffn_up": (N_FF, N_EMBD)})
        else:
            shapes.update({"ffn_gate_inp": (N_EXP, N_EMBD), "ffn_down_exps": (N_EXP, N_EMBD, N_FF_EXP)})
            if L == 1:
                shapes["ffn_down_shexp"] = (N_EMBD, N_SHEXP)
        for nm, shape in shapes.items():
            m = (rng.standard_normal(shape) * 0.1).astype(np.float32)
            name = f"blk.{L}.{nm}.weight"
            if nm == "ffn_down_exps" and exps_type != T.F32:
                q = gquants.quantize(m, exps_type)
                mats[name] = gquants.dequantize(q, exps_type).astype(np.float32).reshape(shape)
                w.add_tensor(name, q, raw_dtype=exps_type)
            else:
                mats[name] = m
                w.add_tensor(name, m)
        if L == 1 and gate_shexp:
            w.add_tensor("blk.1.ffn_gate_inp_shexp.weight", rng.standard_normal(N_EMBD).astype(np.float32))
        if L == 1 and exps_scale:
            w.add_tensor("blk.1.ffn_down_exps.scale", np.ones(N_EXP, dtype=np.float32))
    w.add_tensor("token_embd.weight", rng.standard_normal((8, N_EMBD)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return mats


def _session(tmp_path, **kw):
    """小さな MoE の Session(エンジンは読まない)と方向 d1(全層)。split なら分割した 1 番目のファイルを開く。"""
    from gne.api import Session
    mats = _moe_gguf(tmp_path / "m.gguf", **kw)
    first = next(iter(sorted(tmp_path.glob("m-00001-of-*.gguf")))) if kw.get("split") else tmp_path / "m.gguf"
    s = Session(first, root=tmp_path / "root")
    rng = np.random.default_rng(2)
    raw = {L: rng.standard_normal(N_EMBD).astype(np.float32) for L in range(3)}
    np.savez(s.store.direction_dir(s.meta.model_id) / "d1.npz", layers=np.array(sorted(raw), dtype=np.int32),
             **{f"layer_{L}": v for L, v in raw.items()})
    return s, mats, raw


def _key(s, name, target, layers, in_dim, *, seed=5, n_b=4, gate=None):
    rng = np.random.default_rng(seed)
    ks = KeySet(target=target, kappa={}, mu={}, kn={}, gain={}, model_id=s.meta.model_id)
    for L in layers:
        Bm = rng.standard_normal((n_b, in_dim))
        mu = rng.standard_normal(in_dim)
        kap, info = make_key(mu, orth_basis(Bm, energy=1.0))
        ks.kappa[L], ks.mu[L] = kap.astype(np.float32), mu.astype(np.float32)
        ks.kn[L], ks.gain[L], ks.q_rank[L] = info["kn"], info["gain"], info["q_rank"]
        ks.exb[L] = Bm.astype(np.float32)
        if gate is not None:
            ks.gate_a[L] = float(gate)
    ks.save(key_dir(s) / f"{name}.npz")
    return ks


# --- モデル不要 ----------------------------------------------------------------------------------------

def test_shared_capture_names_rows_and_gate(tmp_path):
    """shared: MoE の層の build_ffn の活性(ffn_swiglu)が ffn_down_shexp の入力。ゲート ffn_shexp_gate(qwen2moe)か
    shared_expert_gate_sigmoid(qwen35moe など)をトークンごとの gate に返す。幅が違えば ARCH_UNSUPPORTED。"""
    s, _, _ = _session(tmp_path)
    tc = target_capture(s, "shared", [1])
    assert isinstance(tc, SharedCapture) and tc.in_dim == {1: N_SHEXP} and tc.gated == {1: True}
    seen = {"ffn_swiglu-1", "ffn_shexp_gate-1", "ffn_moe_argsort-1", "ffn_moe_swiglu-1", "ffn_shexp-1"}
    tc.bind(seen)
    assert tc.names(1) == ["ffn_swiglu-1", "ffn_shexp_gate-1"]
    tc.bind(seen - {"ffn_shexp_gate-1"} | {"shared_expert_gate_sigmoid-1"})
    assert tc.names(1) == ["ffn_swiglu-1", "shared_expert_gate_sigmoid-1"]
    Tn = 6
    x = np.arange(Tn * N_SHEXP, dtype=np.float32).reshape(Tn, N_SHEXP)
    g = np.linspace(0.1, 0.6, Tn, dtype=np.float32).reshape(Tn, 1)
    sel = np.array([1, 4])
    rows, gate = tc.rows({"ffn_swiglu-1": x, "shared_expert_gate_sigmoid-1": g}, 1, sel)
    np.testing.assert_array_equal(rows, x[sel])
    np.testing.assert_allclose(gate, g[sel, 0])
    with pytest.raises(GneError) as ei:                                   # 幅が違う = shared expert の活性ではない
        tc.rows({"ffn_swiglu-1": x[:, :7], "shared_expert_gate_sigmoid-1": g}, 1, sel)
    assert ei.value.code == "ARCH_UNSUPPORTED"
    with pytest.raises(GneError) as ei:                                   # MoE の捕捉名が無い層
        tc.bind({"ffn_swiglu-1"})
    assert ei.value.code == "ARCH_UNSUPPORTED"
    # ゲートの捕捉名が見えなければ警告して gate なし(gate_a を保存しない)
    s.warnings.clear()
    tc.bind({"ffn_swiglu-1", "ffn_moe_argsort-1"})
    assert tc.names(1) == ["ffn_swiglu-1"] and any("gate_a" in w for w in s.warnings)
    assert tc.rows({"ffn_swiglu-1": x}, 1, sel)[1] is None
    # shared expert の無い層・dense の層は BAD_RECIPE
    for L in (0, 2):
        with pytest.raises(GneError) as ei:
            target_capture(s, "shared", [L])
        assert ei.value.code == "BAD_RECIPE"


def test_expert_capture_selects_rows_of_expert(tmp_path):
    """expert:e: 選んだ位置のうち専門家 e が選ばれたトークンの枠だけを行にし、その枠のルーターの重みを gate に返す。
    捕捉の (T·n_used, 幅) はトークンの行 (T, n_used·幅) にそろえる。"""
    s, _, _ = _session(tmp_path)
    tc = target_capture(s, "expert:2", [1, 2])
    assert isinstance(tc, ExpertCapture) and tc.target == "expert:2" and tc.in_dim == {1: N_FF_EXP, 2: N_FF_EXP}
    seen = {f"{b}-{L}" for L in (1, 2) for b in ("ffn_moe_swiglu", "ffn_moe_argsort", "ffn_moe_weights",
                                                   "ffn_moe_weights_norm", "ffn_moe_down")}
    tc.bind(seen)
    assert tc.names(1) == ["ffn_moe_swiglu-1", "ffn_moe_argsort-1", "ffn_moe_weights_norm-1"]   # 実効の重み(正規化の後)
    Tn = 5
    act3 = np.arange(Tn * N_USED * N_FF_EXP, dtype=np.float32).reshape(Tn, N_USED, N_FF_EXP)
    wts = np.linspace(0.05, 0.95, Tn * N_USED, dtype=np.float32).reshape(Tn, N_USED)
    arg = np.array([[2, 0, 1, 3], [1, 3, 2, 0], [0, 1, 2, 3], [3, 2, 0, 1], [1, 0, 3, 2]], dtype=np.int32)
    got = {"ffn_moe_swiglu-1": tc.token_rows("ffn_moe_swiglu-1", act3.reshape(Tn * N_USED, N_FF_EXP), Tn),
           "ffn_moe_weights_norm-1": tc.token_rows("ffn_moe_weights_norm-1", wts.reshape(Tn * N_USED, 1), Tn),
           "ffn_moe_argsort-1": tc.token_rows("ffn_moe_argsort-1", arg, Tn)}
    assert got["ffn_moe_swiglu-1"].shape == (Tn, N_USED * N_FF_EXP) and got["ffn_moe_weights_norm-1"].shape == (Tn, N_USED)
    # 専門家 2 が選ばれた(先頭 n_used = 2 列にある)のはトークン 0(枠 0)と 3(枠 1)。トークン 1・2 は 3 列目以降なので選ばれていない
    rows, gate = tc.rows(got, 1, np.array([0, 1, 2, 3, 4]))
    np.testing.assert_array_equal(rows, np.stack([act3[0, 0], act3[3, 1]]))
    np.testing.assert_allclose(gate, [wts[0, 0], wts[3, 1]])
    rows, gate = tc.rows(got, 1, np.array([1, 2, 4]))                      # 選んだ位置に専門家 2 が無ければ 0 行
    assert rows.shape == (0, N_FF_EXP) and gate.shape == (0,)
    with pytest.raises(GneError) as ei:
        tc.bind(seen - {"ffn_moe_argsort-2"})
    assert ei.value.code == "ARCH_UNSUPPORTED" and "argsort" in ei.value.message
    for t, L, code in (("expert:9", 1, "BAD_RECIPE"), ("expert:1", 0, "BAD_RECIPE")):   # 範囲外の専門家・dense の層
        with pytest.raises(GneError) as ei:
            target_capture(s, t, [L])
        assert ei.value.code == code, (t, L)


def test_route_capture_for_expert_usage(tmp_path):
    """expert_usage の捕捉: 選ばれた専門家を 1 にした行と、重みを散らした行(例ごとの平均 = 選ばれた割合・重みの平均)。"""
    from gne.keyed_build import _RouteCapture
    s, _, _ = _session(tmp_path)
    tc = _RouteCapture(s, [1])
    tc.bind({"ffn_moe_argsort-1", "ffn_moe_weights-1"})
    arg = np.array([[2, 0, 1, 3], [1, 3, 2, 0]], dtype=np.int32)
    wts = np.array([[0.7, 0.3], [0.6, 0.4]], dtype=np.float32)
    x, gate = tc.rows({"ffn_moe_argsort-1": arg, "ffn_moe_weights-1": tc.token_rows("ffn_moe_weights-1", wts.reshape(4, 1), 2)},
                      1, np.array([0, 1]))
    assert gate is None and x.shape == (2, 2 * N_EXP)
    np.testing.assert_array_equal(x[:, :N_EXP], [[1, 0, 1, 0], [0, 1, 0, 1]])
    np.testing.assert_allclose(x[:, N_EXP:], [[0.3, 0, 0.7, 0], [0, 0.6, 0, 0.4]])


def test_moe_refusals():
    """post-norm のある MoE、ffn_down_exps / ffn_down_shexp の .scale は ARCH_UNSUPPORTED(補正を確かめていない)。"""
    for kind in ("shared", "expert"):
        with pytest.raises(GneError) as ei:
            K.post_norm_gamma(SimpleNamespace(meta=SimpleNamespace(arch="gemma2")), 1, kind)
        assert ei.value.code == "ARCH_UNSUPPORTED"
        w = SimpleNamespace(meta=SimpleNamespace(arch="qwen2moe"), has_tensor=lambda n: n.startswith("blk.1.post_ffw_norm"))
        with pytest.raises(GneError) as ei:
            K.post_norm_gamma(w, 1, kind)
        assert ei.value.code == "ARCH_UNSUPPORTED"


def test_exps_scale_refused(tmp_path):
    s, _, _ = _session(tmp_path, exps_scale=True)
    with pytest.raises(GneError) as ei:
        target_capture(s, "expert:0", [1])
    assert ei.value.code == "ARCH_UNSUPPORTED" and ".scale" in ei.value.message


def test_write_vector_divides_gate_and_residual_scale(tmp_path):
    """B の向きは d = r_L / (residual_scale · gate_a)。granite 系は枝の出力に residual_scale を掛けてから残差に足すので割る
    (無い arch は 1)。LoRA は専門家 e のスライスだけ、B_e A_e = α d κᵀ。"""
    s, _, raw = _session(tmp_path, arch="granitemoe", residual_scale=0.25)
    ke = _key(s, "ke", "expert:1", [1], N_FF_EXP, gate=0.4)
    assert K.residual_scale(s.weights) == pytest.approx(0.25)
    info = {}
    d = K.write_vector(s.weights, s.direction_set("d1"), 1, kind="expert", expert=1, gate=0.4, info=info)
    np.testing.assert_allclose(d, raw[1] / (0.25 * 0.4), rtol=1e-6)
    assert info["res_scale"] == pytest.approx(0.25) and info["gate"] == pytest.approx(0.4) and info["out_scale"] == 1.0
    r = s.recipe(units=["1:E1:keyed:a2@d1#ke"], name="ge")
    kb = K.keyed_bundle(s.weights, r, s._key_of, s.direction_set)
    A, B = kb["tensors"]["blk.1.ffn_down_exps.weight"]
    assert A.shape == (N_EXP, 1, N_FF_EXP) and B.shape == (N_EXP, N_EMBD, 1)
    assert not np.any(A[[0, 2, 3]]) and not np.any(B[[0, 2, 3]])
    np.testing.assert_allclose(B[1] @ A[1], 2.0 * np.outer(raw[1] / 0.1, ke.kappa[1]), rtol=1e-5, atol=1e-6)
    assert kb["meta"]["units"][0]["res_scale"] == pytest.approx(0.25)
    # residual_scale の無い arch は 1(qwen2moe)
    (tmp_path / "q").mkdir()
    s2, _, _ = _session(tmp_path / "q")
    assert K.residual_scale(s2.weights) == 1.0


@pytest.mark.parametrize("src", [T.Q8_0, T.Q5_0])
def test_expert_bake_slab_by_slab(tmp_path, monkeypatch, src):
    """専門家の 3 次元テンソルの焼き込みはスラブごと: テンソル全体を float32 にしない。f16 は全スラブを F16 にする
    (大きさが増えるので記録と警告)、keep は元の型で書ける型なら元の型で、編集しないスラブはバイトのまま。編集した
    スラブは W_e + α d κᵀ(書き出す型の丸めまで)。requant_error は編集したスラブだけで測る。"""
    from gguf import GGUFReader

    import gne.gain as G
    from gne.weights import _slab_to_f32
    s, mats, raw = _session(tmp_path, exps_type=src)
    ke = _key(s, "ke", "expert:2", [1], N_FF_EXP)
    name = "blk.1.ffn_down_exps.weight"
    orig_base = G._writer_base

    def no_whole(weights, n):                                    # 専門家のテンソル全体を float32 にしたら失敗
        assert not n.endswith("ffn_down_exps.weight"), n
        return orig_base(weights, n)
    monkeypatch.setattr(G, "_writer_base", no_whole)
    r = s.recipe(units=["1:E2:keyed:a3@d1#ke"], name="eb")
    delta = 3.0 * np.outer(raw[1].astype(np.float64), ke.kappa[1].astype(np.float64))
    W = mats[name]
    src_t = s.weights.tensor(name)
    # 既定(auto): 元が Q8_0 なら Q8_0 のまま(編集しないスラブはバイトのまま)、それ以外(Q5_0)は F16
    s.warnings.clear()
    a = s.gain(r, tmp_path / "auto.gguf", bake=True)
    rec = a["modified"][0]
    assert a["requant"] == "auto" and "fallback" not in rec and rec["edit_kept"] >= 0.9
    if src == T.Q8_0:
        assert rec["to"] == "Q8_0" and rec["untouched"] == "raw" and a["expert_bytes_added"] == 0
    else:
        assert rec["to"] == "F16" and rec["untouched"] == f"{src.name}->F16" and a["expert_bytes_added"] > 0
    (tmp_path / "auto.gguf").unlink()
    # f16
    s.warnings.clear()
    b = s.gain(r, tmp_path / "f16.gguf", bake=True, requant="f16")
    rec = b["modified"][0]
    assert rec["tensor"] == name and rec["from"] == src.name and rec["to"] == "F16" and rec["experts"] == [2]
    assert rec["untouched"] == f"{src.name}->F16" and rec["bytes_from"] == int(src_t.n_bytes)
    assert rec["bytes_to"] == N_EXP * N_EMBD * N_FF_EXP * 2 > rec["bytes_from"]
    assert rec["requant_error"] < 0.01
    assert any("MiB 大きくなりました" in w and "--requant keep" in w for w in s.warnings)
    t = {x.name: x for x in GGUFReader(tmp_path / "f16.gguf").tensors}[name]
    got = np.array(t.data, dtype=np.float32).reshape(N_EXP, N_EMBD, N_FF_EXP)
    for e in (0, 1, 3):
        np.testing.assert_array_equal(got[e], W[e].astype(np.float16).astype(np.float32))
    np.testing.assert_allclose(got[2], W[2] + delta, atol=2e-3)
    (tmp_path / "f16.gguf").unlink()
    # keep: 元の型のまま(大きさは変わらない)、編集しないスラブはバイトのまま
    s.warnings.clear()
    k = s.gain(r, tmp_path / "keep.gguf", bake=True, requant="keep")
    rec = k["modified"][0]
    assert rec["to"] == src.name and rec["untouched"] == "raw" and rec["bytes_to"] == rec["bytes_from"]
    assert 0 < rec["requant_error"] < 0.5 and not any("大きくなりました" in w for w in s.warnings)
    tk = {x.name: x for x in GGUFReader(tmp_path / "keep.gguf").tensors}[name]
    for e in (0, 1, 3):
        assert np.array_equal(np.asarray(tk.data[e]), np.asarray(src_t.data[e]))
    edited = _slab_to_f32(tk, 2).astype(np.float64)
    want = W[2].astype(np.float64) + delta
    assert np.linalg.norm(edited - want) / np.linalg.norm(delta) == pytest.approx(rec["requant_error"], rel=1e-3)
    side = json.loads(Path(str(tmp_path / "keep.gguf") + ".headproj.json").read_text())
    assert side["modified"][0]["experts"] == [2]
    (tmp_path / "keep.gguf").unlink()


def test_expert_auto_falls_back_to_f16_for_tiny_edit(tmp_path):
    """専門家のテンソルも既定(auto)は Q8_0 のまま。ただし Q8_0 の刻みに埋もれる小さな編集(edit_kept < 0.9)だけ F16 にする
    (テンソル全体が F16 になって大きくなるので、大きさの警告も出る)。分割 GGUF への焼き込みも同じ。"""
    s, mats, raw = _session(tmp_path, exps_type=T.Q8_0)
    _key(s, "ke", "expert:2", [1], N_FF_EXP)
    r = s.recipe(units=["1:E2:keyed:a3e-4@d1#ke"], name="et")
    s.warnings.clear()
    d = s.gain(r, tmp_path / "t.gguf", bake=True)
    rec = d["modified"][0]
    assert rec["to"] == "F16" and rec["fallback"]["from"] == "Q8_0" and rec["fallback"]["edit_kept"] < 0.9
    assert d["expert_bytes_added"] > 0 and any("F16 で書きました" in w for w in s.warnings)
    assert any("大きくなりました" in w for w in s.warnings)
    (tmp_path / "sp").mkdir()
    s2, _, _ = _session(tmp_path / "sp", exps_type=T.Q8_0, split=4)
    _key(s2, "ke", "expert:2", [1], N_FF_EXP)
    f = s2.headproj_export(s2.recipe(units=["1:E2:keyed:a3e-4@d1#ke"], name="et"), tmp_path / "out" / "et.gguf", bake=True)
    side = json.loads(Path(f["sidecar"]).read_text(encoding="utf-8"))
    assert f["fallback"] == ["blk.1.ffn_down_exps.weight"] and side["modified"][0]["to"] == "F16"
    k = s2.headproj_export(s2.recipe(units=["1:E2:keyed:a3@d1#ke"], name="ek"), tmp_path / "out2" / "ek.gguf", bake=True)
    side = json.loads(Path(k["sidecar"]).read_text(encoding="utf-8"))
    assert "fallback" not in k and side["modified"][0]["to"] == "Q8_0" and side["modified"][0]["untouched"] == "raw"
    assert k["expert_bytes_added"] == 0


def test_expert_qtype_rules():
    """keep の型: gguf-py で量子化できる型(Q8_0・Q5_0・Q4_0・F16 など)は元の型、できない型(K-quant・IQ)は Q8_0。"""
    from gne.gain import _expert_qtype
    assert _expert_qtype(T.Q5_0, "keep") == T.Q5_0 and _expert_qtype(T.Q8_0, "keep") == T.Q8_0
    assert _expert_qtype(T.F16, "keep") == T.F16
    assert _expert_qtype(T.Q4_K, "keep") == T.Q8_0 and _expert_qtype(T.IQ2_XS, "keep") == T.Q8_0
    assert _expert_qtype(T.Q4_K, "f16") == T.F16 and _expert_qtype(T.Q5_0, "q8_0") == T.Q8_0


ENV_STUB = {"schema_version": "1", "command": "x", "model_id": "m", "llama_cpp_version": "v", "seed": 0,
            "generated_at": "2026-11-08T00:00:00+00:00", "warnings": []}


def test_schema_moe_fields():
    """MoE の鍵でだけ増える項目(鍵の層の行・V30 の専門家の項目・専門家の選ばれ方)をスキーマが受け、知らない項目は断る。"""
    from pydantic import ValidationError

    from gne.schemas import (
        COMMAND_SCHEMAS,
        DirectionKeyExpertsOut,
        DirectionKeyLayer,
        DirectionKeyVerifyOut,
        KeyVerifyRow,
    )
    base = {"layer": 18, "kn": 0.69, "gain": 1.45, "q_rank": 20, "q_rows": 40, "weak": False, "kappa_mu": 1.0,
            "build_auc": 0.95, "rms_a": None}
    DirectionKeyLayer.model_validate(base)                                    # ffn・attn の行は前と同じ形
    DirectionKeyLayer.model_validate({**base, "gate_a": 0.117, "rows_a": 16, "rows_b": 86, "examples_a": 8, "examples_b": 8})
    with pytest.raises(ValidationError):
        DirectionKeyLayer.model_validate({**base, "rows_c": 1})
    row = {"alpha": 1.0, "adapter_scale": 1.0, "n_tokens": 40, "n_big": 10, "kx_threshold": 0.4, "kx_max": 1.3,
           "rel_err_median": 1e-7, "rel_err_p95": 3e-7, "cos_median": 1.0, "leak_b": 0.01, "leak_b_source": "key_examples",
           "leak_b_token_median": 0.02, "ok": True}
    KeyVerifyRow.model_validate(row)
    KeyVerifyRow.model_validate({**row, "other_max": 0.0, "selection_changed": 0})
    with pytest.raises(ValidationError):
        KeyVerifyRow.model_validate({**row, "other": 0.0})
    v = {**ENV_STUB, "command": "direction key-verify", "layer": 18, "kind": "expert", "expert": 31,
         "tensor": "blk.18.ffn_down_exps.weight", "key": "k", "direction": "d", "direction_layer": "own",
         "in_name": "ffn_moe_swiglu-18", "out_name": "ffn_moe_down-18", "d_norm": 3.0, "direction_norm": 0.5,
         "post_norm": None, "kn": 0.69, "leak_b_key_examples": 0.01, "rows": [{**row, "other_max": 0.0, "selection_changed": 0}],
         "v30": True, "judged_leak": True, "gate": 0.117, "res_scale": 0.22}
    DirectionKeyVerifyOut.model_validate(v)
    DirectionKeyVerifyOut.model_validate({**v, "rows": [{**row, "n_tokens": 0, "ok": False}], "v30": False,
                                          "note": "専門家 31 は texts で一度も選ばれなかったので測れません"})
    u = {**ENV_STUB, "command": "direction key-experts", "contrast": "zundamon_endings", "mode": "same",
         "positions": "before:(です|ます)。", "positions_b": "not-before:(です|ます)。", "subset": "even",
         "layers": [{"layer": 18, "n_expert": 32, "n_used": 8, "positions_a": 16, "positions_b": 86, "base_rate": 0.25,
                     "top": [{"expert": 31, "rate_a": 1.0, "rate_b": 0.32, "diff": 0.68, "lift": 3.1, "weight_a": 0.117,
                              "rows_a": 16.0, "rate_dt": {"english": 0.1}},
                             {"expert": 3, "rate_a": 0.5, "rate_b": 0.0, "diff": 0.5, "lift": None, "weight_a": 0.2,
                              "rows_a": 8.0, "rate_dt": {}}]}],
         "replies": {"AB": "given", "dt:english": "generated"}, "iv": None, "notes": [], "seconds": 5.0}
    assert COMMAND_SCHEMAS["direction key-experts"] is DirectionKeyExpertsOut
    DirectionKeyExpertsOut.model_validate(u)


@pytest.mark.parametrize("requant", ["f16", "keep"])
def test_bake_output_matches_schema(tmp_path, requant):
    """専門家の 3 次元テンソルの焼き込みの結果(gain・headproj export)がスキーマ(BakeModified・expert_bytes_added)を通る。
    expert_bytes_added = bytes_to − bytes_from(keep なら 0)。ffn だけの焼き込みには出さない。"""
    from gne.schemas import GainOut
    s, _, _ = _session(tmp_path, exps_type=T.Q8_0)
    _key(s, "ke", "expert:2", [1], N_FF_EXP)
    r = s.recipe(units=["1:E2:keyed:a3@d1#ke"], name="eb")
    for d in (s.gain(r, tmp_path / "g.gguf", bake=True, requant=requant),
              s.headproj_export(r, tmp_path / "h.gguf", bake=True, requant=requant)):
        GainOut.model_validate({**ENV_STUB, "command": "headproj export", **d})
        rec = d["modified"][0]
        assert d["expert_bytes_added"] == rec["bytes_to"] - rec["bytes_from"]
        assert (d["expert_bytes_added"] == 0) == (requant == "keep")
    _key(s, "ks", "shared", [1], N_SHEXP, seed=3)
    d = s.gain(s.recipe(units=["1:S:keyed:a1@d1#ks"], name="sh"), tmp_path / "s.gguf", bake=True)
    GainOut.model_validate({**ENV_STUB, **d})
    assert "expert_bytes_added" not in d and "experts" not in d["modified"][0]


@pytest.mark.parametrize("src", [T.Q8_0, T.Q5_0])
def test_split_bake_expert_slab_by_slab(tmp_path, monkeypatch, src):
    """分割 GGUF への焼き込みも専門家の 3 次元テンソルをスラブごとに書く(テンソル全体を float32 にしない)。keep は元の型で
    その場に書き戻し(大きさ・ヘッダーは同じ、編集しないスラブはバイトのまま)、f16 は通常の焼き込みと同じバイト列(全スラブを
    F16、大きくなるので expert_bytes_added と警告)。書き換えのない分割ファイルはそのまま写す。"""
    import hashlib

    import gne.quant as Q
    from gne.gguf_fast import FastGGUF
    from gne.weights import Weights, _slab_to_f32, tensor_to_f32
    name = "blk.1.ffn_down_exps.weight"
    (tmp_path / "one").mkdir()
    (tmp_path / "sp").mkdir()
    s1, mats, raw = _session(tmp_path / "one", exps_type=src)                 # 同じ中身の分割していないモデル
    s, _, _ = _session(tmp_path / "sp", exps_type=src, split=4)
    assert s.weights.gguf.is_split and not s1.weights.gguf.is_split
    ke = _key(s, "ke", "expert:2", [1], N_FF_EXP)
    _key(s1, "ke", "expert:2", [1], N_FF_EXP)
    delta = 3.0 * np.outer(raw[1].astype(np.float64), ke.kappa[1].astype(np.float64))
    W = mats[name]
    orig = Q.dequantize

    def no_whole(raw_, q):                                     # 専門家のテンソル全体(3 次元)を float32 にしたら失敗
        assert np.ndim(raw_) < 3, np.shape(raw_)
        return orig(raw_, q)
    monkeypatch.setattr(Q, "dequantize", no_whole)
    src_t = s.weights.tensor(name)
    src_part = Path(src_t._mm.filename).name
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    units = ["1:E2:keyed:a3@d1#ke"]

    # keep: 元の型でその場に書き戻す
    s.warnings.clear()
    d = s.headproj_export(s.recipe(units=units, name="k"), tmp_path / "out" / "k.gguf", bake=True, requant="keep")
    assert d["requant"] == "keep" and d["modified"] == [name] and d["expert_bytes_added"] == 0
    assert not any("大きくなりました" in w for w in s.warnings)
    rec = json.loads(Path(d["sidecar"]).read_text(encoding="utf-8"))["modified"][0]
    assert rec["experts"] == [2] and rec["type"] == src.name and rec["to"] == src.name and rec["untouched"] == "raw"
    assert rec["bytes_to"] == rec["bytes_from"] == int(src_t.n_bytes) and 0 < rec["requant_error"] < 0.5
    for p_src, p_out in zip(s.weights.gguf.parts, d["parts"], strict=True):
        assert Path(p_src).stat().st_size == Path(p_out).stat().st_size
        assert (digest(p_src) == digest(p_out)) == (Path(p_src).name != src_part)
    out_t = Weights(d["path"]).tensor(name)
    assert out_t.tensor_type == src
    for e in (0, 1, 3):
        assert np.array_equal(np.asarray(out_t.data[e]), np.asarray(src_t.data[e]))
    edited = _slab_to_f32(out_t, 2).astype(np.float64)
    want = W[2].astype(np.float64) + delta
    assert np.linalg.norm(edited - want) / np.linalg.norm(delta) == pytest.approx(rec["requant_error"], rel=1e-3)

    # f16: 通常の焼き込みと同じ(テンソル全体を F16、編集しないスラブも 1 枚ずつ変換)
    s.warnings.clear()
    d = s.headproj_export(s.recipe(units=units, name="f"), tmp_path / "out" / "f.gguf", bake=True, requant="f16")
    want_bytes = N_EXP * N_EMBD * N_FF_EXP * 2
    assert d["requant"] == "f16" and d["expert_bytes_added"] == want_bytes - int(src_t.n_bytes) > 0
    assert any("大きくなりました" in w and "--requant keep" in w for w in s.warnings)
    rec = json.loads(Path(d["sidecar"]).read_text(encoding="utf-8"))["modified"][0]
    assert rec["to"] == "F16" and rec["untouched"] == f"{src.name}->F16" and rec["bytes_to"] == want_bytes
    w2 = Weights(d["path"])
    out_t = w2.tensor(name)
    assert out_t.tensor_type == T.F16
    got = tensor_to_f32(out_t).reshape(N_EXP, N_EMBD, N_FF_EXP)
    for e in (0, 1, 3):
        np.testing.assert_array_equal(got[e], W[e].astype(np.float16).astype(np.float32))
    np.testing.assert_allclose(got[2], W[2] + delta, atol=2e-3)
    for p_src, p_out in zip(s.weights.gguf.parts, d["parts"], strict=True):     # ほかの分割ファイルはそのまま
        if Path(p_src).name != src_part:
            assert digest(p_src) == digest(p_out)
    for t in FastGGUF(d["path"]).tensors:                                     # ほかのテンソルもそのまま
        if t.name != name:
            assert np.array_equal(np.asarray(t.data), np.asarray(s.weights.tensor(t.name).data)), t.name
    b = s1.gain(s1.recipe(units=units, name="f1"), tmp_path / "one.gguf", bake=True, requant="f16")
    one_t = Weights(b["path"]).tensor(name)
    assert np.array_equal(np.asarray(one_t.data), np.asarray(out_t.data))       # 通常の焼き込みとバイト一致
    assert b["expert_bytes_added"] == d["expert_bytes_added"]


def test_rewrite_gguf_takes_stream(tmp_path):
    """rewrite_gguf は差し替えにバイト列のほか、nbytes と tofile を持つもの(スラブごとに書く)を受け、大きさが違えば断る。"""
    from gne.gguf_fast import FastGGUF, rewrite_gguf
    _moe_gguf(tmp_path / "m.gguf")
    name = "blk.2.ffn_down_exps.weight"
    src = {x.name: x for x in FastGGUF(tmp_path / "m.gguf").tensors}
    assert src[name].tensor_type == T.F32
    new = np.arange(N_EXP * N_EMBD * N_FF_EXP, dtype=np.float16).reshape(N_EXP, N_EMBD, N_FF_EXP)

    class Stream:
        nbytes = new.nbytes

        def tofile(self, f):
            for e in range(N_EXP):
                new[e].tofile(f)
    rewrite_gguf(tmp_path / "m.gguf", tmp_path / "o.gguf", {name: (T.F16, Stream())})
    got = {x.name: x for x in FastGGUF(tmp_path / "o.gguf").tensors}
    assert got[name].tensor_type == T.F16 and np.array_equal(np.asarray(got[name].data), new)
    np.testing.assert_array_equal(np.asarray(got["token_embd.weight"].data), np.asarray(src["token_embd.weight"].data))

    class Short(Stream):
        def tofile(self, f):
            new[0].tofile(f)
    with pytest.raises(ValueError):
        rewrite_gguf(tmp_path / "m.gguf", tmp_path / "o2.gguf", {name: (T.F16, Short())})


def test_mod_text_expert_record():
    from gne.cli import _mod_text
    t = _mod_text({"tensor": "blk.1.ffn_down_exps.weight", "from": "Q8_0", "to": "F16", "experts": [2], "untouched": "Q8_0->F16",
                   "bytes_from": 1_000_000, "bytes_to": 2_000_000, "requant_error": 0.001})
    assert "専門家 E2" in t and "変換(Q8_0->F16)" in t and "1.0 → 2.0 MB" in t and "0.1%" in t
    t = _mod_text({"tensor": "x", "from": "Q8_0", "to": "Q8_0", "experts": [2, 5], "untouched": "raw",
                   "bytes_from": 10, "bytes_to": 10})
    assert "専門家 E2,E5" in t and "バイトのまま" in t and "MB" not in t


def _cli(*args):
    from typer.testing import CliRunner

    from gne.cli import app
    return CliRunner().invoke(app, list(args), env={"COLUMNS": "400"})     # rich が日本語の長い説明を … で切らない幅


def test_cli_help_moe():
    """key-verify の --target に shared / expert:e、headproj export・gain の --requant keep の専門家の扱い、key-experts の入口。"""
    import re
    flat = lambda r: re.sub(r"[\s│]+", "", r.output)  # noqa: E731  rich の折り返しと枠を消して比べる
    r = _cli("direction", "--help")
    assert r.exit_code == 0 and "key-experts" in r.output
    r = _cli("direction", "key-verify", "--help")
    assert r.exit_code == 0 and "shared" in r.output and "expert:e" in flat(r)
    r = _cli("headproj", "export", "--help")
    assert "ffn_down_exps" in flat(r) and "バイトのまま" in flat(r)
    r = _cli("gain", "--help")
    assert "keep" in r.output and "ffn_down_exps" in flat(r)
    r = _cli("direction", "key-experts", "--help")
    assert r.exit_code == 0
    for opt in ("--contrast", "--layers", "--positions", "--positions-b", "--skip", "--max-tokens", "--subset",
                "--replies-a", "--replies-b", "--iv", "--top", "--json"):
        assert opt in r.output, opt
    r = _cli("direction", "key-experts", "--contrast", str(ZUNDAMON), "--layers", "all", "--json")
    assert r.exit_code == 1 and json.loads(r.stdout.strip().splitlines()[-1])["error"]["code"] == "BAD_ARGS"


def test_job_and_mcp_key_experts(monkeypatch):
    """ジョブの種類 direction_key_experts と MCP のツール direction_key_experts(対照集合の名前はパスに直す)。
    direction_key_verify の説明に shared / expert:e。"""
    import anyio

    import gne.api
    import gne.mcp_server as ms
    from gne.jobs import _direction_key_experts, standard_kinds
    assert standard_kinds()["direction_key_experts"] is _direction_key_experts
    calls = []
    fake = SimpleNamespace(progress=object(), direction_key_experts=lambda c, **kw: calls.append((c, kw)) or {"layers": []})
    _direction_key_experts(fake, "c.yaml", layers="6,12", top=3)
    assert calls[-1][0] == "c.yaml" and calls[-1][1]["layers"] == [6, 12] and calls[-1][1]["top"] == 3
    assert calls[-1][1]["progress"] is fake.progress
    with pytest.raises(GneError) as ei:
        _direction_key_experts(fake, "c.yaml")
    assert ei.value.code == "BAD_ARGS"
    started = []

    class FakeRunner:
        def __init__(self, *a, **k):
            pass

        def start(self, kind, params, **k):
            started.append((kind, params))
            return SimpleNamespace(view=lambda: {"id": "j1", "kind": kind, "params": params})
    monkeypatch.setattr(gne.api, "Session", lambda *a, **k: SimpleNamespace(warnings=[]))
    monkeypatch.setattr(ms, "JobRunner", FakeRunner)
    srv = ms.build_server("m.gguf")

    async def go():
        tools = {t.name: t for t in await srv.list_tools()}
        assert "expert:e" in tools["direction_key_verify"].description and "shared" in tools["direction_key_verify"].description
        assert "rate_a" in tools["direction_key_experts"].description
        await srv.call_tool("direction_key_experts", {"contrast": "zundamon_endings", "layers": [18], "top": 4,
                                                      "replies_b": ["a"]})
        err = await srv.call_tool("direction_key_experts", {"contrast": "zundamon_endings", "layers": []})
        return err
    err = anyio.run(go)
    kind, p = started[-1]
    assert kind == "direction_key_experts" and p["contrast"] == str(ZUNDAMON) and p["layers"] == [18] and p["top"] == 4
    assert p["replies_b"] == ["a"] and "positions" not in p
    assert "BAD_ARGS" in json.dumps(err, ensure_ascii=False, default=str)
    assert "direction_key_experts" in ms.INSTRUCTIONS


def test_estimate_keyed(tmp_path):
    """見積もり: eval は keyed のレシピの選択性の分(会話の prefill と返事の生成)を足す。鍵の作成・確かめ・専門家の選ばれ方の
    ジョブも見積もる(会話の数は対照集合と subset で決まる)。"""
    from gne.estimate import Speed, estimate
    s, _, _ = _session(tmp_path)
    ke = _key(s, "ke", "expert:2", [1], N_FF_EXP)
    assert ke.n_pseudo == 0
    sp = Speed(s)
    per = sp.setprefill / 1000
    units = ["1:E2:keyed:a1@d1#ke"]
    base = {"units": units, "doses": [0, 1], "keyed_contrast": str(ZUNDAMON), "keyed_pseudo": 4, "keyed_match": False}
    off = estimate(s, "eval", {**base, "keyed_metrics": False})
    on = estimate(s, "eval", base)
    assert "keyed" not in off["detail"]
    k = on["detail"]["keyed"]
    # 鍵(位置 response・subset even)の反対側 odd: 側 A・側 B(system_a・なし)8 + 8、dont_touch の英語 2。返事は生成する
    assert k["conversations"] == 8 + 8 + 2 and k["conversations_a"] == 8 and k["generated"] == 18
    assert k["pseudo"] == 0 and k["controls"] == 0 and not k["match"]
    assert k["prefills"] == 18 * (1 + 1) + 2 * 18                            # 無介入 + 倍率 1 + 方式の比較(keyed・同じ α の add)
    assert on["seconds"] == pytest.approx(off["seconds"] + k["seconds"], rel=1e-6)
    assert k["seconds"] > k["prefills"] * per
    m = estimate(s, "eval", {**base, "keyed_match": True})["detail"]["keyed"]
    assert m["prefills"] == k["prefills"] + 3 * 18 + 3 * 6 * 8              # add・gain・proj + 大きさ合わせ(側 A の会話)
    sk = estimate(s, "eval", {"units": units, "doses": [0, 1]})["detail"]["keyed"]   # 対照集合が分からない: keyed は飛ばす
    assert sk["skipped"] and sk["seconds"] == 0.0
    assert "keyed" not in estimate(s, "eval", {"units": ["1:dir:d1:1"], "doses": [0, 1]})["detail"]
    reps = ["です。"] * 16
    for kind in ("direction_key", "direction_key_experts"):
        e = estimate(s, kind, {"contrast": str(ZUNDAMON), "layers": [1], "replies_b": reps, "max_tokens": 32})
        assert e["detail"]["conversations"] == 8 + 2 and e["detail"]["generated"] == 2, e     # even: 返事の 8 + 英語 2
        assert e["seconds"] == pytest.approx(10 * per + sp.gen(2, 32))
    e = estimate(s, "direction_key_check", {"key": "ke"})
    assert e["seconds"] == 0.0                                               # 対照集合なし: 鍵に保存した例で測る
    e = estimate(s, "direction_key", {"contrast": str(tmp_path / "none.yaml"), "layers": [1]})
    assert e["detail"]["guess"] and e["seconds"] > 0


# --- モデルを使う: granite-3.1-1b-a400m(granitemoe)-----------------------------------------------------------

# zundamon_endings.yaml の user_prompts の順に、モデル自身の素の返事に見立てた、です・ます調の返事
REPLIES = ["天気が良ければ公園を散歩するのがおすすめです。家ではゆっくり本を読むのもいいと思います。",
           "ご飯と味噌汁と卵焼きがおすすめです。忙しい日はバナナとヨーグルトでも十分です。",
           "料理は手軽に始められる趣味です。散歩や写真も気軽に楽しめます。",
           "雨の日は家で映画を見るのが好きです。温かいお茶を飲みながら過ごします。",
           "まずはしっかり眠ることが大切です。お風呂にゆっくり入るのも効果があります。",
           "私は秋が一番好きです。涼しくて食べ物がおいしい季節です。",
           "野菜と肉を炒めてから水で煮込みます。最後にルーを溶かせば完成です。",
           "寝る前にスマホを見ないようにすると眠りやすくなります。温かい牛乳を飲むのもおすすめです。",
           "京都はお寺や神社が多くておすすめです。北海道では自然とおいしい料理が楽しめます。",
           "時間を決めて短く区切るのが効果的です。机の上を片付けると集中しやすくなります。",
           "どちらもかわいいですが、私は猫が好きです。気ままなところが魅力です。",
           "毎日少しずつ歩くようにしています。野菜を多めに食べることも心がけています。",
           "温かい緑茶がおすすめです。夏は冷たい麦茶もおいしいです。",
           "まず要らない物を捨てることが大切です。物の置き場所を決めると散らかりにくくなります。",
           "相手の好きな物を選ぶのが一番です。手紙を添えると気持ちが伝わります。",
           "毎日同じ時間に寝るようにするといいです。起きたらすぐにカーテンを開けて光を浴びます。"]
POS = "before:(です|ます)。"
G_LAYER = 18                                   # granite(24 層)の後ろ寄りの MoE の層
CONV = [{"role": "user", "content": "朝ごはんに何を食べたらいい？"}, {"role": "assistant", "content": REPLIES[1]}]
CONV2 = [{"role": "user", "content": "眠れない夜はどうしたらいい？"}, {"role": "assistant", "content": REPLIES[7]}]


def _spec_with_replies():
    """zundamon_endings.yaml(同じ contrast_sha)に返事を足した KeySpec(KeyedMetrics に渡す)。"""
    from gne.keyed_build import read_key_spec
    spec = read_key_spec(ZUNDAMON)
    spec.replies["same"] = list(REPLIES)
    return spec


@pytest.fixture(scope="module")
def gsession(tmp_path_factory):
    if not MOE.exists():
        pytest.skip(f"{MOE} がありません")
    from gne.api import Session
    s = Session(MOE, root=tmp_path_factory.mktemp("c07moe"), n_gpu_layers=NGL, n_ctx=1024)
    s.direction(contrast=ZUNDAMON, positions="last")                 # 書く向き r_L(system_a 対 なし、生成なし)
    yield s
    s.close()


@pytest.fixture(scope="module")
def gkey(gsession):
    """側 A の位置(「です。」「ます。」の直前)で最もよく選ばれる専門家の expert:e の鍵。"""
    from gne.keyed_build import build_keys
    u = gsession.direction_key_experts(ZUNDAMON, layers=[G_LAYER], replies_b=REPLIES, top=4)   # keyed_build.expert_usage
    top = u["layers"][0]["top"][0]
    r = build_keys(gsession, ZUNDAMON, name="zunda_exp", layers=[G_LAYER], target=f"expert:{top['expert']}",
                   replies_b=REPLIES, max_tokens=32, q_energy=1.0, pseudo=16)
    return {"usage": u, "expert": top["expert"], "build": r}


@pytest.mark.model
def test_granite_expert_usage_and_key(gsession, gkey):
    """expert_usage: 側 A の位置で選ばれた割合の大きい順。鍵は専門家 e が選ばれた行だけで作り、ルーターの重みの平均を
    gate_a に保存する(κᵀμ = 1、側 B の例の平均に直交)。"""
    from gne.keyed import load_key
    from gne.schemas import DirectionKeyExpertsOut, DirectionKeyOut
    DirectionKeyExpertsOut.model_validate({**ENV_STUB, **gkey["usage"]})      # 実際の出力とスキーマが合う
    DirectionKeyOut.model_validate({**ENV_STUB, **gkey["build"]})
    lay = gkey["usage"]["layers"][0]
    top = lay["top"][0]
    assert lay["layer"] == G_LAYER and lay["n_used"] == 8 and lay["positions_a"] == 16
    assert top["rate_a"] >= 0.75 > lay["base_rate"] and top["rate_a"] > top["rate_b"]
    assert all(lay["top"][i]["rate_a"] >= lay["top"][i + 1]["rate_a"] for i in range(len(lay["top"]) - 1))
    r = gkey["build"]
    e = gkey["expert"]
    assert r["target"] == f"expert:{e}" and r["mode"] == "same" and not r["dropped"]
    row = r["layers"][0]
    assert abs(row["kappa_mu"] - 1) < 1e-9 and 0 < row["gate_a"] < 1
    assert 0 < row["rows_a"] <= 16 and row["rows_b"] < r["counts"]["rows_b"] + 1
    ks = load_key(gsession, "zunda_exp")
    assert ks.target == f"expert:{e}" and ks.kind == "expert" and ks.expert == e and ks.in_dims == {G_LAYER: 512}
    assert ks.gate_a[G_LAYER] == pytest.approx(row["gate_a"]) and ks.info["rows_l"][str(G_LAYER)]["a"] == row["rows_a"]
    k, mu = ks.kappa[G_LAYER].astype(np.float64), ks.mu[G_LAYER].astype(np.float64)
    assert abs(k @ mu - 1) < 1e-3 and np.max(np.abs(ks.exb[G_LAYER].astype(np.float64) @ k)) < 1e-3


@pytest.mark.model
def test_granite_expert_key_check_beats_pseudo(gsession, gkey):
    """保留した例(subset odd)で、トークン単位の AUC と選択性が擬似鍵の p95 を超える(専門家 e が選ばれた行だけで測る)。"""
    from gne.keyed_build import key_check
    c = key_check(gsession, "zunda_exp", contrast=ZUNDAMON, replies_b=REPLIES)
    assert c["held_out"] and c["target"] == f"expert:{gkey['expert']}" and not c["warnings"]
    row = c["layers"][0]
    assert row["auc_token"] > row["pseudo_auc_token"]["p95"] and row["beats_pseudo_auc"]
    assert row["selectivity"] > row["pseudo_selectivity"]["p95"] and row["beats_pseudo"]
    assert row["rho"]["A"]["median"] > 0.5 > row["rho"]["B"]["abs_median"]


@pytest.mark.model
def test_granite_v30_expert_runtime_lora(gsession, gkey):
    """V30: 実行時の 3 次元 LoRA(ffn_down_exps)の差は、専門家 e が選ばれた枠で α (κᵀx) d(ルーターの重みを掛ける前の
    ffn_moe_down)、ほかの専門家の枠は 0、選ばれる専門家は変わらない。"""
    e = gkey["expert"]
    d = gsession.check_keyed(G_LAYER, key="zunda_exp", direction="zundamon_endings", kind="expert", expert=e,
                             alphas=(1.0, 2.0), texts=[CONV, CONV2, "Tell me a short story about a lighthouse keeper."])
    from gne.schemas import DirectionKeyVerifyOut
    DirectionKeyVerifyOut.model_validate({**ENV_STUB, **d})
    assert d["v30"] and d["tensor"] == f"blk.{G_LAYER}.ffn_down_exps.weight", d
    assert d["in_name"] == f"ffn_moe_swiglu-{G_LAYER}" and d["out_name"] == f"ffn_moe_down-{G_LAYER}"
    assert d["res_scale"] == pytest.approx(0.22, rel=1e-3) and 0 < d["gate"] < 1
    for row in d["rows"]:
        assert row["n_tokens"] > 10 and row["rel_err_median"] < 0.03 and row["cos_median"] > 0.99, row
        assert row["other_max"] == 0.0 and row["selection_changed"] == 0
        assert row["leak_b"] < 0.1 and row["leak_b_source"] == "key_examples"


@pytest.mark.model
def test_granite_keyed_metrics_and_runtime(gsession, gkey):
    """実行時の LoRA は発火位置で残差を動かし、それ以外の位置はほとんど動かさない(KeyedMetrics、保留した例)。
    gain の対照は MoE では除外、add・proj は並べる。"""
    from gne.keyed_eval import KeyedMetrics
    e = gkey["expert"]
    r = gsession.recipe(units=[f"{G_LAYER}:E{e}:keyed:a2@zundamon_endings#zunda_exp"], name="ke")
    lo = gsession.lora_for(r)
    assert lo and lo[0][1] == 2.0 and "keyed-" in Path(lo[0][0]).name
    km = KeyedMetrics(gsession, r, contrast=_spec_with_replies(), max_examples=4, pseudo=2, steps=2)
    km.prepare()
    row = km.row(r)
    assert row["layer"] == G_LAYER and row["selectivity"] > 0
    assert row["a_norm"] > 2 * row["b_norm"], row
    blk = km.block()
    methods = {m["method"]: m for m in blk["methods"]}
    assert "MoE" in methods["gain"]["excluded"]
    assert "excluded" not in methods["keyed"] and "excluded" not in methods["add_same_alpha"]
    assert blk["compare"]["b_norm_ratio"]["add_same_alpha"] < 0.5 and blk["pseudo"]["n"] == 2


@pytest.mark.model
def test_granite_key_experts_cli_json(gsession, tmp_path):
    """gne direction key-experts --json(子プロセスを使わず同じプロセスで): スキーマを通り、側 A でよく選ばれる専門家が先頭。"""
    import yaml
    from typer.testing import CliRunner

    from gne.cli import app
    from gne.schemas import DirectionKeyExpertsOut
    (tmp_path / "replies.yaml").write_text(yaml.safe_dump(REPLIES, allow_unicode=True), encoding="utf-8")
    r = CliRunner().invoke(app, ["direction", "key-experts", "--contrast", str(ZUNDAMON), "--layers", str(G_LAYER),
                                 "--replies-b", str(tmp_path / "replies.yaml"), "--top", "3", "--max-tokens", "32",
                                 "--json", "--model", str(MOE), "--root", str(tmp_path / "root"),
                                 "--n-gpu-layers", str(NGL), "--n-ctx", "1024"])
    assert r.exit_code == 0, r.output[-2000:]
    d = json.loads(r.stdout.strip().splitlines()[-1])
    DirectionKeyExpertsOut.model_validate(d)
    lay = d["layers"][0]
    assert d["command"] == "direction key-experts" and lay["layer"] == G_LAYER and len(lay["top"]) == 3
    assert lay["top"][0]["rate_a"] >= lay["top"][1]["rate_a"] >= lay["top"][2]["rate_a"]
    assert d["replies"]["AB"] == "given"


@pytest.mark.model
@pytest.mark.parametrize("requant", ["f16", "keep"])
def test_granite_expert_bake_matches_lora(gsession, gkey, tmp_path, requant):
    """焼き込み(F16 と keep = Q8_0 のまま)を llama.cpp で読み、発火位置の logits が実行時の LoRA 版と合う。編集した
    スラブは W_e + α d_e κᵀ、keep なら編集しないスラブはバイトのまま。書いた GGUF はすぐ消す。"""
    from gguf import GGUFReader

    from gne.engine import Engine
    from gne.keyed import keyed_bundle
    from gne.keyed_build import select_positions
    from gne.weights import _slab_to_f32
    e = gkey["expert"]
    r = gsession.recipe(units=[f"{G_LAYER}:E{e}:keyed:a4@zundamon_endings#zunda_exp"], name=f"kb-{requant}")
    name = f"blk.{G_LAYER}.ffn_down_exps.weight"
    out = tmp_path / f"kb-{requant}.gguf"
    try:
        from gne.schemas import GainOut
        b = gsession.headproj_export(r, out, bake=True, requant=requant)
        GainOut.model_validate({**ENV_STUB, **b})
        rec = b["modified"][0]
        assert rec["tensor"] == name and rec["experts"] == [e] and rec["from"] == "Q8_0"
        assert b["expert_bytes_added"] == rec["bytes_to"] - rec["bytes_from"]
        if requant == "keep":
            assert rec["to"] == "Q8_0" and rec["untouched"] == "raw" and rec["bytes_to"] == rec["bytes_from"]
        else:
            assert rec["to"] == "F16" and rec["bytes_to"] > rec["bytes_from"]
        kb = keyed_bundle(gsession.weights, r, gsession._key_of, gsession.direction_set)
        A, B = kb["tensors"][name]
        dW = B[e].astype(np.float64) @ A[e].astype(np.float64)
        bt = {x.name: x for x in GGUFReader(out).tensors}[name]
        src = gsession.weights.tensor(name)
        err = _slab_to_f32(bt, e).astype(np.float64) - (_slab_to_f32(src, e).astype(np.float64) + dW)
        assert np.linalg.norm(err) / np.linalg.norm(dW) < (0.02 if requant == "keep" else 1e-3)
        other = (e + 1) % src.data.shape[0]
        if requant == "keep":
            assert np.array_equal(np.asarray(bt.data[other]), np.asarray(src.data[other]))
        # 発火位置(「です」「ます」を書く直前の最後の位置)の logits
        en = gsession.engine
        pids = en.tokenize_prompt(en.render_chat(CONV[:1]))
        resp = en.tokenize(REPLIES[1], add_special=False, parse_special=False)
        pos = select_positions(en, resp, POS, n_prompt=len(pids), pieces=[en.token_bytes(x) for x in resp])
        ids = (pids + resp)[:int(pos[-1]) + 1]

        def logits(eng, lora):
            ctx = eng.steer_ctx()
            eng.set_lora(ctx, lora)
            try:
                eng.clear_memory(ctx)
                return eng.prefill(ctx, [ids]).logits.copy()
            finally:
                eng.set_lora(ctx, None)
        lo_ = logits(en, gsession.lora_for(r))
        none = logits(en, None)
        eb = Engine(out, n_gpu_layers=NGL, n_ctx=512, n_seq_max=1)
        try:
            bk = logits(eb, None)
        finally:
            eb.close()
        assert np.abs(lo_ - bk).max() < 0.1 * np.abs(lo_ - none).max(), (np.abs(lo_ - bk).max(), np.abs(lo_ - none).max())
    finally:
        out.unlink(missing_ok=True)
        Path(str(out) + ".headproj.json").unlink(missing_ok=True)


# --- モデルを使う: Qwen1.5-MoE(shared expert あり、GNE_TEST_MOE_SHARED のときだけ。焼き込みはしない)-------------------

Q_LAYERS = [8, 12]


@pytest.fixture(scope="module")
def qsession(tmp_path_factory):
    if not MOE_SHARED or not Path(MOE_SHARED).exists():
        pytest.skip("GNE_TEST_MOE_SHARED(shared expert のある MoE、例 Qwen1.5-MoE-A2.7B)を渡したときだけ")
    from gne.api import Session
    s = Session(Path(MOE_SHARED), root=tmp_path_factory.mktemp("c07moes"), n_gpu_layers=NGL, n_ctx=1024)
    s.direction(contrast=ZUNDAMON, positions="last")
    yield s
    s.close()


@pytest.mark.model
def test_shared_key_check_v30_runtime(qsession):
    """shared: ゲートのある shared expert(qwen2moe の ffn_shexp_gate)。鍵の gate_a、保留集合で擬似鍵を超える、
    V30(ゲートの前の ffn_shexp の差 = α (κᵀx) d)、実行時の LoRA が発火位置で効く。"""
    from gne.keyed_build import build_keys, key_check
    from gne.keyed_eval import KeyedMetrics
    r = build_keys(qsession, ZUNDAMON, name="zunda_shared", layers=Q_LAYERS, target="shared", replies_b=REPLIES,
                   max_tokens=32, q_energy=1.0, pseudo=16)
    assert r["target"] == "shared" and not r["dropped"]
    for row in r["layers"]:
        assert abs(row["kappa_mu"] - 1) < 1e-9 and 0 < row["gate_a"] < 1
    c = key_check(qsession, "zunda_shared", contrast=ZUNDAMON, replies_b=REPLIES)
    assert c["held_out"]
    for row in c["layers"]:
        assert row["beats_pseudo"] and row["beats_pseudo_auc"], row["layer"]
    d = qsession.check_keyed(Q_LAYERS[-1], key="zunda_shared", direction="zundamon_endings", kind="shared",
                             texts=[CONV, CONV2, "Tell me a short story about a lighthouse keeper."])
    assert d["v30"] and d["out_name"] == f"ffn_shexp-{Q_LAYERS[-1]}" and "res_scale" not in d and 0 < d["gate"] < 1, d
    from gne.schemas import DirectionKeyOut, DirectionKeyVerifyOut
    DirectionKeyVerifyOut.model_validate({**ENV_STUB, **d})
    DirectionKeyOut.model_validate({**ENV_STUB, **r})
    for row in d["rows"]:
        assert row["rel_err_median"] < 0.03 and row["leak_b"] < 0.1
    rec = qsession.recipe(units=[f"{Q_LAYERS[-1]}:S:keyed:a2@zundamon_endings#zunda_shared"], name="ks")
    km = KeyedMetrics(qsession, rec, contrast=_spec_with_replies(), max_examples=4, pseudo=2, match=False)
    km.prepare()
    row = km.row(rec)
    assert row["selectivity"] > 0 and row["a_norm"] > 2 * row["b_norm"], row


@pytest.mark.model
def test_qwen_expert_key_v30(qsession):
    """expert:e(60 専門家から 4 つ選ぶ): 側 A でよく選ばれる専門家でも行は少ない(8 例 × 2 位置のうち数行)。V30 は通る。
    key-check は保留集合で測り、トークン単位の AUC が擬似鍵の p95 を超える。"""
    from gne.keyed_build import build_keys, expert_usage, key_check
    u = expert_usage(qsession, ZUNDAMON, layers=[Q_LAYERS[-1]], replies_b=REPLIES, top=3)
    e = u["layers"][0]["top"][0]["expert"]
    r = build_keys(qsession, ZUNDAMON, name="zunda_qexp", layers=[Q_LAYERS[-1]], target=f"expert:{e}",
                   replies_b=REPLIES, max_tokens=32, q_energy=1.0, pseudo=16)
    assert r["layers"][0]["rows_a"] >= 4
    c = key_check(qsession, "zunda_qexp", contrast=ZUNDAMON, replies_b=REPLIES)
    assert c["held_out"] and c["layers"][0]["beats_pseudo_auc"]
    d = qsession.check_keyed(Q_LAYERS[-1], key="zunda_qexp", direction="zundamon_endings", kind="expert", expert=e,
                             texts=[CONV, CONV2, "Tell me a short story about a lighthouse keeper."])
    assert d["v30"], d
    assert all(row["other_max"] == 0.0 and row["selection_changed"] == 0 for row in d["rows"])
