"""CHANGE-07 attention の keyed write のテスト(書き込み行列 attn_output / ssm_out、入力 = ヘッド出力の連結)。

モデルなし: 小さな GGUF(qwen35 arch。線形注意の層と attention の層で、書き込み行列の入力の次元が違う)で、
keyed_build.AttnCapture の捕捉名(final_output / attn_gated / kqv_out、bitnet の attn_sub_norm)と形(3 次元の捕捉を
トークンの行に)、ssm_out の .scale、post-norm などの arch で断ること、層ごとに長さの違う鍵の保存と LoRA の行列
(ssm_out と attn_output)。
モデルあり(Qwen3-0.6B、GNE_TEST_MODEL。CPU なら GNE_TEST_NGL=0): configs/contrast/zundamon_endings.yaml の作り方 (b)
(与えた返事の「です。」「ます。」を書く直前)で attn の鍵を作り、保存項目・κᵀμ = 1・key-check(保留集合で擬似鍵を超える)・
V30(check_keyed kind=attn)・実行時の LoRA が発火位置で効くこと(KeyedMetrics。gain の対照は除外)と生成が変わること・
LoRA と F16 焼き込みの書き出し。
ハイブリッド(線形注意)のモデル: GNE_TEST_HYBRID_MODEL(Qwen3.5-0.8B など、CPU で回せる大きさ)を渡したときだけ、
線形注意の層(ssm_out)と attention の層の鍵の作成・key-check・V30・実行時の LoRA のテンソル。
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gne import keyed as K
from gne.keyed import KeySet, key_dir, make_key, orth_basis
from gne.keyed_build import AttnCapture, target_capture
from gne.steer import GneError

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
HYBRID = os.environ.get("GNE_TEST_HYBRID_MODEL")
NGL = int(os.environ.get("GNE_TEST_NGL", "-1"))
ZUNDAMON = ROOT / "configs" / "contrast" / "zundamon_endings.yaml"

# 小さな qwen35: 層 0・2 が線形注意(V ヘッド 4 × 幅 6 = 24)、層 1・3 が attention(ヘッド 4 × 幅 3 = 12)
N_EMBD, N_FF, N_LIN_HEAD, LIN_HD, N_HEAD, HEAD_DIM = 16, 32, 4, 6, 4, 3


def _hybrid_gguf(path, *, ssm_scale=True):
    from gguf import GGUFWriter
    rng = np.random.default_rng(3)
    w = GGUFWriter(str(path), arch="qwen35")
    w.add_name("tiny-hybrid")
    w.add_block_count(4)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(2)
    w.add_uint32("qwen35.ssm.time_step_rank", N_LIN_HEAD)
    w.add_uint32("qwen35.ssm.group_count", 2)
    w.add_uint32("qwen35.full_attention_interval", 2)
    w.add_token_list([f"t{i}" for i in range(8)])
    mats = {}
    for L in range(4):
        shapes = {"ffn_down": (N_EMBD, N_FF), "ffn_up": (N_FF, N_EMBD)}
        if (L + 1) % 2 == 0:
            shapes.update({"attn_q": (N_HEAD * HEAD_DIM * 2, N_EMBD), "attn_output": (N_EMBD, N_HEAD * HEAD_DIM)})
        else:
            shapes.update({"ssm_alpha": (N_LIN_HEAD, N_EMBD), "ssm_beta": (N_LIN_HEAD, N_EMBD),
                           "ssm_out": (N_EMBD, N_LIN_HEAD * LIN_HD)})
        for nm, shape in shapes.items():
            m = rng.standard_normal(shape).astype(np.float32)
            mats[f"blk.{L}.{nm}.weight"] = m
            w.add_tensor(f"blk.{L}.{nm}.weight", m)
        if ssm_scale and (L + 1) % 2:
            w.add_tensor(f"blk.{L}.ssm_out.scale", np.array([0.5], dtype=np.float32))
    w.add_tensor("token_embd.weight", rng.standard_normal((8, N_EMBD)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return mats


@pytest.fixture
def tiny(tmp_path):
    """小さなハイブリッドの GGUF の Session(エンジンは読まない)と、方向 d1(全層、n_embd)。"""
    from gne.api import Session
    mats = _hybrid_gguf(tmp_path / "h.gguf")
    s = Session(tmp_path / "h.gguf", root=tmp_path / "root")
    rng = np.random.default_rng(2)
    raw = {L: rng.standard_normal(N_EMBD).astype(np.float32) for L in range(4)}
    np.savez(s.store.direction_dir(s.meta.model_id) / "d1.npz", layers=np.array(sorted(raw), dtype=np.int32),
             **{f"layer_{L}": v for L, v in raw.items()})
    return s, mats, raw


def _key(s, name, dims: dict[int, int], *, seed=5, n_b=4, target="attn"):
    rng = np.random.default_rng(seed)
    ks = KeySet(target=target, kappa={}, mu={}, kn={}, gain={}, model_id=s.meta.model_id)
    for L, n in dims.items():
        Bm = rng.standard_normal((n_b, n))
        mu = rng.standard_normal(n)
        kap, info = make_key(mu, orth_basis(Bm, energy=1.0))
        ks.kappa[L], ks.mu[L] = kap.astype(np.float32), mu.astype(np.float32)
        ks.kn[L], ks.gain[L], ks.q_rank[L] = info["kn"], info["gain"], info["q_rank"]
        ks.exb[L] = Bm.astype(np.float32)
    ks.save(key_dir(s) / f"{name}.npz")
    return ks


# --- モデル不要 ----------------------------------------------------------------------------------------

def test_attn_capture_names_and_shapes(tiny):
    """線形注意の層は final_output(ssm_out の入力)、attention の層は attn_gated(無ければ kqv_out、bitnet は
    attn_sub_norm)。入力の次元は層ごと(24 と 12)。3 次元の捕捉([head_dim, n_head, T])はトークンの行にそろえる。"""
    s, _, _ = tiny
    arch = s.weights.arch
    assert [li.mixer for li in arch.layers] == ["linear_attention", "attention"] * 2
    tc = target_capture(s, "attn", [0, 1])
    assert isinstance(tc, AttnCapture) and tc.in_dim == {0: N_LIN_HEAD * LIN_HD, 1: N_HEAD * HEAD_DIM}
    tc.bind({"final_output-0", "attn_gated-1", "kqv_out-1", "ffn_out-1"})
    assert tc.names(0) == ["final_output-0"] and tc.names(1) == ["attn_gated-1"]
    tc.bind({"final_output-0", "kqv_out-1"})
    assert tc.names(1) == ["kqv_out-1"]
    tc.bind({"final_output-0", "kqv_out-1", "attn_sub_norm-1"})        # bitnet: kqv_out の後の norm が書き込み行列の入力
    assert tc.names(1) == ["attn_sub_norm-1"]
    with pytest.raises(GneError) as ei:
        tc.bind({"attn_gated-1"})
    assert ei.value.code == "ARCH_UNSUPPORTED" and "final_output-0" in ei.value.message
    # 3 次元の捕捉: Capture は [head_dim, n_head, T] を (n_head·T, head_dim) で返す → (T, n_head·head_dim)
    tc.bind({"final_output-0", "kqv_out-1"})
    T = 5
    x = np.arange(T * N_HEAD * HEAD_DIM, dtype=np.float32).reshape(T, N_HEAD, HEAD_DIM)
    got = tc.token_rows("kqv_out-1", x.reshape(T * N_HEAD, HEAD_DIM), T)
    np.testing.assert_array_equal(got, x.reshape(T, -1))
    flat = x.reshape(T, -1)
    assert tc.token_rows("kqv_out-1", flat, T) is flat
    sel = np.array([1, 3])
    rows, gate = tc.rows({"kqv_out-1": flat}, 1, sel)
    assert gate is None and np.array_equal(rows, flat[sel])
    with pytest.raises(GneError) as ei:
        tc.rows({"kqv_out-1": flat[:, :7]}, 1, sel)
    assert ei.value.code == "ARCH_UNSUPPORTED"
    # qwen35 の ssm_out.scale は build_lora_mm(ssm_out, x, s) で W x にだけ掛かる(LoRA の出力には掛からない)
    assert s.weights.has_tensor("blk.0.ssm_out.scale") and K.writer_out_scale(s.weights, 0, "attn") == 1.0
    # 最終層は断る(keyed.check_layer_ok)
    with pytest.raises(GneError) as ei:
        K.check_layer_ok(s.weights, 3, "attn")
    assert ei.value.code == "BAD_RECIPE"


def test_attn_refusals():
    """post-norm の arch と、attention の出力に後から倍率を掛ける arch は attn の keyed を断る。shared・expert の捕捉はまだ無い。"""
    for arch in ("gemma2", "olmo2", "mimo2"):
        with pytest.raises(GneError) as ei:
            K.post_norm_gamma(SimpleNamespace(meta=SimpleNamespace(arch=arch)), 1, "attn")
        assert ei.value.code == "ARCH_UNSUPPORTED", arch
    assert K.post_norm_gamma(SimpleNamespace(meta=SimpleNamespace(arch="qwen35")), 1, "attn") is None
    for t in ("shared", "expert:1"):
        with pytest.raises(GneError) as ei:
            target_capture(None, t, [1])
        assert ei.value.code == "ARCH_UNSUPPORTED"


def test_mixed_width_key_and_lora(tiny, tmp_path):
    """線形注意の層(ssm_out、入力 24)と attention の層(attn_output、入力 12)を 1 つの鍵に持てる。LoRA は層ごとに
    A = κᵀ(1 × in)、B = α d / α_ref(n_embd × 1)。同じ長さの鍵は今までと同じ中身(in_dims を書かない)。"""
    from gguf import GGUFReader
    s, _, raw = tiny
    ks = _key(s, "ka", {0: N_LIN_HEAD * LIN_HD, 1: N_HEAD * HEAD_DIM})
    back = K.load_key(s, "ka")
    assert back.in_dims == {0: 24, 1: 12} and back.summary()["in_dims"] == {"0": 24, "1": 12}
    assert back.sha == ks.sha and back.target == "attn"
    with np.load(key_dir(s) / "ka.npz") as z:
        assert z["in_dims"].tolist() == [24, 12]
    same = _key(s, "kb", {0: 24, 2: 24}, seed=9)
    with np.load(key_dir(s) / "kb.npz") as z:
        assert "in_dims" not in z.files and int(z["in_dim"]) == 24
    assert "in_dims" not in same.summary()
    r = s.recipe(units=["0:attn:keyed:a2@d1#ka", "1:attn:keyed:a-1@d1#ka"], name="mixed")
    path, ref = s.keyed_lora(r)
    assert ref == 2.0
    t = {x.name: x for x in GGUFReader(path).tensors}
    for L, name, n_in, alpha in ((0, "blk.0.ssm_out.weight", 24, 2.0), (1, "blk.1.attn_output.weight", 12, -1.0)):
        A = np.array(t[f"{name}.lora_a"].data, dtype=np.float32)
        B = np.array(t[f"{name}.lora_b"].data, dtype=np.float32)
        assert A.shape == (1, n_in) and B.shape == (N_EMBD, 1)
        np.testing.assert_allclose(A[0], ks.kappa[L], rtol=1e-6)
        np.testing.assert_allclose(B[:, 0], alpha / ref * raw[L], rtol=1e-5, atol=1e-6)
    meta = json.loads(str(GGUFReader(path).fields["gne.keyed"].contents()))
    assert {u["target"] for u in meta["units"]} == {"attn"}
    assert {u["tensor"] for u in meta["units"]} == {"blk.0.ssm_out.weight", "blk.1.attn_output.weight"}
    # 鍵の書き込み先と kind が合わなければ BAD_RECIPE
    _key(s, "kf", {0: N_FF}, target="ffn")
    with pytest.raises(GneError) as ei:
        s.recipe(units=["0:attn:keyed:a1@d1#kf"])
    assert ei.value.code == "BAD_RECIPE"


# --- モデルを使う(Qwen3-0.6B)---------------------------------------------------------------------------

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
LAYERS = [8, 14]
POS = "before:(です|ます)。"


def _spec_with_replies():
    """zundamon_endings.yaml(同じ contrast_sha)に返事を足した KeySpec(KeyedMetrics に渡す)。"""
    from gne.keyed_build import read_key_spec
    spec = read_key_spec(ZUNDAMON)
    spec.replies["same"] = list(REPLIES)
    return spec


@pytest.fixture(scope="module")
def msession(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip(f"{MODEL} がありません")
    from gne.api import Session
    s = Session(MODEL, root=tmp_path_factory.mktemp("c07a"), n_gpu_layers=NGL, n_ctx=1024)
    s.direction(contrast=ZUNDAMON, positions="last")                 # 書く向き r_L(system_a 対 なし、生成なし)
    yield s
    s.close()


@pytest.fixture(scope="module")
def akey(msession):
    from gne.keyed_build import build_keys
    return build_keys(msession, ZUNDAMON, name="zunda_attn", layers=LAYERS, target="attn", replies_b=REPLIES,
                      max_tokens=32, q_energy=1.0, pseudo=16)


@pytest.mark.model
def test_attn_key_saved_fields(msession, akey):
    """target attn の鍵: 入力はヘッド出力の連結(n_head·head_dim)、κᵀμ = 1、q_energy = 1 なら側 B の例の平均に直交。"""
    from gne.keyed import load_key
    r = akey
    assert r["target"] == "attn" and r["mode"] == "same" and r["positions"] == POS and not r["dropped"]
    assert r["counts"]["a"] == 8 and r["counts"]["b"] == 8 and r["counts"]["dt"] == 2
    assert r["replies"] == {"AB": "given", "dt:english": "generated"}
    for row in r["layers"]:
        assert abs(row["kappa_mu"] - 1) < 1e-9 and row["q_rank"] == 8 + 2 and 0 < row["kn"] <= 1
    ks = load_key(msession, "zunda_attn")
    li = msession.weights.arch.layers[LAYERS[0]]
    n_in = li.n_head * li.head_dim
    assert ks.target == "attn" and ks.kind == "attn" and ks.layers == LAYERS and ks.in_dims == {L: n_in for L in LAYERS}
    assert ks.max_tokens == 32 and ks.subset == "even" and ks.info["mode"] == "same" and ks.n_pseudo == 16
    for L in LAYERS:
        k, mu = ks.kappa[L].astype(np.float64), ks.mu[L].astype(np.float64)
        assert abs(k @ mu - 1) < 1e-3
        assert ks.exa[L].shape == (8, n_in) and ks.kappa_ctrl[L].shape == (16, n_in)
        assert np.max(np.abs(ks.exb[L].astype(np.float64) @ k)) < 1e-3
    assert not ks.rms_a and not ks.gate_a


@pytest.mark.model
def test_attn_key_check_beats_pseudo(msession, akey):
    """保留した例(subset odd)で、トークン単位の AUC と選択性が擬似鍵の p95 を超える。英語の dont_touch では ρ ≈ 0。"""
    from gne.keyed_build import key_check
    c = key_check(msession, "zunda_attn", contrast=ZUNDAMON, replies_b=REPLIES)
    assert c["held_out"] and c["target"] == "attn" and c["subset"] == "odd" and not c["warnings"]
    for row in c["layers"]:
        assert row["auc_token"] > row["pseudo_auc_token"]["p95"], row["layer"]
        assert row["selectivity"] > row["pseudo_selectivity"]["p95"], row["layer"]
        assert row["beats_pseudo"] and row["beats_pseudo_auc"]
        assert row["rho"]["A"]["median"] > 0.5 > row["rho"]["B"]["abs_median"]
        assert abs(row["dont_touch_median"]["english"]) < 0.3


@pytest.mark.model
def test_v30_attn_with_built_key(msession, akey):
    """V30: 作った鍵で、attn_output の LoRA の差 = α (κᵀx) d(相対誤差の中央値 < 3%)、側 B の例の漏れ < 0.1。"""
    conv = [{"role": "user", "content": "朝ごはんに何を食べたらいい？"}, {"role": "assistant", "content": REPLIES[1]}]
    d = msession.check_keyed(14, key="zunda_attn", direction="zundamon_endings", kind="attn", alphas=(1.0, 2.0),
                             texts=[conv, "Tell me a short story about a lighthouse keeper."])
    assert d["v30"] and d["tensor"] == "blk.14.attn_output.weight" and d["in_name"] == "kqv_out-14", d
    for row in d["rows"]:
        assert row["rel_err_median"] < 0.03 and row["cos_median"] > 0.99, row
        assert row["leak_b"] < 0.1 and row["leak_b_source"] == "key_examples", row
        assert row["kx_max"] > 0.5


@pytest.mark.model
def test_attn_keyed_runtime_fires_at_positions(msession, akey):
    """実行時の LoRA(lora_for)は発火位置(「です。」「ます。」の直前)で残差を動かし、それ以外の位置はほとんど動かさない
    (KeyedMetrics、保留した例)。gain の対照は attn では除外、add・proj は並べる。強めに当てると生成が変わる。"""
    from gne.keyed_eval import KeyedMetrics
    r = msession.recipe(units=["14:attn:keyed:a2@zundamon_endings#zunda_attn"], name="ka")
    lo = msession.lora_for(r)
    assert lo and lo[0][1] == 2.0 and "keyed-" in Path(lo[0][0]).name
    km = KeyedMetrics(msession, r, contrast=_spec_with_replies(), max_examples=4, pseudo=2, steps=2)
    km.prepare()
    row = km.row(r)
    assert row["layer"] == 14 and row["selectivity"] > 0
    assert row["a_norm"] > 2 * row["b_norm"] and row["b_over_a"] < 0.6, row    # 発火位置で大きく、それ以外で小さい
    blk = km.block()
    methods = {m["method"]: m for m in blk["methods"]}
    assert "attn" in methods["gain"]["excluded"]
    assert "excluded" not in methods["keyed"] and "excluded" not in methods["add_same_alpha"]
    assert "proj" in methods and "add" in methods
    assert blk["compare"]["b_norm_ratio"]["add_same_alpha"] < 0.5    # 同じ α を全位置に足すより側 B の乱れがずっと小さい
    assert blk["pseudo"]["n"] == 2 and blk["held_out"]
    st = msession.steer(["14:attn:keyed:a6@zundamon_endings#zunda_attn"], prompt="おすすめの飲み物は？", max_tokens=24)
    assert st["differs"], (st["baseline"]["text"], st["steered"]["text"])


@pytest.mark.model
def test_attn_lora_and_f16_bake(msession, akey, tmp_path):
    """書き出し: LoRA に blk.14.attn_output.weight.lora_a/b(A = κᵀ、B = α d、倍率 1)。F16 の焼き込みは
    W' = W + α d κᵀ(F16 の丸めまで)。"""
    from gguf import GGUFReader

    from gne.keyed import load_key
    from gne.weights import tensor_to_f32
    r = msession.recipe(units=["14:attn:keyed:a2@zundamon_endings#zunda_attn"], name="ka-export")
    ks = load_key(msession, "zunda_attn")
    raw = msession.direction_set("zundamon_endings").raw[14].astype(np.float64)
    name = "blk.14.attn_output.weight"
    d = msession.headproj_export(r, tmp_path / "ka.lora.gguf")
    rd = GGUFReader(d["path"])
    t = {x.name: x for x in rd.tensors}
    A = np.array(t[f"{name}.lora_a"].data, dtype=np.float32)
    B = np.array(t[f"{name}.lora_b"].data, dtype=np.float32)
    li = msession.weights.arch.layers[14]
    assert A.shape == (1, li.n_head * li.head_dim) and B.shape == (msession.meta.n_embd, 1)
    assert set(t) == {f"{name}.lora_a", f"{name}.lora_b"}
    np.testing.assert_allclose(A[0], ks.kappa[14], rtol=1e-6)
    np.testing.assert_allclose(B[:, 0], 2.0 * raw, rtol=1e-4, atol=1e-5)         # 配る LoRA は倍率 1 で α を書く
    keyed_meta = json.loads(str(rd.fields["gne.keyed"].contents()))
    assert keyed_meta["units"][0]["target"] == "attn" and keyed_meta["units"][0]["tensor"] == name
    # F16 の焼き込み
    b = msession.headproj_export(r, tmp_path / "ka-bake.gguf", bake=True)
    assert [m["tensor"] for m in b["modified"]] == [name] and [m["to"] for m in b["modified"]] == ["F16"]
    baked = {x.name: x for x in GGUFReader(b["path"]).tensors}[name]
    W1 = tensor_to_f32(baked).astype(np.float64)
    W0 = msession.weights.head_writer_matrix(14).astype(np.float64)
    delta = 2.0 * np.outer(raw, ks.kappa[14].astype(np.float64))
    err = np.abs(W1 - (W0 + delta))
    assert err.max() < 2e-3 * max(1.0, np.abs(W0).max()), err.max()
    assert np.abs(W1 - W0).max() > 10 * err.max()                   # 編集は F16 の丸めよりずっと大きい
    # 焼き込み版を読んだ llama.cpp の出力は、発火位置(「です」を書く直前)で実行時の LoRA 版と合う
    from gne.engine import Engine
    from gne.keyed_build import select_positions
    e = msession.engine
    pids = e.tokenize_prompt(e.render_chat([{"role": "user", "content": "朝ごはんに何を食べたらいい？"}]))
    resp = e.tokenize(REPLIES[1], add_special=False, parse_special=False)
    first = int(select_positions(e, resp, POS, n_prompt=len(pids), pieces=[e.token_bytes(x) for x in resp])[0])
    ids = (pids + resp)[:first + 1]

    def logits(eng, lora):
        ctx = eng.steer_ctx()
        eng.set_lora(ctx, lora)
        try:
            eng.clear_memory(ctx)
            return eng.prefill(ctx, [ids]).logits.copy()
        finally:
            eng.set_lora(ctx, None)
    lo_ = logits(e, msession.lora_for(r))
    none = logits(e, None)
    eb = Engine(b["path"], n_gpu_layers=NGL, n_ctx=512, n_seq_max=1)
    try:
        bk = logits(eb, None)
    finally:
        eb.close()
    assert np.abs(lo_ - bk).max() < 0.2 * np.abs(lo_ - none).max(), (np.abs(lo_ - bk).max(), np.abs(lo_ - none).max())


# --- ハイブリッド(線形注意)のモデル: GNE_TEST_HYBRID_MODEL のときだけ ------------------------------------------

@pytest.fixture(scope="module")
def hsession(tmp_path_factory):
    if not HYBRID or not Path(HYBRID).exists():
        pytest.skip("GNE_TEST_HYBRID_MODEL(線形注意のあるモデル、例 Qwen3.5-0.8B)がありません")
    from gne.api import Session
    s = Session(Path(HYBRID), root=tmp_path_factory.mktemp("c07h"), n_gpu_layers=NGL, n_ctx=1024)
    arch = s.weights.arch
    if not any(li.mixer == "linear_attention" and li.head_writer for li in arch.layers):
        s.close()
        pytest.skip("GNE_TEST_HYBRID_MODEL に線形注意の層がありません")
    # 書く向きは乱数(V30 は向きの意味に依らない。方向の捕捉は CPU で重いので作らない)
    rng = np.random.default_rng(4)
    raw = {L: rng.standard_normal(s.meta.n_embd).astype(np.float32) for L in range(s.meta.n_layer)}
    np.savez(s.store.direction_dir(s.meta.model_id) / "rnd.npz", layers=np.array(sorted(raw), dtype=np.int32),
             **{f"layer_{L}": v for L, v in raw.items()})
    yield s
    s.close()


@pytest.mark.model
def test_hybrid_linear_attention_keys(hsession):
    """線形注意の層(ssm_out、入力 final_output)と attention の層(attn_output、入力 attn_gated)の鍵を 1 つにまとめて作り、
    key-check(保留集合)・V30・実行時の LoRA のテンソルを確かめる。"""
    from gguf import GGUFReader

    from gne.keyed import last_main_layer, load_key
    from gne.keyed_build import build_keys, key_check
    s = hsession
    arch = s.weights.arch
    top = last_main_layer(s.weights)
    mid = top // 3
    lin = next(li.index for li in arch.layers if li.index >= mid and li.mixer == "linear_attention" and li.head_writer)
    att = next(li.index for li in arch.layers if lin < li.index < top and li.mixer == "attention" and li.head_writer)
    r = build_keys(s, ZUNDAMON, name="hyb", layers=[lin, att], target="attn", replies_b=REPLIES, max_tokens=24,
                   q_energy=1.0, pseudo=8)
    assert not r["dropped"] and r["target"] == "attn"
    ks = load_key(s, "hyb")
    for L in (lin, att):
        li = arch.layers[L]
        assert ks.in_dims[L] == li.n_head * li.head_dim == K.writer_in_dim(s.weights, L, "attn")
        assert abs(float(ks.kappa[L].astype(np.float64) @ ks.mu[L].astype(np.float64)) - 1) < 1e-3
    c = key_check(s, "hyb", contrast=ZUNDAMON, replies_b=REPLIES)
    assert c["held_out"]
    for row in c["layers"]:
        assert row["auc_token"] > row["pseudo_auc_token"]["p95"] and row["beats_pseudo"], row["layer"]
    conv = [{"role": "user", "content": "朝ごはんに何を食べたらいい？"}, {"role": "assistant", "content": REPLIES[1]}]
    for L, tensor, x_name, y_name in ((lin, "ssm_out", "final_output", "linear_attn_out"),
                                      (att, "attn_output", "attn_gated", "attn_output")):
        d = s.check_keyed(L, key="hyb", direction="rnd", kind="attn", alphas=(1.0,), texts=[conv])
        assert d["v30"] and d["tensor"] == f"blk.{L}.{tensor}.weight", d
        assert d["in_name"] == f"{x_name}-{L}" and d["out_name"] == f"{y_name}-{L}", d
        assert d["rows"][0]["rel_err_median"] < 0.03 and d["rows"][0]["cos_median"] > 0.99, d
    rec = s.recipe(units=[f"{lin}:attn:keyed:a1@rnd#hyb", f"{att}:attn:keyed:a1@rnd#hyb"], name="hyb")
    path, _ = s.keyed_lora(rec)
    names = {x.name for x in GGUFReader(path).tensors}
    assert {f"blk.{lin}.ssm_out.weight.lora_a", f"blk.{att}.attn_output.weight.lora_b"} <= names
