"""CHANGE-07 WP3 のテスト: keyed write の実行時(Session.keyed_lora・lora_for)と書き出し(gne gain・headproj export・
焼き込み・分割 GGUF)、レシピの確かめ、control vector で断ること。

モデル不要のものは小さな GGUF(llama arch)を作って Session を開く(エンジンは読まない)。モデルを使うもの(V30・V33)は
Qwen3-0.6B(GNE_TEST_MODEL)を GNE_TEST_NGL で読む(CPU なら GNE_TEST_NGL=0)。鍵は WP4 の作成を使わず、捕捉した活性から
make_key で作る。
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from gne.keyed import KeySet, key_dir, make_key, orth_basis
from gne.steer import GneError, Recipe

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
NGL = int(os.environ.get("GNE_TEST_NGL", "-1"))
TSUN = ROOT / "configs" / "contrast" / "tsun_endings.yaml"

N_EMBD, N_FF, N_HEAD, HEAD_DIM, N_EXP, N_FF_EXP, N_SHEXP = 16, 32, 4, 3, 3, 10, 8   # n_ff は Q8_0 の塊(32)の倍数


def _gguf(path, *, n_layer=4, moe_layers=(), shexp=False, nextn=0, split=0, q8=False):
    """attention(attn_q・attn_output)と FFN(ffn_up・ffn_down、MoE の層は ffn_down_exps と ffn_down_shexp)を持つ
    小さな GGUF(llama arch、F32)。nextn は最後の nextn 層を MTP の層にする。split > 0 なら 1 ファイル split 個までの
    テンソルで分割して書く(1 番目はメタデータだけ)。q8 なら ffn_down を Q8_0 で書く(行列は量子化して戻した値を返す)。
    戻り値は {テンソル名: numpy の行列} と、読むファイルのパス。"""
    from gguf import GGMLQuantizationType, GGUFWriter
    from gguf import quants as gquants
    rng = np.random.default_rng(1)
    kw = dict(split_max_tensors=split, small_first_shard=True) if split else {}
    w = GGUFWriter(str(path), arch="llama", **kw)
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
    mats = {}
    for L in range(n_layer):
        shapes = {"attn_q": (N_HEAD * HEAD_DIM, N_EMBD), "attn_output": (N_EMBD, N_HEAD * HEAD_DIM)}
        if L in moe_layers:
            shapes["ffn_down_exps"] = (N_EXP, N_EMBD, N_FF_EXP)
            if shexp:
                shapes["ffn_down_shexp"] = (N_EMBD, N_SHEXP)
        else:
            shapes["ffn_down"] = (N_EMBD, N_FF)
            shapes["ffn_up"] = (N_FF, N_EMBD)
        if L >= n_layer - nextn:
            shapes["nextn.eh_proj"] = (N_EMBD, 2 * N_EMBD)
        for nm, shape in shapes.items():
            m = rng.standard_normal(shape).astype(np.float32)
            if q8 and nm == "ffn_down":                                  # 実際のモデルに近い大きさ(0.02)で Q8_0 に
                qd = gquants.quantize(m * 0.02, GGMLQuantizationType.Q8_0)
                mats[f"blk.{L}.{nm}.weight"] = gquants.dequantize(qd, GGMLQuantizationType.Q8_0).astype(np.float32).reshape(shape)
                w.add_tensor(f"blk.{L}.{nm}.weight", qd, raw_dtype=GGMLQuantizationType.Q8_0)
                continue
            mats[f"blk.{L}.{nm}.weight"] = m
            w.add_tensor(f"blk.{L}.{nm}.weight", m)
    w.add_tensor("token_embd.weight", rng.standard_normal((8, N_EMBD)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    p = Path(path)
    first = next(iter(sorted(p.parent.glob(f"{p.stem}-00001-of-*.gguf")))) if split else p
    return mats, first


def _dirs(path, n_layer=4, seed=2):
    rng = np.random.default_rng(seed)
    raw = {L: rng.standard_normal(N_EMBD).astype(np.float32) for L in range(n_layer)}
    np.savez(path, layers=np.array(sorted(raw), dtype=np.int32), **{f"layer_{L}": v for L, v in raw.items()})
    return raw


def _key(s, name, target, layers, in_dim, *, seed=5, n_b=4):
    """側 B の標本 n_b 本と μ から作った鍵を <root>/keys/<model_id>/<name>.npz に保存する(exb は側 B の標本)。"""
    rng = np.random.default_rng(seed)
    ks = KeySet(target=target, kappa={}, mu={}, kn={}, gain={}, model_id=s.meta.model_id)
    for L in layers:
        Bm = rng.standard_normal((n_b, in_dim))
        mu = rng.standard_normal(in_dim)
        kap, info = make_key(mu, orth_basis(Bm, energy=1.0))
        ks.kappa[L], ks.mu[L] = kap.astype(np.float32), mu.astype(np.float32)
        ks.kn[L], ks.gain[L], ks.q_rank[L] = info["kn"], info["gain"], info["q_rank"]
        ks.exb[L] = Bm.astype(np.float32)
    ks.save(key_dir(s) / f"{name}.npz")
    return ks


def _session(tmp_path, **kw):
    """小さな GGUF の Session(エンジンは読まない)と、方向 d1・鍵 k1(ffn、層 1・2)・ka(attn、層 1)を置いたもの。"""
    from gne.api import Session
    mats, first = _gguf(tmp_path / "m.gguf", **kw)
    s = Session(first, root=tmp_path / "root")
    raw = _dirs(s.store.direction_dir(s.meta.model_id) / "d1.npz", n_layer=kw.get("n_layer", 4))
    ffn_layers = [L for L in (1, 2) if L not in kw.get("moe_layers", ())]
    k1 = _key(s, "k1", "ffn", ffn_layers, N_FF) if ffn_layers else None
    ka = _key(s, "ka", "attn", [1], N_HEAD * HEAD_DIM, seed=7)
    return s, mats, raw, {"k1": k1, "ka": ka}


def _tensors(path):
    from gguf import GGUFReader
    rd = GGUFReader(path)
    return rd, {t.name: t for t in rd.tensors}


def _lora(t, name):
    """LoRA GGUF の lora_a・lora_b を numpy の形 A (…, r, in)・B (…, out, r) で。"""
    A = np.array(t[f"{name}.lora_a"].data, dtype=np.float32)
    B = np.array(t[f"{name}.lora_b"].data, dtype=np.float32)
    return A, B


# --- モデル不要 ----------------------------------------------------------------------------------------

def test_export_cvec_refuses_keyed_with_hint(tmp_path):
    s, _, _, _ = _session(tmp_path)
    r = s.recipe(units=["1:ffn:keyed:a1.5@d1#k1"])
    with pytest.raises(GneError) as e:
        s.export_cvec(r, tmp_path / "x.gguf")
    assert e.value.code == "BAD_RECIPE" and "keyed" in e.value.message
    assert "LoRA" in (e.value.hint or "")
    assert any("keyed" in w and "control vector" in w for w in s.warnings)
    # add と混ぜたときは keyed を外して警告し、add だけを書く(control vector の bake も同じ)
    s.warnings.clear()
    with pytest.raises(GneError):
        s.bake(r, tmp_path / "b.gguf")
    assert any("keyed" in w and "bake" in w for w in s.warnings)


def test_keyed_lora_matrices_are_alpha_d_kappa(tmp_path):
    """gne gain / headproj export の LoRA: blk.L.ffn_down.weight.lora_a/b で B @ A = α r_L κᵀ。記録は gne.keyed(パスなし)。"""
    s, _, raw, keys = _session(tmp_path)
    r = s.recipe(units=["1:ffn:keyed:a1.5@d1#k1", "2:ffn:keyed:a-0.5@d1#k1"], name="kw")
    d = s.gain(r, tmp_path / "g.lora.gguf")
    rd, t = _tensors(tmp_path / "g.lora.gguf")
    for L, a in ((1, 1.5), (2, -0.5)):
        A, B = _lora(t, f"blk.{L}.ffn_down.weight")
        assert A.shape == (1, N_FF) and B.shape == (N_EMBD, 1)
        exp = a * np.outer(raw[L].astype(np.float64), keys["k1"].kappa[L].astype(np.float64))
        np.testing.assert_allclose(B @ A, exp, rtol=1e-5, atol=1e-6)
    assert d["keyed_rank"] == {"blk.1.ffn_down.weight": 1, "blk.2.ffn_down.weight": 1}
    assert d["keyed_layers"]["1"]["keys"] == ["k1"] and d["keyed_layers"]["1"]["rank"] == 1
    assert "proj_rank" not in d
    meta = json.loads(rd.fields["gne.keyed"].contents())
    assert {(u["layer"], u["alpha"], u["key"], u["direction"]) for u in meta["units"]} == {(1, 1.5, "k1", "d1"), (2, -0.5, "k1", "d1")}
    assert all(u["key_sha"] == keys["k1"].sha for u in meta["units"]) and meta["alpha_ref"] == 1.0
    assert str(tmp_path) not in json.dumps(meta)
    # headproj export も同じ経路(keyed だけのレシピを受ける)
    h = s.headproj_export(r, tmp_path / "h.lora.gguf")
    _, t2 = _tensors(tmp_path / "h.lora.gguf")
    np.testing.assert_array_equal(np.array(t2["blk.1.ffn_down.weight.lora_b"].data), np.array(t["blk.1.ffn_down.weight.lora_b"].data))
    assert h["keyed_rank"] == d["keyed_rank"]
    # 既定のファイル名: keyed だけなら -keyed
    assert s.default_export_path(r).name == "kw-keyed.lora.gguf"
    assert s.default_export_path(r, bake=True).name == "kw-keyed.gguf"
    assert s.default_export_path(s.recipe(units=["1:H0:s2@d1"], name="p"), bake=True).name == "p-headproj.gguf"


def test_keyed_merges_with_proj_and_head_gain(tmp_path):
    """同じ attn_output に head gain・proj・keyed(attn)を連結して 1 つのテンソル。B @ A = 3 つの変化量の和。
    proj_rank は proj の分だけ、keyed_rank は keyed の分だけ数える。"""
    s, mats, raw, keys = _session(tmp_path)
    r = s.recipe(units=["1:H0:s2@d1", "1:H1:x0", "1:attn:keyed:a2@d1#ka", "2:ffn:keyed:a1@d1#k1"])
    d = s.gain(r, tmp_path / "g.lora.gguf")
    rd, t = _tensors(tmp_path / "g.lora.gguf")
    A, B = _lora(t, "blk.1.attn_output.weight")
    assert A.shape == (HEAD_DIM + 2, N_HEAD * HEAD_DIM) and B.shape == (N_EMBD, HEAD_DIM + 2)
    W = mats["blk.1.attn_output.weight"].astype(np.float64)
    r1 = raw[1].astype(np.float64)
    rr = r1 / np.linalg.norm(r1)
    exp = np.zeros_like(W)
    exp[:, 3:6] = -W[:, 3:6]                                          # H1 ×0
    exp[:, 0:3] -= 2.0 * np.outer(rr, rr @ W[:, 0:3])                  # H0 の方向成分を 2 倍ぶん引く
    exp += 2.0 * np.outer(r1, keys["ka"].kappa[1].astype(np.float64))  # keyed attn: α r_L κᵀ
    np.testing.assert_allclose(B @ A, exp, atol=1e-4)
    assert d["proj_rank"] == {"1": 1} and d["proj_layers"]["1"]["rank"] == 1
    assert d["keyed_rank"] == {"blk.1.attn_output.weight": 1, "blk.2.ffn_down.weight": 1}
    assert "gne.headproj" in rd.fields and "gne.keyed" in rd.fields


def test_merge_tensors_3d_experts():
    """専門家の 3 次元 A (E, r, in)・B (E, out, r) も軸 −2・−1 で連結して B @ A が和になる。形が合わなければ断る。"""
    from gne.headproj import merge_tensors
    rng = np.random.default_rng(0)
    A1, B1 = rng.standard_normal((3, 1, 10)), rng.standard_normal((3, 16, 1))
    A2, B2 = rng.standard_normal((3, 2, 10)), rng.standard_normal((3, 16, 2))
    m = merge_tensors({"x": (A1, B1)}, {"x": (A2, B2)})
    A, B = m["x"]
    assert A.shape == (3, 3, 10) and B.shape == (3, 16, 3)
    np.testing.assert_allclose(B @ A, B1 @ A1 + B2 @ A2)
    with pytest.raises(GneError):
        merge_tensors({"x": (A1, B1)}, {"x": (rng.standard_normal((1, 10)), rng.standard_normal((16, 1)))})


def test_f16_bake_adds_exactly_the_lora_delta(tmp_path):
    """keyed の焼き込みの既定(auto)は、元が Q8_0 でないテンソル(ここでは F32)を F16 にする。書き換えたテンソルの差 =
    B @ A(= α r_L κᵀ)、他のテンソルは元のバイトのまま。"""
    s, mats, raw, keys = _session(tmp_path)
    r = s.recipe(units=["1:ffn:keyed:a1.5@d1#k1"], name="kb")
    d = s.gain(r, tmp_path / "baked.gguf", bake=True)                 # requant=auto → 元が F32 なので F16
    assert [(m["tensor"], m["to"]) for m in d["modified"]] == [("blk.1.ffn_down.weight", "F16")]
    assert d["requant_error_max"] < 0.01 and d["keyed_rank"] == {"blk.1.ffn_down.weight": 1}
    side = json.loads(Path(d["sidecar"]).read_text(encoding="utf-8"))
    assert side["keyed"]["units"][0]["key"] == "k1" and side["headproj"] is None
    rd, t = _tensors(tmp_path / "baked.gguf")
    got = np.array(t["blk.1.ffn_down.weight"].data, dtype=np.float32).reshape(N_EMBD, N_FF)
    W = mats["blk.1.ffn_down.weight"].astype(np.float64)
    exp = W + 1.5 * np.outer(raw[1].astype(np.float64), keys["k1"].kappa[1].astype(np.float64))
    np.testing.assert_allclose(got, exp.astype(np.float16).astype(np.float32), rtol=0, atol=1e-6)
    for name, m in mats.items():
        if name != "blk.1.ffn_down.weight":
            assert np.array(t[name].data).tobytes() == m.tobytes(), name
    assert "gne.keyed" in rd.fields
    assert json.loads(rd.fields["gne.patches"].contents())[0]["tensor"] == "blk.1.ffn_down.weight"
    # headproj export の焼き込みも requant を省けば auto(元が F32 なので F16)
    h = s.headproj_export(r, tmp_path / "baked2.gguf", bake=True)
    assert [m["to"] for m in h["modified"]] == ["F16"] and h["requant"] == "auto" and d["requant"] == "auto"
    assert h["edit_kept_min"] == pytest.approx(1.0, abs=1e-3)
    # q8_0 にすると、量子化し直しのずれ(requant_error)と残った編集の割合(edit_kept)を測る。元が F32 の行列は Q8_0 の
    # 刻みに乗っていないので、小さな編集でも丸めのずれは大きいが、編集の向きの成分は平均では残る(警告は edit_kept で決める。
    # 元が Q8_0 で編集が埋もれる場合は test_keyed_bake_auto_keeps_q8_0_and_falls_back)
    q = s.gain(r.scaled(1e-3), tmp_path / "q.gguf", bake=True, requant="q8_0")
    assert q["requant_error_max"] > 0.5 and "edit_kept" in q["modified"][0] and "requant" not in q


def test_auto_requant_rules():
    """keyed の焼き込みの既定(auto)の型: 元が Q8_0 なら Q8_0、それ以外(K-quant・IQ・F16・BF16・F32)は F16。専門家も同じ。
    edit_kept = 1 + ⟨戻した値 − 書きたい値, ΔW⟩ / ‖ΔW‖²(編集が全部残れば 1、消えれば 0)。"""
    from gguf import GGMLQuantizationType as T

    from gne.gain import KEYED_KEEP_MIN, _expert_qtype, auto_qtype, edit_kept
    assert KEYED_KEEP_MIN == 0.9
    assert auto_qtype(T.Q8_0) == T.Q8_0
    for src in (T.Q4_K, T.Q6_K, T.IQ2_XS, T.F16, T.BF16, T.F32, T.Q5_0):
        assert auto_qtype(src) == T.F16, src
    assert _expert_qtype(T.Q8_0, "auto") == T.Q8_0 and _expert_qtype(T.Q5_0, "auto") == T.F16
    rng = np.random.default_rng(0)
    W, dW = rng.standard_normal((4, 8)), rng.standard_normal((4, 8)) * 0.1
    assert edit_kept(W + dW, W + dW, dW) == pytest.approx(1.0)
    assert edit_kept(W, W + dW, dW) == pytest.approx(0.0)                 # 丸めで元に戻った = 編集が消えた
    assert edit_kept(W + 0.5 * dW, W + dW, dW) == pytest.approx(0.5)
    noise = rng.standard_normal((4, 8)) * 0.05
    noise -= (np.sum(noise * dW) / np.sum(dW * dW)) * dW                 # 編集に直交するゆらぎは残り方に入らない
    assert edit_kept(W + dW + noise, W + dW, dW) == pytest.approx(1.0)


def test_keyed_bake_auto_keeps_q8_0_and_falls_back(tmp_path):
    """元が Q8_0 のテンソルは、既定(auto)で Q8_0 のまま書く(大きさが変わらない)。Q8_0 のままでは編集の 1 割以上が丸めに
    消える小さな編集だけ F16 にして(記録の fallback と警告)、keep を明示すれば Q8_0 のまま(丸めに消えた警告)。"""
    from gne.schemas import GainOut
    s, mats, raw, keys = _session(tmp_path, q8=True)
    name = "blk.1.ffn_down.weight"
    src = s.weights.tensor(name)
    r = s.recipe(units=["1:ffn:keyed:a1.5@d1#k1"], name="kq")
    env = {"schema_version": "1", "command": "headproj export", "model_id": "m", "llama_cpp_version": "0", "seed": 0,
           "generated_at": "2026-11-08T00:00:00+00:00", "warnings": []}
    for d in (s.gain(r, tmp_path / "g.gguf", bake=True), s.headproj_export(r, tmp_path / "h.gguf", bake=True)):
        rec = d["modified"][0]
        assert d["requant"] == "auto" and rec["tensor"] == name and rec["to"] == "Q8_0" and "fallback" not in rec
        assert rec["bytes_to"] == rec["bytes_from"] == int(src.n_bytes)
        assert rec["edit_kept"] >= 0.9 and d["edit_kept_min"] == rec["edit_kept"]
        GainOut.model_validate({**env, **d})
    rd, t = _tensors(tmp_path / "h.gguf")
    from gne.weights import tensor_to_f32
    got = tensor_to_f32(t[name]).reshape(N_EMBD, N_FF)
    want = mats[name] + 1.5 * np.outer(raw[1], keys["k1"].kappa[1])
    assert np.abs(got - want).max() <= np.abs(want).max() / 127 + 1e-6          # Q8_0 の丸めまで
    # 小さな編集: Q8_0 のままでは埋もれるので F16
    tiny = r.scaled(1e-4)
    s.warnings.clear()
    f = s.headproj_export(tiny, tmp_path / "t.gguf", bake=True)
    rec = f["modified"][0]
    assert rec["to"] == "F16" and rec["fallback"]["from"] == "Q8_0" and rec["fallback"]["edit_kept"] < 0.9
    assert rec["edit_kept"] == pytest.approx(1.0, abs=0.2)
    assert any("F16 で書きました" in w for w in s.warnings) and not any("丸めに消えて" in w for w in s.warnings)
    GainOut.model_validate({**env, **f})
    s.warnings.clear()
    k = s.headproj_export(tiny, tmp_path / "k.gguf", bake=True, requant="keep")
    assert k["modified"][0]["to"] == "Q8_0" and k["edit_kept_min"] < 0.9
    assert any("丸めに消えて" in w and "edit_kept_min" in w for w in s.warnings)
    assert s.headproj_export(r, tmp_path / "f.gguf", bake=True, requant="f16")["modified"][0]["to"] == "F16"
    with pytest.raises(GneError) as e:
        s.headproj_export(r, tmp_path / "x.gguf", bake=True, requant="q4_k")
    assert e.value.code == "BAD_ARGS"


def test_split_bake_auto_in_place_and_fallback(tmp_path):
    """分割 GGUF への焼き込みの auto: 元が Q8_0 のテンソルはその場に書き戻し(分割ファイルの大きさは同じ)、編集が埋もれる
    小さな編集だけ F16 にしてその分割ファイルを書き直す(結果の fallback)。"""
    from gne.weights import Weights
    s, mats, raw, keys = _session(tmp_path, split=6, q8=True)
    name = "blk.1.ffn_down.weight"
    r = s.recipe(units=["1:ffn:keyed:a1.5@d1#k1"], name="sq")
    d = s.headproj_export(r, tmp_path / "out" / "sq.gguf", bake=True)
    side = json.loads(Path(d["sidecar"]).read_text(encoding="utf-8"))
    assert d["requant"] == "auto" and side["modified"][0]["to"] == "Q8_0" and "fallback" not in d
    assert d["edit_kept_min"] >= 0.9
    for src, out in zip(s.weights.gguf.parts, d["parts"], strict=True):
        assert Path(src).stat().st_size == Path(out).stat().st_size
    assert Weights(d["path"]).tensor(name).tensor_type.name == "Q8_0"
    s.warnings.clear()
    f = s.headproj_export(r.scaled(1e-4), tmp_path / "out2" / "sq.gguf", bake=True)
    side = json.loads(Path(f["sidecar"]).read_text(encoding="utf-8"))
    assert f["fallback"] == [name] and side["modified"][0]["to"] == "F16" and side["modified"][0]["fallback"]["from"] == "Q8_0"
    assert Weights(f["path"]).tensor(name).tensor_type.name == "F16"
    assert any("F16 で書きました" in w for w in s.warnings)


def test_patch_requant_refused_for_keyed(tmp_path):
    from gne.gain import _patch_targets, bake_gain_gguf
    s, _, _, _ = _session(tmp_path)
    r = s.recipe(units=["1:ffn:keyed:a1@d1#k1"])
    with pytest.raises(GneError) as e:
        s.gain(r, tmp_path / "p.gguf", bake=True, requant="patch")
    assert e.value.code == "UNSUPPORTED_QUANT"
    assert _patch_targets(s.weights, r) is None
    with pytest.raises(GneError) as e2:                               # 直接呼んでもパッチにはならない
        bake_gain_gguf(s.weights, r, tmp_path / "p2.gguf", requant="patch")
    assert e2.value.code == "UNSUPPORTED_QUANT"


def test_runtime_cache_path_tracks_key_and_direction(tmp_path):
    """実行時の LoRA: 全体の α を k 倍しても同じファイル(倍率 α_ref が変わる)。鍵や方向の中身を変えたら別のパス。"""
    s, _, raw, keys = _session(tmp_path)
    r = s.recipe(units=["1:ffn:keyed:a1.5@d1#k1", "2:ffn:keyed:a-3@d1#k1"])
    p1, ref1 = s.keyed_lora(r)
    assert p1.exists() and p1.parent == key_dir(s) / "lora" and p1.name.startswith("keyed-") and ref1 == 3.0
    assert s.lora_for(r) == [(str(p1), 3.0)]
    _, t = _tensors(p1)
    A, B = _lora(t, "blk.2.ffn_down.weight")
    np.testing.assert_allclose(B @ A * ref1, -3.0 * np.outer(raw[2], keys["k1"].kappa[2]), rtol=1e-5, atol=1e-6)
    p2, ref2 = s.keyed_lora(r.scaled(0.5))
    assert p2 == p1 and ref2 == 1.5
    # α の比が変われば別のファイル
    assert s.keyed_lora(s.recipe(units=["1:ffn:keyed:a1@d1#k1", "2:ffn:keyed:a-3@d1#k1"]))[0] != p1
    # 方向の中身を変える
    dp = s.store.direction_dir(s.meta.model_id) / "d1.npz"
    _dirs(dp, seed=9)
    os.utime(dp, ns=(dp.stat().st_atime_ns, dp.stat().st_mtime_ns + 10**9))
    p3, _ = s.keyed_lora(r)
    assert p3 != p1
    # 鍵の中身を変える
    _key(s, "k1", "ffn", [1, 2], N_FF, seed=11)
    kp = key_dir(s) / "k1.npz"
    os.utime(kp, ns=(kp.stat().st_atime_ns, kp.stat().st_mtime_ns + 10**9))
    p4, _ = s.keyed_lora(r)
    assert p4 not in (p1, p3)
    # α が全部 0 なら LoRA は当てない
    assert s.lora_for(s.recipe(units=["1:ffn:keyed:a0@d1#k1"])) is None


def test_mtp_and_last_layer_refused(tmp_path):
    s, _, _, _ = _session(tmp_path, n_layer=5, nextn=1)               # 層 4 が MTP、本流の最終層は 3
    _key(s, "k3", "ffn", [3, 4], N_FF)
    for spec, word in (("4:ffn:keyed:a1@d1#k3", "MTP"), ("3:ffn:keyed:a1@d1#k3", "最終層")):
        with pytest.raises(GneError) as e:
            s.recipe(units=[spec])
        assert e.value.code == "BAD_RECIPE" and word in e.value.message, spec


def test_recipe_checks_keys_directions_and_layers(tmp_path):
    s, _, _, _ = _session(tmp_path)
    r = s.recipe(units=["1:ffn:keyed:a1@d1"], default_key="k1")       # default_key 引数
    assert r.default_key == "k1"
    saved = s.recipe_save("ks", ["1:ffn:keyed:a1"], default_direction="d1", default_key="k1")
    back = Recipe.load(saved["path"])
    assert back.default_key == "k1" and back.default_direction == "d1" and back.units[0].key is None
    cases = {"1:ffn:keyed:a1@d1#nokey": "FILE_NOT_FOUND",              # 鍵が無い
             "1:ffn:keyed:a1@nodir#k1": "FILE_NOT_FOUND",              # 方向が無い
             "0:ffn:keyed:a1@d1#k1": "BAD_KEY",                        # 層 0 は鍵に無い
             "1:ffn:keyed:a1@d1#ka": "BAD_RECIPE",                     # 鍵の書き込み先(attn)とユニット(ffn)が違う
             "1:S:keyed:a1@d1#k1": "BAD_RECIPE"}                       # shared expert の無い層
    for spec, code in cases.items():
        with pytest.raises(GneError) as e:
            s.recipe(units=[spec])
        assert e.value.code == code, (spec, e.value.message)
    _key(s, "kbad", "ffn", [1], N_FF + 2)                             # 長さが ffn_down の入力と違う
    with pytest.raises(GneError) as e:
        s.recipe(units=["1:ffn:keyed:a1@d1#kbad"])
    assert e.value.code == "BAD_KEY"


def test_recipe_warns_when_gain_and_ablate(tmp_path):
    """レビュー P3(when は keyed・gain・proj に効かない)と P17(同じ層の gain、同じ方向の ablate)の警告。"""
    s, _, _, _ = _session(tmp_path)
    s.recipe(units=["1:ffn:keyed:a1@d1#k1", "1:5:x2"], when="gen", ablate="d1")
    w = " / ".join(s.warnings)
    assert "when=gen" in w and "keyed・gain" in w
    assert "同じ層の gain" in w and "L1 N5" in w
    assert "ablate" in w and "消えます" in w
    s.warnings.clear()
    s.recipe(units=["1:ffn:keyed:a1@d1#k1", "2:5:x2"])                # 別の層の gain は警告しない
    assert not any("同じ層の gain" in x for x in s.warnings)


def test_unit_details_and_unit_spec(tmp_path):
    """unit_details の keyed の行は α‖r_L‖/ρ_L(dose_unit に関係なく、残差ノルム統計があれば)。_unit_spec は keyed の記法。"""
    from gne.api import _unit_spec
    from gne.stats import ResidStats
    s, _, raw, _ = _session(tmp_path)
    r = s.recipe(units=["1:ffn:keyed:a1.5@d1#k1"], dose_unit="raw")
    _, det = s.vectors(r)
    assert det[0]["mode"] == "keyed" and det[0]["rho"] is None
    assert any("残差ノルム統計" in w for w in s.warnings)
    Ls = np.arange(4)
    ResidStats(Ls, Ls * 0 + 1.0, np.array([10.0, 20.0, 30.0, 40.0]), Ls * 0 + 50.0, Ls * 0 + 25.0, 100, "x", 64).save(
        s.store.resid_path(s.meta.model_id))
    s._resid = None
    _, det = s.vectors(r)
    rn = float(np.linalg.norm(raw[1]))
    assert det[0]["rho"] == 20.0 and det[0]["rel_push"] == pytest.approx(1.5 * rn / 20.0, rel=1e-5)
    assert det[0]["key"] == "k1" and det[0]["direction"] == "d1"
    assert _unit_spec(r.to_dict()["units"][0]) == "1:ffn:keyed:a1.5@d1#k1"
    assert _unit_spec(Recipe([r.units[0]], default_key="k1").to_dict()["units"][0]).startswith("1:ffn:keyed:a1.5")


def test_moe_shared_and_expert_keyed(tmp_path):
    """MoE の層: shared expert の keyed はレシピで落ちない(ニューロン範囲の確かめに当たらない)。専門家 e の keyed は
    3 次元の LoRA(e のスライスだけ)と、焼き込みで W_e += B_e A_e(他の専門家はそのまま)。階数は 3 次元でも 1。"""
    s, mats, raw, _ = _session(tmp_path, moe_layers=(1,), shexp=True)
    ks = _key(s, "ks", "shared", [1], N_SHEXP, seed=3)
    ke = _key(s, "ke", "expert:1", [1], N_FF_EXP, seed=4)
    r = s.recipe(units=["1:S:keyed:a1@d1#ks", "1:E1:keyed:a2@d1#ke"], name="moe")
    assert any("gate_a" in w for w in s.warnings)                   # ゲートが掛かることの注意
    d = s.gain(r, tmp_path / "m.lora.gguf")
    assert d["keyed_rank"] == {"blk.1.ffn_down_shexp.weight": 1, "blk.1.ffn_down_exps.weight": 1}
    _, t = _tensors(tmp_path / "m.lora.gguf")
    A, B = _lora(t, "blk.1.ffn_down_exps.weight")
    assert A.shape == (N_EXP, 1, N_FF_EXP) and B.shape == (N_EXP, N_EMBD, 1)
    assert not A[0].any() and not A[2].any() and not B[0].any() and not B[2].any()
    np.testing.assert_allclose(B[1] @ A[1], 2.0 * np.outer(raw[1], ke.kappa[1]), rtol=1e-5, atol=1e-6)
    b = s.gain(r, tmp_path / "mb.gguf", bake=True)
    assert {m["tensor"]: m["to"] for m in b["modified"]} == {"blk.1.ffn_down_exps.weight": "F16",
                                                             "blk.1.ffn_down_shexp.weight": "F16"}
    _, tb = _tensors(tmp_path / "mb.gguf")
    got = np.array(tb["blk.1.ffn_down_exps.weight"].data, dtype=np.float32).reshape(N_EXP, N_EMBD, N_FF_EXP)
    W = mats["blk.1.ffn_down_exps.weight"]
    np.testing.assert_allclose(got[0], W[0].astype(np.float16).astype(np.float32))
    np.testing.assert_allclose(got[2], W[2].astype(np.float16).astype(np.float32))
    np.testing.assert_allclose(got[1], W[1] + 2.0 * np.outer(raw[1], ke.kappa[1]), atol=5e-3)
    gs = np.array(tb["blk.1.ffn_down_shexp.weight"].data, dtype=np.float32).reshape(N_EMBD, N_SHEXP)
    np.testing.assert_allclose(gs, mats["blk.1.ffn_down_shexp.weight"] + np.outer(raw[1], ks.kappa[1]), atol=5e-3)


def test_split_bake_takes_keyed(tmp_path):
    """分割 GGUF への焼き込みも keyed だけのレシピを受ける(requant を省けば auto。元が F32 なので F16)。書き換えた
    分割ファイルだけ変わる。"""
    import hashlib
    s, mats, raw, keys = _session(tmp_path, split=6)
    assert s.weights.gguf.is_split
    r = s.recipe(units=["1:ffn:keyed:a1.5@d1#k1"], name="sp")
    d = s.headproj_export(r, tmp_path / "out" / "sp.gguf", bake=True)
    assert d["requant"] == "auto" and d["modified"] == ["blk.1.ffn_down.weight"] and d["requant_error_max"] < 0.01
    assert d["keyed_rank"] == {"blk.1.ffn_down.weight": 1}
    side = json.loads(Path(d["sidecar"]).read_text(encoding="utf-8"))
    assert side["keyed"]["units"][0]["layer"] == 1 and side["modified"][0]["to"] == "F16" and side["requant"] == "auto"
    from gne.weights import Weights, tensor_to_f32
    w2 = Weights(d["path"])
    got = tensor_to_f32(w2.tensor("blk.1.ffn_down.weight")).reshape(N_EMBD, N_FF)
    exp = mats["blk.1.ffn_down.weight"] + 1.5 * np.outer(raw[1], keys["k1"].kappa[1])
    np.testing.assert_allclose(got, exp, atol=5e-3)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    changed = {x["file"] for x in side["modified"]}
    for src, out in zip(s.weights.gguf.parts, d["parts"], strict=True):
        assert (digest(src) == digest(out)) == (Path(out).name not in changed)


# --- モデルを使う(Qwen3-0.6B、CPU なら GNE_TEST_NGL=0)------------------------------------------------

LAYER = 14
TEXT_A = "今日はいい天気なのだ。公園でお弁当を食べるのだ。ずんだ餅も持っていくのだ。"
TEXTS_B = ["The weather is nice today. I will eat lunch in the park with my friends.",
           "昨日は図書館で本を読みました。とても静かで、勉強がはかどりました。"]


@pytest.fixture(scope="module")
def msession(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip(f"{MODEL} がありません")
    from gne.api import Session
    s = Session(MODEL, root=tmp_path_factory.mktemp("c07x"), n_gpu_layers=NGL, n_ctx=2048)
    s.direction(contrast=TSUN, positions="last")
    _model_key(s)
    yield s
    s.close()


def _model_key(s, name="v30k"):
    """捕捉した ffn_down の入力から鍵を作る: μ = 側 A(TEXT_A)の位置 1 以降の平均、Q = 側 B(TEXTS_B)の全トークンが張る空間。"""
    from gne.attn import capture_attention
    from gne.capture import pick_act_name
    e = s.engine
    names = e.list_names(e.tokenize(TEXT_A, add_special=True, parse_special=False)[:16])
    act = pick_act_name(names, LAYER)
    got = capture_attention(s, [TEXT_A] + TEXTS_B, layers=[LAYER], want_pattern=False, want_head_out=False, extra=[act])
    e.free_attn_probe()
    xa = got[0].extra[act][1:].astype(np.float64)
    xb = [g.extra[act][1:].astype(np.float64) for g in got[1:]]
    kap, info = make_key(xa.mean(axis=0), orth_basis(np.concatenate(xb), energy=1.0))
    ks = KeySet(target="ffn", kappa={LAYER: kap.astype(np.float32)}, mu={LAYER: xa.mean(axis=0).astype(np.float32)},
                kn={LAYER: info["kn"]}, gain={LAYER: info["gain"]}, q_rank={LAYER: info["q_rank"]},
                exb={LAYER: np.stack([x.mean(axis=0) for x in xb]).astype(np.float32)}, q_mode="samples",
                positions="all", model_id=s.meta.model_id)
    ks.save(key_dir(s) / f"{name}.npz")
    return ks


@pytest.mark.model
def test_v30_keyed_lora_writes_alpha_kx_d(msession):
    """V30: LoRA を当てた ffn_out の差 = α (κᵀx) d(|κᵀx| の大きいトークンで相対誤差の中央値 < 3%)。鍵の作成に使った
    側 B の例では |Δ̄| / (|α|·‖d‖) < 0.1。α = 1 と 2 は同じファイルを倍率で当てる。"""
    d = msession.check_keyed(LAYER, key="v30k", direction="tsun_endings", alphas=(1.0, 2.0),
                             texts=[TEXT_A, "Tell me a short story about a lighthouse keeper."], texts_b=TEXTS_B)
    assert d["v30"], d
    assert d["out_name"] == f"ffn_out-{LAYER}" and d["judged_leak"]
    for row in d["rows"]:
        assert row["rel_err_median"] < 0.03 and row["cos_median"] > 0.99, row
        assert row["leak_b"] < 0.1 and row["leak_b_source"] == "texts_b", row
        assert row["kx_max"] > 0.5                                    # 側 A の文では鍵が合う
    assert [r["adapter_scale"] for r in d["rows"]] == [1.0, 2.0]
    assert d["leak_b_key_examples"] < 1e-3                            # 例の平均は Q が張る空間にある
    # 符号: 方向の先頭の - は α の符号に畳む
    neg = msession.check_keyed(LAYER, key="v30k", direction="-tsun_endings", alphas=(1.0,), texts=[TEXT_A])
    assert neg["rows"][0]["alpha"] == -1.0 and neg["rows"][0]["ok"], neg


@pytest.mark.model
def test_v30_attn_output_keyed(msession):
    """V30 の attention 版: 入力はヘッド出力の連結(CHANGE-06 の head_out)、書き込み行列は attn_output。"""
    from gne.attn import capture_attention
    got = capture_attention(msession, [TEXT_A] + TEXTS_B, layers=[LAYER], want_pattern=False)
    msession.engine.free_attn_probe()
    xs = [g.head_out[LAYER].reshape(g.head_out[LAYER].shape[0], -1)[1:].astype(np.float64) for g in got]
    kap, info = make_key(xs[0].mean(axis=0), orth_basis(np.concatenate(xs[1:]), energy=1.0))
    KeySet(target="attn", kappa={LAYER: kap.astype(np.float32)}, mu={LAYER: xs[0].mean(axis=0).astype(np.float32)},
           kn={LAYER: info["kn"]}, gain={LAYER: info["gain"]}, model_id=msession.meta.model_id).save(key_dir(msession) / "v30a.npz")
    d = msession.check_keyed(LAYER, key="v30a", direction="tsun_endings", kind="attn", alphas=(1.5,),
                             texts=[TEXT_A, "Tell me a short story."], texts_b=TEXTS_B)
    assert d["v30"] and d["tensor"] == f"blk.{LAYER}.attn_output.weight", d
    assert d["rows"][0]["rel_err_median"] < 0.03 and d["rows"][0]["leak_b"] < 0.1, d
    with pytest.raises(GneError) as e:                                # 鍵の書き込み先(attn)と kind が違う(MoE は test_change07_moe.py)
        msession.check_keyed(LAYER, key="v30a", direction="tsun_endings", kind="shared")
    assert e.value.code == "BAD_RECIPE"


def test_bake_score_check_adds_noise_only_on_last_layer():
    """V26・V33 の点数の判定(api.bake_score_check): 層・側ごとに例ごとの差の最大を許容誤差(両側の点数の標準偏差の 1 割)と
    比べ、最後の層だけ合格の線に計算方法の違いだけで出る誤差(noise)を足す。基準の層のずれは noise では通らない。"""
    from gne.api import NOISE_LORA_SCALE, bake_score_check
    assert NOISE_LORA_SCALE == 1e-6                      # 0 ちょうどだと LoRA が外れて計算の仕方が変わるおそれ
    rng = np.random.default_rng(0)
    A = {14: rng.normal(5, 1, 8), 26: rng.normal(20, 10, 8)}
    B = {14: rng.normal(0, 1, 8), 26: rng.normal(0, 10, 8)}
    lora = {"A": {"values": A}, "B": {"values": B}}
    res = {L: 0.1 * float(np.std(np.concatenate([A[L], B[L]]))) for L in A}
    bA = {14: A[14] + 0.5 * res[14], 26: A[26].copy()}
    bA[26][3] += 1.5 * res[26]                           # 最後の層の側 A の 1 例だけ、許容誤差の 1.5 倍ずれる
    baked = {"A": {"values": bA}, "B": {"values": dict(B)}}
    s0, ok0 = bake_score_check(lora, baked, [14, 26], last=26)
    assert not ok0 and s0["26A"]["ok"] is False and s0["26B"]["ok"] and s0["14A"]["ok"]
    assert s0["26A"]["max_abs_diff"] == pytest.approx(1.5 * res[26]) and s0["26A"]["resolution"] == pytest.approx(res[26])
    s1, ok1 = bake_score_check(lora, baked, [14, 26], last=26, noise=0.6 * res[26])
    assert ok1 and s1["26A"]["threshold"] == pytest.approx(1.6 * res[26]) and s1["26A"]["noise"] == pytest.approx(0.6 * res[26])
    assert s1["14A"]["threshold"] == pytest.approx(res[14]) and "noise" not in s1["14A"]
    bad = {"A": {"values": {14: A[14] + 1.5 * res[14], 26: A[26]}}, "B": {"values": dict(B)}}
    s2, ok2 = bake_score_check(lora, bad, [14, 26], last=26, noise=10 * res[14])
    assert not ok2 and s2["14A"]["ok"] is False                                  # 基準の層は noise を足さない
    s3, ok3 = bake_score_check(lora, baked, [14, 26], last=None, noise=1e9)     # 最後の層が基準の層と同じなら足さない
    assert not ok3 and "noise" not in s3["26A"]


@pytest.mark.model
def test_v33_bake_matches_lora_f16(msession, tmp_path):
    """V33: keyed の焼き込み(F16)と LoRA 版が V26 の 4 項目で合う。実行時の LoRA は lora_for で当たる。"""
    r = msession.recipe(units=[f"{LAYER}:ffn:keyed:a4@tsun_endings#v30k"], name="v33")
    lo = msession.lora_for(r)
    assert lo and lo[0][1] == 4.0 and "keys" in lo[0][0]
    d = msession.headproj_export(r, tmp_path / "v33.gguf", bake=True, requant="f16", verify=True, contrast=TSUN)
    assert [m["to"] for m in d["modified"]] == ["F16"] and d["modified"][0]["tensor"] == f"blk.{LAYER}.ffn_down.weight"
    v = d["verify"]
    assert v["pass"] and v["top1_match"] >= 0.9, v
    assert all(x["ok"] for x in v["scores"].values()), v
    # 最後の層だけ、合格の線 = 許容誤差 + 計算方法の違いだけで出る誤差(書き込みがほぼ 0 の LoRA と LoRA なしの差)
    from gne.schemas import BakeVerify
    BakeVerify.model_validate(v)
    last, base = v["last_layer"], v["base_layer"]
    assert last is not None and last != base, v
    nz = v["noise"]
    assert nz["layer"] == last and nz["lora_scale"] == 1e-6 and nz["max_abs_diff"] >= 0 and nz["seconds"] > 0
    assert nz["max_abs_diff"] == max(nz["by_side"].values())
    for side in "AB":
        x = v["scores"][f"{last}{side}"]
        assert x["noise"] == nz["max_abs_diff"] and x["threshold"] == pytest.approx(x["resolution"] + nz["max_abs_diff"])
        if base is not None:                                                             # 基準の層はそのまま
            y = v["scores"][f"{base}{side}"]
            assert "noise" not in y and y["threshold"] == y["resolution"]
    # 鍵が合う文(側 A)では編集が効いていて、焼き込み版と LoRA 版の差はその効き目よりずっと小さい
    from gne.engine import Engine
    ids = msession.engine.tokenize(TEXT_A, add_special=True, parse_special=False)
    msession.close()

    def logits(path, lora):
        e = Engine(path, n_gpu_layers=NGL, n_ctx=512, n_seq_max=1)
        try:
            ctx = e.steer_ctx()
            e.set_lora(ctx, lora)
            e.clear_memory(ctx)
            return e.prefill(ctx, [ids]).logits.copy()
        finally:
            e.close()
    a, b, c = logits(MODEL, lo), logits(d["path"], None), logits(MODEL, None)
    assert np.abs(a - b).max() < 0.2 * np.abs(a - c).max(), (np.abs(a - b).max(), np.abs(a - c).max())
