"""CHANGE-02 A のテスト: gne direction(方向)・方向のユニット(L:dir:NAME:dose)・方向を消す LoRA(V13 の LoRA 版)。

モデルを使うものは Qwen3-0.6B(GNE_TEST_MODEL)。無ければ飛ばす。
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from gne.steer import GneError, Recipe, parse_unit

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))


# --- モデル不要 ----------------------------------------------------------------------------------------

def test_direction_unit_parse_and_recipe(tmp_path):
    u = parse_unit("12:dir:yandere:+0.5")
    assert (u.layer, u.kind, u.direction, u.dose, u.neuron) == (12, "direction", "yandere", 0.5, None)
    assert u.label() == "L12 dir:yandere +0.5"
    with pytest.raises(GneError):
        parse_unit("12:dir:yandere:x2")                       # gain はニューロンだけ
    r = Recipe([u, parse_unit("3:100:-1")], name="mix", when="gen")
    r.save(tmp_path / "r.yaml")
    d = yaml.safe_load((tmp_path / "r.yaml").read_text())
    assert d["units"][0] == {"layer": 12, "dose": 0.5, "kind": "direction", "direction": "yandere"}
    back = Recipe.load(tmp_path / "r.yaml")
    assert back.units[0] == u and back.scaled(0.5).units[0].direction == "yandere" and back.scaled(0.5).units[0].dose == 0.25
    with pytest.raises(GneError):
        Recipe([u], dose_unit="absmax").validate()             # absmax はニューロン用


def test_candidate_layers_and_contrast(tmp_path):
    from gne.direction import candidate_layers, load_contrast
    assert candidate_layers(28) == list(range(8, 23))         # 0.3·28 〜 0.8·28
    assert candidate_layers(28, "10-16:3") == [10, 13, 16] and candidate_layers(28, "0,5,27,30") == [5]
    with pytest.raises(GneError):
        candidate_layers(28, "x")
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"name": "c", "system_a": "A", "system_b": "B", "user_prompts": ["hi", "yo"],
                                 "validate": {"match": "love|愛", "target": "clingy"}}))
    _, v = load_contrast(p)
    assert v.prompts == ["hi", "yo"] and v.match == "love|愛" and v.target == "clingy"
    p.write_text(yaml.safe_dump({"name": "c", "system_a": "A", "system_b": "B", "user_prompts": ["hi"],
                                 "validate": {"match": "("}}))
    with pytest.raises(GneError):
        load_contrast(p)


def _tiny_writers(path):
    """attn_output と ffn_down を持つ小さな GGUF(llama arch)。"""
    from gguf import GGUFWriter
    rng = np.random.default_rng(1)
    w = GGUFWriter(str(path), arch="llama")
    w.add_name("tiny")
    w.add_block_count(2)
    w.add_embedding_length(16)
    w.add_feed_forward_length(24)
    w.add_token_list([f"t{i}" for i in range(8)])
    mats = {}
    for L in range(2):
        for nm, shape in (("attn_output", (16, 12)), ("ffn_down", (16, 24))):
            m = rng.standard_normal(shape).astype(np.float32)
            mats[f"blk.{L}.{nm}.weight"] = m
            w.add_tensor(f"blk.{L}.{nm}.weight", m)
    w.add_tensor("token_embd.weight", rng.standard_normal((8, 16)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return mats


def test_ablation_lora_tensors(tmp_path):
    """W + B A = (I − r rᵀ) W になり、出力から r 成分が消える。LoRA GGUF の形とメタデータ。"""
    from gguf import GGUFReader

    from gne.ablate import export_ablation_lora, residual_writers, writer_info
    from gne.weights import Weights
    mats = _tiny_writers(tmp_path / "m.gguf")
    w = Weights(tmp_path / "m.gguf")
    assert residual_writers(w) == {L: [f"blk.{L}.attn_output.weight", f"blk.{L}.ffn_down.weight"] for L in range(2)}
    info = writer_info(w)
    assert info["verified"] and info["counts"] == {"attn_output": 2, "ffn_down": 2}
    assert not info["embed_in_lora"] and len(info["warnings"]) == 1 and "tied" in info["warnings"][0]   # 出力層が無い = 共有
    r = np.random.default_rng(2).standard_normal(16)
    rep = export_ablation_lora(w, r, tmp_path / "a.lora.gguf", meta={"direction": "t"})
    assert rep["n_tensors"] == 4
    rd = GGUFReader(tmp_path / "a.lora.gguf")
    ru = r / np.linalg.norm(r)
    for name, W in mats.items():
        t = {x.name: x for x in rd.tensors}
        A = np.asarray(t[f"{name}.lora_a"].data).reshape(1, -1)
        B = np.asarray(t[f"{name}.lora_b"].data).reshape(-1, 1)
        assert A.shape == (1, W.shape[1]) and B.shape == (W.shape[0], 1)
        Wp = W + B @ A
        assert np.allclose(Wp, W - np.outer(ru, ru @ W), atol=1e-5)
        assert np.abs(ru @ Wp).max() < 1e-5                                  # 出力に r 成分が無い
    assert rd.fields["general.type"].contents() == "adapter" and json.loads(rd.fields["gne.ablate"].contents())["direction"] == "t"
    # 層ごとの方向(--per-layer)
    rep2 = export_ablation_lora(w, {0: r, 1: -r}, tmp_path / "p.lora.gguf")
    t = {x.name: x for x in GGUFReader(tmp_path / "p.lora.gguf").tensors}
    assert np.allclose(np.asarray(t["blk.1.ffn_down.weight.lora_b"].data).reshape(-1), ru, atol=1e-6) and rep2["n_tensors"] == 4


# --- モデルを使う ----------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def session(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip("モデルがありません")
    from gne.api import Session
    s = Session(MODEL, root=tmp_path_factory.mktemp("c02"))
    yield s
    s.close()


def _contrast(tmp_path):
    p = tmp_path / "pirate.yaml"
    p.write_text(yaml.safe_dump({
        "name": "pirate", "system_a": "You are a pirate. Always talk like a pirate, with arr and matey.",
        "system_b": "You are a helpful assistant.",
        "user_prompts": ["Who are you?", "Tell me about the weather.", "What is your favorite food?"],
        "validate": {"match": r"\barr+\b|matey|ahoy|ye\b", "target": "a pirate"}}))
    return p


@pytest.mark.model
def test_direction_end_to_end(session, tmp_path):
    from gne.direction import DirectionSet
    from gne.reverse import ContrastSet, contrastive_directions
    from gne.schemas import COMMAND_SCHEMAS
    c = _contrast(tmp_path)
    d = session.direction(contrast=c)
    assert d["name"] == "pirate" and d["positions"] == "last" and d["n_rows"] == {"A": 3, "B": 3} and d["select"] is None
    ds = DirectionSet.load(d["path"])
    n_layer = session.meta.n_layer
    assert sorted(ds.raw) == list(range(n_layer - 1)) and ds.meta["positions"] == "last"
    cr = contrastive_directions(session.engine, ContrastSet.load(c), positions="last", with_act=False)
    assert all(np.allclose(ds.raw[L], cr.directions[L]) for L in (3, 14))               # reverse の捕捉と同じ
    z = np.load(d["path"])
    assert z["cos"].shape == (n_layer - 1, n_layer - 1) and np.allclose(np.diag(z["cos"]), 1, atol=1e-5)
    rows = {r["layer"]: r for r in d["layers"]}
    assert rows[0]["cos_prev"] is None and rows[14]["norm"] > 0
    COMMAND_SCHEMAS["direction"].model_validate({**session.envelope("direction"), **d})
    # 層選び: add(足して一致率が上がるか)と ablate(消して下がるか)
    a = session.direction(contrast=c, name="pirate", select="auto", mode="add", layers="10-14:2", dose=4.0, max_tokens=24)
    sel = a["select"]
    assert sel["mode"] == "add" and [r["layer"] for r in sel["rows"]] == [10, 12, 14] and sel["metric"] == "rate"
    assert all(r["rel_dose"] > 0 and 0 <= r["distinct4"] <= 1 and r["kl"] > 0 for r in sel["rows"])
    assert sel["selected"] in (None, 10, 12, 14) and Path(d["path"]).with_suffix(".select.json").exists()
    b = session.direction(contrast=c, name="pirate", select="auto", mode="ablate", layers="12", max_tokens=24)
    sb = b["select"]
    assert sb["mode"] == "ablate" and sb["rows"][0]["layer"] == 12 and sb["base"]["rate"] > 0      # system で海賊になる
    assert list(session.store.ablate_dir(session.meta.model_id).glob("pirate-L12-*.lora.gguf"))
    m = session.direction(contrast=c, name="pirate", select="13")
    assert m["select"] == {"mode": "manual", "selected": 13}
    with pytest.raises(GneError):
        session.direction(contrast=c, select="99")
    # 判定が無い対照集合では層を選べない
    c2 = tmp_path / "noval.yaml"
    c2.write_text(yaml.safe_dump({"name": "noval", "system_a": "A", "system_b": "B", "user_prompts": ["hi"]}))
    with pytest.raises(GneError):
        session.direction(contrast=c2, select="auto", layers="12")


@pytest.mark.model
def test_direction_unit_vectors_and_steer(session, tmp_path):
    session.direction(contrast=_contrast(tmp_path))
    ds = session.direction_set("pirate")
    r = session.recipe(units=["12:dir:pirate:+0.5"])
    vecs, det = session.vectors(r)
    rho = session.resid_stats().rho(12)
    want = 0.5 * rho * ds.raw[12] / np.linalg.norm(ds.raw[12])
    assert set(vecs) == {12} and np.allclose(vecs[12], want, rtol=1e-5)
    assert det[0]["kind"] == "direction" and det[0]["direction"] == "pirate"
    assert det[0]["diff_rel"] == pytest.approx(np.linalg.norm(ds.raw[12]) / rho, rel=1e-5)
    raw = session.vectors(session.recipe(units=["12:dir:pirate:+2"], dose_unit="raw"))[0]
    assert np.allclose(raw[12], 2 * ds.raw[12])                                                   # raw: 平均差の 2 倍
    st = session.steer(r, prompt="Who are you?", max_tokens=16)
    assert st["differs"] and st["units"][0]["direction"] == "pirate"
    assert st["steered"]["text"] == session.generate(vecs, [{"role": "user", "content": "Who are you?"}],
                                                     max_tokens=16)["text"]
    out = session.export_cvec(r, tmp_path / "p.cvec.gguf")
    assert Path(out["path"]).exists()
    with pytest.raises(GneError):
        session.vectors(session.recipe(units=["12:dir:nope:+0.5"]))
    # 逆引きは gne direction の方向でもニューロンを順位づけできる。調書とマップの方向にも出る
    rv = session.reverse(directions="pirate", top=5, layers=[12])
    assert rv["n_candidates"] == 5 and rv["candidates"][0]["layer"] == 12
    assert "pirate" in session.neuron(12, 0)["directions"]


@pytest.mark.model
def test_v13_ablation_lora(session):
    """V13(LoRA 版): 方向 r を全層のライターから消すと、l_out-L の r 成分はどの層でも埋め込みの r 成分と同じになる
    (LoRA は埋め込みを変えられない)。消す前は層ごとに違う。"""
    from gne.verify import Verifier
    rep = Verifier(session.engine, session.weights).v13_ablation()
    assert rep.status == "pass", rep.detail
    assert rep.data["max_dev"] < 5e-3 and rep.data["base_med_dev"] > 5 * rep.data["max_dev"]
    assert rep.data["n_tensors"] == 2 * session.meta.n_layer


@pytest.mark.model
def test_direction_cli_and_job(session, tmp_path):
    from typer.testing import CliRunner

    from gne.cli import app
    from gne.jobs import standard_kinds
    c = _contrast(tmp_path)
    res = CliRunner().invoke(app, ["direction", "-m", str(MODEL), "--root", str(session.store.root), "--contrast", str(c),
                                   "--name", "pirate_cli", "--json"])
    assert res.exit_code == 0, res.output
    d = json.loads(res.stdout.strip().splitlines()[-1])
    assert d["command"] == "direction" and d["name"] == "pirate_cli"
    out = standard_kinds()["direction"](session, contrast=str(c), name="pirate_job", select="12")
    assert out["select"]["selected"] == 12


# --- B: gne ablate(書き出し・bake・V13 の bake 版・V14) --------------------------------------------------

def _tiny_full(path, *, moe=False, iq=False):
    """埋め込み・ライター(MoE なら ffn_down_exps)を持つ小さな GGUF(llama arch、出力層は別)。"""
    from gguf import GGMLQuantizationType, GGUFWriter
    from gguf.quants import quant_shape_to_byte_shape
    rng = np.random.default_rng(3)
    w = GGUFWriter(str(path), arch="llama")
    w.add_name("tiny")
    w.add_block_count(2)
    w.add_embedding_length(32)
    w.add_feed_forward_length(64)
    w.add_token_list([f"t{i}" for i in range(10)])
    if moe:
        w.add_expert_count(2)
        w.add_expert_used_count(1)
    mats = {"token_embd.weight": rng.standard_normal((10, 32)).astype(np.float32),
            "output.weight": rng.standard_normal((10, 32)).astype(np.float32)}
    for L in range(2):
        mats[f"blk.{L}.attn_output.weight"] = rng.standard_normal((32, 32)).astype(np.float32)
        if moe:
            mats[f"blk.{L}.ffn_down_exps.weight"] = rng.standard_normal((2, 32, 64)).astype(np.float32)
        else:
            mats[f"blk.{L}.ffn_down.weight"] = rng.standard_normal((32, 64)).astype(np.float32)
    for n, m in mats.items():
        w.add_tensor(n, m)
    if iq:                                             # 中身はでたらめでよい(型だけを見る検査用)
        bs = quant_shape_to_byte_shape((32, 256), GGMLQuantizationType.IQ2_XXS)
        w.add_tensor("blk.0.attn_q.weight", np.zeros(bs, np.uint8), raw_dtype=GGMLQuantizationType.IQ2_XXS)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return mats


def test_bake_ablation_math(tmp_path):
    """bake: ライター(軸 0)・埋め込み(軸 1)・MoE の ffn_down_exps(軸 1)から r が消え、他のテンソルは変わらない。部分消去。"""
    from gne.ablate import (
        bake_ablation_gguf,
        embed_in_lora,
        export_ablation_lora,
        per_layer_directions,
        unit,
    )
    from gne.weights import Weights, tensor_to_f32
    r = unit(np.random.default_rng(4).standard_normal(32))
    for moe in (False, True):
        mats = _tiny_full(tmp_path / f"m{moe}.gguf", moe=moe)
        w = Weights(tmp_path / f"m{moe}.gguf")
        assert embed_in_lora(w)                                    # 出力層が別
        rep = bake_ablation_gguf(w, r, tmp_path / f"b{moe}.gguf", requant="f16")
        b = Weights(tmp_path / f"b{moe}.gguf")
        got = {t.name: tensor_to_f32(t) for t in b.reader.tensors}
        assert np.abs(got["token_embd.weight"] @ r).max() < 1e-2              # 埋め込み(F16 に丸め)
        for L in range(2):
            assert np.abs(r @ got[f"blk.{L}.attn_output.weight"]).max() < 1e-2
            dn = got[f"blk.{L}.ffn_down_exps.weight"] if moe else got[f"blk.{L}.ffn_down.weight"]
            assert np.abs(np.einsum("i,...ij->...j", r, dn)).max() < 1e-2
        assert np.array_equal(got["output.weight"], mats["output.weight"])     # 出力層は触らない
        assert rep["n_modified"] == 5 and {m["to"] for m in rep["modified"]} == {"F16"}
        if moe:
            with pytest.raises(GneError):
                export_ablation_lora(w, r, tmp_path / "x.lora.gguf")            # MoE の LoRA は未対応
    # 部分消去(scale 0.5)と q8_0、埋め込みを省く
    w = Weights(tmp_path / "mFalse.gguf")
    bake_ablation_gguf(w, r, tmp_path / "h.gguf", scale=0.5, requant="f16", embed=False)
    h = {t.name: tensor_to_f32(t) for t in Weights(tmp_path / "h.gguf").reader.tensors}
    W = _tiny_full(tmp_path / "again.gguf")["blk.0.attn_output.weight"]
    assert np.allclose(r @ h["blk.0.attn_output.weight"], 0.5 * (r @ W), atol=2e-2)
    assert np.allclose(h["token_embd.weight"], _tiny_full(tmp_path / "again.gguf")["token_embd.weight"], atol=1e-2)
    rq = bake_ablation_gguf(w, r, tmp_path / "q.gguf", requant="q8_0")
    assert {m["to"] for m in rq["modified"]} == {"Q8_0"}
    with pytest.raises(GneError):
        bake_ablation_gguf(w, r, tmp_path / "x.gguf", requant="q4_0")
    # LoRA に埋め込みを入れる(出力層が別のモデル)
    lr = export_ablation_lora(w, r, tmp_path / "e.lora.gguf")
    from gguf import GGUFReader
    t = {x.name: x for x in GGUFReader(tmp_path / "e.lora.gguf").tensors}
    E = _tiny_full(tmp_path / "again.gguf")["token_embd.weight"]
    assert lr["embed"] and np.allclose(np.asarray(t["token_embd.weight.lora_a"].data).reshape(-1), E @ r, atol=1e-5)
    # 層ごとの方向: 方向の無い層は前の層のものを使う
    per = per_layer_directions({0: r, 2: -r}, [0, 1, 2, 3])
    assert np.allclose(per[1], r) and np.allclose(per[3], -r)


def test_via_f16_prechecks(tmp_path):
    from gne.ablate import bake_via_f16
    from gne.weights import Weights
    _tiny_full(tmp_path / "iq.gguf", iq=True)
    w = Weights(tmp_path / "iq.gguf")
    r = np.ones(32, np.float32)
    with pytest.raises(GneError) as ei:
        bake_via_f16(w, r, tmp_path / "o.gguf", llama_quantize=str(tmp_path / "missing"))
    assert ei.value.code == "TOOL_NOT_FOUND"
    with pytest.raises(GneError) as ei:
        bake_via_f16(w, r, tmp_path / "o.gguf", llama_quantize="/bin/true")
    assert ei.value.code == "NEEDS_IMATRIX" and "IQ2_XXS" in ei.value.message


@pytest.mark.model
def test_ablate_cli_and_v14(session, tmp_path):
    """gne ablate の LoRA / bake(q8_0・f16・--via-f16)と検証、V13 の bake 版、V14。"""
    from typer.testing import CliRunner

    from gne.ablate import find_llama_quantize
    from gne.cli import app
    from gne.schemas import COMMAND_SCHEMAS
    from gne.verify import Verifier
    c = _contrast(tmp_path)
    session.direction(contrast=c, name="pirate_abl", select="12")
    d = session.ablate("pirate_abl", tmp_path / "p.lora.gguf", verify=True)
    assert d["kind"] == "lora" and d["from_layer"] == 12 and not d["embed"] and d["verify"]["v13"]["pass"]
    COMMAND_SCHEMAS["ablate"].model_validate({**session.envelope("ablate"), **d})
    part = session.ablate("pirate_abl", tmp_path / "h.lora.gguf", scale=0.5, layers=[10, 11, 12], verify=True)
    assert part["layers"] == [10, 11, 12] and part["verify"]["v13"]["pass"] is None             # 部分消去は合否なし
    with pytest.raises(GneError):
        session.ablate("pirate_abl", tmp_path / "x.lora.gguf", scale=0)
    b = session.ablate("pirate_abl", tmp_path / "p-f16.gguf", bake=True, requant="f16", verify=True)
    assert b["kind"] == "bake" and b["embed"] and b["verify"]["v13"]["pass"] and b["verify"]["v14"]["pass"]
    if find_llama_quantize():
        res = CliRunner().invoke(app, ["ablate", "-m", str(MODEL), "--root", str(session.store.root),
                                       "--direction", "pirate_abl", "--bake", "--via-f16", "-o", str(tmp_path / "v.gguf"),
                                       "--json"])
        assert res.exit_code == 0, res.output
        v = json.loads(res.stdout.strip().splitlines()[-1])
        from gne.weights import Weights
        types = {t.tensor_type.name for t in Weights(tmp_path / "v.gguf").reader.tensors if t.name.endswith("ffn_down.weight")}
        assert v["requant"] == "via-f16" and types == {"Q8_0"}                                  # 元の型に戻る
    rep = Verifier(session.engine, session.weights).v13_ablation(bake=True)
    assert rep.status == "pass" and rep.data["variants"][1]["kind"] == "bake" and rep.data["variants"][1]["max_dev"] < 1e-3
    v14 = Verifier(session.engine, session.weights).v14_lora_vs_bake()
    assert v14.status == "pass", v14.detail


# --- C: レシピ・実行時・評価への組み込み ------------------------------------------------------------------

def test_ablate_spec_and_chat_mirror(tmp_path):
    from gne.chat import _with_dose
    from gne.steer import AblateSpec, parse_ablate
    a = parse_ablate("evil@10:0.5")
    assert (a.direction, a.from_layer, a.scale, a.layers) == ("evil", 10, 0.5, "all") and a.label() == "ablate evil@10 ×0.5"
    assert parse_ablate("evil") == AblateSpec("evil") and parse_ablate("x-y.z:1") .direction == "x-y.z"
    with pytest.raises(GneError):
        parse_ablate("evil@:x")
    r = Recipe([parse_unit("12:dir:pirate:+0.5"), parse_unit("3:7:x0")], "m", name="r", ablate=AblateSpec("evil", 0.8, "10-20"))
    r.save(tmp_path / "r.yaml")
    back = Recipe.load(tmp_path / "r.yaml")
    assert back.ablate == AblateSpec("evil", 0.8, "10-20") and back.to_dict()["ablate"]["layers"] == "10-20"
    assert back.scaled(0.5).ablate.scale == pytest.approx(0.4) and back.scaled(0).ablate.scale == 0
    only = Recipe([], "m", ablate=AblateSpec("evil"))
    only.validate()                                          # ablate だけのレシピ
    with pytest.raises(GneError):
        Recipe([], "m").validate()
    with pytest.raises(GneError):
        Recipe([], "m", ablate=AblateSpec("evil", 2.0)).validate()
    neg = _with_dose(r, None, -1)                            # 3 列チャットの − 列
    assert neg.units[0].direction == "pirate" and neg.units[0].dose == -0.5 and neg.units[1].k == 2
    assert neg.ablate.scale == -0.8 and _with_dose(r, None, +1).ablate.scale == 0.8


@pytest.mark.model
def test_ablate_in_recipes_runtime_and_eval(session, tmp_path):
    from gne.schemas import COMMAND_SCHEMAS
    c = _contrast(tmp_path)
    session.direction(contrast=c, name="pirate", select="12")
    sys_a = "You are a pirate. Always talk like a pirate, with arr and matey."
    r = session.recipe(ablate="pirate")                                        # ablate だけ
    lora = session.lora_for(r)
    assert len(lora) == 1 and lora[0][1] == 1.0 and Path(lora[0][0]).exists()
    assert session.lora_for(r.scaled(0.5)) == [(lora[0][0], 0.5)]                # 同じファイルを倍率だけ変えて使う
    assert session.lora_for(r.scaled(0)) is None
    st = session.steer(r, system=sys_a, prompt="Who are you?", max_tokens=24)
    assert st["differs"] and st["recipe"]["ablate"]["direction"] == "pirate" and st["units"] == []
    mix = session.recipe(units=["14:100:x4", "14:200:+0.5"], ablate="pirate@12:0.5")
    assert len(session.lora_for(mix)) == 2                                         # gain と ablate の 2 つの LoRA
    session.warnings = []
    session.export_cvec(mix, tmp_path / "m.cvec.gguf")
    assert any("ablate" in w for w in session.warnings)                            # cvec には入らない
    with pytest.raises(GneError):
        session.recipe(ablate="nope")
    saved = session.recipe_save("withabl", ["14:100:+0.5"], ablate="pirate:0.7")
    assert saved["recipe"]["ablate"]["scale"] == 0.7
    assert session.recipe(path=saved["path"]).ablate.scale == 0.7
    # chat の 3 列: − の列は ablate の倍率を反転(方向を強める)
    from gne.chat import make_columns
    cols = make_columns(session.recipe(ablate="pirate"))
    assert [x.recipe.ablate.scale if x.recipe else None for x in cols] == [-1.0, None, 1.0]
    # eval: ablate のレシピも測れる。比較(ニューロン ×0 と方向の消去)
    suite = tmp_path / "mini.yaml"
    (tmp_path / "ppl").mkdir()
    (tmp_path / "ppl" / "en.txt").write_text("Alice was beginning to get very tired of sitting by her sister on the bank. " * 3)
    suite.write_text(yaml.safe_dump({"name": "mini", "max_tokens": {"exact": 8, "format": 16, "mix": 16, "persona": 16},
                                     "exact": [{"id": "a", "prompt": "What is 17 + 25? Answer with just the number.", "answer": ["42"]}],
                                     "format": [], "mix": [], "persona": {"prompt_sets": ["en"]},
                                     "ppl": [{"id": "en", "path": "ppl/en.txt"}]}))
    ev = session.eval(r, doses=[1], suite=suite)
    assert ev["recipe"]["ablate"]["direction"] == "pirate" and ev["rows"][1]["persona"]["novelty"] >= 0
    assert any("ablate pirate" in u for u in session.evals()[0]["units"])
    cmp = session.eval_compare_ablation("pirate", top=5, doses=[1], suite=suite, max_tokens=24)
    assert [(x["method"], x["dose"]) for x in cmp["rows"]] == [("neurons", 0), ("neurons", 1), ("direction", 0), ("direction", 1)]
    assert len(cmp["neurons"]) == 5 and all(x["cos"] > 0 for x in cmp["neurons"]) and cmp["base_rate"] > 0
    assert cmp["rows"][3]["rate"] is not None and Path(cmp["path"]).exists()
    COMMAND_SCHEMAS["eval"].model_validate({**session.envelope("eval"), **ev})
    from gne.schemas import EvalCompareOut
    EvalCompareOut.model_validate({**session.envelope("eval"), **cmp})


@pytest.mark.model
def test_ablate_in_openai_and_ui_stream(session, tmp_path):
    from fastapi.testclient import TestClient

    from gne.ui.server import create_app
    session.direction(contrast=_contrast(tmp_path), name="pirate", select="12")
    app = create_app(session, warm=False, openai=True)
    with TestClient(app, base_url="http://127.0.0.1") as cl:
        body = {"messages": [{"role": "system", "content": "You are a pirate. Always talk like a pirate."},
                             {"role": "user", "content": "Who are you?"}], "max_tokens": 16}
        base = cl.post("/v1/chat/completions", json=body).json()
        abl = cl.post("/v1/chat/completions", json={**body, "gne": {"ablate": "pirate"}}).json()
        assert abl["gne"]["recipe"]["ablate"]["direction"] == "pirate"
        assert abl["choices"][0]["message"]["content"] != base["choices"][0]["message"]["content"]
        res = cl.post("/api/steer", json={"max_tokens": 8, "columns": [
            {"label": "a", "units": [], "ablate": {"direction": "pirate", "scale": 1.0, "layers": "all", "from_layer": None},
             "messages": [{"role": "user", "content": "hi"}]}]})
        assert res.status_code == 200 and '"type": "end"' in res.text
        sv = cl.post("/api/recipes/uiabl", json={"units": ["14:100:+0.5"], "ablate": {"direction": "pirate", "scale": 0.5}})
        assert sv.status_code == 200, sv.text
        got = next(x for x in cl.get("/api/recipes").json() if x["name"] == "uiabl")
        assert got["ablate"]["scale"] == 0.5 and got["units"][0]["kind"] == "neuron"


@pytest.mark.model
def test_direction_from_responses(session, tmp_path):
    """positions="response": A / B の system で返事を生成し、返事のトークンで測った平均の差。手で計算したものと一致する。"""
    from gne.capture import Capture
    from gne.direction import DirectionSet, load_contrast
    from gne.evalx import _generate
    c = _contrast(tmp_path)
    d = session.direction(contrast=c, name="pirate_resp", positions="response", response_tokens=16)
    ds = DirectionSet.load(d["path"])
    assert d["positions"] == "response" and ds.meta["positions"] == "response" and 0 < d["n_rows"]["A"] <= 3 * 16
    # 手で: 層 12 の A 側の返事のトークンの平均
    e = session.engine
    cs, _ = load_contrast(c)
    rows = []
    for side, system in (("A", cs.system_a), ("B", cs.system_b)):
        convs = [[{"role": "system", "content": system}, {"role": "user", "content": u}] for u in cs.user_prompts]
        outs = _generate(session, [(cv, 16) for cv in convs], None, None, "all", lambda n: None)
        acc = []
        cap = Capture()
        cap.set_wanted(lambda n: n == "l_out-12")
        for cv, o in zip(convs, outs, strict=True):
            pids = e.tokenize_prompt(e.render_chat(cv))
            resp = [t for t in o["token_ids"] if not e.is_eog(t)]
            ctx = e.probe_ctx(cap)
            e.clear_memory(ctx)
            e.prefill(ctx, [pids + resp])
            acc.append(cap.take()["l_out-12"][len(pids):].astype(np.float64))
        rows.append(np.concatenate(acc).mean(axis=0))
    assert np.allclose(ds.raw[12], rows[0] - rows[1], rtol=1e-4, atol=1e-4)
    last = DirectionSet.load(session.direction(contrast=c, name="pirate_last")["path"])
    assert abs(float(ds.unit(12) @ last.unit(12))) < 0.99                      # 指示の直後の方向とは別物
    st = session.steer(session.recipe(units=["12:dir:pirate_resp:+0.5"]), prompt="Who are you?", max_tokens=12)
    assert st["differs"]
    with pytest.raises(GneError):
        session.direction(contrast=c, positions="middle")


@pytest.mark.model
def test_sampling_in_engine_and_ui_stream(session, tmp_path):
    """UI のチャット用のサンプリング: 同じ seed なら同じ列、温度 0 なら greedy と同じ、presence penalty は繰り返しを減らす。"""
    from fastapi.testclient import TestClient

    from gne.engine import Sampling
    from gne.ui.server import create_app
    e = session.engine
    ctx = e.steer_ctx()
    ids = e.tokenize_prompt(e.render_chat([{"role": "user", "content": "Write a short poem about the sea."}]))
    greedy = e.generate_greedy(ctx, [ids], max_tokens=24)[0].token_ids
    assert e.generate_greedy(ctx, [ids], max_tokens=24, sampling=Sampling(temperature=0, presence_penalty=0))[0].token_ids == greedy
    a = e.generate_greedy(ctx, [ids], max_tokens=24, sampling=Sampling(seed=1))[0].token_ids
    b = e.generate_greedy(ctx, [ids], max_tokens=24, sampling=Sampling(seed=1))[0].token_ids
    c = e.generate_greedy(ctx, [ids], max_tokens=24, sampling=Sampling(seed=2))[0].token_ids
    assert a == b and a != c
    rng = np.random.default_rng(0)
    lo = np.zeros(10)
    lo[3] = 5.0
    assert Sampling(presence_penalty=0).pick(lo, {3}, rng) == 3
    assert Sampling(presence_penalty=100, top_k=10, top_p=1.0, min_p=0).pick(lo, {3}, rng) != 3
    session.direction(contrast=_contrast(tmp_path), name="pirate_s", select="12")
    app = create_app(session, warm=False)
    with TestClient(app, base_url="http://127.0.0.1") as cl:
        col = {"label": "x", "units": ["12:dir:pirate_s:+0.5"], "messages": [{"role": "user", "content": "Hello!"}]}
        body = {"max_tokens": 16, "sampling": {"seed": 7}, "columns": [col, col]}
        res = cl.post("/api/steer", json=body)
        ends = [json.loads(line[6:]) for line in res.text.splitlines() if line.startswith("data: ") and '"type": "end"' in line]
        assert len(ends) == 2 and ends[0]["text"] == ends[1]["text"]                  # 同じ seed なら列どうしも同じ
        bad = cl.post("/api/steer", json={**body, "sampling": {"seed": 1, "temp": 2}})
        assert bad.status_code == 400


@pytest.mark.model
def test_openai_api_sampling(session):
    """OpenAI 互換 API のサンプリング: 何も送らなければ greedy、seed が同じなら同じ返事、値の誤りは 400。"""
    from fastapi.testclient import TestClient

    from gne.ui.openai_api import sampling_from
    from gne.ui.server import create_app
    assert sampling_from({}) is None and sampling_from({"temperature": 0}) is None
    sp = sampling_from({"temperature": 0.7, "seed": 3})
    assert (sp.temperature, sp.top_k, sp.top_p, sp.min_p, sp.presence_penalty, sp.seed) == (0.7, 40, 0.95, 0.05, 0.0, 3)
    app = create_app(session, warm=False, openai=True)
    msgs = [{"role": "user", "content": "Write one sentence about the moon."}]
    with TestClient(app, base_url="http://127.0.0.1") as cl:
        greedy = cl.post("/v1/chat/completions", json={"messages": msgs, "max_tokens": 20}).json()
        assert greedy["gne"]["sampling"] is None
        assert greedy["choices"][0]["message"]["content"] == session.generate(None, msgs, max_tokens=20)["text"]
        body = {"messages": msgs, "max_tokens": 20, "temperature": 1.0, "seed": 11}
        a, b = (cl.post("/v1/chat/completions", json=body).json() for _ in range(2))
        c = cl.post("/v1/chat/completions", json={**body, "seed": 12}).json()
        assert a["choices"][0]["message"]["content"] == b["choices"][0]["message"]["content"]
        assert a["gne"]["sampling"]["seed"] == 11 and c["choices"][0]["message"]["content"] != a["choices"][0]["message"]["content"]
        assert cl.post("/v1/chat/completions", json={**body, "top_p": 2}).status_code == 400


def test_recipe_sampling_block(tmp_path):
    """レシピの sampling: 足りない値は llama-server の既定で埋めて保存し、読み直せる。scaled でも残る。範囲外は誤り。"""
    from gne.steer import (
        GneError,
        Recipe,
        llama_server_args,
        parse_unit,
        sampling_values,
    )
    r = Recipe([parse_unit("3:5:+0.5")], name="sp", sampling={"temperature": 1.0, "presence_penalty": 0.9})
    r.validate()
    assert r.sampling == {"temperature": 1.0, "top_k": 40, "top_p": 0.95, "min_p": 0.05, "presence_penalty": 0.9}
    p = tmp_path / "sp.yaml"
    r.save(p)
    q = Recipe.load(p)
    assert q.sampling == r.sampling and q.scaled(0.5).sampling == r.sampling and q.to_dict()["sampling"] == r.sampling
    assert llama_server_args(q.sampling) == "--temp 1 --top-k 40 --top-p 0.95 --min-p 0.05 --presence-penalty 0.9"
    for bad in ({"temperature": 9}, {"top_p": 0}, {"seed": 3}, {"temp": 1}):
        with pytest.raises(GneError):
            Recipe([parse_unit("3:5:+0.5")], sampling=bad).validate()
    with pytest.raises(GneError):
        Recipe([parse_unit("3:5:+0.5")], sampling="hot").validate()
    assert sampling_values({"seed": 7}, base={"temperature": 1.0, "top_k": 20, "top_p": 0.9, "min_p": 0.0,
                                              "presence_penalty": 0.0})["seed"] == 7


@pytest.mark.model
def test_recipe_sampling_in_api_and_cli(session, tmp_path):
    """レシピの sampling は OpenAI 互換 API の既定になり(送った値が優先、温度 0 だけなら greedy)、
    CLI の steer でも使われる(--greedy で切る、--sample-seed で同じ返事)。"""
    from fastapi.testclient import TestClient
    from typer.testing import CliRunner

    from gne.cli import app as cli_app
    from gne.steer import Recipe, parse_unit
    from gne.ui.openai_api import sampling_from
    from gne.ui.server import create_app
    r = Recipe([parse_unit("14:100:+0.5")], name="spr", sampling={"temperature": 1.2, "top_k": 10})
    r.validate()
    sp = sampling_from({}, r)
    assert (sp.temperature, sp.top_k, sp.top_p) == (1.2, 10, 0.95)
    sp = sampling_from({"temperature": 0.5, "seed": 4}, r)
    assert (sp.temperature, sp.top_k, sp.seed) == (0.5, 10, 4)
    assert sampling_from({"temperature": 0}, r) is None
    session.recipe_save("spr", ["14:100:+0.5"], sampling={"temperature": 1.2, "top_k": 10})
    msgs = [{"role": "user", "content": "Say something about cats."}]
    app = create_app(session, warm=False, openai=True)
    with TestClient(app, base_url="http://127.0.0.1") as cl:
        res = cl.post("/v1/chat/completions", json={"messages": msgs, "max_tokens": 8, "seed": 5, "gne": {"recipe": "spr"}}).json()
        assert res["gne"]["sampling"]["temperature"] == 1.2 and res["gne"]["sampling"]["seed"] == 5
    path = session.store.recipes_dir() / "spr.yaml"
    base = ["steer", "-m", str(MODEL), "--root", str(session.store.root), "--recipe", str(path), "-p", "Hi!",
            "--max-tokens", "8", "--json"]
    runs = [CliRunner().invoke(cli_app, base + ["--sample-seed", "9"]) for _ in range(2)]
    assert all(x.exit_code == 0 for x in runs), runs[0].output
    outs = [json.loads(x.stdout.strip().splitlines()[-1]) for x in runs]
    assert outs[0]["sampling"]["temperature"] == 1.2 and outs[0]["sampling"]["seed"] == 9
    assert outs[0]["steered"]["text"] == outs[1]["steered"]["text"]
    g = json.loads(CliRunner().invoke(cli_app, base + ["--greedy"]).stdout.strip().splitlines()[-1])
    assert g["sampling"] is None
    bad = CliRunner().invoke(cli_app, base + ["--greedy", "--temperature", "1"])
    assert bad.exit_code != 0
