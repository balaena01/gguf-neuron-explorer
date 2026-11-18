"""v0.8.0: 部品ごとの書き込みの順位づけ(gne heads writes)、ニューロンの方向成分の射影(L:Nn:sS@NAME)、
s=1 から組を作る配分(gne heads pipeline --rank writes)。

モデルが要らないもの(書き方・LoRA の式・対照・回の合成・確かめる部品の選び方)は、いつも流す。
モデルを使うものは Qwen3-0.6B(GNE_TEST_MODEL)。線形注意の層は Qwen3.5-4B(GNE_TEST_LINEAR_MODEL)で 1 件。
GPU が他の処理で埋まっているときは GNE_TEST_NGL=0 で CPU に載せる。
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from gne.steer import GneError, Recipe, parse_unit

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
LINEAR = Path(os.environ.get("GNE_TEST_LINEAR_MODEL", ROOT / "models" / "Qwen3.5-4B-Q8_0.gguf"))
NGL = int(os.environ.get("GNE_TEST_NGL", "-1"))
EVIL = ROOT / "configs" / "contrast" / "evil.yaml"


def _tiny(path, *, n_head=4, head_dim=3, n_embd=16, n_layer=3, n_ff=24):
    """attention の層(attn_q・attn_output)と ffn_down・ffn_up を持つ小さな GGUF(llama arch)。"""
    from gguf import GGUFWriter
    rng = np.random.default_rng(1)
    w = GGUFWriter(str(path), arch="llama")
    w.add_name("tiny")
    w.add_block_count(n_layer)
    w.add_embedding_length(n_embd)
    w.add_feed_forward_length(n_ff)
    w.add_head_count(n_head)
    w.add_head_count_kv(2)
    w.add_token_list([f"t{i}" for i in range(8)])
    mats = {}
    for L in range(n_layer):
        for nm, shape in (("attn_q", (n_head * head_dim, n_embd)), ("attn_output", (n_embd, n_head * head_dim)),
                          ("ffn_down", (n_embd, n_ff)), ("ffn_up", (n_ff, n_embd))):
            m = rng.standard_normal(shape).astype(np.float32)
            mats[f"blk.{L}.{nm}.weight"] = m
            w.add_tensor(f"blk.{L}.{nm}.weight", m)
    w.add_tensor("token_embd.weight", rng.standard_normal((8, n_embd)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return mats


def _store_session(tmp_path):
    from gne.api import Session
    mats = _tiny(tmp_path / "m.gguf")
    s = Session(tmp_path / "m.gguf", root=tmp_path / "root")
    rng = np.random.default_rng(2)
    p = s.store.direction_dir(s.meta.model_id) / "d1.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, layers=np.array([0, 1, 2], dtype=np.int32), **{f"layer_{L}": rng.standard_normal(16).astype(np.float32)
                                                              for L in range(3)})
    return s, mats


def _unit(v):
    return v / np.linalg.norm(v)


# --- モデル不要 ----------------------------------------------------------------------------------------

def test_neuron_proj_notation_and_recipe(tmp_path):
    u = parse_unit("20:N4096:s1@evil")
    assert (u.layer, u.kind, u.neuron, u.head, u.mode, u.k, u.direction) == (20, "neuron", 4096, None, "proj", 1.0, "evil")
    assert u.label() == "L20 N4096 s1 @evil"
    assert parse_unit("3:n7:s-2").k == -2.0 and parse_unit("3:N7:s2").direction is None
    r = Recipe([u, parse_unit("5:H2:s3@evil")], name="m")
    r.validate()
    r.save(tmp_path / "r.yaml")
    d = yaml.safe_load((tmp_path / "r.yaml").read_text())
    assert d["units"][0] == {"layer": 20, "neuron": 4096, "mode": "proj", "s": 1.0, "direction": "evil"}
    back = Recipe.load(tmp_path / "r.yaml")
    assert back.units[0].kind == "neuron" and back.units[0].neuron == 4096 and back.units[0].mode == "proj"
    assert back.units[1].kind == "head" and back.units[1].head == 2
    from gne.headproj import ProjHead, heads_to_units, parse_proj_specs, proj_heads
    ps = proj_heads(back)
    assert ps[0] == ProjHead(20, 4096, 1.0, "evil", "neuron") and ps[0].spec() == "20:N4096:s1@evil"
    assert ps[1].kind == "head" and ps[1].spec() == "5:H2:s3@evil"
    assert ps[0].key != ProjHead(20, 4096, 1.0, "evil").key                      # 番号が同じでもヘッドとは別
    assert parse_proj_specs(["2:N5:s1"], "d")[0].kind == "neuron"
    us = heads_to_units(ps, "evil")
    assert us[0].kind == "neuron" and us[0].neuron == 4096 and us[0].direction is None and us[1].head == 2
    with pytest.raises(GneError):
        Recipe([parse_unit("3:N7:s2")]).validate()                            # 方向が無い


def test_neuron_proj_matrices_lora_and_bake(tmp_path):
    """ffn_down の列から方向成分を引く: 同じ層・同じ方向は階数 1。ヘッドと混ぜると、テンソルごとに 1 本ずつ。LoRA と焼き込みが式どおり。"""
    from gguf import GGUFReader
    from gne.headproj import ProjBuilder, ProjHead
    s, mats = _store_session(tmp_path)
    b = ProjBuilder(s.weights, s.direction_set)
    tens = b.tensors([ProjHead(1, 3, 1.0, "d1", "neuron"), ProjHead(1, 10, 2.5, "d1", "neuron"), ProjHead(1, 0, 2.0, "d1")])
    assert sorted(tens) == ["blk.1.attn_output.weight", "blk.1.ffn_down.weight"]
    A, B = tens["blk.1.ffn_down.weight"]
    assert A.shape == (1, 24) and B.shape == (16, 1)
    W = mats["blk.1.ffn_down.weight"].astype(np.float64)
    r = _unit(np.load(s.direction_path("d1"))["layer_1"].astype(np.float64))
    exp = W.copy()
    for n, sv in ((3, 1.0), (10, 2.5)):
        exp[:, n] -= sv * r * (r @ W[:, n])
    np.testing.assert_allclose(W + B @ A, exp, atol=1e-5)
    np.testing.assert_allclose(((W + B @ A)[:, 3]) @ r, 0.0, atol=1e-5)          # s=1: その方向に何も書かない
    d = (W + B @ A) - W
    np.testing.assert_allclose(d - np.outer(r, r @ d), 0, atol=1e-5)               # 直交成分は変わらない
    with pytest.raises(GneError):
        b.matrices([ProjHead(1, 3, 1.0, "d1", "neuron")])                         # ヘッドだけの道具(nw heads)は断る
    with pytest.raises(GneError):
        b.tensors([ProjHead(1, 99, 1.0, "d1", "neuron")])                         # 範囲外
    rec = s.recipe(units=["1:N3:s1@d1", "1:N10:s2.5@d1", "2:H1:s1@d1"], name="nr")
    out = s.gain(rec, tmp_path / "n.lora.gguf")
    assert out["proj_neuron_layers"]["1"]["rank"] == 1 and out["proj_neuron_layers"]["1"]["neurons"] == [[3, 1.0, "d1"], [10, 2.5, "d1"]]
    assert set(out["proj_layers"]) == {"2"}
    rd = GGUFReader(tmp_path / "n.lora.gguf")
    t = {x.name: x for x in rd.tensors}
    Al = np.array(t["blk.1.ffn_down.weight.lora_a"].data).reshape(1, 24)
    Bl = np.array(t["blk.1.ffn_down.weight.lora_b"].data).reshape(16, 1)
    np.testing.assert_allclose(Bl @ Al, exp - W, atol=1e-5)
    meta = json.loads(rd.fields["gne.headproj"].contents())
    assert {(x.get("kind", "head"), x["layer"], x.get("neuron", x.get("head"))) for x in meta["heads"]} == \
        {("neuron", 1, 3), ("neuron", 1, 10), ("head", 2, 1)}
    hb = s.gain(rec, tmp_path / "baked.gguf", bake=True, requant="f16")
    assert sorted(m["tensor"] for m in hb["modified"]) == ["blk.1.ffn_down.weight", "blk.2.attn_output.weight"]
    got = np.array({x.name: x for x in GGUFReader(tmp_path / "baked.gguf").tensors}["blk.1.ffn_down.weight"].data,
                   dtype=np.float32).reshape(16, 24)
    np.testing.assert_allclose(got, exp, atol=2e-2)


def test_head_only_records_unchanged(tmp_path):
    """ヘッドだけのレシピは、記録(gne.headproj)・キャッシュの鍵・層のまとめに種類の項目が増えない。"""
    from gne.headproj import ProjBuilder, ProjHead, layer_summary, proj_cache_key, proj_meta
    s, _ = _store_session(tmp_path)
    b = ProjBuilder(s.weights, s.direction_set)
    hs = [ProjHead(1, 0, 2.0, "d1")]
    m = proj_meta(hs, b)
    assert set(m["heads"][0]) == {"layer", "head", "s", "direction", "direction_from", "vector_key"}
    assert proj_cache_key(hs, b) == proj_cache_key([ProjHead(1, 0, 2.0, "d1", "head")], b)
    assert set(layer_summary(s.weights, hs, b.tensors(hs))) == {"proj_layers"}


def test_neuron_random_controls(tmp_path):
    """対照: ヘッドは同数のランダムなヘッド、ニューロンは同数のランダムなニューロン(s の値はそのまま)。同じ種なら同じ。"""
    from gne.headproj import neuron_control_pool, proj_control_recipes, proj_heads
    s, _ = _store_session(tmp_path)
    r = s.recipe(units=["1:H0:s2@d1", "1:N3:s1@d1", "2:N5:s1.5@d1"], name="p")
    pool = neuron_control_pool(s.weights, r, pool="before", score_layers=[1])
    assert {L for L, _ in pool} == {0, 1} and (1, 3) not in pool
    cs = proj_control_recipes(s, r, 3, pool="before", score_layers=[2])
    for c in cs:
        ps = proj_heads(c)
        assert sum(p.kind == "neuron" for p in ps) == 2 and sum(p.kind == "head" for p in ps) == 1
        assert sorted(p.s for p in ps if p.kind == "neuron") == [1.0, 1.5]
        assert not {(p.layer, p.head) for p in ps if p.kind == "neuron"} & {(1, 3), (2, 5)}
    assert [proj_heads(c) for c in cs] == [proj_heads(c) for c in proj_control_recipes(s, r, 3, pool="before", score_layers=[2])]
    # ヘッドだけのレシピの対照は、ニューロンの対照を足す前と同じ乱数の列
    rh = s.recipe(units=["1:H0:s2@d1", "2:H1:s3@d1"], name="h")
    from gne.controls import SEED_BASE
    from gne.headproj import control_pool
    cand = control_pool(s.weights, rh, pool="before", score_layers=[2])
    rng = np.random.default_rng([SEED_BASE, 6, 0])
    want = [cand[int(i)] for i in rng.choice(len(cand), size=2, replace=False)]
    got = [(p.layer, p.head) for p in proj_heads(proj_control_recipes(s, rh, 1, pool="before", score_layers=[2])[0])]
    assert got == want


def test_loop_merge_keeps_neurons_apart_from_heads():
    """回ごとのレシピ → 両方の回: ヘッド H20 とニューロン N20 は番号が同じでも別のユニット。ニューロンも同じ重みの 2 回で 1 つにまとめる。"""
    from gne.loop import plan_from_kinds
    from gne.loopsearch import merge_units
    from gne.steer import Unit
    kinds = [0] * 4 + [1] + [4] * 22 + [2] + [1] + [4] * 22 + [2] + [0] * 4
    rng = np.random.default_rng(1)
    raw = {L: rng.normal(size=8) for L in range(56)}
    ds = SimpleNamespace(raw=raw, unit=lambda L: raw[L] / np.linalg.norm(raw[L]), meta={})
    sess = SimpleNamespace(weights=SimpleNamespace(loop=plan_from_kinds(kinds)), direction_set=lambda n: ds, warn=lambda m: None)
    us = [Unit(21, None, 0.0, "head", mode="proj", k=2.0, head=20), Unit(21, 20, 0.0, "neuron", mode="proj", k=1.0),
          Unit(45, 20, 0.0, "neuron", mode="proj", k=3.0)]
    r = Recipe(us, "m", name="x", default_direction="d", loop_edits="per_visit")
    mx, tw = merge_units(sess, r, "max")
    assert len(tw) == 1 and tw[0]["head"] == "21:N20"
    assert sorted((u.kind, u.layer, u.k) for u in mx.proj_units) == [("head", 21, 2.0), ("neuron", 45, 3.0)]
    assert mx.loop_visits["21:N20"] == [1, 2] and mx.loop_visits["21:H20"] == [1]


def test_parse_kinds_and_verify_rows():
    from gne.unitwrites import parse_kinds, verify_rows
    assert parse_kinds("all") == ("attn", "linear", "neuron") and parse_kinds("heads") == ("attn", "linear")
    assert parse_kinds("neuron,attn") == ("attn", "neuron") and parse_kinds(None) == ("attn", "linear", "neuron")
    with pytest.raises(GneError):
        parse_kinds("mlp")
    rows = [{"label": f"u{i}", "score": 100 - i} for i in range(100)] + [{"label": "neg", "score": -1.0}]
    v = verify_rows(rows, 10, 4)
    assert [r["label"] for r in v[:10]] == [f"u{i}" for i in range(10)] and not any(r["sampled"] for r in v[:10])
    samp = [r for r in v if r["sampled"]]
    assert len(samp) == 4 and samp[0]["label"] == "u10" and samp[-1]["label"] == "u39"     # 上位の下(10〜39 位)から等間隔
    assert all(r["label"] != "neg" for r in verify_rows(rows, 200, 0))                       # 点数が正のものだけ


def test_pipeline_writes_settings_are_checked(tmp_path):
    from gne.api import Session
    s, _ = _store_session(tmp_path)
    spec = {"name": "x", "user_prompts": ["a", "b", "c", "d"], "system_a": "A"}
    with pytest.raises(GneError):
        s.heads_pipeline({**spec, "pipeline": {"rank": "nope"}})
    with pytest.raises(GneError):
        s.heads_pipeline({**spec, "pipeline": {"rank": "writes", "unit_kinds": "mlp"}})
    with pytest.raises(GneError):
        s.heads_pipeline({**spec, "pipeline": {"bogus": 1}})
    assert set(Session.PIPELINE_WRITES) >= {"rank", "unit_kinds", "verify_top", "verify_mode", "n_units", "check_replies"}


# --- モデルを使う ------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def session(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip(f"{MODEL} がありません")
    from gne.api import Session
    s = Session(MODEL, root=tmp_path_factory.mktemp("wr"), n_gpu_layers=NGL, n_ctx=2048)
    s.direction(contrast=EVIL, positions="last")
    yield s
    s.close()


@pytest.mark.model
def test_neuron_proj_check_v24_v25(session):
    """ニューロンの射影の LoRA: MLP の出力の差が −s·r̂·(r̂·W_down[:, n])·a_n と合い、直交成分が変わらない(gain ×0 は変わる)。"""
    d = session.check_neuron_proj(12, 100, "evil", ss=(1.0, 2.0))
    assert d["ok"] and d["v24"] and d["v25"], d["rows"]
    assert all(abs(r["par_after_over_before"] - (1 - r["s"])) < 0.05 for r in d["rows"])
    assert all(r["act_change_max"] < 1e-4 for r in d["rows"])
    from gne.schemas import SCHEMA_VERSION, HeadprojCheckOut
    HeadprojCheckOut.model_validate({"schema_version": SCHEMA_VERSION, "command": "headproj check", "model_id": "x",
                                     "llama_cpp_version": "x", "seed": 0, "generated_at": "x", **d})


@pytest.mark.model
def test_measure_writes_decomposes_and_saves(session):
    """書き込みの順位づけ: 層ごとに部品の和が捕まえた層の出力と合い、点数は A − B の直接の書き込み × (r̂_L·r̂_S)。
    層ごとの更新のベクトル(種類ごと・質問を半分ずつ)と、部品ごとの値が npz に残る。上位を確かめると順位相関が出る。"""
    out = session.heads_writes("evil", verify_top=6, verify_sample=2, verify_examples=8, name="ev_w")
    assert out["select_layer"] == session.meta.n_layer - 2 and out["n"]["A"] == out["n"]["B"] > 0
    assert out["check"]["attn_rel_max"] < 0.05 and out["check"]["mlp_rel_max"] < 0.05     # 部品の和 = 層の出力
    rows = out["rows"]
    assert [r["score"] for r in rows] == sorted((r["score"] for r in rows), reverse=True)
    assert {r["kind"] for r in rows} <= {"attn", "neuron"} and any(r["kind"] == "neuron" for r in rows)
    z = np.load(out["npz"])
    NL, ne = session.meta.n_layer - 1, session.meta.n_embd
    for k in ("attn", "linear", "mlp", "layer"):
        for x in ("a", "b", "diff", "diff_h0", "diff_h1"):
            assert z[f"upd_{k}_{x}"].shape == (NL, ne)
    np.testing.assert_allclose(z["upd_layer_diff"], z["upd_attn_diff"] + z["upd_mlp_diff"], atol=1e-4)
    np.testing.assert_allclose(z["upd_attn_diff"], z["upd_attn_a"] - z["upd_attn_b"], atol=1e-4)
    assert not z["upd_linear_diff"].any()                                                # attention だけのモデル
    assert list(z["layer_mixer"])[:2] == ["attention", "attention"] and not z["layer_visit"].any()
    # 点数 = (a_sel − b_sel)、a_sel = a_own × (r̂_L·r̂_S)
    h = next(r for r in rows if r["kind"] == "attn")
    j = int(np.flatnonzero((z["head_layer"] == h["layer"]) & (z["head_index"] == h["index"]))[0])
    assert abs(float(z["head_a_sel"][j] - z["head_b_sel"][j]) - h["score"]) < 1e-3
    cs = float(z["cos_to_select"][h["layer"]])
    assert abs(float(z["head_a_own"][j]) * cs - float(z["head_a_sel"][j])) < 1e-3
    assert abs(float(z["head_q_a"][:, j].mean()) - float(z["head_a_own"][j])) < 1e-3
    nr = next(r for r in rows if r["kind"] == "neuron")
    assert abs(float(z[f"neuron_a_sel_{nr['layer']}"][nr["index"]] - z[f"neuron_b_sel_{nr['layer']}"][nr["index"]]) - nr["score"]) < 1e-3
    v = out["verify"]
    assert len(v["rows"]) == 8 and sum(r["sampled"] for r in v["rows"]) == 2 and "all" in v["spearman"]
    for r in v["rows"]:
        assert abs(r["gap_drop"] - (r["causal"] + r["b_side_delta"])) < 1e-9
        assert abs(r["effect"] - (r["causal"] - abs(r["b_side_delta"]))) < 1e-9
    js = json.loads(Path(out["path"]).read_text(encoding="utf-8"))
    assert js["npz"] == Path(out["npz"]).name and js["verify"]["mode"] == "proj"
    from gne.schemas import SCHEMA_VERSION, HeadsWritesOut
    HeadsWritesOut.model_validate({"schema_version": SCHEMA_VERSION, "command": "heads writes", "model_id": "x",
                                   "llama_cpp_version": "x", "seed": 0, "generated_at": "x", **out})


@pytest.mark.model
def test_pipeline_rank_writes_end_to_end(session):
    """--rank writes: 順位づけ → 確かめ → s=1 から組 → 応答での確かめ。表は heads allocate と同じ形で、レシピが残る。"""
    d = yaml.safe_load(EVIL.read_text(encoding="utf-8"))
    spec = {"name": "pw", "system_a": d["system_a"], "system_b": d["system_b"], "user_prompts": d["user_prompts"],
            "validate": d["validate"],
            "pipeline": {"rank": "writes", "verify_top": 6, "verify_sample": 0, "n_units": [4, 2], "max_examples": 8,
                         "check_replies": 2, "check_tokens": 16, "target": "B"}}
    out = session.heads_pipeline(spec)
    assert out["settings"]["rank"] == "writes" and out["writes"]["select_layer"] == session.meta.n_layer - 2
    assert Path(out["writes"]["path"]).exists() and Path(out["writes"]["npz"]).exists()
    al = out["allocate"]
    assert al["method"] == "unit1" and [r["n_heads"] for r in al["table"]] == [4, 2]
    for r in al["table"]:
        if r.get("skipped"):
            continue
        assert Path(r["recipe"]).exists() and r["n_used"] <= r["n_heads"] and r["s_max"] >= 1.0
        rec = Recipe.load(r["recipe"])
        assert len(rec.proj_units) == r["n_used"] and all(u.k == pytest.approx(r["alpha"]) for u in rec.proj_units)
        assert r["stop"] in ("target", "raised", "s_max", "drift")
        if r["stop"] == "target":
            assert r["alpha"] == 1.0
    acc = [t for t in al["trace"] if t["accepted"]]
    assert all(b["a_mean"] < a["a_mean"] for a, b in zip(acc, acc[1:]))                 # 採った部品ごとに A 側が下がる
    ck = out["check"]
    assert set(ck["rows"]) == {"none", "recipe"} and ck["n"] == {"A": 2, "B": 2}
    assert all(set(x) >= {"finished", "match_finished", "empty", "loop", "at_limit"} for row in ck["rows"].values() for x in row.values())
    from gne.schemas import SCHEMA_VERSION, HeadsPipelineOut
    HeadsPipelineOut.model_validate({"schema_version": SCHEMA_VERSION, "command": "heads pipeline", "model_id": "x",
                                     "llama_cpp_version": "x", "seed": 0, "generated_at": "x", **out})
    # 残りから追加: 新しい選び方の上でも動き、足す部品の探し方を引き継ぐ
    ap = session.store.heads_dir(session.meta.model_id) / "pw.alloc.json"
    a2 = json.loads(ap.read_text(encoding="utf-8"))
    a2["remain_threshold"] = -1e9                                              # 全部残る(仕組みを確かめるため)
    if not any(r.get("recommended") for r in a2["table"]):
        next(r for r in a2["table"] if not r.get("skipped"))["recommended"] = True
    ap.write_text(json.dumps(a2), encoding="utf-8")
    rf = session.heads_refine("pw", new_heads=[2])
    pd = json.loads(Path(rf["pipeline"]["path"]).read_text(encoding="utf-8"))
    assert pd["settings"]["rank"] == "writes" and pd.get("writes") and not pd.get("check")


@pytest.mark.model
def test_pipeline_causal_settings_unchanged(session):
    """既定(rank causal)の一括の結果の settings には、writes の設定が増えない(今までの結果と同じ形)。"""
    from gne.api import Session
    d = yaml.safe_load(EVIL.read_text(encoding="utf-8"))
    spec = {"name": "pc", "system_a": d["system_a"], "system_b": d["system_b"], "user_prompts": d["user_prompts"],
            "pipeline": {"n_heads": [2], "max_examples": 8, "top_layers": 1, "hard": 4}}
    out = session.heads_pipeline(spec)
    assert set(out["settings"]) == set(Session.PIPELINE_DEFAULTS) and "writes" not in out


@pytest.mark.model
def test_linear_heads_are_ranked(tmp_path):
    """線形注意の層(Qwen3.5)のヘッドも、注意のヘッドと同じ式で順位に入り、部品の和が層の出力と合う。"""
    if not LINEAR.exists():
        pytest.skip(f"{LINEAR} がありません")
    from gne.api import Session
    s = Session(LINEAR, root=tmp_path / "root", n_gpu_layers=NGL, n_ctx=2048)
    try:
        s.direction(contrast=EVIL, positions="last")
        out = s.heads_writes("evil", verify_top=0, max_examples=6, kinds="heads")
        kinds = {r["kind"] for r in out["rows"]}
        assert kinds == {"attn", "linear"} and out["n_units"]["neuron"] == 0
        assert out["check"]["attn_rel_max"] < 0.05
        z = np.load(out["npz"])
        lin = [i for i, m in enumerate(z["layer_mixer"]) if m == "linear_attention"]
        assert lin and np.abs(z["upd_linear_diff"][lin]).sum() > 0 and not np.abs(z["upd_attn_diff"][lin]).sum()
    finally:
        s.close()
