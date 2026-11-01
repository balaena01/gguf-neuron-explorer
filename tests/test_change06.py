"""CHANGE-06 のテスト: ヘッドの方向成分だけを変える介入(mode: proj)、速い物差し、因果スクリーニング、配分、書き出し、eval。

モデルを使うものは Qwen3-0.6B(GNE_TEST_MODEL、attention だけ)。線形注意の住所の分岐は Qwen3.5-4B(GNE_TEST_LINEAR_MODEL)で 1 件。
GPU が他の処理で埋まっているときは GNE_TEST_NGL=0 で CPU に載せる。
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from gne.steer import GneError, Recipe, Unit, parse_unit

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
LINEAR = Path(os.environ.get("GNE_TEST_LINEAR_MODEL", ROOT / "models" / "Qwen3.5-4B-Q8_0.gguf"))
NGL = int(os.environ.get("GNE_TEST_NGL", "-1"))
EVIL = ROOT / "configs" / "contrast" / "evil.yaml"


def _tiny(path, *, n_head=4, head_dim=3, n_embd=16, n_layer=3):
    """attention の層 n_layer 個(attn_q・attn_output)と ffn_down・ffn_up を持つ小さな GGUF(llama arch)。"""
    from gguf import GGUFWriter
    rng = np.random.default_rng(1)
    w = GGUFWriter(str(path), arch="llama")
    w.add_name("tiny")
    w.add_block_count(n_layer)
    w.add_embedding_length(n_embd)
    w.add_feed_forward_length(24)
    w.add_head_count(n_head)
    w.add_head_count_kv(2)
    w.add_token_list([f"t{i}" for i in range(8)])
    mats = {}
    for L in range(n_layer):
        for nm, shape in (("attn_q", (n_head * head_dim, n_embd)), ("attn_output", (n_embd, n_head * head_dim)),
                          ("ffn_down", (n_embd, 24)), ("ffn_up", (24, n_embd))):
            m = rng.standard_normal(shape).astype(np.float32)
            mats[f"blk.{L}.{nm}.weight"] = m
            w.add_tensor(f"blk.{L}.{nm}.weight", m)
    w.add_tensor("token_embd.weight", rng.standard_normal((8, n_embd)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return mats


def _dirs(path, n_layer=3, n_embd=16, seed=2, base_layer=None):
    """DirectionSet の npz(層 0..n_layer-1)。"""
    rng = np.random.default_rng(seed)
    Ls = list(range(n_layer))
    raw = {L: rng.standard_normal(n_embd).astype(np.float32) for L in Ls}
    extra = {} if base_layer is None else {"base_layer": np.int32(base_layer)}
    np.savez(path, layers=np.array(Ls, dtype=np.int32), **{f"layer_{L}": raw[L] for L in Ls}, **extra)
    return raw


def _unit(v):
    return v / np.linalg.norm(v)


# --- モデル不要 ----------------------------------------------------------------------------------------

def test_proj_unit_notation_and_recipe(tmp_path):
    u = parse_unit("33:H16:s15.4@tsundere_r")
    assert (u.layer, u.kind, u.head, u.mode, u.k, u.direction) == (33, "head", 16, "proj", 15.4, "tsundere_r")
    assert u.label() == "L33 H16 s15.4 @tsundere_r"
    v = parse_unit("43:h3:s14.2")
    assert v.direction is None and v.k == 14.2 and v.mode == "proj"
    assert parse_unit("2:H1:s-3").k == -3.0                          # 負の s は方向成分を強める
    with pytest.raises(GneError):
        Recipe([v]).validate()                                     # 方向が無い
    r = Recipe([u, v], name="p", default_direction="tsundere_r", direction_layer="base")
    r.validate()
    r.save(tmp_path / "r.yaml")
    d = yaml.safe_load((tmp_path / "r.yaml").read_text())
    assert d["default_direction"] == "tsundere_r" and d["direction_layer"] == "base"
    assert d["units"][0] == {"layer": 33, "kind": "head", "mode": "proj", "direction": "tsundere_r", "head": 16, "s": 15.4}
    assert "direction" not in d["units"][1]
    back = Recipe.load(tmp_path / "r.yaml")
    assert back.units == [u, v] and back.default_direction == "tsundere_r" and back.direction_layer == "base"
    assert back.proj_units == [u, v] and back.head_units == [u, v]
    half = back.scaled(0.5)                                        # scale(レシピ全体の倍率)は s に掛かる
    assert [x.k for x in half.units] == [7.7, 7.1] and half.default_direction == "tsundere_r"
    # spec の YAML(direction・s)も読める
    (tmp_path / "s.yaml").write_text("default_direction: t\nunits:\n  - {layer: 33, kind: head, head: 16, mode: proj, direction: tsundere_r, s: 15.4}\n"
                                     "  - {layer: 43, kind: head, head: 3,  mode: proj, s: 14.2}\n")
    sp = Recipe.load(tmp_path / "s.yaml")
    assert [x.k for x in sp.units] == [15.4, 14.2] and sp.units[1].direction is None
    with pytest.raises(GneError):
        Recipe([u], direction_layer="nope").validate()
    with pytest.raises(GneError):
        Recipe([Unit(3, None, 0.0, "head", mode="proj", k=2.0, head=None, direction="x")]).validate()


def test_unit_matrices_rank_and_math(tmp_path):
    """V27: 同じ層・同じ方向のヘッドは階数 1、方向が混ざると階数 = 方向の数で警告。W + BA が式どおり。"""
    from gne.direction import DirectionSet
    from gne.headproj import ProjHead, unit_matrices
    from gne.weights import Weights
    mats = _tiny(tmp_path / "m.gguf")
    _dirs(tmp_path / "d1.npz", seed=2)
    _dirs(tmp_path / "d2.npz", seed=3)
    w = Weights(tmp_path / "m.gguf")
    dsets = {"d1": DirectionSet.load(tmp_path / "d1.npz"), "d2": DirectionSet.load(tmp_path / "d2.npz")}
    heads = [ProjHead(1, 0, 1.5, "d1"), ProjHead(1, 2, 6.0, "d1"), ProjHead(2, 3, 2.0, "d1")]
    warn = []
    m = unit_matrices(w, dsets, heads, warn=warn.append)
    assert sorted(m) == [1, 2] and not warn
    A, B = m[1]
    assert A.shape == (1, 12) and B.shape == (16, 1)
    W = mats["blk.1.attn_output.weight"]
    r = _unit(dsets["d1"].raw[1].astype(np.float64))
    expect = W.astype(np.float64).copy()
    for h, s in ((0, 1.5), (2, 6.0)):
        blk = slice(3 * h, 3 * h + 3)
        expect[:, blk] -= s * np.outer(r, r @ W[:, blk])
    assert np.allclose(W + B @ A, expect, atol=1e-5)
    assert np.allclose((W + B @ A)[:, 3:6], W[:, 3:6])                 # 触らないヘッドの列はそのまま
    # 直交成分は変わらない
    d = (W + B @ A) - W
    assert np.allclose(d - np.outer(r, r @ d), 0, atol=1e-5)
    # 方向が混ざると階数 2 と警告
    m2 = unit_matrices(w, dsets, [ProjHead(1, 0, 1.0, "d1"), ProjHead(1, 1, 1.0, "d2"), ProjHead(1, 2, 1.0, "d2")],
                       warn=warn.append)
    assert m2[1][0].shape == (2, 12) and m2[1][1].shape == (16, 2) and warn and "階数が 2" in warn[0]
    # base: 方向セットの基準層の方向を全ヘッドに
    _dirs(tmp_path / "d3.npz", seed=2, base_layer=0)
    d3 = DirectionSet.load(tmp_path / "d3.npz")
    mb = unit_matrices(w, {"d3": d3}, [ProjHead(2, 1, 1.0, "d3")], direction_layer="base")
    assert np.allclose(-mb[2][1][:, 0], _unit(d3.raw[0]), atol=1e-6)
    with pytest.raises(GneError):                                    # 基準層が無い方向で base
        unit_matrices(w, dsets, [ProjHead(2, 1, 1.0, "d1")], direction_layer="base")


def _store_session(tmp_path, *, base_layer=None):
    """小さな GGUF の Session と、その model_id の置き場所に方向 d1 を置いたもの(エンジンは読まない)。"""
    from gne.api import Session
    mats = _tiny(tmp_path / "m.gguf")
    s = Session(tmp_path / "m.gguf", root=tmp_path / "root")
    _dirs(s.store.direction_dir(s.meta.model_id) / "d1.npz", seed=2, base_layer=base_layer)
    return s, mats


def test_export_cvec_rejects_proj_and_gain_lora_merges(tmp_path):
    from gguf import GGUFReader
    s, mats = _store_session(tmp_path)
    r = s.recipe(units=["1:H0:s2@d1", "1:H2:s3@d1"])
    with pytest.raises(GneError) as e:
        s.export_cvec(r, tmp_path / "x.gguf")
    assert e.value.code == "BAD_RECIPE"
    assert any("proj" in x for x in s.warnings)
    # gne gain は proj を同じ LoRA に入れる。同じ層のヘッドの gain とは連結して 1 つのテンソル(階数 = head_dim + 1)
    d = s.gain(s.recipe(units=["1:H0:s2@d1", "1:H2:s3@d1", "1:H1:x0", "2:H3:s1@d1"]), tmp_path / "g.lora.gguf")
    assert d["proj_layers"]["1"]["rank"] == 1 and d["proj_layers"]["1"]["folded"] and d["head_rank"]["1"] == 4
    rd = GGUFReader(tmp_path / "g.lora.gguf")
    t = {x.name: x for x in rd.tensors}
    assert [int(x) for x in t["blk.1.attn_output.weight.lora_b"].shape] == [4, 16]       # ne: [r, n_embd]
    assert [int(x) for x in t["blk.2.attn_output.weight.lora_b"].shape] == [1, 16]
    meta = json.loads(rd.fields["gne.headproj"].contents())
    assert {(x["layer"], x["head"], x["s"]) for x in meta["heads"]} == {(1, 0, 2.0), (1, 2, 3.0), (2, 3, 1.0)}
    # 合成した変化量 = gain の変化量 + proj の変化量
    A = np.array(t["blk.1.attn_output.weight.lora_a"].data).reshape(4, 12)
    B = np.array(t["blk.1.attn_output.weight.lora_b"].data).reshape(16, 4)
    W = mats["blk.1.attn_output.weight"].astype(np.float64)
    rr = _unit(np.load(s.direction_path("d1"))["layer_1"].astype(np.float64))
    exp = np.zeros_like(W)
    exp[:, 3:6] = -W[:, 3:6]
    for h, sv in ((0, 2.0), (2, 3.0)):
        exp[:, 3 * h:3 * h + 3] -= sv * np.outer(rr, rr @ W[:, 3 * h:3 * h + 3])
    assert np.allclose(B @ A, exp, atol=1e-4)


def test_bake_proj_equals_lora_delta(tmp_path):
    """焼き込みは元の重みに LoRA と同じ B @ A を足したもの(F16 で書いて比べる)。書き換えたテンソルと型の記録。"""
    from gguf import GGUFReader
    s, mats = _store_session(tmp_path)
    r = s.recipe(units=["1:H0:s2@d1", "1:H1:x0"])
    d = s.gain(r, tmp_path / "baked.gguf", bake=True, requant="f16")
    assert [m["tensor"] for m in d["modified"]] == ["blk.1.attn_output.weight"] and Path(d["sidecar"]).exists()
    rd = GGUFReader(tmp_path / "baked.gguf")
    t = {x.name: x for x in rd.tensors}
    got = np.array(t["blk.1.attn_output.weight"].data, dtype=np.float32).reshape(16, 12)
    W = mats["blk.1.attn_output.weight"].astype(np.float64)
    rr = _unit(np.load(s.direction_path("d1"))["layer_1"].astype(np.float64))
    exp = W.copy()
    exp[:, 3:6] = 0.0
    exp[:, 0:3] -= 2.0 * np.outer(rr, rr @ W[:, 0:3])
    assert np.allclose(got, exp, atol=2e-2)
    assert np.array_equal(np.array(t["blk.0.attn_output.weight"].data), mats["blk.0.attn_output.weight"])   # 他は元のまま
    rec = json.loads(rd.fields["gne.patches"].contents())
    assert rec[0]["tensor"] == "blk.1.attn_output.weight" and rec[0]["from"] == "F32"
    with pytest.raises(GneError):                                   # スケールパッチでは書けない
        s.gain(r, tmp_path / "p.gguf", bake=True, requant="patch")


def test_random_control_pool(tmp_path):
    """F.2: 対照は同数のヘッド、s の値は元と同じ。母集団 before(物差しの層まで)と same-layers。"""
    from gne.headproj import control_pool, proj_control_recipes, proj_heads
    s, _ = _store_session(tmp_path)
    r = s.recipe(units=["1:H0:s2@d1", "1:H2:s5@d1", "2:H1:s3@d1"], name="p")
    pool = control_pool(s.weights, r, pool="before", score_layers=[1])
    assert {L for L, _ in pool} == {0, 1} and (1, 0) not in pool and (1, 2) not in pool
    same = control_pool(s.weights, r, pool="same-layers")
    assert {L for L, _ in same} == {1, 2} and (2, 1) not in same
    cs = proj_control_recipes(s, r, 4, pool="before", score_layers=[2])
    assert len(cs) == 4
    for c in cs:
        hs = proj_heads(c)
        assert sorted(p.s for p in hs) == [2.0, 3.0, 5.0] and len({(p.layer, p.head) for p in hs}) == 3
        assert not {(p.layer, p.head) for p in hs} & {(1, 0), (1, 2), (2, 1)}
    assert [proj_heads(c) for c in cs] == [proj_heads(c) for c in proj_control_recipes(s, r, 4, pool="before", score_layers=[2])]
    with pytest.raises(GneError):
        control_pool(s.weights, r, pool="nope")


def test_bisect_and_spearman():
    from gne.causal import _bisect, spearman
    a, v, ok, n = _bisect(lambda x: 10 - 3 * x, 1.0, 10.0)
    assert ok and abs(v - 1.0) < 0.1 and abs(a - 3.0) < 0.05 and n <= 16
    a, v, ok, _ = _bisect(lambda x: 10 - x, -5.0, 4.0)
    assert not ok and a == 4.0 and v == 6.0                          # 上限でも届かない
    f = lambda x: (x - 1.0) ** 2 - 0.5                               # noqa: E731  単調でない(方式の比較の丸ごと倍率)
    assert not _bisect(f, 0.0, 4.0)[2]                               # 上限だけ見ると届かない
    a, v, ok, _ = _bisect(f, 0.0, 4.0, grid=8)
    assert ok and abs(a - (1 - 0.5 ** 0.5)) < 0.02
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2], [1, 2]) is None


def test_chat_minus_column_flips_s():
    from gne.chat import make_columns
    r = Recipe([parse_unit("3:H1:s2@d"), parse_unit("3:5:+0.4")], name="r", default_direction="d")
    lo, _, hi = make_columns(r)
    assert lo.recipe.units[0].k == -2.0 and hi.recipe.units[0].k == 2.0 and lo.recipe.default_direction == "d"


# --- モデルを使う ------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def session(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip(f"{MODEL} がありません")
    from gne.api import Session
    s = Session(MODEL, root=tmp_path_factory.mktemp("c06"), n_gpu_layers=NGL, n_ctx=2048)
    s.direction(contrast=EVIL, positions="last")
    yield s
    s.close()


@pytest.mark.model
def test_score_separation_and_base_layer(session):
    """B: A/B が分かれる(基準層の cohen_d)、基準層を方向セットに記録、後段の層とのコサイン。"""
    from gne.direction import base_layer_of
    d = session.direction_score("evil")
    best = d["best_layer"]
    row = next(r for r in d["layers"] if r["layer"] == best)
    assert row["cohen_d"] > 2 and row["A"]["mean"] > row["B"]["mean"] and d["base_recorded"]
    assert row["cos_base"] == pytest.approx(1.0, abs=1e-6)
    assert base_layer_of(session.direction_set("evil")) == best
    assert len(d["values"]["A"][str(best)]) == 12
    late = [r for r in d["layers"] if r["layer"] > best]
    assert late and min(r["cos_base"] for r in late) < 0.99          # 方向は層で回る
    # 介入つき: 点数の下がり方と直交成分のずれ。介入があるときは基準層を記録し直さない
    iv = session.direction_score("evil", layers=[best], units=[f"{best - 1}:H0:s4"])
    x = iv["layers"][0]
    assert set(x["drift"]) == {"A", "B"} and x["drift"]["B"]["orth"] >= 0 and not iv["base_recorded"]
    # 基準層を手で上書き
    m = session.direction_score("evil", layers=[best], set_base=best - 1)
    assert m["base_layer"] == best - 1 and base_layer_of(session.direction_set("evil")) == best - 1
    session.direction_score("evil", set_base=best)


@pytest.mark.model
def test_v24_v25_projection(session):
    """V24: LoRA の出力の差 = 方向成分を s 倍ぶん引いた値(s=1 で 0、s=2 で反転)。V25: 直交成分は不変、gain なら変わる。"""
    d = session.check_head_proj(10, 3, "evil", ss=(1.0, 2.0, 6.0))
    assert d["v24"] and d["v25"], d
    by = {r["s"]: r for r in d["rows"]}
    assert abs(by[1.0]["par_after_over_before"]) < 0.01 and by[2.0]["par_after_over_before"] == pytest.approx(-1.0, abs=0.01)
    assert all(r["rel_err_median"] < 0.03 and r["orth_change_rel_median"] < 0.01 for r in d["rows"])
    assert d["gain"]["orth_change_rel_median"] > 0.5


@pytest.mark.model
def test_screen_heads_both_modes_and_allocate(session):
    """C: ablate と marginal(行の中身・保存・V28 の順位相関)、D: 配分の二分探索と比較表。"""
    from gne.causal import load_causal
    best = session.direction_score("evil", layers=None, record_base=False)["base_layer"]
    sl = [best, best + 4]
    d = session.heads_causal("evil", layers=[best - 2, best - 1], candidates="contrib:4", score_layers=sl, b_examples=4)
    assert d["n_rows"] == 4 and d["primary"] == best
    for r in d["rows"]:
        assert set(r["score_drop"]) == {str(L) for L in sl} and r["contrib"] is not None and r["seconds"] > 0
        assert r["b_side_delta"] is not None and r["drift_orth_b"] >= 0
    assert len(load_causal(session, "evil", "ablate")) == 4
    top = d["rows"][:2]
    m = session.heads_causal("evil", mode="marginal", base_units=[f"{r['layer']}:H{r['head']}:s2" for r in top],
                             hard=4, score_layers=sl, b_examples=4)
    assert all(r["eta"] is not None and r["delta_s"] == 3.0 for r in m["rows"])
    assert sum(r["in_base"] for r in m["rows"]) == 2
    assert m["n_compared"] == 4 and "spearman_ablate_marginal" in m
    assert len(load_causal(session, "evil")) == 8                     # ablate と marginal の行が両方残る
    a = session.heads_allocate("evil", target=float(m["base_scores"][str(sl[-1])]["A"]) * 0.5, score_layer=sl[-1],
                               n_heads=(2, 1), hard=4, pool=3)
    ok = [r for r in a["table"] if not r.get("skipped")]
    assert ok and all(Path(r["recipe"]).exists() and r["remain"] >= 0 and "b_drift_orth" in r for r in ok)
    assert sum(r["recommended"] for r in a["table"]) == (1 if any(r.get("reached") for r in ok) else 0)
    assert Path(session.store.hard_dir(session.meta.model_id) / "evil.json").exists()
    rec = Recipe.load(ok[0]["recipe"])
    assert rec.default_direction == "evil" and all(u.mode == "proj" for u in rec.units)


@pytest.mark.model
def test_v26_bake_matches_lora(session, tmp_path):
    r = session.recipe(units=["12:H3:s4@evil", "12:H7:s2@evil"], name="v26")
    d = session.headproj_export(r, tmp_path / "v26.gguf", bake=True, verify=True)
    v = d["verify"]
    assert v["pass"] and v["top1_match"] >= 0.9 and all(g < 0.5 for g in v["top1_mismatch_gaps"]), v
    assert all(x["ok"] for x in v["scores"].values()) and abs(v["ppl_diff_pct"]) < 0.5
    f = session.headproj_export(r, tmp_path / "v26f.gguf", bake=True, requant="f16", verify=True)["verify"]
    assert f["pass"] and f["kl_first_token"] <= v["kl_first_token"] + 1e-6, f      # F16 なら Q8_0 より近い


@pytest.mark.model
def test_v29_random_controls_in_eval(session, tmp_path):
    """F.1・F.2・V29: proj のレシピの eval に直交成分のずれ、同数・同じ s のランダムなヘッドの対照と null_pct。"""
    (tmp_path / "ppl").mkdir()
    (tmp_path / "ppl" / "en.txt").write_text("Alice was beginning to get very tired of sitting by her sister on the bank. " * 3)
    (tmp_path / "zh_only_chars.txt").write_text("# test\n这们说\n")
    suite = {"name": "mini", "max_tokens": {"exact": 12, "format": 24, "mix": 24, "persona": 16},
             "exact": [{"id": "add", "prompt": "What is 17 + 25? Answer with just the number.", "answer": ["42"]}],
             "format": [], "mix": [], "persona": {"prompt_sets": ["en"]}, "ppl": [{"id": "en", "path": "ppl/en.txt"}]}
    (tmp_path / "mini.yaml").write_text(yaml.safe_dump(suite, allow_unicode=True))
    r = session.recipe(units=["12:H3:s4@evil", "12:H7:s4@evil"], name="v29")
    e = session.eval(r, doses=[1.0], suite=tmp_path / "mini.yaml", random_controls=2)
    row = next(x for x in e["rows"] if x["dose"] == 1.0)
    assert e["direction_metrics"]["direction"] == "evil" and row["drift_orth_b"] is not None
    assert row["dir_null_pct"] is not None and len(row["ctrl"]["rows"]) == 2
    assert all(len(c["units"]) == 2 for c in row["ctrl"]["rows"])
    base = next(x for x in e["rows"] if x["dose"] == 0.0)
    assert base["dir_drop_a"] == 0.0 and base["drift_orth_b"] == 0.0
    key = "rate_null_pct" if (e["direction_metrics"]["base_rate"] or 0) > 0 else "dir_null_pct"
    assert (row[key] < 0.95) == bool(row.get("notes"))               # 対照を超えていなければ明示


@pytest.mark.model
def test_linear_layer_address(tmp_path):
    """線形注意の層(ssm_out の列ブロック)でも V24 が通る(住所の分岐)。"""
    if not LINEAR.exists():
        pytest.skip(f"{LINEAR} がありません")
    from gne.api import Session
    s = Session(LINEAR, root=tmp_path, n_gpu_layers=NGL, n_ctx=1024)
    try:
        L = next(li.index for li in s.weights.arch.layers if li.mixer == "linear_attention" and li.index >= 4)
        assert s.weights.head_block(L, 1)[0].endswith("ssm_out.weight")
        s.direction(contrast=EVIL, positions="last")
        d = s.check_head_proj(L, 1, "evil", ss=(1.0, 2.0))
        assert d["v24"] and d["out_tensor"].startswith("linear_attn_out"), d
    finally:
        s.close()


@pytest.mark.model
def test_cli_json_schemas(tmp_path):
    """V10 と同じ: 新しいコマンドの --json が stdout に JSON だけを出し、スキーマを通る。"""
    import subprocess
    import sys
    from gne.schemas import COMMAND_SCHEMAS
    if not MODEL.exists():
        pytest.skip(f"{MODEL} がありません")
    gne = str(Path(sys.executable).parent / "gne")
    env = {**os.environ, "GNE_ROOT": str(tmp_path / "root")}

    def run(*args):
        p = subprocess.run([gne, *args, "-m", str(MODEL), "--n-gpu-layers", str(NGL), "--n-ctx", "2048", "--json"],
                           capture_output=True, text=True, env=env, timeout=1800)
        last = json.loads([ln for ln in p.stdout.splitlines() if ln.strip()][-1])
        assert p.returncode == 0, (args, last, p.stderr[-1500:])
        COMMAND_SCHEMAS[last["command"]].model_validate(last)
        return last
    run("direction", "--contrast", str(EVIL))                                # サブコマンドなしは従来どおり
    d = run("direction", "score", "--name", "evil")
    b = d["base_layer"]
    run("direction", "score", "--name", "evil", "--layers", str(b), "--unit", f"{b - 2}:H1:s3")
    run("headproj", "check", "--layer", str(b - 2), "--head", "1", "--direction", "evil", "--s", "1,2")
    c = run("heads", "causal", "--name", "evil", "--layers", str(b - 1), "--candidates", "contrib:2", "--b-examples", "4")
    assert c["n_rows"] == 2
    e = run("headproj", "export", "--unit", f"{b - 1}:H{c['rows'][0]['head']}:s4", "--direction", "evil",
            "-o", str(tmp_path / "x.lora.gguf"))
    assert e["proj_layers"][str(b - 1)]["rank"] == 1


# --- レシピで A / B を作る対照集合(追加)---------------------------------------------------------------

def test_recipe_contrast_set_parsing(tmp_path):
    from gne.reverse import ContrastSet, combine
    p = tmp_path / "c.yaml"
    p.write_text("name: pushed\nrecipe_a: [\"8:100:+1.0\"]\nuser_prompts: [\"Hi\", \"Who are you?\"]\n", encoding="utf-8")
    cs = ContrastSet.load(p)
    assert cs.has_recipes and cs.system_a is None and cs.system_b is None and cs.base_dir == tmp_path
    assert cs.convs("A") == [[{"role": "user", "content": "Hi"}], [{"role": "user", "content": "Who are you?"}]]
    assert cs.describe()["A"]["recipe"] == ["8:100:+1.0"] and cs.describe()["B"]["recipe"] is None
    with pytest.raises(GneError) as e:
        cs.require_systems("gne heads contrast")
    assert e.value.code == "BAD_CONTRAST"
    # system だけの対照集合は今までどおり。system が同じなら断る
    q = tmp_path / "s.yaml"
    q.write_text("system_a: x\nsystem_b: y\nuser_prompts: [Hi]\n", encoding="utf-8")
    cq = ContrastSet.load(q)
    assert not cq.has_recipes and cq.convs("A")[0][0] == {"role": "system", "content": "x"}
    cq.require_systems("x")
    q.write_text("system_a: x\nsystem_b: x\nuser_prompts: [Hi]\n", encoding="utf-8")
    with pytest.raises(GneError):
        ContrastSet.load(q)
    q.write_text("system_a: x\nuser_prompts: [Hi]\n", encoding="utf-8")          # 片方の system だけ: B は system なし
    one = ContrastSet.load(q)
    assert one.convs("A")[0][0]["role"] == "system" and one.convs("B") == [[{"role": "user", "content": "Hi"}]]
    with pytest.raises(GneError):
        one.require_systems("gne heads contrast")
    q.write_text("user_prompts: [Hi]\n", encoding="utf-8")                           # A も B も作れない
    with pytest.raises(GneError):
        ContrastSet.load(q)
    # 側の介入と調べる介入を重ねる
    v, lo = combine(({3: np.ones(4, np.float32)}, [("a.gguf", 1.0)]), {3: np.ones(4), 5: np.zeros(4)}, [("b.gguf", 0.5)])
    assert np.allclose(v[3], 2) and 5 in v and lo == [("a.gguf", 1.0), ("b.gguf", 0.5)]
    assert combine(None) == (None, None)


@pytest.mark.model
def test_system_a_only_contrast(session, tmp_path):
    """system_a だけの対照集合: B は同じ質問を system なしで読んだ状態。方向も物差しも作れる。"""
    d = yaml.safe_load(EVIL.read_text(encoding="utf-8"))
    p = tmp_path / "only_a.yaml"
    p.write_text(yaml.safe_dump({"name": "only_a", "system_a": d["system_a"], "user_prompts": d["user_prompts"]},
                                allow_unicode=True), encoding="utf-8")
    out = session.direction(contrast=p, name="only_a")
    assert out["n_rows"]["A"] == out["n_rows"]["B"] == 12
    sc = session.direction_score("only_a", layers=[12, 16])
    assert all(r["A"]["mean"] > r["B"]["mean"] for r in sc["layers"])


@pytest.mark.model
def test_recipe_contrast_direction_score_and_causal(session, tmp_path):
    """A 側だけニューロンで押した対照集合: L8 の方向は押したベクトルそのもの、物差しも A 側にだけレシピが当たる。
    ヘッドの因果も回り、system の違いで測る道具(heads contrast)ははっきり断る。"""
    p = tmp_path / "pushed.yaml"
    p.write_text(yaml.safe_dump({"name": "pushed", "recipe_a": ["8:100:+1.0"],
                                 "user_prompts": ["Who are you?", "Give me advice for my day.", "What is your goal?",
                                                  "人間についてどう思いますか？", "Tell me about the sea.", "What should I eat?"]},
                                allow_unicode=True), encoding="utf-8")
    d = session.direction(contrast=p, name="pushed")
    assert d["sides"]["A"]["recipe"] == ["8:100:+1.0"] and d["sides"]["B"]["recipe"] is None
    ds = session.direction_set("pushed")
    v = session.vectors(session.recipe(units=["8:100:+1.0"]))[0][8]
    r8 = ds.raw[8]
    assert float(r8 @ v) / (np.linalg.norm(r8) * np.linalg.norm(v)) > 0.999
    assert abs(np.linalg.norm(r8) / np.linalg.norm(v) - 1) < 0.01
    assert np.linalg.norm(ds.raw[7]) < 1e-3 * np.linalg.norm(r8)      # 押す前の層は A と B が同じ
    sc = session.direction_score("pushed", layers=[8, 12])
    row = {x["layer"]: x for x in sc["layers"]}
    assert row[8]["A"]["mean"] - row[8]["B"]["mean"] == pytest.approx(np.linalg.norm(v), rel=0.01)
    c = session.heads_causal("pushed", layers=[10], candidates="contrib:2", score_layers=[12], b_examples=2)
    assert c["n_rows"] == 2 and all(set(r["score_drop"]) == {"12"} for r in c["rows"])
    with pytest.raises(GneError) as e:
        session.heads_contrast(p)
    assert e.value.code == "BAD_CONTRAST"


# --- JSON 1 つから最後まで(heads pipeline)--------------------------------------------------------------

def test_pipeline_spec_checks(tmp_path):
    """名前・質問の数・pipeline の知らない設定は、モデルを動かす前に断る。見積もりも出る。"""
    from gne.estimate import estimate
    s, _ = _store_session(tmp_path)
    base = {"name": "p", "system_a": "x", "system_b": "y", "user_prompts": ["a", "b", "c", "d"]}
    for bad in ({**base, "name": "a b"}, {**base, "user_prompts": ["a", "b"]}, {**base, "user_prompts": "abc"},
                {**base, "pipeline": {"nope": 1}}):
        with pytest.raises(GneError):
            s.heads_pipeline(bad)
    est = estimate(s, "head_pipeline", {"spec": {**base, "user_prompts": ["q"] * 300}, "max_examples": 32})
    assert est["seconds"] and est["detail"]["prompts"] == 300 and est["detail"]["screened"] == 32


@pytest.mark.model
def test_pipeline_runs_end_to_end(session, tmp_path):
    """対照集合の表から、方向 → 基準層 → 因果 → 配分 が 1 回で回り、結果とレシピが残る。止めて測る段階は間引いた質問で。"""
    d = yaml.safe_load(EVIL.read_text(encoding="utf-8"))
    spec = {"name": "pipe_t", "system_a": d["system_a"], "system_b": d["system_b"], "user_prompts": d["user_prompts"] * 2,
            "pipeline": {"n_heads": [2], "max_examples": 8, "top_layers": 1, "hard": 4}}
    out = session.heads_pipeline(spec)
    assert out["n_prompts"] == 24 and Path(out["contrast"]).exists() and Path(out["path"]).exists()
    assert out["direction"]["n_rows"]["A"] == 24                       # 方向は全部の質問から
    assert out["score"]["n"]["A"] == 8 and out["causal"]["n_a"] == 8   # 止めて測る段階は 8 問
    assert out["score"]["base_layer"] == session.direction_set("pipe_t").meta["base_layer"]
    al = out["allocate"]
    assert al["initial"]["heads"] and len(al["table"]) == 1 and al["max_examples"] == 8
    S = out["causal"]["score_layers"][-1]
    bs = out["causal"]["base_scores"][str(S)]
    assert out["target"] == pytest.approx(0.5 * (bs["A"] + bs["B"])) and al["remain_threshold"] == pytest.approx(out["target"])
    for r in out["causal"]["layers"] or []:
        assert (r.get("excluded") == "breaks") == (r["drift_orth_b"] > 0.5)
    if out["recommended"]:
        assert Path(out["recommended"]["recipe"]).exists()


# --- 側ごとに別の質問を読む対照集合(追加)---------------------------------------------------------------

JA = ["今日の天気はどうですか？", "おすすめの本を教えてください。", "週末は何をしたらいいですか？", "好きな食べ物は何ですか？",
      "朝ごはんに何を食べるといいですか？", "海についてどう思いますか？", "仕事のやる気を出すにはどうすればいいですか？", "猫と犬はどちらが好きですか？"]
EN = ["How is the weather today?", "Can you recommend a book?", "What should I do this weekend?", "What is your favorite food?",
      "What is a good breakfast?", "What do you think about the sea?", "How can I get motivated at work?", "Do you like cats or dogs?",
      "What is the capital of France?", "Tell me a joke."]


def test_separate_prompts_parsing(tmp_path):
    from gne.direction import load_contrast
    from gne.reverse import ContrastSet
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"name": "ja_en", "prompts_a": JA, "prompts_b": EN}, allow_unicode=True), encoding="utf-8")
    cs, val = load_contrast(p)
    assert cs.separate_prompts and cs.prompts("A") == JA and cs.prompts("B") == EN and cs.user_prompts == JA
    assert cs.convs("B")[0] == [{"role": "user", "content": EN[0]}]
    assert cs.describe()["A"]["n_prompts"] == 8 and cs.describe()["B"]["n_prompts"] == 10
    assert val.prompts == JA and val.add_prompts() == EN                   # 足して見るのは中立(B)の質問
    with pytest.raises(GneError):
        cs.require_systems("gne heads contrast")
    # 片側だけ書けば、もう片側は user_prompts
    p.write_text(yaml.safe_dump({"prompts_a": JA, "user_prompts": EN}, allow_unicode=True), encoding="utf-8")
    c2 = ContrastSet.load(p)
    assert c2.prompts("A") == JA and c2.prompts("B") == EN
    # 質問も system も recipe も同じなら断る。質問が無くても断る
    p.write_text(yaml.safe_dump({"prompts_a": JA, "prompts_b": JA}, allow_unicode=True), encoding="utf-8")
    with pytest.raises(GneError):
        ContrastSet.load(p)
    p.write_text(yaml.safe_dump({"prompts_a": JA, "system_a": "x"}, allow_unicode=True), encoding="utf-8")
    with pytest.raises(GneError):
        ContrastSet.load(p)
    p.write_text(yaml.safe_dump({"prompts_a": "x", "prompts_b": EN}, allow_unicode=True), encoding="utf-8")
    with pytest.raises(GneError):
        ContrastSet.load(p)


@pytest.mark.model
def test_separate_prompts_direction_score_causal_pipeline(session, tmp_path):
    """素のモデルに日本語の質問(A)と英語の質問(B)を読ませる: 方向は側ごとの質問の数で作られ、物差しは A と B を
    別々に間引き、ヘッドの因果と pipeline も回る。"""
    p = tmp_path / "ja_en.yaml"
    p.write_text(yaml.safe_dump({"name": "ja_en", "prompts_a": JA, "prompts_b": EN}, allow_unicode=True), encoding="utf-8")
    d = session.direction(contrast=p, name="ja_en")
    assert d["n_rows"] == {"A": 8, "B": 10} and d["sides"]["B"]["n_prompts"] == 10
    sc = session.direction_score("ja_en", max_examples=6)
    assert sc["n"] == {"A": 6, "B": 6} and len(sc["prompts_b"]) == 6 and sc["prompts_b"][0] == EN[0]
    best = next(r for r in sc["layers"] if r["layer"] == sc["best_layer"])
    assert best["cohen_d"] > 2 and best["A"]["mean"] > best["B"]["mean"]
    c = session.heads_causal("ja_en", layers=[sc["best_layer"] - 1], candidates="contrib:2", b_examples=4)
    assert c["n_rows"] == 2 and c["n_b"] == 4
    out = session.heads_pipeline({"name": "ja_en_p", "prompts_a": JA, "prompts_b": EN,
                                  "pipeline": {"n_heads": [2], "max_examples": 6, "top_layers": 1, "hard": 3}})
    assert out["n_prompts"] == 8 and out["n_prompts_b"] == 10 and out["allocate"]["table"]


# --- 強さの候補づくり(組 × 倍率で生成し、名前を伏せた採点ファイルを作る)------------------------------------------

def _fake_alloc(s, name="cand_t"):
    """配分の結果の形だけを作る(heads allocate を回さずに候補づくりを試す)。"""
    hd = s.store.heads_dir(s.meta.model_id)
    hd.mkdir(parents=True, exist_ok=True)
    rp = hd / f"{name}.alloc-n2.yaml"
    units = ["10:H3:s3@evil", "14:H5:s2@evil"]
    Recipe([parse_unit(u) for u in units], s.meta.model_id, name=f"{name}-n2", default_direction="evil").save(rp)
    al = {"name": name, "direction": "evil", "table": [
        {"n_heads": 2, "units": units, "recipe": str(rp), "recommended": True, "reached": True},
        {"n_heads": 1, "skipped": "効率が正のヘッドが 0 個しかありません"}]}
    p = hd / f"{name}.alloc.json"
    p.write_text(json.dumps(al, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.mark.model
def test_candidates_generate_blind_files_and_tally(session, tmp_path):
    """候補 = 無介入 + 組 × 倍率。同じ種なので無介入は自分と同じ。採点のファイルには候補の名前が出ず、
    鍵で候補に戻せる。採点を書くと候補ごとに集計され、作り直すと前の採点は退避される。"""
    from gne.schemas import SCHEMA_VERSION, HeadsCandidatesOut
    from gne.estimate import estimate
    _fake_alloc(session)
    est = estimate(session, "heads_candidates", {"alloc": "cand_t", "scales": [0.5, 1.0], "max_prompts": 3})
    assert est["detail"]["candidates"] == 3 and est["detail"]["replies"] == 18
    qs = ["Who are you?", "What do you think about humans?", "Tell me about your plans.", "What is love?"]
    d = session.heads_candidates("cand_t", prompts=qs, scales=[0.5, 1.0], max_prompts=3, max_tokens=12, seed=3)
    names = [c["name"] for c in d["candidates"]]
    assert names == ["base", "n2x0.5", "n2x1"] and d["held_out"] == {"A": True, "B": True}
    assert len(d["prompts"]["A"]) == 3 and d["prompts"]["A"][0] == qs[0] and d["prompts"]["A"][-1] == qs[-1]
    base = d["candidates"][0]
    assert base["same_as_base_a"] == 1.0 and base["same_as_base_b"] == 1.0
    assert set(d["replies"]) == {f"{n}|{s}" for n in names for s in "AB"}
    rnd = Path(d["judge"]["round"]).read_text(encoding="utf-8")
    assert d["judge"]["n"] == 9 and rnd.count("[REPLY]") == 9 and "n2x" not in rnd and "base" not in rnd
    key = json.loads((Path(d["judge"]["dir"]) / "round_01.key.json").read_text(encoding="utf-8"))
    assert sorted(k["id"] for k in key) == list(range(1, 10)) and {k["side"] for k in key} == {"A"}
    assert Path(d["judge"]["rubric"]).exists()
    env = {"schema_version": SCHEMA_VERSION, "command": "heads candidates", "model_id": "x", "llama_cpp_version": "x",
           "seed": 0, "generated_at": "x"}
    HeadsCandidatesOut.model_validate({**env, **{k: v for k, v in d.items()}})
    # 採点(ふつうの返事は P 1、組は P 4)→ 集計
    lines = "\n".join(f"{k['id']} {1 if k['cand'] == 'base' else 4} {5 if k['cand'] == 'base' else 3}" for k in key[:-1])
    t = session.heads_candidates_tally("cand_t", scores=lines)
    assert t["scores"]["base"]["A"]["P"] == 1.0 and t["scores"]["base"]["A"]["min_PC"] == 1.0
    assert t["scores_missing"] == [key[-1]["id"]]
    t = session.heads_candidates_tally("cand_t", scores={str(key[-1]["id"]): [4, 3]})
    assert t["scores_missing"] == [] and t["scores"]["n2x1"]["A"]["min_PC"] == 3.0 and t["scores"]["n2x1"]["A"]["P_ge3"] == 1.0
    with pytest.raises(GneError):
        session.heads_candidates_tally("cand_t", scores="1 9 9")
    blind = session.candidates_blind("cand_t")
    assert len(blind["items"]) == 9 and "cand" not in blind["items"][0] and blind["scores"]
    assert session.candidates_list()[0]["name"] == "cand_t"
    # 作り直すと、前の採点は名前を変えて残る(番号が合わなくなるので)
    assert t["best"]["remove"]["base"] == t["scores"]["base"]["A"]["remove"] and "add" not in t["best"]
    session.heads_candidates("cand_t", prompts=qs[:2], scales=[1.0], max_tokens=8, sampling="greedy", judge_b=True)
    dd = Path(d["judge"]["dir"])
    assert not (dd / "round_01.scores.txt").exists() and list(dd.glob("round_01.scores.*.old.txt"))
    key2 = json.loads((dd / "round_01.key.json").read_text(encoding="utf-8"))
    assert {k["side"] for k in key2} == {"A", "B"} and len(key2) == 2 * 2 * 2
    # 負の倍率は強める向き: B 側を採点する。重ねるレシピは無介入(base)にも当たる
    seed_r = tmp_path / "seed.yaml"
    Recipe([parse_unit("20:100:+0.3")], session.meta.model_id, name="seed").save(seed_r)
    e = session.heads_candidates("cand_t", prompts=qs[:2], scales=[1.0, -1.0], max_tokens=8, seed=1, with_recipe=seed_r)
    by = {c["name"]: c for c in e["candidates"]}
    assert by["n2x1"]["judge_sides"] == "A" and by["n2x-1"]["judge_sides"] == "B" and by["base"]["judge_sides"] == "AB"
    assert by["n2x-1"]["goal"] == "add" and e["with_recipe"] and e["judge"]["n"] == 2 * 4
    plain = session.heads_candidates("cand_t", prompts=qs[:2], scales=[1.0], max_tokens=8, seed=1, name="cand_plain")
    assert plain["replies"]["base|A"] != e["replies"]["base|A"]          # 種火が base にも当たっている
    key3 = json.loads((dd / "round_01.key.json").read_text(encoding="utf-8"))
    t = session.heads_candidates_tally("cand_t", scores="\n".join(f"{k['id']} {4 if k['cand'] == 'n2x-1' else 2} 4" for k in key3))
    assert t["best"]["add"]["name"] == "n2x-1" and t["best"]["add"]["value"] == 4.0 and t["best"]["add"]["base"] == 2.0
    assert t["scores"]["n2x1"]["A"]["remove"] == 4.0
    with pytest.raises(GneError):
        session.heads_candidates("cand_t", n_heads=[7])
    with pytest.raises(GneError):
        session.heads_candidates("cand_t", sampling="nope")
