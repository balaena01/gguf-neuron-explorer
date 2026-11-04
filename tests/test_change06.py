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
def test_contexts_fit_the_questions_and_steer_is_freed(session):
    """短い会話しか読まない処理(方向の点数・因果・配分)は、内部を測るコンテキストを最長の会話に合わせて小さく作り、
    生成用のコンテキストは先に解放する(大きいモデルで VRAM からあふれて遅くなるのを避ける)。"""
    e = session.engine
    assert e.fit_ctx(None) == e.n_ctx and e.fit_ctx(1) == 256 and e.fit_ctx(257) == 512 and e.fit_ctx(e.n_ctx) == e.n_ctx
    with pytest.raises(ValueError):
        e.fit_ctx(e.n_ctx + 1)
    from gne.capture import Capture
    cap = Capture()
    c1 = e.probe_ctx(cap, need=100)
    assert c1.n_ctx == 256 and c1.n_ubatch == 256
    assert e.probe_ctx(cap, need=200) is c1                              # 足りていれば作り直さない
    c2 = e.probe_ctx(cap, need=600)
    assert c2 is not c1 and c2.n_ctx == 768
    assert e.probe_ctx(cap, need=100).n_ctx == 256                       # 2 倍より大きければ小さく作り直す
    assert e.probe_ctx(cap).n_ctx == e.n_ctx                             # 長さを言わない呼び出しは今までどおり
    e.steer_ctx()
    d = session.direction_score("evil", layers=[10])
    assert e._steer is None and e._probe.n_ctx < e.n_ctx and d["layers"]
    from gne.attn import capture_attention
    capture_attention(session, ["Hello there, how are you?"], layers=[10], want_pattern=False)
    assert e._attn_probe.n_ctx == 256
    e.free_attn_probe()


@pytest.mark.model
def test_pipeline_resumes_direction_and_causal(session, tmp_path):
    """同じ JSON・同じ設定で流し直すと、作り済みの方向と因果の結果を使い回して配分からやり直す。resume=False は作り直す。
    設定(top_layers)が違えば因果は作り直す。"""
    d = yaml.safe_load(EVIL.read_text(encoding="utf-8"))
    spec = {"name": "pipe_r", "system_a": d["system_a"], "system_b": d["system_b"], "user_prompts": d["user_prompts"],
            "pipeline": {"n_heads": [2], "max_examples": 8, "top_layers": 1, "hard": 4}}
    first = session.heads_pipeline(dict(spec))
    assert not first["reused"] and not first["direction"]["reused"]
    again = session.heads_pipeline(dict(spec))
    assert again["reused"] == ["direction", "causal"] and again["causal"]["reused"]
    assert again["causal"]["top"] == first["causal"]["top"] and again["causal"]["score_layers"] == first["causal"]["score_layers"]
    assert again["causal"]["seconds"] < first["causal"]["seconds"]
    assert [r["units"] for r in again["allocate"]["table"] if not r.get("skipped")] == \
        [r["units"] for r in first["allocate"]["table"] if not r.get("skipped")]
    other = session.heads_pipeline({**spec, "pipeline": {**spec["pipeline"], "top_layers": 2}})
    assert other["reused"] == ["direction"]
    fresh = session.heads_pipeline(dict(spec), resume=False)
    assert fresh["reused"] == []


def test_pipeline_templates_are_valid(tmp_path):
    """configs/pipeline/ の用途別のテンプレートは、そのまま heads pipeline の spec として通る(about は説明だけ)。"""
    import re
    from gne.api import Session
    from gne.direction import load_contrast
    files = sorted((ROOT / "configs" / "pipeline").glob("*.json"))
    assert len(files) == 5
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        about = d.pop("about")
        assert about["ja"]["title"] and about["en"]["title"] and about["ja"]["text"] and about["en"]["text"]
        assert re.fullmatch(r"[\w.\-]+", d["name"])
        assert set(d.get("pipeline") or {}) <= set(Session.PIPELINE_DEFAULTS)
        for u in d.get("recipe_a") or []:
            parse_unit(u)
        p = tmp_path / f"{d['name']}.yaml"
        p.write_text(yaml.safe_dump({k: v for k, v in d.items() if k != "pipeline"}, allow_unicode=True), encoding="utf-8")
        cs, val = load_contrast(p)
        assert min(len(cs.prompts("A")), len(cs.prompts("B"))) >= 16
        assert val.match and val.target and re.compile(val.match)
        key = lambda q: json.dumps(q, ensure_ascii=False)                    # 質問は文字列か会話の履歴(リスト)
        held = {key(q) for q in val.prompts} if val.explicit else set()
        assert not held & {key(q) for side in "AB" for q in cs.prompts(side)}   # 確かめ用の質問は調整に使う質問と重ならない


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


# --- 会話の履歴を書いた質問(多ターン)-----------------------------------------------------------------

HIST = [{"role": "user", "content": "I adopted a puppy yesterday."}, {"role": "assistant", "content": "Congratulations! What is its name?"},
        {"role": "user", "content": "Max. What should I feed him?"}]


def test_multi_turn_questions_parse(tmp_path):
    """1 問は文字列か会話の履歴(user と assistant が交互で user で始まる。assistant で終われば書きかけの返事)。崩れた履歴は読む前に断る。
    同じ返事を 2 つの system で読ませる道具は、履歴の質問を断る。"""
    from gne.reverse import ContrastSet, _conv, check_question, question_text
    check_question("hi")
    check_question(HIST)
    check_question(HIST[:2])                                                 # assistant で終われば書きかけの返事
    with pytest.raises(GneError):
        check_question(HIST[:2], in_progress=False)                          # 読めない道具(fieldtest)は断る
    for bad in ([], [{"role": "assistant", "content": "x"}], [{"role": "user", "content": "a"}, {"role": "assistant", "content": ""}],
                [{"role": "user", "content": 3}],
                [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}], [{"role": "user", "content": "a", "x": 1}], 5):
        with pytest.raises(GneError):
            check_question(bad)
    assert _conv("SYS", HIST)[0] == {"role": "system", "content": "SYS"} and _conv(None, HIST)[-1]["content"].startswith("Max.")
    assert question_text(HIST).startswith("user: I adopted") and question_text("q") == "q"
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"name": "c", "system_a": "Be evil.", "system_b": "Be nice.",
                                 "user_prompts": ["Who are you?", HIST, "Hello", "Hi"]}), encoding="utf-8")
    cs = ContrastSet.load(p)
    assert cs.multi_turn and cs.convs("A")[1][-1]["content"].startswith("Max.") and cs.convs("A")[1][0]["role"] == "system"
    with pytest.raises(GneError):
        cs.require_systems("gne heads contrast")
    p.write_text(yaml.safe_dump({"name": "c", "system_a": "x", "user_prompts": ["a", [{"role": "user"}]]}), encoding="utf-8")
    with pytest.raises(GneError):
        ContrastSet.load(p)


@pytest.mark.model
def test_multi_turn_questions_measured_before_last_reply(session, tmp_path):
    """履歴の質問は、最後の user の発話への返事の直前で測る(assistant の位置は最後の返事の見出し)。方向・点数・候補づくりが通る。"""
    from gne.reverse import ContrastSet, contrast_sequences
    d = yaml.safe_load(EVIL.read_text(encoding="utf-8"))
    qs = d["user_prompts"][:5] + [HIST]
    p = tmp_path / "mt.yaml"
    p.write_text(yaml.safe_dump({"name": "mt", "system_a": d["system_a"], "system_b": d["system_b"], "user_prompts": qs,
                                 "validate": d["validate"]}, allow_unicode=True), encoding="utf-8")
    cs = ContrastSet.load(p)
    e = session.engine
    seqs = contrast_sequences(e, cs)
    ids, start = seqs["A"][-1]
    full = e.detokenize(ids)
    assert "Congratulations" in full and "Max." in full                      # 履歴が全部入っている
    assert "Max." not in e.detokenize(ids[start:]) and "Max." in e.detokenize(ids[:start])   # 測る位置は最後の返事の見出し
    session.direction(contrast=p, name="mt", positions="last")
    sc = session.direction_score("mt", layers=[12])
    assert sc["layers"][0]["A"]["n"] == len(qs)


THINK_TPL = ("{% for m in messages %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>{% endfor %}"
             "{% if add_generation_prompt %}<assistant>{% if enable_thinking %}<think>\n{% else %}<think>\n\n</think>\n\n{% endif %}{% endif %}")


def test_reply_in_progress_renders_open():
    """assistant で終わる質問は書きかけの返事: 閉じずに生成プロンプトの続きに置く。<think> で始まれば thinking の途中で、
    テンプレートが開いた <think> と重ねない。生成プロンプトなしは、その前の会話だけ。印のない assistant は今までどおり。"""
    from gne.engine import render_template
    from gne.fieldtest import load_long_tasks, parse_prompt_sets
    from gne.reverse import _conv
    q = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "<think>\nabc"}]
    assert render_template(THINK_TPL, _conv(None, q)) == "<user>Q</user><assistant><think>\nabc"
    assert render_template(THINK_TPL, _conv("S", q)) == "<system>S</system><user>Q</user><assistant><think>\nabc"
    body = [q[0], {"role": "assistant", "content": "The answer"}]
    assert render_template(THINK_TPL, _conv(None, body)) == "<user>Q</user><assistant><think>\n\n</think>\n\nThe answer"
    assert render_template(THINK_TPL, _conv(None, q), add_generation_prompt=False) == "<user>Q</user>"
    settled = render_template(THINK_TPL, q)                                  # 印がない = 閉じた発話 + 新しい返事
    assert settled.startswith("<user>Q</user><assistant><think>\nabc</assistant><assistant>")
    no_tag = "{% for m in messages %}[{{ m.content }}]{% endfor %}{% if add_generation_prompt %}A:{% endif %}"
    assert render_template(no_tag, _conv(None, q)) == "[Q]A:<think>\nabc"     # <think> を開かないテンプレート
    old = [{"role": "user", "content": "<think>x"}, {"role": "assistant", "content": "<think>\ny"}]
    assert render_template(no_tag, _conv(None, old)) == "[<think>x]A:<think>\ny"   # 履歴の中の <think> は削らない
    with pytest.raises(GneError):
        parse_prompt_sets({"s": [q]})                                        # fieldtest は書きかけを読まない
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "t.json").write_text(json.dumps([q]), encoding="utf-8")
        with pytest.raises(GneError):
            load_long_tasks(Path(td) / "t.json")


@pytest.mark.model
def test_causal_skips_questions_longer_than_the_capture_limit(session, tmp_path):
    """止めて測る段階に長さの制限はない。直接の書き込みは捕捉の上限より長い質問を除いて測り、途中で止まらない。"""
    d = yaml.safe_load(EVIL.read_text(encoding="utf-8"))
    long_q = "Tell me about this. " + "word " * (session.engine.probe_ubatch + 40)
    qs = d["user_prompts"][:5] + [long_q]
    p = tmp_path / "ln.yaml"
    p.write_text(yaml.safe_dump({"name": "ln", "system_a": d["system_a"], "system_b": d["system_b"], "user_prompts": qs,
                                 "validate": d["validate"]}, allow_unicode=True), encoding="utf-8")
    session.direction(contrast=p, name="ln", positions="last")
    r = session.heads_causal("ln", contrast=p, top_layers=1, hard=2)
    assert r["name"] == "ln" and Path(r["path"]).is_file()


@pytest.mark.model
def test_reply_in_progress_is_measured_at_its_end(session, tmp_path):
    """書きかけの返事は最後のトークンで測る。区間は返事の側に入り、生成はその続きから書く。ひな形がそのまま流れる。"""
    from gne.reverse import ContrastSet, _conv, contrast_sequences
    from gne.when import chat_segments
    d = json.loads((ROOT / "configs" / "pipeline" / "reply_in_progress.json").read_text(encoding="utf-8"))
    p = tmp_path / "rp.yaml"
    p.write_text(yaml.safe_dump({"name": "rp", "prompts_a": d["prompts_a"], "prompts_b": d["prompts_b"], "validate": d["validate"]},
                                allow_unicode=True), encoding="utf-8")
    cs = ContrastSet.load(p)
    assert cs.multi_turn and cs.separate_prompts
    e = session.engine
    ids, start = contrast_sequences(e, cs)["A"][0]
    text = cs.prompts("A")[0][-1]["content"]
    assert e.detokenize(ids).rstrip().endswith(text.rstrip()) and "<think>\n\n</think>" not in e.detokenize(ids)   # 閉じず、空の think 欄もない
    assert text[-20:] in e.detokenize(ids[start:]) and cs.prompts("A")[0][0]["content"] in e.detokenize(ids[:start])
    conv = _conv(None, cs.prompts("A")[0])
    all_ids, segs = chat_segments(e, conv)                                   # when の区間でも書きかけは返事の側
    assert all_ids == ids and segs[-1][2] and segs[-1][1] == len(ids) and segs[-1][0] <= start + 1
    session.direction(contrast=p, name="rp", positions="last")
    sc = session.direction_score("rp", layers=[6])
    assert sc["layers"][0]["A"]["n"] == len(cs.prompts("A"))
    r = e.generate_greedy(e.steer_ctx(), [ids], max_tokens=4)[0]              # 続きから書く
    assert r.prompt_ids == ids and r.token_ids


@pytest.mark.model
def test_refine_adds_heads_from_remaining_questions(session, tmp_path):
    """残りから追加: 組を当てても閾値を超える A 側の質問だけで(両側に組を当てたまま)方向を作り直し、一括で足すヘッドを決め、
    元の組と合わせたレシピを作って測り直す。残りが無ければ断る。同じ組で流し直すと同じ名前で、方向と因果を使い回す。"""
    d = yaml.safe_load(EVIL.read_text(encoding="utf-8"))
    spec = {"name": "rf", "system_a": d["system_a"], "system_b": d["system_b"], "user_prompts": d["user_prompts"],
            "validate": d["validate"], "pipeline": {"n_heads": [2], "max_examples": 8, "top_layers": 1, "hard": 4}}
    session.heads_pipeline(spec)
    hd = session.store.heads_dir(session.meta.model_id)
    ap = hd / "rf.alloc.json"
    al = json.loads(ap.read_text(encoding="utf-8"))
    al["remain_threshold"] = 1e9                                             # 何も残らない
    ap.write_text(json.dumps(al), encoding="utf-8")
    with pytest.raises(GneError):
        session.heads_refine("rf", n_heads=2)
    al["remain_threshold"] = -1e9                                            # 全部残る(仕組みを確かめるため)
    ap.write_text(json.dumps(al), encoding="utf-8")
    out = session.heads_refine("rf", n_heads=2, new_heads=[1])
    rm = out["remaining"]
    assert out["name"] == "rf_r1" and rm["n"] == rm["n_a"] == len(d["user_prompts"]) and not rm["padded"]
    assert out["base_units"] == next(r["units"] for r in al["table"] if r["n_heads"] == 2)
    cs_path = Path(json.loads(Path(out["pipeline"]["path"]).read_text(encoding="utf-8"))["contrast"])
    c = yaml.safe_load(cs_path.read_text(encoding="utf-8"))
    base_specs = {f"{u['layer']}:H{u['head']}" for u in c["recipe_a"]["units"]}
    assert base_specs == {f"{u['layer']}:H{u['head']}" for u in c["recipe_b"]["units"]}          # 両側に元の組
    assert len(c["prompts_a"]) == rm["used"] and c["prompts_b"] == d["user_prompts"] and c.get("validate")
    pd = json.loads(Path(out["pipeline"]["path"]).read_text(encoding="utf-8"))
    assert pd["refine"]["base"] == "rf" and pd["refine"]["n_heads"] == 2
    rows = [r for r in out["pipeline"]["table"] if not r.get("skipped")]
    assert len(out["combined_rows"]) == len(rows)                           # 比較表のどの組も、元の組と合わせる
    base_u = out["base_units"]
    for c, r in zip(out["combined_rows"], rows, strict=True):
        assert c["units"] == base_u + r["units"] and c["n_heads"] == len(base_u) + len(r["units"]) and Path(c["recipe"]).exists()
        rr = Recipe.load(c["recipe"])
        assert {u.direction for u in rr.units} == {"rf", "rf_r1"}                        # 元の方向と、作り直した方向
        assert c["remain_before"] == rm["n"] and c["b_drift_orth_base"] >= 0
    if rows:
        cb = out["combined"]
        assert cb["chosen_by"] in ("recommended", "fewest_remaining") and (hd / "rf_r1.combined.yaml").exists()
        assert pd["refine"]["combined_rows"] == out["combined_rows"]
    again = session.heads_refine("rf", n_heads=2, new_heads=[1])
    assert again["name"] == "rf_r1" and again["pipeline"]["reused"] == ["direction", "causal"]
    from gne.schemas import SCHEMA_VERSION, HeadsRefineOut
    HeadsRefineOut.model_validate({"schema_version": SCHEMA_VERSION, "command": "heads refine", "model_id": "x",
                                   "llama_cpp_version": "x", "seed": 0, "generated_at": "x", **out})


def test_pipeline_target_keywords():
    """目標は数、B(B 側の平均)、mid か省略(A と B の真ん中)。ほかは流す前に断る。"""
    from gne.api import Session
    bs = {"A": 10.0, "B": -6.0}
    assert Session._pipeline_target(None, bs) == (2.0, "A と B の平均の真ん中")
    assert Session._pipeline_target("mid", bs)[0] == 2.0 and Session._pipeline_target("b", bs) == (-6.0, "B 側の平均")
    assert Session._pipeline_target("3.5", bs) == (3.5, "settings") and Session._pipeline_target(-1, bs) == (-1.0, "settings")
    with pytest.raises(GneError):
        Session._pipeline_target("half", bs)


@pytest.mark.model
def test_pipeline_target_b_uses_b_mean(session):
    """target="B" は物差しの最後の層での B 側の平均を目標にし、残りもそれを超える例で数える。方向と因果は使い回す。"""
    d = yaml.safe_load(EVIL.read_text(encoding="utf-8"))
    spec = {"name": "pipe_b", "system_a": d["system_a"], "system_b": d["system_b"], "user_prompts": d["user_prompts"],
            "pipeline": {"n_heads": [2], "max_examples": 8, "top_layers": 1, "hard": 4}}
    mid = session.heads_pipeline(dict(spec))
    out = session.heads_pipeline(dict(spec), target="B")
    S = str(out["causal"]["score_layers"][-1])
    assert out["target"] == pytest.approx(out["causal"]["base_scores"][S]["B"]) and out["target_from"] == "B 側の平均"
    assert out["allocate"]["remain_threshold"] == pytest.approx(out["target"]) and out["reused"] == ["direction", "causal"]
    assert out["target"] < mid["target"]
    with pytest.raises(GneError):
        session.heads_pipeline(dict(spec), target="half")


@pytest.mark.model
def test_export_job_takes_units_and_names_files(session):
    """UI の書き出し(headproj_export のジョブ): レシピのユニットの書き方から LoRA を <root>/export/<名前>.lora.gguf に書き、
    落とせる URL を返す。名前の誤りは断る。"""
    from gne.jobs import _headproj_export
    b = session.direction_set("evil").meta.get("base_layer") or 12
    d = _headproj_export(session, units=[f"{b - 1}:H1:s3@evil", f"{b - 1}:H2:s2@evil"], name="ui_t")
    assert d["file"] == "ui_t.lora.gguf" and d["url"] == "/api/export/file/ui_t.lora.gguf" and Path(d["path"]).exists()
    assert d["proj_layers"][str(b - 1)]["rank"] == 1 and d["bytes"] > 0
    with pytest.raises(GneError):
        _headproj_export(session, units=[f"{b - 1}:H1:s3@evil"], name="a b")
    with pytest.raises(GneError):
        _headproj_export(session, name="x")


@pytest.mark.model
def test_eval_direction_metrics_thinned_counted_and_optional(session):
    """eval の方向の指標: 質問を dir_max_examples に等間隔に間引き(一致率の返事も)、進み具合に数える(終わりで done = total)。
    direction_metrics=False なら測らない。"""
    b = session.direction_set("evil").meta.get("base_layer") or 12
    r = session.recipe(units=[f"{b - 1}:H1:s3@evil", f"{b - 1}:H2:s2@evil"])
    seen = []
    old = session.progress
    session.progress = lambda done, total, stage: seen.append((done, total, stage))
    try:
        d = session.eval(r, doses=(0, 1), dir_max_examples=4, name="dm_t")
    finally:
        session.progress = old
    dm = d["direction_metrics"]
    assert dm["n_a"] == 4 and dm["n_b"] == 4 and 0 < dm["n_behavior"] <= 4 and dm["max_examples"] == 4
    assert d["rows"][1].get("dir_drop_a") is not None and d["rows"][1].get("rate") is not None
    assert seen and seen[-1][0] == seen[-1][1] and any("方向の指標" in s for _, _, s in seen)
    assert all(x[0] <= x[1] for x in seen)
    e = session.eval(r, doses=(0, 1), direction_metrics=False, name="dm_f")
    assert "direction_metrics" not in e and e["rows"][1].get("dir_drop_a") is None


def test_local_path_keys_finds_home_paths(tmp_path):
    """人に渡すファイルの確かめ: メタデータの文字列に手元のパス(ホーム・/home/・/tmp/)があれば、そのキーを返す。語彙は見ない。"""
    import os

    from gguf import GGUFWriter

    from gne.ablate import local_path_keys
    p = tmp_path / "m.gguf"
    w = GGUFWriter(str(p), arch="llama")
    w.add_string("general.name", "model")
    w.add_string("quantize.imatrix.dataset", os.path.expanduser("~") + "/calib.txt")
    w.add_string("general.note", "fine")
    w.add_array("tokenizer.ggml.tokens", ["/home/x", "b"])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    w.close()
    assert local_path_keys(p) == ["quantize.imatrix.dataset"]


@pytest.mark.model
def test_strip_meta_leaves_no_gne_records(session, tmp_path):
    """strip_meta: LoRA にも焼き込みにも gne.* のメタデータを入れず、名前に印を付けず、横の .headproj.json も作らない。
    付けないときは今までどおり記録が入る。"""
    from gguf import GGUFReader
    b = session.direction_set("evil").meta.get("base_layer") or 12
    r = session.recipe(units=[f"{b - 1}:H1:s3@evil"], name="secret_name")

    def keys(p):
        return {f.name: f.contents() for f in GGUFReader(str(p)).fields.values() if not f.name.startswith("tokenizer.")}
    plain = session.headproj_export(r, tmp_path / "a.lora.gguf")
    assert any(k.startswith("gne.") for k in keys(plain["path"]))
    lo = session.headproj_export(r, tmp_path / "b.lora.gguf", strip_meta=True)
    k = keys(lo["path"])
    assert lo["strip_meta"] and not any(x.startswith("gne.") for x in k) and "secret_name" not in str(k.get("general.name"))
    assert lo["local_path_keys"] == []
    bk = session.headproj_export(r, tmp_path / "c.gguf", bake=True, strip_meta=True)
    kb = keys(bk["path"])
    orig = {f.name: f.contents() for f in session.weights.reader.fields.values() if f.name == "general.name"}
    assert not any(x.startswith("gne.") for x in kb) and kb.get("general.name") == orig.get("general.name")
    assert not Path(str(bk["path"]) + ".headproj.json").exists() and bk["local_path_keys"] == []
    bk2 = session.headproj_export(r, tmp_path / "d.gguf", bake=True)
    assert "gne.headproj" in keys(bk2["path"]) and Path(str(bk2["path"]) + ".headproj.json").exists()


def test_find_imatrix_next_to_model(tmp_path):
    """モデルの隣の imatrix を探す: <モデル名>.imatrix.gguf → .dat → -imatrix.* → imatrix.gguf / .dat の順。"""
    from gne.ablate import find_imatrix
    m = tmp_path / "Model-IQ2_XS.gguf"
    m.write_bytes(b"x")
    assert find_imatrix(m) is None
    (tmp_path / "imatrix.dat").write_bytes(b"x")
    assert find_imatrix(m).name == "imatrix.dat"
    (tmp_path / "Model-IQ2_XS.imatrix.dat").write_bytes(b"x")
    assert find_imatrix(m).name == "Model-IQ2_XS.imatrix.dat"
    (tmp_path / "Model-IQ2_XS.imatrix.gguf").write_bytes(b"x")
    assert find_imatrix(m).name == "Model-IQ2_XS.imatrix.gguf"


@pytest.mark.model
def test_bake_uses_imatrix_found_next_to_model(session, tmp_path, monkeypatch):
    """「元の型に戻す」で imatrix を渡さなければ、モデルの隣で見つけたものを llama-quantize に渡す(記録を残さないときは
    結果に名前だけ)。渡せばそちらを使う。"""
    import gne.ablate as ab
    got = {}

    def fake(weights, write_mid, dst, *, llama_quantize=None, imatrix=None, keep_f16=False, neutral_imatrix=False):
        got.update(imatrix=imatrix, neutral=neutral_imatrix)
        Path(dst).write_bytes(b"GGUF")
        return {"path": str(dst), "bytes": 4, "modified": []}
    found = tmp_path / "near.imatrix.gguf"
    found.write_bytes(b"x")
    monkeypatch.setattr(ab, "requant_via_f16", fake)
    monkeypatch.setattr(ab, "find_imatrix", lambda p: found)
    monkeypatch.setattr(ab, "local_path_keys", lambda p: [])
    b = session.direction_set("evil").meta.get("base_layer") or 12
    r = session.recipe(units=[f"{b - 1}:H1:s3@evil"])
    d = session.gain(r, tmp_path / "o.gguf", bake=True, via_f16=True, strip_meta=True)
    assert got == {"imatrix": str(found), "neutral": True} and d["imatrix_auto"] and d["imatrix"] == "near.imatrix.gguf"
    d = session.gain(r, tmp_path / "o2.gguf", bake=True, via_f16=True, imatrix="/given/im.dat")
    assert got["imatrix"] == "/given/im.dat" and not d["imatrix_auto"] and d["imatrix"] == "/given/im.dat"


@pytest.mark.model
def test_export_job_file_name_and_overwrite(session):
    """書き出しのファイル名(拡張子なし)を変えられ、overwrite=False なら同じ名前があるときに書かずに断る。"""
    from gne.jobs import _headproj_export
    b = session.direction_set("evil").meta.get("base_layer") or 12
    u = [f"{b - 1}:H1:s3@evil"]
    d = _headproj_export(session, units=u, name="rcp", file_stem="My-model_v2.lora.gguf", overwrite=False)
    assert d["file"] == "My-model_v2.lora.gguf"
    with pytest.raises(GneError) as ei:
        _headproj_export(session, units=u, name="rcp", file_stem="My-model_v2", overwrite=False)
    assert ei.value.code == "FILE_EXISTS"
    assert _headproj_export(session, units=u, name="rcp", file_stem="My-model_v2", overwrite=True)["file"] == "My-model_v2.lora.gguf"
    with pytest.raises(GneError):
        _headproj_export(session, units=u, name="rcp", file_stem="bad name")


@pytest.mark.model
def test_bake_into_unquantized_copy(session, tmp_path):
    """分析は量子化したモデルで、焼き込みは同じ構造の量子化していない GGUF(BF16)に: 方向はこのモデルのものを使い、書き換える
    行列は焼き込む先のもの。keep なら元の型(BF16)のまま書き、ほかのテンソルはそのまま。構造が違えば断り、LoRA 版との比較もしない。"""
    import subprocess

    from gne.ablate import find_llama_quantize
    from gne.weights import Weights
    lq = find_llama_quantize()
    if lq is None:
        pytest.skip("llama-quantize がありません")
    bf = tmp_path / "m-bf16.gguf"
    p = subprocess.run([lq, "--allow-requantize", str(MODEL), str(bf), "BF16"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-500:]
    b = session.direction_set("evil").meta.get("base_layer") or 12
    L = b - 1
    r = session.recipe(units=[f"{L}:H1:s3@evil"])
    out = tmp_path / "baked.gguf"
    d = session.headproj_export(r, out, bake=True, requant="keep", into=bf)
    assert d["into"] == str(bf) and all(m["from"] == "BF16" and m["to"] == "BF16" for m in d["modified"])
    src, dst = Weights(bf), Weights(out)
    name = d["modified"][0]["tensor"]
    W0 = src.head_writer_matrix(L).astype(np.float64)
    W1 = dst.head_writer_matrix(L).astype(np.float64)
    from gne.headproj import ProjBuilder, proj_heads
    tens = ProjBuilder(src, session.direction_set).tensors(proj_heads(r))
    A, B = tens[name]
    want = W0 + B.astype(np.float64) @ A.astype(np.float64)
    assert np.abs(W1 - want).max() <= 0.02 * np.abs(want).max()                  # BF16 の丸めの範囲
    other = next(t for t in src.reader.tensors if t.name != name and t.name.endswith("attn_output.weight"))
    assert np.array_equal(np.asarray(other.data), np.asarray(next(t for t in dst.reader.tensors if t.name == other.name).data))
    with pytest.raises(GneError):
        session.headproj_export(r, tmp_path / "x.gguf", bake=True, requant="keep", into=bf, verify=True)
    small = ROOT / "models" / "SmolLM2-135M-Instruct-f16.gguf"
    if small.exists():
        with pytest.raises(GneError):
            session.headproj_export(r, tmp_path / "y.gguf", bake=True, into=small)


# --- 実地テスト(gne fieldtest)--------------------------------------------------------------------------

def _fake_chat(messages, max_tokens, seed, tools=None):
    """llama-server の代わり: 質問の中身で振る舞いを変える。"""
    last = messages[-1]
    if last["role"] == "tool":
        return {"content": "done"}, "stop", 3
    q = last["content"]
    if tools:
        if "weather" in q:
            return {"content": "", "tool_calls": [{"id": "c1", "function": {"name": "get_weather", "arguments": '{"city": "Osaka"}'}}]}, "tool_calls", 9
        if "broken" in q:
            return {"content": "", "tool_calls": [{"id": "c2", "function": {"name": "get_weather", "arguments": "{oops"}}]}, "tool_calls", 9
        return {"content": "hello"}, "stop", 2
    if "loop" in q:
        return {"content": "same words again " * 400}, "length", max_tokens
    if "empty" in q:
        return {"reasoning_content": "thinking about many different things " + " ".join(str(i) for i in range(400)), "content": ""}, "length", max_tokens
    if "error" in q:
        return {}, "error", 0
    return {"reasoning_content": "ok", "content": "A fine answer: " + " ".join(str(i) for i in range(300))}, "stop", 50


def test_fieldtest_sections_and_compare(tmp_path):
    """長い作業・ツール・質問集の数え方(ループ・本文が空・上限・名前・引数・結果の後)と、ファイルどうしの比較(2 問ぶんまで同程度)。"""
    from gne import fieldtest as ft
    assert ft.tail_distinct("same words " * 400) < 0.5 and ft.tail_distinct(" ".join(str(i) for i in range(900))) > 0.7
    para = " ".join(f"word{i}" for i in range(60)) + " "
    assert ft.long_repeat(para * 30) > 0.5 and ft.long_repeat(" ".join(str(i) for i in range(2000))) == 0.0
    ticks = []
    m, recs = ft.run_long(_fake_chat, ["plain", "a loop please", "empty please", "error please"], 64, 2, ticks.append)
    assert m["n"] == 4 and m["loop"] == 0.25 and m["empty"] == 0.5 and m["error"] == 0.25 and m["hit_limit"] == 0.5 and sum(ticks) == 4
    assert recs[1]["loop"] and not recs[0]["loop"] and recs[0]["content"].startswith("A fine")
    spec = {"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"required": ["city"]}}}],
            "tasks": [{"prompt": "weather in Osaka", "expect": "get_weather"}, {"prompt": "broken call", "expect": "get_weather"},
                      {"prompt": "no call here", "expect": "get_weather"}, {"prompt": "just chat", "expect": None}], "results": {"get_weather": {"t": 1}}}
    m, _ = ft.run_tools(_fake_chat, spec, 64, 2, lambda n: None)
    assert m["n_expect"] == 3 and abs(m["called"] - 2 / 3) < 1e-9 and abs(m["name_ok"] - 2 / 3) < 1e-9 and abs(m["args_ok"] - 1 / 3) < 1e-9
    assert m["no_tool_ok"] == 1.0 and m["followup_ok"] == 1.0 and m["n_followup"] == 2
    m, recs = ft.run_prompts(_fake_chat, ["plain", "a loop", "empty one", "plain 2"], 64, 2, lambda n: None, match="fine answer")
    assert m["match"] == 0.5 and m["loop"] == 0.25 and m["empty"] == 0.25
    assert ft.run_prompts(_fake_chat, ["plain"], 64, 1, lambda n: None)[0]["match"] is None
    # 比較: 2 問ぶんまでは同程度、それより悪ければ worse。perplexity は 1% まで
    base = {"long": {"n": 24, "loop": 0.0, "empty": 0.0, "tail_distinct": 0.98, "tokens": 900}, "tools": {"n": 40, "n_expect": 30, "called": 1.0},
            "prompts": {"h": {"n": 100, "loop": 0.0, "empty": 0.02}}, "ppl": {"en": 6.0, "mean": 6.0}}
    same = {"long": {"n": 24, "loop": 2 / 24, "empty": 0.0, "tail_distinct": 0.95, "tokens": 2000}, "tools": {"n": 40, "n_expect": 30, "called": 28 / 30},
            "prompts": {"h": {"n": 100, "loop": 0.02, "empty": 0.03}}, "ppl": {"en": 6.05, "mean": 6.05}}
    assert not any(r["worse"] for r in ft.compare(base, same))
    bad = {**same, "long": {**same["long"], "loop": 3 / 24}, "tools": {"n": 40, "n_expect": 30, "called": 27 / 30}, "ppl": {"en": 6.2, "mean": 6.2}}
    worse = {(r["section"], r["metric"]) for r in ft.compare(base, bad) if r["worse"]}
    assert worse == {("long", "loop"), ("tools", "called"), ("ppl", "en"), ("ppl", "mean")}


def test_fieldtest_inputs_and_shipped_tasks(tmp_path):
    """同梱の課題が読める。質問集は 名前=ファイル か表で、履歴も書ける。崩れた入力は断る。llama-server が無ければ分かるように断る。"""
    from gne import fieldtest as ft
    assert len(ft.load_long_tasks()) == 24
    t = ft.load_tool_tasks()
    assert len(t["tasks"]) == 40 and set(t["results"]) == {x["function"]["name"] for x in t["tools"]}
    p = tmp_path / "q.yaml"
    p.write_text(yaml.safe_dump(["Hello", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]]), encoding="utf-8")
    assert len(ft.parse_prompt_sets([f"mine={p}"])["mine"]) == 2 and ft.parse_prompt_sets({"x": ["q"]}) == {"x": ["q"]}
    for bad in (["nofile"], {"a b": ["q"]}, {"x": []}, {"x": [{"role": "user"}]}):
        with pytest.raises(GneError):
            ft.parse_prompt_sets(bad)
    (tmp_path / "bad.json").write_text('{"tools": [{"type":"function","function":{"name":"a","parameters":{}}}], "tasks": [{"prompt":"p","expect":"zzz"}]}')
    with pytest.raises(GneError):
        ft.load_tool_tasks(tmp_path / "bad.json")
    assert ft.find_llama_server(str(tmp_path / "none")) is None


@pytest.mark.model
def test_fieldtest_end_to_end_with_llama_server(session, tmp_path, monkeypatch):
    """本物の llama-server で通す(GNE_LLAMA_SERVER か build か PATH にあるとき): 2 ファイルを比べ、結果と文章を別のファイルに書く。
    走った後は読み込んでいたモデルに戻れ、llama-server は残らない。"""
    import subprocess

    from gne import fieldtest as ft
    from gne.schemas import SCHEMA_VERSION, FieldtestOut
    if ft.find_llama_server() is None:
        pytest.skip("llama-server がありません")
    (tmp_path / "long.json").write_text(json.dumps(["Say hello.", "Name a color."]))
    tools = json.loads(ft.TOOL_TASKS.read_text(encoding="utf-8"))
    tools["tasks"] = [{"prompt": "What's the weather like in Osaka right now?", "expect": "get_weather"}, {"prompt": "What is 2 + 2?", "expect": None}]
    (tmp_path / "tools.json").write_text(json.dumps(tools))
    (tmp_path / "q.json").write_text(json.dumps(["Hello", "Who are you?"]))
    monkeypatch.setattr(ft, "load_bench", lambda b, cache, n=None: [{"id": "1", "prompt": "What is 2 + 2? End with: Answer: <number>", "gold": "4"},
                                                                    {"id": "2", "prompt": "What is 3 + 3? End with: Answer: <number>", "gold": "6"}])
    d = session.fieldtest(against=MODEL, name="ft_t", long_tasks=tmp_path / "long.json", tool_tasks=tmp_path / "tools.json",
                          prompts=[f"mine={tmp_path / 'q.json'}"], match="hello", long_tokens=24, tool_tokens=24, prompt_tokens=24, n_parallel=2,
                          bench="gsm8k", bench_tokens=24)
    assert [r["label"] for r in d["runs"]] == ["reference", "target"] and d["compare"] and d["same_level"] is not None
    run = d["runs"][1]
    assert run["long"]["n"] == 2 and run["tools"]["n"] == 2 and run["prompts"]["mine"]["n"] == 2 and run["ppl"]["mean"] > 1
    assert run["bench"]["gsm8k"]["n"] == 2 and run["bench_items"]["gsm8k"]["ids"] == ["1", "2"] and d["settings"]["bench"] == {"gsm8k": 2}
    assert any(r["section"] == "bench:gsm8k" and r["metric"] == "accuracy" and "only_target" in r for r in d["compare"])
    # 同じファイル・同じ種なら perplexity は同じ
    ppl_row = next(r for r in d["compare"] if r["section"] == "ppl" and r["metric"] == "mean")
    assert abs(ppl_row["delta"]) < 1e-6 and not ppl_row["worse"]
    saved = json.loads(Path(d["path"]).read_text(encoding="utf-8"))
    assert saved["name"] == "ft_t" and "reasoning" not in json.dumps(saved["runs"]) and "content" not in json.dumps(saved["runs"][1]["long"])
    assert "reference" in json.loads(Path(d["replies_path"]).read_text(encoding="utf-8"))
    FieldtestOut.model_validate({"schema_version": SCHEMA_VERSION, "command": "fieldtest", "model_id": "x", "llama_cpp_version": "x",
                                 "seed": 0, "generated_at": "x", **d})
    assert session.no_yield is False and session.engine is not None                    # モデルは読み直せる
    me = str(os.getpid())
    kids = subprocess.run(["ps", "--ppid", me, "-o", "comm="], capture_output=True, text=True).stdout
    assert "llama-server" not in kids


def test_llama_server_restarts_after_a_crash(tmp_path):
    """測定の途中で llama-server が落ちても立て直して同じ問いをやり直す。落ち続けたら無限に立て直さず、エラーとして数えて結果の頭に書く。"""
    import stat
    import sys
    from concurrent.futures import ThreadPoolExecutor

    from gne import fieldtest as ft
    fake = tmp_path / "fake-server"
    state = tmp_path / "crash-left"
    fake.write_text(f"""#!{sys.executable}
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, o):
        b = json.dumps(o).encode(); self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._send({{"status": "ok"}})
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        left = int(open({str(state)!r}).read())
        if left != 0:
            open({str(state)!r}, "w").write(str(left - 1))
            os._exit(1)
        self._send({{"choices": [{{"message": {{"content": "ok"}}, "finish_reason": "stop"}}], "usage": {{"completion_tokens": 3}}}})
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
""", encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    srv = ft.LlamaServer(str(fake), "m.gguf", lora=None, n_parallel=2, n_ctx_slot=128, ngl=0, template_kwargs=None, log=tmp_path / "s.log")
    state.write_text("2")
    with srv:
        chat = srv.chat_fn({})
        with ThreadPoolExecutor(4) as ex:                                              # 同時に落ちても立て直しは必要な分だけ
            res = list(ex.map(lambda i: chat([{"role": "user", "content": "q"}], 8, i), range(8)))
        assert [r[1] for r in res] == ["stop"] * 8 and srv.restarts == 2
        state.write_text("-1")                                                         # 落ち続ける
        assert chat([{"role": "user", "content": "q"}], 8, 0) == ({}, "error", 0)
        assert srv.restarts == ft.LlamaServer.MAX_RESTARTS
        assert srv.p.poll() is not None
    w = ft._run_warnings([{"label": "target", "server_restarts": 3, "prompts": {"a": {"error": 0.25}}, "bench": {}, "long": {}}])
    assert len(w) == 2 and "3 回" in w[0] and "25%" in w[1]
    assert ft._run_warnings([{"label": "t", "prompts": {"a": {"error": 0.0}}, "long": {"error": 0}}]) == []


def test_fieldtest_bench_math500_and_ifeval(tmp_path, monkeypatch):
    """MATH-500 は最後の boxed を math-verify で(スレッドからでも別プロセスで)、IFEval は公式の採点器で指示を全部守ったときだけ正解。"""
    from gne import fieldtest as ft
    assert ft.last_boxed(r"a \\boxed{1} b \\boxed{\\frac{1}{2}} c") == r"\\frac{1}{2}" and ft.last_boxed(r"\\boxed{open") is None and ft.last_boxed("x") is None
    it = ft.bench_item("math500", 3, {"problem": "P", "answer": " 7 "})
    assert it["gold"] == "7" and it["prompt"].endswith("within \\boxed{}.")
    row = {"prompt": "Q", "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{"a": None, "b": 1}]}
    assert ft.bench_item("ifeval", 4, row)["kwargs"] == [{"b": 1}]                     # 表の形の None は落とす
    assert set(ft.BENCHES) == {"gsm8k", "mmlu_pro", "math500", "ifeval"}
    pytest.importorskip("math_verify")
    pytest.importorskip("langdetect")
    pytest.importorskip("nltk")
    pytest.importorskip("immutabledict")
    ft.bench_ready(["math500", "ifeval"])
    assert ft.grade_math(["\\left( 3, \\frac{\\pi}{2} \\right)", "12", "5"], ["(3,\\pi/2)", "13", None]) == ([True, False, False], 0)
    # 別プロセスが落ちても、長い生成の結果は失わない(採点できなかった数を数える)
    real = ft._grade_math_rows
    monkeypatch.setattr(ft, "_grade_math_rows", lambda rows, t: (_ for _ in ()).throw(GneError("GRADE_FAILED", "x")))
    assert ft.grade_math(["1", "2"], ["1", "2"]) == ([False, False], 2)
    monkeypatch.setattr(ft, "_grade_math_rows", real)
    item = {"prompt": "About cats.", "instructions": ["punctuation:no_comma", "length_constraints:number_words"],
            "kwargs": [{}, {"relation": "at least", "num_words": 5}]}
    assert ft.grade_ifeval(item, "Cats sleep a lot during the day.") == ([True, True], 0)
    assert ft.grade_ifeval(item, "Cats, dogs.") == ([False, False], 0)
    assert ft.grade_ifeval(item, "") == ([False, False], 0)
    def chat(msgs, mt, seed):
        return {"reasoning_content": "t", "content": "So \\boxed{7}."}, "stop", 5
    items = [{"id": "1", "prompt": "P", "gold": "7"}, {"id": "2", "prompt": "P", "gold": "8"}]
    m, its, recs = ft.run_bench(chat, "math500", items, 32, 2, lambda n: None)
    assert m["accuracy"] == 0.5 and its["correct"] == [True, False] and recs[0]["answer"] == "7"
    def chat2(msgs, mt, seed):
        return {"reasoning_content": "t", "content": "Cats sleep a lot during the day."}, "stop", 5
    m, its, recs = ft.run_bench(chat2, "ifeval", [{**item, "id": "1", "gold": ""}], 32, 1, lambda n: None)
    assert m["accuracy"] == 1.0 and recs[0]["answer"] == "2/2"
    def chat3(msgs, mt, seed):                                                         # 思考で切れて本文が無い
        return {"reasoning_content": "loop", "content": ""}, "length", 32
    m, its, recs = ft.run_bench(chat3, "ifeval", [{**item, "id": "1", "gold": ""}], 32, 1, lambda n: None)
    assert m["accuracy"] == 0.0 and m["hit_limit"] == 1.0 and m["no_answer"] == 1.0
    m, its, recs = ft.run_bench(chat3, "math500", items[:1], 32, 1, lambda n: None)
    assert m["accuracy"] == 0.0 and m["no_answer"] == 1.0


def test_fieldtest_bench_extract_score_and_compare(tmp_path):
    """ベンチマーク: 答えの取り出し(Answer: の最後、boxed、無ければ不正解)、問題の選び方(種を固定、写しを使う)、片方だけ正解での比較。"""
    from gne import fieldtest as ft
    for text, want in (("Work... Answer: 1,234", "1,234"), ("first Answer: 3\nthen **Answer: $42**", "42"), ("so \\boxed{7.5}", "7.5"),
                       ("Answer: **-3**", "-3"), ("no final line 12", None), ("", None)):
        assert ft.extract_answer("gsm8k", text) == want, text
    for text, want in (("reasoning\nAnswer: C", "C"), ("Answer: (B)", "B"), ("**Answer: J**", "J"), ("\\boxed{\\text{D}}", "D"),
                       ("The answer: a good one", None), ("Answer: Absolutely", None)):
        assert ft.extract_answer("mmlu_pro", text) == want, text
    assert ft.is_correct("gsm8k", "1,234", "1234") and ft.is_correct("gsm8k", "42.0", "42") and not ft.is_correct("gsm8k", None, "1")
    assert ft.is_correct("mmlu_pro", "C", "C") and not ft.is_correct("mmlu_pro", "c", "C")
    assert ft.bench_item("gsm8k", 3, {"question": "Q?", "answer": "steps\n#### 1,000"})["gold"] == "1000"
    it = ft.bench_item("mmlu_pro", 5, {"question": "Q?", "options": ["x", "y"], "answer": "B"})
    assert it["gold"] == "B" and "A) x\nB) y" in it["prompt"] and it["id"] == "5"
    calls = []

    def fetch(ds, cfg, split, offset, length):
        calls.append(offset)
        return {"num_rows_total": 1000, "rows": [{"row_idx": i, "row": {"question": f"q{i}", "answer": f"#### {i}"}}
                                                 for i in range(offset, min(1000, offset + length))]}
    a = ft.load_bench("gsm8k", tmp_path, 40, fetch=fetch, pace=0)
    assert len(a) == 40 and len({x["id"] for x in a}) == 40 and all(x["gold"] == x["id"] for x in a)
    n_calls = len(calls)
    assert ft.load_bench("gsm8k", tmp_path, 40, fetch=fetch, pace=0) == a and len(calls) == n_calls        # 2 回目は写しを使う
    assert [x["id"] for x in ft.load_bench("gsm8k", tmp_path / "other", 40, fetch=fetch, pace=0)] == [x["id"] for x in a]   # 同じ種なら同じ問題
    with pytest.raises(GneError):
        ft.load_bench("nope", tmp_path)

    def chat(messages, max_tokens, seed, tools=None):
        q = messages[-1]["content"]
        if "cut" in q:
            return {"reasoning_content": "long"}, "length", max_tokens
        return {"content": "Answer: 5" if "right" in q else "Answer: 6"}, "stop", 10
    items = [{"id": "1", "prompt": "right", "gold": "5"}, {"id": "2", "prompt": "wrong", "gold": "5"}, {"id": "3", "prompt": "cut", "gold": "5"}]
    m, its, recs = ft.run_bench(chat, "gsm8k", items, 32, 2, lambda n: None)
    assert m["n"] == 3 and abs(m["accuracy"] - 1 / 3) < 1e-9 and abs(m["no_answer"] - 1 / 3) < 1e-9 and abs(m["hit_limit"] - 1 / 3) < 1e-9
    assert its == {"ids": ["1", "2", "3"], "correct": [True, False, False]} and "content" in recs[0] and "content" not in its
    assert ft.mcnemar_p(0, 0) == 1.0 and ft.mcnemar_p(5, 5) == 1.0 and ft.mcnemar_p(0, 6) < 0.05 and ft.mcnemar_p(1, 4) > 0.05

    def run(correct):
        ids = [str(i) for i in range(len(correct))]
        return {"bench": {"b": {"n": len(ids), "accuracy": sum(correct) / len(ids), "no_answer": 0.0, "tokens": 9}}, "bench_items": {"b": {"ids": ids, "correct": correct}},
                "long": {}, "tools": {}, "prompts": {}, "ppl": {}}
    ref = run([True] * 100)
    r = next(x for x in ft.compare(ref, run([True] * 97 + [False] * 3)) if x["metric"] == "accuracy")
    assert r["only_reference"] == 3 and r["only_target"] == 0 and not r["worse"]              # 3 問では揺れの範囲
    r = next(x for x in ft.compare(ref, run([True] * 90 + [False] * 10)) if x["metric"] == "accuracy")
    assert r["worse"] and r["p"] < 0.05
