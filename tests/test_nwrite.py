"""方向の新しい作り方(同じ返事を両方の条件で読ませた差・返事の位置で分けた方向)と、計算で作るニューロン(mode nw)のテスト。

モデルなし: 区切りの書き方、返事の位置の区切りの positions(pos:<範囲>)、nw のユニットの書き方とレシピの保存・読み込み、
足す量の式(鍵 1 つ・鍵 2 つ・上限つき)。
モデルあり(Qwen3-0.6B、GNE_TEST_MODEL。CPU で回すときは GNE_TEST_NGL=0): 方向(同じ返事を読ませていること、区切りつき、
今までの形との互換)、MLP の入力の空間の鍵(位置・場面・ほぼ 1 を返す鍵)、ふだんの働きの表、nw の LoRA が式どおりに動くこと
(ほかのニューロンが変わらないこと)、位置ごとの足した量、LoRA の書き出しと読み直し、焼き込み・対応しない形のエラー、
CLI・ジョブ・MCP。
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from gne import nwrite as NW
from gne.direction import DirectionSet, parse_segments, segment_of, segment_text
from gne.keyed_build import parse_positions, select_positions, start_positions
from gne.steer import GneError, Recipe, parse_unit

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
NGL = int(os.environ.get("GNE_TEST_NGL", "-1"))

CONTRAST = {
    "name": "nw_scene",
    "system_a": "You are a pirate. Always answer in rough pirate speech.",
    "user_prompts": ["Thank you for your help yesterday.", "I really like your drawing.", "Thanks for lending me your notebook.",
                     "Your cooking was delicious.", "Thank you for listening to me.", "You look great today."],
    "dont_touch": [{"name": "tasks", "prompts": ["What is 12 times 8?", "What is the capital of France?",
                                                 "How many days are in a week?", "What color is the sky?"]}],
}


class FakeEngine:
    def is_eog(self, t):
        return t == 99

    def token_bytes(self, t):
        return b"x"


# --- モデルなし ----------------------------------------------------------------------------------------

def test_parse_segments():
    segs = parse_segments("0,1-3,4-")
    assert segs == [(0, 1), (1, 4), (4, None)]
    assert [segment_text(s) for s in segs] == ["0", "1-3", "4-"]
    assert [segment_of(segs, k) for k in (0, 1, 3, 4, 100)] == [0, 1, 1, 2, 2]
    assert segment_of(parse_segments("1-2"), 0) is None
    for bad in ("", "a", "3-1", "0,0-2", "2-,5"):
        with pytest.raises(GneError):
            parse_segments(bad)


def test_position_ranges():
    p = parse_positions("pos:1-3")
    assert (p.kind, p.lo, p.hi, p.needs_reply) == ("pos", 1, 4, True)
    assert parse_positions("pos:4-").hi is None and parse_positions("not-pos:0").kind == "not-pos"
    e, reply, P = FakeEngine(), [1, 2, 3, 4, 5, 99], 10
    # 位置 k = 返事の k 番目のトークンを決める位置(列の番号 P − 1 + k)。返事 5 トークンなら位置 0〜5
    assert select_positions(e, reply, "pos:0", n_prompt=P).tolist() == [9]
    assert select_positions(e, reply, "pos:1-3", n_prompt=P).tolist() == [10, 11, 12]
    assert select_positions(e, reply, "pos:4-", n_prompt=P).tolist() == [13, 14]
    assert select_positions(e, reply, "not-pos:1-3", n_prompt=P).tolist() == [9, 13, 14]
    assert start_positions(e, reply, "pos:1-3", n_prompt=P).tolist() == [10, 11]
    assert start_positions(e, reply, "not-pos:1-3", n_prompt=P).tolist() == [9]
    with pytest.raises(GneError):
        parse_positions("pos:3-1")


def test_unit_syntax_and_recipe_roundtrip(tmp_path):
    one = parse_unit("30:N8046:nw:a4@noda#noda_in")
    assert (one.mode, one.layer, one.neuron, one.k, one.direction, one.key, one.key2, one.pair) == ("nw", 30, 8046, 4.0, "noda", "noda_in", None, None)
    two = parse_unit("11:N2045:nw:a1@tsun.seg0#scene&pos0!s8")
    assert (two.key, two.key2, two.s) == ("scene", "pos0", 8.0) and NW.form_of(two) == {"form": "two", "s": 8.0}
    cap = parse_unit("11:N2045+N4648:nw:a0.9@-tsun.seg1#conj~2&one!cap0.3-0.7")
    assert (cap.pair, cap.k, cap.direction, cap.key, cap.cap) == (4648, -0.9, "tsun.seg1", "conj~2", "0.3-0.7")
    assert NW.form_of(cap) == {"form": "cap", "s": 16.0, "a": 0.3, "b": 0.7}
    assert NW.form_of(parse_unit("1:N2+N3:nw:a1@d#k&one")) == {"form": "cap", "s": 16.0, "a": 0.4, "b": 0.8}     # 既定
    for u in (one, two, cap):
        assert parse_unit(NW.spec_of(u)) == u
    r = Recipe([one, two, cap, parse_unit("14:8949:+1.0")], name="t")
    r.validate()
    assert [u.label() for u in r.nw_units][1] == "L11 N2045 nw a1 @tsun.seg0 #scene&pos0"
    r.save(tmp_path / "t.yaml")
    back = Recipe.load(tmp_path / "t.yaml")
    assert back.units == r.units and yaml.safe_load((tmp_path / "t.yaml").read_text())["units"][0]["alpha"] == 4.0
    assert r.scaled(0.5).units[0].k == 2.0 and r.scaled(0.5).units[2].k == -0.45
    # nw 以外のユニットの記録は前と同じ(新しい項目が入らない)
    assert "key2" not in r.to_dict()["units"][3] and r.to_dict()["units"][1]["key2"] == "pos0"
    for bad in ("11:N5:nw:b1", "11:5:nw:a1@d#k", "11:N5:nw:a1@d#k!cap0.8-0.4"):
        with pytest.raises(GneError):
            Recipe([parse_unit(bad)]).validate()
    with pytest.raises(GneError):                                   # 上限つきの形には 2 本目の鍵が要る
        Recipe([parse_unit("11:N5+N6:nw:a1@d#k")]).validate()
    with pytest.raises(GneError):                                   # 鍵が無い
        Recipe([parse_unit("11:N5:nw:a1@d")]).validate()


def test_amount_formulas():
    assert NW.amount(1.0, 1.0, 4.0) == pytest.approx(1.0)
    assert NW.amount(0.5, 0.5, 4.0) < 0.25 and NW.amount(-0.5, -0.5, 4.0) < 0.05          # 小さい反応は潰れ、負ではほぼ書かない
    assert NW.amount(2.0, 2.0, 4.0) > 3.9                                                 # 反応が 1 を超えると 2 乗で増える
    assert NW.amount(1.0, 0.0, 4.0) == 0.0 and NW.amount(-1.0, 1.0, 4.0) == pytest.approx(-0.0183, abs=2e-3)   # 鍵 2 つ: 片方が閉じれば書かない
    rho = np.array([-1.0, 0.0, 0.2, 0.6, 0.8, 1.0, 2.0, 5.0])
    c = NW.amount_cap(rho)                                                                # 既定 s 16・0.4-0.8
    assert abs(c[0]) < 1e-3 and abs(c[1]) < 5e-3 and c[3] == pytest.approx(0.5, abs=0.01)
    assert np.all(c[4:] < 1.05) and c[-1] == pytest.approx(1.0, abs=1e-6)                 # 頭打ち
    assert NW.amount_cap(0.0, s=8, a=0.25, b=0.75) < -0.05                                # 緩い設定は反応 0 の位置で逆向きに書く


def test_direction_set_without_segments_is_unchanged(tmp_path):
    p = tmp_path / "old.npz"
    np.savez(p, layers=np.array([1, 2], dtype=np.int32), layer_1=np.ones(4, np.float32), layer_2=np.ones(4, np.float32),
             scale_1=np.ones(4, np.float32), scale_2=np.ones(4, np.float32), method="mean_diff", positions="response")
    ds = DirectionSet.load(p)
    assert ds.segments == [] and ds.seg == {} and "same_reply" not in ds.meta
    with pytest.raises(GneError):
        ds.segment(0)


def test_help_lists_new_commands():
    from gne.cli import app
    r = CliRunner().invoke(app, ["nw", "--help"])
    assert r.exit_code == 0 and all(x in r.output for x in ("neurons", "key", "verify", "check", "export"))
    r = CliRunner().invoke(app, ["direction", "--help"])
    assert "--same-reply" in r.output and "--segments" in r.output and "--stack-layers" in r.output


# --- モデルあり ----------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """小さいモデルで、方向・鍵・ニューロンの表・レシピを 1 回だけ作る。"""
    if not MODEL.exists():
        pytest.skip(f"{MODEL} がありません")
    from gne.api import Session
    root = tmp_path_factory.mktemp("nw")
    c = root / "c.yaml"
    c.write_text(yaml.safe_dump(CONTRAST, allow_unicode=True), encoding="utf-8")
    s = Session(MODEL, root=root, n_gpu_layers=NGL)
    out = {"s": s, "root": root, "contrast": c}
    out["dir"] = s.direction(contrast=c, positions="response", response_tokens=20, same_reply=True, segments="0,1-3,4-",
                             stack_layers="8,12")
    out["dir_old"] = s.direction(contrast=c, name="nw_old", positions="response", response_tokens=20)
    out["k_pos"] = s.direction_key(c, target="mlpin", layers=[8, 12], positions="pos:1-3", name="k_pos", subset="all", max_tokens=20, pseudo=3)
    out["k_scene"] = s.direction_key(c, role="scene", layers=[8, 12], name="k_scene", subset="all", max_tokens=20, pseudo=3)
    out["k_one"] = s.direction_key(c, role="one", layers=[8, 12], name="k_one", subset="all", max_tokens=20)
    out["usage"] = s.nw_neurons([8, 12], max_tokens=12, corpus_tokens=512, top=5)
    n8 = [x["neuron"] for x in out["usage"]["layers"][0]["rows"]]
    n12 = [x["neuron"] for x in out["usage"]["layers"][1]["rows"]]
    out["units"] = [f"8:N{n8[0]}:nw:a1@nw_scene.seg0#k_scene", f"8:N{n8[1]}:nw:a1@nw_scene.seg1#k_scene&k_pos",
                    f"12:N{n12[0]}+N{n12[1]}:nw:a1@nw_scene.seg1#k_pos&k_one"]
    out["n8"], out["n12"] = n8, n12
    yield out
    s.close()


@pytest.mark.model
def test_direction_same_reply_and_segments(built):
    s, d = built["s"], built["dir"]
    assert d["method"] == "mean_diff_same_reply" and d["same_reply"] is True and d["stack_layers"] == [8, 12]
    assert [x["positions"] for x in d["segments"]] == ["0", "1-3", "4-"] and d["segments"][1]["name"] == "nw_scene.seg1"
    # 同じ返事を読ませているので、両側の位置の数が同じ(区切りごと・全位置とも)
    assert all(x["rows"]["A"] == x["rows"]["B"] for x in d["segments"]) and d["n_rows"]["A"] == d["n_rows"]["B"]
    assert d["segments"][0]["rows"]["A"] == len(CONTRAST["user_prompts"])
    ds = s.direction_set("nw_scene")
    assert ds.segments == ["0", "1-3", "4-"] and ds.meta["same_reply"] is True and ds.meta["method"] == "mean_diff_same_reply"
    assert set(ds.seg[0]) == set(ds.raw) and ds.raw[8].shape == ds.seg[1][8].shape
    seg1 = s.direction_set("nw_scene.seg1")
    assert seg1.name == "nw_scene.seg1" and np.allclose(seg1.raw[8], ds.seg[1][8]) and seg1.meta["segment"] == "1-3"
    with pytest.raises(GneError):
        s.direction_set("nw_scene.seg9")
    # 今までの作り方は記録が mean_diff で、区切りは無い。全位置の平均(layer_L)はどちらもそのまま使える
    old = s.direction_set("nw_old")
    assert built["dir_old"]["method"] == "mean_diff" and built["dir_old"]["same_reply"] is False and old.segments == []
    assert built["dir_old"]["n_rows"]["A"] != 0 and old.raw[8].shape == ds.raw[8].shape
    r = s.recipe(units=["8:dir:nw_scene:0.2", "8:dir:nw_scene.seg2:0.2"])            # add のユニットでどちらも使える
    assert len(s.vectors(r)[0]) == 1
    # 下の層に足した後の残り: いちばん下の層は平均差そのまま、上の層は平均差より小さくなる(下の層の分が伝わるので)
    raw = s.direction(contrast=built["contrast"], name="nw_raw", positions="response", response_tokens=20, same_reply=True, segments="0,1-3,4-")
    rs = s.direction_set("nw_raw")
    assert np.allclose(rs.seg[1][8], ds.seg[1][8], atol=1e-4) and not np.allclose(rs.seg[1][12], ds.seg[1][12], atol=1e-3)
    assert np.linalg.norm(ds.seg[1][12]) < np.linalg.norm(rs.seg[1][12]) and raw["stack_layers"] == []
    for kw in ({"positions": "last", "same_reply": True}, {"positions": "response", "stack_layers": "8,12"},
               {"positions": "response", "segments": "0,1-", "same_reply": True, "stack_layers": "0,8"}):
        with pytest.raises(GneError):
            s.direction(contrast=built["contrast"], name="nw_bad", **kw)


@pytest.mark.model
def test_mlpin_keys(built):
    s = built["s"]
    from gne.keyed import load_key
    kp, ksc, k1 = load_key(s, "k_pos"), load_key(s, "k_scene"), load_key(s, "k_one")
    n_embd = s.meta.n_embd
    for ks in (kp, ksc, k1):
        assert ks.target == "mlpin" and ks.layers == [8, 12] and ks.kappa[8].size == n_embd
    assert built["k_pos"]["mode"] == "same" and built["k_pos"]["positions_b"] == "not-pos:1-3" and kp.n_pseudo == 3
    assert float(kp.kappa[8] @ kp.mu[8]) == pytest.approx(1.0, abs=1e-4)
    dts = built["k_pos"]["dont_touch"]                               # 位置の鍵: ほかの場面は返事の全部の位置が書き込みたくない位置
    assert dts and all(d["positions"] == "pos:0-" for d in dts)
    assert ksc.info["role"] == "scene" and ksc.n_pseudo == 3 and float(ksc.kappa[8] @ ksc.mu[8]) == pytest.approx(1.0, abs=1e-4)
    for r in built["k_scene"]["layers"]:                             # 場面の鍵は、作成に使った質問で比較用の鍵より分ける
        assert r["build_selectivity"] > r["pseudo_selectivity_p95"] and r["build_selectivity"] > 0.5
    for r in built["k_one"]["layers"]:                               # ほぼ 1 を返す鍵
        assert r["response_mean"] == pytest.approx(1.0, abs=0.02) and r["response_sd"] < 0.1
    with pytest.raises(GneError):                                    # 場面の鍵には dont_touch が要る
        c2 = built["root"] / "c2.yaml"
        c2.write_text(yaml.safe_dump({k: v for k, v in CONTRAST.items() if k != "dont_touch"}), encoding="utf-8")
        s.direction_key(c2, role="scene", layers=[8], subset="all", max_tokens=8)
    with pytest.raises(GneError) as ei:                              # MLP の入力の空間の鍵は keyed には使えない
        s.recipe(units=["8:ffn:keyed:a1@nw_scene#k_pos"])
    assert ei.value.code == "BAD_RECIPE"


@pytest.mark.model
def test_neuron_usage_table(built):
    u = built["usage"]
    assert [r["layer"] for r in u["layers"]] == [8, 12] and u["sort"] == "sum"
    for r in u["layers"]:
        assert len(r["rows"]) == 5 and r["n_ff"] == built["s"].meta.n_ff
        tot = [x["rank_max"] + x["rank_mean"] for x in r["rows"]]
        assert tot == sorted(tot) and all(x["mean"] <= x["max"] for x in r["rows"])
        assert r["min"]["max"] > 0                                    # まったく働いていないニューロンは無い
    by_max = built["s"].nw_neurons([8], max_tokens=8, corpus_tokens=256, top=3, sort="max", prompts=["Hello."])["layers"][0]["rows"]
    assert [x["max"] for x in by_max] == sorted(x["max"] for x in by_max)


@pytest.mark.model
def test_weights_follow_the_formula(built):
    s = built["s"]
    r = s.recipe(units=built["units"], name="nw_t")
    assert [u.mode for u in r.units] == ["nw"] * 3
    v = s.nw_verify(r)
    assert v["ok"], v
    for x in v["layers"]:
        assert x["others_changed_max"] == 0.0 and x["act_err"] < 1e-5 and x["out_rel_err_median"] < 0.05
    # LoRA の中身: 層ごとに ffn_gate・ffn_up・ffn_down。差し替えた後の行・列が式どおり(float に戻した重みで)
    rows = NW.nw_plan(s.weights, r, s._key_of, s.direction_set)
    tens = NW.nw_tensors(s.weights, rows)
    assert sorted(tens) == sorted(f"blk.{L}.ffn_{p}.weight" for L in (8, 12) for p in ("gate", "up", "down"))
    one = rows[0]
    j, gate, up, down, g = one["parts"][0]
    A, B = tens["blk.8.ffn_gate.weight"]
    assert A.shape == (2, s.meta.n_embd) and B.shape == (s.meta.n_ff, 2) and B[j, 0] == 1.0 and B.sum() == 2.0
    Wg = NW._gate_matrix(s.weights, 8)
    assert np.allclose(Wg[j] + A[0], 4.0 * one["kappa_gate"], atol=1e-5)                       # gate ← s·κ
    Ad, Bd = tens["blk.8.ffn_down.weight"]
    assert np.allclose(s.weights.down_matrix(8)[:, j] + Bd[:, 0], one["d"] / float(NW.silu(4.0)), atol=1e-4)   # down ← α·d ÷ silu(s)
    capr = rows[2]
    (j1, g1, u1, d1, c1), (j2, g2, u2, d2, c2) = capr["parts"]
    assert np.allclose(d1, -d2) and c1 == -c2 == pytest.approx(1.0 / (16.0 * 0.4)) and np.allclose(u1, capr["kappa_up"])
    assert np.allclose(g1 - g2, 16.0 * 0.4 * capr["kappa_up"])                                  # ゲートの差 = s(b − a)·κ1
    with pytest.raises(GneError):                                                              # 同じニューロンを 2 回は使えない
        s.recipe(units=[built["units"][0], built["units"][0].replace("seg0", "seg1")])
    with pytest.raises(GneError):
        s.recipe(units=[f"8:N{s.meta.n_ff}:nw:a1@nw_scene#k_scene"])
    with pytest.raises(GneError):                                                              # keyed の鍵(ffn_down の入力の空間)は使えない
        s.direction_key(built["contrast"], target="ffn", layers=[8], positions="pos:1-3", name="k_ffn", subset="all", max_tokens=8, pseudo=0)
        s.recipe(units=[f"8:N{built['n8'][0]}:nw:a1@nw_scene#k_ffn"])


@pytest.mark.model
def test_amounts_by_position_and_generation(built):
    s = built["s"]
    r = s.recipe(units=built["units"], name="nw_t")
    ck = s.nw_check(r, built["contrast"], subset="all", max_tokens=12, n_pos=4, replies="base")
    scene, tasks = ck["groups"]["scene"], ck["groups"]["dont_touch:tasks"]
    amt = lambda g, i, b: g["units"][i]["positions"][b]["amount"]["mean"]
    # 位置の鍵つきのユニットは、狙う場面の位置 1〜3 で書き、位置 0 と 4 以降ではほとんど書かない
    assert max(amt(scene, 1, b) for b in ("1", "2", "3")) > 0.5 and abs(amt(scene, 1, "0")) < 0.2
    assert max(amt(scene, 2, b) for b in ("1", "2", "3")) > 0.3
    # 書き込みたくない場面では、どのユニットもほとんど書かない
    for i in range(3):
        assert max(abs(amt(tasks, i, b)) for b in ("0", "1", "2", "3", "4+")) < 0.15
    own = s.nw_check(r, built["contrast"], subset="all", max_tokens=12, n_pos=2)
    assert own["replies"] == "own" and set(own["groups"]) == {"scene", "dont_touch:tasks"}
    # steer・behave・チャットと同じ道(lora_for)で当たる。nw の LoRA は中身で決まる名前に 1 つ
    lora = s.lora_for(r)
    assert len(lora) == 1 and Path(lora[0][0]).name.startswith("nw-") and lora[0][1] == 1.0 and s.lora_for(r) == lora
    assert s.lora_for(r.scaled(0.5)) != lora
    d = s.steer(r, prompt="Thank you for your help.", max_tokens=8, compare=False)
    assert [u["mode"] for u in d["units"]] == ["nw"] * 3 and d["units"][2]["pair"] == built["n12"][1]
    s.recipe_save("nw_t", r)
    b = s.behave([CONTRAST["user_prompts"][0], "What is 2 + 2?"], recipes=["nw_t"], max_tokens=8, match="a", name="nw_behave")
    assert b["n_prompts"] == 2 and len(b["runs"]) == 1


@pytest.mark.model
def test_export_reload_and_bake_error(built, tmp_path):
    from gguf import GGUFReader
    s = built["s"]
    r = s.recipe(units=built["units"], name="nw_t")
    e = s.nw_export(r, tmp_path / "t.lora.gguf")
    assert e["rank"] == {f"blk.{L}.ffn_{p}.weight": 2 for L in (8, 12) for p in ("gate", "up", "down")}
    rd = GGUFReader(str(tmp_path / "t.lora.gguf"))
    names = {t.name for t in rd.tensors}
    assert "blk.8.ffn_gate.weight.lora_a" in names and "blk.12.ffn_down.weight.lora_b" in names and len(names) == 12
    meta = json.loads(bytes(rd.fields["gne.nw"].parts[-1]).decode())
    assert [u["form"] for u in meta["units"]] == ["one", "two", "cap"]
    # 書き出したファイルを読み直して当てると、実行時の LoRA と同じ返事になる
    from gne.evalx import _generate
    q = [(CONTRAST["user_prompts"][0], 10)]
    a = _generate(s, q, None, s.lora_for(r), "all", lambda k: None)[0]["token_ids"]
    b = _generate(s, q, None, [(str(tmp_path / "t.lora.gguf"), 1.0)], "all", lambda k: None)[0]["token_ids"]
    assert a == b
    stripped = s.nw_export(r, tmp_path / "s.lora.gguf", strip_meta=True)
    assert "gne.nw" not in GGUFReader(str(tmp_path / "s.lora.gguf")).fields and stripped["bytes"] > 0
    for f in (lambda: s.gain(r, tmp_path / "x.gguf", bake=True), lambda: s.headproj_export(r, tmp_path / "x.gguf", bake=True),
              lambda: s.bake(r, tmp_path / "x.gguf")):
        with pytest.raises(GneError) as ei:
            f()
        assert ei.value.code == "BAD_ARGS" and "nw" in ei.value.message
    assert not (tmp_path / "x.gguf").exists()


@pytest.mark.model
def test_unsupported_shapes(built, monkeypatch):
    s = built["s"]
    with pytest.raises(GneError) as ei:                              # 確かめていない arch
        monkeypatch.setattr(s.weights.meta, "arch", "gemma3")
        NW.check_mlp(s.weights, 8)
    assert ei.value.code == "ARCH_UNSUPPORTED"
    monkeypatch.undo()
    real = s.weights.has_tensor
    monkeypatch.setattr(s.weights, "has_tensor", lambda n: False if n.endswith("ffn_gate.weight") else real(n))
    with pytest.raises(GneError) as ei:                              # ゲートの無い MLP
        NW.check_mlp(s.weights, 8)
    assert ei.value.code == "ARCH_UNSUPPORTED"
    monkeypatch.setattr(s.weights, "has_tensor", lambda n: True if n.endswith("ffn_up.bias") else real(n))
    with pytest.raises(GneError):                                    # バイアスつき
        NW.check_mlp(s.weights, 8)
    monkeypatch.undo()
    with pytest.raises(GneError):                                    # 最終層
        s.recipe(units=[f"{s.meta.n_layer - 1}:N1:nw:a1@nw_scene#k_scene"])


@pytest.mark.model
@pytest.mark.skipif(not os.environ.get("GNE_TEST_MOE") or not Path(os.environ.get("GNE_TEST_MOE", "")).exists(), reason="GNE_TEST_MOE がありません")
def test_moe_layer_is_rejected():
    from gne.weights import Weights
    w = Weights(os.environ["GNE_TEST_MOE"])
    L = next(L for L in range(w.meta.n_layer) if w.is_moe(L))
    with pytest.raises(GneError) as ei:
        NW.check_mlp(w, L)
    assert ei.value.code == "ARCH_UNSUPPORTED" and "MoE" in ei.value.message


@pytest.mark.model
def test_cli_and_jobs(built):
    from gne.cli import app
    from gne.jobs import JobRunner, standard_kinds
    from gne.jobs import GpuArbiter
    s, root = built["s"], built["root"]
    s.recipe_save("nw_t", built["units"])
    common = ["-m", str(MODEL), "--root", str(root), "--n-gpu-layers", str(NGL), "--json"]
    r = CliRunner().invoke(app, ["nw", "verify", "--recipe", str(s.recipe_path("nw_t")), *common])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output.strip().splitlines()[-1])
    assert d["command"] == "nw verify" and d["ok"] is True
    r = CliRunner().invoke(app, ["nw", "export", "--unit", built["units"][0], "-o", str(root / "cli.lora.gguf"), *common])
    assert r.exit_code == 0 and (root / "cli.lora.gguf").exists(), r.output
    r = CliRunner().invoke(app, ["nw", "export", "--unit", built["units"][0], "--bake", *common])
    assert r.exit_code != 0 and json.loads(r.output.strip().splitlines()[-1])["error"]["code"] == "BAD_ARGS"
    r = CliRunner().invoke(app, ["nw", "key", "--contrast", str(built["contrast"]), "--layers", "8", "--role", "one", "--name", "cli_one",
                                 "--subset", "all", "--max-tokens", "8", *common])
    assert r.exit_code == 0 and json.loads(r.output.strip().splitlines()[-1])["role"] == "one", r.output
    r = CliRunner().invoke(app, ["direction", "--contrast", str(built["contrast"]), "--name", "cli_dir", "--positions", "response",
                                 "--response-tokens", "8", "--same-reply", "--segments", "0,1-", *common])
    d = json.loads(r.output.strip().splitlines()[-1])
    assert r.exit_code == 0 and d["same_reply"] is True and [x["name"] for x in d["segments"]] == ["cli_dir.seg0", "cli_dir.seg1"]
    runner = JobRunner(s, GpuArbiter(), standard_kinds())

    def run(kind, params):
        import time
        j = runner.start(kind, params)
        t0 = time.time()
        while j.status in ("queued", "running") and time.time() - t0 < 600:
            time.sleep(0.2)
        assert j.status == "done", j.view()
        return j.result
    assert len(run("nw_neurons", {"layers": "8", "top": 3, "max_tokens": 8, "corpus_tokens": 256})["layers"][0]["rows"]) == 3
    assert "scene" in run("nw_check", {"recipe": str(s.recipe_path("nw_t")), "contrast": str(built["contrast"]), "subset": "all",
                                       "max_tokens": 8, "n_pos": 2})["groups"]
    assert run("direction_key", {"contrast": str(built["contrast"]), "layers": [8], "role": "scene", "name": "job_scene",
                                 "subset": "all", "max_tokens": 8, "pseudo": 2})["role"] == "scene"
    # 方向 → 鍵の一括でも方向の作り方を選べる
    p = s.direction_key_pipeline({**CONTRAST, "name": "pipe_same", "positions": "pos:1-3"}, layers=[8], target="mlpin", same_reply=True,
                                 subset="all", max_tokens=8, pseudo=2, response_tokens=8, check=False)
    assert p["direction"]["same_reply"] is True and s.direction_set("pipe_same").meta["method"] == "mean_diff_same_reply"
    p2 = s.direction_key_pipeline({**CONTRAST, "name": "pipe_old", "positions": "pos:1-3"}, layers=[8], target="mlpin",
                                  subset="all", max_tokens=8, pseudo=2, response_tokens=8, check=False)
    assert p2["direction"]["same_reply"] is False and s.direction_set("pipe_old").meta["method"] == "mean_diff"


@pytest.mark.model
def test_mcp_tools(tmp_path):
    import anyio
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    gne = str(Path(sys.executable).parent / "gne")

    async def main():
        params = StdioServerParameters(command=gne, args=["mcp", "--model", str(MODEL), "--root", str(tmp_path), "--n-gpu-layers", str(NGL)],
                                       env=dict(os.environ))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as c:
                init = await c.initialize()
                assert "nw" in (init.instructions or "") and "same_reply" in (init.instructions or "")
                tools = {t.name: t for t in (await c.list_tools()).tools}
                assert {"nw_neurons", "nw_verify", "nw_check", "nw_export"} <= set(tools)
                assert {"same_reply", "segments", "stack_layers"} <= set(tools["direction"].input_schema["properties"])
                assert "role" in tools["direction_key"].input_schema["properties"]
                assert "same_reply" in tools["direction_key_pipeline"].input_schema["properties"]
                r = await c.call_tool("nw_verify", {"units": ["8:N5:nw:a1@nope#nokey"]})
                sc = getattr(r, "structured_content", None)
                d = (sc.get("result", sc) if isinstance(sc, dict) else None) or json.loads(r.content[0].text)
                assert d["error"]["code"] in ("FILE_NOT_FOUND", "BAD_RECIPE", "BAD_KEY")
    anyio.run(main)


@pytest.mark.model
def test_prior_intervention_and_one_lora_with_proj(built, tmp_path):
    """先に当てておく介入(iv)を指定できること、proj(CHiPS)と nw を 1 つの LoRA に書き出せること、nw のレシピを対照集合の側に
    当てて方向を作れること(heads pipeline の逆の順)。"""
    from gguf import GGUFReader
    from gne.evalx import _generate
    s, c = built["s"], built["contrast"]
    li = s.weights.arch.layers[8]
    proj = f"8:H{li.n_head - 1}:s2@nw_scene"
    s.recipe_save("nw_iv", [proj])
    # iv: ふだんの働き・式の確かめ・位置ごとの量・鍵
    u = s.nw_neurons([8], max_tokens=8, corpus_tokens=256, top=3, iv="nw_iv", prompts=["Hello."])
    assert u["iv"] and u["iv"]["units"][0]["mode"] == "proj"
    r = s.recipe(units=built["units"][:2], name="nw_two")
    v = s.nw_verify(r, iv=[proj])
    assert v["ok"] and v["iv"]["units"][0]["mode"] == "proj", v
    ck = s.nw_check(r, c, subset="all", max_tokens=8, n_pos=2, iv="nw_iv")
    assert ck["iv"] and "scene" in ck["groups"]
    k = s.direction_key(c, role="scene", layers=[8], name="k_scene_iv", subset="all", max_tokens=8, pseudo=2, iv="nw_iv")
    assert k["iv"]["units"][0]["mode"] == "proj"
    # proj と nw を同じレシピに書いて、1 つの LoRA に(実行時はアダプタ 2 つ。返事は同じになる)
    both = s.recipe(units=[proj, *built["units"][:2]], name="nw_both")
    assert len(s.lora_for(both)) == 2
    e = s.headproj_export(both, tmp_path / "both.lora.gguf")
    names = {x.name for x in GGUFReader(str(tmp_path / "both.lora.gguf")).tensors}
    assert "blk.8.ffn_gate.weight.lora_a" in names and "blk.8.ffn_down.weight.lora_b" in names
    assert any(n.startswith("blk.8.attn_output") or n.startswith("blk.8.ssm_out") for n in names) and [x["form"] for x in e["nw"]] == ["one", "two"]
    q = [(CONTRAST["user_prompts"][0], 10), ("What is 12 times 8?", 10)]
    a = [o["token_ids"] for o in _generate(s, q, None, s.lora_for(both), "all", lambda k: None)]
    b = [o["token_ids"] for o in _generate(s, q, None, [(str(tmp_path / "both.lora.gguf"), 1.0)], "all", lambda k: None)]
    assert a == b
    with pytest.raises(GneError):                                    # 焼き込みは、混ざっていてもエラー
        s.headproj_export(both, tmp_path / "both.gguf", bake=True)
    with pytest.raises(GneError) as ei:                              # 同じ層のニューロンの gain と nw は 1 つにできない
        s.gain(s.recipe(units=["8:5:x2", built["units"][0]]), tmp_path / "g.lora.gguf")
    assert ei.value.code == "BAD_RECIPE"
    # 逆の順: nw のレシピを当てた状態を対照集合の側にして、方向を作る(heads pipeline が使うのと同じ仕組み)
    s.recipe_save("nw_side", built["units"][:2])
    c2 = tmp_path / "side.yaml"
    c2.write_text(yaml.safe_dump({"name": "nw_side_c", "recipe_a": "nw_side", "recipe_b": None,
                                  "user_prompts": CONTRAST["user_prompts"]}, allow_unicode=True), encoding="utf-8")
    d = s.direction(contrast=c2, positions="last")
    assert d["n_rows"]["A"] == len(CONTRAST["user_prompts"]) and float(np.abs(s.direction_set("nw_side_c").raw[10]).sum()) > 0
