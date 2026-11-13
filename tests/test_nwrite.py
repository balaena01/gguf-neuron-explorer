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
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from gne import nwrite as NW
from gne.direction import DirectionSet, parse_segments, segment_of, segment_runs, segment_text
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
    assert r.exit_code == 0 and all(x in r.output for x in ("neurons", "key", "verify", "check", "try", "pipeline", "export"))
    r = CliRunner().invoke(app, ["direction", "--help"])
    assert "--same-reply" in r.output and "--segments" in r.output and "--stack-layers" in r.output and "--stack-fast" in r.output
    r = CliRunner().invoke(app, ["nw", "pipeline", "--help"])
    assert r.exit_code == 0 and "--save" in r.output and "--layers" in r.output and "--stack-fast" in r.output
    r = CliRunner().invoke(app, ["nw", "export", "--help"])
    assert "--strip-meta" in r.output and "--name" in r.output


def test_layer_ranges_are_read_the_same_way():
    """層の指定は、どの入口でも同じ読み方: 範囲・並び・混ぜた書き方・空白区切り・リストの中の文字列。"""
    from gne.cli import parse_layers
    from gne.estimate import _n_stack_layers
    from gne.jobs import _layers
    for text, want in (("11-23", list(range(11, 24))), ("11,18,23", [11, 18, 23]), ("11,14-16,23", [11, 14, 15, 16, 23]),
                       ("11 18 23", [11, 18, 23]), (" 8 , 12 ", [8, 12]), ("12,8,8", [8, 12]), ("5", [5])):
        assert parse_layers(text) == want and _layers(text) == want, text
    assert _layers(["11-13", 20]) == [11, 12, 13, 20] and _layers([8, 12]) == [8, 12] and _layers(7) == [7] and _layers(None) is None
    assert parse_layers("all") is None and parse_layers(None) is None
    for bad in ("", "a", "11-", "23-11", "11--23", "1.5"):
        with pytest.raises(GneError):
            parse_layers(bad)
    # 見積もりの層の数え方も同じ読み方(足せない層 0 と最終層は数えない。読めない指定は 0 で、見積もりでは止めない)
    assert _n_stack_layers("11-23", 32) == 13 and _n_stack_layers([11, 18, 23], 32) == 3 and _n_stack_layers("0,8,31", 32) == 1
    assert _n_stack_layers(None, 32) == 0 and _n_stack_layers("", 32) == 0 and _n_stack_layers("x", 32) == 0


def test_pipeline_summary_and_recipe_file(tmp_path):
    """まとめて作った結果の短い要約と、レシピの保存先の決め方(モデルなし)。"""
    from types import SimpleNamespace

    from gne import fish
    amount = lambda m: {"amount": {"mean": m, "min": m - 0.1, "max": m + 0.1, "n": 4}, "rho_gate": None}
    d = {"name": "tt", "layers": [8, 12], "segments": ["0", "1-"], "form": "cap", "alpha": [1.0, 0.8], "alphas": [1.0, 0.8],
         "units": ["8:N1+N2:nw:a1@tt.seg0#tt_r_p0&tt_r_one"] * 4, "unit_rows": [{"neuron": 1, "pair": 2}] * 4, "key_base": "tt_r",
         "rekeyed": {"rows": []}, "verify": {"ok": True, "layers": []}, "usage": {"layers": [{"rows": [{"neuron": i} for i in range(5000)]}]},
         "check": {"groups": {"scene": {"n": 6, "loops": 0, "units": [{"unit": "u", "positions": {"0": amount(1.0), "1": amount(0.8), "2": {"amount": None}}},
                                                                     {"unit": "v", "positions": {"0": amount(0.6), "1": amount(0.8)}}]},
                              "dont_touch:tasks": {"n": 4, "loops": 0, "units": [{"unit": "u", "positions": {"0": amount(0.0)}}]}}},
         "notes": ["n1"], "stage_seconds": {"direction": 1.234}, "seconds": 9.87, "recipe_path": str(tmp_path / "tt.yaml")}
    sm = fish.summary(d)
    assert sm["ok"] is True and sm["n_units"] == 4 and sm["n_neurons"] == 8 and sm["recipe_path"].endswith("tt.yaml")
    assert sm["check"]["scene"]["amount"]["0"] == {"mean": 0.8, "min": 0.5, "max": 1.1} and "2" not in sm["check"]["scene"]["amount"]
    assert sm["check"]["dont_touch:tasks"]["n"] == 4 and sm["layers"] == [8, 12] and sm["rekeyed"] is True
    assert len(json.dumps(sm, ensure_ascii=False)) < 2000 < len(json.dumps(d, ensure_ascii=False))      # 表は入れない
    text = fish.summary_text(d)
    assert "合格" in text and "ユニット 4 個" in text and "ニューロン 8 個" in text and "tt.yaml" in text and "[scene]" in text
    assert fish.summary_text(sm) == text                                       # 要約からでも、結果そのものからでも同じ文
    assert "保存していません" in fish.summary_text({**d, "recipe_path": None}) and "不合格" in fish.summary_text({**d, "verify": {"ok": False}})
    # 質問ごとの値: 区間ごとの数(ほとんど書けていない・狙った量を超えた・書いてしまった)と、書けていない質問の上位が要約に入る
    q = lambda i, a0, a1, tot1, start: {"i": i, "prompt": f"q{i}", "reply_start": start, "n_tokens": 9, "loop": False,
                                        "segments": {"0": {"amount": a0, "total": a0, "rho": a0, "n_pos": 1},
                                                     "1-": None if a1 is None else {"amount": a1, "total": tot1, "rho": 0.3 + a1 / 2, "n_pos": 8}}}
    rows = [q(0, 0.98, 0.9, 1.0, "Sure,"), q(1, 0.97, 0.05, 0.05, "\u300cwell"), q(2, 0.99, 0.95, 1.4, "Hm,"), q(3, 0.96, None, None, "")]
    bq = NW._question_summary(rows, ["0", "1-"], True, 0.4, 1.2, 0.2)
    assert bq["0"] == {"n": 4, "low": 0, "over": 0, "leak": None, "amount_min": 0.96, "amount_median": pytest.approx(0.975)}
    assert (bq["1-"]["n"], bq["1-"]["low"], bq["1-"]["over"], bq["1-"]["amount_min"]) == (3, 1, 1, 0.05)      # 平均 0.63 に隠れる 1 問
    dt = NW._question_summary([q(0, 0.01, 0.3, 0.3, "12")], ["0", "1-"], False, 0.4, 1.2, 0.2)
    assert dt["1-"]["leak"] == 1 and dt["0"]["leak"] == 0 and dt["0"]["low"] is None
    worst = [{"group": "scene", "segment": "1-", "i": 1, "prompt": "q1", "reply_start": "\u300cwell", "amount": 0.05, "rho": 0.325, "loop": False}]
    d2 = {**d, "check": {"groups": {"scene": {**d["check"]["groups"]["scene"], "by_question": bq, "questions": rows},
                                    "dont_touch:tasks": {**d["check"]["groups"]["dont_touch:tasks"], "by_question": dt, "questions": []}},
                         "question_summary": {"low": 0.4, "over": 1.2, "leak": 0.2, "segments": ["0", "1-"], "cap": [[0.4, 0.8]], "n_low": 1,
                                              "worst": worst}, "path": "/x/tt_check.json"},
          "cap": ["0.4-0.8", "0.3-0.7"], "direction_name": "old_dir", "direction_reused": True}
    sm2 = fish.summary(d2)
    assert sm2["check"]["scene"]["by_question"]["1-"]["low"] == 1 and sm2["questions"]["worst"][0]["prompt"] == "q1" and sm2["questions"]["n_low"] == 1
    assert sm2["direction"] == "old_dir" and sm2["direction_reused"] is True and sm2["cap"] == ["0.4-0.8", "0.3-0.7"] and sm2["check_path"] == "/x/tt_check.json"
    assert "questions" not in json.dumps(sm2["check"]) and len(json.dumps(sm2, ensure_ascii=False)) < 3500     # 質問ごとの行そのものは入れない
    t2 = fish.summary_text(d2)
    assert "平均だけでは" in t2 and "区間 1-: 3 問のうち、ほとんど書けていない(量が 0.4 未満)1 問、狙った量を超えた(合計が 1.2 超)1 問" in t2
    assert "Q: q1  A: \u300cwell" in t2 and "書いてしまった質問(合計の大きさが 0.2 以上): 区間 0 0 / 1 問、区間 1- 1 / 1 問" in t2
    assert "方向 old_dir(すでにあるものを使った)" in t2 and "しきい値 0.4-0.8,0.3-0.7" in t2 and "/x/tt_check.json" in t2 and fish.summary_text(sm2) == t2
    assert "ほとんど書けていない質問はありません" in "\n".join(NW.question_lines({"question_summary": {**d2["check"]["question_summary"], "worst": [], "n_low": 0},
                                                                    "groups": d2["check"]["groups"]}))
    assert sm["questions"] is None and sm["cap"] is None and sm["direction"] == "tt" and "質問ごとの値" not in text      # 前の形の結果もそのまま要約できる
    # どちらの読み方で作った方向か: 速い読み方(stack_fast)で作った方向は、作ったときも、使い回したときも要約に出る
    assert sm["direction_stack_fast"] is False and sm2["direction_stack_fast"] is False and "速い読み方" not in text + t2
    for extra, word in (({}, "方向 tt(速い読み方で作った方向)"), ({"direction_name": "old_dir", "direction_reused": True},
                                                          "方向 old_dir(すでにあるものを使った)(速い読み方で作った方向)")):
        d3 = {**d, **extra, "direction_stack_fast": True}
        assert fish.summary(d3)["direction_stack_fast"] is True and word in fish.summary_text(d3) and fish.summary_text(fish.summary(d3)) == fish.summary_text(d3)
    assert "stack_fast" in fish.SETTINGS and fish.DEFAULTS["stack_fast"] is False                  # 既定は切。表の fish にも書ける
    s = SimpleNamespace(store=SimpleNamespace(recipes_dir=lambda: tmp_path / "recipes"))
    assert fish.recipe_file(s, "tt_nw") == tmp_path / "recipes" / "tt_nw.yaml"
    assert fish.recipe_file(s, str(tmp_path / "x" / "tt_nw.yaml")) == tmp_path / "x" / "tt_nw.yaml"
    for bad in ("a b", "dir/name", str(tmp_path / "a b.yaml")):
        with pytest.raises(GneError):
            fish.recipe_file(s, bad)


def test_segment_runs():
    """層を重ねるときの読み取りのまとめ方: 同じ区切りが続く返事の位置を 1 つの範囲にし、chunk 個までに切る。"""
    segs = parse_segments("0,1-3,4-")
    assert segment_runs(segs, 10) == [(0, 1, 0), (1, 4, 1), (4, 10, 2)]
    assert segment_runs(segs, 10, 4) == [(0, 1, 0), (1, 4, 1), (4, 8, 2), (8, 10, 2)]
    assert segment_runs(segs, 10, 1) == [(k, k + 1, segment_of(segs, k)) for k in range(10)]        # 1 つずつ(元の読み方)
    assert segment_runs(segs, 3) == [(0, 1, 0), (1, 3, 1)] and segment_runs(segs, 1) == [(0, 1, 0)]  # 返事が区切りの境目より短い
    assert segment_runs(segs, 4, 8) == [(0, 1, 0), (1, 4, 1)] and segment_runs(segs, 5, 8) == [(0, 1, 0), (1, 4, 1), (4, 5, 2)]
    assert segment_runs(parse_segments("0-"), 7) == [(0, 7, 0)] and segment_runs(parse_segments("0-"), 7, 3) == [(0, 3, 0), (3, 6, 0), (6, 7, 0)]
    # どの区切りにも入らない位置(None)も読む(KV キャッシュに要る)。隣り合う None は 1 つの範囲
    assert segment_runs(parse_segments("2-3"), 7) == [(0, 2, None), (2, 4, 0), (4, 7, None)]
    assert segment_runs(parse_segments("0,3,5-6"), 9, 2) == [(0, 1, 0), (1, 3, None), (3, 4, 1), (4, 5, None), (5, 7, 2), (7, 9, None)]
    assert segment_runs(segs, 0) == []
    for spec in ("0,1-3,4-", "0-", "2-3", "0,3,5-6", "1-", "0,1,2,3-10,11-"):
        sg = parse_segments(spec)
        for n in (1, 2, 5, 12, 40):
            for chunk in (None, 1, 3, 8):
                runs = segment_runs(sg, n, chunk)
                assert [k for a, b, _ in runs for k in range(a, b)] == list(range(n))                # 抜けも重なりも無く、順に
                assert all(segment_of(sg, k) == i for a, b, i in runs for k in range(a, b))         # 1 つの範囲は 1 つの区切り
                assert chunk is None or all(b - a <= chunk for a, b, _ in runs)


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


def _stack_args(s, contrast, monkeypatch, **kw):
    """層を重ねる計算(direction._stack_segments)に渡る値を 1 回分とっておく(返事・A 側の平均・重ねる前の方向)。"""
    import copy

    from gne import direction as D
    rec, orig = {}, D._stack_segments

    def spy(session, cs, replies, segs, stack, mean_a, seg_dirs, progress, **k2):
        rec.update(cs=cs, replies=copy.deepcopy(replies), segs=segs, stack=list(stack), mean_a=copy.deepcopy(mean_a),
                   seg0=copy.deepcopy(seg_dirs), kw=k2)
        used = orig(session, cs, replies, segs, stack, mean_a, seg_dirs, progress, **k2)
        rec["out"] = copy.deepcopy(seg_dirs)
        return used
    monkeypatch.setattr(D, "_stack_segments", spy)
    s.direction(contrast=contrast, positions="response", same_reply=True, **kw)
    monkeypatch.setattr(D, "_stack_segments", orig)
    return rec


def _stack_run(s, rec, spec, replies, chunk):
    """区切り spec・返事 replies で層を重ねる計算をやり直す(区切りの数が違うときは、とっておいたベクトルを順に使い回す)。"""
    from gne import direction as D
    segs = parse_segments(spec)
    n0 = len(rec["segs"])
    seg = {i: dict(rec["seg0"][i % n0]) for i in range(len(segs))}
    mean_a = {L: [v[i % n0] for i in range(len(segs))] for L, v in rec["mean_a"].items()}
    used = D._stack_segments(s, rec["cs"], replies, segs, rec["stack"], mean_a, seg, None, chunk=chunk)
    assert used == chunk or (chunk == "auto" and 1 <= used <= D.STACK_CHUNK_MAX), (chunk, used)
    rec["used"] = used
    return {(i, L): seg[i][L].astype(np.float64) for i in range(len(segs)) for L in rec["stack"]}


def _stack_diff(a: dict, b: dict) -> tuple[float, float]:
    """(相対誤差の最大 = 差の長さ ÷ 長さ、cos の最小)。"""
    rel = max(float(np.linalg.norm(a[k] - b[k]) / max(np.linalg.norm(b[k]), 1e-30)) for k in a)
    cos = min(float(a[k] @ b[k] / max(np.linalg.norm(a[k]) * np.linalg.norm(b[k]), 1e-30)) for k in a)
    return rel, cos


# 層を重ねるときの読み取りは、返事の位置をまとめて読む。llama.cpp は 1 回に読むトークンの数で計算方法を変えるので、まとめると層の
# 出力がわずかに変わる組み合わせがある(control vector を足さなくても同じ大きさで変わる。docs/verification.md §22.18)。
# - 既定(chunk="auto")は、1 つずつ読んだときと出力が完全に同じになる数だけまとめる。できる方向は、どのモデル・バックエンドでも、
#   1 つずつ読む元の読み方(chunk=1)と配列が完全に同じ。
# - まとめる数を決め打ちにしたとき(chunk=数・None)は、同じ計算だが値がわずかに違ってよい。Qwen3-0.6B の CPU は、2 個以上まとめると
#   1 個のときと層の出力が 1〜2% 変わる。方向は平均の差なので相対誤差はそれより大きく出る。実測(6 問・返事 20 トークン・層 8,10,12、
#   下の組み合わせの全部)の最大は相対誤差 0.032・cos の最小 0.9995 だった(8 問・32 トークン・7 層では 0.067・0.9978)ので、
#   その 3 倍ほどの 0.1 と 0.995 を許す。読む位置が 1 つずれると相対誤差は 0.4 を超える(下で確かめる)。
# - まとめても値が変わらない組み合わせ(SmolLM2 の Q8_0 を CPU で)では、決め打ちでも完全に一致し、既定は 2 個以上まとめる。
# (区切り, 返事を短く切るか, まとめる数の並び)。短く切った返事は 1・2・3・4・9 トークンと元の長さで、区切りの境目より短いものが混ざる
STACK_CASES = [("0,1-3,4-", False, (None, 8, 3)), ("0,1-3,4-", True, (8,)), ("0-", True, (None,)), ("2-3", False, (None,)),
               ("0,3,5-6", True, (2,)), ("0,1-3,4-9,10-", False, (8,))]


@pytest.mark.model
def test_stack_segments_chunked_matches_per_token(built, monkeypatch):
    s = built["s"]
    rec = _stack_args(s, built["contrast"], monkeypatch, name="nw_stk", response_tokens=20, segments="0,1-3,4-", stack_layers="8,10,12")
    assert rec["kw"] in ({}, {"chunk": "auto"}) and rec["stack"] == [8, 10, 12]                        # ふだんは既定のまとめ方(auto)
    full = rec["replies"]
    assert min(len(r) for r in full) >= 12
    short = [r[:n] for r, n in zip(full, (1, 2, 3, 4, 9, 20), strict=True)]       # 区切りの境目より短い返事を混ぜる
    worst = (0.0, 1.0)
    for spec, cut, chunks in STACK_CASES:
        replies = short if cut else full
        old = _stack_run(s, rec, spec, replies, 1)
        auto = _stack_run(s, rec, spec, replies, "auto")
        assert all(np.array_equal(auto[k], old[k]) for k in old), (spec, cut, rec["used"])   # 既定は元の読み方と完全に同じ
        for chunk in chunks:
            new = _stack_run(s, rec, spec, replies, chunk)
            assert all(np.array_equal(new[k], old[k]) for k in new if k[1] == 8)  # いちばん下の層は平均差そのまま
            rel, cos = _stack_diff(new, old)
            worst = (max(worst[0], rel), min(worst[1], cos))
            assert rel < 0.1 and cos > 0.995, (spec, chunk, rel, cos)
    print(f"stack: chunked vs per-token worst rel {worst[0]:.3g}, min cos {worst[1]:.5f}")
    # gne direction が作った方向(既定のまとめ方)は、1 つずつ読んだ方向と完全に同じ。使った数は次の見積もりのために覚えている
    again = _stack_run(s, rec, "0,1-3,4-", full, 1)
    assert all(np.array_equal(again[(i, L)], rec["out"][i][L].astype(np.float64)) for i, L in again)
    assert 1 <= s._stack_chunk <= 8
    # 1 回に 1 個しか読まない区切り方では、まとめても元の読み方と完全に同じ
    one = [r[:2] for r in full]
    a, b = _stack_run(s, rec, "0,1,2", one, None), _stack_run(s, rec, "0,1,2", one, 1)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    # 比べ方が位置のずれを見つけられること: 返事の区切りを 1 つずらすと、許す誤差を大きく超える
    off = _stack_run(s, rec, "0-1,2-4,5-", full, 1)
    rel, cos = _stack_diff(off, _stack_run(s, rec, "0,1-3,4-", full, 1))
    assert rel > 0.4, (rel, cos)


@pytest.mark.model
def test_stack_segments_exact_where_batching_is_exact(tmp_path, monkeypatch):
    """まとめ方で値が変わらない組み合わせ(SmolLM2-135M の Q8_0、CPU、32 個未満ずつ)では、まとめて読んだ方向が 1 つずつ読んだ
    方向と完全に一致する。"""
    small = ROOT / "models" / "SmolLM2-135M-Instruct-Q8_0.gguf"
    if not small.exists():
        pytest.skip(f"{small} がありません")
    from gne.api import Session
    c = tmp_path / "c.yaml"
    c.write_text(yaml.safe_dump({**CONTRAST, "name": "stk_small", "system_b": "You are a polite butler. Always answer formally."},
                                allow_unicode=True), encoding="utf-8")
    s = Session(small, root=tmp_path / "root", n_gpu_layers=0)
    try:
        rec = _stack_args(s, c, monkeypatch, name="stk_small", response_tokens=24, segments="0,1-3,4-", stack_layers="6,9,12,15")
        full = rec["replies"]
        short = [r[:n] for r, n in zip(full, (1, 2, 3, 4, 9, 24), strict=True)]
        used = []
        for spec, cut, chunks in STACK_CASES:
            replies = short if cut else full
            old = _stack_run(s, rec, spec, replies, 1)
            auto = _stack_run(s, rec, spec, replies, "auto")
            assert all(np.array_equal(auto[k], old[k]) for k in old), (spec, cut, rec["used"])
            used.append(rec["used"])
            for chunk in chunks:
                rel, cos = _stack_diff(_stack_run(s, rec, spec, replies, chunk), old)
                assert rel < 1e-6, (spec, chunk, rel, cos)
        assert max(used) >= 2, used                                               # この組み合わせでは、既定でまとめて読んでいる
        off = _stack_run(s, rec, "0-1,2-4,5-", full, 1)
        assert _stack_diff(off, _stack_run(s, rec, "0,1-3,4-", full, 1))[0] > 0.05
    finally:
        s.close()


@pytest.mark.model
def test_direction_stack_fast(built):
    """--stack-fast: 層を重ねる読み直しを、区切りが続く範囲ごとに 1 回で読む。既定ではなく、方向のファイルに読み方が残る。"""
    from gne.estimate import estimate
    s = built["s"]
    kw = dict(contrast=built["contrast"], positions="response", response_tokens=20, same_reply=True, segments="0,1-3,4-")
    d = s.direction(name="nw_fast", **kw, stack_layers="8,12", stack_fast=True)
    fs, ds = s.direction_set("nw_fast"), s.direction_set("nw_scene")                # nw_scene は同じ設定の既定の読み方
    assert d["stack_fast"] is True and d["stack_layers"] == [8, 12] and fs.meta["stack_read"] == "fast"
    assert "stack_fast" not in built["dir"] and "stack_read" not in ds.meta         # 既定の読み方の結果・ファイルは今までの項目のまま
    for i in range(3):
        assert np.array_equal(fs.seg[i][8], ds.seg[i][8]) and np.array_equal(fs.raw[12], ds.raw[12])   # いちばん下の層と全位置の平均は同じ
        a, b = fs.seg[i][12].astype(np.float64), ds.seg[i][12].astype(np.float64)
        assert np.linalg.norm(a - b) / np.linalg.norm(b) < 0.1, i                    # 上の層は、まとめ方の違いの分だけ違ってよい
    with pytest.raises(GneError):
        s.direction(name="nw_bad", **kw, stack_fast=True)                             # stack_layers が要る
    base = {"contrast": str(built["contrast"]), "positions": "response", "response_tokens": 96, "same_reply": True,
            "segments": "0,1-3,4-", "stack_layers": "8-20"}
    fast, slow = estimate(s, "direction", {**base, "stack_fast": True}), estimate(s, "direction", base)
    assert fast["detail"]["stack_chunk"] == "run" and fast["detail"]["stack_evals"] == 12 * len(CONTRAST["user_prompts"]) * 4
    assert fast["seconds"] < slow["seconds"]
    r = CliRunner().invoke(_app(), ["direction", "--contrast", str(built["contrast"]), "--name", "cli_fast", "--positions", "response",
                                    "--response-tokens", "6", "--same-reply", "--segments", "0,1-", "--stack-layers", "8-10", "--stack-fast",
                                    "-m", str(MODEL), "--root", str(built["root"]), "--n-gpu-layers", str(NGL), "--json"])
    out = json.loads(r.output.strip().splitlines()[-1])
    assert r.exit_code == 0 and out["stack_fast"] is True and out["stack_layers"] == [8, 9, 10], r.output


def _app():
    from gne.cli import app
    return app


@pytest.mark.model
def test_direction_estimate_counts_stacked_layers(built):
    """方向の見積もりに、層を重ねる分(層ごとに、返事を 1 本ずつ、区切りごとに数個ずつまとめて読み直す回数)が入る。"""
    from gne.estimate import STACK_READ_STEPS, STACK_SWAP_STEPS, Speed, estimate
    s = built["s"]
    base = {"contrast": str(built["contrast"]), "positions": "response", "response_tokens": 96, "same_reply": True, "segments": "0,1-3,4-"}
    n_a, sp = len(CONTRAST["user_prompts"]), Speed(s)
    memo = getattr(s, "_stack_chunk", None)

    def per(reads: int, swaps: int) -> float:       # 返事 1 本: 質問の部分で 1 回 + 返事の位置の読み取り + control vector の付け替えの分
        return (sp.setprefill + (reads * STACK_READ_STEPS + swaps * STACK_SWAP_STEPS) * sp.step(1)) / 1000
    try:
        # まとめて読める数は走らせるまで分からないので、層を重ねたことのないセッションでは 1 つずつ(いちばん遅い場合)で数える
        s._stack_chunk = None
        plain = estimate(s, "direction", base)
        one = estimate(s, "direction", {**base, "stack_layers": "8"})               # 1 層だけなら読み直しは無い
        st = estimate(s, "direction", {**base, "stack_layers": "8-20"})
        evals = 12 * n_a * (97 + 1)                                                # (層の数 − 1) × 質問の数 × 読み取りの回数
        assert st["detail"]["stack_layers"] == 13 and st["detail"]["stack_evals"] == evals and st["detail"]["stack_chunk"] == 1
        assert st["seconds"] - plain["seconds"] == pytest.approx(12 * n_a * per(97, 3), rel=1e-6)   # 付け替えは区切りごとに 1 回
        assert one["seconds"] == pytest.approx(plain["seconds"]) and "stack_evals" not in one["detail"]
        assert st["seconds"] > 5 * plain["seconds"]                                 # 13 層では、読み直しが大半になる
        # 区切りが 1 つなら付け替えは 1 回
        one_seg = estimate(s, "direction", {**base, "segments": "0-", "stack_layers": "8-20"})
        assert one_seg["seconds"] - estimate(s, "direction", {**base, "segments": "0-"})["seconds"] == pytest.approx(12 * n_a * per(97, 1), rel=1e-6)
        # このセッションで層を重ねたことがあれば、そのときにまとめて読めた数で数える(4 個なら 1 + 1 + 24 回)
        s._stack_chunk = 4
        st4 = estimate(s, "direction", {**base, "stack_layers": "8-20"})
        assert st4["detail"]["stack_chunk"] == 4 and st4["detail"]["stack_evals"] == 12 * n_a * 27
        assert st4["seconds"] - plain["seconds"] == pytest.approx(12 * n_a * per(26, 3), rel=1e-6)
        assert st4["seconds"] < 0.5 * st["seconds"]
        s._stack_chunk = None
        for form in ("8,9,10,11,12,13,14,15,16,17,18,19,20", list(range(8, 21)), ["8-20"], "8-12,13-20"):
            assert estimate(s, "direction", {**base, "stack_layers": form})["seconds"] == pytest.approx(st["seconds"]), form
        # 層の数にほぼ比例する。重ねる前提にならない組み合わせ(same_reply か segments が無い)と、読めない指定では足さない
        three = estimate(s, "direction", {**base, "stack_layers": "8,12,16"})
        assert (three["seconds"] - plain["seconds"]) * 6 == pytest.approx(st["seconds"] - plain["seconds"], rel=1e-6)
        assert estimate(s, "direction", {**base, "segments": None, "stack_layers": "8-20"})["seconds"] == pytest.approx(plain["seconds"])
        assert estimate(s, "direction", {**base, "stack_layers": "x"})["seconds"] == pytest.approx(plain["seconds"])
        # 今までの作り方は A 側・B 側の返事を作る。same_reply は A 側だけ
        old = estimate(s, "direction", {**base, "same_reply": False})
        assert old["seconds"] - plain["seconds"] == pytest.approx(sp.gen(n_a, 96), rel=1e-6)
    finally:
        s._stack_chunk = memo
    # 実測した大きさ(40 問・返事の上限 96 トークン・13 層)を 4B Q8_0 の基準の速さで数える: 1 つずつなら約 880 秒、4 個ずつなら約 280 秒。
    # 実測は 310 秒・138 秒(返事の多くが上限より前に終わり、4B は読み取りが係数より速い)。短くは出ない
    from gne.estimate import REF
    ref = lambda reads: 12 * 40 * (REF["setprefill_ms"] + (reads * STACK_READ_STEPS + 3 * STACK_SWAP_STEPS) * REF["step1_ms"]) / 1000
    assert 310 < ref(97) < 3 * 310 and 138 < ref(26) < 2.5 * 138


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
def test_keys_from_one_read(built, monkeypatch):
    """まとめて作る鍵(fish.keys_step)は、返事を質問の群ごとに 1 回だけ作って読み、全部の区切りの位置の鍵と、形に要る鍵を作る。
    できる鍵は、1 本ずつ作った鍵と同じ(同じ返事・同じ活性の行から同じ式で計算する。許す誤差は、いちばん大きい要素の 1e-6 倍)。"""
    from gne import evalx, fish
    from gne.keyed import load_key
    s, c = built["s"], built["contrast"]
    kw = dict(target="mlpin", layers=[8, 12], subset="all", max_tokens=20, pseudo=3)
    for i, sg in ((0, "0"), (2, "4-")):                               # 1 本ずつ(位置 1-3・場面・ほぼ 1 を返す鍵は built にある)
        s.direction_key(c, positions=f"pos:{sg}", name=f"sep_p{i}", **kw)
    ref = {"p0": "sep_p0", "p1": "k_pos", "p2": "sep_p2", "one": "k_one", "scene": "k_scene"}
    calls = []
    real = evalx._generate
    monkeypatch.setattr(evalx, "_generate", lambda ss, items, *a, **k: (calls.append(len(items)), real(ss, items, *a, **k))[1])

    def same(new, old):
        a, b = load_key(s, new), load_key(s, old)
        assert a.layers == b.layers == [8, 12] and a.positions == b.positions and a.counts == b.counts
        for L in a.layers:
            for x, y in ((a.kappa[L], b.kappa[L]), (a.mu[L], b.mu[L])) + (((a.kappa_ctrl[L], b.kappa_ctrl[L]),) if L in b.kappa_ctrl else ()):
                x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
                assert x.shape == y.shape and float(np.abs(x - y).max()) <= 1e-6 * float(np.abs(y).max()), (new, old, L)
        return a

    d = fish.keys_step(s, str(c), base="one_read", layers=[8, 12], segments="0,1-3,4-", form="cap", pseudo=3, max_tokens=20)
    assert calls == [len(CONTRAST["user_prompts"]), len(CONTRAST["dont_touch"][0]["prompts"])]     # 生成は群ごとに 1 回だけ(鍵 4 本)
    assert [r["key"] for r in d["rows"] if r["layer"] == 8] == ["one_read_p0", "one_read_p1", "one_read_p2", "one_read_one"]
    assert set(d["key_seconds"]) == {"position", "one"}
    for i in range(3):
        same(f"one_read_p{i}", ref[f"p{i}"])
    same("one_read_one", ref["one"])
    calls.clear()
    fish.keys_step(s, str(c), base="two_read", layers=[8, 12], segments="0,1-3,4-", form="two", pseudo=3, max_tokens=20)
    assert len(calls) == 2
    same("two_read_scene", ref["scene"])
    same("two_read_p2", ref["p2"])
    # API: 何本かをまとめて作る。1 本だけなら今までの direction_key と同じ結果の形
    rs = s.direction_keys(c, [("multi_a", "pos:0"), ("multi_b", "pos:1-3")], **kw)
    assert [r["name"] for r in rs] == ["multi_a", "multi_b"] and all(r["shared_read"]["keys"] == 2 for r in rs)
    assert rs[1]["positions_b"] == "not-pos:1-3" and rs[1]["dont_touch"] == built["k_pos"]["dont_touch"]
    assert [x["build_auc"] for x in rs[1]["layers"]] == [x["build_auc"] for x in built["k_pos"]["layers"]]
    same("multi_b", ref["p1"])
    assert "shared_read" not in built["k_pos"]
    with pytest.raises(GneError):                                    # 同じ名前の鍵を 2 本は作れない
        s.direction_keys(c, [("dup", "pos:0"), ("dup", "pos:1-3")], **kw)


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
    # 質問ごとの値: 区間は、ユニットが担当する返事の位置の範囲(位置の鍵の pos:1-3。場面の鍵だけのユニットは方向の区切り .seg0 = 0)
    qs = ck["question_summary"]
    assert qs["segments"] == ["0", "1-3"] and (qs["low"], qs["over"], qs["leak"]) == (0.4, 1.2, 0.2) and qs["cap"] == [[0.4, 0.8]]
    rows = scene["questions"]
    assert [q["i"] for q in rows] == list(range(6)) and [q["prompt"] for q in rows] == CONTRAST["user_prompts"]
    assert all(q["reply_start"] and len(q["reply_start"]) <= 40 and q["n_tokens"] > 3 and set(q["segments"]) == {"0", "1-3"} for q in rows)
    assert all(q["segments"]["1-3"]["n_pos"] == 3 and q["segments"]["0"]["n_pos"] == 1 for q in rows)
    # 区間 1-3 の値は、その区間を担当する 2 つのユニットの平均。質問で平均すると、位置ごとの表の平均と同じになる
    by_q = np.mean([q["segments"]["1-3"]["amount"] for q in rows])
    table = np.mean([amt(scene, i, b) for i in (1, 2) for b in ("1", "2", "3")])
    assert by_q == pytest.approx(table, abs=1e-6)
    tot = np.mean([q["segments"]["1-3"]["total"] for q in rows])           # 合計 = 区間 0 のユニットが同じ位置に書いた分も足したもの
    assert tot == pytest.approx(table + np.mean([amt(scene, 0, b) for b in ("1", "2", "3")]), abs=1e-6)
    bq = scene["by_question"]["1-3"]
    assert bq["n"] == 6 and bq["low"] == sum(q["segments"]["1-3"]["amount"] < 0.4 for q in rows) and bq["leak"] is None
    assert bq["amount_min"] == pytest.approx(min(q["segments"]["1-3"]["amount"] for q in rows))
    assert qs["n_low"] == sum(c["low"] for c in scene["by_question"].values()) and len(qs["worst"]) == min(qs["n_low"], 5)
    tq = tasks["by_question"]
    assert all(c["n"] == 4 and c["leak"] == 0 and c["low"] is None for c in tq.values()) and len(tasks["questions"]) == 4
    assert "path" not in ck and any("平均だけでは" in n for n in ck["notes"])
    # 目安を変えると数が変わる(全部の質問が「書けていない」になる高さ)。結果のファイルに質問ごとの行が入る
    hi = s.nw_check(r, built["contrast"], subset="all", max_tokens=12, n_pos=4, replies="base", low=5.0, over=0.01, top=2, save="nw_t_hi")
    assert hi["groups"]["scene"]["by_question"]["1-3"]["low"] == 6 and hi["question_summary"]["n_low"] == 12 and len(hi["question_summary"]["worst"]) == 2
    w = hi["question_summary"]["worst"]
    assert w[0]["amount"] <= w[1]["amount"] and w[0]["prompt"] in CONTRAST["user_prompts"] and w[0]["reply_start"]
    saved = json.loads(Path(hi["path"]).read_text(encoding="utf-8"))
    assert Path(hi["path"]).name == "nw_t_hi_check.json" and len(saved["groups"]["scene"]["questions"]) == 6 and "path" not in saved
    lines = "\n".join(NW.question_lines(hi))
    assert "平均だけでは" in lines and "区間 1-3: 6 問のうち、ほとんど書けていない(量が 5 未満)6 問" in lines and "Q: " in lines and "しきい値 0.4-0.8" in lines
    with pytest.raises(GneError):
        s.nw_check(r, built["contrast"], subset="all", max_tokens=4, save="bad name")
    own = s.nw_check(r, built["contrast"], subset="all", max_tokens=12, n_pos=2)
    assert own["replies"] == "own" and set(own["groups"]) == {"scene", "dont_touch:tasks"}
    assert own["groups"]["scene"]["by_question"]["0"]["n"] == 6
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

    def strings(path):
        """メタデータの文字列の値(キー → 文字列)。"""
        from gguf import GGUFValueType
        out = {}
        for f in GGUFReader(str(path)).fields.values():
            if f.types and f.types[0] == GGUFValueType.STRING:
                out[f.name] = bytes(f.parts[-1]).decode("utf-8", "replace")
        return out
    # ふつうの書き出し: 元のモデルの名前と識別子、レシピの名前が入る(今までどおり)
    full = strings(tmp_path / "t.lora.gguf")
    assert full["gne.source_model_id"] == s.meta.model_id and full["general.name"] == f"{s.meta.name} nw nw_t" and e["general_name"] == full["general.name"]
    # 人に渡す版: gne.* のキーが 1 つも無く、どの文字列にも元のモデルの名前・識別子・レシピの名前・方向と鍵の名前が出ない
    bare = strings(tmp_path / "s.lora.gguf")
    assert not [k for k in bare if k.startswith("gne.")] and bare["general.name"] == "adapter" and stripped["general_name"] == "adapter"
    assert stripped["strip_meta"] is True and bare["general.architecture"] == s.meta.arch and bare["adapter.type"] == "lora"
    for word in (s.meta.name, s.meta.model_id, "nw_t", "nw_scene", "k_pos", MODEL.stem):
        assert not [k for k, v in bare.items() if word in v], word
    named = s.nw_export(r, tmp_path / "n.lora.gguf", strip_meta=True, name="tone adapter")
    got = strings(tmp_path / "n.lora.gguf")
    assert got["general.name"] == "tone adapter" and named["general_name"] == "tone adapter" and "gne.source_model_id" not in got
    assert strings(tmp_path / "t.lora.gguf").keys() - got.keys() == {"gne.nw", "gne.source_model_id"}
    s.nw_export(r, tmp_path / "m.lora.gguf", name="my name")                     # 名前だけ変える(記録は残す)
    got = strings(tmp_path / "m.lora.gguf")
    assert got["general.name"] == "my name" and got["gne.source_model_id"] == s.meta.model_id and "gne.nw" in got
    with pytest.raises(GneError):
        s.nw_export(r, tmp_path / "z.lora.gguf", name=" ")
    # 中身(テンソル)は同じ: 人に渡す版を当てても同じ返事
    c2 = _generate(s, q, None, [(str(tmp_path / "s.lora.gguf"), 1.0)], "all", lambda k: None)[0]["token_ids"]
    assert c2 == a
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
    from gguf import GGUFReader
    r = CliRunner().invoke(app, ["nw", "export", "--unit", built["units"][0], "-o", str(root / "cli_s.lora.gguf"), "--strip-meta",
                                 "--name", "cli adapter", *common])
    d = json.loads(r.output.strip().splitlines()[-1])
    rd = GGUFReader(str(root / "cli_s.lora.gguf"))
    assert r.exit_code == 0 and d["strip_meta"] is True and d["general_name"] == "cli adapter", r.output
    assert not [k for k in rd.fields if k.startswith("gne.")] and bytes(rd.fields["general.name"].parts[-1]).decode() == "cli adapter"
    r = CliRunner().invoke(app, ["nw", "export", "--unit", built["units"][0], "--bake", *common])
    assert r.exit_code != 0 and json.loads(r.output.strip().splitlines()[-1])["error"]["code"] == "BAD_ARGS"
    r = CliRunner().invoke(app, ["nw", "key", "--contrast", str(built["contrast"]), "--layers", "8", "--role", "one", "--name", "cli_one",
                                 "--subset", "all", "--max-tokens", "8", *common])
    assert r.exit_code == 0 and json.loads(r.output.strip().splitlines()[-1])["role"] == "one", r.output
    r = CliRunner().invoke(app, ["direction", "--contrast", str(built["contrast"]), "--name", "cli_dir", "--positions", "response",
                                 "--response-tokens", "8", "--same-reply", "--segments", "0,1-", *common])
    d = json.loads(r.output.strip().splitlines()[-1])
    assert r.exit_code == 0 and d["same_reply"] is True and [x["name"] for x in d["segments"]] == ["cli_dir.seg0", "cli_dir.seg1"]
    # 重ねる層は、範囲でも書ける(ヘルプと docs の例の書き方)。並び・混ぜた書き方・リストの中の範囲も同じ結果
    r = CliRunner().invoke(app, ["direction", "--contrast", str(built["contrast"]), "--name", "cli_stack", "--positions", "response",
                                 "--response-tokens", "6", "--same-reply", "--segments", "0,1-", "--stack-layers", "8-10", *common])
    d = json.loads(r.output.strip().splitlines()[-1])
    assert r.exit_code == 0 and d["stack_layers"] == [8, 9, 10], r.output
    kw = dict(contrast=built["contrast"], name="api_stack", positions="response", response_tokens=6, same_reply=True, segments="0,1-")
    for form in ("8,9,10", "8,9-10", "8 9 10", ["8-10"], [8, 9, 10]):
        assert s.direction(**kw, stack_layers=form)["stack_layers"] == [8, 9, 10], form
    assert np.allclose(s.direction_set("api_stack").seg[1][10], s.direction_set("cli_stack").seg[1][10], atol=1e-5)
    for bad in ("all", "8-", "x"):
        with pytest.raises(GneError) as ei:
            s.direction(**kw, stack_layers=bad)
        assert ei.value.code == "BAD_ARGS"
    # まとめて作る(gne nw pipeline): 出るのは短い要約。--save で、gne nw verify --recipe にそのまま渡せるレシピを書く
    rp = root / "saved" / "cli_pipe.yaml"
    r = CliRunner().invoke(app, ["nw", "pipeline", "--contrast", str(built["contrast"]), "--layers", "8-9", "--alpha", "1,0.8", "--segments", "0,1-",
                                 "--name", "cli_pipe", "--save", str(rp), "--max-tokens", "8", "--response-tokens", "8", "--pseudo", "2",
                                 "--top", "6", *common])
    line = r.output.strip().splitlines()[-1]
    d = json.loads(line)
    assert r.exit_code == 0 and d["command"] == "nw pipeline" and d["ok"] is True and d["n_units"] == 4 and d["n_neurons"] == 8, r.output
    assert d["recipe_path"] == str(rp) and rp.exists() and len(d["units"]) == 4 and "usage" not in d and len(line) < 6000
    assert set(d["check"]) == {"scene", "dont_touch:tasks"} and d["layers"] == [8, 9] and d["alpha"] == [1.0, 0.8]
    saved = yaml.safe_load(rp.read_text(encoding="utf-8"))
    assert saved["name"] == "cli_pipe" and len(saved["units"]) == 4 and all(u["mode"] == "nw" and u["pair"] is not None for u in saved["units"])
    r = CliRunner().invoke(app, ["nw", "verify", "--recipe", str(rp), *common])
    assert r.exit_code == 0 and json.loads(r.output.strip().splitlines()[-1])["ok"] is True, r.output
    assert d["direction"] == "cli_pipe" and d["direction_reused"] is False and d["cap"] is None and all("!cap" not in u for u in d["units"])
    assert set(d["check"]["scene"]["by_question"]) == {"0", "1-"} and d["questions"]["segments"] == ["0", "1-"]     # 質問ごとの値の数も要約に入る
    assert Path(d["check_path"]).name == "cli_pipe_check.json" and Path(d["check_path"]).exists()
    # すでにある方向を使い回す(--direction)。区切りごとのしきい値(--cap)。方向は作り直さない
    from gne import fish
    made = s.direction_set("cli_pipe")
    before = {L: np.array(v) for L, v in made.seg[1].items()}
    r = CliRunner().invoke(app, ["nw", "pipeline", "--contrast", str(built["contrast"]), "--layers", "8-9", "--alpha", "1,0.8", "--segments", "0,1-",
                                 "--name", "cli_reuse", "--direction", "cli_pipe", "--cap", "0.4-0.8,0.3-0.7", "--max-tokens", "8", "--pseudo", "2",
                                 "--top", "6", "--no-rekey", *common])
    d = json.loads(r.output.strip().splitlines()[-1])
    assert r.exit_code == 0 and d["direction"] == "cli_pipe" and d["direction_reused"] is True and d["key_base"] == "cli_reuse", r.output
    assert d["cap"] == ["0.4-0.8", "0.3-0.7"] and d["stage_seconds"]["direction"] < 1.0 and any("作り直さず" in n for n in d["notes"])
    assert all("@cli_pipe.seg" in u and "#cli_reuse_p" in u for u in d["units"])
    assert all(u.endswith("!cap0.4-0.8") if ".seg0#" in u else u.endswith("!cap0.3-0.7") for u in d["units"]), d["units"]
    with pytest.raises(GneError):
        s.direction_set("cli_reuse")                                           # 新しい方向は作っていない
    assert all(np.array_equal(before[L], s.direction_set("cli_pipe").seg[1][L]) for L in before)
    assert [NW.form_of(u)["a"] for u in s.recipe(units=d["units"]).nw_units] == [0.4, 0.3, 0.4, 0.3]
    for kw, word in (({"layers": "8"}, "重ねて書く前提"), ({"layers": "8-9", "segments": "0,1-3,4-"}, "区切り")):
        with pytest.raises(GneError) as ei:                                    # 層・区切りが合わない方向は、モデルを流す前に止まる
            fish.pipeline(s, str(built["contrast"]), name="x", direction="cli_pipe", **{"segments": "0,1-", **kw})
        assert ei.value.code == "BAD_DIRECTION" and word in ei.value.message, ei.value.message
    with pytest.raises(GneError) as ei:                                        # 重ねる前提で作っていない方向を複数の層に: エラー(--no-stack なら使える)
        fish.reuse_direction(s, "cli_dir", layers="8-9", segments="0,1-")
    assert ei.value.code == "BAD_DIRECTION" and "--no-stack" in ei.value.hint
    assert fish.reuse_direction(s, "cli_dir", layers="8-9", segments="0,1-", stack=False)["reused"] is True
    assert fish.reuse_direction(s, "cli_dir", layers="8", segments="0,1-")["rows"][1]["direction"] == "cli_dir.seg1"
    with pytest.raises(GneError):
        fish.reuse_direction(s, "no_such_direction", layers="8", segments="0,1-")
    from gne.estimate import Speed, _fish_cost, estimate
    ep = {"contrast": str(built["contrast"]), "layers": "8-9", "segments": "0,1-"}
    sp = Speed(s)
    e_new, e_old = _fish_cost(s, sp, "fish_pipeline", ep), _fish_cost(s, sp, "fish_pipeline", {**ep, "direction": "cli_pipe"})
    assert e_new["direction_seconds"] > 0 and e_old["direction_seconds"] == 0 and e_old["seconds"] == pytest.approx(e_new["seconds"] - e_new["direction_seconds"])
    assert estimate(s, "fish_pipeline", {**ep, "direction": "cli_pipe"})["seconds"] < estimate(s, "fish_pipeline", ep)["seconds"]
    # 鍵の見積もり: 返事を作るのは 1 回なので、区切り(= 鍵)を増やしても、返事を作る分は増えない(増えるのは鍵を計算する分だけ)
    k2, k3 = _fish_cost(s, sp, "fish_keys", ep), _fish_cost(s, sp, "fish_keys", {**ep, "segments": "0,1-3,4-"})
    n_q = k2["questions"]
    assert k2["keys"] == 3 and k3["keys"] == 4 and k2["seconds"] > sp.gen(n_q, 96)
    assert k3["seconds"] - k2["seconds"] == pytest.approx(2 * n_q * 0.0035 * s.meta.n_embd / 2560) and k3["seconds"] < 1.5 * k2["seconds"]
    # gne nw check: 平均の表の後に、質問ごとの値の要約が出る。質問ごとの行は --json と結果のファイルに入る
    r = CliRunner().invoke(app, ["nw", "check", "--recipe", str(rp), "--contrast", str(built["contrast"]), "--subset", "all", "--max-tokens", "8",
                                 "--low", "5", "--top", "2", *common[:-1]])
    assert r.exit_code == 0 and "質問ごとの値(平均だけでは、一部の質問でまったく書けていないことが隠れます)" in r.output, r.output
    assert "区間 0: 6 問のうち、ほとんど書けていない(量が 5 未満)6 問" in r.output and "書けていない質問(量の小さい順に 2 件。全部で 12 件)" in r.output
    assert "[dont_touch:tasks] 書いてしまった質問" in r.output and "cli_pipe_check.json" in r.output and "[scene] 6 問、ループ" in r.output
    r = CliRunner().invoke(app, ["nw", "check", "--recipe", str(rp), "--contrast", str(built["contrast"]), "--subset", "all", "--max-tokens", "8",
                                 "--name", "cli_named", *common])
    d = json.loads(r.output.strip().splitlines()[-1])
    assert r.exit_code == 0 and Path(d["path"]).name == "cli_named_check.json" and len(d["groups"]["scene"]["questions"]) == 6, r.output
    assert d["question_summary"]["low"] == 0.4 and set(d["groups"]["scene"]["questions"][0]["segments"]) == {"0", "1-"}
    r = CliRunner().invoke(app, ["nw", "pipeline", "--contrast", str(built["contrast"]), "--layers", "8", "--cap", "0.8-0.4", *common])
    assert r.exit_code != 0 and json.loads(r.output.strip().splitlines()[-1])["error"]["code"] == "BAD_ARGS" and "見積もり" not in r.output
    r = CliRunner().invoke(app, ["nw", "pipeline", "--contrast", str(built["contrast"]), "--layers", "8", "--save", "bad name", *common])
    assert r.exit_code != 0 and json.loads(r.output.strip().splitlines()[-1])["error"]["code"] == "BAD_ARGS"     # 保存先が悪ければ、流す前に止まる
    assert "見積もり" not in r.output
    # 重みに入れる前に試す(gne nw try): 区切り 0 / 1- の方向を、層 8 の出力に区切りごとの強さで直接足す
    r = CliRunner().invoke(app, ["nw", "try", "--direction", "cli_dir", "--contrast", str(built["contrast"]), "--layers", "8", "--alpha", "1,0.5",
                                 "-n", "2", "--max-tokens", "8", *common])
    d = json.loads(r.output.strip().splitlines()[-1])
    assert r.exit_code == 0 and d["command"] == "nw try" and d["alphas"] == [1.0, 0.5] and d["layers"] == [8], r.output
    assert d["scene"]["counts"]["n"] == 2 and d["baseline"]["counts"]["n"] == 2 and d["others"]["counts"]["n"] == 2 and Path(d["path"]).exists()
    assert Path(d["path"]).name.startswith("cli_dir_") and Path(d["path"]).name.endswith("_try.json")
    r = CliRunner().invoke(app, ["nw", "try", "--direction", "cli_dir", "--contrast", str(built["contrast"]), "--layers", "8", "--alpha", "1,1,1",
                                 "-n", "1", "--max-tokens", "4", *common])                # 強さの数が区切りの数と合わない
    assert r.exit_code != 0 and json.loads(r.output.strip().splitlines()[-1])["error"]["code"] == "BAD_ARGS"
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
    jc = run("nw_check", {"recipe": str(s.recipe_path("nw_t")), "contrast": str(built["contrast"]), "subset": "all",
                          "max_tokens": 8, "n_pos": 2, "low": 0.3, "save": "job_check"})
    assert "scene" in jc["groups"] and jc["question_summary"]["low"] == 0.3 and Path(jc["path"]).name == "job_check_check.json"
    assert len(jc["groups"]["scene"]["questions"]) == 6
    t = run("fish_try", {"contrast": str(built["contrast"]), "direction": "nw_scene", "alpha": "1,1,0.8", "n": 2, "max_tokens": 8, "others": False})
    assert t["layers"] == [8, 12] and t["alphas"] == [1.0, 1.0, 0.8] and "others" not in t and t["scene"]["counts"]["n"] == 2
    # まとめて作るジョブ: summary を付けると短い要約だけが返る。save でレシピを書く(名前 → <root>/recipes/<名前>.yaml)
    pj = run("fish_pipeline", {"contrast": str(built["contrast"]), "layers": "8", "segments": "0,1-", "name": "job_pipe", "save": "job_pipe_r",
                               "summary": True, "max_tokens": 8, "response_tokens": 8, "pseudo": 2, "top": 4})
    assert pj["ok"] is True and pj["n_units"] == 2 and pj["n_neurons"] == 4 and len(pj["units"]) == 2 and "usage" not in pj and "keys" not in pj
    assert pj["recipe_path"] == str(s.recipe_path("job_pipe_r")) and s.nw_verify("job_pipe_r")["ok"] and len(json.dumps(pj, ensure_ascii=False)) < 6000
    assert set(pj["check"]["scene"]["by_question"]) == {"0", "1-"} and "questions" not in pj["check"]["scene"]
    # ジョブでも、すでにある方向としきい値を渡せる(しきい値 1 つは全部の区切りに同じ値)
    pr = run("fish_pipeline", {"contrast": str(built["contrast"]), "layers": "8", "segments": "0,1-", "name": "job_reuse", "direction": "job_pipe",
                               "cap": "0.3-0.7", "rekey": False, "summary": True, "max_tokens": 8, "pseudo": 2, "top": 4})
    assert pr["direction"] == "job_pipe" and pr["direction_reused"] and pr["cap"] == "0.3-0.7" and all(u.endswith("!cap0.3-0.7") for u in pr["units"])
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
def test_stack_fast_from_every_entry(built):
    """速い読み方(stack_fast)を、まとめて作る所(Python・CLI・ジョブ)と方向のジョブから指定できる。効くのは方向を作る段だけで、
    どちらの読み方で作った方向かが、方向のファイル・結果・要約に残る。すでにある方向を使うときは効かず、注意が出る。"""
    from gne import fish
    from gne.cli import app
    from gne.estimate import Speed, _fish_cost, estimate
    from gne.jobs import GpuArbiter, JobRunner, standard_kinds
    s, root, c = built["s"], built["root"], str(built["contrast"])
    small = dict(segments="0,1-", max_tokens=8, response_tokens=8, pseudo=2, top=6)
    # Python: 方向の段だけが速い読み方になり、できた方向は gne direction --stack-fast(Session.direction)で作ったものと同じ
    p = fish.pipeline(s, c, name="pf", layers="8,12", alpha="1,0.8", stack_fast=True, compare=False, rekey=False, **small)
    ds = s.direction_set("pf")
    assert p["direction_stack_fast"] is True and p["direction"]["stack_fast"] is True and ds.meta["stack_read"] == "fast"
    assert p["summary"]["direction_stack_fast"] is True and p["summary"]["direction_reused"] is False and p["verify"]["ok"]
    text = fish.summary_text(p)
    assert "方向 pf(速い読み方で作った方向)" in text and "注: 方向 pf は速い読み方" in text and "配るもの" in text
    ref = s.direction(contrast=c, name="pf_ref", positions="response", response_tokens=8, same_reply=True, segments="0,1-",
                      stack_layers="8,12", stack_fast=True)
    rs = s.direction_set("pf_ref")
    assert ref["stack_fast"] is True and all(np.array_equal(ds.seg[i][L], rs.seg[i][L]) for i in range(2) for L in (8, 12))
    # 既定(渡さない)は今までどおり: 既定の読み方で、方向のファイルに読み方の項目は無い
    q = fish.pipeline(s, c, name="pd", layers="8,12", alpha="1,0.8", compare=False, rekey=False, **small)
    assert q["direction_stack_fast"] is False and "stack_read" not in s.direction_set("pd").meta and "速い読み方" not in fish.summary_text(q)
    # 速い読み方で作った方向を使い回す: stack_fast を渡していなくても、要約に「速い読み方で作った方向」と出る
    n_warn = len(s.warnings)
    r1 = fish.pipeline(s, c, name="pf_use", layers="8,12", alpha="1,0.8", direction="pf", rekey=False, **small)
    assert r1["direction_reused"] and r1["direction_stack_fast"] is True and not any("効きません" in w for w in s.warnings[n_warn:])
    t1 = fish.summary_text(r1)
    assert "方向 pf(すでにあるものを使った)(速い読み方で作った方向)" in t1 and "注: 方向 pf は速い読み方" in t1 and "効きません" not in t1
    # すでにある方向と一緒に stack_fast を渡す: 効かないことを警告と注で知らせる(方向は作り直さない。読み方は方向のファイルの記録のまま)
    r2 = fish.pipeline(s, c, name="pd_use", layers="8,12", alpha="1,0.8", direction="pd", stack_fast=True, rekey=False, **small)
    assert r2["direction_reused"] and r2["direction_stack_fast"] is False and "stack_read" not in s.direction_set("pd").meta
    assert any("効きません" in w and "既定の読み方" in w for w in s.warnings[n_warn:]) and any("効きません" in n for n in r2["notes"])
    assert "速い読み方で作った方向)" not in fish.summary_text(r2) and "注: 速い読み方(stack_fast)は、すでにある方向を使うとき" in fish.summary_text(r2)
    # 読み直しが無いとき(層が 1 つ・重ねない)は使わず、注で知らせる。方向のファイルにも残さない
    one = fish.direction_step(s, c, name="pf_one", layers="8", segments="0,1-", response_tokens=8, compare=False, stack_fast=True)
    assert one["stack_fast"] is False and any("使っていません" in n for n in one["notes"]) and "stack_read" not in s.direction_set("pf_one").meta
    flat = fish.direction_step(s, c, name="pf_flat", layers="8,12", segments="0,1-", response_tokens=8, compare=False, stack=False, stack_fast=True)
    assert flat["stack_fast"] is False and flat["stack"] is False and any("使っていません" in n for n in flat["notes"])
    # CLI: gne nw pipeline --stack-fast。要約(--json)と文の両方に読み方が出る。--direction と一緒なら注意
    common = ["-m", str(MODEL), "--root", str(root), "--n-gpu-layers", str(NGL)]
    args = ["nw", "pipeline", "--contrast", c, "--layers", "8,12", "--alpha", "1,0.8", "--segments", "0,1-", "--max-tokens", "8",
            "--response-tokens", "8", "--pseudo", "2", "--top", "6", "--no-rekey"]
    r = CliRunner().invoke(app, [*args, "--name", "cli_fastpipe", "--stack-fast", *common, "--json"])
    d = json.loads(r.output.strip().splitlines()[-1])
    assert r.exit_code == 0 and d["direction_stack_fast"] is True and d["direction"] == "cli_fastpipe" and d["ok"] is True, r.output
    fs = s.direction_set("cli_fastpipe")
    assert fs.meta["stack_read"] == "fast" and all(np.array_equal(fs.seg[i][L], ds.seg[i][L]) for i in range(2) for L in (8, 12))
    r = CliRunner().invoke(app, [*args, "--name", "cli_fastuse", "--direction", "cli_fastpipe", "--stack-fast", *common])
    assert r.exit_code == 0 and "方向 cli_fastpipe(すでにあるものを使った)(速い読み方で作った方向)" in r.output, r.output
    assert "効きません" in r.output and "方向 cli_fastpipe は速い読み方で作ったものです" in r.output
    r = CliRunner().invoke(app, [*args, "--name", "cli_slowuse", "--direction", "pd", "--no-stack-fast", *common, "--json"])
    d = json.loads(r.output.strip().splitlines()[-1])
    assert r.exit_code == 0 and d["direction_stack_fast"] is False and not any("効きません" in n for n in d["notes"]), r.output
    # ジョブ: 方向(direction)・FiSH の方向の段(fish_direction)・まとめて作る(fish_pipeline)。表の fish.stack_fast でも書ける
    runner = JobRunner(s, GpuArbiter(), standard_kinds())

    def run(kind, params):
        import time
        j = runner.start(kind, params)
        t0 = time.time()
        while j.status in ("queued", "running") and time.time() - t0 < 600:
            time.sleep(0.2)
        assert j.status == "done", j.view()
        return j
    jd = run("direction", {"contrast": c, "name": "job_fastdir", "positions": "response", "response_tokens": 8, "same_reply": True,
                           "segments": "0,1-", "stack_layers": "8,12", "stack_fast": True})
    assert jd.result["stack_fast"] is True and s.direction_set("job_fastdir").meta["stack_read"] == "fast"
    assert jd.view()["estimate_stack_first"] is False
    jf = run("fish_direction", {"contrast": c, "name": "job_fishdir", "layers": "8,12", "segments": "0,1-", "response_tokens": 8,
                                "compare": False, "stack_fast": True})
    assert jf.result["stack_fast"] is True and s.direction_set("job_fishdir").meta["stack_read"] == "fast"
    assert all(np.array_equal(s.direction_set("job_fishdir").seg[1][L], ds.seg[1][L]) for L in (8, 12))
    jp = run("fish_pipeline", {"spec": {**CONTRAST, "name": "job_fastpipe", "fish": {"segments": "0,1-", "stack_fast": True, "rekey": False,
                                                                                     "max_tokens": 8, "response_tokens": 8, "pseudo": 2}},
                               "layers": "8,12", "alpha": "1,0.8", "compare": False, "top": 6, "summary": True})
    assert jp.result["direction_stack_fast"] is True and jp.result["ok"] is True and s.direction_set("job_fastpipe").meta["stack_read"] == "fast"
    jo = run("fish_pipeline", {"contrast": c, "name": "job_fastoff", "layers": "8,12", "alpha": "1,0.8", "direction": "job_fastpipe",
                               "stack_fast": False, "rekey": False, "summary": True, **{k: v for k, v in small.items() if k != "response_tokens"}})
    assert jo.result["direction_stack_fast"] is True and jo.result["direction_reused"] is True
    # 見積もり: 速い読み方なら、層を重ねる読み直しを「区切りが続く範囲ごとに 1 回」で数える(短くなる)。既定の読み方は、まとめて読める
    # 数をまだ確かめていないセッションでは 1 つずつで数え、そのことを見積もりに残す(表示に「初回は長めに出ます」と添える)
    memo = getattr(s, "_stack_chunk", None)
    try:
        s._stack_chunk = None
        sp, ep = Speed(s), {"contrast": c, "layers": "8-20", "segments": "0,1-3,4-"}
        slow, fast = _fish_cost(s, sp, "fish_pipeline", ep), _fish_cost(s, sp, "fish_pipeline", {**ep, "stack_fast": True})
        assert slow["stack_first"] is True and fast["stack_first"] is False and fast["direction_seconds"] < 0.5 * slow["direction_seconds"]
        assert slow["seconds"] - fast["seconds"] == pytest.approx(slow["direction_seconds"] - fast["direction_seconds"])   # ほかの段は同じ
        dslow, dfast = _fish_cost(s, sp, "fish_direction", ep), _fish_cost(s, sp, "fish_direction", {**ep, "stack_fast": True})
        assert dslow["stack_first"] is True and dfast["stack_first"] is False and dfast["seconds"] < dslow["seconds"]
        assert dslow["stack_seconds"] - dfast["stack_seconds"] == pytest.approx(dslow["seconds"] - dfast["seconds"])
        spec_fast = {"spec": {**CONTRAST, "name": "est_fast", "fish": {"stack_fast": True}}, "layers": "8-20"}       # 表の fish.stack_fast
        assert _fish_cost(s, sp, "fish_pipeline", spec_fast)["direction_seconds"] == pytest.approx(fast["direction_seconds"])
        for off in ({"stack": False}, {"layers": "8"}, {"direction": "pf"}):     # 読み直しが無い・方向を作らないときは、添えない
            assert _fish_cost(s, sp, "fish_pipeline", {**ep, **off})["stack_first"] is False, off
        base = {"contrast": c, "positions": "response", "same_reply": True, "segments": "0,1-3,4-", "stack_layers": "8-20"}
        assert estimate(s, "direction", base)["detail"]["stack_first"] is True
        assert estimate(s, "direction", {**base, "stack_fast": True})["detail"]["stack_first"] is False
        s._stack_chunk = 4                                                       # このセッションで確かめた数があれば、それで数える(添えない)
        known = _fish_cost(s, sp, "fish_pipeline", ep)
        assert known["stack_first"] is False and fast["direction_seconds"] < known["direction_seconds"] < slow["direction_seconds"]
        assert estimate(s, "direction", base)["detail"]["stack_first"] is False
    finally:
        s._stack_chunk = memo
    r = CliRunner().invoke(app, ["nw", "pipeline", "--contrast", c, "--layers", "8", "--save", "bad name", "--stack-fast", *common, "--json"])
    assert r.exit_code != 0 and json.loads(r.output.strip().splitlines()[-1])["error"]["code"] == "BAD_ARGS"


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
                assert "nw" in (init.instructions or "") and "same_reply" in (init.instructions or "") and "stack_fast" in (init.instructions or "")
                tools = {t.name: t for t in (await c.list_tools()).tools}
                assert {"nw_neurons", "nw_verify", "nw_check", "nw_export", "nw_try", "fish_pipeline"} <= set(tools)
                assert {"direction", "layers", "alpha", "temperature"} <= set(tools["nw_try"].input_schema["properties"])
                assert {"alpha", "save", "summary", "direction", "cap", "stack_fast"} <= set(tools["fish_pipeline"].input_schema["properties"])
                assert "探索" in tools["fish_pipeline"].description and "direction_stack_fast" in tools["fish_pipeline"].description
                assert {"low", "over", "top", "save"} <= set(tools["nw_check"].input_schema["properties"])
                assert {"strip_meta", "adapter_name"} <= set(tools["nw_export"].input_schema["properties"])
                r = await c.call_tool("fish_pipeline", {"contrast": "tone_in_scene", "layers": []})       # 層は必須
                sc = getattr(r, "structured_content", None)
                d = (sc.get("result", sc) if isinstance(sc, dict) else None) or json.loads(r.content[0].text)
                assert d["error"]["code"] == "BAD_ARGS"
                assert {"same_reply", "segments", "stack_layers", "stack_fast"} <= set(tools["direction"].input_schema["properties"])
                assert tools["direction"].input_schema["properties"]["stack_fast"].get("default") is False and "stack_read" in tools["direction"].description
                r = await c.call_tool("direction", {"contrast": "evil", "positions": "response", "stack_fast": True})   # stack_layers が要る
                sc = getattr(r, "structured_content", None)
                d = (sc.get("result", sc) if isinstance(sc, dict) else None) or json.loads(r.content[0].text)
                assert d["error"]["code"] == "BAD_ARGS" and "stack_fast" in d["error"]["message"]
                assert "role" in tools["direction_key"].input_schema["properties"]
                assert "same_reply" in tools["direction_key_pipeline"].input_schema["properties"]
                r = await c.call_tool("nw_verify", {"units": ["8:N5:nw:a1@nope#nokey"]})
                sc = getattr(r, "structured_content", None)
                d = (sc.get("result", sc) if isinstance(sc, dict) else None) or json.loads(r.content[0].text)
                assert d["error"]["code"] in ("FILE_NOT_FOUND", "BAD_RECIPE", "BAD_KEY")
    anyio.run(main)


@pytest.mark.model
def test_try_before_writing(built, monkeypatch):
    """重みに入れる前に試す(Session.nw_try): 区切りつきの方向を、狙う場面の質問の返事の位置どおりに層の出力へ直接足す。
    どの位置を読ませるときにどのベクトルが付いているかを、エンジンの呼び出しを見て確かめる。"""
    s, e = built["s"], built["s"].engine
    ds = s.direction_set("nw_scene")
    al = [1.0, 0.5, 0.25]
    log, cur = [], {"v": None}                       # (読ませた位置, そのとき付いていたベクトル)
    set_cvec, prefill, decode = e.set_cvec, e.prefill, e._decode

    def spy_cvec(ctx, vecs, **kw):
        cur["v"] = None if not vecs else {int(L): np.array(v, dtype=np.float32) for L, v in vecs.items()}
        return set_cvec(ctx, vecs, **kw)

    def spy_prefill(ctx, seqs, **kw):
        if len(seqs) == 1:
            log.extend((kw.get("pos0", 0) + i, cur["v"]) for i in range(len(seqs[0])))
        return prefill(ctx, seqs, **kw)

    def spy_decode(ctx, items, *a, **kw):
        if len(items) == 1:
            log.append((items[0][1], cur["v"]))
        return decode(ctx, items, *a, **kw)
    monkeypatch.setattr(e, "set_cvec", spy_cvec)
    monkeypatch.setattr(e, "prefill", spy_prefill)
    monkeypatch.setattr(e, "_decode", spy_decode)
    q = CONTRAST["user_prompts"][0]
    r = s.nw_try("nw_scene", built["contrast"], alpha=al, prompts=[q], max_tokens=10, others=False, baseline=False)
    monkeypatch.undo()
    P = len(e.tokenize_prompt(e.render_chat([{"role": "user", "content": q}])))
    assert r["layers"] == [8, 12] and r["stack_layers"] == [8, 12] and r["alphas"] == al and r["segments"] == ["0", "1-3", "4-"]
    seen = dict(log)
    assert sorted(seen) == list(range(max(seen) + 1)) and max(seen) >= P + 4          # 質問の全部の位置と、返事の位置 5 以上まで読ませた
    for pos, v in seen.items():
        k = pos - (P - 1)                            # 返事の位置(0 = 質問の最後のトークン = 返事を書き始める直前)
        if k < 0:
            assert v is None, pos                    # 質問の文には足さない
            continue
        i = 0 if k == 0 else (1 if k <= 3 else 2)
        assert set(v) == {8, 12} and all(np.allclose(v[L], al[i] * ds.seg[i][L], atol=1e-6) for L in (8, 12)), (pos, k)
    assert r["added"][2]["norm"][12] == pytest.approx(0.25 * float(np.linalg.norm(ds.seg[2][12])), rel=1e-5)
    # 強さ 0 は、何も足さない返事と同じ(何も足さない返事も、足す側と同じ作り方で 1 問ずつ作る。まとめて生成すると、GPU では
    # 組の作り方の違いだけで返事が分かれることがある)。サンプリングでも同じ種で引くので同じ。強く足すと返事が変わる。数は behave と同じ項目
    z = s.nw_try("nw_scene", built["contrast"], alpha=0, n=3, max_tokens=12)
    assert [x["text"] for x in z["scene"]["replies"]] == [x["text"] for x in z["baseline"]["replies"]]
    zs = s.nw_try("nw_scene", built["contrast"], alpha=0, n=3, max_tokens=12, temperature=0.8, seed=3, others=False)
    assert [x["text"] for x in zs["scene"]["replies"]] == [x["text"] for x in zs["baseline"]["replies"]]
    assert set(z["scene"]["counts"]) == {"n", "empty", "loop", "at_limit", "tokens"} and z["others"]["counts"]["n"] == 3      # validate.match が無いので一致は数えない
    big = s.nw_try("nw_scene", built["contrast"], alpha=6, n=3, max_tokens=12, match="[a-z]", others=False)
    assert [x["text"] for x in big["scene"]["replies"]] != [x["text"] for x in big["baseline"]["replies"]]
    assert "match" in big["scene"]["counts"] and big["match"] == "[a-z]" and not big["own_prompts"]
    # 層を方向の前提(重ねる層)と変えると注が出る。サンプリングは種で変わる
    one = s.nw_try("nw_scene", built["contrast"], layers="8", alpha="1,1,0.8", n=2, max_tokens=12, temperature=0.9, seed=1, others=False, baseline=False)
    two = s.nw_try("nw_scene", built["contrast"], layers="8", alpha="1,1,0.8", n=2, max_tokens=12, temperature=0.9, seed=2, others=False, baseline=False)
    assert any("重ねて書く前提" in n for n in one["notes"]) and one["temperature"] == 0.9
    assert [x["text"] for x in one["scene"]["replies"]] != [x["text"] for x in two["scene"]["replies"]]
    # 区切りの無い方向は、返事の全部の位置に同じベクトル。層は自分で渡す
    old = s.nw_try("nw_old", built["contrast"], layers=[8], alpha=0.5, n=1, max_tokens=6, others=False, baseline=False)
    assert old["segments"] == ["0-"] and any("区切りつきで作られていない" in n for n in old["notes"])
    for kw in ({"layers": "0,8"}, {"layers": [e.n_layer - 1]}, {"alpha": [1, 1]}, {"alpha": "x"}):
        with pytest.raises(GneError):
            s.nw_try("nw_scene", built["contrast"], n=1, max_tokens=4, **kw)
    with pytest.raises(GneError):
        s.nw_try("nw_old", built["contrast"], n=1, max_tokens=4)                          # 重ねる層の無い方向で、層を省いた
    # 結果のファイル: 名前を省くと、設定が違えば別のファイル(前の結果を上書きしない)。同じ設定の流し直しは同じファイル
    kw = dict(n=1, max_tokens=4, others=False, baseline=False)
    a = s.nw_try("nw_scene", built["contrast"], alpha="1,1,0.8", **kw)
    b = s.nw_try("nw_scene", built["contrast"], alpha="1,1,0.5", **kw)                    # 強さが違う
    c = s.nw_try("nw_scene", built["contrast"], alpha="1,1,0.8", prompts=[CONTRAST["user_prompts"][3]], **kw)   # 質問が違う
    d = s.nw_try("nw_scene", built["contrast"], alpha="1,1,0.8", layers="8", **kw)        # 層が違う
    f = s.nw_try("nw_scene", built["contrast"], alpha="1,1,0.8", temperature=0.7, seed=2, **kw)   # 温度と種が違う
    paths = [x["path"] for x in (a, b, c, d, f)]
    assert len(set(paths)) == 5 and all(Path(p).exists() and re.fullmatch(r"nw_scene_[0-9a-f]{6}_try\.json", Path(p).name) for p in paths)
    assert json.loads(Path(a["path"]).read_text(encoding="utf-8"))["alphas"] == [1.0, 1.0, 0.8]      # 後の実行で書き換わっていない
    assert not any("上書き" in n for x in (a, b, c, d, f) for n in x["notes"])
    a2 = s.nw_try("nw_scene", built["contrast"], alpha="1,1,0.8", **kw)
    assert a2["path"] == a["path"] and a2["name"] == a["name"] and any("上書き" in n and "同じ設定" in n for n in a2["notes"])
    # 名前を付けると <名前>_try.json。同じ名前で流し直すと上書きして、注で知らせる
    n1 = s.nw_try("nw_scene", built["contrast"], alpha="1,1,0.8", name="mine", **kw)
    n2 = s.nw_try("nw_scene", built["contrast"], alpha="1,1,0.5", name="mine", **kw)
    assert Path(n1["path"]).name == "mine_try.json" and n2["path"] == n1["path"] and not any("上書き" in n for n in n1["notes"])
    assert any("上書き" in n and "name" in n for n in n2["notes"])


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
