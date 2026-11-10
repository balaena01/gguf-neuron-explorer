"""keyed の効く場所を探す 2 段(gne direction key-causal / key-causal-verify、gne/keyed_causal.py)のテスト。

モデルなし: 書き込み先・α の段の読み方、probe(JSONL)の読み方とエラー、一致の文字を含むトークン(日本語の 1 文字が複数の
トークンにまたがるとき)、全候補の比較用の鍵の 5% 点による印と「偶然でも印が付く数」、比例の目安、第 2 段の数え方(正規表現・
ラベル・返事が変わった数と先頭から同じ部分の割合)、ジョブの包み・見積もり、MCP のツール(見立てた Session)、--json のスキーマ。
モデルあり(Qwen3-0.6B、GNE_TEST_MODEL。CPU で回すときは GNE_TEST_NGL=0): 第 1 段の出力の形(候補 × α の行、比較用の鍵、印、
偶然の数、key-check の参考、保存と一覧)と、行の効き目が「keyed の LoRA を実行時と同じように当てて 1 本ずつ読んだ値」と合うこと、
probe の読み方(作成に使った質問の行を外す、書き込みたくない場面の行)、第 2 段の数え方(正規表現と judge のラベル)、エラー、
ジョブ(結果・進み具合・見積もり・取り消し)、サーバの入口、ブラウザの「効く層を探す」(第 1 段 → 行を選んで第 2 段 →
「+ レシピ」、遅れて届いた読み込みで上書きしない、英語の文言)。
"""

import json
import os
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import anyio
import numpy as np
import pytest
import yaml

from gne.steer import GneError

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
NGL = int(os.environ.get("GNE_TEST_NGL", "-1"))
CHROME = Path(os.environ.get("GNE_CHROME", Path.home() / ".cache/ms-playwright/chromium-1223/chrome-linux64/chrome"))
ZUNDAMON = ROOT / "configs" / "contrast" / "zundamon_endings.yaml"
EN_REPLIES = ["Go for a walk in the park and read a good book.", "Fry the vegetables, add water and the roux, then simmer.",
              "Keep the room dark and avoid screens before bed.", "Pick something they love and add a short letter."]


def _spec():
    """語尾の対照集合(configs/contrast/zundamon_endings.yaml)に、素の返事に見立てたです・ます調の返事と英語の返事を足したもの
    (第 1 段で返事を作らない)。名前は kc_endings。"""
    from tests.test_change07_ui import REPLIES
    d = yaml.safe_load(ZUNDAMON.read_text(encoding="utf-8"))
    d["name"] = "kc_endings"
    d["replies"] = list(REPLIES)
    d["dont_touch"][0]["replies"] = list(EN_REPLIES)
    return d


# --- モデルなし ----------------------------------------------------------------------------------------------

def test_targets_alphas_units():
    from gne.keyed_causal import parse_alphas, parse_targets, target_tag, unit_spec
    assert parse_targets("ffn, attn,ffn") == ["ffn", "attn"]
    assert parse_targets(["S", "expert:3", "shared"]) == ["shared", "expert:3"]
    with pytest.raises(GneError):
        parse_targets("mlp")
    assert parse_alphas("-1,-2, -4") == [-1.0, -2.0, -4.0] and parse_alphas([2, 2, 1]) == [2.0, 1.0] and parse_alphas(None) == []
    for bad in ("1,0", "a", "1,inf"):
        with pytest.raises(GneError) as ei:
            parse_alphas(bad)
        assert ei.value.code == "BAD_ARGS"
    assert [target_tag(t) for t in ("ffn", "attn", "shared", "expert:7")] == ["ffn", "attn", "S", "E7"]
    assert unit_spec(14, "expert:7", -2.0, "d", "k") == "14:E7:keyed:a-2@d#k"
    from gne.steer import parse_unit
    u = parse_unit(unit_spec(8, "attn", 0.5, "dir", "key-attn"))
    assert (u.mode, u.kind, u.k, u.direction, u.key) == ("keyed", "attn", 0.5, "dir", "key-attn")


def test_read_probe(tmp_path):
    from gne.keyed_causal import read_probe
    p = tmp_path / "p.jsonl"
    rows = [{"prompt": "Q1", "prefix": "Answer:", "behavior": " B", "desired": " C"},
            {"prompt": [{"role": "user", "content": "Q2"}], "behavior": "x", "desired": "y", "group": "facts"},
            {"prompt": "Q3", "behavior": "x", "desired": "y", "dont_touch": True, "system": "s"}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    got = read_probe(p)
    assert [r["group"] for r in got] == [None, "facts", "dont_touch"] and got[0]["prefix"] == "Answer:" and got[1]["prefix"] == ""
    assert got[2]["system"] == "s"
    assert read_probe(rows) == got and read_probe(None) == []
    j = tmp_path / "p.json"
    j.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    assert read_probe(j) == got
    for bad, word in (([{"prompt": "q", "behavior": "a"}], "desired"), ([{"prompt": "q", "behavior": "a", "desired": "a"}], "同じ"),
                      ([{"prompt": "q", "behavior": "a", "desired": "b", "x": 1}], "知らないキー"), ([], "リスト"),
                      ([{"prompt": "", "behavior": "a", "desired": "b"}], "prompt"),
                      ([{"prompt": [{"role": "assistant", "content": "x"}], "behavior": "a", "desired": "b"}], "")):
        with pytest.raises(GneError) as ei:
            read_probe(bad)
        assert ei.value.code == "BAD_ARGS" and word in ei.value.message
    bj = tmp_path / "b.jsonl"
    bj.write_text("{oops\n", encoding="utf-8")
    with pytest.raises(GneError) as ei:
        read_probe(bj)
    assert ei.value.code == "BAD_ARGS"
    with pytest.raises(GneError) as ei:
        read_probe(tmp_path / "none.jsonl")
    assert ei.value.code == "FILE_NOT_FOUND"


def test_match_spans_multibyte():
    """一致の文字を含むトークン: 「す」の UTF-8 の 3 バイトが 2 つのトークンにまたがっても、一致の始まりと終わりのトークンを取る。"""
    from gne.keyed_causal import match_spans
    s = "おいしいです。"
    b = s.encode()
    i = b.index("す".encode())
    pieces = [b[:i - 3], b[i - 3:i + 1], b[i + 1:i + 3], b[i + 3:]]       # 「で」+「す」の 1 バイト目 / 「す」の残り / 「。」
    assert b"".join(pieces) == b
    assert match_spans(pieces, "(です|ます)。") == [[1, 2, 3]]
    assert match_spans(pieces, "おいしい") == [[0]]
    assert match_spans([b"ab", b"cd"], "x") == [] and match_spans([b"ab"], "a*") == [[0]]


def _row(target, layer, alpha, mean, pseudo, pmean=None, ppseudo=None):
    return {"target": target, "layer": layer, "alpha": alpha, "reply": {"mean": mean, "ci": 0.1, "n": 4},
            "pseudo": {"values": pseudo, "min": min(pseudo), "max": max(pseudo)},
            "probe": None if pmean is None else {"mean": pmean, "ci": None, "n": 2},
            "probe_pseudo": None if ppseudo is None else {"values": ppseudo, "min": min(ppseudo), "max": max(ppseudo)}}


def test_summarize_marks_and_chance():
    """全候補の比較用の鍵の値を α ごとにまとめた分布の 5% 点より下がった候補に印。偶然の数 = 候補の数 × 5%。"""
    from gne.keyed_causal import summarize
    rng = np.random.default_rng(0)
    rows = []
    for L in range(20):
        for a in (-1.0, -2.0):
            ps = list(rng.normal(0, 0.1 * abs(a), 3))
            mean = -1.0 if L in (3, 7) else float(rng.normal(0, 0.05))
            rows.append(_row("ffn", L, a, mean, ps, pmean=0.0 if L != 3 else -5.0, ppseudo=[0.1, 0.2, -0.1]))
    pool, marks = summarize(rows, [-1.0, -2.0])
    for a in ("-1", "-2"):
        vals = [v for r in rows if f"{r['alpha']:g}" == a for v in r["pseudo"]["values"]]
        assert pool[a]["reply"]["n"] == 60 and abs(pool[a]["reply"]["p5"] - np.percentile(vals, 5)) < 1e-12
        assert marks[a]["reply"]["candidates"] == 20 and marks[a]["reply"]["expected"] == 1.0
        assert marks[a]["probe"]["candidates"] == 20
    for r in rows:
        p5 = pool[f"{r['alpha']:g}"]["reply"]["p5"]
        assert r["mark"] == (r["reply"]["mean"] < p5)
    assert all(r["mark"] for r in rows if r["layer"] in (3, 7))
    assert [r["layer"] for r in rows if r["probe_mark"]] == [3, 3]
    # 効き目が無い行は印の数に入れない(None)
    rows2 = [_row("ffn", 1, 1.0, 0.0, [0.1, 0.2]), {**_row("attn", 1, 1.0, 0.0, [0.0, 0.1]), "reply": None}]
    pool2, marks2 = summarize(rows2, [1.0])
    assert marks2["1"]["reply"]["candidates"] == 1 and rows2[1]["mark"] is None and pool2["1"]["reply"]["n"] == 4
    assert summarize([], [1.0]) == ({"1": {}}, {"1": {}})


def test_scale_ratio():
    from gne.keyed_causal import scale_ratio
    assert scale_ratio([-1, -2, -4], [-0.1, -0.2, -0.4])["ratio"] == pytest.approx(1.0)
    r = scale_ratio([1, 2, 4], [0.1, 0.15, 0.2])
    assert r["consistent"] and r["ratio"] == pytest.approx(0.5)
    r = scale_ratio([1, 2], [0.1, -0.1])
    assert not r["consistent"] and r["ratio"] is None
    assert scale_ratio([1], [0.1]) is None and scale_ratio([1, 2], [None, 0.1]) is None
    r = scale_ratio([1, 2, 4], [-0.01, -0.3, -2.0], [0.05, 0.1, 0.5])        # 小さい α の値が 95% の幅の中: 比は出さない
    assert r["noisy"] and r["ratio"] is None and r["consistent"]
    assert scale_ratio([1, 2], [-0.2, -0.4], [0.05, 0.1])["ratio"] == pytest.approx(1.0)


def test_drop_per_harm():
    """確率の下がり方 ÷ 副作用: (−効き目) ÷ (−副作用)。副作用が下限(0.01)より小さいとき・上がったときは下限で割る。確率が下がって
    いなければ None。"""
    from gne.keyed_causal import SIDE_FLOOR, drop_per_harm
    assert SIDE_FLOOR == 0.01
    assert drop_per_harm({"mean": -1.0}, {"mean": -0.5}) == pytest.approx(2.0)
    assert drop_per_harm({"mean": -1.0}, {"mean": -0.001}) == pytest.approx(100.0)
    assert drop_per_harm({"mean": -1.0}, {"mean": +0.02}) == pytest.approx(100.0)
    assert drop_per_harm({"mean": -1.0}, None) == pytest.approx(100.0)
    assert drop_per_harm({"mean": -0.3}, {"mean": -0.1}, floor=0.2) == pytest.approx(1.5)
    assert drop_per_harm({"mean": 0.1}, {"mean": -0.5}) is None and drop_per_harm({"mean": 0.0}, {"mean": -0.5}) is None
    assert drop_per_harm(None, {"mean": -0.5}) is None


def _rows3():
    """候補 × α の行(2 か所 × α 1・2・4)。ffn L5 はどの α でも印、attn L5 は α 4 だけ。"""
    rows = []
    for tg in ("ffn", "attn"):
        for a in (1.0, 2.0, 4.0):
            r = _row(tg, 5, a, -0.1 * a, [0.0, 0.1])
            r.update(key=f"k-{tg}", unit=f"5:{tg}:keyed:a{a:g}@d#k-{tg}", mark=(tg == "ffn" or a == 4.0), probe_mark=None,
                     side={"b": {"mean": -0.02 * a, "ci": 0.01, "n": 4}, "dt_min": -0.001 * a},
                     pseudo_side={"b": {"min": -0.1, "max": -0.05, "values": [-0.1, -0.05]}},
                     key_check={"margin": 0.3}, linear={"consistent": True, "ratio": 1.0, "noisy": False})
            rows.append(r)
    return rows


def test_candidates_from_rows_and_lines():
    """候補 × α の行を 1 か所 1 行にまとめる(α ごとの効き目・印・副作用・ratio、全部の強さで印)。CLI の表も 1 か所 1 行で、
    --sort と --ratio-alpha で並べ替えと割る強さを変えられる。保存した古い結果にも with_candidates で足す。"""
    from gne.cli import kc_sorted, key_causal_lines
    from gne.keyed_causal import candidates_from_rows, drop_per_harm, with_candidates
    rows = _rows3()
    cs = candidates_from_rows(rows, [1.0, 2.0, 4.0])
    assert [(c["target"], c["layer"]) for c in cs] == [("ffn", 5), ("attn", 5)]
    f, a = cs
    assert f["alphas"] == [1.0, 2.0, 4.0] and set(f["by_alpha"]) == {"1", "2", "4"} and f["key"] == "k-ffn"
    assert f["all_marked"] is True and a["all_marked"] is False and f["all_marked_probe"] is None
    b = f["by_alpha"]["4"]
    assert b["unit"] == "5:ffn:keyed:a4@d#k-ffn" and b["mark"] is True and b["side_b"]["mean"] == pytest.approx(-0.08)
    assert b["ratio"] == pytest.approx(drop_per_harm({"mean": -0.4}, {"mean": -0.08})) == pytest.approx(5.0)
    assert f["by_alpha"]["1"]["ratio"] == pytest.approx(0.1 / 0.02)
    d = {"name": "r", "contrast": "c", "direction": "d", "iv_spec": None, "targets": ["ffn", "attn"], "n_candidates": 2,
         "alphas": [1.0, 2.0, 4.0], "pseudo": 2, "seconds": 1.0, "notes": ["note0"], "measure": {"regex": "x", "source": "positions"},
         "counts": {"a": {"examples": 4, "matched": 4}}, "marks": {}, "pool": {}, "chance": 0.05, "rows": rows, "dropped": []}
    d2 = with_candidates(dict(d))
    assert len(d2["candidates"]) == 2 and d2["ratio_floor"] == 0.01
    lines = key_causal_lines(dict(d))
    body = [ln for ln in lines if ln.strip().startswith(("ffn", "attn"))]
    assert len(body) == 2 and "★" in body[0] and "○" in body[0] and "○" not in body[1]
    assert any("α 4" in ln and "下がり方÷副作用" in ln for ln in lines) and any("でたらめに入れ替えて" in ln for ln in lines)
    # 並べ替え: attn L5 の副作用を小さく・下がり方を大きくすると ratio・effect・side の順で先に来る
    for r in rows:
        if r["target"] == "attn":
            r["reply"]["mean"] *= 3
            r["side"]["b"]["mean"] = -0.001
    cs = candidates_from_rows(rows, [1.0, 2.0, 4.0])
    for srt in ("ratio", "effect", "side"):
        assert [c["target"] for c in kc_sorted(cs, srt, 4.0)] == ["attn", "ffn"], srt
    assert [c["target"] for c in kc_sorted(cs, "place", 4.0)] == ["ffn", "attn"]
    with pytest.raises(GneError):
        kc_sorted(cs, "nope", 4.0)
    with pytest.raises(GneError):
        key_causal_lines({**d, "rows": rows, "candidates": cs}, ratio_alpha=3.0)
    lines = key_causal_lines({**d, "rows": rows, "candidates": cs}, ratio_alpha=1.0, sort="ratio")
    body = [ln for ln in lines if ln.strip().startswith(("ffn", "attn"))]
    assert body[0].strip().startswith("attn") and any("α 1)" in ln for ln in lines)


def test_parse_combos():
    from gne.keyed_causal import parse_combos
    assert parse_combos(None) == []
    assert parse_combos([["11:attn:keyed:a2", "18:attn:keyed:a2"]]) == [["11:attn:keyed:a2", "18:attn:keyed:a2"]]
    assert parse_combos(["11:attn:keyed:a2, 18:attn:keyed:a1.5  23:attn:keyed:a2"]) == [["11:attn:keyed:a2", "18:attn:keyed:a1.5", "23:attn:keyed:a2"]]
    assert parse_combos("8:ffn:keyed:a1,9:ffn:keyed:a1;10:ffn:keyed:a1,11:ffn:keyed:a1") == [["8:ffn:keyed:a1", "9:ffn:keyed:a1"],
                                                                                           ["10:ffn:keyed:a1", "11:ffn:keyed:a1"]]
    with pytest.raises(GneError) as ei:
        parse_combos(["8:ffn:keyed:a1"])
    assert ei.value.code == "BAD_ARGS"


def test_verify_counts_and_changed():
    import re

    from gne.keyed_causal import _changed, _counts
    reps = [{"text": "です。ます。", "finish": "eog", "loop": False, "label": "P"}, {"text": "", "finish": "length", "loop": False, "label": "N"},
            {"text": "のだ。", "finish": "eog", "loop": True, "label": "C"}]
    c = _counts(reps, re.compile("(です|ます)。"), None)
    assert (c["n"], c["match"], c["hits"], c["behavior"], c["empty"], c["loop"], c["at_limit"]) == (3, 1, 2, 1, 1, 1, 1)
    assert c["labels"] == {"P": 1, "N": 1, "C": 1}
    c = _counts(reps, None, ["P", "C"])
    assert c["behavior"] == 2 and "match" not in c
    base = [{"text": "abcd"}, {"text": ""}, {"text": "xyz"}]
    n, fr = _changed([{"text": "abXX"}, {"text": ""}, {"text": "xyz"}], base)
    assert n == 1 and fr == pytest.approx((0.5 + 1 + 1) / 3)


def test_job_wrappers_and_compact(tmp_path):
    from gne.jobs import _direction_key_causal, _direction_key_causal_verify, compact_key_causal, standard_kinds
    k = standard_kinds()
    assert k["direction_key_causal"] is _direction_key_causal and k["direction_key_causal_verify"] is _direction_key_causal_verify
    got = []
    s = SimpleNamespace(progress=None,
                        direction_key_causal=lambda c, **kw: got.append(("c", c, kw)) or {"rows": [_row("ffn", 1, 1.0, 0.0, [0.1, 0.2])]},
                        direction_key_causal_verify=lambda u, **kw: got.append(("v", u, kw)) or {"ok": 1})
    with pytest.raises(GneError):
        _direction_key_causal(s)
    with pytest.raises(GneError):
        _direction_key_causal(s, contrast="a", spec={"name": "x"})
    d = _direction_key_causal(s, spec={"name": "x"}, compact=True, layers="8")
    assert got[-1][1] == {"name": "x"} and got[-1][2]["layers"] == "8" and "values" not in d["rows"][0]["pseudo"]
    d = _direction_key_causal(s, contrast="c.yaml")
    assert got[-1][1] == "c.yaml" and "values" in d["rows"][0]["pseudo"]
    _direction_key_causal_verify(s, units=["8:ffn:keyed:a1"], seeds="1, 2", label="P,C", judge="  ")
    _, u, kw = got[-1]
    assert u == ["8:ffn:keyed:a1"] and kw["seeds"] == [1, 2] and kw["label"] == ["P", "C"] and kw["judge"] is None
    with pytest.raises(GneError):
        _direction_key_causal_verify(s, units=["x"], seeds="1,x")
    r = {"rows": [{"pseudo": {"values": [1], "min": 1}, "probe_pseudo": None, "pseudo_side": {"dt_min": {"values": [1]}}}],
         "candidates": [{"by_alpha": {"1": {"pseudo": {"values": [1], "min": 1}, "pseudo_side_b": {"values": [2]}, "ratio": 3.0}}}]}
    out = compact_key_causal(r)
    assert out["rows"][0] == {"pseudo": {"min": 1}, "probe_pseudo": None, "pseudo_side": {"dt_min": {}}}
    assert out["candidates"][0]["by_alpha"]["1"] == {"pseudo": {"min": 1}, "pseudo_side_b": {}, "ratio": 3.0}


def test_estimate(tmp_path):
    """見積もり: 測る回数 = 1 + 候補 × α の段 × (1 + 比較用の鍵)。返事を与えた対照集合は返事を作らない。第 2 段は条件 = 1 + 2 × 候補。"""
    from gne.estimate import _key_causal_cost, _key_causal_verify_cost
    cp = tmp_path / "kc.yaml"
    cp.write_text(yaml.safe_dump(_spec(), allow_unicode=True), encoding="utf-8")
    sp = SimpleNamespace(gen=lambda n, t, k=8: n * t * 0.01, setprefill=40.0, step=lambda k: 10.0)
    s = SimpleNamespace(meta=SimpleNamespace(n_layer=28), _engine_kw={"n_seq_max": 8},
                        direction_set=lambda n: (_ for _ in ()).throw(GneError("FILE_NOT_FOUND", "x")))
    a = _key_causal_cost(s, sp, {"contrast": str(cp), "layers": "8,14", "targets": "ffn,attn", "alphas": "1,2", "pseudo": 2})
    assert a["candidates"] == 4 and a["runs"] == 1 + 4 * 2 * 3 and a["replies_seconds"] == 0 and a["direction_seconds"] > 0
    assert a["seqs_per_run"] == 8 + 2 and a["seconds"] > a["runs"] * a["seconds_per_run"]
    b = _key_causal_cost(s, sp, {"contrast": str(cp)})                      # 既定: 全層(最終層を除く)× ffn,attn × 3 段 × (1 + 3)
    assert b["candidates"] == 2 * 27 and b["runs"] == 1 + 54 * 3 * 4
    d = {k: v for k, v in _spec().items() if k != "replies"}
    c = _key_causal_cost(s, sp, {"spec": d, "layers": "8", "targets": "ffn"})
    assert c["replies_seconds"] > 0 and c["candidates"] == 1
    assert _key_causal_cost(s, sp, {"contrast": str(tmp_path / "none.yaml")}).get("guess")
    v = _key_causal_verify_cost(s, sp, {"contrast": str(cp), "units": ["8:ffn:keyed:a1", "9:ffn:keyed:a1"], "max_tokens": 100})
    assert v["conditions"] == 5 and v["questions"] == 8 and v["dont_touch"] == 2 and v["seconds"] > 0
    v2 = _key_causal_verify_cost(s, sp, {"contrast": str(cp), "units": "8:ffn:keyed:a1", "dont_touch": False, "seeds": "1,2", "judge": "{q}{a}"})
    assert v2["conditions"] == 3 and v2["dont_touch"] == 0 and v2["seeds"] == 2
    v3 = _key_causal_verify_cost(s, sp, {"contrast": str(cp), "units": ["8:ffn:keyed:a1"], "combos": [["8:ffn:keyed:a1", "9:ffn:keyed:a1"]]})
    assert v3["conditions"] == 1 + 2 * 2 and v3["combos"] == 1


def test_mcp_tools_start_jobs(tmp_path, monkeypatch):
    """MCP の direction_key_causal・direction_key_causal_verify はジョブを投げる(見立てた Session、モデルは読まない)。対照集合は表・
    configs/keyed/ のひな形の名前・パス。結果は MCP 向けに小さくする(compact)。"""
    import gne.api
    import gne.mcp_server as ms
    started = []

    class FakeRunner:
        def __init__(self, *a, **k):
            pass

        def start(self, kind, params, **k):
            started.append((kind, params))
            return SimpleNamespace(view=lambda: {"id": "j1", "kind": kind})
    fake = SimpleNamespace(warnings=[], store=SimpleNamespace(root=tmp_path))
    monkeypatch.setattr(gne.api, "Session", lambda *a, **k: fake)
    monkeypatch.setattr(ms, "JobRunner", FakeRunner)
    srv = ms.build_server("m.gguf")

    async def go():
        names = {t.name for t in await srv.list_tools()}
        assert {"direction_key_causal", "direction_key_causal_verify"} <= names
        await srv.call_tool("direction_key_causal", {"contrast": {"name": "x", "user_prompts": ["a"]}, "alphas": [-1, -2], "layers": "all"})
        await srv.call_tool("direction_key_causal", {"contrast": "sentence_endings", "targets": "ffn", "probe": [{"prompt": "q", "behavior": "a", "desired": "b"}]})
        await srv.call_tool("direction_key_causal", {"contrast": str(ZUNDAMON), "no_iv": True})
        await srv.call_tool("direction_key_causal_verify", {"units": ["8:ffn:keyed:a1"], "result": "r", "label": "P", "dont_touch": False})
        await srv.call_tool("direction_key_causal_verify", {"combos": [["8:ffn:keyed:a1", "9:attn:keyed:a2"]], "result": "r"})
        bad = await srv.call_tool("direction_key_causal", {"contrast": "no_such_template"})
        return bad
    bad = anyio.run(go)
    k, p = started[0]
    assert k == "direction_key_causal" and p["spec"] == {"name": "x", "user_prompts": ["a"]} and p["alphas"] == [-1, -2]
    assert p["compact"] is True and p["pseudo"] == 3 and p["layers"] == "all" and "contrast" not in p
    k, p = started[1]
    assert p["spec"]["name"] == "noda_endings" and p["targets"] == "ffn" and p["probe"][0]["desired"] == "b"
    k, p = started[2]
    assert p["contrast"] == str(ZUNDAMON) and p["no_iv"] is True and "spec" not in p
    k, p = started[3]
    assert k == "direction_key_causal_verify" and p["units"] == ["8:ffn:keyed:a1"] and p["result"] == "r" and p["dont_touch"] is False
    assert p["max_tokens"] == 256 and "temperature" not in p and "combos" not in p
    k, p = started[4]
    assert k == "direction_key_causal_verify" and p["combos"] == [["8:ffn:keyed:a1", "9:attn:keyed:a2"]] and "units" not in p
    assert len(started) == 5 and "FILE_NOT_FOUND" in json.dumps(bad, default=str, ensure_ascii=False)


def test_cli_schemas_registered():
    from gne.schemas import COMMAND_SCHEMAS, DirectionKeyCausalOut, DirectionKeyCausalVerifyOut
    assert COMMAND_SCHEMAS["direction key-causal"] is DirectionKeyCausalOut
    assert COMMAND_SCHEMAS["direction key-causal-verify"] is DirectionKeyCausalVerifyOut


# --- モデルあり ----------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip(f"{MODEL} がありません")
    from gne.api import Session
    d = tmp_path_factory.mktemp("kc")
    s = Session(MODEL, root=d / "root", n_gpu_layers=NGL, n_ctx=2048)
    spec = _spec()
    cp = d / "kc_endings.yaml"
    cp.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
    s.direction(contrast=str(cp), name="kc_endings", positions="last")         # 返事を作らない方向(流れを確かめるため)
    probe = d / "probe.jsonl"
    rows = [{"prompt": spec["user_prompts"][1], "prefix": "おすすめは", "behavior": "ご飯", "desired": "パン"},
            {"prompt": spec["user_prompts"][3], "prefix": "", "behavior": "家", "desired": "外"},
            {"prompt": spec["user_prompts"][0], "prefix": "", "behavior": "公園", "desired": "家"},        # 作成に使った質問(外す)
            {"prompt": "What is two plus two?", "prefix": "Answer:", "behavior": " 5", "desired": " 4", "group": "english"}]
    probe.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    e = SimpleNamespace(s=s, contrast=cp, probe=probe, spec=spec, dir=d, closed=False)
    yield e
    if not e.closed:
        s.close()


@pytest.fixture(scope="module")
def stage1(env):
    t0 = time.time()
    r = env.s.direction_key_causal(str(env.contrast), layers="8,14", targets="ffn,attn", alphas="1,2", pseudo=2,
                                   probe=str(env.probe), name="kc1")
    r["_wall"] = time.time() - t0
    return r


@pytest.mark.model
def test_stage1_shape_marks_and_saved(env, stage1):
    from gne.keyed import load_key
    from gne.keyed_causal import KC_PSEUDO, NOTE, list_results, load_result
    r, s = stage1, env.s
    assert r["name"] == "kc1" and r["direction"] == "kc_endings" and r["targets"] == ["ffn", "attn"] and r["alphas"] == [1.0, 2.0]
    assert r["keys"] == {"ffn": "kc1-ffn", "attn": "kc1-attn"} and r["mode"] == "same"
    assert r["measure"] == {"regex": "(です|ます)。", "source": "positions"} and r["replies"] == {"AB": "given", "dt:english": "given"}
    for k in ("kc1-ffn", "kc1-attn"):
        ks = load_key(s, k)
        assert ks.layers == [8, 14] and ks.n_pseudo >= KC_PSEUDO and ks.subset == "even"
    assert r["n_candidates"] == 4 and r["n_runs"] == 1 + 4 * 2 * 3 and len(r["rows"]) == 8
    assert r["counts"]["a"]["examples"] == 8 and r["counts"]["a"]["matched"] == 8 and r["counts"]["dt"] == {"english": 2}
    assert r["notes"][0] == NOTE and any("5% 点は粗い" in n for n in r["notes"])            # 比較用の鍵は 4 候補 × 2 本 = 8 本
    assert any("positions last" in w for w in r["warnings"])                                  # 方向は positions last で作った
    for row in r["rows"]:
        assert row["unit"] == f"{row['layer']}:{row['target']}:keyed:a{row['alpha']:g}@kc_endings#kc1-{row['target']}"
        assert row["reply"]["n"] == 8 and len(row["pseudo"]["values"]) == 2 and row["key"] == f"kc1-{row['target']}"
        assert set(row["side"]["dt"]) == {"english"} and row["side"]["b"]["n"] == 8 and row["side"]["dt"]["english"]["n"] == 2
        assert row["key_check"]["selectivity"] is not None and row["key_check"]["kn"] is not None
        assert row["linear"] is None or "consistent" in row["linear"]
    for a in ("1", "2"):
        pl = r["pool"][a]["reply"]
        vals = [v for x in r["rows"] if f"{x['alpha']:g}" == a for v in x["pseudo"]["values"]]
        assert pl["n"] == 8 and pl["p5"] == pytest.approx(np.percentile(vals, 5))
        mk = r["marks"][a]["reply"]
        assert mk["candidates"] == 4 and mk["expected"] == pytest.approx(0.2)
        assert mk["marked"] == sum(bool(x["mark"]) for x in r["rows"] if f"{x['alpha']:g}" == a)
        for x in r["rows"]:
            if f"{x['alpha']:g}" == a:
                assert x["mark"] == (x["reply"]["mean"] < pl["p5"])
    assert set(r["key_check"]) == {"ffn", "attn"} and r["key_check"]["ffn"]["held_out"] is True
    from gne.keyed_causal import drop_per_harm
    assert r["ratio_floor"] == 0.01 and len(r["candidates"]) == 4                  # 1 か所(書き込み先 × 層)につき 1 つ
    for cd in r["candidates"]:
        mine = [x for x in r["rows"] if (x["target"], x["layer"]) == (cd["target"], cd["layer"])]
        assert set(cd["by_alpha"]) == {"1", "2"} and cd["all_marked"] == all(bool(x["mark"]) for x in mine)
        for x in mine:
            b = cd["by_alpha"][f"{x['alpha']:g}"]
            assert b["unit"] == x["unit"] and b["ratio"] == drop_per_harm(x["reply"], x["side"]["b"])
    assert r["base"]["reply"] < 0 and r["base"]["dt"]["english"] < 0
    assert Path(r["path"]).exists() and load_result(s, "kc1")["rows"] == r["rows"]
    assert any(x["name"] == "kc1" for x in list_results(s))
    print(f"\n第 1 段(CPU/GPU、4 候補 × 2 段 × 3 本 = {r['n_runs']} 回): {r['_wall']:.1f} 秒、1 回 {r['timing']['per_run']:.2f} 秒")


@pytest.mark.model
def test_stage1_probe(env, stage1):
    r = stage1
    assert r["counts"]["probe"] == {"a": 2, "dt": {"english": 1}, "excluded": 1}
    assert any("probe の 1 行は鍵の作成に使った質問" in n for n in r["notes"])
    assert "probe" in r["pool"]["1"] and r["marks"]["2"]["probe"]["candidates"] == 4
    for row in r["rows"]:
        assert row["probe"]["n"] == 2 and len(row["probe_pseudo"]["values"]) == 2
        assert row["side"]["probe_dt"]["english"]["n"] == 1 and row["side"]["probe_dt"]["english"]["abs_mean"] >= 0


@pytest.mark.model
def test_stage1_effect_matches_runtime_lora(env, stage1):
    """速い対数確率(列をまとめて読む)は、1 本ずつ読めば engine.token_logprobs と同じ値。行の効き目(返事)は、同じ列の並びを
    keyed の LoRA を実行時と同じように(レシピ → Session.lora_for)当てて読んだ値と合う(比較用の鍵にも使う一時的な LoRA が、
    実行時の LoRA と同じ書き込みをしている)。まとめて読む本数が変わると計算の順が変わり、Qwen3-0.6B Q8_0 では 1 トークンで 0.1 以上
    値が動くことがあるので、基準と候補はいつも同じ並びで読む(差を取るのはその同じ並びどうし)。"""
    from gne.keyed_build import read_key_spec
    from gne.keyed_causal import _effects, _Plan, _Scorer
    s, e = env.s, env.s.engine
    spec = read_key_spec(env.contrast)
    plan = _Plan(s, spec, mode="same", regex="(です|ます)。", skip=0, side_tokens=32, probe_rows=[],
                 side_iv={"A": (None, None), "B": (None, None)}, vecs0=None, lora0=None, built_questions=set())
    A = plan.sets[0]
    assert A.label == "A" and len(A.seqs) == 8
    sc = _Scorer(s)
    n0 = sc.ctx.n_seq_max
    try:                                                      # 1 本ずつ読めば token_logprobs と同じ
        sc.ctx.n_seq_max = 1
        got = sc.run(A.seqs[:3], A.wants[:3])
    finally:
        sc.ctx.n_seq_max = n0
    ctx = e.steer_ctx()
    for seq, g in zip(A.seqs[:3], got, strict=True):
        lp = e.token_logprobs(ctx, seq)
        for (pos, tok), v in g.items():
            assert seq[pos + 1] == tok and v == pytest.approx(float(lp[pos]), abs=1e-4)
    row = next(x for x in stage1["rows"] if (x["target"], x["layer"], x["alpha"]) == ("ffn", 14, 2.0))
    lora = s.lora_for(s.recipe(units=[row["unit"]]))
    base = plan.measure(sc, None)
    eff = _effects(plan, plan.measure(sc, lora), base)
    assert eff["reply"]["mean"] == pytest.approx(row["reply"]["mean"], abs=2e-3)
    assert eff["b"]["mean"] == pytest.approx(row["side"]["b"]["mean"], abs=2e-3)
    assert eff["dt"]["english"]["mean"] == pytest.approx(row["side"]["dt"]["english"]["mean"], abs=2e-3)


@pytest.mark.model
def test_stage1_errors(env):
    s = env.s
    with pytest.raises(GneError) as ei:                                       # system の状態の鍵で、正規表現も probe も無い
        s.direction_key_causal(str(ROOT / "configs" / "keyed" / "system_state.json"), layers="8")
    assert ei.value.code == "BAD_ARGS" and "measure_match" in (ei.value.hint or "")
    with pytest.raises(GneError) as ei:
        s.direction_key_causal(str(env.contrast), pseudo=0)
    assert ei.value.code == "BAD_ARGS"
    with pytest.raises(GneError) as ei:
        s.direction_key_causal(str(env.contrast), measure_match="(")
    assert ei.value.code == "BAD_ARGS"
    with pytest.raises(GneError) as ei:
        s.direction_key_causal(str(env.contrast), targets="ffn", layers="27")       # 最終層しか無い
    assert ei.value.code == "BAD_ARGS"


@pytest.mark.model
def test_stage2_verify_counts(env, stage1):
    from gne.keyed_causal import list_results
    s = env.s
    row = next(x for x in stage1["rows"] if (x["target"], x["layer"], x["alpha"]) == ("ffn", 14, 2.0))
    v = s.direction_key_causal_verify([row["unit"], "8:attn:keyed:a2"], result="kc1", max_tokens=10)
    assert v["name"] == "kc1" and v["n_a"] == 8 and v["n_dt"] == {"english": 2} and v["greedy"] and v["mode"] == "same"
    assert v["judge"]["method"] == "regex" and v["judge"]["match"] == "(です|ます)。" and v["judge"]["source"] == "key-causal"
    assert v["base"]["a"]["n"] == 8 and v["base"]["dt"]["english"]["n"] == 2
    assert [x["unit"] for x in v["rows"]] == [row["unit"], "8:attn:keyed:a2@kc_endings#kc1-attn"]
    assert v["rows"][1]["pseudo_unit"] == "8:attn:keyed:a2@kc_endings#kc1-attn~0"
    for x in v["rows"]:
        for which in ("keyed", "pseudo"):
            c = x["a"][which]
            assert c["n"] == 8 and 0 <= c["behavior"] <= 8 and c["changed"] is not None and 0 <= c["same_prefix"] <= 1
            assert x["dt"]["english"][which]["n"] == 2 and {"loop", "empty", "at_limit"} <= set(c)
        assert set(x["samples"]) == {"base", "keyed", "pseudo"} and len(x["samples"]["keyed"]) == 2
    full = json.loads(Path(v["path"]).read_text(encoding="utf-8"))
    assert set(full["conditions"]) == {"base", "c0", "c1", "p0", "p1"} and full["conditions"]["c1"] == [v["rows"][1]["unit"]]
    assert len(full["replies"]["c0"]["a"][0]["replies"]) == 8 and len(full["replies"]["base"]["dt"][0]["replies"]) == 2
    assert next(x for x in list_results(s) if x["name"] == "kc1")["verify"] is True


@pytest.mark.model
def test_stage2_judge_and_errors(env, stage1):
    s = env.s
    row = next(x for x in stage1["rows"] if (x["target"], x["layer"], x["alpha"]) == ("attn", 14, 1.0))
    v = s.direction_key_causal_verify([row["unit"]], result="kc1", max_tokens=6, dont_touch=False, name="kc1j",
                                      judge="質問: {q}\n返事: {a}\n丁寧語なら P、そうでなければ C とだけ答えて。", labels="P,C", label="P")
    assert v["judge"]["method"] == "label" and v["judge"]["labels"] == ["P"] and v["n_dt"] == {}
    c = v["rows"][0]["a"]["keyed"]
    assert c["behavior"] == c["labels"].get("P", 0) and sum(c["labels"].values()) == 8
    for bad, kw, word in (([row["unit"].replace("#kc1-attn", "#kc1-attn~0")], {}, "本物の鍵"),
                          (["14:7:+1"], {}, "keyed のユニットではありません"),
                          ([row["unit"]], {"judge": "{q}{a}"}, "label"),
                          ([], {}, "1 つ以上"),
                          ([row["unit"]], {"pseudo_index": 99}, "比較用の鍵 99")):
        with pytest.raises(GneError) as ei:
            s.direction_key_causal_verify(bad, result="kc1", **kw)
        assert ei.value.code == "BAD_ARGS" and word in ei.value.message, ei.value.message
    with pytest.raises(GneError) as ei:
        s.direction_key_causal_verify([row["unit"]])
    assert ei.value.code == "BAD_ARGS"


@pytest.mark.model
def test_stage2_combo(env, stage1):
    """組(複数のユニットを 1 つの候補)の確かめ: 組の各ユニットは第 1 段の方向・鍵で補い、比較用の組は各ユニットの鍵を同じ番号の
    比較用の鍵に替えたもの。強さは行ごとに違ってよい。1 か所の候補と並ぶ。同じ層・書き込み先が重なる組は断る。"""
    s = env.s
    v = s.direction_key_causal_verify(["14:ffn:keyed:a2"], result="kc1", combos=[["8:attn:keyed:a2", "14:attn:keyed:a1"]],
                                      max_tokens=6, dont_touch=False, name="kc1combo", pseudo_index=1)
    single, combo = v["rows"]
    assert single["combo"] is False and single["units"] == ["14:ffn:keyed:a2@kc_endings#kc1-ffn"]
    assert combo["combo"] is True and combo["units"] == ["8:attn:keyed:a2@kc_endings#kc1-attn", "14:attn:keyed:a1@kc_endings#kc1-attn"]
    assert combo["pseudo_units"] == ["8:attn:keyed:a2@kc_endings#kc1-attn~1", "14:attn:keyed:a1@kc_endings#kc1-attn~1"]
    assert combo["unit"] == " + ".join(combo["units"]) and combo["label"].startswith("attn L8・L14")
    assert combo["a"]["keyed"]["n"] == 8 and combo["a"]["pseudo"]["n"] == 8 and combo["layer"] is None
    full = json.loads(Path(v["path"]).read_text(encoding="utf-8"))
    assert full["conditions"]["c1"] == combo["units"] and full["conditions"]["p1"] == combo["pseudo_units"]
    for bad in ([["8:attn:keyed:a2", "8:attn:keyed:a1"]], [["8:attn:keyed:a2"]]):
        with pytest.raises(GneError) as ei:
            s.direction_key_causal_verify([], result="kc1", combos=bad)
        assert ei.value.code == "BAD_ARGS"


@pytest.mark.model
def test_cli_json_and_text(env, stage1):
    """CLI: gne direction key-causal・key-causal-verify の --json がスキーマを通る(負の α の段もそのまま渡せる)。表の文字には
    印・偶然でも印が付く数・生成なしの目安の注・次に打つコマンドが出る。"""
    from gne.cli import key_causal_lines, key_causal_verify_lines
    from tests.test_change07_cli import gne_json
    w = {"root": env.dir / "root", "dir": env.dir}
    d = gne_json(w, "direction", "key-causal", "--contrast", str(env.contrast), "--layers", "8", "--target", "ffn",
                 "--alphas", "-2", "--pseudo", "1", "--name", "kccli")
    assert d["command"] == "direction key-causal" and d["alphas"] == [-2.0] and len(d["rows"]) == 1 and d["keys"] == {"ffn": "kccli-ffn"}
    v = gne_json(w, "direction", "key-causal-verify", "--from", "kccli", "--unit", d["rows"][0]["unit"], "--max-tokens", "4",
                 "--no-dont-touch")
    assert v["command"] == "direction key-causal-verify" and v["n_dt"] == {} and v["rows"][0]["unit"] == d["rows"][0]["unit"]
    text = "\n".join(key_causal_lines(stage1))
    assert "偶然でも 4 × 5% = 0.2 個は印が付く" in text and "生成なしの目安" in text and "key-causal-verify --from kc1 --unit" in text
    body = [ln for ln in key_causal_lines(stage1) if ln.strip().startswith(("ffn", "attn"))]
    assert len(body) == 4 and "比較用の鍵" in text and "下がり方÷副作用" in text and "でたらめに入れ替えて" in text
    only = key_causal_lines(stage1, only_marked=True)
    n_unmarked = sum(not any(b.get("mark") or b.get("probe_mark") for b in c["by_alpha"].values()) for c in stage1["candidates"])
    assert len(only) == len(key_causal_lines(stage1)) - n_unmarked
    assert "iv のみ" in "\n".join(key_causal_verify_lines(v))
    w2 = gne_json(w, "direction", "key-causal-verify", "--from", "kc1", "--unit", "8:ffn:keyed:a1",
                  "--combo", "8:ffn:keyed:a1,14:attn:keyed:a2", "--max-tokens", "4", "--no-dont-touch", "--name", "kccombo")
    assert [x["combo"] for x in w2["rows"]] == [False, True] and w2["rows"][1]["units"] == ["8:ffn:keyed:a1@kc_endings#kc1-ffn",
                                                                                          "14:attn:keyed:a2@kc_endings#kc1-attn"]
    assert "組 " in "\n".join(key_causal_verify_lines(w2))


# --- ジョブ・サーバ・ブラウザ ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def api(env, stage1):
    from fastapi.testclient import TestClient

    from gne.ui.server import create_app
    from tests.test_change07_ui import _close_app
    app = create_app(env.s, warm=False)
    yield TestClient(app), app
    _close_app(app, env.s)          # 走っているジョブと画面からの読み込み(方向の値など)の終わりを GPU の調停で待ってから閉じる
    env.closed = True


def _run_job(c, kind, params, timeout=900):
    j = c.post("/api/jobs", json={"kind": kind, "params": params})
    assert j.status_code == 200, j.text
    j = j.json()
    stages = set()
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = next(x for x in c.get("/api/jobs").json() if x["id"] == j["id"])
        stages.add(v.get("stage") or "")
        if v["status"] in ("done", "error", "cancelled"):
            assert v["status"] == "done", v["error"]
            v["stages_seen"] = stages
            return v
        time.sleep(0.05)
    raise AssertionError(f"{kind} が終わりません")


@pytest.mark.model
def test_jobs_progress_estimate_cancel(api, env):
    """ジョブ direction_key_causal(入力欄の表 spec で)と direction_key_causal_verify: 結果・見積もり・段ごとの進み具合。走っている間に
    取り消すと cancelled。"""
    c, _ = api
    spec = json.loads(json.dumps(_spec()))
    v = _run_job(c, "direction_key_causal", {"spec": spec, "layers": "8", "targets": "ffn", "alphas": [2], "pseudo": 1, "name": "kcj"})
    r = v["result"]
    assert v["estimate_s"] and v["estimate_s"] > 0 and r["n_runs"] == 3 and len(r["rows"]) == 1 and r["name"] == "kcj"
    assert any(st.startswith("3/3 測る") for st in v["stages_seen"]) and any("鍵 ffn" in st for st in v["stages_seen"])
    assert (env.s.store.contrast_dir() / "kc_endings.yaml").exists()                     # 表は <root>/contrasts/ に書く
    w = _run_job(c, "direction_key_causal_verify", {"result": "kcj", "units": [r["rows"][0]["unit"]], "max_tokens": 6})
    assert w["estimate_s"] > 0 and w["result"]["rows"][0]["a"]["keyed"]["n"] == 8
    j = c.post("/api/jobs", json={"kind": "direction_key_causal",
                                  "params": {"spec": spec, "layers": "2-20", "targets": "ffn,attn", "name": "kc_cancel"}}).json()
    for _ in range(4000):
        x = next(y for y in c.get("/api/jobs").json() if y["id"] == j["id"])
        if x["status"] == "running" and x["done"] > 0:
            break
        time.sleep(0.02)
    c.delete(f"/api/jobs/{j['id']}")
    for _ in range(3000):
        x = next(y for y in c.get("/api/jobs").json() if y["id"] == j["id"])
        if x["status"] not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert x["status"] == "cancelled" and x["done"] < x["total"]


@pytest.mark.model
def test_server_endpoints(api, stage1):
    c, _ = api
    lst = c.get("/api/key-causal").json()
    assert any(x["name"] == "kc1" and x["targets"] == ["ffn", "attn"] for x in lst)
    one = c.get("/api/key-causal/kc1").json()
    assert len(one["rows"]) == 8 and one["notes"][0] == stage1["notes"][0]
    if "verify" in one:
        assert "replies" not in one["verify"]
    assert c.get("/api/key-causal/..%2Fx").status_code in (400, 404)
    r = c.get("/api/key-causal/no_such")
    assert r.status_code == 400 and r.json()["error"]["code"] == "NOT_FOUND"


def _free_port() -> int:
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


@pytest.fixture(scope="module")
def ui_server(api):
    if not CHROME.exists():
        pytest.skip(f"Chromium がありません({CHROME}、GNE_CHROME で指定)")
    import uvicorn
    _, app = api
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    th.join(5)


@pytest.fixture(scope="module")
def browser():
    if not CHROME.exists():
        pytest.skip(f"Chromium がありません({CHROME}、GNE_CHROME で指定)")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=str(CHROME), headless=True,
                              args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        yield b
        b.close()


def _key_tab(browser, url, lang, routes=()):
    pg = browser.new_page(viewport={"width": 1600, "height": 1100}, locale="ja-JP")
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("dialog", lambda dl: dl.accept())
    for pat, fn in routes:
        pg.route(pat, fn)
    pg.goto(url + f"/?lang={lang}")
    pg.wait_for_function("window.gne && window.gne.recipe && window.gne.arch", timeout=120000)
    pg.click("#views button[data-view=direction]")
    pg.wait_for_function("window.gne.directionView && window.gne.directionView.mode", timeout=60000)
    pg.click("#dv-mode button[data-m=key]")
    pg.wait_for_function("window.gne.directionView.key.loaded", timeout=60000)
    return pg, errors


@pytest.mark.model
def test_causal_panel_in_browser(ui_server, browser, env):
    """「効く層を探す」: 入力欄の JSON で第 1 段(ジョブ、進み具合)→ 表(印・偶然でも印が付く数・生成なしの目安の注)→ 行を選んで
    第 2 段(ジョブ)→ 結果の表 →「+ レシピ」で keyed のユニットがレシピに入る。保存した結果の一覧にも入る。"""
    pg, errors = _key_tab(browser, ui_server, "ja")
    pg.evaluate("window.gne.recipe.clear()")
    pg.click("#kx-src button[data-src=json]")
    pg.fill("#kx-text", json.dumps(_spec(), ensure_ascii=False))
    pg.wait_for_function("document.querySelector('#kx-jcheck').textContent.includes('kc_endings')", timeout=10000)
    assert pg.get_attribute("#kz-alphas", "placeholder") == "1,2,4" and pg.get_attribute("#kz-name", "placeholder") == "kc_endings-causal"
    pg.fill("#kx-text", json.dumps({**_spec(), "keyed": {"alpha": -1}}, ensure_ascii=False))   # keyed.alpha が負なら α の段も負
    pg.wait_for_function("document.querySelector('#kz-alphas').placeholder === '-1,-2,-4'", timeout=5000)
    pg.fill("#kx-text", json.dumps(_spec(), ensure_ascii=False))
    pg.fill("#kx-mt", "16")
    pg.fill("#kz-layers", "8,14")
    pg.uncheck("#kz-t-attn")
    pg.fill("#kz-alphas", "2")
    pg.fill("#kz-pseudo", "1")
    pg.fill("#kz-name", "kcweb")
    pg.click("#kz-go")
    pg.wait_for_selector("#kz-res table.kz-tab tbody tr", timeout=600000)
    assert "できました" in pg.evaluate("document.querySelector('#kz-status').textContent")
    res = pg.evaluate("document.querySelector('#kz-res').textContent")
    assert "偶然でも 2 × 5% = 0.1 個は印が付く" in res and "生成なしの目安" in res and "kcweb" in res
    assert pg.evaluate("document.querySelectorAll('#kz-res table.kz-tab tbody tr').length") == 2
    assert pg.evaluate("import('/jobs.js').then((m) => m.KIND.direction_key_causal)") == "効く層を探す(生成なし)"
    pg.wait_for_function("[...document.querySelectorAll('#kz-saved option')].some((o) => o.value === 'kcweb')", timeout=10000)
    assert pg.is_disabled("#kz-vgo")
    pg.check("#kz-res [data-kzsel='1']")
    assert pg.is_enabled("#kz-vgo") and "1 行を選んでいます" in pg.evaluate("document.querySelector('#kz-vsel').textContent")
    pg.fill("#kz-vmt", "6")
    pg.click("#kz-vgo")
    pg.wait_for_selector("#kz-vres table.kz-vtab tbody tr", timeout=600000)
    vres = pg.evaluate("document.querySelector('#kz-vres').textContent")
    assert "14:ffn:keyed:a2@kc_endings#kcweb-ffn" in vres and "english" in vres
    pg.click("#kz-res [data-kzadd='1']")
    assert pg.evaluate("window.gne.recipe.specs()") == ["14:ffn:keyed:a2@kc_endings#kcweb-ffn"]
    pg.check("#kz-onlymark")                                               # 印の付いた行だけ(描き直しても選んだ結果のまま)
    n_mark = sum(bool(x["mark"]) for x in pg.evaluate("window.gne.directionView.key.causal.result.rows"))
    assert pg.evaluate("document.querySelectorAll('#kz-res table.kz-tab tbody tr').length") == n_mark
    assert any(x["name"] == "kcweb" for x in env.s.key_causal_results())
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_causal_table_sort_strength_and_combo(ui_server, browser, stage1):
    """表: 1 か所 1 行で、強さごとの効き目の列と「全部の強さで有意」・下がり方 ÷ 副作用・副作用。見出しのクリックで並べ替え(2 回で
    逆)、強さの切り替えで比と副作用の列が変わる。比較用の鍵の説明が表の注に出る。2 行を選んで「まとめて 1 つの候補として確かめる」
    と、組の行と 1 か所ずつの行が結果の表に並ぶ。"""
    pg, errors = _key_tab(browser, ui_server, "ja")
    kz = "window.gne.directionView.key.causal"
    pg.evaluate(f"{kz}.open('kc1')")
    pg.wait_for_selector("#kz-res table.kz-tab tbody tr", timeout=10000)
    assert pg.evaluate("document.querySelectorAll('#kz-res table.kz-tab tbody tr').length") == 4
    heads = pg.evaluate("[...document.querySelectorAll('#kz-res table.kz-tab thead th')].map((x) => x.textContent)")
    assert "α +1.00" in heads and "α +2.00" in heads and "全部の強さで有意" in heads and "下がり方 ÷ 副作用" in heads and "副作用" in heads
    assert "でたらめに入れ替えて" in pg.evaluate("document.querySelector('#kz-res .kz-legend').textContent")
    assert pg.input_value("#kz-ralpha") == "2"
    cands = stage1["candidates"]

    def col(name):
        i = heads.index(name)
        return pg.evaluate(f"[...document.querySelectorAll('#kz-res table.kz-tab tbody tr')].map((tr) => tr.children[{i}].textContent.trim())")

    def places():
        return pg.evaluate("[...document.querySelectorAll('#kz-res table.kz-tab tbody tr')].map((tr) => tr.children[1].textContent + tr.children[2].textContent)")
    exp = [("—" if c["by_alpha"]["2"]["ratio"] is None else f"{c['by_alpha']['2']['ratio']:.1f}") for c in cands]
    assert col("下がり方 ÷ 副作用") == exp
    pg.select_option("#kz-ralpha", "1")
    exp1 = [("—" if c["by_alpha"]["1"]["ratio"] is None else f"{c['by_alpha']['1']['ratio']:.1f}") for c in cands]
    assert col("下がり方 ÷ 副作用") == exp1
    pg.click("#kz-res th[data-kzsort='layer']")                         # 層の小さい順(同じ層は元の順)
    assert places() == ["ffnL8", "attnL8", "ffnL14", "attnL14"]
    pg.click("#kz-res th[data-kzsort='layer']")                         # もう一度で逆
    assert places() == ["ffnL14", "attnL14", "ffnL8", "attnL8"]
    pg.click("#kz-res th[data-kzsort='ratio']")                         # 下がり方 ÷ 副作用の大きい順(— は後ろ)
    got = col("下がり方 ÷ 副作用")
    vals = [float(x) for x in got if x != "—"]
    assert vals == sorted(vals, reverse=True) and got[:len(vals)] == [f"{v:.1f}" for v in vals]
    # 2 行(ffn L14・attn L14)を選び、まとめて 1 つの候補として確かめる(強さ 1 のユニット)。並べ替えで行が動いても選んだものは残る
    pg.click("#kz-res th[data-kzsort='target']")
    rows = places()
    for i, p_ in enumerate(rows):
        if p_ in ("ffnL14", "attnL14"):
            pg.check(f"#kz-res table.kz-tab tbody tr:nth-child({i + 1}) input[data-kzsel]")
    assert pg.is_enabled("#kz-vcombo") and "強さ +1.00" in pg.evaluate("document.querySelector('#kz-vsel').textContent")
    pg.click("#kz-res th[data-kzsort='layer']")
    assert pg.evaluate("document.querySelectorAll('#kz-res input[data-kzsel]:checked').length") == 2
    pg.fill("#kz-vmt", "4")
    pg.uncheck("#kz-vdt")
    pg.click("#kz-vcombo")
    pg.wait_for_selector("#kz-vres table.kz-vtab tbody tr.kz-combo-row", timeout=600000)
    assert pg.evaluate("document.querySelectorAll('#kz-vres table.kz-vtab tbody tr').length") == 3
    combo = pg.evaluate("document.querySelector('#kz-vres tr.kz-combo-row').textContent")
    assert "14:ffn:keyed:a1@kc_endings#kc1-ffn" in combo and "14:attn:keyed:a1@kc_endings#kc1-attn" in combo
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_causal_late_loads_do_not_overwrite(ui_server, browser, env, stage1):
    """保存した結果を開く読み込みが遅れて届いても、後から開いた結果・後から届いたジョブの結果を上書きしない。"""
    from gne.keyed_causal import causal_dir
    d = json.loads(Path(stage1["path"]).read_text(encoding="utf-8"))
    (causal_dir(env.s) / "kcj.json").write_text(json.dumps({**d, "name": "kcj"}, ensure_ascii=False), encoding="utf-8")
    held = []

    def on_one(route, req):
        if req.url.endswith("/api/key-causal/kc1") and not held:
            held.append(route)
        else:
            route.continue_()
    pg, errors = _key_tab(browser, ui_server, "ja", routes=[("**/api/key-causal/*", on_one)])
    kz = "window.gne.directionView.key.causal"
    pg.evaluate(f"() => {{ {kz}.open('kc1'); }}")                 # 止めておく
    for _ in range(400):
        if held:
            break
        pg.wait_for_timeout(25)
    assert held
    pg.evaluate(f"{kz}.open('kcj')")
    pg.wait_for_function(f"{kz}.result && {kz}.result.name === 'kcj'", timeout=10000)
    with pg.expect_response("**/api/key-causal/kc1"):
        held[0].continue_()
    pg.wait_for_timeout(300)
    assert pg.evaluate(f"{kz}.result.name") == "kcj"
    # 開いている間に、いま投げたジョブ(と見立てたもの)の結果が届いたら、開いた結果は捨てる
    held.clear()
    pg.evaluate(f"() => {{ {kz}.open('kc1'); }}")
    for _ in range(400):
        if held:
            break
        pg.wait_for_timeout(25)
    pg.evaluate(f"""async () => {{ const r = await (await fetch('/api/key-causal/kcj')).json(); {kz}.jobs.run = 'fake1';
                   await {kz}.jobEnded({{id: 'fake1', kind: 'direction_key_causal', status: 'done', result: r}}); }}""")
    with pg.expect_response("**/api/key-causal/kc1"):
        held[0].continue_()
    pg.wait_for_timeout(300)
    assert pg.evaluate(f"{kz}.result.name") == "kcj"
    # 前に投げたジョブ(いまの id でないもの)の結果は表に出さない
    pg.evaluate(f"""async () => {{ const r = await (await fetch('/api/key-causal/kc1')).json();
                   await {kz}.jobEnded({{id: 'old1', kind: 'direction_key_causal', status: 'done', result: r}}); }}""")
    assert pg.evaluate(f"{kz}.result.name") == "kcj"
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_causal_panel_in_english(ui_server, browser, stage1):
    """英語の画面: 「効く層を探す」の欄と表(開いた結果)に日本語の文言が残らない。"""
    pg, errors = _key_tab(browser, ui_server, "en")
    assert pg.evaluate("document.querySelector('#kz-wrap h4').textContent").startswith("Find where a key works")
    pg.evaluate("window.gne.directionView.key.causal.open('kc1')")
    pg.wait_for_selector("#kz-res table.kz-tab tbody tr", timeout=10000)
    pg.check("#kz-res [data-kzsel='0']")
    pg.click("#kz-vwrap .kz-adv > summary")
    res = pg.evaluate("document.querySelector('#kz-res').textContent")
    assert "by chance alone" in res and "guide without generation" in res
    missing = pg.evaluate("[...window.gneI18n.missing]")
    assert missing == [], missing
    assert not errors, errors
    pg.close()
