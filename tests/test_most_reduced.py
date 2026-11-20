"""組の比較表の「A 側が最も下がった組」の印と、推す組が無いときの次の手(v0.9.0)。

推す規則(推奨)は変えない。A 側が最も下がった組は、元の方向・元の目標で測った A 側の射影の値の平均が最小の行
(残りから追加では合わせたレシピの a_mean_after)。同じ値なら部品の少ない行。応答での確かめで壊れた行は replies_break を添える。
モデル不要。
"""
from gne.causal import fill_most_reduced, mark_most_reduced, most_reduced_summary, refine_most_reduced


def _row(n, a_mean, **kw):
    return {"n_heads": n, "a_mean": a_mean, "units": [f"u{n}"], "recipe": f"r{n}.yaml", "remain": 1, "reached": True,
            "recommended": False, **kw}


def test_mark_most_reduced_picks_lowest_a_mean_and_ties_by_size():
    rows = [_row(6, -3.0), _row(4, -5.0), _row(2, -5.0), {"n_heads": 1, "skipped": "候補が足りない"}]
    best = mark_most_reduced(rows)
    assert best is rows[2] and [r.get("most_reduced") for r in rows] == [False, False, True, None]   # 同じ値なら部品の少ない行
    assert mark_most_reduced([{"n_heads": 1, "skipped": "x"}]) is None


def test_summary_counts_s1_row_and_keeps_recommendation():
    """倍率を上げる前の組(s1)も比べる。推す組の印は変えない。応答が壊れた行は replies_break。"""
    al = {"s1": {"n_heads": 4, "n_used": 4, "a_mean": -12.0, "units": ["s"], "recipe": "s1.yaml", "remain": 0, "reached": True},
          "table": [_row(4, -9.0, recommended=True), _row(2, -10.0, replies_break=True)]}
    out = {"allocate": al, "check": None}
    mr = most_reduced_summary(out)
    assert mr["s1"] and mr["a_mean"] == -12.0 and al["s1"]["most_reduced"] and not any(r["most_reduced"] for r in al["table"])
    assert al["table"][0]["recommended"] and not mr["recommended"]
    al["s1"]["a_mean"] = 5.0
    mr = most_reduced_summary(out)
    assert mr["n_heads"] == 2 and mr["replies_break"] and not al["s1"]["most_reduced"] and al["table"][1]["most_reduced"]
    # 応答での確かめ(check)でその組が壊れたと分かったときも
    al["table"][1].pop("replies_break")
    out["check"] = {"n_heads": 2, "break_warning": {"A": {"loop": 3}}}
    assert most_reduced_summary(out)["replies_break"]


def test_refine_rows_use_combined_a_mean():
    rows = [{"key": "s1", "n_heads": 8, "a_mean_after": -9.5, "verdict": "no_gain", "recipe": "a"},
            {"key": "3", "n_heads": 8, "a_mean_after": -11.0, "verdict": "replies_break", "recipe": "b"},
            {"key": "2", "n_heads": 7, "a_mean_after": -9.5, "verdict": "better", "recipe": "c"}]
    mr = refine_most_reduced(rows, "2", True)
    assert mr["key"] == "3" and mr["replies_break"] and not mr["recommended"] and rows[1]["most_reduced"]
    rows[1]["a_mean_after"] = -9.5
    mr = refine_most_reduced(rows, "2", True)
    assert mr["key"] == "2" and mr["recommended"]                                    # 同じ値なら部品の少ない行(推す行と同じ)


def test_fill_old_results_once():
    """v0.9.0 より前の結果: 開くときに印を付ける。印のある結果は付け直さない。"""
    old = {"allocate": {"table": [_row(4, -2.0), _row(2, -4.0)]},
           "refine": {"combined": None, "combined_rows": [{"key": "2", "n_heads": 3, "a_mean_after": 1.0}]}}
    x = fill_most_reduced(old)
    assert x["most_reduced"]["n_heads"] == 2 and x["refine"]["most_reduced"]["key"] == "2" and not x["refine"]["most_reduced"]["recommended"]
    x["allocate"]["table"][0]["most_reduced"] = True                                  # 付いていれば触らない
    x["allocate"]["table"][1]["most_reduced"] = False
    assert fill_most_reduced(x)["allocate"]["table"][0]["most_reduced"]


def test_cli_marks_and_next_step():
    """CLI: 行の印(壊れた組は添える)と、推す組・推す行が無いときの次の手(A 側が最も下がった組と強さの調整)。"""
    from gne.cli import REC_GUIDE, _mr_line, _mr_mark
    assert _mr_mark({"most_reduced": True}) == "  ← A 側が最も下がった"
    assert "応答が壊れた" in _mr_mark({"most_reduced": True, "replies_break": True})
    assert "応答が壊れた" in _mr_mark({"most_reduced": True, "verdict": "replies_break"})
    assert _mr_mark({"most_reduced": False}) == "" and _mr_mark(None) == ""
    line = _mr_line({"n_heads": 6, "a_mean": -3.25, "s1": False})
    assert "6 個の組" in line and "-3.25" in line and "gne recipe-scale" in line and "この強さをレシピに反映" in line
    assert "倍率を上げる前の組(s=1)" in _mr_line({"s1": True, "a_mean": 1.0})
    assert "上限 2 の行の合わせたレシピ" in _mr_line({"key": "2", "a_mean_after": 0.5}, refine=True)
    assert "強さを下げてから" in _mr_line({"n_heads": 2, "a_mean": 0.0, "replies_break": True})
    assert "目安" in REC_GUIDE and "応答は見ていない" in REC_GUIDE


def test_cli_writes_pipeline_text_without_recommendation():
    """heads pipeline(rank writes)の CLI の表示: 推す組が無いとき、表の行に「A 側が最も下がった」(壊れた組は添える)、
    次の手に最も下がった組と強さの調整、「推す組は目安」の一文。"""
    from gne.cli import _pipeline_writes_lines

    def row(n, a, **kw):
        return {"n_heads": n, "n_used": n, "units": [f"12:H{n}:s2@d"], "kinds": {"attn": n}, "s_max": 2.0, "s_min": 2.0, "s_median": 2.0,
                "alpha": 2.0, "stop": "s_max", "reached": False, "remain": 3, "a_mean": a, "b_side_delta": 0.1, "b_drift_orth": 0.05,
                "effect": 1.0, "recipe": f"r{n}.yaml", "recommended": False, **kw}
    al = {"pool": 4, "base_a_mean": 10.0, "strength": "per_unit", "n_a": 6, "notes": [],
          "s1": {**row(4, 6.0), "alpha": 1.0, "s_max": 1.0, "s_min": 1.0}, "table": [row(4, 2.0, replies_break=True), row(2, 4.0)]}
    d = {"path": "x.pipeline.json", "n_prompts": 6, "seconds": 3.0, "name": "d", "direction": {"positions": "last", "seconds": 1.0},
         "score": {"base_layer": 12}, "target": 0.0, "target_from": "mid", "recommended": None, "allocate": al,
         "writes": {"totals_share": {"attn": 0.5}, "kinds": ["attn"], "select_layer": 14, "n": {"A": 6, "B": 6}, "gap": 10.0,
                    "n_units": {"attn": 16}, "seconds": 1.0},
         "verify": {"n": 4, "mode": "proj", "kinds": {"attn": 4}, "seconds": 1.0,
                    "spearman": {"all": {"n": 4, "score_vs_gap_drop": 0.5, "score_vs_a_drop": 0.4}}},
         "no_recommend": {"reason": "どの組も、A 側の下がりが B 側の動きより小さい", "next": ["rank causal"]}}
    d["most_reduced"] = most_reduced_summary(d)
    txt = "\n".join(_pipeline_writes_lines(d))
    assert "上限 4 個" in txt and "← A 側が最も下がった(応答での確かめで応答が壊れた" in txt
    assert txt.count("← A 側が最も下がった") == 1
    assert "推す組なし: 道具の基準では 1 つに決められなかった(どの組も、A 側の下がりが B 側の動きより小さい)" in txt
    assert "A 側が最も下がったのは、4 個の組(A 平均 2.00)" in txt and "gne recipe-scale" in txt and "強さを下げてから使う" in txt
    assert "推す組は目安(合否の判定ではない)" in txt
