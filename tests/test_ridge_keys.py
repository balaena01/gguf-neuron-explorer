"""最小二乗の鍵(key_fit ridge、gne/keyed_ridge.py)の式: モデルを使わない(小さな合成データ)。

- 閉じた式の答えと合うこと(重みを 1 行ずつ付けて、行列をそのまま解いた答えと比べる)
- ますごとの重みが効くこと(行の数の少ない群を見分ける。位置を全部同じ重みで数えると見分けられない)
- 正則化の強さの決め方(質問単位の分割。別の書き方で数え直した値と同じ c・同じ点数になる)
- 比較用の鍵、行の間引き、位置の区分
- 入口(fish.keys_step・pipeline の引数)、要約、gne nw check の群ごとの漏れの表、ひな形の群

モデルあり(Qwen3-0.6B、GNE_TEST_MODEL。CPU で回すときは GNE_TEST_NGL=0):
- 鍵の計算に渡す行が、返事を作り直して 1 本ずつ読んだ MLP の入力と同じ位置・同じ値で、保存した鍵が閉じた式の答えと合うこと
- 入口ごとの引数(CLI の gne nw key / direction key / nw pipeline / nw check、ジョブ、見積もり)、2 段、比較用の鍵、
  既定の作り方の結果の形が今までと同じこと
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from gne import keyed_ridge as KR
from gne.steer import GneError


def _table(rows):
    """rows = [(質問, 群, 位置, ω)]。質問の番号は群ごとに続いている前提。"""
    q = np.array([r[0] for r in rows])
    g = np.array([r[1] for r in rows])
    q_group = np.zeros(int(q.max()) + 1, dtype=np.int64)
    q_group[q] = g
    return {"q": q, "pos": np.array([r[2] for r in rows]), "om": np.array([r[3] for r in rows], dtype=np.float64), "q_group": q_group}


def _data(seed=0, n_t=12, n_big=10, n_small=6, D=8, long=30):
    """狙う場面(群 0)、行の多い群(群 1。長い返事)、行の少ない群(群 2。返事は位置 0 だけ)。
    狙う場面の位置 0 の状態は e0 + e1、群 2 の位置 0 は e0 だけ(狙う場面と半分同じ)、ほかの位置は e2・e3 の向き。"""
    rng = np.random.default_rng(seed)
    rows, X = [], []
    e = np.eye(D)
    qi = 0
    for _ in range(n_t):
        for k in range(6):
            rows.append((qi, 0, k, 1.0))
            X.append((e[0] + e[1] if k == 0 else e[2] + 0.3 * e[4 + k % 3]) + 0.05 * rng.standard_normal(D))
        qi += 1
    for _ in range(n_big):
        for k in range(long):
            rows.append((qi, 1, k, 1.0))
            X.append(e[3] + 0.3 * e[4 + k % 4] + 0.05 * rng.standard_normal(D))
        qi += 1
    for _ in range(n_small):
        rows.append((qi, 2, 0, 1.0))
        X.append(e[0] + 0.05 * rng.standard_normal(D))
        qi += 1
    return _table(rows), np.asarray(X)


RANGES = [(0, 1), (1, 4), (4, None)]


def _weights(tab, ranges, e, cells=True):
    """鍵 e の行の重みを、式のとおりに 1 行ずつ付ける(W の合計 1、U の合計 1。U はますごとに等分)。"""
    q, pos, om, qg = tab["q"], tab["pos"], tab["om"], tab["q_group"]
    g = qg[q]
    bins = KR.bins_of(ranges)
    b = np.array([next(i for i, (lo, hi) in enumerate(bins) if p >= lo and (hi is None or p < hi)) for p in pos])
    kb = bins.index(ranges[e])
    W = (g == 0) & (b == kb)
    U = ~W
    w = np.zeros(len(q))
    w[W] = om[W] / om[W].sum()
    if cells:
        cell = g * len(bins) + b
        cs = np.unique(cell[U])
        for c in cs:
            m = U & (cell == c)
            w[m] = om[m] / om[m].sum() / len(cs)
    else:
        w[U] = om[U] / om[U].sum()
    return W, w


def _direct(X, W, w, c):
    D = X.shape[1]
    G = (X * w[:, None]).T @ X
    b = (w[W][:, None] * X[W]).sum(0)
    k = np.linalg.solve(G + c * np.trace(G) / D * np.eye(D), b)
    return k / (k @ (b / w[W].sum()))


def test_closed_form_matches_direct_solution():
    tab, X = _data()
    r = KR.fit(lambda L: X, [3], tab, RANGES, c=1.0)
    for e in range(3):
        W, w = _weights(tab, RANGES, e)
        assert abs(w[W].sum() - 1) < 1e-12 and abs(w[~W].sum() - 1) < 1e-12
        np.testing.assert_allclose(r["kappa"][e][3], _direct(X, W, w, 1.0), rtol=1e-9, atol=1e-12)
        mu = r["mu"][e][3]
        np.testing.assert_allclose(mu, (w[W][:, None] * X[W]).sum(0), rtol=1e-12)
        assert abs(r["kappa"][e][3] @ mu - 1) < 1e-12          # 書き込みたい位置での反応の平均が 1
    info = r["info"]
    assert info["c"] == 1.0 and not info["cv"] and not info["edge"] and info["cs"] is None
    # ますの数: 狙う場面のほかの 2 区分 + 群 1 の 3 区分 + 群 2 の位置 0
    assert info["cells"] == [6, 6, 6] and info["bins"] == ["0", "1-3", "4-"]
    assert info["rows_by_cell"]["2"] == {"0": 6} and info["rows_want"] == [12, 36, 24]


def test_float16_rows_give_the_same_key():
    tab, X = _data()
    a = KR.fit(lambda L: X.astype(np.float16), [0], tab, RANGES, c=1.0)["kappa"][0][0]
    b = KR.fit(lambda L: X.astype(np.float16).astype(np.float64), [0], tab, RANGES, c=1.0)["kappa"][0][0]
    np.testing.assert_array_equal(a, b)


def test_cell_weights_tell_a_small_group_apart():
    """行の少ない群(群 2: 6 行)の位置 0 は、狙う場面の位置 0 と半分同じ状態。位置を全部同じ重みで数えると、行の多い群(1,200 行)に
    重みを取られて、群 2 での反応が残る。ますごとの重みなら 0 の近くに抑えられ、狙う場面の反応は 1 のまま。"""
    tab, X = _data(n_big=40)
    g = tab["q_group"][tab["q"]]
    small = g == 2
    want = (g == 0) & (tab["pos"] == 0)
    cell = KR.fit(lambda L: X, [0], tab, RANGES, c=0.05)["kappa"][0][0]
    unif = KR.fit(lambda L: X, [0], tab, RANGES, c=0.05, weights="uniform")["kappa"][0][0]
    r_cell, r_unif = X[small] @ cell, X[small] @ unif
    assert r_unif.mean() > 0.4, r_unif.mean()
    assert abs(r_cell.mean()) < 0.2, r_cell.mean()
    assert abs((X[want] @ cell).mean() - 1) < 0.05
    W, w = _weights(tab, RANGES, 0, cells=False)
    np.testing.assert_allclose(unif, _direct(X, W, w, 0.05), rtol=1e-9, atol=1e-12)


def _cv_reference(X, tab, ranges, cs, folds, seed):
    """質問単位の分割での点数を、行の重みを 1 行ずつ付ける書き方で数え直す。"""
    q, qg = tab["q"], tab["q_group"]
    rng = np.random.default_rng(KR.SEED0 + seed)
    fq = np.zeros(len(qg), dtype=int)
    for gi in range(int(qg.max()) + 1):
        idx = np.flatnonzero(qg == gi)
        fq[idx[rng.permutation(len(idx))]] = np.arange(len(idx)) % folds
    fold = fq[q]
    S = np.zeros(len(cs))
    D = X.shape[1]
    for e in range(len(ranges)):
        W, w = _weights(tab, ranges, e)
        Xw = X * np.sqrt(w)[:, None]
        G = Xw.T @ Xw
        for f in range(folds):
            h = fold == f
            tw = W & ~h
            Gt = G - Xw[h].T @ Xw[h]
            b = (w[tw][:, None] * X[tw]).sum(0)
            m = b / w[tw].sum()
            for j, c in enumerate(cs):
                k = np.linalg.solve(Gt + c * np.trace(Gt) / D * np.eye(D), b)
                r = X[h] @ (k / (k @ m))
                S[j] += (w[h] * np.where(W[h], r < 0.8, r > 0.4)).sum()
    return S


def test_regularization_is_chosen_by_question_folds():
    tab, X = _data(seed=3)
    X = X + 0.4 * np.random.default_rng(1).standard_normal(X.shape)       # 分割で差が出るように、ばらつきを足す
    r = KR.fit(lambda L: X, [0], tab, RANGES)
    info = r["info"]
    assert info["cv"] and info["folds"] == 5 and info["cs"] == list(KR.CS) and info["thresholds"] == [0.4, 0.8]
    S = _cv_reference(X, tab, RANGES, KR.CS, 5, 0)
    np.testing.assert_allclose(info["score"], S, rtol=1e-9, atol=1e-12)
    best = max(j for j in range(len(S)) if S[j] <= S.min() + 1e-12)       # 点数が最小。同じなら大きいほう
    assert info["c"] == KR.CS[best]
    assert info["edge"] == (best in (0, len(KR.CS) - 1))
    np.testing.assert_allclose(np.add(info["score_want"], info["score_unwant"]), info["score"])
    # 選ばれた c の鍵は、その c を直接渡した鍵と同じ
    same = KR.fit(lambda L: X, [0], tab, RANGES, c=info["c"])
    for e in range(3):
        np.testing.assert_allclose(r["kappa"][e][0], same["kappa"][e][0], rtol=1e-9, atol=1e-12)
    # 分割は質問単位(同じ質問の位置は同じ組)で、種で決まる
    again = KR.fit(lambda L: X, [0], tab, RANGES)
    assert again["info"]["score"] == info["score"]
    other = KR.fit(lambda L: X, [0], tab, RANGES, seed=5)
    assert other["info"]["score"] != info["score"]


def test_edge_is_reported():
    tab, X = _data()
    r = KR.fit(lambda L: X, [0], tab, RANGES, cs=[1.0, 10.0])
    assert r["info"]["edge"] is True and r["info"]["c"] in (1.0, 10.0)        # 候補が 2 つなら、どちらも端


def test_few_target_questions():
    rows = [(0, 0, k, 1.0) for k in range(5)] + [(1, 1, k, 1.0) for k in range(5)]
    tab = _table(rows)
    X = np.random.default_rng(0).standard_normal((10, 4))
    with pytest.raises(GneError) as ei:
        KR.fit(lambda L: X, [0], tab, [(0, 1)])
    assert ei.value.code == "BAD_CONTRAST" and "key_c" in (ei.value.hint or "")
    assert KR.fit(lambda L: X, [0], tab, [(0, 1)], c=1.0)["kappa"][0][0].shape == (4,)
    # 狙う場面が 3 問なら 3 つに分ける
    rows = [(i, 0, k, 1.0) for i in range(3) for k in range(5)] + [(3, 1, k, 1.0) for k in range(5)]
    X = np.random.default_rng(0).standard_normal((20, 4))
    assert KR.fit(lambda L: X, [0], _table(rows), [(0, 1)])["info"]["folds"] == 3


def test_comparison_keys_use_the_same_fit():
    tab, X = _data()
    r = KR.fit(lambda L: X, [0, 1], tab, RANGES, pseudo=3)
    c = r["info"]["c"]
    direct = KR.fit(lambda L: X, [0, 1], tab, RANGES, c=c, pseudo=3)       # c を渡しても、比較用の鍵は同じ
    from gne.keyed_build import balanced_split
    q, pos, om, g = tab["q"], tab["pos"], tab["om"], tab["q_group"][tab["q"]]
    for e, (lo, hi) in enumerate(RANGES):
        C = r["ctrl"][e][0]
        assert C.shape == (3, X.shape[1])
        np.testing.assert_allclose(C, direct["ctrl"][e][0], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(r["ctrl"][e][1], C)                    # 同じ行を渡したので、層が違っても同じ
        inside = (g == 0) & (pos >= lo) & ((pos < hi) if hi is not None else True)
        qa, qb = np.unique(q[inside]), np.unique(q[(g == 0) & ~inside])
        rng = np.random.default_rng(0)
        for j in range(3):
            ia, _ = balanced_split(rng, len(qa), len(qb))
            W = np.zeros(len(q), dtype=bool)
            for i in ia:
                W |= (inside & (q == qa[i])) if i < len(qa) else ((g == 0) & ~inside & (q == qb[i - len(qa)]))
            # 比較用の鍵: ラベルを入れ替えた W で、同じ式(ますごとの重み・同じ c)
            bins = KR.bins_of(RANGES)
            b = np.array([next(k for k, (l2, h2) in enumerate(bins) if p >= l2 and (h2 is None or p < h2)) for p in pos])
            cell = g * len(bins) + b
            w = np.zeros(len(q))
            w[W] = om[W] / om[W].sum()
            cs = np.unique(cell[~W])
            for cc in cs:
                m = ~W & (cell == cc)
                w[m] = om[m] / om[m].sum() / len(cs)
            np.testing.assert_allclose(C[j], _direct(X, W, w, c), rtol=1e-8, atol=1e-10)
        assert not np.allclose(C[0], r["kappa"][e][0])
    assert KR.fit(lambda L: X, [0], tab, RANGES, c=1.0)["ctrl"] == [{}, {}, {}]


def test_bins():
    assert KR.bins_of([(0, 1), (1, 4), (4, None)]) == [(0, 1), (1, 4), (4, None)]
    assert KR.bins_of([(1, 4)]) == [(0, 1), (1, 4), (4, None)]             # 区間の外は、続いた範囲ごとに別の区分
    assert KR.bins_of([(4, None), (0, 1)]) == [(0, 1), (1, 4), (4, None)]
    assert KR.bins_of([(2, 3), (6, 8)]) == [(0, 2), (2, 3), (3, 6), (6, 8), (8, None)]
    assert [KR.bin_text(b) for b in KR.bins_of([(1, 4)])] == ["0", "1-3", "4-"]
    with pytest.raises(GneError):
        KR.bins_of([(0, 3), (2, 5)])
    with pytest.raises(GneError):
        KR.bins_of([(0, None), (4, None)])


def test_keep_positions():
    ks, w = KR.keep_positions(10, np.random.default_rng(0))
    assert ks.tolist() == list(range(10)) and set(w) == {1.0}
    ks, w = KR.keep_positions(24, np.random.default_rng(0))               # 16 以降が 8 位置ちょうど: 全部
    assert ks.tolist() == list(range(24)) and set(w) == {1.0}
    ks, w = KR.keep_positions(97, np.random.default_rng(0))
    assert ks[:16].tolist() == list(range(16)) and len(ks) == 24 and all(16 <= k < 97 for k in ks[16:])
    assert sorted(ks[16:].tolist()) == ks[16:].tolist() and len(set(ks.tolist())) == 24
    assert set(w[:16]) == {1.0} and set(w[16:]) == {81 / 8}
    assert abs(w.sum() - 97) < 1e-9                                       # 重みの合計は、もとの位置の数


def test_row_store():
    st = KR.RowStore([2, 5], seed=0)
    st.begin("AB", 2)
    X = {L: np.arange(40 * 3, dtype=np.float32).reshape(40, 3) + L for L in (2, 5)}
    st.add(0, 7, 4, lambda L, sel: X[L][sel])         # 返事 4 トークン: 位置 0〜4(列 6〜10)
    st.add(1, 5, 30, lambda L, sel: X[L][sel])        # 返事 30 トークン: 位置 0〜15 と、16〜30 から 8 位置
    st.begin("dt:x", 1)
    st.add(0, 3, 2, lambda L, sel: X[L][sel])
    t = st.table()
    assert st.groups == ["AB", "dt:x"] and t["q_group"].tolist() == [0, 0, 1]
    assert t["q"].tolist() == [0] * 5 + [1] * 24 + [2] * 3 and st.n_rows == 32
    assert t["pos"][:5].tolist() == [0, 1, 2, 3, 4] and t["om"][:5].tolist() == [1.0] * 5
    assert t["om"][5 + 16:5 + 24].tolist() == [15 / 8] * 8
    x2 = st.take(2)
    assert x2.dtype == np.float16 and x2.shape == (32, 3)
    np.testing.assert_array_equal(x2[:5], X[2][6:11])
    np.testing.assert_array_equal(x2[5:21], X[2][4:20])                   # 質問 1 の位置 0〜15 = 列 4〜19
    np.testing.assert_array_equal(x2[29:], X[2][2:5])
    # 乱数は質問の通し番号で決まる: 同じ質問は、いつも同じ位置
    st2 = KR.RowStore([2], seed=0)
    st2.begin("AB", 2)
    st2.add(1, 5, 30, lambda L, sel: X[L][sel])
    assert st2.table()["pos"].tolist() == t["pos"][5:29].tolist()
    # float16 に収まらない値がある層は float32 で持つ
    st3 = KR.RowStore([0])
    st3.begin("AB", 1)
    st3.add(0, 2, 1, lambda L, sel: np.full((2, 3), 1e5, np.float32))
    assert st3.wide == {0} and st3.take(0).dtype == np.float32


def test_arguments():
    assert KR.check_key_fit(None) == "mean" and KR.check_key_fit("ridge") == "ridge"
    with pytest.raises(GneError):
        KR.check_key_fit("lasso")
    assert KR.check_c(None) is None and KR.check_c("") is None and KR.check_c("0.1") == 0.1
    for bad in (0, -1, "x", float("nan")):
        with pytest.raises(GneError):
            KR.check_c(bad)
    assert KR.estimate_seconds(14000, 3000, 2560, 13, 3, pseudo=4) > KR.estimate_seconds(14000, 3000, 2560, 13, 3, cv=False) > 0


# --- 入口・要約・gne nw check の群ごとの数(モデルなし)-----------------------------------------------------------

def _q(i, totals, pos0):
    """check の質問ごとの行(区間 0 / 1- の合計 total と、位置 0 の合計)。"""
    return {"i": i, "prompt": f"q{i}", "reply_start": "", "n_tokens": 9, "loop": False, "pos0_total": pos0,
            "segments": {sg: None if t is None else {"amount": t, "total": t, "rho": 0.0, "n_pos": 1} for sg, t in zip(("0", "1-"), totals)}}


def test_leak_table_counts_per_group():
    from gne import nwrite as NW
    groups = {"scene": {"questions": [_q(0, (1.0, 0.9), 1.0)]},
              "dont_touch:tasks": {"questions": [_q(0, (0.01, 0.0), 0.01), _q(1, (0.05, 0.1), 0.05)]},
              "dont_touch:report": {"questions": [_q(0, (0.0, 0.35), 0.0), _q(1, (0.3, -0.25), 0.3), _q(2, (0.1, None), 0.1)]}}
    lk = NW.leak_table(groups, ["0", "1-"], 0.2)
    assert lk["leak"] == 0.2 and lk["segments"] == ["0", "1-"] and [g["group"] for g in lk["groups"]] == ["tasks", "report"]
    tasks, report = lk["groups"]
    assert (tasks["n"], tasks["any"], tasks["pos0"]["leak"]) == (2, 0, 0)
    assert (report["n"], report["any"]) == (3, 2)                                 # 区間 1- の 0.35 と、区間 0 の 0.3(-0.25 も大きさで数える)
    assert report["segments"]["0"] == {"n": 3, "leak": 1, "total_mean": pytest.approx(0.4 / 3), "abs_max": pytest.approx(0.3)}
    assert report["segments"]["1-"]["n"] == 2 and report["segments"]["1-"]["leak"] == 2 and report["segments"]["1-"]["total_mean"] == pytest.approx(0.05)
    assert report["pos0"]["leak"] == 1 and report["pos0"]["n"] == 3
    assert lk["all"]["n"] == 5 and lk["all"]["any"] == 2 and lk["all"]["group"] == "all"
    # by_question(区間ごとの「書いてしまった」数)と同じ数え方
    for g, row in (("dont_touch:tasks", tasks), ("dont_touch:report", report)):
        bq = NW._question_summary(groups[g]["questions"], ["0", "1-"], False, 0.4, 1.2, 0.2)
        assert {sg: c["leak"] for sg, c in bq.items()} == {sg: row["segments"][sg]["leak"] for sg in ("0", "1-")}
    text = "\n".join(NW.leak_lines({"leaks": lk}))
    assert "変えたくない場面の群ごとの漏れ(狙った量の 0.2 以上を書いた質問の数 / 質問の数" in text and "位置 0 だけ" in text
    assert "report" in text and "2/3" in text and "1/3(+0.13)" in text and "(全部)" in text
    assert NW.leak_lines({"leaks": None}) == [] and NW.leak_lines({}) == []
    one = NW.leak_table({"scene": groups["scene"], "dont_touch:tasks": groups["dont_touch:tasks"]}, ["0", "1-"], 0.2)
    assert "(全部)" not in "\n".join(NW.leak_lines({"leaks": one}))              # 群が 1 つなら、合わせた行は出さない
    assert NW.leak_table({"scene": groups["scene"]}, ["0"], 0.2) == {"leak": 0.2, "segments": ["0"], "groups": [], "all": None}
    # 質問ごとの値の要約の最後に、群ごとの漏れの表が付く
    qs = {"low": 0.4, "over": 1.2, "leak": 0.2, "segments": ["0", "1-"], "cap": [], "n_low": 0, "worst": []}
    lines = NW.question_lines({"question_summary": qs, "groups": {g: {**x, "by_question": {}} for g, x in groups.items()}, "leaks": lk})
    assert lines[-1].startswith("  群ごとに見ます") and any("変えたくない場面の群ごとの漏れ" in x for x in lines)
    # 鍵の作り方の 1 行(gne nw check)
    assert NW.key_fit_line(None) == NW.key_fit_line({"fit": "mean"}) == "鍵の作り方: mean(いまの作り方)"
    assert NW.key_fit_line({"fit": "ridge", "c": {"a_p0": 1.0, "a_p1": 1.0}}) == "鍵の作り方: ridge(最小二乗の鍵、ますごとの重み。正則化の強さ c = 1)"
    assert NW.key_fit_line({"fit": "ridge", "c": {"a": 0.1, "b": 10.0}}).endswith("c = 0.1・10)")
    assert "混ざって" in NW.key_fit_line({"fit": "mixed", "keys": {"a": "mean", "b": "ridge"}})


def test_summary_shows_key_fit_and_leaks():
    from gne import fish
    rec = {"c": 0.1, "cv": True, "cs": list(KR.CS), "folds": 5, "score": [3.0, 1.0, 2.0, 2.5, 4.0], "edge": False, "rows": 13803,
           "questions": 576, "cells": [35, 35, 35], "bins": ["0", "1-3", "4-"], "groups": ["scene", "report"], "seconds": 1.0}
    lk = {"leak": 0.2, "segments": ["0"], "groups": [{"group": "report", "n": 2, "any": 0, "segments": {"0": {"n": 2, "leak": 0, "total_mean": 0.0, "abs_max": 0.0}},
                                                      "pos0": {"n": 2, "leak": 0, "total_mean": 0.0, "abs_max": 0.0}}], "all": None}
    d = {"name": "tt", "layers": [8], "segments": ["0"], "form": "cap", "alpha": 1.0, "units": [], "unit_rows": [], "key_base": "tt_r",
         "rekeyed": {"rows": []}, "verify": {"ok": True}, "check": {"groups": {}, "leaks": lk}, "notes": [], "stage_seconds": {},
         "key_fit": "ridge", "ridge": {"build": {**rec, "c": 1.0}, "rekey": rec}}
    sm = fish.summary(d)
    assert sm["key_fit"] == "ridge" and sm["leaks"] == lk and sm["ridge"]["build"]["c"] == 1.0 and sm["ridge"]["rekey"]["cells"] == [35, 35, 35]
    assert "seconds" not in sm["ridge"]["rekey"] and "score" in sm["ridge"]["rekey"]
    text = fish.summary_text(d)
    assert "鍵の作り方: ridge(最小二乗の鍵、ますごとの重み)。正則化の強さ: 1 段目 c = 1(質問単位の 5 分割で決めた。" in text
    assert "当てた状態で作り直した鍵 c = 0.1(" in text and "候補の端" not in text and fish.summary_text(sm) == text
    edge = fish.summary_text({**d, "ridge": {"build": {**rec, "c": 100.0, "edge": True}, "rekey": None}})
    assert "c = 100(質問単位の 5 分割で決めた。候補 0.01, 0.1, 1, 10, 100。候補の端" in edge
    given = fish.summary_text({**d, "ridge": {"build": {**rec, "cv": False, "cs": None, "folds": None, "c": 2.0}, "rekey": None}})
    assert "1 段目 c = 2(渡した値" in given
    old = fish.summary({**d, "key_fit": None, "ridge": None, "check": {"groups": {}}})        # 前の形の結果(作り方の記録なし)= いまの作り方
    assert old["key_fit"] == "mean" and old["ridge"] is None and old["leaks"] is None
    assert "鍵の作り方: mean(いまの作り方。" in fish.summary_text(old)
    assert fish.DEFAULTS["key_fit"] == "mean" and {"key_fit", "key_c"} <= set(fish.SETTINGS) and "key_c" not in fish.DEFAULTS


def test_entry_arguments_stop_before_the_model():
    """作り方・強さ・書く位置が合わない指定は、モデルを流す前に止まる(session を使わない)。"""
    from gne import fish
    from gne.keyed_build import build_keys_multi
    for kw in ({"key_fit": "lasso"}, {"key_c": 1}, {"key_fit": "ridge", "key_c": 0}, {"key_fit": "ridge", "key_c": "x"},
               {"key_fit": "ridge", "segments": "before:(です|ます)。"}, {"key_fit": "mean", "key_c": 1}):
        kw = {"segments": "0,1-3,4-", **kw}
        with pytest.raises(GneError) as ei:
            fish.keys_step(None, "c.yaml", base="b", layers="8", **kw)
        assert ei.value.code == "BAD_ARGS", kw
        with pytest.raises(GneError) as ei:
            fish.pipeline(None, "c.yaml", name="b", layers="8", alpha=1, **kw)
        assert ei.value.code == "BAD_ARGS", kw
    for kw in ({"key_c": 1}, {"key_fit": "lasso"}):
        with pytest.raises(GneError) as ei:
            build_keys_multi(None, "c.yaml", [("k", "pos:0")], layers=[8], target="mlpin", **kw)
        assert ei.value.code == "BAD_ARGS"


def test_templates_list_non_question_groups(tmp_path):
    """ひな形の「変えたくない場面」には、質問のほかに質問でない発言の群(近況の報告・誘い・挨拶・頼みごと・独り言)が、
    群の名前つきで入っていて、対照集合として読める(群の名前 = dont_touch の name。書式は今までのまま)。"""
    import json
    from pathlib import Path

    from gne.keyed_build import read_key_spec
    from gne.yamlio import write_yaml
    root = Path(__file__).resolve().parent.parent / "configs" / "fish"
    for name in ("tone_in_scene.json", "after_chips.json"):
        d = json.loads((root / name).read_text(encoding="utf-8"))
        about = d.pop("about")
        assert [g["name"] for g in d["dont_touch"]] == ["tasks", "ordinary", "report", "invite", "greet", "request", "partial"], name
        assert all(len(g["prompts"]) >= 4 for g in d["dont_touch"]) and d["fish"]["key_fit"] == "mean", name
        assert not set(d["user_prompts"]) & {q for g in d["dont_touch"] for q in g["prompts"]}, name      # 狙う場面と重ならない
        assert "key_fit" in about["ja"]["text"] and "key_fit" in about["en"]["text"], name
        p = tmp_path / f"{name}.yaml"
        write_yaml(p, {k: v for k, v in d.items() if k != "fish"})
        ks = read_key_spec(p)
        assert [g.name for g in ks.dont_touch] == [g["name"] for g in d["dont_touch"]], name
    text = json.loads((root / "tone_in_scene.json").read_text(encoding="utf-8"))["about"]
    assert "数十問" in text["ja"]["text"] and "several dozen" in text["en"]["text"] and "1 つのモデル・1 つの性格" in text["ja"]["text"]


# --- モデルあり ----------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
NGL = int(os.environ.get("GNE_TEST_NGL", "-1"))
CONTRAST = {
    "name": "rk_scene",
    "system_a": "You are a pirate. Always answer in rough pirate speech.",
    "user_prompts": ["Thank you for your help yesterday.", "I really like your drawing.", "Thanks for lending me your notebook.",
                     "Your cooking was delicious.", "Thank you for listening to me.", "You look great today.",
                     "Thanks a lot for the ride home.", "Your song was wonderful."],
    "dont_touch": [{"name": "tasks", "prompts": ["What is 12 times 8?", "What is the capital of France?", "How many days are in a week?",
                                                 "What color is the sky?"]},
                   {"name": "report", "prompts": ["I went to the park this morning.", "It rained all day here.", "I finished my homework.",
                                                  "I started reading a new book."]}],
}
LAYERS = [8, 12]
MT = 12


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    """小さいモデルで、最小二乗の鍵(区切り 0 / 1-3 / 4-)を 1 回だけ作る。鍵の計算に渡した行と表を覚えておく。"""
    if not MODEL.exists():
        pytest.skip(f"{MODEL} がありません")
    from gne import fish
    from gne.api import Session
    root = tmp_path_factory.mktemp("rk")
    c = root / "c.yaml"
    c.write_text(yaml.safe_dump(CONTRAST, allow_unicode=True), encoding="utf-8")
    s = Session(MODEL, root=root, n_gpu_layers=NGL)
    seen = {}
    real = KR.fit

    def spy(X_of, layers, table, ranges, **kw):
        xs = {L: np.array(X_of(L)) for L in layers}
        seen.update(xs=xs, table={k: np.array(v) for k, v in table.items()}, ranges=list(ranges), kw=dict(kw))
        return real(lambda L: xs[L], layers, table, ranges, **kw)
    KR.fit = spy
    try:
        keys = fish.keys_step(s, str(c), base="rk", layers=LAYERS, segments="0,1-3,4-", form="cap", pseudo=2, max_tokens=MT, key_fit="ridge")
    finally:
        KR.fit = real
    yield {"s": s, "root": root, "contrast": c, "keys": keys, "seen": seen}
    s.close()


@pytest.mark.model
def test_rows_are_the_mlp_input_at_the_kept_positions(made):
    """鍵の計算に渡す行 = 返事を作り直して 1 本ずつ読んだ MLP の入力(正規化の後)の、間引いた位置の値。保存した鍵は、その行に
    式どおりの重み(書き込みたい位置の合計 1、書き込みたくない位置はますごとに等分)を付けて閉じた式で解いた答えと合う。"""
    from gne import nwrite as NW
    from gne.evalx import _generate
    from gne.keyed import load_key
    from gne.keyed_build import read_key_spec
    s, seen = made["s"], made["seen"]
    tab, xs = seen["table"], seen["xs"]
    assert seen["ranges"] == [(0, 1), (1, 4), (4, None)] and seen["kw"]["c"] is None and seen["kw"]["pseudo"] == 2
    ks = read_key_spec(made["contrast"])
    sets = [ks.contrast.convs("B")] + [dt.convs(list(range(len(dt.prompts)))) for dt in ks.dont_touch]
    assert tab["q_group"].tolist() == [0] * 8 + [1] * 4 + [2] * 4
    e = s.engine
    qi = 0
    for convs in sets:
        outs = _generate(s, [(cv, MT) for cv in convs], None, None, "all", lambda k: None)
        for cv, o in zip(convs, outs):
            p = e.tokenize_prompt(e.render_chat(cv))
            rid = [t for t in o["token_ids"] if not e.is_eog(t)]
            keep, om = KR.keep_positions(len(rid) + 1, np.random.default_rng(KR.SEED0 + qi))
            m = tab["q"] == qi
            assert tab["pos"][m].tolist() == keep.tolist() and np.allclose(tab["om"][m], om)
            names = {L: NW._capture_names(s, p + rid, L)["h"] for L in LAYERS}
            got = NW._run_capture(s, p + rid, set(names.values()))
            for L in LAYERS:
                ref = got[names[L]][len(p) - 1 + keep].astype(np.float16).astype(np.float64)
                np.testing.assert_allclose(xs[L][m].astype(np.float64), ref, rtol=2e-3, atol=2e-3 * float(np.abs(ref).max()))
            qi += 1
    assert qi == tab["q_group"].size
    info = made["keys"]["ridge"]
    c = info["c"]
    assert info["cv"] and info["folds"] == 5 and c in KR.CS and info["edge"] == (c in (KR.CS[0], KR.CS[-1]))
    assert info["groups"] == ["scene", "tasks", "report"] and info["bins"] == ["0", "1-3", "4-"] and info["cells"] == [8, 8, 8]
    for e_i, rg in enumerate(seen["ranges"]):
        W, w = _weights(tab, seen["ranges"], e_i)
        k = load_key(s, f"rk_p{e_i}")
        assert k.info["key_fit"] == "ridge" and k.info["ridge"]["c"] == c and k.positions == f"pos:{KR.bin_text(rg)}"
        for L in LAYERS:
            ref = _direct(xs[L].astype(np.float64), W, w, c)
            got = np.asarray(k.kappa[L], dtype=np.float64)
            assert float(np.abs(got - ref).max()) <= 1e-5 * float(np.abs(ref).max()), (e_i, L)
            assert float(got @ np.asarray(k.mu[L], dtype=np.float64)) == pytest.approx(1.0, abs=1e-4)
            assert k.kappa_ctrl[L].shape == (2, s.meta.n_embd) and not np.allclose(k.kappa_ctrl[L][0], got)   # 比較用の鍵(同じ式・同じ c)


@pytest.mark.model
def test_keys_step_result_and_notes(made):
    d = made["keys"]
    assert d["key_fit"] == "ridge" and [r["key"] for r in d["rows"] if r["layer"] == 8] == ["rk_p0", "rk_p1", "rk_p2", "rk_one"]
    assert all(r["q_rank"] is None for r in d["rows"]) and all(r["beats"] is not None for r in d["rows"] if r["kind"] == "position")
    assert any(n.startswith("最小二乗の鍵(ますごとの重み): 行 ") for n in d["notes"])
    from gne.keyed import load_key
    one = load_key(made["s"], "rk_one")                                       # ほぼ 1 を返す鍵の作り方は変わらない
    assert one.info["role"] == "one" and "key_fit" not in one.info


@pytest.mark.model
def test_default_keys_keep_the_old_form(made):
    """既定(key_fit を渡さない)の鍵は、今までの形のまま: 鍵のファイルに作り方の記録が無く、結果の key_fit は mean、ridge は None。"""
    from gne import fish
    from gne.keyed import load_key
    s, c = made["s"], made["contrast"]
    d = fish.keys_step(s, str(c), base="dk", layers=LAYERS, segments="0,1-3,4-", form="cap", pseudo=2, max_tokens=MT)
    assert d["key_fit"] == "mean" and d["ridge"] is None and all(r["q_rank"] is not None for r in d["rows"] if r["kind"] == "position")
    for i in range(3):
        k = load_key(s, f"dk_p{i}")
        assert "key_fit" not in k.info and "ridge" not in k.info and k.q_mode == "examples"
        assert not any("最小二乗" in n for n in d["notes"])
    # 同じ返事・同じ読み取りから作るので、書き込みたい位置の行は同じ(平均 μ が、最小二乗の鍵の m_W と同じ位置の行から来ている)
    a, b = load_key(s, "dk_p0"), load_key(s, "rk_p0")
    assert np.allclose(np.asarray(a.mu[8]), np.asarray(b.mu[8]), rtol=1e-3, atol=1e-3 * float(np.abs(np.asarray(a.mu[8])).max()))


def _job(s, kind, params):
    import time

    from gne.jobs import GpuArbiter, JobRunner, standard_kinds
    runner = JobRunner(s, GpuArbiter(), standard_kinds())
    j = runner.start(kind, params)
    t0 = time.time()
    while j.status in ("queued", "running") and time.time() - t0 < 900:
        time.sleep(0.2)
    assert j.status == "done", j.view()
    return j.result


@pytest.mark.model
def test_pipeline_two_stages_and_check(made):
    """まとめて作る: 1 段目も、当てた状態で鍵を作り直す段も最小二乗の鍵。gne nw check に作り方と群ごとの漏れの表が出る。"""
    from typer.testing import CliRunner

    from gne import fish
    from gne.cli import app
    from gne.keyed import load_key
    s, root, c = made["s"], made["root"], str(made["contrast"])
    small = dict(segments="0,1-", max_tokens=8, response_tokens=8, pseudo=2, top=6, compare=False)
    p = fish.pipeline(s, c, name="rp", layers="8,12", alpha="1,0.8", key_fit="ridge", save="rp_nw", **small)
    assert p["key_fit"] == "ridge" and p["ridge"]["build"]["cv"] and p["ridge"]["rekey"]["cv"] and p["key_base"] == "rp_r"
    assert p["keys"]["key_fit"] == p["rekeyed"]["key_fit"] == "ridge" and p["ridge"]["build"]["bins"] == ["0", "1-"]
    for nm in ("rp_p0", "rp_p1", "rp_r_p0", "rp_r_p1"):
        assert load_key(s, nm).info["key_fit"] == "ridge", nm
    assert load_key(s, "rp_r_p0").info["iv"]                                  # 2 段目は、1 段目のユニットを当てた状態で作った
    assert all("#rp_r_p" in u for u in p["units"]) and p["verify"]["ok"]
    chk = p["check"]
    assert chk["key_fit"]["fit"] == "ridge" and set(chk["key_fit"]["keys"]) == {"rp_r_p0", "rp_r_p1"}
    assert [g["group"] for g in chk["leaks"]["groups"]] == ["tasks", "report"] and chk["leaks"]["segments"] == ["0", "1-"]
    for g in chk["leaks"]["groups"]:
        bq = chk["groups"][f"dont_touch:{g['group']}"]["by_question"]
        assert {sg: x["leak"] for sg, x in bq.items()} == {sg: g["segments"][sg]["leak"] for sg in ("0", "1-")}
        assert g["n"] == 4 and g["pos0"]["n"] == 4
    sm = p["summary"]
    assert sm["key_fit"] == "ridge" and sm["leaks"] == chk["leaks"] and set(sm["ridge"]) == {"build", "rekey"}
    text = fish.summary_text(p)
    assert "鍵の作り方: ridge(最小二乗の鍵、ますごとの重み)。正則化の強さ: 1 段目 c = " in text and "当てた状態で作り直した鍵 c = " in text
    assert "変えたくない場面の群ごとの漏れ" in text
    common = ["-m", str(MODEL), "--root", str(root), "--n-gpu-layers", str(NGL)]
    rp = str(s.recipe_path("rp_nw"))
    r = CliRunner().invoke(app, ["nw", "check", "--recipe", rp, "--contrast", c, "--subset", "all", "--max-tokens", "8", *common])
    assert r.exit_code == 0 and "鍵の作り方: ridge(最小二乗の鍵、ますごとの重み。正則化の強さ c = " in r.output, r.output
    assert "変えたくない場面の群ごとの漏れ" in r.output and "  report " in r.output and "位置 0 だけ" in r.output
    r = CliRunner().invoke(app, ["nw", "check", "--recipe", rp, "--contrast", c, "--subset", "all", "--max-tokens", "8", "--leak", "0.05",
                                 *common, "--json"])
    d = json.loads(r.output.strip().splitlines()[-1])
    assert r.exit_code == 0 and d["leaks"]["leak"] == 0.05 and d["key_fit"]["fit"] == "ridge" and d["question_summary"]["leak"] == 0.05
    # CLI のまとめて作る: 正則化の強さを渡す(両方の段でその値)。方向は使い回す
    r = CliRunner().invoke(app, ["nw", "pipeline", "--contrast", c, "--layers", "8,12", "--alpha", "1,0.8", "--segments", "0,1-", "--max-tokens", "8",
                                 "--pseudo", "0", "--top", "6", "--name", "rp_c", "--direction", "rp", "--key-fit", "ridge", "--key-c", "10",
                                 *common, "--json"])
    d = json.loads(r.output.strip().splitlines()[-1])
    assert r.exit_code == 0 and d["key_fit"] == "ridge" and d["ridge"]["build"]["c"] == 10 and d["ridge"]["rekey"]["c"] == 10, r.output
    assert d["ridge"]["build"]["cv"] is False and d["leaks"]["groups"]
    for bad in (["--key-fit", "lasso"], ["--key-c", "1"], ["--key-fit", "ridge", "--key-c", "0"], ["--key-fit", "ridge", "--segments", "before:x"]):
        r = CliRunner().invoke(app, ["nw", "pipeline", "--contrast", c, "--layers", "8", *bad, *common, "--json"])
        assert r.exit_code != 0 and json.loads(r.output.strip().splitlines()[-1])["error"]["code"] == "BAD_ARGS" and "見積もり" not in r.output, bad


@pytest.mark.model
def test_key_entries_cli_jobs_and_estimates(made):
    from typer.testing import CliRunner

    from gne.cli import app
    from gne.estimate import Speed, _fish_cost, estimate
    from gne.keyed import load_key
    s, root, c = made["s"], made["root"], str(made["contrast"])
    common = ["-m", str(MODEL), "--root", str(root), "--n-gpu-layers", str(NGL), "--json"]
    # gne nw key --role position(gne direction key --target mlpin と同じ)。正則化の強さを渡す
    r = CliRunner().invoke(app, ["nw", "key", "--contrast", c, "--layers", "8", "--role", "position", "--positions", "pos:1-3", "--key-fit", "ridge",
                                 "--key-c", "1", "--pseudo", "2", "--name", "cli_rk", "--subset", "all", "--max-tokens", "8", *common])
    d = json.loads(r.output.strip().splitlines()[-1])
    assert r.exit_code == 0 and d["role"] == "position" and d["key_fit"] == "ridge" and d["ridge"]["c"] == 1 and d["ridge"]["cv"] is False, r.output
    assert d["ridge"]["bins"] == ["0", "1-3", "4-"] and load_key(s, "cli_rk").n_pseudo == 2      # 1 本だけ: 区間と、その外の続いた範囲
    # gne direction key --target mlpin --key-fit ridge(c は 5 分割で決める)
    r = CliRunner().invoke(app, ["direction", "key", "--contrast", c, "--layers", "8", "--target", "mlpin", "--positions", "pos:0", "--key-fit", "ridge",
                                 "--pseudo", "0", "--name", "cli_dk", "--subset", "all", "--max-tokens", "8", *common])
    d = json.loads(r.output.strip().splitlines()[-1])
    assert r.exit_code == 0 and d["key_fit"] == "ridge" and d["ridge"]["cv"] is True and d["ridge"]["bins"] == ["0", "1-"], r.output
    assert d["layers"][0]["q_rank"] is None and d["n_pseudo"] == 0
    r = CliRunner().invoke(app, ["direction", "key", "--contrast", c, "--layers", "8", "--target", "mlpin", "--positions", "pos:0", "--name", "cli_mk",
                                 "--subset", "all", "--max-tokens", "8", "--pseudo", "0", *common])
    d = json.loads(r.output.strip().splitlines()[-1])
    assert r.exit_code == 0 and d["key_fit"] == "mean" and d["ridge"] is None and isinstance(d["layers"][0]["q_rank"], int), r.output
    bads = [["direction", "key", "--layers", "8", "--target", "ffn", "--positions", "pos:0", "--key-fit", "ridge"],     # keyed write の鍵には使えない
            ["direction", "key", "--layers", "8", "--target", "mlpin", "--positions", "pos:0", "--key-c", "1"],
            ["direction", "key", "--layers", "8", "--target", "mlpin", "--positions", "before:x", "--key-fit", "ridge"],
            ["direction", "key", "--layers", "8", "--target", "mlpin", "--positions", "pos:0", "--key-fit", "lasso"],
            ["nw", "key", "--layers", "8", "--role", "scene", "--key-fit", "ridge"],
            ["nw", "key", "--layers", "8", "--role", "position"]]
    for bad in bads:
        r = CliRunner().invoke(app, [*bad, "--contrast", c, "--subset", "all", "--max-tokens", "4", *common])
        assert r.exit_code != 0 and json.loads(r.output.strip().splitlines()[-1])["error"]["code"] == "BAD_ARGS", (bad, r.output)
    # ジョブ: 鍵をまとめて作る(fish_keys)・鍵 1 本(direction_key)・表の fish.key_fit
    jk = _job(s, "fish_keys", {"contrast": c, "layers": "8", "segments": "0,1-", "key_fit": "ridge", "key_c": 1, "pseudo": 0, "max_tokens": 8, "base": "job_rk"})
    assert jk["key_fit"] == "ridge" and jk["ridge"]["c"] == 1 and load_key(s, "job_rk_p1").info["key_fit"] == "ridge"
    jt = _job(s, "fish_keys", {"spec": {**CONTRAST, "name": "job_tab", "fish": {"key_fit": "ridge", "segments": "0,1-", "pseudo": 0, "max_tokens": 8}},
                               "layers": "8"})
    assert jt["key_fit"] == "ridge" and jt["ridge"]["cv"] is True
    jd = _job(s, "direction_key", {"contrast": c, "layers": [8], "target": "mlpin", "positions": "pos:0", "key_fit": "ridge", "key_c": 1,
                                   "pseudo": 0, "subset": "all", "max_tokens": 8, "name": "job_dk"})
    assert jd["key_fit"] == "ridge" and jd["ridge"]["c"] == 1
    # 見積もり: 最小二乗の鍵の計算の分が足される(当てた状態で作り直すなら 2 段分)。渡した正則化の強さなら、分割の分だけ短い
    sp = Speed(s)
    ep = {"contrast": c, "layers": "8,12", "segments": "0,1-3,4-"}
    m, r_ = _fish_cost(s, sp, "fish_pipeline", ep), _fish_cost(s, sp, "fish_pipeline", {**ep, "key_fit": "ridge"})
    assert m["key_fit_seconds"] == 0 and r_["key_fit_seconds"] > 0
    assert r_["seconds"] - m["seconds"] == pytest.approx(r_["key_fit_seconds"]) and r_["keys_seconds"] - m["keys_seconds"] == pytest.approx(r_["key_fit_seconds"] / 2)
    given = _fish_cost(s, sp, "fish_pipeline", {**ep, "key_fit": "ridge", "key_c": 1})
    assert 0 < given["key_fit_seconds"] < r_["key_fit_seconds"]
    once = _fish_cost(s, sp, "fish_pipeline", {**ep, "key_fit": "ridge", "rekey": False})
    assert once["key_fit_seconds"] == pytest.approx(r_["key_fit_seconds"] / 2)
    kk = _fish_cost(s, sp, "fish_keys", {**ep, "key_fit": "ridge"})
    assert kk["key_fit_seconds"] == pytest.approx(r_["key_fit_seconds"] / 2)
    tab = _fish_cost(s, sp, "fish_keys", {"spec": {**CONTRAST, "fish": {"key_fit": "ridge"}}, "layers": "8,12"})      # 表の fish.key_fit
    assert tab["key_fit_seconds"] > 0
    ek = estimate(s, "direction_key", {"contrast": c, "layers": [8], "target": "mlpin", "positions": "pos:0", "key_fit": "ridge"})
    e0 = estimate(s, "direction_key", {"contrast": c, "layers": [8], "target": "mlpin", "positions": "pos:0"})
    assert ek["detail"]["key_fit_seconds"] > 0 and "key_fit_seconds" not in e0["detail"] and ek["seconds"] > e0["seconds"]
