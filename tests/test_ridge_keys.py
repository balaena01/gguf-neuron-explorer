"""最小二乗の鍵(key_fit ridge、gne/keyed_ridge.py)の式: モデルを使わない(小さな合成データ)。

- 閉じた式の答えと合うこと(重みを 1 行ずつ付けて、行列をそのまま解いた答えと比べる)
- ますごとの重みが効くこと(行の数の少ない群を見分ける。位置を全部同じ重みで数えると見分けられない)
- 正則化の強さの決め方(質問単位の分割。別の書き方で数え直した値と同じ c・同じ点数になる)
- 比較用の鍵、行の間引き、位置の区分
"""
import numpy as np
import pytest

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
