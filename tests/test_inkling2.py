"""Inkling 2(arch inkling2_loop、Inkling-2-Mini)への対応(読み込み・情報・マップ・会話だけ。docs/inkling2.md)。

モデルが要らないもの(段の対応・返事の区切り・断りの文)は、いつも流す。
実物を使うものは GNE_TEST_INKLING2 に GGUF のパスを入れたときだけ流す(Inkling 2 を読める llama.cpp の llama-cpp-python が要る。
読めなければ飛ばす)。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from gne.inkling import InklingPlan, ReplyMarks, answer_part, plan_of, refuse
from gne.steer import GneError

ARCH = "inkling2_loop"
# 実物と同じ並び: 層 5・11・17・23 だけ全体を見る attention
SWA = [L % 6 != 5 for L in range(24)]
META = {f"{ARCH}.block_count": 24, f"{ARCH}.attn_res.block_size": 6, f"{ARCH}.attention.sliding_window_pattern": SWA,
        f"{ARCH}.attention.sliding_window": 1024, f"{ARCH}.loop.start": 6, f"{ARCH}.loop.end": 18, f"{ARCH}.loop.count": 3,
        f"{ARCH}.loop.gate_rank": 64, f"{ARCH}.loop.varying": False}


def _plan(**over) -> InklingPlan:
    m = {**META, **over}
    return plan_of(lambda k, d=None: m.get(k, d), ARCH)


def test_steps_follow_llama_cpp_slots():
    p = _plan()
    assert (p.n_stored, p.n_passes, p.width, p.n_steps) == (24, 3, 12, 48)
    st = p.steps()
    assert [s["layer"] for s in st] == list(range(18)) + list(range(6, 18)) * 2 + list(range(18, 24))
    assert [s["pass"] for s in st] == [0] * 6 + [1] * 12 + [2] * 12 + [3] * 12 + [0] * 6
    # llama.cpp のキャッシュの番号: 2 回目は 24〜35、3 回目は 36〜47、繰り返しの後の層は 18〜23(実行の順と違う)
    assert [s["slot"] for s in st] == list(range(18)) + list(range(24, 36)) + list(range(36, 48)) + list(range(18, 24))
    assert sorted(s["slot"] for s in st) == list(range(48))
    assert [s["stream"] for s in st] == [0] * 6 + ([1] * 6 + [2] * 6) * 3 + [3] * 6
    assert [s["step"] for s in st if s["new_stream"]] == [0, 6, 12, 18, 24, 30, 36, 42]
    assert [s["layer"] for s in st if s["full"]] == [5, 11, 17, 11, 17, 11, 17, 23]
    assert p.steps_of(6) == [6, 18, 30] and p.steps_of(3) == [3] and p.steps_of(20) == [44]
    assert p.full_layers() == [5, 11, 17, 23]
    g = p.step_groups()
    assert [(x["steps"], x["layers"], x["pass"], x["slots"]) for x in g] == [
        ([0, 5], [0, 5], 0, [0, 5]), ([6, 17], [6, 17], 1, [6, 17]), ([18, 29], [6, 17], 2, [24, 35]),
        ([30, 41], [6, 17], 3, [36, 47]), ([42, 47], [18, 23], 0, [18, 23])]
    with pytest.raises(GneError):
        p.step(48)
    d = p.to_dict()
    assert d["n_steps"] == 48 and len(d["steps"]) == 48 and "層 6〜17 を 3 回" in d["summary"]


def test_plan_shapes_and_other_archs():
    assert plan_of(lambda k, d=None: d, "qwen35") is None
    with pytest.raises(GneError) as e:
        _plan(**{f"{ARCH}.loop.start": 5})                 # 流れの間隔(6)にそろっていない
    assert e.value.code == "BAD_MODEL"
    with pytest.raises(GneError):
        _plan(**{f"{ARCH}.loop.count": 1})                 # 回数を変えて学習していないのに 1 回
    assert _plan(**{f"{ARCH}.loop.count": 1, f"{ARCH}.loop.varying": True}).n_steps == 24
    with pytest.raises(GneError):
        _plan(**{f"{ARCH}.attention.sliding_window_pattern": SWA[:10]})
    # 繰り返しの無い inkling2
    m = {k.replace(ARCH, "inkling2"): v for k, v in META.items() if ".loop." not in k}
    p = plan_of(lambda k, d=None: m.get(k, d), "inkling2")
    assert p.n_steps == 24 and [s["slot"] for s in p.steps()] == list(range(24)) and all(s["pass"] == 0 for s in p.steps())


def test_sensenova_loop_plan_ignores_inkling():
    from gne.loop import plan_of as loop_plan_of
    m = {**META}
    assert loop_plan_of(lambda k, d=None: m.get(k, d), ARCH) is None     # SenseNova の形として読まない(以前は BAD_MODEL)


def test_refuse_message():
    e = refuse(_plan(), "control vector")
    assert e.code == "ARCH_UNSUPPORTED" and "control vector" in e.message and "層 6〜17 を 3 回" in e.message
    assert "docs/inkling2.md" in e.hint
    from gne.inkling import check_command
    for ok in ("info", "chat", "behave", "eval", "probe", "heads pipeline", "heads refine", "headproj export", None):   # CHiPS の流れと評価は通す
        check_command(_plan(), ok)
    for bad in ("direction", "steer", "nw pipeline", "stats resid"):
        with pytest.raises(GneError) as ex:
            check_command(_plan(), bad)
        assert ex.value.code == "ARCH_UNSUPPORTED" and f"gne {bad}" in ex.value.message


def test_capture_translates_slots_to_steps():
    import numpy as np

    from gne.capture import Capture
    cap = Capture(lambda n: n in ("l_out-18", "attn_out-6", "attn_out-7", "out_mix"))
    cap.inkling = _plan()
    assert cap._canon("l_out-24") == "l_out-18" and cap._canon("l_out-18") == "l_out-42" and cap._canon("l_out-47") == "l_out-41"
    assert cap._canon("out_mix") == "out_mix" and cap._canon("norm") == "norm"
    # 流れが始まる段(6)の attn_out は ffn_inp と同じテンソル: ffn_inp-6 を取って attn_out-6 の名前で返す。始まらない段(7)は取らない
    assert cap._want("ffn_inp-6") and not cap._want("ffn_inp-7") and cap._want("attn_out-7")
    cap._store("ffn_inp-6", np.ones((2, 3), np.float32))
    assert "attn_out-6" in cap._step and "ffn_inp-6" not in cap._step


def test_states_follow_the_graph():
    from gne.inkling_read import Reader
    r = Reader.__new__(Reader)
    r.plan = _plan()
    r.steps = r.plan.steps()
    st = r.states()
    names = lambda xs: [n for _, n in xs]  # noqa: E731
    assert 0 not in st["attn"]                                                      # 層 0 は深さ方向の注意なしで読む
    assert names(st["ffn"][0]) == ["embd_norm", "ffn_inp-0"]
    assert names(st["attn"][6]) == ["embd_norm", "l_out-5"]                          # 流れが始まる段: 前の流れを取っておく前
    assert names(st["ffn"][6]) == ["l_out-5", "embd_norm", "ffn_inp-6"]              # 取っておいた流れは新しい順
    assert names(st["attn"][12]) == ["l_out-5", "embd_norm", "l_out-11"]
    assert names(st["ffn"][13]) == ["l_out-11", "l_out-5", "embd_norm", "ffn_inp-13"]
    assert names(st["attn"][18]) == ["embd_norm", "l_out-5"]                         # 2 回目は層 5 の出力からやり直す
    assert names(st["ffn"][24]) == ["l_out-23", "l_out-5", "embd_norm", "ffn_inp-24"]
    assert names(st["msg"][18]) == ["l_out-11", "l_out-17"] and names(st["msg"][30]) == ["l_out-23", "l_out-29"]
    assert 6 not in st["msg"]
    assert names(st["attn"][42]) == ["l_out-35", "l_out-5", "embd_norm", "l_out-41"]  # 繰り返しの後は 3 回目の流れを読む
    assert names(st["out"]) == ["l_out-41", "l_out-35", "l_out-5", "embd_norm", "l_out-47"]


def test_conv_post_matches_the_formula():
    import numpy as np

    from gne.inkling_read import _conv_post
    rng = np.random.default_rng(0)
    x, w = rng.standard_normal((6, 5)), rng.standard_normal((5, 4))
    out = _conv_post(x, w)
    for t in range(6):
        ref = x[t] + sum(w[:, k] * (x[t - 3 + k] if t - 3 + k >= 0 else 0) for k in range(4))
        np.testing.assert_allclose(out[t], ref / np.sqrt(5), rtol=1e-12)


def test_reasoning_effort_values():
    from gne.chat import effort_choices, effort_value
    tpl = '{%- set effort_map = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99} -%}' \
          '{{ reasoning_effort }}'
    assert effort_choices(tpl) == ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
    assert effort_choices("{{ messages }}") is None and effort_choices("{{ reasoning_effort }}") == []
    assert effort_value(tpl, None) is None and effort_value(tpl, " ") is None
    assert effort_value(tpl, "low") == "low" and effort_value(tpl, "0.5") == 0.5 and effort_value(tpl, 0) == 0.0
    for bad in ("huge", "1.5", "-0.1"):
        with pytest.raises(GneError):
            effort_value(tpl, bad)
    with pytest.raises(GneError):
        effort_value("{{ messages }}", "low")


def test_chips_trial_projection_math():
    import numpy as np

    from gne.inkling_chips import layer_lora
    rng = np.random.default_rng(3)
    d, nh, hd = 12, 3, 4
    Wo, cw, r = rng.standard_normal((d, nh * hd)), rng.standard_normal((d, 4)) * 0.5, rng.standard_normal(d)
    g = (1 + cw[:, -1]) / np.sqrt(5)
    x = rng.standard_normal(nh * hd)
    for correct in (True, False):
        A, B = layer_lora(Wo, cw, r, [(1, 1.0)], hd, correct=correct)
        W1 = Wo + B @ A
        y = W1[:, hd:2 * hd] @ x[hd:2 * hd]                     # ヘッド 1 の書き込み(畳み込みの前)
        now = r @ (g * y)                                        # 畳み込みの後の、今のトークンの r の成分
        if correct:
            assert abs(now) < 1e-9 * np.linalg.norm(y)           # 直した形は今のトークンの分をちょうど消す
        else:
            assert abs(r @ y) < 1e-9 * np.linalg.norm(y) and abs(now) > 1e-6
        np.testing.assert_allclose(W1[:, :hd], Wo[:, :hd])      # ほかのヘッドは変わらない
    A2, _ = layer_lora(Wo, cw, r, [(None, 2.0)], hd)
    A1, _ = layer_lora(Wo, cw, r, [(h, 2.0) for h in range(nh)], hd)
    np.testing.assert_allclose(A1, A2)                           # attention 全体 = 全部のヘッド


def test_chips_trial_tap_vectors_match_the_convolution():
    import numpy as np

    from gne.inkling_chips import _taps
    from gne.inkling_read import _conv_post
    rng = np.random.default_rng(5)
    d, T = 10, 7
    cw, r, y = rng.standard_normal((d, 4)), rng.standard_normal(d), rng.standard_normal((T, d))
    V = _taps(cw, r)
    # 最後のトークンの畳み込みの後の r の成分 = 最後の 4 トークンの書き込みと、タップごとの向きの内積の和
    np.testing.assert_allclose(_conv_post(y, cw)[-1] @ r, sum(V[k] @ y[T - 4 + k] for k in range(4)), rtol=1e-12)


def test_chips_trial_candidates_all():
    from gne.inkling_chips import candidates_all, unit_of
    p = _plan()
    ra = {"rows": [{"kind": "head", "layer": 7, "head": 1, "value": 3.0}, {"kind": "attn", "layer": 12, "head": None, "value": 9.0},
                   {"kind": "ffn", "layer": 12, "head": None, "value": 8.0}, {"kind": "head", "layer": 2, "head": 0, "value": 7.0}],
          "neurons": [{"kind": "neuron", "layer": 9, "expert": 5, "neuron": 77, "value": 5.0},
                      {"kind": "neuron", "layer": 20, "expert": 1, "neuron": 3, "value": -1.0}]}
    c = candidates_all(p, ra, ("head", "attn", "ffn", "neuron"))
    assert [(r["kind"], r["layer"]) for r in c] == [("ffn", 12), ("neuron", 9), ("head", 7)]   # 流れの始まりの層は MLP だけ残す
    assert [(r["kind"]) for r in candidates_all(p, ra, ("head", "attn", "ffn"))] == ["ffn", "head"]
    assert unit_of(c[1], 2.0) == [9, "neuron", [5, 77], 2.0] and unit_of(c[0], 1.0) == [12, "ffn", None, 1.0]


def test_expert_neuron_projection_units():
    """専門家のニューロンの射影(L:Ee:Nn:sS@方向、Inkling 2 だけ)の書き方・レシピ・強さの倍率。"""
    import dataclasses

    from gne.headproj import heads_to_units, proj_heads
    from gne.steer import Recipe, parse_unit
    from gne.unitwrites import unit_of
    u = parse_unit("14:E25:N389:s2.5@dirx")
    assert (u.kind, u.mode, u.expert, u.neuron, u.k) == ("expert_neuron", "proj", 25, 389, 2.5)
    r = Recipe([u, parse_unit("11:H5:s1@dirx"), parse_unit("9:ffn:s1@dirx")], "m", name="t")
    r.validate()
    hs = proj_heads(r)
    assert [h.spec() for h in hs] == ["14:E25:N389:s2.5@dirx", "11:H5:s1@dirx", "9:ffn:s1@dirx"]
    assert hs[0].key != dataclasses.replace(hs[0], expert=26).key              # 専門家が違えば別の部品
    assert [x.label() for x in heads_to_units(hs, "dirx")][0] == "L14 E25 N389 s2.5"
    assert r.scaled(2.0).units[0].k == 5.0
    assert unit_of({"layer": 14, "index": 389, "kind": "neuron", "expert": 25}, 1.0, "d").spec() == "14:E25:N389:s1@d"
    assert unit_of({"layer": 14, "index": 389, "kind": "neuron"}, 1.0, "d").spec() == "14:N389:s1@d"   # ほかのモデルは今までどおり


def test_inkling_positions_and_rows():
    from gne.inkling_pipeline import parse_positions, rows_for
    from gne.steer import GneError
    assert parse_positions(None) == ("last", None) and parse_positions("after:[.!?]") == ("after", "[.!?]")
    with pytest.raises(GneError):
        parse_positions("response:3")

    class E:
        def is_eog(self, t):
            return t == 99

        def token_bytes(self, t):
            return {50: b"Hi", 51: b".", 52: b" Ok"}.get(t, b"x")
    ids, TEXT = [1, 2, 3], 7
    reply = [5, 6, 6, TEXT, 50, 51, 52, 99]                 # 思考 3 トークン、答えの始まりの区切り、答え 3 トークン、EOG
    assert rows_for(E(), "last", None, ids, 1, None, TEXT) == (ids, [2])
    full, pos = rows_for(E(), "answer", None, ids, 1, reply, TEXT)
    assert full == ids + reply[:4] and pos == [6]            # 区切りの位置(その出力が答えの最初のトークンを決める)
    full, pos = rows_for(E(), "response", None, ids, 1, reply, TEXT)
    assert full == ids + reply[:7] and pos == [6, 7, 8]      # 答えのトークンを決める位置
    assert rows_for(E(), "reasoning", None, ids, 1, reply, TEXT)[1] == [2, 3, 4, 5]
    full, pos = rows_for(E(), "after", r"\.", ids, 1, reply, TEXT)
    assert pos == [8]                                        # "." のトークン(列の番号 8)の位置
    assert rows_for(E(), "answer", None, ids, 1, [5, 6, 99], TEXT) is None     # 思考の途中で終わった


def _random_raw(typ, rows, cols, rng):
    import numpy as np
    nb = cols // 64
    if typ == "IQ4_NL":
        d = rng.choice([-1.0, 1.0], (rows, cols // 32)) * rng.uniform(0.005, 0.02, (rows, cols // 32))
        raw = np.zeros((rows, cols // 32, 18), np.uint8)
        raw[:, :, :2] = d.astype(np.float16).view(np.uint8).reshape(rows, cols // 32, 2)
        raw[:, :, 2:] = rng.integers(0, 256, (rows, cols // 32, 16), dtype=np.uint8)
        return raw.reshape(rows, -1)
    if typ == "Q2_0":
        d = rng.choice([-1.0, 1.0], (rows, nb)) * rng.uniform(0.5, 2.0, (rows, nb))
        raw = np.zeros((rows, nb, 18), np.uint8)
        raw[:, :, :2] = d.astype(np.float16).view(np.uint8).reshape(rows, nb, 2)
        raw[:, :, 2:] = rng.integers(0, 256, (rows, nb, 16), dtype=np.uint8)
    elif typ == "NVFP4":
        raw = np.zeros((rows, nb, 36), np.uint8)
        raw[:, :, :4] = rng.integers(40, 60, (rows, nb, 4), dtype=np.uint8)
        raw[:, :, 4:] = rng.integers(0, 256, (rows, nb, 32), dtype=np.uint8)
    else:
        raw = np.zeros((rows, cols // 32, 17), np.uint8)
        raw[:, :, 0] = rng.integers(118, 126, (rows, cols // 32), dtype=np.uint8)
        raw[:, :, 1:] = rng.integers(0, 256, (rows, cols // 32, 16), dtype=np.uint8)
    return raw.reshape(rows, -1)


@pytest.mark.parametrize("typ", ["Q2_0", "NVFP4", "MXFP4", "IQ4_NL"])
def test_same_size_bake_rounding(typ):
    """元の型・元の大きさの焼き込み(射影を意識した丸め): 編集する列だけコードが動き、uᵀw が目標に近づき、倍率はそのまま。"""
    import numpy as np

    from gne.inkling_bake import decode, edit_slab, encode
    rng = np.random.default_rng(1)
    rows, cols = 128, 128
    nb = cols // 64
    raw = _random_raw(typ, rows, cols, rng)
    q, sc, W = decode(raw, typ, rows, cols)
    assert np.array_equal(encode(raw, typ, rows, cols, q), raw)               # 戻せる
    u = rng.standard_normal(rows)
    u /= np.linalg.norm(u)
    cols_e = [3, 70]
    c = u @ W[:, cols_e]
    r = edit_slab(raw, typ, u, {n: float(-x) for n, x in zip(cols_e, c)}, rows, cols)
    q2, sc2, W2 = decode(r["raw"], typ, rows, cols)
    other = [n for n in range(cols) if n not in cols_e]
    assert np.array_equal(q2[:, other], q[:, other]) and np.array_equal(sc2, sc)   # ほかの列・倍率はそのまま
    hd, nbk = {"Q2_0": (2, nb), "NVFP4": (4, nb), "MXFP4": (1, cols // 32), "IQ4_NL": (2, cols // 32)}[typ]
    assert np.array_equal(r["raw"].reshape(rows, nbk, -1)[:, :, :hd], raw.reshape(rows, nbk, -1)[:, :, :hd])
    assert np.all(np.abs(u @ W2[:, cols_e]) < 0.25 * np.abs(c)), (u @ W2[:, cols_e], c)   # 方向の成分はほぼ消えた


@pytest.mark.parametrize("typ", ["Q2_0", "NVFP4", "IQ4_NL"])
def test_same_size_bake_several_directions_together(typ):
    """いくつかの方向(直交しない)の部品が同じ列に重なるとき: LoRA の変化に、方向が張る空間の中でまとめて合わせる。どの方向の
    成分も目標に近く、ほかの列と倍率はそのまま、方向に沿わない変化は span の外だけ。"""
    import numpy as np

    from gne.inkling_bake import basis, decode, edit_columns
    rng = np.random.default_rng(3)
    rows, cols = 256, 128
    raw = _random_raw(typ, rows, cols, rng)
    q, sc, W = decode(raw, typ, rows, cols)
    u1 = rng.standard_normal(rows)
    u2 = 0.6 * u1 + 0.8 * rng.standard_normal(rows)                         # 直交しない 2 つの方向
    U = np.stack([u1 / np.linalg.norm(u1), u2 / np.linalg.norm(u2)], axis=1)
    cs = np.array([1, 5, 64, 100])
    A = np.stack([1.0 * (U[:, 0] @ W[:, cs]), 0.5 * (U[:, 1] @ W[:, cs])])  # 強さ 1 と 0.5 の部品(元の重みから)
    T = -U @ A
    r = edit_columns(raw, typ, U, T, cs, rows, cols)
    q2, sc2, W2 = decode(r["raw"], typ, rows, cols)
    other = [n for n in range(cols) if n not in cs]
    assert np.array_equal(q2[:, other], q[:, other]) and np.array_equal(sc2, sc)
    D = W2[:, cs] - W[:, cs]
    want, got = U.T @ T, U.T @ D
    assert np.allclose(got, r["got"]) and np.allclose(want, r["want"])
    assert np.linalg.norm(got - want) < 0.05 * np.linalg.norm(want), (got, want)   # 両方の方向の成分が目標どおり
    Qb = basis(U)
    assert Qb.shape[1] == 2 and np.isclose(r["off2"], np.sum((D - Qb @ (Qb.T @ D)) ** 2))
    assert basis(np.stack([U[:, 0], U[:, 0]], axis=1)).shape[1] == 1               # 同じ向きの方向は 1 つにまとめる


def test_refine_parts_and_chain_with_expert_units(tmp_path):
    """残りから追加を重ねる(Inkling 2 の専門家のニューロンを含む): 部品の書き方・回の番号・同じ部品の見分け、土台のたどり方、
    合わせたレシピの部品の書き方が保存したレシピと同じ(画面が読み込む部品 = 表の数を測ったレシピ)。"""
    import json as _json
    from types import SimpleNamespace

    from gne.api import Session, refine_parts, refine_proj_spec
    from gne.steer import Recipe, parse_unit
    base = [parse_unit(x) for x in ("11:H5:s1.5@d", "14:E25:N389:s2@d", "9:ffn:s1@d")]
    added = [parse_unit(x) for x in ("14:E26:N389:s1@d2", "14:E25:N389:s0.5@d2")]
    assert [refine_proj_spec(u) for u in base] == ["11:H5:s1.5@d", "14:E25:N389:s2@d", "9:ffn:s1@d"]
    parts = refine_parts(base + added, [0, 0, 0, 1, 1], lambda L: "attn")
    assert [(q["kind"], q["head"], q.get("expert"), q["round"]) for q in parts] == [
        ("attn", 5, None, 0), ("neuron", 389, 25, 0), ("layer_mlp", None, None, 0), ("neuron", 389, 26, 1), ("neuron", 389, 25, 1)]
    keys = [(q["layer"], q["kind"], q["head"], q.get("expert")) for q in parts]
    assert keys.count((14, "neuron", 389, 25)) == 2 and keys.count((14, "neuron", 389, 26)) == 1   # 専門家が違えば別の部品
    # 合わせたレシピ: 保存して読み直しても、部品の書き方は表に載せたものと同じ
    comb = Recipe(base + added, "m", name="x_r1-n2-combined", default_direction="d")
    cpath = tmp_path / "x_r1.combined-n2.yaml"
    comb.save(cpath)
    loaded = Recipe.load(cpath)
    assert [refine_proj_spec(u) for u in loaded.units] == [q["spec"] for q in parts]
    assert all(refine_proj_spec(parse_unit(q["spec"])) == q["spec"] for q in parts)       # 画面の読み込み(書き方)から同じ部品に戻る
    # 土台のたどり方: 1 回目は一括の行の組、重ねると前の回の合わせたレシピ(回の番号つき)
    hd = tmp_path / "heads"
    hd.mkdir()
    r0 = Recipe(base, "m", name="x-n3", default_direction="d")
    r0.save(hd / "x.alloc-n3.yaml")
    fake = SimpleNamespace(store=SimpleNamespace(heads_dir=lambda mid: hd), meta=SimpleNamespace(model_id="m"))
    ch1 = Session._refine_chain(fake, "x", {"n_heads": 3, "recipe": str(hd / "x.alloc-n3.yaml")})
    assert (ch1["root"], ch1["round"], ch1["rounds"], ch1["chain"]) == ("x", 1, [0, 0, 0], ["x"])
    (hd / "x_r1.refine.json").write_text(_json.dumps({"root": "x", "round": 1, "chain": ["x", "x_r1"], "base": "x",
                                                      "combined_rows": [{"key": "2", "n_new": 2, "recipe": str(cpath), "parts": parts}]}),
                                         encoding="utf-8")
    ch2 = Session._refine_chain(fake, "x_r1", {"n_heads": 2})
    assert (ch2["root"], ch2["round"], ch2["rounds"], ch2["recipe"]) == ("x", 2, [0, 0, 0, 1, 1], str(cpath))
    assert ch2["chain"] == ["x", "x_r1", "x_r1"]


def test_chips_bake_units_and_compare():
    import numpy as np

    from gne.inkling_chips import bake_compare, units_of
    assert units_of({"units": [[8, 3, 2.0], [14, None, 1.0]]}) == [[8, "head", 3, 2.0], [14, "attn", None, 1.0]]
    assert units_of({"units": [[9, "neuron", [5, 77], 2.0]]}) == [[9, "neuron", [5, 77], 2.0]]
    p = _plan()
    n = p.n_steps
    base = np.ones((2, n))
    lora = {"proj": (base * 0.5).tolist(), "proj_base": base.tolist(), "out": [0.5, 0.5], "out_base": [1.0, 1.0],
            "greedy": {"tokens": [[1, 2, 3], [4, 5]], "n_tokens": 5, "seconds": 1.0}}
    other = {"proj": (base * 0.55).tolist(), "out": [0.45, 0.5], "greedy": {"tokens": [[1, 2, 9], [4, 5]], "n_tokens": 5, "seconds": 2.0}}
    c = bake_compare(p, lora, other)
    assert abs(c["proj_by_region"]["pass3"]["diff_over_edit_median"] - 0.1) < 1e-9          # 0.05 ÷ 0.5
    assert np.allclose(c["out"]["diff_over_edit"], [0.1, 0.0])
    assert c["greedy"]["same"] == 1 and c["greedy"]["first_diff"] == [2, None] and c["greedy"]["tok_per_s"] == [5.0, 2.5]


def test_chips_trial_candidates_and_pick():
    from gne.inkling_chips import candidates, pick
    p = _plan()
    rows = [{"layer": L, "head": h, "value": v} for L, h, v in
            ((3, 0, 5.0), (6, 1, 4.0), (7, 2, 3.0), (12, 0, 2.5), (19, 4, 2.0), (18, 1, 9.0), (8, 0, -1.0), (9, None, 6.0))]
    c = candidates(p, rows)
    assert [(r["layer"], r["head"]) for r in c] == [(7, 2), (19, 4)]          # 0〜5 は入れない、流れの始まりの層・負の値も外す
    c = candidates(p, rows, skip_stream_start=False)
    assert [(r["layer"], r["head"]) for r in c] == [(18, 1), (6, 1), (7, 2), (12, 0), (19, 4)]
    assert [(r["layer"], r["head"]) for r in candidates(p, rows, regions=("before",))] == [(3, 0)]
    assert [r["layer"] for r in candidates(p, rows, whole=True)] == [9]
    rep = lambda m, e=0: {"counts": {"match": m, "empty": e, "loop": 0, "at_limit": 0, "n": 4}}  # noqa: E731
    res = {"base": {"A": rep(4), "B": rep(0)},
           "sets": {"x": {"A": rep(1), "B": rep(0), "kl_b": 0.1}, "y": {"A": rep(0, 4), "B": rep(0), "kl_b": 0.0},
                    "z": {"A": rep(1), "B": rep(0), "kl_b": 0.05}, "z_naive": {"A": rep(0), "B": rep(0), "kl_b": 0.0}}}
    assert pick(res) == ("z", "y")                               # y は壊れた応答が増えすぎ、素直な射影の組は選ばない


class _FakeEngine:
    """ReplyMarks が使う分だけ: 特別なトークン 1 つを 1 つの番号に、ほかは 1 文字 = 1 トークン。"""
    SPECIAL = {"<|content_thinking|>": 1001, "<|content_text|>": 1002, "<|message_model|>": 1003, "<|end_message|>": 1004,
               "<|content_model_end_sampling|>": 1005}

    def tokenize(self, text, parse_special=True):
        return [self.SPECIAL[text]] if text in self.SPECIAL else [ord(c) for c in text]

    def detokenize(self, ids):
        return "".join(chr(t) for t in ids if t < 1000)

    def is_eog(self, t):
        return t == 1005


def _ids(*parts):
    e = _FakeEngine()
    return [x for p in parts for x in e.tokenize(p)]


def test_reply_marks_split_thinking():
    e = _FakeEngine()
    ids = _ids("<|message_model|>", "<|content_thinking|>", "plan", "<|end_message|>", "<|message_model|>", "<|content_text|>",
               "Hi!", "<|end_message|>", "<|content_model_end_sampling|>")
    m = ReplyMarks(e)
    shown = m.render(e, ids)
    assert shown == "<think>\nplan\n</think>\n\nHi!"
    assert answer_part(shown) == "Hi!"
    # 逐次表示: 区切りのトークンだけが文字になる
    m2 = ReplyMarks(e)
    marks = [m2.piece(t) for t in ids]
    assert [x for x in marks if x] == ["<think>\n", "\n</think>\n\n"]
    # 思考の途中で切れた返事は回答が空
    cut = m.render(e, _ids("<|content_thinking|>", "still thinking"))
    assert cut.startswith("<think>\n") and answer_part(cut) == ""
    # 思考の無い返事はそのまま
    assert answer_part("plain") == "plain" and m.render(e, _ids("<|content_text|>", "ok")) == "ok"


# --- 実物 ----------------------------------------------------------------------------------------------------------

MODEL = os.environ.get("GNE_TEST_INKLING2")
real = pytest.mark.skipif(not MODEL or not Path(MODEL).exists(), reason="GNE_TEST_INKLING2 に Inkling 2 の GGUF がありません")


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    from gne.api import Session
    try:
        s = Session(MODEL, root=tmp_path_factory.mktemp("root"), n_gpu_layers=int(os.environ.get("GNE_TEST_NGL", "-1")),
                    n_ctx=2048, n_seq_max=1)
        s.engine
    except (RuntimeError, GneError) as ex:
        pytest.skip(f"この llama.cpp では読めません: {ex}")
    yield s
    s.close()


@real
def test_real_weights_and_arch():
    from gne.arch import arch_info
    from gne.weights import Weights
    w = Weights(MODEL)
    assert w.loop is None and w.inkling is not None and w.inkling.n_steps == 48
    a = arch_info(w)
    assert a.n_layer == 24 and a.inkling["n_steps"] == 48 and a.loop is None
    assert all(li.mixer == "attention" and li.n_head == 24 and li.n_head_kv == 8 and li.head_dim == 128 for li in a.layers)
    assert all(li.ffn == "moe" and li.n_expert == 128 and li.n_expert_used == 8 and li.n_ff == 1152 for li in a.layers)
    assert all(li.routing == "sigmoid_topk" and li.norm_topk for li in a.layers)
    assert a.inkling["full_layers"] == [5, 11, 17, 23]
    warns = w.check_arch()
    assert len(warns) == 2 and "層 6〜17 を 3 回" in warns[0]
    # 重みの中身を読む処理は止まる
    for fn in (lambda: w.down_column(6, 5), lambda: w.head_write_matrix(6, 0), lambda: w.router(6),
               lambda: w.expert_down(6, 3), lambda: next(w.output_rows()), lambda: w.tensor("blk.6.attn_q.weight"),
               lambda: w.reader):
        with pytest.raises(GneError) as e:
            fn()
        assert e.value.code == "ARCH_UNSUPPORTED"


@real
def test_real_engine_guards_and_graph_order(session):
    import numpy as np

    from gne.capture import Capture
    e = session.engine
    assert e.inkling is not None and not e.cvec_supported
    assert e.n_layer == e.inkling.n_steps == 48               # llama.cpp の n_layer は実行順の段の数
    with pytest.raises(GneError) as ex:
        e.probe_ctx(Capture(lambda n: n.startswith("l_out-")))
    assert ex.value.code == "ARCH_UNSUPPORTED"
    with pytest.raises(GneError):
        e.set_cvec(e.steer_ctx(), {6: np.ones(e.n_embd, np.float32)})
    e.set_cvec(e.steer_ctx(), None)                           # 外すのはいつもよい
    with pytest.raises(GneError):
        e.lora("missing.lora.gguf")
    e.set_lora(e.steer_ctx(), None)
    # 名前だけの読み(gne probe --list-names)は使える。グラフの中の順は実行の順(番号の順ではない)
    ids = e.tokenize_prompt(e.render_chat([{"role": "user", "content": "Hello there."}]))
    with pytest.raises(GneError):
        e.list_names(ids)                                     # 名前を調べた後に値を読む処理の入口なので、ふだんは止める
    e.allow_names = True
    names = e.list_names(ids)
    e.allow_names = False
    assert {f"l_out-{k}" for k in range(48)} <= names        # 名前は実行順の段に読み替える
    # グラフの中の順が実行の順(llama.cpp の番号の順ではない)なら、読み替えた名前は 0〜47 の順に並ぶ
    order = [int(n.split("-")[1]) for n in e._probe.capture.seen if re.fullmatch(r"l_out-\d+", n)]
    assert order == list(range(48))
    e._probe.free()
    e._probe = None


@real
def test_real_info(session):
    d = session.info(load=True)
    ink = d["meta"]["inkling"]
    assert ink["n_steps"] == 48 and d["meta"]["cvec_supported"] is False and d["meta"]["internals_supported"] is False
    assert d["llama_cpp"]["ok"] and d["llama_cpp"]["n_layer"] == 48
    assert not d["bake"]["supported"] and d["residual_writers"] is None
    assert "ffn_down_exps" in ink["experts"]["types"] and ink["expert_used"] == 8
    assert not any("n_layer" in w for w in session.warnings)


@real
def test_real_chat_and_behave(session, tmp_path):
    from gne.chat import ChatState, Column, chat_turn
    st = ChatState([Column("0", None)])
    rep = chat_turn(session, st, "How is the weather today?", max_tokens=400)[0]
    assert rep["n_gen"] > 0 and rep["text"].startswith("<think>\n")          # このモデルは答えの前に思考を書く
    assert st.histories[0][-1]["content"] == answer_part(rep["text"])         # 履歴には回答だけ(思考の欄は入れない)
    assert "<think>" not in st.histories[0][-1]["content"]
    with pytest.raises(GneError) as e:
        session.behave(["Hello"], judge="{q} {a}", max_tokens=8, name="t")
    assert e.value.code == "ARCH_UNSUPPORTED"
    d = session.behave(["How is the weather today?"], match=r"weather", max_tokens=48, name="t")
    assert d["runs"][0]["counts"]["n"] == 1


def _reader(session):
    from gne.inkling_read import Reader
    r = Reader(session)
    if not r.patch_present():
        pytest.skip("この llama.cpp には third_party/inkling2/0001 の名前がありません")
    return r


@real
def test_real_reader_without_patch_refuses(session):
    from gne.inkling_read import Reader
    r = Reader(session)
    if r.patch_present():
        pytest.skip("0001 を当てた llama.cpp なので、断りは確かめられません")
    with pytest.raises(GneError) as e:
        r.require_patch()
    assert e.value.code == "ARCH_UNSUPPORTED" and "0001" in e.value.hint


@real
def test_real_check_against_independent_math(session):
    from gne.inkling_read import check
    d = check(_reader(session), "How is the weather today?", steps=[1, 6, 11, 18, 23, 35, 42, 47], n_tok_expert=1)
    err = d["errors"]
    for k, v in err.items():
        limit = 3e-2 if any(x in k for x in ("W_o", "W_down", "W_gate_up", "SiTU")) else 2e-3
        assert v["max"] < limit, (k, v)
    assert any("繰り返し" in k for k in err) and any("専門家" in k for k in err)


@real
def test_real_study_and_mix(session):
    import numpy as np

    from gne.inkling_read import experts, mix_summary, study
    r = _reader(session)
    A = [[{"role": "user", "content": q}] for q in ("今日の天気はどうですか？", "おすすめの本を教えてください。")]
    B = [[{"role": "user", "content": q}] for q in ("How is the weather today?", "Can you recommend a book?")]
    m = mix_summary(r, A + B, positions="last")
    for st in m["steps"]:
        for side in ("attn", "ffn", "msg"):
            if side in st:
                np.testing.assert_allclose(np.sum(st[side]["weights"], axis=-1), 1.0, atol=1e-4)
    d = study(r, A, B)
    assert len(d["steps"]) == 48 and d["head_sum_rel_max"] < 3e-2       # W_o の積は量子化した重みと活性(相対 1e-2 程度)
    assert {x["region"] for x in d["steps"]} == {"before", "loop", "after"}
    assert all("l_out|ffn_mix" in x["cos"] for x in d["steps"])
    assert {x["pass"] for x in d["loop"]} == {2, 3}
    e = experts(r, A[:1])
    assert len(e["steps"]) == 48 and e["overlap"]


@real
def test_real_reasoning_effort_reaches_the_prompt(session):
    e = session.engine
    with e.with_template_vars(reasoning_effort="none"):
        text = e.render_chat([{"role": "user", "content": "Hi"}])
    assert "Thinking effort level: 0<" in text
    assert "Thinking effort level: 0.9" in e.render_chat([{"role": "user", "content": "Hi"}])


@real
def test_real_chips_trial_lora_and_tiny_run(session, tmp_path):
    from gne.inkling_chips import check_lora, directions, layer_refs, load_spec, trial
    r = _reader(session)
    session.engine.allow_lora_trial = True
    try:
        sp = load_spec(Path(__file__).resolve().parent.parent / "configs" / "pipeline" / "pirate_tone.json")
        D = directions(r, sp["build_a"][:2], sp["build_b"][:2])
        refs = layer_refs(r.plan, D, "after")
        assert check_lora(r, refs, 14, 3, 2.0, tmp_path / "one.lora.gguf") < 1e-3   # LoRA の変化が計算どおり
        res = trial(session, Path(__file__).resolve().parent.parent / "configs" / "pipeline" / "pirate_tone.json", tmp_path / "t",
                    ks=(2,), strengths=(1.0,), whole_ks=(), whole_strengths=(), n_build=2, n_val_a=1, n_val_b=1, n_kl=1,
                    max_tokens=16, region_k=2)
        assert "heads_k2_s1" in res["sets"] and res["sets"]["heads_k2_s1"]["keep_out"] is not None
        assert res["best"] is None or res["best"] in res["sets"]
    finally:
        session.engine.allow_lora_trial = False
    with pytest.raises(GneError):
        session.engine.lora(tmp_path / "t" / "heads_k2_s1.lora.gguf")                  # 試しの外では LoRA は止まる


@real
def test_real_chips_trial_ffn_lora(session, tmp_path):
    from gne.inkling_chips import check_ffn_lora, directions, layer_refs, load_spec
    r = _reader(session)
    session.engine.allow_lora_trial = True
    try:
        sp = load_spec(Path(__file__).resolve().parent.parent / "configs" / "pipeline" / "pirate_tone.json")
        refs = layer_refs(r.plan, directions(r, sp["build_a"][:2], sp["build_b"][:2]), "after")
        for units in ([[9, "neuron", [3, 10], 1.0], [9, "neuron", [40, 7], 2.0]], [[22, "ffn", None, 1.0]]):
            ck = check_ffn_lora(r, refs, units, tmp_path / "f.lora.gguf")
            assert ck["rel_post"] < 1e-3 and ck["rel_pre"] < 1e-3, ck            # 専門家ごとの倍率を差分に入れた式どおり
    finally:
        session.engine.allow_lora_trial = False


@real
def test_real_chips_bake_types(session):
    """焼き込みの型: q8_0・f16・keep のどれでも編集が残る(ファイルは書かない)。"""
    from types import SimpleNamespace

    import numpy as np

    from gne.inkling_chips import bake, layer_refs
    plan = session.weights.inkling
    rng = np.random.default_rng(0)
    D = {k: rng.standard_normal((plan.n_steps, session.weights.arch.n_embd)) for k in ("ffn_mix", "ffn_in")}
    D["out"] = rng.standard_normal(session.weights.arch.n_embd)
    refs = layer_refs(plan, D, "after")
    rd = SimpleNamespace(w=session.weights, plan=plan)
    units = [[11, "head", 5, 2.0], [17, "neuron", [3, 10], 2.0], [22, "neuron", [7, 1], 2.0]]
    for rq, tol in (("q8_0", 0.1), ("f16", 0.01)):            # 1 列だけの小さな編集は Q8_0 の刻みで数 % ずれる
        for m in bake(rd, units, refs, None, requant=rq)["modified"]:
            assert abs(m["edit_kept"] - 1.0) < tol, (rq, m)
    for m in bake(rd, units, refs, None, requant="keep")["modified"]:       # 元の型・元の大きさ(専門家は列のコードだけ動かす)
        assert m["to"] == m["from"] and m["bytes_to"] == m["bytes_from"] and m["edit_kept"] > 0.9, m


@real
def test_real_inkling_pipeline_recipe_and_bake(session, tmp_path):
    """gne heads pipeline(Inkling 2)を小さく流し、レシピが LoRA で当たり、強さの倍率が式どおりに効き、焼き込み(元の型・
    元の大きさ)の確かめが通る(ファイルは書かない)。"""
    import json as _json
    from types import SimpleNamespace

    from gne.headproj import proj_heads
    from gne.inkling_chips import bake
    from gne.inkling_pipeline import OUT, chips_unit, layer_refs_of
    if not _reader(session).patch_present():
        pytest.skip("名前の patch を当てた llama.cpp だけ")
    spec = _json.loads((Path(__file__).resolve().parent.parent / "configs" / "pipeline" / "pirate_tone.json").read_text(encoding="utf-8"))
    spec["user_prompts"] = spec["user_prompts"][:6]
    spec.pop("validate", None)
    d = session.heads_pipeline(spec, max_examples=4, verify_top=3, verify_sample=0, n_units=[2], check_replies=0)
    assert d["inkling"] and d["writes"]["select_layer"] == OUT and d["allocate"]["table"]
    row = next(r for r in d["allocate"]["table"] if not r.get("skipped"))
    rec = session.recipe(path=row["recipe"])
    assert session.lora_for(rec)                                               # LoRA で当たる
    ds = session.direction_set(spec["name"])
    units = [chips_unit(p) for p in proj_heads(rec) if p.s]
    refs = layer_refs_of(session.weights.inkling, ds)
    from gne.inkling_chips import whole_ffn_layers
    whole = whole_ffn_layers(units)
    from gne.inkling_bake import OFF_MAX
    for m in bake(SimpleNamespace(w=session.weights, plan=session.weights.inkling), units, refs, None, requant="auto")["modified"]:
        fb = m.get("fallback") or {}
        if m["to"] != m["from"]:                       # Q8_0 にするのは理由があるときだけ(MLP 全体の層は方向に沿わない変化で決める)
            assert m["to"] == "Q8_0" and fb.get("why") in ("edit_kept", "off_direction", "imatrix") or "edit_kept" in fb, m
            if fb.get("why") == "off_direction":
                assert int(m["tensor"].split(".")[1]) in whole and fb["off_over_removed"] > OFF_MAX, m
        else:
            assert m["bytes_to"] == m["bytes_from"] and m["edit_kept"] > 0.9, m
            if "ffn_down_exps" in m["tensor"] and int(m["tensor"].split(".")[1]) in whole:
                assert m["whole_layer"] and m["off_over_removed"] <= OFF_MAX, m


@real
def test_real_inkling_refine_stacks(session):
    """残りから追加(Inkling 2): 土台を当てたまま方向を作り直して足す組を探し、合わせたレシピ(土台 + 足した組)の数を元の方向で測り、
    重ねると前の回の部品を残す。合わせたレシピの部品は保存したレシピと同じ(画面が読み込む部品 = 表の数を測ったレシピ)。"""
    import json as _json

    from gne.api import refine_proj_spec
    from gne.steer import Recipe
    if not _reader(session).patch_present():
        pytest.skip("名前の patch を当てた llama.cpp だけ")
    spec = _json.loads((Path(__file__).resolve().parent.parent / "configs" / "pipeline" / "pirate_tone.json").read_text(encoding="utf-8"))
    spec.update(name="t_refine", user_prompts=spec["user_prompts"][:6])
    spec.pop("validate", None)
    d = session.heads_pipeline(spec, max_examples=6, verify_top=3, verify_sample=0, n_units=[3], check_replies=0, unit_kinds="heads,layers")
    row = next(r for r in d["allocate"]["table"] if not r.get("skipped"))
    try:
        r1 = session.heads_refine("t_refine", n_heads=int(row["n_heads"]), new_heads=(2,), check_replies=0)
    except Exception as e:                                                   # 土台で全部の質問が目標まで下がったとき
        if getattr(e, "code", "") == "NOTHING_LEFT":
            pytest.skip("土台で残りが無い")
        raise
    assert r1["round"] == 1 and r1["combined_rows"] and r1["inkling"]["score_label"]
    for c in r1["combined_rows"]:
        assert [refine_proj_spec(u) for u in Recipe.load(c["recipe"]).units] == c["units"]
        assert c["n_base"] == len(r1["base_units"]) and c["units"][:c["n_base"]] == r1["base_units"]
    key = next((c["key"] for c in r1["combined_rows"] if c["key"] != "s1"), None)
    if key is None:
        pytest.skip("重ねる行が無い")
    try:
        r2 = session.heads_refine(r1["name"], n_heads=int(key), new_heads=(2,), check_replies=0)
    except Exception as e:
        if getattr(e, "code", "") == "NOTHING_LEFT":
            return
        raise
    c1 = next(c for c in r1["combined_rows"] if c["key"] == key)
    assert r2["round"] == 2 and r2["base_units"] == c1["units"]                # 前の回の合わせたレシピが土台
    for c in r2["combined_rows"]:
        assert c["units"][:len(c1["units"])] == c1["units"] and max(p["round"] for p in c["parts"]) == 2


def test_bake_auto_rule_whole_ffn_layers():
    """焼き込みの auto: 元の型で書いて測り、編集が残らないテンソルと、MLP 全体の部品(L:ffn)を受けて方向に沿わない変化 ÷ 消した
    成分が上限を超える専門家のテンソルだけ Q8_0。ニューロンだけの層は残り方だけで決める。予定の文に、元の大きさのまま書く
    テンソルと Q8_0 にする理由が出る。"""
    from gne.inkling_bake import OFF_MAX
    from gne.inkling_chips import auto_reason, whole_ffn_layers
    from gne.inkling_pipeline import bake_plan_text, plan_entry
    assert auto_reason({"edit_kept": 0.99, "off_over_removed": 1.0}, True) is None
    assert auto_reason({"edit_kept": 0.99, "off_over_removed": OFF_MAX + 0.1}, True) == "off_direction"
    assert auto_reason({"edit_kept": 0.99, "off_over_removed": OFF_MAX + 0.1}, False) is None     # ニューロンだけ
    assert auto_reason({"edit_kept": 0.85, "off_over_removed": 1.0}, False) == "edit_kept"
    recs = [{"tensor": "blk.17.ffn_down_exps.weight", "from": "NVFP4", "to": "NVFP4", "bytes_from": 182 * 2**20, "bytes_to": 182 * 2**20,
             "whole_layer": True, "edit_kept": 0.993, "off_over_removed": 1.01},
            {"tensor": "blk.14.ffn_down_exps.weight", "from": "Q2_0", "to": "Q8_0", "bytes_from": 91 * 2**20, "bytes_to": 344 * 2**20,
             "whole_layer": True, "fallback": {"from": "Q2_0", "why": "off_direction", "edit_kept": 0.996, "off_over_removed": 2.41,
                                               "off_max": OFF_MAX}}]
    text = bake_plan_text({"bytes": 253 * 2**20, "tensors": [plan_entry(m) for m in recs]})
    assert "元の大きさのまま: 層 17 の専門家 NVFP4(MLP 全体の部品。方向に沿わない変化 ÷ 消した成分 1.01" in text, text
    assert "層 14 の専門家 Q2_0 → Q8_0(+253 MiB、方向に沿わない変化 ÷ 消した成分 2.41 が上限" in text and "+253 MiB" in text
    units = [[9, "ffn", None, 1.0, "d"], [9, "neuron", [3, 10], 2.0, "d2"], [14, "neuron", [25, 389], 4.0], [11, "head", 5, 2.0],
             [17, "ffn", None, 0.5], [23, "attn", None, 1.0]]
    assert whole_ffn_layers(units) == {9, 17}
    assert whole_ffn_layers([[14, "neuron", [1, 2], 1.0], [11, "attn", None, 1.0]]) == set()
    grow = {"bytes": int((162.0 + 253.1) * 2**20), "tensors": [
        {"tensor": "blk.9.ffn_down_exps.weight", "type": "NVFP4", "to": "Q8_0", "from_mib": 182.2, "to_mib": 344.2},
        {"tensor": "blk.14.ffn_down_exps.weight", "type": "Q2_0", "to": "Q2_0", "from_mib": 91.1, "to_mib": 91.1},
        {"tensor": "blk.22.ffn_down_exps.weight", "type": "Q2_0", "to": "Q8_0", "from_mib": 91.1, "to_mib": 344.2},
        {"tensor": "blk.11.attn_output.weight", "type": "Q8_0", "to": "Q8_0", "from_mib": 7.2, "to_mib": 7.2}]}
    text = bake_plan_text(grow)
    assert "層 9 の専門家 NVFP4 → Q8_0(+162 MiB)" in text and "層 22 の専門家 Q2_0 → Q8_0(+253 MiB)" in text
    assert "層 14" not in text and "+415 MiB" in text
    assert "元の型・元の大きさのまま" in bake_plan_text({"bytes": 0, "tensors": grow["tensors"][1:2]})
    # 繰り返しの層(6〜17)の MLP 全体の部品を元の大きさのまま書くときだけ、3 回の段に効く注意と Q8_0 での避け方を足す(止めない)
    from types import SimpleNamespace
    from gne.inkling_pipeline import _mark_loop, loop_note
    plan = SimpleNamespace(core_start=6, core_end=18, n_passes=3)
    g = {"bytes": 0, "tensors": [plan_entry({"tensor": "blk.17.ffn_down_exps.weight", "from": "NVFP4", "to": "NVFP4",
                                                       "bytes_from": 1, "bytes_to": 1, "whole_layer": True, "off_over_removed": 1.01}),
                                 plan_entry({"tensor": "blk.22.ffn_down_exps.weight", "from": "Q2_0", "to": "Q2_0", "bytes_from": 1,
                                             "bytes_to": 1, "whole_layer": True, "off_over_removed": 1.93})]}
    _mark_loop(g, plan)
    assert [x["loop_layer"] for x in g["tensors"]] == [True, False] and g["loop"]["passes"] == 3
    note = loop_note(g)
    assert note and "層 17 の MLP 全体" in note and "層 22" not in note and "3 回すべての段" in note and "--requant q8_0" in note
    assert note in bake_plan_text(g)
    g2 = {"bytes": 0, "tensors": [dict(g["tensors"][0], to="Q8_0")]}
    _mark_loop(g2, plan)
    assert loop_note(g2) is None                                 # Q8_0 にしたなら注意は出さない
    # keep で MLP 全体の部品を元の型(IQ4_NL)に書いて編集が丸めに消えたときは注意を出す(auto なら Q8_0 になるので出ない)
    from gne.inkling_pipeline import kept_note
    recs = [{"tensor": "blk.22.ffn_down_exps.weight", "to": "IQ4_NL", "edit_kept": 0.5643},
            {"tensor": "blk.9.ffn_down_exps.weight", "to": "NVFP4", "edit_kept": 1.0014},
            {"tensor": "blk.11.attn_output.weight", "to": "Q8_0", "edit_kept": None}]
    note = kept_note(recs)
    assert note and "blk.22.ffn_down_exps.weight(IQ4_NL)は 56%" in note and "blk.9" not in note
    assert kept_note(recs[1:]) is None and kept_note([]) is None


def test_into_same_type_rules_and_target_check():
    """別の量子化に焼き込むとき: 型ごとの元の型での書き方と、同じモデルかの確かめ(名前・形・層の並び)。"""
    from types import SimpleNamespace

    from gguf import GGMLQuantizationType as Q

    from gne.inkling_chips import same_type_method
    from gne.inkling_pipeline import target_weights
    from gne.quant import qtype_of
    from gne.steer import GneError
    assert same_type_method(qtype_of(42)) == "par" and same_type_method(SimpleNamespace(name="NVFP4")) == "par"
    assert same_type_method(Q.IQ4_NL) == "par" and same_type_method(Q.MXFP4) == "par"
    for q in (Q.Q8_0, Q.Q4_K, Q.Q5_K, Q.Q6_K, Q.Q3_K):
        assert same_type_method(q) == "requant", q
    for q in (Q.IQ2_XXS, Q.IQ2_XS, Q.IQ1_S):
        assert same_type_method(q) == "imatrix", q

    def T(shape):
        return SimpleNamespace(shape=shape)
    plan = _plan()
    src = SimpleNamespace(inkling=plan, _tensors={"a": T([4, 8]), "b": T([8])}, path="src.gguf")
    ok = SimpleNamespace(inkling=_plan(), _tensors={"a": T([4, 8]), "b": T([8])}, path="t.gguf")
    sess = SimpleNamespace(weights=src, _target_weights=lambda into: {"ok": ok, "same": src}.get(into) or bad[into])
    bad = {"shape": SimpleNamespace(inkling=_plan(), _tensors={"a": T([4, 9]), "b": T([8])}, path="x.gguf"),
           "names": SimpleNamespace(inkling=_plan(), _tensors={"a": T([4, 8])}, path="y.gguf"),
           "plan": SimpleNamespace(inkling=_plan(**{f"{ARCH}.loop.count": 2}), _tensors={"a": T([4, 8]), "b": T([8])}, path="z.gguf")}
    assert target_weights(sess, "ok") is ok and target_weights(sess, "same") is src
    for k in ("shape", "names", "plan"):
        with pytest.raises(GneError):
            target_weights(sess, k)


INTO = [p for p in os.environ.get("GNE_TEST_INKLING2_INTO", "").split(",") if p.strip()]


@pytest.mark.skipif(not MODEL or not Path(MODEL).exists() or not INTO, reason="GNE_TEST_INKLING2 と GNE_TEST_INKLING2_INTO(焼き込む先の GGUF、カンマ区切り)が要ります")
@pytest.mark.parametrize("target", INTO or ["-"])
def test_real_bake_into_other_quant_dry_run(tmp_path, target):
    """別の量子化に焼き込む(書かずに計算だけ、GPU は使わない): 同じモデルの確かめ、短い畳み込みのタップが同じ、型ごとの決め方
    (元の型で書くテンソルは大きさが同じ、Q8_0 にするのは理由があるときだけ)、消す量を焼き込む先の重みから作る(部品ごとの違いを出す)。"""
    import numpy as np

    from gne.api import Session
    from gne.inkling_pipeline import taps_identical, target_weights
    if not Path(target).exists():
        pytest.skip(f"{target} がありません")
    if Path(target).resolve() == Path(MODEL).resolve():
        pytest.skip("焼き込む先が元のモデルと同じファイル")
    s = Session(MODEL, root=tmp_path / "root", n_gpu_layers=0)
    try:
        plan = s.weights.inkling
        rng = np.random.default_rng(3)
        d = s.weights.meta.n_embd
        Ls = list(range(plan.n_steps + 1))
        D = {L: rng.standard_normal(d).astype(np.float32) for L in Ls}
        p = s.store.direction_dir(s.meta.model_id) / "rnd.npz"
        np.savez(p, layers=np.array(Ls, dtype=np.int32), **{f"layer_{L}": v for L, v in D.items()},
                 **{f"scale_{L}": np.ones(d, np.float32) for L in Ls}, method="inkling_ffn_input", positions="last", contrast_sha="x",
                 source="test", n_rows_A=1, n_rows_B=1, inkling_gate="after", same_reply=False)
        rec = s.recipe(units=["11:H5:s2@rnd", "9:ffn:s1@rnd", "14:E25:N389:s4@rnd", "20:E1:N3:s4@rnd"], name="into_t")
        tw = target_weights(s, target)
        taps = taps_identical(s.weights, tw)
        assert not taps["different"] and taps["target_types"] == ["F32"]
        r = s.headproj_export(rec, None, bake=True, into=target, dry_run=True)
        assert r["dry_run"] and len(r["carry_over"]) == 4 and all(np.isfinite(x["norm_ratio"]) for x in r["carry_over"])
        for m in r["modified"]:
            if m["to"] == m["from"]:
                assert m["bytes_to"] == m["bytes_from"] and m["edit_kept"] > 0.9, m
            else:
                assert m["to"] == "Q8_0" and m.get("fallback"), m              # 元の型で書けない・編集が残らない・方向に沿わない変化が大きい
        assert "元の大きさ" in r["plan"] or "型を変える" in r["plan"], r["plan"]
    finally:
        s.close()


def test_missing_direction_points_to_the_model_it_was_made_on(tmp_path):
    """-m に別の量子化を渡して方向が見つからないとき: 同じモデル(general.name・arch が同じ)の別のファイルの置き場所にあれば、
    そのファイルを -m に、焼き込む先を --into にと言う(Inkling 2 はそのままの形)。どこにも無ければいつもの文。"""
    import json as _json
    from types import SimpleNamespace

    from gne.api import Session
    from gne.steer import GneError
    from gne.store import Store
    st = Store(tmp_path / "root")
    md = st.root / "models"
    md.mkdir(parents=True)
    for mid, path, name, arch in (("ink-src", "models/Ink-IQ2_XXS.gguf", "Ink", ARCH), ("ink-q2", "models/Ink-Q2_K_XL.gguf", "Ink", ARCH),
                                  ("other", "models/Other.gguf", "Other", ARCH)):
        (md / f"{mid}.json").write_text(_json.dumps({"model_id": mid, "path": path, "name": name, "arch": arch}))
    (st.root / "directions" / "ink-src").mkdir(parents=True)
    (st.root / "directions" / "ink-src" / "tone.npz").write_bytes(b"x")
    (st.root / "directions" / "other").mkdir(parents=True)
    (st.root / "directions" / "other" / "tone2.npz").write_bytes(b"x")
    assert [o["model_id"] for o in st.direction_elsewhere("tone", "ink-q2", model_name="Ink", arch=ARCH)] == ["ink-src"]
    assert st.direction_elsewhere("tone2", "ink-q2", model_name="Ink", arch=ARCH) == []          # 別のモデルの方向は数えない
    me = SimpleNamespace(store=st, meta=SimpleNamespace(model_id="ink-q2", path=Path("models/Ink-Q2_K_XL.gguf"), name="Ink", arch=ARCH))
    with pytest.raises(GneError) as e:
        Session.direction_path(me, "tone")
    assert e.value.code == "FILE_NOT_FOUND" and "models/Ink-IQ2_XXS.gguf" in e.value.message
    assert ("gne -m models/Ink-IQ2_XXS.gguf headproj export --recipe <レシピ> --bake --into models/Ink-Q2_K_XL.gguf --out <GGUF>"
            in e.value.hint)
    with pytest.raises(GneError) as e:
        Session.direction_path(me, "nothing")
    assert e.value.message == "方向 nothing がありません" and "gne direction" in e.value.hint
    me.meta.arch = "llama"                                                                        # ほかのモデル: -m の直し方を言う
    (md / "ink-src.json").write_text(_json.dumps({"model_id": "ink-src", "path": "models/Ink-IQ2_XXS.gguf", "name": "Ink", "arch": "llama"}))
    with pytest.raises(GneError) as e:
        Session.direction_path(me, "tone")
    assert "gne -m models/Ink-IQ2_XXS.gguf" in e.value.hint and "--into models/Ink-Q2_K_XL.gguf" in e.value.hint


def test_bake_checks_free_space_at_the_output(tmp_path, monkeypatch):
    """焼き込みを書く前に、書き込み先のファイルシステムの空きを確かめる(同じ名前の古いファイルは切り詰めるので空きに数える)。
    予定の文に書くファイルの大きさと空き、WSL の Linux 側のディスクならそのことを出す。足りなければ書く前に止める。"""
    import shutil
    from collections import namedtuple

    import gne.inkling_pipeline as ip
    from gne.steer import GneError
    du = namedtuple("du", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: du(100 * 2**30, 90 * 2**30, 10 * 2**30))
    monkeypatch.setattr(ip, "_wsl", lambda: True)
    dst = tmp_path / "sub" / "out.gguf"                              # まだ無いフォルダ: ある親で測る
    sp = ip.out_space(dst, 8 * 2**30)
    assert sp["free"] == 10 * 2**30 and sp["wsl_linux"] == (not str(dst.resolve()).startswith("/mnt/"))
    ip.check_space(sp, dst)
    text = ip.space_text(sp)
    assert "書くファイル 8.00 GiB(書き込み先の空き 10.0 GiB" in text and ("WSL の Linux 側のディスク" in text) == sp["wsl_linux"]
    with pytest.raises(GneError) as e:
        ip.check_space(ip.out_space(dst, 10 * 2**30), dst)          # ゆとり(256 MiB)を残せない
    assert e.value.code == "NO_SPACE"
    dst.parent.mkdir()
    with open(dst, "wb") as f:                                       # 同じ名前の古いファイル(書くときに切り詰める)
        f.truncate(2 * 2**30)
    ip.check_space(ip.out_space(dst, 10 * 2**30), dst)
    monkeypatch.setattr(ip, "_wsl", lambda: False)
    assert "WSL" not in ip.space_text(ip.out_space(dst, 1))


def test_eval_reads_the_answer_after_the_reasoning():
    """評価(Inkling 2): 返事を思考の後の回答に切る(回答の長さで切る・思考の途中で切れたら空・壊れ方の印)、壊れた返事の数、
    問題の間引き、まだ使えないものの断り。"""
    from types import SimpleNamespace

    from gne.evalx import Suite, _ink_refuse, broken_counts, ink_answer, parse_items, thin_suite
    from gne.steer import GneError, Recipe, parse_unit
    E = SimpleNamespace(is_eog=lambda t: t == 0, detokenize=lambda ids: " ".join(f"w{t}" for t in ids))
    T = 99                                                            # <|content_text|> の代わり
    o = {"token_ids": [5, 6, 7, T, 11, 12, 13, 0], "finish": "eog"}
    a = ink_answer(E, o, 32, T)
    assert a["text"] == "w11 w12 w13" and a["token_ids"] == [11, 12, 13] and a["finish"] == "eog"
    assert a["answer"] and a["reasoning_tokens"] == 3 and not a["cut_answer"] and a["answer_at"] == 4
    a2 = ink_answer(E, o, 2, T)                                       # 回答の長さで切る(採点は最初の 2 トークン)
    assert a2["token_ids"] == [11, 12] and a2["finish"] == "length" and not a2["cut_answer"]
    a3 = ink_answer(E, {"token_ids": [5] * 40, "finish": "length"}, 32, T)   # 思考の途中で上限(回答が始まらない)
    assert a3["text"] == "" and not a3["answer"] and a3["loop"] and a3["reasoning_tokens"] == 40
    a4 = ink_answer(E, {"token_ids": [5, 6, T, 11], "finish": "length"}, 32, T)   # 回答の途中で全体の上限
    assert a4["cut_answer"] and a4["finish"] == "length"
    b = broken_counts([a, a3, a4])
    assert b == {"n": 3, "empty": 1, "no_answer": 1, "loop": 1, "cut_answer": 1, "reasoning_median": 3.0}
    assert parse_items(None) == {"exact": 8, "format": 4, "mix": 4, "persona": 4} and parse_items("all") is None
    assert parse_items("2,1,0,1") == {"exact": 2, "format": 1, "mix": 0, "persona": 1}
    with pytest.raises(GneError):
        parse_items("1,2")
    st = Suite("s", Path("x"), {}, {}, [{"id": i} for i in range(24)], [{"id": i} for i in range(12)], [{"id": i} for i in range(8)],
               [{"id": i} for i in range(8)], [{"id": "p"}])
    t = thin_suite(st, parse_items("8,4,0,4"))
    assert [x["id"] for x in t.exact] == [0, 3, 7, 10, 13, 16, 20, 23] and len(t.format) == 4 and t.mix == [] and len(t.persona) == 4
    assert t.ppl == st.ppl and thin_suite(st, None) is st
    r = Recipe([parse_unit("11:H5:s2@d")], "m")
    _ink_refuse(r, when=None, lora_when="all", random_controls=0, head_controls=0)
    for kw in ({"random_controls": 2}, {"lora_when": "reply"}, {"when": "gen"}):
        with pytest.raises(GneError) as e:
            _ink_refuse(r, **{"when": None, "lora_when": "all", "random_controls": 0, "head_controls": 0, **kw})
        assert e.value.code == "ARCH_UNSUPPORTED"


def test_reasoning_token_count():
    """思考の長さ(画面の思考の欄の見出し・gne chat --hide-reasoning・behave の reasoning_tokens): 思考の始まりから回答の区切りまで。"""
    from types import SimpleNamespace

    from gne.inkling import reasoning_tokens
    mk = SimpleNamespace(think=7, text=9)
    assert reasoning_tokens(mk, [7, 1, 2, 3, 9, 4, 5]) == 3
    assert reasoning_tokens(mk, [7, 1, 2, 3, 4]) == 4                  # 回答が始まらない
    assert reasoning_tokens(mk, [1, 2, 3]) is None                    # 思考の区切りが無い


def test_rewrite_gguf_copies_untouched_and_adds_keys(tmp_path):
    """GGUF の書き直し: 差し替えないテンソルはバイトまで同じ(ファイルから少しずつ読んで写す)、足したメタデータが読める。"""
    import numpy as np
    from gguf import GGMLQuantizationType as Q
    from gguf import GGUFReader, GGUFWriter

    from gne.gguf_fast import FastGGUF, rewrite_gguf
    src, dst = tmp_path / "a.gguf", tmp_path / "b.gguf"
    w = GGUFWriter(str(src), "llama")
    rng = np.random.default_rng(0)
    data = {"t0": rng.standard_normal((4, 64)).astype(np.float32), "t1": rng.standard_normal((8, 32)).astype(np.float16),
            "t2": rng.standard_normal(16).astype(np.float32)}
    for k, v in data.items():
        w.add_tensor(k, v)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    new = np.ones((8, 32), np.float32)
    rewrite_gguf(src, dst, {"t1": (Q.F32, new.tobytes())}, extra_kv={"gne.patches": "[]", "gne.inkling_proj": '{"x": 1}'})
    a, b = FastGGUF(src, _part=True), FastGGUF(dst, _part=True)
    ta, tb = {t.name: t for t in a.tensors}, {t.name: t for t in b.tensors}
    for k in ("t0", "t2"):
        assert bytes(a._mm[ta[k].data_offset:ta[k].data_offset + ta[k].n_bytes]) == bytes(b._mm[tb[k].data_offset:tb[k].data_offset + tb[k].n_bytes])
    assert tb["t1"].tensor_type == Q.F32 and np.array_equal(np.frombuffer(b._mm[tb["t1"].data_offset:tb["t1"].data_offset + tb["t1"].n_bytes], np.float32).reshape(8, 32), new)
    r = GGUFReader(str(dst))
    assert "gne.patches" in r.fields and "gne.inkling_proj" in r.fields and "general.architecture" in r.fields
