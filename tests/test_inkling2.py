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
    for ok in ("info", "chat", "behave", "probe", None):
        check_command(_plan(), ok)
    for bad in ("heads pipeline", "direction", "steer", "nw pipeline", "stats resid"):
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
    """焼き込みの型: q8_0 と f16 は編集が残り、keep は Q2_0 を断る(ファイルは書かない)。"""
    from types import SimpleNamespace

    import numpy as np

    from gne.inkling_chips import bake, layer_refs
    from gne.steer import GneError
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
    with pytest.raises(GneError, match="Q2_0"):
        bake(rd, units, refs, None, requant="keep")
