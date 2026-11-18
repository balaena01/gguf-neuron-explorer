"""会話の形を混ぜた対照集合(gne/prior.py)、位置 0 以外のキー(after:<正規表現>・target_reply)、決め手のセルの重みと縮め方
(keyed_ridge.fit の conv_cells・crit・fp_max)、出力層の行の方向と層の走査(gne/tokendir.py)。

モデルなし: 質問の確かめ・会話の組み立て・形のラベル、行の間引き(いつも残す位置)、セルと分割、after の区分、縮め方、方向の式。
モデルあり(Qwen3-0.6B、GNE_TEST_MODEL。CPU なら GNE_TEST_NGL=0): 前のやり取りの応答の穴埋めと覚え書き、形を混ぜた最小二乗の
キー(セルに形を入れる・入れない)、after のキーと target_reply、gne nw check の形ごとの数、トークンの方向と走査、CLI の入口。
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from gne import keyed_ridge as KR
from gne.prior import FORMS, conv_form, conv_form_of, has_null, spec_has_null
from gne.reverse import _conv, check_question, question_text, turns
from gne.steer import GneError

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
NGL = int(os.environ.get("GNE_TEST_NGL", "-1"))

MT_Q = [{"role": "user", "content": "今日は晴れてるね"}, {"role": "assistant", "content": None},
        {"role": "user", "content": "それでね、えーと"}]
SYS_Q = [{"role": "system", "content": "あなたは料理の相談役です。"}, {"role": "user", "content": "あのー、ちょっと"}]


# --- モデルなし ----------------------------------------------------------------------------------------------

def test_questions_with_system_and_null_prior():
    check_question(MT_Q)
    check_question(SYS_Q)
    check_question([SYS_Q[0], *MT_Q])
    with pytest.raises(GneError):                                    # 最後の assistant は null にできない
        check_question([{"role": "user", "content": "a"}, {"role": "assistant", "content": None}])
    with pytest.raises(GneError):                                    # system は先頭だけ
        check_question([{"role": "user", "content": "a"}, {"role": "system", "content": "s"}])
    with pytest.raises(GneError):                                    # 埋めていない応答は組み立てられない
        _conv(None, MT_Q)
    assert has_null(MT_Q) and not has_null(SYS_Q) and not has_null("a")
    assert question_text(MT_Q).startswith("user: 今日は晴れてるね / assistant: … /")
    # その問の system は群の system より優先し、turns には入らない
    assert _conv("群", SYS_Q) == [{"role": "system", "content": "あなたは料理の相談役です。"}, {"role": "user", "content": "あのー、ちょっと"}]
    assert turns(SYS_Q) == [{"role": "user", "content": "あのー、ちょっと"}]
    assert _conv("群", "a")[0] == {"role": "system", "content": "群"}


def test_form_labels():
    filled = [dict(m, content=m["content"] or "x") for m in MT_Q]
    assert conv_form(None, "a") == "single" and conv_form("", "a") == "system"
    assert conv_form(None, filled) == "multi_turn" and conv_form(None, SYS_Q) == "system"
    assert conv_form(None, [SYS_Q[0], *filled]) == "system_multi_turn" and conv_form("s", filled) == "system_multi_turn"
    assert conv_form_of(_conv(None, filled)) == "multi_turn" and conv_form_of(_conv(None, SYS_Q)) == "system"
    assert set(FORMS) == {"single", "multi_turn", "system", "system_multi_turn"}
    assert spec_has_null({"user_prompts": ["a", MT_Q]}) and spec_has_null({"user_prompts": ["a"], "dont_touch": [{"prompts": [MT_Q]}]})
    assert not spec_has_null({"user_prompts": ["a", SYS_Q]})


def test_keep_positions_must():
    rng = lambda: np.random.default_rng(3)
    ks0, w0 = KR.keep_positions(40, rng())
    ks1, w1 = KR.keep_positions(40, rng(), ())
    assert ks0.tolist() == ks1.tolist() and w0.tolist() == w1.tolist()   # いつも残す位置が無ければ今までと同じ
    ks, w = KR.keep_positions(40, rng(), [5, 20, 33])
    assert {5, 20, 33} <= set(ks.tolist()) and list(ks) == sorted(ks)
    m = {int(k): float(x) for k, x in zip(ks, w, strict=True)}
    assert m[20] == 1.0 and m[33] == 1.0
    rest = [k for k in range(16, 40) if k not in (20, 33)]
    assert [x for k, x in m.items() if k >= 16 and k not in (20, 33)] == [len(rest) / KR.SAMPLE] * KR.SAMPLE


def _store(seed=0, n=24, D=12, forms=True, after=None, target_reply=None, texts=None):
    """狙う場面(位置 0 の状態に e0)・群 x・群 y。会話の形は半分ずつ single / multi_turn。after の位置(トークン 7 の直後)に e1。"""
    rng = np.random.default_rng(seed)
    st = KR.RowStore([3], 0, after=after, pieces_of=(lambda t: b"." if t == 7 else b"a"), target_reply=target_reply)
    for gi, lab in enumerate(["AB", "dt:x", "dt:y"]):
        st.begin(lab, n, ["single"] * (n // 2) + ["multi_turn"] * (n - n // 2) if forms else None)
        for i in range(n):
            R = 20
            reply = [1] * R
            reply[4] = 7
            reply[12] = 7

            def rows_of(L, sel, gi=gi, P=10):
                r = rng.normal(size=(len(sel), D)) * 0.3
                k = sel - (P - 1)
                if gi == 0:
                    r[k == 0, 0] += 3
                    r[np.isin(k, [5, 13]), 1] += 3
                if gi == 1:
                    r[np.isin(k, [5, 13]), 1] += 1.5          # 決め手のセル: ほかの群の同じ種類の位置
                return r
            st.add(i, 10, R, rows_of, reply, (texts[gi][i] if texts else ""))
    return st


def _fit(st, ranges, **kw):
    tab = st.table()
    xs = {L: np.concatenate(st._rows[L]) for L in st.layers}
    return KR.fit(lambda L: xs[L], st.layers, tab, ranges, group_names=st.group_names(), **kw)


def test_conv_cells_and_folds():
    st = _store()
    a = _fit(st, [(0, 1)])
    b = _fit(st, [(0, 1)], conv_cells=True)
    assert a["info"]["forms"] == {"single": 36, "multi_turn": 36} and not a["info"]["conv_cells"]
    assert b["info"]["cells"][0] == 2 * a["info"]["cells"][0]          # 形ごとにセルが分かれる
    assert "scene|single" in b["info"]["rows_by_cell"] and "x|multi_turn" in b["info"]["rows_by_cell"]
    tab = b["info"]["cv_cells"][0][3]
    assert any(c["want"] and c["group"] == "scene|single" for c in tab)
    # 形が 1 つなら、分割と答えは今までと同じ
    one = _store(forms=False)
    c1 = _fit(one, [(0, 1)])
    tab1 = one.table()
    tab1.pop("q_form")
    xs = {3: np.concatenate(one._rows[3])}
    c2 = KR.fit(lambda L: xs[L], [3], tab1, [(0, 1)], group_names=one.group_names())
    assert np.allclose(c1["kappa"][0][3], c2["kappa"][0][3]) and c1["info"]["c"] == c2["info"]["c"]


def test_after_positions_target_reply_crit_shrink():
    texts = [["◇" if i % 3 else "x" for i in range(24)], [""] * 24, [""] * 24]
    st = _store(after=r"\.", target_reply="^◇", texts=texts)
    assert st.n_target_reply == 16 and len(st.q_other) == 8 and st.group_names() == ["scene", "x", "y", "scene_other"]
    tab = st.table()
    k = tab["pos"][tab["match"]]
    assert set(k.tolist()) == {5, 13}                                  # トークン 4・12 の直後
    r = _fit(st, ["after"])
    assert r["info"]["bins"] == ["after", "0", "1-3", "4-"] and r["info"]["groups"][-1] == "scene_other"
    dec = [c for c in r["info"]["cv_cells"][0][3] if c["decisive"]]
    assert dec and all(c["group"] in ("x", "y") and c["bin"] == "after" for c in dec)
    rc = _fit(st, ["after"], crit=10.0)
    assert rc["info"]["crit"] == 10.0 and rc["info"]["cells"][0] > r["info"]["cells"][0]
    rs = _fit(st, ["after"], crit=10.0, fp_max=0.0)
    sh = rs["info"]["shrink"][3]
    assert sh["fp_after"] <= sh["fp_before"] and 0 <= sh["scale"] <= 1
    if sh["scale"] < 1:
        assert np.allclose(rs["kappa"][0][3], sh["scale"] * rc["kappa"][0][3])
    with pytest.raises(GneError):                                      # 縮め方は c を分割で決めるときだけ
        _fit(st, ["after"], fp_max=0.1, c=1.0)
    with pytest.raises(GneError):                                      # 重みと縮め方は 1 本ずつ
        _fit(_store(), [(0, 1), (1, 4)], crit=2.0)


def test_token_direction_formula():
    from gne.tokendir import direction_of

    class W:
        class meta:
            tied_output = True
            n_embd = 4
        rows = np.array([[1.0, 0, 0, 0], [0, 2.0, 0, 0], [1.0, 1.0, 0, 0]])

        def output_norm_weight(self):
            return np.array([2.0, 1.0, 1.0, 1.0])

        def embed_rows(self, ids):
            return self.rows[list(ids)]
    d = direction_of(W(), [0])
    assert np.allclose(d["u_hat"], [1, 0, 0, 0]) and np.isclose(d["gdot"], 2.0)       # (γ⊙r)·r̂
    d2 = direction_of(W(), [0], [1])
    u = np.array([1.0, -2.0, 0, 0])
    assert np.allclose(d2["u_hat"], u / np.linalg.norm(u)) and np.isclose(d2["gdot"], (np.array([2.0, 1, 1, 1]) * u) @ u / np.linalg.norm(u))
    with pytest.raises(GneError):
        direction_of(W(), [0], [0])


# --- モデルあり -----------------------------------------------------------------------------------------------

CONTRAST = {
    "name": "cf_route", "system_b": None,
    "user_prompts": ["えっと、明日の会議なんだけど", "そういえばさ、あの件", MT_Q, SYS_Q, "うーん、どうしようかな",
                     [{"role": "user", "content": "最近忙しい？"}, {"role": "assistant", "content": None},
                      {"role": "user", "content": "あ、そうそう、それで"}],
                     [{"role": "system", "content": "あなたは IT の窓口です。"}, {"role": "user", "content": "えーと、その"}],
                     "あ、ちょっと待って"],
    "dont_touch": [{"name": "ask", "prompts": [
        "水の沸点は何度？", "Python でリストを逆順にする方法は？", [SYS_Q[0], {"role": "user", "content": "カレーの隠し味は？"}],
        [{"role": "user", "content": "こんにちは"}, {"role": "assistant", "content": None}, {"role": "user", "content": "おすすめの本を教えて"}],
        "1 キロメートルは何メートル？", "東京の人口は？", "月までの距離は？",
        [{"role": "system", "content": "あなたは IT の窓口です。"}, {"role": "user", "content": "パスワードを忘れました。"}]]}],
}


@pytest.fixture(scope="module")
def sess(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip(f"{MODEL} がありません")
    from gne.api import Session
    root = tmp_path_factory.mktemp("cf")
    c = root / "c.yaml"
    c.write_text(yaml.safe_dump(CONTRAST, allow_unicode=True), encoding="utf-8")
    s = Session(MODEL, root=root, n_gpu_layers=NGL)
    yield {"s": s, "root": root, "contrast": c}
    s.close()


def test_prior_fill_and_cache(sess):
    s, c = sess["s"], sess["contrast"]
    d = s.prior_fill(c, max_tokens=16)
    assert d["filled"] == 3 and d["generated"] == 3 and d["cached"] == 0
    assert d["forms"]["user_prompts"] == {"single": 4, "multi_turn": 2, "system": 2}
    out = yaml.safe_load(Path(d["path"]).read_text(encoding="utf-8"))
    assert not spec_has_null(out) and out["user_prompts"][2][1]["content"]
    assert Path(d["path"]).parent == sess["root"] / "contrasts" / "prior"
    d2 = s.prior_fill(c, max_tokens=16)
    assert d2["generated"] == 0 and d2["cached"] == 3 and d2["path"] == d["path"]
    cache = json.loads((sess["root"] / "prior" / s.meta.model_id / "replies.json").read_text(encoding="utf-8"))
    assert len(cache) == 3
    # 上限を変えると作り直す(別の覚え書き)
    d3 = s.prior_fill(c, max_tokens=8)
    assert d3["generated"] == 3 and d3["path"] != d["path"]


def test_mixed_form_keys(sess):
    s, c = sess["s"], sess["contrast"]
    a = s.direction_key(c, name="cf_a", layers=[8], positions="pos:0", target="mlpin", key_fit="ridge", subset="all",
                        max_tokens=8, pseudo=0, prior_max_tokens=16)
    ri = a["ridge"]
    assert ri["forms"] == {"single": 9, "multi_turn": 3, "system": 4} and not ri["conv_cells"]
    b = s.direction_key(c, name="cf_b", layers=[8], positions="pos:0", target="mlpin", key_fit="ridge", subset="all",
                        max_tokens=8, pseudo=0, key_conv_cells=True, prior_max_tokens=16)
    assert b["ridge"]["conv_cells"] and any("|system" in g for g in b["ridge"]["rows_by_cell"])
    assert b["ridge"]["cv_cells"]
    with pytest.raises(GneError):                                      # 形のセルは最小二乗のキーだけ
        s.direction_key(c, name="cf_x", layers=[8], positions="pos:0", target="mlpin", key_conv_cells=True)


def test_after_key_and_target_reply(sess):
    s, c = sess["s"], sess["contrast"]
    r = s.direction_key(c, name="cf_end", layers=[8], positions="after:[。！？\n]", target="mlpin", key_fit="ridge", subset="all",
                        max_tokens=24, pseudo=0, key_crit=5.0, prior_max_tokens=16)
    assert r["positions"] == "after:[。！？\n]" and r["ridge"]["bins"] == ["after", "0", "1-3", "4-"]
    assert r["ridge"]["crit"] == 5.0
    with pytest.raises(GneError):                                      # after のキーは最小二乗で 1 本ずつ
        s.direction_key(c, name="cf_end2", layers=[8], positions="after:。", target="mlpin")
    with pytest.raises(GneError, match="target_reply"):
        s.direction_key(c, name="cf_tr", layers=[8], positions="pos:0", target="mlpin", key_fit="ridge", subset="all",
                        max_tokens=8, pseudo=0, target_reply="^絶対に出ない文字列ZZZ", prior_max_tokens=16)


def test_check_by_form(sess):
    s, c = sess["s"], sess["contrast"]
    s.direction_key(c, name="cf_p0", layers=[8], positions="pos:0", target="mlpin", key_fit="ridge", subset="all", max_tokens=8,
                    pseudo=0, prior_max_tokens=16)
    s.direction_key(c, name="cf_one", layers=[8], role="one", subset="all", max_tokens=8, prior_max_tokens=16)
    s.direction_token(["<|im_end|>"], layers=[8], delta=4, name="cf_eos", rbar=1.0)
    u = s.nw_neurons([8], top=4)["layers"][0]["rows"]
    unit = f"8:N{u[0]['neuron']}+N{u[1]['neuron']}:nw:a1@cf_eos#cf_p0&cf_one"
    d = s.nw_check(s.recipe(units=[unit]), c, subset="all", max_tokens=8, pos_bins="0,1-3,4-", prior_max_tokens=16)
    bf = d["groups"]["scene"]["by_form"]
    assert set(bf) == {"single", "multi_turn", "system"} and bf["single"]["n"] == 4
    assert set(d["groups"]["scene"]["units"][0]["positions"]) == {"0", "1-3", "4-"}
    assert "finished" in d["groups"]["dont_touch:ask"]


def test_token_direction_and_scan(sess):
    s, c = sess["s"], sess["contrast"]
    d = s.direction_token(["<|im_end|>"], layers=[10, 12], delta=8, name="cf_tok", contrast=c)
    z = np.load(d["path"])
    v = z["layer_10"]
    assert np.isclose(np.linalg.norm(v), d["beta"], rtol=1e-5) and np.allclose(z["layer_12"], v)
    assert np.isclose(d["beta"], 8 * d["rbar"] / d["gdot"]) and str(z["method"]) == "token_row"
    dn = s.direction_token(["<|im_end|>"], layers=[10], delta=8, name="cf_tok_dn", rbar=d["rbar"], down=True)
    assert np.allclose(np.load(dn["path"])["layer_10"], -v)
    sc = s.direction_token_scan(["<|im_end|>"], c, layers=[20], deltas=[4, 64], need=0.25, rbar=d["rbar"])
    row = sc["rows"][0]
    assert set(row["tried"]) == {"4", "64"} and sc["open_n"] == 8 and sc["close_n"] == 8
    if row["need_delta"] is not None:
        assert set(row["leak"]) == {"0.2", "0.5"} and row["leak"]["0.2"]["n"] == 8
    with pytest.raises(GneError):
        s.direction_token(["こんにちは、世界"], layers=[10], delta=8, name="cf_bad", rbar=1.0)   # 1 トークンにならない


def test_cli_entries(sess):
    from typer.testing import CliRunner

    from gne.cli import app
    common = ["-m", str(MODEL), "--root", str(sess["root"]), "--n-gpu-layers", str(NGL)]
    run = CliRunner().invoke
    r = run(app, ["nw", "prior", "--contrast", str(sess["contrast"]), "--max-tokens", "16", *common, "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output.strip().splitlines()[-1])["filled"] == 3
    r = run(app, ["direction", "token", "--token", "<|im_end|>", "--layers", "10", "--delta", "8", "--name", "cf_cli", "--rbar", "1.0",
                  *common])
    assert r.exit_code == 0 and "β 8 ÷" not in r.output and "β" in r.output, r.output
    r = run(app, ["nw", "key", "--contrast", str(sess["contrast"]), "--layers", "8", "--role", "one", "--key-conv-cells", *common])
    assert r.exit_code != 0                                             # 形のセルは位置のキーだけ


def test_first_token(sess):
    s = sess["s"]
    ids = s.first_ids("<|im_end|>")
    assert len(ids) == 1
    d = s.steer(s.recipe(units=["5:7:+0.0"]), prompt="こんにちは", max_tokens=6, first_token="はい")
    want = s.first_ids("はい")
    for side in ("baseline", "steered"):
        assert d[side]["token_ids"][:len(want)] == want and d[side]["text"].startswith("はい")
        assert d[side]["when"]["first_ids"] == want
    assert s.first_ids(None) is None and s.first_ids("") is None


def test_pipeline_token_direction(sess):
    from gne import fish
    s, c = sess["s"], sess["contrast"]
    p = fish.pipeline(s, str(c), name="cf_rt", layers=[10], segments="0", alpha=1, token="<|im_end|>", token_delta=8,
                      key_fit="ridge", pseudo=0, max_tokens=8, response_tokens=8, rekey=False, compare=False, kscale="auto")
    assert all("@cf_rt_tok#" in u and u.endswith("!kauto") for u in p["units"])
    assert p["direction"]["unsegmented"] and p["direction"]["token"]["tokens"]
    with pytest.raises(GneError):
        fish.pipeline(s, str(c), name="cf_rt2", layers=[10], segments="0", token="<|im_end|>")     # Δ が要る


def test_jobs_and_mcp_entries(sess):
    import inspect

    from gne import jobs, mcp_server
    k = jobs.standard_kinds()
    assert {"direction_token", "direction_token_scan", "nw_prior"} <= set(k)
    s, c = sess["s"], sess["contrast"]
    d = k["direction_token"](s, tokens=["<|im_end|>"], layers=[10], delta=8, name="cf_job", rbar=1.0)
    assert d["name"] == "cf_job" and np.isclose(d["beta"], 8 / d["gdot"])
    p = k["nw_prior"](s, contrast=str(c), max_tokens=16)
    assert p["filled"] == 3
    src = inspect.getsource(mcp_server)
    for name in ("async def direction_token(", "async def direction_token_scan(", "async def nw_prior(", "first_token: str | None"):
        assert name in src


def test_after_positions_select():
    from gne.keyed_build import parse_positions, select_positions, start_positions

    class E:
        def is_eog(self, t):
            return t == 0

        def token_bytes(self, t):
            return {1: "あ".encode(), 2: "。".encode(), 3: "い".encode()}[t]
    reply = [1, 2, 3, 3, 2, 1, 0]                      # あ。いい。あ <eog>
    p = parse_positions("after:。")
    assert p.kind == "after" and p.needs_reply and parse_positions("not-after:。").kind == "not-after"
    # 位置 k = 返事の k 番目のトークンを決める位置(列 P − 1 + k)。「。」はトークン 1・4 → 位置 2・5
    assert select_positions(E(), reply, "after:。", n_prompt=10).tolist() == [11, 14]
    assert select_positions(E(), reply, "not-after:。", n_prompt=10).tolist() == [9, 10, 12, 13, 15]
    assert start_positions(E(), reply, "after:。", n_prompt=10, k=3).tolist() == [11]
    with pytest.raises(GneError):
        parse_positions("after:")
