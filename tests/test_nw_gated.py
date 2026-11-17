"""計算で作るニューロン(mode nw)の、量子化の丸めへの対処(kscale)と、成分を読み取って取り除く形(remove)のテスト。

モデルなし: 書き方(!k4・!kauto・!rm・!rmbase)とレシピの保存・読み込み、確かめのエラー、倍率の決め方。
モデルあり(Qwen3-0.6B、GNE_TEST_MODEL。CPU で回すときは GNE_TEST_NGL=0): 倍率を付けても足す量が同じこと、既定(倍率なし)の
LoRA が変わらないこと、鍵が開く文での検算、成分を取り除く形が「その位置の成分に比例して」引くこと、基準値を超えた分だけを
引く形。
"""

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from gne import nwrite as NW
from gne.steer import GneError, Recipe, parse_unit

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
NGL = int(os.environ.get("GNE_TEST_NGL", "-1"))

CONTRAST = {
    "name": "gated_scene",
    "system_a": "You are a pirate. Always answer in rough pirate speech.",
    "user_prompts": ["Thank you for your help yesterday.", "I really like your drawing.", "Thanks for lending me your notebook.",
                     "Your cooking was delicious.", "Thank you for listening to me.", "You look great today."],
    "dont_touch": [{"name": "tasks", "prompts": ["What is 12 times 8?", "What is the capital of France?",
                                                 "How many days are in a week?", "What color is the sky?"]}],
}


def test_syntax_and_roundtrip(tmp_path):
    u = parse_unit("11:N20+N30:nw:a0.5@dir#sc&one!cap0.3-0.7!kauto!rmbase")
    assert (u.kscale, u.remove, u.cap, u.pair) == ("auto", "excess", "0.3-0.7", 30)
    assert NW.spec_of(u) == "11:N20+N30:nw:a0.5@dir#sc&one!cap0.3-0.7!kauto!rmbase"
    u2 = parse_unit("11:N20+N30:nw:a1@dir#sc&one!k4!rm")
    assert (u2.kscale, u2.remove) == ("4", "part") and NW.form_of(u2)["kscale"] == "4"
    old = parse_unit("8:N7:nw:a1@d#k")                                    # 今までのユニットは何も変わらない
    assert old.kscale is None and old.remove is None and NW.spec_of(old) == "8:N7:nw:a1@d#k"
    assert "kscale" not in NW.form_of(old) and "remove" not in NW.form_of(old)
    r = Recipe([u, u2, old], "m", name="rt")
    p = tmp_path / "r.yaml"
    r.save(p)
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert d["units"][0]["kscale"] == "auto" and d["units"][0]["remove"] == "excess" and "kscale" not in d["units"][2]
    back = Recipe.load(p)
    assert [NW.spec_of(x) for x in back.units] == [NW.spec_of(x) for x in r.units]


def test_validation():
    with pytest.raises(GneError):                                          # 成分を取り除く形は 2 個組だけ
        Recipe([parse_unit("8:N7:nw:a1@d#k!rm")], "m").validate()
    bad = parse_unit("8:N7+N9:nw:a1@d#k&one")
    bad.kscale = "-2"
    with pytest.raises(GneError):
        Recipe([bad], "m").validate()
    bad.kscale, bad.remove = None, "all"
    with pytest.raises(GneError):
        Recipe([bad], "m").validate()
    with pytest.raises(GneError):                                          # 書き方の順は !s → !cap → !k → !rm
        parse_unit("8:N7+N9:nw:a1@d#k&one!rm!k4")


def test_scale_rule():
    f = {"form": "cap", "s": 16.0, "a": 0.4, "b": 0.8, "kscale": "auto"}
    st = {"block_max_median": 0.1}
    peak = float(NW.silu(16.0 * 0.6))
    assert NW.kscale_of(f, st) == pytest.approx(peak / (NW.KSCALE_M * 0.1))
    assert NW.kscale_of({**f}, {"block_max_median": 100.0}) == 1.0           # 小さければ 1 より下げない
    assert NW.kscale_of({**f, "kscale": "3"}, None) == 3.0
    assert NW.kscale_of({"form": "one", "s": 4.0}, None) == 1.0
    with pytest.raises(GneError):
        NW.kscale_of(f, None)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip(f"{MODEL} がありません")
    from gne.api import Session
    root = tmp_path_factory.mktemp("gated")
    c = root / "c.yaml"
    c.write_text(yaml.safe_dump(CONTRAST, allow_unicode=True), encoding="utf-8")
    s = Session(MODEL, root=root, n_gpu_layers=NGL)
    s.direction(contrast=c, name="gd", positions="response", response_tokens=20)
    s.direction_key(c, role="scene", layers=[8, 12], name="g_scene", subset="all", max_tokens=20, pseudo=3)
    s.direction_key(c, role="one", layers=[8, 12], name="g_one", subset="all", max_tokens=20)
    usage = s.nw_neurons([8, 12], max_tokens=12, corpus_tokens=512, top=6)
    n8 = [x["neuron"] for x in usage["layers"][0]["rows"]]
    n12 = [x["neuron"] for x in usage["layers"][1]["rows"]]
    yield {"s": s, "c": c, "n8": n8, "n12": n12}
    s.close()


def _plan(s, units):
    r = s.recipe(units=units, name="g")
    return r, NW.nw_plan(s.weights, r, s._key_of, s.direction_set, s.nw_stats)


@pytest.mark.model
def test_scale_keeps_the_amount(built):
    s, n12 = built["s"], built["n12"]
    base_u = f"12:N{n12[0]}+N{n12[1]}:nw:a1@gd#g_scene&g_one"
    _, r0 = _plan(s, [base_u])
    _, r4 = _plan(s, [base_u + "!k4"])
    for (j, g, u, d, c), (j4, g4, u4, d4, c4) in zip(r0[0]["parts"], r4[0]["parts"]):
        assert j == j4 and np.allclose(g, g4) and np.allclose(u / 4.0, u4) and np.allclose(d * 4.0, d4) and c * 4.0 == pytest.approx(c4)
    # 既定(倍率なし)の LoRA は、倍率の指定が無い今までの式と同じ(同じ値・同じファイル名)
    t0 = NW.nw_tensors(s.weights, r0)
    plain = NW.nw_plan(s.weights, s.recipe(units=[base_u]), s._key_of, s.direction_set)       # 大きさを測らない呼び方
    assert NW.cache_hash(t0) == NW.cache_hash(NW.nw_tensors(s.weights, plain))
    _, ra = _plan(s, [base_u + "!kauto"])
    st = s.nw_stats(12)
    assert ra[0]["kscale_used"] == pytest.approx(max(1.0, float(NW.silu(16 * 0.6)) / (NW.KSCALE_M * st["block_max_median"])))
    assert st["block_max_median"] > 0 and st["m"] > 0


@pytest.mark.model
def test_verify_on_open_texts(built):
    s, n12 = built["s"], built["n12"]
    r = s.recipe(units=[f"12:N{n12[0]}+N{n12[1]}:nw:a1@gd#g_scene&g_one!kauto"], name="g")
    v = s.nw_verify(r, contrast=str(built["c"]), n_prompts=3, max_tokens=16)
    assert v["texts_from"] == "contrast" and v["n_texts"] == 3
    x = v["layers"][0]
    assert x["act_max"] > 0 and x["block_max_median"] > 0 and list(x["kscale"].values())[0] >= 1.0
    assert x["others_changed_max"] < 1e-4 and x["out_rel_err_median"] < 0.05, x
    v0 = s.nw_verify(r)                                                     # 今までどおり短い文でも確かめられる
    assert v0["texts_from"] == "texts"


@pytest.mark.model
def test_remove_component(built):
    """どこでも開く鍵(ゲートにもほぼ 1 を返す鍵)で当てると、MLP の出力の r̂ 成分が −α·m·(r̂·x / rms(x)) になる。"""
    s, n8 = built["s"], built["n8"]
    alpha = 0.5
    r, rows = _plan(s, [f"8:N{n8[0]}+N{n8[1]}:nw:a{alpha}@gd#g_one&g_one!cap0.1-0.2!rm"])
    row = rows[0]
    rh = row["d"] / np.linalg.norm(row["d"])
    gm = NW.norm_weight(s.weights, 8)
    st = row["layer_stats"]
    assert np.allclose(row["parts"][0][2] * row["kscale_used"], rh / gm)
    lora = s.nw_lora(r)
    e = s.engine
    ids = e.tokenize("The weather was calm and the market was busy that morning.", add_special=True, parse_special=False)
    names = {"ffn_inp-8", "ffn_norm-8", "ffn_out-8"}
    b = NW._run_capture(s, ids, names)
    g = NW._run_capture(s, ids, names, [(str(lora), 1.0)])
    x = b["ffn_inp-8"][1:]
    want = -alpha * st["m"] * (x @ rh) / np.sqrt(np.mean(x * x, axis=1))       # 取り除く量(r̂ 成分)
    got = (g["ffn_out-8"][1:] - b["ffn_out-8"][1:]) @ rh
    assert np.corrcoef(want, got)[0, 1] > 0.98
    assert np.median(np.abs(got - want)) < 0.1 * np.median(np.abs(want))
    # 基準値を超えた分だけ: 入力の行から b·κ1 を引いた形
    _, rows_b = _plan(s, [f"8:N{n8[0]}+N{n8[1]}:nw:a{alpha}@gd#g_one&g_one!cap0.1-0.2!rmbase"])
    up_b = rows_b[0]["parts"][0][2] * rows_b[0]["kscale_used"]
    assert np.allclose(up_b, rh / gm - rows_b[0]["layer_stats"]["b_up"] * rows_b[0]["kappa_up"])


@pytest.mark.model
def test_heads_flow(built):
    """CHiPS のヘッド(層 8・12 の 2 個)の分を、場面を選んで取り除くユニットを作る。量の当たり方・CHiPS との一致・場面の差。"""
    from gne import headgate as HG
    s = built["s"]
    chips = s.recipe(units=["8:H3:s4@gd", "12:H5:s4@gd"], name="chips_t")
    s.recipe_save("chips_t", chips)
    lay = HG.chips_layers(s, chips)
    assert sorted(lay) == [8, 12] and all(abs(np.linalg.norm(v["rhat"]) - 1) < 1e-5 for v in lay.values())
    d = HG.build(s, "chips_t", str(built["c"]), name="hg", scene_key="g_scene", one_key="g_one", subset="all", max_tokens=24,
                 neurons={8: built["n8"][2:4], 12: built["n12"][2:4]}, save="hg_r")
    assert [r["layer"] for r in d["layers"]] == [8, 12]
    assert all(r["r2_cv"] > 0.5 for r in d["layers"]), d["layers"]
    assert d["units"][0].startswith(f"8:N{built['n8'][2]}+N{built['n8'][3]}:nw:a1@hg_dir#g_scene&g_one!kauto!takehg_amount")
    m = d["match"]
    # 小さいモデル・6 問の対照集合なので緩い目安(Qwen3.5-4B での値は docs/verification.md)
    assert m["open"]["cos_median"] > 0.4 and 0.3 < m["open"]["ratio_median"] < 2.0, m
    assert m["scene"]["n_positions"] > 0 and m["other"]["n_positions"] > 0
    r = s.recipe(path=d["recipe_path"])
    v = s.nw_verify(r, contrast=str(built["c"]), n_prompts=2, max_tokens=12)
    assert all(x["others_changed_max"] < 1e-4 for x in v["layers"])
    assert "\n".join(HG.summary_lines(d)).count("R²") == 2


def test_target_spans():
    from gne.heads_tf import target_spans
    t = "Sure. Ahoy matey, here it is!\nThanks again."
    assert target_spans(t, r"[Mm]atey", "match") == [(t.index("matey"), t.index("matey") + 5)]
    (a, b), = target_spans(t, r"matey", "sentence")
    assert t[a:b].strip() == "Ahoy matey, here it is!"
    assert target_spans(t, r"nothing", "sentence") == []


@pytest.mark.model
def test_heads_tf_screen(built):
    from gne import heads_tf as HT
    s = built["s"]
    d = HT.screen(s, str(built["c"]), direction="gd", target=r"(?i)\b(you|your|ye|matey)\b", layers="6,8,12",
                  layers_heads="12", s=4, max_tokens=160, others=16, name="tf_t")
    assert [x["layer"] for x in d["layers"]] == [6, 8, 12] and d["heads_layers"] == [12]
    assert len(d["heads"]) == s.weights.arch.layers[12].n_head and all(x["b"] >= 0 for x in d["heads"])
    assert d["n_finished"] + d["n_unfinished"] <= d["n_prompts"] and d["n_with_target"] > 0
    assert Path(d["path"]).exists() and "ヘッドごとに測った層" in "\n".join(HT.summary_lines(d))
    with pytest.raises(GneError):
        HT.screen(s, str(built["c"]), direction="gd", target=r"zzzqqq", layers="8", max_tokens=16)


@pytest.mark.model
def test_lora_when(built):
    """LoRA を当てる位置を限る(生成と teacher forcing)。prompt は最初のトークンが全位置と同じ、reply は無介入と同じ。"""
    from gne.evalx import _generate
    from gne.heads_tf import _logprobs
    s, n12 = built["s"], built["n12"]
    r = s.recipe(units=[f"12:N{n12[4]}+N{n12[5]}:nw:a8@gd#g_one&g_one!cap0.1-0.2"], name="lw")
    lora = [(str(s.nw_lora(r)), 1.0)]
    items = [([{"role": "user", "content": q}], 12) for q in ("Tell me about the sea.", "What do you think of my drawing?",
                                                             "How was your day?")]
    none = _generate(s, items, None, None, "all", lambda k: None)
    allp = _generate(s, items, None, lora, "all", lambda k: None)
    again = _generate(s, items, None, lora, "all", lambda k: None, lora_when="all")
    assert [x["token_ids"] for x in allp] == [x["token_ids"] for x in again]          # 既定は今までと同じ
    assert any(a["token_ids"] != b["token_ids"] for a, b in zip(allp, none))           # この LoRA は返事を変える
    e = s.engine
    ctx = e.steer_ctx()                                # 最初のトークンの 1 位と 2 位の差(差が小さい質問は、バッチの組み方の違いで入れ替わりうる)
    seqs = [e.tokenize_prompt(e.render_chat(conv)) for conv, _ in items]
    margin = {}
    for st, lo in (("lora", lora), ("none", None)):
        e.set_lora(ctx, lo)
        e.clear_memory(ctx)
        lg = e.prefill(ctx, seqs).logits
        e.set_lora(ctx, None)
        top2 = np.sort(lg, axis=1)[:, -2:]
        margin[st] = top2[:, 1] - top2[:, 0]
    for batched in (True, False):
        when = "all" if batched else "gen"                                             # gen は 1 本ずつの道(cvec なしでは all と同じ)
        pr = _generate(s, items, None, lora, when, lambda k: None, lora_when="prompt")
        rp = _generate(s, items, None, lora, when, lambda k: None, lora_when="reply")
        for i in range(len(items)):
            if margin["lora"][i] > 1.0:
                assert pr[i]["token_ids"][0] == allp[i]["token_ids"][0]
            if margin["none"][i] > 1.0:
                assert rp[i]["token_ids"][0] == none[i]["token_ids"][0]
    rows = []
    for (conv, _), o in zip(items, none):
        rid = [int(t) for t in o["token_ids"] if not e.is_eog(t)][:8]
        rows.append({"prompt": e.tokenize_prompt(e.render_chat(conv)), "reply": rid, "target": [0, 1], "others": list(range(2, len(rid) - 1))})
    b0 = _logprobs(s, rows)
    la = _logprobs(s, rows, lora)
    lp = _logprobs(s, rows, lora, lora_when="prompt")
    lr = _logprobs(s, rows, lora, lora_when="reply")
    for x0, xa, xp, xr in zip(b0, la, lp, lr):
        assert xp[0] == pytest.approx(xa[0], abs=1e-4) and xr[0] == pytest.approx(x0[0], abs=1e-4)
        assert not np.allclose(xr[1:], x0[1:], atol=1e-3)                              # 返事の位置では効いている
    with pytest.raises(GneError):
        _generate(s, items, None, lora, "all", lambda k: None, lora_when="middle")
