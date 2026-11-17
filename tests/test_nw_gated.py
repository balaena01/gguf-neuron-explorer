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
