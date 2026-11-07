"""CHANGE-07 WP6 のテスト: keyed のレシピの評価(gne/keyed_eval.py・gne/evalx.py・gne/controls.py)。

モデルなし: 大きさ合わせの探索(match_strength)を単調な合成の関数で、擬似鍵 NAME~j の対照レシピ(keyed だけのレシピでも
対照ができる)、control_set の keyed の分岐、direction_of の keyed、選択性の要約の向き、表示の行。
モデルあり(Qwen3-0.6B、GNE_TEST_MODEL。CPU で回すときは GNE_TEST_NGL=0): ずんだもんの対照集合に与えた返事(です・ます調)を
足した 1 つのファイルで方向と鍵を作り、短い eval(倍率 0 と 1、小さいタスク集、対照 2 本)で結果の keyed の形、keyed の
側 B の ‖Δ‖ が同じ α の add より小さいこと、擬似鍵の対照と、鍵の対照集合が分からないときの扱いを確かめる。
"""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from gne.steer import GneError, Recipe, Unit, parse_unit

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
NGL = int(os.environ.get("GNE_TEST_NGL", "-1"))
ZUNDAMON = ROOT / "configs" / "contrast" / "zundamon_endings.yaml"


# --- 大きさ合わせ(モデルなし)---------------------------------------------------------------------------

def test_match_strength_monotone():
    """単調な関数で target にそろう β を見つけ、測った回数が少ないこと(線形なら外挿 1 回)。"""
    from gne.keyed_eval import match_strength
    calls = []

    def lin(b):
        calls.append(b)
        return 3.0 * b
    m = match_strength(lin, 6.0, start=1.0, hi=8.0, rtol=0.01)
    assert m["reached"] and m["bracketed"] and abs(m["beta"] - 2.0) < 1e-9 and m["evals"] == 2 == len(calls)
    assert m["trace"] == [[1.0, 3.0], [2.0, 6.0]]
    # 二乗: 外挿で越え、挟んだ区間を補間で詰める
    m = match_strength(lambda b: b * b, 9.0, start=1.0, hi=8.0, rtol=0.05, steps=8)
    assert m["reached"] and abs(m["value"] - 9.0) <= 0.45 and 2.8 < m["beta"] < 3.2 and m["evals"] <= 6
    # 最初の点で越える(区間 [0, start] を補間)
    m = match_strength(lambda b: 10.0 * b, 3.0, start=1.0, hi=8.0, rtol=0.01)
    assert m["reached"] and abs(m["beta"] - 0.3) < 1e-6 and m["evals"] == 2
    # 打ち消す向き(target が f0 より小さい)
    m = match_strength(lambda b: -1.5 * b, -3.0, start=1.0, hi=8.0, rtol=0.01)
    assert m["reached"] and abs(m["beta"] - 2.0) < 1e-6
    # target = f0 は測らない
    m = match_strength(lambda b: 1 / 0, 0.0)
    assert m["reached"] and m["beta"] == 0.0 and m["evals"] == 0


def test_match_strength_unreachable_and_wrong_way():
    """上限まで強めても届かない(飽和する)なら reached・bracketed とも False で、最も近い点(上限)を返す。
    逆に動く関数でも上限で止まり、回数は steps を超えない。"""
    from gne.keyed_eval import match_strength
    m = match_strength(lambda b: 1.0 - np.exp(-b), 2.0, start=1.0, hi=8.0, steps=10)
    assert not m["reached"] and not m["bracketed"] and m["beta"] == 8.0
    assert [round(b, 2) for b, _ in m["trace"]] == [1.0, 3.16, 6.61, 8.0]         # 原点からの外挿で広げ、上限で止まる
    m = match_strength(lambda b: -b, 2.0, start=1.0, hi=30.0, steps=4)
    assert not m["reached"] and m["evals"] == 4 and m["beta"] == 0.0 and m["value"] == 0.0         # どの点も 0 より遠い
    # 単調でなくても、測った点のうち target に最も近いものを返す
    m = match_strength(lambda b: 5.0 * np.sin(b), 4.0, start=1.0, hi=8.0, steps=6, rtol=0.05)
    assert abs(m["value"] - 4.0) <= abs(5.0 * np.sin(1.0) - 4.0)


# --- 擬似鍵の対照(モデルなし)---------------------------------------------------------------------------

def _fake_session(tmp_path, n_pseudo=3, name="zk"):
    """keyed_control_cap・load_key が使う store と meta、control_recipes が使う vectors だけを持つ。"""
    from gne.keyed import KeySet, key_dir
    from gne.store import Store
    s = SimpleNamespace(store=Store(tmp_path), meta=SimpleNamespace(model_id="m"),
                        vectors=lambda r: ({}, [{"rel_push": 0.25} for _ in r.units]))
    rng = np.random.default_rng(0)
    for nm, n in ((name, n_pseudo), ("bare", 0)):
        Ls = [5, 9]
        ks = KeySet(target="ffn", kappa={L: rng.standard_normal(16).astype(np.float32) for L in Ls},
                    mu={L: rng.standard_normal(16) for L in Ls}, kn={L: 0.5 for L in Ls}, gain={L: 2.0 for L in Ls},
                    kappa_ctrl={L: rng.standard_normal((n, 16)).astype(np.float32) for L in Ls} if n else {},
                    model_id="m", name=nm)
        ks.save(key_dir(s) / f"{nm}.npz")
    return s


def test_keyed_control_recipes_pseudo_keys(tmp_path):
    """鍵を擬似鍵 NAME~j に替えた対照: α・方向・層・書き込み先とほかのユニットはそのまま、鍵を省いたユニットは
    default_key の擬似鍵を書き込む、本数は鍵の擬似鍵の本数まで、擬似鍵の無い鍵・keyed の無いレシピは空。"""
    from gne.controls import keyed_control_cap, keyed_control_recipes, pseudo_keyed
    s = _fake_session(tmp_path)
    r = Recipe([parse_unit("5:ffn:keyed:a1.25@zun_r#zk"), parse_unit("9:ffn:keyed:a-0.5@zun_r"),
                parse_unit("7:dir:zun_r:+0.3"), parse_unit("3:12:x2")], "m", name="zr", default_key="zk")
    assert keyed_control_cap(s, r) == 3
    out = keyed_control_recipes(s, r, 5)
    assert [c.name for c in out] == ["zr-pseudo0", "zr-pseudo1", "zr-pseudo2"]
    for j, c in enumerate(out):
        ku = c.keyed_units
        assert [u.key for u in ku] == [f"zk~{j}", f"zk~{j}"]
        assert [(u.layer, u.k, u.direction, u.kind) for u in ku] == [(5, 1.25, "zun_r", "ffn"), (9, -0.5, "zun_r", "ffn")]
        assert c.units[2:] == r.units[2:] and c.default_key == "zk"
    assert len(keyed_control_recipes(s, r, 2)) == 2
    # すでに擬似鍵を使うレシピでも、本物の名前から番号を付け直す
    assert [u.key for u in pseudo_keyed(out[1], 0)][:2] == ["zk~0", "zk~0"]
    assert keyed_control_recipes(s, Recipe([parse_unit("5:ffn:keyed:a1@zun_r#bare")], "m"), 4) == []
    assert keyed_control_recipes(s, Recipe([parse_unit("7:dir:zun_r:+0.3")], "m"), 4) == []
    assert keyed_control_recipes(s, r, 0) == []


def test_control_set_keyed_branch(tmp_path):
    """keyed だけのレシピでも対照ができる(前は 0 本)。add と混ぜたら add はランダムな向き、鍵は擬似鍵に(同じ j)。
    本数は鍵の擬似鍵の本数まで。"""
    from gne.evalx import control_set
    s = _fake_session(tmp_path)
    only = Recipe([parse_unit("5:ffn:keyed:a1@zun_r#zk")], "m", name="only")
    cs = control_set(s, only, 2)
    assert [c.name for c in cs] == ["only-random0", "only-random1"]
    assert [c.units[0].key for c in cs] == ["zk~0", "zk~1"] and all(c.units[0].k == 1.0 for c in cs)
    assert len(control_set(s, only, 10)) == 3
    mixed = Recipe([parse_unit("5:ffn:keyed:a1@zun_r#zk"), parse_unit("7:dir:zun_r:+0.3")], "m", name="mx")
    cm = control_set(s, mixed, 2)
    assert [c.units[0].key for c in cm] == ["zk~0", "zk~1"]
    assert all(c.units[1].kind == "random" and c.units[1].dose == 0.25 for c in cm)
    assert control_set(s, Recipe([parse_unit("5:ffn:keyed:a1@zun_r#bare")], "m"), 3) == []


def test_direction_of_keyed():
    from gne.evalx import direction_of
    from gne.steer import AblateSpec
    assert direction_of(Recipe([parse_unit("5:ffn:keyed:a1@zun_r#zk")], "m")) == "zun_r"
    assert direction_of(Recipe([parse_unit("5:ffn:keyed:a-1#zk")], "m", default_direction="tsun_r")) == "tsun_r"
    assert direction_of(Recipe([parse_unit("5:ffn:keyed:a1@zun_r#zk"), parse_unit("3:H2:s4@other")], "m")) == "other"
    assert direction_of(Recipe([parse_unit("5:ffn:keyed:a1@zun_r#zk"), parse_unit("7:dir:third:+0.3")], "m")) == "zun_r"
    assert direction_of(Recipe([parse_unit("3:12:x2")], "m", ablate=AblateSpec("abl"))) == "abl"


def test_summary_sign_and_text():
    """選択性は書く向き(α の符号)にそろえた 側 A − 側 B。b_over_a は 側 B の ‖Δ‖ ÷ |側 A の par|。表示の行に方式・除外・
    擬似鍵が出る。"""
    from gne.keyed_build import opposite_subset
    from gne.keyed_eval import KeyedMetrics, keyed_text
    km = object.__new__(KeyedMetrics)
    st = {"A": {"par": -4.0, "norm": 4.2}, "B": {"par": -0.5, "norm": 1.0}, "dt:en": {"par": 0.1, "norm": 0.4}}
    km.sign = -1.0
    s = km.summary(st)
    assert s["selectivity"] == pytest.approx(3.5) and s["b_over_a"] == pytest.approx(0.25)
    assert s["dt_norm"] == pytest.approx(0.4) and s["a_norm"] == 4.2
    km.sign = 1.0
    assert km.summary(st)["selectivity"] == pytest.approx(-3.5)
    assert km.summary({"A": {"par": 1.0, "norm": 1.0}})["selectivity"] is None
    assert [opposite_subset(x) for x in ("even", "odd", "all", None)] == ["odd", "even", "all", "all"]
    blk = {"key": "zk", "direction": "zun_r", "objective_layer": 14, "held_out": True, "subset": "odd",
           "positions": "before:です。", "positions_b": "not-before:です。",
           "counts": {"A": {"examples": 4, "positions": 8}, "B": {"examples": 4, "positions": 90},
                      "dt:english": {"examples": 2, "positions": 30}},
           "methods": [{"method": "keyed", "beta": 1.0, "a_par": 2.0, "b_par": 0.1, "b_norm": 0.5, "dt_norm": 0.3,
                        "selectivity": 1.9, "b_over_a": 0.25},
                       {"method": "add", "beta": 0.8, "matched": True, "a_par": 2.0, "b_par": 2.0, "b_norm": 2.1,
                        "dt_norm": 2.0, "selectivity": 0.0, "b_over_a": 1.05},
                       {"method": "gain", "excluded": "gain の対照は dense の ffn_down の keyed だけです"}],
           "compare": {"b_norm_ratio": {"add": 0.24}, "b_over_a_ratio": {"add": 0.24}},
           "pseudo": {"n": 4, "p50": 0.3, "p95": 0.9, "real": 1.9, "beats": True}, "notes": ["テスト"]}
    text = keyed_text(blk)
    joined = "\n".join(text)
    assert all(isinstance(x, str) for x in text)
    assert "keyed" in joined and "除外: gain の対照" in joined and "english: 2 例 30 位置" in joined
    assert "add 0.24" in joined and "本物 +1.9 は 超える" in joined and "注: テスト" in joined


# --- モデルを使う ------------------------------------------------------------------------------------

PROMPTS = ["週末は何をして過ごすのがおすすめ？", "朝ごはんに何を食べたらいい？", "おすすめの趣味を教えて。", "雨の日は何をする？",
           "疲れたときはどうすればいい？", "好きな季節はいつ？", "カレーの作り方を簡単に教えて。", "眠れない夜はどうしたらいい？",
           "日本のおすすめの観光地は？", "勉強に集中するコツは？", "猫と犬、どっちが好き？", "健康のために何をしている？",
           "おすすめの飲み物は？", "部屋を片付けるコツは？", "友達の誕生日に何を贈ればいい？", "朝早く起きるコツは？"]
# モデル自身の素の返事に見立てた、です・ます調の返事(どの返事も文末が「です。」「ます。」で終わる文を含む)
REPLIES = ["天気が良ければ公園を散歩するのがおすすめです。家ではゆっくり本を読むのもいいと思います。",
           "ご飯と味噌汁と卵焼きがおすすめです。忙しい日はバナナとヨーグルトでも十分です。",
           "料理は手軽に始められる趣味です。散歩や写真も気軽に楽しめます。",
           "雨の日は家で映画を見るのが好きです。温かいお茶を飲みながら過ごします。",
           "まずはしっかり眠ることが大切です。お風呂にゆっくり入るのも効果があります。",
           "私は秋が一番好きです。涼しくて食べ物がおいしい季節です。",
           "野菜と肉を炒めてから水で煮込みます。最後にルーを溶かせば完成です。",
           "寝る前にスマホを見ないようにすると眠りやすくなります。温かい牛乳を飲むのもおすすめです。",
           "京都はお寺や神社が多くておすすめです。北海道では自然とおいしい料理が楽しめます。",
           "時間を決めて短く区切るのが効果的です。机の上を片付けると集中しやすくなります。",
           "どちらもかわいいですが、私は猫が好きです。気ままなところが魅力です。",
           "毎日少しずつ歩くようにしています。野菜を多めに食べることも心がけています。",
           "温かい緑茶がおすすめです。夏は冷たい麦茶もおいしいです。",
           "まず要らない物を捨てることが大切です。物の置き場所を決めると散らかりにくくなります。",
           "相手の好きな物を選ぶのが一番です。手紙を添えると気持ちが伝わります。",
           "毎日同じ時間に寝るようにするといいです。起きたらすぐにカーテンを開けて光を浴びます。"]
EN = {"name": "english",
      "prompts": ["What is a good way to spend the weekend?", "How do I make a simple curry?",
                  "Any tips for falling asleep faster?", "What should I give a friend for their birthday?"],
      "replies": ["Going for a walk in the park is a nice way to relax. Reading a book at home is also good.",
                  "Fry vegetables and meat, then simmer them in water. Add the curry roux at the end.",
                  "Avoid screens before bed. A warm drink can also help you relax.",
                  "Choose something they like. Adding a handwritten note makes it special."]}
LAYERS = [8, 14]
UNITS = [f"{L}:ffn:keyed:a1@zun_eval#zun_eval" for L in LAYERS]


def _mini_suite(tmp_path):
    (tmp_path / "ppl").mkdir()
    (tmp_path / "ppl" / "en.txt").write_text("Alice was beginning to get very tired of sitting by her sister on the bank. " * 2)
    (tmp_path / "zh_only_chars.txt").write_text("# test\n这们说\n")
    d = {"name": "mini", "max_tokens": {"exact": 8, "format": 16, "mix": 16, "persona": 16},
         "exact": [{"id": "add", "prompt": "What is 17 + 25? Answer with just the number.", "answer": ["42"]}],
         "format": [{"id": "j", "prompt": "Return a JSON object with keys \"a\" and \"b\". Output only the JSON.",
                     "check": "json", "keys": ["a", "b"]}],
         "mix": [{"id": "m", "prompt": "春の好きなところを 2 文で教えてください。"}],
         "persona": {"prompt_sets": ["ja"]}, "ppl": [{"id": "en", "path": "ppl/en.txt"}]}
    p = tmp_path / "mini.yaml"
    p.write_text(yaml.safe_dump(d, allow_unicode=True))
    return p


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip(f"{MODEL} がありません")
    from gne.api import Session
    s = Session(MODEL, root=tmp_path_factory.mktemp("c07e"), n_gpu_layers=NGL, n_ctx=1024)
    yield s
    s.close()


@pytest.fixture(scope="module")
def zun(session, tmp_path_factory):
    """ずんだもんの対照集合に返事を足したファイル 1 つで、方向(system_a 対 なし、生成なしの last)と鍵(mode same、
    「です。」「ます。」の直前、subset even、擬似鍵 4 本)を作る。"""
    from gne.keyed_build import build_keys
    d = yaml.safe_load(ZUNDAMON.read_text(encoding="utf-8"))
    d.update({"name": "zun_eval", "replies": REPLIES, "dont_touch": [EN]})
    d.pop("validate", None)
    p = tmp_path_factory.mktemp("c07e_spec") / "zun_eval.yaml"
    p.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    session.direction(contrast=p, positions="last")
    build_keys(session, p, name="zun_eval", layers=LAYERS, max_tokens=48, q_energy=1.0, pseudo=4)
    return p


@pytest.mark.model
def test_eval_keyed_block(session, zun, tmp_path):
    """短い eval: 結果の keyed の形(方式 5 行、層ごとの群、擬似鍵、表示)、keyed の側 B の ‖Δ‖ < 同じ α の add、
    行の keyed(倍率 0 は 0)、擬似鍵の対照(対照の行に鍵 NAME~j と選択性)。対照集合は方向の source から引く。"""
    from gne.evalx import evaluate
    r = session.recipe(units=UNITS, name="zk")
    d = evaluate(session, r, doses=(0, 1), suite=_mini_suite(tmp_path), random_controls=2, direction_metrics=False,
                 keyed_examples=4, keyed_pseudo=4)
    k = d["keyed"]
    assert "skipped" not in k, k
    assert k["contrast_from"] == "direction_source" and k["held_out"] and k["subset"] == "odd" and k["mode"] == "same"
    assert k["layers"] == LAYERS and k["objective_layer"] == 14 and k["key"] == "zun_eval"
    assert k["counts"]["A"]["examples"] == 4 and k["counts"]["B"]["examples"] == 4 and k["counts"]["dt:english"]["examples"] == 2
    assert k["counts"]["B"]["positions"] > k["counts"]["A"]["positions"] >= 4
    assert k["replies"] == {"AB": "given", "dt:english": "given"}
    names = [m["method"] for m in k["methods"]]
    assert names == ["keyed", "add_same_alpha", "add", "gain", "proj"]
    by = {m["method"]: m for m in k["methods"]}
    for m in k["methods"]:
        if "excluded" in m:
            assert isinstance(m["excluded"], str) and m["method"] in ("gain", "proj")
            continue
        assert set(m["layers"]) == {"8", "14"}
        assert set(m["layers"]["14"]) == {"A", "B", "dt:english"}
        for key in ("a_par", "b_par", "b_norm", "dt_norm", "selectivity", "b_over_a"):
            assert isinstance(m[key], float), (m["method"], key)
    kd, a1 = by["keyed"], by["add_same_alpha"]
    assert kd["a_par"] > 0 and kd["selectivity"] > 0
    assert kd["b_norm"] < a1["b_norm"] and kd["dt_norm"] < a1["dt_norm"]          # 側 B と dont_touch をあまり動かさない
    assert kd["b_over_a"] < a1["b_over_a"]
    assert by["add"]["matched"] and abs(by["add"]["a_par"] - kd["a_par"]) <= 0.1 * abs(kd["a_par"])
    assert a1["recipe"]["dose_unit"] == "raw" and a1["units"] == [f"L{L} dir:zun_eval +1" for L in LAYERS]
    assert k["compare"]["b_norm_ratio"]["add_same_alpha"] < 1
    p = k["pseudo"]
    assert p["n"] == 4 and p["keys"] == [f"zun_eval~{j}" for j in range(4)] and len(p["values"]) == 4
    assert p["real"] == kd["selectivity"] and isinstance(p["beats"], bool)
    assert k["text"] and all(isinstance(x, str) for x in k["text"]) and "add_same_alpha" in "\n".join(k["text"])
    assert k["cost"]["prefills"] > 0
    rows = {row["dose"]: row for row in d["rows"]}
    assert rows[0.0]["keyed"]["a_par"] == 0.0 and rows[0.0]["keyed"]["selectivity"] == 0.0
    assert rows[1.0]["keyed"]["layer"] == 14
    assert rows[1.0]["keyed"]["selectivity"] == pytest.approx(kd["selectivity"], rel=1e-6)   # keyed だけのレシピなので同じ
    # 擬似鍵の対照(control_set の keyed の分岐)
    assert d["random_controls"] == 2
    cr = rows[1.0]["ctrl"]["rows"]
    assert [x["keys"] for x in cr] == [["zun_eval~0"], ["zun_eval~1"]]
    assert all(isinstance(x["keyed_selectivity"], float) for x in cr)
    assert rows[1.0]["keyed"]["null_pct"] is not None and rows[1.0]["ctrl"]["keyed_selectivity"]["n"] == 2


@pytest.mark.model
def test_keyed_contrast_required(session, zun, tmp_path):
    """鍵を作った対照集合が方向の source と違うと、対照集合を渡さない限り測らない(BAD_ARGS。eval では keyed に
    skipped)。渡せば測れる。返事が正規表現に一致しなければ BAD_CONTRAST。"""
    from gne.keyed_build import build_keys
    from gne.keyed_eval import KeyedMetrics
    other = tmp_path / "plain.yaml"
    other.write_text(yaml.safe_dump({"name": "plain", "positions": "before:(です|ます)。", "user_prompts": PROMPTS,
                                     "replies": REPLIES, "dont_touch": [EN]}, allow_unicode=True), encoding="utf-8")
    build_keys(session, other, name="plain_k", layers=[8], max_tokens=48, pseudo=0)
    r = session.recipe(units=["8:ffn:keyed:a1@zun_eval#plain_k"], name="pk")
    with pytest.raises(GneError) as ei:
        KeyedMetrics(session, r)
    assert ei.value.code == "BAD_ARGS" and "keyed_contrast" in (ei.value.hint or "")
    km = KeyedMetrics(session, r, contrast=other, max_examples=2, pseudo=4, match=False)
    assert km.n_pseudo == 0 and any("擬似鍵が無い" in n for n in km.notes)
    km.prepare()
    blk = km.block()
    assert [m["method"] for m in blk["methods"]] == ["keyed", "add_same_alpha"] and blk["pseudo"] is None
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"name": "plain", "positions": "before:(です|ます)。", "user_prompts": PROMPTS,
                                   "replies": ["のだ。"] * len(PROMPTS)}, allow_unicode=True), encoding="utf-8")
    km2 = KeyedMetrics(session, r, contrast=bad, max_examples=2, pseudo=0)
    with pytest.raises(GneError) as ei:
        km2.prepare()
    assert ei.value.code == "BAD_CONTRAST"
