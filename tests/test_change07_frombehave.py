"""CHANGE-07 のテスト(CHiPS で残った場面を keyed で抑える): gne behave の結果から、CHiPS のレシピを当てても消したい振る舞いが
残った A 側の質問を集めて、keyed の対照集合の JSON を作る(gne direction key-from-behave、Session.direction_key_from_behave、
MCP direction_key_from_behave、gne/keyed_from_behave.py)。

モデルなし(合成した behave の結果と対照集合): 正規表現の (?i) の付け方と位置の数え方、抜き出し(正規表現・読み手のラベル・種の割合)、
出力の JSON の形と保存先(<root>/keyed/、-o)、警告(レシピ違い・A 側に無い質問・少なすぎ・広すぎる正規表現・一致の無い質問・
B 側の重なり・system で作った A 側。strict ならエラー)、古い behave の結果(質問の記録なし)、behave が質問とレシピの名前を残すこと、
ひな形の keyed.direction・keyed.iv の読み方(一括・ジョブ)、UI の「ひな形」に <root>/keyed/ が出ること、MCP の入口、CLI のヘルプ。
モデルあり(Qwen3-0.6B、GNE_TEST_MODEL。CPU で回すときは GNE_TEST_NGL=0): CLI の --json、作った JSON から direction_key_pipeline で
鍵を作る(方向は keyed.direction を使い回し、iv に CHiPS のレシピ)、ブラウザで <root>/keyed/ のひな形を選ぶと入力欄に入る。
"""

import json
import os
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from gne.steer import GneError

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
NGL = int(os.environ.get("GNE_TEST_NGL", "-1"))
CHROME = Path(os.environ.get("GNE_CHROME", Path.home() / ".cache/ms-playwright/chromium-1223/chrome-linux64/chrome"))
ENV_STUB = {"schema_version": "1", "command": "x", "model_id": "m", "llama_cpp_version": "v", "seed": 0,
            "generated_at": "2026-11-09T00:00:00+00:00", "warnings": []}

# CHiPS の対照集合に見立てたもの(ツンデレ。A 側 = 褒められた・お礼を言われた場面、B 側 = ふつうの質問)
PROMPTS_A = ["今日なんか可愛いね。", "昨日は手伝ってくれてありがとう。", "髪切った？似合ってるね。", "誕生日おめでとう！",
             "君の料理、すごくおいしかったよ。", "いつもありがとう、頼りにしてるよ。", "君の歌、とても上手だったよ。", "傘を貸してくれてありがとう。"]
PROMPTS_B = ["今日の天気はどう？", "おすすめの本は？", "カレーの作り方を教えて。", "朝ごはんに何を食べればいい？",
             "好きな季節は？", "疲れたときはどうすればいい？"]
MATCH = "べ、別に|ふん|Hmph"
TSUN = {"name": "tsun", "prompts_a": PROMPTS_A, "prompts_b": PROMPTS_B,
        "validate": {"match": MATCH, "target": "a tsundere girl who hides her embarrassment with harsh words"}}
# レシピを当てても照れ隠しが残った返事(3 つずつ。返事の冒頭には置かない)と、残らなかった返事
LEFT = "そんなこと言われても、ふん、知らないんだから。でも、ふん、まあいいわ。ほんとに、ふん。"
LEFT_EN = "Well, if you say so. Hmph, it is nothing. Hmph, do not mention it. Really, hmph."
PLAIN = "ありがとう、とても嬉しいです。これからもよろしくお願いします。"


def _store_session(tmp_path, model_id="m1"):
    """Session の代わり(モデルなし): 本物の Store と、Session の方向・対照集合・レシピの引き方。"""
    from gne.api import Session
    from gne.store import Store
    s = SimpleNamespace(store=Store(tmp_path / "root"), meta=SimpleNamespace(model_id=model_id), warnings=[])
    s.warn = s.warnings.append
    s._contrast_file = Session._contrast_file
    s.recipe_path = lambda x: Session.recipe_path(s, x)
    s.direction_path = lambda x: Session.direction_path(s, x)
    s.direction_set = lambda x: Session.direction_set(s, x)
    return s


def _write_contrast(tmp_path, d, name=None):
    p = tmp_path / f"{name or d['name']}.yaml"
    p.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def _fake_direction(s, name, source):
    """方向のファイルに見立てた npz(作った対照集合 source と、その sha を記録)。"""
    from gne.reverse import ContrastSet
    p = s.store.direction_dir(s.meta.model_id) / f"{name}.npz"
    np.savez(p, layers=np.array([1], dtype=np.int32), layer_1=np.ones(4, np.float32), method="mean_diff", positions="last",
             contrast_sha=ContrastSet.load(source).sha, source=str(source), n_rows_A=1, n_rows_B=1)
    return p


def _chips(s, name="chips", units=None):
    """CHiPS のレシピ(proj の組)を <root>/recipes/ に保存する。"""
    units = units or [{"layer": 10, "kind": "head", "head": 2, "mode": "proj", "s": 3}]
    p = s.store.recipes_dir() / f"{name}.yaml"
    p.write_text(yaml.safe_dump({"name": name, "default_direction": "tsun", "units": units}, allow_unicode=True), encoding="utf-8")
    return p


def _rec_dict(path, name):
    from gne.steer import Recipe, combine_recipes
    return combine_recipes([Recipe.load(path)], name=name).to_dict()


def _behave(s, name, texts_by_seed, *, prompts=PROMPTS_A, recipe=None, match=None, labels_by_seed=None, thinking=False,
            loops=None, record_prompts=True, max_tokens=64, n_tokens=40, recipe_names=("chips",)):
    """gne behave の結果に見立てた JSON(<root>/behave/<model_id>/<name>.json)。texts_by_seed[seed][i] が質問 i の返事。"""
    runs = []
    for k, texts in enumerate(texts_by_seed):
        reps = []
        for i, t in enumerate(texts):
            r = {"i": i, "text": t, "full": (f"<think>考え中</think>{t}" if thinking else None), "finish": "eog",
                 "n_tokens": n_tokens * (2 if thinking else 1), "loop": bool(loops and (k, i) in loops)}
            if labels_by_seed is not None:
                r["label"] = labels_by_seed[k][i]
            reps.append(r)
        runs.append({"seed": k, "counts": {}, "replies": reps, "seconds": 0.1})
    d = {"runs": runs, "n_prompts": len(prompts), "max_tokens": max_tokens, "thinking": thinking, "temperature": 0.7,
         "match": match, "judge": labels_by_seed is not None, "labels": ["R", "P", "C"], "name": name, "recipe": recipe,
         "recipe_names": list(recipe_names)}
    if record_prompts:
        d["prompts"] = list(prompts)
    p = s.store.root / "behave" / s.meta.model_id / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture
def world(tmp_path):
    """対照集合 tsun(prompts_a / prompts_b)・方向 tsun・レシピ chips と、3 種の behave の結果 left:
    質問 0・1・5・6・7 は 3 種とも照れ隠しが残り(5 は英語で、種 2 だけ小文字の hmph)、2 は 2 種、3 は 1 種、4 は残らない。
    質問 0 の種 0 の返事は繰り返し(出力の返事には繰り返していない種を選ぶ)。"""
    s = _store_session(tmp_path)
    cp = _write_contrast(tmp_path, TSUN)
    _fake_direction(s, "tsun", cp)
    rp = _chips(s)
    seeds = []
    for k in range(3):
        t = [LEFT, LEFT, LEFT if k < 2 else PLAIN, LEFT if k == 0 else PLAIN, PLAIN,
             LEFT_EN if k < 2 else LEFT_EN.replace("Hmph", "hmph"), LEFT, LEFT]
        seeds.append(t)
    seeds[0][0] = LEFT + "ふん、ふん、ふん。"
    _behave(s, "left", seeds, recipe=_rec_dict(rp, "left"), loops={(0, 0)})
    return SimpleNamespace(s=s, contrast=cp, recipe=rp, tmp=tmp_path, seeds=seeds)


# --- 正規表現と数え方 -------------------------------------------------------------------------------------------

def test_ignore_case_and_counting():
    """validate.match → before: の正規表現: 大文字と小文字のある文字を含めば (?i) を付ける(すでに無視するなら付けない。エスケープ・
    グループの書き方は数えない。かなだけなら付けない)。返事の書き込みたい位置の数の見込み(まばらなら一致の数、密なら区間の数)。"""
    import re

    from gne.keyed_from_behave import has_cased, ignore_case, match_starts, reply_positions, reply_tokens
    assert ignore_case("のだ|なのだ") == "のだ|なのだ" and ignore_case("[ぁ-んァ-ン]") == "[ぁ-んァ-ン]"
    assert ignore_case(MATCH) == "(?i)" + MATCH
    assert ignore_case("(?i)hmph") == "(?i)hmph" and ignore_case("(?ai)hmph") == "(?ai)hmph"
    x = ignore_case("(?s)Hmph.")
    assert x == "(?i)(?s)Hmph." and re.compile(x).flags & re.IGNORECASE
    for p in (r"\w+です", r"ぁ\x41", r"(?P<g>ふん)(?P=g)", r"\d{2}\s*円", r"\N{HIRAGANA LETTER A}"):
        assert not has_cased(p) and ignore_case(p) == p, p
    assert has_cased("ｈｍｐｈ") and has_cased("Ω") and not has_cased("べ、別に")
    rx = re.compile("ふん")
    assert match_starts(LEFT, rx) == 3 and reply_positions(LEFT, rx, 40) == 3
    dense = re.compile("[ぁ-んァ-ン]")
    n = reply_positions(PLAIN, dense, 10)
    assert n <= 10 and n < match_starts(PLAIN, dense)                       # 1 つのトークンに入る始まりは 1 つと数える
    assert reply_positions("", rx, 5) == 0
    assert reply_tokens({"text": "abcd", "n_tokens": 10}) == 10.0
    assert reply_tokens({"text": "ab", "full": "abcd", "n_tokens": 10}) == pytest.approx(5.0)   # 思考ありは答えの部分だけ
    assert reply_tokens({"text": "abc"}) == 3.0


# --- 抜き出しと出力 -----------------------------------------------------------------------------------------------

def test_extract_by_regex_and_output(world):
    """既定: 判定は validate.match(大文字小文字を無視)、全部の種で出た質問だけ。出力は <root>/keyed/<名前>.json で、one_scene.json と
    同じ形(about の ja / en・name・positions・user_prompts・dont_touch の ordinary = B 側の質問・validate・keyed)に、behave の返事
    replies。鍵の対照集合として読め、keyed の設定(direction・alpha −1・iv)が一括の既定になる。"""
    from gne.keyed_build import read_key_spec
    from gne.keyed_from_behave import from_behave
    from gne.keyed_pipeline import file_settings
    from gne.schemas import DirectionKeyFromBehaveOut
    s = world.s
    d = from_behave(s, "left", recipe="chips", direction="tsun")
    DirectionKeyFromBehaveOut.model_validate({**ENV_STUB, **{k: v for k, v in d.items() if k != "warnings"}})
    assert [r["i"] for r in d["extracted"]] == [0, 1, 5, 6, 7] and all(r["rate"] == 1.0 for r in d["extracted"])
    assert d["judge"] == {"method": "match", "source": "validate", "match": MATCH, "ignore_case": True, "labels": [], "min_rate": 1.0}
    assert d["n_partial"] == 2 and d["n_none"] == 1 and d["checks"] == {"recipe": True, "questions_in_a": 8}
    assert d["positions"] == f"before:(?i){MATCH}" and d["positions_source"] == "validate"
    assert d["iv"] == "chips" and d["direction"] == "tsun" and d["contrast"] == "tsun" and d["replies"] is True
    path = Path(d["path"])
    assert path == s.store.root / "keyed" / "left_keyed.json" and d["name"] == "left_keyed"
    j = json.loads(path.read_text(encoding="utf-8"))
    assert list(j) == ["about", "name", "positions", "user_prompts", "replies", "dont_touch", "validate", "keyed"]
    assert j["user_prompts"] == [PROMPTS_A[i] for i in (0, 1, 5, 6, 7)]
    assert j["replies"][0] == LEFT and j["replies"][2] == LEFT_EN          # 質問 0 は繰り返していない種 1 の返事
    assert d["extracted"][0]["seed"] == 1 and not d["extracted"][0]["loop"]
    assert j["dont_touch"] == [{"name": "ordinary", "prompts": PROMPTS_B}]   # positions は省く(返事全体)
    assert j["validate"] == TSUN["validate"] and j["keyed"] == {"direction": "tsun", "alpha": -1, "iv": "chips"}
    a = j["about"]
    for lang in ("ja", "en"):
        assert "left" in a[lang]["title"] and "chips" in a[lang]["title"]
        assert "tsun" in a[lang]["text"] and "iv" in a[lang]["text"] and "−1" in a[lang]["text"]
    assert str(world.tmp) not in json.dumps(a, ensure_ascii=False)          # 説明に手元のパスを書かない
    spec = read_key_spec(path)                                              # JSON は YAML としても読める
    assert spec.contrast.name == "left_keyed" and spec.positions == j["positions"] and [g.name for g in spec.dont_touch] == ["ordinary"]
    assert spec.replies["same"] == j["replies"] and spec.dont_touch[0].positions is None
    assert file_settings(path) == {"direction": "tsun", "alpha": -1, "iv": "chips"}
    sh = d["position_share"]
    assert 0 < sh["share"] < 0.25 and not sh["broad"] and sh["threshold"] == 0.25
    assert d["halves"]["even"] == {"questions": 3, "examples": 3, "positions": 9} and d["no_match"] == 0
    assert d["ordinary"] == {"n": 6, "removed": 0}
    assert any("min_rate" in n for n in d["notes"])
    # min_rate を下げると 2 種で出た質問も入る。名前と -o
    out = world.tmp / "elsewhere" / "x.json"
    d2 = from_behave(s, "left", recipe="chips", direction="tsun", min_rate=0.6, name="left2", out=out)
    assert [r["i"] for r in d2["extracted"]] == [0, 1, 2, 5, 6, 7] and d2["path"] == str(out) and out.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["name"] == "left2" and not (s.store.root / "keyed" / "left2.json").exists()
    assert d2["extracted"][2]["rate"] == pytest.approx(2 / 3)


def test_judge_sources_regex(world, tmp_path):
    """判定の出どころ: behave の --match があればそれ(大文字小文字を区別するので、小文字の hmph の種は「出ていない」)、match を渡せば
    それ。どちらも無ければ validate.match(上のテスト)。"""
    from gne.keyed_from_behave import from_behave
    s = world.s
    p = s.store.root / "behave" / "m1" / "left.json"
    bd = json.loads(p.read_text(encoding="utf-8"))
    bd["match"] = MATCH
    p.write_text(json.dumps(bd, ensure_ascii=False), encoding="utf-8")
    d = from_behave(s, "left", recipe="chips", direction="tsun")
    assert d["judge"]["source"] == "behave" and not d["judge"]["ignore_case"]
    assert [r["i"] for r in d["extracted"]] == [0, 1, 6, 7]                 # 5 は種 2 が小文字
    d = from_behave(s, "left", recipe="chips", direction="tsun", match="Hmph|hmph")
    assert d["judge"]["source"] == "match" and [r["i"] for r in d["extracted"]] == [5]
    with pytest.raises(GneError) as ei:
        from_behave(s, "left", recipe="chips", direction="tsun", match="(")
    assert ei.value.code == "BAD_ARGS"


def test_extract_by_label_and_no_match(world):
    """読み手のラベルで判定する(--label P、複数可)。ラベルでは残ったが正規表現に一致しない質問は数えて知らせる(書き込みたい位置ができない)。
    behave に判定が無い・知らないラベル・正規表現とラベルの両方は BAD_ARGS。"""
    from gne.keyed_from_behave import from_behave
    s = world.s
    rp = world.recipe
    texts = [[LEFT, "照れてなんかないから。", LEFT, PLAIN, PLAIN, PLAIN, LEFT, LEFT]] * 2
    labs = [["P", "P", "P", "R", "C", "R", "P", "C"], ["P", "P", "P", "R", "R", "R", "C", "C"]]
    _behave(s, "lab", texts, recipe=_rec_dict(rp, "lab"), labels_by_seed=labs)
    d = from_behave(s, "lab", recipe="chips", direction="tsun", labels="P")
    assert d["judge"]["method"] == "label" and d["judge"]["labels"] == ["P"] and d["judge"]["match"] is None
    assert [r["i"] for r in d["extracted"]] == [0, 1, 2] and d["n_partial"] == 1
    assert d["no_match"] == 1 and any("一致が無く" in w and "昨日は手伝ってくれてありがとう" in w and "positions で渡す" in w
                                      for w in d["warnings"])
    d = from_behave(s, "lab", recipe="chips", direction="tsun", labels=["P", "C"], name="lab2")
    assert [r["i"] for r in d["extracted"]] == [0, 1, 2, 6, 7]
    for kw, msg in (({"labels": "X"}, "X"), ({"labels": "P", "match": "ふん"}, "どちらか")):
        with pytest.raises(GneError) as ei:
            from_behave(s, "lab", recipe="chips", direction="tsun", **kw)
        assert ei.value.code == "BAD_ARGS" and msg in ei.value.message
    with pytest.raises(GneError) as ei:
        from_behave(s, "left", recipe="chips", direction="tsun", labels="P")       # left は判定なし
    assert ei.value.code == "BAD_ARGS" and "判定がありません" in ei.value.message


# --- 確かめと警告 -------------------------------------------------------------------------------------------------

def test_recipe_and_question_checks(world, tmp_path):
    """behave の結果が、そのレシピを当てて A 側の質問に答えたものか: レシピなし・別のレシピ・ほかのレシピを重ねたもの・A 側に無い質問・
    A 側の質問が 1 つも無い、はどれも警告(strict ならエラー)。"""
    from gne.keyed_from_behave import from_behave
    s = world.s
    rp = world.recipe
    other = _chips(s, "other", [{"layer": 12, "kind": "head", "head": 5, "mode": "proj", "s": 2}])
    stacked = _rec_dict(rp, "x")
    stacked["units"] = stacked["units"] + _rec_dict(other, "x")["units"]
    for nm, rec, frag in (("nore", None, "レシピを当てずに"), ("other", _rec_dict(other, "o"), "違います"),
                          ("stack", stacked, "重ねて")):
        _behave(s, nm, world.seeds, recipe=rec)
        d = from_behave(s, nm, recipe="chips", direction="tsun")
        assert d["checks"]["recipe"] is False and any(frag in w for w in d["warnings"]), (nm, d["warnings"])
        with pytest.raises(GneError) as ei:
            from_behave(s, nm, recipe="chips", direction="tsun", strict=True)
        assert ei.value.code == "BAD_ARGS" and frag in ei.value.message
    mixed = PROMPTS_A[:4] + PROMPTS_B[:4]
    _behave(s, "mixed", world.seeds, prompts=mixed, recipe=_rec_dict(rp, "m"))
    d = from_behave(s, "mixed", recipe="chips", direction="tsun")
    assert d["checks"]["questions_in_a"] == 4 and any("4 問は、CHiPS の対照集合 tsun の A 側の質問にありません" in w for w in d["warnings"])
    _behave(s, "bside", world.seeds, prompts=PROMPTS_B + ["a", "b"], recipe=_rec_dict(rp, "b"))
    d = from_behave(s, "bside", recipe="chips", direction="tsun")
    assert d["checks"]["questions_in_a"] == 0 and any("1 つもありません" in w for w in d["warnings"])
    with pytest.raises(GneError):
        from_behave(s, "bside", recipe="chips", direction="tsun", strict=True)
    d = from_behave(s, "left", recipe="chips", direction="tsun", strict=True)        # 合っていれば strict でも通る
    assert d["path"] and d["checks"]["recipe"] is True


def test_system_contrast_and_b_overlap(world, tmp_path):
    """system で両側を作る対照集合: A 側が system_a なので警告し、B 側の質問(= 同じ質問)のうち user_prompts と同じものは ordinary から
    外して警告する。B 側が全部重なれば dont_touch を書かずに警告する。"""
    from gne.keyed_from_behave import from_behave
    s = world.s
    sysc = {"name": "tsun_sys", "system_a": "あなたはツンデレの女の子です。", "user_prompts": PROMPTS_A, "validate": {"match": MATCH}}
    cp = _write_contrast(tmp_path, sysc)
    d = from_behave(s, "left", recipe="chips", contrast=str(cp), direction="tsun")
    assert any("system_a で作った状態" in w and "向きません" in w for w in d["warnings"])
    assert d["ordinary"] == {"n": 3, "removed": 5} and any("5 問は user_prompts と同じ" in w for w in d["warnings"])
    j = json.loads(Path(d["path"]).read_text(encoding="utf-8"))
    assert j["dont_touch"][0]["prompts"] == [PROMPTS_A[i] for i in (2, 3, 4)]
    assert any("中身の違うファイル" in w for w in d["warnings"])           # 方向 tsun は別の対照集合から作った
    allp = [[LEFT] * 8] * 3
    _behave(s, "all", allp, recipe=_rec_dict(world.recipe, "all"))
    d = from_behave(s, "all", recipe="chips", contrast=str(cp), direction="tsun")
    j = json.loads(Path(d["path"]).read_text(encoding="utf-8"))
    assert "dont_touch" not in j and any("入れる質問が残りません" in w for w in d["warnings"])


def test_few_broad_empty_and_options(world, tmp_path):
    """少なすぎ(鍵の作成と key-check の目安 2 例・8 位置)、広すぎる正規表現(かな 1 文字)、残った質問が無い(JSON を書かない)、
    positions は before: だけ、min_rate の範囲、思考ありの結果と --no-replies(返事を入れず keyed.max_tokens)。"""
    from gne.keyed_from_behave import BROAD_SHARE, from_behave
    s = world.s
    d = from_behave(s, "left", recipe="chips", direction="tsun")
    few = [w for w in d["warnings"] if "目安は 2 例・8 位置以上" in w]
    assert len(few) == 1 and few[0].startswith("key-check(奇数番目)") and "6 位置" in few[0]       # 奇数番目は 2 例の 6 位置
    d = from_behave(s, "left", recipe="chips", direction="tsun", positions="before:[ぁ-んァ-ン]", name="broad")
    sh = d["position_share"]
    assert sh["broad"] and sh["share"] >= BROAD_SHARE
    assert any("一致が広すぎます" in w and "その言語を書いている状態" in w and "--positions" in w for w in d["warnings"])
    d = from_behave(s, "left", recipe="chips", direction="tsun", positions="before:ふん", name="narrow")
    assert d["positions"] == "before:ふん" and d["positions_source"] == "given" and not d["position_share"]["broad"]
    _behave(s, "none", [[PLAIN] * 8] * 2, recipe=_rec_dict(world.recipe, "none"))
    d = from_behave(s, "none", recipe="chips", direction="tsun")
    assert d["path"] is None and d["extracted"] == [] and any("JSON は書きません" in w for w in d["warnings"])
    assert not (s.store.root / "keyed" / "none_keyed.json").exists()
    for kw in ({"positions": "response"}, {"min_rate": 0}, {"min_rate": 1.5}, {"name": "a/b"}):
        with pytest.raises(GneError) as ei:
            from_behave(s, "left", recipe="chips", direction="tsun", **kw)
        assert ei.value.code == "BAD_ARGS", kw
    d = from_behave(s, "left", recipe="chips", direction="tsun", replies=False, name="norep")
    j = json.loads(Path(d["path"]).read_text(encoding="utf-8"))
    assert "replies" not in j and j["keyed"]["max_tokens"] == 64 and d["replies"] is False
    assert any("作り直すので変わります" in w for w in d["warnings"])
    _behave(s, "think", world.seeds, recipe=_rec_dict(world.recipe, "think"), thinking=True)
    d = from_behave(s, "think", recipe="chips", direction="tsun")
    j = json.loads(Path(d["path"]).read_text(encoding="utf-8"))
    assert "replies" not in j and any("思考あり" in w for w in d["warnings"])
    assert d["extracted"][0]["tokens"] < 80                                  # 思考の分を除いた見込み


def test_inputs_old_behave_direction_contrast_recipe(world, tmp_path):
    """入力: 質問の記録の無い古い behave の結果は prompts(質問集)で補う。behave はパスでもよい。方向だけ(方向を作った対照集合)・
    対照集合だけ(方向の名前は対照集合の name)・まだ無い方向。レシピはパスでもよい(iv はパス)。validate.match が無ければ positions が要る。"""
    from gne.keyed_from_behave import from_behave
    s = world.s
    old = _behave(s, "old", world.seeds, recipe=_rec_dict(world.recipe, "old"), record_prompts=False)
    with pytest.raises(GneError) as ei:
        from_behave(s, "old", recipe="chips", direction="tsun")
    assert ei.value.code == "BAD_ARGS" and "prompts" in ei.value.hint
    qf = tmp_path / "q.yaml"
    qf.write_text(yaml.safe_dump(PROMPTS_A, allow_unicode=True), encoding="utf-8")
    d = from_behave(s, str(old), recipe="chips", direction="tsun", prompts=qf)
    assert [r["i"] for r in d["extracted"]] == [0, 1, 5, 6, 7]
    qf.write_text(yaml.safe_dump(PROMPTS_A[:3], allow_unicode=True), encoding="utf-8")
    with pytest.raises(GneError) as ei:
        from_behave(s, "old", recipe="chips", direction="tsun", prompts=qf)
    assert ei.value.code == "BAD_ARGS" and "n_prompts" in ei.value.message
    with pytest.raises(GneError) as ei:
        from_behave(s, "nope", recipe="chips", direction="tsun")
    assert ei.value.code == "FILE_NOT_FOUND"
    with pytest.raises(GneError) as ei:
        from_behave(s, "left", recipe="chips")                                # 方向も対照集合も無い
    assert ei.value.code == "BAD_ARGS"
    d = from_behave(s, "left", recipe="chips", contrast=str(world.contrast), name="c1")
    assert d["direction"] == "tsun" and not any("まだありません" in w for w in d["warnings"])
    d = from_behave(s, "left", recipe="chips", contrast=str(world.contrast), direction="tsun_new", name="c2")
    assert d["direction"] == "tsun_new" and d["keyed"]["direction"] == "tsun_new" and any("まだありません" in w for w in d["warnings"])
    with pytest.raises(GneError) as ei:
        from_behave(s, "left", recipe="chips", direction="tsun_new")         # 対照集合も無ければ探せない
    assert ei.value.code == "FILE_NOT_FOUND"
    ext = tmp_path / "ext" / "chips_ext.yaml"
    ext.parent.mkdir()
    ext.write_text(world.recipe.read_text(encoding="utf-8").replace("name: chips", "name: chips_ext"), encoding="utf-8")
    d = from_behave(s, "left", recipe=str(ext), direction="tsun", name="c3")
    assert d["iv"] == str(ext.resolve()) and d["recipe"] == "chips_ext" and d["checks"]["recipe"] is True
    nov = _write_contrast(tmp_path, {**TSUN, "name": "noval", "validate": {"target": "x"}})
    with pytest.raises(GneError) as ei:                                       # 判定の正規表現も無い
        from_behave(s, "left", recipe="chips", contrast=str(nov), direction="tsun", name="c4")
    assert ei.value.code == "BAD_ARGS" and "判定がありません" in ei.value.message
    with pytest.raises(GneError) as ei:                                       # 書き込みたい位置を決められない
        from_behave(s, "left", recipe="chips", contrast=str(nov), direction="tsun", name="c4", match="ふん")
    assert ei.value.code == "BAD_ARGS" and "validate.match" in ei.value.message and "positions" in ei.value.hint
    d = from_behave(s, "left", recipe="chips", contrast=str(nov), direction="tsun", name="c4", match="ふん|Hmph",
                    positions="before:(?i)ふん|hmph")
    assert d["path"] and d["judge"]["source"] == "match" and json.loads(Path(d["path"]).read_text(encoding="utf-8"))["validate"] == {"target": "x"}


def test_session_behave_records_prompts_and_recipe_names(tmp_path, monkeypatch):
    """Session.behave は結果のファイルに質問(prompts)と当てたレシピの名前(recipe_names)を残す(画面と戻り値には質問を出さない)。"""
    import gne.behave as B
    from gne.api import Session
    s = _store_session(tmp_path)
    rp = _chips(s)
    s.lora_for = lambda r: None
    s.vectors = lambda r: ({}, None)
    s.progress = None
    s._as_recipe = lambda r: r
    monkeypatch.setattr(B, "behave", lambda sess, qs, **kw: {"runs": [{"seed": 0, "counts": {"n": len(qs)}, "replies": []}],
                                                             "n_prompts": len(qs), "max_tokens": kw["max_tokens"], "thinking": False,
                                                             "temperature": 0.0, "match": kw["match"], "judge": False, "labels": []})
    out = Session.behave(s, PROMPTS_A[:3], recipes=["chips"], match="ふん", name="rec1")
    assert "prompts" not in out and out["recipe_names"] == ["chips"]
    saved = json.loads(Path(out["path"]).read_text(encoding="utf-8"))
    assert saved["prompts"] == PROMPTS_A[:3] and saved["recipe_names"] == ["chips"] and saved["recipe"]["units"]
    out = Session.behave(s, PROMPTS_A[:2], name="rec2")
    saved = json.loads(Path(out["path"]).read_text(encoding="utf-8"))
    assert saved["recipe_names"] == [] and saved["recipe"] is None and saved["prompts"] == PROMPTS_A[:2]
    assert rp.exists()


def test_load_prompts_reads_a_side_of_contrast(tmp_path):
    """gne behave --prompts に対照集合を渡すと A 側の質問を読む(prompts_a、無ければ user_prompts)。質問のリストと {prompts} は前のまま。"""
    from gne.behave import load_prompts
    assert load_prompts(_write_contrast(tmp_path, TSUN)) == PROMPTS_A
    both = _write_contrast(tmp_path, {**TSUN, "name": "both", "user_prompts": PROMPTS_B, "prompts_b": None})
    assert load_prompts(both) == PROMPTS_A
    up = _write_contrast(tmp_path, {"name": "up", "system_a": "x", "user_prompts": PROMPTS_B})
    assert load_prompts(up) == PROMPTS_B
    q = tmp_path / "q.json"
    q.write_text(json.dumps({"prompts": ["a", "b"]}), encoding="utf-8")
    assert load_prompts(q) == ["a", "b"]


# --- ひな形の keyed.direction・keyed.iv(一括・ジョブ) --------------------------------------------------------------

def test_keyed_settings_direction_and_iv(tmp_path):
    """表の keyed に direction(方向の名前)と iv(レシピの名前・パス・ユニットのリスト)を書ける。iv は鍵の作成の引数の既定になり、
    direction は引数にしない(一括が読む)。読めない値は BAD_CONTRAST。"""
    from gne.jobs import _direction_key, _direction_key_experts
    from gne.keyed_pipeline import apply_settings, clean_spec
    base = {"name": "t1", "positions": "before:ふん", "user_prompts": ["a", "b", "c", "d"]}
    _, st = clean_spec({**base, "keyed": {"direction": "tsun", "iv": "chips", "alpha": -1}})
    assert st == {"direction": "tsun", "iv": "chips", "alpha": -1}
    assert clean_spec({**base, "keyed": {"iv": ["10:H2:proj:s3@tsun"]}})[1]["iv"] == ["10:H2:proj:s3@tsun"]
    for bad in ({"direction": "a/b"}, {"direction": 3}, {"iv": ""}, {"iv": [1]}, {"iv": {"x": 1}}):
        with pytest.raises(GneError) as ei:
            clean_spec({**base, "keyed": bad})
        assert ei.value.code == "BAD_CONTRAST", bad
    assert apply_settings({"direction": "tsun", "iv": "chips", "alpha": -1, "layers": "3"}, {"layers": None, "iv": None}) == \
        {"layers": "3", "iv": "chips"}
    assert apply_settings({"iv": "chips"}, {"iv": "other"}) == {"iv": "other"}                 # 引数が優先
    d = tmp_path / "contrasts"
    d.mkdir()
    calls = []
    s = SimpleNamespace(store=SimpleNamespace(contrast_dir=lambda: d), progress=None)
    s.direction_key = lambda c, **kw: calls.append(kw) or {}
    s.direction_key_experts = lambda c, **kw: calls.append(kw) or {}
    spec = {**base, "keyed": {"layers": "2", "direction": "tsun", "iv": "chips", "alpha": -1}}
    _direction_key(s, spec=spec)
    assert calls[-1]["iv"] == "chips" and "direction" not in calls[-1] and "alpha" not in calls[-1]
    _direction_key_experts(s, spec=spec)
    assert calls[-1]["iv"] == "chips"


def test_pipeline_fixed_direction_refuses_build(tmp_path):
    """一括は表の keyed.direction の方向を使う。その方向が無いとき・作り直しを頼まれたときは、この対照集合から作らずに断る
    (モデルを流す前に)。"""
    from gne.keyed_pipeline import run
    d = tmp_path / "contrasts"
    d.mkdir()
    have = {}

    def dset(name):
        if name in have:
            return have[name]
        raise GneError("FILE_NOT_FOUND", f"方向 {name} がありません")
    s = SimpleNamespace(store=SimpleNamespace(contrast_dir=lambda: d), progress=None, direction_set=dset)
    spec = {"name": "fx", "positions": "before:ふん", "user_prompts": ["a", "b", "c", "d"],
            "keyed": {"layers": "2", "direction": "chips_dir", "iv": "chips", "alpha": -1}}
    with pytest.raises(GneError) as ei:
        run(s, spec)
    assert ei.value.code == "BAD_ARGS" and "chips_dir" in ei.value.message and "ありません" in ei.value.message
    have["chips_dir"] = SimpleNamespace(meta={}, path=tmp_path / "x.npz")
    with pytest.raises(GneError) as ei:
        run(s, spec, rebuild_direction=True)
    assert ei.value.code == "BAD_ARGS" and "作り直しません" in ei.value.message


def test_ui_keyed_templates_include_root(world):
    """鍵のタブの「ひな形」: configs/keyed/ のひな形(src configs、order の順)の後に、<root>/keyed/ の JSON(src root、新しい順)。
    読めないファイルは飛ばす。configs/keyed/ だけの keyed_templates は前のまま。"""
    from gne.keyed_from_behave import from_behave
    from gne.ui.data import UIData
    s = world.s
    from_behave(s, "left", recipe="chips", direction="tsun", name="older")
    time.sleep(0.02)
    from_behave(s, "left", recipe="chips", direction="tsun", name="newer")
    (s.store.root / "keyed" / "broken.json").write_text("{", encoding="utf-8")
    xs = UIData(s).keyed_templates_all()
    assert [x["src"] for x in xs] == ["configs"] * 3 + ["root"] * 2
    assert [x["file"] for x in xs][:3] == [x["file"] for x in UIData.keyed_templates()]
    assert [x["file"] for x in xs[3:]] == ["newer.json", "older.json"]
    r = xs[3]
    assert "about" not in r["spec"] and r["spec"]["keyed"]["direction"] == "tsun" and r["order"] == 50
    assert "left" in r["ja"]["title"] and r["en"]["title"]


def test_mcp_key_from_behave_and_root_template(world, monkeypatch):
    """MCP: direction_key_from_behave は Session.direction_key_from_behave に引数を渡す。direction_key_pipeline の contrast に
    <root>/keyed/ の名前を渡すと表として読む。instructions に手順。"""
    import anyio

    import gne.api
    import gne.mcp_server as ms
    from gne.keyed_from_behave import from_behave
    s = world.s
    from_behave(s, "left", recipe="chips", direction="tsun")
    seen, started = [], []

    class FakeRunner:
        def __init__(self, *a, **k):
            pass

        def start(self, kind, params, **k):
            started.append((kind, params))
            return SimpleNamespace(view=lambda: {"id": "j1", "kind": kind})
    fake = SimpleNamespace(warnings=[], store=s.store,
                           direction_key_from_behave=lambda b, **kw: seen.append((b, kw)) or {"name": "x", "path": "p"})
    monkeypatch.setattr(gne.api, "Session", lambda *a, **k: fake)
    monkeypatch.setattr(ms, "JobRunner", FakeRunner)
    srv = ms.build_server("m.gguf")

    async def go():
        tools = {t.name: t for t in await srv.list_tools()}
        assert "direction_key_from_behave" in tools and "min_rate" in tools["direction_key_from_behave"].description
        r = await srv.call_tool("direction_key_from_behave", {"behave": "left", "recipe": "chips", "direction": "tsun",
                                                              "labels": ["P"], "min_rate": 0.5, "strict": True})
        await srv.call_tool("direction_key_pipeline", {"contrast": "left_keyed", "layers": [8]})
        return r
    r = anyio.run(go)
    b, kw = seen[-1]
    assert b == "left" and kw["recipe"] == "chips" and kw["direction"] == "tsun" and kw["labels"] == ["P"]
    assert kw["min_rate"] == 0.5 and kw["strict"] is True and kw["replies"] is True
    assert "x" in json.dumps(r, ensure_ascii=False, default=str)
    k, p = started[-1]
    assert k == "direction_key_pipeline" and p["spec"]["name"] == "left_keyed" and p["spec"]["keyed"]["iv"] == "chips"
    assert "direction_key_from_behave" in ms.INSTRUCTIONS


def test_cli_help_lists_key_from_behave():
    from gne.cli import app
    r = CliRunner().invoke(app, ["direction", "--help"])
    assert r.exit_code == 0 and "key-from-behave" in r.output
    r = CliRunner().invoke(app, ["direction", "key-from-behave", "--help"])
    assert r.exit_code == 0
    for opt in ("--behave", "--recipe", "--direction", "--contrast", "--label", "--match", "--min-rate", "--positions",
                "--name", "--out", "--no-replies", "--prompts", "--strict", "--json"):
        assert opt in r.output, opt
    r = CliRunner().invoke(app, ["direction", "key-from-behave", "--recipe", "chips"])
    assert r.exit_code == 2 and "--behave" in r.output


# --- モデルあり: CLI、作った JSON から鍵、ブラウザ ----------------------------------------------------------------------

def _left_texts():
    """作った JSON から鍵を作るための返事: 8 問とも照れ隠しが 3 つずつ残った返事(冒頭には置かない)。言い回しを少しずつ変える。"""
    tails = ["知らないんだから", "まあいいわ", "勘違いしないで", "たまたまよ", "当然でしょ", "別にいいけど", "仕方ないわね", "今回だけよ"]
    return [f"そう言われても、ふん、{t}。でも、ふん、悪い気はしないわ。ほんとに、ふん。" for t in tails]


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    """Session(Qwen3-0.6B、CPU 可)と、対照集合 tsun(prompts_a / prompts_b)から作った本物の方向 tsun(positions last)、CHiPS の
    レシピ chips(L10 のヘッド 2 の proj)、behave の結果 left(8 問とも照れ隠しが残った返事)、それから作った JSON。"""
    if not MODEL.exists():
        pytest.skip(f"{MODEL} がありません")
    from gne.api import Session
    from gne.keyed_from_behave import from_behave
    d = tmp_path_factory.mktemp("c07fb")
    s = Session(MODEL, root=d / "root", n_gpu_layers=NGL, n_ctx=1024)
    cp = _write_contrast(d, TSUN)
    s.direction(contrast=str(cp), name="tsun", positions="last")
    rp = _chips(s)
    _behave(s, "left", [_left_texts()] * 2, recipe=_rec_dict(rp, "left"))
    fb = from_behave(s, "left", recipe="chips", direction="tsun")
    yield SimpleNamespace(s=s, contrast=cp, recipe=rp, dir=d, fb=fb)
    s.close()


@pytest.mark.model
def test_json_builds_key_with_pipeline(real):
    """作った JSON から direction_key_pipeline で鍵を作る: 方向は keyed.direction の tsun を使い回し(警告なし)、鍵は keyed.iv の
    CHiPS のレシピを当てた状態で、与えた返事(behave の返事)の照れ隠しの直前を書き込みたい位置にする。dont_touch の ordinary は
    レシピを当てて返事を作る。作成と反対側の subset で key-check。alpha_hint は −1。"""
    from gne.schemas import DirectionKeyCheckOut, DirectionKeyOut
    fb = real.fb
    assert fb["path"] and len(fb["extracted"]) == 8 and fb["warnings"] == [], fb["warnings"]
    r = real.s.direction_key_pipeline(fb["path"], layers=[8, 14], pseudo=4, max_tokens=16)
    assert r["warnings"] == [] and r["alpha_hint"] == -1 and r["stages"] == ["鍵", "key-check"]
    assert r["direction"]["name"] == "tsun" and r["direction"]["reused"] and r["direction"]["from_settings"]
    k = r["key"]
    DirectionKeyOut.model_validate({**ENV_STUB, **k})
    assert k["name"] == "left_keyed" and k["mode"] == "same" and k["positions"] == f"before:(?i){MATCH}"
    assert k["iv"]["units"][0]["mode"] == "proj" and k["iv"]["units"][0]["head"] == 2
    assert k["replies"]["AB"] == "given" and k["replies"]["dt:ordinary"] == "generated"
    assert k["matched"] == {"A": 4, "of": 4} and k["counts"]["rows_a"] >= 8 and k["counts"]["dt"] == 3
    assert [x["layer"] for x in k["layers"]] == [8, 14] and not k["dropped"]
    c = r["check"]
    DirectionKeyCheckOut.model_validate({**ENV_STUB, **c})
    assert c["subset"] == "odd" and c["held_out"] is True and c["iv"]["units"][0]["head"] == 2
    assert Path(real.s.store.key_dir(real.s.meta.model_id) / "left_keyed.npz").exists()


@pytest.mark.model
def test_cli_key_from_behave_json(real):
    """CLI(--json): 出力はスキーマを通り、警告はエンベロープに入る。残った質問が無ければ終了コード 1 で、JSON は書かない。"""
    from gne.cli import app
    from gne.schemas import COMMAND_SCHEMAS
    s = real.s
    _behave(s, "plain", [[PLAIN] * 8], recipe=_rec_dict(real.recipe, "plain"))
    common = ["--model", str(MODEL), "--root", str(s.store.root), "--json"]
    r = CliRunner().invoke(app, ["direction", "key-from-behave", "--behave", "left", "--recipe", "chips", "--direction", "tsun",
                                 "--min-rate", "0.5", "--name", "cli_left", *common])
    assert r.exit_code == 0, r.output
    d = json.loads(r.stdout.strip().splitlines()[-1])
    COMMAND_SCHEMAS["direction key-from-behave"].model_validate(d)
    assert d["command"] == "direction key-from-behave" and d["name"] == "cli_left" and len(d["extracted"]) == 8
    assert Path(d["path"]) == s.store.root / "keyed" / "cli_left.json" and d["judge"]["min_rate"] == 0.5
    r = CliRunner().invoke(app, ["direction", "key-from-behave", "--behave", "plain", "--recipe", "chips", "--direction", "tsun",
                                 *common])
    assert r.exit_code == 1
    d = json.loads(r.stdout.strip().splitlines()[-1])
    assert d["path"] is None and any("JSON は書きません" in w for w in d["warnings"])
    r = CliRunner().invoke(app, ["direction", "key-from-behave", "--behave", "left", "--recipe", "chips", "--direction", "tsun",
                                 "--label", "P", *common])
    assert r.exit_code == 1 and json.loads(r.stdout.strip().splitlines()[-1])["error"]["code"] == "BAD_ARGS"
    r = CliRunner().invoke(app, ["direction", "key-from-behave", "--behave", "left", "--recipe", "chips", "--direction", "tsun",
                                 "--name", "cli_h", "--model", str(MODEL), "--root", str(s.store.root)])
    assert r.exit_code == 0 and "残った質問: 8 問" in r.output and "次: gne direction key --contrast" in r.output


def _free_port() -> int:
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


@pytest.fixture(scope="module")
def ui_server(real):
    """ブラウザのテスト用のサーバ(上の Session と <root>/keyed/left_keyed.json)。"""
    if not CHROME.exists():
        pytest.skip(f"Chromium がありません({CHROME}、GNE_CHROME で指定)")
    import uvicorn

    from gne.ui.server import create_app
    app = create_app(real.s, warm=False)
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
    live = app.state.live
    for j in live.jobs.list():
        if j.status in ("queued", "running"):
            live.jobs.cancel(j.id)


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


def _key_tab(browser, url, lang):
    pg = browser.new_page(viewport={"width": 1600, "height": 1100}, locale="ja-JP")
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("dialog", lambda dl: dl.accept())
    pg.goto(url + f"/?lang={lang}")
    pg.wait_for_function("window.gne && window.gne.recipe && window.gne.arch", timeout=120000)
    pg.click("#views button[data-view=direction]")
    pg.wait_for_function("window.gne.directionView && window.gne.directionView.mode", timeout=60000)
    pg.click("#dv-mode button[data-m=key]")
    pg.wait_for_function("window.gne.directionView.key.loaded && document.querySelector(\"#kx-tpl-sel option[value='root/left_keyed.json']\")",
                         timeout=30000)
    return pg, errors


@pytest.mark.model
def test_root_template_in_browser(ui_server, browser):
    """鍵のタブの「ひな形」に <root>/keyed/ の JSON が「自分で作ったひな形」の組で出て、選ぶと入力欄に入る。説明の頭に置き場所、
    keyed.iv が介入の欄に(細かい設定を開いて)、要約に keyed.direction の方向、方向の欄に「使い回す」。英語の画面でも文言がそろう。"""
    pg, errors = _key_tab(browser, ui_server, "ja")
    grp = pg.evaluate("document.querySelector(\"#kx-tpl-sel option[value='root/left_keyed.json']\").parentElement.label")
    assert grp == "自分で作ったひな形(<root>/keyed/)"
    assert pg.evaluate("document.querySelector(\"#kx-tpl-sel option[value='sentence_endings.json']\").parentElement.tagName") == "SELECT"
    pg.select_option("#kx-tpl-sel", "root/left_keyed.json")
    text = pg.input_value("#kx-text")
    spec = json.loads(text)
    assert spec["name"] == "left_keyed" and spec["keyed"] == {"direction": "tsun", "alpha": -1, "iv": "chips"} and "about" not in spec
    assert len(spec["user_prompts"]) == 8 and spec["dont_touch"][0]["name"] == "ordinary"
    about = pg.evaluate("document.querySelector('#kx-tpl-about').textContent")
    assert about.startswith("自分で作ったひな形(<root>/keyed/left_keyed.json)") and "gne behave の結果 left" in about
    assert pg.evaluate("document.querySelector('#kx-tpl-about').title") == "<root>/keyed/left_keyed.json"
    assert pg.input_value("#kx-iv") == "chips" and pg.evaluate("document.querySelector('.kx-adv').open")
    jc = pg.evaluate("document.querySelector('#kx-jcheck').textContent")
    assert "keyed.direction の tsun を使い回す" in jc and "質問 8 個" in jc and "ordinary" in jc
    assert pg.evaluate("document.querySelectorAll('#kx-jcheck .err').length") == 0
    assert "system_a が無いので" not in jc
    assert "keyed.direction の tsun を使い回します" in pg.evaluate("document.querySelector('#kx-dirinfo').textContent")
    assert pg.evaluate("document.querySelector('#kx-name').placeholder") == "left_keyed"
    assert not errors, errors
    pg.close()
    pg, errors = _key_tab(browser, ui_server, "en")
    assert pg.evaluate("document.querySelector(\"#kx-tpl-sel option[value='root/left_keyed.json']\").parentElement.label") == \
        "your own templates (<root>/keyed/)"
    pg.select_option("#kx-tpl-sel", "root/left_keyed.json")
    assert pg.evaluate("document.querySelector('#kx-tpl-about').textContent").startswith("Your own template (<root>/keyed/left_keyed.json)")
    assert "reuse tsun from keyed.direction" in pg.evaluate("document.querySelector('#kx-jcheck').textContent")
    missing = [m for m in pg.evaluate("[...window.gneI18n.missing]") if "keyed" in m or "ひな形" in m]
    assert missing == [], missing
    assert not errors, errors
    pg.close()
