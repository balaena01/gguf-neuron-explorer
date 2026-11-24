"""候補を 1 個ずつレシピに足す(v0.10.1)。

モデル不要: gne/steer.py の add_units(同じユニットの判定・--replace・既定の方向・別の方向は別のユニット)、gne recipe-add、
Session.unit_check(レシピの欄の「ユニットを足す」が返す理由の code)を軽いセッションの代役で。
モデルあり(Qwen3-0.6B、CPU): /api/recipe-unit と /api/pipelines/<名前>/writes、ブラウザ(Playwright の Chromium、無ければ飛ばす)で
CHiPS の結果の候補の表(確かめの表・順位の表・止めて選ぶ流れの表と η の表)の「レシピに足す」と「レシピにある」の印、組の
「足りない部品を足す」、別の一括の方向の部品と残りから追加の結果の部品が自分の方向を持つこと、レシピの欄の「ユニットを足す」
(書き方の誤り・範囲外・重複の文)、日本語と英語(未翻訳なし)。
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from gne.steer import ADD_NOTE, GneError, Recipe, add_units, parse_unit, unit_identity

from .test_ui import (MODEL, _dialogs, _seed_pipeline, _seed_pipeline_writes, _seed_pipeline_writes_081,  # noqa: F401
                      _seed_pipeline_writes_refine, _serve, _step, browser)


# --- モデル不要 ------------------------------------------------------------------------------------------------------------

def _recipe(**kw):
    us = [parse_unit("12:H5:s2"), parse_unit("14:N100:s1@other")]
    return Recipe(us, "m", name="r", default_direction="pir", **kw)


def test_add_units_identity_and_replace():
    r = _recipe()
    r2, done = add_units(r, [parse_unit("14:H19:s4"), parse_unit("9:attn:s1@x2"), parse_unit("17:E25:N389:s4@ink")])
    assert [d["action"] for d in done] == ["added"] * 3 and len(r2.units) == 5 and len(r.units) == 2     # 元のレシピは変えない
    assert r2.units[2].direction is None and r2.default_direction == "pir"                                  # 省いた方向は省いたまま(既定)
    assert unit_identity(r2.units[0], "pir") == unit_identity(parse_unit("12:H5:s9@pir"), "pir")           # 強さは見ない、既定で埋める
    with pytest.raises(GneError) as e:                                                                      # 同じ部品・同じ方向
        add_units(r2, [parse_unit("12:H5:s3@pir")])
    assert "もうレシピにあります" in e.value.message and "--replace" in e.value.hint
    r3, done = add_units(r2, [parse_unit("12:H5:s3@pir")], replace=True)
    assert done == [{"label": "L12 H5 s3 @pir", "action": "replaced", "was": "L12 H5 s2"}] and r3.units[0].k == 3.0
    assert [u.label() for u in r3.units][1:] == [u.label() for u in r2.units][1:]                          # 並びはそのまま
    r4, done = add_units(r2, [parse_unit("12:H5:s3@other")])                                               # 別の方向は別のユニット
    assert done[0]["action"] == "added" and len(r4.units) == 6
    with pytest.raises(GneError):                                                                           # 同じ呼び出しの中の重複も止める
        add_units(r, [parse_unit("1:H1:s1"), parse_unit("1:H1:s2")])
    with pytest.raises(GneError) as e:                                                                      # 既定の方向が無い
        add_units(Recipe([parse_unit("1:H1:s1@a")], "m", name="x"), [parse_unit("2:H1:s1")])
    assert "方向がありません" in e.value.message
    with pytest.raises(GneError) as e:                                                                      # keyed は鍵も要る
        add_units(r, [parse_unit("10:ffn:keyed:a1@pir")])
    assert "鍵がありません" in e.value.message
    r5, _ = add_units(r, [parse_unit("14:8949:+1"), parse_unit("21:H7:x0")])                              # 射影以外の書き方も足せる
    assert [u.mode for u in r5.units[2:]] == ["add", "gain"]


def test_cli_recipe_add(tmp_path):
    from gne.cli import app
    src = tmp_path / "r.yaml"
    _recipe().save(src)
    run = CliRunner().invoke
    out = tmp_path / "r2.yaml"
    res = run(app, ["recipe-add", "--recipe", str(src), "--unit", "14:H19:s4", "--unit", "17:E25:N389:s4@ink", "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert "足した: L14 H19 s4" in res.output and "足した: L17 E25 N389 s4 @ink" in res.output and ADD_NOTE in res.output
    r2 = Recipe.load(out)
    assert r2.name == "r2" and len(r2.units) == 4 and r2.units[3].kind == "expert_neuron" and r2.units[3].expert == 25
    assert len(Recipe.load(src).units) == 2                                                                 # --out なら元は変えない
    res = run(app, ["recipe-add", "--recipe", str(out), "--unit", "12:H5:s3"])
    assert res.exit_code == 2 and "もうレシピにあります" in res.output                                      # 重複は止める
    res = run(app, ["recipe-add", "--recipe", str(out), "--unit", "12:H5:s3", "--replace"])
    assert res.exit_code == 0 and "強さを置き換えた: L12 H5 s3" in res.output
    assert Recipe.load(out).units[0].k == 3.0                                                               # --out が無ければ書き換える
    res = run(app, ["recipe-add", "--recipe", str(out), "--unit", "12:Hx:s3"])
    assert res.exit_code == 2                                                                               # 読めない書き方
    old = tmp_path / "old.yaml"                                                                             # 補正の記録の無い古いレシピ
    old.write_text("name: old\nmodel_id: m\ndose_unit: rel\nunits:\n- {layer: 3, kind: head, head: 1, mode: proj, s: 1, direction: a}\n")
    res = run(app, ["recipe-add", "--recipe", str(old), "--unit", "4:H2:s1@a"])
    assert res.exit_code == 0 and "proj_correction: postnorm" in res.output


def _fake_session(*, ink=False, moe_layers=()):
    """Session.unit_check を呼べる軽い代役(層 28、ヘッド 16、ニューロン 3072、方向 pir だけ)。"""
    from gne.api import Session
    li = SimpleNamespace(n_head=16, n_expert=8 if ink else 0, n_ff=512 if ink else 3072)
    plan = SimpleNamespace(n_stored=24) if ink else None
    w = SimpleNamespace(inkling=plan, n_main=28, arch=SimpleNamespace(layers=[li] * 28), is_moe=lambda L: L in moe_layers)

    def direction_set(name):
        if name != "pir":
            raise GneError("FILE_NOT_FOUND", f"方向 {name} がありません")
        return object()

    def recipe(*, units, default_direction=None, default_key=None):
        r = Recipe([parse_unit(u) for u in units], "m", default_direction=default_direction, default_key=default_key)
        r.validate()
        return r
    me = SimpleNamespace(weights=w, meta=SimpleNamespace(n_ff_at=lambda L: 3072), direction_set=direction_set, recipe=recipe)
    return lambda spec, **kw: Session.unit_check(me, spec, **kw)


def test_unit_check_codes_without_model():
    chk = _fake_session(moe_layers=(5,))
    ok = chk("12:H5:s2@pir")
    assert ok["ok"] and ok["unit"]["kind"] == "head" and ok["unit"]["direction"] == "pir" and ok["label"] == "L12 H5 s2 @pir"
    ok = chk("9:attn:s1", default_direction="pir")                                     # @ を省くと既定の方向で埋める
    assert ok["ok"] and ok["unit"]["direction"] == "pir"
    code = lambda spec, **kw: chk(spec, **kw)["code"]                                    # noqa: E731
    assert code("garbage") == "grammar" and code("12:H5:s1") == "no_direction" and code("40:H1:s1@pir") == "layer_range"
    assert chk("40:H1:s1@pir")["params"] == {"layer": 40, "max": 27}
    assert code("12:H16:s1@pir") == "head_range" and code("12:N3072:s1@pir") == "neuron_range"
    assert code("12:H1:s1@nope") == "no_such_direction" and code("12:E1:N2:s1@pir") == "not_moe"
    assert code("5:N7:s1@pir") == "moe_neuron" and code("10:ffn:keyed:a1@pir") == "no_key"
    ink = _fake_session(ink=True)
    assert ink("17:E7:N511:s4@pir")["ok"] and ink("17:E8:N1:s4@pir")["code"] == "expert_range"
    assert ink("17:E1:N512:s4@pir")["code"] == "neuron_range" and ink("24:H1:s1@pir")["code"] == "layer_range"
    assert ink("14:8949:+1")["code"] == "inkling_proj_only"


# --- モデルあり ------------------------------------------------------------------------------------------------------------

def _direction(s, name):
    m = s.meta
    rng = np.random.default_rng(len(name))
    Ls = list(range(m.n_layer))
    np.savez(s.store.direction_dir(m.model_id) / f"{name}.npz", layers=np.array(Ls, np.int32),
             **{f"layer_{L}": rng.standard_normal(m.n_embd).astype(np.float32) for L in Ls},
             **{f"scale_{L}": np.ones(m.n_embd, np.float32) for L in Ls}, method="mean_diff", positions="last", contrast_sha="x",
             source="test", n_rows_A=6, n_rows_B=6)


def _writes_json(s, name, rows):
    p = s.store.heads_dir(s.meta.model_id) / f"{name}.writes.json"
    p.write_text(json.dumps({"rows": rows, "verify": {"rows": [dict(rows[0], gap_drop=2.6, effect=2.4, excluded=False)]},
                             "n_rows_total": 5000, "name": name}))


@pytest.fixture(scope="module")
def add_server(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip("モデルがありません")
    from gne.api import Session
    s = Session(MODEL, root=tmp_path_factory.mktemp("ui_add"), n_ctx=2048, n_seq_max=4, n_gpu_layers=0)
    _seed_pipeline_writes(s)                                  # ppw(書き込みの量の流れ)
    _seed_pipeline_writes_081(s)
    _seed_pipeline_writes_refine(s)                           # ppw81_r1_r1(残りから追加を 2 回)
    _seed_pipeline(s)                                         # ppx(止めて選ぶ流れ)
    for name in ("ppw", "ppx", "ppw81", "ppw81_r1", "ppw81_r1_r1"):
        _direction(s, name)
    row = lambda lab, kind, L, i, sc: {"label": lab, "layer": L, "index": i, "kind": kind, "score": sc, "share": sc / 40}  # noqa: E731
    _writes_json(s, "ppw", [row("12:H5", "attn", 12, 5, 3.0), row("14:N100", "neuron", 14, 100, 2.0), row("12:H3", "attn", 12, 3, 1.0),
                            row("13:N9", "neuron", 13, 9, 0.8), row("15:attn", "layer_attn", 15, -1, 0.5),
                            dict(row("16:E25:N389", "neuron", 16, 389, 0.4), expert=25)])   # 専門家のニューロンの行(Inkling 2 の形)
    yield from _serve(s)


@pytest.mark.model
def test_recipe_unit_and_writes_api(add_server):
    import urllib.error
    import urllib.request

    def call(path, body=None):
        req = urllib.request.Request(add_server + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, None
    post = lambda spec, **kw: call("/api/recipe-unit", {"spec": spec, **kw})[1]   # noqa: E731
    x = post("12:H5:s2@ppw")
    assert x["ok"] and x["unit"]["mode"] == "proj" and x["unit"]["head"] == 5 and x["unit"]["direction"] == "ppw"
    assert post("12:N345:s2", default_direction="ppx")["unit"]["direction"] == "ppx"
    assert post("9:attn:s1@ppw")["ok"] and post("9:ffn:s1@ppw")["ok"] and post("14:300:+1")["ok"]
    assert [post(x_)["code"] for x_ in ("12:H99:s1@ppw", "99:H1:s1@ppw", "12:H5:s1@nope", "12:H5:s1", "garbage", "12:E1:N2:s1@ppw",
                                        "12:N99999:s1@ppw")] == ["head_range", "layer_range", "no_such_direction", "no_direction",
                                                                  "grammar", "not_moe", "neuron_range"]
    w = call("/api/pipelines/ppw/writes")[1]
    assert w["n_rows_total"] == 5000 and [r["rank"] for r in w["rows"]] == [1, 2, 3, 4, 5, 6] and w["rows"][5]["expert"] == 25
    assert w["rows"][0]["verified"]["gap_drop"] == 2.6 and "verified" not in w["rows"][1]
    assert call("/api/pipelines/ppx/writes")[1]["rows"] == []                           # 止めて選ぶ流れには順位が無い
    assert call("/api/pipelines/bad name/writes".replace(" ", "%20"))[0] in (400, 404)


def _open(pg, url, name, lang):
    pg.goto(url + f"/?lang={lang}")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    pg.evaluate("window.gne.recipe.clear(); try { localStorage.removeItem('gne.pipeline.draft') } catch {}")
    pg.click("#views button[data-view=chips]")
    pg.wait_for_function("window.gne.directionView && window.gne.directionView.mode === 'pipe'", timeout=60000)
    _step(pg, "chips", "past")
    pg.wait_for_selector(f"#pp-past tr[data-pp={name}]", timeout=30000)
    pg.click(f"#pp-past tr[data-pp={name}]")
    pg.wait_for_function(f"window.gne.directionView.pipe.data && window.gne.directionView.pipe.data.name === '{name}'", timeout=30000)


def _btn(pg, table, label):
    """表の行(部品の名前の文字を含む行)の「レシピに足す」のボタンの文字。"""
    return pg.evaluate("""([sel, label]) => { const tr = [...document.querySelectorAll(`${sel} tbody tr`)].find((x) => x.querySelector("td.l") && x.querySelector("td.l").innerText.trim().startsWith(label));
      const b = tr && tr.querySelector("[data-add]"); return b ? b.textContent : null; }""", [table, label])


def _click(pg, table, label):
    pg.evaluate("""([sel, label]) => { const tr = [...document.querySelectorAll(`${sel} tbody tr`)].find((x) => x.querySelector("td.l") && x.querySelector("td.l").innerText.trim().startsWith(label));
      tr.querySelector("[data-add]").click(); }""", [table, label])


@pytest.mark.model
def test_add_candidates_in_browser(add_server, browser):
    """書き込みの量の流れ(ppw): 確かめの表の行から足す(強さは含むいちばん小さい組の s、無ければ 1)、「レシピにある」の印と外す、
    レシピの欄で外すと印が戻る、組の「足りない部品を足す」(ある部品の強さは変えない)、順位の表(開くと読む)から足す。"""
    url = add_server
    for lang in ("ja", "en"):
        pg = browser.new_page(viewport={"width": 1600, "height": 1000})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        _dialogs(pg)
        _open(pg, url, "ppw", lang)
        _step(pg, "chips", "alloc")
        top = "#pp-result table.pp-top"
        pg.wait_for_selector(f"{top} [data-add]", timeout=30000)
        add_t = "Add to recipe" if lang == "en" else "レシピに足す"
        assert _btn(pg, top, "L12 H5") == add_t
        _click(pg, top, "L14 N100")                                   # 組 2(s1)と組 4(s2)にある → 小さい組の s1
        _click(pg, top, "L12 H3")                                     # 組 4 だけ → s2
        assert pg.evaluate("window.gne.recipe.specs()") == ["12:H3:s2@ppw", "14:N100:s1@ppw"]
        assert _btn(pg, top, "L12 H3").endswith("In recipe" if lang == "en" else "レシピにある")
        msg = pg.evaluate("document.querySelector('#pp-add-msg').innerText")
        assert ("s2" in msg and "ppw" in msg) and ("measured each unit alone" in msg if lang == "en" else "その部品だけを当てて測った" in msg)
        _click(pg, top, "L12 H3")                                     # もう一度押すと外す
        assert pg.evaluate("window.gne.recipe.specs()") == ["14:N100:s1@ppw"]
        assert _btn(pg, top, "L12 H3") == add_t
        pg.evaluate("window.gne.recipe.remove(0)")                    # ほかの所(レシピの欄)で外しても印が戻る
        pg.wait_for_function(f"(() => {{ const b = [...document.querySelectorAll('{top} [data-add]')][1]; return b && !b.classList.contains('on'); }})()")
        pg.evaluate("window.gne.recipe.addUnit({ layer: 12, kind: 'head', mode: 'proj', head: 5, s: 7, direction: 'ppw' })")
        pg.click("#pp-result table.pp-cmp [data-addset='4']")         # 組 4 のうち足りない 3 個だけ。12:H5 は s7 のまま
        assert pg.evaluate("window.gne.recipe.specs()") == ["12:H3:s2@ppw", "12:H5:s7@ppw", "13:N7:s2@ppw", "14:N100:s2@ppw"]
        assert ("Added 3 of this set's 4 units (1 were already" if lang == "en" else "4 個のうち 3 個を足しました(1 個は") in pg.evaluate("document.querySelector('#pp-add-msg').innerText")
        pg.evaluate("window.gne.recipe.clear()")
        _step(pg, "chips", "causal")                                  # 3. 順位の表(開くと読む)
        pg.click("#pp-result details.pp-rankall summary")
        rk = "#pp-result table.pp-rank"
        pg.wait_for_selector(f"{rk} [data-add]", timeout=30000)
        assert pg.evaluate(f"document.querySelectorAll('{rk} tbody tr').length") == 6
        _click(pg, rk, "L15 attn")                                    # どの組にも無い → s1
        _click(pg, rk, "L13 N9")
        pg.evaluate(f"[...document.querySelectorAll('{rk} [data-add]')][5].click()")   # 専門家のニューロンの行 → L:Ee:Nn
        assert pg.evaluate("window.gne.recipe.specs()") == ["13:N9:s1@ppw", "15:attn:s1@ppw", "16:E25:N389:s1@ppw"]
        if lang == "en":
            assert pg.evaluate("[...window.gneI18n.missing]") == []
        assert not errors, errors
        pg.close()


@pytest.mark.model
def test_add_keeps_each_runs_direction(add_server, browser):
    """別の一括の部品は自分の方向で入り、パネルの行に方向が出る。同じ部品が別の方向で入っていれば「別の方向あり」と出て、足すと
    2 行目になる。止めて選ぶ流れ(ppx)の因果の表と η の表からも足せる。残りから追加の結果(ppw81_r1_r1)の部品はその結果の方向。"""
    url = add_server
    pg = browser.new_page(viewport={"width": 1600, "height": 1000})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    _dialogs(pg)
    _open(pg, url, "ppx", "ja")
    pg.evaluate("window.gne.recipe.loadProj('base', ['12:H3:s1@ppw', '20:H1:s1@ppw'], 'ppw')")
    _step(pg, "chips", "causal")
    top = "#pp-result table.pp-top"
    pg.wait_for_selector(f"{top} [data-add]", timeout=30000)
    assert _btn(pg, top, "L12 H3") == "足す(別の方向あり)"
    _click(pg, top, "L12 H3")                                         # 組 2 に s7 で入っている
    assert pg.evaluate("window.gne.recipe.specs()") == ["12:H3:s1@ppw", "12:H3:s7@ppx", "20:H1:s1@ppw"]
    dirs = pg.evaluate("[...document.querySelectorAll('#recipe-list tr.rc-proj select.rc-pdir')].map((s) => s.value)")
    assert dirs == ["ppw", "ppx", "ppw"]
    _step(pg, "chips", "alloc")
    eta = "#pp-result table.pp-eta"
    pg.wait_for_selector(f"{eta} [data-add]", timeout=30000)
    _click(pg, eta, "L14 H0")                                         # 組 3 に s2 で入っている
    assert "14:H0:s2@ppx" in pg.evaluate("window.gne.recipe.specs()")
    _open(pg, url, "ppw81_r1_r1", "ja")                               # 残りから追加: 足す部品の方向はその結果の方向
    _step(pg, "chips", "alloc")
    pg.wait_for_selector(f"{top} [data-add]", timeout=30000)
    _click(pg, top, "L12 H5")
    assert any(x.endswith("@ppw81_r1_r1") and x.startswith("12:H5:") for x in pg.evaluate("window.gne.recipe.specs()"))
    pg.evaluate("window.gne.recipe.clear()")
    pg.click("#pp-result table.pp-cmp [data-addset='3']")             # 合わせたレシピ(土台 + 足した組)。部品ごとに自分の方向
    sp = pg.evaluate("window.gne.recipe.specs()")
    assert any("@ppw81_r1@" not in x and x.endswith("@ppw81_r1") for x in sp) and any(x.endswith("@ppw81") for x in sp)
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_recipe_panel_unit_input(add_server, browser):
    """レシピの欄の「ユニットを足す」: 書き方を読んでサーバーで確かめ、足す。@ を省くと行の方向(1 つだけのとき)。重複・範囲外・
    読めない書き方は理由を出して足さない。英語の画面の文も辞書にある。"""
    url = add_server
    for lang in ("ja", "en"):
        pg = browser.new_page(viewport={"width": 1600, "height": 1000})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        _dialogs(pg)
        pg.goto(url + f"/?lang={lang}")
        pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
        pg.evaluate("window.gne.recipe.clear()")
        if pg.evaluate("document.querySelector('#recipe').classList.contains('collapsed')"):
            pg.click("#recipe .toggle")
        pg.click("#recipe-tabs button[data-rtab=units]")

        def add(spec):
            pg.fill("#rc-ua", spec)
            pg.press("#rc-ua", "Enter")
            pg.wait_for_function("(() => { const m = document.querySelector('#rc-ua-msg'); return m && m.innerText.trim(); })()", timeout=30000)
            out = pg.evaluate("document.querySelector('#rc-ua-msg').innerText")
            pg.evaluate("document.querySelector('#rc-ua-msg').innerHTML = ''")
            return out
        m = add("12:H5:s2@ppw")
        assert pg.evaluate("window.gne.recipe.specs()") == ["12:H5:s2@ppw"] and "L12 H5 s2 @ppw" in m
        m = add("14:N100:s1.5")                                        # @ を省く → 行の方向 ppw
        assert pg.evaluate("window.gne.recipe.specs()") == ["12:H5:s2@ppw", "14:N100:s1.5@ppw"] and "ppw" in m
        m = add("12:H5:s3@ppw")
        assert ("already in the recipe" if lang == "en" else "もうレシピにあります") in m
        assert ("has no head 99" if lang == "en" else "ヘッド 99 はありません") in add("12:H99:s1@ppw")
        assert ("Cannot read the unit" if lang == "en" else "書き方を読めません") in add("12:Hx")
        assert ("There is no direction nope" if lang == "en" else "方向 nope がありません") in add("12:H1:s1@nope")
        m = add("9:ffn:s1@ppx")
        assert "9:ffn:s1@ppx" in pg.evaluate("window.gne.recipe.specs()")
        assert len(pg.evaluate("window.gne.recipe.specs()")) == 3
        if lang == "en":
            assert pg.evaluate("[...window.gneI18n.missing]") == []
        assert not errors, errors
        pg.close()
