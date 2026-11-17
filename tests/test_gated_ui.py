"""Web UI: nw のユニットの新しい書き方(!k4・!kauto・!rm・!rmbase・!take<鍵>)のレシピのパネルと、FiSH の画面の手順
「CHiPS のヘッドの分を場面で取り除く」(ジョブ fish_heads、gne nw heads)のテスト。

モデルなし: 書き方を読む・書く(JS の parseNw・nwSpec と Python の parse_unit・spec_of が同じ文字に戻る)、レシピの行の読み込みと
強さの変更で後ろの項目が消えないこと、ニューロンの欄の読み方(ブラウザは Playwright の Chromium、無ければ飛ばす)、ジョブの
引数の確かめ。
モデルあり(Qwen3-0.6B、GNE_TEST_MODEL。CPU で回すときは GNE_TEST_NGL=0): ジョブ fish_heads の結果(層ごとの R²・行の数、
CHiPS との一致、ユニット、レシピの保存先)とエラー、保存したレシピの一覧(/api/recipes)の項目。ブラウザ: レシピのパネルの nw の行の
表示(取り除く形・量の鍵・倍率)、強さの変更・保存・読み直しで同じ書き方に戻ること、FiSH の新しい手順の入力・実行・結果の表・
「レシピに足す」・上書きの確かめ、英語の画面の文言。GNE_TEST_SHOTS を渡すと、画面の写し gated-*.png を置く。
"""

import json
import os
import socket
import threading
import time
import urllib.parse
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
NGL = int(os.environ.get("GNE_TEST_NGL", "-1"))
CHROME = Path(os.environ.get("GNE_CHROME", Path.home() / ".cache/ms-playwright/chromium-1223/chrome-linux64/chrome"))
SHOTS = os.environ.get("GNE_TEST_SHOTS")          # 画面の写しを置くフォルダ(任意)
STATIC = ROOT / "gne" / "ui" / "static"

CONTRAST = {
    "name": "gated_ui",
    "system_a": "You are a pirate. Always answer in rough pirate speech.",
    "user_prompts": ["Thank you for your help yesterday.", "I really like your drawing.", "Thanks for lending me your notebook.",
                     "Your cooking was delicious.", "Thank you for listening to me.", "You look great today."],
    "dont_touch": [{"name": "tasks", "prompts": ["What is 12 times 8?", "What is the capital of France?",
                                                 "How many days are in a week?", "What color is the sky?"]}],
}

# 書き方の例(後ろの項目の順は !s → !cap → !k → !rm / !rmbase → !take)
SPECS = [
    "8:N1+N2:nw:a1@hg_dir#g_scene&g_one!kauto!takehg_amount",
    "11:N20+N30:nw:a0.5@dir#sc&one!cap0.3-0.7!kauto!rmbase",
    "11:N21+N31:nw:a1@dir#sc&one!k4!rm",
    "12:N5+N6:nw:a0.8@d.seg0#k&one!s12!cap0.4-0.8!k2.5!takeamt.x-1",
    "8:N7:nw:a1@d#k",
    "8:N8:nw:a-0.5@d#k&p0",
    "9:N3+N4:nw:a1.25@d.seg0#b_p0&b_one!cap0.3-0.7",
]


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


def _shot(pg, name, full=True, selector=None):
    if SHOTS:
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        if selector:
            pg.locator(selector).screenshot(path=str(Path(SHOTS) / name))
        else:
            pg.screenshot(path=str(Path(SHOTS) / name), full_page=full)


# --- モデルなし ---------------------------------------------------------------------------------------------

def test_spec_roundtrip_python():
    from gne.nwrite import spec_of
    from gne.steer import parse_unit
    for x in SPECS:
        assert spec_of(parse_unit(x)) == x, x
    assert spec_of(parse_unit("8:N7:nw:a1@-d#k")) == "8:N7:nw:a-1@d#k"            # @-方向 は強さの符号に畳む


def test_neuron_pairs_for_the_job():
    from gne.jobs import _neuron_pairs
    from gne.steer import GneError
    assert _neuron_pairs(None) is None and _neuron_pairs("") is None and _neuron_pairs({}) is None
    assert _neuron_pairs("8:101,102 12:201,202") == {8: [101, 102], 12: [201, 202]}
    assert _neuron_pairs("8:1,2;12:3,4") == {8: [1, 2], 12: [3, 4]} == _neuron_pairs({"8": [1, 2], "12": ["3", "4"]})
    for bad in ("8:1", "8:1,1", "8-1,2", "x:1,2", "8:1,2,3", {8: [1]}):
        with pytest.raises(GneError):
            _neuron_pairs(bad)


def _static_page(browser, lang="ja"):
    """モデルを使わずに静的な JS を読むページ(http://gne.test/ に gne/ui/static を出す)。"""
    pg = browser.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))

    def handle(route, req):
        path = urllib.parse.urlparse(req.url).path.lstrip("/")
        if path in ("", "blank.html"):
            route.fulfill(status=200, content_type="text/html; charset=utf-8",
                          body="<!doctype html><html><head><meta charset=utf-8><title>t</title></head><body></body></html>")
            return
        f = STATIC / path
        if f.is_file() and f.resolve().is_relative_to(STATIC.resolve()):
            route.fulfill(status=200, content_type="text/javascript; charset=utf-8", body=f.read_text(encoding="utf-8"))
        else:
            route.fulfill(status=404, content_type="application/json", body="[]")
    pg.route("http://gne.test/**", handle)
    pg.goto(f"http://gne.test/blank.html?lang={lang}")
    return pg, errors


def test_spec_roundtrip_js(browser):
    """JS の parseNw → nwSpec が同じ文字に戻る(Python の spec_of と同じ)。レシピの行の読み込み・強さの変更・全体の倍率でも後ろの項目が残る。"""
    pg, errors = _static_page(browser)
    got = pg.evaluate("""(specs) => import('/recipe.js').then((m) => {
      const rows = specs.map((x) => m.parseNw(x));
      const back = rows.map((u) => m.nwSpec(u));
      const r = new m.Recipe('gated_t', () => {});
      // サーバの一覧(/api/recipes)の形: 強さは k、kscale は文字
      r.load('saved', rows.map((u) => ({ ...u, alpha: undefined, k: u.alpha })), null, null);
      const loaded = r.specs(1, false);
      r.set(0, 0.6);
      const after = r.specs(1, false), half = r.specs(0.5);
      const draft = JSON.parse(localStorage.getItem('gne.recipe.gated_t')).units.map((u) => m.nwSpec(u));    // 下書きにも残る
      return { rows, back, loaded, after, half, draft, neg: m.parseNw('8:N7:nw:a1@-d#k'), bad: m.parseNw('8:N7+N9:nw:a1@d#k&one!rm!k4') };
    })""", SPECS)
    assert got["back"] == SPECS
    take = got["rows"][0]
    assert (take["kscale"], take["remove"], take["take"], take["pair"], take["key2"]) == ("auto", None, "hg_amount", 2, "g_one")
    assert (got["rows"][1]["remove"], got["rows"][2]["remove"], got["rows"][2]["kscale"]) == ("excess", "part", "4")
    assert (got["rows"][3]["s"], got["rows"][3]["cap"], got["rows"][3]["kscale"], got["rows"][3]["take"]) == (12, "0.4-0.8", "2.5", "amt.x-1")
    assert got["rows"][4]["kscale"] is None and got["rows"][4]["remove"] is None and got["rows"][4]["take"] is None
    assert got["neg"]["alpha"] == -1 and got["neg"]["direction"] == "d" and got["bad"] is None      # 書き方の順が違えば読まない
    assert sorted(got["loaded"]) == sorted(SPECS) and sorted(got["draft"]) == sorted(got["after"])
    assert "8:N1+N2:nw:a0.6@hg_dir#g_scene&g_one!kauto!takehg_amount" in got["after"]
    assert "11:N20+N30:nw:a0.25@dir#sc&one!cap0.3-0.7!kauto!rmbase" in got["half"]    # 全体の倍率(3 列チャット)でも後ろの項目は残る
    assert not errors, errors
    pg.close()


def test_neuron_field_js(browser):
    pg, errors = _static_page(browser)
    got = pg.evaluate("""() => import('/fish.js').then((m) => [m.parseNeuronPairs('', [8, 12]), m.parseNeuronPairs(' 8 : 101 , 102  12:201,202 ', [8, 12]),
      m.parseNeuronPairs('8:1,2', [8, 12]).error, m.parseNeuronPairs('8:1,1 12:3,4', [8, 12]).error, m.parseNeuronPairs('8:1 12:3,4', [8, 12]).error,
      m.parseNeuronPairs('8:1,2 12:3,4 14:5,6', [8, 12]).error])""")
    assert got[0] == {"neurons": None} and got[1] == {"neurons": {"8": [101, 102], "12": [201, 202]}}
    assert "足りない層 12" in got[2] and "違う番号" in got[3] and "層:番号,番号" in got[4] and "層 14 は" in got[5]
    assert not errors, errors
    pg.close()


# --- モデルあり ---------------------------------------------------------------------------------------------

def _run_job(c, kind, params, timeout=1800, ok=True):
    j = c.post("/api/jobs", json={"kind": kind, "params": params})
    assert j.status_code == 200, j.text
    j = j.json()
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = next(x for x in c.get("/api/jobs").json() if x["id"] == j["id"])
        if v["status"] in ("done", "error", "cancelled"):
            if ok:
                assert v["status"] == "done", v["error"]
            return v
        time.sleep(0.2)
    raise AssertionError(f"{kind} が終わりません")


def _close_app(app, s):
    live = app.state.live
    for j in live.jobs.list():
        if j.status in ("queued", "running"):
            live.jobs.cancel(j.id)
    closed = threading.Event()

    def close():
        with live.arbiter.interactive():
            s.close()
        closed.set()
    threading.Thread(target=close, daemon=True).start()
    closed.wait(300)


def _free_port() -> int:
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """Session と、TestClient・ブラウザ用のサーバ(同じアプリ)。先に方向 gd、場面の鍵 g_scene・ほぼ 1 を返す鍵 g_one(層 8・12)と
    層 8 だけの g_one8、CHiPS のレシピ chips_t(L8 H3・L12 H5)と proj の無いレシピ plain_t を作っておく。"""
    if not MODEL.exists():
        pytest.skip(f"{MODEL} がありません")
    import uvicorn
    from fastapi.testclient import TestClient

    from gne.api import Session
    from gne.ui.server import create_app
    root = tmp_path_factory.mktemp("gatedui") / "root"
    s = Session(MODEL, root=root, n_gpu_layers=NGL, n_ctx=1024)
    c = root / "c_gated.yaml"
    c.write_text(yaml.safe_dump(CONTRAST, allow_unicode=True), encoding="utf-8")
    s.direction(contrast=c, name="gd", positions="response", response_tokens=20)
    s.direction_key(c, role="scene", layers=[8, 12], name="g_scene", subset="all", max_tokens=20, pseudo=3)
    s.direction_key(c, role="one", layers=[8, 12], name="g_one", subset="all", max_tokens=20)
    s.direction_key(c, role="one", layers=[8], name="g_one8", subset="all", max_tokens=20)
    s.recipe_save("chips_t", ["8:H3:s4@gd", "12:H5:s4@gd"])
    s.recipe_save("plain_t", ["14:100:+0.5"])
    usage = s.nw_neurons([8, 12], max_tokens=12, corpus_tokens=512, top=10)
    n = {int(x["layer"]): [r["neuron"] for r in x["rows"]] for x in usage["layers"]}
    app = create_app(s, warm=False)
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield {"client": TestClient(app), "session": s, "url": f"http://127.0.0.1:{port}", "contrast": str(c), "n": n}
    srv.should_exit = True
    th.join(5)
    _close_app(app, s)


@pytest.fixture(scope="module")
def headed(served):
    """ジョブ fish_heads(名前 hga、ニューロンを書いて、レシピ hga_r に保存)。"""
    n = served["n"]
    nr = f"8:{n[8][2]},{n[8][3]} 12:{n[12][2]},{n[12][3]}"
    return _run_job(served["client"], "fish_heads", {"contrast": served["contrast"], "recipe": "chips_t", "name": "hga", "scene_key": "g_scene",
                                                     "one_key": "g_one", "subset": "all", "max_tokens": 24, "neurons": nr, "save": "hga_r"})


@pytest.mark.model
def test_heads_job(served, headed):
    c, s, n = served["client"], served["session"], served["n"]
    r = headed["result"]
    assert r["name"] == "hga" and r["chips"] == "chips_t" and r["contrast_path"] == served["contrast"]
    assert r["direction"] == "hga_dir" and r["amount_key"] == "hga_amount" and (r["scene_key"], r["one_key"]) == ("g_scene", "g_one")
    assert [x["layer"] for x in r["layers"]] == [8, 12] and all(x["n_rows"] > 0 and -1 <= x["r2_cv"] <= 1 for x in r["layers"])
    assert r["layers"][0]["heads"] == ["8:H3:s4@gd"] and r["n_scene"] == 6 and r["n_other"] == 4
    assert r["units"] == [f"8:N{n[8][2]}+N{n[8][3]}:nw:a1@hga_dir#g_scene&g_one!kauto!takehga_amount",
                          f"12:N{n[12][2]}+N{n[12][3]}:nw:a1@hga_dir#g_scene&g_one!kauto!takehga_amount"]
    m = r["match"]
    assert m["layer"] == 12 and all(m[k]["n_positions"] > 0 for k in ("open", "scene", "other"))
    assert Path(r["recipe_path"]) == s.store.recipes_dir() / "hga_r.yaml" and Path(r["recipe_path"]).exists()
    saved = next(x for x in c.get("/api/recipes").json() if x["name"] == "hga_r")      # 一覧に、量子化の倍率と取り除く量の鍵が載る
    assert [(u["kscale"], u["remove"], u["take"], u["key2"]) for u in saved["units"]] == [("auto", None, "hga_amount", "g_one")] * 2
    keys = {k["name"]: k for k in c.get("/api/keys").json()}
    assert keys["hga_amount"]["target"] == "mlpin" and keys["hga_amount"]["layers"] == [8, 12]
    assert keys["g_scene"]["positions"] == "role:scene" and keys["g_one"]["positions"] == "role:one"   # 画面は positions で役割を見分ける


def _open(browser, url, lang):
    pg = browser.new_page(viewport={"width": 1600, "height": 1100}, locale="ja-JP")
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.add_init_script("addEventListener('DOMContentLoaded', () => { window.__asked = []; window.__ans = 'accept';"
                       " window.gneAsk = (msg) => { window.__asked.push(msg); const a = window.__ans; window.__ans = 'accept'; return a === 'accept'; }; })")
    pg.goto(url + f"/?lang={lang}")
    pg.wait_for_function("window.gne && window.gne.recipe && window.gne.arch", timeout=120000)
    return pg, errors


def _open_fish_heads(pg, spec):
    """FiSH の画面を開き、1. 対照集合に JSON を入れて、手順「CHiPS のヘッドの分を場面で取り除く」を開く。一覧が読めるまで待つ。"""
    pg.click("#views button[data-view=direction]")
    pg.wait_for_function("window.gne.directionView && window.gne.directionView.mode", timeout=60000)
    pg.click("#views button[data-view=fish]")
    pg.wait_for_function("window.gne.directionView.fish.loaded", timeout=30000)
    pg.click("#dv-fish [role=tab][data-step=src]")
    pg.click("#fx-src button[data-src=json]")
    pg.evaluate("""(text) => { const ta = document.querySelector('#fx-text'); ta.value = text; ta.dispatchEvent(new Event('input')); }""",
                json.dumps(spec, ensure_ascii=False, indent=2))
    pg.click("#dv-fish [role=tab][data-step=heads]")
    pg.wait_for_function("[...document.querySelectorAll('#fx-hg-recipe option')].some((o) => o.value === 'chips_t')", timeout=30000)


def _recipe_tab(pg, tab="units"):
    if pg.evaluate("document.querySelector('#recipe').classList.contains('collapsed')"):
        pg.click("#recipe .toggle")
    pg.click(f"#recipe-tabs button[data-rtab={tab}]")


@pytest.mark.model
def test_recipe_panel_new_units(served, browser, headed):
    """レシピのパネル: 取り除く形・量の鍵・量子化の倍率の表示。強さを変えて保存し、読み直しても同じ書き方に戻る。"""
    c, s, n = served["client"], served["session"], served["n"]
    from gne.nwrite import spec_of
    from gne.steer import Recipe
    units = headed["result"]["units"] + [f"8:N{n[8][4]}+N{n[8][5]}:nw:a0.5@gd#g_one&g_one!cap0.1-0.2!k4!rm",
                                         f"12:N{n[12][4]}+N{n[12][5]}:nw:a1@gd#g_scene&g_one!kauto!rmbase",
                                         f"12:N{n[12][6]}:nw:a0.8@gd#g_scene"]
    assert c.post("/api/recipes/nwforms", json={"units": units}).status_code == 200
    for lang in ("ja", "en"):
        pg, errors = _open(browser, served["url"], lang)
        pg.evaluate("window.gne.recipe.clear()")
        _recipe_tab(pg)
        pg.wait_for_function("[...document.querySelectorAll('#rc-load option')].some(o => o.value === 'nwforms')", timeout=30000)
        pg.select_option("#rc-load", "nwforms")
        pg.wait_for_function("document.querySelectorAll('#recipe-list tr.rc-nw').length === 5", timeout=10000)
        assert sorted(pg.evaluate("window.gne.recipe.specs(1, false)")) == sorted(units)
        rows = pg.evaluate("[...document.querySelectorAll('#recipe-list tr.rc-nw')].map((tr) => ({ t: tr.textContent.replace(/\\s+/g, ' '), rm: tr.classList.contains('rc-nw-rm'), x: !!tr.querySelector('.rc-nwx') }))")
        by = {r["t"].split(" nw")[0].strip(): r for r in rows}
        take = by[f"L8 N{n[8][2]}+N{n[8][3]}"]
        rm = by[f"L8 N{n[8][4]}+N{n[8][5]}"]
        base = by[f"L12 N{n[12][4]}+N{n[12][5]}"]
        old = by[f"L12 N{n[12][6]}"]
        if lang == "ja":
            assert "取り除く形: 量の鍵 hga_amount" in take["t"] and "量子化の倍率 kauto" in take["t"] and "取り除く" in take["t"] and take["rm"]
            assert "取り除く形: その位置の成分" in rm["t"] and "k4" in rm["t"] and "上限つき" in rm["t"]
            assert "取り除く形: 基準値を超えた分" in base["t"] and "kauto" in base["t"]
            assert not old["rm"] and not old["x"] and "取り除く" not in old["t"]
            _shot(pg, "gated-recipe-panel-ja.png", selector="#recipe")
            # 強さを変えて保存 → ファイルに倍率・形・量の鍵が残る → 読み直しても同じ書き方
            i = pg.evaluate(f"window.gne.recipe.units.findIndex((u) => u.neuron === {n[8][2]})")
            pg.fill(f"#recipe-list tr[data-i=\"{i}\"] input.num", "0.6")
            pg.dispatch_event(f"#recipe-list tr[data-i=\"{i}\"] input.num", "change")
            pg.wait_for_function(f"window.gne.recipe.units[{i}].alpha === 0.6", timeout=5000)
            want = sorted([units[0].replace(":nw:a1@", ":nw:a0.6@")] + units[1:])
            assert sorted(pg.evaluate("window.gne.recipe.specs(1, false)")) == want
            assert pg.evaluate("document.querySelector('#recipe-list tr.rc-nw .rc-nwx').textContent").count("hga_amount") == 1
            pg.fill("#rc-name", "nwforms2")
            pg.dispatch_event("#rc-name", "change")
            pg.click("#rc-save")
            pg.wait_for_function("document.querySelector('#rc-result') && document.querySelector('#rc-result').textContent.includes('nwforms2')", timeout=30000)
            path = s.store.recipes_dir() / "nwforms2.yaml"
            assert sorted(spec_of(u) for u in Recipe.load(path).units) == want
            ys = {(u["neuron"], u.get("pair")): u for u in yaml.safe_load(path.read_text(encoding="utf-8"))["units"]}
            assert ys[(n[8][2], n[8][3])]["kscale"] == "auto" and ys[(n[8][2], n[8][3])]["take"] == "hga_amount" and ys[(n[8][2], n[8][3])]["alpha"] == 0.6
            assert ys[(n[8][4], n[8][5])]["remove"] == "part" and ys[(n[12][4], n[12][5])]["remove"] == "excess" and "kscale" not in ys[(n[12][6], None)]
            pg.evaluate("window.gne.recipe.clear()")
            pg.reload()
            pg.wait_for_function("window.gne && window.gne.recipe && window.gne.arch", timeout=120000)
            _recipe_tab(pg)
            pg.wait_for_function("[...document.querySelectorAll('#rc-load option')].some(o => o.value === 'nwforms2')", timeout=30000)
            pg.select_option("#rc-load", "nwforms2")
            pg.wait_for_function("document.querySelectorAll('#recipe-list tr.rc-nw').length === 5", timeout=10000)
            assert sorted(pg.evaluate("window.gne.recipe.specs(1, false)")) == want
        else:
            assert "removal: amount key hga_amount" in take["t"] and "quantization scale kauto" in take["t"] and "removes" in take["t"]
            assert "removal: component at the position" in rm["t"] and "removal: part above the baseline" in base["t"]
            _shot(pg, "gated-recipe-panel-en.png", selector="#recipe")
            assert pg.evaluate("[...window.gneI18n.missing]") == []
        assert not errors, errors
        pg.close()


@pytest.mark.model
def test_heads_step_in_browser(served, browser):
    """FiSH の手順「CHiPS のヘッドの分を場面で取り除く」: 選べるもの・入力の確かめ・実行・結果の表・レシピに足す・上書きの確かめ。"""
    s = served["session"]
    pg, errors = _open(browser, served["url"], "ja")
    pg.evaluate("window.gne.recipe.clear()")
    _open_fish_heads(pg, CONTRAST)
    f = "window.gne.directionView.fish"
    assert pg.evaluate("document.querySelector('#fx-shared').hidden") and pg.evaluate("document.querySelector('#fx-formset').hidden")   # 共通の設定は読まない
    assert pg.evaluate("document.querySelector('#dv-fish [role=tab][data-step=heads]').classList.contains('sep')")
    opts = pg.evaluate("[...document.querySelectorAll('#fx-hg-recipe option')].map((o) => [o.value, o.textContent])")
    assert opts == [["chips_t", "chips_t(proj 2 個 · 層 8,12)"]]                         # proj の無いレシピは出ない
    assert pg.evaluate("document.querySelector('#fx-hg-layers').textContent") == "CHiPS のヘッドの層: 8, 12(鍵はこの層を全部含むもの)"
    assert "gated_ui" in pg.evaluate("document.querySelector('#fx-hg-src').textContent")
    assert pg.evaluate("[document.querySelector('#fx-hg-scene').value, document.querySelector('#fx-hg-one').value]") == ["g_scene", "g_one"]
    one8 = pg.evaluate("(() => { const o = [...document.querySelectorAll('#fx-hg-one option')].find((x) => x.value === 'g_one8'); return [o.disabled, o.textContent]; })()")
    assert one8[0] is True and "(層 12 が無い)" in one8[1]
    assert pg.evaluate("[...document.querySelectorAll('#fx-hg-scene option')].map((o) => o.value)") == ["g_scene"]     # 役割 scene の鍵だけ
    assert pg.evaluate("document.querySelector('#fx-hg-keyhint').textContent") == ""
    assert pg.evaluate("[document.querySelector('#fx-hg-name').placeholder, document.querySelector('#fx-hg-save').placeholder]") == ["chips_t_hg", "chips_t_hg_r"]
    assert "残したい振る舞いまで弱まる" in pg.evaluate("document.querySelector('section[data-step=heads]').textContent")
    # 入力の確かめ: 投げずに、欄に「直すところ」の印と理由
    pg.evaluate("document.querySelector('section[data-step=heads] details.kx-adv').open = true")
    for text, word in (("8:1,2", "足りない層 12"), ("8:1,1 12:3,4", "違う番号")):
        pg.fill("#fx-hg-neurons", text)
        pg.click("#fx-hg-go")
        pg.wait_for_function("document.querySelector('#fx-status-heads').classList.contains('err')", timeout=5000)
        assert word in pg.evaluate("document.querySelector('#fx-status-heads').textContent")
        assert pg.evaluate("document.querySelector('#fx-hg-neurons').getAttribute('aria-invalid')") == "true"
    pg.fill("#fx-hg-neurons", "")
    assert pg.evaluate("document.querySelector('#fx-hg-neurons').getAttribute('aria-invalid')") is None             # 直したら印を消す
    pg.fill("#fx-hg-name", "hg bad")
    pg.click("#fx-hg-go")
    pg.wait_for_function("document.querySelector('#fx-hg-name').getAttribute('aria-invalid') === 'true'", timeout=5000)
    pg.fill("#fx-hg-name", "hgb")
    assert pg.evaluate("document.querySelector('#fx-hg-save').placeholder") == "hgb_r"
    pg.select_option("#fx-hg-subset", "all")
    pg.fill("#fx-hg-mt", "24")
    pg.evaluate("document.querySelector('section[data-step=heads] details.kx-adv').open = false")
    _shot(pg, "gated-step-input-ja.png")
    posts = []
    pg.on("request", lambda req: posts.append(json.loads(req.post_data)) if req.url.endswith("/api/jobs") and req.method == "POST" else None)
    pg.click("#fx-hg-go")
    pg.wait_for_function("document.querySelectorAll('#fx-hg-fit tbody tr').length === 2", timeout=1500000)
    q = posts[0]
    assert q["kind"] == "fish_heads" and set(q["params"]) == {"spec", "recipe", "name", "scene_key", "one_key", "subset", "max_tokens", "save"}
    assert (q["params"]["name"], q["params"]["save"], q["params"]["subset"], q["params"]["max_tokens"]) == ("hgb", "hgb_r", "all", 24)
    assert "できました" in pg.evaluate("document.querySelector('#fx-status-heads').textContent")
    assert pg.evaluate("document.querySelector('#dv-fish [role=tab][data-step=heads] [data-mark]').className.includes('ok')")
    fit = pg.evaluate("[...document.querySelectorAll('#fx-hg-fit tbody tr')].map((tr) => [...tr.children].map((x) => x.textContent))")
    assert [x[0] for x in fit] == ["L8", "L12"] and all(float(x[2]) <= 1 and int(x[4]) > 0 for x in fit)
    match = pg.evaluate("[...document.querySelectorAll('#fx-hg-match tbody tr')].map((tr) => [tr.dataset.hg, ...[...tr.children].map((x) => x.textContent)])")
    assert [x[0] for x in match] == ["open", "scene", "other"] and [x[1] for x in match] == ["どこでも開く鍵・狙う場面", "本物の鍵・狙う場面", "本物の鍵・dont_touch"]
    assert all(x[2] != "—" and x[3] != "—" and int(x[4]) > 0 for x in match)
    assert pg.evaluate("document.querySelectorAll('#fx-hg-units tbody tr').length") == 2
    assert pg.evaluate("document.querySelector('#fx-hg-units tbody tr td.l').textContent").endswith("!kauto!takehgb_amount")
    saved = pg.evaluate("document.querySelector('#fx-hg-saved').textContent")
    assert saved.startswith("レシピの保存先") and saved.endswith("hgb_r.yaml") and (s.store.recipes_dir() / "hgb_r.yaml").exists()
    assert pg.evaluate("document.querySelector('#fx-hg-match-h').textContent").startswith("CHiPS との一致(層 12")
    for t in ("fit", "match", "units"):                                                 # 表の入れ物は名前つきの領域(キーで横に送れる)
        assert pg.evaluate(f"(() => {{ const d = document.querySelector('#fx-hg-{t}').parentElement; return [d.getAttribute('role'), d.tabIndex, document.getElementById(d.getAttribute('aria-labelledby')) !== null]; }})()") == ["region", 0, True]
    _shot(pg, "gated-step-result-ja.png")
    # レシピに足す → レシピのタブの行
    pg.click("#fx-hg-add")
    pg.wait_for_function("document.querySelectorAll('#recipe-list tr.rc-nw').length === 2", timeout=10000)
    assert "レシピに足しました(2 個)" in pg.evaluate("document.querySelector('#fx-status-heads').textContent")
    assert all(x.endswith("!kauto!takehgb_amount") for x in pg.evaluate("window.gne.recipe.specs(1, false)"))
    assert "取り除く形: 量の鍵 hgb_amount" in pg.evaluate("document.querySelector('#recipe-list tr.rc-nw').textContent")
    _shot(pg, "gated-step-added-ja.png")
    # 同じ名前でもう一度: 上書きを確かめ、やめたら投げない
    pg.wait_for_function(f"{f}.hg.keys.some((k) => k.name === 'hgb_amount') && {f}.hg.recipeNames.has('hgb_r')", timeout=30000)
    n_posts = len(posts)
    pg.evaluate("window.__ans = 'dismiss'")
    pg.click("#fx-hg-go")
    pg.wait_for_function("document.querySelector('#fx-status-heads').textContent === 'やめました'", timeout=5000)
    asked = pg.evaluate("window.__asked")
    assert asked and "方向 hgb_dir と取り除く量の鍵 hgb_amount・レシピ hgb_r を作り直します" in asked[-1]
    assert len(posts) == n_posts
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_heads_step_in_english(served, browser, headed):
    """英語の画面: 手順の欄・入力の確かめ・結果の表(ジョブの結果を入れて描く)・レシピに足した行の文言が辞書にある。"""
    pg, errors = _open(browser, served["url"], "en")
    pg.evaluate("window.gne.recipe.clear()")
    _open_fish_heads(pg, CONTRAST)
    f = "window.gne.directionView.fish"
    assert pg.evaluate("document.querySelector('#dv-fish [role=tab][data-step=heads]').textContent").strip().startswith("Remove CHiPS head contribution by scene")
    assert pg.evaluate("document.querySelector('#fx-hg-layers').textContent") == "CHiPS head layers: 8, 12 (keys must cover all of them)"
    assert pg.evaluate("document.querySelector('#fx-hg-recipe option').textContent") == "chips_t (2 proj · layers 8,12)"
    assert "(no layer 12)" in pg.evaluate("[...document.querySelectorAll('#fx-hg-one option')].map((o) => o.textContent).join(' ')")
    assert "Build the removal units" in pg.evaluate("document.querySelector('#fx-hg-go').textContent")
    pg.evaluate("document.querySelector('section[data-step=heads] details.kx-adv').open = true")
    pg.fill("#fx-hg-neurons", "8:1,2")
    pg.click("#fx-hg-go")
    pg.wait_for_function("document.querySelector('#fx-status-heads').classList.contains('err')", timeout=5000)
    assert pg.evaluate("document.querySelector('#fx-status-heads').textContent").startswith("If you give neurons, give them for every CHiPS head layer 8,12")
    pg.fill("#fx-hg-neurons", "")
    pg.fill("#fx-hg-name", "hga")
    pg.fill("#fx-hg-mt", "24")
    _shot(pg, "gated-step-input-en.png")
    job = {"id": "en1", "kind": "fish_heads", "status": "done", "result": headed["result"]}
    pg.evaluate(f"(async (j) => {{ const p = {f}; p.jobs.heads = j.id; await p.jobEnded(j); }})", job)
    pg.wait_for_function("document.querySelectorAll('#fx-hg-fit tbody tr').length === 2", timeout=10000)
    txt = pg.evaluate("document.querySelector('#fx-heads').textContent")
    for w in ("Fit of the amount to remove (per layer)", "Agreement with CHiPS (change of layer 12's output", "key open everywhere · target scene",
              "real keys · target scene", "real keys · dont_touch", "Units built", "Recipe saved to", "amount key"):
        assert w in txt, w
    assert [x.strip() for x in pg.evaluate("[...document.querySelectorAll('#fx-hg-fit thead th')].map((x) => x.textContent)")] == ["Layers", "heads", "R² (5-fold)", "Regularization", "rows"]
    _shot(pg, "gated-step-result-en.png")
    pg.click("#fx-hg-add")
    pg.wait_for_function("document.querySelectorAll('#recipe-list tr.rc-nw').length === 2", timeout=10000)
    assert "removal: amount key hga_amount" in pg.evaluate("document.querySelector('#recipe-list tr.rc-nw').textContent")
    _shot(pg, "gated-step-added-en.png")
    missing = pg.evaluate("[...window.gneI18n.missing]")
    assert missing == [], missing
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_heads_job_errors(served):
    """引数のエラー(モデルを流す前に止まる)。ブラウザのテストの後に回す(ジョブの帯にエラーを残さないため)。"""
    c = served["client"]
    base = {"contrast": served["contrast"], "recipe": "chips_t", "name": "hgx", "scene_key": "g_scene", "one_key": "g_one"}
    for over, word in (({"scene_key": None}, "scene_key"), ({"subset": "half"}, "subset"), ({"neurons": "8:1,2"}, "足りない層 [12]"),
                       ({"neurons": "8:1"}, "2 個ずつ"), ({"neurons": "8:a,b"}, "層:番号,番号"), ({"recipe": "nothing_here"}, "")):
        e = _run_job(c, "fish_heads", {k: v for k, v in {**base, **over}.items() if v is not None}, ok=False)
        assert e["status"] == "error" and e["error"]["code"] in ("BAD_ARGS", "FILE_NOT_FOUND", "BAD_RECIPE"), e["error"]
        assert word in e["error"]["message"], e["error"]
    e = _run_job(c, "fish_heads", {**base, "recipe": "plain_t"}, ok=False)              # proj の無いレシピ
    assert e["error"]["code"] == "BAD_RECIPE" and "proj" in e["error"]["message"]
