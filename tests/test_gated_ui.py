"""Web UI: レシピのパネルの nw のユニットの新しい書き方(!k4・!kauto・!rm・!rmbase・!take<鍵>)のテスト。

モデルなし: 書き方を読む・書く(JS の parseNw・nwSpec と Python の parse_unit・spec_of が同じ文字に戻る)、レシピの行の読み込みと
強さの変更・全体の倍率で後ろの項目が消えないこと(ブラウザは Playwright の Chromium、無ければ飛ばす)。
"""

import os
import urllib.parse
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CHROME = Path(os.environ.get("GNE_CHROME", Path.home() / ".cache/ms-playwright/chromium-1223/chrome-linux64/chrome"))
STATIC = ROOT / "gne" / "ui" / "static"

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


# --- モデルなし ---------------------------------------------------------------------------------------------

def test_spec_roundtrip_python():
    from gne.nwrite import spec_of
    from gne.steer import parse_unit
    for x in SPECS:
        assert spec_of(parse_unit(x)) == x, x
    assert spec_of(parse_unit("8:N7:nw:a1@-d#k")) == "8:N7:nw:a-1@d#k"            # @-方向 は強さの符号に畳む


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
