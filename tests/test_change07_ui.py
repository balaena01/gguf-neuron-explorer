"""CHANGE-07 のテスト(UI のパッケージ): 鍵つきの書き込み(keyed write)の Web UI。

サーバ(モデルあり、Qwen3-0.6B、GNE_TEST_MODEL。CPU で回すときは GNE_TEST_NGL=0): 対照集合の候補(/api/contrasts)と保存した
鍵(/api/keys・/api/keys/<名前>・/api/model の keys)、ジョブ direction_key(語尾の対照集合 configs/contrast/zundamon_endings.yaml
と与えた返事)と direction_key_check(値とヒストグラムを落とさない)、keyed のユニットと default_key・default_direction の
レシピの保存と読み直し(省いた方向・鍵は既定で埋め、埋めたことを *_default で返す)、keyed の書き出し(headproj_export の
ジョブで LoRA と F16 の焼き込み、cvec の書き出しは keyed を外した警告)。
ブラウザ(Playwright の Chromium、無ければ飛ばす): direction ビューの「鍵(keyed)」で鍵を作り、層ごとの表を見て、key-check の
表と層ごとのヒストグラム(SVG)が描かれる。行の「+ レシピ」で keyed のユニットがレシピに入り、保存して読み直しても残る。英語の文言もある。
"""

import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
NGL = int(os.environ.get("GNE_TEST_NGL", "-1"))
CHROME = Path(os.environ.get("GNE_CHROME", Path.home() / ".cache/ms-playwright/chromium-1223/chrome-linux64/chrome"))
ZUNDAMON = ROOT / "configs" / "contrast" / "zundamon_endings.yaml"
SHOTS = os.environ.get("GNE_TEST_SHOTS")          # 画面の写しを置くフォルダ(任意)

# モデル自身の素の返事に見立てた、です・ます調の返事(zundamon_endings.yaml の user_prompts と同じ順)
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
NAME = "zundamon_endings"                         # 対照集合・方向・鍵の名前


def _run_job(c, kind, params, timeout=900, ok=True):
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
    """走っているジョブを止め、モデルの処理の終わりを GPU の調停で待ってから閉じる(test_ui.py の _serve と同じ)。"""
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


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    """Session(CPU)と TestClient。方向 zundamon_endings(system_a 対 なし、プロンプトの最後)を作っておく。"""
    if not MODEL.exists():
        pytest.skip(f"{MODEL} がありません")
    from fastapi.testclient import TestClient

    from gne.api import Session
    from gne.ui.server import create_app
    s = Session(MODEL, root=tmp_path_factory.mktemp("c07ui"), n_gpu_layers=NGL, n_ctx=1024)
    s.direction(contrast=str(ZUNDAMON), positions="last")
    app = create_app(s, warm=False)
    yield TestClient(app), s
    _close_app(app, s)


@pytest.fixture(scope="module")
def built(api):
    """鍵 zundamon_endings(作り方 (b): 与えた返事の「です。」「ます。」を書く直前、L8・L14、擬似鍵 8 本)をジョブで作る。"""
    c, _ = api
    return _run_job(c, "direction_key", {"contrast": str(ZUNDAMON), "layers": "8,14", "replies_a": REPLIES, "max_tokens": 24,
                                         "pseudo": 8})


# --- サーバ ----------------------------------------------------------------------------------------------

@pytest.mark.model
def test_contrasts_and_empty_keys(api):
    c, s = api
    cs = c.get("/api/contrasts").json()
    z = next(x for x in cs if x["file"] == NAME and x["src"] == "configs")
    assert z["label"] == f"configs/contrast/{NAME}.yaml" and Path(z["path"]) == ZUNDAMON.resolve()
    assert z["mode"] == "same" and z["positions"] == "before:(です|ます)。" and z["dont_touch"] == ["english"]
    assert z["n_prompts"] == 16 and z["system_a"] and not z["replies"]
    assert c.get("/api/keys").json() == [] and c.get("/api/model").json()["keys"] == []
    r = c.get("/api/keys/nope")
    assert r.status_code == 400 and r.json()["error"]["code"] == "NOT_FOUND"
    assert c.get("/api/keys/..%2Fx").status_code in (400, 404)


@pytest.mark.model
def test_direction_key_job_and_key_list(api, built):
    c, s = api
    r = built["result"]
    assert built["kind"] == "direction_key" and r["name"] == NAME and r["mode"] == "same" and r["target"] == "ffn"
    assert [x["layer"] for x in r["layers"]] == [8, 14] and r["n_pseudo"] == 8 and r["matched"] == {"A": 8, "of": 8}
    for x in r["layers"]:
        assert 0 < x["kn"] <= 1 and x["q_rank"] > 0 and x["build_auc"] is not None
    ks = c.get("/api/keys").json()
    assert [k["name"] for k in ks] == [NAME] and ks[0]["layers"] == [8, 14] and ks[0]["target"] == "ffn"
    assert "path" not in ks[0]                                               # 一覧にパスを出さない
    assert [k["name"] for k in c.get("/api/model").json()["keys"]] == [NAME]
    one = c.get(f"/api/keys/{NAME}").json()
    assert one["info"]["mode"] == "same" and set(one["info"]["build_auc"]) == {"8", "14"} and one["n_pseudo"] == 8
    assert one["info"]["dont_touch"][0]["name"] == "english" and one["model_id"] == s.meta.model_id


@pytest.mark.model
def test_direction_key_check_job_keeps_hist(api, built):
    c, _ = api
    v = _run_job(c, "direction_key_check", {"key": NAME, "contrast": str(ZUNDAMON), "replies_a": REPLIES, "max_tokens": 24})
    d = v["result"]
    assert d["held_out"] is True and d["subset"] == "odd" and [x["layer"] for x in d["layers"]] == [8, 14]
    for x in d["layers"]:
        h = x["hist"]                                                        # UI のヒストグラム: compact でないので残る
        assert set(h) == {"edges", "A", "B", "dt:english"} and len(h["edges"]) == len(h["A"]) + 1
        assert sum(h["A"]) > 0 and sum(h["B"]) > sum(h["A"]) and x["values"]["A"]
        assert x["auc"] is not None and x["beats_pseudo"] in (True, False) and x["pseudo_selectivity"]["n"] == 8
    # 対照集合なし: 作成時の例ごとの平均で測る(held_out: false)
    d2 = _run_job(c, "direction_key_check", {"key": NAME, "layers": "14"})["result"]
    assert d2["held_out"] is False and d2["contrast"] is None and [x["layer"] for x in d2["layers"]] == [14]
    e = _run_job(c, "direction_key_check", {"key": "no_such_key"}, ok=False)
    assert e["status"] == "error" and e["error"]["code"] == "FILE_NOT_FOUND"


@pytest.mark.model
def test_recipe_keyed_round_trip(api, built):
    """keyed のユニットを保存して読み直す。@-方向は α の符号に畳む。省いた方向・鍵はレシピの既定(default_direction・
    default_key)で埋めて返し、埋めたことは direction_default・key_default で分かる。"""
    c, s = api
    units = ["14:ffn:keyed:a1.25", f"8:ffn:keyed:a0.5@-{NAME}#{NAME}"]
    r = c.post("/api/recipes/zk", json={"units": units, "default_key": NAME, "default_direction": NAME})
    assert r.status_code == 200, r.text
    saved = yaml.safe_load((s.store.recipes_dir() / "zk.yaml").read_text(encoding="utf-8"))
    assert saved["default_key"] == NAME and saved["default_direction"] == NAME
    u14 = next(u for u in saved["units"] if u["layer"] == 14)
    assert u14 == {"layer": 14, "kind": "ffn", "mode": "keyed", "alpha": 1.25}            # 方向・鍵は省いたまま
    zk = next(x for x in c.get("/api/recipes").json() if x["name"] == "zk")
    assert (zk["default_key"], zk["default_direction"], zk["direction_layer"]) == (NAME, NAME, "own")
    by = {u["layer"]: u for u in zk["units"]}
    assert by[14]["mode"] == "keyed" and by[14]["alpha"] == 1.25 and by[14]["key"] == NAME and by[14]["direction"] == NAME
    assert by[14]["key_default"] is True and by[14]["direction_default"] is True
    assert by[8]["alpha"] == -0.5 and by[8]["direction"] == NAME and by[8]["key"] == NAME          # -方向は α の符号に
    assert "key_default" not in by[8] and "direction_default" not in by[8]
    # 鍵が無い keyed は断る(BAD_RECIPE)、鍵の名前が違えば FILE_NOT_FOUND
    bad = c.post("/api/recipes/zbad", json={"units": [f"14:ffn:keyed:a1@{NAME}"]})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "BAD_RECIPE"
    bad = c.post("/api/recipes/zbad", json={"units": [f"14:ffn:keyed:a1@{NAME}#nope"]})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "FILE_NOT_FOUND"
    assert not (s.store.recipes_dir() / "zbad.yaml").exists()


@pytest.mark.model
def test_keyed_export_via_headproj_job(api, built):
    """keyed のレシピは headproj_export のジョブで LoRA(既定)か焼き込み(F16)。cvec の書き出しは keyed を外した警告を出す。"""
    c, s = api
    unit = f"14:ffn:keyed:a1@{NAME}#{NAME}"
    v = _run_job(c, "headproj_export", {"units": [unit], "name": "zk", "bake": False, "file_stem": "zk-keyed", "overwrite": False})
    x = v["result"]
    assert x["kind"] == "lora" and x["file"] == "zk-keyed.lora.gguf" and x["url"] == "/api/export/file/zk-keyed.lora.gguf"
    assert x["keyed_layers"]["14"]["keys"] == [NAME] and x["keyed_rank"] == {"blk.14.ffn_down.weight": 1}
    assert c.get(x["url"]).status_code == 200
    e = _run_job(c, "headproj_export", {"units": [unit], "name": "zk", "bake": False, "file_stem": "zk-keyed", "overwrite": False},
                 ok=False)
    assert e["error"]["code"] == "FILE_EXISTS"                                       # UI は確かめてから上書きする
    b = _run_job(c, "headproj_export", {"units": [unit], "name": "zk", "bake": True, "requant": "f16", "file_stem": "zk-bake",
                                        "overwrite": False})["result"]
    try:
        assert b["kind"] == "bake" and b["modified"][0]["to"] == "F16" and b["keyed_layers"]["14"]["rank"] == 1
        assert b["requant_error_max"] < 0.5 and "url" not in b
    finally:
        Path(b["path"]).unlink(missing_ok=True)
    ex = c.post("/api/export", json={"units": [f"14:dir:{NAME}:+0.05"], "name": "zkx", "keyed": 2}).json()
    assert ex["file"] == "zkx.cvec.gguf" and "keyed、2 個" in ex["warnings"][0] and "headproj_export" in ex["warnings"][0]
    r = c.post("/api/export", json={"units": [unit], "name": "zky"})                  # keyed だけでは cvec にならない
    assert r.status_code == 400 and r.json()["error"]["code"] == "BAD_RECIPE"


# --- ブラウザ --------------------------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


@pytest.fixture(scope="module")
def ui_server(tmp_path_factory):
    """ブラウザのテスト用のサーバ(CPU)。方向 zundamon_endings と、返事のファイル(フォームの「返事のファイル」に入れる)。"""
    if not MODEL.exists():
        pytest.skip(f"{MODEL} がありません")
    if not CHROME.exists():
        pytest.skip(f"Chromium がありません({CHROME}、GNE_CHROME で指定)")
    import uvicorn

    from gne.api import Session
    from gne.ui.server import create_app
    d = tmp_path_factory.mktemp("c07ui_web")
    replies = d / "replies.yaml"
    replies.write_text(yaml.safe_dump(REPLIES, allow_unicode=True), encoding="utf-8")
    s = Session(MODEL, root=d / "root", n_gpu_layers=NGL, n_ctx=1024)
    s.direction(contrast=str(ZUNDAMON), positions="last")
    app = create_app(s, warm=False)
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield {"url": f"http://127.0.0.1:{port}", "replies": str(replies), "session": s}
    srv.should_exit = True
    th.join(5)
    _close_app(app, s)


@pytest.fixture(scope="module")
def browser():
    if not CHROME.exists():
        pytest.skip(f"Chromium がありません({CHROME}、GNE_CHROME で指定)")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        # GPU は使わない(並行するモデルの処理とぶつからないように)。地図の WebGL はソフトウェアで描く
        b = p.chromium.launch(executable_path=str(CHROME), headless=True,
                              args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        yield b
        b.close()


def _shot(pg, name):
    if SHOTS:
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        pg.screenshot(path=str(Path(SHOTS) / name), full_page=False)


def _open(browser, url, lang):
    pg = browser.new_page(viewport={"width": 1600, "height": 1100}, locale="ja-JP")
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("dialog", lambda dl: dl.accept())
    pg.goto(url + f"/?lang={lang}")
    pg.wait_for_function("window.gne && window.gne.recipe && window.gne.arch", timeout=120000)
    return pg, errors


@pytest.mark.model
def test_key_panel_and_recipe_in_browser(ui_server, browser):
    """direction ビューの「鍵(keyed)」: 鍵を作る(進み具合 → 層ごとの kn・gain・q_rank・作成時の AUC)、この鍵で key-check
    (層ごとの表と ρ のヒストグラム 2 枚)、行の「+ レシピ」で keyed のユニットをレシピへ。レシピのタブで方向・鍵が選べ、保存して
    ページを読み直し、保存したレシピを読んでも keyed の行(既定の鍵を使う行も)が残る。LoRA・焼き込みの欄の既定は -keyed と F16。"""
    url, s = ui_server["url"], ui_server["session"]
    pg, errors = _open(browser, url, "ja")
    pg.evaluate("window.gne.recipe.clear()")
    pg.click("#views button[data-view=direction]")
    pg.wait_for_function("window.gne.directionView && window.gne.directionView.mode", timeout=60000)
    pg.click("#dv-mode button[data-m=key]")
    pg.wait_for_function("document.querySelector('#kx-contrast option') && window.gne.directionView.key.loaded", timeout=30000)
    assert pg.evaluate("document.querySelector('#kx-contrast').value").endswith(f"{NAME}.yaml")     # before:<re> の対照集合が既定
    assert "before" in pg.evaluate("document.querySelector('#kx-cinfo').textContent") or "(b)" in pg.evaluate(
        "document.querySelector('#kx-cinfo').textContent")
    assert "まだありません" in pg.evaluate("document.querySelector('#kx-keys').textContent")
    pg.fill("#kx-name", "zk_web")
    pg.fill("#kx-layers", "8,14")
    pg.fill("#kx-pseudo", "8")
    pg.fill("#kx-mt", "24")
    pg.click(".kx-adv summary")
    pg.fill("#kx-replies", ui_server["replies"])
    pg.click("#kx-go")
    pg.wait_for_selector("#kx-build table.kx-tab tbody tr", timeout=600000)
    rows = pg.evaluate("[...document.querySelectorAll('#kx-build table.kx-tab tbody tr')].map(tr => tr.cells[0].textContent)")
    assert rows == ["L8", "L14"]
    assert pg.evaluate("document.querySelector('#kx-dir').value") == NAME                 # 対照集合と同じ名前の方向を選んでおく
    assert "zk_web" in pg.evaluate("document.querySelector('#kx-keys').textContent")
    _shot(pg, "keyed_build.png")
    # この鍵で key-check(対照集合・返事・反対側の subset を入れて回す)
    pg.click("#kx-build [data-kcheck]")
    assert pg.evaluate("document.querySelector('#kc-subset').value") == "odd"
    pg.wait_for_selector("#kc-res table.kx-check tbody tr", timeout=600000)
    chk = pg.evaluate("[...document.querySelectorAll('#kc-res table.kx-check tbody tr')].map(tr => tr.cells[0].textContent.trim())")
    assert [x.replace(" ★", "") for x in chk] == ["L8", "L14"]
    assert pg.evaluate("document.querySelectorAll('#kc-res svg.kx-hist').length") == 2
    assert pg.evaluate("document.querySelectorAll('#kc-res svg.kx-hist rect.bar.ga').length") > 0
    assert pg.evaluate("document.querySelectorAll('#kc-res svg.kx-hist rect.bar.gb').length") > 0
    assert pg.evaluate("document.querySelectorAll('#kc-res svg.kx-hist path.line').length") == 2           # dont_touch english
    assert "保留した例で測りました" in pg.evaluate("document.querySelector('#kc-res').textContent")
    pg.evaluate("document.querySelector('#kc-res .kx-hists').scrollIntoView()")
    _shot(pg, "keyed_check.png")
    # 「+ レシピ」で L14 の keyed をレシピへ(方向 zundamon_endings、鍵 zk_web、α 1)
    pg.click("#kx-build [data-kadd='14']")
    u = pg.evaluate("window.gne.recipe.keyed()")
    assert u == [{"layer": 14, "kind": "ffn", "mode": "keyed", "alpha": 1, "direction": NAME, "key": "zk_web"}]
    pg.wait_for_selector("#recipe-list tr.rc-keyed", timeout=10000)
    # レシピのタブ: 既定の鍵を使う行を足す(L8、鍵は「既定」)。既定の鍵を zk_web に
    pg.evaluate("window.gne.recipe.addKeyed({layer: 8, kind: 'ffn', alpha: 0.5, direction: null, key: null})")
    pg.click("#list-tabs button[data-tab=recipe]")
    pg.wait_for_selector("#rc-kdk", timeout=10000)
    assert pg.evaluate("document.querySelectorAll('#recipe-list tr.rc-keyed .badge.warn').length") >= 1   # 鍵・方向が無い行に印
    pg.select_option("#rc-kdk", "zk_web")
    pg.select_option("#rc-kdd", NAME)
    pg.wait_for_function("document.querySelectorAll('#recipe-list tr.rc-keyed .badge.warn').length === 0", timeout=5000)
    assert pg.evaluate("window.gne.recipe.specs()") == [f"8:ffn:keyed:a0.5@{NAME}#zk_web", f"14:ffn:keyed:a1@{NAME}#zk_web"]
    assert pg.evaluate("window.gne.recipe.specs(-1)")[1] == f"14:ffn:keyed:a-1@{NAME}#zk_web"
    assert pg.evaluate("window.gne.recipe.addSpecs()") == []                                          # cvec には入らない
    assert pg.evaluate("document.querySelector('#rc-export').disabled") and not pg.evaluate("document.querySelector('#rc-lora').disabled")
    pg.fill("#rc-name", "zk_web_r")
    pg.dispatch_event("#rc-name", "change")
    pg.click("#rc-save")
    pg.wait_for_function("document.querySelector('#rc-result') && document.querySelector('#rc-result').textContent.includes('zk_web_r')",
                         timeout=30000)
    saved = yaml.safe_load((s.store.recipes_dir() / "zk_web_r.yaml").read_text(encoding="utf-8"))
    assert saved["default_key"] == "zk_web" and saved["default_direction"] == NAME
    assert {"layer": 8, "kind": "ffn", "mode": "keyed", "alpha": 0.5} in saved["units"]
    # LoRA・焼き込みの欄: keyed だけなら既定の名前は -keyed、焼き込みの型の既定は F16
    pg.click("#rc-lora")
    pg.wait_for_selector(".rc-lx", timeout=5000)
    assert pg.evaluate("document.querySelector('#lx-file').placeholder") == "zk_web_r-keyed"
    pg.check("input[name=lx-kind][value=bake]")
    assert pg.evaluate("document.querySelector('#lx-requant').value") == "f16"
    _shot(pg, "keyed_recipe.png")
    # 読み直し: 下書きを消し、保存したレシピを読む
    pg.evaluate("window.gne.recipe.clear()")
    pg.reload()
    pg.wait_for_function("window.gne && window.gne.recipe && window.gne.arch", timeout=120000)
    pg.evaluate("document.querySelector('#list').classList.remove('collapsed')")
    pg.click("#list-tabs button[data-tab=recipe]")
    pg.wait_for_function("[...document.querySelectorAll('#rc-load option')].some(o => o.value === 'zk_web_r')", timeout=30000)
    pg.select_option("#rc-load", "zk_web_r")
    pg.wait_for_function("document.querySelectorAll('#recipe-list tr.rc-keyed').length === 2", timeout=10000)
    r = pg.evaluate("({units: window.gne.recipe.keyed(), dk: window.gne.recipe.defaultKey, dd: window.gne.recipe.defaultDirection, skipped: window.gne.recipe.skipped})")
    assert r["dk"] == "zk_web" and r["dd"] == NAME and r["skipped"] == 0
    assert {"layer": 8, "kind": "ffn", "mode": "keyed", "alpha": 0.5, "direction": None, "key": None} in r["units"]   # 既定のまま
    assert {"layer": 14, "kind": "ffn", "mode": "keyed", "alpha": 1, "direction": NAME, "key": "zk_web"} in r["units"]
    assert pg.evaluate("document.querySelector('#recipe-list tr.rc-keyed select[data-kkey]').value") == ""        # 「既定」を選んだまま
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_key_panel_in_english(ui_server, browser):
    """英語の画面: 鍵のパネル(保存した鍵の一覧と key-check の結果)・レシピの keyed の行・eval の keyed の欄の文言が辞書にある。"""
    url = ui_server["url"]
    pg, errors = _open(browser, url, "en")
    pg.evaluate(f"window.gne.recipe.clear(); window.gne.recipe.addKeyed({{layer: 14, kind: 'ffn', alpha: 1, direction: '{NAME}', key: 'nope'}})")
    pg.click("#views button[data-view=direction]")
    pg.wait_for_function("window.gne.directionView && window.gne.directionView.mode", timeout=60000)
    pg.click("#dv-mode button[data-m=key]")
    pg.wait_for_function("window.gne.directionView.key.loaded", timeout=30000)
    assert pg.evaluate("!!window.gne.directionView.key")
    # 保存した結果の形で key-check を描く(ジョブは回さない)
    hist = {"edges": [-0.5, 0, 0.5, 1, 1.5], "A": [0, 1, 3, 2], "B": [5, 9, 1, 0], "dt:english": [2, 3, 0, 0]}
    res = {"key": "k", "contrast": "c", "subset": "odd", "mode": "same", "positions": "before:x", "positions_b": "not-before:x",
           "held_out": False, "overlap": {"subset_overlap": True, "hash_overlap": 0, "built_subset": "odd", "subset": "odd"},
           "best_layer": 8, "all_beat_pseudo": False, "n_pseudo": 8, "seconds": 1.0, "counts": {"A": {"examples": 2, "of": 8}},
           "layers": [{"layer": 8, "auc": 0.9, "auc_token": 0.95, "selectivity": 0.8, "pseudo_auc": {"p95": 0.7},
                       "pseudo_auc_token": {"p95": 0.8}, "pseudo_selectivity": {"p95": 0.5}, "beats_pseudo": True,
                       "beats_pseudo_auc": True, "b_split_auc": 0.5, "dont_touch_median": {"english": 0.01}, "kn": 0.4, "hist": hist}],
           "warnings": ["w"], "notes": ["n"]}
    pg.evaluate(f"(() => {{ const k = window.gne.directionView.key; k.checked = {json.dumps(res)}; k.renderCheck(); }})()")
    assert pg.evaluate("document.querySelectorAll('#kc-res svg.kx-hist').length") == 1
    assert "Overlaps the build examples" in pg.evaluate("document.querySelector('#kc-res').textContent")
    pg.evaluate("document.querySelector('#list').classList.remove('collapsed')")
    pg.click("#list-tabs button[data-tab=recipe]")
    pg.wait_for_selector("#recipe-list tr.rc-keyed", timeout=10000)
    assert "key nope not found" in pg.evaluate("document.querySelector('#recipe-list tr.rc-keyed').textContent")
    pg.click("#rc-lora")
    pg.check("input[name=lx-kind][value=bake]")
    pg.click("#views button[data-view=eval]")
    pg.wait_for_function("window.gne.evalView", timeout=30000)
    assert not pg.evaluate("document.querySelector('#evx-kx').classList.contains('hidden')")            # keyed の欄が出る
    missing = pg.evaluate("[...window.gneI18n.missing]")
    assert missing == [], missing
    assert not errors, errors
    pg.close()


def _fake_eval(keyed):
    """eval の結果の形(行の keyed と、結果の keyed のブロック)。ジョブは回さずに描く。"""
    def row(dose, sel):
        return {"dose": dose, "capability": 0.9, "capability_drop_pt": 0.0, "ppl_mean": 10.0, "ppl_worse_pct": 0.0, "mix_zh": 0.0,
                "mix_other": 0.0, "persona": {"novelty": 0.1 * dose, "distinct4": 0.5, "capped": 0.0, "top_words": []},
                "shippable": True, "ok": {"capability": True, "ppl": True, "mix": True},
                "keyed": {"layer": 14, "a_par": 27.5 * dose, "b_norm": 7.28 * dose, "selectivity": sel, "b_over_a": 0.26}}
    rec = {"name": "zk", "units": [{"layer": 14, "neuron": None, "dose": 0.0, "kind": "ffn", "mode": "keyed", "k": 1.0, "direction": NAME,
                                    "key": "zk_web", "head": None, "seek": None, "expert": None, "seed": None}],
           "default_key": "zk_web", "ablate": None, "sampling": None, "dose_unit": "rel", "when": "all"}
    return {"name": "fake", "recipe": rec, "suite": {"name": "s", "n_tasks": 10, "thresholds": {"capability_drop_pt": 3, "ppl_worse_pct": 5}},
            "when": "all", "rows": [row(0, 0.0), row(1, 26.3)], "samples": {}, "seconds": 1.0, "keyed": keyed}


KEYED_BLOCK = {
    "key": "zk_web", "direction": NAME, "objective_layer": 14, "contrast": NAME, "subset": "odd", "held_out": True,
    "positions": "before:(です|ます)。", "positions_b": "not-before:(です|ます)。",
    "counts": {"A": {"examples": 8, "positions": 20}, "B": {"examples": 8, "positions": 300}, "dt:english": {"examples": 2, "positions": 40}},
    "methods": [
        {"method": "keyed", "desc": "このレシピの keyed のユニット", "beta": 1.0, "matched": None, "a_par": 27.5, "b_par": 1.2,
         "b_norm": 7.28, "dt_norm": 4.32, "selectivity": 26.3, "b_over_a": 0.26,
         "recipe": {"name": "zk-keyed", "units": [{"layer": 14, "kind": "ffn", "mode": "keyed", "k": 1.0, "direction": NAME, "key": "zk_web"}],
                    "default_key": None, "ablate": None, "sampling": None}},
        {"method": "add_same_alpha", "desc": "同じ層・同じ α の L:dir:NAME:α", "beta": 1.0, "matched": None, "a_par": 30.1, "b_par": 25.0,
         "b_norm": 31.7, "dt_norm": 32.0, "selectivity": 5.1, "b_over_a": 1.05},
        {"method": "add", "desc": "全体を β 倍", "beta": 0.9, "matched": True, "a_par": 27.4, "b_par": 22.0, "b_norm": 28.5, "dt_norm": 29.0,
         "selectivity": 5.4, "b_over_a": 1.04},
        {"method": "gain", "desc": "ニューロンの gain", "excluded": "β = 0〜8 で側 A の par が keyed に近づきません"},
        {"method": "proj", "desc": "ヘッドの射影", "beta": 93.0, "matched": False, "a_par": 20.0, "b_par": 8.0, "b_norm": 52.7,
         "dt_norm": 40.0, "selectivity": 12.0, "b_over_a": 2.6}],
    "compare": {"b_norm_ratio": {"add_same_alpha": 0.23, "add": 0.26}, "dt_norm_ratio": {"add": 0.15},
                "b_over_a_ratio": {"add_same_alpha": 0.25, "add": 0.25, "proj": 0.1}, "keyed_smaller": True},
    "pseudo": {"n": 8, "p50": 10.0, "p95": 19.7, "real": 26.3, "beats": True, "null_pct": 1.0},
    "notes": ["proj: 側 A の par を keyed に ±10% でそろえられませんでした"], "warnings": [], "cost": {"prefills": 218, "seconds": 28.0}}


@pytest.mark.model
def test_eval_keyed_params_and_block_in_browser(ui_server, browser):
    """eval ビュー: レシピに keyed があると keyed の選択性の欄が出て、ジョブの params に keyed_contrast・keyed_examples・
    keyed_pseudo・keyed_subset・keyed_match・keyed_metrics が載る。結果の keyed は方式の表(除外の理由・そろったか)・比・擬似鍵・注、
    行の表に keyed の選択性の列。測っていない(skipped)ときは理由とヒント。英語の文言もある。"""
    url = ui_server["url"]
    for lang in ("ja", "en"):
        pg, errors = _open(browser, url, lang)
        posts = []

        def on_jobs(route, req):
            if req.method != "POST":
                route.continue_()
                return
            body = json.loads(req.post_data)
            posts.append(body)
            route.fulfill(status=200, content_type="application/json", body=json.dumps({
                "id": "fake-e1", "kind": body["kind"], "params": body["params"], "status": "queued", "done": 0, "total": 0, "stage": "",
                "eta_s": None, "yields": 0, "seconds": None, "result": None, "error": None}))
        pg.route("**/api/jobs", on_jobs)
        pg.route("**/api/jobs/fake-e1/events", lambda route, req: route.fulfill(status=200, content_type="text/event-stream", body=""))
        pg.evaluate(f"window.gne.recipe.clear(); window.gne.recipe.addKeyed({{layer: 14, kind: 'ffn', alpha: 1, direction: '{NAME}', key: null}});"
                    "window.gne.recipe.setDefaults({key: 'zk_web'})")
        pg.click("#views button[data-view=eval]")
        pg.wait_for_function("window.gne.evalView && !document.querySelector('#evx-kx').classList.contains('hidden')", timeout=30000)
        pg.wait_for_function("document.querySelectorAll('#evx-kc option').length > 1", timeout=10000)
        zpath = pg.evaluate(f"[...document.querySelectorAll('#evx-kc option')].map(o => o.value).find(v => v.endsWith('{NAME}.yaml'))")
        pg.select_option("#evx-kc", zpath)
        pg.select_option("#evx-ks", "even")
        pg.fill("#evx-kp", "4")
        pg.uncheck("#evx-kmatch")
        pg.click("#evx-go")
        for _ in range(100):                     # ルートの処理は Playwright が待っている間に走る(time.sleep では進まない)
            if posts:
                break
            pg.wait_for_timeout(100)
        p = posts[0]["params"]
        assert posts[0]["kind"] == "eval" and p["units"] == [f"14:ffn:keyed:a1@{NAME}#zk_web"]          # 既定の鍵を埋めて送る
        assert (p["keyed_metrics"], p["keyed_contrast"], p["keyed_subset"], p["keyed_examples"], p["keyed_pseudo"], p["keyed_match"]) \
            == (True, zpath, "even", 8, 4, False)
        pg.uncheck("#evx-km")
        pg.click("#evx-go")
        for _ in range(100):
            if len(posts) > 1:
                break
            pg.wait_for_timeout(100)
        q = posts[1]["params"]
        assert q["keyed_metrics"] is False and "keyed_contrast" not in q and "keyed_pseudo" not in q
        # 結果の keyed のブロック
        pg.evaluate(f"(() => {{ const v = window.gne.evalView; v.data = {json.dumps(_fake_eval(KEYED_BLOCK))}; v.dose = 1; v.render(); }})()")
        rows = pg.evaluate("[...document.querySelectorAll('#evx-body .evx-kx-tab tbody tr')].map(tr => tr.className)")
        assert rows == ["best", "", "", "rnd", ""]                                         # keyed の行に印、除外は灰色
        txt = pg.evaluate("document.querySelector('#evx-body .evx-kx').textContent")
        assert "0.23" in txt and "26.300" in txt and "19.700" in txt and "β = 0〜8" in txt
        assert ("超える" if lang == "ja" else "beats") in txt and ("保留" if lang == "ja" else "held out") in txt
        head = pg.evaluate("[...document.querySelectorAll('#evx-body table.evx:not(.evx-kx-tab) thead th')].map(th => th.textContent)")
        assert ("keyed 選択性" if lang == "ja" else "keyed sel.") in head
        cell = pg.evaluate("[...document.querySelectorAll('#evx-body tr[data-dose]')][1].textContent")
        assert "26.30" in cell
        if lang == "ja":
            pg.evaluate("document.querySelector('#evx-body .evx-kx').scrollIntoView()")
            _shot(pg, "keyed_eval.png")
        pg.click("#evx-body [data-krc='0']")                                                # keyed の方式のレシピを recipe タブへ
        assert pg.evaluate("window.gne.recipe.keyed().length") == 1 and pg.evaluate("window.gne.recipe.keyed()[0].key") == "zk_web"
        # 測っていない(対照集合が分からない)
        skipped = {"skipped": "鍵の対照集合が分かりません", "hint": "--keyed-contrast で渡してください"}
        pg.evaluate(f"(() => {{ const v = window.gne.evalView; v.data = {json.dumps(_fake_eval(skipped))}; v.render(); }})()")
        txt = pg.evaluate("document.querySelector('#evx-body .evx-kx').textContent")
        assert "鍵の対照集合が分かりません" in txt and "--keyed-contrast" in txt
        if lang == "en":
            missing = pg.evaluate("[...window.gneI18n.missing]")
            assert missing == [], missing
        assert not errors, errors
        pg.close()
