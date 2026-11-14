"""CHANGE-07 のテスト(UI のパッケージ): 鍵つきの書き込み(keyed write)の Web UI。

サーバ(モデルあり、Qwen3-0.6B、GNE_TEST_MODEL。CPU で回すときは GNE_TEST_NGL=0): 対照集合の候補(/api/contrasts)と保存した
鍵(/api/keys・/api/keys/<名前>・/api/model の keys)、ジョブ direction_key(語尾の対照集合 configs/contrast/zundamon_endings.yaml
と与えた返事)と direction_key_check(値とヒストグラムを落とさない)、keyed のユニットと default_key・default_direction の
レシピの保存と読み直し(省いた方向・鍵は既定で埋め、埋めたことを *_default で返す)、keyed の書き出し(headproj_export の
ジョブで LoRA と F16 の焼き込み、cvec の書き出しは keyed を外した警告)。
ブラウザ(Playwright の Chromium、無ければ飛ばす): direction ビューの「鍵(keyed)」で鍵を作り、層ごとの表を見て、key-check の
表と層ごとのヒストグラム(SVG)が描かれる。行の「+ レシピ」で keyed のユニットがレシピに入り、保存して読み直しても残る。英語の文言もある。
鍵のタブの「JSON を書く」: ひな形(configs/keyed/)を選ぶと入力欄に入り、書き換えた入力欄を置き換える前に確かめる。「方向から key-check
まで」(direction_key_pipeline のジョブ)で方向・鍵・key-check を続けて回し、書き込む方向に作ったばかりの方向が選ばれる。行の
「key-verify」(direction_key_verify のジョブ)。専門家の選ばれ方の表から専門家の鍵づくりの欄に入る(MoE の層は与えて描く)。
保存した鍵の一覧の「見る」は、先に押した鍵の読み込みが遅れて届いても、後から開いた鍵の表を上書きしない。
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
    """鍵 zundamon_endings(作り方 (b): 与えた返事の「です。」「ます。」を書く直前、L8・L14、比較用の鍵 8 本)をジョブで作る。"""
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

def _recipe_tab(pg, tab="units"):
    """右のレシピのパネルを開いて、タブを選ぶ(units 中身 / try 試す / out 書き出す)。中身は保存したレシピの一覧も取り直す。"""
    if pg.evaluate("document.querySelector('#recipe').classList.contains('collapsed')"):
        pg.click("#recipe .toggle")
    pg.click(f"#recipe-tabs button[data-rtab={tab}]")


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
    pg.click("#views button[data-view=keys]")
    pg.wait_for_function("document.querySelector('#kx-contrast option') && window.gne.directionView.key.loaded", timeout=30000)
    # before:<re> の対照集合(一覧で最初のもの)が既定。before:<re> のものは複数あるので、この先で使う NAME を選び直す
    assert pg.evaluate("document.querySelector('#kx-contrast').selectedOptions[0].textContent").endswith(" · before")
    zpath = pg.evaluate(f"[...document.querySelectorAll('#kx-contrast option')].map(o => o.value).find(v => v.endsWith('{NAME}.yaml'))")
    pg.select_option("#kx-contrast", zpath)
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
    _recipe_tab(pg)
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
    # LoRA・焼き込みの欄: keyed だけなら既定の名前は -keyed、焼き込みの型の既定は auto(Q8_0 は元のまま、ほかは F16)
    _recipe_tab(pg, "out")
    pg.click("#rc-lora")
    pg.wait_for_selector(".rc-lx", timeout=5000)
    assert pg.evaluate("document.querySelector('#lx-file').placeholder") == "zk_web_r-keyed"
    pg.check("input[name=lx-kind][value=bake]")
    assert pg.evaluate("document.querySelector('#lx-requant').value") == "auto"
    _shot(pg, "keyed_recipe.png")
    # 読み直し: 下書きを消し、保存したレシピを読む
    pg.evaluate("window.gne.recipe.clear()")
    pg.reload()
    pg.wait_for_function("window.gne && window.gne.recipe && window.gne.arch", timeout=120000)
    pg.evaluate("document.querySelector('#list').classList.remove('collapsed')")
    _recipe_tab(pg)
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
    pg.click("#views button[data-view=keys]")
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
    _recipe_tab(pg)
    pg.wait_for_selector("#recipe-list tr.rc-keyed", timeout=10000)
    assert "key nope not found" in pg.evaluate("document.querySelector('#recipe-list tr.rc-keyed').textContent")
    _recipe_tab(pg, "out")
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
        {"method": "gain", "desc": "ニューロンの gain", "excluded": "β = 0〜8 で書き込みたい位置の par が keyed に近づきません"},
        {"method": "proj", "desc": "ヘッドの射影", "beta": 93.0, "matched": False, "a_par": 20.0, "b_par": 8.0, "b_norm": 52.7,
         "dt_norm": 40.0, "selectivity": 12.0, "b_over_a": 2.6}],
    "compare": {"b_norm_ratio": {"add_same_alpha": 0.23, "add": 0.26}, "dt_norm_ratio": {"add": 0.15},
                "b_over_a_ratio": {"add_same_alpha": 0.25, "add": 0.25, "proj": 0.1}, "keyed_smaller": True},
    "pseudo": {"n": 8, "p50": 10.0, "p95": 19.7, "real": 26.3, "beats": True, "null_pct": 1.0},
    "notes": ["proj: 書き込みたい位置の par を keyed に ±10% でそろえられませんでした"], "warnings": [], "cost": {"prefills": 218, "seconds": 28.0}}


@pytest.mark.model
def test_eval_keyed_params_and_block_in_browser(ui_server, browser):
    """eval ビュー: レシピに keyed があると keyed の選択性の欄が出て、ジョブの params に keyed_contrast・keyed_examples・
    keyed_pseudo・keyed_subset・keyed_match・keyed_metrics が載る。結果の keyed は方式の表(除外の理由・そろったか)・比・比較用の鍵・注、
    行の表に eval の選択性の列。測っていない(skipped)ときは理由とヒント。英語の文言もある。"""
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
        assert ("eval の選択性" if lang == "ja" else "eval selectivity") in head
        cell = pg.evaluate("[...document.querySelectorAll('#evx-body tr[data-dose]')][1].textContent")
        assert "26.30" in cell
        if lang == "ja":
            pg.evaluate("document.querySelector('#evx-body .evx-kx').scrollIntoView()")
            _shot(pg, "keyed_eval.png")
        asked = []
        pg.once("dialog", lambda dl: (asked.append(dl.message), dl.accept()))           # いまのレシピを置き換える前に確かめる
        pg.click("#evx-body [data-krc='0']")                                                # keyed の方式のレシピでレシピを置き換える
        assert asked and pg.evaluate("window.gne.recipe.keyed().length") == 1 and pg.evaluate("window.gne.recipe.keyed()[0].key") == "zk_web"
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


# --- ひな形と入力欄、方向から key-check まで、専門家の選ばれ方、key-verify ----------------------------------------------

def _key_tab(pg):
    pg.click("#views button[data-view=direction]")
    pg.wait_for_function("window.gne.directionView && window.gne.directionView.mode", timeout=60000)
    pg.click("#views button[data-view=keys]")
    pg.wait_for_function("window.gne.directionView.key.loaded && document.querySelector(\"#kx-tpl-sel option[value='sentence_endings.json']\")",
                         timeout=30000)


@pytest.mark.model
def test_key_panel_template_pipeline_and_verify_in_browser(ui_server, browser):
    """鍵のタブの「JSON を書く」: ひな形を選ぶと入力欄に入り(説明つき)、手で直せる。書き換えた入力欄を別のひな形で置き換える前に
    確かめ、書き換えていなければ確かめない。表の keyed.alpha が「+ レシピ」の α の既定になる。読めない JSON はボタンを押しても
    投げない。「方向から key-check まで」で方向(まだ無いので作る)→ 鍵 → 保留した例の key-check を 1 回のジョブで回し、鍵の表・
    key-check の表とヒストグラムが出て、書き込む方向は作ったばかりの方向が選ばれる。2 回目は方向を使い回すと出る。行の
    「key-verify」で V30 の表。ジョブのバーの名前。"""
    url, s = ui_server["url"], ui_server["session"]
    pg, errors = _open(browser, url, "ja")
    dialogs = []
    pg.on("dialog", lambda dl: dialogs.append(dl.message))                  # _open が受け入れる。ここでは数えるだけ
    _key_tab(pg)
    assert not pg.evaluate("document.querySelector('#kx-src-file').classList.contains('hidden')")         # 既定はファイルを選ぶ
    pg.select_option("#kx-tpl-sel", "sentence_endings.json")
    assert pg.evaluate("document.querySelector('#kx-src-file').classList.contains('hidden')")             # JSON を書くに替わる
    assert '"noda_endings"' in pg.input_value("#kx-text") and dialogs == []
    assert "「です。」「ます。」を書く直前" in pg.evaluate("document.querySelector('#kx-tpl-about').textContent")
    jc = pg.evaluate("document.querySelector('#kx-jcheck').textContent")
    assert "noda_endings" in jc and "質問 16 個" in jc and "作り方 (b)" in jc and "english 4 問" in jc
    assert "まだありません" in pg.evaluate("document.querySelector('#kx-dirinfo').textContent")
    assert pg.evaluate("document.querySelector('#kx-name').placeholder") == "noda_endings"
    # 手で直す → 別のひな形を選ぶと確かめる(受け入れる)。(a) の例の keyed.alpha −1 が「+ レシピ」の α に入る
    pg.fill("#kx-text", pg.input_value("#kx-text").replace('"noda_endings"', '"noda_web"'))
    assert "noda_web" in pg.evaluate("document.querySelector('#kx-jcheck').textContent")
    pg.select_option("#kx-tpl-sel", "system_state.json")
    assert len(dialogs) == 1 and '"polite_state"' in pg.input_value("#kx-text")
    assert "作り方 (a)" in pg.evaluate("document.querySelector('#kx-jcheck').textContent")
    pg.evaluate("(() => { const k = window.gne.directionView.key; k.built = {kind: 'saved', r: {name: 'tmp', target: 'ffn', layers: [8], kn: {8: 0.5}, info: {}}}; k.renderBuilt(); })()")
    assert pg.input_value("#kx-alpha") == "-1"
    pg.select_option("#kx-tpl-sel", "sentence_endings.json")                # 書き換えていないので確かめない
    assert len(dialogs) == 1 and '"noda_endings"' in pg.input_value("#kx-text")
    # 読めない JSON: エラーを出し、ボタンを押してもジョブを投げない
    good = pg.input_value("#kx-text")
    pg.fill("#kx-text", good.replace('"name": "noda_endings"', '"name": "a/b"'))
    assert pg.evaluate("document.querySelectorAll('#kx-jcheck .err').length") == 1
    pg.click("#kx-pipe-go")
    assert "直すところ" in pg.evaluate("document.querySelector('#kx-status').textContent")
    pg.fill("#kx-text", good)
    # 書き込む方向に別の方向を選んでおく(一括の後は作ったばかりの方向に替わる)
    pg.evaluate("(() => { const k = window.gne.directionView.key; k.renderBuilt(); })()")
    pg.select_option("#kx-dir", NAME)
    pg.fill("#kx-layers", "8,14")
    pg.fill("#kx-pseudo", "8")
    pg.fill("#kx-mt", "24")
    pg.click(".kx-adv summary")
    pg.fill("#kx-replies", ui_server["replies"])
    pg.fill("#kx-rt", "16")
    pg.click("#kx-pipe-go")
    pg.wait_for_selector("#kc-res table.kx-check tbody tr", timeout=600000)
    pg.wait_for_selector("#kx-build table.kx-tab tbody tr", timeout=10000)
    rows = pg.evaluate("[...document.querySelectorAll('#kx-build table.kx-tab tbody tr')].map(tr => tr.cells[0].textContent)")
    assert rows == ["L8", "L14"]
    pipe = pg.evaluate("document.querySelector('#kx-pipe').textContent")
    assert "noda_endings" in pipe and "作りました" in pipe and "subset odd" in pipe
    assert pg.input_value("#kx-dir") == "noda_endings" and pg.input_value("#kx-alpha") == "1"
    assert pg.input_value("#kc-key") == "noda_endings" and pg.input_value("#kc-subset") == "odd"
    assert pg.input_value("#kc-contrast").endswith("/contrasts/noda_endings.yaml")
    assert "保留した例で測りました" in pg.evaluate("document.querySelector('#kc-res').textContent")
    assert pg.evaluate("document.querySelectorAll('#kc-res svg.kx-hist').length") == 2
    assert "noda_endings" in pg.evaluate("document.querySelector('#kx-keys').textContent")
    assert "使い回します" in pg.evaluate("document.querySelector('#kx-dirinfo').textContent")
    y = yaml.safe_load((s.store.contrast_dir() / "noda_endings.yaml").read_text(encoding="utf-8"))
    assert "about" not in y and y["positions"] == "before:(です|ます)。"
    _shot(pg, "keyed_pipeline.png")
    # key-verify(L14、書き込む方向 noda_endings)
    pg.click("#kx-build [data-kverify='14']")
    pg.wait_for_selector("#kx-vres table.kx-verify tbody tr", timeout=300000)
    assert pg.evaluate("document.querySelectorAll('#kx-vres table.kx-verify tbody tr').length") == 2
    assert "合格(V30)" in pg.evaluate("document.querySelector('#kx-verify').textContent")
    bar = pg.evaluate("document.querySelector('#jobs-bar').textContent")
    assert "LoRA の確かめ(key-verify)" in bar or "方向から key-check まで" in bar
    names = pg.evaluate("import('/jobs.js').then((m) => [m.KIND.direction_key_experts, m.KIND.direction_key_verify, m.KIND.direction_key_pipeline])")
    assert names == ["専門家の選ばれ方", "LoRA の確かめ(key-verify)", "方向から key-check まで"]
    assert not errors, errors
    pg.close()


EXPERTS = {"contrast": "noda_endings", "mode": "same", "positions": "before:(です|ます)。", "positions_b": "not-before:(です|ます)。",
           "subset": "even", "replies": {"AB": "given"}, "iv": None, "notes": ["n"], "seconds": 2.0,
           "layers": [{"layer": 3, "n_expert": 32, "n_used": 8, "positions_a": 15, "positions_b": 700, "base_rate": 0.25,
                       "top": [{"expert": 5, "rate_a": 1.0, "rate_b": 0.32, "diff": 0.68, "lift": 3.1, "weight_a": 0.21, "rows_a": 15.0,
                                "rate_dt": {"english": 0.1}},
                               {"expert": 9, "rate_a": 0.9, "rate_b": 0.0, "diff": 0.9, "lift": None, "weight_a": None, "rows_a": 13.5,
                                "rate_dt": {}}]}]}
VERIFY = {"layer": 3, "kind": "expert", "expert": 5, "tensor": "blk.3.ffn_down_exps.weight", "key": "k_L3E5", "direction": "noda_endings",
          "direction_layer": "own", "in_name": "x", "out_name": "y", "d_norm": 1.5, "direction_norm": 2.0, "post_norm": False, "kn": 0.4,
          "leak_b_key_examples": 0.01, "judged_leak": True, "v30": False, "gate": 0.21, "note": "専門家 5 は texts で一度も選ばれなかったので測れません",
          "rows": [{"alpha": 1.0, "adapter_scale": 1.0, "n_tokens": 0, "n_big": 0, "kx_threshold": 0.0, "kx_max": None, "rel_err_median": None,
                    "rel_err_p95": None, "cos_median": None, "leak_b": None, "leak_b_source": None, "leak_b_token_median": None, "ok": False,
                    "other_max": 0.0, "selection_changed": 0}]}


@pytest.mark.model
def test_key_panel_experts_verify_in_english(ui_server, browser):
    """英語の画面: ひな形(場面の例、説明に starts)と入力欄の要約・エラー、MoE の層が無ければ専門家のボタンは隠れ、MoE の層を
    与えると出る。層の欄に MoE の層が無ければ投げない。専門家の選ばれ方の表(rate_a・rate_b・差・lift・重み・行の見込み)の
    「この専門家で鍵を作る」で、書き込み先 expert:e・層 L・名前が鍵づくりの欄に入る。key-verify の専門家の表と注、一括のまとめ、
    方向の欄の説明(使い回す・作り直す)の文言が辞書にある。ジョブのバーの名前も英語。"""
    url = ui_server["url"]
    pg, errors = _open(browser, url, "en")
    _key_tab(pg)
    assert pg.evaluate("document.querySelector('#kx-exp-go').classList.contains('hidden')")              # Qwen3-0.6B は dense
    pg.select_option("#kx-tpl-sel", "one_scene.json")
    assert "starts" in pg.evaluate("document.querySelector('#kx-tpl-about').textContent")
    assert "tsundere_scene" in pg.evaluate("document.querySelector('#kx-jcheck').textContent")
    txt = pg.input_value("#kx-text")
    pg.fill("#kx-text", txt.replace('"positions": "before:', '"positions": "bogus:'))
    assert "must be response" in pg.evaluate("document.querySelector('#kx-jcheck').textContent")
    spec = json.loads(txt)
    spec.pop("dont_touch")
    spec["user_prompts"] = spec["user_prompts"][:6]
    spec["keyed"] = {"nope": 1}
    pg.fill("#kx-text", json.dumps(spec, ensure_ascii=False))
    jc = pg.evaluate("document.querySelector('#kx-jcheck').textContent")
    assert "No dont_touch" in jc and "Only 6 questions" in jc and "keyed cannot contain nope" in jc
    spec.pop("keyed")
    pg.fill("#kx-text", json.dumps(spec, ensure_ascii=False))
    # 専門家: MoE の層を与えてボタンを出す。層の欄に MoE の層が無ければ投げない
    pg.evaluate("(() => { const k = window.gne.directionView.key; k.moe = new Set([3]); k.$('#kx-exp-go').classList.remove('hidden'); })()")
    pg.fill("#kx-layers", "1")
    pg.click("#kx-exp-go")
    assert "no MoE layer" in pg.evaluate("document.querySelector('#kx-estatus').textContent")
    pg.evaluate(f"(() => {{ const k = window.gne.directionView.key; k.experts = {json.dumps(EXPERTS)}; k.renderExperts(); }})()")
    head = pg.evaluate("[...document.querySelectorAll('#kx-eres table.kx-exps thead th')].map(th => th.textContent)")
    assert head[:7] == ["expert", "rate_a", "rate_b", "diff", "lift", "weight", "expected rows"]
    cells = pg.evaluate("[...document.querySelectorAll('#kx-eres table.kx-exps tbody tr')][0].textContent")
    assert "E5" in cells and "100%" in cells and "32%" in cells and "3.10×" in cells and "15.0" in cells
    pg.click("#kx-eres [data-kexp='3:5']")
    assert pg.input_value("#kx-target") == "expert" and pg.input_value("#kx-expert") == "5" and pg.input_value("#kx-layers") == "3"
    assert pg.input_value("#kx-name") == "tsundere_scene_L3E5"
    assert "expert 5 at L3" in pg.evaluate("document.querySelector('#kx-status').textContent")
    # key-verify(専門家の鍵、測れなかった注)と一括のまとめ、方向の欄
    pg.evaluate(f"(() => {{ const k = window.gne.directionView.key; k.built = {{kind: 'saved', r: {{name: 'k_L3E5', target: 'expert:5', layers: [3], kn: {{3: 0.4}}, info: {{}}}}}};"
                f" k.renderBuilt(); k.verified = {json.dumps(VERIFY)}; k.renderVerify();"
                " k.pipeResult = {name: 'k_L3E5', direction: {name: 'tsundere_scene', reused: true}, key_seconds: 1, check: {subset: 'odd'}, check_seconds: 2, seconds: 3,"
                " warnings: ['w']}; k.renderPipe(); })()")
    vt = pg.evaluate("document.querySelector('#kx-verify').textContent")
    assert "failed (V30)" in vt and "other experts' slots" in vt and "専門家 5" in vt
    assert "reused" in pg.evaluate("document.querySelector('#kx-pipe').textContent")
    assert "does not exist yet" in pg.evaluate("document.querySelector('#kx-dirinfo').textContent")
    pg.evaluate("(() => { const k = window.gne.directionView.key; k.dirs = [...k.dirs, {name: 'tsundere_scene', positions: 'last'}]; k.dirInfo(); })()")
    di = pg.evaluate("document.querySelector('#kx-dirinfo').textContent")
    assert "will be reused" in di and "built with positions last" in di
    pg.check("#kx-rebuild")
    assert "will be rebuilt" in pg.evaluate("document.querySelector('#kx-dirinfo').textContent")
    names = pg.evaluate("import('/jobs.js').then((m) => [m.KIND.direction_key_experts, m.KIND.direction_key_verify, m.KIND.direction_key_pipeline])")
    assert names == ["expert usage", "LoRA check (key-verify)", "Direction to key-check"]
    pg.click("#kx-src button[data-src=file]")
    assert not pg.evaluate("document.querySelector('#kx-src-file').classList.contains('hidden')")
    missing = pg.evaluate("[...window.gneI18n.missing]")
    assert missing == [], missing
    assert not errors, errors
    pg.close()


# key-check の結果(返事の冒頭での ρ の表示を見るためのもの)。位置 0 は低く、位置 1・2 で高い群と、例が 1 つしかない位置
START_CHECK = {"key": "k_sys", "contrast": "polite_state", "subset": "odd", "mode": "system", "positions": "response", "held_out": True,
               "overlap": {"built_subset": "even", "subset": "odd"}, "counts": {}, "n_pseudo": 16, "seconds": 1.0, "best_layer": 8,
               "all_beat_pseudo": True,
               "layers": [{"layer": 8, "auc": 0.8, "auc_token": 0.6, "selectivity": 0.3, "pseudo_selectivity": {"p95": 0.2}, "beats_pseudo": True,
                           "dont_touch_median": {"english": 0.19}, "kn": 0.2,
                           "start_rho": {"B": {"median": [0.11, 0.2, 0.74], "max": [0.3, 0.5, 0.74], "n": [2, 2, 1]},
                                         "dont_touch": {"english": {"median": [-0.05, 1.23, 0.9], "max": [0.1, 1.5, 1.2], "n": [2, 2, 2]},
                                                        "few": {"median": [0.5, None, None], "max": [0.5, None, None], "n": [1, 0, 0]}}}}]}


@pytest.mark.model
def test_key_check_start_rho_cell_and_matched_label(ui_server, browser):
    """key-check の表の、返事の冒頭での ρ(dont_touch の中央値の欄の下の段): 位置 0〜2 のうち中央値が最大の位置とその値を出し、
    例 2 つ以上で 0.4 以上なら赤(サーバの警告 keyed_build._start_hot と同じ決まり)。位置 0 だけを見ると、位置 1・2 で反応して
    警告が出ている群が赤にならなかった(作り方 (a) のひな形 system_state を Qwen3-0.6B で回して見つけた)。鍵の表の
    「正規表現が一致した例」は作り方 (b)(mode same)だけに出す。英語の文言もある。"""
    url = ui_server["url"]
    for lang in ("ja", "en"):
        pg, errors = _open(browser, url, lang)
        _key_tab(pg)
        pg.evaluate(f"(() => {{ const k = window.gne.directionView.key; k.checked = {json.dumps(START_CHECK)}; k.renderCheck(); }})()")
        cell = "document.querySelectorAll('#kc-res table.kx-check tbody tr')[0].cells[9]"
        spans = pg.evaluate(f"[...{cell}.querySelectorAll('span:not(.muted)')].map(s => [s.textContent, s.classList.contains('kx-ng')])")
        if lang == "ja":
            assert spans == [["書き込みたくない位置 0.20(位置 1)", False],      # 例 1 つの位置 2(0.74)は選ばない
                             ["english 1.23(位置 1)", True],                  # 位置 0 は −0.05 でも、位置 1 で反応している
                             ["few 0.50(位置 0)", False]], spans                # 例 2 つ以上の位置が無ければ全部から選ぶが赤にしない
        else:
            assert spans == [["positions to leave alone 0.20 (position 1)", False], ["english 1.23 (position 1)", True],
                             ["few 0.50 (position 0)", False]], spans
        # 鍵の表: 作り方 (a)(mode system)には「正規表現が一致した例」を出さない。(b)(mode same)には出す
        built = {"name": "k_sys", "target": "ffn", "mode": "system", "positions": "response", "q_mode": "examples", "subset": "even",
                 "layers": [{"layer": 8, "kn": 0.2, "gain": 5.0, "q_rank": 13, "build_auc": 1.0}], "matched": {"A": 10, "of": 10},
                 "counts": {"a": 10, "b": 10}, "n_pseudo": 16, "seconds": 2.0}
        for mode, shown in (("system", False), ("same", True)):
            pg.evaluate(f"(() => {{ const k = window.gne.directionView.key; k.built = {{kind: 'build', params: {{}}, r: {json.dumps({**built, 'mode': mode})}}}; k.renderBuilt(); }})()")
            head = pg.evaluate("document.querySelector('#kx-build .kx-res p.note').textContent")
            assert (("正規表現が一致した例 10 / 10" if lang == "ja" else "regex matched in 10 / 10 examples") in head) is shown, (mode, head)
        if lang == "en":
            missing = pg.evaluate("[...window.gneI18n.missing]")
            assert missing == [], missing
        assert not errors, errors
        pg.close()


# 返事を書き始める直前の位置を Q に入れて作った鍵の key-check(dont_touch の群のこの位置を測ったもの)
DTS_CHECK = {**START_CHECK, "mode": "same", "positions": "before:(?s)\\A.",
             "counts": {"A": {"examples": 4, "rows": 4, "of": 4}, "B": {"examples": 4, "rows": 60, "of": 4},
                        "dt:english": {"examples": 2, "rows": 30, "of": 2}, "dts:english": {"examples": 2, "rows": 2, "of": 2},
                        "dt:tasks": {"examples": 1, "rows": 9, "of": 1}, "dts:tasks": {"examples": 1, "rows": 1, "of": 1}},
             "dont_touch_start": {"on": True, "recorded": True, "positions": "last", "groups": {"english": 2, "tasks": 1}, "rows": 3},
             "layers": [{**START_CHECK["layers"][0], "dont_touch_median": {"english": 0.19, "tasks": 0.02},
                         "dont_touch_start_median": {"english": 0.62, "tasks": 0.9},
                         "rho": {"dont_touch_start": {"english": {"n": 2, "median": 0.62}, "tasks": {"n": 1, "median": 0.9}}}}]}


@pytest.mark.model
def test_key_check_dont_touch_reply_start_cell(ui_server, browser):
    """key-check の表の dont_touch の中央値の欄に、返事を書き始める直前の位置での ρ の中央値(鍵の作成でこの位置を Q に入れた鍵
    だけ)を括弧で出し、例 2 つ以上で 0.4 以上なら赤(返事の冒頭の警告と同じ決まり)。数の行では dts:<群> を「書き始める直前」と
    書く。この位置を測っていない結果(START_CHECK)には出さない。英語の文言もある。"""
    url = ui_server["url"]
    for lang in ("ja", "en"):
        pg, errors = _open(browser, url, lang)
        _key_tab(pg)
        pg.evaluate(f"(() => {{ const k = window.gne.directionView.key; k.checked = {json.dumps(DTS_CHECK)}; k.renderCheck(); }})()")
        cell = "document.querySelectorAll('#kc-res table.kx-check tbody tr')[0].cells[9]"
        full = pg.evaluate(f"{cell}.textContent")
        text = full.split("返事の冒頭" if lang == "ja" else "start of the reply")[0]       # 下の段(返事の冒頭)の前まで
        reds = pg.evaluate(f"[...{cell}.querySelectorAll('.kx-ng')].map(s => s.textContent)")
        head = pg.evaluate("document.querySelector('#kc-res .kx-res p.note').textContent")
        if lang == "ja":
            assert text == "english 0.19(書き始める直前 0.62) · tasks 0.02(書き始める直前 0.90)", full
            assert "dt english(書き始める直前) 2/2" in head and "dt english 2/2" in head
        else:
            assert text == "english 0.19 (just before the reply 0.62) · tasks 0.02 (just before the reply 0.90)", text
            assert "dt english (just before the reply) 2/2" in head
        assert "0.62" in reds and "0.90" not in reds                                # tasks は例が 1 つなので赤にしない
        pg.evaluate(f"(() => {{ const k = window.gne.directionView.key; k.checked = {json.dumps(START_CHECK)}; k.renderCheck(); }})()")
        assert "書き始める直前" not in pg.evaluate(f"{cell}.textContent") and "just before" not in pg.evaluate(f"{cell}.textContent")
        if lang == "en":
            missing = pg.evaluate("[...window.gneI18n.missing]")
            assert missing == [], missing
        assert not errors, errors
        pg.close()


def _saved_key(name):
    """/api/keys/<名前> の形の、小さな保存した鍵(層 8・14)。"""
    return {"name": name, "target": "ffn", "layers": [8, 14], "kn": {"8": 0.3, "14": 0.4}, "gain": {"8": 3.3, "14": 2.5},
            "q_rank": {"8": 5, "14": 6}, "positions": "before:(です|ます)。", "q_mode": "examples", "subset": "even",
            "counts": {"a": 10, "b": 100}, "n_pseudo": 16, "info": {"mode": "same", "build_auc": {"8": 0.9, "14": 0.95}, "weak": [], "dropped": {}}}


@pytest.mark.model
def test_key_view_late_load_does_not_overwrite(ui_server, browser):
    """保存した鍵の一覧の「見る」: 先に押した鍵の読み込みが遅れて届いても、その間に開いた別の鍵の表を上書きしない
    (eval ビューと同じく、遅れて届いた古い読み込みは捨てる)。鍵の一覧と中身はページの側で返し、先に押した鍵の中身だけ止めておく。"""
    url = ui_server["url"]
    keys = {"kview_a": _saved_key("kview_a"), "kview_b": _saved_key("kview_b")}
    listing = [{k: v for k, v in x.items() if k not in ("gain", "q_rank", "info")} for x in keys.values()]
    held = []

    def on_keys(route, req):
        name = req.url.split("/api/keys", 1)[1].split("?")[0].lstrip("/")
        if not name:
            route.fulfill(status=200, content_type="application/json", body=json.dumps(listing))
        elif name == "kview_a" and not held:
            held.append(route)
        else:
            route.fulfill(status=200, content_type="application/json", body=json.dumps(keys[name]))
    pg, errors = _open(browser, url, "ja")
    pg.route("**/api/keys**", on_keys)
    _key_tab(pg)
    pg.wait_for_selector("#kx-keys [data-kview=kview_b]", timeout=10000)
    pg.click("#kx-keys [data-kview=kview_a]")
    for _ in range(400):
        if held:
            break
        pg.wait_for_timeout(25)
    assert held
    pg.click("#kx-keys [data-kview=kview_b]")
    pg.wait_for_function("window.gne.directionView.key.built && window.gne.directionView.key.built.r.name === 'kview_b'", timeout=10000)
    with pg.expect_response("**/api/keys/kview_a"):
        held[0].fulfill(status=200, content_type="application/json", body=json.dumps(keys["kview_a"]))
    pg.wait_for_timeout(300)                                                         # 届いた中身をページが読み終えるまで
    assert pg.evaluate("window.gne.directionView.key.built.r.name") == "kview_b"
    head = pg.evaluate("document.querySelector('#kx-build .kx-res p.note').textContent")
    assert "kview_b" in head and "kview_a" not in head
    assert not errors, errors
    pg.close()
