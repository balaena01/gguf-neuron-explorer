"""FiSH の画面の新しい欄のテスト: キーのセルに会話の形を入れる(key_conv_cells)・前のやり取りの応答の上限(prior_max_tokens)・
ユニットの書き方(!kauto・取り除く形 rm / rmbase)・方向のもと(対照集合の差 / トークンの行 token・token_delta)と層の走査の表・
確かめの会話の形ごとの表(by_form)・キーの表の会話の形ごとの数。

モデルあり(Qwen3-0.6B、GNE_TEST_MODEL。CPU で回すときは GNE_TEST_NGL=0): /api/fish/contrast(入力欄の表をファイルにする)と、
画面の「層を走査する」から投げる direction_token_scan のジョブ(1 層。生成しない)。ほかのジョブは投げずに、引数を受け取って確かめる
(結果の表は、結果をページに入れて描く)。ブラウザは Playwright の Chromium(無ければ飛ばす)。ひな形の値が欄に入り、欄を変えると
ジョブの引数・/api/fish/units の引数に出る。英語の画面で未翻訳が無い。GNE_TEST_SHOTS を渡すと、画面の写し fish-new-*.png を置く。
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
SHOTS = os.environ.get("GNE_TEST_SHOTS")          # 画面の写しを置くフォルダ(任意)

CONTRAST = {
    "name": "fish_new_ui",
    "system_a": "You are a pirate. Always answer in rough pirate speech.",
    "user_prompts": ["Thank you for your help yesterday.", "I really like your drawing.", "Thanks for lending me your notebook.",
                     "Your cooking was delicious.", "Thank you for listening to me.", "You look great today."],
    "dont_touch": [{"name": "tasks", "prompts": ["What is 12 times 8?", "What is the capital of France?",
                                                 "How many days are in a week?", "What color is the sky?"]}],
}
# ひな形の fish に、新しい設定を全部書いたもの
NEW_FISH = {"segments": "0", "form": "cap", "alpha": 1, "key_fit": "ridge", "key_conv_cells": True, "prior_max_tokens": 64,
            "kscale": "auto", "remove": "rmbase", "token": "<|im_end|>", "token_delta": 24, "max_tokens": 16}

# 層の走査の結果(token_scan の形。L13 は届かない層)
SCAN = {"contrast": "fish_new_ui", "tokens": [151645], "token_text": ["<|im_end|>"], "minus": [], "rbar": 81.25, "rbar_from": "contrast",
        "gdot": 2.5, "need": 0.95, "need_n": 6, "deltas": [2, 4, 8, 16, 32, 64], "leaks": [0.2, 0.5], "open_n": 6, "close_n": 4,
        "base_open_top1": 0, "base_close_top1": 0, "seconds": 3.2,
        "rows": [{"layer": 12, "need_delta": 32.0, "beta": 1040.0, "open_hit": 6, "open_n": 6, "tried": {},
                  "leak": {"0.2": {"top1": 0, "kl": 0.0123, "changed": 1, "n": 4}, "0.5": {"top1": 2, "kl": 0.3456, "changed": 3, "n": 4}}},
                 {"layer": 13, "need_delta": None, "beta": None, "open_hit": None, "open_n": 6, "tried": {}, "leak": {}},
                 {"layer": 14, "need_delta": 16.0, "beta": 520.0, "open_hit": 6, "open_n": 6, "tried": {},
                  "leak": {"0.2": {"top1": 0, "kl": 0.002, "changed": 0, "n": 4}, "0.5": {"top1": 0, "kl": 0.04, "changed": 0, "n": 4}}}]}

# 足した量の結果(gne nw check の形)のうち、会話の形ごとの表に要るところ。tasks は形が 1 つだけ(表を出さない)
CHECK = {"contrast": "fish_new_ui", "positions": None, "key_fit": {"fit": "ridge", "c": {"b_p0": 1}},
         "question_summary": {"low": 0.25, "over": 1.5, "leak": 0.2, "segments": ["0"], "cap": [[0.4, 0.8]], "n_low": 0, "worst": []},
         "groups": {
             "scene": {"n": 6, "loops": 0, "units": [], "finished": 5,
                       "by_form": {"single": {"n": 4, "segments": {"0": {"n": 4, "rho_ge_b": 0.75, "rho_gt_a": 1.0, "amount_mean": 0.9}}},
                                   "multi_turn": {"n": 2, "segments": {"0": {"n": 2, "rho_ge_b": 0.5, "rho_gt_a": 0.5, "amount_mean": 0.4}}}}},
             "dont_touch:ordinary": {"n": 5, "loops": 0, "units": [], "finished": 4,
                                     "by_form": {"single": {"n": 3, "segments": {"0": {"n": 3, "leak": 0, "total_mean": 0.01}}},
                                                 "system": {"n": 2, "segments": {"0": {"n": 2, "leak": 1, "total_mean": 0.3}}}}},
             "dont_touch:tasks": {"n": 4, "loops": 0, "units": [], "finished": 4,
                                  "by_form": {"single": {"n": 4, "segments": {"0": {"n": 4, "leak": 0, "total_mean": 0.0}}}}}}}


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
    """Session と、TestClient・ブラウザ用のサーバ(同じアプリ)。ジョブは層の走査だけを投げる。"""
    if not MODEL.exists():
        pytest.skip(f"{MODEL} がありません")
    import uvicorn
    from fastapi.testclient import TestClient

    from gne.api import Session
    from gne.ui.server import create_app
    s = Session(MODEL, root=tmp_path_factory.mktemp("fishnew") / "root", n_gpu_layers=NGL, n_ctx=1024)
    app = create_app(s, warm=False)
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield {"client": TestClient(app), "session": s, "url": f"http://127.0.0.1:{port}"}
    srv.should_exit = True
    th.join(5)
    _close_app(app, s)


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


def _shot(pg, name, selector=None):
    if SHOTS:
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        if selector:
            pg.locator(selector).screenshot(path=str(Path(SHOTS) / name))
        else:
            pg.screenshot(path=str(Path(SHOTS) / name), full_page=False)


def _open(browser, url, lang):
    pg = browser.new_page(viewport={"width": 1600, "height": 1100}, locale="ja-JP")
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.add_init_script("addEventListener('DOMContentLoaded', () => { window.gneAsk = () => true; })")
    pg.goto(url + f"/?lang={lang}")
    pg.wait_for_function("window.gne && window.gne.recipe && window.gne.arch", timeout=120000)
    pg.evaluate("try { localStorage.removeItem('gne.fish.draft'); localStorage.removeItem('gne.fish.step'); } catch {}")
    pg.click("#views button[data-view=direction]")
    pg.wait_for_function("window.gne.directionView && window.gne.directionView.mode", timeout=60000)
    pg.click("#views button[data-view=fish]")
    pg.wait_for_function("window.gne.directionView.fish.loaded", timeout=30000)
    return pg, errors


def _step(pg, step):
    pg.evaluate("""(step) => { const b = document.querySelector(`#dv-fish .steps [role=tab][data-step="${step}"]`);
      if (b && b.getAttribute("aria-selected") !== "true") b.click(); }""", step)


F = "window.gne.directionView.fish"


def _insert(pg, fish, name="t_new.json"):
    """ひな形(fish の設定つき)を一覧に足して入れる。入力欄の JSON になり、fish の値が欄に入る。"""
    spec = {**CONTRAST, "fish": fish}
    _step(pg, "src")
    pg.click("#fx-src button[data-src=json]")
    pg.evaluate(f"""async ([file, spec]) => {{ const p = {F};
      p.templates = p.templates.filter((x) => x.file !== file);
      p.templates.push({{ file, order: 99, ja: {{ title: 't', text: 'テスト用' }}, en: {{ title: 't', text: 'for tests' }}, spec }});
      await p.insertTemplate(file, false); }}""", [name, spec])
    pg.evaluate("document.querySelector('#fx-shared details.kx-adv').open = true")          # 細かい設定(前のやり取りの応答の上限)を開く


def _capture(pg):
    """ジョブを投げずに、引数だけを受け取る(onRun が null を返すと「やめました」になる)。"""
    pg.evaluate(f"() => {{ const p = {F}; window.__runs = []; p.onRun = async (kind, params) => {{ window.__runs.push({{ kind, params }}); return null; }}; }}")


def _runs(pg, clear=True):
    return pg.evaluate("(c) => { const r = [...(window.__runs || [])]; if (c) window.__runs = []; return r; }", clear)


def _vals(pg):
    return pg.evaluate("""() => { const q = (s) => document.querySelector(s);
      return { keyfit: q('#fx-keyfit').value, conv: q('#fx-convcells').checked, convOff: q('#fx-convcells').disabled, prior: q('#fx-prior-mt').value,
               kauto: q('#fx-kauto').checked, uform: q('#fx-uform').value, src: q('input[name=fx-dirsrc]:checked').value,
               token: q('#fx-token').value, tokenOff: q('#fx-token').disabled, delta: q('#fx-tdelta').value, scanBox: !q('#fx-scan-box').hidden,
               methodOff: q('#fx-method').disabled, fastOff: q('#fx-fast').disabled }; }""")


# --- サーバ ----------------------------------------------------------------------------------------------

@pytest.mark.model
def test_contrast_endpoint(served):
    """入力欄の表 → <root>/contrasts/<name>.yaml(fish は入れない)。読めない表は 400。"""
    c, s = served["client"], served["session"]
    r = c.post("/api/fish/contrast", json={"spec": {**CONTRAST, "fish": NEW_FISH}})
    assert r.status_code == 200, r.text
    x = r.json()
    p = Path(x["path"])
    assert x["name"] == "fish_new_ui" and p == s.store.contrast_dir() / "fish_new_ui.yaml" and p.exists()
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert "fish" not in d and d["user_prompts"] == CONTRAST["user_prompts"] and d["dont_touch"][0]["name"] == "tasks"
    bad = c.post("/api/fish/contrast", json={"spec": {"name": "bad name!", "user_prompts": []}})
    assert bad.status_code == 400


# --- ブラウザ --------------------------------------------------------------------------------------------

@pytest.mark.model
def test_new_controls_roundtrip(served, browser):
    """ひな形の新しい設定が欄に入り、欄の値がジョブの引数に出る(切り替えたら、表の値より欄の値)。"""
    pg, errors = _open(browser, served["url"], "ja")
    _insert(pg, NEW_FISH)
    v = _vals(pg)
    assert (v["keyfit"], v["conv"], v["convOff"], v["prior"], v["kauto"], v["uform"]) == ("ridge", True, False, "64", True, "rmbase"), v
    assert (v["src"], v["token"], v["tokenOff"], v["delta"], v["scanBox"], v["methodOff"], v["fastOff"]) == ("token", "<|im_end|>", False, "24", True, True, True), v
    assert not pg.evaluate("document.querySelectorAll('#fx-jcheck .err').length")            # 新しい設定も fish に書ける
    # 欄が出る手順: ユニットの書き方はレシピを組む・まとめて作る、方向のもとは方向を作る・まとめて作る
    shown = {}
    for st in ("dir", "keys", "units", "all"):
        _step(pg, st)
        shown[st] = pg.evaluate("['#fx-keyset', '#fx-unitset', '#fx-dirset'].map((s) => !document.querySelector(s).hidden)")
    assert shown == {"dir": [False, False, True], "keys": [True, False, False], "units": [True, True, False], "all": [True, True, True]}
    _capture(pg)
    pg.fill("#fx-layers", "8")
    pg.evaluate(f"{F}.runAll()")
    pg.wait_for_function("(window.__runs || []).length === 1", timeout=5000)
    run = _runs(pg)[0]
    assert run["kind"] == "fish_pipeline"
    p = run["params"]
    assert p["spec"]["fish"] == NEW_FISH and p["layers"] == "8" and p["segments"] == "0"
    assert (p["key_fit"], p["key_conv_cells"], p["prior_max_tokens"], p["kscale"], p["remove"], p["token"], p["token_delta"]) \
        == ("ridge", True, 64, "auto", "rmbase", "<|im_end|>", 24)
    pg.evaluate(f"{F}.runKeys(false)")
    pg.evaluate(f"{F}.runDirection()")
    pg.wait_for_function("(window.__runs || []).length === 2", timeout=5000)
    keys, dirn = _runs(pg)
    assert keys["kind"] == "fish_keys" and keys["params"]["key_conv_cells"] is True and keys["params"]["prior_max_tokens"] == 64
    assert "kscale" not in keys["params"] and "remove" not in keys["params"] and "token" not in keys["params"]   # キーのジョブには渡さない
    assert dirn["kind"] == "fish_direction" and (dirn["params"]["token"], dirn["params"]["token_delta"], dirn["params"]["prior_max_tokens"]) == ("<|im_end|>", 24, 64)
    # 欄を切り替える: 切・足す・対照集合の差・上限は空 → 表に書いてあっても、空・切で渡す(上限は渡さない)
    _step(pg, "all")
    pg.uncheck("#fx-convcells")
    pg.uncheck("#fx-kauto")
    pg.select_option("#fx-uform", "add")
    pg.check("input[name=fx-dirsrc][value=contrast]")
    pg.fill("#fx-prior-mt", "")
    v = _vals(pg)
    assert v["tokenOff"] and not v["scanBox"] and not v["methodOff"] and not v["fastOff"]
    pg.evaluate(f"{F}.runAll()")
    pg.evaluate(f"{F}.runDirection()")
    pg.wait_for_function("(window.__runs || []).length === 2", timeout=5000)
    allp, dirp = [x["params"] for x in _runs(pg)]
    assert (allp["key_conv_cells"], allp["kscale"], allp["remove"], allp["token"]) == (False, "", "", "") and "token_delta" not in allp
    assert "prior_max_tokens" not in allp and dirp["token"] == "" and "token_delta" not in dirp
    # 鍵の作り方を戻すと、セルの欄は使えない(引数も切)。上限つきの形でなければ、取り除く形は選べない
    pg.check("#fx-convcells")
    pg.select_option("#fx-keyfit", "mean")
    assert pg.evaluate("document.querySelector('#fx-convcells').disabled")
    pg.evaluate(f"{F}.runKeys(false)")
    pg.wait_for_function("(window.__runs || []).length === 1", timeout=5000)
    assert _runs(pg)[0]["params"]["key_conv_cells"] is False
    pg.select_option("#fx-uform", "rm")
    pg.select_option("#fx-form", "one")
    assert pg.evaluate("[document.querySelector('#fx-uform').value, [...document.querySelectorAll('#fx-uform option')].map((o) => o.disabled)]") == ["add", [False, True, True]]
    assert "上限つき" in pg.evaluate("document.querySelector('#fx-uform-note').textContent")
    pg.select_option("#fx-form", "cap")
    assert pg.evaluate("document.querySelector('#fx-uform-note').textContent") == ""
    # トークンの行の欄が読めなければ投げない(理由が出る)
    pg.check("input[name=fx-dirsrc][value=token]")
    pg.fill("#fx-token", "")
    pg.evaluate(f"{F}.runDirection()")
    assert "トークンを入れてください" in pg.evaluate("document.querySelector('#fx-status-dir').textContent")
    assert pg.evaluate("[document.querySelector('#fx-token').getAttribute('aria-invalid'), document.activeElement.id]") == ["true", "fx-token"]
    pg.fill("#fx-token", "<|im_end|>")
    assert pg.evaluate("document.querySelector('#fx-token').getAttribute('aria-invalid')") is None     # 直すと印が消える
    pg.fill("#fx-tdelta", "0")
    pg.evaluate(f"{F}.runDirection()")
    assert "0 より大きい数" in pg.evaluate("document.querySelector('#fx-status-dir').textContent")
    pg.fill("#fx-tdelta", "24")
    pg.fill("#fx-usedir", "old_dir")
    pg.evaluate(f"{F}.runAll()")
    assert "一緒に使えません" in pg.evaluate("document.querySelector('#fx-status-all').textContent")
    pg.fill("#fx-usedir", "")
    pg.fill("#fx-prior-mt", "0")
    pg.evaluate(f"{F}.runAll()")
    assert "1 以上の整数" in pg.evaluate("document.querySelector('#fx-status-all').textContent")
    pg.fill("#fx-prior-mt", "")
    pg.fill("#fx-seg", "before:(です|ます)。")
    pg.evaluate(f"{F}.runDirection()")
    assert "before:" in pg.evaluate("document.querySelector('#fx-status-dir').textContent")
    pg.fill("#fx-seg", "0")
    assert not _runs(pg)
    # 空白で区切ったトークンは並びで渡す(行の平均)。kscale の数は画面で選べないことを知らせる
    pg.fill("#fx-token", "◇ <|im_end|>")
    pg.evaluate(f"{F}.runDirection()")
    pg.wait_for_function("(window.__runs || []).length === 1", timeout=5000)
    assert _runs(pg)[0]["params"]["token"] == ["◇", "<|im_end|>"]
    _insert(pg, {**NEW_FISH, "kscale": 4, "token": None, "remove": None}, "t_k4.json")
    v = _vals(pg)
    assert (v["kauto"], v["uform"], v["src"], v["token"]) == (False, "add", "contrast", "")
    assert "fish.kscale の数(4)" in pg.evaluate("document.querySelector('#fx-jcheck').textContent")
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_units_request_carries_unit_form(served, browser):
    """/api/fish/units の引数に kscale・remove が入り、組んだユニットに !kauto・!rmbase が付く(切れば付かない)。"""
    pg, errors = _open(browser, served["url"], "ja")
    _insert(pg, {**NEW_FISH, "token": None})
    pg.fill("#fx-layers", "8")
    pg.evaluate("""() => { window.__fx = []; const of = window.fetch;
      window.fetch = (u, o) => { if (String(u).includes('/api/fish/units')) window.__fx.push(JSON.parse(o.body)); return of(u, o); }; }""")
    pg.evaluate(f"""async () => {{ const p = {F}; p.st.dir = {{ name: 'dnew', segments: ['0'], layers: [8], rows: [] }};
      p.st.keys = {{ base: 'bnew', segments: ['0'], form: 'cap', layers: [8], rows: [] }}; p.st.keyBase = 'bnew';
      p.st.picked = {{ 8: [11, 12] }}; p.st.touched = new Set([8]); await p.compose(); }}""")
    pg.wait_for_function("document.querySelectorAll('#fx-unit-tab tbody tr').length === 1", timeout=10000)
    body = pg.evaluate("window.__fx[window.__fx.length - 1]")
    assert (body["kscale"], body["remove"], body["form"], body["direction"], body["base"]) == ("auto", "rmbase", "cap", "dnew", "bnew")
    assert pg.evaluate(f"{F}.st.units") == ["8:N11+N12:nw:a1@dnew.seg0#bnew_p0&bnew_one!kauto!rmbase"]
    _step(pg, "units")
    _shot(pg, "fish-new-units.png", "#dv-fish .stage")
    pg.select_option("#fx-uform", "rm")
    pg.wait_for_function(f"{F}.st.units.length === 1 && {F}.st.units[0].endsWith('!kauto!rm')", timeout=10000)
    pg.uncheck("#fx-kauto")
    pg.select_option("#fx-uform", "add")
    pg.wait_for_function(f"{F}.st.units.length === 1 && {F}.st.units[0].endsWith('&bnew_one')", timeout=10000)
    body = pg.evaluate("window.__fx[window.__fx.length - 1]")
    assert "kscale" not in body and "remove" not in body
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_scan_in_browser(served, browser):
    """方向のもと = トークンの行: 「層を走査する」で direction_token_scan のジョブ(入力欄の表はファイルにしてから渡す)。表の行で層と Δ が欄に入る。"""
    pg, errors = _open(browser, served["url"], "ja")
    _insert(pg, {**NEW_FISH, "token_delta": None})
    _step(pg, "dir")
    assert pg.evaluate("!document.querySelector('#fx-scan-box').hidden")
    pg.fill("#fx-scan-layers", "8")
    pg.click("#fx-scan-go")
    pg.wait_for_function("document.querySelector('#fx-status-scan').textContent.includes('できました') || document.querySelector('#fx-status-scan').classList.contains('err')",
                         timeout=600000)
    assert "できました" in pg.evaluate("document.querySelector('#fx-status-scan').textContent"), pg.evaluate("document.querySelector('#fx-status-scan').textContent")
    real = pg.evaluate(f"{F}.st.scan")
    assert real["contrast"] == "fish_new_ui" and [r["layer"] for r in real["rows"]] == [8] and real["open_n"] == 6 and real["close_n"] == 4
    assert real["token_text"] == ["<|im_end|>"]
    assert pg.evaluate("document.querySelectorAll('#fx-scan-tab tbody tr').length") == 1
    job = next(j for j in served["client"].get("/api/jobs").json() if j["kind"] == "direction_token_scan")
    assert job["params"]["tokens"] == "<|im_end|>" and job["params"]["layers"] == [8] and job["params"]["contrast"].endswith("fish_new_ui.yaml")
    # 結果の表(いくつかの層。届かない層は「使う」が無い)。行の「使う」で層と Δ が欄に入り、その行に印
    pg.evaluate(f"(r) => {{ const p = {F}; p.st.scan = r; p.renderScan(); }}", SCAN)
    assert pg.evaluate("document.querySelectorAll('#fx-scan-tab tbody tr').length") == 3
    assert pg.evaluate("document.querySelectorAll('#fx-scan-tab tbody button[data-fxscanuse]').length") == 2
    heads = pg.evaluate("[...document.querySelectorAll('#fx-scan-tab thead th')].map((x) => x.textContent.trim())")
    assert heads[:3] == ["層", "要る Δ", "β"] and "dont_touch に ×0.2 を足したとき" in heads and heads.count("1 位になった数") == 2 and heads.count("KL") == 2
    cells = pg.evaluate("[...document.querySelectorAll('#fx-scan-tab tbody tr:first-child td')].map((x) => x.textContent.trim())")
    assert cells[:8] == ["32", "1040.00", "0 / 4", "0.0123", "1 / 4", "2 / 4", "0.3456", "3 / 4"], cells
    assert "誤検出" in pg.evaluate("document.querySelector('#fx-scan-box').textContent")
    assert "6 個のうち 6 個以上(95%)" in pg.evaluate("document.querySelector('#fx-scan-sum').textContent")
    pg.click("#fx-scan-tab tbody tr[data-fxscan='0'] button")
    assert pg.evaluate("[document.querySelector('#fx-layers').value, document.querySelector('#fx-tdelta').value]") == ["12", "32"]
    assert "層 12 と Δ 32" in pg.evaluate("document.querySelector('#fx-status-scan').textContent")
    assert pg.evaluate("document.querySelector('#fx-scan-tab tbody tr[data-fxscan=\"0\"]').classList.contains('fx-sel')")
    pg.click("#fx-scan-tab tbody tr[data-fxscan='2'] td:nth-child(3)")             # 行のどこでも
    assert pg.evaluate("[document.querySelector('#fx-layers').value, document.querySelector('#fx-tdelta').value]") == ["14", "16"]
    pg.click("#fx-scan-tab tbody tr:nth-child(2) td:nth-child(2)")                  # 届かない層は入れない
    assert pg.evaluate("document.querySelector('#fx-layers').value") == "14"
    # トークンの行で作った方向の表示
    tokdir = {"name": "fish_new_ui_tok", "method": "token_row", "segments": ["0"], "layers": [14], "unsegmented": True, "same_reply": False, "stack": False,
              "token": {"tokens": [151645], "token_text": ["<|im_end|>"], "delta": 16, "rbar": 81.25, "gdot": 2.5, "beta": 520.0},
              "rows": [{"layer": 14, "segment": 0, "positions": "0", "direction": "fish_new_ui_tok", "norm": 520.0, "cos_old": None, "norm_old": None}]}
    pg.evaluate(f"(d) => {{ const p = {F}; p.jobs.dir = 'tokdir'; return p.jobEnded({{ id: 'tokdir', kind: 'fish_direction', status: 'done', result: d }}); }}", tokdir)
    txt = pg.evaluate("document.querySelector('#fx-dir-token').textContent")
    assert "トークンの行" in txt and "<|im_end|>" in txt and "Δ 16.00" in txt and "β 520.00" in txt
    assert pg.evaluate(f"{F}.st.name") == "fish_new_ui"                              # キーの名前は <名前> から(_tok を付けない)
    pg.evaluate("window.scrollTo(0, 0)")
    _shot(pg, "fish-new-scan.png", "#dv-fish .stage")
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_forms_tables(served, browser):
    """確かめ: 会話の形が 2 つ以上ある群ごとの、形 × 区間の表と、書き終えた応答の数。キーの表の上の、会話の形ごとの質問の数。"""
    pg, errors = _open(browser, served["url"], "ja")
    _insert(pg, {**NEW_FISH, "token": None})
    pg.evaluate(f"(r) => {{ const p = {F}; p.st.check = r; p.renderCheck(); }}", CHECK)
    assert pg.evaluate("[...document.querySelectorAll('.fx-form-tab')].map((x) => x.dataset.g)") == ["scene", "dont_touch:ordinary"]
    rows = pg.evaluate("[...document.querySelectorAll('.fx-form-tab')].map((t) => [...t.querySelectorAll('tbody tr')].map((tr) => [...tr.children].map((c) => c.textContent.trim())))")
    assert rows == [[["1 ターン", "4", "75%"], ["複数ターン", "2", "50%"]], [["1 ターン", "3", "0 / 3"], ["system あり", "2", "1 / 2"]]]
    assert pg.evaluate("[...document.querySelectorAll('.fx-form-tab thead th')].map((x) => x.textContent)").count("区間 0") == 2
    assert pg.evaluate("document.querySelector('[data-fxfin=\"scene\"]').textContent") == "最後まで書き終えた応答 5 / 6"
    assert "ρ が 0.8 以上" in pg.evaluate("document.querySelector('#fx-check').textContent")
    assert pg.evaluate("document.querySelectorAll('.fx-form-tab td.fx-bad').length") == 1
    _step(pg, "check")
    _shot(pg, "fish-new-check-forms.png", "#dv-fish .stage")
    # 形が 1 つだけなら出さない
    one = {**CHECK, "groups": {"scene": {**CHECK["groups"]["scene"], "by_form": {"single": CHECK["groups"]["scene"]["by_form"]["single"]}}}}
    pg.evaluate(f"(r) => {{ const p = {F}; p.st.check = r; p.renderCheck(); }}", one)
    assert pg.evaluate("document.querySelectorAll('.fx-form-tab').length") == 0
    keys = {"base": "bnew", "rows": [], "segments": ["0"], "form": "cap", "layers": [8], "subset": "all", "pseudo": 0, "seconds": 1.0, "key_fit": "ridge",
            "ridge": {"c": 1, "cv": False, "forms": {"single": 10, "multi_turn": 4, "system": 0, "system_multi_turn": 0}, "conv_cells": True}}
    pg.evaluate(f"(k) => {{ const p = {F}; p.st.keys = k; p.renderKeys(); }}", keys)
    assert pg.evaluate("document.querySelector('#fx-keys-forms').textContent") == "会話の形ごとの質問の数: 1 ターン 10 · 複数ターン 4 · セルに会話の形も入れた"
    keys["ridge"] = {**keys["ridge"], "forms": {"single": 10, "multi_turn": 0, "system": 0, "system_multi_turn": 0}, "conv_cells": False}
    pg.evaluate(f"(k) => {{ const p = {F}; p.st.keys = k; p.renderKeys(); }}", keys)
    assert not pg.evaluate("document.querySelector('#fx-keys-forms')")
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_new_controls_in_english(served, browser):
    """英語の画面: 新しい欄・表・エラーの文が全部辞書にある。"""
    pg, errors = _open(browser, served["url"], "en")
    _insert(pg, NEW_FISH)
    for st in ("src", "dir", "keys", "units", "check", "all"):
        _step(pg, st)
    assert "Add conversation forms to the cells" in pg.evaluate("document.querySelector('#fx-convcells').parentElement.textContent")
    assert pg.evaluate("document.querySelector('#fx-dirset legend').textContent").startswith("Direction source")
    pg.evaluate(f"(r) => {{ const p = {F}; p.st.scan = r; p.renderScan(); }}", SCAN)
    pg.evaluate(f"(r) => {{ const p = {F}; p.st.check = r; p.renderCheck(); }}", CHECK)
    keys = {"base": "bnew", "rows": [], "segments": ["0"], "form": "cap", "layers": [8], "subset": "all", "pseudo": 0, "seconds": 1.0, "key_fit": "ridge",
            "ridge": {"c": 1, "cv": False, "forms": {"single": 10, "multi_turn": 4, "system": 2, "system_multi_turn": 1}, "conv_cells": True}}
    pg.evaluate(f"(k) => {{ const p = {F}; p.st.keys = k; p.renderKeys(); }}", keys)
    assert "Per conversation form" in pg.evaluate("document.querySelector('#fx-check').textContent")
    assert "Δ needed" in pg.evaluate("document.querySelector('#fx-scan-tab').textContent")
    _capture(pg)
    pg.fill("#fx-layers", "8")
    _step(pg, "dir")
    pg.click("#fx-scan-tab tbody tr[data-fxscan='0'] button")                      # 状態の文
    for setup in ("document.querySelector('#fx-token').value = ''", "document.querySelector('#fx-token').value = '<|im_end|>'; document.querySelector('#fx-tdelta').value = ''",
                  "document.querySelector('#fx-tdelta').value = '24'; document.querySelector('#fx-prior-mt').value = '0'"):
        pg.evaluate(f"() => {{ {setup}; {F}.runDirection(); }}")
    pg.evaluate(f"() => {{ document.querySelector('#fx-prior-mt').value = ''; document.querySelector('#fx-usedir').value = 'd'; {F}.runAll(); document.querySelector('#fx-usedir').value = ''; }}")
    pg.evaluate(f"() => {{ const s = document.querySelector('#fx-seg'); s.value = 'before:x'; {F}.runDirection(); s.value = '0'; }}")
    pg.evaluate(f"{F}.runDirection()")                                              # 投げる文(onRun は null を返す)
    pg.evaluate(f"async () => {{ document.querySelector('#fx-scan-layers').value = '8'; await {F}.runScan(); }}")
    assert [x["kind"] for x in _runs(pg)] == ["fish_direction", "direction_token_scan"]
    pg.evaluate(f"() => {{ document.querySelector('#fx-scan-layers').value = '999'; {F}.runScan(); }}")
    _step(pg, "units")
    pg.select_option("#fx-form", "one")
    pg.select_option("#fx-form", "cap")
    _insert(pg, {**NEW_FISH, "kscale": 4}, "t_k4.json")
    tokdir = {"name": "fish_new_ui_tok", "method": "token_row", "segments": ["0"], "layers": [12],
              "token": {"token_text": ["<|im_end|>"], "delta": 32, "rbar": 81.25, "beta": 1040.0},
              "rows": [{"layer": 12, "segment": 0, "positions": "0", "direction": "fish_new_ui_tok", "norm": 1040.0, "cos_old": None, "norm_old": None}]}
    pg.evaluate(f"(d) => {{ const p = {F}; p.st.dir = d; p.renderDirection(); }}", tokdir)
    assert "same direction for every segment" in pg.evaluate("document.querySelector('#fx-dir').textContent")
    assert pg.evaluate("window.gne.directionView.fish.$('#fx-status-scan').textContent").startswith("Enter layers to scan")
    missing = pg.evaluate("[...window.gneI18n.missing]")
    assert missing == [], missing
    assert not errors, errors
    pg.close()
