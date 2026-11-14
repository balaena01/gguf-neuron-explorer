"""FiSH(計算で作るニューロン、docs/fish.md)を Web UI だけで最初から最後まで作るテスト。

サーバ(モデルあり、Qwen3-0.6B、GNE_TEST_MODEL。CPU で回すときは GNE_TEST_NGL=0): ひな形(/api/fish-templates)、ユニットの
組み立て(/api/fish/units)、ジョブ fish_pipeline(方向 → ニューロンの表 → 鍵 → ユニット → 当てた状態で鍵を作り直す → 式どおりか →
足した量)と、段ごとのジョブ(fish_direction・nw_neurons・fish_keys・fish_verify・fish_check・fish_behave)。見積もりが付く。
ブラウザ(Playwright の Chromium、無ければ飛ばす): direction ビューの「FiSH」を開く → ひな形を入れる → 「まとめて作る」→ 方向・
ニューロン・鍵・ユニット・式どおりか・足した量の表 → 「レシピに足す」→ レシピのタブの nw の行(表示・α の変更・保存・読み込み・
LoRA の書き出し)。ニューロンをチェックで選び直すとユニットが組み直され、手で選んだ層は表を取り直しても残る。後から投げた
ジョブがある欄には、先のジョブの結果を入れない。英語の画面で未翻訳が無い。
強さは区切りごと(ひな形は 1 / 1 / 0.8。欄も区切りごと)。「重みに入れる前に試す」(ジョブ fish_try): 方向を層の出力に直接足した返事と数。
「鍵の作り方」(いまの作り方 / 最小二乗)の欄とジョブの引数、鍵の表・まとめて作った結果・足した量の上の作り方の表示、変えたくない場面の群ごとの漏れの表。
"""

import json
import os
import re
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
TEMPLATE = ROOT / "configs" / "fish" / "tone_in_scene.json"
LAYERS = "8,12"


def small_spec(name: str) -> dict:
    """ひな形を小さくした対照集合(狙う場面 8 問、計算 2 問、ふつうの質問 3 問、返事 16 トークン)。"""
    d = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    d.pop("about")
    d["name"] = name
    d["user_prompts"] = d["user_prompts"][:8]
    d["dont_touch"] = d["dont_touch"][:2]                                 # 質問の 2 群だけ(ひな形には質問でない発言の群もある)
    d["dont_touch"][0]["prompts"] = d["dont_touch"][0]["prompts"][:2]
    d["dont_touch"][1]["prompts"] = d["dont_touch"][1]["prompts"][:3]
    d["fish"] = {**d["fish"], "max_tokens": 16, "response_tokens": 16, "pseudo": 2}
    return d


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
    """Session(CPU)と、TestClient・ブラウザ用のサーバ(同じアプリ)。"""
    if not MODEL.exists():
        pytest.skip(f"{MODEL} がありません")
    import uvicorn
    from fastapi.testclient import TestClient

    from gne.api import Session
    from gne.ui.server import create_app
    s = Session(MODEL, root=tmp_path_factory.mktemp("fishui") / "root", n_gpu_layers=NGL, n_ctx=1024)
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
def piped(served):
    """fish_pipeline のジョブの結果(小さい対照集合、L8・L12、上限つきの 2 個組)。"""
    return _run_job(served["client"], "fish_pipeline", {"spec": small_spec("fish_job"), "layers": LAYERS, "top": 8, "save": "fish_job_saved"})


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


def _shot(pg, name, full=True):
    if SHOTS:
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        pg.screenshot(path=str(Path(SHOTS) / name), full_page=full)


def _open(browser, url, lang):
    pg = browser.new_page(viewport={"width": 1600, "height": 1100}, locale="ja-JP")
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("dialog", lambda dl: dl.accept())
    pg.goto(url + f"/?lang={lang}")
    pg.wait_for_function("window.gne && window.gne.recipe && window.gne.arch", timeout=120000)
    return pg, errors


def _open_fish(pg):
    pg.click("#views button[data-view=direction]")
    pg.wait_for_function("window.gne.directionView && window.gne.directionView.mode", timeout=60000)
    pg.click("#dv-mode button[data-m=fish]")
    pg.wait_for_function("window.gne.directionView.fish.loaded", timeout=30000)


def _put_spec(pg, spec):
    pg.evaluate("""(text) => { const ta = document.querySelector('#fx-text'); ta.value = text; ta.dispatchEvent(new Event('input')); }""",
                json.dumps(spec, ensure_ascii=False, indent=2))


# --- サーバ ----------------------------------------------------------------------------------------------

def test_template_file():
    d = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    assert d["about"]["ja"]["title"] and d["about"]["en"]["title"] and d["about"]["ja"]["text"] and d["about"]["en"]["text"]
    assert len(d["user_prompts"]) >= 24 and d["system_a"]
    assert [g["name"] for g in d["dont_touch"]] == ["tasks", "ordinary", "report", "invite", "greet", "request", "partial"]   # 質問でない発言も群ごとに
    assert d["fish"]["key_fit"] == "mean" and "key_fit" in d["about"]["ja"]["text"] and "key_fit" in d["about"]["en"]["text"]   # 既定の作り方
    assert d["fish"]["form"] == "cap" and d["fish"]["segments"] == "0,1-3,4-" and d["fish"]["alpha"] == [1, 1, 0.8]
    assert "layers" not in d["fish"]                                       # 層は使う人が入れる
    assert d["fish"]["stack_fast"] is False                                # 速い読み方は、書けるが既定は切
    assert "stack_fast" in d["about"]["ja"]["text"] and "stack_fast" in d["about"]["en"]["text"] and "探索" in d["about"]["ja"]["text"]


def test_units_and_names_without_model():
    from gne import fish
    assert fish.needed(3, "cap") == 6 and fish.needed(3, "two") == 3 and fish.needed(2, "one") == 2
    assert fish.key_names("b", 2, "cap") == {"position": ["b_p0", "b_p1"], "scene": None, "one": "b_one"}
    u = fish.units("8,12", 2, {8: [1, 2, 3, 4], 12: [5, 6, 7, 8]}, "cap", "d", "b", 0.9)
    assert u == ["8:N1+N2:nw:a0.9@d.seg0#b_p0&b_one", "8:N3+N4:nw:a0.9@d.seg1#b_p1&b_one",
                 "12:N5+N6:nw:a0.9@d.seg0#b_p0&b_one", "12:N7+N8:nw:a0.9@d.seg1#b_p1&b_one"]
    assert fish.units([8], 2, {"8": [1, 2]}, "two", "d", "b", 1) == ["8:N1:nw:a1@d.seg0#b_scene&b_p0", "8:N2:nw:a1@d.seg1#b_scene&b_p1"]
    assert fish.units([8], 1, {8: [9]}, "one", "d", "b", -0.5) == ["8:N9:nw:a-0.5@d.seg0#b_p0"]
    from gne.steer import GneError
    # 区切りごとの強さ: 並びか "1,1,0.8"。1 つだけなら全部の区切りに同じ値。数が区切りと合わなければエラー
    assert fish.alphas_of(0.9, 3) == [0.9, 0.9, 0.9] and fish.alphas_of("0.9", 2) == [0.9, 0.9] and fish.alphas_of([0.7], 2) == [0.7, 0.7]
    assert fish.alphas_of("1,1,0.8", 3) == [1.0, 1.0, 0.8] and fish.alphas_of([1, 1, 0.8], 3) == [1.0, 1.0, 0.8] and fish.alphas_of("1 / 0.5", 2) == [1.0, 0.5]
    assert fish.alphas_of(None, 2) == [0.9, 0.9] and fish.alpha_out([1.0, 1.0]) == 1.0 and fish.alpha_out([1.0, 0.8]) == [1.0, 0.8]
    for bad in ("1,1", [1, 1, 1, 1], "x", [], float("nan")):
        with pytest.raises(GneError):
            fish.alphas_of(bad, 3)
    u3 = fish.units("8,12", 3, {8: [1, 2, 3, 4, 5, 6], 12: [7, 8, 9, 10, 11, 12]}, "cap", "d", "b", "1,1,0.8")
    assert [x.split(":nw:")[1].split("@")[0] for x in u3] == ["a1", "a1", "a0.8", "a1", "a1", "a0.8"] and u3[2] == "8:N5+N6:nw:a0.8@d.seg2#b_p2&b_one"
    assert fish.units([8], 2, {8: [1, 2]}, "one", "d", "b", [1, -0.5]) == ["8:N1:nw:a1@d.seg0#b_p0", "8:N2:nw:a-0.5@d.seg1#b_p1"]
    with pytest.raises(GneError):
        fish.units([8], 2, {8: [1, 2]}, "one", "d", "b", [1, 1, 0.8])       # 強さ 3 つに区切り 2 つ
    with pytest.raises(GneError):
        fish.units([8], 2, {8: [1, 1]}, "one", "d", "b")                    # 重なったニューロン
    with pytest.raises(GneError):
        fish.units([8], 2, {8: [1]}, "one", "d", "b")                       # 足りない
    with pytest.raises(GneError):
        fish.units([8], 2, {8: [1, 2]}, "three", "d", "b")
    # 区切りごとのしきい値(上限つきの形): 強さと同じ考え方。省くとユニットに書かない(既定の 0.4-0.8)。1 つなら全部の区切りに同じ値
    assert fish.caps_of(None, 3) is None and fish.caps_of("", 3) is None and fish.caps_of([], 3) is None
    assert fish.caps_of("0.3-0.7", 3) == ["0.3-0.7"] * 3 and fish.caps_of(["0.30-0.70"], 2) == ["0.3-0.7"] * 2
    assert fish.caps_of("0.4-0.8, 0.3-0.7 / 0.3-0.7", 3) == ["0.4-0.8", "0.3-0.7", "0.3-0.7"] == fish.caps_of(["0.4-0.8", "0.3-0.7", "0.3-0.7"], 3)
    assert fish.cap_out(None) is None and fish.cap_out(["0.3-0.7"] * 3) == "0.3-0.7" and fish.cap_out(["0.4-0.8", "0.3-0.7"]) == ["0.4-0.8", "0.3-0.7"]
    for bad in ("0.4", "0.8-0.4", "0.4-0.4", "a-b", "0.4-0.8,0.3-0.7", ["0.4-0.8"] * 4, "-0.1-0.5"):
        with pytest.raises(GneError):
            fish.caps_of(bad, 3)
    with pytest.raises(GneError):
        fish.caps_of("0.4-0.8", 2, "one")                                    # 上限の無い形には、しきい値が無い
    uc = fish.units("8,12", 3, {8: [1, 2, 3, 4, 5, 6], 12: [7, 8, 9, 10, 11, 12]}, "cap", "d", "b", "1,1,0.8", cap="0.4-0.8,0.3-0.7,0.3-0.7")
    assert uc[0] == "8:N1+N2:nw:a1@d.seg0#b_p0&b_one!cap0.4-0.8" and uc[5] == "12:N11+N12:nw:a0.8@d.seg2#b_p2&b_one!cap0.3-0.7"
    assert [x.split("!cap")[1] for x in uc] == ["0.4-0.8", "0.3-0.7", "0.3-0.7"] * 2
    from gne.nwrite import form_of
    from gne.steer import parse_unit
    assert [form_of(parse_unit(x))["a"] for x in uc[:3]] == [0.4, 0.3, 0.3] and form_of(parse_unit(u[0])) == {"form": "cap", "s": 16.0, "a": 0.4, "b": 0.8}
    assert fish.units("8,12", 3, {8: [1, 2, 3, 4, 5, 6], 12: [7, 8, 9, 10, 11, 12]}, "cap", "d", "b", "1,1,0.8", cap=None) == u3      # 省けば今までと同じ書き方
    assert "cap" in fish.SETTINGS and "cap" not in fish.DEFAULTS
    usage = {"layers": [{"layer": 8, "rows": [{"neuron": j} for j in (5, 3, 9, 1)]}]}
    assert fish.default_neurons(usage, 2, "one") == {8: [5, 3]} and fish.default_neurons(usage, 2, "cap") == {8: [5, 3, 9, 1]}
    with pytest.raises(GneError):
        fish.default_neurons(usage, 3, "cap")


@pytest.mark.model
def test_templates_and_units_api(served):
    c = served["client"]
    tp = c.get("/api/fish-templates").json()
    x = next(t for t in tp if t["file"] == "tone_in_scene.json")
    assert x["src"] == "configs" and x["ja"]["title"] and x["en"]["title"] and "about" not in x["spec"] and x["spec"]["fish"]["form"] == "cap"
    r = c.post("/api/fish/units", json={"layers": "8,12", "segments": "0,1-3,4-", "neurons": {"8": [1, 2, 3, 4, 5, 6], "12": [7, 8, 9, 10, 11, 12]},
                                        "form": "cap", "direction": "d", "base": "b", "alpha": 0.9}).json()
    assert r["needed"] == 6 and len(r["specs"]) == 6 and r["specs"][0] == "8:N1+N2:nw:a0.9@d.seg0#b_p0&b_one"
    assert r["units"][0] == {"layer": 8, "neuron": 1, "pair": 2, "mode": "nw", "alpha": 0.9, "direction": "d.seg0", "key": "b_p0",
                             "key2": "b_one", "s": None, "cap": None}
    assert r["names"] == {"position": ["b_p0", "b_p1", "b_p2"], "scene": None, "one": "b_one"}
    r = c.post("/api/fish/units", json={"layers": "8-9", "segments": "0,1-3,4-", "neurons": {"8": [1, 2, 3], "9": [4, 5, 6]},      # 層は範囲、強さは区切りごと
                                        "form": "one", "direction": "d", "base": "b", "alpha": [1, 1, 0.8]}).json()
    assert r["specs"] == ["8:N1:nw:a1@d.seg0#b_p0", "8:N2:nw:a1@d.seg1#b_p1", "8:N3:nw:a0.8@d.seg2#b_p2",
                          "9:N4:nw:a1@d.seg0#b_p0", "9:N5:nw:a1@d.seg1#b_p1", "9:N6:nw:a0.8@d.seg2#b_p2"] and r["units"][2]["alpha"] == 0.8
    bad = c.post("/api/fish/units", json={"layers": "8", "segments": "0,1-3", "neurons": {"8": [1, 2]}, "form": "one", "direction": "d", "base": "b",
                                          "alpha": [1, 1, 0.8]})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "BAD_ARGS"
    bad = c.post("/api/fish/units", json={"layers": "8", "segments": "0,1-3", "neurons": {"8": [1]}, "form": "cap", "direction": "d", "base": "b"})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "BAD_ARGS"
    r = c.post("/api/fish/units", json={"layers": "8", "segments": "0,1-3", "neurons": {"8": [1, 2, 3, 4]}, "form": "cap", "direction": "d", "base": "b",
                                        "alpha": 1, "cap": ["0.4-0.8", "0.3-0.7"]}).json()                # しきい値は区切りごと
    assert r["specs"] == ["8:N1+N2:nw:a1@d.seg0#b_p0&b_one!cap0.4-0.8", "8:N3+N4:nw:a1@d.seg1#b_p1&b_one!cap0.3-0.7"] and r["units"][1]["cap"] == "0.3-0.7"
    bad = c.post("/api/fish/units", json={"layers": "8", "segments": "0,1-3", "neurons": {"8": [1, 2, 3, 4]}, "form": "cap", "direction": "d", "base": "b",
                                          "cap": "0.4-0.8,0.3-0.7,0.3-0.7"})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "BAD_ARGS"


@pytest.mark.model
def test_pipeline_job(served, piped):
    c, s = served["client"], served["session"]
    r = piped["result"]
    assert piped["estimate_s"] and piped["estimate_s"] > 0                  # 見積もりが付く
    assert piped["estimate_stack_first"] is True                            # このセッションで初めての読み直し: 1 つずつ読む場合で数えた見積もり
    assert r["direction_stack_fast"] is False and r["direction"]["stack_fast"] is False and r["summary"]["direction_stack_fast"] is False
    assert "stack_read" not in s.direction_set("fish_job").meta              # 既定の読み方の方向のファイルは今までの項目のまま
    assert r["name"] == "fish_job" and r["layers"] == [8, 12] and r["segments"] == ["0", "1-3", "4-"] and r["form"] == "cap"
    assert r["alphas"] == [1.0, 1.0, 0.8] and r["alpha"] == [1.0, 1.0, 0.8]             # ひな形の強さ(区切りごと)
    assert Path(r["contrast_path"]).name == "fish_job.yaml"
    assert "fish" not in yaml.safe_load(Path(r["contrast_path"]).read_text(encoding="utf-8"))
    d = r["direction"]
    assert d["same_reply"] and d["stack"] and d["compare"] == "fish_job_old" and len(d["rows"]) == 6
    for x in d["rows"]:
        assert x["norm"] > 0 and x["direction"] == f"fish_job.seg{x['segment']}" and x["cos_old"] is not None and -1 <= x["cos_old"] <= 1.0001
    assert r["neurons_default"] and {int(k): len(v) for k, v in r["neurons"].items()} == {8: 6, 12: 6}
    for row in r["usage"]["layers"]:                                        # 既定の候補 = 表の上から
        assert [x["neuron"] for x in row["rows"][:6]] == r["neurons"][str(row["layer"])]
    k, k2 = r["keys"], r["rekeyed"]
    assert k["base"] == "fish_job" and k2["base"] == "fish_job_r" and r["key_base"] == "fish_job_r"
    assert k["names"] == {"position": ["fish_job_p0", "fish_job_p1", "fish_job_p2"], "scene": None, "one": "fish_job_one"}
    pos = [x for x in k2["rows"] if x["kind"] == "position"]
    assert len(pos) == 6 and all(x["selectivity"] is not None and x["pseudo_p95"] is not None and x["beats"] is not None for x in pos)
    assert [x["kind"] for x in k2["rows"]].count("one") == 2
    assert len(r["units"]) == 6 and all("#fish_job_r_p" in u and "&fish_job_r_one" in u for u in r["units"])
    for u in r["units"]:                                                    # 区切り 0・1 は 1、区切り 2 は 0.8(作り直した鍵のユニットにも)
        assert (":nw:a0.8@fish_job.seg2" in u) if ".seg2" in u else (":nw:a1@fish_job.seg" in u), u
    assert k2["iv"] and any(":nw:a0.8@fish_job.seg2" in x for x in k2["iv"])    # 鍵の作り直しは、区切りごとの強さのユニットを当てた状態で
    assert len(r["unit_rows"]) == 6 and r["unit_rows"][0]["pair"] is not None
    assert r["verify"]["ok"] and [x["layer"] for x in r["verify"]["layers"]] == [8, 12]
    g = r["check"]["groups"]
    assert set(g) == {"scene", "dont_touch:tasks", "dont_touch:ordinary"} and g["scene"]["n"] == 8 and len(g["scene"]["units"]) == 6
    seg0 = next(u for u in g["scene"]["units"] if ".seg0" in u["unit"])
    assert seg0["positions"]["0"]["amount"]["mean"] > 0.5                    # 位置 0 用は、狙う場面の位置 0 で書く
    assert set(r["stage_seconds"]) == {"direction", "neurons", "keys", "rekey", "verify", "check"}
    # 質問ごとの値: 区間ごとの数と、質問ごとの行(質問・返事の出だし・区間ごとの書いた量と鍵の反応)
    qs = r["check"]["question_summary"]
    assert qs["segments"] == ["0", "1-3", "4-"] and qs["cap"] == [[0.4, 0.8]] and len(g["scene"]["questions"]) == 8
    q0 = g["scene"]["questions"][0]
    assert q0["prompt"] == small_spec("fish_job")["user_prompts"][0] and set(q0["segments"]) == {"0", "1-3", "4-"} and q0["segments"]["0"]["amount"] > 0.5
    assert all(set(c) == {"n", "low", "over", "leak", "amount_min", "amount_median"} for c in g["scene"]["by_question"].values())
    assert all(c["low"] is None and c["leak"] is not None for c in g["dont_touch:tasks"]["by_question"].values())
    assert Path(r["check"]["path"]).name == "fish_job_check.json" and r["cap"] is None and r["direction_reused"] is False
    names = {x["name"] for x in c.get("/api/keys").json()}
    assert {"fish_job_p0", "fish_job_one", "fish_job_r_p2", "fish_job_r_one"} <= names
    # レシピとして保存・読み込みできる
    rec = s.recipe(units=r["units"], name="fish_job")
    assert len(rec.nw_units) == 6
    # save を渡したので、レシピのファイルができている(名前 → <root>/recipes/<名前>.yaml)。そのまま読めて、式どおりかも確かめられる
    rp = Path(r["recipe_path"])
    assert rp == s.store.recipes_dir() / "fish_job_saved.yaml" and rp.exists() and s.recipe_path("fish_job_saved") == rp
    saved = s.recipe(path=rp)
    assert saved.name == "fish_job_saved" and saved.to_dict()["units"] == rec.to_dict()["units"]
    assert _run_job(c, "fish_verify", {"units": [u for u in r["units"]]})["result"]["ok"] and s.nw_verify(str(rp))["ok"]
    # 結果には短い要約も入る(式どおりか・ユニットとニューロンの数・保存先・足した量)
    from gne import fish
    sm = r["summary"]
    assert sm["ok"] is True and sm["n_units"] == 6 and sm["n_neurons"] == 12 and sm["recipe_path"] == str(rp) and sm["key_base"] == "fish_job_r"
    assert set(sm["check"]) == set(g) and sm["check"]["scene"]["n"] == 8 and sm["check"]["scene"]["amount"]["0"]["max"] > 0.5
    assert set(sm["check"]["scene"]["by_question"]) == {"0", "1-3", "4-"} and sm["questions"]["n_low"] == qs["n_low"]
    assert len(json.dumps(sm, ensure_ascii=False)) < 7000 < len(json.dumps(r, ensure_ascii=False)) // 10
    text = fish.summary_text(r)
    assert "合格" in text and "ユニット 6 個" in text and "ニューロン 12 個" in text and str(rp) in text


@pytest.mark.model
def test_step_jobs(served, piped):
    """段ごとのジョブ: 方向(今までの作り方・比べなし)、ニューロンの表、鍵(鍵 2 つの形・レシピを当てた状態)、式どおりか、足した量。"""
    c = served["client"]
    spec = small_spec("fish_step")
    d = _run_job(c, "fish_direction", {"spec": spec, "layers": LAYERS, "same_reply": False, "stack": False})["result"]
    assert d["name"] == "fish_step" and not d["same_reply"] and d["compare"] is None and all(x["cos_old"] is None for x in d["rows"])
    n = _run_job(c, "nw_neurons", {"layers": LAYERS, "top": 5})
    assert n["estimate_s"] and [len(x["rows"]) for x in n["result"]["layers"]] == [5, 5]
    picked = {str(x["layer"]): [y["neuron"] for y in x["rows"][:3]] for x in n["result"]["layers"]}
    k = _run_job(c, "fish_keys", {"spec": spec, "layers": LAYERS, "form": "two"})["result"]
    assert k["names"]["scene"] == "fish_step_scene" and k["names"]["one"] is None and not k["with_units"]
    assert {x["kind"] for x in k["rows"]} == {"position", "scene"}
    u = c.post("/api/fish/units", json={"layers": LAYERS, "segments": "0,1-3,4-", "neurons": picked, "form": "two", "direction": "fish_step",
                                        "base": "fish_step", "alpha": 0.8}).json()["specs"]
    k2 = _run_job(c, "fish_keys", {"spec": spec, "layers": LAYERS, "form": "two", "with_units": u, "base": "fish_step_r"})["result"]
    assert k2["with_units"] and k2["base"] == "fish_step_r" and k2["iv"]
    v = _run_job(c, "fish_verify", {"units": u})["result"]
    assert v["ok"] and len(v["layers"]) == 2
    ch = _run_job(c, "fish_check", {"spec": spec, "units": u, "low": 0.3, "top": 3})["result"]
    assert set(ch["groups"]) == {"scene", "dont_touch:tasks", "dont_touch:ordinary"} and ch["max_tokens"] == 16
    assert ch["question_summary"]["low"] == 0.3 and ch["question_summary"]["segments"] == ["0", "1-3", "4-"] and len(ch["question_summary"]["worst"]) <= 3
    assert len(ch["groups"]["scene"]["questions"]) == 8                      # 鍵 2 つの形でも、区間は入力の側の鍵(位置の鍵)から決まる
    # すでにある方向を使い回す: 方向の段を飛ばす。層・区切りが合わない方向ははっきりしたエラー(モデルを流す前に止まる)
    st = small_spec("fish_step")
    st["fish"] = {**st["fish"], "cap": ["0.4-0.8", "0.3-0.7", "0.3-0.7"]}        # 表の fish にも、区切りごとのしきい値を書ける
    e = _run_job(c, "fish_pipeline", {"spec": st, "layers": LAYERS, "direction": "fish_step"}, ok=False)
    assert e["status"] == "error" and e["error"]["code"] == "BAD_DIRECTION" and "--no-stack" in e["error"]["hint"]     # 重ねる前提で作っていない方向
    ru = _run_job(c, "fish_pipeline", {"spec": st, "layers": LAYERS, "direction": "fish_job", "name": "fish_reuse", "rekey": False, "summary": True})
    rr = ru["result"]
    assert rr["direction"] == "fish_job" and rr["direction_reused"] and rr["cap"] == ["0.4-0.8", "0.3-0.7", "0.3-0.7"] and rr["stage_seconds"]["direction"] < 1
    assert all("@fish_job.seg" in x and "#fish_reuse_p" in x for x in rr["units"]) and rr["units"][1].endswith("!cap0.3-0.7")
    assert ru["estimate_s"] < piped["estimate_s"]                              # 見積もりも、方向を作らない分だけ短い
    e = _run_job(c, "fish_pipeline", {"spec": st, "layers": "8", "direction": "fish_job"}, ok=False)
    assert e["status"] == "error" and e["error"]["code"] == "BAD_DIRECTION" and "8,12" in e["error"]["message"]
    e = _run_job(c, "fish_pipeline", {"spec": st, "layers": LAYERS, "cap": "0.4-0.8,0.3-0.7"}, ok=False)
    assert e["status"] == "error" and e["error"]["code"] == "BAD_ARGS" and "しきい値" in e["error"]["message"]
    # 速い読み方(stack_fast): 方向の段のジョブと、まとめて作るジョブ(表の fish.stack_fast)。見積もりは、チェックの有無で変わる
    fd = _run_job(c, "fish_direction", {"spec": small_spec("fish_fast"), "layers": LAYERS, "compare": False, "stack_fast": True})
    sd = _run_job(c, "fish_direction", {"spec": small_spec("fish_slow"), "layers": LAYERS, "compare": False})
    assert fd["result"]["stack_fast"] is True and sd["result"]["stack_fast"] is False and fd["params"]["stack_fast"] is True
    ss = served["session"]
    assert ss.direction_set("fish_fast").meta["stack_read"] == "fast" and "stack_read" not in ss.direction_set("fish_slow").meta
    assert fd["estimate_s"] < sd["estimate_s"] and fd["estimate_stack_first"] is False
    sf = small_spec("fish_fast")
    sf["fish"] = {**sf["fish"], "stack_fast": True}
    fu = _run_job(c, "fish_pipeline", {"spec": sf, "layers": LAYERS, "direction": "fish_fast", "name": "fish_fast_use", "rekey": False, "summary": True})
    assert fu["result"]["direction_reused"] and fu["result"]["direction_stack_fast"] is True             # 使い回した方向の読み方が要約に出る
    assert any("効きません" in n for n in fu["result"]["notes"]) and any("速い読み方(--stack-fast)で作った方向" in n for n in fu["result"]["notes"])
    b = _run_job(c, "fish_behave", {"spec": spec, "units": u, "n": 2, "max_tokens": 8})
    assert b["estimate_s"] and b["result"]["scene"]["counts"]["n"] == 2 and b["result"]["others"]["counts"]["n"] == 2
    assert len(b["result"]["scene"]["replies"]) == 2 and b["result"]["scene"]["replies"][0]["prompt"] == spec["user_prompts"][0]
    assert b["result"]["match"] == spec["validate"]["match"] and not b["result"]["own_prompts"]
    # 重みに入れる前に試す: 方向を層の出力に直接足す(区切りごとの強さ)。見積もりが付く
    tr = _run_job(c, "fish_try", {"spec": spec, "direction": "fish_step", "layers": "8-9", "alpha": [1, 1, 0.8], "n": 2, "max_tokens": 8})
    t = tr["result"]
    assert tr["estimate_s"] and t["layers"] == [8, 9] and t["alphas"] == [1.0, 1.0, 0.8] and t["segments"] == ["0", "1-3", "4-"]
    assert t["scene"]["counts"]["n"] == 2 and t["baseline"]["counts"]["n"] == 2 and t["others"]["counts"]["n"] == 2
    assert "match" in t["scene"]["counts"] and t["match"] == spec["validate"]["match"] and t["scene"]["replies"][0]["prompt"] == spec["user_prompts"][0]
    assert re.fullmatch(r"fish_step_[0-9a-f]{6}_try\.json", Path(t["path"]).name)       # 名前を省くと、設定から作った印が付く
    tr2 = _run_job(c, "fish_try", {"spec": spec, "direction": "fish_step", "layers": "8-9", "alpha": [1, 1, 0.5], "n": 2, "max_tokens": 8})
    assert tr2["result"]["path"] != t["path"] and Path(t["path"]).exists()              # 強さを変えた 2 回目は別のファイル
    tr3 = _run_job(c, "fish_try", {"spec": spec, "direction": "fish_step", "layers": "8", "alpha": [1, 1, 0.8], "n": 1, "max_tokens": 4, "name": "kept"})
    assert Path(tr3["result"]["path"]).name == "kept_try.json"
    e = _run_job(c, "fish_try", {"spec": spec, "direction": "fish_step", "layers": "8", "alpha": [1, 1]}, ok=False)
    assert e["status"] == "error" and e["error"]["code"] == "BAD_ARGS" and "強さ" in e["error"]["message"]
    e = _run_job(c, "fish_pipeline", {"spec": spec, "layers": LAYERS, "alpha": "1,1"}, ok=False)        # 強さの数が区切りと合わない(モデルを流す前に止まる)
    assert e["status"] == "error" and e["error"]["code"] == "BAD_ARGS" and "強さ" in e["error"]["message"]
    # 層が無い・形が違うときははっきりしたエラー
    e = _run_job(c, "fish_pipeline", {"spec": spec}, ok=False)
    assert e["status"] == "error" and e["error"]["code"] == "BAD_ARGS" and "層" in e["error"]["message"]
    e = _run_job(c, "fish_keys", {"spec": spec, "layers": LAYERS, "form": "three"}, ok=False)
    assert e["status"] == "error" and e["error"]["code"] == "BAD_ARGS"


# --- ブラウザ ---------------------------------------------------------------------------------------------

@pytest.mark.model
def test_fish_tab_end_to_end_in_browser(served, browser):
    """FiSH のタブ: ひな形 → まとめて作る → 各段の表 → レシピに足す → レシピのタブの nw の行(α・保存・読み込み・LoRA の書き出し)。"""
    url, s = served["url"], served["session"]
    pg, errors = _open(browser, url, "ja")
    pg.evaluate("window.gne.recipe.clear(); try { localStorage.removeItem('gne.fish.draft'); } catch {}")
    _open_fish(pg)
    f = "window.gne.directionView.fish"
    # ひな形: 選ぶと入力欄に入り、表の fish の値が欄に入る(層は空のまま)
    pg.click("#fx-src button[data-src=json]")
    pg.select_option("#fx-tpl-sel", "tone_in_scene.json")
    pg.wait_for_function("document.querySelector('#fx-text').value.includes('fish_tsundere_thanks')", timeout=5000)
    assert pg.evaluate("document.querySelector('#fx-tpl-about').textContent").startswith("褒められたり")
    assert pg.evaluate("document.querySelector('#fx-form').value") == "cap"
    assert pg.evaluate("[...document.querySelectorAll('#fx-alphas input')].map((x) => x.value)") == ["1", "1", "0.8"]      # 強さの欄は区切りごと
    assert pg.evaluate("[...document.querySelectorAll('#fx-alphas .fx-al span')].map((x) => x.textContent)") == ["0", "1-3", "4-"]
    assert pg.evaluate("document.querySelector('#fx-layers').value") == ""
    assert "狙う場面の質問 32 問" in pg.evaluate("document.querySelector('#fx-jcheck').textContent")
    assert "report(4)" in pg.evaluate("document.querySelector('#fx-jcheck').textContent")                   # 変えたくない場面の群
    assert pg.evaluate("document.querySelector('#fx-keyfit').value") == "mean" and pg.evaluate("document.querySelector('#fx-keyc').disabled")
    assert "層 1 つに 6 個" in pg.evaluate("document.querySelector('#fx-need').textContent")
    pg.fill("#fx-layers", "11-23")                                          # 層は範囲で入れられ、要るニューロンの数と、時間がかかることが出る
    need = pg.evaluate("document.querySelector('#fx-need').textContent")
    assert "13 層で合わせて 78 個" in need and "時間がかかります" in need
    pg.fill("#fx-layers", "")
    # 層を入れないと投げない
    pg.click("#fx-all")
    pg.wait_for_function("document.querySelector('#fx-status-all').classList.contains('err')", timeout=5000)
    assert "層を入れてください" in pg.evaluate("document.querySelector('#fx-status-all').textContent")
    # 小さい対照集合に替えて、まとめて作る
    _put_spec(pg, small_spec("fish_web"))
    assert pg.evaluate("document.querySelector('#fx-mt').value") == "16" and pg.evaluate("document.querySelector('#fx-pseudo').value") == "2"
    pg.fill("#fx-layers", LAYERS)
    assert "2 層で合わせて 12 個" in pg.evaluate("document.querySelector('#fx-need').textContent")
    pg.fill("#fx-top", "8")
    _shot(pg, "fish_1_form.png")
    pg.click("#fx-all")
    pg.wait_for_function("document.querySelector('#fx-prog-all').textContent.length > 0", timeout=30000)        # 進み具合と「止める」
    assert pg.evaluate("!!document.querySelector('#fx-prog-all button[data-fxstop]')")
    pg.wait_for_function("document.querySelectorAll('#fx-unit-tab tbody tr').length === 6", timeout=1500000)
    assert "できました" in pg.evaluate("document.querySelector('#fx-status-all').textContent")
    assert pg.evaluate("document.querySelectorAll('#fx-dir-tab tbody tr').length") == 6
    assert pg.evaluate("document.querySelectorAll('.fx-neu-tab').length") == 2
    assert pg.evaluate("document.querySelectorAll('.fx-neu-tab[data-l=\"8\"] input:checked').length") == 6
    assert "選んだ数 6 / 必要な数 6" in pg.evaluate("document.querySelector('[data-fxcount=\"8\"]').textContent")
    assert pg.evaluate("document.querySelectorAll('#fx-keys-tab tbody tr').length") == 8                          # 位置の鍵 3 本 + ほぼ 1 を返す鍵、× 2 層
    assert "fish_web_r" in pg.evaluate("document.querySelector('#fx-keys').textContent")
    assert pg.evaluate("document.querySelectorAll('#fx-verify-tab tbody tr').length") == 2
    assert pg.evaluate("document.querySelectorAll('#fx-verify .kx-ok').length") == 3                              # 全体と 2 層
    assert pg.evaluate("document.querySelectorAll('.fx-check-tab').length") == 3
    assert pg.evaluate("document.querySelectorAll('.fx-check-tab[data-g=scene] td.fx-in, .fx-check-tab[data-g=scene] td.fx-low').length") > 0
    assert "足りない" in pg.evaluate("document.querySelector('#fx-check-sum').textContent")
    # 質問ごとの値: 場面 × 区間ごとの数の表(狙う場面 3 区間 + 書き込みたくない場面 2 群 × 3 区間)と、平均では隠れることの注意
    assert pg.evaluate("document.querySelectorAll('#fx-q-tab tbody tr').length") == 9
    assert "ほとんど書けていない質問" in pg.evaluate("document.querySelector('#fx-q-sum').textContent")
    assert "平均だけでは" in pg.evaluate("document.querySelector('#fx-check').textContent")
    # 変えたくない場面の群ごとの漏れの表(2 群 + 合わせた行)と、鍵の作り方の表示(既定 = いまの作り方)
    assert pg.evaluate("document.querySelectorAll('#fx-leak-tab tbody tr').length") == 3
    assert pg.evaluate("[...document.querySelectorAll('#fx-leak-tab thead th')].map((x) => x.textContent)") == ["群", "どれかの区間", "区間 0", "区間 1-3", "区間 4-", "位置 0 だけ"]
    for sel in ("#fx-pipe-fit", "#fx-keys-fit"):
        assert pg.evaluate(f"document.querySelector('{sel}').textContent").startswith("鍵の作り方: いまの作り方"), sel
    assert pg.evaluate("document.querySelector('#fx-check-fit').textContent") == "ユニットの鍵の作り方: いまの作り方"
    # 鍵の作り方の欄 → ジョブの引数(いまの作り方なら渡さない。最小二乗なら key_fit、強さを入れたら key_c)。読めなければ止める
    kf = pg.evaluate(f"""(() => {{ const p = {f}, sel = document.querySelector('#fx-keyfit'), c = document.querySelector('#fx-keyc'), seg = document.querySelector('#fx-seg');
        const out = [p.keyFit('all')]; sel.value = 'ridge'; sel.dispatchEvent(new Event('change')); out.push(c.disabled, p.keyFit('all'));
        c.value = '0.1'; out.push(p.keyFit('all')); c.value = '0'; out.push(p.keyFit('all'), document.querySelector('#fx-status-all').textContent);
        c.value = ''; const was = seg.value; seg.value = 'before:(です|ます)。'; out.push(p.keyFit('all')); seg.value = was;
        sel.value = 'mean'; sel.dispatchEvent(new Event('change')); out.push(c.disabled, p.keyFit('all')); return out; }})()""")
    assert kf[0] == {} and kf[1] is False and kf[2] == {"key_fit": "ridge"} and kf[3] == {"key_fit": "ridge", "key_c": 0.1}
    assert kf[4] is None and "正則化の強さは 0 より大きい数" in kf[5] and kf[6] is None and kf[7] is True and kf[8] == {}
    assert pg.evaluate("document.querySelector('#fx-cap').value") == "" and not pg.evaluate("document.querySelector('#fx-cap').disabled")
    assert pg.evaluate("import('/fish.js').then((m) => [m.parseCaps('', 3), m.parseCaps('0.3-0.7', 3), m.parseCaps('0.4-0.8,0.3-0.7,0.3-0.7', 3), m.parseCaps('0.4-0.8,0.3-0.7', 3), m.parseCaps('0.8-0.4', 3)])") \
        == [[], ["0.3-0.7"], ["0.4-0.8", "0.3-0.7", "0.3-0.7"], None, None]
    assert "既定の候補" in pg.evaluate("document.querySelector('#fx-pipe').textContent")
    units = pg.evaluate(f"{f}.st.units")
    assert len(units) == 6 and all("#fish_web_r_p" in u for u in units)
    assert [u.split(":nw:")[1].split("@")[0] for u in units] == ["a1", "a1", "a0.8", "a1", "a1", "a0.8"]          # 区切りごとの強さ
    _shot(pg, "fish_2_built.png")
    # 区切り 2 の強さだけ変えると、その区切りのユニットだけ変わる
    pg.fill("#fx-alpha-2", "0.6")
    pg.wait_for_function(f"{f}.st.units.length === 6 && {f}.st.units[2].includes(':nw:a0.6@') && {f}.st.units[0].includes(':nw:a1@')", timeout=10000)
    pg.fill("#fx-alpha-2", "")                                              # 空の欄があるとユニットを組まない
    pg.wait_for_function("document.querySelector('#fx-add').disabled", timeout=5000)
    pg.fill("#fx-alpha-2", "0.8")
    pg.wait_for_function(f"{f}.st.units.length === 6 && {f}.st.units[2].includes(':nw:a0.8@')", timeout=10000)
    # 後から投げたジョブがある欄には、先のジョブの結果を入れない
    pg.evaluate(f"""(() => {{ const p = {f}; p.jobs.keys = 'newer';
        p.jobEnded({{ id: 'older', kind: 'fish_keys', status: 'done', result: {{ base: 'STALE', rows: [], segments: ['0'], form: 'one', layers: [8] }} }}); }})()""")
    assert pg.evaluate(f"{f}.st.keyBase") == "fish_web_r"
    pg.evaluate(f"{f}.jobs.keys = null")
    # ニューロンを選び直す: 既定の 1 個を外すと数が合わずユニットが消え、別の 1 個を足すと組み直される
    first = pg.evaluate("+document.querySelector('.fx-neu-tab[data-l=\"8\"] input:checked').dataset.fxn")
    other = pg.evaluate("+document.querySelector('.fx-neu-tab[data-l=\"8\"] input:not(:checked)').dataset.fxn")
    pg.uncheck(f".fx-neu-tab[data-l=\"8\"] input[data-fxn=\"{first}\"]")
    pg.wait_for_function("document.querySelector('#fx-add').disabled", timeout=5000)
    assert "選んだ数 5 / 必要な数 6" in pg.evaluate("document.querySelector('[data-fxcount=\"8\"]').textContent")
    pg.check(f".fx-neu-tab[data-l=\"8\"] input[data-fxn=\"{other}\"]")
    pg.wait_for_function("document.querySelectorAll('#fx-unit-tab tbody tr').length === 6", timeout=10000)
    units2 = pg.evaluate(f"{f}.st.units")
    assert f"N{other}" in " ".join(units2) and f"N{first}:" not in " ".join(u for u in units2 if u.startswith("8:")) \
        and f"N{first}+" not in " ".join(units2) and f"+N{first}:" not in " ".join(units2)
    # 手で選んだ層は、表を取り直しても残る(L12 は既定のまま)
    pg.click("#fx-neu-go")
    pg.wait_for_function("document.querySelector('#fx-status-neu').textContent.includes('できました')", timeout=600000)
    assert other in pg.evaluate(f"{f}.st.picked[8]") and first not in pg.evaluate(f"{f}.st.picked[8]")
    pg.click("button[data-fxdef=\"8\"]")
    pg.wait_for_function(f"!{f}.st.touched.has(8) && !document.querySelector('button[data-fxdef]')", timeout=5000)
    # 既定の候補(取り直した表の上から 6 個)でユニットが組み直される(働きがほぼ同じニューロンは、測り直すと順が入れ替わることがある)
    top6 = pg.evaluate(f"{f}.st.usage.layers.find((x) => x.layer === 8).rows.slice(0, 6).map((x) => x.neuron)")
    assert pg.evaluate(f"{f}.st.picked[8]") == top6
    pg.wait_for_function(f"{f}.st.units.length === 6 && {f}.st.units[0].startsWith('8:N{top6[0]}+N{top6[1]}:nw:')", timeout=10000)
    units = pg.evaluate(f"{f}.st.units")
    # レシピに足す → レシピのタブ
    pg.click("#fx-add")
    pg.wait_for_function("document.querySelectorAll('#recipe-list tr.rc-nw').length === 6", timeout=10000)
    assert "レシピに足しました" in pg.evaluate("document.querySelector('#fx-status-units').textContent")
    row = pg.evaluate("document.querySelector('#recipe-list tr.rc-nw').textContent")
    assert "nw" in row and "fish_web.seg0" in row and "fish_web_r_p0" in row and "上限つき" in row
    assert pg.evaluate("window.gne.recipe.addSpecs()") == []                                                     # cvec には入らない
    # α を変える(数の欄)。保存
    pg.fill("#recipe-list tr.rc-nw input.num", "0.6")
    pg.dispatch_event("#recipe-list tr.rc-nw input.num", "change")
    pg.wait_for_function("window.gne.recipe.nws()[0].alpha === 0.6", timeout=5000)
    pg.fill("#rc-name", "fish_web_r")
    pg.dispatch_event("#rc-name", "change")
    pg.click("#rc-save")
    pg.wait_for_function("document.querySelector('#rc-result') && document.querySelector('#rc-result').textContent.includes('fish_web_r')", timeout=30000)
    saved = yaml.safe_load((s.store.recipes_dir() / "fish_web_r.yaml").read_text(encoding="utf-8"))
    nw = [u for u in saved["units"] if u.get("mode") == "nw"]
    assert len(nw) == 6 and nw[0]["alpha"] == 0.6 and nw[0]["pair"] is not None and nw[0]["key2"] == "fish_web_r_one"
    # LoRA の書き出し
    pg.click("#rc-nw")
    pg.wait_for_function("document.querySelector('#rc-result a[download]')", timeout=120000)
    assert pg.evaluate("document.querySelector('#rc-result a[download]').textContent") == "fish_web_r-nw.lora.gguf"
    assert (s.store.root / "export" / "fish_web_r-nw.lora.gguf").stat().st_size > 1000
    _shot(pg, "fish_3_recipe.png", full=False)
    # 読み直し: 下書きを消し、保存したレシピを読む
    pg.evaluate("window.gne.recipe.clear()")
    pg.reload()
    pg.wait_for_function("window.gne && window.gne.recipe && window.gne.arch", timeout=120000)
    pg.evaluate("document.querySelector('#list').classList.remove('collapsed')")
    pg.click("#list-tabs button[data-tab=recipe]")
    pg.wait_for_function("[...document.querySelectorAll('#rc-load option')].some(o => o.value === 'fish_web_r')", timeout=30000)
    pg.select_option("#rc-load", "fish_web_r")
    pg.wait_for_function("document.querySelectorAll('#recipe-list tr.rc-nw').length === 6", timeout=10000)
    got = pg.evaluate("window.gne.recipe.nws()")
    assert got[0]["alpha"] == 0.6 and got[0]["key2"] == "fish_web_r_one" and got[0]["pair"] is not None
    assert sorted(x["alpha"] for x in got) == [0.6, 0.8, 0.8, 1, 1, 1]                                          # 区切りごとの強さが保存・読み込みで残る
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_fish_steps_in_browser(served, browser, piped):
    """段ごとのボタン: 方向 → ニューロンの表 → 鍵(鍵 1 つの形)→ ユニット → 当てた状態で鍵を作り直す → 式どおりか → 足した量。"""
    url = served["url"]
    pg, errors = _open(browser, url, "ja")
    _open_fish(pg)
    f = "window.gne.directionView.fish"
    pg.click("#fx-src button[data-src=json]")
    spec = small_spec("fish_steps")
    spec["fish"]["form"] = "one"
    spec["fish"]["segments"] = "0,1-"
    spec["fish"]["alpha"] = 0.9                                             # 1 つだけ書くと、全部の区切りに同じ値
    _put_spec(pg, spec)
    assert pg.evaluate("[...document.querySelectorAll('#fx-alphas input')].map((x) => x.value)") == ["0.9", "0.9"]
    assert pg.evaluate("[...document.querySelectorAll('#fx-try-alphas input')].map((x) => x.value)") == ["0.9", "0.9"]
    assert pg.evaluate("document.querySelector('#fx-form').value") == "one" and pg.evaluate("document.querySelector('#fx-seg').value") == "0,1-"
    pg.fill("#fx-layers", "8")
    pg.fill("#fx-top", "6")
    assert "層 1 つに 2 個" in pg.evaluate("document.querySelector('#fx-need').textContent")
    assert pg.evaluate("document.querySelector('#fx-add').disabled")
    pg.uncheck("#fx-compare")
    # 「速い読み方」のチェックは 2 か所(まとめて作る・2. 方向を作る)にあり、片方を変えるともう片方もそろう。既定は切。表の fish.stack_fast で入る
    fast = "[...document.querySelectorAll('input.fx-fast')].map((x) => x.checked)"
    assert pg.evaluate(fast) == [False, False]
    assert "探索向け" in pg.evaluate("document.querySelector('#fx-fast').parentElement.textContent")
    tip = pg.evaluate("document.querySelector('#fx-fast-all').parentElement.title")
    assert "cos 0.999 以上" in tip and "配るものを作るときは切のまま" in tip and "大きいモデル" in tip
    _put_spec(pg, {**spec, "fish": {**spec["fish"], "stack_fast": True}})
    assert pg.evaluate(fast) == [True, True]
    _put_spec(pg, spec)                                                     # 書いていない設定は既定(切)に戻る
    assert pg.evaluate(fast) == [False, False]
    pg.check("#fx-fast-all")
    assert pg.evaluate(fast) == [True, True]
    pg.uncheck("#fx-fast")
    assert pg.evaluate(fast) == [False, False]
    pg.check("#fx-fast")
    pg.click("#fx-dir-go")
    pg.wait_for_function("document.querySelectorAll('#fx-dir-tab tbody tr').length === 2", timeout=900000)
    sent = [j for j in served["client"].get("/api/jobs").json() if j["kind"] == "fish_direction" and (j["params"].get("spec") or {}).get("name") == "fish_steps"]
    assert sent and sent[-1]["params"]["stack_fast"] is True                 # チェックがジョブの引数に入る(層が 1 つなので、読み直しは無い)
    assert sent[-1]["result"]["stack_fast"] is False and not pg.evaluate("!!document.querySelector('#fx-dir-read')")
    pg.uncheck("#fx-fast")
    assert "まだ無いもの" in pg.evaluate("document.querySelector('#fx-units').textContent")
    # 重みに入れる前に試す: 方向を層 8 の出力に直接足して返事を作る。強さを変えてもう 1 回 → 回ごとの数が並ぶ
    pg.fill("#fx-try-n", "2")
    pg.fill("#fx-try-mt", "10")
    pg.fill("#fx-try-layers", "0,8")                                        # 層 0 には足せない
    pg.click("#fx-try-go")
    pg.wait_for_function("document.querySelector('#fx-status-try').classList.contains('err')", timeout=5000)
    assert "足す層を入れてください" in pg.evaluate("document.querySelector('#fx-status-try').textContent")
    pg.fill("#fx-try-layers", "")                                           # 空なら上の欄の層
    pg.click("#fx-try-go")
    pg.wait_for_function("document.querySelectorAll('#fx-try-runs tbody tr').length === 1", timeout=900000)
    assert "見積もり" in pg.evaluate("document.querySelector('#fx-status-try').textContent") or "できました" in pg.evaluate("document.querySelector('#fx-status-try').textContent")
    assert pg.evaluate("document.querySelectorAll('#fx-try-scene tbody tr').length") == 2 and pg.evaluate("document.querySelectorAll('#fx-try-others tbody tr').length") == 2
    assert pg.evaluate("document.querySelectorAll('#fx-try-scene thead th').length") == 3          # 質問・足した返事・何も足さない返事
    row = pg.evaluate("[...document.querySelectorAll('#fx-try-runs tbody tr')[0].children].map((x) => x.textContent.trim())")
    assert row[1] == "8" and row[2] == "0: 0.9 / 1-: 0.9" and row[3] == "greedy" and row[4] == "2"
    pg.fill("#fx-try-alpha-1", "0.5")
    pg.fill("#fx-try-temp", "0.8")
    pg.click("#fx-try-go")
    pg.wait_for_function("document.querySelectorAll('#fx-try-runs tbody tr').length === 2", timeout=900000)
    row = pg.evaluate("[...document.querySelectorAll('#fx-try-runs tbody tr')[1].children].map((x) => x.textContent.trim())")
    assert row[2] == "0: 0.9 / 1-: 0.5" and row[3].startswith("0.8")
    assert pg.evaluate(f"{f}.st.tryShown") == 1
    pg.click("#fx-try-runs button[data-fxtry=\"0\"]")                      # 前の回の返事を見る
    pg.wait_for_function(f"{f}.st.tryShown === 0 && document.querySelector('#fx-try-runs tbody tr.fx-sel td').textContent === '1'", timeout=5000)
    assert pg.evaluate(f"{f}.st.tries[1].alphas") == [0.9, 0.5] and pg.evaluate(f"{f}.st.tries[1].temperature") == 0.8
    pg.click("#fx-try-use")                                                 # 試した強さを上の欄に入れる
    assert pg.evaluate("[...document.querySelectorAll('#fx-alphas input')].map((x) => x.value)") == ["0.9", "0.5"]
    pg.fill("#fx-alpha-1", "0.9")
    _shot(pg, "fish_4a_try.png")
    pg.click("#fx-neu-go")
    pg.wait_for_function("document.querySelectorAll('.fx-neu-tab[data-l=\"8\"] input:checked').length === 2", timeout=900000)
    pg.click("#fx-keys-go")
    pg.wait_for_function("document.querySelectorAll('#fx-keys-tab tbody tr').length === 2", timeout=900000)
    pg.wait_for_function("document.querySelectorAll('#fx-unit-tab tbody tr').length === 2", timeout=10000)
    u = pg.evaluate(f"{f}.st.units")
    assert len(u) == 2 and u[0].endswith("@fish_steps.seg0#fish_steps_p0") and ":nw:a0.9" in u[0] and "+N" not in u[0]
    # 強さを変えるとユニットが組み直される(変えた区切りだけ)
    pg.fill("#fx-alpha", "0.7")
    pg.wait_for_function(f"{f}.st.units.length === 2 && {f}.st.units[0].includes(':nw:a0.7@') && {f}.st.units[1].includes(':nw:a0.9@')", timeout=10000)
    # 当てた状態で鍵を作り直す → 鍵の名前が _r に
    pg.click("#fx-rekey")
    pg.wait_for_function(f"{f}.st.keyBase === 'fish_steps_r' && {f}.st.units.length === 2 && {f}.st.units[0].includes('#fish_steps_r_p0')", timeout=900000)
    assert "レシピを当てた状態で作り直した鍵" in pg.evaluate("document.querySelector('#fx-keys').textContent")
    pg.click("#fx-verify-go")
    pg.wait_for_function("document.querySelectorAll('#fx-verify-tab tbody tr').length === 1", timeout=300000)
    assert "合格" in pg.evaluate("document.querySelector('#fx-status-verify').textContent")
    pg.click("#fx-check-go")
    pg.wait_for_function("document.querySelectorAll('.fx-check-tab').length === 3", timeout=900000)
    # 返事を読む: 自分で入れた質問 2 問と、書き込みたくない場面の質問
    pg.fill("#fx-bn", "2")
    pg.fill("#fx-bmt", "12")
    pg.evaluate("document.querySelector('#fx-bq').value = '手伝ってくれてありがとう。\\nその服、似合ってるね。'")
    pg.click("#fx-behave-go")
    pg.wait_for_function("document.querySelectorAll('.fx-behave-tab').length === 2", timeout=900000)
    assert pg.evaluate("document.querySelectorAll('.fx-behave-tab[data-b=scene] tbody tr').length") == 2
    assert pg.evaluate("document.querySelectorAll('.fx-behave-tab[data-b=others] tbody tr').length") == 2
    assert "手伝ってくれてありがとう。" in pg.evaluate("document.querySelector('.fx-behave-tab[data-b=scene]').textContent")
    assert "2 問" in pg.evaluate("document.querySelector('[data-fxb=scene]').textContent")
    assert "鍵づくりに使った質問" not in pg.evaluate("document.querySelector('#fx-behave').textContent")
    assert (served["session"].store.root / "behave" / served["session"].meta.model_id / "fish_steps_fish_scene.json").exists()
    # 区切りを変えると、方向・鍵と合わないことを伝えてユニットを消す
    pg.fill("#fx-seg", "0,1-3,4-")
    pg.wait_for_function("document.querySelector('#fx-add').disabled", timeout=5000)
    assert "方向を作り直してください" in pg.evaluate("document.querySelector('#fx-units').textContent")
    _shot(pg, "fish_4_steps.png")
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_fish_tab_in_english(served, browser, piped):
    """英語の画面: FiSH のタブの全部の欄と表(ジョブの結果を入れて描く)、レシピの nw の行の文言が辞書にある。"""
    url = served["url"]
    pg, errors = _open(browser, url, "en")
    pg.evaluate("window.gne.recipe.clear()")
    _open_fish(pg)
    f = "window.gne.directionView.fish"
    pg.click("#fx-src button[data-src=json]")
    pg.select_option("#fx-tpl-sel", "tone_in_scene.json")
    assert pg.evaluate("document.querySelector('#fx-tpl-about').textContent").startswith("An example")
    pg.click("#fx-all")                                                   # 層なし → 理由が出る
    _put_spec(pg, small_spec("fish_job"))
    pg.fill("#fx-layers", LAYERS)
    job = {"id": "en1", "kind": "fish_pipeline", "status": "done", "result": piped["result"]}
    pg.evaluate(f"(async (j) => {{ const p = {f}; p.jobs.all = j.id; await p.jobEnded(j); }})", job)
    pg.wait_for_function("document.querySelectorAll('#fx-unit-tab tbody tr').length === 6", timeout=10000)
    assert "Build everything" in pg.evaluate("document.querySelector('#fx-all').textContent")
    assert "Target scene" in pg.evaluate("document.querySelector('#fx-check').textContent")
    assert "Values per question" in pg.evaluate("document.querySelector('#fx-check').textContent")
    assert "Leaks per group of scenes to leave alone" in pg.evaluate("document.querySelector('#fx-check').textContent")
    assert pg.evaluate("document.querySelector('#fx-pipe-fit').textContent").startswith("Key method: current method")
    assert pg.evaluate("document.querySelector('#fx-keyfit').parentElement.textContent").strip().startswith("Key method")
    assert "Least squares (weights per cell)" in pg.evaluate("document.querySelector('#fx-keyfit').textContent")
    texts = pg.evaluate(f"""(() => {{ const p = {f}; const k = p.st.keys, pp = p.st.pipe, ck = p.st.check, out = {{}};
        p.st.keys = {{ ...k, key_fit: 'ridge', ridge: {{ c: 100, cv: true, folds: 5, edge: true }} }}; p.renderKeys(); out.keys = document.querySelector('#fx-keys-fit').textContent;
        p.st.pipe = {{ ...pp, key_fit: 'ridge', ridge: {{ build: {{ c: 1, cv: true, folds: 5, edge: false }}, rekey: {{ c: 2, cv: false }} }} }}; p.renderPipe();
        out.pipe = document.querySelector('#fx-pipe-fit').textContent;
        p.st.check = {{ ...ck, key_fit: {{ fit: 'ridge', c: {{ a: 1, b: 1 }} }} }}; p.renderCheck(); out.ck1 = document.querySelector('#fx-check-fit').textContent;
        p.st.check = {{ ...ck, key_fit: {{ fit: 'mixed', keys: {{}} }} }}; p.renderCheck(); out.ck2 = document.querySelector('#fx-check-fit').textContent;
        p.st.keys = k; p.st.pipe = pp; p.st.check = ck; p.renderKeys(); p.renderPipe(); p.renderCheck();
        const sel = document.querySelector('#fx-keyfit'), c = document.querySelector('#fx-keyc'), seg = document.querySelector('#fx-seg');
        sel.value = 'ridge'; c.value = 'x'; p.keyFit('all'); out.bad = document.querySelector('#fx-status-all').textContent;
        c.value = ''; const was = seg.value; seg.value = 'before:x'; p.keyFit('all'); out.before = document.querySelector('#fx-status-all').textContent;
        seg.value = was; sel.value = 'mean'; return out; }})()""")
    assert texts["keys"].startswith("Key method: least squares (weights per cell) · Regularization c = 100(chosen by 5-fold splits over questions)")
    assert "at the edge of the candidates" in texts["keys"] and "first keys c = 1" in texts["pipe"] and "keys rebuilt with the recipe applied c = 2(value given)" in texts["pipe"]
    assert texts["ck1"] == "Key method of the units: least squares (weights per cell) · c = 1" and "are mixed" in texts["ck2"]
    assert texts["bad"].startswith("The regularization strength is a number above 0") and texts["before"].startswith("Least-squares keys are only for reply-position segments")
    assert "Use an existing direction" in pg.evaluate("document.querySelector('#fx-usedir').parentElement.textContent")
    # 速い読み方: チェックの文言、できた方向の読み方の表示、見積もりに添える「初回は長めに出ます」
    assert [x.split("(")[0].strip() for x in pg.evaluate("[...document.querySelectorAll('input.fx-fast')].map((x) => x.parentElement.textContent)")] == ["Fast reading"] * 2
    tip = pg.evaluate("document.querySelector('#fx-fast').parentElement.title")
    assert "cosine of 0.999 or more" in tip and "Keep it off when you build something to distribute" in tip and "large models" in tip
    assert pg.evaluate("document.querySelector('#fx-dir-read').textContent") == "Direction built with the default reading"
    assert not pg.evaluate("document.querySelector('#fx-dir-read').classList.contains('pp-warn')")
    pg.evaluate(f"(d) => {{ const p = {f}; const was = p.st.dir; p.st.dir = {{ ...d, stack_fast: true, reused: true }}; p.renderDirection(); window._fxRead = document.querySelector('#fx-dir').textContent; window._fxWarn = document.querySelector('#fx-dir-read').classList.contains('pp-warn'); p.st.dir = was; p.renderDirection(); }}", piped["result"]["direction"])
    assert "Direction built with fast reading (for exploring" in pg.evaluate("window._fxRead") and "reused an existing one" in pg.evaluate("window._fxRead")
    assert pg.evaluate("window._fxWarn") is True
    pg.evaluate(f"""(async () => {{ const p = {f}; const run = p.onRun, follow = p.follow; p.follow = () => {{}};
        p.onRun = async () => ({{ id: 'est1', estimate_s: 900, estimate_stack_first: true }}); await p.launch('dir', 'fish_direction', {{}}, 'run');
        window._fxEst1 = document.querySelector('#fx-status-dir').textContent;
        p.onRun = async () => ({{ id: 'est2', estimate_s: 80, estimate_stack_first: false }}); await p.launch('dir', 'fish_direction', {{}}, 'run');
        window._fxEst2 = document.querySelector('#fx-status-dir').textContent; p.onRun = run; p.follow = follow; p.jobs.dir = null; }})()""")
    assert "about 900 s" in pg.evaluate("window._fxEst1") and "the first estimate on a model runs long" in pg.evaluate("window._fxEst1")
    assert "about 80 s" in pg.evaluate("window._fxEst2") and "runs long" not in pg.evaluate("window._fxEst2")
    box = pg.evaluate("document.querySelector('.fx-neu-tab input:checked').dataset.fxn")
    pg.uncheck(f".fx-neu-tab input[data-fxn=\"{box}\"]")                   # 数が合わない文・「既定の候補に戻す」
    pg.wait_for_function("document.querySelector('#fx-add').disabled", timeout=5000)
    pg.click("button[data-fxdef]")
    pg.wait_for_function("!document.querySelector('#fx-add').disabled", timeout=10000)
    pg.fill("#fx-seg", "0,1-")                                            # 区切りが合わない文
    pg.wait_for_function("document.querySelector('#fx-add').disabled", timeout=5000)
    pg.fill("#fx-seg", "0,1-3,4-")
    pg.wait_for_function("!document.querySelector('#fx-add').disabled", timeout=10000)
    pg.click("#fx-add")
    pg.wait_for_function("document.querySelectorAll('#recipe-list tr.rc-nw').length === 6", timeout=10000)
    assert "capped" in pg.evaluate("document.querySelector('#recipe-list tr.rc-nw').textContent")
    beh = {"scene": {"counts": {"n": 1, "match": 0, "loop": 1, "empty": 0, "at_limit": 0, "tokens": 5.0},
                     "replies": [{"prompt": "q", "text": "a", "loop": True, "finish": "eog"}]},
           "others": {"counts": {"n": 1, "loop": 0, "empty": 0, "at_limit": 0, "tokens": 5.0},
                      "replies": [{"prompt": "q", "text": "a", "loop": False, "finish": "eog"}]}, "match": "x", "own_prompts": False}
    pg.evaluate(f"(r) => {{ const p = {f}; p.st.behave = r; p.renderBehave(); }}", beh)
    assert "Target-scene questions" in pg.evaluate("document.querySelector('#fx-behave').textContent")
    assert "Try before writing into the weights" in pg.evaluate("document.querySelector('#fx-try-sec h4').textContent")
    assert "13 layers: 78 neurons" in pg.evaluate("(() => { const l = document.querySelector('#fx-layers'); const was = l.value; l.value = '11-23'; l.dispatchEvent(new Event('input')); const s = document.querySelector('#fx-need').textContent; l.value = was; l.dispatchEvent(new Event('input')); return s; })()")
    one = {"counts": {"n": 1, "match": 0, "loop": 1, "empty": 0, "at_limit": 0, "tokens": 5.0}, "replies": [{"prompt": "q", "text": "a", "loop": True, "finish": "eog", "n_tokens": 5}]}
    tr = {"scene": one, "baseline": one, "others": one, "direction": "fish_job", "layers": [8, 12], "segments": ["0", "1-3", "4-"], "alphas": [1, 1, 0.8], "alpha": [1, 1, 0.8],
          "match": "x", "temperature": 0.7, "seed": 1, "own_prompts": False, "notes": ["方向 fish_job は層 8,12 に重ねて書く前提で作ってあります。", "a", "b"]}
    pg.evaluate(f"(r) => {{ const p = {f}; p.st.tries = [r]; p.st.tryShown = 0; p.renderTry(); }}", tr)
    assert "Replies that fall into repetition (loops)" in pg.evaluate("document.querySelector('#fx-try-runs').textContent")
    assert "Reply with the direction added" in pg.evaluate("document.querySelector('#fx-try-scene').textContent")
    pg.fill("#fx-try-layers", "0")                                        # 足せない層の文
    pg.click("#fx-try-go")
    pg.fill("#fx-try-layers", "")
    pg.fill("#fx-alpha-1", "")                                            # 強さが空の文
    pg.click("#fx-all")
    pg.fill("#fx-alpha-1", "1")
    pg.click("#fx-try-use")
    pg.click("#fx-src button[data-src=file]")
    missing = pg.evaluate("[...window.gneI18n.missing]")
    assert missing == [], missing
    assert not errors, errors
    pg.close()


# --- ひな形(configs/fish/)と、文字の直前に書く形 -----------------------------------------------------------------

TEMPLATES = ["tone_in_scene.json", "always_persona.json", "sentence_end.json", "scene_and_sentence_end.json", "two_personas.json",
             "after_chips.json"]                                              # about.order の順


def test_all_templates_are_readable(tmp_path):
    """ひな形が全部読める: 説明(日本語・英語、試した範囲つき)、対照集合として読める、fish の設定が決まりどおり。"""
    from gne import fish
    from gne.keyed_build import read_key_spec
    from gne.yamlio import write_yaml
    files = sorted(p.name for p in TEMPLATE.parent.glob("*.json"))
    assert files == sorted(TEMPLATES)
    orders = []
    for name in TEMPLATES:
        d = json.loads((TEMPLATE.parent / name).read_text(encoding="utf-8"))
        a = d.pop("about")
        orders.append(a["order"])
        assert a["ja"]["title"] and a["en"]["title"] and "試した範囲" in a["ja"]["text"] and "Tried on" in a["en"]["text"], name
        assert "誤差" in a["ja"]["text"] and "Qwen3.5-4B" in a["ja"]["text"] and "Qwen3.5-4B" in a["en"]["text"], name
        st = d.pop("fish")
        assert set(st) <= set(fish.SETTINGS) and "layers" not in st, name       # 層は使う人が入れる
        assert st.get("stack_fast", False) is False and ("stack_fast" in st) == bool(st.get("stack")), name    # 速い読み方: 重ねる形にだけ書いてあり、切
        assert ("stack_fast" in a["ja"]["text"]) == ("stack_fast" in a["en"]["text"]) == bool(st.get("stack")), name
        assert st["form"] in fish.FORMS, name
        segs = fish._segments(st["segments"])
        al = fish.alphas_of(st["alpha"], len(segs))                             # 数 1 つか、区切りごと
        assert al == ([1.0, 1.0, 0.8] if not fish.before_mode(st["segments"]) else [4.0]), name
        if fish.before_mode(st["segments"]):
            assert len(segs) == 1 and st.get("rekey") is False and st["same_reply"] is False and st["form"] in ("one", "two"), name
        else:
            assert segs == ["0", "1-3", "4-"], name
        assert len(d["user_prompts"]) >= 16 and d["system_a"] and d["validate"]["match"], name
        p = tmp_path / f"{d['name']}.yaml"
        write_yaml(p, d)
        ks = read_key_spec(p)
        assert ks.contrast.name == d["name"] and ks.contrast.name.startswith("fish_"), name
        if st["form"] == "two":
            assert ks.dont_touch, name                                          # 場面の鍵には、変えたくない場面の質問が要る
    assert orders == sorted(orders) and len(set(orders)) == len(orders)
    two = json.loads((TEMPLATE.parent / "two_personas.json").read_text(encoding="utf-8"))
    assert two["system_b"] and two["system_a"] != two["system_b"]
    assert json.loads((TEMPLATE.parent / "after_chips.json").read_text(encoding="utf-8"))["fish"]["iv"] == ""


def test_before_mode_without_model():
    from gne import fish
    from gne.steer import GneError
    assert fish.before_mode("before:(です|ます)。") and fish.before_mode(["before:x"]) and not fish.before_mode("0,1-3,4-")
    assert fish._segments(" before:(a,b)。") == ["before:(a,b)。"]              # 正規表現の中のカンマでは分けない
    assert fish.units([30], 1, {30: [7]}, "one", "d", "b", 4, True) == ["30:N7:nw:a4@d#b_p0"]
    assert fish.units([30], 1, {30: [7]}, "two", "d", "b", 4, True) == ["30:N7:nw:a4@d#b_scene&b_p0"]
    with pytest.raises(GneError):
        fish._segments("before:")


@pytest.mark.model
def test_templates_api_lists_all(served):
    tp = served["client"].get("/api/fish-templates").json()
    assert [t["file"] for t in tp] == TEMPLATES
    assert all(t["ja"]["title"] and t["en"]["title"] and "fish" in t["spec"] and "about" not in t["spec"] for t in tp)


@pytest.fixture(scope="module")
def piped_before(served):
    """文末の直前に書く形(ひな形 sentence_end を小さくしたもの、L12、鍵 1 つ)の fish_pipeline の結果。"""
    d = json.loads((TEMPLATE.parent / "sentence_end.json").read_text(encoding="utf-8"))
    d.pop("about")
    d["name"] = "fish_end_job"
    d["fish"] = {**d["fish"], "max_tokens": 32, "response_tokens": 32, "pseudo": 2, "alpha": 2}
    return d, _run_job(served["client"], "fish_pipeline", {"spec": d, "layers": "12", "top": 6})


@pytest.mark.model
def test_pipeline_job_before_mode(served, piped_before):
    spec, j = piped_before
    r = j["result"]
    assert r["segments"] == ["before:(です|ます)。"] and r["form"] == "one" and r["alpha"] == 2
    assert r["rekeyed"] is None and r["key_base"] == "fish_end_job" and any("作り直し" in n for n in r["notes"])   # rekey: false と、書けない理由
    assert "rekey" not in r["stage_seconds"]
    assert "recipe_path" not in r and r["summary"]["recipe_path"] is None            # save を渡さなければ、レシピは書かない(今までどおり)
    assert not list(served["session"].store.recipes_dir().glob("fish_end_job*"))
    assert len(r["units"]) == 1 and r["units"][0].endswith(":nw:a2@fish_end_job#fish_end_job_p0") and ".seg" not in r["units"][0]
    d = r["direction"]
    assert not d["same_reply"] and not d["stack"] and d["rows"][0]["direction"] == "fish_end_job" and d["rows"][0]["norm"] > 0
    assert r["keys"]["rows"][0]["positions"] == "before:(です|ます)。"
    assert r["verify"]["ok"]
    ch = r["check"]
    assert ch["positions"] == "before:(です|ます)。" and ch["replies"] == "base"
    u = ch["groups"]["scene"]["units"][0]["positions"]
    assert set(u) == {"match", "other"} and u["match"]["amount"]["n"] > 0
    assert u["match"]["amount"]["mean"] > 5 * abs(u["other"]["amount"]["mean"])                 # 一致の直前でだけ書く
    en = ch["groups"]["dont_touch:english"]["units"][0]["positions"]
    assert abs(en["other"]["amount"]["mean"]) < 0.2
    # 段ごとの確かめも、区切りの欄の値を渡すと同じ分け方になる
    c = _run_job(served["client"], "fish_check", {"spec": spec, "units": r["units"], "segments": spec["fish"]["segments"], "max_tokens": 32})
    assert set(c["result"]["groups"]["scene"]["units"][0]["positions"]) == {"match", "other"}


@pytest.mark.model
def test_templates_in_browser(served, browser, piped_before):
    """ひな形を 1 つずつ選ぶと、入力欄と欄(区切り・形・強さ・作り方・鍵の作り直し・iv)に入る。文字の直前に書く形の結果の表。英語で未翻訳なし。"""
    url = served["url"]
    for lang in ("ja", "en"):
        pg, errors = _open(browser, url, lang)
        pg.evaluate("try { localStorage.removeItem('gne.fish.draft'); } catch {}")
        _open_fish(pg)
        f = "window.gne.directionView.fish"
        pg.click("#fx-src button[data-src=json]")
        assert pg.evaluate("[...document.querySelectorAll('#fx-tpl-sel option')].map((o) => o.value).filter(Boolean)") == TEMPLATES
        want = {"tone_in_scene.json": ("0,1-3,4-", "cap", "1,1,0.8", "same", True, True, 6),
                "always_persona.json": ("0,1-3,4-", "cap", "1,1,0.8", "same", True, True, 6),
                "sentence_end.json": ("before:(です|ます)。", "one", "4", "own", False, False, 1),
                "scene_and_sentence_end.json": ("before:(です|ます)。", "two", "4", "own", False, False, 1),
                "two_personas.json": ("0,1-3,4-", "cap", "1,1,0.8", "same", True, True, 6),
                "after_chips.json": ("0,1-3,4-", "cap", "1,1,0.8", "same", True, True, 6)}
        for name in ["sentence_end.json", "tone_in_scene.json", "always_persona.json", "scene_and_sentence_end.json", "two_personas.json", "after_chips.json"]:
            pg.select_option("#fx-tpl-sel", name)
            spec_name = json.loads((TEMPLATE.parent / name).read_text(encoding="utf-8"))["name"]
            pg.wait_for_function(f"document.querySelector('#fx-text').value.includes('\"{spec_name}\"')", timeout=5000)
            seg, form, alpha, method, stack, rekey, need = want[name]
            got = pg.evaluate("""({seg: document.querySelector('#fx-seg').value, form: document.querySelector('#fx-form').value,
                alpha: [...document.querySelectorAll('#fx-alphas input')].map((x) => x.value).join(','), method: document.querySelector('#fx-method').value,
                stack: document.querySelector('#fx-stack').checked, rekey: document.querySelector('#fx-rekey-on').checked,
                iv: document.querySelector('#fx-iv').value, errs: document.querySelectorAll('#fx-jcheck .err').length,
                about: document.querySelector('#fx-tpl-about').textContent, layers: document.querySelector('#fx-layers').value})""")
            assert (got["seg"], got["form"], got["alpha"], got["method"], got["stack"], got["rekey"]) == (seg, form, alpha, method, stack, rekey), (name, got)
            assert got["errs"] == 0 and got["iv"] == "" and got["layers"] == "", (name, got)
            assert ("試した範囲" if lang == "ja" else "Tried on") in got["about"], name
            assert pg.evaluate(f"{f}.needed()") == need, name
        # 文字の直前に書く形の結果を描く(ジョブの結果を入れる)
        spec, j = piped_before
        pg.select_option("#fx-tpl-sel", "sentence_end.json")
        _put_spec(pg, spec)
        pg.fill("#fx-layers", "12")
        job = {"id": f"b-{lang}", "kind": "fish_pipeline", "status": "done", "result": j["result"]}
        pg.evaluate(f"(async (j) => {{ const p = {f}; p.jobs.all = j.id; await p.jobEnded(j); }})", job)
        pg.wait_for_function("document.querySelectorAll('#fx-unit-tab tbody tr').length === 1", timeout=10000)
        heads = pg.evaluate("[...document.querySelectorAll('.fx-check-tab[data-g=scene] thead th')].map((x) => x.textContent)")
        assert ("一致の直前" if lang == "ja" else "just before a match") in heads and len(heads) == 5
        assert pg.evaluate("document.querySelectorAll('.fx-check-tab[data-g=scene] td.fx-in, .fx-check-tab[data-g=scene] td.fx-low').length") == 1
        assert "作り直し" in pg.evaluate("document.querySelector('#fx-pipe').textContent")           # サーバの注(日本語のまま)
        _shot(pg, f"fish_5_before_{lang}.png")
        if lang == "en":
            missing = pg.evaluate("[...window.gneI18n.missing]")
            assert missing == [], missing
        assert not errors, errors
        pg.close()
