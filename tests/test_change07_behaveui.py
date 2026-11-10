"""CHANGE-07 のテスト(CHiPS で残った場面から keyed の JSON を作る流れを UI で): behave のジョブ(進み具合・取り消し・見積もり・結果)、
MCP の behave、UI のサーバの入口(/api/behave・/api/chips-source・POST /api/keyed-from-behave)、鍵のタブの「CHiPS で残った場面から作る」。

モデルなし: behave の進み具合(質問ごと・種ごと、判定の分も)と、進み具合からの取り消し、ジョブの包み(質問集の名前・種の書き方・
ラベルのリスト)、見積もり、MCP の behave、from_behave の警告の種類と重さ(issues)・返事(extracted の reply)。
モデルあり(Qwen3-0.6B、GNE_TEST_MODEL。CPU で回すときは GNE_TEST_NGL=0): サーバのジョブで behave(結果・保存・進み具合・取り消し)、
サーバの入口、ブラウザで「behave を回す」→「JSON を作る」→ 入力欄に入る →「方向から key-check まで」、遅れて届いた読み込み
(behave の一覧・JSON を作った結果)で選んだものを上書きしないこと、英語の文言。
"""

import json
import os
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gne.steer import GneError

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
NGL = int(os.environ.get("GNE_TEST_NGL", "-1"))
CHROME = Path(os.environ.get("GNE_CHROME", Path.home() / ".cache/ms-playwright/chromium-1223/chrome-linux64/chrome"))

# テスト用の対照集合: 消したい振る舞いを、素のモデルが日本語の返事でよく書く「です・ます」の文末にしたもの(流れを確かめるため)
PROMPTS_A = ["今日なんか可愛いね。", "昨日は手伝ってくれてありがとう。", "髪切った？似合ってるね。", "誕生日おめでとう！",
             "君の料理、すごくおいしかったよ。", "いつもありがとう、頼りにしてるよ。", "君の歌、とても上手だったよ。", "傘を貸してくれてありがとう。"]
PROMPTS_B = ["今日の天気はどう？", "おすすめの本は？", "カレーの作り方を教えて。", "朝ごはんに何を食べればいい？",
             "好きな季節は？", "疲れたときはどうすればいい？"]
POLITE = {"name": "polite_left", "prompts_a": PROMPTS_A, "prompts_b": PROMPTS_B, "validate": {"match": "です|ます"}}


# --- モデルなし ----------------------------------------------------------------------------------------------

def _fake_generate(texts_for):
    """evalx._generate の代わり: 質問ごとの返事を返し、2 本ずつ tick を呼ぶ。texts_for(item) が返事の文字列。"""
    def gen(session, items, vecs, lora, when, tick, sampling=None, thinking=False):
        outs = []
        for k in range(0, len(items), 2):
            part = items[k:k + 2]
            outs += [{"text": texts_for(it), "token_ids": [1, 2, 3], "finish": "eog"} for it in part]
            tick(len(part))
        return outs
    return gen


def test_behave_progress_per_question_and_cancel(monkeypatch):
    """behave の進み具合: 最初に 0、種ごとに質問の数だけ、判定があればその分も進み、最後は合計に届く。段は種と判定。
    進み具合が例外を投げれば(ジョブの取り消し)そこで止まる。"""
    import gne.evalx as E
    from gne.behave import behave
    monkeypatch.setattr(E, "_generate", _fake_generate(lambda it: "判定: P" if "{" not in str(it[0]) and "返事" in str(it[0])
                                                       else "ありがとうございます。"))
    calls = []
    qs = PROMPTS_A[:5]
    d = behave(SimpleNamespace(), qs, match="ます", judge="質問 {q} 返事 {a}", labels="P,C", seeds=[1, 2], temperature=0.7,
               progress=lambda done, total, stage: calls.append((done, total, stage)))
    assert calls[0] == (0, 20, "behave") and calls[-1][0] == 20 and all(c[1] == 20 for c in calls)
    assert [c[0] for c in calls] == sorted(c[0] for c in calls)
    stages = {c[2] for c in calls}
    assert {"behave: 種 1", "behave: 種 2", "behave: 種 1(判定)", "behave: 種 2(判定)"} <= stages
    assert [r["counts"]["match"] for r in d["runs"]] == [5, 5]
    calls.clear()
    behave(SimpleNamespace(), qs, match="ます", progress=lambda done, total, stage: calls.append((done, total, stage)))
    assert calls[0] == (0, 5, "behave") and calls[-1] == (5, 5, "behave")

    class Stop(Exception):
        pass

    def stop(done, total, stage):
        if done >= 2:
            raise Stop()
    with pytest.raises(Stop):
        behave(SimpleNamespace(), qs, match="ます", seeds=[1, 2, 3], temperature=0.7, progress=stop)


def test_behave_job_wrapper(tmp_path):
    """ジョブ behave の包み: 質問集は configs/contrast/ の名前でもよい。レシピは 1 つなら文字列、種は "1,2,3"、ラベルはリストでもよい。
    空の judge は無いものとして扱う。種の書き方の誤りは BAD_ARGS。種類の一覧にある。"""
    from gne.api import Session
    from gne.jobs import _behave, standard_kinds
    assert standard_kinds()["behave"] is _behave
    got = []
    s = SimpleNamespace(_contrast_file=Session._contrast_file, behave=lambda prompts, **kw: got.append((prompts, kw)) or {"ok": 1})
    assert _behave(s, "zundamon_endings", recipes="chips", seeds="1, 2,3", labels=["P", "C"], judge="  ", match="のだ") == {"ok": 1}
    prompts, kw = got[-1]
    assert Path(prompts).name == "zundamon_endings.yaml" and kw["recipes"] == ["chips"] and kw["seeds"] == [1, 2, 3]
    assert kw["labels"] == "P,C" and kw["judge"] is None and kw["match"] == "のだ"
    q = tmp_path / "q.yaml"
    q.write_text(yaml.safe_dump(PROMPTS_A), encoding="utf-8")
    _behave(s, str(q))
    assert got[-1][0] == str(q) and got[-1][1]["recipes"] == [] and got[-1][1]["seeds"] is None
    _behave(s, ["a", "b"], seeds=[4])
    assert got[-1][0] == ["a", "b"] and got[-1][1]["seeds"] == [4]
    with pytest.raises(GneError) as ei:
        _behave(s, ["a"], seeds="1,x")
    assert ei.value.code == "BAD_ARGS"
    with pytest.raises(GneError) as ei:
        _behave(s, "no_such_contrast")
    assert ei.value.code == "FILE_NOT_FOUND"


def test_behave_estimate(tmp_path):
    """見積もり: 種ごとに質問を max_tokens まで生成 + 判定。対照集合を渡せば A 側の質問の数で数える。"""
    from gne.estimate import _behave_cost
    cp = tmp_path / "c.yaml"
    cp.write_text(yaml.safe_dump(POLITE, allow_unicode=True), encoding="utf-8")
    sp = SimpleNamespace(gen=lambda n, t, k=8: n * t * 0.01, setprefill=40.0, step=lambda k: 10.0)
    s = SimpleNamespace(_engine_kw={"n_seq_max": 8}, _contrast_file=lambda x: (_ for _ in ()).throw(GneError("FILE_NOT_FOUND", "x")))
    a = _behave_cost(s, sp, {"prompts": str(cp), "max_tokens": 100})
    assert a["prompts"] == 8 and a["seeds"] == 1 and not a["judge"] and a["seconds"] > 0
    b = _behave_cost(s, sp, {"prompts": str(cp), "max_tokens": 100, "seeds": [1, 2, 3], "judge": "{q}{a}"})
    assert b["seeds"] == 3 and b["judge"] and b["seconds"] > 3 * a["seconds"] - 3
    c = _behave_cost(s, sp, {"prompts": ["x", "y"], "max_tokens": 10, "seeds": "1,2"})
    assert c["prompts"] == 2 and c["seeds"] == 2
    assert _behave_cost(s, sp, {"prompts": "nope"}).get("guess")


def test_mcp_behave_tool(monkeypatch):
    """MCP の behave: ジョブとして投げる(引数は Session.behave と同じ。既定の値は渡さない)。instructions にも出る。"""
    import anyio

    import gne.api
    import gne.mcp_server as ms
    started = []

    class FakeRunner:
        def __init__(self, *a, **k):
            pass

        def start(self, kind, params, **k):
            started.append((kind, params))
            return SimpleNamespace(view=lambda: {"id": "j1", "kind": kind, "params": params})
    monkeypatch.setattr(gne.api, "Session", lambda *a, **k: SimpleNamespace(warnings=[]))
    monkeypatch.setattr(ms, "JobRunner", FakeRunner)
    srv = ms.build_server("m.gguf")

    async def go():
        tools = {t.name: t for t in await srv.list_tools()}
        assert "behave" in tools and "job_status" in tools["behave"].description
        await srv.call_tool("behave", {"prompts": "polite_left", "recipes": ["chips"], "match": "です|ます", "max_tokens": 64,
                                       "seeds": [1, 2], "temperature": 0.7, "name": "b1"})
        await srv.call_tool("behave", {"prompts": ["a"]})
    anyio.run(go)
    k, p = started[0]
    assert k == "behave" and p == {"prompts": "polite_left", "recipes": ["chips"], "match": "です|ます", "labels": "R,P,C",
                                   "max_tokens": 64, "temperature": 0.7, "seeds": [1, 2], "name": "b1"}
    assert started[1][1] == {"prompts": ["a"], "labels": "R,P,C", "max_tokens": 400, "name": "behave"}
    assert "behave(ジョブ" in ms.INSTRUCTIONS


def test_from_behave_issues_and_reply(tmp_path):
    """from_behave の結果に、警告の種類と重さ(issues: 赤 = error・黄色 = warn)と、残った質問の返事(extracted の reply)。"""
    from tests.test_change07_frombehave import LEFT, PLAIN, _behave, _chips, _fake_direction, _rec_dict, _store_session, _write_contrast
    from gne.keyed_from_behave import from_behave
    s = _store_session(tmp_path)
    cp = _write_contrast(tmp_path, {**POLITE, "validate": {"match": "ふん"}})
    _fake_direction(s, "polite_left", cp)
    rp = _chips(s)
    _behave(s, "b", [[LEFT, PLAIN, LEFT, PLAIN, PLAIN, PLAIN, PLAIN, PLAIN]], prompts=PROMPTS_A, recipe=None)
    d = from_behave(s, "b", recipe="chips", direction="polite_left", positions="before:[ぁ-ん]")
    kinds = {(x["kind"], x["level"]) for x in d["issues"]}
    assert ("recipe", "error") in kinds and ("broad", "error") in kinds and ("few", "warn") in kinds
    assert len(d["issues"]) == len(d["warnings"]) and all(x["message"] in d["warnings"] for x in d["issues"])
    assert [r["reply"] for r in d["extracted"]] == [LEFT, LEFT]
    assert rp.exists() and _rec_dict(rp, "x")["units"]


# --- モデルあり ----------------------------------------------------------------------------------------------

def _run_job(c, kind, params, timeout=900, ok=True):
    j = c.post("/api/jobs", json={"kind": kind, "params": params})
    assert j.status_code == 200, j.text
    j = j.json()
    t0 = time.time()
    stages = set()
    while time.time() - t0 < timeout:
        v = next(x for x in c.get("/api/jobs").json() if x["id"] == j["id"])
        if v["stage"]:
            stages.add(v["stage"])
        if v["status"] in ("done", "error", "cancelled"):
            if ok:
                assert v["status"] == "done", v["error"]
            v["stages_seen"] = stages
            return v
        time.sleep(0.05)
    raise AssertionError(f"{kind} が終わりません")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Session(Qwen3-0.6B、CPU 可)・TestClient・対照集合 polite_left(A 側 8 問・B 側 6 問、validate.match「です|ます」)から作った方向
    (positions last)・CHiPS のレシピ chips(L10 のヘッド 2 の proj)。"""
    if not MODEL.exists():
        pytest.skip(f"{MODEL} がありません")
    from fastapi.testclient import TestClient

    from gne.api import Session
    from gne.ui.server import create_app
    from tests.test_change07_frombehave import _chips
    d = tmp_path_factory.mktemp("c07bui")
    s = Session(MODEL, root=d / "root", n_gpu_layers=NGL, n_ctx=2048)
    cp = s.store.contrast_dir() / "polite_left.yaml"
    cp.write_text(yaml.safe_dump(POLITE, allow_unicode=True), encoding="utf-8")
    s.direction(contrast=str(cp), name="polite_left", positions="last")
    _chips(s, "chips", [{"layer": 10, "kind": "head", "head": 2, "mode": "proj", "s": 1, "direction": "polite_left"}])
    app = create_app(s, warm=False)
    yield SimpleNamespace(s=s, c=TestClient(app), app=app, contrast=cp, dir=d)
    live = app.state.live
    for j in live.jobs.list():
        if j.status in ("queued", "running"):
            live.jobs.cancel(j.id)
    with live.arbiter.interactive():
        s.close()


@pytest.mark.model
def test_behave_job_result_progress_estimate_cancel(env):
    """ジョブ behave: 対照集合の名前ではなくパスを渡すと A 側の質問に答え、種ごとに進む。結果は数だけ(返事は保存したファイルに、質問と
    レシピの名前と一緒に)。見積もりがジョブに付く。走っている間に取り消すと cancelled。"""
    from gne.estimate import estimate
    c, s = env.c, env.s
    p = {"prompts": str(env.contrast), "recipes": ["chips"], "match": "です|ます", "max_tokens": 12, "seeds": [1, 2],
         "temperature": 0.7, "name": "bjob"}
    e = estimate(s, "behave", p)
    assert e["seconds"] > 0 and e["detail"]["prompts"] == 8 and e["detail"]["seeds"] == 2
    v = _run_job(c, "behave", p)
    r = v["result"]
    assert v["estimate_s"] and v["estimate_s"] > 0 and {"behave: 種 1", "behave: 種 2"} & v["stages_seen"]
    assert r["name"] == "bjob" and r["n_prompts"] == 8 and [x["seed"] for x in r["runs"]] == [1, 2]
    assert all("replies" not in x and x["counts"]["n"] == 8 for x in r["runs"]) and r["recipe_names"] == ["chips"]
    saved = json.loads(Path(r["path"]).read_text(encoding="utf-8"))
    assert saved["prompts"] == PROMPTS_A and saved["recipe"]["units"][0]["mode"] == "proj" and len(saved["runs"][0]["replies"]) == 8
    lst = c.get("/api/behave").json()
    assert lst[0]["name"] == "bjob" and lst[0]["seeds"] == [1, 2] and lst[0]["has_prompts"] and lst[0]["recipe_names"] == ["chips"]
    assert "replies" not in json.dumps(lst)
    # 取り消し: 長めのジョブを投げ、走り出したら止める
    j = c.post("/api/jobs", json={"kind": "behave", "params": {**p, "max_tokens": 64, "seeds": [1, 2, 3, 4, 5, 6], "name": "bcancel"}}).json()
    for _ in range(2000):
        x = next(y for y in c.get("/api/jobs").json() if y["id"] == j["id"])
        if x["status"] == "running" and x["done"] > 0:
            break
        time.sleep(0.02)
    c.delete(f"/api/jobs/{j['id']}")
    for _ in range(3000):
        x = next(y for y in c.get("/api/jobs").json() if y["id"] == j["id"])
        if x["status"] not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert x["status"] == "cancelled" and x["done"] < x["total"]
    assert not (s.store.root / "behave" / s.meta.model_id / "bcancel.json").exists()


def _synthetic(s, name, texts, recipe_units=None):
    """保存した behave の結果に見立てたもの(polite_left の A 側の質問、レシピ chips)。"""
    from tests.test_change07_frombehave import _behave, _rec_dict
    rp = s.store.recipes_dir() / "chips.yaml"
    return _behave(s, name, [texts], prompts=PROMPTS_A, recipe=_rec_dict(rp, name), n_tokens=30)


@pytest.mark.model
def test_server_endpoints(env):
    """/api/chips-source: 方向を作った対照集合の要約と positions の既定。POST /api/keyed-from-behave: from_behave と同じ結果に、
    書いた JSON(spec)を添える。名前でないものは 400。残った質問が無ければ spec は無い。"""
    c, s = env.c, env.s
    src = c.get("/api/chips-source", params={"direction": "polite_left"}).json()
    assert src["contrast"] == "polite_left" and src["n_a"] == 8 and src["n_b"] == 6 and src["match"] == "です|ます"
    assert src["positions"] == "before:です|ます" and src["direction_exists"] and not src["system_a"] and src["same_sha"] is True
    assert c.get("/api/chips-source").status_code == 400
    texts = ["ありがとうございます。とても嬉しいです。"] * 6 + ["ありがとう！", "うれしいな。"]
    _synthetic(s, "syn1", texts)
    r = c.post("/api/keyed-from-behave", json={"behave": "syn1", "recipe": "chips", "direction": "polite_left"}).json()
    assert r["path"] and r["spec"]["name"] == "syn1_keyed" and r["spec"]["keyed"] == {"direction": "polite_left", "alpha": -1, "iv": "chips"}
    assert len(r["extracted"]) == 6 and r["extracted"][0]["reply"] == texts[0] and r["n_none"] == 2
    assert r["spec"]["user_prompts"] == PROMPTS_A[:6] and r["spec"]["dont_touch"][0]["prompts"] == PROMPTS_B
    assert isinstance(r["issues"], list)
    bad = c.post("/api/keyed-from-behave", json={"behave": "../x", "recipe": "chips"})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "BAD_ARGS"
    _synthetic(s, "syn_none", ["ありがとう！"] * 8)
    r = c.post("/api/keyed-from-behave", json={"behave": "syn_none", "recipe": "chips", "direction": "polite_left"}).json()
    assert r["path"] is None and r["spec"] is None and any(x["kind"] == "empty" and x["level"] == "error" for x in r["issues"])


@pytest.mark.model
def test_server_c1_json_to_pipeline(env):
    """質問と返事に C1 制御文字(MMLU の問題文によくある U+0080〜U+009F)・U+0085・U+2028 などを含む behave の結果: POST
    /api/keyed-from-behave の spec を入力欄に入れて(about を除き、JSON を通す)「方向から key-check まで」(direction_key_pipeline の
    ジョブ)に渡すと、鍵と key-check まで回る。<root>/contrasts/ に書いた対照集合を読み戻しても質問と返事の文字は変わらない。"""
    from tests.test_change07_frombehave import C1_A, C1_B, _behave, _rec_dict
    c, s = env.c, env.s
    cp = s.store.contrast_dir() / "polite_c1.yaml"
    cp.write_text(yaml.safe_dump({**POLITE, "name": "polite_c1", "prompts_a": C1_A, "prompts_b": C1_B}, allow_unicode=True,
                                 default_style='"'), encoding="utf-8")
    texts = ["ありがとうございます。とても嬉しいです。"] * 7 + ["\x93ありがとう\x94ございます\x85嬉しいです\u2028。"]
    _behave(s, "syn_c1", [texts], prompts=C1_A, recipe=_rec_dict(s.store.recipes_dir() / "chips.yaml", "syn_c1"), n_tokens=30)
    r = c.post("/api/keyed-from-behave", json={"behave": "syn_c1", "recipe": "chips", "direction": "polite_left",
                                               "contrast": str(cp)})
    assert r.status_code == 200, r.text
    r = r.json()
    assert r["path"] and r["spec"]["user_prompts"] == C1_A and r["spec"]["replies"] == texts
    spec = json.loads(json.dumps({k: v for k, v in r["spec"].items() if k != "about"}))    # 入力欄(JSON.stringify → JSON.parse)
    v = _run_job(c, "direction_key_pipeline", {"spec": spec, "layers": "8", "subset": "even", "max_tokens": 16, "pseudo": 2})
    out = v["result"]
    assert out["key"]["name"] == "syn_c1_keyed" and out["key"]["matched"] == {"A": 4, "of": 4} and out["check"]["held_out"]
    assert out["direction"]["reused"] and out["key"]["iv"]["units"][0]["mode"] == "proj"
    from gne.keyed_build import read_key_spec
    back = read_key_spec(out["contrast_path"])
    assert back.contrast.prompts("B") == C1_A and back.dont_touch[0].prompts == C1_B and back.replies["same"] == texts


def _free_port() -> int:
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


@pytest.fixture(scope="module")
def ui_server(env):
    if not CHROME.exists():
        pytest.skip(f"Chromium がありません({CHROME}、GNE_CHROME で指定)")
    import uvicorn

    from gne.ui.server import create_app
    app = create_app(env.s, warm=False)
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


def _key_tab(browser, url, lang, routes=()):
    pg = browser.new_page(viewport={"width": 1600, "height": 1100}, locale="ja-JP")
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("dialog", lambda dl: dl.accept())
    for pat, fn in routes:
        pg.route(pat, fn)
    pg.goto(url + f"/?lang={lang}")
    pg.wait_for_function("window.gne && window.gne.recipe && window.gne.arch", timeout=120000)
    pg.click("#views button[data-view=direction]")
    pg.wait_for_function("window.gne.directionView && window.gne.directionView.mode", timeout=60000)
    pg.click("#dv-mode button[data-m=key]")
    pg.wait_for_function("window.gne.directionView.key.loaded && document.querySelector(\"#kb-dir option[value='polite_left']\")",
                         timeout=60000)
    pg.click("#kb-wrap > summary")
    return pg, errors


@pytest.mark.model
def test_chips_flow_in_browser(ui_server, browser, env):
    """「CHiPS で残った場面から作る」: 方向・レシピを選び(対照集合の要約と positions の既定が出る)、「behave を回す」(ジョブ、進み具合、
    終わると一覧に入って選ばれる)→「JSON を作る」(残った質問の数と一覧、返事を開ける、<root>/keyed/ に保存、入力欄に入り
    keyed.direction・keyed.iv が欄に入る)→「方向から key-check まで」(方向は使い回し、介入はレシピ)。書きかけの入力欄は
    置き換える前に確かめる。"""
    pg, errors = _key_tab(browser, ui_server, "ja")
    dialogs = []
    pg.on("dialog", lambda dl: dialogs.append(dl.message))
    pg.select_option("#kb-dir", "polite_left")
    pg.wait_for_function("document.querySelector('#kb-src').textContent.includes('A 側 8 問')", timeout=10000)
    assert "validate.match" in pg.evaluate("document.querySelector('#kb-src').textContent")
    assert pg.evaluate("document.querySelector('#kb-pos').placeholder") == "before:です|ます"
    assert pg.evaluate("document.querySelector(\"#kb-recipe option[value='chips']\").textContent").endswith("· CHiPS")
    pg.select_option("#kb-recipe", "chips")
    pg.click("#kb-wrap .kb-adv > summary")
    pg.fill("#kb-mt", "48")
    pg.fill("#kb-bname", "web_left")
    pg.click("#kx-src button[data-src=json]")
    pg.fill("#kx-text", '{"name": "draft"}')                                   # 書きかけの入力欄(置き換える前に確かめる)
    pg.click("#kb-run")
    pg.wait_for_function("document.querySelector('#kb-behave').value === 'web_left'", timeout=600000)
    assert "8 問" in pg.evaluate("document.querySelector('#kb-binfo').textContent")
    assert "できました" in pg.evaluate("document.querySelector('#kb-status').textContent")
    assert pg.evaluate("import('/jobs.js').then((m) => m.KIND.behave)") == "振る舞いを数える(behave)"
    # 判定と positions: 0.6B の返事で流れを確かめるため、文の終わり(どの返事にも出る)を「残った」振る舞いと書き込みたい位置にする
    pg.fill("#kb-match", "。|！")
    pg.fill("#kb-pos", "before:(。|！)")
    pg.click("#kb-make")
    pg.wait_for_function("document.querySelector('#kb-mstatus').textContent.includes('keyed/')", timeout=30000)
    res = pg.evaluate("document.querySelector('#kb-res').textContent")
    assert "残った質問" in res and "dont_touch の ordinary 6 問" in res
    n_rows = pg.evaluate("document.querySelectorAll('#kb-res table.kb-rows tbody tr').length")
    assert n_rows >= 1 and len(dialogs) == 1                                # 書きかけを置き換える前に 1 回確かめた
    pg.click("#kb-res table.kb-rows tbody tr details summary")
    assert pg.evaluate("document.querySelector('#kb-res table.kb-rows tbody tr details').open")
    spec = json.loads(pg.input_value("#kx-text"))
    assert spec["name"] == "web_left_keyed" and spec["keyed"] == {"direction": "polite_left", "alpha": -1, "iv": "chips"}
    assert len(spec["user_prompts"]) == n_rows and spec["positions"] == "before:(。|！)"
    assert (env.s.store.root / "keyed" / "web_left_keyed.json").exists()
    assert pg.input_value("#kx-iv") == "chips" and "keyed.direction の polite_left を使い回す" in pg.evaluate("document.querySelector('#kx-jcheck').textContent")
    assert pg.input_value("#kx-tpl-sel") == "root/web_left_keyed.json"
    # 方向から key-check まで(層は鍵を作る欄で)
    pg.fill("#kx-layers", "8")
    pg.fill("#kx-pseudo", "4")
    pg.fill("#kx-mt", "16")
    pg.click("#kb-res [data-kbpipe]")
    pg.wait_for_selector("#kc-res table.kx-check tbody tr", timeout=600000)
    pipe = pg.evaluate("document.querySelector('#kx-pipe').textContent")
    assert "web_left_keyed" in pipe
    assert pg.input_value("#kx-dir") == "polite_left" and pg.input_value("#kx-alpha") == "-1"
    assert (env.s.store.key_dir(env.s.meta.model_id) / "web_left_keyed.npz").exists()
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_chips_late_loads_do_not_overwrite(ui_server, browser, env):
    """遅れて届いた読み込みは捨てる: behave の一覧を読み直している間に手で選び直した結果は、届いた一覧で上書きしない。「JSON を作る」を
    2 回押して最初の結果が後から届いても、後から押したほうの結果を残す。"""
    s = env.s
    _synthetic(s, "late_a", ["ありがとうございます。"] * 8)
    time.sleep(0.02)
    _synthetic(s, "late_b", ["ありがとうございます。とても嬉しいです。"] * 8)
    held = {"list": [], "make": []}
    mode = {"hold_list": False, "hold_make": True}

    def on_list(route, req):
        if mode["hold_list"] and not held["list"]:
            held["list"].append(route)
        else:
            route.continue_()

    def on_make(route, req):
        body = json.loads(req.post_data or "{}")
        if mode["hold_make"] and body.get("behave") == "late_a" and not held["make"]:
            held["make"].append(route)
        else:
            route.continue_()
    pg, errors = _key_tab(browser, ui_server, "ja", routes=[("**/api/behave", on_list), ("**/api/keyed-from-behave", on_make)])
    pg.select_option("#kb-recipe", "chips")
    # 1. 一覧の読み直し(ジョブの後の自動の選択に見立てて late_a を選ばせる)を止めておき、その間に late_b を選ぶ
    pg.select_option("#kb-behave", "late_a")
    mode["hold_list"] = True
    pg.evaluate("() => { window.gne.directionView.key.chips.loadList('late_a'); }")    # 待たない(読み込みは止めてある)
    for _ in range(400):
        if held["list"]:
            break
        pg.wait_for_timeout(25)
    assert held["list"]
    pg.select_option("#kb-behave", "late_b")
    mode["hold_list"] = False
    with pg.expect_response("**/api/behave"):
        held["list"][0].continue_()
    pg.wait_for_timeout(300)
    assert pg.input_value("#kb-behave") == "late_b"
    # 2. 「JSON を作る」: late_a の結果を止めておき、late_b で押し直す。後から届いた late_a の結果は捨てる
    pg.select_option("#kb-behave", "late_a")
    pg.click("#kb-make")
    for _ in range(400):
        if held["make"]:
            break
        pg.wait_for_timeout(25)
    assert held["make"]
    pg.select_option("#kb-behave", "late_b")
    pg.click("#kb-make")
    pg.wait_for_function("window.gne.directionView.key.chips.made && window.gne.directionView.key.chips.made.behave === 'late_b'", timeout=30000)
    with pg.expect_response("**/api/keyed-from-behave"):
        held["make"][0].continue_()
    pg.wait_for_timeout(400)
    assert pg.evaluate("window.gne.directionView.key.chips.made.behave") == "late_b"
    assert json.loads(pg.input_value("#kx-text"))["name"] == "late_b_keyed"
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_chips_panel_in_english(ui_server, browser):
    """英語の画面: 「CHiPS で残った場面から作る」の文言に日本語が残らない。"""
    pg, errors = _key_tab(browser, ui_server, "en")
    assert pg.evaluate("document.querySelector('#kb-wrap > summary').textContent").startswith("Build from what a CHiPS recipe leaves behind")
    pg.select_option("#kb-dir", "polite_left")
    pg.wait_for_function("document.querySelector('#kb-src').textContent.includes('A side 8 questions')", timeout=10000)
    pg.click("#kb-wrap .kb-adv > summary")
    missing = pg.evaluate("[...window.gneI18n.missing]")
    assert missing == [], missing
    assert not errors, errors
    pg.close()
