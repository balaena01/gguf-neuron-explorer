"""CHANGE-07 のテスト(UI から手軽に使うための追加): 鍵つきの書き込みのひな形(configs/keyed/)、入力欄の表(spec)を受ける鍵の
ジョブ、方向 → 鍵 → key-check の一括(ジョブ direction_key_pipeline、MCP direction_key_pipeline、Session.direction_key_pipeline)、
LoRA の確かめのジョブ(direction_key_verify)、見積もり。

モデルなし: ひな形の形(about の ja / en・中身が元の対照集合と同じ・ツンデレの dont_touch の tasks と starts・(a) の例の α が負)と、
鍵の対照集合として読めること、/api/keyed-templates の中身(UIData.keyed_templates)、表を対照集合に書く決まり(about と keyed を
捨てる・名前・keyed の設定の検査・読めない表は消す)、ジョブの包み(spec か contrast のどちらか、keyed の設定が既定になる、
key-verify の書き込み先は鍵から)、MCP の入口(表・ひな形の名前・configs/contrast の名前、compact、instructions)。
モデルあり(Qwen3-0.6B、GNE_TEST_MODEL。CPU で回すときは GNE_TEST_NGL=0): 見積もり、サーバのジョブで一括(ひな形の表 + 与えた
返事)→ 方向を作る → 鍵 → 保留した例での key-check、2 回目は方向を使い回し、作り直しも選べる。spec で鍵だけを作るジョブ、
key-verify のジョブ。granite(GNE_TEST_MOE)では spec で専門家の選ばれ方。
"""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gne.steer import GneError

ROOT = Path(__file__).resolve().parent.parent
KEYED = ROOT / "configs" / "keyed"
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
MOE = Path(os.environ.get("GNE_TEST_MOE", ROOT / "models" / "granite-3.1-1b-a400m-instruct-Q8_0.gguf"))
NGL = int(os.environ.get("GNE_TEST_NGL", "-1"))
ZUNDAMON = ROOT / "configs" / "contrast" / "zundamon_endings.yaml"
TSUNDERE = ROOT / "configs" / "contrast" / "tsundere_thanks.yaml"
ENV_STUB = {"schema_version": "1", "command": "x", "model_id": "m", "llama_cpp_version": "v", "seed": 0,
            "generated_at": "2026-11-08T00:00:00+00:00", "warnings": []}

# モデル自身の素の返事に見立てた、です・ます調の返事(zundamon_endings.yaml・ひな形 sentence_endings.json の user_prompts と同じ順)
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


def _tpl(name):
    return json.loads((KEYED / name).read_text(encoding="utf-8"))


def _fake_session(tmp_path):
    """表を対照集合に書くだけの Session の代わり(<root>/contrasts/ の置き場所だけ)。"""
    d = tmp_path / "contrasts"
    d.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(store=SimpleNamespace(contrast_dir=lambda: d), progress=None)


# --- モデルなし ----------------------------------------------------------------------------------------------

def test_keyed_templates_shape_and_content(tmp_path):
    """ひな形 3 つ: about(order・ja / en の title と text)、作り方 (b) の 2 つは元の対照集合と同じ中身(文末・場面)、
    場面の例は dont_touch に tasks と starts(positions last)があり、starts が要る理由を説明に書く。(a) の例は system_a / system_b、
    positions response、keyed.alpha が負。どれも鍵の対照集合として読め、about と keyed は対照集合に書かない。"""
    from gne.keyed_build import read_key_spec
    from gne.keyed_pipeline import save_spec
    files = sorted(p.name for p in KEYED.glob("*.json"))
    assert files == ["one_scene.json", "sentence_endings.json", "system_state.json"]
    orders = {}
    for f in files:
        d = _tpl(f)
        a = d["about"]
        orders[f] = a["order"]
        for lang in ("ja", "en"):
            assert a[lang]["title"] and len(a[lang]["text"]) > 80, (f, lang)
        assert not any(w in json.dumps(d, ensure_ascii=False) for w in ("/home/", "design-backup")), f
    assert sorted(orders.values()) == [1, 2, 3]
    z, ts = yaml.safe_load(ZUNDAMON.read_text(encoding="utf-8")), yaml.safe_load(TSUNDERE.read_text(encoding="utf-8"))
    one = _tpl("sentence_endings.json")
    assert one["positions"] == "before:(です|ます)。" and one["user_prompts"] == z["user_prompts"] and one["system_a"] == z["system_a"]
    assert one["dont_touch"] == z["dont_touch"] and "keyed" not in one and one["name"] != z["name"]
    two = _tpl("one_scene.json")
    assert two["positions"] == ts["positions"] and two["user_prompts"] == ts["user_prompts"] and two["dont_touch"] == ts["dont_touch"]
    dts = {g["name"]: g for g in two["dont_touch"]}
    assert set(dts) == {"tasks", "starts"} and dts["starts"]["positions"] == "last" and "positions" not in dts["tasks"]
    assert "starts" in two["about"]["ja"]["text"] and "返事を書き始める直前の位置" in two["about"]["ja"]["text"]
    assert "starts" in two["about"]["en"]["text"] and "just before the reply begins" in two["about"]["en"]["text"]
    three = _tpl("system_state.json")
    assert not three["positions"].startswith("before:") and three["system_a"] and three["system_b"]
    assert three["keyed"]["alpha"] < 0 and "(a)" in three["about"]["ja"]["title"]
    s = _fake_session(tmp_path)
    for f in files:
        path, settings = save_spec(s, _tpl(f))
        d = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "about" not in d and "keyed" not in d and path.name == f"{_tpl(f)['name']}.yaml"
        spec = read_key_spec(path)
        assert spec.contrast.name == _tpl(f)["name"]
        assert settings == _tpl(f).get("keyed", {})


def test_ui_keyed_templates_api():
    """/api/keyed-templates の中身: order の順、spec は about を除き keyed は残す。pipeline のひな形は前と同じ形。"""
    from gne.ui.data import UIData
    xs = UIData.keyed_templates()
    assert [x["file"] for x in xs] == ["sentence_endings.json", "one_scene.json", "system_state.json"]
    assert all("about" not in x["spec"] and x["ja"]["title"] and x["en"]["title"] for x in xs)
    assert xs[2]["spec"]["keyed"] == {"alpha": -1}
    ps = UIData.pipeline_templates()
    assert len(ps) == 5 and all("about" not in x["spec"] for x in ps)


def test_clean_spec_save_spec_and_settings(tmp_path):
    """表の決まり: 名前は英数字など、keyed は決まった設定だけ。読めない表(dont_touch の知らないキー)はファイルを残さない。
    keyed の設定は渡していない引数の既定(alpha は引数にしない)。"""
    from gne.keyed_pipeline import apply_settings, clean_spec, file_settings, resolve, save_spec
    base = {"name": "t1", "positions": "before:です", "user_prompts": ["a", "b", "c", "d"]}
    d, st = clean_spec({**base, "about": {"order": 1}, "keyed": {"layers": "3", "alpha": -1}})
    assert "about" not in d and "keyed" not in d and st == {"layers": "3", "alpha": -1}
    with pytest.raises(GneError) as ei:
        clean_spec({**base, "name": "../x"})
    assert ei.value.code == "BAD_ARGS"
    with pytest.raises(GneError) as ei:
        clean_spec({**base, "keyed": {"nope": 1}})
    assert ei.value.code == "BAD_CONTRAST" and "nope" in ei.value.message
    s = _fake_session(tmp_path)
    with pytest.raises(GneError):
        save_spec(s, {**base, "dont_touch": [{"prompts": ["x"], "bogus": 1}]})
    assert not (tmp_path / "contrasts" / "t1.yaml").exists()
    path, st = save_spec(s, {**base, "keyed": {"q_mode": "means"}})
    assert path.exists() and st == {"q_mode": "means"}
    assert apply_settings({"layers": "3", "q_mode": "means", "alpha": -1}, {"layers": None, "q_mode": "samples"}) == \
        {"layers": "3", "q_mode": "samples"}
    tp = KEYED / "system_state.json"
    assert file_settings(tp) == {"alpha": -1} and file_settings(ZUNDAMON) == {}
    assert resolve(s, str(tp)) == (tp, {"alpha": -1})
    with pytest.raises(GneError) as ei:
        resolve(s, str(tmp_path / "none.yaml"))
    assert ei.value.code == "FILE_NOT_FOUND"


def test_job_wrappers_take_spec(tmp_path, monkeypatch):
    """ジョブの包み: direction_key・direction_key_experts は contrast か spec のどちらか(spec は <root>/contrasts/ に書いて、keyed の
    設定を既定に)。direction_key_pipeline は keyed_pipeline.run に表かパスを渡す。direction_key_verify は書き込み先を鍵から読み、
    1 行 1 文の文字列も受ける。種類の一覧に一括と key-verify。"""
    import gne.keyed as K
    import gne.keyed_pipeline as KP
    from gne.jobs import (_direction_key, _direction_key_experts, _direction_key_pipeline, _direction_key_verify,
                          standard_kinds)
    kinds = standard_kinds()
    assert kinds["direction_key_pipeline"] is _direction_key_pipeline and kinds["direction_key_verify"] is _direction_key_verify
    calls = []
    s = _fake_session(tmp_path)
    s.progress = object()
    s.direction_key = lambda c, **kw: calls.append(("key", c, kw)) or {}
    s.direction_key_experts = lambda c, **kw: calls.append(("exp", c, kw)) or {}
    spec = {"name": "sp", "positions": "before:です", "user_prompts": ["a", "b", "c", "d"],
            "keyed": {"layers": "2,4", "q_mode": "means", "alpha": -1}}
    _direction_key(s, spec=spec, pseudo=3)
    kind, c, kw = calls[-1]
    assert c == str(tmp_path / "contrasts" / "sp.yaml") and kw["layers"] == [2, 4] and kw["q_mode"] == "means"
    assert kw["pseudo"] == 3 and "alpha" not in kw and kw["progress"] is s.progress and "spec" not in kw
    _direction_key(s, spec=spec, layers="5", q_mode="examples")                     # 引数が優先
    assert calls[-1][2]["layers"] == [5] and calls[-1][2]["q_mode"] == "examples"
    _direction_key(s, "c.yaml", layers=[1])
    assert calls[-1][1] == "c.yaml" and calls[-1][2]["layers"] == [1]
    for bad in ({}, {"contrast": "c.yaml", "spec": spec}):
        with pytest.raises(GneError) as ei:
            _direction_key(s, layers=[1], **bad)
        assert ei.value.code == "BAD_ARGS"
    _direction_key_experts(s, spec=spec, top=3)
    kind, c, kw = calls[-1]
    assert kind == "exp" and kw["layers"] == [2, 4] and "q_mode" not in kw and kw["top"] == 3
    got = []
    monkeypatch.setattr(KP, "run", lambda sess, contrast, **kw: got.append((contrast, kw)) or {"ok": 1})
    assert _direction_key_pipeline(s, spec=spec, layers="3") == {"ok": 1}
    assert got[-1][0] is spec and got[-1][1]["layers"] == "3" and got[-1][1]["progress"] is s.progress
    _direction_key_pipeline(s, contrast="c.yaml", layers="3")
    assert got[-1][0] == "c.yaml"
    with pytest.raises(GneError):
        _direction_key_pipeline(s, layers="3")
    monkeypatch.setattr(K, "load_key", lambda sess, name: SimpleNamespace(target="expert:7"))
    seen = []
    s.check_keyed = lambda L, **kw: seen.append((L, kw)) or {"v30": True}
    _direction_key_verify(s, "14", "k", "d", texts="一つ目\n\n二つ目\n")
    L, kw = seen[-1]
    assert L == 14 and kw["kind"] == "expert" and kw["expert"] == 7 and kw["texts"] == ["一つ目", "二つ目"]
    assert kw["alphas"] == (1.0, 2.0) and kw["key"] == "k" and kw["direction"] == "d"
    _direction_key_verify(s, 3, "k", "-d", alphas=[0.5], target="attn")
    assert seen[-1][1]["kind"] == "attn" and seen[-1][1]["expert"] is None and seen[-1][1]["alphas"] == (0.5,) and seen[-1][1]["texts"] is None


def test_mcp_direction_key_pipeline(monkeypatch):
    """MCP の direction_key_pipeline: 表は spec に、configs/contrast/ の名前はパスに、configs/keyed/ のひな形の名前は表に読む。
    key-check の結果は compact。rebuild_direction・check=False を渡す。instructions に一括とひな形の場所。"""
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
    spec = {"name": "x", "positions": "before:です", "user_prompts": ["a", "b", "c", "d"]}

    async def go():
        tools = {t.name: t for t in await srv.list_tools()}
        assert "rebuild_direction" in tools["direction_key_pipeline"].description
        await srv.call_tool("direction_key_pipeline", {"contrast": spec, "layers": [8, 14]})
        await srv.call_tool("direction_key_pipeline", {"contrast": "zundamon_endings", "layers": "8", "rebuild_direction": True,
                                                       "check": False, "replies_a": ["a"]})
        await srv.call_tool("direction_key_pipeline", {"contrast": "one_scene", "layers": [8]})
        e1 = await srv.call_tool("direction_key_pipeline", {"contrast": "nope_nope", "layers": [8]})
        e2 = await srv.call_tool("direction_key_pipeline", {"contrast": spec, "layers": []})
        return e1, e2
    e1, e2 = anyio.run(go)
    assert len(started) == 3
    k, p = started[0]
    assert k == "direction_key_pipeline" and p["spec"] == spec and "contrast" not in p and p["layers"] == [8, 14] and p["compact"]
    assert "rebuild_direction" not in p and "check" not in p
    k, p = started[1]
    assert p["contrast"] == str(ZUNDAMON) and "spec" not in p and p["rebuild_direction"] is True and p["check"] is False
    assert p["replies_a"] == ["a"]
    assert started[2][1]["spec"]["name"] == _tpl("one_scene.json")["name"]
    assert "FILE_NOT_FOUND" in json.dumps(e1, ensure_ascii=False, default=str)
    assert "BAD_ARGS" in json.dumps(e2, ensure_ascii=False, default=str)
    assert "direction_key_pipeline" in ms.INSTRUCTIONS and "configs/keyed/" in ms.INSTRUCTIONS


# --- モデルあり: Qwen3-0.6B(CPU でも回る) ------------------------------------------------------------------------

def _run_job(c, kind, params, timeout=900, ok=True):
    j = c.post("/api/jobs", json={"kind": kind, "params": params})
    assert j.status_code == 200, j.text
    j = j.json()
    t0 = time.time()
    stages = set()
    while time.time() - t0 < timeout:
        v = next(x for x in c.get("/api/jobs").json() if x["id"] == j["id"])
        if v["stage"]:
            stages.add(v["stage"].split(":")[0])
        if v["status"] in ("done", "error", "cancelled"):
            if ok:
                assert v["status"] == "done", v["error"]
            v["stages_seen"] = stages
            return v
        time.sleep(0.1)
    raise AssertionError(f"{kind} が終わりません")


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    """Session(CPU)と TestClient。方向は作らない(一括で作る)。"""
    if not MODEL.exists():
        pytest.skip(f"{MODEL} がありません")
    from fastapi.testclient import TestClient

    from gne.api import Session
    from gne.ui.server import create_app
    s = Session(MODEL, root=tmp_path_factory.mktemp("c07pipe"), n_gpu_layers=NGL, n_ctx=1024)
    app = create_app(s, warm=False)
    yield TestClient(app), s
    live = app.state.live
    for j in live.jobs.list():
        if j.status in ("queued", "running"):
            live.jobs.cancel(j.id)
    with live.arbiter.interactive():                 # 走っているモデルの処理の終わりを待ってから閉じる
        s.close()


@pytest.mark.model
def test_estimate_pipeline_and_verify(api):
    """見積もり: 一括は方向(返事を作る)+ 鍵 + 反対側の subset の key-check。表(spec)も読む。key-verify は文 × (1 + α の数)。"""
    from gne.estimate import Speed, estimate
    _, s = api
    sp = Speed(s)
    spec = _tpl("sentence_endings.json")
    e = estimate(s, "direction_key_pipeline", {"spec": spec, "layers": "8", "replies_a": REPLIES, "response_tokens": 16})
    d = e["detail"]
    assert not d.get("guess") and d["direction_reused"] is False and d["direction_seconds"] > 0
    assert d["build"]["conversations"] == 8 + 2 and d["build"]["generated"] == 2         # even: 返事の 8 + 英語 2(英語は生成)
    assert d["check"]["conversations"] == 8 + 2
    assert e["seconds"] == pytest.approx(d["direction_seconds"] + d["build"]["seconds"] + d["check"]["seconds"])
    e2 = estimate(s, "direction_key", {"spec": spec, "layers": "8", "replies_a": REPLIES})
    assert e2["detail"]["conversations"] == 10 and not e2["detail"].get("guess")
    nc = estimate(s, "direction_key_pipeline", {"spec": spec, "layers": "8", "replies_a": REPLIES, "check": False})["detail"]
    assert nc["check"]["seconds"] == 0.0
    v = estimate(s, "direction_key_verify", {"layer": 8, "key": "k", "direction": "d", "texts": "a\nb\nc"})
    assert v["detail"] == {"texts": 3, "alphas": 2} and v["seconds"] > 0
    assert not (s.store.contrast_dir() / f"{spec['name']}.yaml").exists()              # 見積もりでは <root>/contrasts に書かない


@pytest.fixture(scope="module")
def piped(api):
    """ひな形 sentence_endings.json の表 + 与えた返事で、一括(方向 → 鍵 L8・L14 → key-check)をジョブで回す。"""
    c, _ = api
    spec = _tpl("sentence_endings.json")
    return _run_job(c, "direction_key_pipeline", {"spec": spec, "layers": "8,14", "replies_a": REPLIES, "response_tokens": 16,
                                                  "pseudo": 8, "max_tokens": 24})


@pytest.mark.model
def test_pipeline_job_builds_direction_key_and_check(api, piped):
    """一括のジョブ: 表を <root>/contrasts/<name>.yaml に書き(about・keyed なし)、方向(positions response)を作り、鍵(subset even)、
    反対側(odd)で保留した例の key-check。鍵と key-check の部分は CLI の出力と同じスキーマ。進み具合は段ごと(1/3 方向 → 3/3 key-check)。
    作った対照集合・鍵・方向は一覧に出る。"""
    from gne.schemas import DirectionKeyCheckOut, DirectionKeyOut
    c, s = api
    name = _tpl("sentence_endings.json")["name"]
    r = piped["result"]
    assert r["name"] == name and r["contrast"] == name and r["layers"] == [8, 14] and r["alpha_hint"] is None
    assert r["stages"] == ["方向", "鍵", "key-check"] and {"1/3 方向", "2/3 鍵", "3/3 key-check"} <= piped["stages_seen"]
    d = r["direction"]
    assert d["name"] == name and not d["reused"] and not d["rebuilt"] and d["positions"] == "response"
    assert s.direction_set(name).meta.get("positions") == "response"
    cp = Path(r["contrast_path"])
    assert cp.parent == s.store.contrast_dir() and cp.name == f"{name}.yaml"
    y = yaml.safe_load(cp.read_text(encoding="utf-8"))
    assert "about" not in y and "keyed" not in y and y["positions"] == "before:(です|ます)。"
    DirectionKeyOut.model_validate({**ENV_STUB, **r["key"]})
    DirectionKeyCheckOut.model_validate({**ENV_STUB, **r["check"]})
    assert r["key"]["subset"] == "even" and r["key"]["mode"] == "same" and r["key"]["n_pseudo"] == 8
    ck = r["check"]
    assert ck["subset"] == "odd" and ck["held_out"] is True and [x["layer"] for x in ck["layers"]] == [8, 14]
    assert all(x["hist"] for x in ck["layers"])                                     # UI はヒストグラムに使うので落とさない
    assert any(x["file"] == name and x["src"] == "root" for x in c.get("/api/contrasts").json())
    assert name in [k["name"] for k in c.get("/api/keys").json()]
    assert name in [x["name"] for x in c.get("/api/directions").json()]
    assert r["warnings"] == []


@pytest.mark.model
def test_pipeline_reuses_or_rebuilds_direction(api, piped):
    """2 回目: 同じ名前の方向があれば使い回し(段は鍵だけ)、rebuild_direction で作り直す。中身の違う表なら警告。"""
    c, s = api
    spec = _tpl("sentence_endings.json")
    name = spec["name"]
    r = _run_job(c, "direction_key_pipeline", {"spec": spec, "layers": "14", "replies_a": REPLIES, "pseudo": 4, "max_tokens": 24,
                                               "name": "nk2", "check": False})["result"]
    assert r["direction"]["reused"] and r["stages"] == ["鍵"] and "check" not in r and r["warnings"] == []
    assert r["name"] == "nk2" and r["direction"]["name"] == name
    changed = {**spec, "user_prompts": spec["user_prompts"][:12]}                  # 中身が違う: 方向は使い回すが警告
    r = _run_job(c, "direction_key_pipeline", {"spec": changed, "layers": "14", "replies_a": REPLIES[:12], "pseudo": 4,
                                               "max_tokens": 24, "name": "nk3", "check": False})["result"]
    assert r["direction"]["reused"] and any("中身の違う" in w for w in r["warnings"])
    t0 = s.direction_set(name).path.stat().st_mtime
    r = _run_job(c, "direction_key_pipeline", {"spec": changed, "layers": "14", "replies_a": REPLIES[:12], "pseudo": 4,
                                               "max_tokens": 24, "name": "nk3", "check": False, "rebuild_direction": True,
                                               "response_tokens": 8})["result"]
    assert r["direction"]["rebuilt"] and not r["direction"]["reused"] and r["warnings"] == []
    assert s.direction_set(name).path.stat().st_mtime >= t0


@pytest.mark.model
def test_pipeline_refuses_contrast_without_direction_sides(api, tmp_path):
    """方向を作れない表(system_a も recipe も無い (b))で、同じ名前の方向も無ければ、方向の段で断る(理由と案内つき)。"""
    c, _ = api
    spec = {"name": "nodir", "positions": "before:です", "user_prompts": ["a", "b", "c", "d"]}
    v = _run_job(c, "direction_key_pipeline", {"spec": spec, "layers": "8", "replies_a": ["です。"] * 4}, ok=False)
    assert v["status"] == "error" and "方向を作れません" in v["error"]["message"] and "system_a" in v["error"]["hint"]


@pytest.mark.model
def test_direction_key_job_with_spec_and_verify(api, piped):
    """spec で鍵だけを作るジョブ((a) のひな形、positions last で返事を作らない。keyed.alpha は鍵には効かない)と、
    key-verify のジョブ(書き込み先は鍵から、1 行 1 文の文字列も受ける)。"""
    c, s = api
    spec = _tpl("system_state.json")
    r = _run_job(c, "direction_key", {"spec": spec, "layers": "8", "positions": "last", "pseudo": 4})["result"]
    assert r["name"] == spec["name"] and r["mode"] == "system" and r["positions"] == "last"
    assert (s.store.contrast_dir() / f"{spec['name']}.yaml").exists()
    name = _tpl("sentence_endings.json")["name"]
    v = _run_job(c, "direction_key_verify", {"layer": 14, "key": name, "direction": name,
                                             "texts": "おすすめの趣味を教えてください。散歩が好きです。\nI like walking."})["result"]
    from gne.schemas import DirectionKeyVerifyOut
    DirectionKeyVerifyOut.model_validate({**ENV_STUB, **v})
    assert v["kind"] == "ffn" and v["layer"] == 14 and v["v30"] is True and [x["alpha"] for x in v["rows"]] == [1.0, 2.0]


@pytest.mark.model
def test_granite_experts_job_with_spec(tmp_path):
    """MoE(granite、GNE_TEST_MOE): 表(ひな形 sentence_endings.json + 与えた返事)で専門家の選ばれ方のジョブ。"""
    if not MOE.exists():
        pytest.skip(f"{MOE} がありません")
    from gne.api import Session
    from gne.jobs import _direction_key_experts
    s = Session(MOE, root=tmp_path / "root", n_gpu_layers=NGL, n_ctx=1024)
    try:
        r = _direction_key_experts(s, spec=_tpl("sentence_endings.json"), layers="18", replies_a=REPLIES, top=4, max_tokens=16)
        ly = r["layers"][0]
        assert ly["layer"] == 18 and len(ly["top"]) == 4 and ly["positions_a"] > 0
        assert all(0 <= x["rate_a"] <= 1 and "rows_a" in x and "weight_a" in x for x in ly["top"])
        assert (s.store.contrast_dir() / "noda_endings.yaml").exists()
    finally:
        s.close()
