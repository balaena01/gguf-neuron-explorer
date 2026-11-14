"""Phase 5(UI)のテスト。構造情報(ArchInfo)・サーバの API・ブラウザでの描画と hover。

ブラウザのテストは Playwright の Chromium を使う(無ければ飛ばす)。WSL では WSLg の D3D12 経由で実 GPU を使う。
"""

import json
import os
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from gguf import GGUFWriter

from gne.arch import arch_info, sheet_side
from gne.weights import Weights

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
CHROME = Path(os.environ.get("GNE_CHROME", Path.home() / ".cache/ms-playwright/chromium-1223/chrome-linux64/chrome"))


def _write(path, arch, n_layer, tensors_of, kv=None):
    w = GGUFWriter(str(path), arch=arch)
    w.add_name("t")
    w.add_block_count(n_layer)
    w.add_embedding_length(8)
    w.add_feed_forward_length(16)
    w.add_head_count(4)
    w.add_head_count_kv(2)
    w.add_token_list([f"t{i}" for i in range(5)])
    for k, v in (kv or {}).items():
        if isinstance(v, int):
            w.add_uint32(k, v)
    for L in range(n_layer):
        for name, shape in tensors_of(L):
            w.add_tensor(f"blk.{L}.{name}", np.zeros(shape, np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return Weights(path)


def test_arch_hybrid_with_interval_and_ssm(tmp_path):
    def ts(L):
        mixer = [("attn_q.weight", (8, 8))] if (L + 1) % 4 == 0 else [("attn_qkv.weight", (24, 8)), ("ssm_alpha.weight", (4, 8)),
                                                                     ("ssm_beta.weight", (4, 8))]
        return mixer + [("ffn_down.weight", (8, 16))]
    w = _write(tmp_path / "h.gguf", "qwen35", 8, ts, {"qwen35.full_attention_interval": 4, "qwen35.ssm.time_step_rank": 6,
                                                      "qwen35.ssm.group_count": 3})
    a = arch_info(w)
    assert [li.mixer for li in a.layers] == ["linear_attention"] * 3 + ["attention"] + ["linear_attention"] * 3 + ["attention"]
    assert (a.layers[0].n_head, a.layers[0].n_head_kv) == (6, 3) and (a.layers[3].n_head, a.layers[3].n_head_kv) == (4, 2)
    assert a.layers[0].ffn == "dense" and a.layers[0].n_ff == 16 and a.grid_side == 4


def test_arch_ssm_without_interval_is_classified_before_attn_qkv(tmp_path):
    """attn_qkv を持つ線形注意の層を attention と取り違えない(F32)。Mamba 系は recurrent。"""
    w = _write(tmp_path / "s.gguf", "x", 2, lambda L: [("attn_qkv.weight", (24, 8)), ("ssm_beta.weight", (4, 8))] if L == 0
               else [("ssm_in.weight", (16, 8)), ("ssm_d", (4,))])
    a = arch_info(w)
    assert [li.mixer for li in a.layers] == ["linear_attention", "recurrent"]


def test_arch_moe_and_sheet_side(tmp_path):
    w = _write(tmp_path / "m.gguf", "qwen3moe", 2,
               lambda L: [("attn_q.weight", (8, 8)), ("ffn_down_exps.weight", (20, 8, 6))]
               + ([("ffn_down_shexp.weight", (8, 12))] if L == 1 else []),
               {"qwen3moe.expert_count": 20, "qwen3moe.expert_used_count": 2})
    a = arch_info(w)
    l0, l1 = a.layers
    assert (l0.ffn, l0.n_ff, l0.n_expert, l0.n_expert_used) == ("moe", 6, 20, 2)
    assert l1.ffn == "moe+shared" and l1.n_ff_shexp == 12
    assert sheet_side(l0) == 3 * 5 and a.grid_side == 15          # 専門家 20 > 16: 3×3 のタイルを 5×5 に並べる


@pytest.mark.model
@pytest.mark.skipif(not MODEL.exists(), reason="モデルがありません")
def test_server_api(tmp_path):
    from fastapi.testclient import TestClient

    from gne.api import Session
    from gne.ui.server import create_app
    s = Session(MODEL, root=tmp_path)
    c = TestClient(create_app(s))
    d = c.get("/api/model").json()
    assert d["arch"]["n_layer"] == d["meta"]["n_layer"] and len(d["coverage"]) == d["meta"]["n_layer"]
    assert all(li["mixer"] == "attention" and li["ffn"] == "dense" for li in d["arch"]["layers"])
    v = c.get("/api/vocab", headers={"Accept-Encoding": "gzip"})
    assert v.status_code == 200 and len(v.json()) == d["meta"]["n_vocab"]
    assert c.get("/").status_code == 200 and "importmap" in c.get("/").text
    assert c.get("/vendor/three/three.module.min.js").status_code == 200


@pytest.mark.model
@pytest.mark.skipif(not MODEL.exists(), reason="モデルがありません")
def test_u1_api_on_seeded_data(tmp_path):
    from fastapi.testclient import TestClient

    from gne.api import Session
    from gne.ui.server import create_app
    s = Session(MODEL, root=tmp_path)
    _seed(s)
    c = TestClient(create_app(s))
    n_ff = s.meta.n_ff_at(6)
    r = c.get("/api/layer/6/field", params={"name": "sweep"})
    v = np.frombuffer(r.content, np.float32)
    meta = json.loads(r.headers["x-meta"])
    assert v.size == n_ff and v[100] == 20.0 and v[7] == 3.0 and np.isnan(v[0]) and meta["floor"] == 10.0
    r = c.get("/api/layer/6/field", params={"name": "steer_kl"})
    v = np.frombuffer(r.content, np.float32)
    assert v[100] == pytest.approx(0.9) and np.isnan(v[7])                          # 大きい方の KL(符号 +)
    assert np.frombuffer(c.get("/api/layer/6/field", params={"name": "named"}).content, np.float32)[100] == 1.0
    assert not json.loads(c.get("/api/layer/5/field", params={"name": "sweep"}).headers["x-meta"])["has"]
    assert c.get("/api/layer/6/field", params={"name": "nope"}).status_code == 400
    h = c.get("/api/layer/6/hover")
    ids = np.frombuffer(h.content[:n_ff * 24 * 4], np.int32).reshape(n_ff, 24)
    kl = np.frombuffer(h.content[n_ff * 24 * 4:], np.float32).reshape(n_ff, 2)
    assert list(ids[100, :6]) == list(range(1000, 1006)) and list(ids[100, 6:12]) == list(range(2000, 2006))
    assert (ids[7] == -1).all() and kl[100, 0] == pytest.approx(0.9)
    u = c.get("/api/layer/6/units").json()["runs"][0]
    assert u["floor"] == 10.0 and u["units"][0]["name"] == "宇宙人" and u["units"][0]["words"] == ["alien", "planet"]
    d = c.get("/api/unit", params={"layer": 6, "n": 100}).json()
    ex = d["sweep"]["examples"]
    assert [c_["dose"] for c_ in ex["columns"]] == ["-0.7", "+0.8"]                 # 2 つ以上のプロンプトで共有される dose
    rows = {r_["prompt"]: r_ for r_ in ex["rows"]}
    assert list(rows) == ["who", "snake", "free"]                                    # meta の prompts の順
    assert rows["free"]["extra"][0]["dose"] == 3.0 and not rows["free"]["cells"]     # プロンプト固有の dose は extra
    assert rows["who"]["baseline"]["text"] == "base who" and rows["snake"]["baseline"] is None
    assert rows["who"]["cells"]["+0.8"]["finish"] == "length"
    assert d["steer_lens"]["kl_up"] == pytest.approx(0.9) and d["steer_lens"]["pct_up"] == 1.0 and d["name"] == "宇宙人"
    assert "token_ids" not in json.dumps(d)                                          # 重い欄は返さない
    rd = c.get("/api/unit", params={"layer": 6, "n": 0, "kind": "random"}).json()
    assert rd["sweep"]["mass"] == 10.0 and rd["steer_lens"] is None
    p = c.get("/api/profile").json()["layers"][6]
    assert p["sweep"]["frac_null95"] == 0.5 and p["steer_kl"]["n"] == 1 and p["coverage"]["steer_lens"]
    assert c.get("/api/search", params={"q": "alien", "mode": "words"}).json()["hits"][0] == {
        "layer": 6, "neuron": 100, "where": ["sweep:L6-quick"], "best": {"where": "sweep:L6-quick", "pos": 1, "token": "alien"}}
    voc = c.get("/api/vocab").json()
    hit = c.get("/api/search", params={"q": voc[1003], "mode": "lens"}).json()["hits"]
    assert any(h["neuron"] == 100 and h["best"]["where"] == "steer+" and h["best"]["pos"] == 4 for h in hit)
    bad = [i for i, p_ in enumerate(s.vocab.pieces) if p_ and not _utf8(p_)]
    assert bad and all(voc[i].startswith("<0x") for i in bad)                          # 途中で切れたバイト列は <0x..> で見せる
    r = c.post("/api/name", json={"layer": 6, "neuron": 7, "name": " the "}).json()
    assert r["name"] == "the" and r["names"]["6:7"] == "the"
    assert c.get("/api/layer/6/units").json()["runs"][0]["units"][1]["name"] == "the"   # 名前を変えたら読み直す


def _utf8(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


@pytest.fixture(autouse=True)
def _close_sessions(monkeypatch):
    """テストの中で作った Session を、テストの終わりに閉じる(module の fixture のサーバの Session は対象外)。閉じないと
    0.6B のモデルが 1 つずつ GPU に残り、ファイルの後半で VRAM(12 GB)が尽きて、ブラウザの WebGL や後のテストが遅くなって落ちる。
    その Session でアプリ(create_app)を作っていれば、走っているジョブを止め、モデルの処理の終わりを GPU の調停で待ってから
    閉じる(60 秒まで。終わらなければ閉じずに進む)。"""
    import gne.ui.server as ui_server
    from gne.api import Session
    sessions, apps = [], []
    init, make = Session.__init__, ui_server.create_app

    def track_init(self, *a, **kw):
        init(self, *a, **kw)
        sessions.append(self)

    def track_app(*a, **kw):
        app = make(*a, **kw)
        apps.append(app)
        return app
    monkeypatch.setattr(Session, "__init__", track_init)
    monkeypatch.setattr(ui_server, "create_app", track_app)
    yield
    by_session = {id(app.state.session): app for app in apps}
    for app in apps:
        for j in app.state.live.jobs.list():
            if j.status in ("queued", "running"):
                app.state.live.jobs.cancel(j.id)
    for s in sessions:
        app = by_session.get(id(s))
        if app is None:
            s.close()
            continue
        closed = threading.Event()

        def close(s=s, arbiter=app.state.live.arbiter, closed=closed):
            with arbiter.interactive():
                s.close()
            closed.set()
        threading.Thread(target=close, daemon=True).start()
        closed.wait(60)


def _free_port() -> int:
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]

def _recipe_tab(pg, tab="units"):
    """右のレシピのパネルを開いて、タブを選ぶ(units 中身 / try 試す / out 書き出す)。中身は保存したレシピの一覧も取り直す。"""
    if pg.evaluate("document.querySelector('#recipe').classList.contains('collapsed')"):
        pg.click("#recipe .toggle")
    pg.click(f"#recipe-tabs button[data-rtab={tab}]")


def _seed(s, L: int = 6) -> None:
    """U1 用の小さな保存データ: 層 L の sweep(ニューロン 100 と 7、対照 1)、steer lens(100 だけ)、名前。"""
    import json

    from gne.lens import SteerLens
    mid = s.meta.model_id
    runs = s.store.runs_dir(mid)
    runs.mkdir(parents=True, exist_ok=True)
    dose = {"-0.7": {"rel_push": 0.7, "distinct4": 0.98, "capped": 0.5, "logprob": -0.3},
            "+0.8": {"rel_push": 0.8, "distinct4": 0.9, "capped": 1.0, "logprob": -0.5}}
    units = [
        {"kind": "neuron", "neuron": 100, "layer": L, "rank": 1, "z": 5.0, "mass": 20.0, "null_pct": 1.0,
         "words": [["alien", 4, 5.0], ["planet", 2, 2.0]], "per_prompt": {"alien": ["who"], "planet": ["who"]}, "dose": dose},
        {"kind": "neuron", "neuron": 7, "layer": L, "rank": 2, "z": 0.5, "mass": 3.0, "null_pct": 0.1,
         "words": [["the", 2, 0.5]], "per_prompt": {"the": ["who"]}, "dose": dose},
        {"kind": "random", "neuron": 0, "layer": L, "rank": None, "z": 1.0, "mass": 10.0, "null_pct": None,
         "words": [["and", 2, 1.0]], "per_prompt": {"and": ["who"]}, "dose": dose}]
    (runs / f"L{L}-quick.units.jsonl").write_text("".join(json.dumps(u) + "\n" for u in units))
    recs = [{"kind": "baseline", "neuron": None, "prompt": p, "dose": 0.0, "rel_push": 0.0, "text": f"base {p}", "finish": "eog"}
            for p in ("who", "free")]
    for kind, n in (("neuron", 100), ("neuron", 7), ("random", 0)):
        for p, d in (("who", -0.7), ("who", 0.8), ("snake", -0.7), ("snake", 0.8), ("free", 3.0)):
            recs.append({"kind": kind, "neuron": n, "prompt": p, "dose": d, "rel_push": abs(d), "text": f"{kind} {n} {p} {d}",
                         "finish": "length" if d > 0 else "eog", "token_ids": [1, 2, 3]})
    (runs / f"L{L}-quick.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (runs / f"L{L}-quick.meta.json").write_text(json.dumps({"layer": L, "design": "quick", "prompts": {
        "who": "Who are you?", "snake": "Write a snake game.", "free": "Write whatever you like."}}))
    cfg = {"dose": 0.8, "contexts": None, "top_k": 12, "random_controls": 4, "seed": 0}
    sl = SteerLens.empty(L, s.meta.n_ff_at(L), cfg)
    sl.done[100] = True
    sl.up_ids[100] = np.arange(1000, 1012)
    sl.down_ids[100] = np.arange(2000, 2012)
    sl.kl_pos[100], sl.kl_neg[100] = 0.9, 0.1
    sl.ctrl_done[:] = True
    sl.ctrl_kl_pos[:] = [0.1, 0.2, 0.3, 0.4]
    sl.ctrl_kl_neg[:] = [0.1, 0.2, 0.3, 0.4]
    d = s.store.lens_dir(mid)
    d.mkdir(parents=True, exist_ok=True)
    sl.save(d / f"steer_L{L}.npz")
    s.set_name(L, 100, "宇宙人")
    _seed_heads(s)


HEADS_CONTRAST = ROOT / "configs" / "contrast" / "yandere_react.yaml"


def _seed_heads(s, name: str = "seed") -> list[dict]:
    """ヘッドの対照分析(gne heads contrast の出力の形)の小さな保存データ: 層 0〜26 の全ヘッド、L12 H3 が 1 位。"""
    rows = []
    for L in range(s.meta.n_layer - 1):
        for h in range(s.weights.arch.layers[L].n_head):
            top = (L, h) == (12, 3)
            rows.append({"layer": L, "head": h, "kind": "attn", "trigger_mass": 0.05 if top else 0.001 * ((L + h) % 7 - 3),
                         "pattern_shift": 0.2 if top else 0.01 * (h % 5), "out_shift": 0.5 if top else 0.02 * (L % 9), "dir_cos": None,
                         "z": 5.0 if top else 2.0 - 0.01 * (L * 16 + h)})
    rows.sort(key=lambda x: -x["z"])
    zs = np.sort([x["z"] for x in rows])
    for x in rows:
        x["null_pct"] = float(np.searchsorted(zs, x["z"], "left") / zs.size)
    d = s.store.heads_dir(s.meta.model_id)
    (d / f"{name}.candidates.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    (d / f"{name}.json").write_text(json.dumps({
        "name": name, "contrast": str(HEADS_CONTRAST), "direction": None, "layers": list(range(s.meta.n_layer - 1)), "n_prompts": 1,
        "n_with_triggers": 1, "floors": {"trigger_mass": 0.01, "pattern_shift": 0.05, "out_shift": 0.3}, "null_stats": {}, "top": rows[:50],
        "n_heads": len(rows), "samples": [{"prompt": "I had lunch with Emily from work today. She's really funny.", "response": "Oh?",
                                           "n_tokens": 40, "n_response": 3, "triggers": [" Emily"]}]}))
    return rows


def _seed_writers(s, name: str = "wseed") -> None:
    """書き手の分析(gne writers の出力の形)の小さな保存データ: 参照 L20、L14 N77 が 1 位。"""
    rng = np.random.default_rng(3)
    ref = 20
    per, top = {}, []
    for L in range(ref + 1):
        n = s.meta.n_ff_at(L)
        c = (rng.standard_normal(n) * 0.01).astype(np.float32)
        if L == 14:
            c[77] = 0.9
        per[L] = {"contrib": c, "act_a": rng.standard_normal(n).astype(np.float32), "act_b": rng.standard_normal(n).astype(np.float32),
                  "act_diff": rng.standard_normal(n).astype(np.float32), "align": (rng.standard_normal(n) * 0.1).astype(np.float32)}
    d = s.store.root / "writers" / s.meta.model_id
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{name}.npz", ref_layer=ref, ref_norm=10.0, **{f"{k}_{L}": v for L, p in per.items() for k, v in p.items()})
    allc = sorted(((L, i, float(p["contrib"][i])) for L, p in per.items() for i in range(p["contrib"].size)), key=lambda x: -x[2])
    row = lambda L, i, c: {"layer": L, "neuron": i, "contrib": c, "share": c / 10.0, "act_a": 1.0, "act_b": 0.2, "act_diff": 0.8, "align": 0.1}  # noqa: E731
    top = [row(*x) for x in allc[:20]]
    layers = [{"layer": L, "mixer": "attention", "r_norm": 1.0 + L, "cos_ref": L / ref, "mlp_norm": 1.0, "mixer_norm": 1.0,
               "mlp_to_ref": 0.3, "mixer_to_ref": 0.2, "mlp_check": 0.004} for L in range(ref + 1)]
    (d / f"{name}.json").write_text(json.dumps({
        "name": name, "contrast": "configs/contrast/possessive.yaml", "contrast_sha": "x", "ref_layer": ref, "ref_norm": 10.0,
        "n_tokens": {"A": 100, "B": 90}, "n_convs": {"A": 4, "B": 4}, "replies": {"A": "generated", "B": "generated"},
        "post_ffn_norm": False, "totals": {"embed": 0.0, "mixer": 0.42, "mlp": 0.58, "top10": 0.12, "top100": 0.3, "top1000": 0.5},
        "mlp_check_max": 0.004, "layers": layers, "top": top, "bottom": [row(*x) for x in allc[::-1][:5]],
        "samples": [{"prompt": "I'm going out tonight.", "A": "With who?", "B": "Have fun!"}], "seconds": 1.0}))


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip("モデルがありません")
    import uvicorn

    from gne.api import Session
    from gne.ui.server import create_app
    port = _free_port()
    s = Session(MODEL, root=tmp_path_factory.mktemp("ui_root"))
    _seed(s)
    _seed_writers(s)
    srv = uvicorn.Server(uvicorn.Config(create_app(s), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    th.join(5)


class _JaBrowser:
    """ページを既定で日本語の環境(locale ja-JP)で開くブラウザ。UI はブラウザの言語で JP / EN を選ぶので、
    文言を日本語で確かめるテストはこれで開く。英語は ?lang=en。"""

    def __init__(self, b):
        self._b = b

    def new_page(self, **kw):
        return self._b.new_page(**{"locale": "ja-JP", **kw})

    def __getattr__(self, k):
        return getattr(self._b, k)


@pytest.fixture(scope="module")
def browser():
    if not CHROME.exists():
        pytest.skip(f"Chromium がありません({CHROME}、GNE_CHROME で指定)")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=str(CHROME), headless=True,
                              args=["--use-gl=angle", "--use-angle=gl", "--enable-gpu", "--ignore-gpu-blocklist",
                                    "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],   # マイクのテスト
                              env={**os.environ, "MESA_D3D12_DEFAULT_ADAPTER_NAME": "NVIDIA"})
        yield _JaBrowser(b)
        b.close()


@pytest.mark.model
def test_map_renders_and_hover_picks_the_cell_under_the_mouse(server, browser):
    pg = browser.new_page(viewport={"width": 1280, "height": 800})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(server + "/")
    pg.wait_for_function("window.gne && (window.gne.ready || window.gne.error)", timeout=60000)
    assert not errors and not pg.evaluate("window.gne.error")
    n_layer = pg.evaluate("window.gne.arch.n_layer")
    assert pg.evaluate("window.gne.map.nPoints") == n_layer * 3072
    pg.fill("#jump", "L5 N100")
    pg.press("#jump", "Enter")
    pg.wait_for_timeout(300)
    assert pg.evaluate("window.gne.map.front") == 5
    assert pg.evaluate("document.querySelector('#layer-select').value") == "5"          # 選択欄も連動する
    pg.select_option("#layer-select", "7")
    pg.wait_for_timeout(200)
    assert pg.evaluate("window.gne.map.front") == 7
    pg.fill("#jump", "L5 N100")
    pg.press("#jump", "Enter")
    pg.wait_for_timeout(200)
    for n in (0, 100, 3071):
        sp = pg.evaluate(f"window.gne.map.screenOf(5, {n})")
        pg.mouse.move(sp["x"], sp["y"])
        pg.wait_for_timeout(80)
        h = pg.evaluate("window.gne.map.hover")
        assert h and (h["layer"], h["neuron"]) == (5, n), (n, h)
        assert pg.evaluate("document.querySelector('#tooltip').textContent") == f"L5 N{n}"
    pg.click("#reset-view")
    assert pg.evaluate("window.gne.map.front") is None
    pg.wait_for_timeout(200)                                                             # overlay は次のフレームで書き換わる
    labels = pg.evaluate("[...document.querySelectorAll('#overlay span')].map(s => s.textContent)")
    assert labels[0] == "L0" and f"L{n_layer - 1}" in labels and "入力側" in labels and "出力側" in labels   # 最後の層は必ず出す
    assert len(labels) >= 8                                                              # 正面ビューから戻っても全体が見える距離
    pg.close()


@pytest.mark.model
def test_u1_overlay_tooltip_list_and_neuron_panel(server, browser):
    pg = browser.new_page(viewport={"width": 1400, "height": 860})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(server + "/")
    pg.wait_for_function("window.gne && (window.gne.overlayReady || window.gne.error)", timeout=60000)
    assert pg.evaluate("window.gne.overlay.field") == "sweep"                        # sweep があれば既定は sweep
    rgba = lambda n: pg.evaluate(f"(() => {{ const m = window.gne.map, o = (m.offsets[6] + {n}) * 4; "   # noqa: E731
                                 f"return Array.from(m.colorAttr.array.slice(o, o + 4)); }})()")
    assert rgba(100) == [255, 255, 255, 255]                                         # 名前付きは白
    assert rgba(7) == [30, 37, 48, 0]                                                # 基準値(対照の最大 mass)以下
    assert rgba(0) == [16, 20, 27, 0]                                                # データ無し
    assert pg.evaluate("document.querySelectorAll('#rank tbody tr').length") == 2    # random は既定で隠す
    pg.click("#rank tbody tr[data-n='100']")
    pg.wait_for_function("document.querySelector('#neuron-body .n-id')", timeout=30000)
    assert pg.evaluate("document.querySelector('#neuron-body .n-id').textContent") == "L6 N100"
    assert pg.evaluate("window.gne.map.front") == 6
    assert pg.evaluate("document.querySelectorAll('#neuron-body table.ex td.cell').length") == 8   # who 3 + snake 3(ベースライン無しは空)+ free 2(結合)
    assert pg.evaluate("document.querySelector('#neuron-body td.span').colSpan") == 2
    sp = pg.evaluate("window.gne.map.screenOf(6, 100)")
    pg.mouse.move(sp["x"], sp["y"])
    pg.wait_for_function("window.gne.vocab && !document.querySelector('#tooltip').textContent.includes('読み込み中')",
                         timeout=30000)
    pg.mouse.move(sp["x"] + 0.5, sp["y"])
    pg.wait_for_timeout(200)
    tip = pg.evaluate("document.querySelector('#tooltip').innerText")
    assert "宇宙人" in tip and "alien · planet" in tip and "steer+" in tip and "KL 0.900" in tip
    pg.select_option("#color-select", "steer_kl")
    pg.wait_for_function("window.gne.overlay.field === 'steer_kl' && window.gne.overlay.scale", timeout=30000)
    assert rgba(100)[3] > 0 and rgba(7) == [16, 20, 27, 0]
    pg.fill("#search-q", "alien")
    pg.select_option("#search-mode", "words")
    pg.press("#search-q", "Enter")
    pg.wait_for_function("document.querySelector('#search-list tbody tr')", timeout=30000)
    assert pg.evaluate("window.gne.overlay.hits.length") == 1
    assert pg.evaluate("document.querySelector('#search-list tbody tr').dataset.n") == "100"
    pg.click("#search-close")
    pg.wait_for_function("window.gne.overlay.field === 'steer_kl'", timeout=30000)   # 閉じると前の overlay に戻る
    pg.click("#views button[data-view=layers]")
    pg.wait_for_function("document.querySelector('#lv-canvas') && document.querySelector('#lv-canvas').height > 100", timeout=60000)
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_synthetic_70b_point_count(server, browser):
    pg = browser.new_page(viewport={"width": 1280, "height": 800})
    pg.goto(server + "/?synthetic=80x28672")
    pg.wait_for_function("window.gne && window.gne.ready", timeout=60000)
    assert pg.evaluate("window.gne.map.nPoints") == 80 * 28672
    pg.close()


# --- U2: ライブ機能 -----------------------------------------------------------------------------------------

def _sse_events(c, url, body):
    with c.stream("POST", url, json=body) as r:
        assert r.status_code == 200, r.read()
        return [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]


@pytest.mark.model
@pytest.mark.skipif(not MODEL.exists(), reason="モデルがありません")
def test_u2_live_api(tmp_path):
    from fastapi.testclient import TestClient

    from gne.api import Session
    from gne.ui.server import create_app
    s = Session(MODEL, root=tmp_path)
    c = TestClient(create_app(s, warm=False))
    msgs = [{"role": "user", "content": "Who are you?"}]
    ev = _sse_events(c, "/api/steer", {"max_tokens": 8, "columns": [
        {"units": ["14:100:-2"], "messages": msgs}, {"units": [], "messages": msgs}, {"units": ["14:100:+2"], "messages": msgs}]})
    kinds = [e["type"] for e in ev]
    assert kinds[0] == "start" and kinds[1] == "gpu" and kinds[-1] == "done"
    ends = [e for e in ev if e["type"] == "end"]
    assert [e["col"] for e in ends] == [0, 1, 2] and all(e["n_gen"] == 8 for e in ends)
    for e in ends:                                                                   # 逐次の断片をつなぐと全文になる
        assert "".join(x["text"] for x in ev if x["type"] == "text" and x["col"] == e["col"]) == e["text"]
    base = s.generate(None, msgs, max_tokens=8)["text"]
    assert ends[1]["text"] == base                                                  # 無介入の列は普通の生成と同じ
    assert c.post("/api/steer", json={"columns": [{"units": ["999:1:+1"], "messages": msgs}]}).status_code == 400
    assert c.post("/api/steer/nope/stop").json() == {"id": "nope", "stopped": False}

    f = c.post("/api/fires", json={"text": "The aliens landed on Mars."}).json()
    n_ff, L = s.meta.n_ff_at(0), s.meta.n_layer
    r = c.get(f"/api/fires/{f['id']}/field", params={"tok": 2})
    v = np.frombuffer(r.content, np.float32).reshape(L, n_ff)
    top = f["top"][1]["neurons"][0]                                                  # position 2 の最上位(top は position 1 から)
    assert v[top["layer"], top["neuron"]] == pytest.approx(top["score"], rel=1e-5)
    mx = np.frombuffer(c.get(f"/api/fires/{f['id']}/field", params={"tok": -1}).content, np.float32).reshape(L, n_ff)
    ok = np.isfinite(v)                                                              # 最終層は捕捉しない(NaN)
    assert ok.any() and (np.abs(mx[ok]) >= np.abs(v[ok]) - 1e-6).all()               # 全トークンの |最大| はどのトークンより大きい
    nn = c.get(f"/api/fires/{f['id']}/neurons", params={"tok": 2, "sign": "+", "thr": 1.0, "limit": 5}).json()
    assert nn["neurons"] and all(x["score"] >= 1.0 for x in nn["neurons"])
    assert c.get("/api/fires/zzz/field").status_code == 400
    rd = c.post("/api/reads", json={"layer": top["layer"], "neuron": top["neuron"], "text": "The aliens landed on Mars."}).json()
    assert rd["score"][1] == pytest.approx(top["score"], rel=1e-4)                   # reads you と fires は同じ値

    assert c.post("/api/recipes/u2r", json={"units": ["14:100:+1", "13:7:-0.5"]}).status_code == 200
    rs = c.get("/api/recipes").json()
    assert rs[0]["name"] == "u2r" and [u["dose"] for u in rs[0]["units"]] == [1.0, -0.5] and rs[0]["this_model"]
    ex = c.post("/api/export", json={"units": ["14:100:+1", "13:7:-0.5"], "name": "u2x"}).json()
    dl = c.get(ex["url"])
    assert dl.status_code == 200 and dl.content[:4] == b"GGUF" and "attachment" in dl.headers["content-disposition"]
    assert c.get("/api/export/file/..%2F..%2Fsecret.cvec.gguf").status_code in (400, 404)
    assert c.post("/api/export", json={"units": ["14:100:+1"], "name": "../x"}).status_code == 400

    j = c.post("/api/jobs", json={"kind": "lens_logit", "params": {"layer": 3, "neurons": "0-255"}}).json()
    with c.stream("GET", f"/api/jobs/{j['id']}/events") as r:
        views = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert views[-1]["status"] == "done" and views[-1]["result"]["n_neurons"] == 256
    assert c.post("/api/jobs", json={"kind": "nope"}).status_code == 400


@pytest.mark.model
@pytest.mark.skipif(not MODEL.exists(), reason="モデルがありません")
def test_live_audio_parts(tmp_path):
    """UI のチャットに音声の部品(base64)を送れる。mmproj の無いモデルなら NO_MMPROJ のエラー、path は読まない。"""
    import base64

    from fastapi.testclient import TestClient

    from gne.api import Session
    from gne.ui.server import create_app
    s = Session(MODEL, root=tmp_path)
    c = TestClient(create_app(s, warm=False))
    b64 = base64.b64encode(b"RIFF....WAVE").decode()
    msgs = [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]}]
    ev = _sse_events(c, "/api/steer", {"max_tokens": 4, "system": "Respond in text.", "columns": [{"units": [], "messages": msgs}]})
    err = [e for e in ev if e["type"] == "error"]
    assert err and err[0]["error"]["code"] == "NO_MMPROJ"
    bad = [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"path": "/etc/hostname"}}]}]
    assert c.post("/api/steer", json={"columns": [{"units": [], "messages": bad}]}).status_code == 400


@pytest.mark.model
@pytest.mark.skipif(not MODEL.exists(), reason="モデルがありません")
def test_u2_chat_preempts_running_job(tmp_path):
    """ジョブ(steer lens)の実行中に送ったチャットは、ユニットの切れ目で GPU を受け取り、すぐ始まる(§8.1)。"""
    from fastapi.testclient import TestClient

    from gne.api import Session
    from gne.ui.server import create_app
    s = Session(MODEL, root=tmp_path)
    s.stats_resid(tokens=2048)
    c = TestClient(create_app(s, warm=False))
    j = c.post("/api/jobs", json={"kind": "lens_steer", "params": {"layer": 10, "neurons": "0-2047", "random_controls": 4}}).json()
    for _ in range(200):
        v = next(x for x in c.get("/api/jobs").json() if x["id"] == j["id"])
        if v["status"] == "running" and v["done"] >= 3:
            break
        time.sleep(0.1)
    msgs = [{"role": "user", "content": "Hi"}]
    ev = _sse_events(c, "/api/steer", {"max_tokens": 4, "columns": [{"units": [], "messages": msgs}]})
    gpu = next(e for e in ev if e["type"] == "gpu")
    assert gpu["waited_s"] < 2.0
    v = next(x for x in c.get("/api/jobs").json() if x["id"] == j["id"])
    assert v["yields"] >= 1 and v["status"] == "running"                             # 譲ったあと続きから走っている
    c.delete(f"/api/jobs/{j['id']}")


@pytest.mark.model
def test_chat_mic_sends_wav(server, browser):
    """マイク(Chromium の偽の録音デバイス)で話すと、16 kHz の WAV を input_audio で送り、system も添える。
    テストのモデルは音声を聞けないので、エラーを状態の欄に出す。英語の辞書に新しい文言がそろっている。"""
    import base64
    ctx = browser.new_context(permissions=["microphone"], locale="ja-JP", viewport={"width": 1500, "height": 860})
    pg = ctx.new_page()
    errors, sent = [], []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("request", lambda r: sent.append(r.post_data_json) if r.url.endswith("/api/steer") and r.method == "POST" else None)

    def open_chat(url):
        pg.goto(url)
        pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
        pg.fill("#jump", "L6 N100")
        pg.press("#jump", "Enter")
        pg.wait_for_function("document.querySelector('#neuron-body .n-id')", timeout=30000)
        pg.click("#neuron-body button[data-tab=chat]")
        pg.wait_for_selector("#ch-mic", timeout=10000)
    open_chat(server + "/")
    pg.fill("#ch-system", "Respond in text.")
    pg.dispatch_event("#ch-system", "change")
    pg.fill("#ch-text", "and this")
    pg.dispatch_event("#ch-text", "input")
    pg.click("#ch-mic")
    pg.wait_for_function("document.querySelector('#ch-mic').classList.contains('rec')", timeout=10000)
    pg.wait_for_timeout(1200)
    pg.click("#ch-mic")
    pg.wait_for_function("document.querySelector('.ch-user audio')", timeout=30000)
    pg.wait_for_function("!document.querySelector('#ch-send').disabled", timeout=60000)
    assert sent, "送っていない"
    body = sent[-1]
    assert body["system"] == "Respond in text."
    parts = body["columns"][0]["messages"][-1]["content"]
    assert parts[0]["type"] == "input_audio" and parts[1] == {"type": "text", "text": "and this"}
    raw = base64.b64decode(parts[0]["input_audio"]["data"])
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE" and int.from_bytes(raw[24:28], "little") == 16000
    n_sec = (len(raw) - 44) / 2 / 16000
    assert 0.5 < n_sec < 5, n_sec
    assert "mmproj" in pg.evaluate("document.querySelector('#ch-status').textContent")
    open_chat(server + "/?lang=en")
    assert "talk" in pg.evaluate("document.querySelector('#ch-mic').textContent")
    assert pg.evaluate("[...window.gneI18n.missing]") == []
    assert not errors, errors
    ctx.close()


@pytest.mark.model
def test_enter_while_composing_does_nothing(server, browser):
    """日本語の変換を確定する Enter / Esc(isComposing)では、jump・検索・名前の保存が走らない。ふつうの Enter では走る。"""
    pg = browser.new_page()
    errors, names = [], []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("request", lambda rq: names.append(rq.url) if rq.url.endswith("/api/name") else None)
    pg.goto(server + "/")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    ime = {"key": "Enter", "isComposing": True}
    pg.fill("#jump", "L6 N7")
    pg.dispatch_event("#jump", "keydown", ime)
    pg.wait_for_timeout(300)
    assert not pg.evaluate("document.querySelector('#neuron-body .n-id')")
    pg.press("#jump", "Enter")
    pg.wait_for_function("document.querySelector('#neuron-body .n-id')", timeout=30000)
    pg.fill("#neuron-body .n-name", "かり")                                          # 名前: 変換の確定では保存しない
    pg.dispatch_event("#neuron-body .n-name", "keydown", ime)
    pg.wait_for_timeout(300)
    assert names == []
    pg.fill("#neuron-body .n-name", "")                                              # 元の空に戻す(欄を出ても保存しない)
    pg.fill("#search-q", "alien")
    pg.dispatch_event("#search-q", "keydown", ime)
    pg.dispatch_event("#search-q", "keydown", {"key": "Escape", "isComposing": True})   # 変換の取り消しで欄を消さない
    pg.wait_for_timeout(300)
    assert pg.input_value("#search-q") == "alien" and not pg.evaluate("window.gne.overlay.field.startsWith('search:')")
    pg.press("#search-q", "Enter")
    pg.wait_for_function("window.gne.overlay.field === 'search:alien'", timeout=30000)
    assert names == [] and not errors, errors
    pg.close()


@pytest.mark.model
def test_saved_recipe_load_asks_before_replacing(server, browser):
    """保存したレシピを読むとき、いまのレシピが空でなければ確かめる(断れば今のレシピを残し、選択も戻す)。空なら確かめない。"""
    pg = browser.new_page()
    errors, asked = [], []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(server + "/")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    pg.evaluate("""fetch('/api/recipes/ui_replace', {method: 'POST', headers: {'Content-Type': 'application/json'},
                   body: JSON.stringify({units: ['6:7:+0.5']})})""")
    pg.evaluate("window.gne.recipe.clear(); window.gne.recipe.add(6, 100, 1); document.querySelector('#list').classList.remove('collapsed')")
    _recipe_tab(pg)
    pg.wait_for_function("[...document.querySelectorAll('#rc-load option')].some((o) => o.value === 'ui_replace')", timeout=10000)
    pg.once("dialog", lambda d: (asked.append(d.message), d.dismiss()))
    pg.select_option("#rc-load", "ui_replace")
    assert len(asked) == 1 and pg.evaluate("window.gne.recipe.specs()") == ["6:100:+1"] and pg.input_value("#rc-load") == ""
    pg.once("dialog", lambda d: (asked.append(d.message), d.accept()))
    pg.select_option("#rc-load", "ui_replace")
    assert len(asked) == 2 and pg.evaluate("window.gne.recipe.specs()") == ["6:7:+0.5"]
    pg.evaluate("window.gne.recipe.clear()")
    _open_list(pg, "rank")
    _recipe_tab(pg)
    pg.wait_for_function("[...document.querySelectorAll('#rc-load option')].some((o) => o.value === 'ui_replace')", timeout=10000)
    pg.select_option("#rc-load", "ui_replace")                                       # 空なら確かめずに読む
    assert len(asked) == 2 and pg.evaluate("window.gne.recipe.specs()") == ["6:7:+0.5"]
    pg.evaluate("window.gne.recipe.clear()")
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_top_bar_opens_each_screen(server, browser):
    """上の帯の 3 群(探す・作る・確かめる)から、どの画面も直接開ける。いまの画面だけに印(aria-current)。方向・CHiPS・鍵・FiSH・
    強さの候補は direction ビューの中身を入れ替えて出す(中の切り替えは無い)。地図の上の帯(層・並べ方・色・線・この層で計算)は地図だけ。
    方向の画面は開いただけでは測らない。英語の画面でも文言は辞書にある。"""
    want = {"map": "#map-bar", "layers": "#layers-view",                       # 地図の上の帯は地図のときだけ見える(地図の上に重ねるほかの画面と分ける) "writers": "#writers-view", "embed": "#embed-view", "eval": "#eval-view",
            "direction": "#dv-dir", "chips": "#dv-pipe", "keys": "#dv-key", "fish": "#dv-fish", "cand": "#dv-cand"}
    for lang in ("ja", "en"):
        pg = browser.new_page(viewport={"width": 1500, "height": 860})
        errors, posts = [], []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("request", lambda rq: posts.append(rq.url) if rq.method == "POST" and rq.url.endswith("/score") else None)
        pg.route("**/api/evals", lambda route, req: route.fulfill(status=200, content_type="application/json", body="[]"))
        pg.goto(server + f"/?lang={lang}")
        pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
        groups = pg.evaluate("[...document.querySelectorAll('#views .grp')].map(g => [...g.querySelectorAll('button[data-view]')].map(b => b.dataset.view))")
        assert groups == [["map", "layers", "writers", "embed"], ["direction", "chips", "keys", "fish"], ["eval", "cand"]]
        assert not pg.evaluate("document.querySelector('#dv-mode')")
        for v, sel in [*want.items(), ("map", "#map-bar")]:
            pg.click(f"#views button[data-view={v}]")
            pg.wait_for_function(f"document.querySelector('{sel}') && document.querySelector('{sel}').offsetParent !== null", timeout=30000)
            cur = pg.evaluate("[...document.querySelectorAll('#views button[aria-current=true]')].map(b => b.dataset.view)")
            assert cur == [v], (v, cur)
            shown = pg.evaluate("Object.entries(" + json.dumps(want) + ").filter(([k, s]) => { const e = document.querySelector(s); return e && e.offsetParent !== null; }).map(([k]) => k)")
            assert shown == [v], (v, shown)                                             # ほかの画面は隠れている
            if v in ("direction", "chips", "keys", "fish", "cand"):
                assert pg.evaluate("window.gne.directionView.mode") == {"direction": "dir", "chips": "pipe", "keys": "key", "fish": "fish", "cand": "cand"}[v]
        pg.wait_for_load_state("networkidle")
        assert posts == []                                                             # 方向の画面を開いても測らない
        # 地図の上の「この層で計算」: 層の右クリックと同じメニュー。Esc で閉じる
        pg.select_option("#layer-select", "6")
        pg.wait_for_function("window.gne.map.front === 6", timeout=10000)
        pg.click("#map-compute")
        pg.wait_for_selector("#compute-menu", timeout=5000)
        head = pg.evaluate("document.querySelector('#compute-menu .cm-head').textContent")
        assert ("L6 を計算する" if lang == "ja" else "L6") in head
        assert pg.evaluate("document.querySelectorAll('#compute-menu .cm-item').length") == 8
        pg.keyboard.press("Escape")
        assert not pg.evaluate("document.querySelector('#compute-menu')")
        if lang == "en":
            assert pg.evaluate("[...window.gneI18n.missing]") == []
            assert pg.evaluate("document.querySelector('#views button[data-view=map]').textContent") == "map"
        assert not errors, errors
        pg.close()


@pytest.mark.model
def test_recipe_panel_adds_from_any_entry(server, browser):
    """レシピは右のパネル(中身・試す・書き出す)。細くたたんでいても、くわしく見るの「レシピに足す」でパネルが開き、数が増える。
    外しただけでは開かない。広げる・戻す、タブは ← → でも替わる。"""
    pg = browser.new_page(viewport={"width": 1500, "height": 860})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(server + "/")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    pg.evaluate("window.gne.recipe.clear()")
    _recipe_tab(pg)
    assert pg.evaluate("document.querySelector('#recipe-count').textContent") == ""
    pg.click("#recipe .toggle")                                                      # 細くたたむ
    assert pg.evaluate("document.querySelector('#recipe').classList.contains('collapsed')")
    pg.fill("#jump", "L6 N100")
    pg.press("#jump", "Enter")
    pg.wait_for_function("document.querySelector('#neuron-body .n-id')", timeout=30000)
    assert pg.evaluate("document.querySelector('#detail-kinds button[data-kind=neuron]').getAttribute('aria-pressed')") == "true"
    pg.click("#neuron-body [data-rc='1']")                                           # レシピに足す(+)
    assert not pg.evaluate("document.querySelector('#recipe').classList.contains('collapsed')")
    assert pg.evaluate("document.querySelector('#recipe-tabs [aria-selected=true]').dataset.rtab") == "units"
    assert pg.evaluate("document.querySelector('#recipe-count').textContent") == "1"
    assert pg.evaluate("document.querySelectorAll('#recipe-list table.rc tr').length") == 1
    pg.fill("#jump", "L12 H3")                                                       # ヘッド: 同じ「くわしく見る」の中身が替わる
    pg.press("#jump", "Enter")
    pg.wait_for_function("document.querySelector('#head-body .n-id')", timeout=30000)
    assert pg.evaluate("document.querySelector('#neuron-body').classList.contains('hidden')")
    pg.click("#recipe .toggle")
    pg.click("#head-body [data-hk='0']")                                             # レシピに足す(×0)
    assert not pg.evaluate("document.querySelector('#recipe').classList.contains('collapsed')")
    assert pg.evaluate("document.querySelector('#recipe-count').textContent") == "2"
    pg.click("#recipe .toggle")
    pg.click("#head-body [data-hk='0']")                                             # もう一度押すと外す(パネルは開かない)
    assert pg.evaluate("document.querySelector('#recipe').classList.contains('collapsed')")
    assert pg.evaluate("document.querySelector('#recipe-count').textContent") == "1"
    pg.click("#detail-kinds button[data-kind=neuron]")                               # 前に開いたニューロンへ戻れる
    assert not pg.evaluate("document.querySelector('#neuron-body').classList.contains('hidden')")
    pg.click("#recipe .toggle")
    w0 = pg.evaluate("document.querySelector('#recipe').getBoundingClientRect().width")
    pg.click("#recipe-wide")
    assert pg.evaluate("document.querySelector('#recipe').getBoundingClientRect().width") > w0 + 100
    pg.click("#recipe-tabs button[data-rtab=units]")
    pg.keyboard.press("ArrowRight")
    assert pg.evaluate("document.querySelector('#recipe-tabs [aria-selected=true]').dataset.rtab") == "try"
    assert pg.evaluate("document.querySelector('#rc-chat').offsetParent !== null")
    pg.keyboard.press("ArrowRight")
    assert pg.evaluate("document.querySelector('#rc-export').offsetParent !== null") and pg.evaluate("document.querySelector('#rc-chat').offsetParent === null")
    pg.click("#recipe-wide")
    pg.evaluate("window.gne.recipe.clear()")
    assert not errors, errors
    pg.close()


def test_chat_single_column_in_browser(server, browser):
    """チャットの 1 列: + の列(レシピあり)だけを生成し、3 列に戻すときは確かめてから会話を消す。選んだ列の数は覚える。"""
    pg = browser.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(server + "/")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    pg.evaluate("window.gne.recipe.clear()")
    pg.fill("#jump", "L6 N100")
    pg.press("#jump", "Enter")
    pg.wait_for_function("document.querySelector('#neuron-body .n-id')", timeout=30000)
    pg.click("#neuron-body [data-rc='1']")
    _recipe_tab(pg)
    _recipe_tab(pg, "try")
    pg.click("#rc-chat")
    pg.wait_for_selector("#ch-text", timeout=30000)
    pg.click("[data-cols='1']")
    assert pg.evaluate("document.querySelectorAll('.ch-head > div').length") == 1
    assert pg.evaluate("document.querySelector('.ch-log').classList.contains('single')")
    pg.fill("#ch-max", "6")
    pg.dispatch_event("#ch-max", "change")
    pg.fill("#ch-text", "Hello")
    pg.click("#ch-send")
    try:
        pg.wait_for_function("!document.querySelector('#ch-send').disabled", timeout=120000)
    except Exception:
        raise AssertionError(f"返事が終わらない: {pg.evaluate('document.querySelector(\"#ch-status\").textContent')} {errors}") from None
    assert not errors, errors
    cells = pg.evaluate("[...document.querySelectorAll('.ch-cell')].map(e => e.className)")
    assert len(cells) == 1 and "plus" in cells[0]
    assert "秒で 1 列を生成" in pg.evaluate("document.querySelector('#ch-status').textContent")
    assert pg.evaluate("localStorage.getItem('gne.chat.single')") == "1"
    asked = []
    pg.on("dialog", lambda d: (asked.append(d.message), d.accept()))
    pg.click("[data-cols='3']")
    pg.wait_for_function("document.querySelectorAll('.ch-head > div').length === 3", timeout=5000)
    assert asked and pg.evaluate("document.querySelectorAll('.ch-cell').length") == 0
    pg.close()


def test_chat_new_conversation_button(server, browser):
    """入力欄の横の「新しい会話」: 会話があれば確かめてから消し(断れば残す)、入力欄に戻る。1 列のまま。
    生成はブラウザの中で差し替える(GPU を使わずに速く確かめるため)。"""
    pg = browser.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    sse = "".join(f"data: {json.dumps(ev)}\n\n" for ev in (
        {"type": "start", "id": "x1", "columns": 1}, {"type": "gpu", "waited_s": 0.0}, {"type": "column", "col": 0, "n_prompt": 9},
        {"type": "text", "col": 0, "text": "Hi there"}, {"type": "end", "col": 0, "text": "Hi there", "finish": "length", "n_gen": 2},
        {"type": "done", "seconds": 0.1}))
    pg.route("**/api/steer", lambda route, req: route.fulfill(status=200, content_type="text/event-stream", body=sse))
    pg.goto(server + "/")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    pg.evaluate("window.gne.recipe.clear()")
    pg.fill("#jump", "L6 N100")
    pg.press("#jump", "Enter")
    pg.wait_for_function("document.querySelector('#neuron-body .n-id')", timeout=30000)
    pg.click("#neuron-body [data-rc='1']")
    _recipe_tab(pg)
    _recipe_tab(pg, "try")
    pg.click("#rc-chat")
    pg.wait_for_selector("#ch-text", timeout=30000)
    pg.click("[data-cols='1']")
    assert pg.evaluate("!!document.querySelector('.ch-input #ch-reset')")          # 送る・止めるの横
    for text in ("Hello", "Again"):
        pg.fill("#ch-text", text)
        pg.click("#ch-send")
        pg.wait_for_function("!document.querySelector('#ch-send').disabled", timeout=30000)
    assert pg.evaluate("document.querySelectorAll('.ch-cell').length") == 2
    pg.once("dialog", lambda d: d.dismiss())                                       # 断れば残す
    pg.click("#ch-reset")
    assert pg.evaluate("document.querySelectorAll('.ch-cell').length") == 2
    asked = []
    pg.once("dialog", lambda d: (asked.append(d.message), d.accept()))
    pg.click("#ch-reset")
    pg.wait_for_function("document.querySelectorAll('.ch-cell').length === 0", timeout=5000)
    assert asked and pg.evaluate("document.activeElement && document.activeElement.id") == "ch-text"
    assert pg.evaluate("document.querySelectorAll('.ch-head > div').length") == 1   # 1 列のまま
    pg.click("#ch-reset")                                                          # 会話が無ければ確かめない
    assert len(asked) == 1 and not errors, errors
    pg.close()


@pytest.mark.model
def test_u2_recipe_chat_fires_in_browser(server, browser):
    pg = browser.new_page(viewport={"width": 1500, "height": 860})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(server + "/")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    pg.evaluate("window.gne.recipe.clear()")
    pg.fill("#jump", "L6 N100")
    pg.press("#jump", "Enter")
    pg.wait_for_function("document.querySelector('#neuron-body .n-id')", timeout=30000)
    pg.click("#neuron-body [data-rc='1']")
    assert pg.evaluate("window.gne.recipe.specs()") == ["6:100:+1"]
    pg.wait_for_timeout(200)
    assert pg.evaluate("document.querySelectorAll('#overlay span.mk.plus').length") == 1    # 地図の印
    _recipe_tab(pg)
    assert pg.evaluate("document.querySelectorAll('#recipe-list table.rc tr').length") == 1
    _recipe_tab(pg, "try")
    pg.click("#rc-chat")
    pg.wait_for_selector("#ch-text", timeout=30000)
    pg.fill("#ch-max", "6")
    pg.dispatch_event("#ch-max", "change")
    pg.select_option("#ch-when", "ramp")                                            # 位置別ステアリング(§6.15)
    pg.wait_for_selector("#ch-when-x", timeout=5000)
    assert pg.evaluate("window.gne.chat.whenSpec()") == "ramp:1.5-0.5"
    pg.select_option("#ch-when", "gen")
    pg.fill("#ch-text", "Hello")
    pg.click("#ch-send")
    pg.wait_for_function("!document.querySelector('#ch-send').disabled", timeout=120000)
    cells = pg.evaluate("[...document.querySelectorAll('.ch-cell .txt')].map(e => e.textContent)")
    assert len(cells) == 3 and all(cells), cells
    assert "秒で 3 列を生成" in pg.evaluate("document.querySelector('#ch-status').textContent")
    _open_list(pg, "fires")
    pg.fill("#fi-text", "The aliens landed on Mars.")
    pg.dispatch_event("#fi-text", "input")
    pg.click("#fi-run")
    pg.wait_for_function("document.querySelector('#fi-list tr[data-l]')", timeout=60000)
    pg.wait_for_function("window.gne.overlay.field.startsWith('fires:')", timeout=30000)     # 色は一覧の後に別に取りに行く
    pg.wait_for_function("(document.querySelector('#fires-list .ch-status') || {}).textContent?.includes('活性統計が無い')",
                         timeout=30000)                                                # 統計の無い層を名指し(一覧の後に出る)
    assert pg.evaluate("document.querySelectorAll('#fi-list tr.weak').length") > 0                        # 一覧の行に印
    assert pg.evaluate("window.gne.overlay.weak.size") > 0 and "枠の比較から外した" in pg.evaluate("document.querySelector('#legend').textContent")
    assert pg.evaluate("[...document.querySelectorAll('#fi-list td.w')].every(td => td.getBoundingClientRect().width > 80)")   # lens 列に幅がある
    pg.click(".chip-t[data-tok='2']")
    pg.wait_for_function("window.gne.overlay.field.includes('landed')", timeout=30000)
    assert not errors, errors
    pg.close()


# --- U3: 配線と配置 -----------------------------------------------------------------------------------------

def _run_job(c, kind, params, timeout=600):
    j = c.post("/api/jobs", json={"kind": kind, "params": params}).json()
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = next(x for x in c.get("/api/jobs").json() if x["id"] == j["id"])
        if v["status"] in ("done", "error", "cancelled"):
            assert v["status"] == "done", v["error"]
            return v
        time.sleep(0.2)
    raise AssertionError(f"{kind} が終わりません")


@pytest.mark.model
@pytest.mark.skipif(not MODEL.exists(), reason="モデルがありません")
def test_u3_connections_web_layout_api(tmp_path):
    from fastapi.testclient import TestClient

    from gne.api import Session
    from gne.connections import ffn_norm_weight
    from gne.ui.server import create_app
    from gne.weights import tensor_to_f32
    s = Session(MODEL, root=tmp_path)
    c = TestClient(create_app(s, warm=False))
    assert c.get("/api/connections", params={"layer": 5, "n": 100}).status_code == 400   # まだ無い
    _run_job(c, "connections", {"k": 8, "window": 2})
    m = c.get("/api/model").json()
    assert m["connections"]["window"] == 2 and m["connections"]["n_pairs"] > 0
    d = c.get("/api/connections", params={"layer": 5, "n": 100, "k": 5}).json()
    assert len(d["down"]) == 5 and all(5 < e["layer"] <= 7 for e in d["down"])
    assert all(3 <= e["layer"] < 5 for e in d["up"])
    w = s.weights                                                                     # 直接計算と照合(層 6 の中での順位と符号)
    v, g = w.down_matrix(5)[:, 100], ffn_norm_weight(w, 6)
    P = ((tensor_to_f32(w.tensor("blk.6.ffn_gate.weight")) * g) @ v) * ((tensor_to_f32(w.tensor("blk.6.ffn_up.weight")) * g) @ v)
    l6 = [e for e in d["down"] if e["layer"] == 6]
    assert l6 and int(np.argmax(np.abs(P))) == l6[0]["neuron"] and np.sign(P[l6[0]["neuron"]]) == l6[0]["sign"]
    r = c.get("/api/web", params={"layer": 5, "n": 100, "hops": 2, "first": 3, "k": 2, "zmin": 0})
    e = np.frombuffer(r.content, np.int32).reshape(-1, 7)
    first = e[e[:, 5] == 1]
    assert len(first) == 3 and (first[:, 0] == 5).all() and (first[:, 1] == 100).all()   # 1 段目は開いたニューロンから
    assert (np.sign(first[:, 4]) == first[:, 6]).all()                                  # 1 段目は辺の符号 = 経路の符号
    assert (e[:, 5] == -1).any() and (e[e[:, 5] == -1][:, 2:4] == [5, 100]).all()       # 上流の 1 段目は開いたニューロンへ
    sm = c.get("/api/web/summary", params={"layer": 5, "n": 100, "zmin": 0}).json()
    assert {"toward", "away", "fed_by", "held_by", "n_neurons"} <= set(sm)
    assert c.get("/api/layer/3/layout", params={"kind": "writes"}).status_code == 404
    _run_job(c, "layout", {"kind": "writes", "layers": [3]})
    cell = np.frombuffer(c.get("/api/layer/3/layout", params={"kind": "writes"}).content, np.int32)
    assert sorted(cell) == list(range(s.meta.n_ff_at(3)))                               # 並べ替え(全マスを 1 回ずつ)
    assert c.get("/api/model").json()["layouts"] == {"writes": [3]}


@pytest.mark.model
def test_u3_links_and_layout_in_browser(server, browser):
    import urllib.request

    def post(path, body):
        req = urllib.request.Request(server + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())

    def wait(jid):
        for _ in range(3000):
            v = next(x for x in json.loads(urllib.request.urlopen(server + "/api/jobs").read()) if x["id"] == jid)
            if v["status"] != "running" and v["status"] != "queued":
                assert v["status"] == "done", v
                return
            time.sleep(0.2)
    wait(post("/api/jobs", {"kind": "connections", "params": {"k": 6, "window": 1}})["id"])
    wait(post("/api/jobs", {"kind": "layout", "params": {"kind": "writes", "layers": [6]}})["id"])
    pg = browser.new_page(viewport={"width": 1500, "height": 860})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(server + "/")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    pg.fill("#jump", "L6 N100")
    pg.press("#jump", "Enter")
    pg.wait_for_function("window.gne.links && window.gne.links.n > 0", timeout=30000)
    assert "読み手(線の先)を + に押す" in pg.evaluate("document.querySelector('#legend').textContent")
    n_seg = pg.evaluate("window.gne.map._links.geometry.getAttribute('position').count / 2")
    assert n_seg == pg.evaluate("window.gne.links.n")
    cols = pg.evaluate("Array.from(window.gne.map._links.geometry.getAttribute('color').array)")
    signs = pg.evaluate("window.gne.map.linkEdges.map(e => e.sign)")
    for i, sg in enumerate(signs):                                                     # 線の色 = 符号(黄は R≈G、赤紫は R≫G)
        r, g = cols[6 * i], cols[6 * i + 1]
        assert (g > 0.6 * r) if sg > 0 else (g < 0.5 * r), (sg, r, g)
    pg.click("#neuron-body .tabs button[data-tab=details]")
    pg.wait_for_function("document.querySelectorAll('#neuron-body table.links tr').length > 0", timeout=30000)
    before = pg.evaluate("window.gne.map.neuronPosition(6, 7)")
    pg.select_option("#layout-select", "writes")
    pg.wait_for_function("window.gne.map.cellOf[6]", timeout=30000)
    after = pg.evaluate("window.gne.map.neuronPosition(6, 7)")
    cell7 = pg.evaluate("window.gne.map.cellOf[6][7]")
    assert after != before or cell7 == 7
    assert "writes 配置" in pg.evaluate("document.querySelector('#legend').textContent")
    pg.select_option("#layer-select", "6")
    pg.wait_for_timeout(300)
    for n in (7, 100, 2000):                                                           # 配置を変えても hover は同じニューロンを指す
        sp = pg.evaluate(f"window.gne.map.screenOf(6, {n})")
        pg.mouse.move(sp["x"], sp["y"])
        pg.wait_for_timeout(80)
        h = pg.evaluate("window.gne.map.hover")
        assert h and (h["layer"], h["neuron"]) == (6, n), (n, h)
    pg.click("#reset-view")                                                            # 全体ビューでは 3D の線、正面ビューでは 2D の札
    pg.wait_for_timeout(200)
    assert pg.evaluate("window.gne.map._links.visible") and pg.evaluate("document.querySelectorAll('#overlay span.lk-d').length") > 0
    pg.select_option("#layer-select", "6")
    pg.wait_for_timeout(200)
    assert not pg.evaluate("window.gne.map._links.visible") and pg.evaluate("document.querySelectorAll('#overlay span.lk-l').length") > 0
    pg.click("#web-toggle")                                                            # 線: 直接 → web → なし
    pg.wait_for_function("window.gne.links && window.gne.links.web", timeout=30000)
    pg.click("#web-toggle")
    pg.wait_for_function("window.gne.links === null", timeout=30000)
    assert not errors, errors
    pg.close()


# --- U4: 埋め込みの散布図 -------------------------------------------------------------------------------------

def _embed_model():
    from gne.embed import default_embed_model
    return default_embed_model()


@pytest.mark.model
def test_u4_embed_job_and_view(server, browser):
    import urllib.request
    if _embed_model() is None or not _embed_model().exists():
        pytest.skip("埋め込みモデル(models/*Embedding*.gguf か GNE_EMBED_MODEL)がありません")

    def get(path):
        return json.loads(urllib.request.urlopen(server + path).read())
    runs = {r["run"]: r for r in get("/api/model")["runs"]}
    assert "L6-quick" in runs and not runs["L6-quick"]["embedded"]
    req = urllib.request.Request(server + "/api/jobs", data=json.dumps({"kind": "embed", "params": {"run": "L6-quick"}}).encode(),
                                 headers={"Content-Type": "application/json"})
    jid = json.loads(urllib.request.urlopen(req).read())["id"]
    for _ in range(3000):
        v = next(x for x in get("/api/jobs") if x["id"] == jid)
        if v["status"] not in ("queued", "running"):
            break
        time.sleep(0.2)
    assert v["status"] == "done", v
    e = get("/api/embeddings?run=L6-quick")
    assert e["meta"]["n"] == 17 and e["meta"]["dim"] > 0 and len(e["points"]) == len(e["texts"]) == 17   # ベースライン 2 + 15
    xs = np.array([p[:2] for p in e["points"]])
    assert xs.min() >= 0 and xs.max() <= 1                                           # 0〜1 にそろえてある
    assert sorted({p[2] for p in e["points"]}) == [0, 1, 2]                          # ニューロン・ランダム・ベースライン
    pg = browser.new_page(viewport={"width": 1500, "height": 860})
    errors = []
    pg.on("pageerror", lambda er: errors.append(str(er)))
    pg.goto(server + "/")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    pg.click("#views button[data-view=embed]")
    pg.wait_for_function("window.gne.embedView && window.gne.embedView.data", timeout=30000)
    assert pg.evaluate("getComputedStyle(document.querySelector('#legend')).display") == "none"     # 地図の凡例は隠す
    p100 = next(p for p in e["points"] if p[2] == 0 and p[3] == 100)
    for c in ("prompt", "dose", "neuron", "kind"):
        pg.select_option("#ev-color", c)
    sx, sy = pg.evaluate(f"(() => {{ const v = window.gne.embedView, r = v.cv.getBoundingClientRect(); const [x, y] = v._toScreen({p100[0]}, {p100[1]}); return [r.left + x, r.top + y]; }})()")
    pg.mouse.move(sx, sy)
    pg.wait_for_timeout(150)
    hv = pg.evaluate("window.gne.embedView.hover")
    who = pg.evaluate(f"window.gne.embedView.data.points[{hv}]")
    assert hv >= 0
    if who[2] == 0:                                                                  # ニューロンの点ならクリックで開く
        pg.mouse.click(sx, sy)
        pg.wait_for_function(f"window.gne.focus && window.gne.focus.n === {who[3]}", timeout=30000)
    pg.click("#views button[data-view=map]")
    assert pg.evaluate("getComputedStyle(document.querySelector('#legend')).display") != "none"
    assert not errors, errors
    pg.close()


# --- U5: エージェント連携 -------------------------------------------------------------------------------------

@pytest.mark.model
@pytest.mark.skipif(not MODEL.exists(), reason="モデルがありません")
def test_u5_agent_api_and_local_only(tmp_path):
    from fastapi.testclient import TestClient

    from gne.api import Session
    from gne.ui.server import create_app
    s = Session(MODEL, root=tmp_path)
    c = TestClient(create_app(s, warm=False))
    assert c.post("/api/ui/highlight", json={"units": ["99:1"]}).status_code == 400              # 0.6B は 28 層
    assert c.post("/api/ui/highlight", json={"units": ["5:5000"]}).status_code == 400            # 各層 3072
    r = c.post("/api/ui/highlight", json={"units": ["5:100", "L6 N7", "6:8:+1.5"], "label": "候補"}).json()
    assert r["ok"] and r["n"] == 3
    assert c.post("/api/ui/highlight", json={"units": ["abc"]}).status_code == 400
    assert c.post("/api/ui/note", json={"layer": 5, "neuron": 100, "text": "<b>宇宙人</b> の候補"}).json()["notes"] == 1
    assert c.post("/api/ui/open", json={"layer": 5, "neuron": 100, "tab": "details"}).json()["ok"]
    assert c.post("/api/ui/open", json={"layer": 5, "neuron": 100, "tab": "nope"}).status_code == 400
    st = c.get("/api/ui/state").json()
    assert [(h["layer"], h["neuron"], h["dose"]) for h in st["highlights"]] == [(5, 100, None), (6, 7, None), (6, 8, 1.5)]
    assert st["label"] == "候補" and st["notes"][0]["text"] == "<b>宇宙人</b> の候補" and st["last"]["action"] == "open"
    c.post("/api/ui/note", json={"layer": 5, "neuron": 100, "text": ""})                        # 空で消す
    assert c.get("/api/ui/state").json()["notes"] == []
    c.post("/api/ui/clear")
    assert c.get("/api/ui/state").json()["highlights"] == []
    # ローカル専用: 別の Host(DNS rebinding)と、別オリジンからの書き込みは断る。読むだけなら Origin があっても通す
    assert c.get("/api/model", headers={"Host": "evil.example:8791"}).status_code == 403
    assert c.post("/api/ui/note", json={"layer": 5, "neuron": 1, "text": "x"}, headers={"Origin": "http://evil.example"}).status_code == 403
    assert c.post("/api/ui/note", json={"layer": 5, "neuron": 1, "text": "x"}, headers={"Origin": "http://127.0.0.1:8791"}).status_code == 200
    assert c.get("/api/ui/state", headers={"Origin": "http://evil.example"}).status_code == 200


@pytest.mark.model
def test_u5_mcp_ui_tools_reach_the_browser(server, browser, tmp_path):
    """MCP の ui_highlight / ui_note / ui_open(gne mcp --ui-url)が、開いているブラウザの地図と NEURON に届く。"""
    import urllib.request

    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    gne = str(Path(os.sys.executable).parent / "gne")
    urllib.request.urlopen(urllib.request.Request(server + "/api/ui/clear", data=b"{}", headers={"Content-Type": "application/json"}))
    pg = browser.new_page(viewport={"width": 1500, "height": 860})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(server + "/")
    pg.wait_for_function("window.gne && window.gne.overlayReady && window.gne.agent && window.gne.agent.connected", timeout=60000)

    async def run():
        params = StdioServerParameters(command=gne, args=["mcp", "--model", str(MODEL), "--root", str(tmp_path), "--ui-url", server,
                                                          "--n-gpu-layers", "0"], env=dict(os.environ))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as s:
                await s.initialize()

                async def call(tool, **args):
                    r = await s.call_tool(tool, args)
                    sc = getattr(r, "structured_content", None)
                    return (sc.get("result", sc) if isinstance(sc, dict) else sc) if sc else json.loads(r.content[0].text)
                assert (await call("ui_highlight", units=["6:100", "6:7"], label="宇宙人の候補"))["n"] == 2
                assert (await call("ui_note", layer=6, neuron=100, text="sweep 1 位・全対照超え <script>"))["notes"] == 1
                assert (await call("ui_open", layer=6, neuron=100, tab="details"))["ok"]
                bad = await call("ui_highlight", units=["999:1"])
                assert bad["error"]["code"] == "BAD_ARGS"
    import threading

    import anyio
    err = []

    def worker():                                                                    # Playwright が同じスレッドでループを回しているので別スレッドで
        try:
            anyio.run(run)
        except BaseException as e:  # noqa: BLE001
            err.append(e)
    th = threading.Thread(target=worker)
    th.start()
    th.join(300)
    assert not err, err
    pg.wait_for_function("document.querySelector('#neuron-body .n-id') && document.querySelector('#neuron-body .n-id').textContent === 'L6 N100'",
                         timeout=30000)
    pg.wait_for_timeout(300)
    assert pg.evaluate("window.gne.map.front") == 6                                              # ui_open で正面ビューに
    assert pg.evaluate("document.querySelectorAll('#overlay span.ag-ring').length") == 2
    note = pg.evaluate("document.querySelector('#overlay span.ag-lab.has-note').textContent")
    assert "sweep 1 位" in note and "<script>" in note                                         # 文字として出る(タグにならない)
    assert pg.evaluate("document.querySelectorAll('#overlay script').length") == 0
    assert "エージェントの注釈" in pg.evaluate("document.querySelector('#neuron-body').innerText")
    assert pg.evaluate("document.querySelector('#agent-btn').textContent") == "agent ◌2 ✎1"
    pg.click("#agent-btn")
    assert "宇宙人の候補" in pg.evaluate("document.querySelector('#agent-pop').innerText")
    pg.click("#ag-clear")
    pg.wait_for_function("document.querySelectorAll('#overlay span.ag-ring').length === 0", timeout=10000)
    pg.wait_for_function("!document.querySelector('#neuron-body').innerText.includes('エージェントの注釈')", timeout=10000)   # パネルの注釈も消える
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_i18n_english_and_switch_back(server, browser):
    """?lang=en で英語に。画面の日本語の文はすべて英語の辞書にある(window.gneI18n.missing が空)。JP で戻る。"""
    import re
    pg = browser.new_page(viewport={"width": 1500, "height": 860})                  # locale は ja-JP のまま(?lang が勝つ)
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    # 保存済みの eval は空に見せる。前のテスト(test_eval_view_and_job)が同じ作業フォルダに残した eval があると、eval ビューに
    # モデルが書いた日本語の返事(出力の例)が出て、下の「画面に日本語が残らない」の確かめに数えられる(回す順番で落ちる)
    pg.route("**/api/evals", lambda route, req: route.fulfill(status=200, content_type="application/json", body="[]"))
    pg.goto(server + "/?lang=en")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    assert pg.evaluate("document.documentElement.lang") == "en"
    assert pg.evaluate("document.querySelector('#lang button.on').dataset.lang") == "en"
    assert pg.evaluate("document.querySelector('#web-toggle').textContent") == "links: direct"
    assert pg.evaluate("document.querySelector('#reset-view').textContent") == "overview"
    assert pg.evaluate("document.querySelector('#search-q').placeholder") == "Search: alien…"
    pg.click("#rank tbody tr[data-n='100']")
    pg.wait_for_function("document.querySelector('#neuron-body .n-id')", timeout=30000)
    texts = []

    def grab():
        texts.append(pg.evaluate("document.body.innerText"))
    for tab in ("details", "chat", "examples"):
        pg.click(f"#neuron-body button[data-tab={tab}]")
        pg.wait_for_timeout(300)
        grab()
    for tab in ("fires", "rank"):
        _open_list(pg, tab)
        pg.wait_for_timeout(200)
        grab()
    for tab in ("units", "try", "out"):                                              # レシピのパネルのタブ
        _recipe_tab(pg, tab)
        pg.wait_for_timeout(200)
        grab()
    pg.click("#reset-view")
    pg.wait_for_timeout(300)
    pg.click("#overlay [data-layer]", button="right")
    pg.wait_for_function("document.querySelector('#compute-menu')", timeout=10000)
    grab()
    pg.keyboard.press("Escape")
    pg.click("#agent-btn")
    grab()
    pg.click("#agent-btn")
    for view in ("layers", "eval", "embed"):
        pg.click(f"#views button[data-view={view}]")
        pg.wait_for_timeout(1500)
        grab()
    assert pg.evaluate("[...window.gneI18n.missing]") == []
    ja = set()
    for s in texts:
        ja |= set(re.findall(r"[ぁ-んァ-ヶ一-龥々ー]+", s))
    assert ja <= {"宇宙人"}, ja                                                       # 残るのは付けた名前だけ
    pg.click("#lang button[data-lang=ja]")                                          # JP に戻す(読み直し)
    pg.wait_for_function("window.gne && window.gne.overlayReady && document.documentElement.lang === 'ja'", timeout=60000)
    assert pg.evaluate("document.querySelector('#web-toggle').textContent") == "線: 直接"
    assert pg.evaluate("localStorage.getItem('gne.lang')") == "ja"
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_eval_view_and_job(server, browser, tmp_path):
    """eval ビュー(CHANGE-01 A): 保存済みの eval を折れ線と表で見る。「今のレシピで eval」でジョブを投げ、終われば一覧に出る。"""
    import urllib.request

    import yaml

    def get(path):
        return json.loads(urllib.request.urlopen(server + path).read())
    (tmp_path / "ppl").mkdir()
    (tmp_path / "ppl" / "en.txt").write_text("Alice was beginning to get very tired of sitting by her sister on the bank. " * 3)
    suite = tmp_path / "mini.yaml"
    suite.write_text(yaml.safe_dump({"name": "mini", "max_tokens": {"exact": 8, "format": 24, "mix": 24, "persona": 24},
                                     "exact": [{"id": "add", "prompt": "What is 17 + 25? Answer with just the number.", "answer": ["42"]}],
                                     "format": [], "mix": [{"id": "m", "prompt": "春について 1 文で。"}],
                                     "persona": {"prompt_sets": ["en"]}, "ppl": [{"id": "en", "path": "ppl/en.txt"}]}))
    req = urllib.request.Request(server + "/api/jobs", data=json.dumps({"kind": "eval", "params": {
        "units": ["14:100:+1.5"], "doses": [0.5, 1], "suite": str(suite), "name": "ui-seed"}}).encode(),
        headers={"Content-Type": "application/json"})
    jid = json.loads(urllib.request.urlopen(req).read())["id"]
    for _ in range(1500):
        v = next(x for x in get("/api/jobs") if x["id"] == jid)
        if v["status"] not in ("queued", "running"):
            break
        time.sleep(0.2)
    assert v["status"] == "done", v
    assert get("/api/evals")[0]["name"] == "ui-seed" and len(get("/api/evals/ui-seed")["rows"]) == 3
    pg = browser.new_page(viewport={"width": 1500, "height": 860})
    errors = []
    pg.on("pageerror", lambda er: errors.append(str(er)))
    pg.goto(server + "/")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    pg.click("#views button[data-view=eval]")
    pg.wait_for_function("window.gne.evalView && window.gne.evalView.data", timeout=30000)
    assert pg.evaluate("window.gne.evalView.data.name") == "ui-seed"
    assert pg.evaluate("document.querySelectorAll('table.evx tbody tr').length") == 3
    assert pg.evaluate("document.querySelector('#evx-canvas').width") > 0
    assert pg.evaluate("document.querySelector('#evx-go').disabled")                      # レシピが空なら押せない
    pg.click("table.evx tbody tr[data-dose='0.5']")
    assert "0.50" in pg.evaluate("document.querySelector('#evx-sample h4').textContent")
    # 今のレシピで投げる(既定のタスク集、倍率 0 と 1)
    pg.evaluate("window.gne.recipe.add(14, 100, 1.0); window.gne.recipe.name = 'ui-run'; window.gne.evalView.refresh()")
    pg.wait_for_function("!document.querySelector('#evx-go').disabled", timeout=10000)
    pg.fill("#evx-doses", "0,1")
    pg.click("#evx-go")
    pg.wait_for_function("window.gne.evalView.data && window.gne.evalView.data.name === 'ui-run'", timeout=300000)
    assert pg.evaluate("document.querySelectorAll('table.evx tbody tr').length") == 2
    assert pg.evaluate("window.gne.evalView.data.suite.name") == "default"
    # NEURON の Details に、このニューロンを含む eval が出て、名前から eval ビューで開ける(CHANGE-01 D)
    unit = get("/api/unit?layer=14&n=100")
    assert [e["name"] for e in unit["evals"]][:2] == ["ui-run", "ui-seed"]
    pg.click("#views button[data-view=map]")
    pg.fill("#jump", "L14 N100")
    pg.press("#jump", "Enter")
    pg.wait_for_function("window.gne.focus && window.gne.focus.n === 100", timeout=30000)
    pg.click("#neuron-body button[data-tab=details]")
    pg.wait_for_function("document.querySelector('#neuron-body a[data-eval=ui-seed]')", timeout=10000)
    pg.click("#neuron-body a[data-eval=ui-seed]")
    pg.wait_for_function("window.gne.view === 'eval' && window.gne.evalView.data.name === 'ui-seed'", timeout=30000)
    assert not errors, errors
    pg.close()


def _fake_eval(name, neuron):
    """eval ビューに出せる最小の eval の結果(ページの側で /api/evals/<名前> の応答として返す)。ユニットのニューロンで見分ける。"""
    def row(dose):
        return {"dose": dose, "capability": 0.9, "capability_drop_pt": 0, "ppl_mean": 5, "ppl_worse_pct": 0, "mix_zh": 0, "mix_other": 0,
                "persona": {"novelty": 0.1 * dose, "distinct4": 0.9, "capped": 0, "top_words": []}, "shippable": True, "ok": {},
                "failed": {"exact": [], "format": []}}
    return {"name": name, "when": "all", "seconds": 1, "created": "2026-01-01T00:00:00",
            "suite": {"n_tasks": 3, "thresholds": {"capability_drop_pt": 5, "ppl_worse_pct": 10}},
            "recipe": {"units": [{"layer": 14, "kind": "neuron", "neuron": neuron, "mode": "add", "dose": 1.0}]},
            "rows": [row(0), row(1)], "samples": {"1": {"persona": [], "failed": []}}}


@pytest.mark.model
def test_eval_view_late_load_does_not_overwrite(server, browser):
    """eval ビュー: 開いたときに読み込む先頭の eval の中身が遅れて届いても、その間に選び直した eval や、差し替えた表示を
    上書きしない(遅れて届いた古い読み込みは捨てる)。一覧と中身はページの側で返し、先頭の eval の中身だけ止めておく。"""
    evs = {"late_a": _fake_eval("late_a", 100), "late_b": _fake_eval("late_b", 200)}
    held = []

    def on_evals(route, req):
        name = req.url.split("/api/evals", 1)[1].lstrip("/")
        if not name:
            route.fulfill(status=200, content_type="application/json",
                          body=json.dumps([{"name": n, "kind": "eval", "created": "2026-01-01T00:00:00"} for n in evs]))
        elif name == "late_a" and len(held) < 2:                                     # 先頭の eval の中身は止めておく(ページごとに 1 回)
            held.append(route)
        else:
            route.fulfill(status=200, content_type="application/json", body=json.dumps(evs[name]))

    def open_eval(n_held):
        pg = browser.new_page(viewport={"width": 1500, "height": 860})
        pg.on("pageerror", lambda er: errors.append(str(er)))
        pg.route("**/api/evals**", on_evals)
        pg.goto(server + "/")
        pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
        pg.click("#views button[data-view=eval]")
        for _ in range(400):
            if len(held) == n_held:
                break
            pg.wait_for_timeout(25)
        assert len(held) == n_held
        return pg

    def release(pg, i):
        with pg.expect_response("**/api/evals/late_a"):
            held[i].fulfill(status=200, content_type="application/json", body=json.dumps(evs["late_a"]))
        pg.wait_for_timeout(300)                                                     # 届いた中身をページが読み終えるまで
    errors = []
    # 1. 先頭(late_a)を待つ間に late_b を選ぶ → late_a が後から届いても late_b のまま
    pg = open_eval(1)
    pg.wait_for_function("document.querySelector('#evx-pick').options.length === 2", timeout=10000)
    pg.select_option("#evx-pick", "late_b")
    pg.wait_for_function("window.gne.evalView.data && window.gne.evalView.data.name === 'late_b'", timeout=10000)
    release(pg, 0)
    assert pg.evaluate("window.gne.evalView.data.name") == "late_b"
    assert pg.evaluate("document.querySelector('#evx-pick').value") == "late_b"
    body = pg.evaluate("document.querySelector('#evx-body').textContent")
    assert "L14 N200" in body and "L14 N100" not in body
    pg.close()
    # 2. 先頭を待つ間に表示を差し替える(test_head_panel_in_browser の eval の段と同じ形)→ 後から届いても差し替えたまま
    pg = open_eval(2)
    pg.evaluate("""(() => { const ev = window.gne.evalView; ev.data = Object.assign(%s, { name: "mine" }); ev.dose = 1; ev.render(); })()"""
                % json.dumps(_fake_eval("mine", 300)))
    release(pg, 1)
    assert pg.evaluate("window.gne.evalView.data.name") == "mine"
    assert "L14 N300" in pg.evaluate("document.querySelector('#evx-body').textContent")
    assert not errors, errors
    pg.close()


class _Late:
    """ページの側で JSON の応答を返し、決めた 1 つだけ止めておいて後から返す(遅れて届く読み込みを作る)。
    key(url) で応答を選ぶ名前を取り出す。bodies: 名前 → JSON(無い名前は本物のサーバへ)。hold の名前は最初の 1 回だけ止める。"""

    def __init__(self, key, bodies, hold):
        self.key, self.bodies, self.hold, self.held = key, bodies, hold, None

    def __call__(self, route, req):
        k = self.key(req.url)
        if k == self.hold and self.held is None:
            self.held = route
        elif k in self.bodies:
            route.fulfill(status=200, content_type="application/json", body=json.dumps(self.bodies[k]))
        else:
            route.continue_()

    def wait_held(self, pg):
        for _ in range(400):
            if self.held is not None:
                return
            pg.wait_for_timeout(25)
        raise AssertionError(f"{self.hold} の読み込みが来ません")

    def release(self, pg, pattern):
        """止めていた応答を返し(bodies に無ければ本物のサーバへ流す)、ページが読み終えるまで待つ。"""
        with pg.expect_response(pattern):
            if self.hold in self.bodies:
                self.held.fulfill(status=200, content_type="application/json", body=json.dumps(self.bodies[self.hold]))
            else:
                self.held.continue_()
        pg.wait_for_timeout(300)


def _path_key(base):
    """URL の base より後ろの名前(一覧なら "")。"""
    return lambda url: url.split(base, 1)[1].split("?")[0].lstrip("/")


def _open_page(browser, errors):
    pg = browser.new_page(viewport={"width": 1500, "height": 900})
    pg.on("pageerror", lambda er: errors.append(str(er)))
    return pg


@pytest.mark.model
def test_writers_view_late_load_does_not_overwrite(server, browser):
    """writers ビュー: 開いたときに読み込む先頭の分析の中身が遅れて届いても、その間に選び直した分析を上書きしない
    (eval ビューと同じ。一覧と中身はページの側で返し、保存してある wseed を 2 つの名前で見せる)。"""
    import urllib.request
    ls = json.loads(urllib.request.urlopen(server + "/api/writers").read())
    one = json.loads(urllib.request.urlopen(server + "/api/writers/wseed").read())
    item = next(x for x in ls if x["name"] == "wseed")
    late = _Late(_path_key("/api/writers"), {"": [{**item, "name": "wr_a"}, {**item, "name": "wr_b"}],
                                              "wr_a": {**one, "name": "wr_a"}, "wr_b": {**one, "name": "wr_b"}}, hold="wr_a")
    errors = []
    pg = _open_page(browser, errors)
    pg.route("**/api/writers**", late)
    pg.goto(server + "/")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    pg.click("#views button[data-view=writers]")
    late.wait_held(pg)                                                               # 先頭の wr_a の中身を止めている
    pg.wait_for_function("document.querySelector('#wrx-pick').options.length === 2", timeout=10000)
    pg.select_option("#wrx-pick", "wr_b")
    pg.wait_for_function("window.gne.writersView.data && window.gne.writersView.data.name === 'wr_b'", timeout=10000)
    late.release(pg, "**/api/writers/wr_a")
    assert pg.evaluate("window.gne.writersView.data.name") == "wr_b"
    assert pg.evaluate("document.querySelector('#wrx-pick').value") == "wr_b"
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_embed_view_late_load_does_not_overwrite(server, browser):
    """embed ビュー: 開いたときに読み込む先頭の run の埋め込みが遅れて届いても、その間に選び直した run を上書きしない。
    run の一覧(/api/model の runs)と埋め込みはページの側で返す。"""
    def emb(model):
        return {"meta": {"n": 2, "model": model, "dim": 8, "layer": 14, "prompts": ["p0"]},
                "points": [[0.2, 0.3, 0, 100, "p0", 1.0, 0], [0.6, 0.7, 2, 0, "p0", 0, 1]], "texts": ["a", "b"]}
    late = _Late(lambda url: url.split("run=", 1)[1].split("&")[0] if "run=" in url else "",
                 {"ea": emb("fake-a"), "eb": emb("fake-b")}, hold="ea")

    def on_model(route, req):
        r = route.fetch()
        m = r.json()
        m["runs"] = [{"run": "ea", "embedded": True, "size": 1000}, {"run": "eb", "embedded": True, "size": 1000}]
        route.fulfill(status=200, content_type="application/json", body=json.dumps(m))
    errors = []
    pg = _open_page(browser, errors)
    pg.route("**/api/model", on_model)
    pg.route("**/api/embeddings*", late)
    pg.goto(server + "/")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    pg.click("#views button[data-view=embed]")
    late.wait_held(pg)                                                               # 先頭の ea の埋め込みを止めている
    pg.select_option("#ev-run", "eb")
    pg.wait_for_function("window.gne.embedView.data && window.gne.embedView.data.meta.model === 'fake-b'", timeout=10000)
    late.release(pg, "**/api/embeddings?run=ea")
    assert pg.evaluate("window.gne.embedView.run") == "eb"
    assert pg.evaluate("window.gne.embedView.data.meta.model") == "fake-b"
    assert "fake-b" in pg.evaluate("document.querySelector('#ev-status').textContent")
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_fieldtest_late_open_does_not_overwrite(server, browser):
    """eval ビューの「実地テスト」: 結果を開く読み込みが遅れて届いても、その間に開き直した結果を上書きしない
    (ジョブの終わりに開く結果と行のクリックが重なったときなど)。一覧と中身はページの側で返す。"""
    late = _Late(_path_key("/api/fieldtests"), {"": [_fieldtest_list_row("ft_a"), _fieldtest_list_row("ft_b")],
                                                 "ft_a": _fieldtest_row("ft_a"), "ft_b": _fieldtest_row("ft_b")}, hold="ft_a")
    errors = []
    pg = _open_page(browser, errors)
    pg.route("**/api/fieldtests**", late)
    pg.goto(server + "/")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    pg.click("#views button[data-view=eval]")
    pg.wait_for_function("window.gne.evalView && window.gne.evalView.ft", timeout=30000)
    pg.evaluate("document.querySelector('#evx-ft-wrap').open = true")
    pg.wait_for_selector("#ft-list tr[data-ft=ft_b]", timeout=10000)
    pg.click("#ft-list tr[data-ft=ft_a]")
    late.wait_held(pg)
    pg.click("#ft-list tr[data-ft=ft_b]")
    pg.wait_for_function("window.gne.evalView.ft.data && window.gne.evalView.ft.data.name === 'ft_b'", timeout=10000)
    late.release(pg, "**/api/fieldtests/ft_a")
    assert pg.evaluate("window.gne.evalView.ft.data.name") == "ft_b"
    assert pg.evaluate("document.querySelector('#ft-list tr.on').dataset.ft") == "ft_b"
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_recipe_name_kept_when_saved_list_arrives(server, browser):
    """recipe タブを開いた直後(保存済みのレシピの一覧が届く前)に入れた名前が、一覧が届いた後も消えず、そのまま保存できる。
    一覧の応答をページの側で止めておき、名前を入れてから返す(欄から出ていないので change はまだ起きていない)。"""
    import urllib.request
    pg = browser.new_page(viewport={"width": 1500, "height": 860})
    errors = []
    pg.on("pageerror", lambda er: errors.append(str(er)))
    pg.goto(server + "/")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    pg.evaluate("window.gne.recipe.clear(); window.gne.recipe.add(14, 100, 1.0)")
    held = []

    def on_list(route, req):
        if req.method == "GET" and not held:                                         # 最初の一覧だけ止める
            held.append(route)
        else:
            route.continue_()
    pg.route("**/api/recipes", on_list)
    _recipe_tab(pg)
    for _ in range(400):
        if held:
            break
        pg.wait_for_timeout(25)
    assert held
    pg.fill("#rc-name", "ui_early")
    held[0].fulfill(status=200, content_type="application/json",
                    body=json.dumps([{"name": "ui_marker", "this_model": True, "units": [{"layer": 14, "neuron": 100, "dose": 1.0}]}]))
    pg.wait_for_function("[...document.querySelectorAll('#rc-load option')].some((o) => o.value === 'ui_marker')", timeout=10000)
    assert pg.evaluate("document.querySelector('#rc-name').value") == "ui_early"
    assert pg.evaluate("window.gne.recipe.name") == "ui_early"                     # 欄から出る前でもレシピに入っている
    pg.click("#rc-save")
    pg.wait_for_function("document.querySelector('#rc-result').textContent.includes('ui_early')", timeout=10000)
    assert any(x["name"] == "ui_early" for x in json.loads(urllib.request.urlopen(server + "/api/recipes").read()))
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_recipe_ablate_row(server, browser, tmp_path):
    """recipe タブの ablate の行(CHANGE-02 C): 方向を選ぶと 3 列チャットの列に ablate が入り(− の列は倍率を反転)、保存したレシピにも残る。"""
    import urllib.request

    import yaml

    def get(path):
        return json.loads(urllib.request.urlopen(server + path).read())

    def post(path, body):
        req = urllib.request.Request(server + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())
    c = tmp_path / "pirate.yaml"
    c.write_text(yaml.safe_dump({"name": "pirate_ui", "system_a": "You are a pirate.", "system_b": "You are a helpful assistant.",
                                 "user_prompts": ["Who are you?", "Hello!"]}))
    jid = post("/api/jobs", {"kind": "direction", "params": {"contrast": str(c), "select": "12"}})["id"]
    for _ in range(600):
        v = next(x for x in get("/api/jobs") if x["id"] == jid)
        if v["status"] not in ("queued", "running"):
            break
        time.sleep(0.2)
    assert v["status"] == "done", v
    pg = browser.new_page(viewport={"width": 1500, "height": 860})
    errors = []
    pg.on("pageerror", lambda er: errors.append(str(er)))
    pg.goto(server + "/")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    assert "pirate_ui" in pg.evaluate("window.gne.model.directions")
    _recipe_tab(pg)
    pg.select_option("#rc-ab-dir", "pirate_ui")
    pg.wait_for_function("document.querySelector('#rc-ab-scale')", timeout=5000)
    pg.evaluate("(() => { const s = document.querySelector('#rc-ab-scale'); s.value = '0.5'; s.dispatchEvent(new Event('input')); })()")
    assert pg.evaluate("window.gne.recipe.ablate") == {"direction": "pirate_ui", "scale": 0.5, "layers": "all", "from_layer": None}
    _recipe_tab(pg, "try")
    pg.click("#rc-chat")                                                            # ablate だけのレシピでもチャットを開ける
    pg.wait_for_function("window.gne.chat.target === 'recipe'", timeout=10000)
    cols = pg.evaluate("window.gne.chat.columns()")
    assert cols[0]["ablate"]["scale"] == -0.5 and "ablate" not in cols[1] and cols[2]["ablate"]["scale"] == 0.5
    _recipe_tab(pg)
    pg.fill("#rc-name", "ui_abl")
    pg.click("#rc-save")
    pg.wait_for_function("document.querySelector('#rc-result').textContent.includes('ui_abl')", timeout=10000)
    saved = next(x for x in get("/api/recipes") if x["name"] == "ui_abl")
    assert saved["ablate"]["direction"] == "pirate_ui" and saved["ablate"]["scale"] == 0.5
    pg.select_option("#rc-ab-dir", "")
    assert pg.evaluate("window.gne.recipe.ablate") is None
    # 方向のユニットのレシピも読み込める(行に出て、3 列チャットの列に入る)。サンプリングの切り替え
    post("/api/recipes/ui_dir", {"units": ["12:dir:pirate_ui:+0.27", "14:100:+0.5"]})
    pg.evaluate("window.gne.recipe.clear()")
    _open_list(pg, "rank")
    _recipe_tab(pg)
    pg.wait_for_function("[...document.querySelectorAll('#rc-load option')].some((o) => o.value === 'ui_dir')", timeout=10000)
    pg.select_option("#rc-load", "ui_dir")
    assert pg.evaluate("window.gne.recipe.specs()") == ["12:dir:pirate_ui:+0.27", "14:100:+0.5"]
    assert "dir:pirate_ui" in pg.evaluate("document.querySelector('#recipe-list table.rc').innerText")
    assert pg.evaluate("document.querySelector('#recipe-list tr[data-i=\"0\"] input.dose').step") == "0.005"
    assert pg.evaluate("window.gne.recipe.neurons().length") == 1
    _recipe_tab(pg, "try")
    pg.click("#rc-chat")
    pg.wait_for_function("document.querySelector('#ch-sp')", timeout=10000)
    cols = pg.evaluate("window.gne.chat.columns()")
    assert cols[2]["units"] == ["12:dir:pirate_ui:+0.27", "14:100:+0.5"] and cols[0]["units"][0] == "12:dir:pirate_ui:-0.27"
    # サンプリング: 既定は greedy(送る本文に sampling が無い)。型を選ぶと値の欄が出る。値を変えると「自由」に
    assert pg.evaluate("window.gne.chat.samplingBody().body") == {}
    pg.select_option("#ch-sp", "llama_server")
    assert pg.evaluate("document.querySelector('.sp-v[data-k=top_k]').value") == "40"
    pg.fill(".sp-v[data-k=temperature]", "0.7")
    pg.dispatch_event(".sp-v[data-k=temperature]", "change")
    assert pg.evaluate("window.gne.chat.spPreset") == "custom"
    pg.check("#ch-seed-fixed")
    pg.fill("#ch-seed", "42")
    pg.dispatch_event("#ch-seed", "change")
    body = pg.evaluate("window.gne.chat.samplingBody().body")
    assert body == {"sampling": {"temperature": 0.7, "top_k": 40, "top_p": 0.95, "min_p": 0.05, "presence_penalty": 0, "seed": 42}}
    pg.click("#ch-sp-to-recipe")                                                    # この値をレシピに入れて保存する
    assert pg.evaluate("window.gne.recipe.sampling.temperature") == 0.7
    assert pg.evaluate("window.gne.chat.spPreset") == "recipe"
    _recipe_tab(pg)
    pg.wait_for_function("document.querySelector('.rc-sampling')", timeout=10000)
    pg.fill("#rc-name", "ui_sp")                                                    # 保存一覧が後から届いても名前の欄は消えない
    pg.dispatch_event("#rc-name", "change")
    pg.click("#rc-save")
    pg.wait_for_function("document.querySelector('#rc-result').textContent.includes('ui_sp')", timeout=10000)
    saved = next(x for x in get("/api/recipes") if x["name"] == "ui_sp")
    assert saved["sampling"] == {"temperature": 0.7, "top_k": 40, "top_p": 0.95, "min_p": 0.05, "presence_penalty": 0.0}
    assert not errors, errors
    pg.close()


# --- CHANGE-04 G: attention ヘッド(HEAD パネル・地図の heads:<name>・fires のヘッドの注目・レシピのヘッド) ---------------

@pytest.mark.model
@pytest.mark.skipif(not MODEL.exists(), reason="モデルがありません")
def test_heads_api(tmp_path):
    from fastapi.testclient import TestClient

    from gne.api import Session
    from gne.ui.server import create_app
    s = Session(MODEL, root=tmp_path)
    rows = _seed_heads(s)
    c = TestClient(create_app(s, warm=False))
    assert [h["name"] for h in c.get("/api/model").json()["heads"]] == ["seed"]
    assert c.get("/api/heads").json()[0]["n_heads"] == len(rows)
    one = c.get("/api/heads/seed").json()
    assert len(one["rows"]) == len(rows) and "top" not in one and (one["rows"][0]["layer"], one["rows"][0]["head"]) == (12, 3)
    assert one["system_a"].startswith("You are a yandere") and one["triggers"]["I'm going to the movies with Sarah tonight."]
    assert c.get("/api/heads/nope").status_code == 400 and c.get("/api/heads/..%2Fx").status_code in (400, 404)
    # 調書: lens=false は logit lens を待たずに返す。1 回計算したら覚えておく
    d = c.get("/api/head", params={"layer": 12, "head": 3, "lens": False}).json()
    assert d["logit_lens"] is None and d["kind"] == "attn" and d["block"] == [3 * d["head_dim"], 4 * d["head_dim"]]
    assert d["contrasts"][0]["name"] == "seed" and d["contrasts"][0]["rank"] == 1 and d["contrasts"][0]["floors"]["out_shift"] == 0.3
    ll = c.get("/api/head", params={"layer": 12, "head": 3}).json()["logit_lens"]
    assert len(ll["components"]) == 2 and len(ll["components"][0]["up_tokens"]) == 12 and 0 < ll["sv_share"] <= 1
    assert c.get("/api/head", params={"layer": 12, "head": 3, "lens": False}).json()["logit_lens"] == ll
    assert c.get("/api/head", params={"layer": 12, "head": 99}).status_code == 400
    assert c.get("/api/head", params={"layer": 99, "head": 0}).status_code == 400
    # 名前はニューロンと同じファイル("L:Hh")。ニューロンの overlay・検索は壊れない
    r = c.post("/api/head/name", json={"layer": 12, "head": 3, "name": " 嫉妬の係 "}).json()
    assert r["name"] == "嫉妬の係" and r["names"]["12:H3"] == "嫉妬の係"
    c.post("/api/name", json={"layer": 12, "neuron": 5, "name": "n5"})
    v = np.frombuffer(c.get("/api/layer/12/field", params={"name": "named"}).content, np.float32)
    assert v[5] == 1.0 and np.isnan(v[3])
    assert [h["neuron"] for h in c.get("/api/search", params={"q": "嫉妬", "mode": "name"}).json()["hits"]] == []
    assert c.get("/api/head", params={"layer": 12, "head": 3, "lens": False}).json()["name"] == "嫉妬の係"
    # 注目のパターン: 行の和 ≈ 1、因果マスク
    a = c.post("/api/attention", json={"text": "The aliens landed on Mars.", "layer": 12, "head": 3}).json()
    P = np.array(a["pattern"])
    assert P.shape == (a["n_tokens"], a["n_tokens"]) == (len(a["tokens"]), len(a["tokens"]))
    assert np.allclose(P.sum(axis=1), 1, atol=1e-3) and np.allclose(np.triu(P, 1), 0)
    cmp = c.post("/api/attention", json={"text": "Who are you?", "layer": 12, "head": 3, "chat": True, "system": "You are a pirate.",
                                          "compare": True}).json()
    wt, wo = cmp["with"]["tokens"], cmp["without"]["tokens"]
    assert cmp["compare"] and len(wt) > len(wo) and wt[-len(wo):] == wo and "pirate" in "".join(wt[:len(wt) - len(wo)])
    assert c.post("/api/attention", json={"text": "Hi", "layer": 12, "head": 3, "compare": True}).status_code == 400   # system が要る
    assert c.post("/api/attention", json={"text": "Hello there", "layer": 12, "head": 16}).status_code == 400
    # fires のヘッドの注目: (全ヘッド, T)。各トークンへの、後ろの位置からの注目の平均(パターンから直接計算したものと一致)
    f = c.post("/api/fires", json={"text": "The aliens landed on Mars."}).json()
    r = c.get(f"/api/fires/{f['id']}/heads")
    n, T = (int(x) for x in r.headers["x-shape"].split(","))
    meta = json.loads(r.headers["x-meta"])
    m = np.frombuffer(r.content, np.float32).reshape(n, T)
    assert n == s.meta.n_layer * 16 and T == f["n_tokens"] == a["n_tokens"] and np.isnan(m[:, -1]).all()
    i = meta["heads"].index([12, 3])
    want = [(P[p + 1:, p]).mean() for p in range(T - 1)]
    assert np.allclose(m[i, :-1], want, atol=1e-4)
    assert c.get("/api/fires/zzz/heads").status_code == 400
    # レシピのヘッドの行: 保存して読むと head が付いて返る。ヘッドの gain で 3 列チャットが回る
    assert c.post("/api/recipes/hr", json={"units": ["12:H3:x0", "14:100:+1"]}).status_code == 200
    hr = next(x for x in c.get("/api/recipes").json() if x["name"] == "hr")
    hu = next(u for u in hr["units"] if u["kind"] == "head")
    assert (hu["layer"], hu["head"], hu["mode"], hu["k"]) == (12, 3, "gain", 0.0)
    msgs = [{"role": "user", "content": "Who are you?"}]
    ev = _sse_events(c, "/api/steer", {"max_tokens": 4, "columns": [{"units": ["12:H3:x0"], "messages": msgs},
                                                                      {"units": [], "messages": msgs}]})
    ends = [e for e in ev if e["type"] == "end"]
    assert not [e for e in ev if e["type"] == "error"] and [e["n_gen"] for e in ends] == [4, 4]


@pytest.mark.model
def test_head_panel_in_browser(server, browser):
    """板の上辺の刻みにマウスを当てるとヘッド、クリックで HEAD パネル(Pattern / Contrast / Details)。overlay heads:<name>、
    レシピのヘッドの行と印、fires のヘッドの注目、eval の反応性の列、英語の文言。"""
    import urllib.request

    def post(path, body):
        req = urllib.request.Request(server + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())
    pg = browser.new_page(viewport={"width": 1500, "height": 860})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(server + "/")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    pg.evaluate("window.gne.recipe.clear()")
    assert "heads:seed" in pg.evaluate("[...document.querySelectorAll('#color-select option')].map(o => o.value)")
    pg.select_option("#layer-select", "12")
    pg.wait_for_timeout(300)
    for h in (0, 3, 15):                                                             # 刻みの帯の上で、そのヘッドを拾う
        sp = pg.evaluate(f"window.gne.map.screenOfHead(12, {h})")
        pg.mouse.move(sp["x"], sp["y"])
        pg.wait_for_timeout(80)
        hv = pg.evaluate("window.gne.map.hover")
        assert hv and (hv["layer"], hv.get("head")) == (12, h), (h, hv)
    assert pg.evaluate("document.querySelector('#tooltip').innerText").startswith("L12 H15")
    sp = pg.evaluate("window.gne.map.screenOf(12, 0)")                               # シートの左上の角のニューロンは今までどおり
    pg.mouse.move(sp["x"], sp["y"])
    pg.wait_for_timeout(80)
    assert pg.evaluate("window.gne.map.hover.neuron") == 0
    sp = pg.evaluate("window.gne.map.screenOfHead(12, 3)")
    pg.mouse.click(sp["x"], sp["y"])
    pg.wait_for_function("document.querySelector('#head-body .n-id')", timeout=30000)
    assert pg.evaluate("document.querySelector('#head-body .n-id').textContent").startswith("L12 H3")
    assert not pg.evaluate("document.querySelector('#detail').classList.contains('collapsed')")         # くわしく見るがヘッドで開く
    assert not pg.evaluate("document.querySelector('#head-body').classList.contains('hidden')")
    assert pg.evaluate("location.hash") == "#L12/H3" and pg.evaluate("document.querySelectorAll('#overlay span.mk-h.sel').length") == 1
    # Pattern: ペルソナあり/なし、トリガーの表、差の表示
    pg.fill("#hp-text", "I had lunch with Emily from work today.")
    pg.dispatch_event("#hp-text", "input")
    pg.fill("#hp-system", "You are a jealous yandere girl.")
    pg.dispatch_event("#hp-system", "input")
    pg.fill("#hp-trig", "Emily")
    pg.dispatch_event("#hp-trig", "change")
    pg.click("#hp-run")
    pg.wait_for_function("document.querySelector('#hp-canvas') && document.querySelector('#hp-canvas').width > 100", timeout=60000)
    res = pg.evaluate("(() => { const r = window.gne.headPanel.res; return { cmp: r.compare, S: r.S, nb: r.b.tokens.length }; })()")
    assert res["cmp"] and res["S"] == res["nb"]                                     # system なしの会話はまるごと共通の末尾
    assert pg.evaluate("document.querySelectorAll('#head-body table.hp-trig tr').length") == 2   # 見出し + Emily
    pg.click("#hp-out button[data-v=diff]")
    assert pg.evaluate("window.gne.headPanel.heat.sysCol && window.gne.headPanel.heat.colLab.length") == res["S"] + 1
    # Contrast: 指標と基準値、上位の表、地図で色付け
    pg.click("#head-body .tabs button[data-tab=contrast]")
    pg.wait_for_function("document.querySelector('#head-body table.h-top')", timeout=30000)
    assert pg.evaluate("document.querySelectorAll('#head-body table.h-met td.hot').length") == 3   # 基準値を超えた 3 指標(dir_cos は無い)
    pg.click("#head-body button[data-ov=seed]")
    pg.wait_for_function("window.gne.heads && window.gne.heads.field === 'heads:seed'", timeout=30000)
    assert pg.evaluate("window.gne.map.layerParts[12].heads.visible") and not pg.evaluate("window.gne.map.layerParts[11].heads.visible")
    col = pg.evaluate("Array.from(window.gne.map.layerParts[12].heads.geometry.getAttribute('color').array.slice(3 * 18, 3 * 18 + 3))")
    assert col[0] > col[1] > col[2] > 0                                              # H3 は琥珀(基準値超え)
    assert "ヘッドの対照 seed の z" in pg.evaluate("document.querySelector('#legend').textContent")
    # Details: W_O ブロックの logit lens(後から読み込む)
    pg.click("#head-body .tabs button[data-tab=details]")
    pg.wait_for_function("document.querySelectorAll('#head-body .lens').length === 4", timeout=60000)
    # レシピ: ×0 で足す → 印と行。倍率を変えると L:Hh:xk、− 列は 1 の反対側
    pg.click("#head-body [data-hk='0']")
    assert pg.evaluate("window.gne.recipe.specs()") == ["12:H3:x0"]
    pg.wait_for_timeout(200)
    assert pg.evaluate("document.querySelectorAll('#overlay span.mk-h.minus').length") == 1
    _recipe_tab(pg)
    assert "L12 H3" in pg.evaluate("document.querySelector('#recipe-list table.rc').innerText")
    pg.fill("#recipe-list tr.rc-head input.num", "0.5")
    pg.dispatch_event("#recipe-list tr.rc-head input.num", "input")
    assert pg.evaluate("window.gne.recipe.specs()") == ["12:H3:x0.5"] and pg.evaluate("window.gne.recipe.specs(-1)") == ["12:H3:x1.5"]
    assert pg.evaluate("document.querySelector('#rc-export').disabled")             # ヘッドだけでは cvec にならない
    post("/api/recipes/ui_heads", {"units": ["12:H3:x2", "14:100:+1"]})
    _open_list(pg, "rank")
    _recipe_tab(pg)
    pg.wait_for_function("[...document.querySelectorAll('#rc-load option')].some((o) => o.value === 'ui_heads')", timeout=10000)
    asked = []
    pg.once("dialog", lambda d: (asked.append(d.message), d.accept()))             # いまのレシピ(×0.5)を置き換える前に確かめる
    pg.select_option("#rc-load", "ui_heads")
    assert asked and pg.evaluate("window.gne.recipe.specs()") == ["12:H3:x2", "14:100:+1"] and pg.evaluate("window.gne.recipe.skipped") == 0
    pg.click("#recipe-list tr.rc-head a[data-open]")                                # 行から HEAD パネルへ
    pg.fill("#jump", "L5 H2")
    pg.press("#jump", "Enter")
    pg.wait_for_function("document.querySelector('#head-body .n-id') && document.querySelector('#head-body .n-id').textContent.startsWith('L5 H2')", timeout=30000)
    # fires: 文章を流すとヘッドも、選んだトークンへの注目で塗る
    _open_list(pg, "fires")
    pg.fill("#fi-text", "The aliens landed on Mars.")
    pg.dispatch_event("#fi-text", "input")
    pg.click("#fi-run")
    pg.wait_for_function("window.gne.heads && window.gne.heads.field === 'fires_heads'", timeout=60000)
    assert pg.evaluate("window.gne.heads.values.size") == pg.evaluate("window.gne.arch.n_layer")
    pg.click(".chip-t[data-tok='2']")
    pg.wait_for_function("window.gne.heads && window.gne.heads.label && window.gne.heads.label.includes('landed')", timeout=30000)
    assert "ヘッドの注目" in pg.evaluate("document.querySelector('#legend').textContent")
    # eval ビュー: 反応性の列(保存済みの eval の形をそのまま渡す)
    pg.click("#views button[data-view=eval]")
    pg.wait_for_function("window.gne.evalView", timeout=30000)
    # 開いたときに読み込む保存済みの eval が差し替えの後に届いても、差し替えた表示は上書きされない(eval.js が古い読み込みを捨てる。
    # test_eval_view_late_load_does_not_overwrite)
    pg.evaluate("""(() => {
      const row = (dose, w, wo, ctrl) => ({ dose, capability: 0.9, capability_drop_pt: 0, ppl_mean: 5, ppl_worse_pct: 0, mix_zh: 0, mix_other: 0,
        persona: { novelty: 0.1, distinct4: 0.9, capped: 0, top_words: [] }, shippable: true, ok: {}, failed: { exact: [], format: [] },
        reactivity: { with: w, without: wo, reactivity: w - wo, n_pairs: 10, samples: [{ with: "darling, who is she?", without: "Nice!" }],
                      ...(ctrl ? { ctrl: { n: 4, values: [0, 0.1, 0, 0.1], p05: 0, p50: 0.05, p95: 0.1, max: 0.1 }, null_pct: 1.0 } : {}) } });
      const ev = window.gne.evalView;
      ev.data = { name: "rx", when: "all", seconds: 1, suite: { n_tasks: 3, thresholds: { capability_drop_pt: 5, ppl_worse_pct: 10 } },
        recipe: { units: [{ layer: 12, kind: "head", head: 3, mode: "gain", k: 0 }] }, rows: [row(0, 0.1, 0.1, false), row(1, 0.6, 0.1, true)],
        samples: { "1": { persona: [], failed: [] } }, reactivity: { contrast: "x.yaml", name: "yandere_react", n_pairs: 10, head_controls: 4, max_tokens: 64 } };
      ev.dose = 1; ev.render();
    })()""")
    heads_ = pg.evaluate("[...document.querySelectorAll('table.evx thead th')].map(th => th.textContent)")
    assert ["あり", "なし", "反応性", "対照p95", "null"] == heads_[-6:-1]
    cells = pg.evaluate("[...document.querySelectorAll('table.evx tbody tr[data-dose=\"1\"] td')].map(td => td.textContent)")
    assert cells[-6:-1] == ["60%", "10%", "+50%", "+10%", "1.00"]
    assert "L12 H3 ×0.00" in pg.evaluate("document.querySelector('#evx-body').textContent")
    assert not errors, errors
    pg.close()
    # 英語: HEAD パネルの文言はすべて辞書にある
    pg = browser.new_page(viewport={"width": 1500, "height": 860})
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(server + "/?lang=en#L12/H3")
    pg.wait_for_function("window.gne && window.gne.overlayReady && document.querySelector('#head-body .n-id')", timeout=60000)
    pg.fill("#hp-text", "Hello Emily")
    pg.dispatch_event("#hp-text", "input")
    pg.fill("#hp-trig", "Emily")
    pg.dispatch_event("#hp-trig", "change")
    pg.click("#hp-run")
    pg.wait_for_function("document.querySelector('#hp-canvas')", timeout=60000)
    for tab in ("contrast", "details"):
        pg.click(f"#head-body button[data-tab={tab}]")
        pg.wait_for_timeout(500)
    pg.select_option("#color-select", "heads:seed")
    pg.wait_for_function("window.gne.heads && window.gne.heads.field === 'heads:seed'", timeout=30000)
    sp = pg.evaluate("window.gne.map.screenOfHead(12, 3)")
    pg.mouse.move(sp["x"], sp["y"])
    pg.wait_for_timeout(150)
    assert pg.evaluate("[...window.gneI18n.missing]") == []
    assert not errors, errors
    pg.close()


def test_writers_view_and_overlay(server, browser):
    """writers ビュー(書き手の強さ): 割合の棒・上位の表・行のクリックでニューロン、「地図に重ねる」で writers:<name> の色。英語の文言。"""
    import urllib.request
    m = json.loads(urllib.request.urlopen(server + "/api/model").read())
    assert m["writers"][0]["name"] == "wseed" and m["writers"][0]["layers"] == list(range(21))
    f = urllib.request.urlopen(server + "/api/layer/14/field?name=writers:wseed")
    v = np.frombuffer(f.read(), np.float32)
    assert abs(float(v[77]) - 0.9) < 1e-6
    pg = browser.new_page(viewport={"width": 1500, "height": 860})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(server + "/?lang=en")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    pg.click("#views button[data-view=writers]")
    pg.wait_for_selector("#writers-view table.wrx tbody tr")
    rows = pg.evaluate("[...document.querySelectorAll('#writers-view table.wrx tbody tr')].length")
    assert rows >= 20
    assert "MLP +58%" in pg.inner_text("#writers-view")
    assert pg.evaluate("document.querySelector('#wrx-canvas').width") > 0
    pg.click("#wrx-map")
    pg.wait_for_function("window.gne.overlay.field === 'writers:wseed' && window.gne.overlay.scale", timeout=30000)
    assert pg.evaluate("[...window.gneI18n.missing]") == []
    pg.click("#views button[data-view=writers]")
    pg.wait_for_selector("#writers-view table.wrx tbody tr")
    pg.click("#writers-view table.wrx tbody tr[data-layer='14'][data-n='77']")
    pg.wait_for_function("window.gne.focus && window.gne.focus.layer === 14 && window.gne.focus.n === 77", timeout=30000)
    assert not errors, errors


# --- CHANGE-05 G: MoE(EXPERT パネル、専門家の overlay、fires の選ばれた専門家、MoE のニューロンのレシピ) -----------------------

MOE = Path(os.environ.get("GNE_TEST_MOE", ROOT / "models" / "granite-3.1-1b-a400m-instruct-Q8_0.gguf"))


@pytest.fixture(scope="module")
def moe_server(tmp_path_factory):
    """MoE のモデル(GNE_TEST_MOE。granite-3.1-1b-a400m: 24 層 × 専門家 32・使用 8・幅 512、ルーターのバイアス無し)のサーバ。
    層 5・12 の活性統計(専門家の使用統計)と、専門家の対照分析 pirate(層 5・12)を作っておく。"""
    if not MOE.exists():
        pytest.skip(f"MoE のモデルがありません({MOE})")
    import uvicorn
    import yaml

    from gne.api import Session
    from gne.ui.server import create_app
    root = tmp_path_factory.mktemp("ui_moe")
    s = Session(MOE, root=root, n_ctx=1024)
    s.stats_act(layers=[5, 12], tokens=1500)
    c = root / "pirate.yaml"
    c.write_text(yaml.safe_dump({"name": "pirate", "system_a": "You are a pirate. Always talk like a pirate, with arr and matey.",
                                 "system_b": "You are a helpful assistant.", "user_prompts": ["Who are you?", "Tell me about the weather."],
                                 "validate": {"match": "arr|matey", "target": "a pirate"}}))
    top = s.experts_contrast(c, max_tokens=16, layers=[5, 12])["top"][0]
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(create_app(s), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield {"url": f"http://127.0.0.1:{port}", "L": top["layer"], "E": top["expert"], "W": s.weights.arch.layers[top["layer"]].n_ff}
    srv.should_exit = True
    th.join(5)
    s.close()


def _open_list(pg, tab):
    if tab == "recipe":                                                              # レシピは右のパネル
        _recipe_tab(pg)
        return
    if pg.evaluate("document.querySelector('#list').classList.contains('collapsed')"):
        pg.click(".toggle[data-target=list]")
    pg.click(f"#list-tabs button[data-tab={tab}]")


@pytest.mark.model
def test_moe_expert_panel_in_browser(moe_server, browser):
    """EXPERT パネル: jump の L E で開く(L E N はニューロン)、使われ方・対照分析・ニューロンの上位、×0 / ×2 をレシピに(− 列は 2 − k)、
    3 列チャット(専門家全体の gain はパッチで読み直す)、上位のニューロンと地図のクリックで NEURON パネル(通し番号)。"""
    url, L, E, W = moe_server["url"], moe_server["L"], moe_server["E"], moe_server["W"]
    pg = browser.new_page(viewport={"width": 1500, "height": 860})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(url + "/")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    pg.evaluate("window.gne.recipe.clear()")
    assert pg.evaluate("document.querySelector('#jump').placeholder") == "L10 E0 N0"
    assert not pg.evaluate("document.querySelector('#detail-kinds button[data-kind=expert]').classList.contains('hidden')")
    pg.fill("#jump", f"L{L} E{E}")
    pg.press("#jump", "Enter")
    pg.wait_for_function(f"document.querySelector('#expert-body .n-id') && document.querySelector('#expert-body .n-id').textContent.startsWith('L{L} E{E} ')",
                         timeout=30000)
    assert pg.evaluate("location.hash") == f"#L{L}/E{E}"
    d = pg.evaluate("window.gne.expertPanel.d")
    assert d["usage"] and d["contrasts"][0]["name"] == "pirate" and d["contrasts"][0]["rank"] == 1 and len(d["neurons"]) == 20
    body = pg.inner_text("#expert-body")
    assert "softmax top-k · 8/32" in body and "選択率" in body and "一様 25.0%" in body and f"#1/{d['contrasts'][0]['of']}" in body
    assert pg.evaluate("document.querySelectorAll('#expert-body .x-c').length") == 1
    assert pg.evaluate("document.querySelectorAll('#expert-body table.x-top tbody tr').length") == 20
    assert pg.evaluate("!!document.querySelector('#expert-body .x-few')") == (not d["usage"]["reliable"])   # 観測が少なければ注記
    assert pg.evaluate("document.querySelectorAll('#expert-body [data-xr]').length") == 0     # granite にルーターのバイアスは無い
    assert pg.evaluate("document.querySelector('#x-chat').disabled")                          # レシピが空
    frames = lambda: pg.evaluate("window.gne.map.tileFrames.map((f) => [f.layer, f.expert, f.ring, f.color])")  # noqa: E731
    assert [L, E, 0, 0xffffff] in frames()                                                    # 開いている専門家の白い枠
    # レシピ: ×0 → もう一度押すと外す → ×2。3 列チャットの − 列は 2 − k
    pg.click("#expert-body [data-xk='0']")
    assert pg.evaluate("window.gne.recipe.specs()") == [f"{L}:E{E}:x0"] and pg.evaluate("window.gne.recipe.specs(-1)") == [f"{L}:E{E}:x2"]
    assert [L, E, 1, 0xc94fa7] in frames()
    pg.click("#expert-body [data-xk='0']")
    assert pg.evaluate("window.gne.recipe.specs()") == []
    pg.click("#expert-body [data-xk='2']")
    assert pg.evaluate("window.gne.recipe.specs()") == [f"{L}:E{E}:x2"] and pg.evaluate("window.gne.recipe.specs(-1)") == [f"{L}:E{E}:x0"]
    assert [L, E, 1, 0xf2d24b] in frames()
    # route ±1 はルーターのバイアスのある層だけ(granite には無いので、構造の値だけ替えて押せることを確かめる。サーバには送らない)
    pg.evaluate(f"(() => {{ window.gne.arch.layers[{L}].has_router_bias = true; window.gne.expertPanel.render(); }})()")
    pg.click("#expert-body [data-xr='1']")
    assert pg.evaluate("window.gne.recipe.specs()") == [f"{L}:E{E}:x2", f"{L}:E{E}:route:+1"]
    assert pg.evaluate("window.gne.recipe.specs(-1)") == [f"{L}:E{E}:x0", f"{L}:E{E}:route:-1"]
    pg.click("#expert-body [data-xr='1']")                                            # もう一度押すと外す
    pg.evaluate(f"(() => {{ window.gne.arch.layers[{L}].has_router_bias = false; window.gne.expertPanel.render(); }})()")
    assert pg.evaluate("window.gne.recipe.specs()") == [f"{L}:E{E}:x2"]
    pg.click("#x-chat")
    pg.wait_for_selector("#ch-text", timeout=30000)
    assert pg.evaluate("[...document.querySelectorAll('.ch-head .ch-units')].map((x) => x.innerText)") == [f"L{L} E{E} ×0", f"L{L} E{E} ×2"]
    pg.fill("#ch-max", "6")
    pg.dispatch_event("#ch-max", "change")
    pg.fill("#ch-text", "Who are you?")
    pg.dispatch_event("#ch-text", "input")
    pg.click("#ch-send")
    pg.wait_for_function("window.gne.chat && !window.gne.chat.busy && window.gne.chat.turns.length === 1", timeout=180000)
    replies = pg.evaluate("window.gne.chat.turns[0].replies")
    assert len(replies) == 3 and all(replies), (replies, pg.evaluate("window.gne.chat.status"))
    assert not pg.evaluate("window.gne.chat.failed"), pg.evaluate("window.gne.chat.status")
    # 上位のニューロン → NEURON パネル(通し番号)。jump の L E N もニューロン
    n0 = d["neurons"][0]["neuron"]
    pg.fill("#jump", f"L{L} E{E}")
    pg.press("#jump", "Enter")
    pg.wait_for_selector("#expert-body table.x-top tbody tr", timeout=30000)
    pg.click("#expert-body table.x-top tbody tr")
    pg.wait_for_function(f"window.gne.focus && window.gne.focus.n === {E * W + n0}", timeout=30000)
    pg.wait_for_function(f"document.querySelector('#neuron-body .n-id') && document.querySelector('#neuron-body .n-id').textContent.startsWith('L{L} E{E} N{n0}')",
                         timeout=30000)
    assert pg.evaluate("location.hash") == f"#L{L}/E{E}/N{n0}" and pg.evaluate("document.querySelector('#jump').value") == f"L{L} E{E} N{n0}"
    pg.fill("#jump", f"L{L} E{E} N5")
    pg.press("#jump", "Enter")
    pg.wait_for_function(f"window.gne.focus && window.gne.focus.n === {E * W + 5}", timeout=30000)
    # 地図の専門家のタイルのニューロンをクリック → NEURON パネル
    pg.select_option("#layer-select", str(L))
    pg.wait_for_timeout(300)
    sp = pg.evaluate(f"window.gne.map.screenOf({L}, {E * W + 40})")
    pg.mouse.move(sp["x"], sp["y"])
    pg.wait_for_timeout(100)
    hv = pg.evaluate("window.gne.map.hover")
    assert (hv["layer"], hv["expert"], hv["neuron"]) == (L, E, 40)
    pg.mouse.click(sp["x"], sp["y"])
    pg.wait_for_function(f"window.gne.focus && window.gne.focus.n === {E * W + 40}", timeout=30000)
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_moe_overlays_and_fires_routing(moe_server, browser):
    """overlay expert_usage(専門家の選択率)と experts:<name>(専門家の対照分析の z): 値・凡例・hover。fires の routed: 選んだトークンで
    選ばれた専門家の一覧と地図のタイルの枠、一覧のニューロンは E N の形。"""
    import re
    import urllib.request
    url, L, E, W = moe_server["url"], moe_server["L"], moe_server["E"], moe_server["W"]
    f = urllib.request.urlopen(url + f"/api/layer/{L}/field?name=experts:pirate")
    meta = json.loads(f.headers["X-Meta"])
    v = np.frombuffer(f.read(), np.float32)
    assert v.size == 32 * W and abs(float(v[E * W + 3]) - meta["experts"][str(E)]["z"]) < 1e-4
    pg = browser.new_page(viewport={"width": 1500, "height": 860})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(url + "/")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    opts = pg.evaluate("Object.fromEntries([...document.querySelectorAll('#color-select option')].map((o) => [o.value, o.textContent]))")
    assert opts["expert_usage"].endswith("(2 層)") and opts["experts:pirate"].endswith("(2 層)")
    pg.select_option("#color-select", "expert_usage")
    pg.wait_for_function("window.gne.overlay.field === 'expert_usage' && window.gne.overlay.scale", timeout=30000)
    info = pg.evaluate("(() => { const o = window.gne.overlay; return { layers: [...o.values.keys()].sort((a, b) => a - b), n: o.values.get(5).length, "
                       "lit: o.lit.get(5) + o.lit.get(12), uni: o.scale.uniform }; })()")
    assert info["layers"] == [5, 12] and info["n"] == 32 * W and info["lit"] > 0 and abs(info["uni"] - 0.25) < 1e-6
    lg = pg.inner_text("#legend")
    assert "専門家の選択率" in lg and "一様(25.0%)" in lg
    pg.select_option("#layer-select", str(L))
    pg.wait_for_timeout(300)
    sp = pg.evaluate(f"window.gne.map.screenOf({L}, {E * W + 40})")
    pg.mouse.move(sp["x"], sp["y"])
    pg.wait_for_timeout(150)
    tip = pg.inner_text("#tooltip")
    assert tip.startswith(f"L{L} E{E} N40") and f"E{E} の選択率" in tip
    pg.select_option("#color-select", "experts:pirate")
    pg.wait_for_function("window.gne.overlay.field === 'experts:pirate' && window.gne.overlay.scale", timeout=30000)
    assert pg.evaluate(f"window.gne.overlay.lit.get({L})") >= W                        # 1 位の専門家のタイルは光る
    assert "専門家の対照 pirate の z" in pg.inner_text("#legend")
    pg.mouse.move(sp["x"] + 1, sp["y"])
    pg.wait_for_timeout(150)
    assert f"E{E} · z {meta['experts'][str(E)]['z']:.2f}" in pg.inner_text("#tooltip")
    # fires: 選んだトークンで選ばれた専門家
    _open_list(pg, "fires")
    pg.fill("#fi-text", "The pirate sailed across the sea.")
    pg.dispatch_event("#fi-text", "input")
    pg.click("#fi-run")
    pg.wait_for_selector(".chip-t[data-tok='2']", timeout=60000)
    assert pg.evaluate("!!document.querySelector('p.fi-routed')")                       # 全トークンのときは案内だけ
    pg.click(".chip-t[data-tok='2']")
    pg.wait_for_function("window.gne.firesRouted && window.gne.firesRouted.size > 0", timeout=30000)
    nl = pg.evaluate("window.gne.firesRouted.size")
    assert nl >= 20 and pg.evaluate("[...window.gne.firesRouted.values()].every((es) => es.length === 8)")
    lines = pg.evaluate("[...document.querySelectorAll('.fi-routed-list > div')].map((x) => x.innerText)")
    assert len(lines) == nl and re.fullmatch(r"L0\s+(E\d+ ){7}E\d+", lines[0]), lines[:2]
    assert pg.evaluate("window.gne.map.tileFrames.filter((f) => f.ring === 2).length") == 8 * nl
    assert "fires のトークンで選ばれた専門家" in pg.inner_text("#legend")
    pg.wait_for_function("document.querySelector('#fi-list tr[data-l]')", timeout=30000)
    assert re.fullmatch(r"E\d+ N\d+", pg.evaluate("document.querySelector('#fi-list tr[data-l] td:nth-child(2)').innerText"))
    pg.click(".chip-t[data-tok='-1']")
    pg.wait_for_function("window.gne.firesRouted === null", timeout=30000)
    assert pg.evaluate("window.gne.map.tileFrames.filter((f) => f.ring === 2).length") == 0
    pg.click(".chip-t[data-tok='3']")
    pg.wait_for_function("window.gne.firesRouted && window.gne.firesRouted.size > 0", timeout=30000)
    pg.select_option("#color-select", "none")                                         # ほかの色に替えると枠は消える
    pg.wait_for_function("window.gne.firesRouted === null", timeout=30000)
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_moe_neuron_recipe_buttons(moe_server, browser):
    """MoE のニューロン: 住所 L E N と方向押しの注記、+ / − レシピは L:Ee:Nn:±dose、×0 / ×2 は gain、Web タブは使えない。
    recipe タブの行(方向押しの印・gain・専門家全体・route・shared)、− 列の反転、3 列チャットの列見出しの表示、保存と読み込み。"""
    import urllib.request
    url, L, E, W = moe_server["url"], moe_server["L"], moe_server["E"], moe_server["W"]
    flat = E * W + 7
    pg = browser.new_page(viewport={"width": 1500, "height": 860})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(url + f"/#L{L}/E{E}/N7")
    pg.wait_for_function("window.gne && window.gne.overlayReady && document.querySelector('#neuron-body .n-id')", timeout=60000)
    pg.evaluate("window.gne.recipe.clear()")
    nid = pg.inner_text("#neuron-body .n-id")
    assert nid.startswith(f"L{L} E{E} N7") and f"#{flat}" in nid and pg.evaluate("window.gne.focus.n") == flat
    assert "方向押し" in pg.inner_text("#neuron-body .n-moe-note")
    web = pg.evaluate("(() => { const b = document.querySelector('#neuron-body .tabs button[data-tab=web]'); return [b.disabled, b.title]; })()")
    assert web == [True, "MoE の層は connections の対象外"]
    specs = lambda s=1: pg.evaluate(f"window.gne.recipe.specs({s})")  # noqa: E731
    pg.click("#neuron-body [data-rc='1']")
    assert specs() == [f"{L}:E{E}:N7:+1"]
    pg.click("#neuron-body [data-rc='-1']")
    assert specs() == [f"{L}:E{E}:N7:-1"]
    pg.click("#neuron-body [data-gk='0']")
    assert specs() == [f"{L}:E{E}:N7:-1", f"{L}:E{E}:N7:x0"] and specs(-1) == [f"{L}:E{E}:N7:+1", f"{L}:E{E}:N7:x2"]
    pg.click("#neuron-body [data-gk='0']")                                            # もう一度押すと外す
    pg.click("#neuron-body [data-gk='2']")
    assert specs() == [f"{L}:E{E}:N7:-1", f"{L}:E{E}:N7:x2"]
    marks = pg.evaluate("window.gne.map.marks")
    assert {"layer": L, "neuron": flat, "dose": -1, "label": f"L{L} E{E} N7 −1"} in marks
    pg.click("#neuron-body button[data-tab=details]")
    det = pg.inner_text("#neuron-body")
    assert "MoE の層は connections の対象外" in det and "トークンだけで数えた値" in det
    pg.click("#neuron-body button[data-tab=chat]")
    pg.wait_for_selector("#ch-text", timeout=30000)
    if pg.evaluate("window.gne.chat.target") != "neuron":
        pg.click("[data-target=neuron]")
    assert pg.evaluate("window.gne.chat.columns().map((c) => c.units)") == [[f"{L}:E{E}:N7:-0.8"], [], [f"{L}:E{E}:N7:+0.8"]]
    assert pg.evaluate("!!document.querySelector('#neuron-body .ch-tools .badge.push')")
    # recipe タブ: 方向押しの印、gain の行の倍率
    _open_list(pg, "recipe")
    rc = pg.inner_text("#recipe-list table.rc")
    assert f"L{L} E{E} N7" in rc and "選ばれたときだけ" in rc and pg.evaluate("document.querySelectorAll('#recipe-list .badge.push').length") == 1
    assert "方向押し = 専門家のニューロンの add" in pg.inner_text("#recipe-list")
    _recipe_tab(pg)
    pg.fill("#recipe-list tr.rc-gain input.num", "0.5")
    pg.dispatch_event("#recipe-list tr.rc-gain input.num", "input")
    assert specs() == [f"{L}:E{E}:N7:-1", f"{L}:E{E}:N7:x0.5"] and specs(-1) == [f"{L}:E{E}:N7:+1", f"{L}:E{E}:N7:x1.5"]
    # 専門家全体・route・shared(/api/recipes の行の形で読む)と、列見出しの表示
    pg.evaluate("""window.gne.recipe.load('x', [{ layer: 12, kind: 'expert', expert: 3, mode: 'route', dose: 2, neuron: null },
      { layer: 12, kind: 'shared', neuron: 5, mode: 'add', dose: 0.5, expert: null }, { layer: 12, kind: 'expert', expert: 3, mode: 'gain', k: 0, dose: 0 },
      { layer: 12, kind: 'expert_neuron', expert: 30, neuron: 511, mode: 'add', dose: 0.8 }])""")
    assert pg.evaluate("window.gne.recipe.skipped") == 0
    assert specs() == ["12:E3:route:+2", "12:S:N5:+0.5", "12:E3:x0", "12:E30:N511:+0.8"]              # 読んだ順のまま
    assert specs(-1) == ["12:E3:route:-2", "12:S:N5:-0.5", "12:E3:x2", "12:E30:N511:-0.8"]
    assert pg.evaluate("window.gne.recipe.addSpecs()") == ["12:S:N5:+0.5", "12:E30:N511:+0.8"]   # gain と route は cvec に入らない
    labels = pg.evaluate("""import('./chat.js').then((m) => [...window.gne.recipe.specs(-1),
      '14:8949:+1', '21:H7:x0', '30:dir:pirate:+0.3', '12:E37:N511:x2'].map(m.unitLabel))""")
    assert labels == ["L12 E3 route −2", "L12 S N5 −0.5", "L12 E3 ×2", "L12 E30 N511 −0.8", "L14 N8949 +1", "L21 H7 ×0", "L30 dir:pirate +0.3",
                      "L12 E37 N511 ×2"]
    _open_list(pg, "rank")
    _open_list(pg, "recipe")
    rc = pg.inner_text("#recipe-list table.rc")
    assert "L12 E30 N511" in rc and "L12 S N5" in rc and "L12 E3" in rc and "route" in rc
    assert pg.evaluate("document.querySelectorAll('#recipe-list .badge.push').length") == 2
    # 保存したレシピ(サーバ)の読み込み: expert が戻る
    req = urllib.request.Request(url + "/api/recipes/ui_moe", data=json.dumps({"units": [f"{L}:E{E}:N7:+0.8", f"{L}:E{E}:N7:x2", f"{L}:E{E}:x0"]}).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req).read()
    _open_list(pg, "rank")
    _open_list(pg, "recipe")
    pg.wait_for_function("[...document.querySelectorAll('#rc-load option')].some((o) => o.value === 'ui_moe')", timeout=10000)
    pg.once("dialog", lambda d: d.accept())                                        # いまのレシピを置き換える前に確かめる
    pg.select_option("#rc-load", "ui_moe")
    assert specs() == [f"{L}:E{E}:N7:+0.8", f"{L}:E{E}:N7:x2", f"{L}:E{E}:x0"] and pg.evaluate("window.gne.recipe.skipped") == 0
    pg.click("#n-expert")                                                              # EXPERT パネルで開く
    pg.wait_for_function(f"document.querySelector('#expert-body .n-id') && document.querySelector('#expert-body .n-id').textContent.startsWith('L{L} E{E} ')",
                         timeout=30000)
    assert pg.evaluate("document.querySelector('#expert-body [data-xk=\"0\"]').classList.contains('on')")   # 読み込んだ ×0
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_moe_panels_in_english(moe_server, browser):
    """?lang=en で EXPERT パネル・MoE のニューロン・レシピ・fires の routed・専門家の overlay の文言がすべて英語の辞書にある。"""
    url, L, E, W = moe_server["url"], moe_server["L"], moe_server["E"], moe_server["W"]
    pg = browser.new_page(viewport={"width": 1500, "height": 860})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(url + f"/?lang=en#L{L}/E{E}")
    pg.wait_for_function("window.gne && window.gne.overlayReady && document.querySelector('#expert-body .n-id')", timeout=60000)
    pg.evaluate("(() => { window.gne.recipe.clear(); window.gne.expertPanel.render(); })()")
    assert "selection rate" in pg.inner_text("#expert-body")
    pg.click("#expert-body [data-xk='0']")
    pg.click("#expert-body table.x-top tbody tr")
    pg.wait_for_function("document.querySelector('#neuron-body .n-moe-note')", timeout=30000)
    pg.click("#neuron-body [data-rc='1']")
    pg.click("#neuron-body [data-gk='2']")
    for tab in ("details", "chat", "examples"):
        pg.click(f"#neuron-body button[data-tab={tab}]")
        pg.wait_for_timeout(300)
    _open_list(pg, "recipe")
    assert "direction push" in pg.inner_text("#recipe-list")
    _open_list(pg, "fires")
    pg.fill("#fi-text", "Arr matey.")
    pg.dispatch_event("#fi-text", "input")
    pg.click("#fi-run")
    pg.wait_for_selector(".chip-t[data-tok='2']", timeout=60000)
    pg.click(".chip-t[data-tok='2']")
    pg.wait_for_function("window.gne.firesRouted", timeout=30000)
    for f in ("expert_usage", "experts:pirate"):
        pg.select_option("#color-select", f)
        pg.wait_for_function(f"window.gne.overlay.field === '{f}' && window.gne.overlay.scale", timeout=30000)
        pg.select_option("#layer-select", str(L))
        pg.wait_for_timeout(300)
        sp = pg.evaluate(f"window.gne.map.screenOf({L}, {E * W + 40})")
        pg.mouse.move(sp["x"], sp["y"])
        pg.wait_for_timeout(150)
    pg.fill("#jump", "L3 E0")                                                          # 活性統計も対照分析も無い層
    pg.press("#jump", "Enter")
    pg.wait_for_function("document.querySelector('#expert-body .n-id') && document.querySelector('#expert-body .n-id').textContent.startsWith('L3 E0 ')",
                         timeout=30000)
    assert "No activation stats" in pg.inner_text("#expert-body")
    assert pg.evaluate("[...window.gneI18n.missing]") == []
    assert not errors, errors
    pg.close()


# --- CHANGE-06 G: 因果スクリーニング・ヘッドの射影・direction ビュー・eval の方式の比較とヘッド数の曲線 ------------------------

def _seed_causal(s, name: str = "cz", direction: str = "czdir") -> None:
    """方向セット(乱数の向き、対照集合は configs/contrast/evil.yaml、基準層はまだ無い)、因果スクリーニング(ablate・marginal、
    射影の値を測る層 14・20)と強さの配分、方式の比較とヘッド数の曲線の eval、proj のレシピ(default_direction で方向を省いた行を含む)。"""
    import yaml

    from gne.causal import save_causal
    m = s.meta
    rng = np.random.default_rng(0)
    Ls = list(range(m.n_layer))
    vecs = {L: rng.standard_normal(m.n_embd).astype(np.float32) for L in Ls}
    U = np.stack([v / np.linalg.norm(v) for v in vecs.values()])
    np.savez(s.store.direction_dir(m.model_id) / f"{direction}.npz", layers=np.array(Ls, np.int32),
             **{f"layer_{L}": v for L, v in vecs.items()}, **{f"scale_{L}": np.ones(m.n_embd, np.float32) for L in Ls},
             cos=(U @ U.T).astype(np.float32), method="mean_diff", positions="last", contrast_sha="x",
             source=str(ROOT / "configs" / "contrast" / "evil.yaml"), n_rows_A=12, n_rows_B=12)
    heads = [(12, 3, 9.0, 2.5), (12, 5, 4.0, 3.1), (14, 0, 2.0, -0.5), (14, 7, -1.0, 0.4)]   # (層, ヘッド, causal_max, η)
    common = {"b_side_delta": 0.2, "drift_par_b": 0.01, "drift_orth_b": 0.05, "contrib_by_layer": {"14": 1.0, "20": 0.5},
              "contrib": 1.0, "seconds": 1.0}
    abl = [{"layer": L, "head": h, "kind": "attn", "mode": "ablate", "score_drop": {"14": cm / 2, "20": cm}, "causal": cm / 2,
            "causal_max": cm, "ratio": cm / 2, **common} for L, h, cm, _ in heads]
    mar = [{"layer": L, "head": h, "kind": "attn", "mode": "marginal", "score_drop": {"14": 3 * e, "20": 6 * e}, "causal": 3 * e,
            "causal_max": 6 * e, "ratio": 3 * e, "delta_s": 3.0, "eta": e, "eta_by_layer": {"14": e, "20": 2 * e},
            "in_base": (L, h) in ((12, 3), (12, 5)), "excluded": (L, h) == (14, 7), **common} for L, h, _, e in heads]
    summ = {"direction": direction, "contrast": "evil", "score_layers": [14, 20], "primary": 14, "n_a": 12, "n_b": 12,
            "base_scores": {"14": {"A": 30.0, "B": -9.0}, "20": {"A": 80.0, "B": -20.0}}, "updated": "2026-11-01T00:00:00",
            "layer_rows": [{"layer": 12, "kind": "attn", "n_heads": 16, "score_drop": {"14": 8.0, "20": 15.0}, "causal": 8.0,
                            "causal_max": 15.0, "b_side_delta": 1.0, "drift_orth_b": 0.2}]}
    save_causal(s, name, abl, {**summ, "mode": "ablate"})
    save_causal(s, name, mar, {**summ, "mode": "marginal", "spearman_ablate_marginal": 0.4})
    hd = s.store.heads_dir(m.model_id)
    (hd / f"{name}.alloc.json").write_text(json.dumps({
        "name": name, "direction": direction, "score_layer": 20, "target": 0.0, "recommended": 2,
        "table": [{"n_heads": 2, "units": [f"12:H5:s6@{direction}", f"12:H3:s4@{direction}"], "s_max": 6.0, "remain": 1, "a_max": 2.0,
                   "b_drift_orth": 0.1, "b_side_delta": 3.0, "recommended": True, "reached": True,
                   "heads": [{"layer": 12, "head": 5, "s": 6.0, "eta": 3.1}, {"layer": 12, "head": 3, "s": 4.0, "eta": 2.5}]},
                  {"n_heads": 1, "units": [f"12:H5:s12@{direction}"], "s_max": 12.0, "remain": 3, "a_max": 5.0,
                   "b_drift_orth": 0.3, "b_side_delta": 6.0, "recommended": False, "reached": False,
                   "heads": [{"layer": 12, "head": 5, "s": 12.0, "eta": 3.1}]}],
        "eta": [{"layer": L, "head": h, "eta": e} for L, h, _, e in heads]}))
    ed = s.store.eval_dir(m.model_id)
    (ed / "cz-methods.json").write_text(json.dumps({
        "name": "cz-methods", "direction": direction, "score_layer": 14, "base_layer": 14, "target": -5.0, "target_from": "recipe",
        "base_score_a": 30.0, "base_score_b": -9.0, "base_rate": 0.5, "seconds": 3.0, "rows": [
            {"method": "ablate", "desc": "all layers", "alpha": 1.5, "score_a": 2.0, "excluded": "does not reach the target"},
            {"method": "proj", "desc": "proj", "alpha": 1.0, "score_a": -5.0, "drift_orth_b": 0.1, "capability_drop_pt": 0.0,
             "ppl_worse_pct": 1.0, "mix_increase": 0.0, "rate": 0.1, "rate_drop": 0.4, "shippable": True,
             "recipe": {"name": "cz-proj", "units": [{"layer": 12, "kind": "head", "mode": "proj", "head": 5, "k": 6.0, "direction": None}],
                        "default_direction": direction, "ablate": None, "sampling": None}},
            {"method": "head_gain", "desc": "gain", "alpha": 0.9, "score_a": -5.1, "drift_orth_b": 1.5, "capability_drop_pt": 40.0,
             "ppl_worse_pct": 300.0, "mix_increase": 0.0, "rate": 0.0, "rate_drop": 0.5, "shippable": False}]}))
    (ed / "cz-nheads.json").write_text(json.dumps({
        "name": "cz-nheads", "alloc": str(hd / f"{name}.alloc.json"), "direction": direction, "score_layer": 20, "target": 0.0,
        "random_controls": 0, "seconds": 5.0, "rows": [
            {"n_heads": n, "units": u, "s_max": sm, "remain": rm, "recommended": n == 2, "capability_drop_pt": cd, "ppl_worse_pct": 1.0,
             "mix_increase": 0.0, "dir_drop_a": 50.0, "drift_orth_a": 0.2, "drift_orth_b": ob, "rate": 0.1, "rate_drop": 0.3,
             "shippable": True, "notes": []}
            for n, u, sm, rm, cd, ob in [(2, [f"12:H5:s6@{direction}", f"12:H3:s4@{direction}"], 6.0, 1, 0.0, 0.1),
                                         (1, [f"12:H5:s12@{direction}"], 12.0, 3, 5.0, 0.3)]]}))
    (s.store.recipes_dir() / "czr.yaml").write_text(yaml.safe_dump({
        "name": "czr", "model_id": m.model_id, "dose_unit": "rel", "default_direction": direction,
        "units": [{"layer": 12, "kind": "head", "mode": "proj", "head": 3, "s": 6.0},
                  {"layer": 12, "kind": "head", "mode": "proj", "head": 5, "s": 4.0, "direction": direction},
                  {"layer": 14, "kind": "head", "mode": "gain", "head": 0, "k": 0.0}]}))


@pytest.mark.model
def test_causal_proj_direction_api(tmp_path):
    """CHANGE-06 G のサーバ: 因果スクリーニングの一覧・1 件・ヘッドの行と順位、方向セット(層どうしの cos)と速い指標
    (基準層の記録と手での上書き)、proj のレシピの読み書き(default_direction・direction_layer)、proj のユニットでの 3 列チャット。"""
    from fastapi.testclient import TestClient

    from gne.api import Session
    from gne.ui.server import create_app
    # 前のテストの Session が GPU に残っている(閉じていない)ので、ファイルの最後に走るこのテストは CPU に載せる
    with Session(MODEL, root=tmp_path, n_ctx=2048, n_seq_max=4, n_gpu_layers=0) as s:
        _seed_causal(s)
        c = TestClient(create_app(s, warm=False))
        m = c.get("/api/model").json()
        cz = m["causal"][0]
        assert (cz["name"], cz["modes"], cz["primary"], cz["alloc"]) == ("cz", ["ablate", "marginal"], 14, True)
        assert cz["layers"] == {"ablate": [12, 14], "marginal": [12, 14]} and cz["n_rows"] == {"ablate": 4, "marginal": 4}
        assert [d["name"] for d in m["direction_sets"]] == ["czdir"] and m["direction_sets"][0]["base_layer"] is None
        one = c.get("/api/causal/cz").json()
        assert len(one["rows"]) == 8 and one["score_layers"] == [14, 20] and one["alloc"]["recommended"] == 2
        assert one["layer_rows"][0]["layer"] == 12 and one["spearman_ablate_marginal"] == 0.4
        assert c.get("/api/causal/nope").status_code == 400 and c.get("/api/causal/..%2Fx").status_code in (400, 404)
        h = c.get("/api/head", params={"layer": 12, "head": 5, "lens": False}).json()["causal"][0]
        assert h["name"] == "cz" and {r["mode"] for r in h["rows"]} == {"ablate", "marginal"}
        assert h["rank"] == {"ablate": {"rank": 2, "n": 4}, "marginal": {"rank": 1, "n": 4}}
        assert c.get("/api/head", params={"layer": 3, "head": 0, "lens": False}).json()["causal"] == []
        # 方向セット: 層どうしの cos(モデルは使わない)。測ると、基準層が無ければ cohen_d 最大の層を記録する
        d = c.get("/api/directions/czdir").json()
        assert len(d["cos"]) == len(d["layers"]) == s.meta.n_layer and abs(d["cos"][5][5] - 1) < 1e-4 and d["contrast_ok"]
        assert c.get("/api/directions/nope").status_code == 400
        r = c.post("/api/directions/czdir/score", json={}).json()
        assert r["base_recorded"] and r["meta"]["base_layer"] == r["best_layer"] and r["n"] == {"A": 12, "B": 12}
        assert len(r["values"]["A"][str(r["best_layer"])]) == 12 and len(r["prompts"]) == 12
        other = 3 if r["best_layer"] != 3 else 4
        r2 = c.post("/api/directions/czdir/score", json={"set_base": other}).json()
        assert r2["base_layer"] == other and r2["meta"]["base_layer"] == other
        r3 = c.post("/api/directions/czdir/score", json={"record_base": False}).json()           # 測り直しても手で決めた層は残る
        assert not r3["base_recorded"] and c.get("/api/directions").json()[0]["base_layer"] == other
        # proj のレシピ: 方向を省いた行は default_direction で埋めて返す。direction_layer: base で保存できる
        czr = next(x for x in c.get("/api/recipes").json() if x["name"] == "czr")
        pu = [u for u in czr["units"] if u["mode"] == "proj"]
        assert [(u["head"], u["k"], u["direction"]) for u in pu] == [(3, 6.0, "czdir"), (5, 4.0, "czdir")]
        assert (czr["default_direction"], czr["direction_layer"]) == ("czdir", "own")
        assert c.post("/api/recipes/czb", json={"units": ["12:H3:s2.5@czdir", "14:H0:x0"], "direction_layer": "base"}).status_code == 200
        czb = next(x for x in c.get("/api/recipes").json() if x["name"] == "czb")
        assert czb["direction_layer"] == "base" and [u["mode"] for u in czb["units"]] == ["proj", "gain"]
        msgs = [{"role": "user", "content": "Who are you?"}]
        ev = _sse_events(c, "/api/steer", {"max_tokens": 4, "columns": [{"units": ["12:H3:s6@czdir", "12:H5:s-4@czdir"], "messages": msgs},
                                                                          {"units": [], "messages": msgs}]})
        assert not [e for e in ev if e["type"] == "error"] and [e["n_gen"] for e in ev if e["type"] == "end"] == [4, 4]


@pytest.fixture(scope="module")
def causal_server(tmp_path_factory):
    """CHANGE-06 G のブラウザのテスト用のサーバ(_seed_causal のデータ)。"""
    if not MODEL.exists():
        pytest.skip("モデルがありません")
    from gne.api import Session
    s = Session(MODEL, root=tmp_path_factory.mktemp("ui_causal"), n_ctx=2048, n_seq_max=4, n_gpu_layers=0)   # 上と同じ理由で CPU
    _seed_causal(s)
    yield from _serve(s)


def _serve(s):
    """Session を UI のサーバで出す(テストの fixture 用)。閉じる前に、走っているジョブを止め、モデルの処理(direction ビューの
    測定など、ページを閉じても続く)の終わりを GPU の調停で待つ。待たずに閉じると処理中の llama.cpp を解放して落ちる。
    待つのは 300 秒まで(それでも終わらなければ閉じずに進む。テスト全体が止まらないように)。"""
    import uvicorn

    from gne.ui.server import create_app
    app = create_app(s)
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
    closed = threading.Event()

    def close():
        with live.arbiter.interactive():
            s.close()
        closed.set()
    threading.Thread(target=close, daemon=True).start()
    closed.wait(300)


def _head_rgb(pg, L, h):
    return pg.evaluate(f"Array.from(window.gne.map.layerParts[{L}].heads.geometry.getAttribute('color').array.slice({18 * h}, {18 * h + 3}))")


@pytest.mark.model
def test_causal_proj_direction_in_browser(causal_server, browser):
    """HEAD パネルの Causal(このヘッドの行・上位の表の並べ替えと射影の値を測る層・行で別のヘッドへ・配分の組をレシピに)、
    地図の causal:<name>・eta:<name>、recipe タブの proj の行(s・方向・rank)、direction ビュー(基準層の記録と上書き)、
    eval ビューの方向の列・方式の比較・ヘッド数の曲線。英語の文言もすべて辞書にある。"""
    url = causal_server
    for lang in ("ja", "en"):
        pg = browser.new_page(viewport={"width": 1600, "height": 900})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(url + f"/?lang={lang}#L12/H3")
        pg.wait_for_function("window.gne && window.gne.overlayReady && document.querySelector('#head-body .n-id')", timeout=60000)
        pg.evaluate("window.gne.recipe.clear()")
        opts = pg.evaluate("[...document.querySelectorAll('#color-select option')].map(o => o.value)")
        assert "causal:cz" in opts and "eta:cz" in opts
        pg.click("#head-body .tabs button[data-tab=causal]")
        pg.wait_for_function("document.querySelectorAll('#head-body table.cz-top tr[data-go]').length === 4", timeout=30000)
        go = "[...document.querySelectorAll('#head-body table.cz-top tr[data-go]')].map(tr => tr.dataset.go)"
        assert pg.evaluate(go) == ["12:3", "12:5", "14:0", "14:7"]                           # causal_max の順
        assert pg.evaluate("document.querySelector('#head-body table.cz-top tr.sel').dataset.go") == "12:3"
        pg.click("#head-body th[data-czsort=eta]")
        assert pg.evaluate(go)[0] == "12:5"
        pg.select_option("#cz-layer", "20")                                                  # 射影の値を測る層を替えると下がり方の列も替わる
        assert "causal L20" in pg.evaluate("document.querySelector('#head-body table.cz-top').innerText")
        pg.click("#head-body th[data-czsort=causal]")
        cells = pg.evaluate("[...document.querySelectorAll('#head-body table.cz-top tr[data-go] td')].slice(0, 4).map(td => td.textContent)")
        assert cells[1] == "L12 H3" and cells[3] == "9.00"
        # 射影 s1 のボタンと、配分の組の読み込み
        pg.click("#head-body [data-hp]")
        assert pg.evaluate("window.gne.recipe.specs()") == ["12:H3:s1@czdir"]
        asked = []
        pg.once("dialog", lambda d: (asked.append(d.message), d.accept()))                     # レシピを置き換える前に確かめる
        pg.click("#head-body [data-alloc='2']")
        assert asked and pg.evaluate("window.gne.recipe.specs()") == ["12:H5:s6@czdir", "12:H3:s4@czdir"]
        assert pg.evaluate("window.gne.recipe.specs(-1)") == ["12:H5:s-6@czdir", "12:H3:s-4@czdir"]
        assert not pg.evaluate("document.querySelector('#recipe').classList.contains('collapsed')")       # 置き換えたらレシピのパネルが開く
        assert pg.evaluate("document.querySelectorAll('#recipe-list tr.rc-proj').length") == 2
        assert pg.evaluate("[...document.querySelectorAll('#recipe-list .badge.rank1')].length") == 2
        pg.wait_for_function("document.querySelectorAll('#overlay span.mk-h.minus').length === 2", timeout=10000)   # 地図のヘッドの印(パネルが開いて地図の幅が変わった後の描き直しを待つ)
        pg.click("#head-body table.cz-top tr[data-go='14:0']")                                # 行で別のヘッドを開く(Causal のまま)
        pg.wait_for_function("(document.querySelector('#head-body .n-id') || {}).textContent?.startsWith('L14 H0 ')", timeout=30000)
        assert pg.evaluate("window.gne.headPanel.tab") == "causal"
        # 地図: causal:cz は止めたときの下がり方(正 = 黄、負 = 赤紫)、eta:cz は効率(外したヘッドは別の色)
        pg.click("#head-body [data-czov=causal]")
        pg.wait_for_function("window.gne.heads && window.gne.heads.field === 'causal:cz'", timeout=30000)
        r, g_, b = _head_rgb(pg, 12, 3)
        assert r > g_ > b > 0
        r, g_, b = _head_rgb(pg, 14, 7)
        assert r > g_ and b > g_                                                              # 赤紫
        legend = pg.evaluate("document.querySelector('#legend').textContent")
        assert ("causal_max" in legend) and (("対照の z" in legend) if lang == "ja" else ("contrast z" in legend))
        pg.select_option("#color-select", "eta:cz")
        pg.wait_for_function("window.gne.heads && window.gne.heads.field === 'eta:cz'", timeout=30000)
        assert [round(x * 255) for x in _head_rgb(pg, 14, 7)] == [64, 112, 124]
        # recipe タブ: 保存したレシピの proj の行(方向を省いた行は default_direction)と gain の行
        _open_list(pg, "rank")
        _recipe_tab(pg)
        pg.wait_for_function("[...document.querySelectorAll('#rc-load option')].some((o) => o.value === 'czr')", timeout=10000)
        pg.once("dialog", lambda d: d.accept())                                    # いまのレシピを置き換える前に確かめる
        pg.select_option("#rc-load", "czr")
        assert pg.evaluate("window.gne.recipe.specs()") == ["12:H3:s6@czdir", "12:H5:s4@czdir", "14:H0:x0"]
        assert pg.evaluate("window.gne.recipe.skipped") == 0
        pg.fill("#recipe-list tr.rc-proj input.num >> nth=1", "2.5")
        pg.dispatch_event("#recipe-list tr.rc-proj input.num >> nth=1", "input")
        assert pg.evaluate("window.gne.recipe.specs()")[1] == "12:H5:s2.5@czdir"
        assert "s2.5" in pg.evaluate("document.querySelectorAll('#recipe-list tr.rc-proj')[1].innerText")
        # direction ビュー: 開いただけでは測らない(モデルを流す計算は「測り直す」を押したときだけ)
        score_posts = []
        pg.on("request", lambda rq: score_posts.append(rq.url) if rq.method == "POST" and rq.url.endswith("/score") else None)
        pg.click("#views button[data-view=direction]")
        pg.wait_for_function("window.gne.directionView && window.gne.directionView.meta", timeout=60000)
        pg.wait_for_load_state("networkidle")
        assert score_posts == []                                                         # 出るのは前の回(別の言語)に測って残した結果だけ
        assert pg.evaluate("[...window.gne.directionView.scores.values()].every((r) => r.saved)")
        # 「測り直す」で測り、基準層が無ければ cohen_d 最大の層を記録する。別の層を基準層にできる
        pg.click("#dv-run")
        pg.wait_for_function("window.gne.directionView.scores.size && !window.gne.directionView.busy", timeout=180000)
        base = pg.evaluate("window.gne.directionView.meta.base_layer")
        assert base is not None and pg.evaluate("document.querySelector('#dv-base').disabled")
        assert pg.evaluate("document.querySelector('#dv-line').width") > 0 and pg.evaluate("document.querySelector('#dv-hist').width") > 0
        other = 5 if base != 5 else 6
        pg.evaluate(f"window.gne.directionView.sel = {other}; window.gne.directionView.draw()")
        pg.click("#dv-base")
        pg.wait_for_function(f"window.gne.directionView.meta.base_layer === {other} && !window.gne.directionView.busy", timeout=180000)
        pg.click("#dv-ex summary")
        assert pg.evaluate("document.querySelectorAll('#dv-ex table.dv-ex tbody tr').length") == 12
        # 測った結果は残り、開き直すと測らずに出る(GET だけ)
        n_posts = len(score_posts)
        pg.evaluate("(() => { const v = window.gne.directionView; v.scores.clear(); v.select(v.name); })()")
        pg.wait_for_function("window.gne.directionView.scores.size === 1", timeout=30000)
        assert len(score_posts) == n_posts and pg.evaluate("window.gne.directionView.scores.get(window.gne.directionView.name).saved")
        assert pg.evaluate("window.gne.directionView.meta.base_layer") == other
        # eval ビュー: 方式の比較(外した方式は理由)、ヘッド数の曲線(推奨の組、行の組をレシピに)、方向の列
        pg.click("#views button[data-view=eval]")
        pg.wait_for_function("window.gne.evalView && window.gne.evalView.data", timeout=30000)
        pg.wait_for_load_state("networkidle")
        pg.select_option("#evx-pick", "cz-methods")
        pg.wait_for_function("window.gne.evalView.data.name === 'cz-methods' && document.querySelector('table.evx-methods')", timeout=30000)
        assert pg.evaluate("document.querySelectorAll('table.evx-methods tbody tr').length") == 3
        assert "does not reach the target" in pg.evaluate("document.querySelector('table.evx-methods td.excl').textContent")
        assert pg.evaluate("document.querySelector('table.evx-methods tr.best td').textContent") in ("ヘッドの方向成分(proj)", "head direction component (proj)")
        pg.once("dialog", lambda d: d.accept())                                    # いまのレシピを置き換える前に確かめる
        pg.click("table.evx-methods [data-rc]")
        assert pg.evaluate("window.gne.recipe.specs()") == ["12:H5:s6@czdir"]
        pg.select_option("#evx-pick", "cz-nheads")
        pg.wait_for_function("window.gne.evalView.data.name === 'cz-nheads' && document.querySelector('#evx-nh')", timeout=30000)
        assert pg.evaluate("document.querySelector('#evx-nh').width") > 0
        assert pg.evaluate("document.querySelectorAll('#evx-body table.evx tbody tr').length") == 2
        assert pg.evaluate("document.querySelector('#evx-body table.evx tr.rec td').textContent").startswith("2")
        pg.once("dialog", lambda d: d.accept())                                                # いまのレシピを置き換える前に確かめる
        pg.click("#evx-body [data-nh='0']")                                                    # x の小さい順(1 ヘッドの組)
        assert pg.evaluate("window.gne.recipe.specs()") == ["12:H5:s12@czdir"]
        pg.evaluate("""(() => {
          const ev = window.gne.evalView, row = (dose, x) => ({ dose, capability: 1, capability_drop_pt: 0, ppl_mean: 5, ppl_worse_pct: 0, mix_zh: 0,
            mix_other: 0, persona: { novelty: 0.1, distinct4: 1, capped: 0, top_words: [] }, shippable: true, ok: {}, failed: { exact: [], format: [] },
            dir_drop_a: x, drift_par_a: 0.1, drift_orth_a: 0.2, drift_par_b: 0.05, drift_orth_b: 0.1, rate: 0.5 - x / 100, rate_drop: x / 100,
            notes: x ? ["note at dose 1"] : [], direction: { layers: { "14": { score_a: 1, score_b: -1, drop_a: x, delta_b: 0.5, drift_par_a: 0.1,
              drift_orth_a: 0.2, drift_par_b: 0.05, drift_orth_b: 0.1 } } } });
          ev.data = { name: "dx", when: "all", seconds: 1, suite: { n_tasks: 3, thresholds: { capability_drop_pt: 5, ppl_worse_pct: 10 } },
            recipe: { units: [{ layer: 12, kind: "head", mode: "proj", head: 3, k: 6, direction: "czdir" }] }, rows: [row(0, 0), row(1, 30)],
            samples: { "1": { persona: [], failed: [] } }, direction_metrics: { direction: "czdir", layers: [14, 20], behavior: true, base_rate: 0.5 },
            notes: ["top-level note"] };
          ev.dose = 1; ev.render();
        })()""")
        th = pg.evaluate("[...document.querySelector('#evx-body table.evx').querySelectorAll('thead th')].map(th => th.textContent)")
        assert th[-5:-3] == ["A ⊥", "B ⊥"]                                                     # A 下がり・A ⊥・B ⊥・一致率・一致減
        cells = pg.evaluate("[...document.querySelector('#evx-body table.evx').querySelectorAll('tbody tr[data-dose=\"1\"] td')].map(td => td.textContent)")
        assert cells[-6:-1] == ["30.00", "0.200", "0.100", "20%", "30%"]
        body = pg.evaluate("document.querySelector('#evx-body').textContent")
        assert "L12 H3 s6 @czdir" in body and "top-level note" in body and "note at dose 1" in body
        if lang == "en":
            assert pg.evaluate("[...window.gneI18n.missing]") == []
        assert not errors, errors
        pg.close()


# --- direction ビューの「JSON から方向とヘッドを作る」(head_pipeline のジョブ) -------------------------------------------

def _seed_pipeline(s, name: str = "ppx") -> dict:
    """一括の結果(Session.heads_pipeline の出力の形)。方向・因果スクリーニング・配分は _seed_causal の同じ名前のもの。
    入力に使った対照集合(<root>/contrasts/<name>.yaml)と、配分の組のレシピも置く。"""
    import yaml
    _seed_causal(s, name=name, direction=name)
    m = s.meta
    (s.store.direction_dir(m.model_id) / f"{name}.npz").unlink()      # 方向が無ければ direction ビューは「JSON から作る」で開く(測定しない)
    hd = s.store.heads_dir(m.model_id)
    spec = {"name": name, "system_a": "You are a pirate.", "system_b": "You are a helpful assistant.",
            "user_prompts": [f"Question {i}?" for i in range(6)]}
    cp = s.store.contrast_dir() / f"{name}.yaml"
    cp.write_text(yaml.safe_dump(spec))
    units = {3: [f"12:H5:s9@{name}", f"12:H3:s6@{name}", f"14:H0:s2@{name}"], 2: [f"12:H5:s10@{name}", f"12:H3:s7@{name}"]}
    for n, us in units.items():
        (hd / f"{name}.alloc-n{n}.yaml").write_text(yaml.safe_dump({"name": f"{name}-n{n}", "model_id": m.model_id, "default_direction": name,
            "units": [{"layer": int(u.split(":")[0]), "kind": "head", "mode": "proj", "head": int(u.split(":")[1][1:]),
                       "s": float(u.split(":s")[1].split("@")[0])} for u in us]}))
    head = lambda L, h, cm, **kw: {"layer": L, "head": h, "kind": "attn", "mode": "ablate", "score_drop": {"14": cm / 2, "20": cm},  # noqa: E731
                                   "causal": cm / 2, "causal_max": cm, "b_side_delta": 0.2, "drift_orth_b": 0.05,
                                   "contrib_by_layer": {"14": 1.0, "20": 0.5}, "contrib": 1.0, "ratio": cm / 2, **kw}
    out = {
        "name": name, "contrast": str(cp), "n_prompts": 6,
        "sides": {"A": {"system": "You are a pirate.", "recipe": None}, "B": {"system": "You are a helpful assistant.", "recipe": None}},
        "settings": {"positions": "last", "score_positions": "last", "target": None, "n_heads": [3, 2], "hard": 4, "top_layers": 2,
                     "max_examples": 6, "s_max": 30.0, "direction_layer": "own"},
        "direction": {"path": "x.npz", "positions": "last", "n_rows": {"A": 6, "B": 6}, "seconds": 1.0},
        "score": {"base_layer": 14, "best_layer": 14, "n": {"A": 6, "B": 6}, "seconds": 1.0,
                  "layers": [{"layer": L, "cohen_d": 5.0 - abs(L - 14) * 0.3, "cos_base": 1.0 - abs(L - 14) * 0.05, "A": 3.0, "B": -3.0}
                             for L in range(1, s.meta.n_layer - 1)]},
        "causal": {"path": "x.causal.jsonl", "score_layers": [14, 20], "primary": 14, "n_rows": 4, "n_a": 6, "n_b": 6,
                   "base_scores": {"14": {"A": 30.0, "B": -9.0}, "20": {"A": 80.0, "B": -20.0}}, "seconds": 2.0,
                   "layers": [{"layer": 0, "kind": "attn", "n_heads": 16, "score_drop": {"14": 50.0, "20": 99.0}, "causal": 50.0, "causal_max": 99.0,
                               "b_side_delta": 9.0, "drift_orth_b": 1.2, "excluded": "breaks"},
                              {"layer": 12, "kind": "attn", "n_heads": 16, "score_drop": {"14": 8.0, "20": 15.0}, "causal": 8.0, "causal_max": 15.0,
                               "b_side_delta": 1.0, "drift_orth_b": 0.2},
                              {"layer": 14, "kind": "attn", "n_heads": 16, "score_drop": {"14": 3.0, "20": 6.0}, "causal": 3.0, "causal_max": 6.0,
                               "b_side_delta": 0.5, "drift_orth_b": 0.1}],
                   "top": [head(1, 12, 80.0, excluded=True), head(12, 3, 9.0), head(12, 5, 4.0), head(14, 0, 2.0)]},
        "allocate": {"name": name, "direction": name, "score_layer": 20, "score_layers": [14, 20], "target": 25.0, "n_a": 6, "n_b": 6,
                     "initial": {"heads": [f"12:H3:s12@{name}", f"12:H5:s5.333@{name}", f"14:H0:s2.667@{name}"], "alpha": 12.0, "mean_a": 24.0,
                                 "reached": True},
                     "hard": {"n": 2, "index": [1, 4], "prompts": ["Question 1?", "Question 4?"], "scores": [40.0, 33.0], "layer": 20},
                     "eta": [{"layer": 12, "head": 5, "kind": "attn", "eta": 3.1, "causal": 9.3, "b_side_delta": 0.3, "drift_orth_b": 0.02,
                              "excluded": None, "in_base": True},
                             {"layer": 12, "head": 3, "kind": "attn", "eta": 2.5, "causal": 7.5, "b_side_delta": 0.4, "drift_orth_b": 0.03,
                              "excluded": None, "in_base": True},
                             {"layer": 14, "head": 0, "kind": "attn", "eta": 0.5, "causal": 1.5, "b_side_delta": 0.1, "drift_orth_b": 0.9,
                              "excluded": True, "in_base": True}],
                     "table": [{"n_heads": n, "units": us, "heads": [{"layer": int(u.split(":")[0]), "head": int(u.split(":")[1][1:]),
                                                                     "s": float(u.split(":s")[1].split("@")[0]), "eta": 1.0} for u in us],
                                "s_max": 10.0 if n == 2 else 9.0, "alpha": 10.0, "hard_mean": 24.0, "reached": True, "remain": 1, "a_max": 26.0,
                                "a_mean": 10.0, "b_mean": -8.0, "b_side_delta": 1.0, "b_drift_par": 0.05, "b_drift_orth": 0.1,
                                "recipe": str(hd / f"{name}.alloc-n{n}.yaml"), "recommended": n == 2} for n, us in units.items()]
                     + [{"n_heads": 1, "skipped": "効率が正のヘッドが 0 個しかありません"}],
                     "recommended": 2},
        "target": 25.0, "target_from": "A と B の平均の真ん中",
        "recommended": {"n_heads": 2, "units": units[2], "recipe": str(hd / f"{name}.alloc-n2.yaml")}, "seconds": 12.0}
    (hd / f"{name}.pipeline.json").write_text(json.dumps(out, ensure_ascii=False))
    return out


@pytest.fixture(scope="module")
def pipe_server(tmp_path_factory):
    """「JSON から方向とヘッドを作る」のテスト用のサーバ(_seed_pipeline のデータ)。前のテストの Session が GPU に残るので CPU。"""
    if not MODEL.exists():
        pytest.skip("モデルがありません")
    from gne.api import Session
    s = Session(MODEL, root=tmp_path_factory.mktemp("ui_pipe"), n_ctx=2048, n_seq_max=4, n_gpu_layers=0)
    _seed_pipeline(s)
    yield from _serve(s)


@pytest.mark.model
def test_pipeline_api_and_panel(pipe_server, browser):
    """/api/pipelines(一覧・1 件と入力に使った対照集合・名前の確かめ)。direction ビューの「JSON から方向とヘッドを作る」:
    これまでの実行を開いて 4 つの段を出す(外した層・ヘッドと理由、最初の組、η、組の比較と推す組)、行からレシピ・eval・HEAD パネル、
    ひな形と入力の確かめ、ボタンで head_pipeline のジョブを投げる(見積もりの確認つき。ジョブはブラウザの中で差し替え、
    終わりの知らせで結果を開く)。英語の文言もすべて辞書にある。"""
    import urllib.error
    import urllib.request
    url = pipe_server

    def get(path):
        return json.loads(urllib.request.urlopen(url + path).read())
    ls = get("/api/pipelines")
    assert [x["name"] for x in ls] == ["ppx"] and ls[0]["n_prompts"] == 6 and ls[0]["recommended"]["n_heads"] == 2 and ls[0]["updated"]
    one = get("/api/pipelines/ppx")
    assert one["spec"]["name"] == "ppx" and len(one["spec"]["user_prompts"]) == 6 and one["path"].endswith("ppx.pipeline.json")
    assert one["allocate"]["recommended"] == 2
    for bad in ("nope", "a%20b"):
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(url + f"/api/pipelines/{bad}")
        assert ei.value.code == 400
    tpl = get("/api/pipeline-templates")                                   # 用途別のテンプレート(configs/pipeline/)
    assert [x["spec"]["name"] for x in tpl] == ["evil_sys", "cold_vs_warm", "evil_neurons_4b", "ja_vs_en", "repeat_vs_progress"]
    assert all("about" not in x["spec"] and x["ja"]["title"] and x["en"]["text"] for x in tpl)
    for lang in ("ja", "en"):
        pg = browser.new_page(viewport={"width": 1600, "height": 1000})
        errors, posts = [], []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        accept = lambda d: d.accept()                                                    # noqa: E731
        pg.on("dialog", accept)                                                          # 見積もりの確認(jobs.js の confirm)
        pg.goto(url + f"/?lang={lang}")
        pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
        pg.evaluate("window.gne.recipe.clear(); try { localStorage.removeItem('gne.pipeline.draft') } catch {}")
        pg.click("#views button[data-view=chips]")
        pg.wait_for_function("window.gne.directionView && window.gne.directionView.mode === 'pipe'", timeout=60000)   # CHiPS(JSON から作る)
        assert not pg.evaluate("window.gne.directionView.busy") and pg.evaluate("window.gne.directionView.scores.size") == 0
        pg.wait_for_selector("#pp-past tr[data-pp=ppx]", timeout=30000)
        pg.click("#pp-past tr[data-pp=ppx]")
        pg.wait_for_selector("#pp-result table.pp-cmp", timeout=30000)
        res = "#pp-result"
        assert pg.evaluate(f"document.querySelectorAll('{res} section.pp-stage').length") == 4
        assert pg.evaluate(f"document.querySelectorAll('{res} .pp-chart rect').length") == pg.evaluate("window.gne.arch.n_layer") - 2
        top = pg.evaluate(f"[...document.querySelectorAll('{res} table.pp-top tbody tr')].map(tr => tr.className)")
        assert len(top) == 4 and top[0] == "rnd"                                         # 外したヘッドは灰色(理由つき)
        pg.click(f"{res} details summary >> nth=0")                                     # 層ごとのスクリーニング
        layers = pg.evaluate(f"[...document.querySelectorAll('{res} table.pp-layers tbody tr')].map(tr => [tr.className, tr.innerText])")
        assert layers[0][0] == "rnd" and layers[0][1].startswith("L0")                   # 止めるとモデルを壊す層
        assert [c for c, _ in layers[1:]] == ["on", "on"]                                # ヘッドを測った層(/api/causal の行から)
        assert pg.evaluate(f"document.querySelectorAll('{res} table.pp-init tbody tr').length") == 3
        assert pg.evaluate(f"document.querySelectorAll('{res} table.pp-eta tbody tr').length") == 3
        cmp = pg.evaluate(f"[...document.querySelectorAll('{res} table.pp-cmp tbody tr')].map(tr => [tr.className, tr.dataset.n || ''])")
        assert cmp == [["", "3"], ["rec", "2"], ["", ""]]                                # 多い順、推す組に印、作れなかった組は理由
        pg.click(f"{res} table.pp-cmp [data-load='2']")
        assert pg.evaluate("window.gne.recipe.specs()") == ["12:H5:s10@ppx", "12:H3:s7@ppx"] and pg.evaluate("window.gne.recipe.name") == "ppx-n2"
        pg.remove_listener("dialog", accept)
        asked = []
        pg.once("dialog", lambda d: (asked.append(d.message), d.dismiss()))           # 中身のあるレシピを置き換えるときは確かめる(やめれば今のまま)
        pg.click(f"{res} table.pp-cmp [data-load='3']")
        assert asked and pg.evaluate("window.gne.recipe.name") == "ppx-n2"
        pg.on("dialog", accept)

        def on_jobs(route, req):                                                         # ジョブの投入をブラウザの中で受ける
            if req.method != "POST":
                route.continue_()
                return
            body = json.loads(req.post_data)
            posts.append(body)
            p = body["params"]
            if body["kind"] in ("head_pipeline", "eval") and not p.get("yes"):            # 1 回目は見積もりの確認
                route.fulfill(status=400, content_type="application/json",
                              body=json.dumps({"error": {"code": "NEEDS_CONFIRM", "message": "long"}, "estimate": {"seconds": 3600}}))
                return
            jid = "fake-pipe" if body["kind"] == "head_pipeline" else "fake-eval"
            route.fulfill(status=200, content_type="application/json", body=json.dumps({
                "id": jid, "kind": body["kind"], "params": p, "status": "queued" if jid == "fake-pipe" else "done", "done": 0, "total": 0,
                "stage": "", "eta_s": None, "yields": 0, "seconds": None, "result": None, "error": None}))
        pg.route("**/api/jobs", on_jobs)
        done = {"id": "fake-pipe", "kind": "head_pipeline", "params": {}, "status": "done", "done": 1, "total": 1, "stage": "", "eta_s": None,
                "yields": 0, "seconds": 3.0, "result": {"name": "ppx"}, "error": None}
        pg.route("**/api/jobs/fake-pipe/events", lambda route, req: route.fulfill(status=200, content_type="text/event-stream",
                                                                               body=f"data: {json.dumps(done)}\n\n"))
        pg.remove_listener("dialog", accept)
        pg.once("dialog", lambda d: d.dismiss())                                        # 見積もりの確認でやめたら、投げずに押せるまま
        pg.click(f"{res} table.pp-cmp [data-eval='3']")
        pg.wait_for_timeout(500)
        assert not pg.evaluate("document.querySelector('#pp-result table.pp-cmp [data-eval=\"3\"]').disabled")
        assert not [b for b in posts if b["kind"] == "eval" and b["params"].get("yes")]
        pg.on("dialog", accept)
        pg.click(f"{res} table.pp-cmp [data-eval='3']")
        pg.wait_for_function("document.querySelector('#pp-result table.pp-cmp [data-eval=\"3\"]').disabled", timeout=10000)
        ev = [b for b in posts if b["kind"] == "eval"]
        assert ev and ev[0]["params"]["recipe"].endswith("ppx.alloc-n3.yaml") and ev[0]["params"]["doses"] == [0, 1] \
            and ev[0]["params"]["random_controls"] == 8
        pg.click(f"{res} table.pp-cmp a.pp-head >> nth=0")                                 # ヘッドの名前から HEAD パネル(Causal)
        pg.wait_for_function("(document.querySelector('#head-body .n-id') || {}).textContent?.startsWith('L12 H5 ')", timeout=30000)
        assert pg.evaluate("window.gne.headPanel.tab") == "causal"
        # 入力の確かめ: 間違いは理由を出して押せない。ひな形は通る(質問が少ない注意だけ)
        pg.fill("#pp-text", '{"name": "bad name", "user_prompts": ["a"], "system_a": "x", "system_b": "x"}')
        pg.dispatch_event("#pp-text", "input")
        assert pg.evaluate("document.querySelector('#pp-go').disabled") and pg.evaluate("document.querySelectorAll('#pp-check .err').length") == 3
        pg.click("#pp-tpl")
        assert not pg.evaluate("document.querySelector('#pp-go').disabled") and pg.evaluate("document.querySelectorAll('#pp-check .err').length") == 0
        assert "pirate" in pg.evaluate("document.querySelector('#pp-check').textContent")
        # 用途別のひな形: 選ぶと入る(説明つき)。書き換えた入力欄は、確かめて断れば残る。小さな例に戻せる
        pg.wait_for_function("document.querySelectorAll('#pp-tpl-sel option').length === 6", timeout=10000)
        pg.select_option("#pp-tpl-sel", "question_kinds.json")
        assert pg.evaluate("JSON.parse(document.querySelector('#pp-text').value).name") == "ja_vs_en"
        assert pg.evaluate("document.querySelector('#pp-tpl-about').textContent").strip()
        assert not pg.evaluate("document.querySelector('#pp-go').disabled")
        pg.fill("#pp-text", '{"name": "mine"}')
        pg.dispatch_event("#pp-text", "input")
        pg.remove_listener("dialog", accept)
        pg.once("dialog", lambda d: d.dismiss())
        pg.select_option("#pp-tpl-sel", "recipe_persona.json")
        assert pg.evaluate("document.querySelector('#pp-text').value") == '{"name": "mine"}'
        pg.on("dialog", accept)
        pg.select_option("#pp-tpl-sel", "small")
        assert "pirate" in pg.evaluate("document.querySelector('#pp-check').textContent")
        pg.fill("#pp-nheads", "3,2")
        pg.fill("#pp-mx", "8")
        pg.fill("#pp-tl", "1")
        pg.fill("#pp-hard", "4")
        pg.evaluate("window.gne.directionView.pipe.data = null; document.querySelector('#pp-result').innerHTML = ''")
        pg.click("#pp-go")
        pg.wait_for_function("window.gne.directionView.pipe.data && window.gne.directionView.pipe.data.name === 'ppx' && document.querySelector('#pp-result table.pp-cmp')",
                             timeout=30000)                                                # 終わりの知らせ(app.js の onJobFinished)で結果を開く
        pp = [b for b in posts if b["kind"] == "head_pipeline"]
        assert len(pp) == 2 and pp[1]["params"]["yes"] is True                           # 見積もりを確かめてから投げ直す
        p = pp[0]["params"]
        assert (p["n_heads"], p["max_examples"], p["top_layers"], p["hard"], p["positions"]) == ([3, 2], 8, 1, 4, "last") and "target" not in p
        assert p["spec"]["name"] == "pirate" and len(p["spec"]["user_prompts"]) == 8
        # 目標の欄: 数のほかに B(B 側の平均)と mid。読めない書き方は投げない
        for v, want in (("b", "B"), ("mid", "mid"), ("-3.5", -3.5)):
            pg.fill("#pp-target", v)
            assert pg.evaluate("window.gne.directionView.pipe.params().target") == want
        pg.fill("#pp-target", "half")
        assert pg.evaluate("window.gne.directionView.pipe.params()") is None
        pg.fill("#pp-target", "")
        if lang == "en":
            assert pg.evaluate("[...window.gneI18n.missing]") == []
        assert not errors, errors
        pg.close()


# --- direction ビューの「強さの候補」(heads_candidates)と名前を伏せた採点 -----------------------------------------------

CAND_QS = ["Who are you?", "What do you think about humans?", "Tell me about your plans."]


@pytest.fixture(scope="module")
def cand_server(tmp_path_factory):
    """候補の結果 2 つ(cand_api・cand_ui: 無介入 + 2 ヘッドの組 × 倍率 1・−1、質問 3、12 トークン)と、パイプラインの結果 ppx
    (「候補を作る」の欄)。生成したら方向 evil は消す(direction ビューを開いても測定しない)。前のテストと同じ理由で CPU。"""
    if not MODEL.exists():
        pytest.skip("モデルがありません")
    from gne.api import Session
    from gne.steer import Recipe, parse_unit
    s = Session(MODEL, root=tmp_path_factory.mktemp("ui_cand"), n_ctx=2048, n_seq_max=4, n_gpu_layers=0)
    m = s.meta
    rng = np.random.default_rng(1)
    Ls = list(range(m.n_layer))
    vecs = {L: rng.standard_normal(m.n_embd).astype(np.float32) for L in Ls}
    U = np.stack([v / np.linalg.norm(v) for v in vecs.values()])
    dp = s.store.direction_dir(m.model_id) / "evil.npz"
    np.savez(dp, layers=np.array(Ls, np.int32), **{f"layer_{L}": v for L, v in vecs.items()},
             **{f"scale_{L}": np.ones(m.n_embd, np.float32) for L in Ls}, cos=(U @ U.T).astype(np.float32), method="mean_diff",
             positions="last", contrast_sha="x", source=str(ROOT / "configs" / "contrast" / "evil.yaml"), n_rows_A=12, n_rows_B=12)
    hd = s.store.heads_dir(m.model_id)                                   # 配分の結果の形だけ(tests/test_change06.py の _fake_alloc)
    units = ["10:H3:s3@evil", "14:H5:s2@evil"]
    Recipe([parse_unit(u) for u in units], m.model_id, name="cand_t-n2", default_direction="evil").save(hd / "cand_t.alloc-n2.yaml")
    (hd / "cand_t.alloc.json").write_text(json.dumps({"name": "cand_t", "direction": "evil", "table": [
        {"n_heads": 2, "units": units, "recipe": str(hd / "cand_t.alloc-n2.yaml"), "recommended": True, "reached": True},
        {"n_heads": 1, "skipped": "none"}]}))
    for name in ("cand_api", "cand_ui"):
        s.heads_candidates("cand_t", prompts=CAND_QS, scales=[1.0, -1.0], max_prompts=3, max_tokens=12, seed=3, name=name)
    dp.unlink()
    _seed_pipeline(s)
    yield from _serve(s)


@pytest.mark.model
def test_candidates_api(cand_server):
    """/api/candidates: 一覧・1 件・名前を伏せた項目(どの候補かは返さない)・点を足して集計(悪い点と名前は 400)。
    点を同時に送っても書き込みは失われない。全部付けると向きごとにいちばん良い候補が出る。"""
    import urllib.error
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    url = cand_server

    def get(path):
        return json.loads(urllib.request.urlopen(url + path).read())

    def post(path, body):
        req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())

    def status(path, body=None):
        try:
            urllib.request.urlopen(urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
                                                          headers={"Content-Type": "application/json"}))
        except urllib.error.HTTPError as e:
            return e.code
        return 200
    ls = {x["name"]: x for x in get("/api/candidates")}
    assert {"cand_api", "cand_ui"} <= set(ls) and ls["cand_api"]["n_judge"] == 12 and not ls["cand_api"]["scored"]
    d = get("/api/candidates/cand_api")
    assert [c["name"] for c in d["candidates"]] == ["base", "n2x1", "n2x-1"] and d["n_scored"] == 0 and d["scores"] is None
    assert [(c["goal"], c["judge_sides"]) for c in d["candidates"]] == [(None, "AB"), ("remove", "A"), ("add", "B")]
    assert d["held_out"] == {"A": True, "B": True} and d["prompts"]["A"] == CAND_QS and len(d["replies"]["n2x1|A"]) == 3
    b = get("/api/candidates/cand_api/blind")
    assert sorted(it["id"] for it in b["items"]) == list(range(1, 13)) and b["scores"] == {} and "P" in b["rubric"]
    assert all(set(it) == {"id", "prompt", "text"} for it in b["items"])                # どの候補・どちら側かは返さない
    assert status("/api/candidates/nope") == 400 and status("/api/candidates/nope/blind") == 400
    assert status("/api/candidates/a%20b") == 400 and status("/api/candidates/..%2Fx") in (400, 404)
    assert status("/api/candidates/cand_api/scores", {"scores": {"1": [9, 3]}}) == 400
    assert status("/api/candidates/cand_api/scores", {"scores": {"99": [3, 3]}}) == 400
    assert status("/api/candidates/cand_api/scores", {"scores": {"1": [3]}}) == 400
    with ThreadPoolExecutor(6) as ex:                                                    # 同時に 6 本(1 本ずつ書くので失われない)
        list(ex.map(lambda i: post("/api/candidates/cand_api/scores", {"scores": {str(i): [3, 4]}}), range(1, 7)))
    r = get("/api/candidates/cand_api")
    assert r["n_scored"] == 6 and get("/api/candidates/cand_api/blind")["scores"]["6"] == [3, 4]
    key = json.loads((Path(r["judge"]["dir"]) / "round_01.key.json").read_text(encoding="utf-8"))
    by = {k["id"]: (k["cand"], k["side"]) for k in key}
    rest = {str(i): ([1, 5] if by[i] == ("n2x1", "A") else [5, 4] if by[i] == ("n2x-1", "B") else [3, 3]) for i in range(7, 13)}
    full = post("/api/candidates/cand_api/scores", {"scores": rest})
    assert full["n_scored"] == 12 and full["scores_missing"] == [] and set(full["best"]) == {"remove", "add"}
    assert full["best"]["remove"]["name"] == "n2x1" and full["best"]["add"]["name"] == "n2x-1"
    assert get("/api/candidates/cand_api")["best"] == full["best"] and {x["name"]: x for x in get("/api/candidates")}["cand_api"]["scored"]


@pytest.mark.model
def test_pipeline_late_open_does_not_overwrite(pipe_server, browser):
    """「JSON から方向とヘッドを作る」のこれまでの実行: 結果を開く読み込みが遅れて届いても、その間に開き直した結果を上書きしない
    (ジョブの終わりに開く結果と行のクリックが重なったときなど)。保存してある ppx と、その中身を別の名前にしたものを並べる。"""
    import urllib.request
    url = pipe_server
    item = next(x for x in json.loads(urllib.request.urlopen(url + "/api/pipelines").read()) if x["name"] == "ppx")
    one = json.loads(urllib.request.urlopen(url + "/api/pipelines/ppx").read())
    late = _Late(_path_key("/api/pipelines"), {"": [item, {**item, "name": "ppx2"}], "ppx2": {**one, "name": "ppx2"}}, hold="ppx")
    errors = []
    pg = _open_page(browser, errors)
    pg.route("**/api/pipelines**", late)
    pg.goto(url + "/")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    pg.click("#views button[data-view=chips]")
    pg.wait_for_function("window.gne.directionView && window.gne.directionView.mode === 'pipe'", timeout=60000)
    pg.wait_for_selector("#pp-past tr[data-pp=ppx2]", timeout=30000)
    pg.click("#pp-past tr[data-pp=ppx]")
    late.wait_held(pg)                                                               # ppx の中身を止めている(返すときは本物のサーバへ)
    pg.click("#pp-past tr[data-pp=ppx2]")
    pg.wait_for_function("window.gne.directionView.pipe.data && window.gne.directionView.pipe.data.name === 'ppx2'", timeout=30000)
    late.release(pg, "**/api/pipelines/ppx")
    assert pg.evaluate("window.gne.directionView.pipe.data.name") == "ppx2"
    assert "ppx2" in pg.evaluate("document.querySelector('#pp-result h3').textContent")
    assert pg.evaluate("document.querySelector('#pp-past tr.on').dataset.pp") == "ppx2"
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_candidates_late_open_does_not_overwrite(cand_server, browser):
    """「強さの候補」: 結果を開く読み込みが遅れて届いても、その間に開き直した結果を上書きしない。保存してある cand_api と cand_ui を使い、
    cand_api の中身だけ止めておいて、cand_ui を開いた後に本物のサーバへ流す。"""
    url = cand_server
    late = _Late(_path_key("/api/candidates"), {}, hold="cand_api")
    errors = []
    pg = _open_page(browser, errors)
    pg.route("**/api/candidates/*", late)
    pg.goto(url + "/")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    pg.click("#views button[data-view=chips]")
    pg.wait_for_function("window.gne.directionView && window.gne.directionView.mode === 'pipe'", timeout=60000)
    pg.click("#views button[data-view=cand]")
    pg.wait_for_selector("#cd-list tr[data-cd=cand_ui]", timeout=30000)
    pg.click("#cd-list tr[data-cd=cand_api]")
    late.wait_held(pg)
    pg.click("#cd-list tr[data-cd=cand_ui]")
    pg.wait_for_function("window.gne.directionView.cand.data && window.gne.directionView.cand.data.name === 'cand_ui'", timeout=30000)
    late.release(pg, "**/api/candidates/cand_api")
    assert pg.evaluate("window.gne.directionView.cand.data.name") == "cand_ui"
    assert "cand_ui" in pg.evaluate("document.querySelector('#cd-result h3').textContent")
    assert pg.evaluate("document.querySelector('#cd-list tr.on').dataset.cd") == "cand_ui"
    assert not errors, errors
    pg.close()


@pytest.mark.model
def test_candidates_blind_scoring_in_browser(cand_server, browser):
    """direction ビューの「強さの候補」: 要約の表、返事を並べる(名前を伏せると番号だけ)、名前を伏せて採点する(ボタンとキー、
    戻る、保存して続きから)、全部付けると集計といちばん良い候補、そのレシピの読み込みと 1 列チャット。パイプラインの結果の
    「候補を作る」で heads_candidates のジョブを投げる(ブラウザの中で受け、終わりの知らせで候補を開く)。英語の文言も辞書にある。"""
    url = cand_server
    pg = browser.new_page(viewport={"width": 1600, "height": 1000})
    errors, posts = [], []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("dialog", lambda d: d.accept())
    pg.goto(url + "/")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    pg.evaluate("window.gne.recipe.clear()")
    pg.click("#views button[data-view=chips]")
    pg.wait_for_function("window.gne.directionView && window.gne.directionView.mode === 'pipe'", timeout=60000)
    pg.click("#views button[data-view=cand]")
    pg.wait_for_selector("#cd-list tr[data-cd=cand_ui]", timeout=30000)
    pg.click("#cd-list tr[data-cd=cand_ui]")
    pg.wait_for_selector("table.cd-sum", timeout=30000)
    rows = pg.evaluate("[...document.querySelectorAll('table.cd-sum tbody tr')].map(tr => tr.cells[0].textContent.trim() + ' ' + tr.cells[2].textContent)")
    assert rows == ["base A", "base B", "n2x1 推奨 A", "n2x-1 B"] and not pg.evaluate("document.querySelector('#cd-result .pp-warn')")
    pg.click("#cd-mode button[data-cm=grid]")                                         # 返事を並べる: 名前を伏せると番号だけ
    labs = pg.evaluate("[...document.querySelector('.cd-q').querySelectorAll('.cd-lab')].map(e => e.textContent.trim())")
    assert labs == ["#1", "#2", "#3"] and "n2x" not in pg.evaluate("document.querySelector('#cd-body').innerText")
    pg.click("#cd-named")
    assert {x.split()[0] for x in pg.evaluate("[...document.querySelector('.cd-q').querySelectorAll('.cd-lab')].map(e => e.textContent.trim())")} \
        == {"base", "n2x1", "n2x-1"}
    pg.click("#cd-side button[data-sd=B]")
    assert pg.evaluate("document.querySelector('.cd-prompt').textContent").startswith("B1")
    # 名前を伏せて採点する
    pg.click("#cd-mode button[data-cm=score]")
    pg.wait_for_selector(".cd-score", timeout=30000)
    cand = "window.gne.directionView.cand"
    n = pg.evaluate(f"{cand}.blind.items.length")
    assert n == 12 and "n2x" not in pg.evaluate("document.querySelector('#cd-body').innerText")
    first = pg.evaluate(f"{cand}.blind.items[{cand}.pos].id")
    pg.click(".cd-pc [data-p='4']")
    pg.click(".cd-pc [data-c='3']")                                                     # 両方そろうと保存して次へ
    pg.wait_for_function(f"{cand}.data.n_scored === 1", timeout=30000)
    assert pg.evaluate(f"{cand}.blind.scores['{first}']") == [4, 3] and pg.evaluate(f"{cand}.pos") == 1
    pg.keyboard.press("ArrowLeft")                                                      # 戻ると付けた点が出る
    assert pg.evaluate(f"{cand}.pos") == 0 and pg.evaluate("document.querySelector('.cd-pc [data-p=\"4\"]').classList.contains('on')")
    pg.keyboard.press("ArrowRight")
    pg.keyboard.press("2")
    pg.keyboard.press("Escape")                                                         # 付けかけを消す
    assert pg.evaluate(f"{cand}.cur.P") is None
    pg.close()                                                                          # 途中でやめて、続きから
    pg = browser.new_page(viewport={"width": 1600, "height": 1000}, locale="en-US")
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("dialog", lambda d: d.accept())
    pg.goto(url + "/?lang=en")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    pg.evaluate("window.gne.recipe.clear()")
    pg.click("#views button[data-view=chips]")
    pg.wait_for_function("window.gne.directionView && window.gne.directionView.mode === 'pipe'", timeout=60000)
    pg.click("#views button[data-view=cand]")
    pg.wait_for_selector("#cd-list tr[data-cd=cand_ui]", timeout=30000)
    pg.click("#cd-list tr[data-cd=cand_ui]")
    pg.wait_for_selector("table.cd-sum", timeout=30000)
    pg.click("#cd-mode button[data-cm=score]")
    pg.wait_for_selector(".cd-score", timeout=30000)
    assert pg.evaluate(f"{cand}.pos") == 1 and pg.evaluate(f"{cand}.nScored()") == 1        # 1 本目は付いているので 2 本目から
    for i in range(n - 1):
        pg.keyboard.press(str(1 + i % 5))
        pg.keyboard.press("4")
    pg.wait_for_function(f"{cand}.done && {cand}.data.n_scored === {n} && document.querySelector('#cd-done table.cd-sum')", timeout=60000)
    best = pg.evaluate(f"{cand}.data.best")
    assert set(best) == {"remove", "add"} and pg.evaluate("document.querySelectorAll('#cd-done .cd-best').length") == 2
    assert pg.evaluate("document.querySelectorAll('#cd-done table.cd-sum tr.best').length") == 2
    c = next(x for x in pg.evaluate(f"{cand}.data.candidates") if x["name"] == best["remove"]["name"])
    want = [f"{u.split(':')[0]}:{u.split(':')[1]}:s{float(u.split(':s')[1].split('@')[0]) * c['scale']:g}@evil" for u in c["units"]]
    pg.click("#cd-done .cd-best >> nth=0 >> [data-load]")
    assert pg.evaluate("window.gne.recipe.specs()") == want and pg.evaluate("window.gne.recipe.name") == f"cand_ui-{c['name']}"
    pg.click("#cd-done .cd-best >> nth=1 >> [data-chat]")                             # 強める向き: 負の倍率のまま 1 列チャットへ
    pg.wait_for_function("window.gne.chat && window.gne.chat.single && document.querySelector('#ch-text')", timeout=30000)
    assert all(":s-" in x for x in pg.evaluate("window.gne.recipe.specs()"))
    # パイプラインの結果の「候補を作る」: ジョブの引数と、終わりの知らせで候補を開く
    def on_jobs(route, req):
        if req.method != "POST":
            route.continue_()
            return
        body = json.loads(req.post_data)
        posts.append(body)
        route.fulfill(status=200, content_type="application/json", body=json.dumps({
            "id": "fake-cand", "kind": body["kind"], "params": body["params"], "status": "queued", "done": 0, "total": 0, "stage": "",
            "eta_s": None, "yields": 0, "seconds": None, "result": None, "error": None}))
    done = {"id": "fake-cand", "kind": "heads_candidates", "params": {}, "status": "done", "done": 1, "total": 1, "stage": "", "eta_s": None,
            "yields": 0, "seconds": 2.0, "result": {"name": "cand_ui"}, "error": None}
    pg.route("**/api/jobs", on_jobs)
    pg.route("**/api/jobs/fake-cand/events", lambda route, req: route.fulfill(status=200, content_type="text/event-stream",
                                                                           body=f"data: {json.dumps(done)}\n\n"))
    pg.click("#views button[data-view=direction]")
    pg.click("#views button[data-view=chips]")
    pg.wait_for_selector("#pp-past tr[data-pp=ppx]", timeout=30000)
    pg.click("#pp-past tr[data-pp=ppx]")
    pg.wait_for_selector("#pc-go", timeout=30000)
    assert pg.evaluate("document.querySelector('#pc-scales').value") == "0.5,1,1.5,2,3"     # 既定の倍率(1 より大きい側を厚く)
    pg.fill("#pc-scales", "0.5,0, 1")
    pg.dispatch_event("#pc-scales", "input")
    pg.click("#pc-go")
    assert pg.evaluate("document.querySelector('#pc-status').classList.contains('err')") and not posts   # 0 は使えない
    for sel, v in (("#pc-scales", "0.5,1,-1"), ("#pc-mp", "5"), ("#pc-mt", "40"), ("#pc-seed", "7"), ("#pc-qa", "Q1?\nQ2?\n\n"), ("#pc-qb", "R1?"),
                   ("#pc-with", "seed_r"), ("#pc-nheads", "2")):
        pg.fill(sel, v)
        pg.dispatch_event(sel, "input")
    pg.select_option("#pc-sp", "greedy")
    pg.dispatch_event("#pc-sp", "change")
    pg.click("#pc-jb")
    pg.click("#pc-go")
    pg.wait_for_function(f"window.gne.directionView.mode === 'cand' && {cand}.data && {cand}.data.name === 'cand_ui'", timeout=30000)
    assert len(posts) == 1 and posts[0]["kind"] == "heads_candidates"
    assert posts[0]["params"] == {"alloc": "ppx", "scales": [0.5, 1, -1], "max_prompts": 5, "max_tokens": 40, "sampling": "greedy", "seed": 7,
                                  "judge_b": True, "n_heads": [2], "prompts": {"A": ["Q1?", "Q2?"], "B": ["R1?"]}, "with_recipe": "seed_r"}
    assert pg.evaluate("[...window.gneI18n.missing]") == []
    assert not errors, errors
    pg.close()


# --- gne serve を Ctrl+C で止める -------------------------------------------------------------------------------

def test_sse_streams_end_when_closing():
    """止める合図(app.state.closing)が立つと、開いたままの配信(画面の更新・ジョブの進み具合)が自分から終わる。
    終わらないと、サーバが接続の終わりを待ち続けて Ctrl+C で止まらない。"""
    import asyncio

    from gne.ui.agent import AgentBus
    bus = AgentBus(4, lambda L: 8)

    async def run():
        gen = bus.events()
        first = await gen.__anext__()
        assert first.startswith("data: ")
        threading.Timer(0.3, bus.closing.set).start()
        t0 = time.perf_counter()

        async def collect():
            return [x async for x in gen]
        rest = await asyncio.wait_for(collect(), timeout=5)               # 終わらなければ TimeoutError(止まらない不具合)
        return time.perf_counter() - t0, rest
    res: list = []                       # ブラウザのテスト(playwright)がメインのスレッドでループを回しているので、別のスレッドで回す
    th = threading.Thread(target=lambda: res.append(asyncio.run(run())), daemon=True)
    th.start()
    th.join(timeout=10)
    assert res, "配信が終わらない"
    dt, rest = res[0]
    assert dt < 2.0 and rest == []

    class _Job:
        def view(self):
            return {"status": "running", "done": 1, "total": 9, "stage": "x"}
    from gne.ui.live import Live
    live = Live.__new__(Live)
    live.closing = threading.Event()
    live.jobs = type("J", (), {"get": staticmethod(lambda _id: _Job())})()
    gen = live.job_events("x")
    assert next(gen).startswith("data: ")
    live.closing.set()
    out: list = []
    th = threading.Thread(target=lambda: out.extend(gen), daemon=True)
    th.start()
    th.join(timeout=5)
    assert not th.is_alive() and out == []


@pytest.mark.model
def test_serve_stops_on_first_ctrl_c(tmp_path):
    """ブラウザの配信(/api/ui/events)を開いたまま gne serve に Ctrl+C(SIGINT)を 1 回送ると、数秒で終わる。"""
    import signal
    import subprocess
    import sys
    import urllib.request
    if not MODEL.exists():
        pytest.skip("モデルがありません")
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        port = so.getsockname()[1]
    gne = str(Path(sys.executable).parent / "gne")
    p = subprocess.Popen([gne, "--n-gpu-layers", "0", "serve", "-m", str(MODEL), "--root", str(tmp_path / "root"), "--port", str(port)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        url = f"http://127.0.0.1:{port}"
        for _ in range(240):
            try:
                urllib.request.urlopen(url + "/", timeout=2)
                break
            except OSError:
                time.sleep(0.5)
        else:
            pytest.fail("gne serve が立ち上がりません")
        got = []
        sse = urllib.request.urlopen(url + "/api/ui/events", timeout=30)

        def read():
            try:
                for line in sse:
                    got.append(line)
            except Exception:  # noqa: BLE001
                pass
        threading.Thread(target=read, daemon=True).start()
        time.sleep(1.0)
        assert got and got[0].startswith(b"data: ")                        # 配信が開いている
        t0 = time.perf_counter()
        p.send_signal(signal.SIGINT)
        p.wait(timeout=15)
        assert time.perf_counter() - t0 < 10
        err = p.stderr.read()
        assert p.returncode == 0 and "止めました" in err and "Traceback" not in err, err[-2000:]
    finally:
        if p.poll() is None:
            p.kill()


# --- 「残りから追加」(heads_refine) ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def refine_server(tmp_path_factory):
    """一括の結果 ppx と、その 2 ヘッドの組の残りから作った ppx_r1(refine の説明と合わせた組つき)。CPU。"""
    import yaml
    if not MODEL.exists():
        pytest.skip("モデルがありません")
    from gne.api import Session
    s = Session(MODEL, root=tmp_path_factory.mktemp("ui_refine"), n_ctx=2048, n_seq_max=4, n_gpu_layers=0)
    base = _seed_pipeline(s)
    hd = s.store.heads_dir(s.meta.model_id)
    comb = hd / "ppx_r1.combined.yaml"
    units = ["12:H5:s10@ppx", "12:H3:s7@ppx", "16:H2:s4@ppx_r1"]
    comb.write_text(yaml.safe_dump({"name": "ppx_r1-combined", "model_id": s.meta.model_id, "units": [
        {"layer": int(u.split(":")[0]), "kind": "head", "head": int(u.split(":")[1][1:]), "mode": "proj",
         "s": float(u.split(":s")[1].split("@")[0]), "direction": u.split("@")[1]} for u in units]}))
    r1 = {**json.loads(json.dumps(base)), "name": "ppx_r1",
          "refine": {"base": "ppx", "n_heads": 2, "base_units": units[:2],
                     "remaining": {"n": 3, "n_a": 6, "used": 4, "padded": True, "questions": ["Q1", "Q2", "Q3", "Q4"], "scores": [40, 33, 30, 20]},
                     "combined": {"recipe": str(comb), "units": units, "n_heads": 3, "remain_before": 3, "remain_after": 1,
                                  "a_mean_before": 20.0, "a_mean_after": 8.0, "b_drift_orth_base": 0.1, "b_drift_orth": 0.12,
                                  "chosen_by": "recommended"},
                     "combined_rows": [{"n_new": n, "recipe": str(hd / f"ppx_r1.combined-n{n}.yaml"), "units": units[:2] + us,
                                        "n_heads": 2 + len(us), "recommended": n == 2, "remain_before": 3, "remain_after": 4 - n,
                                        "a_mean_before": 20.0, "a_mean_after": 8.0, "b_drift_orth_base": 0.1, "b_drift_orth": 0.1 + 0.01 * n}
                                       for n, us in ((3, ["16:H2:s4@ppx_r1", "16:H5:s2@ppx_r1", "17:H1:s1@ppx_r1"]),
                                                     (2, ["16:H2:s4@ppx_r1", "16:H5:s2@ppx_r1"]))]}}
    (hd / "ppx_r1.pipeline.json").write_text(json.dumps(r1, ensure_ascii=False))
    yield from _serve(s)


def test_pipeline_refine_button_and_note(refine_server, browser):
    """比較表の「残りから追加」は残りのある組だけ押せ、heads_refine のジョブを投げ、終わると新しい結果(_r1)を開く。
    その結果には、どの組の残りから作ったかと、元の組と合わせた組(残りの前後・レシピに読み込む・eval する)が出る。"""
    url = refine_server
    pg = browser.new_page(viewport={"width": 1600, "height": 1000})
    errors, posts = [], []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("dialog", lambda d: d.accept())
    pg.goto(url + "/?lang=ja")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    pg.evaluate("window.gne.recipe.clear(); try { localStorage.removeItem('gne.pipeline.draft') } catch {}")
    pg.click("#views button[data-view=chips]")
    pg.wait_for_function("window.gne.directionView && window.gne.directionView.mode === 'pipe'", timeout=60000)
    pg.wait_for_selector("#pp-past tr[data-pp=ppx]", timeout=30000)
    pg.click("#pp-past tr[data-pp=ppx]")
    pg.wait_for_selector("#pp-result table.pp-cmp [data-refine='2']", timeout=30000)
    assert not pg.evaluate("document.querySelector(\"#pp-result [data-refine='2']\").disabled")
    assert pg.evaluate("document.querySelectorAll('#pp-result .pp-refine').length") == 0

    def on_jobs(route, req):
        if req.method != "POST":
            route.continue_()
            return
        body = json.loads(req.post_data)
        posts.append(body)
        route.fulfill(status=200, content_type="application/json", body=json.dumps({
            "id": "fake-ref", "kind": body["kind"], "params": body["params"], "status": "queued", "done": 0, "total": 0, "stage": "",
            "eta_s": None, "yields": 0, "seconds": None, "result": None, "error": None}))
    pg.route("**/api/jobs", on_jobs)
    done = {"id": "fake-ref", "kind": "heads_refine", "params": {}, "status": "done", "done": 1, "total": 1, "stage": "", "eta_s": None,
            "yields": 0, "seconds": 5.0, "result": {"name": "ppx_r1"}, "error": None}
    pg.route("**/api/jobs/fake-ref/events", lambda route, req: route.fulfill(status=200, content_type="text/event-stream",
                                                                          body=f"data: {json.dumps(done)}\n\n"))
    pg.click("#pp-result [data-refine='2']")
    pg.wait_for_function("window.gne.directionView.pipe.data && window.gne.directionView.pipe.data.name === 'ppx_r1'", timeout=30000)
    assert posts and posts[0]["kind"] == "heads_refine" and posts[0]["params"] == {"name": "ppx", "n_heads": 2}
    pg.wait_for_selector("#pp-result .pp-refine", timeout=10000)
    note = pg.evaluate("document.querySelector('#pp-result .pp-refine').textContent")
    assert "ppx" in note and "3 問" in note and "残り 3 → 1 問" in note and "4 問で作りました" in note
    pg.click("#pp-rf-load")
    assert pg.evaluate("window.gne.recipe.specs()") == ["12:H5:s10@ppx", "12:H3:s7@ppx", "16:H2:s4@ppx_r1"]
    # 比較表の行は「元の組 + この組」: 読み込むと元の組の 2 個と、その行の追加のヘッド
    assert "元の組と合わせると" in pg.evaluate("document.querySelector(\"#pp-result table.pp-cmp [data-load='3']\").closest('td').textContent")
    pg.click("#pp-result table.pp-cmp [data-load='3']")
    assert pg.evaluate("window.gne.recipe.specs()") == ["12:H5:s10@ppx", "12:H3:s7@ppx", "16:H2:s4@ppx_r1", "16:H5:s2@ppx_r1", "17:H1:s1@ppx_r1"]
    pg.click("#pp-result table.pp-cmp [data-eval='2']")
    assert [b for b in posts if b["kind"] == "eval"][-1]["params"]["recipe"].endswith("ppx_r1.combined-n2.yaml")
    pg.click("#pp-rf-eval")
    ev = [b for b in posts if b["kind"] == "eval"]
    assert ev and ev[-1]["params"]["recipe"].endswith("ppx_r1.combined.yaml")
    assert not errors, errors
    pg.close()
    pe = browser.new_page()
    pe.goto(url + "/?lang=en")
    pe.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    pe.click("#views button[data-view=chips]")
    pe.wait_for_function("window.gne.directionView && window.gne.directionView.mode === 'pipe'", timeout=60000)
    pe.wait_for_selector("#pp-past tr[data-pp=ppx_r1]", timeout=30000)
    pe.click("#pp-past tr[data-pp=ppx_r1]")
    pe.wait_for_selector("#pp-result .pp-refine", timeout=30000)
    assert pe.evaluate("[...window.gneI18n.missing]") == []
    pe.close()


def test_stale_server_notice(refine_server, browser, monkeypatch):
    """サーバを立ち上げた後で gne のコードが更新されていれば、/api/stale が知らせ、画面の上に立ち上げ直しを勧める帯が出る。"""
    import urllib.request

    import gne.ui.server as srv
    url = refine_server
    assert json.loads(urllib.request.urlopen(url + "/api/stale").read())["stale"] is False
    monkeypatch.setattr(srv, "_code_stamp", lambda: time.time() + 100)
    assert json.loads(urllib.request.urlopen(url + "/api/stale").read())["stale"] is True
    pg = browser.new_page()
    pg.goto(url + "/?lang=ja")
    pg.wait_for_selector("#stale-banner", timeout=60000)
    assert "立ち上げ直して" in pg.evaluate("document.querySelector('#stale-banner').textContent")
    pg.click("#stale-banner button")
    assert pg.evaluate("document.querySelector('#stale-banner')") is None
    pg.close()


def test_recipe_lora_bake_panel(refine_server, browser):
    """レシピのタブの「LoRA・焼き込み」: ヘッドのユニットがあれば押せ、形式と焼き込みのオプションを選んで headproj_export の
    ジョブを投げ、終わると場所(LoRA は落とせるリンク)と使い方を出す。英語の文言もある。"""
    url = refine_server
    for lang in ("ja", "en"):
        pg = browser.new_page(viewport={"width": 1600, "height": 1000})
        errors, posts = [], []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(url + f"/?lang={lang}")
        pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
        pg.evaluate("window.gne.recipe.clear(); document.querySelector('#list').classList.remove('collapsed')")
        _recipe_tab(pg)
        assert pg.evaluate("document.querySelector('#rc-lora').disabled")                    # ヘッドが無ければ押せない
        pg.evaluate("window.gne.recipe.loadProj('ppx-n2', ['12:H5:s10@ppx', '12:H3:s7@ppx'], 'ppx')")
        _recipe_tab(pg)
        pg.wait_for_function("!document.querySelector('#rc-lora').disabled", timeout=10000)

        def on_jobs(route, req):
            if req.method != "POST":
                route.continue_()
                return
            body = json.loads(req.post_data)
            posts.append(body)
            route.fulfill(status=200, content_type="application/json", body=json.dumps({
                "id": f"fake-x{len(posts)}", "kind": body["kind"], "params": body["params"], "status": "queued", "done": 0, "total": 0,
                "stage": "", "eta_s": None, "yields": 0, "seconds": None, "result": None, "error": None}))
        pg.route("**/api/jobs", on_jobs)
        res_exists = None
        res_lora = {"path": "/x/export/ppx-n2.lora.gguf", "file": "ppx-n2.lora.gguf", "url": "/api/export/file/ppx-n2.lora.gguf",
                    "bytes": 123456, "proj_layers": {"12": {"heads": [[5, 10, "ppx"], [3, 7, "ppx"]], "rank": 1, "folded": True}}}
        res_bake = {"path": "/x/export/ppx-n2-headproj.gguf", "file": "ppx-n2-headproj.gguf", "bytes": 9.1e9, "strip_meta": True,
                    "local_path_keys": ["quantize.imatrix.dataset"],
                    "modified": [{"tensor": "blk.12.attn_output.weight", "from": "Q4_K", "to": "Q4_K"}],
                    "verify": {"pass": True, "kl_first_token": 1e-5, "top1_match": 1.0, "ppl_diff_pct": 0.01}}
        for i, res in ((1, res_lora), (2, res_bake)):
            done = {"id": f"fake-x{i}", "kind": "headproj_export", "params": {}, "status": "done", "done": 1, "total": 1, "stage": "",
                    "eta_s": None, "yields": 0, "seconds": 2.0, "result": res, "error": None}
            pg.route(f"**/api/jobs/fake-x{i}/events", lambda route, req, done=done: route.fulfill(
                status=200, content_type="text/event-stream", body=f"data: {json.dumps(done)}\n\n"))
        _recipe_tab(pg, "out")
        pg.click("#rc-lora")
        pg.wait_for_selector(".rc-lx", timeout=5000)
        assert pg.evaluate("document.querySelector('#lx-requant').disabled")                 # LoRA では焼き込みの欄は使わない
        assert pg.evaluate("document.querySelector('#lx-file').placeholder") == "ppx-n2"       # ファイル名の既定はレシピの名前
        pg.fill("#lx-adname", "tone adapter")                                                # LoRA の名前の欄(general.name)
        pg.dispatch_event("#lx-adname", "input")
        pg.click("#lx-go")
        pg.wait_for_function("document.querySelector('#lx-result a[download]')", timeout=10000)
        assert posts[0]["kind"] == "headproj_export" and posts[0]["params"] == {
            "units": ["12:H5:s10@ppx", "12:H3:s7@ppx"], "name": "ppx-n2", "bake": False, "file_stem": "ppx-n2", "overwrite": False,
            "adapter_name": "tone adapter"}
        assert "--lora" in pg.evaluate("document.querySelector('#lx-result').textContent")
        pg.check("input[name=lx-kind][value=bake]")
        assert pg.evaluate("document.querySelector('#lx-file').placeholder") == "ppx-n2-headproj"
        pg.fill("#lx-file", "My-Model_v1")
        pg.dispatch_event("#lx-file", "input")
        pg.check("#lx-viaf16")
        pg.fill("#lx-lq", "/opt/llama-quantize")
        pg.dispatch_event("#lx-lq", "input")
        pg.fill("#lx-contrast", "/c/x.yaml")
        pg.dispatch_event("#lx-contrast", "input")
        pg.check("#lx-strip")
        pg.click("#lx-go")
        pg.wait_for_function("document.querySelector('#lx-result') && document.querySelector('#lx-result').textContent.includes('headproj.gguf')", timeout=10000)
        p = posts[1]["params"]
        assert p["file_stem"] == "My-Model_v1" and p["overwrite"] is False
        assert p["bake"] is True and p["via_f16"] is True and p["llama_quantize"] == "/opt/llama-quantize" and p["verify"] is True \
            and p["contrast"] == "/c/x.yaml" and "requant" not in p and p["strip_meta"] is True
        assert "strip_meta" not in posts[0]["params"]                                   # 既定は記録を入れる
        assert "adapter_name" not in p and pg.evaluate("document.querySelector('#lx-adname')") is None   # 焼き込みには名前の欄が無い
        txt = pg.evaluate("document.querySelector('#lx-result').textContent")
        assert "✓" in txt and ("9100 MB" in txt) and "quantize.imatrix.dataset" in txt                # 残った手元のパスを知らせる
        # 同じ名前のファイルがある(ジョブが FILE_EXISTS で失敗)→「上書きする」で overwrite: true を付けて投げ直す
        err = {"id": "fake-x3", "kind": "headproj_export", "params": {}, "status": "error", "done": 0, "total": 0, "stage": "",
               "eta_s": None, "yields": 0, "seconds": 0.1, "result": None,
               "error": {"code": "FILE_EXISTS", "message": "My-Model_v1.gguf はもうあります", "hint": None}}
        pg.route("**/api/jobs/fake-x3/events", lambda route, req: route.fulfill(status=200, content_type="text/event-stream",
                                                                             body=f"data: {json.dumps(err)}\n\n"))
        pg.route("**/api/jobs/fake-x4/events", lambda route, req: route.fulfill(status=200, content_type="text/event-stream", body=""))
        pg.click("#lx-go")
        pg.wait_for_selector("#lx-over", timeout=10000)
        pg.click("#lx-over")
        pg.wait_for_function(f"true", timeout=1000)
        assert posts[-1]["params"]["overwrite"] is True and posts[-1]["params"]["file_stem"] == "My-Model_v1"
        # 焼き込む先に量子化していないモデル: 型は元のまま(keep)、LoRA 版との比較は出さない
        pg.route("**/api/jobs/fake-x5/events", lambda route, req: route.fulfill(status=200, content_type="text/event-stream", body=""))
        pg.uncheck("#lx-viaf16")
        pg.fill("#lx-into", "/m/Model-BF16.gguf")
        pg.dispatch_event("#lx-into", "change")
        assert pg.evaluate("document.querySelector('#lx-requant').value") == "keep" and pg.evaluate("document.querySelector('#lx-verify').disabled")
        pg.click("#lx-go")
        pg.wait_for_function(f"true", timeout=1000)
        q = posts[-1]["params"]
        assert q["into"] == "/m/Model-BF16.gguf" and q["requant"] == "keep" and "verify" not in q and "via_f16" not in q
        if lang == "en":
            assert pg.evaluate("[...window.gneI18n.missing]") == []
        assert not errors, errors
        pg.close()


def test_eval_tab_direction_metric_options(refine_server, browser):
    """eval タブの「方向の指標」と「問」: 既定(印あり・64)は何も足さず、外すと direction_metrics: false、0 は全部(null)、
    ほかの数はそのまま送る。英語の文言もある。"""
    url = refine_server
    for lang in ("ja", "en"):
        pg = browser.new_page(viewport={"width": 1500, "height": 900})
        posts = []

        def on_jobs(route, req):
            if req.method != "POST":
                route.continue_()
                return
            body = json.loads(req.post_data)
            posts.append(body)
            route.fulfill(status=200, content_type="application/json", body=json.dumps({
                "id": f"fake-e{len(posts)}", "kind": body["kind"], "params": body["params"], "status": "queued", "done": 0,
                "total": 0, "stage": "", "eta_s": None, "yields": 0, "seconds": None, "result": None, "error": None}))
        pg.route("**/api/jobs", on_jobs)
        pg.route("**/api/jobs/fake-e*/events", lambda route, req: route.fulfill(status=200, content_type="text/event-stream", body=""))
        pg.goto(url + f"/?lang={lang}")
        pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
        pg.evaluate("window.gne.recipe.clear(); window.gne.recipe.loadProj('ppx-n2', ['12:H5:s10@ppx', '12:H3:s7@ppx'], 'ppx')")
        pg.click("#views button[data-view=eval]")
        pg.wait_for_function("window.gne.evalView && !document.querySelector('#evx-go').disabled", timeout=30000)
        pg.fill("#evx-doses", "0,1")
        pg.click("#evx-go")
        pg.uncheck("#evx-dm")
        pg.click("#evx-go")
        pg.check("#evx-dm")
        pg.fill("#evx-dm-n", "0")
        pg.click("#evx-go")
        pg.fill("#evx-dm-n", "16")
        pg.click("#evx-go")
        pg.wait_for_function(f"true", timeout=1000)
        ps = [b["params"] for b in posts if b["kind"] == "eval"]
        assert len(ps) == 4
        assert "direction_metrics" not in ps[0] and "dir_max_examples" not in ps[0]
        assert ps[1]["direction_metrics"] is False
        assert "dir_max_examples" in ps[2] and ps[2]["dir_max_examples"] is None and ps[3]["dir_max_examples"] == 16
        if lang == "en":
            assert pg.evaluate("[...window.gneI18n.missing]") == []
        pg.close()


def _fieldtest_list_row(name):
    """/api/fieldtests の一覧の 1 行の形。"""
    return {"name": name, "target": "a.gguf", "reference": "b.gguf", "same_level": False, "n_worse": 1, "updated": "2026-11-02T12:00:00"}


def _fieldtest_row(name):
    """/api/fieldtests/<名前> の形(比較の表つき)。"""
    return {"name": name, "target": {"model": "/m/a.gguf", "file": "a.gguf", "lora": None}, "reference": {"model": "/m/b.gguf", "file": "b.gguf", "lora": None},
            "settings": {"n_long": 24, "n_tools": 40, "long_tokens": 3072, "prompts": {"mine": 100}, "bench": {"gsm8k": 250}},
            "runs": [], "same_level": False, "worse": ["long.loop"], "updated": "2026-11-02T12:00:00", "seconds": 600,
            "compare": [{"section": "long", "metric": "loop", "label": "短い周期のループ", "reference": 0.0, "target": 0.125, "delta": 0.125, "worse": True},
                        {"section": "tools", "metric": "called", "label": "呼ぶべきときに呼んだ", "reference": 1.0, "target": 1.0, "delta": 0.0, "worse": False},
                        {"section": "prompts:mine", "metric": "empty", "label": "本文が空", "reference": 0.02, "target": 0.02, "delta": 0.0, "worse": False},
                        {"section": "bench:gsm8k", "metric": "accuracy", "label": "正答率", "reference": 0.97, "target": 0.96, "delta": -0.01, "worse": False,
                         "only_reference": 4, "only_target": 2, "p": 0.688},
                        {"section": "ppl", "metric": "mean", "label": "全体(幾何平均)", "reference": 6.77, "target": 6.78, "delta": 0.01, "worse": False}]}


def test_fieldtest_panel(refine_server, browser, tmp_path):
    """eval ビューの「実地テスト」: 入力を fieldtest のジョブにして投げ、終わると比較の表を出す。悪化の行に印。文章は出さない。英語の文言もある。"""
    import urllib.error
    import urllib.request
    url = refine_server
    from gne.store import Store  # noqa: F401
    done_row = _fieldtest_row("ftx")
    for bad in ("nope", "x.replies"):
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(url + f"/api/fieldtests/{bad}")
        assert ei.value.code == 400
    for lang in ("ja", "en"):
        pg = browser.new_page(viewport={"width": 1500, "height": 1000})
        errors, posts = [], []
        pg.on("pageerror", lambda e: errors.append(str(e)))

        def on_jobs(route, req):
            if req.method != "POST":
                route.continue_()
                return
            body = json.loads(req.post_data)
            posts.append(body)
            route.fulfill(status=200, content_type="application/json", body=json.dumps({
                "id": "fake-ft", "kind": body["kind"], "params": body["params"], "status": "queued", "done": 0, "total": 0, "stage": "",
                "eta_s": None, "yields": 0, "seconds": None, "result": None, "error": None}))
        pg.route("**/api/jobs", on_jobs)
        done = {"id": "fake-ft", "kind": "fieldtest", "params": {}, "status": "done", "done": 1, "total": 1, "stage": "", "eta_s": None, "yields": 0,
                "seconds": 600.0, "result": {"name": "ftx"}, "error": None}
        pg.route("**/api/jobs/fake-ft/events", lambda route, req: route.fulfill(status=200, content_type="text/event-stream", body=f"data: {json.dumps(done)}\n\n"))
        pg.route("**/api/fieldtests", lambda route, req: route.fulfill(status=200, content_type="application/json", body=json.dumps(
            [_fieldtest_list_row("ftx")])))
        pg.route("**/api/fieldtests/ftx", lambda route, req: route.fulfill(status=200, content_type="application/json", body=json.dumps(done_row)))
        pg.goto(url + f"/?lang={lang}")
        pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
        pg.click("#views button[data-view=eval]")
        pg.wait_for_function("window.gne.evalView && window.gne.evalView.ft", timeout=30000)
        pg.evaluate("document.querySelector('#evx-ft-wrap').open = true")
        pg.fill("#ft-prompts", "broken line")
        pg.click("#ft-go")
        assert not posts and pg.evaluate("document.querySelector('#ft-status').classList.contains('err')")   # 質問集の書き方の誤り
        pg.fill("#ft-model", "/m/a.gguf")
        pg.fill("#ft-against", "/m/b.gguf")
        pg.fill("#ft-prompts", "mine=/q/x.json\n")
        pg.fill("#ft-match", "abc")
        pg.fill("#ft-long", "2048")
        pg.uncheck("#ft-ppl")
        pg.check("#ft-gsm8k")
        pg.click("#ft-go")
        pg.wait_for_selector("#ft-result .ft-table", timeout=15000)
        p = posts[0]["params"]
        assert posts[0]["kind"] == "fieldtest" and p == {"long_tokens": 2048, "tool_tokens": 1536, "prompt_tokens": 1536, "ppl": False, "model": "/m/a.gguf",
                                                         "against": "/m/b.gguf", "match": "abc", "prompts": ["mine=/q/x.json"], "bench": ["gsm8k"]}
        assert pg.evaluate("document.querySelectorAll('#ft-result tr.ft-sec').length") == 5
        assert "0.688" in pg.evaluate("document.querySelector('#ft-result').textContent")
        assert pg.evaluate("document.querySelectorAll('#ft-result tr.ft-worse').length") == 1
        assert "12.5%" in pg.evaluate("document.querySelector('#ft-result tr.ft-worse').textContent")
        assert "6.780" in pg.evaluate("document.querySelector('#ft-result').textContent")
        assert pg.evaluate("document.querySelectorAll('#ft-list tr[data-ft=ftx]').length") == 1
        if lang == "en":
            assert pg.evaluate("[...window.gneI18n.missing]") == []
        assert not errors, errors
        pg.close()
