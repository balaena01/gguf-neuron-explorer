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


def _free_port() -> int:
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


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
                              args=["--use-gl=angle", "--use-angle=gl", "--enable-gpu", "--ignore-gpu-blocklist"],
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
    assert rgba(7) == [30, 37, 48, 0]                                                # 床(対照の最大 mass)以下
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
    pg.click("#list-tabs button[data-tab=recipe]")
    assert pg.evaluate("document.querySelectorAll('#recipe-list table.rc tr').length") == 1
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
    pg.click("#list-tabs button[data-tab=fires]")
    pg.fill("#fi-text", "The aliens landed on Mars.")
    pg.dispatch_event("#fi-text", "input")
    pg.click("#fi-run")
    pg.wait_for_function("document.querySelector('#fi-list tr[data-l]')", timeout=60000)
    assert pg.evaluate("window.gne.overlay.field").startswith("fires:")
    assert "活性統計が無い" in pg.evaluate("document.querySelector('#fires-list .ch-status').textContent")   # 統計の無い層を名指し
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
    for tab in ("fires", "recipe", "rank"):
        pg.click(f"#list-tabs button[data-tab={tab}]")
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
    pg.click("#list-tabs button[data-tab=recipe]")
    pg.select_option("#rc-ab-dir", "pirate_ui")
    pg.wait_for_function("document.querySelector('#rc-ab-scale')", timeout=5000)
    pg.evaluate("(() => { const s = document.querySelector('#rc-ab-scale'); s.value = '0.5'; s.dispatchEvent(new Event('input')); })()")
    assert pg.evaluate("window.gne.recipe.ablate") == {"direction": "pirate_ui", "scale": 0.5, "layers": "all", "from_layer": None}
    pg.click("#rc-chat")                                                            # ablate だけのレシピでもチャットを開ける
    pg.wait_for_function("window.gne.chat.target === 'recipe'", timeout=10000)
    cols = pg.evaluate("window.gne.chat.columns()")
    assert cols[0]["ablate"]["scale"] == -0.5 and "ablate" not in cols[1] and cols[2]["ablate"]["scale"] == 0.5
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
    pg.click("#list-tabs button[data-tab=rank]")
    pg.click("#list-tabs button[data-tab=recipe]")
    pg.wait_for_function("[...document.querySelectorAll('#rc-load option')].some((o) => o.value === 'ui_dir')", timeout=10000)
    pg.select_option("#rc-load", "ui_dir")
    assert pg.evaluate("window.gne.recipe.specs()") == ["12:dir:pirate_ui:+0.27", "14:100:+0.5"]
    assert "dir:pirate_ui" in pg.evaluate("document.querySelector('#recipe-list table.rc').innerText")
    assert pg.evaluate("document.querySelector('#recipe-list tr[data-i=\"0\"] input.dose').step") == "0.005"
    assert pg.evaluate("window.gne.recipe.neurons().length") == 1
    pg.click("#rc-chat")
    pg.wait_for_function("document.querySelector('#ch-sample')", timeout=10000)
    cols = pg.evaluate("window.gne.chat.columns()")
    assert cols[2]["units"] == ["12:dir:pirate_ui:+0.27", "14:100:+0.5"] and cols[0]["units"][0] == "12:dir:pirate_ui:-0.27"
    pg.check("#ch-sample")
    assert pg.evaluate("window.gne.chat.sample") is True
    assert not errors, errors
    pg.close()
