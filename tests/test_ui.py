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


@pytest.fixture(scope="module")
def browser():
    if not CHROME.exists():
        pytest.skip(f"Chromium がありません({CHROME}、GNE_CHROME で指定)")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=str(CHROME), headless=True,
                              args=["--use-gl=angle", "--use-angle=gl", "--enable-gpu", "--ignore-gpu-blocklist"],
                              env={**os.environ, "MESA_D3D12_DEFAULT_ADAPTER_NAME": "NVIDIA"})
        yield b
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
