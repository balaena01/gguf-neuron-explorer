"""Phase 5(UI)のテスト。構造情報(ArchInfo)・サーバの API・ブラウザでの描画と hover。

ブラウザのテストは Playwright の Chromium を使う(無ければ飛ばす)。WSL では WSLg の D3D12 経由で実 GPU を使う。
"""

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


def _free_port() -> int:
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip("モデルがありません")
    import uvicorn

    from gne.api import Session
    from gne.ui.server import create_app
    port = _free_port()
    s = Session(MODEL, root=tmp_path_factory.mktemp("ui_root"))
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
def test_synthetic_70b_point_count(server, browser):
    pg = browser.new_page(viewport={"width": 1280, "height": 800})
    pg.goto(server + "/?synthetic=80x28672")
    pg.wait_for_function("window.gne && window.gne.ready", timeout=60000)
    assert pg.evaluate("window.gne.map.nPoints") == 80 * 28672
    pg.close()
