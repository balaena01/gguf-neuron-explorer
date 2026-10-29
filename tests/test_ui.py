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
    pg.click("#list-tabs button[data-tab=recipe]")
    pg.wait_for_function("document.querySelector('.rc-sampling')", timeout=10000)
    pg.wait_for_load_state("networkidle")                                           # 保存一覧を読み直して描き直すまで待つ
    pg.fill("#rc-name", "ui_sp")
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
    assert not pg.evaluate("document.querySelector('#head').classList.contains('collapsed')")
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
    # Contrast: 指標と床、上位の表、地図で色付け
    pg.click("#head-body .tabs button[data-tab=contrast]")
    pg.wait_for_function("document.querySelector('#head-body table.h-top')", timeout=30000)
    assert pg.evaluate("document.querySelectorAll('#head-body table.h-met td.hot').length") == 3   # 床を超えた 3 指標(dir_cos は無い)
    pg.click("#head-body button[data-ov=seed]")
    pg.wait_for_function("window.gne.heads && window.gne.heads.field === 'heads:seed'", timeout=30000)
    assert pg.evaluate("window.gne.map.layerParts[12].heads.visible") and not pg.evaluate("window.gne.map.layerParts[11].heads.visible")
    col = pg.evaluate("Array.from(window.gne.map.layerParts[12].heads.geometry.getAttribute('color').array.slice(3 * 18, 3 * 18 + 3))")
    assert col[0] > col[1] > col[2] > 0                                              # H3 は琥珀(床超え)
    assert "ヘッドの対照 seed の z" in pg.evaluate("document.querySelector('#legend').textContent")
    # Details: W_O ブロックの logit lens(後から読み込む)
    pg.click("#head-body .tabs button[data-tab=details]")
    pg.wait_for_function("document.querySelectorAll('#head-body .lens').length === 4", timeout=60000)
    # レシピ: ×0 で足す → 印と行。倍率を変えると L:Hh:xk、− 列は 1 の反対側
    pg.click("#head-body [data-hk='0']")
    assert pg.evaluate("window.gne.recipe.specs()") == ["12:H3:x0"]
    pg.wait_for_timeout(200)
    assert pg.evaluate("document.querySelectorAll('#overlay span.mk-h.minus').length") == 1
    pg.click("#list-tabs button[data-tab=recipe]")
    assert "L12 H3" in pg.evaluate("document.querySelector('#recipe-list table.rc').innerText")
    pg.fill("#recipe-list tr.rc-head input.num", "0.5")
    pg.dispatch_event("#recipe-list tr.rc-head input.num", "input")
    assert pg.evaluate("window.gne.recipe.specs()") == ["12:H3:x0.5"] and pg.evaluate("window.gne.recipe.specs(-1)") == ["12:H3:x1.5"]
    assert pg.evaluate("document.querySelector('#rc-export').disabled")             # ヘッドだけでは cvec にならない
    post("/api/recipes/ui_heads", {"units": ["12:H3:x2", "14:100:+1"]})
    pg.click("#list-tabs button[data-tab=rank]")
    pg.click("#list-tabs button[data-tab=recipe]")
    pg.wait_for_function("[...document.querySelectorAll('#rc-load option')].some((o) => o.value === 'ui_heads')", timeout=10000)
    pg.select_option("#rc-load", "ui_heads")
    assert pg.evaluate("window.gne.recipe.specs()") == ["12:H3:x2", "14:100:+1"] and pg.evaluate("window.gne.recipe.skipped") == 0
    pg.click("#recipe-list tr.rc-head a[data-open]")                                # 行から HEAD パネルへ
    pg.fill("#jump", "L5 H2")
    pg.press("#jump", "Enter")
    pg.wait_for_function("document.querySelector('#head-body .n-id') && document.querySelector('#head-body .n-id').textContent.startsWith('L5 H2')", timeout=30000)
    # fires: 文章を流すとヘッドも、選んだトークンへの注目で塗る
    pg.click("#list-tabs button[data-tab=fires]")
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
    pg.wait_for_load_state("networkidle")                                           # 保存済みの eval を読み終えてから差し替える
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
    assert not pg.evaluate("document.querySelector('#expert').classList.contains('hidden')")
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
