"""qwen4exp(Qwen3.8-Flash-Next)への対応。

実物は最小でも 66 GB あって CI に載らないので、2 つで確かめる。
- 実物のヘッダー(tests/data、メタデータとテンソルの一覧だけ)で、arch の読み取りと分割 GGUF を確かめる。
- 同じ構造の小さい乱数のモデル(tests/tools/make_tiny_qwen4exp.py)を CPU で動かして、同梱の llama.cpp のグラフと gne のつなぎ目を確かめる。
GPU は使わない。
"""
from __future__ import annotations

import ctypes
import gzip
import json
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "tools"))
import make_tiny_qwen4exp as tiny  # noqa: E402

# 小さなモデルのヘッド数は、CUDA の gated delta net が対応する 16・32・64・128 のどれでもない。n_gpu_layers=0 でも GPU が見えると
# その演算が GPU に渡って ggml_abort でプロセスごと落ちる。GPU が見えるときは test_qwen4exp_cpu.py が CPU だけの子プロセスで流す。
pytestmark = pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES", None) != "", reason="GPU が見えると小さなモデルは落ちる(CPU の子プロセスで流す)")

CHROME = Path(os.environ.get("GNE_CHROME", Path.home() / ".cache/ms-playwright/chromium-1223/chrome-linux64/chrome"))
HEADER = json.load(gzip.open(HERE / "data/qwen4exp_iq1s_header.json.gz"))
SPEC = {"name": "tiny", "system_a": "You are cold and dismissive.", "system_b": "You are warm and kind.",
        "user_prompts": ["How was your day?", "I failed my exam.", "Can you help me?", "My cat is sick.",
                         "I got a new job!", "I can't sleep.", "Do you remember me?", "I made dinner for you."]}


# --- 実物のヘッダー -----------------------------------------------------------------------------------------------

def _value(v):
    """(型番号, 中身のバイト列)。整数の配列は値に合わせて U32 か U64。"""
    if isinstance(v, bool):
        return 7, struct.pack("<?", v)
    if isinstance(v, int):
        return (4, struct.pack("<I", v)) if 0 <= v < 2**32 else (11, struct.pack("<q", v))
    if isinstance(v, float):
        return 6, struct.pack("<f", v)
    if isinstance(v, str):
        b = v.encode()
        return 8, struct.pack("<Q", len(b)) + b
    if isinstance(v, list):
        if all(isinstance(x, str) for x in v):
            body = b"".join(_value(x)[1] for x in v)
            return 9, struct.pack("<IQ", 8, len(v)) + body
        if any(x >= 2**32 for x in v):
            return 9, struct.pack("<IQ", 10, len(v)) + struct.pack(f"<{len(v)}Q", *v)
        return 9, struct.pack("<IQ", 4, len(v)) + struct.pack(f"<{len(v)}I", *v)
    raise TypeError(type(v))


def _stub_shard(path: Path, kv: dict, tensors: list) -> None:
    """重みのない GGUF(ヘッダーだけ)。メタデータの読み手は本体を読まないので、これで足りる。"""
    out = [b"GGUF", struct.pack("<IQQ", 3, len(tensors), len(kv))]
    for k, v in kv.items():
        b = k.encode()
        t, body = _value(v)
        out += [struct.pack("<Q", len(b)), b, struct.pack("<I", t), body]
    for name, shape, ty, off in tensors:
        b = name.encode()
        out += [struct.pack("<Q", len(b)), b, struct.pack("<I", len(shape)), struct.pack(f"<{len(shape)}Q", *shape), struct.pack("<IQ", ty, off)]
    path.write_bytes(b"".join(out))


@pytest.fixture(scope="module")
def real_header(tmp_path_factory):
    d = tmp_path_factory.mktemp("header")
    for s in HEADER["shards"]:
        _stub_shard(d / s["file"], s["kv"], s["tensors"])
    return d / HEADER["shards"][0]["file"]


def test_real_header_is_read_as_one_model(real_header):
    from gne.gguf_fast import FastGGUF
    from gne.weights import Weights
    g = FastGGUF(real_header)
    assert g.is_split and len(g.parts) == 3 and len(g.tensors) == 1224
    w = Weights(real_header)
    assert (w.meta.arch, w.meta.n_layer, w.meta.n_embd, w.meta.n_expert, w.meta.n_ff) == ("qwen4exp", 48, 2560, 512, 640)
    assert w.has_tensor("blk.47.attn_output.weight") and w.has_tensor("per_layer_token_embd.weight")
    assert w.meta.file_size == sum(p.stat().st_size for p in g.parts)
    # どのファイルを渡しても同じモデルになる。テンソルの一部がない・足りないファイルは止める
    assert Weights(g.parts[1]).meta.model_id == w.meta.model_id
    with pytest.raises(NotImplementedError, match="分割"):
        _ = w.reader


def test_missing_shard_is_reported(real_header, tmp_path):
    from gne.gguf_fast import FastGGUF
    p = tmp_path / real_header.name
    p.write_bytes(real_header.read_bytes())
    with pytest.raises(ValueError, match="足りません"):
        FastGGUF(p)


def test_real_architecture(real_header):
    from gne.arch import arch_info
    from gne.weights import Weights
    a = arch_info(Weights(real_header))
    assert a.hc_streams == 4 and a.n_layer == 48
    attn = [li for li in a.layers if li.mixer == "attention"]
    linear = [li for li in a.layers if li.mixer == "linear_attention"]
    assert [li.index for li in attn] == list(range(3, 48, 4)) and len(linear) == 36
    assert {(li.n_head, li.head_dim, li.head_writer, li.sparse_top_k) for li in attn} == {(24, 256, "attn_output", 2048)}
    assert {(li.n_head, li.head_dim, li.head_writer, li.sparse_top_k) for li in linear} == {(48, 128, "ssm_out", 0)}
    assert sum(li.n_head for li in a.layers) == 36 * 48 + 12 * 24
    assert [li.index for li in a.layers if li.has_ple] == [1]
    assert {(li.ffn, li.n_ff, li.n_expert, li.n_expert_used, li.n_ff_shexp) for li in a.layers} == {("moe+shared", 640, 512, 10, 640)}


def test_real_quant_types_can_be_read():
    """実物の IQ1_S の書き込み行列と専門家の型。gguf-py が知らない Q2_0 も、型の大きさと dequantize が揃うこと。"""
    from gne.quant import byte_shape, can_dequantize, qtype_of
    types = {t[2] for s in HEADER["shards"] for t in s["tensors"]}
    assert 42 in types                                     # Q2_0
    assert qtype_of(42).name == "Q2_0"
    assert byte_shape((512, 2560, 640), qtype_of(42))[-1] > 0
    for ty in types - {0}:
        assert can_dequantize(qtype_of(ty)), qtype_of(ty).name


@pytest.mark.parametrize("name", ["Q8_0", "Q5_K", "Q6_K", "Q2_0", "IQ4_NL", "IQ4_XS", "IQ2_S", "IQ2_XXS"])
def test_dequantize_undoes_quantize(name):
    from gguf import GGMLQuantizationType as G
    from llama_cpp._ggml import libggml

    from gne.quant import block_sizes, dequantize, qtype_of
    q = next(qtype_of(i) for i in range(50) if _safe_name(qtype_of, i) == name)
    n = 8192 if q.value not in (G.IQ2_XXS.value,) else 8192
    bs, ts = block_sizes(q)
    n = n // bs * bs
    x = (np.random.default_rng(0).standard_normal(n) * 0.2).astype(np.float32)
    dst = np.zeros(n // bs * ts, np.uint8)
    imat = np.ones(n, np.float32)
    libggml.ggml_quantize_init.restype, libggml.ggml_quantize_init.argtypes = None, [ctypes.c_int]
    libggml.ggml_quantize_init(q.value)
    f = libggml.ggml_quantize_chunk
    f.restype, f.argtypes = ctypes.c_size_t, [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    assert f(q.value, x.ctypes.data, dst.ctypes.data, 0, 1, n, imat.ctypes.data) == dst.size
    y = dequantize(dst, q)
    assert y.shape == (n,)
    assert np.corrcoef(x, y)[0, 1] > 0.5                  # 2 bit の型でも向きは残る


def _safe_name(qtype_of, i):
    try:
        return qtype_of(i).name
    except ValueError:
        return None


def test_real_chat_template_renders():
    from gne.engine import render_template
    t = HEADER["shards"][0]["kv"]["tokenizer.chat_template"]
    msgs = [{"role": "user", "content": "Hello"}]
    on = render_template(t, msgs, enable_thinking=True, add_generation_prompt=True)
    off = render_template(t, msgs, enable_thinking=False, add_generation_prompt=True)
    assert on.endswith("<think>\n") and off.endswith("<think>\n\n</think>\n\n") and "Hello" in on


# --- 小さいモデルを CPU で -------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    return tiny.write(tmp_path_factory.mktemp("tiny") / "tiny.gguf")


@pytest.fixture(scope="module")
def session(model_path, tmp_path_factory):
    from gne.api import Session
    s = Session(model_path, root=tmp_path_factory.mktemp("root"), n_gpu_layers=0, n_ctx=1024)
    yield s
    s.close()


TEXT = "Tell me a short story about a lighthouse keeper and the sea."


def test_residual_is_the_mean_of_the_streams(session):
    from gne.capture import Capture
    e = session.engine
    assert e.hc_streams == 4 and not e.cvec_supported
    ids = e.tokenize(TEXT, add_special=False, parse_special=False)
    cap = Capture()
    cap.set_wanted(lambda n: n in ("l_last-2", "l_out-2"))
    ctx = e.probe_ctx(cap)
    e.clear_memory(ctx)
    e.prefill(ctx, [ids])
    mean = cap.take()["l_out-2"]
    assert mean.shape == (len(ids), e.n_embd)
    cap.hc_streams = 0                                     # 生の [n_embd, 4, T]
    e.clear_memory(ctx)
    e.prefill(ctx, [ids])
    raw = cap.take()["l_last-2"]
    cap.hc_streams = 4
    assert raw.shape == (len(ids) * 4, e.n_embd)
    np.testing.assert_allclose(raw.reshape(len(ids), 4, -1).mean(axis=1), mean, rtol=1e-5, atol=1e-5)


def test_verify_checks_what_applies(session):
    from gne.verify import Verifier
    r = Verifier(session.engine, session.weights, layer=2, seed=0)
    assert r.v1_shapes().status == "pass"
    got = r.v26_hc_streams()
    assert got.status == "pass", got.detail


def test_control_vectors_are_refused_not_silently_ignored(session):
    from gne.steer import GneError
    e = session.engine
    ctx = e.steer_ctx()
    with pytest.raises(GneError, match="control vector"):
        e.set_cvec(ctx, {2: np.ones(e.n_embd, np.float32)})
    e.set_cvec(ctx, None)                                  # 外すのは何も起きない


def test_split_file_gives_the_same_output(session, model_path, tmp_path):
    from gne.engine import Engine
    from gne.gguf_fast import split_paths
    first = tiny.write(tmp_path / "tiny.gguf", split_tensors=120)
    assert split_paths(first) and len(split_paths(first)) >= 3
    e = session.engine
    ids = e.tokenize(TEXT, add_special=False, parse_special=False)
    c1 = e.steer_ctx()
    e.clear_memory(c1)
    a = e.prefill(c1, [ids]).logits.copy()
    e2 = Engine(first, n_gpu_layers=0, n_ctx=1024, n_seq_max=1)
    try:
        c2 = e2.steer_ctx()
        e2.clear_memory(c2)
        b = e2.prefill(c2, [ids]).logits.copy()
    finally:
        e2.close()
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


def test_attention_capture_on_both_mixers(session):
    from gne.attn import capture_attention
    lin = capture_attention(session, [TEXT], layers=[0], want_pattern=True)[0]
    att = capture_attention(session, [TEXT], layers=[3], want_pattern=True)[0]
    T = len(lin.ids)
    assert lin.head_out[0].shape == (T, 4, 8) and 0 not in lin.pattern
    assert att.head_out[3].shape == (T, 4, 8) and att.pattern[3].shape == (4, T, T)


@pytest.fixture(scope="module")
def pipeline(session, tmp_path_factory):
    p = tmp_path_factory.mktemp("spec") / "spec.json"
    p.write_text(json.dumps(SPEC))
    return session.heads_pipeline(p, n_heads=[4, 2], max_examples=8, top_layers=2)


def test_pipeline_reaches_a_recipe(pipeline):
    assert pipeline["allocate"]["table"] if "allocate" in pipeline else pipeline
    assert pipeline["recommended"]["units"]


@pytest.mark.parametrize("L,h", [(0, 1), (3, 1), (6, 2)])
def test_head_projection_is_exact_on_both_mixers(session, pipeline, L, h):
    """CHiPS の編集(V24・V25): 書き込みの差は -s r (r·v) で、直交成分は動かない。線形注意と疎な attention の両方で。"""
    from gne.headproj import check_head_proj
    r = check_head_proj(session, L, h, "tiny")
    assert r["ok"], r


def test_recipe_lora_changes_the_output(session, pipeline):
    from gne.headproj import export_proj_lora
    from gne.steer import Recipe
    rec = Recipe.load(pipeline["recommended"]["recipe"])
    out = Path(session.store.heads_dir(session.meta.model_id)) / "t.lora.gguf"
    export_proj_lora(session.weights, rec, out, session.direction_set)
    e = session.engine
    ids = e.tokenize(TEXT, add_special=False, parse_special=False)
    ctx = e.steer_ctx()
    e.set_lora(ctx, None)
    e.clear_memory(ctx)
    a = e.prefill(ctx, [ids]).logits.copy()
    e.set_lora(ctx, [(out, 1.0)])
    e.clear_memory(ctx)
    b = e.prefill(ctx, [ids]).logits.copy()
    e.set_lora(ctx, None)
    assert not np.allclose(a, b)


def test_moe_gain_on_an_expert_neuron(session):
    """専門家のニューロンの gain は LoRA で動く。"""
    from gne.gain import export_gain_lora
    r = session.recipe(units=["2:E3:N5:x0"])
    out = Path(session.store.heads_dir(session.meta.model_id)) / "g.lora.gguf"
    export_gain_lora(session.weights, r, out)
    assert out.stat().st_size > 0


def test_head_lens_works_without_output_norm(session):
    """qwen4exp には output_norm がない(最後の混ぜ合わせが兼ねる)。"""
    g = session.weights.output_norm_weight()
    assert g.shape == (session.meta.n_embd,) and np.isfinite(g).all()


# --- UI --------------------------------------------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(session):
    import uvicorn

    from gne.ui.server import create_app
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(create_app(session), host="127.0.0.1", port=port, log_level="warning"))
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
        pytest.skip(f"Chromium がありません({CHROME})")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=str(CHROME), headless=True)   # GPU は使わない
        yield b
        b.close()


def _real_arch() -> dict:
    layers = []
    for L in range(48):
        a = (L + 1) % 4 == 0
        layers.append(dict(index=L, mixer="attention" if a else "linear_attention", n_head=24 if a else 48, n_head_kv=2 if a else 16,
                           ffn="moe+shared", n_ff=640, n_expert=512, n_expert_used=10, n_ff_shexp=640, has_post_ffn_norm=False,
                           head_dim=256 if a else 128, head_writer="attn_output" if a else "ssm_out",
                           head_out=["attn_gated"] if a else ["final_output"], has_router_bias=False, routing="softmax_topk",
                           norm_topk=None, sparse_top_k=2048 if a else 0, has_ple=L == 1))
    return dict(arch="qwen4exp", n_layer=48, n_embd=2560, n_vocab=248320, layers=layers, grid_side=598, hc_streams=4)


def test_ui_map_stays_fast_at_the_real_size(server, browser):
    """全ニューロンは 1,570 万点。専門家ごとの点にまとめて、すぐ開くこと。層のラベルに疎な attention などが出ること。"""
    pg = browser.new_page(viewport={"width": 1400, "height": 860})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))

    def model(route):
        r = route.fetch()
        d = r.json()
        d["arch"] = _real_arch()
        route.fulfill(response=r, json=d)
    pg.route("**/api/model", model)
    t0 = time.time()
    pg.goto(server + "/?lang=en")
    pg.wait_for_function("window.gne && window.gne.ready", timeout=60000)
    assert time.time() - t0 < 30
    assert pg.evaluate("window.gne.map.coarse") is True
    assert pg.evaluate("window.gne.map.nPoints") == 48 * (512 + 640)      # 共有の専門家はニューロンのまま
    assert "1 point = 1 expert" in pg.evaluate("document.querySelector('#legend').textContent")
    assert "4 streams" in pg.evaluate("document.querySelector('#legend').textContent")
    # 点の色: 通し番号順の配列を渡すと、専門家ごとに一番目立つ点になる
    n = 512 * 640 + 640
    lit = pg.evaluate(f"""(() => {{ const m = window.gne.map, a = new Uint8Array({n} * 4);
        a[4 * (7 * 640 + 3) + 3] = 200; a[4 * (7 * 640 + 3)] = 99; m.setLayerColors(2, a);
        const o = m.offsets[2] * 4 + 7 * 4; return [m.colorAttr.array[o], m.colorAttr.array[o + 3]]; }})()""")
    assert lit == [99, 200]
    assert errors == []


def test_ui_small_model_keeps_every_neuron(server, browser):
    pg = browser.new_page(viewport={"width": 1400, "height": 860})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(server + "/?lang=en")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    assert pg.evaluate("window.gne.map.coarse") is False
    assert pg.evaluate("document.body.dataset.cvec") == "off"
    labels = pg.evaluate("[...document.querySelectorAll('#layer-select option, select option')].map(o => o.textContent).join(' | ')")
    assert "sparse top 16" in labels and "n-gram" in labels
    assert errors == []


def test_ui_panels(server, browser, pipeline):
    pg = browser.new_page(viewport={"width": 1400, "height": 860})
    errors, bad = [], []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("response", lambda r: bad.append((r.status, r.url)) if r.status >= 500 else None)
    pg.goto(server + "/?lang=en")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    for jump, sel, want in (("L0 H1", "#head-body", "linear"), ("L3 H1", "#head-body", "attention"), ("L2 E3", "#expert-body", "softmax top-k")):
        pg.fill("#jump", jump)
        pg.press("#jump", "Enter")
        pg.wait_for_function(f"document.querySelector('{sel}').innerText.includes('{want}')", timeout=30000)
    # 足す押し方のボタンは出ず、理由が出る
    pg.fill("#jump", "L2 E3 N5")
    pg.press("#jump", "Enter")
    pg.wait_for_function("document.querySelector('#neuron-body, #panel-body, .panel-body').innerText.length > 0", timeout=30000)
    pg.wait_for_function("document.body.innerText.includes('add is not available')", timeout=30000)
    assert pg.evaluate("document.querySelectorAll('[data-rc]').length") == 0
    assert errors == [] and bad == []


def test_ui_expert_opens_into_its_neurons(server, browser):
    """1 点 = 1 専門家の地図: 開くと正面に来て中のニューロンが広がり、そのニューロンを指せて、戻すと元の点に戻る。"""
    pg = browser.new_page(viewport={"width": 1400, "height": 860})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(server + "/?lang=en&coarse=1")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    m = "window.gne.map"
    assert pg.evaluate(f"{m}.coarse") is True and pg.evaluate(f"{m}.nPoints") == 8 * (8 + 16)
    pg.evaluate(f"{m}.setLayerColors(2, (() => {{ const a = new Uint8Array((8 * 16 + 16) * 4); a[4 * (3 * 16 + 5) + 3] = 200; a[4 * (3 * 16 + 5)] = 77; return a; }})())")
    assert pg.evaluate(f"{m}.colorAttr.array[{m}.offsets[2] * 4 + 3 * 4 + 3]") == 200
    pg.fill("#jump", "L2 E3")
    pg.press("#jump", "Enter")
    pg.wait_for_function(f"{m}._detail && {m}._detail.expert === 3 && {m}.front === 2", timeout=30000)
    assert "Showing the neurons of L2 E3" in pg.evaluate("document.querySelector('#legend').textContent")
    # 隠した専門家の点は 0、共有の専門家は残る
    assert pg.evaluate(f"[...{m}.colorAttr.array.slice({m}.offsets[2] * 4, {m}.offsets[2] * 4 + 8 * 4)].every((x) => x === 0)")
    assert pg.evaluate(f"{m}._detail.color.array.length") == 16 * 4
    # 色を渡すと、広げた専門家の分がそのニューロンに当たる
    pg.evaluate(f"{m}.setLayerColors(2, (() => {{ const a = new Uint8Array((8 * 16 + 16) * 4); a[4 * (3 * 16 + 5) + 3] = 200; a[4 * (3 * 16 + 5)] = 77; return a; }})())")
    assert pg.evaluate(f"[{m}._detail.color.array[5 * 4], {m}._detail.color.array[5 * 4 + 3]]") == [77, 200]
    # ニューロンの位置が画面に出て、そこを指すとそのニューロンになる
    pos = pg.evaluate(f"{m}.neuronPosition(2, 3 * 16 + 5)")
    assert pos is not None
    hit = pg.evaluate(f"""(() => {{ const s = {m}.screenOf(2, 3 * 16 + 5), r = {m}.renderer.domElement.getBoundingClientRect();
        const h = {m}._pick({m}._ndc({{ clientX: s.x, clientY: s.y }})); return h && [h.layer, h.expert, h.neuron, h.coarse]; }})()""")
    assert hit == [2, 3, 5, False]
    # 行の札が専門家の番号からニューロンの番号に変わる。共有の専門家には名前が付く
    pg.wait_for_function("[...document.querySelectorAll('#overlay span.ex')].some((x) => x.textContent === 'N4')", timeout=10000)
    assert not pg.evaluate("[...document.querySelectorAll('#overlay span.ex')].some((x) => x.textContent.startsWith('E'))")
    assert pg.evaluate("[...document.querySelectorAll('#overlay span')].some((x) => x.textContent === 'shared 16')")
    # 本物のクリックで NEURON パネルが開く
    s = pg.evaluate(f"{m}.screenOf(2, 3 * 16 + 5)")
    pg.mouse.click(s["x"], s["y"])
    pg.wait_for_function("(document.querySelector('#neuron-body')?.innerText || '').includes('L2 E3 N5')", timeout=30000)
    assert pg.evaluate(f"{m}._marker.scale.x <= {m}._detail.cell")     # 選んだ印の輪が隣のニューロンに掛からない
    pg.click("#detail-back")
    pg.wait_for_function(f"!{m}._detail", timeout=10000)
    assert pg.evaluate(f"{m}.colorAttr.array[{m}.offsets[2] * 4 + 3 * 4 + 3]") == 200
    assert errors == []


def test_split_file_refuses_scale_patches_and_loads_from_any_shard(session, tmp_path):
    from gne.engine import Engine
    from gne.gguf_fast import split_paths
    from gne.patch import Patcher, supported
    from gne.quant import qtype_of
    from gne.steer import GneError
    from gne.weights import Weights
    first = tiny.write(tmp_path / "tiny.gguf", split_tensors=120)
    w = Weights(first)
    with pytest.raises(GneError, match="分割 GGUF"):
        Patcher(w)
    assert supported(qtype_of(42)) is False             # 実物の ffn_down_exps の Q2_0 は、結合してもスケールパッチの対応外
    second = split_paths(first)[1]
    e = Engine(second, n_gpu_layers=0, n_ctx=64, n_seq_max=1)
    try:
        assert e.path == split_paths(first)[0] and e.n_layer == 8
    finally:
        e.close()


def test_export_cvec_is_refused_on_this_arch(session, tmp_path):
    from gne.steer import GneError
    with pytest.raises(GneError, match="control vector を読みません") as err:
        session.export_cvec(["1:E0:N0:+0.5"], tmp_path / "x.gguf")
    assert err.value.code == "ARCH_UNSUPPORTED"


def test_ui_hides_whole_expert_gain_on_a_split_file(server, browser):
    pg = browser.new_page(viewport={"width": 1400, "height": 860})
    pg.goto(server + "/?lang=en")
    pg.wait_for_function("window.gne && window.gne.overlayReady", timeout=60000)
    assert pg.evaluate("window.gne.state ? window.gne.state.model.meta.is_split : false") is False
    pg.fill("#jump", "L2 E3")
    pg.press("#jump", "Enter")
    pg.wait_for_selector("#expert-body [data-xk='0']", timeout=30000)
    pg.evaluate("document.body.dataset.split = '1'")
    pg.fill("#jump", "L2 E4")
    pg.press("#jump", "Enter")
    pg.wait_for_function("(document.querySelector('#expert-body')?.innerText || '').includes('not available on a split GGUF')", timeout=30000)
    assert pg.query_selector("#expert-body [data-xk='0']") is None


def test_chips_runs_on_a_split_file(tmp_path):
    """実物は分割ファイルなので、そのまま pipeline → LoRA まで通す。1 ファイル版に同じ LoRA を当てた出力と一致する。"""
    from gne.api import Session
    from gne.headproj import export_proj_lora
    from gne.steer import Recipe
    (tmp_path / "split").mkdir()
    (tmp_path / "one").mkdir()
    first = tiny.write(tmp_path / "split/tiny.gguf", split_tensors=120)
    one = tiny.write(tmp_path / "one/tiny.gguf")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(SPEC))
    s = Session(first, root=tmp_path / "r1", n_gpu_layers=0, n_ctx=1024)
    s1 = Session(one, root=tmp_path / "r2", n_gpu_layers=0, n_ctx=1024)
    try:
        p = s.heads_pipeline(spec, n_heads=[4, 2], max_examples=8, top_layers=2)
        lora = tmp_path / "x.lora.gguf"
        export_proj_lora(s.weights, Recipe.load(p["recommended"]["recipe"]), lora, s.direction_set)
        ids = s.engine.tokenize(TEXT, add_special=False, parse_special=False)

        def logits(e, lo):
            ctx = e.steer_ctx()
            e.set_lora(ctx, lo)
            e.clear_memory(ctx)
            out = e.prefill(ctx, [ids]).logits.copy()
            e.set_lora(ctx, None)
            return out
        edited = logits(s.engine, [(lora, 1.0)])
        assert not np.allclose(logits(s.engine, None), edited)
        np.testing.assert_allclose(edited, logits(s1.engine, [(lora, 1.0)]), rtol=1e-5, atol=1e-5)
    finally:
        s.close()
        s1.close()


@pytest.mark.parametrize("L,h", [(0, 1), (2, 3), (3, 1)])
def test_head_projection_moves_every_stream_only_along_the_direction(session, pipeline, L, h, tmp_path):
    """V24 は書き込み行列の出力を見る。こちらは残差の 4 本の流れ(attention の直後)で、r̂ に直交する差が出ないことを見る。"""
    from gne.capture import Capture
    from gne.headproj import export_proj_lora, head_direction
    from gne.steer import Recipe, Unit
    e = session.engine
    r = head_direction(session.direction_set("tiny"), L, "own")[0].astype(np.float64)
    path = tmp_path / "c.lora.gguf"
    rec = Recipe([Unit(L, None, 0.0, "head", mode="proj", k=1.0, head=h, direction="tiny")], session.meta.model_id, name="c")
    export_proj_lora(session.weights, rec, path, session.direction_set)
    ids = e.tokenize(TEXT, add_special=False, parse_special=False)
    name = f"hc_combine-{L}"
    cap = Capture()
    ctx = e.probe_ctx(cap, need=len(ids))
    cap.hc_streams = 0
    cap.set_wanted(lambda x: x == name)

    def run(lo):
        e.set_lora(ctx, lo)
        e.clear_memory(ctx)
        e.prefill(ctx, [ids])
        return cap.take()[name].reshape(-1, e.hc_streams, e.n_embd).astype(np.float64)
    try:
        d = run([(path, 1.0)]) - run(None)
    finally:
        e.set_lora(ctx, None)
        e.drop_lora(path)
        cap.hc_streams = e.hc_streams
    along = d @ r
    assert np.linalg.norm(d) > 0 and (np.abs(along).sum(axis=0) > 0).all()    # どの流れも動く
    assert np.linalg.norm(d - along[..., None] * r) < 1e-3 * np.linalg.norm(d)


def test_bench_works_without_control_vectors(session):
    """gne bench の実測が見積もりの元になるので、control vector を読まない arch でも測れる。"""
    from gne.bench import run_bench
    r = run_bench(session.engine, layer=2, reps=2, gen_tokens=4)
    assert r["set_plus_prefill_ms"] > 0 and r["gen"]["single"]["tok_per_s"] > 0


def test_writers_split_adds_up_to_the_gap(session, pipeline):
    """attention・専門家・共有の専門家の書き込みの差を足すと、参照の層の A − B の差になる。"""
    d = session.heads_writers("tiny", ref_layer=6, expert_layers=[5, 6], max_examples=8)
    assert d["sum_error"] < 1e-3, d
    assert [r["layer"] for r in d["rows"]] == list(range(7))
    assert any(abs(r["moe"]) > 0 for r in d["rows"]) and any(abs(r["shared"]) > 0 for r in d["rows"])
    assert set(d["experts"]) == {"5", "6"} and d["experts"]["6"][0]["diff"] >= d["experts"]["6"][-1]["diff"]
    assert "prompt" not in json.dumps(d)                     # 質問の文を残さない
