"""CHANGE-05 のテスト: MoE の住所・散布・スケールパッチ・専門家の介入。

モデルを使うものは granite-3.1-1b-a400m(GNE_TEST_MOE、granitemoe)と LFM2-8B-A1B(GNE_TEST_MOE_BIAS、lfm2moe、
ルーターのバイアスあり)。無ければ飛ばす。
"""

import os
from pathlib import Path

import numpy as np
import pytest
from gguf import GGMLQuantizationType as T
from gguf import GGUFWriter
from gguf import quants as gquants
from gguf.constants import GGML_QUANT_SIZES

from gne.steer import GneError, Recipe, parse_unit

ROOT = Path(__file__).resolve().parent.parent
MOE = Path(os.environ.get("GNE_TEST_MOE", ROOT / "models" / "granite-3.1-1b-a400m-instruct-Q8_0.gguf"))
MOE_BIAS = Path(os.environ.get("GNE_TEST_MOE_BIAS", ROOT / "models" / "LFM2-8B-A1B-Q4_K_M.gguf"))
MOE_SHARED = Path(os.environ.get("GNE_TEST_MOE_SHARED", ROOT / "models" / "Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf"))


# --- A. 住所 --------------------------------------------------------------------------------------

def test_parse_moe_units(tmp_path):
    cases = {
        "12:E37:N511:+0.8": ("expert_neuron", "add", 37, 511, 0.8, None, "L12 E37 N511 +0.8"),
        "12:E37:N511:x0": ("expert_neuron", "gain", 37, 511, 0.0, 0.0, "L12 E37 N511 ×0"),
        "12:e3:x2": ("expert", "gain", 3, None, 0.0, 2.0, "L12 E3 ×2"),
        "12:E37:route:-1.5": ("expert", "route", 37, None, -1.5, None, "L12 E37 route -1.5"),
        "12:S:N511:+0.5": ("shared", "add", None, 511, 0.5, None, "L12 S N511 +0.5"),
        "12:S:N5:x3": ("shared", "gain", None, 5, 0.0, 3.0, "L12 S N5 ×3"),
    }
    for spec, (kind, mode, e, n, dose, k, label) in cases.items():
        u = parse_unit(spec)
        assert (u.kind, u.mode, u.expert, u.neuron, u.dose, u.k, u.label()) == (kind, mode, e, n, dose, k, label), spec
    with pytest.raises(GneError):
        parse_unit("12:S:x2")                               # shared expert はニューロンだけ
    r = Recipe([parse_unit(s) for s in cases], "m", "raw")
    r.validate()
    half = r.scaled(0.5)
    assert [u.label() for u in half.units][:4] == ["L12 E37 N511 +0.4", "L12 E37 N511 ×0.5", "L12 E3 ×1.5", "L12 E37 route -0.75"]
    p = tmp_path / "r.yaml"
    r.save(p)
    assert "delta: -1.5" in p.read_text() and [u.label() for u in Recipe.load(p).units] == [u.label() for u in r.units]
    assert len(r.moe_units) == 6 and len(r.gain_units) == 3
    with pytest.raises(GneError):
        Recipe([parse_unit("12:E3:N1:+1")], "m", "absmax").validate()


# --- B. 散布 ---------------------------------------------------------------------------------------

def test_scatter_and_shapes():
    from gne.moe import by_expert, per_token, scatter, selected, usage
    rng = np.random.default_rng(0)
    Tn, used, n_exp, n_ff = 5, 2, 4, 3
    arg = np.stack([rng.permutation(n_exp) for _ in range(Tn)]).astype(np.int32)
    sel = selected(arg, used)
    assert sel.shape == (Tn, used) and (sel == arg[:, :used]).all()
    act = rng.standard_normal((Tn * used, n_ff)).astype(np.float32)      # 捕捉の形 (T·n_used, n_ff)
    a3 = per_token(act, used, Tn)
    assert a3.shape == (Tn, used, n_ff) and np.array_equal(a3[2, 1], act[2 * used + 1])
    w2 = rng.random((Tn, used)).astype(np.float32)
    assert per_token(w2, used, Tn) is w2                                  # [n_used, T] で出る重みはそのまま
    assert per_token(w2.reshape(-1, 1), used, Tn).shape == (Tn, used)
    sc = scatter(a3, sel, n_exp)
    for t in range(Tn):
        for j in range(used):
            assert np.array_equal(sc[t, sel[t, j]], a3[t, j])
        assert np.isnan(sc[t, [e for e in range(n_exp) if e not in sel[t]]]).all()   # 選ばれていない専門家は未観測
    parts = by_expert(a3, sel, n_exp)
    cnt, wsum = usage(sel, w2, n_exp)
    assert [len(p) for p in parts] == list(cnt) and cnt.sum() == Tn * used
    assert np.isclose(wsum.sum(), w2.sum())


# --- D. スケールパッチ(V20)-------------------------------------------------------------------------

PATCH_TYPES = ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K",
               "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_S", "IQ4_NL", "IQ4_XS"]


def _random_blocks(qt: T, n_blocks: int, rng) -> np.ndarray:
    """有効なブロック: ランダムなバイトに、有限の fp16 のスケールを書く(configs/quant_layout.yaml の位置)。"""
    from gne.patch import LAYOUT
    lay = LAYOUT[qt.name]
    b = rng.integers(0, 256, size=(n_blocks, lay["bytes"]), dtype=np.uint8)
    for s in lay["scales"]:
        b[:, s:s + 2] = rng.uniform(0.01, 1.0, n_blocks).astype("<f2").view(np.uint8).reshape(-1, 2)
    return b


def _synthetic_gguf(path: Path, rng) -> dict[str, np.ndarray]:
    """型ごとに (4 行, 512) の 2D テンソルと、(3 専門家, 2 行, 512) の 3D テンソル、F32 のバイアスを持つ GGUF。"""
    w = GGUFWriter(str(path), arch="llama")
    w.add_block_count(1)
    w.add_embedding_length(512)
    w.add_feed_forward_length(512)
    ne0, out = 512, {}
    for name in PATCH_TYPES:
        qt = T[name]
        bs, ts = GGML_QUANT_SIZES[qt]
        raw = _random_blocks(qt, 4 * ne0 // bs, rng).reshape(4, ne0 // bs * ts)
        w.add_tensor(f"t2.{name}", raw, raw_dtype=qt)
        raw3 = _random_blocks(qt, 6 * ne0 // bs, rng).reshape(3, 2, ne0 // bs * ts)
        w.add_tensor(f"t3.{name}", raw3, raw_dtype=qt)
        out[name] = raw
    for name, dt in (("F32", np.float32), ("F16", np.float16)):
        x = rng.standard_normal((4, ne0)).astype(dt)
        w.add_tensor(f"t2.{name}", x)
        out[name] = x
    w.add_tensor("bias", rng.standard_normal(8).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return out


def _deq(weights, name: str) -> np.ndarray:
    from gne.weights import tensor_to_f32
    t = weights.tensor(name)
    if t.tensor_type.name in ("F32", "F16"):
        return np.asarray(t.data, dtype=np.float32).copy()
    return gquants.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32) if t.tensor_type in gquants._type_traits \
        else tensor_to_f32(t)


def test_scale_patch_all_types(tmp_path):
    """V20: 行のスケールを k 倍 → dequantize が元の k 倍(fp16 のスケールの丸め以内)、k = 0 で全 0、ほかの行はビット単位で
    不変、revert で元のバイトに戻る。3D のスラブ(専門家)も同じ。F32 の要素の足し算。"""
    from gne.patch import PatchSet, Patcher, applied, make_work_copy
    from gne.weights import Weights
    rng = np.random.default_rng(3)
    src = tmp_path / "syn.gguf"
    _synthetic_gguf(src, rng)
    work = make_work_copy(src, tmp_path / "work.gguf")
    orig = src.read_bytes()
    w0 = Weights(src)
    pt = Patcher(w0)
    for name in PATCH_TYPES + ["F32", "F16"]:
        base2 = _deq(w0, f"t2.{name}")
        for k in (2.0, -1.0, 0.37, 0.0):
            ps = pt.scale_row(f"t2.{name}", 1, k)
            ps.name = f"{name}-{k}"
            ps.apply(work)
            got = _deq(Weights(work), f"t2.{name}")
            tol = 2e-3 if name not in ("F32",) else 1e-6
            assert np.allclose(got[1], k * base2[1], rtol=tol, atol=tol * np.abs(base2[1]).max()), (name, k)
            if k == 0:
                assert not got[1].any(), name
            assert np.array_equal(got[[0, 2, 3]], base2[[0, 2, 3]]), (name, k)          # ほかの行は不変
            ps.revert(work)
            assert work.read_bytes() == orig, (name, k)                                 # 元のバイトに戻る
        if name in ("F32", "F16"):
            continue
        base3 = _deq(w0, f"t3.{name}")
        ps = pt.scale_slab(f"t3.{name}", 1, 0.0)
        ps.name = f"slab-{name}"
        ps.apply(work)
        got = _deq(Weights(work), f"t3.{name}").reshape(base3.shape)
        assert not got[1].any() and np.array_equal(got[[0, 2]], base3[[0, 2]]), name
        ps.revert(work)
    # 要素の足し算(ルーターのバイアス)と、保存して読み直したパッチ
    b0 = _deq(w0, "bias")
    ps = pt.add_element("bias", 5, 2.5)
    ps.name = "bias"
    p = ps.save(tmp_path / "bias.patch.json")
    ps2 = PatchSet.load(p)
    ps2.apply(work)
    assert applied(work) == ["bias"]
    b1 = _deq(Weights(work), "bias")
    assert np.isclose(b1[5], b0[5] + 2.5) and np.array_equal(np.delete(b1, 5), np.delete(b0, 5))
    ps2.revert(work)
    assert applied(work) == [] and work.read_bytes() == orig
    # 別のパッチが掛かったままのバイトには書かない
    a = pt.scale_row("t2.Q8_0", 0, 2.0)
    a.name = "a"
    a.apply(work)
    work_bytes = bytearray(work.read_bytes())
    off = a.patches[0].offset
    work_bytes[off] ^= 0xFF
    work.write_bytes(bytes(work_bytes))
    with pytest.raises(GneError):
        a.revert(work)
    with pytest.raises(GneError):
        pt.scale_row("t2.Q8_0", 9, 2.0)                                                 # 行が範囲外


# --- モデルを使うもの(granite = GNE_TEST_MOE、LFM2 = GNE_TEST_MOE_BIAS)--------------------------------

def _session(path, tmp_path_factory, tag):
    if not path.exists():
        pytest.skip(f"MoE のモデルがありません({path})")
    from gne.api import Session
    return Session(path, root=tmp_path_factory.mktemp(tag), n_ctx=1024)


@pytest.fixture(scope="module")
def moe(tmp_path_factory):
    s = _session(MOE, tmp_path_factory, "moe")
    yield s
    s.close()


def _capture(s, ids, L, keys, lora=None):
    from gne.capture import Capture
    from gne.moe import moe_names
    e = s.engine
    nm = moe_names(e.list_names(ids, layer=L), L)
    want = {nm[k] for k in keys}
    cap = Capture(lambda n: n in want)
    ctx = e.probe_ctx(cap)
    e.clear_memory(ctx)
    e.set_cvec(ctx, None)
    e.set_lora(ctx, lora)
    try:
        e.prefill(ctx, [ids])
        out = cap.take()
    finally:
        e.set_lora(ctx, None)
    return nm, out


def _ids(s, text="Tell me about the history of the Roman empire."):
    return s.engine.tokenize_prompt(s.engine.render_chat([{"role": "user", "content": text}]))


@pytest.mark.model
def test_moe_arch_and_stats(moe):
    """住所・専門家の重み・専門家ごとの活性統計(n_routed、選ばれた重みの平均、よく選ばれるトークン)。"""
    from gne.stats import ActStats
    w = moe.weights
    li = w.arch.layers[5]
    assert li.ffn == "moe" and li.n_expert == 32 and li.n_expert_used == 8 and li.n_ff == 512 and li.routing == "softmax_topk"
    assert w.expert_down(5, 3).shape == (moe.meta.n_embd, 512) and w.expert_up(5, 3).shape == (512, moe.meta.n_embd)
    assert w.expert_down_norms(5).shape == (32, 512) and w.router(5).shape == (32, moe.meta.n_embd) and w.router_bias(5) is None
    with pytest.raises(GneError):
        moe.recipe(units=["5:100:+1"])                                  # MoE の層のニューロンは L:Ee:Nn
    with pytest.raises(GneError):
        moe.recipe(units=["5:E3:route:+1"])                             # granitemoe にルーターのバイアスは無い
    d = moe.stats_act(layers=[5], tokens=1500)
    row = d["layers"][0]
    assert row["tensor_name"] == "ffn_moe_swiglu-5" and row["moe"]["n_expert"] == 32
    a = ActStats.load(row["path"])
    assert a.is_moe and a.absmax.shape == (32, 512) and int(a.n_routed.sum()) == a.n_tokens * 8
    assert a.top_tokens.shape == (32, 20) and np.all((a.weight_mean > 0) | (a.n_routed == 0))


@pytest.mark.model
def test_expert_neuron_gain_lora(moe):
    """V19: 専門家ごとに積んだ LoRA で (e, n) が k 倍。ほかは不変、選ばれる専門家も変わらない。"""
    from gne.moe import per_token, scatter, selected
    L, ids = 5, _ids(moe)
    li = moe.weights.arch.layers[L]

    def acts(lora=None):
        nm, o = _capture(moe, ids, L, ("act", "argsort"), lora)
        return scatter(per_token(o[nm["act"]], li.n_expert_used, len(ids)), selected(o[nm["argsort"]], li.n_expert_used), li.n_expert)
    a0 = acts()
    obs = ~np.isnan(a0[:, :, 0])
    ex = int(np.argsort(-obs.sum(0))[1])
    n = int(np.nanargmax(np.nanmean(np.abs(a0[:, ex]), 0)))
    for k in (0.0, 2.0, -1.0):
        a1 = acts(moe.lora_for(moe.recipe(units=[f"{L}:E{ex}:N{n}:x{k:g}"])))
        assert np.array_equal(~np.isnan(a1[:, :, 0]), obs)
        sel = obs[:, ex]
        assert np.abs(a1[sel, ex, n] - k * a0[sel, ex, n]).max() < 0.03 * np.abs(a0[sel, ex, n]).max(), k
        rest = a1.copy()
        rest[sel, ex, n] = a0[sel, ex, n]
        assert np.array_equal(np.nan_to_num(rest), np.nan_to_num(a0)), k


@pytest.mark.model
def test_expert_ablate_patch(moe):
    """V21: 専門家のスラブを 0 にするパッチ → その専門家の寄与がちょうど 0、出力の差は元の寄与。revert で元に戻る。"""
    from gne.moe import per_token, selected
    L, ids = 5, _ids(moe)
    li = moe.weights.arch.layers[L]
    T = len(ids)
    nm, o0 = _capture(moe, ids, L, ("argsort", "weighted", "out"))
    sel0 = selected(o0[nm["argsort"]], li.n_expert_used)
    wt0 = per_token(o0[nm["weighted"]], li.n_expert_used, T)
    ex = int(np.bincount(sel0.reshape(-1)).argmax())
    r = moe.recipe(units=[f"{L}:E{ex}:x0"])
    assert moe.lora_for(r) is None
    ps = moe.patches_for(r)
    assert ps.ops == [{"op": "scale_slab", "tensor": f"blk.{L}.ffn_down_exps.weight", "expert": ex, "k": 0.0}]
    with moe.patched(ps):
        _, o1 = _capture(moe, ids, L, ("argsort", "weighted", "out"))
        assert moe.last_patch_reload_s is not None
    sel1 = selected(o1[nm["argsort"]], li.n_expert_used)
    wt1 = per_token(o1[nm["weighted"]], li.n_expert_used, T)
    assert np.array_equal(sel0, sel1)
    assert not wt1[sel1 == ex].any() and np.array_equal(wt1[sel1 != ex], wt0[sel0 != ex])
    contrib = np.stack([wt0[t][sel0[t] == ex].sum(0) if (sel0[t] == ex).any() else np.zeros(wt0.shape[2]) for t in range(T)])
    assert np.allclose(o0[nm["out"]] - o1[nm["out"]], contrib, atol=1e-5)
    _, o2 = _capture(moe, ids, L, ("out",))
    assert np.allclose(o2[nm["out"]], o0[nm["out"]], atol=1e-5)          # 読み直したエンジンなので GPU の丸めの差だけ
    from gne.patch import applied
    assert applied(moe.store.work_path(moe.meta.model_id)) == []
    d = moe.steer(r, prompt="Hello, who are you?", max_tokens=6)                   # steer は専門家のパッチを当てて生成する
    assert d["steered"]["text"] and d["baseline"]["text"]


@pytest.mark.model
def test_moe_add_is_a_direction_push(moe, tmp_path):
    """V22: MoE のニューロンの add は警告と semantics: direction。export-cvec も警告し、gain と route は入れない。"""
    moe.warnings = []
    r = moe.recipe(units=["5:E3:N100:+0.8", "5:E3:x0"], dose_unit="raw")
    assert any("書き込み方向の押し" in w for w in moe.warnings)
    _, det = moe.vectors(r)
    assert det[0]["semantics"] == "direction" and det[0]["expert"] == 3 and det[1]["mode"] == "gain"
    moe.warnings = []
    out = moe.export_cvec(r, tmp_path / "x.cvec.gguf")
    assert Path(out["path"]).exists()
    assert any("書き込み方向の押し" in w for w in moe.warnings) and any("スケールパッチ" in w for w in moe.warnings)


@pytest.mark.model
def test_router_bias_route(tmp_path_factory):
    """V23(lfm2moe): ルーターのバイアスに ±δ → 専門家の選択率が 1 / 0 へ。選び方が同じトークンの重みは変わらない。"""
    from gne.moe import per_token, selected
    s = _session(MOE_BIAS, tmp_path_factory, "moe_bias")
    try:
        L, ids = 6, _ids(s, "Tell me about the history of the Roman empire, the senate and the emperors.")
        li = s.weights.arch.layers[L]
        assert li.has_router_bias and li.routing == "sigmoid_topk" and s.weights.arch.layers[0].ffn == "dense"
        T = len(ids)

        def route():
            nm, o = _capture(s, ids, L, ("argsort", "weights"))
            return selected(o[nm["argsort"]], li.n_expert_used), per_token(o[nm["weights"]], li.n_expert_used, T)
        sel0, w0 = route()
        rate = np.array([(sel0 == e).any(1).mean() for e in range(li.n_expert)])
        lo, hi = int(np.argmin(rate)), int(np.argmax(rate))
        with s.patched(s.patches_for(s.recipe(units=[f"{L}:E{lo}:route:+5"]))):
            sel1, _ = route()
        assert (sel1 == lo).any(1).mean() > 0.9
        with s.patched(s.patches_for(s.recipe(units=[f"{L}:E{hi}:route:-5"]))):
            sel2, w2 = route()
        assert not (sel2 == hi).any()
        same = (sel2 == sel0).all(1)
        if same.any():
            assert np.allclose(np.sort(w2[same], 1), np.sort(w0[same], 1))
    finally:
        s.close()


@pytest.mark.model
def test_experts_contrast_sweep_route(moe, tmp_path):
    """E.1 専門家の対照分析(中立どうしの床・z・null_pct)、調書、E.2 専門家の sweep、moe_route、fires の routed。"""
    import yaml

    from gne.schemas import COMMAND_SCHEMAS
    c = tmp_path / "pirate.yaml"
    c.write_text(yaml.safe_dump({"name": "pirate", "system_a": "You are a pirate. Always talk like a pirate, with arr and matey.",
                                 "system_b": "You are a helpful assistant.", "user_prompts": ["Who are you?", "Tell me about the weather."],
                                 "validate": {"match": "arr|matey", "target": "a pirate"}}))
    d = moe.experts_contrast(c, max_tokens=12)
    li = moe.weights.arch.layers[5]
    assert d["n_experts"] == len(d["layers"]) * li.n_expert and d["n_prompts"] == 2
    assert set(d["floors"]) == {"usage_shift", "weight_shift", "contrib_cos", "contrib_norm"}
    top = d["top"][0]
    assert top["z"] >= d["top"][-1]["z"] and 0 <= top["null_pct"] <= 1
    COMMAND_SCHEMAS["experts contrast"].model_validate({**moe.envelope("experts contrast"), **d})
    ex = moe.expert(top["layer"], top["expert"])
    assert ex["contrasts"][0]["name"] == "pirate" and ex["contrasts"][0]["rank"] == 1 and len(ex["neurons"]) == 20
    COMMAND_SCHEMAS["experts show"].model_validate({**moe.envelope("experts show"), **ex})
    sw = moe.experts_sweep(5, experts=[0, 1], ks=(0.0,), max_tokens=8)
    assert sw["n_experts"] == 2 and {u["expert"] for u in sw["top"]} <= {0, 1} and sw["reload_s"]["mean"] is not None
    COMMAND_SCHEMAS["experts sweep"].model_validate({**moe.envelope("experts sweep"), **sw})
    r = moe.moe_route("The pirate sailed across the sea.", layers=[5, 12])
    assert set(r["layers"]) == {"5", "12"} and len(r["layers"]["5"]) == r["n_tokens"]
    assert all(len(x["experts"]) == li.n_expert_used and abs(sum(x["weights"]) - 1) < 1e-3 for x in r["layers"]["5"])
    f = moe.fires("The pirate sailed across the sea.", top=5)
    assert "routed" in f and len(f["routed"]["5"]) == f["n_tokens"] - 1
    hit = next(x for row in f["top"] for x in row["neurons"] if "expert" in x)
    assert hit["neuron"] == hit["expert"] * li.n_ff + hit["expert_neuron"]


@pytest.mark.model
def test_reverse_and_sweep_on_moe_layers(moe, tmp_path):
    """E.3: 逆引きは全専門家のニューロン(通し番号)を並べ、専門家とその中の番号を付ける。書き込みが 0 の列は NaN にしない。
    MoE の層の sweep は通し番号で動き、semantics: direction。"""
    import yaml
    c = tmp_path / "pirate.yaml"
    c.write_text(yaml.safe_dump({"name": "pirate", "system_a": "You are a pirate. Talk like a pirate.", "system_b": "You are a helpful assistant.",
                                 "user_prompts": ["Who are you?", "Tell me about the weather."], "validate": {"match": "arr", "target": "a pirate"}}))
    d = moe.reverse(contrast=c, top=30, layers=[4, 5, 12])
    assert all(x["cos"] == x["cos"] for x in d["candidates"])                  # NaN が無い
    cd = d["candidates"][0]
    n_ff = moe.weights.arch.layers[cd["layer"]].n_ff
    assert cd["expert"] is not None and cd["neuron"] == cd["expert"] * n_ff + cd["expert_neuron"] and d["by_expert"]
    sw = moe.sweep(cd["layer"], neurons=str(cd["neuron"]), design="quick", random_controls=1, max_tokens=8)
    assert sw["semantics"] == "direction" and sw["n_units"] == 2
    nd = moe.neuron(cd["layer"], cd["neuron"])
    assert nd["expert"] == cd["expert"] and nd["semantics"] == "direction" and nd["address"].startswith(f"L{cd['layer']} E")


def test_gain_bake_by_scale_patch(tmp_path):
    """D: gne gain --bake の既定(auto)は、行の k 倍だけで書けるならスケールパッチ(量子化を保ったまま)。ヘッドがあれば q8_0。
    明示した q8_0 / f16 は従来どおり。"""
    from gne.gain import bake_gain_gguf
    from gne.weights import Weights
    rng = np.random.default_rng(5)
    src = tmp_path / "m.gguf"
    w = GGUFWriter(str(src), arch="llama")
    w.add_block_count(2)
    w.add_embedding_length(256)
    w.add_feed_forward_length(64)
    for L in range(2):
        up = rng.standard_normal((64, 256)).astype(np.float32)
        w.add_tensor(f"blk.{L}.ffn_up.weight", gquants.quantize(up, T.Q8_0), raw_dtype=T.Q8_0)
        w.add_tensor(f"blk.{L}.ffn_down.weight", rng.standard_normal((256, 64)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    wt = Weights(src)
    r = Recipe([parse_unit("1:5:x2"), parse_unit("1:9:x0")], name="g")
    d = bake_gain_gguf(wt, r, tmp_path / "g.gguf", requant="auto")
    assert d["method"] == "patch" and d["modified"][0]["to"] == "Q8_0" and d["patch_bytes"] == 2 * 64 // 32 * 34 * 4
    w2 = Weights(tmp_path / "g.gguf")
    u0, u1 = wt.up_matrix(1), w2.up_matrix(1)
    assert np.allclose(u1[5], 2 * u0[5], rtol=2e-3, atol=2e-3 * np.abs(u0[5]).max()) and not u1[9].any()
    assert np.array_equal(np.delete(u1, [5, 9], 0), np.delete(u0, [5, 9], 0)) and np.array_equal(w2.up_matrix(0), wt.up_matrix(0))
    assert (tmp_path / "g.gguf.patch.json").exists()
    q = bake_gain_gguf(wt, r, tmp_path / "q.gguf", requant="q8_0")
    assert "method" not in q


@pytest.mark.model
def test_shared_expert_qwen2moe(tmp_path_factory):
    """shared expert のある MoE(qwen2moe、GNE_TEST_MOE_SHARED): 統計は MoE の側を取る(shared expert の FFN は dense と同じ
    ffn_swiglu-L で出る)。shared expert のニューロンの gain(LoRA を ffn_up_shexp に)と add(方向押し)。"""
    s = _session(MOE_SHARED, tmp_path_factory, "moe_shared")
    try:
        L = 12
        li = s.weights.arch.layers[L]
        assert li.ffn == "moe+shared" and li.n_ff_shexp > 0 and not li.has_router_bias
        assert not [w for w in s.weights.check_arch() if "ffn_down.weight" in w]
        d = s.stats_act(layers=[L], tokens=600)
        assert d["layers"][0]["tensor_name"].startswith("ffn_moe_") and d["layers"][0]["moe"]["n_expert"] == li.n_expert
        ids = _ids(s)
        _, o = _capture(s, ids, L, ("argsort",))
        from gne.capture import Capture
        e = s.engine

        def shared(lora=None):
            cap = Capture(lambda n: n == f"ffn_swiglu-{L}")
            ctx = e.probe_ctx(cap)
            e.clear_memory(ctx)
            e.set_cvec(ctx, None)
            e.set_lora(ctx, lora)
            try:
                e.prefill(ctx, [ids])
                return cap.take()[f"ffn_swiglu-{L}"]
            finally:
                e.set_lora(ctx, None)
        a0 = shared()
        assert a0.shape == (len(ids), li.n_ff_shexp)
        n = int(np.argmax(np.abs(a0).mean(0)))
        a1 = shared(s.lora_for(s.recipe(units=[f"{L}:S:N{n}:x2"])))
        assert np.abs(a1[:, n] - 2 * a0[:, n]).max() < 0.03 * np.abs(a0[:, n]).max()
        assert np.array_equal(np.delete(a1, n, 1), np.delete(a0, n, 1))
        _, det = s.vectors(s.recipe(units=[f"{L}:S:N{n}:+0.5"]))
        assert det[0]["semantics"] == "direction" and det[0]["norm"] > 0
        assert s.weights.n_units(L) == li.n_expert * li.n_ff                       # shared expert は通し番号の外
    finally:
        s.close()

