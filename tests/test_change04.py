"""CHANGE-04 のテスト: attention ヘッドの住所・捕捉・対照分析・ゲイン・反応性。

モデルを使うものは Qwen3-0.6B(GNE_TEST_MODEL、attention だけ)。無ければ飛ばす。
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from gne.steer import GneError, Recipe, parse_unit

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
REACT = ROOT / "configs" / "contrast" / "yandere_react.yaml"


def _tiny_heads(path, *, n_head=4, head_dim=3, n_embd=16):
    """attention の層 2 つ(attn_q・attn_output)と ffn_down を持つ小さな GGUF(llama arch、ヘッド 4 個 × 幅 3)。"""
    from gguf import GGUFWriter
    rng = np.random.default_rng(1)
    w = GGUFWriter(str(path), arch="llama")
    w.add_name("tiny")
    w.add_block_count(2)
    w.add_embedding_length(n_embd)
    w.add_feed_forward_length(24)
    w.add_head_count(n_head)
    w.add_head_count_kv(2)
    w.add_token_list([f"t{i}" for i in range(8)])
    mats = {}
    for L in range(2):
        for nm, shape in (("attn_q", (n_head * head_dim, n_embd)), ("attn_output", (n_embd, n_head * head_dim)),
                          ("ffn_down", (n_embd, 24))):
            m = rng.standard_normal(shape).astype(np.float32)
            mats[f"blk.{L}.{nm}.weight"] = m
            w.add_tensor(f"blk.{L}.{nm}.weight", m)
    w.add_tensor("token_embd.weight", rng.standard_normal((8, n_embd)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return mats


# --- モデル不要 ----------------------------------------------------------------------------------------

def test_head_unit_and_column_block(tmp_path):
    """住所 L:Hh:xk と W_O の列ブロックの対応。レシピの保存・読み込み・倍率。"""
    from gne.weights import Weights
    u = parse_unit("1:H2:x0")
    assert (u.layer, u.kind, u.head, u.mode, u.k, u.neuron) == (1, "head", 2, "gain", 0.0, None) and u.label() == "L1 H2 ×0"
    assert parse_unit("1:h3:x2.5").head == 3
    with pytest.raises(GneError):
        parse_unit("1:H2:+0.5")                                   # ヘッドは add にできない
    r = Recipe([u, parse_unit("0:5:+0.3")], name="h")
    r.save(tmp_path / "r.yaml")
    d = yaml.safe_load((tmp_path / "r.yaml").read_text())
    assert d["units"][0] == {"layer": 1, "kind": "head", "mode": "gain", "k": 0.0, "head": 2}
    back = Recipe.load(tmp_path / "r.yaml")
    assert back.units[0] == u and back.head_units == [u] and back.scaled(0.5).units[0].k == 0.5
    with pytest.raises(GneError):
        Recipe([parse_unit("1:H2:x0").__class__(1, None, 0.3, "head", head=2)]).validate()   # add のヘッド
    mats = _tiny_heads(tmp_path / "m.gguf")
    w = Weights(tmp_path / "m.gguf")
    li = w.arch.layers[1]
    assert (li.mixer, li.n_head, li.n_head_kv, li.head_dim, li.head_writer) == ("attention", 4, 2, 3, "attn_output")
    assert li.head_out == ("attn_gated", "kqv_out")
    assert w.head_block(1, 2) == ("blk.1.attn_output.weight", 6, 9)
    assert np.allclose(w.head_write_matrix(1, 2), mats["blk.1.attn_output.weight"][:, 6:9])
    with pytest.raises(GneError):
        w.head_block(1, 4)


def test_gqa_group():
    from gne.attn import kv_group
    assert [kv_group(h, 16, 8) for h in range(16)] == [h // 2 for h in range(16)]
    assert [kv_group(h, 16, 4) for h in (0, 3, 4, 15)] == [0, 0, 1, 3]
    assert kv_group(5, 8, 8) == 5
    with pytest.raises(GneError):
        kv_group(0, 6, 4)


def test_head_lora_shapes_and_math(tmp_path):
    """LoRA の A は列ブロックを選ぶ単位行列の断片、B は (k − 1)·W_O[:, blk]。W + BA で列ブロックだけが k 倍。GGUF の形。"""
    from gguf import GGUFReader

    from gne.gain import export_gain_lora, head_gain_by_layer, head_lora_tensors, lora_cache_path
    from gne.weights import Weights
    mats = _tiny_heads(tmp_path / "m.gguf")
    w = Weights(tmp_path / "m.gguf")
    r = Recipe([parse_unit("1:H2:x0"), parse_unit("1:H0:x3"), parse_unit("0:H1:x-1"), parse_unit("1:H2:x2")])
    hb = head_gain_by_layer(r.units)
    assert hb == {0: [(1, -1.0)], 1: [(0, 3.0), (2, 0.0)]}                       # 同じヘッドは掛け合わせる(0 × 2)
    t = head_lora_tensors(w, hb)
    A, B = t["blk.1.attn_output.weight"]
    assert A.shape == (6, 12) and B.shape == (16, 6)                             # rank = ヘッド数 × head_dim
    W = mats["blk.1.attn_output.weight"]
    ref = W.copy()
    ref[:, 0:3] *= 3.0
    ref[:, 6:9] *= 0.0
    assert np.allclose(W + B @ A, ref, atol=1e-5)
    rep = export_gain_lora(w, r, tmp_path / "h.lora.gguf")
    assert rep["heads"] == {"0": [[1, -1.0]], "1": [[0, 3.0], [2, 0.0]]} and rep["head_rank"] == {"0": 3, "1": 6}
    tt = {x.name: x for x in GGUFReader(tmp_path / "h.lora.gguf").tensors}
    assert [int(x) for x in tt["blk.1.attn_output.weight.lora_a"].shape] == [12, 6]       # ggml ne [n_in, r]
    assert [int(x) for x in tt["blk.1.attn_output.weight.lora_b"].shape] == [6, 16]       # ggml ne [r, n_embd]
    # 実行時の LoRA の置き場所: ヘッドが無いレシピは前と同じ名前、ヘッドがあれば別
    neu = Recipe([parse_unit("0:5:x2")])
    assert lora_cache_path(tmp_path, neu) != lora_cache_path(tmp_path, Recipe(neu.units + [parse_unit("1:H2:x0")]))


def test_parse_heads(tmp_path):
    """Listen のヘッドの書き方: L:Hh・L:Ha-b・L:H*、重なりは 1 つに、範囲外はエラー。"""
    from gne.heads import parse_heads
    from gne.weights import Weights
    _tiny_heads(tmp_path / "m.gguf")
    w = Weights(tmp_path / "m.gguf")
    assert parse_heads("1:H2", w) == [(1, 2)]
    assert parse_heads("1:H0-2, 1:h1, 0:H*", w) == [(1, 0), (1, 1), (1, 2), (0, 0), (0, 1), (0, 2), (0, 3)]
    for bad in ("1:H4", "1:H3-1", "2:H0", "1:2", ""):
        with pytest.raises(GneError):
            parse_heads(bad, w)


def test_spans_to_tokens():
    from gne.attn import find_spans, spans_to_tokens
    text = "I had lunch with Emily. 美咲さんと"
    sp = find_spans(text, ["Emily", "re:美.", "lunch with"])
    assert sp == [(6, 16), (17, 22), (24, 26)]
    offs = [(0, 1), (1, 5), (5, 11), (11, 16), (16, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 26), (26, 28)]
    assert spans_to_tokens(offs, sp) == [2, 3, 4, 7, 8]                          # 空の区間(9)は入れない
    assert spans_to_tokens(offs, []) == [] and find_spans("abc", []) == []
    assert find_spans("aaa", ["aa"]) == [(0, 3)]                                 # 重なりはまとめる


def test_reactivity_scoring(tmp_path, monkeypatch):
    """反応性 = トリガーありの一致率 − なしの一致率。対集合の読み込みの誤り。"""
    import gne.evalx as ev
    rc = ev.Reactivity.load(REACT)
    assert rc.name == "yandere_react" and len(rc.pairs) == 10 and rc.pairs[0][0].startswith("I had lunch with Emily")
    answers = {p: ("Who is she, darling?" if i % 3 else "Nice!") for i, (p, _) in enumerate(rc.pairs)}

    def fake_generate(session, items, vecs, lora, when, tick):
        tick(len(items))
        return [{"text": answers.get(p, "Sounds fun."), "token_ids": [1]} for p, _ in items]
    monkeypatch.setattr(ev, "_generate", fake_generate)
    n = []
    d = ev.reactivity_score(None, rc, None, "all", 32, n.append)
    assert d["with"] == pytest.approx(6 / 10) and d["without"] == 0.0 and d["reactivity"] == pytest.approx(0.6)
    assert sum(n) == 20 and len(d["samples"]) == 3
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"name": "b", "system_a": "A", "system_b": "B", "user_prompts": ["x"],
                                   "reactivity": [{"with": "a", "without": "b"}]}))
    with pytest.raises(GneError):
        ev.Reactivity.load(bad)                                                  # validate.match が無い


def test_head_controls_and_null_pct(tmp_path):
    """ヘッドのランダム対照: 同じ k・同じ種類の別のヘッド、seed 固定。null_pct は CHANGE-03 と同じ定義。"""
    from gne.controls import head_control_recipes, null_pct
    from gne.weights import Weights
    _tiny_heads(tmp_path / "m.gguf")

    class S:
        weights = Weights(tmp_path / "m.gguf")
    r = Recipe([parse_unit("1:H2:x0"), parse_unit("0:H1:x2"), parse_unit("0:5:+0.3")], name="r")
    out = head_control_recipes(S, r, 5)
    assert len(out) == 5 and [c.name for c in out] == [f"r-randhead{j}" for j in range(5)]
    for c in out:
        hs = [u for u in c.units if u.kind == "head"]
        assert [u.k for u in hs] == [0.0, 2.0] and c.units[2] == r.units[2]
        assert all((u.layer, u.head) not in {(1, 2), (0, 1)} for u in hs) and len({(u.layer, u.head) for u in hs}) == 2
    again = head_control_recipes(S, r, 5)
    assert [c.units for c in again] == [c.units for c in out]
    assert head_control_recipes(S, Recipe([parse_unit("0:5:+0.3")]), 3) == []
    assert null_pct(0.3, [0.0, 0.1, 0.2, 0.5]) == 0.75


# --- モデルを使う ----------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def session(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip("モデルがありません")
    from gne.api import Session
    s = Session(MODEL, root=tmp_path_factory.mktemp("c04"))
    yield s
    s.close()


@pytest.mark.model
def test_capture_v16_and_head_gain_v15(session):
    """V16: パターンの行の和 ≈ 1・因果マスク・flash attention を切っても greedy が同じ。V15: ヘッド gain の LoRA の差が予測と合う。"""
    from gne.attn import capture_attention, check_capture, check_head_gain
    v16 = check_capture(session)
    assert v16["ok"] and v16["row_sum_err_max"] < 1e-4 and v16["future_max"] < 1e-6 and v16["greedy_match"]
    ac = capture_attention(session, ["Hello there, how are you?"], layers=[3, 5], want_qkv=True)[0]
    T = ac.n_tokens
    assert ac.pattern[3].shape == (16, T, T) and ac.head_out[5].shape == (T, 16, 128)
    assert ac.qkv[3]["Qcur"].shape == (T, 16, 128) and ac.qkv[3]["Kcur"].shape == (T, 8, 128)
    assert "".join(ac.text[a:b] for a, b in ac.offsets) == ac.text
    v15 = check_head_gain(session, 5, 3)
    assert v15["ok"] and [r["k"] for r in v15["rows"]] == [0.0, 2.0, -1.0]
    assert all(r["rel_err_median"] < 0.03 for r in v15["rows"])


@pytest.mark.model
def test_heads_contrast_attention_and_dossier(session, tmp_path):
    from gne.schemas import COMMAND_SCHEMAS
    d = session.heads_contrast(REACT, layers=[2, 9, 13], max_tokens=16)
    assert d["n_heads"] == 48 and d["n_with_triggers"] == 10 and set(d["floors"]) >= {"trigger_mass", "pattern_shift", "out_shift"}
    assert all(0 <= x["null_pct"] <= 1 for x in d["top"]) and d["top"][0]["z"] >= d["top"][-1]["z"]
    assert any(" Emily" in s["triggers"] or "Emily" in "".join(s["triggers"]) for s in d["samples"])
    COMMAND_SCHEMAS["heads contrast"].model_validate({**session.envelope("heads contrast"), **d})
    top = d["top"][0]
    a = session.attention("I had lunch with Emily from work today.", layer=top["layer"], head=top["head"], top=3)
    assert len(a["heads"]) == 1 and len(a["heads"][0]["top"]) == 3 and len(a["pattern"]) == a["n_tokens"]
    assert abs(sum(a["pattern"][a["pos"]]) - 1) < 1e-3
    COMMAND_SCHEMAS["heads attention"].model_validate({**session.envelope("heads attention"), **a})
    session.set_head_name(top["layer"], top["head"], "who-is-she")
    hd = session.head(top["layer"], top["head"])
    assert hd["name"] == "who-is-she" and hd["contrasts"][0]["name"] == "yandere_react" and hd["contrasts"][0]["rank"] == 1
    assert hd["kv_group"] == top["head"] // 2 and len(hd["logit_lens"]["components"]) == 2
    COMMAND_SCHEMAS["heads show"].model_validate({**session.envelope("heads show"), **hd})


@pytest.mark.model
def test_eval_reactivity_with_head_controls(session, tmp_path):
    from gne.schemas import COMMAND_SCHEMAS
    (tmp_path / "ppl").mkdir()
    (tmp_path / "ppl" / "en.txt").write_text("Alice was beginning to get very tired of sitting by her sister on the bank. " * 2)
    suite = tmp_path / "mini.yaml"
    suite.write_text(yaml.safe_dump({"name": "mini", "max_tokens": {"exact": 8, "format": 8, "mix": 8, "persona": 16},
                                     "exact": [{"id": "add", "prompt": "What is 17 + 25? Answer with just the number.",
                                                "answer": ["42"]}],
                                     "persona": {"prompt_sets": ["en"]}, "ppl": [{"id": "en", "path": "ppl/en.txt"}]}))
    r = session.recipe(units=["9:H6:x0", "13:H3:x2"], name="heads")
    d = session.eval(r, doses=[1.0], suite=suite, reactivity=REACT, head_controls=2)
    assert d["reactivity"]["n_pairs"] == 10 and d["reactivity"]["head_controls"] == 2
    b, top = d["rows"]
    assert b["reactivity"]["increase"] == 0 and "ctrl" not in b["reactivity"]
    assert top["reactivity"]["ctrl"]["n"] == 2 and 0 <= top["reactivity"]["null_pct"] <= 1
    assert -1 <= top["reactivity"]["reactivity"] <= 1
    COMMAND_SCHEMAS["eval"].model_validate({**session.envelope("eval"), **d})
    st = session.steer(r, prompt="I had lunch with Emily today.", max_tokens=12)
    assert st["units"][0]["kind"] == "head" and st["units"][0]["head"] == 6
    assert json.loads(json.dumps(session.head(13, 3, lens=False)["gain_k"])) == [2.0]


def test_query_unit_recipe(tmp_path):
    """(実験的、E)query のユニット: YAML の scale、倍率、− の列では外す、gain の LoRA には入らない。"""
    from gne.chat import _with_dose
    from gne.gain import gain_by_layer, head_gain_by_layer
    p = tmp_path / "q.yaml"
    p.write_text(yaml.safe_dump({"name": "q", "units": [
        {"layer": 3, "kind": "head", "head": 2, "mode": "query", "direction": "yandere", "seek": "yandere_react", "scale": 2},
        {"layer": 3, "kind": "head", "head": 1, "mode": "gain", "k": 0}]}))
    r = Recipe.load(p)
    q = r.query_units[0]
    assert (q.mode, q.head, q.k, q.direction, q.seek) == ("query", 2, 2.0, "yandere", "yandere_react")
    assert r.scaled(0.5).query_units[0].k == 1.0 and r.scaled(0).query_units[0].k == 0.0
    assert head_gain_by_layer(r.units) == {3: [(1, 0.0)]} and gain_by_layer(r.units) == {}
    assert _with_dose(r, None, -1).query_units[0].k == 0.0 and _with_dose(r, None, 1).query_units[0].k == 2.0
    r.save(tmp_path / "back.yaml")
    d = yaml.safe_load((tmp_path / "back.yaml").read_text())
    assert d["units"][0]["scale"] == 2.0 and "k" not in d["units"][0] and Recipe.load(tmp_path / "back.yaml").units == r.units
    with pytest.raises(GneError):
        Recipe([parse_unit("3:H2:x0").__class__(3, None, 0.0, "head", mode="query", k=1.0, head=2)]).validate()  # seek なし


@pytest.mark.model
def test_query_v17_on_small_model(session, tmp_path):
    """(実験的、E)V17 が動き、scale 0 は無介入と同じ。Qwen3 は RoPE が全次元なので、u は周波数の低い対に絞る。"""
    from gne.heads import q_rows, static_dims
    assert q_rows(session.weights, 9, 6) == (768, 896)
    dims = static_dims(session.weights, 9)
    assert len(dims) == 32 and dims[0] == 48 and dims[-1] == 127
    session.direction(contrast=REACT)
    d = session.heads_query(9, 6, direction="yandere_react", seek=REACT, scales=(0, 1))
    assert d["rows"][0]["others_shift_max"] == 0 and abs(d["v_mean_a"] - 1) < 1e-6 and abs(d["v_mean_b"]) < 1e-6
    assert d["rows"][1]["others_shift_max"] < 1e-6                          # 同じ層の他のヘッドは変わらない
    from gne.steer import Unit
    rq = Recipe([Unit(9, None, 0.0, "head", mode="query", k=1.0, direction="yandere_react", head=6, seek=str(REACT))],
                session.meta.model_id, name="q")
    g = session.gain(rq, tmp_path / "q.lora.gguf")
    assert len(g["query_loras"]) == 1 and Path(g["query_loras"][0]).exists()
    st = session.steer(rq, prompt="I had lunch with Emily today.", max_tokens=8)
    assert st["units"][0]["mode"] == "query"


@pytest.mark.model
def test_heads_listen_decomposition(session, monkeypatch):
    """Listen: 注目の行の和 = 1、分解(注目 × V × W_O)を全ヘッドで足し直すと attention の出力に戻る(check_rel_err)、
    全ヘッド・全トークンの寄与 = 出力の平均の方向成分、system とユーザーの発話のトークンに役が付く。"""
    from gne.schemas import COMMAND_SCHEMAS
    rng = np.random.default_rng(0)
    vec = {L: rng.standard_normal(session.meta.n_embd).astype(np.float32) for L in range(session.meta.n_layer)}
    monkeypatch.setattr(session, "direction_vec", lambda name, L: vec[L])
    d = session.heads_listen("I'm going to a concert with Jake this weekend.", heads="5:H*,9:H3", system="You are a jealous girlfriend.",
                             direction="rand", max_tokens=12)
    assert [h["layer"] for h in d["heads"]].count(5) == 16 and {"layer": 9, "head": 3} in d["heads"]
    for k, a in d["attn_by_head"].items():
        assert abs(sum(a) - 1.0) < 1e-3, k                                       # 返事の各行の和 1 の平均
    info = d["layers"]["5"]
    assert info["check_rel_err"] is not None and info["check_rel_err"] < 0.03
    assert abs(sum(info["contrib_by_head"]) - info["proj_mean"]) < 1e-4 * max(1.0, abs(info["proj_mean"]))
    assert abs(sum(d["contrib_by_head"]["5:H0"]) - info["contrib_by_head"][0]) < 1e-5
    assert d["layers"]["9"]["check_rel_err"] < 0.03
    roles = d["roles"]
    assert "system" in roles and "user" in roles and roles[d["reply_start"]:] == ["reply"] * (d["n_tokens"] - d["reply_start"])
    toks = "".join(t for t, r in zip(d["tokens"], roles) if r == "user")
    assert "concert with Jake" in toks
    COMMAND_SCHEMAS["heads listen"].model_validate({**session.envelope("heads listen"), **d})
    d2 = session.heads_listen("Hello there.", heads="5:H1", reply="Hi! Who is Jake?")
    assert d2["contrib"] is None and d2["reply"].strip() == "Hi! Who is Jake?" and len(d2["top_attn"]) > 0
