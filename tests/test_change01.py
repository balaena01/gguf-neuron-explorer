"""CHANGE-01 のテスト: 位置別ステアリング when(§6.15)、gain(§6.16)、eval(§6.14)。

モデルが要るものは GNE_TEST_MODEL(既定 models/Qwen3-0.6B-Q8_0.gguf)を使う。
"""

import json
import os
from pathlib import Path

import pytest
import yaml

from gne.steer import GneError, Recipe, Unit
from gne.when import When, _merge, parse_when

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
needs_model = [pytest.mark.model, pytest.mark.skipif(not MODEL.exists(), reason=f"{MODEL} がありません")]


# --- when: モデル不要 ---------------------------------------------------------------------------------

def test_parse_when():
    assert parse_when(None) == When("all") and parse_when("gen") == When("gen")
    assert parse_when("assistant").kind == "assistant"
    assert parse_when("after:結論:") == When("after", text="結論:")
    assert parse_when("ramp:1.5-0") == When("ramp", a=1.5, b=0.0)
    assert str(parse_when("ramp:1-0.25")) == "ramp:1-0.25"
    for bad in ("after:", "ramp:x", "ramp:1", "sometimes"):
        with pytest.raises(GneError):
            parse_when(bad)


def test_merge_segments():
    assert _merge([(0, 5, False), (5, 9, False), (9, 12, True), (12, 12, True), (12, 20, True)]) == \
        [(0, 9, False), (9, 20, True)]


def test_recipe_when_roundtrip(tmp_path):
    r = Recipe([Unit(3, 7, 1.0)], "m", "rel", "x", when="gen")
    r.save(tmp_path / "r.yaml")
    d = yaml.safe_load((tmp_path / "r.yaml").read_text(encoding="utf-8"))
    assert d["when"] == "gen"
    back = Recipe.load(tmp_path / "r.yaml")
    assert back.when == "gen" and back.scaled(0.5).when == "gen" and back.to_dict()["when"] == "gen"
    Recipe([Unit(3, 7, 1.0)]).save(tmp_path / "a.yaml")                       # all は書かない(前の形式と同じ)
    assert "when" not in yaml.safe_load((tmp_path / "a.yaml").read_text(encoding="utf-8"))
    (tmp_path / "bad.yaml").write_text("units: [{layer: 3, neuron: 7, dose: 1.0}]\nwhen: sometimes\n", encoding="utf-8")
    with pytest.raises(GneError):
        Recipe.load(tmp_path / "bad.yaml")


# --- when: 実モデル -----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def session(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip("モデルがありません")
    from gne.api import Session
    s = Session(MODEL, root=tmp_path_factory.mktemp("c01"))
    yield s
    s.close()


@pytest.mark.model
def test_v12_when_gen(session):
    from gne.verify import Verifier
    r = Verifier(session.engine, session.weights).v12_when_gen()
    assert r.status == "pass", r.detail
    assert r.data["prefill_diff_max"] == 0 and r.data["err_gen"] < 1e-3 and r.data["first_token_same"]


@pytest.mark.model
def test_when_modes(session):
    from gne.when import chat_segments, generate_when
    e = session.engine
    msgs = [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "I am a helpful assistant."},
            {"role": "user", "content": "Tell me more about yourself."}]
    ids, segs = chat_segments(e, msgs)
    steered = [e.detokenize(ids[a:b], special=True) for a, b, st in segs if st]
    assert "I am a helpful assistant." in steered[0] and steered[-1].startswith("<|im_start|>assistant")
    assert segs[0][0] == 0 and segs[-1][1] == len(ids)
    vecs = session.vectors(session.recipe(units=["14:100:+1.0"]))[0]
    ctx = e.steer_ctx()
    base, _ = generate_when(e, ctx, msgs, None, "all", max_tokens=24)
    out = {}
    for w in ("all", "gen", "assistant", "after:I", "ramp:1-0"):
        out[w] = generate_when(e, ctx, msgs, vecs, w, max_tokens=24)
    assert out["gen"][0].token_ids[0] == base.token_ids[0] and out["gen"][1]["set_cvec"] == 2
    assert out["assistant"][1]["segments"][-1][2] is True
    assert out["after:I"][1].get("after_step") is not None
    ramp = out["ramp:1-0"][1]["ramp"]
    assert ramp[0] == [0, 1.0] and all(k < 1.0 for _, k in ramp[1:]) and [s for s, _ in ramp[1:]] == [8, 16]
    # all は従来の steer と同じ(全位置に足す)
    assert out["all"][0].token_ids == session.generate(vecs, msgs, max_tokens=24)["token_ids"]


@pytest.mark.model
def test_when_skip_replies(session):
    """skip-replies: 過去の返答の区間だけ素のまま読み、system・ユーザーの発話・生成プロンプトには足す(F118 の写し対策)。"""
    from gne.when import generate_when, parse_when
    assert str(parse_when("skip-replies")) == "skip-replies"
    e = session.engine
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "I am a helpful assistant."}, {"role": "user", "content": "Tell me more about yourself."}]
    vecs = session.vectors(session.recipe(units=["14:100:+1.0"]))[0]
    r, info = generate_when(e, e.steer_ctx(), msgs, vecs, "skip-replies", max_tokens=12)
    _, ia = generate_when(e, e.steer_ctx(), msgs, vecs, "assistant", max_tokens=12)
    ids = e.chat_ids(msgs)
    assert [s[:2] for s in info["segments"]] == [s[:2] for s in ia["segments"]] and r.n_gen > 0
    off = [e.detokenize(ids[a:b], special=True) for a, b, on in info["segments"] if not on]
    assert len(off) == 1 and "I am a helpful assistant." in off[0]                    # 押さないのは過去の返答だけ
    assert info["segments"][0][2] and info["segments"][-1][2] and info["set_cvec"] == 3
    # 過去の返答が無ければ全区間に足す
    _, i1 = generate_when(e, e.steer_ctx(), msgs[:2], vecs, "skip-replies", max_tokens=4)
    assert all(on for _, _, on in i1["segments"])


@pytest.mark.model
def test_steer_and_chat_take_when(session, tmp_path):
    d = session.steer(session.recipe(units=["14:100:+1.0"], when="gen"), prompt="Who are you?", max_tokens=12)
    assert d["when"] == "gen" and d["steered"]["when"]["set_cvec"] == 2
    from gne.schemas import SteerOut
    SteerOut.model_validate({**session.envelope("steer"), **d})
    from gne.chat import make_columns
    cols = make_columns(session.recipe(units=["14:100:+1.0"]), dose=1.0)
    r = session.chat(cols, ["Hi", "And?"], max_tokens=8, when="assistant")
    assert len(r["turns"]) == 2 and all(len(t["replies"]) == 3 for t in r["turns"])
    # export は all として書き出し、警告を出す
    session.warnings = []
    session.export_cvec(session.recipe(units=["14:100:+1.0"], when="gen"), tmp_path / "x.cvec.gguf")
    assert any("when=gen" in w for w in session.warnings)


@pytest.mark.model
def test_openai_api(session):
    from fastapi.testclient import TestClient

    from gne.ui.server import create_app
    c = TestClient(create_app(session, warm=False, openai=True))
    assert c.get("/v1/models").json()["data"][0]["id"] == session.meta.model_id
    body = {"model": "x", "messages": [{"role": "user", "content": [{"type": "text", "text": "Who are you?"}]}],
            "max_tokens": 12, "gne": {"units": ["14:100:+1.0"], "when": "gen"}}
    r = c.post("/v1/chat/completions", json=body).json()
    ch = r["choices"][0]
    assert ch["message"]["role"] == "assistant" and ch["finish_reason"] in ("stop", "length")
    assert r["usage"]["completion_tokens"] <= 12 and r["gne"]["when"]["when"] == "gen"
    ref = session.generate(session.vectors(session.recipe(units=["14:100:+1.0"]))[0],
                           [{"role": "user", "content": "Who are you?"}], max_tokens=12, when="gen")
    assert ch["message"]["content"] == ref["text"]
    # ストリーミング: 断片をつなぐと同じ文、最後に [DONE]
    with c.stream("POST", "/v1/chat/completions", json={**body, "stream": True}) as s:
        lines = [ln for ln in s.iter_lines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(ln[6:]) for ln in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(ch["choices"][0]["delta"].get("content", "") for ch in chunks) == ref["text"]
    assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")
    # 誤り: OpenAI 形式のエラー
    bad = c.post("/v1/chat/completions", json={**body, "gne": {"units": ["14:100:+1"], "when": "sometimes"}})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "BAD_ARGS"
    # --openai を付けなければ出ない
    assert TestClient(create_app(session, warm=False)).get("/v1/models").status_code == 404


@pytest.mark.model
def test_ui_steer_stream_when(session):
    from fastapi.testclient import TestClient

    from gne.ui.server import create_app
    c = TestClient(create_app(session, warm=False))
    msgs = [{"role": "user", "content": "Who are you?"}]
    with c.stream("POST", "/api/steer", json={"max_tokens": 8, "when": "gen", "columns": [
            {"units": ["14:100:+1.0"], "messages": msgs}, {"units": ["14:100:+1.0"], "messages": msgs, "when": "all"}]}) as r:
        ev = [json.loads(ln[6:]) for ln in r.iter_lines() if ln.startswith("data: ")]
    ends = [e for e in ev if e["type"] == "end"]
    assert ends[0]["when"]["when"] == "gen" and ends[1]["when"]["when"] == "all"


# --- gain(§6.16): モデル不要 --------------------------------------------------------------------------

def _tiny_ffn(path, qtype="F32"):
    """ffn_up / ffn_down を持つ小さな GGUF(llama arch)。"""
    import numpy as np
    from gguf import GGMLQuantizationType, GGUFWriter
    from gguf import quants as gq
    rng = np.random.default_rng(0)
    w = GGUFWriter(str(path), arch="llama")
    w.add_name("tiny")
    w.add_block_count(3)
    w.add_embedding_length(32)
    w.add_feed_forward_length(64)
    w.add_token_list([f"t{i}" for i in range(10)])
    ups = {}
    for L in range(3):
        up = rng.standard_normal((64, 32)).astype(np.float32)
        ups[L] = up
        if qtype == "Q8_0":
            w.add_tensor(f"blk.{L}.ffn_up.weight", gq.quantize(up, GGMLQuantizationType.Q8_0), raw_dtype=GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(f"blk.{L}.ffn_up.weight", up)
        w.add_tensor(f"blk.{L}.ffn_down.weight", rng.standard_normal((32, 64)).astype(np.float32))
    w.add_tensor("token_embd.weight", rng.standard_normal((10, 32)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return ups


def test_parse_gain_unit_and_recipe(tmp_path):
    from gne.steer import parse_unit
    u = parse_unit("14:8949:x2")
    assert u.mode == "gain" and u.k == 2.0 and u.label() == "L14 N8949 ×2"
    assert parse_unit("14:8949:x-1").k == -1.0 and parse_unit("14:8949:+1.0").mode == "add"
    with pytest.raises(GneError):
        parse_unit("3:random:7:x2")
    r = Recipe([parse_unit("14:1:+1.0"), parse_unit("14:2:x3")], "m", name="mix")
    r.save(tmp_path / "r.yaml")
    d = yaml.safe_load((tmp_path / "r.yaml").read_text(encoding="utf-8"))
    assert d["units"][0] == {"layer": 14, "neuron": 1, "dose": 1.0}
    assert d["units"][1] == {"layer": 14, "neuron": 2, "mode": "gain", "k": 3.0}
    back = Recipe.load(tmp_path / "r.yaml")
    assert [u.mode for u in back.units] == ["add", "gain"] and len(back.gain_units) == 1
    half = back.scaled(0.5)                            # gain は 1 と k の間を動く(0 で無介入)
    assert half.units[0].dose == 0.5 and half.units[1].k == 2.0 and back.scaled(0).units[1].k == 1.0
    from gne.chat import make_columns
    cols = make_columns(back)
    assert cols[0].recipe.units[1].k == -1.0 and cols[2].recipe.units[1].k == 3.0    # − の列は 1 を挟んで反対側


def test_gain_lora_tensors_and_bake(tmp_path):
    import numpy as np
    from gguf import GGUFReader

    from gne.gain import bake_gain_gguf, export_gain_lora, gain_by_layer
    from gne.steer import parse_unit
    from gne.weights import Weights
    ups = _tiny_ffn(tmp_path / "m.gguf")
    w = Weights(tmp_path / "m.gguf")
    r = Recipe([parse_unit("1:5:x2"), parse_unit("1:9:x0"), parse_unit("2:3:x-1"), parse_unit("1:5:x1.5")], name="g")
    assert gain_by_layer(r.units) == {1: [(5, 3.0), (9, 0.0)], 2: [(3, -1.0)]}       # 重なりは掛け合わせる
    rep = export_gain_lora(w, r, tmp_path / "g.lora.gguf")
    assert rep["rank"] == {"1": 2, "2": 1}
    rd = GGUFReader(tmp_path / "g.lora.gguf")
    f = {x.name: x for x in rd.fields.values()}
    assert f["general.type"].contents() == "adapter" and f["adapter.type"].contents() == "lora"
    assert f["general.architecture"].contents() == "llama" and "adapter.lora.alpha" not in f
    t = {x.name: x for x in rd.tensors}
    A, B = np.asarray(t["blk.1.ffn_up.weight.lora_a"].data), np.asarray(t["blk.1.ffn_up.weight.lora_b"].data)
    assert list(t["blk.1.ffn_up.weight.lora_a"].shape) == [32, 2] and list(t["blk.1.ffn_up.weight.lora_b"].shape) == [2, 64]
    assert np.allclose(A, ups[1][[5, 9]]) and B[5, 0] == 2.0 and B[9, 1] == -1.0 and np.count_nonzero(B) == 2
    # W x + B(A x) = 行 n を k 倍した W x
    x = np.random.default_rng(1).standard_normal(32).astype(np.float32)
    want = ups[1] @ x
    want[5] *= 3.0
    want[9] *= 0.0
    assert np.allclose(ups[1] @ x + B @ (A @ x), want, atol=1e-4)
    # 書き換え: F16 と Q8_0
    for rq in ("f16", "q8_0"):
        d = bake_gain_gguf(w, r, tmp_path / f"g-{rq}.gguf", requant=rq)
        assert [m["tensor"] for m in d["modified"]] == ["blk.1.ffn_up.weight", "blk.2.ffn_up.weight"]
        w2 = Weights(tmp_path / f"g-{rq}.gguf")
        up1 = w2.up_matrix(1)
        tol = 2e-2 if rq == "q8_0" else 2e-3
        assert np.allclose(up1[5], 3 * ups[1][5], atol=tol * 3) and np.abs(up1[9]).max() == 0
        assert np.allclose(w2.up_matrix(0), ups[0])                                  # 触らない層はそのまま
        assert "+gain:g" in w2.meta.name
    with pytest.raises(GneError):
        export_gain_lora(w, Recipe([parse_unit("1:5:+1.0")]), tmp_path / "x.gguf")


# --- gain: 実モデル -----------------------------------------------------------------------------------

@pytest.mark.model
def test_v11_gain(session):
    from gne.verify import Verifier
    r = Verifier(session.engine, session.weights).v11_gain()
    assert r.status == "pass", r.detail
    assert all(row["others_max_diff"] == 0 for row in r.data["rows"])


@pytest.mark.model
def test_gain_steer_and_llama_lora(session, tmp_path):
    msgs = [{"role": "user", "content": "Who are you?"}]
    r = session.recipe(units=["14:100:x8", "14:2613:x0"])
    lora = session.lora_for(r)
    assert lora and Path(lora[0][0]).exists() and session.lora_for(r) == lora                  # 使い回す
    d = session.steer(r, prompt="Who are you?", max_tokens=24)
    assert d["steered"]["when"]["lora"] == [lora[0][0]] and d["units"][0]["mode"] == "gain"
    # gain と add を同時に当てられる
    mix = session.steer(session.recipe(units=["14:100:x8", "14:100:+1.0"]), prompt="Who are you?", max_tokens=12)
    assert mix["steered"]["n_gen"] > 0
    from gne.verify import find_llama_completion, run_llama_completion
    b = find_llama_completion()
    if b:
        ours = session.generate(None, msgs, max_tokens=24, lora=lora)["text"]
        theirs = run_llama_completion(b, session.engine, session.engine.render_chat(msgs), 24, None, lora=Path(lora[0][0]))
        assert ours == theirs
    out = session.gain(r, tmp_path / "x.lora.gguf")
    assert out["kind"] == "lora" and out["rank"] == {"14": 2}
    with pytest.raises(GneError):                       # gain だけのレシピは control vector にならない
        session.export_cvec(r, tmp_path / "x.cvec.gguf")


# --- eval(A): 採点はモデル不要 ----------------------------------------------------------------------------

def test_eval_scoring():
    from gne.evalx import (
        load_suite,
        mix_ratio,
        persona_metrics,
        score_exact,
        score_format,
    )
    st = load_suite()
    assert (len(st.exact), len(st.format), len(st.mix), len(st.persona), len(st.ppl)) == (24, 12, 8, 8, 3)
    assert len(st.zh_only) > 3000 and "这" in st.zh_only and "日" not in st.zh_only       # 日本語と共通の字は入れない
    assert score_exact("The answer is 42.", ["42"]) and not score_exact("420", ["42"])       # 数は語の境界で
    assert score_exact("**Paris**", ["paris"]) and score_exact("東京です。", ["東京"])
    assert not score_exact("Sure!\n42", ["42"])                                              # 最初の行だけを見る
    assert score_format('```json\n{"name": "A", "age": 3}\n```', {"check": "json", "keys": ["name", "age"]})
    assert not score_format('{"name": "A"}', {"check": "json", "keys": ["name", "age"]})
    assert score_format('["a", "b", "c"]', {"check": "json_array", "n": 3})
    assert score_format("a\n\nb\nc", {"check": "lines:3"}) and not score_format("1. a\n2. b", {"check": "lines:3"})
    assert score_format("I like sushi.", {"check": "lang:en"}) and not score_format("寿司が好き", {"check": "lang:en"})
    assert score_format("私は寿司が好きです。", {"check": "lang:ja"}) and not score_format("我喜欢寿司", {"check": "lang:ja"})
    zh, other = mix_ratio("这是测试。日本語です", st.zh_only)
    assert zh > 0.2 and other == 0
    assert mix_ratio("안녕하세요 hello", st.zh_only)[1] == 0.5
    words = ["", "a", "b", "c", "d"]
    base = [{"token_ids": [1, 2, 0], "finish": "eog"}, {"token_ids": [1, 0], "finish": "eog"}]          # 末尾の 0 が EOG
    out = [{"token_ids": [1, 3, 4], "finish": "length"}, {"token_ids": [3, 0], "finish": "eog"}]
    pm = persona_metrics(base, out, words)
    assert pm["novelty"] == pytest.approx((2 / 3 + 1) / 2) and pm["top_words"] == [["c", 2]] and pm["mass"] == 2
    assert pm["capped"] == 0.5 and persona_metrics(base, base, words)["novelty"] == 0


def _mini_suite(tmp_path):
    (tmp_path / "ppl").mkdir()
    (tmp_path / "ppl" / "en.txt").write_text("Alice was beginning to get very tired of sitting by her sister on the bank. " * 3)
    (tmp_path / "zh_only_chars.txt").write_text("# test\n这们说\n")
    d = {"name": "mini", "max_tokens": {"exact": 12, "format": 32, "mix": 32, "persona": 32},
         "exact": [{"id": "add", "prompt": "What is 17 + 25? Answer with just the number.", "answer": ["42"]},
                   {"id": "cap", "prompt": "What is the capital of France? Answer with one word.", "answer": ["paris"]}],
         "format": [{"id": "j", "prompt": "Return a JSON object with keys \"a\" and \"b\". Output only the JSON.",
                     "check": "json", "keys": ["a", "b"]}],
         "mix": [{"id": "m", "prompt": "春の好きなところを 2 文で教えてください。"}],
         "persona": {"prompt_sets": ["en"]}, "ppl": [{"id": "en", "path": "ppl/en.txt"}]}
    p = tmp_path / "mini.yaml"
    p.write_text(yaml.safe_dump(d, allow_unicode=True))
    return p


@pytest.mark.model
def test_token_logprobs_matches_generation(session):
    """teacher-forced の logprob が greedy 生成の logprob と合う(ppl の土台)。"""
    import numpy as np
    e = session.engine
    ctx = e.steer_ctx()
    ids = e.tokenize_prompt(e.render_chat([{"role": "user", "content": "Describe the color orange."}]))
    r = e.generate_greedy(ctx, [ids], max_tokens=24)[0]
    lp = e.token_logprobs(ctx, ids + r.token_ids, chunk=512)
    assert lp.shape == (len(ids) + len(r.token_ids) - 1,)
    tail = lp[len(ids) - 1:]
    diff = np.abs(tail - np.array(r.logprob))
    # 1 トークンずつの decode とまとめた decode ではカーネルが違い、0.6B Q8_0 で最大 0.13 nats ずれる(verification §11.3)
    assert diff.mean() < 0.05 and diff.max() < 0.3 and np.all(lp <= 0)


@pytest.mark.model
def test_eval_end_to_end(session, tmp_path):
    from gne.schemas import COMMAND_SCHEMAS
    suite = _mini_suite(tmp_path)
    r = session.recipe(units=["14:100:+1.5"], name="e2e")
    d = session.eval(r, doses=[1.0, 0.5], suite=suite)
    assert d["doses"] == [0.0, 0.5, 1.0] and [x["dose"] for x in d["rows"]] == [0.0, 0.5, 1.0]        # 0 は必ず入る
    b, top = d["rows"][0], d["rows"][-1]
    assert b["capability_drop_pt"] == 0 and b["ppl_worse_pct"] == 0 and b["persona"]["novelty"] == 0 and b["shippable"]
    assert top["ppl_worse_pct"] > 0 and top["persona"]["novelty"] > 0
    assert set(b["ppl"]) == {"en"} and set(d["samples"]) == {"0", "0.5", "1"}
    assert Path(d["path"]).exists() and d["name"] == "e2e"
    assert session.eval_load("e2e")["rows"] == json.loads(json.dumps(d["rows"]))
    lst = session.evals()
    assert lst[0]["name"] == "e2e" and lst[0]["units"] == ["14:100:+1.5"] and 0.0 in lst[0]["shippable"]
    COMMAND_SCHEMAS["eval"].model_validate({**session.envelope("eval"), **d})
    # 同じ条件なら同じ数(greedy・同じバッチの組み方)
    again = session.eval(r, doses=[0], suite=suite, name="e2e-again")
    assert again["rows"][0]["capability"] == b["capability"] and again["rows"][0]["ppl"] == b["ppl"]
    # when=gen は 1 本ずつ。名前に @gen が付く。gain のレシピも測れる
    g = session.eval(session.recipe(units=["14:100:x4"], name="g"), doses=[1], suite=suite, when="gen")
    assert g["name"] == "g@gen" and g["when"] == "gen" and g["rows"][1]["persona"]["novelty"] >= 0
    # 負の倍率でもベースラインは倍率 0 の行
    n = session.eval(r, doses=[-0.5], suite=suite, name="neg")
    assert [x["dose"] for x in n["rows"]] == [-0.5, 0.0] and n["rows"][1]["capability_drop_pt"] == 0
    assert n["rows"][1]["persona"]["novelty"] == 0 and n["rows"][0]["persona"]["novelty"] > 0
    # judge(採点用の GGUF。ここでは同じモデル)
    j = session.eval(r, doses=[1], suite=suite, judge_model=MODEL, judge_target="a pirate", name="judged")
    assert j["judge"]["target"] == "a pirate" and all(len(x["judge_scores"]) == 4 for x in j["rows"])
    assert all(s_ is None or 1 <= s_ <= 5 for x in j["rows"] for s_ in x["judge_scores"])
    COMMAND_SCHEMAS["eval"].model_validate({**session.envelope("eval"), **j})
    with pytest.raises(GneError):
        session.eval(r, suite=suite, judge_model="x.gguf")                                   # judge は 2 つそろえて
    with pytest.raises(GneError):
        session.eval_load("../x")
    # 調書(CHANGE-01 D): このニューロンを含む eval(新しい順)と、当てたことのある gain の倍率
    dn = session.neuron(14, 100)
    names = [e["name"] for e in dn["evals"]]
    assert names[0] == "judged" and "g@gen" in names and len(names) <= 5 and 4.0 in dn["gain_k"]
    ge = next(e for e in dn["evals"] if e["name"] == "g@gen")
    assert ge["mode"] == "gain" and ge["k"] == 4 and ge["when"] == "gen" and ge["at_max"]["dose"] == 1
    assert dn["coverage"]["eval"] and dn["coverage"]["gain"]
    assert session.neuron(14, 101)["evals"] == []
    COMMAND_SCHEMAS["neuron"].model_validate({**session.envelope("neuron"), **dn})


@pytest.mark.model
def test_eval_job_and_cli(session, tmp_path):
    from gne.jobs import standard_kinds
    suite = _mini_suite(tmp_path)
    out = standard_kinds()["eval"](session, units=["14:100:+1.0"], doses=[1], suite=suite, name="job")
    assert out["name"] == "job" and "samples" not in out and len(out["rows"]) == 2
    with pytest.raises(GneError):
        standard_kinds()["eval"](session, doses=[1])
    from typer.testing import CliRunner

    from gne.cli import app
    res = CliRunner().invoke(app, ["eval", "-m", str(MODEL), "--root", str(session.store.root), "--unit", "14:100:+1.0",
                                   "--doses", "0,1", "--suite", str(suite), "--name", "cli", "--json"])
    assert res.exit_code == 0, res.output
    d = json.loads(res.stdout.strip().splitlines()[-1])
    assert d["command"] == "eval" and d["name"] == "cli" and len(d["rows"]) == 2
