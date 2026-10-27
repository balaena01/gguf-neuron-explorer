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
