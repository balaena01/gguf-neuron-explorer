"""音声の入力(media.py)のテスト。

- 部品の content の読み方・OpenAI 互換 API での扱い(モデル不要)
- 埋め込みの区間の decode(Qwen3-0.6B、GNE_TEST_MODEL): 音声の代わりに「語の埋め込み」を置き場所に入れ、
  同じ語を文で入れたときとプロンプト・ロジット・捕捉・生成が一致すること
- 本物の音声(GNE_TEST_OMNI = LFM3-Omni の GGUF、横に mmproj。GNE_TEST_AUDIO = 話し声の wav)。無ければ飛ばす
"""

import base64
import os
from pathlib import Path

import numpy as np
import pytest

from gne.media import MARKER, audio_message, find_mmproj, has_audio, normalize, split_content
from gne.steer import GneError

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
OMNI = os.environ.get("GNE_TEST_OMNI")
AUDIO = os.environ.get("GNE_TEST_AUDIO")


def test_split_content(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFFfake")
    b64 = base64.b64encode(b"abc").decode()
    text, audios = split_content([{"type": "text", "text": "hi "}, {"type": "input_audio", "input_audio": {"data": b64}},
                                  {"type": "input_audio", "input_audio": {"data": "data:audio/wav;base64," + b64}},
                                  {"type": "input_audio", "input_audio": {"path": str(wav)}}])
    assert text == "hi " + MARKER * 3 and audios == [b"abc", b"abc", b"RIFFfake"]
    assert split_content("plain") == ("plain", [])
    with pytest.raises(GneError):
        split_content([{"type": "image_url", "image_url": {"url": "x"}}])
    with pytest.raises(GneError):
        split_content([{"type": "input_audio", "input_audio": {}}])
    with pytest.raises(GneError):
        split_content([{"type": "input_audio", "input_audio": {"path": str(tmp_path / "none.wav")}}])
    m = audio_message([wav], "and this")
    msgs, audios = normalize([{"role": "system", "content": "s"}, m])
    assert msgs[1]["content"] == MARKER + "and this" and audios == [b"RIFFfake"] and msgs[0]["content"] == "s"
    assert has_audio([m]) and not has_audio(msgs)


def test_find_mmproj(tmp_path):
    model = tmp_path / "Omni-2B-Q8_0.gguf"
    model.write_bytes(b"")
    assert find_mmproj(model) is None
    (tmp_path / "mmproj-Omni-2B-Q8_0.gguf").write_bytes(b"")
    assert find_mmproj(model).name == "mmproj-Omni-2B-Q8_0.gguf"
    (tmp_path / "mmproj-Other-F16.gguf").write_bytes(b"")
    assert find_mmproj(model).name == "mmproj-Omni-2B-Q8_0.gguf"      # 複数ならモデル名の頭がそろうもの


def test_openai_content_refuses_paths():
    from gne.ui.openai_api import _messages
    b64 = base64.b64encode(b"abc").decode()
    out = _messages({"messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": b64, "path": "/etc/hostname", "format": "wav"}},
        {"type": "text", "text": "what?"}]}]})
    assert out[0]["content"] == [{"type": "input_audio", "input_audio": {"data": b64}}, {"type": "text", "text": "what?"}]
    with pytest.raises(GneError):
        _messages({"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"path": "/etc/hostname"}}]}]})
    assert _messages({"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}]}]})[0]["content"] == "a"


# --- 埋め込みの区間の decode(音声の代わりに語の埋め込み)-------------------------------------------

PHRASE = " about cats"


class _WordMedia:
    """音声の代わりに、PHRASE のトークンの埋め込みを置き場所に入れる(media.Media と同じ口)。"""

    def __init__(self, engine, weights):
        self.e = engine
        self.toks = engine.tokenize(PHRASE, parse_special=False)
        self.rows = weights.embed_rows(self.toks)
        self.n_embd = self.rows.shape[1]
        self.calls = 0

    def tokenize(self, text, audios):
        self.calls += 1
        parts = text.split(MARKER)
        assert len(parts) == len(audios) + 1
        ids = []
        for k, p in enumerate(parts):
            ids += self.e.tokenize(p, add_special=False, parse_special=True) if p else []
            if k < len(audios):
                ids += [-(j + 1) for j in range(len(self.toks))]
        return ids

    def row(self, tok):
        return self.rows[-tok - 1], False

    def close(self):
        pass


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip("モデルがありません")
    from gne.api import Session
    s = Session(MODEL, root=tmp_path_factory.mktemp("media"))
    s.engine._media = _WordMedia(s.engine, s.weights)
    yield s
    s.close()


def _pair(session):
    b64 = base64.b64encode(b"fake").decode()
    with_audio = [{"role": "user", "content": [{"type": "text", "text": "Tell me a joke"},
                                               {"type": "input_audio", "input_audio": {"data": b64}}]}]
    as_text = [{"role": "user", "content": "Tell me a joke" + PHRASE}]
    return with_audio, as_text


def _split_decode(e, ctx, ids, cuts, *, logits_all=False):
    """ids を cuts の位置で分けて decode する(音声の区間を別に decode するのと同じ切り方)。最後か全部の位置のロジット。"""
    e.clear_memory(ctx)
    out = []
    bounds = [0, *cuts, len(ids)]
    for a, b in zip(bounds, bounds[1:]):
        part = ids[a:b]
        e._decode(ctx, [(t, a + j, 0, logits_all or b == len(ids) and j == len(part) - 1) for j, t in enumerate(part)])
        if logits_all:
            out += [e._logits(ctx, j) for j in range(len(part))]
    return np.stack(out) if logits_all else e._logits(ctx, len(ids[bounds[-2]:]) - 1)


@pytest.mark.model
def test_embedding_span_matches_tokens(session):
    """置き場所に語の埋め込みを入れたプロンプトは、同じ語を文で入れ、同じ所で decode を分けたものとちょうど一致する。
    (decode を分けること自体で GPU の計算順が変わり、1 回で decode したものとは 0.5 程度ずれる。F2 と同じ)"""
    from gne.capture import Capture
    e = session.engine
    wa, wt = _pair(session)
    ids_a, ids_t = e.chat_ids(wa), e.chat_ids(wt)
    m = e.media
    assert [m.toks[-t - 1] if t < 0 else t for t in ids_a] == ids_t and sum(t < 0 for t in ids_a) == len(m.toks)
    assert "<audio>" in e.detokenize(ids_a) and e.token_text(-1) == "<audio>"
    a = next(i for i, t in enumerate(ids_a) if t < 0)
    cuts = [a, a + len(m.toks)]
    ctx = e.steer_ctx()
    e.clear_memory(ctx)
    pa = e.prefill(ctx, [ids_a], logits_all=True)
    ref = _split_decode(e, ctx, ids_t, cuts, logits_all=True)
    assert np.array_equal(pa.all_logits[0], ref) and np.array_equal(pa.logits[0], ref[-1])
    e.clear_memory(ctx)
    assert np.array_equal(e.prefill(ctx, [ids_a]).logits[0], _split_decode(e, ctx, ids_t, cuts))
    # 2 本を 1 バッチで(音声ありの列となしの列が混ざる): 走り、音声の列の結果は 1 本のときに近い
    e.clear_memory(ctx)
    pb = e.prefill(ctx, [ids_a, ids_t])
    assert np.isfinite(pb.logits).all() and np.corrcoef(pb.logits[0], pa.logits[0])[0, 1] > 0.99
    # 捕捉(probe ctx)でも、同じ所で分けた文の decode と同じ活性
    L = session.meta.n_layer // 2
    cap = Capture(lambda n: n == f"l_out-{L}")
    pctx = e.probe_ctx(cap)
    e.clear_memory(pctx)
    e.prefill(pctx, [ids_a])
    got = cap.take()[f"l_out-{L}"]
    cap.reset()
    _split_decode(e, pctx, ids_t, cuts)
    want = cap.take()[f"l_out-{L}"]
    assert got.shape == (len(ids_a), session.meta.n_embd) and np.array_equal(got, want)


@pytest.mark.model
def test_generation_with_audio_parts(session):
    """生成(when の各モード)が音声の部品で走る。assistant の区間は文で入れたときと同じで、音声の区間を含まない。"""
    from gne.when import generate_when
    e = session.engine
    wa, wt = _pair(session)
    vecs = session.vectors(session.recipe(units=[f"{session.meta.n_layer // 2}:100:+2.0"]))[0]
    calls = e.media.calls
    for when in ("all", "assistant", "gen"):
        ra, ia = generate_when(e, e.steer_ctx(), wa, vecs, when, max_tokens=12)
        rb, _ = generate_when(e, e.steer_ctx(), wa, vecs, when, max_tokens=12)
        _, it = generate_when(e, e.steer_ctx(), wt, vecs, when, max_tokens=12)
        assert ra.token_ids == rb.token_ids and ra.n_gen > 0, when                   # 同じ入力なら同じ列
        assert ia["audio_in"] == len(e.media.toks) and "audio_in" not in it
        if when == "assistant":
            assert ia["segments"] == it["segments"]
            a = next(i for i, t in enumerate(e.chat_ids(wa)) if t < 0)
            assert all(not st for x, y, st in ia["segments"] if x <= a < y)             # 音声の区間には足さない
    assert e.media.calls > calls
    # 多ターン: 前の発話の音声も毎回同じ置き場所になる
    conv = wa + [{"role": "assistant", "content": "Why did the cat sit on the computer?"},
                 {"role": "user", "content": "Why?"}]
    r, info = generate_when(e, e.steer_ctx(), conv, vecs, "assistant", max_tokens=8)
    assert info["audio_in"] == len(e.media.toks) and r.n_gen > 0
    assert any(st for _, _, st in info["segments"][:-1])                              # 過去の返事の区間
    d = session.steer(session.recipe(units=[f"{session.meta.n_layer // 2}:100:+0"]), messages=wa, max_tokens=4, compare=False)
    from gne.schemas import COMMAND_SCHEMAS
    COMMAND_SCHEMAS["steer"].model_validate({**session.envelope("steer"), **d})


# --- 本物の音声(LFM3-Omni)-----------------------------------------------------------------------

@pytest.mark.model
@pytest.mark.skipif(not (OMNI and AUDIO), reason="GNE_TEST_OMNI と GNE_TEST_AUDIO が要ります")
def test_omni_audio(tmp_path):
    from gne.api import Session
    with Session(OMNI, root=tmp_path) as s:
        assert find_mmproj(Path(OMNI)) is not None
        d = s.steer(s.recipe(units=["14:100:+0"]), system="Perform ASR in English.", audio=[AUDIO], max_tokens=40, compare=False)
        st = d["steered"]
        assert st["text"].strip() and st["when"]["audio_in"] > 0
        d2 = s.steer(s.recipe(units=["14:100:+0"]), system="Respond in text.", audio=[AUDIO], max_tokens=40, compare=False)
        assert d2["steered"]["text"].strip()
        assert all(t not in d2["steered"]["token_ids"] for t in s.engine.audio_out_tokens())
