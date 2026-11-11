"""CHANGE-07 WP4 のテスト: 鍵の作成と確認(gne/keyed_build.py)。

モデルなし: トークン片のバイト位置と before / not-before の位置(日本語の 1 文字が複数のトークンにまたがる場合)、
positions の書き方、dont_touch の読み、作成と確認の重なりの検出、擬似鍵の並べ替えの釣り合い、post-norm の rms。
モデルあり(Qwen3-0.6B、GNE_TEST_MODEL。CPU で回すときは GNE_TEST_NGL=0): 与えた返事(です・ます調、語尾の例)で
側 A の作り方 (b)(推論時と同じ文脈の返事の、文末を書く直前の位置)の鍵を作り、保存項目・κᵀμ = 1・q_mode の 3 通り・
V34(保留集合で測る)・V35(本物の鍵が擬似鍵より選択性が高い)を確かめる。側 A の作り方 (a)(system_a、生成なし)と、
介入(iv)を当てた捕捉も 1 件ずつ。
"""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from gne.keyed_build import (
    DontTouch,
    balanced_split,
    byte_offsets,
    check_capture,
    held_out_info,
    load_key_spec,
    match_tokens,
    parse_positions,
    read_key_spec,
    rms_rows,
    select_positions,
    target_capture,
)
from gne.steer import GneError

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
NGL = int(os.environ.get("GNE_TEST_NGL", "-1"))
ZUNDAMON = ROOT / "configs" / "contrast" / "zundamon_endings.yaml"

# 「今」(e4 bb 8a)が 2 つのトークンにまたがり、2 番目の片は「今」の最後のバイトと「日」で始まる
JA_PIECES = [b"\xe4\xbb", b"\x8a" + "日".encode(), "は".encode(), "晴れ".encode(), "です".encode(), "。".encode(),
             "明日".encode(), "も".encode(), "晴れ".encode(), "です".encode(), "。".encode()]


class FakeEngine:
    """select_positions が使う is_eog と token_bytes だけを持つ。トークン番号 = JA_PIECES の添字、99 が EOG。"""

    def __init__(self, pieces):
        self.pieces = pieces

    def token_bytes(self, t):
        return b"" if t == 99 else self.pieces[t]

    def is_eog(self, t):
        return t == 99


# --- モデルなし ----------------------------------------------------------------------------------------

def test_byte_offsets_and_multitoken_japanese():
    """累積バイト位置と、文字位置 → トークン(1 文字が 2 トークンにまたがる場合はその文字の最初のバイトのトークン)。"""
    offs = byte_offsets(JA_PIECES)
    assert offs[:7] == [0, 2, 6, 9, 15, 21, 24]
    assert offs[-1] == sum(len(p) for p in JA_PIECES)
    assert match_tokens(JA_PIECES, "今") == [0]
    assert match_tokens(JA_PIECES, "日は") == [1]            # 「日」は 2 番目の片の途中(「今」の最後のバイトの後)から
    assert match_tokens(JA_PIECES, "です。") == [4, 9]        # 重ならない一致すべて
    assert match_tokens(JA_PIECES, "晴れです") == [3, 8]
    assert match_tokens(JA_PIECES, "x*") == []                # 長さ 0 の一致は数えない
    # 途中で切れた UTF-8(生成が文字の途中で止まった)でも、それより前の位置はずれない
    assert match_tokens(JA_PIECES[:4] + [b"\xe3\x81"], "晴れ") == [3]


def test_select_positions_before_not_before_skip():
    e = FakeEngine(JA_PIECES)
    reply = list(range(len(JA_PIECES))) + [99]               # 末尾の EOG は除く
    P = 20
    before = select_positions(e, reply, "before:です。", n_prompt=P)
    assert before.tolist() == [P + 4 - 1, P + 9 - 1]          # 「です」を決める位置 = そのトークンの 1 つ前
    nb = select_positions(e, reply, "not-before:です。", n_prompt=P)
    universe = [P + j - 1 for j in range(len(JA_PIECES))]     # 返事の各トークンを決める位置
    assert sorted(before.tolist() + nb.tolist()) == universe and not set(before) & set(nb)
    assert select_positions(e, reply, "before:です。", skip=5, n_prompt=P).tolist() == [P + 8]
    assert select_positions(e, reply, "not-before:です。", skip=9, n_prompt=P).tolist() == [P + 9]
    assert select_positions(e, reply, "response", n_prompt=P).tolist() == list(range(P, P + len(JA_PIECES)))
    assert select_positions(e, reply, "response", skip=3, n_prompt=P).tolist() == list(range(P + 3, P + len(JA_PIECES)))
    assert select_positions(e, reply, "last", n_prompt=P).tolist() == [P - 1]
    assert select_positions(e, reply, "last:3", n_prompt=P).tolist() == [P - 3, P - 2, P - 1]
    assert select_positions(e, reply, "assistant", n_prompt=P, start=15).tolist() == list(range(15, P))
    # 返事の最初のトークンに一致: n_prompt = 0 なら -1(生成プロンプトの最後のトークン)、n_prompt > 0 なら P - 1
    assert select_positions(e, reply, "before:今日", n_prompt=0).tolist() == [-1]
    assert select_positions(e, reply, "before:今日", n_prompt=P).tolist() == [P - 1]
    assert select_positions(e, reply, "before:雨", n_prompt=P).tolist() == []


def test_parse_positions():
    assert parse_positions(None).kind == "response"
    assert parse_positions("last:4").k == 4 and parse_positions("last:4").text == "last:4"
    p = parse_positions("before: です")
    assert p.kind == "before" and p.pattern == " です" and p.needs_reply   # 正規表現の空白は残す
    assert parse_positions("not-before:(のだ|なのだ)").pattern == "(のだ|なのだ)"
    assert not parse_positions("assistant").needs_reply
    for bad in ("foo", "last:0", "before:", "before:(", "before:x*", "all"):
        with pytest.raises(GneError) as ei:
            parse_positions(bad)
        assert ei.value.code == "BAD_ARGS"


def _write(path: Path, data) -> Path:
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_load_key_spec_dont_touch(tmp_path):
    """dont_touch: system を省くと system なし(空の文字列は空の system メッセージ)、prompts・replies のパスは対照集合の
    ファイルからの相対。側 A と側 B の文脈が同じ対照集合(mode same 用)も読める。"""
    sub = tmp_path / "sets"
    sub.mkdir()
    _write(sub / "benign.yaml", {"prompts": ["今日の天気は？", "おすすめの本は？"]})
    _write(sub / "benign_replies.yaml", ["晴れです。", "小説がおすすめです。"])
    spec = _write(sub / "k.yaml", {
        "name": "k", "positions": "before:です。", "skip": 1,
        "user_prompts": ["週末は何をする？", "好きな季節は？"], "replies": ["散歩です。", "秋です。"],
        "dont_touch": [{"prompts": ["What is your hobby?"]},
                       {"name": "plain", "system": "丁寧に答えてください。", "prompts": "benign.yaml",
                        "replies": "benign_replies.yaml", "positions": "response"},
                       {"name": "empty", "system": "", "prompts": ["こんにちは"]}]})
    cs, dts = load_key_spec(spec)
    assert cs.system_a is None and cs.system_b is None and cs.prompts("A") == cs.prompts("B")
    assert [d.name for d in dts] == ["dt0", "plain", "empty"]
    assert dts[0].system is None and dts[0].replies is None and dts[0].positions is None
    assert dts[1].prompts == ["今日の天気は？", "おすすめの本は？"] and dts[1].replies == ["晴れです。", "小説がおすすめです。"]
    assert dts[1].positions == "response" and dts[1].system == "丁寧に答えてください。"
    assert dts[2].system == ""
    assert dts[0].convs([0]) == [[{"role": "user", "content": "What is your hobby?"}]]
    assert dts[2].convs([0])[0][0] == {"role": "system", "content": ""}
    full = read_key_spec(spec)
    assert full.positions == "before:です。" and full.skip == 1 and full.replies["same"] == ["散歩です。", "秋です。"]
    assert isinstance(full.dont_touch[0], DontTouch)

    def bad(dt, code="BAD_CONTRAST"):
        p = _write(sub / "bad.yaml", {"name": "b", "system_a": "x", "user_prompts": ["a", "b"], "dont_touch": dt})
        with pytest.raises(GneError) as ei:
            load_key_spec(p)
        assert ei.value.code == code
    bad([{"prompts": ["a"], "sytem": "typo"}])                                 # 知らないキー
    bad([{"prompts": ["a", "b"], "replies": ["only one"]}])                    # 返事の数が違う
    bad([{"system": "x"}])                                                     # prompts が無い
    bad([{"name": "g", "prompts": ["a"]}, {"name": "g", "prompts": ["b"]}])    # name が重なる
    bad([{"prompts": "missing.yaml"}], "FILE_NOT_FOUND")
    bad([{"prompts": ["a"], "positions": "everywhere"}], "BAD_ARGS")
    # 質問の書き方の誤りは、同じ文脈の対照集合への読み替えで隠さない
    p = _write(sub / "bad2.yaml", {"name": "b", "user_prompts": [{"role": "assistant", "content": "x"}]})
    with pytest.raises(GneError) as ei:
        load_key_spec(p)
    assert ei.value.code == "BAD_CONTRAST" and "user" in ei.value.message


def test_zundamon_config_is_mode_same():
    from gne.keyed_build import _resolve_mode
    s = read_key_spec(ZUNDAMON)
    assert s.contrast.system_a and s.contrast.system_b is None
    mode, pb = _resolve_mode(parse_positions(s.positions), s.positions_b)
    assert mode == "same" and pb.text == "not-before:(です|ます)。"
    assert [d.name for d in s.dont_touch] == ["english"] and s.dont_touch[0].system is None


def test_held_out_info_overlap():
    """V34 の判定: 同じ対照集合で subset が重なる、または (system, プロンプト) のハッシュが重なると held_out は False。"""
    key = SimpleNamespace(contrast_sha="abc", subset="even", hashes_a=["h0", "h2"], hashes_b=["h0", "h2"])
    assert held_out_info(key, "abc", "odd", ["h1", "h3"])["held_out"]
    o = held_out_info(key, "abc", "even", ["h0", "h2"])
    assert not o["held_out"] and o["subset_overlap"] and o["hash_overlap"] == 2
    assert not held_out_info(key, "abc", "all", ["h1"])["held_out"]
    o = held_out_info(key, "zzz", "odd", ["h2", "h5"])                       # 別のファイルでも同じ例が入っていれば重なり
    assert not o["held_out"] and not o["same_contrast"] and o["hash_overlap"] == 1
    assert held_out_info(key, "zzz", "all", ["h5"])["held_out"]
    key_all = SimpleNamespace(contrast_sha="abc", subset="all", hashes_a=[], hashes_b=[])
    assert held_out_info(key_all, "abc", "odd", ["h9"])["subset_overlap"]


def test_balanced_split():
    rng = np.random.default_rng(0)
    for n_a, n_b in ((8, 8), (5, 9), (3, 20), (6, 2)):
        ia, ib = balanced_split(rng, n_a, n_b)
        assert len(ia) == n_a and len(ib) == n_b
        assert sorted(np.concatenate([ia, ib]).tolist()) == list(range(n_a + n_b))
    ia, ib = balanced_split(rng, 8, 8)
    assert (ia < 8).sum() == 4 and (ib < 8).sum() == 4                         # どちらにも本物の A が半分ずつ


def test_rms_rows_post_norm():
    rng = np.random.default_rng(3)
    x, W = rng.standard_normal((5, 7)), rng.standard_normal((4, 7))
    y = 2.0 * x @ W.T
    np.testing.assert_allclose(rms_rows(x, W, scale=2.0, eps=1e-5), np.sqrt((y ** 2).mean(axis=1) + 1e-5), rtol=1e-5)


def test_check_capture_and_unsupported_targets():
    check_capture(SimpleNamespace(skipped={"ffn_up-3": "not F32"}), {"ffn_swiglu-3"})        # 要らない名前は気にしない
    for skipped in ({"ffn_swiglu-3": "not F32"}, {"<callback error>": "boom"}):
        with pytest.raises(GneError) as ei:
            check_capture(SimpleNamespace(skipped=skipped), {"ffn_swiglu-3"})
        assert ei.value.code == "ARCH_UNSUPPORTED"
    # 書き込み先の捕捉: attn は tests/test_change07_attn.py、shared・expert は tests/test_change07_moe.py
    # mlpin(計算で作るニューロンの鍵)は tests/test_nwrite.py
    from gne.keyed_build import _TARGETS, AttnCapture, ExpertCapture, FfnCapture, MlpInCapture, SharedCapture
    assert _TARGETS == {"ffn": FfnCapture, "attn": AttnCapture, "shared": SharedCapture, "expert": ExpertCapture,
                        "mlpin": MlpInCapture}
    for t in ("expert", "expert:x", "moe"):              # 書き込み先の書き方が違えば BAD_KEY
        with pytest.raises(GneError) as ei:
            target_capture(None, t, [3])
        assert ei.value.code == "BAD_KEY"


# --- モデルを使う ------------------------------------------------------------------------------------

PROMPTS = ["週末は何をして過ごすのがおすすめ？", "朝ごはんに何を食べたらいい？", "おすすめの趣味を教えて。", "雨の日は何をする？",
           "疲れたときはどうすればいい？", "好きな季節はいつ？", "カレーの作り方を簡単に教えて。", "眠れない夜はどうしたらいい？",
           "日本のおすすめの観光地は？", "勉強に集中するコツは？", "猫と犬、どっちが好き？", "健康のために何をしている？",
           "おすすめの飲み物は？", "部屋を片付けるコツは？", "友達の誕生日に何を贈ればいい？", "朝早く起きるコツは？"]
# モデル自身の素の返事に見立てた、です・ます調の返事(どの返事も文末が「です。」「ます。」で終わる文を含む)
REPLIES = ["天気が良ければ公園を散歩するのがおすすめです。家ではゆっくり本を読むのもいいと思います。",
           "ご飯と味噌汁と卵焼きがおすすめです。忙しい日はバナナとヨーグルトでも十分です。",
           "料理は手軽に始められる趣味です。散歩や写真も気軽に楽しめます。",
           "雨の日は家で映画を見るのが好きです。温かいお茶を飲みながら過ごします。",
           "まずはしっかり眠ることが大切です。お風呂にゆっくり入るのも効果があります。",
           "私は秋が一番好きです。涼しくて食べ物がおいしい季節です。",
           "野菜と肉を炒めてから水で煮込みます。最後にルーを溶かせば完成です。",
           "寝る前にスマホを見ないようにすると眠りやすくなります。温かい牛乳を飲むのもおすすめです。",
           "京都はお寺や神社が多くておすすめです。北海道では自然とおいしい料理が楽しめます。",
           "時間を決めて短く区切るのが効果的です。机の上を片付けると集中しやすくなります。",
           "どちらもかわいいですが、私は猫が好きです。気ままなところが魅力です。",
           "毎日少しずつ歩くようにしています。野菜を多めに食べることも心がけています。",
           "温かい緑茶がおすすめです。夏は冷たい麦茶もおいしいです。",
           "まず要らない物を捨てることが大切です。物の置き場所を決めると散らかりにくくなります。",
           "相手の好きな物を選ぶのが一番です。手紙を添えると気持ちが伝わります。",
           "毎日同じ時間に寝るようにするといいです。起きたらすぐにカーテンを開けて光を浴びます。"]
EN = {"name": "english",
      "prompts": ["What is a good way to spend the weekend?", "How do I make a simple curry?",
                  "Any tips for falling asleep faster?", "What should I give a friend for their birthday?"],
      "replies": ["Going for a walk in the park is a nice way to relax. Reading a book at home is also good.",
                  "Fry vegetables and meat, then simmer them in water. Add the curry roux at the end.",
                  "Avoid screens before bed. A warm drink can also help you relax.",
                  "Choose something they like. Adding a handwritten note makes it special."]}
POS = "before:(です|ます)。"
LAYERS = [8, 14, 20]


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip(f"{MODEL} がありません")
    from gne.api import Session
    s = Session(MODEL, root=tmp_path_factory.mktemp("c07k"), n_gpu_layers=NGL, n_ctx=1024)
    yield s
    s.close()


@pytest.fixture(scope="module")
def spec_b(tmp_path_factory):
    """側 A の作り方 (b) の対照集合: system なしで与えた返事を読み、「です。」「ます。」を書く直前を側 A にする。"""
    d = tmp_path_factory.mktemp("c07k_spec")
    return _write(d / "plain_end.yaml", {"name": "plain_end", "positions": POS, "user_prompts": PROMPTS,
                                         "replies": REPLIES, "dont_touch": [EN]})


@pytest.fixture(scope="module")
def key_b(session, spec_b):
    from gne.keyed_build import build_keys
    return build_keys(session, spec_b, name="plain_end", layers=LAYERS, max_tokens=48, q_energy=1.0, pseudo=16)


@pytest.mark.model
def test_build_saved_fields_and_kappa_mu(session, spec_b, key_b):
    """保存項目、κᵀμ = 1(作成に使った例)、q_energy = 1 なら側 B の例ごとの平均に κ が直交する(V31 の作成側)。"""
    from gne.keyed import example_hash, load_key
    r = key_b
    assert r["mode"] == "same" and r["positions_b"] == "not-before:(です|ます)。" and r["matched"] == {"A": 8, "of": 8}
    assert not r["dropped"] and [row["layer"] for row in r["layers"]] == LAYERS
    assert r["counts"]["a"] == 8 and r["counts"]["b"] == 8 and r["counts"]["dt"] == 2
    assert r["counts"]["rows_a"] >= 16 and r["counts"]["rows_b"] > r["counts"]["rows_a"]
    for row in r["layers"]:
        assert abs(row["kappa_mu"] - 1) < 1e-9 and 0 < row["kn"] <= 1 and row["q_rank"] == 8 + 2 + 2   # 側 B・英語・英語の返事の冒頭
    ks = load_key(session, "plain_end")
    n_ff = session.meta.n_ff_at(8)
    assert ks.target == "ffn" and ks.layers == LAYERS and ks.in_dim == n_ff
    assert ks.q_mode == "examples" and ks.positions == POS and ks.skip == 0 and ks.max_tokens == 48 and ks.subset == "even"
    cs, _ = load_key_spec(spec_b)
    assert ks.contrast_sha == cs.sha and ks.model_id == session.meta.model_id
    assert ks.hashes_a == [example_hash(None, PROMPTS[i]) for i in range(0, 16, 2)] == ks.hashes_b
    assert ks.n_pseudo == 16 and ks.info["mode"] == "same" and ks.info["pseudo_method"] == "balanced"
    assert ks.info["replies"] == {"AB": "given", "dt:english": "given"} and ks.info["iv"] is None
    for L in LAYERS:
        k, mu = ks.kappa[L].astype(np.float64), ks.mu[L].astype(np.float64)
        assert abs(k @ mu - 1) < 1e-3                                        # κ は float32 で保存
        assert ks.exa[L].shape == (8, n_ff) and ks.exb[L].shape == (8, n_ff) and ks.kappa_ctrl[L].shape == (16, n_ff)
        np.testing.assert_allclose(ks.exa[L].astype(np.float64).mean(axis=0), mu, rtol=1e-4, atol=1e-5)
        assert np.max(np.abs(ks.exb[L].astype(np.float64) @ k)) < 1e-3       # 側 B の例の平均は Q に入っている
        for j in range(16):                                                  # 擬似鍵もそれぞれ自分の μ で 1(0 にした本は無い)
            assert np.linalg.norm(ks.kappa_ctrl[L][j]) > 0
        assert not ks.rms_a                                                  # Qwen3 は post-norm ではない


@pytest.mark.model
def test_q_modes_means_examples_samples(session, spec_b, key_b):
    from gne.keyed import load_key
    from gne.keyed_build import build_keys
    rm = build_keys(session, spec_b, name="plain_means", layers=[14], q_mode="means", pseudo=4)
    rs = build_keys(session, spec_b, name="plain_samples", layers=[14], q_mode="samples", q_max_samples=64, q_energy=1.0,
                    pseudo=4)
    ex = next(row for row in key_b["layers"] if row["layer"] == 14)
    m, s = rm["layers"][0], rs["layers"][0]
    assert m["q_rank"] == 3 and m["q_rows"] == 3                             # 側 B・dont_touch の群・その返事の冒頭の平均
    assert s["q_rows"] == 64 and s["q_rank"] == 64                           # トークンの行を 64 行に間引いた
    assert m["kn"] >= ex["kn"] - 1e-9                                        # 群の平均は例の平均の張る空間の中
    for row in (m, s):
        assert abs(row["kappa_mu"] - 1) < 1e-9 and row["build_auc"] > 0.9
    assert load_key(session, "plain_samples").q_mode == "samples"
    assert load_key(session, "plain_samples").info["q_rows"] == {"14": 64}


@pytest.mark.model
def test_key_check_held_out_v34_v35(session, spec_b, key_b, tmp_path):
    """V35: 保留した例(subset odd)で、本物の鍵の選択性(ρ_A − ρ_B)とトークン単位の AUC が擬似鍵の p95 を超える。
    V34: 作成と同じ例を含む集合で測ると警告して held_out: false(strict なら BAD_ARGS)。"""
    from gne.keyed_build import key_check
    c = key_check(session, "plain_end", contrast=spec_b)
    assert c["held_out"] and c["subset"] == "odd" and c["overlap"]["hash_overlap"] == 0 and not c["warnings"]
    assert c["mode"] == "same" and c["positions"] == POS and c["counts"]["A"]["examples"] == 8
    assert c["counts"]["dt:english"]["examples"] == 2
    best = next(row for row in c["layers"] if row["layer"] == c["best_layer"])
    assert best["auc"] > 0.9 and best["auc_token"] > 0.9
    assert best["beats_pseudo"] and best["beats_pseudo_auc"]
    assert best["selectivity"] > best["pseudo_selectivity"]["p95"]
    assert best["auc_token"] > best["pseudo_auc_token"]["p95"]
    for row in c["layers"]:
        assert row["rho"]["A"]["median"] > 0.5 > row["rho"]["B"]["abs_median"]          # 発火位置で 1 前後、それ以外で 0 前後
        assert abs(row["dont_touch_median"]["english"]) < 0.3
        assert row["pseudo_auc"]["n"] == 16 and len(row["values"]["A"]) == 8
        assert sum(row["hist"]["A"]) == row["rho"]["A_token"]["n"]
    # V34: 作成と同じ subset
    c2 = key_check(session, "plain_end", contrast=spec_b, subset="even")
    assert not c2["held_out"] and c2["overlap"]["subset_overlap"] and c2["overlap"]["hash_overlap"] == 8
    assert any("重なって" in w for w in c2["warnings"]) and any("held_out: false" in n for n in c2["notes"])
    with pytest.raises(GneError) as ei:
        key_check(session, "plain_end", contrast=spec_b, subset="even", strict=True)
    assert ei.value.code == "BAD_ARGS"
    # 別のファイルでも同じ (system, プロンプト) が入っていれば重なりを数える
    other = tmp_path / "copy.yaml"
    other.write_text("# 写し\n" + spec_b.read_text(encoding="utf-8"), encoding="utf-8")
    c3 = key_check(session, "plain_end", contrast=other, subset="all", layers=[14])
    assert not c3["held_out"] and not c3["overlap"]["same_contrast"] and c3["overlap"]["hash_overlap"] == 8
    # 対照集合を渡さないと、保存した作成時の例ごとの平均で測る(捕捉なし、甘い値と明記)
    c4 = key_check(session, "plain_end", layers=[14])
    assert not c4["held_out"] and c4["layers"][0]["auc"] == 1.0 and any("作成時" in n for n in c4["notes"])


@pytest.mark.model
def test_mode_a_system_last_no_generation(session, tmp_path):
    """側 A の作り方 (a): system_a(語尾の指示)の文脈 対 system なし。positions last は返事を作らない。"""
    from gne.keyed import load_key
    from gne.keyed_build import build_keys, key_check
    spec = _write(tmp_path / "noda_sys.yaml", {"name": "noda_sys", "system_a": "文の終わりは必ず「〜のだ」「〜なのだ」にしてください。",
                                               "user_prompts": PROMPTS[:8]})
    r = build_keys(session, spec, layers=[12], positions="last", pseudo=4)
    assert r["mode"] == "system" and r["positions"] == "last" and r["positions_b"] == "last"
    assert r["replies"] == {"A": None, "B": None} and r["counts"]["rows_a"] == 4 and r["counts"]["rows_b"] == 4
    assert abs(r["layers"][0]["kappa_mu"] - 1) < 1e-9
    ks = load_key(session, "noda_sys")
    assert set(ks.hashes_a).isdisjoint(ks.hashes_b)                          # system が違うので例のハッシュも違う
    c = key_check(session, "noda_sys", contrast=spec)
    assert c["held_out"] and c["mode"] == "system" and c["layers"][0]["auc"] is not None
    # 側 A と側 B が同じ対照集合を mode system で使うと断る
    same = _write(tmp_path / "same.yaml", {"name": "same", "user_prompts": PROMPTS[:4]})
    with pytest.raises(GneError) as ei:
        build_keys(session, same, layers=[12], positions="last")
    assert ei.value.code == "BAD_CONTRAST" and "before:" in ei.value.hint


@pytest.mark.model
def test_collect_acts_generation_and_iv(session, tmp_path):
    """返事を生成する捕捉と、介入(iv、ここでは層 4 のニューロンの gain)を当てた捕捉。iv は鍵の記録にも残る。"""
    from gne.keyed import load_key
    from gne.keyed_build import build_keys, collect_acts, key_check
    g = collect_acts(session, PROMPTS[:2], layers=[10], positions="response", max_tokens=16)
    assert g["replies"] == "generated" and all(t for t in g["texts"]) and all(1 <= n <= 16 for n in g["n_reply"])
    assert g["groups"]["A"]["means"][10].shape == (2, session.meta.n_ff_at(10))
    iv = ["4:100:x0", "4:200:x0", "4:300:x3"]
    a0 = collect_acts(session, PROMPTS[:2], layers=[10], replies=REPLIES[:2])
    a1 = collect_acts(session, PROMPTS[:2], layers=[10], replies=REPLIES[:2], iv=iv)
    assert a0["replies"] == "given" and a1["iv"]["units"]
    assert a0["groups"]["A"]["rows"] == a1["groups"]["A"]["rows"]
    assert not np.allclose(a0["groups"]["A"]["means"][10], a1["groups"]["A"]["means"][10])
    spec = _write(tmp_path / "noda_iv.yaml", {"name": "noda_iv", "system_a": "文の終わりは必ず「〜のだ」にしてください。",
                                              "user_prompts": PROMPTS[:4]})
    build_keys(session, spec, layers=[12], positions="last", pseudo=2, iv=iv)
    assert load_key(session, "noda_iv").info["iv"]["units"]
    c = key_check(session, "noda_iv", contrast=spec, subset="odd")
    assert any("介入" in n for n in c["notes"])


@pytest.mark.model
def test_build_refusals(session, spec_b):
    from gne.keyed_build import build_keys
    last = session.meta.n_layer - 1
    for kw, code in (({"layers": [last]}, "BAD_RECIPE"),                    # 最終層は行が間引かれる
                     ({"layers": []}, "BAD_ARGS"),
                     ({"layers": [8], "target": "shared"}, "BAD_RECIPE"),     # dense の層に shared expert は無い
                     ({"layers": [8], "positions": "not-before:です"}, "BAD_ARGS"),
                     ({"layers": [8], "q_mode": "all"}, "BAD_ARGS"),
                     ({"layers": [8], "positions": "before:存在しない語尾"}, "BAD_CONTRAST")):
        with pytest.raises(GneError) as ei:
            build_keys(session, spec_b, name="refused", **kw)
        assert ei.value.code == code, (kw, ei.value.message)


# --- 返事の書き出しと、側 A が少ないときの警告(測定で見つかった問題の手当て)----------------------------------------------

def test_start_positions_by_kind():
    """返事の書き出し = 返事の最初の 3 トークンを決める位置(P−1 … P+1)。before は自分の位置のうち書き出しに入るもの、
    not-before は書き出しから一致の位置を除いたもの(skip に関係なく)、response・last は書き出しのすべて。"""
    from gne.keyed_build import START_TOKENS, start_positions
    eng = FakeEngine(JA_PIECES)
    reply = list(range(len(JA_PIECES))) + [99]                           # 末尾の EOG は除く
    P = 5
    assert START_TOKENS == 3
    assert start_positions(eng, reply, "before:今日", n_prompt=P).tolist() == [4]          # 返事の最初の一致 → P−1
    assert start_positions(eng, reply, "before:今日", 1, n_prompt=P).tolist() == []       # skip より前は側 A にならない
    assert start_positions(eng, reply, "not-before:今日", 1, n_prompt=P).tolist() == [5, 6]
    assert start_positions(eng, reply, "before:(です|ます)。", n_prompt=P).tolist() == []  # 返事の途中の一致
    assert start_positions(eng, reply, "not-before:(です|ます)。", n_prompt=P).tolist() == [4, 5, 6]
    assert start_positions(eng, reply, "response", n_prompt=P).tolist() == [4, 5, 6]
    assert start_positions(eng, reply, "response", n_prompt=P, k=1).tolist() == [4]
    assert start_positions(eng, [], "last", n_prompt=P).tolist() == [4]                   # 返事なし: 生成プロンプトの最後だけ
    assert start_positions(eng, [0, 99], "response", n_prompt=P).tolist() == [4]          # 返事の長さまで
    assert start_positions(eng, reply, "response", n_prompt=1).tolist() == [1, 2]         # 列の位置 0(sink)は使わない


def test_few_side_a_threshold_and_messages():
    from gne.keyed_build import MIN_A_EXAMPLES, MIN_A_ROWS, few_side_a, few_side_a_message
    assert (MIN_A_EXAMPLES, MIN_A_ROWS) == (2, 8)
    assert few_side_a(1, 20) and few_side_a(2, 2) and few_side_a(6, 7) and not few_side_a(2, 8)
    m = few_side_a_message(2, 2)
    assert "2 例の 2 位置だけ" in m and "正規表現" in m and "8 位置以上" in m
    assert "正規表現" not in few_side_a_message(1, 1, mode="system")
    c = few_side_a_message(6, 7, where="L12: ", check=True)
    assert c.startswith("L12: key-check の書き込みたい位置が 6 例の 7 位置だけ") and "揺れる" in c


def test_reply_start_share_info_and_message():
    """μ に占める書き出しの重み = 側 A の例ごとの「書き出しの行 ÷ 行」の平均。mode same でだけ測る。書き出しの状態を Q に
    入れた dont_touch の群(返事の冒頭を自動で足した群か、positions last・assistant の群)があれば警告ではなく注の文になる。
    dont_touch の群が無ければ質問を足す案内、返事の冒頭の自動を切っていれば戻す案内の警告。"""
    from gne.keyed_build import START_SHARE_WARN, KeySpec, _reply_start_info, _reply_start_message, _start_share
    ga = {"start": {"k": 3, "n_l": {14: [1, 0, 2]}}, "count_l": {14: [1, 4, 4]}, "rows_l": {14: 9}}
    assert _start_share(ga, 14) == (pytest.approx(0.5), 3)
    cs = SimpleNamespace()
    plain = KeySpec(cs, [DontTouch("tasks", None, ["17 × 23 は？"])])
    info = _reply_start_info(plain, ga, [14], "same", "ffn", ["tasks"])          # tasks の返事の冒頭を自動で入れた
    assert info == {"k": 3, "share": 0.5, "rows": 3, "rows_a": 9, "layer": None, "dont_touch_starts": ["tasks"],
                    "dont_touch_start": True}
    assert info["share"] >= START_SHARE_WARN == 0.25
    msg = _reply_start_message(info, 1)
    assert "50%" in msg and "tasks" in msg and "返事を書き始める直前の位置は Q に" in msg and "繰り返し" in msg
    off = _reply_start_info(plain, ga, [14], "same", "ffn", None)               # 自動を切った
    assert off["dont_touch_starts"] == [] and off["dont_touch_start"] is False
    m_off = _reply_start_message(off, 1)
    assert "dont_touch_start: false" in m_off and "戻して" in m_off
    none = _reply_start_info(KeySpec(cs, []), ga, [14], "same", "ffn", [])      # dont_touch の群が無い
    m_none = _reply_start_message(none, 0)
    assert "dont_touch の群がありません" in m_none and "dont_touch に足して" in m_none
    starts = KeySpec(cs, [DontTouch("tasks", None, ["x"]), DontTouch("starts", None, ["今日の天気は？"], positions="last")])
    info2 = _reply_start_info(starts, ga, [14], "same", "ffn", None)            # 自動なしでも positions last の群は入る
    assert info2["dont_touch_starts"] == ["starts"]
    assert "starts" in _reply_start_message(info2, 2) and "繰り返し" in _reply_start_message(info2, 2)
    info3 = _reply_start_info(starts, ga, [14], "same", "ffn", ["tasks"])       # 自動の群と last の群(dont_touch の並び順)
    assert info3["dont_touch_starts"] == ["tasks", "starts"]
    assert _reply_start_info(plain, ga, [14], "system", "ffn") is None          # mode system では測らない
    # 専門家の鍵は層ごと(いちばん重い層を出す)
    ge = {"start": {"k": 3, "n_l": {6: [0, 0], 12: [1, 1]}}, "count_l": {6: [2, 2], 12: [2, 1]}, "rows_l": {6: 4, 12: 3}}
    e = _reply_start_info(plain, ge, [6, 12], "same", "expert")
    assert e["layer"] == 12 and e["share"] == pytest.approx(0.75) and "L12 の書き込みたい位置" in _reply_start_message(e, 1)


def test_start_rho_hot_and_messages():
    """key-check の start_rho: 側 B と dont_touch の書き出しの位置ごとの ρ の中央値。0.4 以上(例 2 つ以上)で警告の文。
    dont_touch なら触りたくない場面への漏れ、側 B だけなら場面の中の書き出しで書く(繰り返しに落ちやすい)という文。"""
    from gne.keyed_build import START_RHO_WARN, _start_hot, _start_hot_message, _start_rho
    assert START_RHO_WARN == 0.4

    def grp(rows, pos):
        return {"positions": pos, "start": {"k": 3, "proj": {14: [(np.asarray(o, dtype=np.int64),
                                                                    np.asarray([[v, 0.0] for v in vs])) for o, vs in rows]}}}
    G = {"A": grp([([0], [9.0])], "before:x"),
         "B": grp([([0, 1], [0.6, 0.1]), ([0], [0.5]), ([1, 2], [0.0, 0.2])], "not-before:x"),
         "dt:english": grp([([0, 1, 2], [0.1, 0.0, -0.1]), ([0, 1, 2], [0.2, 0.0, 0.0])], "not-before:x"),
         "dt:starts": grp([([0], [0.05]), ([0], [0.0])], "last")}
    sr = _start_rho(G, 14)
    assert sr["B"]["median"][0] == pytest.approx(0.55) and sr["B"]["n"] == [2, 2, 1] and sr["B"]["median"][2] == 0.2
    assert set(sr["dont_touch"]) == {"english", "starts"} and sr["dont_touch"]["starts"]["n"] == [2, 0, 0]
    assert "A" not in sr                                                    # 側 A は測らない
    pos = {g: d["positions"] for g, d in G.items()}
    hot = _start_hot([{"layer": 14, "start_rho": sr}], pos)
    assert hot == [{"layer": 14, "group": "B", "j": 0, "median": pytest.approx(0.55), "starts": False}]
    mb = _start_hot_message(hot)
    assert mb.startswith("書き込みたくない位置(") and "繰り返し" in mb
    G["dt:english"] = grp([([0], [0.7]), ([0], [0.9])], "not-before:x")
    hot2 = _start_hot([{"layer": 14, "start_rho": _start_rho(G, 14)}], pos)
    md = _start_hot_message(hot2)
    assert md.startswith("dont_touch の返事の冒頭で鍵が反応しています") and "english +0.80" in md
    assert "dont_touch に足して" in md and "自動で Q に" in md and "書き込みたくない位置の返事の冒頭でも" in md
    assert "その群の質問を増やして" not in md
    hot_auto = _start_hot([{"layer": 14, "start_rho": _start_rho(G, 14)}], pos, auto={"english": 2})
    assert [h["starts"] for h in hot_auto if h["group"] == "english"] == [True]     # 鍵の作成で返事の冒頭を入れた群
    assert "その群の質問を増やして" in _start_hot_message(hot_auto)
    G["dt:starts"] = grp([([0], [0.6]), ([0], [0.5])], "last")
    md2 = _start_hot_message(_start_hot([{"layer": 14, "start_rho": _start_rho(G, 14)}], pos))
    assert "その群の質問を増やして" in md2                                   # 書き出しを Q に入れた群でも合う
    assert _start_hot([{"layer": 14, "start_rho": None}], pos) == []


def test_start_rho_adds_dont_touch_reply_start():
    """返事を書き始める直前の位置を Q に入れて作った鍵の key-check: dont_touch の群 dts:<名前> の値を、その群の書き出しに位置 0 が
    無い例(返事の冒頭が positions の正規表現に一致して not-before が外した例と、not-before の位置が 1 つも無い例)の位置 0 に
    入れる。位置 0 がある例は同じ位置なので重ねない。dts の群の書き出し(返事の最初の 3 トークン)は使わない。dts の群が無ければ
    (古い鍵・切って作った鍵)今のまま。"""
    from gne.keyed_build import _start_hot, _start_hot_message, _start_rho

    def grp(rows, pos, idx):
        return {"positions": pos, "index_l": {14: idx},
                "start": {"k": 3, "proj": {14: [(np.asarray(o, dtype=np.int64), np.asarray([[v, 0.0] for v in vs])) for o, vs in rows]}}}
    # 例 0・1 は返事の冒頭が一致(位置 0 が無い)、例 2 は一致しない(位置 0 がある)、例 3 は not-before の位置が無い
    G = {"B": grp([([1], [0.0]), ([1], [0.1])], "not-before:x", [0, 1]),
         "dt:english": grp([([1, 2], [0.0, 0.1]), ([1], [0.05]), ([0, 1, 2], [0.2, 0.0, 0.0])], "not-before:x", [0, 1, 2])}
    old = _start_rho(G, 14)
    assert old["dont_touch"]["english"]["n"] == [1, 3, 2] and old["dont_touch"]["english"]["median"][0] == pytest.approx(0.2)
    assert _start_hot([{"layer": 14, "start_rho": old}], {"dt:english": "not-before:x"}, auto={"english": 2}) == []
    G["dts:english"] = {**grp([([0, 1, 2], [5.0, 5.0, 5.0])] * 4, "last", [0, 1, 2, 3]),   # 書き出しは使わない
                        "proj": {14: [np.asarray([[v, 0.0]]) for v in (0.7, 0.9, 0.2, 0.6)]}}
    new = _start_rho(G, 14)
    e = new["dont_touch"]["english"]
    assert set(new["dont_touch"]) == {"english"} and new["B"] == old["B"]
    assert e["n"] == [4, 3, 2] and e["median"][0] == pytest.approx(0.65) and e["max"][0] == pytest.approx(0.9)   # 0.2・0.7・0.9・0.6
    assert e["median"][1:] == old["dont_touch"]["english"]["median"][1:]
    hot = _start_hot([{"layer": 14, "start_rho": new}], {"dt:english": "not-before:x"}, auto={"english": 2})
    assert hot == [{"layer": 14, "group": "english", "j": 0, "median": pytest.approx(0.65), "starts": True}]
    msg = _start_hot_message(hot)
    assert "english +0.65(位置 0)" in msg and "その群の質問を増やして" in msg


def test_plan_adds_dont_touch_reply_start(tmp_path):
    """鍵の作成は、dont_touch の各群に、その質問の返事の冒頭(生成プロンプトの最後のトークン)の群 dts:<名前> を同じ捕捉で
    足す。群の positions が生成プロンプトの位置(last など)なら重ねず、側 B の返事の冒頭は足さない。対照集合の
    dont_touch_start を読む(true / false 以外は BAD_CONTRAST)。"""
    from gne.keyed_build import DT_START_POSITIONS, _plan, _resolve_mode
    assert DT_START_POSITIONS == "last"
    p = _write(tmp_path / "c.yaml", {"name": "c", "positions": "before:です。", "user_prompts": PROMPTS[:4],
                                     "dont_touch": [{"name": "tasks", "prompts": ["17 × 23 は？", "水の沸点は？"]},
                                                    {"name": "starts", "positions": "last", "prompts": ["今日の天気は？"]}]})
    spec = read_key_spec(p)
    assert spec.dont_touch_start is None
    pos_a = parse_positions(spec.positions)
    mode, pos_b = _resolve_mode(pos_a, None)
    sets, info = _plan(None, spec, mode, pos_a, pos_b, "all", None, None, None, None, lambda m: None, dt_start=True)
    g = {x.label: x.groups for x in sets}
    assert g["AB"] == {"A": "before:です。", "B": "not-before:です。"}           # 側 B の返事の冒頭は足さない
    assert g["dt:tasks"] == {"dt:tasks": "not-before:です。", "dts:tasks": "last"}
    assert g["dt:starts"] == {"dt:starts": "last"}                              # もう返事の冒頭の状態なので重ねない
    assert [(d["name"], d["start"]) for d in info["dont_touch"]] == [("tasks", "last"), ("starts", None)]
    sets0, info0 = _plan(None, spec, mode, pos_a, pos_b, "all", None, None, None, None, lambda m: None)
    assert {x.label: x.groups for x in sets0}["dt:tasks"] == {"dt:tasks": "not-before:です。"}   # eval・切った鍵の key-check は足さない
    assert all("start" not in d for d in info0["dont_touch"])
    off = _write(tmp_path / "off.yaml", {**yaml.safe_load(p.read_text(encoding="utf-8")), "dont_touch_start": False})
    assert read_key_spec(off).dont_touch_start is False
    bad = _write(tmp_path / "bad.yaml", {**yaml.safe_load(p.read_text(encoding="utf-8")), "dont_touch_start": "yes"})
    with pytest.raises(GneError) as e:
        read_key_spec(bad)
    assert e.value.code == "BAD_CONTRAST"


THANKS_POS = "before:ありがとう"


@pytest.mark.model
def test_reply_start_warning_note_and_key_check(session, tmp_path):
    """返事の最初で一致する側 A(「ありがとう！…」)。dont_touch の質問の返事の冒頭は自動で Q に入り(結果と鍵のファイルの
    dont_touch_start に行の数)、書き出しの知らせは注になる。自動を切ると警告(戻す案内)、dont_touch の群が無いと警告
    (質問を足す案内)。key-check は層ごとに側 B・dont_touch の start_rho を出す。記録の無い古い鍵も読める。"""
    from gne.keyed import key_dir, load_key
    from gne.keyed_build import build_keys, key_check
    replies = [("ありがとう！" + r) if i % 4 != 3 else r.replace("。", "、ありがとう。", 1) for i, r in enumerate(REPLIES)]
    base = {"name": "thanks_start", "positions": THANKS_POS, "user_prompts": PROMPTS, "replies": replies,
            "dont_touch": [EN]}
    p1 = _write(tmp_path / "thanks_start.yaml", base)
    r1 = build_keys(session, p1, name="thanks_start", layers=[14], q_energy=1.0, pseudo=4)
    rs = r1["reply_start"]
    assert rs["k"] == 3 and rs["share"] >= 0.5 and rs["rows"] >= 6
    assert rs["dont_touch_starts"] == ["english"] and rs["dont_touch_start"] is True
    assert r1["counts"]["rows_a_start"] == rs["rows"]
    assert r1["dont_touch_start"] == {"on": True, "positions": "last", "groups": {"english": 2}, "rows": 2}
    assert r1["counts"]["rows_dt_start"] == 2 and r1["counts"]["dt"] == 2               # dt の数には数えない
    assert [d.get("start") for d in r1["dont_touch"]] == ["last"]
    assert not any("返事の冒頭" in w for w in r1["warnings"]), r1["warnings"]
    assert any("返事の冒頭" in n and "english" in n for n in r1["notes"])
    assert load_key(session, "thanks_start").info["dont_touch_start"] == r1["dont_touch_start"]
    r1off = build_keys(session, p1, name="thanks_start_off", layers=[14], q_energy=1.0, pseudo=4, dont_touch_start=False)
    assert r1off["dont_touch_start"] == {"on": False, "positions": None, "groups": {}, "rows": 0}
    assert r1off["counts"]["rows_dt_start"] == 0 and r1off["reply_start"]["dont_touch_starts"] == []
    assert r1off["layers"][0]["q_rows"] == r1["layers"][0]["q_rows"] - 2                # examples: 英語 2 問の冒頭の 2 行
    assert any("dont_touch_start: false" in w for w in r1off["warnings"]), r1off["warnings"]
    p0 = _write(tmp_path / "thanks_start0.yaml", {k: v for k, v in base.items() if k != "dont_touch"} | {"name": "t0"})
    r0 = build_keys(session, p0, name="thanks_start0", layers=[14], q_energy=1.0, pseudo=4)
    assert r0["dont_touch_start"]["rows"] == 0
    assert any("dont_touch の群がありません" in w for w in r0["warnings"]), r0["warnings"]
    starts = {"name": "starts", "positions": "last", "prompts": ["今日の天気はどう？", "おすすめの本は？", "猫と犬どっちが好き？",
                                                               "好きな季節は？"]}
    p2 = _write(tmp_path / "thanks_start2.yaml", {**base, "name": "thanks_start2", "dont_touch": [EN, starts]})
    r2 = build_keys(session, p2, name="thanks_start2", layers=[14], q_energy=1.0, pseudo=4)
    assert r2["reply_start"]["dont_touch_starts"] == ["english", "starts"]
    assert r2["dont_touch_start"]["groups"] == {"english": 2}                           # starts は last なので重ねない
    assert not any("返事の冒頭" in w for w in r2["warnings"])
    assert any("返事の冒頭" in n and "starts" in n for n in r2["notes"])
    c = key_check(session, "thanks_start2", contrast=p2)
    sr = c["layers"][0]["start_rho"]
    assert set(sr) == {"B", "dont_touch"} and set(sr["dont_touch"]) == {"english", "starts"}
    assert len(sr["B"]["median"]) == 3 and sr["dont_touch"]["starts"]["n"][1:] == [0, 0]
    assert sr["dont_touch"]["english"]["n"][0] == 2                          # 保留した英語 2 問の書き出し
    # 返事の冒頭の記録が無い古い鍵(この変更の前に作った鍵)も読めて、key-check も回る
    old = load_key(session, "thanks_start2")
    old.info.pop("dont_touch_start")
    old.counts.pop("rows_dt_start")
    old.name = "thanks_old"
    old.save(key_dir(session) / "thanks_old.npz")
    back = load_key(session, "thanks_old")
    assert "dont_touch_start" not in back.info and back.summary()["dont_touch_start"] is None
    assert key_check(session, "thanks_old", contrast=p2)["layers"][0]["start_rho"]
    from gne.schemas import DirectionKeyCheckOut, DirectionKeyOut, DirectionKeysOut
    env = {"schema_version": "1", "command": "x", "model_id": "m", "llama_cpp_version": "0", "seed": 0,
           "generated_at": "2026-11-08T00:00:00+00:00", "warnings": []}
    for r in (r1, r1off, r0):
        DirectionKeyOut.model_validate({**env, **{k: v for k, v in r.items() if k != "warnings"}})
    DirectionKeyCheckOut.model_validate({**env, **{k: v for k, v in c.items() if k != "warnings"}})
    DirectionKeysOut.model_validate({**env, "keys": session.keys()})


@pytest.mark.model
def test_few_side_a_warns_on_rows(session, tmp_path):
    """側 A が 2 例あっても 8 位置未満なら警告する(ツンデレの最初の版は 2 例・2 位置で警告が出なかった)。key-check も同じ基準。"""
    from gne.keyed_build import build_keys, key_check
    replies = list(REPLIES)
    for i in (0, 2, 1, 3):                                                   # 作成用(偶数番目)に 2 つ、保留用(奇数番目)に 2 つ
        replies[i] = replies[i].replace("。", "、ありがとう。", 1)
    p = _write(tmp_path / "few.yaml", {"name": "few_a", "positions": THANKS_POS, "user_prompts": PROMPTS, "replies": replies,
                                       "dont_touch": [EN]})
    r = build_keys(session, p, name="few_a", layers=[14], q_energy=1.0, pseudo=4)
    assert r["counts"]["a"] == 2 and r["counts"]["rows_a"] == 2
    assert any("書き込みたい位置が 2 例の 2 位置だけ" in w for w in r["warnings"]), r["warnings"]
    c = key_check(session, "few_a", contrast=p)
    assert any("key-check の書き込みたい位置が 2 例の 2 位置だけ" in w for w in c["warnings"]), c["warnings"]


HEAD_POS = "before:(?s)\\A."            # 返事の最初の文字: どの返事でも返事の冒頭が一致する(書き込みたい位置 = 返事を書き始める直前の位置)


@pytest.mark.model
def test_key_check_measures_dont_touch_reply_start(session, tmp_path):
    """positions が返事の冒頭に一致するとき、鍵の作成は dont_touch の質問の返事を書き始める直前の位置を Q に入れるが、群の
    positions(not-before)はこの位置を外す。key-check も鍵の記録(dont_touch_start)に従ってこの位置を測る: counts の
    dts:<群>、結果の dont_touch_start、層ごとの rho.dont_touch_start・dont_touch_start_median、start_rho の位置 0。AUC・選択性・
    dont_touch の ρ の中央値・ヒストグラムには入れない。漏れていれば start_rho の警告が出る。記録の無い古い鍵と、切って作った
    鍵では今のまま(この位置を測らない)。CLI の表にも出る。"""
    from typer.testing import CliRunner

    from gne.cli import app
    from gne.keyed import KeySet, key_dir
    from gne.keyed_build import build_keys, key_check
    from gne.schemas import DirectionKeyCheckOut
    base = {"name": "head_start", "positions": HEAD_POS, "user_prompts": PROMPTS, "replies": REPLIES, "dont_touch": [EN]}
    p = _write(tmp_path / "head_start.yaml", base)
    r = build_keys(session, p, name="head_start", layers=[14], q_energy=1.0, pseudo=4)
    assert r["dont_touch_start"] == {"on": True, "positions": "last", "groups": {"english": 2}, "rows": 2}
    assert r["reply_start"]["share"] == 1.0 and r["counts"]["rows_a"] == 8
    c = key_check(session, "head_start", contrast=p)                         # subset odd: 保留した日本語 8 問・英語 2 問
    assert c["held_out"] and c["counts"]["dts:english"] == {"examples": 2, "rows": 2, "of": 2}
    assert c["dont_touch_start"] == {"on": True, "recorded": True, "positions": "last", "groups": {"english": 2}, "rows": 2}
    row = c["layers"][0]
    e = row["start_rho"]["dont_touch"]["english"]
    assert e["n"][0] == 2                                                    # 返事の冒頭が一致しても位置 0 を測る
    assert row["dont_touch_start_median"]["english"] == pytest.approx(e["median"][0])
    assert row["rho"]["dont_touch_start"]["english"]["n"] == 2
    assert set(row["values"]) == {"A", "B", "dt:english"} and "dts:english" not in row["hist"]
    env = {"schema_version": "1", "command": "x", "model_id": "m", "llama_cpp_version": "0", "seed": 0,
           "generated_at": "2026-11-09T00:00:00+00:00", "warnings": []}
    DirectionKeyCheckOut.model_validate({**env, **{k: v for k, v in c.items() if k != "warnings"}})
    # 記録の無い古い鍵と、切って作った鍵: この位置は測らず、ほかの値は同じ
    old = KeySet.load(key_dir(session) / "head_start.npz")                  # load_key のキャッシュの鍵は書き換えない
    old.info.pop("dont_touch_start")
    old.counts.pop("rows_dt_start")
    old.name = "head_old"
    old.save(key_dir(session) / "head_old.npz")
    build_keys(session, p, name="head_off", layers=[14], q_energy=1.0, pseudo=4, dont_touch_start=False)
    for nm, rec in (("head_old", False), ("head_off", True)):
        co = key_check(session, nm, contrast=p)
        assert not any(g.startswith("dts:") for g in co["counts"]), nm
        assert co["dont_touch_start"] == {"on": False, "recorded": rec, "positions": None, "groups": {}, "rows": 0}
        ro = co["layers"][0]
        assert ro["start_rho"]["dont_touch"]["english"]["n"][0] == 0 and ro["start_rho"]["dont_touch"]["english"]["median"][0] is None
        assert "dont_touch_start_median" not in ro and "dont_touch_start" not in ro["rho"]
        if nm == "head_old":                                                 # 同じ κ: 位置 0 のほかは同じ値
            for k in ("auc", "auc_token", "selectivity", "dont_touch_median", "values", "hist"):
                assert ro[k] == row[k], k
            assert ro["start_rho"]["B"] == row["start_rho"]["B"] and ro["start_rho"]["dont_touch"]["english"]["n"][1:] == e["n"][1:]
            assert ro["rho"] == {k: v for k, v in row["rho"].items() if k != "dont_touch_start"}
    # 漏れの警告: 確かめる対照集合の dont_touch を、鍵を作った場面の質問と返事そのものにする(返事を書き始める直前の状態が、作成に
    # 使った書き込みたい位置と同じ。subset even なら、その位置での ρ の平均は κᵀμ = 1)
    same = _write(tmp_path / "head_same.yaml", {**base, "dont_touch": [{"name": "same", "prompts": PROMPTS, "replies": REPLIES}]})
    cs = key_check(session, "head_start", contrast=same, subset="even")
    rs = cs["layers"][0]
    a = np.asarray(rs["values"]["A"])
    s0 = rs["start_rho"]["dont_touch"]["same"]
    assert len(a) == 8 and float(a.mean()) == pytest.approx(1.0, abs=1e-3)
    assert s0["n"][0] == 8 and s0["median"][0] == pytest.approx(float(np.median(a)), abs=1e-4)
    assert s0["median"][0] >= 0.4, s0
    assert any(w.startswith("dont_touch の返事の冒頭で鍵が反応しています") and "same" in w and "(位置 0)" in w
               for w in cs["warnings"]), cs["warnings"]
    co = key_check(session, "head_old", contrast=same, subset="even")      # 古い鍵では測らないので、この警告は出ない
    assert co["layers"][0]["start_rho"]["dont_touch"]["same"]["n"][0] == 0
    assert not any("same" in w and "(位置 0)" in w for w in co["warnings"]), co["warnings"]
    # CLI の表: 返事の冒頭の ρ に english の位置 0 が入り、測ったことを 1 行で知らせる
    cli = CliRunner().invoke(app, ["direction", "key-check", "--key", "head_start", "--contrast", str(p), "--model", str(MODEL),
                                   "--root", str(session.store.root), "--n-gpu-layers", str(NGL), "--n-ctx", "1024"])
    assert cli.exit_code == 0, cli.output
    assert "dont_touch の返事を書き始める直前の位置: 鍵の作成と同じく測りました(english 2 位置" in cli.output
    line = next(ln for ln in cli.output.splitlines() if ln.strip().startswith("14 "))
    assert f"english {e['median'][0]:+.2f}" in line.split("返事の冒頭の ρ")[1]
    cli = CliRunner().invoke(app, ["direction", "key-check", "--key", "head_old", "--contrast", str(p), "--model", str(MODEL),
                                   "--root", str(session.store.root), "--n-gpu-layers", str(NGL), "--n-ctx", "1024"])
    assert cli.exit_code == 0 and "記録の無い古い鍵なので" in cli.output
