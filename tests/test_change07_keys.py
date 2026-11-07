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
    for t in ("shared", "expert:3"):                     # attn は tests/test_change07_attn.py
        with pytest.raises(GneError) as ei:
            target_capture(None, t, [3])
        assert ei.value.code == "ARCH_UNSUPPORTED"


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
        assert abs(row["kappa_mu"] - 1) < 1e-9 and 0 < row["kn"] <= 1 and row["q_rank"] == 8 + 2
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
    assert m["q_rank"] == 2 and m["q_rows"] == 2                             # 側 B と dont_touch の群の平均
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
