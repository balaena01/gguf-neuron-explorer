"""CHANGE-07 のテスト(レシピの記法): 鍵つきの書き込み(mode: keyed)のユニット・短縮記法・YAML・scaled・3 列チャット・
レシピの重ね合わせ・確かめの誤り。モデルは使わない。

例は語尾(ずんだもん)と性格(ツンデレ)だけ。方向・鍵の名前はファイルを読まないので、存在しなくてよい。
"""

import numpy as np
import pytest
import yaml

from gne.steer import AblateSpec, GneError, Recipe, Unit, combine_recipes, fold_sign, parse_unit, unit_details


def _keyed(L=19, kind="ffn", alpha=1.25, direction="zundamon_r", key="zundamon_k", expert=None):
    return Unit(L, None, 0.0, kind, mode="keyed", k=alpha, direction=direction, expert=expert, key=key)


# --- 短縮記法 -------------------------------------------------------------------------------------------

def test_keyed_notation_every_kind():
    u = parse_unit("19:ffn:keyed:a1.25@zundamon_r#zundamon_k")
    assert (u.layer, u.kind, u.mode, u.k, u.direction, u.key, u.expert, u.neuron) == \
        (19, "ffn", "keyed", 1.25, "zundamon_r", "zundamon_k", None, None)
    a = parse_unit("20:attn:keyed:a0.5@tsun_r#tsun_k")
    assert (a.kind, a.k, a.direction, a.key) == ("attn", 0.5, "tsun_r", "tsun_k")
    s = parse_unit("12:S:keyed:a2@zundamon_r#zundamon_k")
    assert (s.kind, s.k, s.expert, s.neuron) == ("shared", 2.0, None, None)
    e = parse_unit("12:E37:keyed:a1e-1@zundamon_r#zundamon_k")
    assert (e.kind, e.expert, e.k) == ("expert", 37, pytest.approx(0.1))
    # 大文字・小文字は問わない(他の記法と同じ)
    assert parse_unit("3:FFN:KEYED:A1@x#k").kind == "ffn" and parse_unit("3:s:keyed:a1").kind == "shared"
    assert parse_unit("3:e2:keyed:a1").expert == 2


def test_keyed_notation_defaults_pseudo_key_and_sign():
    u = parse_unit("19:ffn:keyed:a1.25")                          # @DIR・#KEY を省くとレシピの既定
    assert u.direction is None and u.key is None and u.k == 1.25
    assert parse_unit("19:ffn:keyed:a1#zundamon_k").direction is None
    assert parse_unit("19:ffn:keyed:a1@zundamon_r").key is None
    p = parse_unit("19:ffn:keyed:a1.25@zundamon_r#zundamon_k~3")  # 擬似鍵(ランダム対照)
    assert p.key == "zundamon_k~3"
    n = parse_unit("19:ffn:keyed:a1.25@-zundamon_r#zundamon_k")   # 先頭の - は α の符号に畳む
    assert n.direction == "zundamon_r" and n.k == -1.25
    assert parse_unit("19:ffn:keyed:a-1.25@-zundamon_r").k == 1.25
    assert parse_unit("19:ffn:keyed:a-2@zundamon_r").k == -2.0
    assert parse_unit("19:ffn:keyed:a1@--zundamon_r").k == 1.0    # - が 2 つなら元の符号
    for bad in ("19:ffn:keyed:1.25@x#k",                          # a が無い
                "19:ffn:keyed:a1@x#k~", "19:ffn:keyed:a1@x#k~j",  # 擬似鍵の番号は数字
                "19:mlp:keyed:a1", "19:E:keyed:a1", "19:ffn:keyed:a1#k@x"):
        with pytest.raises(GneError) as ei:
            parse_unit(bad)
        assert ei.value.code == "BAD_UNIT"


def test_fold_sign():
    assert fold_sign("-tsun_r", 1.5) == ("tsun_r", -1.5)
    assert fold_sign("tsun_r", -1.5) == ("tsun_r", -1.5)
    assert fold_sign("--tsun_r", 2.0) == ("tsun_r", 2.0)
    assert fold_sign(None, 2.0) == (None, 2.0)
    assert fold_sign("-tsun_r", None) == ("tsun_r", None)


def test_keyed_label_and_existing_labels():
    assert parse_unit("19:ffn:keyed:a1.25@zundamon_r#zundamon_k").label() == "L19 ffn keyed a1.25 @zundamon_r #zundamon_k"
    assert parse_unit("20:attn:keyed:a-0.5").label() == "L20 attn keyed a-0.5"
    assert parse_unit("12:S:keyed:a2#k~1").label() == "L12 S keyed a2 #k~1"
    assert parse_unit("12:E37:keyed:a1@r").label() == "L12 E37 keyed a1 @r"
    assert _keyed(alpha=None).label().startswith("L19 ffn keyed a?")
    # 既存の記法の表示は変わらない
    assert parse_unit("12:S:N511:+0.8").label() == "L12 S N511 +0.8"
    assert parse_unit("33:H16:s15.4@tsundere_r").label() == "L33 H16 s15.4 @tsundere_r"
    assert parse_unit("14:8949:x2").label() == "L14 N8949 ×2"


# --- YAML の往復 --------------------------------------------------------------------------------------

def _recipe_all_kinds():
    units = [parse_unit("19:ffn:keyed:a1.25@zundamon_r#zundamon_k"),
             parse_unit("20:ffn:keyed:a-0.75"),                                 # 方向・鍵はレシピの既定
             parse_unit("21:attn:keyed:a0.5@tsun_r#tsun_k~2"),
             parse_unit("12:S:keyed:a2#zundamon_k"),
             parse_unit("12:E37:keyed:a1.5@tsun_r"),
             parse_unit("33:H16:s15.4@tsundere_r"), parse_unit("14:8949:+1.0"), parse_unit("14:77:x0")]
    return Recipe(units, "m", name="zundamon", default_direction="zundamon_r", default_key="zundamon_k")


def test_keyed_yaml_roundtrip_is_lossless(tmp_path):
    r = _recipe_all_kinds()
    r.validate()
    r.save(tmp_path / "r.yaml")
    d = yaml.safe_load((tmp_path / "r.yaml").read_text(encoding="utf-8"))
    assert d["default_key"] == "zundamon_k" and d["default_direction"] == "zundamon_r"
    assert d["units"][0] == {"layer": 19, "kind": "ffn", "mode": "keyed", "direction": "zundamon_r",
                             "key": "zundamon_k", "alpha": 1.25}
    assert d["units"][1] == {"layer": 20, "kind": "ffn", "mode": "keyed", "alpha": -0.75}
    assert d["units"][4] == {"layer": 12, "kind": "expert", "mode": "keyed", "direction": "tsun_r", "expert": 37, "alpha": 1.5}
    for x in d["units"][:5]:
        assert "k" not in x and "dose" not in x
    assert d["units"][5] == {"layer": 33, "kind": "head", "mode": "proj", "direction": "tsundere_r", "head": 16, "s": 15.4}
    assert all("key" not in x for x in d["units"][5:])                       # keyed 以外に key は書かない
    back = Recipe.load(tmp_path / "r.yaml")
    assert back.units == r.units and back.default_key == "zundamon_k" and back.default_direction == "zundamon_r"
    assert back.keyed_units == r.units[:5] and back.add_units == [r.units[6]] and back.gain_units == [r.units[7]]
    back.save(tmp_path / "r2.yaml")                                          # 2 回目も同じ
    assert (tmp_path / "r2.yaml").read_text(encoding="utf-8") == (tmp_path / "r.yaml").read_text(encoding="utf-8")


def test_keyed_yaml_spec_form_k_and_minus_direction(tmp_path):
    (tmp_path / "s.yaml").write_text(
        "default_direction: zundamon_r\ndefault_key: zundamon_k\nunits:\n"
        "  - {layer: 19, kind: ffn, mode: keyed, direction: zundamon_r, key: zundamon_k, alpha: 1.25}\n"
        "  - {layer: 20, kind: ffn, mode: keyed, direction: \"-zundamon_r\", key: zundamon_k, alpha: 1.25}\n"
        "  - {layer: 21, kind: attn, mode: keyed, k: 0.5}\n"
        "  - {layer: 12, kind: expert, expert: 3, mode: keyed, alpha: -2, direction: \"-tsun_r\", key: \"tsun_k~4\"}\n",
        encoding="utf-8")
    r = Recipe.load(tmp_path / "s.yaml")
    assert [u.k for u in r.units] == [1.25, -1.25, 0.5, 2.0]
    assert [u.direction for u in r.units] == ["zundamon_r", "zundamon_r", None, "tsun_r"]
    assert r.units[3].key == "tsun_k~4" and r.units[3].expert == 3
    r.save(tmp_path / "s2.yaml")
    assert Recipe.load(tmp_path / "s2.yaml").units == r.units


def test_to_dict_and_schema():
    from gne.schemas import RecipeModel
    r = _recipe_all_kinds()
    d = r.to_dict()
    assert d["default_key"] == "zundamon_k"
    assert d["units"][0]["key"] == "zundamon_k" and d["units"][0]["k"] == 1.25
    assert all("key" not in u for u in d["units"][5:])                       # keyed 以外の記録は前と同じ形
    RecipeModel.model_validate(d)
    plain = Recipe([parse_unit("14:8949:+1.0")]).to_dict()
    assert "default_key" not in plain and set(plain["units"][0]) == {
        "layer", "neuron", "dose", "kind", "seed", "mode", "k", "direction", "head", "seek", "expert"}


# --- scaled ------------------------------------------------------------------------------------------

def test_scaled_keeps_every_field_and_scales_alpha():
    r = _recipe_all_kinds()
    r.ablate = AblateSpec("tsun_r", 0.5, "10-12", 11)
    h = r.scaled(0.5)
    for a, b in zip(r.units[:5], h.units[:5], strict=True):
        assert (b.kind, b.direction, b.key, b.expert, b.layer, b.mode) == (a.kind, a.direction, a.key, a.expert, a.layer, a.mode)
        assert b.k == pytest.approx(a.k * 0.5)
    assert h.default_key == "zundamon_k" and h.default_direction == "zundamon_r"
    assert h.ablate.scale == pytest.approx(0.25) and h.ablate.layers == "10-12" and h.ablate.from_layer == 11
    # 他のモードは前と同じ: proj は s × k、add は dose × k、gain は 1 から k_unit への間を k の割合
    assert h.units[5].k == pytest.approx(7.7) and h.units[5].direction == "tsundere_r" and h.units[5].head == 16
    assert h.units[6].dose == pytest.approx(0.5) and h.units[6].neuron == 8949
    assert h.units[7].k == pytest.approx(0.5) and h.units[7].dose == 0.0
    q = Unit(15, None, 0.0, "head", mode="query", k=2.0, direction="tsun_r", head=2, seek="x.yaml")
    rt = Unit(12, None, 1.5, "expert", mode="route", expert=4)
    hq, hr = Recipe([q, rt]).scaled(3.0).units
    assert hq.k == 6.0 and hq.seek == "x.yaml" and hq.head == 2 and hr.dose == 4.5 and hr.expert == 4
    assert r.units[0].k == 1.25                                              # 元のレシピは変わらない
    assert r.scaled(0.0).units[0].k == 0.0


# --- 3 列チャット ---------------------------------------------------------------------------------------

def test_chat_minus_column_flips_alpha():
    from gne.chat import make_columns
    r = Recipe([parse_unit("19:ffn:keyed:a1.25@zundamon_r#zundamon_k"), parse_unit("12:E3:keyed:a2"),
                parse_unit("14:8949:+0.4"), parse_unit("33:H16:s2@tsun_r")],
               name="z", default_direction="zundamon_r", default_key="zundamon_k")
    lo, mid, hi = make_columns(r)
    assert mid.recipe is None
    assert [u.k for u in lo.recipe.units[:2]] == [-1.25, -2.0] and [u.k for u in hi.recipe.units[:2]] == [1.25, 2.0]
    for col in (lo, hi):
        k0, k1 = col.recipe.units[:2]
        assert (k0.mode, k0.kind, k0.direction, k0.key) == ("keyed", "ffn", "zundamon_r", "zundamon_k")
        assert (k1.mode, k1.kind, k1.expert, k1.direction, k1.key) == ("keyed", "expert", 3, None, None)
        assert col.recipe.default_key == "zundamon_k" and col.recipe.default_direction == "zundamon_r"
    assert lo.recipe.units[2].dose == -0.4 and lo.recipe.units[3].k == -2.0
    lo2, _, hi2 = make_columns(r, dose=1.0)                                  # dose は add だけ。α は raw なのでそろえない
    assert hi2.recipe.units[0].k == 1.25 and lo2.recipe.units[0].k == -1.25 and hi2.recipe.units[2].dose == 1.0
    assert r.units[0].k == 1.25


# --- レシピの重ね合わせ -----------------------------------------------------------------------------------

def test_combine_recipes_writes_defaults_into_units():
    a = Recipe([parse_unit("19:ffn:keyed:a1"), parse_unit("20:ffn:keyed:a1@own_r"), parse_unit("33:H16:s2")],
               "m", name="a", default_direction="zundamon_r", default_key="zundamon_k")
    b = Recipe([parse_unit("19:ffn:keyed:a0.5"), parse_unit("21:attn:keyed:a1#own_k")],
               "m", name="b", default_direction="tsun_r", default_key="tsun_k")
    c = combine_recipes([a, b], name="ab")
    c.validate()
    got = [(u.layer, u.direction, u.key) for u in c.units]
    assert got == [(19, "zundamon_r", "zundamon_k"), (20, "own_r", "zundamon_k"), (33, "zundamon_r", None),
                   (19, "tsun_r", "tsun_k"), (21, "tsun_r", "own_k")]
    assert c.default_key == "zundamon_k" and c.default_direction == "zundamon_r"
    assert a.units[0].direction is None and a.units[0].key is None           # 元のユニットは変えない
    # 既定を変えても、重ねた後のユニットの方向・鍵は変わらない
    c2 = c.like(c.units, default_direction="other", default_key="other")
    assert [(u.direction, u.key) for u in c2.keyed_units] == [(d, k) for _, d, k in got if k is not None]


# --- 確かめ ------------------------------------------------------------------------------------------

def _bad(r: Recipe, needle: str):
    with pytest.raises(GneError) as ei:
        r.validate()
    assert ei.value.code == "BAD_RECIPE" and needle in ei.value.message, ei.value.message


def test_validate_keyed_errors():
    _bad(Recipe([_keyed(direction=None)]), "方向が要ります")
    Recipe([_keyed(direction=None)], default_direction="zundamon_r").validate()
    _bad(Recipe([_keyed(key=None)]), "鍵が要ります")
    Recipe([_keyed(key=None)], default_key="zundamon_k").validate()
    _bad(Recipe([_keyed(alpha=None)]), "alpha が要ります")
    _bad(Recipe([_keyed(alpha=150.0)]), "−100〜100")
    Recipe([_keyed(alpha=-100.0)]).validate()
    _bad(Recipe([_keyed(kind="expert")]), "expert")
    Recipe([_keyed(kind="expert", expert=0)]).validate()
    Recipe([_keyed(kind="shared")]).validate()
    Recipe([_keyed(kind="attn")]).validate()
    for kind in ("neuron", "random", "direction", "head", "expert_neuron"):
        _bad(Recipe([Unit(3, 5, 0.0, kind, seed=1, mode="keyed", k=1.0, direction="r", head=1, expert=1, key="k")]),
             "keyed は kind ffn / attn / shared / expert")
    _bad(Recipe([_keyed(direction="bad name")]), "方向の名前")
    _bad(Recipe([_keyed(direction="-zundamon_r")]), "符号に畳み")             # 畳まずに作ったユニット
    _bad(Recipe([_keyed(direction=None)], default_direction="-zundamon_r"), "符号に畳み")
    for key in ("-zundamon_k", "zundamon_k~", "zundamon_k~j", "a/b", "k~1~2"):
        _bad(Recipe([_keyed(key=key)]), "鍵の名前")
    Recipe([_keyed(key="zundamon_k~12")]).validate()
    _bad(Recipe([_keyed(key=None)], default_key="bad key"), "default_key")
    _bad(Recipe([parse_unit("14:8949:+1.0")], default_key="-k"), "default_key")
    # kind ffn・attn は keyed だけ
    _bad(Recipe([Unit(3, None, 0.5, "ffn")]), "keyed だけ")
    _bad(Recipe([Unit(3, None, 0.0, "attn", mode="gain", k=2.0)]), "keyed だけ")
    _bad(Recipe([Unit(3, 1, 0.0, mode="nope")]), "keyed")


def test_validate_existing_behaviour_unchanged():
    """keyed を足しても既存のユニットの確かめは同じ(proj に default_direction の - を付けても前どおり通る)。"""
    Recipe([parse_unit("33:H16:s2")], default_direction="-tsun_r").validate()
    Recipe([parse_unit("12:S:N511:+0.8"), parse_unit("12:E37:x0"), parse_unit("12:E3:route:+2")]).validate()
    with pytest.raises(GneError):
        Recipe([Unit(12, None, 0.0, "shared", mode="gain", k=0.0)]).validate()     # shared の gain はニューロンが要る
    with pytest.raises(GneError):
        Recipe([Unit(12, None, 0.0, "expert", mode="add")]).validate()
    # 鍵の既定は keyed が無くても名前だけ確かめる
    Recipe([parse_unit("14:8949:+1.0")], default_key="zundamon_k").validate()


def test_load_rejects_bad_keyed_yaml(tmp_path):
    p = tmp_path / "b.yaml"
    p.write_text("units:\n  - {layer: 19, kind: ffn, mode: keyed, direction: zundamon_r, alpha: 1}\n", encoding="utf-8")
    with pytest.raises(GneError) as ei:
        Recipe.load(p)
    assert "鍵が要ります" in ei.value.message
    p.write_text("default_key: zundamon_k\nunits:\n  - {layer: 19, kind: neuron, neuron: 3, mode: keyed, direction: r, alpha: 1}\n",
                 encoding="utf-8")
    with pytest.raises(GneError):
        Recipe.load(p)


# --- unit_details ------------------------------------------------------------------------------------

class _Stats:
    def __init__(self, rho):
        self._rho = rho

    def rho(self, L):
        if L not in self._rho:
            raise KeyError(L)
        return self._rho[L]


def test_unit_details_keyed_row():
    from gne.schemas import UnitModel
    r_vec = {19: np.array([3.0, 4.0], dtype=np.float32), 20: np.array([0.0, 2.0], dtype=np.float32),
             21: np.array([6.0, 8.0], dtype=np.float32)}
    seen = []

    def dirs(name, L):
        seen.append((name, L))
        return r_vec[L]

    r = Recipe([parse_unit("19:ffn:keyed:a-1.5@tsun_r#tsun_k"), parse_unit("20:E3:keyed:a2"),
                parse_unit("21:attn:keyed:a1")], default_direction="zundamon_r", default_key="zundamon_k")
    rows = unit_details(None, _Stats({19: 10.0, 20: 4.0}), r, dirs)
    a, b, c = rows
    assert (a["mode"], a["kind"], a["direction"], a["key"], a["alpha"], a["k"]) == ("keyed", "ffn", "tsun_r", "tsun_k", -1.5, -1.5)
    assert a["norm"] == pytest.approx(7.5) and a["rel_push"] == pytest.approx(0.75) and a["diff_rel"] == pytest.approx(0.5)
    assert a["rho"] == 10.0
    assert (b["direction"], b["key"], b["expert"]) == ("zundamon_r", "zundamon_k", 3)
    assert b["rel_push"] == pytest.approx(1.0)
    assert c["rel_push"] is None and c["rho"] is None                        # ρ_L が無い層
    assert c["norm"] == pytest.approx(10.0)                                  # |α|‖r_L‖ は ρ_L が無くても出す
    assert seen == [("tsun_r", 19), ("zundamon_r", 20), ("zundamon_r", 21)]
    for row in rows:
        UnitModel.model_validate(row)
    # 統計・方向の読み込み先が無い(keyed だけのレシピ)・direction_layer が base なら比べる数は None
    for st, dl in ((None, dirs), (_Stats({19: 10.0}), None)):
        x = unit_details(None, st, r, dl)[0]
        assert x["rel_push"] is None and x["direction"] == "tsun_r" and x["key"] == "tsun_k"
    rb = r.like(r.units, direction_layer="base")
    assert unit_details(None, _Stats({19: 10.0}), rb, dirs)[0]["rel_push"] is None


# --- 置き場所 ------------------------------------------------------------------------------------------

def test_store_key_dir(tmp_path):
    from gne.store import Store
    st = Store(tmp_path)
    p = st.key_dir("qwen3-0.6b-abc")
    assert p == tmp_path / "keys" / "qwen3-0.6b-abc" and p.is_dir()
