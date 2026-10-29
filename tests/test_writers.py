"""書き手の強さ(gne writers)のテスト。モデルは Qwen3-0.6B(GNE_TEST_MODEL)。無ければ飛ばす。"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))


def test_load_replies(tmp_path):
    from gne.steer import GneError
    from gne.writers import load_replies
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(["a", "b"]))
    assert load_replies(p, ["x", "y"]) == ["a", "b"] and load_replies(None, ["x"]) is None
    p.write_text(yaml.safe_dump({"replies": {"y": "B", "x": "A"}}))
    assert load_replies(p, ["x", "y"]) == ["A", "B"]
    with pytest.raises(GneError):
        load_replies(p, ["x", "z"])
    p.write_text(yaml.safe_dump(["only one"]))
    with pytest.raises(GneError):
        load_replies(p, ["x", "y"])


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip("モデルがありません")
    from gne.api import Session
    s = Session(MODEL, root=tmp_path_factory.mktemp("writers"))
    yield s
    s.close()


@pytest.mark.model
def test_writers_decomposition(session, tmp_path):
    """MLP の出力を式で求めたものが捕捉した出力と合う。方向が埋め込み・mixer・MLP に分かれ、和が 1。
    寄与の順位・保存・調書・読ませた返事(teacher forcing)。"""
    from gne.schemas import COMMAND_SCHEMAS
    from gne.writers import list_writers, load_layer, neuron_entries
    c = tmp_path / "pirate.yaml"
    c.write_text(yaml.safe_dump({
        "name": "pirate", "system_a": "You are a pirate. Always talk like a pirate, with arr and matey.",
        "system_b": "You are a helpful assistant.", "user_prompts": ["Who are you?", "Tell me about the weather.", "What is your favorite food?"],
        "validate": {"match": "arr|matey", "target": "a pirate"}}))
    d = session.writers(c, max_tokens=16, top=30)
    assert d["n_convs"] == {"A": 3, "B": 3} and d["n_tokens"]["A"] > 0 and d["ref_layer"] == session.meta.n_layer - 2
    assert d["mlp_check_max"] is not None and d["mlp_check_max"] < 0.03                     # 式と捕捉した ffn_out の差
    t = d["totals"]
    assert abs(t["embed"] + t["mixer"] + t["mlp"] - 1.0) < 1e-6                              # 和が ‖r*‖
    assert [x["contrib"] for x in d["top"]] == sorted((x["contrib"] for x in d["top"]), reverse=True)
    assert d["bottom"][0]["contrib"] <= d["top"][-1]["contrib"]
    top = d["top"][0]
    lay = load_layer(session, "pirate", top["layer"])
    assert abs(float(lay["contrib"][top["neuron"]]) - top["contrib"]) < 1e-6 and lay["act_a"].shape == (session.meta.n_ff_at(top["layer"]),)
    e = neuron_entries(session, top["layer"], top["neuron"])
    assert e[0]["name"] == "pirate" and e[0]["rank"] == 1
    assert list_writers(session)[0]["name"] == "pirate"
    COMMAND_SCHEMAS["writers"].model_validate({**session.envelope("writers"), **d})
    # 層ごとの和: 参照の層までの mixer と MLP の射影の和 + 埋め込み = ‖r*‖
    nr = d["ref_norm"]
    s = t["embed"] * nr + sum(x["mixer_to_ref"] + (x["mlp_to_ref"] or 0) for x in d["layers"] if "mixer_to_ref" in x)
    assert abs(s - nr) / nr < 1e-6
    # 読ませた返事: A の返事を生成せず、与えた文を使う
    rp = tmp_path / "replies.yaml"
    rp.write_text(yaml.safe_dump(["Arr, I be a pirate, matey!", "The seas be calm, arr.", "Salted fish, matey!"]))
    g = session.writers(c, name="pirate_given", replies_a=rp, max_tokens=16)
    n_tok = sum(len(session.engine.tokenize(x, add_special=False, parse_special=False)) for x in yaml.safe_load(rp.read_text()))
    assert g["replies"]["A"] == "given" and g["n_tokens"]["A"] == n_tok
    nd = session.neuron(top["layer"], top["neuron"])
    assert {x["name"] for x in nd["writers"]} >= {"pirate", "pirate_given"} and nd["coverage"]["writers"]
    assert json.loads(Path(d["path"]).read_text())["name"] == "pirate"
    assert np.isfinite([x["cos_ref"] for x in d["layers"]]).all()
    # 書き出し: 上位の書き手の分だけ・MLP 全部・mixer・残差の差。MLP 全部の和は捕捉した ΔMLP と同じ
    from gne.writers import export_writes
    z = np.load(Path(d["path"]).with_suffix(".npz"))
    full = export_writes(session, "pirate", "pw_mlp", part="mlp")
    ds = session.direction_set("pw_mlp")
    L = full["layers"][len(full["layers"]) // 2]
    assert np.allclose(ds.raw[L], z[f"mlp_vec_{L}"], atol=1e-5)
    t3 = export_writes(session, "pirate", "pw_top3", part="mlp", top=3)
    assert 1 <= len(t3["layers"]) <= 3
    tl = top["layer"]
    W = session.weights.down_matrix(tl)
    v = session.direction_set("pw_top3").raw[tl]
    assert np.linalg.norm(v) > 0 and np.allclose(v, W[:, [top["neuron"]] + [x["neuron"] for x in d["top"][1:3] if x["layer"] == tl]]
                                                  @ z[f"wdiff_{tl}"][[top["neuron"]] + [x["neuron"] for x in d["top"][1:3] if x["layer"] == tl]], atol=1e-4)
    r = export_writes(session, "pirate", "pw_r", part="r", layers=[d["ref_layer"]])
    assert r["layers"] == [d["ref_layer"]] and abs(r["norms"][str(d["ref_layer"])] - d["ref_norm"]) < 1e-3
    rec = session.recipe(units=[f"{L}:dir:pw_mlp:1"], dose_unit="raw")
    assert np.allclose(session.vectors(rec)[0][L], z[f"mlp_vec_{L}"], atol=1e-5)
