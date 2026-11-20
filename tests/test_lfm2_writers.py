"""LFM2(lfm2・lfm2moe)の heads writers と、長さが 0 の方向の扱い(v0.9.0)。

LFM2 は出力のあとの正規化が無く、MLP の出力に名前が無い("model.layers.{}.ffn_out" は MLP の入力 = 正規化の後に付いた名前)。
混ぜる層の書き込み行列の出力(conv.out_proj / self_attn.out_proj)を前の層の出力に足して、混ぜる層を足した後の残差にする。
短い畳み込みの層は数トークン前までしか見ないので、最初の注意の層より前(LFM2・LFM3 の層 0・1)は、生成プロンプトの最後の位置で
A − B がちょうど 0 になる。その層の方向を使うコマンドは理由つきで止め、書き込みの順位づけはその層を外して知らせる。

モデル不要: 捕捉の表の名前、検算する層の選び方、長さが 0 の方向の止め方と知らせ。
モデルを使うもの: LFM2-8B-A1B(GNE_TEST_LFM2_MODEL、既定は models/LFM2-8B-A1B-Q4_K_M.gguf)。
"""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gne.steer import GneError

ROOT = Path(__file__).resolve().parent.parent
LFM2 = Path(os.environ.get("GNE_TEST_LFM2_MODEL", ROOT / "models" / "LFM2-8B-A1B-Q4_K_M.gguf"))
NGL = int(os.environ.get("GNE_TEST_NGL", "-1"))
EVIL = ROOT / "configs" / "contrast" / "evil.yaml"


def _tiny(path, arch, layers, *, n_embd=16, n_head=4, head_dim=4, n_ff=24):
    from gguf import GGUFWriter
    rng = np.random.default_rng(0)
    w = GGUFWriter(str(path), arch=arch)
    w.add_name("tiny")
    w.add_block_count(len(layers))
    w.add_embedding_length(n_embd)
    w.add_feed_forward_length(n_ff)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_head)
    w.add_token_list([f"t{i}" for i in range(8)])
    shapes = {"attn_q": (n_head * head_dim, n_embd), "attn_output": (n_embd, n_head * head_dim), "ffn_down": (n_embd, n_ff),
              "ffn_up": (n_ff, n_embd), "ffn_gate": (n_ff, n_embd), "shortconv.out_proj": (n_embd, n_embd),
              "shortconv.in_proj": (3 * n_embd, n_embd)}
    for L, names in enumerate(layers):
        for nm in names:
            w.add_tensor(f"blk.{L}.{nm}.weight", rng.standard_normal(shapes.get(nm, (n_embd,))).astype(np.float32))
    w.add_tensor("token_embd.weight", rng.standard_normal((8, n_embd)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


CONV = ["shortconv.out_proj", "shortconv.in_proj", "ffn_down", "ffn_up", "ffn_gate", "attn_norm", "ffn_norm"]
ATTN = ["attn_q", "attn_output", "ffn_down", "ffn_up", "ffn_gate", "attn_norm", "ffn_norm"]


def _ds(raw, name="d"):
    return SimpleNamespace(name=name, raw=raw, meta={})


def test_lfm2_capture_names_and_layers(tmp_path):
    """捕捉の表: lfm2・lfm2moe は混ぜる層の種類ごとの書き込み行列の出力と埋め込みの名前を持ち、正規化の後の名前は持たない。
    出力のあとに正規化の無い arch でも、表に mixer_out があれば検算の層を選ぶ(種類ごとの最初の層と dense の層)。"""
    from gne.postnorm import _pick_layers, checked, entry, layer_names, writer_layers
    from gne.weights import Weights
    for arch in ("lfm2", "lfm2moe"):
        ent = entry(arch)
        assert checked(arch) == "real" and ent["embed"] == "model.embed_tokens"
        assert not ent.get("mid") and not ent.get("ffn_post") and not ent.get("mixer_post")
    w = Weights(_tiny(tmp_path / "lfm2.gguf", "lfm2", [CONV, CONV, ATTN, CONV, ATTN, CONV]))
    assert not writer_layers(w)
    n0, n2 = layer_names(w, 0), layer_names(w, 2)
    assert n0["mixer_out"] == "model.layers.{}.conv.out_proj-0" and n2["mixer_out"] == "model.layers.{}.self_attn.out_proj-2"
    assert n0["embed"] == "model.embed_tokens" and n0["mid"] is None
    Ls = _pick_layers(w, prenorm=True)
    kinds = {w.arch.layers[L].mixer for L in Ls}
    assert kinds == {"conv", "attention"} and all(1 <= L <= 4 for L in Ls)
    assert any(w.arch.layers[L].ffn == "dense" for L in Ls)
    assert _pick_layers(w) == []                                   # 正規化のある層だけを選ぶ今までの選び方では空


def test_zero_length_direction_stops_with_reason():
    """長さが 0 の層の方向: 理由(A 側と B 側の残差が同じ、短い畳み込みの層)と使える最初の層を添えて止める。"""
    from gne.ablate import per_layer_directions
    from gne.direction import require_length, zero_length_layers
    from gne.headproj import head_direction
    v = np.arange(1, 17, dtype=np.float32)
    ds = _ds({0: np.zeros(16, np.float32), 1: np.zeros(16, np.float32), 2: v, 3: -v})
    assert zero_length_layers(ds) == [0, 1]
    with pytest.raises(GneError, match="長さが 0 です。この層では A 側と B 側の残差がまったく同じ") as e:
        head_direction(ds, 1)
    assert e.value.zero_layer == 1 and "層 2 以降" in e.value.hint and "畳み込み" in e.value.message
    with pytest.raises(GneError, match="方向の部品 0:dir:d は実行しません") as e:
        require_length(ds, 0, "方向の部品 0:dir:d")
    assert "層 2 以降" in e.value.hint
    r, src = head_direction(ds, 2)
    assert src == 2 and abs(float(np.linalg.norm(r)) - 1) < 1e-6
    with pytest.raises(GneError, match="層ごとの方向の消去") as e:
        per_layer_directions(ds.raw, [0, 1, 2, 3], "d")
    assert "--layers" in e.value.hint
    assert set(per_layer_directions(ds.raw, [2, 3], "d")) == {2, 3}
    with pytest.raises(GneError) as e:                                 # 全部の層が 0: 対照集合の作り直しを案内する
        require_length(_ds({0: np.zeros(4), 1: np.zeros(4)}), 0)
    assert "対照集合" in e.value.hint


def test_writes_ranking_skips_zero_layers():
    """書き込みの順位づけ: 長さが 0 の層を見つけて外す(ほかの理由の BAD_DIRECTION はそのまま止める)。"""
    from gne.headproj import head_direction
    from gne.unitwrites import _zero_direction_layers
    v = np.ones(8, np.float32)
    ds = _ds({0: np.zeros(8, np.float32), 1: np.zeros(8, np.float32), 2: v, 3: v})
    builder = SimpleNamespace(r_of=lambda L, name: head_direction(ds, L))
    assert _zero_direction_layers(builder, ds, [0, 1, 2, 3]) == [0, 1]

    def other(L, name):
        raise GneError("BAD_DIRECTION", "方向 d の基準層が決まっていません")
    with pytest.raises(GneError, match="基準層"):
        _zero_direction_layers(SimpleNamespace(r_of=other), ds, [2])


# --- モデルを使うもの(LFM2-8B-A1B)-------------------------------------------------------------------

@pytest.mark.model
def test_lfm2_model(tmp_path):
    """実物(LFM2-8B-A1B): 書き込みの和の検算(V27)が通り、heads writers が実行でき(部品の和 = 基準層の差)、既定の一括
    (rank writes)が層 0・1 を外す知らせを出して最後まで進む。層 0 の方向の部品は理由つきで止まる。"""
    if not LFM2.exists():
        pytest.skip(f"LFM2 のモデルがありません({LFM2})")
    import yaml

    from gne.api import Session
    from gne.postnorm import check_writer_sums
    s = Session(LFM2, root=tmp_path / "root", n_gpu_layers=NGL, n_ctx=2048)
    try:
        res = check_writer_sums(s)
        assert res["ok"] and res["prenorm"] and res["checked"] == "real", res
        kinds = {r["mixer"] for r in res["layers"]}
        assert kinds == {"conv", "attention"}
        dense = [r for r in res["layers"] if s.weights.arch.layers[r["layer"]].ffn == "dense"]
        moe = [r for r in res["layers"] if s.weights.arch.layers[r["layer"]].ffn.startswith("moe")]
        assert dense and all(r["ffn_err"][0] < 0.05 for r in dense)        # W_down a と、式で出した MLP の分
        assert moe and all(r["ffn_err"][0] < 0.05 for r in moe)            # 専門家の出力の和と、式で出した MLP の分
        assert any(r["head_err"] is not None and r["head_err"][0] < 0.05 for r in res["layers"])    # ヘッドの出力 → out_proj
        assert all(r["sum_err"] is None for r in res["layers"])            # 和は式そのものなので測らない
        d = yaml.safe_load(EVIL.read_text(encoding="utf-8"))
        s.direction(contrast=EVIL, name="lf", positions="last")
        ds = s.direction_set("lf")
        assert not np.any(ds.raw[0]) and not np.any(ds.raw[1]) and np.any(ds.raw[2])
        hw = s.heads_writers("lf", contrast=EVIL, ref_layer=6, max_examples=6)
        assert hw["mixer_out_layers"] == list(range(7)) and hw["sum_error"] < 1e-4, hw["sum_error"]
        with pytest.raises(GneError, match="長さが 0"):
            s.steer(["0:dir:lf:1"], prompt="Hello.", max_tokens=2)
        s.warnings = []
        out = s.heads_pipeline({"name": "lfw", "system_a": d["system_a"], "system_b": d["system_b"], "user_prompts": d["user_prompts"][:6],
                                "pipeline": {"verify_top": 3, "verify_sample": 0, "n_units": [2], "max_examples": 4, "check_replies": 0}})
        assert out.get("rank") == "writes"
        assert any("層 0, 1 は長さが 0" in x for x in s.warnings), s.warnings
    finally:
        s.close()
