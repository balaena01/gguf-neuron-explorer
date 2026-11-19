"""出力のあとの正規化(post-norm)の見分けと、補正が入るまでの扱い(v0.8.6)。

モデル不要: 構造の表(configs/arch/residual_writers.yaml の output_norm)と、小さな GGUF で層ごとの見分け
(lfm3・gemma3 は拾い、qwen35 の同じ名前のテンソル = MLP の手前の正規化は拾わない。表に無い arch の post_attention_norm は
「確かめていない」)。ニューロン・ffn の射影は断り、ヘッド・attn の射影は知らせを出し、検算・bake は理由つきで断る。
モデルを使うもの: GNE_TEST_LFM3_MODEL(LFM3 用の llama.cpp で読める環境のときだけ)・GNE_TEST_GEMMA3_MODEL。
"""
import os
from pathlib import Path

import numpy as np
import pytest

from gne.steer import GneError

LFM3 = os.environ.get("GNE_TEST_LFM3_MODEL")
GEMMA3 = os.environ.get("GNE_TEST_GEMMA3_MODEL")
NGL = int(os.environ.get("GNE_TEST_NGL", "-1"))


def _tiny(path, arch, layers, *, n_embd=16, n_head=4, head_dim=4, n_ff=24):
    """層ごとのテンソルの名前(blk.L. の後、.weight を除く)を並べた小さな GGUF。値は乱数(構造の見分けだけを見る)。"""
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


ATTN = ["attn_q", "attn_output", "ffn_down", "ffn_up", "ffn_gate", "attn_norm", "ffn_norm"]


def test_post_norms_table():
    """表にある arch は表の名前で、MLP の手前の正規化の arch は post_attention_norm を拾わず、表に無い arch は post_ffw_norm を
    MLP の正規化、post_attention_norm を「確かめていない」(?)とする。"""
    from gne.arch import post_norms
    both = {"post_attention_norm.weight", "post_ffw_norm.weight", "attn_q.weight"}
    assert post_norms("gemma3", both) == ("post_attention_norm", "post_ffw_norm", "rms")
    assert post_norms("lfm3", both) == ("post_attention_norm", "post_ffw_norm", "rms")
    assert post_norms("lfm3", {"attn_q.weight"}) == ("", "", "rms")                          # 正規化の無い lfm3 の層
    assert post_norms("qwen35", {"post_attention_norm.weight", "attn_q.weight"}) == ("", "", "")
    assert post_norms("sensenova_loop", {"post_attention_norm.weight"}) == ("", "", "")
    assert post_norms("someone_new", {"post_attention_norm.weight"}) == ("?", "", "rms")
    assert post_norms("someone_new", {"post_ffw_norm.weight"}) == ("", "post_ffw_norm", "rms")
    assert post_norms("grok", {"attn_output_norm.weight", "layer_output_norm.weight"}) == ("attn_output_norm", "layer_output_norm", "rms")
    assert post_norms("muse-spark", both)[2] == "scale"


def test_layer_structure_from_tiny_gguf(tmp_path):
    """小さな GGUF の層ごとの見分け: lfm3 の畳み込みの層は mixer conv、正規化は表の名前。qwen35 の post_attention_norm は拾わない。"""
    from gne.weights import Weights
    pn = ["post_attention_norm", "post_ffw_norm"]
    lf = Weights(_tiny(tmp_path / "lfm3.gguf", "lfm3", [["shortconv.out_proj", "shortconv.in_proj", "ffn_down", "attn_norm"] + pn, ATTN + pn]))
    assert [(li.mixer, li.post_norm_mixer, li.post_norm_ffn) for li in lf.arch.layers] == \
        [("conv", "post_attention_norm", "post_ffw_norm"), ("attention", "post_attention_norm", "post_ffw_norm")]
    g = Weights(_tiny(tmp_path / "g3.gguf", "gemma3", [ATTN + pn]))
    assert g.arch.layers[0].post_norm_mixer == "post_attention_norm" and g.arch.layers[0].post_norm_ffn == "post_ffw_norm"
    q = Weights(_tiny(tmp_path / "q35.gguf", "qwen35", [ATTN + ["post_attention_norm"]]))
    assert (q.arch.layers[0].post_norm_mixer, q.arch.layers[0].post_norm_ffn) == ("", "")
    u = Weights(_tiny(tmp_path / "new.gguf", "qwen3", [ATTN + ["post_attention_norm"]]))     # 表に無い(qwen3 に本来は無い名前)
    assert u.arch.layers[0].post_norm_mixer == "?"


def test_guards_on_post_norm_layers(tmp_path):
    """ニューロン・ffn の射影は断る。ヘッド・attn の射影は知らせの文を返す(qwen35 では返さない)。検算は理由つきで断る。
    bake(add)は正規化の理由で断る。"""
    from gne.headproj import ProjHead, _refuse_postnorm_check, neuron_layer_ok, postnorm_mixer_warning
    from gne.steer import bake_bias_gguf
    from gne.weights import Weights
    pn = ["post_attention_norm", "post_ffw_norm"]
    g = Weights(_tiny(tmp_path / "g3.gguf", "gemma3", [ATTN + pn, ATTN + pn]))
    with pytest.raises(GneError, match="MLP の出力のあとに正規化"):
        neuron_layer_ok(g, 1)
    heads = [ProjHead(1, 0, 2.0, "d"), ProjHead(1, -1, 1.0, "d", "attn"), ProjHead(0, 5, 1.0, "d", "neuron")]
    msg = postnorm_mixer_warning(g, heads)
    assert msg and "層 1 " in msg and "5 割" in msg and "層 0" not in msg
    with pytest.raises(GneError, match="正規化") as e:
        _refuse_postnorm_check(g, 1, "mixer")
    assert "次の版" in (e.value.hint or "")
    with pytest.raises(GneError, match="正規化があるので.*bake は実行しません"):
        bake_bias_gguf(g, {1: np.zeros(16, np.float32)}, tmp_path / "g3.gguf", tmp_path / "out.gguf")
    q = Weights(_tiny(tmp_path / "q35.gguf", "qwen35", [ATTN + ["post_attention_norm"]] * 2))
    assert postnorm_mixer_warning(q, heads) is None
    neuron_layer_ok(q, 1)


@pytest.mark.model
@pytest.mark.parametrize("which", ["lfm3", "gemma3"])
def test_post_norm_models(which, tmp_path):
    """実物(LFM3・Gemma 3): ヘッドの射影の steer で知らせが出て、一括(rank causal)の結果に postnorm が入り、検算は理由つきで断る。
    verify の V2 は MLP の出力(LFM3 は正規化の後の出力)と比べて通る。"""
    path = LFM3 if which == "lfm3" else GEMMA3
    if not path or not Path(path).exists():
        pytest.skip(f"GNE_TEST_{which.upper()}_MODEL がありません")
    import yaml

    from gne.api import Session
    try:
        s = Session(Path(path), root=tmp_path / "root", n_gpu_layers=NGL, n_ctx=2048)
    except Exception as ex:  # noqa: BLE001  この llama.cpp が arch を読めない環境
        pytest.skip(f"このモデルを読めない環境です: {ex}")
    try:
        att = [li.index for li in s.weights.arch.layers if li.mixer == "attention" and li.post_norm_mixer]
        assert att, "注意の層の正規化を見分けていない"
        L = att[len(att) // 2]
        evil = Path(__file__).resolve().parent.parent / "configs" / "contrast" / "evil.yaml"
        s.direction(contrast=evil, name="pn", positions="last")
        s.steer([f"{L}:H0:s1@pn"], prompt="Hello.", max_tokens=4)
        assert any("正規化" in w and "5 割" in w for w in s.warnings)
        from gne.headproj import check_head_proj
        with pytest.raises(GneError, match="正規化"):
            check_head_proj(s, L, 0, "pn")
        d = yaml.safe_load(evil.read_text(encoding="utf-8"))
        out = s.heads_pipeline({"name": "pnc", "system_a": d["system_a"], "system_b": d["system_b"], "user_prompts": d["user_prompts"][:8],
                                "pipeline": {"rank": "causal", "n_heads": [2], "top_layers": 1, "max_examples": 6, "hard": 4}})
        assert out.get("postnorm") and out["postnorm"]["layers"]
        base = {"system_a": d["system_a"], "system_b": d["system_b"], "user_prompts": d["user_prompts"][:8]}
        dflt = s.heads_pipeline({**base, "name": "pnw", "pipeline": {"n_heads": [2], "top_layers": 1, "max_examples": 6, "hard": 4}})
        assert dflt["rank"] == "causal" and "正規化" in dflt.get("rank_note", "")       # 既定は止めて選ぶ流れに切り替え、知らせる
        with pytest.raises(GneError, match="正規化") as e:                               # 書き込みの量での順位づけを指定したら断る
            s.heads_pipeline({**base, "name": "pnx", "pipeline": {"rank": "writes", "max_examples": 6, "verify_top": 4,
                                                                   "verify_sample": 0, "n_units": [2], "check_replies": 0}})
        assert e.value.code == "ARCH_UNSUPPORTED"
        from gne.verify import Verifier
        v2 = Verifier(s.engine, s.weights, layer=s.engine.n_layer // 2).v2_ffn_linearity()
        assert v2.status == "pass", v2.detail
        s.stats_resid(tokens=1024)                                 # 残差の偏り: Gemma 3 は約 6 倍で知らせる、LFM3 は約 1.3 倍で知らせない
        ds = s.dose_spread()
        assert ds is not None and ds["warn"] == (which == "gemma3"), ds
        s.steer([f"{L}:100:+0.1"], prompt="Hello.", max_tokens=4)
        assert any("偏って" in w for w in s.warnings) == (which == "gemma3")
        assert any("偏って" in w for w in s.info(load=False)["arch_warnings"]) == (which == "gemma3")
    finally:
        s.close()


def test_resid_spread_stats(tmp_path):
    """残差の偏りの量: 保存・読み込み(古い統計 = 偏りの量が無い形も読める)・層を足したときの合わせ方・まとめ。"""
    from gne.api import _merge_resid
    from gne.stats import SPREAD_WARN, ResidStats
    L = np.array([2, 3, 4, 5], dtype=np.int32)
    p50 = np.array([10.0, 60.0, 60.0, 60.0], dtype=np.float32)
    st = ResidStats(L, p50 * 0.5, p50, p50 * 2, p50, 100, "sha", 512, True, np.array([9.0, 10.0, 12.0, 6.0], dtype=np.float32))
    assert st.spread(3) == pytest.approx(6.0) and st.spread(9) is None
    sm = st.spread_summary()                                      # 層 3 以降だけ(層 2 の 1.1 倍は入れない)
    assert sm["median"] == pytest.approx(6.0) and sm["max"] == pytest.approx(10.0) and sm["max_layer"] == 5
    assert sm["warn"] and sm["warn_at"] == SPREAD_WARN
    st.save(tmp_path / "new.npz")
    assert ResidStats.load(tmp_path / "new.npz").spread(4) == pytest.approx(5.0)
    old = ResidStats(L, p50 * 0.5, p50, p50 * 2, p50, 100, "sha", 512, True)          # v0.8.6 より前の形
    np.savez(tmp_path / "old.npz", layers=old.layers, p10=old.p10, p50=old.p50, p90=old.p90, mean=old.mean,
             n_tokens=old.n_tokens, corpus_sha=old.corpus_sha, chunk_tokens=old.chunk_tokens, exclude_pos0=old.exclude_pos0)
    lo = ResidStats.load(tmp_path / "old.npz")
    assert lo.p50_rest is None and lo.spread(3) is None and lo.spread_summary() is None and "p50_rest" not in lo.to_dict()
    add = ResidStats(np.array([6], dtype=np.int32), *(np.array([60.0], dtype=np.float32),) * 4, 100, "sha", 512, True,
                     np.array([30.0], dtype=np.float32))
    m = _merge_resid(lo, add)                                     # 古い統計に層を足す: 足した層だけ偏りの量がある
    assert list(m.layers) == [2, 3, 4, 5, 6] and m.spread(3) is None and m.spread(6) == pytest.approx(2.0)
    assert m.to_dict()["p50_rest"][:4] == [None] * 4
    assert _merge_resid(lo, ResidStats(np.array([6], dtype=np.int32), *(np.array([1.0], dtype=np.float32),) * 4, 1, "sha", 512)).p50_rest is None


def test_check_arch_messages(tmp_path):
    """gne info の警告: 正規化の場所(MLP・注意)ごとに、何を実行し何を実行しないかを書く。qwen35 には出さない。"""
    from gne.weights import Weights
    pn = ["post_attention_norm", "post_ffw_norm"]
    g = Weights(_tiny(tmp_path / "g3.gguf", "gemma3", [ATTN + pn]))
    ws = g.check_arch()
    assert any("MLP の出力のあとに正規化" in w and "実行しません" in w for w in ws)
    assert any("注意(混ぜる層)の出力のあとに正規化" in w and "--rank causal" in w for w in ws)
    q = Weights(_tiny(tmp_path / "q35.gguf", "qwen35", [ATTN + ["post_attention_norm"]]))
    assert not any("正規化" in w for w in q.check_arch())
    u = Weights(_tiny(tmp_path / "new.gguf", "qwen3", [ATTN + ["post_attention_norm"]]))
    assert any("確かめていません" in w for w in u.check_arch())
