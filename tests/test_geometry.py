"""層ごとの更新ベクトルの幾何(gne heads geometry、gne/geometry.py)。

モデルが要らないもの(式・基準・行の作り方・構造の判定・保存・CLI・出力の型)は、いつも流す。小さな合成の GGUF(llama arch)の
保存先に、合成の heads writes の材料と方向ベクトルを置いて、最後まで動くことを確かめる。
MCP のツールは Qwen3-0.6B(GNE_TEST_MODEL)で gne mcp を起動して確かめる(モデルは流さない)。
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gne import geometry as G
from gne.steer import GneError

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))


def _rand(n, d, seed=0, scales=None):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    if scales is not None:
        X *= np.asarray(scales)[:, None]
    return X


# --- 式 -------------------------------------------------------------------------------------------------

def test_cosine_matrix_is_normalized_gram():
    X = _rand(7, 30, scales=[0.1, 1, 3, 10, 0.5, 2, 50])
    C = G.cosine_matrix(X)
    Xt = X / np.linalg.norm(X, axis=1, keepdims=True)
    assert np.allclose(C, Xt @ Xt.T)
    for i in range(7):
        for j in range(7):
            assert C[i, j] == pytest.approx(X[i] @ X[j] / (np.linalg.norm(X[i]) * np.linalg.norm(X[j])), abs=1e-12)
    assert np.allclose(np.diag(C), 1.0)
    # 0 の行は 0 のまま(割り算で NaN にしない)
    Z = np.vstack([X[:2], np.zeros((1, 30))])
    assert np.all(np.isfinite(G.cosine_matrix(Z))) and np.all(G.cosine_matrix(Z)[2] == 0)


def test_svd_matches_eigenvectors_and_sign_rule():
    X = _rand(6, 25, seed=1, scales=[1, 2, 3, 4, 5, 6])
    m = G.svd_modes(X, 6)
    U, s, Vt = m["U"], m["s"], m["Vt"]
    # U の列は X Xᵀ の固有ベクトル(固有値 σ²)、V の行は Xᵀ X の固有ベクトル
    assert np.allclose(X @ X.T @ U, U * s ** 2)
    assert np.allclose(Vt @ X.T @ X, (s ** 2)[:, None] * Vt)
    w, Q = np.linalg.eigh(X @ X.T)
    order = np.argsort(-w)
    for j in range(6):
        assert abs(Q[:, order[j]] @ U[:, j]) == pytest.approx(1.0, abs=1e-8)        # 符号を除いて一致
    assert np.allclose(sorted(w, reverse=True), s ** 2)
    assert m["explained"].sum() == pytest.approx(1.0)
    assert np.all(U.sum(axis=0) >= 0)                                              # 符号は U の列の和が正
    assert np.allclose(U * s @ Vt, X)                                              # 全部の向きで元に戻る


def test_row_explained_is_projection_share():
    X = _rand(8, 40, seed=2, scales=np.geomspace(0.1, 20, 8))
    m = G.svd_modes(X, 5)
    rx = G.row_explained(m["U"], m["s"], X)
    for k in range(1, 6):
        P = m["Vt"][:k].T @ m["Vt"][:k]
        want = ((X @ P) ** 2).sum(axis=1) / (X ** 2).sum(axis=1)
        assert np.allclose(rx[:, k - 1], want)
    assert np.all(np.diff(rx, axis=1) >= -1e-12)                                    # 累積なので増える
    # explained_by: 同じ向きで行列全体を説明する割合は、寄与率の累積と一致する
    assert np.allclose(G.explained_by(m["Vt"], X), np.cumsum(m["explained"])[:5])


def test_subspace_overlap():
    V = np.linalg.qr(_rand(20, 4, seed=3))[0].T                                     # 4 本の正規直交の向き
    assert np.allclose(G.subspace_overlap(V, V), 1.0)
    assert np.allclose(G.subspace_overlap(V, V[[1, 0, 2, 3]] * -1)[1:], 1.0)        # 並べ替え・符号は 2 本目からは同じ
    W = np.linalg.qr(np.hstack([V.T, _rand(20, 4, seed=4)]))[0].T[4:8]              # V と直交する 4 本
    assert np.allclose(G.subspace_overlap(V, W), 0.0, atol=1e-12)


def _writes_like(QA, QB):
    """measure_writes と同じ作り方: 側ごとの平均の差と、偶数番目・奇数番目の質問の半分ずつの差。QA・QB は (質問, 層, d)。"""
    hA, hB = np.arange(len(QA)) % 2 == 0, np.arange(len(QB)) % 2 == 0
    d = QA.mean(0) - QB.mean(0)
    d0 = QA[hA].mean(0) - QB[hB].mean(0)
    d1 = QA[~hA].mean(0) - QB[~hB].mean(0)
    return d, d0, d1


def test_halves_mean_and_noise_estimate():
    rng = np.random.default_rng(5)
    QA = rng.standard_normal((16, 4, 12)) + 0.8
    QB = rng.standard_normal((16, 4, 12))
    d, d0, d1 = _writes_like(QA, QB)
    assert np.allclose(d, (d0 + d1) / 2)                                           # 半々なら全体の差 = 半分の平均
    hc, nr = G.half_stats(d, d0, d1)
    assert hc[0] == pytest.approx(d0[0] @ d1[0] / np.linalg.norm(d0[0]) / np.linalg.norm(d1[0]))
    assert nr[2] == pytest.approx(np.linalg.norm((d0[2] - d1[2]) / 2) / np.linalg.norm(d[2]))
    # 質問の数が奇数なら一致しない
    dd, e0, e1 = _writes_like(QA[:15], QB)
    assert not np.allclose(dd, (e0 + e1) / 2)
    # 狙った違いが無いとき、(d0 − d1)/2 の大きさの 2 乗は d の大きさの 2 乗と平均で同じ(雑音の目安として偏らない)
    r = []
    for t in range(400):
        g = np.random.default_rng(100 + t)
        a, b = g.standard_normal((16, 1, 30)), g.standard_normal((16, 1, 30))
        x, x0, x1 = _writes_like(a, b)
        r.append((np.linalg.norm((x0 - x1) / 2) ** 2, np.linalg.norm(x) ** 2))
    r = np.array(r)
    assert r[:, 0].mean() / r[:, 1].mean() == pytest.approx(1.0, abs=0.06)


def test_svd_analysis_baselines_on_orthogonal_rows():
    norms = np.array([5.0, 1.0, 3.0, 0.5])
    X = np.zeros((4, 10))
    X[np.arange(4), np.arange(4)] = norms                                          # 行が互いに直交
    s_rows, _ = G.svd_analysis(X, X, X, "rows", 4)
    assert np.allclose(s_rows["explained"], 0.25) and np.allclose(s_rows["baseline_orthogonal"], 0.25)
    s_none, _ = G.svd_analysis(X, X, X, "none", 4)
    want = np.sort(norms ** 2)[::-1] / (norms ** 2).sum()
    assert np.allclose(s_none["explained"], want) and np.allclose(s_none["baseline_orthogonal"], want)
    # 半分どうしが同じなら、雑音の基準は 0、半分で求めた向きの説明率は寄与率の累積、重なりは 1
    assert np.allclose(s_none["baseline_noise"], 0) and s_none["noise_total"] == pytest.approx(0)
    assert np.allclose(s_none["half_cross"], s_none["cumulative"]) and np.allclose(s_none["half_overlap"], 1)


def test_svd_analysis_noise_baseline_scaling():
    X = _rand(5, 20, seed=6, scales=[1, 2, 4, 8, 16])
    N = _rand(5, 20, seed=7) * 0.1
    X0, X1 = X + N, X - N                                                          # (X0 − X1)/2 = N
    s_none, _ = G.svd_analysis(X, X0, X1, "none", 3)
    sn = np.linalg.svd(N, compute_uv=False)
    assert np.allclose(s_none["baseline_noise"], sn[:3] ** 2 / (X ** 2).sum())
    assert s_none["noise_total"] == pytest.approx((N ** 2).sum() / (X ** 2).sum())
    s_rows, _ = G.svd_analysis(X, X0, X1, "rows", 3)                                # 行ごとに ‖x_i‖ で割った雑音
    Nr = N / np.linalg.norm(X, axis=1, keepdims=True)
    assert s_rows["noise_total"] == pytest.approx((Nr ** 2).sum() / 5)


# --- 材料の行・ループ・累積・構造 ---------------------------------------------------------------------------

def _material(n_layer=5, d=12, seed=8, visit=None, stored=None, mixer=None, n_a=16, n_b=16):
    rng = np.random.default_rng(seed)
    z = {"layer_index": np.arange(n_layer, dtype=np.int32),
         "layer_mixer": np.array(mixer or (["attention", "linear_attention"] * n_layer)[:n_layer]),
         "layer_visit": np.array(visit or [0] * n_layer, np.int32),
         "layer_stored": np.array(stored or list(range(n_layer)), np.int32),
         "check_layer_rel": np.full(n_layer, 0.003, np.float32), "select_layer": np.int32(n_layer - 1),
         "n_a": np.int32(n_a), "n_b": np.int32(n_b), "gap": np.float32(1.0),
         "cos_to_select": np.full(n_layer, np.nan, np.float32)}
    for k in ("attn", "linear", "mlp"):
        for x in ("a", "b", "diff", "diff_h0", "diff_h1"):
            z[f"upd_{k}_{x}"] = np.zeros((n_layer, d), np.float32)
    for L in range(n_layer):
        k = "attn" if z["layer_mixer"][L] == "attention" else "linear"
        for part in (k, "mlp"):
            d0, d1 = rng.standard_normal(d) * (L + 1), rng.standard_normal(d) * (L + 1)
            z[f"upd_{part}_diff_h0"][L], z[f"upd_{part}_diff_h1"][L] = d0, d1
            z[f"upd_{part}_diff"][L] = (d0 + d1) / 2
    for x in ("a", "b", "diff", "diff_h0", "diff_h1"):
        z[f"upd_layer_{x}"] = z[f"upd_attn_{x}"] + z[f"upd_linear_{x}"] + z[f"upd_mlp_{x}"]
    return z


def test_build_rows_order_skips_and_exclusion():
    z = _material()
    z["upd_mlp_diff"][2] = 0                                                       # MLP の無い層 → MLP の行を除く
    rows, X, X0, X1, skipped = G.build_rows(z, ("attn", "mlp"), None, moe_layers=[4], exclude_dims=[3, 5])
    labels = [r["label"] for r in rows]
    assert labels == [f"L{L}/attn" for L in range(5)] + ["L0/mlp", "L1/mlp", "L3/mlp"]   # 部品ごとにまとめて層の順。MoE の層 4 の MLP も除く
    assert skipped == {"attn": 0, "mlp": 2}
    assert np.all(X[:, [3, 5]] == 0) and np.all(X0[:, [3, 5]] == 0)
    v = z["upd_attn_diff"][1] + z["upd_linear_diff"][1]
    assert np.allclose(X[1][[0, 1, 2, 4]], v[[0, 1, 2, 4]])
    assert rows[1]["top_dims"]["dims"][0] == int(np.argmax(v ** 2))                 # 上位の次元は除く前の値で
    assert rows[1]["mixer"] == "linear_attention" and rows[5]["mixer"] == "mlp"
    r2, *_ = G.build_rows(z, ("layer",), [1, 3], moe_layers=[3])
    assert [r["label"] for r in r2] == ["L1"]                                       # MoE の層は層全体の行も作らない
    with pytest.raises(GneError):
        G.build_rows(z, ("layer",), [40])


def test_loop_pairs_follow_stored_layers():
    z = _material(n_layer=6, visit=[0, 1, 1, 2, 2, 0], stored=[0, 1, 2, 1, 2, 5])
    lp = G.loop_pairs(z, None)
    assert lp["shift"] == 2 and [(p["first"], p["second"]) for p in lp["pairs"]] == [(1, 3), (2, 4)]
    a, b = z["upd_layer_diff"][1].astype(float), z["upd_layer_diff"][3].astype(float)
    assert lp["pairs"][0]["layer"]["cos"] == pytest.approx(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    assert G.loop_pairs(z, [1, 2, 4]) ["pairs"][0]["first"] == 2                    # 両方の層が選んだ層にある組だけ
    assert G.loop_pairs(_material(), None) is None


def test_cumsum_check_matches_residual_difference():
    z = _material()
    raw = {L: np.cumsum(z["upd_layer_diff"].astype(float), axis=0)[L] for L in range(5)}
    out = G.cumsum_check(z, raw)
    assert all(c["cos"] == pytest.approx(1.0) and c["norm_ratio"] == pytest.approx(1.0) for c in out)
    raw[4] = -raw[4]
    assert G.cumsum_check(z, raw)[4]["cos"] == pytest.approx(-1.0)


def _arch(name, ffn="dense", mixer_norm="", ffn_norm="", hc=0, n=3):
    return SimpleNamespace(arch=name, hc_streams=hc, layers=[SimpleNamespace(
        index=i, ffn=ffn if i == 1 else "dense", has_post_ffn_norm=bool(ffn_norm) and i == 2,
        post_norm_mixer=mixer_norm if i == 2 else "", post_norm_ffn=ffn_norm if i == 2 else "") for i in range(n)])


TABLE = {"verified": ["qwen35", "qwen3", "llama", "qwen4exp", "sensenova_loop"]}


def test_arch_check():
    ok = G.arch_check(_arch("qwen35"), TABLE)
    assert ok["refuse"] is None and ok["checked"] and not ok["notes"]
    r = G.arch_check(_arch("gemma3", mixer_norm="post_attention_norm", ffn_norm="post_ffw_norm"), TABLE)["refuse"]
    assert "注意" in r and "MLP" in r and "正規化" in r
    assert "MLP の出力のあと" in G.arch_check(_arch("foo", ffn_norm="post_ffw_norm"), TABLE)["refuse"]
    assert G.arch_check(_arch("qwen3", hc=4), TABLE)["refuse"]
    unk = G.arch_check(_arch("bar", mixer_norm="?"), TABLE)                           # 表に無い arch の post_attention_norm
    assert unk["refuse"] is None and {n["code"] for n in unk["notes"]} == {"post_norm_unknown", "unchecked_arch"}
    moe = G.arch_check(_arch("qwen35moe", ffn="moe"), TABLE)
    assert moe["refuse"] is None and moe["moe_layers"] == [1] and {n["code"] for n in moe["notes"]} == {"moe", "unchecked_arch"}
    # 表を渡さなければ configs/arch/residual_writers.yaml を読む
    assert G.arch_check(_arch("sensenova_loop"))["checked"]


def test_parse_and_compare_names(tmp_path):
    assert G.parse_parts("mlp,layer") == ("layer", "mlp")
    assert G.parse_normalize("both") == ("rows", "none") and G.parse_normalize("rows") == ("rows",)
    assert G.parse_index_list("0-3,8", "層") == [0, 1, 2, 3, 8] and G.parse_index_list(None, "層") is None
    for bad in (lambda: G.parse_parts("heads"), lambda: G.parse_normalize("x"), lambda: G.parse_index_list("3-1", "層"),
                lambda: G.parse_index_list("a", "層")):
        with pytest.raises(GneError):
            bad()
    for n in ("d", "d_r1", "d_r1_r1", "d_r2", "dx_r1", "d_rx", "d_r1.seg0"):
        (tmp_path / f"{n}.npz").write_bytes(b"")
    assert G.compare_names(tmp_path, "d") == ["d_r1", "d_r2", "d_r1_r1"]
    assert G.compare_names(tmp_path / "none", "d") == []


# --- 保存先を使って最後まで(合成の小さな GGUF)------------------------------------------------------------------

def _session_with_material(tmp_path, *, n_a=16, n_b=16, refine=True):
    from gne.api import Session
    from tests.test_writes_rank import _tiny
    _tiny(tmp_path / "m.gguf", n_embd=16, n_layer=3)
    s = Session(tmp_path / "m.gguf", root=tmp_path / "root")
    hd = s.store.heads_dir(s.meta.model_id)
    hd.mkdir(parents=True, exist_ok=True)
    z = _material(n_layer=3, d=16, mixer=["attention"] * 3, n_a=n_a, n_b=n_b)
    np.savez_compressed(hd / "d1.writes.npz", **z)
    (hd / "d1.writes.json").write_text(json.dumps({"direction": "d1", "select_layer": 2, "positions": "last", "n": {"A": n_a, "B": n_b}}))
    dd = s.store.direction_dir(s.meta.model_id)
    dd.mkdir(parents=True, exist_ok=True)
    C = np.cumsum(z["upd_layer_diff"].astype(np.float32), axis=0)                    # 残差の平均の差 = 更新の累積
    np.savez(dd / "d1.npz", layers=np.arange(3, dtype=np.int32), positions="last", **{f"layer_{L}": C[L] for L in range(3)})
    if refine:
        rng = np.random.default_rng(9)
        np.savez(dd / "d1_r1.npz", layers=np.arange(3, dtype=np.int32),
                 **{f"layer_{L}": rng.standard_normal(16).astype(np.float32) for L in range(3)})
        z2 = _material(n_layer=3, d=16, seed=10, mixer=["attention"] * 3)
        np.savez_compressed(hd / "d1_r1.writes.npz", **z2)
    return s, z


def test_heads_geometry_end_to_end(tmp_path):
    from gne.schemas import HeadsGeometryOut
    s, z = _session_with_material(tmp_path)
    d = s.heads_geometry("d1", parts="layer,attn,mlp", top_k=4)
    assert Path(d["path"]).exists() and Path(d["npz"]).exists()
    assert [r["label"] for r in d["rows"]] == ["L0", "L1", "L2", "L0/attn", "L1/attn", "L2/attn", "L0/mlp", "L1/mlp", "L2/mlp"]
    m = d["material"]
    assert m["cumsum_cos_select"] == pytest.approx(1.0, abs=1e-6) and m["halves_exact"] and m["halves_mean_err_max"] < 1e-6
    rows, X, X0, X1, _ = G.build_rows(z, ("layer", "attn", "mlp"), None)
    assert np.allclose(np.array(d["similarity"]["matrix"]), G.cosine_matrix(X), atol=1e-4)
    zz = np.load(d["npz"])
    assert np.allclose(zz["similarity"], G.cosine_matrix(X), atol=1e-6)
    r = s.direction_set("d1")
    rS = r.unit(2).astype(float)
    assert d["rows"][0]["cos_select"] == pytest.approx(X[0] @ rS / np.linalg.norm(X[0]), abs=1e-6)
    assert d["rows"][2]["cos_own"] == pytest.approx(X[2] @ rS / np.linalg.norm(X[2]), abs=1e-6)
    assert d["svd"]["rows"]["k"] == 4 and set(d["svd"]) == {"rows", "none"}
    assert [c["name"] for c in d["compare"]] == ["d1_r1"] and d["compare"][0]["material"] == "d1_r1.writes.npz"
    assert {n["code"] for n in d["notes"]} >= {"few_questions", "compare_meaning"}
    env = {**s.envelope("heads geometry"), **json.loads(Path(d["path"]).read_text(encoding="utf-8"))}
    HeadsGeometryOut.model_validate(env)
    # 方向ベクトルが材料と合わない(累積の cos が低い)と知らせる
    dd = s.store.direction_dir(s.meta.model_id)
    np.savez(dd / "d1.npz", layers=np.arange(3, dtype=np.int32), **{f"layer_{L}": -np.ones(16, np.float32) * (L + 1) for L in range(3)})
    d2 = s.heads_geometry("d1", compare="", out_name="d1b")
    assert "cumsum_mismatch" in {n["code"] for n in d2["notes"]} and d2["compare"] == []


def test_offdiag_stats_skip_same_layer_parts(tmp_path):
    s, z = _session_with_material(tmp_path, refine=False)
    d = s.heads_geometry("d1", parts="layer,attn,mlp")
    rows, X, *_ = G.build_rows(z, ("layer", "attn", "mlp"), None)
    C = G.cosine_matrix(X)
    keep = [(i, j) for i in range(len(rows)) for j in range(len(rows)) if i != j
            and not (rows[i]["layer"] == rows[j]["layer"] and (rows[i]["part"] == "layer") != (rows[j]["part"] == "layer"))]
    sm = d["similarity"]
    assert sm["offdiag_excluded_pairs"] == 6                                       # 3 層 × (層全体と注意・層全体と MLP)
    assert sm["offdiag_abs_max"] == pytest.approx(max(abs(C[i, j]) for i, j in keep))
    assert sm["offdiag_abs_mean"] == pytest.approx(np.mean([abs(C[i, j]) for i, j in keep]))
    assert np.allclose(np.array(sm["matrix"]), C, atol=1e-4)                        # 行列そのものには残す
    assert s.heads_geometry("d1", out_name="d1l")["similarity"]["offdiag_excluded_pairs"] == 0


def test_table_aligns_by_display_width():
    out = G._table(["行", "種類", "cos r̂_l"], [["L0", "線形注意", "+0.99"], ["L10/attn", "MLP", "−"]], left=2)
    assert G._width("線形注意") == 8 and G._width("r̂") == 1 and G._width("cos r̂_l") == 7
    # 右寄せの列の終わりがそろう(表示の幅で数える)
    ends = [G._width(x) for x in out]
    assert ends[0] == ends[1] == ends[2]


def test_heads_geometry_odd_halves_and_errors(tmp_path):
    s, _ = _session_with_material(tmp_path, n_a=15, refine=False)
    d = s.heads_geometry("d1", normalize="rows", min_half_cos=0.99)
    assert "odd_halves" in {n["code"] for n in d["notes"]} and set(d["svd"]) == {"rows"}
    assert d["n_unreliable"] == sum(1 for r in d["rows"] if not r["reliable"]) and d["n_unreliable"] > 0
    with pytest.raises(GneError) as e:
        s.heads_geometry("nothing")
    assert e.value.code == "FILE_NOT_FOUND"
    with pytest.raises(GneError) as e:
        s.heads_geometry("d1", compare="no_such_direction")
    assert e.value.code == "FILE_NOT_FOUND"
    with pytest.raises(GneError):
        s.heads_geometry("d1", exclude_dims="99")


def test_cli_heads_geometry_json_and_text(tmp_path):
    from typer.testing import CliRunner

    from gne.cli import app
    from gne.schemas import ErrorOut, HeadsGeometryOut
    _session_with_material(tmp_path)
    base = ["--model", str(tmp_path / "m.gguf"), "--root", str(tmp_path / "root")]
    r = CliRunner().invoke(app, ["heads", "geometry", "--name", "d1", "--json", *base])
    assert r.exit_code == 0, r.output
    HeadsGeometryOut.model_validate(json.loads(r.output.strip().splitlines()[-1]))
    r = CliRunner().invoke(app, ["heads", "geometry", "--name", "d1", "--parts", "attn,mlp", *base])
    assert r.exit_code == 0 and "特異値分解" in r.output and "読み取りの注意" in r.output and "L2/mlp" in r.output
    r = CliRunner().invoke(app, ["heads", "geometry", "--name", "missing", "--json", *base])
    assert r.exit_code != 0
    err = ErrorOut.model_validate(json.loads(r.output.strip().splitlines()[-1]))
    assert err.error.code == "FILE_NOT_FOUND"


# --- MCP(Qwen3-0.6B の GGUF で gne mcp を起動する。モデルは流さない)------------------------------------------------

@pytest.mark.model
@pytest.mark.skipif(not MODEL.exists(), reason=f"{MODEL} がありません")
def test_mcp_heads_geometry(tmp_path):
    import anyio
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    from gne.api import Session
    s = Session(MODEL, root=tmp_path)
    n_layer, n_embd = s.weights.n_main - 1, s.meta.n_embd
    hd = s.store.heads_dir(s.meta.model_id)
    hd.mkdir(parents=True, exist_ok=True)
    z = _material(n_layer=n_layer, d=n_embd, mixer=["attention"] * n_layer)
    np.savez_compressed(hd / "g.writes.npz", **z)
    (hd / "g.writes.json").write_text(json.dumps({"direction": "g", "select_layer": n_layer - 1, "n": {"A": 16, "B": 16}}))
    gne = str(Path(sys.executable).parent / "gne")

    async def main():
        params = StdioServerParameters(command=gne, args=["mcp", "--model", str(MODEL), "--root", str(tmp_path)], env=dict(os.environ))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as cs:
                await cs.initialize()
                assert "heads_geometry" in {t.name for t in (await cs.list_tools()).tools}
                res = await cs.call_tool("heads_geometry", {"name": "g", "top_k": 3})
                sc = getattr(res, "structured_content", None)
                out = (sc.get("result", sc) if isinstance(sc, dict) else sc) if sc else json.loads(res.content[0].text)
                assert "error" not in out, out
                assert "matrix" not in out["similarity"] and out["svd"]["none"]["k"] == 3 and "U" not in out["svd"]["none"]
                assert {n["code"] for n in out["notes"]} >= {"no_direction"}
                assert Path(out["path"]).exists()
                bad = await cs.call_tool("heads_geometry", {"name": "missing"})
                sc = getattr(bad, "structured_content", None)
                out = (sc.get("result", sc) if isinstance(sc, dict) else sc) if sc else json.loads(bad.content[0].text)
                assert out["error"]["code"] == "FILE_NOT_FOUND"
    anyio.run(main)
