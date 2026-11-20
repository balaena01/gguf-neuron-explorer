"""層ごとの更新ベクトルの幾何(`gne heads geometry`)。

材料は `gne heads writes` が保存した <root>/heads/<model_id>/<名前>.writes.npz の、層ごとの更新の差
d_l = (A 側の平均の更新) − (B 側の平均の更新)(その層が残差に足した量、正規化の前、測る位置の平均)。モデルは流さない。

1. 行: 層ごと(layer)、注意の出力全体(attn = 注意 + 線形注意。ユニット L:attn と同じ範囲)、MLP の出力全体(mlp)。
   部品の行を足すときは、部品ごとにまとめて並べる(layer の行の全部 → attn の行の全部 → mlp の行の全部)。中身の無い行は除く。
2. 類似度の行列: C_ij = (d_i·d_j) / (‖d_i‖‖d_j‖) = (X̃ X̃ᵀ)_ij(X̃ は行の大きさをそろえた行列)。
3. 特異値分解: X = U Σ Vᵀ(rows なら X̃)。寄与率 σ_k² / Σσ², 層ごとの説明率 ‖V_kᵀV_k d_l‖² / ‖d_l‖²。
   基準: 行が互いに直交していたときの値、半分どうしの差 (d_h0 − d_h1)/2(雑音の目安)の特異値、半分ずつで求めた向きでもう半分を
   説明できる割合と、半分どうしの部分空間の重なり。
4. 信頼性: 行ごとの cos(d_h0, d_h1) と ‖(d_h0 − d_h1)/2‖ / ‖d‖。
5. 方向ベクトルとの関係: cos(d_l, r̂_l)(CHiPS がその層で使う方向)・cos(d_l, r̂_S)(選ぶ層の方向)。材料の確かめに、更新の累積と
   方向ベクトル(残差の平均の差)の cos。
6. 2 本目の方向の候補との比べ: 既定は heads refine の方向(<名前>_r1 など)。
ループのモデル(同じ重みを 2 回使う)では、同じ重みの 1 回目と 2 回目の行の組を表にする。

対象は、正規化が副層の入力の手前にだけある構造(出力のあとに正規化がある構造では、残差に入るのは正規化の後の値で、材料の更新は
その前の値なので、意味が変わる)。
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np

from .steer import GneError

# 基準の値(ここだけで変える)
HALF_COS_MIN = 0.5        # 半分どうしの cos がこれ未満の行に、信用しない印を付ける
CUMSUM_COS_MIN = 0.9      # 選ぶ層で、更新の累積と方向ベクトルの cos がこれ未満なら、材料と方向が合わない疑いを知らせる
FEW_QUESTIONS = 32        # 片側の質問がこれより少ないと、半分ずつの数は揺れやすいと知らせる
TOP_K = 6                 # 寄与率・説明率・保存する向きの数の既定
TOP_DIMS = 5              # 行ごとに出す、大きい次元の数

PARTS = ("layer", "attn", "mlp")
PART_JA = {"layer": "層全体", "attn": "注意", "mlp": "MLP"}
MIXER_JA = {"attention": "注意", "linear_attention": "線形注意", "recurrent": "再帰", "mlp": "MLP", "none": "なし"}
NORMALIZE = ("rows", "none")
READING = [
    "類似度の行列のかたまりや特異値分解の上位の向きは、その層たちの更新の差が似ていることしか示しません。狙った振る舞いが"
    "そこで作られていることは意味しません。A 側と B 側の質問の違い(話題・長さ・言語・system の長さ)も、同じように差に映ります"
    "(側ごとに別の質問を読ませる対照集合では、とくにそうです)。",
    "更新の差は、測る位置(既定はプロンプトの最後のトークン)での平均です。応答を書いている最中の更新ではありません。",
    "cos は大きさを見ません。小さな更新の層どうしの cos は雑音で大きく動くので、大きさと信用しない印と並べて読みます。"
    "少数の次元に偏った層は、その次元だけで cos が決まることがあります(行ごとの「上位の次元」の割合)。",
    "行の大きさをそろえない特異値分解の上位は、どの層の更新が大きいかを主に映します(後ろの層ほど大きい)。行が互いに直交して"
    "いたときの値・雑音の目安と並べて読みます。",
    "直接の書き込み(その層が足した量)の比べで、後ろの層がそれをどう変えるかは見ません。",
    "2 本目の向きの候補が見つかっても、それが応答に効くかは別に確かめます。",
]


# --- 指定の読み方 ------------------------------------------------------------------------------------------

def parse_parts(spec) -> tuple[str, ...]:
    """行の単位: layer / attn / mlp のカンマ区切り(例 layer、attn,mlp、layer,attn,mlp)。並びは PARTS の順にそろえる。"""
    parts = [x.strip().lower() for x in (spec.split(",") if isinstance(spec, str) else (spec or ["layer"])) if str(x).strip()]
    bad = [x for x in parts if x not in PARTS]
    if bad or not parts:
        raise GneError("BAD_ARGS", f"行の単位は layer・attn・mlp のカンマ区切りです({spec!r})",
                       "layer(層ごと)、attn,mlp(注意の出力全体と MLP の出力全体)、layer,attn,mlp")
    return tuple(p for p in PARTS if p in parts)


def parse_normalize(spec) -> tuple[str, ...]:
    """特異値分解の前に行の大きさをそろえるか: rows / none / both。"""
    s = str(spec or "both").strip().lower()
    if s == "both":
        return NORMALIZE
    if s in NORMALIZE:
        return (s,)
    raise GneError("BAD_ARGS", f"normalize は rows(行の大きさをそろえる)/ none(そろえない)/ both です({spec!r})")


def parse_index_list(spec, what: str) -> list[int] | None:
    """'10-30'・'3,7,12'・'0-3,8' か整数の並び。None・空は None(全部)。"""
    if spec is None or spec == "" or spec == []:
        return None
    if isinstance(spec, (list, tuple)):
        return sorted({int(x) for x in spec})
    out: set[int] = set()
    for part in str(spec).split(","):
        m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", part)
        if not m:
            raise GneError("BAD_ARGS", f"{what}の指定 {spec!r} を読めません(例 10-30、3,7,12)")
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        if b < a:
            raise GneError("BAD_ARGS", f"{what}の範囲が逆です({part.strip()})")
        out |= set(range(a, b + 1))
    return sorted(out)


# --- 構造の判定 -------------------------------------------------------------------------------------------

def arch_check(arch, table: dict | None = None) -> dict:
    """更新ベクトルを「その層が残差に足した量」と読める構造か。戻り値 {arch, refuse(理由か None), checked, moe_layers, notes}。
    出力のあとの正規化は、層ごと・副層ごとの判定(ArchInfo の post_norm_mixer・post_norm_ffn。configs/arch/residual_writers.yaml の
    output_norm の表から作る)で見る。確かめた構造かは、同じ表の verified。"""
    if table is None:
        from .ablate import writers_table
        table = writers_table()
    name = str(getattr(arch, "arch", "") or "")
    layers = list(getattr(arch, "layers", []) or [])
    notes: list[dict] = []
    refuse = None
    pm = [li for li in layers if getattr(li, "post_norm_mixer", "") not in ("", "?")]
    pf = [li for li in layers if getattr(li, "post_norm_ffn", "") or getattr(li, "has_post_ffn_norm", False)]
    if getattr(arch, "hc_streams", 0):
        refuse = "残差が複数本の流れの構造(hyper-connections)で、層ごとの更新の材料がありません"
    elif pm or pf:
        what = [f"注意(混ぜる層)の出力のあと({len(pm)} 層)" if pm else "", f"MLP の出力のあと({len(pf)} 層)" if pf else ""]
        refuse = (f"{name} は{'と'.join(w for w in what if w)}に正規化がある構造です。残差に入るのは正規化の後の値で、材料の更新は"
                  "その前の値なので、層ごとの更新を比べる量の意味が変わります(この構造には未対応。heads writes もこの構造では実行しません)")
    pu = [int(li.index) for li in layers if getattr(li, "post_norm_mixer", "") == "?"]
    if pu and refuse is None:
        notes.append({"code": "post_norm_unknown", "text": f"層 {', '.join(map(str, pu[:6]))}{' …' if len(pu) > 6 else ''} の post_attention_norm が、"
                                                           "注意の出力のあとの正規化か MLP の手前の正規化か、この arch では確かめていません。"
                                                           "出力のあとなら、注意と層全体の行は残差に入る量を表しません"})
    moe = [int(li.index) for li in layers if str(getattr(li, "ffn", "dense")).startswith("moe")]
    if moe and refuse is None:
        notes.append({"code": "moe", "text": f"MoE の層({len(moe)} 層)は、材料に MLP の更新がありません(dense の MLP だけを測っています)。"
                                             "その層は、層全体と MLP の行を作らず、注意の行だけを使います"})
    verified = [str(x) for x in (table.get("verified") or [])]
    checked = name in verified
    if not checked and refuse is None:
        notes.append({"code": "unchecked_arch", "text": f"{name} は確かめていない構造です(確かめたのは {', '.join(verified)})。"
                                                        "出力のあとの正規化は見つかっていないので、そのまま計算しました"})
    return {"arch": name, "refuse": refuse, "checked": checked, "moe_layers": moe, "notes": notes}


# --- 材料 -------------------------------------------------------------------------------------------------

def load_material(heads_dir: Path, name: str) -> tuple[dict, dict]:
    """<名前>.writes.npz(配列)と .writes.json(まとめ)。"""
    if not re.fullmatch(r"[\w.\-]+", str(name or "")):
        raise GneError("BAD_ARGS", f"名前 {name!r} は英数字・_・-・. だけです")
    pz, pj = heads_dir / f"{name}.writes.npz", heads_dir / f"{name}.writes.json"
    if not pz.exists():
        raise GneError("FILE_NOT_FOUND", f"heads writes の材料 {name} がありません({pz.name})",
                       "gne heads writes --name <方向>(または gne heads pipeline)で作ります")
    z = np.load(pz)
    need = ["layer_index", "upd_layer_diff", "upd_layer_diff_h0", "upd_layer_diff_h1"]
    miss = [k for k in need if k not in z.files]
    if miss:
        raise GneError("BAD_ARGS", f"{pz.name} に層ごとの更新がありません({', '.join(miss)})",
                       "gne heads writes(v0.8.0 以降)で作り直してください")
    arrays = {k: z[k] for k in z.files if not k.startswith("neuron_")}
    js = json.loads(pj.read_text(encoding="utf-8")) if pj.exists() else {}
    return arrays, js


def part_vectors(z: dict, part: str, suffix: str = "diff") -> np.ndarray:
    """部品の種類ごとの更新の差 (層の数, n_embd)。attn = 注意 + 線形注意(どちらかしか持たない層は、持つ方)。"""
    if part == "layer":
        return z[f"upd_layer_{suffix}"].astype(np.float64)
    if part == "attn":
        return z[f"upd_attn_{suffix}"].astype(np.float64) + z[f"upd_linear_{suffix}"].astype(np.float64)
    return z[f"upd_mlp_{suffix}"].astype(np.float64)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _cos(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else None


def top_dims(v: np.ndarray, n: int = TOP_DIMS) -> dict:
    """‖v‖² のうち、大きい順に n 個の次元が占める割合と、その次元の番号。"""
    sq = v * v
    tot = float(sq.sum())
    idx = np.argsort(-sq)[:n]
    return {"share": float(sq[idx].sum() / tot) if tot > 0 else None, "dims": [int(i) for i in idx]}


def build_rows(z: dict, parts, layers: list[int] | None, moe_layers=(), exclude_dims=()) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, dict]:
    """行(部品ごとにまとめて、層の順)と、行列 X・X_h0・X_h1。除いた行の数を部品ごとに返す。
    exclude_dims の次元は 0 にしてから計算する(行ごとの大きい次元は除く前の値で出す)。"""
    Ls = [int(x) for x in z["layer_index"]]
    pos = {L: i for i, L in enumerate(Ls)}
    use = Ls if layers is None else [L for L in Ls if L in set(layers)]
    if layers is not None and not use:
        raise GneError("BAD_ARGS", f"材料に指定の層がありません(材料の層は {Ls[0]}〜{Ls[-1]})")
    mixer = [str(x) for x in z.get("layer_mixer", np.array(["?"] * len(Ls)))]
    visit = [int(x) for x in z.get("layer_visit", np.zeros(len(Ls), int))]
    stored = [int(x) for x in z.get("layer_stored", np.array(Ls))]
    chk = z.get("check_layer_rel")
    moe = set(int(x) for x in moe_layers)
    rows, xs, x0s, x1s = [], [], [], []
    skipped = {p: 0 for p in parts}
    for p in parts:
        V, V0, V1 = part_vectors(z, p), part_vectors(z, p, "diff_h0"), part_vectors(z, p, "diff_h1")
        for L in use:
            i = pos[L]
            if p in ("layer", "mlp") and L in moe:
                skipped[p] += 1
                continue
            v = V[i]
            if not np.any(v):
                skipped[p] += 1
                continue
            r = {"label": f"L{L}" if p == "layer" else f"L{L}/{p}", "layer": L, "part": p,
                 "mixer": "mlp" if p == "mlp" else mixer[i], "visit": visit[i], "stored": stored[i],
                 "top_dims": top_dims(v)}
            if chk is not None and p == "layer" and np.isfinite(chk[i]):
                r["check_layer_rel"] = float(chk[i])
            v, v0, v1 = v.copy(), V0[i].copy(), V1[i].copy()
            if len(exclude_dims):
                v[list(exclude_dims)] = 0.0
                v0[list(exclude_dims)] = 0.0
                v1[list(exclude_dims)] = 0.0
            rows.append(r)
            xs.append(v)
            x0s.append(v0)
            x1s.append(v1)
    if not rows:
        raise GneError("BAD_ARGS", "中身のある行がありません(指定の層・部品の更新がどれも 0)")
    return rows, np.stack(xs), np.stack(x0s), np.stack(x1s), skipped


# --- 計算(行列だけを見る部分。テストはここを直接たたく)------------------------------------------------------

def unit_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, n, out=np.zeros_like(X), where=n > 0)


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    """C = X̃ X̃ᵀ(行どうしのコサイン類似度)。"""
    Xt = unit_rows(X)
    return Xt @ Xt.T


def svd_modes(M: np.ndarray, k: int) -> dict:
    """M = U Σ Vᵀ の上位 k 個。符号は U の列の和が正になる向き(和が 0 なら絶対値の最大の要素が正)。"""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    k = min(int(k), len(s))
    U, s, Vt = U[:, :k].copy(), s.copy(), Vt[:k].copy()
    for j in range(k):
        c = U[:, j].sum()
        if abs(c) < 1e-12:
            c = U[np.argmax(np.abs(U[:, j])), j]
        if c < 0:
            U[:, j] *= -1
            Vt[j] *= -1
    tot = float((s ** 2).sum())
    ex = (s ** 2) / tot if tot > 0 else np.zeros_like(s)
    return {"U": U, "s": s, "Vt": Vt, "explained": ex, "k": k, "total": tot}


def row_explained(U: np.ndarray, s: np.ndarray, M: np.ndarray) -> np.ndarray:
    """層ごとの説明率: 上位 j 本(j = 1..k)の向きで、行 i をどれだけ説明できるか ‖V_jᵀV_j m_i‖² / ‖m_i‖²。
    m_i の V_j 成分は U_ij σ_j なので、その 2 乗の累積を ‖m_i‖² で割る。"""
    k = U.shape[1]
    num = np.cumsum((U * s[None, :k]) ** 2, axis=1)
    den = (M ** 2).sum(axis=1, keepdims=True)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def explained_by(Vt: np.ndarray, M: np.ndarray) -> np.ndarray:
    """向き Vt(k × d、正規直交)の上位 j 本(j = 1..k)で、行列 M 全体のどれだけを説明できるか(‖M‖² に対する割合)。"""
    tot = float((M ** 2).sum())
    if tot <= 0:
        return np.zeros(Vt.shape[0])
    return np.cumsum(((M @ Vt.T) ** 2).sum(axis=0)) / tot


def subspace_overlap(Va: np.ndarray, Vb: np.ndarray) -> np.ndarray:
    """上位 j 本どうし(j = 1..k)の部分空間の重なり: 主角の cos の 2 乗の平均 ‖Va_j Vb_jᵀ‖_F² / j(1 = 同じ部分空間、0 = 直交)。"""
    k = min(Va.shape[0], Vb.shape[0])
    return np.array([float(((Va[:j] @ Vb[:j].T) ** 2).sum()) / j for j in range(1, k + 1)])


def half_stats(X: np.ndarray, X0: np.ndarray, X1: np.ndarray) -> tuple[list, np.ndarray]:
    """行ごとの半分どうしの cos と、雑音の目安の比 ‖(X0 − X1)/2‖ / ‖X‖。"""
    hc = [_cos(X0[i], X1[i]) for i in range(len(X))]
    n = np.linalg.norm(X, axis=1)
    nz = np.linalg.norm((X0 - X1) / 2, axis=1)
    return hc, np.divide(nz, n, out=np.full_like(n, np.nan), where=n > 0)


def svd_analysis(X: np.ndarray, X0: np.ndarray, X1: np.ndarray, mode: str, k: int) -> tuple[dict, dict]:
    """1 つの版(rows = 行の大きさをそろえる / none = そろえない)の特異値分解と基準。戻り値 (まとめ, 配列 U・s・Vt)。
    - 直交の基準: 行が互いに直交なら出る寄与率(rows は全部 1/行の数、none は ‖x_i‖² を大きい順に並べた割合)
    - 雑音の基準: 半分どうしの差 N = (X0 − X1)/2(rows では行ごとに ‖x_i‖ で割る)の上位の特異値の 2 乗 ÷ ‖M‖²(データの全体に
      対する割合)と、全体 ‖N‖² ÷ ‖M‖²
    - 半分ずつでの確かめ: X0 で求めた上位 j 本で X1 を説明できる割合(逆も、その平均)と、半分どうしの部分空間の重なり"""
    n_rows = np.linalg.norm(X, axis=1)
    if mode == "rows":
        M = unit_rows(X)
        N = np.divide((X0 - X1) / 2, n_rows[:, None], out=np.zeros_like(X), where=n_rows[:, None] > 0)
        M0, M1 = unit_rows(X0), unit_rows(X1)
    else:
        M, N, M0, M1 = X, (X0 - X1) / 2, X0, X1
    m = svd_modes(M, k)
    kk = m["k"]
    tot = m["total"]
    if mode == "rows":
        orth = np.full(kk, 1.0 / len(M))
    else:
        sq = np.sort(n_rows ** 2)[::-1]
        orth = sq[:kk] / sq.sum()
    sn = np.linalg.svd(N, compute_uv=False)
    noise_k = (sn[:kk] ** 2) / tot if tot > 0 else np.zeros(kk)
    m0, m1 = svd_modes(M0, kk), svd_modes(M1, kk)
    cross = (explained_by(m0["Vt"], M1) + explained_by(m1["Vt"], M0)) / 2
    overlap = subspace_overlap(m0["Vt"], m1["Vt"])
    rx = row_explained(m["U"], m["s"], M)
    out = {"mode": mode, "k": kk, "explained": [float(x) for x in m["explained"][:kk]],
           "cumulative": [float(x) for x in np.cumsum(m["explained"])[:kk]],
           "singular_values": [float(x) for x in m["s"][:kk]],
           "baseline_orthogonal": [float(x) for x in orth],
           "baseline_noise": [float(x) for x in noise_k], "noise_total": float((N ** 2).sum() / tot) if tot > 0 else None,
           "half_cross": [float(x) for x in cross], "half_overlap": [float(x) for x in overlap],
           "U": [[float(x) for x in row] for row in m["U"]],
           "row_explained": [[float(x) for x in row] for row in rx]}
    return out, {"U": m["U"], "s": m["s"], "Vt": m["Vt"]}


def loop_pairs(z: dict, layers: list[int] | None, exclude_dims=()) -> dict | None:
    """同じ重みの 1 回目と 2 回目の行の組(layer_visit 1 と 2、layer_stored が同じ)。部品ごとの cos と大きさ。"""
    if "layer_visit" not in z or not np.any(np.asarray(z["layer_visit"]) > 0):
        return None
    Ls = [int(x) for x in z["layer_index"]]
    vis, st = [int(x) for x in z["layer_visit"]], [int(x) for x in z["layer_stored"]]
    keep = None if layers is None else set(layers)
    second = {st[j]: j for j in range(len(Ls)) if vis[j] == 2}
    pairs, shifts = [], set()
    vec = {p: (part_vectors(z, p), part_vectors(z, p, "diff_h0"), part_vectors(z, p, "diff_h1")) for p in PARTS}
    for i in range(len(Ls)):
        if vis[i] != 1 or st[i] not in second:
            continue
        j = second[st[i]]
        if keep is not None and not (Ls[i] in keep and Ls[j] in keep):
            continue
        shifts.add(Ls[j] - Ls[i])
        row = {"stored": st[i], "first": Ls[i], "second": Ls[j]}
        for p in PARTS:
            a, a0, a1 = (v[i].copy() for v in vec[p])
            b, b0, b1 = (v[j].copy() for v in vec[p])
            for x in (a, a0, a1, b, b0, b1):
                if len(exclude_dims):
                    x[list(exclude_dims)] = 0.0
            if not np.any(a) or not np.any(b):
                row[p] = None
                continue
            row[p] = {"cos": _cos(a, b), "norm_first": float(np.linalg.norm(a)), "norm_second": float(np.linalg.norm(b)),
                      "half_cos_first": _cos(a0, a1), "half_cos_second": _cos(b0, b1),
                      # 半分ずつで測った 1 回目と 2 回目の cos(偶数番目どうし・奇数番目どうしの平均。値の揺れの目安)
                      "cos_halves": [_cos(a0, b0), _cos(a1, b1)]}
        pairs.append(row)
    if not pairs:
        return None
    return {"shift": sorted(shifts)[0] if len(shifts) == 1 else sorted(shifts), "pairs": pairs}


def cumsum_check(z: dict, raw: dict[int, np.ndarray]) -> list[dict]:
    """更新の累積 Σ_{l'≤l} d_l' と方向ベクトルの元の値 r_l(残差の平均の差)の cos と大きさの比(層ごと)。測る位置が両側で同じ
    テンプレートのトークンなら、埋め込みの差は 0 で、累積は r_l に一致する。"""
    Ls = [int(x) for x in z["layer_index"]]
    C = np.cumsum(z["upd_layer_diff"].astype(np.float64), axis=0)
    out = []
    for i, L in enumerate(Ls):
        if L not in raw:
            continue
        r = np.asarray(raw[L], np.float64)
        nr = float(np.linalg.norm(r))
        out.append({"layer": L, "cos": _cos(C[i], r), "norm_ratio": float(np.linalg.norm(C[i]) / nr) if nr > 0 else None})
    return out


def compare_names(direction_dir: Path, name: str) -> list[str]:
    """既定の比べる方向: heads refine で作った方向(<名前>_r1、<名前>_r1_r1、<名前>_r2 など)。"""
    if not direction_dir.exists():
        return []
    pat = re.compile(rf"^{re.escape(name)}(_r\d+)+$")
    names = [p.name[:-4] for p in direction_dir.glob(f"{name}_r*.npz") if pat.match(p.name[:-4])]
    return sorted(names, key=lambda x: (len(x), x))


# --- 質問ごとの更新(heads writes --save-q の <名前>.writes_q.npz)-----------------------------------------------

Q_MATCH_TOL = 0.01        # 質問ごとの更新の平均と材料の側ごとの平均の相対誤差の上限(float16 の丸めは 1e-3 程度)
PERMUTATIONS = 200        # ラベル入れ替えの基準の回数の既定


def save_q(heads_dir: Path, name: str, qarr: dict) -> Path:
    """質問ごと・層ごとの更新(measure_writes の keep_q)を <名前>.writes_q.npz に保存する(圧縮しない。float16 はほとんど縮まない)。"""
    heads_dir.mkdir(parents=True, exist_ok=True)
    p = heads_dir / f"{name}.writes_q.npz"
    np.savez(p, **qarr)
    return p


def load_q(heads_dir: Path, name: str, z: dict) -> tuple[dict | None, dict | None]:
    """(質問ごとの更新, 知らせ)。無ければ (None, None)。質問の平均が材料の側ごとの平均と合わない(材料を作り直したあとの古い
    ファイルなど)ときは使わず、知らせを返す。戻り値の A・B は {"attn": (質問, 層, d), "mlp": …}(float32)。"""
    p = heads_dir / f"{name}.writes_q.npz"
    if not p.exists():
        return None, None
    q = np.load(p)

    def get(k):
        return q[f"q_upd_{k}"].astype(np.float32) * q[f"q_scale_{k}"].astype(np.float32)[None, :, None]
    A, B = {"attn": get("attn_a"), "mlp": get("mlp_a")}, {"attn": get("attn_b"), "mlp": get("mlp_b")}
    err = 0.0
    for side, D in (("a", A), ("b", B)):
        refs = {"attn": z[f"upd_attn_{side}"].astype(np.float64) + z[f"upd_linear_{side}"].astype(np.float64),
                "mlp": z[f"upd_mlp_{side}"].astype(np.float64)}
        for k in ("attn", "mlp"):
            if D[k].shape[1:] != refs[k].shape:
                return None, {"code": "q_mismatch", "text": f"{p.name} の形が材料と違うので、質問ごとの更新は使いません"}
            nr = float(np.linalg.norm(refs[k]))
            if nr > 0:
                err = max(err, float(np.linalg.norm(D[k].astype(np.float64).mean(0) - refs[k]) / nr))
    if err > Q_MATCH_TOL:
        return None, {"code": "q_mismatch", "text": f"{p.name} の質問ごとの更新の平均が材料の平均と合いません(相対誤差 {err:.1%})。材料を作り直した"
                                                    "あとの古いファイルの疑いがあるので使いません(heads writes --save-q で作り直してください)"}
    return {"A": A, "B": B, "paired": bool(q["q_paired"]), "index_a": [int(x) for x in q["q_index_a"]],
            "index_b": [int(x) for x in q["q_index_b"]], "match_err": err, "file": p.name, "bytes": p.stat().st_size,
            "contrast": str(q["q_contrast"]) if "q_contrast" in q.files else None,
            "contrast_sha": str(q["q_contrast_sha"]) if "q_contrast_sha" in q.files else None}, None


def q_rows(qd: dict, rows: list[dict], layer_pos: dict[int, int], exclude_dims=()) -> tuple[np.ndarray, np.ndarray]:
    """行ごとの質問ごとの更新 (A の質問, 行, d)・(B の質問, 行, d)。層全体 = 注意 + MLP。"""
    def one(D):
        out = np.zeros((D["attn"].shape[0], len(rows), D["attn"].shape[2]), np.float32)
        for i, r in enumerate(rows):
            j = layer_pos[r["layer"]]
            out[:, i] = (D["attn"][:, j] + D["mlp"][:, j]) if r["part"] == "layer" else D[r["part"]][:, j]
        if len(exclude_dims):
            out[:, :, list(exclude_dims)] = 0.0
        return out
    return one(qd["A"]), one(qd["B"])


def _sym_eigvals(G: np.ndarray) -> np.ndarray:
    """対称行列(行列の束)の固有値を大きい順に。"""
    return np.linalg.eigvalsh(G)[..., ::-1]


def label_null(QA: np.ndarray, QB: np.ndarray, paired: bool, n_perm: int = PERMUTATIONS, seed: int = 0,
               k: int = TOP_K, chunk: int = 10) -> dict:
    """A と B のラベルを入れ替えて作る基準(§2.5)。対になった対照集合は質問ごとの差の符号を無作為に入れ替え、対にならない対照集合は
    両側の質問を混ぜて同じ数ずつに分け直す。入れ替えた差ごとに、行の大きさ・半分どうしの cos・行どうしの |cos|・特異値分解の
    寄与率(両方の版)を出し、n_perm 回の分布を返す。半分は、入れ替えたあとの各側の並びの偶数番目・奇数番目。
    QA・QB は (質問, 行, d)。"""
    rng = np.random.default_rng(seed)
    nA, nB = QA.shape[0], QB.shape[0]
    nr, d = QA.shape[1], QA.shape[2]
    if paired:
        U = (QA - QB).reshape(nA, nr * d)
    else:
        U = np.concatenate([QA, QB]).reshape(nA + nB, nr * d)
    N = U.shape[0]
    kk = min(int(k), nr)
    norms, hcos, ex_rows, ex_none = [], [], [], []
    cos_abs = []
    done = 0
    while done < n_perm:
        P = min(chunk, n_perm - done)
        W, W0, W1 = np.zeros((P, N)), np.zeros((P, N)), np.zeros((P, N))
        for p_ in range(P):
            if paired:
                sg = rng.choice([-1.0, 1.0], size=nA)
                ev = np.arange(nA) % 2 == 0
                W[p_] = sg / nA
                W0[p_] = np.where(ev, sg / max(1, ev.sum()), 0.0)
                W1[p_] = np.where(~ev, sg / max(1, (~ev).sum()), 0.0)
            else:
                pi = rng.permutation(N)
                a, b = pi[:nA], pi[nA:]
                W[p_, a], W[p_, b] = 1.0 / nA, -1.0 / nB
                for Wh, sl in ((W0, slice(0, None, 2)), (W1, slice(1, None, 2))):
                    ah, bh = a[sl], b[sl]
                    Wh[p_, ah], Wh[p_, bh] = 1.0 / max(1, len(ah)), -1.0 / max(1, len(bh))
        D = (W @ U).reshape(P, nr, d)
        D0 = (W0 @ U).reshape(P, nr, d)
        D1 = (W1 @ U).reshape(P, nr, d)
        nd = np.linalg.norm(D, axis=2)
        norms.append(nd)
        n0, n1 = np.linalg.norm(D0, axis=2), np.linalg.norm(D1, axis=2)
        hcos.append(np.einsum("prd,prd->pr", D0, D1) / np.maximum(n0 * n1, 1e-30))
        Dt = D / np.maximum(nd, 1e-30)[:, :, None]
        C = Dt @ Dt.transpose(0, 2, 1)
        cos_abs.append(np.abs(C).astype(np.float32))
        er = _sym_eigvals(C)
        ex_rows.append(er[:, :kk] / np.maximum(er.sum(axis=1, keepdims=True), 1e-30))
        en = _sym_eigvals(D @ D.transpose(0, 2, 1))
        ex_none.append(en[:, :kk] / np.maximum(en.sum(axis=1, keepdims=True), 1e-30))
        done += P
    return {"scheme": "sign_flip" if paired else "shuffle", "n_perm": int(n_perm), "seed": int(seed),
            "norms": np.concatenate(norms), "half_cos": np.concatenate(hcos), "cos_abs": np.concatenate(cos_abs),
            "explained": {"rows": np.concatenate(ex_rows), "none": np.concatenate(ex_none)}}


def within_row(EA: np.ndarray, EB: np.ndarray, paired: bool, r_own: np.ndarray | None, rS: np.ndarray | None, k: int) -> dict:
    """同じ行(層・部品)の中で、質問ごとの差を分解する(§2.6)。EA・EB はその行の A 側・B 側の質問ごとの更新 (質問, d)。
    1. 質問ごとの差 e_q: 対になった対照集合では u_q^A − u_q^B、対にならない対照集合では u_q^A − ū^B(行の平均は d)。
    2. 1 本の方向での説明率 Σ(e_q·v)² / Σ‖e_q‖²(v = d̂・r̂_l・r̂_S)。
    3. e_q を並べた行列の特異値分解(中心をずらさない)の寄与率と、1 番目の向きと d̂・r̂_l の cos。
    4. 1 本の方向で取り残す分 E (I − d̂d̂ᵀ) の特異値分解の寄与率(Σ‖e_q‖² に対する割合)。
    5. 取り残しの向きの安定度: 偶数番目・奇数番目の質問で別々に求めた向きどうしの |cos| と部分空間の重なり。
       対にならない対照集合では、片方の半分で求めた向きに沿った A 側のもう半分の質問の広がり ÷ B 側の質問どうしの広がり
       (1 前後なら、ふつうの質問どうしの違いと同じ程度)。
    6. 質問ごとの cos(e_q, d̂)。"""
    EA, EB = EA.astype(np.float64), EB.astype(np.float64)
    E = EA - EB if paired else EA - EB.mean(0)
    dv = E.mean(0)
    nd = float(np.linalg.norm(dv))
    dh = dv / nd if nd > 0 else dv
    sq = (E ** 2).sum(axis=1)
    tot = float(sq.sum())
    nA = E.shape[0]

    def share(v):
        return float(((E @ v) ** 2).sum() / tot) if v is not None and tot > 0 else None
    kk = max(1, min(int(k), nA))
    U, s, Vt = np.linalg.svd(E, full_matrices=False)
    w1 = Vt[0] * (1.0 if Vt[0] @ dh >= 0 else -1.0)
    Ep = E - np.outer(E @ dh, dh)
    _, sp, Vp = np.linalg.svd(Ep, full_matrices=False)
    out = {"n": nA, "share_mean": share(dh), "share_own": share(r_own), "share_select": share(rS),
           "explained": [float(x) for x in (s[:kk] ** 2) / tot] if tot > 0 else [],
           "cos_w1_mean": float(w1 @ dh), "cos_w1_own": float(w1 @ r_own) if r_own is not None else None,
           "leftover_total": float((Ep ** 2).sum() / tot) if tot > 0 else None,
           "leftover_explained": [float(x) for x in (sp[:kk] ** 2) / tot] if tot > 0 else []}
    ev = np.arange(nA) % 2 == 0
    kh = max(1, min(kk, int(ev.sum()) - 1, int((~ev).sum()) - 1))
    if ev.sum() >= 2 and (~ev).sum() >= 2:
        V0 = np.linalg.svd(Ep[ev], full_matrices=False)[2][:kh]
        V1 = np.linalg.svd(Ep[~ev], full_matrices=False)[2][:kh]
        out["leftover_stability"] = [float(abs(V0[j] @ V1[j])) for j in range(kh)]
        out["leftover_overlap"] = [float(x) for x in subspace_overlap(V0, V1)]
        if not paired:
            F = EB - EB.mean(0)
            rat = []
            for j in range(kh):
                r_ = []
                for Vh, held in ((V0, ~ev), (V1, ev)):
                    a_ = float(((Ep[held] @ Vh[j]) ** 2).mean())
                    b_ = float(((F @ Vh[j]) ** 2).mean())
                    r_.append(a_ / b_ if b_ > 0 else None)
                rat.append(float(np.mean([x for x in r_ if x is not None])) if any(x is not None for x in r_) else None)
            out["leftover_ab_ratio"] = rat
    nq = np.sqrt(sq)
    out["q_cos"] = [float(x) for x in np.divide(E @ dh, nq, out=np.zeros_like(nq), where=nq > 0)]
    return out


def _questions(qd: dict) -> dict[int, str]:
    """質問の番号 → 文(A 側。材料を作った対照集合が同じものと確かめられるときだけ)。"""
    path, sha = qd.get("contrast"), qd.get("contrast_sha")
    if not path or not Path(path).exists():
        return {}
    try:
        from .reverse import ContrastSet
        cs = ContrastSet.load(path)
    except (GneError, OSError, ValueError):
        return {}
    if sha and cs.sha != sha:
        return {}
    out = {}
    for i, q in enumerate(cs.prompts("A")):
        x = q if isinstance(q, str) else next((m.get("content") or "" for m in reversed(q) if m.get("role") == "user"), "")
        out[i] = x if len(x) <= 120 else x[:119] + "…"
    return out


# --- まとめ -----------------------------------------------------------------------------------------------

def analyze(session, name: str, *, parts="layer", normalize="both", top_k: int = TOP_K, layers=None,
            min_half_cos: float = HALF_COS_MIN, compare=None, exclude_dims=None, out_name: str | None = None,
            permutations: int = PERMUTATIONS, seed: int = 0) -> dict:
    """heads writes の材料 name の層ごとの更新を比べ、<out_name>.geometry.json と .geometry.npz に保存する。モデルは流さない。
    質問ごとの更新(<name>.writes_q.npz、heads writes --save-q)があれば、ラベル入れ替えの基準(permutations 回、0 で切る)と
    同じ層の中の質問に沿った分解も出す。"""
    t0 = time.perf_counter()
    parts_t, modes = parse_parts(parts), parse_normalize(normalize)
    if int(top_k) < 1:
        raise GneError("BAD_ARGS", "top_k は 1 以上です")
    out_name = out_name or name
    if not re.fullmatch(r"[\w.\-]+", str(out_name)):
        raise GneError("BAD_ARGS", f"保存名 {out_name!r} は英数字・_・-・. だけです")
    ac = arch_check(session.weights.arch)
    if ac["refuse"]:
        raise GneError("ARCH_UNSUPPORTED", ac["refuse"])
    hd = session.store.heads_dir(session.meta.model_id)
    z, js = load_material(hd, name)
    n_embd = int(z["upd_layer_diff"].shape[1])
    lay = parse_index_list(layers, "層")
    ex = parse_index_list(exclude_dims, "次元") or []
    if any(d >= n_embd for d in ex):
        raise GneError("BAD_ARGS", f"除く次元は 0〜{n_embd - 1} です")
    rows, X, X0, X1, skipped = build_rows(z, parts_t, lay, ac["moe_layers"], ex)
    notes = list(ac["notes"])
    S = int(z["select_layer"]) if "select_layer" in z else js.get("select_layer")
    n_a, n_b = int(z.get("n_a", js.get("n", {}).get("A", 0))), int(z.get("n_b", js.get("n", {}).get("B", 0)))
    # 材料の確かめ
    full = z["upd_layer_diff"].astype(np.float64)
    halves = (z["upd_layer_diff_h0"].astype(np.float64) + z["upd_layer_diff_h1"].astype(np.float64)) / 2
    nf = np.linalg.norm(full, axis=1)
    mh = np.divide(np.linalg.norm(full - halves, axis=1), nf, out=np.zeros_like(nf), where=nf > 0)
    material = {"name": name, "direction": js.get("direction"), "select_layer": S, "n_a": n_a, "n_b": n_b,
                "positions": js.get("positions"), "subset": js.get("subset"), "max_examples": js.get("max_examples"),
                "loop_direction": js.get("loop_direction"), "n_layers": len(z["layer_index"]), "n_embd": n_embd,
                "halves_exact": bool(n_a % 2 == 0 and n_b % 2 == 0), "halves_mean_err_max": float(mh.max())}
    if not material["halves_exact"]:
        notes.append({"code": "odd_halves", "text": f"質問の数が奇数(A {n_a}・B {n_b})なので、全体の差は 2 つの半分の平均と少しずれます"
                                                    f"(最大 {mh.max():.1%})。雑音の目安はそのぶん粗くなります"})
    chk = z.get("check_layer_rel")
    if chk is not None and np.isfinite(chk).any():
        material["check_layer_rel_median"] = float(np.nanmedian(chk))
        material["check_layer_rel_max"] = float(np.nanmax(chk))
    qd, qnote = load_q(hd, name, z)
    if qnote:
        notes.append(qnote)
    if qd is not None:
        material["q"] = {k: qd[k] for k in ("file", "bytes", "paired", "match_err")} | {"n_a": len(qd["index_a"]), "n_b": len(qd["index_b"])}
    else:
        notes.append({"code": "no_q", "text": "質問ごとの更新が無いので、ラベル入れ替えの基準と、同じ層の中の質問に沿った分解は出していません"
                                              f"(gne heads writes --name <方向> --save-q で {name}.writes_q.npz を作ります)"})
    if min(n_a, n_b) < FEW_QUESTIONS:
        notes.append({"code": "few_questions", "text": f"質問が少ない(A {n_a}・B {n_b})ので、半分ずつ(各側 {n_a // 2}・{n_b // 2} 問)の"
                                                       "数は揺れやすく、結論には足りません"})
    # 方向ベクトル
    ds = None
    dname = js.get("direction") or name
    try:
        ds = session.direction_set(dname)
    except GneError:
        notes.append({"code": "no_direction", "text": f"方向ベクトル {dname} が見つからないので、方向ベクトルとの関係と材料の累積の確かめは出しません"})
    rS = None

    def rhat(dset, L):
        if dset is None or L not in dset.raw:
            return None
        r = np.asarray(dset.raw[L], np.float64).copy()
        if ex:
            r[ex] = 0.0
        return _unit(r) if np.any(r) else None
    if ds is not None:
        rS = rhat(ds, S) if S is not None else None
        cum = cumsum_check(z, ds.raw)
        material["cumsum"] = cum
        at = next((c for c in cum if c["layer"] == S), None)
        if at is not None:
            material["cumsum_cos_select"] = at["cos"]
            material["cumsum_cos_min"] = min((c["cos"] for c in cum if c["cos"] is not None), default=None)
            if ac["moe_layers"]:
                notes.append({"code": "cumsum_moe", "text": "MoE の層の MLP の更新が材料に無いので、累積の確かめは合いません(読み飛ばしてください)"})
            elif at["cos"] is not None and at["cos"] < CUMSUM_COS_MIN:
                notes.append({"code": "cumsum_mismatch", "text": f"選ぶ層 L{S} で、更新の累積と方向ベクトル {dname} の cos が {at['cos']:.3f} です"
                                                               f"({CUMSUM_COS_MIN} 未満)。方向ベクトルが材料と別の質問・位置で作られた疑いがあり、"
                                                               "方向ベクトルとの関係の列は当てになりません"})
        dpos = ds.meta.get("positions")
        if dpos and js.get("positions") and str(dpos) != str(js.get("positions")):
            notes.append({"code": "positions", "text": f"方向ベクトルの測る位置({dpos})と材料の測る位置({js.get('positions')})が違います"})
    # 行ごと
    hc, nr = half_stats(X, X0, X1)
    r_own = [rhat(ds, r["layer"]) for r in rows]          # 行の層の方向 r̂_l(CHiPS がその層で使う方向)
    for i, r in enumerate(rows):
        r["norm"] = float(np.linalg.norm(X[i]))
        r["half_cos"] = hc[i]
        r["noise_ratio"] = None if not np.isfinite(nr[i]) else float(nr[i])
        r["reliable"] = bool(hc[i] is not None and hc[i] >= float(min_half_cos))
        ro = r_own[i]
        r["cos_own"] = _cos(X[i], ro) if ro is not None else None
        r["cos_select"] = _cos(X[i], rS) if rS is not None else None
    n_unrel = sum(1 for r in rows if not r["reliable"])
    # 類似度の行列
    C = cosine_matrix(X)
    # 対角以外のまとめの数からは、同じ層の層全体の行と部品の行の組を除く(層全体 = 注意 + MLP なので、それだけで cos が高い)
    keep = ~np.eye(len(C), dtype=bool)
    part_of = np.array([r["part"] for r in rows])
    layer_of = np.array([r["layer"] for r in rows])
    same = (layer_of[:, None] == layer_of[None, :]) & ((part_of[:, None] == "layer") != (part_of[None, :] == "layer"))
    keep &= ~same
    off = C[keep] if keep.any() else np.zeros(1)
    adj = [float(C[i, i + 1]) for i in range(len(rows) - 1)
           if rows[i]["part"] == rows[i + 1]["part"] and rows[i + 1]["layer"] == rows[i]["layer"] + 1]
    sim = {"order": "parts", "parts": list(parts_t), "n": len(rows),
           "offdiag_abs_max": float(np.abs(off).max()), "offdiag_abs_mean": float(np.abs(off).mean()),
           "offdiag_excluded_pairs": int(same.sum() // 2),
           "adjacent": {"n": len(adj), "min": min(adj), "median": float(np.median(adj)), "max": max(adj)} if adj else None,
           "matrix": [[round(float(x), 4) for x in row] for row in C]}
    # 特異値分解
    svd, arrs = {}, {"similarity": C.astype(np.float32)}
    for mode in modes:
        s_, a_ = svd_analysis(X, X0, X1, mode, int(top_k))
        Vt = a_["Vt"]
        s_["cos_select"] = [_cos(v, rS) for v in Vt] if rS is not None else None
        s_["cos_own"] = [[(_cos(v, ro) if ro is not None else None) for v in Vt] for ro in r_own] if ds is not None else None
        svd[mode] = s_
        arrs[f"V_{mode}"] = Vt.astype(np.float32)
        arrs[f"U_{mode}"] = a_["U"].astype(np.float32)
        arrs[f"s_{mode}"] = a_["s"].astype(np.float32)
    # ループの対
    loop = loop_pairs(z, lay, ex)
    # 2 本目の方向の候補との比べ
    cmp_names = compare_names(session.store.direction_dir(session.meta.model_id), name) if compare is None else \
        [x.strip() for x in (compare.split(",") if isinstance(compare, str) else compare) if str(x).strip()]
    comps = []
    for cn in cmp_names:
        try:
            dD = session.direction_set(cn)
        except GneError as e:
            if compare is not None:
                raise GneError("FILE_NOT_FOUND", f"比べる方向ベクトル {cn} がありません") from e
            continue
        zD = None
        pD = hd / f"{cn}.writes.npz"
        if pD.exists():
            zl = np.load(pD)
            zD = {k: zl[k] for k in zl.files if k.startswith("upd_") or k == "layer_index"}
        posD = {int(L): i for i, L in enumerate(zD["layer_index"])} if zD is not None else {}
        crow = []
        for i, r in enumerate(rows):
            rd = rhat(dD, r["layer"])
            x = {"i": i, "cos_direction": None, "cos_update": None, "cos_material": None}
            if rd is not None:
                ro = r_own[i]
                x["cos_direction"] = _cos(rd, ro) if ro is not None else None
                x["cos_update"] = _cos(rd, X[i])
                for mode in modes:
                    x[f"cos_v_{mode}"] = [_cos(rd, v) for v in arrs[f"V_{mode}"].astype(np.float64)]
            if zD is not None and r["layer"] in posD:
                v = part_vectors(zD, r["part"])[posD[r["layer"]]].copy()
                if ex:
                    v[ex] = 0.0
                x["cos_material"] = _cos(v, X[i])
            crow.append(x)
        rdS = rhat(dD, S) if S is not None else None
        summ = {"cos_direction_select": _cos(rdS, rS) if rdS is not None and rS is not None else None}
        if rdS is not None:
            for mode in modes:
                cs_ = [_cos(rdS, v) for v in arrs[f"V_{mode}"].astype(np.float64)]
                j = int(np.argmax([abs(c or 0.0) for c in cs_]))
                summ[f"best_v_{mode}"] = {"k": j + 1, "cos": cs_[j]}
        comps.append({"name": cn, "kind": "refine" if compare is None else "given", "material": pD.name if zD is not None else None,
                      "select": summ, "rows": crow})
    if comps:
        notes.append({"code": "compare_meaning", "text": "heads refine の方向は、組を両側に当てた状態で、残りの質問だけを A にした残差の平均の差です。"
                                                         "特異値分解の向きは層ごとの更新の差なので、意味の違う量どうしを、同じ残差の空間の向きとして比べています"})
    # 質問ごとの更新があるとき: ラベル入れ替えの基準(§2.5)と、同じ層の中の質問に沿った分解(§2.6)
    null_meta, within, wsum = None, None, None
    if qd is not None:
        pos = {int(L): i for i, L in enumerate(z["layer_index"])}
        QA, QB = q_rows(qd, rows, pos, ex)
        paired = qd["paired"]
        dq = (QA - QB).mean(0) if paired else QA.mean(0) - QB.mean(0)
        nX = np.linalg.norm(X, axis=1)
        material["q"]["row_err_max"] = float(np.max(np.linalg.norm(dq - X, axis=1) / np.maximum(nX, 1e-30)))
        if int(permutations) > 0:
            nl = label_null(QA, QB, paired, int(permutations), int(seed), int(top_k))
            null_meta = {k: nl[k] for k in ("scheme", "n_perm", "seed")}
            for i, r in enumerate(rows):
                v = nl["norms"][:, i]
                r["norm_null95"] = float(np.percentile(v, 95))
                r["norm_p"] = float((np.sum(v >= r["norm"]) + 1) / (len(v) + 1))
                r["half_cos_null95"] = float(np.percentile(nl["half_cos"][:, i], 95))
            c95 = np.percentile(nl["cos_abs"], 95, axis=0)
            arrs["cos_null95"] = c95.astype(np.float32)
            up = np.triu(keep, 1)
            sim["above_null95"] = {"n": int((np.abs(C)[up] > c95[up]).sum()), "of": int(up.sum())}
            for mode in svd:
                svd[mode]["null95"] = [float(x) for x in np.percentile(nl["explained"][mode], 95, axis=0)[:svd[mode]["k"]]]
        kw = max(1, min(int(top_k), QA.shape[0] // 2 - 1))
        texts = _questions(qd)
        within = []
        for i, r in enumerate(rows):
            w = within_row(QA[:, i], QB[:, i], paired, r_own[i], rS, kw)
            within.append({"i": i, "label": r["label"], **w})
        arrs["within_q_cos"] = np.array([w["q_cos"] for w in within], np.float32)
        lr = [w for w, r in zip(within, rows, strict=True) if r["part"] == "layer" and r["reliable"]] or within
        mq = np.mean([w["q_cos"] for w in lr], axis=0)
        order = [int(j) for j in np.argsort(mq)]
        wsum = {"paired": paired, "k": kw, "rows_used": len(lr),
                "q_mean_cos": [float(x) for x in mq],
                "lowest": [{"q": qd["index_a"][j], "mean_cos": float(mq[j]), "text": texts.get(qd["index_a"][j])} for j in order[:5]],
                "highest": [{"q": qd["index_a"][j], "mean_cos": float(mq[j]), "text": texts.get(qd["index_a"][j])} for j in order[::-1][:3]]}
        if QA.shape[0] < 2 * FEW_QUESTIONS:
            notes.append({"code": "within_few", "text": f"A 側の質問が {QA.shape[0]} 問なので、質問に沿った分解の上位の向きは雑音で決まりやすく、"
                                                        "安定度(半分どうしの |cos|)と並べてしか読めません"})
    if n_unrel:
        notes.append({"code": "unreliable", "text": f"{n_unrel} 行は、半分どうしの cos が {min_half_cos} 未満です(信用しない印)"})
    if "layer" in parts_t and len(parts_t) > 1:
        notes.append({"code": "layer_and_parts", "text": "層全体の行は注意と MLP の行の和なので、同じ層の層全体と部品の行どうしの cos は"
                                                         "それだけで高くなります(対角以外の |cos| の平均・最大からは、その組を除きました。"
                                                         "行列には残っています)"})
    if ex:
        notes.append({"code": "excluded_dims", "text": f"次元 {', '.join(map(str, ex))} を 0 にしてから計算しました(行ごとの上位の次元は、除く前の値)"})
    out = {"name": out_name, "material": material, "arch": {k: ac[k] for k in ("arch", "checked", "moe_layers")},
           "settings": {"parts": list(parts_t), "normalize": list(modes), "top_k": int(top_k), "layers": lay,
                        "min_half_cos": float(min_half_cos), "exclude_dims": ex, "compare": cmp_names,
                        "permutations": int(permutations), "seed": int(seed)},
           "rows": rows, "skipped_rows": skipped, "n_unreliable": n_unrel, "similarity": sim, "svd": svd, "loop": loop,
           "compare": comps, "null": null_meta, "within": within, "within_summary": wsum, "notes": notes, "reading": READING}
    hd.mkdir(parents=True, exist_ok=True)
    pj, pz = hd / f"{out_name}.geometry.json", hd / f"{out_name}.geometry.npz"
    np.savez_compressed(pz, **arrs, row_layer=np.array([r["layer"] for r in rows], np.int32),
                        row_part=np.array([r["part"] for r in rows]))
    out["seconds"] = time.perf_counter() - t0
    out["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out["path"], out["npz"] = str(pj), str(pz)
    pj.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# --- 人が読む表示(CLI) -------------------------------------------------------------------------------------

def _f(x, fmt: str) -> str:
    return "—" if x is None else format(x, fmt)


def _width(s: str) -> int:
    """端末での表示の幅(全角・幅広の文字は 2、結合文字は 0、ほかは 1)。"""
    return sum(0 if unicodedata.combining(ch) else 2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def _table(header: list[str], body: list[list[str]], left: int = 1, indent: str = "  ") -> list[str]:
    """表示の幅で桁をそろえた表。最初の left 列は左寄せ、ほかは右寄せ。"""
    cells = [[str(c) for c in row] for row in [header, *body]]
    n = max(len(r) for r in cells)
    w = [max(_width(r[j]) for r in cells if j < len(r)) for j in range(n)]

    def pad(c, j):
        sp = " " * (w[j] - _width(c))
        return c + sp if j < left else sp + c
    return [indent + " ".join(pad(c, j) for j, c in enumerate(r)).rstrip() for r in cells]


def human_lines(d: dict) -> list[str]:
    m, st = d["material"], d["settings"]
    S = m.get("select_layer")
    lines = [f"{d['path']}(材料 {m['name']}、方向 {m.get('direction')}、選ぶ層 L{S}、A {m['n_a']} 問・B {m['n_b']} 問、"
             f"行 {d['similarity']['n']}({'・'.join(PART_JA[p] for p in st['parts'])}、部品ごとにまとめて層の順)、{d['seconds']:.1f} 秒)",
             f"構造: {d['arch']['arch']}({'確かめた構造' if d['arch']['checked'] else '確かめていない構造'})"]
    ck = [f"半分ずつの平均と全体の差 最大 {m['halves_mean_err_max']:.1e}"]
    if m.get("cumsum_cos_select") is not None:
        ck.append(f"更新の累積と方向ベクトルの cos: 選ぶ層 {m['cumsum_cos_select']:.3f}(全部の層の最小 {_f(m.get('cumsum_cos_min'), '.3f')})")
    if m.get("check_layer_rel_median") is not None:
        ck.append(f"部品から組み立てた更新と残差の差 中央値 {m['check_layer_rel_median']:.2%}・最大 {m['check_layer_rel_max']:.2%}")
    lines.append("材料の確かめ: " + " / ".join(ck))
    lines.append(f"行ごと(cos r̂_l = その層の方向との cos、cos r̂_S = 選ぶ層の方向との cos。2 乗がその方向に沿う割合。"
                 f"× = 半分どうしの cos が {st['min_half_cos']} 未満):")
    has_null = bool(d.get("null"))
    if has_null:
        lines.append(f"  (大きさ÷基準 = 大きさ ÷ A と B のラベルを入れ替えて作った差の大きさの 95 パーセンタイル({d['null']['n_perm']} 回)。"
                     "1 以下なら、質問ごとのばらつきだけでも出る大きさ)")
    lines += _table(["行", "種類", "回", "大きさ"] + (["大きさ÷基準"] if has_null else []) +
                    ["半分cos", "雑音比", "cos r̂_l", "cos r̂_S", "上位5次元", "印"],
                    [[r["label"], MIXER_JA.get(r["mixer"], r["mixer"]), str(r["visit"] or ""), f"{r['norm']:.3g}"] +
                     ([_f(r["norm"] / r["norm_null95"] if r.get("norm_null95") else None, ".2f")] if has_null else []) +
                     [_f(r["half_cos"], ".2f"), _f(r["noise_ratio"], ".2f"), _f(r["cos_own"], "+.2f"), _f(r["cos_select"], "+.2f"),
                      _f(r["top_dims"]["share"], ".2f"), "" if r["reliable"] else "×"] for r in d["rows"]], left=2)
    sm = d["similarity"]
    line = f"類似度の行列: 対角以外の |cos| 平均 {sm['offdiag_abs_mean']:.3f}・最大 {sm['offdiag_abs_max']:.3f}" + \
        (f"(同じ層の層全体と部品の組 {sm['offdiag_excluded_pairs']} 個を除く)" if sm.get("offdiag_excluded_pairs") else "")
    if sm.get("adjacent"):
        a = sm["adjacent"]
        line += f"。隣の層どうし(同じ部品)の cos: 最小 {a['min']:+.2f}・中央値 {a['median']:+.2f}・最大 {a['max']:+.2f}"
    if sm.get("above_null95"):
        line += f"。ラベル入れ替えの 95% を超える |cos| の組: {sm['above_null95']['n']} / {sm['above_null95']['of']}"
    lines.append(line + f"(行列は {Path(d['npz']).name} と JSON)")
    for mode, s in d["svd"].items():
        lines.append(f"特異値分解({'行の大きさをそろえる' if mode == 'rows' else '行の大きさをそろえない'}。基準は、データの全体に対する割合):")
        cs = s.get("cos_select") or [None] * s["k"]
        nn = s.get("null95")
        lines += _table(["k", "寄与率"] + (["入れ替え95%"] if nn else []) + ["累積", "直交の基準", "雑音の基準", "半分で求めた向きの説明率",
                                                                           "半分どうしの重なり", "cos r̂_S"],
                        [[str(j + 1), f"{s['explained'][j]:.3f}"] + ([f"{nn[j]:.3f}"] if nn else []) +
                         [f"{s['cumulative'][j]:.3f}", f"{s['baseline_orthogonal'][j]:.3f}",
                          f"{s['baseline_noise'][j]:.3f}", f"{s['half_cross'][j]:.3f}", f"{s['half_overlap'][j]:.3f}", _f(cs[j], "+.2f")]
                         for j in range(s["k"])], left=0)
        if nn:
            lines.append("  (入れ替え95% = ラベルを入れ替えた差の寄与率の 95 パーセンタイル。寄与率はその差自身の全体に対する割合)")
        lines.append(f"  (雑音の目安の全体 ‖(X_h0 − X_h1)/2‖² ÷ ‖X‖² = {_f(s['noise_total'], '.3f')}。半分で求めた向きの説明率と重なりは累積)")
    lp = d.get("loop")
    if lp:
        lines.append(f"同じ重みの 1 回目と 2 回目(段のずれ {lp['shift']}。cos の右のかっこは半分ずつで測った値):")
        def c(x):
            return "—" if not x else f"{x['cos']:+.2f}({_f(x['cos_halves'][0], '+.2f')},{_f(x['cos_halves'][1], '+.2f')})"
        lines += _table(["重みの層", "1回目", "2回目", "層全体 cos", "注意 cos", "MLP cos", "大きさ(層全体、1回目 / 2回目)"],
                        [[str(p_["stored"]), str(p_["first"]), str(p_["second"]), c(p_.get("layer")), c(p_.get("attn")), c(p_.get("mlp")),
                          f"{p_['layer']['norm_first']:.3g} / {p_['layer']['norm_second']:.3g}" if p_.get("layer") else "—"]
                         for p_ in lp["pairs"]], left=0)
    w, ws = d.get("within"), d.get("within_summary")
    if w:
        lines.append(f"同じ層の中の質問ごとの差の分解({'対になった対照集合: e_q = u_q^A − u_q^B' if ws['paired'] else '側ごとに別の質問: e_q = u_q^A − (B 側の平均)'}。"
                     "割合は Σ‖e_q‖² に対して。取り残し = 平均の向き d̂ に直交する分):")
        head = ["行", "d̂ で説明", "r̂_l で説明", "r̂_S で説明", "上位1本", "取り残し", "取り残し1番目", "安定度(1番目)"] +             ([] if ws["paired"] else ["A/B の広がりの比(1番目)"])
        body = []
        for x in w:
            row = [x["label"], _f(x["share_mean"], ".2f"), _f(x["share_own"], ".2f"), _f(x["share_select"], ".2f"),
                   _f((x.get("explained") or [None])[0], ".2f"), _f(x["leftover_total"], ".2f"),
                   _f((x.get("leftover_explained") or [None])[0], ".3f"), _f((x.get("leftover_stability") or [None])[0], ".2f")]
            if not ws["paired"]:
                row.append(_f((x.get("leftover_ab_ratio") or [None])[0], ".2f"))
            body.append(row)
        lines += _table(head, body, left=1)
        lines.append("  (安定度 = 偶数番目・奇数番目の質問で別々に求めた取り残しの向きどうしの |cos|。A/B の広がりの比は、片方の半分で"
                     "求めた向きに沿った A 側のもう半分の広がり ÷ B 側の質問どうしの広がり。1 前後ならふつうの質問どうしの違いと同じ程度)")
        lo = [f"{x['mean_cos']:+.2f} {x['text'] or '#' + str(x['q'])}" for x in ws["lowest"]]
        lines.append(f"平均の向き d̂ との cos が低い A 側の質問(信用できる層全体の行 {ws['rows_used']} 行の平均): " + " | ".join(lo))
    for cp in d.get("compare") or []:
        sel = cp["select"]
        parts = [f"選ぶ層での方向ベクトルどうしの cos {_f(sel.get('cos_direction_select'), '+.3f')}"]
        for mode in d["svd"]:
            b = sel.get(f"best_v_{mode}")
            if b:
                parts.append(f"{'そろえる' if mode == 'rows' else 'そろえない'}版の上位の向きとの cos の絶対値の最大: V{b['k']} {_f(b['cos'], '+.3f')}")
        lines.append(f"比べる方向 {cp['name']}" + (f"(材料 {cp['material']} あり)" if cp.get("material") else "") + ": " + "、".join(parts))
    for n in d.get("notes") or []:
        lines.append("注意: " + n["text"])
    lines.append("読み取りの注意:")
    lines += [f"  - {x}" for x in d.get("reading") or []]
    return lines
