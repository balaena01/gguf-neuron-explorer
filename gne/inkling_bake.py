"""Inkling 2: 専門家の down の重み(ffn_down_exps)に、元の型・元の大きさのまま射影の編集を書き込む(焼き込みの keep / auto)。

このモデルの ffn_down_exps は NVFP4 か Q2_0(別の量子化には MXFP4・IQ4_NL もある)。llama.cpp の量子化で書き戻すと、NVFP4 は
小さな編集の 3〜5 割が丸めで消え、Q2_0 は(この GGUF のブロックの倍率に負の値があり、llama.cpp の量子化はそれを作らないので)
編集しない値の半分ほどが変わり、IQ4_NL は MLP 全体の部品で編集が半分ほどしか残らない。そこで量子化し直さず、元のコードと
ブロックの倍率をそのまま残し、編集する列(ニューロン)のコードだけを 1 段ずつ動かして、LoRA が足す変化に、方向が張る空間の
中で合わせる(射影を意識した丸め。いくつかの方向は列ごとにまとめて解く。下の edit_columns):
- 行列の 1 行(出力の 1 次元)は 1152 個のニューロンで、量子化のブロックは行に沿って並ぶ(Q2_0 は 64 個、NVFP4 は 64 個で倍率は
  16 個ごと、MXFP4・IQ4_NL は 32 個)。1 つの列は各行のブロックに 1 個ずつ入るので、列のコードを動かしても、ほかの列の値は
  まったく変わらない
- 1 段の大きさは Q2_0 が |d|(値は (q − 1)·d、q = 0〜3)、NVFP4・MXFP4 は隣の値まで(値は {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12} × 倍率。
  NVFP4 は 16 個ごとの UE4M3 の倍率、MXFP4 は 32 個ごとの E8M0 の倍率 2^(e − 128))、IQ4_NL は表の隣の値まで(値は
  {−127, −104, …, 113} × fp16 の倍率 d、d は負もある)
- 方向に沿わない変化(動かしたコードの、方向の空間の外の成分)は残る。その大きさは結果の off_over_removed(÷ 消した成分)"""
from __future__ import annotations

import os

import numpy as np

KV = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float64)
_LEVELS = sorted(set(KV.tolist()))
_CODE = {v: int(np.where(KV == v)[0][0]) for v in _LEVELS}
UP = np.array([_CODE[_LEVELS[_LEVELS.index(KV[c]) + 1]] if _LEVELS.index(KV[c]) + 1 < len(_LEVELS) else -1 for c in range(16)])
DN = np.array([_CODE[_LEVELS[_LEVELS.index(KV[c]) - 1]] if _LEVELS.index(KV[c]) > 0 else -1 for c in range(16)])
KV4 = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float64)   # IQ4_NL の表
TYPES = ("Q2_0", "NVFP4", "MXFP4", "IQ4_NL")


def _ue4m3(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.int64)
    exp, man = (x >> 3) & 0xF, x & 0x7
    raw = np.where(exp == 0, np.ldexp(man.astype(np.float64), -9), np.ldexp(1.0 + man / 8.0, exp - 7))
    return np.where((x == 0) | (x == 0x7F), 0.0, raw * 0.5)


def decode(raw: np.ndarray, typ: str, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1 つの専門家のバイト列 (rows, 行のバイト数) → コード (rows, cols)、値 1 単位の倍率 (rows, cols)、値 (rows, cols)。"""
    nb = cols // 64
    if typ == "Q2_0":
        b = raw.reshape(rows, nb, 18)
        d = b[:, :, :2].copy().view(np.float16).astype(np.float64)[:, :, 0]
        qs = b[:, :, 2:]
        q = np.stack([(qs >> (2 * k)) & 3 for k in range(4)], axis=-1).reshape(rows, cols).astype(np.int64)
        sc = np.repeat(d, 64, axis=1)
        return q, sc, (q - 1) * sc
    if typ == "IQ4_NL":                                        # 32 個ごと: d(fp16)2 バイト + 4 ビット × 32(表の番号)
        b = raw.reshape(rows, cols // 32, 18)
        d = b[:, :, :2].copy().view(np.float16).astype(np.float64)[:, :, 0]
        qs = b[:, :, 2:]
        q = np.concatenate([qs & 0x0F, qs >> 4], axis=-1).reshape(rows, cols).astype(np.int64)
        sc = np.repeat(d, 32, axis=1)
        return q, sc, KV4[q] * sc
    if typ == "MXFP4":                                         # 32 個ごと: e(E8M0)1 バイト + 4 ビット × 32
        nb32 = cols // 32
        b = raw.reshape(rows, nb32, 17)
        d = np.ldexp(1.0, b[:, :, 0].astype(np.int64) - 128)
        qs = b[:, :, 1:]
        q = np.concatenate([qs & 0x0F, qs >> 4], axis=-1).reshape(rows, cols).astype(np.int64)
        sc = np.repeat(d, 32, axis=1)
        return q, sc, KV[q] * sc
    if typ != "NVFP4":
        raise ValueError(typ)
    b = raw.reshape(rows, nb, 36)
    d = _ue4m3(b[:, :, :4])
    qs = b[:, :, 4:].reshape(rows, nb, 4, 8)
    q = np.concatenate([qs & 0x0F, qs >> 4], axis=-1).reshape(rows, cols).astype(np.int64)
    sc = np.repeat(d.reshape(rows, nb * 4), 16, axis=1)
    return q, sc, KV[q] * sc


def encode(raw: np.ndarray, typ: str, rows: int, cols: int, q: np.ndarray) -> np.ndarray:
    """コード q を元のバイト列に書き戻す(ブロックの倍率はそのまま)。"""
    nb = cols // 64
    out = np.array(raw, dtype=np.uint8, copy=True).reshape(rows, nb, -1)
    if typ == "Q2_0":
        qq = q.reshape(rows, nb, 16, 4).astype(np.uint8)
        out[:, :, 2:] = qq[..., 0] | (qq[..., 1] << 2) | (qq[..., 2] << 4) | (qq[..., 3] << 6)
    elif typ in ("MXFP4", "IQ4_NL"):
        nb32 = cols // 32
        out = np.array(raw, dtype=np.uint8, copy=True).reshape(rows, nb32, 17 if typ == "MXFP4" else 18)
        qq = q.reshape(rows, nb32, 32).astype(np.uint8)
        out[:, :, (1 if typ == "MXFP4" else 2):] = qq[..., :16] | (qq[..., 16:] << 4)
    else:
        qq = q.reshape(rows, nb, 4, 16).astype(np.uint8)
        out[:, :, 4:] = (qq[..., :8] | (qq[..., 8:] << 4)).reshape(rows, nb, 32)
    return out.reshape(np.shape(raw))


def _moves(q: np.ndarray, sc: np.ndarray, typ: str, up: bool) -> tuple[np.ndarray, np.ndarray]:
    """値を 1 段上げる(up)/ 下げるときの値の変化と新しいコード(動かせないところは変化 0・コードそのまま)。"""
    if typ == "Q2_0":
        dq = np.where(sc >= 0, 1, -1) * (1 if up else -1)
        nq = q + dq
        ok = (nq >= 0) & (nq <= 3) & (sc != 0)
        return np.where(ok, np.abs(sc) * (1 if up else -1), 0.0), np.where(ok, nq, q)
    if typ == "IQ4_NL":                                       # 表の隣の番号。d が負なら番号を上げると値は下がる
        a, b = np.minimum(q + 1, 15), np.maximum(q - 1, 0)
        da = np.where(q < 15, (KV4[a] - KV4[q]) * sc, 0.0)
        db = np.where(q > 0, (KV4[b] - KV4[q]) * sc, 0.0)
        pa = da > 0 if up else da < 0
        pb = db > 0 if up else db < 0
        return np.where(pa, da, np.where(pb, db, 0.0)), np.where(pa, a, np.where(pb, b, q))
    nq = (UP if up else DN)[q]
    ok = (nq >= 0) & (sc > 0)
    nq = np.where(ok, nq, q)
    return np.where(ok, (KV[nq] - KV[q]) * sc, 0.0), nq


def edit_slab(raw: np.ndarray, typ: str, u: np.ndarray, delta_by_col: dict[int, float], rows: int, cols: int) -> dict:
    """1 つの専門家のバイト列に、列ごとの uᵀw の変化 delta_by_col を書き込む(方向 1 つの edit_columns)。戻り値: 新しいバイト列と、
    確かめの数(目標の uᵀw の変化・実際の変化・方向に沿わない変化の大きさ・動かしたコードの数)。ほかの列のコードはそのまま。"""
    cs = np.array(sorted(delta_by_col), np.int64)
    dl = np.array([delta_by_col[int(c)] for c in cs], np.float64)
    u = np.asarray(u, np.float64)
    r = edit_columns(raw, typ, u[:, None], np.outer(u, dl) / max(float(u @ u), 1e-300), cs, rows, cols)
    return {"raw": r["raw"], "want": dl, "along": r["got"][0], "off": r["off_col"] * r["t_col"], "flips": r["codes_col"],
            "w_norm": r["w_norm"]}


# --- いくつかの方向をまとめて(列ごとに) ------------------------------------------------------------------------------
# 列 w(1 つの専門家の 1 つのニューロンの書き込み)の目標は、LoRA が足す変化 T(いくつかの方向の部品の和、方向は直交しない)。
# 方向が張る空間 span(U) の中で、T との差 r が小さくなるように、コードを 1 段ずつ動かす。1 段(行 i、値の変化 dv)で ‖r‖² は
# gain = 2·dv·(Q_i·r) − dv²·|Q_i|² 下がり(Q は span(U) の正規直交の基底、Q_i はその行 i)、方向に沿わない変化のエネルギーは
# cost = (D_i + dv)² − D_i² 増える(D はそれまでの変化)。gain / cost の大きい順に、残りの半分ほどを 1 回で動かすことを、
# 残りが ‖Qᵀ T‖ の JOINT_TOL 以内になるまで繰り返す。方向を 1 つずつ続けて書くと、後の方向で動かしたコードが先の方向の成分を
# ずらし(2 方向の全体の部品で 15%・49%)、方向に沿わない変化も重なった。同じ量を少ない数のコードで動かすより(|u_i| の順)、
# 小さな段を多めに動かすほうが、方向に沿わない変化のエネルギーは小さい(方向 1 つでも消した成分の 2.4 倍 → 1.3 倍、NVFP4)。

# auto で MLP 全体の部品を受ける専門家のテンソルを元の大きさのまま書く上限: 方向に沿わない変化 ÷ 消した成分(全部の編集する列で
# まとめた値)。方向 1 つの元の大きさの焼き込みで、GPU の比べで LoRA と合った組(層 9 の NVFP4・層 22 の Q2_0 の MLP 全体、強さ 1)
# の値(以前のやり方で 1.91・2.22)のうち大きいほう。これより大きいテンソルは Q8_0 にする
OFF_MAX = 2.25
JOINT_TOL = 0.01      # span(U) の中の残り ÷ 目標
WORKERS = max(1, min(6, (os.cpu_count() or 2) - 1))   # 専門家を並べて計算するスレッドの数
CANDIDATES = 512      # 列ごとに候補にする行の数(|Q_i| ÷ 1 段の大きさ の大きい順)。候補で届かない列は全部の行で続ける
MAX_ROUNDS = 80
HALF = 0.5            # 1 回で動かす量(残りに沿った分)の上限 ÷ 残り


def basis(U: np.ndarray) -> np.ndarray:
    """U(行 × 方向)が張る空間の正規直交の基底(同じ向きの方向は 1 つにまとめる)。"""
    Qb, Rr = np.linalg.qr(np.asarray(U, np.float64))
    d = np.abs(np.diag(Rr))
    return Qb[:, d > 1e-8 * max(float(d.max()) if d.size else 0.0, 1e-300)]


def _greedy(q, sc, typ, Qg, nq2, D, res, tn, tol):
    """候補の行(P, m)の上で、gain / cost の順にコードを動かす。q・D・res をその場で書き換える。"""
    active = np.ones(q.shape[1], bool)
    for _ in range(MAX_ROUNDS):
        en = np.linalg.norm(res, axis=0)
        active &= en > tol * np.maximum(tn, 1e-300)
        cols = np.nonzero(active)[0]
        if not len(cols):
            break
        qa, sa, Da, ra = q[:, cols], sc[:, cols], D[:, cols], res[:, cols]
        Qa = Qg[:, cols]
        proj = np.einsum("pmr,rm->pm", Qa, ra.astype(np.float32), optimize=True)
        n2 = nq2[:, cols]
        best = None
        for up in (True, False):
            dv, nq = _moves(qa, sa, typ, up)
            dv32 = dv.astype(np.float32)
            gain = 2 * dv32 * proj - dv32 * dv32 * n2
            cost = (Da + dv32) ** 2 - Da ** 2
            score = np.where(gain > 0, np.where(cost <= 0, np.float32(np.inf), gain / np.maximum(cost, np.float32(1e-30))),
                             np.float32(-1))
            if best is None:
                best = [score, dv, nq]
            else:
                b = score > best[0]
                best = [np.where(b, score, best[0]), np.where(b, dv, best[1]), np.where(b, nq, best[2])]
        score, dv, nq = best
        order = np.argsort(-score, axis=0, kind="stable")
        so = np.take_along_axis(score, order, 0)
        ok = so > 0
        lin = np.take_along_axis(dv * proj, order, 0) / np.maximum(en[cols], 1e-300)[None, :]
        cum = np.cumsum(np.where(ok, lin, 0.0), 0)
        allow = ok & ((cum <= HALF * en[cols][None, :]) | (np.arange(len(so))[:, None] == 0))
        # 並べた順に足したときの残り(span の中のベクトル)。許す長さの中で残りがいちばん小さくなるところまで動かす
        # (段ごとの gain は今の残りに対する値なので、まとめて動かすと span の中の別の向きにずれることがある)
        step = np.take_along_axis(dv[..., None] * Qa.astype(np.float64), order[..., None], 0)
        rc = ra.T[None, :, :] - np.cumsum(np.where(ok[..., None], step, 0.0), 0)
        nr = np.where(allow, np.sum(rc * rc, -1), np.inf)
        last = np.argmin(nr, 0)
        better = nr[last, np.arange(len(cols))] < en[cols] ** 2
        take = (np.arange(len(so))[:, None] <= last[None, :]) & better[None, :] & ok
        sel = np.zeros_like(take)
        np.put_along_axis(sel, order, take, 0)
        mv = np.where(sel, dv, 0.0)
        q[:, cols] = np.where(sel, nq, qa)
        D[:, cols] = Da + mv
        res[:, cols] = ra - np.einsum("pmr,pm->rm", Qa.astype(np.float64), mv, optimize=True)
        active[cols[~better]] = False                   # 残りを減らせる動かし方がもう無い列


def solve_columns(q0: np.ndarray, sc: np.ndarray, typ: str, U: np.ndarray, T: np.ndarray, *, tol: float = JOINT_TOL,
                  P: int = CANDIDATES) -> tuple[np.ndarray, np.ndarray]:
    """列(q0・sc・T は (行, 列))のコードを動かして、span(U) の中で変化 D を T に合わせる。戻り値: 新しいコードと D。"""
    Qb = basis(U)
    R, m = T.shape
    t = Qb.T @ T
    tn = np.linalg.norm(t, axis=0)
    rn = np.linalg.norm(Qb, axis=1)
    vu, _ = _moves(q0, sc, typ, True)
    vd, _ = _moves(q0, sc, typ, False)
    step = np.minimum(np.where(vu != 0, np.abs(vu), np.inf), np.where(vd != 0, np.abs(vd), np.inf))
    fin = np.isfinite(step)
    key = np.where(fin, rn[:, None] / np.where(fin, step, 1.0), -1.0)
    P = min(P, R)
    idx = np.argpartition(-key, P - 1, axis=0)[:P] if P < R else np.broadcast_to(np.arange(R)[:, None], (R, m)).copy()
    qc = np.take_along_axis(q0, idx, 0).copy()
    Dc = np.zeros((P, m))
    res = t.copy()
    _greedy(qc, np.take_along_axis(sc, idx, 0), typ, Qb[idx].astype(np.float32), (rn[idx] ** 2).astype(np.float32), Dc, res, tn, tol)
    q = q0.copy()
    np.put_along_axis(q, idx, qc, 0)
    D = np.zeros((R, m))
    np.put_along_axis(D, idx, Dc, 0)
    left = np.nonzero(np.linalg.norm(res, axis=0) > tol * np.maximum(tn, 1e-300))[0]
    if len(left) and P < R:                              # 候補で届かない列: 全部の行で続ける
        ql, Dl, rl = q[:, left].copy(), D[:, left].copy(), res[:, left].copy()
        _greedy(ql, sc[:, left], typ, np.broadcast_to(Qb.astype(np.float32)[:, None, :], (R, len(left), Qb.shape[1])),
                np.broadcast_to((rn ** 2).astype(np.float32)[:, None], (R, len(left))), Dl, rl, tn[left], tol)
        q[:, left], D[:, left] = ql, Dl
    return q, D


def edit_columns(raw: np.ndarray, typ: str, U: np.ndarray, T: np.ndarray, cs, rows: int, cols: int) -> dict:
    """1 つの専門家のバイト列の列 cs に、LoRA の変化 T((行, len(cs)))を、span(U) の中で合わせて書く(元のコードと倍率のまま、
    ほかの列はバイトのまま)。戻り値: 新しいバイト列、方向ごとの目標と実際(Uᵀ T・Uᵀ D、D は書いた変化)、方向に沿わない変化
    (span(U) の外)のエネルギーと列ごとの大きさ ÷ ‖T の列‖、動かしたコードの数。"""
    q, sc, W = decode(raw, typ, rows, cols)
    cs = np.asarray(cs, np.int64)
    qn, D = solve_columns(q[:, cs], sc[:, cs], typ, U, T)
    qq = q.copy()
    qq[:, cs] = qn
    new = encode(raw, typ, rows, cols, qq)
    Qb = basis(U)
    off = D - Qb @ (Qb.T @ D)
    tc = np.linalg.norm(T, axis=0)
    return {"raw": new, "want": U.T @ T, "got": U.T @ D, "off2": float(np.sum(off * off)), "T2": float(np.sum(T * T)),
            "off_col": np.linalg.norm(off, axis=0) / np.maximum(tc, 1e-300), "t_col": tc, "codes_col": np.sum(qn != q[:, cs], axis=0),
            "w_norm": np.linalg.norm(W[:, cs], axis=0)}

