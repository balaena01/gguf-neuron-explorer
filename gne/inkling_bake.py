"""Inkling 2: 専門家の down の重み(ffn_down_exps)に、元の型・元の大きさのまま射影の編集を書き込む(焼き込みの keep / auto)。

このモデルの ffn_down_exps は NVFP4 か Q2_0。llama.cpp の量子化で書き戻すと、NVFP4 は小さな編集の 3〜5 割が丸めで消え、
Q2_0 は(この GGUF のブロックの倍率に負の値があり、llama.cpp の量子化はそれを作らないので)編集しない値の半分ほどが変わる。
そこで量子化し直さず、元のコードとブロックの倍率をそのまま残し、編集する列(ニューロン)ごとに、方向の成分 uᵀw が目標
(1 − s)·uᵀw になるように、いちばん少ない数のコードを 1 段ずつ動かす(射影を意識した丸め):
- 行列の 1 行(出力の 1 次元)は 1152 個のニューロンで、量子化のブロックは行に沿って並ぶ(Q2_0 は 64 個、NVFP4 は 64 個で倍率は
  16 個ごと)。1 つの列は各行のブロックに 1 個ずつ入るので、列のコードを動かしても、ほかの列の値はまったく変わらない
- 動かすのは |u_i| の大きい行から(1 段で uᵀw をいちばん大きく動かし、方向に沿わない変化が相対的に小さい)。最後に、残りに
  いちばん合う 1 段を足す
- 1 段の大きさは Q2_0 が |d|(値は (q − 1)·d、q = 0〜3)、NVFP4・MXFP4 は隣の値まで(値は {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12} × 倍率。
  NVFP4 は 16 個ごとの UE4M3 の倍率、MXFP4 は 32 個ごとの E8M0 の倍率 2^(e − 128))。MXFP4 は別の量子化(UD-Q2_K_XL の層 20 など)に
  焼き込むときに出てくる
方向に沿わない変化(動かしたコードの、u に直交する成分)は消えた成分の 2 倍ほど残る(列 1 本あたり。docs/inkling2.md)。"""
from __future__ import annotations

import numpy as np

KV = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float64)
_LEVELS = sorted(set(KV.tolist()))
_CODE = {v: int(np.where(KV == v)[0][0]) for v in _LEVELS}
UP = np.array([_CODE[_LEVELS[_LEVELS.index(KV[c]) + 1]] if _LEVELS.index(KV[c]) + 1 < len(_LEVELS) else -1 for c in range(16)])
DN = np.array([_CODE[_LEVELS[_LEVELS.index(KV[c]) - 1]] if _LEVELS.index(KV[c]) > 0 else -1 for c in range(16)])
TYPES = ("Q2_0", "NVFP4", "MXFP4")


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
    elif typ == "MXFP4":
        nb32 = cols // 32
        out = np.array(raw, dtype=np.uint8, copy=True).reshape(rows, nb32, 17)
        qq = q.reshape(rows, nb32, 32).astype(np.uint8)
        out[:, :, 1:] = qq[..., :16] | (qq[..., 16:] << 4)
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
    nq = (UP if up else DN)[q]
    ok = (nq >= 0) & (sc > 0)
    nq = np.where(ok, nq, q)
    return np.where(ok, (KV[nq] - KV[q]) * sc, 0.0), nq


def round_columns(q: np.ndarray, sc: np.ndarray, typ: str, u: np.ndarray, cols: np.ndarray, delta: np.ndarray) -> dict:
    """列 cols の uᵀw を delta だけ動かす、いちばん少ないコードの動かし方(列ごと、まとめて計算)。戻り値: 新しいコード・列ごとの
    動かした数・届かなかった残り(uᵀw の目標との差)。"""
    order = np.argsort(-np.abs(u))
    uo = u[order]
    qo, so = q[order][:, cols], sc[order][:, cols]                            # (rows, k)
    need = np.asarray(delta, np.float64)
    vu, nu_ = _moves(qo, so, typ, True)
    vd, nd = _moves(qo, so, typ, False)
    want_up = (np.sign(need)[None, :] * np.sign(uo)[:, None]) > 0           # 値を上げると uᵀw が目標の向きに動く行
    dv = np.where(want_up, vu, vd)
    nq = np.where(want_up, nu_, nd)
    c = np.abs(uo[:, None] * dv)                                              # 1 段で動く |uᵀw|(動かせなければ 0)
    tgt = np.abs(need)
    cum = np.cumsum(c, axis=0)
    j = np.clip((cum < tgt[None, :]).sum(axis=0), 0, len(order) - 1)          # cum >= tgt になる最初の行
    k = np.arange(len(cols))
    got_hi = cum[j, k]
    got_lo = np.where(j > 0, cum[np.maximum(j - 1, 0), k], 0.0)
    use_hi = np.abs(tgt - got_hi) < np.abs(tgt - got_lo)
    last = np.where(use_hi, j, j - 1)                                         # この行まで(含む)を動かす(-1 なら動かさない)
    got = np.where(use_hi, got_hi, got_lo)
    sel = (np.arange(len(order))[:, None] <= last[None, :]) & (c > 0)
    rem = tgt - got
    # 残りにいちばん合う 1 段を、まだ動かしていない行から足す(残りより小さくなるときだけ)
    rest = (np.arange(len(order))[:, None] > last[None, :]) & (c > 0)
    fit = np.where(rest, np.abs(c - rem[None, :]), np.inf)
    jb = np.argmin(fit, axis=0)
    better = (rem > 0) & (fit[jb, k] < np.abs(rem))
    sel[jb[better], k[better]] = True
    got = got + np.where(better, c[jb, k], 0.0)
    q2 = q.copy()
    sub = np.where(sel, nq, qo)
    q2[np.ix_(order, cols)] = sub
    return {"q": q2, "flips": sel.sum(axis=0), "residual": (tgt - got) * np.sign(need)}


def edit_slab(raw: np.ndarray, typ: str, u: np.ndarray, delta_by_col: dict[int, float], rows: int, cols: int) -> dict:
    """1 つの専門家のバイト列に、列ごとの uᵀw の変化 delta_by_col を書き込む。戻り値: 新しいバイト列と、確かめの数(目標の
    uᵀw の変化・実際の変化・方向に沿わない変化の大きさ・動かしたコードの数)。ほかの列のコードはそのまま。"""
    q, sc, W = decode(raw, typ, rows, cols)
    cs = np.array(sorted(delta_by_col), np.int64)
    dl = np.array([delta_by_col[int(c)] for c in cs], np.float64)
    cur, flips = q, np.zeros(len(cs), np.int64)
    for _ in range(PASSES):                # 1 回で届かない列(大きな強さ・値の端のコード)は、残りをもう一度(同じ行をさらに 1 段)
        _, _, Wc = decode(encode(raw, typ, rows, cols, cur), typ, rows, cols) if cur is not q else (None, None, W)
        rest = dl - u @ (Wc[:, cs] - W[:, cs])
        todo = np.abs(rest) > REST_TOL * np.maximum(np.abs(dl), 1e-30)
        if not todo.any():
            break
        r = round_columns(cur, sc, typ, u, cs[todo], rest[todo])
        if not r["flips"].any():
            break
        cur = r["q"]
        flips[todo] += r["flips"]
    new = encode(raw, typ, rows, cols, cur)
    _, _, W2 = decode(new, typ, rows, cols)
    dw = W2[:, cs] - W[:, cs]
    along = u @ dw
    off = np.linalg.norm(dw - np.outer(u, along), axis=0)
    return {"raw": new, "want": dl, "along": along, "off": off, "flips": flips,
            "w_norm": np.linalg.norm(W[:, cs], axis=0)}


PASSES = 4          # 列ごとの丸めを繰り返す上限(1 回目で残りが目標の 2% 以内なら 1 回)
REST_TOL = 0.02
