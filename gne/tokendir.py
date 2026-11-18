"""出力層の重み行列の行(トークンの行)から方向を作る・層を走査する(gne direction token / token-scan、docs/fish.md §5「トークンの行の方向」)。

方向: u = (足すトークンの行の平均) − (引くトークンの行の平均。無ければ 0)、û = u ÷ ‖u‖。層 L の残差に β·û を足すと、最終層の
あと(正規化 γ・rms で割る)で、足すトークンのロジット(引くトークンがあれば、その平均との差)は

    Δ ≈ β · ((γ ⊙ (r̄_plus − r̄_minus)) · û) ÷ r̄      (r̄ = 最終層の出力の、正規化の前の rms の中央値)

だけ上がる(後ろの層が足した分を変えないとしたときの見積もり)。目標の Δ から β = Δ · r̄ ÷ ((γ ⊙ u)·û) で長さを決める。
r̄ は、対照集合の入力の位置 0(応答の最初のトークンを決める位置 = 入力の最後のトークン)で測る(--rbar で直接渡せる)。
down を付けると −β·û(そのトークンを出にくくする向き)。書いた方向は、ほかの方向と同じく nw のユニット
(`L:Na+Nb:nw:a1@NAME#KEY&ONE`)や `L:dir:NAME:1` で使える(α = 1 で β·û をそのまま足す)。

層の走査(token_scan): 層ごとに、狙う側の入力(対照集合の user_prompts。system_b の文脈)の位置 0 にだけ β(Δ)·û を足し、足すトークンが
1 位になった入力の割合が need 以上になる最小の Δ(deltas の中から)を探す。その Δ の leak 倍(既定 0.2・0.5)を、dont_touch の入力の
位置 0 に足したときの、足すトークンが 1 位になった数・位置 0 の KL(足さないときの分布から)・1 位が変わった数を出す。
入力の最後のトークンの手前までを 1 回だけ読んでおき、候補ごとにその状態を複製して最後のトークンだけを読む(位置 0 だけに足すのと同じ)。
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np

from .steer import GneError

DELTAS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
NEED = 0.95
LEAKS = (0.2, 0.5)


def resolve_token(engine, spec) -> int:
    """トークンの指定 → ID。整数か "#123"(ID)、それ以外は文字(特殊トークンの書き方も可。例 "<|im_end|>"、"◇")で、1 トークンに
    なるものだけ。"""
    s = str(spec).strip()
    if not s:
        raise GneError("BAD_ARGS", "トークンが空です")
    if s.lstrip("#").isdigit() and (s.startswith("#") or isinstance(spec, int)):
        t = int(s.lstrip("#"))
        if not 0 <= t < engine.n_vocab:
            raise GneError("BAD_ARGS", f"トークン ID {t} が語彙(0〜{engine.n_vocab - 1})の外です")
        return t
    ids = engine.tokenize(str(spec), add_special=False, parse_special=True)
    if len(ids) != 1:
        raise GneError("BAD_ARGS", f"{spec!r} は 1 トークンになりません({len(ids)} トークン: {ids[:8]})",
                       "1 トークンになる文字か、#<ID> で渡してください")
    return int(ids[0])


def _tokens(engine, specs) -> list[int]:
    if specs is None:
        return []
    if isinstance(specs, (str, int)):
        specs = [specs]
    return [resolve_token(engine, x) for x in specs]


def direction_of(weights, plus: Sequence[int], minus: Sequence[int] = ()) -> dict:
    """足す・引くトークンの行 → {u_hat, gdot, u_norm, row_norms, cos_plus_minus}。gdot = (γ ⊙ u)·û(Δ と β の換算の分母)。"""
    if not plus:
        raise GneError("BAD_ARGS", "足すトークンがありません")
    gam = np.asarray(weights.output_norm_weight(), dtype=np.float64)
    P = np.asarray(weights.embed_rows(list(plus)) if weights.meta.tied_output else _out_rows(weights, plus), dtype=np.float64)
    u = P.mean(0)
    cos = None
    if minus:
        M = np.asarray(weights.embed_rows(list(minus)) if weights.meta.tied_output else _out_rows(weights, minus), dtype=np.float64)
        cos = float(P.mean(0) @ M.mean(0) / max(np.linalg.norm(P.mean(0)) * np.linalg.norm(M.mean(0)), 1e-30))
        u = u - M.mean(0)
    n = float(np.linalg.norm(u))
    if n < 1e-12:
        raise GneError("BAD_ARGS", "足すトークンと引くトークンの行の平均が同じです(方向が決まりません)")
    uh = u / n
    gdot = float((gam * u) @ uh)
    if gdot <= 0.1 * float(np.linalg.norm(gam * u)):
        raise GneError("BAD_DIRECTION", f"この方向では、最終層の正規化のあとでトークンのロジットがほとんど上がりません((γ⊙u)·û = {gdot:.4g})")
    return {"u_hat": uh, "gdot": gdot, "u_norm": n, "row_norms": [float(np.linalg.norm(r)) for r in P], "cos_plus_minus": cos}


def _out_rows(weights, ids: Sequence[int]) -> np.ndarray:
    """出力層の行列(tied でないとき)の行。"""
    want = {int(t): k for k, t in enumerate(ids)}
    out = np.zeros((len(ids), weights.meta.n_embd), dtype=np.float32)
    for st, rows in weights.output_rows(8192):
        for t, k in want.items():
            if st <= t < st + rows.shape[0]:
                out[k] = rows[t - st]
    return out


def _prompt_ids(session, contrast) -> tuple[list[list[int]], list[list[int]], str]:
    """対照集合 → (狙う側の入力のトークン列, dont_touch の入力のトークン列, 名前)。狙う側は system_b の文脈(推論時と同じ)。"""
    from .keyed_build import _as_spec
    from .prior import ensure_filled
    from .reverse import _conv
    spec = _as_spec(ensure_filled(session, contrast))
    cs = spec.contrast
    e = session.engine
    ids = lambda c: e.tokenize_prompt(e.render_chat(c))
    open_ = [ids(_conv(cs.system_b, q)) for q in cs.prompts("B")]
    close = [ids(c) for dt in spec.dont_touch for c in dt.convs(range(len(dt.prompts)))]
    return open_, close, cs.name


def final_rms(session, seqs: list[list[int]], *, max_n: int = 400) -> float:
    """入力の位置 0(最後のトークン)での、最終層の出力(正規化の前)の rms の中央値。多ければ等間隔に max_n 本で測る。"""
    from . import keyed as K
    from .nwrite import _run_capture
    if not seqs:
        raise GneError("BAD_ARGS", "r̄ を測る入力がありません")
    L = K.last_main_layer(session.weights)
    name = f"l_out-{L}"
    pick = seqs if len(seqs) <= max_n else [seqs[int(i)] for i in np.linspace(0, len(seqs) - 1, max_n)]
    vals = []
    for ids in pick:
        x = _run_capture(session, ids, {name})[name][-1]
        vals.append(float(np.sqrt(np.mean(x * x))))
    return float(np.median(vals))


def _default_seqs(session) -> list[list[int]]:
    import yaml

    from .store import CONFIGS
    d = yaml.safe_load((CONFIGS / "prompts" / "default.yaml").read_text(encoding="utf-8")) or {}
    qs = d.get("prompts") if isinstance(d, dict) else d
    qs = [q for q in (qs or []) if isinstance(q, str)][:64]
    e = session.engine
    return [e.tokenize_prompt(e.render_chat([{"role": "user", "content": q}])) for q in qs]


def check_name(name: str) -> str:
    import re
    if not name or not re.fullmatch(r"[\w.][\w.\-]*", str(name)) or "." in str(name):
        raise GneError("BAD_ARGS", f"方向の名前 {name!r} が不正です(英数字・_・- だけ)")
    return str(name)


def token_direction(session, tokens, *, layers, delta: float, name: str, minus=None, down: bool = False, contrast=None,
                    rbar: float | None = None) -> dict:
    """トークンの行から方向を作り、層ごとに ±β·û を書いた方向のファイルを保存する(<root>/directions/<model_id>/<name>.npz)。
    tokens・minus はトークンの指定のリスト(resolve_token)。delta は目標のロジットの上がり Δ。r̄ は contrast の入力
    (狙う側と dont_touch の全部)の位置 0 で測る(省くと configs/prompts/default.yaml の質問)。rbar を渡すとその値を使う。"""
    t0 = time.perf_counter()
    check_name(name)
    e, w = session.engine, session.weights
    plus, mns = _tokens(e, tokens), _tokens(e, minus)
    Ls = sorted({int(L) for L in (layers or [])})
    if not Ls:
        raise GneError("BAD_ARGS", "方向を書く層(layers)を指定してください")
    from . import keyed as K
    last = K.last_main_layer(w)
    bad = [L for L in Ls if not 0 <= L < last]
    if bad:
        raise GneError("BAD_ARGS", f"層 {bad} には書けません(0〜{last - 1}。最終層は除く)")
    try:
        delta = float(delta)
    except (TypeError, ValueError):
        raise GneError("BAD_ARGS", f"delta(ロジットの上がり Δ){delta!r} を読めません") from None
    if not np.isfinite(delta) or delta <= 0:
        raise GneError("BAD_ARGS", f"delta(ロジットの上がり Δ)は 0 より大きい数です({delta!r})")
    d = direction_of(w, plus, mns)
    src = "rbar"
    if rbar is None:
        seqs = _default_seqs(session) if contrast is None else [x for part in _prompt_ids(session, contrast)[:2] for x in part]
        rbar = final_rms(session, seqs)
        src = "contrast" if contrast is not None else "default_prompts"
    rbar = float(rbar)
    if not np.isfinite(rbar) or rbar <= 0:
        raise GneError("BAD_ARGS", f"rbar は 0 より大きい数です({rbar!r})")
    beta = delta * rbar / d["gdot"]
    v = ((-1.0 if down else 1.0) * beta * d["u_hat"]).astype(np.float32)
    path = session.store.direction_dir(session.meta.model_id) / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    desc = (f"token rows +{plus}" + (f" -{mns}" if mns else "") + f" delta={delta:g} rbar={rbar:.6g} beta={beta:.6g}"
            + (" down" if down else ""))
    np.savez(path, layers=np.array(Ls, dtype=np.int32), **{f"layer_{L}": v for L in Ls}, method="token_row", positions="pos:0",
             source=desc, n_rows_A=len(plus), n_rows_B=len(mns), tokens=np.array(plus, dtype=np.int64),
             minus=np.array(mns, dtype=np.int64), delta=float(delta), rbar=rbar, beta=float(beta), gdot=d["gdot"],
             down=bool(down))
    return {"name": name, "path": str(path), "layers": Ls, "tokens": plus, "token_text": [e.token_text(t) for t in plus],
            "minus": mns, "minus_text": [e.token_text(t) for t in mns], "delta": delta, "rbar": rbar, "rbar_from": src,
            "gdot": d["gdot"], "beta": beta, "down": bool(down), "u_norm": d["u_norm"], "row_norms": d["row_norms"],
            "cos_plus_minus": d["cos_plus_minus"], "seconds": time.perf_counter() - t0}


def _lsm(z: np.ndarray) -> np.ndarray:
    m = z.max(-1, keepdims=True)
    return z - (m + np.log(np.exp(z - m).sum(-1, keepdims=True)))


class _LastTokenReader:
    """入力の最後のトークンの手前までを列 0〜B−1 に読んでおき、候補ごとに列 B〜2B−1 へ複製して、control vector を変えて最後の
    トークンだけを読む(複製は読み終えたら消す)。線形注意・SSM の層があるモデルでは、状態を途中まで消して読み直すことが
    できないので、手前までの状態を複製して使う。B = n_seq_max ÷ 2。"""

    def __init__(self, session, seqs: list[list[int]]):
        from llama_cpp import llama_cpp as C
        self.C = C
        self.e = session.engine
        if int(self.e.n_seq_max) < 2:
            raise GneError("BAD_ARGS", "層の走査には n_seq_max 2 以上が要ります(--n-seq-max)")
        self.ctx = self.e.steer_ctx()
        self.mem = C.llama_get_memory(self.ctx.ptr)
        self.B = int(self.e.n_seq_max) // 2
        self.seqs = seqs

    def batches(self):
        for i in range(0, len(self.seqs), self.B):
            yield self.seqs[i:i + self.B]

    def load(self, bb: list[list[int]]) -> None:
        e = self.e
        e.set_lora(self.ctx, None)
        e.set_cvec(self.ctx, None)
        e.clear_memory(self.ctx)
        e.prefill(self.ctx, [x[:-1] for x in bb])

    def last(self, bb: list[list[int]], V: dict | None) -> np.ndarray:
        e, C, B = self.e, self.C, self.B
        for j in range(len(bb)):
            C.llama_memory_seq_rm(self.mem, B + j, -1, -1)
            C.llama_memory_seq_cp(self.mem, j, B + j, -1, -1)
        try:
            if V:
                e.set_cvec(self.ctx, V)
            e._decode(self.ctx, [(x[-1], len(x) - 1, B + j, True) for j, x in enumerate(bb)])
            z = np.stack([e._logits(self.ctx, j) for j in range(len(bb))]).astype(np.float64)
        finally:
            e.set_cvec(self.ctx, None)
            for j in range(len(bb)):
                C.llama_memory_seq_rm(self.mem, B + j, -1, -1)
        return z


def token_scan(session, tokens, contrast, *, layers=None, minus=None, deltas: Sequence[float] = DELTAS, need: float = NEED,
               leaks: Sequence[float] = LEAKS, rbar: float | None = None,
               progress: Callable[[int, int, str], None] | None = None) -> dict:
    """層ごとに、狙う側の入力の位置 0 で足すトークンが 1 位になる最小の Δ と、dont_touch の入力への影響を測る(生成しない)。
    layers を省くと、最終層を除く全部の層。結果の rows は層ごとに {layer, need_delta, beta, open_hit(その Δ で 1 位になった数),
    leak: {倍率: {top1(1 位になった数), kl(平均), changed(1 位が変わった数)}}}。need_delta が None の層は、deltas の最大でも届かない。"""
    t0 = time.perf_counter()
    e, w = session.engine, session.weights
    plus, mns = _tokens(e, tokens), _tokens(e, minus)
    from . import keyed as K
    last = K.last_main_layer(w)
    Ls = sorted({int(L) for L in layers}) if layers else list(range(0, last))
    bad = [L for L in Ls if not 0 <= L < last]
    if bad:
        raise GneError("BAD_ARGS", f"層 {bad} は走査できません(0〜{last - 1})")
    ds = sorted({float(x) for x in deltas})
    if not ds or ds[0] <= 0:
        raise GneError("BAD_ARGS", "deltas は 0 より大きい数のリストです")
    if not 0 < float(need) <= 1:
        raise GneError("BAD_ARGS", f"need は 0 より大きく 1 以下の割合です({need!r})")
    open_, close, cname = _prompt_ids(session, contrast)
    if not open_:
        raise GneError("BAD_CONTRAST", "狙う側の入力(user_prompts)がありません")
    d = direction_of(w, plus, mns)
    rb = float(rbar) if rbar is not None else final_rms(session, open_ + close)
    uh = d["u_hat"]
    beta_of = lambda delta: delta * rb / d["gdot"]
    pset = np.asarray(plus)
    n_need = int(np.ceil(float(need) * len(open_) - 1e-9))
    reader_o = _LastTokenReader(session, open_)
    reader_c = _LastTokenReader(session, close)
    total = len(Ls) * (1 + (len(leaks) if close else 0))
    done = 0

    def tick(msg):
        nonlocal done
        done += 1
        if progress:
            progress(min(done, total), total, msg)

    def vec(L, delta):
        return {L: (beta_of(delta) * uh).astype(np.float32)}

    # 足さないとき
    base_o, base_c = [], []
    for bb in reader_o.batches():
        reader_o.load(bb)
        base_o.append(reader_o.last(bb, None))
    for bb in reader_c.batches():
        reader_c.load(bb)
        base_c.append(reader_c.last(bb, None))
    base_o = np.concatenate(base_o) if base_o else np.zeros((0, e.n_vocab))
    base_c = np.concatenate(base_c) if base_c else np.zeros((0, e.n_vocab))
    base_hit_o = int(np.isin(base_o.argmax(1), pset).sum())
    base_hit_c = int(np.isin(base_c.argmax(1), pset).sum()) if close else 0
    lp_c = _lsm(base_c) if close else None
    top_c = base_c.argmax(1) if close else None
    # 狙う側: バッチごとに 1 回だけ読み、全部の層 × Δ を最後のトークンの読み直しで測る
    hits = {(L, delta): 0 for L in Ls for delta in ds}
    for bb in reader_o.batches():
        reader_o.load(bb)
        for L in Ls:
            for delta in ds:
                z = reader_o.last(bb, vec(L, delta))
                hits[(L, delta)] += int(np.isin(z.argmax(1), pset).sum())
    need_of = {}
    for L in Ls:
        need_of[L] = next((delta for delta in ds if hits[(L, delta)] >= n_need), None)
        tick(f"L{L}")
    # dont_touch: 層ごとの必要な Δ の leak 倍
    acc = {(L, f): {"top1": 0, "changed": 0, "kl": []} for L in Ls for f in leaks}
    if close:
        k = 0
        for bb in reader_c.batches():
            reader_c.load(bb)
            n = len(bb)
            p0 = np.exp(lp_c[k:k + n])
            for L in Ls:
                if need_of[L] is None:
                    continue
                for f in leaks:
                    z = reader_c.last(bb, vec(L, float(f) * need_of[L]))
                    a = acc[(L, f)]
                    a["top1"] += int(np.isin(z.argmax(1), pset).sum())
                    a["changed"] += int((z.argmax(1) != top_c[k:k + n]).sum())
                    a["kl"] += list((p0 * (lp_c[k:k + n] - _lsm(z))).sum(1))
            k += n
        for L in Ls:
            for f in leaks:
                tick(f"L{L} ×{f:g}")
    rows = []
    for L in Ls:
        nd = need_of[L]
        row = {"layer": L, "need_delta": nd, "beta": None if nd is None else beta_of(nd),
               "open_hit": None if nd is None else hits[(L, nd)], "open_n": len(open_),
               "tried": {format(delta, "g"): hits[(L, delta)] for delta in ds}, "leak": {}}
        if nd is not None and close:
            for f in leaks:
                a = acc[(L, f)]
                row["leak"][str(f)] = {"top1": a["top1"], "kl": float(np.mean(a["kl"])), "changed": a["changed"], "n": len(close)}
        rows.append(row)
    return {"contrast": cname, "tokens": plus, "token_text": [e.token_text(t) for t in plus], "minus": mns, "rbar": rb,
            "rbar_from": "rbar" if rbar is not None else "contrast", "gdot": d["gdot"], "need": float(need), "need_n": n_need,
            "deltas": ds, "leaks": [float(x) for x in leaks], "open_n": len(open_), "close_n": len(close),
            "base_open_top1": base_hit_o, "base_close_top1": base_hit_c, "rows": rows, "seconds": time.perf_counter() - t0}


def scan_lines(d: dict) -> list[str]:
    """token_scan の結果を表の行に。"""
    out = [f"トークン {d['token_text']}(r̄ {d['rbar']:.4f}、(γ⊙u)·û {d['gdot']:.4f})、狙う側 {d['open_n']} 入力で {d['need_n']} 以上が 1 位になる最小の Δ。"
           f"足さないとき 1 位: 狙う側 {d['base_open_top1']}、dont_touch {d['base_close_top1']}/{d['close_n']}",
           "| 層 | 必要な Δ | β | " + " | ".join(f"×{f:g}: 1 位 / KL / 1 位が変わった" for f in d["leaks"]) + " |",
           "|---|---|---|" + "---|" * len(d["leaks"])]
    for r in d["rows"]:
        cells = [f"{r['leak'][str(f)]['top1']} / {r['leak'][str(f)]['kl']:.4f} / {r['leak'][str(f)]['changed']}" if str(f) in r["leak"]
                 else "—" for f in d["leaks"]]
        out.append(f"| L{r['layer']} | {'—' if r['need_delta'] is None else format(r['need_delta'], 'g')} | "
                   f"{'—' if r['beta'] is None else format(r['beta'], '.2f')} | " + " | ".join(cells) + " |")
    return out
