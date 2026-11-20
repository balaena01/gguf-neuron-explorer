"""出力のあとに正規化がある構造(post-norm)の土台(v0.9.0): 書き込みを捕捉する名前の解決と、「書き込みの和」の検算。

残差に入るのは、混ぜる層(注意・畳み込みなど)と MLP の書き込み y を正規化した γ ⊙ y / rms(y)(rms(y) = sqrt(mean(y²) + eps))。
捕捉の名前は arch ごとに configs/arch/residual_writers.yaml の capture の表に書く(llama.cpp のグラフの名前は arch で違い、
同じ名前が別のもの(attn_out が W_O の出力・注意を足した後の残差・ゲートの前の連結)を指すこともあるため)。

check_writer_sums は、数層について次を測る(gne verify の V27。表の名前を使う機能は、使う前に require_writer_sums で通す):
  (a) 混ぜる層: 捕捉した y(書き込み行列の出力。名前が無ければ W_O · ヘッドの出力)から作った γ ⊙ y / rms(y) が、
      残差に足された値(mid − l_out(L−1)、または正規化の後の捕捉)と合う
  (b) MLP: γ ⊙ (W_down a) / rms と、残差に足された値(l_out(L) − mid、または正規化の後の捕捉)が合う
  (c) l_out(L) = l_out(L−1) + (a の値) + (b の値)
許容は V2 と同じ(量子化した行列の積の誤差。書き込み行列の型ごと)。和(c)は 1e-3。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .steer import GneError

SUM_TOL = 1e-3


@lru_cache(maxsize=1)
def _table() -> dict:
    from .ablate import writers_table
    return writers_table().get("capture") or {}


def entry(arch: str) -> dict | None:
    """arch の捕捉の表(無ければ None)。"""
    return _table().get(arch)


def layer_names(weights, L: int) -> dict[str, str | None]:
    """層 L の捕捉の名前("-L" 付き)。表に無い arch は空の dict。"""
    ent = entry(weights.meta.arch)
    if not ent:
        return {}
    li = weights.arch.layers[weights.stored(L)] if hasattr(weights, "stored") else weights.arch.layers[L]

    def nm(x):
        return f"{x}-{L}" if x else None
    mo = ent.get("mixer_out")
    if isinstance(mo, dict):
        mo = mo.get(li.mixer)
    return {"head_out": nm(ent.get("head_out")), "head_pre": nm(ent.get("head_pre")), "head_gate": nm(ent.get("head_gate")),
            "mixer_out": nm(mo), "mixer_post": nm(ent.get("mixer_post")), "mid": nm(ent.get("mid")),
            "ffn_post": nm(ent.get("ffn_post"))}


def checked(arch: str) -> str | None:
    """表での確かめの度合い: real(実物で確かめた)/ graph(グラフを読んだだけ)/ None(表に無い)。"""
    ent = entry(arch)
    return None if not ent else str(ent.get("checked") or "graph")


def eps_of(weights) -> float:
    return float(weights.field(f"{weights.meta.arch}.attention.layer_norm_rms_epsilon", 1e-6) or 1e-6)


def gamma(weights, L: int, side: str) -> np.ndarray | None:
    """層 L の出力のあとの正規化の γ(side は mixer / ffn)。正規化が無い・確かめていない("?")層は None。"""
    from .weights import tensor_to_f32
    ai = getattr(weights, "arch", None)
    if ai is None:                                   # 構造の情報の無い重み(テストの偽物など)は正規化なしとみなす
        return None
    li = ai.layers[L]
    nm = getattr(li, "post_norm_mixer", "") if side == "mixer" else getattr(li, "post_norm_ffn", "")
    if not nm or nm == "?":
        return None
    return np.asarray(tensor_to_f32(weights.tensor(f"blk.{weights.stored(L)}.{nm}.weight")), np.float64).reshape(-1)


def post_add(y: np.ndarray, g: np.ndarray | None, eps: float, kind: str = "rms") -> np.ndarray:
    """出力のあとの正規化: γ ⊙ y / sqrt(mean(y²) + eps)(kind scale なら γ ⊙ y)。g が None なら y のまま。y は (T, n_embd)。"""
    y = np.asarray(y, np.float64)
    if g is None:
        return y
    if kind == "scale":
        return y * g[None, :]
    return g[None, :] * y / np.sqrt(np.mean(y * y, axis=1, keepdims=True) + eps)


def head_out_parts(weights, names: set[str], L: int) -> tuple[str, str | None] | None:
    """層 L のヘッドの出力の捕捉: (名前, ゲートの名前 or None)。ゲートがあれば ヘッドの出力 = 名前の値 ⊙ ゲート(ヘッドごと)。
    表の head_pre + head_gate(LFM3)、表の head_out、arch の候補(attn_gated / kqv_out / final_output)の順。見えなければ None。"""
    nm = layer_names(weights, L)
    if nm.get("head_pre") and nm.get("head_gate"):
        if nm["head_pre"] in names and nm["head_gate"] in names:
            return nm["head_pre"], nm["head_gate"]
        return None
    if nm.get("head_out") and nm["head_out"] in names:
        return nm["head_out"], None
    for base in weights.arch.layers[L].head_out:
        if f"{base}-{L}" in names:
            return f"{base}-{L}", None
    return None


def gated(pre: np.ndarray, gate: np.ndarray | None, n_head: int, head_dim: int) -> np.ndarray:
    """ヘッドの出力 (T, n_head, head_dim)。gate (T, n_head) があればヘッドごとに掛ける。"""
    T = pre.shape[0] if pre.shape[0] * n_head * head_dim == pre.size else pre.size // (n_head * head_dim)
    x = np.asarray(pre).reshape(T, n_head, head_dim)
    if gate is None:
        return x
    return x * np.asarray(gate).reshape(T, n_head)[:, :, None]


def writer_layers(weights) -> list[int]:
    """出力のあとに正規化がある(確かめた)層。"""
    return [li.index for li in weights.arch.layers[:weights.n_main]
            if li.post_norm_mixer not in ("", "?") or li.post_norm_ffn not in ("", "?")]


def _pick_layers(weights) -> list[int]:
    """検算する層: 混ぜる層の種類ごとに最初の 1 層(層 1 以降)、真ん中、最後の 1 つ前。"""
    n = weights.n_main
    pn = set(writer_layers(weights))
    out, kinds = [], set()
    for li in weights.arch.layers[1:n - 1]:
        if li.index in pn and li.mixer not in kinds:
            kinds.add(li.mixer)
            out.append(li.index)
    for L in (n // 2, n - 2):
        if L in pn and L not in out and L >= 1:
            out.append(L)
    return sorted(out)


def _tol(weights, L: int) -> tuple[float, float]:
    from .verify import V2_TOL, V2_TOL_DEFAULT
    li = weights.arch.layers[L]
    for nm in (f"blk.{weights.stored(L)}.{li.head_writer}.weight" if li.head_writer else None,
               f"blk.{weights.stored(L)}.ffn_down.weight"):
        if nm and weights.has_tensor(nm):
            q = getattr(weights.tensor(nm).tensor_type, "name", "")
            if q in V2_TOL:
                return V2_TOL[q]
    return V2_TOL_DEFAULT


def check_writer_sums(session, layers: list[int] | None = None, text: str | None = None) -> dict:
    """層ごとに (a)(b)(c) を測る(モジュールの説明)。戻り値 {ok, arch, checked, layers: [{layer, mixer, mixer_err, ffn_err,
    sum_err, ok, why}], text_tokens}。出力のあとに正規化が無いモデルは ok で skipped。"""
    from .capture import Capture, pick_act_name
    e, w = session.engine, session.weights
    arch = w.meta.arch
    pn = writer_layers(w)
    if not pn:
        return {"ok": True, "arch": arch, "checked": checked(arch), "skipped": "出力のあとに正規化がある層はありません", "layers": []}
    if entry(arch) is None:
        return {"ok": False, "arch": arch, "checked": None, "layers": [],
                "why": f"arch {arch} の捕捉の名前が configs/arch/residual_writers.yaml の capture の表にありません"}
    Ls = sorted(set(layers) if layers else set(_pick_layers(w)))
    text = text or "The lighthouse keeper watched the ships cross the bay at dawn, and wrote each name in a small book."
    ids = e.tokenize_prompt(e.render_chat([{"role": "user", "content": text}]))
    seen = set(e.list_names(ids[:16]))
    eps = eps_of(w)
    want: set[str] = set()
    plan = {}
    for L in Ls:
        nm = layer_names(w, L)
        li = w.arch.layers[L]
        p = {"prev": f"l_out-{L - 1}", "out": f"l_out-{L}", "mid": nm.get("mid") if nm.get("mid") in seen else None,
             "mpost": nm.get("mixer_post") if nm.get("mixer_post") in seen else None,
             "fpost": nm.get("ffn_post") if nm.get("ffn_post") in seen else None,
             "mout": nm.get("mixer_out") if nm.get("mixer_out") in seen else None,
             "heads": head_out_parts(w, seen, L) if li.mixer == "attention" and li.head_dim else None,
             "act": pick_act_name(seen, L) if li.ffn == "dense" else None}
        plan[L] = p
        want |= {x for x in (p["prev"], p["out"], p["mid"], p["mpost"], p["fpost"], p["mout"], p["act"]) if x}
        if p["heads"]:
            want |= {x for x in p["heads"] if x}
    cap = Capture(lambda n: n in want)
    ctx = e.probe_ctx(cap)
    e.clear_memory(ctx)
    e.set_cvec(ctx, None)
    e.set_lora(ctx, None)
    e.prefill(ctx, [ids])
    got = cap.take()
    T = len(ids)
    rows = []

    def rel(a, b):
        den = np.linalg.norm(b, axis=1)
        keep = den > 1e-6 * max(1.0, float(den.max()))
        r = np.linalg.norm(a - b, axis=1)[keep] / den[keep]
        return (float(np.median(r)), float(r.max())) if r.size else (None, None)

    for L in Ls:
        p, li = plan[L], w.arch.layers[L]
        why = []
        if p["prev"] not in got or p["out"] not in got:
            rows.append({"layer": L, "mixer": li.mixer, "ok": False, "why": "l_out が捕捉できません"})
            continue
        prev, out = got[p["prev"]].astype(np.float64), got[p["out"]].astype(np.float64)
        if p["mid"]:
            m_add = got[p["mid"]].astype(np.float64) - prev
            f_add = out - got[p["mid"]].astype(np.float64)
        elif p["mpost"] and p["fpost"]:
            m_add, f_add = got[p["mpost"]].astype(np.float64), got[p["fpost"]].astype(np.float64)
        else:
            rows.append({"layer": L, "mixer": li.mixer, "ok": False,
                         "why": "残差に足した値の捕捉の名前(mid か、mixer_post と ffn_post)が見えません"})
            continue
        s_err = float(np.linalg.norm(prev + m_add + f_add - out) / max(float(np.linalg.norm(out)), 1e-12))
        if s_err > SUM_TOL:
            why.append(f"l_out の和が合いません(相対誤差 {s_err:.2e})")
        tm, tx = _tol(w, L)
        m_err = f_err = h_err = None
        y = yh = None
        if p["heads"] and li.head_writer:            # ヘッドの出力(ゲートを掛けて作る arch も)から W_O で作った書き込み
            pre, gt = p["heads"]
            ho = gated(got[pre], got.get(gt) if gt else None, li.n_head, li.head_dim).reshape(T, -1).astype(np.float64)
            yh = ho @ w.head_writer_matrix(L).astype(np.float64).T
        if p["mout"]:
            y = got[p["mout"]].astype(np.float64).reshape(T, -1)
            if yh is not None:                       # ヘッドの出力の作り方(ゲート)も確かめる
                h_err = rel(yh, y)
                if h_err[0] is None or h_err[0] > tm or h_err[1] > tx:
                    why.append(f"ヘッドの出力から作った書き込みが、捕捉した書き込み行列の出力と合いません(相対誤差 中央値 {h_err[0]:.3g} / "
                               f"最大 {h_err[1]:.3g}、許容 {tm}/{tx})" if h_err[0] is not None else "ヘッドの出力が 0 です")
        else:
            y = yh
        if y is not None and li.post_norm_mixer not in ("", "?"):
            m_err = rel(post_add(y, gamma(w, L, "mixer"), eps, li.post_norm_kind or "rms"), m_add)
            if m_err[0] is None or m_err[0] > tm or m_err[1] > tx:
                why.append(f"混ぜる層の書き込みから作った値が、残差に足された値と合いません(相対誤差 中央値 {m_err[0]:.3g} / 最大 {m_err[1]:.3g}、"
                           f"許容 {tm}/{tx})" if m_err[0] is not None else "混ぜる層の書き込みが 0 です")
        elif li.post_norm_mixer not in ("", "?"):
            why.append("混ぜる層の書き込み(書き込み行列の出力かヘッドの出力)が捕捉できません")
        if p["act"] and li.post_norm_ffn not in ("", "?"):
            a = got[p["act"]].astype(np.float64)
            yf = a @ w.down_matrix(L).astype(np.float64).T * float((w.meta.ffn_down_scale or {}).get(L, 1.0))
            f_err = rel(post_add(yf, gamma(w, L, "ffn"), eps, li.post_norm_kind or "rms"), f_add)
            if f_err[0] is None or f_err[0] > tm or f_err[1] > tx:
                why.append(f"MLP の書き込みから作った値が、残差に足された値と合いません(相対誤差 中央値 {f_err[0]:.3g} / 最大 {f_err[1]:.3g}、"
                           f"許容 {tm}/{tx})" if f_err[0] is not None else "MLP の書き込みが 0 です")
        rows.append({"layer": L, "mixer": li.mixer, "head_err": h_err, "mixer_err": m_err, "ffn_err": f_err, "sum_err": s_err,
                     "ok": not why, "why": "、".join(why) or None})
    ok = bool(rows) and all(r["ok"] for r in rows)
    return {"ok": ok, "arch": arch, "checked": checked(arch), "layers": rows, "text_tokens": T}


def require_writer_sums(session, what: str) -> dict:
    """表の名前を使う前の確かめ(セッションごとに 1 回)。合わなければ、what を実行しない理由つきの GneError。"""
    res = getattr(session, "_writer_sums", None)
    if res is None:
        res = check_writer_sums(session)
        session._writer_sums = res
    if not res["ok"]:
        bad = next((r for r in res["layers"] if not r["ok"]), None)
        detail = res.get("why") or (f"層 {bad['layer']}: {bad['why']}" if bad else "検算する層がありません")
        src = ("llama.cpp のグラフを読んで書いた名前で、実物では確かめていません" if res.get("checked") == "graph"
               else "実物で確かめた名前ですが、いまの llama.cpp では合いませんでした")
        raise GneError("ARCH_UNSUPPORTED", f"{what}は実行しません。理由: arch {res['arch']} の書き込みの捕捉の名前と式が、"
                                           f"捕捉した値と合いませんでした({detail})",
                       f"configs/arch/residual_writers.yaml の capture の表({src})を確かめてください。gne verify の V27 で同じ検算を流せます")
    return res


def writer_sums_vresult(session):
    """gne verify の V27(書き込みの和)。"""
    from .verify import VResult
    res = check_writer_sums(session)
    session._writer_sums = res
    title = "書き込みの和(出力のあとの正規化)"
    if res.get("skipped"):
        return VResult("V27", title, "skip", res["skipped"], res)
    if not res["layers"]:
        return VResult("V27", title, "fail", res.get("why") or "検算する層がありません", res)

    def f(x):
        return "—" if not x else f"{x[0] * 100:.2f}% / {x[1] * 100:.2f}%"
    parts = [f"L{r['layer']}({r['mixer']}): " + (f"ヘッド {f(r.get('head_err'))}、" if r.get("head_err") else "")
             + f"混ぜる層 {f(r.get('mixer_err'))}、MLP {f(r.get('ffn_err'))}、和 {r.get('sum_err', 0):.1e}"
             + ("" if r["ok"] else f" — {r['why']}") for r in res["layers"]]
    src = {"real": "実物で確かめた名前", "graph": "グラフを読んだだけの名前"}.get(res.get("checked"), "")
    return VResult("V27", title, "pass" if res["ok"] else "fail",
                   f"{src}。相対誤差(中央値 / 最大): " + "; ".join(parts), res)


# --- 補正つきの射影(v0.9.0 B2)-------------------------------------------------------------------------
# 残差の r̂ の成分は (γ⊙r̂)·y / rms(y)。書き込み y から v̂ = normalize(γ⊙r̂) の成分を引けば、rms に関係なく 0 になる(s=1)。
# s=2 は折り返し(大きさが変わらない)なので成分はちょうど反対。ほかの s では rms が変わるので比例せず、s を上げると頭打ちになる。

SIDE_RATIO_WARN = 3.0     # 補正で残差が r̂ 以外へ動く量 ÷ r̂ に沿って動く量 がこれを超える層は警告する(LFM3・Gemma 3 で測った値で置いた)
KIND_SIDE = {"head": "mixer", "attn": "mixer", "linear": "mixer", "neuron": "ffn", "ffn": "ffn"}


def writer_side(name_or_kind: str) -> str:
    """書き込み行列の名前か proj の種類 → mixer / ffn。"""
    if name_or_kind in KIND_SIDE:
        return KIND_SIDE[name_or_kind]
    return "ffn" if ".ffn_down" in name_or_kind else "mixer"


def corrected_dir(weights, L: int, r: np.ndarray, side: str, *, correction: bool = True) -> np.ndarray:
    """層 L の side(mixer / ffn)で書き込みから引く単位ベクトル。出力のあとに正規化があり補正するなら normalize(γ⊙r̂)、
    それ以外は r̂ のまま(正規化の無いモデル・補正を切ったときは、v0.8.x までと同じ)。"""
    r = np.asarray(r, np.float64).reshape(-1)
    rh = r / max(float(np.linalg.norm(r)), 1e-30)
    if not correction:
        return rh.astype(np.float32)
    g = gamma(weights, L, side)
    if g is None:
        return rh.astype(np.float32)
    v = g * rh
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else rh.astype(np.float32)


def side_ratio(weights, L: int, r: np.ndarray, side: str) -> float | None:
    """補正つきで消したとき、残差に足す値が r̂ 以外の向きへ動く量 ÷ r̂ に沿って動く量(γ⊙v̂ の r̂ に直交な部分 ÷ r̂ の成分)。
    正規化の無い層は None(0 と同じ)。"""
    g = gamma(weights, L, side)
    if g is None:
        return None
    rh = np.asarray(r, np.float64).reshape(-1)
    rh = rh / max(float(np.linalg.norm(rh)), 1e-30)
    v = g * rh
    n = float(np.linalg.norm(v))
    if n == 0:
        return float("inf")
    gv = g * (v / n)
    along = float(rh @ gv)
    return float(np.linalg.norm(gv - along * rh) / max(abs(along), 1e-30))


def side_warning(weights, items: list[tuple[int, str, np.ndarray]]) -> str | None:
    """(層, side, r) の並び → 比が SIDE_RATIO_WARN を超える層の警告(無ければ None)。"""
    bad = []
    for L, side, r in items:
        x = side_ratio(weights, L, r, side)
        if x is not None and x > SIDE_RATIO_WARN:
            bad.append((L, side, x))
    if not bad:
        return None
    bad = sorted(set(bad))
    txt = "、".join(f"層 {L}{'(MLP)' if s == 'ffn' else ''} {x:.1f} 倍" for L, s, x in bad[:6]) + (" …" if len(bad) > 6 else "")
    return (f"この方向を補正つきで取り除くと、{txt}: 残差に足す値が、r̂ に沿って動く量に比べてその倍だけ、ほかの向きへも動きます"
            f"(この構造のモデルでは中央値でも 1 倍前後。{SIDE_RATIO_WARN:g} 倍を超える層を知らせています)。副作用が出やすい層です")


def predict_ratio(y: np.ndarray, yu: np.ndarray, u: np.ndarray, s: float, kind: str = "rms", eps: float = 1e-6) -> np.ndarray:
    """書き込み y (T, n) の一部 yu(その部品の書き込み。層全体なら y)を、u 方向(単位ベクトル v̂)に s だけ引いたときの、
    残差の r̂ の成分の変化 ÷ (s × 強さ 1 のときの変化)。トークンごと。頭打ちの表示に使う(1 なら s に比例、0.5 なら半分)。
    層全体では s=1 と s=2(折り返し)でちょうど 1。"""
    if kind == "scale":
        return np.ones(len(y))
    a = yu @ u                                                       # (T,) 部品の v̂ 成分
    r0 = np.sqrt(np.mean(y * y, axis=1) + eps)

    def change(k):                                                   # 残差の r̂ 成分 ∝ (u·y)/rms(y)(‖γ⊙r̂‖ は消える)
        y1 = y - k * a[:, None] * u[None, :]
        return (y1 @ u) / np.sqrt(np.mean(y1 * y1, axis=1) + eps) - (y @ u) / r0
    one = change(1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(one) > 1e-12, change(s) / (s * one), 1.0)


FORECAST_TEXT = "Tell me a short story about a lighthouse keeper and the sea, and how the light guided the ships home."


def projection_forecast(session, recipe, scales=(1.0,), text: str | None = None) -> list[dict]:
    """レシピの射影のユニットのうち、出力のあとに正規化がある層のものについて、短い文 1 つで、全体の倍率 k(scales)を掛けた
    ときの効きの見込み(トークンの中央値)を返す。
      corrected / uncorrected: 層が残差に足す値の r̂ の成分の変化 ÷ 元の成分(補正つき・補正なし。−1 で全部消える)
      vs_proportional: 補正つきで、比例のときに比べた効き(1 なら比例、小さいほど頭打ち)
    記録の無い古いレシピの知らせと、頭打ちの表示に使う。"""
    from .attn import capture_attention
    from .capture import pick_act_name
    from .headproj import head_direction, proj_heads
    w, e = session.weights, session.engine
    heads = [p for p in proj_heads(recipe) if p.s]
    pn = []
    for p in heads:
        li = w.arch.layers[p.layer]
        side = writer_side(p.kind)
        if (li.post_norm_mixer if side == "mixer" else li.post_norm_ffn) not in ("", "?"):
            pn.append((p, side))
    if not pn:
        return []
    text = text or FORECAST_TEXT
    ids = e.tokenize(text, add_special=True, parse_special=False)
    names = set(e.list_names(ids[:16]))
    Ls = sorted({p.layer for p, _ in pn})
    acts = {L: pick_act_name(names, L) for L in Ls if any(s == "ffn" and p.layer == L for p, s in pn)}
    att = [L for L in Ls if any(s == "mixer" and p.layer == L for p, s in pn) and w.arch.layers[L].head_dim]
    ac = capture_attention(session, [ids], layers=att, want_pattern=False, want_head_out=bool(att),
                           extra=[a for a in acts.values() if a])[0]
    T = ac.n_tokens
    eps = eps_of(w)
    out = []
    for p, side in pn:
        L = p.layer
        li = w.arch.layers[L]
        r, _ = head_direction(session.direction_set(p.direction), L, recipe.direction_layer)
        rh = np.asarray(r, np.float64) / max(float(np.linalg.norm(r)), 1e-30)
        g = gamma(w, L, side)
        v = corrected_dir(w, L, rh, side).astype(np.float64)
        if side == "mixer":
            if L not in ac.head_out:
                continue
            X = ac.head_out[L].astype(np.float64)
            WO = w.head_writer_matrix(L).astype(np.float64)
            y = X.reshape(T, -1) @ WO.T
            if p.kind == "head":
                _, lo, hi = w.head_block(L, p.head)
                yu = X[:, p.head, :] @ WO[:, lo:hi].T
            else:
                yu = y
        else:
            a_nm = acts.get(L)
            if not a_nm or a_nm not in ac.extra:
                continue
            a = ac.extra[a_nm].astype(np.float64)
            Wd = w.down_matrix(L).astype(np.float64)
            y = a @ Wd.T
            yu = np.outer(a[:, p.head], Wd[:, p.head]) if p.kind == "neuron" else y
        y, yu = y[1:], yu[1:]
        kind = li.post_norm_kind or "rms"
        base = post_add(y, g, eps, kind) @ rh
        den = np.where(np.abs(base) > 1e-12, base, np.nan)
        for k in scales:
            s_ = p.s * float(k)
            c_new = (post_add(y - s_ * np.outer(yu @ v, v), g, eps, kind) @ rh - base) / den
            c_old = (post_add(y - s_ * np.outer(yu @ rh, rh), g, eps, kind) @ rh - base) / den
            sat = predict_ratio(y, yu, v, s_, kind, eps)
            out.append({"unit": p.spec() if hasattr(p, "spec") else f"{L}:{p.head}", "layer": L, "kind": p.kind, "s": s_,
                        "scale": float(k), "corrected": float(np.nanmedian(c_new)), "uncorrected": float(np.nanmedian(c_old)),
                        "vs_proportional": float(np.nanmedian(sat))})
    return out


def forecast_text(rows: list[dict], old: bool) -> str | None:
    """見込みの知らせの文。old なら補正なしとの比較、そうでなければ頭打ち(比例のときの 0.8 倍未満)の行だけ。"""
    if not rows:
        return None
    if old:
        parts = [f"{r['unit']} s={r['s']:g}: 補正なし {r['uncorrected'] * 100:+.0f}% → 補正つき {r['corrected'] * 100:+.0f}%" for r in rows[:6]]
        return ("効きの見込み(短い文 1 つで計算した、層が残差に足す値の r̂ の成分の変化。−100% で全部消える): " + "、".join(parts)
                + (" …" if len(rows) > 6 else ""))
    sat = [r for r in rows if r["vs_proportional"] < 0.8]
    if not sat:
        return None
    parts = [f"{r['unit']} s={r['s']:g}: 比例のときの {r['vs_proportional']:.2f} 倍" for r in sat[:6]]
    return ("強さの頭打ち(出力のあとの正規化のため、強さを上げても効きが比例しない。短い文 1 つでの見込み): " + "、".join(parts)
            + (" …" if len(sat) > 6 else ""))
