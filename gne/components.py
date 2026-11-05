"""方向の差を、層ごとに書き手で分ける(gne heads writers)。

残差(流れがあれば流れの平均)は Σ_L (mixer の書き込み + FFN の書き込み) なので、A − B の差を参照の方向 r̂ に射影すると、
足し合わせて参照の層の差になる。MoE の層では FFN を、選ばれた専門家の分(ffn_moe_out)と共有の専門家の分(ffn_shexp_gated)に
分け、指定した層は専門家ごとにも分ける。測る位置は heads pipeline の点数と同じ(既定は最後のトークン)。

後ろの層での変換を見ない一次の見方で、止めたときの因果の効き目(heads causal)とは別物。どこを探すかを決めるために使う。
"""
from __future__ import annotations

import numpy as np

from .steer import GneError


def component_writes(session, ds, cs, *, ref_layer: int | None = None, positions: str = "last", subset: str = "all",
                     max_examples: int | None = 64, expert_layers: list[int] | None = None, top: int = 12, progress=None) -> dict:
    from .causal import kind_of
    from .direction import Scorer, _score_capture, base_layer_of
    from .moe import moe_names, per_token, selected
    if cs.has_recipes:
        raise GneError("BAD_ARGS", "レシピで作る対照集合には対応していません")
    e, arch = session.engine, session.weights.arch
    ref = int(ref_layer) if ref_layer is not None else base_layer_of(ds)
    if ref is None:
        raise GneError("BAD_DIRECTION", f"方向 {ds.name} の基準層が決まっていません", "--ref-layer で指定してください")
    sc = Scorer(session, ds, cs, layers=[ref], positions=positions, subset=subset, max_examples=max_examples)
    rh, hc, Ls, n_embd = sc.r[ref], e.hc_streams, list(range(ref + 1)), session.meta.n_embd
    names = e.list_names(sc.items["A"][0][0][:16])
    moe = {L: moe_names(names, L) for L in Ls if arch.layers[L].ffn.startswith("moe")}
    expert_layers = [L for L in (expert_layers or []) if L in moe]
    for L in expert_layers:
        if not {"argsort", "weighted"} <= set(moe[L]):
            raise GneError("ARCH_UNSUPPORTED", f"層 {L} の専門家の捕捉名が見つかりません")
    want = {f"l_out-{L}" for L in Ls}
    for L in Ls:
        want |= {f"hc_combine-{L}"} if hc else {f"ffn_out-{L}"}
        if L in moe:
            want |= {moe[L]["out"]} if "out" in moe[L] else set()
            want |= {f"ffn_shexp_gated-{L}"} & names
    for L in expert_layers:
        want |= {moe[L]["argsort"], moe[L]["weighted"]}
    cap = _score_capture(e)
    cap.set_wanted(lambda nm: nm in want)
    ctx = e.probe_ctx(cap, need=sc.max_len)
    e.set_lora(ctx, None)
    e.set_cvec(ctx, None)
    parts = {s: {k: np.zeros(len(Ls)) for k in ("attention", "moe", "shared")} for s in "AB"}
    expert = {s: {L: {"rate": np.zeros(arch.layers[L].n_expert), "write": np.zeros(arch.layers[L].n_expert)}
                  for L in expert_layers} for s in "AB"}
    score = {s: 0.0 for s in "AB"}
    total, done = sc.n("A") + sc.n("B"), 0
    for s in "AB":
        for ids, rows in sc.items[s]:
            e.clear_memory(ctx)
            e.prefill(ctx, [ids])
            got = cap.take()
            T = len(ids)
            rows = np.arange(T)[rows]            # 測る位置は slice のことがある
            prev = np.zeros((len(rows), n_embd))
            for L in Ls:
                if f"l_out-{L}" not in got:
                    raise GneError("ARCH_UNSUPPORTED", f"l_out-{L} が捕捉できません")
                after = got[f"l_out-{L}"][rows].astype(np.float64)
                if hc:
                    mid = got[f"hc_combine-{L}"].reshape(-1, hc, n_embd).mean(axis=1)[rows].astype(np.float64)
                elif f"ffn_out-{L}" in got:
                    mid = after - got[f"ffn_out-{L}"][rows].astype(np.float64)
                else:
                    raise GneError("ARCH_UNSUPPORTED", f"層 {L} の FFN の出力(ffn_out)が捕捉できません")
                ffn = after - mid
                nm_out = moe.get(L, {}).get("out")
                f_m = got[nm_out][rows].astype(np.float64) if nm_out in got else None
                f_s = got[f"ffn_shexp_gated-{L}"][rows].astype(np.float64) if f"ffn_shexp_gated-{L}" in got else None
                c = np.ones(len(rows))
                if f_m is not None:              # 流れの倍率の分(ffn は ffn_out の定数倍。V26)を両方に掛ける
                    f = f_m if f_s is None else f_m + f_s
                    c = np.einsum("ij,ij->i", ffn, f) / np.maximum(np.einsum("ij,ij->i", f, f), 1e-30)
                    moe_w = c[:, None] * f_m
                    shared_w = np.zeros_like(moe_w) if f_s is None else c[:, None] * f_s
                else:                            # dense の層の FFN は moe の欄に入れる
                    moe_w, shared_w = ffn, np.zeros_like(ffn)
                parts[s]["attention"][L] += float(((mid - prev) @ rh).mean())
                parts[s]["moe"][L] += float((moe_w @ rh).mean())
                parts[s]["shared"][L] += float((shared_w @ rh).mean())
                if L in expert_layers:
                    used = arch.layers[L].n_expert_used
                    sel = selected(got[moe[L]["argsort"]], used)[rows]
                    wd = per_token(got[moe[L]["weighted"]].astype(np.float64), used, T)[rows]
                    for i in range(len(rows)):
                        np.add.at(expert[s][L]["rate"], sel[i], 1.0 / len(rows))
                        np.add.at(expert[s][L]["write"], sel[i], c[i] * (wd[i] @ rh) / len(rows))
                prev = after
                if L == ref:
                    score[s] += float((after @ rh).mean())
            done += 1
            if progress:
                progress(done, total, f"heads writers: {s}")
    na, nb = sc.n("A"), sc.n("B")
    diff = {k: parts["A"][k] / na - parts["B"][k] / nb for k in ("attention", "moe", "shared")}
    gap = score["A"] / na - score["B"] / nb
    rows_out = [{"layer": L, "kind": kind_of(session, L), **{k: float(diff[k][L]) for k in diff}} for L in Ls]
    summed = float(sum(diff[k].sum() for k in diff))
    experts = {}
    for L in expert_layers:
        ra, rb = expert["A"][L]["rate"] / na, expert["B"][L]["rate"] / nb
        wa, wb = expert["A"][L]["write"] / na, expert["B"][L]["write"] / nb
        d = wa - wb
        order = np.argsort(-d)[:top]
        experts[str(L)] = [{"expert": int(x), "rate_a": float(ra[x]), "rate_b": float(rb[x]), "write_a": float(wa[x]),
                            "write_b": float(wb[x]), "diff": float(d[x])} for x in order]
    return {"direction": ds.name, "ref_layer": ref, "positions": positions, "n": {"A": na, "B": nb}, "gap": float(gap),
            "sum_error": abs(summed - gap) / max(abs(gap), 1e-12), "totals": {k: float(diff[k].sum()) for k in diff},
            "rows": rows_out, "experts": experts}
