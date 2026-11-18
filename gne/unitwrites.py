"""部品ごとの書き込みの順位づけ(`gne heads writes`)と、それを使う配分(`gne heads pipeline --rank writes`)。

1. 順位づけ(measure_writes): 対照集合の質問 1 問につき 1 回だけ prefill して、全部の層の
   - ヘッド(注意・線形注意)の出力の手前(attn_gated / kqv_out / final_output)
   - MLP の活性(ffn_down の入力)
   を捕まえ、部品ごとに「その層の方向 r̂_L にどれだけ書いたか」を出す。
     ヘッド h:     x_h · (r̂_Lᵀ W_O[:, blk_h])        (W_O は attn_output か ssm_out)
     ニューロン n: a_n · (r̂_L · W_down[:, n])
   2 本の流れ(mHC)の中の層では、副層の出力 y は 2 本に post0·y・post1·y として足され、混ぜる行列は二重確率なので、
   2 本の平均の残差の増分は c·y(c = (post0 + post1)/2)。書き込みにはこの c を掛ける(流れ 1 本の層は c = 1)。
   重みを 2 回使うモデル(sensenova_loop)で書き換えを両方の回に当てるときは、同じ重みの 2 回分を足す。
   点数(順位の鍵)は、選ぶ層 S で「s=1 の射影で直接に縮まる A と B の差」:
     点数 = Σ_回 [A 側の平均 − B 側の平均](書き込み · r̂_L) × (r̂_L · r̂_S)
   後ろの層での変換は見ない一次の見方(direct logit attribution と同じ)。後ろの書き手を動かしているだけの部品は見落とす。
2. 確かめ(verify_units): 点数の上位を 1 個ずつ s=1 で射影して(mode ablate なら出力を 0 にして)流し直し、因果スクリーニングと
   同じ物差し(S での A 側の下がり方・B 側の変化・B 側の直交成分のずれ)で測る。点数との順位相関も出す。
3. 配分(allocate_units): 確かめた部品を「A 側の下がり − B 側の動きの大きさ」の大きい順に s=1 で 1 個ずつ組に足す(組全体を
   流し直して、A 側の平均が下がり、組の効き = A 側の下がり − |B 側の変化| が増え、B 側の直交成分のずれが drift_max 以下で、
   B 側が A 側へ動く量が A と B の差の b_move_max 倍以下のときだけ採る)。目標に届けば強さは全部 1。届かなければ、組の全部の s を
   同じ倍率 α(1 ≤ α ≤ s_max)で上げる(_raise)。s > 1 の射影は成分を反転させるので、B 側の書き込みも反転して B 側が A 側へ動く
   ことがある。α ごとに A 側の平均・B 側の変化・直交成分のずれ・組の効きを記録し、目標に届けば届く最小の α、届かなければ
   上の 2 つの上限の中で組の効きがいちばん大きい α を使う(いちばん大きい α ではない)。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .headproj import ProjBuilder, ProjHead, heads_to_units, neuron_layer_ok
from .steer import GneError, Recipe

KINDS = ("attn", "linear", "neuron")


def kind_of_layer(session, L: int) -> str:
    return "attn" if session.weights.arch.layers[L].mixer == "attention" else "linear"


def _shared(session) -> bool:
    w = session.weights
    return getattr(w, "loop", None) is not None and not getattr(w, "per_visit_names", False)


def unit_layers(session, up_to: int, kinds) -> tuple[list[int], list[int]]:
    """候補の部品がある層(ヘッドの層, ニューロンの層)。重みを 2 回使うモデルで両方の回に当てるときは、1 回目の層で表す。"""
    w = session.weights
    plan = getattr(w, "loop", None)
    shared = _shared(session)
    hl, nl = [], []
    for li in w.arch.layers:
        L = li.index
        if L > up_to or (shared and plan.second(L)):
            continue
        if ("attn" in kinds or "linear" in kinds) and li.head_dim and li.head_writer:
            if kind_of_layer(session, L) in kinds:
                hl.append(L)
        if "neuron" in kinds and li.ffn == "dense" and not li.has_post_ffn_norm \
                and w.has_tensor(f"blk.{w.stored(L)}.ffn_down.weight"):
            nl.append(L)
    return hl, nl


def visits_of(session, L: int) -> list[int]:
    from .causal import _visits
    return _visits(session, L)


def parse_kinds(spec) -> tuple[str, ...]:
    """部品の種類の指定: heads(注意 + 線形注意)/ all(+ ニューロン)/ neurons / attn,linear,neuron の並び。"""
    if spec in (None, "", "all"):
        return KINDS
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s == "heads":
            return ("attn", "linear")
        if s in ("neurons", "neuron"):
            return ("neuron",)
        parts = [x.strip() for x in s.split(",") if x.strip()]
    else:
        parts = [str(x).strip().lower() for x in spec]
    bad = [x for x in parts if x not in KINDS]
    if bad or not parts:
        raise GneError("BAD_ARGS", f"部品の種類は heads / all / neurons か、attn・linear・neuron の並びです({spec!r})")
    return tuple(k for k in KINDS if k in parts)


# --- 1. 書き込みの順位づけ ---------------------------------------------------------------------------------

def measure_writes(session, ds, cs, *, select_layer: int, positions: str = "last", subset: str = "all",
                   max_examples: int | None = None, kinds=KINDS, direction_layer: str = "own", top_q: int = 256,
                   candidate_layers: list[int] | None = None, progress=None) -> tuple[dict, dict]:
    """部品ごとの書き込み(と層ごとの更新のベクトル)を測る。戻り値 (まとめ, 保存する配列)。candidate_layers を渡すと、
    順位に入れる部品をその層のものに絞る(層ごとの更新のベクトルは全部の層で測る)。"""
    from .attn import _names, _probe, head_out_names, writer_out_name
    from .capture import pick_act_name
    from .direction import Scorer
    from .reverse import combine, side_interventions
    t0 = time.perf_counter()
    e, w = session.engine, session.weights
    if e.hc_streams:
        raise GneError("ARCH_UNSUPPORTED", "残差が複数本の流れのモデル(hyper-connections)の書き込みの順位づけは未対応です")
    S = int(select_layer)
    n_layer = session.meta.n_layer
    if not 1 <= S <= n_layer - 2:
        raise GneError("BAD_ARGS", f"選ぶ層は 1〜{n_layer - 2} です(L{S})")
    kinds = tuple(kinds)
    sc = Scorer(session, ds, cs, layers=[S], positions=positions, subset=subset, max_examples=max_examples)
    sides_iv = side_interventions(session, cs) if cs.has_recipes else {"A": (None, None), "B": (None, None)}
    plan = getattr(w, "loop", None)
    shared = _shared(session)
    builder = ProjBuilder(w, session.direction_set, direction_layer=direction_layer, keep_w=4)
    rS = ds.unit(S).astype(np.float64)
    hl, nl = unit_layers(session, S - 1, kinds)
    if candidate_layers is not None:
        keep = {int(x) for x in candidate_layers}
        hl, nl = [L for L in hl if L in keep], [L for L in nl if L in keep]
        if not hl and not nl:
            raise GneError("BAD_ARGS", f"candidate_layers に、選ぶ層 L{S} より前の部品のある層がありません")
    exec_layers = list(range(n_layer - 1))               # 最後の層は行が間引かれるので測れない
    # 捕まえる名前
    items = {s: [(ids, rows) for ids, rows in sc.items[s] if len(ids) <= e.probe_ubatch] for s in "AB"}
    skipped = {s: len(sc.items[s]) - len(items[s]) for s in "AB"}
    if not items["A"] or not items["B"]:
        raise GneError("BAD_ARGS", f"質問がどれも attention の捕捉の上限 {e.probe_ubatch} トークンより長く、書き込みを測れません")
    probe, ctx = _probe(e, need=max(len(ids) for s in "AB" for ids, _ in items[s]))
    names = _names(e, probe, ctx, items["A"][0][0][:16])
    arch = w.arch
    head_exec = [L for L in exec_layers if arch.layers[L].head_dim and arch.layers[L].head_writer]
    hon = head_out_names(session, names, head_exec)
    act = {L: pick_act_name(names, L) for L in exec_layers if arch.layers[L].ffn == "dense"}
    mixn = {L: writer_out_name(session, names, L) for L in head_exec}
    # 注意の出力に名前の無い arch(llama など)は ffn_inp-L(= 層の入力 + 注意の出力)を使い、l_out-(L−1) を引く(層 0 は除く)
    mixn = {L: n for L, n in mixn.items() if n and (not n.startswith("ffn_inp") or (L > 0 and L not in mhc_set(plan, L)))}
    ffnn = {L: f"ffn_out-{L}" for L in exec_layers if f"ffn_out-{L}" in names}
    mhc = {L for L in exec_layers if plan is not None and plan.kinds[L] != 0}
    want = set(hon.values()) | {v for v in act.values() if v} | set(mixn.values()) | set(ffnn.values()) | {f"l_out-{S}"} \
        | {f"l_out-{L}" for L in exec_layers}         # 層ごとの残差の差(更新のベクトルの確かめ)
    for L in mhc:
        want |= {f"mhc_attn_gates-{L}", f"mhc_ffn_gates-{L}"}
    probe.set_wanted(lambda nm: nm in want)
    lacking_heads = [L for L in head_exec if L not in hon and (L in hl or (shared and plan is not None and plan.stored(L) in hl))]
    if lacking_heads:
        raise GneError("ARCH_UNSUPPORTED", f"層 {lacking_heads[:6]} のヘッドの出力の手前のテンソルが捕捉できません")
    nlist_exec = sorted({V for L in nl for V in visits_of(session, L)})
    lacking_act = [L for L in nlist_exec if not act.get(L)]
    if lacking_act:
        raise GneError("ARCH_UNSUPPORTED", f"層 {lacking_act[:6]} の MLP の活性(ffn_down の入力)が捕捉できません")
    scale = w.meta.ffn_down_scale or {}
    # 質問ごと: 層ごとの (c·ヘッドの出力) の位置の平均 (n_in,)、(c·活性) の位置の平均 (n_ff,)、捕まえた出力の位置の平均 (n_embd,)
    per = {s: {"mx": {L: [] for L in hon}, "ma": {L: [] for L in act if act[L]}, "mo": {L: [] for L in mixn},
               "fo": {L: [] for L in ffnn}, "lo": {L: [] for L in exec_layers}, "pS": []} for s in "AB"}
    total, done = len(items["A"]) + len(items["B"]), 0
    try:
        for side in "AB":
            sv, sl = combine(sides_iv.get(side), None, None)
            e.set_lora(ctx, sl)
            for ids, rows in items[side]:
                e.clear_memory(ctx)
                e.set_cvec(ctx, sv)
                e.prefill(ctx, [ids])
                got = probe.take()
                T = len(ids)
                rr = np.arange(T)[rows]
                for L in exec_layers:
                    ca = cm = None
                    if L in mhc:
                        ga, gf = got.get(f"mhc_attn_gates-{L}"), got.get(f"mhc_ffn_gates-{L}")
                        if ga is None or gf is None:
                            raise GneError("ARCH_UNSUPPORTED", f"層 {L} のルーターの値(2 本の流れへの書き込みの重み)が捕捉できません")
                        ca = ga[rr, :2].astype(np.float64).mean(axis=1)
                        cm = gf[rr, :2].astype(np.float64).mean(axis=1)

                    def wmean(x, c):
                        x = x[rr].astype(np.float64)
                        return (x if c is None else x * c[:, None]).mean(axis=0)
                    if L in hon:
                        a = got.get(hon[L])
                        if a is None:
                            raise GneError("ARCH_UNSUPPORTED", f"{hon[L]} が捕捉できません")
                        per[side]["mx"][L].append(wmean(a, ca).astype(np.float32))
                    if L in per[side]["ma"]:
                        a = got.get(act[L])
                        if a is None:
                            raise GneError("ARCH_UNSUPPORTED", f"{act[L]} が捕捉できません")
                        per[side]["ma"][L].append((wmean(a, cm) * scale.get(w.stored(L), 1.0)).astype(np.float32))
                    if L in mixn and mixn[L] in got:
                        per[side]["mo"][L].append(wmean(got[mixn[L]], ca).astype(np.float32))
                    if L in ffnn and ffnn[L] in got:
                        per[side]["fo"][L].append(wmean(got[ffnn[L]], cm).astype(np.float32))
                for L in exec_layers:
                    if f"l_out-{L}" in got:
                        per[side]["lo"][L].append(got[f"l_out-{L}"][rr].astype(np.float64).mean(axis=0).astype(np.float32))
                lo = got.get(f"l_out-{S}")
                if lo is None:
                    raise GneError("ARCH_UNSUPPORTED", f"l_out-{S} が捕捉できません")
                per[side]["pS"].append(float((lo[rr].astype(np.float64) @ rS).mean()))
                done += 1
                if progress:
                    progress(done, total, "heads writes: 捕捉")
    finally:
        e.set_lora(ctx, None)
        e.set_cvec(ctx, None)
        e.free_attn_probe()
    nq = {s: len(items[s]) for s in "AB"}
    half = {s: (np.arange(nq[s]) % 2 == 0) for s in "AB"}
    gap = float(np.mean(per["A"]["pS"]) - np.mean(per["B"]["pS"]))
    n_embd = session.meta.n_embd
    NL = len(exec_layers)
    arrays: dict[str, np.ndarray] = {}
    upd = {f"{k}_{x}": np.zeros((NL, n_embd), np.float32) for k in ("attn", "linear", "mlp", "layer")
           for x in ("a", "b", "diff", "diff_h0", "diff_h1")}
    chk_attn = np.full(NL, np.nan, np.float32)
    chk_mlp = np.full(NL, np.nan, np.float32)
    cos_sel = np.full(NL, np.nan, np.float32)

    stacks: dict = {}

    def stack(side, key, L):
        k_ = (side, key, L)
        if k_ not in stacks:
            if len(stacks) > 8:                        # 層ごとに使い終わるので、古いものは捨てる(メモリ)
                stacks.pop(next(iter(stacks)))
            stacks[k_] = np.stack(per[side][key][L]).astype(np.float64)
        return stacks[k_]

    def means(side, key, L):
        X = stack(side, key, L)
        return X.mean(axis=0), X[half[side]].mean(axis=0), X[~half[side]].mean(axis=0) if (~half[side]).any() else X.mean(axis=0)

    # 層ごとの更新のベクトル(正規化の前、残差に足した量。2 本の流れは平均への増分)
    for L in exec_layers:
        Ls = w.stored(L)
        if L in hon:
            Wm = builder.W(Ls if shared else L)
            k = "attn" if arch.layers[L].mixer == "attention" else "linear"
            mA, mA0, mA1 = means("A", "mx", L)
            mB, mB0, mB1 = means("B", "mx", L)
            va, vb = Wm @ mA, Wm @ mB
            upd[f"{k}_a"][L], upd[f"{k}_b"][L], upd[f"{k}_diff"][L] = va, vb, va - vb
            upd[f"{k}_diff_h0"][L], upd[f"{k}_diff_h1"][L] = Wm @ (mA0 - mB0), Wm @ (mA1 - mB1)
            if L in mixn and per["A"]["mo"][L]:
                real = stack("A", "mo", L).mean(axis=0)
                if mixn[L].startswith("ffn_inp"):           # ffn_inp − 層の入力(前の層の出力)
                    if not per["A"]["lo"][L - 1]:
                        real = None
                    else:
                        real = real - np.stack(per["A"]["lo"][L - 1]).astype(np.float64).mean(axis=0)
                if real is not None:
                    chk_attn[L] = float(np.linalg.norm(va - real) / max(np.linalg.norm(real), 1e-12))
        if L in per["A"]["ma"]:
            Wd = builder.W(Ls if shared else L, "neuron") if _dense_ok(w, L) else None
            if Wd is not None:
                mA, mA0, mA1 = means("A", "ma", L)
                mB, mB0, mB1 = means("B", "ma", L)
                va, vb = Wd @ mA, Wd @ mB
                upd["mlp_a"][L], upd["mlp_b"][L], upd["mlp_diff"][L] = va, vb, va - vb
                upd["mlp_diff_h0"][L], upd["mlp_diff_h1"][L] = Wd @ (mA0 - mB0), Wd @ (mA1 - mB1)
                if L in ffnn and per["A"]["fo"][L]:
                    real = stack("A", "fo", L).mean(axis=0)
                    chk_mlp[L] = float(np.linalg.norm(va - real) / max(np.linalg.norm(real), 1e-12))
    for x in ("a", "b", "diff", "diff_h0", "diff_h1"):
        upd[f"layer_{x}"] = upd[f"attn_{x}"] + upd[f"linear_{x}"] + upd[f"mlp_{x}"]
    # 層の更新のベクトル(部品から組み立てたもの)と、捕まえた残差の差 l_out-L − l_out-(L−1) の A 側の平均の差。2 本の流れの層で
    # 増分に掛けた c(ルーターの値)が正しいかも、ここで分かる
    chk_layer = np.full(NL, np.nan, np.float32)
    for L in exec_layers[1:]:
        if per["A"]["lo"][L] and per["A"]["lo"][L - 1] and len(per["A"]["lo"][L]) == len(per["A"]["lo"][L - 1]):
            real = (np.stack(per["A"]["lo"][L]).astype(np.float64) - np.stack(per["A"]["lo"][L - 1]).astype(np.float64)).mean(axis=0)
            chk_layer[L] = float(np.linalg.norm(upd["layer_a"][L] - real) / max(np.linalg.norm(real), 1e-12))
    # 部品ごとの書き込み
    rows_h = []                                        # (層, 番号, 種類, a_own, b_own, a_sel, b_sel)
    hq = {"A": [], "B": []}
    for L in hl:
        r, _ = builder.r_of(L, ds.name)
        r = r.astype(np.float64)
        cs_ = float(r @ rS)
        li = arch.layers[L]
        U = (r @ builder.W(L)).reshape(li.n_head, li.head_dim)
        own = {s: np.zeros((nq[s], li.n_head)) for s in "AB"}
        sel = {s: np.zeros((nq[s], li.n_head)) for s in "AB"}
        for V in visits_of(session, L):
            if V not in hon:
                continue
            for s in "AB":
                X = stack(s, "mx", V).reshape(nq[s], li.n_head, li.head_dim)
                q = (X * U[None]).sum(axis=2)
                own[s] += q
                if V <= S:
                    sel[s] += q * cs_
        for V in visits_of(session, L):
            cos_sel[V] = cs_ if np.isnan(cos_sel[V]) else cos_sel[V]
        k = kind_of_layer(session, L)
        for h in range(li.n_head):
            rows_h.append((L, h, k, own["A"][:, h].mean(), own["B"][:, h].mean(), sel["A"][:, h].mean(), sel["B"][:, h].mean()))
        hq["A"].append(own["A"])
        hq["B"].append(own["B"])
    neu = {}                                           # 層 → (a_own, b_own, a_sel, b_sel) (n_ff,)
    for L in nl:
        r, _ = builder.r_of(L, ds.name)
        r = r.astype(np.float64)
        cs_ = float(r @ rS)
        g = builder.W(L, "neuron").T.astype(np.float64) @ r       # (n_ff,)
        acc = {k: np.zeros_like(g) for k in ("ao", "bo", "as", "bs")}
        for V in visits_of(session, L):
            if V not in per["A"]["ma"]:
                continue
            mA = stack("A", "ma", V).mean(axis=0)
            mB = stack("B", "ma", V).mean(axis=0)
            acc["ao"] += mA * g
            acc["bo"] += mB * g
            if V <= S:
                acc["as"] += mA * g * cs_
                acc["bs"] += mB * g * cs_
        for V in visits_of(session, L):
            cos_sel[V] = cs_ if np.isnan(cos_sel[V]) else cos_sel[V]
        neu[L] = (acc["ao"], acc["bo"], acc["as"], acc["bs"], g, cs_)
    # 順位
    units = []                                         # (点数, 種類, 層, 番号, a_own, b_own, a_sel, b_sel)
    for L, h, k, ao, bo, as_, bs in rows_h:
        units.append((as_ - bs, k, L, h, ao, bo, as_, bs))
    big = []
    for L, (ao, bo, as_, bs, _, _) in neu.items():
        sc_ = as_ - bs
        top = np.argsort(-sc_)[:max(top_q, 64)]       # 全部を行にすると重いので、層ごとに上位だけ並べる(合計は配列で)
        big += [(float(sc_[n]), "neuron", L, int(n), float(ao[n]), float(bo[n]), float(as_[n]), float(bs[n])) for n in top]
    units += big
    units.sort(key=lambda x: -x[0])
    totals = {k: 0.0 for k in KINDS}
    for x in rows_h:
        totals[x[2]] += x[5] - x[6]
    for L, (_, _, as_, bs, _, _) in neu.items():
        totals["neuron"] += float((as_ - bs).sum())

    def label(k, L, i):
        return f"{L}:{'N' if k == 'neuron' else 'H'}{i}"
    rows = [{"label": label(k, L, i), "layer": L, "index": i, "kind": k, "score": float(sc_),
             "share": float(sc_ / gap) if abs(gap) > 1e-12 else None,
             "own": float(ao - bo), "a_own": float(ao), "b_own": float(bo), "a_sel": float(as_), "b_sel": float(bs)}
            for sc_, k, L, i, ao, bo, as_, bs in units]
    # 質問ごとの値(上位の部品)
    topu = [r for r in rows[:top_q]]
    tq = {"A": np.zeros((nq["A"], len(topu)), np.float32), "B": np.zeros((nq["B"], len(topu)), np.float32)}
    hidx = {(L, h): j for j, (L, h, *_x) in enumerate(rows_h)}
    HQ = {s: (np.concatenate(hq[s], axis=1) if hq[s] else np.zeros((nq[s], 0))) for s in "AB"}
    for j, r_ in enumerate(topu):
        if r_["kind"] != "neuron":
            for s in "AB":
                tq[s][:, j] = HQ[s][:, hidx[(r_["layer"], r_["index"])]]
        else:
            L, n = r_["layer"], r_["index"]
            g = neu[L][4]
            for s in "AB":
                v = np.zeros(nq[s])
                for V in visits_of(session, L):
                    if V in per[s]["ma"]:
                        v += stack(s, "ma", V)[:, n] * g[n]
                tq[s][:, j] = v
    lp = plan
    arrays.update({
        "layer_index": np.array(exec_layers, np.int32),
        "layer_mixer": np.array([arch.layers[L].mixer for L in exec_layers]),
        "layer_visit": np.array([lp.visit(L) if lp is not None else 0 for L in exec_layers], np.int32),
        "layer_stored": np.array([w.stored(L) for L in exec_layers], np.int32),
        "cos_to_select": cos_sel, "check_attn_rel": chk_attn, "check_mlp_rel": chk_mlp, "check_layer_rel": chk_layer,
        "select_layer": np.int32(S), "gap": np.float32(gap), "n_a": np.int32(nq["A"]), "n_b": np.int32(nq["B"]),
        **{f"upd_{k}": v for k, v in upd.items()},
        "head_layer": np.array([x[0] for x in rows_h], np.int32), "head_index": np.array([x[1] for x in rows_h], np.int32),
        "head_kind": np.array([x[2] for x in rows_h]),
        "head_a_own": np.array([x[3] for x in rows_h], np.float32), "head_b_own": np.array([x[4] for x in rows_h], np.float32),
        "head_a_sel": np.array([x[5] for x in rows_h], np.float32), "head_b_sel": np.array([x[6] for x in rows_h], np.float32),
        "head_q_a": HQ["A"].astype(np.float32), "head_q_b": HQ["B"].astype(np.float32),
        "top_label": np.array([r_["label"] for r_ in topu]), "top_q_a": tq["A"], "top_q_b": tq["B"],
    })
    for L, (ao, bo, as_, bs, _, _) in neu.items():
        arrays[f"neuron_a_own_{L}"] = ao.astype(np.float32)
        arrays[f"neuron_b_own_{L}"] = bo.astype(np.float32)
        arrays[f"neuron_a_sel_{L}"] = as_.astype(np.float32)
        arrays[f"neuron_b_sel_{L}"] = bs.astype(np.float32)
    by_kind_n = {k: sum(1 for x in rows_h if x[2] == k) for k in ("attn", "linear")}
    by_kind_n["neuron"] = int(sum(v[0].size for v in neu.values()))
    summary = {"direction": ds.name, "select_layer": S, "positions": positions, "subset": subset,
               "max_examples": max_examples, "kinds": list(kinds), "direction_layer": direction_layer,
               "n": nq, "skipped_long": skipped, "gap": gap,
               "base": {"A": float(np.mean(per["A"]["pS"])), "B": float(np.mean(per["B"]["pS"]))},
               "n_units": by_kind_n, "head_layers": hl, "neuron_layers": nl,
               "loop_edits": ("shared" if shared else "per_visit") if plan is not None else None,
               "totals": {k: v for k, v in totals.items()},
               "totals_share": {k: (v / gap if abs(gap) > 1e-12 else None) for k, v in totals.items()},
               "check": {"attn_rel_max": float(np.nanmax(chk_attn)) if np.isfinite(chk_attn).any() else None,
                         "mlp_rel_max": float(np.nanmax(chk_mlp)) if np.isfinite(chk_mlp).any() else None,
                         "layer_rel_median": float(np.nanmedian(chk_layer)) if np.isfinite(chk_layer).any() else None,
                         "layer_rel_max": float(np.nanmax(chk_layer)) if np.isfinite(chk_layer).any() else None},
               "rows": rows, "seconds": time.perf_counter() - t0}
    return summary, arrays


def mhc_set(plan, L: int) -> set:
    """2 本の流れの中の層なら {L}(ffn_inp は使えない)、ほかは空。"""
    return {L} if plan is not None and plan.kinds[L] != 0 else set()


def _dense_ok(w, L: int) -> bool:
    try:
        neuron_layer_ok(w, L)
        return True
    except GneError:
        return False


# --- 2. 確かめ(1 個ずつ流し直す)---------------------------------------------------------------------------

def unit_of(row: dict, s: float, direction: str) -> ProjHead:
    return ProjHead(int(row["layer"]), int(row["index"]), float(s), direction,
                    "neuron" if row["kind"] == "neuron" else "head")


def ablate_unit_tensors(st, row: dict) -> dict:
    """部品の出力を 0 にする LoRA(ヘッドは列ブロック、ニューロンは ffn_down の 1 列)。"""
    L, i = int(row["layer"]), int(row["index"])
    if row["kind"] != "neuron":
        return st.builder.ablate_tensors(L, [i])
    W = st.builder.W(L, "neuron")
    A = np.zeros((1, W.shape[1]), np.float32)
    A[0, i] = 1.0
    return {st.builder.block(L, i, "neuron")[0]: (A, (-W[:, i:i + 1]).astype(np.float32))}


def verify_rows(rows: list[dict], top: int, sample: int = 0) -> list[dict]:
    """確かめる部品: 点数が正の上位 top 個と、その下(top〜4·top 位)から等間隔に sample 個(順位相関を上位だけの狭い範囲で
    測らないため。sample の部品も組の候補になる)。"""
    pos = [r for r in rows if r["score"] > 0]
    out = [dict(r, sampled=False) for r in pos[:max(1, int(top))]]
    rest = pos[int(top):4 * int(top)]
    if sample and rest:
        idx = sorted({int(i) for i in np.linspace(0, len(rest) - 1, min(int(sample), len(rest))).round()})
        out += [dict(rest[i], sampled=True) for i in idx]
    return out


def verify_units(session, st, rows: list[dict], *, mode: str = "proj", progress=None) -> dict:
    """上位の部品を 1 個ずつ s=1 で射影して(ablate なら出力を 0 にして)、因果スクリーニングと同じ物差しで測る。"""
    from .causal import _row_common, spearman
    if mode not in ("proj", "ablate"):
        raise GneError("BAD_ARGS", "確かめ方は proj(s=1 で射影)か ablate(出力を 0)です")
    out = []
    for i, r in enumerate(rows):
        t0 = time.perf_counter()
        tens = st.builder.tensors([unit_of(r, 1.0, st.ds.name)]) if mode == "proj" else ablate_unit_tensors(st, r)
        with st.tmp.use(tens) as lora:
            got = st.measure(lora)
        m = _row_common(st, got, st.base)
        m["gap_drop"] = m["causal"] + (m.get("b_side_delta") or 0.0)   # A と B の差の縮み = A の下がり − B の下がり
        # 組に入れる順の鍵: A 側の下がり − B 側の動きの大きさ。B 側が A 側の向きに動く(差は縮むが、ふつうの入力が振る舞いの側へ
        # 寄る)部品を、差の縮みだけで上に置かないため
        m["effect"] = m["causal"] - abs(m.get("b_side_delta") or 0.0)
        out.append({**{k: r[k] for k in ("label", "layer", "index", "kind", "score", "share", "own")},
                    "sampled": bool(r.get("sampled")), "mode": mode, **m,
                    "excluded": bool((m.get("drift_orth_b") or 0) > 0.5), "seconds": time.perf_counter() - t0})
        if progress:
            progress(i + 1, len(rows), f"heads writes: 確かめ {r['label']}")
    corr = {}
    for k in ("all", *KINDS):
        sel = [x for x in out if k == "all" or x["kind"] == k]
        if len(sel) >= 3:
            corr[k] = {"n": len(sel), "score_vs_gap_drop": spearman([x["score"] for x in sel], [x["gap_drop"] for x in sel]),
                       "score_vs_a_drop": spearman([x["score"] for x in sel], [x["causal"] for x in sel]),
                       "score_vs_effect": spearman([x["score"] for x in sel], [x["effect"] for x in sel])}
    return {"mode": mode, "rows": out, "spearman": corr}


# --- 3. 配分(s=1 から)-----------------------------------------------------------------------------------

B_MOVE_MAX = 0.10        # B 側が A 側へ動いてよい量(A と B の差に対する割合)。公開のひな形での値から決めた(docs §3.9)
GAIN_TOL = 0.01          # 倍率を上げて「組の効きが良くなった」とみなす最小の増え方(A と B の差に対する割合)
GAIN_PATIENCE = 2        # 良くならない格子の点がこの数だけ続いたら、それ以上は上げない


class SetScore:
    """組の物差し(A 側の下がり・B 側の変化・直交成分のずれ → 組の効きと上限)。base_* は無介入の値。"""

    def __init__(self, base_a: float, base_b: float, drift_max: float, b_move_max: float):
        self.base_a, self.base_b = float(base_a), float(base_b)
        self.gap = abs(self.base_a - self.base_b) or 1.0
        self.sign = 1.0 if self.base_a >= self.base_b else -1.0
        self.drift_max, self.b_move_max = float(drift_max), float(b_move_max)

    def point(self, alpha: float, a_mean: float, b_mean: float, b_orth: float) -> dict:
        b_delta = float(b_mean) - self.base_b
        toward = self.sign * b_delta / self.gap                     # B 側が A 側へ動いた量(差に対する割合。正 = A 側へ)
        why = "b_move" if toward > self.b_move_max else "drift" if b_orth > self.drift_max else None
        return {"alpha": float(alpha), "a_mean": float(a_mean), "b_delta": b_delta, "b_toward": toward,
                "b_orth": float(b_orth), "effect": (self.base_a - float(a_mean)) * self.sign - abs(b_delta),
                "feasible": why is None, "over": why}


def allocate_units(session, st, verified: list[dict], *, target: float, n_units=(24, 12, 6), s_max: float = 30.0,
                   drift_max: float = 0.5, b_move_max: float = B_MOVE_MAX, name: str, direction_layer: str = "own",
                   remain_threshold: float | None = None, max_examples: int | None = None, progress=None) -> dict:
    """確かめた部品で組を作る(モジュールの説明の 3)。表の形は heads allocate と同じ(n_heads = 組の部品の数の上限)。
    表とは別に、倍率を上げる前の組(採った部品を全部 s=1)を s1 に入れる。"""
    from .direction import drift
    t_all = time.perf_counter()
    S = st.primary
    ds = st.ds
    a_all = list(range(st.scorer.n("A")))
    thr = float(target) if remain_threshold is None else float(remain_threshold)
    base_a = float(st.base["A"][S].mean())
    b_base = st.scorer.run(None, sides="B")["B"]["values"]
    sc = SetScore(base_a, float(st.base["B"][S].mean()), drift_max, b_move_max)
    steps = [0]

    def tick(msg):
        steps[0] += 1
        if progress:
            progress(steps[0], max(steps[0], est), msg)

    def run(units: list[ProjHead], alpha: float = 1.0) -> dict:
        with st.tmp.use(st.builder.tensors([u for u in units if u.s])) as lora:
            got = st.scorer.run(lora, sides="A", which={"A": a_all})
            gb = st.scorer.run(lora, sides="B", which={"B": st.b_idx}, keep=True)
        tick("allocate")
        orth = drift(st.base["B_resid"][S], gb["B"]["resid"][S], st.scorer.r[S])[1]
        return sc.point(alpha, float(got["A"]["values"][S].mean()), float(gb["B"]["values"][S].mean()), orth)
    pool = sorted([r for r in verified if not r.get("excluded") and (r.get("effect") or 0) > 0],
                  key=lambda r: -r["effect"])
    nmax = max(int(x) for x in n_units)
    est = len(pool) + len(n_units) * 16 + 1
    # s=1 で 1 個ずつ足す(A 側の平均が下がり、組の効きが増え、上限の中のときだけ)
    chosen: list[ProjHead] = []
    trace = []
    at1: list[dict] = []                         # 採った i+1 個の組の s=1 での点
    cur = sc.point(1.0, base_a, sc.base_b, 0.0)
    for r in pool:
        if len(chosen) >= nmax or cur["a_mean"] <= target:
            break
        u = unit_of(r, 1.0, ds.name)
        g = run(chosen + [u])
        ok = g["a_mean"] < cur["a_mean"] - 1e-6 and g["effect"] > cur["effect"] and g["feasible"]
        trace.append({"label": r["label"], "kind": r["kind"], "accepted": bool(ok), "a_mean": g["a_mean"],
                      "b_delta": g["b_delta"], "b_drift_orth": g["b_orth"], "effect": g["effect"],
                      **({"over": g["over"]} if g["over"] else {})})
        if ok:
            chosen.append(u)
            cur = g
            at1.append(g)

    def finish(units: list[ProjHead], alpha: float, label: str) -> dict:
        heads = [ProjHead(u.layer, u.head, alpha, u.direction, u.kind) for u in units]
        with st.tmp.use(st.builder.tensors(heads)) as lora:
            got = st.scorer.run(lora, sides="AB", which={"A": a_all})
            gb = st.scorer.run(lora, sides="B", which={"B": st.b_idx}, keep=True)
        va = got["A"]["values"][S]
        par, orth = drift(st.base["B_resid"][S], gb["B"]["resid"][S], st.scorer.r[S])
        rec = Recipe(heads_to_units(heads, ds.name), session.meta.model_id, name=f"{name}-{label}", default_direction=ds.name,
                     direction_layer=direction_layer)
        rpath = session.store.heads_dir(session.meta.model_id) / f"{name}.alloc-{label}.yaml"
        rec.save(rpath)
        ss = [p.s for p in heads]
        bm = float(got["B"]["values"][S].mean())
        pt = SetScore(base_a, float(b_base[S].mean()), drift_max, b_move_max).point(alpha, float(va.mean()), bm, orth)
        return {"n_used": len(heads),
                "heads": [{"layer": p.layer, "head": p.head, "kind": p.kind if p.kind == "neuron" else
                           kind_of_layer(session, p.layer), "s": p.s} for p in heads],
                "kinds": {k: sum(1 for p in heads if (p.kind == "neuron" and k == "neuron") or
                                 (p.kind == "head" and kind_of_layer(session, p.layer) == k)) for k in KINDS},
                "units": [p.spec() for p in heads], "s_max": max(ss), "s_median": float(np.median(ss)), "alpha": alpha,
                "hard_mean": float(va.mean()), "remain": int((va > thr).sum()), "a_max": float(va.max()),
                "a_mean": float(va.mean()), "b_mean": bm, "b_side_delta": pt["b_delta"], "b_toward": pt["b_toward"],
                "effect": pt["effect"], "feasible": pt["feasible"],
                "closed": (base_a - float(va.mean())) / (base_a - target) if abs(base_a - target) > 1e-9 else None,
                "b_drift_par": par, "b_drift_orth": orth, "recipe": str(rpath),
                "by_layer": {str(L): {"A": float(got["A"]["values"][L].mean()), "B": float(got["B"]["values"][L].mean())}
                             for L in st.score_layers}}

    table = []
    for n in sorted({int(x) for x in n_units}, reverse=True):
        t0 = time.perf_counter()
        units = chosen[:n]
        if not units:
            table.append({"n_heads": n, "n_units": n, "skipped": "s=1 で射影の値を下げる部品がありません"})
            continue
        prev = next((r for r in table if r.get("n_used") == len(units) and not r.get("skipped")), None)
        if prev is not None:                        # 上限を下げても組が同じ(採った部品がそれより少ない)
            table.append({"n_heads": n, "n_units": n, "skipped": f"{prev['n_heads']} 個の上限の組と同じ({len(units)} 個)",
                          "same_as": prev["n_heads"]})
            continue
        p1 = at1[len(units) - 1]
        alpha, stop, n_iter, rtrace = 1.0, "target", 0, [dict(p1, phase="s1")]
        if p1["a_mean"] > target:
            alpha, stop, rtrace = _raise(lambda a, units=units: run([ProjHead(u.layer, u.head, a, u.direction, u.kind)
                                                                     for u in units], a),
                                         p1, target, s_max, sc)
            n_iter = len(rtrace) - 1
        row = finish(units, alpha, f"n{n}")
        table.append({"n_heads": n, "n_units": n, **row,
                      "reached": bool(row["a_mean"] <= target + 1e-6 or stop in ("target", "raised")),
                      "iterations": n_iter, "stop": stop, "capped": None, "raise": rtrace,
                      "seconds": time.perf_counter() - t0})
    s1 = None
    if chosen:                                      # 倍率を上げる前の組(取り除くだけでどこまで届くか)
        t0 = time.perf_counter()
        s1 = {"n_heads": len(chosen), **finish(chosen, 1.0, "s1"), "seconds": time.perf_counter() - t0}
        s1["reached"] = bool(s1["a_mean"] <= target + 1e-6)
    rec_n, why, no_rec = _recommend(table, len(a_all), sc)
    for r in table:
        r["recommended"] = r.get("n_heads") == rec_n and rec_n is not None
    out = {"name": name, "direction": ds.name, "contrast_sha": st.cs.sha, "score_layer": S, "score_layers": st.score_layers,
           "max_examples": max_examples, "drift_max": drift_max, "b_move_max": b_move_max, "target": float(target),
           "positions": st.scorer.positions, "subset": st.scorer.subset, "direction_layer": direction_layer,
           "method": "unit1", "s_max": s_max, "remain_threshold": thr, "n_a": len(a_all), "n_b": len(st.b_idx),
           "base_a_mean": base_a, "base_b_mean": float(st.base["B"][S].mean()), "gap": sc.gap,
           "base_by_layer": {str(L): {"A": float(st.base["A"][L].mean()), "B": float(b_base[L].mean())} for L in st.score_layers},
           "initial": None, "pool": len(pool), "trace": trace, "table": table, "s1": s1, "recommended": rec_n,
           "recommended_by": why, "no_recommend": no_rec, "seconds": time.perf_counter() - t_all}
    if progress:
        progress(steps[0], steps[0], "allocate")
    p = session.store.heads_dir(session.meta.model_id) / f"{name}.alloc.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    out["path"] = str(p)
    return out


def _recommend(table: list[dict], n_a: int, sc: SetScore) -> tuple[int | None, str | None, dict | None]:
    """推す組の上限の数・推した理由(reached / best_effect)・推さないときの理由。
    1. 届いた組があれば heads allocate と同じ規則(残る数が最少から A 側の 5% 以内、B 側のずれが最小の 1.25 倍以内で、数が最少)。
    2. どれも届かなければ、上限の中で組の効き(A 側の下がり − |B 側の変化|)が正の組のうち、効きが最大から差の 5% 以内で数が最少。
    3. どちらも無ければ推さない(理由を返す)。"""
    rows = [r for r in table if not r.get("skipped")]
    ok = [r for r in rows if r.get("reached")]
    if ok:
        r_min = min(r["remain"] for r in ok)
        tol_n = max(1, int(np.ceil(0.05 * n_a)))
        near = [r for r in ok if r["remain"] <= r_min + tol_n]
        d_min = min(r["b_drift_orth"] for r in near)
        same = [r for r in near if r["b_drift_orth"] <= d_min * 1.25 + 1e-4]
        return min(r["n_heads"] for r in same), "reached", None
    good = [r for r in rows if r.get("feasible", True) and (r.get("effect") or 0) > 0]
    if good:
        e_max = max(r["effect"] for r in good)
        same = [r for r in good if r["effect"] >= e_max - 0.05 * sc.gap]
        return min(r["n_heads"] for r in same), "best_effect", None
    if not rows:
        reason = "s=1 で A 側の射影の値を下げる部品がない"
    else:
        reason = "どの組も、A 側の下がりが B 側の動きより小さいか、B 側の動き・直交成分のずれの上限を超える"
    return None, None, {"reason": reason, "next": ["rank causal", "unit_kinds", "refine"]}


def _raise(run_at, p1: dict, target: float, s_max: float, sc: SetScore, *, n_grid: int = 10,
           iters: int = 8) -> tuple[float, str, list[dict]]:
    """組の全部の s を同じ倍率 α で上げる。α は 1 から s_max まで等比の格子で上にたどり(α と射影の値の関係は単調とは限らない)、
    点ごとに A 側の平均・B 側の変化・直交成分のずれ・組の効き(SetScore.point)を記録する。
    - 目標に届き、上限の中 → 手前の点とのあいだを二分探索して、届く最小の α(raised)
    - 上限を超えた(B 側が A 側へ動く b_move / 直交成分のずれ drift)→ 手前の点とのあいだで上限の中の最大の α を探し、そこで止める
    - 組の効きが差の GAIN_TOL 倍より増えない点が GAIN_PATIENCE 個続いた → no_gain
    - 格子の終わり → s_max
    届かずに止めたときは、測った上限の中の点のうち組の効きがいちばん大きい α を返す(いちばん大きい α ではない。増え方が
    GAIN_TOL 倍以下なら低い α を残す)。p1 は α=1 の点。戻り値 (α, 止まった理由, 測った点の並び)。"""
    tol = GAIN_TOL * sc.gap
    pts = [float(x) for x in np.geomspace(1.0, max(s_max, 1.0), n_grid + 1)[1:]]
    trace = [dict(p1, phase="s1")]
    best = p1 if p1["feasible"] else None
    prev, bad = 1.0, 0

    def better(p):
        return p["feasible"] and (best is None or p["effect"] > best["effect"] + tol)

    def shrink(lo: float, hi: float) -> float:
        """[lo, hi] で(hi は目標に届き、上限の中)、届く最小の α を二分探索で詰める。"""
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            pm = dict(run_at(mid), phase="shrink")
            trace.append(pm)
            if pm["a_mean"] <= target and pm["feasible"]:
                hi = mid
            else:
                lo = mid
            if hi - lo < 0.02 * hi:
                break
        return hi
    stop = "s_max"
    for a in pts:
        p = dict(run_at(a), phase="grid")
        trace.append(p)
        if not p["feasible"]:                       # 上限を超えた: 手前の点とのあいだで、上限の中の最大の α
            lo, hi = prev, a
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                pm = dict(run_at(mid), phase="edge")
                trace.append(pm)
                if not pm["feasible"]:
                    hi = mid
                    continue
                lo = mid
                if pm["a_mean"] <= target:          # 上限の手前で目標に届いたら、届く最小の α を探す
                    return shrink(prev, mid), "raised", trace
                if better(pm):
                    best = pm
            stop = p["over"]
            break
        if p["a_mean"] <= target:
            return shrink(prev, a), "raised", trace
        if better(p):
            best, bad = p, 0
        else:
            bad += 1
            if bad >= GAIN_PATIENCE:
                stop = "no_gain"
                break
        prev = a
    return (best["alpha"] if best else 1.0), stop, trace


# --- 保存 -------------------------------------------------------------------------------------------------

def save_writes(session, name: str, summary: dict, arrays: dict, verify: dict | None = None,
                show: int = 200) -> tuple[Path, Path]:
    """<root>/heads/<model_id>/<name>.writes.npz(配列)と .writes.json(まとめ。行は上位 show 個と種類ごとの上位 50 個)。"""
    d = session.store.heads_dir(session.meta.model_id)
    d.mkdir(parents=True, exist_ok=True)
    pz, pj = d / f"{name}.writes.npz", d / f"{name}.writes.json"
    np.savez_compressed(pz, **arrays)
    rows = summary["rows"]
    keep = rows[:show]
    seen = {r["label"] for r in keep}
    for k in KINDS:
        for r in [x for x in rows if x["kind"] == k][:50]:
            if r["label"] not in seen:
                keep.append(r)
                seen.add(r["label"])
    js = {**{k: v for k, v in summary.items() if k != "rows"}, "rows": keep, "n_rows_total": len(rows),
          **({"verify": verify} if verify is not None else {}), "npz": pz.name, "name": name,
          "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    pj.write_text(json.dumps(js, ensure_ascii=False, indent=1), encoding="utf-8")
    return pz, pj
