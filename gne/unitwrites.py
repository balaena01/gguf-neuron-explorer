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

from .headproj import TENSOR_KIND, WHOLE, ProjBuilder, ProjHead, heads_to_units, neuron_layer_ok
from .steer import GneError, Recipe

KINDS = ("attn", "linear", "neuron")
LAYER_KINDS = ("layer_attn", "layer_mlp")     # 層の書き込み全体(注意の出力全体・MLP の出力全体)を 1 つの部品にする(v0.8.1)
ALL_KINDS = KINDS + LAYER_KINDS
GROUPS = {"heads": ("attn", "linear"), "neurons": ("neuron",), "neuron": ("neuron",), "all": KINDS, "layers": LAYER_KINDS}
UNIT_KIND = {"neuron": "neuron", "layer_attn": "attn", "layer_mlp": "ffn"}   # 行の種類 → ProjHead の kind(ほかは head)


def kind_of_layer(session, L: int) -> str:
    return "attn" if session.weights.arch.layers[L].mixer == "attention" else "linear"


def _shared(session) -> bool:
    w = session.weights
    return getattr(w, "loop", None) is not None and not getattr(w, "per_visit_names", False)


def unit_layers(session, up_to: int, kinds, loop_direction: str = "first") -> tuple[list[int], list[int]]:
    """候補の部品がある層(ヘッドの層, ニューロンの層)。重みを 2 回使うモデルで両方の回に当てるときは、loop_direction が
    first なら 1 回目の段、second なら 2 回目の段で表す(部品の番号の段の方向 r̂ で測り、射影する。重みは同じ)。
    層全体の部品(layer_attn・layer_mlp)を求めたときは、その層のヘッド・ニューロンも測る(層全体の書き込みはその和)。"""
    if loop_direction not in ("first", "second"):
        raise GneError("BAD_ARGS", f"loop_direction は first(1 回目の段の方向)か second(2 回目の段の方向)です({loop_direction!r})")
    w = session.weights
    plan = getattr(w, "loop", None)
    shared = _shared(session)
    hl, nl = [], []
    for li in w.arch.layers:
        L = li.index
        if shared and plan.second(L):
            continue
        rep = L + plan.shift if shared and loop_direction == "second" and plan.visit(L) == 1 else L
        if rep > up_to:
            continue
        if ("attn" in kinds or "linear" in kinds or "layer_attn" in kinds) and li.head_dim and li.head_writer:
            if kind_of_layer(session, L) in kinds or "layer_attn" in kinds:
                hl.append(rep)
        if ("neuron" in kinds or "layer_mlp" in kinds) and li.ffn == "dense" and not li.post_norm_ffn \
                and w.has_tensor(f"blk.{w.stored(L)}.ffn_down.weight"):
            nl.append(rep)
    return sorted(hl), sorted(nl)


def visits_of(session, L: int) -> list[int]:
    from .causal import _visits
    return _visits(session, L)


def parse_kinds(spec) -> tuple[str, ...]:
    """部品の種類の指定: heads(注意 + 線形注意)/ all(+ ニューロン)/ neurons / layers(層の注意の出力全体・MLP の出力全体)か、
    attn・linear・neuron・layer_attn・layer_mlp とこれらの名前のカンマ区切り(例 heads,layers)。"""
    if spec in (None, ""):
        return KINDS
    parts = [x.strip() for x in spec.lower().split(",") if x.strip()] if isinstance(spec, str) else \
        [str(x).strip().lower() for x in spec]
    got = set()
    for x in parts:
        if x in GROUPS:
            got |= set(GROUPS[x])
        elif x in ALL_KINDS:
            got.add(x)
        else:
            got = None
            break
    if not got:
        raise GneError("BAD_ARGS", f"部品の種類は heads / all / neurons / layers か、attn・linear・neuron・layer_attn・layer_mlp の"
                                   f"カンマ区切りです({spec!r})")
    return tuple(k for k in ALL_KINDS if k in got)


# --- 1. 書き込みの順位づけ ---------------------------------------------------------------------------------

def _zero_direction_layers(builder, ds, layers) -> list[int]:
    """方向の長さが 0 の層(A と B の残差の差が 0。短い畳み込みの層の最初の数層など)。head_direction が断る層。"""
    out = []
    for L in layers:
        try:
            builder.r_of(L, ds.name)
        except GneError as e:
            if getattr(e, "zero_layer", None) is None:
                raise
            out.append(int(L))
    return out


def measure_writes(session, ds, cs, *, select_layer: int, positions: str = "last", subset: str = "all",
                   max_examples: int | None = None, kinds=KINDS, direction_layer: str = "own", top_q: int = 256,
                   candidate_layers: list[int] | None = None, loop_direction: str = "first",
                   keep_q: bool = False, progress=None) -> tuple[dict, dict]:
    """部品ごとの書き込み(と層ごとの更新のベクトル)を測る。戻り値 (まとめ, 保存する配列)。candidate_layers を渡すと、
    順位に入れる部品をその層のものに絞る(層ごとの更新のベクトルは全部の層で測る)。loop_direction は unit_layers。
    keep_q なら、質問ごと・層ごとの更新(注意の出力全体と MLP の出力全体)も q_* の配列で返す(gne heads geometry の材料。
    float16 で、層ごとに最大の絶対値が 1024 になるよう q_scale_* で割ってある)。"""
    from .attn import _names, _probe, head_out_names, writer_out_name
    from .capture import pick_act_name
    from .direction import Scorer
    from .reverse import combine, side_interventions
    t0 = time.perf_counter()
    e, w = session.engine, session.weights
    if e.hc_streams:
        raise GneError("ARCH_UNSUPPORTED", "残差が複数本の流れのモデル(hyper-connections)の書き込みの順位づけは未対応です")
    S = int(select_layer)
    n_layer = session.weights.n_main                    # MTP(nextn)の層は本流で実行されないので数えない
    if not 1 <= S <= n_layer - 2:
        raise GneError("BAD_ARGS", f"選ぶ層は 1〜{n_layer - 2} です(L{S})")
    kinds = tuple(kinds)
    sc = Scorer(session, ds, cs, layers=[S], positions=positions, subset=subset, max_examples=max_examples)
    sides_iv = side_interventions(session, cs) if cs.has_recipes else {"A": (None, None), "B": (None, None)}
    plan = getattr(w, "loop", None)
    shared = _shared(session)
    builder = ProjBuilder(w, session.direction_set, direction_layer=direction_layer, keep_w=4)
    rS = ds.unit(S).astype(np.float64)
    hl, nl = unit_layers(session, S - 1, kinds, loop_direction)
    pn = [L for L in hl if w.arch.layers[w.stored(L)].post_norm_mixer not in ("", "?")]
    if pn:                                           # 出力のあとの正規化: 書き込みの量の点数が残差での寄与を表さない(B3 の補正までは実行しない)
        raise GneError("ARCH_UNSUPPORTED", f"層 {', '.join(map(str, pn[:6]))}{' …' if len(pn) > 6 else ''} は注意(混ぜる層)の出力のあとに正規化が"
                       "あるので、書き込みの量での順位づけは実行しません。理由: 点数(正規化の前の書き込み)が残差での寄与を表さない"
                       "ためです(LFM3-1.7B・Gemma 3 1B で測った、正規化のあとの寄与との順位相関は中央値 0.85、最小 −0.12)",
                       "代わりに --rank causal(止めて選ぶ)を使ってください。正規化を補正した点数が次の版で入るまでの扱いです")
    pu = [L for L in hl if w.arch.layers[w.stored(L)].post_norm_mixer == "?"]
    if pu:                                           # 表に無い arch の post_attention_norm(MLP の手前か、出力のあとか、確かめていない)
        session.warn(f"層 {', '.join(map(str, pu[:6]))} の post_attention_norm が、注意の出力のあとの正規化か MLP の手前の正規化か、"
                     "この arch では確かめていません。出力のあとなら、書き込みの量の点数は残差での寄与を表しません")
    if candidate_layers is not None:                  # 重みを 2 回使うモデルでは、どちらの回の段で書いても同じ重みを指す
        keep = {int(x) for x in candidate_layers}
        if shared:
            keep |= {plan.stored(x) for x in keep} | {plan.stored(x) + plan.shift for x in keep if plan.visit(plan.stored(x)) == 1}
        hl, nl = [L for L in hl if L in keep], [L for L in nl if L in keep]
        if not hl and not nl:
            raise GneError("BAD_ARGS", f"candidate_layers に、選ぶ層 L{S} より前の部品のある層がありません")
    exec_layers = list(range(n_layer - 1))               # 最後の層は行が間引かれるので測れない
    # 捕まえる名前
    items = {s: [(ids, rows) for ids, rows in sc.items[s] if len(ids) <= e.probe_ubatch] for s in "AB"}
    skipped = {s: len(sc.items[s]) - len(items[s]) for s in "AB"}
    qidx = {s: [sc.index_side[s][j] for j, (ids, _) in enumerate(sc.items[s]) if len(ids) <= e.probe_ubatch] for s in "AB"}
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
    qup = {f"{k}_{x.lower()}": np.zeros((nq[x], NL, n_embd), np.float16) for k in ("attn", "mlp") for x in "AB"} if keep_q else None
    qsc = {k: np.zeros(NL, np.float32) for k in qup} if keep_q else None

    def keep_q_rows(k, L, W, key):
        """質問ごとの更新(W · 質問ごとの出力の平均)を float16 で持つ。層ごとに最大の絶対値が 1024 になるよう割る(桁あふれを避ける)。"""
        for x in "AB":
            Q = stack(x, key, L) @ W.T
            m = float(np.abs(Q).max()) / 1024.0 or 1.0
            qup[f"{k}_{x.lower()}"][:, L] = (Q / m).astype(np.float16)
            qsc[f"{k}_{x.lower()}"][L] = m
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
            if qup is not None:
                keep_q_rows("attn", L, Wm, "mx")
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
                if qup is not None:
                    keep_q_rows("mlp", L, Wd, "ma")
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
    # 層ごとの直接の書き込み(注意全体・MLP 全体を s=1 で射影したときに選ぶ層で縮む A−B の差。層の更新のベクトルから。部品の種類の
    # 指定に関係なく出す。重みを 2 回使うモデルで両方の回に当てるときは、同じ重みの 2 回分を、部品の番号の段の方向で足す)
    lay: dict[int, list[float]] = {}
    zero = _zero_direction_layers(builder, ds, sorted(set(exec_layers) | set(hl) | set(nl)))   # A と B の差が 0 の層(v0.9.0)
    if zero:
        session.warn(f"方向 {ds.name} の層 {', '.join(map(str, zero[:8]))}{' …' if len(zero) > 8 else ''} は長さが 0 です"
                     "(A 側と B 側の残差がまったく同じ層。短い畳み込みの層は数トークン前までしか見ないので、両側で違う system "
                     "プロンプトがまだ届いていません)。この層では方向に沿った書き込みが決まらないので、層ごとの書き込みの表と部品の候補から"
                     "外しました")
    hl = [L for L in hl if L not in zero]
    nl = [L for L in nl if L not in zero]
    for L in exec_layers:
        if L >= S or not (arch.layers[L].head_dim or arch.layers[L].ffn == "dense"):
            continue
        rep = L
        if shared and plan.visit(L) != 0:
            st_ = plan.stored(L)
            rep = st_ + plan.shift if loop_direction == "second" else st_
        if rep in zero:
            continue
        r_, _ = builder.r_of(rep, ds.name)
        r_ = r_.astype(np.float64)
        c_ = float(r_ @ rS)
        x = lay.setdefault(rep, [0.0, 0.0])
        x[0] += float((upd["attn_diff"][L] + upd["linear_diff"][L]) @ r_) * c_
        x[1] += float(upd["mlp_diff"][L] @ r_) * c_
    layer_scores = sorted(({"layer": L, "attn": v[0], "mlp": v[1], "attn_share": v[0] / gap if abs(gap) > 1e-12 else None,
                            "mlp_share": v[1] / gap if abs(gap) > 1e-12 else None} for L, v in lay.items()),
                          key=lambda d_: -max(d_["attn"], d_["mlp"]))
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
        if k in kinds:
            units.append((as_ - bs, k, L, h, ao, bo, as_, bs))
    big = []
    if "neuron" in kinds:
        for L, (ao, bo, as_, bs, _, _) in neu.items():
            sc_ = as_ - bs
            top = np.argsort(-sc_)[:max(top_q, 64)]   # 全部を行にすると重いので、層ごとに上位だけ並べる(合計は配列で)
            big += [(float(sc_[n]), "neuron", L, int(n), float(ao[n]), float(bo[n]), float(as_[n]), float(bs[n])) for n in top]
    units += big
    if "layer_attn" in kinds:                          # 層の注意の出力全体 = その層の全部のヘッドの和
        for L in hl:
            xs = [x for x in rows_h if x[0] == L]
            units.append((sum(x[5] - x[6] for x in xs), "layer_attn", L, -1, sum(x[3] for x in xs), sum(x[4] for x in xs),
                          sum(x[5] for x in xs), sum(x[6] for x in xs)))
    if "layer_mlp" in kinds:                           # 層の MLP の出力全体 = その層の全部のニューロンの和
        for L, (ao, bo, as_, bs, _, _) in neu.items():
            units.append((float((as_ - bs).sum()), "layer_mlp", L, -1, float(ao.sum()), float(bo.sum()), float(as_.sum()), float(bs.sum())))
    units.sort(key=lambda x: -x[0])
    totals = {k: 0.0 for k in ALL_KINDS}
    for x in rows_h:
        totals[x[2]] += x[5] - x[6]
        totals["layer_attn"] += x[5] - x[6]
    for L, (_, _, as_, bs, _, _) in neu.items():
        totals["neuron"] += float((as_ - bs).sum())
        totals["layer_mlp"] += float((as_ - bs).sum())

    def label(k, L, i):
        if k in LAYER_KINDS:
            return f"{L}:{UNIT_KIND[k]}"
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
    hcols = {}                                         # 層 → その層のヘッドの列(層の注意の出力全体の質問ごとの値)
    for j, (L, *_x) in enumerate(rows_h):
        hcols.setdefault(L, []).append(j)
    for j, r_ in enumerate(topu):
        if r_["kind"] == "layer_attn":
            for s in "AB":
                tq[s][:, j] = HQ[s][:, hcols[r_["layer"]]].sum(axis=1)
        elif r_["kind"] == "layer_mlp":
            L = r_["layer"]
            g = neu[L][4]
            for s in "AB":
                v = np.zeros(nq[s])
                for V in visits_of(session, L):
                    if V in per[s]["ma"]:
                        v += stack(s, "ma", V) @ g
                tq[s][:, j] = v
        elif r_["kind"] != "neuron":
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
    if qup is not None:                                # 質問ごとの更新(keep_q)。対になる = 両側で同じ質問の同じ番号
        arrays.update({f"q_upd_{k}": v for k, v in qup.items()}, **{f"q_scale_{k}": v for k, v in qsc.items()},
                      q_index_a=np.array(qidx["A"], np.int32), q_index_b=np.array(qidx["B"], np.int32),
                      q_paired=np.bool_(not cs.separate_prompts and qidx["A"] == qidx["B"]))
    for L, (ao, bo, as_, bs, _, _) in neu.items():
        arrays[f"neuron_a_own_{L}"] = ao.astype(np.float32)
        arrays[f"neuron_b_own_{L}"] = bo.astype(np.float32)
        arrays[f"neuron_a_sel_{L}"] = as_.astype(np.float32)
        arrays[f"neuron_b_sel_{L}"] = bs.astype(np.float32)
    by_kind_n = {k: sum(1 for x in rows_h if x[2] == k) if k in kinds else 0 for k in ("attn", "linear")}
    by_kind_n["neuron"] = int(sum(v[0].size for v in neu.values())) if "neuron" in kinds else 0
    by_kind_n["layer_attn"] = len(hl) if "layer_attn" in kinds else 0
    by_kind_n["layer_mlp"] = len(neu) if "layer_mlp" in kinds else 0
    summary = {"direction": ds.name, "select_layer": S, "positions": positions, "subset": subset,
               "max_examples": max_examples, "kinds": list(kinds), "direction_layer": direction_layer,
               "n": nq, "skipped_long": skipped, "gap": gap,
               "base": {"A": float(np.mean(per["A"]["pS"])), "B": float(np.mean(per["B"]["pS"]))},
               "n_units": by_kind_n, "head_layers": hl, "neuron_layers": nl, "layer_scores": layer_scores,
               "loop_edits": ("shared" if shared else "per_visit") if plan is not None else None,
               **({"loop_direction": loop_direction} if shared else {}),
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
    return ProjHead(int(row["layer"]), int(row["index"]), float(s), direction, UNIT_KIND.get(row["kind"], "head"))


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
    if mode == "ablate" and any(r["kind"] in LAYER_KINDS for r in rows):
        raise GneError("BAD_ARGS", "層全体の部品(layer_attn・layer_mlp)は ablate では確かめられません(proj で確かめる)")
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
    for k in ("all", *ALL_KINDS):
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
                   remain_threshold: float | None = None, max_examples: int | None = None, strength: str = "per_unit",
                   hard: int = 16, progress=None) -> dict:
    """確かめた部品で組を作る(モジュールの説明の 3)。表の形は heads allocate と同じ(n_heads = 組の部品の数の上限)。
    表とは別に、倍率を上げる前の組(採った部品を全部 s=1)を s1 に入れる。strength は s=1 の組が目標に届かないときの強さの
    決め方: per_unit(既定。部品ごとに、残りやすい質問 hard 個での効率に比例して上げる、_tune)/ uniform(全部を同じ倍率、_raise。
    v0.8.0〜0.8.2)。"""
    from .direction import drift
    if strength not in STRENGTHS:
        raise GneError("BAD_ARGS", f"strength は {' / '.join(STRENGTHS)} です({strength!r})")
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

    def others(got, gb) -> dict:
        """選ぶ層のほかの測る層での、A 側の平均・B 側の変化・直交成分のずれ(ループのモデルでは 1 回目の出口も見る)。"""
        return {str(L): {"a_mean": float(got["A"]["values"][L].mean()),
                         "b_delta": float(gb["B"]["values"][L].mean()) - float(st.base["B"][L].mean()),
                         "b_orth": drift(st.base["B_resid"][L], gb["B"]["resid"][L], st.scorer.r[L])[1]}
                for L in st.score_layers if L != S}

    def run(units: list[ProjHead], alpha: float | None = 1.0, a_idx: list[int] | None = None) -> dict:
        """組を測る。a_idx で A 側の質問を絞れる(部品ごとの強さの効率は、残りやすい質問だけで測る)。a_vals に質問ごとの A の値。"""
        if alpha is None:                            # 部品ごとの強さ: 記録には最大の s を入れる
            alpha = max((u.s for u in units), default=1.0)
        with st.tmp.use(st.builder.tensors([u for u in units if u.s])) as lora:
            got = st.scorer.run(lora, sides="A", which={"A": a_all if a_idx is None else list(a_idx)})
            gb = st.scorer.run(lora, sides="B", which={"B": st.b_idx}, keep=True)
        tick("allocate")
        orth = drift(st.base["B_resid"][S], gb["B"]["resid"][S], st.scorer.r[S])[1]
        out = {**sc.point(alpha, float(got["A"]["values"][S].mean()), float(gb["B"]["values"][S].mean()), orth)}
        if a_idx is None:
            out["layers"] = others(got, gb)
            out["a_vals"] = [float(x) for x in got["A"]["values"][S]]
        return out
    pool = sorted([r for r in verified if not r.get("excluded") and (r.get("effect") or 0) > 0],
                  key=lambda r: -r["effect"])
    nmax = max(int(x) for x in n_units)
    est = len(pool) + len(n_units) * 16 + 1

    # s=1 で 1 個ずつ足す(A 側の平均が下がり、組の効きが増え、上限の中のときだけ)
    chosen: list[ProjHead] = []
    trace = []
    at1: dict[tuple, dict] = {}                  # 組(部品の並び)→ s=1 での点
    cur = sc.point(1.0, base_a, sc.base_b, 0.0)
    w = session.weights

    def tkey(u: ProjHead) -> tuple[int, str]:  # 書き込む行列(重みを持つ層と、ヘッドの行列か MLP の行列か)
        return (w.stored(u.layer), TENSOR_KIND[u.kind])
    for r in pool:
        if len(chosen) >= nmax or cur["a_mean"] <= target:
            break
        u = unit_of(r, 1.0, ds.name)
        # 層全体の部品と、その層の同じ行列の個別の部品は二重に引くことになるので、両方は入れない: 層全体を試すときは、
        # 組の中の同じ行列の個別の部品と入れ替えて比べる。層全体が組にあれば、その行列の個別の部品は試さない
        if u.kind in WHOLE:
            out_ = [c for c in chosen if c.kind not in WHOLE and tkey(c) == tkey(u)]
        else:
            if any(c.kind in WHOLE and tkey(c) == tkey(u) for c in chosen):
                trace.append({"label": r["label"], "kind": r["kind"], "accepted": False, "covered": True})
                continue
            out_ = []
        trial = [c for c in chosen if c not in out_] + [u]
        g = run(trial)
        ok = g["a_mean"] < cur["a_mean"] - 1e-6 and g["effect"] > cur["effect"] and g["feasible"]
        trace.append({"label": r["label"], "kind": r["kind"], "accepted": bool(ok), "a_mean": g["a_mean"],
                      "b_delta": g["b_delta"], "b_drift_orth": g["b_orth"], "effect": g["effect"],
                      **({"over": g["over"]} if g["over"] else {}),
                      **({"replaces": [c.spec().split(":s")[0] for c in out_]} if out_ else {})})
        if ok:
            chosen = trial
            cur = g
            at1[tuple(c.spec() for c in chosen)] = g

    def finish(units: list[ProjHead], alpha: float | None, label: str, strengths: list[float] | None = None) -> dict:
        sv = strengths if strengths is not None else [alpha] * len(units)
        heads = [ProjHead(u.layer, u.head, float(x), u.direction, u.kind) for u, x in zip(units, sv, strict=True)]
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
        pt = SetScore(base_a, float(b_base[S].mean()), drift_max, b_move_max).point(max(ss), float(va.mean()), bm, orth)
        def kind_of(p):
            return {"neuron": "neuron", "attn": "layer_attn", "ffn": "layer_mlp"}.get(p.kind) or kind_of_layer(session, p.layer)
        return {"n_used": len(heads),
                "heads": [{"layer": p.layer, "head": p.head, "kind": kind_of(p), "s": p.s} for p in heads],
                "kinds": {k: sum(1 for p in heads if kind_of(p) == k) for k in ALL_KINDS},
                "units": [p.spec() for p in heads], "s_max": max(ss), "s_median": float(np.median(ss)), "s_min": min(ss),
                "alpha": alpha,
                "hard_mean": float(va.mean()), "remain": int((va > thr).sum()), "a_max": float(va.max()),
                "a_mean": float(va.mean()), "b_mean": bm, "b_side_delta": pt["b_delta"], "b_toward": pt["b_toward"],
                "effect": pt["effect"], "feasible": pt["feasible"],
                "closed": (base_a - float(va.mean())) / (base_a - target) if abs(base_a - target) > 1e-9 else None,
                "b_drift_par": par, "b_drift_orth": orth, "recipe": str(rpath),
                "by_layer": {str(L): {"A": float(got["A"]["values"][L].mean()), "B": float(got["B"]["values"][L].mean()),
                                      "b_orth": drift(st.base["B_resid"][L], gb["B"]["resid"][L], st.scorer.r[L])[1]}
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
        p1 = at1.get(tuple(c.spec() for c in units)) or run(units)     # 入れ替えがあると、上限で切った組は測っていない
        alpha, stop, n_iter, sv = 1.0, "target", 0, None
        rtrace = [{k: v for k, v in p1.items() if k != "a_vals"} | {"phase": "s1"}]
        if p1["a_mean"] > target:
            if strength == "per_unit":
                if "a_vals" not in p1:
                    p1 = run(units)
                probes = []
                sv, stop, rtrace = _tune(lambda x, a_idx, units=units: run(
                    [ProjHead(u.layer, u.head, xi, u.direction, u.kind) for u, xi in zip(units, x, strict=True)], None, a_idx),
                    [1.0] * len(units), p1, target, s_max, sc, hard=int(hard), probes=probes)
                for pb in probes:                    # 選んだ質問の文(画面でたためる一覧)
                    pb["questions"] = [_q_text(st.scorer.prompts_side["A"][i]) for i in pb["hard"]]
                alpha = None
            else:
                alpha, stop, rtrace = _raise(lambda a, units=units: run([ProjHead(u.layer, u.head, a, u.direction, u.kind)
                                                                         for u in units], a),
                                             p1, target, s_max, sc)
                rtrace = [{k: v for k, v in p.items() if k != "a_vals"} for p in rtrace]
            n_iter = len(rtrace) - 1
        row = finish(units, alpha, f"n{n}", sv)
        if strength == "per_unit" and p1["a_mean"] > target:
            row["probes"] = probes
        table.append({"n_heads": n, "n_units": n, **row,
                      "reached": bool(row["a_mean"] <= target + 1e-6 or stop in ("target", "raised")),
                      "iterations": n_iter, "stop": stop, "capped": None, "raise": rtrace, "strength": strength,
                      "seconds": time.perf_counter() - t0})
    s1 = None
    if chosen:                                      # 倍率を上げる前の組(取り除くだけでどこまで届くか)
        t0 = time.perf_counter()
        s1 = {"n_heads": len(chosen), **finish(chosen, 1.0, "s1"), "seconds": time.perf_counter() - t0}
        s1["reached"] = bool(s1["a_mean"] <= target + 1e-6)
    notes = []                                      # 知らせ(code と、CLI・MCP 向けの文 text)
    if s1 is not None and not s1["reached"] and s1.get("closed") is not None and s1["closed"] < 0.5:
        notes.append({"code": "s1_low", "closed": float(s1["closed"]),
                      "text": f"倍率を上げる前の組(取り除くだけ)は、目標までの差の {100 * s1['closed']:.0f}% しか縮めません。直接書き込む"
                              "部品の成分を取り除くだけでは届かない題材で、強さを上げて(成分を反転させて)届かせています"})
    plan = getattr(session.weights, "loop", None)
    if plan is not None and _shared(session):
        inner = sorted({plan.stored(u.layer) for u in chosen if plan.visit(u.layer) != 0})
        if inner:
            notes.append({"code": "loop_inner", "lo": inner[0], "hi": inner[-1], "first": [plan.lo1, plan.hi1],
                          "text": f"組にループの中の部品(段 {inner[0]}〜{inner[-1]} の重み)が入っています。重みは 2 回とも使われるので、"
                                  f"1 回目(段 {plan.lo1}〜{plan.hi1})にも効きます"
                                  + (f"(表の段 {plan.hi1} の値)" if loop_exit(session, st.score_layers) else "")})
    rec_n, why, no_rec = _recommend(table, len(a_all), sc)
    for r in table:
        r["recommended"] = r.get("n_heads") == rec_n and rec_n is not None
    out = {"name": name, "direction": ds.name, "contrast_sha": st.cs.sha, "score_layer": S, "score_layers": st.score_layers,
           "max_examples": max_examples, "drift_max": drift_max, "b_move_max": b_move_max, "target": float(target),
           "positions": st.scorer.positions, "subset": st.scorer.subset, "direction_layer": direction_layer,
           "method": "unit1", "strength": strength, "hard": int(hard), "s_max": s_max, "remain_threshold": thr,
           "n_a": len(a_all), "n_b": len(st.b_idx),
           "base_a_mean": base_a, "base_b_mean": float(st.base["B"][S].mean()), "gap": sc.gap,
           "base_by_layer": {str(L): {"A": float(st.base["A"][L].mean()), "B": float(b_base[L].mean())} for L in st.score_layers},
           "loop_exit": loop_exit(session, st.score_layers),
           "initial": None, "pool": len(pool), "trace": trace, "table": table, "s1": s1, "recommended": rec_n,
           "recommended_by": why, "no_recommend": no_rec, "notes": notes, "seconds": time.perf_counter() - t_all}
    if progress:
        progress(steps[0], steps[0], "allocate")
    p = session.store.heads_dir(session.meta.model_id) / f"{name}.alloc.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    out["path"] = str(p)
    return out


def _q_text(q, n: int = 160) -> str:
    """質問の短い文(会話の履歴なら最後の user の発話)。"""
    x = q if isinstance(q, str) else next((m.get("content") or "" for m in reversed(q) if m.get("role") == "user"), "")
    return x if len(x) <= n else x[:n - 1] + "…"


def loop_exit(session, score_layers) -> int | None:
    """同じ層を 2 回実行するモデルで、1 回目の出口の段(測る層に入っているとき)。表に、そこでの値を並べる。"""
    plan = getattr(session.weights, "loop", None)
    return plan.hi1 if plan is not None and plan.hi1 in score_layers else None


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
                if hi - lo < 0.02 * hi:              # 2% まで詰めれば十分(目標に届く最小の α を詰めるのと同じ)
                    break
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


STRENGTHS = ("per_unit", "uniform")   # 組の強さの決め方(v0.8.3 から per_unit が既定。v0.8.0〜0.8.2 は uniform だけ)
TUNE_ITERS = 6           # per_unit: 効率を測り直して上げる回数の上限
PROBE_UP = 1.5           # 効率を測るとき、その部品の s を何倍にしてみるか
LINE_T = (0.5, 1.0, 2.0, 4.0, 8.0)    # 上げ幅の線探索(いちばん効く部品の s が 1 + t 倍。_raise の格子と同じくらいの刻みから)


def s_summary(sv) -> dict:
    return {"s_max": float(max(sv)), "s_median": float(np.median(sv)), "s_min": float(min(sv))}


def hard_of(a_vals, target: float, hard: int) -> list[int]:
    """射影の値が残りやすい A 側の質問(目標を超えたまま残っている質問のうち、値の大きい順に hard 個まで)。無ければ値の大きい順。"""
    v = np.asarray(a_vals, dtype=np.float64)
    order = [int(i) for i in np.argsort(-v)]
    over = [i for i in order if v[i] > target]
    return (over or order)[:max(1, int(hard))]


def _tune(run_s, s0: list[float], p1: dict, target: float, s_max: float, sc: SetScore, *, hard: int = 16,
          iters: int = TUNE_ITERS, probes: list | None = None) -> tuple[list[float], str, list[dict]]:
    """部品ごとの強さ(v0.8.3 からの既定)。s0(全部 1)から、毎回
    1. 射影の値が残りやすい A 側の質問(目標を超えたまま残っている質問、hard 個まで。回ごとに選び直す)で、部品ごとに s を PROBE_UP
       倍にしてみて(ほかはそのまま)、A 側の下がり − B 側の動きの大きさの増え方 g_i と、強さ 1 あたりの効率 e_i = g_i / Δs_i を出す。
       上限(B 側が A 側へ動く・直交成分のずれ)を超える試しは e_i = −∞(その部品は上げない)。
    2. 上げ幅を効率に比例して配る: d_i = s_i · max(e_i, 0) / max_j e_j(いちばん効く部品の s が 1 + t 倍。効かない部品はそのまま)。
    3. s + t·d を t = LINE_T(0.5〜8)で、全部の A 側の質問で試し、_raise と同じ規則(目標に届けば届く最小の t、上限を超えたら手前、
       組の効きが差の GAIN_TOL 倍より増えない点が GAIN_PATIENCE 個続いたら止める)で、組の効きがいちばん良い点を採る。
       1 部品ずつの試しは小さく(部品が多いと 1 個を 1.5 倍にしても組はほとんど変わらない)、s が小さいうちは効かない部品もあるので、
       効率の向きで組の効きが増えないときだけ、全部を同じ割合で上げる向き(d_i = s_i)で線探索する。
    どちらの向きでも組の効きが増えなければ止める(no_gain。上限を超えたためなら b_move / drift)。部品は外さず、s は 1 以上・s_max 以下。
    run_s(sv, a_idx) は強さ sv の組を、A 側は a_idx の質問(None で全部)で測る(a_vals に質問ごとの値)。p1 は s0 の点(a_vals つき)。
    probes にリストを渡すと、回ごとに選んだ質問(hard: 添字、a_vals: その値)・部品ごとの s と効率(eff。上げられない部品は None と
    理由 blocked: s_max / b_move / drift)を足す(画面の「強さの決め方」の表)。
    戻り値 (部品ごとの s, 止まった理由 raised / no_gain / iters / b_move / drift, 測った点の並び)。"""
    tol = GAIN_TOL * sc.gap
    n = len(s0)
    s = [float(x) for x in s0]
    cur = dict(p1, phase="s1", **s_summary(s))
    trace = [{k: v for k, v in cur.items() if k != "a_vals"}]
    stop = "iters"

    def clip(v):
        return min(s_max, max(1.0, v))

    def keep(p):
        trace.append({k: v for k, v in p.items() if k != "a_vals"})

    def line(d, step, how):
        """s + t·d の線探索。戻り値 ("raised", 強さ) / ("best", (強さ, 点)) / ("none", 上限を超えた理由か None)。"""
        def at(t):
            sv = [clip(s[i] + t * d[i]) for i in range(n)]
            return sv, dict(run_s(sv, None), phase="line", step=step, t=t, dir=how, **s_summary(sv))
        best, bad, prev_t = None, 0, 0.0
        for t in LINE_T:
            sv, pt = at(t)
            keep(pt)
            if not pt["feasible"]:
                return ("best", best) if best else ("none", pt["over"])
            if pt["a_mean"] <= target:               # 目標に届いた: 手前とのあいだで、届く最小の t
                lo, hi, hs = prev_t, t, sv
                for _ in range(6):
                    if hi - lo < 0.02 * hi:
                        break
                    mid = 0.5 * (lo + hi)
                    sm, pm = at(mid)
                    pm["phase"] = "shrink"
                    keep(pm)
                    if pm["a_mean"] <= target and pm["feasible"]:
                        hi, hs = mid, sm
                    else:
                        lo = mid
                return "raised", hs
            if pt["effect"] > (best[1]["effect"] if best else cur["effect"]) + tol:
                best, bad = (sv, pt), 0
            else:
                bad += 1
                if bad >= GAIN_PATIENCE:
                    break
            prev_t = t
        return ("best", best) if best else ("none", None)
    for it in range(iters):
        hq = hard_of(cur["a_vals"], target, hard)   # 1. 残りやすい質問で、部品ごとの効率
        a_h = float(np.mean([cur["a_vals"][i] for i in hq]))
        e = [float("-inf")] * n
        blocked: list[str | None] = [None] * n
        for i in range(n):
            up = clip(s[i] * PROBE_UP)
            if up <= s[i] + 1e-9:
                blocked[i] = "s_max"
                continue
            sv = list(s)
            sv[i] = up
            pr = run_s(sv, hq)
            if pr["feasible"]:
                e[i] = ((a_h - pr["a_mean"]) * sc.sign - (abs(pr["b_delta"]) - abs(cur["b_delta"]))) / (up - s[i])
            else:
                blocked[i] = pr.get("over") or "limit"
        if probes is not None:
            probes.append({"step": it + 1, "hard": [int(i) for i in hq], "a_vals": [float(cur["a_vals"][i]) for i in hq],
                           "a_hard": a_h, "s": [float(x) for x in s],
                           "eff": [None if x == float("-inf") else float(x) for x in e], "blocked": blocked})
        top = max(e)
        res = ("none", None)
        if top > 0:                                  # 2〜3. 効率に比例した向き
            res = line([s[i] * e[i] / top if e[i] > 0 else 0.0 for i in range(n)], it + 1, "efficiency")
        if res[0] == "none" and any(x < s_max - 1e-9 for x in s):
            # 効率の向きで良くならないとき(1 個ずつの試しが小さすぎる組など)だけ、全部を同じ割合で上げる向き
            # (上限を超える試しになった部品と s_max の部品は動かない)
            over = res[1]
            res = line([s[i] if e[i] > float("-inf") else 0.0 for i in range(n)], it + 1, "uniform")
            if res[0] == "none" and res[1] is None:
                res = ("none", over)
        if res[0] == "raised":
            return res[1], "raised", trace
        if res[0] == "none":
            stop = res[1] or "no_gain"
            break
        s, cur = res[1][0], dict(res[1][1], phase="step")
        keep(cur)
    return s, stop, trace


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
    for k in ALL_KINDS:
        for r in [x for x in rows if x["kind"] == k][:50]:
            if r["label"] not in seen:
                keep.append(r)
                seen.add(r["label"])
    js = {**{k: v for k, v in summary.items() if k != "rows"}, "rows": keep, "n_rows_total": len(rows),
          **({"verify": verify} if verify is not None else {}), "npz": pz.name, "name": name,
          "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    pj.write_text(json.dumps(js, ensure_ascii=False, indent=1), encoding="utf-8")
    return pz, pj
