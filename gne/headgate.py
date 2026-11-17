"""CHiPS のヘッドの分を、場面を選んで取り除く(gne nw heads、docs/fish.md「ヘッドの分を場面で取り除く」)。

CHiPS(gne headproj、mode proj)はヘッド h が残差に書くベクトルの方向 r̂ の成分を (1 − s_h) 倍にする。層 L では残差が
−T_L·r̂_L だけ変わる(T_L = Σ_h s_h · r̂ᵀ W[:, blk_h] x_h。x_h はヘッドの出力の手前のテンソル)。CHiPS はどの場面でも働く。

ここでは、ヘッドの重みは変えず、同じ層の MLP の 2 個 1 組(上限つきの形)が、鍵が開いた場面でだけ、T_L を MLP の入力 h から
当てて取り除く:
- 取り除く量の鍵(take): h → T_L の最小二乗(リッジ、切片あり。切片はほぼ 1 を返す鍵 κ1 で書く)。h は「その層のヘッドの分を
  まだ取り除いていない状態」= CHiPS を当てた実行の、MLP の入力の正規化の前の残差 x に T_L·r̂_L を足し戻して正規化したもの
  (MLP は、層 L のヘッドの分を後から取り除くので)。行は、対照集合の狙う場面と dont_touch の質問の、無介入の返事の位置と
  プロンプトの最後の数位置。正則化の強さは質問単位の 5 分割の R² で決める。
- 方向: CHiPS が層 L で使う r̂_L そのもの(direction_layer own・base のどちらでも)を、方向のファイル <名前>_dir に書く。
- ユニット: L:Na+Nb:nw:a1@<名前>_dir#<場面の鍵>&<ほぼ 1 を返す鍵>!kauto!take<名前>_amount。
確かめ: 当たり方(R²)と、CHiPS との一致(いちばん後ろのヘッドの層の出力の変化の cos と大きさの比。どこでも開く鍵で当てた
ときと、本物の鍵で当てたとき。返事の位置で)。CHiPS を常に当てたときとの比べは、レシピを並べて gne eval / behave で行う。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import keyed as K
from .steer import GneError, Recipe

CS = (0.01, 0.1, 1.0, 10.0, 100.0)


def _ridge_cv(X, y, folds, cs=CS):
    """y ≈ X u + c0(切片は正則化しない)。5 分割の R² で正則化の強さを選ぶ。"""
    Xa = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    D = Xa.shape[1]
    G = Xa.T @ Xa
    tau = float(np.trace(G)) / D
    r2 = []
    for c in cs:
        pred = np.zeros(len(y))
        for f in sorted(set(folds.tolist())):
            hm = folds == f
            Gt = G - Xa[hm].T @ Xa[hm]
            A1 = Gt.copy()
            A1[np.diag_indices(D)] += c * tau
            A1[-1, -1] -= c * tau
            pred[hm] = Xa[hm] @ np.linalg.solve(A1, Xa[~hm].T @ y[~hm])
        r2.append(float(1 - np.mean((y - pred) ** 2) / max(float(np.var(y)), 1e-30)))
    j = int(np.argmax(r2))
    A1 = G.copy()
    A1[np.diag_indices(D)] += cs[j] * tau
    A1[-1, -1] -= cs[j] * tau
    u = np.linalg.solve(A1, Xa.T @ y)
    return u[:-1], float(u[-1]), cs[j], r2


def chips_layers(session, chips: Recipe) -> dict[int, dict]:
    """CHiPS のレシピ → 層ごとの {A: r̂ᵀW をヘッドの列ブロックだけ残して s を掛けたもの, rhat}(T_L = A·x_in)。"""
    from .headproj import ProjBuilder, proj_heads
    heads = [p for p in proj_heads(chips) if p.s]
    if not heads:
        raise GneError("BAD_RECIPE", "CHiPS のレシピに proj のユニット(L:Hh:sS@方向)がありません")
    b = ProjBuilder(session.weights, session.direction_set, direction_layer=chips.direction_layer)
    out = {}
    for L, (A, B) in b.matrices(heads).items():
        if A.shape[0] != 1:
            raise GneError("BAD_RECIPE", f"層 {L} のヘッドが別々の方向を使っています。1 つの層では 1 つの方向にしてください")
        out[int(L)] = {"A": A[0].astype(np.float64), "rhat": (-B[:, 0]).astype(np.float64),
                       "heads": [p.spec() for p in heads if p.layer == L]}
    return out


def _save_direction(session, name: str, lay: dict[int, dict]) -> Path:
    p = session.store.direction_dir(session.meta.model_id) / f"{name}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    Ls = sorted(lay)
    np.savez(p, layers=np.array(Ls, dtype=np.int32), **{f"layer_{L}": lay[L]["rhat"].astype(np.float32) for L in Ls},
             method="chips_heads", positions="none", source="gne nw heads")
    session.__dict__.get("_dir_cache", {}).pop(name, None)
    return p


def _save_amount_key(session, name: str, kappa: dict[int, np.ndarray], info: dict) -> Path:
    n = session.meta.n_embd
    ks = K.KeySet(target=K.MLPIN, kappa={L: v.astype(np.float32) for L, v in kappa.items()},
                  mu={L: np.zeros(n, np.float32) for L in kappa}, kn={L: 1.0 for L in kappa}, gain={L: 1.0 for L in kappa},
                  info=info, model_id=session.meta.model_id, name=name)
    return ks.save(K.key_dir(session) / f"{name}.npz")


def build(session, chips, contrast, *, name: str, scene_key: str, one_key: str, neurons: dict | None = None,
          subset: str = "even", max_tokens: int = 96, n_prompt_pos: int = 8, n_match: int = 6, save: str | None = None,
          progress=None) -> dict:
    """CHiPS のレシピ(名前・パスか Recipe)と対照集合から、取り除く量の鍵・方向・ユニット(とレシピ)を作り、当たり方と CHiPS との
    一致を返す。scene_key は場面の鍵(gne nw key --role scene)、one_key はほぼ 1 を返す鍵(--role one)で、どちらも CHiPS の
    ヘッドの層を含むこと。neurons は {層: [a, b]}(省くと gne nw neurons の働きの小さい順の上 2 個)。"""
    import re
    from .direction import _subset
    from .evalx import _generate
    from .keyed_build import _as_spec
    from .nwrite import _PRE_NORM, SWIGLU_ARCHS, _run_capture, check_mlp, mlp_in_name, norm_weight
    from .reverse import _conv
    if not re.fullmatch(r"[\w.][\w.\-]*", name or ""):
        raise GneError("BAD_ARGS", f"名前 {name!r} は英数字・_・.・- だけにしてください")
    t0 = time.perf_counter()
    tick = progress or (lambda *_: None)
    cr = chips if isinstance(chips, Recipe) else session.recipe(path=session.recipe_path(chips))
    lay = chips_layers(session, cr)
    Ls = sorted(lay)
    w, e = session.weights, session.engine
    for L in Ls:
        check_mlp(w, L)
        if L == K.last_main_layer(w):
            raise GneError("BAD_RECIPE", f"層 {L} は最終層です")
    for kn in (scene_key, one_key):
        ks = session._key_of(kn)
        miss = [L for L in Ls if L not in ks.kappa]
        if ks.target != K.MLPIN or miss:
            raise GneError("BAD_KEY", f"鍵 {kn} は MLP の入力の空間(mlpin)で、層 {Ls} を含む必要があります(足りない層 {miss})")
    dir_name, amt_name = f"{name}_dir", f"{name}_amount"
    _save_direction(session, dir_name, lay)
    # 材料: 狙う場面(subset)と dont_touch の質問に、無介入で返事を作る
    spec = _as_spec(contrast)
    cs = spec.contrast
    prompts = cs.prompts("B")
    convs = [_conv(cs.system_b, prompts[i]) for i in _subset(len(prompts), subset)]
    scene_n = len(convs)
    for dt in spec.dont_touch:
        convs += dt.convs(list(range(len(dt.prompts))))
    tick("replies", 0, len(convs))
    outs = _generate(session, [(c, int(max_tokens)) for c in convs], None, None, "all", lambda k: None)
    seqs = [(e.tokenize_prompt(e.render_chat(c)), [int(t) for t in o["token_ids"] if not e.is_eog(t)]) for c, o in zip(convs, outs)]
    chips_lora = session.lora_for(cr)
    seen = e.list_names(seqs[0][0][:16])
    from .attn import head_out_names
    hon = head_out_names(session, set(seen), Ls)
    pre = {L: f"{_PRE_NORM[SWIGLU_ARCHS[w.meta.arch]]}-{L}" for L in Ls}
    miss = [L for L in Ls if L not in hon or pre[L] not in seen]
    if miss:
        raise GneError("ARCH_UNSUPPORTED", f"層 {miss} のヘッドの出力の手前か、MLP の入力の残差が捕捉できません")
    gam = {L: norm_weight(w, L) for L in Ls}
    X = {L: [] for L in Ls}
    Y = {L: [] for L in Ls}
    grp = []
    for qi, (p, rid) in enumerate(seqs):
        ids = p + rid
        g = _run_capture(session, ids, {hon[L] for L in Ls} | set(pre.values()), chips_lora)
        at = list(range(max(0, len(p) - n_prompt_pos), len(ids)))
        for L in Ls:
            T = g[hon[L]][at] @ lay[L]["A"]
            x = g[pre[L]][at] + np.outer(T, lay[L]["rhat"])          # 層 L のヘッドの分をまだ取り除いていない状態
            h = gam[L] * x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + 1e-6)
            X[L].append(h)
            Y[L].append(T)
        grp += [qi] * len(at)
        tick("rows", qi + 1, len(seqs))
    grp = np.asarray(grp)
    folds = grp % 5
    kappa, rows = {}, []
    one = session._key_of(one_key)
    for L in Ls:
        Xa, ya = np.concatenate(X[L]), np.concatenate(Y[L])
        u, c0, c, r2 = _ridge_cv(Xa, ya, folds)
        kappa[L] = u + c0 * np.asarray(one.kappa_of(L), dtype=np.float64).reshape(-1)
        sc = grp < scene_n
        rows.append({"layer": L, "heads": lay[L]["heads"], "r2_cv": max(r2), "c": c, "r2_by_c": dict(zip(map(str, CS), r2)),
                     "T_mean_scene": float(np.mean(ya[sc])) if sc.any() else None,
                     "T_mean_other": float(np.mean(ya[~sc])) if (~sc).any() else None, "n_rows": int(len(ya))})
    _save_amount_key(session, amt_name, kappa, {"source": "gne nw heads", "chips": [x for L in Ls for x in lay[L]["heads"]],
                                                "contrast": cs.name, "subset": subset, "max_tokens": int(max_tokens)})
    # ニューロン
    if neurons is None:
        us = session.nw_neurons(Ls)
        neurons = {int(x["layer"]): [r["neuron"] for r in x["rows"][:2]] for x in us["layers"]}
    units = [f"{L}:N{neurons[L][0]}+N{neurons[L][1]}:nw:a1@{dir_name}#{scene_key}&{one_key}!kauto!take{amt_name}" for L in Ls]
    rec = session.recipe(units=units, name=name)
    match = compare(session, cr, rec, seqs[:min(n_match, scene_n)], seqs[scene_n:scene_n + n_match], one_key)
    path = None
    if save:
        from .fish import recipe_file, save_recipe
        path = save_recipe(session, units, recipe_file(session, save))
    return {"name": name, "direction": dir_name, "amount_key": amt_name, "scene_key": scene_key, "one_key": one_key,
            "layers": rows, "units": units, "recipe_path": path, "match": match, "n_scene": scene_n, "n_other": len(seqs) - scene_n,
            "seconds": round(time.perf_counter() - t0, 1)}


def compare(session, chips: Recipe, rec: Recipe, scene_seqs, other_seqs, one_key: str) -> dict:
    """いちばん後ろのヘッドの層の出力(l_out)の、無介入との差を返事の位置で比べる。
    open: ユニットの場面の鍵をほぼ 1 を返す鍵に替えた(どこでも開く)LoRA と CHiPS の、向きの一致(cos)と大きさの比。
    scene / other: 本物の鍵で当てたときの、狙う場面・dont_touch での、CHiPS の変化に対する大きさの比(と scene の cos)。"""
    from .nwrite import _run_capture
    Lmax = max(int(u.layer) for u in rec.nw_units)
    nm = f"l_out-{Lmax}"
    chips_l = session.lora_for(chips)
    real = [(str(session.nw_lora(rec)), 1.0)]
    from dataclasses import replace
    open_rec = rec.like([replace(u, key=one_key) for u in rec.nw_units])
    opened = [(str(session.nw_lora(open_rec)), 1.0)]

    def diffs(seqs, lora):
        cs_, ra = [], []
        for p, rid in seqs:
            ids = p + rid
            b = _run_capture(session, ids, {nm})[nm][len(p):]
            c = _run_capture(session, ids, {nm}, chips_l)[nm][len(p):] - b
            d = _run_capture(session, ids, {nm}, lora)[nm][len(p):] - b
            nc = np.linalg.norm(c, axis=1) + 1e-12
            cs_ += ((d * c).sum(1) / (np.linalg.norm(d, axis=1) * nc + 1e-12)).tolist()
            ra += (np.linalg.norm(d, axis=1) / nc).tolist()
        return {"cos_median": float(np.median(cs_)), "ratio_median": float(np.median(ra)), "n_positions": len(ra)} if ra else None
    return {"layer": Lmax, "open": diffs(scene_seqs, opened), "scene": diffs(scene_seqs, real), "other": diffs(other_seqs, real)}


def summary_lines(d: dict) -> list[str]:
    out = [f"方向 {d['direction']}、取り除く量の鍵 {d['amount_key']}(狙う場面 {d['n_scene']} 問・dont_touch {d['n_other']} 問の返事で作成)"]
    for r in d["layers"]:
        out.append(f"  L{r['layer']} {', '.join(r['heads'])}: R²(5 分割){r['r2_cv']:.3f}(正則化 {r['c']:g})、行 {r['n_rows']}")
    m = d["match"]
    for k, lab in (("open", "どこでも開く鍵"), ("scene", "本物の鍵・狙う場面"), ("other", "本物の鍵・dont_touch")):
        x = m.get(k)
        if x:
            out.append(f"  CHiPS との一致(層 {m['layer']} の出力、返事の位置){lab}: cos {x['cos_median']:.3f}、大きさの比 {x['ratio_median']:.3f}")
    out.append("ユニット: " + "  ".join(d["units"]))
    if d.get("recipe_path"):
        out.append(f"レシピ: {d['recipe_path']}")
    return out


def to_json(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False, default=str)
