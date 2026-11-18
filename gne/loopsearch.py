"""同じ層を 2 回実行するモデル(sensenova_loop、docs/sensenova_loop.md)で、1 回目と 2 回目を別々に扱う CHiPS の道具。

- 回ごとに測る: 射影の値を、1 回目の出口・2 回目の出口・最後の層の 1 つ前で並べる(heads pipeline --score-layers loop)。
- 回ごとに探す(2 回目の層に別の名前がある llama.cpp が要る):
  visits_screen: ヘッドを 1 回目だけ・2 回目だけ・両方で止めて、効き目を並べ、回の間の順位相関を出す。
  loop_pipeline: 1 回目で効くヘッドを選ぶ → それを当てた状態で 2 回目で効くヘッドを選ぶ、を 1 回で流す。
- 仕上げ: loop_merge で、回ごとに選んだヘッドを「両方の回に当てる」レシピ(修正なしの llama.cpp で動く形)に直し、
  両方に当てた状態で測り直す。同じヘッドを両方の回で選んだときは、2 つの射影を続けて掛ける(compose)か、強いほうだけを
  使う(max)。
"""

from __future__ import annotations

import dataclasses
import json
import re
import time
from pathlib import Path

import numpy as np

from .steer import GneError, Recipe


def _idx(u) -> int:
    """proj のユニットの番号(ヘッドはヘッドの番号、ニューロンはニューロンの番号)。"""
    return int(u.neuron) if u.kind == "neuron" else int(u.head)


def _vkey(lp, u) -> str:
    """loop_visits の鍵: ヘッドは "21:H20"(今までと同じ)、ニューロンは "21:N4096"。"""
    return f"{lp.stored(int(u.layer))}:{'N' if u.kind == 'neuron' else 'H'}{_idx(u)}"


def _plan(session):
    lp = session.weights.loop
    if lp is None:
        raise GneError("ARCH_UNSUPPORTED", "これは同じ層を 2 回実行するモデル(sensenova_loop)のための道具です")
    return lp


def require_per_visit(session, what: str) -> None:
    """回ごとに当てる道具を、2 回目の層に別の名前が無い llama.cpp で断る(理由と代わりの方法つき)。"""
    from .loop import PER_VISIT_FIX, per_visit_refusal
    _plan(session)
    if not session.engine.loop_runtime_names():
        raise GneError("ARCH_UNSUPPORTED", per_visit_refusal(what), PER_VISIT_FIX)


def loop_score_layers(session) -> list[int]:
    lp = _plan(session)
    return [lp.lo2 - 1, lp.hi2 - 1, lp.n_layer - 2]


def visit_ranges(session) -> dict[str, list[int]]:
    """候補の層: first = 1 回目より前と 1 回目(段 0〜27)、second = 2 回目と後ろ(段 28〜53。最後の層は測れないので除く)。"""
    lp = _plan(session)
    return {"first": list(range(0, lp.lo2)), "second": list(range(lp.lo2, lp.n_layer - 1))}


# --- 回ごとに止めて測る ------------------------------------------------------------------------------

def visits_screen(session, direction: str, *, contrast=None, layers: list[int] | None = None, top_layers: int = 4,
                  score_layers: list[int] | None = None, select_layer: int | None = None, positions: str = "last",
                  max_examples: int | None = 64, direction_layer: str = "own", name: str | None = None) -> dict:
    """ヘッドを 1 回目だけ(段 L)・2 回目だけ(段 L + 24)・両方(重みを持つ層 L の重みを 2 回とも)止めて、射影の値の
    下がり方を並べる。layers は重みを持つ層の番号(1 回目の段の番号、4〜27)。省くと、1 回目だけ・2 回目だけで層ごとに
    全部のヘッドを止め、どちらかで下がり方の大きい top_layers 層を選ぶ。結果は <root>/heads/<model_id>/<name>.visits.json。"""
    from .causal import prepare, screen_heads, screen_layers, spearman
    lp = _plan(session)
    require_per_visit(session, "1 回目だけ・2 回目だけヘッドを止めて測る因果スクリーニング")
    t0 = time.perf_counter()
    ds, cs, _ = session._direction_contrast(direction, contrast)
    name = name or ds.name
    if not re.fullmatch(r"[\w.\-]+", name):
        raise GneError("BAD_ARGS", f"名前 {name!r} は英数字・_・-・. だけにしてください")
    session.engine.free_steer()
    Ls = list(score_layers) if score_layers else loop_score_layers(session)
    S = int(select_layer) if select_layer is not None else max(Ls)
    st = prepare(session, ds, cs, score_layers=Ls, primary=S, positions=positions, max_examples=max_examples,
                 direction_layer=direction_layer)
    loop_layers = [L for L in range(lp.lo1, lp.hi1 + 1)
                   if session.weights.arch.layers[L].head_dim and session.weights.arch.layers[L].head_writer]
    layer_rows = None
    with session.loop_mode("per_visit"):
        if layers is None:
            first = screen_layers(session, ds, cs, setup=st, layers=loop_layers, progress=session.progress)
            second = screen_layers(session, ds, cs, setup=st, layers=[L + lp.shift for L in loop_layers],
                                   progress=session.progress)
            layer_rows = []
            for a, b in zip(first, second, strict=True):
                layer_rows.append({"layer": a["layer"], "first": a["score_drop"], "second": b["score_drop"],
                                   "drift_orth_b": max(a.get("drift_orth_b") or 0, b.get("drift_orth_b") or 0)})
            ok = [r for r in layer_rows if r["drift_orth_b"] <= 0.5]
            key = lambda r: -max(r["first"][str(S)], r["second"][str(S)], max(r["first"].values()), max(r["second"].values()))
            layers = sorted(r["layer"] for r in sorted(ok, key=key)[:max(1, top_layers)])
        layers = [int(L) for L in layers]
        bad = [L for L in layers if L not in loop_layers]
        if bad:
            raise GneError("BAD_ARGS", f"layers は 2 回実行する層のうちヘッドのある層です(段 {lp.lo1}〜{lp.hi1}、{bad})")
        heads = [(L, h) for L in layers for h in range(session.weights.arch.layers[L].n_head)]
        rows_first = screen_heads(session, ds, cs, candidates=heads, mode="ablate", setup=st, progress=session.progress)
        rows_second = screen_heads(session, ds, cs, candidates=[(L + lp.shift, h) for L, h in heads], mode="ablate",
                                   setup=st, progress=session.progress)
    with session.loop_mode("shared"):
        rows_both = screen_heads(session, ds, cs, candidates=heads, mode="ablate", setup=st, progress=session.progress)
    by = lambda rows, off: {(r["layer"] - off, r["head"]): r for r in rows}
    f, s2, b = by(rows_first, 0), by(rows_second, lp.shift), by(rows_both, 0)
    out_rows = []
    for L, h in heads:
        one = lambda r: {"score_drop": r["score_drop"], "causal": r["score_drop"][str(S)],
                         "b_side_delta": r.get("b_side_delta"), "drift_orth_b": r.get("drift_orth_b")}
        out_rows.append({"layer": L, "head": h, "label": f"{L}:H{h}", "first": one(f[(L, h)]),
                         "second": one(s2[(L, h)]), "both": one(b[(L, h)])})
    col = lambda k: [r[k]["causal"] for r in out_rows]
    corr = {"first_second": spearman(col("first"), col("second")), "first_both": spearman(col("first"), col("both")),
            "second_both": spearman(col("second"), col("both"))}
    # 両方で止めた効き目と、片方ずつの効き目の和の差(回の間で補い合うか、打ち消し合うか)
    for r in out_rows:
        r["both_minus_sum"] = r["both"]["causal"] - r["first"]["causal"] - r["second"]["causal"]
    out = {"name": name, "direction": ds.name, "contrast_sha": cs.sha, "score_layers": st.score_layers, "select_layer": S,
           "positions": positions, "max_examples": max_examples, "layers": layers, "layer_rows": layer_rows,
           "n_heads": len(out_rows), "spearman": corr,
           "base_scores": {str(L): {"A": float(st.base["A"][L].mean()), "B": float(st.base["B"][L].mean())} for L in st.score_layers},
           "rows": sorted(out_rows, key=lambda r: -max(r["first"]["causal"], r["second"]["causal"], r["both"]["causal"])),
           "seconds": time.perf_counter() - t0}
    p = session.store.heads_dir(session.meta.model_id) / f"{name}.visits.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    out["path"] = str(p)
    return out


# --- 2 段の流れ --------------------------------------------------------------------------------------

def loop_pipeline(session, spec, *, n_first: int = 3, n_second: int = 3, target="B", name: str | None = None,
                  how: str = "max", resume: bool = True, **pipe_over) -> dict:
    """1 回目で効くヘッドを選ぶ(段 0〜27 の候補、段 27 の射影の値で選ぶ)→ その組を 1 回目だけに当てた状態で、2 回目で効くヘッドを
    選ぶ(段 28〜53 の候補、段 54 で選ぶ)→ 2 つを合わせて、両方の回に当てるレシピに直して測り直す(loop_merge)。
    どちらの段も heads pipeline の rank causal(方向 → 基準層 → 因果 → 配分)。2 段目は、1 段目の組を対照集合の両側に重ねて当てる。"""
    import yaml
    lp = _plan(session)
    require_per_visit(session, "1 回目と 2 回目で別々にヘッドを選ぶ流れ")
    t0 = time.perf_counter()
    if isinstance(spec, str | Path):
        p = Path(spec)
        if not p.exists():
            raise GneError("FILE_NOT_FOUND", f"{p} がありません")
        text = p.read_text(encoding="utf-8")
        spec = json.loads(text) if p.suffix == ".json" else yaml.safe_load(text)
    raw = dict(spec)
    raw.pop("about", None)
    base_pipe = dict(raw.pop("pipeline", None) or {})
    for k in ("score_layers", "select_layer", "candidate_layers", "n_heads"):
        base_pipe.pop(k, None)
    base_pipe.update({k: v for k, v in pipe_over.items() if v is not None})
    base_pipe["target"] = target
    # 2 段の流れは、回ごとの候補の層(candidate_layers)と、その回の出口の層での因果スクリーニングで作ったもの。
    # 一括の既定(v0.8.0 から rank writes)に関係なく、v0.7.0 までの選び方(rank causal)で流す
    base_pipe.setdefault("rank", "causal")
    name = str(name or raw.get("name") or "")
    if not re.fullmatch(r"[\w.\-]+", name):
        raise GneError("BAD_ARGS", f"名前 {name!r} は英数字・_・-・. だけにしてください")
    rng = visit_ranges(session)
    hd = session.store.heads_dir(session.meta.model_id)

    def pick(p: dict, n: int) -> dict:
        row = next((r for r in p["allocate"]["table"] if int(r["n_heads"]) == int(n)), None)
        if row is None or row.get("skipped") or not row.get("recipe"):
            raise GneError("NOT_FOUND", f"{p['name']}: {n} ヘッドの組ができませんでした")
        return row

    with session.loop_mode("per_visit"):
        s1 = {**raw, "name": f"{name}_v1",
              "pipeline": {**base_pipe, "score_layers": "loop", "select_layer": lp.lo2 - 1,
                           "candidate_layers": rng["first"], "n_heads": [int(n_first)]}}
        p1 = session.heads_pipeline(s1, resume=resume)
        r1 = pick(p1, n_first)
        stack = lambda side: {"stack": ([raw[side]] if raw.get(side) is not None else []) + [str(r1["recipe"])]}
        s2 = {**raw, "name": f"{name}_v2", "recipe_a": stack("recipe_a"), "recipe_b": stack("recipe_b"),
              "pipeline": {**base_pipe, "score_layers": "loop", "select_layer": lp.n_layer - 2,
                           "candidate_layers": rng["second"], "n_heads": [int(n_second)]}}
        p2 = session.heads_pipeline(s2, resume=resume)
        r2 = pick(p2, n_second)
        rec1, rec2 = Recipe.load(r1["recipe"]), Recipe.load(r2["recipe"])
        explicit = lambda r: [dataclasses.replace(u, direction=u.direction or r.default_direction) if u.mode == "proj" else u
                              for u in r.units]
        visits = {}
        for u in rec1.proj_units + rec2.proj_units:
            visits.setdefault(_vkey(lp, u), []).append(lp.visit(int(u.layer)))
        comb = rec1.like(explicit(rec1) + explicit(rec2), name=f"{name}_loop", loop_edits="per_visit",
                         default_direction=rec1.default_direction, loop_visits={k: sorted(v) for k, v in visits.items()})
        cpath = hd / f"{name}_loop.per_visit.yaml"
        comb.save(cpath)
    merged = loop_merge(session, cpath, how="both", direction=f"{name}_v1", out_name=f"{name}_loop", default=how,
                        positions=base_pipe.get("score_positions") or "last")
    out = {"name": name, "n_first": int(n_first), "n_second": int(n_second), "target": target,
           "first": {"pipeline": p1["path"], "direction": p1["name"], "units": r1["units"], "recipe": r1["recipe"],
                     "by_layer": r1.get("by_layer"), "reached": r1.get("reached"),
                     "base_by_layer": p1["allocate"].get("base_by_layer")},
           "second": {"pipeline": p2["path"], "direction": p2["name"], "units": r2["units"], "recipe": r2["recipe"],
                      "by_layer": r2.get("by_layer"), "reached": r2.get("reached"),
                      "base_by_layer": p2["allocate"].get("base_by_layer")},
           "per_visit_recipe": str(cpath), "merge": merged, "seconds": time.perf_counter() - t0}
    p = hd / f"{name}.loop_pipeline.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    out["path"] = str(p)
    return out


# --- 仕上げ: 両方の回に当てる形に直す ---------------------------------------------------------------

def merge_units(session, recipe: Recipe, how: str) -> tuple[Recipe, list[dict]]:
    """per_visit のレシピ → shared のレシピ。ユニットの層の番号(実行の段)はそのまま残す(shared では、その段の重みを持つ層の
    ヘッドを両方の回で書き換え、方向は own ならその段の r)。同じヘッド(重みを持つ層とヘッドが同じ)を両方の回で選んだ組は、
    how=compose なら 2 つとも残し(続けて掛ける)、max なら |s| の大きいほうだけを残す。返り値は (レシピ, 両方の回で選んだヘッドの表)。"""
    from .direction import base_layer_of
    from .loop import twin_effect
    lp = _plan(session)
    if how not in ("compose", "max"):
        raise GneError("BAD_ARGS", "how は compose(続けて掛ける)か max(強いほうだけ)です")
    groups: dict[tuple[int, int, str], list] = {}
    for u in recipe.proj_units:                      # ヘッドとニューロンは番号が重なっても別(種類も鍵に入れる)
        groups.setdefault((lp.stored(int(u.layer)), _idx(u), u.kind), []).append(u)
    drop, twins = set(), []
    for (Ls, h, kind), us in groups.items():
        if len(us) < 2:
            continue
        us = sorted(us, key=lambda u: int(u.layer))
        u1, u2 = us[0], us[-1]
        d1 = u1.direction or recipe.default_direction
        d2 = u2.direction or recipe.default_direction
        L1 = int(u1.layer) if recipe.direction_layer == "own" else None
        L2 = int(u2.layer) if recipe.direction_layer == "own" else None
        r1 = session.direction_set(d1)
        r2 = session.direction_set(d2)
        v1 = r1.unit(L1 if L1 is not None else base_layer_of(r1)).astype(np.float64)
        v2 = r2.unit(L2 if L2 is not None else base_layer_of(r2)).astype(np.float64)
        c = float(v1 @ v2 / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-12))
        e1, e2 = twin_effect(float(u1.k), float(u2.k), c)
        keep = max(us, key=lambda u: abs(float(u.k)))
        if how == "max":
            drop |= {id(u) for u in us if u is not keep}
        twins.append({"head": f"{Ls}:{'N' if kind == 'neuron' else 'H'}{h}", "units": [u.label() for u in us], "s": [float(u.k) for u in us],
                      "layers": [int(u.layer) for u in us], "cos": c,
                      "compose_effective": {"along_r1": e1, "along_r2": e2}, "max_kept": keep.label(),
                      "max_s": float(keep.k)})
    units = [u for u in recipe.units if id(u) not in drop]
    other = sorted({u.mode for u in units if u.mode not in ("proj", "add")})
    if other:
        session.warn(f"{'・'.join(other)} のユニットは、両方の回に当てる形では、その段の重みを持つ層を 2 回とも書き換えます")
    visits = dict(recipe.loop_visits or {})
    if not visits:
        for u in recipe.proj_units:
            visits.setdefault(_vkey(lp, u), []).append(lp.visit(int(u.layer)))
    out = recipe.like(units, loop_edits="shared", loop_visits={**{k: sorted(set(v)) for k, v in visits.items()},
                                                               "_merge": how})
    return out, twins


def measure(session, direction: str, recipes: dict[str, Recipe | None], *, contrast=None, layers: list[int] | None = None,
            positions: str = "last", max_examples: int | None = None, target_layer: int | None = None) -> dict:
    """レシピごとに、測る層ごとの A 側・B 側の射影の値の平均・B 側の平均を超えて残る A 側の質問の数・B 側の直交成分のずれ・
    A と B の差のうち縮めた割合(closure)を測る。recipes の値 None は無介入。per_visit のレシピは、2 回目の層に別の名前が
    ある llama.cpp でだけ測れる(無ければ skipped)。"""
    from .direction import Scorer, drift
    ds, cs, _ = session._direction_contrast(direction, contrast)
    session.engine.free_steer()
    Ls = list(layers) if layers else (loop_score_layers(session) if session.weights.loop is not None else None)
    if not Ls:
        raise GneError("BAD_ARGS", "測る層がありません")
    sc = Scorer(session, ds, cs, layers=Ls, positions=positions, max_examples=max_examples)
    base = sc.run(None, None, sides="AB", keep=True)
    rows = {}
    names = session.engine.loop_runtime_names() if session.weights.loop is not None else True
    for key, r in recipes.items():
        if r is None:
            got = base
        elif r.loop_edits == "per_visit" and not names:
            rows[key] = {"skipped": "この llama.cpp は 2 回目の層に別の名前が無いので、回ごとに当てた値は測れません"}
            continue
        else:
            vecs = session.vectors(r)[0] if r.add_units else None
            got = sc.run(session.lora_for(r), vecs, sides="AB", keep=True)
        per = {}
        for L in Ls:
            a0, b0 = float(base["A"]["values"][L].mean()), float(base["B"]["values"][L].mean())
            a1, b1 = float(got["A"]["values"][L].mean()), float(got["B"]["values"][L].mean())
            gap = a0 - b0
            per[str(L)] = {"A": a1, "B": b1, "remain": int((got["A"]["values"][L] > b0).sum()),
                           "closure": (a0 - a1) / gap if abs(gap) > 1e-9 else None,
                           "b_drift_orth": drift(base["B"]["resid"][L], got["B"]["resid"][L], sc.r[L])[1]}
        rows[key] = {"by_layer": per, "loop_edits": None if r is None else (r.loop_edits or session.loop_edits_mode())}
    return {"direction": ds.name, "layers": Ls, "n_a": sc.n("A"), "n_b": sc.n("B"), "rows": rows}


def loop_merge(session, recipe, *, how: str = "max", direction: str | None = None, contrast=None,
               out_name: str | None = None, default: str = "max", positions: str = "last", measure_it: bool = True) -> dict:
    """回ごとに選んだレシピ(per_visit)を、両方の回に当てるレシピ(shared)に直して書き出し、両方に当てた状態で測り直す。
    how=compose / max / both(両方を書き出して並べる。代表は default)。結果のレシピは修正なしの llama.cpp で動く。"""
    lp = _plan(session)
    if isinstance(recipe, Recipe):
        r = recipe
    else:
        r = Recipe.load(recipe if Path(str(recipe)).exists() else session.recipe_path(str(recipe)))
    if (r.loop_edits or "shared") != "per_visit":
        session.warn("このレシピはすでに両方の回に当てる形(shared)です。測り直しだけをします")
    hows = ["compose", "max"] if how == "both" else [how]
    hd = session.store.heads_dir(session.meta.model_id)
    base = out_name or (r.name or "recipe").removesuffix(".per_visit")
    out = {"source": r.name, "how": how, "default": default if how == "both" else how, "recipes": {}, "twins": None}
    merged: dict[str, Recipe] = {}
    for hw in hows:
        m, twins = merge_units(session, r, hw)
        m = m.like(m.units, name=f"{base}.shared-{hw}")
        p = hd / f"{base}.shared-{hw}.yaml"
        m.save(p)
        merged[hw] = m
        out["recipes"][hw] = str(p)
        out["twins"] = twins
    rep = out["default"]
    rp = hd / f"{base}.shared.yaml"
    merged[rep].like(merged[rep].units, name=f"{base}.shared").save(rp)
    out["recipe"] = str(rp)
    out["n_units"] = {"per_visit": len(r.proj_units), **{hw: len(m.proj_units) for hw, m in merged.items()}}
    out["n_heads_distinct"] = len({(lp.stored(int(u.layer)), _idx(u), u.kind) for u in r.proj_units})
    if measure_it:
        d = direction or r.default_direction or next((u.direction for u in r.proj_units if u.direction), None)
        if not d:
            raise GneError("BAD_ARGS", "測る方向が分かりません(--direction)")
        recs = {"none": None, "per_visit": dataclasses.replace(r, loop_edits="per_visit")
                if (r.loop_edits or "shared") == "per_visit" else None}
        if recs["per_visit"] is None:
            recs.pop("per_visit")
        recs.update({f"shared-{hw}": m for hw, m in merged.items()})
        out["measure"] = measure(session, d, recs, contrast=contrast, positions=positions)
    p = hd / f"{base}.loop_merge.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    out["path"] = str(p)
    return out


# --- ループの外の最後の層が方向の成分をどれだけ書き足すか ------------------------------------------------

def tail_decompose(session, direction: str, recipes: dict[str, Recipe | None], *, from_layer: int, to_layer: int,
                   contrast=None, positions: str = "last", max_examples: int | None = None) -> dict:
    """段 from_layer から to_layer までの射影の値の変化を切り分ける(側ごと、質問の平均)。
    (i) 残差の大きさ: 段ごとの射影の値・残差の大きさ・cos(射影の値 ÷ 大きさ)。
    (ii) 方向の違い: to_layer の残差を from_layer の方向で測った値(と、その逆)。
    (iii) 書き足し: 間の各段の attention の書き込み(attn_residual − 前の段の l_out)と MLP の書き込み(ffn_out)を、
         to_layer の方向と from_layer の方向に射影した値。和は「to_layer の残差 − from_layer の残差」の射影に一致する。
    間の段は、2 本の流れを持たない層(ふつうの層)に限る。"""
    from .capture import Capture
    from .direction import Scorer
    from .reverse import combine
    ds, cs, _ = session._direction_contrast(direction, contrast)
    e = session.engine
    lo, hi = int(from_layer), int(to_layer)
    if not 0 <= lo < hi <= e.n_layer - 2:
        raise GneError("BAD_ARGS", f"段は 0 ≦ 始め < 終わり ≦ {e.n_layer - 2} です")
    lp = session.weights.loop
    mid = list(range(lo + 1, hi + 1))
    if lp is not None and any(lp.inner(L) or lp.visit(L) for L in mid):
        raise GneError("BAD_ARGS", f"間の段 {mid} に、2 回実行する層があります(書き込みの切り分けは、ふつうの層だけ)")
    session.engine.free_steer()
    sc = Scorer(session, ds, cs, layers=[lo, hi], positions=positions, max_examples=max_examples)
    Ls = list(range(lo, hi + 1))
    r = {L: ds.unit(L).astype(np.float64) for L in Ls}
    want = {f"l_out-{L}" for L in Ls} | {f"attn_residual-{L}" for L in mid} | {f"ffn_out-{L}" for L in mid}
    cap = Capture(lambda n: n in want)
    ctx = e.probe_ctx(cap, need=sc.max_len)
    out = {"direction": ds.name, "from_layer": lo, "to_layer": hi, "n_a": sc.n("A"), "n_b": sc.n("B"), "rows": {}}
    try:
        for key, rec in recipes.items():
            if rec is not None and rec.loop_edits == "per_visit" and not e.loop_runtime_names():
                out["rows"][key] = {"skipped": "回ごとに当てた値は、2 回目の層に別の名前がある llama.cpp でだけ測れます"}
                continue
            lora = None if rec is None else session.lora_for(rec)
            vecs = None if rec is None or not rec.add_units else session.vectors(rec)[0]
            res = {}
            for side in "AB":
                sv, sl = combine(sc.sides.get(side), vecs, lora)
                e.set_lora(ctx, sl)
                e.set_cvec(ctx, sv)
                acc: dict[str, list] = {}
                for ids, rows in sc.items[side]:
                    e.clear_memory(ctx)
                    e.prefill(ctx, [ids])
                    got = cap.take()
                    miss = sorted(want - set(got))
                    if miss:
                        raise GneError("ARCH_UNSUPPORTED", f"{', '.join(miss[:3])} が捕捉できません")
                    x = {L: got[f"l_out-{L}"][rows].astype(np.float64) for L in Ls}
                    f = lambda k, v: acc.setdefault(k, []).append(float(np.mean(v)))
                    for L in Ls:
                        nrm = np.linalg.norm(x[L], axis=1)
                        f(f"proj-{L}", x[L] @ r[L])
                        f(f"norm-{L}", nrm)
                        f(f"cos-{L}", (x[L] @ r[L]) / np.maximum(nrm, 1e-12))
                    f("to_on_from", x[hi] @ r[lo])
                    f("from_on_to", x[lo] @ r[hi])
                    for L in mid:
                        att = got[f"attn_residual-{L}"][rows].astype(np.float64) - x[L - 1]
                        mlp = got[f"ffn_out-{L}"][rows].astype(np.float64)
                        for tag, rv in (("to", r[hi]), ("from", r[lo])):
                            f(f"attn-{L}-{tag}", att @ rv)
                            f(f"mlp-{L}-{tag}", mlp @ rv)
                res[side] = {k: float(np.mean(v)) for k, v in acc.items()}
            out["rows"][key] = res
    finally:
        e.set_cvec(ctx, None)
        e.set_lora(ctx, None)
    return out
