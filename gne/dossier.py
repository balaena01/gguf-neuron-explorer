"""エージェント向けのまとめ: ニューロン 1 個の調書、モデルの見取り図、層の要約、検索。設計書 §6.13 (b)(d)。

持っているデータ(統計・lens・sweep・逆引き)を集めるだけで、無いものは coverage に false を立てる。
トークン列は文字列で返す。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .steer import GneError


# --- 保存済みデータの読み出し ----------------------------------------------------------------------------

def _units_files(session) -> list[tuple[Path, int, str]]:
    """(units.jsonl, 層, 設計)の一覧。"""
    out = []
    for p in sorted(session.store.runs_dir(session.meta.model_id).glob("*.units.jsonl")):
        try:
            with open(p, encoding="utf-8") as f:
                first = json.loads(f.readline())
        except (json.JSONDecodeError, OSError):
            continue
        meta = p.with_name(p.name.removesuffix(".units.jsonl") + ".meta.json")
        design = json.loads(meta.read_text()).get("design", "?") if meta.exists() else "?"
        out.append((p, int(first["layer"]), design))
    return out


def _read_jsonl(p: Path) -> list[dict]:
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _directions(session) -> list[tuple[str, Path]]:
    return [(p.name.split(".")[0], p) for p in sorted(session.store.reverse_dir(session.meta.model_id).glob("*.directions.npz"))]


def _lens_paths(session, L: int) -> tuple[Path, Path]:
    d = session.store.lens_dir(session.meta.model_id)
    return d / f"logit_L{L}.npz", d / f"steer_L{L}.npz"


def _cos_for_layer(session, z, L: int) -> np.ndarray | None:
    from .reverse import cos_by_layer
    if f"layer_{L}" not in z.files:
        return None
    scale = {L: z[f"scale_{L}"]} if f"scale_{L}" in z.files else None
    return cos_by_layer(session.weights, {L: z[f"layer_{L}"]}, scale)[L]


def _names(session) -> dict[str, str]:
    p = session.store.root / "names" / f"{session.meta.model_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def set_name(session, L: int, n: int, name: str) -> None:
    p = session.store.root / "names" / f"{session.meta.model_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    d = _names(session)
    if name:
        d[f"{L}:{n}"] = name
    else:
        d.pop(f"{L}:{n}", None)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


# --- 調書 ---------------------------------------------------------------------------------------------

def unit_evals(session, L: int, n: int, *, limit: int = 5) -> list[dict]:
    """保存済みの eval(§6.14)のうち、このニューロンをユニットに含むもの(新しい順に limit 件)。"""
    out = []
    files = sorted(session.store.eval_dir(session.meta.model_id).glob("*.json"), key=lambda p: -p.stat().st_mtime)
    for p in files:
        try:
            e = json.loads(p.read_text())
            units = e["recipe"]["units"]
        except (ValueError, KeyError):
            continue
        u = next((x for x in units if x["layer"] == L and x.get("neuron") == n and x.get("kind", "neuron") == "neuron"), None)
        if u is None:
            continue
        rows = e.get("rows", [])
        ok = [r["dose"] for r in rows if r.get("shippable") and r["dose"] > 0]
        top = max(rows, key=lambda r: r["dose"]) if rows else None
        out.append({"name": e.get("name", p.stem), "created": e.get("created"), "when": e.get("when"),
                    "mode": u.get("mode", "add"), "dose": u.get("dose"), "k": u.get("k"), "n_units": len(units),
                    "max_shippable_dose": max(ok) if ok else None,
                    "at_max": None if top is None else {
                        "dose": top["dose"], "capability_drop_pt": top.get("capability_drop_pt"),
                        "ppl_worse_pct": top.get("ppl_worse_pct"), "novelty": top.get("persona", {}).get("novelty")}})
        if len(out) >= limit:
            break
    return out


def gain_tried(session, L: int, n: int) -> list[float]:
    """steer・chat・eval でこのニューロンに当てた gain の倍率(実行時 LoRA の置き場所 <root>/gain/<model_id>/ から)。"""
    from gguf import GGUFReader
    ks = set()
    for p in session.store.gain_dir(session.meta.model_id).glob("*.lora.gguf"):
        try:
            f = GGUFReader(p).fields.get("gne.recipe")
            units = json.loads(f.contents())["units"] if f is not None else []
        except Exception:  # noqa: BLE001  壊れた・書きかけのファイルは飛ばす
            continue
        ks |= {float(u["k"]) for u in units if u.get("mode") == "gain" and u["layer"] == L and u.get("neuron") == n}
    return sorted(ks)


def _pick_rollouts(recs: list[dict], n: int, max_rel: float = 1.6) -> list[dict]:
    """プロンプトごとに |dose| の大きい(ただし max_rel 以下を優先)rollout を 1 本ずつ、足りなければ残りから。"""
    by_prompt: dict[str, list[dict]] = {}
    for r in recs:
        by_prompt.setdefault(r["prompt"], []).append(r)
    first = []
    for rs in by_prompt.values():
        ok = [r for r in rs if (r["rel_push"] or 0) <= max_rel + 1e-9] or rs
        first.append(max(ok, key=lambda r: (abs(r["dose"]), r["dose"])))
    rest = [r for r in recs if r not in first]
    return (first + rest)[:n]


def neuron_dossier(session, L: int, n: int, *, rollouts: int = 4, compute: bool = False) -> dict:
    from .lens import LogitLens, SteerLens, logit_lens_layer
    from .stats import ActStats
    m = session.meta
    if not 0 <= L < m.n_layer or not 0 <= n < m.n_ff_at(L):
        raise GneError("BAD_ARGS", f"L{L} N{n} は範囲外です")
    tok = session._tok
    cov = {"sweep": False, "steer_lens": False, "logit_lens": False, "directions": False, "activation": False,
           "connections": False}
    out: dict = {"layer": L, "neuron": n, "name": _names(session).get(f"{L}:{n}"),
                 "down_norm": float(session.weights.down_norms(L)[n]), "sweep": None, "steer_lens": None,
                 "logit_lens": None, "directions": {}, "activation": None, "connections": None}

    # sweep: その層の run のうち、このニューロンの rollout が一番多いもの
    best = None
    for p, layer, design in _units_files(session):
        if layer != L:
            continue
        units = _read_jsonl(p)
        neu = [u for u in units if u["kind"] == "neuron"]
        hit = next((u for u in neu if u["neuron"] == n), None)
        if hit is None:
            continue
        run = p.with_name(p.name.removesuffix(".units.jsonl") + ".jsonl")
        recs = [r for r in _read_jsonl(run)] if run.exists() else []
        mine = [r for r in recs if r["kind"] == "neuron" and r["neuron"] == n]
        if best is None or len(mine) > best[0]:
            best = (len(mine), p, design, hit, len(neu), mine, [r for r in recs if r["kind"] == "baseline"])
    if best:
        _, p, design, hit, n_ranked, mine, base = best
        out["sweep"] = {"run": p.name.removesuffix(".units.jsonl"), "design": design, "rank": hit.get("rank"),
                        "n_ranked": n_ranked, "mass": hit["mass"], "null_pct": hit["null_pct"],
                        "words": hit["words"][:10], "per_prompt": {w: hit["per_prompt"].get(w, [])
                                                                   for w, _, _ in hit["words"][:10]},
                        "dose": hit["dose"],
                        "rollouts": [{"prompt": r["prompt"], "dose": r["dose"], "text": r["text"], "finish": r["finish"]}
                                     for r in _pick_rollouts(mine, rollouts)],
                        "baselines": [{"prompt": r["prompt"], "text": r["text"]} for r in base]}
        cov["sweep"] = True

    # steer lens
    lp, sp = _lens_paths(session, L)
    sl = SteerLens.load(sp) if sp.exists() else None
    if (sl is None or not sl.done[n]) and compute and L > 0:
        from .lens import steer_lens_layer
        cfg = sl.config if sl is not None else {"dose": 0.8, "contexts": None, "top_k": 12, "random_controls": 16,
                                                  "seed": session.seed}
        sv = session._need_rho([L])                                   # 既存の層ファイルの条件に合わせて追記する
        sl = steer_lens_layer(session.engine, session.weights, sv, L, neurons=np.array([n]), dose=cfg["dose"],
                              contexts=cfg["contexts"], top_k=cfg["top_k"], random_controls=cfg["random_controls"],
                              seed=cfg["seed"], out=sp)
    if sl is not None and sl.done[n]:
        null = sl.null("pos")
        out["steer_lens"] = {"up": tok(sl.up_ids[n]), "down": tok(sl.down_ids[n]),
                             "kl_up": float(sl.kl_pos[n]), "kl_down": float(sl.kl_neg[n]),
                             "kl_null_p95": float(np.percentile(null, 95)) if null.size else None,
                             "pct_up": float(sl.null_pct(sl.kl_pos[n], "pos")),
                             "pct_down": float(sl.null_pct(sl.kl_neg[n], "neg")), "dose": sl.config["dose"]}
        cov["steer_lens"] = True

    # logit lens
    ll, row = None, None
    if lp.exists():
        ll = LogitLens.load(lp)
        hits = np.nonzero(ll.neurons == n)[0]
        row = int(hits[0]) if hits.size else None
    if row is None and compute:
        ll, row = logit_lens_layer(session.weights, L, neurons=np.array([n])), 0
    if row is not None:
        out["logit_lens"] = {"up": tok(ll.up_ids[row]), "down": tok(ll.down_ids[row]), "peak_z": float(ll.peak_z[row])}
        cov["logit_lens"] = True

    # 逆引きの方向
    for name, p in _directions(session):
        z = np.load(p)
        cos = _cos_for_layer(session, z, L)
        if cos is None:
            continue
        out["directions"][name] = {"cos": float(cos[n]), "rank_in_layer": int((np.abs(cos) > abs(cos[n])).sum()) + 1}
        cov["directions"] = True

    # 活性の統計
    ap = session.store.act_path(m.model_id, L)
    if ap.exists():
        a = ActStats.load(ap)
        out["activation"] = {"absmax": float(a.absmax[n]), "q999": float(a.q999[n]), "q99": float(a.q99[n]),
                             "rms": float(a.rms[n]), "mean": float(a.mean[n])}
        cov["activation"] = True
    # eval と gain(CHANGE-01): このニューロンを含む保存済みの eval と、当てたことのある gain の倍率
    out["evals"] = unit_evals(session, L, n)
    out["gain_k"] = gain_tried(session, L, n)
    cov["eval"], cov["gain"] = bool(out["evals"]), bool(out["gain_k"])
    out["coverage"] = cov
    return out


# --- 見取り図・層の要約 --------------------------------------------------------------------------------------

def model_overview(session) -> dict:
    from .lens import SteerLens
    m = session.meta
    info = session.info(load=False)
    resid = session.resid_stats()
    units = _units_files(session)
    dirs = _directions(session)
    dir_layers = {name: set(int(x) for x in np.load(p)["layers"]) for name, p in dirs}
    layers = []
    for L in range(m.n_layer):
        lp, sp = _lens_paths(session, L)
        n_logit = int(np.load(lp)["neurons"].size) if lp.exists() else 0
        n_steer = int(SteerLens.load(sp).done.sum()) if sp.exists() else 0
        rho = None
        if resid is not None and L in resid.layers:
            rho = resid.rho(L)
        layers.append({"layer": L, "rho": rho, "act_stats": session.store.act_path(m.model_id, L).exists(),
                       "logit_lens": n_logit, "steer_lens": n_steer,
                       "sweep_runs": [p.name.removesuffix(".units.jsonl") for p, layer, _ in units if layer == L],
                       "directions": [name for name, ls in dir_layers.items() if L in ls]})
    recipes = sorted(p.stem for p in session.store.recipes_dir().glob("*.yaml"))
    return {"meta": info["meta"], "arch_warnings": info["arch_warnings"], "bake": info["bake"],
            "layers": layers, "directions": [name for name, _ in dirs], "recipes": recipes}


def layer_summary(session, L: int, *, top: int = 20) -> dict:
    from .lens import LogitLens, SteerLens
    m = session.meta
    if not 0 <= L < m.n_layer:
        raise GneError("BAD_ARGS", f"層 {L} は範囲外です")
    tok = session._tok
    resid = session.resid_stats()
    out: dict = {"layer": L, "rho": None, "steer_lens": None, "sweep": [], "logit_lens": None, "directions": []}
    if resid is not None and L in resid.layers:
        i = int(np.nonzero(resid.layers == L)[0][0])
        out["rho"] = {"p10": float(resid.p10[i]), "p50": float(resid.p50[i]), "p90": float(resid.p90[i])}
    lp, sp = _lens_paths(session, L)
    if sp.exists():
        sl = SteerLens.load(sp)
        d = sl.done
        if d.any():
            p95 = {sg: float(np.percentile(sl.null(sg), 95)) if sl.null(sg).size else None for sg in ("pos", "neg")}
            kp, kn = sl.kl_pos[d], sl.kl_neg[d]
            out["steer_lens"] = {
                "n_done": int(d.sum()), "kl_pos": {"p50": float(np.median(kp)), "p90": float(np.percentile(kp, 90))},
                "kl_neg": {"p50": float(np.median(kn)), "p90": float(np.percentile(kn, 90))}, "null_p95": p95,
                "frac_above_null_p95": {"pos": float(np.mean(kp > p95["pos"])) if p95["pos"] is not None else None,
                                        "neg": float(np.mean(kn > p95["neg"])) if p95["neg"] is not None else None}}
    for p, layer, design in _units_files(session):
        if layer != L:
            continue
        units = _read_jsonl(p)
        neu = [u for u in units if u["kind"] == "neuron"]
        ctrl = np.array([u["mass"] for u in units if u["kind"] == "random"])
        out["sweep"].append({
            "run": p.name.removesuffix(".units.jsonl"), "design": design, "n_neurons": len(neu),
            "n_controls": int(ctrl.size),
            "frac_above_all_controls": float(np.mean([u["null_pct"] is not None and u["null_pct"] >= 1.0 for u in neu]))
            if neu else None,
            "control_mass_p95": float(np.percentile(ctrl, 95)) if ctrl.size else None,
            "top": [{"neuron": u["neuron"], "mass": u["mass"], "null_pct": u["null_pct"],
                     "words": [w for w, _, _ in u["words"][:5]]} for u in neu[:top]]})
    if lp.exists():
        ll = LogitLens.load(lp)
        o = np.argsort(-ll.peak_z)[:10]
        out["logit_lens"] = {"n": int(ll.neurons.size),
                             "top_peak": [{"neuron": int(ll.neurons[i]), "peak_z": float(ll.peak_z[i]),
                                           "up": tok(ll.up_ids[i][:6])} for i in o]}
    for name, p in _directions(session):
        cos = _cos_for_layer(session, np.load(p), L)
        if cos is None:
            continue
        o = np.argsort(-np.abs(cos))[:10]
        out["directions"].append({"name": name, "top": [{"neuron": int(i), "cos": float(cos[i])} for i in o]})
    return out


# --- 検索 ------------------------------------------------------------------------------------------------

def search(session, q: str, *, mode: str = "lens", layer: int | None = None, limit: int = 50) -> dict:
    """lens: steer lens / logit lens の上位語に q を含むニューロン。words: sweep の語。name: 付けた名前。"""
    from .lens import LogitLens, SteerLens
    if mode not in ("lens", "words", "name"):
        raise GneError("BAD_ARGS", "mode は lens / words / name です")
    qq = q.casefold().strip()
    if not qq:
        raise GneError("BAD_ARGS", "検索語が空です")
    hits: dict[tuple[int, int], set[str]] = {}
    Ls = range(session.meta.n_layer) if layer is None else [layer]
    if mode == "lens":
        voc = session.vocab
        match = np.array([i for i, p in enumerate(voc.pieces) if qq in p.decode("utf-8", "replace").casefold()],
                         dtype=np.int64)
        for L in Ls:
            lp, sp = _lens_paths(session, L)
            if sp.exists():
                sl = SteerLens.load(sp)
                for where, ids in (("steer+", sl.up_ids), ("steer-", sl.down_ids)):
                    for nn in np.nonzero(np.isin(ids, match).any(axis=1) & sl.done)[0]:
                        hits.setdefault((L, int(nn)), set()).add(where)
            if lp.exists():
                ll = LogitLens.load(lp)
                for where, ids in (("logit+", ll.up_ids), ("logit-", ll.down_ids)):
                    for row in np.nonzero(np.isin(ids, match).any(axis=1))[0]:
                        hits.setdefault((L, int(ll.neurons[row])), set()).add(where)
    elif mode == "words":
        for p, L, _ in _units_files(session):
            if layer is not None and L != layer:
                continue
            for u in _read_jsonl(p):
                if u["kind"] == "neuron" and any(qq in str(w).casefold() for w, _, _ in u["words"][:20]):
                    hits.setdefault((L, int(u["neuron"])), set()).add(f"sweep:{p.name.removesuffix('.units.jsonl')}")
    else:
        for k, v in _names(session).items():
            if qq in v.casefold():
                L, n = (int(x) for x in k.split(":"))
                if layer is None or L == layer:
                    hits.setdefault((L, n), set()).add("name")
    rows = [{"layer": L, "neuron": n, "where": sorted(w)} for (L, n), w in sorted(hits.items(), key=lambda x: (-len(x[1]), x[0]))]
    return {"q": q, "mode": mode, "n_hits": len(rows), "hits": rows[:limit]}
