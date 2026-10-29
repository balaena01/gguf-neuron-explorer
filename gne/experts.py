"""専門家の対照分析(gne experts contrast、CHANGE-05 §E.1)と専門家の調書。

ペルソナあり(A)/なし(B)で同じ返事を読ませ(teacher forcing)、返事のトークンで MoE の層ごと・専門家ごとに:
    usage_shift   選ばれた割合の差(A − B)
    weight_shift  実効のルーティング重み(選ばれなければ 0)の平均の差
    contrib_cos   専門家の寄与(ffn_moe_weighted の e のスライス、選ばれなければ 0)の平均の差と、その層の状態の向き r_L の cos。
                  r_L は方向があればその層の r_L、無ければ l_out-L の平均の差(A − B)
    contrib_norm  寄与の平均の差のノルム ÷ 典型の寄与のノルム(選ばれたトークンでの ‖寄与‖ の平均)
雑音の床は中立どうし(B と B2)の同じ指標。z は床の平均と標準偏差で測った大きさの平均、null_pct は床の z の分布での位置。
読み方: usage_shift が大きい = ペルソナのときに呼ばれる専門家、contrib_cos が大きい = ペルソナの方向を書く専門家。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import yaml

from .steer import GneError

METRICS = ("usage_shift", "weight_shift", "contrib_cos", "contrib_norm")
SIGNED = {"usage_shift", "weight_shift", "contrib_cos"}
DEFAULT_SYSTEM_B2 = "You are a helpful, friendly assistant. Answer clearly and politely."


class _Side:
    """片側(A か B)の層ごとの積算: 選ばれた回数・重みの和・寄与の和・‖寄与‖ の和、返事のトークン数、l_out の和。"""

    def __init__(self, L: int, n_expert: int, n_embd: int):
        self.cnt = np.zeros(n_expert, np.int64)
        self.wsum = np.zeros(n_expert, np.float64)
        self.csum = np.zeros((n_expert, n_embd), np.float64)
        self.cnorm = np.zeros(n_expert, np.float64)
        self.lsum = np.zeros(n_embd, np.float64)
        self.n = 0

    def add(self, sel: np.ndarray, wts: np.ndarray, weighted: np.ndarray, lout: np.ndarray) -> None:
        n_expert = self.cnt.shape[0]
        flat_e = sel.reshape(-1)
        self.cnt += np.bincount(flat_e, minlength=n_expert)
        self.wsum += np.bincount(flat_e, weights=wts.reshape(-1).astype(np.float64), minlength=n_expert)
        c = weighted.reshape(-1, weighted.shape[-1]).astype(np.float64)
        np.add.at(self.csum, flat_e, c)
        self.cnorm += np.bincount(flat_e, weights=np.linalg.norm(c, axis=1), minlength=n_expert)
        self.lsum += lout.sum(0)
        self.n += sel.shape[0]


def _metrics(a: _Side, b: _Side, r: np.ndarray | None) -> dict[str, np.ndarray]:
    na, nb = max(a.n, 1), max(b.n, 1)
    usage = a.cnt / na - b.cnt / nb
    weight = a.wsum / na - b.wsum / nb
    diff = a.csum / na - b.csum / nb                                  # 選ばれないトークンの寄与は 0 として平均
    typical = (a.cnorm + b.cnorm) / np.maximum(a.cnt + b.cnt, 1)
    norm = np.linalg.norm(diff, axis=1) / np.maximum(typical, 1e-9)
    if r is None:
        r = a.lsum / na - b.lsum / nb
    rn = float(np.linalg.norm(r))
    dn = np.linalg.norm(diff, axis=1)
    cos = (diff @ r) / np.maximum(dn * rn, 1e-12) if rn > 0 else np.zeros(len(dn))
    cos[dn == 0] = 0.0
    return {"usage_shift": usage, "weight_shift": weight, "contrib_cos": cos, "contrib_norm": norm,
            "rate_a": a.cnt / na, "rate_b": b.cnt / nb}


def experts_contrast(session, contrast: str | Path, *, name: str | None = None, direction: str | None = None,
                     system_b2: str | None = None, layers: list[int] | None = None, max_tokens: int = 48,
                     progress=None) -> dict:
    """対照集合で MoE の層の専門家ごとの指標を測り、<root>/experts/<model_id>/<name>.candidates.jsonl(z の大きい順)と
    .json に書く。"""
    from .capture import Capture
    from .direction import load_contrast
    from .evalx import _generate
    from .heads import _suffix_len
    from .moe import moe_names, per_token, selected
    t0 = time.perf_counter()
    cs, _ = load_contrast(contrast)
    name = name or cs.name
    sys_b2 = system_b2 or (yaml.safe_load(Path(contrast).read_text(encoding="utf-8")) or {}).get("system_b2") or DEFAULT_SYSTEM_B2
    e, w = session.engine, session.weights
    arch = w.arch
    Ls = [L for L in (layers if layers is not None else range(e.n_layer)) if arch.layers[L].ffn.startswith("moe")]
    if not Ls:
        raise GneError("BAD_ARGS", "MoE の層がありません(gne experts は MoE のモデルの機能です)")
    dname = direction
    if dname is None:
        try:
            session.direction_path(cs.name)
            dname = cs.name
        except GneError:
            dname = None
    dirs = {}
    if dname is not None:
        ds = session.direction_set(dname)
        dirs = {L: np.asarray(ds.raw[L], np.float64) for L in Ls if L in ds.raw}
    n = len(cs.user_prompts)
    steps, done = 1 + n, 0

    def tick(msg):
        nonlocal done
        done += 1
        if progress:
            progress(done, steps, msg)
    convs_a = [[{"role": "system", "content": cs.system_a}, {"role": "user", "content": u}] for u in cs.user_prompts]
    resp = [o["text"] for o in _generate(session, [(c, max_tokens) for c in convs_a], None, None, "all", lambda k: None)]
    tick("experts: responses")
    probe = e.tokenize_prompt(e.render_chat(convs_a[0]))
    names = e.list_names(probe)
    nms = {L: moe_names(names, L) for L in Ls}
    for L, nm in nms.items():
        if not {"argsort", "weights", "weighted"} <= set(nm):
            raise GneError("ARCH_UNSUPPORTED", f"層 {L} の MoE の捕捉名が見つかりません({nm})")
    want = {nm[k] for nm in nms.values() for k in ("argsort", "weights", "weighted")} | {f"l_out-{L}" for L in Ls}
    cap = Capture(lambda nm_: nm_ in want)
    n_embd = session.meta.n_embd
    sides = {s: {L: _Side(L, arch.layers[L].n_expert, n_embd) for L in Ls} for s in ("A", "B", "B2")}
    samples = []
    for i, (u, r) in enumerate(zip(cs.user_prompts, resp, strict=True)):
        tail = [{"role": "user", "content": u}, {"role": "assistant", "content": r}]
        ids = {s: e.tokenize_prompt(e.render_chat([{"role": "system", "content": sy}] + tail, add_generation_prompt=False))
               for s, sy in (("A", cs.system_a), ("B", cs.system_b), ("B2", sys_b2))}
        head_a = e.tokenize_prompt(e.render_chat([{"role": "system", "content": cs.system_a}, {"role": "user", "content": u}]))
        S = min(_suffix_len(ids["A"], ids["B"]), _suffix_len(ids["A"], ids["B2"]))
        p_a = len(head_a) if ids["A"][:len(head_a)] == head_a else None
        if p_a is None or p_a < len(ids["A"]) - S or p_a >= len(ids["A"]):
            session.warn(f"プロンプト {i + 1}: 返事の位置を合わせられないので飛ばしました")
            tick(f"experts: {i + 1}/{n}")
            continue
        n_resp = len(ids["A"]) - p_a                                   # 返事のトークン(どの側でも末尾の同じ並び)
        last = e.n_layer - 1 in Ls                     # 最後の層は出力のトークンだけを計算するので、全位置を出させる
        for s in ("A", "B", "B2"):
            ctx = e.probe_ctx(cap, logits_all=last)
            e.clear_memory(ctx)
            e.set_cvec(ctx, None)
            e.prefill(ctx, [ids[s]], logits_all=last)
            out = cap.take()
            T = len(ids[s])
            for L in Ls:
                nm, li = nms[L], arch.layers[L]
                sel = selected(out[nm["argsort"]], li.n_expert_used)[T - n_resp:]
                wts = per_token(out[nm["weights"]], li.n_expert_used, T)[T - n_resp:]
                wd = per_token(out[nm["weighted"]], li.n_expert_used, T)[T - n_resp:]
                sides[s][L].add(sel, wts, wd, out[f"l_out-{L}"][T - n_resp:])
        samples.append({"prompt": u, "response": r[:200], "n_response": n_resp})
        tick(f"experts: {i + 1}/{n}")
    if not samples:
        raise GneError("BAD_CONTRAST", "返事の位置を合わせられたプロンプトがありません")
    rows, nrows = [], []
    for L in Ls:
        ra = dirs.get(L)
        for side_b, out_rows in (("B", rows), ("B2", nrows)):
            a, b = (sides["A"][L], sides["B"][L]) if side_b == "B" else (sides["B"][L], sides["B2"][L])
            m = _metrics(a, b, ra)
            for ex in range(len(m["usage_shift"])):
                out_rows.append({"layer": L, "expert": ex, **{k: float(m[k][ex]) for k in m}})
    stats = {}
    for k in METRICS:
        pool = np.array([abs(x[k]) if k in SIGNED else x[k] for x in nrows])
        stats[k] = {"mean": float(pool.mean()), "std": float(pool.std()) or 1e-9, "p95": float(np.percentile(pool, 95)),
                    "signed": k in SIGNED}

    def zscore(x):
        return float(np.mean([((abs(x[k]) if k in SIGNED else x[k]) - stats[k]["mean"]) / stats[k]["std"] for k in METRICS]))
    nz = np.sort([zscore(x) for x in nrows])
    for x in rows:
        x["z"] = zscore(x)
        x["null_pct"] = float(np.searchsorted(nz, x["z"], "left") / nz.size)
    rows.sort(key=lambda x: -x["z"])
    out_dir = session.store.experts_dir(session.meta.model_id)
    cand = out_dir / f"{name}.candidates.jsonl"
    with cand.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    d = {"name": name, "contrast": str(Path(contrast).resolve()), "contrast_sha": cs.sha, "direction": dname, "system_b2": sys_b2,
         "layers": Ls, "n_prompts": len(samples), "max_tokens": max_tokens,
         "floors": {k: v["p95"] for k, v in stats.items()}, "null_stats": stats, "top": rows[:50],
         "n_experts": len(rows), "candidates": str(cand), "samples": samples, "seconds": time.perf_counter() - t0}
    p = out_dir / f"{name}.json"
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    return {**d, "path": str(p)}


def load_expert_candidates(session, name: str) -> list[dict]:
    p = session.store.experts_dir(session.meta.model_id) / f"{name}.candidates.jsonl"
    if not p.exists():
        raise GneError("FILE_NOT_FOUND", f"専門家の対照分析 {name} がありません(gne experts contrast で作れます)")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def list_expert_contrasts(session) -> list[dict]:
    d = session.store.experts_dir(session.meta.model_id)
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            x = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        out.append({"name": x.get("name", p.stem), "n_prompts": x.get("n_prompts"), "layers": x.get("layers"),
                    "top": (x.get("top") or [])[:5]})
    return out


def expert_dossier(session, L: int, ex: int, *, top: int = 20) -> dict:
    """専門家の調書: 使用統計(stats act)、対照分析での指標、ニューロンの上位(活性の大きさと書き込みの大きさ)。"""
    from .stats import ActStats
    w = session.weights
    li = w.moe_layer(L)
    if not 0 <= ex < li.n_expert:
        raise GneError("BAD_ARGS", f"L{L} の専門家 {ex} は範囲外です(0..{li.n_expert - 1})")
    d = {"layer": L, "expert": ex, "n_expert": li.n_expert, "n_expert_used": li.n_expert_used, "n_ff": li.n_ff,
         "routing": li.routing, "router_bias": None, "usage": None, "contrasts": [], "neurons": []}
    b = w.router_bias(L)
    if b is not None:
        d["router_bias"] = float(b[ex])
    down = w.expert_down_norms(L)[ex]
    p = session.store.act_path(session.meta.model_id, L)
    a = ActStats.load(p) if p.exists() else None
    if a is not None and a.is_moe:
        d["usage"] = {"n_routed": int(a.n_routed[ex]), "rate": float(a.n_routed[ex] / max(a.n_tokens, 1)),
                      "weight_mean": float(a.weight_mean[ex]), "n_tokens": a.n_tokens,
                      "top_tokens": [{"token": session.engine.token_text(int(t)), "count": int(c)}
                                     for t, c in zip(a.top_tokens[ex], a.top_counts[ex], strict=True) if t >= 0][:12],
                      "reliable": bool(a.n_routed[ex] >= 200)}
        score = a.q999[ex] * down
        order = np.argsort(-score)[:top]
        d["neurons"] = [{"neuron": int(n), "q999": float(a.q999[ex][n]), "down_norm": float(down[n]),
                         "write": float(score[n])} for n in order]
    else:
        order = np.argsort(-down)[:top]
        d["neurons"] = [{"neuron": int(n), "q999": None, "down_norm": float(down[n]), "write": None} for n in order]
    for c in list_expert_contrasts(session):
        try:
            rows = load_expert_candidates(session, c["name"])
        except GneError:
            continue
        for rank, x in enumerate(rows, 1):
            if x["layer"] == L and x["expert"] == ex:
                d["contrasts"].append({"name": c["name"], "rank": rank, "of": len(rows), **{k: x[k] for k in (*METRICS, "z", "null_pct",
                                                                                                         "rate_a", "rate_b")}})
                break
    return d


def experts_sweep(session, L: int, *, ks=(0.0, 2.0), design: str = "quick", prompt_set: str = "en", max_tokens: int = 48,
                  experts: list[int] | None = None, run: str | None = None, top_k: int = 5, progress=None) -> dict:
    """専門家単位のスイープ(CHANGE-05 §E.2): 層 L の各専門家を k 倍(既定 0 = 消す、2 = 強める)するスケールパッチを当て、
    設計のプロンプトで greedy に生成する。1 専門家・1 倍率ごとに「パッチ → 読み直し → 生成 → 戻す」。
    順位は語の出方(gne rank と同じ rank_records)。ランダムな対照の代わりに、同じ層の全専門家の分布を帰無にする
    (null_pct = その層の専門家のうち mass が小さいものの割合)。記録は <root>/runs/<model_id>/<run>.jsonl
    (kind = "expert"、neuron = 専門家の番号、dose = k)。"""
    from .patch import Patcher
    from .rank import rank_records
    from .sweep import load_design
    t0 = time.perf_counter()
    w = session.weights
    li = w.moe_layer(L)
    des = load_design(design, prompt_set=prompt_set)
    keys = list(dict.fromkeys(k for k, _ in des.pairs))              # 倍率はパッチで決めるので、プロンプトだけを使う
    exs = list(range(li.n_expert)) if experts is None else [int(x) for x in experts]
    for ex in exs:
        if not 0 <= ex < li.n_expert:
            raise GneError("BAD_ARGS", f"L{L} の専門家 {ex} は範囲外です(0..{li.n_expert - 1})")
    name = run or f"L{L}-experts-{design}"
    out = session.run_path(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    e0 = session.engine
    ids = {k: e0.tokenize_prompt(e0.render_chat([{"role": "user", "content": des.prompts[k]}])) for k in keys}

    def gen():
        en = session.engine
        ctx = en.steer_ctx()
        en.set_cvec(ctx, None)
        return en.generate_greedy(ctx, [ids[k] for k in keys], max_tokens=max_tokens, top_k=top_k)

    def record(kind, ex, key, k, r):
        return {"model_id": session.meta.model_id, "layer": L, "kind": kind, "neuron": ex, "ref_neuron": ex, "prompt": key,
                "dose": float(k), "dose_unit": "expert_k", "rel_push": 0.0, "add": None, "text": r.text, "finish": r.finish,
                "n_gen": r.n_gen, "token_ids": r.token_ids, "logprob": r.logprob, "top_ids": r.top_ids, "top_lps": r.top_lps,
                "seed": 0, "batch": keys}
    records = [record("baseline", None, key, 0.0, r) for key, r in zip(keys, gen(), strict=True)]
    pt = Patcher(w)
    reload_s, steps = [], len(exs) * len(ks)
    for i, ex in enumerate(exs):
        for j, k in enumerate(ks):
            ps = pt.scale_slab(f"blk.{L}.ffn_down_exps.weight", ex, float(k))
            ps.name = f"sweep-L{L}-E{ex}-k{float(k):g}"
            with session.patched(ps):
                reload_s.append(session.last_patch_reload_s)
                outs = gen()
            records += [record("expert", ex, key, k, r) for key, r in zip(keys, outs, strict=True)]
            if progress:
                progress(i * len(ks) + j + 1, steps, f"experts sweep L{L}")
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    meta = {"model_id": session.meta.model_id, "layer": L, "design": design, "prompt_set": prompt_set, "ks": list(ks),
            "max_tokens": max_tokens, "experts": exs, "kind": "experts"}
    out.with_name(out.name.removesuffix(".jsonl") + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    words = session.vocab.words()
    ranked = [u for u in rank_records(records, word_unit="tokens", token_words=words) if u["kind"] == "expert"]
    masses = np.sort([u["mass"] for u in ranked])
    for u in ranked:
        u["expert"] = u.pop("neuron")
        below = int(np.searchsorted(masses, u["mass"], side="left"))
        u["null_pct"] = below / max(len(masses) - 1, 1)                # 同じ層の専門家どうし(自分を除く)
        u["rank"] = None
    for r_, u in enumerate(ranked, 1):
        u["rank"] = r_
    return {"run": name, "path": str(out), "layer": L, "design": design, "prompt_set": prompt_set, "ks": [float(k) for k in ks],
            "n_experts": len(exs), "n_records": len(records), "top": ranked[:30],
            "reload_s": {"mean": float(np.mean(reload_s)) if reload_s else None, "max": float(np.max(reload_s)) if reload_s else None},
            "seconds": time.perf_counter() - t0}


def moe_route(session, text: str, *, layers: list[int] | None = None, chat: bool = False) -> dict:
    """文を流し、MoE の層ごとに各トークンで選ばれた専門家と実効のルーティング重み(大きい順)を返す(CHANGE-05 §H)。
    chat=True なら text をユーザーの発話としてテンプレートに入れる。position 0 は注意の受け皿なので除く。"""
    from .capture import Capture
    from .moe import moe_names, per_token, selected
    e, w = session.engine, session.weights
    ids = (e.tokenize_prompt(e.render_chat([{"role": "user", "content": text}])) if chat
           else e.tokenize(text, add_special=True, parse_special=False))
    Ls = [L for L in (range(e.n_layer) if layers is None else layers) if w.arch.layers[L].ffn.startswith("moe")]
    if not Ls:
        raise GneError("BAD_ARGS", "MoE の層がありません")
    names = e.list_names(ids[:16])
    nms = {L: moe_names(names, L) for L in Ls}
    want = {nm[k] for nm in nms.values() for k in ("argsort", "weights") if k in nm}
    cap = Capture(lambda n: n in want)
    last = e.n_layer - 1 in Ls
    ctx = e.probe_ctx(cap, logits_all=last)
    e.clear_memory(ctx)
    e.set_cvec(ctx, None)
    e.prefill(ctx, [ids], logits_all=last)
    out = cap.take()
    T = len(ids)
    rows = {}
    for L in Ls:
        nm, li = nms[L], w.arch.layers[L]
        sel = selected(out[nm["argsort"]], li.n_expert_used)
        wts = per_token(out[nm["weights"]], li.n_expert_used, T) if "weights" in nm else np.zeros_like(sel, np.float32)
        rows[str(L)] = [{"experts": [int(x) for x in sel[t]], "weights": [round(float(x), 4) for x in wts[t]]} for t in range(1, T)]
    return {"tokens": [e.token_text(t, special=False) for t in ids[1:]], "n_tokens": T - 1, "layers": rows}
