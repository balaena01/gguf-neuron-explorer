"""Inkling 2 で CHiPS の形の取り除き(ヘッドの書き込みから方向の成分を引く)が、応答の水準で効くかを試す(試し。ふつうの機能には
していない)。gne inkling chips-trial。docs/inkling2.md の「CHiPS の試し」。

- 方向: MLP が読む混ぜた入力(ffn_mix。2 回目・3 回目は既定で繰り返しのゲートの後の ffn_in)の、最後のトークンの A − B。
  層 6〜17 は 3 回目の段、ほかの層はその 1 つの段で作る(層ごとの「基準の方向」)。各段の方向も、測るために作る。
- 射影: 書き込み y(W_o の出力。短い畳み込みの前)から u を引く。直した形は u = normalize(g ⊙ r)、g = (1 + 今のトークンのタップ) / √5
  (短い畳み込みの、チャンネルごとの今のトークンの倍率)。素直な形は u = r。ヘッド h なら W_o[:, blk_h] ← (I − s u uᵀ) W_o[:, blk_h]、
  attention 全体なら W_o ← (I − s u uᵀ) W_o。同じ層の部品は同じ u なので、層ごとにランク 1 の LoRA にまとまる。
  層 6〜17 の重みは 3 回で共有なので、書き換えは 3 回すべてに効く。
- 順位: 部品ごとの「基準の方向に沿った書き込み(短い畳み込みの後、最後のトークン)」の A 側の平均 − B 側の平均。繰り返しの層は
  3 回目の段の値で並べ、1 回目・2 回目の値も並べる。
- 測るもの: 各段の MLP の入力(と最後に混ぜた出力)のその段の方向への射影(A 側・B 側、応答に使う質問の最後のトークン)、
  最後のトークンの分布の差(KL)、応答(gne behave の数え方。思考の欄を除いた回答だけ)。

gne heads writes / pipeline の順位づけと配分は、(1) 書き込みと流れの間に短い畳み込みがある、(2) 基準が残差ではなく混ぜた入力、
(3) 層の番号が段と保存されている層の 2 通り、(4) 重みの読み出しと捕捉が Inkling では止めてある、のでそのままは使えない。
LoRA の書き出し(ablate.write_lora_tensors)と応答の数え方(behave.behave)は共通のものを使う。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .inkling_read import Reader, Weights2, _unit, want

STREAM_START_DEFAULT = True          # 流れが始まる層(0・6・12・18)の attention を候補から外す


def ref_kind(st: dict, gate: str) -> str:
    """段の MLP の入力として読む名前: 2 回目・3 回目は gate="after" なら ffn_in(ゲートの後)、"before" なら ffn_mix。"""
    return "ffn_in" if (st["pass"] >= 2 and gate == "after") else "ffn_mix"


def directions(reader: Reader, A: list, B: list) -> dict:
    """最後のトークンの A − B: 各段の ffn_mix・ffn_in と、最後に混ぜた出力 out_mix。"""
    reader.require_patch()
    plan, d_model = reader.plan, reader.w.meta.n_embd
    seqs = [reader.ids(c) for c in A + B]
    reader.fit(seqs)
    acc = {sd: {"ffn_mix": [], "ffn_in": [], "out": []} for sd in "AB"}
    for i, ids in enumerate(seqs):
        if reader.progress:
            reader.progress(i, len(seqs), "chips-trial: 方向")
        cap = reader.run(ids, want({"ffn_mix", "ffn_in", "out_mix"}))
        T, sd = len(ids), ("A" if i < len(A) else "B")
        m, g = np.zeros((plan.n_steps, d_model)), np.zeros((plan.n_steps, d_model))
        for s in range(plan.n_steps):
            m[s] = cap[f"ffn_mix-{s}"].reshape(T, -1)[-1]
            if f"ffn_in-{s}" in cap:
                g[s] = cap[f"ffn_in-{s}"].reshape(T, -1)[-1]
        acc[sd]["ffn_mix"].append(m)
        acc[sd]["ffn_in"].append(g)
        acc[sd]["out"].append(cap["out_mix"].reshape(T, -1)[-1])
    return {k: np.mean(acc["A"][k], axis=0) - np.mean(acc["B"][k], axis=0) for k in ("ffn_mix", "ffn_in", "out")}


def step_dirs(plan, D: dict, gate: str) -> np.ndarray:
    """各段の方向(単位ベクトル): 2 回目・3 回目は ref_kind に従う。"""
    return np.stack([_unit(D[ref_kind(st, gate)][st["step"]]) for st in plan.steps()])


def layer_refs(plan, D: dict, gate: str) -> dict[int, tuple[int, np.ndarray]]:
    """保存されている層 → (基準の段, 基準の方向)。繰り返しの層は 3 回目(最後の回)の段。"""
    out = {}
    for L in range(plan.n_stored):
        s = plan.steps_of(L)[-1]
        out[L] = (s, _unit(D[ref_kind(plan.step(s), gate)][s]))
    return out


def rank(reader: Reader, A: list, B: list, refs: dict, *, head_tokens: int = 4) -> list[dict]:
    """部品(ヘッドと attention 全体)ごとの、基準の方向に沿った書き込み(短い畳み込みの後、最後のトークン)の A − B を、
    その層が実行されるすべての段で。"""
    plan, w2 = reader.plan, Weights2(reader.w)
    d_model = reader.w.meta.n_embd
    li = reader.w.arch.layers[0]
    nh, hd = li.n_head, li.head_dim
    seqs = [reader.ids(c) for c in A + B]
    reader.fit(seqs)
    sums = {sd: np.zeros((plan.n_steps, nh)) for sd in "AB"}
    cnt = {"A": 0, "B": 0}
    Wo = {L: w2.full(f"blk.{L}.attn_output.weight").astype(np.float64).reshape(d_model, nh, hd) for L in range(plan.n_stored)}
    CW = {L: w2.full(f"blk.{L}.attn_post_conv.weight").astype(np.float64) for L in range(plan.n_stored)}
    for i, ids in enumerate(seqs):
        if reader.progress:
            reader.progress(i, len(seqs), "chips-trial: 順位")
        cap = reader.run(ids, want({"attn_gated"}))
        T, sd = len(ids), ("A" if i < len(A) else "B")
        cnt[sd] += 1
        for st in plan.steps():
            s, L = st["step"], st["layer"]
            g = cap[f"attn_gated-{s}"].reshape(T, nh, hd)[-head_tokens:].astype(np.float64)
            y = np.einsum("dhk,thk->thd", Wo[L], g)                                   # (tok, head, d)
            cw = CW[L]
            n = cw.shape[1]
            o = y[-1].copy()
            for k in range(n):
                t = head_tokens - n + k
                if t >= 0:
                    o += y[t] * cw[:, k]
            o /= math.sqrt(n + 1)
            sums[sd][s] += o @ refs[L][1]
    diff = sums["A"] / max(cnt["A"], 1) - sums["B"] / max(cnt["B"], 1)              # (steps, head)
    rows = []
    for L in range(plan.n_stored):
        ss = plan.steps_of(L)
        ref_s = refs[L][0]
        for h in [*range(nh), None]:
            val = (lambda s: float(diff[s].sum() if h is None else diff[s, h]))  # noqa: E731
            rows.append({"layer": L, "head": h, "ref_step": ref_s, "value": val(ref_s),
                         "by_pass": {str(plan.step(s)["pass"] or 0): val(s) for s in ss}})
    return rows


def region_of(plan, L: int) -> str:
    return "before" if L < plan.core_start else ("loop" if L < plan.core_end else "after")


def candidates(plan, rows: list[dict], *, regions=("loop", "after"), skip_stream_start: bool = STREAM_START_DEFAULT,
               whole: bool = False) -> list[dict]:
    """候補(順位の正の部品、値の大きい順)。regions は before / loop / after。skip_stream_start なら流れが始まる層の attention を外す。"""
    out = []
    for r in rows:
        if (r["head"] is None) != whole or region_of(plan, r["layer"]) not in regions:
            continue
        if skip_stream_start and r["layer"] % plan.block == 0:
            continue
        if r["value"] > 0:
            out.append(r)
    return sorted(out, key=lambda r: -r["value"])


def lora_tensors(reader: Reader, units: list[tuple[int, int | None, float]], refs: dict, *, correct: bool = True) -> dict:
    """部品 (層, ヘッド or None = attention 全体, 強さ s) → LoRA のテンソル {blk.L.attn_output.weight: (A (1, in), B (d, 1))}。
    ΔW = −u (Σ_部品 s uᵀ W_o M_部品)、u = normalize(g ⊙ r)(correct)か r。"""
    w2 = Weights2(reader.w)
    li = reader.w.arch.layers[0]
    hd = li.head_dim
    by_layer: dict[int, list[tuple[int | None, float]]] = {}
    for L, h, s in units:
        by_layer.setdefault(int(L), []).append((h, float(s)))
    out = {}
    for L, us in by_layer.items():
        cw = w2.full(f"blk.{L}.attn_post_conv.weight").astype(np.float64)
        Wo = w2.full(f"blk.{L}.attn_output.weight").astype(np.float64)                 # (d, n_head·hd)
        A, B = layer_lora(Wo, cw, refs[L][1], us, hd, correct=correct)
        out[f"blk.{L}.attn_output.weight"] = (A.astype(np.float32), B.astype(np.float32))
    return out


def layer_lora(Wo: np.ndarray, cw: np.ndarray, r: np.ndarray, us: list[tuple[int | None, float]], hd: int, *,
               correct: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """1 つの層の LoRA (A (1, in), B (d, 1))。ΔW = B A = −u (Σ s uᵀ W_o M)、M はヘッドの列(None は全部)。
    u = normalize(g ⊙ r)、g = (1 + cw[:, −1]) / √(タップ数 + 1)(correct)か r。"""
    r = np.asarray(r, np.float64)
    if correct:
        g = (1.0 + cw[:, -1]) / math.sqrt(cw.shape[1] + 1)
        u = _unit(g * r)
    else:
        u = _unit(r)
    row = u @ Wo
    a = np.zeros_like(row)
    for h, s in us:
        if h is None:
            a += s * row
        else:
            a[h * hd:(h + 1) * hd] += s * row[h * hd:(h + 1) * hd]
    return a.reshape(1, -1), (-u).reshape(-1, 1)


def write_lora(reader: Reader, units, refs, path: Path, *, correct: bool = True) -> str:
    from .ablate import write_lora_tensors
    write_lora_tensors(reader.w, lora_tensors(reader, units, refs, correct=correct), path, name=path.stem,
                       kv={"gne.inkling_chips_trial": json.dumps({"units": units, "correct": correct})})
    return str(path)


def continuations(reader: Reader, convs: list, n: int = 32) -> list[list[int]]:
    """書き換える前のモデルの、貪欲に生成した最初の n トークン(思考の欄の始まりを含む)。KL を測る位置に使う。"""
    e = reader.e
    ctx = e.steer_ctx()
    e.set_lora(ctx, None)
    out = []
    for c in convs:
        r = e.generate_greedy(ctx, [reader.ids(c)], max_tokens=n, stop_on_eog=False)
        out.append([int(t) for t in r[0].token_ids][:n])
    return out


def _free_probe(e) -> None:
    if e._probe is not None:
        e._probe.free()
        e._probe = None


def measure(reader: Reader, convs: list, conts: list[list[int]], sdirs: np.ndarray, out_dir: np.ndarray, gate: str,
            lora: str | None = None) -> dict:
    """質問の後に、書き換える前のモデルの続き(conts)をつないで 1 回流し、
    - 最後の質問のトークンでの、各段の MLP の入力のその段の方向への射影 (n, steps) と、最後に混ぜた出力の射影 (n,)
    - 続きのトークンを予想する位置での、書き換える前と後の分布の差 KL(元 ‖ 後)の平均 (n,)(lora が無ければ 0)
    を返す。最後の質問のトークンは役割の見出しを予想するだけでほぼ決まっているので、KL は続きの位置で測る。"""
    from .engine import log_softmax
    plan, e = reader.plan, reader.e
    seqs = [reader.ids(c) for c in convs]
    fulls = [ids + cont for ids, cont in zip(seqs, conts, strict=True)]
    reader.fit(fulls)
    proj = np.zeros((len(seqs), plan.n_steps))
    outp = np.zeros(len(seqs))
    kls = np.zeros(len(seqs))
    flips = np.zeros(len(seqs))
    dmax = np.zeros(len(seqs))
    names = want({"ffn_mix", "ffn_in", "out_mix"})
    for i, (ids, full) in enumerate(zip(seqs, fulls, strict=True)):
        logps = []
        for lr in ([lora, None] if lora else [None]):
            reader.cap.set_wanted(names if lr == lora else (lambda n: False))
            ctx = e.probe_ctx(reader.cap, logits_all=True, need=max(reader.need, len(full)))
            e.set_lora(ctx, [(lr, 1.0)] if lr else None)
            try:
                reader.cap.reset()
                e.clear_memory(ctx)
                res = e.prefill(ctx, [full], logits_all=True)
                cap = reader.cap.take()
            finally:
                e.set_lora(ctx, None)
            T, p = len(full), len(ids) - 1
            if lr == lora:
                for st in plan.steps():
                    s = st["step"]
                    proj[i, s] = cap[f"{ref_kind(st, gate)}-{s}"].reshape(T, -1)[p] @ sdirs[s]
                outp[i] = cap["out_mix"].reshape(T, -1)[p] @ out_dir
            logps.append(log_softmax(np.asarray(res.all_logits[0][p:T - 1], np.float32)))
        if lora:
            kls[i] = float(kl(logps[1], logps[0]).mean())
            flips[i] = float((logps[1].argmax(-1) != logps[0].argmax(-1)).mean())
            dmax[i] = float(np.abs(logps[1] - logps[0]).max())
    return {"proj": proj, "out": outp, "kl": kls, "flip": flips, "dmax": dmax}


def kl(base_logp: np.ndarray, logp: np.ndarray) -> np.ndarray:
    """KL(元 ‖ 書き換え後)(質問ごと)。"""
    p = np.exp(base_logp.astype(np.float64))
    return (p * (base_logp - logp)).sum(axis=-1)


def internal_row(plan, base: dict, ed: dict) -> dict:
    """A − B の差が、書き換えでどれだけ残ったか(段ごと・区分ごと・最後の出力)と、A 側・B 側の平均の動き。"""
    gap0 = base["A"]["proj"].mean(0) - base["B"]["proj"].mean(0)
    gap1 = ed["A"]["proj"].mean(0) - ed["B"]["proj"].mean(0)
    keep = gap1 / np.where(np.abs(gap0) > 1e-9, gap0, np.nan)
    o0 = base["A"]["out"].mean() - base["B"]["out"].mean()
    o1 = ed["A"]["out"].mean() - ed["B"]["out"].mean()
    regions = {}
    for st in plan.steps():
        k = ("before" if st["layer"] < plan.core_start else "after") if not st["pass"] else f"pass{st['pass']}"
        regions.setdefault(k, []).append(keep[st["step"]])
    return {"keep_by_step": keep.tolist(), "keep_by_region": {k: float(np.nanmedian(v)) for k, v in regions.items()},
            "keep_out": float(o1 / o0) if abs(o0) > 1e-9 else None,
            "a_move_out": float(ed["A"]["out"].mean() - base["A"]["out"].mean()),
            "b_move_out": float(ed["B"]["out"].mean() - base["B"]["out"].mean()), "gap_out_base": float(o0),
            "kl_a": float(ed["A"]["kl"].mean()), "kl_b": float(ed["B"]["kl"].mean()),
            "flip_a": float(ed["A"]["flip"].mean()), "flip_b": float(ed["B"]["flip"].mean()),
            "dmax_b": float(ed["B"]["dmax"].max())}


def replies(session, convs: list, lora: str | None, match: str, max_tokens: int) -> dict:
    from .behave import behave
    r = behave(session, convs, lora=[(lora, 1.0)] if lora else None, match=match, max_tokens=max_tokens)
    run = r["runs"][0]
    return {"counts": run["counts"], "replies": [{"i": x["i"], "text": x["text"], "full": x.get("full"), "finish": x["finish"],
                                                  "loop": x["loop"]} for x in run["replies"]]}


def loop_effect(reader: Reader, A: list, B: list, refs: dict, sdirs: np.ndarray, lora: str | None) -> dict:
    """attention の書き込み(畳み込みの後)の、その層の基準の方向への射影と、繰り返しの MLP 側のメッセージ(gain ⊙ メッセージ)の、
    その段の方向への射影を、最後のトークンの A − B で段ごとに。"""
    plan, e = reader.plan, reader.e
    seqs = [reader.ids(c) for c in A + B]
    reader.fit(seqs)
    acc = {sd: {"attn": np.zeros(plan.n_steps), "msg": np.zeros(plan.n_steps), "n": 0} for sd in "AB"}
    for i, ids in enumerate(seqs):
        reader.cap.set_wanted(want({"attn_out", "loop_msg", "loop_gain_ffn"}))
        ctx = e.probe_ctx(reader.cap, need=max(reader.need, len(ids)))
        e.set_lora(ctx, [(lora, 1.0)] if lora else None)
        try:
            reader.cap.reset()
            e.clear_memory(ctx)
            e.prefill(ctx, [ids])
            cap = reader.cap.take()
        finally:
            e.set_lora(ctx, None)
        T, sd = len(ids), ("A" if i < len(A) else "B")
        acc[sd]["n"] += 1
        for st in plan.steps():
            s, L = st["step"], st["layer"]
            acc[sd]["attn"][s] += cap[f"attn_out-{s}"].reshape(T, -1)[-1] @ refs[L][1]
            if f"loop_msg-{s}" in cap:
                m = cap[f"loop_msg-{s}"].reshape(T, 4, -1)[-1, 3] * cap[f"loop_gain_ffn-{s}"].reshape(T, -1)[-1]
                acc[sd]["msg"][s] += m @ sdirs[s]
    return {k: (acc["A"][k] / acc["A"]["n"] - acc["B"][k] / acc["B"]["n"]).tolist() for k in ("attn", "msg")}


def load_spec(path) -> dict:
    """pirate_tone の形のひな形: 方向と順位は user_prompts(system_a / system_b)、応答は validate.prompts(方向に使っていない質問)。"""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if not (d.get("user_prompts") and d.get("system_a") and (d.get("validate") or {}).get("prompts") and d["validate"].get("match")):
        from .steer import GneError
        raise GneError("BAD_ARGS", f"{path}: user_prompts・system_a・validate.prompts・validate.match のあるひな形を渡してください"
                       "(例 configs/pipeline/pirate_tone.json)")
    conv = lambda q, sys_: ([{"role": "system", "content": sys_}] if sys_ else []) + [{"role": "user", "content": q}]  # noqa: E731
    return {"build_a": [conv(q, d["system_a"]) for q in d["user_prompts"]],
            "build_b": [conv(q, d.get("system_b")) for q in d["user_prompts"]],
            "val_a": [conv(q, d["system_a"]) for q in d["validate"]["prompts"]],
            "val_b": [conv(q, d.get("system_b")) for q in d["validate"]["prompts"]],
            "match": d["validate"]["match"], "name": d.get("name") or Path(path).stem}


def trial(session, spec_path, out_dir: Path, *, ks=(4, 8, 16, 32), strengths=(1.0, 2.0, 4.0, 8.0), whole_ks=(2, 4),
          whole_strengths=(1.0, 2.0, 4.0), gate: str = "after", skip_stream_start: bool = STREAM_START_DEFAULT,
          n_build: int | None = None, n_val_a: int = 30, n_val_b: int = 15, n_kl: int = 30, max_tokens: int = 1024,
          region_k: int = 8, progress=None, log=None) -> dict:
    """試しの一式(方向 → 順位 → 組ごとに LoRA を作り、中の値・最後のトークンの KL・応答を測る)。結果は組ごとに out_dir/result.json に
    足していき、同じ out_dir で流し直すと済んだ組は飛ばす。"""
    sp = load_spec(spec_path)
    reader = Reader(session, progress=progress)
    reader.require_patch()
    reader.e.allow_lora_trial = True                  # engine.py: Inkling で LoRA を通すのは、この試しだけ
    plan = reader.plan
    out_dir.mkdir(parents=True, exist_ok=True)
    rp = out_dir / "result.json"
    res = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
    say = log or (lambda *_: None)

    def save():
        rp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    bA, bB = sp["build_a"][:n_build], sp["build_b"][:n_build]
    vA, vB = sp["val_a"][:max(n_val_a, n_kl)], sp["val_b"][:max(n_val_b, n_kl)]
    dz = out_dir / "directions.npz"
    if dz.exists():
        z = np.load(dz)
        D = {k: z[k] for k in ("ffn_mix", "ffn_in", "out")}
    else:
        D = directions(reader, bA, bB)
        np.savez(dz, **D)
    refs, sdirs, odir = layer_refs(plan, D, gate), step_dirs(plan, D, gate), _unit(D["out"])
    if "rank" not in res:
        res["rank"] = rank(reader, bA, bB, refs)
        res["config"] = {"gate": gate, "skip_stream_start": skip_stream_start, "n_build": len(bA), "n_val_a": n_val_a,
                         "n_val_b": n_val_b, "n_kl": n_kl, "max_tokens": max_tokens, "spec": str(spec_path)}
        save()
    rows = res["rank"]
    if "conts" not in res:
        reader.e.free_steer()
        res["conts"] = {"A": continuations(reader, vA[:n_kl]), "B": continuations(reader, vB[:n_kl])}
        save()
    conts = res["conts"]

    def internal(lora):
        reader.e.free_steer()                             # GPU のメモリ: 生成用と捕捉用のコンテキストを同時に持たない
        out = {sd: measure(reader, (vA if sd == "A" else vB)[:n_kl], conts[sd], sdirs, odir, gate, lora) for sd in "AB"}
        _free_probe(reader.e)
        return out
    base = internal(None)
    if "base" not in res:
        res["base"] = {"A": replies(session, vA[:n_val_a], None, sp["match"], max_tokens),
                       "B": replies(session, vB[:n_val_b], None, sp["match"], max_tokens),
                       "proj_a": base["A"]["proj"].mean(0).tolist(), "proj_b": base["B"]["proj"].mean(0).tolist(),
                       "out_a": float(base["A"]["out"].mean()), "out_b": float(base["B"]["out"].mean())}
        save()
    res.setdefault("sets", {})

    def run_set(key: str, units, *, correct=True, with_replies=True):
        if key in res["sets"] and (res["sets"][key].get("A") or not with_replies):
            return res["sets"][key]
        path = out_dir / f"{key}.lora.gguf"
        write_lora(reader, units, refs, path, correct=correct)
        ed = internal(str(path))
        row = {"units": [[int(L), h, float(s)] for L, h, s in units], "correct": correct, **internal_row(plan, base, ed)}
        if with_replies:
            row["A"] = replies(session, vA[:n_val_a], str(path), sp["match"], max_tokens)
            row["B"] = replies(session, vB[:n_val_b], str(path), sp["match"], max_tokens)
        reader.e.drop_lora(path)
        res["sets"][key] = row
        save()
        say(key, row)
        return row
    heads = candidates(plan, rows, skip_stream_start=skip_stream_start)
    for k in ks:
        for s in strengths:
            run_set(f"heads_k{k}_s{s:g}", [(r["layer"], r["head"], s) for r in heads[:k]])
    whole = candidates(plan, rows, skip_stream_start=skip_stream_start, whole=True)
    for k in whole_ks:
        for s in whole_strengths:
            run_set(f"attn_k{k}_s{s:g}", [(r["layer"], None, s) for r in whole[:k]])
    # 層 0〜5 だけ・層 18〜23 だけ(同じくらい最後の出力の差を縮める強さで比べる)
    region_sets = {}
    for reg in ("before", "after"):
        cand = candidates(plan, rows, regions=(reg,), skip_stream_start=skip_stream_start)[:region_k]
        region_sets[reg] = cand
        for s in strengths:
            run_set(f"{reg}_k{region_k}_s{s:g}", [(r["layer"], r["head"], s) for r in cand], with_replies=False)
    red = {reg: {s: 1 - res["sets"][f"{reg}_k{region_k}_s{s:g}"]["keep_out"] for s in strengths} for reg in region_sets}
    target = 0.8 * min(max(v.values()) for v in red.values())
    res["region_target_reduction"] = target
    for reg, cand in region_sets.items():
        s_best = min(strengths, key=lambda s: abs(red[reg][s] - target))
        res.setdefault("region_pick", {})[reg] = s_best
        run_set(f"{reg}_k{region_k}_s{s_best:g}", [(r["layer"], r["head"], s_best) for r in cand])
    # いちばん良い組と悪い組(決め方は summary の説明)。いちばん良い組を素直な射影でも流す
    best, worst = pick(res)
    res["best"], res["worst"] = best, worst
    if best:
        b = res["sets"][best]
        run_set(f"{best}_naive", [tuple(u) for u in b["units"]], correct=False)
        res["loop_effect"] = {"base": loop_effect(reader, vA[:n_kl], vB[:n_kl], refs, sdirs, None)}
        path = out_dir / f"{best}.lora.gguf"
        write_lora(reader, [tuple(u) for u in b["units"]], refs, path)
        res["loop_effect"]["best"] = loop_effect(reader, vA[:n_kl], vB[:n_kl], refs, sdirs, str(path))
        reader.e.drop_lora(path)
    save()
    return res


def broken(row: dict) -> int:
    """壊れた応答の数(A 側と B 側の、空・繰り返し・長さの上限で切れた、の和)。"""
    return sum(row[sd]["counts"][k] for sd in ("A", "B") if sd in row for k in ("empty", "loop", "at_limit"))


def pick(res: dict) -> tuple[str | None, str | None]:
    """いちばん良い組: 壊れた応答が無介入より 2 つ以上増えず、B 側の一致(海賊の言葉)が増えない組のうち、A 側の一致がいちばん少ない組
    (同じなら B 側の KL が小さい組)。いちばん悪い組: 壊れた応答がいちばん多い組(同じなら B 側の KL が大きい組)。素直な射影の組と、
    応答を数えていない組は選ばない。"""
    b0 = broken(res["base"])
    m0b = res["base"]["B"]["counts"]["match"]
    sets = {k: v for k, v in res["sets"].items() if "A" in v and not k.endswith("_naive")}
    ok = [k for k, v in sets.items() if broken(v) <= b0 + 2 and v["B"]["counts"]["match"] <= m0b]
    best = min(ok, key=lambda k: (sets[k]["A"]["counts"]["match"], sets[k]["kl_b"])) if ok else None
    worst = max(sets, key=lambda k: (broken(sets[k]), sets[k]["kl_b"])) if sets else None
    return best, worst


def check_lora(reader: Reader, refs: dict, L: int, h: int, s: float, path: Path, *, correct: bool = True) -> float:
    """1 つのヘッドの LoRA を当てたとき、その層の最初の段の W_o の出力の変化が、計算どおり −s u (uᵀ W_o[:, blk_h] x_h) か
    (x_h はゲートの後のヘッドの出力。その段より前は書き換わらないので、当てる前と後で同じ)。相対誤差を返す。"""
    plan, e, w2 = reader.plan, reader.e, Weights2(reader.w)
    hd = reader.w.arch.layers[0].head_dim
    write_lora(reader, [(L, h, s)], refs, path, correct=correct)
    step = plan.steps_of(L)[0]
    ids = reader.ids("Hello there, how are you today?")
    reader.fit([ids])
    got = []
    for lora in (None, str(path)):
        reader.cap.set_wanted(want({"attn_gated", "attn_wo"}, [step]))
        ctx = e.probe_ctx(reader.cap, need=max(reader.need, len(ids)))
        e.set_lora(ctx, [(lora, 1.0)] if lora else None)
        try:
            reader.cap.reset()
            e.clear_memory(ctx)
            e.prefill(ctx, [ids])
            got.append(reader.cap.take())
        finally:
            e.set_lora(ctx, None)
    e.drop_lora(path)
    T = len(ids)
    x = got[0][f"attn_gated-{step}"].reshape(T, -1)[:, h * hd:(h + 1) * hd].astype(np.float64)
    Wo = w2.full(f"blk.{L}.attn_output.weight").astype(np.float64)[:, h * hd:(h + 1) * hd]
    A, B = lora_tensors(reader, [(L, h, s)], refs, correct=correct)[f"blk.{L}.attn_output.weight"]
    u = -B.reshape(-1).astype(np.float64)
    pred = -s * np.outer((x @ Wo.T) @ u, u)
    d = got[1][f"attn_wo-{step}"].reshape(T, -1) - got[0][f"attn_wo-{step}"].reshape(T, -1)
    return float(np.linalg.norm(d - pred) / max(np.linalg.norm(pred), 1e-30))


# --- 単位の種類を広げる: ヘッド・attention 全体・MLP 全体・専門家のニューロン ------------------------------------------------------

def _taps(cw: np.ndarray, r: np.ndarray) -> np.ndarray:
    """短い畳み込みのタップごとの「r に沿った成分を取り出す向き」(4, d): 前のトークン k は r ⊙ w_k / √5、今のトークンは
    r ⊙ (1 + w_last) / √5(= g ⊙ r)。書き込み y_t をこれと内積すると、最後のトークンの畳み込みの後の r の成分への寄与になる。"""
    n = cw.shape[1]
    return np.stack([r * cw[:, k] / math.sqrt(n + 1) for k in range(n - 1)] + [r * (1.0 + cw[:, -1]) / math.sqrt(n + 1)])


def expert_q(reader: Reader, refs: dict, progress=None) -> dict:
    """層ごと・専門家ごとに q[L, e, k] = v_k · W_down_e(k は短い畳み込みのタップ、v は _taps)を、専門家 1 つずつ戻しながら作る
    (全部の重みを一度にメモリに置かない)。専門家ごとの倍率(ffn_down_exps.scale。無い層は 1)も返す。"""
    import resource
    import time
    w2 = Weights2(reader.w)
    plan = reader.plan
    li = reader.w.arch.layers[0]
    nE, nf = li.n_expert, li.n_ff
    t0 = time.perf_counter()
    q = np.zeros((plan.n_stored, nE, 4, nf), np.float32)
    sc = np.ones((plan.n_stored, nE), np.float32)
    for L in range(plan.n_stored):
        if progress:
            progress(L, plan.n_stored, "Inkling 2: 専門家の重み")
        V = _taps(w2.full(f"blk.{L}.ffn_post_conv.weight").astype(np.float64), refs[L][1].astype(np.float64)).astype(np.float32)
        for e in range(nE):
            q[L, e] = V @ w2.slab(f"blk.{L}.ffn_down_exps.weight", e)
        s = w2.scale(f"blk.{L}.ffn_down_exps.scale")
        if s is not None:
            sc[L] = s
    return {"q": q, "scale": sc, "seconds": time.perf_counter() - t0,
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def rank_all(reader: Reader, A: list, B: list, refs: dict, eq: dict, *, keep: int = 2000, head_tokens: int = 4) -> dict:
    """全部の種類の部品の、基準の方向に沿った書き込み(短い畳み込みの後、最後のトークン)の A − B を、その層の全部の段で。
    ニューロンの書き込みは「選んだ重み × 専門家の倍率 × 活性 × q」(選ばれなかったトークンでは 0)を、最後の 4 トークンについて足す。
    ニューロンは区分(before / loop / after)ごとに、基準の段の値の大きい順に keep 個だけ残す。"""
    import time
    plan, w2 = reader.plan, Weights2(reader.w)
    d_model = reader.w.meta.n_embd
    li = reader.w.arch.layers[0]
    nh, hd, nE, nf, k_used = li.n_head, li.head_dim, li.n_expert, li.n_ff, li.n_expert_used
    q, sc = eq["q"], eq["scale"]
    seqs = [reader.ids(c) for c in A + B]
    reader.fit(seqs)
    t0 = time.perf_counter()
    heads = {sd: np.zeros((plan.n_steps, nh)) for sd in "AB"}
    whole = {sd: np.zeros((plan.n_steps, 2)) for sd in "AB"}
    neur = {sd: np.zeros((plan.n_steps, nE, nf), np.float32) for sd in "AB"}
    sel = {sd: np.zeros((plan.n_steps, nE)) for sd in "AB"}               # 最後のトークンで専門家が選ばれた回数
    cnt = {"A": 0, "B": 0}
    errs = []
    Wo = {L: w2.full(f"blk.{L}.attn_output.weight").astype(np.float64).reshape(d_model, nh, hd) for L in range(plan.n_stored)}
    CW = {L: w2.full(f"blk.{L}.attn_post_conv.weight").astype(np.float64) for L in range(plan.n_stored)}
    for i, ids in enumerate(seqs):
        if reader.progress:
            reader.progress(i, len(seqs), "chips-trial: 全部の部品の順位")
        cap = reader.run(ids, want({"attn_gated", "attn_out", "ffn_out", "ffn_moe_situ", "ffn_moe_argsort", "ffn_moe_weights_scaled"}))
        T, sd = len(ids), ("A" if i < len(A) else "B")
        cnt[sd] += 1
        for st in plan.steps():
            s, L = st["step"], st["layer"]
            r = refs[L][1]
            g = cap[f"attn_gated-{s}"].reshape(T, nh, hd)[-head_tokens:].astype(np.float64)
            y = np.einsum("dhk,thk->thd", Wo[L], g)
            o = y[-1].copy()
            for k in range(CW[L].shape[1]):
                t = head_tokens - CW[L].shape[1] + k
                if t >= 0:
                    o += y[t] * CW[L][:, k]
            o /= math.sqrt(CW[L].shape[1] + 1)
            heads[sd][s] += o @ r
            whole[sd][s, 0] += cap[f"attn_out-{s}"].reshape(T, -1)[-1] @ r
            fo = cap[f"ffn_out-{s}"].reshape(T, -1)[-1] @ r
            whole[sd][s, 1] += fo
            top = cap[f"ffn_moe_argsort-{s}"].reshape(T, -1)[:, :k_used]
            wts = cap[f"ffn_moe_weights_scaled-{s}"].reshape(T, k_used)
            act = cap[f"ffn_moe_situ-{s}"].reshape(T, k_used, nf)
            tot = 0.0
            for kk in range(4):
                t = T - 4 + kk
                if t < 0:
                    continue
                e = top[t]
                c = (wts[t] * sc[L, e])[:, None] * act[t] * q[L, e, kk]
                np.add.at(neur[sd][s], e, c)
                tot += float(c.sum())
            np.add.at(sel[sd][s], top[T - 1], 1)
            errs.append(abs(tot - fo) / max(abs(fo), 1e-6))
    nA, nB = max(cnt["A"], 1), max(cnt["B"], 1)
    dh = heads["A"] / nA - heads["B"] / nB
    dw = whole["A"] / nA - whole["B"] / nB
    dn = neur["A"] / nA - neur["B"] / nB                                   # (steps, E, n_ff)
    rows = []
    for L in range(plan.n_stored):
        ss = plan.steps_of(L)
        rs = refs[L][0]
        bp = lambda f: {str(plan.step(s)["pass"] or 0): float(f(s)) for s in ss}  # noqa: E731
        for h in range(nh):
            rows.append({"kind": "head", "layer": L, "head": h, "value": float(dh[rs, h]), "by_pass": bp(lambda s, h=h: dh[s, h])})
        rows.append({"kind": "attn", "layer": L, "head": None, "value": float(dw[rs, 0]), "by_pass": bp(lambda s: dw[s, 0])})
        rows.append({"kind": "ffn", "layer": L, "head": None, "value": float(dw[rs, 1]), "by_pass": bp(lambda s: dw[s, 1])})
    nrows = []
    for reg in ("before", "loop", "after"):
        Ls = [L for L in range(plan.n_stored) if region_of(plan, L) == reg]
        vals = np.stack([dn[refs[L][0]] for L in Ls])                          # (len(Ls), E, n_ff)
        flat = vals.reshape(-1)
        idx = np.argpartition(-flat, min(keep, flat.size - 1))[:keep]
        idx = idx[np.argsort(-flat[idx])]
        for x in idx:
            li_, e, n = np.unravel_index(x, vals.shape)
            L = Ls[li_]
            ss = plan.steps_of(L)
            nrows.append({"kind": "neuron", "layer": int(L), "expert": int(e), "neuron": int(n), "value": float(flat[x]),
                          "by_pass": {str(plan.step(s)["pass"] or 0): float(dn[s, e, n]) for s in ss},
                          "sel_rate": {str(plan.step(s)["pass"] or 0): float((sel["A"][s, e] + sel["B"][s, e]) / (nA + nB)) for s in ss}})
    # 集中の度合い: 区分ごとの上位のニューロンが、いくつの専門家に入っているか
    conc = {}
    for reg in ("loop", "after", "before"):
        rr = [x for x in nrows if region_of(plan, x["layer"]) == reg and x["value"] > 0]
        for top_n in (100, 1000):
            sub = rr[:top_n]
            ex = {}
            for x in sub:
                ex[(x["layer"], x["expert"])] = ex.get((x["layer"], x["expert"]), 0.0) + x["value"]
            v = sorted(ex.values(), reverse=True)
            conc[f"{reg}_top{top_n}"] = {"n": len(sub), "experts": len(ex), "layers": len({x["layer"] for x in sub}),
                                         "share_top5_experts": float(sum(v[:5]) / max(sum(v), 1e-12)) if v else None}
    pos_total = {reg: float(np.clip(np.stack([dn[refs[L][0]] for L in range(plan.n_stored) if region_of(plan, L) == reg]), 0, None).sum())
                 for reg in ("before", "loop", "after")}
    return {"rows": rows, "neurons": nrows, "concentration": conc, "neuron_pos_total": pos_total,
            "neuron_sum_vs_ffn_out_rel": {"median": float(np.median(errs)), "max": float(np.max(errs))},
            "seconds": time.perf_counter() - t0}


def candidates_all(plan, ra: dict, kinds: tuple[str, ...], *, regions=("loop", "after"),
                   skip_stream_start: bool = STREAM_START_DEFAULT) -> list[dict]:
    """種類 kinds(head / attn / ffn / neuron)の候補を、値の大きい順に 1 つの並びにする。流れが始まる層では attention の部品
    (head・attn)だけを外す(MLP の部品は残す)。"""
    rows = [r for r in ra["rows"] + ra["neurons"] if r["kind"] in kinds and r["value"] > 0 and region_of(plan, r["layer"]) in regions]
    if skip_stream_start:
        rows = [r for r in rows if not (r["kind"] in ("head", "attn") and r["layer"] % plan.block == 0)]
    return sorted(rows, key=lambda r: -r["value"])


def unit_of(r: dict, s: float) -> list:
    """順位の行 → LoRA の部品 [層, 種類, 番号, 強さ](番号: ヘッド h、ニューロン [e, n]、全体は None)。"""
    idx = r.get("head") if r["kind"] == "head" else ([r["expert"], r["neuron"]] if r["kind"] == "neuron" else None)
    return [int(r["layer"]), r["kind"], idx, float(s)]


def lora_tensors_all(reader: Reader, units: list, refs: dict, *, correct: bool = True) -> dict:
    """部品 [層, 種類, 番号, 強さ] → LoRA のテンソル。attention(head・attn)は attn_output(ランク 1)。MLP(ffn・neuron)は
    ffn_down_exps の 3 次元(専門家ごとにランク 1): ΔW_e = −u (Σ s s_e uᵀ W_e M)。llama.cpp は専門家ごとの倍率 s_e を元の積に
    だけ掛け、LoRA の差分には掛けないので、s_e を差分に入れておく。u は MLP の短い畳み込みの今のトークンの倍率で直した向き。"""
    w2 = Weights2(reader.w)
    li = reader.w.arch.layers[0]
    hd, nE, nf = li.head_dim, li.n_expert, li.n_ff
    att = [(L, (None if k == "attn" else i), s) for L, k, i, s in units if k in ("head", "attn")]
    out = lora_tensors(reader, att, refs, correct=correct) if att else {}
    ffn: dict[int, list] = {}
    for L, k, i, s in units:
        if k in ("ffn", "neuron"):
            ffn.setdefault(int(L), []).append((k, i, float(s)))
    for L, us in ffn.items():
        r = refs[L][1].astype(np.float64)
        cw = w2.full(f"blk.{L}.ffn_post_conv.weight").astype(np.float64)
        g = (1.0 + cw[:, -1]) / math.sqrt(cw.shape[1] + 1)
        u = _unit(g * r) if correct else _unit(r)
        sc = w2.scale(f"blk.{L}.ffn_down_exps.scale")
        sc = np.ones(nE) if sc is None else sc.astype(np.float64)
        Aall = np.zeros((nE, 1, nf))
        experts = range(nE) if any(k == "ffn" for k, _, _ in us) else sorted({i[0] for k, i, _ in us})
        for e in experts:
            row = u @ w2.slab(f"blk.{L}.ffn_down_exps.weight", e).astype(np.float64)          # (n_ff,)
            for k, i, s in us:
                if k == "ffn":
                    Aall[e, 0] += s * sc[e] * row
                elif i[0] == e:
                    Aall[e, 0, i[1]] += s * sc[e] * row[i[1]]
        B = np.repeat((-u).reshape(1, -1, 1), nE, axis=0)
        out[f"blk.{L}.ffn_down_exps.weight"] = (Aall.astype(np.float32), B.astype(np.float32))
    return out


def write_lora_all(reader: Reader, units: list, refs: dict, path: Path, *, correct: bool = True) -> dict:
    from .ablate import write_lora_tensors
    return write_lora_tensors(reader.w, lora_tensors_all(reader, units, refs, correct=correct), path, name=path.stem,
                              kv={"gne.inkling_chips_trial": json.dumps({"units": units, "correct": correct})})


def check_ffn_lora(reader: Reader, refs: dict, units: list, path: Path) -> dict:
    """MLP の LoRA(ffn_down_exps)を当てたとき、その層の最初の段の MLP の出力(畳み込みの前 ffn_moe_out・後 ffn_out)の変化が、
    当てる前に捕まえた選び方・重み・活性からの計算どおりか。r の成分(畳み込みの後)の変化も比べる。"""
    plan, e, w2 = reader.plan, reader.e, Weights2(reader.w)
    li = reader.w.arch.layers[0]
    k_used = li.n_expert_used
    L = int(units[0][0])
    tens = lora_tensors_all(reader, units, refs)
    write_lora_all(reader, units, refs, path)
    Aall, B = tens[f"blk.{L}.ffn_down_exps.weight"]
    step = plan.steps_of(L)[0]
    ids = reader.ids("Hello there, how are you today? Tell me something about the sea.")
    reader.fit([ids])
    got = []
    for lora in (None, str(path)):
        reader.cap.set_wanted(want({"ffn_moe_out", "ffn_out", "ffn_moe_situ", "ffn_moe_argsort", "ffn_moe_weights_scaled"}, [step]))
        ctx = e.probe_ctx(reader.cap, need=max(reader.need, len(ids)))
        e.set_lora(ctx, [(lora, 1.0)] if lora else None)
        try:
            reader.cap.reset()
            e.clear_memory(ctx)
            e.prefill(ctx, [ids])
            got.append(reader.cap.take())
        finally:
            e.set_lora(ctx, None)
    e.drop_lora(path)
    T = len(ids)
    top = got[0][f"ffn_moe_argsort-{step}"].reshape(T, -1)[:, :k_used]
    wts = got[0][f"ffn_moe_weights_scaled-{step}"].reshape(T, k_used).astype(np.float64)
    act = got[0][f"ffn_moe_situ-{step}"].reshape(T, k_used, -1).astype(np.float64)
    a = np.einsum("tjn,tjn->tj", Aall[top, 0].astype(np.float64), act)                  # (T, k): A_e · 活性
    pred_pre = np.einsum("tj,d->td", wts * a, B[0, :, 0].astype(np.float64))
    from .inkling_read import _conv_post
    cw = w2.full(f"blk.{L}.ffn_post_conv.weight").astype(np.float64)
    pred_post = _conv_post(pred_pre, cw)
    d_pre = got[1][f"ffn_moe_out-{step}"].reshape(T, -1).astype(np.float64) - got[0][f"ffn_moe_out-{step}"].reshape(T, -1)
    d_post = got[1][f"ffn_out-{step}"].reshape(T, -1).astype(np.float64) - got[0][f"ffn_out-{step}"].reshape(T, -1)
    r = refs[L][1].astype(np.float64)
    rel = lambda x, y: float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-30))  # noqa: E731
    mine = {int(i[0]) for _, k, i, _ in units if k == "neuron"}
    n_sel = int(T) if any(k == "ffn" for _, k, _, _ in units) else int(sum(1 for t in range(T) if mine & set(top[t].tolist())))
    return {"layer": L, "type": reader.w._tensors[f"blk.{L}.ffn_down_exps.weight"].tensor_type.name, "n_tokens": int(T),
            "n_tokens_selected": n_sel,
            "has_scale": f"blk.{L}.ffn_down_exps.scale" in reader.w._tensors,
            "rel_pre": rel(d_pre, pred_pre), "rel_post": rel(d_post, pred_post),
            "r_change_measured": float((d_post @ r).sum()), "r_change_formula": float((pred_post @ r).sum()),
            "lora_bytes": int(Aall.nbytes + B.nbytes)}


def trial_all(session, spec_path, out_dir: Path, *, ks=(8, 16, 32), strengths=(1.0, 2.0, 4.0, 8.0),
              families=(("hw", ("head", "attn", "ffn")), ("all", ("head", "attn", "ffn", "neuron"))),
              gate: str = "after", skip_stream_start: bool = STREAM_START_DEFAULT, n_build: int | None = None,
              n_val_a: int = 30, n_val_b: int = 15, n_kl: int = 30, max_tokens: int = 1024, check_layers=(9, 14, 22),
              progress=None) -> dict:
    """種類を広げた組(hw = ヘッド・attention 全体・MLP 全体、all = それにニューロン)を、trial と同じ out_dir・同じ方向・同じ無介入の
    結果の上に足す。ヘッドだけの組は trial の heads_k*。"""
    import resource
    sp = load_spec(spec_path)
    reader = Reader(session, progress=progress)
    reader.require_patch()
    reader.e.allow_lora_trial = True
    plan = reader.plan
    rp = out_dir / "result.json"
    res = json.loads(rp.read_text(encoding="utf-8"))

    def save():
        rp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    z = np.load(out_dir / "directions.npz")
    D = {k: z[k] for k in ("ffn_mix", "ffn_in", "out")}
    refs, sdirs, odir = layer_refs(plan, D, gate), step_dirs(plan, D, gate), _unit(D["out"])
    bA, bB = sp["build_a"][:n_build], sp["build_b"][:n_build]
    vA, vB = sp["val_a"][:max(n_val_a, n_kl)], sp["val_b"][:max(n_val_b, n_kl)]
    conts = res["conts"]
    if "rank_all" not in res:
        reader.e.free_steer()
        qz = out_dir / "expert_q.npz"                    # 前に作ってあれば使う(同じ方向から作ったもの)
        if qz.exists():
            z = np.load(qz)
            eq = {"q": z["q"], "scale": z["scale"], "seconds": float(z["seconds"]), "max_rss_mb": float(z["max_rss_mb"])}
        else:
            eq = expert_q(reader, refs, progress=progress)
            np.savez(qz, **eq)
        ra = rank_all(reader, bA, bB, refs, eq)
        ra["expert_q"] = {"seconds": eq["seconds"], "max_rss_mb_after_q": eq["max_rss_mb"], "q_mb": eq["q"].nbytes / 2**20}
        ra["max_rss_mb_after_rank"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        res["rank_all"] = ra
        save()
    ra = res["rank_all"]
    if "ffn_lora_checks" not in res:
        checks = []
        for L in check_layers:
            ns = [r for r in ra["neurons"] if r["layer"] == L][:3]
            for units in ([unit_of(r, 1.0) for r in ns] if ns else None, [[L, "ffn", None, 1.0]]):
                if units:
                    checks.append({"units": units, **check_ffn_lora(reader, refs, units, out_dir / f"check_L{L}.lora.gguf")})
        res["ffn_lora_checks"] = checks
        save()
    bad = [ck for ck in res["ffn_lora_checks"] if not (ck["rel_post"] < 1e-2 and ck["rel_pre"] < 1e-2)]
    if bad:                                              # MoE の LoRA が式どおりに効いていない: 組を流さずに止める
        from .steer import GneError
        raise GneError("INTERNAL", "専門家の LoRA(ffn_down_exps)が式どおりに効いていません: " + "; ".join(
            f"層 {ck['layer']} 畳み込みの前 {ck['rel_pre']:.2e}・後 {ck['rel_post']:.2e}" for ck in bad))

    def internal(lora):
        reader.e.free_steer()
        out = {sd: measure(reader, (vA if sd == "A" else vB)[:n_kl], conts[sd], sdirs, odir, gate, lora) for sd in "AB"}
        _free_probe(reader.e)
        return out
    base = internal(None)

    def run_set(key, units, *, correct=True):
        if key in res["sets"] and (res["sets"][key].get("A") or res["sets"][key].get("same_as")):
            return
        same = next((k for k, v in res["sets"].items() if v.get("units") == units and v.get("correct", True) == correct
                     and v.get("A") and not v.get("same_as")), None)
        if same is not None:                             # 同じ部品の組(ニューロンが上位に入らないとき all と hw は同じ): 流し直さない
            res["sets"][key] = {"same_as": same, "units": units, "correct": correct}
            save()
            return
        path = out_dir / f"{key}.lora.gguf"
        info = write_lora_all(reader, units, refs, path, correct=correct)
        ed = internal(str(path))
        row = {"units": units, "correct": correct, "lora_bytes": info["bytes"], **internal_row(plan, base, ed)}
        row["A"] = replies(session, vA[:n_val_a], str(path), sp["match"], max_tokens)
        row["B"] = replies(session, vB[:n_val_b], str(path), sp["match"], max_tokens)
        reader.e.drop_lora(path)
        res["sets"][key] = row
        save()
    for fam, kinds in families:
        cand = candidates_all(plan, ra, kinds, skip_stream_start=skip_stream_start)
        for k in ks:
            for s in strengths:
                run_set(f"{fam}_k{k}_s{s:g}", [unit_of(r, s) for r in cand[:k]])
    best, worst = pick(res)
    res["best"], res["worst"] = best, worst
    if best and not best.startswith(("heads_", "attn_", "before_", "after_")) and f"{best}_naive" not in res["sets"]:
        run_set(f"{best}_naive", res["sets"][best]["units"], correct=False)
    if best:
        path = out_dir / f"{best}.lora.gguf"
        units = res["sets"][best]["units"]
        if best.startswith(("heads_", "attn_", "before_", "after_")):
            write_lora(reader, [tuple(u) for u in units], refs, path)
        else:
            write_lora_all(reader, units, refs, path)
        res.setdefault("loop_effect", {})["base"] = loop_effect(reader, vA[:n_kl], vB[:n_kl], refs, sdirs, None)
        res["loop_effect"]["best"] = loop_effect(reader, vA[:n_kl], vB[:n_kl], refs, sdirs, str(path))
        res["loop_effect"]["best_key"] = best
        reader.e.drop_lora(path)
    save()
    return res


# --- 焼き込み(GGUF に書き込む) -----------------------------------------------------------------------------------------

def bake(reader: Reader, units: list, refs: dict | None, dst: Path | None, *, requant: str = "q8_0", correct: bool = True,
         strip_meta: bool = False, tens: dict | None = None, extra_meta: dict | None = None, prepared: dict | None = None) -> dict:
    """部品 units を GGUF に焼き込む(LoRA と同じ変化量)。requant(書き換えるテンソルの型):
    - q8_0 / f16: 書き換えるテンソルをその型にする(gne のほかのモデルの焼き込みと同じ。専門家のテンソルは 1 つの型しか持てない
      ので、編集しない専門家も変換し、テンソル全体が大きくなる。F16 は元の Q2_0・NVFP4 の値をそのまま表せる)
    - keep: 元の型・元の大きさのまま。attention の出力(Q8_0・K-quant)は llama.cpp の量子化で書き戻す。専門家のテンソル
      (NVFP4・Q2_0)は量子化し直さず、元のコードと倍率を残して、編集する列のコードだけを動かす(inkling_bake.py の射影を
      意識した丸め。編集しない列・専門家はバイトのまま)
    - auto: keep と同じに書いて測り、次のテンソルだけ Q8_0 にする(auto_reason): 編集の残り方(edit_kept、いくつかの方向なら
      方向ごとの最小)が KEYED_KEEP_MIN(0.9)より小さいテンソル(gne の keyed の焼き込みの auto と同じ決め方)と、MLP 全体の
      部品(L:ffn)を受ける専門家のテンソルで、方向に沿わない変化 ÷ 消した成分(off_over_removed)が inkling_bake.OFF_MAX を
      超えるもの。元が Q8_0 の attention の出力は Q8_0 のまま
    専門家ごとの倍率(ffn_down_exps.scale)はそのままにし、重みの変化量は倍率で割って入れる(実行時に倍率が掛かって LoRA と同じ
    s_e·W_e − s·u uᵀ s_e·W_e になる)。層 6〜17 の重みは 3 回で共有なので、焼き込みも 3 回すべてに効く。
    書き換えたテンソルとそのやり方は GGUF の gne.patches と <dst>.chips.json に残す(strip_meta なら GGUF には入れない)。
    元の型のまま書く(keep・auto)ときの型ごとのやり方(same_type_method): NVFP4・Q2_0・MXFP4・IQ4_NL は射影を意識した丸め
    (いくつかの方向は列ごとにまとめて解く)、llama.cpp が imatrix なしで量子化できる型(Q8_0・K-quant・IQ4_XS など)は量子化し直し(専門家のテンソルは編集した専門家のスラブだけ。編集の残り方が
    0.9 より小さければ auto は Q8_0)、imatrix の要る型(IQ2_XXS・IQ2_XS・IQ1_S)は keep なら止め、auto なら Q8_0。"""
    from gguf import GGMLQuantizationType as _Q

    from .gain import KEYED_KEEP_MIN, _SlabStream, edit_kept, expert_slab_edits
    from .gguf_fast import rewrite_gguf
    from .inkling_bake import OFF_MAX
    from .quant import dequantize, quantize
    from .steer import GneError
    if requant not in ("keep", "q8_0", "f16", "auto"):
        raise GneError("BAD_ARGS", "--requant は auto・keep・q8_0・f16 です")
    w2 = Weights2(reader.w)
    T = reader.w._tensors
    src = Path(reader.w.path)
    if tens is None:                                         # tens を渡すと(いくつかの方向の部品をまとめた LoRA の行列)それを書く
        tens = lora_tensors_all(reader, units, refs, correct=correct)

    def enc(W, qt):
        if qt == _Q.F16:
            return np.ascontiguousarray(W, dtype=np.float32).astype(np.float16).view(np.uint8).reshape(W.shape[0], -1)
        return np.frombuffer(quantize(W.reshape(-1, W.shape[-1]), qt), dtype=np.uint8).reshape(W.shape[0], -1)

    def dec(data, qt, shape):
        if qt == _Q.F16:
            return np.ascontiguousarray(data).view(np.float16).astype(np.float32).reshape(shape)
        return dequantize(np.asarray(data, dtype=np.uint8).reshape(int(np.prod(shape[:-1])), -1), qt).reshape(shape)

    def convert_experts(t, A, B, qt, rec):
        stats: dict = {}
        edited, err = expert_slab_edits(t, A, B, qt, encode=enc, decode=dec, stats=stats)
        probe = next(iter(edited.values()))
        st = _SlabStream(t, qt, {e: v.reshape(probe.shape) for e, v in edited.items()}, probe.shape, np.uint8)
        st.tofile = _convert_tofile(st, enc)
        rec.update(to=qt.name, method="requant" if qt == t.tensor_type else "convert", experts=sorted(edited), edit_kept=stats.get("kept"),
                   requant_error=err,
                   bytes_to=int(st.nbytes))
        return (qt, st)
    replace: dict[str, tuple] = {}
    recs = []
    whole = whole_ffn_layers(units)
    if prepared is not None:                                 # 書かずに計算した結果(dst=None の戻り値)をそのまま書く
        replace, recs = prepared["_replace"], prepared["modified"]
    for name, (A, B) in sorted(tens.items() if prepared is None else ()):
        t = T[name]
        rec = {"tensor": name, "from": t.tensor_type.name, "bytes_from": int(t.n_bytes), "op": "rewrite"}
        if A.ndim == 3:                                      # ffn_down_exps: 専門家ごと(スラブごと)
            L = int(name.split(".")[1])
            sc = w2.scale(f"blk.{L}.ffn_down_exps.scale")
            if sc is not None:
                A = A / sc.reshape(-1, 1, 1).astype(np.float32)
            rec["scale_divided"] = sc is not None
            if L in whole:
                rec["whole_layer"] = True
            if requant in ("q8_0", "f16"):
                replace[name] = convert_experts(t, A, B, {"q8_0": _Q.Q8_0, "f16": _Q.F16}[requant], rec)
            elif same_type_method(t.tensor_type) != "par":       # 射影を意識した丸めのない型(--into で別の量子化に書くとき)
                replace[name] = _same_type_experts(t, A, B, requant, rec, enc, dec, convert_experts)
            else:
                n_exp, rows, cols = int(t.data.shape[0]), int(B.shape[1]), int(A.shape[2])
                slab = int(t.n_bytes) // n_exp
                edited = {}
                rec.update(_par_experts(t, A, B, n_exp, rows, cols, slab, edited))
                why = auto_reason(rec, L in whole) if requant == "auto" else None
                if why:                                          # 元の型では編集が残らない・方向に沿わない変化が大きい: Q8_0
                    fb = {"from": t.tensor_type.name, "why": why, "edit_kept": rec["edit_kept"],
                          "off_over_removed": rec["off_over_removed"], "off_max": OFF_MAX}
                    rec = {k: v for k, v in rec.items() if k in ("tensor", "from", "bytes_from", "op", "scale_divided", "whole_layer")}
                    replace[name] = convert_experts(t, A, B, _Q.Q8_0, rec)
                    rec["fallback"] = fb
                else:
                    replace[name] = (t.tensor_type, _SlabStream(t, t.tensor_type, edited, (slab,), np.uint8))
        else:                                                # attn_output (out, in)
            shape = (int(B.shape[0]), int(A.shape[-1]))
            W = w2.full(name).reshape(shape).astype(np.float32)
            dW = (B.astype(np.float64) @ A.astype(np.float64)).astype(np.float32)
            W2 = W + dW

            def put(qt):
                raw = enc(W2, qt)
                back = dec(raw, qt, shape)
                return raw, edit_kept(back, W2, dW), float(np.linalg.norm(back - W2) / max(np.linalg.norm(dW), 1e-12))
            if requant in ("q8_0", "f16"):
                qt = {"q8_0": _Q.Q8_0, "f16": _Q.F16}[requant]
            else:
                qt = t.tensor_type
            if qt == t.tensor_type and same_type_method(qt) == "imatrix":   # imatrix の要る型: llama.cpp では量子化し直せない
                if requant == "keep":
                    raise GneError("UNSUPPORTED_QUANT", f"{name}: {qt.name} は imatrix が要るので、元の型のまま書き戻せません",
                                   "--requant auto(Q8_0 にする)か q8_0 / f16 にしてください")
                rec["fallback"] = {"from": qt.name, "why": "imatrix"}
                qt = _Q.Q8_0
            raw, kept, err = put(qt)
            if requant == "auto" and qt != _Q.Q8_0 and kept < KEYED_KEEP_MIN:
                rec["fallback"] = {"from": qt.name, "edit_kept": kept}
                qt = _Q.Q8_0
                raw, kept, err = put(qt)
            if qt == t.tensor_type and raw.nbytes != t.n_bytes:
                raise GneError("INTERNAL", f"{name}: 書き戻す大きさ {raw.nbytes} が元の {t.n_bytes} と違います")
            rec.update(to=qt.name, method="requant" if qt == t.tensor_type else "convert", edit_kept=kept, requant_error=err,
                       bytes_to=int(raw.nbytes))
            replace[name] = (qt, raw.tobytes())
        rec["type"] = rec["from"]
        recs.append(rec)
    if dst is None:                                          # 書かずに、残り方と大きさだけ(_replace は prepared で書くとき用)
        return {"path": None, "modified": recs, "_replace": replace}
    dst.parent.mkdir(parents=True, exist_ok=True)
    kv = None if strip_meta else {"gne.patches": json.dumps(recs, ensure_ascii=False),
                                  "gne.inkling_proj": json.dumps({"units": units, "requant": requant, "correct": correct,
                                                                  **(extra_meta or {})}, ensure_ascii=False)}
    rewrite_gguf(src, dst, replace, extra_kv=kv)
    side = Path(str(dst) + ".chips.json")
    side.write_text(json.dumps({"units": units, "requant": requant, "correct": correct, **(extra_meta or {}), "modified": recs},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    return {"path": str(dst), "bytes": dst.stat().st_size, "bytes_src": src.stat().st_size, "modified": recs, "sidecar": str(side)}


_SAME_TYPE: dict = {}


def auto_reason(rec: dict, whole: bool) -> str | None:
    """焼き込みの auto で、元の型・元の大きさで書いた専門家のテンソル(記録 rec)を Q8_0 にする理由(None なら元の型のまま):
    edit_kept(どれかの方向の編集の残り方が 0.9 より小さい)・off_direction(MLP 全体の部品を受けるテンソルで、方向に沿わない
    変化 ÷ 消した成分が OFF_MAX を超える)。ニューロンの部品だけのテンソルは変化が小さいので、残り方だけで決める。"""
    from .gain import KEYED_KEEP_MIN
    from .inkling_bake import OFF_MAX
    if rec.get("edit_kept") is not None and rec["edit_kept"] < KEYED_KEEP_MIN:
        return "edit_kept"
    if whole and (rec.get("off_over_removed") or 0.0) > OFF_MAX:
        return "off_direction"
    return None


def _par_experts(t, A, B, n_exp: int, rows: int, cols: int, slab: int, edited: dict) -> dict:
    """専門家のテンソル(NVFP4・Q2_0・MXFP4・IQ4_NL)を元の型・元の大きさのまま書く(射影を意識した丸め)。専門家ごとに、編集する
    列の LoRA の変化(いくつかの方向の部品の和)を、方向が張る空間の中でまとめて合わせる(inkling_bake.edit_columns)。専門家は
    スレッドで並べて計算する(順番によらず同じ結果)。edited に専門家ごとの新しいバイト列を入れ、記録の値を返す: 方向ごとの
    編集の残り方(edit_kept は最小)と目標との差、方向に沿わない変化 ÷ 消した成分(全体と列ごとの中央値)、動かしたコードの数。"""
    from concurrent.futures import ThreadPoolExecutor

    from .inkling_bake import WORKERS, edit_columns
    typ = t.tensor_type.name
    k = A.shape[1]
    todo = [e for e in range(n_exp) if np.any(A[e]) and np.any(B[e])]

    def one(e):
        jk = [j for j in range(k) if np.any(A[e, j]) and np.any(B[e, :, j])]
        Ae, Be = A[e].astype(np.float64), B[e].astype(np.float64)
        cs = np.nonzero(np.any(Ae[jk] != 0, axis=0))[0]
        r = edit_columns(np.ascontiguousarray(t.data[e]).reshape(rows, -1), typ, -Be[:, jk], Be[:, jk] @ Ae[jk][:, cs], cs, rows, cols)
        return e, jk, r
    tt, tg, rr = np.zeros(k), np.zeros(k), np.zeros(k)
    off2 = t2 = w2 = 0.0
    flips, offs, wn = 0, [], []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for e, jk, r in ex.map(one, todo):
            raw_e = np.asarray(r["raw"], dtype=np.uint8)
            if raw_e.nbytes != slab:
                raise GneError("INTERNAL", f"{t.name}: 専門家 {e} の大きさ {raw_e.nbytes} が元の {slab} と違います")
            edited[e] = raw_e.reshape(-1)
            w, g = r["want"], r["got"]
            tt[jk] += np.sum(w * w, 1)
            tg[jk] += np.sum(w * g, 1)
            rr[jk] += np.sum((g - w) ** 2, 1)
            off2 += r["off2"]
            t2 += r["T2"]
            w2 += float(np.sum(r["w_norm"] ** 2))
            flips += int(np.sum(r["codes_col"]))
            offs += list(r["off_col"])
            wn += list(r["off_col"] * r["t_col"] / np.maximum(r["w_norm"], 1e-30))
    has = tt > 0
    kept_by = np.where(has, tg / np.maximum(tt, 1e-300), 1.0)
    return {"to": typ, "method": "projection-aware rounding", "experts": sorted(edited), "edit_kept": float(kept_by[has].min()) if has.any() else 1.0,
            "edit_kept_by_direction": [round(float(x), 4) for x in kept_by[has]],
            "residual_by_direction": [round(float(x), 4) for x in np.sqrt(rr[has] / np.maximum(tt[has], 1e-300))],
            "codes_changed": flips, "columns": len(offs), "off_over_removed": float(np.sqrt(off2 / max(t2, 1e-300))),
            "off_over_weights": float(np.sqrt(off2 / max(w2, 1e-300))), "removed_over_weights": float(np.sqrt(t2 / max(w2, 1e-300))),
            "off_over_removed_median": float(np.median(offs)) if offs else None,
            "off_over_column_median": float(np.median(wn)) if wn else None, "bytes_to": int(t.n_bytes)}


def same_type_method(qt) -> str:
    """元の型のまま書くときのやり方: par(NVFP4・Q2_0、射影を意識した丸め)/ requant(llama.cpp が imatrix なしで量子化できる型)/
    imatrix(imatrix が要る型。元の型には書き戻せない)。"""
    from .inkling_bake import TYPES
    name = getattr(qt, "name", str(qt))
    if name in TYPES:
        return "par"
    if name not in _SAME_TYPE:
        from .quant import block_sizes, quantize
        try:
            bs, _ = block_sizes(qt)
            quantize(np.zeros((1, max(256, int(bs))), np.float32), qt)
            _SAME_TYPE[name] = "requant"
        except NotImplementedError:
            _SAME_TYPE[name] = "imatrix"
    return _SAME_TYPE[name]


def _same_type_experts(t, A, B, requant: str, rec: dict, enc, dec, convert_experts):
    """NVFP4・Q2_0 のほかの型の専門家のテンソルを元の型で書く: 編集した専門家のスラブだけを llama.cpp で量子化し直す(ほかの専門家は
    バイトのまま)。imatrix の要る型と、編集の残り方が 0.9 より小さいとき(auto)は Q8_0。編集しない列の値の変わり方も記録する。"""
    from gguf import GGMLQuantizationType as _Q

    from .gain import KEYED_KEEP_MIN, _SlabStream, expert_slab_edits
    from .steer import GneError
    from .weights import _slab_to_f32
    m = same_type_method(t.tensor_type)
    if m == "imatrix":
        if requant == "keep":
            raise GneError("UNSUPPORTED_QUANT", f"{t.name}: {t.tensor_type.name} は imatrix が要るので、元の型のまま書き戻せません",
                           "--requant auto(Q8_0 にする)か q8_0 / f16 にしてください")
        st = convert_experts(t, A, B, _Q.Q8_0, rec)
        rec["fallback"] = {"from": t.tensor_type.name, "why": "imatrix"}
        return st
    n_exp = int(t.data.shape[0])
    slab = int(t.n_bytes) // n_exp
    stats: dict = {}
    edited, err = expert_slab_edits(t, A, B, t.tensor_type, encode=enc, decode=dec, stats=stats)
    kept = stats.get("kept", 1.0)
    un2 = w2 = 0.0                                           # 編集しない列の値の変わり方(量子化し直しの丸め)
    for e, raw in edited.items():
        cols = np.nonzero(np.any(A[e] != 0, axis=0))[0]
        rest = np.setdiff1d(np.arange(A.shape[2]), cols)
        if not len(rest):
            continue
        W = _slab_to_f32(t, e)
        back = dec(raw, t.tensor_type, W.shape)
        un2 += float(np.sum((back[:, rest] - W[:, rest]) ** 2))
        w2 += float(np.sum(W[:, rest] ** 2))
    rec.update(to=t.tensor_type.name, method="requant", experts=sorted(edited), edit_kept=kept, requant_error=err,
               bytes_to=int(t.n_bytes), **({"unedited_rel_change": float(np.sqrt(un2 / w2))} if w2 > 0 else {}))
    if requant == "auto" and kept < KEYED_KEEP_MIN:
        fb = {"from": t.tensor_type.name, "edit_kept": kept}
        for k in ("method", "experts", "edit_kept", "requant_error", "unedited_rel_change"):
            rec.pop(k, None)
        st = convert_experts(t, A, B, _Q.Q8_0, rec)
        rec["fallback"] = fb
        return st
    for e, raw in edited.items():
        if raw.nbytes != slab:
            raise GneError("INTERNAL", f"{t.name}: 専門家 {e} の大きさ {raw.nbytes} が元の {slab} と違います")
    return (t.tensor_type, _SlabStream(t, t.tensor_type, {e: np.asarray(v, np.uint8).reshape(-1) for e, v in edited.items()},
                                       (slab,), np.uint8))


def whole_ffn_layers(units: list) -> set[int]:
    """MLP 全体の部品(L:ffn)がある層。焼き込みの auto では、この層の専門家のテンソルの方向に沿わない変化も確かめる(auto_reason)。"""
    return {int(u[0]) for u in units if len(u) > 1 and u[1] == "ffn"}


def _convert_tofile(st, enc):
    """型を変えて書くときの _SlabStream.tofile: 編集した専門家は作ったもの、ほかは戻して変換したもの(1 枚ずつ)。"""
    from .weights import _slab_to_f32

    def tofile(fout):
        for e in range(st.n):
            data = st.edited[e] if e in st.edited else enc(_slab_to_f32(st.t, e), st.qt)
            data.tofile(fout)
    return tofile


def units_of(row: dict) -> list:
    """試しの組の units を [層, 種類, 番号, 強さ] にそろえる(ヘッドだけの組は [層, ヘッド, 強さ]、ヘッドが None なら attention 全体)。"""
    return [list(u) if len(u) == 4 else [int(u[0]), "attn" if u[1] is None else "head", u[1], float(u[2])] for u in row["units"]]


def greedy(reader: Reader, convs: list, lora: str | None, n: int) -> dict:
    """貪欲生成のトークン列(思考の欄も含めて全部)と、かかった時間。"""
    import time
    e = reader.e
    ctx = e.steer_ctx()
    e.set_lora(ctx, [(lora, 1.0)] if lora else None)
    toks = []
    t0 = time.perf_counter()
    try:
        for c in convs:
            r = e.generate_greedy(ctx, [reader.ids(c)], max_tokens=n, stop_on_eog=True)
            toks.append([int(t) for t in r[0].token_ids])
    finally:
        e.set_lora(ctx, None)
    sec = time.perf_counter() - t0
    return {"tokens": toks, "seconds": sec, "n_tokens": int(sum(map(len, toks))),
            "texts": [e.detokenize(t) for t in toks]}


def bake_check(session, trial_dir: Path | None, units: list, spec_path, *, mode: str, n_a: int = 4, n_b: int = 2,
               reply_tokens: int = 256, gate: str = "after", ds=None, lora_path: str | None = None) -> dict:
    """焼き込みの確かめの片側。mode:
    - lora: 元のモデルに LoRA を当てて、中の値(各段の MLP の入力の方向への射影と最後に混ぜた出力の射影、最後の質問のトークン)と
      貪欲の応答を測る。比べる大きさのために、LoRA なしの中の値も測る
    - baked: 焼き込んだ GGUF(このセッションのモデル)で同じものを LoRA なしで測る
    - replies: 応答だけ(中の値を読めない llama.cpp、例えば名前の patch を当てていない本家のビルドで読み込めるかの確かめ)
    ds を渡すと(gne heads pipeline の方向、inkling_pipeline の形)、その方向の段ごとの行と最後の出力の行で測り、射影もそれで作る。"""
    sp = load_spec(spec_path)
    reader = Reader(session)
    plan = reader.plan
    if ds is not None:
        from .inkling_pipeline import OUT, layer_refs_of
        sdirs = np.stack([ds.unit(s).astype(np.float64) for s in range(plan.n_steps)])
        odir = ds.unit(OUT).astype(np.float64)
        refs_ds = layer_refs_of(plan, ds)
        D = None
    else:
        z = np.load(Path(trial_dir) / "directions.npz")
        D = {k: z[k] for k in ("ffn_mix", "ffn_in", "out")}
        sdirs, odir = step_dirs(plan, D, gate), _unit(D["out"])
    convs = sp["val_a"][:n_a] + sp["val_b"][:n_b]
    out: dict = {"mode": mode, "units": units, "n_a": n_a, "n_b": n_b, "reply_tokens": reply_tokens}
    lora = None
    if mode == "lora" and lora_path:                         # 作り済みの LoRA(レシピの LoRA。いくつかの方向の部品もまとめたもの)
        reader.e.allow_lora_trial = True
        lora = str(lora_path)
        out["lora_bytes"] = Path(lora).stat().st_size
    elif mode == "lora":
        reader.e.allow_lora_trial = True
        refs = refs_ds if D is None else layer_refs(plan, D, gate)
        lora = str((Path(trial_dir) if trial_dir else session.store.root / "inkling" / session.meta.model_id) / "bakecheck.lora.gguf")
        Path(lora).parent.mkdir(parents=True, exist_ok=True)
        out["lora_bytes"] = write_lora_all(reader, units, refs, Path(lora))["bytes"]
    if mode in ("lora", "baked"):
        reader.require_patch()
        seqs = [reader.ids(c) for c in convs]
        m = _measure_end(reader, seqs, sdirs, odir, gate, lora)
        out["proj"], out["out"] = m["proj"].tolist(), m["out"].tolist()
        if mode == "lora":
            m0 = _measure_end(reader, seqs, sdirs, odir, gate, None)
            out["proj_base"], out["out_base"] = m0["proj"].tolist(), m0["out"].tolist()
        _free_probe(reader.e)
    reader.e.free_steer()
    out["greedy"] = greedy(reader, convs, lora, reply_tokens)
    if lora and not lora_path:
        reader.e.drop_lora(Path(lora))
    return out


def bake_compare(plan, lora: dict, other: dict) -> dict:
    """lora 側と焼き込み側の比べ。中の値の差は、LoRA が動かした大きさ |LoRA − 元| と元の大きさ |元| で割る(区分ごと)。"""
    out: dict = {}
    if "proj" in other:
        pl, pb, po = np.asarray(lora["proj"]), np.asarray(lora["proj_base"]), np.asarray(other["proj"])
        diff, edit = np.abs(po - pl), np.abs(pl - pb)
        regs: dict = {}
        for st in plan.steps():
            rg = ("before" if st["layer"] < plan.core_start else "after") if not st["pass"] else f"pass{st['pass']}"
            regs.setdefault(rg, []).append(st["step"])
        out["proj_by_region"] = {rg: {"diff_over_edit_median": float(np.median(diff[:, ss].sum(1) / np.maximum(edit[:, ss].sum(1), 1e-9))),
                                      "diff_over_base_max": float((diff[:, ss] / np.maximum(np.abs(pb[:, ss]), 1e-9)).max()),
                                      "edit_over_base_median": float(np.median(edit[:, ss].sum(1) / np.maximum(np.abs(pb[:, ss]).sum(1), 1e-9)))}
                                 for rg, ss in regs.items()}
        ol, ob, oo = np.asarray(lora["out"]), np.asarray(lora["out_base"]), np.asarray(other["out"])
        out["out"] = {"lora": ol.tolist(), "base": ob.tolist(), "other": oo.tolist(),
                      "diff_over_edit": (np.abs(oo - ol) / np.maximum(np.abs(ol - ob), 1e-9)).tolist()}
    gl, go = lora["greedy"]["tokens"], other["greedy"]["tokens"]
    firsts = []
    for a, b in zip(gl, go, strict=True):
        k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        firsts.append(None if k is None and len(a) == len(b) else (k if k is not None else min(len(a), len(b))))
    out["greedy"] = {"same": sum(1 for f in firsts if f is None), "n": len(firsts), "first_diff": firsts,
                     "lens": [[len(a), len(b)] for a, b in zip(gl, go, strict=True)],
                     "tok_per_s": [lora["greedy"]["n_tokens"] / max(lora["greedy"]["seconds"], 1e-9),
                                   other["greedy"]["n_tokens"] / max(other["greedy"]["seconds"], 1e-9)]}
    return out


# --- 答えの始まりでの確かめ(思考の後) ----------------------------------------------------------------------------------

def _gen(reader: Reader, convs: list, lora: str | None, n: int) -> list[list[int]]:
    e = reader.e
    ctx = e.steer_ctx()
    e.set_lora(ctx, [(lora, 1.0)] if lora else None)
    try:
        return [[int(t) for t in e.generate_greedy(ctx, [reader.ids(c)], max_tokens=n, stop_on_eog=True)[0].token_ids] for c in convs]
    finally:
        e.set_lora(ctx, None)


def _answer_pos(toks: list[int], text_id: int) -> int | None:
    """生成したトークンの中の、回答の始まりの区切り(<|content_text|>)の位置(その出力が回答の最初のトークンを予想する)。"""
    return toks.index(text_id) if text_id in toks else None


def _end_vectors(reader: Reader, seqs: list[list[int]], gate: str) -> list[dict]:
    """トークン列の最後の位置での、各段の MLP の入力(基準の種類)と最後に混ぜた出力。"""
    plan = reader.plan
    reader.fit(seqs)
    out = []
    for ids in seqs:
        cap = reader.run(ids, want({"ffn_mix", "ffn_in", "out_mix"}))
        T = len(ids)
        out.append({"steps": np.stack([cap[f"{ref_kind(st, gate)}-{st['step']}"].reshape(T, -1)[-1] for st in plan.steps()]).astype(np.float64),
                    "mix": np.stack([cap[f"ffn_mix-{st['step']}"].reshape(T, -1)[-1] for st in plan.steps()]).astype(np.float64),
                    "out": cap["out_mix"].reshape(T, -1)[-1].astype(np.float64)})
    return out


def _region_steps(plan) -> dict[str, list[int]]:
    regs: dict[str, list[int]] = {}
    for st in plan.steps():
        rg = ("before" if st["layer"] < plan.core_start else "after") if not st["pass"] else f"pass{st['pass']}"
        regs.setdefault(rg, []).append(st["step"])
    return regs


def _measure_end(reader: Reader, seqs: list[list[int]], sdirs: np.ndarray, odir: np.ndarray, gate: str, lora: str | None) -> dict:
    """トークン列の最後の位置での射影 (n, steps) と最後に混ぜた出力の射影 (n,)(lora を当てて)。"""
    plan, e = reader.plan, reader.e
    reader.fit(seqs)
    proj, outp = np.zeros((len(seqs), plan.n_steps)), np.zeros(len(seqs))
    names = want({"ffn_mix", "ffn_in", "out_mix"})
    for i, ids in enumerate(seqs):
        reader.cap.set_wanted(names)
        ctx = e.probe_ctx(reader.cap, need=max(reader.need, len(ids)))
        e.set_lora(ctx, [(lora, 1.0)] if lora else None)
        try:
            reader.cap.reset()
            e.clear_memory(ctx)
            e.prefill(ctx, [ids])
            cap = reader.cap.take()
        finally:
            e.set_lora(ctx, None)
        T = len(ids)
        for st in plan.steps():
            proj[i, st["step"]] = cap[f"{ref_kind(st, gate)}-{st['step']}"].reshape(T, -1)[-1] @ sdirs[st["step"]]
        outp[i] = cap["out_mix"].reshape(T, -1)[-1] @ odir
    return {"proj": proj, "out": outp}


def _keep(plan, base: dict, ed: dict) -> dict:
    """A − B の差が残った割合(区分ごとの中央値と最後の出力)。base / ed = {"A": {...}, "B": {...}}(_measure_end)。"""
    gap0 = base["A"]["proj"].mean(0) - base["B"]["proj"].mean(0)
    gap1 = ed["A"]["proj"].mean(0) - ed["B"]["proj"].mean(0)
    keep = gap1 / np.where(np.abs(gap0) > 1e-9, gap0, np.nan)
    o0 = base["A"]["out"].mean() - base["B"]["out"].mean()
    o1 = ed["A"]["out"].mean() - ed["B"]["out"].mean()
    return {"keep_by_region": {rg: float(np.nanmedian(keep[ss])) for rg, ss in _region_steps(plan).items()},
            "keep_out": float(o1 / o0) if abs(o0) > 1e-9 else None}


def answer_check(session, spec_path, out_dir: Path, *, gate: str = "after", n_dir: int = 16, n_traj: int = 6, n_val: int = 15,
                 n_val_a: int = 30, n_val_b: int = 15, max_tokens: int = 1024, strong: str = "heads_k32_s8", progress=None) -> dict:
    """思考の後の、回答の始まりでの確かめ(1 回だけの決まった確かめ。格子は増やさない)。
    1. 強い組(strong)の応答 3 つを、思考の欄も含めて残す(思考が海賊の指示を言い直しているか)
    2. 2 つ目の方向: 書き換える前のモデルの A 側・B 側の応答(方向を作る質問)の、回答の始まりの区切り(<|content_text|>)の位置での
       A − B。最後の質問のトークンの方向との cos を区分ごとに。ヘッドだけ・全部の種類のいちばん良い組を、同じ部品のままこの方向で
       流し直す(中の値: 最後の質問のトークンと回答の始まり、応答)
    3. 最後の質問のトークンの方向への射影が、書き換える前の応答(思考 → 回答)の中でどう続くか(位置の区切りごとの A − B)"""
    from .inkling import ReplyMarks
    sp = load_spec(spec_path)
    reader = Reader(session, progress=progress)
    reader.require_patch()
    reader.e.allow_lora_trial = True
    plan = reader.plan
    rp = out_dir / "result.json"
    res = json.loads(rp.read_text(encoding="utf-8"))
    ac = res.setdefault("answer_check", {})

    def save():
        rp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    text_id = ReplyMarks(reader.e).text
    z = np.load(out_dir / "directions.npz")
    D1 = {k: z[k] for k in ("ffn_mix", "ffn_in", "out")}
    s1, o1 = step_dirs(plan, D1, gate), _unit(D1["out"])
    regs = _region_steps(plan)
    vA, vB = sp["val_a"], sp["val_b"]
    # 1. 強い組の応答(思考も)
    if "strong_replies" not in ac and strong in res["sets"]:
        path = out_dir / f"{strong}.lora.gguf"
        write_lora_all(reader, units_of(res["sets"][strong]), layer_refs(plan, D1, gate), path)
        r = replies(session, vA[:3], str(path), sp["match"], max_tokens)
        reader.e.drop_lora(path)
        ac["strong_replies"] = {"set": strong, **r}
        save()
    # 2. 方向を作る質問への、書き換える前の応答と、回答の始まりの位置
    zp = out_dir / "answer_directions.npz"
    if not zp.exists():
        reader.e.free_steer()
        info = {"A": [], "B": []}
        vecs = {}
        for sd, convs in (("A", sp["build_a"][:n_dir]), ("B", sp["build_b"][:n_dir])):
            gens = _gen(reader, convs, None, max_tokens)
            seqs = []
            for c, g in zip(convs, gens, strict=True):
                k = _answer_pos(g, text_id)
                info[sd].append({"n_gen": len(g), "answer_pos": k})
                if k is not None:
                    seqs.append(reader.ids(c) + g[:k + 1])
            if not seqs:
                from .steer import GneError
                raise GneError("INTERNAL", f"{sd} 側の応答に回答の始まりがありません(思考の途中で長さの上限に届いた)", "--max-tokens を増やしてください")
            vecs[sd] = _end_vectors(reader, seqs, gate)
        D2 = {"ffn_in": np.zeros_like(D1["ffn_in"]), "ffn_mix": np.zeros_like(D1["ffn_mix"])}
        for st in plan.steps():
            sidx = st["step"]
            D2[ref_kind(st, gate)][sidx] = np.mean([v["steps"][sidx] for v in vecs["A"]], 0) - np.mean([v["steps"][sidx] for v in vecs["B"]], 0)
            D2["ffn_mix"][sidx] = np.mean([v["mix"][sidx] for v in vecs["A"]], 0) - np.mean([v["mix"][sidx] for v in vecs["B"]], 0)
        D2["out"] = np.mean([v["out"] for v in vecs["A"]], 0) - np.mean([v["out"] for v in vecs["B"]], 0)
        np.savez(zp, **D2)
        ac["dir_build"] = {sd: {"n": len(info[sd]), "with_answer": sum(1 for x in info[sd] if x["answer_pos"] is not None),
                                "answer_pos_median": float(np.median([x["answer_pos"] for x in info[sd] if x["answer_pos"] is not None] or [np.nan])),
                                "n_gen_median": float(np.median([x["n_gen"] for x in info[sd]]))} for sd in "AB"}
        save()
    z2 = np.load(zp)
    D2 = {k: z2[k] for k in ("ffn_mix", "ffn_in", "out")}
    s2, o2 = step_dirs(plan, D2, gate), _unit(D2["out"])
    cos = np.array([float(s1[i] @ s2[i]) for i in range(plan.n_steps)])
    ac["cos_by_region"] = {rg: {"median": float(np.median(cos[ss])), "min": float(cos[ss].min()), "max": float(cos[ss].max())}
                           for rg, ss in regs.items()}
    ac["cos_out"] = float(o1 @ o2)
    save()
    # 書き換える前の応答(確かめの質問): 回答の始まりまでのトークン(中の値を回答の始まりで測る)と、射影の続き方
    if "val_gen" not in ac:
        reader.e.free_steer()
        ac["val_gen"] = {sd: _gen(reader, (vA if sd == "A" else vB)[:n_val], None, max_tokens) for sd in "AB"}
        save()
    ends = {sd: [reader.ids(c) + g[:_answer_pos(g, text_id) + 1]
                 for c, g in zip((vA if sd == "A" else vB)[:n_val], ac["val_gen"][sd], strict=True) if _answer_pos(g, text_id) is not None]
            for sd in "AB"}
    prompt_end = {sd: [reader.ids(c) for c in (vA if sd == "A" else vB)[:n_val]] for sd in "AB"}
    # 3. 最後の質問のトークンの方向への射影の、応答の中での続き方
    if "trajectory" not in ac:
        buckets = ["prompt_end", "think_0-25", "think_25-50", "think_50-75", "think_75-100", "answer_start", "answer_0-50", "answer_50-100"]
        acc = {sd: {rg: {b: [] for b in buckets} for rg in list(regs) + ["out"]} for sd in "AB"}
        for sd in "AB":
            convs = (vA if sd == "A" else vB)[:n_traj]
            for c, g in zip(convs, ac["val_gen"][sd][:n_traj], strict=True):
                k = _answer_pos(g, text_id)
                if k is None:
                    continue
                ids = reader.ids(c)
                full = ids + g
                reader.fit([full])
                cap = reader.run(full, want({"ffn_mix", "ffn_in", "out_mix"}))
                T, P = len(full), len(ids)
                pr = np.stack([cap[f"{ref_kind(st, gate)}-{st['step']}"].reshape(T, -1) @ s1[st["step"]] for st in plan.steps()])  # (steps, T)
                po = cap["out_mix"].reshape(T, -1) @ o1
                del cap
                # 位置の区切り: 質問の最後 = P−1、思考 = P .. P+k−1、回答の始まり = P+k、回答 = P+k+1 ..
                spans = {"prompt_end": [P - 1], "answer_start": [P + k]}
                th = list(range(P, P + k))
                for q, b in enumerate(buckets[1:5]):
                    spans[b] = th[len(th) * q // 4: len(th) * (q + 1) // 4]
                an = list(range(P + k + 1, T))
                spans["answer_0-50"], spans["answer_50-100"] = an[:len(an) // 2], an[len(an) // 2:]
                for b, pos in spans.items():
                    if not pos:
                        continue
                    for rg, ss in regs.items():
                        acc[sd][rg][b].append(float(pr[ss][:, pos].mean()))
                    acc[sd]["out"][b].append(float(po[pos].mean()))
        def m(x):
            return float(np.mean(x)) if x else None

        def gap(rg, b):
            return m(acc["A"][rg][b]) - m(acc["B"][rg][b]) if acc["A"][rg][b] and acc["B"][rg][b] else None
        ac["trajectory"] = {"buckets": buckets, "n": {sd: len(acc[sd]["out"]["prompt_end"]) for sd in "AB"},
                            "gap": {rg: {b: gap(rg, b) for b in buckets} for rg in list(regs) + ["out"]},
                            "a": {rg: {b: m(acc["A"][rg][b]) for b in buckets} for rg in list(regs) + ["out"]},
                            "b": {rg: {b: m(acc["B"][rg][b]) for b in buckets} for rg in list(regs) + ["out"]}}
        save()
    # 2 の続き: いちばん良い組を 2 つ目の方向で流し直す
    hs = {k: v for k, v in res["sets"].items() if k.startswith("heads_")}
    al = {k: v for k, v in res["sets"].items() if k.startswith(("hw_", "all_"))}
    picks = [pick({"base": res["base"], "sets": x})[0] for x in (hs, al) if x]
    ac["picked"] = picks
    bases = {}
    for key in [k for k in picks if k]:
        if key in ac.get("reruns", {}):
            continue
        units = units_of(res["sets"][key])
        row = {}
        for dname, D, sd_, od_ in (("prompt_dir", D1, s1, o1), ("answer_dir", D2, s2, o2)):
            path = out_dir / f"{key}.{dname}.lora.gguf"
            write_lora_all(reader, units, layer_refs(plan, D, gate), path)
            reader.e.free_steer()
            for pos, seqs, mdir, sdm, odm in (("at_prompt_end", prompt_end, "on_prompt_dir", s1, o1),
                                              ("at_answer_start", ends, "on_answer_dir", s2, o2)):
                bk = (pos, mdir)
                if bk not in bases:
                    bases[bk] = {sd: _measure_end(reader, seqs[sd], sdm, odm, gate, None) for sd in "AB"}
                ed = {sd: _measure_end(reader, seqs[sd], sdm, odm, gate, str(path)) for sd in "AB"}
                row.setdefault(dname, {})[f"{pos}/{mdir}"] = _keep(plan, bases[bk], ed)
            _free_probe(reader.e)
            if dname == "answer_dir":
                row[dname]["A"] = replies(session, vA[:n_val_a], str(path), sp["match"], max_tokens)
                row[dname]["B"] = replies(session, vB[:n_val_b], str(path), sp["match"], max_tokens)
            reader.e.drop_lora(path)
        ac.setdefault("reruns", {})[key] = row
        save()
    return res
