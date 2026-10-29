"""書き手の強さ `gne writers`: ある状態(対照集合の A、例えばヤンデレの system)で、各ニューロンが残差にどれだけ書き足したか。

活性の差(act_diff)は「どれだけ光り方が変わったか」、逆引きの cos は「書く向きが方向に近いか」(重みだけ)を見る。ここでは
その積、つまり**実際に書き足した量**を返事のトークンで測る。

層 L の MLP の出力は Σ_n a_n · W_down[:, n](post-FFN norm のある arch(Gemma・LFM3)では、さらに γ ⊙ (…) / rms で割る)なので、
トークン t でニューロン n が書き足したベクトルは a_n(t) · γ ⊙ W_down[:, n] / rms_t(post norm が無ければ γ = 1・rms = 1)。
A と B の返事のトークンでの平均の差を取り、参照の層 L* の方向 r̂ = r_{L*} / ‖r_{L*}‖ に射影したものを、そのニューロンの
「方向への直接の寄与」とする(後ろの層での変換は見ない、direct logit attribution と同じ一次の見方)。

残差は「埋め込み + 各層の mixer(attention・畳み込み・線形注意)の出力 + 各層の MLP の出力」の和なので、
r_{L*} = Δ埋め込み + Σ_{l ≤ L*}(Δmixer_l + ΔMLP_l) がちょうど成り立つ。ΔMLP_l は上の和で、Δmixer_l = r_l − r_{l−1} − ΔMLP_l
(mixer の出力のテンソル名は arch で違うので、差で求める)。これを r̂ に射影すると、方向の何割を MLP / mixer / 埋め込みが
書いたかが分かる。

捕捉は 2 回に分ける。1 回目は会話ごとに 1 回 prefill して全層の l_out と埋め込みを取り、方向を決める。2 回目は層ごとに、
全会話を流してその層の活性だけを取る(post norm の rms をトークンごとに求めるのに W_down の積が要るので、W_down を層ごとに
1 回だけ読む)。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import yaml

from .capture import Capture, pick_act_name
from .steer import GneError

EMBED_NAMES = ("inp_embd", "model.embed_tokens")
MLP_OUT_NAMES = ("model.layers.{}.post_ffn_norm", "ffn_post_norm", "ffn_out")   # 確かめに使う MLP の出力(post norm の後を先に)


def load_replies(path: str | Path | None, prompts: list[str]) -> list[str] | None:
    """返事のファイル(YAML / JSON): user_prompts と同じ長さのリストか、{プロンプト: 返事} の表。"""
    if path is None:
        return None
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(d, dict) and "replies" in d:
        d = d["replies"]
    if isinstance(d, dict):
        missing = [p for p in prompts if p not in d]
        if missing:
            raise GneError("BAD_ARGS", f"{path}: 返事の無いプロンプトがあります({missing[0]!r} など {len(missing)} 個)")
        return [str(d[p]) for p in prompts]
    if isinstance(d, list) and len(d) == len(prompts):
        return [str(x) for x in d]
    raise GneError("BAD_ARGS", f"{path}: 返事は user_prompts と同じ長さのリストか、{{プロンプト: 返事}} の表です")


def _post_norm(weights, L: int):
    """層 L の post-FFN norm の重み(無ければ None)と eps。"""
    from .weights import tensor_to_f32
    name = f"blk.{L}.post_ffw_norm.weight"
    if not weights.has_tensor(name):
        return None, 0.0
    g = np.asarray(tensor_to_f32(weights.tensor(name)), dtype=np.float32).reshape(-1)
    eps = float(weights.field(f"{weights.meta.arch}.attention.layer_norm_rms_epsilon", 1e-6) or 1e-6)
    return g, eps


def writer_contributions(session, contrast: str | Path, *, name: str | None = None, ref_layer: int | None = None,
                         replies_a: str | Path | None = None, replies_b: str | Path | None = None,
                         max_tokens: int = 96, top: int = 50, progress=None) -> dict:
    """対照集合 A / B の返事のトークンで、層ごと・ニューロンごとの書き足しを測って <root>/writers/<model_id>/<name>.npz と .json に保存する。
    replies_a / replies_b を渡すと、その側の返事は生成せずにその文を読ませる(teacher forcing。別のモデルの返事を借りるとき)。"""
    from .direction import load_contrast
    from .evalx import _generate
    t0 = time.perf_counter()
    cs, _ = load_contrast(contrast)
    name = name or cs.name
    e, w = session.engine, session.weights
    m = session.meta
    n_l = e.n_layer
    Ls = list(range(n_l - 1))                                  # 最終層は行が間引かれるので除く
    ref = n_l - 2 if ref_layer is None else int(ref_layer)
    if ref not in Ls:
        raise GneError("BAD_ARGS", f"--ref-layer は 0〜{n_l - 2} です")
    d = yaml.safe_load(Path(contrast).read_text(encoding="utf-8")) or {}
    given = {"A": load_replies(replies_a, cs.user_prompts) or d.get("replies_a"),
             "B": load_replies(replies_b, cs.user_prompts) or d.get("replies_b")}
    for side in "AB":
        if given[side] is not None and len(given[side]) != len(cs.user_prompts):
            raise GneError("BAD_CONTRAST", f"replies_{side.lower()} の数が user_prompts と違います")
    probe = e.tokenize_prompt(e.render_chat([{"role": "user", "content": cs.user_prompts[0]}]))
    seen = e.list_names(probe[:16])
    act_name = {L: pick_act_name(seen, L) for L in Ls}
    Ls_n = [L for L in Ls if act_name[L] and w.has_tensor(f"blk.{L}.ffn_down.weight")]
    emb_name = next((x for x in EMBED_NAMES if x in seen), None)
    mlp_name = {L: next((f"{b}-{L}" for b in MLP_OUT_NAMES if f"{b}-{L}" in seen), None) for L in Ls}
    n_conv = len(cs.user_prompts)
    steps, done = 2 + 2 * n_conv + len(Ls_n), 0

    def tick(msg):
        nonlocal done
        done += 1
        if progress:
            progress(done, steps, msg)

    # 会話(system + ユーザー + 返事)と、返事の始まり
    seqs: dict[str, list[tuple[list[int], int]]] = {"A": [], "B": []}
    texts: dict[str, list[str]] = {"A": [], "B": []}
    for side, system in (("A", cs.system_a), ("B", cs.system_b)):
        convs = [([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": u}]
                 for u in cs.user_prompts]
        if given[side] is None:
            outs = _generate(session, [(c, max_tokens) for c in convs], None, None, "all", lambda k: None)
            resp_ids = [[t for t in o["token_ids"] if not e.is_eog(t)] for o in outs]
            texts[side] = [o["text"] for o in outs]
        else:
            resp_ids = [e.tokenize(r, add_special=False, parse_special=False) for r in given[side]]
            texts[side] = list(given[side])
        for c, r in zip(convs, resp_ids, strict=True):
            pids = e.tokenize_prompt(e.render_chat(c))
            if r:
                seqs[side].append((pids + r, len(pids)))
        tick(f"writers: {side} の返事")
    if not seqs["A"] or not seqs["B"]:
        raise GneError("BAD_CONTRAST", "返事がありません")
    if max(len(s) for side in "AB" for s, _ in seqs[side]) > e.n_ctx:
        raise GneError("BAD_ARGS", "会話が n_ctx を超えています")

    # 1 回目: 全層の l_out と埋め込み → 層ごとの方向 r_L
    names1 = {f"l_out-{L}": L for L in Ls}
    cap = Capture(lambda nm: nm in names1 or nm == emb_name)
    res_sum = {side: {L: np.zeros(m.n_embd) for L in Ls} for side in "AB"}
    emb_sum = {side: np.zeros(m.n_embd) for side in "AB"}
    cnt = {"A": 0, "B": 0}
    for side in "AB":
        for ids, st in seqs[side]:
            ctx = e.probe_ctx(cap)
            e.clear_memory(ctx)
            e.set_cvec(ctx, None)
            e.prefill(ctx, [ids])
            got = cap.take()
            for nm, L in names1.items():
                res_sum[side][L] += got[nm][st:].astype(np.float64).sum(axis=0)
            if emb_name:
                emb_sum[side] += got[emb_name][st:].astype(np.float64).sum(axis=0)
            else:
                emb_sum[side] += w.embed_rows(ids[st:]).astype(np.float64).sum(axis=0)
            cnt[side] += len(ids) - st
            tick(f"writers: 残差 {side}")
    r = {L: res_sum["A"][L] / cnt["A"] - res_sum["B"][L] / cnt["B"] for L in Ls}
    d_emb = emb_sum["A"] / cnt["A"] - emb_sum["B"] / cnt["B"]
    if not emb_name:
        session.warn("埋め込みのテンソルが捕捉できないので、token_embd の行で代わりにしました(埋め込みを拡大する arch ではずれます)")
    r_ref = r[ref]
    nrm = float(np.linalg.norm(r_ref))
    if nrm == 0:
        raise GneError("BAD_CONTRAST", "A と B の残差が同じです")
    rh = r_ref / nrm

    # 2 回目: 層ごとに活性を取り、MLP の出力とニューロンごとの寄与
    per: dict[int, dict[str, np.ndarray]] = {}
    mlp: dict[int, np.ndarray] = {}
    mlp_err: dict[int, float] = {}
    for L in Ls_n:
        W = w.down_matrix(L).astype(np.float32)                 # (n_embd, n_ff)
        g, eps = _post_norm(w, L)
        s = (w.meta.ffn_down_scale or {}).get(L, 1.0)
        an, mn = act_name[L], mlp_name[L]
        cap.set_wanted(lambda nm, an=an, mn=mn: nm in (an, mn))
        got_out = {side: np.zeros(m.n_embd) for side in "AB"}  # 捕捉した MLP の出力(式の確かめ)
        sa = {side: np.zeros(W.shape[1]) for side in "AB"}      # Σ_t a_n(t)(post norm なら / rms_t)
        raw_a = {side: np.zeros(W.shape[1]) for side in "AB"}   # Σ_t a_n(t)(表示用の平均の活性)
        out = {side: np.zeros(m.n_embd) for side in "AB"}       # Σ_t MLP の出力
        for side in "AB":
            for ids, st in seqs[side]:
                ctx = e.probe_ctx(cap)
                e.clear_memory(ctx)
                e.set_cvec(ctx, None)
                e.prefill(ctx, [ids])
                gt = cap.take()
                a = gt[an][st:].astype(np.float32) * s                          # (T, n_ff)
                if mn and mn in gt:
                    got_out[side] += gt[mn][st:].astype(np.float64).sum(axis=0)
                raw_a[side] += a.sum(axis=0, dtype=np.float64)
                if g is None:
                    sa[side] += a.sum(axis=0, dtype=np.float64)
                    out[side] += (a @ W.T).sum(axis=0, dtype=np.float64)
                else:
                    y = a @ W.T                                                   # (T, n_embd) post norm の前
                    inv = 1.0 / np.sqrt(np.mean(np.square(y, dtype=np.float64), axis=1) + eps)   # (T,)
                    sa[side] += (a * inv[:, None].astype(np.float32)).sum(axis=0, dtype=np.float64)
                    out[side] += ((y * inv[:, None]) * g[None, :]).sum(axis=0, dtype=np.float64)
        ds = sa["A"] / cnt["A"] - sa["B"] / cnt["B"]                              # (n_ff,)
        gg = g if g is not None else np.ones(m.n_embd, dtype=np.float32)
        gw = (W.T @ (rh * gg)).astype(np.float64)                                 # (n_ff,) 実際の書く向き(γ ⊙ w_n)・r̂
        wn = np.sqrt(np.square(W).T @ np.square(gg)).astype(np.float64)           # ‖γ ⊙ w_n‖
        per[L] = {"contrib": (ds * gw).astype(np.float32), "act_a": (raw_a["A"] / cnt["A"]).astype(np.float32),
                  "act_b": (raw_a["B"] / cnt["B"]).astype(np.float32),
                  "act_diff": (raw_a["A"] / cnt["A"] - raw_a["B"] / cnt["B"]).astype(np.float32),
                  "align": (gw / np.maximum(wn, 1e-12)).astype(np.float32)}
        mlp[L] = out["A"] / cnt["A"] - out["B"] / cnt["B"]
        if mn:
            real = got_out["A"] / cnt["A"] - got_out["B"] / cnt["B"]
            mlp_err[L] = float(np.linalg.norm(mlp[L] - real) / max(np.linalg.norm(real), 1e-12))
        tick(f"writers: L{L}")

    # 層ごとの分け方(参照の方向 r̂ への射影。和が ‖r_ref‖ になる)
    layers, prev = [], d_emb
    for L in Ls:
        dm = mlp.get(L)
        dmix = r[L] - prev - (dm if dm is not None else 0.0)
        row = {"layer": L, "mixer": w.arch.layers[L].mixer if L < len(w.arch.layers) else "?",
               "r_norm": float(np.linalg.norm(r[L])), "cos_ref": float(r[L] @ rh / max(np.linalg.norm(r[L]), 1e-12)),
               "mlp_norm": None if dm is None else float(np.linalg.norm(dm)), "mixer_norm": float(np.linalg.norm(dmix))}
        if L <= ref:
            row["mlp_to_ref"] = None if dm is None else float(dm @ rh)
            row["mixer_to_ref"] = float(dmix @ rh)
        row["mlp_check"] = mlp_err.get(L)                                       # 式の MLP の出力と捕捉した出力の差
        layers.append(row)
        prev = r[L]
    totals = {"embed": float(d_emb @ rh) / nrm,
              "mixer": sum(x["mixer_to_ref"] for x in layers if "mixer_to_ref" in x) / nrm,
              "mlp": sum((x["mlp_to_ref"] or 0.0) for x in layers if "mlp_to_ref" in x) / nrm}
    # ニューロンの順位(参照の方向への寄与)
    allc = [(L, n, float(per[L]["contrib"][n])) for L in per if L <= ref for n in range(per[L]["contrib"].size)]
    allc.sort(key=lambda x: -x[2])

    def row_of(L, n, c):
        p = per[L]
        return {"layer": L, "neuron": n, "contrib": c, "share": c / nrm, "act_a": float(p["act_a"][n]),
                "act_b": float(p["act_b"][n]), "act_diff": float(p["act_diff"][n]), "align": float(p["align"][n])}
    pos = [row_of(*x) for x in allc[:top]]
    neg = [row_of(*x) for x in allc[::-1][:max(10, top // 3)]]
    contribs = np.array([x[2] for x in allc])
    for k in (10, 100, 1000):
        totals[f"top{k}"] = float(np.sort(contribs)[::-1][:k].sum()) / nrm if contribs.size else 0.0
    out_dir = session.store.root / "writers" / m.model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{name}.npz", ref_layer=ref, ref_norm=nrm,
                        **{f"{k}_{L}": v for L, p in per.items() for k, v in p.items()})
    summary = {"name": name, "contrast": str(Path(contrast).resolve()), "contrast_sha": cs.sha, "ref_layer": ref,
               "ref_norm": nrm, "n_tokens": cnt, "n_convs": {s_: len(seqs[s_]) for s_ in "AB"},
               "replies": {"A": "given" if given["A"] is not None else "generated",
                           "B": "given" if given["B"] is not None else "generated"},
               "post_ffn_norm": any(_post_norm(w, L)[0] is not None for L in Ls_n[:1]),
               "totals": totals, "mlp_check_max": max(mlp_err.values()) if mlp_err else None,
               "layers": layers, "top": pos, "bottom": neg,
               "samples": [{"prompt": u, "A": texts["A"][i][:240] if i < len(texts["A"]) else "",
                            "B": texts["B"][i][:240] if i < len(texts["B"]) else ""}
                           for i, u in enumerate(cs.user_prompts[:6])],
               "seconds": time.perf_counter() - t0}
    p = out_dir / f"{name}.json"
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    return {**summary, "path": str(p)}


def list_writers(session) -> list[dict]:
    d = session.store.root / "writers" / session.meta.model_id
    out = []
    for p in sorted(d.glob("*.json"), key=lambda q: -q.stat().st_mtime) if d.exists() else []:
        try:
            s = json.loads(p.read_text())
        except ValueError:
            continue
        out.append({"name": s["name"], "ref_layer": s["ref_layer"], "totals": s["totals"], "created": p.stat().st_mtime})
    return out


def load_layer(session, name: str, L: int) -> dict[str, np.ndarray] | None:
    """保存した書き足しの層 L(contrib・act_a・act_b・act_diff・align)。無ければ None。"""
    p = session.store.root / "writers" / session.meta.model_id / f"{name}.npz"
    if not p.exists():
        raise GneError("FILE_NOT_FOUND", f"書き手の分析 {name} がありません(gne writers で作れます)")
    z = np.load(p)
    if f"contrib_{L}" not in z.files:
        return None
    return {k: z[f"{k}_{L}"] for k in ("contrib", "act_a", "act_b", "act_diff", "align")}


def neuron_entries(session, L: int, n: int) -> list[dict]:
    """ニューロンの調書用: 保存済みの分析ごとの寄与と順位。"""
    d = session.store.root / "writers" / session.meta.model_id
    out = []
    for p in sorted(d.glob("*.npz"), key=lambda q: -q.stat().st_mtime) if d.exists() else []:
        z = np.load(p)
        if f"contrib_{L}" not in z.files:
            continue
        ref = int(z["ref_layer"])
        allv = np.concatenate([z[k] for k in z.files if k.startswith("contrib_") and int(k.split("_")[1]) <= ref])
        c = float(z[f"contrib_{L}"][n])
        out.append({"name": p.stem, "ref_layer": ref, "contrib": c, "share": c / float(z["ref_norm"]),
                    "rank": int((allv > c).sum()) + 1, "n_ranked": int(allv.size),
                    "act_a": float(z[f"act_a_{L}"][n]), "act_b": float(z[f"act_b_{L}"][n]),
                    "act_diff": float(z[f"act_diff_{L}"][n])})
    return out
