"""方向(`gne direction`、本体設計 §6.17)。

対照集合(reverse と同じ YAML: system_a / system_b / user_prompts)から、全層の r_L = mean_A − mean_B を作って保存する。
捕捉の位置は既定で生成プロンプトの最終トークン(positions: last、§6.17)。保存先 <root>/directions/<model_id>/<name>.npz
(reverse の *.directions.npz と同じキーなので、逆引き・マップの cos の重ね合わせにもそのまま使える)。

層の選択(--select auto)は候補層ごとに方向を当てて、検証用プロンプトでの効果を測る:
- mode=ablate(設計どおり): r_L を全層のライターから消す rank-1 LoRA(gne/ablate.py)を当て、A 側(system_a 付き)の出力で
  判定の一致率がどれだけ下がるかを効果、B 側(system_b 付き)の次トークン分布の KL(base‖ablated)を副作用とする。
- mode=add(追加): 層 L に dose · (mean_A − mean_B)_L を足し(dose は平均差の何倍か。層ごとの大きさの違いをそろえるため、
  rel dose で一律にはしない)、system なしの出力で判定の一致率がどれだけ上がるかを効果、distinct-4(反復していないか)を
  副作用の目安とする。KL も参考に出す(足せば分布が変わるのは当然なので基準にはしない)。各層の結果には、同じ大きさを
  レシピの rel dose に直した値(dose · ‖平均差‖ / ρ_L)も付ける。

判定は対照集合の validate.match(正規表現、大文字小文字は無視)か、--judge-model(別の GGUF に 1〜5 で採点させる)。
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .rank import distinct4
from .reverse import ContrastSet
from .steer import GneError

Progress = Callable[[int, int, str], None]


@dataclass
class Validate:
    prompts: list[str]
    match: str | None
    target: str | None


def load_contrast(path: str | Path) -> tuple[ContrastSet, Validate]:
    cs = ContrastSet.load(path)
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    v = d.get("validate") or {}
    if v.get("match"):
        try:
            re.compile(v["match"])
        except re.error as e:
            raise GneError("BAD_CONTRAST", f"validate.match の正規表現が読めません: {e}") from e
    return cs, Validate(list(v.get("prompts") or cs.user_prompts), v.get("match"), v.get("target"))


@dataclass
class DirectionSet:
    name: str
    path: Path
    raw: dict[int, np.ndarray]           # 層 → mean_A − mean_B
    scale: dict[int, np.ndarray]
    meta: dict

    @classmethod
    def load(cls, path: str | Path) -> DirectionSet:
        p = Path(path)
        z = np.load(p)
        raw = {int(L): z[f"layer_{L}"].astype(np.float32) for L in z["layers"]}
        scale = {int(k[6:]): z[k] for k in z.files if k.startswith("scale_")}
        meta = {k: (z[k].item() if z[k].ndim == 0 else z[k].tolist()) for k in
                ("method", "positions", "contrast_sha", "source", "n_rows_A", "n_rows_B") if k in z.files}
        return cls(p.name.split(".")[0], p, raw, scale, meta)

    def unit(self, L: int) -> np.ndarray:
        if L not in self.raw:
            raise GneError("BAD_DIRECTION", f"方向 {self.name} に層 {L} がありません(0〜{max(self.raw)})")
        d = self.raw[L]
        n = float(np.linalg.norm(d))
        return d / n if n > 0 else d


def save_directions(path: Path, cr, cs: ContrastSet, source: str) -> None:
    Ls = sorted(cr.directions)
    U = np.stack([cr.directions[L] / max(float(np.linalg.norm(cr.directions[L])), 1e-30) for L in Ls])
    np.savez(path, layers=np.array(Ls, dtype=np.int32), **{f"layer_{L}": cr.directions[L] for L in Ls},
             **{f"scale_{L}": cr.scale[L] for L in Ls}, cos=(U @ U.T).astype(np.float32),
             method=cr.method, positions=cr.positions, contrast_sha=cs.sha, source=source,
             n_rows_A=cr.n_rows["A"], n_rows_B=cr.n_rows["B"])


def candidate_layers(n_layer: int, spec: str | list[int] | None = None) -> list[int]:
    """既定 0.3·n_layer 〜 0.8·n_layer。"a-b"、"a-b:step"、"a,b,c" も受け付ける。層 0 と最終層は除く。"""
    if spec is None or spec == "":
        Ls = list(range(round(0.3 * n_layer), round(0.8 * n_layer) + 1))
    elif isinstance(spec, list):
        Ls = [int(x) for x in spec]
    else:
        Ls = []
        for part in str(spec).split(","):
            m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*(?::\s*(\d+))?)?\s*", part)
            if not m:
                raise GneError("BAD_ARGS", f"層の指定 {spec!r} を読めません(例 10-24、10-24:2、12,16,20)")
            a, b, st = m.groups()
            Ls += list(range(int(a), int(b) + 1, int(st or 1))) if b else [int(a)]
    return sorted({L for L in Ls if 1 <= L < n_layer - 1})


def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True)))


def last_logprobs(session, convs: list[list[dict]], vecs, lora) -> np.ndarray:
    """会話ごとの生成プロンプト末尾の次トークン分布(log-softmax)。"""
    e = session.engine
    ctx = e.steer_ctx()
    seqs = [e.tokenize_prompt(e.render_chat(c)) for c in convs]
    out = []
    e.set_lora(ctx, lora)
    try:
        e.set_cvec(ctx, vecs or None)
        for c0 in range(0, len(seqs), e.n_seq_max):
            e.clear_memory(ctx)
            out.append(e.prefill(ctx, seqs[c0:c0 + e.n_seq_max]).logits.astype(np.float64))
    finally:
        e.set_cvec(ctx, None)
        e.set_lora(ctx, None)
    return _log_softmax(np.concatenate(out))


def kl(base_lp: np.ndarray, lp: np.ndarray) -> float:
    """平均 KL(base‖other)。"""
    return float(np.mean(np.sum(np.exp(base_lp) * (base_lp - lp), axis=1)))


def select_layer(session, ds: DirectionSet, cs: ContrastSet, val: Validate, *, mode: str = "ablate",
                 layers: list[int] | None = None, dose: float = 0.5, max_tokens: int = 64, kl_max: float = 0.1,
                 min_distinct: float = 0.8, per_layer: bool = False, judge=None,
                 progress: Progress | None = None) -> dict:
    from .ablate import (
        embed_in_lora,
        embed_projection,
        export_ablation_lora,
        projection_rows,
        residual_writers,
        vector_key,
        write_ablation_lora,
    )
    from .evalx import _generate
    if mode not in ("ablate", "add"):
        raise GneError("BAD_ARGS", "mode は ablate か add です")
    rx = re.compile(val.match, re.IGNORECASE) if val.match else None
    if rx is None and judge is None:
        raise GneError("BAD_CONTRAST", "層を選ぶには対照集合の validate.match(判定の正規表現)か --judge-model が要ります")
    e = session.engine
    Ls = [L for L in (layers or candidate_layers(e.n_layer)) if L in ds.raw]
    if not Ls:
        raise GneError("BAD_ARGS", "候補の層がありません")
    t0 = time.perf_counter()
    if mode == "ablate":
        gen_convs = [[{"role": "system", "content": cs.system_a}, {"role": "user", "content": u}] for u in val.prompts]
        kl_convs = [[{"role": "system", "content": cs.system_b}, {"role": "user", "content": u}] for u in cs.user_prompts]
    else:
        gen_convs = [[{"role": "user", "content": u}] for u in val.prompts]
        kl_convs = gen_convs
    steps = len(Ls) + 1 + (1 if per_layer and mode == "ablate" else 0)
    done = 0

    def tick(msg: str):
        nonlocal done
        done += 1
        if progress:
            progress(done, steps, msg)

    def measure(vecs, lora) -> dict:
        outs = _generate(session, [(c, max_tokens) for c in gen_convs], vecs, lora, "all", lambda n: None)
        row = {"distinct4": float(np.mean([distinct4(o["token_ids"]) for o in outs])),
               "kl": kl(base_lp, last_logprobs(session, kl_convs, vecs, lora)) if base_lp is not None else 0.0}
        if rx is not None:
            row["rate"] = float(np.mean([bool(rx.search(o["text"])) for o in outs]))
        if judge is not None:
            sc = [judge.score(o["text"]) for o in outs]
            vals = [x for x in sc if x is not None]
            row["judge"] = float(np.mean(vals)) if vals else None
        row["sample"] = outs[0]["text"][:200]
        return row

    base_lp = None
    base = measure(None, None)
    base_lp = last_logprobs(session, kl_convs, None, None)
    base["kl"] = 0.0
    tick("direction: base")
    key = "rate" if rx is not None else "judge"
    sign = -1.0 if mode == "ablate" else 1.0

    def effect(row) -> float | None:
        if row.get(key) is None or base.get(key) is None:
            return None
        return sign * (row[key] - base[key])

    rows = []
    loras: dict[int, Path] = {}
    if mode == "ablate":                            # 全候補の rᵀW をまとめて 1 回で計算する
        adir = session.store.ablate_dir(session.meta.model_id)
        emb = embed_in_lora(session.weights)                  # tied でなければ埋め込みも消す(gne ablate の既定と同じ)
        need = {L: adir / f"{ds.name}-L{L}-{vector_key(ds.raw[L])}{'-e' if emb else ''}.lora.gguf" for L in Ls}
        todo = [L for L, p in need.items() if not p.exists()]
        if todo:
            R = np.stack([ds.unit(L) for L in todo])
            rows_all = projection_rows(session.weights, R)
            E = embed_projection(session.weights, R) if emb else None
            for i, L in enumerate(todo):
                tmp = need[L].with_suffix(".part")
                write_ablation_lora(session.weights, {n: R[i] for n in rows_all}, {n: v[i] for n, v in rows_all.items()},
                                    tmp, meta={"direction": ds.name, "source_layer": L,
                                               "layers": sorted(residual_writers(session.weights))},
                                    embed=(E[:, i], R[i]) if emb else None)
                tmp.replace(need[L])
        loras = need
    sv = session._need_rho(Ls) if mode == "add" else None
    for L in Ls:
        if mode == "add":
            row = measure({L: (dose * ds.raw[L]).astype(np.float32)}, None)
            row["rel_dose"] = dose * float(np.linalg.norm(ds.raw[L])) / sv.rho(L)
        else:
            row = measure(None, [(str(loras[L]), 1.0)])
        row["layer"] = L
        row["effect"] = effect(row)
        row["ok"] = row["kl"] <= kl_max if mode == "ablate" else row["distinct4"] >= min_distinct
        rows.append(row)
        tick(f"direction: L{L}")
    per = None
    if per_layer and mode == "ablate":
        p = session.store.ablate_dir(session.meta.model_id) / f"{ds.name}-perlayer.lora.gguf"
        export_ablation_lora(session.weights, ds.raw, p, meta={"direction": ds.name, "per_layer": True})
        per = measure(None, [(str(p), 1.0)])
        per["effect"] = effect(per)
        per["ok"] = per["kl"] <= kl_max
        tick("direction: per-layer")
    ok = [r for r in rows if r["ok"] and r["effect"] is not None and r["effect"] > 0]   # 効果が無い層は選ばない
    best = max(ok, key=lambda r: (r["effect"], -r["kl"])) if ok else None
    return {"mode": mode, "metric": key, "dose": dose if mode == "add" else None, "max_tokens": max_tokens,
            "criteria": {"kl_max": kl_max} if mode == "ablate" else {"min_distinct": min_distinct},
            "n_prompts": len(gen_convs), "base": base, "rows": rows, "per_layer": per,
            "selected": best["layer"] if best else None, "seconds": time.perf_counter() - t0}


def layer_table(ds: DirectionSet, rho: Callable[[int], float | None]) -> list[dict]:
    """層ごとの ‖平均差‖、‖平均差‖/ρ_L(rel dose に直した大きさ)、前の層との cos。"""
    out, prev = [], None
    for L in sorted(ds.raw):
        u = ds.unit(L)
        n = float(np.linalg.norm(ds.raw[L]))
        r = rho(L)
        out.append({"layer": L, "norm": n, "rel": n / r if r else None,
                    "cos_prev": float(u @ prev) if prev is not None else None})
        prev = u
    return out


def save_select(path: Path, sel: dict) -> None:
    path.write_text(json.dumps(sel, ensure_ascii=False, indent=1))
