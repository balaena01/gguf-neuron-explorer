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
from .controls import control_vectors
from .controls import null_pct as ctrl_null_pct
from .controls import summarize as summarize_ctrl
from .steer import GneError, random_direction

Progress = Callable[[int, int, str], None]


@dataclass
class Validate:
    prompts: list[str]
    match: str | None
    target: str | None
    explicit: bool = True                    # validate.prompts を書いたか(書かなければ A 側の質問)
    neutral: list[str] | None = None         # 中立の側(B)の質問。足して上がるかを見る測り方(add)で、書いていないときに使う

    def add_prompts(self) -> list[str]:
        """「中立に足して振る舞いが出るか」を見る質問: 書いた validate.prompts、無ければ B 側の質問。"""
        return list(self.prompts) if self.explicit or self.neutral is None else list(self.neutral)


def load_contrast(path: str | Path) -> tuple[ContrastSet, Validate]:
    from .reverse import check_questions
    cs = ContrastSet.load(path)
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    v = d.get("validate") or {}
    check_questions(v.get("prompts"), f"{path}: validate.prompts")
    if v.get("match"):
        try:
            re.compile(v["match"])
        except re.error as e:
            raise GneError("BAD_CONTRAST", f"validate.match の正規表現が読めません: {e}") from e
    return cs, Validate(list(v.get("prompts") or cs.prompts("A")), v.get("match"), v.get("target"),
                        explicit=bool(v.get("prompts")), neutral=cs.prompts("B"))


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
                ("method", "positions", "contrast_sha", "source", "n_rows_A", "n_rows_B", "base_layer", "base_positions")
                if k in z.files}
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
             **{f"scale_{L}": cr.scale[L] for L in Ls}, **{f"act_{L}": a for L, a in (cr.act_diff or {}).items()},
             **{f"actstd_{L}": a for L, a in (cr.act_std or {}).items()},
             cos=(U @ U.T).astype(np.float32),
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
                 min_distinct: float = 0.8, per_layer: bool = False, judge=None, random_controls: int = 0,
                 progress: Progress | None = None) -> dict:
    """候補層ごとに方向を当てて効果(判定の一致率の変化、符号は「望む向き」が正)と KL を測り、効果の大きい層を選ぶ。
    random_controls > 0 なら、同じ大きさのランダムな向きを対照にする(CHANGE-03 A)。add は層ごとに同じノルムの
    ランダムなベクトルを足し、ablate はランダムな単位ベクトルを全層から消す(層によらないので 1 回だけ測る)。
    各層の行に対照の分布と null_pct を足し、ok の条件に「効果 > 対照の p95」を加える。"""
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
    from .reverse import combine, side_interventions
    ivs = side_interventions(session, cs) if cs.has_recipes else {"A": (None, None), "B": (None, None)}
    if mode == "ablate":                            # A 側(system_a か recipe_a)で一致率が下がるか、B 側の KL
        gen_convs = cs.convs("A", val.prompts)
        kl_convs = cs.convs("B")
        gen_iv, kl_iv = ivs["A"], ivs["B"]
    else:                                           # 中立(system なし。recipe_b があればそれを当てた状態)に足して上がるか
        gen_convs = [[{"role": "user", "content": u}] for u in val.add_prompts()]
        kl_convs = gen_convs
        gen_iv = kl_iv = ivs["B"]
    n_ctrl = max(0, int(random_controls))
    # add の対照は、選ばれ得る層(基準を満たし効果 > 0)だけで流す(後から足す)。ablate の対照は層によらず 1 回
    steps = len(Ls) + 1 + (1 if per_layer and mode == "ablate" else 0) + (n_ctrl if mode == "ablate" else 0)
    done = 0

    def tick(msg: str):
        nonlocal done
        done += 1
        if progress:
            progress(done, steps, msg)

    def measure(vecs, lora) -> dict:
        outs = _generate(session, [(c, max_tokens) for c in gen_convs], *combine(gen_iv, vecs, lora), "all", lambda n: None)
        row = {"distinct4": float(np.mean([distinct4(o["token_ids"]) for o in outs])),
               "kl": kl(base_lp, last_logprobs(session, kl_convs, *combine(kl_iv, vecs, lora))) if base_lp is not None else 0.0}
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
    base_lp = last_logprobs(session, kl_convs, *combine(kl_iv))
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
    ablate_ctrl = []                                # ablate の対照: ランダムな向きを全層から消す(層によらない)
    if mode == "ablate" and n_ctrl:
        from .controls import SEED_BASE
        adir = session.store.ablate_dir(session.meta.model_id)
        Rc = np.stack([random_direction(session.meta.n_embd, SEED_BASE + j, 0) for j in range(n_ctrl)])
        cpaths = [adir / f"random{SEED_BASE + j}-{vector_key(Rc[j])}{'-e' if emb else ''}.lora.gguf" for j in range(n_ctrl)]
        todo = [j for j in range(n_ctrl) if not cpaths[j].exists()]
        if todo:
            rows_all = projection_rows(session.weights, Rc[todo])
            E = embed_projection(session.weights, Rc[todo]) if emb else None
            for i, j in enumerate(todo):
                tmp = cpaths[j].with_suffix(".part")
                write_ablation_lora(session.weights, {n: Rc[j] for n in rows_all}, {n: v[i] for n, v in rows_all.items()},
                                    tmp, meta={"direction": f"random{SEED_BASE + j}", "control": True,
                                               "layers": sorted(residual_writers(session.weights))},
                                    embed=(E[:, i], Rc[j]) if emb else None)
                tmp.replace(cpaths[j])
        for j in range(n_ctrl):
            c = measure(None, [(str(cpaths[j]), 1.0)])
            ablate_ctrl.append({"seed": SEED_BASE + j, "effect": effect(c), "kl": c["kl"], "distinct4": c["distinct4"]})
            tick(f"direction: random{j}")

    def add_controls(row, ctrl):
        if not ctrl:
            return
        es = summarize_ctrl([c["effect"] for c in ctrl])
        ks = summarize_ctrl([c["kl"] for c in ctrl])
        row["ctrl"] = {"n": len(ctrl), "effect": es, "kl": ks, "rows": ctrl}
        row["ctrl_effect_p50"], row["ctrl_effect_p95"], row["ctrl_effect_max"] = es["p50"], es["p95"], es["max"]
        row["ctrl_kl_p50"] = ks["p50"]
        row["null_pct"] = ctrl_null_pct(row["effect"], [c["effect"] for c in ctrl])

    sv = session._need_rho(Ls) if mode == "add" else None
    vecs: dict[int, np.ndarray] = {}
    for L in Ls:
        if mode == "add":
            vecs[L] = (dose * ds.raw[L]).astype(np.float32)
            row = measure({L: vecs[L]}, None)
            row["rel_dose"] = dose * float(np.linalg.norm(ds.raw[L])) / sv.rho(L)
        else:
            row = measure(None, [(str(loras[L]), 1.0)])
        row["layer"] = L
        row["effect"] = effect(row)
        row["ok"] = row["kl"] <= kl_max if mode == "ablate" else row["distinct4"] >= min_distinct
        rows.append(row)
        tick(f"direction: L{L}")
    cand = {id(r) for r in rows if r["ok"] and r["effect"] is not None and r["effect"] > 0}
    if mode == "add" and n_ctrl:
        steps += n_ctrl * len(cand)
    for row in rows:
        ctrl = ablate_ctrl
        if mode == "add":
            if id(row) not in cand:
                if n_ctrl:
                    row["ctrl_skipped"] = True                  # 選ばれ得ない層には対照を流さない
                continue
            L = row["layer"]
            ctrl = []
            for j, cv in enumerate(control_vectors(session.meta.n_embd, vecs[L], L, n_ctrl)):
                c = measure({L: cv}, None)
                ctrl.append({"seed": j, "effect": effect(c), "kl": c["kl"], "distinct4": c["distinct4"]})
                tick(f"direction: L{L} random{j}")
        add_controls(row, ctrl)
        if ctrl and row.get("ctrl_effect_p95") is not None:      # 対照を超えない層は選ばない
            row["ok"] = bool(row["ok"] and row["effect"] is not None and row["effect"] > row["ctrl_effect_p95"])
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
            "n_prompts": len(gen_convs), "random_controls": n_ctrl, "base": base, "rows": rows, "per_layer": per,
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


def thin(v: list, n: int | None) -> list:
    """多いリストを n 個に等間隔に間引く(n が None か足りていればそのまま)。"""
    if not n or len(v) <= n:
        return list(v)
    return [v[int(i)] for i in sorted({int(x) for x in np.linspace(0, len(v) - 1, int(n)).round()})]


def behavior(session, cs: ContrastSet, val: Validate, recipe, *, max_tokens: int = 64,
             base_lp: np.ndarray | None = None, max_examples: int | None = None, with_kl: bool = True) -> dict:
    """消したい振る舞いがどれだけ出るか: A 側(system_a 付き)の検証プロンプトの出力で判定の一致率、
    B 側(system_b 付き)の次トークン分布の KL(base‖recipe)。recipe が None なら無介入。
    max_examples で質問を等間隔に間引く。with_kl=False なら KL を測らない(B 側を読まない)。"""
    from .evalx import _generate
    rx = re.compile(val.match, re.IGNORECASE) if val.match else None
    from .reverse import combine, side_interventions
    vecs = session.vectors(recipe)[0] if recipe is not None else None
    lora = session.lora_for(recipe) if recipe is not None else None
    ivs = side_interventions(session, cs) if cs.has_recipes else {"A": (None, None), "B": (None, None)}
    gen = cs.convs("A", thin(list(val.prompts), max_examples))
    kl_convs = thin(cs.convs("B"), max_examples)
    outs = _generate(session, [(c, max_tokens) for c in gen], *combine(ivs["A"], vecs, lora), "all", lambda n: None)
    lp = last_logprobs(session, kl_convs, *combine(ivs["B"], vecs, lora)) if with_kl else None
    return {"rate": float(np.mean([bool(rx.search(o["text"])) for o in outs])) if rx else None,
            "kl_b": kl(base_lp, lp) if base_lp is not None and lp is not None else 0.0, "lp": lp,
            "sample": outs[0]["text"][:200] if outs else ""}


def behavior_null(session, cs: ContrastSet, val: Validate, recipe, *, random_controls: int = 16, max_tokens: int = 64,
                  mode: str | None = None, sampling=None, seeds: int = 1, prompts: list[str] | None = None) -> dict:
    """レシピの効果(判定の一致率の変化)が、同じ大きさのランダムな対照レシピ(controls.control_recipes)の何割より
    大きいか(null_pct、CHANGE-03 A)。mode="add" は system なしの検証プロンプトで一致率が上がるのを、mode="ablate" は
    system_a 付きで下がるのを効果とみなす(既定は add ユニットが無く ablate があれば ablate、それ以外は add)。
    sampling(engine.Sampling)を渡すと、各プロンプトを seeds 本ずつサンプリングで生成する。無介入・レシピ・対照で同じ乱数を使う
    (違いは押し方だけ)。prompts で検証プロンプトを差し替えられる(場面のある組など)。"""
    from .controls import control_recipes, null_pct, summarize
    from .evalx import _generate
    from .when import parse_when
    if not val.match:
        raise GneError("BAD_CONTRAST", "対照の効果を測るには対照集合の validate.match(判定の正規表現)が要ります")
    rx = re.compile(val.match, re.IGNORECASE)
    mode = mode or ("ablate" if recipe.ablate is not None and not recipe.add_units else "add")
    if mode not in ("add", "ablate"):
        raise GneError("BAD_ARGS", "mode は add か ablate です")
    ps = list(prompts) if prompts else (list(val.prompts) if mode == "ablate" else val.add_prompts())
    n_rep = max(1, int(seeds)) if sampling is not None else 1
    from .reverse import combine, side_interventions
    ivs = side_interventions(session, cs) if cs.has_recipes else {"A": (None, None), "B": (None, None)}
    if mode == "ablate":
        convs = cs.convs("A", ps)
        when, sign, side_iv = "all", -1.0, ivs["A"]
    else:
        convs = [[{"role": "user", "content": u}] for u in ps]
        when, sign, side_iv = str(parse_when(recipe.when)), 1.0, ivs["B"]
    items = [(c, max_tokens) for _ in range(n_rep) for c in convs]          # 種ごとに全プロンプト(同じ順)

    def rate(r) -> tuple[float, str]:
        vecs = session.vectors(r)[0] if r is not None and r.add_units else None
        lora = session.lora_for(r) if r is not None else None
        outs = _generate(session, items, *combine(side_iv, vecs, lora), when, lambda n: None, sampling=sampling)
        hits = [bool(rx.search(o["text"])) for o in outs]
        hit_text = next((o["text"] for o, h in zip(outs, hits, strict=True) if h), None)
        return float(np.mean(hits)), ((hit_text if r is not None and hit_text else outs[0]["text"])[:240] if outs else "")

    base, _ = rate(None)
    got, sample = rate(recipe)
    ctrl = []
    for c in control_recipes(session, recipe, random_controls):
        cr, _ = rate(c)
        ctrl.append({"name": c.name, "rate": cr, "effect": sign * (cr - base)})
    effect = sign * (got - base)
    return {"mode": mode, "when": when, "n_prompts": len(convs), "n_samples": len(items), "max_tokens": max_tokens,
            "sampling": None if sampling is None else sampling.to_dict(), "seeds": n_rep,
            "prompts": None if not prompts else ps, "base_rate": base, "rate": got, "effect": effect,
            "sample": sample, "random_controls": len(ctrl), "ctrl": summarize([c["effect"] for c in ctrl]),
            "ctrl_rows": ctrl, "null_pct": null_pct(effect, [c["effect"] for c in ctrl])}


def top_writer_neurons(session, ds: DirectionSet, n: int) -> list[tuple[int, int, float]]:
    """方向に書き込むニューロンの上位 n 個 [(層, ニューロン, cos)]。層ごとの r_L と W_down の列の cos(逆引きと同じ標準化)で、
    + 向き(押すと A 側に寄る)のものだけ。層 0 は除く。"""
    from .reverse import rank_by_direction
    cands, _ = rank_by_direction(session.weights, ds.raw, top=max(4 * n, 200), exclude_layers=(0,),
                                 scale=ds.scale or None)
    pos = [c for c in cands if c.sign > 0]
    pos.sort(key=lambda c: -c.cos)
    return [(c.layer, c.neuron, c.cos) for c in pos[:n]]


def response_directions(session, cs: ContrastSet, *, max_tokens: int = 96, with_act: bool = True,
                        progress: Progress | None = None):
    """返事のトークンから作る方向(positions="response"、persona vector の作り方)。
    A(system_a)と B(system_b)で各プロンプトに greedy で返事を生成し、その会話(system・プロンプト・返事)を流して
    返事のトークンでの l_out-L の平均の差を取る。「指示の直後」ではなく「その性格で話している最中」の状態の差になる。
    with_act なら ffn_down の入力(ニューロンの活性)の平均の差も返事のトークンで取る(逆引きの act_diff、部品のニューロン探し用)。
    戻り値は reverse.ContrastResult と同じ形。"""
    from .capture import Capture, pick_act_name
    from .evalx import _generate
    from .reverse import ContrastResult
    e = session.engine
    Ls = list(range(e.n_layer - 1))                         # 最終層は行が間引かれ得るので除く
    names = {f"l_out-{L}": L for L in Ls}
    act_names: dict[str, int] = {}
    if with_act:
        from .reverse import _conv
        probe = e.tokenize_prompt(e.render_chat(_conv(None, cs.user_prompts[0])))
        seen = e.list_names(probe[:16])
        from .moe import moe_names
        for L in Ls:
            nm = pick_act_name(seen, L)
            if nm and "argsort" not in moe_names(seen, L):      # MoE の層の ffn_swiglu は shared expert の活性
                act_names[nm] = L
    act_sums = {side: {L: 0.0 for L in act_names.values()} for side in "AB"}
    act_sq = {L: 0.0 for L in act_names.values()}             # A∪B の活性の二乗和(効果量 = 差 ÷ 標準偏差 に使う)
    sums = {side: {L: np.zeros(e.n_embd) for L in Ls} for side in "AB"}
    sq = {L: np.zeros(e.n_embd) for L in Ls}
    cnt = {"A": 0, "B": 0}
    total, step = len(cs.prompts("A")) + len(cs.prompts("B")), 0
    from .reverse import side_interventions
    ivs = side_interventions(session, cs) if cs.has_recipes else {"A": (None, None), "B": (None, None)}
    for side in "AB":
        convs = cs.convs(side)
        vecs, lora = ivs[side]
        outs = _generate(session, [(c, max_tokens) for c in convs], vecs, lora, "all", lambda n: None)
        cap = Capture()
        cap.set_wanted(lambda n: n in names or n in act_names)
        need = max(len(e.tokenize_prompt(e.render_chat(c))) for c in convs) + max_tokens   # 質問 + 返事の上限
        for c, o in zip(convs, outs, strict=True):
            pids = e.tokenize_prompt(e.render_chat(c))
            resp = [t for t in o["token_ids"] if not e.is_eog(t)]
            if not resp:
                continue
            ctx = e.probe_ctx(cap, need=need)
            e.clear_memory(ctx)
            e.set_cvec(ctx, vecs or None)
            e.set_lora(ctx, lora or None)
            try:
                e.prefill(ctx, [pids + resp])
            finally:
                e.set_cvec(ctx, None)
                e.set_lora(ctx, None)
            got = cap.take()
            for nm, L in names.items():
                rows = got[nm][len(pids):].astype(np.float64)        # 返事のトークンだけ
                sums[side][L] += rows.sum(axis=0)
                sq[L] += np.square(rows).sum(axis=0)
            for nm, L in act_names.items():
                a = got[nm][len(pids):].astype(np.float64)
                act_sums[side][L] = act_sums[side][L] + a.sum(axis=0)
                act_sq[L] = act_sq[L] + np.square(a).sum(axis=0)
            cnt[side] += len(resp)
            step += 1
            if progress:
                progress(step, total, "direction: response")
    if not cnt["A"] or not cnt["B"]:
        raise GneError("BAD_CONTRAST", "返事が生成されませんでした")
    n = cnt["A"] + cnt["B"]
    dirs = {L: (sums["A"][L] / cnt["A"] - sums["B"][L] / cnt["B"]).astype(np.float32) for L in Ls}
    scale = {L: np.sqrt(np.maximum(sq[L] / n - ((sums["A"][L] + sums["B"][L]) / n) ** 2, 0)).astype(np.float32) for L in Ls}
    act = {L: (act_sums["A"][L] / cnt["A"] - act_sums["B"][L] / cnt["B"]).astype(np.float32) for L in act_names.values()}
    # ニューロンごとの活性の標準偏差(A∪B)。act_diff ÷ これ = 効果量で、もともと活性の大きいニューロンが上に来すぎない
    act_std = {L: np.sqrt(np.maximum(act_sq[L] / n - ((act_sums["A"][L] + act_sums["B"][L]) / n) ** 2, 0)).astype(np.float32)
                  for L in act_names.values()}
    return ContrastResult(dirs, act, "mean_diff", "response", cnt, scale, act_std)


# --- 速い指標(CHANGE-06 B、`gne direction score`)------------------------------------------------------
#
# 射影の値 = 測定位置での残差 l_out-L と、その層の方向の単位ベクトル r̂_L の内積(例ごとに位置の平均)。対照集合の A 側
# (system_a + プロンプト)と B 側(system_b + プロンプト)を、返事を生成せずに生成プロンプトの最後まで読ませるだけで測る。
# 介入(レシピの LoRA・control vector)を当てたときの、残差の差の方向成分・直交成分(drift_par / drift_orth、残差の大きさで
# 割ったもの)も出す。強さの探索(heads causal・heads allocate)はすべてこれで回し、生成は確定候補の評価だけに使う。

def base_layer_of(ds: DirectionSet) -> int | None:
    """方向セットの基準層: gne direction score で記録した base_layer、無ければ gne direction --select で選んだ層。"""
    b = ds.meta.get("base_layer")
    if b is not None and int(b) in ds.raw:
        return int(b)
    sel = ds.path.with_name(ds.path.name.split(".")[0] + ".select.json")
    if sel.exists():
        try:
            chosen = json.loads(sel.read_text()).get("selected")
        except (OSError, ValueError):
            chosen = None
        if chosen is not None and int(chosen) in ds.raw:
            return int(chosen)
    return None


def set_base_layer(path: str | Path, L: int, *, positions: str | None = None) -> None:
    """方向セットの npz に基準層(base_layer)を書き込む(他の配列はそのまま)。"""
    p = Path(path)
    with np.load(p) as z:
        d = {k: z[k] for k in z.files}
    if str(int(L)) not in {str(int(x)) for x in d["layers"]}:
        raise GneError("BAD_ARGS", f"方向に層 {L} がありません")
    d["base_layer"] = np.int32(L)
    if positions:
        d["base_positions"] = np.array(positions)
    tmp = p.with_name(p.name[:-4] + ".tmp.npz") if p.name.endswith(".npz") else p.with_suffix(".tmp.npz")
    np.savez(tmp, **d)
    tmp.replace(p)


def score_layers_default(session, ds: DirectionSet, base: int) -> list[int]:
    """因果スクリーニングで射影の値を測る層の既定: 基準層と、後段の 2 層(基準層と最後の層の中間、最後の層)。"""
    top = max(L for L in ds.raw if L <= session.meta.n_layer - 2)
    mid = base + (top - base) // 2
    return sorted({base, mid, top})


def _subset(n: int, subset: str) -> list[int]:
    if subset in ("all", None):
        return list(range(n))
    if subset == "even":                       # 測定用(方向・因果・強さを決める)
        return list(range(0, n, 2))
    if subset == "odd":                        # 確認用(最終の評価だけ)
        return list(range(1, n, 2))
    raise GneError("BAD_ARGS", f"subset は all / even / odd です({subset!r})")


def _score_rows(n_tok: int, start: int, positions: str) -> slice:
    if positions == "last":
        return slice(n_tok - 1, n_tok)
    if positions == "assistant":               # 生成プロンプト(assistant の見出し)のトークン
        return slice(max(start, 1), n_tok)
    m = re.fullmatch(r"last:(\d+)", positions or "")
    if m and int(m.group(1)) >= 1:
        return slice(max(1, n_tok - int(m.group(1))), n_tok)
    raise GneError("BAD_ARGS", f"positions は last / last:K / assistant です({positions!r})")


def _score_capture(e):
    from .capture import Capture
    cap = getattr(e, "_score_capture", None)
    if cap is None:
        cap = e._score_capture = Capture()
    return cap


class Scorer:
    """対照集合の A 側・B 側を prefill だけして、層ごとに 測定位置の残差 · r̂_L を測る(CHANGE-06 B)。
    トークン化は 1 回だけ。run() に LoRA・control vector を渡すと、それを当てた状態で測る。"""

    def __init__(self, session, ds: DirectionSet, cs: ContrastSet, *, layers, positions: str = "last",
                 subset: str = "all", max_examples: int | None = None):
        from .reverse import contrast_sequences
        e = session.engine
        self.session, self.ds, self.positions, self.subset = session, ds, positions, subset
        self.layers = sorted({int(L) for L in layers})
        if not self.layers:
            raise GneError("BAD_ARGS", "測る層がありません")
        for L in self.layers:
            if L not in ds.raw:
                raise GneError("BAD_ARGS", f"方向 {ds.name} に層 {L} がありません(0〜{max(ds.raw)})")
            if L >= e.n_layer - 1:
                raise GneError("BAD_ARGS", f"最終層 {L} は行が間引かれるので測れません")
        self.r = {L: ds.unit(L).astype(np.float64) for L in self.layers}
        seqs = contrast_sequences(e, cs)

        def pick(n: int) -> list[int]:
            idx = _subset(n, subset)
            if max_examples and len(idx) > max_examples:     # 質問が多いときは等間隔に間引く(何百問の対照集合で試行を速く)
                sel = sorted({int(i) for i in np.linspace(0, len(idx) - 1, int(max_examples)).round()})
                idx = [idx[i] for i in sel]
            if not idx:
                raise GneError("BAD_CONTRAST", "プロンプトがありません")
            return idx
        # 側ごとに質問が違う対照集合(prompts_a / prompts_b)では、A と B を別々に選ぶ。index・prompts は A 側
        self.index_side = {side: pick(len(cs.prompts(side))) for side in "AB"}
        self.index = self.index_side["A"]
        self.prompts_side = {side: [cs.prompts(side)[i] for i in self.index_side[side]] for side in "AB"}
        self.prompts = self.prompts_side["A"]
        self.items = {side: [(seqs[side][i][0], _score_rows(len(seqs[side][i][0]), seqs[side][i][1], positions))
                             for i in self.index_side[side]] for side in "AB"}
        self.max_len = max(len(ids) for side in "AB" for ids, _ in self.items[side])   # コンテキストをこの長さに合わせる
        from .reverse import side_interventions
        # レシピで A / B を作る対照集合: 側ごとのレシピをいつも当て、調べる介入(run の lora・vecs)はその上に重ねる
        self.sides = side_interventions(session, cs) if cs.has_recipes else {"A": (None, None), "B": (None, None)}
        for side in "AB":
            for ids, _ in self.items[side]:
                if len(ids) > e.n_ctx:
                    raise GneError("CONTEXT_FULL", f"{len(ids)} トークンの会話が n_ctx={e.n_ctx} に入りません")

    def n(self, side: str) -> int:
        return len(self.items[side])

    def run(self, lora=None, vecs=None, *, sides: str = "AB", which: dict[str, list[int]] | None = None,
            keep: bool = False) -> dict[str, dict]:
        """{side: {"values": {L: (n,)}, "idx": [例の番号], "resid": {L: [(P, n_embd)]}(keep のとき)}}。
        which[side] で測る例(self.items の添字)を絞れる。"""
        from .reverse import combine
        e = self.session.engine
        cap = _score_capture(e)
        names = {f"l_out-{L}": L for L in self.layers}
        cap.set_wanted(lambda nm: nm in names)
        ctx = e.probe_ctx(cap, need=self.max_len)
        out: dict[str, dict] = {}
        try:
            for side in sides:
                sv, sl = combine(self.sides.get(side), vecs, lora)
                e.set_lora(ctx, sl)
                e.set_cvec(ctx, sv)
                items = self.items[side]
                sel = list(range(len(items))) if which is None or side not in which else list(which[side])
                vals = {L: [] for L in self.layers}
                res: dict[int, list] = {L: [] for L in self.layers}
                for i in sel:
                    ids, rows = items[i]
                    e.clear_memory(ctx)
                    e.prefill(ctx, [ids])
                    got = cap.take()
                    for nm, L in names.items():
                        if nm not in got:
                            raise GneError("ARCH_UNSUPPORTED", f"{nm} が捕捉できません")
                        x = got[nm][rows].astype(np.float64)
                        vals[L].append(float((x @ self.r[L]).mean()))
                        if keep:
                            res[L].append(x)
                out[side] = {"values": {L: np.array(v) for L, v in vals.items()}, "idx": sel}
                if keep:
                    out[side]["resid"] = res
        finally:
            e.set_cvec(ctx, None)
            e.set_lora(ctx, None)
        return out


def drift(base_resid: list[np.ndarray], new_resid: list[np.ndarray], r: np.ndarray) -> tuple[float, float]:
    """介入前後の残差の差 Δ の、r̂ に沿った成分と直交成分の大きさ ÷ 介入前の残差の大きさ(位置ごと、全例の平均)。"""
    par, orth = [], []
    for x0, x1 in zip(base_resid, new_resid, strict=True):
        d = x1 - x0
        p = d @ r
        o = np.linalg.norm(d - np.outer(p, r), axis=1)
        nrm = np.maximum(np.linalg.norm(x0, axis=1), 1e-12)
        par += list(np.abs(p) / nrm)
        orth += list(o / nrm)
    return (float(np.mean(par)) if par else 0.0, float(np.mean(orth)) if orth else 0.0)


def _stats(v: np.ndarray) -> dict:
    return {"mean": float(v.mean()) if v.size else None, "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "n": int(v.size), "max": float(v.max()) if v.size else None, "min": float(v.min()) if v.size else None}


def cohen_d(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or b.size < 2:
        return None
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sd) if sd > 0 else None


def score(session, ds: DirectionSet, cs: ContrastSet, *, layers=None, positions: str = "last", units=None, lora=None,
          side: str = "both", subset: str = "all", record_base: bool | None = None, progress: Progress | None = None,
          max_examples: int | None = None) -> dict:
    """速い指標(CHANGE-06 B)。層ごとに A 側・B 側の射影の値の mean / sd / cohen_d、例ごとの射影の値、基準層の方向とのコサイン。
    units(レシピかユニットの書き方のリスト)か lora([(パス, 倍率)])を渡すと、その介入を当てた状態でも測り、射影の値の変化と
    残差の差の方向成分・直交成分(drift_par / drift_orth)を出す。介入が無く layers を省いたときは、cohen_d が最大の層を
    基準層(base_layer)として方向セットに記録する。記録するのは基準層がまだ無いときだけ(record_base=True で上書き、
    False で記録しない)。"""
    t0 = time.perf_counter()
    if side not in ("both", "A", "B"):
        raise GneError("BAD_ARGS", "side は both / A / B です")
    auto_layers = layers is None
    n_layer = session.meta.n_layer
    Ls = sorted(L for L in ds.raw if 1 <= L <= n_layer - 2) if auto_layers else sorted({int(L) for L in layers})
    sc = Scorer(session, ds, cs, layers=Ls, positions=positions, subset=subset, max_examples=max_examples)
    recipe = None
    if units is not None:
        recipe = units if hasattr(units, "units") else session.recipe(units=list(units))
    vecs = None
    if recipe is not None:
        vecs = session.vectors(recipe)[0] if recipe.add_units else None
        lora = list(lora or []) + list(session.lora_for(recipe) or [])
    has_iv = bool(lora) or bool(vecs)
    sides = "AB" if side == "both" else side
    base = sc.run(sides=sides, keep=has_iv)
    if progress:
        progress(1, 2 if has_iv else 1, "direction score")
    after = sc.run(lora, vecs, sides=sides, keep=True) if has_iv else None
    if progress and has_iv:
        progress(2, 2, "direction score")
    d_by_layer = {L: (cohen_d(base["A"]["values"][L], base["B"]["values"][L]) if side == "both" else None) for L in Ls}
    best = max((L for L in Ls if d_by_layer[L] is not None), key=lambda L: d_by_layer[L], default=None)
    recorded = base_layer_of(ds)
    # 記録するのは、介入なし・全層・両側のときで、record_base=True か、まだ基準層が記録されていない(None)とき。
    # 手で決めた基準層(--set-base)を、測り直すたびに上書きしないため
    can = auto_layers and not has_iv and side == "both" and best is not None
    do_record = can and (record_base is True or (record_base is None and ds.meta.get("base_layer") is None))
    base_L = best if do_record else (recorded if recorded is not None else best)
    rows = []
    for L in Ls:
        row: dict = {"layer": L, "cohen_d": d_by_layer[L],
                     "cos_base": float(ds.unit(L) @ ds.unit(base_L)) if base_L is not None and base_L in ds.raw else None}
        for sd in sides:
            row[sd] = _stats(base[sd]["values"][L])
        if after is not None:
            row["after"] = {sd: _stats(after[sd]["values"][L]) for sd in sides}
            if side == "both":
                row["after"]["cohen_d"] = cohen_d(after["A"]["values"][L], after["B"]["values"][L])
            row["drop"] = {sd: float(base[sd]["values"][L].mean() - after[sd]["values"][L].mean()) for sd in sides}
            row["drift"] = {}
            for sd in sides:
                par, orth = drift(base[sd]["resid"][L], after[sd]["resid"][L], sc.r[L])
                row["drift"][sd] = {"par": par, "orth": orth}
        rows.append(row)
    if do_record:
        set_base_layer(ds.path, best, positions=positions)
    out = {"direction": ds.name, "path": str(ds.path), "contrast_sha": cs.sha, "positions": positions, "subset": subset,
           "n": {sd: sc.n(sd) for sd in sides}, "layers": rows, "best_layer": best, "base_layer": base_L,
           "base_recorded": bool(do_record),
           "values": {sd: {str(L): base[sd]["values"][L].tolist() for L in Ls} for sd in sides},
           "values_after": None if after is None else {sd: {str(L): after[sd]["values"][L].tolist() for L in Ls} for sd in sides},
           "prompts": sc.prompts, **({"prompts_b": sc.prompts_side["B"]} if cs.separate_prompts else {}),
           "intervention": None if not has_iv else
           {"recipe": None if recipe is None else recipe.to_dict(), "lora": [str(p) for p, _ in (lora or [])],
            "vectors": sorted(int(L) for L in (vecs or {}))},
           "seconds": time.perf_counter() - t0}
    return out
