"""ニューロンごとの強さ(符号つき)を (1+λ) 進化戦略で探す(docs/persona_workflow.md の手順 5)。

- 目的 = 返事ごとの min(性格, 自然さ) の平均。片方だけ高くても上がらない(壊れた返事で性格だけ高い組を選ばない)。
- 毎回(round)、親と λ−1 個の候補を同じプロンプト・同じ種で生成して採点し、同じ回の中で比べる。
  候補が親を上回ったときだけ乗り換える。種は回ごとに変える(固定すると、たまたま当たった組が残る、verification.md F95)。
- 候補の 1 つは全体の倍率だけを ±10〜20% 変え、残りは半分ほどの強さを σ で揺らす(符号が変わることもある)。
- 最後に、探すのに使っていないプロンプト(valid_prompts)で親・始めの組・押さない状態を比べる。
- 採点は --judge self(モデル自身)か file(名前を伏せたファイル、Claude や人が採点。common.py を参照)。

START はレシピの YAML(units に layer・neuron・dose)か、1 行に 1 つ「L:N:dose」を書いたテキスト。
使い方: python tune.py MODEL CONFIG.yaml START OUT_DIR [--when all] [--rounds 10] [--lam 4] [--judge self]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    generate,
    judge_file,
    judge_self,
    load_config,
    summarize,
    units_of,
    write_rubric,
)

from gne.api import Session


def load_start(path: str) -> tuple[list[str], np.ndarray]:
    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        us = yaml.safe_load(p.read_text(encoding="utf-8"))["units"]
        return [f"{u['layer']}:{u['neuron']}" for u in us], np.array([float(u["dose"]) for u in us])
    names, doses = [], []
    for ln in p.read_text(encoding="utf-8").split():
        L, n, d = ln.split(":")
        names.append(f"{L}:{n}")
        doses.append(float(d))
    return names, np.array(doses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("config")
    ap.add_argument("start")
    ap.add_argument("out_dir")
    ap.add_argument("--when", default="all", help="all / gen / assistant / ramp:a-b(生成も採点もこの押し方)")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--lam", type=int, default=4, help="1 回に採点する組の数(親を含む)")
    ap.add_argument("--sigma", type=float, default=0.06)
    ap.add_argument("--max-dose", type=float, default=1.2)
    ap.add_argument("--seeds", default="7,31", help="1 回に使う種(回ごとに +1000 ずつずらす)")
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--judge", choices=("self", "file"), default="self")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.judge == "file":
        write_rubric(out, cfg, a.max_tokens)
    names, x0 = load_start(a.start)
    seeds = tuple(int(v) for v in a.seeds.split(","))
    s = Session(a.model)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()

    def judge_round(tag, cands, prompts, sds):
        items = []
        for name, x in cands:
            items += [dict(r, cand=name) for r in generate(s, prompts, units_of(names, x), a.when, sds, max_tokens=a.max_tokens)]
        scores = judge_self(s, items, cfg["target"]) if a.judge == "self" else judge_file(out, tag, items)
        res = {}
        for name, _ in cands:
            res[name] = summarize([sc for it, sc in zip(items, scores, strict=True) if it["cand"] == name])
        (out / f"round_{tag}.result.json").write_text(json.dumps(
            {"result": res, "rows": [dict(it, persona=p, coherence=c) for it, (p, c) in zip(items, scores, strict=True)]},
            ensure_ascii=False, indent=1), encoding="utf-8")
        return res

    x, sigma, hist = x0.copy(), a.sigma, []
    for g in range(1, a.rounds + 1):
        cands = [("parent", x.copy())]
        for j in range(1, a.lam):
            if j == 1:
                z = x * (1 + rng.choice([-1, 1]) * rng.uniform(0.1, 0.2))
            else:
                z = x + rng.normal(0, sigma, size=x.size) * (rng.random(x.size) < 0.5)
            cands.append((f"c{j}", np.clip(z, -a.max_dose, a.max_dose)))
        res = judge_round(f"{g:02d}", cands, cfg["prompts"], tuple(sd + 1000 * g for sd in seeds))
        best = max((k for k in res if k != "parent"), key=lambda k: res[k]["score"])
        accepted = res[best]["score"] > res["parent"]["score"]
        if accepted:
            x = dict(cands)[best].copy()
            sigma = min(0.15, sigma * 1.1)
        else:
            sigma = max(0.03, sigma * 0.85)
        hist.append({"round": g, "accepted": best if accepted else None, "sigma": sigma, "x": x.tolist(), "tried": res})
        print(f"round {g}: 親 {res['parent']['score']:.2f}(性格 {res['parent']['persona']:.2f} / 自然さ {res['parent']['coherence']:.2f})"
              f"  候補 " + " ".join(f"{k} {v['score']:.2f}" for k, v in res.items() if k != "parent") +
              f"  → {'採用 ' + best if accepted else 'そのまま'}  σ {sigma:.2f}  {time.perf_counter() - t0:.0f}s", flush=True)
        (out / "result.json").write_text(json.dumps({"names": names, "start": x0.tolist(), "hist": hist,
                                                     "units": units_of(names, x)}, ensure_ascii=False, indent=1), encoding="utf-8")
    val = {}
    if cfg["valid_prompts"]:
        val = judge_round("valid", [("parent", x), ("start", x0), ("unsteered", np.zeros_like(x0))], cfg["valid_prompts"],
                          tuple(sd + 101 for sd in seeds))
        for k, v in val.items():
            print(f"確かめ {k:10}: {v['score']:.2f}(性格 {v['persona']:.2f} / 自然さ {v['coherence']:.2f})", flush=True)
    (out / "result.json").write_text(json.dumps({"names": names, "start": x0.tolist(), "hist": hist, "units": units_of(names, x),
                                                 "validation": val}, ensure_ascii=False, indent=1), encoding="utf-8")
    recipe = {"name": f"{cfg.get('name', 'persona')}_tuned", "dose_unit": "rel", "when": a.when,
              "units": [{"layer": int(n.split(":")[0]), "neuron": int(n.split(":")[1]), "dose": round(float(d), 3)}
                        for n, d in zip(names, x, strict=True) if abs(d) > 1e-3]}
    (out / "recipe.yaml").write_text(yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"レシピ: {out / 'recipe.yaml'}(gne の --recipe や ~/.gne/recipes に置いて使う)")


if __name__ == "__main__":
    main()
