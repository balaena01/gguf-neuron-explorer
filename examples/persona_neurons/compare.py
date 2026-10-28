"""1 つのレシピを押し方(when)だけ変えて、性格と自然さを比べる(docs/persona_workflow.md の手順 6)。

どの押し方も 1 本ずつ generate_when で、同じプロンプトには同じ種を使う(押し方どうしで対になる)。
能力の低下は別に `gne eval --recipe R --when W` で測る。

使い方: python compare.py MODEL CONFIG.yaml RECIPE.yaml OUT_DIR [--when all assistant gen ramp:0.5-1] [--judge self|file]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    generate,
    judge_file,
    judge_self,
    load_config,
    summarize,
    write_rubric,
)

from gne.api import Session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("config")
    ap.add_argument("recipe")
    ap.add_argument("out_dir")
    ap.add_argument("--when", nargs="+", default=["all", "assistant", "gen"])
    ap.add_argument("--seeds", default="5007,5031")
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--judge", choices=("self", "file"), default="self")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.judge == "file":
        write_rubric(out, cfg, a.max_tokens)
    units = [f"{u['layer']}:{u['neuron']}:{float(u['dose']):+.4f}" for u in yaml.safe_load(Path(a.recipe).read_text())["units"]]
    seeds = tuple(int(v) for v in a.seeds.split(","))
    s = Session(a.model)
    items = []
    for w in a.when:
        items += [dict(r, cand=w) for r in generate(s, cfg["prompts"], units, w, seeds, max_tokens=a.max_tokens, batched=False)]
        print(f"{w}: 生成しました", flush=True)
    scores = judge_self(s, items, cfg["target"]) if a.judge == "self" else judge_file(out, "when", items)
    res = {w: summarize([sc for it, sc in zip(items, scores, strict=True) if it["cand"] == w]) for w in a.when}
    for w, v in res.items():
        print(f"{w:14}: {v['score']:.2f}(性格 {v['persona']:.2f} / 自然さ {v['coherence']:.2f})")
    (out / "compare.json").write_text(json.dumps(
        {"recipe": a.recipe, "result": res, "rows": [dict(it, persona=p, coherence=c) for it, (p, c) in zip(items, scores, strict=True)]},
        ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
