"""lexsearch.py の候補を 1 個ずつ押して、部品の目標で採点する(docs/persona_workflow.md の手順 4)。

部品ごとに上位 N 個を、反応する場面のプロンプト(設定の prompts)で dose ごとに生成し、モデル自身に採点させる。
押さない状態の点(base)と、ランダムなニューロンを同じように押した点(対照)も出す。対照の最大を超えたものだけが残す候補。
anti の部品は全体の目標(target)で採点する(抑えると性格が出やすくなるか)。

使い方: python screen.py MODEL CONFIG.yaml LEXSEARCH.json OUT.json [--per-part 4] [--doses 0.5,1] [--controls 6] [--prompts 6]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import generate, judge_self, load_config, summarize

from gne.api import Session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("config")
    ap.add_argument("lexsearch")
    ap.add_argument("out")
    ap.add_argument("--per-part", type=int, default=4)
    ap.add_argument("--doses", default="0.5,1")
    ap.add_argument("--controls", type=int, default=6, help="ランダムなニューロンの数(対照)")
    ap.add_argument("--prompts", type=int, default=6, help="使うプロンプトの数(prompts の先頭から、Who are you? は除く)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    cfg = load_config(a.config)
    hits = json.loads(Path(a.lexsearch).read_text(encoding="utf-8"))["hits"]
    doses = [float(d) for d in a.doses.split(",")]
    prompts = [p for p in cfg["prompts"] if "who are you" not in p.lower() and "誰" not in p][:a.prompts]
    s = Session(a.model)
    layers = sorted({h["layer"] for hs in hits.values() for h in hs})
    rng = np.random.default_rng(a.seed)

    def score(units, target):
        rows = generate(s, prompts, units, "all", (a.seed,))
        return summarize(judge_self(s, rows, target)), [r["text"][:200] for r in rows]
    res = {}
    for name, hs in hits.items():
        target = cfg["parts"][name]["target"] if name in cfg["parts"] else cfg["target"]
        base, _ = score([], target)
        # 対照: 候補と同じ層の範囲から、ランダムなニューロンを同じ強さ(プラス)で押す
        ctrl = []
        for _ in range(a.controls):
            L = int(rng.choice(layers))
            n = int(rng.integers(0, s.meta.n_ff_at(L)))
            ctrl.append(max(score([f"{L}:{n}:+{d}"], target)[0]["persona"] for d in doses))
        rows = []
        seen = set()
        for h in hs:
            key = (h["layer"], h["neuron"])
            if key in seen:
                continue
            seen.add(key)
            row = {"unit": f"{h['layer']}:{h['neuron']}", "sign": h["sign"], "token": h["token"], "by_dose": {}}
            for d in doses:
                sc, texts = score([f"{h['layer']}:{h['neuron']}:{h['sign']}{d}"], target)
                row["by_dose"][str(d)] = {**sc, "texts": texts}
            row["best"] = max(v["persona"] for v in row["by_dose"].values())
            rows.append(row)
            print(f"{name:12} {row['unit']:>10}:{row['sign']} {row['token']!r:12} " +
                  " ".join(f"{d}: 性格 {row['by_dose'][str(d)]['persona']:.2f} 自然さ {row['by_dose'][str(d)]['coherence']:.2f}"
                           for d in doses) + f"  (素 {base['persona']:.2f}、対照の最大 {max(ctrl):.2f})", flush=True)
            if len(rows) >= a.per_part:
                break
        res[name] = {"target": target, "base": base, "controls": ctrl, "rows": rows,
                     "passed": [r["unit"] + ":" + r["sign"] for r in rows if r["best"] > max(ctrl)]}
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("対照の最大を超えた候補:", {k: v["passed"] for k, v in res.items()})


if __name__ == "__main__":
    main()
