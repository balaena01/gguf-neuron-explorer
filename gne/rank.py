"""語彙のかたまりによる順位づけ。設計書 §6.8(参考実装の指標を、コードを流用せず再実装)。

1 ユニット(ニューロンかランダム対照)ごとに:
- rel_push ≤ max_rel の rollout から語の集合を取る(rollout 内では 1 回だけ数える)。ベースラインに出る語は除く。
- 全ユニットの平均 mean_w に対する Poisson z: z = (count − mean_w) / sqrt(mean_w + 0.5)、count ≥ 2 のみ。
- mass = 上位 5 語の正の z の和(並べ替えのキー)。null_pct = mass がそれ未満のランダム対照の割合(1.0 = 全対照より上)。
- 補助: dose ごとの distinct-4-gram 比(反復度)・length 終了率(capped)・平均 logprob、語がどのプロンプトで出たか(per_prompt)。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence

import numpy as np

_REGEX = re.compile(r"[^\W\d_]{3,}")


def rollout_words(rec: dict, word_unit: str, token_words: Sequence[str | None] | None) -> set[str]:
    if word_unit == "regex":
        return set(_REGEX.findall(rec["text"].casefold()))
    if token_words is None:
        raise ValueError("word_unit=tokens には語彙(token_words)が要ります")
    ids = rec["token_ids"][:-1] if rec.get("finish") == "eog" else rec["token_ids"]
    return {w for w in (token_words[t] for t in ids if 0 <= t < len(token_words)) if w}


def distinct4(ids: Sequence[int]) -> float:
    if len(ids) < 4:
        return 1.0
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return len(set(grams)) / len(grams)


def _dose_key(d: float) -> str:
    return f"{d:+g}"


def rank_records(records: list[dict], *, word_unit: str = "tokens", token_words: Sequence[str | None] | None = None,
                 max_rel: float = 1.6, top_words: int = 5, keep_words: int = 20,
                 progress: Callable[[int, int, str], None] | None = None) -> list[dict]:
    """rollout のレコード(sweep の JSONL)→ ユニットごとの要約(mass の大きい順)。"""
    baseline_words: set[str] = set()
    units: dict[tuple[str, int], list[dict]] = defaultdict(list)
    layer = None
    for r in records:
        if r["kind"] == "baseline":
            baseline_words |= rollout_words(r, word_unit, token_words)
        else:
            units[(r["kind"], int(r["neuron"]))].append(r)
            layer = r["layer"]
    if not units:
        return []

    counts: dict[tuple[str, int], Counter] = {}
    where: dict[tuple[str, int], dict[str, set[str]]] = {}
    total: Counter = Counter()
    for i, (key, recs) in enumerate(units.items()):
        c: Counter = Counter()
        wp: dict[str, set[str]] = defaultdict(set)
        for r in recs:
            if r["rel_push"] is None or r["rel_push"] > max_rel + 1e-9:
                continue
            for w in rollout_words(r, word_unit, token_words) - baseline_words:
                c[w] += 1
                wp[w].add(r["prompt"])
        counts[key], where[key] = c, wp
        total.update(c)
        if progress and (i + 1) % 64 == 0:
            progress(i + 1, len(units), "rank")
    n_units = len(units)

    out = []
    for key, recs in units.items():
        zs = []
        for w, cnt in counts[key].items():
            if cnt < 2:
                continue
            mean = total[w] / n_units
            zs.append((w, cnt, (cnt - mean) / math.sqrt(mean + 0.5)))
        zs.sort(key=lambda x: -x[2])
        mass = float(sum(z for _, _, z in zs[:top_words] if z > 0))
        dose = {}
        for d in sorted({r["dose"] for r in recs}):
            rs = [r for r in recs if r["dose"] == d]
            dose[_dose_key(d)] = {
                "rel_push": float(np.mean([r["rel_push"] for r in rs])),
                "distinct4": float(np.mean([distinct4(r["token_ids"]) for r in rs])),
                "capped": float(np.mean([r["finish"] == "length" for r in rs])),
                "logprob": float(np.mean([np.mean(r["logprob"]) if r["logprob"] else 0.0 for r in rs])),
            }
        out.append({"kind": key[0], "neuron": key[1], "layer": layer, "z": float(zs[0][2]) if zs else 0.0,
                    "mass": mass, "null_pct": None,
                    "words": [[w, int(c), round(float(z), 3)] for w, c, z in zs[:keep_words]],
                    "per_prompt": {w: sorted(where[key][w]) for w, _, _ in zs[:keep_words]},
                    "dose": dose})

    ctrl = np.sort([u["mass"] for u in out if u["kind"] == "random"])
    for u in out:
        if not ctrl.size:
            continue
        if u["kind"] == "random":                   # 対照どうしは自分を除いて比べる
            others = ctrl.size - 1
            below = int(np.searchsorted(ctrl, u["mass"], side="left"))
            u["null_pct"] = below / others if others else None
        else:
            u["null_pct"] = float(np.searchsorted(ctrl, u["mass"], side="left") / ctrl.size)
    out.sort(key=lambda u: (-u["mass"], u["kind"], u["neuron"]))
    r = 0
    for u in out:
        if u["kind"] == "neuron":
            r += 1
            u["rank"] = r
        else:
            u["rank"] = None
    return out
