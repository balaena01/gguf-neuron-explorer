"""振る舞いの計測(gne behave): レシピを当てて質問集に答えさせ、正規表現の一致と、読み手のモデルの判定を数える。

判定の文面(--judge)は使う人が渡す。{q} に質問、{a} に返事が入り、返事の最後に出てきた --labels の文字を判定にする。
読み手は、介入していない同じモデル。返事は <root>/behave/<model_id>/<名前>.json に残し、画面には数だけを出す。
思考ありのモデルでは --thinking で思考ありにして、</think> の後だけを答えとして数える(思考の途中で切れたら空)。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .steer import GneError


def load_prompts(path: str | Path) -> list:
    """JSON / YAML の質問の並び(文字列か会話の履歴)。{"prompts": [...]} の形も読む。対照集合なら A 側の質問
    (prompts_a、無ければ user_prompts)。"""
    import yaml
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8")) if p.suffix == ".json" else yaml.safe_load(p.read_text(encoding="utf-8"))
    if isinstance(d, dict):
        d = d.get("prompts") or d.get("prompts_a") or d.get("user_prompts")
    if not isinstance(d, list) or not d:
        raise GneError("BAD_ARGS", f"{p}: 質問の並びが読めません(文字列か会話の履歴のリスト)")
    return d


def loop_like(ids: list[int]) -> bool:
    """4 トークンの並びの種類が半分を切る返事(同じことを繰り返している)。"""
    if len(ids) < 32:
        return False
    g = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return len(set(g)) / len(g) < 0.5


def answer_of(text: str, thinking: bool) -> str:
    if not thinking:
        return text or ""
    return text.split("</think>", 1)[1] if "</think>" in (text or "") else ""


def behave(session, prompts: list, *, lora=None, vectors=None, match: str | None = None, judge: str | None = None,
           labels: str = "R,P,C", max_tokens: int = 400, thinking: bool = False, temperature: float = 0.0,
           seeds: list[int] | None = None, judge_tokens: int = 8, progress=None) -> dict:
    from .engine import Sampling
    from .evalx import _generate
    rx = re.compile(match) if match else None
    labs = [x.strip() for x in labels.split(",") if x.strip()]
    if judge and ("{q}" not in judge or "{a}" not in judge):
        raise GneError("BAD_ARGS", "判定の文面には {q}(質問)と {a}(返事)が要ります")
    seeds = seeds or [0]
    # 進み具合: 種ごとに質問の数だけ(判定があればその分も)進む。ジョブの取り消しは progress が例外を投げて止める
    total = len(seeds) * len(prompts) * (2 if judge else 1)
    done = 0

    def tick(k: int, stage: str) -> None:
        nonlocal done
        done += k
        if progress:
            progress(min(done, total), total, stage)
    if progress:
        progress(0, total, "behave")
    runs = []
    for seed in seeds:
        sp = Sampling(temperature=temperature, seed=seed, presence_penalty=0.0, min_p=0.0) if temperature > 0 else None
        t0 = time.perf_counter()
        stage = f"behave: 種 {seed}" if len(seeds) > 1 else "behave"
        outs = _generate(session, [(q, max_tokens) for q in prompts], vectors, lora, "all", lambda k, st=stage: tick(k, st),
                         sampling=sp, thinking=thinking)
        replies = [{"i": i, "text": answer_of(o["text"], thinking), "full": o["text"] if thinking else None,
                    "finish": o["finish"], "n_tokens": len(o["token_ids"]), "loop": loop_like(o["token_ids"])} for i, o in enumerate(outs)]
        counts = {"n": len(replies), "empty": sum(not r["text"].strip() for r in replies), "loop": sum(r["loop"] for r in replies),
                  "at_limit": sum(r["finish"] not in ("eog", "stop", "eos") for r in replies),
                  "tokens": round(sum(r["n_tokens"] for r in replies) / max(len(replies), 1), 1)}
        if rx:
            counts["match"] = sum(bool(rx.search(r["text"])) for r in replies)
        if judge:
            idx = [k for k, r in enumerate(replies) if r["text"].strip()]
            items = [(judge.replace("{q}", q if isinstance(q, str) else json.dumps(q, ensure_ascii=False))
                      .replace("{a}", replies[k]["text"][:6000]), judge_tokens) for k in idx for q in [prompts[replies[k]["i"]]]]
            got = _generate(session, items, None, None, "all", lambda k, st=stage: tick(k, f"{st}(判定)"))   # 読み手は介入なし・思考なし
            tick(len(replies) - len(idx), f"{stage}(判定)")                       # 空の返事は判定しない
            pat = re.compile(r"\b(" + "|".join(map(re.escape, labs)) + r")\b")
            for r in replies:
                r["label"] = "N"
            for k, o in zip(idx, got, strict=True):
                m = pat.findall(o["text"] or "")
                replies[k]["label"] = m[-1] if m else "?"
            counts["labels"] = {x: sum(r["label"] == x for r in replies) for x in labs + ["N", "?"]}
        runs.append({"seed": seed, "counts": counts, "replies": replies, "seconds": time.perf_counter() - t0})
    return {"runs": runs, "n_prompts": len(prompts), "max_tokens": max_tokens, "thinking": thinking, "temperature": temperature,
            "match": match, "judge": bool(judge), "labels": labs}
