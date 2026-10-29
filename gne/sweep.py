"""総当たり(rollouts)の実行部。設計書 §6.7。

Phase 2 では reverse --verify(候補に quick 設計の rollout を回す)から使う。順位づけ(rank.py)と
`gne sweep` コマンドは Phase 3。

- ユニット列 = 対象ニューロン + ランダム対照(全体に均等に散らす。中断しても対照の割合が保たれる)。
- ユニットごとに、同じ dose のプロンプト群を 1 回の set_cvec で 1 バッチに生成する。
- ベースライン(無介入)は、介入と同じバッチ構成(同じプロンプトの組)で生成する(F2)。
- 再開: 出力 JSONL の (kind, neuron, prompt, dose) を読み、済みはスキップ。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .engine import Engine
from .steer import GneError, Unit, unit_vector
from .store import CONFIGS

Progress = Callable[[int, int, str], None]


@dataclass
class Design:
    name: str
    pairs: list[tuple[str, float]]       # (プロンプトのキー, dose)
    prompts: dict[str, str]              # キー → 本文

    def groups(self) -> list[tuple[float, list[str]]]:
        """dose ごとのプロンプトの組(最初に出た順)。"""
        out: dict[float, list[str]] = {}
        for key, dose in self.pairs:
            out.setdefault(float(dose), []).append(key)
        return list(out.items())


def load_design(name: str, *, prompt_set: str = "en", designs_path: str | Path | None = None,
                prompts_path: str | Path | None = None) -> Design:
    dd = yaml.safe_load(Path(designs_path or CONFIGS / "designs.yaml").read_text(encoding="utf-8"))["designs"]
    pp = yaml.safe_load(Path(prompts_path or CONFIGS / "prompts" / "default.yaml").read_text(encoding="utf-8"))
    if name not in dd:
        raise GneError("BAD_ARGS", f"設計 {name!r} がありません({', '.join(dd)})")
    if prompt_set not in pp:
        raise GneError("BAD_ARGS", f"プロンプト集 {prompt_set!r} がありません({', '.join(pp)})")
    d = dd[name]
    if "pairs" in d:
        pairs = [(str(k), float(v)) for k, v in d["pairs"]]
    else:
        pairs = [(str(k), float(v)) for v in d["doses"] for k in d["prompts"]]
    prompts = pp[prompt_set]
    missing = sorted({k for k, _ in pairs} - set(prompts))
    if missing:
        raise GneError("BAD_ARGS", f"プロンプト集 {prompt_set!r} に {missing} がありません")
    return Design(name, pairs, {k: prompts[k] for k, _ in pairs})


def unit_schedule(neurons: Sequence[int], n_controls: int, controls: Sequence[int] | None = None) -> list[tuple[str, int]]:
    """[(kind, id)]。対照を全体に均等に散らす。controls は回す対照の番号(既定は 0..n_controls−1、ワーカーで分けるとき)。"""
    cs = list(range(n_controls)) if controls is None else [int(x) for x in controls]
    n_c = len(cs)
    total = len(neurons) + n_c
    slots = set(int((j + 0.5) * total / n_c) for j in range(n_c)) if n_c else set()
    out, it, j = [], iter(neurons), 0
    for i in range(total):
        if i in slots and j < n_c:
            out.append(("random", cs[j]))
            j += 1
        else:
            nxt = next(it, None)
            if nxt is None:
                out.append(("random", cs[j]))
                j += 1
            else:
                out.append(("neuron", int(nxt)))
    return out


def control_seed(seed: int, j: int) -> int:
    return seed * 100003 + j


def _done_keys(path: Path) -> set[tuple]:
    keys = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue                                 # 中断で途中まで書かれた最後の行
                if r["kind"] == "baseline":
                    keys.add(("baseline", tuple(r["batch"]), r["prompt"]))
                else:
                    keys.add((r["kind"], r["neuron"], r["prompt"], round(float(r["dose"]), 6)))
    return keys


def run_sweep(engine: Engine, weights, stats, *, L: int, neurons: Sequence[int], design: Design, out: str | Path,
              model_id: str, dose_unit: str = "rel", random_controls: int = 64, max_tokens: int = 128,
              top_k: int = 5, resume: bool = True, seed: int = 0, signs: dict[int, int] | None = None,
              progress: Progress | None = None, controls: Sequence[int] | None = None) -> dict:
    """層 L の neurons(+ 対照)を design で押して生成し、JSONL に 1 rollout 1 行で追記する。
    signs[n] = -1 のニューロンは dose の符号を反転して押す(逆引きで cos が負の候補)。"""
    if L == 0:
        raise GneError("BAD_ARGS", "層 0 は control vector で押せません")
    if random_controls and dose_unit != "rel":
        raise GneError("BAD_ARGS", "ランダム対照は dose_unit=rel でだけ使えます")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rho = stats.rho(L)
    meta = {"model_id": model_id, "layer": L, "design": design.name, "pairs": design.pairs, "prompts": design.prompts,
            "dose_unit": dose_unit, "resid_p50": round(rho, 6), "max_tokens": max_tokens, "top_k": top_k,
            "random_controls": random_controls, "seed": seed, "llama_cpp_version": engine.llama_cpp_version()}
    meta_path = out.with_name(out.name.removesuffix(".jsonl") + ".meta.json")
    if resume and meta_path.exists():
        old = json.loads(meta_path.read_text())
        diff = [k for k in meta if k not in ("llama_cpp_version",) and json.dumps(old.get(k)) != json.dumps(meta[k])]
        if diff:
            raise GneError("RUN_MISMATCH", f"{out.name} は別の条件で作られています({diff})", "--run で別の名前にしてください")
    elif not resume and out.exists():
        out.unlink()
    meta["prompt_ids"] = {k: engine.tokenize_prompt(engine.render_chat([{"role": "user", "content": t}]))
                          for k, t in design.prompts.items()}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1))

    done = _done_keys(out) if resume else set()
    ctx = engine.steer_ctx()
    groups = design.groups()
    sched = unit_schedule(list(neurons), random_controls, controls)
    signs = signs or {}
    norms = weights.down_norms(L)
    t0, n_new = time.monotonic(), 0

    def gen(keys: list[str]):
        return engine.generate_greedy(ctx, [meta["prompt_ids"][k] for k in keys], max_tokens=max_tokens, top_k=top_k)

    def record(kind, neuron, ref, key, dose, rel_push, add, r, batch, sd):
        return {"model_id": model_id, "layer": L, "kind": kind, "neuron": neuron, "ref_neuron": ref, "prompt": key,
                "dose": dose, "dose_unit": dose_unit, "rel_push": rel_push, "add": add, "text": r.text,
                "finish": r.finish, "n_gen": r.n_gen, "token_ids": r.token_ids, "logprob": r.logprob,
                "top_ids": r.top_ids, "top_lps": r.top_lps, "seed": sd, "batch": batch}

    with open(out, "a", encoding="utf-8") as f:
        def write(rec):
            nonlocal n_new
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_new += 1

        try:
            engine.set_cvec(ctx, None)
            for batch in dict.fromkeys(tuple(keys) for _, keys in groups):   # ベースライン(バッチ構成ごと、順序つき)
                if all(("baseline", batch, k) in done for k in batch):
                    continue
                for k, r in zip(batch, gen(list(batch)), strict=True):
                    write(record("baseline", None, None, k, 0.0, 0.0, 0.0, r, list(batch), seed))
            f.flush()
            for i, (kind, uid) in enumerate(sched):
                sgn = signs.get(uid, 1) if kind == "neuron" else 1
                for dose0, keys in groups:
                    dose = round(dose0 * sgn, 6)
                    if all((kind, uid, k, dose) in done for k in keys):
                        continue
                    if kind == "neuron":
                        unit, sd = Unit(L, uid, dose), seed
                    else:
                        sd = control_seed(seed, uid)
                        unit = Unit(L, None, dose, "random", sd)
                    v = unit_vector(weights, stats, unit, dose_unit)
                    nrm = float(np.linalg.norm(v))
                    add = float(np.sign(dose)) * nrm / float(norms[uid]) if kind == "neuron" and norms[uid] > 0 else None
                    engine.set_cvec(ctx, {L: v})
                    for k, r in zip(keys, gen(keys), strict=True):
                        write(record(kind, uid, uid if kind == "neuron" else None, k, dose, nrm / rho, add, r,
                                     keys, sd))
                    f.flush()
                if progress:
                    progress(i + 1, len(sched), f"sweep L{L}")
        finally:
            engine.set_cvec(ctx, None)
    return {"path": str(out), "meta": str(meta_path), "layer": L, "n_units": len(sched), "n_new_records": n_new,
            "seconds": time.monotonic() - t0}


def part_paths(path: str | Path) -> list[Path]:
    """run の JSONL と、ワーカーごとの部分(<run>.part-<name>.jsonl、CHANGE-03 B)。"""
    p = Path(path)
    stem = p.name.removesuffix(".jsonl")
    parts = sorted(p.parent.glob(f"{stem}.part-*.jsonl"))
    return ([p] if p.exists() else []) + [q for q in parts if not q.name.endswith(".units.jsonl")]


def read_rollouts(path: str | Path, *, parts: bool = True) -> list[dict]:
    """run の rollout。parts なら、ワーカーごとの部分も結合して読む(ベースラインは重なっても語の和集合なので影響しない)。"""
    out = []
    for q in (part_paths(path) if parts else [Path(path)]):
        with open(q, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out
