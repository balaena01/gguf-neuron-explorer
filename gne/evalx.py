"""副作用チェック: レシピの評価(設計書 §6.14)。

レシピは「性格が変わった」だけでは配れない。どれだけ変わったか(persona)と、どれだけ壊れたか(exact / format / ppl / mix)を、
レシピ全体の倍率(dose)ごとに同じ表に並べる。倍率 0 が無介入(ベースライン)。

- exact: 正解との一致(答えの語が出力の最初の行に含まれるか)
- format: 形式の検査(JSON が取り出せる、行数、答えの言語)
- ppl: 参照テキストの teacher-forced な平均 logprob から求めた perplexity(レシピは全位置に当てる)。テキストをまたいだ値は
  幾何平均なので、悪化率は「テキストごとの ppl の比」の幾何平均になる
- mix: 日本語の自由回答のうち、簡体字固有の文字と、ハングル・キリル文字などの割合
- persona: sweep と同じプロンプトで生成し、無介入との語彙の差(新しく出た語の割合と、複数のプロンプトに出た語の数)、
  distinct-4、長さ、max_tokens で打ち切られた割合
- judge(任意): 別の GGUF モデルに「目標の性格が出ているか 1〜5」を採点させる

生成は greedy。when = all のときは同じ倍率のタスクをまとめて生成し(n_seq_max 本ずつ)、それ以外は 1 本ずつ generate_when。
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .rank import distinct4, rollout_words
from .steer import GneError, Recipe

CONFIG = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_SUITE = CONFIG / "eval" / "default.yaml"
PROMPTS = CONFIG / "prompts" / "default.yaml"

_OTHER_SCRIPTS = [(0x0400, 0x052F), (0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF),   # キリル、ハングル
                  (0x0E00, 0x0E7F), (0x0600, 0x06FF), (0x0900, 0x097F), (0x0590, 0x05FF)]   # タイ、アラビア、デーヴァナーガリー、ヘブライ
_KANA = re.compile(r"[぀-ヿ]")
_MD = re.compile(r"[*_`#>]+")


@dataclass
class Suite:
    name: str
    path: Path
    max_tokens: dict
    thresholds: dict
    exact: list[dict]
    format: list[dict]
    mix: list[dict]
    persona: list[dict]
    ppl: list[dict]
    zh_only: set[str] = field(default_factory=set)

    @property
    def n_tasks(self) -> int:
        return len(self.exact) + len(self.format) + len(self.mix) + len(self.persona) + len(self.ppl)


def load_suite(path: str | Path | None = None) -> Suite:
    p = Path(path) if path else DEFAULT_SUITE
    if not p.exists():
        raise GneError("FILE_NOT_FOUND", f"タスク集 {p} がありません")
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    prompts = yaml.safe_load(PROMPTS.read_text(encoding="utf-8"))
    persona = [{"id": f"{ps}.{k}", "prompt": v} for ps in (d.get("persona") or {}).get("prompt_sets", ["en"])
               for k, v in prompts[ps].items()]
    zh_file = p.parent / "zh_only_chars.txt"
    zh = set()
    if zh_file.exists():
        zh = {c for line in zh_file.read_text(encoding="utf-8").splitlines() if not line.startswith("#") for c in line.strip()}
    ppl = [{**t, "text": (p.parent / t["path"]).read_text(encoding="utf-8")} for t in d.get("ppl", [])]
    mt = {"exact": 32, "format": 96, "mix": 96, "persona": 96, **(d.get("max_tokens") or {})}
    th = {"capability_drop_pt": 5.0, "ppl_worse_pct": 10.0, "mix_increase": 0.0, **(d.get("thresholds") or {})}
    return Suite(d.get("name") or p.stem, p, mt, th, d.get("exact", []), d.get("format", []), d.get("mix", []), persona, ppl, zh)


# --- 採点 ---------------------------------------------------------------------------------------------

def _norm(s: str) -> str:
    return _MD.sub("", s).casefold().strip()


def first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def score_exact(text: str, answers: list) -> bool:
    """答え(別解のどれか)が出力の最初の行に含まれるか。英数字の答えは語の境界で探す。"""
    line = _norm(first_line(text))
    for a in answers:
        a = _norm(str(a))
        if re.fullmatch(r"[\w.+\-₀-₉]+", a, re.ASCII) or a.isascii():
            if re.search(rf"(?<![\w.]){re.escape(a)}(?![\w])", line):
                return True
        elif a in line:
            return True
    return False


def _strip_fence(text: str) -> str:
    t = text.strip()
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", t, re.DOTALL)
    return m.group(1).strip() if m else t


def score_format(text: str, task: dict) -> bool:
    check = task["check"]
    t = _strip_fence(text)
    if check in ("json", "json_array"):
        o, c = ("{", "}") if check == "json" else ("[", "]")
        a, b = t.find(o), t.rfind(c)
        if a < 0 or b <= a:
            return False
        try:
            v = json.loads(t[a:b + 1])
        except ValueError:
            return False
        if check == "json":
            return isinstance(v, dict) and all(k in v for k in task.get("keys", []))
        return isinstance(v, list) and len(v) == int(task.get("n", len(v)))
    if check.startswith("lines:"):
        return len([ln for ln in t.splitlines() if ln.strip()]) == int(check.split(":")[1])
    if check.startswith("lang:"):
        letters = [ch for ch in t if ch.isalpha()]
        if not letters:
            return False
        if check == "lang:en":
            return not _KANA.search(t) and sum(ch.isascii() for ch in letters) / len(letters) >= 0.9
        return sum(not ch.isascii() for ch in letters) / len(letters) >= 0.5 and bool(_KANA.search(t))
    raise GneError("BAD_SUITE", f"format の check={check!r} は分かりません")


def mix_ratio(text: str, zh_only: set[str]) -> tuple[float, float]:
    """(簡体字固有の文字の割合, ハングル・キリル文字などの割合)。分母は空白・記号・数字を除いた文字。"""
    chars = [ch for ch in text if ch.isalpha()]
    if not chars:
        return 0.0, 0.0
    zh = sum(ch in zh_only for ch in chars)
    other = sum(any(a <= ord(ch) <= b for a, b in _OTHER_SCRIPTS) for ch in chars)
    return zh / len(chars), other / len(chars)


def persona_metrics(base: list[dict], outs: list[dict], token_words) -> dict:
    """同じプロンプトの無介入の出力と比べた語彙の差。

    novelty: 各プロンプトで、出力の語のうち無介入の出力に無い語の割合(の平均)
    mass: 新しく出た語のうち 2 つ以上のプロンプトに出たもの、上位 5 語の出たプロンプト数の和(複数の話題で同じ方向に寄った量)。
    rank.py の mass は対照群の平均に対する z だが、eval は 1 レシピなので数そのものを使う(docs/divergence.md)"""
    nov, counts = [], Counter()
    for b, o in zip(base, outs, strict=True):
        wb, wo = rollout_words(b, "tokens", token_words), rollout_words(o, "tokens", token_words)
        new = wo - wb
        nov.append(len(new) / max(1, len(wo)))
        counts.update(new)
    multi = [(w, c) for w, c in counts.most_common() if c >= 2]
    return {"novelty": float(np.mean(nov)) if nov else 0.0, "mass": float(sum(c for _, c in multi[:5])),
            "top_words": [[w, c] for w, c in multi[:10]],
            "distinct4": float(np.mean([distinct4(o["token_ids"]) for o in outs])) if outs else 1.0,
            "length": float(np.mean([len(o["token_ids"]) for o in outs])) if outs else 0.0,
            "capped": float(np.mean([o["finish"] == "length" for o in outs])) if outs else 0.0}


# --- 生成 ---------------------------------------------------------------------------------------------

def _msgs(p: str | list[dict]) -> list[dict]:
    return [{"role": "user", "content": p}] if isinstance(p, str) else p


def _generate(session, items: list[tuple[str | list[dict], int]], vecs, lora, when: str,
              tick: Callable[[int], None]) -> list[dict]:
    """items: [(プロンプトか会話, max_tokens)]。when = all ならまとめて(n_seq_max 本ずつ、n_ctx に収まる分)、それ以外は 1 本ずつ。"""
    from .when import generate_when
    e = session.engine
    ctx = e.steer_ctx()
    outs: list[dict | None] = [None] * len(items)

    def keep(r):                       # token_ids は EOG まで含めたまま(rank.rollout_words と同じ形)
        return {"text": r.text, "token_ids": r.token_ids, "finish": r.finish}
    if when != "all":
        for i, (prompt, mt) in enumerate(items):
            r, _ = generate_when(e, ctx, _msgs(prompt), vecs, when, max_tokens=mt, lora=lora)
            outs[i] = keep(r)
            tick(1)
        return outs
    e.set_lora(ctx, lora)
    try:
        e.set_cvec(ctx, vecs or None)
        seqs = [e.tokenize_prompt(e.render_chat(_msgs(p))) for p, _ in items]
        by_mt: dict[int, list[int]] = {}
        for i, (_, mt) in enumerate(items):
            by_mt.setdefault(mt, []).append(i)
        for mt, idx in by_mt.items():
            while idx:
                part, used = [], 0
                for i in idx:
                    need = len(seqs[i]) + mt
                    if part and (len(part) == e.n_seq_max or used + need > e.n_ctx):
                        break
                    part.append(i)
                    used += need
                idx = idx[len(part):]
                if used > e.n_ctx:
                    raise GneError("CONTEXT_FULL", f"タスク {part[0]} が n_ctx={e.n_ctx} に入りません")
                for i, r in zip(part, e.generate_greedy(ctx, [seqs[i] for i in part], max_tokens=mt), strict=True):
                    outs[i] = keep(r)
                tick(len(part))
    finally:
        e.set_cvec(ctx, None)
        e.set_lora(ctx, None)
    return outs


def _ppl(session, texts: list[dict], vecs, lora, tick: Callable[[int], None]) -> dict[str, float]:
    """チャンクの大きさで数値がわずかに変わる(verification §11.3)ので、どの倍率でも同じ 512(steer ctx の n_ubatch)で測る。"""
    e = session.engine
    ctx = e.steer_ctx()
    out = {}
    e.set_lora(ctx, lora)
    try:
        e.set_cvec(ctx, vecs or None)
        for t in texts:
            ids = e.tokenize(t["text"], add_special=True, parse_special=False)[:e.n_ctx]
            lp = e.token_logprobs(ctx, ids, chunk=512)
            out[t["id"]] = float(np.exp(-lp.mean()))
            tick(1)
    finally:
        e.set_cvec(ctx, None)
        e.set_lora(ctx, None)
    return out


class Judge:
    """別の GGUF モデルに「目標の性格が出ているか 1〜5」を採点させる(任意)。"""

    PROMPT = ("Target persona: {target}\n\nResponse to rate:\n\"\"\"\n{text}\n\"\"\"\n\n"
              "How strongly does the response show the target persona? Reply with a single digit from 1 to 5 "
              "(1 = not at all, 5 = very strongly). Reply with the digit only.")

    def __init__(self, model: str | Path, target: str, *, n_gpu_layers: int = -1):
        from .engine import Engine
        self.e = Engine(model, n_gpu_layers=n_gpu_layers, n_ctx=4096, n_seq_max=1)
        self.target = target

    def score(self, text: str) -> int | None:
        msgs = [{"role": "user", "content": self.PROMPT.format(target=self.target, text=text[:2000])}]
        ctx = self.e.steer_ctx()
        r = self.e.generate_greedy(ctx, [self.e.tokenize_prompt(self.e.render_chat(msgs))], max_tokens=6)[0]
        m = re.search(r"[1-5]", r.text)
        return int(m.group(0)) if m else None

    def close(self) -> None:
        self.e.close()


# --- 評価の本体 ----------------------------------------------------------------------------------------

def evaluate(session, recipe: Recipe, *, doses=(0, 0.25, 0.5, 0.75, 1.0), suite: str | Path | None = None,
             when: str | None = None, judge: Judge | None = None,
             progress: Callable[[int, int, str], None] | None = None) -> dict:
    """dose(レシピ全体の倍率)ごとに全タスクを流し、指標の表を返す。0 が無ければ足す(ベースライン、最初に測る)。
    倍率 0 も同じ when の経路で生成する(バッチの組み方で数値が変わるので、比べる行どうしで経路をそろえる)。
    負の倍率(反対向き)も測れる。表は倍率の小さい順。"""
    from .when import parse_when
    st = load_suite(suite)
    doses = sorted({float(d) for d in doses} | {0.0})
    order = [0.0] + [d for d in doses if d != 0]
    w = str(parse_when(when or recipe.when))
    tok = session.vocab.words()
    items = ([("exact", t, st.max_tokens["exact"]) for t in st.exact] + [("format", t, st.max_tokens["format"]) for t in st.format]
             + [("mix", t, st.max_tokens["mix"]) for t in st.mix] + [("persona", t, st.max_tokens["persona"]) for t in st.persona])
    t0 = time.perf_counter()
    rows, samples, base_persona = [], {}, None
    total, done = len(doses) * (len(items) + len(st.ppl)), 0

    def tick(n: int, d: float = 0.0):
        nonlocal done
        done += n
        if progress:
            progress(done, total, f"eval dose {d:g}")
    for d in order:
        r = recipe.scaled(d) if d != 0 else None
        vecs = session.vectors(r)[0] if r is not None else None
        lora = session.lora_for(r) if r is not None else None
        outs = _generate(session, [(t["prompt"], mt) for _, t, mt in items], vecs, lora, w, lambda n, d=d: tick(n, d))
        ppl = _ppl(session, st.ppl, vecs, lora, lambda n, d=d: tick(n, d)) if st.ppl else {}
        by = {"exact": [], "format": [], "mix": [], "persona": []}
        for (cat, t, _), o in zip(items, outs, strict=True):
            by[cat].append((t, o))
        ex = [score_exact(o["text"], t["answer"]) for t, o in by["exact"]]
        fm = [score_format(o["text"], t) for t, o in by["format"]]
        mixes = [mix_ratio(o["text"], st.zh_only) for _, o in by["mix"]]
        pers_out = [o for _, o in by["persona"]]
        if d == 0:
            base_persona = pers_out
        pm = persona_metrics(base_persona, pers_out, tok)
        row = {"dose": d, "exact": float(np.mean(ex)) if ex else None, "format": float(np.mean(fm)) if fm else None,
               "capability": float(np.mean(ex + fm)) if ex + fm else None,
               "ppl": ppl, "ppl_mean": float(np.exp(np.mean(np.log(list(ppl.values()))))) if ppl else None,   # 幾何平均
               "mix_zh": float(np.mean([a for a, _ in mixes])) if mixes else 0.0,
               "mix_other": float(np.mean([b for _, b in mixes])) if mixes else 0.0,
               "persona": pm,
               "failed": {"exact": [t["id"] for (t, _), ok in zip(by["exact"], ex, strict=True) if not ok],
                          "format": [t["id"] for (t, _), ok in zip(by["format"], fm, strict=True) if not ok]}}
        if judge is not None:
            scores = [judge.score(o["text"]) for o in pers_out]
            vals = [s for s in scores if s is not None]
            row["judge"] = float(np.mean(vals)) if vals else None
            row["judge_scores"] = scores
        rows.append(row)
        samples[f"{d:g}"] = {"persona": [{"id": t["id"], "text": o["text"][:400]} for t, o in by["persona"]],
                             "mix": [{"id": t["id"], "text": o["text"][:200]} for t, o in by["mix"][:3]],
                             "failed": [{"id": t["id"], "text": o["text"][:200]} for (t, o), ok in
                                        zip(by["exact"] + by["format"], ex + fm, strict=True) if not ok][:6]}
    base = rows[0]
    rows.sort(key=lambda x: x["dose"])
    th = st.thresholds
    for row in rows:
        drop = (base["capability"] - row["capability"]) * 100 if base["capability"] is not None else 0.0
        worse = ((row["ppl_mean"] / base["ppl_mean"] - 1) * 100) if base["ppl_mean"] else 0.0
        mix_up = (row["mix_zh"] + row["mix_other"]) - (base["mix_zh"] + base["mix_other"])
        row["capability_drop_pt"] = drop
        row["ppl_worse_pct"] = worse
        row["mix_increase"] = mix_up
        row["ok"] = {"capability": drop <= th["capability_drop_pt"], "ppl": worse <= th["ppl_worse_pct"],
                     "mix": mix_up <= th["mix_increase"] + 1e-9}
        row["shippable"] = all(row["ok"].values())
    return {"recipe": recipe.to_dict(), "when": w, "suite": {"name": st.name, "path": str(st.path), "n_tasks": st.n_tasks,
            "thresholds": th}, "doses": doses, "rows": rows, "samples": samples,
            "judge": {"target": judge.target, "model": str(judge.e.path)} if judge else None,
            "seconds": time.perf_counter() - t0}
