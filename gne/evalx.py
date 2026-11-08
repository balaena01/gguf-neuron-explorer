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

import contextlib

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
              tick: Callable[[int], None], sampling=None, thinking: bool = False) -> list[dict]:
    """items: [(プロンプトか会話, max_tokens)]。when = all ならまとめて(n_seq_max 本ずつ、n_ctx に収まる分)、それ以外は 1 本ずつ。
    sampling(engine.Sampling)を渡すとサンプリングで生成する。乱数は「seed + まとまりの番号」と、まとまりの中の位置で決まるので、
    同じ items を同じ順に渡せば(押し方を変えても)同じ乱数で引く。既定は greedy。"""
    import dataclasses
    from .when import generate_when
    e = session.engine
    ctx = e.steer_ctx()
    outs: list[dict | None] = [None] * len(items)

    def keep(r):                       # token_ids は EOG まで含めたまま(rank.rollout_words と同じ形)
        return {"text": r.text, "token_ids": r.token_ids, "finish": r.finish}
    if when != "all":
        for i, (prompt, mt) in enumerate(items):
            sp = None if sampling is None else dataclasses.replace(sampling, seed=sampling.seed + 7919 * i)
            r, _ = generate_when(e, ctx, _msgs(prompt), vecs, when, max_tokens=mt, lora=lora, sampling=sp)
            outs[i] = keep(r)
            tick(1)
        return outs
    e.set_lora(ctx, lora)
    try:
        e.set_cvec(ctx, vecs or None)
        seqs = [e.tokenize_prompt(e.render_chat(_msgs(p), enable_thinking=thinking)) for p, _ in items]
        by_mt: dict[int, list[int]] = {}
        for i, (_, mt) in enumerate(items):
            by_mt.setdefault(mt, []).append(i)
        n_batch = 0
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
                sp = None if sampling is None else dataclasses.replace(sampling, seed=sampling.seed + 7919 * n_batch)
                n_batch += 1
                for i, r in zip(part, e.generate_greedy(ctx, [seqs[i] for i in part], max_tokens=mt, sampling=sp), strict=True):
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


# --- 反応性(CHANGE-04 F) ------------------------------------------------------------------------------

@dataclass
class Reactivity:
    """対照集合の reactivity(トリガーあり/なしの対)と validate.match(性格の判定)。"""
    name: str
    path: Path
    pairs: list[tuple[str, str]]
    match: str

    @classmethod
    def load(cls, path: str | Path) -> Reactivity:
        p = Path(path)
        if not p.exists():
            raise GneError("FILE_NOT_FOUND", f"対照集合 {p} がありません")
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        pairs = [(str(x["with"]), str(x["without"])) for x in (d.get("reactivity") or []) if x.get("with") and x.get("without")]
        match = (d.get("validate") or {}).get("match")
        if not pairs or not match:
            raise GneError("BAD_CONTRAST", f"{p}: 反応性には reactivity(with / without の対)と validate.match が要ります")
        try:
            re.compile(match)
        except re.error as e:
            raise GneError("BAD_CONTRAST", f"validate.match の正規表現が読めません: {e}") from e
        return cls(d.get("name") or p.stem, p, pairs, match)


def reactivity_score(session, rc: Reactivity, recipe: Recipe | None, when: str, max_tokens: int,
                     tick: Callable[[int], None], judge: Judge | None = None) -> dict:
    """トリガーあり/なしの各プロンプト(system なし)に返事をさせ、判定の一致率(と judge)の差 = 反応性。"""
    rx = re.compile(rc.match, re.IGNORECASE)
    vecs = session.vectors(recipe)[0] if recipe is not None and recipe.add_units else None
    lora = session.lora_for(recipe) if recipe is not None else None
    items = [(p, max_tokens) for pair in rc.pairs for p in pair]
    outs = _generate(session, items, vecs, lora, when, tick)
    hit = [bool(rx.search(o["text"])) for o in outs]
    w, wo = float(np.mean(hit[0::2])), float(np.mean(hit[1::2]))
    d = {"with": w, "without": wo, "reactivity": w - wo, "n_pairs": len(rc.pairs),
         "samples": [{"with": outs[2 * i]["text"][:200], "without": outs[2 * i + 1]["text"][:200]} for i in range(min(3, len(rc.pairs)))]}
    if judge is not None:
        sc = [judge.score(o["text"]) for o in outs]
        jw = [x for x in sc[0::2] if x is not None]
        jo = [x for x in sc[1::2] if x is not None]
        d["judge_with"] = float(np.mean(jw)) if jw else None
        d["judge_without"] = float(np.mean(jo)) if jo else None
        d["judge_reactivity"] = (d["judge_with"] - d["judge_without"]) if jw and jo else None
    return d


# --- 方向を使うレシピの指標(CHANGE-06 F.1)-----------------------------------------------------------------

def direction_of(recipe: Recipe | None) -> str | None:
    """レシピが使う方向の名前(proj・keyed・方向のユニット・ablate の順)。ランダム対照の消去(random-SEED)は除く。
    keyed(CHANGE-07)は書き込む向きの方向(打ち消しの - は α の符号に畳んであるので名前だけ)。"""
    if recipe is None:
        return None
    for u in recipe.units:
        if u.mode == "proj":
            return u.direction or recipe.default_direction
    for u in recipe.units:
        if u.mode == "keyed" and (u.direction or recipe.default_direction):
            return str(u.direction or recipe.default_direction).lstrip("-")
    for u in recipe.units:
        if u.kind == "direction" and u.direction:
            return u.direction
    if recipe.ablate is not None and not recipe.ablate.direction.startswith("random-"):
        return recipe.ablate.direction
    return None


class DirMetrics:
    """方向を使うレシピの、速い指標(射影の値)の変化と残差の差の方向成分・直交成分(drift_par / drift_orth、A 側・B 側)。
    外から見える指標として、対照集合に validate.match があれば A 側(system_a 付き)の判定の一致率も測る(behavior)。
    射影の値は基準層と最も後ろの層で測り、行に平らに出す値(drift_*・dir_drop_a)は最も後ろの層のもの。"""

    def __init__(self, session, name: str, *, contrast: str | Path | None = None, layers: list[int] | None = None,
                 behavior: bool = True, max_tokens: int = 64, max_examples: int | None = 64):
        from .direction import Scorer, base_layer_of, load_contrast, score_layers_default
        from .direction import score as score_fn
        self.session = session
        ds = session.direction_set(name)
        src = contrast or ds.meta.get("source")
        if not src or not Path(src).exists():
            raise GneError("BAD_ARGS", f"方向 {name} を作った対照集合 {src} が見つかりません")
        self.cs, self.val = load_contrast(src)
        if base_layer_of(ds) is None:
            score_fn(session, ds, self.cs)
            ds = session.direction_set(name)
        b = base_layer_of(ds)
        self.layers = sorted(set(layers)) if layers else sorted({b, max(score_layers_default(session, ds, b))})
        self.top = max(self.layers)
        self.name, self.ds = name, ds
        # 質問が多い対照集合(何百問)では、1 回の測定で全部を読むと大きいモデルで何十分もかかるので、等間隔に間引く
        from .direction import thin
        self.max_examples = max_examples
        self.sc = Scorer(session, ds, self.cs, layers=self.layers, max_examples=max_examples)
        self.behavior = behavior and bool(self.val.match)
        self.max_tokens = max_tokens
        self.n_score = self.sc.n("A") + self.sc.n("B")
        self.n_beh = len(thin(list(self.val.prompts), max_examples)) if self.behavior else 0
        self.cost = self.n_score + self.n_beh             # 1 回の測定で読む・作る会話の数(進み具合に数える)
        self.tick: Callable[[int, str], None] = lambda n, label="": None
        self.base = None
        self.base_rate = None

    def prepare(self) -> None:
        """無介入の射影の値と一致率(測る前に 1 回)。"""
        self.base = self.sc.run(keep=True)
        self.tick(self.n_score, "方向の指標(無介入)")
        if self.behavior:
            from .direction import behavior as beh
            self.base_rate = beh(self.session, self.cs, self.val, None, max_tokens=self.max_tokens,
                                 max_examples=self.max_examples, with_kl=False)["rate"]
            self.tick(self.n_beh, "方向の指標(無介入)")

    def measure(self, r: Recipe | None, label: str = "方向の指標") -> dict:
        from .direction import drift
        if r is None:
            got = self.base
        else:
            lora = self.session.lora_for(r)
            vecs = self.session.vectors(r)[0] if r.add_units else None
            got = self.sc.run(lora, vecs, keep=True)
            self.tick(self.n_score, label)
        layers = {}
        for L in self.layers:
            x = {"score_a": float(got["A"]["values"][L].mean()), "score_b": float(got["B"]["values"][L].mean()),
                 "drop_a": float(self.base["A"]["values"][L].mean() - got["A"]["values"][L].mean()),
                 "delta_b": float(got["B"]["values"][L].mean() - self.base["B"]["values"][L].mean())}
            for side in "AB":
                x[f"drift_par_{side.lower()}"], x[f"drift_orth_{side.lower()}"] = drift(
                    self.base[side]["resid"][L], got[side]["resid"][L], self.sc.r[L])
            layers[str(L)] = x
        top = layers[str(self.top)]
        out = {"direction": self.name, "layer": self.top, "layers": layers, "dir_drop_a": top["drop_a"],
               **{k: top[k] for k in ("drift_par_a", "drift_orth_a", "drift_par_b", "drift_orth_b")}}
        if self.behavior:
            from .direction import behavior as beh
            rate = self.base_rate
            if r is not None:
                rate = beh(self.session, self.cs, self.val, r, max_tokens=self.max_tokens, max_examples=self.max_examples,
                           with_kl=False)["rate"]
                self.tick(self.n_beh, label)
            out["rate"], out["rate_drop"] = rate, None if rate is None or self.base_rate is None else self.base_rate - rate
        return out


def control_set(session, recipe: Recipe, n: int, *, pool: str = "before", score_layers: list[int] | None = None) -> list[Recipe]:
    """レシピのランダム対照。add のユニット・ablate は control_recipes(同じ大きさのランダムな向き)、proj のユニットは
    同数のランダムなヘッドに同じ s で(CHANGE-06 F.2)、keyed のユニットは鍵を比較用の鍵 NAME~j に替えて(CHANGE-07 E.2。
    α・方向・層はそのまま)。どれも無ければ空。keyed を含むときの本数は n と鍵の比較用の鍵の本数の小さい方。"""
    from .controls import control_recipes, keyed_control_cap, pseudo_keyed
    if n <= 0:
        return []
    keyed = bool(recipe.keyed_units)
    if keyed:
        n = min(n, keyed_control_cap(session, recipe))
        if n <= 0:
            return []                                # 比較用の鍵の無い鍵(pseudo = 0 で作った鍵)
    if not recipe.proj_units:
        if recipe.add_units or recipe.ablate is not None:
            out = control_recipes(session, recipe, n)
        else:
            out = [recipe.like(recipe.units, name=f"{recipe.name or 'recipe'}-random{j}") for j in range(n)] if keyed else []
    else:
        from .headproj import proj_control_recipes
        base = control_recipes(session, recipe, n) if (recipe.add_units or recipe.ablate is not None) else [recipe] * n
        out = []
        for j, b in enumerate(base):
            c = proj_control_recipes(session, b, n, pool=pool, score_layers=score_layers)[j]
            out.append(c.like(c.units, name=f"{recipe.name or 'recipe'}-random{j}"))
    if keyed:                                        # CHANGE-07 E.2: 鍵だけを比較用の鍵 j に(ほかの対照と同じ j 本目に重ねる)
        out = [c.like(pseudo_keyed(c, j), name=c.name) for j, c in enumerate(out)]
    return out


# --- 評価の本体 ----------------------------------------------------------------------------------------

def evaluate(session, recipe: Recipe, *, doses=(0, 0.25, 0.5, 0.75, 1.0), suite: str | Path | None = None,
             when: str | None = None, judge: Judge | None = None, random_controls: int = 0,
             reactivity: str | Path | None = None, head_controls: int = 0, reactivity_tokens: int = 64,
             control_pool: str = "before", direction_metrics: bool = True, baseline: dict | None = None,
             dir_max_examples: int | None = 64, keyed_contrast: str | Path | None = None, keyed_metrics: bool = True,
             keyed_examples: int | None = 8, keyed_match: bool = True, keyed_pseudo: int = 8,
             keyed_subset: str | None = None, progress: Callable[[int, int, str], None] | None = None) -> dict:
    """dose(レシピ全体の倍率)ごとに全タスクを流し、指標の表を返す。0 が無ければ足す(ベースライン、最初に測る)。
    倍率 0 も同じ when の経路で生成する(バッチの組み方で数値が変わるので、比べる行どうしで経路をそろえる)。
    負の倍率(反対向き)も測れる。表は倍率の小さい順。
    random_controls > 0 なら、0 以外の各倍率で同じ大きさのランダムな対照レシピ(controls.control_recipes)も同じタスクで測り、
    行の ctrl に persona・能力・ppl の分布を入れる(CHANGE-03 A。judge は対照では回さない)。
    reactivity(対照集合の YAML)を渡すと、各倍率で反応性(トリガーあり/なしの判定の一致率の差、CHANGE-04 F)も測る。
    head_controls > 0 なら、ヘッドのユニットを同じ k のランダムなヘッドに替えた対照で反応性を測り、null_pct を出す。
    方向を使うレシピ(proj・方向のユニット・ablate)では、各行に速い指標(射影の値)の変化と直交成分のずれ(drift_par / drift_orth、
    A 側・B 側)を出す(CHANGE-06 F.1)。proj のユニットの対照は同数のランダムなヘッド(control_pool: before / same-layers、F.2)で、
    目的の指標(dir_drop_a、validate.match があれば判定の一致率の減少 rate_drop)と副作用(drift_orth_b)の null_pct を出す。
    baseline(前の evaluate の戻り値の "_baseline")を渡すと、倍率 0 の測定を使い回す(方式の比較・ヘッド数の曲線用)。
    keyed のユニットを持つレシピ(CHANGE-07 E)では、鍵の対照集合(keyed_contrast。省くと方向を作った対照集合が鍵と同じ
    ものならそれ)を鍵の作成と反対側の subset(keyed_subset で変えられる)で prefill だけして(keyed_eval.KeyedMetrics)、
    各行に keyed(書き込みたい位置・書き込みたくない位置・dont_touch での残差の r̂ 方向の変化と ‖Δ‖、eval の選択性)を、
    結果の keyed に方式の比較(keyed・同じ α の add・書き込みたい位置をそろえた add / gain / proj)と比較用の鍵の eval の
    選択性(keyed_pseudo 本)を出す。
    keyed_examples は会話の組ごとに読む数、keyed_match=False で大きさ合わせ(bisection 相当の探索)を省く、
    keyed_metrics=False で測らない。対照集合が分からなければ結果の keyed に skipped(理由)だけを書く。"""
    from .controls import head_control_recipes
    from .controls import null_pct as ctrl_null_pct
    from .controls import summarize as summarize_ctrl
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
    n_ctrl = max(0, int(random_controls))
    dm = None
    dname = direction_of(recipe) if direction_metrics else None
    if dname is not None:
        try:
            dm = DirMetrics(session, dname, max_examples=dir_max_examples)
        except GneError as e:
            session.warn(f"方向 {dname} の射影の値と直交成分のずれは測りません: {e.message}")
    km, kblock = None, None
    if keyed_metrics and recipe.keyed_units:        # CHANGE-07 E: keyed の eval の選択性(prefill だけ)
        from .keyed_eval import KeyedMetrics
        try:
            km = KeyedMetrics(session, recipe, contrast=keyed_contrast, subset=keyed_subset, max_examples=keyed_examples,
                              pseudo=keyed_pseudo, match=keyed_match)
        except GneError as e:
            session.warn(f"keyed の eval の選択性は測りません: {e.message}")
            kblock = {"skipped": e.message, **({"hint": e.hint} if e.hint else {})}
    if n_ctrl and not control_set(session, recipe, 1, pool=control_pool, score_layers=dm.layers if dm else None):
        n_ctrl = 0                                  # gain だけのレシピには向きが無いので対照を作れない
    if n_ctrl and recipe.keyed_units:               # keyed の比較用の鍵の対照は、鍵に保存した比較用の鍵の本数まで
        from .controls import keyed_control_cap
        cap = keyed_control_cap(session, recipe)
        if cap < n_ctrl:
            session.warn(f"random_controls を、鍵に保存した比較用の鍵の本数 {cap} に減らしました(gne direction key の pseudo で増やせます)")
        n_ctrl = min(n_ctrl, cap)
    total, done = (len(doses) + n_ctrl * (len(doses) - 1)) * (len(items) + len(st.ppl)), 0
    rc = Reactivity.load(reactivity) if reactivity else None
    n_hctrl = max(0, int(head_controls)) if rc is not None else 0
    if n_hctrl and not head_control_recipes(session, recipe, 1):
        n_hctrl = 0                                 # ヘッドの無いレシピにはヘッドの対照を作れない
    if rc is not None:
        total += (len(doses) + n_hctrl * (len(doses) - 1)) * 2 * len(rc.pairs)
    n_nonzero = len({float(x) for x in doses} - {0.0})
    if dm is not None:                              # 方向の指標: 無介入 1 回 + 倍率ごと + 対照ごと
        total += dm.cost * (1 + n_nonzero + n_ctrl * n_nonzero)
    if km is not None:                              # keyed: 無介入 1 回 + 倍率ごと + 対照ごと + 方式の比較と比較用の鍵
        total += km.n_full * (1 + n_nonzero + n_ctrl * n_nonzero) + km.block_cost()

    def tick(n: int, d: float = 0.0):
        nonlocal done
        done += n
        if progress:
            progress(done, total, f"eval dose {d:g}")

    def dtick(n: int, label: str):
        nonlocal done
        done += n
        if progress:
            progress(min(done, total), total, label)
    if dm is not None:
        dm.tick = dtick
        dm.prepare()
    if km is not None:
        km.tick = dtick
        try:
            km.prepare()
        except GneError as e:
            session.warn(f"keyed の eval の選択性は測りません: {e.message}")
            kblock = {"skipped": e.message, **({"hint": e.hint} if e.hint else {})}
            km = None

    def measure(r: Recipe | None, d: float):
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
        row = {"dose": d, "exact": float(np.mean(ex)) if ex else None, "format": float(np.mean(fm)) if fm else None,
               "capability": float(np.mean(ex + fm)) if ex + fm else None,
               "ppl": ppl, "ppl_mean": float(np.exp(np.mean(np.log(list(ppl.values()))))) if ppl else None,   # 幾何平均
               "mix_zh": float(np.mean([a for a, _ in mixes])) if mixes else 0.0,
               "mix_other": float(np.mean([b for _, b in mixes])) if mixes else 0.0,
               "failed": {"exact": [t["id"] for (t, _), ok in zip(by["exact"], ex, strict=True) if not ok],
                          "format": [t["id"] for (t, _), ok in zip(by["format"], fm, strict=True) if not ok]}}
        return row, by, ex, fm

    for d in order:
        r = recipe.scaled(d) if d != 0 else None
        ps = session.patches_for(r)                    # CHANGE-05: 専門家全体の gain・route は作業用コピーで読み直して測る
        with session.patched(ps) if ps is not None else contextlib.nullcontext():
            if d == 0 and baseline is not None:
                row, by, ex, fm = (json.loads(json.dumps(baseline["row"])), baseline["by"], baseline["ex"], baseline["fm"])
                tick(len(items) + len(st.ppl), d)
            else:
                row, by, ex, fm = measure(r, d)
            if d == 0:
                base_meas = {"row": json.loads(json.dumps(row)), "by": by, "ex": ex, "fm": fm}
            if dm is not None:
                row["direction"] = dm.measure(r, f"方向の指標 倍率 {d:g}")
                for k in ("dir_drop_a", "drift_par_a", "drift_orth_a", "drift_par_b", "drift_orth_b", "rate", "rate_drop"):
                    if k in row["direction"]:
                        row[k] = row["direction"][k]
            if km is not None:                         # CHANGE-07 E.1: レシピ全体を当てた書き込みたい位置・書き込みたくない位置の残差の変化
                row["keyed"] = km.row(r, f"keyed 倍率 {d:g}")
            pers_out = [o for _, o in by["persona"]]
            if d == 0:
                base_persona = pers_out
            row["persona"] = persona_metrics(base_persona, pers_out, tok)
            if judge is not None:
                scores = [judge.score(o["text"]) for o in pers_out]
                vals = [s for s in scores if s is not None]
                row["judge"] = float(np.mean(vals)) if vals else None
                row["judge_scores"] = scores
            if rc is not None:
                rx = reactivity_score(session, rc, r, w, reactivity_tokens, lambda n, d=d: tick(n, d), judge)
                if n_hctrl and d != 0:
                    cr_ = [reactivity_score(session, rc, c, w, reactivity_tokens, lambda n, d=d: tick(n, d))["reactivity"]
                           for c in head_control_recipes(session, r, n_hctrl)]
                    rx["ctrl"] = {"n": len(cr_), "values": cr_, **summarize_ctrl(cr_)}
                    rx["null_pct"] = ctrl_null_pct(rx["reactivity"], cr_)
                row["reactivity"] = rx
        if n_ctrl and d != 0:
            crow = []
            for j, c in enumerate(control_set(session, r, n_ctrl, pool=control_pool, score_layers=dm.layers if dm else None)):
                pc = session.patches_for(c)                    # 対照も同じ専門家の gain・route を残すので、同じパッチで測る
                with session.patched(pc) if pc is not None else contextlib.nullcontext():
                    cr, cby, _, _ = measure(c, d)
                    cdm = dm.measure(c, f"方向の指標 倍率 {d:g} 対照 {j + 1}/{n_ctrl}") if dm is not None else None
                    ckm = km.row(c, f"keyed 倍率 {d:g} 対照 {j + 1}/{n_ctrl}") if km is not None else None
                pm = persona_metrics(base_persona, [o for _, o in cby["persona"]], tok)
                crow.append({"name": c.name, "capability": cr["capability"], "ppl_mean": cr["ppl_mean"],
                             "mix": cr["mix_zh"] + cr["mix_other"], "novelty": pm["novelty"], "mass": pm["mass"],
                             "distinct4": pm["distinct4"],
                             **({"units": [u.label() for u in c.proj_units]} if c.proj_units else {}),
                             **({"keys": sorted({u.key or c.default_key for u in c.keyed_units})} if c.keyed_units else {}),
                             **({k: cdm.get(k) for k in ("dir_drop_a", "drift_orth_b", "rate_drop")} if cdm else {}),
                             **({"keyed_selectivity": ckm["selectivity"], "keyed_b_norm": ckm["b_norm"]} if ckm else {})})
            row["ctrl"] = {"n": len(crow), "rows": crow}
        rows.append(row)
        samples[f"{d:g}"] = {"persona": [{"id": t["id"], "text": o["text"][:400]} for t, o in by["persona"]],
                             "mix": [{"id": t["id"], "text": o["text"][:200]} for t, o in by["mix"][:3]],
                             "failed": [{"id": t["id"], "text": o["text"][:200]} for (t, o), ok in
                                        zip(by["exact"] + by["format"], ex + fm, strict=True) if not ok][:6]}
    if km is not None:                               # CHANGE-07 E: 方式の比較と比較用の鍵(倍率 1、keyed のユニットだけ)
        try:
            kblock = km.block()
        except GneError as e:
            session.warn(f"keyed の方式の比較は測れませんでした: {e.message}")
            kblock = {"skipped": e.message, **({"hint": e.hint} if e.hint else {})}
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
        c = row.get("ctrl")
        if c:                                        # 対照の分布(能力の低下・ppl の悪化は無介入との差に直す)
            cr = c["rows"]
            c["novelty"] = summarize_ctrl([x["novelty"] for x in cr])
            c["mass"] = summarize_ctrl([x["mass"] for x in cr])
            c["capability_drop_pt"] = summarize_ctrl(
                [(base["capability"] - x["capability"]) * 100 if base["capability"] is not None and x["capability"] is not None
                 else None for x in cr])
            c["ppl_worse_pct"] = summarize_ctrl(
                [(x["ppl_mean"] / base["ppl_mean"] - 1) * 100 if base["ppl_mean"] and x["ppl_mean"] else None for x in cr])
            row["ctrl_novelty_p95"] = c["novelty"]["p95"]
            row["ctrl_capability_drop_p50"] = c["capability_drop_pt"]["p50"]
            row["ctrl_ppl_worse_p50"] = c["ppl_worse_pct"]["p50"]
            row["novelty_null_pct"] = ctrl_null_pct(row["persona"]["novelty"], [x["novelty"] for x in cr])
            if row.get("dir_drop_a") is not None:          # CHANGE-06 F.2: 目的の指標と副作用の null_pct
                row["dir_null_pct"] = ctrl_null_pct(row["dir_drop_a"], [x.get("dir_drop_a") for x in cr])
                row["drift_orth_b_null_pct"] = ctrl_null_pct(row["drift_orth_b"], [x.get("drift_orth_b") for x in cr])
                c["dir_drop_a"] = summarize_ctrl([x.get("dir_drop_a") for x in cr])
                c["drift_orth_b"] = summarize_ctrl([x.get("drift_orth_b") for x in cr])
            if row.get("rate_drop") is not None:
                row["rate_null_pct"] = ctrl_null_pct(row["rate_drop"], [x.get("rate_drop") for x in cr])
                c["rate_drop"] = summarize_ctrl([x.get("rate_drop") for x in cr])
            ksel = (row.get("keyed") or {}).get("selectivity")
            if ksel is not None and any(x.get("keyed_selectivity") is not None for x in cr):   # CHANGE-07 E.2
                c["keyed_selectivity"] = summarize_ctrl([x.get("keyed_selectivity") for x in cr])
                row["keyed"]["null_pct"] = ctrl_null_pct(ksel, [x.get("keyed_selectivity") for x in cr])
                row["keyed"]["ctrl_p95"] = c["keyed_selectivity"]["p95"]
                if recipe.keyed_units and row["keyed"]["null_pct"] is not None and row["keyed"]["null_pct"] < 0.95:
                    row.setdefault("notes", []).append(
                        f"keyed の eval の選択性 {ksel:+.3g} が、鍵を比較用の鍵に替えた対照の {row['keyed']['null_pct'] * 100:.0f}% より大きい"
                        "だけです(0.95 未満: 鍵が状態を選んでいるとは言えません)")
            # 外から見える指標(判定の一致率)を主にする。無介入で一致率が 0 なら下がりようが無いので射影の値の下がり方で見る
            key = "rate_null_pct" if row.get("rate_null_pct") is not None and (dm.base_rate or 0) > 0 else "dir_null_pct"
            if recipe.proj_units and row.get(key) is not None and row[key] < 0.95:      # V29: 対照を超えていないことを明示
                row.setdefault("notes", []).append(
                    f"目的の指標({'判定の一致率の減少' if key == 'rate_null_pct' else '射影の値の下がり方'})が、同数・同じ s のランダムなヘッドの"
                    f" {row[key] * 100:.0f}% より大きいだけです(0.95 未満: ヘッドの選び方が効いているとは言えません)")
    if rc is not None:
        b0 = base.get("reactivity", {}).get("reactivity")
        for row in rows:
            x = row.get("reactivity")
            if x is not None and b0 is not None:
                x["increase"] = x["reactivity"] - b0              # 無介入からの反応性の増え方
    notes = [f"倍率 {r_['dose']:g}: {n}" for r_ in rows for n in r_.get("notes", [])]
    if kblock is not None:
        notes += [f"keyed: {n}" for n in kblock.get("notes", [])]
        if "skipped" in kblock:
            notes.append(f"keyed: eval の選択性は測っていません({kblock['skipped']})")
    return {"recipe": recipe.to_dict(), "when": w, "random_controls": n_ctrl,
            **({"keyed": kblock} if kblock is not None else {}),
            **({"control_pool": control_pool} if recipe.proj_units and n_ctrl else {}),
            **({"direction_metrics": {"direction": dm.name, "layers": dm.layers, "contrast": dm.cs.name,
                                      "behavior": dm.behavior, "base_rate": dm.base_rate, "n_a": dm.sc.n("A"),
                                      "n_b": dm.sc.n("B"), "n_behavior": dm.n_beh, "max_examples": dm.max_examples}}
               if dm is not None else {}),
            **({"notes": notes} if notes else {}), "_baseline": base_meas,
            "reactivity": None if rc is None else {"contrast": str(rc.path), "name": rc.name, "n_pairs": len(rc.pairs),
                                                   "head_controls": n_hctrl, "max_tokens": reactivity_tokens}, "suite": {"name": st.name, "path": str(st.path), "n_tasks": st.n_tasks,
            "thresholds": th}, "doses": doses, "rows": rows, "samples": samples,
            "judge": {"target": judge.target, "model": str(judge.e.path)} if judge else None,
            "seconds": time.perf_counter() - t0}
