"""Inkling 2 の CHiPS の流れ(gne heads pipeline・レシピの LoRA・焼き込み)。docs/inkling2.md の「CHiPS」。

ほかのモデルの流れ(api.heads_pipeline → unitwrites)と同じ段(方向 → 書き込みの順位づけ → 1 個ずつ確かめる → 組を作る →
応答での確かめ)を、このモデルの形に合わせて流す。決めたこと(段階 3 の試しで確かめた形):
- 方向は MLP が読む混ぜた入力(段ごと、48 行)の A − B。層 6〜17 は 2・3 回目を繰り返しのゲートの後で読み、部品の射影の基準は
  その層の最後の回(3 回目)の段。ほかの層は 1 つの段。最後に混ぜた出力の A − B も 49 行目(番号 48)に持つ
- 射影は短い畳み込みの今のトークンの倍率で直した向き normalize(g ⊙ r) を引く LoRA(inkling_chips.lora_tensors_all)。専門家の
  ニューロンは ffn_down_exps の 3 次元(専門家ごとの倍率を差分に入れる)
- 部品は ヘッド・attention 全体・MLP 全体・専門家のニューロン。流れが始まる層(0・6・12・18)の attention と、層 0〜5 は候補に入れない
  (層 0〜5 は include_before で入れる)
- 測る点は 各区分の最後の段(5・17・29・41・47)と最後に混ぜた出力(番号 48、組を作る目標)
- 方向と測る位置(positions): last(質問の最後のトークン)/ answer(思考の後の、答えの始まり)/ response(答えのトークン全部)/
  reasoning(思考のトークン全部)/ after:<正規表現>(答えの中の一致の直後)/ all / assistant(質問のトークン)。answer・response・
  reasoning・after は、書き換える前のモデルの返事(greedy)を作って、その上で読む
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .steer import GneError

OUT = 48                        # 方向・測る点の番号: 最後に混ぜた出力(段 0〜47 の後)
GATE = "after"
POSITIONS = ("last", "answer", "response", "reasoning", "all", "assistant")
REPLY_KINDS = ("answer", "response", "reasoning", "after")
REASON_TOKENS = 1024            # 返事を作るときの長さの上限(思考 + 答えの始まり)
ANSWER_TOKENS = 96              # response・after で読む答えのトークンの上限(ほかのモデルの --response-tokens と同じ)
RANK_ROWS = 16                  # 書き込みの順位づけで 1 つの会話から読む位置の上限(等間隔に間引く)
POSITION_NOTE = ("このモデルは答えの前に必ず思考を書きます。質問の最後のトークン(last)で作った方向は、思考と答えを書く位置では"
                 "差が 1 割ほどしか残らないことがありました(公開のひな形での測定)。答えの口調や書き方を狙うときは、answer(思考の後の、"
                 "答えの始まり)か response(答えのトークン全部)も試してください")
UNIT_NOTE = "層 6〜17 の重みは 3 回の繰り返しで共有なので、その層の部品の書き換えは 3 回すべてに効きます"


# --- 位置 ------------------------------------------------------------------------------------------------------------

def parse_positions(spec) -> tuple[str, str | None]:
    """positions の書き方 → (種類, 正規表現)。"""
    s = "last" if spec in (None, "") else str(spec).strip()
    if s.startswith("after:"):
        pat = s[len("after:"):]
        if not pat:
            raise GneError("BAD_ARGS", f"positions {s!r} に正規表現がありません")
        try:
            rx = re.compile(pat)
        except re.error as e:
            raise GneError("BAD_ARGS", f"positions {s!r} の正規表現が不正です: {e}") from e
        if rx.fullmatch(""):
            raise GneError("BAD_ARGS", f"positions {s!r} の正規表現が空の文字列に一致します(位置が決まりません)")
        return "after", pat
    if s not in POSITIONS:
        raise GneError("BAD_ARGS", f"Inkling 2 の positions は {' / '.join(POSITIONS)} / after:<正規表現> です({s!r})", POSITION_NOTE)
    return s, None


def _text_id(engine) -> int:
    from .inkling import TEXT_TOKEN
    ids = engine.tokenize(TEXT_TOKEN, parse_special=True)
    if len(ids) != 1:
        raise GneError("INTERNAL", f"答えの始まりの区切り {TEXT_TOKEN} が 1 つのトークンになりません")
    return int(ids[0])


def base_replies(session, items: dict[str, list[tuple[int, list[int]]]], *, sha: str, budget: int = REASON_TOKENS + ANSWER_TOKENS,
                 progress=None) -> dict[str, dict[int, list[int]]]:
    """書き換える前のモデルの greedy の返事(思考から答えまで、budget トークンまで)。items = {side: [(質問の番号, ids)]}。
    <root>/inkling/<model_id>/replies/<対照集合の sha>-<budget>.json に覚えて使い回す(止めて流し直したとき)。"""
    e = session.engine
    p = session.store.root / "inkling" / session.meta.model_id / "replies" / f"{sha}-{budget}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    cache: dict = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    todo = [(sd, i, ids) for sd, xs in items.items() for i, ids in xs if f"{sd}:{i}" not in cache]
    if todo:
        e.free_steer()
        ctx = e.steer_ctx()
        e.set_lora(ctx, None)
        e.set_cvec(ctx, None)
        longest = max(len(ids) for _, _, ids in todo) + budget
        per = max(1, min(int(e.n_seq_max), int(e.n_ctx) // max(longest, 1)))
        done = 0
        for k in range(0, len(todo), per):
            chunk = todo[k:k + per]
            rs = e.generate_greedy(ctx, [ids for _, _, ids in chunk], max_tokens=budget, stop_on_eog=True)
            for (sd, i, _), r in zip(chunk, rs, strict=True):
                cache[f"{sd}:{i}"] = [int(t) for t in r.token_ids]
            done += len(chunk)
            p.write_text(json.dumps(cache), encoding="utf-8")
            if progress:
                progress(done, len(todo), "Inkling 2: 書き換える前の返事")
    return {sd: {i: cache[f"{sd}:{i}"] for i, _ in xs} for sd, xs in items.items()}


def _strip_eog(engine, toks: list[int]) -> list[int]:
    r = list(toks)
    while r and engine.is_eog(r[-1]):
        r.pop()
    return r


def rows_for(engine, kind: str, pattern: str | None, ids: list[int], start: int, reply: list[int] | None,
             text_id: int) -> tuple[list[int], list[int]] | None:
    """1 つの会話の (読む列, 読む位置)。返事が要る種類で答えの始まりが無ければ None(思考の途中で長さの上限に届いた)。
    位置 p の値は「列の p+1 番目のトークンを決める」位置(質問の最後のトークン = 返事の最初のトークンを決める位置)。"""
    P = len(ids)
    if kind == "last":
        return ids, [P - 1]
    if kind == "all":
        return ids, list(range(1, P))
    if kind == "assistant":
        return ids, list(range(max(int(start), 1), P))
    r = _strip_eog(engine, reply or [])
    if text_id not in r:
        return None
    k = r.index(text_id)                                     # 返事の中の答えの始まりの区切り
    if kind == "answer":
        return ids + r[:k + 1], [P + k]
    if kind == "reasoning":
        return ids + r[:k + 1], list(range(P - 1, P + k))
    ans = r[k + 1:k + 1 + ANSWER_TOKENS]
    full = ids + r[:k + 1] + ans
    if kind == "response":                                   # 答えのトークンを決める位置(区切りの位置から、答えの最後の 1 つ前まで)
        pos = list(range(P + k, P + k + len(ans)))
        return (full, pos) if pos else None
    from .keyed_build import select_positions             # after:<正規表現>: 答えの文字の中の一致の直後
    pos = [int(x) for x in select_positions(engine, ans, f"after:{pattern}", n_prompt=P + k + 1)]
    pos = [x for x in pos if x < len(full)]
    return (full, pos) if pos else None


def _thin(xs: list, n: int | None) -> list:
    if not n or len(xs) <= n:
        return list(xs)
    return [xs[int(i)] for i in sorted({int(x) for x in np.linspace(0, len(xs) - 1, int(n)).round()})]


def contrast_items(session, cs, positions: str, *, max_examples: int | None = None, progress=None) -> dict:
    """対照集合の両側の、読む会話と位置。{"items": {side: [(ids, rows)]}, "index": {side: [質問の番号]}, "skipped": {side: n},
    "prompts": {side: [質問]}}。max_examples で等間隔に間引く。"""
    from .reverse import contrast_sequences
    e = session.engine
    kind, pat = parse_positions(positions)
    seqs = contrast_sequences(e, cs)
    idx = {sd: _thin(list(range(len(seqs[sd]))), max_examples) for sd in "AB"}
    replies = None
    if kind in REPLY_KINDS:
        replies = base_replies(session, {sd: [(i, seqs[sd][i][0]) for i in idx[sd]] for sd in "AB"}, sha=cs.sha, progress=progress)
    tid = _text_id(e)
    items: dict[str, list] = {"A": [], "B": []}
    used: dict[str, list[int]] = {"A": [], "B": []}
    skipped = {"A": 0, "B": 0}
    for sd in "AB":
        for i in idx[sd]:
            ids, start = seqs[sd][i]
            got = rows_for(e, kind, pat, ids, start, None if replies is None else replies[sd][i], tid)
            if got is None:
                skipped[sd] += 1
                continue
            items[sd].append(got)
            used[sd].append(i)
    for sd in "AB":
        if len(items[sd]) < 2:
            raise GneError("BAD_CONTRAST", f"{sd} 側で位置 {positions} を読める会話が {len(items[sd])} 個しかありません"
                                           + (f"({skipped[sd]} 個は思考の途中で長さの上限に届いた)" if skipped[sd] else ""),
                           "positions last にするか、質問を増やしてください")
    return {"items": items, "index": used, "skipped": skipped,
            "prompts": {sd: [cs.prompts(sd)[i] for i in used[sd]] for sd in "AB"}, "kind": kind}


# --- 方向 ------------------------------------------------------------------------------------------------------------

def _names(plan) -> dict[int, str]:
    """段 → その段の基準の捕捉の名前(2・3 回目はゲートの後の ffn_in、ほかは ffn_mix)、OUT → out_mix。"""
    from .inkling_chips import ref_kind
    out = {st["step"]: f"{ref_kind(st, GATE)}-{st['step']}" for st in plan.steps()}
    out[OUT] = "out_mix"
    return out


def make_direction(session, cs, name: str, positions: str, *, source: str, max_examples: int | None = None, progress=None) -> dict:
    """段ごとの MLP の入力(基準の種類)と最後に混ぜた出力の、A 側の平均 − B 側の平均。方向の .npz(層の番号 = 段の番号、48 = 出力)。"""
    from .direction import DirectionSet
    from .inkling_read import Reader, want
    t0 = time.perf_counter()
    ci = contrast_items(session, cs, positions, max_examples=max_examples, progress=progress)
    rd = Reader(session, progress=progress)
    rd.require_patch()
    plan = rd.plan
    names = _names(plan)
    sums = {sd: None for sd in "AB"}
    n_rows = {"A": 0, "B": 0}
    rd.fit([ids for sd in "AB" for ids, _ in ci["items"][sd]])
    tot = sum(len(ci["items"][sd]) for sd in "AB")
    done = 0
    for sd in "AB":
        for ids, rows in ci["items"][sd]:
            cap = rd.run(ids, want({"ffn_mix", "ffn_in", "out_mix"}))
            T = len(ids)
            v = np.stack([cap[names[k]].reshape(T, -1)[rows].astype(np.float64).mean(0) for k in sorted(names)])
            sums[sd] = v if sums[sd] is None else sums[sd] + v
            n_rows[sd] += len(rows)
            done += 1
            if progress:
                progress(done, tot, "Inkling 2: 方向")
    D = sums["A"] / len(ci["items"]["A"]) - sums["B"] / len(ci["items"]["B"])
    Ls = sorted(names)
    path = session.store.direction_dir(session.meta.model_id) / f"{name}.npz"
    U = np.stack([D[k] / max(float(np.linalg.norm(D[k])), 1e-30) for k in range(len(Ls))])
    np.savez(path, layers=np.array(Ls, dtype=np.int32), **{f"layer_{L}": D[k].astype(np.float32) for k, L in enumerate(Ls)},
             **{f"scale_{L}": np.ones(D.shape[1], np.float32) for L in Ls}, cos=(U @ U.T).astype(np.float32),
             method="inkling_ffn_input", positions=str(positions), contrast_sha=cs.sha, source=source,
             n_rows_A=n_rows["A"], n_rows_B=n_rows["B"], inkling_gate=GATE, same_reply=False)
    ds = DirectionSet.load(path)
    return {"path": str(path), "positions": str(positions), "n_rows": n_rows, "n_items": {sd: len(ci["items"][sd]) for sd in "AB"},
            "skipped": ci["skipped"], "seconds": time.perf_counter() - t0, "ds": ds}


def is_inkling_direction(ds) -> bool:
    return str(ds.meta.get("method") or "") == "inkling_ffn_input"


def require_inkling_direction(ds) -> None:
    if not is_inkling_direction(ds):
        raise GneError("BAD_DIRECTION", f"方向 {ds.name} は Inkling 2 の形(段ごとの MLP の入力)で作られていません",
                       "gne heads pipeline(Inkling 2 で流す)で作った方向を使ってください")


def layer_refs_of(plan, ds) -> dict[int, tuple[int, np.ndarray]]:
    """保存されている層 → (基準の段, その段の方向の単位ベクトル)。繰り返しの層は最後の回の段。"""
    require_inkling_direction(ds)
    out = {}
    for L in range(plan.n_stored):
        s = plan.steps_of(L)[-1]
        out[L] = (s, ds.unit(s).astype(np.float64))
    return out


# --- 部品 → LoRA ---------------------------------------------------------------------------------------------------

def chips_unit(p) -> list:
    """ProjHead → inkling_chips の部品 [層, 種類, 番号, 強さ]。"""
    if p.kind == "head":
        return [int(p.layer), "head", int(p.head), float(p.s)]
    if p.kind in ("attn", "ffn"):
        return [int(p.layer), p.kind, None, float(p.s)]
    if p.kind == "neuron":
        if getattr(p, "expert", None) is None:
            raise GneError("BAD_RECIPE", f"{p.spec()}: Inkling 2 の MLP は MoE なので、ニューロンは専門家の番号つき(L:Ee:Nn:sS@方向)です")
        return [int(p.layer), "neuron", [int(p.expert), int(p.head)], float(p.s)]
    raise GneError("BAD_RECIPE", f"{p.spec()}: Inkling 2 で射影できる部品は ヘッド・attention 全体・MLP 全体・専門家のニューロンです")


def check_unit(session, p) -> None:
    """Inkling 2 の部品の範囲と、候補から外している部品の知らせ用の確かめ(範囲外は GneError)。"""
    plan = session.weights.inkling
    li = session.weights.arch.layers[0]
    if not 0 <= int(p.layer) < plan.n_stored:
        raise GneError("BAD_RECIPE", f"層 {p.layer} は範囲外です(0〜{plan.n_stored - 1}。Inkling 2 の部品は保存されている 24 層の番号)")
    if p.kind == "head" and not 0 <= int(p.head) < li.n_head:
        raise GneError("BAD_RECIPE", f"L{p.layer} のヘッド {p.head} は範囲外です(0..{li.n_head - 1})")
    if p.kind == "neuron":
        e = getattr(p, "expert", None)
        if e is None:
            chips_unit(p)
        if not 0 <= int(e) < li.n_expert:
            raise GneError("BAD_RECIPE", f"L{p.layer} の専門家 {e} は範囲外です(0..{li.n_expert - 1})")
        if not 0 <= int(p.head) < li.n_ff:
            raise GneError("BAD_RECIPE", f"L{p.layer} E{e} のニューロン {p.head} は範囲外です(0..{li.n_ff - 1})")


def _merge(a: dict, b: dict) -> dict:
    """同じ行列の LoRA をランクの向きにつなぐ(違う方向の部品が同じ行列に入るとき)。"""
    out = dict(a)
    for k, (A, B) in b.items():
        if k not in out:
            out[k] = (A, B)
            continue
        A0, B0 = out[k]
        ax_a, ax_b = (0, 1) if A.ndim == 2 else (1, 2)
        out[k] = (np.concatenate([A0, A], axis=ax_a), np.concatenate([B0, B], axis=ax_b))
    return out


class InkBuilder:
    """ProjHead の並び → LoRA の行列(api の ProjBuilder の代わり。tensors だけを使う)。方向は部品ごとの名前で引く。"""

    def __init__(self, session, direction_set, *, correction: bool = True):
        self.s, self.dir_of, self.correction = session, direction_set, bool(correction)
        self.plan = session.weights.inkling
        self._refs: dict[str, dict] = {}

    def refs(self, name: str) -> dict:
        if name not in self._refs:
            self._refs[name] = layer_refs_of(self.plan, self.dir_of(name))
        return self._refs[name]

    def tensors(self, heads, warn=None) -> dict:
        from .inkling_chips import lora_tensors_all
        rd = SimpleNamespace(w=self.s.weights, plan=self.plan)
        by: dict[str, list] = {}
        for p in heads:
            if p.s:
                check_unit(self.s, p)
                by.setdefault(p.direction, []).append(chips_unit(p))
        out: dict = {}
        for name, units in by.items():
            out = _merge(out, lora_tensors_all(rd, units, self.refs(name), correct=self.correction))
        return out

    def ablate_tensors(self, L, heads):
        raise GneError("ARCH_UNSUPPORTED", "Inkling 2 では、部品の出力を 0 にする確かめ(verify_mode ablate)はまだ使えません", "verify_mode proj(既定)")


def export_lora(session, heads, path: Path, direction_set, *, correction: bool = True, strip_meta: bool = False) -> dict:
    from .ablate import write_lora_tensors
    tens = InkBuilder(session, direction_set, correction=correction).tensors(heads)
    if not tens:
        raise GneError("BAD_RECIPE", "射影の部品がありません")
    session.engine.allow_lora_trial = True               # Inkling 2 の射影の LoRA(このモジュールの作り方)だけを通す
    return write_lora_tensors(session.weights, tens, path, name="adapter" if strip_meta else path.stem,
                              kv=None if strip_meta else {"gne.inkling_proj": json.dumps([p.spec() for p in heads if p.s])})


# --- 測る -----------------------------------------------------------------------------------------------------------

def default_points(plan) -> list[int]:
    """測る点: 各区分の最後の段と、最後に混ぜた出力。"""
    pts = {plan.core_start - 1, plan.n_steps - 1, OUT}
    w = plan.width
    for k in range(plan.n_passes):
        pts.add(plan.core_start + (k + 1) * w - 1)
    return sorted(x for x in pts if x >= 0)


def point_label(plan, k: int) -> str:
    if k == OUT:
        return "最後に混ぜた出力"
    st = plan.step(k)
    return f"段 {k}(層 {st['layer']}" + (f"・{st['pass']} 回目" if st["pass"] else "") + ")"


def parse_points(plan, spec) -> list[int] | None:
    """score_layers の指定(段の番号の並び、out = 最後に混ぜた出力)→ 測る点。None なら既定。"""
    if spec in (None, "", []) or (isinstance(spec, str) and spec.strip().lower() in ("base", "default")):
        return None
    xs = spec.split(",") if isinstance(spec, str) else list(spec)
    out = set()
    for x in xs:
        x = str(x).strip().lower()
        if not x:
            continue
        if x == "out":
            out.add(OUT)
        elif re.fullmatch(r"\d+-\d+", x):
            a, b = (int(v) for v in x.split("-"))
            out.update(range(a, b + 1))
        elif x.isdigit():
            out.add(int(x))
        else:
            raise GneError("BAD_ARGS", f"Inkling 2 の score_layers は段の番号(0〜{plan.n_steps - 1})と out です({spec!r})")
    bad = [x for x in out if not (0 <= x < plan.n_steps or x == OUT)]
    if bad:
        raise GneError("BAD_ARGS", f"段 {bad} は範囲外です(0〜{plan.n_steps - 1}、out)")
    return sorted(out | {OUT})


class InkScorer:
    """direction.Scorer の代わり: 対照集合の会話の、測る位置の 段ごとの MLP の入力・最後に混ぜた出力 · 方向。"""

    def __init__(self, session, ds, cs, *, layers, positions: str = "last", max_examples: int | None = None,
                 items: dict | None = None, progress=None):
        from .inkling_read import Reader
        require_inkling_direction(ds)
        self.session, self.ds, self.positions, self.subset = session, ds, str(positions), "all"
        self.reader = Reader(session)
        self.reader.require_patch()
        self.plan = self.reader.plan
        self.layers = sorted({int(x) for x in layers})
        self.r = {L: ds.unit(L).astype(np.float64) for L in self.layers}
        ci = items or contrast_items(session, cs, positions, max_examples=max_examples, progress=progress)
        self.ci = ci
        self.items = ci["items"]
        self.index_side = ci["index"]
        self.index = self.index_side["A"]
        self.prompts_side = ci["prompts"]
        self.prompts = self.prompts_side["A"]
        self.max_len = max(len(ids) for sd in "AB" for ids, _ in self.items[sd])
        nm = _names(self.plan)
        self.names = {nm[L]: L for L in self.layers}

    def n(self, side: str) -> int:
        return len(self.items[side])

    def run(self, lora=None, vecs=None, *, sides: str = "AB", which: dict | None = None, keep: bool = False) -> dict:
        if vecs:
            raise GneError("ARCH_UNSUPPORTED", "Inkling 2 では control vector を当てて測れません")
        e = self.session.engine
        cap = self.reader.cap
        names = self.names
        cap.set_wanted(lambda n: n in names)
        ctx = e.probe_ctx(cap, need=self.max_len)
        out: dict = {}
        try:
            e.set_lora(ctx, lora or None)
            for side in sides:
                items = self.items[side]
                sel = list(range(len(items))) if which is None or side not in which else list(which[side])
                vals = {L: [] for L in self.layers}
                res: dict = {L: [] for L in self.layers}
                for i in sel:
                    ids, rows = items[i]
                    cap.reset()
                    e.clear_memory(ctx)
                    e.prefill(ctx, [ids])
                    got = cap.take()
                    T = len(ids)
                    for nm, L in names.items():
                        x = got[nm].reshape(T, -1)[rows].astype(np.float64)
                        vals[L].append(float((x @ self.r[L]).mean()))
                        if keep:
                            res[L].append(x)
                out[side] = {"values": {L: np.array(v) for L, v in vals.items()}, "idx": sel}
                if keep:
                    out[side]["resid"] = res
        finally:
            e.set_lora(ctx, None)
        return out


def prepare(session, ds, cs, *, points: list[int], positions: str, max_examples: int | None, b_examples: int = 16,
            progress=None):
    """causal.prepare の代わり(Setup に InkScorer と InkBuilder を入れる)。"""
    from .causal import Setup, TmpLora
    sc = InkScorer(session, ds, cs, layers=points, positions=positions, max_examples=max_examples, progress=progress)
    nb = sc.n("B")
    b_idx = list(range(nb)) if b_examples <= 0 or b_examples >= nb else \
        [int(x) for x in np.linspace(0, nb - 1, b_examples).round().astype(int)]
    session.engine.allow_lora_trial = True               # このモジュールが作る射影の LoRA を当てる
    st = Setup(session, ds, cs, sc, sorted(points), OUT,
               InkBuilder(session, session.direction_set, correction=True), TmpLora(session), sorted(set(b_idx)))
    st.base = st.measure(None)
    return st


# --- 書き込みの順位づけ --------------------------------------------------------------------------------------------

def expert_q(session, ds, progress=None) -> dict:
    """専門家の down の重みを、タップごとの向きとの内積に縮めたもの(方向ごとに 1 回。<root>/inkling/<model_id>/q-<方向>.npz に覚える)。"""
    from .inkling_chips import expert_q as _eq
    import hashlib
    key = hashlib.sha256(np.load(ds.path)["layers"].tobytes() + b"".join(ds.raw[k].tobytes() for k in sorted(ds.raw))).hexdigest()[:12]
    p = session.store.root / "inkling" / session.meta.model_id / f"q-{ds.name}-{key}.npz"
    if p.exists():
        z = np.load(p)
        return {"q": z["q"], "scale": z["scale"], "seconds": float(z["seconds"]), "max_rss_mb": float(z["max_rss_mb"]), "reused": True}
    p.parent.mkdir(parents=True, exist_ok=True)
    eq = _eq(SimpleNamespace(w=session.weights, plan=session.weights.inkling), layer_refs_of(session.weights.inkling, ds), progress=progress)
    np.savez(p, **eq)
    return {**eq, "reused": False}


def rank_units(session, ds, cs, *, positions: str, kinds, include_before: bool = False, skip_stream_start: bool = True,
               max_examples: int | None = None, eq: dict | None = None, items: dict | None = None, progress=None) -> dict:
    """全部の部品の、その層の基準の方向に沿った書き込み(短い畳み込みの後)の A − B(測る位置の平均)。行は unitwrites と同じ形
    (label・layer・index・kind・score・share・own …)に、専門家のニューロンは expert、繰り返しの層は回ごとの値 by_pass を足す。
    1 つの会話から読む位置は RANK_ROWS 個まで(等間隔に間引く)。"""
    from .inkling_chips import _taps, region_of
    from .inkling_read import Reader, Weights2, want
    t0 = time.perf_counter()
    plan = session.weights.inkling
    w2 = Weights2(session.weights)
    li = session.weights.arch.layers[0]
    nh, hd, nE, nf, k_used = li.n_head, li.head_dim, li.n_expert, li.n_ff, li.n_expert_used
    d = session.weights.meta.n_embd
    refs = layer_refs_of(plan, ds)
    eq = eq or expert_q(session, ds, progress=progress)
    q, sc = eq["q"], eq["scale"]
    ci = items or contrast_items(session, cs, positions, max_examples=max_examples, progress=progress)
    rd = Reader(session, progress=progress)
    rd.require_patch()
    U, CW = {}, {}
    for L in range(plan.n_stored):
        Wo = w2.full(f"blk.{L}.attn_output.weight").astype(np.float64).reshape(d, nh, hd)
        cw = w2.full(f"blk.{L}.attn_post_conv.weight").astype(np.float64)
        CW[L] = cw
        U[L] = np.einsum("kd,dhe->khe", _taps(cw, refs[L][1]), Wo)          # (4, nh, hd): タップ k のヘッドの書き込みの r の成分
    acc = {sd: {"heads": np.zeros((plan.n_steps, nh)), "attn": np.zeros(plan.n_steps), "ffn": np.zeros(plan.n_steps),
                "neur": np.zeros((plan.n_steps, nE, nf), np.float32)} for sd in "AB"}
    rd.fit([ids for sd in "AB" for ids, _ in ci["items"][sd]])
    tot = sum(len(ci["items"][sd]) for sd in "AB")
    done = 0
    names = want({"attn_gated", "attn_out", "ffn_out", "ffn_moe_situ", "ffn_moe_argsort", "ffn_moe_weights_scaled"})
    for sd in "AB":
        for ids, rows in ci["items"][sd]:
            rows = _thin(list(rows), RANK_ROWS)
            cap = rd.run(ids, names)
            T = len(ids)
            wr = 1.0 / len(rows)
            for st in plan.steps():
                s, L = st["step"], st["layer"]
                r = refs[L][1]
                g = cap[f"attn_gated-{s}"].reshape(T, nh, hd)
                ao = cap[f"attn_out-{s}"].reshape(T, -1)
                fo = cap[f"ffn_out-{s}"].reshape(T, -1)
                top = cap[f"ffn_moe_argsort-{s}"].reshape(T, -1)[:, :k_used]
                wts = cap[f"ffn_moe_weights_scaled-{s}"].reshape(T, k_used)
                act = cap[f"ffn_moe_situ-{s}"].reshape(T, k_used, nf)
                a = acc[sd]
                for p in rows:
                    for k in range(4):
                        t = p - 3 + k
                        if t < 0:
                            continue
                        a["heads"][s] += wr * np.einsum("he,he->h", g[t].astype(np.float64), U[L][k])
                        e = top[t]
                        c = (wts[t] * sc[L, e])[:, None] * act[t] * q[L, e, k]
                        np.add.at(a["neur"][s], e, (wr * c).astype(np.float32))
                    a["attn"][s] += wr * float(ao[p] @ r)
                    a["ffn"][s] += wr * float(fo[p] @ r)
            del cap
            done += 1
            if progress:
                progress(done, tot, "Inkling 2: 書き込みの順位づけ")
    nA, nB = len(ci["items"]["A"]), len(ci["items"]["B"])
    dif = {k: acc["A"][k] / nA - acc["B"][k] / nB for k in ("heads", "attn", "ffn", "neur")}
    from .unitwrites import parse_kinds
    want_kinds = set(parse_kinds(kinds))
    rows_out = []
    cand_regions = {"loop", "after"} | ({"before"} if include_before else set())

    def bp(f, L):
        return {str(plan.step(s)["pass"] or 0): float(f(s)) for s in plan.steps_of(L)}
    excluded = {"stream_start": [], "before": []}
    for L in range(plan.n_stored):
        rs = refs[L][0]
        region = region_of(plan, L)
        if region not in cand_regions:
            excluded["before"].append(L)
            continue
        ss = skip_stream_start and L % plan.block == 0
        if ss:
            excluded["stream_start"].append(L)
        if "attn" in want_kinds and not ss:
            for h in range(nh):
                rows_out.append({"label": f"L{L}:H{h}", "layer": L, "index": h, "kind": "attn", "score": float(dif["heads"][rs, h]),
                                 "by_pass": bp(lambda s, h=h: dif["heads"][s, h], L), "region": region})
        if "layer_attn" in want_kinds and not ss:
            rows_out.append({"label": f"L{L}:attn", "layer": L, "index": -1, "kind": "layer_attn", "score": float(dif["attn"][rs]),
                             "by_pass": bp(lambda s: dif["attn"][s], L), "region": region})
        if "layer_mlp" in want_kinds:
            rows_out.append({"label": f"L{L}:ffn", "layer": L, "index": -1, "kind": "layer_mlp", "score": float(dif["ffn"][rs]),
                             "by_pass": bp(lambda s: dif["ffn"][s], L), "region": region})
        if "neuron" in want_kinds:
            v = dif["neur"][rs]
            flat = v.reshape(-1)
            keep = min(256, flat.size)
            idx = np.argpartition(-flat, keep - 1)[:keep]
            for x in idx[np.argsort(-flat[idx])]:
                e, n = divmod(int(x), nf)
                rows_out.append({"label": f"L{L}:E{e}:N{n}", "layer": L, "index": n, "expert": e, "kind": "neuron",
                                 "score": float(flat[x]), "by_pass": bp(lambda s, e=e, n=n: dif["neur"][s, e, n], L), "region": region})
    rows_out.sort(key=lambda r: -r["score"])
    pos_total = sum(r["score"] for r in rows_out if r["score"] > 0) or 1.0
    for r in rows_out:
        r["share"] = r["score"] / pos_total
        r["own"] = r["score"]
    n_units = {k: sum(1 for r in rows_out if r["kind"] == k) for k in ("attn", "linear", "neuron", "layer_attn", "layer_mlp")}
    return {"rows": rows_out, "n": {"A": nA, "B": nB}, "n_units": n_units, "kinds": sorted(want_kinds),
            "excluded": excluded, "expert_q": {"seconds": eq.get("seconds"), "reused": eq.get("reused")},
            "seconds": time.perf_counter() - t0}


# --- 一括(gne heads pipeline) -------------------------------------------------------------------------------------

INK_DEFAULTS = {"unit_kinds": "all,layers", "check_tokens": 1024, "max_examples": 32}
NOT_BUILT = {"same_reply": "返事を両方の側で読ませる方向(same_reply)", "holdout": "質問を半分に分ける確かめ(holdout)",
             "loop_direction": "重みを 2 回使うモデル用の loop_direction", "candidate_layers": "候補の層の指定(candidate_layers)",
             "select_layer": "選ぶ層の指定(select_layer。Inkling 2 は最後に混ぜた出力で選ぶ)",
             "always_layers": "rank causal 用の always_layers", "include_score_layers": "rank causal 用の include_score_layers"}


def pipeline(session, spec: dict, pipe: dict, given: dict, t0: float) -> dict:
    """api.heads_pipeline の Inkling 2 の流れ。spec は対照集合の中身(pipeline を除いたもの)、pipe は既定を混ぜた設定、given は
    ひな形・引数で渡された設定。結果は <root>/heads/<model_id>/<name>.pipeline.json(ほかのモデルの rank writes と同じ形)。"""
    from .causal import most_reduced_summary
    from .direction import load_contrast
    from .reverse import check_questions
    from .unitwrites import allocate_units, verify_rows, verify_units
    from .yamlio import write_yaml
    plan = session.weights.inkling
    if pipe.get("rank") == "causal" and "rank" in given:
        raise GneError("ARCH_UNSUPPORTED", "Inkling 2 では --rank causal(層ごとにヘッドを止めて残差で測る選び方)はまだ使えません",
                       "--rank writes(既定)で流してください")
    pipe["rank"] = "writes"
    for k, what in NOT_BUILT.items():
        if given.get(k) not in (None, False, "", []):
            raise GneError("ARCH_UNSUPPORTED", f"Inkling 2 では {what} はまだ使えません(止めました)")
    if pipe.get("verify_mode") not in (None, "proj"):
        raise GneError("ARCH_UNSUPPORTED", "Inkling 2 では verify_mode ablate はまだ使えません", "verify_mode proj(既定)")
    arch_defaults = {}
    for k, v in INK_DEFAULTS.items():
        if k not in given:
            pipe[k] = v
            arch_defaults[k] = v
    kind, _ = parse_positions(pipe.get("positions"))
    sp = pipe.get("score_positions")
    if "score_positions" in given and str(sp) != str(pipe.get("positions")):
        raise GneError("ARCH_UNSUPPORTED", "Inkling 2 では、測る位置は方向を作る位置と同じです(score_positions は positions に合わせます)")
    pipe["score_positions"] = pipe.get("positions") or "last"
    name = str(pipe.pop("name", None) or spec.get("name") or "")
    if not re.fullmatch(r"[\w.\-]+", name):
        raise GneError("BAD_ARGS", f"名前 {name!r} は英数字・_・-・. だけにしてください(spec の name)")
    spec["name"] = name
    for key in ("user_prompts", "prompts_a", "prompts_b"):
        check_questions(spec.get(key), key)
    path = session.store.contrast_dir() / f"{name}.yaml"
    write_yaml(path, spec)
    try:
        cs, val = load_contrast(path)
        if min(len(cs.prompts("A")), len(cs.prompts("B"))) < 4:
            raise GneError("BAD_CONTRAST", "質問は両側とも 4 個以上にしてください(A と B の分かれ方と効き目を測るため)")
        if cs.has_recipes:
            raise GneError("ARCH_UNSUPPORTED", "Inkling 2 では、側をレシピで作る対照集合(recipe_a / recipe_b)はまだ使えません"
                                               "(すべての測定に側ごとの LoRA を重ねる処理を作っていません)", "system_a / system_b で側を作ってください")
    except GneError:
        path.unlink(missing_ok=True)
        raise
    mx = int(pipe["max_examples"]) if pipe.get("max_examples") else None
    points = parse_points(plan, pipe.get("score_layers")) or default_points(plan)
    include_before = bool(given.get("include_before"))
    out: dict = {"name": name, "contrast": str(path), "n_prompts": len(cs.prompts("A")),
                 **({"n_prompts_b": len(cs.prompts("B"))} if cs.separate_prompts else {}), "sides": cs.describe(),
                 "settings": dict(pipe), "rank": "writes", "arch_defaults": arch_defaults or None,
                 "inkling": {"positions": str(pipe["positions"]), "position_kind": kind, "position_note": POSITION_NOTE,
                             "unit_note": UNIT_NOTE, "points": [{"key": k, "label": point_label(plan, k)} for k in points],
                             "include_before": include_before, "reference": "MLP の入力(層 6〜17 は 3 回目、繰り返しのゲートの後)"}}
    session.engine.free_steer()
    ts = time.perf_counter()                                           # 1. 方向
    if pipe.get("from_direction"):
        ds = session.direction_set(str(pipe["from_direction"]))
        require_inkling_direction(ds)
        d = {"path": str(ds.path), "positions": ds.meta.get("positions"), "n_rows": {"A": ds.meta.get("n_rows_A"), "B": ds.meta.get("n_rows_B")},
             "from": ds.name}
        if str(ds.meta.get("positions")) != str(pipe["positions"]):
            raise GneError("BAD_ARGS", f"方向 {ds.name} は位置 {ds.meta.get('positions')} で作られています(今の positions は {pipe['positions']})")
        import shutil
        shutil.copyfile(ds.path, session.store.direction_dir(session.meta.model_id) / f"{name}.npz")
    else:
        d = make_direction(session, cs, name, str(pipe["positions"]), source=str(path), max_examples=mx, progress=session.progress)
    ds = session.direction_set(name)
    out["direction"] = {"path": d["path"], "positions": d["positions"], "n_rows": d["n_rows"], "seconds": time.perf_counter() - ts,
                        "reused": False, **({"skipped": d["skipped"]} if d.get("skipped") else {}),
                        **({"from": d["from"]} if d.get("from") else {})}
    # 2. 段ごとの A と B の分かれ方(ほかのモデルの「基準層」の代わり。Inkling 2 は層ごとの基準が決まっているので選ばない)
    ts = time.perf_counter()
    st = prepare(session, ds, cs, points=points, positions=str(pipe["positions"]), max_examples=mx, progress=session.progress)
    out["score"] = {"base_layer": None, "best_layer": None, "n": {"A": st.scorer.n("A"), "B": st.scorer.n("B")},
                    "layers": [{"layer": L, "label": point_label(plan, L), "A": float(st.base["A"][L].mean()),
                                "B": float(st.base["B"][L].mean())} for L in st.score_layers],
                    "seconds": time.perf_counter() - ts}
    ts = time.perf_counter()                                           # 3. 書き込みの順位づけ
    rk = rank_units(session, ds, cs, positions=str(pipe["positions"]), kinds=pipe["unit_kinds"], include_before=include_before,
                    max_examples=mx, items=st.scorer.ci, progress=session.progress)
    gap = float(st.base["A"][OUT].mean() - st.base["B"][OUT].mean())
    out["writes"] = {"select_layer": OUT, "select_label": point_label(plan, OUT), "kinds": rk["kinds"], "n": rk["n"], "gap": gap,
                     "n_units": rk["n_units"], "totals_share": {}, "layer_scores": [], "excluded": rk["excluded"],
                     "expert_q": rk["expert_q"], "seconds": time.perf_counter() - ts, "top": rk["rows"][:40]}
    ts = time.perf_counter()                                           # 4. 上位を 1 個ずつ s=1 で確かめる
    top = verify_rows(rk["rows"], int(pipe["verify_top"]), int(pipe["verify_sample"]))
    ver = verify_units(session, st, top, mode="proj", progress=session.progress)
    by = {r["label"]: r for r in rk["rows"]}
    for r in ver["rows"]:                                              # 専門家の番号・回ごとの値を戻す(配分が部品を作るのに使う)
        src = by.get(r["label"]) or {}
        for k in ("expert", "by_pass", "region"):
            if k in src:
                r[k] = src[k]
    out["verify"] = {"mode": ver["mode"], "n": len(ver["rows"]), "spearman": ver["spearman"],
                     "kinds": {k: sum(1 for r in ver["rows"] if r["kind"] == k) for k in ("attn", "linear", "neuron", "layer_attn", "layer_mlp")},
                     "excluded": [r["label"] for r in ver["rows"] if r.get("excluded")], "top": ver["rows"][:20],
                     "seconds": time.perf_counter() - ts}
    p_rank = session.store.heads_dir(session.meta.model_id) / f"{name}.writes.json"
    p_rank.write_text(json.dumps({"rows": rk["rows"][:2000], "verify": ver["rows"], "n": rk["n"], "kinds": rk["kinds"]},
                                 ensure_ascii=False), encoding="utf-8")
    out["writes"]["path"] = str(p_rank)
    bs = {"A": float(st.base["A"][OUT].mean()), "B": float(st.base["B"][OUT].mean())}
    tgt, tgt_from = session._pipeline_target(pipe["target"], bs)
    out["causal"] = {"score_layers": st.score_layers, "primary": OUT, "n_a": st.scorer.n("A"), "n_b": len(st.b_idx),
                     "base_scores": {str(L): {"A": float(st.base["A"][L].mean()), "B": float(st.base["B"][L].mean())}
                                     for L in st.score_layers}}
    a = allocate_units(session, st, ver["rows"], target=float(tgt), n_units=[int(x) for x in pipe["n_units"]],
                       s_max=float(pipe["s_max"]), name=name, direction_layer="own", remain_threshold=float(tgt), max_examples=mx,
                       strength=str(pipe.get("strength") or "per_unit"), hard=int(pipe.get("hard") or 16), progress=session.progress)
    a["notes"] = list(a.get("notes") or []) + [{"code": "inkling_shared", "text": UNIT_NOTE}]
    p_ = Path(a["path"])
    p_.write_text(json.dumps({k: v for k, v in a.items() if k != "path"}, ensure_ascii=False, indent=1), encoding="utf-8")
    out["allocate"] = a
    out["reused"] = []
    out["target"], out["target_from"] = float(tgt), tgt_from
    rec = next((r for r in a["table"] if r.get("recommended")), None)
    out["recommended"] = None if rec is None else {"n_heads": rec["n_heads"], "units": rec["units"], "recipe": rec["recipe"],
                                                   "reached": bool(rec["reached"]), "by": a.get("recommended_by")}
    if a.get("no_recommend"):
        out["no_recommend"] = a["no_recommend"]
    n_check = pipe.get("check_replies")
    n_check = 16 if n_check is None else int(n_check)
    ok_rows = [r for r in a["table"] if not r.get("skipped")]
    chk = rec or (min(ok_rows, key=lambda r: (r["remain"], r["b_drift_orth"])) if ok_rows else None)
    if n_check > 0 and chk is not None:                                # 5. 応答での確かめ(思考を除いた答えで数える)
        out["check"] = session.reply_check(cs, val, chk["recipe"], n=n_check, max_tokens=int(pipe["check_tokens"]))
        out["check"].update(n_heads=chk["n_heads"], recommended=rec is not None)
        if rec is not None and out["check"].get("break_warning"):
            session._recommend_without_broken(out, a, rec, cs, val, n_check, int(pipe["check_tokens"]))
    out["most_reduced"] = most_reduced_summary(out)
    out["seconds"] = time.perf_counter() - t0
    out["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    p = session.store.heads_dir(session.meta.model_id) / f"{name}.pipeline.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=_json), encoding="utf-8")
    out["path"] = str(p)
    return out


def _json(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


# --- 焼き込み -------------------------------------------------------------------------------------------------------

def bake_recipe(session, heads, dst: Path, *, requant: str = "auto", vram_free_mb: float | None = None,
                strip_meta: bool = False) -> dict:
    """レシピの射影の部品を GGUF に焼き込む(inkling_chips.bake)。requant は auto(既定: 元の型・元の大きさで書き、編集が残らない
    テンソルだけ Q8_0)・keep・q8_0・f16。書き換えるテンソルごとのファイル・VRAM の増え方を出し、vram_free_mb(読み込んだときの
    空き)を渡すと、収まらないときは止める(auto・keep は書く前に一度計算して、Q8_0 になるテンソルの分を数える)。"""
    from .inkling_chips import bake
    refs_by: dict[str, list] = {}
    for p in heads:
        if p.s:
            check_unit(session, p)
            refs_by.setdefault(p.direction, []).append(chips_unit(p))
    if len(refs_by) != 1:
        raise GneError("ARCH_UNSUPPORTED", "Inkling 2 の焼き込みは、1 つの方向の射影のレシピだけです" if refs_by else "射影の部品がありません")
    (dname, units), = refs_by.items()
    refs = layer_refs_of(session.weights.inkling, session.direction_set(dname))
    rd = SimpleNamespace(w=session.weights, plan=session.weights.inkling)
    if requant in ("auto", "keep"):                          # 書く前に一度計算して、型と大きさを決める
        pre = bake(rd, units, refs, None, requant=requant)["modified"]
        grow = {"bytes": int(sum(m["bytes_to"] - m["bytes_from"] for m in pre)),
                "tensors": [{"tensor": m["tensor"], "type": m["from"], "to": m["to"], "from_mib": m["bytes_from"] / 2**20,
                             "to_mib": m["bytes_to"] / 2**20} for m in pre],
                "expert_layers": sorted({int(m["tensor"].split(".")[1]) for m in pre if "ffn_down_exps" in m["tensor"] and m["to"] != m["from"]})}
    else:
        grow = estimate_growth(session, units, requant)
    if vram_free_mb is not None and grow["bytes"] / 2**20 > vram_free_mb - VRAM_MARGIN_MB:
        raise GneError("VRAM_FULL", f"焼き込むと VRAM が {grow['bytes'] / 2**20:.0f} MiB 増えますが、空きは {vram_free_mb:.0f} MiB です"
                                    f"(書き換える専門家の層 {grow['expert_layers']})",
                       "部品の層を減らすか、LoRA のまま使ってください(書き換える専門家の層 1 つで Q8_0 は Q2_0 の層 +253 MiB・NVFP4 の層 +162 MiB)")
    r = bake(rd, units, refs, Path(dst), requant=requant, strip_meta=strip_meta)
    r["growth"] = grow
    return r


def gpu_free_mb() -> float | None:
    """GPU の空き(MiB、最初の GPU)。llama.cpp(ggml)の装置の一覧から読む。GPU が無い・読めなければ None。"""
    import ctypes
    try:
        from llama_cpp._ggml import libggml as G
        G.ggml_backend_dev_count.restype = ctypes.c_size_t
        G.ggml_backend_dev_get.restype = ctypes.c_void_p
        G.ggml_backend_dev_get.argtypes = [ctypes.c_size_t]
        G.ggml_backend_dev_type.restype = ctypes.c_int
        G.ggml_backend_dev_type.argtypes = [ctypes.c_void_p]
        G.ggml_backend_dev_memory.restype = None
        G.ggml_backend_dev_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        for i in range(int(G.ggml_backend_dev_count())):
            dev = G.ggml_backend_dev_get(i)
            if G.ggml_backend_dev_type(dev) == 1:            # GGML_BACKEND_DEVICE_TYPE_GPU
                free, total = ctypes.c_size_t(0), ctypes.c_size_t(0)
                G.ggml_backend_dev_memory(dev, ctypes.byref(free), ctypes.byref(total))
                return free.value / 2**20
    except Exception:                                        # noqa: BLE001
        return None
    return None


VRAM_MARGIN_MB = 256            # 焼き込んだ GGUF を読むときに残しておく空き


def estimate_growth(session, units: list, requant: str) -> dict:
    """焼き込みでファイル(と VRAM)が増える大きさ: 書き換えるテンソルの元の大きさと、書き出す型での大きさの差。"""
    from .quant import block_sizes, qtype_of
    T = session.weights._tensors
    names = set()
    for L, k, _, _ in units:
        names.add(f"blk.{L}.attn_output.weight" if k in ("head", "attn") else f"blk.{L}.ffn_down_exps.weight")
    tot, per = 0, []
    for nm in sorted(names):
        t = T[nm]
        n_el = int(np.prod(t.shape))
        if requant == "keep":
            to = int(t.n_bytes)
        else:
            q = qtype_of({"q8_0": 8, "f16": 1}[requant])
            bs, ts = block_sizes(q)
            to = n_el * ts // bs
        per.append({"tensor": nm, "type": t.tensor_type.name, "from_mib": t.n_bytes / 2**20, "to_mib": to / 2**20})
        tot += to - int(t.n_bytes)
    return {"bytes": tot, "tensors": per,
            "expert_layers": sorted({int(x["tensor"].split(".")[1]) for x in per if "ffn_down_exps" in x["tensor"]})}
