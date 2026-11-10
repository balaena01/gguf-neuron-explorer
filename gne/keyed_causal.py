"""keyed(鍵つきの書き込み、CHANGE-07)の「効く場所を探す」: gne direction key-causal(第 1 段)と key-causal-verify(第 2 段)。

key-check は鍵が場面を分けるか(鍵の反応 ρ = κᵀx の差)しか見ない。どの層・どの書き込み先に書けば振る舞いが変わるかは別の問題で、
key-check の余裕が大きい層に書いても振る舞いが変わらないことがある。そこで 2 段で探す。自動で 1 つに決める規則は入れない。

第 1 段 key_causal(生成なしのふるい分け):
- 候補(層 × 書き込み先)ごとに、対照集合の偶数番目(subset even)で鍵を作る(書き込み先ごとに全候補の層を 1 つの鍵のファイルに。
  比較用の鍵つき)。測るのは作成に使っていない奇数番目(odd)。
- 介入(iv。既定は対照集合の keyed.iv)を当てた状態を基準にし、候補の keyed のユニット(α の段ごと)を足したときの変化を、
  返事を読ませるだけ(prefill)で測る。返事は対照集合の replies か、iv を当てて 1 回だけ作ったもの(どの条件でも同じトークン列)。
  - 効き目(返事): 書き込みたい場面の返事で、振る舞いの部分(既定は positions の正規表現に一致した文字。measure_match で別に)の
    トークンの対数確率の和が、どれだけ下がるか(例ごとに一致の平均、例で平均。負 = 振る舞いが出にくくなる)。
  - 効き目(probe): 質問ごとに「返事の前に置く文字」「振る舞いの答え」「望む答え」を書いたファイルを渡すと、その位置での
    log p(振る舞いの答え) − log p(望む答え) の変化(負 = 望む答えに寄る)。
  - 副作用: dont_touch の各群の返事の、最初の side_tokens トークンの 1 トークンあたりの対数確率の変化。書き込みたくない位置
    (mode same は同じ返事の一致以外の位置、mode system は B 側の返事)も同じ。probe の書き込みたくない場面の行は差の変化。
  - 比較用の鍵: 候補ごとに pseudo 本(ラベルをシャッフルして作った鍵 NAME~j)を同じ α・同じ測り方で。全候補の比較用の鍵の値を
    α ごとにまとめた分布の 5% 点より下がった候補に印を付け、「候補の数 × 5% = 偶然でもこのくらいは印が付く数」を並べる。
  - 効き目が α に比例しているかの目安(scale_ratio)と、key-check の値(参考)。
- これは生成なしの目安で、比較用の鍵と見分けがつかない候補を外すためのもの。効くことの保証ではない(生成で行動が変わった
  場面でも、すぐ答えさせる形の確率の差が動かないことがある)。効くかどうかは第 2 段の生成で数える。

第 2 段 key_causal_verify(生成での確かめ):
- 使う人が選んだ候補(keyed のユニット)と、それぞれの比較用の鍵 1 本で、作成に使っていない奇数番目の質問(と dont_touch の
  奇数番目の質問)に behave と同じ生成をさせ、振る舞いが出た数(正規表現か judge のラベル)を iv だけのときと比べる。
  dont_touch では返事が iv だけのときから変わった数(greedy のとき)、どれも繰り返し・空の返事・上限で切れた数を出す。

結果は <root>/keys/<model_id>/causal/<名前>.json(第 1 段)と <名前>.verify.json(第 2 段、返事も)に保存する。
"""

from __future__ import annotations

import bisect
import dataclasses
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import keyed as K
from .steer import GneError

Progress = Callable[[int, int, str], None]

DEFAULT_PSEUDO = 3            # 候補ごとに測る比較用の鍵の本数
KC_PSEUDO = 8                 # 鍵の作成で作る比較用の鍵の本数の下限(key-check の参考値の p95 用。測るのは pseudo 本)
DEFAULT_ALPHAS = (1.0, 2.0, 4.0)   # α の段(対照集合の keyed.alpha の符号を掛ける)
SIDE_TOKENS = 32              # 副作用を測る返事のトークンの数(返事の最初から)
START_TOKENS = 3              # dont_touch の返事の冒頭(返事の最初の 3 トークン)
CHANCE = 0.05                 # 印の基準: 比較用の鍵の分布の 5% 点。偶然でも候補の 5% には印が付く
MIN_POOL = 20                 # 比較用の鍵がこれより少ないと 5% 点が粗いと知らせる
VERIFY_TOKENS = 256           # 第 2 段の返事の長さの既定
SIDE_FLOOR = 0.01             # 「確率の下がり方 ÷ 副作用」の副作用の下限(1 トークンあたりの対数確率の低下。これより小さい副作用・
                              # 上がった副作用はこの値で割る。0 に近い副作用で割って比が際限なく大きくなるのを防ぐ)
NOTE = ("これは生成なしの目安です。比較用の鍵(ラベルをシャッフルして作った鍵)と見分けがつかない候補を外すためのもので、"
        "効くことの保証ではありません(生成で振る舞いが変わる場面でも、生成なしの確率の差は動かないことがあります)。"
        "効くかどうかは第 2 段(gne direction key-causal-verify、UI は「選んだ行を生成で確かめる」)の生成で数えてください")
_NAME_RE = re.compile(r"[\w.][\w.\-]*")


# --- 書き込み先 ----------------------------------------------------------------------------------------------

def target_tag(target: str) -> str:
    """書き込み先 → 鍵の名前の後ろとユニットの書き方に使う短い印(ffn / attn / S / Ee)。"""
    kind, e = K.parse_target(target)
    return {"ffn": "ffn", "attn": "attn", "shared": "S"}.get(kind, f"E{e}")


def unit_spec(layer: int, target: str, alpha: float, direction: str, key: str) -> str:
    """keyed のユニットの書き方(L:ffn:keyed:aα@方向#鍵)。"""
    return f"{int(layer)}:{target_tag(target)}:keyed:a{float(alpha):g}@{direction}#{key}"


def parse_targets(targets) -> list[str]:
    """書き込み先の指定("ffn,attn"・["ffn", "expert:3"])→ 正規化したリスト(重複なし)。"""
    if targets is None:
        return []
    xs = [t.strip() for t in (targets.split(",") if isinstance(targets, str) else targets) if str(t).strip()]
    out = []
    for t in xs:
        t = "shared" if t.lower() in ("s", "shared") else t.lower()
        kind, e = K.parse_target(t)
        tt = K.target_of(kind, e)
        if tt not in out:
            out.append(tt)
    return out


def parse_alphas(alphas) -> list[float]:
    """α の段("-1,-2,-4"・[-1, -2])→ 0 を除いた数のリスト(書いた順、重複なし)。"""
    if alphas is None:
        return []
    xs = alphas.split(",") if isinstance(alphas, str) else list(alphas)
    out: list[float] = []
    for x in xs:
        if isinstance(x, str) and not x.strip():
            continue
        try:
            a = float(x)
        except (TypeError, ValueError) as e:
            raise GneError("BAD_ARGS", f"α の段は -1,-2,-4 のように書きます({alphas!r})") from e
        if not np.isfinite(a) or a == 0:
            raise GneError("BAD_ARGS", f"α は 0 でない数です({x!r})")
        if a not in out:
            out.append(a)
    return out


def candidate_layers(session, ds, target: str, layers=None) -> tuple[list[int], list[dict]]:
    """書き込み先 target に書ける層(MTP・最終層・書き込み先の無い層・方向に無い層を除く)と、外した層の理由。
    layers を省くと全層(外した層の理由は返さない)。ds が None(方向をこれから作る)なら方向の層は見ない。"""
    w = session.weights
    kind, e = K.parse_target(target)
    want = list(range(session.meta.n_layer)) if layers is None else sorted({int(L) for L in layers})
    ok, skipped = [], []
    for L in want:
        why = None
        try:
            K.check_layer_ok(w, L, kind, e)
        except GneError as ex:
            why = ex.message
        if why is None and ds is not None and L not in ds.raw:
            why = f"方向 {ds.name} に層 {L} がありません"
        if why is None:
            ok.append(L)
        elif layers is not None:
            skipped.append({"target": target, "layer": L, "error": why})
    return ok, skipped


# --- probe(選択式の振る舞い向け)-------------------------------------------------------------------------

_PROBE_KEYS = {"prompt", "system", "prefix", "behavior", "desired", "group", "dont_touch"}


def read_probe(probe) -> list[dict]:
    """probe(JSONL / JSON のパス、または行のリスト)→ 行のリスト。1 行は {prompt(質問か会話の履歴), system?, prefix?(返事の前に
    置く文字、例 "Answer:"), behavior(振る舞いの答え、例 " B"), desired(望む答え), group?(書き込みたくない場面の行なら群の名前)
    か dont_touch?: true}。group も dont_touch も無い行は書き込みたい場面の行。"""
    if probe is None:
        return []
    if isinstance(probe, (list, tuple)):
        rows = list(probe)
        where = "probe"
    else:
        p = Path(str(probe))
        where = p.name
        if not p.exists():
            raise GneError("FILE_NOT_FOUND", f"probe {p} がありません")
        text = p.read_text(encoding="utf-8")
        try:
            if p.suffix == ".json":
                d = json.loads(text)
                rows = d.get("rows") if isinstance(d, dict) else d
            else:
                rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        except ValueError as e:
            raise GneError("BAD_ARGS", f"{where}: JSON として読めません({e})") from e
    if not isinstance(rows, list) or not rows:
        raise GneError("BAD_ARGS", f"{where}: probe は行(表)のリストです")
    out = []
    for i, r in enumerate(rows):
        w = f"{where} の {i + 1} 行目"
        if not isinstance(r, dict):
            raise GneError("BAD_ARGS", f"{w} は表です({{prompt, prefix?, behavior, desired, group?}})")
        extra = set(r) - _PROBE_KEYS
        if extra:
            raise GneError("BAD_ARGS", f"{w} に知らないキー {sorted(extra)} があります({', '.join(sorted(_PROBE_KEYS))})")
        q = r.get("prompt")
        if not ((isinstance(q, str) and q.strip()) or (isinstance(q, list) and q)):
            raise GneError("BAD_ARGS", f"{w}: prompt は質問(文字列)か会話の履歴です")
        if isinstance(q, list):
            from .reverse import check_questions
            try:
                check_questions([q], f"{w}: prompt")
            except GneError as e:
                raise GneError("BAD_ARGS", e.message, e.hint) from e
        b, d = r.get("behavior"), r.get("desired")
        if not (isinstance(b, str) and b and isinstance(d, str) and d):
            raise GneError("BAD_ARGS", f"{w}: behavior(振る舞いの答え)と desired(望む答え)は空でない文字列です")
        if b == d:
            raise GneError("BAD_ARGS", f"{w}: behavior と desired が同じです")
        pre = r.get("prefix", "")
        if not isinstance(pre, str):
            raise GneError("BAD_ARGS", f"{w}: prefix は文字列です")
        sysm = r.get("system")
        if sysm is not None and not isinstance(sysm, str):
            raise GneError("BAD_ARGS", f"{w}: system は文字列です")
        g = r.get("group")
        if g is not None and not (isinstance(g, str) and g.strip()):
            raise GneError("BAD_ARGS", f"{w}: group は群の名前(文字列)です")
        if g is None and r.get("dont_touch"):
            g = "dont_touch"
        out.append({"prompt": q, "system": sysm, "prefix": pre, "behavior": b, "desired": d, "group": g})
    return out


# --- 位置 --------------------------------------------------------------------------------------------------

def match_spans(pieces: Sequence[bytes], pattern) -> list[list[int]]:
    """正規表現に一致する箇所(重ならない一致すべて、長さ 0 は除く)ごとの、一致の文字を含むトークンの番号(昇順)。
    文字の位置は keyed_build.match_tokens と同じく、トークン片のバイト長の累積でトークンに戻す。"""
    from .keyed_build import byte_offsets
    rx = re.compile(pattern) if isinstance(pattern, str) else pattern
    offs = byte_offsets(pieces)
    text = b"".join(pieces).decode("utf-8", "surrogateescape")
    cb = [0]
    for ch in text:
        cb.append(cb[-1] + len(ch.encode("utf-8", "surrogateescape")))
    out = []
    for m in rx.finditer(text):
        if m.end() == m.start():
            continue
        a = bisect.bisect_right(offs, cb[m.start()]) - 1
        b = bisect.bisect_right(offs, cb[m.end()] - 1) - 1
        out.append(list(range(a, max(a, b) + 1)))
    return out


def _answer_ids(e, prefix: str, ans: str) -> tuple[list[int], list[int]]:
    """返事の前に置く文字と答え → (前に置く文字のトークン, 答えのトークン)。続けて区切ったときに前の部分が同じなら、その残りを
    答えにする(トークンの区切りが文字の境目でまたがらないとき)。違えば答えだけを区切る。"""
    pre = e.tokenize(prefix, add_special=False, parse_special=False) if prefix else []
    full = e.tokenize(prefix + ans, add_special=False, parse_special=False)
    if full[:len(pre)] == pre and len(full) > len(pre):
        return pre, full[len(pre):]
    return pre, e.tokenize(ans, add_special=False, parse_special=False)


# --- 速い対数確率(steer ctx で複数の列をまとめて読む)---------------------------------------------------

class _Scorer:
    """列(トークン)のリストと、列ごとに「位置 → その次に来るトークンの番号」を渡し、teacher forcing の対数確率を返す。
    n_seq_max 本・n_ctx に収まる分ずつまとめて読み、ロジットは欲しい位置だけ取る。生成用のコンテキストを使う(読む間は
    ほかの処理に GPU を譲らない。譲るのは run の切れ目だけ)。"""

    def __init__(self, session):
        self.e = session.engine
        self.ctx = self.e.steer_ctx()
        self.rows = 0

    def run(self, seqs: list[list[int]], wants: list[dict[int, list[int]]], vecs=None, lora=None) -> list[dict[tuple[int, int], float]]:
        e, ctx = self.e, self.ctx
        out: list[dict[tuple[int, int], float]] = [{} for _ in seqs]
        order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
        n_ctx = int(getattr(ctx, "n_ctx", 0) or e.n_ctx)
        for i in order:
            if len(seqs[i]) > n_ctx:
                raise GneError("CONTEXT_FULL", f"{len(seqs[i])} トークンの列が n_ctx={n_ctx} に入りません")
        e.set_lora(ctx, lora or None)
        e.set_cvec(ctx, vecs or None)
        try:
            while order:
                grp, used = [], 0
                for i in order:
                    if len(grp) < ctx.n_seq_max and used + len(seqs[i]) <= n_ctx:
                        grp.append(i)
                        used += len(seqs[i])
                order = [i for i in order if i not in grp]
                e.clear_memory(ctx)
                flat = []
                for sid, i in enumerate(grp):
                    w = wants[i]
                    flat += [(t, p, sid, p in w) for p, t in enumerate(seqs[i])]
                for c0 in range(0, len(flat), ctx.n_batch):
                    chunk = flat[c0:c0 + ctx.n_batch]
                    e._decode(ctx, chunk)
                    for j, (_, p, sid, lg) in enumerate(chunk):
                        if not lg:
                            continue
                        row = e._logits(ctx, j)
                        m = float(row.max())
                        lse = m + float(np.log(np.exp(row - m, dtype=np.float32).sum(dtype=np.float64)))
                        i = grp[sid]
                        for tok in wants[i][p]:
                            out[i][(p, int(tok))] = float(row[int(tok)]) - lse
                        self.rows += 1
        finally:
            e.set_cvec(ctx, None)
            e.set_lora(ctx, None)
        return out


# --- 測る列 ------------------------------------------------------------------------------------------------

@dataclass
class _MSet:
    """1 つの介入(側の介入 + iv)で読む列と、列ごとに何を測るか。"""
    label: str                                 # A / B / dt:<名前> / probe
    iv: tuple
    seqs: list[list[int]] = field(default_factory=list)
    wants: list[dict[int, list[int]]] = field(default_factory=list)
    meta: list[dict] = field(default_factory=list)


def _want(w: dict[int, list[int]], pos: int, tok: int) -> tuple[int, int]:
    w.setdefault(int(pos), [])
    if int(tok) not in w[int(pos)]:
        w[int(pos)].append(int(tok))
    return int(pos), int(tok)


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def _stat(d: np.ndarray) -> dict | None:
    """例ごとの差 → {mean, ci(95% の幅の半分 = 1.96·sd/√n), n}。値が無ければ None。"""
    v = np.asarray([x for x in d if x is not None and np.isfinite(x)], dtype=np.float64)
    if not v.size:
        return None
    ci = float(1.96 * v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else None
    return {"mean": float(v.mean()), "ci": ci, "n": int(v.size)}


class _Plan:
    """第 1 段で読む列(作成に使っていない奇数番目)と、介入ごとの測り方。"""

    def __init__(self, session, spec, *, mode: str, regex: str | None, skip: int, side_tokens: int, probe_rows: list[dict],
                 side_iv: dict, vecs0, lora0, built_questions: set[str]):
        from .direction import _subset
        from .reverse import _conv, combine, question_text
        e = session.engine
        self.mode, self.regex, self.side_tokens = mode, regex, int(side_tokens)
        self.sets: list[_MSet] = []
        self.counts: dict = {}
        self.notes: list[str] = []
        cs = spec.contrast
        rx = re.compile(regex) if regex else None
        side = "B" if mode == "same" else "A"
        prompts = cs.prompts(side)
        idx = _subset(len(prompts), "odd")
        reps = spec.replies.get("same") if mode == "same" else spec.replies.get("A")
        A = _MSet("A", combine(side_iv[side], vecs0, lora0))
        n_match = n_span = 0
        for i in idx:
            conv = _conv(cs.system(side), prompts[i])
            pids = e.tokenize_prompt(e.render_chat(conv))
            rid = e.tokenize(str(reps[i]), add_special=False, parse_special=False) if reps is not None else []
            rid = [t for t in rid if not e.is_eog(t)]
            if not rid:
                continue
            P = len(pids)
            w: dict[int, list[int]] = {}
            spans = []
            if rx is not None:
                pieces = [e.token_bytes(t) for t in rid]
                spans = [[_want(w, P + j - 1, rid[j]) for j in sp] for sp in match_spans(pieces, rx) if sp[0] >= skip]
            in_span = {pt[0] - (P - 1) for sp in spans for pt in sp}
            rest = [_want(w, P + j - 1, rid[j]) for j in range(min(len(rid), self.side_tokens))
                    if mode == "same" and j not in in_span]
            if not spans and not rest:
                continue
            n_match += bool(spans)
            n_span += len(spans)
            A.seqs.append(pids + rid)
            A.wants.append(w)
            A.meta.append({"kind": "a", "spans": spans, "rest": rest})
        self.sets.append(A)
        self.counts["a"] = {"examples": len(idx), "matched": n_match, "spans": n_span}
        if mode == "system":                                    # 書き込みたくない側(B)の返事
            B = _MSet("B", combine(side_iv["B"], vecs0, lora0))
            pb = cs.prompts("B")
            rb = spec.replies.get("B")
            for i in _subset(len(pb), "odd"):
                self._reply_item(B, e, _conv(cs.system("B"), pb[i]), None if rb is None else rb[i])
            self.sets.append(B)
            self.counts["b"] = len(B.seqs)
        self.counts["dt"] = {}
        for dt in spec.dont_touch:
            S = _MSet(f"dt:{dt.name}", combine(None, vecs0, lora0))
            for i in _subset(len(dt.prompts), "odd"):
                self._reply_item(S, e, _conv(dt.system, dt.prompts[i]), None if dt.replies is None else dt.replies[i])
            self.sets.append(S)
            self.counts["dt"][dt.name] = len(S.seqs)
        # probe: 書き込みたい場面の行と、書き込みたくない場面の行(群ごと)。作成に使った質問の行は外す
        self.probe_rows: list[dict] = []
        excluded = 0
        if probe_rows:
            Pz = _MSet("probe", combine(side_iv["B"], vecs0, lora0))
            for r in probe_rows:
                if question_text(r["prompt"]) in built_questions:
                    excluded += 1
                    continue
                pids = e.tokenize_prompt(e.render_chat(_conv(r["system"], r["prompt"])))
                pre_b, ans_b = _answer_ids(e, r["prefix"], r["behavior"])
                pre_d, ans_d = _answer_ids(e, r["prefix"], r["desired"])
                k = len(self.probe_rows)
                self.probe_rows.append({"group": r["group"]})
                if pre_b == pre_d and len(ans_b) == 1 and len(ans_d) == 1:     # 1 トークンの答え: 同じ列の同じ位置で両方
                    w = {}
                    pos = len(pids) + len(pre_b) - 1
                    Pz.seqs.append(pids + pre_b)
                    Pz.wants.append(w)
                    Pz.meta.append({"kind": "probe", "row": k, "parts": [("b", [_want(w, pos, ans_b[0])]), ("d", [_want(w, pos, ans_d[0])])]})
                    continue
                for which, pre, ans in (("b", pre_b, ans_b), ("d", pre_d, ans_d)):
                    w = {}
                    base = len(pids) + len(pre)
                    Pz.seqs.append(pids + pre + ans)
                    Pz.wants.append(w)
                    Pz.meta.append({"kind": "probe", "row": k, "parts": [(which, [_want(w, base + j - 1, t) for j, t in enumerate(ans)])]})
            self.sets.append(Pz)
        groups = [r["group"] for r in self.probe_rows]
        self.counts["probe"] = {"a": sum(g is None for g in groups), "dt": {g: groups.count(g) for g in dict.fromkeys(groups) if g},
                                "excluded": excluded}
        if excluded:
            self.notes.append(f"probe の {excluded} 行は鍵の作成に使った質問なので外しました(作成に使っていない質問だけで測ります)")

    def _reply_item(self, S: _MSet, e, conv, reply: str | None) -> None:
        pids = e.tokenize_prompt(e.render_chat(conv))
        rid = [t for t in e.tokenize(str(reply or ""), add_special=False, parse_special=False) if not e.is_eog(t)]
        if not rid:
            return
        P = len(pids)
        w: dict[int, list[int]] = {}
        toks = [_want(w, P + j - 1, rid[j]) for j in range(min(len(rid), self.side_tokens))]
        start = toks[:START_TOKENS]
        S.seqs.append(pids + rid)
        S.wants.append(w)
        S.meta.append({"kind": "reply", "toks": toks, "start": start})

    @property
    def n_seqs(self) -> int:
        return sum(len(s.seqs) for s in self.sets)

    def measure(self, sc: _Scorer, extra: list | None) -> dict:
        """介入(各組の介入 + extra の LoRA)を当てて読み、値を返す: a(例ごとの振る舞いの部分の対数確率の和の、一致の平均)・
        a_rest(書き込みたくない位置の 1 トークンあたり)・b・dt(群 → (1 トークンあたり, 冒頭))・probe(行 → 振る舞い − 望む答え)。"""
        from .reverse import combine
        out: dict = {"a": [], "a_rest": [], "b": [], "dt": {}, "probe": [None] * len(self.probe_rows)}
        plp = [{"b": 0.0, "d": 0.0} for _ in self.probe_rows]
        for S in self.sets:
            if not S.seqs:
                if S.label.startswith("dt:"):
                    out["dt"][S.label[3:]] = ([], [])
                continue
            vecs, lora = combine(S.iv, None, extra)
            got = sc.run(S.seqs, S.wants, vecs, lora)
            if S.label.startswith("dt:"):
                out["dt"][S.label[3:]] = ([], [])
            for m, g in zip(S.meta, got, strict=True):
                if m["kind"] == "a":
                    sp = [sum(g[pt] for pt in s) for s in m["spans"]]
                    out["a"].append(float(np.mean(sp)) if sp else None)
                    out["a_rest"].append(float(np.mean([g[pt] for pt in m["rest"]])) if m["rest"] else None)
                elif m["kind"] == "reply":
                    tok = float(np.mean([g[pt] for pt in m["toks"]]))
                    st = float(np.mean([g[pt] for pt in m["start"]]))
                    if S.label == "B":
                        out["b"].append(tok)
                    else:
                        out["dt"][S.label[3:]][0].append(tok)
                        out["dt"][S.label[3:]][1].append(st)
                else:
                    for which, pts in m["parts"]:
                        plp[m["row"]][which] += sum(g[pt] for pt in pts)
        for k in range(len(self.probe_rows)):
            out["probe"][k] = plp[k]["b"] - plp[k]["d"]
        return out


def _diff(x: list, b: list) -> np.ndarray:
    return np.asarray([np.nan if (u is None or v is None) else u - v for u, v in zip(x, b, strict=True)], dtype=np.float64)


def _effects(plan: _Plan, got: dict, base: dict) -> dict:
    """基準(iv だけ)との差の要約: reply(効き目)・probe(効き目)・b・dt(群ごと)・probe_dt(群ごと)。"""
    out: dict = {"reply": _stat(_diff(got["a"], base["a"])) if plan.regex else None}
    rows = plan.probe_rows
    pa = [k for k, r in enumerate(rows) if r["group"] is None]
    pd = _diff(got["probe"], base["probe"]) if rows else np.zeros(0)
    out["probe"] = _stat(pd[pa]) if pa else None
    out["b"] = _stat(_diff(got["a_rest"], base["a_rest"])) if plan.mode == "same" else _stat(_diff(got["b"], base["b"]))
    dt = {}
    for g, (tok, st) in got["dt"].items():
        s = _stat(_diff(tok, base["dt"][g][0]))
        if s is not None:
            s["start"] = _mean(_diff(st, base["dt"][g][1]))
        dt[g] = s
    out["dt"] = dt
    out["dt_min"] = min((v["mean"] for v in dt.values() if v), default=None)
    pdt = {}
    for g in dict.fromkeys(r["group"] for r in rows if r["group"]):
        ks = [k for k, r in enumerate(rows) if r["group"] == g]
        s = _stat(pd[ks])
        if s is not None:
            s["abs_mean"] = float(np.nanmean(np.abs(pd[ks])))
        pdt[g] = s
    out["probe_dt"] = pdt
    return out


def scale_ratio(alphas: list[float], effects: list[float | None], cis: list[float | None] | None = None) -> dict | None:
    """効き目が α に比例しているかの目安: いちばん大きい |α| での効き目 ÷ いちばん小さい |α| での効き目を、|α| の比で割ったもの
    (1 = 比例、1 より小さい = 頭打ち、大きい = α より速く伸びる)。向き(符号)が α の段でそろわなければ consistent: False。
    いちばん小さい |α| での効き目が 95% の幅(cis)の中なら、比は雑音の割り算になるので出さない(noisy: True)。"""
    cis = cis or [None] * len(alphas)
    pts = [(abs(a), e, c) for a, e, c in zip(alphas, effects, cis, strict=True) if e is not None]
    if len(pts) < 2:
        return None
    pts.sort(key=lambda x: x[0])
    (a0, e0, c0), (a1, e1, _) = pts[0], pts[-1]
    signs = {np.sign(e) for _, e, _ in pts if e != 0}
    out = {"consistent": len(signs) <= 1, "ratio": None, "noisy": bool(c0 is not None and abs(e0) <= c0)}
    if a1 > a0 and e0 != 0 and np.sign(e0) == np.sign(e1) and not out["noisy"]:
        out["ratio"] = float((e1 / e0) / (a1 / a0))
    return out


def drop_per_harm(effect: dict | None, side: dict | None, floor: float = SIDE_FLOOR) -> float | None:
    """確率の下がり方 ÷ 副作用。下がり方 = −(効き目の平均)(振る舞いの部分の対数確率が下がった量、正)、副作用 = −(書き込みたくない
    位置の 1 トークンあたりの対数確率の変化の平均)(下がった量)。副作用が floor より小さいとき(上がったときも)は floor で割る。
    確率が下がっていない(下がり方 ≤ 0)か、値が無ければ None。大きいほど、書き込みたくない場面を変えずに振る舞いを下げる。"""
    if not effect or effect.get("mean") is None:
        return None
    drop = -float(effect["mean"])
    if drop <= 0:
        return None
    harm = -float(side["mean"]) if side and side.get("mean") is not None else 0.0
    return float(drop / max(harm, float(floor)))


def candidates_from_rows(rows: list[dict], alphas: list[float], floor: float = SIDE_FLOOR) -> list[dict]:
    """候補 × α の行 → 1 か所(書き込み先 × 層)につき 1 つ: {target, layer, key, alphas, by_alpha: {α: {unit, reply, mark, probe,
    probe_mark, pseudo, probe_pseudo, side_b, dt_min, ratio}}, all_marked(返事の効き目がどの α でも印。効き目が無ければ None)、
    all_marked_probe、key_check、linear}。ratio は drop_per_harm(返事の効き目、無ければ probe の効き目)。行の順(書き込み先・層)のまま。"""
    out: dict[tuple, dict] = {}
    for r in rows:
        k = (r["target"], int(r["layer"]))
        c = out.setdefault(k, {"target": r["target"], "layer": int(r["layer"]), "key": r.get("key"), "alphas": [], "by_alpha": {},
                               "key_check": r.get("key_check"), "linear": r.get("linear")})
        a = float(r["alpha"])
        side = r.get("side") or {}
        eff = r.get("reply") or r.get("probe")
        c["alphas"].append(a)
        c["by_alpha"][f"{a:g}"] = {"unit": r.get("unit"), "reply": r.get("reply"), "mark": r.get("mark"), "probe": r.get("probe"),
                                   "probe_mark": r.get("probe_mark"), "pseudo": r.get("pseudo"), "probe_pseudo": r.get("probe_pseudo"),
                                   "side_b": side.get("b"), "dt_min": side.get("dt_min"),
                                   "pseudo_side_b": (r.get("pseudo_side") or {}).get("b"),
                                   "ratio": drop_per_harm(eff, side.get("b"), floor)}
    for c in out.values():
        for m, key in (("mark", "all_marked"), ("probe_mark", "all_marked_probe")):
            ms = [c["by_alpha"][f"{a:g}"].get(m) for a in alphas if f"{a:g}" in c["by_alpha"]]
            c[key] = None if not ms or any(x is None for x in ms) else all(bool(x) for x in ms)
    return list(out.values())


def summarize(rows: list[dict], alphas: list[float], *, chance: float = CHANCE) -> tuple[dict, dict]:
    """全候補の比較用の鍵の値を α ごとにまとめた分布(pool: {α: {reply / probe: {n, p5, p50, p95}}})と、その 5% 点より下がった
    候補の印(行の mark・probe_mark を書き込む)と数(marks: {α: {reply / probe: {candidates, marked, expected}}})。
    expected = 候補の数 × chance(偶然でもこのくらいは印が付く数)。"""
    pool: dict = {}
    marks: dict = {}
    for a in alphas:
        key = f"{a:g}"
        pool[key], marks[key] = {}, {}
        rs = [r for r in rows if r["alpha"] == a]
        for m, pk, mk in (("reply", "pseudo", "mark"), ("probe", "probe_pseudo", "probe_mark")):
            vals = [v for r in rs for v in ((r.get(pk) or {}).get("values") or []) if v is not None]
            if not vals:
                continue
            p5 = float(np.percentile(vals, 100 * chance))
            pool[key][m] = {"n": len(vals), "p5": p5, "p50": float(np.percentile(vals, 50)), "p95": float(np.percentile(vals, 95))}
            n_c = 0
            n_m = 0
            for r in rs:
                e = r.get(m)
                if not e:
                    r[mk] = None
                    continue
                n_c += 1
                r[mk] = bool(e["mean"] < p5)
                n_m += r[mk]
            marks[key][m] = {"candidates": n_c, "marked": n_m, "expected": round(n_c * chance, 2)}
    return pool, marks


# --- 返事を用意する(作成・key-check・測定で同じトークン列を読む)--------------------------------------------------

def _fill_replies(session, spec, mode: str, side_iv: dict, vecs0, lora0, max_tokens: int, tick) -> tuple[object, dict]:
    """対照集合の返事を、無いところだけ iv を当てて 1 回作り(greedy)、返事を入れた KeySpec を返す。鍵の作成(偶数番目)・
    key-check と測定(奇数番目)が同じ返事を読む。戻り値 (KeySpec, {組: given / generated})。"""
    from .evalx import _generate
    from .keyed_build import DontTouch
    from .reverse import _conv, combine
    cs = spec.contrast
    src: dict = {}
    replies = dict(spec.replies)

    def gen(convs, iv):
        if not convs:
            return []
        vecs, lora = iv
        outs = _generate(session, [(c, int(max_tokens)) for c in convs], vecs, lora, "all", lambda k: tick(k))
        return [o["text"] for o in outs]
    if mode == "same":
        have = spec.replies.get("same") or spec.replies.get("B")
        if have is None:
            have = gen(cs.convs("B"), combine(side_iv["B"], vecs0, lora0))
            src["AB"] = "generated"
        else:
            src["AB"] = "given"
        replies["same"] = list(have)
    else:
        for side in "AB":
            have = spec.replies.get(side)
            if have is None:
                have = gen(cs.convs(side), combine(side_iv[side], vecs0, lora0))
                src[side] = "generated"
            else:
                src[side] = "given"
            replies[side] = list(have)
    dts = []
    for dt in spec.dont_touch:
        if dt.replies is None:
            r = gen([_conv(dt.system, q) for q in dt.prompts], (vecs0, lora0))
            src[f"dt:{dt.name}"] = "generated"
        else:
            r = list(dt.replies)
            src[f"dt:{dt.name}"] = "given"
        dts.append(DontTouch(dt.name, dt.system, dt.prompts, r, dt.positions))
    return dataclasses.replace(spec, replies=replies, dont_touch=dts), src


def _n_replies_to_make(spec, mode: str) -> int:
    cs = spec.contrast
    n = 0
    if mode == "same":
        if (spec.replies.get("same") or spec.replies.get("B")) is None:
            n += len(cs.prompts("B"))
    else:
        n += sum(len(cs.prompts(s)) for s in "AB" if spec.replies.get(s) is None)
    n += sum(len(dt.prompts) for dt in spec.dont_touch if dt.replies is None)
    return n


def _built_questions(spec, mode: str) -> set[str]:
    """鍵の作成(偶数番目)に使った質問の文(probe の行を外す用)。"""
    from .direction import _subset
    from .reverse import question_text
    cs = spec.contrast
    out = set()
    for side in (("B",) if mode == "same" else ("A", "B")):
        ps = cs.prompts(side)
        out |= {question_text(ps[i]) for i in _subset(len(ps), "even")}
    for dt in spec.dont_touch:
        out |= {question_text(dt.prompts[i]) for i in _subset(len(dt.prompts), "even")}
    return out


def _validate_regex(spec, settings_regex: str | None) -> tuple[str | None, str]:
    from .keyed_build import parse_positions
    if settings_regex:
        try:
            if re.compile(settings_regex).fullmatch(""):
                raise GneError("BAD_ARGS", f"measure_match {settings_regex!r} が空の文字列に一致します")
        except re.error as e:
            raise GneError("BAD_ARGS", f"measure_match の正規表現が不正です: {e}") from e
        return settings_regex, "measure_match"
    pos = parse_positions(spec.positions or "response")
    if pos.kind == "before":
        return pos.pattern, "positions"
    return None, "none"


# --- 保存 ----------------------------------------------------------------------------------------------------

def causal_dir(session) -> Path:
    d = K.key_dir(session) / "causal"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _check_name(name: str, what: str = "名前") -> str:
    if not _NAME_RE.fullmatch(str(name or "")):
        raise GneError("BAD_ARGS", f"{what} {name!r} は英数字・_・-・. だけにしてください")
    return str(name)


def load_result(session, name: str | Path) -> dict:
    """保存した第 1 段の結果(名前か .json のパス)。"""
    p = Path(str(name))
    if not (p.suffix == ".json" and p.exists()):
        p = causal_dir(session) / f"{_check_name(str(name))}.json"
    if not p.exists():
        raise GneError("FILE_NOT_FOUND", f"key-causal の結果 {name} がありません", "gne direction key-causal で作ってください")
    return with_candidates(json.loads(p.read_text(encoding="utf-8")))


def list_results(session) -> list[dict]:
    """保存した第 1 段の結果の要約(新しい順)。第 2 段の結果(.verify.json)があればその名前も。"""
    d = causal_dir(session)
    out = []
    for p in sorted(d.glob("*.json"), key=lambda q: -q.stat().st_mtime):
        if p.name.endswith(".verify.json"):
            continue
        try:
            x = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(x, dict) or "rows" not in x:
            continue
        v = d / f"{p.stem}.verify.json"
        out.append({"name": x.get("name") or p.stem, "contrast": x.get("contrast"), "direction": x.get("direction"),
                    "targets": x.get("targets"), "alphas": x.get("alphas"), "n_candidates": x.get("n_candidates"),
                    "marks": x.get("marks"), "updated": p.stat().st_mtime, "verify": v.exists()})
    return out


# --- 第 1 段 -------------------------------------------------------------------------------------------------

def key_causal(session, contrast, *, layers=None, targets=None, alphas=None, pseudo: int = DEFAULT_PSEUDO, iv=None,
               no_iv: bool = False, direction: str | None = None, rebuild_direction: bool = False, response_tokens: int = 96,
               measure_match: str | None = None, probe=None, max_tokens: int | None = None, side_tokens: int = SIDE_TOKENS,
               name: str | None = None, q_mode: str | None = None, seed: int = 0, dont_touch_start: bool | None = None,
               keep_rows: bool = True, progress: Progress | None = None) -> dict:
    """第 1 段: 生成なしのふるい分け(モジュールの説明)。contrast は鍵の対照集合の表かパス(表は <root>/contrasts/<name>.yaml に
    書く)。layers を省くと全層、targets の既定は ffn,attn(書けない書き込み先・層は外す)、alphas の既定は (1, 2, 4) に表の
    keyed.alpha の符号を掛けたもの、iv の既定は表の keyed.iv(no_iv で当てない)、direction の既定は表の keyed.direction、
    無ければ対照集合の name(無ければ作る。keyed.direction の方向は作らない)。鍵は書き込み先ごとに <name>-<ffn|attn|S|Ee>
    (name の既定は <対照集合の名前>-causal)で保存する(第 2 段と「+ レシピ」で使う)。"""
    from .keyed_build import _resolve_mode, build_keys, intervention, key_check, parse_positions, read_key_spec
    from .keyed_pipeline import _Stages, direction_plan, make_direction, resolve, reused_direction
    from .reverse import side_interventions
    t0 = time.perf_counter()
    progress = progress if progress is not None else session.progress
    warnings: list[str] = []
    notes: list[str] = []

    def warn(msg: str) -> None:
        if msg not in warnings:
            warnings.append(msg)
            session.warn(msg)
    pseudo = int(pseudo)
    if pseudo < 1:
        raise GneError("BAD_ARGS", "pseudo(候補ごとの比較用の鍵の本数)は 1 以上です")
    side_tokens = int(side_tokens)
    if side_tokens < 1:
        raise GneError("BAD_ARGS", "side_tokens は 1 以上です")
    path, settings = resolve(session, contrast)
    spec = read_key_spec(path)
    cs = spec.contrast
    name = _check_name(name or f"{cs.name}-causal")
    probe_rows = read_probe(probe)
    regex, regex_src = _validate_regex(spec, measure_match)
    pos_a = parse_positions(spec.positions or "response")
    mode, pos_b = _resolve_mode(pos_a, spec.positions_b)
    if regex is None and not probe_rows:
        raise GneError("BAD_ARGS", "効き目を測る正規表現がありません(対照集合の positions が before:<正規表現> でないとき)",
                       "measure_match に振る舞いの部分の正規表現を渡すか、probe(質問ごとの答えの組)を渡してください")
    sign = float(np.sign(settings.get("alpha") or 1.0) or 1.0)
    alist = parse_alphas(alphas) or [sign * a for a in DEFAULT_ALPHAS]
    explicit_targets = targets is not None
    tlist = parse_targets(targets) if explicit_targets else ["ffn", "attn"]
    if not tlist:
        raise GneError("BAD_ARGS", "書き込み先(targets)を 1 つ以上指定してください(ffn / attn / shared / expert:e)")
    from .jobs import _layers
    Ls_req = _layers(layers) if layers not in (None, "", "all") else None
    max_tokens = int(max_tokens or settings.get("max_tokens") or 96)
    q_mode = q_mode or settings.get("q_mode") or "examples"
    iv_spec = None if no_iv else (iv if iv not in (None, "", []) else settings.get("iv"))
    # 方向(無ければ作る。表の keyed.direction は作らない)
    dname, fixed, existing, build_dir = direction_plan(session, spec, settings, direction, rebuild_direction)
    vecs0, lora0, iv_desc = intervention(session, iv_spec, warn)
    side_iv = side_interventions(session, cs) if cs.has_recipes else {"A": (None, None), "B": (None, None)}
    # 書き込み先と候補の層(書けない書き込み先は、既定の ffn,attn なら外して知らせる)
    from .keyed_build import target_capture
    cands: dict[str, list[int]] = {}
    dropped: list[dict] = []
    for tg in tlist:
        try:
            Ls, sk = candidate_layers(session, None if build_dir else existing, tg, Ls_req)   # 方向をこれから作るなら層だけで
            if Ls:
                target_capture(session, tg, Ls)
        except GneError as ex:
            if explicit_targets:
                raise
            warn(f"書き込み先 {tg} は使えないので外しました: {ex.message}")
            continue
        dropped += sk
        if not Ls:
            if explicit_targets:
                warn(f"書き込み先 {tg} に書ける層がありません")
            continue
        cands[tg] = Ls
    if not cands:
        raise GneError("BAD_ARGS", "候補の層がありません(書き込み先と層を確かめてください)",
                       "MoE の層は shared か expert:e、dense の層は ffn・attn。最終層と MTP の層は使えません")
    n_cand = sum(len(v) for v in cands.values())
    labels = (["方向"] if build_dir else []) + (["返事"] if _n_replies_to_make(spec, mode) else []) \
        + [x for tg in cands for x in (f"鍵 {target_tag(tg)}", f"key-check {target_tag(tg)}")] + ["測る"]
    st = _Stages(progress, labels)
    si = 0
    prev = session.progress
    timing: dict = {}
    out: dict = {"name": name, "contrast": cs.name, "contrast_path": str(path), "contrast_sha": cs.sha, "mode": mode,
                 "positions": pos_a.text, "positions_b": pos_b.text, "measure": {"regex": regex, "source": regex_src},
                 "alphas": alist, "targets": list(cands), "pseudo": pseudo, "side_tokens": side_tokens, "max_tokens": max_tokens,
                 "iv": iv_desc, "iv_spec": iv_spec, "q_mode": q_mode, "stages": labels}
    try:
        # 1. 方向
        if build_dir:
            session.progress = st.start(si)
            si += 1
            out["direction_info"] = make_direction(session, path, dname, existing, response_tokens)
        else:
            out["direction_info"], w = reused_direction(existing, spec, dname, fixed)
            for x in w:
                warn(x)
        out["direction"] = dname
        ds = session.direction_set(dname)
        if build_dir:                                                 # 作った方向に無い層は外す
            for tg in list(cands):
                keep = [L for L in cands[tg] if L in ds.raw]
                dropped += [{"target": tg, "layer": L, "error": f"方向 {dname} に層 {L} がありません"} for L in cands[tg] if L not in ds.raw]
                cands[tg] = keep
        # 2. 返事(無いところだけ、iv を当てて 1 回)
        ts = time.perf_counter()
        n_make = _n_replies_to_make(spec, mode)
        if n_make:
            f = st.start(si)
            si += 1
            done = [0]

            def tick(k: int) -> None:
                done[0] += k
                if f:
                    f(min(done[0], n_make), n_make, "返事")
            spec2, rsrc = _fill_replies(session, spec, mode, side_iv, vecs0, lora0, max_tokens, tick)
        else:
            spec2, rsrc = _fill_replies(session, spec, mode, side_iv, vecs0, lora0, max_tokens, lambda k: None)
        out["replies"] = rsrc
        timing["replies"] = time.perf_counter() - ts
        # 3. 鍵(書き込み先ごとに全候補の層)と key-check(参考)
        ts = time.perf_counter()
        keys: dict[str, str] = {}
        kc: dict = {}
        built: dict[str, list[int]] = {}
        for tg, Ls in cands.items():
            kname = f"{name}-{target_tag(tg)}"
            f = st.start(si)
            si += 1
            session.progress = f
            if not Ls:
                si += 1
                continue
            try:
                r = build_keys(session, spec2, name=kname, layers=Ls, max_tokens=max_tokens, q_mode=q_mode, subset="even",
                               pseudo=max(pseudo, KC_PSEUDO), iv=iv_spec, target=tg, seed=seed,
                               dont_touch_start=dont_touch_start, progress=f)
            except GneError as ex:
                if ex.code != "BAD_KEY":
                    raise
                warn(f"書き込み先 {tg} はどの層でも鍵にならなかったので外しました: {ex.message}")
                dropped += [{"target": tg, "layer": L, "error": ex.message} for L in Ls]
                si += 1
                continue
            for x in r.get("warnings") or []:                     # build_keys が session.warn 済み。結果にだけ書き込み先を付けて残す
                if f"{target_tag(tg)}: {x}" not in warnings:
                    warnings.append(f"{target_tag(tg)}: {x}")
            for d0 in r.get("dropped") or []:
                dropped.append({"target": tg, "layer": int(d0["layer"]), "error": d0["error"]})
            built[tg] = [x["layer"] for x in r["layers"]]
            keys[tg] = kname
            f = st.start(si)
            si += 1
            session.progress = f
            c = key_check(session, kname, contrast=spec2, subset="odd", iv=iv_spec, progress=f)
            kc[tg] = {"held_out": c["held_out"], "warnings": c.get("warnings") or [],
                      "layers": {str(x["layer"]): {"selectivity": x["selectivity"],
                                                   "p95": (x.get("pseudo_selectivity") or {}).get("p95"),
                                                   "beats": x.get("beats_pseudo"), "kn": x.get("kn"), "auc": x.get("auc")}
                                 for x in c["layers"]}}
        timing["keys"] = time.perf_counter() - ts
        cands = built
        if not cands:
            raise GneError("BAD_KEY", "どの候補でも鍵を作れませんでした", "q_mode を examples / means にするか、例を増やしてください")
        out["keys"] = keys
        out["key_check"] = kc
        # 4. 測る(基準 = iv だけ、候補 × α × (本物 + 比較用の鍵))
        ts = time.perf_counter()
        f = st.start(si)
        session.progress = f
        skip = int(spec.skip or 0)
        plan = _Plan(session, spec2, mode=mode, regex=regex, skip=skip, side_tokens=side_tokens, probe_rows=probe_rows,
                     side_iv=side_iv, vecs0=vecs0, lora0=lora0, built_questions=_built_questions(spec, mode))
        notes += plan.notes
        out["counts"] = plan.counts
        if regex and plan.counts["a"]["matched"] < 2:
            warn(f"作成に使っていない例で、効き目を測る正規表現 {regex} に一致した返事が {plan.counts['a']['matched']} 個だけです"
                 "(効き目の値が数例で揺れます。質問を増やすか、正規表現を見直してください)")
        if regex and not plan.counts["a"]["matched"]:
            raise GneError("BAD_CONTRAST", f"作成に使っていない例の返事に、効き目を測る正規表現 {regex} が一致しません",
                           "measure_match を返事に出てくる言い方に合わせるか、一致する返事を対照集合の replies で与えてください")
        if probe_rows and not plan.probe_rows:
            warn("probe の行がすべて鍵の作成に使った質問だったので、probe は測りませんでした")
        n_sign = len({np.sign(a) for a in alist})
        n_cand = sum(len(v) for v in cands.values())
        n_runs = 1 + n_cand * len(alist) * (1 + pseudo)
        total = 1 + n_cand * n_sign * (1 + pseudo)                    # 進み具合は LoRA ごと(α の段は同じ LoRA の倍率)
        sc = _Scorer(session)
        done = 0
        if f:
            f(0, total, "基準(iv だけ)")
        base = plan.measure(sc, None)
        done += 1
        out["base"] = {"reply": _mean(base["a"]), "b": _mean(base["a_rest"] if mode == "same" else base["b"]),
                       "dt": {g: _mean(v[0]) for g, v in base["dt"].items()},
                       "probe": _mean([v for v, r in zip(base["probe"], plan.probe_rows, strict=True) if r["group"] is None])}
        from .causal import TmpLora
        tmp = TmpLora(session, K.key_dir(session) / "tmp")
        rows: list[dict] = []
        t_run = []
        for tg, Ls in cands.items():
            kname = keys[tg]
            kind, ex = K.parse_target(tg)
            for L in Ls:
                variants = [None] + list(range(pseudo))
                per: dict = {}                                        # (variant, α) → 効き目の要約
                for j in variants:
                    kspec = kname if j is None else f"{kname}~{j}"
                    for sgn in sorted({float(np.sign(a)) for a in alist}):
                        u = K.KeyedUnit(int(L), sgn, dname, kspec, kind, ex)
                        tens = K.keyed_tensors(session.weights, [u], session._key_of, session.direction_set)
                        with tmp.use(tens) as lora:
                            for a in alist:
                                if np.sign(a) != sgn:
                                    continue
                                t1 = time.perf_counter()
                                got = plan.measure(sc, [(lora[0][0], abs(a))])
                                per[(j, a)] = _effects(plan, got, base)
                                t_run.append(time.perf_counter() - t1)
                        done += 1
                        if f:
                            f(min(done, total), total, f"{target_tag(tg)} L{L}" + ("" if j is None else f" 比較用の鍵 {j}"))
                main = [per[(None, a)]["reply"] or per[(None, a)]["probe"] or {} for a in alist]
                lin = scale_ratio(alist, [m.get("mean") for m in main], [m.get("ci") for m in main])
                kcl = (kc.get(tg) or {}).get("layers", {}).get(str(L)) or {}
                for a in alist:
                    e0 = per[(None, a)]
                    ps = [per[(j, a)] for j in range(pseudo)]
                    pr = [(x["reply"] or {}).get("mean") for x in ps]
                    pp = [(x["probe"] or {}).get("mean") for x in ps]
                    sel, p95 = kcl.get("selectivity"), kcl.get("p95")
                    row = {"target": tg, "layer": int(L), "alpha": a, "key": kname, "unit": unit_spec(L, tg, a, dname, kname),
                           "reply": e0["reply"], "pseudo": _range(pr) if regex else None,
                           "probe": e0["probe"], "probe_pseudo": _range(pp) if plan.probe_rows else None,
                           "side": {"b": e0["b"], "dt": e0["dt"], "dt_min": e0["dt_min"], "probe_dt": e0["probe_dt"]},
                           "pseudo_side": {"dt_min": _range([x["dt_min"] for x in ps]),
                                           "b": _range([(x["b"] or {}).get("mean") for x in ps])},
                           "key_check": {"selectivity": sel, "p95": p95,
                                         "margin": None if sel is None or p95 is None else sel - p95,
                                         "beats": kcl.get("beats"), "kn": kcl.get("kn")},
                           "linear": lin}
                    rows.append(row)
        timing["measure"] = time.perf_counter() - ts
        timing["per_run"] = float(np.mean(t_run)) if t_run else None
    finally:
        session.progress = prev
    pool, marks = summarize(rows, alist)
    cand_rows = candidates_from_rows(rows, alist)
    for a, pm in pool.items():
        for m, v in pm.items():
            if v["n"] < MIN_POOL:
                notes.append(f"α {a} の{'返事' if m == 'reply' else ' probe '}の比較用の鍵は {v['n']} 本だけなので、5% 点は粗い値です"
                             "(候補の層を増やすか、pseudo を増やしてください)")
    out.update({"direction": dname, "targets": list(cands), "n_candidates": n_cand, "n_runs": n_runs, "rows": rows if keep_rows else [],
                "candidates": cand_rows if keep_rows else [], "ratio_floor": SIDE_FLOOR,
                "pool": pool, "marks": marks, "chance": CHANCE, "dropped": dropped,
                "notes": [NOTE] + notes, "warnings": warnings, "timing": timing, "rows_read": sc.rows,
                "seconds": time.perf_counter() - t0})
    p = causal_dir(session) / f"{name}.json"
    p.write_text(json.dumps({**out, "rows": rows, "candidates": cand_rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    out["path"] = str(p)
    return out


def with_candidates(d: dict) -> dict:
    """保存した第 1 段の結果に、1 か所 1 行のまとめ(candidates)が無ければ足す(この形を入れる前の結果を読むとき)。"""
    if d.get("rows") and not d.get("candidates"):
        d["candidates"] = candidates_from_rows(d["rows"], [float(a) for a in d.get("alphas") or sorted({r["alpha"] for r in d["rows"]})])
        d.setdefault("ratio_floor", SIDE_FLOOR)
    return d


def _range(vals: list) -> dict | None:
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return None
    return {"values": [float(x) for x in vals], "min": float(min(v)), "max": float(max(v))}


# --- 第 2 段 -------------------------------------------------------------------------------------------------

def _counts(replies: list[dict], rx, labels_b: list[str] | None) -> dict:
    """返事のリスト(behave の replies の形)→ {n, behavior(振る舞いが出た数), match, hits(一致の総数), labels, loop, empty, at_limit}。"""
    out = {"n": len(replies), "loop": sum(bool(r.get("loop")) for r in replies),
           "empty": sum(not (r.get("text") or "").strip() for r in replies),
           "at_limit": sum(r.get("finish") not in ("eog", "stop", "eos") for r in replies)}
    if rx is not None:
        out["match"] = sum(bool(rx.search(r.get("text") or "")) for r in replies)
        out["hits"] = sum(len(rx.findall(r.get("text") or "")) for r in replies)
    if any("label" in r for r in replies):
        labs: dict[str, int] = {}
        for r in replies:
            labs[r.get("label", "N")] = labs.get(r.get("label", "N"), 0) + 1
        out["labels"] = labs
    if labels_b:
        out["behavior"] = sum(r.get("label") in labels_b for r in replies)
    elif rx is not None:
        out["behavior"] = out["match"]
    return out


def _changed(replies: list[dict], base: list[dict]) -> tuple[int, float | None]:
    """iv だけのときの返事(base)から変わった返事の数と、先頭から同じ部分の割合の平均(iv だけの返事の文字数に対する、
    先頭から一致した文字数。1 = 変わらない)。"""
    n, fr = 0, []
    for a, b in zip(replies, base, strict=True):
        x, y = a.get("text") or "", b.get("text") or ""
        n += x != y
        k = 0
        for c1, c2 in zip(x, y):
            if c1 != c2:
                break
            k += 1
        fr.append(k / len(y) if y else float(x == y))
    return n, (float(np.mean(fr)) if fr else None)


def parse_combos(combos) -> list[list[str]]:
    """組の指定 → ユニットの書き方のリストのリスト。1 つの組はリストか、カンマ・空白で区切った文字列(「11:attn:keyed:a2,18:attn:keyed:a2」)。"""
    if combos is None:
        return []
    if isinstance(combos, str):
        combos = [c for c in combos.split(";") if c.strip()]
    out = []
    for c in combos:
        us = [u for u in re.split(r"[\s,]+", c) if u] if isinstance(c, str) else [str(u).strip() for u in c if str(u).strip()]
        if len(us) < 2:
            raise GneError("BAD_ARGS", f"組には 2 つ以上のユニットが要ります({c!r})。1 つなら units で渡してください")
        out.append(us)
    return out


def key_causal_verify(session, units, *, result=None, contrast=None, iv=None, no_iv: bool = False,
                      direction: str | None = None, pseudo_index: int = 0, match: str | None = None,
                      judge: str | None = None, labels: str = "R,P,C", label=None, max_tokens: int = VERIFY_TOKENS,
                      thinking: bool = False, temperature: float = 0.0, seeds=None, dont_touch: bool = True,
                      name: str | None = None, combos=None, progress: Progress | None = None) -> dict:
    """第 2 段: 選んだ候補(keyed のユニットの書き方のリスト)を生成で確かめる(モジュールの説明)。result は第 1 段の結果の名前
    (対照集合・iv・方向・鍵・測った正規表現の既定に使う)。ユニットの @方向・#鍵 を省くと、第 1 段の方向と書き込み先の鍵。
    比較用の鍵は同じユニットの鍵を NAME~pseudo_index にしたもの。判定は behave と同じ: 正規表現 match(既定は第 1 段で測った
    正規表現、無ければ対照集合の validate.match)か、judge(文面)と label(振る舞いとみなすラベル)。
    combos は複数のユニットを 1 つの候補として確かめる組(parse_combos。強さはユニットの書き方どおりで、組の中で違ってよい)。
    組の比較用の鍵は、組の各ユニットの鍵を同じ番号の比較用の鍵(NAME~pseudo_index)に替えた組。"""
    import yaml

    from .behave import behave
    from .keyed_build import _resolve_mode, intervention, parse_positions, read_key_spec
    from .keyed_pipeline import _Stages, resolve
    from .reverse import _conv, combine, side_interventions
    from .steer import parse_unit
    t0 = time.perf_counter()
    progress = progress if progress is not None else session.progress
    warnings: list[str] = []

    def warn(msg: str) -> None:
        if msg not in warnings:
            warnings.append(msg)
            session.warn(msg)
    if isinstance(labels, (list, tuple)):
        labels = ",".join(str(x) for x in labels)
    res = load_result(session, result) if result else None
    if contrast is None and res is None:
        raise GneError("BAD_ARGS", "対照集合(contrast)か第 1 段の結果(result)を渡してください")
    if contrast is None:
        contrast = res["contrast_path"]
        if not Path(str(contrast)).exists():
            raise GneError("FILE_NOT_FOUND", f"第 1 段の対照集合 {contrast} がありません", "contrast で渡してください")
    path, settings = resolve(session, contrast)
    spec = read_key_spec(path)
    cs = spec.contrast
    if res is not None and res.get("contrast_sha") and res["contrast_sha"] != cs.sha:
        warn(f"対照集合 {cs.name} は、第 1 段({res.get('name')})のときと中身が違います")
    if isinstance(units, str):
        units = [u for u in re.split(r"[\s;]+", units) if u]
    units = list(units or [])
    combo_list = parse_combos(combos)
    if not units and not combo_list:
        raise GneError("BAD_ARGS", "確かめる候補(keyed のユニット、例 14:ffn:keyed:a-2@方向#鍵)を 1 つ以上渡してください")
    dname = direction or (res or {}).get("direction") or settings.get("direction") or cs.name
    keys_of = (res or {}).get("keys") or {}
    iv_spec = None if no_iv else (iv if iv not in (None, "", []) else ((res or {}).get("iv_spec") if res else settings.get("iv")))
    vecs0, lora0, iv_desc = intervention(session, iv_spec, warn)
    side_iv = side_interventions(session, cs) if cs.has_recipes else {"A": (None, None), "B": (None, None)}
    pos_a = parse_positions(spec.positions or "response")
    mode, _ = _resolve_mode(pos_a, spec.positions_b)
    def resolve_unit(spec_u: str) -> dict:
        u = parse_unit(str(spec_u))
        if u.mode != "keyed":
            raise GneError("BAD_ARGS", f"{spec_u!r} は keyed のユニットではありません(例 14:ffn:keyed:a-2@方向#鍵)")
        tg = K.target_of(u.kind, u.expert)
        key = u.key or keys_of.get(tg)
        if not key:
            raise GneError("BAD_ARGS", f"{spec_u}: 鍵がありません(#鍵 を書くか、第 1 段の結果 result を渡してください)")
        base_key, j0 = K.split_key_spec(key)
        if j0 is not None:
            raise GneError("BAD_ARGS", f"{spec_u}: 確かめる候補には本物の鍵を書いてください(比較用の鍵 {key} は自動で足します)")
        ks = session._key_of(base_key)
        if ks.n_pseudo <= int(pseudo_index):
            raise GneError("BAD_ARGS", f"鍵 {base_key} に比較用の鍵 {pseudo_index} がありません(比較用の鍵 {ks.n_pseudo} 本)")
        d = u.direction or dname
        return {"unit": unit_spec(u.layer, tg, u.k, d, base_key),
                "pseudo_unit": unit_spec(u.layer, tg, u.k, d, f"{base_key}~{int(pseudo_index)}"),
                "layer": int(u.layer), "target": tg, "alpha": float(u.k), "key": base_key, "direction": d}
    cands = []
    for spec_u in units:                                      # 1 か所の候補
        c = resolve_unit(spec_u)
        cands.append({**c, "combo": False, "units": [c["unit"]], "pseudo_units": [c["pseudo_unit"]]})
    for cu in combo_list:                                     # 組(複数のユニットを 1 つの候補として)
        rs = [resolve_unit(x) for x in cu]
        if len({(r["layer"], r["target"]) for r in rs}) < len(rs):
            raise GneError("BAD_ARGS", f"組に同じ層・書き込み先のユニットが重なっています({', '.join(cu)})")
        cands.append({"combo": True, "units": [r["unit"] for r in rs], "pseudo_units": [r["pseudo_unit"] for r in rs],
                      "unit": " + ".join(r["unit"] for r in rs), "pseudo_unit": " + ".join(r["pseudo_unit"] for r in rs),
                      "label": f"{'・'.join(sorted({target_tag(r['target']) for r in rs}))} "
                               f"L{'・L'.join(str(r['layer']) for r in rs)}"
                               + (f" α {rs[0]['alpha']:g}" if len({r['alpha'] for r in rs}) == 1 else ""),
                      "layer": None, "target": None, "alpha": None, "key": None, "direction": rs[0]["direction"]})
    # 判定
    labs_b = [x.strip() for x in (label.split(",") if isinstance(label, str) else (label or [])) if str(x).strip()] or None
    if judge and not labs_b:
        raise GneError("BAD_ARGS", "judge を使うときは、振る舞いとみなすラベル label(例 P)を渡してください")
    if not match and not judge:
        match = ((res or {}).get("measure") or {}).get("regex")
        msrc = "key-causal"
        if not match:
            try:
                raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
                match = ((raw.get("validate") or {}).get("match")) if isinstance(raw, dict) else None
            except (OSError, yaml.YAMLError):
                match = None
            msrc = "validate"
        if not match and pos_a.kind == "before":
            match, msrc = pos_a.pattern, "positions"
        if not match:
            raise GneError("BAD_ARGS", "振る舞いを数える正規表現がありません", "match に正規表現を渡すか、judge と label を渡してください")
    else:
        msrc = "match" if match else "judge"
    try:
        rx = re.compile(match) if match else None
    except re.error as e:
        raise GneError("BAD_ARGS", f"match の正規表現が不正です: {e}") from e
    # 質問: 書き込みたい場面(作成に使っていない奇数番目)と dont_touch の奇数番目
    from .direction import _subset
    if mode == "same":
        pa = cs.prompts("B")
        a_convs = [_conv(cs.system("B"), pa[i]) for i in _subset(len(pa), "odd")]
        a_iv = combine(side_iv["B"], vecs0, lora0)
    else:                                                     # system の状態の鍵: 推論時と同じく system なしで答えさせる
        pa = cs.prompts("A")
        a_convs = [_conv(None, pa[i]) for i in _subset(len(pa), "odd")]
        a_iv = (vecs0, lora0)
    dt_convs, dt_group = [], []
    if dont_touch:
        for dt in spec.dont_touch:
            for i in _subset(len(dt.prompts), "odd"):
                dt_convs.append(_conv(dt.system, dt.prompts[i]))
                dt_group.append(dt.name)
    if not a_convs:
        raise GneError("BAD_CONTRAST", "作成に使っていない奇数番目の質問がありません(質問を 2 つ以上にしてください)")
    conds = [("base", None)] + [(f"c{k}", c["units"]) for k, c in enumerate(cands)] + [(f"p{k}", c["pseudo_units"]) for k, c in enumerate(cands)]
    st = _Stages(progress, [f"{i + 1}/{len(conds)}" for i in range(len(conds))])
    seeds = [int(s) for s in (seeds or [0])]
    prev = session.progress
    gens: dict[str, dict] = {}
    try:
        for ci, (cid, us) in enumerate(conds):
            f = st.start(ci)
            extra = None
            if us is not None:
                kp, ref = session.keyed_lora(session.recipe(units=list(us), name="key-causal-verify"))
                extra = [(str(kp), float(ref))]
            parts = {}
            n_parts = 2 if dt_convs else 1
            for pi, (seg, convs, ivp) in enumerate((("a", a_convs, a_iv), ("dt", dt_convs, (vecs0, lora0)))):
                if not convs:
                    continue
                vecs, lora = combine(ivp, None, extra)

                def sub(done, total, stage="", pi=pi):
                    if f:
                        f(int(1000 * pi + 1000 * (done / total if total else 0)), 1000 * n_parts,
                          ("iv のみ" if us is None else " + ".join(us)) + (f"({stage})" if stage else ""))
                d = behave(session, convs, lora=lora, vectors=vecs, match=match, judge=judge, labels=labels,
                           max_tokens=int(max_tokens), thinking=thinking, temperature=float(temperature), seeds=seeds,
                           progress=sub)
                parts[seg] = d["runs"]
            gens[cid] = parts
    finally:
        session.progress = prev

    def seg_replies(cid: str, seg: str, group: str | None = None) -> list[dict]:
        out = []
        for run in gens[cid].get(seg) or []:
            for r in run["replies"]:
                if group is None or dt_group[r["i"]] == group:
                    out.append(r)
        return out
    greedy = float(temperature) <= 0
    groups = list(dict.fromkeys(dt_group))
    base = {"a": _counts(seg_replies("base", "a"), rx, labs_b),
            "dt": {g: _counts(seg_replies("base", "dt", g), rx, labs_b) for g in groups}}
    out_rows = []
    for k, c in enumerate(cands):
        row = {**c, "a": {}, "dt": {}}
        for which, cid in (("keyed", f"c{k}"), ("pseudo", f"p{k}")):
            ra = seg_replies(cid, "a")
            cnt = _counts(ra, rx, labs_b)
            cnt["changed"], cnt["same_prefix"] = _changed(ra, seg_replies("base", "a")) if greedy else (None, None)
            row["a"][which] = cnt
            for g in groups:
                rg = seg_replies(cid, "dt", g)
                cg = _counts(rg, rx, labs_b)
                cg["changed"], cg["same_prefix"] = _changed(rg, seg_replies("base", "dt", g)) if greedy else (None, None)
                row["dt"].setdefault(g, {})[which] = cg
        row["samples"] = {which: [(r.get("text") or "")[:200] for r in seg_replies(cid, "a")[:2]]
                          for which, cid in (("base", "base"), ("keyed", f"c{k}"), ("pseudo", f"p{k}"))}
        out_rows.append(row)
    name = _check_name(name or ((res or {}).get("name") if res else None) or f"{cs.name}-causal")
    out = {"name": name, "result": (res or {}).get("name"), "contrast": cs.name, "contrast_path": str(path), "mode": mode,
           "direction": dname, "iv": iv_desc, "iv_spec": iv_spec,
           "judge": {"method": "label" if labs_b else "regex", "match": match, "source": msrc, "labels": labs_b,
                     "judge": bool(judge)},
           "n_a": len(a_convs), "n_dt": {g: dt_group.count(g) for g in groups}, "seeds": seeds, "temperature": float(temperature),
           "max_tokens": int(max_tokens), "greedy": greedy, "base": base, "rows": out_rows,
           "notes": ["作成に使っていない奇数番目の質問で数えました。比較用の鍵(ラベルをシャッフルして作った鍵)でも同じくらい変わるなら、"
                     "その変化は鍵が場面を分けたからではなく、書き込んだこと自体によるものです"]
           + ([] if greedy else ["温度があるので、返事が変わった数(changed)は数えていません"])
           + (["system の状態の鍵(作り方 (a))なので、書き込みたい場面の質問は system なし(推論時と同じ)で答えさせました"] if mode == "system" else []),
           "warnings": warnings, "seconds": time.perf_counter() - t0}
    p = causal_dir(session) / f"{name}.verify.json"
    full = {**out, "replies": {cid: {seg: [{"seed": run["seed"], "replies": run["replies"]} for run in runs]
                                     for seg, runs in parts.items()} for cid, parts in gens.items()},
            "conditions": {cid: us for cid, us in conds}, "dt_group": dt_group}
    p.write_text(json.dumps(full, ensure_ascii=False, indent=1), encoding="utf-8")
    out["path"] = str(p)
    return out
