"""鍵つきの書き込み(keyed write、CHANGE-07 E)の評価: 選択性(E.1)と擬似鍵の対照(E.2)。

gne eval は keyed のユニットを持つレシピで、能力・ppl・言語混入の行(evalx)に加えて次を測る(生成なし、prefill だけ):

- 物差し: 鍵の対照集合(鍵のファイルには対照集合のパスが無いので evaluate の keyed_contrast で渡す。省くと方向を作った
  対照集合が鍵と同じもの(contrast_sha が同じ)ならそれ)を、鍵の作成と反対側の subset(even ↔ odd)で読み、鍵の作成の
  位置で群に分ける。側 A = 発火させたい位置(before:<re> など)、側 B = 同じ返事のそれ以外の位置(not-before:<re>)か
  system_b の文脈、dont_touch の群。各群の位置で、keyed の層の残差 l_out-L の介入前後の差 Δ の、方向 r̂_L に沿った成分
  (par、符号つき)・大きさ ‖Δ‖(norm)・直交成分(orth)を、例ごとに位置で平均してから例で平均する。返事は対照集合に
  書いたものか、無介入で 1 回だけ生成したものを、どの介入でも同じ列として読む。
- 比べる方式(目的の層 = 最も後ろの keyed の層で、側 A の par を keyed にそろえる):
  keyed(レシピの keyed のユニットだけ)/ add_same_alpha(同じ層・同じ α の L:dir:NAME:α、raw)/ add(同じ向きの足し算を
  全体の倍率 β で合わせたもの)/ gain(層ごとに、側 A での寄与 μ_n·(r̂ᵀ W_down[:, n]) の大きい順に、β = GAIN_AIM で
  keyed と同じ大きさを書ける数(GAIN_NEURONS の範囲)のニューロンを k = 1 ± β)/ proj(keyed の層の全ヘッドの r̂ 成分を
  s = ∓β)。β は match_strength(外挿と補間の探索、MATCH_STEPS 回まで)で探す。上限まで強めても届かない方式は
  matched: false のまま値を出す(比べるときは側 A を 1 押すあたりの側 B の乱れ b_over_a で見る)。
  理想は「keyed は側 A を対照と同じだけ動かし、側 B・dont_touch はずっと少なく動かす」。
- 擬似鍵(E.2): 鍵を擬似鍵 NAME~j(例の単位でラベルを釣り合わせて並べ替えた鍵)に替えた keyed の選択性
  (書く向きにそろえた側 A − 側 B の par)の分布と、本物の鍵の選択性。p95 を超えなければ notes に書く。

gain・proj の対照は β = 1 の LoRA を 1 本作り、アダプタの倍率 β で当てる(どちらも ΔW が強さに線形なので、その強さで
作った LoRA と同じ)。記録の recipe は合わせた強さで書いたもの。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .steer import GneError, Recipe, Unit

DEFAULT_EXAMPLES = 8          # 会話の組(側 A・B の会話、dont_touch の群)ごとに読む会話の数(多ければ等間隔に間引く)
DEFAULT_PSEUDO = 8            # 選択性を測る擬似鍵の本数(E.2、鍵の擬似鍵の本数まで)
MATCH_STEPS = 6               # 大きさ合わせで 1 方式に使う側 A の測定の回数の上限
MATCH_RTOL = 0.1              # 側 A の par を keyed の値の ±10% にそろえる
GAIN_NEURONS = (16, 512)      # gain の対照: 層ごとのニューロンの数の下限と上限
GAIN_AIM = 4.0                # gain の対照のニューロンは、β = GAIN_AIM で keyed と同じ大きさ(|α|·‖r_L‖)を書ける数だけ選ぶ
MATCH_HI = {"add": 8.0, "gain": 16.0, "gain_down": 2.0, "proj": 100.0}  # β の上限(gain_down は打ち消す向きの k = 1 − β、proj は s の上限)
MIN_TARGET = 0.05             # keyed の側 A の par が |α|·‖r_L‖ のこの割合より小さければ大きさを合わせない


# --- 大きさ合わせ ----------------------------------------------------------------------------------------

def match_strength(f: Callable[[float], float], target: float, *, start: float = 1.0, hi: float = 8.0,
                   f0: float = 0.0, steps: int = MATCH_STEPS, rtol: float = MATCH_RTOL) -> dict:
    """f(β)(β ≥ 0 の強さ、f(0) = f0)を target にそろえる β を (0, hi] で探す(CHANGE-07 E.1 の大きさ合わせ)。
    β = start から始め、届かなければ原点と届いた中で最も近い点を結ぶ直線で外挿して広げ(1 回に 1.25〜4 倍、hi まで)、
    越えたら挟んだ区間を線形補間で詰める(区間の端から 1 割は離す)。|f − target| ≤ rtol·|target − f0| で止める。
    f は β について単調に target へ進むとみなす(そうでなくても、測った点のうち target に最も近いものを返す)。
    戻り値 {"beta", "value", "reached"(rtol 以内), "bracketed"(target を挟んだか), "evals", "trace": [[β, f(β)], ...]}。"""
    need = abs(float(target) - float(f0))
    trace: list[list[float]] = []
    if need == 0:
        return {"beta": 0.0, "value": float(f0), "reached": True, "bracketed": True, "evals": 0, "trace": trace}
    sgn = 1.0 if target > f0 else -1.0

    def prog(v: float) -> float:              # target の向きに f0 からどれだけ進んだか(target で need)
        return sgn * (v - f0)
    lo_b, lo_p = 0.0, 0.0
    hi_b = hi_p = None
    best = (0.0, float(f0))
    b = min(max(float(start), 1e-9), float(hi))
    for _ in range(max(1, int(steps))):
        v = float(f(b))
        trace.append([b, v])
        p = prog(v)
        if abs(v - target) < abs(best[1] - target):
            best = (b, v)
        if abs(p - need) <= rtol * need:
            break
        if p > need:
            if hi_b is None or b < hi_b:
                hi_b, hi_p = b, p
        elif p >= lo_p and b > lo_b:
            lo_b, lo_p = b, p
        if hi_b is None:                      # まだ届いていない: 広げる
            if b >= hi:
                break
            guess = lo_b * need / lo_p if lo_p > 0 and lo_b > 0 else 4 * b
            b = float(min(hi, max(min(guess, 4 * b), 1.25 * b)))
        else:                                 # 挟んだ: 補間で詰める
            w = hi_b - lo_b
            guess = lo_b + w * (need - lo_p) / (hi_p - lo_p) if hi_p != lo_p else lo_b + 0.5 * w
            b = float(min(hi_b - 0.1 * w, max(lo_b + 0.1 * w, guess)))
    reached = abs(best[1] - target) <= rtol * need
    return {"beta": best[0], "value": best[1], "reached": bool(reached), "bracketed": bool(reached or hi_b is not None),
            "evals": len(trace), "trace": trace}


# --- 物差しの準備 ------------------------------------------------------------------------------------------

@dataclass
class _Item:
    ids: list[int]                        # プロンプト + 返事のトークン
    pos: np.ndarray                       # 捕捉する位置(どれかの群に入る位置の和集合、昇順)
    gidx: dict[str, np.ndarray]           # 群 → pos の中の添字


@dataclass
class _EvalSet:
    label: str
    iv: tuple                             # 側の介入(対照集合の recipe_a / recipe_b)。調べる介入はこの上に重ねる
    items: list[_Item] = field(default_factory=list)
    replies: str | None = None            # given / generated / None(返事を使わない位置)
    texts: list[str] = field(default_factory=list)


def _thin_idx(n: int, m: int | None) -> list[int]:
    if not m or n <= m:
        return list(range(n))
    return sorted({int(x) for x in np.linspace(0, n - 1, int(m)).round()})


def resolve_keyed_contrast(session, ks, ds, contrast) -> tuple[object | None, str]:
    """鍵の評価に使う対照集合。渡されたものがあればそれ。無ければ方向を作った対照集合(方向の source)が鍵と同じ対照集合
    (contrast_sha が同じ)ならそれ。どちらも無ければ (None, 理由)。"""
    if contrast is not None:
        return contrast, "given"
    from .keyed_build import read_key_spec
    src = (getattr(ds, "meta", None) or {}).get("source")
    if not src or not Path(str(src)).exists():
        return None, f"鍵 {ks.name} の対照集合が分かりません(鍵のファイルには対照集合のパスがありません)"
    try:
        spec = read_key_spec(src)
    except GneError as e:
        return None, f"方向 {ds.name} の対照集合を鍵の対照集合として読めません: {e.message}"
    if ks.contrast_sha and spec.contrast.sha != ks.contrast_sha:
        return None, f"方向 {ds.name} の対照集合は鍵 {ks.name} を作った対照集合と違います(contrast_sha が違う)"
    return spec, "direction_source"


def _capture(e):
    from .capture import Capture
    cap = getattr(e, "_keyed_eval_capture", None)
    if cap is None:
        cap = e._keyed_eval_capture = Capture()
    return cap


def _mean(xs) -> float | None:
    return float(np.mean(xs)) if len(xs) else None


# --- 本体 --------------------------------------------------------------------------------------------------

class KeyedMetrics:
    """keyed のレシピの選択性(CHANGE-07 E.1)と擬似鍵の対照(E.2)を prefill だけで測る。
    使い方: km = KeyedMetrics(session, recipe, contrast=...); km.prepare()(無介入の残差、1 回)→ km.row(r)(eval の倍率ごと、
    レシピ全体)・km.row(c)["selectivity"](対照レシピ)→ km.block()(方式の比較と擬似鍵、倍率 1 の keyed のユニットだけ)。
    tick(n, label) は読んだ会話の数で呼ぶ(eval の進み具合)。"""

    def __init__(self, session, recipe: Recipe, *, contrast=None, subset: str | None = None,
                 max_examples: int | None = DEFAULT_EXAMPLES, pseudo: int = DEFAULT_PSEUDO, match: bool = True,
                 layers: list[int] | None = None, steps: int = MATCH_STEPS, rtol: float = MATCH_RTOL,
                 gain_neurons: tuple[int, int] = GAIN_NEURONS):
        from .controls import keyed_control_cap
        from .keyed import keyed_units, load_key, split_key_spec
        from .keyed_build import eval_plan
        self.session, self.recipe = session, recipe
        self.ku = keyed_units(recipe)
        if not self.ku:
            raise GneError("BAD_RECIPE", "keyed のユニットがありません")
        self.notes: list[str] = []
        self.warnings: list[str] = []
        self._key_of = getattr(session, "_key_of", None) or (lambda n: load_key(session, n))
        keys = list(dict.fromkeys(split_key_spec(k.key)[0] for k in self.ku))
        dirs = list(dict.fromkeys(k.direction for k in self.ku))
        self.key_name, self.dname = keys[0], dirs[0]
        if len(keys) > 1:
            self.notes.append(f"鍵が {len(keys)} 種類あります({', '.join(keys)})。位置は最初の鍵 {keys[0]} の作成の条件で測りました")
        if len(dirs) > 1:
            self.notes.append(f"方向が {len(dirs)} 種類あります({', '.join(dirs)})。r̂ は最初の方向 {dirs[0]} で測りました")
        signs = {np.sign(k.alpha) for k in self.ku if k.alpha}
        self.sign = float(np.sign(self.ku[0].alpha) or 1.0)
        if len(signs) > 1:
            self.notes.append("足す(α > 0)と打ち消す(α < 0)のユニットが混ざっています。選択性の向きは最初のユニットの符号です")
        self.ks = self._key_of(keys[0])
        self.ds = session.direction_set(self.dname)
        spec, how = resolve_keyed_contrast(session, self.ks, self.ds, contrast)
        if spec is None:
            raise GneError("BAD_ARGS", how, "鍵を作った対照集合を keyed_contrast(CLI では --keyed-contrast)で渡してください")
        self.contrast_from = how
        self.plan = eval_plan(session, self.ks, spec, subset=subset, warn=self._warn)
        if not self.plan["overlap"]["held_out"]:
            self.notes.append(f"測った例が鍵の作成に使った例と重なっています(作成 {self.ks.subset}・評価 {self.plan['subset']})。"
                              "選択性は甘くなります")
        n_layer = session.meta.n_layer
        kl = sorted({k.layer for k in self.ku})
        self.layers = sorted({int(L) for L in (layers or kl)})
        for L in self.layers:
            if L >= n_layer - 1:
                raise GneError("BAD_ARGS", f"最終層 {L} は行が間引かれるので測れません")
            if L not in self.ds.raw:
                raise GneError("BAD_ARGS", f"方向 {self.dname} に層 {L} がありません")
        self.obj = max(self.layers)                       # 目的の層(行と方式の比較の値はこの層)
        self.r = {L: self.ds.unit(L).astype(np.float64) for L in self.layers}
        self.max_examples = max_examples
        self.n_pseudo = max(0, min(int(pseudo), keyed_control_cap(session, recipe, self._key_of)))
        if pseudo and not self.n_pseudo:
            self.notes.append(f"鍵 {self.key_name} に擬似鍵が無いので擬似鍵の対照は測りません"
                              "(gne direction key の pseudo を 1 以上にして作り直してください)")
        self.match, self.steps, self.rtol = bool(match), int(steps), float(rtol)
        self.gain_neurons = (int(gain_neurons[0]), int(gain_neurons[1]))
        self.sets: list[_EvalSet] = []
        self.base: list[list[dict[int, np.ndarray]]] | None = None
        self.max_len = 0
        self.prefills = 0
        self.tick: Callable[[int, str], None] = lambda n, label="": None
        # 進み具合の見積もり(prepare の前。位置の無い会話は後で落ちるので上限)
        n_sets = [len(_thin_idx(len(s.convs), max_examples)) for s in self.plan["sets"]]
        self.n_full = sum(n_sets)
        self.n_a = sum(n for s, n in zip(self.plan["sets"], n_sets, strict=True) if "A" in s.groups)

    def _warn(self, msg: str) -> None:
        self.warnings.append(msg)
        self.session.warn(msg)

    # --- 準備と測定 ---
    def block_cost(self) -> int:
        """block() で読む会話の数の見積もり(上限)。"""
        n_methods = 2 + (3 if self.match else 0)
        return n_methods * self.n_full + self.n_pseudo * self.n_full + (3 * self.steps * self.n_a if self.match else 0)

    def prepare(self) -> None:
        """返事を用意し(対照集合に書いたものか、無介入で 1 回だけ生成)、位置を選び、無介入の残差を捕捉する。"""
        from .evalx import _generate
        from .keyed_build import parse_positions, select_positions
        s, e = self.session, self.session.engine
        skip, mt = self.plan["skip"], self.plan["max_tokens"]
        self.sets = []
        for st in self.plan["sets"]:
            idx = _thin_idx(len(st.convs), self.max_examples)
            convs = [st.convs[i] for i in idx]
            reps = None if st.replies is None else [st.replies[i] for i in idx]
            groups = {g: parse_positions(p) for g, p in st.groups.items()}
            pids = [e.tokenize_prompt(e.render_chat(c)) for c in convs]
            starts: list[int | None] = [None] * len(convs)
            if any(p.kind == "assistant" for p in groups.values()):
                for i, c in enumerate(convs):
                    head = e.tokenize_prompt(e.render_chat(c, add_generation_prompt=False))
                    starts[i] = len(head) if pids[i][:len(head)] == head else len(pids[i]) - 1
            if not any(p.needs_reply for p in groups.values()):
                resp, mode, texts = [[] for _ in convs], None, []
            elif reps is not None:
                resp = [e.tokenize(str(r), add_special=False, parse_special=False) for r in reps]
                mode, texts = "given", [str(r) for r in reps]
            else:
                vecs, lora = st.iv
                outs = _generate(s, [(c, int(mt)) for c in convs], vecs, lora, "all", lambda k: None)
                resp = [[t for t in o["token_ids"] if not e.is_eog(t)] for o in outs]
                mode, texts = "generated", [o["text"] for o in outs]
                self.tick(len(convs), "keyed: 返事の生成")
            es = _EvalSet(st.label, st.iv, [], mode, texts)
            need_pieces = any(p.pattern for p in groups.values())
            for i in range(len(convs)):
                pieces = [e.token_bytes(t) for t in resp[i]] if need_pieces else None
                sel = {g: select_positions(e, resp[i], p, skip, n_prompt=len(pids[i]), start=starts[i], pieces=pieces)
                       for g, p in groups.items()}
                sel = {g: v for g, v in sel.items() if v.size}
                if not sel:
                    continue
                pos = np.unique(np.concatenate(list(sel.values())))
                es.items.append(_Item(pids[i] + resp[i], pos, {g: np.searchsorted(pos, v) for g, v in sel.items()}))
            self.sets.append(es)
        items = [it for es in self.sets for it in es.items]
        if not any("A" in it.gidx for it in items):
            raise GneError("BAD_CONTRAST", f"側 A の位置がありません(positions {self.plan['positions']})",
                           "返事に鍵の正規表現が一致していません。一致する返事を対照集合の replies で与えてください")
        if not any(g != "A" for it in items for g in it.gidx):
            raise GneError("BAD_CONTRAST", "側 B・dont_touch の位置がありません")
        self.max_len = max(len(it.ids) for it in items)
        if self.max_len > e.n_ctx:
            raise GneError("CONTEXT_FULL", f"{self.max_len} トークンの会話が n_ctx={e.n_ctx} に入りません")
        self.n_full = len(items)
        self.n_a = sum("A" in it.gidx for it in items)
        self.base = self._run(None, None, self.layers, None, "keyed: 無介入")

    def _run(self, vecs, lora, layers, groups, label) -> list[list[dict[int, np.ndarray] | None]]:
        """全会話(groups を渡すとその群を含む会話だけ)を (vecs, lora) を重ねて prefill し、層 → 位置の残差(pos の行)。"""
        from .reverse import combine
        e = self.session.engine
        cap = _capture(e)
        names = {f"l_out-{L}": L for L in layers}
        cap.set_wanted(lambda nm: nm in names)
        ctx = e.probe_ctx(cap, need=self.max_len)
        out: list[list[dict[int, np.ndarray] | None]] = []
        n = 0
        try:
            for es in self.sets:
                sv, sl = combine(es.iv, vecs, lora)
                e.set_lora(ctx, sl)
                e.set_cvec(ctx, sv)
                rows: list[dict[int, np.ndarray] | None] = []
                for it in es.items:
                    if groups is not None and not any(g in it.gidx for g in groups):
                        rows.append(None)
                        continue
                    e.clear_memory(ctx)
                    e.prefill(ctx, [it.ids])
                    got = cap.take()
                    d = {}
                    for nm, L in names.items():
                        if nm not in got:
                            raise GneError("ARCH_UNSUPPORTED", f"{nm} が捕捉できません")
                        if got[nm].shape[0] != len(it.ids):
                            raise GneError("ARCH_UNSUPPORTED", f"{nm} の行の数 {got[nm].shape[0]} が列の長さ {len(it.ids)} と違います")
                        d[L] = np.asarray(got[nm][it.pos], dtype=np.float32)
                    rows.append(d)
                    n += 1
                out.append(rows)
        finally:
            e.set_cvec(ctx, None)
            e.set_lora(ctx, None)
        self.prefills += n
        self.tick(n, label)
        return out

    def measure(self, vecs=None, lora=None, *, layers: list[int] | None = None, groups=None,
                label: str = "keyed") -> dict[int, dict[str, dict]]:
        """(vecs, lora) を当てたときの、層 → 群 → {par(Δ·r̂、符号つき)、norm(‖Δ‖)、orth(‖Δ の直交成分‖)、n_ex、n_pos}。
        例ごとに位置で平均してから例で平均する。groups を渡すとその群を含む会話だけ読む。"""
        if self.base is None:
            raise GneError("BAD_ARGS", "KeyedMetrics.prepare() を先に呼んでください")
        Ls = sorted(layers or self.layers)
        got = self._run(vecs, lora, Ls, groups, label)
        acc: dict[int, dict[str, dict[str, list]]] = {L: {} for L in Ls}
        for es, base_rows, new_rows in zip(self.sets, self.base, got, strict=True):
            for it, b, x in zip(es.items, base_rows, new_rows, strict=True):
                if x is None:
                    continue
                for L in Ls:
                    D = x[L].astype(np.float64) - b[L].astype(np.float64)
                    for g, gi in it.gidx.items():
                        if groups is not None and g not in groups:
                            continue
                        d = D[gi]
                        par = d @ self.r[L]
                        nrm = np.linalg.norm(d, axis=1)
                        orth = np.sqrt(np.maximum(nrm ** 2 - par ** 2, 0.0))
                        a = acc[L].setdefault(g, {"par": [], "norm": [], "orth": [], "n_pos": 0})
                        a["par"].append(float(par.mean()))
                        a["norm"].append(float(nrm.mean()))
                        a["orth"].append(float(orth.mean()))
                        a["n_pos"] += int(d.shape[0])
        return {L: {g: {"par": _mean(a["par"]), "norm": _mean(a["norm"]), "orth": _mean(a["orth"]),
                        "n_ex": len(a["par"]), "n_pos": a["n_pos"]} for g, a in acc[L].items()} for L in Ls}

    def summary(self, st: dict[str, dict]) -> dict:
        """1 層の群の値 → 行の要約。selectivity = 書く向き(最初の keyed のユニットの α の符号)にそろえた側 A − 側 B の par、
        b_over_a = 側 B の ‖Δ‖ ÷ |側 A の par|(側 A を 1 押すあたりの側 B の乱れ。小さいほどよい)。dt_* は dont_touch の群の平均。"""
        A, B = st.get("A") or {}, st.get("B") or {}
        dts = [v for g, v in st.items() if g.startswith("dt:")]
        a_par, b_par, b_norm = A.get("par"), B.get("par"), B.get("norm")
        sel = None if a_par is None or b_par is None else self.sign * (a_par - b_par)
        return {"a_par": a_par, "a_norm": A.get("norm"), "b_par": b_par, "b_norm": b_norm,
                "dt_par": _mean([v["par"] for v in dts]) if dts else None,
                "dt_norm": _mean([v["norm"] for v in dts]) if dts else None,
                "selectivity": sel,
                "b_over_a": None if b_norm is None or not a_par else float(b_norm / abs(a_par))}

    def _iv(self, r: Recipe) -> tuple:
        s = self.session
        vecs = s.vectors(r)[0] if r.add_units else None
        return (vecs or None), s.lora_for(r)

    def row(self, r: Recipe | None, label: str = "keyed") -> dict:
        """eval の 1 倍率の行の keyed の指標(目的の層で、レシピ全体を当てたもの)。r = None(倍率 0)は全部 0。"""
        if r is None:
            z = {"a_par": 0.0, "a_norm": 0.0, "b_par": 0.0, "b_norm": 0.0, "dt_par": 0.0 if self._has_dt() else None,
                 "dt_norm": 0.0 if self._has_dt() else None, "selectivity": 0.0, "b_over_a": None}
            return {"layer": self.obj, **z}
        st = self.measure(*self._iv(r), layers=[self.obj], label=label)
        return {"layer": self.obj, **self.summary(st[self.obj])}

    def _has_dt(self) -> bool:
        return any(g.startswith("dt:") for es in self.sets for it in es.items for g in it.gidx)

    # --- 方式の比較 ---
    def _add_vecs(self, beta: float = 1.0) -> dict[int, np.ndarray]:
        """keyed のユニットと同じ層・同じ α・同じ向き(keyed が書く生の r)を全位置に足す control vector。"""
        from .keyed import raw_direction
        v: dict[int, np.ndarray] = {}
        for k in self.ku:
            r, _ = raw_direction(self.session.direction_set(k.direction), k.layer, self.recipe.direction_layer)
            v[k.layer] = v.get(k.layer, 0.0) + float(k.alpha) * r
        return {L: (float(beta) * x).astype(np.float32) for L, x in v.items()}

    def _add_units(self, beta: float) -> list[Unit]:
        return [Unit(k.layer, None, float(beta) * k.alpha, "direction", direction=k.direction) for k in self.ku]

    def _gain_units(self) -> tuple[list[Unit] | None, str]:
        """gain の対照: keyed の層ごとに、側 A での寄与 c_n = μ_n·(r̂ᵀ W_down[:, n])·s が正のニューロンを大きい順に、寄与の
        和が |α|·‖r_L‖ / GAIN_AIM に届く数(gain_neurons の範囲)だけ選ぶ(μ は鍵の側 A の平均活性。打ち消す向きは k = 1 − β で
        β ≤ 2 なので 1 / min(GAIN_AIM, 1))。k = 1 ± 1(β = 1)。戻り値 (ユニット, 説明) か (None, 除外の理由)。"""
        from .keyed import raw_direction, split_key_spec
        from .steer import _down_scale
        w = self.session.weights
        if w.meta.arch == "bitnet":
            return None, "bitnet は ffn_down の入力が ffn_sub_norm の後なので、ニューロンの gain の対照は作りません"
        units: list[Unit] = []
        seen: set[int] = set()
        for k in self.ku:
            if k.kind == "attn":
                return None, ("attn の keyed にはニューロンの gain の対照を作りません(書き込み先がヘッド出力の連結で、"
                              "ffn_down のニューロンではない。ヘッドの対照は proj)")
            if k.kind != "ffn" or w.is_moe(k.layer):
                return None, "gain の対照は dense の ffn_down の keyed だけです"
            if k.layer in seen:
                continue
            seen.add(k.layer)
            ks = self._key_of(split_key_spec(k.key)[0])
            mu = np.asarray(ks.mu[k.layer], dtype=np.float64)
            r, _ = raw_direction(self.session.direction_set(k.direction), k.layer, self.recipe.direction_layer)
            W = w.down_matrix(k.layer)
            if W.shape[1] != mu.size:
                return None, f"L{k.layer}: 鍵の長さ {mu.size} が ffn_down の入力 {W.shape[1]} と違います"
            contrib = mu * ((r / np.linalg.norm(r)) @ W.astype(np.float64)) * _down_scale(w, k.layer)
            order = np.argsort(-contrib)
            order = order[contrib[order] > 0]
            if not order.size:
                return None, f"L{k.layer}: 側 A で r̂ に書くニューロンがありません"
            aim = GAIN_AIM if self.sign > 0 else min(GAIN_AIM, 1.0)
            want = sum(abs(u.alpha) for u in self.ku if u.layer == k.layer) * float(np.linalg.norm(r)) / aim
            n = int(np.searchsorted(np.cumsum(contrib[order]), want)) + 1
            n = min(max(n, self.gain_neurons[0]), self.gain_neurons[1], int(order.size))
            units += [Unit(k.layer, int(i), 0.0, mode="gain", k=1.0 + self.sign) for i in order[:n]]
        per = {L: sum(u.layer == L for u in units) for L in sorted(seen)}
        return units, ("keyed の層ごとに、側 A で r̂ に書く寄与の大きいニューロン("
                       + "・".join(f"L{L} {c} 個" for L, c in per.items()) + f")を k = 1 {'+' if self.sign > 0 else '−'} β 倍")

    def _proj_units(self, s: float) -> tuple[list[Unit] | None, str]:
        """proj の対照: keyed の層の全ヘッドの r̂ 成分を s 倍ぶん引く(s = −1 で 2 倍に強める)。"""
        arch = self.session.weights.arch
        units: list[Unit] = []
        for L in sorted({k.layer for k in self.ku}):
            li = arch.layers[L]
            if li.head_dim and li.head_writer:
                units += [Unit(L, None, 0.0, "head", mode="proj", k=float(s), head=h, direction=self.dname)
                          for h in range(li.n_head)]
        if not units:
            return None, "keyed の層に attention のヘッドがありません"
        return units, "keyed の層の全ヘッドの書き込みの r̂ 成分を s = −β(強める)倍ぶん引く"

    def _method(self, method: str, desc: str, st: dict, *, beta=None, matched=None, evals=None, trace=None,
                units: list[Unit] | None = None, recipe: Recipe | None = None) -> dict:
        row = {"method": method, "desc": desc, "beta": beta, "matched": matched, "evals": evals,
               **self.summary(st[self.obj]),
               "layers": {str(L): {g: {k: v[k] for k in ("par", "norm", "orth", "n_ex", "n_pos")} for g, v in st[L].items()}
                          for L in sorted(st)}}
        if units is not None:
            row["units"] = [u.label() for u in units]
        if recipe is not None:
            row["recipe"] = recipe.to_dict()
        if trace is not None:
            row["trace"] = trace
        return row

    def block(self) -> dict:
        """方式の比較(keyed・同じ α の add・大きさを合わせた add・gain・proj)と擬似鍵の選択性。eval の結果の keyed。"""
        from .controls import pseudo_keyed
        from .controls import summarize as summ
        t0, p0 = time.perf_counter(), self.prefills
        s, e = self.session, self.session.engine
        kr = self.recipe.like(self.recipe.keyed_units, name=f"{self.recipe.name or 'recipe'}-keyed", ablate=None)
        if len(kr.units) != len(self.recipe.units) or self.recipe.ablate is not None:
            self.notes.append("方式の比較と擬似鍵は keyed のユニットだけで測りました(レシピのほかのユニットは外しました。"
                              "行の keyed はレシピ全体)")
        obj = self.obj
        methods: list[dict] = []
        drop: list[str] = []
        st_k = self.measure(*self._iv(kr), label="keyed: keyed")
        methods.append(self._method("keyed", "このレシピの keyed のユニット", st_k, beta=1.0, recipe=kr))
        target = st_k[obj]["A"]["par"] if "A" in st_k[obj] else None
        add_ok = all(k.layer >= 1 for k in self.ku)
        st_a1 = None
        if add_ok:
            st_a1 = self.measure(self._add_vecs(1.0), None, label="keyed: add 同じ α")
            u1 = self._add_units(1.0)
            methods.append(self._method("add_same_alpha", "同じ層・同じ α の L:dir:NAME:α(raw、全位置に足す)", st_a1,
                                        beta=1.0, units=u1,
                                        recipe=Recipe(u1, s.meta.model_id, "raw", f"{kr.name}-add",
                                                      direction_layer=self.recipe.direction_layer)))
        else:
            methods.append({"method": "add_same_alpha", "excluded": "層 0 には control vector を足せません"})
        scale = max(abs(k.alpha) for k in self.ku) * float(np.linalg.norm(self.ds.raw[obj]))
        weak = target is None or abs(target) < MIN_TARGET * scale
        if self.match and weak:
            self.notes.append(f"keyed が側 A をほとんど動かしていない(L{obj} の par {0.0 if target is None else target:+.3g}、"
                              f"全部書いたときの {MIN_TARGET:.0%} 未満)ので、add・gain・proj の大きさは合わせていません")
        elif self.match:
            for spec in self._match_specs(st_a1, add_ok):
                if "excluded" in spec:
                    methods.append({"method": spec["method"], "desc": spec.get("desc"), "excluded": spec["excluded"]})
                    continue
                methods.append(self._match_one(spec, target, drop))
        # 擬似鍵(E.2)
        pseudo = None
        if self.n_pseudo:
            vals, keys = [], []
            for j in range(self.n_pseudo):
                pr = kr.like(pseudo_keyed(kr, j), name=f"{kr.name}-pseudo{j}")
                vecs, lora = self._iv(pr)
                drop += [p for p, _ in (lora or [])]
                st = self.measure(vecs, lora, layers=[obj], label=f"keyed: 擬似鍵 {j + 1}/{self.n_pseudo}")
                vals.append(self.summary(st[obj])["selectivity"])
                keys.append(f"{self.key_name}~{j}")
            real = methods[0]["selectivity"]
            sm = summ(vals)
            beats = None if real is None or sm["p95"] is None else bool(real > sm["p95"])
            from .controls import null_pct
            pseudo = {"n": len(vals), "keys": keys, "values": vals, "p50": sm["p50"], "p95": sm["p95"], "max": sm["max"],
                      "real": real, "beats": beats, "null_pct": null_pct(real, vals)}
            if beats is False:
                self.notes.append(f"本物の鍵の選択性 {real:+.3g} が擬似鍵 {len(vals)} 本の p95 {sm['p95']:+.3g} を超えていません"
                                  "(鍵が状態ではなく例の本文を拾っている可能性。key-check と対照集合を見直してください)")
        compare = self._compare(methods)
        for p in set(drop):
            try:
                e.drop_lora(p)
            except Exception:                          # 解放の失敗は測定に関係しない
                pass
        out = {"key": self.ks.spec or self.key_name, "key_sha": self.ks.sha, "direction": self.dname,
               "units": [k.spec() for k in self.ku], "sign": self.sign, "contrast": self.plan["contrast"],
               "contrast_sha": self.plan["contrast_sha"], "contrast_from": self.contrast_from, "subset": self.plan["subset"],
               "held_out": bool(self.plan["overlap"]["held_out"]), "overlap": self.plan["overlap"], "mode": self.plan["mode"],
               "positions": self.plan["positions"], "positions_b": self.plan["positions_b"], "skip": self.plan["skip"],
               "layers": self.layers, "objective_layer": obj, "counts": self.counts(), "replies": self.replies(),
               "base": self.base_stats(), "methods": methods, "compare": compare, "pseudo": pseudo,
               "settings": {"max_examples": self.max_examples, "match": self.match, "steps": self.steps, "rtol": self.rtol,
                            "gain_neurons": list(self.gain_neurons), "gain_aim": GAIN_AIM, "n_pseudo": self.n_pseudo,
                            "min_target": MIN_TARGET,
                            "hi": dict(MATCH_HI)},
               "notes": list(self.notes), "warnings": list(self.warnings),
               "cost": {"prefills": self.prefills - p0, "prefills_total": self.prefills,
                        "seconds": time.perf_counter() - t0}}
        out["text"] = keyed_text(out)
        return out

    def _match_specs(self, st_a1, add_ok: bool) -> list[dict]:
        """大きさを合わせる方式: make(β) → (vecs, lora)、hi、start、説明、ユニット(β)。"""
        s = self.session
        specs: list[dict] = []
        if add_ok:
            f1 = st_a1[self.obj]["A"]["par"] if st_a1 is not None and "A" in st_a1[self.obj] else None
            specs.append({"method": "add", "desc": "同じ向きを全位置に足す(add_same_alpha の全体を β 倍)",
                          "make": lambda b: (self._add_vecs(b), None), "hi": MATCH_HI["add"], "start": 1.0,
                          "known": {1.0: f1} if f1 is not None else {}, "units": self._add_units,
                          "recipe": lambda b: Recipe(self._add_units(b), s.meta.model_id, "raw", "keyed-add",
                                                     direction_layer=self.recipe.direction_layer)})
        else:
            specs.append({"method": "add", "excluded": "層 0 には control vector を足せません"})
        gu, gdesc = self._gain_units()
        if gu is None:
            specs.append({"method": "gain", "excluded": gdesc})
        else:
            path = s.lora_for(Recipe(gu, s.meta.model_id, name="keyed-gain"))[0][0]
            sg = self.sign

            def gain_units(b, gu=gu):
                return [Unit(u.layer, u.neuron, 0.0, mode="gain", k=1.0 + sg * b) for u in gu]
            specs.append({"method": "gain", "desc": gdesc, "make": lambda b, p=path: (None, [(p, float(b))]),
                          "hi": MATCH_HI["gain" if sg > 0 else "gain_down"], "start": 1.0, "known": {}, "units": gain_units,
                          "lora": [path], "recipe": lambda b: Recipe(gain_units(b), s.meta.model_id, name="keyed-gain")})
        pu, pdesc = self._proj_units(-1.0)
        if pu is None:
            specs.append({"method": "proj", "excluded": pdesc})
        else:
            def proj_spec(sgn: float) -> dict:
                units, _ = self._proj_units(-sgn)
                rec = Recipe(units, s.meta.model_id, name="keyed-proj", direction_layer=self.recipe.direction_layer)
                path = s.proj_lora(rec)

                def pu_of(b, units=units):
                    return [Unit(u.layer, None, 0.0, "head", mode="proj", k=-sgn * b, head=u.head, direction=u.direction)
                            for u in units]
                return {"make": lambda b, p=path: (None, [(str(p), float(b))]), "units": pu_of, "lora": [str(path)],
                        "recipe": lambda b: Recipe(pu_of(b), s.meta.model_id, name="keyed-proj",
                                                   direction_layer=self.recipe.direction_layer)}
            specs.append({"method": "proj", "desc": pdesc, "hi": MATCH_HI["proj"], "start": 1.0, "known": {},
                          "flip": proj_spec, **proj_spec(1.0)})
        return specs

    def _match_one(self, spec: dict, target: float, drop: list[str]) -> dict:
        obj = self.obj

        def runner(sp):
            known = dict(sp.get("known") or {})

            def f(b: float) -> float:
                if b in known:
                    return known[b]
                st = self.measure(*sp["make"](b), layers=[obj], groups={"A"}, label=f"keyed: {spec['method']} の大きさ合わせ")
                v = st[obj]["A"]["par"]
                known[b] = v
                return v
            return match_strength(f, target, start=spec["start"], hi=spec["hi"], steps=self.steps, rtol=self.rtol)
        sp = dict(spec)
        drop += sp.get("lora", [])
        m = runner(sp)
        desc = spec["desc"]
        first = m["trace"][0][1] if m["trace"] else 0.0
        if "flip" in spec and np.sign(target) * first <= 0 and not m["reached"]:
            sp2 = {**spec, **spec["flip"](-1.0), "known": {}}
            drop += sp2.get("lora", [])
            m2 = runner(sp2)
            if abs(m2["value"] - target) < abs(m["value"] - target):
                sp, m = sp2, m2
                desc = "keyed の層の全ヘッドの書き込みの r̂ 成分を s = +β 倍ぶん引く(強めると側 A が逆に動くので、消す・反転する向き)"
        if float(np.sign(target)) * m["value"] < 0.1 * abs(target):        # 側 A をそろえる向きにほとんど動かない
            return {"method": spec["method"], "desc": desc, "beta": m["beta"], "evals": m["evals"], "trace": m["trace"],
                    "excluded": f"β = 0〜{spec['hi']:g} で側 A の par が keyed の {target:+.3g} に近づきません"
                                f"(最も近くて {m['value']:+.3g}、β = {m['beta']:.3g})"}
        st = self.measure(*sp["make"](m["beta"]), label=f"keyed: {spec['method']}")
        units = sp["units"](m["beta"])
        row = self._method(spec["method"], desc, st, beta=m["beta"], matched=m["reached"], evals=m["evals"],
                           trace=m["trace"], units=units, recipe=sp["recipe"](m["beta"]))
        if not m["reached"]:
            self.notes.append(f"{spec['method']}: 側 A の par を keyed に ±{self.rtol:.0%} でそろえられませんでした"
                              f"(β = {m['beta']:.3g} で {m['value']:+.3g}、keyed は {target:+.3g})")
        return row

    def _compare(self, methods: list[dict]) -> dict:
        """keyed と対照の、側 B・dont_touch の乱れの比(keyed ÷ 対照。1 より小さいほど keyed が選択的)。
        b_norm_ratio・dt_norm_ratio は同じ α の add と側 A をそろえられた対照(matched)だけ、b_over_a_ratio(側 A を 1 押す
        あたりの側 B の ‖Δ‖ の比)はそろえられなかった対照も含む。keyed_smaller は keyed の側 B の ‖Δ‖ が同じ α の add と
        そろえた対照のどれよりも小さいか。"""
        k = methods[0]
        out: dict = {"b_norm_ratio": {}, "dt_norm_ratio": {}, "b_over_a_ratio": {}, "keyed_smaller": None}
        smaller = []
        for m in methods[1:]:
            if "excluded" in m:
                continue
            if k.get("b_over_a") is not None and m.get("b_over_a"):
                out["b_over_a_ratio"][m["method"]] = float(k["b_over_a"] / m["b_over_a"])
            if not (m["method"] == "add_same_alpha" or m.get("matched")):
                continue
            for key, dst in (("b_norm", "b_norm_ratio"), ("dt_norm", "dt_norm_ratio")):
                if k.get(key) is not None and m.get(key):
                    out[dst][m["method"]] = float(k[key] / m[key])
            if k.get("b_norm") is not None and m.get("b_norm") is not None:
                ok = k["b_norm"] < m["b_norm"]
                smaller.append(ok)
                if not ok:
                    self.notes.append(f"側 B の ‖Δ‖ が {m['method']} より小さくありません"
                                      f"(keyed {k['b_norm']:.3g}、{m['method']} {m['b_norm']:.3g}。keyed の利点が見えません)")
        out["keyed_smaller"] = all(smaller) if smaller else None
        return out

    # --- 記録 ---
    def counts(self) -> dict:
        out: dict[str, dict] = {}
        for es in self.sets:
            for it in es.items:
                for g, gi in it.gidx.items():
                    c = out.setdefault(g, {"examples": 0, "positions": 0})
                    c["examples"] += 1
                    c["positions"] += int(gi.size)
        return out

    def replies(self) -> dict:
        return {es.label: es.replies for es in self.sets}

    def base_stats(self) -> dict:
        """無介入の残差の大きさ(位置の ‖x‖ の平均)と r̂ 成分(x·r̂ の平均)。par・norm の大きさを読む目安。"""
        out: dict[str, dict] = {}
        for L in self.layers:
            acc: dict[str, dict[str, list]] = {}
            for es, rows in zip(self.sets, self.base or [], strict=True):
                for it, b in zip(es.items, rows, strict=True):
                    x = b[L].astype(np.float64)
                    for g, gi in it.gidx.items():
                        a = acc.setdefault(g, {"resid": [], "proj": []})
                        a["resid"].append(float(np.linalg.norm(x[gi], axis=1).mean()))
                        a["proj"].append(float((x[gi] @ self.r[L]).mean()))
            out[str(L)] = {g: {"resid": _mean(a["resid"]), "proj": _mean(a["proj"])} for g, a in acc.items()}
        return out


# --- 表示 --------------------------------------------------------------------------------------------------

def _f(x, fmt: str = "+.3g") -> str:
    return "—" if x is None else format(x, fmt)


def keyed_text(k: dict) -> list[str]:
    """eval の結果の keyed を人が読む行にする(CLI の表示用)。"""
    c = k.get("counts") or {}
    ca, cb = c.get("A") or {}, c.get("B") or {}
    dts = [g for g in c if g.startswith("dt:")]
    lines = [f"keyed の選択性(鍵 {k['key']}、方向 {k['direction']}、L{k['objective_layer']} の残差、"
             f"{'保留' if k['held_out'] else '作成と重なる'} subset {k['subset']})",
             f"  側 A = {k['positions']}: {ca.get('examples', 0)} 例 {ca.get('positions', 0)} 位置 / "
             f"側 B = {k['positions_b']}: {cb.get('examples', 0)} 例 {cb.get('positions', 0)} 位置"
             + "".join(f" / {g[3:]}: {c[g]['examples']} 例 {c[g]['positions']} 位置" for g in dts),
             f"  {'方式':<16} {'β':>7} {'A の r̂':>9} {'B の r̂':>9} {'B の ‖Δ‖':>9} {'dt の ‖Δ‖':>10} {'選択性':>9} {'B/A':>7}"]
    for m in k.get("methods") or []:
        if "excluded" in m:
            lines.append(f"  {m['method']:<16} 除外: {m['excluded']}")
            continue
        mark = "" if m.get("matched") in (None, True) else " (合わず)"
        lines.append(f"  {m['method']:<16} {_f(m.get('beta'), '.3g'):>7} {_f(m.get('a_par')):>9} {_f(m.get('b_par')):>9} "
                     f"{_f(m.get('b_norm'), '.3g'):>9} {_f(m.get('dt_norm'), '.3g'):>10} {_f(m.get('selectivity')):>9} "
                     f"{_f(m.get('b_over_a'), '.3g'):>7}{mark}")
    cmp_ = k.get("compare") or {}
    if cmp_.get("b_norm_ratio"):
        lines.append("  側 B の ‖Δ‖ の比(keyed ÷ 対照、同じ α の add とそろえた対照): "
                     + "、".join(f"{m} {v:.2f}" for m, v in cmp_["b_norm_ratio"].items()))
    if cmp_.get("b_over_a_ratio"):
        lines.append("  側 A を 1 押すあたりの側 B の ‖Δ‖ の比(keyed ÷ 対照): "
                     + "、".join(f"{m} {v:.2f}" for m, v in cmp_["b_over_a_ratio"].items()))
    p = k.get("pseudo")
    if p:
        verdict = {True: "超える", False: "超えない", None: "判定なし"}[p.get("beats")]
        lines.append(f"  擬似鍵 {p['n']} 本の選択性: p50 {_f(p.get('p50'))} p95 {_f(p.get('p95'))} → 本物 {_f(p.get('real'))} は {verdict}")
    for n in k.get("notes") or []:
        lines.append(f"  注: {n}")
    return lines
