"""逆引き: 対照プロンプト(目標ペルソナ vs 中立)から方向を作り、全層・全ニューロンを類似度で順位づけする。設計書 §6.9。

見つかるのは「残差空間でそのペルソナに相関する方向」に書き込むニューロンで、各ニューロンは多義的なので、
候補は steer lens と rollout(対照群つき)で検証する(--verify)。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from .capture import Capture, pick_act_name
from .engine import IN_PROGRESS, Engine
from .steer import GneError

Progress = Callable[[int, int, str], None]
Positions = Literal["all", "last", "assistant"]


@dataclass
class ContrastSet:
    """対照集合。A(目標)と B(中立)の 2 つの状態で質問を読ませる。ふつうは両側で同じ user_prompts を読み、
    prompts_a / prompts_b を書くと側ごとに別の質問を読む(片側だけ書けば、もう片側は user_prompts。数は違ってよい)。
    別の質問を読むときは、方向に狙った違い以外の差(話題・長さ・言語)も混ざるので、それ以外をできるだけそろえる。

    状態は system プロンプトで作る(system_a / system_b)か、レシピを当てて作る(recipe_a / recipe_b、どちらか片側だけでも可)。
    レシピの側は、そのレシピのユニット(ニューロン・方向の add は control vector、gain・ヘッドは LoRA)を全位置に当てて読ませる。
    system を省いた側は system なしの会話になる(system_a だけを書けば、B は同じ質問を system なしで読んだ状態)。recipe_* はレシピ YAML のパス(対照集合のファイルからの相対パスも可)、
    保存したレシピの名前、ユニットの書き方のリスト、またはレシピと同じ形の表。"""
    name: str
    system_a: str | None
    system_b: str | None
    user_prompts: list                      # 両側で同じ質問(側ごとの質問だけのときは A の質問)。1 問は文字列か会話の履歴
    sha: str
    recipe_a: object = None
    recipe_b: object = None
    base_dir: Path | None = None
    prompts_a: list[str] | None = None      # A 側だけの質問(無ければ user_prompts)
    prompts_b: list[str] | None = None

    @property
    def separate_prompts(self) -> bool:
        return self.prompts("A") != self.prompts("B")

    def prompts(self, side: str) -> list[str]:
        own = self.prompts_a if side == "A" else self.prompts_b
        return list(own) if own is not None else list(self.user_prompts)

    @property
    def has_recipes(self) -> bool:
        return self.recipe_a is not None or self.recipe_b is not None

    def system(self, side: str) -> str | None:
        return self.system_a if side == "A" else self.system_b

    def convs(self, side: str, prompts: Sequence[str] | None = None) -> list[list[dict]]:
        """その側の会話(system があれば system + ユーザーの発話)。"""
        return [_conv(self.system(side), u) for u in (self.prompts(side) if prompts is None else prompts)]

    @property
    def multi_turn(self) -> bool:
        return any(is_multi(q) for side in "AB" for q in self.prompts(side))

    def require_systems(self, what: str) -> None:
        """system の違いでしか測れない道具(同じ返事を 2 つの system で読ませるもの)は、レシピの対照集合を断る。"""
        if self.multi_turn:
            raise GneError("BAD_CONTRAST", f"{what} は 1 ターンの質問だけを読みます(会話の履歴を書いた質問には使えません)",
                           "会話の履歴の質問は gne direction・direction score・heads causal・heads allocate・heads pipeline・eval で使えます")
        if self.has_recipes or self.system_a is None or self.system_b is None or self.separate_prompts:
            raise GneError("BAD_CONTRAST", f"{what} は同じ質問を system_a と system_b で読ませて比べる道具なので、レシピで A / B を"
                           f"作る対照集合(recipe_a / recipe_b)、system を省いた対照集合、側ごとに質問が違う対照集合"
                           f"(prompts_a / prompts_b)には使えません",
                           "レシピの対照集合は gne direction・direction score・heads causal・heads allocate・reverse・eval で使えます")

    @classmethod
    def load(cls, path: str | Path) -> ContrastSet:
        p = Path(path)
        if not p.exists():
            raise GneError("FILE_NOT_FOUND", f"対照プロンプト集 {p} がありません")
        text = p.read_text(encoding="utf-8")
        d = yaml.safe_load(text) or {}
        ra, rb = d.get("recipe_a"), d.get("recipe_b")
        up, ua, ub = d.get("user_prompts"), d.get("prompts_a"), d.get("prompts_b")
        for key, v in (("user_prompts", up), ("prompts_a", ua), ("prompts_b", ub)):
            check_questions(v, f"{p}: {key}")
        pa, pb = (ua if ua is not None else up), (ub if ub is not None else up)
        if not pa or not pb:
            raise GneError("BAD_CONTRAST", f"{p}: 質問が要ります(両側で同じなら user_prompts、側ごとに違うなら prompts_a と prompts_b)")
        same_prompts = list(pa) == list(pb)
        if ra is None and rb is None and same_prompts and d.get("system_a") is None and d.get("system_b") is None:
            raise GneError("BAD_CONTRAST", f"{p}: system_a か system_b(片方だけでも可。省いた側は system なしで読む)、"
                                           "recipe_a / recipe_b、または側ごとの質問 prompts_a / prompts_b が要ります")
        if ra is None and rb is None and same_prompts and d.get("system_a") == d.get("system_b"):
            raise GneError("BAD_CONTRAST", f"{p}: A と B が同じです(system も recipe も質問も違いがありません)")
        return cls(d.get("name") or p.stem, d.get("system_a"), d.get("system_b"), list(up if up is not None else pa),
                   hashlib.sha256(text.encode()).hexdigest()[:16], ra, rb, p.resolve().parent,
                   None if ua is None else list(ua), None if ub is None else list(ub))

    def recipe(self, session, side: str):
        """その側のレシピ(Recipe)か None。"""
        spec = self.recipe_a if side == "A" else self.recipe_b
        if spec is None:
            return None
        if isinstance(spec, list):
            return session.recipe(units=[str(x) for x in spec], name=f"{self.name}-{side}")
        if isinstance(spec, dict):
            import tempfile

            from .yamlio import write_yaml
            with tempfile.TemporaryDirectory() as td:          # 読み戻して同じ文字になるか確かめて書く(U+0085 なども)
                p = Path(td) / "recipe.yaml"
                write_yaml(p, {"name": f"{self.name}-{side}", **spec}, code="BAD_RECIPE")
                return session.recipe(path=p)
        q = Path(str(spec))
        for cand in ([q] if q.is_absolute() else [self.base_dir / q if self.base_dir else q, q]):
            if cand.suffix in (".yaml", ".yml") and cand.exists():
                return session.recipe(path=cand)
        return session.recipe(path=session.recipe_path(str(spec)))

    def describe(self) -> dict:
        """出力に残す、A / B の作り方。"""
        def one(side, system, spec):
            return {"system": system, "recipe": spec if spec is None or isinstance(spec, str) else
                    (list(spec) if isinstance(spec, list) else dict(spec)),
                    "n_prompts": len(self.prompts(side)), "own_prompts": (self.prompts_a if side == "A" else self.prompts_b) is not None}
        return {"A": one("A", self.system_a, self.recipe_a), "B": one("B", self.system_b, self.recipe_b)}


Intervention = tuple  # (vecs: dict[int, np.ndarray] | None, lora: list[tuple[str, float]] | None)


def side_interventions(session, cs: ContrastSet) -> dict[str, Intervention]:
    """A / B それぞれで当てる介入(control vector と LoRA)。レシピの無い側は (None, None)。"""
    out: dict[str, Intervention] = {}
    for side in "AB":
        r = cs.recipe(session, side)
        if r is None:
            out[side] = (None, None)
            continue
        if session.patches_for(r) is not None:
            raise GneError("BAD_CONTRAST", f"recipe_{side.lower()}: 専門家全体の gain・route(スケールパッチ)は対照集合の側には当てられません")
        vecs = session.vectors(r)[0] if r.add_units else None
        out[side] = (vecs or None, session.lora_for(r))
    return out


def combine(iv: Intervention | None, vecs=None, lora=None) -> Intervention:
    """側の介入に、さらに当てる介入(調べている介入)を重ねる。ベクトルは層ごとに足し、LoRA は並べる。"""
    va, la = iv or (None, None)
    v = {L: np.asarray(x, dtype=np.float32) for L, x in (va or {}).items()}
    for L, x in (vecs or {}).items():
        v[L] = v[L] + x if L in v else np.asarray(x, dtype=np.float32)
    lo = list(la or []) + list(lora or [])
    return (v or None), (lo or None)


# --- 質問: 文字列(1 ターン)か、会話の履歴(user と assistant が交互で user で始まるメッセージの列。assistant で終われば書きかけの返事) -----------

def is_multi(q) -> bool:
    return not isinstance(q, str)


def check_question(q, where: str = "質問", *, in_progress: bool = True) -> None:
    """1 問を確かめる。文字列か、[{"role": "user", "content": …}, {"role": "assistant", …}, …, {"role": "user", …}]。
    assistant で終われば書きかけの返事(その最後のトークンで測り、生成はその続きから)。in_progress=False の道具は断る。"""
    if isinstance(q, str):
        return
    if not isinstance(q, list) or not q:
        raise GneError("BAD_CONTRAST", f"{where}: 1 問は文字列か、会話の履歴(メッセージのリスト)です")
    for i, m in enumerate(q):
        if not isinstance(m, dict) or set(m) - {"role", "content"} or not isinstance(m.get("content"), str):
            raise GneError("BAD_CONTRAST", f"{where}: 会話の {i + 1} 番目は {{\"role\": …, \"content\": 文字列}} の形にしてください")
        want = "user" if i % 2 == 0 else "assistant"
        if m.get("role") != want:
            raise GneError("BAD_CONTRAST", f"{where}: 会話は user と assistant が交互で、user で始めてください({i + 1} 番目が "
                                           f"{m.get('role')!r}、{want!r} のはず)")
    if q[-1]["role"] != "user":
        if not in_progress:
            raise GneError("BAD_CONTRAST", f"{where}: ここでは会話の履歴を user の発話で終えてください(書きかけの返事は読めません)")
        if not q[-1]["content"]:
            raise GneError("BAD_CONTRAST", f"{where}: 最後の assistant の発話(書きかけの返事)が空です。返事の直前を測るなら user で終えてください")


def check_questions(v, where: str, *, in_progress: bool = True) -> None:
    if v is None:
        return
    if not isinstance(v, list):
        raise GneError("BAD_CONTRAST", f"{where} は質問のリストです(1 問は文字列か会話の履歴)")
    for i, q in enumerate(v):
        check_question(q, f"{where} の {i + 1} 問目", in_progress=in_progress)


def turns(q) -> list[dict]:
    """質問 → メッセージの列(system は含まない)。assistant で終わる履歴の最後には書きかけの印を付ける。"""
    if isinstance(q, str):
        return [{"role": "user", "content": q}]
    out = [{"role": m["role"], "content": m["content"]} for m in q]
    if out[-1]["role"] == "assistant":
        out[-1][IN_PROGRESS] = True
    return out


def question_text(q, *, sep: str = " / ") -> str:
    """表示・採点のファイル用の 1 行。会話の履歴は「user: … / assistant: … / user: …」。"""
    if isinstance(q, str):
        return q
    return sep.join(f"{m['role']}: {m['content']}" for m in q)


def _conv(system: str | None, user) -> list[dict]:
    """system が None なら system なしの会話(空の文字列は空の system として残す)。user は質問(文字列か会話の履歴)。"""
    return ([] if system is None else [{"role": "system", "content": system}]) + turns(user)


def contrast_sequences(engine: Engine, cs: ContrastSet) -> dict[str, list[tuple[list[int], int]]]:
    """{"A": [(ids, assistant の開始位置)], "B": [...]}。開始位置 = 生成プロンプト(assistant 冒頭)の最初のトークン。"""
    out: dict[str, list[tuple[list[int], int]]] = {"A": [], "B": []}
    for side, system in (("A", cs.system_a), ("B", cs.system_b)):
        for u in cs.prompts(side):
            msgs = _conv(system, u)
            ids = engine.tokenize_prompt(engine.render_chat(msgs))
            head = engine.tokenize_prompt(engine.render_chat(msgs, add_generation_prompt=False))
            start = len(head) if ids[:len(head)] == head else len(ids) - 1
            out[side].append((ids, min(start, len(ids) - 1)))
    return out


def _rows(x: np.ndarray, start: int, positions: Positions) -> np.ndarray:
    if positions == "all":
        return x[1:]                     # position 0 は attention sink なので除く(§5.6)
    if positions == "last":
        return x[-1:]
    return x[max(start, 1):]


@dataclass
class ContrastResult:
    directions: dict[int, np.ndarray]        # 層 → 方向(A − B)
    act_diff: dict[int, np.ndarray]          # 層 → (n_ff,) mean(a|A) − mean(a|B)
    method: str
    positions: str
    n_rows: dict[str, int]
    scale: dict[int, np.ndarray]             # 層 → (n_embd,) 残差の次元ごとの標準偏差(A∪B)
    act_std: dict[int, np.ndarray] | None = None   # 層 → (n_ff,) 活性の標準偏差(A∪B)。返事のトークンで測るときだけ


def contrastive_directions(engine: Engine, cs: ContrastSet, *, layers: Sequence[int] | None = None,
                           method: Literal["mean_diff", "pca1"] = "mean_diff", positions: Positions = "all",
                           with_act: bool = True, progress: Progress | None = None,
                           interventions: dict[str, Intervention] | None = None) -> ContrastResult:
    """目標群 A と中立群 B を probe ctx で 1 本ずつ prefill し、l_out-L の平均差(または差の第一主成分)と、
    ffn_down 入力の平均差(activation_contrast)を同じ順伝播で取る。最終層は行が間引かれ得るので除く。
    interventions は側ごとに当てる (control vector, LoRA)(レシピで A / B を作る対照集合、side_interventions)。"""
    Ls = [L for L in (range(engine.n_layer - 1) if layers is None else layers) if 0 <= L < engine.n_layer - 1]
    seqs = contrast_sequences(engine, cs)
    need = max(len(ids) for side in "AB" for ids, _ in seqs[side])     # コンテキストを最長の会話に合わせる(VRAM を空ける)
    names = engine.list_names(seqs["A"][0][0][:16])
    act_names = {}
    if with_act:
        from .moe import moe_names
        for L in Ls:
            nm = pick_act_name(names, L)
            if nm and "argsort" not in moe_names(names, L):   # MoE の層の ffn_swiglu は shared expert の活性なので使わない
                act_names[nm] = L
    resid_names = {f"l_out-{L}": L for L in Ls}
    cap = Capture(lambda n: n in resid_names or n in act_names)
    ctx = engine.probe_ctx(cap, need=need)
    engine.set_cvec(ctx, None)
    sums = {side: {L: 0.0 for L in Ls} for side in "AB"}
    sq = {L: 0.0 for L in Ls}                                  # A∪B の行の二乗和(次元ごとのばらつき用)
    act_sums = {side: {L: 0.0 for L in act_names.values()} for side in "AB"}
    per_prompt = {side: {L: [] for L in Ls} for side in "AB"}
    counts = {"A": 0, "B": 0}
    total, step = len(seqs["A"]) + len(seqs["B"]), 0
    for side in "AB":
        vecs, lora = (interventions or {}).get(side) or (None, None)
        ctx = engine.probe_ctx(cap, need=need)
        engine.set_cvec(ctx, vecs or None)
        engine.set_lora(ctx, lora or None)
        for ids, start in seqs[side]:
            ctx = engine.probe_ctx(cap, need=need)
            engine.clear_memory(ctx)
            engine.prefill(ctx, [ids])
            out = cap.take()
            n_rows = None
            for name, L in resid_names.items():
                r = _rows(out[name], start, positions).astype(np.float64)
                sums[side][L] = sums[side][L] + r.sum(axis=0)
                sq[L] = sq[L] + np.square(r).sum(axis=0)
                per_prompt[side][L].append(r.mean(axis=0))
                n_rows = r.shape[0]
            for name, L in act_names.items():
                act_sums[side][L] = act_sums[side][L] + _rows(out[name], start, positions).astype(np.float64).sum(axis=0)
            counts[side] += n_rows or 0
            step += 1
            if progress:
                progress(step, total, "reverse")
        engine.set_cvec(ctx, None)
        engine.set_lora(ctx, None)
    dirs: dict[int, np.ndarray] = {}
    for L in Ls:
        md = sums["A"][L] / counts["A"] - sums["B"][L] / counts["B"]
        if method == "mean_diff":
            dirs[L] = md.astype(np.float32)
        elif method == "pca1":
            D = np.stack(per_prompt["A"][L]) - np.stack(per_prompt["B"][L])      # (n_prompts, n_embd)
            pc = np.linalg.svd(D, full_matrices=False)[2][0]
            pc *= np.sign(pc @ md) or 1.0
            dirs[L] = (pc * (pc @ md)).astype(np.float32)                         # 平均差をその軸に射影した大きさ
        else:
            raise GneError("BAD_ARGS", f"method={method!r} は mean_diff か pca1 です")
    act = {L: (act_sums["A"][L] / counts["A"] - act_sums["B"][L] / counts["B"]).astype(np.float32)
           for L in act_names.values()}
    n = counts["A"] + counts["B"]
    scale = {}
    for L in Ls:
        mu = (sums["A"][L] + sums["B"][L]) / n
        scale[L] = np.sqrt(np.maximum(sq[L] / n - mu ** 2, 0)).astype(np.float32)
    return ContrastResult(dirs, act, method, positions, counts, scale)


@dataclass
class Candidate:
    layer: int
    neuron: int
    cos: float
    sign: int                 # +1: + で押すと目標方向、-1: − で押すと目標方向
    act_diff: float | None
    steer_kl: float | None
    rank_score: float
    expert: int | None = None          # MoE の層(CHANGE-05): neuron は通し番号、expert と expert_neuron がその中身
    expert_neuron: int | None = None


def cos_by_layer(weights, directions: dict[int, np.ndarray],
                 scale: dict[int, np.ndarray] | None = None) -> dict[int, np.ndarray]:
    """cos(u_n, direction[L]) を層ごとに (n_ff,) で。
    scale(残差の次元ごとの標準偏差)があれば、両方をその次元で割った空間で cos を取る。少数の次元だけが
    飛び抜けて大きい(massive activation)モデルで、その次元に書き込むニューロンが上位を占めるのを防ぐ(F17)。"""
    out = {}
    for L, d in directions.items():
        U = weights.unit_matrix(L)
        if scale is not None and L in scale:
            s = np.maximum(scale[L], 1e-6 * float(scale[L].max()) + 1e-12)
            d = d / s
            U = U / s[:, None]
            un = np.linalg.norm(U, axis=0, keepdims=True)
            U = U / np.where(un > 0, un, 1.0)                    # 書き込みが全部 0 の列(使われない専門家)は 0 のまま
        nrm = float(np.linalg.norm(d))
        out[L] = (U.T @ (d / nrm)).astype(np.float32) if nrm > 0 else np.zeros(U.shape[1], np.float32)
    return out


def _pct_rank(x: np.ndarray) -> np.ndarray:
    """大きいほど 1 に近い順位(0〜1)。"""
    o = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return o / max(len(x) - 1, 1)


def combine_scores(cos: np.ndarray, act_diff: np.ndarray | None, steer_kl: np.ndarray | None) -> np.ndarray:
    """順位の平均。|cos|、cos の符号にそろえた act_diff、steer KL のうち、ある指標だけを使う(NaN は除く)。"""
    parts = [_pct_rank(np.abs(cos))]
    for x in (None if act_diff is None else act_diff * np.sign(cos), steer_kl):
        if x is None:
            continue
        ok = np.isfinite(x)
        if ok.any():
            r = np.full(len(x), np.nan)
            r[ok] = _pct_rank(x[ok])
            parts.append(r)
    return np.nanmean(np.stack(parts), axis=0)


def rank_by_direction(weights, directions: dict[int, np.ndarray], *, act_diff: dict[int, np.ndarray] | None = None,
                      steer_kl: dict[int, tuple[np.ndarray, np.ndarray]] | None = None, top: int = 200,
                      exclude_layers: Sequence[int] = (), scale: dict[int, np.ndarray] | None = None,
                      ) -> tuple[list[Candidate], dict[int, np.ndarray]]:
    """全層・全ニューロンを順位づけし、上位 top 件と層ごとの cos を返す。
    steer_kl は層 → (kl_pos, kl_neg)。cos の符号に合う側(+ なら kl_pos)を使う。"""
    cos = cos_by_layer(weights, directions, scale)
    Ls = [L for L in sorted(cos) if L not in set(exclude_layers)]
    nan = {L: np.full(cos[L].size, np.nan, np.float32) for L in Ls}
    lay = np.concatenate([np.full(cos[L].size, L) for L in Ls])
    neu = np.concatenate([np.arange(cos[L].size) for L in Ls])
    c = np.concatenate([cos[L] for L in Ls])
    a = np.concatenate([act_diff.get(L, nan[L]) for L in Ls]) if act_diff else None
    k = None
    if steer_kl:
        k = np.concatenate([np.where(cos[L] >= 0, *steer_kl[L]) if L in steer_kl else nan[L] for L in Ls])
    score = combine_scores(c, a, k)
    order = np.argsort(-score, kind="stable")[:top]
    cands = [Candidate(int(lay[i]), int(neu[i]), float(c[i]), 1 if c[i] >= 0 else -1,
                       None if a is None or not np.isfinite(a[i]) else float(a[i]),
                       None if k is None or not np.isfinite(k[i]) else float(k[i]), float(score[i])) for i in order]
    if getattr(weights.meta, "n_expert", 0):
        for cd in cands:
            if weights.is_moe(cd.layer):
                cd.expert, cd.expert_neuron = weights.moe_split(cd.layer, cd.neuron)
    return cands, cos


def candidate_dicts(cands: list[Candidate]) -> list[dict]:
    return [asdict(c) for c in cands]
