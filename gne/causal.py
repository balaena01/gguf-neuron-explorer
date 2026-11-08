"""因果スクリーニング(`gne heads causal`)と効率ベースの強さ配分(`gne heads allocate`)。CHANGE-06 C・D。

どちらも生成せず、速い指標(direction.Scorer: 測定位置の残差 · r̂_L = 方向への射影の値)だけで回す。

- 層単位(screen_layers): 層のヘッドを全部止める(残差に書き込む重み行列の出力を 0 にする LoRA)と、score_layers の A 側の射影の値が
  どれだけ下がるか。下がる層を見つける。
- ヘッド単位(screen_heads):
  - ablate: ヘッド 1 個の出力を 0 にしたときの下がり方 d_h(causal)。最初の候補出し。
  - marginal: base_units(いまの組)を当てた状態で、そのヘッドの s を delta_s だけ上げたときの、効きにくい例(hard_subset)
    での下がり方 ÷ delta_s = 効率 η_h。配り直し用。
  ablate と marginal は順位が一致しない(丸ごと止めたときの効き目と、方向成分を強めたときの効き目は別物)。
- 直接の寄与 contrib: 無介入で、ヘッドが残差に書く r̂_Sᵀ W[:, blk_h] x_h の A 側の平均(S は射影の値を測る層)。止めたときに
  直接の経路だけで見込める下がり方。ratio = causal / contrib が 1 より大きければ間接の経路(後段のユニットを通る)、
  1 より小さければ補い合い。射影の値を測る層より後ろのヘッドは contrib が 0 になる(後段の層の射影の値で見る)。
- B 側: 各ヘッドで B 側の射影の値の変化(b_side_delta)と直交成分のずれ(drift_orth_b)も測る。

射影の値の下がり方は「止める前 − 止めた後」(正 = そのヘッドが A 側の向きに押していた)。目標は A 側の射影の値を下げること
(振る舞いを足したいときは、A と B を入れ替えた対照集合で方向を作る)。
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .direction import (DirectionSet, Scorer, base_layer_of, drift, score, score_layers_default)
from .headproj import ProjBuilder, ProjHead, heads_to_units
from .reverse import ContrastSet
from .steer import GneError, Recipe

Progress = Callable[[int, int, str], None]
BIG_LORA = 4_000_000                     # これより大きい一時 LoRA は F16 で書く(層の全ヘッドを止めるとき)


class TmpLora:
    """一時的な LoRA を書いて当て、使い終わったら解放してファイルも消す(何百個も当てるので)。"""

    def __init__(self, session):
        self.s = session
        self.dir = session.store.heads_dir(session.meta.model_id) / "tmp"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n = 0

    @contextmanager
    def use(self, tensors: dict[str, tuple[np.ndarray, np.ndarray]]):
        from .ablate import write_lora_tensors
        if not tensors:
            yield None
            return
        self.n += 1
        path = self.dir / f"t{os.getpid()}-{self.n}.lora.gguf"
        size = sum(A.size + B.size for A, B in tensors.values())
        write_lora_tensors(self.s.weights, tensors, path, name="gne tmp",
                           dtype=np.float16 if size > BIG_LORA else np.float32)
        try:
            yield [(str(path), 1.0)]
        finally:
            self.s.engine.drop_lora(path)
            path.unlink(missing_ok=True)


@dataclass
class Setup:
    """スクリーニングと配分で共有するもの(射影の値の測り方・無介入の射影の値・行列の材料)。"""
    session: object
    ds: DirectionSet
    cs: ContrastSet
    scorer: Scorer
    score_layers: list[int]
    primary: int
    builder: ProjBuilder
    tmp: TmpLora
    b_idx: list[int]
    base: dict = field(default_factory=dict)     # 無介入: {"A": {L: (n,)}, "B": {L: (nb,)}, "B_resid": {L: [...]}}

    def measure(self, lora, *, a_idx: list[int] | None = None, b: bool = True) -> dict:
        """A 側(a_idx で絞れる)の射影の値と、B 側(b_idx)の射影の値と残差(直交成分のずれ用)。"""
        ga = self.scorer.run(lora, sides="A", which=None if a_idx is None else {"A": a_idx})
        out = {"A": ga["A"]["values"]}
        if b:
            gb = self.scorer.run(lora, sides="B", which={"B": self.b_idx}, keep=True)
            out["B"], out["B_resid"] = gb["B"]["values"], gb["B"]["resid"]
        return out


def prepare(session, direction: str | DirectionSet, contrast, *, score_layers=None, primary: int | None = None,
            positions: str = "last", subset: str = "all", b_examples: int = 16, direction_layer: str = "own",
            max_examples: int | None = None) -> Setup:
    from .direction import load_contrast
    ds = session.direction_set(direction) if isinstance(direction, str) else direction
    cs = contrast if isinstance(contrast, ContrastSet) else load_contrast(contrast)[0]
    base = base_layer_of(ds)
    if base is None:                              # 基準層が無ければ、ここで測って記録する(B の手順を飛ばしたとき)
        rep = score(session, ds, cs, positions=positions, subset=subset, max_examples=max_examples)
        session.warn(f"方向 {ds.name} の基準層が無いので測りました(L{rep['best_layer']}、cohen_d 最大)")
        ds = session.direction_set(ds.name) if isinstance(direction, str) else DirectionSet.load(ds.path)
        base = base_layer_of(ds)
    Ls = sorted({int(x) for x in score_layers}) if score_layers else score_layers_default(session, ds, base)
    prim = int(primary) if primary is not None else (base if base in Ls else Ls[0])
    if prim not in Ls:
        Ls = sorted(set(Ls) | {prim})
    sc = Scorer(session, ds, cs, layers=Ls, positions=positions, subset=subset, max_examples=max_examples)
    nb = sc.n("B")
    b_idx = list(range(nb)) if b_examples <= 0 or b_examples >= nb else \
        [int(x) for x in np.linspace(0, nb - 1, b_examples).round().astype(int)]
    st = Setup(session, ds, cs, sc, Ls, prim, ProjBuilder(session.weights, session.direction_set,
                                                          direction_layer=direction_layer),
               TmpLora(session), sorted(set(b_idx)))
    st.base = st.measure(None)
    return st


def _row_common(st: Setup, got: dict, ref: dict, a_ref_idx: list[int] | None = None) -> dict:
    """ref(介入前)と got(介入後)の射影の値の差。A の下がり方は層ごと、B は主の層。"""
    drop = {}
    for L in st.score_layers:
        a0 = ref["A"][L] if a_ref_idx is None else ref["A"][L][a_ref_idx]
        drop[str(L)] = float(a0.mean() - got["A"][L].mean())
    P = st.primary
    out = {"score_drop": drop, "causal": drop[str(P)], "causal_max": max(drop.values())}
    if "B" in got and "B" in ref:
        out["b_side_delta"] = float(got["B"][P].mean() - ref["B"][P].mean())
        if "B_resid" in got and "B_resid" in ref:
            out["drift_par_b"], out["drift_orth_b"] = drift(ref["B_resid"][P], got["B_resid"][P], st.scorer.r[P])
    return out


def head_layers(session, up_to: int | None = None) -> list[int]:
    arch = session.weights.arch
    return [li.index for li in arch.layers if li.head_dim and li.head_writer and (up_to is None or li.index <= up_to)]


def kind_of(session, L: int) -> str:
    return "attn" if session.weights.arch.layers[L].mixer == "attention" else "linear"


# --- 層単位 ----------------------------------------------------------------------------------------------

def screen_layers(session, ds, cs, *, mode: str = "ablate", layers=None, score_layers=None, setup: Setup | None = None,
                  progress: Progress | None = None, **kw) -> list[dict]:
    """各層のヘッドを全部止めて、score_layers の射影の値の下がり方を測る(層数ぶんの試行)。"""
    if mode != "ablate":
        raise GneError("BAD_ARGS", "層単位のスクリーニングは mode=ablate だけです(marginal はヘッド単位で)")
    st = setup or prepare(session, ds, cs, score_layers=score_layers, **kw)
    Ls = head_layers(session, max(st.score_layers)) if layers is None else [int(L) for L in layers]
    rows = []
    for i, L in enumerate(Ls):
        t0 = time.perf_counter()
        with st.tmp.use(st.builder.ablate_tensors(L)) as lora:
            got = st.measure(lora)
        rows.append({"layer": L, "kind": kind_of(session, L), "n_heads": session.weights.arch.layers[L].n_head,
                     "mode": "ablate", **_row_common(st, got, st.base), "seconds": time.perf_counter() - t0})
        if progress:
            progress(i + 1, len(Ls), f"causal: L{L}")
    return rows


# --- 直接の寄与 ------------------------------------------------------------------------------------------

def head_contrib(st: Setup, heads: list[tuple[int, int]], *, chunk: int = 4) -> dict[tuple[int, int], dict[str, float]]:
    """無介入の A 側で、ヘッドが書く r̂_Sᵀ W[:, blk_h] x_h の平均(層 S ごと。S より後ろのヘッドは 0)。"""
    from .attn import capture_attention
    s, w = st.session, st.session.weights
    Ls = sorted({L for L, _ in heads})
    U = {}                                        # (S, L) → (n_head, head_dim)
    for L in Ls:
        li = w.arch.layers[L]
        W = st.builder.W(L)
        for S in st.score_layers:
            if L <= S:
                U[(S, L)] = (st.scorer.r[S] @ W).reshape(li.n_head, li.head_dim)
    acc = {hk: {str(S): 0.0 for S in st.score_layers} for hk in heads}
    # 直接の書き込みはヘッドの出力を捕まえて測るので、捕捉の上限より長い質問は除く(止めて測る段階は長さの制限がない)
    items = [x for x in st.scorer.items["A"] if len(x[0]) <= s.engine.probe_ubatch]
    if not items:
        raise GneError("BAD_ARGS", f"A 側の質問がどれも attention の捕捉の上限 {s.engine.probe_ubatch} トークンより長く、直接の書き込みを測れません",
                       "短い質問を混ぜてください")
    n = 0
    for c0 in range(0, len(items), chunk):
        part = items[c0:c0 + chunk]
        caps = capture_attention(s, [ids for ids, _ in part], layers=Ls, want_pattern=False)
        for ac, (_, rows) in zip(caps, part, strict=True):
            n += 1
            for (L, h) in heads:
                if L not in ac.head_out:
                    continue
                x = ac.head_out[L][rows, h, :].astype(np.float64)      # (P, head_dim)
                for S in st.score_layers:
                    if (S, L) in U:
                        acc[(L, h)][str(S)] += float((x @ U[(S, L)][h]).mean())
    s.engine.free_attn_probe()
    return {hk: {k: v / max(n, 1) for k, v in d.items()} for hk, d in acc.items()}


def resolve_candidates(st: Setup, layers: list[int], candidates) -> tuple[list[tuple[int, int]], dict | None]:
    """all / contrib:N / 明示のリスト([(L, h)] か "L:Hh,…")→ (ヘッド, contrib:N のとき計算した寄与)。"""
    from .heads import parse_heads
    w = st.session.weights
    if isinstance(candidates, str) and candidates not in ("all",) and not candidates.startswith("contrib:"):
        return parse_heads(candidates, w, any_mixer=True), None
    if isinstance(candidates, list | tuple) and candidates and not isinstance(candidates, str):
        return [(int(L), int(h)) for L, h in candidates], None
    every = [(L, h) for L in layers for h in range(w.arch.layers[L].n_head)]
    if candidates in (None, "all"):
        return every, None
    m = str(candidates).split(":", 1)
    if len(m) != 2 or not m[1].isdigit():
        raise GneError("BAD_ARGS", f"--candidates {candidates!r} は all / contrib:N / L:Hh,… です")
    n = int(m[1])
    con = head_contrib(st, every)

    def key(hk):                                  # そのヘッドより後ろの射影の値を測る層のうち、一番前の層での寄与
        L = hk[0]
        S = next((S for S in st.score_layers if S >= L), None)
        return con[hk][str(S)] if S is not None else 0.0
    return sorted(every, key=lambda hk: -key(hk))[:n], con


# --- ヘッド単位 ------------------------------------------------------------------------------------------

def screen_heads(session, ds, cs, *, layers=None, candidates=None, mode: str = "ablate", base_units=None,
                 delta_s: float = 3.0, hard_subset: list[int] | None = None, score_layers=None,
                 setup: Setup | None = None, b_side_max: float | None = None, progress: Progress | None = None,
                 **kw) -> list[dict]:
    """ヘッドを 1 個ずつ測る。mode=ablate は出力を 0 にしたときの下がり方、marginal は base_units の状態から s を delta_s
    上げたときの hard_subset での下がり方 ÷ delta_s(効率 eta)。行には直接の寄与 contrib と ratio、B 側への影響も入る。
    b_side_max を渡すと |b_side_delta| がそれを超える行に excluded を付ける。"""
    if mode not in ("ablate", "marginal"):
        raise GneError("BAD_ARGS", "mode は ablate か marginal です")
    st = setup or prepare(session, ds, cs, score_layers=score_layers, **kw)
    Ls = head_layers(session, max(st.score_layers)) if layers is None else [int(L) for L in layers]
    heads, con = resolve_candidates(st, Ls, candidates)
    con = con if con is not None else head_contrib(st, heads)
    base_heads: list[ProjHead] = list(base_units or [])
    if mode == "marginal" and not base_heads:
        raise GneError("BAD_ARGS", "marginal には base_units(いまの組)が要ります(--base-recipe か --base-unit)")
    if mode == "marginal":
        if hard_subset is None:
            hard_subset = hard_examples(st, base_heads, n=16)["idx"]
        with st.tmp.use(st.builder.tensors(base_heads)) as lora:
            ref = st.measure(lora, a_idx=hard_subset)
    else:
        ref = st.base if hard_subset is None else {**st.base, "A": {L: v[hard_subset] for L, v in st.base["A"].items()}}
    rows = []
    for i, (L, h) in enumerate(heads):
        t0 = time.perf_counter()
        if mode == "ablate":
            tens = st.builder.ablate_tensors(L, [h])
        else:
            hs = [ProjHead(p.layer, p.head, p.s + (delta_s if (p.layer, p.head) == (L, h) else 0.0), p.direction)
                  for p in base_heads]
            if not any((p.layer, p.head) == (L, h) for p in base_heads):
                hs.append(ProjHead(L, h, float(delta_s), st.ds.name))
            tens = st.builder.tensors(hs)
        with st.tmp.use(tens) as lora:
            got = st.measure(lora, a_idx=hard_subset)
        row = {"layer": L, "head": h, "kind": kind_of(session, L), "mode": mode, **_row_common(st, got, ref)}
        cb = con.get((L, h), {})
        row["contrib_by_layer"] = cb
        row["contrib"] = cb.get(str(st.primary))
        c = row["contrib"]
        row["ratio"] = (row["causal"] / c) if c is not None and abs(c) > 1e-9 else None
        if mode == "marginal":
            row["delta_s"] = float(delta_s)
            row["eta"] = row["causal"] / delta_s
            row["eta_by_layer"] = {k: v / delta_s for k, v in row["score_drop"].items()}
            row["in_base"] = any((p.layer, p.head) == (L, h) for p in base_heads)
        if b_side_max is not None and row.get("b_side_delta") is not None:
            row["excluded"] = abs(row["b_side_delta"]) > b_side_max
        row["seconds"] = time.perf_counter() - t0
        rows.append(row)
        if progress:
            progress(i + 1, len(heads), f"causal {mode}: L{L} H{h}")
    return rows


def spearman(a: list[float], b: list[float]) -> float | None:
    """順位相関(同順位は平均順位)。V28。"""
    if len(a) < 3:
        return None

    def ranks(x):
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[order] = np.arange(len(x))
        for v in np.unique(x):
            m = x == v
            r[m] = r[m].mean()
        return r
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


# --- 保存 ------------------------------------------------------------------------------------------------

def causal_paths(session, name: str) -> tuple[Path, Path]:
    d = session.store.heads_dir(session.meta.model_id)
    return d / f"{name}.causal.jsonl", d / f"{name}.causal.json"


def save_causal(session, name: str, rows: list[dict], summary: dict) -> tuple[Path, Path]:
    """<name>.causal.jsonl(ヘッドの行。同じ mode・同じヘッドの行は置き換え、他の mode の行は残す)と .causal.json(まとめ)。"""
    pj, ps = causal_paths(session, name)
    keep = []
    new = {(r["mode"], r["layer"], r["head"]) for r in rows}
    if pj.exists():
        for ln in pj.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                if (r.get("mode"), r.get("layer"), r.get("head")) not in new:
                    keep.append(r)
    allr = keep + rows
    allr.sort(key=lambda r: (r["mode"], -((r.get("eta") if r["mode"] == "marginal" else r.get("causal_max")) or 0)))
    pj.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in allr), encoding="utf-8")
    old = json.loads(ps.read_text(encoding="utf-8")) if ps.exists() else {}
    modes = dict(old.get("modes") or {})
    for m in {r["mode"] for r in rows}:
        modes[m] = summary
    out = {**old, **{k: v for k, v in summary.items() if k != "layer_rows"}, "modes": modes,
           "layer_rows": summary.get("layer_rows") or old.get("layer_rows") or []}
    ps.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return pj, ps


def load_causal(session, name: str | Path, mode: str | None = None) -> list[dict]:
    p = Path(name)
    if not (p.suffix == ".jsonl" and p.exists()):
        p = causal_paths(session, str(name))[0]
    if not p.exists():
        raise GneError("FILE_NOT_FOUND", f"因果スクリーニングの結果 {p} がありません", "gne heads causal で作ってください")
    rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [r for r in rows if mode is None or r.get("mode") == mode]


def list_causal(session) -> list[dict]:
    d = session.store.heads_dir(session.meta.model_id)
    out = []
    for p in sorted(d.glob("*.causal.json")):
        try:
            x = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        out.append({"name": p.name[:-len(".causal.json")], "direction": x.get("direction"),
                    "score_layers": x.get("score_layers"), "primary": x.get("primary"),
                    "modes": sorted((x.get("modes") or {}).keys()), "updated": x.get("updated")})
    return out


# --- 効きにくい例 ----------------------------------------------------------------------------------------

def hard_examples(st: Setup, heads: list[ProjHead], *, n: int = 16, layer: int | None = None) -> dict:
    """いまの組を当てた状態で、A 側のうち射影の値が高く残る上位 n 件(Scorer の添字・対照集合の番号・プロンプト・射影の値)。"""
    L = st.primary if layer is None else layer
    with st.tmp.use(st.builder.tensors(heads)) as lora:
        got = st.scorer.run(lora, sides="A")
    v = got["A"]["values"][L]
    order = [int(i) for i in np.argsort(-v)[:max(1, min(n, len(v)))]]
    return {"idx": order, "index": [st.scorer.index[i] for i in order], "prompts": [st.scorer.prompts[i] for i in order],
            "scores": [float(v[i]) for i in order], "layer": L, "heads": [p.spec() for p in heads]}


def save_hard(session, name: str, hard: dict) -> Path:
    p = session.store.hard_dir(session.meta.model_id) / f"{name}.json"
    p.write_text(json.dumps({k: v for k, v in hard.items() if k != "idx"}, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def load_hard(session, st: Setup, name_or_path: str | Path) -> list[int]:
    """保存した効きにくい例 → いまの Scorer の添字(プロンプトの文で引き直す)。"""
    p = Path(name_or_path)
    if not p.exists():
        p = session.store.hard_dir(session.meta.model_id) / f"{name_or_path}.json"
    if not p.exists():
        raise GneError("FILE_NOT_FOUND", f"効きにくい例 {name_or_path} がありません")
    d = json.loads(p.read_text(encoding="utf-8"))
    pos = {q: i for i, q in enumerate(st.scorer.prompts)}
    idx = [pos[q] for q in d.get("prompts", []) if q in pos]
    if not idx:
        raise GneError("BAD_ARGS", f"{p} のプロンプトがいまの対照集合(subset)にありません")
    return idx


# --- 配分 ------------------------------------------------------------------------------------------------

GREEDY_MAX_STEPS = 300


def _bisect(f: Callable[[float], float], target: float, hi: float, *, iters: int = 14, tol: float = 0.02,
            grid: int = 1):
    """f(α) が target 以下になる最小の α を [0, hi] で探す。戻り値 (α, f(α), 届いたか, 回数)。
    grid > 1 なら先に hi/grid 刻みで粗く見て、最初に target を下回った区間の中で二分探索する(f が単調でない方式用。
    届かなければ f が最小の点を返す)。grid = 1 は f が単調に減るとみなして上限だけを先に見る。"""
    n = 0
    pts = [hi * (k + 1) / grid for k in range(grid)]
    lo, found, best = 0.0, None, None
    for a in pts:
        v = f(a)
        n += 1
        if best is None or v < best[1]:
            best = (a, v)
        if v <= target:
            found = (a, v)
            break
        lo = a
    if found is None:
        return best[0], best[1], False, n
    hi, res = found[0], found
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        v = f(mid)
        n += 1
        if v > target:
            lo = mid
        else:
            hi, res = mid, (mid, v)
        if abs(v - target) <= tol * max(1.0, abs(target)):
            break
    return res[0], res[1], True, n


def allocate(session, ds, cs, *, candidates, target: float, score_layer: int, hard_subset=None,
             n_heads=(6, 5, 4), proportional_to: str = "eta", from_zero: bool = True, b_side_max: float | None = None,
             base_units: list[ProjHead] | None = None, delta_s: float = 3.0, hard: int = 16, pool: int = 30,
             s_max: float | None = 30.0, remain_threshold: float = 0.0, name: str | None = None, positions: str = "last",
             subset: str = "all", direction_layer: str = "own", score_layers=None, b_examples: int = 16,
             max_examples: int | None = None, drift_max: float | None = 0.5, method: str = "proportional",
             progress: Progress | None = None) -> dict:
    """効率ベースの強さ配分(CHANGE-06 D)。
    1. いまの組(base_units。無ければ candidates の ablate の行の上位 max(n_heads) 個に causal 比例で、A 側の平均が target に
       なるよう二分探索した組)を当てた状態で、A 側の射影の値が高く残る hard 件を効きにくい例にする(hard/<model_id>/<name>.json)。
    2. いまの組 + causal の上位 pool 個の効率 η を marginal で測る。
    3. n_heads の各値について、η の上位 n 個に s ∝ η(from_zero=False ならいまの s に足す)で配り、効きにくい例の射影の値の
       平均が target になるまで全体の大きさを二分探索する(s の最大は s_max)。
    4. 比較表(ヘッド数・各ヘッドの s・最大の s・A 側で remain_threshold を超えて残る件数・A 側の射影の値の最大・B 側の直交成分の
       ずれ・所要時間)と、残る件数と B 側のずれが同程度なら最少のヘッド数の組に recommended を付ける。
    method="greedy" は 3 を置き換える。η の上位 n 個を候補に、毎回いちばん射影の値が下がるヘッドの s だけを delta_s 上げる。比を固定
    しないので、ヘッドの効き方が直線でなくても追える。止まるのは、目標に届いたとき(stop=target)、B 側の直交成分のずれが drift_max を
    超えるとき(drift。超える手前の組を使う)、どのヘッドも下がらないとき(no_gain)、s_max を渡して全部が当たったとき(s_max)。
    目標は射影の値の目標で、振る舞いの目標ではない。止まった組は振る舞いで確かめる。"""
    from .direction import load_contrast
    t_all = time.perf_counter()
    if proportional_to not in ("eta", "causal"):
        raise GneError("BAD_ARGS", "proportional_to は eta か causal です")
    if method not in ("proportional", "greedy"):
        raise GneError("BAD_ARGS", "method は proportional か greedy です")
    if method == "proportional" and s_max is None:
        s_max = 30.0
    ds = session.direction_set(ds) if isinstance(ds, str) else ds
    cs = cs if isinstance(cs, ContrastSet) else load_contrast(cs)[0]
    sl = sorted({int(x) for x in (score_layers or [])} | {int(score_layer)})
    st = prepare(session, ds, cs, score_layers=sl, primary=int(score_layer), positions=positions, subset=subset,
                 b_examples=b_examples, direction_layer=direction_layer, max_examples=max_examples)
    S = int(score_layer)
    name = name or ds.name
    steps = [0]

    def tick(msg):
        steps[0] += 1
        if progress:
            progress(steps[0], max(steps[0], est), msg)
    rows_in = [r for r in (candidates or []) if isinstance(r, dict)]
    pairs = [(int(c[0]), int(c[1])) for c in (candidates or []) if not isinstance(c, dict)]
    if pairs:                                     # ablate の行が無いヘッドは先に測る
        rows_in += screen_heads(session, ds, cs, candidates=pairs, mode="ablate", setup=st)
    abl = [r for r in rows_in if r.get("mode", "ablate") == "ablate" and not r.get("excluded")]

    def causal_at(r):
        return (r.get("score_drop") or {}).get(str(S), r.get("causal")) or 0.0
    abl.sort(key=lambda r: -causal_at(r))
    if not abl and not base_units:
        raise GneError("BAD_ARGS", "候補(heads causal の ablate の行)か base_units が要ります")
    est = 1 + pool + (sum(int(x) for x in n_heads) * 20 if method == "greedy" else len(n_heads) * 16)
    a_all = list(range(st.scorer.n("A")))

    def mean_score(heads: list[ProjHead], idx: list[int]) -> float:
        with st.tmp.use(st.builder.tensors([p for p in heads if p.s])) as lora:
            got = st.scorer.run(lora, sides="A", which={"A": idx})
        tick("allocate")
        return float(got["A"]["values"][S].mean())

    # 1. いまの組
    if base_units:
        cur = list(base_units)
        init = None
    else:
        top = [r for r in abl if causal_at(r) > 0][:max(n_heads)]
        if not top:
            raise GneError("BAD_ARGS", f"L{S} の射影の値を下げるヘッドが候補にありません(causal ≤ 0)")
        wmax = max(causal_at(r) for r in top)
        wts = [(r["layer"], r["head"], causal_at(r) / wmax) for r in top]
        # 仮の組は効きにくい例を選ぶためのもの。greedy で s_max が無くても、壊れる強さの状態で η を測らないように上限を置く
        alpha, val, reached, _ = _bisect(lambda a: mean_score([ProjHead(L, h, a * w, ds.name) for L, h, w in wts], a_all),
                                         target, 30.0 if s_max is None else s_max)
        cur = [ProjHead(L, h, alpha * w, ds.name) for L, h, w in wts]
        init = {"heads": [p.spec() for p in cur], "alpha": alpha, "mean_a": val, "reached": reached}
        ipath = session.store.heads_dir(session.meta.model_id) / f"{name}.alloc-initial.yaml"   # 効率で配り直す前の組も試せるように残す
        Recipe(heads_to_units(cur, ds.name), session.meta.model_id, name=f"{name}-initial", default_direction=ds.name,
               direction_layer=direction_layer).save(ipath)
        init["recipe"] = str(ipath)
    # 効きにくい例
    if hard_subset is not None:
        h_idx = load_hard(session, st, hard_subset) if isinstance(hard_subset, str | Path) else list(hard_subset)
        hard_rep = None
    else:
        hard_rep = hard_examples(st, cur, n=hard, layer=S)
        h_idx = hard_rep["idx"]
        save_hard(session, name, hard_rep)
    # 2. 効率
    seen = {(p.layer, p.head) for p in cur}
    cand = [(p.layer, p.head) for p in cur] + [(r["layer"], r["head"]) for r in abl
                                               if (r["layer"], r["head"]) not in seen][:pool]
    mrows = screen_heads(session, ds, cs, candidates=cand, mode="marginal", base_units=cur, delta_s=delta_s,
                         hard_subset=h_idx, setup=st, b_side_max=b_side_max)
    for r in mrows:                                # s を少し上げただけで中立の残差が大きく動くヘッドは配らない
        if drift_max is not None and (r.get("drift_orth_b") or 0) > drift_max:
            r["excluded"] = True
    by = {(r["layer"], r["head"]): r for r in mrows}
    cz = {(r["layer"], r["head"]): causal_at(r) for r in abl}

    def weight(r):
        return r["eta"] if proportional_to == "eta" else cz.get((r["layer"], r["head"]), 0.0)
    elig = [r for r in mrows if weight(r) > 0 and not r.get("excluded")]
    elig.sort(key=lambda r: -weight(r))
    def greedy(keys: list[tuple[int, int]]) -> tuple[list[ProjHead], float, bool, int, str]:
        """毎回いちばん射影の値が下がるヘッドだけを delta_s 上げる。戻り値 (組, 効きにくい例の平均, 届いたか, 回数, 止まった理由)。"""
        cur_s: dict[tuple[int, int], float] = {}
        val = float(st.base["A"][S][h_idx].mean())
        make_ = lambda d: [ProjHead(L, h, x, ds.name) for (L, h), x in d.items()]   # noqa: E731
        stop, n_eval = "steps", 0
        for _ in range(GREEDY_MAX_STEPS):
            best = None
            for k in keys:
                x = cur_s.get(k, 0.0) + delta_s
                if s_max is not None and x > s_max:
                    continue
                trial = {**cur_s, k: x}
                v = mean_score(make_(trial), h_idx)
                n_eval += 1
                if best is None or v < best[1]:
                    best = (trial, v)
            if best is None:
                stop = "s_max"
                break
            if best[1] >= val:
                stop = "no_gain"
                break
            if drift_max is not None:
                with st.tmp.use(st.builder.tensors(make_(best[0]))) as lora:
                    gb_ = st.scorer.run(lora, sides="B", which={"B": st.b_idx}, keep=True)
                if drift(st.base["B_resid"][S], gb_["B"]["resid"][S], st.scorer.r[S])[1] > drift_max:
                    stop = "drift"
                    break
            cur_s, val = best
            if val <= target:
                stop = "target"
                break
        return make_(cur_s), val, stop == "target", n_eval, stop

    # 3. 配る
    table = []
    b_all0 = st.scorer.run(None, sides="B")["B"]["values"][S]
    for n in sorted({int(x) for x in n_heads}, reverse=True):
        t0 = time.perf_counter()
        if len(elig) < n:
            table.append({"n_heads": n, "skipped": f"効率が正のヘッドが {len(elig)} 個しかありません"})
            continue
        pick = elig[:n]
        wmax = max(weight(r) for r in pick)
        w = {(r["layer"], r["head"]): weight(r) / wmax for r in pick}
        s0 = {} if from_zero else {(p.layer, p.head): p.s for p in cur}

        def make(a, w=w, s0=s0):
            return [ProjHead(L, h, s0.get((L, h), 0.0) + a * wt, ds.name) for (L, h), wt in w.items()]
        capped = None
        if method == "greedy":
            heads, val, reached, n_iter, stop = greedy([(r["layer"], r["head"]) for r in pick])
            alpha = None
            if not heads:
                table.append({"n_heads": n, "skipped": "どのヘッドも射影の値を下げませんでした", "stop": stop})
                continue
        else:
            hi = s_max - max(s0.values(), default=0.0)
            alpha, val, reached, n_iter = _bisect(lambda a: mean_score(make(a), h_idx), target, max(hi, 0.0))
            heads = make(alpha)
            stop = "target" if reached else "s_max"
            if not reached:                        # 比を固定しているので、上限に当たったヘッドが全体を止める
                capped = next(({"layer": p.layer, "head": p.head, "s": p.s} for p in heads if w[(p.layer, p.head)] == 1.0), None)
        with st.tmp.use(st.builder.tensors(heads)) as lora:
            got = st.scorer.run(lora, sides="AB", which={"A": a_all})
            gb = st.scorer.run(lora, sides="B", which={"B": st.b_idx}, keep=True)
        va = got["A"]["values"][S]
        par, orth = drift(st.base["B_resid"][S], gb["B"]["resid"][S], st.scorer.r[S])
        units = heads_to_units(heads, ds.name)
        rec = Recipe(units, session.meta.model_id, name=f"{name}-n{n}", default_direction=ds.name,
                     direction_layer=direction_layer)
        rpath = session.store.heads_dir(session.meta.model_id) / f"{name}.alloc-n{n}.yaml"
        rec.save(rpath)
        table.append({"n_heads": n, "heads": [{"layer": p.layer, "head": p.head, "s": p.s,
                                               "eta": by[(p.layer, p.head)]["eta"]} for p in heads],
                      "units": [p.spec() for p in heads], "s_max": max(p.s for p in heads), "alpha": alpha,
                      "hard_mean": val, "reached": reached, "iterations": n_iter, "stop": stop, "capped": capped, "n_used": len(heads),
                      "remain": int((va > remain_threshold).sum()), "a_max": float(va.max()), "a_mean": float(va.mean()),
                      "b_mean": float(got["B"]["values"][S].mean()),
                      "b_side_delta": float(got["B"]["values"][S].mean() - b_all0.mean()),
                      "b_drift_par": par, "b_drift_orth": orth, "recipe": str(rpath), "seconds": time.perf_counter() - t0})
    # 4. 推す組
    ok = [r for r in table if r.get("reached")]
    rec_n = None
    if ok:
        r_min = min(r["remain"] for r in ok)
        tol_n = max(1, int(np.ceil(0.05 * len(a_all))))      # 残る件数は A 側の 5%(最低 1 件)まで同程度とみなす
        near = [r for r in ok if r["remain"] <= r_min + tol_n]
        d_min = min(r["b_drift_orth"] for r in near)
        same = [r for r in near if r["b_drift_orth"] <= d_min * 1.25 + 1e-4]   # B 側のずれは最小の 1.25 倍まで
        rec_n = min(r["n_heads"] for r in same)
    for r in table:
        r["recommended"] = r.get("n_heads") == rec_n and rec_n is not None
    out = {"name": name, "direction": ds.name, "contrast_sha": cs.sha, "score_layer": S, "score_layers": st.score_layers,
           "max_examples": max_examples, "drift_max": drift_max,
           "target": target, "positions": positions, "subset": subset, "direction_layer": direction_layer,
           "method": method, "proportional_to": proportional_to, "from_zero": from_zero, "delta_s": delta_s, "s_max": s_max,
           "remain_threshold": remain_threshold, "n_a": len(a_all), "n_b": len(st.b_idx),
           "base_a_mean": float(st.base["A"][S].mean()), "base_b_mean": float(st.base["B"][S].mean()),
           "initial": init, "current": [p.spec() for p in cur], "hard": {"n": len(h_idx), **({k: v for k, v in hard_rep.items() if k != "idx"} if hard_rep else {})},
           "eta": [{k: r.get(k) for k in ("layer", "head", "kind", "eta", "causal", "b_side_delta", "drift_orth_b",
                                           "excluded", "in_base")} for r in sorted(mrows, key=lambda r: -r["eta"])],
           "table": table, "recommended": rec_n, "seconds": time.perf_counter() - t_all}
    if progress:                                   # 見積もりの回数に届かなくても、進捗の行を閉じる
        progress(steps[0], steps[0], "allocate")
    p = session.store.heads_dir(session.meta.model_id) / f"{name}.alloc.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    out["path"] = str(p)
    return out
