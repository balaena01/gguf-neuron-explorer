"""ランダム対照(CHANGE-03 A): 方向やレシピの効果が「同じ大きさならどの向きを足しても出る」ものではないかを確かめる。

ニューロンの sweep と同じ考え方で、同じノルムで向きだけ一様ランダムなベクトルを対照にする。r_L と直交化はしない。
- control_vectors: 方向 r_L の「足す」測り方の対照(層ごとに同じノルムのランダムなベクトル)。
- control_recipes: レシピの add ユニット(ニューロン・方向・ランダム)を、それぞれ同じ層・同じ rel の大きさの
  ランダムなユニットに差し替えた対照レシピ。ablate はランダムな単位ベクトルの消去("random-SEED")に差し替える。
  gain・query のユニットはそのまま残す(向きを持たないので)。
- null_pct: 効果が対照の何割より大きいか(1.0 = 全対照より上)。sweep の null_pct と同じ定義。
"""

from __future__ import annotations

import numpy as np

from .steer import AblateSpec, Recipe, Unit, random_direction

DEFAULT_CONTROLS = 16
SEED_BASE = 7919                    # 対照の seed(層・ユニットごとに seed_base + j·SEED_STRIDE + i)
SEED_STRIDE = 1009


def control_vector(n_embd: int, norm: float, j: int, layer: int) -> np.ndarray:
    """層 layer の j 本目の対照: ノルム norm の一様ランダムなベクトル(seed 固定)。"""
    return (norm * random_direction(n_embd, SEED_BASE + j, layer)).astype(np.float32)


def control_vectors(n_embd: int, vec: np.ndarray, layer: int, n: int) -> list[np.ndarray]:
    norm = float(np.linalg.norm(vec))
    return [control_vector(n_embd, norm, j, layer) for j in range(n)]


def control_recipes(session, recipe: Recipe, n: int) -> list[Recipe]:
    """レシピの add ユニットを、同じ層・同じ rel の大きさ(‖v‖ / ρ_L)のランダムなユニットに、ablate をランダムな
    向きの消去に差し替えた n 本。add ユニットも ablate も無ければ(gain だけ)空のリスト(対照は作れない)。"""
    if n <= 0 or not (recipe.add_units or recipe.ablate is not None):
        return []
    details = session.vectors(recipe)[1] if recipe.add_units else [{"rel_push": None}] * len(recipe.units)
    rel = {}
    for i, (u, d) in enumerate(zip(recipe.units, details, strict=True)):
        if u.mode == "add":
            rel[i] = float(d["rel_push"] or 0.0)
    out = []
    for j in range(n):
        units = []
        for i, u in enumerate(recipe.units):
            if u.mode != "add":                         # gain・query は向きを持たないのでそのまま
                units.append(u)
            else:
                units.append(Unit(u.layer, None, rel[i], kind="random", seed=SEED_BASE + j * SEED_STRIDE + i))
        ab = recipe.ablate
        if ab is not None:
            ab = AblateSpec(f"random-{SEED_BASE + j}", ab.scale, ab.layers, None)
        out.append(Recipe(units, recipe.model_id, "rel", name=f"{recipe.name or 'recipe'}-random{j}", when=recipe.when,
                          ablate=ab, sampling=recipe.sampling))
    return out


def head_control_recipes(session, recipe: Recipe, n: int) -> list[Recipe]:
    """ヘッドのランダム対照(CHANGE-04 F): レシピのヘッドのユニットを、同じ種類(attention / 線形注意)のランダムなヘッドに
    同じ k で差し替えた n 本。ヘッド以外のユニットと ablate はそのまま残す。ヘッドが無ければ空のリスト。"""
    heads = [u for u in recipe.units if u.kind == "head"]
    if n <= 0 or not heads:
        return []
    arch = session.weights.arch
    pool: dict[str, list[tuple[int, int]]] = {}
    for li in arch.layers:
        if li.head_dim and li.head_writer:
            pool.setdefault(li.mixer, []).extend((li.index, h) for h in range(li.n_head))
    used = {(u.layer, u.head) for u in heads}
    out = []
    for j in range(n):
        rng = np.random.default_rng([SEED_BASE, j])
        taken = set(used)
        units = []
        for u in recipe.units:
            if u.kind != "head":
                units.append(u)
                continue
            cand = [x for x in pool[arch.layers[u.layer].mixer] if x not in taken]
            L, h = cand[int(rng.integers(len(cand)))]
            taken.add((L, h))
            units.append(Unit(L, None, 0.0, "head", mode="gain", k=u.k, head=h))
        out.append(Recipe(units, recipe.model_id, recipe.dose_unit, name=f"{recipe.name or 'recipe'}-randhead{j}",
                          when=recipe.when, ablate=recipe.ablate, sampling=recipe.sampling))
    return out


def null_pct(value: float | None, ctrl: list[float | None]) -> float | None:
    """value より小さい対照の割合。value か対照が無ければ None。"""
    xs = np.sort([float(x) for x in ctrl if x is not None])
    if value is None or xs.size == 0:
        return None
    return float(np.searchsorted(xs, float(value), side="left") / xs.size)


def summarize(values: list[float | None]) -> dict:
    """対照の値の分布(p05・p50・p95・max)。"""
    xs = np.array([float(x) for x in values if x is not None])
    if xs.size == 0:
        return {"n": 0, "p05": None, "p50": None, "p95": None, "max": None}
    return {"n": int(xs.size), "p05": float(np.percentile(xs, 5)), "p50": float(np.percentile(xs, 50)),
            "p95": float(np.percentile(xs, 95)), "max": float(xs.max())}
