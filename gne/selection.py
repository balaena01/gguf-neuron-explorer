"""ニューロンの選び方の指定(§7 の --neurons): all / random:384 / 1,5,9-12。カンマで混ぜられる(random:384,2742)。"""

from __future__ import annotations

import re

import numpy as np

from .steer import GneError


def parse_neurons(spec: str | None, n_ff: int, *, seed: int = 0) -> np.ndarray:
    """昇順のニューロン番号の配列。random:K は seed で再現できる K 個(重複なし)。"""
    if spec is None or spec == "all":
        return np.arange(n_ff)
    out: list[int] = []
    for part in spec.split(","):
        m = re.fullmatch(r"\s*random:(\d+)\s*", part)
        if m:
            k = min(int(m.group(1)), n_ff)
            out.extend(np.random.default_rng(seed).choice(n_ff, size=k, replace=False).tolist())
            continue
        m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", part)
        if not m:
            raise GneError("BAD_ARGS", f"ニューロンの指定 {spec!r} を読めません", "all / random:384 / 1,5,9-12")
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        out.extend(range(a, b + 1))
    arr = np.unique(np.array(out, dtype=np.int64))
    if arr.size and (arr[0] < 0 or arr[-1] >= n_ff):
        raise GneError("BAD_ARGS", f"ニューロン番号は 0..{n_ff - 1} です")
    return arr
