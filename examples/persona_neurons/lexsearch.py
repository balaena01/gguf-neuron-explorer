"""logit lens から、部品の語を書き込むニューロンを探す(docs/persona_workflow.md の手順 3)。

各層のニューロンの「押すと増える語」(up)と「減る語」(down)の上位に、設定の terms が出るものを集める。
up で当たったものはプラス、down で当たったものはマイナスで押す候補になる。anti の部品はその逆(抑える候補)。
logit lens が無い層は計算する(重みだけ、CPU。27B で 1 層 1〜2 分)。

使い方: python lexsearch.py MODEL CONFIG.yaml OUT.json [--layers 35-50] [--top 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config

from gne.api import Session
from gne.lens import LogitLens

SUFFIXES = ("", "s", "es", "ed", "ing", "y", "ly", "ness", "ive", "ion")


def matches(token: str, term: str) -> bool:
    """英字の語は「語そのもの＋決まった語尾」だけ(mine が determine に当たらない)。それ以外(中国語など)は部分一致。"""
    if term.isascii() and term.isalpha():
        return token.isalpha() and any(token == term + s for s in SUFFIXES)
    return term in token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--layers", default=None, help="例 35-50(既定は最終層を除く全層)")
    ap.add_argument("--top", type=int, default=8, help="上位何語までを見るか")
    a = ap.parse_args()
    cfg = load_config(a.config)
    s = Session(a.model)
    if a.layers:
        lo, hi = (int(v) for v in a.layers.split("-"))
        layers = list(range(lo, hi + 1))
    else:
        layers = list(range(s.meta.n_layer - 1))
    words = [b.decode("utf-8", "replace").strip().lower() for b in s.vocab.pieces]
    groups = [("part", k, v.get("terms", [])) for k, v in cfg["parts"].items()] + [("anti", k, v) for k, v in cfg["anti"].items()]
    hits = {k: [] for _, k, _ in groups}
    ldir = s.store.lens_dir(s.meta.model_id)
    for L in layers:
        path = ldir / f"logit_L{L}.npz"
        if not path.exists():
            print(f"logit lens L{L} を計算します", flush=True)
            s.lens_logit(L)
        ll = LogitLens.load(path)
        for side, ids, zs in (("up", ll.up_ids, ll.up_z), ("down", ll.down_ids, ll.down_z)):
            for row in range(ids.shape[0]):
                toks = [words[int(t)] for t in ids[row][:a.top]]
                for kind, name, terms in groups:
                    pos = next((i for i, tk in enumerate(toks) if tk and any(matches(tk, t.lower()) for t in terms)), None)
                    if pos is None:
                        continue
                    sign = "+" if side == "up" else "-"
                    if kind == "anti":
                        sign = "-" if sign == "+" else "+"           # 反対の語を増やすニューロンは、マイナスで押して抑える
                    hits[name].append({"kind": kind, "layer": L, "neuron": int(ll.neurons[row]), "side": side, "sign": sign,
                                       "pos": pos, "token": toks[pos], "z": float(zs[row][pos]), "tokens": toks})
    for hs in hits.values():
        hs.sort(key=lambda h: (h["pos"], -abs(h["z"])))
    Path(a.out).write_text(json.dumps({"layers": layers, "hits": hits}, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, hs in hits.items():
        print(f"== {name}: {len(hs)} 個")
        for h in hs[:8]:
            print(f"   {h['layer']}:{h['neuron']}:{h['sign']}  {h['side']} #{h['pos']} {h['token']!r} z {h['z']:+.1f}  {h['tokens'][:6]}")


if __name__ == "__main__":
    main()
