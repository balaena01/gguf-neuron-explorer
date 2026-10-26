"""UI が読むデータ(別冊 §4.3 overlay、§4.6 hover、§5 NEURON、§6 layers ビュー)。

保存済みの結果(sweep の units / rollout、steer lens、logit lens、逆引きの方向、活性統計、名前)を
層ごとに読み、ファイルの更新時刻が変わるまでメモリに置く。計算はしない(逆引きの cos だけは重みとの内積)。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from ..dossier import _directions, _lens_paths, _names, _read_jsonl, _units_files
from ..steer import GneError

FIELDS = ("sweep", "sweep_null", "steer_kl", "logit_peak", "act_q999", "named")   # と "cos:<方向の名前>"
HEAVY = ("token_ids", "logprob", "top_ids", "top_lps")                             # rollout のうち UI で使わない欄


def piece_text(b: bytes) -> str:
    """トークンのバイト列を文字列に。UTF-8 として途中で切れているもの(バイト単位の BPE の断片)は <0xE6> の形で見せる。"""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return "".join(f"<0x{x:02X}>" for x in b)


def _mtime(p: Path) -> float | None:
    try:
        return p.stat().st_mtime
    except OSError:
        return None


class UIData:
    def __init__(self, session):
        self.s = session
        self._cache: dict = {}
        self._locks: dict = {}
        self._guard = threading.Lock()

    # --- キャッシュ(キーごとのロック。ファイルの更新時刻が変わったら読み直す) --------------------------------
    def _get(self, key, paths: list[Path], build):
        stamp = tuple(_mtime(p) for p in paths)
        hit = self._cache.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            hit = self._cache.get(key)
            if hit is not None and hit[0] == stamp:
                return hit[1]
            val = build()
            self._cache[key] = (stamp, val)
            return val

    def vocab_text(self) -> list[str]:
        if "vocab" not in self._cache:
            self._cache["vocab"] = (None, [piece_text(p) for p in self.s.vocab.pieces])
        return self._cache["vocab"][1]

    def tok(self, ids) -> list[str]:
        """語番号 → 文字列(推論エンジンを読み込まずに、語彙だけから)。-1 は飛ばす。"""
        v = self.vocab_text()
        return [v[int(i)] for i in ids if 0 <= int(i) < len(v)]

    def _check(self, L: int) -> int:
        if not 0 <= L < self.s.meta.n_layer:
            raise GneError("BAD_ARGS", f"層 {L} は範囲外です")
        return self.s.meta.n_ff_at(L)

    def names(self) -> dict[str, str]:
        p = self.s.store.root / "names" / f"{self.s.meta.model_id}.json"
        return self._get("names", [p], lambda: _names(self.s))

    def steer_lens(self, L: int):
        from ..lens import SteerLens
        _, sp = _lens_paths(self.s, L)
        return self._get(("steer", L), [sp], lambda: SteerLens.load(sp) if sp.exists() else None)

    def logit_lens(self, L: int):
        from ..lens import LogitLens
        lp, _ = _lens_paths(self.s, L)
        return self._get(("logit", L), [lp], lambda: LogitLens.load(lp) if lp.exists() else None)

    def act(self, L: int):
        from ..stats import ActStats
        ap = self.s.store.act_path(self.s.meta.model_id, L)
        return self._get(("act", L), [ap], lambda: ActStats.load(ap) if ap.exists() else None)

    def cos(self, name: str, L: int) -> np.ndarray | None:
        from ..dossier import _cos_for_layer
        path = dict(_directions(self.s)).get(name)
        if path is None:
            raise GneError("NOT_FOUND", f"逆引きの方向 {name!r} がありません", "gne reverse --contrast で作る")
        return self._get(("cos", name, L), [path], lambda: _cos_for_layer(self.s, np.load(path), L))

    def connections(self):
        from ..connections import Connections
        p = self.s.store.connections_path(self.s.meta.model_id)
        return self._get("connections", [p], lambda: Connections(p) if p.exists() else None)

    def layout(self, L: int, kind: str) -> np.ndarray | None:
        p = self.s.store.layout_dir(self.s.meta.model_id) / f"{kind}_L{L}.npy"
        return self._get(("layout", kind, L), [p], lambda: np.load(p).astype(np.int32) if p.exists() else None)

    def layouts(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for p in sorted(self.s.store.layout_dir(self.s.meta.model_id).glob("*_L*.npy")):
            kind, L = p.stem.rsplit("_L", 1)
            out.setdefault(kind, []).append(int(L))
        return {k: sorted(v) for k, v in out.items()}

    def links(self, L: int, n: int, k: int = 14) -> dict:
        """開いたニューロンの connections(Details と direct links)。相手の lens は、押される向きの極(書き込む先)と、
        相手が + に光ったときに書く語(駆動するもの)。"""
        from ..connections import lens_tokens
        c = self.connections()
        if c is None:
            raise GneError("NOT_FOUND", "connections がありません", "層の右クリックの「connections」(全層ペア、CPU)で作れます")
        self._check(L)
        out = {}
        for way in ("down", "up"):
            rows = []
            for (a, b, z, s) in c.edges(L, n, way, k):
                pole = ("+" if s > 0 else "-") if way == "down" else "+"
                words = [t for t in self.tok(lens_tokens(self, a, b, pole, 6)) if not t.startswith("<0x")][:4]   # バイト単位の語は外す
                rows.append({"layer": a, "neuron": b, "z": round(z, 2), "sign": s, "lens": words,
                             "name": self.names().get(f"{a}:{b}")})
            out[way] = rows
        return {"layer": L, "neuron": n, **out, "meta": {k_: c.meta[k_] for k_ in ("k", "window", "rho_scaled")}}

    def runs(self, L: int) -> list[dict]:
        """層 L の sweep の run(units.jsonl ごと)。ニューロンの多い順。"""
        files = [(p, d) for p, layer, d in _units_files(self.s) if layer == L]

        def build():
            out = []
            for p, design in files:
                units = _read_jsonl(p)
                neu = [u for u in units if u["kind"] == "neuron"]
                ctrl = [u for u in units if u["kind"] == "random"]
                out.append({"run": p.name.removesuffix(".units.jsonl"), "design": design, "units": units,
                            "n_neurons": len(neu), "n_controls": len(ctrl),
                            "floor": max((u["mass"] for u in ctrl), default=None)})
            out.sort(key=lambda r: -r["n_neurons"])
            return out
        return self._get(("runs", L), [p for p, _ in files], build)

    def _run(self, L: int, run: str | None) -> dict | None:
        rs = self.runs(L)
        if run is None:
            return rs[0] if rs else None
        hit = next((r for r in rs if r["run"] == run), None)
        if hit is None:
            raise GneError("NOT_FOUND", f"層 {L} に run {run!r} がありません")
        return hit

    def records(self, run: str) -> tuple[list[dict], dict]:
        """run の rollout(重い欄を除く)と meta.json。"""
        p = self.s.run_path(run)
        meta = p.with_name(p.name.removesuffix(".jsonl") + ".meta.json")

        def build():
            recs = []
            for r in _read_jsonl(p):
                for k in HEAVY:
                    r.pop(k, None)
                recs.append(r)
            return recs, (json.loads(meta.read_text()) if meta.exists() else {})
        return self._get(("records", run), [p, meta], build)

    def warm(self) -> None:
        """初回の待ちを減らすため、裏で先に読んでおく(rollout と、全層の逆引き cos)。失敗しても UI は動く。"""
        try:
            for p, _L, _ in _units_files(self.s):
                self.records(p.name.removesuffix(".units.jsonl"))
            for name, path in _directions(self.s):
                for L in (int(x) for x in np.load(path)["layers"]):
                    if 0 <= L < self.s.meta.n_layer:
                        self.cos(name, L)
        except Exception:  # noqa: BLE001
            pass

    # --- overlay の値(§4.3): Float32Array[n_ff]、データの無いニューロンは NaN --------------------------------
    def field(self, L: int, name: str, *, run: str | None = None) -> tuple[np.ndarray, dict]:
        n_ff = self._check(L)
        v = np.full(n_ff, np.nan, np.float32)
        meta: dict = {"layer": L, "field": name, "has": False}
        if name in ("sweep", "sweep_null"):
            r = self._run(L, run)
            if r is not None:
                for u in r["units"]:
                    if u["kind"] == "neuron" and 0 <= u["neuron"] < n_ff:
                        x = u["mass"] if name == "sweep" else u["null_pct"]
                        v[u["neuron"]] = np.nan if x is None else x
                meta.update(has=True, run=r["run"], design=r["design"], floor=r["floor"])
        elif name == "steer_kl":
            sl = self.steer_lens(L)
            if sl is not None and sl.done.any():
                d = sl.done
                kp, kn = sl.kl_pos, sl.kl_neg
                v[d] = np.where(kp[d] >= kn[d], kp[d], -kn[d])           # 大きい方の KL、符号はその向き
                nulls = [sl.null(sg) for sg in ("pos", "neg")]
                meta.update(has=True, n_done=int(d.sum()),
                            null_p95=[float(np.percentile(x, 95)) if x.size else None for x in nulls])
        elif name == "logit_peak":
            ll = self.logit_lens(L)
            if ll is not None:
                ok = ll.neurons < n_ff
                v[ll.neurons[ok]] = ll.peak_z[ok]
                meta.update(has=True)
        elif name == "act_q999":
            a = self.act(L)
            if a is not None:
                v[:] = a.q999[:n_ff]
                meta.update(has=True)
        elif name == "named":
            for k in self.names():
                LL, n = (int(x) for x in k.split(":"))
                if LL == L and 0 <= n < n_ff:
                    v[n] = 1.0
            meta.update(has=bool(np.isfinite(v).any()))
        elif name.startswith("cos:"):
            c = self.cos(name[4:], L)
            if c is not None:
                v[:] = c[:n_ff]
                meta.update(has=True)
        else:
            raise GneError("BAD_ARGS", f"overlay {name!r} はありません", f"{', '.join(FIELDS)}, cos:<方向>")
        return v, meta

    # --- hover(§4.6): steer ±6・logit ±6 の語番号と KL± ------------------------------------------------------
    def hover(self, L: int) -> tuple[bytes, dict]:
        n_ff = self._check(L)
        ids = np.full((n_ff, 24), -1, np.int32)
        kl = np.full((n_ff, 2), np.nan, np.float32)
        meta = {"layer": L, "steer": False, "logit": False}
        sl = self.steer_lens(L)
        if sl is not None and sl.done.any():
            d = sl.done
            k = min(6, sl.up_ids.shape[1])
            ids[d, 0:k] = sl.up_ids[d, :k]
            ids[d, 6:6 + k] = sl.down_ids[d, :k]
            kl[d, 0], kl[d, 1] = sl.kl_pos[d], sl.kl_neg[d]
            meta["steer"] = True
        ll = self.logit_lens(L)
        if ll is not None:
            ok = ll.neurons < n_ff
            k = min(6, ll.up_ids.shape[1])
            ids[ll.neurons[ok], 12:12 + k] = ll.up_ids[ok, :k]
            ids[ll.neurons[ok], 18:18 + k] = ll.down_ids[ok, :k]
            meta["logit"] = True
        return ids.tobytes() + kl.tobytes(), meta

    # --- LIST の rank(§2): 層の run ごとのユニット要約 ---------------------------------------------------------
    def layer_units(self, L: int) -> dict:
        self._check(L)
        names = self.names()
        out = []
        for r in self.runs(L):
            rows = []
            for u in r["units"]:
                d8 = u.get("dose", {}).get("+0.8") or {}
                rows.append({"kind": u["kind"], "neuron": u["neuron"], "rank": u.get("rank"), "mass": u["mass"],
                             "null_pct": u["null_pct"], "z": u.get("z"), "words": [w for w, _, _ in u["words"][:5]],
                             "distinct4": d8.get("distinct4"),
                             "name": names.get(f"{L}:{u['neuron']}") if u["kind"] == "neuron" else None})
            out.append({"run": r["run"], "design": r["design"], "n_neurons": r["n_neurons"],
                        "n_controls": r["n_controls"], "floor": r["floor"], "units": rows})
        return {"layer": L, "runs": out}

    # --- NEURON パネル(§5): Examples の格子と Details --------------------------------------------------------
    def unit(self, L: int, n: int, *, kind: str = "neuron", run: str | None = None) -> dict:
        n_ff = self._check(L)
        if kind not in ("neuron", "random"):
            raise GneError("BAD_ARGS", "kind は neuron / random です")
        if kind == "neuron" and not 0 <= n < n_ff:
            raise GneError("BAD_ARGS", f"L{L} N{n} は範囲外です")
        tok = self.tok
        out: dict = {"layer": L, "neuron": n, "kind": kind, "name": None, "down_norm": None, "sweep": None,
                     "steer_lens": None, "logit_lens": None, "directions": {}, "activation": None}
        if kind == "neuron":
            out["name"] = self.names().get(f"{L}:{n}")
            out["down_norm"] = float(self.s.weights.down_norms(L)[n])

        # sweep: 指定の run、無ければこのユニットを含む run のうちニューロンの多いもの
        runs = [self._run(L, run)] if run else self.runs(L)
        for r in runs:
            if r is None:
                continue
            u = next((x for x in r["units"] if x["kind"] == kind and x["neuron"] == n), None)
            if u is None:
                continue
            recs, meta = self.records(r["run"])
            mine = [x for x in recs if x["kind"] == kind and x["neuron"] == n]
            base = {x["prompt"]: x for x in recs if x["kind"] == "baseline"}
            n_rank = r["n_neurons"] if kind == "neuron" else r["n_controls"]
            out["sweep"] = {"run": r["run"], "design": r["design"], "rank": u.get("rank"), "n_ranked": n_rank,
                            "mass": u["mass"], "null_pct": u["null_pct"], "z": u.get("z"), "floor": r["floor"],
                            "words": [[w, c, z, u["per_prompt"].get(w, [])] for w, c, z in u["words"][:20]],
                            "dose_table": [{"dose": k, **v} for k, v in sorted(u.get("dose", {}).items(),
                                                                                key=lambda kv: float(kv[0]))],
                            "examples": _examples(mine, base, meta.get("prompts", {}))}
            break

        if kind == "neuron":
            sl = self.steer_lens(L)
            if sl is not None and sl.done[n]:
                null = sl.null("pos")
                out["steer_lens"] = {"up": tok(sl.up_ids[n]), "down": tok(sl.down_ids[n]),
                                     "kl_up": float(sl.kl_pos[n]), "kl_down": float(sl.kl_neg[n]),
                                     "kl_null_p95": float(np.percentile(null, 95)) if null.size else None,
                                     "pct_up": float(sl.null_pct(sl.kl_pos[n], "pos")),
                                     "pct_down": float(sl.null_pct(sl.kl_neg[n], "neg")), "dose": sl.config["dose"]}
            ll = self.logit_lens(L)
            if ll is not None:
                hit = np.nonzero(ll.neurons == n)[0]
                if hit.size:
                    i = int(hit[0])
                    out["logit_lens"] = {"up": tok(ll.up_ids[i]), "down": tok(ll.down_ids[i]),
                                         "up_z": [float(x) for x in ll.up_z[i]], "down_z": [float(x) for x in ll.down_z[i]],
                                         "peak_z": float(ll.peak_z[i])}
            for name, _ in _directions(self.s):
                c = self.cos(name, L)
                if c is not None:
                    out["directions"][name] = {"cos": float(c[n]), "rank_in_layer": int((np.abs(c) > abs(c[n])).sum()) + 1}
            a = self.act(L)
            if a is not None:
                out["activation"] = {"absmax": float(a.absmax[n]), "q999": float(a.q999[n]), "q99": float(a.q99[n]),
                                     "rms": float(a.rms[n]), "mean": float(a.mean[n])}
        out["coverage"] = {k: bool(out[k]) for k in ("sweep", "steer_lens", "logit_lens", "directions", "activation")}
        return out

    # --- 検索(header): 当たったトークンと、その lens の中での順位を添えて、強い順に --------------------------------
    def search(self, q: str, *, mode: str = "lens", limit: int = 500) -> dict:
        qq = q.casefold().strip()
        if not qq:
            raise GneError("BAD_ARGS", "検索語が空です")
        if mode not in ("lens", "words", "name"):
            raise GneError("BAD_ARGS", "mode は lens / words / name です")
        hits: dict[tuple[int, int], dict] = {}

        def add(L, n, where, pos, token, score):
            h = hits.setdefault((L, n), {"layer": L, "neuron": n, "where": [], "best": None})
            h["where"].append(where)
            if h["best"] is None or score < h["best"]["score"]:
                h["best"] = {"where": where, "pos": pos, "token": token, "score": score}

        if mode == "lens":
            voc = self.vocab_text()
            match = np.array([i for i, t in enumerate(voc) if qq in t.casefold()], dtype=np.int64)
            for L in range(self.s.meta.n_layer):
                sl, ll = self.steer_lens(L), self.logit_lens(L)
                lists = []
                if sl is not None:
                    lists += [("steer+", sl.up_ids, None, sl.done, 0), ("steer−", sl.down_ids, None, sl.done, 100)]
                if ll is not None:
                    lists += [("logit+", ll.up_ids, ll.neurons, None, 10), ("logit−", ll.down_ids, ll.neurons, None, 110)]
                for where, ids, neurons, done, base in lists:
                    m = np.isin(ids, match)
                    if done is not None:
                        m &= done[:, None]
                    for row in np.nonzero(m.any(axis=1))[0]:
                        pos = int(np.argmax(m[row]))
                        n = int(neurons[row]) if neurons is not None else int(row)
                        add(L, n, where, pos + 1, voc[int(ids[row, pos])], base + pos)   # 押す側(+)・steer・上位ほど強い
        elif mode == "words":
            for L in range(self.s.meta.n_layer):
                for r in self.runs(L):
                    for u in r["units"]:
                        if u["kind"] != "neuron":
                            continue
                        for i, (w, _c, z) in enumerate(u["words"][:20]):
                            if qq in str(w).casefold():
                                add(L, int(u["neuron"]), f"sweep:{r['run']}", i + 1, w, -float(z))
                                break
        else:
            for k, v in self.names().items():
                if qq in v.casefold():
                    L, n = (int(x) for x in k.split(":"))
                    add(L, n, "name", None, v, 0)
        rows = sorted(hits.values(), key=lambda h: (h["best"]["score"], h["layer"], h["neuron"]))
        for h in rows:
            h["where"] = sorted(set(h["where"]))
            h["best"].pop("score")
        return {"q": q, "mode": mode, "n_hits": len(rows), "hits": rows[:limit]}

    # --- layers ビュー(§6): 層ごとの要約 ---------------------------------------------------------------------
    def profile(self) -> dict:
        m = self.s.meta
        resid = self.s.resid_stats()
        dirs = [name for name, _ in _directions(self.s)]
        dir_layers = {name: {int(x) for x in np.load(p)["layers"]} for name, p in _directions(self.s)}
        # 逆引きの閾値はマップの cos overlay の既定と同じ「全層の |cos| の 99.9% 点」(F40)
        thr = {}
        for name in dirs:
            allc = [np.abs(self.cos(name, L)) for L in sorted(dir_layers[name]) if 0 <= L < m.n_layer]
            allc = [c for c in allc if c is not None]
            thr[name] = float(np.quantile(np.concatenate(allc), 0.999, method="lower")) if allc else None
        layers = []
        for L in range(m.n_layer):
            row: dict = {"layer": L, "rho": None, "sweep": None, "steer_kl": None, "repeat": None, "directions": {},
                         "coverage": {}}
            if resid is not None and L in resid.layers:
                i = int(np.nonzero(resid.layers == L)[0][0])
                row["rho"] = [float(resid.p10[i]), float(resid.p50[i]), float(resid.p90[i])]
            r = self._run(L, None)
            if r is not None:
                neu = [u for u in r["units"] if u["kind"] == "neuron"]
                ctrl = [u for u in r["units"] if u["kind"] == "random"]
                k95 = sum((u["null_pct"] or 0) >= 0.95 for u in neu)
                row["sweep"] = {"run": r["run"], "n": len(neu), "n_null95": int(k95),
                                "frac_null95": k95 / len(neu) if neu else None}

                def d4(us):
                    x = [u.get("dose", {}).get("+0.8", {}).get("distinct4") for u in us]
                    x = [y for y in x if y is not None]
                    return float(np.median(x)) if x else None
                row["repeat"] = {"neuron": d4(neu), "random": d4(ctrl)}
            sl = self.steer_lens(L)
            if sl is not None and sl.done.any():
                kp, kn = sl.kl_pos[sl.done], sl.kl_neg[sl.done]
                row["steer_kl"] = {"pos": [float(np.median(kp)), float(np.percentile(kp, 90))],
                                   "neg": [float(np.median(kn)), float(np.percentile(kn, 90))], "n": int(sl.done.sum())}
            for name in dirs:
                if L not in dir_layers[name]:
                    continue
                c = np.abs(self.cos(name, L))
                top = np.sort(c)[-max(1, c.size // 100):]
                row["directions"][name] = {"top1_mean": float(top.mean()), "max": float(c.max()),
                                           "n_above": int((c >= thr[name]).sum()), "thr": thr[name]}
            lp, sp = _lens_paths(self.s, L)
            row["coverage"] = {"sweep": r is not None, "steer_lens": sp.exists(), "logit_lens": lp.exists(),
                               "act_stats": self.s.store.act_path(m.model_id, L).exists(),
                               "directions": sorted(row["directions"])}
            layers.append(row)
        return {"layers": layers, "directions": dirs}


def _cell(r: dict) -> dict:
    return {"dose": r["dose"], "rel_push": r.get("rel_push"), "text": r["text"], "finish": r.get("finish"),
            "n_gen": r.get("n_gen")}


def _examples(mine: list[dict], base: dict[str, dict], prompts: dict[str, str]) -> dict:
    """Examples の格子(§5.2)。列 = ベースライン + 2 つ以上のプロンプトで共有される dose(昇順)。
    そのプロンプトにしかない dose(free × +3 など)は行ごとの extra に入れる(空の列をまたいで表示)。"""
    by_dose: dict[float, set[str]] = {}
    for r in mine:
        by_dose.setdefault(float(r["dose"]), set()).add(r["prompt"])
    shared = sorted(d for d, ps in by_dose.items() if len(ps) >= 2)
    order = list(prompts) or sorted({r["prompt"] for r in mine})
    rows = []
    for p in order:
        rs = [r for r in mine if r["prompt"] == p]
        if not rs and p not in base:
            continue
        b = base.get(p)
        rows.append({"prompt": p, "text": prompts.get(p, p),
                     "baseline": {"text": b["text"], "finish": b.get("finish"), "n_gen": b.get("n_gen")} if b else None,
                     "cells": {f"{float(r['dose']):+g}": _cell(r) for r in rs if float(r["dose"]) in shared},
                     "extra": [_cell(r) for r in sorted(rs, key=lambda r: r["dose"]) if float(r["dose"]) not in shared]})
    rel = {}
    for r in mine:
        rel.setdefault(f"{float(r['dose']):+g}", r.get("rel_push"))
    return {"columns": [{"dose": f"{d:+g}", "rel_push": rel.get(f"{d:+g}")} for d in shared], "rows": rows}
