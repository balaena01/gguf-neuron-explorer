"""UI のライブ機能(別冊 U2): 3 列チャットの逐次生成、fires、reads you、レシピ、cvec の書き出し、ジョブ。

GPU を使う処理は GpuArbiter を通す。ジョブ(sweep・lens など)はユニットの切れ目で GPU を譲るので、
チャットや fires はジョブの実行中でも数秒以内に始まる(§8.1)。
"""

from __future__ import annotations

import codecs
import json
import queue
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..jobs import GpuArbiter, JobRunner, standard_kinds
from ..steer import GneError

FIRES_KEEP = 4             # 覚えておく fires の文章の数
FIRES_TOKEN_KEEP = 8       # 文章ごとに覚えておくトークンの全層フィールドの数


class _Stopped(Exception):
    pass


class Live:
    def __init__(self, session, data):
        self.s = session
        self.data = data
        self.arbiter = GpuArbiter()
        self.jobs = JobRunner(session, self.arbiter, standard_kinds())
        self._fires: OrderedDict[str, dict] = OrderedDict()
        self._streams: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    # --- 共通 -------------------------------------------------------------------------------------------
    def _gpu(self, fn):
        """対話的な GPU の処理。ジョブが走っていれば、その切れ目まで待ってから先に通る。警告も集める。"""
        with self.arbiter.interactive():
            self.s.warnings = []
            out = fn()
            return out, list(self.s.warnings)

    def recipe(self, units: list[str], name: str | None = None, ablate: str | dict | None = None):
        if not units and not ablate:
            return None
        return self.s.recipe(units=list(units) or None, dose_unit="rel", name=name, ablate=ablate)

    # --- 3 列チャット(§5.3)。SSE の行を順に返す ---------------------------------------------------------
    def steer_stream(self, body: dict):
        """body: {columns: [{label, units: ["14:8949:+1"], ablate?, messages: [...], when?}], max_tokens, system, when}。
        ablate は {direction, scale, layers, from_layer} か "NAME@層:倍率"(§6.17)。
        sampling: {seed, temperature?, top_k?, top_p?, min_p?, presence_penalty?} ならサンプリング(既定は start.sh と同じ値)。
        どの列も同じ seed で引くので、列の違いは介入の違いだけになる。
        列ごとに会話の履歴(最後がユーザーの発話)を受け取り、1 列ずつ逐次生成する。状態はクライアントが持つ。
        when は足す位置(§6.15)。列ごとに指定が無ければ全体の when、それも無ければ all。"""
        cols = body.get("columns") or []
        if not cols or len(cols) > 6:
            raise GneError("BAD_ARGS", "columns は 1〜6 列です")
        max_tokens = int(body.get("max_tokens", 128))
        if not 1 <= max_tokens <= 2048:
            raise GneError("BAD_ARGS", "max_tokens は 1〜2048 です")
        recipes = [self.recipe(c.get("units") or [], ablate=c.get("ablate")) for c in cols]   # 形式の誤りは始める前に返す
        from ..when import generate_when, parse_when
        whens = [parse_when(c.get("when") or body.get("when") or "all") for c in cols]
        sampling = None
        if body.get("sampling"):
            from ..engine import Sampling
            sp = dict(body["sampling"])
            allowed = {"seed", "temperature", "top_k", "top_p", "min_p", "presence_penalty"}
            if set(sp) - allowed:
                raise GneError("BAD_ARGS", f"sampling に使えるのは {', '.join(sorted(allowed))} です")
            sampling = Sampling(**{k: (int(v) if k in ("seed", "top_k") else float(v)) for k, v in sp.items()})
        system = body.get("system")
        sid = uuid.uuid4().hex[:8]
        stop = threading.Event()
        self._streams[sid] = stop
        q: queue.Queue = queue.Queue()

        def work():
            t0 = time.perf_counter()
            try:
                with self.arbiter.interactive():
                    q.put({"type": "gpu", "waited_s": time.perf_counter() - t0})
                    self.s.warnings = []
                    e = self.s.engine
                    ctx = e.steer_ctx()
                    for i, (c, r, w) in enumerate(zip(cols, recipes, whens, strict=True)):
                        if stop.is_set():
                            break
                        msgs = ([{"role": "system", "content": system}] if system else []) + list(c.get("messages") or [])
                        ids = e.tokenize_prompt(e.render_chat(msgs))
                        if len(ids) + max_tokens > e.n_ctx:
                            raise GneError("CONTEXT_FULL", f"会話が長すぎます({len(ids)} トークン、n_ctx {e.n_ctx})",
                                           "会話をやり直すか、max_tokens を減らしてください")
                        vecs = self.s.vectors(r)[0] if r is not None else None
                        q.put({"type": "column", "col": i, "n_prompt": len(ids)})
                        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

                        def cb(_si, tok, _i=i, _dec=dec):
                            if stop.is_set():
                                raise _Stopped()
                            piece = _dec.decode(e.token_bytes(tok))
                            if piece:
                                q.put({"type": "text", "col": _i, "text": piece})
                        try:
                            out, winfo = generate_when(e, ctx, msgs, vecs, w, max_tokens=max_tokens, on_token=cb,
                                                       lora=self.s.lora_for(r), sampling=sampling)
                            q.put({"type": "end", "col": i, "text": out.text, "finish": out.finish, "n_gen": out.n_gen,
                                   "when": winfo})
                        except _Stopped:
                            q.put({"type": "end", "col": i, "text": None, "finish": "stopped", "n_gen": None})
                            break
                    if self.s.warnings:
                        q.put({"type": "warn", "warnings": list(self.s.warnings)})
            except GneError as ex:
                q.put({"type": "error", "error": {"code": ex.code, "message": ex.message, "hint": ex.hint}})
            except Exception as ex:  # noqa: BLE001
                q.put({"type": "error", "error": {"code": "INTERNAL", "message": f"{type(ex).__name__}: {ex}"}})
            finally:
                q.put({"type": "done", "seconds": time.perf_counter() - t0})

        def events():
            yield _sse({"type": "start", "id": sid, "columns": len(cols)})
            th = threading.Thread(target=work, name=f"gne-steer-{sid}", daemon=True)
            th.start()
            try:
                while True:
                    ev = q.get()
                    yield _sse(ev)
                    if ev["type"] == "done":
                        break
            finally:
                stop.set()                        # 接続が切れたら生成も止める
                self._streams.pop(sid, None)
        return events()

    def stop(self, sid: str) -> dict:
        ev = self._streams.get(sid)
        if ev is not None:
            ev.set()
        return {"id": sid, "stopped": ev is not None}

    # --- fires(§5.5) ---------------------------------------------------------------------------------
    def fires(self, text: str, *, top: int = 50) -> dict:
        from ..fires import fires
        if not text or not text.strip():
            raise GneError("BAD_ARGS", "文章が空です")
        r, warns = self._gpu(lambda: fires(self.s, text, top=top, keep_max=True))
        fid = uuid.uuid4().hex[:8]
        with self._lock:
            self._fires[fid] = {"ids": r.pop("ids"), "max": r.pop("max_field"), "tokens": r["tokens"], "top": r["top"],
                                "layers": r["layers"], "norm": r["normalization"], "by_token": OrderedDict()}
            while len(self._fires) > FIRES_KEEP:
                self._fires.popitem(last=False)
        return {"id": fid, **r, "warnings": warns}

    def _fire(self, fid: str) -> dict:
        f = self._fires.get(fid)
        if f is None:
            raise GneError("NOT_FOUND", f"fires {fid} はもうありません", "文章をもう一度流してください")
        return f

    def fires_fields(self, fid: str, tok: int) -> tuple[bytes, list[int], dict]:
        """全層のスコア(層ごとの n_ff を順に並べた Float32Array)。tok = -1 なら全トークンでの |最大|、
        それ以外は position tok(1 始まり)の値。トークンの値は覚えていなければ prefill し直す(1 秒未満)。"""
        from ..fires import fire_scores
        f = self._fire(fid)
        if tok == -1:
            fields = f["max"]
        elif 1 <= tok < len(f["ids"]):
            fields = f["by_token"].get(tok)
            if fields is None:
                (kept, scores, _, _), _ = self._gpu(lambda: fire_scores(self.s, f["ids"], f["layers"]))
                fields = {L: s[tok - 1].astype(np.float32) for L, s in zip(kept, scores, strict=True)}
                with self._lock:
                    f["by_token"][tok] = fields
                    while len(f["by_token"]) > FIRES_TOKEN_KEEP:
                        f["by_token"].popitem(last=False)
        else:
            raise GneError("BAD_ARGS", f"tok は -1 か 1〜{len(f['ids']) - 1} です")
        m = self.s.meta
        sizes = [m.n_ff_at(L) for L in range(m.n_layer)]
        parts = [fields[L] if L in fields else np.full(n, np.nan, np.float32) for L, n in enumerate(sizes)]
        mx = float(max((np.nanmax(np.abs(p)) for p in parts if np.isfinite(p).any()), default=0.0))
        weak = sorted(int(L) for L, src in f["norm"].items() if src == "text_p99")   # 活性統計が無く文章内 p99 で割った層
        return b"".join(p.tobytes() for p in parts), sizes, {"tok": tok, "max": mx, "layers": sorted(fields), "text_p99": weak}

    def fires_neurons(self, fid: str, tok: int, *, lmin: int = 0, lmax: int | None = None, sign: str = "both",
                      thr: float = 0.0, q: str = "", limit: int = 100) -> dict:
        """そのトークン(-1 は全トークンの最大)で最も光るニューロン。層の範囲・符号・閾値・lens の語で絞る。"""
        body, sizes, meta = self.fires_fields(fid, tok)
        v = np.frombuffer(body, np.float32)
        offs = np.cumsum([0] + sizes)
        lmax = len(sizes) - 1 if lmax is None else lmax
        qq = q.casefold().strip()
        match = None
        if qq:
            voc = self.data.vocab_text()
            match = np.array([i for i, t in enumerate(voc) if qq in t.casefold()], dtype=np.int64)
        rows = []
        for L in range(max(0, lmin), min(len(sizes) - 1, lmax) + 1):
            x = v[offs[L]:offs[L + 1]]
            ok = np.isfinite(x) & (np.abs(x) >= thr)
            if sign == "+":
                ok &= x > 0
            elif sign == "-":
                ok &= x < 0
            if match is not None:
                ok &= self._lens_has(L, match, len(x))
            for n in np.nonzero(ok)[0]:
                rows.append((abs(float(x[n])), L, int(n), float(x[n])))
        rows.sort(reverse=True)
        out = []
        norm = self._fire(fid)["norm"]
        for _, L, n, sc in rows[:limit]:
            out.append({"layer": L, "neuron": n, "score": sc, "lens": self._lens_pole(L, n, sc >= 0),
                        "norm": norm.get(str(L))})
        return {"tok": tok, "n_hits": len(rows), "neurons": out, "text_p99": meta["text_p99"]}

    def _lens_pole(self, L: int, n: int, positive: bool) -> dict | None:
        """光った向きに対応する lens の極(正に光ったら + 極、負なら − 極)。steer lens があればそれ、無ければ logit lens。"""
        sl = self.data.steer_lens(L)
        if sl is not None and sl.done[n]:
            return {"src": "steer", "pole": "+" if positive else "−", "tokens": self.data.tok((sl.up_ids if positive else sl.down_ids)[n][:4])}
        ll = self.data.logit_lens(L)
        hit = np.nonzero(ll.neurons == n)[0] if ll is not None else []
        if len(hit):
            ids = (ll.up_ids if positive else ll.down_ids)[int(hit[0])]
            return {"src": "logit", "pole": "+" if positive else "−", "tokens": self.data.tok(ids[:4])}
        return None

    def _lens_has(self, L: int, match: np.ndarray, n_ff: int) -> np.ndarray:
        ok = np.zeros(n_ff, bool)
        sl = self.data.steer_lens(L)
        if sl is not None:
            ok |= (np.isin(sl.up_ids, match) | np.isin(sl.down_ids, match)).any(axis=1) & sl.done
        ll = self.data.logit_lens(L)
        if ll is not None:
            hit = (np.isin(ll.up_ids, match) | np.isin(ll.down_ids, match)).any(axis=1)
            ok[ll.neurons[hit]] = True
        return ok

    # --- reads you(1 ニューロンの活性をトークンごとに) -------------------------------------------------------
    def reads(self, L: int, n: int, text: str) -> dict:
        from ..fires import fire_ids, fire_scores
        self.data._check(L)
        if not 0 <= n < self.s.meta.n_ff_at(L):
            raise GneError("BAD_ARGS", f"L{L} N{n} は範囲外です")

        def fn():
            ids = fire_ids(self.s, text)
            kept, scores, raw, norm = fire_scores(self.s, ids, [L])
            if L not in kept:
                raise GneError("ARCH_UNSUPPORTED", f"層 {L} の活性を捕捉できませんでした")
            toks = [self.s.engine.token_text(t, special=False) for t in ids]
            return {"first": toks[0], "tokens": toks[1:], "score": [float(x) for x in scores[0][:, n]],
                    "act": [float(x) for x in raw[0][:, n]], "normalization": norm[str(L)]}
        r, warns = self._gpu(fn)
        return {"layer": L, "neuron": n, **r, "skipped_position0": True, "warnings": warns}

    # --- レシピと書き出し --------------------------------------------------------------------------------
    def recipes(self) -> list[dict]:
        from ..steer import Recipe
        out = []
        for p in sorted(self.s.store.recipes_dir().glob("*.yaml")):
            try:
                r = Recipe.load(p)
            except GneError:
                continue
            out.append({"name": p.stem, "model_id": r.model_id, "dose_unit": r.dose_unit,
                        "units": [{"layer": u.layer, "neuron": u.neuron, "dose": u.dose, "kind": u.kind, "mode": u.mode,
                                   "k": u.k, "direction": u.direction} for u in r.units],
                        "ablate": r.ablate.to_dict() if r.ablate else None, "when": r.when,
                        "this_model": r.model_id in (None, self.s.meta.model_id)})
        return out

    def export(self, units: list[str], name: str, *, verify: bool = False) -> dict:
        import re
        if not re.fullmatch(r"[\w.\-]+", name or ""):
            raise GneError("BAD_ARGS", f"名前 {name!r} は英数字・_・-・. だけにしてください")
        path = self.s.store.root / "export" / f"{name}.cvec.gguf"
        recipe = self.s.recipe(units=list(units), dose_unit="rel", name=name)        # GGUF の中のレシピにも名前を入れる
        r, warns = self._gpu(lambda: self.s.export_cvec(recipe, path, verify=verify))
        saved = (self.s.store.recipes_dir() / f"{name}.yaml").exists()
        return {**r, "file": f"{name}.cvec.gguf", "url": f"/api/export/file/{name}.cvec.gguf", "warnings": warns,
                "recipe_saved": saved}

    def export_path(self, file: str) -> Path:
        p = (self.s.store.root / "export" / file).resolve()
        if p.parent != (self.s.store.root / "export").resolve() or not p.name.endswith(".cvec.gguf") or not p.exists():
            raise GneError("NOT_FOUND", f"{file} はありません")
        return p

    # --- ジョブ(§8.3) ----------------------------------------------------------------------------------
    def job_events(self, job_id: str):
        job = self.jobs.get(job_id)

        def events():
            last = None
            while True:
                v = job.view()
                key = (v["status"], v["done"], v["total"], v["stage"])
                if key != last:
                    yield _sse(v)
                    last = key
                if v["status"] in ("done", "error", "cancelled"):
                    break
                time.sleep(0.5)
        return events()


def _sse(d: dict) -> str:
    return f"data: {json.dumps(d, ensure_ascii=False)}\n\n"
