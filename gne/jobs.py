"""長時間の処理(sweep / lens / stats など)をバックグラウンドで回すジョブと、GPU の調停。
設計書 §6.13 (d) の job_start / job_status、別冊 §8.1。

- GPU を使う処理は GpuArbiter を通す。対話的な呼び出し(steer・neuron など)が待っていれば、ジョブは
  ユニットの切れ目(進捗の報告のたび)で GPU を譲り、終わったら続きから進む。
- 1 つの Engine を共有するので、GPU を握っていない間は Engine に触らない。
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field

from .steer import GneError


class GpuArbiter:
    """対話的な要求を優先する排他。ジョブは yield_point() で譲る。"""

    def __init__(self):
        self._cv = threading.Condition()
        self._owner: int | None = None
        self._waiting = 0                     # 待っている対話的な要求の数

    @contextmanager
    def interactive(self):
        me = threading.get_ident()
        with self._cv:
            self._waiting += 1
            while self._owner is not None and self._owner != me:
                self._cv.wait()
            self._waiting -= 1
            nested = self._owner == me
            self._owner = me
        try:
            yield
        finally:
            if not nested:
                with self._cv:
                    self._owner = None
                    self._cv.notify_all()

    @contextmanager
    def job(self):
        me = threading.get_ident()
        with self._cv:
            while self._owner is not None or self._waiting:
                self._cv.wait()
            self._owner = me
        try:
            yield
        finally:
            with self._cv:
                self._owner = None
                self._cv.notify_all()

    def yield_point(self) -> bool:
        """ジョブのスレッドから呼ぶ。待っている対話的な要求があれば GPU を譲り、終わるまで待つ。譲ったら True。"""
        me = threading.get_ident()
        with self._cv:
            if self._owner != me or not self._waiting:
                return False
            self._owner = None
            self._cv.notify_all()
            while self._owner is not None or self._waiting:
                self._cv.wait()
            self._owner = me
            return True


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    status: str = "queued"                   # queued | running | done | error | cancelled
    done: int = 0
    total: int = 0
    stage: str = ""
    started: float | None = None
    finished: float | None = None
    result: dict | None = None
    error: dict | None = None
    yields: int = 0
    cancel: bool = False
    estimate: dict | None = None             # CHANGE-03 C: 投げたときの見積もり
    _t_last: float = field(default=0.0, repr=False)

    def view(self) -> dict:
        eta = None
        if self.status == "running" and self.done and self.total and self.started:
            el = time.monotonic() - self.started
            eta = el / self.done * (self.total - self.done)
        return {"id": self.id, "kind": self.kind, "params": self.params, "status": self.status, "done": self.done,
                "total": self.total, "stage": self.stage, "eta_s": eta, "yields": self.yields,
                "seconds": (self.finished or time.monotonic()) - self.started if self.started else None,
                "result": self.result, "error": self.error,
                "estimate_s": None if self.estimate is None else self.estimate.get("seconds")}


class _Cancelled(Exception):
    pass


def _sweep_rank(s, layer: int, **p) -> dict:
    """sweep のあとに rank まで回す(UI から 1 回で層の順位表まで作る)。"""
    r = s.sweep(layer, **p)
    k = s.rank(r["run"], show=0)
    return {"sweep": {k_: v for k_, v in r.items() if k_ != "units"}, "rank": {k_: v for k_, v in k.items() if k_ != "units"}}


def _eval(s, recipe: str | None = None, units: list[str] | None = None, dose_unit: str | None = None,
          compare_methods: bool = False, match_effect: float | None = None, score_layer: int | None = None,
          alloc: str | None = None, n_heads: list[int] | None = None, **p) -> dict:
    """eval(§6.14)。recipe は保存したレシピの名前、units は単発指定(どちらか)。compare_methods は方式の比較(CHANGE-06 F.3)、
    alloc は heads allocate の組のヘッド数ごとの評価(F.4、recipe も units も要らない)。"""
    if alloc:
        return s.eval_alloc_curve(alloc, n_heads=n_heads, suite=p.get("suite"), when=p.get("when"),
                                  random_controls=int(p.get("random_controls") or 0))
    if bool(recipe) == bool(units):
        raise GneError("BAD_ARGS", "recipe(名前)か units のどちらかを渡してください")
    r = s.recipe(path=s.recipe_path(recipe)) if recipe else s.recipe(units=list(units), dose_unit=dose_unit)
    if compare_methods:
        return s.eval_compare_methods(r, match_effect=match_effect, score_layer=score_layer, suite=p.get("suite"),
                                      when=p.get("when"))
    d = s.eval(r, **p)
    return {k: v for k, v in d.items() if k != "samples"}


def _headproj_export(s, recipe: str | None = None, out: str | None = None, *, units: list[str] | None = None,
                     name: str | None = None, direction_layer: str | None = None, file_stem: str | None = None,
                     overwrite: bool = True, **p) -> dict:
    """ヘッドの proj と gain(LoRA に入るユニット)を LoRA アダプタか焼き込み GGUF に書き出す。recipe(保存したレシピの名前か
    パス)か units(UI のレシピのユニットの書き方)。out を省くと <root>/export/<名前>.lora.gguf(焼き込みは <名前>-headproj.gguf)。
    file_stem を渡すとファイル名はそれ(LoRA は <file_stem>.lora.gguf、焼き込みは <file_stem>.gguf)。overwrite=False なら、
    同じ名前のファイルがあれば書かずに FILE_EXISTS を返す(UI は確かめてから上書きする)。LoRA は小さいので、UI から落とせる
    URL も返す。"""
    import re
    from pathlib import Path
    from .steer import GneError
    if units:
        r = s.recipe(units=list(units), name=name or None, direction_layer=direction_layer)
    elif recipe:
        r = s.recipe(path=s.recipe_path(recipe), direction_layer=direction_layer)
    else:
        raise GneError("BAD_ARGS", "recipe か units が要ります")
    nm = name or r.name or "headproj"
    if not re.fullmatch(r"[\w.\-]+", nm):
        raise GneError("BAD_ARGS", f"名前 {nm!r} は英数字・_・-・. だけにしてください")
    bake = bool(p.get("bake"))
    if file_stem is not None:
        file_stem = file_stem.strip().removesuffix(".gguf").removesuffix(".lora")
        if not re.fullmatch(r"[\w.\-]+", file_stem):
            raise GneError("BAD_ARGS", f"ファイル名 {file_stem!r} は英数字・_・-・. だけにしてください")
    stem = file_stem or (f"{nm}-headproj" if bake else nm)
    path = Path(out) if out else s.store.export_dir() / (f"{stem}.gguf" if bake else f"{stem}.lora.gguf")
    if not overwrite and path.exists():
        raise GneError("FILE_EXISTS", f"{path.name} はもうあります", "上書きするか、別の名前にしてください")
    d = s.headproj_export(r, path, **p)
    d["file"] = Path(d["path"]).name
    if not bake and Path(d["path"]).parent.resolve() == s.store.export_dir().resolve():
        d["url"] = f"/api/export/file/{d['file']}"
    d["name"] = nm
    return d


def standard_kinds() -> dict[str, Callable[..., dict]]:
    """MCP(job_start)と Web UI が共通で使うジョブの種類。"""
    return {
        "stats_resid": lambda s, **p: s.stats_resid(**p),
        "stats_act": lambda s, **p: s.stats_act(**p),
        "lens_logit": lambda s, layer, **p: s.lens_logit(layer, **p),
        "lens_steer": lambda s, layer, **p: s.lens_steer(layer, **p),
        "sweep": lambda s, layer, **p: s.sweep(layer, **p),
        "sweep_rank": _sweep_rank,
        "rank": lambda s, run, **p: s.rank(run, **p),
        "reverse": lambda s, **p: s.reverse(**p),
        "connections": lambda s, **p: s.connections(**p),
        "layout": lambda s, **p: s.layout(**p),
        "embed": lambda s, run, **p: s.embed(run, **p),
        "eval": _eval,
        "direction": lambda s, **p: s.direction(**p),
        "heads_contrast": lambda s, **p: s.heads_contrast(**p),
        "writers": lambda s, **p: s.writers(**p),
        "experts_contrast": lambda s, **p: s.experts_contrast(**p),
        "experts_sweep": lambda s, layer, **p: s.experts_sweep(layer, **p),
        "ablate": lambda s, direction, out, **p: s.ablate(direction, out, **p),
        "heads_causal": lambda s, direction, **p: s.heads_causal(direction, **p),        # CHANGE-06 C
        "heads_allocate": lambda s, direction, **p: s.heads_allocate(direction, **p),    # CHANGE-06 D
        "head_pipeline": lambda s, spec, **p: s.heads_pipeline(spec, **p),             # 方向 → 基準層 → 因果 → 配分
        "heads_candidates": lambda s, alloc, **p: s.heads_candidates(alloc, **p),       # 組 × 倍率で生成 → 採点のファイル
        "heads_refine": lambda s, name, **p: s.heads_refine(name, **p),                 # 残りの質問で追加のヘッドを探す
        "fieldtest": lambda s, **p: s.fieldtest(**p),                                   # 本番と同じ動かし方で、ファイルどうしを比べる
        "headproj_export": _headproj_export,
    }


class JobRunner:
    """ジョブを 1 本ずつ順に回すスレッド。kinds: 種類 → fn(session, **params) -> dict。"""

    def __init__(self, session, arbiter: GpuArbiter, kinds: dict[str, Callable[..., dict]]):
        self.session, self.arbiter, self.kinds = session, arbiter, kinds
        self.jobs: dict[str, Job] = {}
        self._queue: list[str] = []
        self._cv = threading.Condition()
        self._thread = threading.Thread(target=self._loop, name="gne-jobs", daemon=True)
        self._thread.start()

    def start(self, kind: str, params: dict | None = None) -> Job:
        if kind not in self.kinds:
            raise GneError("BAD_ARGS", f"ジョブの種類 {kind!r} はありません({', '.join(self.kinds)})")
        from .estimate import NeedsConfirm, gate
        try:
            params, est = gate(self.session, kind, dict(params or {}))     # CHANGE-03 C: 予算と 30 分の確認
        except NeedsConfirm as e:
            e.error.estimate = e.estimate
            raise e.error from e
        job = Job(uuid.uuid4().hex[:8], kind, dict(params or {}))
        job.estimate = est
        with self._cv:
            self.jobs[job.id] = job
            self._queue.append(job.id)
            self._cv.notify_all()
        return job

    def list(self) -> list[Job]:
        return list(self.jobs.values())

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise GneError("BAD_ARGS", f"ジョブ {job_id} はありません")
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        job.cancel = True
        if job.status == "queued":
            job.status = "cancelled"
        return job

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._queue:
                    self._cv.wait()
                job = self.jobs[self._queue.pop(0)]
            if job.status == "cancelled":
                continue
            self._run(job)

    def _run(self, job: Job) -> None:
        me = threading.get_ident()
        s = self.session

        def progress(done: int, total: int, stage: str) -> None:
            if threading.get_ident() != me:          # 譲っている間の対話的な処理からの進捗は無視
                return
            job.done, job.total, job.stage = done, total, stage
            if job.cancel:
                raise _Cancelled()
            if not getattr(s, "no_yield", False) and self.arbiter.yield_point():   # 実地テストは llama-server に GPU を渡している間は譲らない
                job.yields += 1

        job.status, job.started = "running", time.monotonic()
        try:
            with self.arbiter.job():
                prev = s.progress
                s.progress = progress
                try:
                    job.result = self.kinds[job.kind](s, **job.params)
                finally:
                    s.progress = prev
            job.status = "done"
        except _Cancelled:
            job.status = "cancelled"
        except GneError as e:
            job.status, job.error = "error", {"code": e.code, "message": e.message, "hint": e.hint}
        except Exception as e:  # noqa: BLE001
            job.status, job.error = "error", {"code": "INTERNAL", "message": f"{type(e).__name__}: {e}",
                                              "hint": traceback.format_exc(limit=3)}
        finally:
            job.finished = time.monotonic()
