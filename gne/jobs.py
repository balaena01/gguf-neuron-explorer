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
    _t_last: float = field(default=0.0, repr=False)

    def view(self) -> dict:
        eta = None
        if self.status == "running" and self.done and self.total and self.started:
            el = time.monotonic() - self.started
            eta = el / self.done * (self.total - self.done)
        return {"id": self.id, "kind": self.kind, "params": self.params, "status": self.status, "done": self.done,
                "total": self.total, "stage": self.stage, "eta_s": eta, "yields": self.yields,
                "seconds": (self.finished or time.monotonic()) - self.started if self.started else None,
                "result": self.result, "error": self.error}


class _Cancelled(Exception):
    pass


def _sweep_rank(s, layer: int, **p) -> dict:
    """sweep のあとに rank まで回す(UI から 1 回で層の順位表まで作る)。"""
    r = s.sweep(layer, **p)
    k = s.rank(r["run"], show=0)
    return {"sweep": {k_: v for k_, v in r.items() if k_ != "units"}, "rank": {k_: v for k_, v in k.items() if k_ != "units"}}


def _eval(s, recipe: str | None = None, units: list[str] | None = None, dose_unit: str | None = None, **p) -> dict:
    """eval(§6.14)。recipe は保存したレシピの名前、units は単発指定(どちらか)。"""
    if bool(recipe) == bool(units):
        raise GneError("BAD_ARGS", "recipe(名前)か units のどちらかを渡してください")
    r = s.recipe(path=s.recipe_path(recipe)) if recipe else s.recipe(units=list(units), dose_unit=dose_unit)
    d = s.eval(r, **p)
    return {k: v for k, v in d.items() if k != "samples"}


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
        "ablate": lambda s, direction, out, **p: s.ablate(direction, out, **p),
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
        job = Job(uuid.uuid4().hex[:8], kind, dict(params or {}))
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
            if self.arbiter.yield_point():
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
