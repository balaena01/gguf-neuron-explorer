"""複数 GPU・複数プロセスのワーカー(CHANGE-03 B)。

GPU ごとに別の Python 環境・別のプロセスで `gne worker` を立て、同じ置き場所(--root)のファイルのジョブキューで仕事を分ける。
- `<root>/workers/<name>.json`: ワーカーが起動時と待っている間に書く(model_id・モデルのパス・バックエンド・VRAM・pid・
  最後に応答した時刻)。投入側はこれを見て、同じ model_id で生きているワーカーに投げる。
- `<root>/jobs/queue/<id>.json` → `running/<id>.json` → `done/<id>.json`。取るのは rename(同じファイルシステムの中で
  原子的なので、2 つのワーカーが同じジョブを取ることはない)。`cancel/<id>` があれば止める。
- 分けられるのは sweep と lens_steer(ニューロンの範囲で分け、ランダム対照も分ける)。sweep はワーカーごとの
  `<run>.part-<name>.jsonl` に書き、`gne rank` と UI が結合して読む。lens_steer は `steer_L{L}.part-<name>.npz` に書き、
  最後の部分を終えたワーカーが `steer_L{L}.npz` に結合する。
- GPU の調停はプロセスの中だけ(別の GPU なので競合しない)。1 枚の GPU に 2 つのワーカーを載せることもできる(遅くなる)。
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from pathlib import Path

import numpy as np

from .steer import GneError

SPLITTABLE = ("sweep", "lens_steer")
ALIVE_S = 30.0                       # これより古い心拍のワーカーは死んだとみなす


def _jobs_dir(root: Path) -> Path:
    d = Path(root) / "jobs"
    for sub in ("queue", "running", "done", "cancel"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def _write(path: Path, d: dict) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str))
    tmp.replace(path)


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# --- ワーカーの登録 --------------------------------------------------------------------------------------

def backend_info() -> dict:
    """バックエンドと VRAM(わかる範囲)。"""
    from .bench import gpu_name
    info = {"host": socket.gethostname(), "gpu": gpu_name()}
    try:
        from llama_cpp import llama_cpp as C
        info["gpu_offload"] = bool(C.llama_supports_gpu_offload())
        sysinfo = C.llama_print_system_info().decode("utf-8", "replace")
        info["backend"] = ("CUDA" if "CUDA" in sysinfo else "ROCm" if ("HIP" in sysinfo or "ROCM" in sysinfo.upper())
                           else "Vulkan" if "VULKAN" in sysinfo.upper() else "CPU")
    except Exception:  # noqa: BLE001
        info["backend"] = "?"
    return info


def heartbeat(root: Path, name: str, session, *, status: str = "idle", job: str | None = None,
              info: dict | None = None) -> dict:
    d = {"name": name, "model_id": session.meta.model_id, "model": str(session.path), "pid": os.getpid(),
         "status": status, "job": job, "last_seen": time.time(), **(info or {})}
    p = Path(root) / "workers" / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    _write(p, d)
    return d


def list_workers(root: Path, *, model_id: str | None = None, alive_only: bool = True) -> list[dict]:
    out = []
    for p in sorted((Path(root) / "workers").glob("*.json")):
        d = _read(p)
        if d is None:
            continue
        d["alive"] = time.time() - float(d.get("last_seen", 0)) < ALIVE_S
        if alive_only and not d["alive"]:
            continue
        if model_id is not None and d.get("model_id") != model_id:
            continue
        out.append(d)
    return out


# --- 投入 ----------------------------------------------------------------------------------------------

def _spec(xs) -> str:
    """番号の列 → "0-99,120,130-140"(連続は範囲にまとめる)。"""
    xs = sorted(int(x) for x in xs)
    out, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ",".join(out)


def split_params(session, kind: str, params: dict, names: list[str]) -> list[dict]:
    """ニューロンと対照を len(names) 個に分ける(ニューロンは連続した範囲、対照は番号で)。"""
    from .selection import parse_neurons
    L = int(params["layer"])
    default = "random:384" if kind == "sweep" else "all"
    sel = parse_neurons(params.get("neurons", default), session.meta.n_ff_at(L), seed=session.seed)
    n_ctrl = int(params.get("random_controls", 64))
    names = names[:max(1, min(len(names), int(sel.size)))]      # どの部分にもニューロンが入るように
    parts = []
    for name, nn, cc in zip(names, np.array_split(np.sort(sel), len(names)),
                            np.array_split(np.arange(n_ctrl), len(names)), strict=True):
        parts.append({**params, "neurons": _spec(nn), "controls": [int(x) for x in cc], "part": name})
    return parts


def submit(session, kind: str, params: dict, *, workers: str | list[str] = "auto") -> dict:
    """ジョブをファイルのキューに入れる。workers = "auto"(同じ model_id で生きているワーカー全部)か名前のリスト。
    sweep / lens_steer は分ける。他の種類は 1 つのジョブにして、最初のワーカーに宛てる。"""
    root = session.store.root
    alive = list_workers(root, model_id=session.meta.model_id)
    if workers == "auto":
        names = [w["name"] for w in alive]
    else:
        names = [x for x in (workers.split(",") if isinstance(workers, str) else workers) if x]
        known = {w["name"]: w for w in list_workers(root, alive_only=False)}
        bad = [n for n in names if n in known and known[n].get("model_id") != session.meta.model_id]
        if bad:
            raise GneError("WORKER_MISMATCH", f"ワーカー {', '.join(bad)} は別のモデルを載せています",
                           f"このモデル({session.meta.model_id})を載せたワーカーに投げてください")
    if not names:
        raise GneError("NO_WORKERS", f"このモデル({session.meta.model_id})を載せた生きているワーカーがありません",
                       "gne worker -m <model> --name <名前> を先に起動してください")
    group = uuid.uuid4().hex[:8]
    jd = _jobs_dir(root)
    plist = split_params(session, kind, params, names) if kind in SPLITTABLE and len(names) > 1 else [dict(params)]
    if kind in SPLITTABLE and len(names) > 1:
        targets = [p["part"] for p in plist]
    else:
        targets = [names[0]]
        if kind in SPLITTABLE and len(names) == 1:
            plist[0] = {**plist[0], "part": names[0]}
    ids = []
    for p, w in zip(plist, targets, strict=True):
        jid = f"{group}-{w}"
        _write(jd / "queue" / f"{jid}.json", {"id": jid, "group": group, "kind": kind, "params": p, "worker": w,
                                               "model_id": session.meta.model_id, "n_parts": len(plist),
                                               "status": "queued", "created": time.time()})
        ids.append(jid)
    return {"group": group, "jobs": ids, "workers": targets, "kind": kind}


def job_files(root: Path, group: str | None = None) -> list[dict]:
    jd = _jobs_dir(root)
    out = []
    for sub in ("queue", "running", "done"):
        for p in sorted((jd / sub).glob("*.json")):
            d = _read(p)
            if d and (group is None or d.get("group") == group):
                out.append(d)
    return out


def wait(root: Path, group: str, *, poll: float = 2.0, timeout: float | None = None, progress=None) -> list[dict]:
    """group の全ジョブが終わるまで待つ。progress(done, total, msg) に部分ごとの進捗の合計を渡す。"""
    t0 = time.monotonic()
    while True:
        js = job_files(root, group)
        if progress and js:
            progress(sum(int(j.get("done") or 0) for j in js), sum(int(j.get("total") or 0) for j in js) or 1,
                     " / ".join(f"{j['worker']}:{j['status']}" for j in js))
        if js and all(j["status"] in ("done", "error", "cancelled") for j in js):
            return js
        if timeout is not None and time.monotonic() - t0 > timeout:
            raise GneError("TIMEOUT", f"ジョブ {group} が {timeout:.0f} 秒で終わりませんでした")
        time.sleep(poll)


def cancel(root: Path, job_id: str) -> None:
    (_jobs_dir(root) / "cancel" / job_id).touch()


# --- ワーカーの本体 ----------------------------------------------------------------------------------------

class _Cancelled(Exception):
    pass


def _claim(root: Path, name: str, model_id: str) -> dict | None:
    jd = _jobs_dir(root)
    for p in sorted((jd / "queue").glob("*.json"), key=lambda q: q.stat().st_mtime):
        d = _read(p)
        if d is None or d.get("model_id") != model_id or d.get("worker") not in (None, name):
            continue
        dst = jd / "running" / p.name
        try:
            os.rename(p, dst)                       # 先に rename できたワーカーが取る
        except FileNotFoundError:
            continue
        d.update(status="running", claimed_by=name, started=time.time())
        _write(dst, d)
        return d
    return None


def _merge_if_last(session, d: dict) -> dict | None:
    """lens_steer の部分を全部終えていれば steer_L{L}.npz に結合する(最後に終えたワーカーが行う)。"""
    if d["kind"] != "lens_steer" or d.get("n_parts", 1) <= 1:
        return None
    root = session.store.root
    js = job_files(root, d["group"])
    if not js or any(j["status"] != "done" for j in js if j["id"] != d["id"]):
        return None
    lock = _jobs_dir(root) / "done" / f"{d['group']}.merge.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return None
    from .lens import merge_parts
    L = int(d["params"]["layer"])
    ld = session.store.lens_dir(session.meta.model_id)
    parts = [ld / f"steer_L{L}.part-{j['params']['part']}.npz" for j in js]
    out = ld / f"steer_L{L}.npz"
    res = merge_parts([p for p in parts if p.exists()], out)
    for p in parts:
        p.unlink(missing_ok=True)
    return {"merged": str(out), "n_done": int(res.done.sum()), "n_controls": int(res.ctrl_done.sum())}


def run_job(session, d: dict, *, name: str) -> dict:
    """1 つのジョブを回す(standard_kinds の関数)。進捗はジョブのファイルに書き、cancel の印を見て止める。"""
    from .jobs import standard_kinds
    root = session.store.root
    jd = _jobs_dir(root)
    path = jd / "running" / f"{d['id']}.json"
    last = [0.0]

    def progress(done, total, msg=""):
        if (jd / "cancel" / d["id"]).exists():
            raise _Cancelled
        if time.monotonic() - last[0] > 1.0:
            last[0] = time.monotonic()
            d.update(done=int(done), total=int(total), stage=str(msg))
            _write(path, d)
            heartbeat(root, name, session, status="busy", job=d["id"])
    prev = session.progress
    session.progress = progress
    try:
        fn = standard_kinds()[d["kind"]]
        res = fn(session, **d["params"])
        d.update(status="done", result=json.loads(json.dumps(res, default=str)))
    except _Cancelled:
        d.update(status="cancelled")
    except GneError as e:
        d.update(status="error", error={"code": e.code, "message": e.message, "hint": e.hint})
    except Exception as e:  # noqa: BLE001
        d.update(status="error", error={"code": "INTERNAL", "message": f"{type(e).__name__}: {e}", "hint": None})
    finally:
        session.progress = prev
    d["finished"] = time.time()
    _write(jd / "done" / f"{d['id']}.json", d)
    path.unlink(missing_ok=True)
    if d["status"] == "done":
        try:
            m = _merge_if_last(session, d)
            if m:
                d["merge"] = m
                _write(jd / "done" / f"{d['id']}.json", d)
        except Exception as e:  # noqa: BLE001
            d["merge_error"] = f"{type(e).__name__}: {e}"
            _write(jd / "done" / f"{d['id']}.json", d)
    return d


def run_worker(session, name: str, *, poll: float = 2.0, once: bool = False, max_jobs: int | None = None,
               log=None) -> int:
    """ジョブを取って回し続ける。once なら、キューが空になったら戻る。返り値は回したジョブの数。"""
    root = session.store.root
    info = backend_info()
    n = 0
    while True:
        heartbeat(root, name, session, info=info)
        d = _claim(root, name, session.meta.model_id)
        if d is None:
            if once:
                return n
            time.sleep(poll)
            continue
        if log:
            log(f"{name}: {d['kind']} {d['id']} を開始")
        heartbeat(root, name, session, status="busy", job=d["id"], info=info)
        d = run_job(session, d, name=name)
        n += 1
        if log:
            log(f"{name}: {d['kind']} {d['id']} → {d['status']}" + (f"(結合 {d['merge']['merged']})" if d.get("merge") else ""))
        if max_jobs is not None and n >= max_jobs:
            heartbeat(root, name, session, info=info)
            return n
