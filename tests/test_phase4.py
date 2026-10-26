"""Phase 4 の単体テスト(モデル不要): 3 列の組み立て、GPU の調停、ジョブ、調書の rollout の選び方、MCP の補助。"""

import threading
import time

import pytest

from gne.chat import ChatState, make_columns
from gne.dossier import _pick_rollouts
from gne.jobs import GpuArbiter, JobRunner
from gne.mcp_server import _contrast_path, _strip_ids
from gne.steer import GneError, Recipe, Unit


def test_make_columns_default_and_dose():
    r = Recipe([Unit(14, 8949, 1.0), Unit(13, 7956, -1.0)], name=None)
    cols = make_columns(r)
    assert [c.recipe is None for c in cols] == [False, True, False]
    assert [u.dose for u in cols[0].recipe.units] == [-1.0, 1.0]           # − の列は符号を反転
    assert [u.dose for u in cols[2].recipe.units] == [1.0, -1.0]
    cols = make_columns(r, dose=0.5)
    assert [u.dose for u in cols[2].recipe.units] == [0.5, -0.5] and [u.dose for u in cols[0].recipe.units] == [-0.5, 0.5]
    other = Recipe([Unit(3, 1, 0.8)], name="x")
    cols = make_columns(None, left=other, right=other)
    assert cols[0].label == "x" and cols[1].recipe is None
    with pytest.raises(GneError):
        make_columns(None, left=other)


def test_chat_state_histories_are_per_column():
    st = ChatState(make_columns(Recipe([Unit(1, 1, 1.0)])), system="sys")
    st.histories[0].append({"role": "user", "content": "a"})
    assert st.messages(0)[0] == {"role": "system", "content": "sys"} and len(st.messages(1)) == 1
    st.reset()
    assert st.histories == [[], [], []]


def test_arbiter_interactive_cuts_in_at_yield_point():
    arb = GpuArbiter()
    order = []
    in_job = threading.Event()
    go_interactive = threading.Event()

    def job():
        with arb.job():
            in_job.set()
            for i in range(5):
                go_interactive.wait(2)
                time.sleep(0.05)
                order.append(f"job{i}")
                arb.yield_point()

    t = threading.Thread(target=job)
    t.start()
    in_job.wait(2)
    go_interactive.set()

    def interactive():
        with arb.interactive():
            order.append("inter")
    ti = threading.Thread(target=interactive)
    ti.start()
    ti.join(5)
    t.join(5)
    assert "inter" in order and order.index("inter") < len(order) - 1      # ジョブの途中で割り込めた
    assert order[-1] == "job4"


def test_arbiter_nested_interactive_does_not_deadlock():
    arb = GpuArbiter()
    with arb.interactive():
        with arb.interactive():
            pass
    with arb.job():
        assert arb.yield_point() is False                                   # 待ちが無ければ譲らない


class _FakeSession:
    def __init__(self):
        self.progress = None


def _wait(job, timeout=5):
    t0 = time.monotonic()
    while job.status in ("queued", "running") and time.monotonic() - t0 < timeout:
        time.sleep(0.02)
    return job


def test_job_runner_progress_cancel_and_error():
    arb = GpuArbiter()

    def counting(s, n=5, delay=0.01):
        for i in range(n):
            time.sleep(delay)
            s.progress(i + 1, n, "count")
        return {"n": n}

    def failing(s):
        raise GneError("BAD_ARGS", "boom", "hint")

    jr = JobRunner(_FakeSession(), arb, {"count": counting, "fail": failing})
    j = _wait(jr.start("count", {"n": 4}))
    assert j.status == "done" and j.result == {"n": 4} and (j.done, j.total) == (4, 4)
    j = _wait(jr.start("fail"))
    assert j.status == "error" and j.error["code"] == "BAD_ARGS"
    j = jr.start("count", {"n": 200, "delay": 0.01})
    time.sleep(0.1)
    jr.cancel(j.id)
    assert _wait(j).status == "cancelled" and j.done < 200
    with pytest.raises(GneError):
        jr.start("nope")
    v = jr.get(j.id).view()
    assert v["status"] == "cancelled" and "eta_s" in v


def test_pick_rollouts_prefers_one_per_prompt_within_max_rel():
    recs = [{"prompt": p, "dose": d, "rel_push": abs(d), "text": f"{p}{d}"}
            for p in ("who", "snake") for d in (-0.7, 0.5, 0.8)] + [{"prompt": "free", "dose": 3.0, "rel_push": 3.0, "text": "f"}]
    got = _pick_rollouts(recs, 3)
    assert [(r["prompt"], r["dose"]) for r in got] == [("who", 0.8), ("snake", 0.8), ("free", 3.0)]
    assert len(_pick_rollouts(recs, 10)) == 7


def test_strip_ids_and_inline_contrast(tmp_path):
    assert _strip_ids({"a": 1, "token_ids": [1], "b": [{"token_ids": [2], "c": 3}]}) == {"a": 1, "b": [{"c": 3}]}

    class S:
        class store:
            root = tmp_path
    p = _contrast_path(S, None, "evil", "good", ["hi"], "mine")
    assert p.exists() and "evil" in p.read_text()
    with pytest.raises(GneError):
        _contrast_path(S, None, "evil", None, None, None)
    with pytest.raises(GneError):
        _contrast_path(S, "missing-contrast-name", None, None, None, None)
