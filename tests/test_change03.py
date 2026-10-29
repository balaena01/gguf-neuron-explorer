"""CHANGE-03 A のテスト: 方向・レシピ・eval のランダム対照(同じ大きさで向きだけランダム)。

モデルを使うものは Qwen3-0.6B(GNE_TEST_MODEL)。無ければ飛ばす。
"""

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from gne.steer import AblateSpec, Recipe, parse_unit

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))


class _FakeSession:
    """control_recipes が使う vectors(recipe) → (vecs, details) だけを持つ。"""

    def vectors(self, recipe):
        return {}, [{"rel_push": None if u.mode == "gain" else 0.1 * (i + 1)} for i, u in enumerate(recipe.units)]


def test_control_count_and_norm():
    from gne.controls import SEED_BASE, control_recipes, control_vectors
    v = np.random.default_rng(0).standard_normal(64).astype(np.float32) * 3
    cs = control_vectors(64, v, 7, 16)
    assert len(cs) == 16 and all(c.shape == (64,) and c.dtype == np.float32 for c in cs)
    assert np.allclose([np.linalg.norm(c) for c in cs], np.linalg.norm(v), rtol=1e-5)          # 同じノルム
    G = np.stack([c / np.linalg.norm(c) for c in cs])
    assert np.abs(G @ G.T - np.eye(16)).max() < 0.7                                           # 向きはばらばら
    assert all(np.array_equal(a, b) for a, b in zip(cs, control_vectors(64, v, 7, 16), strict=True))   # seed 固定
    assert not np.allclose(cs[0], control_vectors(64, v, 8, 1)[0])                            # 層ごとに違う
    assert control_vectors(64, v, 7, 0) == []
    # レシピの対照: add ユニットは同じ層・同じ rel の大きさのランダムに、gain は残す、ablate はランダムな向きの消去に
    r = Recipe([parse_unit("5:10:+0.8"), parse_unit("6:dir:x:+0.5"), parse_unit("7:3:x0")], name="r", when="gen",
               ablate=AblateSpec("evil", 0.5, "all", None))
    out = control_recipes(_FakeSession(), r, 4)
    assert len(out) == 4 and [x.name for x in out] == [f"r-random{j}" for j in range(4)]
    for j, c in enumerate(out):
        a, b, g = c.units
        assert (a.kind, a.layer, a.dose) == ("random", 5, pytest.approx(0.1)) and (b.kind, b.layer, b.dose) == ("random", 6, pytest.approx(0.2))
        assert g == r.units[2] and c.when == "gen" and c.dose_unit == "rel"
        assert c.ablate.direction == f"random-{SEED_BASE + j}" and c.ablate.scale == 0.5
    seeds = {u.seed for c in out for u in c.units if u.kind == "random"}
    assert len(seeds) == 8                                                                     # ユニット・対照ごとに別の seed
    assert control_recipes(_FakeSession(), Recipe([parse_unit("7:3:x2")]), 4) == []           # gain だけなら作れない
    assert control_recipes(_FakeSession(), r, 0) == []


def test_null_pct():
    from gne.controls import null_pct, summarize
    ctrl = [0.1, 0.2, 0.6, 0.9]
    assert null_pct(0.5, ctrl) == 0.5 and null_pct(1.0, ctrl) == 1.0 and null_pct(0.0, ctrl) == 0.0
    assert null_pct(0.2, [0.2, 0.2]) == 0.0                                                    # 同じ値は「上」に数えない
    assert null_pct(0.5, [None, 0.1]) == 1.0 and null_pct(None, ctrl) is None and null_pct(0.5, []) is None
    s = summarize(list(range(21)))
    assert s["n"] == 21 and s["p50"] == 10 and s["p95"] == pytest.approx(19) and s["max"] == 20 and s["p05"] == pytest.approx(1)
    assert summarize([None]) == {"n": 0, "p05": None, "p50": None, "p95": None, "max": None}


# --- モデルを使う ----------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def session(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip("モデルがありません")
    from gne.api import Session
    s = Session(MODEL, root=tmp_path_factory.mktemp("c03"))
    yield s
    s.close()


def _contrast(tmp_path):
    p = tmp_path / "pirate.yaml"
    p.write_text(yaml.safe_dump({
        "name": "pirate", "system_a": "You are a pirate. Always talk like a pirate, with arr and matey.",
        "system_b": "You are a helpful assistant.",
        "user_prompts": ["Who are you?", "Tell me about the weather.", "What is your favorite food?"],
        "validate": {"match": r"\barr+\b|matey|ahoy|ye\b", "target": "a pirate"}}))
    return p


def _mini_suite(tmp_path):
    (tmp_path / "ppl").mkdir()
    (tmp_path / "ppl" / "en.txt").write_text("Alice was beginning to get very tired of sitting by her sister on the bank. " * 3)
    d = {"name": "mini", "max_tokens": {"exact": 12, "format": 24, "mix": 16, "persona": 24},
         "exact": [{"id": "add", "prompt": "What is 17 + 25? Answer with just the number.", "answer": ["42"]}],
         "format": [{"id": "j", "prompt": "Return a JSON object with keys \"a\" and \"b\". Output only the JSON.",
                     "check": "json", "keys": ["a", "b"]}],
         "persona": {"prompt_sets": ["en"]}, "ppl": [{"id": "en", "path": "ppl/en.txt"}]}
    p = tmp_path / "mini.yaml"
    p.write_text(yaml.safe_dump(d, allow_unicode=True))
    return p


_KEYS = ("effect", "kl", "rate", "distinct4", "sample", "rel_dose")


@pytest.mark.model
def test_random_controls_zero_is_unchanged(session, tmp_path):
    """random_controls=0 は従来と同じ結果(対照の項目が無い)。対照を入れても方向そのものの測定値は変わらない。"""
    from gne.schemas import COMMAND_SCHEMAS
    c = _contrast(tmp_path)
    session.direction(contrast=c)
    kw = {"contrast": c, "name": "pirate", "select": "auto", "mode": "add", "layers": "10-14:4", "dose": 4.0,
          "max_tokens": 16}
    a = session.direction(**kw, random_controls=0)["select"]
    b = session.direction(**kw, random_controls=2)["select"]
    assert a["random_controls"] == 0 and b["random_controls"] == 2
    for ra, rb in zip(a["rows"], b["rows"], strict=True):
        assert {k: ra[k] for k in _KEYS} == {k: rb[k] for k in _KEYS}
        assert not any(k.startswith("ctrl") or k == "null_pct" for k in ra)
        assert ra["ok"] == (ra["distinct4"] >= a["criteria"]["min_distinct"])                    # 従来の基準だけ
        if ra["ok"] and ra["effect"] > 0:                       # 選ばれ得る層だけ対照を流す
            assert rb["ctrl"]["n"] == 2 and len(rb["ctrl"]["rows"]) == 2 and 0 <= rb["null_pct"] <= 1
            assert rb["ok"] == (rb["effect"] > rb["ctrl_effect_p95"])
        else:
            assert rb["ctrl_skipped"] and "ctrl" not in rb and rb["ok"] == ra["ok"]
    assert any("ctrl" in r for r in b["rows"]) or b["selected"] is None
    # ablate: ランダムな向きの消去は層によらないので 1 回だけ測り、全層の行で同じ対照を使う
    ab = session.direction(contrast=c, name="pirate", select="auto", mode="ablate", layers="10-14:4", max_tokens=16,
                           random_controls=2)["select"]
    assert ab["rows"][0]["ctrl"]["rows"] == ab["rows"][1]["ctrl"]["rows"] and len(ab["rows"][0]["ctrl"]["rows"]) == 2
    assert len(list(session.store.ablate_dir(session.meta.model_id).glob("random*.lora.gguf"))) == 2
    # 対照の大きさ: 層の方向と同じノルム
    from gne.controls import control_vectors
    ds = session.direction_set("pirate")
    v = (4.0 * ds.raw[10]).astype(np.float32)
    assert np.allclose(np.linalg.norm(control_vectors(session.meta.n_embd, v, 10, 2)[1]), np.linalg.norm(v), rtol=1e-5)
    # レシピの対照: 同じ層で同じノルムの control vector になる
    from gne.controls import control_recipes
    r = session.recipe(units=["12:100:+0.8", "14:dir:pirate:+0.5"], name="mix")
    vr = session.vectors(r)[0]
    for cr in control_recipes(session, r, 2):
        vc = session.vectors(cr)[0]
        assert sorted(vc) == sorted(vr) and all(np.isclose(np.linalg.norm(vc[L]), np.linalg.norm(vr[L]), rtol=1e-4) for L in vr)
    # eval: random_controls=0 は従来どおり(ctrl なし)、入れても本体の行は同じ
    suite = _mini_suite(tmp_path)
    r1 = session.recipe(units=["14:100:+1.5"], name="c03")
    e0 = session.eval(r1, doses=[1.0], suite=suite, name="c03-a")
    e1 = session.eval(r1, doses=[1.0], suite=suite, name="c03-b", random_controls=2)
    assert e0["random_controls"] == 0 and e1["random_controls"] == 2
    for x, y in zip(e0["rows"], e1["rows"], strict=True):
        assert "ctrl" not in x and x["capability"] == y["capability"] and x["ppl_mean"] == y["ppl_mean"]
        assert x["persona"]["novelty"] == y["persona"]["novelty"]
    top = e1["rows"][-1]
    assert "ctrl" not in e1["rows"][0] and top["ctrl"]["n"] == 2 and top["ctrl_novelty_p95"] is not None
    assert 0 <= top["novelty_null_pct"] <= 1 and top["ctrl_ppl_worse_p50"] is not None
    COMMAND_SCHEMAS["eval"].model_validate({**session.envelope("eval"), **e1})
    # レシピの効果(steer --contrast): 対照レシピと比べた null_pct
    bn = session.behavior_null(session.recipe(units=["14:dir:pirate:+4.0"], name="p"), c, random_controls=2, max_tokens=16)
    assert bn["mode"] == "add" and bn["random_controls"] == 2 and len(bn["ctrl_rows"]) == 2 and 0 <= bn["null_pct"] <= 1
    assert bn["effect"] == pytest.approx(bn["rate"] - bn["base_rate"])


# --- C: 見積もり・予算(CHANGE-03 C) ---------------------------------------------------------------------

def test_parse_budget_and_spec():
    from gne.estimate import fit_count, fmt_seconds, parse_budget
    from gne.steer import GneError
    from gne.workers import _spec
    assert parse_budget("30m") == 1800 and parse_budget("2h") == 7200 and parse_budget("90s") == 90 and parse_budget("5") == 300
    assert parse_budget(None) is None
    with pytest.raises(GneError):
        parse_budget("soon")
    assert fit_count(2.0, 100, 10) == 45 and fit_count(2.0, 5, 10) == 1 and fit_count(1.0, 100, hi=7) == 7
    assert fmt_seconds(45) == "45 秒" and fmt_seconds(1800) == "30 分" and fmt_seconds(7200) == "2.0 時間"
    assert _spec([5, 0, 1, 2, 9, 7, 8]) == "0-2,5,7-9"


@pytest.mark.model
def test_estimate_gate_and_budget(session):
    from gne.estimate import NeedsConfirm, estimate, gate
    from gne.jobs import GpuArbiter, JobRunner, standard_kinds
    from gne.steer import GneError
    big = {"layer": 14, "neurons": "all", "design": "main_rel"}
    est = estimate(session, "sweep", big)
    assert est["seconds"] > 1800 and est["needs_confirm"] and est["detail"]["units"] == session.meta.n_ff_at(14) + 64
    with pytest.raises(NeedsConfirm):
        gate(session, "sweep", big)
    p, e = gate(session, "sweep", {**big, "yes": True})
    assert p == big and e["seconds"] == est["seconds"]
    p, e = gate(session, "sweep", {**big, "budget": "5m"})
    assert p["design"] == "quick" and p["random_controls"] == 16 and e["seconds"] <= 300 * 1.05
    assert len(p["neurons"].split(",")) >= 1 and e["budget"]["neurons"] > 0
    p, e = gate(session, "connections", {"budget": "10s"})
    assert p["window"] >= 1 and e["seconds"] <= 10 * 1.05 or p["window"] == 1
    runner = JobRunner(session, GpuArbiter(), standard_kinds())
    with pytest.raises(GneError) as ei:
        runner.start("sweep", big)
    assert ei.value.code == "NEEDS_CONFIRM" and ei.value.estimate["seconds"] == est["seconds"]
    assert estimate(session, "lens_steer", {"layer": 14, "neurons": "0-9", "random_controls": 2})["detail"]["units"] == 12


# --- B: ワーカー(CHANGE-03 B) ---------------------------------------------------------------------------

@pytest.mark.model
def test_workers_split_merge_equals_single(session, tmp_path):
    """lens steer と sweep を 2 つのワーカーに分けて回し、結合したものが 1 回で回したものと同じになる。"""
    from gne.api import Session
    from gne.lens import SteerLens
    from gne.workers import heartbeat, job_files, list_workers, run_worker, submit
    root = tmp_path / "shared"
    a = Session(MODEL, root=root)
    try:
        a.stats_resid()
        heartbeat(root, "wa", a)
        heartbeat(root, "wb", a)
        assert {w["name"] for w in list_workers(root, model_id=a.meta.model_id)} == {"wa", "wb"}
        sub = submit(a, "lens_steer", {"layer": 14, "neurons": "0-23", "random_controls": 4}, workers="auto")
        assert len(sub["jobs"]) == 2 and sorted(sub["workers"]) == ["wa", "wb"]
        js = {j["worker"]: j for j in job_files(root, sub["group"])}
        assert js["wa"]["params"]["neurons"] == "0-11" and js["wb"]["params"]["controls"] == [2, 3]
        assert run_worker(a, "wa", once=True) == 1 and run_worker(a, "wb", once=True) == 1
        done = job_files(root, sub["group"])
        assert all(j["status"] == "done" for j in done) and sum(bool(j.get("merge")) for j in done) == 1
        merged = SteerLens.load(root / "lens" / a.meta.model_id / "steer_L14.npz")
        assert merged.done[:24].all() and merged.ctrl_done.all() and not list((root / "lens" / a.meta.model_id).glob("*.part-*"))
        one = Session(MODEL, root=tmp_path / "single")
        try:
            one.stats_resid()
            one.lens_steer(14, neurons="0-23", random_controls=4)
            single = SteerLens.load(tmp_path / "single" / "lens" / one.meta.model_id / "steer_L14.npz")
            assert np.corrcoef(single.kl_pos[:24], merged.kl_pos[:24])[0, 1] > 0.99
            assert np.allclose(single.ctrl_kl_pos, merged.ctrl_kl_pos, rtol=1e-3)
            # sweep: 部分ごとの JSONL を rank が結合して読む
            sub2 = submit(a, "sweep", {"layer": 14, "neurons": "0-5", "design": "quick", "random_controls": 2,
                                       "max_tokens": 12, "run": "w"}, workers=["wa", "wb"])
            run_worker(a, "wa", once=True)
            run_worker(a, "wb", once=True)
            assert all(j["status"] == "done" for j in job_files(root, sub2["group"]))
            r2 = a.rank("w")
            assert len(r2["parts"]) == 2 and r2["n_neurons"] == 6 and r2["n_controls"] == 2
            one.sweep(14, neurons="0-5", design="quick", random_controls=2, max_tokens=12, run="w")
            r1 = one.rank("w")
            key = lambda r: [(u["kind"], u["neuron"], round(u["mass"], 6)) for u in r["units"]]  # noqa: E731
            assert sorted(key(r1)) == sorted(key(r2))
        finally:
            one.close()
        heartbeat(root, "other", a)
        import json as _json
        p = root / "workers" / "other.json"
        d = _json.loads(p.read_text())
        d["model_id"] = "different"
        p.write_text(_json.dumps(d))
        from gne.steer import GneError
        with pytest.raises(GneError):
            submit(a, "lens_steer", {"layer": 14, "neurons": "0-3"}, workers=["other"])   # 別のモデルには投げない
    finally:
        a.close()


@pytest.mark.model
def test_behavior_null_with_sampling(session, tmp_path):
    """steer --contrast をレシピのサンプリングで: プロンプト × 種の本数、同じ種なら同じ結果、プロンプトの差し替え。"""
    from gne.engine import Sampling
    c = _contrast(tmp_path)
    session.direction(contrast=c)
    r = session.recipe(units=["14:dir:pirate:+4.0"], name="p")
    sp = Sampling(temperature=0.8, top_k=40, top_p=0.95, min_p=0.05, presence_penalty=0.0, seed=3)
    a = session.behavior_null(r, c, random_controls=2, max_tokens=12, sampling=sp, seeds=2)
    b = session.behavior_null(r, c, random_controls=2, max_tokens=12, sampling=sp, seeds=2)
    assert a["n_samples"] == 6 and a["seeds"] == 2 and a["sampling"]["seed"] == 3 and a["max_tokens"] == 12
    assert (a["rate"], a["base_rate"], [x["rate"] for x in a["ctrl_rows"]]) == (b["rate"], b["base_rate"], [x["rate"] for x in b["ctrl_rows"]])
    pf = tmp_path / "ps.yaml"
    pf.write_text(yaml.safe_dump(["Ahoy there!", "Where is the treasure?"]))
    g = session.behavior_null(r, c, random_controls=1, max_tokens=8, prompts=pf)
    assert g["n_prompts"] == 2 and g["n_samples"] == 2 and g["sampling"] is None and g["prompts"] == ["Ahoy there!", "Where is the treasure?"]
