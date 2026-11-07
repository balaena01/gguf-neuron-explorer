"""CHANGE-07 の eval の配線のテスト: keyed のレシピの評価(evalx.evaluate の keyed_*)を Session.eval・CLI・ジョブ・MCP から
使えること。

モデルなし: EvalRow・EvalOut が keyed を受けること、Session.eval が keyed_* を evaluate に渡すこと(対照集合の名前の解決・
引数の誤り)、gne eval の --keyed-* が Session.eval に届き、keyed の表示(方式の比較の行・行の選択性の列・skipped の理由と
ヒント)が出ること、ジョブ eval と MCP の eval ツールが同じ引数を渡すこと(evaluate・ジョブは差し替えて速く)。
モデルあり(Qwen3-0.6B、GNE_TEST_MODEL。CPU で回すときは GNE_TEST_NGL=0): ずんだもんの語尾の対照集合に与えた返事を足した
ファイルで方向と鍵を作り、小さいタスク集で `gne eval --json`(スキーマと keyed のブロック)、表示、鍵の対照集合が
分からないときの skipped を確かめる。
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest
import yaml
from typer.testing import CliRunner

from gne.schemas import COMMAND_SCHEMAS, ErrorOut, EvalOut, EvalRow, ProgressLine
from gne.steer import GneError, Recipe, parse_unit

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
NGL = os.environ.get("GNE_TEST_NGL", "-1")
ZUNDAMON = ROOT / "configs" / "contrast" / "zundamon_endings.yaml"
UNITS = ["8:ffn:keyed:a1@zun_wire#zun_wire", "14:ffn:keyed:a1@zun_wire#zun_wire"]

ENV_STUB = {"schema_version": "1", "command": "eval", "model_id": "m", "llama_cpp_version": "v", "seed": 0,
            "generated_at": "2026-11-07T00:00:00+00:00", "warnings": []}


# --- 作り物の結果 -------------------------------------------------------------------------------------

def _row(dose: float, sel: float | None, null_pct: float | None = None) -> dict:
    row = {"dose": dose, "exact": 1.0, "format": 1.0, "capability": 1.0, "ppl": {"en": 10.0}, "ppl_mean": 10.0,
           "mix_zh": 0.0, "mix_other": 0.0,
           "persona": {"novelty": 0.1, "mass": 0.0, "top_words": [], "distinct4": 0.9, "length": 20.0, "capped": 0.0},
           "failed": {"exact": [], "format": []}, "capability_drop_pt": 0.0, "ppl_worse_pct": 0.0, "mix_increase": 0.0,
           "ok": {"capability": True, "ppl": True, "mix": True}, "shippable": True}
    if sel is not None:
        row["keyed"] = {"layer": 14, "a_par": sel, "a_norm": abs(sel), "b_par": 0.0, "b_norm": 0.5, "dt_par": 0.0,
                        "dt_norm": 0.4, "selectivity": sel, "b_over_a": 0.2 if sel else None,
                        **({"null_pct": null_pct} if null_pct is not None else {})}
    return row


def _result(keyed: dict | None, sel=(0.0, 2.5)) -> dict:
    rows = [_row(0.0, sel[0]), _row(1.0, sel[1])] if sel else [_row(0.0, None), _row(1.0, None)]
    d = {"name": "zw", "path": "/x/zw.json", "created": "2026-11-07T00:00:00",
         "recipe": {"model_id": "m", "dose_unit": "rel", "units": [], "name": "zw"}, "when": "all",
         "suite": {"name": "mini", "path": "mini.yaml", "n_tasks": 3,
                   "thresholds": {"capability_drop_pt": 5, "ppl_worse_pct": 10, "mix_increase": 0}},
         "doses": [0.0, 1.0], "rows": rows, "samples": {}, "judge": None, "seconds": 1.0, "random_controls": 0}
    if keyed is not None:
        d["keyed"] = keyed
        d["notes"] = [f"keyed: {n}" for n in keyed.get("notes", [])] + (
            [f"keyed: 選択性は測っていません({keyed['skipped']})"] if "skipped" in keyed else [])
    return d


BLOCK = {"key": "zk", "direction": "zd", "objective_layer": 14, "methods": [{"method": "keyed", "selectivity": 2.5}],
         "pseudo": {"n": 2, "p50": 0.3, "p95": 0.9, "real": 2.5, "beats": True}, "notes": ["鍵の注"],
         "text": ["keyed の選択性(鍵 zk、方向 zd、L14 の残差、保留 subset odd)", "  keyed  …  +2.5", "  注: 鍵の注"]}
SKIPPED = {"skipped": "鍵 zk の対照集合が分かりません", "hint": "鍵を作った対照集合を --keyed-contrast で渡してください"}


def test_schemas_accept_keyed():
    """EvalRow・EvalOut(どちらも extra="forbid")が行の keyed と結果の keyed(ブロックと skipped)を受ける。"""
    EvalRow.model_validate(_row(1.0, 2.5, 0.97))
    EvalOut.model_validate({**ENV_STUB, **_result(BLOCK)})
    EvalOut.model_validate({**ENV_STUB, **_result(SKIPPED, sel=None)})
    assert COMMAND_SCHEMAS["eval"] is EvalOut
    with pytest.raises(Exception):
        EvalOut.model_validate({**ENV_STUB, **_result(BLOCK), "keyed_x": 1})


# --- Session.eval → evaluate ----------------------------------------------------------------------------

def _fake_session(tmp_path):
    from gne.api import Session
    s = SimpleNamespace(warnings=[], progress=None, meta=SimpleNamespace(model_id="m"),
                        store=SimpleNamespace(eval_dir=lambda mid: tmp_path),
                        _contrast_file=Session._contrast_file)
    s.warn = s.warnings.append
    s._as_recipe = lambda r: r
    return s


def test_session_eval_passes_keyed_args(monkeypatch, tmp_path):
    """keyed_* が evaluate に届く。対照集合は名前でも受け(configs/contrast/)、keyed_examples の 0 は全部(None)、
    空の keyed_subset・keyed_contrast は省いたのと同じ。"""
    import gne.evalx
    from gne.api import Session
    got = []

    def fake_evaluate(session, recipe, **kw):
        got.append(kw)
        return {"when": "all", "_baseline": {}, "keyed": {"skipped": "x"}}
    monkeypatch.setattr(gne.evalx, "evaluate", fake_evaluate)
    s = _fake_session(tmp_path)
    r = Recipe([parse_unit("14:ffn:keyed:a1@zd#zk")], "m", name="zw")
    d = Session.eval(s, r, doses=(0, 1), keyed_contrast="zundamon_endings", keyed_examples=0, keyed_pseudo=3,
                     keyed_match=False, keyed_subset="all")
    kw = got[-1]
    assert kw["keyed_contrast"] == ZUNDAMON and kw["keyed_examples"] is None and kw["keyed_pseudo"] == 3
    assert kw["keyed_match"] is False and kw["keyed_subset"] == "all" and kw["keyed_metrics"] is True
    assert d["keyed"] == {"skipped": "x"} and Path(d["path"]).exists() and "_baseline" not in d
    Session.eval(s, r, keyed_contrast="", keyed_subset="", keyed_metrics=False)
    kw = got[-1]
    assert kw["keyed_contrast"] is None and kw["keyed_subset"] is None and kw["keyed_metrics"] is False
    assert kw["keyed_examples"] == 8 and kw["keyed_pseudo"] == 8 and kw["keyed_match"] is True     # 既定
    # 誤り: subset の名前、負の数、無い対照集合(長い eval の前に止める)
    for bad in ({"keyed_subset": "half"}, {"keyed_examples": -1}, {"keyed_pseudo": -2}):
        with pytest.raises(GneError) as e:
            Session.eval(s, r, **bad)
        assert e.value.code == "BAD_ARGS", bad
    with pytest.raises(GneError) as e:
        Session.eval(s, r, keyed_contrast="no_such_contrast")
    assert e.value.code == "FILE_NOT_FOUND"
    # keyed の無いレシピでは keyed_contrast を使わず、警告だけ
    n = len(got)
    Session.eval(s, Recipe([parse_unit("14:100:+0.5")], "m", name="plain"), keyed_contrast="no_such_contrast")
    assert len(got) == n + 1 and got[-1]["keyed_contrast"] is None
    assert any("keyed_contrast は使いません" in w for w in s.warnings)


# --- CLI ------------------------------------------------------------------------------------------------

class _CliSession:
    """gne eval が呼ぶ Session のメソッドだけ(eval の引数を記録して、作り物の結果を返す)。"""

    def __init__(self, result):
        self.result, self.calls, self.warnings = result, [], []

    def recipe(self, **kw):
        return Recipe([parse_unit(u) for u in kw.get("units") or []], "m", name="zw")

    def eval(self, r, **kw):
        self.calls.append(kw)
        return self.result

    def envelope(self, command):
        return {**ENV_STUB, "command": command}

    def close(self):
        pass


def _cli_eval(monkeypatch, result, *args):
    import gne.cli
    fake = _CliSession(result)
    monkeypatch.setattr(gne.cli.Common, "session", lambda self, model=None: fake)
    monkeypatch.setattr(gne.cli, "_gate", lambda *a, **k: {})
    r = CliRunner().invoke(gne.cli.app, ["eval", "--unit", "14:ffn:keyed:a1@zd#zk", *args])
    return r, fake


def test_cli_help_lists_keyed_options():
    from gne.cli import app
    r = CliRunner().invoke(app, ["eval", "--help"])
    assert r.exit_code == 0
    for opt in ("--keyed-contrast", "--keyed-examples", "--keyed-pseudo", "--keyed-subset", "--no-keyed-match",
                "--no-keyed"):
        assert opt in r.output, opt


def test_cli_eval_passes_and_shows_keyed(monkeypatch):
    """--keyed-* が Session.eval に届き、keyed の表示(text の行、行の選択性の列と説明)が出る。keyed の注は 2 回出さない。"""
    r, fake = _cli_eval(monkeypatch, _result(BLOCK), "--keyed-contrast", "zundamon_endings", "--keyed-examples", "0",
                        "--keyed-pseudo", "3", "--keyed-subset", "all", "--no-keyed-match")
    assert r.exit_code == 0, r.output
    kw = fake.calls[-1]
    assert kw["keyed_contrast"] == "zundamon_endings" and kw["keyed_examples"] is None and kw["keyed_pseudo"] == 3
    assert kw["keyed_subset"] == "all" and kw["keyed_match"] is False and kw["keyed_metrics"] is True
    out = r.output
    assert "keyed の選択性(鍵 zk" in out and "注: 鍵の注" in out and out.count("鍵の注") == 1
    head = next(ln for ln in out.splitlines() if ln.lstrip().startswith("倍率"))
    assert "選択性" in head and "null" not in head
    row1 = next(ln for ln in out.splitlines() if ln.lstrip().startswith("1 "))
    assert "+2.5" in row1 and "行の選択性 = L14" in out
    # 既定と --no-keyed
    r, fake = _cli_eval(monkeypatch, _result(None, sel=None), "--no-keyed")
    assert r.exit_code == 0, r.output
    kw = fake.calls[-1]
    assert kw["keyed_metrics"] is False and kw["keyed_contrast"] is None and kw["keyed_examples"] == 8
    assert kw["keyed_pseudo"] == 8 and kw["keyed_subset"] is None and kw["keyed_match"] is True
    assert "選択性" not in r.output


def test_cli_eval_keyed_null_and_skipped(monkeypatch):
    """擬似鍵の対照があれば選択性の列に null も。対照集合が分からなければ理由とヒントを表示する(注と重ねない)。"""
    d = _result(BLOCK)
    d["rows"][1] = _row(1.0, 2.5, 0.97)
    r, _ = _cli_eval(monkeypatch, d)
    head = next(ln for ln in r.output.splitlines() if ln.lstrip().startswith("倍率"))
    assert head.rstrip().endswith("null") and "97%" in r.output and "擬似鍵に替えた対照" in r.output
    r, _ = _cli_eval(monkeypatch, _result(SKIPPED, sel=None))
    assert r.exit_code == 0, r.output
    assert "keyed の選択性は測っていません: 鍵 zk の対照集合が分かりません" in r.output
    assert "ヒント: 鍵を作った対照集合を --keyed-contrast で渡してください" in r.output
    assert r.output.count("対照集合が分かりません") == 1 and "選択性 =" not in r.output
    r, _ = _cli_eval(monkeypatch, _result(BLOCK), "--json")
    assert r.exit_code == 0, r.output
    EvalOut.model_validate(json.loads(r.output.strip().splitlines()[-1]))


# --- ジョブ・MCP --------------------------------------------------------------------------------------

def test_job_eval_passes_keyed_args():
    from gne.jobs import _eval, standard_kinds
    assert standard_kinds()["eval"] is _eval
    calls = []
    s = SimpleNamespace(recipe=lambda **kw: Recipe([parse_unit(u) for u in kw["units"]], "m"),
                        eval=lambda r, **kw: calls.append(kw) or {"name": "zw", "samples": {}, "keyed": BLOCK})
    p = {"keyed_contrast": "zundamon_endings", "keyed_metrics": True, "keyed_examples": 4, "keyed_match": False,
         "keyed_pseudo": 2, "keyed_subset": "odd"}
    d = _eval(s, units=["14:ffn:keyed:a1@zd#zk"], doses=[0, 1], **p)
    assert calls[-1] == {"doses": [0, 1], **p}
    assert "samples" not in d and d["keyed"] is BLOCK


def test_mcp_eval_tool_passes_keyed_args(monkeypatch):
    """MCP の eval ツール: keyed_* をジョブの引数に入れる(対照集合の名前はパスに直す)。方式の比較では入れない。"""
    import gne.api
    import gne.mcp_server as ms
    started = []

    class FakeRunner:
        def __init__(self, *a, **k):
            pass

        def start(self, kind, params, **k):
            started.append((kind, params))
            return SimpleNamespace(view=lambda: {"id": "j1", "kind": kind, "params": params})
    monkeypatch.setattr(gne.api, "Session", lambda *a, **k: SimpleNamespace(warnings=[]))
    monkeypatch.setattr(ms, "JobRunner", FakeRunner)
    srv = ms.build_server("m.gguf")

    async def go(args):
        return await srv.call_tool("eval", args)
    anyio.run(go, {"units": UNITS, "keyed_contrast": "zundamon_endings", "keyed_examples": 0, "keyed_pseudo": 2,
                   "keyed_match": False, "keyed_subset": "odd"})
    kind, p = started[-1]
    assert kind == "eval" and p["units"] == UNITS
    assert p["keyed_contrast"] == str(ZUNDAMON) and p["keyed_examples"] is None and p["keyed_pseudo"] == 2
    assert p["keyed_match"] is False and p["keyed_subset"] == "odd" and p["keyed_metrics"] is True
    anyio.run(go, {"units": UNITS})
    p = started[-1][1]
    assert p["keyed_metrics"] is True and p["keyed_examples"] == 8 and p["keyed_pseudo"] == 8
    assert "keyed_contrast" not in p and "keyed_subset" not in p
    anyio.run(go, {"units": UNITS, "keyed_metrics": False})
    assert started[-1][1]["keyed_metrics"] is False
    anyio.run(go, {"units": UNITS, "compare_methods": True, "keyed_contrast": "zundamon_endings"})
    assert not any(k.startswith("keyed") for k in started[-1][1])
    assert "keyed_contrast" in ms.INSTRUCTIONS and "skipped" in ms.INSTRUCTIONS


# --- モデルを使う ------------------------------------------------------------------------------------

# モデル自身の素の返事に見立てた、です・ます調の返事(zundamon_endings.yaml の user_prompts と同じ順)
REPLIES = ["天気が良ければ公園を散歩するのがおすすめです。家ではゆっくり本を読むのもいいと思います。",
           "ご飯と味噌汁と卵焼きがおすすめです。忙しい日はバナナとヨーグルトでも十分です。",
           "料理は手軽に始められる趣味です。散歩や写真も気軽に楽しめます。",
           "雨の日は家で映画を見るのが好きです。温かいお茶を飲みながら過ごします。",
           "まずはしっかり眠ることが大切です。お風呂にゆっくり入るのも効果があります。",
           "私は秋が一番好きです。涼しくて食べ物がおいしい季節です。",
           "野菜と肉を炒めてから水で煮込みます。最後にルーを溶かせば完成です。",
           "寝る前にスマホを見ないようにすると眠りやすくなります。温かい牛乳を飲むのもおすすめです。",
           "京都はお寺や神社が多くておすすめです。北海道では自然とおいしい料理が楽しめます。",
           "時間を決めて短く区切るのが効果的です。机の上を片付けると集中しやすくなります。",
           "どちらもかわいいですが、私は猫が好きです。気ままなところが魅力です。",
           "毎日少しずつ歩くようにしています。野菜を多めに食べることも心がけています。",
           "温かい緑茶がおすすめです。夏は冷たい麦茶もおいしいです。",
           "まず要らない物を捨てることが大切です。物の置き場所を決めると散らかりにくくなります。",
           "相手の好きな物を選ぶのが一番です。手紙を添えると気持ちが伝わります。",
           "毎日同じ時間に寝るようにするといいです。起きたらすぐにカーテンを開けて光を浴びます。"]
EN = {"name": "english",
      "prompts": ["What is a good way to spend the weekend?", "How do I make a simple curry?",
                  "Any tips for falling asleep faster?", "What should I give a friend for their birthday?"],
      "replies": ["Going for a walk in the park is a nice way to relax. Reading a book at home is also good.",
                  "Fry vegetables and meat, then simmer them in water. Add the curry roux at the end.",
                  "Avoid screens before bed. A warm drink can also help you relax.",
                  "Choose something they like. Adding a handwritten note makes it special."]}


def _mini_suite(d: Path) -> Path:
    (d / "ppl").mkdir()
    (d / "ppl" / "en.txt").write_text("Alice was beginning to get very tired of sitting by her sister on the bank. " * 2)
    (d / "zh_only_chars.txt").write_text("# test\n这们说\n")
    s = {"name": "mini", "max_tokens": {"exact": 8, "format": 16, "mix": 16, "persona": 16},
         "exact": [{"id": "add", "prompt": "What is 17 + 25? Answer with just the number.", "answer": ["42"]}],
         "format": [{"id": "j", "prompt": "Return a JSON object with keys \"a\" and \"b\". Output only the JSON.",
                     "check": "json", "keys": ["a", "b"]}],
         "mix": [{"id": "m", "prompt": "春の好きなところを 2 文で教えてください。"}],
         "persona": {"prompt_sets": ["ja"]}, "ppl": [{"id": "en", "path": "ppl/en.txt"}]}
    p = d / "mini.yaml"
    p.write_text(yaml.safe_dump(s, allow_unicode=True))
    return p


def _run_gne(w, *args, json_out=True):
    env = {**os.environ, "GNE_ROOT": str(w["root"]), "GNE_MODEL": str(MODEL), "CUDA_VISIBLE_DEVICES": os.environ.get(
        "CUDA_VISIBLE_DEVICES", ""), "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}
    return subprocess.run([sys.executable, "-m", "gne.cli", *args, *(["--json"] if json_out else []), "--n-gpu-layers",
                           NGL, "--n-ctx", "1024"], capture_output=True, text=True, env=env, timeout=900, cwd=str(w["dir"]))


def gne_json(w, *args):
    """stdout が JSON だけで、最後の行がコマンドのスキーマを通ることを確かめる。"""
    p = _run_gne(w, *args)
    objs = [json.loads(ln) for ln in p.stdout.splitlines() if ln.strip()]
    assert objs, f"stdout が空です: {p.stderr[-2000:]}"
    for o in objs[:-1]:
        ProgressLine.model_validate(o)
    last = objs[-1]
    if p.returncode != 0:
        ErrorOut.model_validate(last)
    assert p.returncode == 0, (last, p.stderr[-2000:])
    COMMAND_SCHEMAS[last["command"]].model_validate(last)
    return last


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """ずんだもんの対照集合に返事(です・ます調と英語)を足したファイル 1 つで、方向(system_a 対 なし、生成なしの last)と
    鍵(mode same、「です。」「ます。」の直前、層 8・14、擬似鍵 4 本)を作る。生成するのは eval のタスク集だけ。"""
    if not MODEL.exists():
        pytest.skip(f"{MODEL} がありません")
    d = tmp_path_factory.mktemp("c07wire")
    spec = yaml.safe_load(ZUNDAMON.read_text(encoding="utf-8"))
    spec.update({"name": "zun_wire", "replies": REPLIES, "dont_touch": [EN]})
    spec.pop("validate", None)
    cp = d / "zun_wire.yaml"
    cp.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
    w = {"root": d / "root", "dir": d, "contrast": cp, "suite": _mini_suite(d)}
    gne_json(w, "direction", "--contrast", str(cp), "--positions", "last")
    k = gne_json(w, "direction", "key", "--contrast", str(cp), "--layers", "8,14", "--max-tokens", "24", "--pseudo", "4")
    assert k["name"] == "zun_wire" and k["mode"] == "same"
    return w


def _eval_args(w, *extra):
    return ["eval", *[a for u in UNITS for a in ("--unit", u)], "--doses", "0,1", "--suite", str(w["suite"]),
            "--no-direction-metrics", "--keyed-examples", "2", "--keyed-pseudo", "2", "--yes", *extra]


@pytest.mark.model
def test_eval_json_keyed_block(work):
    """gne eval --json: スキーマを通り、結果の keyed(方式の比較・擬似鍵・text)と行の keyed の選択性がある。"""
    d = gne_json(work, *_eval_args(work, "--keyed-contrast", str(work["contrast"]), "--name", "zw_json"))
    assert d["command"] == "eval"
    k = d["keyed"]
    assert "skipped" not in k, k
    assert k["contrast_from"] == "given" and k["held_out"] and k["subset"] == "odd" and k["objective_layer"] == 14
    assert k["counts"]["A"]["examples"] == 2 and k["settings"]["max_examples"] == 2 and k["settings"]["match"] is True
    assert [m["method"] for m in k["methods"]][:2] == ["keyed", "add_same_alpha"]
    assert k["pseudo"]["n"] == 2 and k["pseudo"]["keys"] == ["zun_wire~0", "zun_wire~1"]
    assert k["text"] and all(isinstance(x, str) for x in k["text"])
    rows = {r["dose"]: r for r in d["rows"]}
    assert rows[0.0]["keyed"]["selectivity"] == 0.0 and isinstance(rows[1.0]["keyed"]["selectivity"], float)
    assert rows[1.0]["keyed"]["layer"] == 14
    saved = json.loads(Path(d["path"]).read_text(encoding="utf-8"))
    assert saved["keyed"]["pseudo"]["n"] == 2


@pytest.mark.model
def test_eval_text_shows_keyed(work):
    """表示: 方向の対照集合から鍵の対照集合を引く(--keyed-contrast なし)。大きさ合わせを省いても keyed の表と行の選択性が出る。"""
    p = _run_gne(work, *_eval_args(work, "--no-keyed-match", "--name", "zw_text"), json_out=False)
    assert p.returncode == 0, p.stderr[-2000:]
    out = p.stdout
    assert "keyed の選択性(鍵 zun_wire、方向 zun_wire、L14 の残差、保留 subset odd)" in out
    assert "add_same_alpha" in out and "擬似鍵 2 本の選択性" in out and "行の選択性 = L14" in out
    head = next(ln for ln in out.splitlines() if ln.lstrip().startswith("倍率"))
    assert "選択性" in head


@pytest.mark.model
def test_eval_keyed_skipped(work):
    """鍵を方向と別の対照集合で作り、--keyed-contrast を渡さなければ keyed は skipped と hint(eval 自体は通る)。"""
    other = work["dir"] / "plain.yaml"
    spec = {"name": "plain", "positions": "before:(です|ます)。", "user_prompts": yaml.safe_load(
        ZUNDAMON.read_text(encoding="utf-8"))["user_prompts"], "replies": REPLIES}
    other.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
    gne_json(work, "direction", "key", "--contrast", str(other), "--layers", "14", "--max-tokens", "24", "--pseudo", "0",
             "--name", "plain_k")
    d = gne_json(work, "eval", "--unit", "14:ffn:keyed:a1@zun_wire#plain_k", "--doses", "0,1", "--suite",
                 str(work["suite"]), "--no-direction-metrics", "--yes", "--name", "zw_skip")
    k = d["keyed"]
    assert "pseudo" not in k and "対照集合" in k["skipped"] and "--keyed-contrast" in k["hint"]
    assert all("keyed" not in r for r in d["rows"])
    assert any("keyed の選択性は測りません" in w for w in d["warnings"])
    assert any("選択性は測っていません" in n for n in d["notes"])
