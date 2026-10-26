"""V10: すべてのコマンドの --json が stdout に JSON だけを出し、スキーマ(pydantic)を通る。
実際に `gne` を子プロセスで起動して確かめる(C 側のログが stdout に混ざらないことも含めて)。"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gne.schemas import COMMAND_SCHEMAS, ErrorOut, ProgressLine
from gne.verify import find_llama_completion

ROOT = Path(__file__).resolve().parent.parent
QWEN = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
LLAMA = Path(os.environ.get("GNE_TEST_LLAMA_MODEL", ROOT / "models" / "SmolLM2-135M-Instruct-Q8_0.gguf"))
GNE = [str(Path(sys.executable).parent / "gne")]

pytestmark = [pytest.mark.model, pytest.mark.skipif(not QWEN.exists(), reason=f"{QWEN} がありません")]


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return tmp_path_factory.mktemp("gne_root")


def gne_json(root, *args, expect_ok=True):
    env = {**os.environ, "GNE_ROOT": str(root)}
    p = subprocess.run([*GNE, *args, "--json"], capture_output=True, text=True, env=env, timeout=600)
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    assert lines, f"stdout が空です: {p.stderr[-2000:]}"
    objs = [json.loads(ln) for ln in lines]                  # JSON 以外が混ざると例外
    for o in objs[:-1]:
        ProgressLine.model_validate(o)
    last = objs[-1]
    if expect_ok:
        assert p.returncode == 0, (p.returncode, last)
        COMMAND_SCHEMAS[last["command"]].model_validate(last)
    else:
        assert p.returncode != 0
        ErrorOut.model_validate(last)
    return last


def test_info_probe(root):
    d = gne_json(root, "info", "-m", str(QWEN))
    assert d["meta"]["arch"] == "qwen3" and d["bake"]["supported"] is False
    d = gne_json(root, "probe", "--list-names", "--layer", "5", "-m", str(QWEN))
    assert d["act_name"] == "ffn_swiglu-5"


def test_stats_steer_export_import(root, tmp_path):
    d = gne_json(root, "stats", "resid", "-m", str(QWEN), "--tokens", "4096")
    assert d["stats"]["n_tokens"] > 3000 and len(d["stats"]["layers"]) == 27
    d = gne_json(root, "stats", "act", "-m", str(QWEN), "--layers", "13-14", "--tokens", "2048")
    assert [r["layer"] for r in d["layers"]] == [13, 14]
    d = gne_json(root, "steer", "-m", str(QWEN), "--unit", "14:1234:+0.8", "-p", "Who are you?",
                 "--compare", "--max-tokens", "24")
    assert d["differs"] is True and d["baseline"]["token_ids"]
    d = gne_json(root, "steer", "-m", str(QWEN), "--unit", "14:1234:+0.8", "--dose-unit", "absmax", "-p", "Hi",
                 "--max-tokens", "8")
    assert d["recipe"]["dose_unit"] == "absmax"
    d = gne_json(root, "steer", "-m", str(QWEN), "--unit", "27:5:+0.5", "-p", "Hi", "--max-tokens", "8")
    assert d["units"][0]["rho"] > 0 and any("stats resid" in w for w in d["warnings"])   # 最終層の ρ を自動で足す
    args = ["export-cvec", "-m", str(QWEN), "--unit", "14:1234:+0.8", "-o", str(tmp_path / "x.cvec.gguf")]
    if find_llama_completion():
        args.append("--verify")
    d = gne_json(root, *args)
    assert d["verify"] is None or d["verify"]["match"]
    d = gne_json(root, "export-cvec", "-m", str(QWEN), "--unit", "14:1234:+0.8", "--format", "safetensors",
                 "-o", str(tmp_path / "x.safetensors"))
    d = gne_json(root, "import-cvec", "-m", str(QWEN), str(tmp_path / "x.cvec.gguf"), "--source", "generator")
    assert list(d["layer_norms"]) == ["13"]


def test_verify_and_bench(root):
    d = gne_json(root, "verify", "-m", str(QWEN), "--no-bench")
    assert d["n_fail"] == 0
    d = gne_json(root, "bench", "-m", str(QWEN), "--reps", "2", "--gen-tokens", "8")
    assert d["bench"]["prefill_ms"] > 0


def test_errors_are_json(root, tmp_path):
    d = gne_json(root, "bake", "-m", str(QWEN), "--unit", "14:1:+0.8", "--out", str(tmp_path / "o.gguf"),
                 expect_ok=False)
    assert d["error"]["code"] == "ARCH_UNSUPPORTED" and d["model_id"]
    d = gne_json(root, "steer", "-m", str(QWEN), "--unit", "14-1-0.8", "-p", "x", expect_ok=False)
    assert d["error"]["code"] == "BAD_UNIT"
    d = gne_json(root, "info", "-m", str(tmp_path / "missing.gguf"), expect_ok=False)
    assert d["error"]["code"] == "MODEL_NOT_FOUND" and d["model_id"] is None


@pytest.mark.skipif(not LLAMA.exists(), reason=f"{LLAMA} がありません")
def test_bake_llama_v9(root, tmp_path):
    d = gne_json(root, "bake", "-m", str(LLAMA), "--unit", "15:100:+0.8", "--unit", "12:200:-0.6",
                 "--out", str(tmp_path / "baked.gguf"), "--verify")
    v = d["verify"]
    assert d["report"]["added"] == ["blk.12.ffn_down.bias", "blk.15.ffn_down.bias"]
    assert v["match"] and v["differs_from_baseline"] and v["kl_next"] < 0.05


def test_lens_and_reverse(root):
    d = gne_json(root, "lens", "logit", "-m", str(QWEN), "--layer", "14", "--neurons", "1,2,3", "--top-k", "6")
    assert [r["neuron"] for r in d["neurons"]] == [1, 2, 3] and len(d["neurons"][0]["up"]) == 6
    d = gne_json(root, "lens", "steer", "-m", str(QWEN), "--layer", "14", "--neurons", "5,6",
                 "--random-controls", "4", "--top-k", "6")
    assert d["n_controls"] == 4 and d["null"]["pos"]["n"] == 4 and [r["neuron"] for r in d["neurons"]] == [5, 6]
    d = gne_json(root, "reverse", "-m", str(QWEN), "--contrast", str(ROOT / "configs" / "contrast" / "evil.yaml"),
                 "--top", "30", "--verify", "--verify-top", "2", "--verify-controls", "1", "--max-tokens", "16")
    assert d["n_candidates"] == 30 and all(c["layer"] != 0 for c in d["candidates"])
    assert len(d["verify"]["examples"]) == 2 and all(ex["text"] for ex in d["verify"]["examples"])
    d = gne_json(root, "reverse", "-m", str(QWEN), "--directions", "evil", "--top", "10")
    assert d["info"]["method"] == "mean_diff" and d["n_candidates"] == 10
