"""test_qwen4exp.py を CPU だけの子プロセスで流す(理由はそのファイルの先頭)。"""
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", reason="GPU が見えないので test_qwen4exp.py がそのまま走る")
def test_qwen4exp_on_cpu_only():
    target = Path(__file__).with_name("test_qwen4exp.py")
    r = subprocess.run([sys.executable, "-m", "pytest", str(target), "-q", "-p", "no:cacheprovider"], capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=False)
    assert r.returncode == 0, r.stdout[-3000:]
