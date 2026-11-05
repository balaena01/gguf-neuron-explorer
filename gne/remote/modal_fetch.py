"""Hugging Face のモデルを手元を通さず Modal の Volume に落とす(gne modal fetch、docs/modal.md)。GPU は使わない。

modal run は同じファイルの全部のイメージをビルドするので、CUDA のビルドに巻き込まれないよう modal_app.py と分けている。
"""
import json
from pathlib import Path

import modal

MODELS = "/models"

app = modal.App("gne-fetch")
models = modal.Volume.from_name("gne-models", create_if_missing=True, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub[hf_xet]==0.36.0")


@app.function(image=image, cpu=4, memory=8192, timeout=4 * 3600, volumes={MODELS: models})
def fetch(repo: str, include: list[str]) -> list[str]:
    from huggingface_hub import snapshot_download
    root = Path(MODELS) / repo
    snapshot_download(repo, allow_patterns=include or None, local_dir=root)
    models.commit()
    return [str(p) for p in sorted(root.rglob("*.gguf"))]


@app.local_entrypoint()
def main(repo: str, include_json: str = "[]") -> None:
    for p in fetch.remote(repo, json.loads(include_json)):
        print(p)
