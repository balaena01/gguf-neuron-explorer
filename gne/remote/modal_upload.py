"""Modal の Volume の上のファイルを、手元を通さず Hugging Face に上げる(gne modal hf-upload、docs/modal.md)。GPU は使わない。

トークンは Modal のシークレット huggingface(HF_TOKEN)から読む: modal secret create huggingface HF_TOKEN=<書き込み権限のトークン>
シークレットの無い人の gne modal fetch を止めないよう、modal_fetch.py とは別のアプリにしている。
"""
import json
import os
from pathlib import Path

import modal

DATA = "/data"
app = modal.App("gne-upload")
data = modal.Volume.from_name("gne-data", create_if_missing=True, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub[hf_xet]==0.36.0")


@app.function(image=image, cpu=4, memory=8192, timeout=6 * 3600, volumes={DATA: data.read_only()},
              secrets=[modal.Secret.from_name("huggingface")])
def upload(repo: str, paths: list[str], private: bool, card: str = "") -> list[str]:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, private=private, exist_ok=True, repo_type="model")
    done = []
    if card:
        api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=repo)
        done.append("README.md")
    for p in paths:
        src = Path(DATA) / p.lstrip("/")
        if not src.is_file():
            raise FileNotFoundError(f"/data/{p.lstrip('/')} がありません")
        api.upload_file(path_or_fileobj=str(src), path_in_repo=src.name, repo_id=repo)
        done.append(src.name)
        print("上げた", src.name, flush=True)
    return done


@app.local_entrypoint()
def main(repo: str, paths_json: str, private: bool = True, card_path: str = "") -> None:
    card = Path(card_path).read_text(encoding="utf-8") if card_path else ""
    for name in upload.remote(repo, json.loads(paths_json), private, card):
        print(name)
