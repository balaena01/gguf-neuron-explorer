"""rollout の文埋め込みと UMAP(別冊 UI 設計 §7、ジョブ `embed`)。

1 つの run(層)の全 rollout(ニューロン・ランダム対照・ベースライン)を、GGUF の埋め込みモデル(llama-cpp-python の
embedding=True。既定は Qwen3-Embedding-0.6B)で文埋め込みにし、UMAP(cosine)で 2D にする。
出力: runs/<run>.umap.json({meta, points: [[x, y, kind, neuron, prompt, dose, idx], ...], texts: [...]})と
runs/<run>.emb.npy(埋め込み、f16。UMAP だけやり直すとき用)。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from .steer import GneError

# クラスタを見る用途なので、全文に同じ指示を付ける(Qwen3-Embedding は指示つきで 1〜5% 良い、と README)
INSTRUCT = "Instruct: Identify the topic, content and style of the given text\nQuery:"
KINDS = {"neuron": 0, "random": 1, "baseline": 2}


def default_embed_model() -> Path | None:
    """GNE_EMBED_MODEL、無ければ models/ の *Embedding*.gguf。"""
    env = os.environ.get("GNE_EMBED_MODEL")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent.parent / "models"
    found = sorted(here.glob("*[Ee]mbed*.gguf"))
    return found[0] if found else None


class Embedder:
    def __init__(self, path: str | Path, *, n_gpu_layers: int = -1, n_ctx: int = 1024):
        import llama_cpp
        if not Path(path).exists():
            raise GneError("FILE_NOT_FOUND", f"埋め込みモデル {path} がありません",
                           "GNE_EMBED_MODEL で GGUF の埋め込みモデル(Qwen3-Embedding-0.6B など)を指定してください")
        self.path = Path(path)
        self.llm = llama_cpp.Llama(model_path=str(path), embedding=True, n_ctx=n_ctx, n_batch=n_ctx, n_ubatch=n_ctx,
                                   n_gpu_layers=n_gpu_layers, verbose=False)

    def embed(self, texts: list[str], *, batch: int = 16, progress=None) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            chunk = [f"{INSTRUCT}{t[:2000]}" for t in texts[i:i + batch]]   # 長すぎる文は切る(rollout は 128 トークン程度)
            out.append(np.asarray(self.llm.embed(chunk, normalize=True, truncate=True), np.float32))
            if progress:
                progress(min(i + batch, len(texts)), len(texts), "embed")
        return np.concatenate(out, axis=0)

    def close(self) -> None:
        self.llm.close()


def embed_run(session, run: str, *, model: str | Path | None = None, seed: int = 0, progress=None) -> dict:
    from .layout import _umap
    from .sweep import read_rollouts
    path = session.run_path(run)
    if not path.exists():
        raise GneError("FILE_NOT_FOUND", f"run {path} がありません")
    recs = [r for r in read_rollouts(path) if r.get("kind") in KINDS and r.get("text") is not None]
    if len(recs) < 5:
        raise GneError("BAD_ARGS", f"run {run} の rollout が少なすぎます({len(recs)} 本)")
    model = Path(model) if model else default_embed_model()
    if model is None:
        raise GneError("FILE_NOT_FOUND", "埋め込みモデルがありません",
                       "models/ に Qwen3-Embedding-0.6B-Q8_0.gguf などを置くか、GNE_EMBED_MODEL で指定してください")
    t0 = time.perf_counter()
    em = Embedder(model)
    try:
        E = em.embed([r["text"] for r in recs], progress=progress)
    finally:
        em.close()
    t_embed = time.perf_counter() - t0
    if progress:
        progress(len(recs), len(recs), "UMAP")
    Y = np.asarray(_umap(E, metric="cosine", seed=seed), np.float32)
    span = float(np.max(Y.max(0) - Y.min(0))) or 1.0                     # 縦横を同じ倍率で 0〜1 に(短い軸は中央に寄せる)
    Y = (Y - Y.min(0)) / span + (1 - (Y.max(0) - Y.min(0)) / span) / 2
    base = path.with_name(path.name.removesuffix(".jsonl"))
    np.save(str(base) + ".emb.npy", E.astype(np.float16))
    prompts = sorted({r["prompt"] for r in recs})
    points = [[round(float(x), 5), round(float(y), 5), KINDS[r["kind"]], r.get("neuron"), r["prompt"], r.get("dose"), i]
              for i, ((x, y), r) in enumerate(zip(Y, recs, strict=True))]
    meta = {"run": run, "model": model.name, "n": len(recs), "dim": int(E.shape[1]), "prompts": prompts,
            "kinds": list(KINDS), "instruct": INSTRUCT, "seconds_embed": t_embed, "seconds": time.perf_counter() - t0,
            "layer": recs[0].get("layer")}
    out = Path(str(base) + ".umap.json")
    out.write_text(json.dumps({"meta": meta, "points": points, "texts": [r["text"][:600] for r in recs]},
                              ensure_ascii=False), encoding="utf-8")
    return {"path": str(out), **meta}
