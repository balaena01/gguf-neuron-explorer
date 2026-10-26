"""`gne serve`: Web UI のサーバ(FastAPI)。別冊 UI 設計 §8。127.0.0.1 にだけ bind する。

U0: /api/model(ModelMeta + ArchInfo + coverage)、/api/vocab(語彙の文字列)、静的配信。
U1: 層ごとの overlay の値・hover の語・rank の表、ニューロンの詳細、検索、名前、layers ビューの要約。
大きい配列は JSON にせず octet-stream で返し、形は X-Shape、付帯情報は X-Meta(JSON)に入れる(§8.2)。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import Body, FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ..steer import GneError

STATIC = Path(__file__).resolve().parent / "static"


def _binary(body: bytes, shape: str, meta: dict) -> Response:
    return Response(body, media_type="application/octet-stream",
                    headers={"X-Shape": shape, "X-Meta": json.dumps(meta, ensure_ascii=True), "Cache-Control": "no-store"})


def create_app(session, *, warm: bool = True) -> FastAPI:
    from ..arch import arch_info
    from .data import UIData
    app = FastAPI(title="gne", docs_url=None, redoc_url=None)
    app.add_middleware(GZipMiddleware, minimum_size=4096)
    app.state.session = session
    data = UIData(session)
    app.state.data = data
    if warm:
        threading.Thread(target=data.warm, name="gne-ui-warm", daemon=True).start()
    cache: dict = {}

    @app.exception_handler(GneError)
    async def _gne_error(_req, e: GneError):
        return JSONResponse({"error": {"code": e.code, "message": e.message, "hint": e.hint}}, status_code=400)

    @app.get("/api/model")
    def model():
        if "arch" not in cache:
            cache["arch"] = arch_info(session.weights).to_dict()
        ov = session.model_overview()
        return {"meta": ov["meta"], "arch": cache["arch"], "coverage": ov["layers"],
                "arch_warnings": ov["arch_warnings"], "bake": ov["bake"], "recipes": ov["recipes"],
                "directions": ov["directions"], "names": data.names()}

    @app.get("/api/vocab")
    def vocab():
        return data.vocab_text()

    @app.get("/api/layer/{L}/field")
    def field(L: int, name: str, run: str | None = None):
        v, meta = data.field(L, name, run=run)
        return _binary(v.tobytes(), str(v.size), meta)

    @app.get("/api/layer/{L}/hover")
    def hover(L: int):
        body, meta = data.hover(L)
        n = session.meta.n_ff_at(L)
        return _binary(body, f"{n},24;{n},2", meta)

    @app.get("/api/layer/{L}/units")
    def units(L: int):
        return data.layer_units(L)

    @app.get("/api/unit")
    def unit(layer: int, n: int, kind: str = "neuron", run: str | None = None):
        return data.unit(layer, n, kind=kind, run=run)

    @app.post("/api/name")
    def name(layer: int = Body(...), neuron: int = Body(...), name: str = Body("")):
        data._check(layer)
        session.set_name(layer, neuron, name.strip())
        return {"layer": layer, "neuron": neuron, "name": name.strip() or None, "names": data.names()}

    @app.get("/api/search")
    def search(q: str, mode: str = "lens", limit: int = 500):
        return data.search(q, mode=mode, limit=limit)

    @app.get("/api/profile")
    def profile():
        return data.profile()

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/", StaticFiles(directory=STATIC), name="static")
    return app


def serve(model: str | Path, *, host: str = "127.0.0.1", port: int = 8791, root: str | Path | None = None,
          n_gpu_layers: int = -1, n_ctx: int = 4096, seed: int = 0) -> None:
    import uvicorn

    from ..api import Session
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise GneError("BAD_ARGS", "UI はローカル専用です(--host は 127.0.0.1)", "別冊 UI 設計 §1 の非ゴール: マルチユーザー・認証")
    session = Session(model, root=root, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, seed=seed)
    uvicorn.run(create_app(session), host=host, port=port, log_level="warning")
