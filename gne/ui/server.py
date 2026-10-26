"""`gne serve`: Web UI のサーバ(FastAPI)。別冊 UI 設計 §8。127.0.0.1 にだけ bind する。

U0: /api/model(ModelMeta + ArchInfo + coverage)、/api/vocab(語彙の文字列)、静的配信。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..steer import GneError

STATIC = Path(__file__).resolve().parent / "static"


def create_app(session) -> FastAPI:
    from ..arch import arch_info
    app = FastAPI(title="gne", docs_url=None, redoc_url=None)
    app.add_middleware(GZipMiddleware, minimum_size=4096)
    app.state.session = session
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
                "directions": ov["directions"]}

    @app.get("/api/vocab")
    def vocab():
        if "vocab" not in cache:
            cache["vocab"] = [p.decode("utf-8", "replace") for p in session.vocab.pieces]
        return cache["vocab"]

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
