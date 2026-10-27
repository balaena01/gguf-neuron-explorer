"""`gne serve`: Web UI のサーバ(FastAPI)。別冊 UI 設計 §8。127.0.0.1 にだけ bind する。

U0: /api/model(ModelMeta + ArchInfo + coverage)、/api/vocab(語彙の文字列)、静的配信。
U1: 層ごとの overlay の値・hover の語・rank の表、ニューロンの詳細、検索、名前、layers ビューの要約。
U2: 3 列チャット(SSE)、fires、reads you、レシピ、cvec の書き出し、ジョブ(SSE の進捗)。GPU は live.Live が調停する。
U5: エージェント連携。MCP の ui_* が POST /api/ui/{highlight,note,open,clear} を叩き、ブラウザへ SSE(/api/ui/events)。
ローカル専用なので、Host が 127.0.0.1 / localhost 以外の要求(DNS rebinding)と、別のオリジンからの書き込み(POST / DELETE)は断る。
大きい配列は JSON にせず octet-stream で返し、形は X-Shape、付帯情報は X-Meta(JSON)に入れる(§8.2)。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import Body, FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..steer import GneError

STATIC = Path(__file__).resolve().parent / "static"


def _binary(body: bytes, shape: str, meta: dict) -> Response:
    return Response(body, media_type="application/octet-stream",
                    headers={"X-Shape": shape, "X-Meta": json.dumps(meta, ensure_ascii=True), "Cache-Control": "no-store"})


def create_app(session, *, warm: bool = True, openai: bool = False) -> FastAPI:
    from ..arch import arch_info
    from .agent import AgentBus
    from .data import UIData
    from .live import Live
    app = FastAPI(title="gne", docs_url=None, redoc_url=None)
    app.add_middleware(GZipMiddleware, minimum_size=4096)
    app.state.session = session
    data = UIData(session)
    app.state.data = data
    live = Live(session, data)
    if openai:                                     # §6.15: gne serve --openai
        from .openai_api import register
        register(app, session, live)
    app.state.live = live
    agent = AgentBus(session.meta.n_layer, session.meta.n_ff_at)
    app.state.agent = agent

    @app.middleware("http")
    async def local_only(request, call_next):
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
        if host not in ("127.0.0.1", "localhost", "::1", "testserver"):
            return JSONResponse({"error": {"code": "FORBIDDEN", "message": f"Host {host!r} は受け付けません(ローカル専用)"}},
                                status_code=403)
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin:
            o = origin.split("//", 1)[-1].split("/", 1)[0].rsplit(":", 1)[0].strip("[]")
            if o not in ("127.0.0.1", "localhost", "::1"):
                return JSONResponse({"error": {"code": "FORBIDDEN", "message": "別のオリジンからの書き込みは受け付けません"}},
                                    status_code=403)
        return await call_next(request)
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
                "directions": ov["directions"], "names": data.names(), "layouts": data.layouts(), "runs": data.run_list(),
                "connections": data.connections().meta if data.connections() is not None else None}

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

    @app.get("/api/layer/{L}/layout")
    def layout(L: int, kind: str):
        data._check(L)
        cell = data.layout(L, kind)
        if cell is None:
            return JSONResponse({"error": {"code": "NOT_FOUND", "message": f"L{L} の配置 {kind} はまだありません",
                                           "hint": "層の右クリックの「配置」で作れます"}}, status_code=404)
        return _binary(cell.tobytes(), str(cell.size), {"layer": L, "kind": kind})

    @app.get("/api/embeddings")
    def embeddings(run: str):
        e = data.embeddings(run)
        if e is None:
            return JSONResponse({"error": {"code": "NOT_FOUND", "message": f"run {run} の埋め込みはまだありません",
                                           "hint": "embed ビューの「埋め込みを計算」で作れます(GGUF の埋め込みモデルが要ります)"}},
                                status_code=404)
        return e

    @app.get("/api/connections")
    def connections(layer: int, n: int, k: int = 14):
        return data.links(layer, n, k)

    @app.get("/api/web")
    def web(layer: int, n: int, hops: int = 2, first: int = 8, k: int = 4, zmin: float = 2.0, cap: int = 20000):
        c = data.connections()
        if c is None:
            raise GneError("NOT_FOUND", "connections がありません", "層の右クリックの「connections」で作れます")
        e = c.web(layer, n, hops=max(1, min(hops, 5)), first=first, k=k, zmin=zmin, cap=cap)
        return _binary(e.tobytes(), f"{e.shape[0]},7", {"n_edges": int(e.shape[0])})

    @app.get("/api/web/summary")
    def web_summary(layer: int, n: int, hops: int = 2, first: int = 8, k: int = 4, zmin: float = 2.0, cap: int = 20000):
        from ..connections import web_summary as summarize
        c = data.connections()
        if c is None:
            raise GneError("NOT_FOUND", "connections がありません", "層の右クリックの「connections」で作れます")
        e = c.web(layer, n, hops=max(1, min(hops, 5)), first=first, k=k, zmin=zmin, cap=cap)
        return {"n_edges": int(e.shape[0]), **summarize(data, c, layer, n, e)}

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

    # --- U2: ライブ機能 ----------------------------------------------------------------------------------
    sse_headers = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}

    @app.post("/api/steer")
    def steer(body: dict = Body(...)):
        return StreamingResponse(live.steer_stream(body), media_type="text/event-stream", headers=sse_headers)

    @app.post("/api/steer/{sid}/stop")
    def steer_stop(sid: str):
        return live.stop(sid)

    @app.post("/api/fires")
    def fires(text: str = Body(...), top: int = Body(50)):
        return live.fires(text, top=top)

    @app.get("/api/fires/{fid}/field")
    def fires_field(fid: str, tok: int = -1):
        body, sizes, meta = live.fires_fields(fid, tok)
        return _binary(body, ",".join(map(str, sizes)), meta)

    @app.get("/api/fires/{fid}/neurons")
    def fires_neurons(fid: str, tok: int = -1, lmin: int = 0, lmax: int | None = None, sign: str = "both",
                      thr: float = 0.0, q: str = "", limit: int = 100):
        return live.fires_neurons(fid, tok, lmin=lmin, lmax=lmax, sign=sign, thr=thr, q=q, limit=limit)

    @app.post("/api/reads")
    def reads(layer: int = Body(...), neuron: int = Body(...), text: str = Body(...)):
        return live.reads(layer, neuron, text)

    @app.get("/api/recipes")
    def recipes():
        return live.recipes()

    @app.post("/api/recipes/{name}")
    def recipe_save(name: str, units: list[str] = Body(...), ablate: dict | None = Body(None)):
        return session.recipe_save(name, units, ablate=ablate)

    @app.get("/api/evals")
    def evals():
        return session.evals()

    @app.get("/api/evals/{name}")
    def eval_one(name: str):
        return session.eval_load(name)

    @app.post("/api/export")
    def export(units: list[str] = Body(...), name: str = Body(...), verify: bool = Body(False)):
        return live.export(units, name, verify=verify)

    @app.get("/api/export/file/{file}")
    def export_file(file: str):
        return FileResponse(live.export_path(file), media_type="application/octet-stream", filename=file)

    @app.post("/api/jobs")
    def job_start(kind: str = Body(...), params: dict = Body({})):
        return live.jobs.start(kind, params).view()

    @app.get("/api/jobs")
    def job_list():
        return [j.view() for j in live.jobs.list()]

    @app.get("/api/jobs/{job_id}/events")
    def job_events(job_id: str):
        return StreamingResponse(live.job_events(job_id), media_type="text/event-stream", headers=sse_headers)

    @app.delete("/api/jobs/{job_id}")
    def job_cancel(job_id: str):
        return live.jobs.cancel(job_id).view()

    # --- U5: エージェント連携 --------------------------------------------------------------------------------
    @app.post("/api/ui/highlight")
    def ui_highlight(units: list[str] = Body([]), label: str | None = Body(None)):
        return agent.highlight(units, label)

    @app.post("/api/ui/note")
    def ui_note(layer: int = Body(...), neuron: int = Body(...), text: str = Body("")):
        return agent.note(layer, neuron, text)

    @app.post("/api/ui/open")
    def ui_open(layer: int = Body(...), neuron: int = Body(...), tab: str | None = Body(None)):
        return agent.open(layer, neuron, tab)

    @app.post("/api/ui/clear")
    def ui_clear():
        return agent.clear()

    @app.get("/api/ui/state")
    def ui_state():
        return agent.snapshot()

    @app.get("/api/ui/events")
    def ui_events():
        return StreamingResponse(agent.events(), media_type="text/event-stream", headers=sse_headers)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/", StaticFiles(directory=STATIC), name="static")
    return app


def serve(model: str | Path, *, host: str = "127.0.0.1", port: int = 8791, root: str | Path | None = None,
          n_gpu_layers: int = -1, n_ctx: int = 4096, seed: int = 0, openai: bool = False) -> None:
    import uvicorn

    from ..api import Session
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise GneError("BAD_ARGS", "UI はローカル専用です(--host は 127.0.0.1)", "別冊 UI 設計 §1 の非ゴール: マルチユーザー・認証")
    session = Session(model, root=root, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, seed=seed)
    uvicorn.run(create_app(session, openai=openai), host=host, port=port, log_level="warning")
