"""`gne serve`: Web UI のサーバ(FastAPI)。別冊 UI 設計 §8。127.0.0.1 にだけ bind する。

U0: /api/model(ModelMeta + ArchInfo + coverage)、/api/vocab(語彙の文字列)、静的配信。
U1: 層ごとの overlay の値・hover の語・rank の表、ニューロンの詳細、検索、名前、layers ビューの要約。
U2: 3 列チャット(SSE)、fires、reads you、レシピ、cvec の書き出し、ジョブ(SSE の進捗)。GPU は live.Live が調停する。
U5: エージェント連携。MCP の ui_* が POST /api/ui/{highlight,note,open,clear} を叩き、ブラウザへ SSE(/api/ui/events)。
CHANGE-04 G: ヘッドの調書(/api/head)・名前、保存済みのヘッドの対照分析(/api/heads)、注目のパターン(/api/attention)、
fires の文章でのヘッドの注目(/api/fires/{id}/heads)。GPU を使うものは live.Live の調停を通す。
CHANGE-06 G: 因果スクリーニングと強さの配分(/api/causal)、方向セットと速い指標(/api/directions、.../score は GPU)、
方向とヘッドの一括の結果(/api/pipelines。実行は head_pipeline のジョブ)、強さの候補と名前を伏せた採点(/api/candidates。
生成は heads_candidates のジョブ)。
CHANGE-07 G: 鍵つきの書き込み(keyed)の鍵の一覧(/api/keys)、対照集合の候補(/api/contrasts)とひな形(/api/keyed-templates)。
鍵の作成・確認・専門家の選ばれ方・LoRA の確かめ・方向から key-check までの一括は direction_key・direction_key_check・
direction_key_experts・direction_key_verify・direction_key_pipeline のジョブ、keyed のレシピの書き出しは headproj_export のジョブ。
keyed の効く場所を探す 2 段は direction_key_causal・direction_key_causal_verify のジョブで、保存した結果は /api/key-causal。
ローカル専用なので、Host が 127.0.0.1 / localhost 以外の要求(DNS rebinding)と、別のオリジンからの書き込み(POST / DELETE)は断る。
大きい配列は JSON にせず octet-stream で返し、形は X-Shape、付帯情報は X-Meta(JSON)に入れる(§8.2)。
"""

from __future__ import annotations

import json
import re
import sys
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


def _code_stamp() -> float:
    """gne のコード(Python と UI のファイル)の最終更新時刻。サーバを立ち上げた後でコードが更新されたかを見る。"""
    root = Path(__file__).resolve().parent.parent
    return max((p.stat().st_mtime for p in root.rglob("*") if p.suffix in (".py", ".js", ".html", ".css")), default=0.0)


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
    agent.closing = live.closing                   # 止める合図は 1 つ(serve の Ctrl+C で立てる)
    app.state.closing = live.closing
    started_stamp = _code_stamp()

    @app.get("/api/stale")
    def stale():
        """サーバを立ち上げた後で gne のコードが更新されたか。画面のファイルは毎回読み直すので、更新されていると新しい画面と
        古いサーバが食い違う(新しいボタンのジョブを知らない、など)。UI はこれを見て立ち上げ直しを勧める。"""
        now = _code_stamp()
        return {"stale": now > started_stamp + 1.0, "started": started_stamp, "now": now}

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
        body = {"error": {"code": e.code, "message": e.message, "hint": e.hint}}
        if getattr(e, "estimate", None) is not None:                 # NEEDS_CONFIRM(CHANGE-03 C)
            body["estimate"] = e.estimate
        return JSONResponse(body, status_code=400)

    @app.get("/api/model")
    def model():
        if "arch" not in cache:
            cache["arch"] = arch_info(session.weights).to_dict()
        ov = session.model_overview()
        return {"meta": ov["meta"], "arch": cache["arch"], "coverage": ov["layers"], "speed": ov.get("speed"),
                "arch_warnings": ov["arch_warnings"], "bake": ov["bake"], "recipes": ov["recipes"],
                "directions": ov["directions"], "names": data.names(), "layouts": data.layouts(), "runs": data.run_list(),
                "connections": data.connections().meta if data.connections() is not None else None,
                "heads": data.heads_list(), "writers": data.writers_list(),
                "causal": data.causal_list(), "direction_sets": data.directions_list(),       # CHANGE-06 G
                "keys": data.keys_list()}                                                     # CHANGE-07 G: 鍵つきの書き込みの鍵

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

    # --- CHANGE-04 G: attention ヘッド(HEAD パネル、地図の heads:<name>) ---------------------------------------
    @app.get("/api/head")
    def head(layer: int, head: int, lens: bool = True):
        return data.head(layer, head, lens=lens)

    @app.post("/api/head/name")
    def head_name(layer: int = Body(...), head: int = Body(...), name: str = Body("")):
        data._check(layer)
        session.set_head_name(layer, head, name.strip())
        return {"layer": layer, "head": head, "name": name.strip() or None, "names": data.names()}

    @app.get("/api/heads")
    def heads():
        return data.heads_list()

    @app.get("/api/heads/{name}")
    def heads_one(name: str):
        return data.heads_one(name)

    # --- CHANGE-06 G: 因果スクリーニング・強さの配分(HEAD パネルの Causal、地図の causal:<name>・eta:<name>)、
    # 方向の速い指標(direction ビュー) --------------------------------------------------------------------------
    @app.get("/api/causal")
    def causal_list():
        return data.causal_list()

    @app.get("/api/causal/{name}")
    def causal_one(name: str):
        return data.causal_one(name)

    @app.get("/api/directions")
    def directions():
        return data.directions_list()

    @app.get("/api/directions/{name}")
    def direction_one(name: str):
        return data.direction_one(name)

    # --- CHANGE-07 G: 鍵つきの書き込み(direction ビューの「鍵」)。作成と確認はジョブ(direction_key・direction_key_check) -----
    @app.get("/api/keys")
    def keys():
        """保存した鍵の一覧(<root>/keys/<model_id>/、パスなし)。"""
        return data.keys_list()

    @app.get("/api/keys/{name}")
    def key_one(name: str):
        return data.key_one(name)

    # keyed の効く場所(鍵のタブの「効く層を探す」)。回すのはジョブ direction_key_causal・direction_key_causal_verify
    @app.get("/api/key-causal")
    def key_causal_list():
        """保存した key-causal の結果の要約(新しい順)。"""
        return data.key_causal_list()

    @app.get("/api/key-causal/{name}")
    def key_causal_one(name: str):
        """key-causal の第 1 段の結果(候補 × α の行)と、あれば第 2 段の結果(返事は除く)。"""
        return data.key_causal_one(name)

    # CHiPS で残った場面から keyed の JSON を作る(鍵のタブ)。behave を回すのはジョブ behave
    @app.get("/api/behave")
    def behave_list():
        """保存した behave の結果の要約(新しい順、返事は返さない)。"""
        return data.behave_list()

    @app.get("/api/chips-source")
    def chips_source(direction: str | None = None, contrast: str | None = None):
        """CHiPS の方向と対照集合の要約(A 側・B 側の質問の数、validate.match、positions の既定)。"""
        return data.chips_source(direction or None, contrast or None)

    @app.post("/api/keyed-from-behave")
    def keyed_from_behave(behave: str = Body(...), recipe: str = Body(...), direction: str | None = Body(None),
                          contrast: str | None = Body(None), labels: list[str] | str | None = Body(None),
                          match: str | None = Body(None), min_rate: float | None = Body(None), positions: str | None = Body(None),
                          name: str | None = Body(None), replies: bool = Body(True), strict: bool = Body(False)):
        """behave の結果から、CHiPS のレシピを当てても振る舞いが残った質問を集めて keyed の対照集合の JSON を <root>/keyed/ に
        書く(gne direction key-from-behave と同じ。モデルは動かさない)。結果に、書いた JSON の中身 spec を添える。"""
        for nm, v in (("behave", behave), ("recipe", recipe), ("name", name)):
            if v and not re.fullmatch(r"[\w.\-]+", v):
                raise GneError("BAD_ARGS", f"{nm} は名前で渡してください({v!r})")
        d = session.direction_key_from_behave(behave, recipe=recipe, direction=direction or None, contrast=contrast or None,
                                              labels=labels or None, match=match or None, min_rate=min_rate,
                                              positions=positions or None, name=name or None, replies=replies, strict=strict)
        spec = json.loads(Path(d["path"]).read_text(encoding="utf-8")) if d.get("path") else None
        return {**d, "spec": spec}

    @app.get("/api/contrasts")
    def contrasts():
        """鍵を作る対照集合の候補(configs/contrast/ と <root>/contrasts/ の YAML の要約)。"""
        return data.contrasts_list()

    @app.get("/api/fieldtests")
    def fieldtests():
        """実地テスト(fieldtest のジョブ)の保存済みの結果。生成した文章は返さない。"""
        return data.fieldtests_list()

    @app.get("/api/fieldtests/{name}")
    def fieldtest_one(name: str):
        return data.fieldtest_one(name)

    @app.get("/api/pipeline-templates")
    def pipeline_templates():
        """用途別の質問集のテンプレート(configs/pipeline/、「ひな形」で選ぶ)。"""
        return data.pipeline_templates()

    @app.get("/api/keyed-templates")
    def keyed_templates():
        """鍵つきの書き込みの対照集合のひな形(configs/keyed/ と、自分で作った <root>/keyed/。鍵のタブの「ひな形」で選ぶ)。"""
        return data.keyed_templates_all()

    @app.get("/api/pipelines")
    def pipelines():
        """方向とヘッドの一括(head_pipeline のジョブ)の保存済みの結果。"""
        return data.pipelines_list()

    @app.get("/api/pipelines/{name}")
    def pipeline_one(name: str):
        return data.pipeline_one(name)

    @app.get("/api/candidates")
    def candidates():
        """強さの候補(heads_candidates のジョブ)の保存済みの結果。"""
        return data.candidates_list()

    @app.get("/api/candidates/{name}")
    def candidates_one(name: str):
        return data.candidates_one(name)

    @app.get("/api/candidates/{name}/blind")
    def candidates_blind(name: str):
        """名前を伏せた採点の項目。どの候補の返事かは返さない。"""
        return data.candidates_blind(name)

    @app.post("/api/candidates/{name}/scores")
    def candidates_scores(name: str, scores: dict | str = Body(..., embed=True)):
        """点を足して集計した結果(Session.heads_candidates_tally)。"""
        return data.candidates_score(name, scores)

    @app.post("/api/directions/{name}/score")
    def direction_score(name: str, set_base: int | None = Body(None), record_base: bool | None = Body(None),
                        layers: list[int] | None = Body(None)):
        """gne direction score(Session.direction_score)。モデルを流すので GPU の調停を通す。set_base で基準層を手で上書きする。
        層を省くと全層を測り、基準層がまだ無ければ cohen_d が最大の層を記録する(record_base=False で記録しない)。"""
        data.direction_one(name)                                 # 名前を確かめる(無ければ 404 相当)
        r, warns = live._gpu(lambda: session.direction_score(name, layers=layers, set_base=set_base, record_base=record_base))
        return {**r, "warnings": warns, "meta": data.direction_one(name)}

    # --- CHANGE-05 G: MoE の専門家(EXPERT パネル、地図の expert_usage・experts:<name>) -----------------------
    @app.get("/api/expert")
    def expert(layer: int, expert: int):
        return session.expert(layer, expert)

    @app.get("/api/experts")
    def experts():
        from ..experts import list_expert_contrasts
        return {"contrasts": list_expert_contrasts(session)}

    @app.get("/api/experts/{name}")
    def experts_one(name: str):
        from ..experts import load_expert_candidates
        return {"name": name, "rows": load_expert_candidates(session, name)}

    @app.post("/api/attention")
    def attention(text: str = Body(...), layer: int | None = Body(None), head: int | None = Body(None), pos: int = Body(-1),
                  top: int = Body(5), system: str | None = Body(None), chat: bool = Body(False), compare: bool = Body(False)):
        return live.attention(text, layer=layer, head=head, pos=pos, top=top, system=system, chat=chat, compare=compare)

    @app.post("/api/listen")
    def listen(user: str = Body(...), heads: str = Body(...), system: str | None = Body(None), reply: str | None = Body(None),
               max_tokens: int = Body(64), direction: str | None = Body(None), top: int = Body(20)):
        return live.listen(user, heads=heads, system=system, reply=reply, max_tokens=max_tokens, direction=direction, top=top)

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

    @app.get("/api/fires/{fid}/heads")
    def fires_heads(fid: str):
        body, shape, meta = live.fires_heads(fid)
        return _binary(body, ",".join(map(str, shape)), meta)

    @app.get("/api/fires/{fid}/neurons")
    def fires_neurons(fid: str, tok: int = -1, lmin: int = 0, lmax: int | None = None, sign: str = "both",
                      thr: float = 0.0, q: str = "", limit: int = 100):
        return live.fires_neurons(fid, tok, lmin=lmin, lmax=lmax, sign=sign, thr=thr, q=q, limit=limit)

    @app.post("/api/reads")
    def reads(layer: int = Body(...), neuron: int = Body(...), text: str = Body(...)):
        return live.reads(layer, neuron, text)

    @app.get("/api/recipes")
    def recipes():
        out = []
        for x in live.recipes():                     # CHANGE-06: proj のユニットの方向の既定と、方向の層(own / base)
            ex = data.recipe_extras(x["name"])
            for u in x["units"]:
                if u.get("mode") == "proj" and not u.get("direction"):
                    u["direction"] = ex["default_direction"]
                if u.get("mode") == "keyed":         # CHANGE-07: 省いた方向・鍵はレシピの既定で埋める(埋めたことは *_default で)
                    if not u.get("direction") and ex["default_direction"]:
                        u["direction"], u["direction_default"] = ex["default_direction"], True
                    if not u.get("key") and ex["default_key"]:
                        u["key"], u["key_default"] = ex["default_key"], True
            out.append({**x, **ex})
        return out

    @app.post("/api/recipes/{name}")
    def recipe_save(name: str, units: list[str] = Body(...), ablate: dict | None = Body(None),
                    sampling: dict | None = Body(None), direction_layer: str | None = Body(None),
                    default_direction: str | None = Body(None), default_key: str | None = Body(None)):
        """direction_layer は proj・keyed に当てる方向の層(CHANGE-06・07、own なら書かない)。default_direction・default_key は
        @方向・#鍵を省いた proj・keyed のユニットの既定(CHANGE-07)。"""
        return session.recipe_save(name, units, ablate=ablate, sampling=sampling, default_direction=default_direction or None,
                                   direction_layer=direction_layer if direction_layer and direction_layer != "own" else None,
                                   default_key=default_key or None)

    @app.get("/api/evals")
    def evals():
        return session.evals()

    @app.get("/api/evals/{name}")
    def eval_one(name: str):
        return session.eval_load(name)

    @app.post("/api/export")
    def export(units: list[str] = Body(...), name: str = Body(...), verify: bool = Body(False), keyed: int = Body(0)):
        """control vector の書き出し。keyed はレシピから外した鍵つきの書き込みの数(CHANGE-07、cvec に入らないので警告に出す)。"""
        return live.export(units, name, verify=verify, keyed=keyed)

    @app.get("/api/fish-templates")
    def fish_templates():
        """FiSH のタブの「ひな形」(configs/fish/*.json)。"""
        return data.fish_templates()

    @app.post("/api/fish/units")
    def fish_units(layers: str = Body(...), segments: str = Body(...), neurons: dict = Body(...), form: str = Body("cap"),
                   direction: str = Body(...), base: str = Body(...), alpha: float = Body(0.9)):
        """FiSH のユニットを組む(層 × 区切りごとに 1 個、上限つきの形は 2 個組)。モデルは使わない。specs は書き方の文字列、
        units は UI のレシピの行、needed は層 1 つに要るニューロンの数。"""
        from .. import fish
        segs = fish._segments(segments)
        specs = fish.units(layers, len(segs), {int(k): v for k, v in neurons.items()}, form, direction, base, alpha,
                           fish.before_mode(segs))
        return {"specs": specs, "units": fish.unit_rows(session, specs), "needed": fish.needed(len(segs), form),
                "names": fish.key_names(base, len(segs), form)}

    @app.post("/api/nw/export")
    def nw_export(units: list[str] = Body(...), name: str = Body(...)):
        """計算で作るニューロン(mode: nw)の LoRA の書き出し(焼き込みは無い)。"""
        return live.nw_export(units, name)

    @app.get("/api/export/file/{file}")
    def export_file(file: str):
        return FileResponse(live.export_path(file), media_type="application/octet-stream", filename=file)

    @app.post("/api/jobs")
    def job_start(kind: str = Body(...), params: dict = Body({})):
        return live.jobs.start(kind, params).view()

    @app.get("/api/jobs")
    def job_list():
        return [j.view() for j in live.jobs.list()]

    @app.get("/api/writers")
    def writers_list():
        return data.writers_list()

    @app.get("/api/writers/{name}")
    def writers_one(name: str):
        return data.writers_summary(name)

    @app.get("/api/workers")
    def workers():
        """CHANGE-03 B: 生きているワーカーと、ファイルのキューのジョブ(結果は省く)。"""
        from ..workers import job_files, list_workers
        root = session.store.root
        return {"workers": list_workers(root), "jobs": [{k: v for k, v in j.items() if k != "result"}
                                                         for j in job_files(root)][-20:]}

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
          n_gpu_layers: int = -1, n_ctx: int = 4096, seed: int = 0, openai: bool = False,
          mmproj: str | Path | None = None) -> None:
    import uvicorn

    from ..api import Session
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise GneError("BAD_ARGS", "UI はローカル専用です(--host は 127.0.0.1)", "別冊 UI 設計 §1 の非ゴール: マルチユーザー・認証")
    session = Session(model, root=root, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, seed=seed, mmproj=mmproj)
    app = create_app(session, openai=openai)
    server = _Server(uvicorn.Config(app, host=host, port=port, log_level="warning"), app.state.closing)
    try:
        server.run()
    except KeyboardInterrupt:                      # uvicorn.run と同じく、止めた後に送り直される Ctrl+C は黙って終わる
        pass
    if server.started:
        sys.stderr.write("gne serve: 止めました\n")


def _server_class():
    import uvicorn

    class Server(uvicorn.Server):
        """Ctrl+C で止まる uvicorn。ブラウザが開いたままにする配信(SSE)は終わらないので、そのままだとサーバが接続の
        終わりを待ち続ける(Python 3.12 の asyncio は、強制終了でも残った接続が閉じるまで待つ)。1 回目で止める合図を立てて
        配信を閉じさせ、2 回目(か 5 秒たっても残っていれば)残った接続を切る。3 回目はすぐ終わる。"""

        def __init__(self, config, closing):
            super().__init__(config)
            self.closing = closing
            self.loop = None
            self.n_signals = 0

        async def startup(self, sockets=None):
            import asyncio
            self.loop = asyncio.get_running_loop()
            await super().startup(sockets)

        def handle_exit(self, sig, frame):
            import os
            import sys
            self.n_signals += 1
            self.closing.set()
            if self.n_signals == 1:
                sys.stderr.write("gne serve: 止めています(開いている配信を閉じています。もう一度 Ctrl+C で接続を切ります)\n")
                if self.loop is not None:
                    self.loop.call_soon_threadsafe(self.loop.call_later, 5.0, self.drop_connections)
            elif self.n_signals == 2:
                sys.stderr.write("gne serve: 残った接続を切って止めます\n")
                if self.loop is not None:
                    self.loop.call_soon_threadsafe(self.drop_connections)
            else:
                os._exit(130)
            super().handle_exit(sig, frame)

        def drop_connections(self):
            self.force_exit = True
            for c in list(self.server_state.connections):
                t = getattr(c, "transport", None)
                if t is not None and not t.is_closing():
                    t.close()

    return Server


def _Server(config, closing):
    return _server_class()(config, closing)
