"""OpenAI 互換の /v1/chat/completions(本体設計 §6.15、`gne serve --openai`)。

位置別ステアリング(when)は control vector の GGUF では表せないので、ほかのアプリから使えるように
UI サーバに最小限の OpenAI 互換 API を同居させる。リクエストの `gne`(OpenAI の SDK では extra_body={"gne": {...}})で
レシピと when を渡す:

    {"model": "...", "messages": [...], "max_tokens": 256, "stream": true,
     "gne": {"recipe": "evil" | {"units": ["14:8949:+1.0"], "dose_unit": "rel", "when": "gen"},
             "units": ["14:8949:+1.0"], "scale": 0.8, "when": "gen"}}

生成は greedy だけ(temperature などは無視する)。tools・logprobs・n は扱わない。GPU は UI のチャットと同じ調停を通る。
"""

from __future__ import annotations

import codecs
import json
import queue
import threading
import time
import uuid

from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from ..steer import GneError, Recipe

MAX_TOKENS_DEFAULT = 512


def _text(content) -> str:
    """content は文字列か、[{type: "text", text: ...}, ...]。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in (None, "text"))
    return "" if content is None else str(content)


def _messages(body: dict) -> list[dict]:
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise GneError("BAD_ARGS", "messages がありません")
    out = []
    for m in msgs:
        role = m.get("role")
        if role == "developer":
            role = "system"
        if role not in ("system", "user", "assistant"):
            raise GneError("BAD_ARGS", f"role={role!r} は扱えません(system / user / assistant)")
        out.append({"role": role, "content": _text(m.get("content"))})
    return out


def resolve_recipe(session, g: dict) -> tuple[Recipe | None, str]:
    """body["gne"] からレシピと when を決める。レシピが無ければ (None, "all")。"""
    rec, units = g.get("recipe"), g.get("units")
    r = None
    if isinstance(rec, str) and rec:
        r = session.recipe(path=session.recipe_path(rec))
    elif isinstance(rec, dict):
        us = rec.get("units") or []
        specs = [u if isinstance(u, str) else f"{u['layer']}:{u['neuron']}:{float(u['dose']):+g}" for u in us]
        r = session.recipe(units=specs, dose_unit=rec.get("dose_unit"), name=rec.get("name"), when=rec.get("when"))
    elif units:
        r = session.recipe(units=list(units))
    elif rec is not None:
        raise GneError("BAD_ARGS", "gne.recipe はレシピ名か {units: [...]} です")
    if r is not None and g.get("scale") not in (None, 1, 1.0):
        r = r.scaled(float(g["scale"]))
    when = g.get("when") or (r.when if r is not None else "all")
    return r, when


def _err(e: GneError, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": {"message": e.message + (f"({e.hint})" if e.hint else ""), "type": "invalid_request_error",
                                   "code": e.code}}, status_code=status)


def register(app: FastAPI, session, live) -> None:
    model_id = session.meta.model_id

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "gne"}]}

    @app.post("/v1/chat/completions")
    def completions(body: dict = Body(...)):
        try:
            msgs = _messages(body)
            recipe, when = resolve_recipe(session, body.get("gne") or {})
            from ..when import parse_when
            w = parse_when(when)
            max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or MAX_TOKENS_DEFAULT)
            if not 1 <= max_tokens <= 4096:
                raise GneError("BAD_ARGS", "max_tokens は 1〜4096 です")
        except GneError as e:
            return _err(e)
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
        q: queue.Queue = queue.Queue()
        stop = threading.Event()

        def work():
            from ..when import generate_when
            try:
                with live.arbiter.interactive():
                    e = session.engine
                    vecs = session.vectors(recipe)[0] if recipe is not None else None
                    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

                    def cb(_si, tok):
                        if stop.is_set():
                            raise _Stopped()
                        piece = dec.decode(e.token_bytes(tok))
                        if piece:
                            q.put(("delta", piece))
                    try:
                        r, info = generate_when(e, e.steer_ctx(), msgs, vecs, w, max_tokens=max_tokens, on_token=cb)
                    except _Stopped:
                        q.put(("end", None))
                        return
                    tail = dec.decode(b"", final=True)
                    if tail:
                        q.put(("delta", tail))
                    q.put(("end", {"finish": "stop" if r.finish == "eog" else "length", "text": r.text,
                                   "usage": {"prompt_tokens": info["n_prompt"], "completion_tokens": r.n_gen,
                                             "total_tokens": info["n_prompt"] + r.n_gen},
                                   "gne": {"recipe": recipe.to_dict() if recipe else None, "when": info}}))
            except GneError as ex:
                q.put(("error", ex))
            except Exception as ex:  # noqa: BLE001
                q.put(("error", GneError("INTERNAL", f"{type(ex).__name__}: {ex}")))

        th = threading.Thread(target=work, name=f"gne-openai-{cid}", daemon=True)
        th.start()
        if not body.get("stream"):
            while True:
                kind, x = q.get()
                if kind == "error":
                    return _err(x, 500 if x.code == "INTERNAL" else 400)
                if kind == "end":
                    break
            return {"id": cid, "object": "chat.completion", "created": created, "model": model_id,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": x["text"]},
                                 "finish_reason": x["finish"]}],
                    "usage": x["usage"], "gne": x["gne"]}

        def chunk(delta: dict, finish=None) -> str:
            d = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            return f"data: {json.dumps(d, ensure_ascii=False)}\n\n"

        def events():
            yield chunk({"role": "assistant", "content": ""})
            try:
                while True:
                    kind, x = q.get()
                    if kind == "delta":
                        yield chunk({"content": x})
                    elif kind == "error":
                        yield f"data: {json.dumps({'error': {'message': x.message, 'code': x.code}}, ensure_ascii=False)}\n\n"
                        break
                    else:
                        yield chunk({}, finish=(x or {}).get("finish", "stop"))
                        break
                yield "data: [DONE]\n\n"
            finally:
                stop.set()                         # 接続が切れたら生成も止める
        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})


class _Stopped(Exception):
    pass
