"""MCP サーバ `gne mcp`(stdio)。設計書 §6.13 (d)。ツールは gne.api の写像。

    claude mcp add gne -- gne mcp --model <path>

- 戻り値は --json と同じ形の dict(トークン列は文字列)。失敗は {"error": {"code", "message", "hint"}}。
- GPU を使う呼び出しは worker スレッドで GpuArbiter を通して実行する。長い処理は job_start で投げ、
  job_status で進捗を見る。ジョブはユニットの切れ目で対話的な呼び出しに GPU を譲る。
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import anyio

from .jobs import GpuArbiter, JobRunner, standard_kinds
from .steer import GneError
from .store import CONFIGS

INSTRUCTIONS = """\
gguf-neuron-explorer: llama.cpp 上の GGUF モデルの MLP ニューロンを調べ、押し(steer)、レシピを control vector に書き出す。
進め方(詳しくは docs/agent_playbook.md):
1. model_overview → layer_summary(L) で、どの層にどのデータ(ρ・lens・sweep・逆引き)があるかを見る。無いものは job_start で作る。
2. 候補は neuron(L, n) の調書で判断する。null_pct(sweep が対照群の何割より上か)と kl_null_p95(steer lens の対照の 95% 点)を必ず見る。対照群を超えないものは報告しない。
3. 「名乗り・作者名が変わる」はランダム方向でも起こる安い効果。複数のプロンプトで同じテーマが出るものだけを本物とみなす。
4. rel dose: 0.5 で微かな挿入、0.8 で話題を上書き、1.0 以上は崩壊域。層 0〜2 には rel dose がそのまま通用しない。
5. steer(units, compare=True) で無介入との差を確かめてから recipe_save → export_cvec。
ユニットの書き方: "L:N:dose"(例 "14:8949:+1.0")、ランダム方向は "L:random:SEED:dose"。
"""


def _err(e: Exception) -> dict:
    if isinstance(e, GneError):
        return {"error": {"code": e.code, "message": e.message, "hint": e.hint}}
    return {"error": {"code": "INTERNAL", "message": f"{type(e).__name__}: {e}", "hint": None}}


def _strip_ids(d: Any) -> Any:
    """token_ids を落とす(トークン列は文字列で返す、§6.13 (d))。"""
    if isinstance(d, dict):
        return {k: _strip_ids(v) for k, v in d.items() if k != "token_ids"}
    if isinstance(d, list):
        return [_strip_ids(x) for x in d]
    return d


def build_server(model: str | Path, *, root: str | Path | None = None, ui_url: str | None = None,
                 n_gpu_layers: int = -1, n_ctx: int = 4096, seed: int = 0):
    from mcp.server.mcpserver import MCPServer

    from .api import Session
    session = Session(model, root=root, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, seed=seed)
    arbiter = GpuArbiter()
    jobs = JobRunner(session, arbiter, standard_kinds())
    srv = MCPServer("gne", instructions=INSTRUCTIONS, version="0.1.0", log_level="WARNING")

    async def call(fn, *a, gpu: bool = True, **kw) -> dict:
        def run():
            session.warnings = []
            try:
                if gpu:
                    with arbiter.interactive():
                        out = fn(*a, **kw)
                else:
                    out = fn(*a, **kw)
                out = _strip_ids(out)
                if session.warnings:
                    out = {**out, "warnings": list(session.warnings)}
                return out
            except Exception as e:  # noqa: BLE001
                return _err(e)
        return await anyio.to_thread.run_sync(run)

    @srv.tool()
    async def model_overview() -> dict:
        """ModelMeta、arch の警告、bake の可否、層ごとの coverage(ρ_L・活性統計・logit lens・steer lens・sweep・逆引き)。"""
        return await call(session.model_overview, gpu=False)

    @srv.tool()
    async def layer_summary(layer: int, top: int = 20) -> dict:
        """層の要約: ρ_L、steer lens の KL(中央値・p90・対照の 95% 点を超える割合)、sweep の対照群超えの割合と上位ユニット、logit lens の上位、逆引きの上位。"""
        return await call(session.layer_summary, layer, top=top, gpu=False)

    @srv.tool()
    async def neuron(layer: int, neuron: int, rollouts: int = 4, compute: bool = False) -> dict:
        """ニューロン 1 個の調書(sweep の順位・語・rollout、steer lens、logit lens、逆引きの cos、活性統計、coverage)。
        compute=True なら、無い logit lens と steer lens をこのニューロンだけ計算する(数秒)。"""
        return await call(session.neuron, layer, neuron, rollouts=rollouts, compute=compute, gpu=compute)

    @srv.tool()
    async def search(q: str, mode: str = "lens", layer: int | None = None, limit: int = 50) -> dict:
        """一致するニューロンの一覧。mode: lens(steer/logit lens の上位語)、words(sweep の語)、name(付けた名前)。"""
        return await call(session.search, q, mode=mode, layer=layer, limit=limit, gpu=False)

    @srv.tool()
    async def steer(units: list[str], prompt: str | None = None, messages: list[dict[str, str]] | None = None,
                    max_tokens: int = 128, compare: bool = True, dose_unit: str = "rel",
                    system: str | None = None) -> dict:
        """ユニット(例 ["14:8949:+1.0", "13:7956:-1.0"])で押して生成する(greedy)。compare=True なら無介入も並べる。"""
        def fn():
            r = session.recipe(units=units, dose_unit=dose_unit)
            return session.steer(r, prompt=prompt, messages=messages, system=system, max_tokens=max_tokens,
                                 compare=compare)
        return await call(fn)

    @srv.tool()
    async def steer_lens(layer: int, neurons: str, dose: float = 0.8, random_controls: int = 16) -> dict:
        """指定ニューロン(例 "2742" や "1,5,9-12")の steer lens(± で押したときの次トークンの変化、KL と対照)。少数向け。"""
        return await call(session.lens_steer, layer, neurons=neurons, dose=dose, random_controls=random_controls)

    @srv.tool()
    async def fires(text: str, top: int = 50) -> dict:
        """文章を流し、トークンごとに最も光る(活性 ÷ 基準 が大きい)ニューロンを top 件ずつ返す。"""
        return await call(session.fires, text, top=top)

    @srv.tool()
    async def reverse(contrast: str | None = None, system_a: str | None = None, system_b: str | None = None,
                      user_prompts: list[str] | None = None, name: str | None = None, top: int = 50,
                      method: str = "mean_diff", positions: str = "all") -> dict:
        """逆引き: 対照プロンプト集(configs/contrast/ の名前かパス、または system_a / system_b / user_prompts を直接)から
        方向を作り、全層・全ニューロンを順位づけする。検証(rollout)は job_start("reverse", {..., "verify": true})。"""
        def fn():
            path = _contrast_path(session, contrast, system_a, system_b, user_prompts, name)
            return session.reverse(contrast=path, top=top, method=method, positions=positions, name=name)
        return await call(fn)

    @srv.tool()
    async def recipe_save(name: str, units: list[str], dose_unit: str = "rel") -> dict:
        """レシピを <root>/recipes/<name>.yaml に保存する。"""
        return await call(session.recipe_save, name, units, dose_unit=dose_unit, gpu=False)

    @srv.tool()
    async def export_cvec(name: str, format: str = "cvec", verify: bool = False) -> dict:
        """保存したレシピを control vector GGUF(または npz / json / safetensors)に書き出す。verify=True で llama-completion と比べる。"""
        def fn():
            from .cli import _EXT
            if format not in _EXT:
                raise GneError("BAD_FORMAT", f"format は {', '.join(_EXT)} のどれかです")
            r = session.recipe(path=session.recipe_path(name))
            out = session.store.export_dir() / f"{name}{_EXT[format]}"
            return session.export_cvec(r, out, fmt=format, verify=verify)
        return await call(fn, gpu=True)

    @srv.tool()
    async def bake(name: str, dst: str, verify: bool = True) -> dict:
        """保存したレシピを FFN バイアスとして焼き付けた GGUF を dst に作る(対応 arch のみ、Qwen 系は不可)。"""
        def fn():
            r = session.recipe(path=session.recipe_path(name))
            return session.bake(r, dst, verify=verify)
        return await call(fn)

    @srv.tool()
    async def set_name(layer: int, neuron: int, name: str) -> dict:
        """ニューロンに名前を付ける(空文字で消す)。調書と search(mode="name") に出る。"""
        return await call(session.set_name, layer, neuron, name, gpu=False)

    @srv.tool()
    async def job_start(kind: str, params: dict[str, Any] | None = None) -> dict:
        """長時間の処理を投げる。kind: stats_resid / stats_act / lens_logit / lens_steer / sweep / rank / reverse。
        params は gne.api の同名メソッドの引数(例 sweep: {"layer": 22, "neurons": "random:64", "design": "quick"})。"""
        try:
            return jobs.start(kind, params).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def job_status(job_id: str | None = None) -> dict:
        """ジョブの進捗(done / total / eta_s)と結果。job_id を省くと全ジョブの一覧。"""
        try:
            if job_id is None:
                return {"jobs": [j.view() for j in jobs.jobs.values()]}
            return _strip_ids(jobs.get(job_id).view())
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def job_cancel(job_id: str) -> dict:
        """ジョブを止める(次のユニットの切れ目で止まる。途中までの結果は残り、同じ条件で続きから再開できる)。"""
        try:
            return jobs.cancel(job_id).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    for action in ("highlight", "note", "open"):
        _register_ui_tool(srv, action, ui_url)
    return srv


def _contrast_path(session, contrast, system_a, system_b, user_prompts, name) -> Path:
    if contrast:
        p = Path(contrast)
        if not p.exists():
            p = CONFIGS / "contrast" / f"{contrast}.yaml"
        if not p.exists():
            raise GneError("FILE_NOT_FOUND", f"対照プロンプト集 {contrast} がありません",
                           "configs/contrast/ の名前かパス、または system_a / system_b / user_prompts を渡してください")
        return p
    if not (system_a and system_b and user_prompts):
        raise GneError("BAD_ARGS", "contrast か、system_a・system_b・user_prompts の 3 つを渡してください")
    import yaml
    nm = name or "contrast"
    if not re.fullmatch(r"[\w.\-]+", nm):
        raise GneError("BAD_ARGS", "name は英数字・_・-・. だけにしてください")
    p = session.store.root / "contrast" / f"{nm}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump({"name": nm, "system_a": system_a, "system_b": system_b,
                                 "user_prompts": list(user_prompts)}, allow_unicode=True), encoding="utf-8")
    return p


def _register_ui_tool(srv, action: str, ui_url: str | None) -> None:
    docs = {"highlight": "UI のマップでユニットを強調する(UI が起動しているとき)。",
            "note": "UI のマップのニューロンに短い注釈を貼る(UI が起動しているとき)。",
            "open": "UI の NEURON パネルでニューロンを開く(UI が起動しているとき)。"}

    async def tool(layer: int | None = None, neuron: int | None = None, text: str | None = None,
                   units: list[str] | None = None) -> dict:
        if not ui_url:
            return {"error": {"code": "UI_NOT_RUNNING", "message": "UI に接続していません",
                              "hint": "gne serve を起動し、gne mcp --ui-url http://127.0.0.1:8791 で繋いでください"}}
        body = json.dumps({"layer": layer, "neuron": neuron, "text": text, "units": units}).encode()
        req = urllib.request.Request(f"{ui_url.rstrip('/')}/api/agent/{action}", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            def post():
                with urllib.request.urlopen(req, timeout=3) as r:     # noqa: S310  127.0.0.1 の UI だけ
                    return json.loads(r.read() or b"{}")
            return await anyio.to_thread.run_sync(post)
        except (urllib.error.URLError, OSError) as e:
            return {"error": {"code": "UI_NOT_RUNNING", "message": f"UI に届きません: {e}", "hint": None}}

    tool.__name__ = f"ui_{action}"
    tool.__doc__ = docs[action]
    srv.tool(name=f"ui_{action}")(tool)


def main(model: str, *, root: str | None = None, ui_url: str | None = None, n_gpu_layers: int = -1,
         n_ctx: int = 4096, seed: int = 0) -> None:
    srv = build_server(model, root=root, ui_url=ui_url, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, seed=seed)
    srv.run("stdio")
