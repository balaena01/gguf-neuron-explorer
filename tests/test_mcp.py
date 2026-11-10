"""Phase 4: MCP サーバ `gne mcp` を stdio でつなぎ、受け入れの一連(model_overview → layer_summary → neuron →
steer(compare=True) → recipe_save → export_cvec)と、ジョブの最中の割り込み・エラーの形を確かめる。"""

import json
import os
import sys
from pathlib import Path

import anyio
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
GNE = str(Path(sys.executable).parent / "gne")

pytestmark = [pytest.mark.model, pytest.mark.skipif(not MODEL.exists(), reason=f"{MODEL} がありません")]


def _payload(r):
    sc = getattr(r, "structured_content", None)
    if sc:
        return sc.get("result", sc) if isinstance(sc, dict) else sc
    return json.loads(r.content[0].text)


def test_mcp_acceptance_sequence(tmp_path):
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    async def main():
        params = StdioServerParameters(command=GNE, args=["mcp", "--model", str(MODEL), "--root", str(tmp_path)],
                                       env=dict(os.environ))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as s:
                init = await s.initialize()
                assert init.server_info.name == "gne" and "null_pct" in (init.instructions or "")
                names = {t.name for t in (await s.list_tools()).tools}
                assert {"model_overview", "layer_summary", "neuron", "search", "steer", "steer_lens", "fires",
                        "reverse", "recipe_save", "export_cvec", "bake", "job_start", "job_status",
                        "ui_highlight", "ui_note", "ui_open",
                        "expert", "experts_contrast", "experts_sweep", "moe_route", "expert_patch"} <= names   # CHANGE-05

                async def call(tool, **args):
                    return _payload(await s.call_tool(tool, args))

                ov = await call("model_overview")
                assert len(ov["layers"]) == ov["meta"]["n_layer"]
                ls = await call("layer_summary", layer=14)
                assert ls["layer"] == 14
                d = await call("neuron", layer=14, neuron=7)
                assert d["coverage"]["sweep"] is False and "token_ids" not in json.dumps(d)
                st = await call("steer", units=["14:7:+0.8"], prompt="Who are you?", max_tokens=16, compare=True)
                assert st["baseline"]["text"] and st["steered"]["text"] and "token_ids" not in st["steered"]
                assert any("stats resid" in w for w in st.get("warnings", []))        # ρ を自動で作った
                rs = await call("recipe_save", name="t", units=["14:7:+0.8"])
                assert Path(rs["path"]).exists()
                ex = await call("export_cvec", name="t")
                assert Path(ex["path"]).exists() and list(ex["layer_norms"]) == ["14"]

                job = await call("job_start", kind="sweep", params={"layer": 14, "neurons": "1,2", "design": "quick",
                                                                    "random_controls": 1, "max_tokens": 8, "run": "j"})
                inter = await call("steer", units=["14:1:+0.5"], prompt="Hi", max_tokens=4, compare=False)
                assert "error" not in inter
                for _ in range(200):
                    js = await call("job_status", job_id=job["id"])
                    if js["status"] not in ("queued", "running"):
                        break
                    await anyio.sleep(0.2)
                assert js["status"] == "done" and js["result"]["n_new_records"] == 3 + 3 * 3

                err = await call("export_cvec", name="no-such-recipe")
                assert err["error"]["code"] == "FILE_NOT_FOUND"
                ui = await call("ui_open", layer=1, neuron=1)
                assert ui["error"]["code"] == "UI_NOT_RUNNING"

    anyio.run(main)


def test_mcp_contrast_from_systems_keeps_c1_and_line_separators(tmp_path, monkeypatch):
    """MCP の reverse に system_a・system_b・user_prompts を直接渡すと <root>/contrast/ に対照集合を書く。C1 制御文字(MMLU の
    問題文によくある U+0080〜U+009F)・U+0085・U+2028・絵文字を含んでも、書いた対照集合を読むと文字が変わらない
    (yaml.safe_dump で書くと U+0085 が空白になった)。heads_pipeline の表はジョブ head_pipeline に渡る(一括の書き方は
    tests/test_change06.py)。Session は見立てたもので、モデルは読まない。"""
    from types import SimpleNamespace

    import gne.api
    import gne.mcp_server as ms
    from gne.reverse import ContrastSet
    qs = ["\x93善と悪\x94についてどう思う？", "ありがとう\x85助かった。", "それは\u2028本当？", "誕生日🎂おめでとう"]
    seen, started = [], []

    class FakeRunner:
        def __init__(self, *a, **k):
            pass

        def start(self, kind, params, **k):
            started.append((kind, params))
            return SimpleNamespace(view=lambda: {"id": "j1", "kind": kind})
    fake = SimpleNamespace(warnings=[], store=SimpleNamespace(root=tmp_path),
                           reverse=lambda contrast, **kw: seen.append(contrast) or {"name": "x"})
    monkeypatch.setattr(gne.api, "Session", lambda *a, **k: fake)
    monkeypatch.setattr(ms, "JobRunner", FakeRunner)
    srv = ms.build_server("m.gguf")
    spec = {"name": "c1_mcp", "system_a": "悪役\x85として", "system_b": "ふつうに\u2028", "user_prompts": qs}

    async def go():
        await srv.call_tool("reverse", dict(spec))
        await srv.call_tool("heads_pipeline", {"spec": dict(spec)})
    anyio.run(go)
    cs = ContrastSet.load(seen[-1])
    assert cs.prompts("A") == qs and (cs.system_a, cs.system_b) == (spec["system_a"], spec["system_b"])
    assert started[-1] == ("head_pipeline", {"spec": spec})
