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
