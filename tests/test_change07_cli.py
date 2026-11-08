"""CHANGE-07 WP5 のテスト: 鍵つきの書き込み(keyed write)の CLI・MCP・ジョブ・出力スキーマ。

モデルなし: CLI のヘルプと引数の誤り(--json のエラーの形)、表示の補助(keyed のユニット・書き出しの行・--iv の読み方)、
ジョブの種類(direction_key・direction_key_check、MCP 向けに値を落とす compact、書き出しの既定のファイル名)、
GainOut が keyed・焼き込みの項目を受けること。
モデルあり(Qwen3-0.6B、GNE_TEST_MODEL。CPU で回すときは GNE_TEST_NGL=0): `gne` を子プロセスで起動し、語尾の対照集合
(configs/contrast/zundamon_endings.yaml)と与えた返事で、direction key・key-check・keys・key-verify・headproj export
(keyed の LoRA と f16 の焼き込み)・gain の --json がスキーマを通ることと、エラーコード(FILE_NOT_FOUND・BAD_KEY・BAD_ARGS)。
MCP(gne mcp を stdio でつなぐ): 新しいツールがあり、direction_key・direction_key_check のジョブと keyed_export が往復する。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import anyio
import pytest
import yaml
from typer.testing import CliRunner

from gne.schemas import (
    COMMAND_SCHEMAS,
    DirectionKeyCheckOut,
    DirectionKeyOut,
    ErrorOut,
    GainOut,
    ProgressLine,
)
from gne.steer import GneError

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GNE_TEST_MODEL", ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
NGL = os.environ.get("GNE_TEST_NGL", "-1")
ZUNDAMON = ROOT / "configs" / "contrast" / "zundamon_endings.yaml"
LAYERS = "8,14"

# モデル自身の素の返事に見立てた、です・ます調の返事(zundamon_endings.yaml の user_prompts と同じ順)
REPLIES = ["天気が良ければ公園を散歩するのがおすすめです。家ではゆっくり本を読むのもいいと思います。",
           "ご飯と味噌汁と卵焼きがおすすめです。忙しい日はバナナとヨーグルトでも十分です。",
           "料理は手軽に始められる趣味です。散歩や写真も気軽に楽しめます。",
           "雨の日は家で映画を見るのが好きです。温かいお茶を飲みながら過ごします。",
           "まずはしっかり眠ることが大切です。お風呂にゆっくり入るのも効果があります。",
           "私は秋が一番好きです。涼しくて食べ物がおいしい季節です。",
           "野菜と肉を炒めてから水で煮込みます。最後にルーを溶かせば完成です。",
           "寝る前にスマホを見ないようにすると眠りやすくなります。温かい牛乳を飲むのもおすすめです。",
           "京都はお寺や神社が多くておすすめです。北海道では自然とおいしい料理が楽しめます。",
           "時間を決めて短く区切るのが効果的です。机の上を片付けると集中しやすくなります。",
           "どちらもかわいいですが、私は猫が好きです。気ままなところが魅力です。",
           "毎日少しずつ歩くようにしています。野菜を多めに食べることも心がけています。",
           "温かい緑茶がおすすめです。夏は冷たい麦茶もおいしいです。",
           "まず要らない物を捨てることが大切です。物の置き場所を決めると散らかりにくくなります。",
           "相手の好きな物を選ぶのが一番です。手紙を添えると気持ちが伝わります。",
           "毎日同じ時間に寝るようにするといいです。起きたらすぐにカーテンを開けて光を浴びます。"]

ENV_STUB = {"schema_version": "1", "command": "x", "model_id": "m", "llama_cpp_version": "v", "seed": 0,
            "generated_at": "2026-11-07T00:00:00+00:00", "warnings": []}


# --- モデルなし ----------------------------------------------------------------------------------------

def _cli(*args):
    from gne.cli import app
    return CliRunner().invoke(app, list(args))


def test_help_lists_key_commands_and_options():
    r = _cli("direction", "--help")
    assert r.exit_code == 0
    for cmd in ("key", "key-check", "keys", "key-verify", "score"):
        assert cmd in r.output
    r = _cli("direction", "key", "--help")
    assert r.exit_code == 0
    for opt in ("--contrast", "--layers", "--positions", "--positions-b", "--skip", "--max-tokens", "--q-mode",
                "--q-energy", "--q-max-rank", "--q-max-samples", "--subset", "--pseudo", "--replies-a", "--replies-b",
                "--iv", "--target", "--seed", "--name", "--json"):
        assert opt in r.output, opt
    r = _cli("direction", "key-check", "--help")
    for opt in ("--key", "--contrast", "--subset", "--layers", "--strict", "--replies-a", "--iv", "--max-tokens"):
        assert opt in r.output, opt
    for cmd in (("headproj", "export"), ("gain",)):
        r = _cli(*cmd, "--help")
        assert "--key" in r.output and "--direction" in r.output


def test_cli_validation_errors_are_json():
    """--layers が無ければ typer の使い方の誤り。all は鍵では使えない(BAD_ARGS、モデルを読む前に返す)。"""
    r = _cli("direction", "key", "--contrast", str(ZUNDAMON))
    assert r.exit_code == 2 and "--layers" in r.output
    r = _cli("direction", "key", "--contrast", str(ZUNDAMON), "--layers", "all", "--json")
    assert r.exit_code == 1
    d = ErrorOut.model_validate(json.loads(r.stdout.strip().splitlines()[-1]))
    assert d.error.code == "BAD_ARGS" and d.command == "direction key"
    r = _cli("direction", "key", "--contrast", str(ZUNDAMON), "--layers", "8-x", "--json")
    assert json.loads(r.stdout.strip().splitlines()[-1])["error"]["code"] == "BAD_ARGS"
    r = _cli("direction", "key-check", "--json")                  # --key は必須
    assert r.exit_code == 2


def test_display_helpers():
    from gne.cli import _iv_arg, _keyed_export_lines, _mod_text, _target_who, _units_text
    assert _iv_arg(None) is None and _iv_arg([]) is None
    assert _iv_arg(["tsun"]) == "tsun"                               # 保存したレシピの名前
    assert _iv_arg(["14:1:+0.5"]) == ["14:1:+0.5"]                   # ユニットの書き方
    assert _iv_arg(["14:1:+0.5", "15:H2:x0"]) == ["14:1:+0.5", "15:H2:x0"]
    assert _target_who("ffn") == "ffn" and _target_who("attn") == "attn" and _target_who("shared") == "S"
    assert _target_who("expert:3") == "E3"
    row = {"layer": 19, "neuron": None, "kind": "ffn", "seed": None, "dose": 0.0, "norm": 30.0, "rel_push": 0.25,
           "rho": 120.0, "add": None, "down_norm": None, "mode": "keyed", "k": -1.5, "alpha": -1.5,
           "direction": "zundamon_endings", "key": "zundamon_endings", "diff_rel": 0.17}
    t = _units_text([row, {**row, "kind": "expert", "expert": 3, "norm": 0.0, "rel_push": None}])
    assert "L19 ffn keyed α=-1.5 @zundamon_endings #zundamon_endings" in t and "残差ノルム ρ_L の 0.25 倍" in t
    assert "L19 E3 keyed" in t and "dose" not in t                  # add の行として出さない
    d = {"keyed_layers": {"19": {"tensors": {"blk.19.ffn_down.weight": 2}, "keys": ["a", "b"], "directions": ["d"],
                                 "rank": 2}}, "requant_error_max": 0.7}
    lines = _keyed_export_lines(d)
    assert "L19: 鍵つきの書き込み blk.19.ffn_down.weight 階数 2(鍵 a, b、方向 d)" in lines[0]
    assert "70.0%" in lines[1] and "--requant f16" in lines[1]
    assert len(_keyed_export_lines(d, requant_line=False)) == 1
    assert _mod_text({"tensor": "t", "from": "Q8_0", "to": "F16", "requant_error": 0.01}) == "t(Q8_0 → F16、丸めで失った分は編集の 1.0%)"
    assert _mod_text("blk.1.ffn_down.weight") == "blk.1.ffn_down.weight"


def test_schemas_registered_and_gain_out_keyed_fields():
    for cmd, cls in (("direction key", DirectionKeyOut), ("direction key-check", DirectionKeyCheckOut)):
        assert COMMAND_SCHEMAS[cmd] is cls
    assert {"direction keys", "direction key-verify"} <= set(COMMAND_SCHEMAS)
    recipe = {"name": "z", "model_id": "m", "dose_unit": "rel", "units": [], "default_key": "zundamon_endings"}
    base = {**ENV_STUB, "command": "headproj export", "kind": "bake", "recipe": recipe, "path": "p.gguf", "bytes": 1}
    GainOut.model_validate({**base, "modified": [{"tensor": "blk.19.ffn_down.weight", "from": "Q8_0", "to": "F16",
                                                  "requant_error": 0.003}], "requant_error_max": 0.003,
                            "keyed_layers": {"19": {"tensors": {"blk.19.ffn_down.weight": 1}, "keys": ["k"],
                                                    "directions": ["d"], "rank": 1}},
                            "keyed_rank": {"blk.19.ffn_down.weight": 1}, "sidecar": "p.gguf.headproj.json"})
    GainOut.model_validate({**base, "modified": ["blk.19.ffn_down.weight"], "parts": ["a-00001-of-00002.gguf"],
                            "requant": "f16", "requant_error_max": 0.01})            # 分割の焼き込みは名前だけ


class _FakeSession:
    """ジョブの関数が呼ぶ Session のメソッドだけ(呼ばれた引数を記録する)。"""

    def __init__(self, tmp_path):
        self.calls = []
        self.progress = object()
        self.store = type("Store", (), {"export_dir": lambda _s: tmp_path})()

    def direction_key(self, contrast, **kw):
        self.calls.append(("direction_key", contrast, kw))
        return {"name": "k"}

    def direction_key_check(self, key, **kw):
        self.calls.append(("direction_key_check", key, kw))
        return {"key": key, "layers": [{"layer": 14, "values": {"A": [1.0]}, "hist": {"edges": [0, 1]},
                                        "pseudo_auc": {"p95": 0.5, "values": [0.5]},
                                        "pseudo_selectivity": {"p95": 0.1, "values": [0.1]}}]}

    def recipe(self, **kw):
        from gne.steer import Recipe, parse_unit
        self.calls.append(("recipe", kw))
        r = Recipe([parse_unit(u) for u in kw.get("units") or []], "m", name=kw.get("name"))
        r.default_key, r.default_direction = kw.get("default_key"), kw.get("default_direction")
        return r

    def default_export_path(self, recipe, *, bake=False, kind="headproj", name=None):
        from gne.api import Session
        return Session.default_export_path(self, recipe, bake=bake, kind=kind, name=name)

    def headproj_export(self, r, path, **kw):
        self.calls.append(("headproj_export", str(path), kw))
        return {"path": str(path)}


def test_job_kinds(tmp_path):
    from gne.jobs import _direction_key, _direction_key_check, _headproj_export, standard_kinds
    kinds = standard_kinds()
    assert kinds["direction_key"] is _direction_key and kinds["direction_key_check"] is _direction_key_check
    s = _FakeSession(tmp_path)
    _direction_key(s, "c.yaml", layers="8,14", q_mode="means")
    name, contrast, kw = s.calls[-1]
    assert (name, contrast, kw["layers"], kw["q_mode"]) == ("direction_key", "c.yaml", [8, 14], "means")
    assert kw["progress"] is s.progress                                 # ジョブの進捗をそのまま渡す
    with pytest.raises(GneError) as e:
        _direction_key(s, "c.yaml")
    assert e.value.code == "BAD_ARGS"
    d = _direction_key_check(s, "k", layers=[14], compact=True)
    assert "values" not in d["layers"][0] and "hist" not in d["layers"][0]
    assert "values" not in d["layers"][0]["pseudo_auc"] and d["layers"][0]["pseudo_selectivity"]["p95"] == 0.1
    assert "values" in _direction_key_check(s, "k")["layers"][0]          # UI はヒストグラムに使う
    # 書き出しの既定のファイル名: keyed だけなら <名前>-keyed、焼き込みは -keyed.gguf。key・direction はレシピの既定に
    d = _headproj_export(s, units=["19:ffn:keyed:a1"], name="z", key="zk", direction="zd")
    assert Path(d["path"]).name == "z-keyed.lora.gguf" and d["file"] == "z-keyed.lora.gguf"
    rk = [c for c in s.calls if c[0] == "recipe"][-1][1]
    assert rk["default_key"] == "zk" and rk["default_direction"] == "zd"
    d = _headproj_export(s, units=["19:ffn:keyed:a1@zd#zk"], name="z", bake=True)
    assert Path(d["path"]).name == "z-keyed.gguf"
    d = _headproj_export(s, units=["33:H16:s6@zd"], name="z")
    assert Path(d["path"]).name == "z.lora.gguf"
    d = _headproj_export(s, units=["19:ffn:keyed:a1@zd#zk"], name="z", file_stem="mine")
    assert Path(d["path"]).name == "mine.lora.gguf"


def test_mcp_instructions_mention_keyed():
    from gne.mcp_server import INSTRUCTIONS
    assert "keyed" in INSTRUCTIONS and "direction_key_check" in INSTRUCTIONS and "zundamon_endings" in INSTRUCTIONS
    assert "介入なしで作る" in INSTRUCTIONS and "subset=\"odd\"" in INSTRUCTIONS


# --- モデルを使う ------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """作業場所: 保存先(GNE_ROOT)と返事のファイル。方向は zundamon_endings(system_a 対 なし、生成プロンプトの最後)で作る。"""
    if not MODEL.exists():
        pytest.skip(f"{MODEL} がありません")
    d = tmp_path_factory.mktemp("c07cli")
    (d / "replies.yaml").write_text(yaml.safe_dump(REPLIES, allow_unicode=True), encoding="utf-8")
    w = {"root": d / "root", "dir": d, "replies": d / "replies.yaml"}
    gne_json(w, "direction", "--contrast", str(ZUNDAMON), "--positions", "last")
    return w


def gne_json(w, *args, expect_ok=True):
    """gne を子プロセスで起動し、stdout が JSON だけで、最後の行がスキーマ(失敗なら ErrorOut)を通ることを確かめる。"""
    env = {**os.environ, "GNE_ROOT": str(w["root"]), "GNE_MODEL": str(MODEL), "CUDA_VISIBLE_DEVICES": os.environ.get(
        "CUDA_VISIBLE_DEVICES", ""), "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}
    p = subprocess.run([sys.executable, "-m", "gne.cli", *args, "--json", "--n-gpu-layers", NGL, "--n-ctx", "1024"],
                       capture_output=True, text=True, env=env, timeout=600, cwd=str(w["dir"]))
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    assert lines, f"stdout が空です: {p.stderr[-2000:]}"
    objs = [json.loads(ln) for ln in lines]
    for o in objs[:-1]:
        ProgressLine.model_validate(o)
    last = objs[-1]
    if expect_ok:
        assert p.returncode == 0, (p.returncode, last, p.stderr[-2000:])
        COMMAND_SCHEMAS[last["command"]].model_validate(last)
    else:
        assert p.returncode != 0
        ErrorOut.model_validate(last)
    return last


@pytest.fixture(scope="module")
def key(work):
    """側 A の作り方 (b): 与えた返事(です・ます調)の「です。」「ます。」を書く直前。dont_touch の英語は 24 トークン生成。"""
    return gne_json(work, "direction", "key", "--contrast", str(ZUNDAMON), "--layers", LAYERS, "--replies-a",
                    str(work["replies"]), "--max-tokens", "24", "--pseudo", "8")


@pytest.mark.model
def test_direction_key_json(work, key):
    d = key
    assert d["command"] == "direction key" and d["name"] == "zundamon_endings" and d["mode"] == "same"
    assert d["positions"] == "before:(です|ます)。" and d["positions_b"] == "not-before:(です|ます)。"
    assert [r["layer"] for r in d["layers"]] == [8, 14] and d["subset"] == "even" and d["n_pseudo"] == 8
    assert d["matched"] == {"A": 8, "of": 8} and d["replies"]["AB"] == "given" and d["replies"]["dt:english"] == "generated"
    for r in d["layers"]:
        assert abs(r["kappa_mu"] - 1) < 1e-6 and 0 < r["kn"] <= 1 and r["q_rank"] > 0
    assert Path(d["path"]).exists() and Path(d["path"]).parent.name == d["model_id"]
    assert str(work["root"]) not in json.dumps(d["notes"])                 # 注に手元のパスを書かない


@pytest.mark.model
def test_key_check_keys_verify_json(work, key):
    c = gne_json(work, "direction", "key-check", "--key", "zundamon_endings", "--contrast", str(ZUNDAMON),
                 "--replies-a", str(work["replies"]), "--max-tokens", "24")
    assert c["held_out"] is True and c["subset"] == "odd" and [r["layer"] for r in c["layers"]] == [8, 14]
    for r in c["layers"]:
        assert r["auc"] is not None and r["auc_token"] is not None and r["selectivity"] is not None
        assert r["pseudo_selectivity"]["n"] == 8 and r["beats_pseudo"] in (True, False)
        assert "english" in r["dont_touch_median"] and r["hist"]["edges"]
    c2 = gne_json(work, "direction", "key-check", "--key", "zundamon_endings", "--layers", "14")   # 対照集合なし: 作成時の例
    assert c2["held_out"] is False and c2["contrast"] is None and any("held_out" in w for w in c2["warnings"])
    e = gne_json(work, "direction", "key-check", "--key", "zundamon_endings", "--contrast", str(ZUNDAMON), "--subset",
                 "even", "--replies-a", str(work["replies"]), "--strict", expect_ok=False)
    assert e["error"]["code"] == "BAD_ARGS"
    e = gne_json(work, "direction", "key-check", "--key", "no_such_key", expect_ok=False)
    assert e["error"]["code"] == "FILE_NOT_FOUND"
    (Path(key["path"]).parent / "broken.npz").write_bytes(b"not a key")
    e = gne_json(work, "direction", "key-check", "--key", "broken", expect_ok=False)
    assert e["error"]["code"] == "BAD_KEY"
    ks = gne_json(work, "direction", "keys")
    by = {k["name"]: k for k in ks["keys"]}
    assert by["zundamon_endings"]["layers"] == [8, 14] and "error" in by["broken"]
    (Path(key["path"]).parent / "broken.npz").unlink()
    v = gne_json(work, "direction", "key-verify", "--layer", "14", "--key", "zundamon_endings", "--direction",
                 "zundamon_endings", "--alphas", "1")
    assert v["v30"] is True and v["tensor"] == "blk.14.ffn_down.weight" and len(v["rows"]) == 1


@pytest.mark.model
def test_keyed_export_json(work, key, tmp_path):
    """headproj export: --key で #KEY を省いたユニット、既定の名前は <レシピ名>-keyed、焼き込みの既定は auto
    (元が Q8_0 のテンソルは Q8_0 のまま。残った編集の割合 edit_kept も出る)。"""
    d = gne_json(work, "headproj", "export", "--unit", "14:ffn:keyed:a1@zundamon_endings", "--key", "zundamon_endings")
    assert d["kind"] == "lora" and Path(d["path"]).name == "headproj-keyed.lora.gguf" and Path(d["path"]).exists()
    assert d["keyed_layers"]["14"]["keys"] == ["zundamon_endings"] and d["keyed_rank"] == {"blk.14.ffn_down.weight": 1}
    assert d["recipe"]["default_key"] == "zundamon_endings"
    out = tmp_path / "z-keyed.gguf"
    b = gne_json(work, "headproj", "export", "--bake", "--unit", "14:ffn:keyed:a1@zundamon_endings#zundamon_endings",
                 "-o", str(out))
    assert b["kind"] == "bake" and b["modified"][0]["to"] == "Q8_0" and b["requant_error_max"] < 0.5
    assert b["requant"] == "auto" and b["modified"][0]["edit_kept"] >= 0.9 and "fallback" not in b["modified"][0]
    assert b["modified"][0]["requant_error"] == b["requant_error_max"] and b["keyed_layers"]["14"]["rank"] == 1
    out.unlink()
    g = gne_json(work, "gain", "--unit", "8:ffn:keyed:a0.5", "--unit", "14:ffn:keyed:a1", "--key", "zundamon_endings",
                 "--direction", "zundamon_endings")
    assert Path(g["path"]).name == "gain-keyed.lora.gguf" and set(g["keyed_layers"]) == {"8", "14"}
    e = gne_json(work, "headproj", "export", "--unit", "14:ffn:keyed:a1@zundamon_endings#no_such_key", expect_ok=False)
    assert e["error"]["code"] == "FILE_NOT_FOUND"


def _payload(r):
    sc = getattr(r, "structured_content", None)
    if sc:
        return sc.get("result", sc) if isinstance(sc, dict) else sc
    return json.loads(r.content[0].text)


@pytest.mark.model
def test_mcp_keyed_tools(work, key):
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}

    async def main():
        params = StdioServerParameters(command=sys.executable,
                                       args=["-m", "gne.cli", "mcp", "--model", str(MODEL), "--root", str(work["root"]),
                                             "--n-gpu-layers", NGL], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as s:
                init = await s.initialize()
                assert "keyed" in (init.instructions or "")
                names = {t.name for t in (await s.list_tools()).tools}
                assert {"direction_key", "direction_key_check", "direction_key_verify", "keyed_export", "keys",
                        "headproj_export"} <= names

                async def call(tool, **args):
                    return _payload(await s.call_tool(tool, args))

                async def wait(job):
                    assert job.get("error") is None and job.get("id"), job
                    for _ in range(600):
                        js = await call("job_status", job_id=job["id"])
                        if js["status"] not in ("queued", "running"):
                            return js
                        await anyio.sleep(0.2)
                    raise AssertionError("ジョブが終わりません")

                ks = await call("keys")
                assert [k["name"] for k in ks["keys"]] == ["zundamon_endings"]
                js = await wait(await call("direction_key", contrast="zundamon_endings", layers=[14], name="z_mcp",
                                           replies_a=str(work["replies"]), max_tokens=16, pseudo=4))
                assert js["status"] == "done", js
                r = js["result"]
                DirectionKeyOut.model_validate({**ENV_STUB, **r})
                assert r["name"] == "z_mcp" and r["layers"][0]["layer"] == 14
                js = await wait(await call("direction_key_check", key="z_mcp", contrast="zundamon_endings",
                                           replies_a=str(work["replies"]), max_tokens=16))
                assert js["status"] == "done", js
                c = js["result"]
                assert c["held_out"] is True and c["layers"][0]["auc"] is not None
                assert "values" not in c["layers"][0] and "hist" not in c["layers"][0]      # MCP 向けに落とす
                ex = await call("keyed_export", units=["14:ffn:keyed:a1"], name="zm", key="z_mcp",
                                direction="zundamon_endings")
                assert "error" not in ex, ex
                assert Path(ex["path"]).name == "zm-keyed.lora.gguf" and Path(ex["path"]).exists()
                assert ex["keyed_layers"]["14"]["keys"] == ["z_mcp"]
                v = await call("direction_key_verify", layer=14, key="z_mcp", direction="zundamon_endings", alphas=[1])
                assert v.get("v30") is True and len(v["rows"]) == 1, v
                err = await call("keyed_export")
                assert err["error"]["code"] == "BAD_ARGS"
                err = await call("direction_key", contrast="zundamon_endings", layers=[])
                assert err["error"]["code"] == "BAD_ARGS"

    anyio.run(main)
