"""gne modal のテスト。Modal には接続しない。modal コマンドは偽物、modal のライブラリは最小の代役に差し替える。"""
import importlib.util
import json
import os
import stat
import subprocess
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gne.modal_cli import APP_FILE, FETCH_FILE, modal_app, parse_seconds

sys.path.insert(0, str(Path(__file__).parent / "tools"))
import make_tiny_qwen4exp as tiny  # noqa: E402

runner = CliRunner()


@pytest.fixture
def fake_modal(tmp_path, monkeypatch):
    """呼ばれた引数を calls.jsonl に書くだけの modal コマンド。"""
    log = tmp_path / "calls.jsonl"
    exe = tmp_path / "bin" / "modal"
    exe.parent.mkdir()
    exe.write_text(f"#!{sys.executable}\nimport json, sys\nopen({str(log)!r}, 'a').write(json.dumps(sys.argv[1:]) + chr(10))\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{exe.parent}{os.pathsep}{os.environ['PATH']}")
    return lambda: [json.loads(x) for x in log.read_text().splitlines()]


def _opt(args, name):
    return args[args.index(name) + 1]


def test_run_passes_the_command_through_and_stays_detached(fake_modal):
    r = runner.invoke(modal_app, ["run", "--gpu", "H100", "--timeout", "2h", "--", "heads", "pipeline", "--spec", "/data/inputs/s.json", "--yes"])
    assert r.exit_code == 0, r.output
    args = fake_modal()[0]
    assert args[:3] == ["run", "--detach", f"{APP_FILE}::main"]
    assert json.loads(_opt(args, "--argv-json")) == ["heads", "pipeline", "--spec", "/data/inputs/s.json", "--yes"]
    assert _opt(args, "--gpu") == "H100" and _opt(args, "--timeout") == "7200"


def test_run_defaults_leave_the_gpu_to_the_app(fake_modal):
    assert runner.invoke(modal_app, ["run", "--attach", "--", "info", "-m", "/models/x.gguf"]).exit_code == 0
    args = fake_modal()[0]
    assert "--gpu" not in args and "--detach" not in args and _opt(args, "--timeout") == "21600"


def test_run_can_start_a_script(fake_modal):
    assert runner.invoke(modal_app, ["run", "--script", "/data/inputs/x.py", "--", "--n", "3"]).exit_code == 0
    args = fake_modal()[0]
    assert _opt(args, "--script") == "/data/inputs/x.py" and json.loads(_opt(args, "--argv-json")) == ["--n", "3"]


def test_run_needs_a_command(fake_modal):
    assert runner.invoke(modal_app, ["run"]).exit_code != 0


@pytest.mark.parametrize("text,sec", [("90", 90), ("90m", 5400), ("6h", 21600), ("24h", 86400)])
def test_timeout_units(text, sec):
    assert parse_seconds(text) == sec


@pytest.mark.parametrize("text", ["25h", "0", "soon", "-1h"])
def test_timeout_out_of_range(text):
    import typer
    with pytest.raises(typer.BadParameter):
        parse_seconds(text)


def test_put_names_the_path_inside_the_container(fake_modal, tmp_path):
    f = tmp_path / "spec.json"
    f.write_text("{}")
    r = runner.invoke(modal_app, ["put", str(f)])
    assert r.exit_code == 0 and "/data/inputs/spec.json" in r.output
    assert fake_modal()[0] == ["volume", "put", "gne-data", str(f), "inputs/spec.json"]


def test_fetch_and_volume_commands(fake_modal, tmp_path):
    assert runner.invoke(modal_app, ["fetch", "a/b", "--include", "*IQ1_S*"]).exit_code == 0
    assert runner.invoke(modal_app, ["ls", "--models"]).exit_code == 0
    assert runner.invoke(modal_app, ["get", "/gne/m1", str(tmp_path)]).exit_code == 0
    calls = fake_modal()
    assert calls[0][:2] == ["run", f"{FETCH_FILE}::main"]          # CUDA のビルドに巻き込まれないよう別のファイル
    assert json.loads(_opt(calls[0], "--include-json")) == ["*IQ1_S*"] and _opt(calls[0], "--repo") == "a/b"
    assert calls[1] == ["volume", "ls", "gne-models"]
    assert calls[2] == ["volume", "get", "gne-data", "gne/m1", str(tmp_path)]


@pytest.fixture
def fake_modal_json(tmp_path, monkeypatch):
    """app list と billing report に決まった JSON を返し、呼ばれた引数を記録する modal コマンド。"""
    log = tmp_path / "calls.jsonl"
    exe = tmp_path / "bin" / "modal"
    exe.parent.mkdir()
    apps = [{"app_id": "ap-run", "description": "gne", "state": "ephemeral", "tasks": "1"},
            {"app_id": "ap-old", "description": "gne", "state": "stopped", "tasks": "0"},
            {"app_id": "ap-other", "description": "someone-else", "state": "deployed", "tasks": "1"},
            {"app_id": "ap-fetch", "description": "gne-fetch", "state": "stopping...", "tasks": "1"}]
    bill = [{"Cost": "1.25"}, {"Cost": "0.50"}]
    exe.write_text(f"#!{sys.executable}\nimport json, sys\nopen({str(log)!r}, 'a').write(json.dumps(sys.argv[1:]) + chr(10))\n"
                   f"if sys.argv[1:3] == ['app', 'list']: print(json.dumps({apps!r}))\n"
                   f"if sys.argv[1:3] == ['billing', 'report']: print(json.dumps({bill!r}))\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{exe.parent}{os.pathsep}{os.environ['PATH']}")
    return lambda: [json.loads(x) for x in log.read_text().splitlines()]


def test_ps_lists_only_running_gne_apps(fake_modal_json):
    r = runner.invoke(modal_app, ["ps"])
    assert r.exit_code == 0 and "ap-run" in r.output and "ap-fetch" in r.output
    assert "ap-old" not in r.output and "ap-other" not in r.output and "2 件" in r.output


def test_stop_all_stops_only_gne_apps(fake_modal_json):
    assert runner.invoke(modal_app, ["stop", "--all"]).exit_code == 0
    stops = [c for c in fake_modal_json() if c[:2] == ["app", "stop"]]
    assert stops == [["app", "stop", "-y", "ap-run"], ["app", "stop", "-y", "ap-fetch"]]
    assert runner.invoke(modal_app, ["stop"]).exit_code != 0


def test_cost_sums_the_billing_report(fake_modal_json):
    r = runner.invoke(modal_app, ["cost"])
    assert r.exit_code == 0 and "$1.75" in r.output


def test_hf_upload_is_private_unless_asked(fake_modal, tmp_path):
    from gne.modal_cli import UPLOAD_FILE
    card = tmp_path / "README.md"
    card.write_text("card")
    assert runner.invoke(modal_app, ["hf-upload", "--repo", "me/x", "export/a.gguf", "/export/b.gguf", "--card", str(card)]).exit_code == 0
    assert runner.invoke(modal_app, ["hf-upload", "--repo", "me/x", "export/a.gguf", "--public"]).exit_code == 0
    c1, c2 = fake_modal()
    assert c1[:2] == ["run", f"{UPLOAD_FILE}::main"] and json.loads(_opt(c1, "--paths-json")) == ["export/a.gguf", "export/b.gguf"]
    assert "--no-private" not in c1 and _opt(c1, "--card-path") == str(card.resolve())
    assert "--no-private" in c2
    assert runner.invoke(modal_app, ["hf-upload", "--repo", "me/x", "export/new.gguf", "--remove", "old.gguf"]).exit_code == 0
    assert json.loads(_opt(fake_modal()[-1], "--remove-json")) == ["old.gguf"]


def test_missing_modal_command_says_how_to_install(monkeypatch):
    monkeypatch.setenv("PATH", "")
    r = runner.invoke(modal_app, ["run", "--", "info"])
    assert r.exit_code == 2 and "uv tool install modal" in r.output


# --- コンテナの中身(modal のライブラリは代役)-----------------------------------------------------------------------


class _Chain:
    """Image も Volume も、呼ばれたら自分を返すだけの代役。commit の回数を数える。"""

    def __init__(self):
        self.commits = 0

    def __getattr__(self, name):
        return lambda *a, **k: self

    def commit(self):
        self.commits += 1


@pytest.fixture
def app_module(monkeypatch):
    vols = {}

    class App:
        def __init__(self, name):
            pass

        def function(self, **k):
            return lambda f: f

        def local_entrypoint(self):
            return lambda f: f

    stub = types.SimpleNamespace(
        App=App, is_local=lambda: True, Image=_Chain(),
        Volume=types.SimpleNamespace(from_name=lambda name, **k: vols.setdefault(name, _Chain())))
    monkeypatch.setitem(sys.modules, "modal", stub)
    spec = importlib.util.spec_from_file_location("gne_modal_app_test", APP_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.vols = vols
    return mod


def test_only_tracked_files_can_go_up(app_module, tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "contrast").mkdir()
    (tmp_path / "contrast" / "public.yaml").write_text("a")
    (tmp_path / "contrast" / "mine.yaml").write_text("b")
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "x.json").write_text("c")
    subprocess.run(["git", "-C", str(tmp_path), "add", "contrast/public.yaml"], check=True)
    ignore = app_module.tracked_only(tmp_path)
    assert not ignore(Path("contrast")) and not ignore(Path("contrast/public.yaml"))
    assert ignore(Path("contrast/mine.yaml")) and ignore(Path("private")) and ignore(Path("private/x.json"))


def test_outside_a_git_checkout_it_refuses(app_module, tmp_path):
    with pytest.raises(RuntimeError, match="git"):
        app_module.tracked_only(tmp_path)


def test_run_gne_runs_the_command_and_returns_its_exit_code(app_module, tmp_path, monkeypatch):
    model = tiny.write(tmp_path / "tiny.gguf")
    repo = str(Path(__file__).parent.parent)
    monkeypatch.setenv("PYTHONPATH", repo)
    monkeypatch.setenv("GNE_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert app_module.run_gne(["info", "-m", str(model), "--no-load"], threads=2) == 0
    assert app_module.run_gne(["no-such-command"], threads=2) != 0


def test_results_are_committed_even_when_the_command_fails(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "llama_libs", lambda: [])
    monkeypatch.setattr(app_module, "run_gne", lambda argv, threads, script="": 3)
    with pytest.raises(RuntimeError, match="終了コード 3"):
        app_module.run(["heads", "pipeline"])
    assert app_module.vols["gne-data"].commits == 1
    monkeypatch.setattr(app_module, "run_gne", lambda argv, threads, script="": 0)
    assert app_module.run(["info"]) == 0
    assert app_module.vols["gne-data"].commits == 2


PROC_NET_TCP = """  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 00000000:0016 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 1 1
   1: 0100007F:2257 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 2 1
   2: 0A000002:0016 0A000003:C350 01 00000000:00000000 00:00000000 00000000     0        0 3 1
   3: 0100007F:2257 0100007F:C351 01 00000000:00000000 00:00000000 00000000     0        0 4 1
"""


def test_only_established_ssh_connections_count(app_module):
    assert app_module.ssh_connections(PROC_NET_TCP) == 1       # 待ち受けと、UI のポートの接続は数えない
    assert app_module.ssh_connections(PROC_NET_TCP.splitlines()[0]) == 0


def test_ssh_server_only_forwards_to_the_ui(app_module):
    cfg = app_module.sshd_config("/k", "/a", 8791)
    for line in ("PasswordAuthentication no", "PermitTTY no", "AllowTcpForwarding local", "PermitOpen 127.0.0.1:8791", "ForceCommand /bin/false"):
        assert line in cfg


def test_tunnel_is_local_and_pins_the_server_key(tmp_path):
    from gne.modal_cli import TUNNEL_RE, make_keys, ssh_args
    args = ssh_args("r1.modal.host", 1234, Path("/k"), Path("/kh"), 8800)
    assert "127.0.0.1:8800:127.0.0.1:8791" in args and "StrictHostKeyChecking=yes" in args and args[-1] == "root@r1.modal.host"
    m = TUNNEL_RE.search("[ta-1] GNE_TUNNEL r1.modal.host 1234")
    assert m[1] == "r1.modal.host" and m[2] == "1234"
    key, pub, host_key, host_pub = make_keys(tmp_path)
    assert key.exists() and pub.startswith("ssh-ed25519") and host_pub.startswith("ssh-ed25519") and "PRIVATE" in host_key


def test_stopping_the_ui_stops_the_app_on_modal(fake_modal, capfd):
    """modal run を止めるだけではコンテナが動き続け、課金も続く。"""
    import shutil

    from gne.modal_cli import APP_RE, stop_app
    assert APP_RE.search("https://modal.com/apps/me/main/ap-AbC123")[0] == "ap-AbC123"
    assert stop_app(shutil.which("modal"), "ap-AbC123") is True
    assert fake_modal() == [["app", "stop", "-y", "ap-AbC123"]]
    assert stop_app(shutil.which("modal"), None) is False       # 分からないときは黙らない
    assert "modal app stop" in capfd.readouterr().err


def test_serve_command_line_never_holds_a_key(fake_modal):
    r = runner.invoke(modal_app, ["serve", "-m", "/models/x.gguf", "--gpu", "H200", "--idle", "10m", "--dry-run"])
    assert r.exit_code == 0, r.output
    args = json.loads(r.output.strip().splitlines()[-1])
    assert args[:2] == ["run", f"{APP_FILE}::serve_main"] and _opt(args, "--idle") == "600" and _opt(args, "--timeout") == "7200"
    assert "PRIVATE" not in r.output and "ssh-ed25519" not in r.output


def test_threads_reach_the_engine(tmp_path, monkeypatch):
    from gne.engine import Engine
    model = tiny.write(tmp_path / "tiny.gguf")
    monkeypatch.setenv("GNE_THREADS", "3")
    e = Engine(model, n_gpu_layers=0, n_ctx=64, n_seq_max=1)
    try:
        assert e.n_threads == 3
    finally:
        e.close()
