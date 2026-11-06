"""gne modal: 手元に載らないモデルのために、gne のコマンドを Modal の GPU で回す(docs/modal.md)。

ここは modal コマンドを呼ぶだけで、modal のライブラリは読まない。実際の定義は remote/modal_app.py にある。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import typer

APP_FILE = Path(__file__).resolve().parent / "remote" / "modal_app.py"
FETCH_FILE = APP_FILE.with_name("modal_fetch.py")
UPLOAD_FILE = APP_FILE.with_name("modal_upload.py")
MODELS_VOL, DATA_VOL = "gne-models", "gne-data"
modal_app = typer.Typer(no_args_is_help=True, help="Modal の GPU で gne を回す(docs/modal.md)。料金は https://modal.com/pricing")


def _modal() -> str:
    exe = shutil.which("modal")
    if not exe:
        sys.stderr.write("modal コマンドがありません。uv tool install modal のあと modal setup でログインしてください\n")
        raise typer.Exit(2)
    return exe


def _call(args: list[str], dry_run: bool = False) -> None:
    cmd = [_modal(), *args]
    if dry_run:
        print(json.dumps(cmd[1:], ensure_ascii=False))
        return
    code = subprocess.call(cmd)
    if code:
        raise typer.Exit(code)


def parse_seconds(s: str) -> int:
    """6h・90m・3600 を秒にする。Modal の上限は 24 時間。"""
    m = re.fullmatch(r"(\d+)([smh]?)", s.strip())
    if not m:
        raise typer.BadParameter(f"{s!r}。6h・90m・3600 の形で書いてください")
    sec = int(m[1]) * {"": 1, "s": 1, "m": 60, "h": 3600}[m[2]]
    if not 1 <= sec <= 24 * 3600:
        raise typer.BadParameter("1 秒から 24 時間までです")
    return sec


@modal_app.command("fetch")
def fetch(repo: str = typer.Argument(..., help="Hugging Face のリポジトリ(例 unsloth/Qwen3.8-Flash-Next-GGUF)"),
          include: Optional[list[str]] = typer.Option(None, "--include", help="落とすファイルのパターン(例 '*UD-IQ1_S*')。省くと全部"),
          dry_run: bool = typer.Option(False, "--dry-run", help="実行する modal コマンドを出すだけ")):
    """モデルを Modal の Volume に直接落とす。手元のディスクは使わず、GPU も使わない。終わると -m に渡すパスを出す。"""
    if not include:
        sys.stderr.write("--include が無いのでリポジトリの全ファイルを落とします(量子化の種類ごとに数十〜数百 GB になります)\n")
    _call(["run", f"{FETCH_FILE}::main", "--repo", repo, "--include-json", json.dumps(include or [])], dry_run)


@modal_app.command("put")
def put(local: Path = typer.Argument(..., exists=True, help="上げるファイルかフォルダ"),
        remote: Optional[str] = typer.Argument(None, help="Volume の上の場所(既定 inputs/<名前>)"),
        force: bool = typer.Option(False, "--force", help="同じ名前があれば上書きする"),
        dry_run: bool = typer.Option(False, "--dry-run", help="実行する modal コマンドを出すだけ")):
    """質問集やレシピを上げる。Modal に上がるのは、ここで名指ししたものと、git で追跡している gne/・configs/ だけ。"""
    dst = (remote or f"inputs/{local.name}").lstrip("/")
    _call(["volume", "put", *(["--force"] if force else []), DATA_VOL, str(local), dst], dry_run)
    if not dry_run:
        print(f"/data/{dst}")


@modal_app.command("run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(ctx: typer.Context,
        gpu: Optional[str] = typer.Option(None, "--gpu", help="GPU の種類(既定 A100-80GB、空きがなければ H100)。例 H200。重みを読まない確認は T4 で足りる"),
        cpu: float = typer.Option(8, "--cpu", help="CPU のコア数(gne のスレッド数もこれにする)"),
        memory: int = typer.Option(32768, "--memory", help="メモリ(MiB)"),
        timeout: str = typer.Option("6h", "--timeout", help="時間の上限。1 回の実行ごと。最大 24h"),
        script: Optional[str] = typer.Option(None, "--script", help="gne の代わりに実行する Python のスクリプト(コンテナの中のパス。gne modal put で上げる)"),
        detach: bool = typer.Option(True, "--detach/--attach", help="手元の接続が切れても続ける。途中で止めるのは modal app stop。--attach でも止まるとは限らないので、modal app list で確かめる"),
        dry_run: bool = typer.Option(False, "--dry-run", help="実行する modal コマンドを出すだけ")):
    """gne のコマンドを Modal で実行する。例: gne modal run -- heads pipeline --spec /data/inputs/s.json -m /models/x.gguf --yes

    動いている時間だけ課金される。コマンドが終わるとコンテナも止まる。結果は /data/gne に残り、gne modal get で持ち帰る。"""
    argv = list(ctx.args)
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv and not script:
        raise typer.BadParameter("-- のあとに gne のコマンドを書いてください")
    args = ["run", *(["--detach"] if detach else []), f"{APP_FILE}::main", "--argv-json", json.dumps(argv, ensure_ascii=False),
            "--cpu", str(cpu), "--memory", str(memory), "--timeout", str(parse_seconds(timeout))]
    if gpu:
        args += ["--gpu", gpu]
    if script:
        args += ["--script", script]
    _call(args, dry_run)


TUNNEL_RE = re.compile(r"GNE_TUNNEL (\S+) (\d+)")
APP_RE = re.compile(r"\bap-[A-Za-z0-9]+")


def stop_app(exe: str, app_id: Optional[str]) -> bool:
    """modal run を止めても Modal 側のコンテナは動き続けるので、課金を止めるには明示的に止める。"""
    if not app_id:
        sys.stderr.write("注意: App ID が分からず、Modal 側を止められません。modal app list で確かめて modal app stop してください\n")
        return False
    if subprocess.run([exe, "app", "stop", "-y", app_id], capture_output=True).returncode:
        sys.stderr.write(f"注意: {app_id} を止められませんでした。課金が続きます。modal app stop {app_id} を実行してください\n")
        return False
    sys.stderr.write(f"{app_id} を止めました\n")
    return True


def ssh_args(host: str, port: int, key: Path, known_hosts: Path, local_port: int, remote_port: int = 8791) -> list[str]:
    """手元の 127.0.0.1 だけに UI を出す。サーバの鍵は作った本人のものだけを信用する。"""
    return ["ssh", "-N", "-i", str(key), "-p", str(port), "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
            "-o", "IdentitiesOnly=yes", "-o", f"UserKnownHostsFile={known_hosts}", "-o", "StrictHostKeyChecking=yes",
            "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30", "-o", "LogLevel=ERROR", f"root@{host}"]


def make_keys(work: Path) -> tuple[Path, str, str, str]:
    """この接続だけの使い捨ての鍵を作る。返り値は (手元の秘密鍵, 手元の公開鍵, サーバの秘密鍵, サーバの公開鍵)。"""
    for name in ("client", "host"):
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "gne-modal", "-f", str(work / name)], check=True)
    return (work / "client", (work / "client.pub").read_text().strip(), (work / "host").read_text(),
            (work / "host.pub").read_text().strip())


@modal_app.command("serve")
def serve(model: str = typer.Option(..., "--model", "-m", help="コンテナの中のパス(/models/...)"),
          gpu: Optional[str] = typer.Option(None, "--gpu", help="GPU の種類(既定 A100-80GB、空きがなければ H100)。例 H200"),
          cpu: float = typer.Option(8, "--cpu", help="CPU のコア数"),
          memory: int = typer.Option(32768, "--memory", help="メモリ(MiB)"),
          timeout: str = typer.Option("2h", "--timeout", help="時間の上限。最大 24h"),
          idle: str = typer.Option("15m", "--idle", help="SSH の接続がこれだけ無ければ止める"),
          port: int = typer.Option(8791, "--port", help="手元のポート"),
          dry_run: bool = typer.Option(False, "--dry-run", help="実行する modal コマンドを出すだけ")):
    """Web UI を Modal の GPU で動かし、手元の 127.0.0.1 で開く。動いている間は GPU の料金がかかる。Ctrl+C で止まる。

    Modal の公開 URL は使わない。gne serve はコンテナの中でも 127.0.0.1 だけで待ち受け、この接続だけの使い捨ての鍵を
    持つ SSH のトンネルでつなぐ。チャットの内容は Modal のコンテナを通るが、ログには出ない。"""
    args = ["run", f"{APP_FILE}::serve_main", "--model", model, "--cpu", str(cpu), "--memory", str(memory),
            "--timeout", str(parse_seconds(timeout)), "--idle", str(parse_seconds(idle))]
    if gpu:
        args += ["--gpu", gpu]
    if dry_run:
        print(json.dumps(args, ensure_ascii=False))
        return
    exe = _modal()
    for tool in ("ssh", "ssh-keygen"):
        if not shutil.which(tool):
            sys.stderr.write(f"{tool} がありません\n")
            raise typer.Exit(2)
    with tempfile.TemporaryDirectory(prefix="gne-modal-") as d:
        work = Path(d)
        key, client_pub, host_key, host_pub = make_keys(work)
        known = work / "known_hosts"
        proc = subprocess.Popen([exe, *args], env={**os.environ, "GNE_SSH_PUB": client_pub, "GNE_SSH_HOST_KEY": host_key},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        target: dict = {}

        def read_output():
            for line in proc.stdout:
                a = APP_RE.search(line)
                if a:
                    target.setdefault("app", a[0])
                m = TUNNEL_RE.search(line)
                if m:
                    target["host"] = (m[1], int(m[2]))     # 横取りで作り直されると、新しいトンネルの行が来る
                else:
                    sys.stdout.write(line)
                    sys.stdout.flush()
        threading.Thread(target=read_output, daemon=True).start()
        ssh, current = None, None
        try:
            sys.stderr.write("コンテナを起動しています。初回はイメージのビルドで長くかかります\n")
            while proc.poll() is None:
                time.sleep(1)
                want = target.get("host")
                if want and (want != current or ssh is None or ssh.poll() is not None):
                    if ssh is not None and ssh.poll() is None:
                        ssh.terminate()
                    if want != current:
                        known.write_text(f"[{want[0]}]:{want[1]} {host_pub}\n")
                        sys.stderr.write(f"開けます: http://127.0.0.1:{port}/(Ctrl+C で止めると課金も止まります)\n")
                    else:
                        time.sleep(5)                       # 切れたらつなぎ直す
                    current = want
                    ssh = subprocess.Popen(ssh_args(want[0], want[1], key, known, port))
        except KeyboardInterrupt:
            pass
        finally:
            if ssh is not None and ssh.poll() is None:
                ssh.terminate()
            stop_app(exe, target.get("app"))
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    proc.kill()
    raise typer.Exit(0 if proc.returncode in (0, -15, None) else proc.returncode)


def running_apps(exe: str) -> list[dict]:
    """止まっていない gne のアプリ(gne と gne-fetch)。"""
    out = subprocess.run([exe, "app", "list", "--json"], capture_output=True, text=True)
    if out.returncode:
        sys.stderr.write(out.stderr[-500:])
        raise typer.Exit(out.returncode)
    apps = json.loads(out.stdout or "[]")
    return [a for a in apps if a.get("description") in ("gne", "gne-fetch", "gne-upload") and not str(a.get("state", "")).startswith("stopped")]


@modal_app.command("ps")
def ps():
    """動いている gne のアプリ(止まりかけを含む)。課金が続いているかを確かめるのに使う。"""
    apps = running_apps(_modal())
    for a in apps:
        print(f"{a['app_id']}  {a['description']:9} {a['state']:12} タスク {a.get('tasks')}  {a.get('created_at')}")
    print(f"動いているアプリ: {len(apps)} 件")


@modal_app.command("stop")
def stop(app_ids: Optional[list[str]] = typer.Argument(None, help="止めるアプリの ID(省くと --all が要る)"),
         all_: bool = typer.Option(False, "--all", help="動いている gne のアプリを全部止める")):
    """アプリを止める(コンテナも止まり、課金も止まる)。止まったかは gne modal ps で確かめる。"""
    exe = _modal()
    ids = list(app_ids or []) or ([a["app_id"] for a in running_apps(exe)] if all_ else [])
    if not ids:
        raise typer.BadParameter("止めるアプリの ID か --all を指定してください")
    bad = [i for i in ids if not stop_app(exe, i)]
    raise typer.Exit(1 if bad else 0)


@modal_app.command("cost")
def cost(period: str = typer.Option("this month", "--for", help="期間(today / this month / last month など)")):
    """Modal の課金の合計(modal billing report)。月の上限は Modal の Usage & Billing で決める。"""
    out = subprocess.run([_modal(), "billing", "report", "--for", period, "--json"], capture_output=True, text=True)
    if out.returncode:
        sys.stderr.write(out.stderr[-500:])
        raise typer.Exit(out.returncode)
    rows = json.loads(out.stdout or "[]")
    total = sum(float(r.get("Cost", r.get("cost", 0)) or 0) for r in rows)
    print(f"{period}: ${total:.2f}")


@modal_app.command("hf-upload")
def hf_upload(paths: list[str] = typer.Argument(..., help="Volume の上のファイル(例 export/x-00001-of-00004.gguf)"),
              repo: str = typer.Option(..., "--repo", help="Hugging Face のリポジトリ(例 user/name)"),
              public: bool = typer.Option(False, "--public", help="公開リポジトリにする(既定は非公開)"),
              card: Optional[Path] = typer.Option(None, "--card", exists=True, help="モデルカード(README.md として上げる)"),
              dry_run: bool = typer.Option(False, "--dry-run", help="実行する modal コマンドを出すだけ")):
    """Volume の上のファイルを、手元を通さず Hugging Face に上げる。トークンは Modal のシークレット huggingface(HF_TOKEN)から読む
    (modal secret create huggingface HF_TOKEN=<トークン>)。上げたものは外から見えるので、名前とモデルカードを確かめてから使う。"""
    args = ["run", f"{UPLOAD_FILE}::main", "--repo", repo, "--paths-json", json.dumps([p.lstrip("/") for p in paths])]
    if public:
        args += ["--no-private"]
    if card:
        args += ["--card-path", str(card.resolve())]
    _call(args, dry_run)


@modal_app.command("ls")
def ls(path: str = typer.Argument("", help="Volume の上の場所"),
       models: bool = typer.Option(False, "--models", help="モデルの置き場を見る(既定は結果の置き場)"),
       dry_run: bool = typer.Option(False, "--dry-run", help="実行する modal コマンドを出すだけ")):
    """Volume の中を見る。-m に渡すパスは /models/<ここの表示>、コンテナの中の結果は /data/<ここの表示>。"""
    _call(["volume", "ls", MODELS_VOL if models else DATA_VOL, *([path] if path else [])], dry_run)


@modal_app.command("get")
def get(remote: str = typer.Argument(..., help="Volume の上の場所(例 gne/<model_id>)"),
        local: Path = typer.Argument(Path("."), help="持ち帰る先"),
        dry_run: bool = typer.Option(False, "--dry-run", help="実行する modal コマンドを出すだけ")):
    """結果を手元に持ち帰る。"""
    _call(["volume", "get", DATA_VOL, remote.lstrip("/"), str(local)], dry_run)
