"""Modal で gne のコマンドを GPU で回す。手元に載らない大きいモデル用(docs/modal.md)。

ふだんは gne modal の下のコマンドから使う。このファイルは modal と標準ライブラリだけを読むので、gne の依存を入れていない
環境の modal コマンドからも実行できる(modal run gne/remote/modal_app.py::main --argv-json '["info", "-m", "/models/x.gguf"]')。

コンテナに載せるのは git で管理している gne/ と configs/ だけ。追跡していないファイル(手元だけの質問集など)は載らない。
ほかのファイルは gne modal put で名指ししたものだけが上がる。
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import modal

MODELS, DATA = "/models", "/data"
GPU = ["A100-80GB", "H100"]             # 80 GB。候補の順に空いているものを使う
CPU = 8
MEMORY = 32768                          # MiB。重みは VRAM に載せるので、ファイルの大きさまでは要らない
TIMEOUT = 6 * 3600                      # 秒。止め忘れの上限。Modal の上限は 24 時間
CUDA = "12.8.1"                         # ホストの CUDA より新しくしない
UI_PORT = 8791                          # コンテナの中の gne serve。127.0.0.1 だけで待ち受ける
UI_TIMEOUT = 2 * 3600                   # 秒
UI_IDLE = 15 * 60                       # 秒。SSH の接続がこれだけ無ければ止める(止め忘れの課金を防ぐ)


def tracked_only(top: Path) -> Callable[[Path], bool]:
    """add_local_dir の ignore。top からの相対パスを受け取り、git で追跡していないものは True(載せない)。"""
    out = subprocess.run(["git", "-C", str(top), "ls-files", "-z"], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{top} が git の作業ツリーではありません。gne modal はリポジトリを clone した場所から使ってください")
    keep = {Path(x) for x in out.stdout.split("\0") if x}
    dirs = {p for f in keep for p in f.parents}
    return lambda rel: Path(rel) not in keep and Path(rel) not in dirs


def run_gne(argv: list[str], threads: int | None = None, script: str = "") -> int:
    """gne のコマンド(script があれば Python のスクリプト)を別プロセスで実行し、終了コードを返す。出力はそのまま Modal のログに流れる。"""
    env = dict(os.environ)
    if threads:                         # 指定しないと gne と numpy はホストの全コア数からスレッド数を決め、CPU の課金が増える
        n = str(int(threads))
        env.update(GNE_THREADS=n, OMP_NUM_THREADS=n, OPENBLAS_NUM_THREADS=n)
    head = [script] if script else ["-c", "from gne.cli import app; app()"]
    return subprocess.call([sys.executable, *head, *argv], env=env)


def sshd_config(host_key: str, authorized: str, port: int = UI_PORT) -> str:
    """鍵だけで入れて、UI のポートへの転送しかできない SSH サーバの設定。"""
    return f"""Port 22
HostKey {host_key}
AuthorizedKeysFile {authorized}
PidFile /run/sshd.pid
PubkeyAuthentication yes
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin prohibit-password
UsePAM no
StrictModes no
PermitTTY no
X11Forwarding no
AllowAgentForwarding no
AllowTcpForwarding local
PermitOpen 127.0.0.1:{port}
ForceCommand /bin/false
"""


def ssh_connections(proc_net_tcp: str, port: int = 22) -> int:
    """/proc/net/tcp の中身から、ローカルのポート port の確立済みの接続を数える。"""
    n = 0
    for line in proc_net_tcp.splitlines()[1:]:
        f = line.split()
        if len(f) > 3 and f[3] == "01" and int(f[1].split(":")[1], 16) == port:
            n += 1
    return n


def gpu_info() -> str:
    """使っている GPU の名前と VRAM。GPU が無ければ空。"""
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def llama_libs() -> list[str]:
    """CUDA 版でビルドできていれば libggml-cuda が入っている。"""
    code = ("import importlib.util, pathlib; d = pathlib.Path(importlib.util.find_spec('llama_cpp').origin).parent; "
            "print(*sorted(p.name for p in d.rglob('*.so*')))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return r.stdout.split() or [f"読めません: {r.stderr.strip()[-300:]}"]


app = modal.App("gne")
models = modal.Volume.from_name("gne-models", create_if_missing=True, version=2)
data = modal.Volume.from_name("gne-data", create_if_missing=True, version=2)

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA}-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("build-essential", "cmake", "git")
    # add_python の Python は clang で作られていて、指定しないと clang を探す
    .uv_pip_install("llama-cpp-python==0.3.36", env={"CC": "gcc", "CXX": "g++", "CMAKE_ARGS": "-DGGML_CUDA=on",
                                                     "CMAKE_BUILD_PARALLEL_LEVEL": "8"})
    .uv_pip_install("gguf==0.19.0", "numpy==2.5.4", "typer==0.27.3", "pyyaml==6.0.3", "jinja2==3.1.6", "pydantic==2.14.1",
                    "safetensors==0.8.0")
    .apt_install("openssh-server")
    .uv_pip_install("fastapi==0.162.0", "uvicorn==0.54.0", "umap-learn==0.5.12")
    .env({"PYTHONPATH": "/gne", "GNE_ROOT": f"{DATA}/gne"})
)
_VOLUMES = {MODELS: models.read_only(), DATA: data}
if modal.is_local():                    # コンテナの中ではこのファイルは別の場所にあり、ソースはもう載っている
    _repo = Path(__file__).resolve().parents[2]
    image = (image.add_local_dir(_repo / "gne", "/gne/gne", ignore=tracked_only(_repo / "gne"))
             .add_local_dir(_repo / "configs", "/gne/configs", ignore=tracked_only(_repo / "configs")))


def _run(argv: list[str], threads: int, script: str = "") -> int:
    """横取り(preemption)されると Modal が同じ引数で最初から実行し直す。heads pipeline や sweep は保存済みの続きから動く。"""
    print("GPU:", gpu_info() or "なし", "| llama_cpp のライブラリ:", llama_libs(), flush=True)
    try:
        code = run_gne(argv, threads, script)
    finally:
        data.commit()
    if code:
        raise RuntimeError(f"{script or 'gne ' + ' '.join(argv[:2])} が終了コード {code} で止まりました")
    return code


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY, timeout=TIMEOUT, volumes=_VOLUMES)
def run(argv: list[str], threads: int = CPU, script: str = "") -> int:
    return _run(argv, threads, script)


def _serve(model: str, client_pub: str, host_key: str, idle: int, threads: int) -> None:
    """gne serve を 127.0.0.1 で動かし、SSH だけを外に出す。手元の gne modal serve が鍵で入ってポートを転送する。"""
    work = Path(tempfile.mkdtemp())
    key, authorized, cfg = work / "host_key", work / "authorized_keys", work / "sshd_config"
    key.write_text(host_key)
    key.chmod(0o600)
    authorized.write_text(client_pub + "\n")
    cfg.write_text(sshd_config(str(key), str(authorized)))
    Path("/run/sshd").mkdir(exist_ok=True)
    print("GPU:", gpu_info() or "なし", flush=True)
    sshd = subprocess.Popen(["/usr/sbin/sshd", "-D", "-e", "-f", str(cfg)])
    env = {**os.environ, "GNE_THREADS": str(threads), "OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads)}
    ui = subprocess.Popen([sys.executable, "-c", "from gne.cli import app; app()", "serve", "-m", model, "--port", str(UI_PORT)], env=env)
    try:
        with modal.forward(22, unencrypted=True) as tunnel:
            host, port = tunnel.tcp_socket
            print(f"GNE_TUNNEL {host} {port}", flush=True)
            seen = last_commit = time.monotonic()
            while ui.poll() is None:
                time.sleep(15)
                now = time.monotonic()
                if ssh_connections(Path("/proc/net/tcp").read_text()):
                    seen = now
                elif now - seen > idle:
                    print(f"SSH の接続が {idle} 秒ないので止めます", flush=True)
                    break
                if now - last_commit > 300:     # UI で保存したレシピなどを Volume に残す
                    data.commit()
                    last_commit = now
    finally:
        ui.send_signal(signal.SIGINT)
        try:
            ui.wait(30)
        except subprocess.TimeoutExpired:
            ui.kill()
        sshd.terminate()
        data.commit()


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY, timeout=UI_TIMEOUT, volumes=_VOLUMES)
def serve(model: str, client_pub: str, host_key: str, idle: int = UI_IDLE, threads: int = CPU) -> None:
    _serve(model, client_pub, host_key, idle, threads)


@app.local_entrypoint()
def serve_main(model: str, gpu: str = "", cpu: float = CPU, memory: int = MEMORY, timeout: int = UI_TIMEOUT, idle: int = UI_IDLE) -> None:
    """鍵は引数に書かず環境変数で受け取る(手元のプロセスの一覧に出さない)。"""
    opts = {"cpu": cpu, "memory": memory, "timeout": timeout}
    if gpu:
        opts["gpu"] = gpu
    serve.with_options(**opts).remote(model, os.environ["GNE_SSH_PUB"], os.environ["GNE_SSH_HOST_KEY"], idle, int(cpu))


@app.local_entrypoint()
def main(argv_json: str, gpu: str = "", cpu: float = CPU, memory: int = MEMORY, timeout: int = TIMEOUT, script: str = "") -> None:
    opts = {"cpu": cpu, "memory": memory, "timeout": timeout}
    if gpu:
        opts["gpu"] = gpu
    run.with_options(**opts).remote(json.loads(argv_json), threads=int(cpu), script=script)
