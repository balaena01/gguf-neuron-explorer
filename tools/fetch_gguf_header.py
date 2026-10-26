"""GGUF のヘッダ(メタデータとテンソル情報)だけを HTTP の Range 要求で取る。重みは落とさない。
UI の構造描画(別冊 U0)を、大きなモデルで手元に重みを置かずに確かめるため。

    python tools/fetch_gguf_header.py URL OUT
"""

import struct
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gne.gguf_fast import FastGGUF  # noqa: E402


def fetch(url: str, out: Path, start_mb: int = 8, max_mb: int = 256) -> Path:
    size = start_mb
    while size <= max_mb:
        req = urllib.request.Request(url, headers={"Range": f"bytes=0-{size * 2**20 - 1}"})
        with urllib.request.urlopen(req, timeout=120) as r:      # noqa: S310
            out.write_bytes(r.read())
        try:
            FastGGUF(out)
            return out
        except (struct.error, IndexError, ValueError):
            size *= 2
    raise RuntimeError(f"{max_mb} MB までにヘッダが収まりませんでした")


if __name__ == "__main__":
    p = fetch(sys.argv[1], Path(sys.argv[2]))
    print(p, p.stat().st_size)
