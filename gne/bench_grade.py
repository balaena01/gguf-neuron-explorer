"""MATH-500 の採点(math-verify)。signal.alarm で時間を切るのでスレッドでは動かない。
子プロセスの主スレッドで走らせる: 標準入力に [[正解, 答え], ...] の JSON、標準出力に [true / false, ...] の JSON。"""
from __future__ import annotations

import json
import sys


def main() -> None:
    from math_verify import parse, verify
    rows = json.load(sys.stdin)
    out = []
    for gold, pred in rows:
        try:
            out.append(bool(verify(parse(f"\\boxed{{{gold}}}"), parse(f"\\boxed{{{pred}}}"))))
        except Exception:  # noqa: BLE001  1 問の失敗は不正解に数える
            out.append(False)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
