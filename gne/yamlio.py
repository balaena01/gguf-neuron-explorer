"""対照集合・レシピなどを YAML(と、YAML としても読む JSON)で書く。

gne は対照集合・レシピ・ひな形を PyYAML で読む(JSON の対照集合も YAML として読む)。PyYAML は次の文字をそのまま書いた
ファイルを読めないか、別の文字に読む: タブ・改行以外の C0 制御文字、DEL と C1 制御文字(U+007F〜U+009F。MMLU の問題文に
よくある U+0093 など)、U+FFFE・U+FFFF、対になっていない代理(サロゲート)の符号位置(ここまで ReaderError)、行の区切りとして
読む U+0085・U+2028・U+2029(U+0085 は空白になる)。yaml.safe_dump は C1 制御文字などはエスケープするが、U+0085 は引用符なしか
一重引用符のまま書くので、読み戻すと空白になる。ここではどれも読み戻して同じ文字になるように書く。

- dump_yaml: yaml.safe_dump(allow_unicode、既定はキーの順のまま)と同じ見た目の YAML。行の区切りとして読まれる文字を含む
  文字列だけ二重引用符で書く(\\N・\\L・\\P にエスケープされる)。
- dump_json: YAML としても同じ中身に読める JSON(gne direction key-from-behave の <root>/keyed/<name>.json)。
- write_yaml: dump_yaml の中身を読み直して元の表と同じか確かめてから書く(違えば書かずに GneError)。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import yaml

_RAW_UNSAFE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u2028\u2029\ud800-\udfff\ufffe\uffff]")
_FOLDED = re.compile("[\x85\u2028\u2029]")          # yaml.safe_dump がそのまま書き、読むと別の文字になりうるもの


def dump_json(d) -> str:
    """表 → JSON の文字列(末尾に改行)。YAML として読んでも同じ中身になるように書く: ensure_ascii=False のまま(日本語・絵文字は
    そのまま)、YAML がそのまま読めない文字・別の文字に読む文字(_RAW_UNSAFE)だけを \\uXXXX にする。ensure_ascii=True に
    しないのは、BMP の外の文字(絵文字)が \\ud83c\\udf82 のような代理の対になり、PyYAML はそれを 1 文字に戻さず 2 つの代理の
    符号位置として読むから(質問の文字が壊れる)。JSON の中でこれらの文字が出るのは文字列の中だけで、その前の \\ は json.dumps
    が \\\\ にしているので、置き換えても JSON として正しく、json.loads でも同じ中身に読める。"""
    text = json.dumps(d, ensure_ascii=False, indent=2)
    return _RAW_UNSAFE.sub(lambda m: f"\\u{ord(m.group()):04x}", text) + "\n"


class _Dumper(yaml.SafeDumper):
    """yaml.safe_dump と同じ書き方で、行の区切りとして読まれる文字(U+0085・U+2028・U+2029)を含む文字列だけを二重引用符で
    書く。"""


def _represent_str(dumper, s: str):
    if _FOLDED.search(s):
        return dumper.represent_scalar("tag:yaml.org,2002:str", s, style='"')
    return yaml.SafeDumper.represent_str(dumper, s)


_Dumper.add_representer(str, _represent_str)


def dump_yaml(d, *, sort_keys: bool = False) -> str:
    """表 → YAML の文字列(yaml.safe_dump(d, allow_unicode=True, sort_keys=sort_keys) と同じ見た目で、読み戻すと同じ中身になる)。"""
    return yaml.dump(d, Dumper=_Dumper, allow_unicode=True, sort_keys=sort_keys)


def same(a, b) -> bool:
    """読み直した中身 a と元の表 b が同じか(NaN どうしは同じとみなす。YAML はリストとタプルを区別しないのでタプルはリストとして比べる)。"""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b or (isinstance(a, int | float) and isinstance(b, int | float)
                                             and not isinstance(a, bool) and not isinstance(b, bool) and a == b)


def check_yaml(text: str, d, where: str, code: str = "BAD_CONTRAST") -> None:
    """YAML の文字列 text を読み直して、元の表 d と同じ中身か確かめる(違えば GneError(code))。"""
    from .steer import GneError
    try:
        back = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise GneError(code, f"{where}: 書く YAML を読み直せません({e})。書きませんでした") from e
    if not same(back, d):
        raise GneError(code, f"{where}: 書く YAML を読み直すと中身が変わります。書きませんでした")


def write_yaml(path: str | Path, d, *, code: str = "BAD_CONTRAST", sort_keys: bool = False) -> str:
    """表 d を path に YAML(dump_yaml)で書く。書く中身を読み直して d と同じか確かめてから書く(違えば書かずに GneError(code)、
    前のファイルはそのまま)。戻り値は書いた文字列。"""
    p = Path(path)
    text = dump_yaml(d, sort_keys=sort_keys)
    check_yaml(text, d, p.name, code)
    p.write_text(text, encoding="utf-8")
    return text
