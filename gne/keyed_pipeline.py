"""鍵つきの書き込み(keyed、CHANGE-07)を、対照集合 1 つから「方向 → 鍵 → key-check」まで 1 回で回す。

UI の鍵のタブの「方向から key-check まで」、ジョブ direction_key_pipeline、MCP direction_key_pipeline が使う。
対照集合は heads_pipeline と同じく、表(UI の入力欄の JSON、configs/keyed/ のひな形)かファイルのパスで受け、表なら
<root>/contrasts/<name>.yaml に書いてから使う(ひな形の説明 about と設定 keyed は書かない)。

1. 方向: gne direction --contrast C --positions response と同じもの(A 側 = system_a・recipe_a、B 側 = system_b・recipe_b、返事の
   トークンでの残差の平均の差)。同じ名前の方向がもうあれば使い回し、rebuild_direction=True なら作り直す。
2. 鍵: gne direction key と同じ(Session.direction_key)。
3. key-check: 作成と反対側の subset(even ↔ odd)で、同じ対照集合・同じ返事で測る(gne direction key-check と同じ)。

表の最上位に任意で keyed: {layers, target, q_mode, subset, pseudo, max_tokens, alpha, direction, iv} を書ける(引数が優先)。alpha は
UI が「+ レシピ」の α の欄に入れる値で、ここでは結果の alpha_hint に返すだけ。direction は書き込む方向の名前(省くと対照集合の
name)で、別の対照集合で作った方向(CHiPS の方向など)を使い回すときに書く。その方向が無くてもこの対照集合からは作らず、作り直しもしない。
iv は鍵の作成と key-check で当てる介入(gne direction key --iv と同じ。保存したレシピの名前かパス、またはユニットの書き方のリスト)。
gne direction key-from-behave が作る JSON は direction・alpha・iv を書く。
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from pathlib import Path

import yaml

from .steer import GneError
from .yamlio import write_yaml

SETTINGS = ("layers", "target", "q_mode", "subset", "pseudo", "max_tokens", "alpha", "direction", "iv", "same_reply")   # 表の keyed に書ける設定
NOT_ARGS = ("alpha", "direction", "same_reply")        # 鍵の作成の引数にしない設定(alpha は「+ レシピ」の α、direction は run が読む)
OPPOSITE = {"even": "odd", "odd": "even", "all": "all"}


def _settings(raw, where: str) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise GneError("BAD_CONTRAST", f"{where}: keyed は設定の表です({', '.join(SETTINGS)})")
    bad = sorted(set(raw) - set(SETTINGS))
    if bad:
        raise GneError("BAD_CONTRAST", f"{where}: keyed に {', '.join(bad)} は使えません({', '.join(SETTINGS)})")
    d = raw.get("direction")
    if d is not None and not (isinstance(d, str) and re.fullmatch(r"[\w.\-]+", d)):
        raise GneError("BAD_CONTRAST", f"{where}: keyed.direction は方向の名前です(英数字・_・-・. だけ。{d!r})")
    iv = raw.get("iv")
    if iv is not None and not (isinstance(iv, str) and iv or isinstance(iv, list) and iv and all(isinstance(u, str) for u in iv)):
        raise GneError("BAD_CONTRAST", f"{where}: keyed.iv は保存したレシピの名前かパス、またはユニットの書き方のリストです({iv!r})")
    return dict(raw)


def clean_spec(spec: dict) -> tuple[dict, dict]:
    """表 → (対照集合に書く中身, keyed の設定)。ひな形の説明 about と keyed は対照集合に入れない。"""
    if not isinstance(spec, dict):
        raise GneError("BAD_ARGS", "spec は対照集合の表です(name・user_prompts・positions・dont_touch など)")
    d = dict(spec)
    d.pop("about", None)
    settings = _settings(d.pop("keyed", None), "spec")
    name = str(d.get("name") or "")
    if not re.fullmatch(r"[\w.\-]+", name):
        raise GneError("BAD_ARGS", f"名前 {name!r} は英数字・_・-・. だけにしてください(spec の name。対照集合・方向・鍵のファイル名になります)")
    d["name"] = name
    return d, settings


def save_spec(session, spec: dict) -> tuple[Path, dict]:
    """表を <root>/contrasts/<name>.yaml に書き、鍵の対照集合として読めるか確かめる(読めなければ消してエラー)。
    戻り値 (パス, keyed の設定)。同じ名前のファイルは書き換える(heads_pipeline と同じ)。YAML は yaml.safe_dump と同じ見た目で、
    行の区切りとして読まれる文字(U+0085 など。MMLU の問題文によくある)を含む文字列も読み戻すと同じ文字になるように書く
    (yamlio.write_yaml。書く中身を読み直して同じか確かめてから書く)。"""
    from .keyed_build import read_key_spec
    from .reverse import check_questions
    d, settings = clean_spec(spec)
    for key in ("user_prompts", "prompts_a", "prompts_b"):
        check_questions(d.get(key), key)
    path = session.store.contrast_dir() / f"{d['name']}.yaml"
    write_yaml(path, d)
    try:
        read_key_spec(path)
    except GneError:
        path.unlink(missing_ok=True)
        raise
    return path, settings


def file_settings(path: str | Path) -> dict:
    """ファイルで渡した対照集合の keyed の設定(configs/keyed/ のひな形をパスで渡したとき)。読めなければ空。"""
    try:
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return _settings(d.get("keyed"), str(path)) if isinstance(d, dict) else {}


def resolve(session, contrast) -> tuple[Path, dict]:
    """対照集合(表かパス)→ (パス, keyed の設定)。"""
    if isinstance(contrast, dict):
        return save_spec(session, contrast)
    if contrast is None or str(contrast) == "":
        raise GneError("BAD_ARGS", "対照集合(contrast のパスか spec の表)を渡してください")
    p = Path(str(contrast))
    if not p.exists():
        raise GneError("FILE_NOT_FOUND", f"対照集合 {p} がありません")
    return p, file_settings(p)


def apply_settings(settings: dict, params: dict, keys=SETTINGS) -> dict:
    """keyed の設定を、渡していない引数(None か無い)の既定にする。alpha・direction は引数にしない(NOT_ARGS)。"""
    out = dict(params)
    for k in keys:
        if k not in NOT_ARGS and out.get(k) is None and settings.get(k) is not None:
            out[k] = settings[k]
    return out


class _Stages:
    """段ごとの進み具合を 1 本の進み具合にまとめる(段 i の done / total を全体の i/n 〜 (i+1)/n に置く)。"""

    def __init__(self, progress: Callable | None, labels: list[str]):
        self.progress, self.labels, self.i = progress, labels, 0

    def start(self, i: int) -> Callable | None:
        self.i = i
        if self.progress is None:
            return None
        n, lab = len(self.labels), self.labels[i]

        def f(done: int, total: int, stage: str = "") -> None:
            frac = done / total if total else 0.0
            self.progress(int(1000 * (i + min(1.0, max(0.0, frac)))), 1000 * n, f"{i + 1}/{n} {lab}" + (f": {stage}" if stage else ""))
        f(0, 1, "")
        return f


def direction_plan(session, spec, settings: dict, direction: str | None = None,
                   rebuild_direction: bool = False) -> tuple[str, bool, object | None, bool]:
    """書き込む方向の決め方: (方向の名前, 表の keyed.direction の方向か, もうある方向(無ければ None), 作るか)。
    direction(引数)> 表の keyed.direction > 対照集合の name。表の keyed.direction の方向は別の対照集合で作ったものなので、
    無ければ作らずにエラー、作り直しもしない。"""
    cname = spec.contrast.name
    fixed = not direction and bool(settings.get("direction"))           # 表の keyed.direction(別の対照集合で作った方向)
    dname = direction or settings.get("direction") or cname
    if not re.fullmatch(r"[\w.\-]+", dname):
        raise GneError("BAD_ARGS", f"方向の名前 {dname!r} は英数字・_・-・. だけにしてください")
    try:
        existing = session.direction_set(dname)
    except (GneError, OSError, ValueError, KeyError):
        existing = None
    if fixed and (existing is None or rebuild_direction):
        raise GneError("BAD_ARGS", f"表の keyed.direction の方向 {dname} " + ("は作り直しません" if existing is not None else "がありません")
                       + "(別の対照集合で作った方向なので、この対照集合からは作りません)",
                       "その方向を作った対照集合で gne direction を回すか、keyed.direction を消して direction で方向を渡してください")
    return dname, fixed, existing, existing is None or bool(rebuild_direction)


def make_direction(session, path: Path, dname: str, existing, response_tokens: int = 96, same_reply: bool = False) -> dict:
    """方向を作る(gne direction --contrast C --positions response と同じ)。結果の direction の記録。"""
    from .direction import load_contrast
    ts = time.perf_counter()
    try:
        load_contrast(path)                                       # 方向を作れない対照集合(A と B が同じ)は先に断る
    except GneError as e:
        raise GneError(e.code, f"方向を作れません: {e.message}",
                       "方向は A 側(system_a か recipe_a)と B 側の返事の差で作ります。書き込みたい振る舞いを頼む system_a を"
                       "対照集合に書くか、同じ名前の方向を先に作ってください") from e
    d = session.direction(contrast=path, name=dname, positions="response", response_tokens=int(response_tokens),
                          same_reply=bool(same_reply))
    return {"name": dname, "reused": False, "rebuilt": existing is not None, "path": d["path"], "same_reply": bool(same_reply),
            "positions": d["positions"], "n_rows": d["n_rows"], "seconds": time.perf_counter() - ts}


def reused_direction(existing, spec, dname: str, fixed: bool) -> tuple[dict, list[str]]:
    """もうある方向を使い回すときの記録と警告(中身の違う対照集合・positions response 以外で作った方向)。"""
    warnings: list[str] = []
    m = existing.meta
    pos = m.get("positions")
    if not fixed and m.get("contrast_sha") and m.get("contrast_sha") != spec.contrast.sha:
        warnings.append(f"方向 {dname} は、いまの対照集合とは中身の違うファイルから作られています(作り直すなら"
                        "「方向を作り直す」、MCP は rebuild_direction=True)")
    if not fixed and pos and pos != "response":
        warnings.append(f"方向 {dname} は positions {pos} で作られています(この一括は response で作ります。"
                        "そろえるなら作り直す)")
    return ({"name": dname, "reused": True, "rebuilt": False, "path": str(existing.path), "same_reply": bool(m.get("same_reply")),
             "positions": pos, "n_rows": {"A": int(m.get("n_rows_A") or 0), "B": int(m.get("n_rows_B") or 0)},
             "seconds": 0.0, "from_settings": fixed}, warnings)


def run(session, contrast, *, layers=None, name: str | None = None, direction: str | None = None,
        rebuild_direction: bool = False, response_tokens: int = 96, same_reply: bool | None = None, check: bool = True,
        check_subset: str | None = None,
        compact: bool = False, progress: Callable | None = None, **build_kw) -> dict:
    """対照集合(表かパス)から、方向(positions response)→ 鍵 → 作成と反対側の subset での key-check を続けて回す。
    layers は必須(引数か表の keyed.layers)。name は鍵の名前(省くと対照集合の name)、direction は方向の名前(省くと表の
    keyed.direction、無ければ対照集合の name。keyed.direction の方向は作らず、作り直さない)。
    build_kw は Session.direction_key の引数(target・positions・q_mode・subset・pseudo・max_tokens・replies_a / replies_b・iv など)。
    compact なら key-check の例ごとの ρ・ヒストグラム・比較用の鍵ごとの値を落とす(MCP 向け)。"""
    from .jobs import _layers
    from .keyed_build import read_key_spec
    t0 = time.perf_counter()
    progress = progress if progress is not None else session.progress
    path, settings = resolve(session, contrast)
    kw = apply_settings(settings, {"layers": layers, **build_kw})
    Ls = _layers(kw.pop("layers", None))
    if not Ls:
        raise GneError("BAD_ARGS", "鍵の層(layers)を指定してください(例 [8, 14, 20] か \"8,14,20\"、または表の keyed.layers)")
    spec = read_key_spec(path)
    cname = spec.contrast.name
    kname = name or cname
    if not re.fullmatch(r"[\w.\-]+", kname):
        raise GneError("BAD_ARGS", f"鍵の名前 {kname!r} は英数字・_・-・. だけにしてください")
    dname, fixed, existing, build_dir = direction_plan(session, spec, settings, direction, rebuild_direction)
    same = bool(settings.get("same_reply")) if same_reply is None else bool(same_reply)      # 方向の作り方(既定は今までどおり)
    warnings: list[str] = []
    labels = (["方向"] if build_dir else []) + ["鍵"] + (["key-check"] if check else [])
    st = _Stages(progress, labels)
    i = 0
    out: dict = {"name": kname, "contrast": cname, "contrast_path": str(path), "layers": Ls,
                 "alpha_hint": settings.get("alpha"), "stages": labels}
    prev = session.progress
    try:
        if build_dir:                                                     # 1. 方向(gne direction --positions response と同じ)
            session.progress = st.start(i)
            i += 1
            out["direction"] = make_direction(session, path, dname, existing, response_tokens, same)
        else:
            out["direction"], w = reused_direction(existing, spec, dname, fixed)
            warnings += w
            if not fixed and bool(existing.meta.get("same_reply")) != same:
                warnings.append(f"方向 {dname} は{'同じ返事を両方の条件で読ませる作り方' if existing.meta.get('same_reply') else '今までの作り方(それぞれ自分の返事)'}で"
                                "作られています(頼まれた作り方と違います。そろえるなら作り直す)")
        ts = time.perf_counter()                                          # 2. 鍵
        f = st.start(i)
        i += 1
        session.progress = f
        r = session.direction_key(path, name=kname, layers=Ls, progress=f, **kw)
        out["key"] = r
        out["key_seconds"] = time.perf_counter() - ts
        if check:                                                         # 3. 作成と反対側の subset で key-check
            ts = time.perf_counter()
            f = st.start(i)
            session.progress = f
            sub = check_subset or OPPOSITE.get(r.get("subset") or kw.get("subset") or "even", "odd")
            ck = {k: kw[k] for k in ("replies_a", "replies_b", "iv") if kw.get(k) is not None}
            c = session.direction_key_check(kname, contrast=path, subset=sub, progress=f, **ck)
            if compact:
                for row in c["layers"]:
                    row.pop("values", None)
                    row.pop("hist", None)
                    for k in ("pseudo_auc", "pseudo_auc_token", "pseudo_selectivity"):
                        if isinstance(row.get(k), dict):
                            row[k].pop("values", None)
            out["check"] = c
            out["check_seconds"] = time.perf_counter() - ts
    finally:
        session.progress = prev
    out["warnings"] = warnings
    out["seconds"] = time.perf_counter() - t0
    return out
