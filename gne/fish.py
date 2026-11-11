"""FiSH(Formula-instantiated Scene-gated Hidden units、docs/fish.md)を順に作る段(Web UI の FiSH のタブとジョブが使う)。

中身は今ある道具をつなぐだけで、新しい計算はしない:
1. 方向: Session.direction(positions response、same_reply・segments・stack_layers)。今までの作り方の方向も同じ区切りで作って、
   層 × 区切りごとの長さと向きの近さ(cos)を表にする。
2. ニューロンの表: Session.nw_neurons(働きの小さい順)。既定の候補は表の上から必要な個数。
3. 鍵: 区切りごとの位置の鍵(target mlpin、positions pos:<範囲>)、場面の鍵(鍵 2 つの形)、ほぼ 1 を返す鍵(上限つきの形)。
   比較用の鍵との比べは、鍵に保存した作成時の例での値(甘い。作成に使った例なので。実際に足す量は check で見る)。
4. ユニット: 層 × 区切りごとに 1 個(上限つきの形は 2 個組)。
5. 確かめ: nwrite.verify(式どおりか)と nwrite.check(場面・位置ごとに実際に足した量)。
層・強さは使う人が決める(ここでは選ばない)。強さは区切りごとに変えられる(alpha に "1,1,0.8" か [1, 1, 0.8]。1 つなら全区切りに同じ値)。
重みに入れる前に試す(try_step): 区切りつきの方向を、狙う場面の質問の返事の位置どおりに、層の出力へ直接足して生成する
(control vector を生成の途中で付け替える。ニューロンも鍵も使わない)。層・区切りごとの強さを、作る前に返事で比べるのに使う。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .steer import GneError

FORMS = ("one", "two", "cap")                 # 鍵 1 つ / 鍵 2 つ(場面 かつ 位置)/ 上限つきの 2 個組
PER_SEGMENT = {"one": 1, "two": 1, "cap": 2}  # 区切り 1 つに要るニューロンの数
SETTINGS = ("layers", "segments", "form", "alpha", "same_reply", "stack", "pseudo", "max_tokens", "response_tokens", "iv", "rekey")
DEFAULTS = {"segments": "0,1-3,4-", "form": "cap", "alpha": 0.9, "same_reply": True, "stack": True, "pseudo": 4,
            "max_tokens": 96, "response_tokens": 96}


def _form(form: str) -> str:
    if form not in FORMS:
        raise GneError("BAD_ARGS", f"形は {' / '.join(FORMS)} です({form!r})",
                       "one = 鍵 1 つ、two = 鍵 2 つ(場面 かつ 位置)、cap = 上限つきの 2 個組")
    return form


def _layers(layers) -> list[int]:
    from .jobs import _layers as parse
    Ls = parse(layers)
    if not Ls:
        raise GneError("BAD_ARGS", "層(layers)を指定してください(例 \"11-23\" か [11, 18, 23])", "層は使う人が決めます")
    return sorted({int(L) for L in Ls})


def before_mode(segments) -> bool:
    """書く位置が「正規表現に一致する文字の直前」(before:<正規表現>)か。返事の位置の区切り(0,1-3,4-)でなければ True。"""
    return isinstance(segments, str) and segments.strip().startswith("before:") or (
        isinstance(segments, (list, tuple)) and len(segments) == 1 and str(segments[0]).strip().startswith("before:"))


def _segments(segments) -> list[str]:
    """書く位置の並び。返事の位置の区切り("0,1-3,4-" → ["0", "1-3", "4-"])か、文字の直前("before:<正規表現>" → その 1 つ。
    keyed write の positions と同じ書き方。正規表現の中のカンマでは分けない)。"""
    from .direction import parse_segments, segment_text
    if before_mode(segments):
        from .keyed_build import parse_positions
        text = (segments if isinstance(segments, str) else str(segments[0])).strip()
        return [parse_positions(text).text]
    segs = [segment_text(x) for x in parse_segments(segments or DEFAULTS["segments"])]
    if not segs:
        raise GneError("BAD_ARGS", "位置の区切りがありません", "0,1-3,4- のように書きます")
    return segs


def alphas_of(alpha, n_seg: int) -> list[float]:
    """強さ → 区切りごとの強さのリスト。数 1 つ(0.9・"0.9"・[0.9])は全部の区切りに同じ値。区切りごとに変えるなら、区切りの順に
    "1,1,0.8" か [1, 1, 0.8](数は区切りの数と同じにする)。"""
    if alpha is None:
        alpha = DEFAULTS["alpha"]
    try:
        if isinstance(alpha, str):
            xs = [float(x) for x in alpha.replace("/", ",").replace(" ", ",").split(",") if x.strip()]
        elif isinstance(alpha, (list, tuple)):
            xs = [float(x) for x in alpha]
        else:
            xs = [float(alpha)]
    except (TypeError, ValueError):
        raise GneError("BAD_ARGS", f"強さ {alpha!r} を読めません", "0.9 か、区切りごとに 1,1,0.8 のように書きます") from None
    if not xs or not all(np.isfinite(x) for x in xs):
        raise GneError("BAD_ARGS", f"強さ {alpha!r} を読めません", "0.9 か、区切りごとに 1,1,0.8 のように書きます")
    if len(xs) == 1:
        return xs * int(n_seg)
    if len(xs) != int(n_seg):
        raise GneError("BAD_ARGS", f"強さが {len(xs)} 個で、区切りの数 {n_seg} と合いません",
                       "1 つだけ書けば全部の区切りに同じ値、区切りごとに変えるなら区切りと同じ数を書きます(例 区切り 0,1-3,4- に 1,1,0.8)")
    return xs


def alpha_out(al: list[float]):
    """結果に入れる強さ: 全部の区切りが同じ値なら数 1 つ、違えば区切りごとのリスト。"""
    return al[0] if len(set(al)) == 1 else list(al)


def resolve(session, contrast=None, spec: dict | None = None) -> tuple[str, dict, str]:
    """対照集合(パスか、UI の入力欄の表)→ (パス, 表の fish の設定, 対照集合の名前)。表は <root>/contrasts/<name>.yaml に書く
    (fish・about・keyed は対照集合に入れない)。"""
    from .keyed_build import read_key_spec
    if (contrast is None) == (spec is None):
        raise GneError("BAD_ARGS", "対照集合は contrast(パス)か spec(表)のどちらかで渡してください")
    settings: dict = {}
    if spec is not None:
        if not isinstance(spec, dict):
            raise GneError("BAD_ARGS", "spec は対照集合の表です(name・system_a・user_prompts・dont_touch など)")
        d = dict(spec)
        raw = d.pop("fish", None)
        if raw is not None:
            if not isinstance(raw, dict) or set(raw) - set(SETTINGS):
                raise GneError("BAD_ARGS", f"fish に書けるのは {', '.join(SETTINGS)} です")
            settings = dict(raw)
        from .keyed_pipeline import save_spec
        path, _ = save_spec(session, d)
        contrast = str(path)
    ks = read_key_spec(Path(str(contrast)))
    if not ks.dont_touch:
        session.warn("対照集合に dont_touch(書き込みたくない場面の質問の群)がありません。ほかの場面でも鍵が反応しやすくなります")
    return str(contrast), settings, ks.contrast.name


def key_names(base: str, n_seg: int, form: str) -> dict:
    """鍵の名前: 区切りごとの位置の鍵 <base>_p<i>、場面の鍵 <base>_scene(鍵 2 つの形)、ほぼ 1 を返す鍵 <base>_one(上限つき)。"""
    _form(form)
    return {"position": [f"{base}_p{i}" for i in range(n_seg)], "scene": f"{base}_scene" if form == "two" else None,
            "one": f"{base}_one" if form == "cap" else None}


def needed(n_seg: int, form: str) -> int:
    """層 1 つに要るニューロンの数。"""
    return int(n_seg) * PER_SEGMENT[_form(form)]


def default_neurons(usage: dict, n_seg: int, form: str) -> dict[int, list[int]]:
    """既定の候補: 表(働きの小さい順)の上から、層ごとに必要な個数。"""
    n = needed(n_seg, form)
    out = {}
    for row in usage["layers"]:
        js = [int(r["neuron"]) for r in row["rows"][:n]]
        if len(js) < n:
            raise GneError("BAD_ARGS", f"L{row['layer']}: 表のニューロンが {len(js)} 個で、必要な {n} 個に足りません", "top を増やしてください")
        out[int(row["layer"])] = js
    return out


def units(layers, n_seg: int, neurons: dict, form: str, direction: str, base: str, alpha=0.9, whole: bool = False) -> list[str]:
    """層 × 区切りごとのユニット(書き方の文字列)。neurons は {層: [番号, …]}(層ごとに needed 個。区切り 0 から順に使う)。
    alpha は強さ(数 1 つなら全部の区切りに同じ値。"1,1,0.8" か [1, 1, 0.8] なら区切りごと。alphas_of)。
    whole(書く位置が before:<正規表現> のとき): 方向は区切りに分けず、direction をそのまま使う。"""
    _form(form)
    al = alphas_of(alpha, n_seg)
    names = key_names(base, n_seg, form)
    per = PER_SEGMENT[form]
    out = []
    for L in _layers(layers):
        js = neurons.get(L, neurons.get(str(L)))
        js = [int(j) for j in (js or [])]
        if len(js) < n_seg * per or len(set(js)) != len(js):
            raise GneError("BAD_ARGS", f"L{L}: ニューロンは重ならない {n_seg * per} 個を選んでください(いま {len(js)} 個)",
                           f"区切り {n_seg} 個 × {per} 個({form})")
        for i in range(n_seg):
            pk = names["position"][i]
            head = f"{L}:N{js[i * per]}" + (f"+N{js[i * per + 1]}" if form == "cap" else "")
            keys = f"#{names['scene']}&{pk}" if form == "two" else (f"#{pk}&{names['one']}" if form == "cap" else f"#{pk}")
            out.append(f"{head}:nw:a{al[i]:g}@{direction}{'' if whole else f'.seg{i}'}{keys}")
    return out


def unit_rows(session, specs: list[str]) -> list[dict]:
    """ユニットの書き方 → UI のレシピの行(recipe.js の nw の行と同じ形)。"""
    from .steer import parse_unit
    us = [parse_unit(str(x)) for x in specs]            # 鍵・方向がまだ無くても行にできるように、書き方だけを読む
    return [{"layer": u.layer, "neuron": u.neuron, "pair": u.pair, "mode": "nw", "alpha": float(u.k), "direction": u.direction,
             "key": u.key, "key2": u.key2, "s": u.s, "cap": u.cap} for u in us]


def direction_step(session, contrast: str, *, name: str, layers=None, segments=None, same_reply: bool = True, stack: bool = True,
                   response_tokens: int = 96, compare: bool = True) -> dict:
    """方向を作り、層 × 区切りごとの長さと、今までの作り方(それぞれ自分の返事を読む・重ねない)との向きの近さを返す。
    stack なら、下の層に書いた後の残りを上の層の方向にする(layers の層に重ねて書く前提)。"""
    t0 = time.perf_counter()
    Ls = _layers(layers)
    segs = _segments(segments)
    whole = before_mode(segs)                    # 文字の直前に書く: 方向は返事の全部の位置の平均(区切らない・重ねない)
    if whole:
        stack = False
    r = session.direction(contrast=contrast, name=name, positions="response", same_reply=bool(same_reply),
                          segments=None if whole else segs, stack_layers=Ls if stack else None, response_tokens=int(response_tokens))
    ds = session.direction_set(name)
    old = None
    old_name = f"{name}_old"
    if compare and (same_reply or stack):
        session.direction(contrast=contrast, name=old_name, positions="response", segments=None if whole else segs,
                          response_tokens=int(response_tokens))
        old = session.direction_set(old_name)
    vec = (lambda x, i, L: x.raw[L]) if whole else (lambda x, i, L: x.seg[i][L])
    rows = []
    for L in Ls:
        for i, sg in enumerate(segs):
            d = np.asarray(vec(ds, i, L), dtype=np.float64)
            n = float(np.linalg.norm(d))
            cos = None
            if old is not None:
                o = np.asarray(vec(old, i, L), dtype=np.float64)
                no = float(np.linalg.norm(o))
                cos = float(d @ o / (n * no)) if n > 0 and no > 0 else None
            rows.append({"layer": L, "segment": i, "positions": sg, "direction": name if whole else f"{name}.seg{i}", "norm": n,
                         "cos_old": cos, "norm_old": None if old is None else float(np.linalg.norm(vec(old, i, L)))})
    return {"name": name, "segments": segs, "layers": Ls, "same_reply": bool(same_reply), "stack": bool(stack),
            "method": r.get("method"), "compare": old_name if old is not None else None, "rows": rows,
            "n_prompts": r.get("n_prompts"), "seconds": time.perf_counter() - t0}


def _build_stats(session, name: str) -> dict:
    """鍵に保存した作成時の例での、鍵と比較用の鍵の比べ(層ごと)。作成に使った例なので甘い値。"""
    try:
        d = session.direction_key_check(name)
    except GneError:
        return {}
    out = {}
    for r in d.get("layers", []):
        ps = r.get("pseudo_selectivity") or {}
        out[int(r["layer"])] = {"selectivity": r.get("selectivity"), "pseudo_p95": ps.get("p95"), "auc": r.get("auc"), "beats": r.get("beats_pseudo")}
    return out


def keys_step(session, contrast: str, *, base: str, layers=None, segments=None, form: str = "cap", pseudo: int = 4,
              max_tokens: int = 96, subset: str = "all", iv=None, seed: int = 0) -> dict:
    """区切りごとの位置の鍵と、形に要る鍵(鍵 2 つの形は場面の鍵、上限つきの形はほぼ 1 を返す鍵)をまとめて作る。
    iv(レシピの名前・パスか、ユニットの書き方のリスト)を渡すと、それを当てた状態で作る。"""
    t0 = time.perf_counter()
    Ls = _layers(layers)
    segs = _segments(segments)
    names = key_names(base, len(segs), _form(form))
    rows, notes = [], []

    def add(kind, positions, r):
        st = _build_stats(session, r["name"]) if kind != "one" else {}
        for x in r["layers"]:
            L = int(x["layer"])
            s = st.get(L, {})
            sel = s.get("selectivity", x.get("build_selectivity"))
            p95 = s.get("pseudo_p95", x.get("pseudo_selectivity_p95"))
            rows.append({"key": r["name"], "kind": kind, "positions": positions, "layer": L, "kn": x.get("kn"), "q_rank": x.get("q_rank"),
                         "selectivity": sel, "pseudo_p95": p95, "build_auc": x.get("build_auc", s.get("auc")),
                         "beats": None if kind == "one" or sel is None or p95 is None else bool(sel > p95)})
        for n in r.get("notes") or []:
            if n not in notes and "build_auc" not in n:
                notes.append(n)

    for i, sg in enumerate(segs):
        r = session.direction_key(contrast, target="mlpin", layers=Ls, positions=sg if sg.startswith("before:") else f"pos:{sg}",
                                  name=names["position"][i],
                                  subset=subset, max_tokens=int(max_tokens), pseudo=int(pseudo), iv=iv, seed=int(seed),
                                  progress=session.progress)
        add("position", sg, r)
    if names["scene"]:
        add("scene", None, session.direction_key(contrast, role="scene", layers=Ls, name=names["scene"], subset=subset,
                                                 max_tokens=int(max_tokens), pseudo=int(pseudo), iv=iv, seed=int(seed),
                                                 progress=session.progress))
    if names["one"]:
        add("one", None, session.direction_key(contrast, role="one", layers=Ls, name=names["one"], subset=subset,
                                               max_tokens=int(max_tokens), iv=iv, progress=session.progress))
    return {"base": base, "form": form, "segments": segs, "layers": Ls, "names": names, "rows": rows, "subset": subset,
            "pseudo": int(pseudo), "max_tokens": int(max_tokens), "iv": _iv_text(iv),
            "notes": notes + ["比べは作成に使った例での値です(甘い)。実際に足す量は「確かめる」の表で見ます"],
            "seconds": time.perf_counter() - t0}


def iv_units(session, iv, extra: list[str]):
    """先に当てておく介入 iv(名前・パスか、ユニットのリスト)に、ユニット extra を足したもの(ユニットのリストか Recipe)。"""
    if iv is None:
        return list(extra)
    if isinstance(iv, (list, tuple)):
        return [str(x) for x in iv] + list(extra)
    from .steer import combine_recipes
    p = Path(str(iv))
    r = session.recipe(path=p if p.suffix in (".yaml", ".yml") and p.exists() else session.recipe_path(str(iv)))
    if not extra:
        return r
    return combine_recipes([r, session.recipe(units=list(extra), name="fish")], name="iv")


def _iv_text(iv):
    if iv is None or isinstance(iv, str):
        return iv
    if hasattr(iv, "units"):
        return f"recipe({len(iv.units)} units)"
    return [str(x) for x in iv]


def verify_step(session, specs: list[str], *, iv=None, texts=None) -> dict:
    r = session.recipe(units=[str(x) for x in specs], name="fish")
    if isinstance(texts, str):
        texts = [x for x in texts.splitlines() if x.strip()]
    return session.nw_verify(r, iv=iv, texts=list(texts) if texts else None)


def check_step(session, specs: list[str], contrast: str, *, subset: str = "all", max_tokens: int = 96, iv=None,
               replies: str = "own", n_pos: int = 8, segments=None) -> dict:
    """足した量。書く位置が before:<正規表現> のときは、無介入の返事をレシピを当てて読み(書くと一致する文字が出なくなるので)、
    一致の直前の位置と、それ以外の位置に分けて出す。"""
    r = session.recipe(units=[str(x) for x in specs], name="fish")
    if segments is not None and before_mode(segments):
        return session.nw_check(r, contrast, subset=subset, max_tokens=int(max_tokens), iv=iv, replies="base",
                                positions=_segments(segments)[0])
    return session.nw_check(r, contrast, subset=subset, max_tokens=int(max_tokens), iv=iv, replies=replies, n_pos=int(n_pos))


def behave_step(session, specs: list[str], contrast: str, *, name: str, prompts=None, n: int = 12, max_tokens: int = 120,
                match: str | None = None, iv=None, others: bool = True) -> dict:
    """レシピを当てて返事を作り、読めるように返す(gne behave と同じ数え方)。狙う場面の質問(prompts を渡せばそれ、無ければ対照集合の
    質問の先頭 n 問)と、書き込みたくない場面の質問(dont_touch の群から n 問まで)。match(正規表現。無ければ対照集合の validate.match)に
    一致した数も出すが、返事の質は読んで確かめる。結果は <root>/behave/<model_id>/<name>_fish_scene.json・…_others.json にも残る。"""
    import json

    from .behave import load_prompts
    from .keyed_build import read_key_spec
    t0 = time.perf_counter()
    if isinstance(prompts, str):
        prompts = [x.strip() for x in prompts.splitlines() if x.strip()]
    own = bool(prompts)
    qs = list(prompts) if own else list(load_prompts(contrast))[:int(n)]
    ks = read_key_spec(Path(str(contrast)))
    if match is None or not str(match).strip():
        import yaml
        raw = yaml.safe_load(Path(str(contrast)).read_text(encoding="utf-8")) or {}
        match = ((raw.get("validate") or {}).get("match")) if isinstance(raw, dict) else None
    rec = session.recipe(units=[str(x) for x in specs], name=f"{name}_fish")
    recs = [rec]
    if iv is not None:
        ivr = iv_units(session, iv, [])
        recs.insert(0, ivr if hasattr(ivr, "units") else session.recipe(units=list(ivr), name="iv"))

    from .reverse import _conv
    sys_b = ks.contrast.system_b                 # 両側に指示がある対照集合: 使うときと同じく、B 側の指示を付けて答えさせる

    def one(tag, questions, system=None):
        asked = [_conv(system, q) for q in questions] if system is not None else list(questions)
        r = session.behave(asked, recipes=recs, match=match or None, max_tokens=int(max_tokens), name=f"{name}_fish_{tag}")
        d = json.loads(Path(r["path"]).read_text(encoding="utf-8"))
        run = d["runs"][0]
        return {"counts": run["counts"], "path": r["path"],
                "replies": [{"prompt": questions[x["i"]] if isinstance(questions[x["i"]], str) else questions[x["i"]][-1]["content"],
                             "text": x["text"], "loop": bool(x["loop"]), "finish": x["finish"]} for x in run["replies"]]}

    out = {"scene": one("scene", qs, sys_b)}
    dts = [q for dt in ks.dont_touch for q in dt.prompts][:int(n)]
    if others and dts:
        out["others"] = one("others", dts)
    return {**out, "match": match, "max_tokens": int(max_tokens), "own_prompts": own, "units": list(specs), "system_b": bool(sys_b),
            "seconds": time.perf_counter() - t0}


def pipeline(session, contrast: str, *, name: str, layers=None, segments=None, form: str = "cap", alpha=0.9,
             same_reply: bool = True, stack: bool = True, compare: bool = True, pseudo: int = 4, max_tokens: int = 96,
             response_tokens: int = 96, iv=None, rekey: bool = True, neurons: dict | None = None, top: int = 20) -> dict:
    """方向 → ニューロンの表 → 鍵 → ユニット →(そのユニットを当てた状態で鍵を作り直す)→ 式どおりか → 足した量、を順に回す。
    ニューロンは既定の候補(表の上から)を使う(neurons を渡せばそれ)。層と強さは渡されたものをそのまま使う。
    alpha は数 1 つ(全部の区切りに同じ値)か、区切りごとの "1,1,0.8"・[1, 1, 0.8]。"""
    t0 = time.perf_counter()
    Ls = _layers(layers)
    segs = _segments(segments)
    _form(form)
    al = alphas_of(alpha, len(segs))            # 数が区切りと合わなければ、モデルを流す前に止める
    stage = {}

    def lap(k, t):
        stage[k] = time.perf_counter() - t

    t = time.perf_counter()
    d = direction_step(session, contrast, name=name, layers=Ls, segments=segs, same_reply=same_reply, stack=stack,
                       response_tokens=response_tokens, compare=compare)
    lap("direction", t)
    t = time.perf_counter()
    usage = session.nw_neurons(Ls, top=max(int(top), needed(len(segs), form)), iv=iv)
    picked = {int(k): [int(j) for j in v] for k, v in (neurons or default_neurons(usage, len(segs), form)).items()}
    lap("neurons", t)
    t = time.perf_counter()
    keys = keys_step(session, contrast, base=name, layers=Ls, segments=segs, form=form, pseudo=pseudo, max_tokens=max_tokens, iv=iv)
    whole = before_mode(segs)
    specs = units(Ls, len(segs), picked, form, name, name, al, whole)
    lap("keys", t)
    keys2 = None
    notes = []
    if whole:
        # 文字の直前に書く形: 書くとその文字が出なくなり、当てた状態の返事には書き込みたい位置がほとんど残らないので、作り直さない
        rekey = False
        notes.append("書く位置が before:<正規表現> なので、レシピを当てた状態での鍵の作り直しはしていません(書くと、一致する文字が出なくなるため)")
    if rekey:
        t = time.perf_counter()
        base2 = f"{name}_r"
        keys2 = keys_step(session, contrast, base=base2, layers=Ls, segments=segs, form=form, pseudo=pseudo, max_tokens=max_tokens,
                          iv=iv_units(session, iv, specs))
        specs = units(Ls, len(segs), picked, form, name, base2, al, whole)
        lap("rekey", t)
    t = time.perf_counter()
    ver = verify_step(session, specs, iv=iv)
    lap("verify", t)
    t = time.perf_counter()
    chk = check_step(session, specs, contrast, max_tokens=max_tokens, iv=iv, segments=segs)
    lap("check", t)
    return {"name": name, "contrast_path": contrast, "layers": Ls, "segments": segs, "form": form, "alpha": alpha_out(al), "alphas": al,
            "direction": d, "usage": usage, "neurons": picked, "neurons_default": neurons is None, "keys": keys, "rekeyed": keys2,
            "key_base": f"{name}_r" if rekey else name, "units": specs, "unit_rows": unit_rows(session, specs),
            "verify": ver, "check": chk, "iv": keys["iv"], "notes": notes, "stage_seconds": stage, "seconds": time.perf_counter() - t0}


# --- 重みに入れる前に試す ---------------------------------------------------------------------------------------

def _counts(rows: list[dict], rx) -> dict:
    """返事の数(gne behave と同じ数え方): 空・繰り返し(4 トークンの並びの種類が半分を切る)・長さの上限で切れた・平均の長さ・正規表現の一致。"""
    c = {"n": len(rows), "empty": sum(not (r["text"] or "").strip() for r in rows), "loop": sum(bool(r["loop"]) for r in rows),
         "at_limit": sum(r["finish"] not in ("eog", "stop", "eos") for r in rows),
         "tokens": round(sum(r["n_tokens"] for r in rows) / max(len(rows), 1), 1)}
    if rx is not None:
        c["match"] = sum(bool(rx.search(r["text"] or "")) for r in rows)
    return c


def try_step(session, contrast: str, *, direction: str, layers=None, alpha=None, prompts=None, n: int = 12, max_tokens: int = 120,
             match: str | None = None, others: bool = True, baseline: bool = True, temperature: float = 0.0, seed: int = 0,
             iv=None, name: str | None = None) -> dict:
    """重みに入れる前に試す: 区切りつきの方向(gne direction --segments で作った direction)を、ニューロンにも鍵にも入れずに、
    層 layers の出力へ直接足して返事を作る(control vector を、返事の位置の区切りが変わるところで付け替える)。
    - 足すのは狙う場面の質問(prompts を渡せばそれ、無ければ対照集合の質問の先頭 n 問)の返事だけ。場面は質問の群で決める
      (重みに入れたときの場面の鍵の代わり)。位置 0(返事を書き始める直前 = 生成プロンプトの最後のトークン)から足し、質問の文には足さない。
    - alpha は区切りごとの強さ(alphas_of。1 で、その区切りの方向をそのままの長さで足す)。layers を省くと、方向を作ったときの
      重ねる層(stack_layers)。
    - baseline: 同じ質問に何も足さずに答えさせた返事。others: 書き込みたくない場面(dont_touch)の質問(n 問まで)に、何も足さずに
      答えさせた返事(狙う場面にしか足さないので、何も足さない返事と同じ。並べて読むために出す)。
    - temperature > 0 でサンプリング(質問ごとに seed をずらす)。iv は先に当てておく介入(全部の位置に当てる)。
    - 返事と一緒に、機械的に数えられるもの(繰り返し・空・長さの上限で切れた数・正規表現の一致)を出す。
    制約: 足せるのは層 1 〜 最終層の 1 つ手前(control vector の制約)。付け替えは 1 本の列ごとなので、狙う場面は 1 問ずつ生成する。
    結果は <root>/behave/<model_id>/<name>_try.json にも残す。"""
    import json
    import re

    from .behave import load_prompts, loop_like
    from .direction import parse_segments
    from .engine import KEEP, Sampling
    from .evalx import _generate
    from .keyed_build import intervention, read_key_spec
    from .reverse import _conv
    t0 = time.perf_counter()
    e = session.engine
    ds = session.direction_set(str(direction))
    seg_text = list(ds.segments) or ["0-"]                  # 区切りの無い方向は、返事の全部の位置に同じベクトル
    ranges = parse_segments(seg_text)
    al = alphas_of(alpha, len(seg_text))
    name = name or ds.name
    if not re.fullmatch(r"[\w.\-]+", name):
        raise GneError("BAD_ARGS", f"名前 {name!r} は英数字・_・-・. だけにしてください")
    stack = [int(L) for L in (ds.meta.get("stack_layers") or [])]
    if layers is None or (isinstance(layers, str) and not layers.strip()):
        if not stack:
            raise GneError("BAD_ARGS", "足す層(layers)を指定してください(例 \"11-23\")",
                           "重ねる層(--stack-layers)を付けて作った方向なら、省くとその層を使います")
        Ls = sorted(stack)
    else:
        Ls = _layers(layers)
    have = ds.seg[0] if ds.segments else ds.raw
    for L in Ls:
        if not 1 <= L <= e.n_layer - 2:
            raise GneError("BAD_ARGS", f"層 {L} には足せません(足せるのは 1〜{e.n_layer - 2}。層 0 と最終層は control vector で操作できません)")
        if L not in have:
            raise GneError("BAD_DIRECTION", f"方向 {ds.name} に層 {L} がありません")
    notes = []
    if not ds.segments:
        notes.append(f"方向 {ds.name} は区切りつきで作られていないので、返事の全部の位置に同じベクトルを足しました")
    if stack and sorted(stack) != Ls:
        notes.append(f"方向 {ds.name} は層 {','.join(map(str, sorted(stack)))} に重ねて書く前提で作ってあります。違う層の組({','.join(map(str, Ls))})に"
                     "足すと、足す量がその前提とずれます(層の組を決めたら、その層で方向を作り直すと正確です)")
    if isinstance(prompts, str):
        prompts = [x.strip() for x in prompts.splitlines() if x.strip()]
    own = bool(prompts)
    qs = list(prompts) if own else list(load_prompts(contrast))[:int(n)]
    ks = read_key_spec(Path(str(contrast)))
    sys_b = ks.contrast.system_b
    if match is None or not str(match).strip():
        import yaml
        raw = yaml.safe_load(Path(str(contrast)).read_text(encoding="utf-8")) or {}
        match = ((raw.get("validate") or {}).get("match")) if isinstance(raw, dict) else None
    rx = re.compile(match) if match else None
    iv_vecs, iv_lora, _ = intervention(session, iv)
    iv_vecs = {int(L): np.asarray(v, dtype=np.float32) for L, v in (iv_vecs or {}).items()}

    def vec(i):
        """区切り i のときに足すベクトル({層: ベクトル}。先に当てておく介入の分も入れる)。区切りの外(i が None)は介入の分だけ。"""
        out = dict(iv_vecs)
        if i is not None and al[i]:
            src = ds.seg[i] if ds.segments else ds.raw
            for L in Ls:
                v = (al[i] * np.asarray(src[L], dtype=np.float64)).astype(np.float32)
                out[L] = out[L] + v if L in out else v
        return out or None

    V = [vec(i) for i in range(len(seg_text))]
    V_out = vec(None)

    def seg_of(k: int):
        for i, (lo, hi) in enumerate(ranges):
            if k >= lo and (hi is None or k < hi):
                return i
        return None

    at = lambda k: V_out if seg_of(k) is None else V[seg_of(k)]
    sp0 = Sampling(temperature=float(temperature), seed=int(seed), presence_penalty=0.0, min_p=0.0) if temperature and temperature > 0 else None
    dts = [q for dt in ks.dont_touch for q in dt.prompts][:int(n)] if others else []
    total = len(qs) * (2 if baseline else 1) + len(dts)
    done = 0

    def tick(k: int, stage: str) -> None:
        nonlocal done
        done += k
        if session.progress:
            session.progress(min(done, total), total, stage)

    def row(q, r_text, ids, finish):
        return {"prompt": q if isinstance(q, str) else q[-1]["content"], "text": r_text, "loop": bool(loop_like(ids)), "finish": finish,
                "n_tokens": len(ids)}

    # 狙う場面: 1 問ずつ。質問の最後のトークン(位置 0)から足し、区切りが変わるところで付け替える
    import dataclasses
    ctx = e.steer_ctx()
    scene = []
    e.set_lora(ctx, iv_lora)
    try:
        for i, q in enumerate(qs):
            ids = e.tokenize_prompt(e.render_chat(_conv(sys_b, q), enable_thinking=False))
            if len(ids) + int(max_tokens) > e.n_ctx:
                raise GneError("CONTEXT_FULL", f"質問 {i} が n_ctx={e.n_ctx} に入りません")
            e.set_cvec(ctx, V_out)
            e.clear_memory(ctx)
            e.prefill(ctx, [ids[:-1]])
            e.set_cvec(ctx, at(0))
            pf = e.prefill(ctx, [[ids[-1]]], pos0=len(ids) - 1)

            def sched(step, outs):
                # step 番目のトークンが決まった後、それを読ませる前に呼ばれる。読ませる位置 = 返事の位置 step + 1
                return KEEP if seg_of(step) == seg_of(step + 1) else at(step + 1)
            sp = None if sp0 is None else dataclasses.replace(sp0, seed=sp0.seed + 7919 * i)
            r = e.generate_greedy(ctx, [ids], max_tokens=int(max_tokens), prefilled=pf, schedule=sched, sampling=sp)[0]
            scene.append(row(q, r.text, r.token_ids, r.finish))
            tick(1, "try: 狙う場面")
    finally:
        e.set_cvec(ctx, None)
        e.set_lora(ctx, None)

    def plain(questions, system, stage):
        asked = [_conv(system, q) for q in questions] if system is not None else list(questions)
        outs = _generate(session, [(c, int(max_tokens)) for c in asked], iv_vecs or None, iv_lora, "all", lambda k: tick(k, stage), sampling=sp0)
        return [row(q, o["text"], o["token_ids"], o["finish"]) for q, o in zip(questions, outs, strict=True)]

    out = {"scene": {"counts": _counts(scene, rx), "replies": scene}}
    if baseline:
        rows = plain(qs, sys_b, "try: 何も足さない")
        out["baseline"] = {"counts": _counts(rows, rx), "replies": rows}
    if dts:
        rows = plain(dts, None, "try: 書き込みたくない場面")
        out["others"] = {"counts": _counts(rows, rx), "replies": rows}
    norms = [{"segment": i, "positions": seg_text[i], "alpha": al[i],
              "norm": {int(L): float(al[i] * np.linalg.norm((ds.seg[i] if ds.segments else ds.raw)[L])) for L in Ls}} for i in range(len(seg_text))]
    notes.append("狙う場面の質問にだけ足しています(場面は質問の群で決めています)。重みに入れると、場面は鍵が決めるので、ほかの場面で"
                 "書かないかは「足した量」で確かめます")
    notes.append("数は機械的に数えたものです(繰り返し・空・正規表現の一致)。口調が出たか、相手の発言に合っているかは、返事を読んで確かめます")
    res = {**out, "direction": ds.name, "layers": Ls, "segments": seg_text, "alpha": alpha_out(al), "alphas": al, "added": norms,
           "stack_layers": sorted(stack) or None, "match": match, "max_tokens": int(max_tokens), "temperature": float(temperature or 0.0),
           "seed": int(seed), "own_prompts": own, "system_b": bool(sys_b), "iv": _iv_text(iv),
           "notes": notes, "seconds": time.perf_counter() - t0}
    path = session.store.root / "behave" / session.meta.model_id / f"{name}_try.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return {**res, "path": str(path)}
