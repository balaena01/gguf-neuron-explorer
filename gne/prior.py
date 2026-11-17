"""会話の形を混ぜた対照集合: 会話の形のラベルと、前のやり取りのアシスタントの応答の穴埋め(docs/fish.md §3「会話の形を混ぜる」)。

1 問は、文字列(1 ターン)か、メッセージのリスト。リストの先頭に {"role": "system", "content": …} を置くと、その問だけの
システムプロンプトになる(群の system_a / system_b / dont_touch の system より優先)。前のやり取りのアシスタントの応答を
{"role": "assistant", "content": null} と書くと、無介入のモデルに書かせて埋める。

- 埋めるときに使うのは、その問の system(あれば)と、その応答より前のメッセージだけ。群の system・レシピ・LoRA・control vector は
  使わない(同じ問は、両側・どの群でも同じ応答になる)。greedy、思考なし、上限 max_tokens(既定 PRIOR_MAX_TOKENS)。
- 埋めた応答は <root>/prior/<model_id>/replies.json に覚えておく(会話と上限のハッシュ → 応答)。同じ会話・同じ上限なら作り直さない。
- 埋めた対照集合は <root>/contrasts/prior/<元の名前>-<ハッシュ>.yaml に書く(ensure_filled)。元のファイルからの相対パス
  (dont_touch の prompts / replies のファイル、replies*、レシピのファイル)は絶対パスに直す。

会話の形のラベル(conv_form): single(system なしの 1 ターン)/ multi_turn(system なしで、ユーザーの発話が 2 つ以上)/
system(system ありの 1 ターン)/ system_multi_turn(system ありで、ユーザーの発話が 2 つ以上)。system は、その問の system か
群の system(空の文字列も system あり)。
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

from .steer import GneError

PRIOR_MAX_TOKENS = 256
FORMS = ("single", "multi_turn", "system", "system_multi_turn")
_LISTS = ("user_prompts", "prompts_a", "prompts_b")


def own_system(q) -> str | None:
    """その問だけの system(リストの先頭の system メッセージの中身)。無ければ None。"""
    if isinstance(q, list) and q and isinstance(q[0], dict) and q[0].get("role") == "system":
        return q[0].get("content")
    return None


def conv_form(system: str | None, q) -> str:
    """問 q を、群の system のもとで読むときの会話の形(FORMS のどれか)。"""
    has_sys = own_system(q) is not None or system is not None
    users = 1 if isinstance(q, str) else sum(1 for m in q if isinstance(m, dict) and m.get("role") == "user")
    if has_sys:
        return "system_multi_turn" if users > 1 else "system"
    return "multi_turn" if users > 1 else "single"


def conv_form_of(conv: list[dict]) -> str:
    """組み立てた会話(メッセージのリスト。system を含む)の形。"""
    has_sys = any(m.get("role") == "system" for m in conv)
    users = sum(1 for m in conv if m.get("role") == "user")
    if has_sys:
        return "system_multi_turn" if users > 1 else "system"
    return "multi_turn" if users > 1 else "single"


def has_null(q) -> bool:
    """埋める応答({"role": "assistant", "content": null})があるか。"""
    return isinstance(q, list) and any(isinstance(m, dict) and m.get("role") == "assistant" and m.get("content") is None for m in q)


def _question_lists(d: dict) -> list[tuple[str, list]]:
    """表の中の質問のリスト(インラインのものだけ)→ [(場所, リスト)]。"""
    out = [(k, d[k]) for k in _LISTS if isinstance(d.get(k), list)]
    dt = d.get("dont_touch")
    groups = [dt] if isinstance(dt, dict) else (dt if isinstance(dt, list) else [])
    for i, g in enumerate(groups):
        if isinstance(g, dict) and isinstance(g.get("prompts"), list):
            out.append((f"dont_touch[{i}]", g["prompts"]))
    return out


def spec_has_null(d: dict) -> bool:
    return any(has_null(q) for _, xs in _question_lists(d) for q in xs)


def _cache_path(session) -> Path:
    return session.store.root / "prior" / session.meta.model_id / "replies.json"


def _load_cache(p: Path) -> dict:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_cache(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part")
    tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def _key(msgs: list[dict], max_tokens: int) -> str:
    return hashlib.sha256(json.dumps([msgs, int(max_tokens)], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def check_max_tokens(max_tokens) -> int:
    mt = PRIOR_MAX_TOKENS if max_tokens is None else max_tokens
    try:
        mt = int(mt)
    except (TypeError, ValueError):
        raise GneError("BAD_ARGS", f"前のやり取りの応答の上限(prior_max_tokens){max_tokens!r} を読めません") from None
    if mt < 1:
        raise GneError("BAD_ARGS", f"前のやり取りの応答の上限(prior_max_tokens)は 1 以上です({max_tokens!r})")
    return mt


def fill_questions(session, questions: list, *, max_tokens: int | None = None,
                   progress: Callable[[int, int, str], None] | None = None) -> tuple[list, dict]:
    """questions の中の {"role": "assistant", "content": null} を、無介入のモデルの応答で埋めたコピーを返す。→ (質問, 記録)。
    記録: filled(埋めた応答の数)、generated(新しく作った数)、cached(覚え書きから使った数)、cut(上限で切れた数)、max_tokens。"""
    from .evalx import _generate
    mt = check_max_tokens(max_tokens)
    qs = [copy.deepcopy(q) for q in questions]
    path = _cache_path(session)
    cache = _load_cache(path)
    stats = {"filled": 0, "generated": 0, "cached": 0, "cut": 0, "max_tokens": mt}
    total = sum(1 for q in qs if isinstance(q, list) for m in q if isinstance(m, dict) and m.get("role") == "assistant"
                and m.get("content") is None)
    if not total:
        return qs, stats
    changed = False
    while True:
        todo = []
        for qi, q in enumerate(qs):
            if not isinstance(q, list):
                continue
            j = next((j for j, m in enumerate(q) if m.get("role") == "assistant" and m.get("content") is None), None)
            if j is not None:
                todo.append((qi, j))
        if not todo:
            break
        need: list[tuple[int, int, list[dict], str]] = []
        for qi, j in todo:
            msgs = [{"role": m["role"], "content": m["content"]} for m in qs[qi][:j]]
            k = _key(msgs, mt)
            hit = cache.get(k)
            if isinstance(hit, dict) and isinstance(hit.get("text"), str):
                qs[qi][j]["content"] = hit["text"]
                stats["cached"] += 1
                stats["cut"] += int(hit.get("finish") == "length")
            else:
                need.append((qi, j, msgs, k))
        if need:
            outs = _generate(session, [(msgs, mt) for _, _, msgs, _ in need], None, None, "all", lambda n: None)
            for (qi, j, _, k), o in zip(need, outs, strict=True):
                text = str(o["text"]).strip()
                qs[qi][j]["content"] = text
                cache[k] = {"text": text, "finish": o.get("finish")}
                stats["generated"] += 1
                stats["cut"] += int(o.get("finish") == "length")
            changed = True
        stats["filled"] += len(todo)
        if progress:
            progress(stats["filled"], total, "前のやり取りの応答")
    if changed:
        _save_cache(path, cache)
    return qs, stats


def _absolutize(d: dict, base: Path) -> dict:
    """元のファイルからの相対パスを絶対パスに直す(埋めた対照集合は別の場所に書くので)。"""
    def fix(v):
        if isinstance(v, str) and v and not Path(v).is_absolute() and (base / v).exists():
            return str((base / v).resolve())
        return v
    for k in ("replies", "replies_a", "replies_b", "recipe_a", "recipe_b"):
        if k in d:
            d[k] = fix(d[k])
    dt = d.get("dont_touch")
    for g in ([dt] if isinstance(dt, dict) else (dt if isinstance(dt, list) else [])):
        if isinstance(g, dict):
            for k in ("prompts", "replies"):
                if k in g:
                    g[k] = fix(g[k])
    return d


def counts(d: dict) -> dict:
    """表の質問の会話の形ごとの数 → {場所: {形: 数}}(system は群の system を見る)。"""
    out = {}
    for where, xs in _question_lists(d):
        if where.startswith("dont_touch"):
            i = int(where[len("dont_touch["):-1])
            dt = d["dont_touch"]
            g = dt if isinstance(dt, dict) else dt[i]
            sys_ = g.get("system")
            label = f"dont_touch:{g.get('name') or f'dt{i}'}"
        else:
            sys_ = d.get("system_a") if where == "prompts_a" else d.get("system_b")
            label = where
        c: dict[str, int] = {}
        for q in xs:
            f = conv_form(sys_, q)
            c[f] = c.get(f, 0) + 1
        out[label] = c
    return out


def fill_file(session, contrast, *, max_tokens: int | None = None, out: str | Path | None = None,
              progress: Callable[[int, int, str], None] | None = None) -> dict:
    """対照集合のファイルの、埋める応答を全部埋めて書く。out を省くと <root>/contrasts/prior/<名前>-<ハッシュ>.yaml。
    埋めるものが無く out も省いたときは書かない(path は元のまま)。→ {path, source, filled, generated, cached, cut, max_tokens, forms}。"""
    import yaml

    from .yamlio import write_yaml
    src = Path(str(contrast))
    if not src.exists():
        raise GneError("FILE_NOT_FOUND", f"対照集合 {src} がありません")
    text = src.read_text(encoding="utf-8")
    d = yaml.safe_load(text) or {}
    if not isinstance(d, dict):
        raise GneError("BAD_CONTRAST", f"{src}: 対照集合は YAML / JSON の表です")
    mt = check_max_tokens(max_tokens)
    if not spec_has_null(d) and out is None:
        return {"path": str(src), "source": str(src), "filled": 0, "generated": 0, "cached": 0, "cut": 0, "max_tokens": mt,
                "forms": counts(d)}
    d = _absolutize(copy.deepcopy(d), src.resolve().parent)
    lists = _question_lists(d)
    flat = [q for _, xs in lists for q in xs]
    filled, st = fill_questions(session, flat, max_tokens=mt, progress=progress)
    k = 0
    for _, xs in lists:
        xs[:] = filled[k:k + len(xs)]
        k += len(xs)
    if out is None:
        h = hashlib.sha256(f"{session.meta.model_id}\n{mt}\n{text}".encode()).hexdigest()[:12]
        name = str(d.get("name") or src.stem)
        out = session.store.contrast_dir() / "prior" / f"{name}-{h}.yaml"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(out, d)
    from .keyed_build import read_key_spec
    read_key_spec(out)                                        # 読めるか確かめる
    return {"path": str(out), "source": str(src), **st, "forms": counts(d)}


def ensure_filled(session, contrast, *, max_tokens: int | None = None, progress=None) -> str:
    """対照集合のパスに埋める応答があれば埋めたファイルのパスを、無ければ元のパスを返す。パス以外(表・ContrastSet)はそのまま。"""
    if not isinstance(contrast, (str, Path)):
        return contrast
    p = Path(str(contrast))
    if p.suffix.lower() not in (".yaml", ".yml", ".json") or not p.exists():
        return contrast
    try:
        import yaml
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return contrast
    if not isinstance(d, dict) or not spec_has_null(d):
        return contrast
    r = fill_file(session, p, max_tokens=max_tokens, progress=progress)
    session.warn(f"前のやり取りの応答 {r['filled']} 個を無介入のモデルで埋めた対照集合を使います({r['path']}。新しく作った "
                 f"{r['generated']}、覚えていた {r['cached']}、上限 {r['max_tokens']} で切れた {r['cut']})")
    return r["path"] if isinstance(contrast, str) else Path(r["path"])
