"""長時間の処理(sweep / lens / stats など)をバックグラウンドで回すジョブと、GPU の調停。
設計書 §6.13 (d) の job_start / job_status、別冊 §8.1。

- GPU を使う処理は GpuArbiter を通す。対話的な呼び出し(steer・neuron など)が待っていれば、ジョブは
  ユニットの切れ目(進捗の報告のたび)で GPU を譲り、終わったら続きから進む。
- 1 つの Engine を共有するので、GPU を握っていない間は Engine に触らない。
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .steer import GneError


class GpuArbiter:
    """対話的な要求を優先する排他。ジョブは yield_point() で譲る。"""

    def __init__(self):
        self._cv = threading.Condition()
        self._owner: int | None = None
        self._waiting = 0                     # 待っている対話的な要求の数

    @contextmanager
    def interactive(self):
        me = threading.get_ident()
        with self._cv:
            self._waiting += 1
            while self._owner is not None and self._owner != me:
                self._cv.wait()
            self._waiting -= 1
            nested = self._owner == me
            self._owner = me
        try:
            yield
        finally:
            if not nested:
                with self._cv:
                    self._owner = None
                    self._cv.notify_all()

    @contextmanager
    def job(self):
        me = threading.get_ident()
        with self._cv:
            while self._owner is not None or self._waiting:
                self._cv.wait()
            self._owner = me
        try:
            yield
        finally:
            with self._cv:
                self._owner = None
                self._cv.notify_all()

    def yield_point(self) -> bool:
        """ジョブのスレッドから呼ぶ。待っている対話的な要求があれば GPU を譲り、終わるまで待つ。譲ったら True。"""
        me = threading.get_ident()
        with self._cv:
            if self._owner != me or not self._waiting:
                return False
            self._owner = None
            self._cv.notify_all()
            while self._owner is not None or self._waiting:
                self._cv.wait()
            self._owner = me
            return True


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    status: str = "queued"                   # queued | running | done | error | cancelled
    done: int = 0
    total: int = 0
    stage: str = ""
    started: float | None = None
    finished: float | None = None
    result: dict | None = None
    error: dict | None = None
    yields: int = 0
    cancel: bool = False
    estimate: dict | None = None             # CHANGE-03 C: 投げたときの見積もり
    _t_last: float = field(default=0.0, repr=False)

    def view(self) -> dict:
        eta = None
        if self.status == "running" and self.done and self.total and self.started:
            el = time.monotonic() - self.started
            eta = el / self.done * (self.total - self.done)
        return {"id": self.id, "kind": self.kind, "params": self.params, "status": self.status, "done": self.done,
                "total": self.total, "stage": self.stage, "eta_s": eta, "yields": self.yields,
                "seconds": (self.finished or time.monotonic()) - self.started if self.started else None,
                "result": self.result, "error": self.error,
                "estimate_s": None if self.estimate is None else self.estimate.get("seconds"),
                # 層を重ねる読み直しを「1 つずつ読む」場合で数えた見積もり(初回。実際はこれより短いことが多い)
                "estimate_stack_first": bool(self.estimate and (self.estimate.get("detail") or {}).get("stack_first"))}


class _Cancelled(Exception):
    pass


def _sweep_rank(s, layer: int, **p) -> dict:
    """sweep のあとに rank まで回す(UI から 1 回で層の順位表まで作る)。"""
    r = s.sweep(layer, **p)
    k = s.rank(r["run"], show=0)
    return {"sweep": {k_: v for k_, v in r.items() if k_ != "units"}, "rank": {k_: v for k_, v in k.items() if k_ != "units"}}


def _eval(s, recipe: str | None = None, units: list[str] | None = None, dose_unit: str | None = None,
          compare_methods: bool = False, match_effect: float | None = None, score_layer: int | None = None,
          alloc: str | None = None, n_heads: list[int] | None = None, **p) -> dict:
    """eval(§6.14)。recipe は保存したレシピの名前、units は単発指定(どちらか)。compare_methods は方式の比較(CHANGE-06 F.3)、
    alloc は heads allocate の組のヘッド数ごとの評価(F.4、recipe も units も要らない)。ほかの引数は Session.eval と同じ
    (keyed のレシピの keyed_contrast・keyed_metrics・keyed_examples・keyed_match・keyed_pseudo・keyed_subset も、CHANGE-07 E)。"""
    if alloc:
        return s.eval_alloc_curve(alloc, n_heads=n_heads, suite=p.get("suite"), when=p.get("when"),
                                  random_controls=int(p.get("random_controls") or 0))
    if bool(recipe) == bool(units):
        raise GneError("BAD_ARGS", "recipe(名前)か units のどちらかを渡してください")
    r = s.recipe(path=s.recipe_path(recipe)) if recipe else s.recipe(units=list(units), dose_unit=dose_unit)
    if compare_methods:
        return s.eval_compare_methods(r, match_effect=match_effect, score_layer=score_layer, suite=p.get("suite"),
                                      when=p.get("when"))
    d = s.eval(r, **p)
    return {k: v for k, v in d.items() if k != "samples"}


def _headproj_export(s, recipe: str | None = None, out: str | None = None, *, units: list[str] | None = None,
                     name: str | None = None, direction_layer: str | None = None, file_stem: str | None = None,
                     overwrite: bool = True, key: str | None = None, direction: str | None = None,
                     adapter_name: str | None = None, **p) -> dict:
    """ヘッドの proj・鍵つきの書き込み(keyed、CHANGE-07)と gain(LoRA に入るユニット)を LoRA アダプタか焼き込み GGUF に
    書き出す。recipe(保存したレシピの名前かパス)か units(UI のレシピのユニットの書き方)。key・direction は #KEY・@NAME を
    省いたユニットの鍵と方向。out を省くと Session.default_export_path(<root>/export/<名前>.lora.gguf、焼き込みは
    <名前>-headproj.gguf、keyed だけなら <名前>-keyed.lora.gguf / -keyed.gguf)。file_stem を渡すとファイル名はそれ(LoRA は
    <file_stem>.lora.gguf、焼き込みは <file_stem>.gguf)。overwrite=False なら、同じ名前のファイルがあれば書かずに
    FILE_EXISTS を返す(UI は確かめてから上書きする)。LoRA は小さいので、UI から落とせる URL も返す。
    adapter_name は LoRA の名前の欄(general.name。name はレシピの名前なので別の引数。空の文字列は省いたのと同じ)。"""
    import re
    from pathlib import Path
    from .steer import GneError
    if units:
        r = s.recipe(units=list(units), name=name or None, direction_layer=direction_layer, default_key=key,
                     default_direction=direction)
    elif recipe:
        r = s.recipe(path=s.recipe_path(recipe), direction_layer=direction_layer, default_key=key, default_direction=direction)
    else:
        raise GneError("BAD_ARGS", "recipe か units が要ります")
    nm = name or r.name or "headproj"
    if not re.fullmatch(r"[\w.\-]+", nm):
        raise GneError("BAD_ARGS", f"名前 {nm!r} は英数字・_・-・. だけにしてください")
    bake = bool(p.get("bake"))
    if file_stem is not None:
        file_stem = file_stem.strip().removesuffix(".gguf").removesuffix(".lora")
        if not re.fullmatch(r"[\w.\-]+", file_stem):
            raise GneError("BAD_ARGS", f"ファイル名 {file_stem!r} は英数字・_・-・. だけにしてください")
    if out:
        path = Path(out)
    elif file_stem:
        path = s.store.export_dir() / (f"{file_stem}.gguf" if bake else f"{file_stem}.lora.gguf")
    else:
        path = s.default_export_path(r, bake=bake, kind="headproj", name=nm)
    if not overwrite and path.exists():
        raise GneError("FILE_EXISTS", f"{path.name} はもうあります", "上書きするか、別の名前にしてください")
    if adapter_name is not None and str(adapter_name).strip():
        p["name"] = str(adapter_name).strip()
    d = s.headproj_export(r, path, **p)
    d["file"] = Path(d["path"]).name
    if not bake and Path(d["path"]).parent.resolve() == s.store.export_dir().resolve():
        d["url"] = f"/api/export/file/{d['file']}"
    d["name"] = nm
    return d


def _layers(x) -> list[int] | None:
    """層の指定(リスト・"8,14,20"・"10-16"・混ぜた "8,10-12")→ 層のリスト。リストの中の文字列("11-23")も同じ読み方で開く。"""
    from .cli import parse_layers
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        out: list[int] = []
        for L in x:
            if isinstance(L, str):
                out.extend(parse_layers(L) or [])
            else:
                out.append(int(L))
        return out
    if isinstance(x, int):
        return [x]
    return parse_layers(str(x))


def _key_contrast(s, contrast, spec, params: dict, keys) -> tuple:
    """鍵のジョブの対照集合: contrast(パス)か spec(UI の入力欄の表。heads_pipeline と同じく <root>/contrasts/<name>.yaml に
    書いてから使う)。spec の keyed の設定(keys のもの)を、渡していない引数の既定にする。戻り値 (パス, 引数)。"""
    if (contrast is None) == (spec is None):
        raise GneError("BAD_ARGS", "対照集合は contrast(パス)か spec(表)のどちらかで渡してください")
    if spec is None:
        return contrast, params
    from .keyed_pipeline import apply_settings, save_spec
    path, settings = save_spec(s, spec)
    return str(path), apply_settings(settings, params, keys)


def _direction_key(s, contrast=None, layers=None, spec: dict | None = None, **p) -> dict:
    """鍵つきの書き込みの鍵を作る(CHANGE-07、gne direction key)。contrast は対照集合のパスか、spec に対照集合の表(UI の入力欄)。
    layers は必須(リストか "8,14,20"。spec の keyed.layers でも可)。ほかの引数は Session.direction_key(keyed_build.build_keys)と同じ
    (key_fit="ridge"・key_c は、FiSH の位置の鍵 = target mlpin・positions pos:<範囲> を最小二乗で作るとき)。"""
    contrast, p = _key_contrast(s, contrast, spec, {"layers": layers, **p},
                                ("layers", "target", "q_mode", "subset", "pseudo", "max_tokens", "iv"))
    Ls = _layers(p.pop("layers", None))
    if not Ls:
        raise GneError("BAD_ARGS", "鍵の層(layers)を指定してください(例 [8, 14, 20] か \"8,14,20\")")
    return s.direction_key(contrast, layers=Ls, progress=s.progress, **p)


def _direction_key_check(s, key: str, layers=None, compact: bool = False, **p) -> dict:
    """鍵の分離の確認(CHANGE-07、gne direction key-check)。compact なら例ごとの ρ(values)・ヒストグラム(hist)・
    比較用の鍵ごとの値を落とす(MCP 向け。UI はヒストグラムに使うので落とさない)。"""
    d = s.direction_key_check(key, layers=_layers(layers), progress=s.progress, **p)
    if compact:
        for r in d["layers"]:
            r.pop("values", None)
            r.pop("hist", None)
            for k in ("pseudo_auc", "pseudo_selectivity"):
                r[k].pop("values", None)
    return d


def _direction_key_experts(s, contrast=None, layers=None, spec: dict | None = None, **p) -> dict:
    """expert:e の鍵の専門家を選ぶ材料(CHANGE-07 MoE、gne direction key-experts): 書き込みたい位置・書き込みたくない位置で各専門家が
    選ばれた割合と重み。contrast は対照集合のパスか、spec に対照集合の表(UI の入力欄)。layers は必須(リストか "6,12,18")。
    ほかの引数は keyed_build.expert_usage と同じ。"""
    contrast, p = _key_contrast(s, contrast, spec, {"layers": layers, **p}, ("layers", "subset", "max_tokens", "iv"))
    Ls = _layers(p.pop("layers", None))
    if not Ls:
        raise GneError("BAD_ARGS", "調べる MoE の層(layers)を指定してください(例 [6, 12, 18] か \"6,12,18\")")
    return s.direction_key_experts(contrast, layers=Ls, progress=s.progress, **p)


def _direction_key_pipeline(s, contrast=None, spec: dict | None = None, **p) -> dict:
    """方向(gne direction --positions response と同じ)→ 鍵 → 作成と反対側の subset での key-check を 1 回で(CHANGE-07、
    keyed_pipeline.run)。contrast は対照集合のパスか、spec に対照集合の表(UI の入力欄。<root>/contrasts/<name>.yaml に書く)。
    同じ名前の方向があれば使い回し、rebuild_direction=True で作り直す。"""
    if (contrast is None) == (spec is None):
        raise GneError("BAD_ARGS", "対照集合は contrast(パス)か spec(表)のどちらかで渡してください")
    from .keyed_pipeline import run
    return run(s, spec if spec is not None else contrast, progress=s.progress, **p)


def _direction_key_verify(s, layer, key: str, direction: str, alphas=None, target: str | None = None,
                          direction_layer: str = "own", texts=None) -> dict:
    """V30(CHANGE-07、gne direction key-verify): keyed の LoRA が式どおり ΔW x = α (κᵀx) d を書くか(数秒)。target を省くと
    鍵の書き込み先。texts は流す文か会話のリスト、または 1 行に 1 文の文字列(省くと語尾の違う日本語と英語の 2 文。
    専門家の鍵はその専門家が選ばれる文を)。"""
    from .keyed import load_key, parse_target
    kind, expert = parse_target(target or load_key(s, str(key)).target)
    if isinstance(texts, str):
        texts = [x for x in texts.splitlines() if x.strip()]
    return s.check_keyed(int(layer), key=str(key), direction=str(direction), alphas=tuple(float(a) for a in (alphas or (1.0, 2.0))),
                         kind=kind, expert=expert, direction_layer=direction_layer, texts=list(texts) if texts else None)


def _fish_args(s, contrast, spec, p: dict, keys) -> tuple[str, dict, str]:
    """FiSH のジョブの対照集合と引数: 渡していない引数(None)は、表の fish の設定、その次に既定で埋める。"""
    from . import fish
    path, settings, cname = fish.resolve(s, contrast, spec)
    q = {k: v for k, v in p.items() if v is not None}
    for k in keys:
        if k not in q and (k in settings or k in fish.DEFAULTS):
            q[k] = settings.get(k, fish.DEFAULTS.get(k))
    iv = q.get("iv")
    if isinstance(iv, str) and not iv.strip():
        q.pop("iv")
    return path, q, cname


def _fish_direction(s, contrast=None, spec: dict | None = None, **p) -> dict:
    """FiSH の方向(docs/fish.md): 返事の位置で分けた方向を作り、層 × 区切りごとの長さと、今までの作り方との向きの近さを返す。
    stack_fast=True で、層を重ねる読み直しを速い読み方にする(既定は切。方向が既定の読み方とわずかに違う。結果の stack_fast に残る)。"""
    from . import fish
    path, q, cname = _fish_args(s, contrast, spec, p, ("layers", "segments", "same_reply", "stack", "stack_fast", "response_tokens"))
    q.pop("iv", None)
    d = fish.direction_step(s, path, name=q.pop("name", None) or cname, **q)
    return {**d, "contrast_path": path}


def _fish_keys(s, contrast=None, spec: dict | None = None, with_units=None, **p) -> dict:
    """FiSH の鍵をまとめて作る(区切りごとの位置の鍵、形に要る場面の鍵・ほぼ 1 を返す鍵)。with_units(ユニットのリスト)を渡すと、
    それを当てた状態で作る(iv に足す。レシピを当てた状態で鍵を作り直すとき)。返事は質問の群ごとに 1 回だけ作り、区切りごとの
    位置の鍵は 1 回の読み取りから全部作る(1 本ずつ作ったときと同じ鍵になる)。
    key_fit="ridge" で、位置の鍵を最小二乗の鍵(ますごとの重み)にする(既定は mean = いまの作り方。表の fish.key_fit でも書ける)。
    key_c はその正則化の強さ(省くと質問単位の 5 分割で決める)。結果の key_fit・ridge に残る。"""
    from . import fish
    path, q, cname = _fish_args(s, contrast, spec, p, ("layers", "segments", "form", "pseudo", "max_tokens", "iv", "key_fit", "key_c"))
    if with_units:
        q["iv"] = fish.iv_units(s, q.get("iv"), [str(u) for u in with_units])
    d = fish.keys_step(s, path, base=q.pop("base", None) or cname, **q)
    return {**d, "contrast_path": path, "with_units": bool(with_units)}


def _fish_verify(s, units, iv=None, texts=None) -> dict:
    """FiSH のユニットの LoRA が式どおりか(gne nw verify)。"""
    from . import fish
    return fish.verify_step(s, list(units), iv=iv or None, texts=texts)


def _fish_check(s, units, contrast=None, spec: dict | None = None, **p) -> dict:
    """FiSH のユニットが、場面・返事の位置ごとに実際に足した量(gne nw check)。質問ごとの値(区間ごとに、ほとんど書けていない
    質問の数・狙った量を超えた質問の数と、質問ごとの行)も入る。low・over・leak・top でその目安と件数、save で結果のファイルの名前。
    結果の leaks は、書き込みたくない場面の群ごと・区間ごとの漏れの表(leak 以上を書いた質問の数、書いた量の平均、位置 0 だけの数)、
    key_fit は、ユニットの鍵がどちらの作り方(mean / ridge)で作ったものか。"""
    from . import fish
    path, q, _ = _fish_args(s, contrast, spec, p, ("max_tokens", "iv", "segments"))
    return {**fish.check_step(s, list(units), path, **q), "contrast_path": path}


def _fish_behave(s, units, contrast=None, spec: dict | None = None, **p) -> dict:
    """FiSH のユニットを当てて返事を作り、読めるように返す(狙う場面の質問と、書き込みたくない場面の質問)。"""
    from . import fish
    path, q, cname = _fish_args(s, contrast, spec, p, ("iv",))
    return {**fish.behave_step(s, list(units), path, name=q.pop("name", None) or cname, **q), "contrast_path": path}


def _fish_try(s, contrast=None, spec: dict | None = None, direction: str | None = None, **p) -> dict:
    """重みに入れる前に試す(gne nw try、fish.try_step): 区切りつきの方向を、狙う場面の質問の返事の位置どおりに層の出力へ直接足して
    返事を作る。direction を省くと対照集合の名前の方向。layers を省くと方向を作ったときの重ねる層。alpha は区切りごとの強さ。"""
    from . import fish
    path, q, cname = _fish_args(s, contrast, spec, p, ("iv", "alpha", "layers"))
    return {**fish.try_step(s, path, direction=direction or q.get("name") or cname, **q), "contrast_path": path}


def _fish_pipeline(s, contrast=None, spec: dict | None = None, summary: bool = False, **p) -> dict:
    """FiSH を最初から最後まで(方向 → ニューロンの表 → 鍵 → ユニット → 当てた状態で鍵を作り直す → 式どおりか → 足した量)。
    層(layers)は必須("11-23" のような範囲でもよい)。ニューロンは既定の候補(表の上から)か neurons({層: [番号…]})。強さ alpha は
    渡した値をそのまま使う(数 1 つなら全部の区切りに同じ値、"1,1,0.8" か [1, 1, 0.8] なら区切りごと)。
    save(名前か .yaml のパス)を渡すと、できたユニットをレシピのファイルに書く(結果の recipe_path)。summary=True なら、結果の
    代わりに短い要約(fish.summary: 式どおりか・ユニットとニューロンの数・保存先・足した量・質問ごとの値の数)と、ユニットの書き方の
    リストだけを返す。direction(すでにある方向の名前)を渡すと、方向を作り直さずにそれを使う(層・区切りが合わなければエラー)。
    cap は上限つきの形のしきい値("0.4-0.8" か、区切りごとに "0.4-0.8,0.3-0.7,0.3-0.7"・リスト。省くと既定)。
    stack_fast=True で、方向を作る段の読み直しを速い読み方にする(既定は切。方向がわずかに変わる。探索向け。direction と一緒に
    渡すと効かず、注意が出る)。どちらの読み方の方向を使ったかは、結果・要約の direction_stack_fast。
    key_fit="ridge" で、位置の鍵を最小二乗の鍵(ますごとの重み)にする(既定は mean = いまの作り方。当てた状態で鍵を作り直す段にも
    同じ作り方を使う)。key_c はその正則化の強さ(省くと 5 分割で決める)。どちらで作ったかは、結果・要約の key_fit・ridge。"""
    from . import fish
    path, q, cname = _fish_args(s, contrast, spec, p, fish.SETTINGS)
    if isinstance(q.get("neurons"), dict):
        q["neurons"] = {int(k): v for k, v in q["neurons"].items()}
    d = fish.pipeline(s, path, name=q.pop("name", None) or cname, **q)
    return {**d["summary"], "units": d["units"], "contrast_path": d["contrast_path"]} if summary else d


def compact_key_causal(d: dict) -> dict:
    """key-causal の結果を MCP 向けに小さくする: 比較用の鍵ごとの値(values)と、副作用の群ごとの細目を落とす(要約は残す)。"""
    for r in d.get("rows") or []:
        for k in ("pseudo", "probe_pseudo"):
            if isinstance(r.get(k), dict):
                r[k].pop("values", None)
        ps = r.get("pseudo_side") or {}
        for k in list(ps):
            if isinstance(ps[k], dict):
                ps[k].pop("values", None)
    for c in d.get("candidates") or []:
        for b in (c.get("by_alpha") or {}).values():
            for k in ("pseudo", "probe_pseudo", "pseudo_side_b"):
                if isinstance(b.get(k), dict):
                    b[k].pop("values", None)
    return d


def _direction_key_causal(s, contrast=None, spec: dict | None = None, compact: bool = False, **p) -> dict:
    """keyed の効く場所を探す第 1 段(gne direction key-causal、keyed_causal.key_causal): 候補(層 × 書き込み先)ごとに鍵を作り、
    生成なしで効き目・副作用・比較用の鍵の値を測る。contrast は対照集合のパスか、spec に対照集合の表(UI の入力欄。
    <root>/contrasts/<name>.yaml に書く)。compact なら比較用の鍵ごとの値を落とす(MCP 向け)。"""
    if (contrast is None) == (spec is None):
        raise GneError("BAD_ARGS", "対照集合は contrast(パス)か spec(表)のどちらかで渡してください")
    d = s.direction_key_causal(spec if spec is not None else contrast, progress=s.progress, **p)
    return compact_key_causal(d) if compact else d


def _direction_key_causal_verify(s, units=None, seeds=None, **p) -> dict:
    """第 2 段(gne direction key-causal-verify、keyed_causal.key_causal_verify): 選んだ候補(keyed のユニットのリストか、空白・
    改行区切りの文字列)と比較用の鍵 1 本ずつで、作成に使っていない質問に生成させて振る舞いを数える。seeds はリストか "1,2,3"。
    combos は複数のユニットを 1 つの候補として確かめる組(組ごとにユニットのリストか、カンマ区切りの文字列)。"""
    if isinstance(seeds, str):
        try:
            seeds = [int(x) for x in seeds.replace(" ", "").split(",") if x]
        except ValueError as e:
            raise GneError("BAD_ARGS", f"seeds は 1,2,3 のように書きます({seeds!r})") from e
    if isinstance(p.get("label"), str):
        p["label"] = [x for x in p["label"].split(",") if x.strip()]
    if p.get("judge") is not None and not str(p["judge"]).strip():
        p["judge"] = None
    return s.direction_key_causal_verify(units or [], seeds=list(seeds) if seeds else None, progress=s.progress, **p)


def _behave(s, prompts, recipes=None, seeds=None, **p) -> dict:
    """振る舞いを数える(gne behave、Session.behave)。prompts は質問集のパス(対照集合なら A 側の質問)、configs/contrast/ の名前、
    または質問のリスト。recipes はレシピの名前かパス(1 つなら文字列でも可)。seeds はリストか "1,2,3"。進み具合は質問ごと・種ごと。"""
    if isinstance(prompts, str) and not Path(prompts).exists():
        prompts = s._contrast_file(prompts)                    # configs/contrast/ の名前(無ければ FILE_NOT_FOUND)
    if isinstance(recipes, str):
        recipes = [recipes]
    if isinstance(p.get("labels"), (list, tuple)):
        p["labels"] = ",".join(str(x) for x in p["labels"])
    if p.get("judge") is not None and not str(p["judge"]).strip():
        p["judge"] = None
    if isinstance(seeds, str):
        try:
            seeds = [int(x) for x in seeds.replace(" ", "").split(",") if x]
        except ValueError as e:
            raise GneError("BAD_ARGS", f"seeds は 1,2,3 のように書きます({seeds!r})") from e
    return s.behave(prompts, recipes=list(recipes or []), seeds=list(seeds) if seeds else None, **p)


def standard_kinds() -> dict[str, Callable[..., dict]]:
    """MCP(job_start)と Web UI が共通で使うジョブの種類。"""
    return {
        "stats_resid": lambda s, **p: s.stats_resid(**p),
        "stats_act": lambda s, **p: s.stats_act(**p),
        "lens_logit": lambda s, layer, **p: s.lens_logit(layer, **p),
        "lens_steer": lambda s, layer, **p: s.lens_steer(layer, **p),
        "sweep": lambda s, layer, **p: s.sweep(layer, **p),
        "sweep_rank": _sweep_rank,
        "rank": lambda s, run, **p: s.rank(run, **p),
        "reverse": lambda s, **p: s.reverse(**p),
        "connections": lambda s, **p: s.connections(**p),
        "layout": lambda s, **p: s.layout(**p),
        "embed": lambda s, run, **p: s.embed(run, **p),
        "eval": _eval,
        "direction": lambda s, **p: s.direction(**p),
        "heads_contrast": lambda s, **p: s.heads_contrast(**p),
        "writers": lambda s, **p: s.writers(**p),
        "experts_contrast": lambda s, **p: s.experts_contrast(**p),
        "experts_sweep": lambda s, layer, **p: s.experts_sweep(layer, **p),
        "ablate": lambda s, direction, out, **p: s.ablate(direction, out, **p),
        "heads_causal": lambda s, direction, **p: s.heads_causal(direction, **p),        # CHANGE-06 C
        "heads_allocate": lambda s, direction, **p: s.heads_allocate(direction, **p),    # CHANGE-06 D
        "head_pipeline": lambda s, spec, **p: s.heads_pipeline(spec, **p),             # 方向 → 基準層 → 因果 → 配分
        "heads_candidates": lambda s, alloc, **p: s.heads_candidates(alloc, **p),       # 組 × 倍率で生成 → 採点のファイル
        "heads_refine": lambda s, name, **p: s.heads_refine(name, **p),                 # 残りの質問で追加のヘッドを探す
        "fieldtest": lambda s, **p: s.fieldtest(**p),                                   # 本番と同じ動かし方で、ファイルどうしを比べる
        "headproj_export": _headproj_export,
        "direction_key": _direction_key,                                                # CHANGE-07: 鍵の作成
        "direction_key_check": _direction_key_check,                                    # CHANGE-07: 鍵の分離の確認
        "direction_key_experts": _direction_key_experts,                                # CHANGE-07 MoE: 専門家の選ばれ方
        "direction_key_pipeline": _direction_key_pipeline,                              # CHANGE-07: 方向 → 鍵 → key-check
        "direction_key_verify": _direction_key_verify,                                  # CHANGE-07 V30: LoRA が式どおりか
        "direction_key_causal": _direction_key_causal,                                  # keyed の効く場所: 生成なしのふるい分け
        "direction_key_causal_verify": _direction_key_causal_verify,                    # keyed の効く場所: 生成での確かめ
        "behave": _behave,                                                              # 振る舞いを数える(gne behave)
        "nw_neurons": lambda s, layers, **p: s.nw_neurons(_layers(layers), **p),         # 計算で作るニューロン: ふだんの働きの表
        "nw_check": lambda s, recipe, contrast, **p: s.nw_check(recipe, contrast, **p),  # 計算で作るニューロン: 位置ごとの鍵の反応と足した量
        "fish_direction": _fish_direction,                                              # FiSH(Web UI): 方向
        "fish_keys": _fish_keys,                                                        # FiSH: 鍵をまとめて作る
        "fish_verify": _fish_verify,                                                    # FiSH: 式どおりか
        "fish_check": _fish_check,                                                      # FiSH: 場面・位置ごとに足した量
        "fish_behave": _fish_behave,                                                    # FiSH: レシピを当てた返事を読む
        "fish_try": _fish_try,                                                          # FiSH: 重みに入れる前に試す(層の出力に直接足す)
        "fish_pipeline": _fish_pipeline,                                                # FiSH: 最初から最後まで
    }


class JobRunner:
    """ジョブを 1 本ずつ順に回すスレッド。kinds: 種類 → fn(session, **params) -> dict。"""

    def __init__(self, session, arbiter: GpuArbiter, kinds: dict[str, Callable[..., dict]]):
        self.session, self.arbiter, self.kinds = session, arbiter, kinds
        self.jobs: dict[str, Job] = {}
        self._queue: list[str] = []
        self._cv = threading.Condition()
        self._thread = threading.Thread(target=self._loop, name="gne-jobs", daemon=True)
        self._thread.start()

    def start(self, kind: str, params: dict | None = None) -> Job:
        if kind not in self.kinds:
            raise GneError("BAD_ARGS", f"ジョブの種類 {kind!r} はありません({', '.join(self.kinds)})")
        from .estimate import NeedsConfirm, gate
        try:
            params, est = gate(self.session, kind, dict(params or {}))     # CHANGE-03 C: 予算と 30 分の確認
        except NeedsConfirm as e:
            e.error.estimate = e.estimate
            raise e.error from e
        job = Job(uuid.uuid4().hex[:8], kind, dict(params or {}))
        job.estimate = est
        with self._cv:
            self.jobs[job.id] = job
            self._queue.append(job.id)
            self._cv.notify_all()
        return job

    def list(self) -> list[Job]:
        return list(self.jobs.values())

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise GneError("BAD_ARGS", f"ジョブ {job_id} はありません")
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        job.cancel = True
        if job.status == "queued":
            job.status = "cancelled"
        return job

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._queue:
                    self._cv.wait()
                job = self.jobs[self._queue.pop(0)]
            if job.status == "cancelled":
                continue
            self._run(job)

    def _run(self, job: Job) -> None:
        me = threading.get_ident()
        s = self.session

        def progress(done: int, total: int, stage: str) -> None:
            if threading.get_ident() != me:          # 譲っている間の対話的な処理からの進捗は無視
                return
            job.done, job.total, job.stage = done, total, stage
            if job.cancel:
                raise _Cancelled()
            if not getattr(s, "no_yield", False) and self.arbiter.yield_point():   # 実地テストは llama-server に GPU を渡している間は譲らない
                job.yields += 1

        job.status, job.started = "running", time.monotonic()
        try:
            with self.arbiter.job():
                prev = s.progress
                s.progress = progress
                try:
                    job.result = self.kinds[job.kind](s, **job.params)
                finally:
                    s.progress = prev
            job.status = "done"
        except _Cancelled:
            job.status = "cancelled"
        except GneError as e:
            job.status, job.error = "error", {"code": e.code, "message": e.message, "hint": e.hint}
        except Exception as e:  # noqa: BLE001
            job.status, job.error = "error", {"code": "INTERNAL", "message": f"{type(e).__name__}: {e}",
                                              "hint": traceback.format_exc(limit=3)}
        finally:
            job.finished = time.monotonic()
