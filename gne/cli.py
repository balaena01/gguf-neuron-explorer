"""gne CLI。設計書 §7。gne.api の薄い皮。

共通オプション(--model, --root, --n-gpu-layers, --n-ctx, --n-seq-max, --seed, -v, --json)は
`gne --model X steer ...` と `gne steer --model X ...` のどちらの位置でも受け付ける。
--json のとき stdout には JSON だけを出す(§6.13 (a))。人向けの表示は stderr。
"""

from __future__ import annotations

import functools
import inspect
import json
import re
import sys
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from .schemas import SCHEMA_VERSION
from .steer import GneError

app = typer.Typer(add_completion=False, no_args_is_help=True, help="GGUF の MLP ニューロンを調べて押す")
stats_app = typer.Typer(no_args_is_help=True, help="統計(残差ノルム ρ_L・ニューロン活性)")
app.add_typer(stats_app, name="stats")
lens_app = typer.Typer(no_args_is_help=True, help="一次ふるい(logit lens・steer lens)")
app.add_typer(lens_app, name="lens")


@dataclass
class Common:
    model: Optional[Path] = None
    root: Optional[Path] = None
    n_gpu_layers: int = -1
    n_ctx: int = 4096
    n_seq_max: int = 8
    seed: int = 0
    verbose: bool = False
    json: bool = False

    def merged(self, over: dict) -> Common:
        kw = {k: v for k, v in over.items() if v is not None and k not in ("verbose", "json")}
        return replace(self, **kw, verbose=self.verbose or bool(over.get("verbose")),
                       json=self.json or bool(over.get("json")))

    def session(self, model: Optional[Path] = None):
        from .api import Session
        path = model or self.model
        if path is None:
            raise GneError("BAD_ARGS", "GGUF を指定してください", "--model か環境変数 GNE_MODEL")
        return Session(path, root=self.root, n_gpu_layers=self.n_gpu_layers, n_ctx=self.n_ctx,
                       n_seq_max=self.n_seq_max, seed=self.seed, verbose=self.verbose, progress=self.progress)

    def progress(self, done: int, total: int, stage: str) -> None:
        now = time.monotonic()
        if done != total and now - getattr(self, "_last_progress", 0.0) < 0.5:   # 0.5 秒ごとに間引く
            return
        self._last_progress = now
        if self.json:
            _emit({"progress": {"stage": stage, "done": done, "total": total}})
        else:
            print(f"\r{stage}: {done}/{total}", end="\n" if done == total else "", file=sys.stderr, flush=True)

    def human(self, text: str) -> None:
        print(text, file=sys.stderr if self.json else sys.stdout, flush=True)


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n")
    sys.stdout.flush()


def _json_default(x):
    import numpy as np
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return str(x)


def _common_params() -> list[inspect.Parameter]:
    """各コマンドにも付ける共通オプション。既定値 None で受け、グループ側(gne --model X ...)の値に重ねる。"""
    K = inspect.Parameter.KEYWORD_ONLY
    return [
        inspect.Parameter("model", K, annotation=Optional[Path],
                          default=typer.Option(None, "--model", "-m", envvar="GNE_MODEL", help="GGUF ファイル")),
        inspect.Parameter("root", K, annotation=Optional[Path],
                          default=typer.Option(None, "--root", envvar="GNE_ROOT", help="保存先(既定 ~/.gne)")),
        inspect.Parameter("n_gpu_layers", K, annotation=Optional[int],
                          default=typer.Option(None, "--n-gpu-layers", help="GPU に載せる層数(-1 = 全部)")),
        inspect.Parameter("n_ctx", K, annotation=Optional[int],
                          default=typer.Option(None, "--n-ctx", help="コンテキスト長(シーケンス間で共有)")),
        inspect.Parameter("n_seq_max", K, annotation=Optional[int],
                          default=typer.Option(None, "--n-seq-max", help="1 バッチの最大シーケンス数")),
        inspect.Parameter("seed", K, annotation=Optional[int], default=typer.Option(None, "--seed")),
        inspect.Parameter("verbose", K, annotation=bool,
                          default=typer.Option(False, "--verbose", "-v", help="llama.cpp のログを出す")),
        inspect.Parameter("json", K, annotation=bool,
                          default=typer.Option(False, "--json", help="stdout に JSON だけを出す")),
    ]


_COMMON_NAMES = [p.name for p in _common_params()]


def gne_command(typer_app: typer.Typer, name: str, schema_name: str | None = None):
    """本体 f(c: Common, holder: dict, ...) -> (payload, human_text, exit_code) をコマンドとして登録する。
    holder["session"] に Session を入れておくと、エンベロープの model_id と後始末に使う。"""

    def deco(f: Callable):
        sig = inspect.signature(f, eval_str=True)
        own = list(sig.parameters.values())[2:]                       # c と holder を除く
        ctx_param = inspect.Parameter("ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=typer.Context)

        @functools.wraps(f)
        def wrapper(ctx: typer.Context, **kwargs):
            over = {k: kwargs.pop(k) for k in _COMMON_NAMES}
            c = (ctx.obj or Common()).merged(over)
            _run(c, schema_name or name, f, kwargs)

        wrapper.__signature__ = sig.replace(parameters=[ctx_param, *own, *_common_params()])
        wrapper.__annotations__ = {}
        typer_app.command(name)(wrapper)
        return f

    return deco


def _envelope(command: str, c: Common, session=None) -> dict:
    if session is not None:
        return session.envelope(command)
    from .engine import Engine
    return {"schema_version": SCHEMA_VERSION, "command": command, "model_id": None,
            "llama_cpp_version": Engine.llama_cpp_version(), "seed": c.seed,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "warnings": []}


def _run(c: Common, command: str, f: Callable, kwargs: dict) -> None:
    holder: dict = {}
    code = 0
    try:
        payload, human, code = f(c, holder, **kwargs)
        s = holder.get("session")
        if c.json:
            _emit({**_envelope(command, c, s), **payload})
        else:
            if human:
                c.human(human)
            for w in (s.warnings if s else []):
                print(f"警告: {w}", file=sys.stderr)
    except GneError as e:
        _fail(c, command, holder.get("session"), e.code, e.message, e.hint)
        code = 1
    except Exception as e:  # noqa: BLE001
        if c.verbose:
            import traceback
            traceback.print_exc()
        _fail(c, command, holder.get("session"), "INTERNAL", f"{type(e).__name__}: {e}", None)
        code = 1
    finally:
        s = holder.get("session")
        if s is not None:
            s.close()
    if code:
        raise typer.Exit(code)


def _fail(c: Common, command: str, session, code: str, message: str, hint: str | None) -> None:
    if c.json:
        _emit({**_envelope(command, c, session), "error": {"code": code, "message": message, "hint": hint}})
    else:
        print(f"エラー[{code}]: {message}" + (f"\nヒント: {hint}" if hint else ""), file=sys.stderr)


@app.callback()
def main(ctx: typer.Context,
         model: Optional[Path] = typer.Option(None, "--model", "-m", envvar="GNE_MODEL", help="GGUF ファイル"),
         root: Optional[Path] = typer.Option(None, "--root", envvar="GNE_ROOT", help="保存先(既定 ~/.gne)"),
         n_gpu_layers: int = typer.Option(-1, "--n-gpu-layers", help="GPU に載せる層数(-1 = 全部)"),
         n_ctx: int = typer.Option(4096, "--n-ctx", help="コンテキスト長(シーケンス間で共有)"),
         n_seq_max: int = typer.Option(8, "--n-seq-max", help="1 バッチの最大シーケンス数"),
         seed: int = typer.Option(0, "--seed"),
         verbose: bool = typer.Option(False, "--verbose", "-v", help="llama.cpp のログを出す"),
         json_: bool = typer.Option(False, "--json", help="stdout に JSON だけを出す")):
    ctx.obj = Common(model, root, n_gpu_layers, n_ctx, n_seq_max, seed, verbose, json_)


# --- 表示の補助 ------------------------------------------------------------------------------------------

def _width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def _table(rows: list[tuple[str, str]]) -> str:
    w = max(_width(k) for k, _ in rows)
    return "\n".join(f"{k}{' ' * (w - _width(k))}  {v}" for k, v in rows)


def parse_layers(spec: Optional[str]) -> Optional[list[int]]:
    """"all" / "10-16" / "1,5,9-12" → 層のリスト(all は None)。"""
    if spec is None or spec == "all":
        return None
    out: list[int] = []
    for part in spec.split(","):
        m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", part)
        if not m:
            raise GneError("BAD_ARGS", f"層の指定 {spec!r} を読めません", "all / 10-16 / 1,5,9-12")
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        out.extend(range(a, b + 1))
    return sorted(set(out))


def _units_text(units: list[dict]) -> str:
    rows = []
    for u in units:
        who = f"N{u['neuron']}" if u["kind"] == "neuron" else f"random{u['seed']}"
        rp = f"rel_push {u['rel_push']:.2f}、ρ={u['rho']:.2f}" if u.get("rel_push") is not None else "ρ 不明"
        add = f"、活性に {u['add']:+.3g} 相当" if u.get("add") is not None else ""
        rows.append(f"  L{u['layer']} {who} dose {u['dose']:+g}: ‖v‖={u['norm']:.2f}({rp}{add})")
    return "\n".join(rows)


# --- info / probe / verify / bench ---------------------------------------------------------------------

@gne_command(app, "info")
def info(c: Common, holder: dict,
         load: bool = typer.Option(True, "--load/--no-load", help="llama.cpp で読み込めるか確かめる(R6)")):
    """GGUF メタデータ、arch 判定、警告、dequantize 可否、chat template、bake の可否。"""
    s = holder["session"] = c.session()
    d = s.info(load=load)
    m = d["meta"]
    b = d["bake"]
    rows = [
        ("file", f"{m['path']} ({m['file_size'] / 2**30:.2f} GiB)"),
        ("model_id", m["model_id"]),
        ("name / arch", f"{m['name']} / {m['arch']}"),
        ("n_layer / n_embd / n_ff", f"{m['n_layer']} / {m['n_embd']} / {m['n_ff']}"
         + (" (層ごとに異なる)" if m["n_ff_per_layer"] and len(set(m["n_ff_per_layer"])) > 1 else "")),
        ("n_vocab / n_expert", f"{m['n_vocab']} / {m['n_expert']}"),
        ("ffn_down 量子化型", m["ffn_down_qtype"]),
        ("output", "token_embd を共有(tied)" if m["tied_output"] else "output.weight"),
        ("post_ffw_norm / ffn_down.scale",
         f"{'あり' if m['has_post_ffn_norm'] else 'なし'} / {'あり' if m['ffn_down_scale'] else 'なし'}"),
        ("chat template", "あり" if m["has_chat_template"] else "なし"),
        ("bake", (f"可({b['mode']})" if b["supported"] else "不可") + f": {b['reason']}(表: {b['checked_llama_cpp']})"),
        ("統計", f"resid {'あり' if d['stats']['resid'] else 'なし'}、act の層 {d['stats']['act_layers'] or 'なし'}"),
    ]
    lc = d["llama_cpp"]
    if lc is not None:
        rows.append(("llama.cpp", f"読み込み可(n_layer {lc['n_layer']}, n_embd {lc['n_embd']}, n_vocab {lc['n_vocab']})"
                     if lc["ok"] else f"読み込めません: {lc['error']}(新しい llama.cpp でのビルドが必要かもしれません、R6)"))
    text = _table(rows)
    text += "".join(f"\n警告: {w}" for w in d["arch_warnings"]) or "\n警告なし: dense FFN、ニューロン押しは control vector と等価です"
    return d, text, 0


@gne_command(app, "probe")
def probe(c: Common, holder: dict,
          list_names: bool = typer.Option(False, "--list-names", help="cb_eval で観測できるテンソル名の一覧"),
          layer: Optional[int] = typer.Option(None, "--layer", help="この層の名前だけ表示"),
          prompt: str = typer.Option("Who are you?", "--prompt")):
    """cb_eval で観測できるテンソル名(1 回 prefill)。"""
    from .capture import pick_act_name
    if not list_names:
        raise GneError("BAD_ARGS", "--list-names を付けてください")
    s = holder["session"] = c.session()
    e = s.engine
    names = e.list_names(e.tokenize_prompt(e.render_chat([{"role": "user", "content": prompt}])))
    groups: dict[str, set[int]] = {}
    other = []
    for n in names:
        mt = re.fullmatch(r"(.+)-(\d+)", n)
        if mt:
            groups.setdefault(mt.group(1), set()).add(int(mt.group(2)))
        elif not re.fullmatch(r"node_\d+", n) and " (" not in n:
            other.append(n)
    act_layer = layer if layer is not None else e.n_layer // 2
    act = pick_act_name(names, act_layer)
    shown = sorted(x for x in names if layer is None or x.endswith(f"-{layer}"))
    payload = {"layer": layer, "names": shown, "groups": {k: sorted(v) for k, v in sorted(groups.items())},
               "act_name": act}
    if layer is not None:
        return payload, "\n".join(shown) + f"\n→ ffn_down の入力: {act or '見つかりません'}", 0
    lines = [f"層ごとの名前(層の範囲、{e.n_layer} 層):"]
    for base in sorted(groups):
        ls = sorted(groups[base])
        rng = f"{ls[0]}..{ls[-1]}" if ls == list(range(ls[0], ls[-1] + 1)) else ",".join(map(str, ls))
        lines.append(f"  {base:<24} {rng}")
    lines.append("層に属さない名前: " + ", ".join(sorted(other)))
    lines.append(f"→ ffn_down の入力(層 {act_layer}): {act or '見つかりません'}")
    return payload, "\n".join(lines), 0


@gne_command(app, "verify")
def verify(c: Common, holder: dict,
           layer: Optional[int] = typer.Option(None, "--layer", help="検証する層(既定 n_layer/2)"),
           llama_completion: Optional[str] = typer.Option(None, "--llama-completion",
                                                          help="V4 の生成比較に使う llama-completion(既定: build/ か PATH)"),
           hf_model: Optional[str] = typer.Option(None, "--hf-model", help="V5: HF 形式のモデルディレクトリ"),
           hf_python: Optional[str] = typer.Option(None, "--hf-python", help="V5: torch と transformers がある python"),
           bench: bool = typer.Option(True, "--bench/--no-bench", help="V7 のベンチも回す"),
           markdown: Optional[Path] = typer.Option(None, "--markdown", help="結果の表を Markdown で保存")):
    """Phase 0 の等価性・層対応テスト一式(V1〜V8)。"""
    from dataclasses import asdict
    from .bench import bench_markdown, run_bench
    from .verify import VResult, Verifier, find_llama_completion
    s = holder["session"] = c.session()
    lc = llama_completion or find_llama_completion()
    v = Verifier(s.engine, s.weights, layer=layer, seed=c.seed)
    steps = [("V1", v.v1_shapes), ("V2", v.v2_ffn_linearity), ("V3", v.v3_cvec_layer),
             ("V4", lambda: v.v4_direction_file(lc)), ("V5", lambda: v.v5_neuron_push(hf_model, hf_python)),
             ("V6", v.v6_attention_sink), ("V12", v.v12_when_gen)]
    results, bench_result = [], None

    def show(r):
        results.append(r)
        c.human(f"[{r.status.upper():4}] {r.id} {r.title}: {r.detail}")

    for vid, fn in steps:
        try:
            show(fn())
        except Exception as ex:  # noqa: BLE001
            show(VResult(vid, "例外", "fail", repr(ex)))
    if bench:
        bench_result = run_bench(s.engine, layer=v.L, reps=5)
        show(VResult("V7", "ベンチ", "info",
                     f"set_cvec+prefill {bench_result['set_plus_prefill_ms']:.1f} ms(うち再予約 {bench_result['reserve_ms']:.1f} ms)、"
                     f"prefill {bench_result['prefill_ms']:.1f} ms、decode 1 step {bench_result['decode_step_ms']:.1f} ms、"
                     f"生成 128 tok×4 {bench_result['gen']['batch']['ms']:.0f} ms", bench_result))
    else:
        show(VResult("V7", "ベンチ", "skip", "--no-bench"))
    show(v.v8_tokenize())
    n_fail = sum(not r.ok for r in results)
    if markdown:
        md = ["| ID | 結果 | 内容 |", "|---|---|---|"] + [f"| {r.id} | {r.status} | {r.detail.replace('|', '/')} |"
                                                       for r in results]
        markdown.write_text("\n".join(md) + "\n" + ("\n" + bench_markdown(bench_result) if bench_result else ""))
    payload = {"layer": v.L, "results": [json.loads(json.dumps(asdict(r), default=_json_default)) for r in results],
               "n_pass": len(results) - n_fail, "n_fail": n_fail}
    text = (f"\n{s.meta.name}({s.meta.model_id}): {len(results) - n_fail}/{len(results)} 合格"
            + (f"、{n_fail} 件不合格" if n_fail else ""))
    return payload, text, 1 if n_fail else 0


@gne_command(app, "bench")
def bench(c: Common, holder: dict,
          layer: Optional[int] = typer.Option(None, "--layer"),
          reps: int = typer.Option(10, "--reps"),
          gen_tokens: int = typer.Option(128, "--gen-tokens"),
          markdown: Optional[Path] = typer.Option(None, "--markdown", help="Markdown の表をこのファイルに追記")):
    """set_cvec / prefill / decode / 生成の実測(docs/measurements.md 向け)。"""
    from .bench import bench_markdown, run_bench
    s = holder["session"] = c.session()
    r = run_bench(s.engine, layer=layer, reps=reps, gen_tokens=gen_tokens, seed=c.seed)
    md = bench_markdown(r)
    if markdown:
        with open(markdown, "a") as fh:
            fh.write(md + "\n")
    return {"bench": json.loads(json.dumps(r, default=_json_default))}, md, 0


# --- stats ------------------------------------------------------------------------------------------------

_CORPUS = typer.Option(None, "--corpus", help="テキストファイル(既定: configs/corpus/default.txt)")
_TOKENS = typer.Option(None, "--tokens", help="使うトークン数の上限(コーパス全体から等間隔に選ぶ)")
_LAYERS = typer.Option("all", "--layers", help="all / 10-16 / 1,5,9-12")
_LAST = typer.Option(False, "--include-last-layer", help="最終層も含める(logits_all で全行を出す、R5)")
_CHUNK = typer.Option(512, "--chunk-tokens", help="1 シーケンスのトークン数")


@gne_command(stats_app, "resid", "stats resid")
def stats_resid(c: Common, holder: dict, corpus: Optional[Path] = _CORPUS, tokens: Optional[int] = _TOKENS,
                layers: str = _LAYERS, include_last_layer: bool = _LAST, chunk_tokens: int = _CHUNK):
    """残差ノルム ρ_L(position ≥ 1 の ‖l_out-L‖ の中央値など)を計算して保存する。"""
    s = holder["session"] = c.session()
    d = s.stats_resid(corpus=corpus, tokens=tokens, layers=parse_layers(layers),
                      include_last_layer=include_last_layer, chunk_tokens=chunk_tokens)
    st = d["stats"]
    rows = [f"  L{L:<3} p10 {a:8.2f}  p50 {b:8.2f}  p90 {q:8.2f}"
            for L, a, b, q in zip(st["layers"], st["p10"], st["p50"], st["p90"], strict=True)]
    text = (f"{d['path']}({st['n_tokens']} トークン/層、コーパス {d['corpus']['sha']}、{d['seconds']:.1f} 秒)\n"
            + "\n".join(rows))
    return d, text, 0


@gne_command(stats_app, "act", "stats act")
def stats_act(c: Common, holder: dict, corpus: Optional[Path] = _CORPUS, tokens: Optional[int] = _TOKENS,
              layers: str = _LAYERS, include_last_layer: bool = _LAST, chunk_tokens: int = _CHUNK):
    """ニューロン活性の統計(|a| の absmax・q50/q99/q999・RMS・平均)を計算して保存する。"""
    s = holder["session"] = c.session()
    d = s.stats_act(corpus=corpus, tokens=tokens, layers=parse_layers(layers),
                    include_last_layer=include_last_layer, chunk_tokens=chunk_tokens)
    rows = [f"  L{r['layer']:<3} {r['tensor_name']:<18} absmax 中央値 {r['absmax_median']:.3g}  "
            f"q999 中央値 {r['q999_median']:.3g}  rms 中央値 {r['rms_median']:.3g}" for r in d["layers"]]
    text = f"{len(d['layers'])} 層を保存(コーパス {d['corpus']['sha']}、{d['seconds']:.1f} 秒)\n" + "\n".join(rows)
    return d, text, 0


# --- steer / export / import / bake --------------------------------------------------------------------

_RECIPE = typer.Option(None, "--recipe", help="レシピ YAML")
_UNIT = typer.Option(None, "--unit", help="L:N:dose(例 14:8949:+1.0)または L:random:SEED:dose。複数可")
_DOSE_UNIT = typer.Option(None, "--dose-unit", help="rel / absmax / raw(既定はレシピの値、--unit なら rel)")
_SCALE = typer.Option(1.0, "--scale", help="全ユニットの dose に掛ける係数")
_WHEN = typer.Option(None, "--when", help="足す位置(§6.15): all / gen / assistant / after:<文字列> / ramp:<a>-<b>。既定はレシピの値(無ければ all)")


@gne_command(app, "steer")
def steer(c: Common, holder: dict, recipe: Optional[Path] = _RECIPE, unit: Optional[list[str]] = _UNIT,
          dose_unit: Optional[str] = _DOSE_UNIT, scale: float = _SCALE,
          prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="ユーザーの発話"),
          system: Optional[str] = typer.Option(None, "--system", help="system プロンプト"),
          max_tokens: int = typer.Option(128, "--max-tokens"),
          compare: bool = typer.Option(False, "--compare", help="無介入の生成も並べる"),
          when: Optional[str] = _WHEN):
    """レシピまたは単発指定でプロンプトに回答する(greedy、1 本ずつ)。"""
    s = holder["session"] = c.session()
    r = s.recipe(units=unit, path=recipe, dose_unit=dose_unit, scale=scale, when=when)
    d = s.steer(r, prompt=prompt, system=system, max_tokens=max_tokens, compare=compare)
    parts = [f"ユニット({r.dose_unit}、位置 {r.when}):\n" + _units_text(d["units"])]
    if d["baseline"] is not None:
        parts.append(f"── 無介入 ──({d['baseline']['finish']}, {d['baseline']['n_gen']} tok)\n{d['baseline']['text']}")
    label = ", ".join(u.label() for u in r.units)
    parts.append(f"── 介入: {label} ──({d['steered']['finish']}, {d['steered']['n_gen']} tok)\n{d['steered']['text']}")
    if d["differs"] is not None:
        parts.append("無介入との差: " + ("あり" if d["differs"] else "なし(同じ greedy 列)"))
    return d, "\n\n".join(parts), 0


_EXT = {"cvec": ".cvec.gguf", "npz": ".npz", "json": ".json", "safetensors": ".safetensors"}


@gne_command(app, "export-cvec")
def export_cvec(c: Common, holder: dict, recipe: Optional[Path] = _RECIPE, unit: Optional[list[str]] = _UNIT,
                dose_unit: Optional[str] = _DOSE_UNIT, scale: float = _SCALE,
                out: Optional[Path] = typer.Option(None, "--out", "-o", help="出力ファイル(既定 <root>/export/<レシピ名>.*)"),
                fmt: str = typer.Option("cvec", "--format", help="cvec / npz / json / safetensors"),
                verify: bool = typer.Option(False, "--verify", help="llama-completion --control-vector で greedy 出力を比べる"),
                verify_prompt: str = typer.Option("Who are you?", "--verify-prompt"),
                llama_completion: Optional[str] = typer.Option(None, "--llama-completion")):
    """レシピ → control vector GGUF(llama-cli / llama-server の --control-vector 用)。他形式も可。"""
    if fmt not in _EXT:
        raise GneError("BAD_FORMAT", f"--format {fmt!r} は cvec / npz / json / safetensors のどれかです")
    s = holder["session"] = c.session()
    r = s.recipe(units=unit, path=recipe, dose_unit=dose_unit, scale=scale)
    path = out or s.store.export_dir() / f"{r.name or 'units'}{_EXT[fmt]}"
    d = s.export_cvec(r, path, fmt=fmt, verify=verify, verify_prompt=verify_prompt, llama_completion=llama_completion)
    text = f"{d['path']}({fmt}、層 {', '.join(d['layer_norms'])})\n" + _units_text(d["units"])
    v = d["verify"]
    if v:
        text += (f"\n本家での再現({Path(v['tool']).name}、greedy {v['n_tokens']} トークン): "
                 f"{'一致' if v['match'] else '不一致'}、無介入との差: {'あり' if v['differs_from_baseline'] else 'なし'}")
        if not v["match"]:
            text += f"\n  gne:   {v['ours']!r}\n  llama: {v['theirs']!r}"
    return d, text, 0 if not v or v["match"] else 1


@gne_command(app, "import-cvec")
def import_cvec(c: Common, holder: dict, file: Path = typer.Argument(..., help="control vector GGUF"),
                source: str = typer.Option("exact", "--source", help="exact / generator(llama-cvector-generator の 1 ずれ補正)"),
                name: Optional[str] = typer.Option(None, "--name", help="保存名(既定はファイル名)")):
    """既存の control vector GGUF を方向として取り込む(逆引き用、<root>/reverse/<model_id>/)。"""
    if source not in ("exact", "generator"):
        raise GneError("BAD_ARGS", "--source は exact か generator です")
    s = holder["session"] = c.session()
    d = s.import_cvec(file, source=source, name=name)
    text = f"{d['path']}(source={source})\n" + "\n".join(f"  L{L}: ‖v‖={n:.2f}" for L, n in d["layer_norms"].items())
    return d, text, 0


@gne_command(app, "bake")
def bake(c: Common, holder: dict, recipe: Optional[Path] = _RECIPE, unit: Optional[list[str]] = _UNIT,
         dose_unit: Optional[str] = _DOSE_UNIT, scale: float = _SCALE,
         in_path: Optional[Path] = typer.Option(None, "--in", help="元の GGUF(既定 --model)"),
         out: Path = typer.Option(..., "--out", help="焼いた GGUF の出力先"),
         verify: bool = typer.Option(False, "--verify", help="焼いた GGUF と 元モデル + cvec の greedy 出力を比べる(V9)"),
         verify_prompt: str = typer.Option("Who are you?", "--verify-prompt")):
    """レシピを FFN バイアス(blk.L.ffn_down.bias)として焼き付けた GGUF を作る(対応 arch のみ)。"""
    if in_path and c.model and in_path.resolve() != c.model.resolve():
        raise GneError("BAD_ARGS", "--in と --model が違うファイルです")
    s = holder["session"] = c.session(in_path)
    r = s.recipe(units=unit, path=recipe, dose_unit=dose_unit, scale=scale)
    d = s.bake(r, out, verify=verify, verify_prompt=verify_prompt)
    rep, v = d["report"], d["verify"]
    text = (f"{rep['dst']}({rep['arch']}、mode={rep['mode']}、追加 {rep['added'] or 'なし'}、加算 {rep['modified'] or 'なし'})\n"
            + _units_text(d["units"]))
    ok = True
    if v:
        verdict = "完全一致" if v["exact"] else ("合格(僅差の分岐)" if v["match"] else "不一致")
        text += (f"\n検証(greedy {v['n_tokens']} トークン): 元モデル + cvec と{verdict}"
                 + (f"、{v['summary']}、次トークン KL {v['kl_next']:.2e}" if v["kl_next"] is not None else "")
                 + f"、無介入との差: {'あり' if v['differs_from_baseline'] else 'なし'}")
        ok = v["differs_from_baseline"] and (v["match"] or not v["ours"])
    return d, text, 0 if ok else 1


# --- lens / reverse(Phase 2) -----------------------------------------------------------------------------

_NEURONS = typer.Option("all", "--neurons", help="all / random:384 / 1,5,9-12")
_SHOW = typer.Option(20, "--show", help="表示するニューロン数(all / random のとき上位から)")


def _toks(ts: list[str], n: int = 8) -> str:
    return " ".join(repr(t) for t in ts[:n])


@gne_command(lens_app, "logit", "lens logit")
def lens_logit(c: Common, holder: dict, layer: int = typer.Option(..., "--layer"), neurons: str = _NEURONS,
               top_k: int = typer.Option(12, "--top-k"), show: int = _SHOW):
    """層の全(または指定)ニューロンの logit lens(重みのみ、推論なし)。"""
    s = holder["session"] = c.session()
    d = s.lens_logit(layer, neurons=neurons, top_k=top_k, show=show)
    lines = [f"{d['path']}({d['n_neurons']} ニューロン、{d['seconds']:.1f} 秒)"]
    for r in d["neurons"]:
        lines.append(f"  N{r['neuron']:<6} peak z {r['peak_z']:5.1f}  ↑ {_toks(r['up'])}\n{'':16}↓ {_toks(r['down'])}")
    return d, "\n".join(lines), 0


@gne_command(lens_app, "steer", "lens steer")
def lens_steer(c: Common, holder: dict, layer: int = typer.Option(..., "--layer"), neurons: str = _NEURONS,
               dose: float = typer.Option(0.8, "--dose", help="rel dose"),
               random_controls: int = typer.Option(64, "--random-controls"),
               top_k: int = typer.Option(12, "--top-k"), show: int = _SHOW,
               resume: bool = typer.Option(True, "--resume/--no-resume", help="保存済みの結果の続きから")):
    """層の全(または指定)ニューロンの steer lens(±dose で押したときの次トークンの変化、ランダム対照つき)。"""
    s = holder["session"] = c.session()
    d = s.lens_steer(layer, neurons=neurons, dose=dose, random_controls=random_controls, top_k=top_k,
                     resume=resume, show=show)
    nl = d["null"]
    lines = [f"{d['path']}(済み {d['n_done']}/{s.meta.n_ff_at(layer)}、対照 {d['n_controls']}、{d['seconds']:.1f} 秒)",
             f"対照の KL(+): 中央値 {nl['pos']['p50'] or 0:.3f} / 95% {nl['pos']['p95'] or 0:.3f} / 最大 {nl['pos']['max'] or 0:.3f}"]
    for r in d["neurons"]:
        lines.append(f"  N{r['neuron']:<6} KL+ {r['kl_pos']:.3f}({r['pct_pos']:.0%})  KL- {r['kl_neg']:.3f}({r['pct_neg']:.0%})"
                     f"\n{'':10}+ {_toks(r['up'])}\n{'':10}- {_toks(r['down'])}")
    return d, "\n".join(lines), 0


@gne_command(app, "reverse")
def reverse(c: Common, holder: dict,
            contrast: Optional[Path] = typer.Option(None, "--contrast", help="対照プロンプト集 YAML(configs/contrast/)"),
            directions: Optional[str] = typer.Option(None, "--directions", help="保存済みの方向(名前かパス。import-cvec の結果も可)"),
            method: str = typer.Option("mean_diff", "--method", help="mean_diff / pca1"),
            positions: str = typer.Option("all", "--positions", help="all / last / assistant"),
            top: int = typer.Option(200, "--top", help="候補の数"),
            layers: str = typer.Option("all", "--layers", help="all / 10-16 / 1,5,9-12"),
            verify: bool = typer.Option(False, "--verify", help="上位の候補に quick 設計の rollout を回す(対照つき)"),
            verify_top: Optional[int] = typer.Option(None, "--verify-top", help="--verify で回す候補の数(既定 --top)"),
            verify_controls: int = typer.Option(8, "--verify-controls", help="--verify の層ごとのランダム対照の数"),
            max_tokens: int = typer.Option(128, "--max-tokens"),
            name: Optional[str] = typer.Option(None, "--name", help="保存名(既定は対照プロンプト集の name)"),
            standardize: bool = typer.Option(True, "--standardize/--no-standardize",
                                             help="残差の次元ごとのばらつきで割った空間で cos を取る(F17)"),
            show: int = typer.Option(20, "--show")):
    """対照方向を作り、全層・全ニューロンを cos で順位づけする(逆引き)。"""
    if method not in ("mean_diff", "pca1") or positions not in ("all", "last", "assistant"):
        raise GneError("BAD_ARGS", "--method は mean_diff / pca1、--positions は all / last / assistant です")
    s = holder["session"] = c.session()
    d = s.reverse(contrast=contrast, directions=directions, method=method, positions=positions, top=top,
                  layers=parse_layers(layers), verify=verify, verify_top=verify_top, verify_controls=verify_controls,
                  max_tokens=max_tokens, name=name, standardize=standardize)
    lines = [f"{d['candidates_path']}({d['n_candidates']} 候補、{d['seconds']:.1f} 秒)",
             "層ごとの |cos| 上位 1% の平均: " + " ".join(f"L{r['layer']}:{r['top1pct_abs_cos']:.3f}" for r in d["layers"])]
    for i, cd in enumerate(d["candidates"][:show]):
        extra = "".join(f"  {k} {cd[k]:+.3g}" for k in ("act_diff", "steer_kl") if cd[k] is not None)
        lines.append(f"  {i + 1:>3}. L{cd['layer']} N{cd['neuron']}  cos {cd['cos']:+.3f}(押す向き {'+' if cd['sign'] > 0 else '-'}){extra}")
    if d["verify"]:
        lines.append(f"rollout({d['verify']['design']}): " + ", ".join(r["path"] for r in d["verify"]["runs"]))
        for ex in d["verify"]["examples"][:min(show, 10)]:
            lines.append(f"  L{ex['layer']} N{ex['neuron']} ({'+' if ex['sign'] > 0 else '-'}) {ex['prompt']}: "
                         f"{ex['text'][:160]!r}")
    return d, "\n".join(lines), 0


# --- sweep / rank(Phase 3) --------------------------------------------------------------------------------

@gne_command(app, "sweep")
def sweep(c: Common, holder: dict, layer: int = typer.Option(..., "--layer"),
          neurons: str = typer.Option("random:384", "--neurons", help="all / random:384 / 1,5,9-12(カンマで混ぜられる)"),
          design: str = typer.Option("main_rel", "--design", help="configs/designs.yaml の名前(main_rel / quick / calib)"),
          random_controls: int = typer.Option(64, "--random-controls"),
          max_tokens: int = typer.Option(128, "--max-tokens"),
          top_k: int = typer.Option(5, "--top-k", help="各ステップで記録する候補の数"),
          run: Optional[str] = typer.Option(None, "--run", help="run の名前(既定 L<層>-<設計>)"),
          prompt_set: str = typer.Option("en", "--prompt-set", help="configs/prompts/default.yaml の組(en / ja)"),
          dose_unit: str = typer.Option("rel", "--dose-unit"),
          resume: bool = typer.Option(True, "--resume/--no-resume", help="出力の JSONL の続きから")):
    """ニューロン(とランダム対照)を押して文章を生成し、rollout を JSONL に書く(総当たり)。"""
    s = holder["session"] = c.session()
    d = s.sweep(layer, neurons=neurons, design=design, random_controls=random_controls, max_tokens=max_tokens,
                top_k=top_k, run=run, prompt_set=prompt_set, dose_unit=dose_unit, resume=resume)
    text = (f"{d['path']}({d['n_neurons']} ニューロン + 対照 {d['n_controls']}、{d['design']}、"
            f"新しい rollout {d['n_new_records']}、{d['seconds']:.0f} 秒)\n次: gne rank --run {d['run']}")
    return d, text, 0


@gne_command(app, "rank")
def rank(c: Common, holder: dict, run: str = typer.Option(..., "--run", help="run の名前か JSONL のパス"),
         show: int = typer.Option(40, "--show"),
         find: Optional[list[int]] = typer.Option(None, "--find", help="このニューロンの順位を出す(複数可)"),
         word_unit: str = typer.Option("tokens", "--word-unit", help="tokens(トークン id 単位、既定)/ regex"),
         max_rel: float = typer.Option(1.6, "--max-rel", help="語を数える rollout の rel_push の上限")):
    """sweep の rollout から units.jsonl を作り、語彙のかたまり(mass)で順位づけして上位を表示する。"""
    s = holder["session"] = c.session()
    d = s.rank(run, word_unit=word_unit, max_rel=max_rel, show=show, find=find)
    cm = d["control_mass"]
    lines = [f"{d['units_path']}({d['n_neurons']} ニューロン + 対照 {d['n_controls']}、語の単位 {word_unit})",
             f"対照の mass: 中央値 {cm['p50'] or 0:.2f} / 95% {cm['p95'] or 0:.2f} / 最大 {cm['max'] or 0:.2f}、"
             f"全対照より上のニューロン {d['n_above_all_controls']}"]

    def row(u):
        who = f"N{u['neuron']}" if u["kind"] == "neuron" else f"random{u['neuron']}"
        rk = f"#{u['rank']:<4}" if u["rank"] else "     "
        null = f"{u['null_pct']:.0%}" if u["null_pct"] is not None else "-"
        ws = ", ".join(f"{w}({cnt})" for w, cnt, _ in u["words"][:6])
        return f"  {rk} {who:<11} mass {u['mass']:6.2f}  null {null:>4}  {ws}"

    lines += [row(u) for u in d["units"]]
    for u in d["find"]:
        lines.append("--find:" + row(u))
    return d, "\n".join(lines), 0


# --- chat / neuron / mcp(Phase 4) -------------------------------------------------------------------------

@gne_command(app, "chat")
def chat(c: Common, holder: dict, recipe: Optional[Path] = _RECIPE, unit: Optional[list[str]] = _UNIT,
         dose_unit: Optional[str] = _DOSE_UNIT,
         dose: Optional[float] = typer.Option(None, "--dose", help="各ユニットの dose の大きさをそろえる(符号は保つ)"),
         left: Optional[Path] = typer.Option(None, "--left", help="左の列のレシピ(既定は −レシピ)"),
         right: Optional[Path] = typer.Option(None, "--right", help="右の列のレシピ(既定は +レシピ)"),
         center: Optional[Path] = typer.Option(None, "--center", help="中央の列のレシピ(既定は無介入)"),
         prompt: Optional[list[str]] = typer.Option(None, "--prompt", "-p",
                                                    help="ユーザーの発話(複数で多ターン)。無ければ対話モード"),
         system: Optional[str] = typer.Option(None, "--system"),
         max_tokens: int = typer.Option(256, "--max-tokens"),
         stream: bool = typer.Option(True, "--stream/--no-stream", help="生成しながら表示する"),
         when: Optional[str] = _WHEN):
    """−/0/+ の 3 列比較チャット(ユーザーの発話は共有、返答は列ごと)。"""
    from .chat import ChatState, chat_turn, make_columns
    s = holder["session"] = c.session()
    if when:
        from .when import parse_when
        parse_when(when)
    base = s.recipe(units=unit, path=recipe, dose_unit=dose_unit) if (unit or recipe) else None
    load = lambda p: s.recipe(path=p) if p else None  # noqa: E731
    cols = make_columns(base, dose=dose, left=load(left), right=load(right), center=load(center))
    live = stream and not c.json

    def on_column(i, col):
        print(f"\n── [{i + 1}] {col.label} ──", file=sys.stderr if c.json else sys.stdout, flush=True)

    def on_text(_i, piece):
        print(piece, end="", flush=True)

    if prompt:
        state = ChatState(cols, system)
        turns = []
        for user in prompt:
            if live:
                print(f"\n> {user}", flush=True)
            turns.append({"user": user, "replies": chat_turn(s, state, user, max_tokens=max_tokens,
                                                             on_text=on_text if live else None,
                                                             on_column=on_column if live else None,
                                                             when_override=when)})
        d = {"columns": [{"label": col.label, "recipe": col.recipe.to_dict() if col.recipe else None} for col in cols],
             "system": system, "turns": turns}
        if live:
            print()
            return d, "", 0
        lines = []
        for t in d["turns"]:
            lines.append(f"> {t['user']}")
            for col, rep in zip(d["columns"], t["replies"], strict=True):
                lines.append(f"── {col['label']} ──({rep['finish']}, {rep['n_gen']} tok)\n{rep['text']}")
        return d, "\n".join(lines), 0
    if c.json:
        raise GneError("BAD_ARGS", "--json では --prompt で発話を渡してください(対話モードは使えません)")
    state = ChatState(cols, system)
    print("列: " + " | ".join(col.label for col in cols) + "\n/reset で会話をやり直し、/quit か Ctrl-D で終了", file=sys.stderr)
    turns = []
    while True:
        try:
            user = input("\n> ")
        except EOFError:
            break
        if user.strip() in ("/quit", "/exit"):
            break
        if user.strip() == "/reset":
            state.reset()
            continue
        if not user.strip():
            continue
        replies = chat_turn(s, state, user, max_tokens=max_tokens, on_text=on_text if stream else None,
                            on_column=on_column if stream else None, when_override=when)
        if not stream:
            for col, rep in zip(cols, replies, strict=True):
                print(f"── {col.label} ──\n{rep['text']}")
        else:
            print()
        turns.append({"user": user, "replies": replies})
    return {"columns": [{"label": col.label, "recipe": col.recipe.to_dict() if col.recipe else None} for col in cols],
            "system": system, "turns": turns}, "", 0


@gne_command(app, "neuron")
def neuron(c: Common, holder: dict, layer: int = typer.Argument(..., help="層 L"),
           n: int = typer.Argument(..., help="ニューロン番号"),
           rollouts: int = typer.Option(4, "--rollouts", help="調書に入れる rollout の数"),
           compute: bool = typer.Option(False, "--compute", help="無い logit lens / steer lens をこのニューロンだけ計算する")):
    """ニューロン 1 個の調書(持っている情報を全部まとめる)。"""
    s = holder["session"] = c.session()
    d = s.neuron(layer, n, rollouts=rollouts, compute=compute)
    lines = [f"L{layer} N{n}" + (f"「{d['name']}」" if d["name"] else "") + f"  ‖W_down‖={d['down_norm']:.3f}",
             "coverage: " + ", ".join(f"{k} {'○' if v else '×'}" for k, v in d["coverage"].items())]
    sw = d["sweep"]
    if sw:
        lines.append(f"sweep({sw['run']}、{sw['design']}): #{sw['rank']}/{sw['n_ranked']}  mass {sw['mass']:.2f}  "
                     f"null {sw['null_pct']:.0%}  語 " + ", ".join(f"{w}({cnt})" for w, cnt, _ in sw["words"][:8]))
        for r in sw["rollouts"]:
            lines.append(f"  [{r['prompt']} {r['dose']:+}] {r['text'][:150]!r}")
    if d["steer_lens"]:
        st = d["steer_lens"]
        lines.append(f"steer lens: KL+ {st['kl_up']:.3f}({st['pct_up']:.0%}) KL- {st['kl_down']:.3f}({st['pct_down']:.0%})"
                     f"  対照 p95 {st['kl_null_p95'] or 0:.3f}\n  + {_toks(st['up'])}\n  - {_toks(st['down'])}")
    if d["logit_lens"]:
        lg = d["logit_lens"]
        lines.append(f"logit lens(peak z {lg['peak_z']:.1f}):\n  ↑ {_toks(lg['up'])}\n  ↓ {_toks(lg['down'])}")
    for name, v in d["directions"].items():
        lines.append(f"逆引き {name}: cos {v['cos']:+.3f}(層内 {v['rank_in_layer']} 位)")
    if d["activation"]:
        a = d["activation"]
        lines.append(f"活性: absmax {a['absmax']:.3g}  q999 {a['q999']:.3g}  rms {a['rms']:.3g}")
    return d, "\n".join(lines), 0


@app.command("mcp")
def mcp_cmd(ctx: typer.Context,
            model: Optional[Path] = typer.Option(None, "--model", "-m", envvar="GNE_MODEL", help="GGUF ファイル"),
            ui_url: Optional[str] = typer.Option(None, "--ui-url", help="gne serve の URL(ui_* ツール用)"),
            root: Optional[Path] = typer.Option(None, "--root", envvar="GNE_ROOT"),
            n_gpu_layers: Optional[int] = typer.Option(None, "--n-gpu-layers",
                                                       help="GPU に載せる層数(-1 = 全部)。gne serve と同時なら 0 で GPU を UI に譲れる")):
    """MCP サーバ(stdio)。Claude Code から `claude mcp add gne -- gne mcp --model <path>` で接続する。"""
    from .mcp_server import main as mcp_main
    c: Common = ctx.obj or Common()
    path = model or c.model
    if path is None:
        print("エラー[BAD_ARGS]: --model を指定してください", file=sys.stderr)
        raise typer.Exit(2)
    mcp_main(str(path), root=str(root or c.root) if (root or c.root) else None, ui_url=ui_url,
             n_gpu_layers=c.n_gpu_layers if n_gpu_layers is None else n_gpu_layers, n_ctx=c.n_ctx, seed=c.seed)


@app.command("serve")
def serve_cmd(ctx: typer.Context,
              model: Optional[Path] = typer.Option(None, "--model", "-m", envvar="GNE_MODEL", help="GGUF ファイル"),
              port: int = typer.Option(8791, "--port"),
              host: str = typer.Option("127.0.0.1", "--host", help="ローカル専用(127.0.0.1)"),
              root: Optional[Path] = typer.Option(None, "--root", envvar="GNE_ROOT"),
              openai: bool = typer.Option(False, "--openai", help="OpenAI 互換の /v1/chat/completions も出す(レシピと when は gne で指定、§6.15)")):
    """Web UI(http://127.0.0.1:8791)。"""
    from .ui.server import serve
    c: Common = ctx.obj or Common()
    path = model or c.model
    if path is None:
        print("エラー[BAD_ARGS]: --model を指定してください", file=sys.stderr)
        raise typer.Exit(2)
    try:
        print(f"gne serve: http://{host}:{port}/" + (f"  (OpenAI 互換: http://{host}:{port}/v1)" if openai else ""), file=sys.stderr)
        serve(path, host=host, port=port, root=root or c.root, n_gpu_layers=c.n_gpu_layers, n_ctx=c.n_ctx, seed=c.seed,
              openai=openai)
    except GneError as e:
        print(f"エラー[{e.code}]: {e.message}", file=sys.stderr)
        raise typer.Exit(2) from e


if __name__ == "__main__":
    app()
