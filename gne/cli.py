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
heads_app = typer.Typer(no_args_is_help=True, help="attention ヘッド(対照分析・注目先・調書、CHANGE-04)")
app.add_typer(heads_app, name="heads")


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
        who = {"neuron": f"N{u['neuron']}", "random": f"random{u['seed']}"}.get(u["kind"], f"dir:{u.get('direction')}")
        if u.get("mode") == "gain":
            rows.append(f"  L{u['layer']} {who} ×{u['k']:g}(gain: 出力を {u['k']:g} 倍、LoRA で当てる)")
            continue
        rp = f"rel_push {u['rel_push']:.2f}、ρ={u['rho']:.2f}" if u.get("rel_push") is not None else "ρ 不明"
        add = f"、活性に {u['add']:+.3g} 相当" if u.get("add") is not None else ""
        if u.get("diff_rel") is not None:
            add += f"、平均差の {u['norm'] / (u['diff_rel'] * u['rho']):.2f} 倍"
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
    rw = d.get("residual_writers")
    if rw:
        rows.append(("残差ライター", ", ".join(f"{k} {n}" for k, n in rw["counts"].items())
                     + ("(グラフで確認済み)" if rw["verified"] else "(テンソル名で推定)")))
    lc = d["llama_cpp"]
    if lc is not None:
        rows.append(("llama.cpp", f"読み込み可(n_layer {lc['n_layer']}, n_embd {lc['n_embd']}, n_vocab {lc['n_vocab']})"
                     if lc["ok"] else f"読み込めません: {lc['error']}(新しい llama.cpp でのビルドが必要かもしれません、R6)"))
    sp = d.get("speed")
    if sp:
        rows.append(("GPU / CPU の層", f"GPU {sp['gpu_layers']} / CPU {sp['cpu_layers']}、推定の生成 {sp['decode_tok_s']:.0f} tok/s(1 本)"
                     + ("、gne bench の実測から" if sp["source"] == "bench" else "、既定の係数から")))
    text = _table(rows)
    text += "".join(f"\n警告: {w}" for w in d["arch_warnings"]) or "\n警告なし: dense FFN、ニューロン押しは control vector と等価です"
    if d.get("workflow"):
        from .estimate import fmt_seconds
        text += "\n大型モデルの手順(全数の lens・sweep は重いので、逆引きから絞る):" + "".join(
            f"\n  {i}. {w['step']}(約 {fmt_seconds(w['seconds'])})" for i, w in enumerate(d["workflow"], 1))
    if rw and rw["warnings"]:
        text += "".join(f"\n射影消去: {w}" for w in rw["warnings"])
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
             ("V6", v.v6_attention_sink), ("V11", v.v11_gain), ("V12", v.v12_when_gen)]
    small = s.meta.file_size < 2 * 2**30                         # bake を 2 つ作るので、大きいモデルでは LoRA だけ
    steps += [("V13", lambda: v.v13_ablation(bake=small)),
              ("V14", v.v14_lora_vs_bake if small else
               lambda: VResult("V14", "射影消去の LoRA と bake の一致", "skip",
                               "2 GiB 以上のモデルは省きます(gne ablate --bake --verify で確かめられます)"))]
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
    """set_cvec / prefill / decode / 生成の実測(docs/measurements.md 向け)。結果は <root>/bench/<model_id>.json にも保存し、
    ジョブの時間の見積もりに使う。"""
    from .bench import bench_markdown, run_bench
    s = holder["session"] = c.session()
    r = run_bench(s.engine, layer=layer, reps=reps, gen_tokens=gen_tokens, seed=c.seed)
    bp = s.store.root / "bench" / f"{s.meta.model_id}.json"           # 見積もり(CHANGE-03 C)が実測を使う
    bp.parent.mkdir(parents=True, exist_ok=True)
    bp.write_text(json.dumps(r, default=_json_default, ensure_ascii=False, indent=1))
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
_UNIT = typer.Option(None, "--unit", help="L:N:dose(例 14:8949:+1.0)、L:random:SEED:dose、gain は L:N:x2、方向は L:dir:NAME:dose、ヘッドは L:Hh:xk(例 21:H7:x0)。複数可")
_WORKERS = typer.Option(None, "--workers", help="auto か名前のカンマ区切り: gne worker に分けて投げ、終わるまで待つ(CHANGE-03 B)")
_YES = typer.Option(False, "--yes", help="見積もりが 30 分を超えても走らせる")
_BUDGET = typer.Option(None, "--budget", help="時間の予算(30m・2h・90s)。収まるように候補や窓を絞る")
_DOSE_UNIT = typer.Option(None, "--dose-unit", help="rel / absmax / raw(既定はレシピの値、--unit なら rel)")
_SCALE = typer.Option(1.0, "--scale", help="全ユニットの dose に掛ける係数")
_ABLATE = typer.Option(None, "--ablate", help="方向の射影消去(§6.17): NAME、NAME@層、NAME:倍率(例 evil@10:0.5)。LoRA で当てる")
_WHEN = typer.Option(None, "--when", help="足す位置(§6.15): all / gen / assistant / after:<文字列> / ramp:<a>-<b>。既定はレシピの値(無ければ all)")
_TEMP = typer.Option(None, "--temperature", help="サンプリングの温度。サンプリングの値をどれか付けるか、レシピに sampling があればサンプリング"
                                                 "(足りない値はレシピの sampling、無ければ llama-server の既定)。0 だけなら greedy")
_TOP_K = typer.Option(None, "--top-k", help="サンプリングの top-k(0 で切らない)")
_TOP_P = typer.Option(None, "--top-p", help="サンプリングの top-p")
_MIN_P = typer.Option(None, "--min-p", help="サンプリングの min-p")
_PRES = typer.Option(None, "--presence-penalty", help="サンプリングの presence penalty(この返事で出たトークンのロジットから引く)")
_SSEED = typer.Option(None, "--sample-seed", help="サンプリングの乱数(既定は毎回ランダム)")
_GREEDY = typer.Option(False, "--greedy", help="レシピに sampling があっても greedy で生成する")


def _sampling(recipe, greedy, temperature, top_k, top_p, min_p, presence_penalty, sample_seed):
    """CLI の値とレシピの sampling → engine.Sampling(greedy なら None)。"""
    import random

    from .engine import Sampling
    from .steer import sampling_values
    given = {k: v for k, v in (("temperature", temperature), ("top_k", top_k), ("top_p", top_p), ("min_p", min_p),
                               ("presence_penalty", presence_penalty)) if v is not None}
    if greedy:
        if given or sample_seed is not None:
            raise GneError("BAD_ARGS", "--greedy とサンプリングの値は一緒に使えません")
        return None
    base = recipe.sampling if recipe is not None else None
    if not given and sample_seed is None and base is None:
        return None
    if set(given) == {"temperature"} and given["temperature"] <= 0:
        return None
    vals = sampling_values(given, base=base, seed=False)
    return Sampling(**vals, seed=sample_seed if sample_seed is not None else random.randrange(2 ** 31))


def _sampling_text(sp) -> str:
    if sp is None:
        return "greedy"
    return (f"サンプリング(温度 {sp.temperature:g}・top-k {sp.top_k}・top-p {sp.top_p:g}・min-p {sp.min_p:g}・"
            f"presence {sp.presence_penalty:g}・seed {sp.seed})")


@gne_command(app, "steer")
def steer(c: Common, holder: dict, recipe: Optional[Path] = _RECIPE, unit: Optional[list[str]] = _UNIT,
          dose_unit: Optional[str] = _DOSE_UNIT, scale: float = _SCALE,
          prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="ユーザーの発話"),
          system: Optional[str] = typer.Option(None, "--system", help="system プロンプト"),
          max_tokens: int = typer.Option(128, "--max-tokens"),
          compare: bool = typer.Option(False, "--compare", help="無介入の生成も並べる(サンプリングなら同じ seed)"),
          when: Optional[str] = _WHEN, ablate: Optional[str] = _ABLATE,
          temperature: Optional[float] = _TEMP, top_k: Optional[int] = _TOP_K, top_p: Optional[float] = _TOP_P,
          min_p: Optional[float] = _MIN_P, presence_penalty: Optional[float] = _PRES,
          sample_seed: Optional[int] = _SSEED, greedy: bool = _GREEDY,
          contrast: Optional[Path] = typer.Option(None, "--contrast", help="対照集合 YAML: その validate で効果を測り、ランダムな対照レシピと比べる(null_pct)"),
          random_controls: int = typer.Option(16, "--random-controls", help="--contrast: 対照レシピ(同じ大きさのランダムな向き)の本数"),
          contrast_sample: bool = typer.Option(False, "--contrast-sample", help="--contrast: greedy ではなくレシピのサンプリング(--temperature などで上書き)で生成する。乱数は無介入・レシピ・対照で同じ"),
          contrast_seeds: int = typer.Option(4, "--contrast-seeds", help="--contrast-sample: 1 プロンプトあたりの本数"),
          contrast_tokens: int = typer.Option(64, "--contrast-tokens", help="--contrast: 生成する長さ"),
          contrast_prompts: Optional[Path] = typer.Option(None, "--contrast-prompts", help="--contrast: 検証プロンプトを差し替える YAML(文字列のリスト、または対照集合なら reactivity の with か user_prompts)")):
    """レシピまたは単発指定でプロンプトに回答する(既定は greedy、1 本ずつ)。"""
    s = holder["session"] = c.session()
    r = s.recipe(units=unit, path=recipe, dose_unit=dose_unit, scale=scale, when=when, ablate=ablate)
    sp = _sampling(r, greedy, temperature, top_k, top_p, min_p, presence_penalty, sample_seed)
    if prompt is None and contrast is None:
        raise GneError("BAD_ARGS", "--prompt か --contrast が要ります")
    if prompt is not None:
        d = s.steer(r, prompt=prompt, system=system, max_tokens=max_tokens, compare=compare, sampling=sp)
    else:                                       # --contrast だけ: 生成はせず、対照集合での効果だけを測る
        vecs, details = s.vectors(r)
        d = {"messages": [], "recipe": r.to_dict(), "units": details, "layer_norms": s._norms(vecs), "when": r.when,
             "sampling": None, "baseline": None, "steered": None, "differs": None}
    if contrast is not None:
        csp = None
        if contrast_sample:                       # レシピを詰めたときの条件に合わせる。seed は固定(既定 0)で、同じ乱数で比べる
            from .engine import Sampling
            from .steer import SAMPLING_PRESETS, sampling_values
            given = {k: v for k, v in (("temperature", temperature), ("top_k", top_k), ("top_p", top_p), ("min_p", min_p),
                                       ("presence_penalty", presence_penalty)) if v is not None}
            vals = sampling_values(given, base=r.sampling or SAMPLING_PRESETS["common"], seed=False)
            csp = Sampling(**vals, seed=sample_seed if sample_seed is not None else 0)
        d["behavior"] = s.behavior_null(r, contrast, random_controls=random_controls, max_tokens=contrast_tokens,
                                        sampling=csp, seeds=contrast_seeds, prompts=contrast_prompts)
    parts = [f"ユニット({r.dose_unit}、位置 {r.when}、{_sampling_text(sp)}):\n" + (_units_text(d["units"]) or "  なし")
             + (f"\n  {r.ablate.label()}(LoRA)" if r.ablate else "")]
    if d["baseline"] is not None:
        parts.append(f"── 無介入 ──({d['baseline']['finish']}, {d['baseline']['n_gen']} tok)\n{d['baseline']['text']}")
    label = ", ".join([u.label() for u in r.units] + ([r.ablate.label()] if r.ablate else []))
    if d["steered"] is not None:
        parts.append(f"── 介入: {label} ──({d['steered']['finish']}, {d['steered']['n_gen']} tok)\n{d['steered']['text']}")
    if d["differs"] is not None:
        parts.append("無介入との差: " + ("あり" if d["differs"] else "なし(同じ列)"))
    b = d.get("behavior")
    if b is not None:
        cs = b["ctrl"]
        how = (f"サンプリング(温度 {b['sampling']['temperature']:g}・seed {b['sampling']['seed']})× {b['seeds']} 本"
               if b.get("sampling") else "greedy")
        line = (f"対照集合での効果({b['mode']}、{b['n_prompts']} プロンプト × {how} = {b.get('n_samples', b['n_prompts'])} 本、"
                f"{b.get('max_tokens', 64)} トークン、位置 {b['when']}): 一致率 {b['base_rate']:.2f} → "
                f"{b['rate']:.2f}(効果 {b['effect']:+.2f})\n  例: {b['sample'][:160]!r}")
        if b["random_controls"]:
            line += (f"\n  ランダムな対照 {b['random_controls']} 本: 効果 p50 {cs['p50']:+.2f}・p95 {cs['p95']:+.2f}・"
                     f"最大 {cs['max']:+.2f} → null_pct {b['null_pct']:.2f}"
                     + ("" if b["null_pct"] is not None and b["null_pct"] >= 0.95 else "(0.95 未満: 対照と区別できない)"))
        else:
            line += "\n  対照なし(add ユニットも ablate も無いレシピ)"
        parts.append(line)
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
    if d.get("llama_server_args"):
        text += f"\nレシピのサンプリング(cvec には入らないので llama-server 側で付ける): {d['llama_server_args']}"
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


@gne_command(app, "gain")
def gain(c: Common, holder: dict, recipe: Optional[Path] = _RECIPE, unit: Optional[list[str]] = _UNIT,
         scale: float = _SCALE,
         out: Optional[Path] = typer.Option(None, "--out", "-o", help="出力(既定 <root>/export/<レシピ名>.lora.gguf、--bake なら -gain.gguf)"),
         bake: bool = typer.Option(False, "--bake", help="LoRA ではなく、W_up を書き換えた GGUF を作る"),
         in_path: Optional[Path] = typer.Option(None, "--in", help="--bake の元の GGUF(既定 --model)"),
         requant: str = typer.Option("q8_0", "--requant", help="--bake で書き換えた ffn_up / W_O の型: q8_0 / f16")):
    """gain ユニット(L:N:x2 など、出力を k 倍。ヘッドは L:Hh:xk)を LoRA アダプタ GGUF か書き換え GGUF に書き出す(§6.16)。"""
    if in_path and c.model and in_path.resolve() != c.model.resolve():
        raise GneError("BAD_ARGS", "--in と --model が違うファイルです")
    s = holder["session"] = c.session(in_path)
    r = s.recipe(units=unit, path=recipe, scale=scale)
    name = r.name or "gain"
    out = out or s.store.export_dir() / (f"{name}-gain.gguf" if bake else f"{name}.lora.gguf")
    d = s.gain(r, out, bake=bake, requant=requant)
    if bake:
        text = f"{d['path']}({d['bytes'] / 1e6:.1f} MB)、書き換え: " + ", ".join(
            f"{m['tensor']}({m['from']} → {m['to']})" for m in d["modified"])
    else:
        text = (f"{d['path']}({d['bytes'] / 1e6:.2f} MB、LoRA)\n"
                + "\n".join(f"  L{L}: " + ", ".join(f"N{int(n)} ×{k:g}" for n, k in nk) for L, nk in (d.get("layers") or {}).items())
                + "".join(f"\n  L{L}: " + ", ".join(f"H{int(h)} ×{k:g}" for h, k in hk) for L, hk in (d.get("heads") or {}).items())
                + "".join(f"\n  クエリ操作: {p}" for p in (d.get("query_loras") or []))
                + "\n使い方: llama-cli -m <model> --lora " + d["path"])
    return d, text, 0


def _doses(text: str) -> list[float]:
    try:
        return [float(x) for x in text.split(",") if x.strip()]
    except ValueError as e:
        raise GneError("BAD_ARGS", f"--doses {text!r} はカンマ区切りの数です(例 0,0.5,1)") from e


@gne_command(app, "eval")
def eval_cmd(c: Common, holder: dict, recipe: Optional[Path] = _RECIPE, unit: Optional[list[str]] = _UNIT,
             dose_unit: Optional[str] = _DOSE_UNIT,
             doses: str = typer.Option("0,0.25,0.5,0.75,1", "--doses", help="レシピ全体に掛ける倍率(0 = 無介入は必ず入る)"),
             suite: Optional[Path] = typer.Option(None, "--suite", help="タスク集 YAML(既定 configs/eval/default.yaml)"),
             when: Optional[str] = _WHEN,
             judge_model: Optional[Path] = typer.Option(None, "--judge-model", help="性格の採点に使う別の GGUF(任意)"),
             judge_target: Optional[str] = typer.Option(None, "--judge-target", help="採点させる目標の性格(例 'a yandere anime girl')"),
             name: Optional[str] = typer.Option(None, "--name", help="保存名(既定はレシピ名)"),
             ablate: Optional[str] = _ABLATE,
             compare_ablation: Optional[str] = typer.Option(None, "--compare-ablation",
                                                            help="方向 NAME の「上位ニューロンの消去(gain ×0)」と「方向の消去」を並べる(§6.17)"),
             top_neurons: int = typer.Option(20, "--top-neurons", help="--compare-ablation で消すニューロンの数"),
             from_layer: Optional[int] = typer.Option(None, "--from-layer", help="--compare-ablation: どの層で測った r を消すか"),
             random_controls: int = typer.Option(0, "--random-controls", help="各倍率で同じ大きさのランダムな対照レシピ N 本も測る(0 = なし)"),
             reactivity: Optional[Path] = typer.Option(None, "--reactivity", help="反応性も測る対照集合 YAML(reactivity の対と validate.match)"),
             head_controls: int = typer.Option(0, "--head-controls", help="--reactivity: ヘッドを同じ k のランダムなヘッドに替えた対照の本数"),
             yes: bool = _YES):
    """副作用チェック: 倍率ごとに能力(exact / format)・ppl・言語の混入・性格の変化量を測り、配布可の倍率を示す(§6.14)。"""
    s = holder["session"] = c.session()
    if compare_ablation:
        if unit or recipe or ablate:
            raise GneError("BAD_ARGS", "--compare-ablation はレシピ(--recipe / --unit / --ablate)と一緒には使えません")
        d = s.eval_compare_ablation(compare_ablation, top=top_neurons, from_layer=from_layer, doses=_doses(doses),
                                    suite=suite, when=when)
        lines = [f"消去の比較 {d['direction']}(r は L{d['from_layer']}、上位 {d['top']} ニューロン vs 方向、{d['seconds']:.0f} 秒)→ {d['path']}",
                 f"振る舞いの判定: {d['match'] or '—'}(無介入の一致率 {format(d['base_rate'], '.2f') if d['base_rate'] is not None else '—'})",
                 f"  {'方法':<9} {'倍率':>5} {'一致率':>6} {'減少':>6} {'KL(B)':>7} {'能力低下':>8} {'ppl悪化':>8} {'混入':>6}"]
        for r_ in d["rows"]:
            f2 = lambda x, fmt: "—" if x is None else format(x, fmt)  # noqa: E731
            lines.append(f"  {('ニューロン' if r_['method'] == 'neurons' else '方向'):<9} {r_['dose']:>5g} {f2(r_['rate'], '.2f'):>6} "
                         f"{f2(r_['rate_drop'], '+.2f'):>6} {f2(r_['kl_b'], '.3f'):>7} {r_['capability_drop_pt']:>7.1f}pt "
                         f"{r_['ppl_worse_pct']:>+7.1f}% {r_['mix_increase'] * 100:>+5.2f}%" + ("  配布可" if r_["shippable"] else ""))
        lines.append("消したニューロン: " + ", ".join(f"L{x['layer']} N{x['neuron']}({x['cos']:+.2f})" for x in d["neurons"][:10])
                     + (" …" if len(d["neurons"]) > 10 else ""))
        return d, "\n".join(lines), 0
    r = s.recipe(units=unit, path=recipe, dose_unit=dose_unit, ablate=ablate)
    _gate(s, "eval", {"doses": _doses(doses), "suite": suite, "when": when, "random_controls": random_controls}, yes, None)
    d = s.eval(r, doses=_doses(doses), suite=suite, when=when, judge_model=judge_model, judge_target=judge_target,
               name=name, random_controls=random_controls, reactivity=reactivity, head_controls=head_controls)
    th = d["suite"]["thresholds"]
    nc = d.get("random_controls") or 0
    head = f"{'倍率':>5} {'能力':>6} {'低下pt':>6} {'ppl':>8} {'悪化%':>7} {'混入':>6} {'新語':>5} {'d4':>5} {'打切':>4}"
    if d["judge"]:
        head += f" {'judge':>5}"
    if nc:
        head += f" │ {'対照新語p95':>10} {'null':>5} {'対照低下p50':>10} {'対照悪化p50':>10}"
    rcd = d.get("reactivity")
    if rcd:
        head += f" │ {'あり':>5} {'なし':>5} {'反応性':>6}" + (f" {'対照p95':>7} {'null':>5}" if rcd["head_controls"] else "")
    lines = [head]
    for row in d["rows"]:
        pm = row["persona"]
        line = (f"{row['dose']:>5g} {row['capability'] * 100:>5.1f}% {row['capability_drop_pt']:>6.1f} "
                f"{row['ppl_mean']:>8.2f} {row['ppl_worse_pct']:>+6.1f}% {(row['mix_zh'] + row['mix_other']) * 100:>5.2f}% "
                f"{pm['novelty']:>5.2f} {pm['distinct4']:>5.2f} {pm['capped'] * 100:>3.0f}%")
        if d["judge"]:
            line += f" {row['judge'] if row['judge'] is not None else float('nan'):>5.2f}"
        if nc:
            if row.get("ctrl"):
                f2 = lambda x, fmt: "—" if x is None else format(x, fmt)  # noqa: E731
                line += (f" │ {f2(row['ctrl_novelty_p95'], '.2f'):>10} {f2(row['novelty_null_pct'], '.2f'):>5} "
                         f"{f2(row['ctrl_capability_drop_p50'], '.1f'):>10} {f2(row['ctrl_ppl_worse_p50'], '+.1f'):>9}%")
            else:
                line += f" │ {'':>10} {'':>5} {'':>10} {'':>10}"
        if rcd:
            x = row.get("reactivity") or {}
            line += f" │ {x.get('with', 0):>5.2f} {x.get('without', 0):>5.2f} {x.get('reactivity', 0):>+6.2f}"
            if rcd["head_controls"]:
                cx = x.get("ctrl")
                line += (f" {cx['p95']:>+7.2f} {x['null_pct'] * 100:>4.0f}%" if cx and cx.get("p95") is not None
                         else f" {'':>7} {'':>5}")
        lines.append(line + ("  配布可" if row["shippable"] else "  " + "・".join(k for k, ok in row["ok"].items() if not ok) + " ×"))
    ok = [r_["dose"] for r_ in d["rows"] if r_["shippable"] and r_["dose"] > 0]
    verdict = (f"配布可の最大倍率: {max(ok):g}" if ok else "配布可の倍率はありません(0 を除く)")
    text = (f"eval {d['name']}(when={d['when']}、{d['suite']['n_tasks']} タスク、{d['seconds']:.0f} 秒)→ {d['path']}\n"
            + "\n".join(lines) + f"\n基準: 能力 −{th['capability_drop_pt']:g}pt 以内・ppl +{th['ppl_worse_pct']:g}% 以内・混入が増えない。"
            + verdict)
    if nc:
        text += (f"\n対照: 同じ大きさのランダムな向きのレシピ {nc} 本(倍率ごと)。新語が対照 p95 を超え(null ≥ 0.95)、"
                 "能力・ppl の悪化が対照と同程度なら、悪化は向きではなく大きさのせい")
    elif random_controls:
        text += "\n対照は作れませんでした(gain だけのレシピには向きが無い)"
    if rcd:
        text += (f"\n反応性({rcd['name']}、{rcd['n_pairs']} 対、{rcd['max_tokens']} トークン): トリガーあり/なしの判定の一致率と、その差。"
                 + (f"対照は同じ k のランダムなヘッド {rcd['head_controls']} 本" if rcd["head_controls"] else "")
                 + ("" if rcd["head_controls"] or not head_controls else "ヘッドの無いレシピなのでヘッドの対照は作れません"))
    top = d["rows"][-1]["persona"]["top_words"][:8]
    if top:
        text += f"\n倍率 {d['rows'][-1]['dose']:g} で複数のプロンプトに新しく出た語: " + ", ".join(f"{w}({n})" for w, n in top)
    return d, text, 0


@gne_command(app, "ablate")
def ablate(c: Common, holder: dict,
           direction: str = typer.Option(..., "--direction", help="gne direction で作った方向の名前(逆引きの方向も可)"),
           from_layer: Optional[int] = typer.Option(None, "--from-layer", help="どの層で測った r を使うか(既定は --select で選んだ層)"),
           per_layer: bool = typer.Option(False, "--per-layer", help="層ごとに、その層で測った r_L を消す"),
           layers: str = typer.Option("all", "--layers", help="消す層: all / 10-40 / 1,5,9-12"),
           scale: float = typer.Option(1.0, "--scale", help="消す割合(1 で全部、0.5 で半分)"),
           out: Optional[Path] = typer.Option(None, "--out", "-o", help="出力(既定 <root>/export/<方向>-ablate.lora.gguf、--bake なら -ablate.gguf)"),
           bake: bool = typer.Option(False, "--bake", help="LoRA ではなく、重みを書き換えた GGUF を作る"),
           in_path: Optional[Path] = typer.Option(None, "--in", help="--bake の元の GGUF(既定 --model)"),
           requant: str = typer.Option("q8_0", "--requant", help="--bake で書き換えたテンソルの型: q8_0 / f16"),
           via_f16: bool = typer.Option(False, "--via-f16", help="--bake: F16 で書いて llama-quantize で元の型に戻す"),
           imatrix: Optional[Path] = typer.Option(None, "--imatrix", help="--via-f16: IQ1〜IQ2 系のモデルに要る imatrix"),
           llama_quantize: Optional[str] = typer.Option(None, "--llama-quantize", help="--via-f16 の llama-quantize のパス"),
           skip_embed: bool = typer.Option(False, "--skip-embed", help="埋め込み(token_embd)は射影しない"),
           verify: bool = typer.Option(False, "--verify", help="書き出したもので V13(r 成分)と、bake なら V14(LoRA と greedy 一致)を見る")):
    """方向 r の射影消去(abliteration)を LoRA アダプタ GGUF か、重みを書き換えた GGUF に書き出す(§6.17)。"""
    if in_path and c.model and in_path.resolve() != c.model.resolve():
        raise GneError("BAD_ARGS", "--in と --model が違うファイルです")
    if via_f16 and not bake:
        raise GneError("BAD_ARGS", "--via-f16 は --bake と一緒に使います")
    s = holder["session"] = c.session(in_path)
    out = out or s.store.export_dir() / (f"{direction}-ablate.gguf" if bake else f"{direction}-ablate.lora.gguf")
    d = s.ablate(direction, out, from_layer=from_layer, per_layer=per_layer, layers=parse_layers(layers), scale=scale,
                 embed=not skip_embed, bake=bake, requant=requant, via_f16=via_f16,
                 imatrix=str(imatrix) if imatrix else None, llama_quantize=llama_quantize, verify=verify)
    src = "層ごとの r_L" if d.get("per_layer") else f"L{d['from_layer']} で測った r"
    Ls = d["layers"]
    head = (f"{d['path']}({d['bytes'] / 2**20:.1f} MiB、{'LoRA' if d['kind'] == 'lora' else 'bake ' + d['requant']}、"
            f"{src}、層 {Ls[0]}〜{Ls[-1]}({len(Ls)} 層)、scale {d['scale']:g}、埋め込み {'あり' if d['embed'] else 'なし'}、"
            f"{d['seconds']:.0f} 秒)")
    lines = [head]
    if d["kind"] == "lora":
        lines.append(f"使い方: llama-cli -m <model> --lora {d['path']}(--lora-scaled {d['path']} 0.5 で半分だけ消す)")
    ok = True
    v = d["verify"]
    if v and "v13" in v:
        t = v["v13"]
        lines.append(f"V13: |r·l_out − 期待値| / ‖l_out‖ 最大 {t['max_dev']:.2e}"
                     + (f"({'合格' if t['pass'] else '不合格'})" if t["pass"] is not None else f"({t['note']})"))
        ok &= t["pass"] is not False
    if v and "v14" in v:
        t = v["v14"]
        lines.append(f"V14: LoRA と bake の greedy {t['n_tokens']} トークン: " + "、".join(t["summary"])
                     + f"({'合格' if t['pass'] else '不合格'})" + (f"({t['note']})" if t["note"] else ""))
        ok &= t["pass"]
    return d, "\n".join(lines), 0 if ok else 1


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
               resume: bool = typer.Option(True, "--resume/--no-resume", help="保存済みの結果の続きから"),
               direction: Optional[str] = typer.Option(None, "--direction", help="--budget: 書き込みの向きがこの方向に近い順に選ぶ(既定は最後に作った方向)"),
               yes: bool = _YES, budget: Optional[str] = _BUDGET, workers: Optional[str] = _WORKERS):
    """層の全(または指定)ニューロンの steer lens(±dose で押したときの次トークンの変化、ランダム対照つき)。"""
    s = holder["session"] = c.session()
    g = _gate(s, "lens_steer", {"layer": layer, "neurons": neurons, "random_controls": random_controls,
                                **({"direction": direction} if direction else {})}, yes, budget)
    if workers:
        d = _via_workers(s, "lens_steer", {"layer": layer, "neurons": g["neurons"], "dose": dose,
                                           "random_controls": random_controls, "top_k": top_k, "resume": resume,
                                           "show": show}, workers)
        text = d.pop("text")
        return d, text, 0
    d = s.lens_steer(layer, neurons=g["neurons"], dose=dose, random_controls=random_controls, top_k=top_k,
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
            show: int = typer.Option(20, "--show"),
            yes: bool = _YES, budget: Optional[str] = _BUDGET):
    """対照方向を作り、全層・全ニューロンを cos で順位づけする(逆引き)。"""
    if method not in ("mean_diff", "pca1") or positions not in ("all", "last", "assistant"):
        raise GneError("BAD_ARGS", "--method は mean_diff / pca1、--positions は all / last / assistant です")
    s = holder["session"] = c.session()
    vb = None
    if verify:
        g = _gate(s, "reverse", {"verify": True, "top": top, "verify_top": verify_top, "verify_controls": verify_controls,
                                 "max_tokens": max_tokens}, yes, budget)
        vb = g.get("verify_budget_s")
    d = s.reverse(contrast=contrast, directions=directions, method=method, positions=positions, top=top,
                  layers=parse_layers(layers), verify=verify, verify_top=verify_top, verify_controls=verify_controls,
                  max_tokens=max_tokens, name=name, standardize=standardize, verify_budget_s=vb)
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


@gne_command(app, "direction")
def direction(c: Common, holder: dict,
              contrast: Path = typer.Option(..., "--contrast", help="対照集合 YAML(reverse と同じ形。層選びには validate.match)"),
              name: Optional[str] = typer.Option(None, "--name", help="保存名(既定は対照集合の name)"),
              positions: str = typer.Option("last", "--positions", help="last(生成プロンプトの最終トークン)/ all / assistant / response(A・B の system で返事を生成し、返事のトークンで測る)"),
              response_tokens: int = typer.Option(96, "--response-tokens", help="--positions response で生成する返事の長さ"),
              select: Optional[str] = typer.Option(None, "--select", help="auto(候補層ごとに効果を測って選ぶ)か層の番号"),
              layer: Optional[int] = typer.Option(None, "--layer", help="層を手で指定する(--select <番号> と同じ)"),
              mode: str = typer.Option("ablate", "--mode", help="--select auto の測り方: ablate(消して効果が下がるか)/ add(足して効果が上がるか)"),
              layers: Optional[str] = typer.Option(None, "--layers", help="候補の層(既定 0.3〜0.8·n_layer。10-24、10-24:2、12,16)"),
              dose: float = typer.Option(1.0, "--dose", help="mode=add で足す大きさ(平均差 mean_A − mean_B の何倍か)"),
              max_tokens: int = typer.Option(64, "--max-tokens"),
              kl_max: float = typer.Option(0.1, "--kl-max", help="mode=ablate: B 側の次トークン KL の上限"),
              min_distinct: float = typer.Option(0.8, "--min-distinct", help="mode=add: distinct-4 の下限(反復していないか)"),
              per_layer: bool = typer.Option(False, "--per-layer", help="mode=ablate: 層ごとの r_L を各層に当てる変種も測る"),
              judge_model: Optional[Path] = typer.Option(None, "--judge-model", help="判定に使う別の GGUF(validate.match の代わり)"),
              judge_target: Optional[str] = typer.Option(None, "--judge-target", help="judge に渡す目標の性格(既定 validate.target か system_a)"),
              random_controls: int = typer.Option(16, "--random-controls", help="--select auto: 層ごとの同じ大きさのランダムな向きの対照の本数(0 で対照なし)"),
              yes: bool = _YES):
    """対照集合から全層の方向 r_L(mean_A − mean_B)を作って保存し、--select auto で効果の大きい層を選ぶ(§6.17)。
    作った方向は L:dir:NAME:dose のユニットで足せる(gne steer / chat / eval / export-cvec)。"""
    if mode not in ("ablate", "add"):
        raise GneError("BAD_ARGS", "--mode は ablate か add です")
    if layer is not None:
        if select not in (None, str(layer)):
            raise GneError("BAD_ARGS", "--layer と --select はどちらか一方にしてください")
        select = str(layer)
    s = holder["session"] = c.session()
    _gate(s, "direction", {"contrast": str(contrast), "positions": positions, "select": select, "mode": mode,
                           "layers": layers, "max_tokens": max_tokens, "response_tokens": response_tokens,
                           "random_controls": random_controls}, yes, None)
    d = s.direction(contrast=contrast, name=name, positions=positions, select=select, mode=mode, layers=layers,
                    dose=dose, max_tokens=max_tokens, kl_max=kl_max, min_distinct=min_distinct, per_layer=per_layer,
                    judge_model=judge_model, judge_target=judge_target, response_tokens=response_tokens,
                    random_controls=random_controls)
    lines = [f"{d['path']}(A/B 各 {d['n_prompts']} 会話、行数 {d['n_rows']}、位置 {d['positions']}、{d['seconds']:.0f} 秒)",
             "層ごとの ‖平均差‖/ρ(rel dose に直した大きさ)と前の層との cos:"]
    lines.append("  " + " ".join(f"L{r['layer']}:{r['rel']:.2f}" if r["rel"] is not None else f"L{r['layer']}:—"
                                 for r in d["layers"]))
    lines.append("  " + " ".join(f"L{r['layer']}:{r['cos_prev']:+.2f}" for r in d["layers"] if r["cos_prev"] is not None))
    sel = d["select"]
    if sel and sel["mode"] == "manual":
        lines.append(f"層 {sel['selected']} を選びました(手で指定)")
    elif sel:
        key = sel["metric"]
        b = sel["base"]
        head = (f"層選び(mode={sel['mode']}、{sel['n_prompts']} プロンプト、{sel['max_tokens']} トークン"
                + (f"、dose {sel['dose']:g}" if sel["dose"] is not None else "") + f"、{sel['seconds']:.0f} 秒)")
        lines += [head, f"  無介入: {key} {b[key]:.2f}  distinct-4 {b['distinct4']:.2f}",
                  f"  {'層':>4} {'効果':>6} {key:>6} {'KL':>7} {'d4':>5}" + (f" {'rel':>5}" if sel["mode"] == "add" else "")
                  + (f" {'対照p95':>7} {'null':>5}" if sel.get("random_controls") else "")]
        for r in sel["rows"] + ([{**sel["per_layer"], "layer": "層別"}] if sel["per_layer"] else []):
            mark = "  ← 選択" if r["layer"] == sel["selected"] else ("" if r["ok"] else "  ×")
            eff = "—" if r["effect"] is None else f"{r['effect']:+.2f}"
            val = "—" if r.get(key) is None else f"{r[key]:.2f}"
            rel = f" {r['rel_dose']:>5.2f}" if r.get("rel_dose") is not None else ""
            ctl = ""
            if sel.get("random_controls") and r.get("ctrl_effect_p95") is not None:
                ctl = f" {r['ctrl_effect_p95']:>+7.2f} {r['null_pct'] * 100:>4.0f}%"
            elif sel.get("random_controls"):
                ctl = f" {'—':>7} {'—':>5}"
            lines.append(f"  {r['layer']!s:>4} {eff:>6} {val:>6} {r['kl']:>7.3f} {r['distinct4']:>5.2f}{rel}{ctl}{mark}")
        crit = sel["criteria"]
        lines.append("基準: " + (f"KL ≤ {crit['kl_max']:g}" if "kl_max" in crit else f"distinct-4 ≥ {crit['min_distinct']:g}")
                     + (f"、効果 > 対照(同じ大きさのランダムな向き {sel['random_controls']} 本)の p95"
                        + ("。基準を満たさない層・効果が 0 以下の層(—)は対照を流していない" if sel["mode"] == "add" else "")
                        if sel.get("random_controls") else "")
                     + ("。選べる層がありません" if sel["selected"] is None else ""))
        if sel["selected"] is not None and sel["mode"] == "add":
            rd = next(r["rel_dose"] for r in sel["rows"] if r["layer"] == sel["selected"])
            lines.append(f"試す(平均差の {sel['dose']:g} 倍 = rel {rd:.2f}): "
                         f"gne steer --unit {sel['selected']}:dir:{d['name']}:+{rd:.2f} -p \"Who are you?\" --compare")
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
          resume: bool = typer.Option(True, "--resume/--no-resume", help="出力の JSONL の続きから"),
          yes: bool = _YES, budget: Optional[str] = _BUDGET, workers: Optional[str] = _WORKERS):
    """ニューロン(とランダム対照)を押して文章を生成し、rollout を JSONL に書く(総当たり)。"""
    s = holder["session"] = c.session()
    g = _gate(s, "sweep", {"layer": layer, "neurons": neurons, "design": design, "random_controls": random_controls,
                           "max_tokens": max_tokens, "prompt_set": prompt_set}, yes, budget)
    if workers:
        name = run or f"L{layer}-{g['design']}"
        d = _via_workers(s, "sweep", {"layer": layer, "neurons": g["neurons"], "design": g["design"],
                                      "random_controls": g["random_controls"], "max_tokens": max_tokens, "top_k": top_k,
                                      "run": name, "prompt_set": prompt_set, "dose_unit": dose_unit, "resume": resume},
                         workers)
        text = d.pop("text")
        return d, text + f"\n次: gne rank --run {name}(ワーカーごとの part-* を結合して読む)", 0
    d = s.sweep(layer, neurons=g["neurons"], design=g["design"], random_controls=g["random_controls"], max_tokens=max_tokens,
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
         when: Optional[str] = _WHEN, ablate: Optional[str] = _ABLATE,
         temperature: Optional[float] = _TEMP, top_k: Optional[int] = _TOP_K, top_p: Optional[float] = _TOP_P,
         min_p: Optional[float] = _MIN_P, presence_penalty: Optional[float] = _PRES,
         sample_seed: Optional[int] = _SSEED, greedy: bool = _GREEDY):
    """−/0/+ の 3 列比較チャット(ユーザーの発話は共有、返答は列ごと)。サンプリングなら 3 列とも同じ seed で引く。"""
    from .chat import ChatState, chat_turn, make_columns
    s = holder["session"] = c.session()
    if when:
        from .when import parse_when
        parse_when(when)
    base = s.recipe(units=unit, path=recipe, dose_unit=dose_unit, ablate=ablate) if (unit or recipe or ablate) else None
    sp = _sampling(base, greedy, temperature, top_k, top_p, min_p, presence_penalty, sample_seed)
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
                                                             when_override=when, sampling=sp)})
        d = {"columns": [{"label": col.label, "recipe": col.recipe.to_dict() if col.recipe else None} for col in cols],
             "system": system, "turns": turns, "sampling": None if sp is None else sp.to_dict()}
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
    print("列: " + " | ".join(col.label for col in cols) + f"({_sampling_text(sp)})"
          + "\n/reset で会話をやり直し、/quit か Ctrl-D で終了", file=sys.stderr)
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
                            on_column=on_column if stream else None, when_override=when, sampling=sp)
        if not stream:
            for col, rep in zip(cols, replies, strict=True):
                print(f"── {col.label} ──\n{rep['text']}")
        else:
            print()
        turns.append({"user": user, "replies": replies})
    return {"columns": [{"label": col.label, "recipe": col.recipe.to_dict() if col.recipe else None} for col in cols],
            "system": system, "turns": turns}, "", 0




def _via_workers(s, kind: str, params: dict, workers: str) -> dict:
    """ファイルのキューに入れて、全部の部分が終わるまで待つ。"""
    from .workers import submit, wait
    sub = submit(s, kind, params, workers=workers)
    sys.stderr.write(f"ワーカー {', '.join(sub['workers'])} に {len(sub['jobs'])} 個に分けて投げました(group {sub['group']})\n")
    t0 = time.monotonic()
    js = wait(s.store.root, sub["group"], progress=s.progress)
    ok = all(j["status"] == "done" for j in js)
    lines = [f"{kind}(ワーカー {len(js)} 個、壁時間 {time.monotonic() - t0:.0f} 秒)"]
    for j in js:
        el = (j.get("finished") or 0) - (j.get("started") or 0)
        lines.append(f"  {j['worker']}: {j['status']}、{el:.0f} 秒" + (f"、{j['error']['message']}" if j.get("error") else ""))
        if j.get("merge"):
            lines.append(f"  結合: {j['merge']['merged']}(ニューロン {j['merge']['n_done']}、対照 {j['merge']['n_controls']})")
    info = {"group": sub["group"], "workers": sub["workers"], "ok": ok, "wall_seconds": time.monotonic() - t0,
            "parts": [{"worker": j["worker"], "status": j["status"], "error": j.get("error"),
                       "seconds": (j.get("finished") or 0) - (j.get("started") or 0)} for j in js]}
    if not ok:
        raise GneError("WORKER_FAILED", "\n".join(lines))
    if kind == "sweep":                       # 部分の結果を足して、ふつうの sweep と同じ形にする
        rs = [j["result"] for j in js]
        out = {"run": params["run"], "path": str(s.run_path(params["run"])), "meta": rs[0]["meta"],
               "layer": params["layer"], "design": params["design"], "prompt_set": params["prompt_set"],
               "n_neurons": sum(r["n_neurons"] for r in rs), "n_controls": params["random_controls"],
               "n_units": sum(r["n_units"] for r in rs), "n_new_records": sum(r["n_new_records"] for r in rs),
               "seconds": info["wall_seconds"]}
    else:
        out = s.lens_summary(params["layer"], neurons=params["neurons"], show=params.get("show", 20),
                             seconds=info["wall_seconds"])
    return {**out, "workers": info, "text": "\n".join(lines)}


def _gate(s, kind: str, params: dict, yes: bool, budget: Optional[str]) -> dict:
    """CHANGE-03 C: 予算で絞り、見積もりが 30 分を超えて --yes が無ければ止める。返り値は実行する引数。"""
    from .estimate import NeedsConfirm, fmt_seconds, gate
    try:
        p, est = gate(s, kind, {**params, "yes": yes, **({"budget": budget} if budget else {})})
    except NeedsConfirm as e:
        e.error.hint = (e.error.hint or "") + f"(見積もりの内訳: {e.estimate['detail']})"
        raise e.error from e
    if est is not None and est.get("seconds"):
        b = est.get("budget")
        sys.stderr.write(f"見積もり: {fmt_seconds(est['seconds'])}" + (f"(予算で絞った: {b})" if b else "") + "\n")
    return p


def _metric(x, fmt: str) -> str:
    return "—" if x is None else format(x, fmt)


@gne_command(app, "worker")
def worker(c: Common, holder: dict, name: str = typer.Option(..., "--name", help="ワーカーの名前(例 xtx・5060)"),
           poll: float = typer.Option(2.0, "--poll", help="キューを見る間隔(秒)"),
           once: bool = typer.Option(False, "--once", help="キューが空になったら終わる"),
           max_jobs: Optional[int] = typer.Option(None, "--max-jobs", help="この数のジョブを回したら終わる")):
    """ワーカー(CHANGE-03 B): <root>/jobs/ のファイルのキューから、自分あて(か宛先なし)で同じモデルのジョブを取って回す。
    GPU ごとに別の環境・別のプロセスで立て、--root を共有する(docs/setup.md)。投入は gne sweep / lens steer の --workers。"""
    from .workers import run_worker
    if not re.fullmatch(r"[\w.\-]+", name):
        raise GneError("BAD_ARGS", "--name は英数字・_・-・. だけにしてください")
    s = holder["session"] = c.session()
    s.engine                                              # 先に読み込んで VRAM を確保する
    sys.stderr.write(f"ワーカー {name}: {s.meta.model_id}、置き場所 {s.store.root}\n")
    n = run_worker(s, name, poll=poll, once=once, max_jobs=max_jobs, log=lambda m: sys.stderr.write(m + "\n"))
    return {"name": name, "model_id": s.meta.model_id, "n_jobs": n}, f"ワーカー {name}: {n} 個のジョブを回しました", 0


@gne_command(app, "connections")
def connections_cmd(c: Common, holder: dict, k: int = typer.Option(24, "--k", help="ニューロンごとに残す上流・下流の数"),
                    window: Optional[int] = typer.Option(None, "--window", help="|j − i| ≤ window の層ペアだけ(既定は全ペア)"),
                    chunk: int = typer.Option(1024, "--chunk"), yes: bool = _YES, budget: Optional[str] = _BUDGET):
    """ニューロン間の配線(層 i の書き込みが層 j の活性に与える寄与)を重みだけから求める(別冊 §8.4、CPU)。"""
    s = holder["session"] = c.session()
    g = _gate(s, "connections", {"window": window}, yes, budget)
    d = s.connections(k=k, window=g.get("window"), chunk=chunk)
    text = (f"{d['path']}({d['n_pairs']} 層ペア、window {d['window'] if d['window'] is not None else 'なし'}、"
            f"{d['seconds']:.0f} 秒" + (f"、ピーク {d['peak_rss_gb']:.1f} GB" if d.get("peak_rss_gb") else "") + ")")
    return d, text, 0


@gne_command(app, "writers")
def writers_cmd(c: Common, holder: dict,
                contrast: Path = typer.Option(..., "--contrast", help="対照集合 YAML(A = 見たい状態、B = 中立)"),
                name: Optional[str] = typer.Option(None, "--name", help="保存名(既定は対照集合の name)"),
                ref_layer: Optional[int] = typer.Option(None, "--ref-layer", help="寄与を測る方向の層(既定は最後から 2 番目)"),
                replies_a: Optional[Path] = typer.Option(None, "--replies-a", help="A の返事を生成せずに読ませる(YAML / JSON の、プロンプトと同じ長さのリストか {プロンプト: 返事})"),
                replies_b: Optional[Path] = typer.Option(None, "--replies-b"),
                max_tokens: int = typer.Option(96, "--max-tokens", help="生成する返事の長さ"),
                show: int = typer.Option(20, "--show"), yes: bool = _YES):
    """書き手の強さ: A の状態(例 ヤンデレの system)で、各ニューロンが残差に実際に書き足した量(活性の差 × 書く向き)と、
    参照の層の方向への直接の寄与。方向を埋め込み・mixer(attention・畳み込み)・MLP が何割ずつ書いたかも出す。"""
    s = holder["session"] = c.session()
    _gate(s, "writers", {"contrast": str(contrast), "max_tokens": max_tokens}, yes, None)
    d = s.writers(contrast, name=name, ref_layer=ref_layer, replies_a=replies_a, replies_b=replies_b,
                  max_tokens=max_tokens, top=max(show, 50))
    t = d["totals"]
    lines = [f"{d['path']}(返事のトークン A {d['n_tokens']['A']} / B {d['n_tokens']['B']}、参照 L{d['ref_layer']}、{d['seconds']:.0f} 秒)",
             f"L{d['ref_layer']} の方向(‖r‖ = {d['ref_norm']:.2f})を書いた割合: 埋め込み {t['embed']:+.0%}・mixer {t['mixer']:+.0%}・MLP {t['mlp']:+.0%}"
             f"(MLP のうち上位 10 個 {t['top10']:+.0%}・100 個 {t['top100']:+.0%}・1000 個 {t['top1000']:+.0%})",
             f"  {'層':>4} {'種類':<17} {'mixer':>7} {'MLP':>7} {'‖r_L‖':>7} {'cos(r_L, r*)':>12}"]
    nr = d["ref_norm"]
    for x in d["layers"]:
        if "mixer_to_ref" not in x:
            continue
        mlp = "—" if x["mlp_to_ref"] is None else f"{x['mlp_to_ref'] / nr:+.1%}"
        lines.append(f"  {x['layer']:>4} {x['mixer']:<17} {x['mixer_to_ref'] / nr:>+7.1%} {mlp:>7} {x['r_norm']:>7.2f} {x['cos_ref']:>12.2f}")
    lines.append(f"方向へ書き足したニューロンの上位(寄与 = 活性の差 × 書く向きの r* 成分、割合は ‖r*‖ に対して):")
    for x in d["top"][:show]:
        lines.append(f"  L{x['layer']:<3} N{x['neuron']:<6} {x['share']:+.2%}  活性 A {x['act_a']:+.2f} / B {x['act_b']:+.2f}  向き {x['align']:+.2f}")
    if d.get("mlp_check_max") is not None:
        lines.append(f"確かめ: 式で求めた MLP の出力と捕捉した出力の差 最大 {d['mlp_check_max']:.1%}")
    return d, "\n".join(lines), 0


@gne_command(app, "writers-export")
def writers_export(c: Common, holder: dict, name: str = typer.Option(..., "--name", help="gne writers の分析の名前"),
                   out: str = typer.Option(..., "--out", help="書き出す方向の名前"),
                   part: str = typer.Option("mlp", "--part", help="mlp / mixer / all(mixer + MLP)/ r(残差の差そのもの)"),
                   top: Optional[int] = typer.Option(None, "--top", help="--part mlp: 参照の方向への寄与の上位 N 個のニューロンだけ"),
                   layers: Optional[str] = typer.Option(None, "--layers", help="all / 10-16 / 1,5,9-12")):
    """書き手の分析の書き足しを、層ごとの方向として書き出す。"L:dir:<out>:1"(dose_unit raw)で、A の状態で書かれた量を
    その層にそのまま足す(倍率で強める)。上位の書き手だけを足して性格が出るかを確かめる用。"""
    from .writers import export_writes
    if not re.fullmatch(r"[\w.\-]+", out):
        raise GneError("BAD_ARGS", "--out は英数字・_・-・. だけにしてください")
    s = holder["session"] = c.session()
    d = export_writes(s, name, out, part=part, top=top, layers=parse_layers(layers))
    text = (f"{d['path']}({part}{'' if top is None else f'、上位 {top} 個'}、{len(d['layers'])} 層)\n"
            + "  ‖v_L‖: " + " ".join(f"L{L}:{n:.2f}" for L, n in d["norms"].items())
            + f"\n使い方(raw の倍率): --dose-unit raw " + " ".join(f"--unit {u}" for u in d["units"][:3]) + (" …" if len(d["units"]) > 3 else ""))
    return d, text, 0


@gne_command(heads_app, "contrast", "heads contrast")
def heads_contrast(c: Common, holder: dict,
                   contrast: Path = typer.Option(..., "--contrast", help="対照集合 YAML(triggers があれば trigger_mass も測る)"),
                   name: Optional[str] = typer.Option(None, "--name", help="保存名(既定は対照集合の name)"),
                   direction: Optional[str] = typer.Option(None, "--direction", help="dir_cos に使う方向(既定は対照集合と同じ名前の方向)"),
                   system_b2: Optional[str] = typer.Option(None, "--system-b2", help="床に使う別の中立の system(既定は同じくらいの長さの中立の文)"),
                   layers: str = typer.Option("all", "--layers", help="all / 10-16 / 1,5,9-12"),
                   max_tokens: int = typer.Option(48, "--max-tokens", help="比べる返事の長さ(system_a で greedy に作る)"),
                   sort: str = typer.Option("z", "--sort", help="表の並び: z / trigger_mass / pattern_shift / out_shift / dir_cos(保存は z の順)"),
                   show: int = typer.Option(20, "--show")):
    """同じ会話をペルソナあり(system_a)と中立(system_b)で流し、ヘッドごとに注目と出力の差を測って順位づける。
    雑音の床は中立どうし(system_b と system_b2)の差。"""
    from .heads import METRICS
    if sort not in ("z", *METRICS):
        raise GneError("BAD_ARGS", f"--sort は z / {' / '.join(METRICS)} のどれかです")
    s = holder["session"] = c.session()
    d = s.heads_contrast(contrast, name=name, direction=direction, system_b2=system_b2, layers=parse_layers(layers),
                         max_tokens=max_tokens)
    fl = d["floors"]
    rows = d["top"]
    if sort != "z":
        from .heads import load_candidates
        rows = sorted((x for x in load_candidates(s, d["name"]) if x[sort] is not None), key=lambda x: -x[sort])
    lines = [f"{d['path']}({d['n_heads']} ヘッド、{d['n_prompts']} プロンプト(トリガーあり {d['n_with_triggers']})、"
             f"方向 {d['direction'] or 'なし'}、{d['seconds']:.0f} 秒)",
             "床(中立どうしの p95): " + "、".join(f"{k} {v:.3f}" for k, v in fl.items()),
             f"  {'順':>3} {'住所':>9} {'種類':>6} {'trigger':>8} {'pattern':>8} {'out':>6} {'dir_cos':>8} {'z':>6} {'null':>5}"]
    for i, x in enumerate(rows[:show], 1):
        lines.append(f"  {i:>3} {('L' + str(x['layer']) + ' H' + str(x['head'])):>9} {x['kind']:>6} "
                     f"{_metric(x['trigger_mass'], '+.3f'):>8} {_metric(x['pattern_shift'], '.3f'):>8} "
                     f"{_metric(x['out_shift'], '.2f'):>6} {_metric(x['dir_cos'], '+.3f'):>8} {_metric(x['z'], '.1f'):>6} "
                     f"{_metric(None if x['null_pct'] is None else x['null_pct'] * 100, '.0f'):>4}%")
    lines.append("読み方: trigger が大きいヘッドは「気づく係」、dir_cos が大きいヘッドは「ペルソナの方向を書く係」。"
                 "試す: gne steer --unit L:Hh:x0 / x2 --compare")
    return d, "\n".join(lines), 0


@gne_command(heads_app, "attention", "heads attention")
def heads_attention(c: Common, holder: dict, text: str = typer.Option(..., "--text", "-t", help="流す文"),
                    pos: int = typer.Option(-1, "--pos", help="見るトークンの位置(負は後ろから)"),
                    layer: Optional[int] = typer.Option(None, "--layer"), head: Optional[int] = typer.Option(None, "--head"),
                    top: int = typer.Option(5, "--top"), chat: bool = typer.Option(False, "--chat", help="text をユーザーの発話としてテンプレートに入れる"),
                    system: Optional[str] = typer.Option(None, "--system", help="--chat の system")):
    """選んだトークンが各ヘッドでどこを見ているか(上位 top)と、per-head 出力のノルム(fires のヘッド版)。"""
    s = holder["session"] = c.session()
    d = s.attention(text, pos=pos, layer=layer, head=head, top=top, system=system, chat=chat)
    lines = [f"{d['n_tokens']} トークン、位置 {d['pos']} {d['tokens'][d['pos']]!r}"]
    rows = sorted(d["heads"], key=lambda x: -x["norm"])
    for x in rows[: (len(rows) if layer is not None else 30)]:
        t = " ".join(f"{y['token']!r}({y['weight']:.2f})" for y in x.get("top", []))
        lines.append(f"  L{x['layer']:<3} H{x['head']:<3} {x['kind']:<6} ‖out‖ {x['norm']:7.2f}  {t}")
    return d, "\n".join(lines), 0


@gne_command(heads_app, "show", "heads show")
def heads_show(c: Common, holder: dict, layer: int = typer.Argument(..., help="層 L"),
               head: int = typer.Argument(..., help="ヘッド番号"),
               lens: bool = typer.Option(True, "--lens/--no-lens", help="W_O ブロックの logit lens(CPU で数秒〜)")):
    """ヘッド 1 個の調書(種類・対照分析での順位・名前・当てた gain・W_O ブロックの logit lens)。"""
    s = holder["session"] = c.session()
    d = s.head(layer, head, lens=lens)
    lines = [f"L{layer} H{head}" + (f"「{d['name']}」" if d["name"] else "") +
             f"  {d['kind']}、{d['n_head']} ヘッド(K/V {d['n_head_kv']})、幅 {d['head_dim']}、{d['writer']} の列 {d['block'][0]}:{d['block'][1]}"
             + (f"、K/V グループ {d['kv_group']}" if d["kv_group"] is not None else "")]
    for x in d["contrasts"]:
        lines.append(f"対照分析 {x['name']}: #{x['rank']}/{x['n_ranked']}  trigger {_metric(x['trigger_mass'], '+.3f')}  "
                     f"pattern {_metric(x['pattern_shift'], '.3f')}  out {_metric(x['out_shift'], '.2f')}  "
                     f"dir_cos {_metric(x['dir_cos'], '+.3f')}  z {_metric(x['z'], '.1f')}")
    if d["gain_k"]:
        lines.append("当てた gain: " + ", ".join(f"×{k:g}" for k in d["gain_k"]))
    if d["logit_lens"]:
        ll = d["logit_lens"]
        lines.append(f"W_O ブロックの logit lens(1 本目の特異値の割合 {ll['sv_share']:.0%}):")
        for i, cp in enumerate(ll["components"], 1):
            lines.append(f"  主成分 {i}: ↑ {_toks(cp['up_tokens'])}\n{'':11}↓ {_toks(cp['down_tokens'])}")
    return d, "\n".join(lines), 0


@gne_command(heads_app, "query", "heads query")
def heads_query(c: Common, holder: dict, layer: int = typer.Option(..., "--layer"), head: int = typer.Option(..., "--head"),
                direction: str = typer.Option(..., "--direction", help="ペルソナの方向(gne direction の名前)。これが入ったときだけ働く"),
                seek: Path = typer.Option(..., "--seek", help="探しに行く語(triggers)を持つ対照集合 YAML"),
                scales: str = typer.Option("0,0.5,1,2,4", "--scales")):
    """(実験的、CHANGE-04 E)ヘッドの Q に rank-1 の LoRA を足し、ペルソナの方向が入ったときだけトリガーを見に行かせる。
    scale ごとに、そのヘッドのトリガーへの注目をペルソナあり/なしで測る(V17)。レシピでは
    {layer, kind: head, head, mode: query, direction, seek, scale}。"""
    s = holder["session"] = c.session()
    d = s.heads_query(layer, head, direction=direction, seek=seek, scales=_doses(scales))
    lines = [f"L{layer} H{head}(K/V グループ {d['kv_group']}、u は次元 {d['dims'][0]}〜{d['dims'][1] - 1} の {d['dims'][2]} 個、"
             f"トリガーのトークン {d['n_trigger_tokens']}、{d['seconds']:.0f} 秒)",
             f"v·norm(x)(返事を書き始める位置): ペルソナあり {d['v_mean_a']:.2f} / なし {d['v_mean_b']:.2f}",
             f"  {'scale':>6} {'あり':>8} {'なし':>8} {'他のヘッドの最大の変化':>12}"]
    for r in d["rows"]:
        lines.append(f"  {r['scale']:>6g} {r['head_a']:>8.4f} {r['head_b']:>8.4f} {r['others_shift_max']:>12.4f}")
    lines.append(("単調に増える" if d["monotonic"] else "単調ではない") +
                 f"(あり {d['gain_a']:+.3f}、なし {d['gain_b']:+.3f})")
    return d, "\n".join(lines), 0


@gne_command(heads_app, "name", "heads name")
def heads_name(c: Common, holder: dict, layer: int = typer.Argument(...), head: int = typer.Argument(...),
               name: str = typer.Argument("", help="名前(空で消す)")):
    """ヘッドに名前を付ける(ニューロンと同じ names に "L:Hh" で入る)。"""
    s = holder["session"] = c.session()
    d = s.set_head_name(layer, head, name)
    return d, f"L{layer} H{head}: {name or '(名前を消しました)'}", 0


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
    for wr in d.get("writers", []):
        lines.append(f"書き手 {wr['name']}: 方向 L{wr['ref_layer']} への寄与 {wr['share']:+.2%}(#{wr['rank']}/{wr['n_ranked']})、"
                     f"活性 A {wr['act_a']:+.2f} / B {wr['act_b']:+.2f}")
    for e in d["evals"]:
        how = f"x{e['k']:g}" if e["mode"] == "gain" else f"{e['dose']:+g}"
        ship = "配布可なし" if e["max_shippable_dose"] is None else f"配布可の最大倍率 {e['max_shippable_dose']:g}"
        top = e["at_max"]
        lines.append(f"eval {e['name']}({how}、when={e['when']}、ユニット {e['n_units']}): {ship}"
                     + (f"、倍率 {top['dose']:g} で能力 −{top['capability_drop_pt']:.1f}pt・新語 {top['novelty']:.2f}" if top else ""))
    if d["gain_k"]:
        lines.append("当てたことのある gain: " + " ".join(f"x{k:g}" for k in d["gain_k"]))
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
