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
experts_app = typer.Typer(no_args_is_help=True, help="MoE の専門家(対照分析・調書、CHANGE-05)")
app.add_typer(experts_app, name="experts")
# gne direction(方向を作る)は、サブコマンドが無ければ従来どおり動くグループ。gne direction score は速い指標(CHANGE-06 B)
direction_app = typer.Typer(no_args_is_help=False, invoke_without_command=True)
app.add_typer(direction_app, name="direction")
nw_app = typer.Typer(no_args_is_help=True, help="FiSH(Formula-instantiated Scene-gated Hidden units): 計算で作るニューロン(mode: nw。ニューロンを学習なしで計算した重みで上書きする、docs/fish.md)")
app.add_typer(nw_app, name="nw")
from .modal_cli import modal_app  # noqa: E402
app.add_typer(modal_app, name="modal")
headproj_app = typer.Typer(no_args_is_help=True, help="ヘッドの方向成分の射影(mode: proj)の書き出し(CHANGE-06 E)")
app.add_typer(headproj_app, name="headproj")


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
    mmproj: Optional[Path] = None

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
                       n_seq_max=self.n_seq_max, seed=self.seed, verbose=self.verbose, progress=self.progress,
                       mmproj=self.mmproj)

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
        inspect.Parameter("mmproj", K, annotation=Optional[Path],
                          default=typer.Option(None, "--mmproj", envvar="GNE_MMPROJ",
                                               help="音声の入力に使う mmproj(既定はモデルと同じフォルダの mmproj*.gguf)")),
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


def gne_callback(typer_app: typer.Typer, name: str):
    """gne_command と同じ形の本体を、サブコマンドを持つグループの callback として登録する(サブコマンドが無いときだけ動く)。"""

    def deco(f: Callable):
        sig = inspect.signature(f, eval_str=True)
        own = list(sig.parameters.values())[2:]
        ctx_param = inspect.Parameter("ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=typer.Context)

        @functools.wraps(f)
        def wrapper(ctx: typer.Context, **kwargs):
            if ctx.invoked_subcommand is not None:
                return
            over = {k: kwargs.pop(k) for k in _COMMON_NAMES}
            c = (ctx.obj or Common()).merged(over)
            _run(c, name, f, kwargs)

        wrapper.__signature__ = sig.replace(parameters=[ctx_param, *own, *_common_params()])
        wrapper.__annotations__ = {}
        typer_app.callback(invoke_without_command=True)(wrapper)
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
         json_: bool = typer.Option(False, "--json", help="stdout に JSON だけを出す"),
         mmproj: Optional[Path] = typer.Option(None, "--mmproj", envvar="GNE_MMPROJ",
                                               help="音声の入力に使う mmproj(既定はモデルと同じフォルダの mmproj*.gguf)")):
    ctx.obj = Common(model, root, n_gpu_layers, n_ctx, n_seq_max, seed, verbose, json_, mmproj)


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


def _keyed_who(kind: str, expert=None) -> str:
    """keyed の書き込み先の短い書き方(ユニットの記法と同じ: ffn / attn / S / Ee)。"""
    return {"ffn": "ffn", "attn": "attn", "shared": "S"}.get(kind, f"E{expert}")


def _units_text(units: list[dict]) -> str:
    rows = []
    for u in units:
        if u.get("mode") == "keyed":                         # CHANGE-07: α は raw(d = r_L)
            a = u.get("alpha", u.get("k"))
            size = ""
            if u.get("norm"):
                size = f"、鍵が反応する位置で |α|‖r_L‖={u['norm']:.2f}" + (
                    f"(残差ノルム ρ_L の {u['rel_push']:.2f} 倍)" if u.get("rel_push") is not None else "")
            rows.append(f"  L{u['layer']} {_keyed_who(u['kind'], u.get('expert'))} keyed α={a:+g} @{u.get('direction')} "
                        f"#{u.get('key')}(鍵 κ が反応する状態だけ方向を書く、LoRA で当てる{size})")
            continue
        who = {"neuron": f"N{u['neuron']}", "random": f"random{u['seed']}", "head": f"H{u.get('head')}"}.get(
            u["kind"], f"dir:{u.get('direction')}")
        if u.get("mode") == "proj":                          # CHANGE-06
            rows.append(f"  L{u['layer']} {who} s{u['k']:g} @{u.get('direction')}"
                        f"(書き込みの方向成分を {u['k']:g} 倍ぶん引く、LoRA で当てる)")
            continue
        if u.get("mode") == "query":
            rows.append(f"  L{u['layer']} {who} query {u.get('direction')} ×{u['k']:g}")
            continue
        if u.get("mode") == "route":
            rows.append(f"  L{u['layer']} E{u.get('expert')} route {u['dose']:+g}")
            continue
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
    # 確かめる対象が無い検査は省く(失敗ではない)。MoE の検査は tests/test_change05.py と gne experts、残差の流れは V26
    why = {}
    if not s.engine.cvec_supported:
        why |= dict.fromkeys(("V3", "V4", "V12"), f"{s.meta.arch} のグラフは control vector を読みません")
    if s.meta.n_expert:
        why |= dict.fromkeys(("V2", "V11", "V13", "V14"), "全層 MoE です。この検査は dense の ffn_down が前提です(MoE は V24 と gne experts で確かめます)")
    steps = [(vid, (lambda vid=vid, fn=fn: VResult(vid, "省略", "skip", why[vid]) if vid in why else fn())) for vid, fn in steps]
    if s.engine.hc_streams:
        steps.append(("V26", v.v26_hc_streams))

    def show(r):
        results.append(r)
        c.human(f"[{r.status.upper():4}] {r.id} {r.title}: {r.detail}")

    for vid, fn in steps:
        try:
            show(fn())
        except Exception as ex:  # noqa: BLE001
            show(VResult(vid, "例外", "fail", repr(ex)))
    if bench and not s.engine.cvec_supported:
        show(VResult("V7", "ベンチ", "skip", "control vector を読まない arch です"))
    elif bench:
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
_UNIT = typer.Option(None, "--unit", help="L:N:dose(例 14:8949:+1.0)、L:random:SEED:dose、gain は L:N:x2、方向は L:dir:NAME:dose、ヘッドは L:Hh:xk(例 21:H7:x0)、ヘッドの方向成分の射影は L:Hh:sS@NAME(例 33:H16:s6@evil)、鍵つきの書き込みは L:ffn:keyed:aα@NAME#KEY(例 19:ffn:keyed:a1@zundamon_endings#zundamon_endings、attention は L:attn:keyed:…)。複数可")
_WORKERS = typer.Option(None, "--workers", help="auto か名前のカンマ区切り: gne worker に分けて投げ、終わるまで待つ(CHANGE-03 B)")
_YES = typer.Option(False, "--yes", help="見積もりが 30 分を超えても走らせる")
_BUDGET = typer.Option(None, "--budget", help="時間の予算(30m・2h・90s)。収まるように候補や窓を絞る")
_DOSE_UNIT = typer.Option(None, "--dose-unit", help="rel / absmax / raw(既定はレシピの値、--unit なら rel)")
_SCALE = typer.Option(1.0, "--scale", help="全ユニットの dose に掛ける係数")
_ABLATE = typer.Option(None, "--ablate", help="方向の射影消去(§6.17): NAME、NAME@層、NAME:倍率(例 evil@10:0.5)。LoRA で当てる")
_WHEN = typer.Option(None, "--when", help="足す位置(§6.15): all / gen / assistant / skip-replies / after:<文字列> / ramp:<a>-<b>。既定はレシピの値(無ければ all)")
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
          contrast_prompts: Optional[Path] = typer.Option(None, "--contrast-prompts", help="--contrast: 検証プロンプトを差し替える YAML(文字列のリスト、または対照集合なら reactivity の with か user_prompts)"),
          audio: Optional[list[Path]] = typer.Option(None, "--audio", help="音声ファイル(wav / mp3 / flac)をユーザーの発話にする。--prompt があれば音声の後に文を足す。mmproj が要る")):
    """レシピまたは単発指定でプロンプトに回答する(既定は greedy、1 本ずつ)。"""
    s = holder["session"] = c.session()
    r = s.recipe(units=unit, path=recipe, dose_unit=dose_unit, scale=scale, when=when, ablate=ablate)
    sp = _sampling(r, greedy, temperature, top_k, top_p, min_p, presence_penalty, sample_seed)
    if prompt is None and contrast is None and not audio:
        raise GneError("BAD_ARGS", "--prompt か --audio か --contrast が要ります")
    if prompt is not None or audio:
        d = s.steer(r, prompt=prompt, system=system, max_tokens=max_tokens, compare=compare, sampling=sp,
                    audio=audio)
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
         requant: str = typer.Option("auto", "--requant", help="--bake の書き方: auto(パッチで書ければスケールパッチ、書けなければ q8_0。keyed を含むならテンソルごとに、元が Q8_0 なら Q8_0 のまま、それ以外は f16。Q8_0 のままでは編集の 1 割以上が丸めに消えるテンソルだけ f16)/ patch / q8_0 / f16 / keep(元が F32・F16・BF16 ならその型。専門家のテンソル ffn_down_exps は元の型を保ち、編集しない専門家はバイトのまま。K-quant・IQ の専門家は q8_0)"),
         key: Optional[str] = typer.Option(None, "--key", help="#KEY を省いた keyed のユニットの鍵(gne direction key で作ったもの)"),
         direction: Optional[str] = typer.Option(None, "--direction", help="@NAME を省いた proj・keyed のユニットの方向"),
         strip_meta: bool = typer.Option(False, "--strip-meta", help="gne の記録(gne.* のメタデータ・レシピ・名前の印)を残さない(人に渡す版)")):
    """gain ユニット(L:N:x2 など、出力を k 倍。ヘッドは L:Hh:xk)を LoRA アダプタ GGUF か書き換え GGUF に書き出す(§6.16)。
    ヘッドの方向成分の射影(proj)と鍵つきの書き込み(keyed、CHANGE-07)も同じ LoRA・焼き込みに入る。"""
    if in_path and c.model and in_path.resolve() != c.model.resolve():
        raise GneError("BAD_ARGS", "--in と --model が違うファイルです")
    s = holder["session"] = c.session(in_path)
    r = s.recipe(units=unit, path=recipe, scale=scale, default_key=key, default_direction=direction)
    out = out or s.default_export_path(r, bake=bake, kind="gain")
    d = s.gain(r, out, bake=bake, requant=requant, strip_meta=strip_meta)
    if bake:
        text = f"{d['path']}({d['bytes'] / 1e6:.1f} MB)、書き換え: " + ", ".join(_mod_text(m) for m in d["modified"])
        if d.get("method") == "patch":
            text += f"\nスケールパッチ({d['patch_bytes'] / 1e3:.1f} KB、量子化はそのまま)。当てた操作は {d['path']}.patch.json"
        text += "".join("\n" + ln for ln in _keyed_export_lines(d))
    else:
        rows = ([f"  L{L}: " + ", ".join(f"N{int(n)} ×{k:g}" for n, k in nk) for L, nk in (d.get("layers") or {}).items()]
                + [f"  L{L}: " + ", ".join(f"H{int(h)} ×{k:g}" for h, k in hk) for L, hk in (d.get("heads") or {}).items()]
                + [f"  L{L}: " + ", ".join(f"H{int(h)} s{sv:g}@{dn}" for h, sv, dn in x["heads"])
                   + f"(方向成分の射影、階数 {x['rank']}{'' if x['folded'] else '、方向が混ざっているので 1 に畳めない'})"
                   for L, x in (d.get("proj_layers") or {}).items()]
                + _keyed_export_lines(d)
                + [f"  クエリ操作: {p}" for p in (d.get("query_loras") or [])])
        text = (f"{d['path']}({d['bytes'] / 1e6:.2f} MB、LoRA)" + "".join("\n" + r for r in rows)
                + "\n使い方: llama-cli -m <model> --lora " + d["path"])
    return d, text, 0


def _mod_text(m) -> str:
    """焼き込みで書き換えたテンソル 1 つ(分割の焼き込みは名前だけ)。keyed・proj は量子化し直しで失った分も出す。"""
    if isinstance(m, str):
        return m
    s = f"{m['tensor']}({m.get('from', m.get('type', '?'))} → {m.get('to', '?')}"
    if m.get("experts") is not None:            # 専門家の 3 次元テンソル(keyed の expert:e、CHANGE-07)
        s += "、専門家 " + ",".join(f"E{e}" for e in m["experts"])
        s += "、ほかの専門家は" + ("バイトのまま" if m.get("untouched") == "raw" else f"変換({m.get('untouched')})")
        if m.get("bytes_to") is not None and int(m["bytes_to"]) != int(m.get("bytes_from", m["bytes_to"])):
            s += f"、{int(m['bytes_from']) / 1e6:.1f} → {int(m['bytes_to']) / 1e6:.1f} MB"
    if m.get("requant_error") is not None:
        s += f"、丸めで失った分は編集の {m['requant_error'] * 100:.1f}%"
    if m.get("edit_kept") is not None:
        s += f"、残った編集 {m['edit_kept'] * 100:.1f}%"
    if m.get("fallback"):
        s += "、Q8_0 では編集が埋もれるので F16"
    return s + ")"


def _keyed_export_lines(d: dict, *, requant_line: bool = True) -> list[str]:
    """gain・headproj export の結果の keyed の行(CHANGE-07): 層ごとの書き込み先と階数(= その層の鍵の数)・鍵・方向と、
    焼き込みなら量子化し直しで失った分の最大(requant_line)。"""
    lines = []
    for L, x in (d.get("keyed_layers") or {}).items():
        tens = ", ".join(f"{n} 階数 {r}" for n, r in x["tensors"].items())
        lines.append(f"  L{L}: 鍵つきの書き込み {tens}(鍵 {', '.join(x['keys'])}、方向 {', '.join(x['directions'])})")
    err, kept = d.get("requant_error_max"), d.get("edit_kept_min")
    if requant_line and d.get("keyed_layers") and err is not None:
        bad = (kept < 0.9) if kept is not None else err > 0.5     # 残った編集の割合で見る(無ければ丸めのずれ)
        lines.append(f"  量子化し直しで失った分は編集の大きさの最大 {err * 100:.1f}%"
                     + (f"、残った編集は最小 {kept * 100:.1f}%" if kept is not None else "")
                     + ("(編集が丸めに埋もれています。--requant f16 か auto、または LoRA で配ってください)" if bad else ""))
    return lines


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
             control_pool: str = typer.Option("before", "--control-pool", help="proj の対照のヘッドの母集団: before(射影の値を測る層より前の全ヘッド)/ same-layers(同じ層、厳しい対照)"),
             compare_methods: bool = typer.Option(False, "--compare-methods", help="proj のレシピと同じ方向で、全層射影・ヘッドの方向成分・ヘッドの丸ごと倍率・ニューロンの消去を、効果をそろえて比べる(CHANGE-06 F.3)"),
             match_effect: Optional[float] = typer.Option(None, "--match-effect", help="--compare-methods: そろえる A 側の射影の値(省くとこのレシピの水準)"),
             score_layer: Optional[int] = typer.Option(None, "--score-layer", help="--compare-methods: 射影の値を測る層"),
             alloc: Optional[str] = typer.Option(None, "--alloc", help="gne heads allocate の結果(名前かパス)の組を、ヘッド数ごとに評価する(CHANGE-06 F.4)"),
             n_heads: Optional[str] = typer.Option(None, "--n-heads", help="--alloc: 評価するヘッド数(例 6,5,4,3)"),
             direction_metrics: bool = typer.Option(True, "--direction-metrics/--no-direction-metrics",
                                                    help="方向を使うレシピで、方向の指標(射影の値の下がり方・ずれ・判定の一致率)も測る"),
             dir_max_examples: int = typer.Option(64, "--dir-max-examples", help="方向の指標で読む質問の数(A・B それぞれ。0 = 全部)"),
             keyed: bool = typer.Option(True, "--keyed/--no-keyed",
                                        help="keyed のユニットを持つレシピで、eval の選択性(書き込みたい位置・書き込みたくない位置の残差の変化)・方式の比較・"
                                             "比較用の鍵(ラベルをシャッフルして同じ手順で作った鍵)との比較も測る"),
             keyed_contrast: Optional[str] = typer.Option(None, "--keyed-contrast",
                                                          help="keyed: 鍵を作った対照集合(configs/contrast/ の名前かパス)。"
                                                               "省くと方向を作った対照集合が鍵と同じならそれ"),
             keyed_examples: int = typer.Option(8, "--keyed-examples", help="keyed: 会話の組ごとに読む数(0 = 全部)"),
             keyed_pseudo: int = typer.Option(8, "--keyed-pseudo", help="keyed: eval の選択性を比べる比較用の鍵の本数(鍵のファイルにある比較用の鍵の本数まで)"),
             keyed_subset: Optional[str] = typer.Option(None, "--keyed-subset",
                                                        help="keyed: 読む例 even / odd / all(既定は鍵の作成と反対側)"),
             keyed_match: bool = typer.Option(True, "--keyed-match/--no-keyed-match",
                                              help="keyed: 方式の比較で add・gain・proj の大きさを、書き込みたい位置で keyed にそろえる(省くと速い)"),
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
    if alloc:                                            # CHANGE-06 F.4: ヘッド数の削減曲線
        d = s.eval_alloc_curve(alloc, n_heads=_ints(n_heads), suite=suite, when=when, random_controls=random_controls,
                               name=name)
        lines = [f"ヘッド数の削減曲線 {d['name']}(方向 {d['direction']}、L{d['score_layer']}、目標 {d['target']:g}、"
                 f"{d['seconds']:.0f} 秒)→ {d['path']}",
                 f"  {'数':>3} {'最大s':>6} {'残る':>5} {'能力低下':>8} {'ppl悪化':>8} {'混入':>6} {'下がり':>7} {'B直交':>7} {'一致率':>6}"]
        for r_ in d["rows"]:
            lines.append(f"  {r_['n_heads']:>3} {r_['s_max']:>6.2f} {r_['remain']:>5} {r_['capability_drop_pt']:>7.1f}pt "
                         f"{r_['ppl_worse_pct']:>+7.1f}% {r_['mix_increase'] * 100:>+5.2f}% {_f(r_.get('dir_drop_a'), '+7.2f')} "
                         f"{_f(r_.get('drift_orth_b'), '7.4f')} {_f(r_.get('rate'), '6.2f')}"
                         + ("  ← allocate の推し" if r_["recommended"] else ""))
        return d, "\n".join(lines), 0
    r = s.recipe(units=unit, path=recipe, dose_unit=dose_unit, ablate=ablate)
    if compare_methods:                                  # CHANGE-06 F.3
        d = s.eval_compare_methods(r, match_effect=match_effect, score_layer=score_layer, top_neurons=top_neurons,
                                   suite=suite, when=when, name=name)
        lines = [f"方式の比較 {d['name']}(方向 {d['direction']}、L{d['score_layer']} の A 側の射影の値を {d['target']:.2f} にそろえた。"
                 f"無介入 A {d['base_score_a']:.2f} / B {d['base_score_b']:.2f}、{d['seconds']:.0f} 秒)→ {d['path']}",
                 f"  {'方式':<10} {'α':>6} {'A射影':>7} {'B直交':>7} {'能力低下':>8} {'ppl悪化':>8} {'混入':>6} {'一致率':>6}"]
        for r_ in d["rows"]:
            if r_.get("excluded"):
                lines.append(f"  {r_['method']:<10} 除外: {r_['excluded']}")
                continue
            lines.append(f"  {r_['method']:<10} {r_['alpha']:>6.3f} {r_['score_a']:>7.2f} {r_['drift_orth_b']:>7.4f} "
                         f"{r_['capability_drop_pt']:>7.1f}pt {r_['ppl_worse_pct']:>+7.1f}% {r_['mix_increase'] * 100:>+5.2f}% "
                         f"{_f(r_['rate'], '6.2f')}" + ("  配布可" if r_["shippable"] else ""))
        lines.append("同じ A 側の射影の値の水準で比べた副作用(A射影 = A 側の射影の値)。報告には判定の一致率(外から見える指標)を主にする")
        return d, "\n".join(lines), 0
    _gate(s, "eval", {"doses": _doses(doses), "suite": suite, "when": when, "random_controls": random_controls,
                      "recipe": r, "keyed_metrics": keyed, "keyed_contrast": keyed_contrast,      # keyed の選択性の分も見積もる
                      "keyed_examples": keyed_examples or None, "keyed_pseudo": keyed_pseudo, "keyed_match": keyed_match,
                      "keyed_subset": keyed_subset}, yes, None)
    d = s.eval(r, doses=_doses(doses), suite=suite, when=when, judge_model=judge_model, judge_target=judge_target,
               name=name, random_controls=random_controls, reactivity=reactivity, head_controls=head_controls,
               control_pool=control_pool, direction_metrics=direction_metrics, dir_max_examples=dir_max_examples or None,
               keyed_contrast=keyed_contrast, keyed_metrics=keyed, keyed_examples=keyed_examples or None,
               keyed_match=keyed_match, keyed_pseudo=keyed_pseudo, keyed_subset=keyed_subset)
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
    dmx = d.get("direction_metrics")
    if dmx:                                              # CHANGE-06 F.1: 方向への射影の値と直交成分のずれ
        head += f" │ {'下がり':>7} {'A直交':>7} {'B直交':>7}" + (f" {'一致率':>6}" if dmx.get("behavior") else "") \
            + (f" {'null':>5}" if nc else "")
    kd = d.get("keyed")                                  # CHANGE-07 E: 結果の keyed(方式の比較・比較用の鍵、または skipped)
    ksel = any((row.get("keyed") or {}).get("selectivity") is not None for row in d["rows"])
    knull = ksel and any((row.get("keyed") or {}).get("null_pct") is not None for row in d["rows"])
    if ksel:                                             # 行の keyed: 目的の層の eval の選択性(比較用の鍵の対照があれば null)
        head += f" │ {'選択性':>7}" + (f" {'null':>5}" if knull else "")
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
        if dmx:
            line += (f" │ {_f(row.get('dir_drop_a'), '+7.2f')} {_f(row.get('drift_orth_a'), '7.4f')} {_f(row.get('drift_orth_b'), '7.4f')}"
                     + (f" {_f(row.get('rate'), '6.2f')}" if dmx.get("behavior") else ""))
            if nc:
                npct = row.get("rate_null_pct") if row.get("rate_null_pct") is not None else row.get("dir_null_pct")
                line += f" {_f(None if npct is None else npct * 100, '4.0f')}%"
        if ksel:
            kr = row.get("keyed") or {}
            line += f" │ {_f(kr.get('selectivity'), '+.3g'):>7}"
            if knull:
                kn = kr.get("null_pct")
                line += f" {kn * 100:>4.0f}%" if kn is not None else f" {'—':>5}"
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
    if dmx:
        text += (f"\n方向 {dmx['direction']}(L{dmx['layers'][-1]}): 下がり = A 側の射影の値の下がり方、A直交・B直交 = 残差の差の直交成分 ÷ 残差"
                 + ("、一致率 = A 側の判定(validate.match)" if dmx.get("behavior") else ""))
    if kd and "skipped" in kd:
        text += f"\nkeyed の eval の選択性は測っていません: {kd['skipped']}" + (f"\nヒント: {kd['hint']}" if kd.get("hint") else "")
    elif kd:
        text += "\n" + "\n".join(kd.get("text") or [])
    if ksel:
        kL = next(row["keyed"].get("layer") for row in d["rows"] if row.get("keyed"))
        text += (f"\n行の選択性 = L{kL} の eval の選択性(残差の r̂ 方向の変化の、書き込みたい位置 − 書き込みたくない位置。"
                 "書く向きにそろえた値で、レシピ全体を当てたもの。key-check の選択性とは別の量)"
                 + ("。null = 鍵を比較用の鍵に替えた対照の何割より大きいか" if knull else ""))
    for n_ in d.get("notes") or []:
        if kd and n_.startswith("keyed: "):              # keyed の注は上の keyed の表示に入っている
            continue
        text += f"\n注意: {n_}"
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
        who = f"E{cd['expert']} N{cd['expert_neuron']}" if cd.get("expert") is not None else f"N{cd['neuron']}"
        lines.append(f"  {i + 1:>3}. L{cd['layer']} {who}  cos {cd['cos']:+.3f}(押す向き {'+' if cd['sign'] > 0 else '-'}){extra}")
    if d["verify"]:
        lines.append(f"rollout({d['verify']['design']}): " + ", ".join(r["path"] for r in d["verify"]["runs"]))
        for ex in d["verify"]["examples"][:min(show, 10)]:
            lines.append(f"  L{ex['layer']} N{ex['neuron']} ({'+' if ex['sign'] > 0 else '-'}) {ex['prompt']}: "
                         f"{ex['text'][:160]!r}")
    return d, "\n".join(lines), 0


@gne_callback(direction_app, "direction")
def direction(c: Common, holder: dict,
              contrast: Optional[Path] = typer.Option(None, "--contrast", help="対照集合 YAML(reverse と同じ形。層選びには validate.match)"),
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
              same_reply: bool = typer.Option(False, "--same-reply", help="--positions response: B 側も A 側の返事を読む(同じ返事を両方の条件で読ませた差。読んでいる語の違いから来る成分が入らない)。既定は今までどおり、それぞれ自分の返事を読んだ差"),
              segments: Optional[str] = typer.Option(None, "--segments", help="--positions response: 返事の位置で分けた方向も作る(例 0,1-3,4-。位置 0 = 返事を書き始める直前)。区切り i の方向は NAME.seg<i> という名前で使える"),
              stack_layers: Optional[str] = typer.Option(None, "--stack-layers", help="--same-reply と --segments と一緒に: 区切りつきの方向を、これらの層(例 11,18,23)に重ねて書く前提の値にする(上の層は、下の層に足した後の残り)。1 トークンずつ読むので時間がかかる"),
              yes: bool = _YES):
    """対照集合から全層の方向 r_L(mean_A − mean_B)を作って保存し、--select auto で効果の大きい層を選ぶ(§6.17)。
    作った方向は L:dir:NAME:dose のユニットで足せる(gne steer / chat / eval / export-cvec)。
    gne direction score は、方向の分離度(層ごとの cohen_d)と介入の効き目を生成なしで測る(CHANGE-06 B)。
    gne direction key / key-check / keys / key-verify / key-experts は鍵つきの書き込み(keyed、CHANGE-07)の鍵を作る・確かめる(key-experts は MoE の expert:e の鍵の専門家を選ぶ材料)。
    key-from-behave は CHiPS のレシピを当てても振る舞いが残った質問(gne behave の結果)から keyed の対照集合の JSON を作る。
    key-causal / key-causal-verify は keyed の効く場所(層・書き込み先・α)を、生成なしのふるい分けと生成での確かめの 2 段で探す。"""
    if contrast is None:
        raise GneError("BAD_ARGS", "--contrast が要ります", "方向を作る: gne direction --contrast <対照集合>、"
                                                          "測る: gne direction score --name <方向> --contrast <対照集合>")
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
                    random_controls=random_controls, same_reply=same_reply, segments=segments, stack_layers=stack_layers)
    lines = [f"{d['path']}(A/B 各 {d['n_prompts']} 会話、行数 {d['n_rows']}、位置 {d['positions']}、{d['seconds']:.0f} 秒)",
             "作り方: " + ("同じ返事(A 側の返事)を A 側・B 側の条件の両方で読ませた差(--same-reply)" if d.get("same_reply")
                         else "それぞれ自分の条件の文を読んだ差(今までの作り方)"),
             *([("区切りつきの方向: " + "、".join(f"{x['name']}(位置 {x['positions']})" for x in d["segments"])
                 + (f"。層 {','.join(map(str, d['stack_layers']))} は、下の層に足した後の残り" if d.get("stack_layers") else ""))]
               if d.get("segments") else []),
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
                   system_b2: Optional[str] = typer.Option(None, "--system-b2", help="基準値に使う別の中立の system(既定は同じくらいの長さの中立の文)"),
                   layers: str = typer.Option("all", "--layers", help="all / 10-16 / 1,5,9-12"),
                   max_tokens: int = typer.Option(48, "--max-tokens", help="比べる返事の長さ(system_a で greedy に作る)"),
                   sort: str = typer.Option("z", "--sort", help="表の並び: z / trigger_mass / pattern_shift / out_shift / dir_cos(保存は z の順)"),
                   show: int = typer.Option(20, "--show")):
    """同じ会話をペルソナあり(system_a)と中立(system_b)で流し、ヘッドごとに注目と出力の差を測って順位づける。
    基準値は中立どうし(system_b と system_b2)の差の p95。"""
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
             "基準値(中立どうしの差の p95): " + "、".join(f"{k} {v:.3f}" for k, v in fl.items()),
             f"  {'順':>3} {'住所':>9} {'種類':>6} {'trigger':>8} {'pattern':>8} {'out':>6} {'dir_cos':>8} {'z':>6} {'null':>5}"]
    for i, x in enumerate(rows[:show], 1):
        lines.append(f"  {i:>3} {('L' + str(x['layer']) + ' H' + str(x['head'])):>9} {x['kind']:>6} "
                     f"{_metric(x['trigger_mass'], '+.3f'):>8} {_metric(x['pattern_shift'], '.3f'):>8} "
                     f"{_metric(x['out_shift'], '.2f'):>6} {_metric(x['dir_cos'], '+.3f'):>8} {_metric(x['z'], '.1f'):>6} "
                     f"{_metric(None if x['null_pct'] is None else x['null_pct'] * 100, '.0f'):>4}%")
    lines.append("読み方: trigger が大きいヘッドは「気づく係」、dir_cos が大きいヘッドは「ペルソナの方向を書く係」。"
                 "試す: gne steer --unit L:Hh:x0 / x2 --compare")
    return d, "\n".join(lines), 0


@gne_command(experts_app, "contrast", "experts contrast")
def experts_contrast(c: Common, holder: dict,
                     contrast: Path = typer.Option(..., "--contrast", help="対照集合 YAML"),
                     name: Optional[str] = typer.Option(None, "--name", help="保存名(既定は対照集合の name)"),
                     direction: Optional[str] = typer.Option(None, "--direction", help="contrib_cos に使う方向(既定は対照集合と同じ名前の方向、無ければ l_out の差)"),
                     system_b2: Optional[str] = typer.Option(None, "--system-b2", help="基準値に使う別の中立の system"),
                     layers: str = typer.Option("all", "--layers", help="all / 10-16 / 1,5,9-12(MoE の層だけ測る)"),
                     max_tokens: int = typer.Option(48, "--max-tokens", help="比べる返事の長さ(system_a で greedy に作る)"),
                     sort: str = typer.Option("z", "--sort", help="表の並び: z / usage_shift / weight_shift / contrib_cos / contrib_norm"),
                     show: int = typer.Option(20, "--show")):
    """同じ返事をペルソナあり(system_a)と中立(system_b)で読ませ、MoE の層の専門家ごとに選ばれ方と寄与の差を測って順位づける。
    基準値は中立どうし(system_b と system_b2)の差の p95。"""
    from .experts import METRICS, load_expert_candidates
    if sort not in ("z", *METRICS):
        raise GneError("BAD_ARGS", f"--sort は z / {' / '.join(METRICS)} のどれかです")
    s = holder["session"] = c.session()
    d = s.experts_contrast(contrast, name=name, direction=direction, system_b2=system_b2, layers=parse_layers(layers),
                           max_tokens=max_tokens)
    rows = d["top"] if sort == "z" else sorted(load_expert_candidates(s, d["name"]),
                                               key=lambda x: -(abs(x[sort]) if sort != "contrib_norm" else x[sort]))
    lines = [f"{d['path']}({d['n_experts']} 専門家、{d['n_prompts']} プロンプト、方向 {d['direction'] or 'l_out の差'}、{d['seconds']:.0f} 秒)",
             "基準値(中立どうしの差の p95): " + "、".join(f"{k} {v:.3f}" for k, v in d["floors"].items()),
             f"  {'順':>3} {'住所':>8} {'選択率 A→B':>12} {'usage':>7} {'weight':>7} {'cos':>6} {'norm':>6} {'z':>5} {'null':>5}"]
    for i, x in enumerate(rows[:show], 1):
        lines.append(f"  {i:>3} {('L' + str(x['layer']) + ' E' + str(x['expert'])):>8} {x['rate_a']:5.2f}→{x['rate_b']:<5.2f} "
                     f"{x['usage_shift']:+7.3f} {x['weight_shift']:+7.3f} {x['contrib_cos']:+6.2f} {x['contrib_norm']:6.2f} "
                     f"{x['z']:5.1f} {x['null_pct'] * 100:4.0f}%")
    lines.append("読み方: usage が大きい専門家は「ペルソナのときに呼ばれる」、cos が大きい専門家は「ペルソナの方向を書く」。"
                 "試す: gne steer --unit L:Ee:x0(消す)/ L:Ee:x2(強める)--compare")
    return d, "\n".join(lines), 0


@gne_command(experts_app, "sweep", "experts sweep")
def experts_sweep(c: Common, holder: dict, layer: int = typer.Option(..., "--layer", "-l", help="MoE の層"),
                  experts: Optional[str] = typer.Option(None, "--experts", help="試す専門家(例 0,3,7。既定は全部)"),
                  ks: str = typer.Option("0,2", "--ks", help="専門家の出力に掛ける倍率(0 = 消す、2 = 強める)"),
                  design: str = typer.Option("quick", "--design", help="configs/designs.yaml の設計(プロンプトだけを使う)"),
                  prompt_set: str = typer.Option("en", "--prompt-set"),
                  max_tokens: int = typer.Option(48, "--max-tokens"),
                  run: Optional[str] = typer.Option(None, "--run", help="記録の名前(既定 L<層>-experts-<設計>)"),
                  show: int = typer.Option(15, "--show"), yes: bool = _YES):
    """層の各専門家を消す・強めるスケールパッチで生成し、語の出方で順位づける(1 専門家ごとに読み直す)。
    帰無は同じ層の全専門家の分布(null = mass がその専門家より小さい専門家の割合)。"""
    s = holder["session"] = c.session()
    exs = [int(x) for x in experts.split(",")] if experts else None
    kk = tuple(float(x) for x in ks.split(","))
    n = len(exs) if exs else s.weights.moe_layer(layer).n_expert
    _gate(s, "experts sweep", {"n": n * len(kk), "max_tokens": max_tokens}, yes, None)
    d = s.experts_sweep(layer, ks=kk, design=design, prompt_set=prompt_set, max_tokens=max_tokens, experts=exs, run=run)
    lines = [f"{d['path']}(L{layer}、専門家 {d['n_experts']} 個 × 倍率 {d['ks']}、読み直し平均 {d['reload_s']['mean']:.2f} 秒、{d['seconds']:.0f} 秒)"]
    for u in d["top"][:show]:
        lines.append(f"  {u['rank']:>3}. L{layer} E{u['expert']:<3} mass {u['mass']:5.1f}  null {u['null_pct'] * 100:3.0f}%  "
                     + " ".join(w[0] for w in u["words"][:8]))
    lines.append("試す: gne steer --unit L:Ee:x0 / x2 --compare、専門家の調書は gne experts show L E")
    return d, "\n".join(lines), 0


@gne_command(experts_app, "show", "experts show")
def experts_show(c: Common, holder: dict, layer: int = typer.Argument(..., help="層 L"),
                 expert: int = typer.Argument(..., help="専門家の番号"),
                 top: int = typer.Option(20, "--top", help="ニューロンの上位の数")):
    """専門家 1 個の調書(使用統計・選ばれやすいトークン・対照分析での順位・ニューロンの上位)。"""
    s = holder["session"] = c.session()
    d = s.expert(layer, expert, top=top)
    lines = [f"L{layer} E{expert}  専門家 {d['n_expert']} 個中 {d['n_expert_used']} 個を選ぶ({d['routing']})、幅 {d['n_ff']}"
             + (f"、ルーターのバイアス {d['router_bias']:+.4f}" if d["router_bias"] is not None else "")]
    u = d["usage"]
    if u:
        lines.append(f"使用: 選択率 {u['rate']:.1%}({u['n_routed']}/{u['n_tokens']} トークン{'' if u['reliable'] else '、少ないので分位点は目安'})、"
                     f"重みの平均 {u['weight_mean']:.3f}")
        lines.append("よく選ぶトークン: " + " ".join(f"{x['token']!r}×{x['count']}" for x in u["top_tokens"]))
    else:
        lines.append("使用統計なし(gne stats act --layers で作れます)")
    for x in d["contrasts"]:
        lines.append(f"対照分析 {x['name']}: #{x['rank']}/{x['of']}  選択率 {x['rate_a']:.2f}→{x['rate_b']:.2f}  "
                     f"usage {x['usage_shift']:+.3f}  cos {x['contrib_cos']:+.2f}  z {x['z']:.1f}")
    lines.append("ニューロン(書き込みの大きさ q999 × ‖W_down[:, n]‖ の順): " +
                 ", ".join(f"N{x['neuron']}" for x in d["neurons"][:12]))
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


@gne_command(heads_app, "listen", "heads listen")
def heads_listen(c: Common, holder: dict, user: str = typer.Option(..., "--user", "-u", help="ユーザーの発話"),
                 heads: str = typer.Option(..., "--heads", help="見るヘッド(例 14:H0-3,16:H2、L:H* で層の全ヘッド)"),
                 system: Optional[str] = typer.Option(None, "--system", help="system(@FILE でファイルから)"),
                 reply: Optional[str] = typer.Option(None, "--reply", help="返事(省くと greedy に作る)"),
                 max_tokens: int = typer.Option(64, "--max-tokens", help="返事を作るときの長さ"),
                 direction: Optional[str] = typer.Option(None, "--direction", help="寄与を測る方向(保存済みの名前。その層の分を使う)"),
                 top: int = typer.Option(20, "--top")):
    """返事のトークンが、選んだヘッドで文脈のどのトークンに注目し、各トークンが方向にどれだけ寄与したか(注目 × ゲート × OV の分解)。"""
    s = holder["session"] = c.session()
    if system and system.startswith("@"):
        system = Path(system[1:]).read_text(encoding="utf-8")
    d = s.heads_listen(user, heads=heads, system=system, reply=reply, max_tokens=max_tokens, direction=direction, top=top)
    hs = ",".join(f"L{x['layer']}H{x['head']}" for x in d["heads"])
    lines = [f"{d['n_tokens']} トークン(返事は {d['reply_start']} から): {d['reply'][:160]!r}", f"ヘッド {hs}"]
    for k, x in d["by_role"].items():
        lines.append(f"  {k:<8} 注目 {x['attn']:.3f}" + ("" if x["contrib"] is None else f"  寄与 {x['contrib']:+.4f}"))
    for L, x in (d["layers"] or {}).items():
        lines.append(f"L{L}: 層全体の寄与 {x['contrib_total']:+.4f}(ヘッドごと " + " ".join(f"{v:+.3f}" for v in x["contrib_by_head"])
                     + f")、分解の誤差 {x['check_rel_err'] if x['check_rel_err'] is not None else '—'}")
    for title, key in (("寄与の上位", "top_contrib"), ("注目の上位", "top_attn")):
        if d[key]:
            lines.append(title + ": " + " ".join(f"{x['token']!r}({x['value']:+.3f})" for x in d[key]))
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
              openai: bool = typer.Option(False, "--openai", help="OpenAI 互換の /v1/chat/completions も出す(レシピと when は gne で指定、§6.15)"),
              mmproj: Optional[Path] = typer.Option(None, "--mmproj", envvar="GNE_MMPROJ",
                                                    help="音声の入力(input_audio)に使う mmproj(既定はモデルと同じフォルダの mmproj*.gguf)")):
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
              openai=openai, mmproj=mmproj or c.mmproj)
    except GneError as e:
        print(f"エラー[{e.code}]: {e.message}", file=sys.stderr)
        raise typer.Exit(2) from e



# --- CHANGE-06: ヘッドの方向成分の射影 ----------------------------------------------------------------------

def _ints(text: Optional[str]) -> Optional[list[int]]:
    if text is None or text == "":
        return None
    try:
        return [int(x) for x in text.split(",") if x.strip()]
    except ValueError as e:
        raise GneError("BAD_ARGS", f"{text!r} はカンマ区切りの整数です(例 38,50,62)") from e


def _f(x, fmt: str) -> str:
    return "—" if x is None else format(x, fmt)


@gne_command(direction_app, "score", "direction score")
def direction_score(c: Common, holder: dict,
                    name: str = typer.Option(..., "--name", help="方向の名前(gne direction で作ったもの)"),
                    contrast: Optional[Path] = typer.Option(None, "--contrast", help="対照集合 YAML(既定は方向を作った対照集合)"),
                    layers: Optional[str] = typer.Option(None, "--layers", help="測る層(例 38,50,62。既定は全層で、介入が無ければ cohen_d 最大の層を基準層として記録する)"),
                    positions: str = typer.Option("last", "--positions", help="測定位置: last / last:K(最後の K トークン)/ assistant(生成プロンプトのトークン)"),
                    recipe: Optional[Path] = _RECIPE, unit: Optional[list[str]] = _UNIT,
                    side: str = typer.Option("both", "--side", help="both / A / B"),
                    subset: str = typer.Option("all", "--subset", help="all / even(測定用)/ odd(確認用)"),
                    set_base: Optional[int] = typer.Option(None, "--set-base", help="基準層を手で決めて記録する"),
                    record: Optional[bool] = typer.Option(None, "--record/--no-record", help="cohen_d が最大の層を基準層として記録する(既定はまだ記録が無いときだけ)")):
    """速い指標(CHANGE-06 B): 生成せず、対照集合の A 側・B 側をプロンプトの最後まで読ませて、層ごとに方向への射影の値
    (残差 · r̂_L)を測る。--unit / --recipe を渡すと、その介入での射影の値の変化と、残差の差の方向成分・直交成分(drift_par / drift_orth)も出す。"""
    s = holder["session"] = c.session()
    d = s.direction_score(name, contrast=contrast, layers=_ints(layers), positions=positions, units=unit,
                          recipe=recipe, side=side, subset=subset, set_base=set_base, record_base=record)
    iv = d["intervention"] is not None
    sides = [x for x in "AB" if x in d["n"]]
    head = f"  {'層':>4} {'d':>6} " + " ".join(f"{x + ' 平均':>9} {'sd':>6}" for x in sides) + f" {'cos基準':>7}"
    if iv:
        head += " │ " + " ".join(f"{x + ' 下がり':>8} {x + ' 直交':>7}" for x in sides)
    lines = [f"方向 {d['direction']}(A {d['n'].get('A', 0)} / B {d['n'].get('B', 0)} 例、位置 {d['positions']}、"
             f"{d['subset']}、{d['seconds']:.1f} 秒)", head]
    for r in d["layers"]:
        line = f"  {r['layer']:>4} {_f(r['cohen_d'], '6.2f')} " + " ".join(
            f"{r[x]['mean']:>9.2f} {r[x]['sd']:>6.2f}" for x in sides) + f" {_f(r['cos_base'], '+7.2f')}"
        if iv:
            line += " │ " + " ".join(f"{r['drop'][x]:>+8.2f} {r['drift'][x]['orth']:>7.4f}" for x in sides)
        if r["layer"] == d["base_layer"]:
            line += "  ← 基準層" + ("(記録)" if d["base_recorded"] else "")
        lines.append(line)
    if not iv:
        lines.append("次: gne heads causal --name " + d["direction"] + " で因果で効くヘッドを探す")
    return d, "\n".join(lines), 0


# --- CHANGE-07: 鍵つきの書き込み(keyed write)--------------------------------------------------------------

_KEY_POS = ("書き込みたい位置: response / assistant / last / last:K / before:<正規表現>(推論時と同じ文脈の返事で、一致する文字を"
            "書く直前。書き込みたい位置の取り方 (b))。既定は対照集合の positions、無ければ response")
_KEY_POS_B = "書き込みたくない位置(既定: before:<re> なら not-before:<re>、それ以外は書き込みたい位置と同じ指定)"
_KEY_REPLIES_A = ("書き込みたい側の返事を読ませるファイル(YAML / JSON、質問と同じ長さのリストか {質問: 返事})。"
                  "before:<re> のときは書き込みたい位置と書き込みたくない位置が同じ返事にあるので 1 つだけ渡す")
_KEY_REPLIES_B = "書き込みたくない側の返事を読ませるファイル"
_KEY_IV = "この介入を当てた状態で捕捉する(レシピ YAML・保存したレシピの名前、またはユニットの書き方。複数可)"
_REPLY_KIND = {"given": "与えた", "generated": "生成"}


def _iv_arg(iv: Optional[list[str]]):
    """--iv → keyed_build.intervention に渡す形: レシピのパスか保存したレシピの名前 1 つ、またはユニットの書き方のリスト。"""
    if not iv:
        return None
    if len(iv) == 1 and (Path(iv[0]).exists() or ":" not in iv[0]):
        return iv[0]
    return list(iv)


def _floats(text: str, opt: str) -> list[float]:
    try:
        return [float(x) for x in text.split(",") if x.strip()]
    except ValueError as e:
        raise GneError("BAD_ARGS", f"{opt} {text!r} はカンマ区切りの数です(例 1,2)") from e


def _fc(x, spec: str) -> str:
    """表の列: 数を spec(例 '>8.3f')で書き、None は同じ幅の —。"""
    return format("—", spec.split(".")[0]) if x is None else format(x, spec)


def _target_who(target: str) -> str:
    """鍵の書き込み先(ffn / attn / shared / expert:e)→ ユニットの記法の書き方(ffn / attn / S / Ee)。"""
    if target.startswith("expert:"):
        return f"E{target.split(':', 1)[1]}"
    return _keyed_who(target)


def _replies_text(replies: dict | None) -> str:
    if not replies:
        return ""
    return "、返事 " + "・".join(f"{g} {_REPLY_KIND.get(v, 'なし')}" for g, v in replies.items())


@gne_command(direction_app, "key", "direction key")
def direction_key(c: Common, holder: dict,
                  contrast: Path = typer.Option(..., "--contrast", help="対照集合 YAML(system_a・system_b と質問、任意で dont_touch・positions・replies。例 configs/contrast/zundamon_endings.yaml)"),
                  layers: str = typer.Option(..., "--layers", help="鍵を作る層(必須。例 8,14,20 / 10-16。Q の基底は層ごとに作るので全層は既定にしない。最終層と MTP の層は不可)"),
                  name: Optional[str] = typer.Option(None, "--name", help="鍵の名前(既定は対照集合の name)。<root>/keys/<model_id>/<名前>.npz に保存"),
                  positions: Optional[str] = typer.Option(None, "--positions", help=_KEY_POS),
                  positions_b: Optional[str] = typer.Option(None, "--positions-b", help=_KEY_POS_B),
                  skip: Optional[int] = typer.Option(None, "--skip", help="返事の先頭から落とすトークン数(既定は対照集合の skip、無ければ 0)"),
                  max_tokens: int = typer.Option(96, "--max-tokens", help="返事を生成するときの長さ"),
                  q_mode: str = typer.Option("examples", "--q-mode", help="Q(書き込みたくない位置と dont_touch の活性が張る部分空間の基底)を張る行: examples(例ごとの平均、既定)/ means(群ごとの平均。よく効くが選択性は粗い)/ samples(トークンの活性。選択性は高いが重い)"),
                  q_energy: float = typer.Option(0.99, "--q-energy", help="Q の打ち切り(特異値の二乗の累積の割合)"),
                  q_max_rank: Optional[int] = typer.Option(None, "--q-max-rank", help="Q の階数の上限"),
                  q_max_samples: int = typer.Option(4096, "--q-max-samples", help="--q-mode samples で使うトークンの上限(超えたら間引く)"),
                  subset: str = typer.Option("even", "--subset", help="使う質問: even(既定。key-check は odd で確かめる)/ odd / all"),
                  pseudo: int = typer.Option(16, "--pseudo", help="比較用の鍵(書き込みたい側・書き込みたくない側のラベルを例ごとにシャッフルして、同じ手順で作った鍵)の本数。"
                                                                "シャッフルは --seed で決まる"),
                  replies_a: Optional[Path] = typer.Option(None, "--replies-a", help=_KEY_REPLIES_A),
                  replies_b: Optional[Path] = typer.Option(None, "--replies-b", help=_KEY_REPLIES_B),
                  iv: Optional[list[str]] = typer.Option(None, "--iv", help=_KEY_IV),
                  target: str = typer.Option("ffn", "--target", help="書き込み先: mlpin(計算で作るニューロン用。dense の MLP の入力 = 正規化の後の空間。返事の位置の鍵は --positions pos:1-3 のように書く)/ ffn(dense の ffn_down の入力)/ attn(attn_output・ssm_out の入力 = ヘッド出力の連結)/ shared(MoE の層の shared expert の ffn_down_shexp の入力)/ expert:e(ffn_down_exps の専門家 e の入力。専門家 e が選ばれた位置だけで作る)"),
                  dont_touch_start: Optional[bool] = typer.Option(None, "--dont-touch-start/--no-dont-touch-start", help="dont_touch の各群の質問の、返事を書き始める直前の位置も Q に入れる(既定は対照集合の dont_touch_start、無ければ入れる)。書き込みたくない側の質問のこの位置は入れない")):
    """keyed write の鍵 κ_L を作る(CHANGE-07)。書き込みたい位置の活性の平均 μ_L を、書き込みたくない位置と dont_touch の
    活性が張る空間 Q_L に直交させ、κ_Lᵀμ_L = 1 に正規化して保存する(方向とは別のファイル)。鍵は無介入で作る(--iv は重ねるときだけ)。
    作ったら gne direction key-check で、作成に使っていない例(既定 subset odd)で分離を確かめる。"""
    Ls = parse_layers(layers)
    if not Ls:
        raise GneError("BAD_ARGS", "--layers に層を指定してください(all は使えません)", "例 --layers 8,14,20")
    s = holder["session"] = c.session()
    d = s.direction_key(contrast, name=name, layers=Ls, positions=positions, positions_b=positions_b, skip=skip,
                        max_tokens=max_tokens, q_mode=q_mode, q_energy=q_energy, q_max_rank=q_max_rank,
                        q_max_samples=q_max_samples, subset=subset, pseudo=pseudo, replies_a=replies_a,
                        replies_b=replies_b, iv=_iv_arg(iv), target=target, seed=c.seed, dont_touch_start=dont_touch_start,
                        progress=s.progress)
    d.pop("warnings", None)                        # 警告はエンベロープの warnings に入っている
    n = d["counts"]
    moe = any("rows_a" in r for r in d["layers"])            # 書き込み先 shared・expert:e の鍵(層ごとの行の数を出す)
    how = "推論時と同じ文脈の返事" if d["mode"] == "same" else "system・recipe の文脈"
    lines = [f"{d['path']}(鍵 {d['name']}、書き込み先 {d['target']}、書き込みたい位置は{how}(mode {d['mode']})、{d['seconds']:.1f} 秒)",
             f"  位置: 書き込みたい位置 {d['positions']}・書き込みたくない位置 {d['positions_b']}、skip {d['skip']}、subset {d['subset']}、"
             f"q_mode {d['q_mode']}、比較用の鍵 {d['n_pseudo']} 本",
             f"  例: 書き込みたい側 {n['a']} 例({n['rows_a']} 位置、書き込みたい位置のある例 {d['matched']['A']}/{d['matched']['of']})・"
             f"書き込みたくない側 {n['b']} 例({n['rows_b']} 位置)" + (f"・dont_touch {n['dt']} 例({n['rows_dt']} 位置)" if n.get("dt") else "")
             + _replies_text(d.get("replies"))]
    ds_ = d.get("dont_touch_start")
    if ds_ and ds_.get("on") and ds_.get("groups"):          # dont_touch の質問の、返事を書き始める直前の位置を Q に入れた行
        lines.append(f"  dont_touch の質問の、返事を書き始める直前の位置: {ds_['rows']} 行を Q に入れました("
                     + "・".join(f"{g} {r}" for g, r in ds_["groups"].items()) + ")")
    elif ds_ and not ds_.get("on") and d.get("dont_touch"):
        lines.append("  dont_touch の質問の、返事を書き始める直前の位置: Q に入れていません(--no-dont-touch-start か、対照集合の dont_touch_start: false)")
    rs = d.get("reply_start")
    if rs:                                                   # mode same: 書き込みたい位置のうち返事の冒頭にあるもの
        lines.append(f"  返事の冒頭: 書き込みたい位置の {rs['rows']} / {rs['rows_a']} が返事の最初の {rs['k']} トークンを決める位置"
                     + (f"(L{rs['layer']})" if rs.get("layer") is not None else "")
                     + f"、μ に占める重み {rs['share'] * 100:.0f}%"
                     + (f"(返事を書き始める直前の位置を Q に入れた dont_touch: {', '.join(rs['dont_touch_starts'])})" if rs["dont_touch_starts"] else ""))
    lines += [f"  {'層':>4} {'kn':>6} {'増幅率':>7} {'q_rank':>6} {'作成時AUC':>9}"
             + (f" {'rows_a':>6} {'rows_b':>6} {'gate_a':>7}" if moe else "")]
    for r in d["layers"]:
        lines.append(f"  {r['layer']:>4} {r['kn']:>6.3f} {r['gain']:>7.2f} {r['q_rank']:>6} {_fc(r['build_auc'], '>9.3f')}"
                     + (f" {r['rows_a']:>6} {r['rows_b']:>6} {_fc(r.get('gate_a'), '>7.3f')}" if moe else "")
                     + ("  ← 弱い(kn < 0.05、書き込みたくない位置に漏れやすい)" if r["weak"] else ""))
    lines.append("  kn = ‖k‖/‖μ‖(1 に近いほど、書き込みたい位置の平均が書き込みたくない位置の部分空間の外にある)、増幅率 = 1/kn"
                 + ("。rows_a・rows_b = この層で書き込み先の行があった書き込みたい位置・書き込みたくない位置の数(専門家は"
                    "選ばれた位置だけ)、gate_a = 書き込みたい位置でのゲート・ルーターの重みの平均(LoRA の B をこれで割る)" if moe else ""))
    for x in d["dropped"]:
        lines.append(f"  L{x['layer']}: 外しました({x['error']})")
    lines += [f"注: {t}" for t in d["notes"]]
    if d.get("samples"):
        lines.append(f"返事の例: {d['samples'][0][:120]!r}")
    lines.append(f"次: gne direction key-check --key {d['name']} --contrast {contrast}(既定で subset odd = 作成に使っていない例)")
    return d, "\n".join(lines), 0


def _start_rho_text(sr: dict | None) -> str:
    """key-check の層の start_rho を 1 行に: 群ごとの、返事の冒頭の位置 0(返事を書き始める直前の位置)での ρ の中央値。"""
    if not sr:
        return ""
    items = ([("書き込みたくない側", sr["B"])] if sr.get("B") else []) + list((sr.get("dont_touch") or {}).items())
    out = [f"{g} {_f(v['median'][0], '+.2f')}" for g, v in items if v.get("median") and v["median"][0] is not None]
    return "・".join(out)


@gne_command(direction_app, "key-check", "direction key-check")
def direction_key_check(c: Common, holder: dict,
                        key: str = typer.Option(..., "--key", help="鍵の名前(NAME~j で比較用の鍵 j)"),
                        contrast: Optional[Path] = typer.Option(None, "--contrast", help="確かめる対照集合(作成と同じなら --subset odd が保留集合)。省くと鍵に保存した作成時の例で測る(甘い値)"),
                        subset: str = typer.Option("odd", "--subset", help="使う質問: odd(既定)/ even / all。作成と重なると held_out: false"),
                        layers: Optional[str] = typer.Option(None, "--layers", help="測る層(既定は鍵の全層)"),
                        positions: Optional[str] = typer.Option(None, "--positions", help="書き込みたい位置(既定は鍵を作ったときの値)"),
                        positions_b: Optional[str] = typer.Option(None, "--positions-b", help="書き込みたくない位置(既定は鍵を作ったときの値)"),
                        skip: Optional[int] = typer.Option(None, "--skip", help="返事の先頭から落とすトークン数(既定は鍵を作ったときの値)"),
                        max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="返事を生成するときの長さ(既定は鍵を作ったときの値)"),
                        replies_a: Optional[Path] = typer.Option(None, "--replies-a", help=_KEY_REPLIES_A),
                        replies_b: Optional[Path] = typer.Option(None, "--replies-b", help=_KEY_REPLIES_B),
                        iv: Optional[list[str]] = typer.Option(None, "--iv", help=_KEY_IV),
                        strict: bool = typer.Option(False, "--strict", help="作成に使った例と重なっていたら測らずにエラー(BAD_ARGS)")):
    """鍵の分離の確認(CHANGE-07 C、生成なしで済む位置なら生成なし)。各位置で鍵の反応の強さ ρ = κᵀx を測り、層ごとに
    書き込みたい位置 対 書き込みたくない位置の AUC(例単位とトークン単位)、key-check の選択性(例ごとの ρ の平均の、
    書き込みたい位置 − 書き込みたくない位置)と比較用の鍵(ラベルをシャッフルして同じ手順で作った鍵)の p95、
    dont_touch の ρ の中央値を出す。作成に使った例と重なると held_out: false(甘い値)。判定の目安: 保留集合で選択性が
    比較用の鍵の p95 を超えること。"""
    s = holder["session"] = c.session()
    d = s.direction_key_check(key, contrast=contrast, subset=subset, layers=parse_layers(layers) if layers else None,
                              positions=positions, positions_b=positions_b, skip=skip, max_tokens=max_tokens,
                              replies_a=replies_a, replies_b=replies_b, iv=_iv_arg(iv), strict=strict,
                              progress=s.progress)
    d.pop("warnings", None)
    src = f"対照集合 {d['contrast']}・subset {d['subset']}" if d["contrast"] else "鍵に保存した作成時の例"
    lines = [f"鍵 {d['key']}(書き込み先 {d['target']}、{src}、mode {d['mode']}、書き込みたい位置 {d['positions']}・"
             f"書き込みたくない位置 {d['positions_b']}、{d['seconds']:.1f} 秒)"]
    ov = d["overlap"]
    if d["held_out"]:
        lines.append(f"保留集合: はい(作成に使っていない例 {ov['n_checked']} 個)")
    else:
        why = ([f"同じ対照集合で subset が重なる(作成 {ov.get('built_subset')}・確認 {ov.get('subset')})"] if ov.get("subset_overlap") else []) \
            + ([f"作成と同じ (system, 質問) が {ov['hash_overlap']} 個"] if ov.get("hash_overlap") else [])
        lines.append("保留集合: いいえ(" + "、".join(why or ["作成に使った例"]) + ")。AUC・選択性は甘い値です")
    dts = d.get("dont_touch_start") or {}
    has_dt = any(g.startswith("dt:") for g in d.get("counts") or {})
    if dts.get("on"):
        grp = "・".join(f"{g} {n} 位置" for g, n in dts["groups"].items())
        lines.append(f"dont_touch の返事を書き始める直前の位置: 鍵の作成と同じく測りました({grp or 'なし'}。返事の冒頭の ρ の位置 0 に入ります)")
    elif has_dt and dts.get("recorded") is False:
        lines.append("dont_touch の返事を書き始める直前の位置: 記録の無い古い鍵なので、dont_touch の群の positions に入るときだけ測りました")
    lines.append(f"  {'層':>4} {'AUC':>6} {'AUC(tok)':>8} {'選択性':>7} {'比較用p95':>7} {'比較用AUCp95':>10} {'2分割AUC':>7}  判定")
    for r in d["layers"]:
        pa = (r["pseudo_auc_token"] if r["auc_token"] is not None else r["pseudo_auc"])["p95"]
        mark = {True: "○", False: "×"}.get(r["beats_pseudo"], "—")
        line = (f"  {r['layer']:>4} {_fc(r['auc'], '>6.3f')} {_fc(r['auc_token'], '>8.3f')} {_fc(r['selectivity'], '>7.3f')} "
                f"{_fc(r['pseudo_selectivity']['p95'], '>7.3f')} {_fc(pa, '>10.3f')} {_fc(r['b_split_auc'], '>7.3f')}  {mark}")
        dt = "・".join(f"{g} {_f(v, '+.3f')}" for g, v in (r.get("dont_touch_median") or {}).items())
        if dt:
            line += f"  dont_touch の ρ 中央値 {dt}"
        st = _start_rho_text(r.get("start_rho"))
        if st:
            line += f"  返事の冒頭の ρ {st}"
        if r["layer"] == d["best_layer"]:
            line += "  ← 最良"
        lines.append(line)
    lines.append("  選択性 = key-check の選択性(例ごとの鍵の反応の強さ ρ の平均の、書き込みたい位置 − 書き込みたくない位置)。"
                 "比較用p95・比較用AUCp95 = 比較用の鍵の選択性・AUC の p95(AUC はトークン単位の AUC があればトークン単位)。"
                 "判定 ○ = 選択性が比較用の鍵の p95 を超え、AUC > 0.5。2分割AUC = 書き込みたくない位置を 2 つに分けて作った鍵の"
                 "AUC(参考値)。返事の冒頭の ρ = 書き込みたくない側・dont_touch の、返事を書き始める直前の位置での ρ の中央値"
                 "(0.4 以上で警告。dont_touch のこの位置は、鍵の作成で Q に入れた鍵なら、返事の冒頭が positions の正規表現に"
                 "一致しても測る)")
    ab = d["all_beat_pseudo"]
    lines.append("全層で比較用の鍵を超えた: " + {True: "はい", False: "いいえ"}.get(ab, "判定なし(比較用の鍵か書き込みたくない位置が無い)"))
    lines += [f"注: {t}" for t in d["notes"]]
    if not d["held_out"]:
        lines.append("次: 作成に使っていない例(--subset odd か別の対照集合)で測り直してください")
    elif d["best_layer"] is not None and ab is not False:
        lines.append(f"次: gne steer --unit {d['best_layer']}:{_target_who(d['target'])}:keyed:a1@<方向>#{d['key']} "
                     "-p \"…\" --compare で生成を確かめる(α は raw、1 で観測した A − B の差をそのまま書く)")
    return d, "\n".join(lines), 0


@gne_command(direction_app, "keys", "direction keys")
def direction_keys(c: Common, holder: dict):
    """保存した鍵の一覧(<root>/keys/<model_id>/、CHANGE-07)。層・書き込み先・kn・作成の条件。"""
    s = holder["session"] = c.session()
    ks = s.keys()
    lines = [f"鍵 {len(ks)} 個" + ("" if ks else "(gne direction key --contrast <対照集合> --layers <層> で作ります)")]
    for k in ks:
        if "error" in k:
            lines.append(f"  {k['name']}: 読めません({k['error']})")
            continue
        kn = " ".join(f"L{L}:{v:.2f}" for L, v in k["kn"].items())
        lines.append(f"  {k['name']}  {k['target']}  q_mode {k['q_mode']}  位置 {k['positions']}  subset {k['subset']}  "
                     f"比較用の鍵 {k['n_pseudo']}  kn {kn}")
    return {"keys": ks}, "\n".join(lines), 0


@gne_command(direction_app, "key-experts", "direction key-experts")
def direction_key_experts(c: Common, holder: dict,
                          contrast: Path = typer.Option(..., "--contrast", help="対照集合 YAML(鍵を作るときと同じもの。例 configs/contrast/zundamon_endings.yaml)"),
                          layers: str = typer.Option(..., "--layers", help="調べる MoE の層(必須。例 6,12,18 / 10-16)"),
                          positions: Optional[str] = typer.Option(None, "--positions", help=_KEY_POS),
                          positions_b: Optional[str] = typer.Option(None, "--positions-b", help=_KEY_POS_B),
                          skip: Optional[int] = typer.Option(None, "--skip", help="返事の先頭から落とすトークン数(既定は対照集合の skip、無ければ 0)"),
                          max_tokens: int = typer.Option(96, "--max-tokens", help="返事を生成するときの長さ"),
                          subset: str = typer.Option("even", "--subset", help="使う質問: even(既定。鍵の作成と同じ)/ odd / all"),
                          replies_a: Optional[Path] = typer.Option(None, "--replies-a", help=_KEY_REPLIES_A),
                          replies_b: Optional[Path] = typer.Option(None, "--replies-b", help=_KEY_REPLIES_B),
                          iv: Optional[list[str]] = typer.Option(None, "--iv", help=_KEY_IV),
                          top: int = typer.Option(8, "--top", help="層ごとに出す専門家の数(書き込みたい位置で選ばれた割合の大きい順)")):
    """expert:e の鍵(gne direction key --target expert:e)の専門家を選ぶ材料(CHANGE-07 MoE)。鍵と同じ対照集合・位置・返事で、
    書き込みたい位置・書き込みたくない位置で各専門家が選ばれた割合(rate_a・rate_b、例ごとの割合の平均)と、選ばれたときの
    ルーターの重みの平均(weight_a、鍵の gate_a の見込み)を層ごとに出す。専門家の鍵は書き込みたい位置で専門家 e が選ばれた
    行だけでできるので、rate_a の
    大きい専門家ほど行が多く(rows_a)、鍵が例の本文を拾いにくい。MoE でない層は BAD_RECIPE。"""
    Ls = parse_layers(layers)
    if not Ls:
        raise GneError("BAD_ARGS", "--layers に層を指定してください(all は使えません)", "例 --layers 6,12,18")
    s = holder["session"] = c.session()
    d = s.direction_key_experts(contrast, layers=Ls, positions=positions, positions_b=positions_b, skip=skip,
                                max_tokens=max_tokens, subset=subset, replies_a=replies_a, replies_b=replies_b,
                                iv=_iv_arg(iv), top=top, progress=s.progress)
    lines = [f"専門家の選ばれ方(対照集合 {d['contrast']}、mode {d['mode']}、書き込みたい位置 {d['positions']}・"
             f"書き込みたくない位置 {d['positions_b']}、"
             f"subset {d['subset']}{_replies_text(d.get('replies'))}、{d['seconds']:.1f} 秒)"]
    for lay in d["layers"]:
        lines.append(f"  L{lay['layer']}({lay['n_expert']} 専門家から {lay['n_used']}、書き込みたい位置 {lay['positions_a']} 個・"
                     f"書き込みたくない位置 {lay['positions_b']} 個、どれも同じだけ選ばれるなら {lay['base_rate']:.2f})")
        lines.append(f"    {'専門家':>6} {'rate_a':>6} {'rate_b':>6} {'差':>6} {'lift':>6} {'重み':>6} {'rows_a見込み':>9}")
        for r in lay["top"]:
            dt = "・".join(f"{g} {v:.2f}" for g, v in (r.get("rate_dt") or {}).items())
            lines.append(f"    {'E' + str(r['expert']):>6} {r['rate_a']:>6.2f} {r['rate_b']:>6.2f} {r['diff']:>+6.2f} "
                         f"{_fc(r['lift'], '>6.2f')} {_fc(r['weight_a'], '>6.3f')} {r['rows_a']:>9.1f}"
                         + (f"  dont_touch {dt}" if dt else ""))
    lines += [f"注: {t}" for t in d["notes"]]
    best = [(lay["layer"], lay["top"][0]["expert"]) for lay in d["layers"] if lay["top"]]
    if best:
        L, e = best[0]
        lines.append(f"次: gne direction key --contrast {contrast} --layers {L} --target expert:{e}(rate_a が大きく rate_b が"
                     "小さい専門家ほど、書き込みたい位置の行が多く鍵が状態を拾いやすい)")
    return d, "\n".join(lines), 0


@gne_command(direction_app, "key-from-behave", "direction key-from-behave")
def direction_key_from_behave(c: Common, holder: dict,
                              behave: str = typer.Option(..., "--behave", help="gne behave の結果の名前(<root>/behave/<model_id>/<名前>.json)かパス。CHiPS のレシピを当てて A 側の質問に答えさせたもの"),
                              recipe: str = typer.Option(..., "--recipe", help="CHiPS のレシピ(保存したレシピの名前かパス)。鍵はこれを当てた状態で作るので、出力の keyed.iv に入れる"),
                              direction: Optional[str] = typer.Option(None, "--direction", help="CHiPS の方向の名前(出力の keyed.direction)。--contrast を省くと、この方向を作った対照集合を使う"),
                              contrast: Optional[str] = typer.Option(None, "--contrast", help="CHiPS の方向の対照集合(パスか configs/contrast/ の名前)。B 側の質問・validate.match・A 側の質問を取る。--direction を省くと方向の名前はこの name"),
                              label: Optional[list[str]] = typer.Option(None, "--label", help="「残った」とみなす読み手の判定のラベル(例 --label P。複数可)。省くと正規表現で判定する"),
                              match: Optional[str] = typer.Option(None, "--match", help="「残った」とみなす正規表現(大文字小文字を区別)。省くと behave の結果の --match、無ければ対照集合の validate.match(大文字小文字を無視)"),
                              min_rate: float = typer.Option(1.0, "--min-rate", help="振る舞いが出た種の割合がこれ以上の質問を入れる(既定 1.0 = 全部の種。A 側には振る舞いが確実に出る質問だけを入れる)"),
                              positions: Optional[str] = typer.Option(None, "--positions", help="書き込みたい位置 before:<正規表現>(既定は before:<validate.match>。大文字と小文字のある文字を含めば (?i) を付ける)。一致が広すぎると警告するので、振る舞いの書き始めに当たる正規表現を渡す"),
                              name: Optional[str] = typer.Option(None, "--name", help="出力の名前(既定 <behave の名前>_keyed)。鍵の名前の既定にもなる"),
                              out: Optional[Path] = typer.Option(None, "--out", "-o", help="書き出す JSON のパス(既定 <root>/keyed/<名前>.json。鍵のタブのひな形に出る)"),
                              replies: bool = typer.Option(True, "--replies/--no-replies", help="抜き出した質問の behave の返事を出力の replies に入れる(既定。鍵の作成は返事を作らずにそれを読む)。--no-replies なら鍵の作成で返事を作る"),
                              prompts: Optional[Path] = typer.Option(None, "--prompts", help="behave に渡した質問集(質問の記録が無い古い behave の結果のときだけ)"),
                              strict: bool = typer.Option(False, "--strict", help="behave の結果が、そのレシピを当てて A 側の質問に答えたものでなければエラー(既定は警告)")):
    """CHiPS のレシピを当てても消したい振る舞いが残った A 側の質問だけを集めて、keyed(鍵つきの書き込み)の対照集合の JSON を作る
    (モデルは動かさない)。user_prompts = 抜き出した質問、dont_touch の ordinary = CHiPS の対照集合の B 側の質問、
    keyed = {direction: CHiPS の方向, alpha: -1, iv: CHiPS のレシピ}。次に鍵のタブか gne direction key --contrast <出力> で鍵を作る。"""
    s = holder["session"] = c.session()
    d = s.direction_key_from_behave(behave, recipe=recipe, direction=direction, contrast=contrast, labels=label or None,
                                    match=match, min_rate=min_rate, positions=positions, name=name, out=out, replies=replies,
                                    prompts=prompts, strict=strict)
    d.pop("warnings", None)                        # 警告はエンベロープの warnings に入っている
    j = d["judge"]
    how = (f"読み手のラベル {', '.join(j['labels'])}" if j["method"] == "label"
           else f"正規表現 {j['match']}" + ("(大文字小文字を無視)" if j["ignore_case"] else "")
           + {"behave": "(behave の --match)", "validate": "(対照集合の validate.match)", "match": ""}[j["source"]])
    rate = "全部の種" if j["min_rate"] >= 1 else f"種の {j['min_rate']:g} 以上"
    lines = [(f"{d['path']}(" if d["path"] else "書き出していません(") + f"behave {d['behave']}・{d['n_prompts']} 問・種 {d['n_seeds']} 個、"
             f"レシピ {d['recipe']}、方向 {d['direction']}、対照集合 {d['contrast']})",
             f"  残った質問: {len(d['extracted'])} 問({how} が{rate}で出たもの。一部の種だけ {d['n_partial']} 問、出なかった {d['n_none']} 問)",
             f"  書き込みたい位置: {d['positions']}" + ("(validate.match から)" if d["positions_source"] == "validate" else ""),
             ]
    ps = d["position_share"]
    if ps["share"] is not None:
        lines.append(f"  一致の広さ: 返事のトークンの約 {ps['share'] * 100:.1f}% で一致の始まり(一致 {ps['starts']}・トークン {ps['tokens']:g}、"
                     f"{ps['threshold'] * 100:.0f}% 以上は広すぎる)" + ("  ← 広すぎる" if ps["broad"] else ""))
    h = d["halves"]
    if d["extracted"]:
        lines.append(f"  書き込みたい位置の見込み: 鍵の作成(偶数番目){h['even']['examples']} 例・{h['even']['positions']} 位置、"
                     f"key-check(奇数番目){h['odd']['examples']} 例・{h['odd']['positions']} 位置、一致の無い質問 {d['no_match']} 問"
                     + ("" if d["replies"] else "(返事を入れないので、鍵の作成では作り直した返事で変わります)"))
    lines.append(f"  dont_touch の ordinary: B 側の質問 {d['ordinary']['n']} 問" + (f"(user_prompts と同じ {d['ordinary']['removed']} 問を外した)"
                                                                                if d["ordinary"]["removed"] else ""))
    for r in d["extracted"][:12]:
        lines.append(f"    {r['i']:>3} {r['rate']:.2f} 一致 {r['starts']:>2}  {r['question']}" + ("  (繰り返し)" if r["loop"] else ""))
    if len(d["extracted"]) > 12:
        lines.append(f"    … ほか {len(d['extracted']) - 12} 問")
    lines += [f"注: {t}" for t in d["notes"]]
    if d["path"]:
        lines.append(f"次: gne direction key --contrast {d['path']} --layers <層> --iv {d['iv']}"
                     "(鍵のタブならひな形から選んで「方向から key-check まで」。方向は keyed.direction のものを使い回す)")
    return d, "\n".join(lines), 0 if d["path"] else 1


@gne_command(direction_app, "key-verify", "direction key-verify")
def direction_key_verify(c: Common, holder: dict,
                         layer: int = typer.Option(..., "--layer", help="層 L"),
                         key: str = typer.Option(..., "--key", help="鍵の名前"),
                         direction: str = typer.Option(..., "--direction", help="書く向きの方向の名前(先頭の - は α の符号に畳む)"),
                         alphas: str = typer.Option("1,2", "--alphas", help="確かめる α(カンマ区切り)"),
                         target: str = typer.Option("ffn", "--target", help="書き込み先(鍵の書き込み先と同じ): ffn / attn / shared(MoE の層の shared expert の ffn_down_shexp)/ expert:e(ffn_down_exps の専門家 e。例 expert:31)"),
                         direction_layer: str = typer.Option("own", "--direction-layer", help="own(その層の方向)/ base(基準層の方向)"),
                         text: Optional[list[str]] = typer.Option(None, "--text", help="流す文(既定は語尾の違う日本語と英語の 2 文)。複数可。expert:e では専門家 e が選ばれる文(鍵の書き込みたい位置を含む文など)を渡す")):
    """V30(CHANGE-07): keyed の LoRA が式どおり ΔW x = α (κᵀx) d を書くか。無介入で重み行列の入力 x と出力を捕捉し、
    LoRA を当てた出力の差と比べる(|κᵀx| の大きいトークンで相対誤差の中央値 < 3%、鍵の作成に使った書き込みたくない側の例で
    漏れ < 0.1)。
    shared はゲートを掛ける前の ffn_shexp、expert:e はルーターの重みを掛ける前の ffn_moe_down の、専門家 e が選ばれた枠で測り、
    ほかの専門家の枠の差(0 のはず)と、LoRA で選ばれる専門家が変わったトークンの数(0 のはず)も確かめる。"""
    from .keyed import parse_target
    kind, expert = parse_target(target)
    s = holder["session"] = c.session()
    d = s.check_keyed(layer, key=key, direction=direction, alphas=_floats(alphas, "--alphas"), kind=kind, expert=expert,
                      direction_layer=direction_layer, texts=text or None)
    lines = [f"L{layer} {d['tensor']}(鍵 {d['key']}、方向 {d['direction']}、‖d‖={d['d_norm']:.3g}、kn {d['kn']:.3f}"
             + ("、post-norm の補正あり(一次近似)" if d.get("post_norm") else "")
             + (f"、gate_a {d['gate']:.3g} で B を割った" if d.get("gate") is not None else "")
             + (f"、residual_scale {d['res_scale']:g} で B を割った" if d.get("res_scale") is not None else "") + ")"]
    for r in d["rows"]:
        lines.append(f"  α {r['alpha']:+g}: 相対誤差の中央値 {_f(r['rel_err_median'], '.2e')}(|κᵀx| の上位 {r['n_big']}/{r['n_tokens']} "
                     f"{'枠' if kind == 'expert' else 'トークン'})、cos {_f(r['cos_median'], '.4f')}、書き込みたくない側への漏れ {_f(r['leak_b'], '.3g')}"
                     + (f"、ほかの専門家の枠の差 {r['other_max']:.2g}、選ばれ方の変化 {r['selection_changed']} トークン"
                        if "other_max" in r else "") + f"  {'○' if r['ok'] else '×'}")
    if d.get("note"):
        lines.append(f"注: {d['note']}")
    lines.append(f"V30 {'合格' if d['v30'] else '不合格'}(相対誤差 < 3%"
                 + ("、書き込みたくない側への漏れ < 0.1" if d["judged_leak"] else "、書き込みたくない側への漏れは判定なし")
                 + ("、ほかの専門家の枠の差 < 1e-3、選ばれ方の変化 0" if kind == "expert" else "") + ")")
    return d, "\n".join(lines), 0 if d["v30"] else 1


def _kc_e(e: dict | None, w: int = 13) -> str:
    """key-causal の効き目の列: 平均±95% の幅(無ければ —)。"""
    if not e:
        return format("—", f">{w}")
    ci = e.get("ci")
    return format(f"{e['mean']:+.3f}" + (f"±{ci:.3f}" if ci is not None else ""), f">{w}")


def _kc_range(r: dict | None, w: int = 15) -> str:
    return format("—", f">{w}") if not r else format(f"{r['min']:+.3f}〜{r['max']:+.3f}", f">{w}")


KC_SORTS = ("place", "effect", "ratio", "side")


def _kc_eff_short(e: dict | None, mark, w: int = 9) -> str:
    """1 か所 1 行の表の効き目の欄: 平均(印が付けば ★)。"""
    if not e:
        return format("—", f">{w}")
    return format(f"{e['mean']:+.2f}" + ("★" if mark else ""), f">{w}")


def kc_ratio_alpha(d: dict, ratio_alpha: float | None = None) -> float:
    """「確率の下がり方 ÷ 副作用」と副作用を出す α(既定は |α| が最大のもの)。"""
    alphas = [float(a) for a in d["alphas"]]
    if ratio_alpha is None:
        return max(alphas, key=abs)
    for a in alphas:
        if abs(a - float(ratio_alpha)) < 1e-9:
            return a
    raise GneError("BAD_ARGS", f"--ratio-alpha {ratio_alpha:g} は測った α({', '.join(f'{a:g}' for a in alphas)})のどれかです")


def kc_sorted(cands: list[dict], sort: str, a: float) -> list[dict]:
    """1 か所 1 行の候補の並べ替え: place(書き込み先・層の順)/ effect(α での確率の下がり方の大きい順)/ ratio(下がり方 ÷ 副作用の
    大きい順)/ side(副作用の小さい順)。値の無い候補は後ろ。"""
    if sort not in KC_SORTS:
        raise GneError("BAD_ARGS", f"--sort は {' / '.join(KC_SORTS)} です({sort!r})")
    key = f"{a:g}"
    if sort == "place":
        return list(cands)

    def val(c):
        b = (c.get("by_alpha") or {}).get(key) or {}
        e = b.get("reply") or b.get("probe")
        if sort == "effect":
            return None if not e else e["mean"]
        if sort == "ratio":
            return None if b.get("ratio") is None else -b["ratio"]
        sb = b.get("side_b")
        return None if not sb else -sb["mean"]
    return sorted(cands, key=lambda c: (val(c) is None, val(c) if val(c) is not None else 0.0))


def key_causal_lines(d: dict, *, only_marked: bool = False, ratio_alpha: float | None = None, sort: str = "place") -> list[str]:
    """第 1 段(key-causal)の結果の表(CLI と MCP の説明用)。1 か所(書き込み先 × 層)につき 1 行で、強さ(α)ごとの確率の下がり方と
    印、全部の強さで印が付いたか、ratio_alpha での「確率の下がり方 ÷ 副作用」と副作用を並べる。"""
    from .keyed_causal import with_candidates
    d = with_candidates(d)
    has_reply = d["measure"]["regex"] is not None
    has_probe = any(r.get("probe") for r in d["rows"])
    alphas = [float(a) for a in d["alphas"]]
    ra = kc_ratio_alpha(d, ratio_alpha)
    iv = d.get("iv_spec")
    lines = [f"効く場所のふるい分け(生成なし。対照集合 {d['contrast']}、方向 {d['direction']}、"
             + (f"介入 {iv if isinstance(iv, str) else ' '.join(iv)}、" if iv else "介入なし、")
             + f"書き込み先 {','.join(d['targets'])}、候補 {d['n_candidates']} か所、強さ(α){','.join(f'{a:g}' for a in alphas)}、"
             f"候補ごとの比較用の鍵 {d['pseudo']} 本、{d['seconds']:.1f} 秒)",
             f"注: {d['notes'][0]}"]
    c = d.get("counts") or {}
    ca = c.get("a") or {}
    if has_reply:
        src = "positions の正規表現" if d["measure"]["source"] == "positions" else "--measure-match"
        lines.append(f"確率の下がり方(効き目、返事): 作成に使っていない {ca.get('examples', 0)} 問のうち、{d['measure']['regex']}({src})に"
                     f"一致した {ca.get('matched', 0)} 問の返事で、一致した文字の対数確率の変化(負 = 振る舞いが出にくくなる)")
    if has_probe:
        pc = c.get("probe") or {}
        lines.append(f"効き目(probe): {pc.get('a', 0)} 行で、log p(振る舞いの答え) − log p(望む答え) の変化(負 = 望む答えに寄る)")
    for a, mk in d["marks"].items():
        for m, v in mk.items():
            pl = d["pool"][a][m]
            lines.append(f"α {a}({'返事' if m == 'reply' else 'probe'}): 印 {v['marked']} / 候補 {v['candidates']}"
                         f"(偶然でも {v['candidates']} × {d['chance'] * 100:.0f}% = {v['expected']:g} 個は印が付く)、"
                         f"比較用の鍵の 5% 点 {pl['p5']:+.3f}({pl['n']} 本)")
    head = f"  {'書込':>4} {'層':>3} |"
    if has_reply:
        head += "".join(f" {'α ' + format(a, 'g'):>9}" for a in alphas) + f" {'全部有意':>4} |"
    if has_probe:
        head += "".join(f" {'probe ' + format(a, 'g'):>9}" for a in alphas) + f" {'全部有意':>4} |"
    head += f" {'下がり方÷副作用':>8} {'副作用':>7} {'dt最小':>7} | {'kc余裕':>7} {'比例':>6}  (右の 4 列は α {ra:g})"
    lines.append(head)
    key = f"{ra:g}"
    for cd in kc_sorted(d["candidates"], sort, ra):
        ba = cd["by_alpha"]
        marked = any(b.get("mark") or b.get("probe_mark") for b in ba.values())
        if only_marked and not marked:
            continue
        ln = f"  {_target_who(cd['target']):>4} {cd['layer']:>3} |"
        if has_reply:
            ln += "".join(" " + _kc_eff_short((ba.get(f"{a:g}") or {}).get("reply"), (ba.get(f"{a:g}") or {}).get("mark")) for a in alphas)
            ln += f" {('○' if cd.get('all_marked') else ''):>4} |"
        if has_probe:
            ln += "".join(" " + _kc_eff_short((ba.get(f"{a:g}") or {}).get("probe"), (ba.get(f"{a:g}") or {}).get("probe_mark")) for a in alphas)
            ln += f" {('○' if cd.get('all_marked_probe') else ''):>4} |"
        b = ba.get(key) or {}
        lin = cd.get("linear") or {}
        lr = lin.get("ratio")
        lin_s = "向き×" if lin and not lin.get("consistent") else (f"×{lr:.2f}" if lr is not None else "—")
        ln += (f" {_fc(b.get('ratio'), '>8.1f')} {_fc((b.get('side_b') or {}).get('mean'), '>+7.3f')} {_fc(b.get('dt_min'), '>+7.3f')} |"
               f" {_fc((cd.get('key_check') or {}).get('margin'), '>+7.3f')} {lin_s:>6}")
        lines.append(ln)
    floor = d.get("ratio_floor") or 0.01
    lines.append("  α ごとの列 = 確率の下がり方(keyed を当てたときの、振る舞いの部分の対数確率の変化。iv だけの状態との差の例ごとの平均。"
                 "負 = 下がった)。★ = その強さで有意(全部の候補の比較用の鍵の 5% 点より大きく下がった)。比較用の鍵 = 書き込みたい場面と"
                 "書き込みたくない場面のラベルをでたらめに入れ替えて、同じ手順で作った鍵(偶然でどれくらい変わるかの基準)。"
                 "全部有意 = どの強さでも ★。副作用 = 書き込みたくない位置の返事の 1 トークンあたりの対数確率の変化(負 = 下がった)。"
                 f"下がり方÷副作用 = (−確率の下がり方)÷(−副作用)。副作用が {floor:g} より小さいとき(上がったときも)は {floor:g} で割り、"
                 "確率が下がっていなければ —。大きいほど、書き込みたくない場面を変えずに振る舞いを下げる。dt最小 = dont_touch の群の返事の"
                 "1 トークンあたりの変化のうち最も下がった群。kc余裕 = key-check の選択性 − 比較用の鍵の p95(参考)。比例 = 最大の |α| での"
                 "効き目 ÷ 最小の |α| での効き目を |α| の比で割った値(1 = 比例、向き× = 強さで符号がそろわない、— = 最小の |α| での効き目が"
                 "95% の幅の中)")
    lines.append("  選び方の目安: 確率の下がり方・副作用・全部の強さで有意か、の 3 つを並べて自分で選ぶ。確率の下がり方だけで選ぶと、返事を"
                 "壊して確率を下げている候補を拾うことがある(副作用も比較用の鍵並みに大きい)。--sort ratio / effect / side で並べ替えられる")
    for x in d.get("dropped") or []:
        lines.append(f"  {x['target']} L{x['layer']}: 外しました({x['error']})")
    lines += [f"注: {t}" for t in d["notes"][1:]]
    marked = [r for r in d["rows"] if r.get("mark") or r.get("probe_mark")]
    pick = marked[0]["unit"] if marked else (d["rows"][0]["unit"] if d["rows"] else "<ユニット>")
    lines.append(f"次: gne direction key-causal-verify --from {d['name']} --unit {pick}(選んだ候補を生成で確かめる。--unit は複数可。"
                 "複数の層をまとめて 1 つの候補にするなら --combo \"<ユニット>,<ユニット>\")")
    return lines


@gne_command(direction_app, "key-causal", "direction key-causal")
def direction_key_causal(c: Common, holder: dict,
                         contrast: Path = typer.Option(..., "--contrast", help="鍵の対照集合(YAML か JSON。configs/keyed/ のひな形や gne direction key-from-behave の JSON)"),
                         layers: Optional[str] = typer.Option(None, "--layers", help="候補の層(既定は全層。例 4-20 / 8,14,20。最終層と MTP の層は除く)"),
                         target: str = typer.Option("ffn,attn", "--target", help="書き込み先(カンマ区切り): ffn / attn / shared / expert:e。既定 ffn,attn(書けない書き込み先は外す)"),
                         alphas: Optional[str] = typer.Option(None, "--alphas", help="α の段(例 --alphas=-1,-2,-4)。既定は 1,2,4 に対照集合の keyed.alpha の符号を掛けたもの"),
                         pseudo: int = typer.Option(3, "--pseudo", help="候補ごとに測る比較用の鍵(ラベルをシャッフルして作った鍵)の本数"),
                         iv: Optional[list[str]] = typer.Option(None, "--iv", help="当てておく介入(既定は対照集合の keyed.iv)。" + _KEY_IV),
                         no_iv: bool = typer.Option(False, "--no-iv", help="対照集合の keyed.iv も当てない"),
                         direction: Optional[str] = typer.Option(None, "--direction", help="書き込む方向(既定は対照集合の keyed.direction、無ければ対照集合の name。無ければ作る)"),
                         measure_match: Optional[str] = typer.Option(None, "--measure-match", help="効き目を測る振る舞いの部分の正規表現(既定は positions の before: の正規表現)"),
                         probe: Optional[Path] = typer.Option(None, "--probe", help="選択式の振る舞い向け: 1 行 1 問の JSONL({\"prompt\", \"prefix\": \"Answer:\", \"behavior\": \" B\", \"desired\": \" C\"}、書き込みたくない場面の行は \"group\": \"<群>\")"),
                         max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="返事を作るときの長さ(既定は対照集合の keyed.max_tokens、無ければ 96)"),
                         side_tokens: int = typer.Option(32, "--side-tokens", help="副作用を測る返事のトークンの数(返事の最初から)"),
                         name: Optional[str] = typer.Option(None, "--name", help="結果と鍵の名前(既定 <対照集合の名前>-causal。鍵は <名前>-ffn などで保存)"),
                         only_marked: bool = typer.Option(False, "--only-marked", help="表には印の付いた行だけを出す(結果の JSON には全部入る)"),
                         ratio_alpha: Optional[float] = typer.Option(None, "--ratio-alpha", help="「確率の下がり方 ÷ 副作用」と副作用を出す強さ(既定は |α| が最大のもの)"),
                         sort: str = typer.Option("place", "--sort", help="表の並べ方: place(書き込み先・層の順、既定)/ effect(確率の下がり方の大きい順)/ ratio(下がり方 ÷ 副作用の大きい順)/ side(副作用の小さい順)。--ratio-alpha の強さで"),
                         yes: bool = _YES):
    """keyed の効く場所を探す第 1 段(生成なしのふるい分け)。候補(層 × 書き込み先)ごとに鍵を作り(偶数番目)、作成に使っていない
    奇数番目の返事で、keyed を当てたときに振る舞いの部分の対数確率がどれだけ下がるか、dont_touch の返事がどれだけ変わるかを、
    比較用の鍵と並べて測る。比較用の鍵の 5% 点より下がった候補に印を付け、偶然でも印が付く数(候補の数 × 5%)を並べる。
    生成なしの目安なので、効くかどうかは gne direction key-causal-verify の生成で数える。"""
    s = holder["session"] = c.session()
    _gate(s, "direction_key_causal", {"contrast": str(contrast), "layers": layers, "targets": target, "alphas": alphas,
                                      "pseudo": pseudo, "probe": str(probe) if probe else None, "max_tokens": max_tokens}, yes, None)
    d = s.direction_key_causal(contrast, layers=layers, targets=target, alphas=alphas, pseudo=pseudo, iv=_iv_arg(iv), no_iv=no_iv,
                               direction=direction, measure_match=measure_match, probe=probe, max_tokens=max_tokens,
                               side_tokens=side_tokens, name=name, seed=c.seed, progress=s.progress)
    d.pop("warnings", None)
    if sort not in KC_SORTS:
        raise GneError("BAD_ARGS", f"--sort は {' / '.join(KC_SORTS)} です({sort!r})")
    return d, "\n".join(key_causal_lines(d, only_marked=only_marked, ratio_alpha=ratio_alpha, sort=sort)), 0


def _kcv_count(x: dict | None, key: str = "behavior") -> str:
    if not x:
        return "—"
    v = x.get(key)
    return "—" if v is None else f"{v}/{x['n']}"


def key_causal_verify_lines(d: dict) -> list[str]:
    """第 2 段(key-causal-verify)の結果の表。"""
    j = d["judge"]
    how = (f"読み手のラベル {', '.join(j['labels'])}" if j["method"] == "label" else f"正規表現 {j['match']}")
    dts = "、".join(f"{g} {n} 問" for g, n in d["n_dt"].items())
    lines = [f"生成での確かめ(書き込みたい場面 {d['n_a']} 問(作成に使っていない奇数番目)" + (f"、dont_touch {dts}" if dts else "")
             + f"、返事 {d['max_tokens']} トークン、" + ("greedy" if d["greedy"] else f"温度 {d['temperature']:g}・種 {len(d['seeds'])} 個")
             + f"、{d['seconds']:.1f} 秒)", f"  判定: {how}"]
    b = d["base"]
    lines.append(f"  iv のみ: 振る舞い {_kcv_count(b['a'])}、繰り返し {b['a']['loop']}、空 {b['a']['empty']}、上限 {b['a']['at_limit']}"
                 + ("" if "hits" not in b["a"] else f"、一致の総数 {b['a']['hits']}"))
    for r in d["rows"]:
        k, p = r["a"]["keyed"], r["a"]["pseudo"]
        lines.append(f"  組 {r.get('label') or ''}: {r['unit']}" if r.get("combo") else f"  {r['unit']}")
        lines.append(f"    書き込みたい場面: 振る舞い keyed {_kcv_count(k)} / 比較用の鍵 {_kcv_count(p)}(iv のみ {_kcv_count(b['a'])})"
                     + ("" if "hits" not in k else f"、一致の総数 {k['hits']} / {p['hits']}(iv のみ {b['a']['hits']})")
                     + ("" if k.get("changed") is None else f"、返事が変わった {k['changed']}/{k['n']} / {p['changed']}/{p['n']}"))
        lines.append(f"    繰り返し・空・上限: keyed {k['loop']}・{k['empty']}・{k['at_limit']} / 比較用の鍵 {p['loop']}・{p['empty']}・{p['at_limit']}")
        for g, v in r["dt"].items():
            kk, pp = v["keyed"], v["pseudo"]
            lines.append(f"    dont_touch {g}: 返事が変わった " + ("—" if kk.get("changed") is None else
                                                          f"{kk['changed']}/{kk['n']} / {pp['changed']}/{pp['n']}"
                                                          f"(先頭から同じ部分 {kk['same_prefix'] * 100:.0f}% / {pp['same_prefix'] * 100:.0f}%)")
                         + f"、振る舞い {_kcv_count(kk)} / {_kcv_count(pp)}(iv のみ {_kcv_count(b['dt'].get(g))})"
                         + f"、繰り返し・空・上限 {kk['loop']}・{kk['empty']}・{kk['at_limit']} / {pp['loop']}・{pp['empty']}・{pp['at_limit']}")
    lines += [f"注: {t}" for t in d["notes"]]
    lines.append(f"返事: {d['path']}")
    return lines


@gne_command(direction_app, "key-causal-verify", "direction key-causal-verify")
def direction_key_causal_verify(c: Common, holder: dict,
                                unit: Optional[list[str]] = typer.Option(None, "--unit", help="確かめる候補の keyed のユニット(key-causal の表の行。例 14:ffn:keyed:a-2@方向#鍵。複数可)。1 つずつ別の候補になる"),
                                combo: Optional[list[str]] = typer.Option(None, "--combo", help="複数の層をまとめて 1 つの候補として確かめる組(ユニットをカンマで区切る。例 \"11:attn:keyed:a2,18:attn:keyed:a2,23:attn:keyed:a2\"。強さは行ごとに違ってよい。複数可、--unit と併用可)"),
                                result: Optional[str] = typer.Option(None, "--from", help="第 1 段(gne direction key-causal)の結果の名前。対照集合・介入・方向・鍵・正規表現の既定になる"),
                                contrast: Optional[Path] = typer.Option(None, "--contrast", help="鍵の対照集合(既定は --from の結果のもの)"),
                                iv: Optional[list[str]] = typer.Option(None, "--iv", help="当てておく介入(既定は --from の結果のもの)。" + _KEY_IV),
                                no_iv: bool = typer.Option(False, "--no-iv", help="介入を当てない"),
                                pseudo_index: int = typer.Option(0, "--pseudo-index", help="比べる比較用の鍵の番号(NAME~j の j)"),
                                match: Optional[str] = typer.Option(None, "--match", help="振る舞いを数える正規表現(既定は第 1 段で測った正規表現、無ければ対照集合の validate.match)"),
                                judge: Optional[str] = typer.Option(None, "--judge", help="読み手の判定の文面({q} に質問、{a} に返事。gne behave と同じ)"),
                                labels: str = typer.Option("R,P,C", "--labels", help="判定のラベル(gne behave と同じ)"),
                                label: Optional[list[str]] = typer.Option(None, "--label", help="振る舞いとみなすラベル(--judge のとき。例 --label P)"),
                                max_tokens: int = typer.Option(256, "--max-tokens", help="返事の長さ"),
                                temperature: float = typer.Option(0.0, "--temperature", help="温度(0 = greedy)"),
                                seeds: Optional[str] = typer.Option(None, "--seeds", help="種(カンマ区切り。温度があるとき)"),
                                no_dont_touch: bool = typer.Option(False, "--no-dont-touch", help="dont_touch の質問は答えさせない"),
                                name: Optional[str] = typer.Option(None, "--name", help="結果の名前(既定は --from の名前。<名前>.verify.json に返事も保存)"),
                                yes: bool = _YES):
    """keyed の効く場所を探す第 2 段(生成での確かめ)。選んだ候補と、それぞれの比較用の鍵 1 本で、作成に使っていない質問
    (と dont_touch の質問)に gne behave と同じ生成をさせ、振る舞いが出た数を介入(iv)だけのときと比べる。dont_touch では
    返事が変わった数、どれも繰り返し・空の返事・上限で切れた数を出す。--combo の組は、複数のユニットを一緒に当てて 1 つの候補に
    し、比較用の鍵も組の各ユニットを同じ番号の比較用の鍵に替えた組にする。"""
    if not unit and not combo:
        raise GneError("BAD_ARGS", "--unit か --combo で確かめる候補を渡してください")
    s = holder["session"] = c.session()
    sd = [int(x) for x in seeds.split(",") if x.strip()] if seeds else None
    _gate(s, "direction_key_causal_verify", {"units": list(unit or []), "combos": list(combo or []), "result": result,
                                             "contrast": str(contrast) if contrast else None,
                                             "max_tokens": max_tokens, "seeds": sd, "judge": judge}, yes, None)
    d = s.direction_key_causal_verify(list(unit or []), result=result, contrast=contrast, iv=_iv_arg(iv), no_iv=no_iv,
                                      pseudo_index=pseudo_index, match=match, judge=judge, labels=labels, label=label or None,
                                      max_tokens=max_tokens, temperature=temperature, seeds=sd, dont_touch=not no_dont_touch,
                                      name=name, combos=list(combo or []) or None, progress=s.progress)
    d.pop("warnings", None)
    return d, "\n".join(key_causal_verify_lines(d)), 0


@gne_command(heads_app, "causal", "heads causal")
def heads_causal(c: Common, holder: dict,
                 name: str = typer.Option(..., "--name", help="方向の名前"),
                 contrast: Optional[Path] = typer.Option(None, "--contrast", help="対照集合(既定は方向を作った対照集合)"),
                 mode: str = typer.Option("ablate", "--mode", help="ablate(出力を 0 にして下がり方を測る、最初の候補出し)/ marginal(いまの組から s を上げたときの効率 η、配り直し用)。両者の順位は一致しない"),
                 layers: str = typer.Option("auto", "--layers", help="auto(層単位で止めて、下がる層を --top-layers 個)/ all / 10-16 / 1,5,9-12"),
                 candidates: Optional[str] = typer.Option(None, "--candidates", help="all / contrib:N(直接の寄与の上位 N)/ L:Hh,…(明示)"),
                 score_layers: Optional[str] = typer.Option(None, "--score-layers", help="射影の値を測る層(既定は基準層と後段 2 層)"),
                 primary: Optional[int] = typer.Option(None, "--primary", help="causal / contrib / ratio を出す層(既定は基準層)"),
                 base_recipe: Optional[Path] = typer.Option(None, "--base-recipe", help="marginal: いまの組のレシピ"),
                 base_unit: Optional[list[str]] = typer.Option(None, "--base-unit", help="marginal: いまの組(L:Hh:sS@NAME、複数可)"),
                 delta_s: float = typer.Option(3.0, "--delta-s", help="marginal: s を上げる量"),
                 hard: int = typer.Option(16, "--hard", help="marginal: 効きにくい例の数"),
                 hard_subset: Optional[str] = typer.Option(None, "--hard-subset", help="marginal: 保存した効きにくい例(名前かパス)"),
                 b_side_max: Optional[float] = typer.Option(None, "--b-side-max", help="B 側の射影の値の変化の絶対値がこれを超えるヘッドに除外の印"),
                 top_layers: int = typer.Option(4, "--top-layers", help="--layers auto で 1 個ずつ測る層の数"),
                 positions: str = typer.Option("last", "--positions"), subset: str = typer.Option("all", "--subset"),
                 b_examples: int = typer.Option(16, "--b-examples", help="B 側への影響を測る例の数(0 で全部)"),
                 out_name: Optional[str] = typer.Option(None, "--out-name", help="保存名(既定は方向の名前)"),
                 direction_layer: str = typer.Option("own", "--direction-layer", help="marginal の射影の方向: own(ヘッドの層)/ base(基準層)"),
                 sort: str = typer.Option("auto", "--sort", help="表の並び: auto(ablate は causal_max、marginal は eta)/ causal / contrib / ratio / eta"),
                 show: int = typer.Option(30, "--show"),
                 max_examples: Optional[int] = typer.Option(None, "--max-examples", help="止めて測る質問の数(等間隔に間引く。既定は全部)"),
                 include_score_layers: bool = typer.Option(False, "--include-score-layers", help="射影の値を測る層とそれより後のヘッドも候補にする(既定は外す。測る量を直接動かすだけで、振る舞いは変わりにくい)")):
    """因果スクリーニング(CHANGE-06 C): ヘッドを止めた(ablate)/ 強めた(marginal)ときの、生成なしの射影の値の下がり方。
    各行に直接の寄与 contrib(r̂ᵀW[:, blk]·x の平均)と causal / contrib の比を並べる(1 より大きければ間接の経路、
    小さければ補い合い。射影の値を測る層より後ろのヘッドは contrib が 0 なので causal_max で見る)。
    ablate と marginal は順位が入れ替わる: 最初の候補出しは ablate、配り直しは marginal。"""
    s = holder["session"] = c.session()
    d = s.heads_causal(name, contrast=contrast, mode=mode, layers=layers, candidates=candidates,
                       score_layers=_ints(score_layers), primary=primary, base_recipe=base_recipe, base_units=base_unit,
                       delta_s=delta_s, hard=hard, hard_subset=hard_subset, b_side_max=b_side_max, top_layers=top_layers,
                       positions=positions, subset=subset, b_examples=b_examples, name=out_name,
                       direction_layer=direction_layer, max_examples=max_examples, include_score_layers=include_score_layers)
    P = d["primary"]
    lines = [f"{d['path']}({d['n_rows']} ヘッド、mode={mode}、測る層 L{','.join(map(str, d['score_layers']))}(主 L{P})、"
             f"{d['seconds']:.0f} 秒)",
             "無介入の射影の値: " + "、".join(f"L{L} A {x['A']:.2f} / B {x['B']:.2f}" for L, x in d["base_scores"].items())]
    if d.get("layers"):
        lines.append("層単位(全ヘッドを止めたときの下がり方): " + " ".join(
            f"L{r['layer']}:{r['causal_max']:+.2f}" for r in sorted(d["layers"], key=lambda r: -r["causal_max"])[:12]))
    rows = d["rows"]
    if sort != "auto":
        rows = sorted(rows, key=lambda r: -(r.get(sort) if r.get(sort) is not None else -1e9))
    lines.append(f"  {'住所':>9} {'種類':>6} " + " ".join(f"{'L' + k:>7}" for k in map(str, d["score_layers"]))
                 + f" {'contrib':>8} {'ratio':>6}" + (f" {'eta':>7}" if mode == "marginal" else "") + f" {'B変化':>7} {'B直交':>7}")
    for r in rows[:show]:
        lines.append(f"  {('L' + str(r['layer']) + ' H' + str(r['head'])):>9} {r['kind']:>6} "
                     + " ".join(f"{r['score_drop'][str(L)]:>+7.2f}" for L in d["score_layers"])
                     + f" {_f(r['contrib'], '+8.3f')} {_f(r['ratio'], '6.2f')}"
                     + (f" {r['eta']:>+7.3f}" if mode == "marginal" else "")
                     + f" {_f(r.get('b_side_delta'), '+7.2f')} {_f(r.get('drift_orth_b'), '7.4f')}"
                     + ("  除外" if r.get("excluded") else "") + ("  (組)" if r.get("in_base") else ""))
    if mode == "marginal" and d.get("spearman_ablate_marginal") is not None:
        lines.append(f"ablate と marginal の順位相関(Spearman、{d['n_compared']} ヘッド): {d['spearman_ablate_marginal']:+.2f}")
    lines.append("次: gne heads allocate --name " + d["direction"] + " --target <A 側の射影の値の目標>")
    return d, "\n".join(lines), 0


def _allocate_note(r: dict, d: dict) -> list[str]:
    """止まった理由の説明。目標に届いた組は黙っている(s_max に当たった組の比の話は、見落とすと結果を読み違える)。"""
    stop, c = r.get("stop"), r.get("capped")
    if stop == "target" or stop is None:
        return []
    if stop == "s_max" and c:
        return [f"      L{c['layer']} H{c['head']} が上限 s={c['s']:g} に当たりました。比を保つ配分なので、他のヘッドも一緒に止まっています。"
                "--method greedy か --s-max を試せます(上限を上げるだけだと、目標が遠いときに壊れる強さまで上がります)"]
    why = {"drift": f"B 側の直交成分のずれが {d.get('drift_max')} を超えるので、その手前で止めました",
           "no_gain": "どのヘッドを上げても射影の値が下がらないので止めました", "s_max": "全部のヘッドが s の上限に当たりました",
           "steps": "上げる回数の上限に達しました"}
    return [f"      {why[stop]}"] if stop in why else []


@gne_command(heads_app, "writers", "heads writers")
def heads_writers(c: Common, holder: dict,
                  name: str = typer.Option(..., "--name", help="方向の名前"),
                  contrast: Optional[Path] = typer.Option(None, "--contrast", help="対照集合(既定は方向を作った対照集合)"),
                  ref_layer: Optional[int] = typer.Option(None, "--ref-layer", help="寄与を測る方向の層(既定は基準層)"),
                  expert_layers: Optional[str] = typer.Option(None, "--expert-layers", help="専門家ごとにも分ける層(例 36,42-46)"),
                  max_examples: int = typer.Option(64, "--max-examples", help="使う質問の数(等間隔に間引く)"),
                  top: int = typer.Option(12, "--top", help="層ごとに出す専門家の数"),
                  out_name: Optional[str] = typer.Option(None, "--out-name", help="保存名(既定は方向の名前)")):
    """方向の差を、層ごとに attention・専門家・共有の専門家の書き込みに分ける。どこを探すかを決めるために使う。"""
    s = holder["session"] = c.session()
    d = s.heads_writers(name, contrast=contrast, ref_layer=ref_layer, expert_layers=parse_layers(expert_layers) if expert_layers else None,
                        max_examples=max_examples, name=out_name, top=top)
    gap = d["gap"] or 1.0
    t = d["totals"]
    lines = [f"{d['path']}(L{d['ref_layer']} の方向、A {d['n']['A']} 問・B {d['n']['B']} 問、{d['positions']}、A−B の差 {d['gap']:.2f}、"
             f"足し合わせの誤差 {d['sum_error'] * 100:.1f}%)",
             f"合計: attention {t['attention'] / gap:+.0%}・専門家 {t['moe'] / gap:+.0%}・共有 {t['shared'] / gap:+.0%}",
             f"{'層':>4} {'種類':<8} {'attention':>10} {'専門家':>8} {'共有':>8}"]
    for r in d["rows"]:
        if max(abs(r["attention"]), abs(r["moe"]), abs(r["shared"])) >= 0.02 * abs(gap):
            lines.append(f"{r['layer']:>4} {r['kind']:<8} {r['attention']:>+10.2f} {r['moe']:>+8.2f} {r['shared']:>+8.2f}")
    for L, rows in d["experts"].items():
        lines.append(f"L{L} の専門家(差の大きい順): " + " ".join(f"E{x['expert']}({x['diff']:+.2f}、A {x['rate_a']:.0%}/B {x['rate_b']:.0%})" for x in rows))
    return d, "\n".join(lines), 0


@gne_command(heads_app, "allocate", "heads allocate")
def heads_allocate(c: Common, holder: dict,
                   name: str = typer.Option(..., "--name", help="方向の名前"),
                   contrast: Optional[Path] = typer.Option(None, "--contrast", help="対照集合(既定は方向を作った対照集合)"),
                   candidates: Optional[str] = typer.Option(None, "--candidates", help="heads causal の結果(名前か .causal.jsonl。既定は方向の名前)"),
                   target: float = typer.Option(..., "--target", help="効きにくい例の射影の値の平均の目標(例 −3。B 側の平均あたりが出発点)"),
                   score_layer: Optional[int] = typer.Option(None, "--score-layer", help="射影の値を測る層(既定は heads causal の最も後ろの層)"),
                   n_heads: str = typer.Option("6,5,4", "--n-heads", help="作る組のヘッド数"),
                   hard: int = typer.Option(16, "--hard", help="効きにくい例の数"),
                   hard_subset: Optional[str] = typer.Option(None, "--hard-subset", help="保存した効きにくい例(名前かパス)"),
                   proportional_to: str = typer.Option("eta", "--proportional-to", help="eta(効率)/ causal(止めたときの効き目)"),
                   from_zero: bool = typer.Option(True, "--from-zero/--add", help="0 から配り直す / いまの s に足す"),
                   b_side_max: Optional[float] = typer.Option(None, "--b-side-max", help="B 側の射影の値を動かすヘッドを除く閾値"),
                   base_recipe: Optional[Path] = typer.Option(None, "--base-recipe", help="いまの組(無ければ ablate の上位から作る)"),
                   delta_s: float = typer.Option(3.0, "--delta-s"), pool: int = typer.Option(30, "--pool", help="効率を測る候補(いまの組 + 効き目の上位 N)"),
                   s_max: Optional[float] = typer.Option(None, "--s-max", help="s の上限(既定 proportional は 30、greedy はなし)"),
                   method: str = typer.Option("proportional", "--method", help="proportional(s ∝ η で全体を上げる)/ greedy(いちばん効くヘッドを 1 本ずつ上げる)"),
                   remain_threshold: float = typer.Option(0.0, "--remain-threshold", help="「残る」とみなす射影の値"),
                   direction_layer: str = typer.Option("own", "--direction-layer"),
                   out_name: Optional[str] = typer.Option(None, "--out-name", help="保存名(既定は方向の名前)"),
                   candidate_layers: Optional[str] = typer.Option(None, "--candidate-layers", help="候補のヘッドをこの層に絞る(例 3-44)"),
                   include_score_layers: bool = typer.Option(False, "--include-score-layers", help="射影の値を測る層とそれより後のヘッドも候補にする(既定は外す。測る量を直接動かすだけで、振る舞いは変わりにくい)")):
    """効率ベースの強さ配分(CHANGE-06 D): 効きにくい例を選び、効率 η を測って s ∝ η で配り、全体の大きさを二分探索して、
    ヘッド数を変えた組の比較表を出す。残る件数と B 側のずれが同程度なら、ヘッド数が最小の組を推す。"""
    s = holder["session"] = c.session()
    d = s.heads_allocate(name, contrast=contrast, candidates=candidates, target=target, score_layer=score_layer,
                         n_heads=_ints(n_heads) or [6, 5, 4], hard=hard, hard_subset=hard_subset,
                         proportional_to=proportional_to, from_zero=from_zero, b_side_max=b_side_max,
                         base_recipe=base_recipe, delta_s=delta_s, pool=pool, s_max=s_max, method=method,
                         remain_threshold=remain_threshold, direction_layer=direction_layer, name=out_name,
                         include_score_layers=include_score_layers, candidate_layers=parse_layers(candidate_layers))
    lines = [f"{d['path']}(L{d['score_layer']}、目標 {d['target']:g}、A {d['n_a']} 例・効きにくい例 {d['hard']['n']}、"
             f"無介入 A {d['base_a_mean']:.2f} / B {d['base_b_mean']:.2f}、{d['seconds']:.0f} 秒)",
             f"  {'数':>3} {'最大s':>6} {'残る':>5} {'A最大':>7} {'B直交':>7} {'B変化':>7} {'秒':>5}  組"]
    for r in d["table"]:
        if r.get("skipped"):
            lines.append(f"  {r['n_heads']:>3}  {r['skipped']}")
            continue
        lines.append(f"  {r['n_heads']:>3} {r['s_max']:>6.2f} {r['remain']:>5} {r['a_max']:>7.2f} {r['b_drift_orth']:>7.4f} "
                     f"{r['b_side_delta']:>+7.2f} {r['seconds']:>5.0f}  " + " ".join(r["units"])
                     + ("" if r["reached"] else "  (目標に届かない)") + ("  ← 推す" if r["recommended"] else ""))
        lines += _allocate_note(r, d)
    if d["recommended"] is not None:
        rp = next(r["recipe"] for r in d["table"] if r.get("recommended"))
        lines.append(f"レシピ: {rp}。確かめる: gne eval --recipe {rp} --random-controls 16")
    return d, "\n".join(lines), 0


@gne_command(app, "behave")
def behave_cmd(c: Common, holder: dict,
               prompts: Path = typer.Option(..., "--prompts", exists=True, help="質問集(JSON / YAML の、文字列か会話の履歴のリスト。対照集合なら A 側の質問)"),
               recipe: Optional[list[Path]] = typer.Option(None, "--recipe", help="当てるレシピ(複数なら重ねる)"),
               match: Optional[str] = typer.Option(None, "--match", help="返事に一致を数える正規表現"),
               judge: Optional[Path] = typer.Option(None, "--judge", exists=True, help="読み手への文面のファイル({q} に質問、{a} に返事)"),
               labels: str = typer.Option("R,P,C", "--labels", help="読み手の判定の文字(返事の最後に出てきたものを採る)"),
               max_tokens: int = typer.Option(400, "--max-tokens"),
               thinking: bool = typer.Option(False, "--thinking", help="思考ありで答えさせ、</think> の後だけを数える"),
               temperature: float = typer.Option(0.0, "--temperature", help="0 なら greedy"),
               seeds: Optional[str] = typer.Option(None, "--seeds", help="温度ありのとき、種ごとに 1 回ずつ(1,2,3)"),
               name: str = typer.Option("behave", "--name", help="保存名")):
    """振る舞いを数える: レシピを当てて質問集に答えさせ、正規表現の一致・読み手の判定・空・繰り返し・長さの上限で切れた数を出す。
    返事は <root>/behave/<model_id>/<名前>.json に残し、画面には数だけを出す。長さで結果が変わることがあるので、--max-tokens は十分に取る。"""
    s = holder["session"] = c.session()
    d = s.behave(prompts, recipes=list(recipe or []), match=match, judge=judge.read_text(encoding="utf-8") if judge else None,
                 labels=labels, max_tokens=max_tokens, thinking=thinking, temperature=temperature, seeds=_ints(seeds), name=name)
    lines = [f"{d['path']}({d['n_prompts']} 問、{d['max_tokens']} トークン" + ("、思考あり" if thinking else "")
             + (f"、温度 {temperature:g}" if temperature > 0 else "、greedy") + ")"]
    for r in d["runs"]:
        c_ = r["counts"]
        lines.append(f"  種 {r['seed']}: " + (f"一致 {c_['match']}・" if "match" in c_ else "")
                     + (" ".join(f"{k} {v}" for k, v in c_["labels"].items()) + "・" if "labels" in c_ else "")
                     + f"空 {c_['empty']}・繰り返し {c_['loop']}・切れた {c_['at_limit']}・平均 {c_['tokens']:g} トークン")
    return d, "\n".join(lines), 0


@app.command("recipe-combine")
def recipe_combine(recipes: list[Path] = typer.Argument(..., exists=True, help="重ねるレシピ(2 つ以上)"),
                   out: Path = typer.Option(..., "--out", "-o", help="書き出すレシピ"),
                   name: Optional[str] = typer.Option(None, "--name", help="レシピの名前(既定は出力のファイル名)")):
    """複数のレシピを 1 つにまとめる。方向の違う CHiPS の組を重ねるときなどに使う(各ユニットは自分の方向を持つ)。"""
    from .steer import Recipe, combine_recipes
    r = combine_recipes([Recipe.load(p) for p in recipes], name=name or out.stem)
    r.save(out)
    dirs = sorted({u.direction for u in r.units if u.mode in ("proj", "keyed") and u.direction})
    keys = sorted({u.key for u in r.units if u.mode == "keyed" and u.key})      # CHANGE-07
    print(f"{out}({len(r.units)} ユニット" + (f"、方向 {', '.join(dirs)}" if dirs else "")
          + (f"、鍵 {', '.join(keys)}" if keys else "") + ")")


@gne_command(headproj_app, "ablate", "headproj export")
def headproj_ablate(c: Common, holder: dict,
                    direction: str = typer.Option(..., "--direction", help="gne direction で作った方向の名前"),
                    layers: str = typer.Option("all", "--layers", help="消す層: all / 1-47 / 1,5,9-12"),
                    per_layer: bool = typer.Option(False, "--per-layer", help="層ごとに、その層で測った r_L を消す(既定は基準層の r)"),
                    scale: float = typer.Option(1.0, "--scale", help="消す割合(1 で全部)"),
                    out: Optional[Path] = typer.Option(None, "--out", "-o", help="出力(既定 <root>/export/<方向>-heads-ablate.lora.gguf)"),
                    bake: bool = typer.Option(False, "--bake", help="LoRA ではなく、残差に書き込む重み行列を書き換えた GGUF を作る(分割のままでも可)")):
    """attention の書き込みだけから方向を消す(abliteration の attention 部分)。指定した層の全ヘッドに proj(s=--scale)を当てる
    レシピと同じで、LoRA は層ごとに 1 本。MLP が MoE で gne ablate の LoRA が使えないモデルでも使える。"""
    from .steer import Recipe, Unit
    s = holder["session"] = c.session()
    arch = s.weights.arch
    Ls = parse_layers(layers) or list(range(s.meta.n_layer))
    units = [Unit(L, None, 0.0, "head", mode="proj", k=float(scale), head=h, direction=direction)
             for L in Ls if arch.layers[L].head_dim and arch.layers[L].head_writer for h in range(arch.layers[L].n_head)]
    if not units:
        raise GneError("BAD_ARGS", "指定した層にヘッドがありません")
    rec = Recipe(units, s.meta.model_id, name=f"{direction}-heads-ablate", default_direction=direction,
                 direction_layer="own" if per_layer else "base")
    out = out or s.store.export_dir() / (f"{direction}-heads-ablate.gguf" if bake else f"{direction}-heads-ablate.lora.gguf")
    d = s.headproj_export(rec, out, bake=bake, requant="keep" if bake else "q8_0")
    lines = [f"{d['path']}({d['bytes'] / 1e6:.2f} MB、{'焼き込み' if bake else 'LoRA'}、{len(units)} ヘッド・{len({u.layer for u in units})} 層、"
             f"{'層ごとの r_L' if per_layer else '基準層の r'}、割合 {scale:g})"]
    if not bake:
        lines.append("使い方: llama-cli -m <model> --lora " + d["path"])
    return d, "\n".join(lines), 0


@gne_command(headproj_app, "export", "headproj export")
def headproj_export(c: Common, holder: dict, recipe: Optional[Path] = _RECIPE, unit: Optional[list[str]] = _UNIT,
                    out: Optional[Path] = typer.Option(None, "--out", "-o", help="出力(既定 <root>/export/<レシピ名>.lora.gguf、--bake なら -headproj.gguf。keyed だけのレシピは <レシピ名>-keyed.lora.gguf / -keyed.gguf)"),
                    bake: bool = typer.Option(False, "--bake", help="LoRA ではなく、書き換えた重み行列(残差に書き込むもの)だけ量子化し直した GGUF を作る"),
                    in_path: Optional[Path] = typer.Option(None, "--in", help="--bake の元の GGUF(既定 --model)"),
                    requant: Optional[str] = typer.Option(None, "--requant", help="--bake で書き換えたテンソルの型: auto / q8_0 / f16 / keep(元が F32・F16・BF16 ならその型。専門家のテンソル ffn_down_exps(keyed の expert:e)は元の型を保ち、編集しない専門家はバイトのまま。K-quant・IQ の専門家は q8_0、分割 GGUF なら元の型)。既定は q8_0、keyed を含むなら auto: テンソルごとに、元が Q8_0 なら Q8_0 のまま(F16 にすると計算方法が変わり、LoRA 版とかえってずれた)、それ以外(K-quant・IQ・F16 など)は f16。Q8_0 のままでは rank 1 の小さな編集の 1 割以上が丸めに消えるテンソル(結果の edit_kept < 0.9)だけ f16 にする。専門家のテンソルも同じ規則(f16 にすると全体が F16 になって大きくなる)"),
                    key: Optional[str] = typer.Option(None, "--key", help="#KEY を省いた keyed のユニットの鍵(gne direction key で作ったもの)"),
                    into: Optional[Path] = typer.Option(None, "--into", help="--bake: 焼き込む先の別の GGUF(同じ構造の量子化していない F16 / BF16 など)。方向はこのモデル(--model)のものを使う。GPU は使わない"),
                    via_f16: bool = typer.Option(False, "--via-f16", help="--bake: F16 で書いて llama-quantize で元の型に戻す"),
                    imatrix: Optional[Path] = typer.Option(None, "--imatrix"), llama_quantize: Optional[str] = typer.Option(None, "--llama-quantize"),
                    verify: bool = typer.Option(False, "--verify", help="--bake: 焼き込み版と LoRA 版を比べる(V26・V33: 方向への射影の値・KL・最尤トークン・ppl。最後の層の射影の値の合格の線には、書き込みがほぼ 0 の LoRA と LoRA なしの差 = 計算方法の違いだけで出る誤差を足す)"),
                    contrast: Optional[Path] = typer.Option(None, "--contrast", help="--verify の射影の値に使う対照集合(既定は方向を作った対照集合)"),
                    direction: Optional[str] = typer.Option(None, "--direction", help="@NAME を省いた proj・keyed のユニットの方向"),
                    direction_layer: Optional[str] = typer.Option(None, "--direction-layer", help="own(ヘッドの層の方向)/ base(基準層の方向)"),
                    lora_f16: bool = typer.Option(False, "--lora-f16", help="LoRA を F16 で書く(大きさが半分。精度は十分)"),
                    strip_meta: bool = typer.Option(False, "--strip-meta", help="gne の記録(gne.* のメタデータ・レシピ・名前の印・横の .headproj.json)を残さない(人に渡す版)")):
    """proj のユニット(L:Hh:sS@NAME、レシピの mode: proj)を LoRA アダプタ GGUF か焼き込み GGUF に書き出す(CHANGE-06 E、gne gain の別名)。
    同じ層・同じ方向のヘッドは階数 1 の LoRA に畳む。スケールパッチ(CHANGE-05 D)では表せないので、焼き込みは書き換える
    テンソルだけ量子化し直す。鍵つきの書き込み(keyed、L:ffn:keyed:aα@NAME#KEY、CHANGE-07)も受ける(層・鍵ごとに階数 1。
    配るなら LoRA が第一候補。焼き込みの既定は auto: 元が Q8_0 のテンソルは Q8_0 のまま、それ以外は f16)。"""
    if in_path and c.model and in_path.resolve() != c.model.resolve():
        raise GneError("BAD_ARGS", "--in と --model が違うファイルです")
    s = holder["session"] = c.session(in_path)
    r = s.recipe(units=unit, path=recipe, default_direction=direction, direction_layer=direction_layer, default_key=key)
    out = out or s.default_export_path(r, bake=bake, kind="headproj")
    d = s.headproj_export(r, out, bake=bake, requant=requant, via_f16=via_f16, imatrix=str(imatrix) if imatrix else None,
                          llama_quantize=llama_quantize, verify=verify, contrast=contrast, strip_meta=strip_meta, into=into,
                          lora_f16=lora_f16)
    lines = [f"{d['path']}({d['bytes'] / 1e6:.2f} MB、{'焼き込み' if bake else 'LoRA'}"
             + ("、gne の記録なし" if d.get("strip_meta") else "") + ")"]
    if d.get("local_path_keys"):
        lines.append("注意: メタデータに手元のパスらしい文字列が残っています: " + ", ".join(d["local_path_keys"]))
    for L, x in (d.get("proj_layers") or {}).items():
        lines.append(f"  L{L}: " + ", ".join(f"H{int(h)} s{sv:g}@{dn}" for h, sv, dn in x["heads"])
                     + f"(階数 {x['rank']}" + ("" if x["folded"] else "、方向が混ざっていて 1 に畳めない") + ")")
    lines += _keyed_export_lines(d, requant_line=not (bake and d.get("parts")))
    if bake and d.get("parts"):                 # 分割 GGUF: 同じ型で書き戻した
        how = {"keep": "元と同じ型で書き戻し", "auto": "auto で書き戻し(Q8_0 は元の型のまま、それ以外は F16"
               + (f"。{len(d['fallback'])} テンソルは編集が埋もれるので F16" if d.get("fallback") else "") + ")"}
        lines.append(f"書き換え: {len(d['modified'])} テンソルを" + how.get(d["requant"], f"{d['requant']} で書き直し")
                     + "(量子化し直しで失った分は編集の大きさの最大 "
                     f"{d['requant_error_max'] * 100:.1f}%)、分割 {len(d['parts'])} ファイル。記録 {d['sidecar']}")
        if d.get("expert_bytes_added"):
            lines.append(f"  専門家のテンソルで {d['expert_bytes_added'] / 1e6:.1f} MB 大きくなりました(編集しない専門家も変換。"
                         "--requant keep なら元の型のまま)")
    elif bake:
        lines.append("書き換え: " + ", ".join(_mod_text(m) for m in d["modified"]))
    else:
        lines.append("使い方: llama-cli -m <model> --lora " + d["path"])
    ok = True
    v = d.get("verify")
    if v:
        lines.append(f"{'V33' if r.keyed_units and not r.proj_units else 'V26'}: 射影の値 {'一致' if all(x['ok'] for x in v['scores'].values()) else '不一致'}、"
                     f"最初のトークンの KL {v['kl_first_token']:.2e}、最尤トークンの一致 {v['top1_match'] * 100:.0f}%、"
                     f"ppl の差 {_f(v['ppl_diff_pct'], '+.3f')}%({'合格' if v['pass'] else '不合格'})"
                     + (f"。分かれた例の差 {', '.join(f'{g:.3f}' for g in v['top1_mismatch_gaps'])} nats" if v["top1_mismatch_gaps"] else ""))
        lines.append("  射影の値の差の最大 / 合格の線: " + "、".join(
            f"L{k[:-1]} {k[-1]} {x['max_abs_diff']:.3g} / {x.get('threshold', x['resolution']):.3g}{'' if x['ok'] else ' ×'}"
            for k, x in v["scores"].items()))
        nz = v.get("noise")
        if nz:
            lines.append(f"  最後の層 L{nz['layer']} の合格の線 = 許容誤差(射影の値の標準偏差の 1 割)+ 計算方法の違いだけで出る誤差 "
                         f"{nz['max_abs_diff']:.3g}(書き込みがほぼ 0 の LoRA(倍率 {nz['lora_scale']:g})と LoRA なしの射影の値の差の"
                         f"最大、{nz['seconds']:.1f} 秒)")
        ok = v["pass"]
    return d, "\n".join(lines), 0 if ok else 1


@gne_command(headproj_app, "check", "headproj check")
def headproj_check(c: Common, holder: dict, layer: int = typer.Option(..., "--layer"), head: int = typer.Option(..., "--head"),
                   direction: str = typer.Option(..., "--direction", help="方向の名前"),
                   ss: str = typer.Option("1,2,6", "--s", help="確かめる s")):
    """V24・V25: proj の LoRA で、ヘッドが残差に書き込む重み行列の出力の差が numpy の計算(方向成分を s 倍ぶん引く)と合い、直交成分が変わらないか。"""
    s = holder["session"] = c.session()
    d = s.check_head_proj(layer, head, direction, ss=[float(x) for x in ss.split(",")])
    lines = [f"L{layer} H{head}({d['out_tensor']}、方向 {direction} の L{d['direction_from']}、ヘッドの書き込みと方向の |cos| の中央値 "
             f"{_f(d['dir_cos_head'], '.3f')})"]
    for r in d["rows"]:
        lines.append(f"  s {r['s']:g}: 相対誤差の中央値 {r['rel_err_median']:.2e}、方向成分 {r['par_after_over_before']:+.3f} 倍"
                     f"(期待 {1 - r['s']:+g})、直交成分の変化 {r['orth_change_rel_median']:.2e}  {'○' if r['ok'] else '×'}")
    lines.append(f"  対比 gain ×{d['gain']['k']:g}: 直交成分の変化 {_f(d['gain']['orth_change_rel_median'], '.3f')}")
    lines.append(f"V24 {'合格' if d['v24'] else '不合格'}・V25 {'合格' if d['v25'] else '不合格'}")
    return d, "\n".join(lines), 0 if d["ok"] else 1


@gne_command(heads_app, "pipeline", "heads pipeline")
def heads_pipeline(c: Common, holder: dict,
                   spec: Path = typer.Option(..., "--spec", help="対照集合の JSON か YAML(name・user_prompts か側ごとの prompts_a / prompts_b・system_a / system_b か recipe_a / recipe_b、任意で pipeline の設定)"),
                   target: Optional[str] = typer.Option(None, "--target", help="効きにくい例の射影の値の目標: 数、B(B 側の平均まで消す)、mid(既定、A と B の真ん中)"),
                   n_heads: Optional[str] = typer.Option(None, "--n-heads", help="比べる組のヘッド数(既定 6,5,4)"),
                   max_examples: Optional[int] = typer.Option(None, "--max-examples", help="止めて測る段階で使う質問の数(既定 64、等間隔に間引く)"),
                   positions: Optional[str] = typer.Option(None, "--positions", help="方向を作る位置: last(既定)/ response など"),
                   top_layers: Optional[int] = typer.Option(None, "--top-layers"), hard: Optional[int] = typer.Option(None, "--hard"),
                   include_score_layers: bool = typer.Option(False, "--include-score-layers", help="射影の値を測る層とそれより後のヘッドも候補にする(既定は外す。測る量を直接動かすだけで、振る舞いは変わりにくい)"),
                   fresh: bool = typer.Option(False, "--fresh", help="作り済みの方向と因果の結果を使い回さず、最初から作り直す"),
                   yes: bool = _YES):
    """対照集合から、方向づくり → 基準層 → 因果スクリーニング(層ごと → ヘッド 1 個ずつ止める)→ 強さの配分(効き目に比例した
    最初の組を二分探索 → 射影の値が残りやすい質問で効率を測って 0 から配り直す → ヘッド数を変えた組の比較)を 1 回で回す。"""
    s = holder["session"] = c.session()
    over = {"target": target, "n_heads": _ints(n_heads), "max_examples": max_examples, "positions": positions,
            "top_layers": top_layers, "hard": hard, "resume": False if fresh else None,
            "include_score_layers": True if include_score_layers else None}
    _gate(s, "head_pipeline", {"spec": str(spec), **{k: v for k, v in over.items() if v is not None}}, yes, None)
    d = s.heads_pipeline(spec, **over)
    sc, ca, al = d["score"], d["causal"], d["allocate"]
    lines = [f"{d['path']}({d['n_prompts']} 問、{d['seconds']:.0f} 秒"
             + (f"、作り済みの{'・'.join({'direction': '方向', 'causal': '因果'}[x] for x in d['reused'])}を使い回した" if d.get("reused") else "")
             + ")",
             f"1. 方向 {d['name']}({d['direction']['positions']}、{d['direction']['seconds']:.0f} 秒)",
             f"2. 基準層 L{sc['base_layer']}(cohen_d 最大、止めて測る段階は {sc['n'].get('A')} 問)",
             f"3. 因果: 測る層 L{','.join(map(str, ca['score_layers']))}、{ca['n_rows']} ヘッド、上位 " +
             " ".join(f"L{r['layer']}H{r['head']}({r['causal_max']:+.2f}{'、壊すだけなので除外' if r.get('excluded') else ''})" for r in ca["top"][:6])
             + f"({ca['seconds']:.0f} 秒)"
             + "".join(f"\n   層 L{r['layer']} は止めると中立の入力も大きく崩れる(ずれ {r['drift_orth_b']:.2f})ので選ばなかった"
                       for r in (ca.get("layers") or []) if r.get("excluded"))]
    if al.get("initial"):
        lines.append(f"4. 最初の組(効き目に比例、全体を二分探索): " + " ".join(al["initial"]["heads"]))
    lines.append(f"5. 効率で配り直した組(目標 {d['target']:.2f} = {d['target_from']}):")
    for r in al["table"]:
        if r.get("skipped"):
            lines.append(f"   {r['n_heads']} 個: {r['skipped']}")
            continue
        lines.append(f"   {r['n_heads']} 個: 最大 s {r['s_max']:.2f}、残る {r['remain']}/{al['n_a']}、B 直交 {r['b_drift_orth']:.4f}  "
                     + " ".join(r["units"]) + ("  ← 推す" if r["recommended"] else "") + ("" if r["reached"] else "(目標に届かない)"))
    if d["recommended"]:
        lines.append(f"レシピ: {d['recommended']['recipe']}。確かめる: gne eval --recipe {d['recommended']['recipe']} --random-controls 16")
    return d, "\n".join(lines), 0



def _candidates_text(d: dict) -> str:
    j = d["judge"]
    ho = d.get("held_out") or {}
    lines = [f"{d['path']}({len(d['candidates'])} 候補 × A {len(d['prompts']['A'])} 問・B {len(d['prompts']['B'])} 問、"
             + ("greedy" if d["sampling"] is None else f"温度 {d['sampling']['temperature']:g}・種 {d['sampling']['seed']}")
             + (f"、重ねたレシピ {Path(d['with_recipe']).name}" if d.get("with_recipe") else "")
             + (f"、{d['seconds']:.0f} 秒" if d.get("seconds") else "") + ")"]
    if not (ho.get("A") and ho.get("B")):
        lines.append("  注意: 調整に使った質問で生成しています(" + "・".join(k for k in "AB" if not ho.get(k)) +
                     " 側)。調整に使っていない質問を --prompts で渡すと、決めるのに使えます")
    sc = d.get("scores") or {}
    has_sc = bool(sc)
    hdr = f"  {'候補':<12} {'側':>2} {'向き':<4} {'一致':>5} {'d4':>5} {'反対側の変化':>6}"
    if has_sc:
        hdr += f" {'P':>5} {'C':>5} {'目的':>5}"
    lines.append(hdr)
    goal_ja = {"remove": "消す", "add": "強める", None: "—"}
    for c in d["candidates"]:
        for side in c["judge_sides"]:
            lo, other = side.lower(), "b" if side == "A" else "a"
            ln = (f"  {c['name']:<12} {side:>2} {goal_ja[c.get('goal')]:<4} {_metric(c.get('rate_' + lo), '.0%'):>5} "
                  f"{c['d4_' + lo]:>5.2f} {_metric(None if c['name'] == 'base' else 1 - c['same_as_base_' + other], '.0%'):>6}")
            if has_sc:
                v = (sc.get(c["name"]) or {}).get(side)
                g = c.get("goal") or ("remove" if side == "A" else "add")
                ln += f" {v['P']:>5.2f} {v['C']:>5.2f} {v[g]:>5.2f}" if v else f" {'—':>5} {'—':>5} {'—':>5}"
            lines.append(ln + ("  ← allocate の推し" if c.get("recommended") and c.get("scale") == 1.0 else ""))
    lines.append("  一致 = validate.match に合う返事の割合、d4 = 繰り返しの少なさ、反対側の変化 = 採点しない側で無介入と違う返事の割合"
                 "(同じ種なので、変わらなければ一致する)")
    if has_sc:
        lines.append("  目的 = 消す向きは min(6 − P, C)、強める向きは min(P, C) の平均")
        if d.get("scores_missing"):
            lines.append(f"  採点の無い番号: {len(d['scores_missing'])} 件")
        for g, b in (d.get("best") or {}).items():
            c = next(x for x in d["candidates"] if x["name"] == b["name"])
            lines.append(f"{goal_ja[g]}向きでいちばん良い: {b['name']}(目的 {b['value']:.2f}、無介入 {_f(b['base'], '.2f')})。"
                         f"確かめる: gne eval --recipe {c['recipe']} --doses {c['scale']:g}")
    else:
        lines.append(f"採点: {j['round']} の {j['n']} 本を {j['rubric']} の基準で付けて、{j['scores']} に「番号 P C」を書く")
        lines.append(f"集計: gne heads candidates --tally {d['name']}")
    return "\n".join(lines)


@gne_command(heads_app, "candidates", "heads candidates")
def heads_candidates(c: Common, holder: dict,
                     alloc: Optional[str] = typer.Option(None, "--alloc", help="heads allocate / pipeline の配分の結果(名前か .alloc.json)"),
                     tally: Optional[str] = typer.Option(None, "--tally", help="採点を書いた候補(名前か .candidates.json)を集計する"),
                     prompts: Optional[Path] = typer.Option(None, "--prompts", help="調整に使っていない質問(YAML / JSON のリストか {A: [...], B: [...]})"),
                     scales: str = typer.Option("0.5,1,1.5,2,3", "--scales", help="組の s 全体にかける倍率(負は B 側で強める向き)"),
                     n_heads: Optional[str] = typer.Option(None, "--n-heads", help="使う組のヘッド数(既定は比較表の全部)"),
                     max_prompts: int = typer.Option(24, "--max-prompts", help="側ごとの質問の上限(多ければ等間隔に間引く)"),
                     max_tokens: int = typer.Option(120, "--max-tokens"),
                     sampling: str = typer.Option("llama_server", "--sampling", help="greedy・llama_server・common か、値の JSON"),
                     sample_seed: int = typer.Option(0, "--sample-seed", help="サンプリングの種(候補・無介入で同じ種を使う)"),
                     judge_b: bool = typer.Option(False, "--judge-b", help="B 側(ふつうの状態)の返事も採点のファイルに入れる"),
                     contrast: Optional[Path] = typer.Option(None, "--contrast", help="対照集合(既定は方向を作った対照集合)"),
                     out_name: Optional[str] = typer.Option(None, "--out-name", help="保存名(既定は配分の名前)"),
                     with_recipe: Optional[Path] = typer.Option(None, "--with-recipe", help="全部の候補(無介入の base も)に重ねるレシピ(ニューロンの種火など)"),
                     rubric: Optional[Path] = typer.Option(None, "--rubric", help="採点基準のファイル(既定は目標の振る舞いから作る)"),
                     yes: bool = _YES):
    """強さの候補づくり: 配分の比較表の組 × 全体の倍率(と無介入)で、A 側と B 側の質問に返事を作る。正の倍率は A 側の振る舞いを
    消す向き、負の倍率(例 --scales -0.5,-1)は B 側で強める向きで、それぞれの側の返事を、どの候補か伏せた採点用のファイルに書く。
    判定の一致率・繰り返し・反対側の変化も出す。採点を書いたら --tally で集計する。"""
    s = holder["session"] = c.session()
    if tally:
        if alloc:
            raise GneError("BAD_ARGS", "--alloc と --tally は一緒に使えません")
        d = s.heads_candidates_tally(tally)
        return d, _candidates_text(d), 0
    if not alloc:
        raise GneError("BAD_ARGS", "--alloc(候補を作る)か --tally(採点を集計する)を指定してください")
    sm = sampling.strip()
    sm_v = json.loads(sm) if sm.startswith("{") else sm
    sc = [float(x) for x in scales.split(",") if x.strip()]
    nh = _ints(n_heads) if n_heads else None
    _gate(s, "heads_candidates", {"alloc": alloc, "scales": sc, "n_heads": nh, "max_prompts": max_prompts,
                                  "max_tokens": max_tokens}, yes, None)
    d = s.heads_candidates(alloc, prompts=prompts, scales=sc, n_heads=nh, max_prompts=max_prompts, max_tokens=max_tokens,
                           sampling=sm_v, seed=sample_seed, judge_b=judge_b, contrast=contrast, name=out_name,
                           with_recipe=with_recipe, rubric=rubric)
    return d, _candidates_text(d), 0


@gne_command(heads_app, "refine", "heads refine")
def heads_refine(c: Common, holder: dict,
                 name: str = typer.Option(..., "--name", help="配分の結果(一括の名前か heads allocate の保存名)"),
                 n_heads: Optional[int] = typer.Option(None, "--n-heads", help="残りを測る組のヘッド数(既定は推す組)"),
                 new_heads: str = typer.Option("3,2,1", "--new-heads", help="追加する組のヘッド数(カンマ区切り)"),
                 out_name: Optional[str] = typer.Option(None, "--out-name", help="保存名(既定 <name>_r1、別の組なら _r2 …)"),
                 fresh: bool = typer.Option(False, "--fresh", help="作り済みの方向と因果の結果を使い回さない"),
                 yes: bool = _YES):
    """組を当てても目標まで下がらない A 側の質問(残り)だけで、追加のヘッドを探す。残りの質問と B 側の質問で、組を当てたまま
    方向を作り直し、一括(基準層 → 因果 → 配分)で足すヘッドと強さを決め、元の組と合わせたレシピを作って残りを測り直す。"""
    s = holder["session"] = c.session()
    nh = _ints(new_heads) or [3, 2, 1]
    _gate(s, "heads_refine", {"name": name, "new_heads": nh}, yes, None)
    d = s.heads_refine(name, n_heads=n_heads, new_heads=nh, out_name=out_name, resume=not fresh)
    rm, cb, pp = d["remaining"], d["combined"], d["pipeline"]
    lines = [f"{d['path']}({d['seconds']:.0f} 秒)",
             f"1. {d['n_heads']} ヘッドの組 " + " ".join(d["base_units"]) + f" で、L{d['score_layer']} の射影の値が {d['threshold']:.2f} を超える"
             f" A 側の質問: {rm['n']}/{rm['n_a']} 問" + (f"(少ないので射影の値の高い順に {rm['used']} 問で方向を作った)" if rm["padded"] else ""),
             f"2. 残りの質問で一括 → 方向 {pp['direction']}" + (f"(作り済みの{'・'.join({'direction': '方向', 'causal': '因果'}[x] for x in pp['reused'])}を使い回した)" if pp["reused"] else "")]
    for r in pp["table"]:
        if r.get("skipped"):
            lines.append(f"   {r['n_heads']} 個: {r['skipped']}")
            continue
        lines.append(f"   {r['n_heads']} 個: 最大 s {r['s_max']:.2f}、残る {r['remain']}  " + " ".join(r["units"])
                     + ("  ← 推す" if r["recommended"] else "") + ("" if r["reached"] else "(目標に届かない)"))
    rows = d.get("combined_rows") or []
    if rows:
        lines.append(f"3. 元の組と合わせた組(元の方向 L{d['score_layer']} で測り直す。合わせる前: 残り {rm['n']}/{rm['n_a']} 問、"
                     f"B 側の直交成分のずれ {rows[0]['b_drift_orth_base']:.4f})")
        for c in rows:
            lines.append(f"   + {c['n_new']} 個(計 {c['n_heads']} ヘッド): 残り {c['remain_after']} 問、A の平均 {c['a_mean_before']:.2f} → "
                         f"{c['a_mean_after']:.2f}、B のずれ {c['b_drift_orth']:.4f}  {c['recipe']}"
                         + ("  ← 代表" if cb and c["recipe"] == cb["recipe"] else ""))
        lines.append(f"代表は {Path(cb['recipe']).with_name(d['name'] + '.combined.yaml')} にも置いた"
                     + ("(推す組)" if cb["chosen_by"] == "recommended" else "(目標に届いた組が無いので、合わせた残りが最も少ない組)"))
        lines.append(f"確かめる: gne eval --recipe {cb['recipe']} --random-controls 16")
    else:
        lines.append("3. 追加の組を作れなかったので、合わせたレシピはありません")
    return d, "\n".join(lines), 0


def _fieldtest_text(d: dict) -> str:
    def fmt(section, key, v):
        if v is None:
            return "—"
        if section == "ppl":
            return f"{v:.3f}"
        if key == "tokens":
            return f"{v:.0f}"
        if key == "tail_distinct":
            return f"{v:.2f}"
        return f"{v * 100:.1f}%"
    title = {"long": f"長い作業({d['settings']['n_long']} 問、{d['settings']['long_tokens']} トークンまで)",
             "tools": f"ツール呼び出し({d['settings']['n_tools']} 問)", "ppl": "perplexity"}
    lines = [f"{d['path']}({d['seconds'] / 60:.0f} 分)", f"対象: {d['target']['file']}" + (f" + LoRA {Path(d['target']['lora']).name}" if d["target"]["lora"] else "")]
    if d["reference"]:
        lines.append(f"比較: {d['reference']['file']}")
    lines += [f"注意: {w}" for w in d.get("warnings") or []]
    rows = d["compare"]
    if rows is None:                                  # 比べる相手が無いとき: 対象の数字だけ
        run = d["runs"][-1]
        rows = []
        for sec in ("long", "tools"):
            rows += [{"section": sec, "metric": k, "label": k, "reference": None, "target": v, "worse": False}
                     for k, v in (run.get(sec) or {}).items() if not k.startswith("n")]
        for name, m in (run.get("bench") or {}).items():
            rows += [{"section": f"bench:{name}", "metric": k, "label": k, "reference": None, "target": v, "worse": False} for k, v in m.items() if k != "n"]
        for name, m in (run.get("prompts") or {}).items():
            rows += [{"section": f"prompts:{name}", "metric": k, "label": k, "reference": None, "target": v, "worse": False}
                     for k, v in m.items() if k != "n"]
        rows += [{"section": "ppl", "metric": k, "label": k, "reference": None, "target": v, "worse": False} for k, v in (run.get("ppl") or {}).items()]
    cur = None
    for r in rows:
        if r["section"] != cur:
            cur = r["section"]
            key = cur.split(":", 1)[-1]
            head = title.get(cur) or (f"ベンチマーク {key}({(d['settings'].get('bench') or {}).get(key, '?')} 問、{d['settings'].get('bench_tokens')} トークンまで)" if cur.startswith("bench:")
                                      else f"質問集 {key}({d['settings']['prompts'].get(key, '?')} 問、{d['settings']['prompt_tokens']} トークンまで)")
            lines.append(head)
            lines.append(f"  {'':<26} {'比較':>9} {'対象':>9}")
        lines.append(f"  {r['label']:<26} {fmt(r['section'], r['metric'], r['reference']):>9} {fmt(r['section'], r['metric'], r['target']):>9}"
                     + (f"  片方だけ正解: 比較 {r['only_reference']} / 対象 {r['only_target']}(p = {r['p']:.3f})" if "only_target" in r else "")
                     + ("  ← 悪化" if r.get("worse") else ""))
    if d["same_level"] is not None:
        lines.append("結論: すべて同程度(割合は 2 問ぶん、perplexity は 1% までの差を同程度とみなす)" if d["same_level"]
                     else "結論: 悪化した項目があります: " + ", ".join(d["worse"]))
    lines.append(f"生成した文章: {d['replies_path']}")
    return "\n".join(lines)


@gne_command(app, "fieldtest")
def fieldtest_cmd(c: Common, holder: dict,
                  against: Optional[Path] = typer.Option(None, "--against", help="比べるファイル(例 元のモデル)。省くと --model だけ測る(--lora があれば LoRA なしと比べる)"),
                  lora: Optional[str] = typer.Option(None, "--lora", help="--model に当てる LoRA(パス か パス:倍率)"),
                  name: Optional[str] = typer.Option(None, "--name", help="保存名(既定はファイル名)"),
                  prompts: Optional[list[str]] = typer.Option(None, "--prompts", help="自分の質問集 名前=ファイル(JSON か YAML のリスト、何度でも)"),
                  match: Optional[str] = typer.Option(None, "--match", help="質問集の返事(本文)に当てる正規表現。一致率を出す"),
                  long_tasks: Optional[Path] = typer.Option(None, "--long-tasks", help="長い作業の課題(既定 configs/fieldtest/long_tasks.json)"),
                  tool_tasks: Optional[Path] = typer.Option(None, "--tool-tasks", help="ツールの課題(既定 configs/fieldtest/tool_tasks.json)"),
                  long_tokens: int = typer.Option(3072, "--long-tokens"), tool_tokens: int = typer.Option(1536, "--tool-tokens"),
                  prompt_tokens: int = typer.Option(1536, "--prompt-tokens"), temperature: float = typer.Option(0.6, "--temperature"),
                  presence_penalty: float = typer.Option(0.0, "--presence-penalty", help="サンプリングの presence penalty(モデルカードの設定に合わせる)"),
                  n_parallel: int = typer.Option(4, "--n-parallel", help="同時に生成する本数"),
                  reasoning_effort: Optional[str] = typer.Option("medium", "--reasoning-effort", help="チャットテンプレートに渡す値"),
                  template_kwargs: Optional[str] = typer.Option(None, "--template-kwargs", help='チャットテンプレートに渡す JSON(例 {"enable_thinking": false} で思考なし)'),
                  llama_server: Optional[str] = typer.Option(None, "--llama-server", envvar="GNE_LLAMA_SERVER", help="llama-server のパス"),
                  server_args: Optional[str] = typer.Option(None, "--server-args", help="llama-server に足す引数"),
                  ppl: bool = typer.Option(True, "--ppl/--no-ppl", help="perplexity も測る"),
                  skip_long: bool = typer.Option(False, "--skip-long"), skip_tools: bool = typer.Option(False, "--skip-tools"),
                  bench: Optional[str] = typer.Option(None, "--bench", help="ベンチマーク(カンマ区切り): gsm8k(既定 250 問)・mmlu_pro(既定 300 問)・math500(既定 250 問)・ifeval(既定 250 問。math500 と ifeval は bench extra が要る)。データは取得して <root>/fieldtest/data に置く"),
                  bench_n: Optional[int] = typer.Option(None, "--bench-n", help="ベンチマークの問題数(種を固定して選ぶ)"),
                  bench_tokens: int = typer.Option(4096, "--bench-tokens", help="ベンチマークの返事の上限"),
                  yes: bool = _YES):
    """実地テスト: 本番と同じ動かし方(llama-server、思考あり、サンプリングあり)で、長い作業・ツール呼び出し・質問集・perplexity を測り、
    ファイルどうしを比べる。eval は同じファイルの倍率違いを短い問題で見るもの。こちらは焼き込んだファイルをそのまま測れる。"""
    s = holder["session"] = c.session()
    args = {"against": str(against) if against else None, "lora": lora, "prompts": list(prompts or []), "long_tokens": long_tokens,
            "prompt_tokens": prompt_tokens, "n_parallel": n_parallel, "skip_long": skip_long, "skip_tools": skip_tools,
            "long_tasks": str(long_tasks) if long_tasks else None, "tool_tasks": str(tool_tasks) if tool_tasks else None,
            "bench": bench, "bench_n": bench_n, "bench_tokens": bench_tokens}
    _gate(s, "fieldtest", args, yes, None)
    try:
        tkw = json.loads(template_kwargs) if template_kwargs else None
    except ValueError as e:
        raise GneError("BAD_ARGS", f"--template-kwargs は JSON で書いてください: {e}") from e
    d = s.fieldtest(against=against, lora=lora, name=name, template_kwargs=tkw, prompts=list(prompts or []), match=match, long_tasks=long_tasks, tool_tasks=tool_tasks,
                    long_tokens=long_tokens, tool_tokens=tool_tokens, prompt_tokens=prompt_tokens, temperature=temperature, presence_penalty=presence_penalty, n_parallel=n_parallel,
                    reasoning_effort=reasoning_effort, llama_server=llama_server, server_args=server_args, ppl=ppl, skip_long=skip_long, skip_tools=skip_tools,
                    bench=bench, bench_n=bench_n, bench_tokens=bench_tokens)
    return d, _fieldtest_text(d), 0


# --- 計算で作るニューロン(mode: nw)--------------------------------------------------------------------------

@gne_command(nw_app, "neurons", "nw neurons")
def nw_neurons(c: Common, holder: dict,
               layers: str = typer.Option(..., "--layers", help="測る層(例 11,18,23)"),
               top: int = typer.Option(20, "--top", help="層ごとに出す数(働きの小さい順)"),
               sort: str = typer.Option("sum", "--sort", help="並べ方: sum(最大の順位 + 平均の順位、既定)/ max / mean"),
               max_tokens: int = typer.Option(96, "--max-tokens", help="ふつうの質問への返事を生成する長さ"),
               corpus_tokens: int = typer.Option(2048, "--corpus-tokens", help="既定のコーパスから読むトークン数(0 で読まない)"),
               iv: Optional[list[str]] = typer.Option(None, "--iv", help="この介入(レシピの名前かパス、またはユニットの書き方)を当てた状態で測る(例: CHiPS のレシピ)")):
    """各ニューロンのふだんの働き(|ゲート後の活性| × 出力の列の長さ = 1 位置で残差に書く長さ)を測り、働きの小さい順の表を出す。
    上書きするニューロンは、この表を見て選ぶ(番号を nw のユニットに書く)。上書きすると、そのニューロンの元の働きは無くなる。"""
    Ls = parse_layers(layers)
    if not Ls:
        raise GneError("BAD_ARGS", "--layers に層を指定してください", "例 --layers 11,18,23")
    s = holder["session"] = c.session()
    d = s.nw_neurons(Ls, top=top, sort=sort, max_tokens=max_tokens, corpus_tokens=corpus_tokens, iv=_iv_arg(iv))
    lines = [f"ふだんの働き({d['n_prompts']} 問の返事 + コーパス {d['corpus_tokens']} トークン、{d['seconds']:.0f} 秒)。並べ方: {d['sort']}"]
    for r in d["layers"]:
        om = r["out_norm_median"]
        lines.append(f"L{r['layer']}({r['positions']} 位置、MLP の出力の長さの中央値 {_f(om, '.2f')}。全ニューロンの中央値: 平均 {r['median']['mean']:.4f}・"
                     f"最大 {r['median']['max']:.3f}、いちばん小さい最大 {r['min']['max']:.3f})")
        lines.append(f"  {'番号':>6} {'平均':>8} {'最大':>7} {'最大÷出力':>9} {'順位 最大/平均':>14}")
        for x in r["rows"]:
            lines.append(f"  {x['neuron']:>6} {x['mean']:>8.4f} {x['max']:>7.3f} {_f(x['max_share'], '.1%'):>9} {x['rank_max']:>7}/{x['rank_mean']}")
    lines += [f"注: {n}" for n in d["notes"]]
    return d, "\n".join(lines), 0


@gne_command(nw_app, "key", "nw key")
def nw_key(c: Common, holder: dict,
           contrast: Path = typer.Option(..., "--contrast", help="対照集合(user_prompts = 狙う場面の質問、dont_touch = 反応させたくない質問の群)"),
           layers: str = typer.Option(..., "--layers", help="鍵を作る層(例 11,18,23)"),
           role: str = typer.Option("scene", "--role", help="scene(狙う場面の質問への返事の間ずっと反応する鍵)/ one(どの位置でもほぼ 1 を返す鍵。上限つきの形で使う)"),
           name: Optional[str] = typer.Option(None, "--name", help="鍵の名前(既定 <対照集合の name>_<role>)"),
           subset: str = typer.Option("even", "--subset", help="使う質問: even(既定)/ odd / all"),
           max_tokens: int = typer.Option(96, "--max-tokens", help="返事を生成するときの長さ"),
           pseudo: int = typer.Option(16, "--pseudo", help="比較用の鍵(質問のラベルをでたらめに入れ替えて同じ手順で作った鍵)の本数(scene だけ)"),
           iv: Optional[list[str]] = typer.Option(None, "--iv", help="この介入(レシピの名前かパス、またはユニットの書き方)を当てた状態の返事と活性で作る")):
    """計算で作るニューロンの「場面の鍵」と「ほぼ 1 を返す鍵」を作る(MLP の入力の空間、target mlpin)。
    返事の位置の区切りで反応する「位置の鍵」は gne direction key --target mlpin --positions pos:1-3 で作る。"""
    Ls = parse_layers(layers)
    if not Ls:
        raise GneError("BAD_ARGS", "--layers に層を指定してください", "例 --layers 11,18,23")
    s = holder["session"] = c.session()
    d = s.direction_key(contrast, role=role, name=name, layers=Ls, subset=subset, max_tokens=max_tokens, pseudo=pseudo,
                        iv=_iv_arg(iv), seed=c.seed, progress=s.progress)
    lines = [f"{d['path']}(鍵 {d['name']}、{d['role']}、MLP の入力の空間、{d['seconds']:.1f} 秒)"]
    for r in d["layers"]:
        if d["role"] == "one":
            lines.append(f"  L{r['layer']}: 反応の平均 {r['response_mean']:.3f} ± {r['response_sd']:.3f}(最小 {r['response_min']:.2f}・最大 {r['response_max']:.2f}、{r['rows']} 位置)")
        else:
            lines.append(f"  L{r['layer']}: kn {r['kn']:.3f}、狙う質問 {r['examples_a']}・それ以外 {r['examples_b']}、作成に使った質問での差 {r['build_selectivity']:+.3f}"
                         f"(比較用の鍵の p95 {_f(r['pseudo_selectivity_p95'], '+.3f')})")
    lines += [f"注: {n}" for n in d["notes"]]
    return d, "\n".join(lines), 0


@gne_command(nw_app, "verify", "nw verify")
def nw_verify(c: Common, holder: dict, recipe: Optional[Path] = _RECIPE, unit: Optional[list[str]] = _UNIT,
              direction: Optional[str] = typer.Option(None, "--direction", help="@NAME を省いたユニットの方向"),
              key: Optional[str] = typer.Option(None, "--key", help="#KEY を省いたユニットの鍵"),
              text: Optional[list[str]] = typer.Option(None, "--text", help="読ませる文(既定は英語と日本語の 2 文)。複数可"),
              iv: Optional[list[str]] = typer.Option(None, "--iv", help="この介入(レシピの名前かパス、またはユニットの書き方)を先に当てた状態で確かめる")):
    """nw の LoRA が式どおりに動くか。上書きしたニューロンのゲート・入力・活性が式どおりか、ほかのニューロンが変わっていないか、
    MLP の出力の変化が (新しい活性 × 新しい列) − (元の活性 × 元の列) になっているかを、層ごとに確かめる。"""
    s = holder["session"] = c.session()
    r = s.recipe(units=unit, path=recipe, default_direction=direction, default_key=key)
    d = s.nw_verify(r, texts=text or None, iv=_iv_arg(iv))
    lines = []
    for x in d["layers"]:
        lines.append(f"L{x['layer']} N{','.join(map(str, x['neurons']))}: ゲートの誤差 {x['gate_err']:.2e}、入力の誤差 {x['up_err']:.2e}、活性の誤差 {x['act_err']:.1e}、"
                     f"ほかのニューロンの変化 {x['others_changed_max']:.1e}、出力の変化の相対誤差(中央値){x['out_rel_err_median']:.2e}  {'○' if x['ok'] else '×'}")
    lines.append(f"{'合格' if d['ok'] else '不合格'}(ゲート・入力・出力の誤差 < 5%、活性の誤差 < 0.1%、ほかのニューロンの変化 < 1e-4。"
                 "ゲート・入力には、量子化のままの掛け算と float に戻した行を引く計算のずれが 1〜2% 残る)")
    return d, "\n".join(lines), 0 if d["ok"] else 1


@gne_command(nw_app, "check", "nw check")
def nw_check(c: Common, holder: dict, recipe: Optional[Path] = _RECIPE, unit: Optional[list[str]] = _UNIT,
             contrast: Path = typer.Option(..., "--contrast", help="対照集合(user_prompts = 狙う場面、dont_touch = 書き込みたくない場面)"),
             direction: Optional[str] = typer.Option(None, "--direction", help="@NAME を省いたユニットの方向"),
             key: Optional[str] = typer.Option(None, "--key", help="#KEY を省いたユニットの鍵"),
             subset: str = typer.Option("odd", "--subset", help="使う質問: odd(既定。鍵の作成に使っていない側)/ even / all"),
             replies: str = typer.Option("own", "--replies", help="own(レシピを当てて作った返事を、当てたまま読む = 実際に起きること)/ base(無介入の返事を、レシピを当てて読む)"),
             max_tokens: int = typer.Option(96, "--max-tokens"),
             n_pos: int = typer.Option(8, "--positions", help="位置 0 からいくつまで 1 位置ずつ出すか(それ以降はまとめる)"),
             iv: Optional[list[str]] = typer.Option(None, "--iv", help="この介入(レシピの名前かパス、またはユニットの書き方)を先に当てた状態で測る")):
    """位置の種類ごとに、鍵の反応と実際に足した量(α·d を 1 として)を出す。狙う場面(対照集合の質問)と、書き込みたくない場面
    (dont_touch の群)で、返事の位置 0(書き始める直前)から順に。足す量が 1 を大きく超える位置、書き込みたくない場面で 0 から
    離れる位置、負(逆向き)になる位置が無いかを見る。"""
    s = holder["session"] = c.session()
    r = s.recipe(units=unit, path=recipe, default_direction=direction, default_key=key)
    d = s.nw_check(r, contrast, subset=subset, max_tokens=max_tokens, replies=replies, n_pos=n_pos, iv=_iv_arg(iv))
    lines = [f"対照集合 {d['contrast']}(subset {d['subset']}、返事 {d['replies']}、{d['seconds']:.0f} 秒)。値は足した量の平均(α·d を 1。かっこ内は最小〜最大)"]
    for g, x in d["groups"].items():
        lines.append(f"[{g}] {x['n']} 問、ループ {x['loops']}")
        for u in x["units"]:
            cells = []
            for b, p in u["positions"].items():
                a = p["amount"]
                cells.append(f"{b}: " + ("—" if a is None else f"{a['mean']:+.2f}({a['min']:+.2f}〜{a['max']:+.2f})"))
            lines.append(f"  {u['unit']}\n    " + "  ".join(cells))
    lines += [f"注: {n}" for n in d["notes"]]
    return d, "\n".join(lines), 0


@gne_command(nw_app, "export", "nw export")
def nw_export(c: Common, holder: dict, recipe: Optional[Path] = _RECIPE, unit: Optional[list[str]] = _UNIT,
              out: Optional[Path] = typer.Option(None, "--out", "-o", help="出力(既定 <root>/export/<レシピ名>-nw.lora.gguf)"),
              direction: Optional[str] = typer.Option(None, "--direction", help="@NAME を省いたユニットの方向"),
              key: Optional[str] = typer.Option(None, "--key", help="#KEY を省いたユニットの鍵"),
              bake: bool = typer.Option(False, "--bake", help="焼き込み(対応していない。はっきりしたエラーを返す)"),
              strip_meta: bool = typer.Option(False, "--strip-meta", help="gne の記録を残さない(人に渡す版)")):
    """nw のユニットを LoRA アダプタ GGUF に書き出す(llama.cpp の --lora で当てる)。ffn_gate・ffn_up・ffn_down への差し替えで、
    ニューロン 1 個あたり 3 つのテンソルに階数 1。焼き込みは無い(Q8_0 の ffn_down は 32 個の重みが 1 つの倍率を共有するので、
    1 列だけを方向に差し替えると誤差が 9 割を超える)。"""
    s = holder["session"] = c.session()
    r = s.recipe(units=unit, path=recipe, default_direction=direction, default_key=key)
    if bake:
        s._no_nw(r, bake=True)
        raise GneError("BAD_ARGS", "計算で作るニューロン(nw)は焼き込めません", "LoRA アダプタ(--bake なし)で書き出してください")
    out = out or (s.store.root / "export" / f"{r.name or 'recipe'}-nw.lora.gguf")
    d = s.nw_export(r, out, strip_meta=strip_meta)
    d.pop("warnings", None)
    lines = [f"{d['path']}({d['bytes'] / 1e6:.2f} MB、LoRA)"]
    for x in d["nw"]:
        lines.append(f"  L{x['layer']} N{','.join(map(str, x['neurons']))}: {x['form']}、s {x['s']:g}、α {x['alpha']:+g}、方向 {x['direction']}、鍵 {x['key']}"
                     + (f" & {x['key2']}" if x["key2"] else "") + f"、鍵が反応する位置で |α|‖d‖ = {x['norm']:.2f}")
    lines.append("llama.cpp では --lora <このファイル> で当てます(倍率 1)。")
    return d, "\n".join(lines), 0


if __name__ == "__main__":
    app()
