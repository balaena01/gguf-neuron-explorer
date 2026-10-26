"""V7: set_cvec / prefill / decode / 生成の実測。設計書 §9 V7, §10。結果は docs/measurements.md に記録する。"""

from __future__ import annotations

import platform
import subprocess
import time

import numpy as np
from llama_cpp import llama_cpp as C

from .capture import Capture, pick_act_name
from .engine import Engine

_TEXTS = [
    "The quick brown fox jumps over the lazy dog while the farmer watches from the old wooden porch at dawn.",
    "In 1905, Albert Einstein published four papers that changed physics, including special relativity and the photoelectric effect.",
    "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所で泣いていた事だけは記憶している。",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\nprint(fibonacci(10))",
]


def _median_ms(xs: list[float]) -> float:
    return float(np.median(xs) * 1000)


def gpu_name() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "?"
    except (OSError, subprocess.SubprocessError):
        return "?"


def run_bench(engine: Engine, *, layer: int | None = None, reps: int = 10, gen_tokens: int = 128,
              n_seq: int = 4, seq_tokens: int = 20, seed: int = 0) -> dict:
    e = engine
    L = layer if layer is not None else e.n_layer // 2
    rng = np.random.default_rng(seed)
    seqs = [e.tokenize(t)[:seq_tokens] for t in (_TEXTS * n_seq)[:n_seq]]
    if any(len(s) < seq_tokens for s in seqs):
        raise RuntimeError("ベンチ用の文が短すぎます")
    ctx = e.steer_ctx()

    def vec() -> dict[int, np.ndarray]:
        v = rng.standard_normal(e.n_embd).astype(np.float32)
        return {L: v * (10.0 / np.linalg.norm(v))}

    def prefill() -> float:
        e.clear_memory(ctx)
        t = time.perf_counter()
        e.prefill(ctx, seqs)
        return time.perf_counter() - t

    e.set_cvec(ctx, vec())
    for _ in range(3):
        prefill()

    steady = [prefill() for _ in range(reps)]

    t_set, t_after = [], []
    for _ in range(reps):
        t = time.perf_counter()
        e.set_cvec(ctx, vec())
        t_set.append(time.perf_counter() - t)
        t_after.append(prefill())

    t_set_fixed, t_after_fixed = [], []
    for _ in range(reps):
        t = time.perf_counter()
        e.set_cvec(ctx, vec(), layer_range=(1, e.n_layer - 1))
        t_set_fixed.append(time.perf_counter() - t)
        t_after_fixed.append(prefill())

    # decode 1 step(n_seq 本)
    step = []
    for _ in range(reps):
        prefill()
        items = [(s[-1], len(s), i, True) for i, s in enumerate(seqs)]
        t = time.perf_counter()
        e._decode(ctx, items)
        C.llama_synchronize(ctx.ptr)        # llama_decode は GPU の完了を待たずに戻る
        step.append(time.perf_counter() - t)

    # 生成(EOG で止めない)。n_seq 本と 1 本
    gen = {}
    for k, ss in (("batch", seqs), ("single", seqs[:1])):
        ts = []
        for _ in range(max(2, reps // 3)):
            t = time.perf_counter()
            rs = e.generate_greedy(ctx, ss, max_tokens=gen_tokens, stop_on_eog=False)
            ts.append(time.perf_counter() - t)
        n_tok = sum(r.n_gen for r in rs)
        gen[k] = {"n_seq": len(ss), "ms": _median_ms(ts), "tok_per_s": n_tok / float(np.median(ts))}
    e.set_cvec(ctx, None)

    # probe ctx: 512 トークン 1 本、全層の l_out を捕捉する場合と何も捕捉しない場合
    long_ids = (e.tokenize(" ".join(_TEXTS)) * 20)[:e.probe_ubatch]
    names = e.list_names(long_ids[:16])
    act_names = {pick_act_name(names, L) for L in range(e.n_layer)} - {None}
    cap = Capture()
    pctx = e.probe_ctx(cap)
    probe = {}
    for k, wanted in (("none", lambda n: False), ("all_l_out", lambda n: n.startswith("l_out-")),
                      ("all_ffn_act", lambda n: n in act_names)):
        cap.set_wanted(wanted)
        ts = []
        for _ in range(max(3, reps // 2)):
            e.clear_memory(pctx)
            t = time.perf_counter()
            e.prefill(pctx, [long_ids])
            ts.append(time.perf_counter() - t)
            cap.take()
        probe[k] = _median_ms(ts)

    reserve = _median_ms(t_after) - _median_ms(steady)
    return {
        "model": str(e.path.name), "layer": L, "n_seq": n_seq, "seq_tokens": seq_tokens, "reps": reps,
        "gpu": gpu_name(), "n_gpu_layers": e.n_gpu_layers, "n_ctx": e.n_ctx, "llama_cpp_python": e.llama_cpp_version(),
        "python": platform.python_version(),
        "prefill_ms": _median_ms(steady),
        "set_cvec_call_ms": _median_ms(t_set),
        "prefill_after_set_ms": _median_ms(t_after),
        "reserve_ms": reserve,
        "set_cvec_fixed_call_ms": _median_ms(t_set_fixed),
        "prefill_after_set_fixed_ms": _median_ms(t_after_fixed),
        "set_plus_prefill_ms": _median_ms([a + b for a, b in zip(t_set, t_after)]),
        "decode_step_ms": _median_ms(step),
        "gen": gen, "gen_tokens": gen_tokens,
        "probe_prefill_512_ms": probe, "probe_tokens": len(long_ids),
    }


def bench_markdown(r: dict) -> str:
    g = r["gen"]
    return "\n".join([
        f"### {r['model']}(層 L{r['layer']}、{r['gpu']}、llama-cpp-python {r['llama_cpp_python']}、n_ctx {r['n_ctx']}、各 {r['reps']} 回の中央値)",
        "",
        "| 処理 | 時間 |",
        "|---|---|",
        f"| prefill {r['n_seq']} seq × {r['seq_tokens']} tok(cvec 変更なし) | {r['prefill_ms']:.1f} ms |",
        f"| `set_cvec` の呼び出しだけ | {r['set_cvec_call_ms']:.2f} ms |",
        f"| `set_cvec` 直後の prefill(グラフ再予約込み) | {r['prefill_after_set_ms']:.1f} ms |",
        f"| → 再予約の上乗せ | {r['reserve_ms']:.1f} ms |",
        f"| `set_cvec` + prefill 合計 | {r['set_plus_prefill_ms']:.1f} ms |",
        f"| 層範囲を 1..n_layer-1 に固定した `set_cvec` 直後の prefill | {r['prefill_after_set_fixed_ms']:.1f} ms |",
        f"| decode 1 step({r['n_seq']} seq) | {r['decode_step_ms']:.1f} ms |",
        f"| 生成 {r['gen_tokens']} tok × {g['batch']['n_seq']} seq | {g['batch']['ms']:.0f} ms({g['batch']['tok_per_s']:.0f} tok/s) |",
        f"| 生成 {r['gen_tokens']} tok × 1 seq | {g['single']['ms']:.0f} ms({g['single']['tok_per_s']:.0f} tok/s) |",
        f"| probe ctx prefill {r['probe_tokens']} tok(捕捉なし / 全層 l_out / 全層 FFN 活性) | "
        f"{r['probe_prefill_512_ms']['none']:.1f} / {r['probe_prefill_512_ms']['all_l_out']:.1f} / "
        f"{r['probe_prefill_512_ms']['all_ffn_act']:.1f} ms |",
        "",
    ])
