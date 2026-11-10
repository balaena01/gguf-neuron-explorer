"""ジョブの時間の見積もりと、予算に合わせた絞り込み(CHANGE-03 C)。

基準は RTX 3080 の Qwen3.5-4B Q8_0 の実測(docs/measurements.md)。生成の 1 step と、set_cvec + prefill(4 本 × 約 20 トークン)
の時間を「固定分 + 実効の大きさ E に比例する分」で表し、GPU に載らない層は CPU の速さで足す。E = ファイルの GB + 0.79 × パラメータ数
(十億)。ファイルの大きさだけだと、2 bit の 27B(8.4 GB)が 4B Q8_0(4.5 GB)の 2 倍弱になるが、実測は 4 倍近かった(量子化が
細かいほど dequantize の計算が重い)。係数は 4B と Saluki 27B の層選びの実測に合わせた。`gne bench` の結果が
`<root>/bench/<model_id>.json` にあれば、その実測で置き換える。生成は max_tokens まで続く場合の上限で、EOG で早く止まれば短い。

| 量 | 基準(4B、3080) | 式 |
|---|---|---|
| 生成 1 step(1 本) | 8.8 ms | 3.0 + 0.74·E(ms) |
| 生成 1 step の 1 本増えるごとの上乗せ | 1.8 ms | 0.25 + 0.2·E |
| set_cvec + prefill(4 本 × 20 トークン) | 46 ms | 17.5 + 3.65·E |
| CPU に置いた層(ファイル 1 GB あたり、1 step) | — | 50 ms(DRAM 約 20 GB/s) |
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REF = {"step1_ms": 8.8, "step_extra_ms": 1.8, "setprefill_ms": 46.0}   # 4B Q8_0、3080
CPU_MS_PER_GB = 50.0
CONFIRM_S = 30 * 60                         # これを超える見積もりは --yes なしでは走らせない


def parse_budget(text: str | None) -> float | None:
    """"30m" / "2h" / "90s" / "45" → 秒(数だけなら分)。"""
    if text is None or text == "":
        return None
    t = str(text).strip().lower()
    mult = {"s": 1, "m": 60, "h": 3600}
    try:
        if t[-1] in mult:
            return float(t[:-1]) * mult[t[-1]]
        return float(t) * 60
    except (ValueError, IndexError) as e:
        from .steer import GneError
        raise GneError("BAD_ARGS", f"--budget {text!r} は 30m・2h・90s のように書きます") from e


class Speed:
    """モデルの速さの見積もり(ms)。"""

    def __init__(self, session):
        m = session.meta
        self.gb = session.weights.meta.file_size / 1e9
        self.params_b = sum(math.prod(int(x) for x in t.shape) for t in session.weights.gguf.tensors) / 1e9
        self.eff = self.gb + 0.79 * self.params_b
        n_gpu = session._engine_kw.get("n_gpu_layers", -1)
        n_layer = m.n_layer
        self.gpu_layers = n_layer if n_gpu is None or n_gpu < 0 else min(n_gpu, n_layer)
        self.cpu_frac = 1.0 - self.gpu_layers / max(1, n_layer)
        self.source = "default"
        cpu = self.cpu_frac * self.gb * CPU_MS_PER_GB
        self.step1 = 3.0 + 0.74 * self.eff + cpu
        self.step_extra = 0.25 + 0.2 * self.eff + 0.3 * cpu
        self.setprefill = 17.5 + 3.65 * self.eff + 4 * cpu
        bp = session.store.root / "bench" / f"{m.model_id}.json"
        if bp.exists():
            try:
                b = json.loads(bp.read_text())
                n = max(1, int(b.get("n_seq", 4)))
                single = b["gen"]["single"]["ms"] / max(1, b["gen_tokens"])
                batch = b["gen"]["batch"]["ms"] / max(1, b["gen_tokens"])
                self.step1 = single
                self.step_extra = max(0.0, (batch - single) / max(1, n - 1))
                self.setprefill = float(b["set_plus_prefill_ms"])
                self.source = "bench"
            except (KeyError, ValueError, TypeError, ZeroDivisionError):
                pass

    def step(self, k: int) -> float:
        """k 本まとめた生成の 1 step(ms)。"""
        return self.step1 + self.step_extra * max(0, k - 1)

    def gen(self, n: int, tokens: int, n_seq_max: int = 8) -> float:
        """n 本を n_seq_max 本ずつまとめて tokens まで生成する時間(秒)。"""
        out, left = 0.0, n
        while left > 0:
            k = min(left, n_seq_max)
            out += tokens * self.step(k) / 1000
            left -= k
        return out

    def to_dict(self) -> dict:
        return {"model_gb": round(self.gb, 2), "params_b": round(self.params_b, 2), "effective": round(self.eff, 1),
                "gpu_layers": self.gpu_layers, "cpu_fraction": round(self.cpu_frac, 3),
                "step1_ms": round(self.step1, 2), "step_extra_ms": round(self.step_extra, 2),
                "setprefill_ms": round(self.setprefill, 1), "source": self.source}


def _design_cost(sp: Speed, design_name: str, max_tokens: int, prompt_set: str = "en") -> float:
    """sweep の 1 ユニット(ニューロンか対照)の秒数: dose ごとに set_cvec して、その dose のプロンプトをまとめて生成。"""
    from .sweep import load_design
    d = load_design(design_name, prompt_set=prompt_set)
    t = 0.0
    for _, keys in d.groups():
        t += sp.setprefill / 1000 + max_tokens * sp.step(len(keys)) / 1000
    return t


def _n_neurons(session, L: int, neurons) -> int:
    from .selection import parse_neurons
    return int(parse_neurons(neurons, session.weights.n_units(L), seed=session.seed).size)


def estimate(session, kind: str, params: dict) -> dict:
    """ジョブ kind(job_start の種類)の見積もり: {"seconds", "detail", "speed"}。見積もれない種類は seconds None。"""
    sp = Speed(session)
    p = dict(params or {})
    m = session.meta
    detail: dict = {}
    sec: float | None = None
    if kind == "lens_steer":
        n = _n_neurons(session, int(p["layer"]), p.get("neurons") or "all") + int(p.get("random_controls", 64))
        sec = n * 3.1 * sp.setprefill / 1000                       # 実測で 1 ユニット ≈ 3.1 ×(set_cvec + prefill)
        detail = {"units": n}
    elif kind in ("sweep", "sweep_rank"):
        n = _n_neurons(session, int(p["layer"]), p.get("neurons", "random:384")) + int(p.get("random_controls", 64))
        per = _design_cost(sp, p.get("design", "main_rel"), int(p.get("max_tokens", 128)), p.get("prompt_set", "en"))
        sec = n * per
        detail = {"units": n, "seconds_per_unit": round(per, 2), "design": p.get("design", "main_rel")}
    elif kind == "experts sweep":                                  # CHANGE-05 §E.2: 1 専門家・1 倍率ごとにパッチ → 読み直し → 生成
        n = int(p.get("n", 0))
        reload = 0.2 + 0.08 * m.file_size / 1e9                     # granite 1.4 GB で 0.31 秒(ページキャッシュあり)
        per = reload + _design_cost(sp, "quick", int(p.get("max_tokens", 48)))
        sec = n * per
        detail = {"patches": n, "seconds_per_patch": round(per, 2), "reload_s": round(reload, 2)}
    elif kind == "reverse":
        n = int(p.get("verify_top") or p.get("top", 200)) if p.get("verify") else 0
        n_ctrl = int(p.get("verify_controls", 8))
        per = _design_cost(sp, p.get("verify_design", "quick"), int(p.get("max_tokens", 128)))
        n_layers = min(n, max(1, int(0.5 * m.n_layer))) if n else 0   # 対照は候補の層ごと(候補は層の半分ほどに散る)
        fixed = 6.0 * sp.setprefill / REF["setprefill_ms"] + 30
        sec = fixed + (n + (n_ctrl + 1) * n_layers) * per                # +1 は層ごとのベースライン
        if p.get("verify_budget_s") is not None:
            sec = min(sec, fixed + float(p["verify_budget_s"]))
        detail = {"candidates": n, "seconds_per_unit": round(per, 2)}
    elif kind == "direction":
        from .direction import candidate_layers, load_contrast
        cs, val = load_contrast(p["contrast"])
        n_conv = len(cs.prompts("A")) + len(cs.prompts("B"))
        sec = n_conv * sp.setprefill / 1000 * 2
        if p.get("positions") == "response":
            sec += 2 * sp.gen(len(cs.user_prompts), int(p.get("response_tokens", 96)))
        if p.get("select") == "auto":
            layers = p.get("layers")
            n_l = len(candidate_layers(m.n_layer, layers if isinstance(layers, str) else None))
            per = sp.gen(len(val.prompts), int(p.get("max_tokens", 64))) + sp.setprefill / 1000
            n_ctrl = int(p.get("random_controls", 16))
            ctrl_layers = n_l if p.get("mode", "ablate") == "ablate" else min(n_l, 3)   # add は選ばれ得る層だけ(多くて数層)
            sec += (n_l + 1 + n_ctrl * (1 if p.get("mode", "ablate") == "ablate" else ctrl_layers)) * per
            detail = {"layers": n_l, "seconds_per_measure": round(per, 2), "controls": n_ctrl}
    elif kind == "eval":
        from .evalx import load_suite
        st = load_suite(p.get("suite"))
        doses = p.get("doses") or [0, 0.25, 0.5, 0.75, 1]
        n_d = len(set(map(float, doses)) | {0.0})
        when = p.get("when") or "all"
        per = 0.0
        for cat, items in (("exact", st.exact), ("format", st.format), ("mix", st.mix), ("persona", st.persona)):
            if items:
                per += sp.gen(len(items), st.max_tokens[cat], 8 if when == "all" else 1)
        per += len(st.ppl) * 4 * sp.setprefill / 1000
        n_ctrl = int(p.get("random_controls", 0))
        kd = _keyed_eval(session, sp, p, n_d - 1, n_ctrl)          # CHANGE-07: keyed のレシピの eval の選択性(prefill だけ)
        if kd is not None:
            n_ctrl = kd["controls"]                                # keyed の対照は鍵の比較用の鍵の本数まで
        sec = per * (n_d + n_ctrl * (n_d - 1))
        detail = {"doses": n_d, "seconds_per_dose": round(per, 1)}
        if kd is not None:
            sec += kd["seconds"]
            detail["keyed"] = kd
    elif kind in ("direction_key", "direction_key_check", "direction_key_experts"):   # CHANGE-07: 鍵の作成・確かめ
        kd = _keyed_job(session, sp, kind, p)
        sec = kd["seconds"]
        detail = kd
    elif kind == "direction_key_pipeline":                     # CHANGE-07: 方向(返事を作る)+ 鍵 + key-check
        kd = _keyed_pipeline(session, sp, p)
        sec = kd["seconds"]
        detail = kd
    elif kind == "direction_key_verify":                       # CHANGE-07 V30: 文を無介入 1 回 + α ごとに 1 回流す(生成なし)
        n_t = len(p.get("texts") or []) if not isinstance(p.get("texts"), str) else len([x for x in p["texts"].splitlines() if x.strip()])
        n_t = n_t or 2
        n_a = len(p.get("alphas") or (1.0, 2.0))
        sec = (1 + n_a) * n_t * sp.setprefill / 1000 + 1.0       # + LoRA を作る分
        detail = {"texts": n_t, "alphas": n_a}
    elif kind == "direction_key_causal":                       # keyed の効く場所の第 1 段: 返事 + 鍵と key-check(書き込み先ごと)+ 測る回数ぶんの prefill
        kd = _key_causal_cost(session, sp, p)
        sec = kd["seconds"]
        detail = kd
    elif kind == "direction_key_causal_verify":                # 第 2 段: (iv のみ + 候補 + 比較用の鍵)× 質問の生成
        kd = _key_causal_verify_cost(session, sp, p)
        sec = kd["seconds"]
        detail = kd
    elif kind == "behave":                                     # 種ごとに質問を max_tokens まで生成 + 判定(読み手、8 トークン)
        kd = _behave_cost(session, sp, p)
        sec = kd["seconds"]
        detail = kd
    elif kind == "connections":
        from .connections import dense_layers
        Ls = dense_layers(session.weights)
        w = p.get("window")
        pairs = [(i, j) for jj, j in enumerate(Ls) for i in Ls[:jj] if w is None or j - i <= int(w)]
        ref = 9216 * 2560 * 9216
        sec = sum(4.0 * m.n_ff_at(i) * m.n_embd * m.n_ff_at(j) / ref for i, j in pairs)   # 4B で 4.0 秒/ペア(CPU)
        detail = {"pairs": len(pairs), "window": w}
    elif kind == "heads_contrast":
        from .reverse import ContrastSet
        cs = ContrastSet.load(p["contrast"])
        n = len(cs.user_prompts)
        sec = sp.gen(n, int(p.get("max_tokens", 48))) + 3 * n * 4 * sp.setprefill / 1000
        detail = {"prompts": n}
    elif kind == "writers":
        from .reverse import ContrastSet
        cs = ContrastSet.load(p["contrast"])
        n = len(cs.user_prompts)
        n_l = m.n_layer
        sec = 2 * sp.gen(n, int(p.get("max_tokens", 96))) + 2 * n * (1 + n_l) * 2 * sp.setprefill / 1000
        detail = {"prompts": n, "prefills": 2 * n * (1 + n_l)}
    elif kind in ("heads_causal", "heads_allocate"):          # CHANGE-06: 生成なし。1 回 = A 側 + B 側(16 例)の prefill
        from .reverse import ContrastSet
        from .steer import GneError
        src = p.get("contrast")
        if src is None and p.get("direction"):
            try:
                src = session.direction_set(p["direction"]).meta.get("source")
            except GneError:
                src = None
        n_a = len(ContrastSet.load(src).user_prompts) if src and Path(src).exists() else 32
        b = int(p.get("b_examples", 16)) or n_a
        n_head = max((li.n_head for li in session.weights.arch.layers if li.head_dim), default=1)
        n_hl = sum(1 for li in session.weights.arch.layers if li.head_dim and li.head_writer)
        if kind == "heads_causal":
            if p.get("mode", "ablate") == "marginal":
                trials = 1 + 30
                per_a = int(p.get("hard", 16))
            else:
                auto = p.get("layers", "auto") == "auto" and p.get("candidates") in (None, "all")
                trials = (n_hl + int(p.get("top_layers", 4)) * n_head) if auto else n_hl * n_head
                per_a = n_a
        else:
            ns = [int(x) for x in (p.get("n_heads") or (6, 5, 4))]
            trials = 20 + int(p.get("pool", 30)) + (sum(ns) * 20 if p.get("method") == "greedy" else len(ns) * 16)
            per_a = int(p.get("hard", 16))
        sec = trials * (per_a + b) * sp.setprefill / 1000 * 1.2
        detail = {"trials": trials, "prefills_per_trial": per_a + b}
    elif kind == "head_pipeline":                               # 方向(全部の質問)+ 射影の値 + 因果 + 配分(間引いた質問)
        spec = p.get("spec") or {}
        if not isinstance(spec, dict):
            import yaml
            sp_path = Path(str(spec))
            spec = (json.loads(sp_path.read_text(encoding="utf-8")) if sp_path.suffix == ".json"
                    else yaml.safe_load(sp_path.read_text(encoding="utf-8"))) if sp_path.exists() else {}
        pipe = {**(spec.get("pipeline") or {}), **{k: v for k, v in p.items() if k != "spec"}}
        n = max(len(spec.get("user_prompts") or []), len(spec.get("prompts_a") or []), len(spec.get("prompts_b") or [])) or 32
        mx = min(n, int(pipe.get("max_examples") or 64))
        b = min(16, mx)
        n_head = max((li.n_head for li in session.weights.arch.layers if li.head_dim), default=1)
        n_hl = sum(1 for li in session.weights.arch.layers if li.head_dim and li.head_writer)
        trials_causal = n_hl + int(pipe.get("top_layers", 4)) * n_head
        trials_alloc = 20 + 30 + len(pipe.get("n_heads") or (6, 5, 4)) * 16
        hard = int(pipe.get("hard", 16))
        prefills = 2 * n * 2 + 2 * mx + trials_causal * (mx + b) + trials_alloc * (hard + b)
        sec = prefills * sp.setprefill / 1000 * 1.2
        if pipe.get("positions") == "response":
            sec += 2 * sp.gen(n, 96)
        detail = {"prompts": n, "screened": mx, "trials": trials_causal + trials_alloc}
    elif kind == "heads_refine":                               # 残りを測る + 残りの質問で一括(方向 → 因果 → 配分)+ 合わせた組を測る
        import yaml
        name = str(p.get("name") or "")
        hd = session.store.heads_dir(session.meta.model_id)
        pj = hd / f"{name}.pipeline.json"
        n_a, n_b = 32, 32
        if pj.exists():
            pd = json.loads(pj.read_text(encoding="utf-8"))
            n_a, n_b = int(pd.get("n_prompts") or 32), int(pd.get("n_prompts_b") or pd.get("n_prompts") or 32)
        spec = {"user_prompts": ["q"] * max(4, n_a // 3), "prompts_b": ["q"] * n_b,
                "pipeline": {"n_heads": list(p.get("new_heads") or (3, 2, 1))}}
        sub = estimate(session, "head_pipeline", {"spec": spec})
        sec = (sub.get("seconds") or 0.0) + (2 * n_a + 3 * min(64, n_b)) * sp.setprefill / 1000 * 1.2
        detail = {"prompts_a": n_a, "remaining_guess": max(4, n_a // 3), "pipeline": sub.get("detail")}
    elif kind == "fieldtest":                                  # 1 ファイルあたり: 長い作業 + ツール(2 往復)+ 質問集。多めに見る
        import yaml
        from .fieldtest import LONG_TASKS, TOOL_TASKS
        np_ = int(p.get("n_parallel", 4))
        def _count(path, key=None):
            try:
                pth = Path(str(path))
                d = json.loads(pth.read_text(encoding="utf-8")) if pth.suffix == ".json" else yaml.safe_load(pth.read_text(encoding="utf-8"))
                return len(d[key] if key else d)
            except (OSError, ValueError, TypeError, KeyError):
                return 0
        n_long = 0 if p.get("skip_long") else _count(p.get("long_tasks") or LONG_TASKS)
        n_tool = 0 if p.get("skip_tools") else _count(p.get("tool_tasks") or TOOL_TASKS, "tasks")
        sets = p.get("prompts") or []
        vals = list(sets.values()) if isinstance(sets, dict) else [str(x).split("=", 1)[-1] for x in sets]
        n_q = sum(len(v) if isinstance(v, list) else _count(v) for v in vals)
        n_runs = 2 if (p.get("against") or p.get("lora")) else 1
        bn = p.get("bench") or []
        bn = [x for x in bn.split(",") if x.strip()] if isinstance(bn, str) else list(bn)
        from .fieldtest import BENCHES
        n_b = int(p.get("bench_n") or 0) * len(bn) or sum((BENCHES.get(b.strip()) or {"n": 300})["n"] for b in bn)
        tok_b = max(800, int(int(p.get("bench_tokens") or 4096) * 0.5) if int(p.get("bench_tokens") or 4096) > 4096 else 800)   # 上限を上げるのはループする長い返事を測るとき
        one = (sp.gen(n_long, int(p.get("long_tokens", 3072)), np_) * 0.7 + sp.gen(n_tool, 400, np_) * 2
               + sp.gen(n_q, int(p.get("prompt_tokens", 1536)), np_) * 0.5 + sp.gen(n_b, tok_b, np_) + 120)
        sec = n_runs * one
        detail = {"files": n_runs, "long_tasks": n_long, "tool_tasks": n_tool, "prompts": n_q, "bench": n_b}
    elif kind == "heads_candidates":                           # 候補(組 × 倍率 + 無介入)× 両側の質問を生成する
        n_rows = len(p.get("n_heads") or []) or 3
        ap = p.get("alloc")
        if ap and not p.get("n_heads"):
            pth = Path(str(ap))
            if not pth.exists():
                pth = session.store.heads_dir(session.meta.model_id) / f"{ap}.alloc.json"
            if pth.exists():
                n_rows = sum(1 for r in json.loads(pth.read_text(encoding="utf-8")).get("table", []) if not r.get("skipped"))
        n_c = 1 + n_rows * len(p.get("scales") or (0.5, 1.0, 1.5, 2.0, 3.0))
        n_q = int(p.get("max_prompts", 24))
        sec = n_c * 2 * sp.gen(n_q, int(p.get("max_tokens", 120)))
        detail = {"candidates": n_c, "prompts_per_side": n_q, "replies": n_c * 2 * n_q}
    elif kind == "stats_resid":
        sec = 10.0 * sp.setprefill / REF["setprefill_ms"]
    return {"kind": kind, "seconds": None if sec is None else float(sec), "detail": detail, "speed": sp.to_dict(),
            "needs_confirm": bool(sec is not None and sec > CONFIRM_S)}


def _behave_cost(session, sp: Speed, p: dict) -> dict:
    """behave の見積もり: 種ごとに、質問を n_seq_max 本ずつまとめて max_tokens まで生成する(EOG で早く止まれば短い)。判定が
    あれば、読み手が返事ごとに 8 トークン生成する(返事を読む prefill の分を set_cvec + prefill 1 回とみる)。思考ありも同じ長さまで。"""
    from .behave import load_prompts
    qs = p.get("prompts")
    try:
        if isinstance(qs, str) and not Path(qs).exists():
            qs = session._contrast_file(qs)
        n = len(load_prompts(qs)) if isinstance(qs, str | Path) else len(qs or [])
    except Exception:  # noqa: BLE001  質問集が読めなくても見積もりで止めない(本番がエラーを返す)
        return {"seconds": 32 * sp.step(8) / 1000 * 400, "guess": True}
    seeds = p.get("seeds")
    if isinstance(seeds, str):
        seeds = [x for x in seeds.split(",") if x.strip()]
    n_seeds = max(1, len(seeds or [0]))
    mt = int(p.get("max_tokens") or 400)
    gen = sp.gen(n, mt, session._engine_kw.get("n_seq_max", 8) or 8) + n * sp.setprefill / 1000 / 4
    judge = (sp.gen(n, 8) + n * sp.setprefill / 1000) if p.get("judge") else 0.0
    return {"prompts": n, "seeds": n_seeds, "max_tokens": mt, "judge": bool(p.get("judge")),
            "seconds": n_seeds * (gen + judge) + 1.0}


# --- CHANGE-07: 鍵つきの書き込み(keyed)----------------------------------------------------------------------
# 鍵の作成(direction_key)・確かめ(direction_key_check)・専門家の選ばれ方(direction_key_experts)と、eval の keyed の選択性
# (keyed_eval.KeyedMetrics)は、どれも対照集合の会話を 1 本ずつ prefill して捕捉する(返事を与えていない組は無介入で 1 回だけ
# 生成する)。会話 1 本(60〜120 トークン)の prefill は set_cvec + prefill(4 本 × 20 トークン)1 回とみる。
# 実測(granite-3.1-1b-a400m Q8_0、CPU、ずんだもんの語尾の対照集合で返事は与えたもの、expert:e の鍵): 鍵の作成 5.5 秒、
# key-check 6 秒。この式ではどちらも 6.2 秒(会話 8 本 + dont_touch の英語 2 本、英語の返事は 32 トークン生成)。


def _key_sets(spec, pos_a: str, pos_b: str | None, subset: str, replies_a=None, replies_b=None) -> list[tuple[int, list[str], bool]]:
    """keyed_build._plan と同じ会話の組を、会話の数・位置・返事があるかだけ数える(介入は作らない。見積もり用)。
    戻り値 [(会話の数, [位置の書き方], 返事が与えてあるか)]。"""
    from .direction import _subset
    from .keyed_build import _resolve_mode, parse_positions
    cs = spec.contrast
    pa = parse_positions(pos_a)
    mode, pb = _resolve_mode(pa, pos_b)
    out = []
    if mode == "system":
        for side, pos, rep in (("A", pa, replies_a), ("B", pb, replies_b)):
            n = len(_subset(len(cs.prompts(side)), subset))
            out.append((n, [pos.text], rep is not None or spec.replies.get(side) is not None))
    else:
        rep = replies_b if replies_b is not None else replies_a
        n = len(_subset(len(cs.prompts("B")), subset))
        given = rep is not None or spec.replies.get("same") is not None or spec.replies.get("B") is not None
        out.append((n, [pa.text, pb.text], given))
    for dt in spec.dont_touch:
        out.append((len(_subset(len(dt.prompts), subset)), [dt.positions or pb.text], dt.replies is not None))
    return out


def _plan_cost(sp: Speed, sets, max_tokens: int, max_examples: int | None = None) -> dict:
    """_key_sets の会話の組 → 会話の数・書き込みたい位置を含む会話の数・生成する会話の数と、prefill 1 回分 + 生成の秒数。
    max_examples は組ごとに読む会話の数の上限(eval の keyed_examples、等間隔に間引く)。"""
    from .keyed_build import parse_positions
    n = n_a = n_gen = 0
    gen_s = 0.0
    for i, (k0, groups, given) in enumerate(sets):
        k = min(k0, max_examples) if max_examples else k0
        n += k
        n_a += k if i == 0 else 0                     # 書き込みたい位置は最初の組(mode system の A、mode same の AB)
        if not given and any(parse_positions(x).needs_reply for x in groups):
            n_gen += k
            gen_s += sp.gen(k, int(max_tokens))
    return {"conversations": n, "conversations_a": n_a, "generated": n_gen, "max_tokens": int(max_tokens),
            "seconds": gen_s + n * sp.setprefill / 1000}


def _spec_of(p: dict):
    """見積もりの対照集合: p の spec(UI の入力欄の表)か contrast(パス)→ (KeySpec, keyed の設定)。表は一時ファイルに書いて
    読む(本番は <root>/contrasts/ に書く。見積もりでは置き場所を変えない)。"""
    import tempfile

    from .keyed_build import _as_spec, read_key_spec
    from .keyed_pipeline import clean_spec, file_settings
    from .yamlio import write_yaml
    if p.get("spec") is not None:
        d, settings = clean_spec(p["spec"])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / f"{d['name']}.yaml"
            write_yaml(path, d)
            return read_key_spec(path), settings
    c = p["contrast"]
    return _as_spec(c), (file_settings(c) if isinstance(c, str | Path) else {})


def _keyed_pipeline(session, sp: Speed, p: dict) -> dict:
    """direction_key_pipeline の見積もり: 方向(A・B の返事を response_tokens まで作って流す。同じ名前の方向があって作り直さない
    なら 0)+ 鍵の作成 + 作成と反対側の subset での key-check。"""
    try:
        spec, settings = _spec_of(p)
        q = {**{k: v for k, v in settings.items() if k != "alpha"}, **{k: v for k, v in p.items() if v is not None}}
        cs = spec.contrast
        dname = q.get("direction") or cs.name
        try:
            session.direction_set(dname)
            have = True
        except Exception:  # noqa: BLE001
            have = False
        dir_s = 0.0
        if not have or q.get("rebuild_direction"):
            n_conv = len(cs.prompts("A")) + len(cs.prompts("B"))
            dir_s = n_conv * sp.setprefill / 1000 * 2 + sp.gen(len(cs.prompts("A")), int(q.get("response_tokens") or 96)) \
                + sp.gen(len(cs.prompts("B")), int(q.get("response_tokens") or 96))
        pos = q.get("positions") or spec.positions or "response"
        pos_b = q.get("positions_b") or spec.positions_b
        subset = q.get("subset") or "even"
        mt = int(q.get("max_tokens") or 96)
        build = _plan_cost(sp, _key_sets(spec, pos, pos_b, subset, q.get("replies_a"), q.get("replies_b")), mt)
        check = {"seconds": 0.0}
        if q.get("check", True) is not False:
            sub = q.get("check_subset") or {"even": "odd", "odd": "even"}.get(subset, "all")
            check = _plan_cost(sp, _key_sets(spec, pos, pos_b, sub, q.get("replies_a"), q.get("replies_b")), mt)
        return {"direction_seconds": dir_s, "direction_reused": have and not q.get("rebuild_direction"),
                "build": build, "check": check, "seconds": dir_s + build["seconds"] + check["seconds"]}
    except Exception:  # noqa: BLE001  対照集合が読めなくても見積もりで止めない(本番がエラーを返す)
        return {"seconds": 3 * 32 * sp.setprefill / 1000, "guess": True}


def _keyed_job(session, sp: Speed, kind: str, p: dict) -> dict:
    """direction_key・direction_key_check・direction_key_experts の見積もり(会話の prefill と返事の生成)。"""
    from .keyed_build import _as_spec
    try:
        if kind == "direction_key_check":
            from .keyed import load_key
            ks = load_key(session, str(p["key"]))
            if p.get("contrast") is None:                       # 鍵に保存した作成時の例で測る(捕捉なし)
                return {"conversations": 0, "conversations_a": 0, "generated": 0, "max_tokens": 0, "seconds": 0.0}
            spec = _as_spec(p["contrast"])
            pos = p.get("positions") or ks.positions or "response"
            pos_b = p.get("positions_b") or (ks.info.get("positions_b") if p.get("positions") is None else None)
            mt = p.get("max_tokens") or ks.max_tokens
            subset = p.get("subset") or "odd"
        else:
            spec, settings = _spec_of(p)
            p = {**{k: v for k, v in settings.items() if k != "alpha"}, **{k: v for k, v in p.items() if v is not None}}
            pos = p.get("positions") or spec.positions or "response"
            pos_b = p.get("positions_b") or spec.positions_b
            mt = p.get("max_tokens") or 96
            subset = p.get("subset") or "even"
        sets = _key_sets(spec, pos, pos_b, subset, p.get("replies_a"), p.get("replies_b"))
        return _plan_cost(sp, sets, int(mt))
    except Exception:  # noqa: BLE001  対照集合が読めなくても見積もりで止めない(本番がエラーを返す)
        return {"conversations": None, "seconds": 32 * sp.setprefill / 1000, "guess": True}


# keyed の効く場所(keyed_causal)。測る 1 回は、作成に使っていない奇数番目の返事(書き込みたい場面・dont_touch)と probe の列を
# まとめて読む prefill。実測(Qwen3-0.6B Q8_0、GPU、configs/keyed/sentence_endings.json の奇数番目 8 問 + 英語 2 問、返事 48 トークン)で
# 1 回 0.15 秒 = 列 1 本あたり set_cvec + prefill(4 本 × 20 トークン)の 0.7 倍ほど。KC_PER_SEQ はその倍率。
KC_PER_SEQ = 0.8


def _key_causal_cost(session, sp: Speed, p: dict) -> dict:
    """direction_key_causal の見積もり: 方向(無ければ)+ 返事(無いところだけ 1 回)+ 書き込み先ごとの鍵の作成と key-check
    (返事を与えた prefill)+ 測る回数(1 + 候補 × α の段 × (1 + 比較用の鍵))× 奇数番目の列の prefill。"""
    try:
        from .direction import _subset
        from .keyed_build import _resolve_mode, parse_positions
        from .keyed_causal import DEFAULT_ALPHAS, DEFAULT_PSEUDO, _n_replies_to_make, parse_alphas, parse_targets, read_probe
        spec, settings = _spec_of(p)
        cs = spec.contrast
        mode, _ = _resolve_mode(parse_positions(spec.positions or "response"), spec.positions_b)
        mt = int(p.get("max_tokens") or settings.get("max_tokens") or 96)
        dname = p.get("direction") or settings.get("direction") or cs.name
        try:
            session.direction_set(dname)
            dir_s = 0.0
        except Exception:  # noqa: BLE001
            dir_s = len(cs.prompts("A")) * sp.setprefill / 1000 * 2 + sp.gen(len(cs.prompts("A")), 96) + sp.gen(len(cs.prompts("B")), 96)
        gen_s = sp.gen(_n_replies_to_make(spec, mode), mt)
        n_conv = (len(cs.prompts("B")) if mode == "same" else len(cs.prompts("A")) + len(cs.prompts("B"))) \
            + sum(len(dt.prompts) for dt in spec.dont_touch)
        targets = parse_targets(p.get("targets")) if p.get("targets") else ["ffn", "attn"]
        layers = p.get("layers")
        from .jobs import _layers
        Ls = _layers(layers) if layers not in (None, "", "all") else None
        n_l = len(Ls) if Ls else max(1, session.meta.n_layer - 1)
        keys_s = len(targets) * n_conv * sp.setprefill / 1000              # 偶数番目で作成 + 奇数番目で key-check = 全部の会話 1 回
        alphas = parse_alphas(p.get("alphas")) or list(DEFAULT_ALPHAS)
        pseudo = int(p.get("pseudo") or DEFAULT_PSEUDO)
        side = "B" if mode == "same" else "A"
        n_seq = len(_subset(len(cs.prompts(side)), "odd")) + (len(_subset(len(cs.prompts("B")), "odd")) if mode == "system" else 0) \
            + sum(len(_subset(len(dt.prompts), "odd")) for dt in spec.dont_touch)
        if p.get("probe") is not None:
            try:
                n_seq += 2 * len(read_probe(p["probe"]))
            except Exception:  # noqa: BLE001
                pass
        n_cand = len(targets) * n_l
        n_runs = 1 + n_cand * len(alphas) * (1 + pseudo)
        per = n_seq * KC_PER_SEQ * sp.setprefill / 1000
        return {"candidates": n_cand, "runs": n_runs, "seqs_per_run": n_seq, "seconds_per_run": round(per, 3),
                "direction_seconds": dir_s, "replies_seconds": gen_s, "keys_seconds": keys_s,
                "seconds": dir_s + gen_s + keys_s + n_runs * per + 2.0}
    except Exception:  # noqa: BLE001  対照集合が読めなくても見積もりで止めない(本番がエラーを返す)
        return {"seconds": 60 * sp.setprefill / 1000, "guess": True}


def _key_causal_verify_cost(session, sp: Speed, p: dict) -> dict:
    """direction_key_causal_verify の見積もり: 条件(iv のみ + 候補 + 候補ごとの比較用の鍵)ごとに、書き込みたい場面と dont_touch の
    奇数番目の質問を max_tokens まで生成する(種ごと)。判定があれば読み手の分も。"""
    try:
        from .direction import _subset
        from .keyed_build import _resolve_mode, parse_positions
        c = p.get("contrast") or p.get("spec")
        if c is None and p.get("result"):
            from .keyed_causal import load_result
            c = load_result(session, p["result"])["contrast_path"]
        spec, _ = _spec_of({"spec": c} if isinstance(c, dict) else {"contrast": c})
        cs = spec.contrast
        mode, _ = _resolve_mode(parse_positions(spec.positions or "response"), spec.positions_b)
        side = "B" if mode == "same" else "A"
        n_a = len(_subset(len(cs.prompts(side)), "odd"))
        n_dt = 0 if p.get("dont_touch") is False else sum(len(_subset(len(dt.prompts), "odd")) for dt in spec.dont_touch)
        units = p.get("units") or []
        if isinstance(units, str):
            units = [u for u in units.split() if u]
        from .keyed_causal import parse_combos
        try:
            n_combo = len(parse_combos(p.get("combos")))
        except Exception:  # noqa: BLE001
            n_combo = 0
        n_c = 1 + 2 * (len(units) + n_combo)
        seeds = p.get("seeds")
        if isinstance(seeds, str):
            seeds = [x for x in seeds.split(",") if x.strip()]
        n_s = max(1, len(seeds or [0]))
        mt = int(p.get("max_tokens") or 256)
        k = session._engine_kw.get("n_seq_max", 8) or 8
        per = sp.gen(n_a, mt, k) + (sp.gen(n_dt, mt, k) if n_dt else 0.0)
        judge = (sp.gen(n_a + n_dt, 8) + (n_a + n_dt) * sp.setprefill / 1000) if p.get("judge") else 0.0
        return {"conditions": n_c, "combos": n_combo, "questions": n_a, "dont_touch": n_dt, "seeds": n_s, "max_tokens": mt,
                "seconds": n_c * n_s * (per + judge) + 1.0 * n_c}
    except Exception:  # noqa: BLE001
        return {"seconds": 3 * 32 * sp.step(8) / 1000 * 256, "guess": True}


def _eval_recipe(session, p: dict):
    """eval の見積もりのレシピ: params の recipe(Recipe か保存したレシピの名前・パス)か units(ユニットの書き方)。"""
    from .steer import Recipe, parse_unit
    r = p.get("recipe")
    if isinstance(r, Recipe):
        return r
    if p.get("units"):
        return Recipe([parse_unit(u) for u in p["units"]], session.meta.model_id)
    if r:
        return Recipe.load(session.recipe_path(r))
    return None


def _keyed_eval(session, sp: Speed, p: dict, n_nonzero: int, n_ctrl: int) -> dict | None:
    """eval が keyed のレシピで余分にする仕事(keyed_eval.KeyedMetrics、evalx.evaluate の進み具合と同じ数え方): 返事の生成
    (対照集合に返事が無い組だけ、1 回)+ 会話の prefill(無介入 1 回 + 0 以外の倍率ごと + 対照(比較用の鍵)ごと + 方式の比較と
    比較用の鍵の選択性 block_cost)。keyed でない・keyed_metrics=False なら None。対照集合が分からず eval が keyed を飛ばすなら 0 秒。
    戻り値 {"seconds", "controls"(比較用の鍵の本数で抑えた対照の数), 会話の数など}。"""
    if p.get("keyed_metrics") is False:
        return None
    try:
        r = _eval_recipe(session, p)
    except Exception:  # noqa: BLE001
        return None
    if r is None or not r.keyed_units:
        return None
    from .keyed_eval import DEFAULT_EXAMPLES, DEFAULT_PSEUDO, MATCH_STEPS
    mx = p.get("keyed_examples", DEFAULT_EXAMPLES) or None
    match = p.get("keyed_match", True) is not False
    pseudo = int(p.get("keyed_pseudo", DEFAULT_PSEUDO) or 0)

    def plan():                                   # keyed_eval.KeyedMetrics と keyed_build.eval_plan と同じ対照集合・位置・subset
        from .keyed import keyed_units, load_key, split_key_spec
        from .keyed_build import _as_spec, opposite_subset
        from .keyed_eval import resolve_keyed_contrast
        ku = keyed_units(r)
        ks = load_key(session, split_key_spec(ku[0].key)[0])
        kc = p.get("keyed_contrast")                 # 名前も受ける(Session.eval と同じ)
        kc = session._contrast_file(kc) if kc else None
        spec, _ = resolve_keyed_contrast(session, ks, session.direction_set(ku[0].direction), kc)
        if spec is None:
            return None, ks
        sub = p.get("keyed_subset") or opposite_subset(ks.subset)
        sets = _key_sets(_as_spec(spec), ks.positions or "response", ks.info.get("positions_b"), sub)
        return _plan_cost(sp, sets, int(ks.max_tokens), mx), ks
    try:
        cost, ks = plan()
    except Exception:  # noqa: BLE001  鍵・対照集合が読めなくても見積もりで止めない(eval が keyed を飛ばして warnings に書く)
        k = mx or DEFAULT_EXAMPLES
        cost, ks = {"conversations": 3 * k, "conversations_a": k, "generated": 0, "max_tokens": 0,
                    "seconds": 3 * k * sp.setprefill / 1000, "guess": True}, None
    if cost is None:                                             # eval は keyed を測らない(skipped)
        return {"seconds": 0.0, "controls": n_ctrl, "skipped": True}
    if ks is not None:
        n_ctrl = min(n_ctrl, int(ks.n_pseudo))
        pseudo = min(pseudo, int(ks.n_pseudo))
    n, n_a = cost["conversations"], cost["conversations_a"]
    block = (2 + (3 if match else 0)) * n + pseudo * n + (3 * MATCH_STEPS * n_a if match else 0)
    prefills = n * (1 + n_nonzero + n_ctrl * n_nonzero) + block
    gen_s = cost["seconds"] - n * sp.setprefill / 1000
    sec = gen_s + prefills * sp.setprefill / 1000
    return {**cost, "prefills": prefills, "pseudo": pseudo, "match": match, "controls": n_ctrl, "seconds": round(sec, 1)}


def fmt_seconds(s: float | None) -> str:
    if s is None:
        return "—"
    if s < 90:
        return f"{s:.0f} 秒"
    if s < 5400:
        return f"{s / 60:.0f} 分"
    return f"{s / 3600:.1f} 時間"


def fit_count(per_unit_s: float, budget_s: float, fixed_s: float = 0.0, *, lo: int = 1, hi: int | None = None) -> int:
    """予算に収まるユニット数。"""
    n = int(max(0.0, budget_s - fixed_s) // max(per_unit_s, 1e-9))
    n = max(lo, n)
    return min(n, hi) if hi is not None else n


def fit_window(session, budget_s: float) -> int:
    """connections の予算に収まる最大の window(1 以上)。"""
    m = session.meta
    from .connections import dense_layers
    Ls = dense_layers(session.weights)
    ref = 9216 * 2560 * 9216
    best = 1
    for w in range(1, len(Ls)):
        pairs = [(i, j) for jj, j in enumerate(Ls) for i in Ls[:jj] if j - i <= w]
        if sum(4.0 * m.n_ff_at(i) * m.n_embd * m.n_ff_at(j) / ref for i, j in pairs) > budget_s:
            break
        best = w
    return best


def recommended_workflow(session) -> list[dict]:
    """大型モデル向けの手順と見積もり: 逆引き → 上位 N の lens → quick sweep → eval。"""
    sp = Speed(session)
    L = session.meta.n_layer // 2
    lens_per = 3.1 * sp.setprefill / 1000
    quick = _design_cost(sp, "quick", 128)
    steps = [
        {"step": "gne reverse --contrast <対照集合> --top 200", "seconds": 6.0 * sp.setprefill / REF["setprefill_ms"] + 30},
        {"step": f"gne lens steer --layer {L} --neurons <逆引きの上位 200>", "seconds": (200 + 64) * lens_per},
        {"step": "gne sweep --design quick --neurons <lens の KL 上位 50>", "seconds": (50 + 16) * quick},
        {"step": "gne eval --recipe <レシピ>", "seconds": estimate(session, "eval", {})["seconds"]},
    ]
    return steps


def is_large(session) -> bool:
    """全層が GPU に載らないか、全数の lens steer が 1 層 1 時間を超えるモデル。"""
    sp = Speed(session)
    n = max(session.meta.n_ff_at(L) for L in range(session.meta.n_layer))
    return sp.cpu_frac > 0 or n * 3.1 * sp.setprefill / 1000 > 3600


def ceil_div(a: int, b: int) -> int:
    return int(math.ceil(a / b))


# --- 予算と確認(job_start・CLI) ------------------------------------------------------------------------

class NeedsConfirm(Exception):
    """見積もりが CONFIRM_S を超えた(--yes か予算が要る)。estimate を持つ。"""

    def __init__(self, est: dict):
        from .steer import GneError
        self.estimate = est
        self.error = GneError("NEEDS_CONFIRM", f"{est['kind']} の見積もりは {fmt_seconds(est['seconds'])} です(30 分を超えるので止めました)",
                              "--yes(MCP は params の yes: true)で走らせるか、--budget 30m で絞ってください")
        super().__init__(self.error.message)


def apply_budget(session, kind: str, params: dict, budget_s: float) -> tuple[dict, dict]:
    """予算に収まるように params を絞る。返り値は (新しい params, 何をしたか)。
    - lens_steer: 層 L に書き込む向きが方向(params の direction、無ければ最後に作った方向)に近い順の上位から、収まる本数だけ。
    - sweep: quick 設計にし、候補は同じ層の steer lens の KL の大きい順から収まる本数(steer lens が無ければランダム)。
    - reverse(verify): 確かめる候補の数を収まる本数に。
    - connections: 収まる最大の window。"""
    p = dict(params)
    note: dict = {"budget_s": budget_s}
    sp = Speed(session)
    if kind == "lens_steer":
        L = int(p["layer"])
        n_ctrl = int(p.get("random_controls", 64))
        per = 3.1 * sp.setprefill / 1000
        n = fit_count(per, budget_s, n_ctrl * per, hi=session.meta.n_ff_at(L))
        top = _writer_top(session, L, n, p.get("direction"))
        p["neurons"] = ",".join(str(x) for x in top)
        note.update({"neurons": len(top), "from": "direction" if top else "none"})
    elif kind in ("sweep", "sweep_rank"):
        L = int(p["layer"])
        p["design"] = p.get("design") if p.get("design") and p.get("design") != "main_rel" else "quick"
        n_ctrl = min(int(p.get("random_controls", 64)), 16)
        p["random_controls"] = n_ctrl
        per = _design_cost(sp, p["design"], int(p.get("max_tokens", 128)), p.get("prompt_set", "en"))
        n = fit_count(per, budget_s, n_ctrl * per, hi=session.meta.n_ff_at(L))
        top = _lens_top(session, L, n)
        p["neurons"] = ",".join(str(x) for x in top) if top else f"random:{n}"
        note.update({"neurons": n, "design": p["design"], "from": "steer lens" if top else "random"})
    elif kind == "reverse" and p.get("verify"):
        # 候補の層の数は順位づけの後でないと分からないので、本数は reverse の中で決める(対照は候補の層ごと)
        p["verify_budget_s"] = budget_s - 6.0 * sp.setprefill / REF["setprefill_ms"] - 30
        note.update({"verify_budget_s": p["verify_budget_s"]})
    elif kind == "connections":
        p["window"] = fit_window(session, budget_s)
        note.update({"window": p["window"]})
    return p, note


def _writer_top(session, L: int, n: int, direction: str | None) -> list[int]:
    """層 L のニューロンを、方向の r_L との |cos| の大きい順に n 個(方向が無ければ空)。"""
    import numpy as np
    try:
        if direction is None:
            files = sorted(session.store.direction_dir(session.meta.model_id).glob("*.npz"), key=lambda p: -p.stat().st_mtime)
            files = [f for f in files if not f.name.endswith(".select.npz")]
            if not files:
                return []
            direction = str(files[0])
        ds = session.direction_set(direction)
    except Exception:  # noqa: BLE001
        return []
    if L not in ds.raw:
        return []
    r = ds.raw[L] / max(float(np.linalg.norm(ds.raw[L])), 1e-12)
    cos = np.abs(r @ session.weights.unit_matrix(L))
    return [int(x) for x in np.argsort(-cos)[:n]]


def _lens_top(session, L: int, n: int) -> list[int]:
    """同じ層の steer lens(済みのニューロン)の KL の大きい順に n 個。"""
    import numpy as np
    from .lens import SteerLens
    p = session.store.lens_dir(session.meta.model_id) / f"steer_L{L}.npz"
    if not p.exists():
        return []
    sl = SteerLens.load(p)
    kl = np.where(sl.done, np.maximum(np.nan_to_num(sl.kl_pos), np.nan_to_num(sl.kl_neg)), -1.0)
    order = [int(x) for x in np.argsort(-kl) if sl.done[x]]
    return order[:n]


def gate(session, kind: str, params: dict) -> tuple[dict, dict | None]:
    """job_start と CLI の前に呼ぶ。params の budget(秒か "30m")で絞り、yes が無く見積もりが 30 分を超えれば NeedsConfirm。
    返り値は (実行する params、見積もり)。見積もれない種類はそのまま。"""
    p = dict(params or {})
    yes = bool(p.pop("yes", False))
    budget = p.pop("budget", None)
    note = None
    if budget is not None:
        b = budget if isinstance(budget, (int, float)) else parse_budget(budget)
        p, note = apply_budget(session, kind, p, float(b))
    try:
        est = estimate(session, kind, p)
    except Exception:  # noqa: BLE001  見積もりの失敗で処理そのものは止めない
        return p, None
    if note is not None:
        est["budget"] = note
    if est["seconds"] is not None and est["seconds"] > CONFIRM_S and not yes:
        raise NeedsConfirm(est)
    return p, est
