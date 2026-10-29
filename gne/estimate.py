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
        self.gb = Path(session.weights.path).stat().st_size / 1e9
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
        n_conv = 2 * len(cs.user_prompts)
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
        sec = per * (n_d + n_ctrl * (n_d - 1))
        detail = {"doses": n_d, "seconds_per_dose": round(per, 1)}
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
    elif kind == "stats_resid":
        sec = 10.0 * sp.setprefill / REF["setprefill_ms"]
    return {"kind": kind, "seconds": None if sec is None else float(sec), "detail": detail, "speed": sp.to_dict(),
            "needs_confirm": bool(sec is not None and sec > CONFIRM_S)}


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
