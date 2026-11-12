"""FiSH(Formula-instantiated Scene-gated Hidden units): 計算で作るニューロン(mode nw)。MLP のニューロンを、学習なしで計算した重みで上書きする。docs/fish.md。

ゲートつきの MLP(SwiGLU、バイアスなし)は、層の入力 h(正規化の後)から

    out = W_down · ( silu(W_gate h) ⊙ (W_up h) )

を残差に足す。ニューロン j は W_gate・W_up の j 行と W_down の j 列で決まる。ここでは、ふだんあまり働いていない
ニューロン j の 3 か所を差し替えて、「鍵が反応する位置でだけ方向 d を書く」ニューロンにする。

- 鍵 1 つの形:   gate[j] ← s·κ、up[j] ← κ、down[:, j] ← α·d ÷ silu(s)
                  足す量 = silu(s·ρ)·ρ ÷ silu(s) × α·d(ρ = κᵀh。ρ = 1 でちょうど α·d、小さい反応は 2 乗に近い形で潰れ、
                  負の反応ではほぼ書かない)
- 鍵 2 つの形:   gate[j] ← s·κ_A、up[j] ← κ_B、down[:, j] ← α·d ÷ silu(s)
                  足す量 = silu(s·ρ_A)·ρ_B ÷ silu(s) × α·d(A かつ B。例: 場面の鍵と、返事の位置の鍵)
- 上限つきの形:  ニューロン 2 個を組にする。κ1 は「どの位置でもほぼ 1 を返す鍵」(バイアスの代わり)。
                  1 個目 gate ← s(κ − a·κ1)、up ← κ1、down ← +α·d ÷ (s(b − a))
                  2 個目 gate ← s(κ − b·κ1)、up ← κ1、down ← −α·d ÷ (s(b − a))
                  足す量 = [silu(s(ρ − a)) − silu(s(ρ − b))] ÷ (s(b − a)) × (κ1ᵀh) × α·d(ρ < a でほぼ 0、ρ > b でほぼ 1 で頭打ち)

鍵は MLP の入力の空間(target mlpin、gne direction key --target mlpin)。方向は gne direction の方向(その層の平均差 r_L を
そのままの大きさで使う。keyed と同じ)。当て方は実行時の LoRA(ffn_gate・ffn_up・ffn_down への差し替え)だけで、焼き込みは無い
(Q8_0 の ffn_down は 32 個で 1 つの倍率を共有するので、1 列だけを大きな値に替えると誤差が 9 割を超える)。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import keyed as K
from .steer import GneError

# ゲートつきの MLP を build_ffn(…, LLM_FFN_SILU, LLM_FFN_PAR)で作り、MLP の入力に下の名前を付ける arch
# (llama.cpp の src/models/<arch>.cpp を読んで確かめたものだけ。ほかは ARCH_UNSUPPORTED)
SWIGLU_ARCHS = {"llama": "ffn_norm", "qwen2": "ffn_norm", "qwen3": "ffn_norm", "qwen35": "attn_post_norm"}
S_DEFAULT, S_CAP, CAP_DEFAULT = 4.0, 16.0, (0.4, 0.8)


def silu(z):
    z = np.asarray(z, dtype=np.float64)
    return z / (1.0 + np.exp(-z))


def amount(rho_gate, rho_up, s: float = S_DEFAULT):
    """鍵 1 つ・鍵 2 つの形で足す量(α·d を 1 として)。鍵 1 つなら rho_up = rho_gate。"""
    return silu(float(s) * np.asarray(rho_gate, dtype=np.float64)) * np.asarray(rho_up, dtype=np.float64) / float(silu(s))


def amount_cap(rho, one=1.0, s: float = S_CAP, a: float = CAP_DEFAULT[0], b: float = CAP_DEFAULT[1]):
    """上限つきの形で足す量(α·d を 1 として)。one = κ1ᵀh(ほぼ 1)。"""
    rho = np.asarray(rho, dtype=np.float64)
    return (silu(s * (rho - a * one)) - silu(s * (rho - b * one))) / (s * (b - a)) * one


def check_mlp(weights, L: int) -> None:
    """層 L が、計算で作るニューロンを入れられる MLP か(dense・ゲートつき・SwiGLU・バイアスなし)。違えばエラー。"""
    m = weights.meta
    if not 0 <= int(L) < m.n_layer:
        raise GneError("BAD_RECIPE", f"層 {L} は範囲外です(0..{m.n_layer - 1})")
    if m.n_expert and weights.is_moe(L):
        raise GneError("ARCH_UNSUPPORTED", f"層 {L} は MoE です。計算で作るニューロン(nw)は dense の MLP の層だけです")
    names = [f"blk.{L}.ffn_{p}.weight" for p in ("gate", "up", "down")]
    missing = [n for n in names if not weights.has_tensor(n)]
    if missing:
        raise GneError("ARCH_UNSUPPORTED", f"層 {L} に {', '.join(missing)} がありません。計算で作るニューロン(nw)は、ゲートつきの MLP"
                                           "(out = W_down(act(W_gate h) ⊙ W_up h))の層だけです")
    if m.arch not in SWIGLU_ARCHS:
        raise GneError("ARCH_UNSUPPORTED", f"arch {m.arch} の MLP の形(活性化関数・ゲートの掛け方)は確かめていないので、nw は使えません",
                       f"確かめてある arch: {', '.join(sorted(SWIGLU_ARCHS))}(SiLU のゲート × 入力、バイアスなし)")
    bias = [n for p in ("gate", "up", "down") for n in (f"blk.{L}.ffn_{p}.bias",) if weights.has_tensor(n)]
    if bias:
        raise GneError("ARCH_UNSUPPORTED", f"層 {L} の MLP にバイアス({', '.join(bias)})があります。nw はバイアスの無い MLP だけです")
    if (m.ffn_down_scale or {}).get(L) not in (None, 1.0) or any(weights.has_tensor(f"blk.{L}.ffn_{p}.scale") for p in ("gate", "up", "down")):
        raise GneError("ARCH_UNSUPPORTED", f"層 {L} の MLP の重みに倍率(.scale)があります。nw は確かめていません")
    if K.is_mtp_layer(weights, L):
        raise GneError("BAD_RECIPE", f"層 {L} は MTP(nextn)の層です。推論の本流では実行されません")


def mlp_in_name(seen: set[str], L: int) -> str:
    """層 L の MLP の入力(正規化の後)の捕捉名。arch で決まる名前を、見えた名前から探す。"""
    for base in dict.fromkeys(list(SWIGLU_ARCHS.values())):
        if f"{base}-{L}" in seen and f"ffn_gate-{L}" in seen:
            return f"{base}-{L}"
    raise GneError("ARCH_UNSUPPORTED", f"層 {L} の MLP の入力(正規化の後)のテンソルが捕捉できません")


def _gate_matrix(weights, L: int) -> np.ndarray:
    from .weights import tensor_to_f32
    return tensor_to_f32(weights.tensor(f"blk.{L}.ffn_gate.weight"))


def form_of(u) -> dict:
    """ユニットの形: one(鍵 1 つ)/ two(鍵 2 つ)/ cap(上限つき、2 個を組)と、s・a・b。"""
    if u.pair is not None:
        a, b = CAP_DEFAULT
        if u.cap:
            m = re.fullmatch(r"([\d.]+)-([\d.]+)", str(u.cap))
            a, b = float(m.group(1)), float(m.group(2))
        return {"form": "cap", "s": float(u.s or S_CAP), "a": a, "b": b}
    return {"form": "two" if u.key2 else "one", "s": float(u.s or S_DEFAULT)}


def spec_of(u) -> str:
    f = form_of(u)
    return (f"{u.layer}:N{u.neuron}" + (f"+N{u.pair}" if u.pair is not None else "") + f":nw:a{u.k:g}"
            + (f"@{u.direction}" if u.direction else "") + (f"#{u.key}" if u.key else "") + (f"&{u.key2}" if u.key2 else "")
            + (f"!s{f['s']:g}" if u.s is not None else "") + (f"!cap{u.cap}" if u.cap else ""))


def nw_plan(weights, recipe, key_of, dir_of) -> list[dict]:
    """レシピの nw のユニット → 行(ユニットごと)。各行に、差し替えるニューロンごとの (番号, ゲートの行, 入力の行, 出力の列) と、
    量の換算(そのニューロンの活性に掛けると「α·d を 1 とした足す量」への寄与になる倍率)を入れる。確かめもここで行う。"""
    rows, used = [], {}
    for u in recipe.nw_units:
        L = int(u.layer)
        check_mlp(weights, L)
        if L == K.last_main_layer(weights):
            raise GneError("BAD_RECIPE", f"層 {L} は最終層です。捕捉の行が間引かれるので鍵を作れません", "最終層より前の層を使ってください")
        n_ff, n_embd = weights.meta.n_ff_at(L), weights.meta.n_embd
        f = form_of(u)
        js = [int(u.neuron)] + ([int(u.pair)] if u.pair is not None else [])
        for j in js:
            if not 0 <= j < n_ff:
                raise GneError("BAD_RECIPE", f"{u.label()}: L{L} のニューロン {j} は範囲外です(0..{n_ff - 1})")
            if (L, j) in used:
                raise GneError("BAD_RECIPE", f"L{L} のニューロン {j} を 2 つのユニット({used[(L, j)]}、{u.label()})で上書きしようとしています",
                               "ユニットごとに別のニューロンを使ってください(gne nw neurons で働きの小さいものを選べます)")
            used[(L, j)] = u.label()
        dname = u.direction or recipe.default_direction
        kname = u.key or recipe.default_key
        d = np.asarray(K.raw_direction(dir_of(dname), L)[0], dtype=np.float64)
        kg = _kappa(key_of, kname, L, n_embd, u)
        ku = _kappa(key_of, u.key2, L, n_embd, u) if u.key2 else kg
        s, alpha = f["s"], float(u.k)
        if f["form"] == "cap":
            g = 1.0 / (s * (f["b"] - f["a"]))
            parts = [(js[0], s * (kg - f["a"] * ku), ku, alpha * g * d, g), (js[1], s * (kg - f["b"] * ku), ku, -alpha * g * d, -g)]
        else:
            g = 1.0 / float(silu(s))
            parts = [(js[0], s * kg, ku, alpha * g * d, g)]
        rows.append({"unit": u, "layer": L, "direction": dname, "key": kname, "key2": u.key2, **f, "alpha": alpha,
                     "kappa_gate": kg, "kappa_up": ku, "d": d, "parts": parts, "norm": float(abs(alpha) * np.linalg.norm(d))})
    return rows


def _kappa(key_of, name: str | None, L: int, n_embd: int, u) -> np.ndarray:
    if not name:
        raise GneError("BAD_RECIPE", f"{u.label()}: 鍵がありません(key か、レシピの default_key)")
    ks = key_of(name)
    if ks.target != K.MLPIN:
        raise GneError("BAD_RECIPE", f"{u.label()}: 鍵 {name} の書き込み先は {ks.target} です。nw には MLP の入力の空間の鍵(mlpin)が要ります",
                       "gne direction key --target mlpin … で作ります(keyed の ffn の鍵は ffn_down の入力の空間で、次元が違います)")
    if L not in ks.kappa:
        raise GneError("BAD_RECIPE", f"{u.label()}: 鍵 {name} に層 {L} がありません({ks.layers})")
    k = np.asarray(ks.kappa_of(L), dtype=np.float64).reshape(-1)
    if k.size != n_embd:
        raise GneError("BAD_KEY", f"鍵 {name} の層 {L} の長さ {k.size} が n_embd {n_embd} と違います")
    return k


def nw_tensors(weights, rows: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """行 → LoRA のテンソル {テンソル名: (A, B)}。層ごとに、ffn_gate・ffn_up は A = (新しい行 − 元の行)、B = e_j、
    ffn_down は A = e_j、B = (新しい列 − 元の列)。元の行・列は float に戻した値(量子化のままの掛け算との小さなずれが残る)。"""
    by_layer: dict[int, list] = {}
    for r in rows:
        by_layer.setdefault(r["layer"], []).extend(r["parts"])
    out = {}
    for L, parts in sorted(by_layer.items()):
        n_ff = weights.meta.n_ff_at(L)
        Wg, Wu, Wd = _gate_matrix(weights, L), weights.up_matrix(L), weights.down_matrix(L)
        E = np.zeros((len(parts), n_ff), dtype=np.float32)
        Ag, Au, Bd = [], [], []
        for i, (j, gate, up, down, _) in enumerate(parts):
            E[i, j] = 1.0
            Ag.append(np.asarray(gate - Wg[j].astype(np.float64), dtype=np.float32))
            Au.append(np.asarray(up - Wu[j].astype(np.float64), dtype=np.float32))
            Bd.append(np.asarray(down - Wd[:, j].astype(np.float64), dtype=np.float32))
        out[f"blk.{L}.ffn_gate.weight"] = (np.stack(Ag), np.ascontiguousarray(E.T))
        out[f"blk.{L}.ffn_up.weight"] = (np.stack(Au), np.ascontiguousarray(E.T))
        out[f"blk.{L}.ffn_down.weight"] = (E.copy(), np.stack(Bd, axis=1))
    return out


def nw_meta(rows: list[dict]) -> dict:
    return {"units": [{"layer": r["layer"], "neurons": [p[0] for p in r["parts"]], "form": r["form"], "s": r["s"],
                       **({"a": r["a"], "b": r["b"]} if r["form"] == "cap" else {}), "alpha": r["alpha"],
                       "direction": r["direction"], "key": r["key"], "key2": r["key2"], "norm": r["norm"]} for r in rows]}


def cache_hash(tens: dict) -> str:
    h = hashlib.sha256()
    for name in sorted(tens):
        for x in tens[name]:
            x = np.ascontiguousarray(x)
            h.update(f"{name}|{x.shape}|".encode())
            h.update(x.tobytes())
    return h.hexdigest()[:16]


NEUTRAL_NAME = "adapter"                       # strip_meta で名前を指定しないときの general.name


def export_nw_lora(weights, recipe, path: str | Path, key_of, dir_of, *, strip_meta: bool = False, name: str | None = None) -> dict:
    """レシピの nw のユニットを LoRA アダプタ GGUF に書く(倍率 1 で当てる)。
    name はファイルの名前の欄(general.name)。省くと「<元のモデルの名前> nw <レシピの名前>」。
    strip_meta(人に渡す版)は、gne の記録(gne.nw)に加えて、元のモデルを特定できる値も書かない: 元のモデルの識別子
    (gne.source_model_id)を書かず、名前の欄は name(省くと "adapter")にして、元のモデルの名前・レシピの名前を入れない。"""
    from .ablate import write_lora_tensors
    rows = nw_plan(weights, recipe, key_of, dir_of)
    if not rows:
        raise GneError("BAD_RECIPE", "nw のユニットがありません(例 --unit 11:N2045:nw:a1@NAME#KEY)")
    tens = nw_tensors(weights, rows)
    kv = None if strip_meta else {"gne.nw": json.dumps(nw_meta(rows), ensure_ascii=False)}
    if name is not None and not str(name).strip():
        raise GneError("BAD_ARGS", "名前(name)が空です")
    label = str(name).strip() if name is not None else (
        NEUTRAL_NAME if strip_meta else f"{weights.meta.name} nw" + (f" {recipe.name}" if recipe.name else ""))
    rep = write_lora_tensors(weights, tens, path, name=label, kv=kv, source_id=not strip_meta)
    return {**rep, "general_name": label, "strip_meta": bool(strip_meta), "nw": nw_meta(rows)["units"]}


# --- ふだんの働き(上書きするニューロンを選ぶ材料)---------------------------------------------------------------

def neuron_usage(session, layers: Sequence[int], *, prompts: Sequence[str] | None = None, max_tokens: int = 96,
                 corpus_tokens: int = 2048, top: int = 20, sort: str = "sum", iv=None) -> dict:
    """層ごとに、各ニューロンのふだんの働きを測り、働きの小さい順の表を返す。
    働き = |silu(gate)·up| × ‖down の列‖(そのニューロンが 1 位置で残差に書くベクトルの長さ)。読ませるのは、ふつうの質問
    (既定のプロンプト集の英語・日本語)への無介入の返事と、既定のコーパスの一部。sort: sum(最大の順位 + 平均の順位、既定)/
    max / mean。表を見て使う人が選ぶ(番号はレシピのユニットに書く)。iv(レシピの名前かパス、またはユニットの書き方)を渡すと、
    その介入を当てた状態で測る(例: CHiPS のレシピを当てた状態で、働きの小さいニューロンを選ぶ)。"""
    import yaml

    from .capture import Capture, pick_act_name
    from .evalx import _generate
    from .stats import corpus_chunks, load_corpus
    e, w = session.engine, session.weights
    Ls = sorted({int(L) for L in layers})
    for L in Ls:
        check_mlp(w, L)
    if sort not in ("sum", "max", "mean"):
        raise GneError("BAD_ARGS", f"sort は sum / max / mean です({sort!r})")
    if prompts is None:
        d = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "prompts" / "default.yaml").read_text(encoding="utf-8"))
        prompts = [p for k in ("en", "ja") for p in (d.get(k) or {}).values()]
    t0 = time.perf_counter()
    from .keyed_build import intervention
    vecs, lora, iv_desc = intervention(session, iv)
    seqs = []
    if prompts:
        outs = _generate(session, [(p, int(max_tokens)) for p in prompts], vecs, lora, "all", lambda k: None)
        for p, o in zip(prompts, outs, strict=True):
            pid = e.tokenize_prompt(e.render_chat([{"role": "user", "content": p}]))
            seqs.append(pid + [t for t in o["token_ids"] if not e.is_eog(t)])
    if corpus_tokens:
        seqs += corpus_chunks(e, load_corpus(), 512, int(corpus_tokens))
    seen = e.list_names(seqs[0][:16])
    names = {}
    for L in Ls:
        nm = pick_act_name(seen, L)
        if nm is None:
            raise GneError("ARCH_UNSUPPORTED", f"層 {L} のニューロンの活性が捕捉できません")
        names[L] = nm
    wanted = set(names.values()) | {f"ffn_out-{L}" for L in Ls}
    cap = Capture(lambda n: n in wanted)
    dn = {L: np.asarray(w.down_norms(L), dtype=np.float64) for L in Ls}
    acc = {L: {"sum": np.zeros(dn[L].size), "max": np.zeros(dn[L].size), "n": 0, "out": []} for L in Ls}
    for ids in seqs:
        ctx = e.probe_ctx(cap, need=len(ids))
        e.set_cvec(ctx, vecs or None)
        e.set_lora(ctx, lora or None)
        try:
            e.clear_memory(ctx)
            e.prefill(ctx, [ids])
            got = cap.take()
        finally:
            e.set_cvec(ctx, None)
            e.set_lora(ctx, None)
        for L in Ls:
            c = np.abs(got[names[L]][1:]).astype(np.float64) * dn[L][None, :]
            a = acc[L]
            a["sum"] += c.sum(0)
            a["max"] = np.maximum(a["max"], c.max(0))
            a["n"] += c.shape[0]
            if f"ffn_out-{L}" in got:
                a["out"].append(np.linalg.norm(got[f"ffn_out-{L}"][1:], axis=1))
    rk = lambda v: np.argsort(np.argsort(v)) + 1
    out_layers = []
    for L in Ls:
        a = acc[L]
        mean, mx = a["sum"] / max(a["n"], 1), a["max"]
        r_max, r_mean = rk(mx), rk(mean)
        key = {"sum": r_max + r_mean, "max": mx, "mean": mean}[sort]
        order = np.argsort(key, kind="stable")[:int(top)]
        out_med = float(np.median(np.concatenate(a["out"]))) if a["out"] else None
        out_layers.append({"layer": L, "n_ff": int(mean.size), "positions": int(a["n"]), "out_norm_median": out_med,
                           "median": {"mean": float(np.median(mean)), "max": float(np.median(mx))},
                           "min": {"mean": float(mean.min()), "max": float(mx.min())},
                           "rows": [{"neuron": int(j), "mean": float(mean[j]), "max": float(mx[j]),
                                     "max_share": None if not out_med else float(mx[j] / out_med),
                                     "rank_max": int(r_max[j]), "rank_mean": int(r_mean[j]), "down_norm": float(dn[L][j])}
                                    for j in order]})
    return {"layers": out_layers, "sort": sort, "top": int(top), "n_prompts": len(prompts or []), "max_tokens": int(max_tokens),
            "corpus_tokens": int(corpus_tokens), "iv": iv_desc,
            "notes": ["働きの小さい順の表です。どれを上書きするかは表を見て選び、レシピのユニットに番号を書きます。",
                      "上書きすると、そのニューロンの元の働きは無くなります。"],
            "seconds": time.perf_counter() - t0}


# --- 鍵(場面の鍵・ほぼ 1 を返す鍵)----------------------------------------------------------------------------
# 位置の鍵(返事の位置の区切りで反応する鍵)は、ふつうの鍵づくり(keyed_build.build_keys、target mlpin、positions pos:<範囲>)で
# 作る。ここは、質問の種類で反応して返事の間ずっと開く「場面の鍵」と、上限つきの形で使う「ほぼ 1 を返す鍵」。

SCENE_BUCKETS = ("pos:0", "pos:1-3", "pos:4-15", "pos:16-")


def build_role_key(session, contrast, *, role: str, layers, name: str | None = None, subset: str = "even",
                   max_tokens: int = 96, pseudo: int = 16, iv=None, seed: int = 0, q_energy: float = 0.99,
                   progress=None) -> dict:
    """role = scene: 対照集合の質問(user_prompts)への返事の位置で反応し、dont_touch の群の質問への返事では反応しない鍵。
      返事の位置を 0 / 1〜3 / 4〜15 / 16 以降に分けた例ごとの平均を行にして、keyed と同じ式(κ = k/(μᵀk)、k = μ − QQᵀμ)で作る。
      比較用の鍵は、質問のラベル(狙う質問か、dont_touch か)をでたらめに入れ替えて同じ手順で作る。
    role = one: どの位置でもほぼ 1 を返す鍵(κ1ᵀh ≈ 1 の最小二乗)。上限つきの形で、バイアスの代わりに使う。
    どちらも推論時と同じ文脈(system_b。省けば system なし)で、モデル自身の返事(iv があればそれを当てた状態)を読む。"""
    from .keyed_build import (MlpInCapture, _as_spec, balanced_split, collect_acts, intervention)
    from .direction import _subset
    from .reverse import _conv
    if role not in ("scene", "one"):
        raise GneError("BAD_ARGS", f"role は scene / one です({role!r})")
    t0 = time.perf_counter()
    spec = _as_spec(contrast)
    cs = spec.contrast
    Ls = sorted({int(L) for L in (layers or [])})
    if not Ls:
        raise GneError("BAD_ARGS", "鍵の層(layers)を指定してください")
    for L in Ls:
        K.check_layer_ok(session.weights, L, K.MLPIN)
    name = name or f"{cs.name}_{role}"
    if not re.fullmatch(r"[\w.][\w.\-]*", name):
        raise GneError("BAD_ARGS", f"鍵の名前 {name!r} が不正です(英数字・_・.・- だけ)")
    if role == "scene" and not spec.dont_touch:
        raise GneError("BAD_CONTRAST", "場面の鍵には、反応させたくない質問の群(dont_touch)が要ります",
                       "対照集合の dont_touch に、ふつうの質問・計算や事実の質問などを入れてください")
    prompts = cs.prompts("B")
    idx = _subset(len(prompts), subset)
    sets = [("A", [_conv(cs.system_b, prompts[i]) for i in idx])]
    for dt in spec.dont_touch:
        sets.append((f"dt:{dt.name}", dt.convs(_subset(len(dt.prompts), subset))))
    pos = {f"s{i}": b for i, b in enumerate(SCENE_BUCKETS)}
    tc = MlpInCapture(session, Ls)
    _, _, iv_desc = intervention(session, iv)
    got = {}
    for k, (label, convs) in enumerate(sets):
        got[label] = collect_acts(session, convs, layers=Ls, positions=pos, max_tokens=max_tokens, iv=iv, target=tc,
                                  samples=2048 if role == "one" else 0, seed=seed)
        if progress:
            progress(k + 1, len(sets), f"key: {label}")
    n_q = {label: len(convs) for label, convs in sets}

    def ex_rows(label, L):
        """(質問ごとの行のリスト): 質問 i の、区切りごとの例の平均の行。"""
        g = got[label]["groups"]
        per = {i: [] for i in range(n_q[label])}
        for b in pos:
            for row, i in zip(g[b]["means"][L], g[b]["index_l"][L], strict=True):
                per[i].append(row)
        return [np.stack(v) for i, v in sorted(per.items()) if v]

    rng = np.random.default_rng(seed)
    kappa, mu, kn, gain, qrank, exa, exb, ctrl, rows_out = {}, {}, {}, {}, {}, {}, {}, {}, []
    for L in Ls:
        A = ex_rows("A", L)
        B = [r for label, _ in sets[1:] for r in ex_rows(label, L)]
        if role == "one":
            X = np.concatenate([got[label]["groups"][b]["samples"][L] for label, _ in sets for b in pos
                                if got[label]["groups"][b]["samples"][L].shape[0]]).astype(np.float64)
            lam = 1e-2 * float(np.trace(X.T @ X)) / X.shape[1]
            k1 = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.sum(0))
            r = X @ k1
            kappa[L], mu[L], kn[L], gain[L] = k1, X.mean(0), 1.0, 1.0
            rows_out.append({"layer": L, "rows": int(X.shape[0]), "response_mean": float(r.mean()), "response_sd": float(r.std()),
                             "response_min": float(r.min()), "response_max": float(r.max())})
            continue
        if not A or not B:
            raise GneError("BAD_CONTRAST", f"L{L}: 場面の鍵の行がありません(返事が空です)")
        m = np.concatenate(A).mean(0)
        kap, info = K.make_key(m, K.orth_basis(np.concatenate(B), energy=q_energy), warn=session.warn, label=f"L{L}")
        kappa[L], mu[L], kn[L], gain[L], qrank[L] = kap, m, info["kn"], info["gain"], info["q_rank"]
        ea, eb = np.stack([x.mean(0) for x in A]), np.stack([x.mean(0) for x in B])
        exa[L], exb[L] = ea.astype(np.float32), eb.astype(np.float32)
        pool = A + B
        C = []
        for _ in range(int(pseudo)):
            ia, ib = balanced_split(rng, len(A), len(B))
            try:
                C.append(K.make_key(np.concatenate([pool[i] for i in ia]).mean(0),
                                    K.orth_basis(np.concatenate([pool[i] for i in ib]), energy=q_energy))[0])
            except GneError:
                C.append(np.zeros_like(kap))
        if C:
            ctrl[L] = np.stack(C).astype(np.float32)
        sel = float((ea @ kap).mean() - (eb @ kap).mean())
        psel = [float((ea @ c).mean() - (eb @ c).mean()) for c in C]
        rows_out.append({"layer": L, "kn": info["kn"], "q_rank": info["q_rank"], "examples_a": len(A), "examples_b": len(B),
                         "build_selectivity": sel, "pseudo_selectivity_p95": float(np.quantile(psel, 0.95)) if psel else None})
    ks = K.KeySet(target=K.MLPIN, kappa=kappa, mu=mu, kn=kn, gain=gain, q_rank=qrank, exa=exa, exb=exb, kappa_ctrl=ctrl,
                  q_mode="examples", positions=f"role:{role}", max_tokens=int(max_tokens), subset=subset, contrast_sha=cs.sha,
                  counts={"a": n_q["A"], "b": sum(v for k, v in n_q.items() if k != "A")},
                  info={"role": role, "contrast": cs.name, "iv": iv_desc, "buckets": list(SCENE_BUCKETS), "pseudo": len(ctrl.get(Ls[0], [])),
                        "seed": seed, "dont_touch": [dt.name for dt in spec.dont_touch]},
                  model_id=session.meta.model_id, name=name)
    path = ks.save(K.key_dir(session) / f"{name}.npz")
    return {"name": name, "path": str(path), "target": K.MLPIN, "role": role, "contrast": cs.name, "subset": subset,
            "max_tokens": int(max_tokens), "layers": rows_out, "n_pseudo": ks.n_pseudo, "iv": iv_desc,
            "notes": (["build_selectivity は作成に使った質問での値です(甘い)。作成に使っていない質問では gne nw check で確かめます"]
                      if role == "scene" else ["上限つきの形(L:Nn+Nm:nw:…#鍵&この鍵)の 2 つ目の鍵に使います"]),
            "sha": ks.sha, "seconds": time.perf_counter() - t0}


# --- 確かめ --------------------------------------------------------------------------------------------------

VERIFY_TEXTS = ["The quick brown fox jumps over the lazy dog. It was a bright cold day in April.",
                "今日は天気が良いので、散歩に出かけることにしました。公園の桜が咲き始めています。"]


def _capture_names(session, ids, L):
    e = session.engine
    seen = e.list_names(ids[:16])
    from .capture import pick_act_name
    act = pick_act_name(seen, L)
    if act is None:
        raise GneError("ARCH_UNSUPPORTED", f"層 {L} のニューロンの活性が捕捉できません")
    return {"h": mlp_in_name(seen, L), "g": f"ffn_gate-{L}", "u": f"ffn_up-{L}", "a": act, "out": f"ffn_out-{L}"}


def _run_capture(session, ids, names: set[str], lora=None, vecs=None) -> dict:
    from .capture import Capture
    e = session.engine
    cap = session.__dict__.get("_nw_capture")
    if cap is None:
        cap = session.__dict__["_nw_capture"] = Capture()
    cap.set_wanted(lambda n: n in names)
    cap.skipped = {}
    ctx = e.probe_ctx(cap, need=len(ids))
    e.set_cvec(ctx, vecs or None)
    e.set_lora(ctx, lora or None)
    try:
        e.clear_memory(ctx)
        e.prefill(ctx, [list(ids)])
        got = cap.take()
    finally:
        e.set_cvec(ctx, None)
        e.set_lora(ctx, None)
    miss = [n for n in names if n not in got]
    if miss:
        raise GneError("ARCH_UNSUPPORTED", f"{', '.join(sorted(miss))} が捕捉できません")
    return {k: v.astype(np.float64) for k, v in got.items()}


def verify(session, recipe, *, texts: Sequence[str] | None = None, iv=None) -> dict:
    """nw の LoRA が式どおりに動くか。層ごとに、その層のユニットだけの LoRA を当てて短い文を読ませ、次を確かめる:
    上書きしたニューロンのゲート・入力が (新しい行)ᵀh になっているか、活性が silu(ゲート)·入力 か、ほかのニューロンの活性が
    変わっていないか、MLP の出力の変化が (新しい活性 × 新しい列) − (元の活性 × 元の列) の和になっているか。
    元の行・列は float に戻した値を引くので、量子化のままの掛け算との小さなずれ(入力の大きさの 1% ほど)は残る。
    iv を渡すと、その介入(レシピかユニット)を先に当てた状態で確かめる(当てない側・当てる側の両方に iv が掛かる)。"""
    from .ablate import write_lora_tensors
    from .keyed_build import intervention
    iv_vecs, iv_lora, iv_desc = intervention(session, iv)
    iv_lora = list(iv_lora or [])
    e, w = session.engine, session.weights
    rows = nw_plan(w, recipe, session._key_of, session.direction_set)
    if not rows:
        raise GneError("BAD_RECIPE", "nw のユニットがありません")
    texts = list(texts or VERIFY_TEXTS)
    seqs = [e.tokenize(t, add_special=True, parse_special=False) for t in texts]
    out_layers = []
    tmp_dir = K.key_dir(session) / "lora"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for L in sorted({r["layer"] for r in rows}):
        sub = [r for r in rows if r["layer"] == L]
        tens = nw_tensors(w, sub)
        path = tmp_dir / f"nw-verify-{cache_hash(tens)}.lora.gguf"
        write_lora_tensors(w, tens, path, name="nw verify")
        nm = _capture_names(session, seqs[0], L)
        Wd = w.down_matrix(L).astype(np.float64)
        js = [p[0] for r in sub for p in r["parts"]]
        stats = {"gate": [], "up": [], "act": [], "others": [], "out_rel": [], "out_norm": []}
        try:
            for ids in seqs:
                b = _run_capture(session, ids, set(nm.values()), iv_lora, iv_vecs)
                g = _run_capture(session, ids, set(nm.values()), [*iv_lora, (str(path), 1.0)], iv_vecs)
                h = g[nm["h"]][1:]
                pred = -(b[nm["a"]][1:, js] @ Wd[:, js].T)
                # 誤差は「予測の最大」か「その層のふつうのニューロンの値の大きさ」の大きいほうに対する比(鍵が反応しない文でも割れるように)
                floor_g = float(np.sqrt(np.mean(np.square(b[nm["g"]][1:]))))
                floor_u = float(np.sqrt(np.mean(np.square(b[nm["u"]][1:]))))
                for r in sub:
                    for (j, gate, up, down, _) in r["parts"]:
                        gj, uj = h @ gate, h @ up
                        stats["gate"].append(float(np.abs(g[nm["g"]][1:, j] - gj).max() / max(float(np.abs(gj).max()), floor_g)))
                        stats["up"].append(float(np.abs(g[nm["u"]][1:, j] - uj).max() / max(float(np.abs(uj).max()), floor_u)))
                        aj = silu(g[nm["g"]][1:, j]) * g[nm["u"]][1:, j]
                        stats["act"].append(float(np.abs(g[nm["a"]][1:, j] - aj).max() / max(float(np.abs(aj).max()), 1e-12)))
                        pred = pred + np.outer(g[nm["a"]][1:, j], down)
                oth = np.delete(g[nm["a"]][1:] - b[nm["a"]][1:], js, axis=1)
                stats["others"].append(float(np.abs(oth).max()))
                dout = g[nm["out"]][1:] - b[nm["out"]][1:]
                pn = np.linalg.norm(pred, axis=1)
                big = pn >= np.quantile(pn, 0.5)
                stats["out_rel"].append(float(np.median(np.linalg.norm(dout - pred, axis=1)[big] / np.maximum(pn[big], 1e-12))))
                stats["out_norm"].append(float(pn.max()))
        finally:
            e.drop_lora(path)
            path.unlink(missing_ok=True)
        row = {"layer": L, "neurons": js, "gate_err": max(stats["gate"]), "up_err": max(stats["up"]), "act_err": max(stats["act"]),
               "others_changed_max": max(stats["others"]), "out_rel_err_median": float(np.median(stats["out_rel"])),
               "out_change_max": max(stats["out_norm"])}
        # ゲート・入力の誤差には、量子化のままの掛け算と float に戻した行を引く計算のずれ(1〜2% ほど)が残る
        row["ok"] = bool(row["gate_err"] < 0.05 and row["up_err"] < 0.05 and row["act_err"] < 1e-3
                         and row["others_changed_max"] < 1e-4 and row["out_rel_err_median"] < 0.05)
        out_layers.append(row)
    return {"layers": out_layers, "ok": all(r["ok"] for r in out_layers), "n_texts": len(texts), "iv": iv_desc,
            "thresholds": {"gate_err": 0.05, "up_err": 0.05, "act_err": 1e-3, "others_changed_max": 1e-4, "out_rel_err_median": 0.05}}


def check(session, recipe, contrast, *, subset: str = "odd", max_tokens: int = 96, replies: str = "own", n_pos: int = 8,
          iv=None, positions: str | None = None) -> dict:
    """位置の種類ごとに、鍵の反応と実際に足した量を出す。対照集合の質問(狙う場面)と dont_touch の群(書き込みたくない場面)に
    ついて、返事の位置 0〜n_pos−1 とそれ以降で、ユニットごとの「鍵の反応(ゲートの鍵・入力の鍵)」と「足した量(α·d を 1 として)」の
    平均・最小・最大を返す。replies = own: レシピを当てて作った返事を、当てたまま読む(実際に起きること)。base: 無介入の返事を、
    レシピを当てて読む(書く前の状態での見積もり)。iv を渡すと、その介入(レシピかユニット)を先に当てた状態で測る
    (返事の生成にも読むときにも掛かる。base の返事は iv だけを当てて作る)。
    positions(before:<正規表現>): 返事の位置の番号ではなく、「正規表現に一致する文字の直前の位置(match)」と「それ以外の位置(other)」に
    分けて出す(文字の直前に書く鍵用。書くとその文字が出なくなるので、replies="base" と一緒に使う)。"""
    from .direction import _subset
    from .keyed_build import intervention
    iv_vecs, iv_lora, iv_desc = intervention(session, iv)
    iv_lora = list(iv_lora or [])
    from .evalx import _generate
    from .keyed_build import _as_spec
    from .reverse import _conv
    if replies not in ("own", "base"):
        raise GneError("BAD_ARGS", f"replies は own / base です({replies!r})")
    t0 = time.perf_counter()
    e, w = session.engine, session.weights
    rows = nw_plan(w, recipe, session._key_of, session.direction_set)
    if not rows:
        raise GneError("BAD_RECIPE", "nw のユニットがありません")
    lora = [*iv_lora, (str(session.nw_lora(recipe)), 1.0)]
    spec = _as_spec(contrast)
    cs = spec.contrast
    prompts = cs.prompts("B")
    groups = [("scene", [_conv(cs.system_b, prompts[i]) for i in _subset(len(prompts), subset)])]
    groups += [(f"dont_touch:{dt.name}", dt.convs(_subset(len(dt.prompts), subset))) for dt in spec.dont_touch]
    Ls = sorted({r["layer"] for r in rows})
    names = None
    buckets = [str(k) for k in range(n_pos)] + [f"{n_pos}+"]
    if positions is not None:
        from .keyed_build import parse_positions, select_positions
        if parse_positions(positions).kind != "before":
            raise GneError("BAD_ARGS", f"check の positions は before:<正規表現> です({positions!r})")
        buckets = ["match", "other"]
    res = {}
    for label, convs in groups:
        outs = _generate(session, [(c, int(max_tokens)) for c in convs], iv_vecs, lora if replies == "own" else (iv_lora or None),
                         "all", lambda k: None)
        acc = {id(r): {b: {"amount": [], "rho_gate": [], "rho_up": []} for b in buckets} for r in rows}
        loops = 0
        for c, o in zip(convs, outs, strict=True):
            p = e.tokenize_prompt(e.render_chat(c))
            rid = [t for t in o["token_ids"] if not e.is_eog(t)]
            from .behave import loop_like
            loops += bool(loop_like(o["token_ids"]))
            ids = p + rid
            if names is None:
                names = {L: _capture_names(session, ids, L) for L in Ls}
            got = _run_capture(session, ids, {names[L][k] for L in Ls for k in ("h", "a")}, lora, iv_vecs)
            for r in rows:
                L = r["layer"]
                h = got[names[L]["h"]][len(p) - 1:]
                a = got[names[L]["a"]][len(p) - 1:]
                amt = sum(g * a[:, j] for (j, _, _, _, g) in r["parts"])
                rg, ru = h @ r["kappa_gate"], h @ r["kappa_up"]
                hit = None
                if positions is not None:                       # 列の番号 → 返事の位置 k = 番号 − (P − 1)
                    hit = {int(x) - (len(p) - 1) for x in select_positions(e, rid, positions, n_prompt=len(p))}
                for k in range(amt.shape[0]):
                    b = ("match" if k in hit else "other") if hit is not None else (str(k) if k < n_pos else f"{n_pos}+")
                    acc[id(r)][b]["amount"].append(float(amt[k]))
                    acc[id(r)][b]["rho_gate"].append(float(rg[k]))
                    acc[id(r)][b]["rho_up"].append(float(ru[k]))
        st = lambda v: None if not v else {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}
        res[label] = {"n": len(convs), "loops": loops,
                      "units": [{"unit": spec_of(r["unit"]), "layer": r["layer"], "form": r["form"],
                                 "positions": {b: {k: st(v) for k, v in d.items()} for b, d in acc[id(r)].items()}} for r in rows]}
    return {"contrast": cs.name, "subset": subset, "replies": replies, "max_tokens": int(max_tokens), "iv": iv_desc, "groups": res,
            "positions": positions,
            "notes": ["amount は、その位置で実際に足した量(α·d を 1 として。負は逆向き)。rho_gate・rho_up は鍵の反応です。",
                      "scene は狙う場面(対照集合の質問)、dont_touch:<名前> は書き込みたくない場面です。"],
            "seconds": time.perf_counter() - t0}
