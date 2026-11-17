"""teacher forcing の目安でヘッドを確かめる(gne heads tf、docs/head-component-steering.md「基準層より後ろのヘッド」)。

gne の因果の確かめ(gne heads pipeline)は「ヘッドを止めたときの、基準層での射影の値の変化」で測るので、基準層より後ろの
ヘッドは仕組みのうえで測れない。ここでは、無介入の返事を読ませ(teacher forcing)、ヘッドの方向成分を弱めたとき
(proj、強さ s)に、
- (a) 対象の文のトークンの対数確率がどれだけ下がるか(それ以外のトークンの変化を引いたもの。大きいほど対象の文に効く)
- (b) それ以外のトークンの対数確率の変化の絶対値の平均(小さいほど、ほかへの影響が少ない)
- 返事の終わりのトークンの対数確率の変化
を、層ごと(その層の全ヘッドを同時に)→ ヘッドごとに測る。返事は生成しない(読み直すだけ)。

対象の文の決め方: 対照集合の片側(既定 a = system_a / user_prompts の側)の質問に無介入で返事を作り、**書き終えた返事だけ**を
使う(返事の終わりに出る振る舞いは、長さの上限で切れた返事では「出なかった」が確かでないため)。返事の中で target(正規表現)に
一致した文字を含む文(span sentence。句点・改行で区切る)か、一致した文字そのもの(span match)のトークンを「対象の文」とする。
それ以外のトークンは返事ごとに others 個まで(種を決めて)選ぶ。
ヘッドごとに測る層: layers_heads を渡せばその層。省くと、層ごとの (a) が正で、かつ層の (a) の中央値以上の層。
"""
from __future__ import annotations

import json
import re
import time

import numpy as np

from .steer import GneError

SENT_RE = re.compile(r"[^。．.!?！？\n]*(?:[。．.!?！？]+|\n|$)")


def target_spans(text: str, target: str, span: str = "sentence") -> list[tuple[int, int]]:
    """返事の文字列で、対象の文の文字の範囲 [a, b) の並び。"""
    rx = re.compile(target)
    hits = [(m.start(), m.end()) for m in rx.finditer(text)]
    if span == "match" or not hits:
        return hits
    out = []
    for m in SENT_RE.finditer(text):
        a, b = m.start(), m.end()
        if b > a and any(a <= h0 < b or a < h1 <= b for h0, h1 in hits):
            out.append((a, b))
    return out


def _token_offsets(e, ids: list[int]) -> list[int]:
    out, cum = [], 0
    for t in ids:
        out.append(cum)
        cum += len(e.token_bytes(t))
    return out + [cum]


def build_rows(session, contrast, *, target: str, side: str = "a", subset: str = "all", span: str = "sentence",
               max_tokens: int = 256, others: int = 64, seed: int = 0) -> dict:
    """対照集合の片側の質問に無介入で返事を作り、書き終えた返事ごとに (プロンプト, 返事 + 終わり, 対象のトークンの番号,
    ほかのトークンの番号) を返す。"""
    from .direction import _subset
    from .evalx import _generate
    from .keyed_build import _as_spec
    from .reverse import _conv
    if side not in ("a", "b"):
        raise GneError("BAD_ARGS", f"side は a / b です({side!r})")
    if span not in ("sentence", "match"):
        raise GneError("BAD_ARGS", f"span は sentence / match です({span!r})")
    try:
        re.compile(target)
    except re.error as ex:
        raise GneError("BAD_ARGS", f"target の正規表現を読めません: {ex}") from None
    e = session.engine
    cs = _as_spec(contrast).contrast
    prompts = cs.prompts("A" if side == "a" else "B")
    system = cs.system_a if side == "a" else cs.system_b
    convs = [_conv(system, prompts[i]) for i in _subset(len(prompts), subset)]
    outs = _generate(session, [(c, int(max_tokens)) for c in convs], None, None, "all", lambda k: None)
    rng = np.random.default_rng(seed)
    rows, n_unfinished = [], 0
    for c, o in zip(convs, outs):
        if o["finish"] != "eog":
            n_unfinished += 1
            continue
        rid = [int(t) for t in o["token_ids"] if not e.is_eog(t)]
        eog = next((int(t) for t in reversed(o["token_ids"]) if e.is_eog(t)), None)     # 返事を終えたトークン
        if eog is None:
            n_unfinished += 1
            continue
        text = e.detokenize(rid)
        off = _token_offsets(e, rid)
        byte_spans = [(len(text[:a].encode("utf-8")), len(text[:b].encode("utf-8"))) for a, b in target_spans(text, target, span)]
        tgt = sorted({j for j in range(len(rid)) for a, b in byte_spans if off[j] < b and off[j + 1] > a})
        rest = [j for j in range(len(rid)) if j not in set(tgt)]
        if len(rest) < 8:
            continue
        oth = sorted(rng.choice(rest, min(int(others), len(rest)), replace=False).tolist())
        rows.append({"prompt": e.tokenize_prompt(e.render_chat(c)), "reply": rid + [int(eog)], "target": tgt, "others": oth})
    return {"rows": rows, "n_prompts": len(convs), "n_unfinished": n_unfinished, "n_with_target": sum(bool(r["target"]) for r in rows)}


def _logprobs(session, rows: list[dict], lora=None, batch: int = 4, lora_when: str = "all") -> list[np.ndarray]:
    """行ごとに、選んだ位置(対象・ほか・終わり)の、返事のトークンの対数確率。lora_when で LoRA を当てる位置を限る
    (all = 全位置、prompt = プロンプトのトークンだけ、reply = 返事のトークンだけ。生成の lora_when と同じ決め方で、
    返事の最初のトークンはプロンプトの最後の位置の読み取りから決まる)。"""
    from .when import check_lora_when
    check_lora_when(lora_when)
    e = session.engine
    ctx = e.steer_ctx()
    out = []
    state_a = (None if lora_when == "reply" else lora) if lora else None       # プロンプトを読む間
    state_b = (None if lora_when == "prompt" else lora) if lora else None      # 返事のトークンを読む間
    try:
        for b0 in range(0, len(rows), batch):
            part = rows[b0:b0 + batch]
            e.clear_memory(ctx)
            e.set_cvec(ctx, None)
            phases, want = ([], []), []
            for k, r in enumerate(part):
                p, rep = r["prompt"], r["reply"]
                sel = set(r["target"]) | set(r["others"]) | {len(rep) - 1}
                for pos, t in enumerate(p + rep[:-1]):
                    j = pos - (len(p) - 1)                       # この位置の出力が予測する返事のトークンの番号
                    phases[0 if pos < len(p) else 1].append((t, pos, k, 0 <= j < len(rep) and j in sel, (k, j, rep[j] if 0 <= j < len(rep) else -1)))
            res = [dict() for _ in part]

            def run(cur):
                e._decode(ctx, [it[:4] for it in cur])
                for jj, it in enumerate(cur):
                    if it[3]:
                        lo = e._logits(ctx, jj).astype(np.float64)
                        k, j, tok = it[4]
                        m = lo.max()
                        res[k][j] = float(lo[tok] - (m + np.log(np.exp(lo - m).sum())))
            for ph, st in zip(phases, (state_a, state_b), strict=True):
                e.set_lora(ctx, st or None)
                cur, n_out = [], 0
                for it in ph:
                    if cur and (len(cur) >= ctx.n_batch or (it[3] and n_out >= 128)):
                        run(cur)
                        cur, n_out = [], 0
                    cur.append(it)
                    n_out += int(it[3])
                if cur:
                    run(cur)
            for k, r in enumerate(part):
                order = r["target"] + r["others"] + [len(r["reply"]) - 1]
                out.append(np.array([res[k][j] for j in order]))
    finally:
        e.set_lora(ctx, None)
    return out


def metrics(lp: list[np.ndarray], base: list[np.ndarray], rows: list[dict]) -> dict:
    dd, bb, ee = [], [], []
    for x, b0, r in zip(lp, base, rows):
        ch = x - b0
        nt = len(r["target"])
        oth = ch[nt:-1]
        bb.append(float(np.abs(oth).mean()))
        ee.append(float(ch[-1]))
        if nt:
            dd.append(float(ch[:nt].mean() - oth.mean()))
    return {"a": -float(np.mean(dd)) if dd else None, "b": float(np.mean(bb)), "eos": float(np.mean(ee))}


def screen(session, contrast, *, direction: str, target: str, layers=None, layers_heads=None, s: float = 10.0,
           direction_layer: str = "own", side: str = "a", subset: str = "all", span: str = "sentence", max_tokens: int = 256,
           others: int = 64, seed: int = 0, name: str | None = None, progress=None, lora_when: str = "all") -> dict:
    """層ごと → ヘッドごとの (a)・(b)・終わりのトークンの変化。layers は層の並びか "1-29"(省くと 1〜最終層の 1 つ前)。"""
    from .jobs import _layers as parse_layers
    t0 = time.perf_counter()
    tick = progress or (lambda *_: None)
    w = session.weights
    n_layer = w.meta.n_layer
    Ls = sorted({int(L) for L in (parse_layers(layers) if layers is not None else range(1, n_layer - 1))})
    Ls = [L for L in Ls if w.arch.layers[L].head_writer is not None and w.arch.layers[L].n_head]
    if not Ls:
        raise GneError("BAD_ARGS", "ヘッドのある層がありません")
    data = build_rows(session, contrast, target=target, side=side, subset=subset, span=span, max_tokens=max_tokens,
                      others=others, seed=seed)
    rows = data["rows"]
    if not data["n_with_target"]:
        raise GneError("NO_TARGET", f"書き終えた返事 {len(rows)} 本に、target {target!r} に一致する文がありません",
                       "target の正規表現か、対照集合の側(side)を確かめてください")
    base = _logprobs(session, rows)

    def measure(units, tag):
        r = session.recipe(units=units, direction_layer=direction_layer, name=f"tf_{tag}")
        return metrics(_logprobs(session, rows, session.lora_for(r), lora_when=lora_when), base, rows)
    by_layer = []
    for i, L in enumerate(Ls):
        nh = w.arch.layers[L].n_head
        m = measure([f"{L}:H{h}:s{s:g}@{direction}" for h in range(nh)], f"L{L}")
        by_layer.append({"layer": L, "n_heads": nh, **m})
        tick("layers", i + 1, len(Ls))
    if layers_heads is None:
        a_vals = [x["a"] for x in by_layer if x["a"] is not None]
        med = float(np.median(a_vals)) if a_vals else 0.0
        HL = [x["layer"] for x in by_layer if x["a"] is not None and x["a"] > 0 and x["a"] >= med]
        rule = f"(a) が正で、層の (a) の中央値 {med:.4g} 以上の層"
    else:
        HL = sorted({int(L) for L in parse_layers(layers_heads)})
        rule = "指定"
    by_head = []
    todo = [(L, h) for L in HL for h in range(w.arch.layers[L].n_head)]
    for i, (L, h) in enumerate(todo):
        by_head.append({"layer": L, "head": h, **measure([f"{L}:H{h}:s{s:g}@{direction}"], f"L{L}H{h}")})
        tick("heads", i + 1, len(todo))
    ranked = sorted([x for x in by_head if x["a"] is not None], key=lambda x: -x["a"])
    out = {"direction": direction, "direction_layer": direction_layer, "s": s, "lora_when": lora_when, "target": target, "span": span, "side": side,
           "n_prompts": data["n_prompts"], "n_finished": len(rows), "n_unfinished": data["n_unfinished"],
           "n_with_target": data["n_with_target"], "layers": by_layer, "heads_layers": HL, "heads_rule": rule, "heads": by_head,
           "top": ranked[:20], "seconds": round(time.perf_counter() - t0, 1)}
    if name:
        if not re.fullmatch(r"[\w.\-]+", name):
            raise GneError("BAD_ARGS", f"名前 {name!r} は英数字・_・-・. だけにしてください")
        p = session.store.heads_dir(session.meta.model_id) / f"{name}.tf.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        out["path"] = str(p)
    return out


def summary_lines(d: dict) -> list[str]:
    out = [f"書き終えた返事 {d['n_finished']} 本(上限で切れて除いた {d['n_unfinished']} 本)、対象の文を含む {d['n_with_target']} 本。"
           f"方向 {d['direction']}({d['direction_layer']})、強さ s = {d['s']:g}"]
    out.append("層ごと(全ヘッドを同時に): " + "  ".join(
        f"L{x['layer']} a={x['a']:.4f} b={x['b']:.4f}" if x["a"] is not None else f"L{x['layer']} -" for x in d["layers"]))
    out.append(f"ヘッドごとに測った層: {d['heads_layers']}({d['heads_rule']})")
    for x in d["top"][:10]:
        out.append(f"  L{x['layer']} H{x['head']}: (a) {x['a']:.4f}、(b) {x['b']:.4f}、(a)/(b) {x['a'] / max(x['b'], 1e-9):.3f}、"
                   f"終わりのトークン {x['eos']:+.4f}")
    if d.get("path"):
        out.append(f"結果: {d['path']}")
    return out
