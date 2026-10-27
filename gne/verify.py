"""Phase 0 の検証テスト V1〜V8。設計書 §9。`gne verify` と pytest の両方から使う。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .capture import Capture, pick_act_name
from .engine import Engine, log_softmax
from .steer import cvec_buffer, export_cvec_gguf, load_cvec_like_common
from .weights import Weights

PROMPTS = [
    "Who are you?",
    "Describe the color orange in two sentences.",
    "あなたは誰ですか？自己紹介してください。",
]

# V2 の許容値(相対誤差の中央値, 最大)。誤差の主因は重みではなく、量子化 GEMM が活性を q8_1 に丸めること
# と半精度の累積なので、重みの型による差は小さい。重みが粗い型は念のため緩める(R7)。
V2_TOL = {
    "F32": (0.02, 0.10), "F16": (0.02, 0.10), "BF16": (0.02, 0.10), "Q8_0": (0.02, 0.10),
    "Q6_K": (0.03, 0.12), "Q5_K": (0.03, 0.12), "Q5_0": (0.03, 0.12), "Q5_1": (0.03, 0.12),
    "Q4_K": (0.04, 0.15), "Q4_0": (0.04, 0.15), "Q4_1": (0.04, 0.15), "IQ4_XS": (0.04, 0.15), "IQ4_NL": (0.04, 0.15),
}
V2_TOL_DEFAULT = (0.06, 0.20)


@dataclass
class VResult:
    id: str
    title: str
    status: str                     # "pass" | "fail" | "info" | "skip"
    detail: str
    data: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("pass", "info", "skip")


class Verifier:
    def __init__(self, engine: Engine, weights: Weights, *, layer: int | None = None, seed: int = 0):
        self.e, self.w = engine, weights
        n = engine.n_layer
        self.L = layer if layer is not None else n // 2
        if not 1 <= self.L <= n - 3:
            raise ValueError(f"--layer は 1..{n - 3} にしてください(L-1, L, L+1, L+2 を捕捉するため)")
        self.rng = np.random.default_rng(seed)
        self.prompt_ids = [engine.tokenize_prompt(engine.render_chat([{"role": "user", "content": p}]))
                           for p in PROMPTS]
        self._rho: float | None = None

    # --- helpers --------------------------------------------------------------------------------

    def capture(self, names: set[str], seqs: list[list[int]], vectors=None, layer_range=None) -> dict[str, np.ndarray]:
        cap = self._cap = getattr(self, "_cap", None) or Capture()
        cap.set_wanted(lambda n: n in names)
        ctx = self.e.probe_ctx(cap)
        self.e.clear_memory(ctx)
        self.e.set_cvec(ctx, vectors, layer_range=layer_range)
        self.e.prefill(ctx, seqs)
        out = cap.take()
        missing = names - out.keys()
        if missing:
            raise RuntimeError(f"捕捉できなかったテンソル: {sorted(missing)} (skipped={cap.skipped})")
        return out

    def rho(self) -> float:
        """層 L の残差ノルム中央値(各シーケンスの position ≥ 1)。V3/V4/V5 のベクトルの大きさに使う。"""
        if self._rho is None:
            norms = []
            for ids in self.prompt_ids:
                x = self.capture({f"l_out-{self.L}"}, [ids])[f"l_out-{self.L}"]
                norms.append(np.linalg.norm(x[1:], axis=1))
            self._rho = float(np.median(np.concatenate(norms)))
        return self._rho

    # --- V1 -------------------------------------------------------------------------------------

    def v1_shapes(self) -> VResult:
        m = self.w.meta
        bad = []
        for L in range(m.n_layer):
            name = f"blk.{L}.ffn_down.weight"
            if not self.w.has_tensor(name):
                bad.append(f"{name} なし")
                continue
            ne = [int(x) for x in self.w.tensor(name).shape]
            if ne != [m.n_ff_at(L), m.n_embd]:
                bad.append(f"{name} ne={ne}")
        checked = sorted({1, self.L, m.n_layer - 2})
        for L in checked:
            mat = self.w.down_matrix(L)
            if mat.shape != (m.n_embd, m.n_ff_at(L)) or not np.isfinite(mat).all():
                bad.append(f"L{L} dequant {mat.shape}")
        st = "fail" if bad else "pass"
        return VResult("V1", "形状と dequantize", st,
                       f"全 {m.n_layer} 層の ne=[n_ff, n_embd]、L{checked} を dequantize → (n_embd, n_ff)=({m.n_embd}, {m.n_ff}) {m.ffn_down_qtype}"
                       + (f" / 不一致: {bad}" if bad else ""), {"bad": bad})

    # --- V2 -------------------------------------------------------------------------------------

    def v2_ffn_linearity(self) -> VResult:
        L = self.L
        names = self.list_names(L)
        act = pick_act_name(names, L)
        if act is None:
            return VResult("V2", "FFN 線形性", "fail", f"ffn_down の入力テンソルが見つかりません: {sorted(names)}")
        caps = [self.capture({act, f"ffn_out-{L}"}, [ids]) for ids in self.prompt_ids]
        a = np.concatenate([c[act] for c in caps])
        out = np.concatenate([c[f"ffn_out-{L}"] for c in caps])
        W = self.w.down_matrix(L)
        s = (self.w.meta.ffn_down_scale or {}).get(L, 1.0)
        pred = s * (a @ W.T)
        rel = np.linalg.norm(pred - out, axis=1) / np.linalg.norm(out, axis=1)
        # 対照: ニューロン添字をずらすと一致しないこと(テストが添字の違いを見分けられる)
        rel_shift = np.linalg.norm(s * (np.roll(a, 1, axis=1) @ W.T) - out, axis=1) / np.linalg.norm(out, axis=1)
        tol_med, tol_max = V2_TOL.get(self.w.meta.ffn_down_qtype, V2_TOL_DEFAULT)
        med, mx = float(np.median(rel)), float(rel.max())
        ok = med < tol_med and mx < tol_max and float(np.median(rel_shift)) > 0.5
        return VResult("V2", "FFN 線形性(W_down の向き・添字、down の後に変換なし)", "pass" if ok else "fail",
                       f"L{L} `{act}` → W_down@a vs `ffn_out-{L}`: 相対誤差 中央値 {med:.4f} / 最大 {mx:.4f}"
                       f"(許容 {tol_med}/{tol_max}, {len(rel)} トークン)。添字を 1 ずらすと中央値 {np.median(rel_shift):.2f}",
                       {"act_name": act, "median": med, "max": mx, "shift_median": float(np.median(rel_shift)),
                        "n_tokens": int(len(rel))})

    # --- V3 -------------------------------------------------------------------------------------

    def v3_cvec_layer(self) -> VResult:
        L, n_embd = self.L, self.e.n_embd
        ids = self.prompt_ids[0]
        v = self.rng.standard_normal(n_embd).astype(np.float32)
        v *= 5.0 * self.rho() / np.linalg.norm(v)
        names = {f"l_out-{L - 1}", f"l_out-{L}", f"l_out-{L + 1}", f"l_out-{L + 2}"}
        base = self.capture(names, [ids], None)
        st = self.capture(names, [ids], {L: v})
        again = self.capture(names, [ids], None)          # 解除で元に戻るか
        d_prev = st[f"l_out-{L - 1}"] - base[f"l_out-{L - 1}"]
        d_L = st[f"l_out-{L}"] - base[f"l_out-{L}"]
        d_next = st[f"l_out-{L + 1}"] - base[f"l_out-{L + 1}"]
        err_L = float((np.linalg.norm(d_L - v, axis=1) / np.linalg.norm(v)).max())
        restore = float(max(np.abs(again[k] - base[k]).max() for k in names))
        # §5.3: 範囲を広げても前回の値が復活しない(常に全層ゼロ埋めバッファを渡しているので)
        v2 = self.rng.standard_normal(n_embd).astype(np.float32)
        v2 *= self.rho() / np.linalg.norm(v2)
        self.capture(names, [ids], {L: v})
        widened = self.capture(names, [ids], {L + 2: v2}, layer_range=(L, L + 2))
        stale = float(np.abs(widened[f"l_out-{L}"] - base[f"l_out-{L}"]).max())
        ok = (np.abs(d_prev).max() == 0 and err_L < 1e-3 and np.abs(d_next).max() > 0
              and restore == 0 and stale == 0)
        return VResult("V3", "cvec 層対応", "pass" if ok else "fail",
                       f"set_cvec({{{L}: v}}), ‖v‖=5ρ={np.linalg.norm(v):.1f}: Δl_out-{L - 1} 最大 {np.abs(d_prev).max():.3g}、"
                       f"Δl_out-{L} と v の相対誤差 最大 {err_L:.2e}({d_L.shape[0]} 位置)、Δl_out-{L + 1} 最大 {np.abs(d_next).max():.3g}、"
                       f"解除後の差 {restore:.3g}、範囲を L..L+2 に広げたときの l_out-{L} の差 {stale:.3g}",
                       {"d_prev_max": float(np.abs(d_prev).max()), "err_L": err_L,
                        "d_next_max": float(np.abs(d_next).max()), "restore": restore, "stale": stale})

    # --- V11(§6.16 gain)----------------------------------------------------------------------------------

    def v11_gain(self, ks=(0.0, 2.0, -1.0), bake: bool = True) -> VResult:
        """gain の等価性: 層 L のニューロン n を k 倍にしたとき、ffn_down の入力(活性)の n 番目が素の活性 × k になり、
        他のニューロンは変わらない。(a) LoRA アダプタを当てたとき、(b) W_up を書き換えた GGUF、の両方で見る。"""
        from .gain import bake_gain_gguf, export_gain_lora
        from .steer import Recipe, Unit
        L = self.L
        n = int(self.rng.integers(self.w.meta.n_ff_at(L)))
        ids = self.prompt_ids[0]
        names = self.e.list_names(ids, L)
        act = pick_act_name(names, L)
        if act is None:
            return VResult("V11", "gain の等価性", "skip", f"層 {L} の ffn_down 入力が捕捉できません")

        def grab(engine, lora=None):
            cap = Capture()
            cap.set_wanted(lambda x: x == act)
            ctx = engine.probe_ctx(cap)
            engine.clear_memory(ctx)
            engine.set_lora(ctx, lora)
            engine.prefill(ctx, [ids])
            engine.set_lora(ctx, None)
            return cap.take()[act]
        base = grab(self.e)
        scale = float(np.abs(base[:, n]).max())
        rows, ok = [], True
        with tempfile.TemporaryDirectory() as td:
            for k in ks:
                r = Recipe([Unit(L, n, 0.0, mode="gain", k=k)], self.w.meta.model_id, name="v11")
                variants = [("LoRA", None)]
                lpath = Path(td) / f"k{k:g}.lora.gguf"
                export_gain_lora(self.w, r, lpath)
                variants[0] = ("LoRA", grab(self.e, [(lpath, 1.0)]))
                if bake:
                    gpath = Path(td) / f"k{k:g}.gguf"
                    bake_gain_gguf(self.w, r, gpath)
                    e2 = Engine(gpath, n_gpu_layers=getattr(self.e, "n_gpu_layers", -1), n_ctx=self.e.n_ctx, n_seq_max=1)
                    try:
                        variants.append(("書き換え", grab(e2)))
                    finally:
                        e2.close()
                for kind, a in variants:
                    others = np.delete(np.arange(base.shape[1]), n)
                    d_other = float(np.abs(a[:, others] - base[:, others]).max())
                    if k == 0:
                        # 設計の目安は絶対値 < 1e-3 だが、LoRA 版は W x(量子化)と A x(dequantize 値)の差が残る。
                        # 活性の大きさはモデルで違うので、素の活性の最大値に対する比で見る(divergence.md)
                        err = float(np.abs(a[:, n]).max())
                        good = err < 0.03 * max(scale, 1e-6)
                    else:
                        rel = np.abs(a[:, n] - k * base[:, n]) / np.maximum(np.abs(k * base[:, n]), 1e-6)
                        err = float(np.median(rel))
                        good = err < 0.03
                    good = good and d_other <= 1e-3 * max(1.0, float(np.abs(base).max()))
                    ok &= good
                    rows.append({"k": k, "kind": kind, "err": err, "others_max_diff": d_other, "pass": bool(good)})
        detail = (f"L{L} N{n}(`{act}`、{len(ids)} トークン、|a| 最大 {scale:.3g}): "
                  + "; ".join(f"k={r['k']:g} {r['kind']}: " + (f"|a| 最大 {r['err']:.2e}(素の最大の {r['err'] / max(scale, 1e-6):.1%})" if r["k"] == 0 else f"相対誤差 中央値 {r['err']:.2e}")
                              + f"、他のニューロンの差 最大 {r['others_max_diff']:.2e}" for r in rows))
        return VResult("V11", "gain の等価性(LoRA / 書き換え)", "pass" if ok else "fail", detail,
                       {"layer": L, "neuron": n, "act": act, "rows": rows, "act_max": scale})

    # --- V13(§6.17 方向の射影消去、LoRA 版)--------------------------------------------------------------

    def _resid_all(self, engine: Engine, ids: list[int], lora=None) -> dict[int, np.ndarray]:
        """全層(最終層を除く)の l_out-L(位置 0 = attention sink を除く)。"""
        Ls = list(range(engine.n_layer - 1))                    # 最終層は行が間引かれ得るので除く
        names = {f"l_out-{L}": L for L in Ls}
        cap = Capture()
        cap.set_wanted(lambda n: n in names)
        ctx = engine.probe_ctx(cap)
        engine.clear_memory(ctx)
        engine.set_cvec(ctx, None)
        engine.set_lora(ctx, lora)
        try:
            engine.prefill(ctx, [ids])
        finally:
            engine.set_lora(ctx, None)
        return {names[k]: v.astype(np.float64)[1:] for k, v in cap.take().items()}

    def v13_ablation(self, r: np.ndarray | None = None, *, bake: bool = False) -> VResult:
        """方向 r を全層の残差ライターから消したとき、l_out-L の r 成分が「消えずに残るはずの分」と一致するか。
        - LoRA: 埋め込みも LoRA に入れられるモデル(出力層が別)では 0、tied のモデルでは埋め込みの r 成分のまま
          (LoRA で埋め込みを変えられないので、それ以上増えも減りもしない)。
        - bake(bake=True、F16): 埋め込みも射影するので 0。
        |r·l_out − 期待値| / ‖l_out‖ が量子化の誤差の水準(< 1e-2)であることを見る。消す前のずれも並べる。"""
        from .ablate import bake_ablation_gguf, embed_in_lora, export_ablation_lora, unit
        r = unit(self.rng.standard_normal(self.w.meta.n_embd) if r is None else r)
        ids = self.prompt_ids[0]
        er = self.w.embed_rows(ids)[1:].astype(np.float64) @ r    # 埋め込みの r 成分(inp_embd は捕捉できない)
        base = self._resid_all(self.e, ids)
        variants = []
        with tempfile.TemporaryDirectory() as td:
            lp = Path(td) / "v13.lora.gguf"
            rep = export_ablation_lora(self.w, r, lp)
            variants.append(("LoRA", self._resid_all(self.e, ids, [(lp, 1.0)]), 0.0 if rep["embed"] else er, rep["n_tensors"]))
            if bake:
                bp = Path(td) / "v13-bake.gguf"
                brep = bake_ablation_gguf(self.w, r, bp, requant="f16")
                e2 = Engine(bp, n_gpu_layers=getattr(self.e, "n_gpu_layers", -1), n_ctx=self.e.n_ctx, n_seq_max=self.e.n_seq_max)
                try:
                    variants.append(("bake", self._resid_all(e2, ids), 0.0, brep["n_modified"]))
                finally:
                    e2.close()
        out, ok = [], True
        for kind, res, expect, n_t in variants:
            rows = []
            for L, la in sorted(res.items()):
                lb = base[L]
                dev = np.abs(la @ r - expect) / np.linalg.norm(la, axis=1)
                bdev = np.abs(lb @ r - expect) / np.linalg.norm(lb, axis=1)
                perp_a, perp_b = la - np.outer(la @ r, r), lb - np.outer(lb @ r, r)
                ch = np.linalg.norm(perp_a - perp_b, axis=1) / np.linalg.norm(perp_b, axis=1)
                rows.append({"layer": L, "dev_max": float(dev.max()), "base_dev_med": float(np.median(bdev)),
                             "perp_change_med": float(np.median(ch))})
            mx = max(x["dev_max"] for x in rows)
            ok &= mx < 1e-2
            out.append({"kind": kind, "expect": "0" if np.isscalar(expect) else "e·r", "n_tensors": n_t, "max_dev": mx,
                        "base_med_dev": float(np.median([x["base_dev_med"] for x in rows])), "rows": rows})
        detail = "; ".join(f"{v['kind']}({v['n_tensors']} テンソル、r 成分の期待値 {v['expect']}): 最大 {v['max_dev']:.2e}"
                           f"(消す前は中央値 {v['base_med_dev']:.2e})、r に直交する成分の変化 層 0 で "
                           f"{v['rows'][0]['perp_change_med']:.3f}・最終で {v['rows'][-1]['perp_change_med']:.3f}" for v in out)
        lora = out[0]
        return VResult("V13", "方向の射影消去", "pass" if ok else "fail",
                       f"ランダムな方向({len(ids)} トークン、層 0〜{lora['rows'][-1]['layer']}、埋め込みを LoRA に"
                       f"{'入れる' if embed_in_lora(self.w) else '入れない(tied)'}): " + detail,
                       {"variants": out, "max_dev": max(v["max_dev"] for v in out), "base_med_dev": lora["base_med_dev"],
                        "n_tensors": lora["n_tensors"], "rows": lora["rows"]})

    def v14_lora_vs_bake(self, r: np.ndarray | None = None, n_tokens: int = 32) -> VResult:
        """同じ方向を消す LoRA と bake(F16)で、greedy の先頭 n_tokens が一致するか。埋め込みは両方そろえる
        (tied のモデルは LoRA に埋め込みを入れられないので bake も埋め込みを省く)。q8_0 の bake も参考に並べる。"""
        from .ablate import bake_ablation_gguf, embed_in_lora, export_ablation_lora, unit
        r = unit(self.rng.standard_normal(self.w.meta.n_embd) if r is None else r)
        r = (r * 1.0).astype(np.float32)
        embed = embed_in_lora(self.w)
        seqs = self.prompt_ids[:2]

        def run(engine, lora=None):
            ctx = engine.steer_ctx()
            engine.set_cvec(ctx, None)
            engine.set_lora(ctx, lora)
            try:
                return [engine.generate_greedy(ctx, [s], max_tokens=n_tokens)[0] for s in seqs]
            finally:
                engine.set_lora(ctx, None)
        base = [x.token_ids for x in run(self.e)]
        rows, ok = [], True
        with tempfile.TemporaryDirectory() as td:
            lp = Path(td) / "v14.lora.gguf"
            export_ablation_lora(self.w, r, lp, embed=embed)
            lora = run(self.e, [(lp, 1.0)])
            for requant in ("f16", "q8_0"):
                bp = Path(td) / f"v14-{requant}.gguf"
                bake_ablation_gguf(self.w, r, bp, requant=requant, embed=embed)
                e2 = Engine(bp, n_gpu_layers=getattr(self.e, "n_gpu_layers", -1), n_ctx=self.e.n_ctx, n_seq_max=self.e.n_seq_max)
                try:
                    baked = run(e2)
                finally:
                    e2.close()
                for i, (a, b) in enumerate(zip(lora, baked, strict=True)):
                    # 完全一致か、最初の分岐が僅差(相手の選択が自分の最良から 0.5 nats 以内)なら合格(V9 と同じ基準)
                    cmp = compare_greedy(a.token_ids, a.top_ids, a.top_lps, b.token_ids, b.top_ids, b.top_lps)
                    rows.append({"requant": requant, "prompt": i, "match": cmp["first_diff"] is None, "ok": bool(cmp["ok"]),
                                 "first_diff": cmp["first_diff"], "summary": cmp["summary"],
                                 "lora_differs_from_base": a.token_ids != base[i]})
                    if requant == "f16":
                        ok &= bool(cmp["ok"])
        detail = ", ".join(f"{x['requant']} プロンプト{x['prompt'] + 1}: " + ("一致" if x["match"] else x["summary"]
                                                                            + ("" if x["ok"] else "、僅差ではない"))
                           for x in rows)
        return VResult("V14", "射影消去の LoRA と bake の一致", "pass" if ok else "fail",
                       f"greedy {n_tokens} トークン、埋め込み{'あり' if embed else 'なし'}: {detail}"
                       f"(無介入との差: {'あり' if any(x['lora_differs_from_base'] for x in rows) else 'なし'})",
                       {"rows": rows, "embed": embed})

    # --- V12(§6.15 位置別ステアリング)----------------------------------------------------------------

    def v12_when_gen(self) -> VResult:
        """when=gen: prefill は素のまま(l_out-L が素の値と完全一致)、生成 1 ステップ目から v が加わる。
        さらに gne の generate_when(gen) の最初のトークンが無介入と同じで、全体として無介入とも all とも違うことを見る。"""
        from .when import generate_when
        L, n_embd = self.L, self.e.n_embd
        ids = self.prompt_ids[0]
        v = self.rng.standard_normal(n_embd).astype(np.float32)
        v *= 5.0 * self.rho() / np.linalg.norm(v)
        name = f"l_out-{L}"
        cap = Capture()
        cap.set_wanted(lambda n: n == name)
        ctx = self.e.probe_ctx(cap)

        def run(gen_vec):
            self.e.clear_memory(ctx)
            self.e.set_cvec(ctx, None)
            pre = self.e.prefill(ctx, [ids])
            a = cap.take()[name]
            tok = int(pre.logits[0].argmax())
            self.e.set_cvec(ctx, gen_vec)
            self.e._decode(ctx, [(tok, len(ids), 0, True)])
            b = cap.take()[name]
            self.e.set_cvec(ctx, None)
            return a, b
        a0, b0 = run(None)
        a1, b1 = run({L: v})
        prefill_same = float(np.abs(a1 - a0).max())
        err_gen = float(np.linalg.norm((b1 - b0)[-1] - v) / np.linalg.norm(v))
        # 実際の経路: generate_when の gen / all / 無介入
        msgs = [{"role": "user", "content": PROMPTS[0]}]
        sctx = self.e.steer_ctx()
        g_base, _ = generate_when(self.e, sctx, msgs, None, "all", max_tokens=24)
        g_gen, info = generate_when(self.e, sctx, msgs, {L: v}, "gen", max_tokens=24)
        g_all, _ = generate_when(self.e, sctx, msgs, {L: v}, "all", max_tokens=24)
        first_same = bool(g_gen.token_ids[:1] == g_base.token_ids[:1])
        ok = prefill_same == 0 and err_gen < 1e-3 and first_same and info["set_cvec"] == 2
        return VResult("V12", "位置別ステアリング(when=gen)", "pass" if ok else "fail",
                       f"l_out-{L}: prefill の差 最大 {prefill_same:.3g}(0 であること)、生成 1 ステップ目の差と v の相対誤差 {err_gen:.2e}。"
                       f"generate_when(gen): 最初のトークンが無介入と{'同じ' if first_same else '違う'}、set_cvec {info['set_cvec']} 回、"
                       f"無介入と{'同じ' if g_gen.token_ids == g_base.token_ids else '違う'}、all と{'同じ' if g_gen.token_ids == g_all.token_ids else '違う'}",
                       {"prefill_diff_max": prefill_same, "err_gen": err_gen, "first_token_same": first_same,
                        "set_cvec": info["set_cvec"], "gen_eq_base": g_gen.token_ids == g_base.token_ids,
                        "gen_eq_all": g_gen.token_ids == g_all.token_ids})

    # --- V4 -------------------------------------------------------------------------------------

    def v4_direction_file(self, llama_completion: str | None = None, n_tokens: int = 32) -> VResult:
        L, n_layer, n_embd = self.L, self.e.n_layer, self.e.n_embd
        n = int(self.rng.integers(self.w.meta.n_ff_at(L)))
        v = (0.8 * self.rho() * self.w.unit_direction(L, n)).astype(np.float32)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "v4.cvec.gguf"
            export_cvec_gguf({L: v}, path, model_hint=self.w.meta.arch, n_layer=n_layer)
            n_embd_f, data = load_cvec_like_common([(path, 1.0)])
            buf, _, _ = cvec_buffer(n_layer, n_embd, {L: v})
            same_buf = n_embd_f == n_embd and data.size == buf.size and np.array_equal(data, buf)
            # ローダの規約で読み戻したバッファで V3 と同じ捕捉
            names = {f"l_out-{L - 1}", f"l_out-{L}"}
            ids = self.prompt_ids[0]
            a = self.capture(names, [ids], {L: v})
            back = {i + 1: data[i * n_embd:(i + 1) * n_embd] for i in range(n_layer - 1)}
            b = self.capture(names, [ids], back, layer_range=(1, n_layer - 1))
            same_cap = all(np.array_equal(a[k], b[k]) for k in names)
            detail = (f"direction.{L} に 0.8ρ·u_{n} を書き出し → common の規則で読み戻したバッファが一致: {same_buf}、"
                      f"それを全層範囲で set_cvec した捕捉が一致: {same_cap}")
            data_out = {"neuron": n, "same_buf": bool(same_buf), "same_capture": bool(same_cap)}
            ok = same_buf and same_cap
            if llama_completion:
                prompt = self.e.render_chat([{"role": "user", "content": PROMPTS[0]}])
                ctx = self.e.steer_ctx()
                gen = {}
                for tag, vec in (("base", None), ("steer", {L: v})):
                    self.e.set_cvec(ctx, vec)
                    r = self.e.generate_greedy(ctx, [self.prompt_ids[0]], max_tokens=n_tokens)[0]
                    ours = self.e.detokenize(r.token_ids[:-1] if r.finish == "eog" else r.token_ids)
                    theirs = run_llama_completion(llama_completion, self.e, prompt, n_tokens,
                                                  path if tag == "steer" else None)
                    gen[tag] = {"ours": ours, "llama": theirs, "match": ours == theirs}
                self.e.set_cvec(ctx, None)
                differs = gen["steer"]["ours"] != gen["base"]["ours"]
                ok = ok and gen["base"]["match"] and gen["steer"]["match"] and differs
                detail += (f"。llama-completion --control-vector の greedy {n_tokens} トークン: 無介入 一致 {gen['base']['match']}、"
                           f"介入 一致 {gen['steer']['match']}(介入で出力が変わる: {differs})")
                data_out["generation"] = gen
            else:
                detail += "。llama-completion が無いので生成の比較は省略"
        return VResult("V4", "direction.N 対応(GGUF 書き出し → 本家ローダ)", "pass" if ok else "fail", detail, data_out)

    # --- V5 -------------------------------------------------------------------------------------

    def v5_neuron_push(self, hf_model: str | None, hf_python: str | None, n_tokens: int = 32) -> VResult:
        if not hf_model or not hf_python:
            return VResult("V5", "ニューロン押しの等価性(HF 実装)", "skip", "--hf-model と --hf-python が無いので省略(任意)")
        L = self.L
        n = int(self.rng.integers(self.w.meta.n_ff_at(L)))
        col = self.w.down_column(L, n)
        add = 0.8 * self.rho() / float(np.linalg.norm(col))     # rel dose 0.8 相当の活性への加算量
        ids = self.prompt_ids[0]
        eog = [t for t in range(self.e.n_vocab) if self.e.is_eog(t)]
        req = {"prompt_ids": ids, "layer": L, "neuron": n, "add": add, "max_tokens": n_tokens,
               "n_vocab": self.e.n_vocab, "eog_ids": eog, "dump_down": True}
        ref, hf_down = run_hf_reference(hf_python, hf_model, req)
        ctx = self.e.steer_ctx()
        ours = {}
        for tag, vec in (("base", None), ("steer", {L: add * col})):
            self.e.set_cvec(ctx, vec)
            self.e.clear_memory(ctx)
            pre = self.e.prefill(ctx, [ids])
            r = self.e.generate_greedy(ctx, [ids], max_tokens=n_tokens, prefilled=pre)[0]
            ours[tag] = {"ids": r.token_ids, "next_lp": log_softmax(pre.logits[0]), "top_ids": r.top_ids, "top_lps": r.top_lps}
        self.e.set_cvec(ctx, None)
        res, ok = {}, True
        for tag in ("base", "steer"):
            res[tag] = compare_greedy(ours[tag]["ids"], ours[tag]["top_ids"], ours[tag]["top_lps"],
                                      ref[tag]["ids"], ref[tag]["top_ids"], ref[tag]["top_lps"])
            ok = ok and res[tag]["ok"]
        # 介入による次トークン分布の変化 Δ(最終プロンプト位置)が HF と一致するか
        d_ours = ours["steer"]["next_lp"] - ours["base"]["next_lp"]
        d_hf = np.asarray(ref["steer"]["next_lp"], np.float32) - np.asarray(ref["base"]["next_lp"], np.float32)
        top = np.argsort(-np.maximum(ours["base"]["next_lp"], ours["steer"]["next_lp"]))[:200]
        corr = float(np.corrcoef(d_ours[top], d_hf[top])[0, 1])
        ok = ok and corr > 0.9
        detail = (f"L{L} N{n} add={add:.2f}: 無介入 {res['base']['summary']} / 介入 {res['steer']['summary']}。"
                  f"介入による次トークン log p の変化の相関(上位 200 語) {corr:.3f}")
        r4 = {}
        if hf_down is not None:
            r4 = neuron_order_check(self.w.down_matrix(L), hf_down, self.rng)
            detail += (f"。R4: HF down_proj と GGUF の列の相関 中央値 {r4['diag_median']:.4f}、"
                       f"抜き取り {r4['n_sample']} 列で最も相関する HF 列が同じ番号: {r4['argmax_match']:.3f}")
            ok = ok and r4["argmax_match"] == 1.0
        return VResult("V5", "ニューロン押しの等価性(HF 実装)", "pass" if ok else "fail", detail,
                       {"neuron": n, "add": add, "corr_delta": corr, "r4": r4, **{k: v for k, v in res.items()}})

    # --- V6 -------------------------------------------------------------------------------------

    def v6_attention_sink(self) -> VResult:
        layers = sorted({1, self.L // 2, self.L, self.e.n_layer - 2})
        ratios = {}
        for L in layers:
            r = []
            for ids in self.prompt_ids:
                x = self.capture({f"l_out-{L}"}, [ids])[f"l_out-{L}"]
                nrm = np.linalg.norm(x, axis=1)
                r.append(nrm[0] / np.median(nrm[1:]))
            ratios[L] = float(np.median(r))
        ok = all(v >= 3 for v in ratios.values())
        return VResult("V6", "position 0 の残差ノルム(attention sink)", "info",
                       "‖l_out‖[pos0] / 中央値[pos≥1]: " + ", ".join(f"L{L} {v:.1f}×" for L, v in ratios.items())
                       + ("(全層 ≥3×)" if ok else "(3× 未満の層あり: 警告)"), {"ratios": ratios, "all_ge3": ok})

    # --- V8 -------------------------------------------------------------------------------------

    def v8_tokenize(self) -> VResult:
        convs = [
            [{"role": "user", "content": PROMPTS[0]}],
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": PROMPTS[2]}],
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! 何かお手伝いできますか？"},
             {"role": "user", "content": "Tell me a joke."}],
        ]
        bad = []
        from llama_cpp import llama_cpp as C
        add_bos = bool(C.llama_vocab_get_add_bos(self.e.vocab))
        bos = C.llama_vocab_bos(self.e.vocab)
        for conv in convs:
            text = self.e.render_chat(conv)
            ids = self.e.tokenize_prompt(text)
            body = ids[1:] if add_bos and not text.startswith(self.e.token_text(bos)) else ids
            back = self.e.detokenize(body, special=True)
            if back != text:
                bad.append(f"往復不一致: {text[:40]!r} → {back[:40]!r}")
            if add_bos and ids[:2] == [bos, bos]:
                bad.append("BOS が二重")
            if add_bos and ids[0] != bos:
                bad.append("BOS がない")
        return VResult("V8", "トークナイズ往復と BOS", "fail" if bad else "pass",
                       f"{len(convs)} 会話: テンプレート描画 → tokenize → detokenize が一致、add_bos={add_bos}"
                       + (f" / {bad}" if bad else ""), {"bad": bad, "add_bos": add_bos})

    # --- names ----------------------------------------------------------------------------------

    def list_names(self, L: int | None = None) -> set[str]:
        return self.e.list_names(self.prompt_ids[0], L)


# --- 外部プロセス -------------------------------------------------------------------------------

def run_llama_completion(binary: str, engine: Engine, prompt: str, n_tokens: int, cvec: Path | None,
                         lora: Path | None = None) -> str:
    """llama-completion で greedy 生成した文字列(プロンプトは表示しない)。lora は gain の LoRA アダプタ(§6.16)。"""
    cmd = [binary, "-m", str(engine.path), "-p", prompt, "-n", str(n_tokens), "-no-cnv", "--no-display-prompt",
           "--temp", "0", "--top-k", "1", "--seed", "0", "-c", str(engine.n_ctx), "-ngl", str(engine.n_gpu_layers),
           "--no-warmup", "-lv", "0"]
    if cvec is not None:
        cmd += ["--control-vector", str(cvec)]
    if lora is not None:
        cmd += ["--lora", str(lora)]
    env = dict(os.environ)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"llama-completion が失敗しました: {p.stderr[-2000:]}")
    # completion.cpp は生成の後に必ず "\n\n" を、EOG で止まったときはその前に " [end of text]\n" を出す
    out = p.stdout.removesuffix("\n\n")
    return out.removesuffix(" [end of text]\n")


def run_hf_reference(python: str, model_dir: str, req: dict) -> dict:
    script = Path(__file__).resolve().parent.parent / "tools" / "hf_reference.py"
    with tempfile.TemporaryDirectory() as td:
        rq, rs = Path(td) / "req.json", Path(td) / "ref.json"
        rq.write_text(json.dumps(req))
        p = subprocess.run([python, str(script), "--model", model_dir, "--request", str(rq), "--out", str(rs)],
                           capture_output=True, text=True, timeout=1200)
        if p.returncode != 0:
            raise RuntimeError(f"hf_reference.py が失敗しました: {p.stderr[-3000:]}")
        down = Path(td) / f"down_L{req['layer']}.npy"
        return json.loads(rs.read_text()), (np.load(down) if down.exists() else None)


def neuron_order_check(gguf_down: np.ndarray, hf_down: np.ndarray, rng, n_sample: int = 256) -> dict:
    """R4: GGUF の W_down[:, n] と HF の down_proj.weight[:, n] が同じニューロンか(列ごとの相関)。"""
    if gguf_down.shape != hf_down.shape:
        return {"diag_median": float("nan"), "argmax_match": 0.0, "n_sample": 0,
                "error": f"shape {gguf_down.shape} vs {hf_down.shape}"}
    g = gguf_down - gguf_down.mean(axis=0)
    h = hf_down - hf_down.mean(axis=0)
    g /= np.linalg.norm(g, axis=0)
    h /= np.linalg.norm(h, axis=0)
    diag = (g * h).sum(axis=0)
    cols = rng.choice(g.shape[1], size=min(n_sample, g.shape[1]), replace=False)
    best = np.argmax(g[:, cols].T @ h, axis=1)
    return {"diag_median": float(np.median(diag)), "diag_min": float(diag.min()),
            "argmax_match": float(np.mean(best == cols)), "n_sample": int(len(cols))}


def compare_greedy(a_ids, a_top, a_lps, b_ids, b_top, b_lps, near: float = 0.5) -> dict:
    """greedy 列の比較。一致しない場合、最初の分岐で「相手が選んだトークンが自分の最良から near nats 以内」
    が双方で成り立てば(僅差の分岐、数値誤差で説明できる)ok。"""
    n = min(len(a_ids), len(b_ids))
    first = next((i for i in range(n) if a_ids[i] != b_ids[i]), None)
    if first is None:
        same_len = len(a_ids) == len(b_ids)
        return {"ok": True, "first_diff": None, "summary": f"先頭 {n} トークン一致" + ("" if same_len else "(長さ違い)")}

    def gap(top, lps, other_tok):
        top = list(top[first])
        return lps[first][0] - lps[first][top.index(other_tok)] if other_tok in top else float("inf")

    gap_a, gap_b = gap(a_top, a_lps, b_ids[first]), gap(b_top, b_lps, a_ids[first])
    ok = gap_a < near and gap_b < near
    return {"ok": ok, "first_diff": first, "gap_ours": gap_a, "gap_ref": gap_b,
            "summary": f"{first} トークン目で分岐(相手の選択との差 ours {gap_a:.2f} / ref {gap_b:.2f} nats)"}


def find_llama_completion() -> str | None:
    here = Path(__file__).resolve().parent.parent / "build" / "llama.cpp" / "build" / "bin" / "llama-completion"
    if here.exists():
        return str(here)
    return shutil.which("llama-completion")
