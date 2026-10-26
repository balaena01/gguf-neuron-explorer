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

def run_llama_completion(binary: str, engine: Engine, prompt: str, n_tokens: int, cvec: Path | None) -> str:
    """llama-completion で greedy 生成した文字列(プロンプトは表示しない)。"""
    cmd = [binary, "-m", str(engine.path), "-p", prompt, "-n", str(n_tokens), "-no-cnv", "--no-display-prompt",
           "--temp", "0", "--top-k", "1", "--seed", "0", "-c", str(engine.n_ctx), "-ngl", str(engine.n_gpu_layers),
           "--no-warmup", "-lv", "0"]
    if cvec is not None:
        cmd += ["--control-vector", str(cvec)]
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
