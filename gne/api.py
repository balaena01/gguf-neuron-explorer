"""Python API。設計書 §6.13 (c)。CLI はこの薄い皮。

    from gne.api import open_model
    with open_model("models/Qwen3.5-4B-Q8_0.gguf") as s:
        out = s.steer(["14:8949:+1.0", "13:7956:-1.0"], prompt="Who are you?")

戻り値は --json と同じ形の dict(エンベロープ抜き)。エンベロープは envelope() で作る。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .engine import Engine
from .schemas import SCHEMA_VERSION
from .stats import (ActStats, ResidStats, StatsView, load_corpus, neuron_activation_stats,
                    residual_norms)
from .steer import (GneError, Recipe, bake_bias_gguf, bias_support, export_vectors, import_cvec_gguf,
                    load_cvec_like_common, parse_unit, recipe_vectors, unit_details)
from .store import Store
from .weights import Weights

Progress = Callable[[int, int, str], None]


def open_model(path: str | Path, **kw) -> Session:
    return Session(path, **kw)


class Session:
    def __init__(self, path: str | Path, *, root: str | Path | None = None, n_gpu_layers: int = -1,
                 n_ctx: int = 4096, n_seq_max: int = 8, seed: int = 0, verbose: bool = False,
                 progress: Progress | None = None):
        self.path = Path(path)
        if not self.path.exists():
            raise GneError("MODEL_NOT_FOUND", f"{self.path} がありません")
        self.weights = Weights(self.path)
        self.meta = self.weights.meta
        self.store = Store(root)
        self.store.save_meta(self.meta)
        self.seed = seed
        self.progress = progress
        self.warnings: list[str] = []
        self._engine_kw = dict(n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, n_seq_max=n_seq_max, verbose=verbose)
        self._engine: Engine | None = None
        self._resid: ResidStats | None = None

    # --- lifecycle ------------------------------------------------------------------------------

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = Engine(self.path, **self._engine_kw)
            tpl = template_override(self.meta.model_id, self.meta.arch)
            if tpl is not None:
                self._engine.chat_template = tpl[1]
                self.warn(f"チャットテンプレートを {tpl[0]} で上書きしました")
        return self._engine

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def envelope(self, command: str) -> dict:
        return {"schema_version": SCHEMA_VERSION, "command": command, "model_id": self.meta.model_id,
                "llama_cpp_version": Engine.llama_cpp_version(), "seed": self.seed,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "warnings": list(self.warnings)}

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    # --- info -----------------------------------------------------------------------------------

    def info(self, *, load: bool = True) -> dict:
        m = self.meta
        meta = {"path": str(m.path), "model_id": m.model_id, "name": m.name, "arch": m.arch, "n_layer": m.n_layer,
                "n_embd": m.n_embd, "n_ff": m.n_ff, "n_ff_per_layer": m.n_ff_per_layer, "n_vocab": m.n_vocab,
                "n_expert": m.n_expert, "has_post_ffn_norm": m.has_post_ffn_norm,
                "ffn_down_scale": {str(k): v for k, v in m.ffn_down_scale.items()} if m.ffn_down_scale else None,
                "tied_output": m.tied_output, "has_chat_template": bool(m.chat_template),
                "ffn_down_qtype": m.ffn_down_qtype, "file_size": m.file_size}
        llama = None
        if load:
            try:
                e = self.engine
                llama = {"ok": True, "n_layer": e.n_layer, "n_embd": e.n_embd, "n_vocab": e.n_vocab}
                if e.n_layer != m.n_layer:
                    self.warn(f"llama.cpp の n_layer {e.n_layer} と block_count {m.n_layer} が違います(MTP 層など)")
            except Exception as ex:  # noqa: BLE001
                llama = {"ok": False, "error": str(ex)}
        mode, reason, checked = bias_support(m.arch)
        rp = self.store.resid_path(m.model_id)
        acts = sorted(int(p.stem.split("_L")[1]) for p in self.store.stats_dir(m.model_id).glob("act_L*.npz"))
        return {"meta": meta, "arch_warnings": self.weights.check_arch(), "llama_cpp": llama,
                "bake": {"supported": mode is not None, "mode": mode, "reason": reason, "checked_llama_cpp": checked},
                "stats": {"resid": str(rp) if rp.exists() else None, "act_layers": acts}}

    # --- stats ----------------------------------------------------------------------------------

    def stats_resid(self, *, corpus: str | Path | None = None, tokens: int | None = None,
                    layers: list[int] | None = None, include_last_layer: bool = False,
                    chunk_tokens: int = 512) -> dict:
        c = load_corpus(corpus)
        t0 = time.perf_counter()
        st = residual_norms(self.engine, c, layers, chunk_tokens=chunk_tokens, include_last_layer=include_last_layer,
                            max_tokens=tokens, progress=self.progress)
        path = self.store.resid_path(self.meta.model_id)
        if path.exists() and layers is not None:
            st = _merge_resid(ResidStats.load(path), st)
        st.save(path)
        self._resid = st
        return {"path": str(path), "stats": st.to_dict(), "corpus": {"path": c.path, "sha": c.sha},
                "seconds": time.perf_counter() - t0}

    def stats_act(self, *, corpus: str | Path | None = None, tokens: int | None = None,
                  layers: list[int] | None = None, include_last_layer: bool = False,
                  chunk_tokens: int = 512) -> dict:
        c = load_corpus(corpus)
        t0 = time.perf_counter()
        res = neuron_activation_stats(self.engine, c, layers, chunk_tokens=chunk_tokens,
                                      include_last_layer=include_last_layer, max_tokens=tokens, progress=self.progress)
        rows = []
        for L, a in sorted(res.items()):
            p = self.store.act_path(self.meta.model_id, L)
            a.save(p)
            rows.append({"layer": L, "tensor_name": a.tensor_name, "path": str(p), "n_tokens": a.n_tokens,
                         "absmax_median": float(np.median(a.absmax)), "q999_median": float(np.median(a.q999)),
                         "rms_median": float(np.median(a.rms))})
        return {"layers": rows, "corpus": {"path": c.path, "sha": c.sha}, "seconds": time.perf_counter() - t0}

    def resid_stats(self) -> ResidStats | None:
        if self._resid is None:
            p = self.store.resid_path(self.meta.model_id)
            if p.exists():
                self._resid = ResidStats.load(p)
        return self._resid

    def stats_view(self, recipe: Recipe | None = None, *, auto: bool = True) -> StatsView:
        """レシピに要る統計を用意する。auto なら足りない統計を既定コーパスで計算する(対話的な確認はしない)。"""
        need_rel = recipe is None or recipe.dose_unit == "rel" or any(u.kind == "random" for u in recipe.units)
        resid = self.resid_stats()
        if recipe is not None and need_rel:
            missing = [u.layer for u in recipe.units if resid is None or u.layer not in resid.layers]
            if missing:
                if not auto:
                    raise GneError("STATS_MISSING", f"層 {sorted(set(missing))} の残差ノルム統計がありません",
                                   "gne stats resid を先に実行してください")
                self.warn("残差ノルム統計が無いので既定コーパスで計算しました(gne stats resid)")
                last = self.meta.n_layer - 1 in missing
                if resid is None:
                    self.stats_resid()
                    if last:
                        self.stats_resid(layers=[self.meta.n_layer - 1], include_last_layer=True)
                else:
                    self.stats_resid(layers=sorted(set(missing)), include_last_layer=last)
                resid = self.resid_stats()
        if recipe is not None and recipe.dose_unit == "absmax":
            missing = sorted({u.layer for u in recipe.units
                              if not self.store.act_path(self.meta.model_id, u.layer).exists()})
            if missing:
                if not auto:
                    raise GneError("STATS_MISSING", f"層 {missing} の活性統計がありません", "gne stats act を先に実行してください")
                self.warn(f"層 {missing} の活性統計が無いので既定コーパスで計算しました(gne stats act)")
                self.stats_act(layers=missing, include_last_layer=self.meta.n_layer - 1 in missing)

        def act_loader(L: int) -> ActStats | None:
            p = self.store.act_path(self.meta.model_id, L)
            return ActStats.load(p) if p.exists() else None

        return StatsView(resid, act_loader)

    # --- recipe ---------------------------------------------------------------------------------

    def recipe(self, *, units: list[str] | None = None, path: str | Path | None = None,
               dose_unit: str | None = None, scale: float = 1.0, name: str | None = None) -> Recipe:
        if bool(units) == bool(path):
            raise GneError("BAD_ARGS", "--recipe か --unit のどちらか一方を指定してください")
        if path:
            if not Path(path).exists():
                raise GneError("FILE_NOT_FOUND", f"レシピ {path} がありません")
            r = Recipe.load(path)
            if dose_unit:
                r.dose_unit = dose_unit
        else:
            r = Recipe([parse_unit(u) for u in units], self.meta.model_id, dose_unit or "rel", name)
        r.validate()
        if r.model_id and r.model_id != self.meta.model_id:
            self.warn(f"レシピの model_id {r.model_id!r} が読み込んだモデル {self.meta.model_id!r} と違います")
        for u in r.units:
            if not 0 <= u.layer < self.meta.n_layer:
                raise GneError("BAD_RECIPE", f"層 {u.layer} は範囲外です(0..{self.meta.n_layer - 1})")
            if u.kind == "neuron" and not 0 <= u.neuron < self.meta.n_ff_at(u.layer):
                raise GneError("BAD_RECIPE", f"L{u.layer} のニューロン {u.neuron} は範囲外です(0..{self.meta.n_ff_at(u.layer) - 1})")
            if u.layer <= 2:
                self.warn(f"層 {u.layer}: 層 0〜2 は残差が小さく rel dose がそのまま通用しないことがあります(R9)")
        return r.scaled(scale) if scale != 1.0 else r

    def _as_recipe(self, recipe: Recipe | list[str]) -> Recipe:
        return recipe if isinstance(recipe, Recipe) else self.recipe(units=list(recipe))

    def vectors(self, recipe: Recipe) -> tuple[dict[int, np.ndarray], list[dict]]:
        sv = self.stats_view(recipe)
        return recipe_vectors(self.weights, sv, recipe), unit_details(self.weights, sv, recipe)

    @staticmethod
    def _norms(vecs: dict[int, np.ndarray]) -> dict[str, float]:
        return {str(L): float(np.linalg.norm(v)) for L, v in sorted(vecs.items())}

    def messages(self, prompt: str | None = None, messages: list[dict] | None = None,
                 system: str | None = None) -> list[dict]:
        if messages:
            return messages
        if prompt is None:
            raise GneError("BAD_ARGS", "--prompt が要ります")
        return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]

    def generate(self, vectors: dict[int, np.ndarray] | None, messages: list[dict], *, max_tokens: int = 128) -> dict:
        """1 本で greedy 生成する(バッチの組み方で結果が変わらないように、F2)。"""
        e = self.engine
        ids = e.tokenize_prompt(e.render_chat(messages))
        ctx = e.steer_ctx()
        try:
            e.set_cvec(ctx, vectors)
        except ValueError as ex:
            raise GneError("BAD_RECIPE", str(ex), "ブロック 0 は control vector で操作できません(bake なら可)") from ex
        try:
            r = e.generate_greedy(ctx, [ids], max_tokens=max_tokens)[0]
        finally:
            e.set_cvec(ctx, None)
        return {"text": r.text, "finish": r.finish, "n_gen": r.n_gen, "token_ids": r.token_ids}

    # --- steer ----------------------------------------------------------------------------------

    def steer(self, recipe: Recipe | list[str], *, prompt: str | None = None, messages: list[dict] | None = None,
              system: str | None = None, max_tokens: int = 128, compare: bool = True) -> dict:
        recipe = self._as_recipe(recipe)
        msgs = self.messages(prompt, messages, system)
        vecs, details = self.vectors(recipe)
        base = self.generate(None, msgs, max_tokens=max_tokens) if compare else None
        st = self.generate(vecs, msgs, max_tokens=max_tokens)
        return {"messages": msgs, "recipe": recipe.to_dict(), "units": details, "layer_norms": self._norms(vecs),
                "baseline": base, "steered": st, "differs": None if base is None else base["token_ids"] != st["token_ids"]}

    # --- export / import ------------------------------------------------------------------------

    def export_cvec(self, recipe: Recipe | list[str], path: str | Path, *, fmt: str = "cvec", verify: bool = False,
                    verify_prompt: str = "Who are you?", n_tokens: int = 32,
                    llama_completion: str | None = None) -> dict:
        recipe = self._as_recipe(recipe)
        vecs, details = self.vectors(recipe)
        if fmt == "cvec" and 0 in vecs:
            raise GneError("BAD_RECIPE", "ブロック 0 は control vector に書けません(llama.cpp の仕様)",
                           "--format npz/json/safetensors か gne bake を使ってください")
        meta = {"model_id": self.meta.model_id, "recipe": recipe.to_dict(), "dose_unit": recipe.dose_unit,
                "n_embd": self.meta.n_embd, "gne_version": _version()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        export_vectors(vecs, path, fmt=fmt, n_layer=self.meta.n_layer, model_hint=self.meta.arch, meta=meta)
        check = None
        if verify:
            if fmt != "cvec":
                raise GneError("BAD_ARGS", "--verify は --format cvec のときだけ使えます")
            check = self.check_cvec_with_llama(Path(path), verify_prompt, n_tokens, llama_completion)
        return {"path": str(path), "format": fmt, "recipe": recipe.to_dict(), "units": details,
                "layer_norms": self._norms(vecs), "verify": check}

    def check_cvec_with_llama(self, path: Path, prompt: str, n_tokens: int, binary: str | None) -> dict:
        """書き出した cvec を本家のローダ(llama-completion --control-vector)で読ませ、greedy 出力を比べる。"""
        from .verify import find_llama_completion, run_llama_completion
        binary = binary or find_llama_completion()
        if not binary:
            raise GneError("TOOL_MISSING", "llama-completion が見つかりません", "--llama-completion でパスを渡してください")
        n_embd, data = load_cvec_like_common([(path, 1.0)])
        back = {i + 1: data[i * n_embd:(i + 1) * n_embd] for i in range(data.size // n_embd)
                if np.any(data[i * n_embd:(i + 1) * n_embd])}
        msgs = [{"role": "user", "content": prompt}]
        ours = self.generate(back, msgs, max_tokens=n_tokens)
        base = self.generate(None, msgs, max_tokens=n_tokens)
        theirs = run_llama_completion(binary, self.engine, self.engine.render_chat(msgs), n_tokens, path)
        same = ours["text"] == theirs
        return {"tool": binary, "prompt": prompt, "n_tokens": n_tokens, "ours": ours["text"], "theirs": theirs,
                "match": same, "exact": same, "first_diff": None,
                "summary": "文字列が一致" if same else "文字列が不一致", "kl_next": None,
                "differs_from_baseline": ours["token_ids"] != base["token_ids"]}

    def import_cvec(self, path: str | Path, *, source: str = "exact", name: str | None = None) -> dict:
        if source == "exact":
            self.warn("--source exact: direction.N を層 N の値として扱いました。llama-cvector-generator で作った"
                      "ファイルなら --source generator(1 ずれ補正、§5.2)を使ってください")
        vecs = import_cvec_gguf(path, source=source)
        if not vecs:
            raise GneError("EMPTY_CVEC", f"{path} に非ゼロの方向がありません")
        n = name or Path(path).name.split(".")[0]
        out = self.store.reverse_dir(self.meta.model_id) / f"{n}.directions.npz"
        np.savez(out, layers=np.array(sorted(vecs), dtype=np.int32),
                 **{f"layer_{L}": v for L, v in vecs.items()}, source=source, source_file=str(path))
        return {"source_file": str(path), "source": source, "layer_norms": self._norms(vecs), "path": str(out)}

    # --- bake -----------------------------------------------------------------------------------

    def bake(self, recipe: Recipe | list[str], dst: str | Path, *, verify: bool = False,
             verify_prompt: str = "Who are you?", n_tokens: int = 32) -> dict:
        recipe = self._as_recipe(recipe)
        vecs, details = self.vectors(recipe)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        rep = bake_bias_gguf(self.weights, vecs, self.path, dst, recipe=recipe)
        check = None
        if verify:
            check = self.check_bake(vecs, Path(dst), verify_prompt, n_tokens)
        return {"recipe": recipe.to_dict(), "units": details, "report": rep.__dict__, "verify": check}

    def check_bake(self, vecs: dict[int, np.ndarray], dst: Path, prompt: str, n_tokens: int,
                   kl_max: float = 0.05) -> dict:
        """V9: 焼いた GGUF の greedy 出力と 元モデル + cvec の greedy 出力を比べる。焼いた結果が無介入と違うことも見る。

        bias と control vector は数学的に等価だが、加算の順番((x + Wa) + v と x + (Wa + b))の丸めの差(1e-8 程度)が
        量子化 GEMM の活性の int8 丸めで増幅され、次トークン分布で KL 1e-3〜1e-2 程度ずれる(docs/verification.md F8)。
        完全一致か、最初の分岐が僅差(0.5 nats 未満)で説明でき KL < kl_max なら合格とする。"""
        from .engine import log_softmax
        from .verify import compare_greedy
        msgs = [{"role": "user", "content": prompt}]
        e = self.engine
        ids = e.tokenize_prompt(e.render_chat(msgs))
        ctx = e.steer_ctx()

        def run(engine, ctx_, v):
            engine.set_cvec(ctx_, v)
            engine.clear_memory(ctx_)
            pre = engine.prefill(ctx_, [ids])
            r = engine.generate_greedy(ctx_, [ids], max_tokens=n_tokens, prefilled=pre)[0]
            engine.set_cvec(ctx_, None)
            return r, log_softmax(pre.logits[0])

        base, _ = run(e, ctx, None)
        ref, ref_lp = (None, None) if 0 in vecs else run(e, ctx, vecs)
        if ref is None:
            self.warn("層 0 は control vector で再現できないので、bake の比較は無介入との違いだけを見ます")
        kw = dict(self._engine_kw)
        self.close()                                   # VRAM を空けてから焼いた方を読む
        with Engine(dst, **kw) as e2:
            baked, baked_lp = run(e2, e2.steer_ctx(), None)
        out = {"tool": "llama-cpp-python(焼いた GGUF を読み込み)", "prompt": prompt, "n_tokens": n_tokens,
               "ours": ref.text if ref else "", "theirs": baked.text, "match": False, "exact": False,
               "first_diff": None, "summary": None, "kl_next": None,
               "differs_from_baseline": baked.token_ids != base.token_ids}
        if ref is not None:
            cmp = compare_greedy(ref.token_ids, ref.top_ids, ref.top_lps, baked.token_ids, baked.top_ids, baked.top_lps)
            p = np.exp(ref_lp.astype(np.float64))
            kl = float((p * (ref_lp - baked_lp)).sum())
            out.update(exact=cmp["first_diff"] is None and len(ref.token_ids) == len(baked.token_ids),
                       first_diff=cmp["first_diff"], summary=cmp["summary"], kl_next=kl,
                       match=bool(cmp["ok"]) and kl < kl_max)
        return out


def template_override(model_id: str, arch: str) -> tuple[str, str] | None:
    """configs/templates/<model_id>.yaml か <arch>.yaml の chat_template(§5.7 の 4)。"""
    import yaml
    from .store import CONFIGS
    for key in (model_id, arch):
        p = CONFIGS / "templates" / f"{key}.yaml"
        if p.exists():
            tpl = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("chat_template")
            if tpl:
                return str(p), tpl
    return None


def _merge_resid(old: ResidStats, new: ResidStats) -> ResidStats:
    """層を指定して計算したとき、既存の統計の他の層を残す(コーパスが同じ場合だけ)。"""
    if old.corpus_sha != new.corpus_sha or old.chunk_tokens != new.chunk_tokens:
        return new
    keep = [i for i, L in enumerate(old.layers) if L not in set(new.layers.tolist())]
    cat = lambda a, b: np.concatenate([a[keep], b])  # noqa: E731
    order = np.argsort(cat(old.layers, new.layers))
    return ResidStats(cat(old.layers, new.layers)[order], cat(old.p10, new.p10)[order],
                      cat(old.p50, new.p50)[order], cat(old.p90, new.p90)[order], cat(old.mean, new.mean)[order],
                      new.n_tokens, new.corpus_sha, new.chunk_tokens)


def _version() -> str:
    from . import __version__
    return __version__


__all__ = ["open_model", "Session", "GneError"]
