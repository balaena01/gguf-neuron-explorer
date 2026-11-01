"""Python API。設計書 §6.13 (c)。CLI はこの薄い皮。

    from gne.api import open_model
    with open_model("models/Qwen3.5-4B-Q8_0.gguf") as s:
        out = s.steer(["14:8949:+1.0", "13:7956:-1.0"], prompt="Who are you?")

戻り値は --json と同じ形の dict(エンベロープ抜き)。エンベロープは envelope() で作る。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .engine import Engine
from .schemas import SCHEMA_VERSION
from .stats import (ActStats, ResidStats, StatsView, load_corpus, neuron_activation_stats,
                    residual_norms)
from .steer import (GneError, Recipe, bake_bias_gguf, bias_support, export_vectors, import_cvec_gguf,
                    load_cvec_like_common, parse_unit, recipe_vectors, unit_details, Unit)
from .store import CONFIGS, Store
from .weights import Weights

Progress = Callable[[int, int, str], None]


def open_model(path: str | Path, **kw) -> Session:
    return Session(path, **kw)


MIN_ROUTED = 200            # MoE: 専門家の観測トークン数がこれより少ないと分位点を信頼できない(CHANGE-05 §B.1)


class Session:
    def __init__(self, path: str | Path, *, root: str | Path | None = None, n_gpu_layers: int = -1,
                 n_ctx: int = 4096, n_seq_max: int = 8, seed: int = 0, verbose: bool = False,
                 progress: Progress | None = None, mmproj: str | Path | None = None):
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
        self._engine_kw = dict(n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, n_seq_max=n_seq_max, verbose=verbose,
                               mmproj=mmproj)       # mmproj: 音声の入力(media.py)。無ければモデルの横の mmproj*.gguf
        self._engine: Engine | None = None
        self._resid: ResidStats | None = None
        self.last_patch_reload_s: float | None = None   # パッチを当てたコピーの読み直しにかかった秒(CHANGE-05)

    # --- lifecycle ------------------------------------------------------------------------------

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = self._make_engine(self.path)
        return self._engine

    def _make_engine(self, path) -> Engine:
        e = Engine(path, **self._engine_kw)
        tpl = template_override(self.meta.model_id, self.meta.arch)
        if tpl is not None:
            e.chat_template = tpl[1]
            self.warn(f"チャットテンプレートを {tpl[0]} で上書きしました")
        return e

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
        from .ablate import writer_info
        from .estimate import Speed, is_large, recommended_workflow
        sp = Speed(self)
        speed = {**sp.to_dict(), "cpu_layers": m.n_layer - sp.gpu_layers,
                 "decode_tok_s": 1000.0 / sp.step1, "large": is_large(self)}
        return {"meta": meta, "arch_warnings": self.weights.check_arch(), "llama_cpp": llama,
                "bake": {"supported": mode is not None, "mode": mode, "reason": reason, "checked_llama_cpp": checked},
                "stats": {"resid": str(rp) if rp.exists() else None, "act_layers": acts},
                "residual_writers": writer_info(self.weights), "speed": speed,
                "workflow": recommended_workflow(self) if speed["large"] else None}

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
            row = {"layer": L, "tensor_name": a.tensor_name, "path": str(p), "n_tokens": a.n_tokens,
                   "absmax_median": float(np.median(a.absmax)), "q999_median": float(np.median(a.q999)),
                   "rms_median": float(np.median(a.rms))}
            if a.is_moe:                         # CHANGE-05 §B.1: 観測数の少ない専門家は分位点を信頼できない
                row["moe"] = {"n_expert": int(len(a.n_routed)), "n_routed_min": int(a.n_routed.min()),
                              "n_routed_median": float(np.median(a.n_routed)),
                              "rare": [int(e) for e in np.flatnonzero(a.n_routed < MIN_ROUTED)]}
            rows.append(row)
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
               dose_unit: str | None = None, scale: float = 1.0, name: str | None = None,
               when: str | None = None, ablate: str | dict | None = None, default_direction: str | None = None,
               direction_layer: str | None = None) -> Recipe:
        """ablate は方向の射影消去(§6.17)。"NAME[@層][:倍率]" か {direction, scale, layers, from_layer}。
        ablate だけ(ユニットなし)のレシピも作れる。"""
        from .steer import AblateSpec
        if units and path:
            raise GneError("BAD_ARGS", "--recipe か --unit のどちらか一方を指定してください")
        if not units and not path and not ablate:
            raise GneError("BAD_ARGS", "--recipe か --unit(または --ablate)を指定してください")
        if not units and not path:
            r = Recipe([], self.meta.model_id, dose_unit or "rel", name)
        elif path:
            if not Path(path).exists():
                raise GneError("FILE_NOT_FOUND", f"レシピ {path} がありません")
            r = Recipe.load(path)
            if dose_unit:
                r.dose_unit = dose_unit
        else:
            r = Recipe([parse_unit(u) for u in units], self.meta.model_id, dose_unit or "rel", name)
        if when:
            r.when = when
        if ablate:
            r.ablate = AblateSpec.from_dict(ablate)
        if default_direction and not r.default_direction:      # CHANGE-06: @NAME を省いた proj のユニットの方向
            r.default_direction = default_direction
        if direction_layer:
            r.direction_layer = direction_layer
        r.validate()
        if r.ablate is not None:
            self.direction_path(r.ablate.direction)              # 方向が無ければここで返す
        if r.model_id and r.model_id != self.meta.model_id:
            self.warn(f"レシピの model_id {r.model_id!r} が読み込んだモデル {self.meta.model_id!r} と違います")
        for u in r.units:
            if not 0 <= u.layer < self.meta.n_layer:
                raise GneError("BAD_RECIPE", f"層 {u.layer} は範囲外です(0..{self.meta.n_layer - 1})")
            if u.kind == "neuron" and self.weights.is_moe(u.layer):
                raise GneError("BAD_RECIPE", f"層 {u.layer} は MoE です。専門家のニューロンは {u.layer}:E<専門家>:N{u.neuron} の形で指定します",
                               "shared expert のニューロンなら L:S:Nn")
            if u.kind == "neuron" and not 0 <= u.neuron < self.meta.n_ff_at(u.layer):
                raise GneError("BAD_RECIPE", f"L{u.layer} のニューロン {u.neuron} は範囲外です(0..{self.meta.n_ff_at(u.layer) - 1})")
            if u.kind in ("expert_neuron", "expert"):
                li = self.weights.moe_layer(u.layer)                    # MoE でない層ならここで返す
                if not 0 <= u.expert < li.n_expert:
                    raise GneError("BAD_RECIPE", f"L{u.layer} の専門家 {u.expert} は範囲外です(0..{li.n_expert - 1})")
                if u.kind == "expert_neuron" and not 0 <= u.neuron < li.n_ff:
                    raise GneError("BAD_RECIPE", f"L{u.layer} E{u.expert} のニューロン {u.neuron} は範囲外です(0..{li.n_ff - 1})")
                if u.mode == "route" and not li.has_router_bias:
                    raise GneError("ARCH_UNSUPPORTED", f"層 {u.layer} にはルーターのバイアスがありません({self.meta.arch})",
                                   "route はルーターのバイアス(exp_probs_b)を持つ arch(lfm2moe・deepseek2 など)だけです")
            if u.kind == "shared":
                li = self.weights.arch.layers[u.layer]
                if not li.n_ff_shexp:
                    raise GneError("BAD_RECIPE", f"層 {u.layer} に shared expert はありません")
                if not 0 <= u.neuron < li.n_ff_shexp:
                    raise GneError("BAD_RECIPE", f"L{u.layer} S のニューロン {u.neuron} は範囲外です(0..{li.n_ff_shexp - 1})")
            if u.kind in ("expert_neuron", "shared") and u.mode == "add":
                self.warn(f"{u.label()}: MoE の add は書き込み方向の押しです(専門家が選ばれたかに関係なく全トークンに足す)。"
                          "選ばれたときだけ効くニューロン押しは gain(x)です")
            if u.kind == "head":
                self.weights.head_block(u.layer, u.head)            # 層にヘッドが無い・範囲外ならここで返す
            if u.mode == "proj":                                    # CHANGE-06: 方向が無ければここで返す
                self.direction_set(u.direction or r.default_direction)
            if u.layer <= 2:
                self.warn(f"層 {u.layer}: 層 0〜2 は残差が小さく rel dose がそのまま通用しないことがあります(R9)")
        return r.scaled(scale) if scale != 1.0 else r

    def _as_recipe(self, recipe: Recipe | list[str]) -> Recipe:
        return recipe if isinstance(recipe, Recipe) else self.recipe(units=list(recipe))

    def vectors(self, recipe: Recipe) -> tuple[dict[int, np.ndarray], list[dict]]:
        """add ユニットの control vector(層ごと)と、全ユニットの詳細。gain だけのレシピなら統計は要らない。"""
        if not recipe.add_units:
            return {}, unit_details(self.weights, None, recipe)
        sv = self.stats_view(Recipe(recipe.add_units, recipe.model_id, recipe.dose_unit, recipe.name))
        return (recipe_vectors(self.weights, sv, recipe, self.direction_vec),
                unit_details(self.weights, sv, recipe, self.direction_vec))

    def direction_path(self, name: str) -> Path:
        """方向の名前 → ファイル。<root>/directions/ を先に、無ければ逆引きの <root>/reverse/ を探す。パスも可。"""
        p = Path(name)
        if p.suffix == ".npz" and p.exists():
            return p
        for q in (self.store.direction_dir(self.meta.model_id) / f"{name}.npz",
                  self.store.reverse_dir(self.meta.model_id) / f"{name}.directions.npz"):
            if q.exists():
                return q
        raise GneError("FILE_NOT_FOUND", f"方向 {name} がありません", "gne direction --contrast <対照集合> で作ってください")

    def direction_set(self, name: str):
        from .direction import DirectionSet
        p = self.direction_path(name)
        key, mt = str(p.resolve()), p.stat().st_mtime
        cache = self.__dict__.setdefault("_dirsets", {})
        if key not in cache or cache[key][0] != mt:
            cache[key] = (mt, DirectionSet.load(p))
        return cache[key][1]

    def direction_vec(self, name: str, L: int) -> np.ndarray:
        """方向ユニット(L:dir:NAME:dose)が足す元のベクトル(その層の mean_A − mean_B)。"""
        ds = self.direction_set(name)
        if L not in ds.raw:
            raise GneError("BAD_RECIPE", f"方向 {name} に層 {L} がありません(0〜{max(ds.raw)})")
        return ds.raw[L]

    @staticmethod
    def _norms(vecs: dict[int, np.ndarray]) -> dict[str, float]:
        return {str(L): float(np.linalg.norm(v)) for L, v in sorted(vecs.items())}

    def messages(self, prompt: str | None = None, messages: list[dict] | None = None,
                 system: str | None = None, audio: list[str | Path] | None = None) -> list[dict]:
        """audio: 音声ファイル。ユーザーの発話を音声(と、あれば prompt の文)にする(media.py)。"""
        if messages:
            return messages
        if audio:
            from .media import audio_message
            user = audio_message(audio, prompt)
        elif prompt is None:
            raise GneError("BAD_ARGS", "--prompt が要ります")
        else:
            user = {"role": "user", "content": prompt}
        return ([{"role": "system", "content": system}] if system else []) + [user]

    def generate(self, vectors: dict[int, np.ndarray] | None, messages: list[dict], *, max_tokens: int = 128,
                 when: str = "all", lora: list[tuple[str, float]] | None = None, sampling=None) -> dict:
        """1 本で生成する(バッチの組み方で結果が変わらないように、F2)。既定は greedy、sampling(engine.Sampling)で
        サンプリング。when は §6.15 の位置別ステアリング、lora は gain(§6.16)の LoRA アダプタ。"""
        from .when import generate_when
        e = self.engine
        r, info = generate_when(e, e.steer_ctx(), messages, vectors, when, max_tokens=max_tokens, lora=lora,
                                sampling=sampling)
        return {"text": r.text, "finish": r.finish, "n_gen": r.n_gen, "token_ids": r.token_ids, "when": info}

    # --- steer ----------------------------------------------------------------------------------

    def steer(self, recipe: Recipe | list[str], *, prompt: str | None = None, messages: list[dict] | None = None,
              system: str | None = None, max_tokens: int = 128, compare: bool = True, when: str | None = None,
              sampling=None, audio: list[str | Path] | None = None) -> dict:
        """sampling(engine.Sampling)を渡すと、無介入と介入を同じ seed で引く(違いは介入だけになる)。
        audio: 音声ファイルをユーザーの発話にする(mmproj が要る)。"""
        recipe = self._as_recipe(recipe)
        msgs = self.messages(prompt, messages, system, audio)
        vecs, details = self.vectors(recipe)
        base = self.generate(None, msgs, max_tokens=max_tokens, sampling=sampling) if compare else None
        lora = self.lora_for(recipe)
        with self.patched(self.patches_for(recipe)):           # 専門家全体の gain・route(作業用コピーで読み直す)
            st = self.generate(vecs, msgs, max_tokens=max_tokens, when=when or recipe.when, lora=lora, sampling=sampling)
        return {"messages": msgs, "recipe": recipe.to_dict(), "units": details, "layer_norms": self._norms(vecs),
                "when": when or recipe.when, "sampling": None if sampling is None else sampling.to_dict(),
                "baseline": base, "steered": st, "differs": None if base is None else base["token_ids"] != st["token_ids"]}

    # --- export / import ------------------------------------------------------------------------

    def lora_for(self, recipe: Recipe | None) -> list[tuple[str, float]] | None:
        """レシピの gain ユニット(§6.16)と ablate(§6.17)を当てる LoRA([(パス, 倍率)])。どちらも無ければ None。
        作ったものは使い回す。ablate の LoRA は倍率 1 で作り、scale はアダプタの倍率で掛ける(eval の倍率列でも 1 ファイル)。"""
        if recipe is None:
            return None
        out = []
        lora_units = [u for u in recipe.gain_units if u.kind != "expert"]   # 専門家全体の gain はパッチ(patches_for)
        if lora_units:
            from .gain import export_gain_lora, lora_cache_path
            gr = Recipe(lora_units, self.meta.model_id, name=recipe.name)
            path = lora_cache_path(self.store.gain_dir(self.meta.model_id), gr)
            if not path.exists():
                tmp = path.with_suffix(".part")
                export_gain_lora(self.weights, gr, tmp)
                tmp.replace(path)
            out.append((str(path), 1.0))
        if recipe.ablate is not None and recipe.ablate.scale != 0:
            out.append((str(self.ablation_lora(recipe.ablate)), float(recipe.ablate.scale)))
        if recipe.proj_units and any(u.k for u in recipe.proj_units):     # CHANGE-06: ヘッドの方向成分の射影
            out.append((str(self.proj_lora(recipe)), 1.0))
        for u in recipe.query_units:                     # CHANGE-04 E: ヘッドの条件付きクエリ操作(実験的)
            if u.k:
                out.append((str(self.query_lora(u)), 1.0))
        return out or None

    # --- スケールパッチ(CHANGE-05 §C・§D)-----------------------------------------------------------

    def patches_for(self, recipe: Recipe | None):
        """専門家全体の gain(ffn_down_exps のスラブを k 倍、k = 0 で消す)とルーターのバイアス(route)のパッチ。無ければ None。"""
        if recipe is None:
            return None
        from .patch import Patcher, PatchSet
        units = [u for u in recipe.units if u.kind == "expert" and u.mode in ("gain", "route")]
        if not units:
            return None
        pt = Patcher(self.weights)
        ps = PatchSet("", self.meta.model_id)
        for u in units:
            if u.mode == "gain":
                ps.extend(pt.scale_slab(f"blk.{u.layer}.ffn_down_exps.weight", u.expert, float(u.k)))
            else:
                name = self.weights.router_bias_name(u.layer)
                if name is None:
                    raise GneError("ARCH_UNSUPPORTED", f"層 {u.layer} にはルーターのバイアスがありません({self.meta.arch})")
                ps.extend(pt.add_element(name, u.expert, float(u.dose)))
        ps.name = "p-" + hashlib.sha256(json.dumps(ps.ops, sort_keys=True).encode()).hexdigest()[:12]
        return ps

    @contextmanager
    def patched(self, ps):
        """パッチを作業用コピーに当て、そのコピーでモデルを読み直す。抜けるとき元に戻し、次に使うときに元のモデルを読む。
        作業用コピーは 1 回だけ作る(cp --reflink=auto)。読み直しは mmap とページキャッシュで数秒(docs/measurements.md)。"""
        if ps is None or not ps.patches:
            yield self.engine
            return
        from .patch import applied, make_work_copy
        work = make_work_copy(self.path, self.store.work_path(self.meta.model_id))
        stale = [n for n in applied(work) if n != ps.name]
        if stale:
            from .patch import PatchSet
            for n in stale:                                   # 前に落ちたときのパッチが残っていれば戻す
                f = self.store.patch_dir(self.meta.model_id) / f"{n}.patch.json"
                if not f.exists():
                    raise GneError("PATCH_CONFLICT", f"作業用コピーに知らないパッチ {n} が掛かっています", f"{work} を消して作り直してください")
                PatchSet.load(f).revert(work)
        ps.save(self.store.patch_dir(self.meta.model_id) / f"{ps.name}.patch.json")
        if self._engine is not None:                          # VRAM を空けてから読み直す
            self._engine.close()
            self._engine = None
        ps.apply(work)
        t0 = time.perf_counter()
        try:
            self._engine = self._make_engine(work)
            self.last_patch_reload_s = time.perf_counter() - t0
            yield self._engine
        finally:
            if self._engine is not None:
                self._engine.close()
                self._engine = None
            ps.revert(work)

    def proj_lora(self, recipe: Recipe) -> Path:
        """レシピの proj のユニット → LoRA(<root>/heads/<model_id>/proj-<鍵>.lora.gguf に作って使い回す)。鍵は
        ヘッド・s・方向の中身(作り直したら LoRA も作り直す)・direction_layer で決まる。"""
        from .headproj import ProjBuilder, export_proj_lora, proj_cache_key, proj_heads
        heads = [p for p in proj_heads(recipe) if p.s]
        b = ProjBuilder(self.weights, self.direction_set, direction_layer=recipe.direction_layer)
        path = self.store.heads_dir(self.meta.model_id) / f"proj-{proj_cache_key(heads, b)}.lora.gguf"
        if not path.exists():
            tmp = path.with_suffix(".part")
            export_proj_lora(self.weights, recipe.like([u for u in recipe.proj_units if u.k]), tmp, self.direction_set,
                             warn=self.warn)
            tmp.replace(path)
        return path

    def query_lora(self, u) -> Path:
        """query のユニット → attn_q の rank-1 LoRA(<root>/heads/<model_id>/ に作って使い回す)。u・v の計算に対照集合の捕捉が要る。"""
        from .heads import query_vectors, write_query_lora
        seek = self._contrast_file(u.seek)
        key = json.dumps([u.layer, u.head, u.direction, str(seek), float(u.k),
                          vector_key_of(self.direction_set(u.direction).path)])
        path = self.store.heads_dir(self.meta.model_id) / f"query-{hashlib.sha256(key.encode()).hexdigest()[:16]}.lora.gguf"
        if not path.exists():
            qv = query_vectors(self, u.layer, u.head, direction=u.direction, seek=seek)
            tmp = path.with_suffix(".part")
            write_query_lora(self, u.layer, u.head, qv, float(u.k), tmp)
            tmp.replace(path)
        return path

    @staticmethod
    def _contrast_file(name: str | Path) -> Path:
        p = Path(name)
        if p.exists():
            return p
        q = CONFIGS / "contrast" / f"{name}.yaml"
        if q.exists():
            return q
        raise GneError("FILE_NOT_FOUND", f"対照集合 {name} がありません(configs/contrast/ の名前かパス)")

    def ablation_lora(self, spec) -> Path:
        """AblateSpec → 倍率 1 の消去 LoRA(<root>/ablate/<model_id>/ に作って使い回す)。"""
        from .ablate import export_ablation_lora, layer_list, vector_key
        r, meta = self.ablation_vector(spec.direction, from_layer=spec.from_layer)
        Ls = layer_list(spec.layers)
        tag = "" if Ls is None else "-L" + hashlib.sha256(json.dumps(Ls).encode()).hexdigest()[:8]
        path = self.store.ablate_dir(self.meta.model_id) / f"{spec.direction}-{vector_key(r)}{tag}.lora.gguf"
        if not path.exists():
            tmp = path.with_suffix(".part")
            export_ablation_lora(self.weights, r, tmp, layers=Ls, meta=meta)
            tmp.replace(path)
        return path

    def gain(self, recipe: Recipe | list[str], out: str | Path, *, bake: bool = False, requant: str = "auto",
             via_f16: bool = False, imatrix: str | None = None, llama_quantize: str | None = None) -> dict:
        """gain ユニットを LoRA アダプタ GGUF(既定)か、W_up(ヘッドは W_O)を書き換えた GGUF(bake)に書き出す(§6.16、CHANGE-04 D)。"""
        from .gain import bake_gain_gguf, export_gain_lora
        recipe = self._as_recipe(recipe)
        if recipe.add_units:
            self.warn("add のユニットは gain の書き出しに入りません(gne export-cvec で control vector に書き出せます)")
        routes = [u for u in recipe.units if u.mode == "route"]
        if not recipe.gain_units and not recipe.query_units and not routes and not recipe.proj_units:
            raise GneError("BAD_RECIPE", "gain のユニットがありません(例 --unit 14:8949:x2、ヘッドは 21:H7:x0、"
                                         "方向成分の射影は 33:H16:s6@NAME)")
        if not bake and (routes or any(u.kind == "expert" for u in recipe.gain_units)):
            raise GneError("BAD_RECIPE", "専門家全体の gain と route は LoRA にできません(全ランクになる)",
                           "--bake でスケールパッチを当てた GGUF にしてください(CHANGE-05 §D)")
        rep: dict = {"path": None, "bytes": 0}
        extra, extra_meta, summary = {}, None, {}
        if recipe.proj_units:                            # CHANGE-06: ヘッドの方向成分の射影も同じ LoRA / 焼き込みに入れる
            from .headproj import ProjBuilder, layer_summary, proj_heads, proj_meta
            heads = proj_heads(recipe)
            b = ProjBuilder(self.weights, self.direction_set, direction_layer=recipe.direction_layer)
            extra = b.tensors(heads, self.warn)
            extra_meta = proj_meta(heads, b)
            summary = layer_summary(self.weights, heads, extra)
            if bake and requant == "patch":
                raise GneError("UNSUPPORTED_QUANT", "方向成分の射影はスケールパッチでは書けません(列方向で、単純な倍率ではない)",
                               "--requant q8_0 / f16、または gne headproj export --bake --via-f16")
        if via_f16 and not bake:
            raise GneError("BAD_ARGS", "--via-f16 は --bake と一緒に使います")
        if recipe.gain_units or routes or extra:
            if bake and via_f16:                         # F16 で書いて llama-quantize で元の型に戻す(CHANGE-06 E.2)
                from .ablate import requant_via_f16
                rep = requant_via_f16(self.weights, lambda mid: bake_gain_gguf(self.weights, recipe, mid, requant="f16",
                                                                              extra=extra, extra_meta=extra_meta,
                                                                              via_f16=True),
                                      out, llama_quantize=llama_quantize, imatrix=imatrix)
                rep["modified"] = [{**m, "to": m["from"]} for m in rep["modified"]]
            elif bake:
                rep = bake_gain_gguf(self.weights, recipe, out, requant=requant, extra=extra, extra_meta=extra_meta)
            else:
                rep = export_gain_lora(self.weights, recipe, out, extra=extra, extra_meta=extra_meta)
        rep.update(summary)
        if bake and extra:                               # CHANGE-06 E.2: 書き換えたテンソルと元の型のサイドカー
            side = Path(str(rep["path"]) + ".headproj.json")
            side.write_text(json.dumps({"recipe": recipe.to_dict(), "headproj": extra_meta, "modified": rep.get("modified"),
                                        "requant": "via-f16" if via_f16 else requant}, ensure_ascii=False, indent=1),
                            encoding="utf-8")
            rep["sidecar"] = str(side)
        if recipe.query_units:                           # CHANGE-04 E: クエリ操作は別の LoRA(llama.cpp の --lora を並べる)
            import shutil
            stem = Path(out).name.removesuffix(".gguf").removesuffix(".lora")
            q = []
            for u in recipe.query_units:
                dst = Path(out).with_name(f"{stem}-query-L{u.layer}-H{u.head}.lora.gguf")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.query_lora(u), dst)
                q.append(str(dst))
            rep["query_loras"] = q
            if rep["path"] is None:
                rep["path"], rep["bytes"] = q[0], Path(q[0]).stat().st_size
        return {"kind": "bake" if bake else "lora", "recipe": recipe.to_dict(), **rep}

    def _warn_when(self, recipe: Recipe, what: str) -> None:
        if recipe.ablate is not None:
            self.warn(f"ablate({recipe.ablate.label()})は {what} では表せないので外しました(§6.17)。"
                      "gne ablate で LoRA アダプタか書き換え GGUF に書き出してください")
        if recipe.gain_units:
            self.warn(f"gain のユニット({len(recipe.gain_units)} 個)は {what} では表せないので外しました(§6.16)。"
                      "gne gain で LoRA アダプタか書き換え GGUF に書き出してください")
        if recipe.proj_units:
            self.warn(f"ヘッドの方向成分の射影(proj、{len(recipe.proj_units)} 個)は {what} では表せないので外しました"
                      "(control vector は足し算だけ、CHANGE-06)。gne headproj export(か gne gain)で LoRA に書き出してください")
        if recipe.when != "all":
            self.warn(f"レシピの when={recipe.when} は {what} では表せないので、全位置(all)として書き出しました(§6.15)。"
                      "位置別のまま使うなら gne steer / chat か gne serve --openai を使ってください")

    def export_cvec(self, recipe: Recipe | list[str], path: str | Path, *, fmt: str = "cvec", verify: bool = False,
                    verify_prompt: str = "Who are you?", n_tokens: int = 32,
                    llama_completion: str | None = None) -> dict:
        recipe = self._as_recipe(recipe)
        self._warn_when(recipe, "control vector")
        if not recipe.add_units:
            raise GneError("BAD_RECIPE", "add のユニットがありません(gain・proj だけのレシピは control vector になりません)",
                           "gne gain / gne headproj export で LoRA アダプタか書き換え GGUF に書き出してください")
        moe_add = [u.label() for u in recipe.add_units if u.kind in ("expert_neuron", "shared")]
        if moe_add:                                           # CHANGE-05 §C・V22
            self.warn(f"{', '.join(moe_add)}: MoE の add は書き込み方向の押しとして書き出します(専門家が選ばれたかに関係なく全トークンに足す)")
        rest = [u.label() for u in recipe.units if u.mode in ("gain", "route")]
        if rest:
            self.warn(f"{', '.join(rest)}: gain と route は control vector に入りません"
                      "(ニューロン・ヘッド・専門家のニューロンの gain は LoRA、専門家全体の gain と route はスケールパッチ)")
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
        from .steer import llama_server_args
        return {"path": str(path), "format": fmt, "recipe": recipe.to_dict(), "units": details,
                "layer_norms": self._norms(vecs), "verify": check,
                "llama_server_args": llama_server_args(recipe.sampling) if recipe.sampling else None}

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
        self._warn_when(recipe, "bake")
        if not recipe.add_units:
            raise GneError("BAD_RECIPE", "add のユニットがありません", "gain は gne gain --bake で書き換え GGUF にできます")
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


    # --- Phase 2: lens / reverse ----------------------------------------------------------------

    def _tok(self, ids) -> list[str]:
        return [self.engine.token_text(int(i)) for i in ids]

    def _need_rho(self, layers: list[int]) -> StatsView:
        return self.stats_view(Recipe([Unit(L, 0, 1.0) for L in layers]))

    def lens_logit(self, L: int, *, neurons: str | None = None, top_k: int = 12, show: int = 20) -> dict:
        from .lens import LogitLens, logit_lens_layer
        from .selection import parse_neurons
        self._check_layer(L)
        sel = parse_neurons(neurons, self.weights.n_units(L), seed=self.seed)
        t0 = time.perf_counter()
        ll = logit_lens_layer(self.weights, L, neurons=sel, top_k=top_k, progress=self.progress)
        path = self.store.lens_dir(self.meta.model_id) / f"logit_L{L}.npz"
        if path.exists() and (neurons not in (None, "all")):
            ll = _merge_logit(LogitLens.load(path), ll)
        ll.save(path)
        pick = sel if neurons not in (None, "all") and not str(neurons).startswith("random:") else \
            ll.neurons[np.argsort(-ll.peak_z)[:show]]
        row = {int(n): i for i, n in enumerate(ll.neurons)}
        rows = [{"neuron": int(n), "peak_z": float(ll.peak_z[row[int(n)]]),
                 "up": self._tok(ll.up_ids[row[int(n)]]), "up_z": ll.up_z[row[int(n)]].tolist(),
                 "down": self._tok(ll.down_ids[row[int(n)]]), "down_z": ll.down_z[row[int(n)]].tolist()}
                for n in pick]
        return {"layer": L, "path": str(path), "n_neurons": int(sel.size), "top_k": top_k,
                "seconds": time.perf_counter() - t0, "neurons": rows}

    def lens_steer(self, L: int, *, neurons: str | None = None, dose: float = 0.8, random_controls: int = 64,
                   top_k: int = 12, resume: bool = True, show: int = 20, part: str | None = None,
                   controls: list[int] | None = None) -> dict:
        """part を渡すと(ワーカー、CHANGE-03 B)steer_L{L}.part-<part>.npz に書き、controls の対照だけを計算する。"""
        from .lens import steer_lens_layer
        from .selection import parse_neurons
        self._check_layer(L, cvec=True)
        sel = parse_neurons(neurons, self.weights.n_units(L), seed=self.seed)
        sv = self._need_rho([L])
        t0 = time.perf_counter()
        path = self.store.lens_dir(self.meta.model_id) / (f"steer_L{L}.part-{part}.npz" if part else f"steer_L{L}.npz")
        res = steer_lens_layer(self.engine, self.weights, sv, L, neurons=sel, dose=dose, contexts=None, top_k=top_k,
                               random_controls=random_controls, seed=self.seed, out=path, resume=resume,
                               progress=self.progress, controls=controls)
        return self._lens_rows(L, res, sel, neurons, path, dose, float(sv.rho(L)), show, time.perf_counter() - t0, self._tok)

    def _lens_rows(self, L, res, sel, neurons, path, dose, rho, show, seconds, tok) -> dict:
        null = {sg: _summ(res.null(sg)) for sg in ("pos", "neg")}
        explicit = neurons not in (None, "all") and not str(neurons).startswith("random:")
        if explicit:
            pick = sel
        else:
            score = np.fmax(res.kl_pos[sel], res.kl_neg[sel])
            pick = sel[np.argsort(-np.nan_to_num(score, nan=-1))[:show]]
        rows = [{"neuron": int(n), "kl_pos": float(res.kl_pos[n]), "kl_neg": float(res.kl_neg[n]),
                 "pct_pos": float(res.null_pct(res.kl_pos[n], "pos")), "pct_neg": float(res.null_pct(res.kl_neg[n], "neg")),
                 "up": tok(res.up_ids[n]), "up_delta": res.up_delta[n].tolist(),
                 "down": tok(res.down_ids[n]), "down_delta": res.down_delta[n].tolist()} for n in pick]
        return {"layer": L, "path": str(path), "dose": dose, "rho": rho, "n_done": int(res.done.sum()),
                "n_selected": int(sel.size), "n_controls": int(res.ctrl_done.sum()), "null": null,
                "seconds": seconds, "neurons": rows}

    def lens_summary(self, L: int, *, neurons: str | None = None, show: int = 20, seconds: float = 0.0) -> dict:
        """保存済みの steer lens の要約(lens_steer と同じ形)。モデルを GPU に読み込まない(語は語彙だけで読む)。"""
        from .lens import SteerLens
        from .selection import parse_neurons
        path = self.store.lens_dir(self.meta.model_id) / f"steer_L{L}.npz"
        if not path.exists():
            raise GneError("FILE_NOT_FOUND", f"{path} がありません")
        res = SteerLens.load(path)
        sel = parse_neurons(neurons, self.weights.n_units(L), seed=self.seed)
        return self._lens_rows(L, res, sel, neurons, path, float(res.config["dose"]), float(res.config["rho"]), show,
                               seconds, lambda ids: [self.vocab.text(int(i)) for i in ids])

    def _check_layer(self, L: int, *, cvec: bool = False) -> None:
        if not 0 <= L < self.meta.n_layer:
            raise GneError("BAD_ARGS", f"層 {L} は範囲外です(0..{self.meta.n_layer - 1})")
        if cvec and L == 0:
            raise GneError("BAD_ARGS", "層 0 は control vector で押せません(llama.cpp の仕様)")
        if not self.weights.has_tensor(f"blk.{L}.ffn_down.weight") and not self.weights.has_tensor(f"blk.{L}.ffn_down_exps.weight"):
            raise GneError("ARCH_UNSUPPORTED", f"層 {L} に ffn_down.weight(dense)も ffn_down_exps.weight(MoE)もありません")

    def reverse(self, *, contrast: str | Path | None = None, directions: str | Path | None = None,
                method: str = "mean_diff", positions: str = "all", top: int = 200, layers: list[int] | None = None,
                verify: bool = False, verify_top: int | None = None, verify_controls: int = 8,
                verify_design: str = "quick", max_tokens: int = 128, name: str | None = None,
                verify_budget_s: float | None = None,
                standardize: bool = True) -> dict:
        from .lens import SteerLens
        from .reverse import ContrastSet, candidate_dicts, contrastive_directions, rank_by_direction
        if bool(contrast) == bool(directions):
            raise GneError("BAD_ARGS", "--contrast か --directions のどちらか一方を指定してください")
        t0 = time.perf_counter()
        rdir = self.store.reverse_dir(self.meta.model_id)
        act, scale = None, None
        info: dict = {"method": method, "positions": positions, "standardize": standardize}
        if contrast:
            cs = ContrastSet.load(contrast)
            name = name or cs.name
            from .reverse import side_interventions
            cr = contrastive_directions(self.engine, cs, layers=layers, method=method, positions=positions,
                                        progress=self.progress,
                                        interventions=side_interventions(self, cs) if cs.has_recipes else None)
            dirs, act, scale = cr.directions, cr.act_diff, cr.scale
            info.update(source=str(contrast), contrast_sha=cs.sha, n_rows=cr.n_rows, n_prompts=len(cs.user_prompts))
            np.savez(rdir / f"{name}.directions.npz", layers=np.array(sorted(dirs), dtype=np.int32),
                     **{f"layer_{L}": v for L, v in dirs.items()}, **{f"act_{L}": a for L, a in act.items()},
                     **{f"scale_{L}": v for L, v in scale.items()},
                     method=method, positions=positions, contrast_sha=cs.sha, source=str(contrast))
        else:
            p = Path(directions)
            if not p.exists():
                p = self.direction_path(str(directions))         # 逆引きの方向、無ければ gne direction の方向
            z = np.load(p)
            dirs = {int(L): z[f"layer_{L}"] for L in z["layers"] if layers is None or int(L) in layers}
            act = {int(k[4:]): z[k] for k in z.files if k.startswith("act_")} or None
            scale = {int(k[6:]): z[k] for k in z.files if k.startswith("scale_")} or None
            if standardize and scale is None:
                self.warn("この方向には残差のばらつきの情報が無いので、標準化せずに cos を取りました")
            name = name or p.name.split(".")[0]
            info.update(source=str(p), method=str(z["method"]) if "method" in z.files else "imported")
        steer_kl = {}
        for L in dirs:
            sp = self.store.lens_dir(self.meta.model_id) / f"steer_L{L}.npz"
            if sp.exists():
                sl = SteerLens.load(sp)
                steer_kl[L] = (sl.kl_pos, sl.kl_neg)
        cands, cos = rank_by_direction(self.weights, dirs, act_diff=act, steer_kl=steer_kl or None, top=top,
                                       exclude_layers=(0,), scale=scale if standardize else None)
        cpath = rdir / f"{name}.candidates.jsonl"
        cd = candidate_dicts(cands)
        cpath.write_text("".join(json.dumps(c) + "\n" for c in cd))
        layer_summary = [{"layer": L, "max_abs_cos": float(np.abs(c).max()),
                          "top1pct_abs_cos": float(np.sort(np.abs(c))[-max(1, c.size // 100):].mean()),
                          "dir_norm": float(np.linalg.norm(dirs[L]))} for L, c in sorted(cos.items())]
        verified = None
        if verify:
            n_v = verify_top or top
            if verify_budget_s is not None:              # CHANGE-03 C: 候補の層が決まってから、対照(層ごと)込みで予算に収める
                from .estimate import Speed, _design_cost
                per = _design_cost(Speed(self), verify_design, max_tokens)
                n_v = 1
                for k in range(1, min(len(cands), verify_top or top) + 1):
                    n_layers = len({c.layer for c in cands[:k]})
                    if (k + (verify_controls + 1) * n_layers) * per > verify_budget_s:   # +1 は層ごとのベースライン
                        break
                    n_v = k
                info["verify_budget"] = {"budget_s": verify_budget_s, "verify_top": n_v,
                                         "layers": len({c.layer for c in cands[:n_v]}), "seconds_per_unit": per}
            verified = self._verify_candidates(cands[:n_v], name, verify_design, verify_controls, max_tokens)
        by_expert = None
        if any(c.expert is not None for c in cands):            # MoE: 上位の候補が特定の専門家に集まっているか(CHANGE-05 §E.3)
            from collections import Counter
            cnt = Counter((c.layer, c.expert) for c in cands if c.expert is not None)
            by_expert = [{"layer": L, "expert": e, "n": n} for (L, e), n in cnt.most_common(20)]
            info["semantics"] = "direction"
        return {"name": name, "info": info, "candidates_path": str(cpath), "n_candidates": len(cands),
                "candidates": cd, "layers": layer_summary, "steer_lens_layers": sorted(steer_kl),
                "verify": verified, "seconds": time.perf_counter() - t0, "by_expert": by_expert}

    # --- Phase 3: sweep / rank ------------------------------------------------------------------

    @property
    def vocab(self):
        if getattr(self, "_vocab", None) is None:
            from .vocab import Vocab
            self._vocab = Vocab(self.path)
        return self._vocab

    def run_path(self, run: str) -> Path:
        """--run の名前(<root>/runs/<model_id>/<run>.jsonl)か JSONL のパス。"""
        p = Path(run)
        if p.suffix == ".jsonl" and (p.exists() or p.parent != Path(".")):
            return p
        return self.store.runs_dir(self.meta.model_id) / f"{run}.jsonl"

    def sweep(self, L: int, *, neurons: str | None = "random:384", design: str = "main_rel", random_controls: int = 64,
              max_tokens: int = 128, top_k: int = 5, run: str | None = None, prompt_set: str = "en",
              dose_unit: str = "rel", resume: bool = True, part: str | None = None,
              controls: list[int] | None = None) -> dict:
        """part を渡すと(ワーカー、CHANGE-03 B)<run>.part-<part>.jsonl に書き、controls の対照だけを回す。"""
        from .selection import parse_neurons
        from .sweep import load_design, run_sweep
        self._check_layer(L, cvec=True)
        sel = parse_neurons(neurons, self.weights.n_units(L), seed=self.seed)
        des = load_design(design, prompt_set=prompt_set)
        sv = self._need_rho([L]) if dose_unit == "rel" else self.stats_view(
            Recipe([Unit(L, int(sel[0]), 1.0)], dose_unit=dose_unit))
        name = run or f"L{L}-{design}"
        out = self.run_path(name)
        if part:
            out = out.with_name(f"{out.name.removesuffix('.jsonl')}.part-{part}.jsonl")
        r = run_sweep(self.engine, self.weights, sv, L=L, neurons=sel.tolist(), design=des, out=out,
                      model_id=self.meta.model_id, dose_unit=dose_unit, random_controls=random_controls,
                      max_tokens=max_tokens, top_k=top_k, resume=resume, seed=self.seed, progress=self.progress,
                      controls=controls)
        return {"run": name, **r, "design": design, "prompt_set": prompt_set, "n_neurons": int(sel.size),
                "n_controls": random_controls,
                **({"semantics": "direction"} if self.meta.n_expert and self.weights.is_moe(L) else {})}

    def rank(self, run: str, *, word_unit: str = "tokens", max_rel: float = 1.6, show: int = 40,
             find: list[int] | None = None) -> dict:
        from .rank import rank_records
        from .sweep import part_paths, read_rollouts
        if word_unit not in ("tokens", "regex"):
            raise GneError("BAD_ARGS", "--word-unit は tokens か regex です")
        path = self.run_path(run)
        if not part_paths(path):                          # 本体か、ワーカーごとの部分(part-*)のどちらかがあればよい
            raise GneError("FILE_NOT_FOUND", f"run {path} がありません", "gne sweep --run で作ってください")
        t0 = time.perf_counter()
        recs = read_rollouts(path)
        words = self.vocab.words() if word_unit == "tokens" else None
        units = rank_records(recs, word_unit=word_unit, token_words=words, max_rel=max_rel, progress=self.progress)
        out = path.with_name(path.name.removesuffix(".jsonl") + ".units.jsonl")
        out.write_text("".join(json.dumps(u, ensure_ascii=False) + "\n" for u in units), encoding="utf-8")
        ctrl = np.array([u["mass"] for u in units if u["kind"] == "random"])
        neu = [u for u in units if u["kind"] == "neuron"]
        n_above = sum(1 for u in neu if u["null_pct"] is not None and u["null_pct"] >= 1.0)
        found = []
        for n in find or []:
            hit = next((u for u in units if u["kind"] == "neuron" and u["neuron"] == n), None)
            if hit is None:
                self.warn(f"N{n} はこの run にありません")
            else:
                found.append(hit)
        return {"run": str(path), "parts": [p.name for p in part_paths(path)], "units_path": str(out),
                "n_units": len(units), "n_neurons": len(neu),
                "n_controls": int(ctrl.size), "n_baseline_records": sum(r["kind"] == "baseline" for r in recs),
                "word_unit": word_unit, "max_rel": max_rel, "control_mass": _summ(ctrl),
                "n_above_all_controls": n_above, "units": units[:show], "find": found,
                "seconds": time.perf_counter() - t0}

    # --- Phase 4: chat / dossier / agent ------------------------------------------------------------

    def neuron(self, L: int, n: int, *, rollouts: int = 4, compute: bool = False) -> dict:
        from .dossier import neuron_dossier
        return neuron_dossier(self, L, n, rollouts=rollouts, compute=compute)

    def model_overview(self) -> dict:
        from .dossier import model_overview
        return model_overview(self)

    def layer_summary(self, L: int, *, top: int = 20) -> dict:
        from .dossier import layer_summary
        return layer_summary(self, L, top=top)

    def search(self, q: str, *, mode: str = "lens", layer: int | None = None, limit: int = 50) -> dict:
        from .dossier import search
        return search(self, q, mode=mode, layer=layer, limit=limit)

    def fires(self, text: str, *, top: int = 50) -> dict:
        from .fires import fires
        return fires(self, text, top=top)

    # --- Phase 5 U3: connections と配置(重みだけ、CPU) -------------------------------------------------

    def connections(self, *, k: int = 24, window: int | None = None, chunk: int = 1024) -> dict:
        """全層ペアの connections を計算して <root>/connections/<model_id>.npz に保存する(別冊 §8.4)。"""
        from .connections import compute_connections, save_connections
        resid = self.resid_stats()
        rho = {int(L): float(p) for L, p in zip(resid.layers, resid.p50, strict=True)} if resid is not None else None
        if rho is None:
            self.warn("残差ノルムの統計が無いので rms で割らずに計算します(層をまたいだ z が深い層に偏ります。gne stats resid で作れます)")
        r = compute_connections(self.weights, k=k, window=window, rho=rho, chunk=chunk, progress=self.progress)
        path = self.store.connections_path(self.meta.model_id)
        save_connections(path, r)
        return {"path": str(path), **r["meta"]}

    def layout(self, kind: str = "writes", *, layers: list[int] | str | None = None, seed: int | None = None) -> dict:
        """シート内の配置(別冊 §4.4)を計算して <root>/layout/<model_id>/<kind>_L{L}.npy に保存する。"""
        from .layout import compute_layout
        Ls = list(range(self.meta.n_layer)) if layers in (None, "all") else [int(x) for x in layers]
        out = []
        for i, L in enumerate(Ls):
            if self.progress:
                self.progress(i, len(Ls), f"layout {kind} L{L}")
            cell, info = compute_layout(self, kind, L, seed=self.seed if seed is None else seed)
            p = self.store.layout_dir(self.meta.model_id) / f"{kind}_L{L}.npy"
            np.save(p, cell)
            out.append({"layer": L, "path": str(p), **info})
        if self.progress:
            self.progress(len(Ls), len(Ls), f"layout {kind}")
        return {"kind": kind, "layers": out}

    def embed(self, run: str, *, model: str | Path | None = None) -> dict:
        """run の全 rollout を文埋め込み → UMAP 2D にして runs/<run>.umap.json に保存する(別冊 §7)。"""
        from .embed import embed_run
        return embed_run(self, run, model=model, seed=self.seed, progress=self.progress)

    # --- 書き手の強さ -------------------------------------------------------------------------------

    def writers(self, contrast: str | Path, *, name: str | None = None, ref_layer: int | None = None,
                replies_a: str | Path | None = None, replies_b: str | Path | None = None, max_tokens: int = 96,
                top: int = 50) -> dict:
        """対照集合の A / B の返事のトークンで、各ニューロンが残差に書き足した量と、参照の層の方向への直接の寄与。
        方向を埋め込み・mixer(attention など)・MLP がそれぞれ何割書いたかも返す。"""
        from .writers import writer_contributions
        return writer_contributions(self, contrast, name=name, ref_layer=ref_layer, replies_a=replies_a,
                                    replies_b=replies_b, max_tokens=max_tokens, top=top, progress=self.progress)

    # --- attention ヘッド(CHANGE-04) -------------------------------------------------------------

    def heads_contrast(self, contrast: str | Path, *, name: str | None = None, direction: str | None = None,
                       system_b2: str | None = None, layers: list[int] | None = None, max_tokens: int = 48) -> dict:
        """対照集合でヘッドごとに trigger_mass・pattern_shift・out_shift・dir_cos を測り、中立どうしを床にして順位づける。"""
        from .heads import heads_contrast
        return heads_contrast(self, contrast, name=name, direction=direction, system_b2=system_b2, layers=layers,
                              max_tokens=max_tokens, progress=self.progress)

    def experts_contrast(self, contrast: str | Path, *, name: str | None = None, direction: str | None = None,
                         system_b2: str | None = None, layers: list[int] | None = None, max_tokens: int = 48) -> dict:
        """MoE の層の専門家ごとに usage_shift・weight_shift・contrib_cos・contrib_norm を測り、中立どうしを床にして順位づける(CHANGE-05 §E.1)。"""
        from .experts import experts_contrast
        return experts_contrast(self, contrast, name=name, direction=direction, system_b2=system_b2, layers=layers,
                                max_tokens=max_tokens, progress=self.progress)

    def experts_sweep(self, L: int, *, ks=(0.0, 2.0), design: str = "quick", prompt_set: str = "en", max_tokens: int = 48,
                      experts: list[int] | None = None, run: str | None = None) -> dict:
        """専門家単位のスイープ(各専門家を消す・強めるスケールパッチで生成し、語の出方で順位づける、CHANGE-05 §E.2)。"""
        from .experts import experts_sweep
        return experts_sweep(self, L, ks=ks, design=design, prompt_set=prompt_set, max_tokens=max_tokens, experts=experts,
                             run=run, progress=self.progress)

    def moe_route(self, text: str, *, layers: list[int] | None = None, chat: bool = False) -> dict:
        """トークンごとに選ばれた専門家とルーティング重み(MoE の層ごと)。"""
        from .experts import moe_route
        return moe_route(self, text, layers=layers, chat=chat)

    def expert(self, L: int, e: int, *, top: int = 20) -> dict:
        """専門家の調書(使用統計・対照分析での順位・ニューロンの上位)。"""
        from .experts import expert_dossier
        return expert_dossier(self, L, e, top=top)

    def attention(self, text: str, *, pos: int = -1, layer: int | None = None, head: int | None = None, top: int = 5,
                  system: str | None = None, chat: bool = False) -> dict:
        """選んだトークンが各ヘッドでどこを見ているか(上位 top)と per-head 出力のノルム(gne heads attention)。"""
        from .heads import attention_view
        return attention_view(self, text, pos=pos, layer=layer, head=head, top=top, system=system, chat=chat)

    def heads_listen(self, user: str, *, heads: str | list[tuple[int, int]], system: str | None = None, reply: str | None = None,
                     max_tokens: int = 64, direction: str | None = None, top: int = 20) -> dict:
        """返事のトークンが選んだヘッドで文脈のどのトークンに注目し、各トークンが方向にどれだけ寄与したか(gne heads listen)。"""
        from .heads import listen_view
        return listen_view(self, user, heads=heads, system=system, reply=reply, max_tokens=max_tokens, direction=direction, top=top)

    def head(self, L: int, h: int, *, lens: bool = True) -> dict:
        """ヘッドの調書。"""
        from .heads import head_dossier
        return head_dossier(self, L, h, lens=lens)

    def heads_query(self, L: int, h: int, *, direction: str, seek: str | Path, scales=(0.0, 0.5, 1.0, 2.0, 4.0)) -> dict:
        """CHANGE-04 E(実験的)の V17: クエリ操作の scale ごとに、ヘッド h のトリガーへの注目がペルソナあり/なしでどう変わるか。"""
        from .heads import check_query
        return check_query(self, L, h, direction=direction, seek=self._contrast_file(seek), scales=scales)

    def set_head_name(self, L: int, h: int, name: str) -> dict:
        from .heads import set_head_name
        set_head_name(self, L, h, name)
        return {"layer": L, "head": h, "name": name or None}

    def set_name(self, L: int, n: int, name: str) -> dict:
        from .dossier import set_name
        set_name(self, L, n, name)
        return {"layer": L, "neuron": n, "name": name or None}

    def recipe_save(self, name: str, units: list[str] | Recipe, *, dose_unit: str = "rel",
                    ablate: str | dict | None = None, when: str | None = None, sampling: dict | None = None) -> dict:
        """sampling はこのレシピを試すときのサンプリング(seed は持たない)。"""
        from .steer import sampling_values
        if not re.fullmatch(r"[\w.\-]+", name or ""):
            raise GneError("BAD_ARGS", f"レシピ名 {name!r} は英数字・_・-・. だけにしてください")
        r = units if isinstance(units, Recipe) else self.recipe(units=list(units), dose_unit=dose_unit, name=name,
                                                               ablate=ablate, when=when)
        if sampling:
            r.sampling = sampling_values(sampling, seed=False)
        r.name = name
        path = self.store.recipes_dir() / f"{name}.yaml"
        r.save(path)
        return {"name": name, "path": str(path), "recipe": r.to_dict()}

    def recipe_path(self, name_or_path: str | Path) -> Path:
        p = Path(name_or_path)
        if p.suffix in (".yaml", ".yml") and p.exists():
            return p
        q = self.store.recipes_dir() / f"{name_or_path}.yaml"
        if not q.exists():
            raise GneError("FILE_NOT_FOUND", f"レシピ {name_or_path} がありません", f"{q} に recipe_save で保存してください")
        return q

    def chat(self, columns, prompts: list[str], *, system: str | None = None, max_tokens: int = 256,
             on_text=None, on_column=None, when: str | None = None, sampling=None) -> dict:
        from .chat import ChatState, chat_turn
        state = ChatState(columns, system)
        turns = []
        for u in prompts:
            turns.append({"user": u, "replies": chat_turn(self, state, u, max_tokens=max_tokens, on_text=on_text,
                                                          on_column=on_column, when_override=when, sampling=sampling)})
        return {"columns": [{"label": c.label, "recipe": c.recipe.to_dict() if c.recipe else None} for c in columns],
                "system": system, "turns": turns}

    # --- direction(§6.17)-----------------------------------------------------------------------

    def direction(self, *, contrast: str | Path, name: str | None = None, positions: str = "last",
                  select: str | int | None = None, mode: str = "ablate", layers: str | list[int] | None = None,
                  dose: float = 0.5, max_tokens: int = 64, kl_max: float = 0.1, min_distinct: float = 0.8,
                  per_layer: bool = False, judge_model: str | Path | None = None, judge_target: str | None = None,
                  response_tokens: int = 96, random_controls: int = 16) -> dict:
        """対照集合から全層の方向 r_L を作って保存し、select="auto" なら効果の大きい層を選ぶ(数字なら手で指定)。
        random_controls > 0 なら層選びに同じ大きさのランダムな向きの対照を入れ、対照の p95 を超えた層だけを選ぶ(CHANGE-03 A)。
        positions="response" は返事のトークンから作る(A / B の system で response_tokens まで返事を生成して流す)。"""
        from .direction import (DirectionSet, candidate_layers, layer_table, load_contrast, response_directions,
                                save_directions, save_select, select_layer)
        from .evalx import Judge
        from .reverse import contrastive_directions
        if positions not in ("last", "all", "assistant", "response"):
            raise GneError("BAD_ARGS", "positions は last / all / assistant / response です")
        t0 = time.perf_counter()
        cs, val = load_contrast(contrast)
        name = name or cs.name
        if not re.fullmatch(r"[\w.\-]+", name):
            raise GneError("BAD_ARGS", f"名前 {name!r} は英数字・_・-・. だけにしてください")
        if positions == "response":
            cr = response_directions(self, cs, max_tokens=response_tokens, progress=self.progress)
        else:
            from .reverse import side_interventions
            cr = contrastive_directions(self.engine, cs, positions=positions, with_act=False, progress=self.progress,
                                        interventions=side_interventions(self, cs) if cs.has_recipes else None)
        path = self.store.direction_dir(self.meta.model_id) / f"{name}.npz"
        save_directions(path, cr, cs, str(contrast))
        ds = DirectionSet.load(path)
        resid = self.resid_stats()

        def rho(L):
            return float(resid.rho(L)) if resid is not None and L in resid.layers else None
        sel = None
        if select is not None and str(select) != "auto":
            if not str(select).isdigit() or int(select) not in ds.raw:
                raise GneError("BAD_ARGS", f"--select は auto か層の番号です({min(ds.raw)}〜{max(ds.raw)})")
            sel = {"mode": "manual", "selected": int(select)}
        elif select == "auto":
            judge = (Judge(judge_model, judge_target or val.target or cs.system_a or cs.name) if judge_model else None)
            try:
                sel = select_layer(self, ds, cs, val, mode=mode, layers=candidate_layers(self.meta.n_layer, layers),
                                   dose=dose, max_tokens=max_tokens, kl_max=kl_max, min_distinct=min_distinct,
                                   per_layer=per_layer, judge=judge, random_controls=random_controls,
                                   progress=self.progress)
                if sel.get("selected") is not None:
                    chosen = next(r for r in sel["rows"] if r["layer"] == sel["selected"])
                    sel["selected_null_pct"] = chosen.get("null_pct")
            finally:
                if judge is not None:
                    judge.close()
        if sel is not None:
            save_select(path.with_suffix(".select.json"), sel)
        return {"name": name, "path": str(path), "contrast": str(contrast), "contrast_sha": cs.sha, "positions": positions,
                **({"sides": cs.describe()} if (cs.has_recipes or cs.separate_prompts or cs.system_a is None
                                                or cs.system_b is None) else {}),
                "n_prompts": len(cs.user_prompts), "n_rows": cr.n_rows, "layers": layer_table(ds, rho),
                "validation": {"n_prompts": len(val.prompts), "match": val.match, "target": val.target},
                "select": sel, "seconds": time.perf_counter() - t0}

    def ablation_vector(self, direction: str, *, from_layer: int | None = None, per_layer: bool = False):
        """消す方向: from_layer の r(既定は gne direction --select で選んだ層)、per_layer なら層ごとの r_L(dict)。
        保存済みの方向に無い "random-SEED" は、ランダムな単位ベクトル(消去の対照、CHANGE-03 A)。"""
        m = re.fullmatch(r"random-(\d+)", direction or "")
        if m and not per_layer:
            try:
                self.direction_path(direction)
            except GneError:
                from .steer import random_direction
                return random_direction(self.meta.n_embd, int(m.group(1)), 0), {"direction": direction, "control": True}
        ds = self.direction_set(direction)
        if per_layer:
            return ds.raw, {"direction": ds.name, "per_layer": True}
        if from_layer is None:
            sel = ds.path.with_suffix(".select.json")
            chosen = json.loads(sel.read_text()).get("selected") if sel.exists() else None
            if chosen is None:
                raise GneError("BAD_ARGS", f"方向 {ds.name} の層が決まっていません",
                               "--from-layer で指定するか、gne direction --select auto / --layer で選んでください")
            from_layer = int(chosen)
        if from_layer not in ds.raw:
            raise GneError("BAD_ARGS", f"方向 {ds.name} に層 {from_layer} がありません(0〜{max(ds.raw)})")
        return ds.raw[from_layer], {"direction": ds.name, "from_layer": from_layer}

    def ablate(self, direction: str, out: str | Path, *, from_layer: int | None = None, per_layer: bool = False,
               layers: list[int] | None = None, scale: float = 1.0, embed: bool = True, bake: bool = False,
               requant: str = "q8_0", via_f16: bool = False, imatrix: str | None = None,
               llama_quantize: str | None = None, verify: bool = False) -> dict:
        """方向の射影消去を LoRA アダプタ GGUF(既定)か、重みを書き換えた GGUF(bake)に書き出す(§6.17)。
        verify なら、LoRA は V13(r 成分が残る分だけか)、bake は V13 と V14(LoRA と greedy が一致するか)を書き出したファイルで見る。"""
        from .ablate import bake_ablation_gguf, bake_via_f16, export_ablation_lora
        if not 0 < scale <= 1.5:
            raise GneError("BAD_ARGS", "--scale は 0 より大きく 1.5 以下にしてください(1 で完全に消す)")
        r, meta = self.ablation_vector(direction, from_layer=from_layer, per_layer=per_layer)
        t0 = time.perf_counter()
        if not bake:
            rep = export_ablation_lora(self.weights, r, out, layers=layers, scale=scale, embed=embed, meta=meta,
                                       progress=self.progress)
            kind = "lora"
        elif via_f16:
            rep = bake_via_f16(self.weights, r, out, layers=layers, scale=scale, embed=embed, meta=meta,
                               llama_quantize=llama_quantize, imatrix=imatrix, progress=self.progress)
            kind = "bake"
        else:
            rep = bake_ablation_gguf(self.weights, r, out, layers=layers, scale=scale, embed=embed, requant=requant,
                                     meta=meta, progress=self.progress)
            kind = "bake"
        for w_ in rep.get("warnings", []):
            self.warn(w_)
        check = self._check_ablation(r, Path(out), kind, rep) if verify else None
        return {"kind": kind, **meta, **{k: v for k, v in rep.items() if k != "warnings"}, "verify": check,
                "seconds": time.perf_counter() - t0}

    def _check_ablation(self, r, out: Path, kind: str, rep: dict, n_tokens: int = 32) -> dict:
        """書き出したファイルで V13 と(bake なら)V14 を見る。"""
        from .ablate import embed_in_lora, export_ablation_lora, residual_writers, unit
        from .verify import Verifier
        if isinstance(r, dict):
            return {"skipped": "層ごとの方向(--per-layer)の検証は未対応です"}
        v = Verifier(self.engine, self.weights, seed=self.seed)
        ru = unit(r)
        ids = v.prompt_ids[0]
        layers = rep.get("layers")
        full = layers == sorted(residual_writers(self.weights))
        er = self.weights.embed_rows(ids)[1:].astype(np.float64) @ ru
        if kind == "lora":
            res = v._resid_all(self.engine, ids, [(out, 1.0)])
            expect = 0.0 if rep.get("embed") else er
        else:
            e2 = Engine(out, n_gpu_layers=self._engine_kw["n_gpu_layers"], n_ctx=self.engine.n_ctx, n_seq_max=self.engine.n_seq_max)
            try:
                res = v._resid_all(e2, ids)
                lp = out.with_name(out.name + ".v14.lora.gguf")
                export_ablation_lora(self.weights, r, lp, layers=layers, scale=rep["scale"],
                                     embed=rep["embed"] and embed_in_lora(self.weights))
                seqs = v.prompt_ids[:2]

                def run(engine, lora=None):
                    ctx = engine.steer_ctx()
                    engine.set_lora(ctx, lora)
                    try:
                        return [engine.generate_greedy(ctx, [s_], max_tokens=n_tokens)[0] for s_ in seqs]
                    finally:
                        engine.set_lora(ctx, None)
                a, b = run(self.engine, [(lp, 1.0)]), run(e2)
                lp.unlink(missing_ok=True)
            finally:
                e2.close()
            expect = 0.0 if rep.get("embed") else er
        scale = rep["scale"]
        devs = []
        for L, la in sorted(res.items()):
            # 部分消去(scale < 1)なら r 成分は (1 − s) 倍ぶん残るので、比べるのは全部消したときだけ
            devs.append(float((np.abs(la @ ru - expect) / np.linalg.norm(la, axis=1)).max()))
        out_ = {"v13": {"max_dev": max(devs), "pass": bool(max(devs) < 1e-2) if (scale == 1.0 and full) else None,
                        "note": None if (scale == 1.0 and full) else "部分消去・層の範囲指定では r 成分は残るので合否は付けません"}}
        if kind == "bake":
            from .verify import compare_greedy
            cmps = [compare_greedy(x.token_ids, x.top_ids, x.top_lps, y.token_ids, y.top_ids, y.top_lps)
                    for x, y in zip(a, b, strict=True)]
            out_["v14"] = {"n_tokens": n_tokens, "match": [c["first_diff"] is None for c in cmps],
                           "summary": [c["summary"] for c in cmps], "pass": all(c["ok"] for c in cmps),
                           "note": "量子化し直しの誤差で、僅差のトークンから分かれることがあります(F16 なら誤差が小さい)"
                           if not all(c["first_diff"] is None for c in cmps) else None}
        return out_

    # --- eval(§6.14)---------------------------------------------------------------------------

    def eval(self, recipe: Recipe | list[str], *, doses=(0, 0.25, 0.5, 0.75, 1.0), suite: str | Path | None = None,
             when: str | None = None, judge_model: str | Path | None = None, judge_target: str | None = None,
             name: str | None = None, random_controls: int = 0, reactivity: str | Path | None = None,
             head_controls: int = 0, control_pool: str = "before") -> dict:
        """レシピの倍率ごとの副作用(能力・ppl・言語混入)と性格の変化量を測り、<root>/evals/<model_id>/<name>.json に保存する。
        random_controls > 0 なら各倍率で同じ大きさのランダムな対照レシピも測る(行の ctrl・ctrl_novelty_p95 など)。
        reactivity(対照集合の YAML)で反応性(トリガーあり/なしの差)も測り、head_controls 本のランダムなヘッドと比べる。"""
        from .evalx import Judge, evaluate
        recipe = self._as_recipe(recipe)
        if not recipe.units and recipe.ablate is None:
            raise GneError("BAD_RECIPE", "ユニットがありません")
        if (judge_model is None) != (judge_target is None):
            raise GneError("BAD_ARGS", "--judge-model と --judge-target は両方指定してください")
        judge = Judge(judge_model, judge_target) if judge_model else None
        try:
            d = evaluate(self, recipe, doses=doses, suite=suite, when=when, judge=judge,
                         random_controls=random_controls, reactivity=reactivity, head_controls=head_controls,
                         control_pool=control_pool, progress=self.progress)
            d.pop("_baseline", None)
        finally:
            if judge is not None:
                judge.close()
        name = name or recipe.name or "units-" + hashlib.sha256(
            json.dumps([recipe.to_dict()["units"], recipe.to_dict()["ablate"]], sort_keys=True).encode()).hexdigest()[:8]
        if not re.fullmatch(r"[\w.\-]+", name):
            raise GneError("BAD_ARGS", f"名前 {name!r} は英数字・_・-・. だけにしてください")
        if d["when"] != "all":
            name += "@" + re.sub(r"[^\w.\-]+", "_", d["when"])
        path = self.store.eval_dir(self.meta.model_id) / f"{name}.json"
        d = {"name": name, "path": str(path), "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **d}
        path.write_text(json.dumps(d, ensure_ascii=False, indent=1))
        return d

    def eval_compare_ablation(self, direction: str, *, top: int = 20, from_layer: int | None = None,
                              doses=(0, 0.5, 1.0), suite: str | Path | None = None, when: str | None = None,
                              max_tokens: int = 64) -> dict:
        """§6.17: 同じ方向から作った「上位 top ニューロンの消去(gain ×0)」と「方向の消去(ablate)」を同じ表に並べる。
        能力・ppl・混入は eval、消したい振る舞いの残り具合は方向の対照集合の判定(validate.match)で測る。"""
        from .direction import behavior, load_contrast, top_writer_neurons
        from .steer import AblateSpec, Unit
        t0 = time.perf_counter()
        ds = self.direction_set(direction)
        _, meta = self.ablation_vector(direction, from_layer=from_layer)
        neurons = top_writer_neurons(self, ds, top)
        rn = Recipe([Unit(L, n, 0.0, mode="gain", k=0.0) for L, n, _ in neurons], self.meta.model_id,
                    name=f"{ds.name}-neurons{top}")
        rd = Recipe([], self.meta.model_id, name=f"{ds.name}-ablate",
                    ablate=AblateSpec(ds.name, 1.0, "all", meta["from_layer"]))
        ev = {"neurons": self.eval(rn, doses=doses, suite=suite, when=when),
              "direction": self.eval(rd, doses=doses, suite=suite, when=when)}
        src = ds.meta.get("source")
        cs_val = load_contrast(src) if src and Path(src).exists() else None
        if cs_val is None:
            self.warn(f"方向 {ds.name} の対照集合 {src} が見つからないので、振る舞いの残り具合は測りません")
        elif not cs_val[1].match:
            self.warn("対照集合に validate.match が無いので、振る舞いの残り具合(一致率)は測りません")
        base = behavior(self, *cs_val, None, max_tokens=max_tokens) if cs_val else None
        rows = []
        for method, recipe in (("neurons", rn), ("direction", rd)):
            for row in ev[method]["rows"]:
                d = row["dose"]
                b = None
                if cs_val is not None:
                    b = base if d == 0 else behavior(self, *cs_val, recipe.scaled(d), max_tokens=max_tokens,
                                                     base_lp=base["lp"])
                rows.append({"method": method, "dose": d,
                             "rate": None if b is None else b["rate"],
                             "rate_drop": None if b is None or b["rate"] is None else base["rate"] - b["rate"],
                             "kl_b": None if b is None else b["kl_b"],
                             "capability_drop_pt": row["capability_drop_pt"], "ppl_worse_pct": row["ppl_worse_pct"],
                             "mix_increase": row["mix_increase"], "novelty": row["persona"]["novelty"],
                             "shippable": row["shippable"], "sample": None if b is None else b["sample"]})
        out = {"direction": ds.name, "from_layer": meta["from_layer"], "top": top,
               "neurons": [{"layer": L, "neuron": n, "cos": c} for L, n, c in neurons],
               "evals": {k: v["name"] for k, v in ev.items()}, "contrast": src,
               "match": cs_val[1].match if cs_val else None, "base_rate": None if base is None else base["rate"],
               "rows": rows, "seconds": time.perf_counter() - t0}
        path = self.store.eval_dir(self.meta.model_id) / f"{ds.name}-compare-ablation.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        return {**out, "path": str(path)}

    def eval_compare_methods(self, recipe: Recipe | list[str], *, match_effect: float | None = None,
                             score_layer: int | None = None, top_neurons: int = 20, suite: str | Path | None = None,
                             when: str | None = None, contrast: str | Path | None = None, name: str | None = None) -> dict:
        """CHANGE-06 F.3: 同じ方向について、全層射影(ablate)・ヘッドの方向成分の射影(このレシピの proj)・同じヘッドの
        丸ごと倍率(gain、k_h = 1 − α·s_h)・上位ニューロンの消去(gain、k = 1 − α)を、A 側の点数(速い物差し、score_layer)が
        同じ水準(match_effect、省けばこのレシピの水準)になるよう全体の倍率を二分探索で合わせてから、副作用を同じ表に並べる。
        合わせられなかった方式は理由を書いて除外する。"""
        from .causal import _bisect
        from .direction import Scorer, base_layer_of, behavior, drift, score_layers_default, top_writer_neurons
        from .evalx import evaluate
        from .headproj import proj_heads
        from .steer import AblateSpec, Unit
        t0 = time.perf_counter()
        recipe = self._as_recipe(recipe)
        heads = proj_heads(recipe)
        if not heads:
            raise GneError("BAD_RECIPE", "--compare-methods は proj のユニットを持つレシピで使います(比べる基準の組)")
        dname = heads[0].direction
        ds, cs, val = self._direction_contrast(dname, contrast)
        base = base_layer_of(ds)
        if base is None:
            self.direction_score(dname, contrast=contrast)
            ds = self.direction_set(dname)
            base = base_layer_of(ds)
        if score_layer is None:                          # 組のヘッドより後ろの、最初の物差しの層
            sls = score_layers_default(self, ds, base)
            score_layer = next((L for L in sls if L >= max(p.layer for p in heads)), sls[-1])
        S = int(score_layer)
        sc = Scorer(self, ds, cs, layers=[S])
        b0 = sc.run(keep=True)
        base_a = float(b0["A"]["values"][S].mean())

        def mean_a(r: Recipe | None) -> float:
            lora = self.lora_for(r) if r is not None else None
            return float(sc.run(lora, sides="A")["A"]["values"][S].mean())
        target = float(match_effect) if match_effect is not None else mean_a(recipe)
        s_hi = max(abs(p.s) for p in heads)
        neurons = [(L, n, c_) for L, n, c_ in top_writer_neurons(self, ds, 4 * top_neurons) if L <= S][:top_neurons]
        methods = {
            "ablate": (lambda a: Recipe([], self.meta.model_id, name=f"{ds.name}-ablate",
                                        ablate=AblateSpec(ds.name, a, "all", base)), 1.5,
                       f"全層の書き込みから L{base} の r を消す(scale = α)"),
            "proj": (lambda a: recipe.scaled(a), max(1.0, 30.0 / s_hi), "このレシピの proj(s = α · s_h)"),
            "head_gain": (lambda a: recipe.like([Unit(p.layer, None, 0.0, "head", mode="gain", k=1.0 - a * p.s, head=p.head)
                                                 for p in heads], name=f"{ds.name}-headgain"), 2.0,
                          "同じヘッドの出力を丸ごと k_h = 1 − α·s_h 倍"),
            "neurons": (lambda a: Recipe([Unit(L, n, 0.0, mode="gain", k=1.0 - a) for L, n, _ in neurons],
                                         self.meta.model_id, name=f"{ds.name}-neurons{len(neurons)}"), 1.0,
                        f"方向に書く上位 {len(neurons)} ニューロン(L{S} まで)を k = 1 − α 倍(α = 1 で消去)"),
        }
        rows, baseline = [], None
        base_rate = behavior(self, cs, val, None)["rate"] if val.match else None
        for key, (make, hi, desc) in methods.items():
            tm = time.perf_counter()
            if key == "neurons" and not neurons:
                rows.append({"method": key, "desc": desc, "excluded": f"L{S} より前に方向に書くニューロンがありません"})
                continue
            if key == "proj" and match_effect is None:
                alpha, val_a, reached = 1.0, target, True
            else:
                # 方式によっては点数が倍率に単調でない(丸ごと倍率は直交成分も反転する)ので、粗く見てから詰める
                alpha, val_a, reached, _ = _bisect(lambda a, make=make: mean_a(make(a)), target, hi, grid=8)
            if not reached:
                rows.append({"method": key, "desc": desc, "alpha": alpha, "score_a": val_a,
                             "excluded": f"α = 0〜{hi:g} で A 側の点数は最小でも {val_a:.2f}(α = {alpha:.3g})で、目標 {target:.2f} に届きません"})
                continue
            r = make(alpha)
            got = sc.run(self.lora_for(r), keep=True)
            par, orth = drift(b0["B"]["resid"][S], got["B"]["resid"][S], sc.r[S])
            ev = evaluate(self, r, doses=(1.0,), suite=suite, when=when or recipe.when, direction_metrics=False,
                          baseline=baseline)
            baseline = ev.pop("_baseline")
            er = next(x for x in ev["rows"] if x["dose"] == 1.0)
            rate = behavior(self, cs, val, r)["rate"] if val.match else None
            rows.append({"method": key, "desc": desc, "alpha": alpha, "score_a": val_a,
                         "score_b": float(got["B"]["values"][S].mean()), "drift_par_b": par, "drift_orth_b": orth,
                         "capability_drop_pt": er["capability_drop_pt"], "ppl_worse_pct": er["ppl_worse_pct"],
                         "mix_increase": er["mix_increase"], "novelty": er["persona"]["novelty"],
                         "distinct4": er["persona"]["distinct4"], "shippable": er["shippable"],
                         "rate": rate, "rate_drop": None if rate is None or base_rate is None else base_rate - rate,
                         "recipe": r.to_dict(), "seconds": time.perf_counter() - tm})
        out = {"name": name or f"{recipe.name or ds.name}-compare-methods", "direction": ds.name, "score_layer": S,
               "base_layer": base, "target": target, "target_from": "match_effect" if match_effect is not None else "recipe",
               "base_score_a": base_a, "base_score_b": float(b0["B"]["values"][S].mean()), "match": val.match,
               "base_rate": base_rate, "rows": rows, "seconds": time.perf_counter() - t0}
        path = self.store.eval_dir(self.meta.model_id) / f"{out['name']}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        return {**out, "path": str(path)}

    def eval_alloc_curve(self, alloc: str | Path, *, n_heads: list[int] | None = None, suite: str | Path | None = None,
                         when: str | None = None, random_controls: int = 0, name: str | None = None) -> dict:
        """CHANGE-06 F.4: gne heads allocate の各組(ヘッド数ごと)を eval で評価して、ヘッド数の削減曲線を作る。"""
        from .evalx import evaluate
        t0 = time.perf_counter()
        p = Path(alloc)
        if not p.exists():
            p = self.store.heads_dir(self.meta.model_id) / f"{alloc}.alloc.json"
        if not p.exists():
            raise GneError("FILE_NOT_FOUND", f"配分の結果 {alloc} がありません", "gne heads allocate で作ってください")
        al = json.loads(p.read_text(encoding="utf-8"))
        table = {r["n_heads"]: r for r in al["table"] if not r.get("skipped")}
        want = sorted(n_heads or table, reverse=True)
        miss = [n for n in want if n not in table]
        if miss:
            raise GneError("BAD_ARGS", f"配分の結果に {miss} 個の組がありません(あるのは {sorted(table)})",
                           f"gne heads allocate --n-heads {','.join(map(str, want))} で作り直してください")
        rows, baseline = [], None
        for n in want:
            t = table[n]
            r = Recipe.load(t["recipe"])
            ev = evaluate(self, r, doses=(1.0,), suite=suite, when=when, random_controls=random_controls,
                          baseline=baseline, progress=self.progress)
            baseline = ev.pop("_baseline")
            er = next(x for x in ev["rows"] if x["dose"] == 1.0)
            rows.append({"n_heads": n, "units": t["units"], "s_max": t["s_max"], "remain": t["remain"], "a_max": t["a_max"],
                         "alloc_b_drift_orth": t["b_drift_orth"], "recommended": t.get("recommended", False),
                         **{k: er.get(k) for k in ("capability_drop_pt", "ppl_worse_pct", "mix_increase", "dir_drop_a",
                                                   "drift_orth_a", "drift_orth_b", "rate", "rate_drop", "dir_null_pct",
                                                   "rate_null_pct", "shippable", "notes")},
                         "novelty": er["persona"]["novelty"], "recipe": t["recipe"]})
        out = {"name": name or f"{al['name']}-nheads", "alloc": str(p), "direction": al["direction"],
               "score_layer": al["score_layer"], "target": al["target"], "rows": rows, "random_controls": random_controls,
               "seconds": time.perf_counter() - t0}
        path = self.store.eval_dir(self.meta.model_id) / f"{out['name']}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        return {**out, "path": str(path)}

    def behavior_null(self, recipe: Recipe, contrast: str | Path, *, random_controls: int = 16, max_tokens: int = 64,
                      mode: str | None = None, sampling=None, seeds: int = 1, prompts: str | Path | list[str] | None = None) -> dict:
        """CHANGE-03 A: レシピの効果(対照集合の validate.match の一致率の変化)と、同じ大きさのランダムな対照レシピ
        random_controls 本の効果を比べる。null_pct は「効果が対照の何割より大きいか」(0.95 未満なら対照と区別できない)。
        sampling(engine.Sampling)を渡すと、各プロンプトを seeds 本ずつサンプリングで生成する(レシピを詰めたときの条件に
        合わせる)。prompts は検証プロンプトの差し替え(文字列のリスト、またはその YAML。対照集合の YAML なら reactivity の
        with か user_prompts)。"""
        from .direction import behavior_null, load_contrast
        cs, val = load_contrast(contrast)
        ps = prompts if prompts is None or isinstance(prompts, list) else load_prompt_list(prompts)
        d = behavior_null(self, cs, val, recipe, random_controls=random_controls, max_tokens=max_tokens, mode=mode,
                          sampling=sampling, seeds=seeds, prompts=ps)
        return {"contrast": str(contrast), "match": val.match,
                **({"prompts_file": str(prompts)} if prompts is not None and not isinstance(prompts, list) else {}), **d}

    # --- ヘッドの方向成分の射影(CHANGE-06)-------------------------------------------------------------

    def _direction_contrast(self, direction: str, contrast: str | Path | None):
        """方向セットと、測るのに使う対照集合(省けば方向を作った対照集合)。"""
        from .direction import load_contrast
        ds = self.direction_set(direction)
        if contrast is None:
            src = ds.meta.get("source")
            if not src or not Path(src).exists():
                raise GneError("BAD_ARGS", f"方向 {ds.name} を作った対照集合 {src} が見つかりません", "--contrast で渡してください")
            contrast = src
        cs, val = load_contrast(self._contrast_file(contrast))
        return ds, cs, val

    def direction_score(self, name: str, *, contrast: str | Path | None = None, layers: list[int] | None = None,
                        positions: str = "last", units: list[str] | None = None, recipe: str | Path | None = None,
                        side: str = "both", subset: str = "all", set_base: int | None = None,
                        record_base: bool | None = None, max_examples: int | None = None) -> dict:
        """CHANGE-06 B: 速い物差し。層ごとの A / B の点数と cohen_d、基準層(cohen_d 最大、記録する)、後段の層の方向との cos。
        units / recipe を渡すと、その介入での点数の変化と直交成分のずれ。set_base で基準層を手で上書きする。"""
        from .direction import set_base_layer
        from .direction import score as score_fn
        ds, cs, _ = self._direction_contrast(name, contrast)
        if set_base is not None:
            set_base_layer(ds.path, int(set_base), positions=positions)
            ds = self.direction_set(name)
        r = None
        if units or recipe:
            r = self.recipe(units=list(units) if units else None, path=self.recipe_path(recipe) if recipe else None,
                            default_direction=ds.name)
        d = score_fn(self, ds, cs, layers=layers, positions=positions, units=r, side=side, subset=subset,
                     record_base=False if set_base is not None else record_base, progress=self.progress,
                     max_examples=max_examples)
        if set_base is not None:
            d["base_layer"], d["base_recorded"], d["base_manual"] = int(set_base), True, True
        return {"contrast": str(ds.meta.get("source") if contrast is None else contrast), **d}

    def heads_causal(self, direction: str, *, contrast: str | Path | None = None, mode: str = "ablate",
                     layers: str | list[int] | None = "auto", candidates: str | list | None = None,
                     score_layers: list[int] | None = None, primary: int | None = None,
                     base_recipe: str | Path | None = None, base_units: list[str] | None = None, delta_s: float = 3.0,
                     hard: int = 16, hard_subset: str | list[int] | None = None, b_side_max: float | None = None,
                     top_layers: int = 4, positions: str = "last", subset: str = "all", b_examples: int = 16,
                     name: str | None = None, direction_layer: str = "own", max_examples: int | None = None,
                     drift_max: float | None = 0.5) -> dict:
        """CHANGE-06 C: 因果スクリーニング。ablate は層単位(layers="auto" のとき)→ 下がり方の大きい top_layers 層のヘッドを
        1 個ずつ止める。marginal は base_recipe / base_units の組から s を delta_s 上げたときの効きにくい例での効率 η。
        結果は <root>/heads/<model_id>/<name>.causal.jsonl(と .causal.json)。"""
        from .causal import (hard_examples, head_layers, load_causal, prepare, save_causal, save_hard, screen_heads,
                             screen_layers, spearman)
        from .headproj import parse_proj_specs, proj_heads
        t0 = time.perf_counter()
        ds, cs, _ = self._direction_contrast(direction, contrast)
        name = name or ds.name
        if not re.fullmatch(r"[\w.\-]+", name):
            raise GneError("BAD_ARGS", f"名前 {name!r} は英数字・_・-・. だけにしてください")
        st = prepare(self, ds, cs, score_layers=score_layers, primary=primary, positions=positions, subset=subset,
                     b_examples=b_examples, direction_layer=direction_layer, max_examples=max_examples)
        layer_rows, extra = None, {}
        if isinstance(layers, str) and layers not in ("auto", "all"):
            from .cli import parse_layers
            layers = parse_layers(layers)
        if mode == "ablate":
            if layers == "auto" and candidates in (None, "all"):
                layer_rows = screen_layers(self, ds, cs, setup=st, progress=self.progress)
                # 止めると中立(B 側)の残差が自分の大きさの drift_max 倍以上動く層は、方向に関係なくモデルを壊しているので選ばない
                for r in layer_rows:
                    if drift_max is not None and (r.get("drift_orth_b") or 0) > drift_max:
                        r["excluded"] = "breaks"
                pos = sorted((r for r in layer_rows if r["causal_max"] > 0 and not r.get("excluded")), key=lambda r: -r["causal_max"])
                Ls = sorted(r["layer"] for r in pos[:max(1, top_layers)])
            elif layers in ("auto", "all", None):
                Ls = head_layers(self, max(st.score_layers))
            else:
                Ls = list(layers)
            rows = screen_heads(self, ds, cs, layers=Ls, candidates=candidates, mode="ablate", setup=st,
                                b_side_max=b_side_max, progress=self.progress)
            for r in rows:
                if drift_max is not None and (r.get("drift_orth_b") or 0) > drift_max:
                    r["excluded"] = True
        elif mode == "marginal":
            if base_recipe:
                base = proj_heads(Recipe.load(self.recipe_path(base_recipe)))
            elif base_units:
                base = parse_proj_specs(list(base_units), ds.name)
            else:
                raise GneError("BAD_ARGS", "marginal には --base-recipe か --base-unit(いまの組)が要ります")
            if hard_subset is None:
                hr = hard_examples(st, base, n=hard)
                save_hard(self, name, hr)
                h_idx = hr["idx"]
                extra["hard"] = {k: v for k, v in hr.items() if k != "idx"}
            elif isinstance(hard_subset, list):
                h_idx = [int(x) for x in hard_subset]
            else:
                from .causal import load_hard
                h_idx = load_hard(self, st, hard_subset)
            if candidates is None:                      # いまの組 + 止めたときの効き目の上位 30
                try:
                    abl = load_causal(self, name, "ablate")
                except GneError:
                    abl = []
                seen = {(p.layer, p.head) for p in base}
                abl.sort(key=lambda r: -(r.get("causal_max") or 0))
                candidates = [(p.layer, p.head) for p in base] + [(r["layer"], r["head"]) for r in abl
                                                                  if (r["layer"], r["head"]) not in seen][:30]
                if len(candidates) == len(base):
                    self.warn("止めたときの効き目(heads causal --mode ablate)の行が無いので、いまの組のヘッドだけを測ります")
            Ls = None if layers in ("auto", "all", None) else list(layers)
            rows = screen_heads(self, ds, cs, layers=Ls, candidates=candidates, mode="marginal", base_units=base,
                                delta_s=delta_s, hard_subset=h_idx, setup=st, b_side_max=b_side_max, progress=self.progress)
            extra["base_units"] = [p.spec() for p in base]
            try:                                        # V28: ablate と marginal の順位相関
                abl = {(r["layer"], r["head"]): r for r in load_causal(self, name, "ablate")}
                both = [(abl[(r["layer"], r["head"])]["causal"], r["eta"]) for r in rows if (r["layer"], r["head"]) in abl]
                extra["spearman_ablate_marginal"] = spearman([a for a, _ in both], [b for _, b in both]) if both else None
                extra["n_compared"] = len(both)
            except GneError:
                extra["spearman_ablate_marginal"] = None
        else:
            raise GneError("BAD_ARGS", "--mode は ablate か marginal です")
        summary = {"direction": ds.name, "contrast": str(cs.name), "contrast_sha": cs.sha, "mode": mode,
                   "score_layers": st.score_layers, "primary": st.primary, "positions": positions, "subset": subset,
                   "n_a": st.scorer.n("A"), "n_b": len(st.b_idx), "direction_layer": direction_layer,
                   "max_examples": max_examples, "drift_max": drift_max,
                   "base_scores": {str(L): {"A": float(st.base["A"][L].mean()), "B": float(st.base["B"][L].mean())}
                                   for L in st.score_layers},
                   "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
        if layer_rows is not None:
            summary["layer_rows"] = layer_rows
        pj, ps = save_causal(self, name, rows, summary)
        key = (lambda r: -r["eta"]) if mode == "marginal" else (lambda r: -(r.get("causal_max") or 0))
        return {"name": name, "path": str(pj), "summary_path": str(ps), **summary,
                "layers": layer_rows, "rows": sorted(rows, key=key), "n_rows": len(rows),
                "seconds": time.perf_counter() - t0}

    def heads_allocate(self, direction: str, *, contrast: str | Path | None = None, candidates: str | Path | None = None,
                       target: float, score_layer: int | None = None, n_heads=(6, 5, 4), hard: int = 16,
                       hard_subset: str | Path | None = None, proportional_to: str = "eta", from_zero: bool = True,
                       b_side_max: float | None = None, base_recipe: str | Path | None = None, delta_s: float = 3.0,
                       pool: int = 30, s_max: float = 30.0, remain_threshold: float = 0.0, positions: str | None = None,
                       subset: str | None = None, direction_layer: str = "own", name: str | None = None,
                       max_examples: int | None = None) -> dict:
        """CHANGE-06 D: 効率ベースの強さ配分。candidates は heads causal の結果(名前か .causal.jsonl のパス、既定は方向の名前)。"""
        from .causal import allocate, causal_paths, load_causal
        from .headproj import proj_heads
        ds, cs, _ = self._direction_contrast(direction, contrast)
        cname = str(candidates or ds.name)
        rows = load_causal(self, cname)
        sp = causal_paths(self, Path(cname).name.removesuffix(".causal.jsonl"))[1]
        summ = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
        if score_layer is None:
            sl = summ.get("score_layers") or []
            score_layer = int(sl[-1]) if sl else int(summ.get("primary") or 0)
        base = proj_heads(Recipe.load(self.recipe_path(base_recipe))) if base_recipe else None
        return allocate(self, ds, cs, candidates=rows, target=float(target), score_layer=int(score_layer),
                        hard_subset=hard_subset, n_heads=tuple(n_heads), proportional_to=proportional_to,
                        from_zero=from_zero, b_side_max=b_side_max, base_units=base, delta_s=delta_s, hard=hard,
                        pool=pool, s_max=s_max, remain_threshold=remain_threshold, name=name or ds.name,
                        positions=positions or summ.get("positions") or "last", subset=subset or summ.get("subset") or "all",
                        direction_layer=direction_layer, score_layers=summ.get("score_layers"),
                        max_examples=max_examples if max_examples is not None else summ.get("max_examples"),
                        progress=self.progress)

    PIPELINE_DEFAULTS = {"positions": "last", "score_positions": "last", "target": None, "n_heads": [6, 5, 4], "hard": 16,
                         "top_layers": 4, "max_examples": 64, "s_max": 30.0, "direction_layer": "own"}

    def heads_pipeline(self, spec: dict | str | Path, **over) -> dict:
        """対照集合(JSON か YAML、または dict)から、方向づくり → 基準層 → 因果スクリーニング → 強さの配分 を 1 回で回す。
        spec は対照集合と同じ中身(name・user_prompts か側ごとの prompts_a / prompts_b・system_a / system_b か
        recipe_a / recipe_b・validate)に、任意で
        pipeline: {positions, score_positions, target, n_heads, hard, top_layers, max_examples, s_max, direction_layer}。
        over(引数)は pipeline より優先。target を省くと、物差しの最後の層での A 側と B 側の平均の真ん中(「残る」はこれを超える例)。
        止めると中立の入力の残差が大きく動く(直交成分のずれ > 0.5)層とヘッドは、モデルを壊しているだけなので候補から外す。
        方向づくりには質問を全部使い、止めて測る段階(物差し・因果・配分)は max_examples 個に等間隔に間引く。
        結果は <root>/heads/<model_id>/<name>.pipeline.json。"""
        import yaml
        t0 = time.perf_counter()
        if isinstance(spec, str | Path):
            p = Path(spec)
            if not p.exists():
                raise GneError("FILE_NOT_FOUND", f"{p} がありません")
            text = p.read_text(encoding="utf-8")
            spec = json.loads(text) if p.suffix == ".json" else yaml.safe_load(text)
        if not isinstance(spec, dict):
            raise GneError("BAD_ARGS", "spec は対照集合の表(name・user_prompts・system_a / system_b か recipe_a / recipe_b)です")
        spec = dict(spec)
        spec.pop("about", None)                    # テンプレート(configs/pipeline/)の説明。対照集合には入れない
        pipe = {**self.PIPELINE_DEFAULTS, **(spec.pop("pipeline", None) or {}),
                **{k: v for k, v in over.items() if v is not None}}
        unknown = sorted(set(pipe) - set(self.PIPELINE_DEFAULTS) - {"name"})
        if unknown:
            raise GneError("BAD_ARGS", f"pipeline に {', '.join(unknown)} は使えません({', '.join(self.PIPELINE_DEFAULTS)})")
        name = str(pipe.pop("name", None) or spec.get("name") or "")
        if not re.fullmatch(r"[\w.\-]+", name):
            raise GneError("BAD_ARGS", f"名前 {name!r} は英数字・_・-・. だけにしてください(spec の name)")
        spec["name"] = name
        for key in ("user_prompts", "prompts_a", "prompts_b"):
            v = spec.get(key)
            if v is not None and (not isinstance(v, list) or not all(isinstance(u, str) for u in v)):
                raise GneError("BAD_CONTRAST", f"{key} は質問の文字列のリストです")
        path = self.store.contrast_dir() / f"{name}.yaml"
        path.write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8")
        from .direction import load_contrast
        try:
            cs, _ = load_contrast(path)
            if min(len(cs.prompts("A")), len(cs.prompts("B"))) < 4:
                raise GneError("BAD_CONTRAST", "質問は両側とも 4 個以上にしてください(A と B の分かれ方と効き目を測るため)")
        except GneError:
            path.unlink(missing_ok=True)
            raise
        mx = int(pipe["max_examples"]) if pipe["max_examples"] else None
        sp = pipe["score_positions"]
        out: dict = {"name": name, "contrast": str(path), "n_prompts": len(cs.prompts("A")),
                     **({"n_prompts_b": len(cs.prompts("B"))} if cs.separate_prompts else {}), "sides": cs.describe(),
                     "settings": dict(pipe)}
        ts = time.perf_counter()                                   # 1. 方向(全部の質問)
        d = self.direction(contrast=path, name=name, positions=pipe["positions"])
        out["direction"] = {"path": d["path"], "positions": d["positions"], "n_rows": d["n_rows"],
                            "seconds": time.perf_counter() - ts}
        ts = time.perf_counter()                                   # 2. 基準層(cohen_d が最大の層)
        sc = self.direction_score(name, contrast=path, positions=sp, record_base=True, max_examples=mx)
        out["score"] = {"base_layer": sc["base_layer"], "best_layer": sc["best_layer"], "n": sc["n"],
                        "layers": [{k: r[k] for k in ("layer", "cohen_d", "cos_base")} | {"A": r["A"]["mean"], "B": r["B"]["mean"]}
                                   for r in sc["layers"]], "seconds": time.perf_counter() - ts}
        ts = time.perf_counter()                                   # 3. 因果(層ごと → 効く層のヘッドを 1 個ずつ止める)
        c = self.heads_causal(name, contrast=path, mode="ablate", layers="auto", top_layers=int(pipe["top_layers"]),
                              positions=sp, max_examples=mx, direction_layer=pipe["direction_layer"])
        out["causal"] = {"path": c["path"], "score_layers": c["score_layers"], "primary": c["primary"],
                         "base_scores": c["base_scores"], "n_rows": c["n_rows"], "n_a": c["n_a"], "n_b": c["n_b"],
                         "layers": c.get("layers"), "measured_layers": sorted({r["layer"] for r in c["rows"]}),
                         "top": c["rows"][:20], "seconds": time.perf_counter() - ts}
        S = int(c["score_layers"][-1])                             # 4. 配分(最初の比率 → 効率で配り直す → ヘッド数の比較)
        bs = c["base_scores"][str(S)]
        tgt = pipe["target"] if pipe["target"] is not None else 0.5 * (float(bs["A"]) + float(bs["B"]))
        a = self.heads_allocate(name, contrast=path, target=float(tgt), score_layer=S, n_heads=tuple(int(x) for x in pipe["n_heads"]),
                                hard=int(pipe["hard"]), s_max=float(pipe["s_max"]), positions=sp,
                                direction_layer=pipe["direction_layer"], max_examples=mx, remain_threshold=float(tgt))
        out["allocate"] = a
        out["target"], out["target_from"] = float(tgt), "settings" if pipe["target"] is not None else "A と B の平均の真ん中"
        rec = next((r for r in a["table"] if r.get("recommended")), None)
        out["recommended"] = None if rec is None else {"n_heads": rec["n_heads"], "units": rec["units"], "recipe": rec["recipe"]}
        out["seconds"] = time.perf_counter() - t0
        out["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        p = self.store.heads_dir(self.meta.model_id) / f"{name}.pipeline.json"
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        out["path"] = str(p)
        return out

    CANDIDATE_RUBRIC = """# 採点基準(名前を伏せた採点。この基準で全部付ける)

目標の振る舞い(A 側の状態): {target}
返事は最大 {max_tokens} トークンで切れている。切れていること自体は減点しない。
round_01.scores.txt に「番号 P C」を 1 行ずつ書く(例: 17 3 4)。全部の番号を書く。

## P(目標の振る舞いがどれだけ出ているか、1〜5)
1 = まったく出ていない(ふつうのアシスタントの返事)
2 = かすかな気配が 1 か所
3 = はっきり出ているが、弱いか、ふつうの返事と混ざる
4 = ほぼ一貫して出ている
5 = 典型的に、全体に出ている

## C(自然さ、1〜5)
1 = 崩れている(意味不明、意味のない繰り返し、言語が壊れる)
2 = かなり崩れている
3 = 意味は通るが、変な言い回し・脱線・言語の混ざりがある
4 = 流暢で返事として成り立つ。小さな違和感だけ
5 = 完全に自然

- どの返事がどの候補かは分からないようにしてある。
- 日本語の返事は、日本語そのものの自然さも C で見る。
"""

    def heads_candidates(self, alloc: str | Path, *, prompts: list[str] | dict | str | Path | None = None,
                         scales=(0.5, 0.75, 1.0, 1.25), n_heads: list[int] | None = None, max_prompts: int = 24,
                         max_tokens: int = 120, sampling: str | dict | None = "llama_server", seed: int = 0,
                         judge_b: bool = False, contrast: str | Path | None = None, name: str | None = None,
                         with_recipe: str | Path | None = None, rubric: str | Path | None = None) -> dict:
        """配分の比較表の組 × 全体の倍率 で候補を作り、A 側(振る舞いが出ている状態)と B 側(ふつうの状態)で生成する。
        正の倍率は A 側の振る舞いを消す向き(A 側の返事を採点する)、負の倍率は符号を返して強める向き(B 側の返事を採点する)。
        自動の指標(判定の一致率・繰り返し・無介入と同じ返事の割合)を出し、採点する側の返事を、どの候補か伏せて並べた
        ファイルに書く(judge_b なら全部の候補の両側)。採点を書いたら heads_candidates_tally で候補ごとに集計する。
        prompts は調整に使っていない質問(リスト、{"A": [...], "B": [...]}、またはそのファイル)。省くと対照集合の質問
        (validate.prompts があればそれ)で、結果の held_out が False になる。with_recipe は全部の候補(無介入の base も)に
        重ねるレシピ(ニューロンの種火など)。rubric は採点基準の文章かファイル(省くと目標の振る舞いから作る)。"""
        import random

        import yaml

        from .engine import Sampling
        from .evalx import _generate
        from .rank import distinct4
        from .reverse import combine, side_interventions
        from .steer import SAMPLING_PRESETS, sampling_values
        t0 = time.perf_counter()
        ap = Path(alloc)
        if not ap.exists():
            ap = self.store.heads_dir(self.meta.model_id) / f"{alloc}.alloc.json"
        if not ap.exists():
            raise GneError("FILE_NOT_FOUND", f"配分の結果 {alloc} がありません", "gne heads allocate か gne heads pipeline で作ってください")
        al = json.loads(ap.read_text(encoding="utf-8"))
        rows = [r for r in al["table"] if not r.get("skipped") and (n_heads is None or r["n_heads"] in set(n_heads))]
        if not rows:
            raise GneError("BAD_ARGS", f"配分の結果に使える組がありません(n_heads={n_heads})")
        ds, cs, val = self._direction_contrast(al["direction"], contrast)
        out_name = name or f"{al['name']}"
        if not re.fullmatch(r"[\w.\-]+", out_name):
            raise GneError("BAD_ARGS", f"名前 {out_name!r} は英数字・_・-・. だけにしてください")
        # 質問
        val_both = val.explicit and not cs.separate_prompts         # validate.prompts は両側で同じ質問のときだけ B にも使う
        held_out = {"A": prompts is not None or val.explicit, "B": prompts is not None or val_both}
        if isinstance(prompts, str | Path):
            pth = Path(prompts)
            if not pth.exists():
                raise GneError("FILE_NOT_FOUND", f"{pth} がありません")
            text = pth.read_text(encoding="utf-8")
            prompts = json.loads(text) if pth.suffix == ".json" else yaml.safe_load(text)
        if isinstance(prompts, dict):
            pa = prompts.get("A") or prompts.get("prompts_a") or prompts.get("user_prompts")
            pb = prompts.get("B") or prompts.get("prompts_b") or prompts.get("user_prompts")
        elif prompts is not None:
            pa = pb = prompts
        else:
            pa = val.prompts if val.explicit else cs.prompts("A")
            pb = val.prompts if val_both else cs.prompts("B")
        for v in (pa, pb):
            if not isinstance(v, list) or not v or not all(isinstance(x, str) for x in v):
                raise GneError("BAD_ARGS", "prompts は質問の文字列のリスト(または {A: [...], B: [...]})です")

        def thin(v):
            if len(v) <= max_prompts:
                return list(v)
            return [v[int(i)] for i in sorted({int(x) for x in np.linspace(0, len(v) - 1, max_prompts).round()})]
        pa, pb = thin(pa), thin(pb)
        # サンプリング(候補・無介入で同じ乱数: 違いは介入だけ)
        if sampling in (None, "greedy"):
            sp = None
        else:
            if isinstance(sampling, dict):
                sd = dict(sampling)
                seed = sd.pop("seed", seed)
            elif str(sampling) in SAMPLING_PRESETS:
                sd = SAMPLING_PRESETS[str(sampling)]
            else:
                raise GneError("BAD_ARGS", f"サンプリング {sampling!r} はありません(greedy・{'・'.join(SAMPLING_PRESETS)}、または値の dict)")
            sp = Sampling(**sampling_values(sd, seed=False), seed=int(seed) if seed is not None else random.randrange(2 ** 31))
        if not scales or any(float(x) == 0 for x in scales):
            raise GneError("BAD_ARGS", "scales は 0 以外の倍率です(無介入は base として必ず入ります)")
        wr = None
        if with_recipe is not None:
            wr = Recipe.load(self.recipe_path(with_recipe))
        cands = [{"name": "base", "n_heads": 0, "scale": 0.0, "units": [], "recipe": None}]
        for r in rows:
            for sc in scales:
                cands.append({"name": f"n{r['n_heads']}x{float(sc):g}", "n_heads": r["n_heads"], "scale": float(sc),
                              "units": r["units"], "recipe": r["recipe"], "recommended": bool(r.get("recommended"))})
        goals = {("remove" if c["scale"] > 0 else "add") for c in cands[1:]}
        for c in cands:                           # 採点する側: 消す向きは A 側、強める向きは B 側、無介入は使う側の両方
            if judge_b:
                c["judge_sides"] = "AB"
            elif c["name"] == "base":
                c["judge_sides"] = ("A" if "remove" in goals else "") + ("B" if "add" in goals else "")
            else:
                c["judge_sides"] = "A" if c["scale"] > 0 else "B"
            c["goal"] = None if c["name"] == "base" else ("remove" if c["scale"] > 0 else "add")
        ivs = side_interventions(self, cs) if cs.has_recipes else {"A": (None, None), "B": (None, None)}
        rx = re.compile(val.match, re.IGNORECASE) if val.match else None
        conv = {"A": cs.convs("A", pa), "B": cs.convs("B", pb)}
        total, done = len(cands) * (len(pa) + len(pb)), [0]

        def tick(n):
            done[0] += n
            if self.progress:
                self.progress(done[0], total, "candidates")
        replies = {}
        when = wr.when if wr is not None else "all"
        for c in cands:
            r = None if c["recipe"] is None else Recipe.load(c["recipe"]).scaled(c["scale"])
            if wr is not None:                    # 重ねるレシピ + 候補のヘッド(方向の既定は候補のもの)
                r = wr.like(list(wr.units) + (list(r.units) if r is not None else []),
                            **({"default_direction": r.default_direction, "direction_layer": r.direction_layer} if r is not None else {}))
            vecs = self.vectors(r)[0] if r is not None and r.add_units else None
            lora = self.lora_for(r) if r is not None else None
            for side in "AB":
                outs = _generate(self, [(m, max_tokens) for m in conv[side]], *combine(ivs[side], vecs, lora), when, tick,
                                 sampling=sp)
                replies[(c["name"], side)] = outs
            for side in "AB":
                outs = replies[(c["name"], side)]
                base = replies[("base", side)]
                c[f"rate_{side.lower()}"] = (float(np.mean([bool(rx.search(o["text"])) for o in outs])) if rx else None)
                c[f"d4_{side.lower()}"] = float(np.mean([distinct4(o["token_ids"]) for o in outs]))
                c[f"same_as_base_{side.lower()}"] = float(np.mean([o["text"] == b["text"] for o, b in zip(outs, base, strict=True)]))
        # 名前を伏せた採点のファイル
        d = self.store.heads_dir(self.meta.model_id) / f"{out_name}.cand"
        d.mkdir(parents=True, exist_ok=True)
        old_sc = d / "round_01.scores.txt"
        if old_sc.exists():                       # 前の候補の採点は番号が合わなくなるので、名前を変えて残す
            old_sc.rename(d / f"round_01.scores.{time.strftime('%Y%m%d-%H%M%S')}.old.txt")
        items = [{"cand": c["name"], "side": side, "prompt": q, "text": o["text"]}
                 for c in cands for side in c["judge_sides"]
                 for q, o in zip(pa if side == "A" else pb, replies[(c["name"], side)], strict=True)]
        order = list(range(len(items)))
        random.Random(out_name).shuffle(order)
        lines, key = [], []
        for k, i in enumerate(order, 1):
            body = items[i]["text"].replace("<think>", "").replace("</think>", "").strip()
            lines.append(f"#{k}\n[USER] {items[i]['prompt']}\n[REPLY] {body}\n")
            key.append({**items[i], "id": k})
        (d / "round_01.txt").write_text("\n".join(lines), encoding="utf-8")
        (d / "round_01.key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
        target = val.target or (f"system「{cs.system_a}」で頼んだ振る舞い" if cs.system_a else
                                f"対照集合 {cs.name} の A 側の振る舞い")
        rub = d / "RUBRIC.md"
        if rubric is not None:
            rp_ = Path(str(rubric))
            rub_text = rp_.read_text(encoding="utf-8") if len(str(rubric)) < 1024 and rp_.is_file() else str(rubric)
        else:
            rub_text = self.CANDIDATE_RUBRIC.format(target=target, max_tokens=max_tokens)
        rub.write_text(rub_text, encoding="utf-8")
        out = {"name": out_name, "alloc": str(ap), "direction": ds.name, "contrast": cs.name, "held_out": held_out,
               "with_recipe": None if wr is None else str(self.recipe_path(with_recipe)), "when": when,
               "sampling": None if sp is None else sp.to_dict(), "max_tokens": max_tokens, "scales": [float(x) for x in scales],
               "prompts": {"A": pa, "B": pb}, "match": val.match, "target": target, "candidates": cands,
               "replies": {f"{n}|{side}": [o["text"] for o in v] for (n, side), v in replies.items()},
               "judge": {"dir": str(d), "round": str(d / "round_01.txt"), "rubric": str(rub),
                         "scores": str(d / "round_01.scores.txt"), "n": len(items)},
               "scores": None, "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": time.perf_counter() - t0}
        p = self.store.heads_dir(self.meta.model_id) / f"{out_name}.candidates.json"
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        out["path"] = str(p)
        return out

    def candidates_path(self, name: str | Path) -> Path:
        p = Path(name)
        if not p.exists():
            if not re.fullmatch(r"[\w.\-]+", str(name)):
                raise GneError("BAD_ARGS", f"候補の名前 {name!r} が読めません")
            p = self.store.heads_dir(self.meta.model_id) / f"{name}.candidates.json"
        if not p.exists():
            raise GneError("FILE_NOT_FOUND", f"候補の結果 {name} がありません", "gne heads candidates で作ってください")
        return p

    def candidates_list(self) -> list[dict]:
        """保存した候補の結果の一覧(新しい順)。"""
        out = []
        for p in sorted(self.store.heads_dir(self.meta.model_id).glob("*.candidates.json"), key=lambda x: -x.stat().st_mtime):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({"name": d.get("name", p.name[:-len(".candidates.json")]), "path": str(p), "direction": d.get("direction"),
                        "n_candidates": len(d.get("candidates") or []), "n_judge": (d.get("judge") or {}).get("n"),
                        "scored": bool(d.get("scores")), "updated": d.get("updated")})
        return out

    def candidates_blind(self, name: str | Path) -> dict:
        """名前を伏せた採点の項目(番号・質問・返事)と、いま書いてある点。どの候補かは返さない。"""
        out = json.loads(self.candidates_path(name).read_text(encoding="utf-8"))
        d = Path(out["judge"]["dir"])
        key = json.loads((d / "round_01.key.json").read_text(encoding="utf-8"))
        sc = self._read_scores(d / "round_01.scores.txt")
        rub = d / "RUBRIC.md"
        return {"name": out["name"], "rubric": rub.read_text(encoding="utf-8") if rub.exists() else "",
                "items": [{"id": k["id"], "prompt": k["prompt"], "text": k["text"]} for k in key],
                "scores": {str(k): list(v) for k, v in sc.items()}}

    @staticmethod
    def _read_scores(path: Path) -> dict[int, tuple[int, int]]:
        sc: dict[int, tuple[int, int]] = {}
        if path.exists():
            for ln in path.read_text(encoding="utf-8").splitlines():
                parts = ln.replace(",", " ").split()
                if len(parts) >= 3 and all(x.lstrip("#").isdigit() for x in parts[:3]):
                    sc[int(parts[0].lstrip("#"))] = (int(parts[1]), int(parts[2]))
        return sc

    def heads_candidates_tally(self, name: str | Path, *, scores: str | dict | None = None) -> dict:
        """採点(round_01.scores.txt の「番号 P C」)を候補・側ごとに集計して、候補の結果に書き足す。
        scores(「番号 P C」の行の文字列か {番号: [P, C]})を渡すと、まず採点のファイルにそれを書く(前の点に上書きで足す)。"""
        p = self.candidates_path(name)
        out = json.loads(p.read_text(encoding="utf-8"))
        d = Path(out["judge"]["dir"])
        sp, kp = d / "round_01.scores.txt", d / "round_01.key.json"
        if scores is not None:
            cur = self._read_scores(sp)
            if isinstance(scores, str):
                tmp = d / ".scores.tmp"
                tmp.write_text(scores, encoding="utf-8")
                new = self._read_scores(tmp)
                tmp.unlink()
            else:
                new = {int(k): (int(v[0]), int(v[1])) for k, v in scores.items()}
            n_items = len(json.loads(kp.read_text(encoding="utf-8")))
            bad = [k for k, (a, b) in new.items() if not (1 <= k <= n_items and 1 <= a <= 5 and 1 <= b <= 5)]
            if bad:
                raise GneError("BAD_ARGS", f"番号は 1〜{n_items}、P と C は 1〜5 です(合わない番号: {bad[:5]})")
            cur.update(new)
            sp.write_text("".join(f"{k} {a} {b}\n" for k, (a, b) in sorted(cur.items())), encoding="utf-8")
        if not sp.exists():
            raise GneError("FILE_NOT_FOUND", f"採点のファイル {sp} がありません", f"{d / 'RUBRIC.md'} の基準で付けてください")
        sc = self._read_scores(sp)
        key = json.loads(kp.read_text(encoding="utf-8"))
        missing = [k["id"] for k in key if k["id"] not in sc]
        agg: dict = {}
        for k in key:
            if k["id"] in sc:
                agg.setdefault((k["cand"], k["side"]), []).append(sc[k["id"]])
        scores = {}
        for (cand, side), v in agg.items():
            P = np.array([x[0] for x in v], dtype=float)
            C = np.array([x[1] for x in v], dtype=float)
            scores.setdefault(cand, {})[side] = {"n": len(v), "P": float(P.mean()), "C": float(C.mean()),
                                                  "min_PC": float(np.minimum(P, C).mean()), "P_ge3": float((P >= 3).mean())}
        # 目的の点: 消す向き(A 側)は「振る舞いが消えて自然」= min(6 − P, C)、強める向き(B 側)は min(P, C)
        for cand, by in scores.items():
            for side, v in by.items():
                pc = np.array(agg[(cand, side)], dtype=float)
                v["remove"] = float(np.minimum(6 - pc[:, 0], pc[:, 1]).mean())
                v["add"] = float(np.minimum(pc[:, 0], pc[:, 1]).mean())
        out["scores"] = scores
        out["scores_missing"] = missing
        best = {}
        for goal, side in (("remove", "A"), ("add", "B")):
            got = [(c["name"], scores[c["name"]][side][goal]) for c in out["candidates"]
                   if c.get("goal") == goal and side in (scores.get(c["name"]) or {})]
            if got:
                nm, v = max(got, key=lambda x: x[1])
                b = (scores.get("base") or {}).get(side, {}).get(goal)
                best[goal] = {"name": nm, "value": v, "base": b, "side": side}
        out["best"] = best
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        out["path"] = str(p)
        return out

    def headproj_export(self, recipe: Recipe | list[str], out: str | Path, *, bake: bool = False, requant: str = "q8_0",
                        via_f16: bool = False, imatrix: str | None = None, llama_quantize: str | None = None,
                        verify: bool = False, contrast: str | Path | None = None) -> dict:
        """CHANGE-06 E: proj のユニット(と gain)を LoRA(既定)か、書き込み行列だけ量子化し直した GGUF に書き出す
        (gne gain と同じ経路)。verify なら焼き込み版と LoRA 版を V26 の 4 項目で比べる。"""
        recipe = self._as_recipe(recipe)
        if not recipe.proj_units and not recipe.gain_units:
            raise GneError("BAD_RECIPE", "proj のユニットがありません(例 --unit 33:H16:s6@NAME)")
        d = self.gain(recipe, out, bake=bake, requant="q8_0" if via_f16 else requant, via_f16=via_f16,
                      imatrix=imatrix, llama_quantize=llama_quantize)
        if verify:
            if not bake:
                raise GneError("BAD_ARGS", "--verify は --bake と一緒に使います(焼き込み版と LoRA 版を比べる)")
            d["verify"] = self.check_proj_bake(recipe, Path(d["path"]), contrast=contrast)
        return d

    def check_head_proj(self, L: int, h: int, direction: str, *, ss=(1.0, 2.0, 6.0)) -> dict:
        """V24・V25(射影の正しさ・直交成分が変わらないこと)。"""
        from .headproj import check_head_proj
        return check_head_proj(self, L, h, direction, ss=ss)

    def check_proj_bake(self, recipe: Recipe, baked: Path, *, contrast: str | Path | None = None,
                        suite: str | Path | None = None) -> dict:
        """V26: 焼き込み版と LoRA 版で (a) A・B の点数、(b) 最初のトークンの分布の KL、(c) 最尤トークンの一致率、
        (d) perplexity を比べる。LoRA 版を測ってからモデルを閉じ、焼き込み版を読む(大きいモデルでも 1 つずつ)。"""
        from .direction import Scorer, base_layer_of, last_logprobs
        from .evalx import _ppl, load_suite
        from .headproj import proj_heads
        heads = proj_heads(recipe)
        dname = heads[0].direction if heads else recipe.default_direction
        ds, cs, _ = self._direction_contrast(dname, contrast)
        b = base_layer_of(ds)
        Ls = sorted({x for x in (b, max(L for L in ds.raw if L <= self.meta.n_layer - 2)) if x is not None})
        convs = cs.convs("A") + cs.convs("B")
        texts = load_suite(suite).ppl

        def measure(sess, lora):
            sc = Scorer(sess, ds, cs, layers=Ls)
            got = sc.run(lora)
            lp = last_logprobs(sess, convs, None, lora)
            ppl = _ppl(sess, texts, None, lora, lambda n: None) if texts else {}
            return got, lp, ppl
        lora = self.lora_for(recipe)
        a, lpa, ppa = measure(self, lora)
        self.close()
        other = Session(baked, root=self.store.root, n_gpu_layers=self._engine_kw["n_gpu_layers"],
                        n_ctx=self._engine_kw["n_ctx"], n_seq_max=self._engine_kw["n_seq_max"])
        try:
            bb, lpb, ppb = measure(other, None)
        finally:
            other.close()
        scores = {}
        ok_s = True
        for L in Ls:
            for side in "AB":
                va, vb = a[side]["values"][L], bb[side]["values"][L]
                sd = float(np.std(np.concatenate([a["A"]["values"][L], a["B"]["values"][L]])))
                diff = float(np.abs(va - vb).max())
                res = 0.1 * sd                                    # 物差しの分解能: 両側の点数の広がりの 1 割
                scores[f"{L}{side}"] = {"lora": float(va.mean()), "baked": float(vb.mean()), "max_abs_diff": diff,
                                        "resolution": res, "ok": diff <= res}
                ok_s &= diff <= res
        kl = float(np.mean(np.sum(np.exp(lpa) * (lpa - lpb), axis=1)))
        ta, tb = np.argmax(lpa, axis=1), np.argmax(lpb, axis=1)
        top1 = float(np.mean(ta == tb))
        # 最尤トークンが分かれた例は、LoRA 版での 2 つの差(nats)。0.5 未満なら僅差(V9・V14 と同じ基準)
        gaps = [float(lpa[i, ta[i]] - lpa[i, tb[i]]) for i in range(len(ta)) if ta[i] != tb[i]]
        pm = lambda d: float(np.exp(np.mean(np.log(list(d.values()))))) if d else None  # noqa: E731
        pa, pb = pm(ppa), pm(ppb)
        ppl_diff = None if pa is None else (pb / pa - 1) * 100
        return {"scores": scores, "kl_first_token": kl, "top1_match": top1, "top1_mismatch_gaps": gaps,
                "n_prompts": len(convs), "ppl_lora": pa, "ppl_baked": pb, "ppl_diff_pct": ppl_diff,
                "pass": bool(ok_s and all(g < 0.5 for g in gaps) and (ppl_diff is None or abs(ppl_diff) < 0.5)),
                "note": None if not gaps else "最尤トークンが分かれた例はどれも僅差(0.5 nats 未満)なら合格。"
                                             "量子化し直しの丸めで分かれる(--requant f16 なら誤差が小さい)"}

    def evals(self) -> list[dict]:
        """保存済みの eval の一覧(新しい順)。"""
        out = []
        for p in sorted(self.store.eval_dir(self.meta.model_id).glob("*.json"), key=lambda p: -p.stat().st_mtime):
            try:
                d = json.loads(p.read_text())
            except ValueError:
                continue
            if "recipe" not in d or "suite" not in d:        # 方式の比較・ヘッド数の曲線・消去の比較(CHANGE-06 F.3・F.4、§6.17)
                kind = ("compare-methods" if "target_from" in d else "nheads" if "alloc" in d
                        else "compare-ablation" if "neurons" in d else "other")
                out.append({"name": d.get("name", p.stem), "kind": kind, "direction": d.get("direction"),
                            "created": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(p.stat().st_mtime))})
                continue
            last = d["rows"][-1] if d.get("rows") else {}
            out.append({"name": d.get("name", p.stem), "kind": "eval", "created": d.get("created"), "when": d.get("when"),
                        "doses": d.get("doses"), "units": [_unit_spec(u) for u in d["recipe"]["units"]]
                        + ([_ablate_label(d["recipe"]["ablate"])] if d["recipe"].get("ablate") else []),
                        "shippable": [r["dose"] for r in d.get("rows", []) if r.get("shippable")],
                        "max_dose_drop_pt": last.get("capability_drop_pt")})
        return out

    def eval_load(self, name: str) -> dict:
        if not re.fullmatch(r"[\w.\-@]+", name or ""):
            raise GneError("BAD_ARGS", f"名前 {name!r} が不正です")
        p = self.store.eval_dir(self.meta.model_id) / f"{name}.json"
        if not p.exists():
            raise GneError("FILE_NOT_FOUND", f"eval {name} がありません")
        return json.loads(p.read_text())

    def _verify_candidates(self, cands, name: str, design_name: str, n_controls: int, max_tokens: int) -> dict:
        """候補を層ごとにまとめ、quick 設計の rollout(対照つき)を回す。順位づけは Phase 3 の rank。"""
        from .sweep import load_design, read_rollouts, run_sweep
        design = load_design(design_name)
        by_layer: dict[int, list] = {}
        for c in cands:
            by_layer.setdefault(c.layer, []).append(c)
        sv = self._need_rho(sorted(by_layer))
        runs, texts = [], {}
        for L, cs in sorted(by_layer.items()):
            out = self.store.runs_dir(self.meta.model_id) / f"reverse-{name}-L{L}.jsonl"
            r = run_sweep(self.engine, self.weights, sv, L=L, neurons=[c.neuron for c in cs], design=design, out=out,
                          model_id=self.meta.model_id, random_controls=n_controls, max_tokens=max_tokens,
                          seed=self.seed, signs={c.neuron: c.sign for c in cs}, progress=self.progress)
            runs.append(r)
            for rec in read_rollouts(out):
                if rec["kind"] in ("neuron", "baseline") and rec["prompt"] == design.pairs[0][0]:
                    texts[(L, rec["neuron"])] = rec["text"]
        first = design.pairs[0][0]
        examples = [{"layer": c.layer, "neuron": c.neuron, "sign": c.sign, "prompt": first,
                     "text": texts.get((c.layer, c.neuron), "")} for c in cands]
        baselines = {f"L{L}": texts.get((L, None), "") for L in sorted(by_layer)}
        return {"design": design_name, "runs": runs, "examples": examples, "baselines": baselines}

def load_prompt_list(path: str | Path) -> list[str]:
    """プロンプトの組の YAML: 文字列のリスト、{prompts: [...]}、または対照集合(reactivity の with、無ければ user_prompts)。"""
    import yaml
    p = Path(path)
    if not p.exists():
        raise GneError("FILE_NOT_FOUND", f"{p} がありません")
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    if isinstance(d, dict):
        if d.get("reactivity"):
            d = [x["with"] for x in d["reactivity"] if x.get("with")]
        else:
            d = d.get("prompts") or d.get("user_prompts")
    if not isinstance(d, list) or not d or not all(isinstance(x, str) for x in d):
        raise GneError("BAD_ARGS", f"{p}: プロンプトの組は文字列のリスト(または対照集合の YAML)です")
    return d


def vector_key_of(path: Path) -> str:
    """ファイルの中身で決まる短い鍵(方向を作り直したら query の LoRA も作り直す)。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def _unit_spec(u: dict) -> str:
    """to_dict() のユニット → L:N:+dose / L:N:xk / L:Hh:xk(query は L:Hh:query:方向:×scale、表示用)。"""
    if u.get("mode") == "query":
        return f"{u['layer']}:H{u['head']}:query:{u.get('direction')}:x{u['k']:g}"
    if u.get("mode") == "proj":                                  # CHANGE-06
        return f"{u['layer']}:H{u['head']}:s{u['k']:g}" + (f"@{u['direction']}" if u.get("direction") else "")
    tail = f"x{u['k']:g}" if u.get("mode") == "gain" else f"{u['dose']:+g}"
    mid = (f"random:{u['seed']}" if u.get("kind") == "random" else f"H{u['head']}" if u.get("kind") == "head"
           else f"dir:{u.get('direction')}" if u.get("kind") == "direction" else str(u["neuron"]))
    return f"{u['layer']}:{mid}:{tail}"


def _ablate_label(a: dict) -> str:
    from .steer import AblateSpec
    return AblateSpec.from_dict(a).label()


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


def _summ(x: np.ndarray) -> dict:
    if not x.size:
        return {"n": 0, "p50": None, "p95": None, "max": None}
    return {"n": int(x.size), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def _merge_logit(old, new):
    """同じ層の logit lens を、ニューロン番号の和集合でまとめる(新しい方を優先)。"""
    from .lens import LogitLens
    if old.up_ids.shape[1] != new.up_ids.shape[1]:
        return new
    keep = ~np.isin(old.neurons, new.neurons)
    cat = lambda a, b: np.concatenate([a[keep], b])  # noqa: E731
    nn = cat(old.neurons, new.neurons)
    o = np.argsort(nn)
    return LogitLens(new.layer, nn[o], cat(old.up_ids, new.up_ids)[o], cat(old.up_z, new.up_z)[o],
                     cat(old.down_ids, new.down_ids)[o], cat(old.down_z, new.down_z)[o])


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
