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
               when: str | None = None, ablate: str | dict | None = None) -> Recipe:
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

    def gain(self, recipe: Recipe | list[str], out: str | Path, *, bake: bool = False, requant: str = "auto") -> dict:
        """gain ユニットを LoRA アダプタ GGUF(既定)か、W_up(ヘッドは W_O)を書き換えた GGUF(bake)に書き出す(§6.16、CHANGE-04 D)。"""
        from .gain import bake_gain_gguf, export_gain_lora
        recipe = self._as_recipe(recipe)
        if recipe.add_units:
            self.warn("add のユニットは gain の書き出しに入りません(gne export-cvec で control vector に書き出せます)")
        routes = [u for u in recipe.units if u.mode == "route"]
        if not recipe.gain_units and not recipe.query_units and not routes:
            raise GneError("BAD_RECIPE", "gain のユニットがありません(例 --unit 14:8949:x2、ヘッドは 21:H7:x0)")
        if not bake and (routes or any(u.kind == "expert" for u in recipe.gain_units)):
            raise GneError("BAD_RECIPE", "専門家全体の gain と route は LoRA にできません(全ランクになる)",
                           "--bake でスケールパッチを当てた GGUF にしてください(CHANGE-05 §D)")
        rep: dict = {"path": None, "bytes": 0}
        if recipe.gain_units or routes:
            if bake:
                rep = bake_gain_gguf(self.weights, recipe, out, requant=requant)
            else:
                rep = export_gain_lora(self.weights, recipe, out)
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
        if recipe.when != "all":
            self.warn(f"レシピの when={recipe.when} は {what} では表せないので、全位置(all)として書き出しました(§6.15)。"
                      "位置別のまま使うなら gne steer / chat か gne serve --openai を使ってください")

    def export_cvec(self, recipe: Recipe | list[str], path: str | Path, *, fmt: str = "cvec", verify: bool = False,
                    verify_prompt: str = "Who are you?", n_tokens: int = 32,
                    llama_completion: str | None = None) -> dict:
        recipe = self._as_recipe(recipe)
        self._warn_when(recipe, "control vector")
        if not recipe.add_units:
            raise GneError("BAD_RECIPE", "add のユニットがありません(gain だけのレシピは control vector になりません)",
                           "gne gain で LoRA アダプタか書き換え GGUF に書き出してください")
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
            cr = contrastive_directions(self.engine, cs, layers=layers, method=method, positions=positions,
                                        progress=self.progress)
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
            cr = contrastive_directions(self.engine, cs, positions=positions, with_act=False, progress=self.progress)
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
            judge = Judge(judge_model, judge_target or val.target or cs.system_a) if judge_model else None
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
             head_controls: int = 0) -> dict:
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
                         progress=self.progress)
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

    def evals(self) -> list[dict]:
        """保存済みの eval の一覧(新しい順)。"""
        out = []
        for p in sorted(self.store.eval_dir(self.meta.model_id).glob("*.json"), key=lambda p: -p.stat().st_mtime):
            try:
                d = json.loads(p.read_text())
            except ValueError:
                continue
            last = d["rows"][-1] if d.get("rows") else {}
            out.append({"name": d.get("name", p.stem), "created": d.get("created"), "when": d.get("when"),
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
