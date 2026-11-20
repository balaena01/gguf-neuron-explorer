"""Python API。設計書 §6.13 (c)。CLI はこの薄い皮。

    from gne.api import open_model
    with open_model("models/Qwen3.5-4B-Q8_0.gguf") as s:
        out = s.steer(["14:8949:+1.0", "13:7956:-1.0"], prompt="Who are you?")

戻り値は --json と同じ形の dict(エンベロープ抜き)。エンベロープは envelope() で作る。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .engine import NO_CVEC, Engine
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
# 焼き込みの確かめ(V26・V33)で、計算方法の違いだけで出る射影の値の誤差を測るときの LoRA の倍率(書き込みはほぼ 0)。同じ LoRA
# ファイルをこの倍率で当てる(keyed の LoRA は α を最大の |α| で割って書くので、いちばん大きいユニットが α = 1e-6 になる)。
# 0 ちょうどにすると llama.cpp が LoRA を外して計算の仕方が LoRA 版と変わるおそれがあるので、ごく小さい値にする(書き込みの
# 効きは 0.6B の最後の層で 1e-4 未満。測りたい誤差は 0.3〜1.9 だった、verification.md F139)。
NOISE_LORA_SCALE = 1e-6


def refine_verdicts(rows: list[dict], *, n_a: int, gap: float, sign: float, threshold: float,
                    b_move_max: float = 0.10, drift_max: float = 0.5) -> list[dict]:
    """残りから追加の合わせたレシピの行に、良くなるか(verdict: better / worse / no_gain と理由 why)を付け、推す順に並べた良くなる行を返す。
    数はどれも元の方向・元の目標で、前 = 土台だけ、後 = 土台 + 足した組。
    - worse: 残りが増える(remain_up)、組の効きが差の 1% より下がる(effect_down)、B 側が A 側へ差の 10% より動く(b_move)、
      B 側の直交成分のずれが 0.5 を超える(drift)。B 側の 2 つは、土台だけのときより悪くなったときだけ(土台がもう超えていることがある)。推さない。
    - better: worse でなく、残りが減るか、組の効きが差の 1% より上がる。それ以外は no_gain。
    - 推す順: 目標に届く行があれば、残りが最少から A 側の 5% 以内 → B ⊥ が最小の 1.25 倍以内 → 足した部品が少ない → 全体が少ない。
      届く行が無ければ、組の効きが最大から差の 5% 以内 → 足した部品が少ない。"""
    tol_e, tol_n = 0.01 * gap, max(1, int(np.ceil(0.05 * n_a)))
    for c in rows:
        toward = sign * float(c.get("b_delta") or 0.0) / gap
        toward0 = sign * float(c.get("b_delta_base") or 0.0) / gap
        # B 側の上限: 合わせると上限を超え、しかも土台だけのときより悪くなる(土台がもう超えていても、足して悪くならなければ足した組のせいではない)
        why = [k for k, bad in (("remain_up", c["remain_after"] > c["remain_before"]),
                                ("effect_down", c["effect"] < c["effect_base"] - tol_e),
                                ("b_move", toward > b_move_max and toward > toward0 + 0.01),
                                ("drift", c["b_drift_orth"] > drift_max and c["b_drift_orth"] > float(c.get("b_drift_orth_base") or 0.0) + 0.01)) if bad]
        good = not why and (c["remain_after"] < c["remain_before"] or c["effect"] > c["effect_base"] + tol_e)
        c.update(verdict="worse" if why else "better" if good else "no_gain", why=why, b_toward=toward)
    ok = [c for c in rows if c["verdict"] == "better"]
    tie = lambda c: (c.get("n_added", 0), c.get("n_heads", 0), rows.index(c))      # noqa: E731
    reach = [c for c in ok if c["a_mean_after"] <= threshold + 1e-6]
    if reach:
        r_min = min(c["remain_after"] for c in reach)
        near = [c for c in reach if c["remain_after"] <= r_min + tol_n]
        d_min = min(c["b_drift_orth"] for c in near)
        first = sorted([c for c in near if c["b_drift_orth"] <= d_min * 1.25 + 1e-4], key=tie)
    elif ok:
        e_max = max(c["effect"] for c in ok)
        first = sorted([c for c in ok if c["effect"] >= e_max - 5 * tol_e], key=tie)
    else:
        return []
    return first + sorted([c for c in ok if c not in first], key=lambda c: (-c["effect"],) + tie(c))


def reply_breaks(rows: dict) -> dict:
    """応答での確かめの行(none・recipe → 側 → 数)から、組を当てると繰り返し・空の応答が側の質問の 1/4(2 問以上)増える側。"""
    out = {}
    for side in "AB":
        x0, x1 = rows["none"][side], rows["recipe"][side]
        more = (x1["loop"] - x0["loop"]) + (x1["empty"] - x0["empty"])
        if x1["n"] and more >= max(2, x1["n"] // 4):
            out[side] = {"loop": x1["loop"], "empty": x1["empty"], "n": x1["n"], "loop_none": x0["loop"], "empty_none": x0["empty"]}
    return out


class Session:
    def __init__(self, path: str | Path, *, root: str | Path | None = None, n_gpu_layers: int = -1,
                 n_ctx: int = 4096, n_seq_max: int = 8, seed: int = 0, verbose: bool = False,
                 progress: Progress | None = None, mmproj: str | Path | None = None):
        self.path = Path(path)
        if not self.path.exists():
            raise GneError("MODEL_NOT_FOUND", f"{self.path} がありません")
        self.weights = Weights(self.path)
        self.meta = self.weights.meta
        # 同じ層を 2 回実行するモデル(sensenova_loop、docs/sensenova_loop.md): 書き換えを両方の回に当てるか(既定)、
        # ユニットの層の回だけに当てるか(GNE_LOOP_EDITS=per_visit。2 回目の層に別の名前を持つ llama.cpp が要る)
        mode = os.environ.get("GNE_LOOP_EDITS", "shared").strip() or "shared"
        if mode not in ("shared", "per_visit"):
            raise GneError("BAD_ARGS", f"GNE_LOOP_EDITS は shared か per_visit です({mode!r})")
        from .loop import set_session_mode
        if self.weights.loop is not None:
            self.weights.per_visit_names = mode == "per_visit"
        set_session_mode(mode if self.weights.loop is not None else None)   # 記録の無いレシピを保存するときの当て方
        self.store = Store(root)
        self.store.save_meta(self.meta)
        self.seed = seed
        self.progress = progress
        self.warnings: list[str] = []
        self._engine_kw = dict(n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, n_seq_max=n_seq_max, verbose=verbose,
                               mmproj=mmproj)       # mmproj: 音声の入力(media.py)。無ければモデルの横の mmproj*.gguf
        self._engine: Engine | None = None
        # v0.9.0: 出力のあとに正規化がある層の射影(proj)・ablate を補正するか(--no-proj-correction で False。レシピの記録が優先)
        self.proj_correction = True
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
        if e.loop is not None:
            e.loop_edits = "per_visit" if self.weights.per_visit_names else "shared"
            if self.weights.per_visit_names and not e.loop_runtime_names():
                e.close()
                raise GneError("ARCH_UNSUPPORTED", "GNE_LOOP_EDITS=per_visit には、2 回目の層に別のテンソル名がある llama.cpp が要ります"
                               "(この llama.cpp は 2 回目の層が 1 回目と同じ名前なので、片方の回だけに LoRA を当てられません)",
                               "GNE_LOOP_EDITS を外すと、書き換えは両方の回に効きます(docs/sensenova_loop.md)")
        tpl = template_override(self.meta.model_id, self.meta.arch)
        if tpl is not None:
            e.chat_template = tpl[1]
            self.warn(f"チャットテンプレートを {tpl[0]} で上書きしました")
        return e

    def _pv_path(self, path: Path) -> Path:
        """作った LoRA を使い回す置き場所。1 回目と 2 回目に別々に当てる形(per_visit)は、同じ中身でも当て方が違うので別の名前
        (<名前>.pv.lora.gguf)にする。両方の回に当てる形とほかのモデルは今までどおり。"""
        if self.weights.loop is None or not self.weights.per_visit_names:
            return path
        return path.with_name(path.name.removesuffix(".lora.gguf") + ".pv.lora.gguf")

    def loop_edits_mode(self) -> str | None:
        """いまの当て方(同じ層を 2 回実行するモデルだけ。shared / per_visit)。ほかのモデルは None。"""
        if self.weights.loop is None:
            return None
        return "per_visit" if self.weights.per_visit_names else "shared"

    @contextlib.contextmanager
    def loop_mode(self, mode: str | None, what: str = "この指定"):
        """同じ層を 2 回実行するモデルで、この中だけ当て方を変える(shared = 両方の回、per_visit = 1 回目と 2 回目を別々)。
        per_visit は、2 回目の層に別の名前がある llama.cpp でだけ(無ければ、理由と代わりの方法を書いて断る)。
        ほかのモデルと mode=None では何もしない。"""
        from .loop import LOOP_EDITS, PER_VISIT_FIX, per_visit_refusal, set_session_mode
        w = self.weights
        if w.loop is None or mode is None:
            yield
            return
        if mode not in LOOP_EDITS:
            raise GneError("BAD_ARGS", f"当て方は shared か per_visit です({mode!r})")
        if mode == "per_visit" and not self.engine.loop_runtime_names():
            raise GneError("ARCH_UNSUPPORTED", per_visit_refusal(what), PER_VISIT_FIX)
        prev = w.per_visit_names
        w.per_visit_names = mode == "per_visit"
        set_session_mode(mode)
        try:
            yield
        finally:
            w.per_visit_names = prev
            set_session_mode("per_visit" if prev else "shared")

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
                "ffn_down_qtype": m.ffn_down_qtype, "file_size": m.file_size,
                "cvec_supported": m.arch not in NO_CVEC, "is_split": self.weights.gguf.is_split, "hc_streams": int(self.weights.field(f"{m.arch}.hyper_connection.count", 0) or 0)}
        llama = None
        if load:
            try:
                e = self.engine
                llama = {"ok": True, "n_layer": e.n_layer, "n_embd": e.n_embd, "n_vocab": e.n_vocab}
                if e.loop is not None:                 # 同じ層を 2 回実行するモデル: この llama.cpp で何ができるか
                    llama["loop"] = {"cvec_inner": e.loop_cvec_reaches(), "per_visit_names": e.loop_runtime_names(),
                                     "edits": e.loop_edits, **e.loop_graph(), "inner_resid": e.loop_inner_resid()}
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
        aw = self.weights.check_arch()
        ds = self.dose_spread()
        if ds and ds.get("warn"):
            aw = [*aw, f"このモデルでは、残差の大きさが少数の次元に偏っています(大きい {ds['top']} 成分を除くと、残差の大きさの中央値は"
                       f"約 1/{ds['median']:.1f})。そのため強さ(rel)1 は、偏りを除いた大きさの約 {ds['median']:.1f} 倍にあたります"
                       "(偏りの少ないモデルでは約 1.3 倍)。弱めの強さから試してください"]
        return {"meta": meta, "arch_warnings": aw, "llama_cpp": llama,
                "bake": {"supported": mode is not None, "mode": mode, "reason": reason, "checked_llama_cpp": checked},
                "stats": {"resid": str(rp) if rp.exists() else None, "act_layers": acts, "dose_spread": ds},
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
               direction_layer: str | None = None, default_key: str | None = None,
               proj_correction: str | None = None) -> Recipe:
        """ablate は方向の射影消去(§6.17)。"NAME[@層][:倍率]" か {direction, scale, layers, from_layer}。
        ablate だけ(ユニットなし)のレシピも作れる。default_key は #KEY を省いた keyed のユニットの鍵(CHANGE-07)。
        keyed のユニットは方向と鍵があるか・層が鍵にあるか・長さ・書ける層か(MTP・最終層は断る)をここで確かめる。"""
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
        if default_key and not r.default_key:                  # CHANGE-07: #KEY を省いた keyed のユニットの鍵
            r.default_key = default_key
        if direction_layer:
            r.direction_layer = direction_layer
        if proj_correction == "unrecorded":                    # 画面で読み込んだ古いレシピ(記録なし)をユニットで送ってくるとき
            r.proj_correction = None
        elif proj_correction:                                  # v0.9.0: 射影の補正(postnorm / none)。画面の切り替え・API から
            r.proj_correction = proj_correction
        elif units and not path:                               # その場で作るレシピは、いまの当て方を記録する(記録の無い扱いにしない)
            r.proj_correction = self.proj_correction_label(None)
        r.validate()
        if r.ablate is not None:
            self.direction_path(r.ablate.direction)              # 方向が無ければここで返す
        if r.model_id and r.model_id != self.meta.model_id:
            self.warn(f"レシピの model_id {r.model_id!r} が読み込んだモデル {self.meta.model_id!r} と違います")
        for u in r.units:
            if not 0 <= u.layer < self.weights.n_main:     # MTP の層は本流で実行されず、LoRA にも入れられない
                raise GneError("BAD_RECIPE", f"層 {u.layer} は範囲外です(0..{self.weights.n_main - 1}"
                               + ("。MTP の層は本流で実行されません)" if self.weights.n_main < self.meta.n_layer else ")"))
            if u.mode in ("keyed", "nw"):                           # CHANGE-07: まとめて _check_keyed で確かめる(nw は nw_plan)
                continue
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
                if u.kind in ("neuron", "ffn"):                     # ニューロン・MLP 全体の射影: dense の MLP の層だけ
                    from .headproj import neuron_layer_ok
                    neuron_layer_ok(self.weights, u.layer, allow_post_norm=self.proj_correction_for(r))
                if u.kind == "attn":                                # 注意の出力全体の射影: ヘッドが残差に書く行列のある層だけ
                    self.weights.head_block(u.layer, 0)
            if u.layer <= 2:
                self.warn(f"層 {u.layer}: 層 0〜2 は残差が小さく rel dose がそのまま通用しないことがあります(R9)")
        if r.keyed_units:
            self._check_keyed(r)
        if r.nw_units:                                              # 計算で作るニューロン: 層の形・鍵・方向・ニューロンの重なり
            from .nwrite import nw_plan
            # 形だけを確かめる(kscale auto・remove の大きさは、ここでは測らない。当てるとき・書き出すときに測る)
            nw_plan(self.weights, r, self._key_of, self.direction_set,
                    lambda L, up: {"block_max_median": 1.0, "m": 1.0, "b_up": 0.0, "uh_q90": 1.0})
        if r.when != "all":                                         # レビュー P3: LoRA は位置で切り替えられない
            lora_kinds = [k for k, us in (("keyed", r.keyed_units), ("gain", r.gain_units), ("proj", r.proj_units),
                                          ("nw", r.nw_units)) if us]
            if lora_kinds:
                self.warn(f"when={r.when} は {'・'.join(lora_kinds)} には効きません(LoRA は生成の最初から最後まで当たります。"
                          "位置で切り替えられるのは add の control vector だけです)")
        return r.scaled(scale) if scale != 1.0 else r

    def _warn_once(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warn(msg)

    def proj_correction_for(self, recipe: Recipe | None) -> bool:
        """このレシピの射影・ablate を補正つきで当てるか(v0.9.0)。セッションで切った(--no-proj-correction)か、レシピの記録が
        none なら補正なし。記録の無い古いレシピは補正つき(proj_record_note が知らせる)。"""
        return bool(self.proj_correction) and not (recipe is not None and recipe.proj_correction == "none")

    def _keyed_corr(self, recipe: Recipe) -> Recipe:
        """keyed の計画に渡すレシピ: セッションで補正を切ったとき(--no-proj-correction)は記録を none にした写し
        (注意の出力のあとに正規化がある層の attn の keyed は、補正なしでは実行しない = v0.8.x・B1 と同じ)。"""
        return recipe if self.proj_correction_for(recipe) or recipe.proj_correction == "none" else recipe.like(recipe.units, proj_correction="none")

    def proj_correction_label(self, recipe: Recipe | None = None) -> str:
        return "postnorm" if self.proj_correction_for(recipe) else "none"

    def proj_record_note(self, recipe: Recipe) -> str | None:
        """記録の無い古いレシピを、出力のあとに正規化がある層の射影・ablate に補正つきで当てるときの知らせ(無ければ None)。
        使うたびに警告に入れる(画面・MCP・API の結果にも出る)。"""
        from .headproj import proj_heads
        from .postnorm import writer_side
        if recipe.proj_correction is not None or not self.proj_correction:
            return None
        Ls = sorted({p.layer for p in proj_heads(recipe) if p.s
                     and (self.weights.arch.layers[p.layer].post_norm_mixer if writer_side(p.kind) == "mixer"
                          else self.weights.arch.layers[p.layer].post_norm_ffn) not in ("", "?")})
        abl = recipe.ablate is not None and any(li.post_norm_mixer not in ("", "?") or li.post_norm_ffn not in ("", "?")
                                                for li in self.weights.arch.layers)
        if not Ls and not abl:
            return None
        what = "・".join(([f"層 {', '.join(map(str, Ls[:8]))}{' …' if len(Ls) > 8 else ''} の射影"] if Ls else []) + (["ablate"] if abl else []))
        who = f"レシピ {recipe.name} には" if recipe.name else "このレシピには"
        return (f"{who}射影の補正の記録がありません(v0.8.x までに作ったレシピ)。{what}は、出力のあとの"
                "正規化を補正して当てます(前の版では補正なし。同じ強さでも効き方が変わり、s=1 では強くなります)。前と同じ効き方にするには、"
                "レシピに proj_correction: none を書くか、--no-proj-correction を付けてください(画面ではレシピの欄の「射影の補正」)")

    def _key_of(self, name: str):
        """鍵の名前("NAME" か比較用の鍵 "NAME~j")→ KeySet(<root>/keys/<model_id>/、CHANGE-07)。"""
        from .keyed import load_key
        return load_key(self, name)

    def _check_keyed(self, r: Recipe) -> None:
        """keyed のユニット(CHANGE-07)の確かめ: 方向と鍵があるか、層が鍵にあるか、鍵の書き込み先と長さ、書ける層か
        (MTP・最終層は断る)、post-norm の一次近似(keyed_plan が警告)。重ね方の注意(レビュー P17): 同じ層の gain は
        鍵の入力 x を変える、同じ方向の ablate は焼き込みの順によっては keyed を消す。"""
        from .keyed import keyed_plan, keyed_units
        ku = keyed_units(r)
        keyed_plan(self.weights, self._keyed_corr(r), self._key_of, self.direction_set, direction_layer=r.direction_layer,
                   warn=self._warn_once)
        same_input = {"ffn": ("neuron",), "attn": ("head",), "shared": ("shared",), "expert": ("expert_neuron",)}
        for k in ku:
            hit = [g.label() for g in r.gain_units if g.layer == k.layer and g.kind in same_input[k.kind]
                   and (k.kind != "expert" or g.expert == k.expert)]
            if hit:
                self._warn_once(f"{k.spec()}: 同じ層の gain({', '.join(hit)})は鍵が読む入力 x を変えるので、κᵀμ = 1 からずれます"
                                "(鍵は無介入で作ったもの)。重ねるなら、gain を当てた状態で key-check を測り直してください")
        if r.ablate is not None:
            same = sorted({k.direction for k in ku if k.direction == r.ablate.direction})
            if same:
                self._warn_once(f"keyed と ablate({r.ablate.label()})が同じ方向 {same[0]} を使っています。実行時は別のアダプタなので"
                                "両方効きますが、keyed を焼き込んだ後に ablate を焼き込むと (I − r̂r̂ᵀ)(W + α r κᵀ) で keyed の"
                                "書き込みが消えます(焼くなら ablate を先に)")

    def _as_recipe(self, recipe: Recipe | list[str]) -> Recipe:
        return recipe if isinstance(recipe, Recipe) else self.recipe(units=list(recipe))

    def vectors(self, recipe: Recipe) -> tuple[dict[int, np.ndarray], list[dict]]:
        """add ユニットの control vector(層ごと)と、全ユニットの詳細。gain だけのレシピなら統計は要らない。
        keyed(CHANGE-07)の行は dose_unit に関係なく、その層の ρ_L(あれば。無ければ計算はせずに注意だけ)と方向の r_L で
        比べる大きさ α‖r_L‖/ρ_L を埋める。"""
        keyed = bool(recipe.keyed_units)
        if not recipe.add_units:
            sv = self._keyed_stats(recipe) if keyed else None
            return {}, unit_details(self.weights, sv, recipe, self.direction_vec if keyed else None)
        sv = self.stats_view(Recipe(recipe.add_units, recipe.model_id, recipe.dose_unit, recipe.name))
        if keyed:
            sv = self._keyed_stats(recipe, sv)
        if recipe.dose_unit == "rel" and sv.resid is not None:
            self._warn_spread(sv.resid, sorted({u.layer for u in recipe.add_units}))
        return (recipe_vectors(self.weights, sv, recipe, self.direction_vec),
                unit_details(self.weights, sv, recipe, self.direction_vec))

    def _warn_spread(self, resid: ResidStats, layers: list[int]) -> None:
        """rel の強さを使う層で、残差の大きさが少数の次元に偏っていれば(偏りの量 ≥ SPREAD_WARN)知らせる(v0.9.0)。"""
        from .stats import SPREAD_WARN
        hit = [(L, x) for L in layers if (x := resid.spread(L)) is not None and x >= SPREAD_WARN]
        if hit:
            x = max(v for _, v in hit)
            self._warn_once(f"層 {', '.join(str(L) for L, _ in hit[:6])}{' …' if len(hit) > 6 else ''} では、残差の大きさが少数の次元に"
                            f"偏っているので、強さ(rel)1 は偏りを除いた残差の大きさの約 {x:.1f} 倍にあたります(偏りの少ないモデルでは"
                            "約 1.3 倍)。弱めの強さ(例 0.1〜0.2)から試してください")

    def dose_spread(self) -> dict | None:
        """残差の大きさの偏り(info・画面用)。統計が無ければ None、v0.9.0 より前の統計なら {"old": True}。"""
        rs = self.resid_stats()
        if rs is None:
            return None
        sm = rs.spread_summary()
        return sm if sm is not None else ({"old": True} if rs.p50_rest is None else None)

    def _keyed_stats(self, recipe: Recipe, sv: StatsView | None = None) -> StatsView:
        """keyed の行に要る ρ_L(残差ノルム統計)。α は raw なので書き込みには要らず、統計が無い層は計算せずに注意だけ出す。"""
        if sv is None:
            sv = StatsView(self.resid_stats())
        elif sv.resid is None:                       # absmax の add と混ぜたとき: 活性統計の読み口はそのまま、ρ だけ足す
            sv.resid = self.resid_stats()
        miss = sorted({u.layer for u in recipe.keyed_units if sv.resid is None or u.layer not in sv.resid.layers})
        if miss:
            self._warn_once(f"keyed の層 {miss} の残差ノルム統計が無いので、比べる大きさ α‖r_L‖/ρ_L は出していません"
                            "(gne stats resid で計算できます)")
        return sv

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
        """方向の名前 → DirectionSet。NAME.seg<i> は、区切りつきで作った方向 NAME の区切り i だけを 1 本の方向として見る版。"""
        from .direction import DirectionSet
        try:
            p = self.direction_path(name)
        except GneError:
            m = re.fullmatch(r"(.+)\.seg(\d+)", str(name))
            if not m:
                raise
            return self.direction_set(m.group(1)).segment(int(m.group(2)))
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
        from .direction import require_length
        require_length(ds, L, f"方向の部品 {L}:dir:{name}")
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
                 when: str = "all", lora: list[tuple[str, float]] | None = None, sampling=None, lora_when: str = "all",
                 first_token=None) -> dict:
        """1 本で生成する(バッチの組み方で結果が変わらないように、F2)。既定は greedy、sampling(engine.Sampling)で
        サンプリング。when は §6.15 の位置別ステアリング、lora は gain(§6.16)の LoRA アダプタ。
        lora_when は LoRA を当てる位置: all(既定)/ prompt(プロンプトを読む間だけ)/ reply(生成するトークンだけ)。
        first_token(文字か #ID、またはそのリスト)を渡すと、応答の最初をそのトークン(の列)に決めて続きを書かせる
        (原因の切り分け用。text・token_ids は決めたトークンを含む)。"""
        from .when import generate_when
        e = self.engine
        r, info = generate_when(e, e.steer_ctx(), messages, vectors, when, max_tokens=max_tokens, lora=lora,
                                sampling=sampling, lora_when=lora_when, first_ids=self.first_ids(first_token))
        return {"text": r.text, "finish": r.finish, "n_gen": r.n_gen, "token_ids": r.token_ids, "when": info}

    def first_ids(self, first_token) -> list[int] | None:
        """応答の最初に決めるトークンの指定 → ID の列。文字(1 トークン以上になってよい)・#ID・それらのリスト。"""
        if first_token is None or first_token == "" or first_token == []:
            return None
        e = self.engine
        out: list[int] = []
        for x in (first_token if isinstance(first_token, (list, tuple)) else [first_token]):
            s = str(x).strip()
            if isinstance(x, int) or (s.startswith("#") and s[1:].isdigit()):
                from .tokendir import resolve_token
                out.append(resolve_token(e, x if isinstance(x, int) else s))
            else:
                ids = e.tokenize(str(x), add_special=False, parse_special=True)
                if not ids:
                    raise GneError("BAD_ARGS", f"応答の最初のトークン {x!r} が空です")
                out += [int(t) for t in ids]
        return out

    # --- steer ----------------------------------------------------------------------------------

    def steer(self, recipe: Recipe | list[str], *, prompt: str | None = None, messages: list[dict] | None = None,
              system: str | None = None, max_tokens: int = 128, compare: bool = True, when: str | None = None,
              sampling=None, audio: list[str | Path] | None = None, lora_when: str = "all", first_token=None) -> dict:
        """sampling(engine.Sampling)を渡すと、無介入と介入を同じ seed で引く(違いは介入だけになる)。
        audio: 音声ファイルをユーザーの発話にする(mmproj が要る)。first_token を渡すと、無介入・介入の両方で応答の最初を
        そのトークンに決めて続きを書かせる(generate の first_token)。"""
        recipe = self._as_recipe(recipe)
        msgs = self.messages(prompt, messages, system, audio)
        vecs, details = self.vectors(recipe)
        base = self.generate(None, msgs, max_tokens=max_tokens, sampling=sampling, first_token=first_token) if compare else None
        lora = self.lora_for(recipe)
        with self.patched(self.patches_for(recipe)):           # 専門家全体の gain・route(作業用コピーで読み直す)
            st = self.generate(vecs, msgs, max_tokens=max_tokens, when=when or recipe.when, lora=lora, sampling=sampling,
                               lora_when=lora_when, first_token=first_token)
        return {"messages": msgs, "recipe": recipe.to_dict(), "units": details, "layer_norms": self._norms(vecs),
                "when": when or recipe.when, "sampling": None if sampling is None else sampling.to_dict(),
                "baseline": base, "steered": st, "differs": None if base is None else base["token_ids"] != st["token_ids"],
                **({"proj_correction": self.proj_correction_label(recipe)} if recipe.proj_units or recipe.ablate else {})}

    # --- export / import ------------------------------------------------------------------------

    def lora_for(self, recipe: Recipe | None) -> list[tuple[str, float]] | None:
        """レシピの gain ユニット(§6.16)と ablate(§6.17)を当てる LoRA([(パス, 倍率)])。どちらも無ければ None。
        作ったものは使い回す。ablate の LoRA は倍率 1 で作り、scale はアダプタの倍率で掛ける(eval の倍率列でも 1 ファイル)。
        keyed(CHANGE-07)も同じで、α/α_ref で作った LoRA を倍率 α_ref で当てる(全体を k 倍しても同じファイル)。"""
        if recipe is None:
            return None
        if self.weights.loop is not None:         # 同じ層を 2 回実行するモデル: レシピの当て方で作る(記録の無いレシピはいまの当て方)
            import dataclasses
            mode = recipe.loop_edits or self.loop_edits_mode()
            with self.loop_mode(mode, f"レシピ {recipe.name or ''}".rstrip()):
                return self._lora_for(dataclasses.replace(recipe, loop_edits=mode))
        return self._lora_for(recipe)

    def _lora_for(self, recipe: Recipe) -> list[tuple[str, float]] | None:
        out = []
        lora_units = [u for u in recipe.gain_units if u.kind != "expert"]   # 専門家全体の gain はパッチ(patches_for)
        if lora_units:
            from .gain import export_gain_lora, lora_cache_path
            gr = Recipe(lora_units, self.meta.model_id, name=recipe.name)
            path = self._pv_path(lora_cache_path(self.store.gain_dir(self.meta.model_id), gr))
            if not path.exists():
                tmp = path.with_suffix(".part")
                export_gain_lora(self.weights, gr, tmp)
                tmp.replace(path)
            out.append((str(path), 1.0))
        note = self.proj_record_note(recipe)                    # 記録の無い古いレシピを補正つきで当てる知らせ(使うたびに)
        if note:
            self._warn_once(note)
            fc = self.proj_forecast_text(recipe, old=True)       # 補正なし・補正つきの効きの見込み(くわしく)
            if fc:
                self._warn_once(fc)
        if recipe.ablate is not None and recipe.ablate.scale != 0:
            out.append((str(self.ablation_lora(recipe.ablate, correction=self.proj_correction_for(recipe))), float(recipe.ablate.scale)))
        if recipe.proj_units and any(u.k for u in recipe.proj_units):     # CHANGE-06: ヘッドの方向成分の射影
            out.append((str(self.proj_lora(recipe)), 1.0))
        if recipe.keyed_units and any(u.k for u in recipe.keyed_units):   # CHANGE-07: 鍵つきの書き込み
            kpath, ref = self.keyed_lora(recipe)
            out.append((str(kpath), float(ref)))
        if recipe.nw_units and any(u.k for u in recipe.nw_units):         # 計算で作るニューロン(ffn_gate・ffn_up・ffn_down の差し替え)
            out.append((str(self.nw_lora(recipe)), 1.0))
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
        corr = self.proj_correction_for(recipe)
        for msg in self.proj_notes(heads, corr, recipe.direction_layer):   # 補正の有無ごとの知らせ(使うたびに)
            self._warn_once(msg)
        if corr and any(abs(p.s) > 2 for p in heads):           # 大きな s: 頭打ちの見込み(正規化のある層だけ、短い文 1 つで)
            fc = self.proj_forecast_text(recipe, old=False)
            if fc:
                self._warn_once(fc)
        b = ProjBuilder(self.weights, self.direction_set, direction_layer=recipe.direction_layer, correction=corr)
        path = self._pv_path(self.store.heads_dir(self.meta.model_id) / f"proj-{proj_cache_key(heads, b)}.lora.gguf")
        if not path.exists():
            tmp = path.with_suffix(".part")
            export_proj_lora(self.weights, recipe.like([u for u in recipe.proj_units if u.k]), tmp, self.direction_set,
                             warn=self.warn, correction=corr)
            tmp.replace(path)
        return path

    def proj_forecast(self, recipe: Recipe, scales=(1.0,)) -> list[dict]:
        """postnorm.projection_forecast(レシピと倍率ごとに覚えておく)。正規化の無いモデル・射影の無いレシピは空。"""
        from .postnorm import projection_forecast, writer_layers
        if not recipe.proj_units or not writer_layers(self.weights):
            return []
        key = (json.dumps(recipe.to_dict(), sort_keys=True, default=str), tuple(float(x) for x in scales))
        cache = self.__dict__.setdefault("_proj_forecast", {})
        if key not in cache:
            try:
                cache[key] = projection_forecast(self, recipe, scales)
            except GneError as ex:                            # 見込みが計算できなくても、当てること自体は止めない
                self._warn_once(f"射影の効きの見込みを計算できませんでした({ex.message})")
                cache[key] = []
        return cache[key]

    def proj_forecast_text(self, recipe: Recipe, *, old: bool) -> str | None:
        from .postnorm import forecast_text
        return forecast_text(self.proj_forecast(recipe), old)

    def proj_notes(self, heads, correction: bool, direction_layer: str = "own") -> list[str]:
        """射影のユニットが出力のあとに正規化がある層に当たるときの知らせ。補正つき: γ と r̂ がほぼ直交する層の副作用の警告と、
        強さが比例しないこと。補正なし: 成分が残ること(v0.8.x と同じ計算、A の知らせ)。"""
        from .headproj import head_direction, postnorm_mixer_warning
        from .postnorm import side_warning, writer_side
        w = self.weights
        pn = [p for p in heads if p.s and (w.arch.layers[p.layer].post_norm_mixer if writer_side(p.kind) == "mixer"
                                            else w.arch.layers[p.layer].post_norm_ffn) not in ("", "?")]
        if not pn:
            return []
        if not correction:
            msg = postnorm_mixer_warning(w, [p for p in heads if writer_side(p.kind) == "mixer"])
            return [msg] if msg else []
        out = []
        items = []
        for p in pn:
            try:
                r, _ = head_direction(self.direction_set(p.direction), p.layer, direction_layer)
            except GneError:
                continue
            items.append((p.layer, writer_side(p.kind), r))
        sw = side_warning(w, items)
        if sw:
            out.append(sw)
        Ls = sorted({p.layer for p in pn})
        if any(abs(p.s) not in (1.0, 2.0) for p in pn):
            out.append(f"層 {', '.join(map(str, Ls[:8]))}{' …' if len(Ls) > 8 else ''} の射影は、出力のあとの正規化を補正して当てます。"
                       "強さ s=1 でその部品の分を取り除き、s=2 でちょうど反対にしますが、ほかの s では効きが s に比例せず、大きな s では"
                       "頭打ちになります(gne headproj check で s ごとの予測と実測を確かめられます)")
        return out

    def keyed_lora(self, recipe: Recipe) -> tuple[Path, float]:
        """レシピの keyed のユニット → 実行時の LoRA(<root>/keys/<model_id>/lora/keyed-<鍵>.lora.gguf に作って使い回す)と
        アダプタの倍率 α_ref(= max|α|)。B は α/α_ref で書くので、全体の強さだけを変えても同じファイル。キャッシュ鍵は
        層・書き込み先・α の比・鍵の中身の sha・書く向き d の生のバイト(方向の中身・post-norm の補正・scale)・direction_layer で
        決まる(鍵や方向を作り直したら別のパス。Engine は同じパスの LoRA を 1 回しか読まないため、レビュー P15)。
        .part に書いてから置き換える。"""
        from .keyed import export_keyed_lora, keyed_cache_key, keyed_lora_path, keyed_plan
        kr = self._keyed_corr(recipe.like([u for u in recipe.keyed_units if u.k]))
        if not kr.units:
            raise GneError("BAD_RECIPE", "keyed のユニットがありません(α が 0 でないもの。例 --unit 19:ffn:keyed:a1@NAME#KEY)")
        rows = keyed_plan(self.weights, kr, self._key_of, self.direction_set, direction_layer=kr.direction_layer,
                          warn=self._warn_once)
        h, ref = keyed_cache_key(self.weights, kr, self._key_of, self.direction_set, direction_layer=kr.direction_layer,
                                 rows=rows)
        path = self._pv_path(keyed_lora_path(self, h))
        if not path.exists():
            tmp = path.with_suffix(".part")
            export_keyed_lora(self.weights, kr, tmp, self._key_of, self.direction_set, alpha_ref=ref,
                              direction_layer=kr.direction_layer)
            tmp.replace(path)
        return path, ref

    def nw_lora(self, recipe: Recipe) -> Path:
        """レシピの nw のユニット → 実行時の LoRA(<root>/keys/<model_id>/lora/nw-<中身のハッシュ>.lora.gguf に作って使い回す)。
        ファイル名は LoRA の中身で決まるので、鍵・方向・強さを変えれば別のファイルになる。"""
        from .ablate import write_lora_tensors
        from .nwrite import cache_hash, nw_meta, nw_plan, nw_tensors
        nr = recipe.like([u for u in recipe.nw_units if u.k])
        if not nr.units:
            raise GneError("BAD_RECIPE", "nw のユニットがありません(α が 0 でないもの。例 --unit 11:N2045:nw:a1@NAME#KEY)")
        rows = nw_plan(self.weights, nr, self._key_of, self.direction_set, self.nw_stats)
        tens = nw_tensors(self.weights, rows)
        d = self.store.key_dir(self.meta.model_id) / "lora"
        d.mkdir(parents=True, exist_ok=True)
        path = self._pv_path(d / f"nw-{cache_hash(tens)}.lora.gguf")
        if not path.exists():
            tmp = path.with_suffix(".part")
            write_lora_tensors(self.weights, tens, tmp, name=f"{self.meta.name} nw",
                               kv={"gne.nw": json.dumps(nw_meta(rows), ensure_ascii=False)})
            tmp.replace(path)
        return path

    def nw_export(self, recipe: Recipe | str | Path, out: str | Path, *, strip_meta: bool = False, name: str | None = None) -> dict:
        """レシピの nw のユニットを LoRA アダプタ GGUF に書き出す(llama.cpp の --lora で当てる)。焼き込みは無い。
        strip_meta(人に渡す版): gne の記録と、元のモデルを特定できる値(識別子・名前の欄の元のモデルの名前とレシピの名前)を書かない。
        name はファイルの名前の欄(general.name。省くと、ふつうは「<元のモデルの名前> nw <レシピの名前>」、strip_meta では "adapter")。"""
        from .nwrite import export_nw_lora
        r = recipe if isinstance(recipe, Recipe) else self.recipe(path=self.recipe_path(recipe))
        others = sorted({u.mode for u in r.units if u.mode != "nw"})
        rep = export_nw_lora(self.weights, r, out, self._key_of, self.direction_set, strip_meta=strip_meta, name=name,
                             stats_of=self.nw_stats)
        if others or r.ablate is not None:
            self.warn(f"このファイルには nw のユニットだけが入ります(レシピのほかの介入 {'・'.join(others + (['ablate'] if r.ablate else []))} は"
                      "入りません。proj・keyed・gain と一緒に 1 つの LoRA にするなら gne headproj export を使います)")
        return {**rep, "recipe": r.name, "warnings": list(self.warnings)}

    def nw_stats(self, layer: int, up_dir=None) -> dict:
        """層の活性・残差の大きさ(中立の文で測る。kscale auto と成分を取り除く形が使う。gne/nwrite.layer_stats)。"""
        from .nwrite import layer_stats
        return layer_stats(self, int(layer), up_dir)

    def nw_neurons(self, layers, **kw) -> dict:
        """各ニューロンのふだんの働きを測り、働きの小さい順の表を返す(gne nw neurons)。"""
        from .nwrite import neuron_usage
        return neuron_usage(self, layers, **kw)

    def nw_verify(self, recipe: Recipe | str | Path, **kw) -> dict:
        """nw の LoRA が式どおりに動くか(gne nw verify)。"""
        from .nwrite import verify
        r = recipe if isinstance(recipe, Recipe) else self.recipe(path=self.recipe_path(recipe))
        return verify(self, r, **kw)

    def nw_check(self, recipe: Recipe | str | Path, contrast: str | Path, **kw) -> dict:
        """位置の種類ごとの、鍵の反応と実際に足した量(gne nw check)。"""
        from .nwrite import check
        r = recipe if isinstance(recipe, Recipe) else self.recipe(path=self.recipe_path(recipe))
        return check(self, r, contrast, **kw)

    def nw_try(self, direction: str, contrast: str | Path, **kw) -> dict:
        """重みに入れる前に試す(gne nw try): 区切りつきの方向を、狙う場面の質問の返事の位置どおりに、層の出力へ直接足して
        返事を作る(fish.try_step)。層・区切りごとの強さを、ニューロンと鍵を作る前に返事で比べる。"""
        from .fish import try_step
        return try_step(self, str(contrast), direction=str(direction), **kw)

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

    def ablation_lora(self, spec, *, correction: bool = True) -> Path:
        """AblateSpec → 倍率 1 の消去 LoRA(<root>/ablate/<model_id>/ に作って使い回す)。correction: 出力のあとに正規化がある
        層の書き込みから normalize(γ⊙r̂) を引くか(v0.9.0)。正規化のあるモデルで補正したものは名前に -pn を付ける(補正なしは
        v0.8.x までと同じ名前・同じ中身)。"""
        from .ablate import ablation_side_warning, export_ablation_lora, layer_list, vector_key
        from .postnorm import writer_layers
        r, meta = self.ablation_vector(spec.direction, from_layer=spec.from_layer)
        Ls = layer_list(spec.layers)
        tag = "" if Ls is None else "-L" + hashlib.sha256(json.dumps(Ls).encode()).hexdigest()[:8]
        if correction and writer_layers(self.weights):
            tag += "-pn"
            from .ablate import _select_layers, unit
            sw = ablation_side_warning(self.weights, _select_layers(self.weights, Ls), {L: unit(r) for L in range(self.meta.n_layer)}, True)
            if sw:
                self._warn_once(sw)
        path = self._pv_path(self.store.ablate_dir(self.meta.model_id) / f"{spec.direction}-{vector_key(r)}{tag}.lora.gguf")
        if not path.exists():
            tmp = path.with_suffix(".part")
            export_ablation_lora(self.weights, r, tmp, layers=Ls, meta=meta, correction=correction)
            tmp.replace(path)
        return path

    def _target_weights(self, into: str | Path):
        """焼き込む先の別の GGUF(同じ構造で型だけ違うもの、例えば量子化していない F16 / BF16)。重みは memmap で読み、
        GPU には載せない。構造(arch・層の数・残差の幅)が違えば断る。"""
        from .weights import Weights
        p = Path(into)
        if not p.is_file():
            raise GneError("FILE_NOT_FOUND", f"焼き込む先 {p} がありません")
        if p.resolve() == Path(self.weights.path).resolve():
            return self.weights
        tw = Weights(p)
        a, b = self.weights.meta, tw.meta
        diff = [k for k in ("arch", "n_layer", "n_embd") if getattr(a, k) != getattr(b, k)]
        if diff:
            raise GneError("BAD_ARGS", f"焼き込む先 {p.name} は構造が違います({', '.join(f'{k}: {getattr(a, k)} ≠ {getattr(b, k)}' for k in diff)})",
                           "同じモデルの別の型(量子化していない F16 / BF16 など)にしてください")
        return tw

    def gain(self, recipe: Recipe | list[str], out: str | Path, *, bake: bool = False, requant: str = "auto",
             via_f16: bool = False, imatrix: str | None = None, llama_quantize: str | None = None,
             strip_meta: bool = False, into: str | Path | None = None, lora_f16: bool = False,
             name: str | None = None) -> dict:
        """gain ユニットを LoRA アダプタ GGUF(既定)か、W_up(ヘッドは W_O)を書き換えた GGUF(bake)に書き出す(§6.16、CHANGE-04 D)。
        ヘッドの方向成分の射影(proj、CHANGE-06)と鍵つきの書き込み(keyed、CHANGE-07)も同じ LoRA / 焼き込みに入れる
        (同じテンソルは A・B を連結。記録は gne.headproj と gne.keyed)。keyed を焼き込むときの requant の既定(auto)は
        テンソルごと: 元が Q8_0 なら Q8_0 のまま、それ以外は F16(gain.auto_qtype)。Q8_0 のままでは rank 1 の小さな編集の 1 割以上が
        刻みに埋もれる(edit_kept < gain.KEYED_KEEP_MIN)テンソルだけ F16(レビュー P13、verification.md F132・F139)。
        量子化し直しで編集が埋もれたら警告する(_warn_requant)。
        strip_meta なら gne の記録(gne.* のメタデータ・レシピ・名前の印・横の .headproj.json)を残さない(人に渡す版)。
        書いたあと、メタデータに手元のパスらしい文字列が残っていないかも確かめる(local_path_keys)。
        LoRA では、strip_meta のとき名前の欄(general.name)を "adapter" にして、元のモデルの名前も入れない。name で名前の欄を
        決められる(LoRA だけ。strip_meta なしでも使える。焼き込みは元のモデルのメタデータをそのまま写すので対象外)。
        into を渡すと、そのモデル(同じ構造の別の型、例えば量子化していない F16 / BF16)の重みに書く。方向はこのモデルで作った
        ものを使い、書き換える行列は焼き込む先のものから作る(keyed の ΔW = α d κᵀ は W に依らないので、このモデルで作る)。
        GPU は使わない。"""
        from .gain import bake_gain_gguf, export_gain_lora
        from .headproj import merge_tensors
        recipe = self._as_recipe(recipe)
        self._no_nw(recipe, bake=bake)                   # nw は焼き込めない(LoRA には入る)
        if name is not None and bake:
            raise GneError("BAD_ARGS", "名前の欄の指定(name)は LoRA の書き出しだけです(焼き込みは元のモデルのメタデータをそのまま写す)")
        if name is not None and not str(name).strip():
            raise GneError("BAD_ARGS", "名前(name)が空です")
        W = self.weights if into is None else self._target_weights(into)
        if recipe.add_units:
            self.warn("add のユニットは gain の書き出しに入りません(gne export-cvec で control vector に書き出せます)")
        routes = [u for u in recipe.units if u.mode == "route"]
        if not recipe.gain_units and not recipe.query_units and not routes and not recipe.proj_units and not recipe.keyed_units \
                and not recipe.nw_units:
            raise GneError("BAD_RECIPE", "gain のユニットがありません(例 --unit 14:8949:x2、ヘッドは 21:H7:x0、"
                                         "方向成分の射影は 33:H16:s6@NAME、鍵つきの書き込みは 19:ffn:keyed:a1@NAME#KEY)")
        if not bake and (routes or any(u.kind == "expert" for u in recipe.gain_units)):
            raise GneError("BAD_RECIPE", "専門家全体の gain と route は LoRA にできません(全ランクになる)",
                           "--bake でスケールパッチを当てた GGUF にしてください(CHANGE-05 §D)")
        rep: dict = {"path": None, "bytes": 0}
        ptens, extra_meta, summary, extra_kv, kb = {}, None, {}, {}, None
        if recipe.proj_units:                            # CHANGE-06: ヘッドの方向成分の射影も同じ LoRA / 焼き込みに入れる
            from .headproj import ProjBuilder, layer_summary, proj_heads, proj_meta
            heads = proj_heads(recipe)
            b = ProjBuilder(W, self.direction_set, direction_layer=recipe.direction_layer, correction=self.proj_correction_for(recipe))
            ptens = b.tensors(heads, self.warn)
            extra_meta = {**proj_meta(heads, b), "proj_correction": self.proj_correction_label(recipe)}
            summary = layer_summary(W, heads, ptens)
            if bake and requant == "patch":
                raise GneError("UNSUPPORTED_QUANT", "方向成分の射影はスケールパッチでは書けません(列方向で、単純な倍率ではない)",
                               "--requant q8_0 / f16、または gne headproj export --bake --via-f16")
        if recipe.keyed_units:                           # CHANGE-07: 鍵つきの書き込みも同じ LoRA / 焼き込みに入れる
            if bake and requant == "patch":
                raise GneError("UNSUPPORTED_QUANT", "鍵つきの書き込み(keyed)はスケールパッチでは書けません(rank 1 の足し算で、単純な倍率ではない)",
                               "--requant auto(既定)/ f16 / q8_0 / keep、または LoRA(--bake なし)で配ってください")
            kb = self._keyed_bundle(recipe, W)
            extra_kv["gne.keyed"] = json.dumps(kb["meta"], ensure_ascii=False)
            summary = {**summary, "keyed_layers": kb["layers"], "keyed_rank": kb["rank"]}
        nwt = {}
        if recipe.nw_units:                              # 計算で作るニューロンも同じ LoRA に入れる(ffn_gate・ffn_up・ffn_down の差し替え)
            from .nwrite import nw_meta, nw_plan, nw_tensors
            if into is not None:
                raise GneError("BAD_ARGS", "nw のユニットは --into と一緒には書き出せません(差し替える元の行・列を、当てる先のモデルから読むため)")
            nrows = nw_plan(W, recipe.like([u for u in recipe.nw_units if u.k]), self._key_of, self.direction_set, self.nw_stats)
            clash = sorted({u.layer for u in recipe.gain_units if u.kind == "neuron"} & {r["layer"] for r in nrows})
            if clash:
                raise GneError("BAD_RECIPE", f"層 {', '.join(map(str, clash))} に、ニューロンの gain と nw の両方があります(どちらも ffn_up を変えるので、"
                                             "1 つの LoRA にできません)", "gne nw export で nw だけを別のアダプタにしてください")
            nwt = nw_tensors(W, nrows)
            extra_kv["gne.nw"] = json.dumps(nw_meta(nrows), ensure_ascii=False)
            summary = {**summary, "nw": nw_meta(nrows)["units"]}
        extra = merge_tensors(ptens, kb["tensors"] if kb else {}, nwt)
        if via_f16 and not bake:
            raise GneError("BAD_ARGS", "--via-f16 は --bake と一緒に使います")
        if strip_meta and (recipe.query_units or (bake and requant in ("patch", "auto") and not extra and not via_f16
                                                  and (routes or any(u.kind == "expert" for u in recipe.gain_units)))):
            raise GneError("BAD_ARGS", "gne の記録を残さない書き出しは、ヘッドの proj・keyed・gain とニューロンの gain だけです"
                                       "(クエリ操作の LoRA と、専門家全体のスケールパッチは未対応)")
        imatrix_auto = False
        if bake and via_f16 and not imatrix:             # 渡されなければ、モデルの隣の <モデル名>.imatrix.gguf などを使う
            from .ablate import find_imatrix
            found = find_imatrix(W.path)
            if found is not None:
                imatrix, imatrix_auto = str(found), True
        if recipe.gain_units or routes or extra:
            if bake and via_f16:                         # F16 で書いて llama-quantize で元の型に戻す(CHANGE-06 E.2)
                from .ablate import requant_via_f16
                rep = requant_via_f16(W, lambda mid: bake_gain_gguf(W, recipe, mid, requant="f16",
                                                                              extra=extra, extra_meta=extra_meta,
                                                                              extra_kv=extra_kv, via_f16=True,
                                                                              strip_meta=strip_meta),
                                      out, llama_quantize=llama_quantize, imatrix=imatrix, neutral_imatrix=strip_meta)
                # 中間の F16 の大きさ(bytes_to・専門家のテンソルの増分)は llama-quantize で元の型に戻るので残さない
                rep["modified"] = [{**{k: v for k, v in m.items() if k != "bytes_to"}, "to": m["from"]}
                                   for m in rep["modified"]]
                rep.pop("expert_bytes_added", None)
            elif bake:
                rep = bake_gain_gguf(W, recipe, out, requant=requant, extra=extra, extra_meta=extra_meta,
                                     extra_kv=extra_kv, strip_meta=strip_meta, keyed=kb is not None)
                if requant == "auto" and kb is not None:
                    rep["requant"] = "auto"
            else:
                rep = export_gain_lora(W, recipe, out, extra=extra, extra_meta=extra_meta, extra_kv=extra_kv,
                                       strip_meta=strip_meta, f16=lora_f16, name=name)
                if ptens:                                # proj だけの階数(keyed と同じテンソルに連結しても proj の分)
                    rep["proj_rank"] = {n.split(".")[1]: int(B.shape[-1]) for n, (_, B) in ptens.items()}
                else:
                    rep.pop("proj_rank", None)
        rep.update(summary)
        if kb is not None and bake:
            self._warn_requant(rep)
            if not via_f16:
                self._warn_expert_size(rep)
        if into is not None:
            rep["into"] = Path(W.path).name if strip_meta else str(W.path)
        if bake and via_f16:
            rep["imatrix"] = None if imatrix is None else (Path(imatrix).name if strip_meta else str(imatrix))
            rep["imatrix_auto"] = imatrix_auto
        if strip_meta and rep.get("path"):              # 人に渡す版: 横の記録を作らず、手元のパスが残っていないかを見る
            from .ablate import local_path_keys
            side = Path(str(rep["path"]) + ".headproj.json")
            side.unlink(missing_ok=True)                 # 前に同じ名前で書いた記録が残っていれば消す
            rep["strip_meta"] = True
            rep["local_path_keys"] = local_path_keys(rep["path"])
            if rep["local_path_keys"]:
                self.warn("メタデータに手元のパスらしい文字列が残っています: " + ", ".join(rep["local_path_keys"])
                          + "(imatrix を使うと llama-quantize が、それを作った文章のパスを quantize.imatrix.dataset に書くことがある)")
        elif bake and extra:                             # CHANGE-06 E.2: 書き換えたテンソルと元の型のサイドカー
            side = Path(str(rep["path"]) + ".headproj.json")
            side.write_text(json.dumps({"recipe": recipe.to_dict(), "headproj": extra_meta,
                                        **({"keyed": kb["meta"]} if kb is not None else {}), "modified": rep.get("modified"),
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
        return {"kind": "bake" if bake else "lora", "recipe": recipe.to_dict(), **rep,
                **({"proj_correction": self.proj_correction_label(recipe)} if recipe.proj_units else {})}

    def _keyed_bundle(self, recipe: Recipe, W=None) -> dict:
        """書き出し用の keyed の LoRA(α そのまま、倍率 1)と記録。W(焼き込む先)は構造が同じことだけ確かめる
        (ΔW = α d κᵀ は W に依らないので、行列はこのモデルで作る)。"""
        from .keyed import keyed_bundle
        kb = keyed_bundle(self.weights, self._keyed_corr(recipe), self._key_of, self.direction_set,
                          direction_layer=recipe.direction_layer, warn=self._warn_once)
        if W is not None and W is not self.weights:
            miss = [n for n in kb["tensors"] if not W.has_tensor(n)]
            if miss:
                raise GneError("BAD_ARGS", f"焼き込む先に {miss[0]} がありません(構造の違うモデル)")
        return kb

    def _warn_requant(self, rep: dict, limit: float = 0.5) -> None:
        """量子化し直しで編集が埋もれたときの警告(keyed、レビュー P13)。量子化し直しても残った編集の割合 edit_kept_min が
        gain.KEYED_KEEP_MIN(0.9)を下回ったら。edit_kept が無い結果(古い記録など)は、量子化し直しで失った分 ÷ 編集の大きさ
        requant_error_max が limit を超えたら。requant_error は編集と無関係な丸めのゆらぎ(その型の量子化そのもののゆらぎ)も
        数えるので、刻みより小さい編集では編集が残っていても大きく出る(verification.md F140)。既定(auto)で Q8_0 から F16 に
        切り替えたテンソル(記録の fallback)も知らせる。"""
        from .gain import KEYED_KEEP_MIN
        err, kept = rep.get("requant_error_max"), rep.get("edit_kept_min")
        if (kept is not None and kept < KEYED_KEEP_MIN) or (kept is None and err is not None and err > limit):
            what = []
            if kept is not None:
                what.append(f"残った編集は {kept * 100:.0f}%(edit_kept_min={kept:.2f})")
            if err is not None:
                what.append(f"丸めのずれは編集の {err * 100:.0f}% 相当(requant_error_max={err:.2f})")
            self.warn("量子化し直しで、残差に書き込む重み行列の編集が丸めに消えています(" + "、".join(what) + ")。"
                      "--requant f16 か auto(既定。Q8_0 で埋もれるテンソルだけ F16)にするか、LoRA で配ってください")
        fb = [m["tensor"] for m in rep.get("modified") or [] if isinstance(m, dict) and m.get("fallback")] \
            or list(rep.get("fallback") or [])                  # 分割 GGUF への焼き込みは modified が名前だけ
        if fb:
            self.warn(f"{len(fb)} テンソル({fb[0]} など)は、Q8_0 のままでは編集の 1 割以上が丸めに消えるので F16 で書きました"
                      "(--requant auto の規則。元の型のままにするなら --requant keep)")

    def _warn_expert_size(self, rep: dict) -> None:
        """専門家の 3 次元テンソル(ffn_down_exps、keyed の expert:e)を元と違う型で書き直してファイルが大きくなったときの
        警告(rep の expert_bytes_added、通常の焼き込みと分割 GGUF への焼き込み)。1 つのテンソルの型は 1 つなので、編集しない
        専門家のスラブも同じ型に変換される(gain._expert_bake)。"""
        tot = int(rep.get("expert_bytes_added") or 0)
        if tot > 0:
            grown = [m["tensor"] for m in rep.get("modified") or []
                     if isinstance(m, dict) and "experts" in m and int(m.get("bytes_to", 0)) > int(m.get("bytes_from", 0))]
            what = f"専門家のテンソル {len(grown)} 本({grown[0]} など)" if grown else "専門家のテンソル"
            self.warn(f"{what}を元と違う型で書き直したので、ファイルが "
                      f"{tot / 2 ** 20:.0f} MiB 大きくなりました(1 つのテンソルの型は 1 つなので、編集しない専門家も変換されます)。"
                      "元の型で書き戻すには --requant keep(編集しない専門家はバイトのまま。通常の焼き込みでは K-quant・IQ の型は"
                      " Q8_0 になります)。配るなら LoRA(--bake なし)が小さく正確です")

    def _no_nw(self, recipe: Recipe, *, bake: bool) -> None:
        """nw のユニットは、この書き出しには入れられない(焼き込みは無い。LoRA は gne nw export)。"""
        if not recipe.nw_units:
            return
        if bake:
            raise GneError("BAD_ARGS", "計算で作るニューロン(nw)は焼き込めません",
                           "Q8_0 の ffn_down は 32 個の重みが 1 つの倍率を共有するので、1 列だけを方向に差し替えると誤差が 9 割を超えます。"
                           "gne nw export で LoRA アダプタにして、llama.cpp の --lora で当ててください")
        # LoRA なら、gne gain / gne headproj export が proj・keyed・gain と同じファイルに入れる(Session.gain)

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
        if recipe.keyed_units:
            self.warn(f"鍵つきの書き込み(keyed、{len(recipe.keyed_units)} 個)は {what} では表せないので外しました"
                      "(入力の活性に鍵が反応するときだけ書くので、定数の足し算にならない、CHANGE-07)。"
                      "gne headproj export(か gne gain)で LoRA に書き出してください")
        if recipe.when != "all":
            self.warn(f"レシピの when={recipe.when} は {what} では表せないので、全位置(all)として書き出しました(§6.15)。"
                      "位置別のまま使うなら gne steer / chat か gne serve --openai を使ってください")

    def export_cvec(self, recipe: Recipe | list[str], path: str | Path, *, fmt: str = "cvec", verify: bool = False,
                    verify_prompt: str = "Who are you?", n_tokens: int = 32,
                    llama_completion: str | None = None) -> dict:
        recipe = self._as_recipe(recipe)
        if fmt == "cvec" and self.meta.arch in NO_CVEC:
            raise GneError("ARCH_UNSUPPORTED", f"{self.meta.arch} のグラフは control vector を読みません(--control-vector を付けても何も起きません)",
                           "gne gain / gne headproj export で LoRA アダプタにしてください")
        self._warn_when(recipe, "control vector")
        if not recipe.add_units:
            raise GneError("BAD_RECIPE", "add のユニットがありません(gain・proj・keyed だけのレシピは control vector になりません)",
                           "gne gain / gne headproj export で LoRA アダプタか書き換え GGUF に書き出してください")
        moe_add = [u.label() for u in recipe.add_units if u.kind in ("expert_neuron", "shared")]
        if moe_add:                                           # CHANGE-05 §C・V22
            self.warn(f"{', '.join(moe_add)}: MoE の add は書き込み方向の押しとして書き出します(専門家が選ばれたかに関係なく全トークンに足す)")
        rest = [u.label() for u in recipe.units if u.mode in ("gain", "route")]
        if rest:
            self.warn(f"{', '.join(rest)}: gain と route は control vector に入りません"
                      "(ニューロン・ヘッド・専門家のニューロンの gain は LoRA、専門家全体の gain と route はスケールパッチ)")
        vecs, details = self.vectors(recipe)
        lp = self.weights.loop
        if fmt == "cvec" and lp is not None and any(lp.inner(L) for L in vecs):
            inner = sorted(L for L in vecs if lp.inner(L))
            self.warn(f"層 {', '.join(map(str, inner[:8]))} は残差が 2 本の流れの中の層です。元の llama.cpp(llama-server・llama-cli など)は"
                      "そこに足した control vector を次の層に渡さないので、2 本の流れの両方に足すように直した llama.cpp で使ってください"
                      "(docs/sensenova_loop.md)")
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
        self._no_nw(recipe, bake=True)
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
        if not 0 <= L < self.weights.n_main:           # MTP の層は本流で実行されない(押しても何も起きない)
            raise GneError("BAD_ARGS", f"層 {L} は範囲外です(0..{self.weights.n_main - 1}"
                           + (f"。層 {self.weights.n_main}〜{self.meta.n_layer - 1} は MTP の層で、本流では実行されません)" if self.weights.n_main < self.meta.n_layer else ")"))
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
        """対照集合でヘッドごとに trigger_mass・pattern_shift・out_shift・dir_cos を測り、中立どうしの差の p95 を基準値にして順位づける。"""
        from .heads import heads_contrast
        return heads_contrast(self, contrast, name=name, direction=direction, system_b2=system_b2, layers=layers,
                              max_tokens=max_tokens, progress=self.progress)

    def experts_contrast(self, contrast: str | Path, *, name: str | None = None, direction: str | None = None,
                         system_b2: str | None = None, layers: list[int] | None = None, max_tokens: int = 48) -> dict:
        """MoE の層の専門家ごとに usage_shift・weight_shift・contrib_cos・contrib_norm を測り、中立どうしの差の p95 を基準値にして順位づける(CHANGE-05 §E.1)。"""
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
                    ablate: str | dict | None = None, when: str | None = None, sampling: dict | None = None,
                    default_direction: str | None = None, direction_layer: str | None = None,
                    default_key: str | None = None, loop_edits: str | None = None,
                    proj_correction: str | None = None) -> dict:
        """sampling はこのレシピを試すときのサンプリング(seed は持たない)。default_direction・direction_layer・default_key は
        @方向・#鍵を省いた proj・keyed のユニットの既定(CHANGE-06・07)。"""
        from .steer import sampling_values
        if not re.fullmatch(r"[\w.\-]+", name or ""):
            raise GneError("BAD_ARGS", f"レシピ名 {name!r} は英数字・_・-・. だけにしてください")
        r = units if isinstance(units, Recipe) else self.recipe(units=list(units), dose_unit=dose_unit, name=name,
                                                               ablate=ablate, when=when, default_direction=default_direction,
                                                               direction_layer=direction_layer, default_key=default_key,
                                                               proj_correction=proj_correction)
        if sampling:
            r.sampling = sampling_values(sampling, seed=False)
        if loop_edits is not None:                 # 同じ層を 2 回実行するモデルの当て方(shared / per_visit)
            if loop_edits not in ("shared", "per_visit"):
                raise GneError("BAD_ARGS", f"loop_edits は shared か per_visit です({loop_edits!r})")
            r.loop_edits = loop_edits
        note = None
        if proj_correction in ("postnorm", "none"):          # v0.9.0: 画面の「射影の補正」
            r.proj_correction = proj_correction
        if r.proj_correction is None:                        # 記録の無い古いレシピ: いま当てている方(補正つき)で記録する
            was = self.proj_record_note(r)
            r.proj_correction = self.proj_correction_label(None)
            if was:
                note = ("記録の無い古いレシピを、いま当てている「補正つき」(proj_correction: postnorm)として保存しました。前の版とは"
                        "効き方が違います。前と同じ効き方で残すには、「射影の補正」を「補正なし」にしてから保存し直してください")
        r.name = name
        path = self.store.recipes_dir() / f"{name}.yaml"
        r.save(path)
        return {"name": name, "path": str(path), "recipe": r.to_dict(), **({"note": note} if note else {})}

    def recipe_path(self, name_or_path: str | Path) -> Path:
        p = Path(name_or_path)
        if p.suffix in (".yaml", ".yml") and p.exists():
            return p
        q = self.store.recipes_dir() / f"{name_or_path}.yaml"
        if not q.exists():
            raise GneError("FILE_NOT_FOUND", f"レシピ {name_or_path} がありません", f"{q} に recipe_save で保存してください")
        return q

    def chat(self, columns, prompts: list[str], *, system: str | None = None, max_tokens: int = 256,
             on_text=None, on_column=None, when: str | None = None, sampling=None, lora_when: str = "all") -> dict:
        from .chat import ChatState, chat_turn
        state = ChatState(columns, system)
        turns = []
        for u in prompts:
            turns.append({"user": u, "replies": chat_turn(self, state, u, max_tokens=max_tokens, on_text=on_text,
                                                          on_column=on_column, when_override=when, sampling=sampling,
                                                          lora_when=lora_when)})
        return {"columns": [{"label": c.label, "recipe": c.recipe.to_dict() if c.recipe else None} for c in columns],
                "system": system, "turns": turns}

    # --- direction(§6.17)-----------------------------------------------------------------------

    def direction(self, *, contrast: str | Path, name: str | None = None, positions: str = "last",
                  select: str | int | None = None, mode: str = "ablate", layers: str | list[int] | None = None,
                  dose: float = 0.5, max_tokens: int = 64, kl_max: float = 0.1, min_distinct: float = 0.8,
                  per_layer: bool = False, judge_model: str | Path | None = None, judge_target: str | None = None,
                  response_tokens: int = 96, random_controls: int = 16, same_reply: bool = False,
                  segments: str | list | None = None, stack_layers: str | list[int] | None = None,
                  stack_fast: bool = False) -> dict:
        """対照集合から全層の方向 r_L を作って保存し、select="auto" なら効果の大きい層を選ぶ(数字なら手で指定)。
        random_controls > 0 なら層選びに同じ大きさのランダムな向きの対照を入れ、対照の p95 を超えた層だけを選ぶ(CHANGE-03 A)。
        positions="response" は返事のトークンから作る(A / B の system で response_tokens まで返事を生成して流す)。
        same_reply(positions response だけ): B 側も A 側の返事を読む(同じ返事を両方の条件で読ませた差。読んでいる語の違いから来る
        成分が入らない)。segments("0,1-3,4-"): 返事の位置で分けた方向も作る(名前 NAME.seg<i> で 1 本の方向として使える)。
        stack_layers: 区切りつきの方向を、これらの層に重ねて書く前提の値にする(direction.response_directions)。
        リストか、"11-23"・"11,18,23"・"11,14-16" のような文字列(ほかの layers と同じ読み方)。
        stack_fast: stack_layers の読み直しを速い読み方にする(区切りが続く範囲ごとに 1 回で読む)。上の層の方向が、既定の読み方と
        わずかに違うことがある(direction.response_directions)。"""
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
        if isinstance(stack_layers, str) and not stack_layers.strip():
            stack_layers = None
        if stack_layers is not None:                    # --layers などと同じ読み方("11-23"・"11,18,23"・混ぜた書き方・リスト)
            from .jobs import _layers
            if isinstance(stack_layers, str) and stack_layers.strip() == "all":
                raise GneError("BAD_ARGS", "stack_layers に all は使えません", "重ねて書く層を 11-23 か 11,18,23 のように書きます")
            stack_layers = _layers(stack_layers)
        if stack_fast and not stack_layers:
            raise GneError("BAD_ARGS", "stack_fast は stack_layers と一緒に使います", "gne direction … --stack-layers 11-23 --stack-fast")
        if (same_reply or segments or stack_layers) and positions != "response":
            raise GneError("BAD_ARGS", "same_reply・segments・stack_layers は positions response のときだけ使えます",
                           "gne direction --contrast C --positions response --same-reply")
        if positions == "response":
            cr = response_directions(self, cs, max_tokens=response_tokens, progress=self.progress, same_reply=same_reply,
                                     segments=segments, stack_layers=stack_layers, stack_fast=stack_fast)
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
        how = {"method": cr.method, "same_reply": bool(getattr(cr, "same_reply", False))}
        if ds.segments:
            how["segments"] = [{"index": i, "positions": sg, "name": f"{name}.seg{i}",
                                "rows": {"A": cr.seg_rows["A"][i], "B": cr.seg_rows["B"][i]}} for i, sg in enumerate(ds.segments)]
            how["stack_layers"] = list(getattr(cr, "stack_layers", []) or [])
            if getattr(cr, "stack_fast", False):
                how["stack_fast"] = True
        return {"name": name, "path": str(path), "contrast": str(contrast), "contrast_sha": cs.sha, "positions": positions, **how,
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
        from .direction import require_length
        require_length(ds, from_layer)
        return ds.raw[from_layer], {"direction": ds.name, "from_layer": from_layer}

    def ablate(self, direction: str, out: str | Path, *, from_layer: int | None = None, per_layer: bool = False,
               layers: list[int] | None = None, scale: float = 1.0, embed: bool = True, bake: bool = False,
               requant: str = "q8_0", via_f16: bool = False, imatrix: str | None = None,
               llama_quantize: str | None = None, verify: bool = False, correction: bool | None = None) -> dict:
        """方向の射影消去を LoRA アダプタ GGUF(既定)か、重みを書き換えた GGUF(bake)に書き出す(§6.17)。
        verify なら、LoRA は V13(r 成分が残る分だけか)、bake は V13 と V14(LoRA と greedy が一致するか)を書き出したファイルで見る。"""
        from .ablate import bake_ablation_gguf, bake_via_f16, export_ablation_lora
        if not 0 < scale <= 1.5:
            raise GneError("BAD_ARGS", "--scale は 0 より大きく 1.5 以下にしてください(1 で完全に消す)")
        r, meta = self.ablation_vector(direction, from_layer=from_layer, per_layer=per_layer)
        corr = self.proj_correction if correction is None else bool(correction)
        t0 = time.perf_counter()
        if not bake:
            rep = export_ablation_lora(self.weights, r, out, layers=layers, scale=scale, embed=embed, meta=meta,
                                       progress=self.progress, correction=corr)
            kind = "lora"
        elif via_f16:
            rep = bake_via_f16(self.weights, r, out, layers=layers, scale=scale, embed=embed, meta=meta,
                               llama_quantize=llama_quantize, imatrix=imatrix, progress=self.progress, correction=corr)
            kind = "bake"
        else:
            rep = bake_ablation_gguf(self.weights, r, out, layers=layers, scale=scale, embed=embed, requant=requant,
                                     meta=meta, progress=self.progress, correction=corr)
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
        er = v.embed_component(ids, ru)                  # 埋め込みを拡大する arch(Gemma 系)は捕捉した inp_scaled から
        if kind == "lora":
            res = v._resid_all(self.engine, ids, [(out, 1.0)])
            expect = 0.0 if rep.get("embed") else er
        else:
            e2 = Engine(out, n_gpu_layers=self._engine_kw["n_gpu_layers"], n_ctx=self.engine.n_ctx, n_seq_max=self.engine.n_seq_max)
            try:
                res = v._resid_all(e2, ids)
                lp = out.with_name(out.name + ".v14.lora.gguf")
                export_ablation_lora(self.weights, r, lp, layers=layers, scale=rep["scale"],
                                     embed=rep["embed"] and embed_in_lora(self.weights),
                                     correction=rep.get("proj_correction", "postnorm") != "none")
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
             when: str | None = None, lora_when: str = "all", judge_model: str | Path | None = None, judge_target: str | None = None,
             name: str | None = None, random_controls: int = 0, reactivity: str | Path | None = None,
             head_controls: int = 0, control_pool: str = "before", direction_metrics: bool = True,
             dir_max_examples: int | None = 64, keyed_contrast: str | Path | None = None, keyed_metrics: bool = True,
             keyed_examples: int | None = 8, keyed_match: bool = True, keyed_pseudo: int = 8,
             keyed_subset: str | None = None) -> dict:
        """レシピの倍率ごとの副作用(能力・ppl・言語混入)と性格の変化量を測り、<root>/evals/<model_id>/<name>.json に保存する。
        random_controls > 0 なら各倍率で同じ大きさのランダムな対照レシピも測る(行の ctrl・ctrl_novelty_p95 など)。
        reactivity(対照集合の YAML)で反応性(トリガーあり/なしの差)も測り、head_controls 本のランダムなヘッドと比べる。
        方向を使うレシピ(proj など)では方向の指標(射影の値の下がり方・ずれ・判定の一致率)も測る。direction_metrics=False で
        測らない。dir_max_examples はその測定で読む質問の数(A・B それぞれ、一致率の返事も。多ければ等間隔に間引く)。
        keyed のユニットを持つレシピ(CHANGE-07 E)では eval の選択性も測る(行の keyed と結果の keyed、evalx.evaluate)。
        keyed_contrast は鍵を作った対照集合(configs/contrast/ の名前かパス。省くと方向を作った対照集合が鍵と同じならそれ、
        どちらも無ければ結果の keyed は skipped)、keyed_examples は会話の組ごとに読む数(None か 0 で全部)、keyed_pseudo は
        比較用の鍵の本数、keyed_subset は読む例(even / odd / all、省くと鍵の作成と反対側)、keyed_match=False で方式の比較の
        大きさ合わせを省く、keyed_metrics=False で測らない。"""
        from .evalx import Judge, evaluate
        recipe = self._as_recipe(recipe)
        if not recipe.units and recipe.ablate is None:
            raise GneError("BAD_RECIPE", "ユニットがありません")
        if (judge_model is None) != (judge_target is None):
            raise GneError("BAD_ARGS", "--judge-model と --judge-target は両方指定してください")
        keyed_subset = keyed_subset or None
        if keyed_subset is not None and keyed_subset not in ("even", "odd", "all"):
            raise GneError("BAD_ARGS", f"keyed_subset は even / odd / all です({keyed_subset!r})")
        keyed_examples = None if keyed_examples is None else int(keyed_examples)
        keyed_pseudo = int(keyed_pseudo)
        if (keyed_examples or 0) < 0 or keyed_pseudo < 0:
            raise GneError("BAD_ARGS", "keyed_examples と keyed_pseudo は 0 以上にしてください")
        keyed_contrast = keyed_contrast or None
        if keyed_contrast is not None and not recipe.keyed_units:
            self.warn("keyed のユニットが無いレシピなので keyed_contrast は使いません")
            keyed_contrast = None
        elif keyed_contrast is not None and keyed_metrics:     # 名前も受ける。無ければ長い eval の前に止める
            keyed_contrast = self._contrast_file(keyed_contrast)
        judge = Judge(judge_model, judge_target) if judge_model else None
        try:
            d = evaluate(self, recipe, doses=doses, suite=suite, when=when, judge=judge, lora_when=lora_when,
                         random_controls=random_controls, reactivity=reactivity, head_controls=head_controls,
                         control_pool=control_pool, direction_metrics=direction_metrics,
                         dir_max_examples=dir_max_examples, keyed_contrast=keyed_contrast,
                         keyed_metrics=bool(keyed_metrics), keyed_examples=keyed_examples or None,
                         keyed_match=bool(keyed_match), keyed_pseudo=keyed_pseudo, keyed_subset=keyed_subset,
                         progress=self.progress)
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
        fc = self.proj_forecast(recipe, scales=[x for x in doses if x])   # v0.9.0: 正規化のある層の射影の、倍率ごとの効きの見込み
        if fc:
            d["proj_forecast"] = fc
        d["proj_correction"] = self.proj_correction_label(recipe)
        old = self.proj_record_note(recipe)                           # 記録の無い古いレシピ・頭打ちの知らせを結果の notes にも残す(画面の評価に出る)
        extra_notes = [x for x in (old, self.proj_forecast_text(recipe, old=bool(old))) if x]
        if extra_notes:
            d["notes"] = [*(d.get("notes") or []), *extra_notes]
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
        丸ごと倍率(gain、k_h = 1 − α·s_h)・上位ニューロンの消去(gain、k = 1 − α)を、A 側の射影の値(速い指標、score_layer)が
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
        if score_layer is None:                          # 組のヘッドより後ろの、最初の射影の値を測る層
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
                # 方式によっては射影の値が倍率に単調でない(丸ごと倍率は直交成分も反転する)ので、粗く見てから詰める
                alpha, val_a, reached, _ = _bisect(lambda a, make=make: mean_a(make(a)), target, hi, grid=8)
            if not reached:
                rows.append({"method": key, "desc": desc, "alpha": alpha, "score_a": val_a,
                             "excluded": f"α = 0〜{hi:g} で A 側の射影の値は最小でも {val_a:.2f}(α = {alpha:.3g})で、目標 {target:.2f} に届きません"})
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
        """CHANGE-06 B: 速い指標。層ごとの A / B の射影の値と cohen_d、基準層(cohen_d 最大、記録する)、後段の層の方向との cos。
        units / recipe を渡すと、その介入での射影の値の変化と直交成分のずれ。set_base で基準層を手で上書きする。"""
        from .direction import set_base_layer
        from .direction import score as score_fn
        ds, cs, _ = self._direction_contrast(name, contrast)
        self.engine.free_steer()                  # 生成しないので、生成用のコンテキストを空けて VRAM を使えるようにする
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

    def behave(self, prompts: str | Path | list, *, recipes: list | None = None, match: str | None = None,
               judge: str | None = None, labels: str = "R,P,C", max_tokens: int = 400, thinking: bool = False,
               temperature: float = 0.0, seeds: list[int] | None = None, name: str = "behave") -> dict:
        """レシピ(複数なら重ねる)を当てて質問集に答えさせ、正規表現の一致と読み手の判定を数える(behave.py)。"""
        from .behave import behave, load_prompts
        from .steer import combine_recipes
        if not re.fullmatch(r"[\w.\-]+", name):
            raise GneError("BAD_ARGS", f"名前 {name!r} は英数字・_・-・. だけにしてください")
        qs = load_prompts(prompts) if isinstance(prompts, str | Path) else list(prompts)
        rs = [Recipe.load(self.recipe_path(r)) if isinstance(r, str | Path) else self._as_recipe(r) for r in (recipes or [])]
        rec = combine_recipes(rs, name=name) if rs else None
        vecs = self.vectors(rec)[0] if rec is not None and rec.add_units else None
        d = behave(self, qs, lora=self.lora_for(rec), vectors=vecs or None, match=match, judge=judge, labels=labels,
                   max_tokens=max_tokens, thinking=thinking, temperature=temperature, seeds=seeds, progress=self.progress)
        out = self.store.root / "behave" / self.meta.model_id / f"{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        # 質問と当てたレシピの名前も残す(direction_key_from_behave が、CHiPS のレシピを当てて A 側の質問に答えたものかを確かめる)
        d.update(name=name, recipe=rec.to_dict() if rec is not None else None, recipe_names=[r.name for r in rs], prompts=qs)
        out.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        return {**{k: v for k, v in d.items() if k not in ("runs", "prompts")}, "path": str(out),
                "runs": [{k: v for k, v in r.items() if k != "replies"} for r in d["runs"]]}

    def heads_writers(self, direction: str, *, contrast: str | Path | None = None, ref_layer: int | None = None,
                      expert_layers: list[int] | None = None, max_examples: int | None = 64, positions: str = "last",
                      name: str | None = None, top: int = 12) -> dict:
        """方向の A − B の差を、層ごとに attention・専門家・共有の専門家の書き込みに分ける(components.py)。質問の文は保存しない。"""
        from .components import component_writes
        ds, cs, _ = self._direction_contrast(direction, contrast)
        name = name or ds.name
        if not re.fullmatch(r"[\w.\-]+", name):
            raise GneError("BAD_ARGS", f"名前 {name!r} は英数字・_・-・. だけにしてください")
        self.engine.free_steer()
        d = component_writes(self, ds, cs, ref_layer=ref_layer, positions=positions, max_examples=max_examples,
                             expert_layers=expert_layers, top=top, progress=self.progress)
        out = self.store.heads_dir(self.meta.model_id) / f"{name}.writers.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        return {**d, "name": name, "path": str(out)}

    def heads_causal(self, direction: str, *, contrast: str | Path | None = None, mode: str = "ablate",
                     layers: str | list[int] | None = "auto", candidates: str | list | None = None,
                     score_layers: list[int] | None = None, primary: int | None = None,
                     base_recipe: str | Path | None = None, base_units: list[str] | None = None, delta_s: float = 3.0,
                     hard: int = 16, hard_subset: str | list[int] | None = None, b_side_max: float | None = None,
                     top_layers: int = 4, positions: str = "last", subset: str = "all", b_examples: int = 16,
                     name: str | None = None, direction_layer: str = "own", max_examples: int | None = None,
                     drift_max: float | None = 0.5, include_score_layers: bool = False,
                     select_layer: int | None = None, candidate_layers: list[int] | None = None,
                     always_layers: list[int] | None = None) -> dict:
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
        self.engine.free_steer()                  # 生成しないので、生成用のコンテキストを空けて VRAM を使えるようにする
        if select_layer is not None:              # 選ぶのに使う層: この層での下がり方で順位をつけ、この層より前のヘッドを候補にする
            if primary is not None and int(primary) != int(select_layer):
                raise GneError("BAD_ARGS", "--primary と --select-layer は同じ層にしてください")
            primary = int(select_layer)
        st = prepare(self, ds, cs, score_layers=score_layers, primary=primary, positions=positions, subset=subset,
                     b_examples=b_examples, direction_layer=direction_layer, max_examples=max_examples)
        layer_rows, extra = None, {}
        if isinstance(layers, str) and layers not in ("auto", "all"):
            from .cli import parse_layers
            layers = parse_layers(layers)
        if mode == "ablate":
            if layers == "auto" and candidates in (None, "all"):
                # candidate_layers: 候補のヘッドを探す層をここに絞る(同じ層を 2 回実行するモデルで、1 回目だけ・2 回目だけを探すとき)
                cl = None if candidate_layers is None else sorted({int(x) for x in candidate_layers})
                if cl is not None:
                    from .causal import head_layers as _hl
                    cl = [L for L in _hl(self, max(st.score_layers)) if L in cl]
                    if not cl:
                        raise GneError("BAD_ARGS", "candidate_layers に、ヘッドのある層がありません")
                layer_rows = screen_layers(self, ds, cs, setup=st, layers=cl, progress=self.progress)
                # 止めると中立(B 側)の残差が自分の大きさの drift_max 倍以上動く層は、方向に関係なくモデルを壊しているので選ばない
                for r in layer_rows:
                    if drift_max is not None and (r.get("drift_orth_b") or 0) > drift_max:
                        r["excluded"] = "breaks"
                # 射影の値を測る層とそれより後の層のヘッドは、測る量を直接動かすので効率が高く見えるが、振る舞いはほとんど変わらなかった
                # (docs/head-component-steering.md)。既定では 1 個ずつ測る層から外す
                # 選ぶのに使う層(primary)を指定したときは、その層での下がり方で順位をつけ、その層より前のヘッドを候補にする
                # (同じ層を 2 回実行するモデルで、1 回目の出口で選ぶときなど)。指定しないときは今までどおり
                first = st.primary if select_layer is not None else min(st.score_layers)
                rank = "causal" if select_layer is not None else "causal_max"
                ok = [r for r in layer_rows if r[rank] > 0 and not r.get("excluded")]
                early = [r for r in ok if r["layer"] < first]
                if not include_score_layers and not early and ok:
                    self.warn(f"射影の値を測る層 L{first} より前に、止めて射影の値が下がる層がないので、測る層のヘッドも測ります")
                pos = sorted(ok if include_score_layers or not early else early, key=lambda r: -r[rank])
                Ls = sorted(r["layer"] for r in pos[:max(1, top_layers)])
                if always_layers:                 # 層の順位に関係なく、いつもヘッド単位で測る層(候補の境より前で、壊すだけでない層)
                    okL = {r["layer"] for r in layer_rows if not r.get("excluded")}
                    Ls = sorted(set(Ls) | {int(L) for L in always_layers
                                           if int(L) in okL and (include_score_layers or int(L) < first)})
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
                   **({"top_layers": top_layers} if mode == "ablate" and layer_rows is not None else {}),
                   **({"score_layers_given": True} if score_layers or select_layer is not None else {}),
                   **({"select_layer": int(select_layer)} if select_layer is not None else {}),
                   **({"candidate_layers": sorted({int(x) for x in candidate_layers})} if candidate_layers is not None else {}),
                   **({"always_layers": sorted({int(x) for x in always_layers})} if always_layers else {}),
                   **({"loop_edits": self.loop_edits_mode()} if self.weights.loop is not None else {}),
                   "base_scores": {str(L): {"A": float(st.base["A"][L].mean()), "B": float(st.base["B"][L].mean())}
                                   for L in st.score_layers},
                   "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
        if layer_rows is not None:
            summary["layer_rows"] = layer_rows
        pj, ps = save_causal(self, name, rows, summary)
        key = (lambda r: -r["eta"]) if mode == "marginal" else \
            (lambda r: -(r.get("causal" if select_layer is not None else "causal_max") or 0))
        return {"name": name, "path": str(pj), "summary_path": str(ps), **summary,
                "layers": layer_rows, "rows": sorted(rows, key=key), "n_rows": len(rows),
                "seconds": time.perf_counter() - t0}

    def heads_allocate(self, direction: str, *, contrast: str | Path | None = None, candidates: str | Path | None = None,
                       target: float, score_layer: int | None = None, n_heads=(6, 5, 4), hard: int = 16,
                       hard_subset: str | Path | None = None, proportional_to: str = "eta", from_zero: bool = True,
                       b_side_max: float | None = None, base_recipe: str | Path | None = None, delta_s: float = 3.0,
                       pool: int = 30, s_max: float | None = None, remain_threshold: float = 0.0, positions: str | None = None,
                       subset: str | None = None, direction_layer: str = "own", name: str | None = None,
                       max_examples: int | None = None, method: str = "proportional", include_score_layers: bool = False,
                       candidate_layers: list[int] | None = None, report_initial: bool = True) -> dict:
        """CHANGE-06 D: 効率ベースの強さ配分。candidates は heads causal の結果(名前か .causal.jsonl のパス、既定は方向の名前)。
        method は proportional(s ∝ η で全体を二分探索、s_max の既定 30)か greedy(1 本ずつ上げる、s_max の既定はなし)。"""
        from .causal import allocate, causal_paths, load_causal
        from .headproj import proj_heads
        ds, cs, _ = self._direction_contrast(direction, contrast)
        self.engine.free_steer()                  # 生成しないので、生成用のコンテキストを空けて VRAM を使えるようにする
        cname = str(candidates or ds.name)
        rows = load_causal(self, cname)
        sp = causal_paths(self, Path(cname).name.removesuffix(".causal.jsonl"))[1]
        summ = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
        if score_layer is None:
            sl = summ.get("score_layers") or []
            if not sl and summ.get("primary") is None:      # 黙って L0 で測らない
                raise GneError("BAD_ARGS", f"{sp.name} が無いので射影の値を測る層が分かりません",
                               "--score-layer を付けるか、.causal.jsonl と同じ名前の .causal.json を横に置いてください")
            score_layer = int(sl[-1]) if sl else int(summ.get("primary"))
        if candidate_layers is not None:          # 候補のヘッドを指定した層に絞る
            keep = {int(x) for x in candidate_layers}
            rows = [r for r in rows if r.get("head") is None or int(r["layer"]) in keep]
        if not include_score_layers:              # 射影の値を測る層とそれより後のヘッドは候補にしない(heads_causal と同じ理由)
            first = int(score_layer) if summ.get("select_layer") is not None else \
                min([int(x) for x in summ.get("score_layers") or []] + [int(score_layer)])
            kept = [r for r in rows if r.get("head") is None or int(r["layer"]) < first]
            if any(r.get("head") is not None for r in kept):
                rows = kept
            else:
                self.warn(f"射影の値を測る層 L{first} より前のヘッドが候補にないので、測る層のヘッドも使います"
                          "(heads causal を前の層で測ると外せます)")
        base = proj_heads(Recipe.load(self.recipe_path(base_recipe))) if base_recipe else None
        return allocate(self, ds, cs, candidates=rows, target=float(target), score_layer=int(score_layer),
                        hard_subset=hard_subset, n_heads=tuple(n_heads), proportional_to=proportional_to,
                        from_zero=from_zero, b_side_max=b_side_max, base_units=base, delta_s=delta_s, hard=hard,
                        pool=pool, s_max=s_max, method=method, remain_threshold=remain_threshold, name=name or ds.name,
                        positions=positions or summ.get("positions") or "last", subset=subset or summ.get("subset") or "all",
                        direction_layer=direction_layer, score_layers=summ.get("score_layers"),
                        max_examples=max_examples if max_examples is not None else summ.get("max_examples"),
                        progress=self.progress, report_initial=report_initial)

    PIPELINE_DEFAULTS = {"positions": "last", "score_positions": "last", "target": None, "n_heads": [6, 5, 4], "hard": 16,
                         "top_layers": 4, "max_examples": 64, "s_max": 30.0, "direction_layer": "own", "resume": True,
                         "include_score_layers": False, "same_reply": False, "score_layers": None, "select_layer": None,
                         "candidate_layers": None, "always_layers": None}
    # rank: writes(v0.8.0 からの既定: 全部の部品を書き込みの量で順位づけ → 上位を s=1 で確かめる → s=1 から組を作る)/
    # causal(v0.7.0 までの既定: 層 → ヘッドを止めて測る因果スクリーニング → 効率で配り直す)。unit_kinds の既定は heads(ニューロンは
    # 指定で足す)。rank 以外は writes のときだけ使う設定(causal の結果の settings には rank だけを載せる)
    PIPELINE_WRITES = {"rank": "writes", "unit_kinds": "heads", "verify_top": 64, "verify_sample": 16, "verify_mode": "proj",
                       "n_units": [24, 12, 6], "check_replies": None, "check_tokens": 512, "loop_direction": None,
                       "holdout": False, "strength": "per_unit"}
    CAUSAL_ONLY = ("n_heads", "top_layers", "always_layers", "include_score_layers")   # rank causal だけで使う設定(hard は両方で使う)

    def _loop_pipeline_defaults(self, pipe: dict) -> dict | None:
        """同じ層を 2 回実行するモデル(sensenova_loop)の一括の既定。測る層を指定しなければ loop(段 27・51・54)・選ぶ層 54 にする
        (rank writes の選ぶ層の既定も最後に測れる層 = 段 54 なので同じ)。rank causal のときは、ループの外の段 52・53 もいつもヘッド単位で
        測る(層を止めた下がり方の順位に関係なく。rank writes は全部の層を順位づけるので要らない)。v0.7.0 より前の既定は
        score_layers: base、段 52・53 を足さないのは always_layers: none。ほかのモデルでは何もしない(None)。"""
        lp = self.weights.loop
        if lp is None:
            return None
        used = {}
        if pipe.get("score_layers") in (None, "", []):
            pipe["score_layers"] = "loop"
            used["score_layers"] = "loop"
            if pipe.get("select_layer") in (None, ""):
                pipe["select_layer"] = lp.n_layer - 2
                used["select_layer"] = lp.n_layer - 2
        if pipe.get("always_layers") is None and pipe.get("rank", "causal") == "causal":
            tail = [L for L in range(lp.hi2, lp.n_layer - 2)
                    if self.weights.arch.layers[L].head_dim and self.weights.arch.layers[L].head_writer]
            pipe["always_layers"] = tail
            used["always_layers"] = tail
        return used or None

    def _loop_direction(self, pipe: dict) -> str:
        """重みを 2 回使うモデルで両方の回に当てるとき、ループの中の部品をどちらの回の段の方向で測り・射影するか(unitwrites.unit_layers)。"""
        v = pipe.get("loop_direction")
        return str(v) if v not in (None, "") else "first"

    def _rank_fallback(self, pipe: dict, rank_given: bool) -> str | None:
        """rank を指定していない(既定の writes)のに、書き込みの順位づけに未対応のモデル(残差が複数本の流れ = hyper-connections)なら、
        pipe の rank を causal にして理由を返す(モデルは読み込まず、GGUF のメタデータで決める)。指定したときは何もしない。"""
        if pipe.get("rank") != "writes" or rank_given:
            return None
        why = writes_block_reason(self.weights, self.meta.arch)
        if why:
            pipe["rank"] = "causal"
            until = _writes_block_until(why)
            return f"{why}ので、既定の選び方を rank causal(止めて選ぶ)に切り替えて実行しました" + (f"({until})" if until else "")
        return None

    def _writes_block_reason(self) -> str | None:
        return writes_block_reason(self.weights, self.meta.arch)

    @staticmethod
    def _layers_of(spec) -> list[int] | None:
        """層の並びの指定("28-53"・"4,5,6"・[28, 29]・None)→ 並べた層の番号か None。"""
        if spec in (None, "", []):
            return None
        if isinstance(spec, str):
            from .cli import parse_layers
            return sorted(set(parse_layers(spec)))
        return sorted({int(x) for x in spec})

    def score_layers_of(self, spec) -> list[int] | None:
        """射影の値を測る層の指定: None(既定 = 基準層と後段の 2 層)、層の番号の並び("27,51,54" か [27, 51, 54])、
        loop(同じ層を 2 回実行するモデル: 1 回目の出口・2 回目の出口・最後の層の 1 つ前。最後の層は捕捉の行が間引かれるので測れない)。"""
        if spec in (None, "", []) or (isinstance(spec, str) and spec.strip().lower() in ("base", "default")):
            return None                           # base: 基準層と後段の 2 層(同じ層を 2 回実行するモデルでも、今までの既定)
        if isinstance(spec, str) and spec.strip().lower() == "loop":
            lp = self.weights.loop
            if lp is None:
                raise GneError("BAD_ARGS", "score_layers の loop は、同じ層を 2 回実行するモデルだけで使えます")
            return [lp.lo2 - 1, lp.hi2 - 1, lp.n_layer - 2]
        if isinstance(spec, str):
            from .cli import parse_layers
            Ls = parse_layers(spec)
        else:
            Ls = [int(x) for x in spec]
        n = self.weights.n_main                                          # MTP の層は本流で実行されないので数えない
        bad = [L for L in Ls if not 0 <= L <= n - 2]
        if bad or not Ls:
            raise GneError("BAD_ARGS", f"射影の値を測る層は 0〜{n - 2} です({bad or spec})")
        return sorted(set(Ls))

    def heads_pipeline(self, spec: dict | str | Path, **over) -> dict:
        """対照集合(JSON か YAML、または dict)から、方向づくり → 基準層 → 因果スクリーニング → 強さの配分 を 1 回で回す。
        spec は対照集合と同じ中身(name・user_prompts か側ごとの prompts_a / prompts_b・system_a / system_b か
        recipe_a / recipe_b・validate)に、任意で
        pipeline: {positions, score_positions, target, n_heads, hard, top_layers, max_examples, s_max, direction_layer}。
        over(引数)は pipeline より優先。target を省くと、射影の値を測る最後の層での A 側と B 側の平均の真ん中(「残る」はこれを超える例)。
        止めると中立の入力の残差が大きく動く(直交成分のずれ > 0.5)層とヘッドは、モデルを壊しているだけなので候補から外す。
        方向づくりには質問を全部使い、止めて測る段階(射影の値・因果・配分)は max_examples 個に等間隔に間引く。
        resume(既定 True)なら、同じ対照集合・同じ設定で作り済みの方向と因果の結果を使い回す(止めて流し直したとき、配分から
        やり直せる)。結果は <root>/heads/<model_id>/<name>.pipeline.json。"""
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
        spec_pipe = spec.pop("pipeline", None) or {}
        given = {**{k: v for k, v in spec_pipe.items() if v is not None}, **{k: v for k, v in over.items() if v is not None}}
        pipe = {**self.PIPELINE_DEFAULTS, **self.PIPELINE_WRITES, **spec_pipe,
                **{k: v for k, v in over.items() if v is not None}}
        unknown = sorted(set(pipe) - set(self.PIPELINE_DEFAULTS) - set(self.PIPELINE_WRITES) - {"name"})
        if unknown:
            raise GneError("BAD_ARGS", f"pipeline に {', '.join(unknown)} は使えません"
                                       f"({', '.join(list(self.PIPELINE_DEFAULTS) + list(self.PIPELINE_WRITES))})")
        if pipe["rank"] not in ("causal", "writes"):
            raise GneError("BAD_ARGS", f"rank は causal(層 → ヘッドを止めて測る、今までの既定)か writes(全部の部品を書き込みの量で"
                                       f"順位づけ)です({pipe['rank']!r})")
        rank_given = "rank" in given
        rank_note = self._rank_fallback(pipe, rank_given)
        writes = pipe["rank"] == "writes"
        if writes:
            from .unitwrites import parse_kinds
            parse_kinds(pipe["unit_kinds"])                             # 書き方の誤りは流す前に返す
            if pipe["verify_mode"] not in ("proj", "ablate"):
                raise GneError("BAD_ARGS", "verify_mode は proj(s=1 で射影)か ablate(出力を 0)です")
            from .unitwrites import STRENGTHS
            if (pipe.get("strength") or "per_unit") not in STRENGTHS:
                raise GneError("BAD_ARGS", f"strength は {' / '.join(STRENGTHS)} です({pipe.get('strength')!r})")
        else:                                       # v0.7.0 までの流れ: writes の設定は使わない(settings には rank だけを残す)
            for k in self.PIPELINE_WRITES:
                if k != "rank":
                    pipe.pop(k, None)
        self._pipeline_target(pipe["target"], {"A": 0.0, "B": 0.0})          # 目標の書き方の誤りは、流す前に返す
        name = str(pipe.pop("name", None) or spec.get("name") or "")
        if not re.fullmatch(r"[\w.\-]+", name):
            raise GneError("BAD_ARGS", f"名前 {name!r} は英数字・_・-・. だけにしてください(spec の name)")
        spec["name"] = name
        from .reverse import check_questions
        for key in ("user_prompts", "prompts_a", "prompts_b"):
            check_questions(spec.get(key), key)
        spec_rep = None
        if writes and pipe.get("holdout"):            # 選ぶ用と表に出す用に質問を分ける(偶数番目で選び、奇数番目で表の数を測る)
            spec, spec_rep = self._holdout_split(spec)
        path = self.store.contrast_dir() / f"{name}.yaml"
        from .yamlio import write_yaml
        write_yaml(path, spec)                     # 読み戻して同じ文字になるか確かめてから書く(U+0085 なども)
        from .direction import load_contrast
        try:
            cs, val = load_contrast(path)
            if min(len(cs.prompts("A")), len(cs.prompts("B"))) < 4:
                raise GneError("BAD_CONTRAST", "質問は両側とも 4 個以上にしてください(A と B の分かれ方と効き目を測るため)")
        except GneError:
            path.unlink(missing_ok=True)
            raise
        mx = int(pipe["max_examples"]) if pipe["max_examples"] else None
        sp = pipe["score_positions"]
        arch_default = self._loop_pipeline_defaults(pipe)          # 同じ層を 2 回実行するモデルだけの既定(引数で渡せば今までどおり)
        sls = self.score_layers_of(pipe["score_layers"])          # 測る層(None = 既定)
        cand = self._layers_of(pipe["candidate_layers"])
        sel = None if pipe["select_layer"] in (None, "") else int(pipe["select_layer"])
        if sel is not None and sls is not None and sel not in sls:
            raise GneError("BAD_ARGS", f"選ぶのに使う層 L{sel} が、射影の値を測る層 {sls} に入っていません")
        out: dict = {"name": name, "contrast": str(path), "n_prompts": len(cs.prompts("A")),
                     **({"n_prompts_b": len(cs.prompts("B"))} if cs.separate_prompts else {}), "sides": cs.describe(),
                     "settings": dict(pipe), "rank": pipe["rank"], **({"rank_note": rank_note} if rank_note else {}),
                     **({"arch_defaults": arch_default} if arch_default else {})}
        if writes:                                  # rank causal だけで使う設定を引数で渡したとき(使わないことを知らせる。ひな形の pipeline の値は除く)
            unused = sorted(k for k in self.CAUSAL_ONLY if over.get(k) not in (None, False, "none"))
            if unused:
                out["unused_settings"] = unused
        reused: list[str] = []                     # resume: 同じ対照集合・同じ設定で作り済みの段は使い回す(止めて流し直したとき)
        self.engine.free_steer()                   # 生成しないので、生成用のコンテキストを空けて VRAM を使えるようにする
        ts = time.perf_counter()                                   # 1. 方向(全部の質問)
        d = None
        if pipe["resume"] and pipe["positions"] != "response":
            try:
                m = self.direction_set(name).meta
                if m.get("contrast_sha") == cs.sha and m.get("positions") == pipe["positions"]:
                    d = {"path": str(self.direction_set(name).path), "positions": m["positions"],
                         "n_rows": {"A": int(m.get("n_rows_A") or 0), "B": int(m.get("n_rows_B") or 0)}}
                    reused.append("direction")
            except GneError:
                d = None
        if d is None:                                      # same_reply は positions response のときだけ(違えば direction が断る)
            d = self.direction(contrast=path, name=name, positions=pipe["positions"], same_reply=bool(pipe["same_reply"]))
        out["direction"] = {"path": d["path"], "positions": d["positions"], "n_rows": d["n_rows"],
                            "seconds": time.perf_counter() - ts, "reused": "direction" in reused}
        ts = time.perf_counter()                                   # 2. 基準層(cohen_d が最大の層)
        sc = self.direction_score(name, contrast=path, positions=sp, record_base=True, max_examples=mx)
        out["score"] = {"base_layer": sc["base_layer"], "best_layer": sc["best_layer"], "n": sc["n"],
                        "layers": [{k: r[k] for k in ("layer", "cohen_d", "cos_base")} | {"A": r["A"]["mean"], "B": r["B"]["mean"]}
                                   for r in sc["layers"]], "seconds": time.perf_counter() - ts}
        if writes:                                                 # 3〜5. 書き込みの順位づけ → 確かめ → s=1 から組を作る
            try:
                return self._pipeline_writes(out, name, path, cs, val, pipe, sp, mx, sls, sel, reused, t0, cand, spec_rep)
            except GneError as e:
                if rank_given or e.code != "ARCH_UNSUPPORTED":
                    raise
                # 既定(rank を指定していない)で、この arch の捕捉に未対応だったときは、v0.7.0 までの選び方で続ける
                self.warn(f"このモデルでは書き込みの量での順位づけを実行できないので、止めて選ぶ流れ(rank causal)で実行します。理由: {e.message}")
                pipe["rank"] = "causal"
                for k in self.PIPELINE_WRITES:
                    if k != "rank":
                        pipe.pop(k, None)
                more = self._loop_pipeline_defaults(pipe)
                if more:
                    out["arch_defaults"] = {**(out.get("arch_defaults") or {}), **more}
                out.update(settings=dict(pipe), rank="causal",
                           rank_note=f"書き込みの量での順位づけを実行できないので、既定の選び方を rank causal(止めて選ぶ)に切り替えて実行しました。理由: {e.message}")
                out.pop("unused_settings", None)
                for k in ("writes", "verify"):
                    out.pop(k, None)
        ts = time.perf_counter()                                   # 3. 因果(層ごと → 効く層のヘッドを 1 個ずつ止める)
        c = self._pipeline_causal_reuse(name, cs, sp, mx, pipe) if pipe["resume"] and "direction" in reused else None
        if c is not None:
            reused.append("causal")
        else:
            c = self.heads_causal(name, contrast=path, mode="ablate", layers="auto", top_layers=int(pipe["top_layers"]),
                                  positions=sp, max_examples=mx, direction_layer=pipe["direction_layer"],
                                  include_score_layers=bool(pipe["include_score_layers"]),
                                  score_layers=sls, select_layer=sel, candidate_layers=cand,
                                  always_layers=self._layers_of(pipe["always_layers"]) if pipe["always_layers"] != "none" else None)
        out["causal"] = {"reused": "causal" in reused, "path": c["path"], "score_layers": c["score_layers"], "primary": c["primary"],
                         "base_scores": c["base_scores"], "n_rows": c["n_rows"], "n_a": c["n_a"], "n_b": c["n_b"],
                         "layers": c.get("layers"), "measured_layers": sorted({r["layer"] for r in c["rows"]}),
                         "top": c["rows"][:20], "seconds": time.perf_counter() - ts}
        # 4. 配分(最初の比率 → 効率で配り直す → ヘッド数の比較)。目標と配分に使う層は、select_layer(無ければ測る最後の層)
        S = int(sel) if sel is not None else int(c["score_layers"][-1])
        bs = c["base_scores"][str(S)]
        tgt, tgt_from = self._pipeline_target(pipe["target"], bs)
        a = self.heads_allocate(name, contrast=path, target=float(tgt), score_layer=S, n_heads=tuple(int(x) for x in pipe["n_heads"]),
                                hard=int(pipe["hard"]), s_max=float(pipe["s_max"]), positions=sp,
                                direction_layer=pipe["direction_layer"], max_examples=mx, remain_threshold=float(tgt),
                                include_score_layers=bool(pipe["include_score_layers"]), candidate_layers=cand)
        cs_chk, val_chk = cs, val
        if spec_rep is not None:                                   # 表の数を、選ぶのに使っていない質問で測り直す
            cs_chk, val_chk, out["holdout"] = self._pipeline_holdout(st, ds, spec_rep, a, pipe, sp, mx)
            p_ = Path(a["path"])
            p_.write_text(json.dumps({k: v for k, v in a.items() if k != "path"}, ensure_ascii=False, indent=1), encoding="utf-8")
        out["allocate"] = a
        out["reused"] = reused
        out["target"], out["target_from"] = float(tgt), tgt_from
        rec = next((r for r in a["table"] if r.get("recommended")), None)
        out["recommended"] = None if rec is None else {"n_heads": rec["n_heads"], "units": rec["units"], "recipe": rec["recipe"]}
        self._postnorm_pipeline_note(out)
        out["seconds"] = time.perf_counter() - t0
        out["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        p = self.store.heads_dir(self.meta.model_id) / f"{name}.pipeline.json"
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        out["path"] = str(p)
        return out

    def _holdout_split(self, spec: dict) -> tuple[dict, dict]:
        """対照集合の質問を、偶数番目(選ぶ用)と奇数番目(表の数を測る用)に分けた 2 つの表。側ごとに 8 問以上ずつ要る。"""
        keys = [k for k in ("user_prompts", "prompts_a", "prompts_b") if spec.get(k)]
        sel, rep = dict(spec), dict(spec)
        for k in keys:
            q = list(spec[k])
            if len(q) < 16:
                raise GneError("BAD_ARGS", f"holdout には、{k} が 16 問以上要ります({len(q)} 問。半分ずつ、選ぶのと表の数を測るのに使う)")
            sel[k], rep[k] = q[0::2], q[1::2]
        v = spec.get("validate")
        if isinstance(v, dict) and v.get("prompts"):           # 応答の確かめの質問(held-out)は、表に出す側だけに残す
            sel["validate"] = {k_: v_ for k_, v_ in v.items() if k_ != "prompts"}
        rep["name"] = f"{spec['name']}.holdout"
        return sel, rep

    def _pipeline_holdout(self, st, ds, spec_rep: dict, a: dict, pipe: dict, sp: str, mx: int | None) -> tuple[object, object, dict]:
        """選ぶのに使っていない質問(奇数番目)で、表の組(と倍率を上げる前の組)の数を測り直す。戻り値 (対照集合, validate, まとめ)。"""
        from .causal import prepare
        from .direction import drift, load_contrast
        from .headproj import proj_heads
        from .yamlio import write_yaml
        path = self.store.contrast_dir() / f"{spec_rep['name']}.yaml"
        write_yaml(path, spec_rep)
        cs_r, val_r = load_contrast(path)
        S = st.primary
        sr = prepare(self, ds, cs_r, score_layers=st.score_layers, primary=S, positions=sp,
                     direction_layer=pipe["direction_layer"], max_examples=mx)
        ba, bb = float(sr.base["A"][S].mean()), float(sr.base["B"][S].mean())
        tgt, _ = self._pipeline_target(pipe["target"], {"A": ba, "B": bb})
        gap = abs(ba - bb) or 1.0
        for row in [r for r in a["table"] if not r.get("skipped")] + ([a["s1"]] if a.get("s1") else []):
            heads = proj_heads(Recipe.load(row["recipe"]))
            with sr.tmp.use(sr.builder.tensors(heads)) as lora:
                got = sr.measure(lora)
            va = got["A"][S]
            bd = float(got["B"][S].mean()) - bb
            row["holdout"] = {"a_mean": float(va.mean()), "remain": int((va > tgt).sum()), "b_side_delta": bd,
                              "b_toward": (1.0 if ba >= bb else -1.0) * bd / gap,
                              "b_drift_orth": drift(sr.base["B_resid"][S], got["B_resid"][S], sr.scorer.r[S])[1],
                              "effect": (ba - float(va.mean())) - abs(bd), "reached": bool(float(va.mean()) <= tgt + 1e-6)}
        info = {"contrast": str(path), "n_a": sr.scorer.n("A"), "n_b": sr.scorer.n("B"), "target": float(tgt),
                "base_a_mean": ba, "base_b_mean": bb}
        a["holdout"] = info
        return cs_r, val_r, info

    def _pipeline_writes(self, out: dict, name: str, path: Path, cs, val, pipe: dict, sp: str, mx: int | None,
                         sls: list[int] | None, sel: int | None, reused: list[str], t0: float,
                         candidate_layers: list[int] | None = None, spec_rep: dict | None = None) -> dict:
        """一括の rank writes の 3〜5 段(heads_pipeline から)。gne/unitwrites.py。"""
        from .causal import prepare
        from .direction import score_layers_default
        from .unitwrites import allocate_units, measure_writes, parse_kinds, save_writes, verify_rows, verify_units
        S = int(sel) if sel is not None else self.weights.n_main - 2      # 既定: 最後に測れる層(MTP の層は数えない)
        kinds = parse_kinds(pipe["unit_kinds"])
        ds = self.direction_set(name)
        ts = time.perf_counter()                                   # 3. 書き込みの順位づけ(質問は全部)
        summ, arrays = measure_writes(self, ds, cs, select_layer=S, positions=sp, kinds=kinds,
                                      direction_layer=pipe["direction_layer"], candidate_layers=candidate_layers,
                                      loop_direction=self._loop_direction(pipe), progress=self.progress)
        out["writes"] = {k: summ[k] for k in ("select_layer", "kinds", "n", "skipped_long", "gap", "base", "n_units",
                                               "totals", "totals_share", "check", "zero_layers", "zero_note")}
        out["writes"]["layer_scores"] = summ["layer_scores"][:12]          # 層ごとの直接の書き込みの上位(層全体の部品の選び方の目安)
        out["writes"]["seconds"] = time.perf_counter() - ts
        ts = time.perf_counter()                                   # 4. 上位を 1 個ずつ s=1 で確かめる(因果スクリーニングと同じ物差し)
        from .direction import base_layer_of
        b = base_layer_of(self.direction_set(name))
        Ls = sorted(set(sls or score_layers_default(self, ds, b)) | {S})
        Ls = [L for L in Ls if L <= S] or [S]
        st = prepare(self, ds, cs, score_layers=Ls, primary=S, positions=sp, direction_layer=pipe["direction_layer"],
                     max_examples=mx)
        top = verify_rows(summ["rows"], int(pipe["verify_top"]), int(pipe["verify_sample"]))
        ver = verify_units(self, st, top, mode=pipe["verify_mode"], progress=self.progress)
        pz, pjw = save_writes(self, name, summ, arrays, ver)
        out["writes"].update(path=str(pjw), npz=str(pz))
        out["verify"] = {"mode": ver["mode"], "n": len(ver["rows"]), "spearman": ver["spearman"],
                         "kinds": {k: sum(1 for r in ver["rows"] if r["kind"] == k)
                                   for k in ("attn", "linear", "neuron", "layer_attn", "layer_mlp")},
                         "excluded": [r["label"] for r in ver["rows"] if r.get("excluded")],
                         "top": ver["rows"][:20], "seconds": time.perf_counter() - ts}
        bs = {"A": float(st.base["A"][S].mean()), "B": float(st.base["B"][S].mean())}
        tgt, tgt_from = self._pipeline_target(pipe["target"], bs)
        out["causal"] = {"score_layers": st.score_layers, "primary": S, "n_a": st.scorer.n("A"), "n_b": len(st.b_idx),
                         "base_scores": {str(L): {"A": float(st.base["A"][L].mean()), "B": float(st.base["B"][L].mean())}
                                         for L in st.score_layers}}
        a = allocate_units(self, st, ver["rows"], target=float(tgt), n_units=[int(x) for x in pipe["n_units"]],
                           s_max=float(pipe["s_max"]), name=name, direction_layer=pipe["direction_layer"],
                           remain_threshold=float(tgt), max_examples=mx, strength=str(pipe.get("strength") or "per_unit"),
                           hard=int(pipe.get("hard") or 16), progress=self.progress)
        cs_chk, val_chk = cs, val
        if spec_rep is not None:                                   # 表の数を、選ぶのに使っていない質問で測り直す
            cs_chk, val_chk, out["holdout"] = self._pipeline_holdout(st, ds, spec_rep, a, pipe, sp, mx)
            p_ = Path(a["path"])
            p_.write_text(json.dumps({k: v for k, v in a.items() if k != "path"}, ensure_ascii=False, indent=1), encoding="utf-8")
        out["allocate"] = a
        out["reused"] = reused
        out["target"], out["target_from"] = float(tgt), tgt_from
        rec = next((r for r in a["table"] if r.get("recommended")), None)
        # reached: 目標に届いた組 / best_effect: どれも届かないので、組の効き(A 側の下がり − |B 側の変化|)がいちばん大きい組
        out["recommended"] = None if rec is None else {"n_heads": rec["n_heads"], "units": rec["units"], "recipe": rec["recipe"],
                                                       "reached": bool(rec["reached"]), "by": a.get("recommended_by")}
        if a.get("no_recommend"):
            out["no_recommend"] = a["no_recommend"]
        n_check = pipe["check_replies"]
        n_check = 16 if n_check is None else int(n_check)
        ok_rows = [r for r in a["table"] if not r.get("skipped")]
        chk = rec or (min(ok_rows, key=lambda r: (r["remain"], r["b_drift_orth"])) if ok_rows else None)
        if n_check > 0 and chk is not None:                        # 5. 応答での確かめ(書き終えた応答の validate.match の一致数)
            out["check"] = self.reply_check(cs_chk, val_chk, chk["recipe"], n=n_check, max_tokens=int(pipe["check_tokens"]))
            out["check"].update(n_heads=chk["n_heads"], recommended=rec is not None)
            if rec is not None and out["check"].get("break_warning"):
                self._recommend_without_broken(out, a, rec, cs_chk, val_chk, n_check, int(pipe["check_tokens"]))
        self._postnorm_pipeline_note(out)
        out["seconds"] = time.perf_counter() - t0
        out["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        p = self.store.heads_dir(self.meta.model_id) / f"{name}.pipeline.json"
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        out["path"] = str(p)
        return out

    def _recommend_without_broken(self, out: dict, a: dict, rec: dict, cs, val, n: int, max_tokens: int, tries: int = 3) -> None:
        """推す組を当てると応答が壊れる(繰り返し・空が増える)とき、同じ規則で次の組を推し、応答での確かめを流し直す(tries 組まで)。
        壊れた組には replies_break を付ける。どれも壊れれば推さない(no_recommend)。配分の JSON の推す組も書き直す。"""
        from types import SimpleNamespace

        from .unitwrites import _recommend
        tried = [{"n_heads": rec["n_heads"], "break_warning": out["check"]["break_warning"]}]
        rec["replies_break"] = True
        new, by, chk = None, None, out["check"]
        for _ in range(tries - 1):
            rest = [r for r in a["table"] if not r.get("replies_break")]
            n_, by_, _no = _recommend(rest, int(a.get("n_a") or 1), SimpleNamespace(gap=float(a.get("gap") or 1.0)))
            row = next((r for r in rest if n_ is not None and r.get("n_heads") == n_ and not r.get("skipped")), None)
            if row is None:
                break
            ck = self.reply_check(cs, val, row["recipe"], n=n, max_tokens=max_tokens)
            ck.update(n_heads=row["n_heads"], recommended=True)
            tried.append({"n_heads": row["n_heads"], "break_warning": ck.get("break_warning")})
            chk = ck
            if ck.get("break_warning"):
                row["replies_break"] = True
                continue
            new, by = row, by_
            break
        for r in a["table"]:
            r["recommended"] = new is not None and r is new
        a["recommended"], a["recommended_by"] = (new["n_heads"], by) if new is not None else (None, None)
        out["check"] = {**chk, "tried": tried}
        if new is None:
            out["recommended"] = None
            out["no_recommend"] = a["no_recommend"] = {
                "reason": "推す組を当てると応答が壊れる(繰り返し・空が増える)", "next": ["rank causal", "unit_kinds", "refine"]}
        else:
            out["recommended"] = {"n_heads": new["n_heads"], "units": new["units"], "recipe": new["recipe"],
                                  "reached": bool(new["reached"]), "by": by, "after_replies_break": [t_["n_heads"] for t_ in tried[:-1]]}
        p_ = Path(a["path"])
        p_.write_text(json.dumps({k: v for k, v in a.items() if k != "path"}, ensure_ascii=False, indent=1), encoding="utf-8")

    def _postnorm_pipeline_note(self, out: dict) -> None:
        """組にヘッド・層全体(attn)の射影が入り、その層の注意の出力のあとに正規化があれば、結果に知らせ(postnorm)を入れる
        (B の補正が入るまで、射影は式どおりでない。組の数は残差で測ったものなので本物)。警告にも入れる。"""
        from .headproj import POSTNORM_MIXER_NOTE, postnorm_mixer_layers, proj_heads
        rows = [r for r in (out.get("allocate") or {}).get("table") or [] if not r.get("skipped") and r.get("recipe")]
        s1 = (out.get("allocate") or {}).get("s1")
        sure, unsure = set(), set()
        for r in rows + ([s1] if s1 and s1.get("recipe") else []):
            try:
                a_, b_ = postnorm_mixer_layers(self.weights, proj_heads(Recipe.load(r["recipe"])))
            except (GneError, OSError):
                continue
            sure |= set(a_)
            unsure |= set(b_)
        if not sure and not unsure:
            return
        if self.proj_correction:                     # v0.9.0: 補正つき(既定)
            from .postnorm import SIDE_RATIO_WARN
            note = ("注意(混ぜる層)の出力のあとに正規化があるので、射影は正規化を補正して当てます(書き込みから normalize(γ⊙r̂) を引く。"
                    "s=1 でその部品の分をちょうど取り除き、ほかの s では比例せず、大きな s では頭打ち)。この構造のモデルでは、取り除くときに"
                    f"残差が r̂ 以外の向きへも動きやすい(中央値で r̂ に沿った変化の 1 倍前後、{SIDE_RATIO_WARN:g} 倍を超える層は警告)")
            out["postnorm"] = {"layers": sorted(sure), "unsure_layers": sorted(unsure), "note": note, "proj_correction": "postnorm"}
        else:
            out["postnorm"] = {"layers": sorted(sure), "unsure_layers": sorted(unsure), "note": POSTNORM_MIXER_NOTE,
                               "proj_correction": "none"}
        self._warn_once(f"組のヘッド・層全体(attn)の射影のうち、層 {', '.join(map(str, sorted(sure | unsure)[:8]))} は{out['postnorm']['note']}")

    def reply_check(self, cs, val, recipe: str | Path | Recipe | None, *, n: int = 16, max_tokens: int = 512,
                    prompts: dict | None = None) -> dict:
        """組を当てたときと無介入で、対照集合の両側の質問(validate.prompts があれば A 側はそれ。prompts={"A": [...], "B": [...]}
        で差し替え)に greedy で応答を書かせ、書き終えた応答のうち validate.match に一致した数・空・繰り返し・上限で切れた数を並べる。
        側の作り方(system・レシピ)はそのまま当てる。射影の値が目標に届いても応答が変わらないことがあるので、その確かめに使う。"""
        from .behave import loop_like
        from .evalx import _generate
        from .reverse import combine, side_interventions
        t0 = time.perf_counter()
        r = recipe if isinstance(recipe, Recipe) or recipe is None else Recipe.load(self.recipe_path(recipe))
        val_both = val.explicit and not cs.separate_prompts
        pa = (prompts or {}).get("A") or (val.prompts if val.explicit else cs.prompts("A"))
        pb = (prompts or {}).get("B") or (val.prompts if val_both else cs.prompts("B"))

        def thin(v):
            if len(v) <= n:
                return list(v)
            return [v[int(i)] for i in sorted({int(x) for x in np.linspace(0, len(v) - 1, n).round()})]
        pa, pb = thin(pa), thin(pb)
        ivs = side_interventions(self, cs) if cs.has_recipes else {"A": (None, None), "B": (None, None)}
        rx = re.compile(val.match) if val.match else None
        conv = {"A": cs.convs("A", pa), "B": cs.convs("B", pb)}
        out = {"n": {"A": len(pa), "B": len(pb)}, "max_tokens": max_tokens, "match": val.match,
               "held_out": {"A": bool((prompts or {}).get("A")) or val.explicit, "B": bool((prompts or {}).get("B")) or val_both},
               "rows": {}}
        for key, rr in (("none", None), ("recipe", r)):
            vecs = self.vectors(rr)[0] if rr is not None and rr.add_units else None
            lora = self.lora_for(rr) if rr is not None else None
            row = {}
            for side in "AB":
                outs = _generate(self, [(m, max_tokens) for m in conv[side]], *combine(ivs[side], vecs, lora), "all", lambda k: None)
                fin = [o for o in outs if o["finish"] in ("eog", "stop", "eos")]
                row[side] = {"n": len(outs), "finished": len(fin),
                             "match_finished": sum(bool(rx.search(o["text"] or "")) for o in fin) if rx else None,
                             "empty": sum(not (o["text"] or "").strip() for o in outs),
                             "loop": sum(loop_like(o["token_ids"]) for o in outs),
                             "at_limit": len(outs) - len(fin)}
            out["rows"][key] = row
        if rx is not None:                            # A 側(振る舞いが出ている側)の一致がほとんど減らないときの知らせ
            a0, a1 = out["rows"]["none"]["A"], out["rows"]["recipe"]["A"]
            out["a_changed"] = (a0["match_finished"] or 0) - (a1["match_finished"] or 0)
            out["note"] = None if out["a_changed"] >= max(1, len(pa) // 6) else \
                "射影の値は目標に届いても、A 側の応答の一致はほとんど減っていません(振る舞いが変わっていない可能性)"
            # 書き込みの量で選ぶと効かず、v0.7.0 までの選び方(基準層の手前の層のヘッド)が効く題材がある(Qwen3.5-4B の disclaimer)
            out["suggest"] = None if out["note"] is None else "rank causal"
            if a0["finished"] and (a0["match_finished"] or 0) < 0.5 * a0["finished"]:
                out["match_warning"] = (f"無介入の A 側で validate.match に一致したのは {a0['match_finished'] or 0}/{a0['finished']} だけです。"
                                        "正規表現が応答に合っていないか、モデルが A 側の振る舞いをしていない疑いがあり、この確かめの数は当てになりません")
        # 組を当てると応答が壊れる(繰り返し・空が増える)とき。射影の値・B 側のずれは最後の位置の 1 点でしか測らないので、ここで見る
        brk = reply_breaks(out["rows"])
        if brk:
            out["break_warning"] = brk
        out["seconds"] = time.perf_counter() - t0
        return out

    def heads_writes(self, direction: str, *, contrast: str | Path | None = None, select_layer: int | None = None,
                     kinds="all", positions: str = "last", subset: str = "all", max_examples: int | None = None,
                     verify_top: int = 64, verify_sample: int = 16, verify_mode: str = "proj", verify_examples: int | None = 64,
                     score_layers: list[int] | None = None, direction_layer: str = "own", name: str | None = None,
                     top_q: int = 256, loop_direction: str = "first", save_q: bool = False) -> dict:
        """全部の部品(注意ヘッド・線形注意のヘッド・MLP のニューロン。kinds に layers を足すと層の書き込み全体も)を、方向への
        書き込みの量で順位づけし、上位を 1 個ずつ
        s=1 で射影して(verify_mode ablate なら止めて)確かめる(gne/unitwrites.py)。結果は <root>/heads/<model_id>/<name>.writes.json
        と .writes.npz(層ごとの更新のベクトル・部品ごとの書き込み)。verify_top 0 で確かめを飛ばす。save_q なら、質問ごと・層ごとの
        更新を別のファイル <name>.writes_q.npz に保存する(gne heads geometry の質問に沿った分解・ラベル入れ替えの基準の材料)。"""
        from .causal import prepare
        from .direction import base_layer_of, score_layers_default
        from .unitwrites import measure_writes, parse_kinds, save_writes, verify_rows, verify_units
        t0 = time.perf_counter()
        ds, cs, _ = self._direction_contrast(direction, contrast)
        name = name or ds.name
        if not re.fullmatch(r"[\w.\-]+", name):
            raise GneError("BAD_ARGS", f"名前 {name!r} は英数字・_・-・. だけにしてください")
        self.engine.free_steer()
        S = int(select_layer) if select_layer is not None else self.weights.n_main - 2
        summ, arrays = measure_writes(self, ds, cs, select_layer=S, positions=positions, subset=subset,
                                      max_examples=max_examples, kinds=parse_kinds(kinds), direction_layer=direction_layer,
                                      top_q=top_q, loop_direction=loop_direction, keep_q=bool(save_q), progress=self.progress)
        qarr = {k: arrays.pop(k) for k in [k for k in arrays if k.startswith("q_")]}   # .writes.npz の形は変えない
        if qarr:                                       # 質問の文を引くための対照集合(geometry が同じものか sha で確かめる)
            src = Path(contrast).resolve() if contrast else ds.meta.get("source")
            qarr.update(q_contrast=np.array(str(src or "")), q_contrast_sha=np.array(str(cs.sha)))
        ver = None
        if verify_top and verify_top > 0:
            b = base_layer_of(ds)
            Ls = sorted(set(score_layers or (score_layers_default(self, ds, b) if b is not None else [])) | {S})
            Ls = [L for L in Ls if L <= S] or [S]
            st = prepare(self, ds, cs, score_layers=Ls, primary=S, positions=positions, subset=subset,
                         direction_layer=direction_layer, max_examples=verify_examples)
            ver = verify_units(self, st, verify_rows(summ["rows"], int(verify_top), int(verify_sample)), mode=verify_mode,
                               progress=self.progress)
        pz, pj = save_writes(self, name, summ, arrays, ver)
        pq = None
        if qarr:
            from .geometry import save_q
            pq = save_q(self.store.heads_dir(self.meta.model_id), name, qarr)
        return {**{k: v for k, v in summ.items() if k != "rows"}, "rows": summ["rows"][:200], "verify": ver, "name": name,
                "path": str(pj), "npz": str(pz), **({"q_npz": str(pq), "q_bytes": pq.stat().st_size} if pq else {}),
                "seconds": time.perf_counter() - t0}

    def _refine_chain(self, name: str, row: dict) -> dict:
        """残りから追加の土台。name が一括の結果なら土台はその行の組(0 回目 = 元の組)。name が残りから追加の結果なら、
        土台はその行の「元の組と合わせたレシピ」(元の組 + それまでの回の追加の全部)。戻り値 {root, round, recipe, rounds}
        (root = 元の一括の名前、round = 今回の回の番号、recipe = 土台のレシピのパス、rounds = 土台の部品ごとの何回目に入ったか)。"""
        hd = self.store.heads_dir(self.meta.model_id)
        rp = hd / f"{name}.refine.json"
        if not rp.exists():
            r0 = Recipe.load(row["recipe"])
            return {"root": name, "round": 1, "recipe": row["recipe"], "rounds": [0] * len(r0.units), "chain": [name]}
        ref = json.loads(rp.read_text(encoding="utf-8"))
        key = "s1" if row.get("s1") else str(row["n_heads"])
        c = next((x for x in ref.get("combined_rows") or [] if str(x.get("key") or x.get("n_new")) == key), None)
        if c is None:
            raise GneError("BAD_ARGS", f"{name} の {key} の行には、元の組と合わせたレシピがありません",
                           "元の一括の結果の行から、残りから追加をやり直してください")
        if "round" not in ref:                     # v0.8.3 までの結果: 1 回目(土台が一括の結果)の合わせたレシピだけが正しい
            base = ref.get("base") or ""
            if (hd / f"{base}.refine.json").exists():
                root = base
                for _ in range(64):                 # 元の一括の結果までたどる
                    f_ = hd / f"{root}.refine.json"
                    nxt = json.loads(f_.read_text(encoding="utf-8")).get("base") if f_.exists() else None
                    if not nxt or nxt == root:
                        break
                    root = nxt
                raise GneError("BAD_ARGS", f"{name} は v0.8.3 までに残りから追加を重ねた結果で、合わせたレシピに前の回の部品が入っていません",
                               f"元の一括の結果 {root} の行から、残りから追加をやり直してください")
            nb = len(ref.get("base_units") or [])
            return {"root": base, "round": 2, "recipe": c["recipe"], "chain": [base, name],
                    "rounds": [0] * nb + [1] * (len(Recipe.load(c["recipe"]).units) - nb)}
        return {"root": ref["root"], "round": int(ref["round"]) + 1, "recipe": c["recipe"], "chain": list(ref.get("chain") or [ref["root"]]) + [name],
                "rounds": [int(p_["round"]) for p_ in c.get("parts") or []] or [0] * len(Recipe.load(c["recipe"]).units)}

    def heads_refine(self, name: str, *, n_heads: int | None = None, new_heads=(3, 2, 1), out_name: str | None = None,
                     min_questions: int = 4, resume: bool = True, rank: str | None = None, unit_kinds: str | None = None,
                     check_replies: int | None = None, check_tokens: int | None = None) -> dict:
        """配分の組を当てても目標まで下がらない A 側の質問(「残り」)だけで、組に足す部品を探す。
        土台: name が一括の結果ならその行の組。name が残りから追加の結果なら、その行の「元の組と合わせたレシピ」(元の組 + それまでの
        回の追加の全部)。どの回も、元の一括の方向・対照集合・目標(配分の remain_threshold)に戻って測る。
        1. 土台を当てて元の A 側の全部の質問を測り、射影の値を測る層での射影の値が目標を超える質問を残りとする。
        2. 新しい対照集合を作る: A = 残りの質問、B = 元の B 側の質問。両側に元の側の作り方(system・レシピ)と土台の全部を当てる
           (土台を当てたうえでまだ残っている違いを方向にする)。validate は元のもの、一括の設定は元の一括のもの。
        3. それを heads pipeline に流す(方向 <name>_rK → 基準層 → 因果 → 配分)。見つかるのは、土台に足す部品と強さ
           (rank writes の部品ごとの強さは、足す組だけに効く。土台の強さはそのまま)。
        4. 比較表のどの組も(rank writes なら倍率を上げる前の組 s1 も)、土台と合わせたレシピ(<name>_rK.combined-n<N>.yaml・
           .combined-s1.yaml)を作り、元の方向・元の目標で、残り・A 平均・B Δ・B ⊥・組の効き(ループのモデルは 1 回目の出口の段も)を、
           土台だけのとき(前)と合わせたとき(後)で測る。足した組は土台を当てた状態で探したので、単独では使わない。
           代表(推す組、無ければ合わせた残りが最も少ない組)は <name>_rK.combined.yaml にも置く。
        5. 代表の合わせたレシピで、元の一括と同じ応答での確かめ(check_replies 問、既定は元の一括の設定、0 で行わない)。
        残りが min_questions 個より少なければ、射影の値の高い順に min_questions 個まで足して方向を作る(残りの数はそのまま報告)。
        rank・unit_kinds は足す部品の探し方(省くと前の回の設定: 元が rank writes なら同じ流れで、ニューロンも足せる。
        rank writes では new_heads は組の部品の数の上限 n_units になる)。"""
        import dataclasses
        import shutil

        import yaml

        from .direction import Scorer, drift, load_contrast
        from .unitwrites import kind_of_layer
        t0 = time.perf_counter()
        hd = self.store.heads_dir(self.meta.model_id)
        ap = hd / f"{name}.alloc.json"
        if not re.fullmatch(r"[\w.\-]+", str(name)) or not ap.exists():
            raise GneError("FILE_NOT_FOUND", f"配分の結果 {name} がありません", "gne heads pipeline か gne heads allocate で作ってください")
        al = json.loads(ap.read_text(encoding="utf-8"))
        rows = [r for r in al["table"] if not r.get("skipped")]
        if not rows:
            raise GneError("BAD_ARGS", f"配分の結果 {name} に使える組がありません")
        rfp = hd / f"{name}.refine.json"
        ref0 = json.loads(rfp.read_text(encoding="utf-8")) if rfp.exists() else {}
        if n_heads is None and "round" in ref0:     # 残りから追加の結果に重ねる: 既定は合わせたレシピの数で推した行(v0.8.3 までの結果は従来どおり)
            key = next((str(c.get("key") or c.get("n_new")) for c in ref0.get("combined_rows") or [] if c.get("recommended")), None)
            if key is None or key == "s1":
                raise GneError("BAD_ARGS", f"{name} には重ねる土台に推す行がありません"
                               + ("(どの行も足すと良くならない)" if key is None else "(推す行が倍率を上げる前の組 s1)"),
                               "重ねる行を n_heads(表の行の上限の数)で選んでください")
            n_heads = int(key)
        if n_heads is None:
            row = next((r for r in rows if r.get("recommended")), rows[0])
        else:
            row = next((r for r in rows if int(r["n_heads"]) == int(n_heads)), None)
            if row is None:
                raise GneError("BAD_ARGS", f"{n_heads} ヘッドの組がありません({', '.join(str(r['n_heads']) for r in rows)})")
        ch = self._refine_chain(name, row)
        root, rnd = ch["root"], ch["round"]
        rap = hd / f"{root}.alloc.json"
        if not rap.exists():
            raise GneError("FILE_NOT_FOUND", f"元の一括の配分の結果 {root} がありません")
        ral = al if root == name else json.loads(rap.read_text(encoding="utf-8"))
        prior_r = Recipe.load(ch["recipe"])
        explicit = lambda r: [dataclasses.replace(u, direction=u.direction or r.default_direction) if u.mode == "proj" else u
                              for u in r.units]
        prior_units = explicit(prior_r)
        rounds0 = list(ch["rounds"])
        if len(rounds0) != len(prior_units):
            rounds0 = (rounds0 + [rnd - 1] * len(prior_units))[:len(prior_units)]
        ds, cs, val = self._direction_contrast(ral["direction"], None)   # 元の方向・元の対照集合
        S, thr, pos = int(ral["score_layer"]), float(ral.get("remain_threshold") or 0.0), ral.get("positions") or "last"
        X = ral.get("loop_exit")
        X = int(X) if X is not None and int(X) != S else None
        Ls = sorted({S} | ({X} if X is not None else set()))
        self.engine.free_steer()
        sc = Scorer(self, ds, cs, layers=Ls, positions=pos)          # A 側は間引かずに全部の質問
        scb = Scorer(self, ds, cs, layers=Ls, positions=pos, max_examples=ral.get("max_examples") or 64)

        def measure(rec):
            lora = self.lora_for(rec) if rec is not None else None
            vecs = self.vectors(rec)[0] if rec is not None and rec.add_units else None
            return sc.run(lora, vecs, sides="A")["A"]["values"], scb.run(lora, vecs, sides="B", keep=True)["B"]
        a0, b0 = measure(None)
        a0m, b0m = float(a0[S].mean()), float(b0["values"][S].mean())
        sign = 1.0 if a0m >= b0m else -1.0

        def summary(a, b) -> dict:
            va, bm = a[S], float(b["values"][S].mean())
            out_ = {"remain": int((va > thr).sum()), "a_mean": float(va.mean()), "a_max": float(va.max()), "b_mean": bm,
                    "b_delta": bm - b0m, "b_drift_orth": drift(b0["resid"][S], b["resid"][S], sc.r[S])[1],
                    "effect": sign * (a0m - float(va.mean())) - abs(bm - b0m), "reached": bool(float(va.mean()) <= thr + 1e-6)}
            if X is not None:
                out_["exit"] = {"layer": X, "a_mean": float(a[X].mean()), "a_mean_none": float(a0[X].mean()),
                                "b_delta": float(b["values"][X].mean()) - float(b0["values"][X].mean()),
                                "b_drift_orth": drift(b0["resid"][X], b["resid"][X], sc.r[X])[1]}
            return out_
        a, b_prior = measure(prior_r)
        before = summary(a, b_prior)
        a = a[S]
        # 元の一括が選ぶ質問と測る質問を分けていたら、合わせたレシピの数を選ぶのに使っていない質問(測る側)でも測る(元の表と同じ並べ方)
        ho_root = ral.get("holdout") or {}
        hmeasure = hsum = None
        if ho_root.get("contrast") and Path(ho_root["contrast"]).exists():
            cs_h, _ = load_contrast(ho_root["contrast"])
            sch = Scorer(self, ds, cs_h, layers=[S], positions=pos)
            schb = Scorer(self, ds, cs_h, layers=[S], positions=pos, max_examples=ral.get("max_examples") or 64)

            def hmeasure(rec):
                lora = self.lora_for(rec) if rec is not None else None
                vecs = self.vectors(rec)[0] if rec is not None and rec.add_units else None
                return sch.run(lora, vecs, sides="A")["A"]["values"][S], schb.run(lora, vecs, sides="B", keep=True)["B"]
            ha0, hb0 = hmeasure(None)
            ha0m, hb0m, thr_h = float(ha0.mean()), float(hb0["values"][S].mean()), float(ho_root.get("target", thr))

            def hsum(va, b) -> dict:
                bm = float(b["values"][S].mean())
                return {"remain": int((va > thr_h).sum()), "a_mean": float(va.mean()), "b_delta": bm - hb0m,
                        "b_drift_orth": drift(hb0["resid"][S], b["resid"][S], sch.r[S])[1],
                        "effect": (1.0 if ha0m >= hb0m else -1.0) * (ha0m - float(va.mean())) - abs(bm - hb0m),
                        "reached": bool(float(va.mean()) <= thr_h + 1e-6)}
            hbefore = hsum(*hmeasure(prior_r))
        order = [int(i) for i in np.argsort(-a)]
        rem = [i for i in order if a[i] > thr]
        what = f"{row['n_heads']} 個の組" + (f"(元の組 + {rnd - 1} 回分の追加)" if rnd > 1 else "")
        if not rem:
            raise GneError("NOTHING_LEFT", f"{what}で、A 側の全部の質問({len(a)} 問)が目標 {thr:.2f} まで下がっています",
                           "追加の部品は要りません")
        pick = rem if len(rem) >= min_questions else order[:max(min_questions, len(rem))]
        qa = [sc.prompts_side["A"][i] for i in pick]
        k = 1
        while out_name is None:                    # 同じ組からの追加は同じ名前(流し直すと方向・因果を使い回す)、別の組なら次の番号
            cand = f"{name}_r{k}"
            f = hd / f"{cand}.refine.json"
            try:
                same = resume and json.loads(f.read_text(encoding="utf-8")).get("n_heads") == int(row["n_heads"])
            except (OSError, ValueError):
                same = False
            if not f.exists() or same:
                out_name = cand
            k += 1
        if not re.fullmatch(r"[\w.\-]+", out_name):
            raise GneError("BAD_ARGS", f"名前 {out_name!r} は英数字・_・-・. だけにしてください")

        def side_recipe(side: str) -> dict:
            r0 = cs.recipe(self, side)                                  # 元の対照集合の側のレシピ(土台とは重ならない)
            merged = (r0.like(list(r0.units) + prior_units, name=f"{out_name}-{side}") if r0 is not None
                      else prior_r.like(prior_units, name=f"{out_name}-{side}"))
            tmp = hd / f".{out_name}-{side}.tmp.yaml"
            merged.save(tmp)
            d = yaml.safe_load(tmp.read_text(encoding="utf-8"))
            tmp.unlink(missing_ok=True)
            d.pop("name", None)
            return d
        raw = yaml.safe_load(Path(str(ds.meta.get("source"))).read_text(encoding="utf-8")) or {}

        def settings_of(n: str) -> dict:
            pj_ = hd / f"{n}.pipeline.json"
            return (json.loads(pj_.read_text(encoding="utf-8")).get("settings") or {}) if pj_.exists() else {}
        st, st_prev = settings_of(root), settings_of(name)
        pipe = {k_: st[k_] for k_ in ("positions", "score_positions", "top_layers", "max_examples", "s_max", "direction_layer",
                                      "loop_direction") if st.get(k_) is not None}
        pipe.setdefault("score_positions", pos)
        if isinstance(st.get("target"), str):                           # B / mid は、残りの一括でも同じ意味
            pipe["target"] = st["target"]
        rk = rank or st_prev.get("rank") or st.get("rank") or "causal"
        rank_note = None
        if rank is None and rk == "writes":                             # 前の回が writes でも、出力のあとに正規化があるモデルでは止めて選ぶ
            probe = {"rank": "writes"}
            rank_note = self._rank_fallback(probe, False)
            if rank_note:
                rk = "causal"
                self._warn_once(rank_note)
        if rk == "writes":                                              # 新しい選び方で足す部品を探す(ニューロンも選べる)
            pipe.pop("top_layers", None)
            pipe.update(rank="writes", unit_kinds=unit_kinds or st_prev.get("unit_kinds") or st.get("unit_kinds") or "heads",
                        n_units=[int(x) for x in new_heads], check_replies=0,
                        **{k_: st[k_] for k_ in ("verify_top", "verify_sample", "verify_mode", "select_layer", "strength")
                           if st.get(k_) is not None})
            if st.get("hard") is not None:
                pipe["hard"] = max(1, min(int(st["hard"]), len(qa)))
        elif rk == "causal":                                            # 元が v0.7.0 までの流れ(rank の記録が無い古い結果も)
            if unit_kinds not in (None, "heads"):
                why = self._writes_block_reason()
                if why:                                                 # 書き込みの量で選べないモデル: 行き止まりにしない
                    until = _writes_block_until(why)
                    raise GneError("ARCH_UNSUPPORTED", f"このモデルでは、ニューロンや層全体を足す選び方は使えません。理由: {why}ためです"
                                   "(ニューロン・層全体を足せるのは、書き込みの量で選ぶ流れ --rank writes だけです)" + (f"。{until}" if until else ""),
                                   "ヘッドだけで足すには --unit-kinds を外してください(止めて選ぶ流れで足します)")
                raise GneError("BAD_ARGS", "ニューロンや層全体を足せるのは、書き込みの量で選ぶ流れ(--rank writes)だけです",
                               "--rank writes を付けてください。ヘッドだけで足すなら --unit-kinds を外してください")
            pipe.pop("loop_direction", None)
            pipe.update(rank="causal", n_heads=[int(x) for x in new_heads], hard=max(1, min(int(st.get("hard") or 16), len(qa))))
        else:
            raise GneError("BAD_ARGS", f"rank は causal か writes です({rk!r})")
        spec = {"name": out_name, **({"system_a": cs.system_a} if cs.system_a is not None else {}),
                **({"system_b": cs.system_b} if cs.system_b is not None else {}),
                "recipe_a": side_recipe("A"), "recipe_b": side_recipe("B"), "prompts_a": qa, "prompts_b": cs.prompts("B"),
                **({"validate": raw["validate"]} if raw.get("validate") else {}), "pipeline": pipe}
        p = self.heads_pipeline(spec, resume=resume)

        def proj_spec(u) -> str:                    # レシピの欄・UI の部品の書き方(L:Hh:sS@方向 など)
            who = {"neuron": f"N{u.neuron}", "attn": "attn", "ffn": "ffn"}.get(u.kind, f"H{u.head}")
            return f"{u.layer}:{who}:s{float(u.k):.4g}@{u.direction}"

        def parts_of(units, rounds) -> list[dict]:
            out_ = []
            for u, r_ in zip(units, rounds, strict=True):
                kind = {"neuron": "neuron", "attn": "layer_attn", "ffn": "layer_mlp"}.get(u.kind)
                if kind is None:
                    try:
                        kind = kind_of_layer(self, u.layer)
                    except (AttributeError, IndexError):
                        kind = "attn"
                out_.append({"spec": proj_spec(u), "layer": int(u.layer), "head": u.neuron if u.kind == "neuron" else u.head,
                             "kind": kind, "s": float(u.k), "direction": u.direction, "round": int(r_)})
            return out_

        plan_ = getattr(self.weights, "loop", None)
        shared_ = plan_ is not None and not getattr(self.weights, "per_visit_names", False)

        def cross_of(parts) -> list[dict]:
            """同じ層を 2 回実行するモデルで、土台の部品と今回足した部品が、同じ重み(保存されている層・同じ行列)を別の回の段で使う組。
            探したときは土台と足した組を別々の LoRA で足し合わせたが、合わせたレシピでは 2 つの射影を続けて掛ける(headproj の _build)ので、
            効き方が違う(強さが 1 を超えると大きく違うことがある)。表の数は合わせたレシピで測った値。"""
            if not shared_:
                return []
            mat = lambda q: "neuron" if q["kind"] == "neuron" else ("ffn" if q["kind"] == "layer_mlp" else "attn")   # noqa: E731
            out_ = []
            for b_ in [q for q in parts if q["round"] < rnd]:
                for a_ in [q for q in parts if q["round"] == rnd]:
                    if plan_.stored(b_["layer"]) == plan_.stored(a_["layer"]) and b_["layer"] != a_["layer"] and mat(b_) == mat(a_):
                        out_.append({"base": b_["spec"], "added": a_["spec"], "stored": int(plan_.stored(a_["layer"]))})
            return out_

        def dups_of(parts) -> list[dict]:           # 同じ部品(層・種類・番号)が 2 つ以上の方向で入っている
            g: dict[tuple, list[dict]] = {}
            for q in parts:
                g.setdefault((q["layer"], q["kind"], q["head"]), []).append(q)
            return [{"layer": k_[0], "kind": k_[1], "head": k_[2], "rounds": [q["round"] for q in v], "s": [q["s"] for q in v],
                     "directions": [q["direction"] for q in v]} for k_, v in g.items() if len(v) > 1]
        alloc = p["allocate"]
        rows_new = [(str(r["n_heads"]), int(r["n_heads"]), r) for r in alloc["table"] if not r.get("skipped") and r.get("recipe")]
        if (alloc.get("s1") or {}).get("recipe"):
            rows_new.append(("s1", None, alloc["s1"]))
        comb_rows = []
        for key, n_new, r in rows_new:
            new_units = explicit(Recipe.load(r["recipe"]))
            cr = prior_r.like(prior_units + new_units, name=f"{out_name}-{'s1' if n_new is None else f'n{n_new}'}-combined",
                              proj_correction=self.proj_correction_label(prior_r))     # 合わせたレシピは当てた方を記録する
            cpath = hd / f"{out_name}.combined-{'s1' if n_new is None else f'n{n_new}'}.yaml"
            cr.save(cpath)
            a2, b2 = measure(cr)
            after = summary(a2, b2)
            hrow = None
            if hmeasure is not None:
                hafter = hsum(*hmeasure(cr))
                hrow = {"n_a": int(len(ha0)), "target": thr_h, "remain_before": hbefore["remain"], "remain_after": hafter["remain"],
                        "a_mean_before": hbefore["a_mean"], "a_mean_after": hafter["a_mean"], "b_delta_base": hbefore["b_delta"],
                        "b_delta": hafter["b_delta"], "b_drift_orth_base": hbefore["b_drift_orth"], "b_drift_orth": hafter["b_drift_orth"],
                        "effect_base": hbefore["effect"], "effect": hafter["effect"], "reached": hafter["reached"]}
            parts = parts_of(prior_units, rounds0) + parts_of(new_units, [rnd] * len(new_units))
            comb_rows.append({"key": key, "n_new": n_new, "s1": n_new is None, "recipe": str(cpath),
                              "units": [q["spec"] for q in parts], "parts": parts, "dups": dups_of(parts), "cross_visit": cross_of(parts),
                              "n_heads": len(parts), "n_base": len(prior_units), "n_added": len(new_units),
                              "recommended_added": bool(r.get("recommended")), "recommended": False, "reached": bool(r.get("reached")),
                              "reached_combined": after["reached"],
                              "remain_before": before["remain"], "remain_after": after["remain"],
                              "a_mean_before": before["a_mean"], "a_mean_after": after["a_mean"],
                              "a_max_before": before["a_max"], "a_max_after": after["a_max"],
                              "b_mean": b0m, "b_mean_base": before["b_mean"], "b_mean_after": after["b_mean"],
                              "b_delta_base": before["b_delta"], "b_delta": after["b_delta"],
                              "b_drift_orth_base": before["b_drift_orth"], "b_drift_orth": after["b_drift_orth"],
                              "effect_base": before["effect"], "effect": after["effect"],
                              **({"exit_before": before["exit"], "exit_after": after["exit"]} if X is not None else {}),
                              "added_only": {k_: r.get(k_) for k_ in ("remain", "a_mean", "b_side_delta", "b_drift_orth", "effect", "reached")},
                              **({"holdout": hrow} if hrow is not None else {})})
        # 推す行: 使うのは合わせたレシピなので、元の方向・元の目標で測った合わせたレシピの数で選ぶ(規則は refine_verdicts)
        order = refine_verdicts(comb_rows, n_a=int(len(a)), gap=abs(a0m - b0m) or 1.0, sign=sign, threshold=thr)
        n_chk = check_replies if check_replies is not None else st.get("check_replies")
        n_chk = 16 if n_chk is None else int(n_chk)
        cs_chk, val_chk, ho = cs, val, {}
        if n_chk > 0:
            rpj = hd / f"{root}.pipeline.json"
            ho = (json.loads(rpj.read_text(encoding="utf-8")).get("holdout") or {}) if rpj.exists() else {}
            if ho.get("contrast") and Path(ho["contrast"]).exists():    # 元の一括が選ぶ質問と測る質問を分けていたら、測る質問で
                cs_chk, val_chk = load_contrast(ho["contrast"])
        comb, check, why_none = None, None, ("no_better" if not order else None)
        for c in order[:3]:                         # 5. 推す行の合わせたレシピで、元の一括と同じ応答での確かめ。壊れていれば次の行
            if n_chk <= 0:
                comb = c
                break
            ck = self.reply_check(cs_chk, val_chk, c["recipe"], n=n_chk, max_tokens=int(check_tokens or st.get("check_tokens") or 512))
            ck.update(key=c["key"], n_heads=c["n_heads"], held_out_contrast=bool(ho.get("contrast")))
            c["check"] = {k_: ck.get(k_) for k_ in ("rows", "break_warning", "match_warning", "note")}
            check = ck
            if ck.get("break_warning"):
                c["verdict"], c["why"] = "replies_break", list(c.get("why") or []) + ["replies_break"]
                continue
            comb = c
            break
        if comb is None and order:
            why_none = "replies_break"
        stale = hd / f"{out_name}.combined.yaml"
        if comb is not None:
            comb["recommended"] = True
            comb = {**comb, "chosen_by": "combined"}
            shutil.copyfile(comb["recipe"], stale)   # 代表は <名前>.combined.yaml にも置く
            if check is not None:
                check["recommended"] = True
        else:
            stale.unlink(missing_ok=True)           # 推す行が無い(足さないほうがよい): 前の代表を残さない
        by_round: dict[int, int] = {}
        for q in (comb or {}).get("parts") or parts_of(prior_units, rounds0):
            by_round[q["round"]] = by_round.get(q["round"], 0) + 1
        out = {"name": out_name, "base": name, "root": root, "round": rnd, "chain": ch["chain"] + [out_name],
               "direction": ds.name, "n_heads": int(row["n_heads"]), "base_units": [proj_spec(u) for u in prior_units],
               "base_parts": parts_of(prior_units, rounds0), "base_recipe": ch["recipe"], "base_measure": before,
               "none_measure": {"a_mean": a0m, "b_mean": b0m}, "score_layer": S, "threshold": thr, "loop_exit": X,
               "rank": rk, **({"rank_note": rank_note} if rank_note else {}), **({"postnorm": p["postnorm"]} if p.get("postnorm") else {}),
               "units_by_round": {str(k_): v for k_, v in sorted(by_round.items())},
               "remaining": {"n": len(rem), "n_a": int(len(a)), "used": len(pick),
                             "padded": len(pick) > len(rem), "questions": qa, "scores": [float(a[i]) for i in pick]},
               "pipeline": {"path": p["path"], "direction": out_name, "recommended": p.get("recommended"),
                            "reused": p.get("reused") or [],
                            "table": [{k_: r.get(k_) for k_ in ("n_heads", "units", "s_max", "remain", "reached", "recommended", "skipped")}
                                      for r in alloc["table"]]},
               "combined": comb, "combined_rows": comb_rows, "check": check, "no_recommend": why_none,
               "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": time.perf_counter() - t0}
        (hd / f"{out_name}.refine.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        pd = json.loads(Path(p["path"]).read_text(encoding="utf-8"))      # 一括の結果にも、どの組の残りから作ったかを書く
        pd["refine"] = {k_: out[k_] for k_ in ("base", "root", "round", "chain", "direction", "n_heads", "base_units", "base_parts",
                                               "base_measure", "none_measure", "threshold", "score_layer", "loop_exit", "units_by_round",
                                               "remaining", "combined", "combined_rows", "check", "no_recommend")}
        Path(p["path"]).write_text(json.dumps(pd, ensure_ascii=False, indent=1), encoding="utf-8")
        out["path"] = str(hd / f"{out_name}.refine.json")
        return out

    # --- 同じ層を 2 回実行するモデル: 1 回目と 2 回目を別々に扱う CHiPS(gne/loopsearch.py) ----------------------------

    def heads_visits(self, direction: str, **kw) -> dict:
        """ヘッドを 1 回目だけ・2 回目だけ・両方で止めた効き目と、回の間の順位相関(2 回目の層に別の名前がある llama.cpp が要る)。"""
        from .loopsearch import visits_screen
        return visits_screen(self, direction, **kw)

    def heads_causal_visit(self, direction: str, visit: str, **kw) -> dict:
        """因果スクリーニングを、ヘッドを 1 回目だけ(first)・2 回目だけ(second)・両方(both)で止めて行う。first / second は
        2 回目の層に別の名前がある llama.cpp が要る。候補の層は、2 回実行する層の、その回の段。"""
        lp = self.weights.loop
        if lp is None:
            raise GneError("ARCH_UNSUPPORTED", "--visits は同じ層を 2 回実行するモデルだけで使えます")
        if visit not in ("first", "second", "both"):
            raise GneError("BAD_ARGS", "--visits は first / second / both / all です")
        rng = {"first": range(lp.lo1, lp.hi1 + 1), "second": range(lp.lo2, lp.hi2 + 1), "both": range(lp.lo1, lp.hi1 + 1)}[visit]
        mode = "shared" if visit == "both" else "per_visit"
        if kw.get("candidate_layers") is None:
            kw["candidate_layers"] = list(rng)
        with self.loop_mode(mode, "片方の回だけヘッドを止める因果スクリーニング"):
            return self.heads_causal(direction, **kw)

    def heads_loop_pipeline(self, spec, **kw) -> dict:
        """1 回目で効くヘッドを選ぶ → それを当てた状態で 2 回目で効くヘッドを選ぶ → 両方の回に当てるレシピに直して測り直す。"""
        from .loopsearch import loop_pipeline
        return loop_pipeline(self, spec, **kw)

    def heads_loop_merge(self, recipe, **kw) -> dict:
        """回ごとに選んだレシピを、両方の回に当てるレシピ(修正なしの llama.cpp で動く形)に直し、測り直す。"""
        from .loopsearch import loop_merge
        return loop_merge(self, recipe, **kw)

    def heads_measure(self, direction: str, recipes: list[str], *, contrast=None, score_layers=None,
                      positions: str = "last", max_examples: int | None = None, tail: int | None = None) -> dict:
        """レシピを当てたときの、測る層ごとの射影の値(A・B の平均、残る質問の数、縮めた割合、B 側のずれ)。"""
        from .loopsearch import measure
        recs: dict = {"none": None}
        for x in recipes:
            p = Path(str(x))
            r = Recipe.load(p if p.exists() else self.recipe_path(str(x)))
            recs[r.name or p.stem] = r
        out = measure(self, direction, recs, contrast=contrast, layers=self.score_layers_of(score_layers),
                      positions=positions, max_examples=max_examples)
        if tail is not None:                       # 最後の段までの変化の切り分け(残差の大きさ・方向の違い・層ごとの書き足し)
            from .loopsearch import tail_decompose
            out["tail"] = tail_decompose(self, direction, recs, from_layer=int(tail), to_layer=max(out["layers"]),
                                         contrast=contrast, positions=positions, max_examples=max_examples)
        return out

    @staticmethod
    def _pipeline_target(spec, bs: dict) -> tuple[float, str]:
        """一括の目標: 数(射影の値を測る最後の層での射影の値)、"B"(B 側の平均、振る舞いを B 側の水準まで消す)、"mid" か省略
        (A と B の平均の真ん中)。返り値は (目標の値, どこから決めたか)。"""
        if spec is None or (isinstance(spec, str) and spec.strip().lower() in ("", "mid", "middle")):
            return 0.5 * (float(bs["A"]) + float(bs["B"])), "A と B の平均の真ん中"
        if isinstance(spec, str) and spec.strip().upper() == "B":
            return float(bs["B"]), "B 側の平均"
        try:
            return float(spec), "settings"
        except (TypeError, ValueError):
            raise GneError("BAD_ARGS", f"target は数か B(B 側の平均)か mid(A と B の真ん中)です({spec!r})") from None

    def fieldtest(self, model: str | Path | None = None, **kw) -> dict:
        """実地テスト(gne/fieldtest.py): 本番と同じ動かし方(llama-server、思考あり)で長い作業・ツール呼び出し・質問集・perplexity を
        測り、against のファイルと比べる。結果は <root>/fieldtest/<name>.json、生成した文章は <name>.replies.json。"""
        from .fieldtest import fieldtest as run
        return run(self, model, **kw)

    def _pipeline_causal_reuse(self, name: str, cs, positions: str, mx: int | None, pipe: dict) -> dict | None:
        """一括の 3 段目(因果)を使い回せるなら、heads_causal と同じ形の結果。同じ対照集合・測る位置・質問の数・方向の層・
        外す閾値・層の数で、層ごとの行があるものだけ(層の数を記録していない古い結果は、同じとみなす)。"""
        from .causal import causal_paths, load_causal
        pj, ps = causal_paths(self, name)
        if not (pj.exists() and ps.exists()):
            return None
        try:
            m = (json.loads(ps.read_text(encoding="utf-8")).get("modes") or {}).get("ablate") or {}
        except (OSError, ValueError):
            return None
        same = (m.get("contrast_sha") == cs.sha and m.get("positions") == positions and m.get("max_examples") == mx
                and m.get("direction_layer") == pipe["direction_layer"] and m.get("drift_max") == 0.5
                and int(m.get("top_layers", pipe["top_layers"])) == int(pipe["top_layers"]) and m.get("layer_rows"))
        want = self.score_layers_of(pipe.get("score_layers"))     # 測る層・選ぶ層を指定したときは、同じ層で測った結果だけ
        if same and want is None and (m.get("score_layers_given") or pipe.get("select_layer") not in (None, "")):
            same = False                                          # 既定の層で測るのに、前の結果は指定した層で測ったもの
        cl = self._layers_of(pipe.get("candidate_layers"))      # 候補の層・いつも測る層・当て方も同じものだけ
        al = pipe.get("always_layers")
        al = None if al in (None, "none") else (self._layers_of(al) or None)
        if same and (m.get("candidate_layers") != cl or m.get("loop_edits") != self.loop_edits_mode()
                     or m.get("always_layers") != al):
            same = False
        if same and want is not None:
            same = [int(x) for x in m.get("score_layers") or []] == want and (
                pipe.get("select_layer") in (None, "") or int(m.get("primary", -1)) == int(pipe["select_layer"]))
        if not same:
            return None
        rows = load_causal(self, name, "ablate")
        if not rows:
            return None
        return {"path": str(pj), "score_layers": m["score_layers"], "primary": m["primary"], "base_scores": m["base_scores"],
                "n_rows": len(rows), "n_a": m["n_a"], "n_b": m["n_b"], "layers": m["layer_rows"],
                "rows": sorted(rows, key=lambda r: -(r.get("causal" if m.get("select_layer") is not None else "causal_max") or 0))}

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
                         scales=(0.5, 1.0, 1.5, 2.0, 3.0), n_heads: list[int] | None = None, max_prompts: int = 24,
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
        from dataclasses import replace

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
        direction, combined_from = al["direction"], None
        rf = ap.with_name(ap.name[:-len(".alloc.json")] + ".refine.json") if ap.name.endswith(".alloc.json") else None
        if rf is not None and rf.exists():
            # 残りから追加の結果: 候補は土台と足した組を合わせたレシピで、元の一括の方向・対照集合の質問で作る(足した組は単独では使わない)
            ref = json.loads(rf.read_text(encoding="utf-8"))
            if "round" not in ref and (rf.with_name(f"{ref.get('base')}.refine.json")).exists():
                raise GneError("BAD_ARGS", f"{al['name']} は v0.8.3 までに残りから追加を重ねた結果で、合わせたレシピに前の回の部品が入っていません",
                               "元の一括の結果の行から、残りから追加をやり直してください")
            comb = {str(c.get("key") or c.get("n_new")): c for c in ref.get("combined_rows") or []}
            rows = [{**r, "units": comb[str(r["n_heads"])]["units"], "recipe": comb[str(r["n_heads"])]["recipe"]}
                    for r in rows if str(r["n_heads"]) in comb]
            if not rows:
                raise GneError("BAD_ARGS", f"{al['name']} の組には、元の組と合わせたレシピがありません")
            root = ref.get("root") or ref.get("base")
            rap = self.store.heads_dir(self.meta.model_id) / f"{root}.alloc.json"
            direction = json.loads(rap.read_text(encoding="utf-8"))["direction"] if rap.exists() else (ref.get("direction") or direction)
            combined_from = al["name"]
        ds, cs, val = self._direction_contrast(direction, contrast)
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
        from .reverse import check_questions, is_multi, turns
        for side, v in (("A", pa), ("B", pb)):
            if not isinstance(v, list) or not v:
                raise GneError("BAD_ARGS", "prompts は質問のリスト(または {A: [...], B: [...]})です。1 問は文字列か会話の履歴")
            check_questions(v, f"prompts の {side} 側")

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
                # 重ねるレシピの keyed(CHANGE-07)は、方向と鍵の既定を先にユニットへ書き込む(候補の既定で上書きされないように)
                w_units = [replace(u, direction=u.direction or wr.default_direction, key=u.key or wr.default_key)
                           if u.mode == "keyed" else u for u in wr.units]
                r = wr.like(w_units + (list(r.units) if r is not None else []),
                            **({"default_direction": r.default_direction, "direction_layer": r.direction_layer,
                                "default_key": r.default_key or wr.default_key} if r is not None else {}))
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
            q = items[i]["prompt"]
            hist = "".join(f"[{m['role'].upper()}] {m['content']}\n" for m in turns(q)) if is_multi(q) else f"[USER] {q}\n"
            lines.append(f"#{k}\n{hist}[REPLY] {body}\n")
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
               **({"combined_from": combined_from} if combined_from else {}),
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

    def headproj_export(self, recipe: Recipe | list[str], out: str | Path, *, bake: bool = False, requant: str | None = None,
                        via_f16: bool = False, imatrix: str | None = None, llama_quantize: str | None = None,
                        verify: bool = False, contrast: str | Path | None = None, strip_meta: bool = False,
                        into: str | Path | None = None, lora_f16: bool = False, name: str | None = None) -> dict:
        """CHANGE-06 E: proj のユニット(と gain)を LoRA(既定)か、残差に書き込む重み行列だけ量子化し直した GGUF に書き出す
        (gne gain と同じ経路)。鍵つきの書き込み(keyed、CHANGE-07)だけのレシピと、proj・keyed(・gain)を混ぜたレシピも受ける。
        requant を省くと q8_0、keyed を含むなら auto(テンソルごと: 元が Q8_0 なら Q8_0 のまま、それ以外は F16。Q8_0 のままでは
        rank 1 の小さな編集の 1 割以上が刻みに埋もれるテンソルだけ F16。gain.auto_qtype・KEYED_KEEP_MIN、レビュー P13、
        verification.md F132・F139)。keyed を含まないレシピの auto は q8_0。
        verify なら焼き込み版と LoRA 版を V26 の 4 項目で比べる(keyed だけなら keyed の方向で測る)。
        name は LoRA の名前の欄(general.name。省くと「<元のモデルの名前> gain <レシピの名前>」、strip_meta では "adapter")。"""
        recipe = self._as_recipe(recipe)
        self._no_nw(recipe, bake=bake)                   # nw は焼き込めない(LoRA には入る)
        if recipe.proj_units:                            # 出力のあとに正規化がある層の射影: 補正の有無ごとの知らせ(v0.9.0)
            from .headproj import proj_heads
            for msg in self.proj_notes(proj_heads(recipe), self.proj_correction_for(recipe), recipe.direction_layer):
                self._warn_once(msg)
        note = self.proj_record_note(recipe)
        if note:
            self._warn_once(note)
        if name is not None and bake:
            raise GneError("BAD_ARGS", "名前の欄の指定(name)は LoRA の書き出しだけです(焼き込みは元のモデルのメタデータをそのまま写す)")
        if not recipe.proj_units and not recipe.gain_units and not recipe.keyed_units and not recipe.nw_units:
            raise GneError("BAD_RECIPE", "proj のユニットがありません(例 --unit 33:H16:s6@NAME、鍵つきの書き込みは 19:ffn:keyed:a1@NAME#KEY)")
        if requant is None:
            requant = "auto" if recipe.keyed_units else "q8_0"
        if requant == "auto" and not recipe.keyed_units:
            requant = "q8_0"
        if requant not in ("auto", "q8_0", "f16", "keep", "patch"):
            raise GneError("BAD_ARGS", f"--requant は auto / q8_0 / f16 / keep です({requant!r})")
        if into is not None and verify and Path(into).resolve() != Path(self.weights.path).resolve():
            raise GneError("BAD_ARGS", "別のモデルに焼き込むときは、LoRA 版との比較(verify)はできません",
                           "焼き込む先のモデルを読み直して eval で確かめてください")
        if bake and into is None and self.weights.gguf.is_split:     # 分割のままでは、同じ型で書き戻す(大きさを変えない)
            from .headproj import bake_proj_split
            rq = requant
            kb = self._keyed_bundle(recipe) if recipe.keyed_units else None
            d = bake_proj_split(self.weights, recipe, out, self.direction_set, requant=rq, warn=self.warn,
                                extra=kb["tensors"] if kb else None, extra_meta={"keyed": kb["meta"]} if kb else None,
                                correction=self.proj_correction_for(recipe))
            d.update(kind="bake", requant=rq, recipe=recipe.to_dict(), proj_correction=self.proj_correction_label(recipe))
            if kb is not None:
                d.update(keyed_layers=kb["layers"], keyed_rank=kb["rank"])
                self._warn_requant(d)
                self._warn_expert_size(d)
        else:
            d = self.gain(recipe, out, bake=bake, requant="q8_0" if via_f16 else requant, via_f16=via_f16,
                          imatrix=imatrix, llama_quantize=llama_quantize, strip_meta=strip_meta, into=into, lora_f16=lora_f16,
                          name=name)
        if verify:
            if not bake:
                raise GneError("BAD_ARGS", "--verify は --bake と一緒に使います(焼き込み版と LoRA 版を比べる)")
            d["verify"] = self.check_proj_bake(recipe, Path(d["path"]), contrast=contrast)
        return d

    # --- 鍵つきの書き込み(CHANGE-07)--------------------------------------------------------------------

    def direction_key(self, contrast: str | Path, **kw) -> dict:
        """対照集合から keyed の鍵を作る(gne direction key)。中身は gne/keyed_build.py の build_keys。
        key_fit="ridge"(と key_c)で、FiSH の位置の鍵(target mlpin・positions pos:<範囲>)を最小二乗の鍵にする(gne/keyed_ridge.py)。"""
        role = kw.pop("role", None)
        if role in ("scene", "one"):                     # 計算で作るニューロン用の、場面の鍵・ほぼ 1 を返す鍵(target mlpin)
            from .nwrite import build_role_key
            if str(kw.pop("target", "mlpin") or "mlpin") != "mlpin":
                raise GneError("BAD_ARGS", "role scene / one の鍵は target mlpin だけです")
            if kw.pop("key_fit", None) not in (None, "mean") or kw.pop("key_c", None) is not None:
                raise GneError("BAD_ARGS", "最小二乗の鍵(key_fit ridge・key_c)は位置の鍵だけです",
                               "場面の鍵(scene)とほぼ 1 を返す鍵(one)の作り方は 1 つです")
            keep = ("layers", "name", "subset", "max_tokens", "pseudo", "iv", "seed", "q_energy", "progress", "gen_cache",
                    "prior_max_tokens")
            extra = sorted(k for k, v in kw.items() if k not in keep and v not in (None, False))
            if extra:
                raise GneError("BAD_ARGS", f"role {role} の鍵では {', '.join(extra)} は使えません")
            return build_role_key(self, contrast, role=role, **{k: v for k, v in kw.items() if k in keep and v is not None})
        if role not in (None, "position"):
            raise GneError("BAD_ARGS", f"role は position / scene / one です({role!r})")
        from .keyed_build import build_keys
        return build_keys(self, contrast, **kw)

    def direction_token(self, tokens, **kw) -> dict:
        """出力層の重み行列のトークンの行から方向を作る(gne direction token。tokendir.token_direction)。tokens はトークンの指定
        (文字か #ID)のリスト。layers・delta(目標のロジットの上がり Δ)・name は必須。minus(引くトークン)・down(逆向き)・
        contrast(r̄ を測る入力)・rbar(r̄ を直接)。"""
        from .tokendir import token_direction
        return token_direction(self, tokens, **kw)

    def direction_token_scan(self, tokens, contrast: str | Path, **kw) -> dict:
        """層の走査(gne direction token-scan。tokendir.token_scan): 層ごとに、狙う側の入力の位置 0 でトークンが 1 位になる最小の Δ と、
        dont_touch の入力に 0.2 倍・0.5 倍を足したときの影響。"""
        from .tokendir import token_scan
        kw.setdefault("progress", self.progress)
        return token_scan(self, tokens, contrast, **kw)

    def prior_fill(self, contrast: str | Path, **kw) -> dict:
        """対照集合の前のやり取りの応答(content: null)を無介入のモデルで埋めた対照集合を書く(gne nw prior。prior.fill_file)。"""
        from .prior import fill_file
        kw.setdefault("progress", self.progress)
        return fill_file(self, contrast, **kw)

    def direction_keys(self, contrast: str | Path, entries, **kw) -> list[dict]:
        """書き込みたい位置だけが違う鍵を何本か、返事を 1 回だけ作って読んで作る(keyed_build.build_keys_multi)。entries は
        [(鍵の名前, positions), …]。ほかの引数は direction_key と同じ。できる鍵は 1 本ずつ作ったときと同じ。"""
        from .keyed_build import build_keys_multi
        return build_keys_multi(self, contrast, entries, **kw)

    def direction_key_experts(self, contrast: str | Path, **kw) -> dict:
        """expert:e の鍵の専門家を選ぶ: 鍵と同じ対照集合・位置で、書き込みたい位置・書き込みたくない位置で各専門家が選ばれた
        割合と重み(keyed_build.expert_usage。層ごとに書き込みたい位置で選ばれた割合の大きい順)。MoE でない層は BAD_RECIPE。"""
        from .keyed_build import expert_usage
        return expert_usage(self, contrast, **kw)

    def direction_key_check(self, key: str, **kw) -> dict:
        """鍵の確かめ(gne direction key-check: 層ごとの AUC・ρ の分布・比較用の鍵との比較・保留集合)。中身は keyed_build.key_check。"""
        from .keyed_build import key_check
        return key_check(self, key, **kw)

    def direction_key_from_behave(self, behave: str | Path, **kw) -> dict:
        """CHiPS のレシピを当てても消したい振る舞いが残った A 側の質問(gne behave の結果)を集めて、keyed の対照集合の JSON を
        <root>/keyed/<name>.json に書く(gne direction key-from-behave)。モデルは動かさない。中身は gne/keyed_from_behave.py の
        from_behave(引数 recipe・direction・contrast・labels・match・min_rate・positions・name・out・replies・prompts・strict)。"""
        from .keyed_from_behave import from_behave
        return from_behave(self, behave, **kw)

    def direction_key_pipeline(self, contrast: str | Path | dict, **kw) -> dict:
        """方向(gne direction --positions response と同じ)→ 鍵 → 作成と反対側の subset での key-check を 1 回で回す。
        contrast は対照集合のパスか表(表は <root>/contrasts/<name>.yaml に書く。about と keyed の設定は書かない)。
        同じ名前の方向があれば使い回し、rebuild_direction=True で作り直す。中身は gne/keyed_pipeline.py の run。"""
        from .keyed_pipeline import run
        return run(self, contrast, **kw)

    def direction_key_causal(self, contrast: str | Path | dict, **kw) -> dict:
        """keyed の効く場所を探す第 1 段(gne direction key-causal): 候補(層 × 書き込み先)ごとに鍵を作り(subset even)、
        作成に使っていない奇数番目の返事で、keyed を当てたときの振る舞いの部分の対数確率の下がり方(と probe の答えの差の変化)・
        副作用・比較用の鍵の値を生成なしで測る。比較用の鍵の 5% 点より下がった候補に印を付け、偶然でも印が付く数を並べる。
        生成なしの目安なので、効くかどうかは第 2 段(direction_key_causal_verify)で数える。中身は gne/keyed_causal.py の key_causal。"""
        from .keyed_causal import key_causal
        return key_causal(self, contrast, **kw)

    def direction_key_causal_verify(self, units: list[str] | str, **kw) -> dict:
        """第 2 段(gne direction key-causal-verify): 選んだ候補(keyed のユニット)と、それぞれの比較用の鍵 1 本で、作成に使って
        いない質問に behave と同じ生成をさせ、振る舞いが出た数を iv だけのときと比べる。中身は keyed_causal.key_causal_verify。"""
        from .keyed_causal import key_causal_verify
        return key_causal_verify(self, units, **kw)

    def key_causal_results(self) -> list[dict]:
        """保存した key-causal の結果(<root>/keys/<model_id>/causal/)の要約。"""
        from .keyed_causal import list_results
        return list_results(self)

    def keys(self) -> list[dict]:
        """<root>/keys/<model_id>/ の鍵の一覧と要約(層・書き込み先・kn・作成の条件、パスなし)。読めないファイルは error だけ。"""
        from .keyed import list_keys
        return list_keys(self)

    def check_keyed(self, L: int, *, key: str, direction: str, **kw) -> dict:
        """V30: keyed の LoRA の出力の差 = α (κᵀx) d(|κᵀx| の大きいトークンで相対誤差の中央値 < 3%)、鍵の作成に使った
        書き込みたくない側の例では |Δ̄| / (|α|·‖d‖) < 0.1。keyed.check_keyed。"""
        from .keyed import check_keyed
        return check_keyed(self, L, key=key, direction=direction, **kw)

    def default_export_path(self, recipe: Recipe, *, bake: bool = False, kind: str = "headproj",
                            name: str | None = None) -> Path:
        """書き出しの既定のファイル名(<root>/export/)。LoRA は <名前>.lora.gguf、焼き込みは <名前>-<kind>.gguf
        (kind は headproj か gain)。keyed だけのレシピは <名前>-keyed.lora.gguf / <名前>-keyed.gguf(CHANGE-07)。"""
        nm = name or recipe.name or kind
        keyed_only = bool(recipe.keyed_units) and all(u.mode == "keyed" for u in recipe.units)
        if keyed_only:
            stem = f"{nm}-keyed"
            return self.store.export_dir() / (f"{stem}.gguf" if bake else f"{stem}.lora.gguf")
        return self.store.export_dir() / (f"{nm}-{kind}.gguf" if bake else f"{nm}.lora.gguf")

    def check_head_proj(self, L: int, h: int, direction: str, *, ss=(1.0, 2.0, 6.0)) -> dict:
        """V24・V25(射影の正しさ・直交成分が変わらないこと)。"""
        from .headproj import check_head_proj
        return check_head_proj(self, L, h, direction, ss=ss)

    def check_whole_proj(self, L: int, kind: str, direction: str, *, ss=(1.0, 2.0, 6.0)) -> dict:
        """層の書き込み全体(attn・ffn)の射影の V24・V25(出力の差が式どおりか・直交成分が変わらないこと)。"""
        from .headproj import check_whole_proj
        return check_whole_proj(self, L, kind, direction, ss=ss)

    def check_neuron_proj(self, L: int, n: int, direction: str, *, ss=(1.0, 2.0, 6.0)) -> dict:
        """ニューロンの射影の V24・V25(MLP の出力の差が式どおりか・直交成分が変わらないこと)。"""
        from .headproj import check_neuron_proj
        return check_neuron_proj(self, L, n, direction, ss=ss)

    def check_proj_bake(self, recipe: Recipe, baked: Path, *, contrast: str | Path | None = None,
                        suite: str | Path | None = None) -> dict:
        """V26: 焼き込み版と LoRA 版で (a) A・B の射影の値、(b) 最初のトークンの分布の KL、(c) 最尤トークンの一致率、
        (d) perplexity を比べる。LoRA 版を測ってからモデルを閉じ、焼き込み版を読む(大きいモデルでも 1 つずつ)。
        射影する方向は proj の方向、proj が無ければ keyed の方向(V33、CHANGE-07)。

        射影の値の判定は、層ごと・側ごとに、例ごとの |LoRA 版 − 焼き込み版| の最大を許容誤差(その層の両側の射影の値の標準偏差の
        1 割)と比べる。最後の層(基準の層と違うとき)だけは、合格の線を「許容誤差 + 計算方法の違いだけで出る誤差」にする。
        その誤差は、LoRA 版を測ったのと同じセッションで、書き込みがほぼ 0 の LoRA(同じ LoRA ファイルを倍率 NOISE_LORA_SCALE で
        当てる)と LoRA なしの射影の値を測り、その層の両側の例ごとの差の最大を取る。Q8_0 の掛け算は途中の値も 8 ビットに丸めるので、LoRA を
        当てるだけで変わる計算の小さな違いが後ろの層で広がり、最後の層の射影の値は焼き込みと関係なく許容誤差の 7〜8 割動いた
        (verification.md F139)。基準の層の射影の値・最尤トークン・ppl の判定はそのまま。"""
        from .direction import Scorer, base_layer_of, last_logprobs
        from .evalx import _ppl, load_suite
        from .headproj import proj_heads
        from .keyed import keyed_units
        heads = proj_heads(recipe)
        ku = keyed_units(recipe) if recipe.keyed_units else []
        dname = heads[0].direction if heads else ku[0].direction if ku else recipe.default_direction
        ds, cs, _ = self._direction_contrast(dname, contrast)
        b = base_layer_of(ds)
        Ls = sorted({x for x in (b, max(L for L in ds.raw if L <= self.meta.n_layer - 2)) if x is not None})
        last = Ls[-1] if Ls[-1] != b else None                 # 合格の線に計算方法の違いの誤差を足す層
        convs = cs.convs("A") + cs.convs("B")
        texts = load_suite(suite).ppl

        def measure(sess, lora):
            sc = Scorer(sess, ds, cs, layers=Ls)
            got = sc.run(lora)
            lp = last_logprobs(sess, convs, None, lora)
            ppl = _ppl(sess, texts, None, lora, lambda n: None) if texts else {}
            return got, lp, ppl, sc
        lora = self.lora_for(recipe)
        a, lpa, ppa, sc = measure(self, lora)
        # 計算方法の違いだけで出る誤差: 書き込みがほぼ 0 の LoRA と LoRA なしの射影の値の差(射影の値だけ。モデルは開き直さない)。
        # 倍率を 0 ちょうどにすると llama.cpp が LoRA を外して計算の仕方が LoRA 版と変わるおそれがあるので、ごく小さい倍率にする
        noise = None
        if lora and last is not None:
            t0 = time.perf_counter()
            tiny = sc.run([(p, math.copysign(NOISE_LORA_SCALE, float(s_))) for p, s_ in lora], sides="AB")
            none = sc.run(None, sides="AB")
            by_side = {side: float(np.abs(np.asarray(tiny[side]["values"][last]) - np.asarray(none[side]["values"][last])).max())
                       for side in "AB"}
            noise = {"layer": last, "max_abs_diff": max(by_side.values()), "by_side": by_side,
                     "lora_scale": NOISE_LORA_SCALE, "seconds": time.perf_counter() - t0}
        self.close()
        other = Session(baked, root=self.store.root, n_gpu_layers=self._engine_kw["n_gpu_layers"],
                        n_ctx=self._engine_kw["n_ctx"], n_seq_max=self._engine_kw["n_seq_max"])
        try:
            bb, lpb, ppb, _ = measure(other, None)
        finally:
            other.close()
        scores, ok_s = bake_score_check(a, bb, Ls, last=last, noise=noise["max_abs_diff"] if noise else 0.0)
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
                "base_layer": b, "last_layer": last, "noise": noise,
                "pass": bool(ok_s and all(g < 0.5 for g in gaps) and (ppl_diff is None or abs(ppl_diff) < 0.5)),
                "note": None if not gaps else "最尤トークンが分かれた例はどれも僅差(0.5 nats 未満)なら合格。"
                                             "量子化し直しの丸めや、書き換えたテンソルの計算方法(Q8_0 か F16 か)の違いで分かれる"}

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

    def heads_geometry(self, name: str, **kw) -> dict:
        """heads writes の材料 name の層ごとの更新ベクトル(A 側と B 側の差)を、層どうしのコサイン類似度・特異値分解・半分ずつの
        一致・方向ベクトルとの cos で比べる(gne/geometry.py)。モデルは流さない(GPU を使わない)。結果は
        <root>/heads/<model_id>/<保存名>.geometry.json と .geometry.npz。"""
        from .geometry import analyze
        return analyze(self, name, **kw)

    def heads_geometry_export(self, result: str, out: str, source: str, *, force: bool = False) -> dict:
        """heads geometry の結果 result の向き(svd-rows:K / svd-none:K / within:K)を方向ベクトル out として保存する(gne/geometry.py)。"""
        from .geometry import export_direction
        return export_direction(self, result, out, source, force=force)


def bake_score_check(lora: dict, baked: dict, layers, *, last: int | None = None, noise: float = 0.0) -> tuple[dict, bool]:
    """V26・V33 の射影の値の判定。lora・baked は Scorer.run の結果({側: {"values": {L: 例ごとの射影の値}}})。層ごと・側ごとに
    {lora・baked(平均), max_abs_diff(例ごとの |LoRA − 焼き込み| の最大), resolution(許容誤差 = その層の LoRA 版の両側の
    射影の値の標準偏差の 1 割), threshold(合格の線), ok} を返す。合格の線は許容誤差で、last の層だけ許容誤差 + noise
    (計算方法の違いだけで出る誤差、check_proj_bake)。last の行には noise も入れる。戻り値 (scores, 全部 ok か)。"""
    scores, ok_all = {}, True
    for L in layers:
        sd = float(np.std(np.concatenate([lora["A"]["values"][L], lora["B"]["values"][L]])))
        res = 0.1 * sd                                    # 許容誤差: 両側の射影の値の広がりの 1 割
        line = res + float(noise) if L == last else res
        for side in "AB":
            va, vb = np.asarray(lora[side]["values"][L]), np.asarray(baked[side]["values"][L])
            diff = float(np.abs(va - vb).max())
            row = {"lora": float(va.mean()), "baked": float(vb.mean()), "max_abs_diff": diff, "resolution": res}
            if L == last:
                row["noise"] = float(noise)
            row.update(threshold=line, ok=diff <= line)
            scores[f"{L}{side}"] = row
            ok_all &= diff <= line
    return scores, bool(ok_all)


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
    """to_dict() のユニット → L:N:+dose / L:N:xk / L:Hh:xk(query は L:Hh:query:方向:×scale、表示用)。
    keyed(CHANGE-07)は L:ffn|attn|S|Ee:keyed:aα[@方向][#鍵]。"""
    if u.get("mode") == "nw":
        a = u.get("k") if u.get("k") is not None else u.get("alpha")
        return (f"{u['layer']}:N{u.get('neuron')}" + (f"+N{u['pair']}" if u.get("pair") is not None else "")
                + ":nw:" + ("a?" if a is None else f"a{a:g}") + (f"@{u['direction']}" if u.get("direction") else "")
                + (f"#{u['key']}" if u.get("key") else "") + (f"&{u['key2']}" if u.get("key2") else "")
                + (f"!s{u['s']:g}" if u.get("s") is not None else "") + (f"!cap{u['cap']}" if u.get("cap") else ""))
    if u.get("mode") == "keyed":
        kind = u.get("kind")
        who = {"ffn": "ffn", "attn": "attn", "shared": "S"}.get(kind, f"E{u.get('expert')}")
        a = u.get("k") if u.get("k") is not None else u.get("alpha")
        return (f"{u['layer']}:{who}:keyed:" + ("a?" if a is None else f"a{a:g}")
                + (f"@{u['direction']}" if u.get("direction") else "")
                + (f"#{u['key']}" if u.get("key") else ""))
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


def writes_block_reason(weights, arch: str) -> str | None:
    """このモデルで書き込みの量での順位づけ(rank writes)を実行しない理由(「…ので」の前に置く文)。実行できれば None。
    残差が複数本の流れ(hyper-connections)と、注意の出力のあとに正規化がある層(v0.9.0、補正した点数が入るまで)。
    UI の CHiPS の欄(app.js の writesBlocked)も同じ条件。"""
    hc = int(weights.field(f"{arch}.hyper_connection.count", 0) or 0)
    if hc:
        return f"このモデルは残差が {hc} 本の流れ(hyper-connections)で、書き込みの量での順位づけに対応していない"
    pn = [li.index for li in weights.arch.layers[:weights.n_main] if li.mixer == "attention" and li.post_norm_mixer not in ("", "?")]
    if pn:
        return f"注意の層 {len(pn)} 個で出力のあとに正規化があり、書き込みの量の点数(正規化の前)が残差での寄与を表さない"
    return None


def _writes_block_until(why: str) -> str:
    """writes_block_reason の文に添える、いつまでの扱いか(出力のあとの正規化だけ。補正した点数が入れば実行する)。"""
    return "正規化を補正した点数が次の版で入るまでの扱いです" if "正規化" in why else ""


def _merge_resid(old: ResidStats, new: ResidStats) -> ResidStats:
    """層を指定して計算したとき、既存の統計の他の層を残す(コーパスが同じ場合だけ)。"""
    if old.corpus_sha != new.corpus_sha or old.chunk_tokens != new.chunk_tokens:
        return new
    keep = [i for i, L in enumerate(old.layers) if L not in set(new.layers.tolist())]
    cat = lambda a, b: np.concatenate([a[keep], b])  # noqa: E731
    order = np.argsort(cat(old.layers, new.layers))
    nan = lambda st: (st.p50_rest if st.p50_rest is not None                       # noqa: E731  古い統計(偏りの量が無い)は NaN
                      else np.full(len(st.layers), np.nan, dtype=np.float32))
    rest = cat(nan(old), nan(new))[order]
    return ResidStats(cat(old.layers, new.layers)[order], cat(old.p10, new.p10)[order],
                      cat(old.p50, new.p50)[order], cat(old.p90, new.p90)[order], cat(old.mean, new.mean)[order],
                      new.n_tokens, new.corpus_sha, new.chunk_tokens,
                      p50_rest=None if old.p50_rest is None and new.p50_rest is None else rest)


def _version() -> str:
    from . import __version__
    return __version__


__all__ = ["open_model", "Session", "GneError"]
