"""保存先と ID。設計書 §6.11。

~/.gne/(または --root、環境変数 GNE_ROOT)
  models/<model_id>.json           ModelMeta のキャッシュ
  stats/<model_id>/resid.npz, act_L{L}.npz
  lens/<model_id>/...
  runs/<model_id>/...
  reverse/<model_id>/<name>.directions.npz
  gain/<model_id>/<hash>.lora.gguf  gain の実行時 LoRA
  evals/<model_id>/<name>.json      gne eval の結果
  directions/<model_id>/<name>.npz  gne direction の方向(層ごとの mean_A − mean_B)
  ablate/<model_id>/*.lora.gguf     方向を消す LoRA(層選びと gne ablate)
  recipes/*.yaml
  export/*
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

CONFIGS = Path(os.environ.get("GNE_CONFIGS", Path(__file__).resolve().parent.parent / "configs"))


def default_root() -> Path:
    return Path(os.environ.get("GNE_ROOT", Path.home() / ".gne"))


class Store:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else default_root()

    def _dir(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def stats_dir(self, model_id: str) -> Path:
        return self._dir("stats", model_id)

    def resid_path(self, model_id: str) -> Path:
        return self.stats_dir(model_id) / "resid.npz"

    def act_path(self, model_id: str, L: int) -> Path:
        return self.stats_dir(model_id) / f"act_L{L}.npz"

    def lens_dir(self, model_id: str) -> Path:
        return self._dir("lens", model_id)

    def runs_dir(self, model_id: str) -> Path:
        return self._dir("runs", model_id)

    def reverse_dir(self, model_id: str) -> Path:
        return self._dir("reverse", model_id)

    def gain_dir(self, model_id: str) -> Path:
        """gain(§6.16)の実行時 LoRA の置き場所。"""
        return self._dir("gain", model_id)

    def direction_dir(self, model_id: str) -> Path:
        """gne direction(§6.17)の方向の置き場所。"""
        return self._dir("directions", model_id)

    def ablate_dir(self, model_id: str) -> Path:
        """方向を消す LoRA(§6.17)の置き場所。"""
        return self._dir("ablate", model_id)

    def eval_dir(self, model_id: str) -> Path:
        """gne eval(§6.14)の結果の置き場所。"""
        return self._dir("evals", model_id)

    def connections_path(self, model_id: str) -> Path:
        return self._dir("connections") / f"{model_id}.npz"

    def layout_dir(self, model_id: str) -> Path:
        return self._dir("layout", model_id)

    def recipes_dir(self) -> Path:
        return self._dir("recipes")

    def export_dir(self) -> Path:
        return self._dir("export")

    def save_meta(self, meta) -> Path:
        p = self._dir("models") / f"{meta.model_id}.json"
        d = asdict(meta)
        d["path"] = str(d["path"])
        d.pop("chat_template", None)
        p.write_text(json.dumps(d, ensure_ascii=False, indent=1))
        return p
