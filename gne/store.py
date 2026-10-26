"""保存先と ID。設計書 §6.11。

~/.gne/(または --root、環境変数 GNE_ROOT)
  models/<model_id>.json           ModelMeta のキャッシュ
  stats/<model_id>/resid.npz, act_L{L}.npz
  lens/<model_id>/...
  runs/<model_id>/...
  reverse/<model_id>/<name>.directions.npz
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
