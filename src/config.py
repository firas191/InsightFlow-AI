"""Loads config/config.yaml as the single source of truth for the whole project.

Usage:
    from src.config import CFG
    CFG.labels["category"]        # list of category labels
    CFG.models["base_encoder"]    # model name
    CFG.get("agent", "retrieve_k")
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Repo root = parent of this file's parent (src/ -> repo root).
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"


class Config:
    """Thin attribute-style wrapper around the YAML config."""

    def __init__(self, data: dict[str, Any], root: Path):
        self._data = data
        self.root = root

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(key) from exc

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self._data
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def path(self, *keys: str) -> Path:
        """Resolve a config path value (relative to repo root) to an absolute Path."""
        value = self.get(*keys)
        if value is None:
            raise KeyError(f"No path configured at {keys}")
        return (self.root / value).resolve()

    # Convenience: head name -> label list, in fixed index order.
    @property
    def heads(self) -> dict[str, list[str]]:
        return self._data["labels"]


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg_path = Path(path) if path else CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data, ROOT)


# Importable singleton.
CFG = load_config()
