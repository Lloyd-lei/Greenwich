"""All filesystem locations. No literal paths anywhere else in the package."""
from __future__ import annotations

import os
from pathlib import Path

import platformdirs

_APP = "alphamotion"


def cache_dir() -> Path:
    p = Path(os.environ.get("ALPHAMOTION_CACHE", "")
             or platformdirs.user_cache_dir(_APP))
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_dir() -> Path:
    p = Path(os.environ.get("ALPHAMOTION_DATA", "")
             or platformdirs.user_data_dir(_APP))
    p.mkdir(parents=True, exist_ok=True)
    return p


def weights_dir() -> Path:
    p = cache_dir() / "weights"
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir() -> Path:
    p = data_dir() / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def db_path() -> Path:
    return data_dir() / "alphamotion.sqlite3"


def assets_root() -> Path:
    """Packaged static assets (embodiment specs, frontend)."""
    return Path(__file__).parent / "assets"
