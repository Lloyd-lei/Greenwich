"""Embodiment registry: bundled skeletons + user-ingested ones.

Bundled bodies ship as (spec, dof) npz pairs — 224 KB for the whole roster, no
meshes required for encoding/decoding. Meshes are only needed for rendering and
can be attached later per body. User URDFs ingested at runtime are persisted
through the service DB and land in data_dir().
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..engine.descriptor import build_from_cache, build_from_mjcf
from ..paths import assets_root, data_dir


@dataclass
class Embodiment:
    name: str
    source: str                      # bundled | user_urdf | user_mjcf
    spec: object                     # SkeletonSpec
    dof: np.ndarray
    rest: np.ndarray
    qnames: list | None
    xml: str | None


def bundled_names() -> list[str]:
    root = assets_root() / "embodiments"
    return sorted(p.name.removesuffix("_spec.npz")
                  for p in root.glob("*_spec.npz"))


def user_dir() -> Path:
    p = data_dir() / "embodiments"
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_names() -> list[str]:
    return sorted(p.name.removesuffix("_spec.npz")
                  for p in user_dir().glob("*_spec.npz"))


def mesh_map() -> dict:
    """User-attached MJCF paths for bundled bodies (vendor meshes are not
    redistributable, so this lives in data_dir as editable runtime config)."""
    import json
    p = data_dir() / "robot_meshes.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load(name: str) -> Embodiment:
    root = assets_root() / "embodiments"
    if (root / f"{name}_spec.npz").exists():
        spec, dof, rest, qn, xml = build_from_cache(root, name)
        meta = _meta(root, name)
        xml = xml or meta.get("xml") or mesh_map().get(name)
        return Embodiment(name, "bundled", spec, dof, rest,
                          qn or meta.get("qnames"), xml)
    u = user_dir()
    if (u / f"{name}_spec.npz").exists():
        spec, dof, rest, qn, xml = build_from_cache(u, name)
        meta = _meta(u, name)
        return Embodiment(name, meta.get("source", "user_urdf"), spec, dof,
                          rest, meta.get("qnames"), meta.get("xml"))
    raise KeyError(f"unknown embodiment '{name}'; "
                   f"bundled={bundled_names()}, user={user_names()}")


def load_from_xml(xml: str | Path, name: str) -> Embodiment:
    spec, dof, rest, qn, xml_s = build_from_mjcf(xml, name)
    return Embodiment(name, "user_mjcf", spec, dof, rest, qn, xml_s)


def _meta(root: Path, name: str) -> dict:
    import json
    p = root / f"{name}_meta.json"
    return json.loads(p.read_text()) if p.exists() else {}
