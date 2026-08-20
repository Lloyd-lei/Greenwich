"""Validation helpers for project-local motion and robot uploads."""
from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import numpy as np


def inspect_smpl_npz(path: str | Path) -> dict:
    """Return a small, validated description of a supported motion NPZ."""
    with np.load(path, allow_pickle=False) as data:
        if "poses" in data.files:
            poses = np.asarray(data["poses"])
            if poses.ndim != 2 or poses.shape[0] < 1 or poses.shape[1] < 72:
                raise ValueError(
                    f"expected poses [frames, >=72], got {poses.shape}")
            frames = int(poses.shape[0])
            pose_dimensions = int(poses.shape[1])
            representation = "axis_angle"
        elif "local_rot6d" in data.files:
            rotations = np.asarray(data["local_rot6d"])
            if (rotations.ndim != 3 or rotations.shape[0] < 1
                    or rotations.shape[1] < 22 or rotations.shape[2] != 6):
                raise ValueError(
                    "expected local_rot6d [frames, >=22, 6], "
                    f"got {rotations.shape}")
            frames = int(rotations.shape[0])
            pose_dimensions = int(rotations.shape[1] * rotations.shape[2])
            representation = "local_rot6d"
        else:
            raise ValueError("NPZ has neither poses nor local_rot6d motion data")

        fps = next((float(np.asarray(data[key]).reshape(()))
                    for key in ("mocap_framerate", "mocap_frame_rate", "fps")
                    if key in data.files), 30.0)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid motion frame rate: {fps}")
    return {"frames": frames, "fps": fps,
            "pose_dimensions": pose_dimensions,
            "representation": representation}


def missing_urdf_resources(path: str | Path,
                           package_root: str | Path | None = None) -> list[str]:
    """List mesh references that cannot be resolved inside an upload."""
    urdf = Path(path).resolve()
    root = Path(package_root or urdf.parent).resolve()
    try:
        document = ElementTree.parse(urdf)
    except (ElementTree.ParseError, OSError) as exc:
        raise ValueError(f"invalid URDF XML: {exc}") from exc

    missing = set()
    for mesh in document.findall(".//mesh"):
        value = str(mesh.get("filename") or "").strip()
        if not value:
            continue
        relative = value.removeprefix("file://")
        candidates: list[Path]
        if relative.startswith("package://"):
            relative = relative.removeprefix("package://")
            parts = Path(relative).parts
            candidates = [root / relative]
            if len(parts) > 1:
                candidates.append(root / Path(*parts[1:]))
        else:
            candidate = Path(relative)
            candidates = ([candidate] if candidate.is_absolute()
                          else [urdf.parent / candidate])
        valid = False
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_relative_to(root) and resolved.is_file():
                valid = True
                break
        if not valid:
            missing.add(value)
    return sorted(missing)
