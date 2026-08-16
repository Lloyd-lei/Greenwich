"""Compact browser payloads for lightweight library hover previews."""
from __future__ import annotations

import numpy as np


def skeleton_preview_payload(global_positions, parents, max_frames: int = 30):
    """Project a root-relative 3-D skeleton into a normalized 2-D flipbook.

    The full robot viewer is intentionally not used for hover previews: loading
    dozens of MJCF meshes while browsing would stall the editor.  An isometric
    projection keeps lateral, vertical, and depth motion legible in a small
    canvas and produces a payload of only a few kilobytes.
    """
    points = np.asarray(global_positions, np.float64)
    if points.ndim != 3 or points.shape[-1] != 3 or not len(points):
        raise ValueError("global_positions must be non-empty [T,J,3]")
    if not np.isfinite(points).all():
        raise ValueError("global_positions contains NaN or infinity")
    parent = np.asarray(parents, np.int64)
    if parent.shape != (points.shape[1],):
        raise ValueError("parents must contain one entry per joint")

    count = min(max(int(max_frames), 1), len(points))
    sample = np.linspace(0, len(points) - 1, count).round().astype(np.int64)
    p = points[sample]
    # Slight isometric projection: X is lateral, Y is up, Z contributes depth.
    xy = np.stack([p[..., 0] + 0.28 * p[..., 2],
                   p[..., 1] + 0.10 * p[..., 2]], axis=-1)
    lo, hi = xy.min(axis=(0, 1)), xy.max(axis=(0, 1))
    center = (lo + hi) * 0.5
    scale = max(float((hi - lo).max()), 1e-6)
    xy = (xy - center) / scale
    edges = [[int(pj), int(j)] for j, pj in enumerate(parent) if pj >= 0]
    return {
        "frames": int(count),
        "source_frames": int(len(points)),
        "points": np.round(xy, 4).tolist(),
        "edges": edges,
    }
