"""Stride odometry: recover world root translation from foot contacts.

The code space is deliberately root-relative — no world translation exists in
the representation. For playback we reconstruct it the way the research stack
validated (contact integration, net-displacement corr 0.988): whichever foot
is planted is pinned to the world, so the root moves opposite to that foot's
root-relative velocity. In-place motions integrate to ~zero by construction.
"""
from __future__ import annotations

import numpy as np

_FOOT_WORDS = ("ankle", "foot", "toe")


def foot_indices(joint_names) -> list[int]:
    hits = [i for i, n in enumerate(joint_names)
            if any(w in n.lower() for w in _FOOT_WORDS)]
    # prefer the most distal per side: keep all — stance weighting sorts it out
    return hits


def stride_odometry(gp: np.ndarray, joint_names, fps: float = 30.0):
    """gp [T,J,3] cm, Y-up, root-relative -> root_t [T,3] cm world (y stays 0).

    Stance weight per foot = softmax of (low height, low speed); the root
    velocity is minus the stance-weighted foot velocity in the ground plane.
    Frames with no plausible stance (both feet high/fast — flight) contribute
    zero. A per-frame step cap keeps decoder jitter from teleporting the root.
    """
    T = len(gp)
    root_t = np.zeros((T, 3), np.float64)
    feet = foot_indices(joint_names)
    if len(feet) < 2 or T < 3:
        return root_t
    fp = gp[:, feet, :].astype(np.float64)             # [T,F,3]
    v = np.zeros_like(fp)
    v[1:] = fp[1:] - fp[:-1]                           # cm/frame
    h = fp[..., 1] - fp[..., 1].min()                  # height above lowest
    speed = np.linalg.norm(v[..., [0, 2]], axis=-1)    # ground-plane speed
    # stance score: on the ground AND not sliding fast
    hs = np.median(h) + 1e-6
    ss = np.median(speed) + 1e-6
    w = np.exp(-h / hs) * np.exp(-speed / ss)          # [T,F]
    wsum = w.sum(-1, keepdims=True)
    conf = wsum[:, 0] / (w.max() + 1e-9)               # 0 when nothing planted
    vroot = -(w[..., None] * v).sum(1) / np.maximum(wsum, 1e-9)  # [T,3]
    vroot_xz = vroot[:, [0, 2]]
    # flight / uncertain frames -> no translation
    vroot_xz[conf < 0.15] = 0.0
    # cap: nothing plausible moves faster than 4 reach/s; use robust cap
    cap = 6.0 * (np.percentile(np.abs(vroot_xz), 90) + 1e-6)
    vroot_xz = np.clip(vroot_xz, -cap, cap)
    # light smoothing (3-frame box) to kill decoder jitter before integration
    k = np.ones(3) / 3.0
    for c in range(2):
        vroot_xz[:, c] = np.convolve(vroot_xz[:, c], k, mode="same")
    root_t[:, 0] = np.cumsum(vroot_xz[:, 0])
    root_t[:, 2] = np.cumsum(vroot_xz[:, 1])
    return root_t
