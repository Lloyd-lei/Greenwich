"""Stride odometry: recover world root translation from foot contacts.

The code space is deliberately root-relative — no world translation exists in
the representation. Whichever foot is planted is pinned to the world, so the
root moves opposite to that foot's root-relative velocity.

MUST run in the RENDERER's own world frame, on the posed mesh (0814 audit,
g1/walk24): computed in the cache Y-up frame — from the position head, the
decoded-rot FK, or the projected-q FK — the integrated direction lands 15-85
degrees off after the renderer's axis conjugation (the Y-up->Z-up map does not
commute with the root rotation). Computed from the mesh's own foot bodies in
the final frame, stance-foot slide drops 2.39 -> 0.20 cm/frame. Renderers call
stance_offsets() on the foot trajectories they already forward-kinematic.
"""
from __future__ import annotations

import numpy as np

_FOOT_WORDS = ("ankle", "foot", "toe")


def foot_indices(joint_names) -> list[int]:
    return [i for i, n in enumerate(joint_names)
            if any(w in n.lower() for w in _FOOT_WORDS)]


def stance_offsets(foot_world: np.ndarray) -> np.ndarray:
    """foot_world [T,F,3] (renderer world, z-up, meters, zero root
    translation) -> root offset [T,2] (x,y meters) that pins the stance foot.

    The validated formula (g1/walk24: stance slide 2.39 -> 0.20 cm/frame):
    stance weight = exp(-height/med) * exp(-speed/med) per foot per frame,
    root velocity = minus the stance-weighted foot velocity, integrated.
    Flight frames (no plausible support) contribute zero.
    """
    T, F = foot_world.shape[:2]
    if F < 1 or T < 3:
        return np.zeros((T, 2))
    v = np.zeros_like(foot_world)
    v[1:] = foot_world[1:] - foot_world[:-1]
    h = foot_world[..., 2] - foot_world[..., 2].min()
    speed = np.linalg.norm(v[..., :2], axis=-1)
    w = np.exp(-h / (np.median(h) + 1e-9)) \
        * np.exp(-speed / (np.median(speed) + 1e-9))
    wsum = w.sum(-1)
    vroot = -(w[..., None] * v[..., :2]).sum(1) \
        / np.maximum(wsum[:, None], 1e-9)
    vroot[wsum < 0.2 * np.median(wsum)] = 0.0
    return np.cumsum(vroot, axis=0)


def foot_bodies(model) -> list[int]:
    """Mesh body ids to treat as feet (by name; fallback: two lowest bodies
    in the rest pose)."""
    import mujoco as mj
    ids = [i for i in range(model.nbody)
           if any(w in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or "")
                  .lower() for w in _FOOT_WORDS)]
    if ids:
        return ids
    order = np.argsort(model.body_pos[:, 2])
    return [int(b) for b in order[:2] if int(b) != 0]


def stride_odometry(gp: np.ndarray, joint_names, fps: float = 30.0):
    """gp [T,J,3] cm, Y-up, root-relative FK positions -> root_t [T,3] cm
    world (y stays 0).

    Stance weight per foot: low height x low ground-plane speed (soft, per
    frame). Root velocity = minus the stance-weighted foot velocity. Frames
    where nothing is plausibly planted (flight) contribute zero.
    """
    T = len(gp)
    root_t = np.zeros((T, 3), np.float64)
    feet = foot_indices(joint_names)
    if len(feet) < 2 or T < 3:
        return root_t
    fp = gp[:, feet, :].astype(np.float64)             # [T,F,3]
    v = np.zeros_like(fp)
    v[1:] = fp[1:] - fp[:-1]                           # cm/frame
    h = fp[..., 1] - fp[..., 1].min()
    speed = np.linalg.norm(v[..., [0, 2]], axis=-1)
    w = np.exp(-h / (np.median(h) + 1e-6)) \
        * np.exp(-speed / (np.median(speed) + 1e-6))   # [T,F]
    wsum = w.sum(-1)
    vroot = -(w[..., None] * v).sum(1) / np.maximum(wsum[:, None], 1e-9)
    # flight: both feet high+fast -> no support, no translation evidence
    vroot[wsum < 0.2 * np.median(wsum)] = 0.0
    root_t[:, 0] = np.cumsum(vroot[:, 0])
    root_t[:, 2] = np.cumsum(vroot[:, 2])
    return root_t
