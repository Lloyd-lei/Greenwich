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


def stance_offsets(foot_world: np.ndarray, *, return_report: bool = False):
    """foot_world [T,F,3] (renderer world, z-up, meters, zero root
    translation) -> root offset [T,2] (x,y meters) that pins the stance foot.

    A hysteretic contact selector chooses one support foot from height and
    horizontal speed, then integrates the exact opposite support velocity.
    Unlike a soft average of both feet, this does not let the swing foot pull
    the root away from the planted foot. Flight frames contribute zero.
    """
    T, F = foot_world.shape[:2]
    if F < 1 or T < 3:
        out = np.zeros((T, 2))
        report = {"available": False, "reason": "fewer than 3 frames/1 foot"}
        return (out, report) if return_report else out
    v = np.zeros_like(foot_world)
    v[1:] = foot_world[1:] - foot_world[:-1]
    height = foot_world[..., 2]
    base = float(np.percentile(height, 2))
    h = np.maximum(height - base, 0.0)
    speed = np.linalg.norm(v[..., :2], axis=-1)
    moving = speed[speed > 1e-6]
    speed_scale = max(0.005, float(np.median(moving)) if len(moving) else 0.005)
    height_scale = max(0.025, min(0.12, float(np.percentile(h, 35))))
    score = h / height_scale + 0.35 * speed / speed_scale

    support = np.zeros(T, np.int64)
    support[0] = int(np.argmin(score[0]))
    active = np.ones(T, bool)
    # At least 5 cm of clearance is required before declaring flight; tall
    # jumps may use up to 12 cm because ankle body origins sit above the sole.
    flight_height = max(0.05, min(0.12, 0.25 * float(np.ptp(height))))
    active[0] = h[0, support[0]] <= flight_height
    for t in range(1, T):
        best = int(np.argmin(score[t]))
        current = int(support[t - 1])
        # Hysteresis prevents one-frame left/right chatter in double support.
        support[t] = best if score[t, best] + 0.35 < score[t, current] \
            else current
        active[t] = h[t, support[t]] <= flight_height

    vroot = np.zeros((T, 2), np.float64)
    rows = np.arange(T)
    vroot[active] = -v[rows[active], support[active], :2]
    # A bad imported frame must not teleport the root.  8 cm/frame is already
    # 2.4 m/s at 30 fps and therefore leaves ample room for fast locomotion.
    magnitude = np.linalg.norm(vroot, axis=1)
    clipped = magnitude > 0.08
    vroot[clipped] *= 0.08 / magnitude[clipped, None]
    offsets = np.cumsum(vroot, axis=0)
    if not return_report:
        return offsets

    before = speed[rows[active], support[active]] if active.any() \
        else np.zeros(1)
    corrected = foot_world.copy()
    corrected[..., :2] += offsets[:, None, :]
    after_v = np.zeros((T, F), np.float64)
    after_v[1:] = np.linalg.norm(
        corrected[1:, :, :2] - corrected[:-1, :, :2], axis=-1)
    after = after_v[rows[active], support[active]] if active.any() \
        else np.zeros(1)
    after_p50 = float(np.percentile(after, 50) * 100)
    after_p90 = float(np.percentile(after, 90) * 100)
    report = {
        "available": True,
        "passed": bool(after_p50 <= 0.2 and after_p90 <= 1.5),
        "contact_fraction": round(float(active.mean()), 4),
        "slide_cm_frame_before_p50": round(float(np.percentile(before, 50) * 100), 4),
        "slide_cm_frame_before_p90": round(float(np.percentile(before, 90) * 100), 4),
        "slide_cm_frame_after_p50": round(after_p50, 4),
        "slide_cm_frame_after_p90": round(after_p90, 4),
        "clipped_frames": int(clipped.sum()),
    }
    return offsets, report


def foot_bodies(model) -> list[int]:
    """One distal contact body per foot, with a low-body fallback.

    Some MJCFs name ankle, foot, and toe bodies. Returning all of them makes
    the odometry solver average three points from the same leg and biases the
    result. Prefer toe, then foot, then ankle independently on each side.
    """
    import mujoco as mj
    names = [(mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or "").lower()
             for i in range(model.nbody)]

    def side_of(name: str) -> str:
        if "left" in name or name.startswith("l_") or "_l_" in name:
            return "left"
        if "right" in name or name.startswith("r_") or "_r_" in name:
            return "right"
        return ""

    chosen = []
    for side in ("left", "right"):
        candidates = [i for i, name in enumerate(names)
                      if side_of(name) == side
                      and any(word in name for word in _FOOT_WORDS)]
        if candidates:
            def rank(i):
                return next((r for r, word in enumerate(("toe", "foot", "ankle"))
                             if word in names[i]), 9)
            chosen.append(min(candidates, key=rank))
    if len(chosen) == 2:
        return chosen
    ids = [i for i, name in enumerate(names)
           if any(word in name for word in _FOOT_WORDS)]
    if ids:
        return sorted(ids, key=lambda i: float(model.body_pos[i, 2]))[:2]
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
