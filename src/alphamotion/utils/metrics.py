"""Every GT-free evaluation metric in one place (owner requirement).

All cross-embodiment quantities are directions, angles or ratios normalised by
each body's own reach — that is what makes human-vs-robot comparisons legal.
Positions ALWAYS come through FK.
"""
from __future__ import annotations

import numpy as np
import torch

from ..engine.nets.rotations import rot6d_to_matrix
from ..engine.spatial import fk_pos, key_joints

KEYS = ("head", "l_wrist", "r_wrist", "l_ankle", "r_ankle")


# ----------------------------------------------------------- follow / amp ---

def _key_idx(spec):
    kj, _n, _t = key_joints(spec)
    return dict(zip(("root",) + KEYS, kj))


def dir_series(p: np.ndarray, kidx: dict) -> dict:
    out = {}
    for k in KEYS:
        v = p[:, kidx[k]] - p[:, kidx["root"]]
        out[k] = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return out


def follow_score(src_rot6d, src_spec, tgt_rot6d, tgt_spec) -> float:
    """Direction time-correlation of the five key joints, averaged.
    1.0 = perfect following; pipeline floor (shuffled time) = ~0.0."""
    ps = fk_pos(np.asarray(src_rot6d), src_spec)
    pt = fk_pos(np.asarray(tgt_rot6d), tgt_spec)
    T = min(len(ps), len(pt))
    ds = dir_series(ps[:T], _key_idx(src_spec))
    dt = dir_series(pt[:T], _key_idx(tgt_spec))
    cs = []
    for k in KEYS:
        cs.append(np.mean([np.corrcoef(ds[k][:, a], dt[k][:, a])[0, 1]
                           for a in range(3)]))
    return float(np.mean(cs))


def amplitude_ratio(src_rot6d, src_spec, tgt_rot6d, tgt_spec) -> float:
    """Robot amplitude / source amplitude, each normalised by its own reach.
    1.0 = no shrink. Means hide tails; report per-clip, not averaged blindly."""
    ps = fk_pos(np.asarray(src_rot6d), src_spec)
    pt = fk_pos(np.asarray(tgt_rot6d), tgt_spec)
    ks, kt = _key_idx(src_spec), _key_idx(tgt_spec)
    rs = np.linalg.norm(ps - ps[:, ks["root"]][:, None], axis=-1).max()
    rt = np.linalg.norm(pt - pt[:, kt["root"]][:, None], axis=-1).max()
    ratios = []
    for k in KEYS:
        a_s = (np.linalg.norm(ps[:, ks[k]] - ps[:, ks["root"]], axis=-1)
               / max(rs, 1e-6)).std()
        a_t = (np.linalg.norm(pt[:, kt[k]] - pt[:, kt["root"]], axis=-1)
               / max(rt, 1e-6)).std()
        ratios.append(a_t / max(a_s, 1e-9))
    return float(np.mean(ratios))


# -------------------------------------------------------------- physical ----

def jerk(rot6d, spec) -> float:
    p = fk_pos(np.asarray(rot6d), spec)
    return float(np.abs(np.diff(p, 3, axis=0)).mean())


def limit_hit_frac(q: torch.Tensor, dof: torch.Tensor, margin=0.02) -> float:
    """Fraction of limited dof-frames within `margin` of a joint stop."""
    from ..engine.constraints import _limits
    lo, hi, lim = _limits(dof, 1.0)
    if not lim.any():
        return 0.0
    span = (hi - lo).clamp(min=1e-6)
    near = (((q - lo[None]) / span[None] < margin)
            | ((hi[None] - q) / span[None] < margin)) & lim[None]
    return float(near[:, lim].double().mean())


# ------------------------------------------------------------------ arms ----

def arm_qc(h_rot6d, h_spec, r_rot6d, r_spec) -> dict:
    """Elbow fidelity + wrist rotation speeds (the HAND_BOTTLENECK metrics).

    Reading: jitter > 2x WITH src wrist speed < ~2 deg/f = still source +
    decoder activity floor (a source problem); jitter > 2x with normal source
    speed = the decoder inventing motion (a model problem)."""
    def pair(names, *keys):
        low = [n.lower() for n in names]
        out = []
        for k in keys:
            hit = [i for i, n in enumerate(low) if k in n]
            out.append(hit[0] if hit else None)
        return out

    def elbow_angle(p, s, e, w):
        a = p[:, s] - p[:, e]
        b = p[:, w] - p[:, e]
        a /= np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9
        b /= np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9
        return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))

    def rot_speed(R):
        dR = R[1:] @ R[:-1].transpose(0, 2, 1)
        tr = np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1)
        return np.degrees(np.arccos(tr))

    hp = fk_pos(np.asarray(h_rot6d), h_spec)
    rp = fk_pos(np.asarray(r_rot6d), r_spec)
    T = min(len(hp), len(rp))
    hn, rn = list(h_spec.joint_names), list(r_spec.joint_names)
    hs, he, hw = pair(hn, "l_shoulder", "r_shoulder"), \
        pair(hn, "l_elbow", "r_elbow"), pair(hn, "l_wrist", "r_wrist")
    rs, re, rw = pair(rn, "left_shoulder_pitch", "right_shoulder_pitch"), \
        pair(rn, "left_elbow", "right_elbow"), \
        pair(rn, "left_wrist", "right_wrist")
    if None in hs + he + hw or None in rs + re + rw:
        return {"available": False}
    err, jit, spd = [], [], []
    for i in (0, 1):
        ha = elbow_angle(hp[:T], hs[i], he[i], hw[i])
        ra = elbow_angle(rp[:T], rs[i], re[i], rw[i])
        err.append(float(np.mean(ra - ha)))
        Rh = rot6d_to_matrix(torch.as_tensor(
            np.asarray(h_rot6d)[:T, [hw[i]]], dtype=torch.float32)).numpy()[:, 0]
        Rr = rot6d_to_matrix(torch.as_tensor(
            np.asarray(r_rot6d)[:T, [rw[i]]], dtype=torch.float32)).numpy()[:, 0]
        sh, sr = float(np.median(rot_speed(Rh))), float(np.median(rot_speed(Rr)))
        jit.append(sr / max(sh, 1e-6))
        spd.append((round(sh, 2), round(sr, 2)))
    return {"available": True,
            "elbow_err_deg": {"left": round(err[0], 1),
                              "right": round(err[1], 1)},
            "elbow_asym_deg": round(abs(err[0] - err[1]), 1),
            "wrist_jitter_x": {"left": round(jit[0], 2),
                               "right": round(jit[1], 2)},
            "wrist_speed_deg_per_frame": {"left": spd[0], "right": spd[1]}}


# --------------------------------------------------------------- roundtrip --

@torch.no_grad()
def reencode_fidelity(greenwich, pose9, spec, dof) -> float:
    """Normalised code agreement of encode(decode(encode(x))) vs encode(x).
    Chance-corrected: 0 = random, 1 = perfect. Release decoder baseline: 0.61."""
    codes = greenwich.encode(pose9, spec, dof)
    rot = greenwich.decode(codes, spec, dof)
    p9b, _ = greenwich.pose9(rot.cpu(), spec, is_global=True)
    codes2 = greenwich.encode(p9b.to(pose9.device), spec, dof)
    agree = float((codes == codes2).float().mean())
    chance = 1.0 / 9.0
    return (agree - chance) / (1 - chance)
