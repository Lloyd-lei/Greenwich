"""Skeleton-level spatial utilities, vendored from the research stack verbatim
so packaged numbers match research numbers bit-for-bit.

Sources: v5_common.py (KEY_PATTERNS/rest_positions/key_joints/fk_pos),
train_global.py (build_global), train_ae.py (static_cond).
"""
from __future__ import annotations

import re

import numpy as np
import torch

from .nets.blocks import hop_distance
from .nets.rotations import SkeletonSpec, fk_global, matrix_to_rot6d, rot6d_to_matrix

_DEPTH: dict = {}

KEY_PATTERNS = [
    ("head",    [r"^head$", r"head", r"neck"]),
    ("l_wrist", [r"^l_wrist$", r"left_wrist", r"wrist.*left", r"left.*hand",
                 r"left_end_effector", r"l_?hand"]),
    ("r_wrist", [r"^r_wrist$", r"right_wrist", r"wrist.*right", r"right.*hand",
                 r"right_end_effector", r"r_?hand"]),
    ("l_ankle", [r"^l_ankle$", r"left_ankle", r"ankle.*left", r"left_foot",
                 r"foot.*left", r"anklepitchleft"]),
    ("r_ankle", [r"^r_ankle$", r"right_ankle", r"ankle.*right", r"right_foot",
                 r"foot.*right", r"anklepitchright"]),
]


def rest_positions(spec):
    """Rest-pose global joint positions [J,3] (cm, Y-up, root at origin) — identity
    rotations everywhere, i.e. just the offsets accumulated down the tree."""
    off = np.asarray(spec.rest_offsets, np.float64)
    p = np.zeros((spec.J, 3))
    for j in range(spec.J):
        k = int(spec.parents[j])
        if k >= 0:
            p[j] = p[k] + off[j]
    return p


def _geometric_key(spec):
    """Body-agnostic fallback: pick extremities off the REST pose. head = highest
    joint; ankles = lowest joint on each side; wrists = most lateral joint above the
    root. Works for pal_talos ('body00'...'body32') where names carry nothing."""
    p = rest_positions(spec)
    x, y = p[:, 0], p[:, 1]
    J = spec.J
    root = int(np.where(np.asarray(spec.parents) < 0)[0][0])
    idx = {"head": int(np.argmax(y))}
    for tag, sgn in (("l_ankle", +1), ("r_ankle", -1)):
        cand = [j for j in range(J) if j != root and np.sign(x[j]) == sgn]
        cand = cand or [j for j in range(J) if j != root]
        idx[tag] = int(min(cand, key=lambda j: y[j]))
    for tag, sgn in (("l_wrist", +1), ("r_wrist", -1)):
        cand = [j for j in range(J) if j != root and np.sign(x[j]) == sgn
                and y[j] > y[root]]
        cand = cand or [j for j in range(J) if j != root and np.sign(x[j]) == sgn]
        cand = cand or [j for j in range(J) if j != root]
        idx[tag] = int(max(cand, key=lambda j: abs(x[j])))
    return idx


def key_joints(spec, verbose=False):
    """-> (idx [6] int list, names [6], tags [6]). Order is ALWAYS
    (root, head, l_wrist, r_wrist, l_ankle, r_ankle) so the 6x9 state vector means the
    same thing on every skeleton — that is what makes the cross-embodiment InfoNCE and
    the cross-embodiment token transfer legal."""
    names = [n.lower() for n in spec.joint_names]
    root = int(np.where(np.asarray(spec.parents) < 0)[0][0])
    geo = _geometric_key(spec)
    out, used = {"root": root}, {root}
    for tag, pats in KEY_PATTERNS:
        hit = None
        for pat in pats:
            cands = [j for j, n in enumerate(names)
                     if re.search(pat, n) and j not in used]
            if cands:
                # deepest match wins: 'left_wrist_yaw_link' over 'left_wrist_roll_link'
                hit = max(cands, key=lambda j: _depth(spec, j))
                break
        if hit is None:
            hit = geo[tag]
            if hit in used:                        # degenerate tiny skeleton
                hit = max(range(spec.J), key=lambda j: (j not in used, _depth(spec, j)))
        out[tag] = int(hit); used.add(int(hit))
    tags = ["root", "head", "l_wrist", "r_wrist", "l_ankle", "r_ankle"]
    idx = [out[t] for t in tags]
    if verbose:
        print(f"[key joints] {spec.name}: " +
              ", ".join(f"{t}={spec.joint_names[j]}" for t, j in zip(tags, idx)))
    return idx, [spec.joint_names[j] for j in idx], tags


_DEPTH = {}


def _depth(spec, j):
    key = (id(spec), j)
    if key not in _DEPTH:
        d, k = 0, int(spec.parents[j])
        while k >= 0:
            d += 1; k = int(spec.parents[k])
        _DEPTH[key] = d
    return _DEPTH[key]

def fk_pos(rot6d, spec):
    """global rot6d [T,J,6] -> RIGID-BONE positions [T,J,3] cm.

    The raw decoder position stream is never used anywhere in V5 (p1_fkfix: rendering
    it gave rubber bones and ~12 cm error). Every position in every V5 number and
    every V5 frame comes through here.
    """
    R = rot6d_to_matrix(torch.as_tensor(np.asarray(rot6d), dtype=torch.float64)).numpy()
    par = np.asarray(spec.parents); off = np.asarray(spec.rest_offsets, np.float64)
    p = np.zeros((R.shape[0], len(par), 3))
    for j in range(len(par)):
        k = int(par[j])
        if k >= 0:
            p[:, j] = p[:, k] + np.einsum("tab,b->ta", R[:, k], off[j])
    return p


def build_global(local6d, spec, device, chunk=4096):
    """FK all frames -> global_rot6d [N,J,6], global_pos [N,J,3] (root-rel, /reach)."""
    N = local6d.shape[0]
    outs_r, outs_p = [], []
    reach = None
    for i in range(0, N, chunk):
        b = local6d[i:i+chunk].to(device)
        gR, gp = fk_global(b, spec)             # [n,J,3,3],[n,J,3]
        outs_r.append(matrix_to_rot6d(gR).cpu())
        outs_p.append(gp.cpu())
    gr6 = torch.cat(outs_r, 0)
    gp = torch.cat(outs_p, 0)
    reach = gp.norm(dim=-1).max().item()
    return gr6, gp, max(reach, 1e-6)


def static_cond(spec: SkeletonSpec, device):
    """Per-skeleton static conditioning tensors (train_ae.static_cond port)."""
    hop = hop_distance(spec.parents)
    depth = hop[int(np.where(spec.parents < 0)[0][0])]
    return {
        "rest_off": torch.as_tensor(spec.rest_offsets, device=device),
        "bonelen": torch.as_tensor(spec.norm_bonelen, device=device),
        "depth": torch.as_tensor(depth, device=device),
        "hop": torch.as_tensor(hop, device=device),
        "J": spec.J,
        "height": spec.height,
    }
