"""Shared utilities for the topology-agnostic SE3 experiment.

Uniform SkeletonSpec + batched FK + rotation helpers, so every skeleton
(human BVH, robot CSV, ...) is handled by exactly the same code path.
No per-embodiment special-casing -- that is the whole point (bitter lesson).
"""
from __future__ import annotations
import numpy as np
import torch



# ----------------------------- rotation helpers -----------------------------

def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """[...,3,3] -> [...,6] = [column0(3), column1(3)], consistent with rot6d_to_matrix
    which reads x[0:3] as the first column. (Prev bug: .reshape row-major INTERLEAVED
    the two columns, scrambling every rotation.)"""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """[...,6] -> [...,3,3] via Gram-Schmidt."""
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _skew(v):
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], -1),
        torch.stack([v[..., 2], z, -v[..., 0]], -1),
        torch.stack([-v[..., 1], v[..., 0], z], -1),
    ], -2)


def rot_axis_angle(axis, cos, sin):
    """Rodrigues: rotation matrix about unit `axis` by angle with given cos/sin. [...,3,3]."""
    a = torch.nn.functional.normalize(axis, dim=-1)
    K = _skew(a)
    I = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(K.shape)
    cos = cos[..., None, None]; sin = sin[..., None, None]
    return I + sin * K + (1 - cos) * (K @ K)


def minimal_rot(a0, a1):
    """Minimal rotation taking unit a0 to unit a1. [...,3,3]. Handles anti-parallel."""
    a0 = torch.nn.functional.normalize(a0, dim=-1)
    a1 = torch.nn.functional.normalize(a1, dim=-1)
    v = torch.cross(a0, a1, dim=-1)
    c = (a0 * a1).sum(-1, keepdim=True)
    s = v.norm(dim=-1, keepdim=True)
    axis = torch.where(s > 1e-6, v, torch.tensor([1.0, 0.0, 0.0], device=a0.device).expand_as(v))
    cos = c.squeeze(-1); sin = s.squeeze(-1)
    return rot_axis_angle(axis, cos, sin)


def swing_twist_decompose(R, a0):
    """Given global rot R and rest bone axis a0, return (swing_dir=R@a0, twist_cos, twist_sin)."""
    a0 = torch.nn.functional.normalize(a0, dim=-1)
    a1 = (R @ a0.unsqueeze(-1)).squeeze(-1)                 # swing dir
    Sw = minimal_rot(a0, a1)
    Tw = Sw.transpose(-1, -2) @ R                           # rotation about a0
    # reference perp vector to a0
    ref = torch.tensor([0.0, 0.0, 1.0], device=R.device).expand_as(a0)
    alt = torch.tensor([0.0, 1.0, 0.0], device=R.device).expand_as(a0)
    ref = torch.where((a0 * ref).sum(-1, keepdim=True).abs() > 0.9, alt, ref)
    r0 = torch.nn.functional.normalize(ref - (ref * a0).sum(-1, keepdim=True) * a0, dim=-1)
    r1 = (Tw @ r0.unsqueeze(-1)).squeeze(-1)
    cos = (r0 * r1).sum(-1)
    sin = (a0 * torch.cross(r0, r1, dim=-1)).sum(-1)
    return a1, cos, sin


def swing_twist_compose(a0, swing_dir, cos, sin):
    """Rebuild global rot R = Swing(a0->swing_dir) @ Twist(a0, angle)."""
    a0 = torch.nn.functional.normalize(a0, dim=-1)
    a1 = torch.nn.functional.normalize(swing_dir, dim=-1)
    n = torch.sqrt(cos * cos + sin * sin).clamp(min=1e-6)
    Tw = rot_axis_angle(a0, cos / n, sin / n)
    Sw = minimal_rot(a0, a1)
    return Sw @ Tw


def geodesic_deg(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """Mean geodesic angle (degrees) between two batches of rot matrices."""
    Rrel = Ra.transpose(-1, -2) @ Rb
    tr = Rrel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.rad2deg(torch.arccos(cos))


# ----------------------------- skeleton spec --------------------------------

class SkeletonSpec:
    """Uniform description of ANY skeleton, consumed identically by the model."""

    def __init__(self, name, parents, rest_offsets, joint_names):
        self.name = name
        self.parents = np.asarray(parents, dtype=np.int64)      # [J], root parent = -1
        self.rest_offsets = np.asarray(rest_offsets, np.float32)  # [J,3] offset from parent
        self.joint_names = list(joint_names)
        self.J = len(self.parents)
        assert self.rest_offsets.shape == (self.J, 3)
        assert len(self.joint_names) == self.J
        # normalized bone length per joint (scale-free): |offset| / skeleton height
        blen = np.linalg.norm(self.rest_offsets, axis=-1)
        self.height = float(blen.sum()) or 1.0
        self.norm_bonelen = (blen / self.height).astype(np.float32)  # [J]

    def order_root_first(self):
        """Assert topological order (parent index < child index)."""
        for j, p in enumerate(self.parents):
            if p >= 0:
                assert p < j, f"{self.name}: joint {j} parent {p} not before it"


def fk_global(local_rot6d: torch.Tensor, spec: SkeletonSpec, device=None):
    """Batched forward kinematics.

    Args:
        local_rot6d: [B, J, 6] local joint rotations (root = global root rot).
    Returns:
        global_rot: [B, J, 3, 3]
        global_pos: [B, J, 3]  (root at origin -> root-relative)
    """
    B, J, _ = local_rot6d.shape
    R = rot6d_to_matrix(local_rot6d)                       # [B,J,3,3]
    parents = spec.parents
    off = torch.as_tensor(spec.rest_offsets, dtype=local_rot6d.dtype,
                          device=local_rot6d.device)        # [J,3]
    gR = [None] * J
    gp = [None] * J
    for j in range(J):
        p = int(parents[j])
        if p < 0:
            gR[j] = R[:, j]
            gp[j] = torch.zeros(B, 3, dtype=local_rot6d.dtype, device=local_rot6d.device)
        else:
            gR[j] = gR[p] @ R[:, j]
            gp[j] = gp[p] + (gR[p] @ off[j])
    global_rot = torch.stack(gR, dim=1)
    global_pos = torch.stack(gp, dim=1)
    return global_rot, global_pos


# ----------------------------- spec builders --------------------------------

