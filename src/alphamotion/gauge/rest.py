"""Rest-frame repair: re-express ANY skeleton's joint frames in the gauge's
rest convention without moving a single point.

Gauge rule (contract.RotationConvention.rest_identity), grounded in the 0818
measurements against robot ground truth:
    identity rotation <=> a NEUTRAL pose (SMPL T-pose for SMPL-defined
    skeletons; otherwise the rig's bind pose or a detected standing frame),
    with the subject FACING +Z (root yaw normalised) at that pose;
    rest_off = world bone vectors at that pose.

What the release codec tolerates (AMASS -> unitree_g1_29dof vs GMR GT):
    native SMPL T-pose convention            23.2 deg / 6.9 cm
    standing-frame calibration, de-yawed     28.5 deg / 7.1 cm   (acceptable)
    standing-frame calibration WITH yaw      43-46 deg / 9-10 cm (yaw kills it)
    calibration at an arbitrary frame        85-90 deg           (fatal)
    "repair" onto SMPL's T-pose DIRECTION table from another rig (SOMA):
                                             worse than the neutral frame —
    joint definitions differ between rigs, and twist is not observable
    from directions.  So the repair is a de-yawed neutral-frame calibration,
    never a direction table.

Mechanics.  A per-joint constant rotation Q_j re-frames without moving points:
    R'_j(t) = R_j(t) @ Q_j            (global rotations)
    off'_j  = Q_par(j)^T @ off_j      (rest offsets; root offset untouched)
    L'_j(t) = Q_par(j)^T @ L_j(t) @ Q_j   (local rotations; root: L_0 @ Q_0)
FK positions are invariant for every Q.  calibrate_neutral picks
Q_j = (Y^T R_j(t*))^T where t* is the neutral frame and Y the root yaw there.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..engine.nets.rotations import SkeletonSpec, fk_global, matrix_to_rot6d, rot6d_to_matrix


def _angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def yaw_matrix(yaw: float) -> np.ndarray:
    """Rotation about +Y by `yaw` (radians), Y-up right-handed."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)


def root_yaw(R_root: np.ndarray) -> float:
    """Yaw (radians) of the root frame's forward (+Z) axis versus world +Z."""
    f = R_root @ np.array([0.0, 0.0, 1.0])
    return float(np.arctan2(f[0], f[2]))


@dataclass
class RestRepair:
    """Per-joint frame rotations + the repaired spec (see module docstring)."""
    spec: SkeletonSpec                 # repaired spec
    source: SkeletonSpec               # original spec
    Q: np.ndarray                      # [J,3,3]  R' = R @ Q
    method: str = ""
    angle_deg: np.ndarray = None       # per joint |Q_j|
    meta: dict = field(default_factory=dict)

    @property
    def is_identity(self) -> bool:
        return bool(np.all(self.angle_deg < 1e-3))

    def summary(self) -> dict:
        return {"method": self.method,
                "max_angle_deg": float(self.angle_deg.max()),
                "mean_angle_deg": float(self.angle_deg.mean()), **self.meta}


def make_repair(spec: SkeletonSpec, Q: np.ndarray, method: str, **meta) -> RestRepair:
    Q = np.asarray(Q, np.float64)
    off = spec.rest_offsets.astype(np.float64).copy()
    for j in range(1, spec.J):
        off[j] = Q[spec.parents[j]].T @ off[j]
    new = SkeletonSpec(spec.name, spec.parents, off.astype(np.float32),
                       list(spec.joint_names))
    ang = np.array([_angle(Q[j]) for j in range(spec.J)])
    return RestRepair(spec=new, source=spec, Q=Q, method=method, angle_deg=ang,
                      meta=dict(meta))


def calibrate_neutral(spec: SkeletonSpec, global_rot: torch.Tensor,
                      deyaw: bool = True) -> RestRepair:
    """Neutral-frame calibration.  `global_rot` [J,6] or [J,3,3]: the global
    joint rotations at the chosen neutral frame (T-pose / bind / standing).
    After repair that frame carries identity rotations everywhere and, with
    deyaw=True, the subject faces +Z there (its world yaw is removed from
    every joint so the pose <-> facing relation the codec learned is kept)."""
    R = global_rot
    if R.shape[-1] == 6:
        R = rot6d_to_matrix(R)
    R = R.detach().cpu().double().numpy()
    yaw = root_yaw(R[0]) if deyaw else 0.0
    Y = yaw_matrix(yaw)
    Q = np.stack([(Y.T @ R[j]).T for j in range(spec.J)])
    return make_repair(spec, Q, "neutral_frame", removed_yaw_deg=float(np.degrees(yaw)))


def find_neutral_frame(local_rot6d: torch.Tensor, spec: SkeletonSpec,
                       stride: int = 1) -> int:
    """Index of the most 'standing' frame: upright, wrists low and near the
    body.  Same rule the research stack used for the bones seed (0808)."""
    _, gp = fk_global(local_rot6d[::stride].float(), spec)
    names = [n.lower() for n in spec.joint_names]
    def find(*keys):
        for k in keys:
            for i, n in enumerate(names):
                if k in n:
                    return i
        return None
    iH = find("head", "neck")
    iL, iR = find("l_wrist", "lefthand", "left_wrist", "l_hand"), \
             find("r_wrist", "righthand", "right_wrist", "r_hand")
    up = torch.nn.functional.normalize(gp[:, iH], dim=-1)[:, 1] if iH is not None else torch.zeros(len(gp))
    score = up * 100
    if iL is not None and iR is not None:
        score = score - (gp[:, iL, 1] + gp[:, iR, 1]) * 0.5 \
                      - (gp[:, iL].norm(dim=-1) + gp[:, iR].norm(dim=-1)) * 0.5
    return int(score.argmax()) * stride


def reframe_pose(global_rot6d: torch.Tensor, repair: RestRepair) -> torch.Tensor:
    """[...,J,6] global rot6d in the SOURCE convention -> repaired (R' = R Q)."""
    Q = torch.as_tensor(repair.Q, dtype=global_rot6d.dtype, device=global_rot6d.device)
    return matrix_to_rot6d(rot6d_to_matrix(global_rot6d) @ Q)


def reframe_local(local_rot6d: torch.Tensor, repair: RestRepair) -> torch.Tensor:
    """[...,J,6] LOCAL rot6d (root slot = global root) -> repaired convention.
    L'_j = Q_par^T L_j Q_j ; root: L_0 Q_0.  FK positions unchanged."""
    Q = torch.as_tensor(repair.Q, dtype=local_rot6d.dtype, device=local_rot6d.device)
    L = rot6d_to_matrix(local_rot6d)
    par = repair.spec.parents
    out = L.clone()
    out[..., 0, :, :] = L[..., 0, :, :] @ Q[0]
    for j in range(1, repair.spec.J):
        out[..., j, :, :] = Q[par[j]].T @ L[..., j, :, :] @ Q[j]
    return matrix_to_rot6d(out)
