"""Temporal composition utilities shared by the API and product tests."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .nets.rotations import matrix_to_rot6d, rot6d_to_matrix


def resample_continuous(values: np.ndarray, frames: int) -> np.ndarray:
    """Linearly resample a time-major continuous array without repeated rows."""
    x = np.asarray(values)
    if frames < 1:
        raise ValueError("frames must be positive")
    if len(x) == frames:
        return x.copy()
    if len(x) < 2:
        return np.repeat(x, frames, axis=0)
    old = np.linspace(0.0, 1.0, len(x))
    new = np.linspace(0.0, 1.0, frames)
    flat = x.reshape(len(x), -1)
    out = np.stack([np.interp(new, old, flat[:, i])
                    for i in range(flat.shape[1])], axis=1)
    return out.reshape((frames, *x.shape[1:])).astype(x.dtype, copy=False)


def interpolate_lattice(codes: torch.Tensor, frames: int) -> torch.Tensor:
    """Interpolate ordered FSQ coordinates along time, then snap to lattice."""
    if frames < 1:
        raise ValueError("frames must be positive")
    if len(codes) == frames:
        return codes.clone()
    x = codes.float().reshape(len(codes), -1).T[None]
    y = F.interpolate(x, size=frames, mode="linear", align_corners=True)
    return y[0].T.reshape(frames, *codes.shape[1:]).round().long()


def bridge_root(prev_root: np.ndarray, next_root: np.ndarray, frames: int):
    """Cubic root bridge with continuous boundary velocity.

    Inputs are a previous absolute trajectory and a next segment trajectory
    whose first row is local zero.  Returns `frames` interior points plus the
    absolute anchor where the next segment must start.  Horizontal momentum is
    continued; vertical displacement returns to the next clip's baseline while
    preserving take-off/landing derivatives.
    """
    if frames < 1:
        raise ValueError("bridge frames must be positive")
    prev = np.asarray(prev_root, np.float64)
    nxt = np.asarray(next_root, np.float64)
    if prev.ndim != 2 or prev.shape[1] != 3 or nxt.ndim != 2 or nxt.shape[1] != 3:
        raise ValueError("root trajectories must be [T,3]")
    start = prev[-1]
    v0 = prev[-1] - prev[-2] if len(prev) > 1 else np.zeros(3)
    v1 = nxt[1] - nxt[0] if len(nxt) > 1 else np.zeros(3)
    intervals = frames + 1
    displacement = 0.5 * (v0 + v1) * intervals
    displacement[1] = 0.0  # Y-up: the next segment starts at the same floor datum
    m0, m1 = v0 * intervals, v1 * intervals
    u = np.arange(1, frames + 1, dtype=np.float64) / intervals
    u2, u3 = u * u, u * u * u
    h10 = u3 - 2 * u2 + u
    h01 = -2 * u3 + 3 * u2
    h11 = u3 - u2
    local = h10[:, None] * m0 + h01[:, None] * displacement + h11[:, None] * m1
    return start + local, start + displacement


def repair_generated_holds(rot6d: torch.Tensor, stage: np.ndarray,
                           threshold_deg: float = 0.02):
    """Spread discrete-code pose holds across the following transition.

    FSQ paths are intentionally discrete. A bridge can therefore decode as
    ``A, A, B`` while its root keeps travelling, which looks like foot skate.
    This controller-facing pass changes generated frames only and replaces the
    repeated middle frame with the geodesic midpoint ``A, A/B, B``. Observed
    frames and both sides of every generated span remain untouched.
    """
    from scipy.spatial.transform import Rotation, Slerp

    stage = np.asarray(stage, np.int32)
    if rot6d.ndim != 3 or len(stage) != len(rot6d):
        raise ValueError("rot6d and stage must share a time dimension")
    R = rot6d_to_matrix(rot6d).detach().cpu().numpy()
    if len(R) < 3:
        return rot6d, {"repaired_frames": 0, "holds_before": 0,
                       "holds_after": 0}

    rel = R[1:] @ np.swapaxes(R[:-1], -1, -2)
    cos = np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1.0) * 0.5,
                  -1.0, 1.0)
    step = np.degrees(np.arccos(cos)).max(axis=1)
    eligible = (stage[:-1] == 1) | (stage[1:] == 1)
    held = (step < threshold_deg) & eligible
    original_holds = int(held.sum())
    repaired: set[int] = set()

    i = 0
    while i < len(held):
        if not held[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(held) and held[j + 1]:
            j += 1
        last = j + 1
        left = i if stage[i] != 1 or i == 0 else i - 1
        right = last if stage[last] != 1 else last + 1
        if left < right < len(R):
            key = np.array([0.0, 1.0])
            for joint in range(R.shape[1]):
                interp = Slerp(
                    key, Rotation.from_matrix(R[[left, right], joint]))
                for frame in range(left + 1, right):
                    if stage[frame] != 1:
                        continue
                    alpha = (frame - left) / (right - left)
                    R[frame, joint] = interp([alpha]).as_matrix()[0]
                    repaired.add(frame)
        i = j + 1

    out = matrix_to_rot6d(torch.as_tensor(
        R, device=rot6d.device, dtype=rot6d.dtype))
    rel2 = R[1:] @ np.swapaxes(R[:-1], -1, -2)
    cos2 = np.clip((np.trace(rel2, axis1=-2, axis2=-1) - 1.0) * 0.5,
                   -1.0, 1.0)
    step2 = np.degrees(np.arccos(cos2)).max(axis=1)
    return out, {"repaired_frames": len(repaired),
                 "holds_before": original_holds,
                 "holds_after": int(((step2 < threshold_deg)
                                      & eligible).sum())}


def repair_generated_joint_holds(q: torch.Tensor, global_rot: torch.Tensor,
                                 stage: np.ndarray,
                                 threshold_deg: float = 0.02):
    """Remove holds that reappear after mechanical projection.

    Interpolating global rotations before projection is not sufficient: the
    per-frame inverse-kinematics solve can map nearby rotations back onto the
    same joint-limit branch.  At this point ``q`` is already feasible and
    branch-clean, so interpolation inside the joint limits is the stable
    operation.  Only generated interior frames are changed; observed and
    task-constrained frames remain exact anchors.
    """
    stage = np.asarray(stage, np.int32)
    if q.ndim != 3 or global_rot.ndim != 4 or len(stage) != len(q):
        raise ValueError("q, global_rot and stage must share time")
    if len(q) < 3:
        return q, {"repaired_frames": 0, "holds_before": 0,
                   "holds_after": 0}

    R = global_rot.detach().cpu().numpy()
    rel = R[1:] @ np.swapaxes(R[:-1], -1, -2)
    cos = np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1.0) * 0.5,
                  -1.0, 1.0)
    step = np.degrees(np.arccos(cos)).max(axis=1)
    eligible = (stage[:-1] == 1) | (stage[1:] == 1)
    held = (step < threshold_deg) & eligible
    original_holds = int(held.sum())

    out = q.clone()
    repaired: set[int] = set()
    i = 0
    while i < len(held):
        if not held[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(held) and held[j + 1]:
            j += 1
        last = j + 1
        left = i if stage[i] != 1 or i == 0 else i - 1
        right = last if stage[last] != 1 else last + 1
        if left < right < len(out):
            for frame in range(left + 1, right):
                if stage[frame] != 1:
                    continue
                alpha = (frame - left) / (right - left)
                out[frame] = ((1.0 - alpha) * out[left]
                              + alpha * out[right])
                repaired.add(frame)
        i = j + 1

    # Joint-space equality is the relevant post-projection hold condition.
    qstep = torch.rad2deg(
        (out[1:] - out[:-1]).abs().reshape(len(out) - 1, -1).amax(dim=1))
    after = int(((qstep.detach().cpu().numpy() < threshold_deg)
                 & eligible).sum())
    return out, {"repaired_frames": len(repaired),
                 "holds_before": original_holds,
                 "holds_after": after}


def repair_projection_branch_flips(q: torch.Tensor, jump_deg: float = 120.0,
                                   max_span: int = 8,
                                   boundary_deg: float = 45.0):
    """Remove short inverse-kinematics branch islands from hinge angles.

    Independent per-frame projection can choose ``+97, -60, -60, +94`` for
    the same elbow even though the decoded rotations are temporally smooth.
    Only a pair of implausibly large jumps enclosing a short island is fixed,
    and only when the poses on the two outer boundaries agree. Sustained turns
    and one-way fast actions are left unchanged.
    """
    if q.ndim != 3:
        raise ValueError("q must have shape [T,J,3]")
    arr = q.detach().cpu().numpy().copy()
    flat = arr.reshape(len(arr), -1)
    jump = np.deg2rad(jump_deg)
    boundary = np.deg2rad(boundary_deg)
    repaired: set[tuple[int, int]] = set()
    before = 0
    for axis in range(flat.shape[1]):
        transitions = np.flatnonzero(np.abs(np.diff(flat[:, axis])) > jump)
        before += len(transitions)
        p = 0
        while p + 1 < len(transitions):
            left, right = int(transitions[p]), int(transitions[p + 1])
            if (right - left <= max_span
                    and abs(flat[right + 1, axis]
                            - flat[left, axis]) < boundary):
                values = np.linspace(flat[left, axis], flat[right + 1, axis],
                                     right - left + 2)
                flat[left + 1:right + 1, axis] = values[1:-1]
                for frame in range(left + 1, right + 1):
                    repaired.add((frame, axis))
                p += 2
            else:
                p += 1
    after = int((np.abs(np.diff(flat, axis=0)) > jump).sum())
    out = torch.as_tensor(arr, device=q.device, dtype=q.dtype)
    return out, {"repaired_values": len(repaired),
                 "large_jumps_before": before,
                 "large_jumps_after": after}


def repair_generated_joint_jumps(q: torch.Tensor, global_rot: torch.Tensor,
                                 stage: np.ndarray,
                                 threshold_deg: float = 90.0,
                                 radius: int = 8):
    """Smooth physically impossible branch jumps inside generated spans.

    Per-frame mechanical projection can select a different valid angle branch
    at a generated-to-observed boundary.  This pass detects the failure in
    global joint rotation space, then interpolates only the adjacent generated
    joint coordinates.  Observed frames (stage 0) and task-constrained frames
    (stage 2) are immutable anchors.
    """
    stage = np.asarray(stage, np.int32)
    if q.ndim != 3 or global_rot.ndim != 4 or len(stage) != len(q):
        raise ValueError("q, global_rot and stage must share time")
    if radius < 1:
        raise ValueError("radius must be positive")
    if len(q) < 2:
        return q, {"repaired_frames": 0, "repaired_joints": 0,
                   "large_jumps_before": 0}

    R = global_rot.detach().cpu().numpy()
    rel = R[1:] @ np.swapaxes(R[:-1], -1, -2)
    cos = np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1.0) * 0.5,
                  -1.0, 1.0)
    joint_step = np.degrees(np.arccos(cos))
    generated_edge = ((stage[:-1] == 1) | (stage[1:] == 1))[:, None]
    bad = np.argwhere((joint_step > threshold_deg) & generated_edge)
    if not len(bad):
        return q, {"repaired_frames": 0, "repaired_joints": 0,
                   "large_jumps_before": 0}

    # Aggregate nearby failures by generated run and joint. A single
    # interpolation then repairs both sides of a short branch island without
    # repeatedly rewriting the same frames.
    groups: dict[tuple[int, int, int], list[int]] = {}
    for transition, joint in bad:
        generated_frame = (int(transition) if stage[transition] == 1
                           else int(transition) + 1)
        run_start = generated_frame
        while run_start > 0 and stage[run_start - 1] == 1:
            run_start -= 1
        run_end = generated_frame
        while run_end + 1 < len(stage) and stage[run_end + 1] == 1:
            run_end += 1
        groups.setdefault((run_start, run_end, int(joint)), []).append(
            int(transition))

    out = q.clone()
    repaired_frames: set[int] = set()
    repaired_joints: set[tuple[int, int]] = set()
    for (run_start, run_end, joint), transitions in groups.items():
        left = max(run_start - 1, min(transitions) - radius)
        right = min(run_end + 1, max(transitions) + 1 + radius)
        if left >= right:
            continue
        left_q = out[left, joint].clone()
        right_q = out[right, joint].clone()
        for frame in range(left + 1, right):
            if stage[frame] != 1:
                continue
            alpha = (frame - left) / (right - left)
            out[frame, joint] = ((1.0 - alpha) * left_q
                                 + alpha * right_q)
            repaired_frames.add(frame)
            repaired_joints.add((frame, joint))

    return out, {
        "repaired_frames": len(repaired_frames),
        "repaired_joints": len(repaired_joints),
        "large_jumps_before": int(len(bad)),
        "threshold_deg": float(threshold_deg),
    }
