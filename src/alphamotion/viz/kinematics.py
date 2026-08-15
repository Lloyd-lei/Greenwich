"""Shared MuJoCo pose plumbing for every AlphaMotion render surface.

The interactive viewer and the MP4 exporter must show the same motion.  Keep
the coordinate conversion, mesh selection, joint mapping, and root placement
here so one surface cannot silently diverge from another.
"""
from __future__ import annotations

import numpy as np


# Row-vector conversion for root-relative joint positions: Y-up -> Z-up.
YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float64)


def visual_mesh_geom_ids(model) -> list[int]:
    """Return render meshes, excluding duplicate collision geometry.

    Vendor MJCF group numbers are not consistent: H1 uses group 2 for its
    body and group 1 for its hands, while other robots use group 1 or 2 for
    the whole visual model.  Collision flags are the stable signal.
    """
    import mujoco as mj

    meshes = [i for i in range(model.ngeom)
              if model.geom_type[i] == mj.mjtGeom.mjGEOM_MESH]
    visual = [i for i in meshes
              if model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0]
    return visual or meshes


def joint_qpos_map(model, qnames) -> np.ndarray:
    """Descriptor joint axes -> MuJoCo qpos addresses."""
    import mujoco as mj

    tab = -np.ones((len(qnames), 3), np.int64)
    for j, names in enumerate(qnames):
        for k, name in enumerate(names):
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                tab[j, k] = int(model.jnt_qposadr[jid])
    return tab


def source_joint_map(spec, trace) -> list[int]:
    """Map descriptor joints to trace q columns by name when available."""
    if not getattr(trace, "joint_names", None):
        return list(range(spec.J))
    lut = {name: i for i, name in enumerate(trace.joint_names)}
    return [lut.get(name, -1) for name in spec.joint_names]


def free_root_address(model) -> int:
    import mujoco as mj

    roots = [int(model.jnt_qposadr[j]) for j in range(model.njnt)
             if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE]
    return roots[0] if roots else -1


def root_world_offsets(root_t, frames: int) -> np.ndarray:
    """Convert root translation [T,3] cm Y-up to [T,3] m Z-up.

    Corpus root trajectories are first-frame anchored, but subtracting the
    first row here also makes user traces deterministic if they are not.
    The empirically audited horizontal convention is (x,y,z)->(z,x,y).
    """
    if root_t is None:
        return np.zeros((frames, 3), np.float64)
    root = np.asarray(root_t, np.float64)
    if root.shape != (frames, 3):
        raise ValueError(f"root_t must have shape ({frames}, 3), got {root.shape}")
    if not np.isfinite(root).all():
        raise ValueError("root_t contains NaN or infinity")
    root = root - root[0]
    return np.stack([root[:, 2], root[:, 0], root[:, 1]], axis=1) / 100.0


def smooth_camera_path(path: np.ndarray, fps: float,
                       window_s: float = 0.30) -> np.ndarray:
    """Low-pass a root trajectory without introducing temporal phase lag.

    A camera should follow locomotion, not the frame-to-frame movement of a
    wrist or foot.  Edge padding avoids the startup/loop snap introduced by
    zero-padded convolutions, while an odd Hann window keeps the filter
    centred in time and keeps endpoint drift bounded.
    """
    points = np.asarray(path, np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"camera path must be [T,3], got {points.shape}")
    if len(points) < 3 or not np.isfinite(points).all():
        return points.copy()
    width = max(3, int(round(max(float(fps), 1.0) * window_s)))
    width += 1 - width % 2
    width = min(width, len(points) if len(points) % 2 else len(points) - 1)
    if width < 3:
        return points.copy()
    kernel = np.hanning(width)
    kernel /= kernel.sum()
    pad = width // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([
        np.convolve(padded[:, axis], kernel, mode="valid")
        for axis in range(3)
    ], axis=1)


def apply_trace_pose(model, data, trace, spec, qpos_map, src_of,
                     root_adr: int, frame: int, root_xyz=(0.0, 0.0, 0.0)) -> None:
    """Write one trace frame into MuJoCo and run forward kinematics."""
    import mujoco as mj
    from scipy.spatial.transform import Rotation

    data.qpos[:] = model.qpos0
    if root_adr >= 0:
        data.qpos[root_adr:root_adr + 3] = root_xyz
        data.qpos[root_adr + 3:root_adr + 7] = Rotation.from_matrix(
            YUP_TO_ZUP.T @ trace.rootR[frame] @ YUP_TO_ZUP
        ).as_quat(scalar_first=True)
    for j in range(spec.J):
        sj = src_of[j]
        if sj < 0 or sj >= trace.q.shape[1]:
            continue
        for k in range(3):
            if qpos_map[j, k] >= 0:
                data.qpos[qpos_map[j, k]] = trace.q[frame, sj, k]
    mj.mj_forward(model, data)


def first_frame_ground_height(model, data, trace, spec, qpos_map, src_of,
                              root_adr: int) -> float:
    """Place the first frame on z=0 once; never re-ground later frames."""
    if root_adr < 0:
        return 0.0
    apply_trace_pose(model, data, trace, spec, qpos_map, src_of, root_adr, 0)
    low = lowest_visual_z(model, data)
    return -low if np.isfinite(low) else 0.0


def lowest_visual_z(model, data) -> float:
    """Exact lowest world-space vertex among the visible robot meshes."""
    lows = []
    for gid in visual_mesh_geom_ids(model):
        mid = int(model.geom_dataid[gid])
        if mid < 0:
            continue
        va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        vertices = np.asarray(model.mesh_vert[va:va + vn])
        if not len(vertices):
            continue
        rotation = np.asarray(data.geom_xmat[gid]).reshape(3, 3)
        world_z = vertices @ rotation[2] + float(data.geom_xpos[gid, 2])
        lows.append(float(world_z.min()))
    return min(lows) if lows else float("nan")


def apply_ground_safe_pose(model, data, trace, spec, qpos_map, src_of,
                           root_adr: int, frame: int, root_xyz,
                           floor_z: float = 0.0) -> float:
    """Apply a pose and lift only genuine floor penetration.

    This is deliberately one-sided: airborne motion is preserved, while a
    codec/retarget mismatch cannot draw feet below the floor. Returns the
    applied vertical safety correction in metres.
    """
    apply_trace_pose(model, data, trace, spec, qpos_map, src_of, root_adr,
                     frame, root_xyz)
    if root_adr < 0:
        return 0.0
    low = lowest_visual_z(model, data)
    correction = max(0.0, floor_z - low) if np.isfinite(low) else 0.0
    if correction > 1e-5:
        data.qpos[root_adr + 2] += correction
        import mujoco as mj
        mj.mj_forward(model, data)
    return correction
