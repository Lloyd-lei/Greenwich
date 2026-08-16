"""Shared MuJoCo pose plumbing for every AlphaMotion render surface.

The interactive viewer and the MP4 exporter must show the same motion.  Keep
the coordinate conversion, mesh selection, joint mapping, and root placement
here so one surface cannot silently diverge from another.
"""
from __future__ import annotations

import numpy as np


# Row-vector conversion for root-relative joint positions: Y-up -> Z-up.
YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float64)

# Descriptor skeletons use X for left/right.  MuJoCo's humanoid convention is
# X-forward/Y-left, so an overlay on an attached MJCF needs a +90 degree yaw
# (counter-clockwise around Z).  This is the row-vector form of Rz(+pi/2).
ZUP_CCW_90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], np.float64)


def _align_vector_step(source: np.ndarray, target: np.ndarray,
                       angle: float | None = None) -> np.ndarray:
    """Minimal column-vector rotation taking source toward target."""
    a = np.asarray(source, np.float64)
    b = np.asarray(target, np.float64)
    a /= np.linalg.norm(a) + 1e-12
    b /= np.linalg.norm(b) + 1e-12
    cross = np.cross(a, b)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    full = float(np.arctan2(sine, cosine))
    if full < 1e-9:
        return np.eye(3)
    if sine < 1e-9:  # antiparallel; choose a stable perpendicular axis
        cross = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(cross) < 1e-6:
            cross = np.cross(a, [0.0, 0.0, 1.0])
    axis = cross / (np.linalg.norm(cross) + 1e-12)
    theta = full if angle is None else min(full, max(float(angle), 0.0))
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def balanced_root_rotations(root_rotations: np.ndarray,
                            max_tilt_deg: float = 8.0) -> np.ndarray:
    """Remove retargeting lean bias and cap root roll/pitch.

    Root yaw and frame-to-frame dynamics are preserved.  First a single
    minimal correction aligns the median pelvis-up vector with world up, then
    only excess per-frame tilt beyond ``max_tilt_deg`` is removed.  This is a
    kinematic balance guard, not a physics controller.
    """
    rotations = np.asarray(root_rotations, np.float64)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(f"root rotations must be [T,3,3], got {rotations.shape}")
    if not np.isfinite(rotations).all() or not len(rotations):
        return rotations.copy()
    target_up = np.array([0.0, 1.0, 0.0])
    up = rotations[:, :, 1]
    median_up = np.median(up, axis=0)
    bias = _align_vector_step(median_up, target_up)
    out = bias[None] @ rotations
    limit = np.deg2rad(max(0.0, float(max_tilt_deg)))
    for t in range(len(out)):
        current_up = out[t, :, 1]
        tilt = float(np.arccos(np.clip(
            np.dot(current_up, target_up)
            / (np.linalg.norm(current_up) + 1e-12), -1.0, 1.0)))
        if tilt > limit:
            out[t] = _align_vector_step(
                current_up, target_up, tilt - limit) @ out[t]
    return out


def preview_joint_positions(rest_yup_cm: np.ndarray,
                            align_to_mjcf: bool = False) -> np.ndarray:
    """Convert canonical rest joints to grounded Z-up preview coordinates.

    Meshless descriptor previews retain their canonical heading.  An attached
    MJCF uses MuJoCo's X-forward/Y-left frame, which differs by a +90 degree
    yaw from the descriptor frame.
    """
    points = np.asarray(rest_yup_cm, np.float64) @ YUP_TO_ZUP / 100.0
    if align_to_mjcf:
        points = points @ ZUP_CCW_90
    points[:, 2] -= float(points[:, 2].min())
    return points


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


def world_offsets_to_root_cm(offsets: np.ndarray) -> np.ndarray:
    """Inverse of :func:`root_world_offsets`, anchored at frame zero."""
    world = np.asarray(offsets, np.float64)
    if world.ndim != 2 or world.shape[1] != 3:
        raise ValueError(f"world offsets must be [T,3], got {world.shape}")
    world = world - world[0]
    return np.stack([world[:, 1], world[:, 2], world[:, 0]], axis=1) * 100.0


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
        balanced = getattr(trace, "_balanced_rootR", None)
        if balanced is None:
            balanced = balanced_root_rotations(trace.rootR)
            # MotionTrace is an in-memory rendering contract; caching avoids
            # recomputing the whole trajectory for every displayed frame.
            trace._balanced_rootR = balanced
        data.qpos[root_adr + 3:root_adr + 7] = Rotation.from_matrix(
            YUP_TO_ZUP.T @ balanced[frame] @ YUP_TO_ZUP
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


def contact_stabilized_root_offsets(model, data, trace, spec, qpos_map,
                                    src_of, root_adr: int,
                                    ground_z: float):
    """Target-mesh world trajectory with planted-foot correction.

    Source root motion remains the initial guess.  The final target robot is
    posed first, then the stance solver removes its measured support-foot
    residual.  This handles morphology and joint-projection differences that
    a copied human trajectory cannot account for.
    """
    from ..engine.odometry import foot_bodies, stance_offsets
    offsets = root_world_offsets(getattr(trace, "root_t", None), trace.frames)
    if getattr(trace, "contact_stabilized", False):
        return offsets, {"available": True, "already_baked": True}
    if root_adr < 0:
        return offsets, {"available": False, "reason": "no free root"}
    feet = foot_bodies(model)
    if not feet:
        return offsets, {"available": False, "reason": "no foot bodies"}
    foot_world = np.zeros((trace.frames, len(feet), 3), np.float64)
    for t in range(trace.frames):
        xyz = offsets[t].copy()
        xyz[2] += ground_z
        apply_ground_safe_pose(model, data, trace, spec, qpos_map, src_of,
                               root_adr, t, xyz)
        for i, body_id in enumerate(feet):
            foot_world[t, i] = data.xpos[body_id]
    correction, report = stance_offsets(foot_world, return_report=True)
    offsets[:, :2] += correction
    report["foot_bodies"] = [int(x) for x in feet]
    report["correction_span_m"] = [
        round(float(x), 4) for x in np.ptp(correction, axis=0)]
    return offsets, report


def stabilize_trace_root_translation(trace, xml: str, body: str):
    """Bake target-specific contact correction into a trace root trajectory."""
    import mujoco as mj
    from ..engine.descriptor import build_from_mjcf
    model = mj.MjModel.from_xml_path(str(xml))
    data = mj.MjData(model)
    spec, _dof, _rest, qnames, _ = build_from_mjcf(xml, body)
    qpos_map = joint_qpos_map(model, qnames)
    src_of = source_joint_map(spec, trace)
    root_adr = free_root_address(model)
    ground_z = first_frame_ground_height(
        model, data, trace, spec, qpos_map, src_of, root_adr)
    offsets, report = contact_stabilized_root_offsets(
        model, data, trace, spec, qpos_map, src_of, root_adr, ground_z)
    return world_offsets_to_root_cm(offsets), report
