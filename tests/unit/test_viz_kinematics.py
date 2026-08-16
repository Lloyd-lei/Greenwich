from types import SimpleNamespace

import mujoco as mj
import numpy as np

from alphamotion.viz.kinematics import (balanced_root_rotations,
                                        preview_joint_positions,
                                        root_world_offsets,
                                        smooth_camera_path,
                                        visual_mesh_geom_ids,
                                        world_offsets_to_root_cm)


def test_root_offsets_keep_all_three_axes():
    root = np.array([[10, 20, 30], [12, 25, 37]], np.float64)
    out = root_world_offsets(root, 2)
    np.testing.assert_allclose(out[0], 0)
    np.testing.assert_allclose(out[1], [0.07, 0.02, 0.05])
    np.testing.assert_allclose(
        root_world_offsets(world_offsets_to_root_cm(out), 2), out)


def test_root_balance_removes_bias_and_caps_dynamic_tilt():
    from scipy.spatial.transform import Rotation
    rotations = Rotation.from_euler(
        "xyz", [[12, 20, 0], [17, 50, 0], [7, 80, 0]], degrees=True
    ).as_matrix()
    fixed = balanced_root_rotations(rotations, max_tilt_deg=8.0)
    up = fixed[:, :, 1]
    tilt = np.degrees(np.arccos(np.clip(up[:, 1], -1.0, 1.0)))

    original_up = rotations[:, :, 1]
    original_tilt = np.degrees(np.arccos(np.clip(
        original_up[:, 1], -1.0, 1.0)))
    assert float(np.median(tilt)) < float(np.median(original_tilt)) * 0.6
    assert float(tilt.max()) <= 8.0001
    # Proper rotations are preserved.
    np.testing.assert_allclose(
        fixed @ np.swapaxes(fixed, -1, -2),
        np.repeat(np.eye(3)[None], len(fixed), axis=0), atol=1e-7)
    np.testing.assert_allclose(np.linalg.det(fixed), 1.0, atol=1e-7)


def test_mjcf_preview_rotates_skeleton_counter_clockwise_90_degrees():
    # Descriptor X is left/right.  After the +90 degree Z yaw, positive X
    # points along MuJoCo's positive Y (the robot's left side).
    rest = np.array([[0, -100, 0], [25, 0, 0], [-25, 0, 0]], np.float64)
    canonical = preview_joint_positions(rest)
    aligned = preview_joint_positions(rest, align_to_mjcf=True)

    np.testing.assert_allclose(canonical[1, :2], [0.25, 0.0])
    np.testing.assert_allclose(aligned[1, :2], [0.0, 0.25])
    np.testing.assert_allclose(aligned[2, :2], [0.0, -0.25])
    assert aligned[:, 2].min() == 0.0


def test_visual_mesh_selection_uses_collision_flags_not_group_number():
    model = SimpleNamespace(
        ngeom=5,
        geom_type=np.array([mj.mjtGeom.mjGEOM_MESH] * 4 +
                           [mj.mjtGeom.mjGEOM_BOX]),
        geom_group=np.array([2, 3, 1, 0, 0]),
        geom_contype=np.array([0, 1, 0, 1, 0]),
        geom_conaffinity=np.array([0, 1, 0, 1, 0]),
    )
    assert visual_mesh_geom_ids(model) == [0, 2]


def test_camera_path_filters_jitter_without_time_shift():
    path = np.zeros((31, 3), np.float64)
    path[:, 0] = np.linspace(0.0, 3.0, len(path))
    path[:, 1] = np.where(np.arange(len(path)) % 2, 0.12, -0.12)
    smooth = smooth_camera_path(path, fps=30.0)

    assert smooth.shape == path.shape
    assert np.max(np.abs(np.diff(smooth[:, 1]))) <= 0.031
    assert np.all(np.diff(smooth[:, 0]) >= 0.0)
    assert abs(smooth[0, 0] - path[0, 0]) < 0.07
    assert abs(smooth[-1, 0] - path[-1, 0]) < 0.07
