from types import SimpleNamespace

import mujoco as mj
import numpy as np

from alphamotion.viz.kinematics import (root_world_offsets,
                                        smooth_camera_path,
                                        visual_mesh_geom_ids)


def test_root_offsets_keep_all_three_axes():
    root = np.array([[10, 20, 30], [12, 25, 37]], np.float64)
    out = root_world_offsets(root, 2)
    np.testing.assert_allclose(out[0], 0)
    np.testing.assert_allclose(out[1], [0.07, 0.02, 0.05])


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
