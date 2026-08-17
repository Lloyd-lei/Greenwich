import importlib.util
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from alphamotion.engine.nets.rotations import rot6d_to_matrix


SCRIPT = Path(__file__).parents[2] / "scripts" / "import_smpl_library.py"
SPEC = importlib.util.spec_from_file_location("import_smpl_library", SCRIPT)
IMPORTER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(IMPORTER)


def test_import_window_starts_at_source_frame_zero_not_activity_peak():
    poses = np.zeros((120, 66), np.float64)
    poses[0, 3] = 0.1
    # A deliberately energetic tail would have won the old activity crop.
    poses[60:, 3] = np.linspace(0.5, 2.5, 60)
    data = {
        "poses": poses,
        "trans": np.zeros((120, 3), np.float64),
        "mocap_framerate": np.asarray(30.0),
    }

    local6d, _root, fps = IMPORTER._window(data)
    first_joint = rot6d_to_matrix(torch.from_numpy(local6d[0, 1])).numpy()

    assert fps == 30.0
    np.testing.assert_allclose(
        first_joint, Rotation.from_rotvec([0.1, 0.0, 0.0]).as_matrix(),
        atol=1e-6)


def test_full_source_keeps_duration_hands_shape_and_gender():
    poses = np.zeros((473, 156), np.float64)
    poses[:, 66:156] = 0.25
    data = {
        "poses": poses,
        "trans": np.zeros((473, 3), np.float64),
        "betas": np.arange(16, dtype=np.float64),
        "gender": np.asarray("male"),
        "mocap_framerate": np.asarray(120.0),
    }

    motion = IMPORTER._motion(data)

    # 473 source frames at 120 Hz span 3.933 s -> 119 frames at 30 Hz.
    assert motion["local_rot6d"].shape == (119, 22, 6)
    assert motion["hand_pose"].shape == (119, 90)
    np.testing.assert_allclose(motion["hand_pose"], 0.25)
    np.testing.assert_array_equal(motion["betas"], np.arange(10))
    assert motion["gender"] == "male"
    assert motion["model_family"] == "smplh"
