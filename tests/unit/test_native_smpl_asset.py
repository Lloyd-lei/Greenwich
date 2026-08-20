import importlib.util
from pathlib import Path

import numpy as np


def _importer_module():
    path = Path(__file__).parents[2] / "scripts/import_smpl_library.py"
    spec = importlib.util.spec_from_file_location("import_smpl_library", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_native_genmo_asset_uses_local_rotation_contract():
    module = _importer_module()
    frames = 7
    local = np.zeros((frames, 22, 6), np.float32)
    local[..., 0] = 1.0
    local[..., 4] = 1.0
    root = np.arange(frames * 3, dtype=np.float32).reshape(frames, 3)
    payload = {
        "local_rot6d": local,
        "root_cm": root,
        "hand_pose": np.zeros((frames, 90), np.float32),
        "betas": np.zeros(10, np.float32),
        "model_family": np.asarray(0, np.uint8),
        "fps": np.asarray(30.0),
    }

    motion = module._motion(payload)

    np.testing.assert_array_equal(motion["local_rot6d"], local)
    np.testing.assert_array_equal(motion["root_cm"], root)
    assert motion["source_fps"] == 30.0
    assert motion["model_family"] == "smpl"
