import numpy as np
import torch

from alphamotion.perception.genmo import (smpl_root_translation,
                                          smpl_to_global_rot6d)


def _artifact():
    frames = 5
    return {
        "body_params_global": {
            "global_orient": torch.zeros(frames, 3),
            "body_pose": torch.zeros(frames, 63),
            "transl": torch.tensor([
                [9.0, 1.0, -2.0], [9.1, 1.0, -2.2],
                [9.2, 1.0, -2.4], [9.3, 1.0, -2.6],
                [9.4, 1.0, -2.8],
            ]),
        },
        "segment_info": [{"start": 2, "end": 5, "type": "text"}],
    }


def test_genmo_motion_keeps_rotation_and_anchored_world_translation():
    artifact = _artifact()
    rot = smpl_to_global_rot6d(artifact, "text")
    root = smpl_root_translation(artifact, "text")
    assert rot.shape == (3, 22, 6)
    assert root.shape == (3, 3)
    np.testing.assert_allclose(root[0], 0.0, atol=1e-8)
    np.testing.assert_allclose(root[-1], [20.0, 0.0, -40.0], atol=1e-4)
