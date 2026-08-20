import numpy as np
import torch

from alphamotion.perception.hybrid import (_resample,
                                            humanml_to_global_rot6d)


def _identity_humanml(frames: int = 3) -> np.ndarray:
    data = np.zeros((frames, 263), np.float32)
    identity = np.array([1, 0, 0, 0, 1, 0], np.float32)
    data[:, 67:193] = np.tile(identity, 21)
    data[:, 3] = 1.0
    return data


def test_humanml_native_rotations_and_root_match_alphamotion_contract():
    data = _identity_humanml()
    data[0, 1:3] = [1.0, 0.0]
    data[1, 1:3] = [0.0, 2.0]

    rot6d, root = humanml_to_global_rot6d(data)

    assert rot6d.shape == (3, 22, 6)
    assert root.shape == (3, 3)
    torch.testing.assert_close(
        rot6d, torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)
        .expand_as(rot6d))
    np.testing.assert_allclose(root, [[0, 0, 0], [100, 0, 0],
                                      [100, 0, 200]], atol=1e-5)


def test_hybrid_resample_preserves_contract_and_endpoints():
    rot6d, root = humanml_to_global_rot6d(_identity_humanml(4))
    root[-1] = [120.0, 5.0, -40.0]

    out_rot, out_root = _resample(rot6d, root, 11)

    assert out_rot.shape == (11, 22, 6)
    assert out_root.shape == (11, 3)
    np.testing.assert_allclose(out_root[0], root[0])
    np.testing.assert_allclose(out_root[-1], root[-1])
    assert torch.isfinite(out_rot).all()
