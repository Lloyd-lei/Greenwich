import numpy as np
import torch

from alphamotion.engine.nets.rotations import (SkeletonSpec, fk_global,
                                               geodesic_deg, matrix_to_rot6d,
                                               rot6d_to_matrix)


def test_rot6d_roundtrip():
    g = torch.Generator().manual_seed(0)
    R = rot6d_to_matrix(torch.randn(64, 6, generator=g))
    R2 = rot6d_to_matrix(matrix_to_rot6d(R))
    # float32 Gram-Schmidt reconstruction is good to ~0.1 deg
    assert float(geodesic_deg(R, R2).max()) < 0.5


def test_fk_chain():
    spec = SkeletonSpec("t", [-1, 0, 1], np.array(
        [[0, 0, 0], [0, 1, 0], [0, 1, 0]], np.float32), ["a", "b", "c"])
    ident = torch.zeros(1, 3, 6)
    ident[..., 0] = 1
    ident[..., 4] = 1
    _R, p = fk_global(ident, spec)
    assert float(p[0, 2, 1]) == 2.0     # straight chain of two unit bones
