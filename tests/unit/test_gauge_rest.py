"""Gauge rest repair: FK invariance, neutral frame -> identity, de-yaw, gates."""
import numpy as np
import torch

from alphamotion.engine.nets.rotations import SkeletonSpec, fk_global, matrix_to_rot6d, rot6d_to_matrix
from alphamotion.gauge import (GAUGE, calibrate_neutral, find_neutral_frame,
                               fk_invariance_cm, reframe_local, reframe_pose,
                               validate_rest)
from alphamotion.gauge.rest import root_yaw, yaw_matrix


def _spec():
    names = ["Pelvis", "L_Hip", "R_Hip", "Neck", "Head", "L_Knee", "L_Ankle",
             "L_Shoulder", "L_Wrist"]
    parents = [-1, 0, 0, 0, 3, 1, 5, 3, 7]
    off = np.array([[0, 90, 0], [10, -5, 0], [-10, -5, 0], [0, 40, 0], [0, 15, 0],
                    [0, -40, 0], [0, -40, 0], [15, 0, 0], [30, 0, 0]], np.float32)
    return SkeletonSpec("toy", parents, off, names)


def _rand_local(T, J, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(T, J, 3, generator=g) * 0.4
    # small random rotations via axis-angle
    th = a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    k = a / th
    K = torch.zeros(T, J, 3, 3)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    R = torch.eye(3) + torch.sin(th)[..., None] * K + (1 - torch.cos(th))[..., None] * K @ K
    return matrix_to_rot6d(R)


def test_reframe_is_fk_invariant_and_neutral_frame_is_identity():
    spec = _spec()
    L = _rand_local(12, spec.J)
    gR, _ = fk_global(L, spec)
    rep = calibrate_neutral(spec, gR[3], deyaw=False)
    L2 = reframe_local(L, rep)
    assert fk_invariance_cm(L, spec, L2, rep.spec) < 1e-3
    gR2, _ = fk_global(L2, rep.spec)
    assert torch.allclose(gR2[3], torch.eye(3).expand(spec.J, 3, 3), atol=1e-4)
    # global-path reframe agrees with local-path reframe
    g6 = matrix_to_rot6d(gR)
    assert torch.allclose(rot6d_to_matrix(reframe_pose(g6, rep)), gR2, atol=1e-4)


def test_deyaw_makes_neutral_frame_face_forward():
    spec = _spec()
    L = _rand_local(6, spec.J, seed=1)
    Y = torch.as_tensor(yaw_matrix(np.radians(70)), dtype=torch.float32)
    L[0] = matrix_to_rot6d(torch.eye(3).expand(spec.J, 3, 3))      # frame 0 = standing ...
    L[:, 0] = matrix_to_rot6d(Y @ rot6d_to_matrix(L[:, 0]))       # ... subject turned 70 deg
    gR, _ = fk_global(L, spec)
    rep = calibrate_neutral(spec, gR[0], deyaw=True)
    assert abs(rep.meta["removed_yaw_deg"]) > 1.0
    gR2, _ = fk_global(reframe_local(L, rep), rep.spec)
    # de-yaw keeps the world facing in the ROTATIONS (root = pure yaw at the
    # neutral frame) and puts +Z facing into the REST OFFSETS — so rotations
    # and positions stay consistent, which is what the codec learned.
    assert abs(root_yaw(gR2[0, 0].numpy()) - root_yaw(gR[0, 0].numpy())) < 1e-3
    assert validate_rest(rep.spec).ok
    assert fk_invariance_cm(L, spec, reframe_local(L, rep), rep.spec) < 1e-3


def test_validate_rest_flags_yaw_and_passes_neutral():
    spec = _spec()
    assert validate_rest(spec).ok
    Y = yaw_matrix(np.radians(40))
    off = spec.rest_offsets @ Y.T.astype(np.float32)
    turned = SkeletonSpec("toy_yaw", spec.parents, off, spec.joint_names)
    rep = validate_rest(turned)
    assert not rep.ok and "rest_facing_yaw_deg" in rep.failures
    assert abs(abs(rep.checks["rest_facing_yaw_deg"]["value"]) - 40) < 1


def test_find_neutral_frame_prefers_standing():
    spec = _spec()
    L = _rand_local(10, spec.J, seed=2)
    L[4] = matrix_to_rot6d(torch.eye(3).expand(spec.J, 3, 3))       # frame 4 = perfect standing
    assert find_neutral_frame(L, spec) == 4


def test_gauge_version_present():
    assert GAUGE.version and GAUGE.time.fps == 30 and GAUGE.time.window == 60
