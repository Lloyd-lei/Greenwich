import numpy as np
import torch

from alphamotion.engine.timeline import (bridge_root, interpolate_lattice,
                                         repair_generated_holds,
                                         repair_generated_joint_jumps,
                                         repair_generated_joint_holds,
                                         repair_projection_branch_flips,
                                         resample_continuous)


def test_continuous_retime_preserves_endpoints_without_duplicate_steps():
    x = np.stack([np.linspace(0, 1, 60), np.linspace(2, -1, 60)], axis=1)
    y = resample_continuous(x, 90)
    np.testing.assert_allclose(y[0], x[0])
    np.testing.assert_allclose(y[-1], x[-1])
    assert not np.any(np.linalg.norm(np.diff(y, axis=0), axis=1) == 0)


def test_lattice_retime_is_continuous_and_bounded():
    codes = torch.arange(5)[:, None, None].expand(-1, 2, 3)
    out = interpolate_lattice(codes, 9)
    assert out.shape == (9, 2, 3)
    assert int(out.min()) == 0 and int(out.max()) == 4
    assert torch.equal(out[0], codes[0]) and torch.equal(out[-1], codes[-1])


def test_bridge_root_moves_and_matches_boundary_velocity():
    prev = np.zeros((10, 3)); prev[:, 0] = np.arange(10) * 2.0
    nxt = np.zeros((10, 3)); nxt[:, 0] = np.arange(10) * 2.0
    interior, anchor = bridge_root(prev, nxt, 8)
    assert interior.shape == (8, 3)
    assert np.all(np.diff(interior[:, 0]) > 0)
    np.testing.assert_allclose(interior[0, 0] - prev[-1, 0], 2.0,
                               atol=0.35)
    np.testing.assert_allclose(anchor[0] - interior[-1, 0], 2.0,
                               atol=0.35)
    assert anchor[1] == prev[-1, 1]


def test_generated_hold_repair_changes_only_generated_middle_frame():
    # identity, duplicate identity, then a 90-degree Z rotation
    from alphamotion.engine.nets.rotations import matrix_to_rot6d
    R = torch.eye(3).repeat(3, 1, 1)
    R[2] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    r6 = matrix_to_rot6d(R[:, None])
    out, report = repair_generated_holds(r6, np.array([0, 1, 0]))
    assert report["repaired_frames"] == 1
    assert torch.allclose(out[0], r6[0])
    assert torch.allclose(out[2], r6[2])
    assert not torch.allclose(out[1], r6[1])


def test_projection_branch_island_is_repaired_but_sustained_turn_is_not():
    q = torch.zeros(8, 1, 3, dtype=torch.float64)
    q[:, 0, 0] = torch.tensor([1.5, 1.6, -1.1, -1.1, 1.55, 1.5, .1, -1.4])
    out, report = repair_projection_branch_flips(q)
    assert report["repaired_values"] == 2
    assert abs(float(out[2, 0, 0]) - 1.58) < 0.1
    assert abs(float(out[3, 0, 0]) - 1.57) < 0.1
    # The final one-way transition has no matching return and is preserved.
    assert out[-1, 0, 0] == q[-1, 0, 0]


def test_joint_hold_repair_survives_projection_collapse():
    q = torch.zeros(4, 1, 3, dtype=torch.float64)
    q[-1, 0, 0] = 1.0
    R = torch.eye(3, dtype=torch.float64).repeat(4, 1, 1, 1)
    a = torch.tensor(1.0, dtype=torch.float64)
    R[-1, 0] = torch.stack([
        torch.stack([torch.cos(a), -torch.sin(a), torch.tensor(0.)]),
        torch.stack([torch.sin(a), torch.cos(a), torch.tensor(0.)]),
        torch.tensor([0., 0., 1.]),
    ])
    out, report = repair_generated_joint_holds(
        q, R, np.array([0, 1, 1, 0]))
    assert report["repaired_frames"] == 2
    assert 0.2 < float(out[1, 0, 0]) < float(out[2, 0, 0]) < 0.8
    assert torch.equal(out[0], q[0]) and torch.equal(out[-1], q[-1])


def test_joint_hold_repair_fixes_generated_to_observed_boundary():
    q = torch.zeros(3, 1, 3, dtype=torch.float64)
    q[0, 0, 0] = -1.0
    R = torch.eye(3, dtype=torch.float64).repeat(3, 1, 1, 1)
    a = torch.tensor(-1.0, dtype=torch.float64)
    R[0, 0] = torch.stack([
        torch.stack([torch.cos(a), -torch.sin(a), torch.tensor(0.)]),
        torch.stack([torch.sin(a), torch.cos(a), torch.tensor(0.)]),
        torch.tensor([0., 0., 1.]),
    ])
    out, report = repair_generated_joint_holds(
        q, R, np.array([1, 1, 0]))
    assert report["repaired_frames"] == 1
    assert -0.8 < float(out[1, 0, 0]) < -0.2
    assert torch.equal(out[0], q[0]) and torch.equal(out[2], q[2])


def test_generated_joint_jump_repair_preserves_observed_anchors():
    q = torch.zeros(4, 1, 3, dtype=torch.float64)
    q[:, 0, 2] = torch.tensor([0.0, 0.5, 2.6, -1.25])
    a = q[:, 0, 2]
    R = torch.zeros(4, 1, 3, 3, dtype=torch.float64)
    R[:, 0, 0, 0] = torch.cos(a)
    R[:, 0, 0, 1] = -torch.sin(a)
    R[:, 0, 1, 0] = torch.sin(a)
    R[:, 0, 1, 1] = torch.cos(a)
    R[:, 0, 2, 2] = 1.0

    out, report = repair_generated_joint_jumps(
        q, R, np.array([0, 1, 1, 0]), radius=8)
    assert report["large_jumps_before"] == 2
    assert report["repaired_frames"] == 2
    assert torch.equal(out[0], q[0]) and torch.equal(out[-1], q[-1])
    assert float(torch.rad2deg(
        (out[1:, 0, 2] - out[:-1, 0, 2]).abs()).max()) < 90.0


def test_generated_joint_jump_repair_never_changes_constrained_frames():
    q = torch.zeros(4, 1, 3, dtype=torch.float64)
    q[:, 0, 2] = torch.tensor([0.0, 0.5, 2.6, -1.25])
    a = q[:, 0, 2]
    R = torch.zeros(4, 1, 3, 3, dtype=torch.float64)
    R[:, 0, 0, 0] = torch.cos(a)
    R[:, 0, 0, 1] = -torch.sin(a)
    R[:, 0, 1, 0] = torch.sin(a)
    R[:, 0, 1, 1] = torch.cos(a)
    R[:, 0, 2, 2] = 1.0

    out, report = repair_generated_joint_jumps(
        q, R, np.array([0, 2, 2, 0]))
    assert report["repaired_frames"] == 0
    assert torch.equal(out, q)
