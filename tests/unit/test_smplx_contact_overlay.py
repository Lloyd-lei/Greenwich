import numpy as np
import pytest

from alphamotion.viz.live import align_contact_frames
from alphamotion.viz.smplx_skin import sole_positions_from_skin


def test_sole_positions_are_derived_from_visible_skin_vertices():
    vertices = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3)
    patches = (
        np.array([0, 1]), np.array([2, 3]),
        np.array([4, 5]), np.array([6, 7]),
    )

    positions = sole_positions_from_skin(vertices, patches)

    assert positions.shape == (2, 4, 3)
    np.testing.assert_allclose(positions[:, 0], vertices[:, [0, 1]].mean(axis=1))
    np.testing.assert_allclose(positions[:, 3], vertices[:, [6, 7]].mean(axis=1))


def test_sole_positions_reject_invalid_patch_vertices():
    vertices = np.zeros((2, 8, 3), np.float32)
    patches = (np.array([0]), np.array([1]), np.array([2]), np.array([9]))

    with pytest.raises(ValueError, match="invalid vertices"):
        sole_positions_from_skin(vertices, patches)


def test_contact_states_follow_preview_fps_instead_of_raw_frame_count():
    contact = np.zeros((425, 4), np.uint8)
    contact[120, 0] = 1
    contact_type = np.zeros_like(contact)
    contact_type[120, 0] = 2

    states, types = align_contact_frames(
        contact, contact_type, 107, label_fps=120.0, target_fps=30.0)

    assert states.shape == (107, 4)
    assert states[30, 0] == 1
    assert types[30, 0] == 2
