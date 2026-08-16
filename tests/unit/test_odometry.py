from types import SimpleNamespace

import numpy as np

from alphamotion.engine.odometry import foot_bodies, stance_offsets


def test_stance_offsets_pin_selected_support_and_preserve_flight():
    # Left foot slides +2 cm/frame while planted, then both feet are airborne.
    feet = np.zeros((8, 2, 3), np.float64)
    feet[:, 0, 0] = np.arange(8) * 0.02
    feet[:, 1, 0] = np.arange(8) * 0.05
    feet[:, 1, 2] = 0.20
    feet[5:, :, 2] += 0.50

    offsets, report = stance_offsets(feet, return_report=True)
    corrected = feet.copy()
    corrected[..., :2] += offsets[:, None, :]

    np.testing.assert_allclose(np.diff(corrected[:5, 0, 0]), 0.0,
                               atol=1e-9)
    np.testing.assert_allclose(np.diff(offsets[5:], axis=0), 0.0,
                               atol=1e-9)
    assert report["slide_cm_frame_after_p90"] == 0.0
    assert report["slide_cm_frame_before_p50"] > 0.0
    assert report["passed"] is True


def test_foot_bodies_choose_one_distal_contact_per_side(monkeypatch):
    names = ["world", "left_ankle", "left_foot", "left_toe",
             "right_ankle", "right_foot", "right_toe"]
    model = SimpleNamespace(
        nbody=len(names),
        body_pos=np.zeros((len(names), 3), np.float64),
    )
    import mujoco as mj
    monkeypatch.setattr(mj, "mj_id2name",
                        lambda _model, _kind, index: names[index])

    assert foot_bodies(model) == [3, 6]
