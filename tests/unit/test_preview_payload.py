import numpy as np
import pytest

from alphamotion.viz.preview import skeleton_preview_payload


def test_preview_payload_downsamples_normalizes_and_builds_edges():
    points = np.zeros((60, 3, 3), np.float64)
    points[:, 1, 1] = 1.0
    points[:, 2, 0] = np.linspace(0.0, 2.0, 60)
    payload = skeleton_preview_payload(points, [-1, 0, 1], max_frames=30)

    assert payload["frames"] == 30
    assert payload["source_frames"] == 60
    assert payload["edges"] == [[0, 1], [1, 2]]
    xy = np.asarray(payload["points"])
    assert xy.shape == (30, 3, 2)
    assert np.max(np.ptp(xy, axis=(0, 1))) == pytest.approx(1.0, abs=1e-4)


def test_preview_payload_rejects_invalid_shape():
    with pytest.raises(ValueError, match="T,J,3"):
        skeleton_preview_payload(np.zeros((3, 2)), [-1, 0])
