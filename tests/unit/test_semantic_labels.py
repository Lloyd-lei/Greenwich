import numpy as np
import pytest

from alphamotion.embodiment.semantics_map import deterministic_part_labels
from alphamotion.embodiment.urdf_ingest import safe_body_name
from alphamotion.engine.nets.rotations import SkeletonSpec


def test_deterministic_semantics_cover_anonymous_topology():
    parents = np.asarray([-1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 10,
                          0, 12, 13])
    offsets = np.asarray([
        [0, 0, 0], [0, 20, 0], [0, 20, 0], [15, 10, 0], [18, 0, 0],
        [18, 0, 0], [-15, 10, 0], [-18, 0, 0], [-18, 0, 0],
        [8, -18, 0], [0, -22, 0], [0, -20, 0], [-8, -18, 0],
        [0, -22, 0], [0, -20, 0],
    ], dtype=float)
    names = [f"body{i:02d}" for i in range(len(parents))]
    spec = SkeletonSpec("anonymous", parents, offsets, names)
    labels = deterministic_part_labels(spec)
    assert set(labels) == set(names)
    assert set(labels.values()) <= {
        "head", "torso", "left arm", "right arm", "left leg", "right leg"}
    assert {"head", "torso", "left arm", "right arm",
            "left leg", "right leg"} <= set(labels.values())


def test_body_name_cannot_escape_data_directory():
    assert safe_body_name("../../My Robot") == "my_robot"
    with pytest.raises(ValueError):
        safe_body_name("../..")
