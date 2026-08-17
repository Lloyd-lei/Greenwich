import pytest
from pydantic import ValidationError

from alphamotion.service.schemas import (MAX_TIMELINE_FRAMES, PlayRequest,
                                         SE3Control, Segment, TimelineRequest)


def test_segment_contracts():
    with pytest.raises(ValidationError):
        Segment(kind="library")
    with pytest.raises(ValidationError):
        Segment(kind="gap", pins={32: 1})
    assert Segment(kind="gap", n=3, pins={4: 12}).pins == {4: 12}
    cut = Segment(kind="library", library_id=2, n=20,
                  source_frames=60, source_start=10, source_end=30)
    assert (cut.source_start, cut.source_end) == (10, 30)
    with pytest.raises(ValidationError):
        Segment(kind="library", library_id=2, n=20,
                source_frames=60, source_start=30, source_end=61)
    with pytest.raises(ValidationError):
        Segment(kind="gap", n=20, source_frames=60,
                source_start=10, source_end=30)


def test_segment_world_transform_contracts():
    transformed = Segment(
        kind="library", library_id=2, n=20,
        world_position_m=(1.0, -2.0, 0.5),
        world_rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        world_end_position_m=(3.0, 4.0, 0.5),
        world_end_rotation_wxyz=(0.0, 0.0, 0.0, 1.0))
    assert transformed.world_position_m == (1.0, -2.0, 0.5)
    assert transformed.world_rotation_wxyz == (1.0, 0.0, 0.0, 0.0)
    assert transformed.world_end_position_m == (3.0, 4.0, 0.5)
    assert transformed.world_end_rotation_wxyz == (0.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValidationError):
        Segment(kind="library", library_id=2, n=20,
                world_position_m=(1001.0, 0.0, 0.0))
    with pytest.raises(ValidationError):
        Segment(kind="library", library_id=2, n=20,
                world_rotation_wxyz=(2.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValidationError):
        Segment(kind="library", library_id=2, n=20,
                world_end_position_m=(0.0, -1001.0, 0.0))
    with pytest.raises(ValidationError):
        Segment(kind="library", library_id=2, n=20,
                world_end_rotation_wxyz=(0.0, 0.0, 0.0, 0.0))


def test_se3_contracts():
    with pytest.raises(ValidationError):
        SE3Control(joint=0, frame_start=4, frame_end=4)
    with pytest.raises(ValidationError):
        SE3Control(joint=0, frame_start=0, frame_end=4, delta_m=[1, 2])


def test_request_lengths_are_bounded():
    with pytest.raises(ValidationError):
        PlayRequest(library_id=0, n=0)
    assert PlayRequest(library_id=0, n=8423).n == 8423
    assert Segment(kind="library", library_id=0, n=8423).n == 8423
    with pytest.raises(ValidationError):
        TimelineRequest(segments=[])
    with pytest.raises(ValidationError):
        TimelineRequest(segments=[
            Segment(kind="library", library_id=i,
                    n=MAX_TIMELINE_FRAMES // 4 + 1)
            for i in range(4)])
    with pytest.raises(ValidationError):
        TimelineRequest(
            segments=[Segment(kind="library", library_id=0, n=10)],
            se3=[SE3Control(joint=0, frame_start=8, frame_end=12)])


def test_non_finite_and_unbounded_controls_are_rejected():
    with pytest.raises(ValidationError):
        Segment(kind="gap", temperature=float("nan"))
    with pytest.raises(ValidationError):
        SE3Control(joint=0, frame_start=0, frame_end=2,
                   delta_m=[6.0, 0.0, 0.0])
