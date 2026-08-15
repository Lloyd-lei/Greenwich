import pytest
from pydantic import ValidationError

from alphamotion.service.schemas import (PlayRequest, SE3Control, Segment,
                                         TimelineRequest)


def test_segment_contracts():
    with pytest.raises(ValidationError):
        Segment(kind="library")
    with pytest.raises(ValidationError):
        Segment(kind="gap", pins={32: 1})
    assert Segment(kind="gap", n=3, pins={4: 12}).pins == {4: 12}


def test_se3_contracts():
    with pytest.raises(ValidationError):
        SE3Control(joint=0, frame_start=4, frame_end=4)
    with pytest.raises(ValidationError):
        SE3Control(joint=0, frame_start=0, frame_end=4, delta_m=[1, 2])


def test_request_lengths_are_bounded():
    with pytest.raises(ValidationError):
        PlayRequest(library_id=0, n=0)
    with pytest.raises(ValidationError):
        TimelineRequest(segments=[])
    with pytest.raises(ValidationError):
        TimelineRequest(segments=[
            Segment(kind="library", library_id=i, n=1201)
            for i in range(3)])
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
