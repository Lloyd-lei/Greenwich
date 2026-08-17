"""API request/response contracts."""
from __future__ import annotations

from typing import Literal

from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)


# The imported SMPL corpus contains long-form DanceDB clips (up to 8,423
# frames at the canonical 30 FPS). These finite bounds cover the real corpus.
MAX_SEGMENT_FRAMES = 10_000
MAX_TIMELINE_FRAMES = 32_000


class APIModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, str_strip_whitespace=True)


class SE3Control(APIModel):
    joint: int = Field(ge=0)
    frame_start: int = Field(ge=0)
    frame_end: int = Field(gt=0)
    delta_m: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    delta_rot_deg: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])

    @field_validator("delta_m", "delta_rot_deg")
    @classmethod
    def _vec3(cls, value, info):
        if len(value) != 3:
            raise ValueError("must contain exactly three values")
        out = [float(x) for x in value]
        limit = 5.0 if info.field_name == "delta_m" else 360.0
        if any(abs(x) > limit for x in out):
            unit = "metres" if info.field_name == "delta_m" else "degrees"
            raise ValueError(f"values must stay within +/-{limit:g} {unit}")
        return out

    @model_validator(mode="after")
    def _ordered_frames(self):
        if self.frame_end <= self.frame_start:
            raise ValueError("frame_end must be greater than frame_start")
        return self


class Segment(APIModel):
    """One timeline block.

    kind=library : a curated clip (library_id), duration n (retiming);
    kind=gap     : Equator-generated bridge between neighbours, budget n;
    kind=prompt  : GENMO text->motion (perception extra), text + n;
    kind=video   : GENMO video->motion, asset path + n.
    """
    kind: Literal["library", "gap", "prompt", "video"]
    library_id: int | None = Field(default=None, ge=0)
    text: str | None = Field(default=None, max_length=500)
    video_asset: str | None = Field(default=None, max_length=4096)
    n: int = Field(default=60, ge=1, le=MAX_SEGMENT_FRAMES)
    # A split Library block keeps a non-destructive reference to the rendered
    # source material. ``source_start`` is inclusive and ``source_end`` is
    # exclusive; ``n`` remains the editable output duration of this block.
    source_frames: int | None = Field(
        default=None, ge=1, le=MAX_SEGMENT_FRAMES)
    source_start: int = Field(default=0, ge=0)
    source_end: int | None = Field(default=None, ge=1)
    pins: dict[int, int] | None = None
    seed: int = Field(default=0, ge=0, le=2**63 - 1)
    temperature: float = Field(default=0.9, gt=0.0, le=3.0)

    @field_validator("pins")
    @classmethod
    def _valid_pins(cls, value):
        if value is None:
            return value
        out = {}
        for slot, token in value.items():
            slot, token = int(slot), int(token)
            if not 0 <= slot < 32:
                raise ValueError("pin slots must be in [0, 31]")
            if not 0 <= token < 15625:
                raise ValueError("pin tokens must be in [0, 15624]")
            out[slot] = token
        return out

    @model_validator(mode="after")
    def _required_source(self):
        if self.kind == "library" and self.library_id is None:
            raise ValueError("library segments require library_id")
        if self.kind == "prompt" and not self.text:
            raise ValueError("prompt segments require text")
        if self.kind == "video" and not self.video_asset:
            raise ValueError("video segments require video_asset")
        has_trim = (self.source_frames is not None or self.source_start != 0
                    or self.source_end is not None)
        if has_trim and self.kind != "library":
            raise ValueError("source ranges are only supported on library segments")
        if has_trim:
            if self.source_frames is None:
                raise ValueError("source_frames is required for a source range")
            end = self.source_end if self.source_end is not None \
                else self.source_frames
            if not self.source_start < end <= self.source_frames:
                raise ValueError(
                    "source range must satisfy 0 <= start < end <= source_frames")
        return self


class TimelineRequest(APIModel):
    segments: list[Segment] = Field(min_length=1, max_length=32)
    target_body: str = Field(default="unitree_h1", min_length=1,
                             max_length=128)
    title: str = Field(default="", max_length=240)
    se3: list[SE3Control] = Field(default_factory=list, max_length=64)
    render: bool = True
    fps: float = Field(default=30.0, ge=1.0, le=120.0)

    @model_validator(mode="after")
    def _bounded_timeline(self):
        frames = sum(segment.n for segment in self.segments)
        if frames > MAX_TIMELINE_FRAMES:
            raise ValueError(
                f"timeline may contain at most {MAX_TIMELINE_FRAMES} frames")
        if any(control.frame_end > frames for control in self.se3):
            raise ValueError("SE3 frame ranges must lie inside the timeline")
        return self


class PlayRequest(APIModel):
    library_id: int = Field(ge=0)
    target_body: str = Field(default="unitree_h1", min_length=1,
                             max_length=128)
    n: int | None = Field(default=None, ge=1, le=MAX_SEGMENT_FRAMES)
    render: bool = True


class JumpRequest(APIModel):
    """Portal jump: bridge from a motion's frame into a library window."""
    motion_id: int = Field(gt=0)
    at_slot: int = Field(ge=0, lt=32)
    dest_library_id: int = Field(ge=0)
    bridge_n: int = Field(default=45, ge=1, le=MAX_SEGMENT_FRAMES)
    target_body: str = Field(default="unitree_h1", min_length=1,
                             max_length=128)
    render: bool = True


class IngestResponse(APIModel):
    job_id: str
