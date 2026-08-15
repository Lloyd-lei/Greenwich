"""API request/response contracts."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SE3Control(BaseModel):
    joint: int
    frame_start: int
    frame_end: int
    delta_m: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    delta_rot_deg: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class Segment(BaseModel):
    """One timeline block.

    kind=library : a curated clip (library_id), duration n (retiming);
    kind=gap     : Equator-generated bridge between neighbours, budget n;
    kind=prompt  : GENMO text->motion (perception extra), text + n;
    kind=video   : GENMO video->motion, asset path + n.
    """
    kind: str
    library_id: int | None = None
    text: str | None = None
    video_asset: str | None = None
    n: int = 60
    pins: dict[int, int] | None = None
    seed: int = 0
    temperature: float = 0.9


class TimelineRequest(BaseModel):
    segments: list[Segment]
    target_body: str = "unitree_h1"
    title: str = ""
    se3: list[SE3Control] = Field(default_factory=list)
    render: bool = True
    fps: float = 30.0


class PlayRequest(BaseModel):
    library_id: int
    target_body: str = "unitree_h1"
    n: int | None = None                  # None = native window length
    render: bool = True


class JumpRequest(BaseModel):
    """Portal jump: bridge from a motion's frame into a library window."""
    motion_id: int
    at_slot: int
    dest_library_id: int
    bridge_n: int = 45
    target_body: str = "unitree_h1"
    render: bool = True


class IngestResponse(BaseModel):
    job_id: str
