"""MotionTrace — the one on-disk motion contract every stage speaks.

npz keys: q [T,J,3] hinge angles, rootR [T,3,3], gp [T,J,3] cm (Y-up,
root-relative source positions used for ground placement), stage [T] int
(0=observed 1=generated 2=refined), fps scalar, title, target.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REQUIRED = ("q", "rootR", "gp", "stage", "fps")


@dataclass
class MotionTrace:
    q: np.ndarray
    rootR: np.ndarray
    gp: np.ndarray
    stage: np.ndarray
    fps: float = 30.0
    title: str = ""
    target: str = ""
    tokens: np.ndarray | None = None      # the 32 rainbow codes, when known
    joint_names: list | None = None       # q columns, for name-based render mapping
    root_t: np.ndarray | None = None      # [T,3] cm world root translation

    def __post_init__(self):
        T = len(self.q)
        if self.q.ndim != 3 or self.q.shape[-1] != 3:
            raise ValueError(f"q must be time-major [T,J,3], got {self.q.shape}")
        if self.rootR.shape != (T, 3, 3):
            raise ValueError(f"rootR must be ({T},3,3), got {self.rootR.shape}")
        if self.gp.shape != (T, self.q.shape[1], 3):
            raise ValueError(f"gp must be time-major [T,J,3], got {self.gp.shape}")
        if len(self.stage) != T:
            raise ValueError("stage must contain one value per frame")
        if not np.isin(self.stage, (0, 1, 2)).all():
            raise ValueError("stage values must be 0, 1 or 2")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        if self.joint_names is not None and len(self.joint_names) != self.q.shape[1]:
            raise ValueError("joint_names must contain one name per q joint")
        if self.root_t is not None and np.asarray(self.root_t).shape != (T, 3):
            raise ValueError(f"root_t must be ({T},3)")
        for name, value in (("q", self.q), ("rootR", self.rootR),
                            ("gp", self.gp), ("root_t", self.root_t)):
            if value is not None and not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinity")

    @property
    def frames(self) -> int:
        return len(self.q)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {}
        if self.tokens is not None:
            extra["tokens"] = np.asarray(self.tokens, np.int32)
        if self.joint_names is not None:
            extra["joint_names"] = np.asarray(self.joint_names)
        if self.root_t is not None:
            extra["root_t"] = np.asarray(self.root_t, np.float32)
        np.savez_compressed(path, q=self.q, rootR=self.rootR, gp=self.gp,
                            stage=self.stage.astype(np.int32),
                            fps=np.float32(self.fps),
                            title=np.asarray(self.title),
                            target=np.asarray(self.target), **extra)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MotionTrace":
        d = np.load(path, allow_pickle=True)
        missing = [k for k in REQUIRED if k not in d]
        if missing:
            raise ValueError(f"trace {path} missing keys {missing}")
        return cls(q=d["q"], rootR=d["rootR"], gp=d["gp"], stage=d["stage"],
                   fps=float(d["fps"]),
                   title=str(d["title"]) if "title" in d else "",
                   target=str(d["target"]) if "target" in d else "",
                   tokens=d["tokens"] if "tokens" in d else None,
                   joint_names=[str(x) for x in d["joint_names"]]
                   if "joint_names" in d else None,
                   root_t=d["root_t"] if "root_t" in d else None)
