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

    def __post_init__(self):
        T = len(self.q)
        assert self.rootR.shape == (T, 3, 3), self.rootR.shape
        assert self.gp.shape[0] == T and self.gp.shape[-1] == 3
        assert len(self.stage) == T

    @property
    def frames(self) -> int:
        return len(self.q)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {}
        if self.tokens is not None:
            extra["tokens"] = np.asarray(self.tokens, np.int32)
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
                   tokens=d["tokens"] if "tokens" in d else None)
