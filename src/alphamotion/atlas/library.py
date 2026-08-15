"""The curated generative clip library: 4096 family-balanced windows.

Each entry = 32 rainbow tokens + 4 boundary-frame codes. Nothing else is
stored — playback is regeneration: endpoints_from_codes -> detokenize(any n)
-> Greenwich decode onto any embodiment. 23 MB for 4096 clips.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .families import FAMILIES


class Library:
    def __init__(self, npz_path: str | Path):
        npz_path = Path(npz_path)
        d = np.load(npz_path)
        self.tokens = d["tokens"]
        self.bounds = d["bounds"]
        meta = json.loads((npz_path.parent / "library_meta.json").read_text())
        self.names = meta["clips"]
        self.families = meta["families"]
        self.window = meta.get("window", 60)

    def __len__(self) -> int:
        return len(self.tokens)

    def search(self, q: str = "", family: str = "", offset: int = 0,
               limit: int = 24) -> dict:
        rows = []
        for i in range(len(self.tokens)):
            if family and self.families[i] != family:
                continue
            if q and q.lower() not in self.names[i].lower():
                continue
            rows.append(i)
        page = rows[offset:offset + limit]
        return {"total": len(rows), "items": [
            {"id": int(i), "name": self.names[i], "family": self.families[i]}
            for i in page]}

    def entry(self, i: int):
        return (self.tokens[int(i)], self.bounds[int(i)],
                self.names[int(i)], self.families[int(i)])


def load_default() -> Library:
    from ..weights import resolve
    return Library(resolve("library") / "library.npz")
