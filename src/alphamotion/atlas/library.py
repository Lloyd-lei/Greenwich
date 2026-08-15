"""The curated clip library: 4096 family-balanced windows.

Each entry = the window's RAW dual-stream codes (packed nibbles, mmapped
library_codes.npy) + 32 rainbow tokens + 4 boundary-frame codes. Playback
decodes the raw codes — bit-faithful to the corpus on any embodiment.
Tokens/bounds serve the editor (pins, bridges, atlas edges); they cannot
replace the raw codes because A3 never learned to reconstruct the rotation
stream (measured 0814: argmax on slots 128:256 lands ~24 m off; with the raw
stream the decode is exact).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .families import FAMILIES, family_of


class Library:
    def __init__(self, npz_path: str | Path):
        npz_path = Path(npz_path)
        d = np.load(npz_path)
        self.tokens = d["tokens"]
        self.bounds = d["bounds"]
        meta = json.loads((npz_path.parent / "library_meta.json").read_text())
        self.names = meta["clips"]
        # Old library builds used an unbounded ``hop`` regex and silently
        # classified names such as ``knife_chop`` as jumps. Names are the
        # durable source of truth, so repair labels at load time as well as in
        # future builders.
        self.families = [family_of(name) for name in self.names]
        self.window = meta.get("window", 60)
        codes_npy = npz_path.parent / "library_codes.npy"
        # mmap: 4096 windows x 60f x 256 slots x 10 packed bytes ~ 630 MB
        self._packed = np.load(codes_npy, mmap_mode="r") \
            if codes_npy.exists() else None
        root_npy = npz_path.parent / "library_root.npy"
        self._root = np.load(root_npy, mmap_mode="r") \
            if root_npy.exists() else None
        rm = npz_path.parent / "library_root_meta.json"
        self._root_bodies = json.loads(rm.read_text())["bodies"] \
            if rm.exists() else []

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def has_raw(self) -> bool:
        return self._packed is not None

    def raw_codes(self, i: int) -> np.ndarray:
        """[window, 256, 20] int8 — the window's exact corpus codes."""
        if self._packed is None:
            raise RuntimeError(
                "library_codes.npy missing — this library build predates raw "
                "playback; rebuild with scripts/build_library.py")
        pk = np.asarray(self._packed[int(i)])          # [w,256,10] uint8
        out = np.empty((*pk.shape[:2], 20), np.int8)
        out[..., 0::2] = pk & 0x0F
        out[..., 1::2] = (pk >> 4) & 0x0F
        return out

    def root_delta(self, i: int, body: str,
                   body_reach: float | None = None,
                   human_reach: float | None = None):
        """[window,3] cm Y-up — the window's root trajectory, first frame =
        origin (owner design: data passthrough, not inference). Exact for
        bodies with GMR ground truth; otherwise the human trajectory scaled
        by reach ratio. None if this library predates root storage."""
        if self._root is None:
            return None
        if body in self._root_bodies:
            return np.asarray(self._root[int(i),
                              self._root_bodies.index(body)], np.float64)
        hu = np.asarray(self._root[int(i),
                        self._root_bodies.index("human_smpl")], np.float64)
        s = (body_reach / human_reach) \
            if body_reach and human_reach else 1.0
        return hu * s

    def search(self, q: str = "", family: str = "", offset: int = 0,
               limit: int = 24) -> dict:
        rows = []
        for i in range(len(self.tokens)):
            if family and self.families[i] != family:
                continue
            if q and q.lower() not in self.names[i].lower():
                continue
            rows.append(i)
        # A clip-level label does not guarantee every 60-frame crop contains
        # the named event. For jump browsing, rank windows by measured root
        # elevation so the product shows actual airborne windows first.
        if family == "jump" and self._root is not None:
            rows.sort(key=lambda i: self.motion_metrics(i)["vertical_range_cm"],
                      reverse=True)
        page = rows[offset:offset + limit]
        return {"total": len(rows), "items": [
            {"id": int(i), "name": self.names[i], "family": self.families[i],
             **self.motion_metrics(i)}
            for i in page]}

    def motion_metrics(self, i: int) -> dict:
        """Cheap source-trajectory diagnostics for library selection."""
        if self._root is None or "human_smpl" not in self._root_bodies:
            return {"vertical_range_cm": None, "path_m": None}
        root = np.asarray(
            self._root[int(i), self._root_bodies.index("human_smpl")],
            np.float64)
        vertical = float(np.ptp(root[:, 1]))
        horizontal = root[:, [0, 2]]
        path_m = float(np.linalg.norm(np.diff(horizontal, axis=0),
                                      axis=1).sum() / 100.0)
        return {"vertical_range_cm": round(vertical, 2),
                "path_m": round(path_m, 3)}

    def entry(self, i: int):
        return (self.tokens[int(i)], self.bounds[int(i)],
                self.names[int(i)], self.families[int(i)])

    def resolve_portal(self, clip: str, tokens: np.ndarray) -> int | None:
        """Resolve an Atlas hit to a playable raw-code library row.

        Some historical Atlas builds stored clip labels in a fixed-width
        string column, so otherwise valid labels can be truncated. Generated
        Atlas rows may also use a user-facing title instead of the source clip
        name. Prefer an unambiguous label match, then accept only a bit-exact
        32-token match. We deliberately do not map merely similar tokens to a
        raw clip: doing so would make the portal button promise a destination
        that the index did not actually identify.
        """
        exact = [i for i, name in enumerate(self.names) if name == clip]
        if exact:
            return exact[0]

        prefix = [i for i, name in enumerate(self.names)
                  if name.startswith(clip) or clip.startswith(name)]
        if len(prefix) == 1:
            return prefix[0]

        query = np.asarray(tokens, dtype=self.tokens.dtype)
        if query.shape != (self.tokens.shape[1],):
            return None
        matches = np.flatnonzero(np.all(self.tokens == query[None], axis=1))
        return int(matches[0]) if len(matches) else None


def load_default() -> Library:
    from ..weights import resolve
    return Library(resolve("library") / "library.npz")
