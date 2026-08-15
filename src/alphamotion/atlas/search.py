"""Atlas Map — the navigable index over motion-token space.

The atlas is a FIXED-CAPACITY table of token sequences ("rainbow codes"): K
windows of real or generated motion, each reduced to Equator's 32 discrete
tokens. Because every motion in the product lives in the same token space, any
code position of any motion can be looked up here and jump ("portal") into
every other motion that passes through the same token — motion space becomes a
graph you can walk, not a black box you can only sample.

Data layout (atlas.npz):
    tokens  int32 [K, 32]     the rainbow codes
    clip    int32 [K]         index into meta["clips"]
    start   int32 [K]         first corpus frame of the window
    family  int16 [K]         index into families.FAMILIES
plus atlas_meta.json {"clips": [...], "window": 60, "stride": 30}.

Runtime additions (user generations) go through add() into a ring: the table
is a fixed-size KV-cache — when full, the oldest dynamic entry is evicted;
the built corpus prefix is never evicted.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .families import FAMILIES, family_id


class AtlasIndex:
    def __init__(self, tokens: np.ndarray, clip: np.ndarray, start: np.ndarray,
                 family: np.ndarray, clips: list[str], window: int = 60,
                 stride: int = 30, capacity: int | None = None):
        self.tokens = tokens.astype(np.int32)
        self.clip = clip.astype(np.int32)
        self.start = start.astype(np.int32)
        self.family = family.astype(np.int16)
        self.clips = list(clips)
        # Recompute frozen labels from clip names. This also repairs atlases
        # built before the word-boundary fix for ``hop``/``chop``.
        self.family = np.asarray(
            [family_id(self.clips[int(c)]) for c in self.clip],
            dtype=np.int16)
        self.window, self.stride = window, stride
        self.frozen = len(tokens)              # corpus prefix, never evicted
        self.capacity = capacity or max(2 * self.frozen, self.frozen + 4096)
        self._ring: list[int] = []             # dynamic row ids in insert order
        self._build_inverted()

    # ------------------------------------------------------------ indexing --
    def _build_inverted(self) -> None:
        self.post: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for w in range(len(self.tokens)):
            for s in range(32):
                self.post[int(self.tokens[w, s])].append((w, s))

    def add(self, tokens: np.ndarray, name: str, family_idx: int) -> int:
        """Register a generated motion (fixed-capacity ring on dynamic rows)."""
        if len(self.tokens) >= self.capacity and self._ring:
            evict = self._ring.pop(0)
            for s in range(32):
                lst = self.post[int(self.tokens[evict, s])]
                self.post[int(self.tokens[evict, s])] = \
                    [(w, sl) for (w, sl) in lst if w != evict]
            self.tokens[evict] = np.asarray(tokens, np.int32)
            self.clip[evict] = len(self.clips)
            self.start[evict] = 0
            self.family[evict] = family_idx
            self.clips.append(name)
            w = evict
        else:
            w = len(self.tokens)
            self.tokens = np.vstack([self.tokens, np.asarray(tokens, np.int32)])
            self.clip = np.append(self.clip, len(self.clips))
            self.start = np.append(self.start, 0)
            self.family = np.append(self.family, family_idx)
            self.clips.append(name)
        self._ring.append(w)
        for s in range(32):
            self.post[int(self.tokens[w, s])].append((w, s))
        # Ring replacement keeps the same table length, so the lazy histogram
        # cache cannot detect the mutation by shape alone.
        if hasattr(self, "_H"):
            del self._H
        return w

    # ------------------------------------------------------------- queries --
    def portals(self, tokens: np.ndarray, slot: int, k: int = 8,
                exclude_clip: int | None = None):
        """From token `tokens[slot]`: ranked windows that pass through the same
        token, re-ranked by n-gram context overlap around the slot."""
        t = int(tokens[slot])
        hits = self.post.get(t, [])
        scored = []
        lo, hi = max(0, slot - 2), min(32, slot + 3)
        ctx = np.asarray(tokens[lo:hi])
        qh = _hist(np.asarray(tokens)[None])[0]
        for (w, s) in hits:
            if exclude_clip is not None and int(self.clip[w]) == exclude_clip:
                continue
            l2, h2 = max(0, s - 2), min(32, s + 3)
            ctx2 = self.tokens[w, l2:h2]
            m = min(len(ctx), len(ctx2))
            overlap = float((ctx[:m] == ctx2[:m]).mean()) if m else 0.0
            # context n-gram alone leaves big score ties (a 5-token window has
            # 6 possible values); whole-window histogram cosine breaks them by
            # global similarity, which is what lifted same-family precision@8
            # from 4.3x random to the gate level
            hist = float(self._hists()[w] @ qh)
            scored.append((0.6 * overlap + 0.4 * hist, w, s))
        scored.sort(key=lambda x: -x[0])
        return [{"window": int(w), "slot": int(s), "score": round(sc, 3),
                 "clip": self.clips[int(self.clip[w])],
                 "family": FAMILIES[int(self.family[w])],
                 "frame": int(self.start[w]) + int(s) * self.window // 32}
                for sc, w, s in scored[:k]]

    def knn(self, tokens: np.ndarray, k: int = 8):
        """Whole-motion neighbours by token-histogram cosine."""
        q = _hist(np.asarray(tokens)[None])[0]
        H = self._hists()
        sim = H @ q
        order = np.argsort(-sim)[:k]
        return [{"window": int(w), "score": round(float(sim[w]), 3),
                 "clip": self.clips[int(self.clip[w])],
                 "family": FAMILIES[int(self.family[w])]}
                for w in order]

    def walk(self, seed_window: int, steps: int = 6, seed: int = 0):
        """Graph walk: repeatedly portal-jump from a random slot — the
        'motion wikipedia' browse primitive."""
        rng = np.random.default_rng(seed)
        path = [int(seed_window)]
        cur = int(seed_window)
        for _ in range(steps):
            slot = int(rng.integers(0, 32))
            ps = self.portals(self.tokens[cur], slot, k=4,
                              exclude_clip=int(self.clip[cur]))
            if not ps:
                break
            cur = ps[int(rng.integers(0, len(ps)))]["window"]
            path.append(cur)
        return path

    def _hists(self) -> np.ndarray:
        if not hasattr(self, "_H") or len(self._H) != len(self.tokens):
            self._H = _hist(self.tokens)
        return self._H

    # ---------------------------------------------------------------- io ----
    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.savez_compressed(path, tokens=self.tokens, clip=self.clip,
                            start=self.start, family=self.family)
        meta = {"clips": self.clips, "window": self.window,
                "stride": self.stride, "frozen": self.frozen,
                "capacity": self.capacity}
        (path.parent / "atlas_meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, npz_path: str | Path) -> "AtlasIndex":
        npz_path = Path(npz_path)
        d = np.load(npz_path)
        meta = json.loads((npz_path.parent / "atlas_meta.json").read_text())
        idx = cls(d["tokens"], d["clip"], d["start"], d["family"],
                  meta["clips"], meta.get("window", 60),
                  meta.get("stride", 30), meta.get("capacity"))
        idx.frozen = meta.get("frozen", len(d["tokens"]))
        idx._ring = list(range(idx.frozen, len(idx.tokens)))
        return idx


def _hist(tokens: np.ndarray, dim: int = 1024) -> np.ndarray:
    """Hashed token histogram, L2-normalised — cheap whole-motion signature."""
    H = np.zeros((len(tokens), dim), np.float32)
    idx = tokens % dim
    for i in range(len(tokens)):
        np.add.at(H[i], idx[i], 1.0)
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-9
    return H


def load_default() -> AtlasIndex:
    from ..weights import resolve
    return AtlasIndex.load(resolve("atlas") / "atlas.npz")
