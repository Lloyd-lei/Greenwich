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
import os
from bisect import bisect_right
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
        self.datasets = meta.get("datasets", ["current"] * len(self.names))
        self.sources = meta.get("sources", ["Curated 4096"] * len(self.names))
        self.data_roles = meta.get("data_roles", [
            "original" if value == "imported_smpl" else "curated"
            for value in self.datasets])
        self.asset_ids = meta.get("asset_ids", [None] * len(self.names))
        self.origin_ids = meta.get("origin_ids", list(self.asset_ids))
        self.augmentations = meta.get("augmentations", [""] * len(self.names))
        self.augmentation_values = meta.get(
            "augmentation_values", [None] * len(self.names))
        self.labels = meta.get("labels", [[] for _ in self.names])
        self.variant_counts = meta.get("variant_counts", [0] * len(self.names))
        self.source_frames = meta.get(
            "source_frames", [meta.get("window", 60)] * len(self.names))
        self.source_models = meta.get("source_models", [None] * len(self.names))
        self.source_genders = meta.get(
            "source_genders", ["neutral"] * len(self.names))
        self._vertical_ranges = meta.get("vertical_ranges_cm")
        self._path_lengths = meta.get("path_lengths_m")
        if len(self.datasets) != len(self.names) or len(self.sources) != len(self.names):
            raise ValueError("library dataset/source metadata length mismatch")
        if any(len(values) != len(self.names) for values in (
                self.data_roles, self.asset_ids, self.origin_ids,
                self.augmentations, self.augmentation_values, self.labels,
                self.variant_counts)):
            raise ValueError("library data-studio metadata length mismatch")
        if any(len(values) != len(self.names) for values in (
                self.source_frames, self.source_models, self.source_genders)):
            raise ValueError("library source-motion metadata length mismatch")
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
        source_rot_npy = npz_path.parent / "library_source_rot6d.npy"
        self._source_rot = np.load(source_rot_npy, mmap_mode="r") \
            if source_rot_npy.exists() else None
        source_dir = npz_path.parent / "source_clips"
        self._source_dir = source_dir if source_dir.is_dir() else None
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

    def source_local_rot6d(self, i: int) -> np.ndarray | None:
        """Exact continuous SMPL local rotations for imported source clips.

        Release-era curated shards predate this sidecar and return ``None``;
        callers may then fall back to the codec. Imported shards must use this
        data for any UI labelled "Original".
        """
        motion = self.source_motion(i)
        return None if motion is None else motion["local_rot6d"]

    def source_motion(self, i: int) -> dict | None:
        """Full exact source parameters, or ``None`` for legacy windows."""
        i = int(i)
        if self._source_dir is not None:
            path = self._source_dir / f"{i:06d}.npz"
            if not path.is_file():
                raise RuntimeError(f"source sidecar missing: {path}")
            with np.load(path, allow_pickle=False) as data:
                gender = int(np.asarray(data["gender"]).reshape(()))
                family = int(np.asarray(data["model_family"]).reshape(()))
                return {
                    "local_rot6d": np.asarray(data["local_rot6d"], np.float32),
                    "root_cm": np.asarray(data["root_cm"], np.float32),
                    "hand_pose": np.asarray(data["hand_pose"], np.float32),
                    "betas": np.asarray(data["betas"], np.float32),
                    "gender": ("neutral", "male", "female")[gender],
                    "model_family": ("smpl", "smplh")[family],
                    "fps": 30.0,
                }
        if self._source_rot is not None:
            local = np.asarray(self._source_rot[i], np.float32)
            return {"local_rot6d": local,
                    "root_cm": self.root_delta(i, "human_smpl"),
                    "hand_pose": np.zeros((len(local), 90), np.float32),
                    "betas": np.zeros(10, np.float32), "gender": "neutral",
                    "model_family": "smplh", "fps": 30.0}
        return None

    def frames(self, i: int) -> int:
        return int(self.source_frames[int(i)])

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

    def playback_root_delta(self, i: int, body: str,
                            body_reach: float | None = None,
                            human_reach: float | None = None):
        """Full source root for imported clips; legacy window otherwise."""
        motion = self.source_motion(i)
        if motion is None:
            return self.root_delta(i, body, body_reach, human_reach)
        root = np.asarray(motion["root_cm"], np.float64)
        scale = (body_reach / human_reach) \
            if body_reach and human_reach else 1.0
        return root * scale

    def enrich_catalog(self, catalog: dict[tuple[str, str], dict]) -> None:
        """Attach BodyDataStudio lineage without mutating encoded shards."""
        for i, name in enumerate(self.names):
            # Newly synchronized Data Studio shards carry authoritative IDs
            # and lineage in their own metadata. Re-matching their names can
            # confuse an augmented title containing ``__`` with its parent.
            if self.asset_ids[i]:
                continue
            source = str(self.sources[i])
            title = name.split("__", 1)[-1]
            record = catalog.get((source.casefold(), title.casefold()))
            if record is None:
                continue
            self.data_roles[i] = record["role"]
            self.asset_ids[i] = record["asset_id"]
            self.origin_ids[i] = record["origin_id"]
            self.augmentations[i] = record["augmentation"]
            self.augmentation_values[i] = record["augmentation_value"]
            self.labels[i] = record["labels"]
            self.variant_counts[i] = len(record["variants"])

    def search(self, q: str = "", family: str = "", offset: int = 0,
               limit: int = 24, dataset: str = "", data_role: str = "",
               augmentation: str = "", label: str = "",
               source: str = "", allowed_indices: set[int] | None = None,
               allowed_asset_ids: set[str] | None = None) -> dict:
        rows = []
        for i in range(len(self.tokens)):
            if (allowed_indices is not None or allowed_asset_ids is not None) \
                    and i not in (allowed_indices or set()) \
                    and str(self.asset_ids[i] or "") not in (allowed_asset_ids or set()):
                continue
            if family and self.families[i] != family:
                continue
            if dataset and self.datasets[i] != dataset:
                continue
            if data_role and self.data_roles[i] != data_role:
                continue
            if augmentation and self.augmentations[i] != augmentation:
                continue
            if label and label not in self.labels[i]:
                continue
            if source and self.sources[i] != source:
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
             "dataset": self.datasets[i], "source": self.sources[i],
             "data_role": self.data_roles[i],
             "asset_id": self.asset_ids[i], "origin_id": self.origin_ids[i],
             "augmentation": self.augmentations[i],
             "augmentation_value": self.augmentation_values[i],
             "labels": self.labels[i], "variant_count": self.variant_counts[i],
             "frames": self.frames(i),
             **self.motion_metrics(i)}
            for i in page]}

    def dataset_summary(self) -> list[dict]:
        values = []
        for key in dict.fromkeys(self.datasets):
            values.append({"id": key, "label": _dataset_label(key),
                           "count": self.datasets.count(key)})
        return values

    def motion_metrics(self, i: int) -> dict:
        """Cheap source-trajectory diagnostics for library selection."""
        if self._vertical_ranges is not None and self._path_lengths is not None:
            return {"vertical_range_cm": float(self._vertical_ranges[int(i)]),
                    "path_m": float(self._path_lengths[int(i)])}
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
    libraries: list[Library] = []
    imported = os.environ.get("ALPHAMOTION_IMPORTED_LIBRARY", "").strip()
    if imported:
        root = Path(imported)
        paths = ([root / "library.npz"] if (root / "library.npz").is_file()
                 else sorted(root.glob("*/library.npz")))
        libraries.extend(Library(path) for path in paths)
    # The release corpus is a model/index fixture, not user data. Keep it as a
    # fallback for clean-room development, but do not mix it into a connected
    # Data Studio library unless explicitly requested.
    include_curated = os.environ.get("ALPHAMOTION_INCLUDE_CURATED", "0") == "1"
    if include_curated or not libraries:
        libraries.insert(0, Library(resolve("library") / "library.npz"))
    try:
        from ..data_studio import catalog_lookup
        catalog = catalog_lookup()
        for library in libraries:
            library.enrich_catalog(catalog)
    except Exception:
        pass
    return libraries[0] if len(libraries) == 1 else CompositeLibrary(libraries)


def _dataset_label(key: str) -> str:
    return {"current": "Current curated", "imported_smpl": "Data Studio"}.get(
        key, key.replace("_", " ").title())


class CompositeLibrary:
    """One stable ID space over the release library and local SMPL shards."""

    def __init__(self, libraries: list[Library]):
        if not libraries:
            raise ValueError("at least one library is required")
        windows = {library.window for library in libraries}
        if len(windows) != 1:
            raise ValueError("all library shards must use the same window")
        self.libraries = libraries
        self.window = libraries[0].window
        self.offsets = []
        total = 0
        for library in libraries:
            self.offsets.append(total)
            total += len(library)
        self.names = [name for library in libraries for name in library.names]
        self.families = [family for library in libraries
                         for family in library.families]
        self.datasets = [dataset for library in libraries
                         for dataset in library.datasets]
        self.sources = [source for library in libraries
                        for source in library.sources]
        self.data_roles = [value for library in libraries
                           for value in library.data_roles]
        self.asset_ids = [value for library in libraries
                          for value in library.asset_ids]
        self.origin_ids = [value for library in libraries
                           for value in library.origin_ids]
        self.augmentations = [value for library in libraries
                              for value in library.augmentations]
        self.augmentation_values = [value for library in libraries
                                    for value in library.augmentation_values]
        self.labels = [value for library in libraries for value in library.labels]
        self.variant_counts = [value for library in libraries
                               for value in library.variant_counts]
        self.source_frames = [frames for library in libraries
                              for frames in library.source_frames]
        self.tokens = np.concatenate([library.tokens for library in libraries])

    def __len__(self) -> int:
        return len(self.names)

    @property
    def has_raw(self) -> bool:
        return all(library.has_raw for library in self.libraries)

    def _local(self, index: int) -> tuple[Library, int]:
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard = bisect_right(self.offsets, index) - 1
        return self.libraries[shard], index - self.offsets[shard]

    def raw_codes(self, index: int) -> np.ndarray:
        library, local = self._local(index)
        return library.raw_codes(local)

    def source_local_rot6d(self, index: int) -> np.ndarray | None:
        library, local = self._local(index)
        return library.source_local_rot6d(local)

    def source_motion(self, index: int) -> dict | None:
        library, local = self._local(index)
        return library.source_motion(local)

    def frames(self, index: int) -> int:
        library, local = self._local(index)
        return library.frames(local)

    def root_delta(self, index: int, body: str, body_reach=None,
                   human_reach=None):
        library, local = self._local(index)
        return library.root_delta(local, body, body_reach, human_reach)

    def playback_root_delta(self, index: int, body: str, body_reach=None,
                            human_reach=None):
        library, local = self._local(index)
        return library.playback_root_delta(
            local, body, body_reach, human_reach)

    def motion_metrics(self, index: int) -> dict:
        library, local = self._local(index)
        return library.motion_metrics(local)

    def entry(self, index: int):
        library, local = self._local(index)
        return library.entry(local)

    def search(self, q: str = "", family: str = "", offset: int = 0,
               limit: int = 24, dataset: str = "", data_role: str = "",
               augmentation: str = "", label: str = "",
               source: str = "", allowed_indices: set[int] | None = None,
               allowed_asset_ids: set[str] | None = None) -> dict:
        rows = []
        query = q.lower()
        for i, name in enumerate(self.names):
            if (allowed_indices is not None or allowed_asset_ids is not None) \
                    and i not in (allowed_indices or set()) \
                    and str(self.asset_ids[i] or "") not in (allowed_asset_ids or set()):
                continue
            if family and self.families[i] != family:
                continue
            if dataset and self.datasets[i] != dataset:
                continue
            if data_role and self.data_roles[i] != data_role:
                continue
            if augmentation and self.augmentations[i] != augmentation:
                continue
            if label and label not in self.labels[i]:
                continue
            if source and self.sources[i] != source:
                continue
            if query and query not in name.lower():
                continue
            rows.append(i)
        if family == "jump":
            rows.sort(key=lambda i: self.motion_metrics(i)["vertical_range_cm"]
                      or 0.0, reverse=True)
        page = rows[offset:offset + limit]
        return {"total": len(rows), "items": [
            {"id": i, "name": self.names[i], "family": self.families[i],
             "dataset": self.datasets[i], "source": self.sources[i],
             "data_role": self.data_roles[i],
             "asset_id": self.asset_ids[i], "origin_id": self.origin_ids[i],
             "augmentation": self.augmentations[i],
             "augmentation_value": self.augmentation_values[i],
             "labels": self.labels[i], "variant_count": self.variant_counts[i],
             "frames": self.frames(i),
             **self.motion_metrics(i)} for i in page]}

    def dataset_summary(self) -> list[dict]:
        return [{"id": key, "label": _dataset_label(key),
                 "count": self.datasets.count(key)}
                for key in dict.fromkeys(self.datasets)]

    def resolve_portal(self, clip: str, tokens: np.ndarray) -> int | None:
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
