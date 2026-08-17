"""Publish saved BodyDataStudio augmentations into Motion Studio.

BodyDataStudio remains the owner of raw/derived SMPL files and lineage.  This
script builds the small Greenwich index sidecar required for editing and robot
retargeting; every full source motion remains an individual ``source_clips``
record, rather than being flattened into a bundle-length preview.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from import_smpl_library import WINDOW, _motion  # noqa: E402
from alphamotion.config import CONFIG  # noqa: E402
from alphamotion.data_studio import catalog_records  # noqa: E402


def _source_file(record: dict) -> Path | None:
    """Resolve both the new Linux cache and the old read-only Windows index."""
    raw = str(record.get("container") or "")
    direct = Path(raw)
    if direct.is_file():
        return direct
    normalized = raw.replace("\\", "/")
    marker = "/_preview_cache/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        candidates = (
            Path(CONFIG.data_studio_root) / "_preview_cache" / suffix,
            Path(CONFIG.data_studio_cache) / suffix,
        )
        for path in candidates:
            if path.is_file():
                return path
    name = Path(normalized).name
    for root in (Path(CONFIG.data_studio_cache),
                 Path(CONFIG.data_studio_root) / "_preview_cache"):
        if root.is_dir() and name:
            match = next(root.rglob(name), None)
            if match is not None:
                return match
    return None


def build(output_root: Path, shard_name: str = "zz_data_studio_augmented") -> dict:
    from alphamotion.service.pool import POOL

    records = catalog_records()
    by_id = {row["asset_id"]: row for row in records}
    publishable = []
    missing = []
    for record in records:
        if record["role"] != "augmented":
            continue
        source_path = _source_file(record)
        if source_path is None:
            missing.append({"asset_id": record["asset_id"],
                            "container": record["container"]})
            continue
        origin = by_id.get(record["origin_id"], {})
        publishable.append((record, origin, source_path))

    if not publishable:
        return {"published": 0, "missing": missing,
                "message": "No saved augmentations are ready to publish"}

    POOL.warm()
    greenwich, equator = POOL.greenwich, POOL.equator
    spec, dof, _rest = POOL.human
    count = len(publishable)
    packed = np.empty((count, WINDOW, 256, 10), np.uint8)
    roots = np.empty((count, 1, WINDOW, 3), np.float16)
    tokens = np.empty((count, 32), np.int32)
    bounds = np.empty((count, 4, 256, 20), np.int8)
    meta = {key: [] for key in (
        "clips", "datasets", "sources", "original_fps", "source_frames",
        "source_models", "source_genders", "vertical_ranges_cm",
        "path_lengths_m", "data_roles", "asset_ids", "origin_ids",
        "augmentations", "augmentation_values", "labels", "variant_counts")}

    output_root.mkdir(parents=True, exist_ok=True)
    final = output_root / shard_name
    staging = output_root / f".{shard_name}.building"
    if staging.exists():
        shutil.rmtree(staging)
    source_dir = staging / "source_clips"
    source_dir.mkdir(parents=True)

    for index, (record, origin, path) in enumerate(publishable):
        with np.load(path, allow_pickle=False) as data:
            motion = _motion(data)
        full_local = np.asarray(motion["local_rot6d"], np.float32)
        full_root = np.asarray(motion["root_cm"], np.float32)
        if len(full_local) >= WINDOW:
            local6d, root = full_local[:WINDOW], full_root[:WINDOW]
        else:
            pad = WINDOW - len(full_local)
            local6d = np.concatenate(
                (full_local, np.repeat(full_local[-1:], pad, axis=0)))
            root = np.concatenate(
                (full_root, np.repeat(full_root[-1:], pad, axis=0)))
        pose9, _ = greenwich.pose9(
            torch.from_numpy(local6d), spec, is_global=False)
        codes = greenwich.encode(pose9, spec, dof)
        token, _ = equator.tokenize(codes)
        code_np = codes.detach().cpu().numpy().astype(np.uint8)
        packed[index] = code_np[..., 0::2] | (code_np[..., 1::2] << 4)
        roots[index, 0] = root
        tokens[index] = token.detach().cpu().numpy().astype(np.int32)
        bounds[index] = code_np[[0, 1, -2, -1]].astype(np.int8)
        np.savez_compressed(
            source_dir / f"{index:06d}.npz",
            local_rot6d=full_local.astype(np.float16),
            root_cm=full_root.astype(np.float16),
            hand_pose=np.asarray(motion["hand_pose"], np.float16),
            betas=np.asarray(motion["betas"], np.float32),
            gender=np.asarray({"neutral": 0, "male": 1, "female": 2}[
                str(motion["gender"])], np.uint8),
            model_family=np.asarray(
                1 if motion["model_family"] == "smplh" else 0, np.uint8))

        source = str(origin.get("source_key") or record["source_key"])
        horizontal = full_root[:, [0, 2]]
        values = {
            "clips": f"Augmented/{source}__{record['title']}",
            "datasets": "imported_smpl",
            "sources": source,
            "original_fps": float(motion["source_fps"]),
            "source_frames": len(full_local),
            "source_models": str(motion["model_family"]),
            "source_genders": str(motion["gender"]),
            "vertical_ranges_cm": round(float(np.ptp(full_root[:, 1])), 2),
            "path_lengths_m": round(float(np.linalg.norm(
                np.diff(horizontal, axis=0), axis=1).sum() / 100.0), 3),
            "data_roles": "augmented",
            "asset_ids": record["asset_id"],
            "origin_ids": record["origin_id"],
            "augmentations": record["augmentation"],
            "augmentation_values": record["augmentation_value"],
            "labels": record["labels"],
            "variant_counts": len(record["variants"]),
        }
        for key, value in values.items():
            meta[key].append(value)

    np.save(staging / "library_codes.npy", packed)
    np.save(staging / "library_root.npy", roots)
    np.savez_compressed(
        staging / "library.npz", tokens=tokens, bounds=bounds,
        clip=np.arange(count, dtype=np.int32), start=np.zeros(count, np.int32),
        family=np.zeros(count, np.int8))
    meta.update({"window": WINDOW, "catalog_database": str(CONFIG.data_studio_db),
                 "import_contract": "saved Data Studio SMPL-family assets"})
    (staging / "library_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    (staging / "library_root_meta.json").write_text(json.dumps({
        "bodies": ["human_smpl"], "units": "cm", "basis": "Y-up",
        "anchor": "first frame of window = origin"}), encoding="utf-8")
    report = {"published": count, "missing": missing,
              "asset_ids": meta["asset_ids"]}
    (staging / "sync_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    os.replace(staging, final)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(os.environ.get(
        "ALPHAMOTION_IMPORTED_LIBRARY", "data/imported_smpl")))
    parser.add_argument("--shard", default="zz_data_studio_augmented")
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.shard), indent=2))


if __name__ == "__main__":
    main()
