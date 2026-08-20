"""Import local AMASS/MOYO SMPL-family archives as playable library shards.

Only motion NPZs with the standard SMPL-family pose prefix are accepted.
HumanRig, Mixamo, model parameter files, shapes and unrelated cache products
are deliberately excluded. Every accepted capture is resampled to 30 fps and
stored in full, including body, hands, trajectory, shape and gender metadata.
The Greenwich/Atlas index remains a separate 60-frame window because that is
the model's training contract; it must never be presented as the whole source.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

SOURCES = {
    "BMLmovi": ("BMLmovi.tar.bz2", "tar"),
    "CMU": ("CMU.tar.bz2", "tar"),
    "DanceDB": ("DanceDB.tar.bz2", "tar"),
    "KIT": ("KIT.tar.bz2", "tar"),
    "MOYO": ("MOYO_smplh_gendered.zip", "zip"),
}
WINDOW = 60
TARGET_FPS = 30.0
SMPL_JOINTS = 22


def _members(path: Path, kind: str) -> Iterator[tuple[str, bytes]]:
    if kind == "tar":
        # Streaming mode is essential for bzip2: random member lookup seeks
        # through the complete compressed archive for every asset.
        with tarfile.open(path, "r|bz2") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".npz"):
                    continue
                if member.name.endswith("/shape.npz"):
                    continue
                handle = archive.extractfile(member)
                if handle is not None:
                    yield member.name, handle.read()
    else:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.endswith(".npz") and "stageii" in name.lower():
                    yield name, archive.read(name)


def _fps(data) -> float:
    for key in ("mocap_framerate", "mocap_frame_rate", "fps"):
        if key in data:
            value = float(np.asarray(data[key]).reshape(()))
            if np.isfinite(value) and value > 0:
                return value
    return TARGET_FPS


def _motion(data) -> dict[str, np.ndarray | float | int | str]:
    # AlphaMotion-native SMPL assets (including GENMO Data Studio results)
    # already use the audited local-rotation/Y-up contract.  Accept them
    # directly so sharing does not round-trip through lossy axis-angle files.
    if "local_rot6d" in data and "root_cm" in data:
        local6d = np.asarray(data["local_rot6d"], np.float32)
        root = np.asarray(data["root_cm"], np.float32)
        if local6d.ndim != 3 or local6d.shape[1:] != (SMPL_JOINTS, 6):
            raise ValueError("local_rot6d must be [T,22,6]")
        if root.shape != (len(local6d), 3) or len(local6d) < 1:
            raise ValueError("root_cm must be [T,3] with at least one frame")
        if not np.isfinite(local6d).all() or not np.isfinite(root).all():
            raise ValueError("motion contains NaN or infinity")
        hands = np.asarray(data["hand_pose"], np.float32) \
            if "hand_pose" in data else np.zeros((len(local6d), 90), np.float32)
        if hands.shape != (len(local6d), 90):
            hands = np.zeros((len(local6d), 90), np.float32)
        betas = np.asarray(data["betas"], np.float32).reshape(-1)[:10] \
            if "betas" in data else np.zeros(10, np.float32)
        if len(betas) < 10:
            betas = np.pad(betas, (0, 10 - len(betas)))
        family = "smplh" if ("model_family" in data and int(
            np.asarray(data["model_family"]).reshape(()))) else "smpl"
        return {"local_rot6d": local6d, "root_cm": root,
                "hand_pose": hands, "betas": betas, "gender": "neutral",
                "model_family": family, "source_fps": _fps(data)}
    poses = np.asarray(data["poses"], np.float64)
    trans = np.asarray(data["trans"], np.float64)
    if poses.ndim != 2 or poses.shape[1] < SMPL_JOINTS * 3:
        raise ValueError(f"poses must be [T,>={SMPL_JOINTS * 3}]")
    if trans.shape != (len(poses), 3) or len(poses) < 1:
        raise ValueError("trans must be [T,3] with at least one frame")
    # Validate every pose component we will persist, including the 90 SMPL-H
    # hand parameters.  Checking only the body prefix can silently admit a
    # corrupt hand stream that later explodes the skinned mesh.
    persisted_pose_width = 156 if poses.shape[1] >= 156 else SMPL_JOINTS * 3
    if not np.isfinite(poses[:, :persisted_pose_width]).all() \
            or not np.isfinite(trans).all():
        raise ValueError("motion contains NaN or infinity")

    source_fps = _fps(data)
    indices = np.rint(np.arange(
        max(2, int(np.floor((len(poses) - 1) * TARGET_FPS / source_fps)) + 1)
    ) * source_fps / TARGET_FPS).astype(np.int64)
    indices = np.clip(indices, 0, len(poses) - 1)
    aa = poses[indices, :SMPL_JOINTS * 3].reshape(-1, SMPL_JOINTS, 3)
    hands = np.zeros((len(indices), 90), np.float32)
    model_family = "smpl"
    if poses.shape[1] >= 156:
        hands[:] = poses[indices, 66:156]
        model_family = "smplh"
    root = trans[indices]

    # AMASS/MOYO are Z-up. The bundled human descriptor and library root
    # convention are Y-up. SMPL body rotations remain local; only the global
    # root orientation and world translation receive the world-basis change.
    basis = Rotation.from_euler("x", -90, degrees=True).as_matrix()
    local = Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(
        len(aa), SMPL_JOINTS, 3, 3)
    local[:, 0] = basis @ local[:, 0]
    local6d = local[..., :, :2].transpose(0, 1, 3, 2).reshape(
        len(aa), SMPL_JOINTS, 6).astype(np.float32)
    root = (root @ basis.T)
    root = ((root - root[:1]) * 100.0).astype(np.float32)
    betas = np.zeros(10, np.float32)
    if "betas" in data:
        value = np.asarray(data["betas"], np.float32).reshape(-1)
        if not np.isfinite(value).all():
            raise ValueError("betas contain NaN or infinity")
        betas[:min(10, len(value))] = value[:10]
    gender = "neutral"
    if "gender" in data:
        gender = str(np.asarray(data["gender"]).reshape(()).item()).lower()
        if gender not in ("male", "female", "neutral"):
            gender = "neutral"
    return {"local_rot6d": local6d, "root_cm": root,
            "hand_pose": hands, "betas": betas, "gender": gender,
            "model_family": model_family, "source_fps": source_fps}


def _window(data) -> tuple[np.ndarray, np.ndarray, float]:
    """The model/index window, derived from (but distinct from) full source."""
    motion = _motion(data)
    local6d = np.asarray(motion["local_rot6d"], np.float32)
    root = np.asarray(motion["root_cm"], np.float32)
    if len(local6d) >= WINDOW:
        local6d, root = local6d[:WINDOW], root[:WINDOW]
    else:
        pad = WINDOW - len(local6d)
        local6d = np.concatenate(
            [local6d, np.repeat(local6d[-1:], pad, axis=0)])
        root = np.concatenate([root, np.repeat(root[-1:], pad, axis=0)])
    return local6d, root.astype(np.float16), float(motion["source_fps"])


def _count(path: Path, kind: str) -> int:
    if kind == "zip":
        with zipfile.ZipFile(path) as archive:
            return sum(name.endswith(".npz") and "stageii" in name.lower()
                       for name in archive.namelist())
    # Counts come from archive headers only. This first streaming pass is
    # cheap relative to decoding and guarantees exact memmap dimensions.
    with tarfile.open(path, "r|bz2") as archive:
        return sum(member.isfile() and member.name.endswith(".npz")
                   and not member.name.endswith("/shape.npz")
                   for member in archive)


def build_source(source: str, archive: Path, kind: str, output: Path,
                 greenwich, equator, human) -> dict:
    expected = _count(archive, kind)
    shard = output / source
    shard.mkdir(parents=True, exist_ok=True)
    packed = np.lib.format.open_memmap(
        shard / "library_codes.npy", mode="w+", dtype=np.uint8,
        shape=(expected, WINDOW, 256, 10))
    roots = np.lib.format.open_memmap(
        shard / "library_root.npy", mode="w+", dtype=np.float16,
        shape=(expected, 1, WINDOW, 3))
    source_dir = shard / "source_clips"
    source_dir.mkdir(exist_ok=True)
    tokens = np.empty((expected, 32), np.int32)
    bounds = np.empty((expected, 4, 256, 20), np.int8)
    names, sources, datasets, original_fps, source_frames = [], [], [], [], []
    source_models, source_genders, rejected = [], [], []
    vertical_ranges, path_lengths = [], []
    spec, dof, _rest = human

    accepted = 0
    for scanned, (member, raw) in enumerate(_members(archive, kind), 1):
        try:
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                motion = _motion(data)
            full_local = np.asarray(motion["local_rot6d"], np.float32)
            full_root = np.asarray(motion["root_cm"], np.float32)
            if len(full_local) >= WINDOW:
                local6d, root = full_local[:WINDOW], full_root[:WINDOW]
            else:
                pad = WINDOW - len(full_local)
                local6d = np.concatenate(
                    [full_local, np.repeat(full_local[-1:], pad, axis=0)])
                root = np.concatenate(
                    [full_root, np.repeat(full_root[-1:], pad, axis=0)])
            fps = float(motion["source_fps"])
            pose9, _ = greenwich.pose9(
                torch.from_numpy(local6d), spec, is_global=False)
            codes = greenwich.encode(pose9, spec, dof)
            token, _endpoints = equator.tokenize(codes)
            code_np = codes.detach().cpu().numpy().astype(np.uint8)
            packed[accepted] = (code_np[..., 0::2]
                                | (code_np[..., 1::2] << 4))
            roots[accepted, 0] = root
            np.savez_compressed(
                source_dir / f"{accepted:06d}.npz",
                local_rot6d=full_local.astype(np.float16),
                root_cm=full_root.astype(np.float16),
                hand_pose=np.asarray(motion["hand_pose"], np.float16),
                betas=np.asarray(motion["betas"], np.float32),
                gender=np.asarray({"neutral": 0, "male": 1, "female": 2}[
                    str(motion["gender"])], np.uint8),
                model_family=np.asarray(
                    1 if motion["model_family"] == "smplh" else 0,
                    np.uint8))
            tokens[accepted] = token.detach().cpu().numpy().astype(np.int32)
            bounds[accepted] = code_np[[0, 1, -2, -1]].astype(np.int8)
            name = Path(member).stem
            names.append(f"Imported/{source}__{name}")
            sources.append(source)
            datasets.append("imported_smpl")
            original_fps.append(fps)
            source_frames.append(len(full_local))
            source_models.append(str(motion["model_family"]))
            source_genders.append(str(motion["gender"]))
            vertical_ranges.append(round(float(np.ptp(full_root[:, 1])), 2))
            horizontal = full_root[:, [0, 2]]
            path_lengths.append(round(float(np.linalg.norm(
                np.diff(horizontal, axis=0), axis=1).sum() / 100.0), 3))
            accepted += 1
        except Exception as exc:  # one bad capture must not abort the corpus
            rejected.append({"member": member, "error": str(exc)[:300]})
        if scanned % 100 == 0:
            print(f"[{source}] scanned={scanned}/{expected} "
                  f"accepted={accepted} rejected={len(rejected)}", flush=True)

    packed.flush()
    roots.flush()
    if accepted != expected:
        # Shrink through ordinary arrays only when validation rejected rows.
        compact_codes = np.asarray(packed[:accepted]).copy()
        compact_roots = np.asarray(roots[:accepted]).copy()
        del packed, roots
        np.save(shard / "library_codes.npy", compact_codes)
        np.save(shard / "library_root.npy", compact_roots)
    np.savez_compressed(
        shard / "library.npz", tokens=tokens[:accepted],
        bounds=bounds[:accepted], clip=np.arange(accepted, dtype=np.int32),
        start=np.zeros(accepted, np.int32),
        family=np.zeros(accepted, np.int8))
    (shard / "library_meta.json").write_text(json.dumps({
        "clips": names, "window": WINDOW, "datasets": datasets,
        "sources": sources, "original_fps": original_fps,
        "source_frames": source_frames, "source_models": source_models,
        "source_genders": source_genders,
        "vertical_ranges_cm": vertical_ranges, "path_lengths_m": path_lengths,
        "import_contract": ("full SMPL-family source at 30 fps; body + hands "
                            "+ trajectory + betas + gender; separate 60-frame "
                            "Greenwich index window"),
    }, ensure_ascii=False))
    (shard / "library_root_meta.json").write_text(json.dumps({
        "bodies": ["human_smpl"], "units": "cm", "basis": "Y-up",
        "anchor": "first frame of window = origin",
    }))
    (shard / "import_report.json").write_text(json.dumps({
        "source": source, "archive": str(archive), "scanned": expected,
        "accepted": accepted, "rejected": rejected,
    }, indent=2, ensure_ascii=False))
    print(f"[{source}] complete: {accepted}/{expected} playable", flush=True)
    return {"source": source, "accepted": accepted,
            "rejected": len(rejected)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", nargs="*", choices=sorted(SOURCES),
                        default=list(SOURCES))
    args = parser.parse_args()

    from alphamotion.service.pool import POOL
    warm = POOL.warm()
    print(f"models ready on {warm['device']}", flush=True)
    reports = []
    for source in args.sources:
        filename, kind = SOURCES[source]
        archive = args.input / filename
        if not archive.is_file():
            raise FileNotFoundError(archive)
        reports.append(build_source(
            source, archive, kind, args.output, POOL.greenwich,
            POOL.equator, POOL.human))
    # Rebuild the global summary from every complete shard so a targeted
    # --sources rerun does not hide sources imported by an earlier pass.
    reports = []
    for source in SOURCES:
        report_path = args.output / source / "import_report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text())
            reports.append({"source": source, "accepted": report["accepted"],
                            "rejected": len(report["rejected"])})
    (args.output / "import_summary.json").write_text(json.dumps({
        "dataset": "imported_smpl", "sources": reports,
        "accepted": sum(row["accepted"] for row in reports),
        "rejected": sum(row["rejected"] for row in reports),
        "excluded": ["HumanRig (not SMPL)", "Mixamo (not SMPL)",
                     "SMPL-H model parameter files (not motions)"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
