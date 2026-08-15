"""GENMO perception adapters (optional extra): video->SMPL and text->SMPL.

The heavy models (GENMO checkpoint ~5.5 GB, T5) are NOT bundled or re-hosted;
they run in a separate python environment pointed to by
ALPHAMOTION_GENMO_PYTHON + ALPHAMOTION_GENMO_REPO, communicating through the
args + last-stdout-JSON worker protocol. Without that env configured these
functions raise with actionable instructions instead of pretending.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from ..config import CONFIG
from ..paths import cache_dir

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                16, 17, 18, 19]


def _require_env():
    if not CONFIG.genmo_python or not CONFIG.genmo_repo:
        raise RuntimeError(
            "GENMO perception is not configured. Install GENMO in its own "
            "environment, then set ALPHAMOTION_GENMO_PYTHON=<env>/bin/python "
            "and ALPHAMOTION_GENMO_REPO=<genmo checkout>. Weights download "
            "from the official source on first run.")
    if not Path(CONFIG.genmo_python).exists():
        raise RuntimeError(f"genmo python not found: {CONFIG.genmo_python}")


def _aa_to_R(aa: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = np.divide(aa, np.where(th < 1e-8, 1.0, th))
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    eye = np.broadcast_to(np.eye(3), K.shape).copy()
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return eye + s * K + (1 - c) * (K @ K)


def smpl_to_global_rot6d(smpl_params: dict, segment: str | None = None):
    """GENMO smpl_params dict -> GLOBAL rot6d [T,22,6] (torch, cpu).
    Local->global composition included — feeding local axis-angle straight to
    FK is the audited 11.6 cm silent bug."""
    p = smpl_params["body_params_global"]
    sl = slice(None)
    if segment:
        segs = [x for x in smpl_params.get("segment_info", [])
                if x["type"] == segment]
        if not segs:
            raise ValueError(f"no '{segment}' segment in artifact")
        sl = slice(segs[0]["start"], segs[0]["end"])
    go = p["global_orient"].float().numpy().reshape(-1, 1, 3)[sl]
    bp = p["body_pose"].float().numpy().reshape(-1, 21, 3)[sl]
    aa = np.concatenate([go, bp], 1)
    Rl = _aa_to_R(aa)
    Rg = np.zeros_like(Rl)
    for j, par in enumerate(SMPL_PARENTS):
        Rg[:, j] = Rl[:, j] if par < 0 else Rg[:, par] @ Rl[:, j]
    r6 = Rg[..., :, :2].transpose(0, 1, 3, 2).reshape(len(aa), 22, 6)
    return torch.from_numpy(r6).float()


def _run_genmo(inputs: list[str], text_len: int) -> dict:
    _require_env()
    staging = cache_dir() / "genmo_staging"
    staging.mkdir(parents=True, exist_ok=True)
    cmd = [CONFIG.genmo_python, "scripts/demo/demo_smpl.py",
           "--input_list", *inputs, "--no_render", "--static_cam",
           "--text_length", str(text_len), "--output_root", str(staging)]
    proc = subprocess.run(cmd, cwd=CONFIG.genmo_repo, text=True,
                          capture_output=True, timeout=1900)
    if proc.returncode != 0:
        raise RuntimeError("GENMO failed: "
                           + (proc.stderr[-1500:] or proc.stdout[-1500:]))
    stem = Path(inputs[0]).stem
    art = staging / f"{stem}_mix" / "smpl_params.pt"
    if not art.is_file():
        raise RuntimeError(f"GENMO artifact missing: {art}")
    return torch.load(art, map_location="cpu", weights_only=False)


def reference_video() -> Path:
    """The cached reference clip whose preprocessing is prewarmed — GENMO needs
    one video for camera intrinsics even in text mode."""
    ref = cache_dir() / "genmo_reference.mp4"
    if not ref.exists():
        raise RuntimeError(
            "no reference video cached; copy any short person video to "
            f"{ref} once (its perception cache warms on first run)")
    return ref


def motion_from_prompt(text: str, seconds: float = 5.0) -> torch.Tensor:
    frames = max(30, int(seconds * 30))
    art = _run_genmo([str(reference_video()), f"text:{text}"], frames)
    return smpl_to_global_rot6d(art, segment="text")


def motion_from_video(video_path: str) -> torch.Tensor:
    src = Path(video_path)
    if not src.is_file():
        raise RuntimeError(f"video not found: {src}")
    staged = cache_dir() / "genmo_staging" / "uploads" / src.name
    staged.parent.mkdir(parents=True, exist_ok=True)
    if not staged.exists():
        shutil.copy2(src, staged)
    art = _run_genmo([str(staged)], 60)
    return smpl_to_global_rot6d(art)
