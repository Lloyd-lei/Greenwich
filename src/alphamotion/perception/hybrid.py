"""Low-memory perception backend: MoMask text motion + GVHMR video HMR.

Both external models terminate before control returns, so their CUDA weights
never overlap.  The public API intentionally matches the old GENMO adapter:
``motion_from_prompt`` and ``motion_from_video`` return global SMPL-22 rot6d
plus first-frame-anchored Y-up root translation in centimetres.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from ..config import CONFIG
from ..engine.nets.rotations import matrix_to_rot6d, rot6d_to_matrix
from ..paths import cache_dir
from .genmo import SMPL_PARENTS, smpl_root_translation, smpl_to_global_rot6d


def _worker() -> Path:
    return Path(__file__).resolve().parent / "workers" / "momask_worker.py"


def status() -> dict:
    python = Path(CONFIG.motion_python) if CONFIG.motion_python else None
    momask = Path(CONFIG.momask_repo) if CONFIG.momask_repo else None
    gvhmr = Path(CONFIG.gvhmr_repo) if CONFIG.gvhmr_repo else None
    text_ready = bool(
        python and python.is_file() and momask and
        (momask / "gen_t2m.py").is_file() and
        (momask / "checkpoints" / "t2m").is_dir() and _worker().is_file())
    video_ready = bool(
        python and python.is_file() and gvhmr and
        (gvhmr / "tools" / "demo" / "demo.py").is_file() and
        (gvhmr / "inputs" / "checkpoints" / "gvhmr" /
         "gvhmr_siga24_release.ckpt").is_file() and
        (gvhmr / "inputs" / "checkpoints" / "hmr2" /
         "epoch=10-step=25000.ckpt").is_file() and
        (gvhmr / "inputs" / "checkpoints" / "vitpose" /
         "vitpose-h-multi-coco.pth").is_file() and
        (gvhmr / "inputs" / "checkpoints" / "yolo" /
         "yolov8x.pt").is_file())
    missing = []
    if not python or not python.is_file():
        missing.append("motion_python")
    if not text_ready:
        missing.append("momask")
    if not video_ready:
        missing.append("gvhmr")
    return {"ready": text_ready or video_ready, "text": text_ready,
            "video": video_ready, "backend": "MoMask + GVHMR",
            "missing": missing}


def _run(cmd: list[str], cwd: Path, timeout: int) -> dict:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True,
                          capture_output=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError((proc.stderr or proc.stdout)[-2500:])
    for line in reversed(proc.stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {}


def _humanml_root(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover root quaternion (wxyz) and translation from HumanML3D."""
    rot_ang = np.zeros(len(data), np.float64)
    if len(data) > 1:
        rot_ang[1:] = data[:-1, 0]
    rot_ang = np.cumsum(rot_ang)
    quat = np.zeros((len(data), 4), np.float64)
    quat[:, 0], quat[:, 2] = np.cos(rot_ang), np.sin(rot_ang)

    delta = np.zeros((len(data), 3), np.float64)
    if len(data) > 1:
        # Avoid NumPy's paired advanced-indexing semantics here: these are
        # independent X/Z columns for every frame, just like MoMask's
        # ``r_pos[..., 1:, [0, 2]]`` tensor assignment.
        delta[1:, 0] = data[:-1, 1]
        delta[1:, 2] = data[:-1, 2]
    # q^-1 * v * q, matching MoMask recover_root_rot_pos.
    qvec = -quat[:, 1:]
    uv = np.cross(qvec, delta)
    uuv = np.cross(qvec, uv)
    delta = delta + 2.0 * (quat[:, :1] * uv + uuv)
    root = np.cumsum(delta, axis=0)
    root[:, 1] = data[:, 3]
    return quat, root


def humanml_to_global_rot6d(data: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """Native HumanML3D 263-D frames -> AlphaMotion's SMPL-22 contract."""
    data = np.asarray(data, np.float32)
    if data.ndim != 2 or data.shape[1] != 263 or not len(data):
        raise ValueError("MoMask output must have shape [frames, 263]")
    quat, root = _humanml_root(data)
    w, x, y, z = [quat[:, i] for i in range(4)]
    root_R = np.stack([
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ], axis=-1).reshape(-1, 3, 3)
    root_6d = np.concatenate([root_R[:, :, 0], root_R[:, :, 1]], axis=-1)
    local = np.concatenate([root_6d[:, None],
                            data[:, 67:193].reshape(-1, 21, 6)], axis=1)
    local_R = rot6d_to_matrix(torch.from_numpy(local).float()).numpy()
    global_R = np.empty_like(local_R)
    for joint, parent in enumerate(SMPL_PARENTS):
        global_R[:, joint] = (local_R[:, joint] if parent < 0 else
                              global_R[:, parent] @ local_R[:, joint])
    rot6d = matrix_to_rot6d(torch.from_numpy(global_R)).float()
    root_cm = (root - root[:1]) * 100.0
    return rot6d, root_cm.astype(np.float64)


def _resample(rot6d: torch.Tensor, root: np.ndarray,
              frames: int) -> tuple[torch.Tensor, np.ndarray]:
    if len(rot6d) == frames:
        return rot6d, root
    # Linear 6D interpolation followed by Gram-Schmidt is stable here and
    # avoids Euler discontinuities; Greenwich consumes the same representation.
    x = rot6d.permute(1, 2, 0).reshape(1, -1, len(rot6d))
    x = torch.nn.functional.interpolate(
        x, size=frames, mode="linear", align_corners=True)
    out = x.reshape(22, 6, frames).permute(2, 0, 1)
    out = matrix_to_rot6d(rot6d_to_matrix(out))
    src = np.linspace(0.0, 1.0, len(root))
    dst = np.linspace(0.0, 1.0, frames)
    root_out = np.stack([np.interp(dst, src, root[:, i])
                         for i in range(3)], axis=1)
    return out.float(), root_out.astype(np.float64)


def motion_from_prompt(text: str, seconds: float = 5.0
                       ) -> tuple[torch.Tensor, np.ndarray]:
    probe = status()
    if not probe["text"]:
        raise RuntimeError(f"MoMask is not configured: {probe['missing']}")
    frames = max(30, int(round(seconds * 30.0)))
    native = max(4, int(round(seconds * 20.0)))
    key = hashlib.sha256(f"{text}\0{native}".encode()).hexdigest()[:20]
    artifact = cache_dir() / "motion_perception" / "momask" / f"{key}.npz"
    if not artifact.is_file():
        _run([CONFIG.motion_python, str(_worker()),
              "--repo", CONFIG.momask_repo, "--text", text,
              "--frames", str(native), "--output", str(artifact)],
             Path(CONFIG.momask_repo), 600)
    with np.load(artifact) as saved:
        rot, root = humanml_to_global_rot6d(saved["humanml"])
    return _resample(rot, root, frames)


def motion_from_video(video_path: str, frames: int | None = None
                      ) -> tuple[torch.Tensor, np.ndarray]:
    probe = status()
    if not probe["video"]:
        raise RuntimeError(f"GVHMR is not configured: {probe['missing']}")
    source = Path(video_path).resolve()
    if not source.is_file():
        raise RuntimeError(f"video not found: {source}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    out_root = cache_dir() / "motion_perception" / "gvhmr" / digest
    artifact = out_root / source.stem / "hmr4d_results.pt"
    if not artifact.is_file():
        cmd = [CONFIG.motion_python, "tools/demo/demo.py",
               "--video", str(source), "--output_root", str(out_root)]
        proc = subprocess.run(cmd, cwd=CONFIG.gvhmr_repo, text=True,
                              capture_output=True, timeout=1900)
        # SimpleVO occasionally cannot estimate a two-view transform (even on
        # GVHMR's own tennis example with newer pycolmap). Reuse the expensive
        # cached 2D preprocessing and retry the model's documented static-
        # camera path. This preserves moving-camera world motion whenever VO
        # succeeds while making ordinary fixed-camera input dependable.
        if proc.returncode and not artifact.is_file():
            fallback = subprocess.run(
                [*cmd, "--static_cam"], cwd=CONFIG.gvhmr_repo, text=True,
                capture_output=True, timeout=1900)
            if fallback.returncode and not artifact.is_file():
                details = fallback.stderr or fallback.stdout
                raise RuntimeError(details[-2500:])
        # The official demo renders after saving the parameters.  A headless
        # renderer failure is harmless when the required artifact exists.
        if proc.returncode and not artifact.is_file():
            raise RuntimeError((proc.stderr or proc.stdout)[-2500:])
    pred = torch.load(artifact, map_location="cpu", weights_only=False)
    packed = {"body_params_global": pred["smpl_params_global"]}
    rot, root = (smpl_to_global_rot6d(packed),
                 smpl_root_translation(packed))
    return _resample(rot, root, frames) if frames is not None else (rot, root)
