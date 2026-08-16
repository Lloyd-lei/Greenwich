"""SMPL-X source-motion skinning and compact animated hover previews."""
from __future__ import annotations

import base64
import io
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from ..config import CONFIG
from .kinematics import YUP_TO_ZUP


_MODEL_LOCK = threading.RLock()


def model_path() -> Path:
    path = Path(CONFIG.genmo_repo) / "inputs/checkpoints/body_models"
    expected = path / "smplx" / "SMPLX_NEUTRAL.npz"
    if not expected.is_file():
        raise FileNotFoundError(
            "SMPL-X neutral body model is missing: " + str(expected))
    return path


@lru_cache(maxsize=2)
def _model(device: str):
    import smplx
    body = smplx.create(
        str(model_path()), model_type="smplx", gender="neutral",
        num_betas=10, num_pca_comps=12, flat_hand_mean=True)
    return body.to(device).eval()


def skin_global_rot6d(rot6d, parents, root_cm=None, device="cpu"):
    """Skin global 22-joint SMPL rotations into a neutral SMPL-X surface.

    The stored motion convention is centimetres with Y up. Viser uses metres
    with Z up, so the returned vertices are ready to render directly.
    """
    from ..engine.nets.rotations import rot6d_to_matrix

    global_R = rot6d_to_matrix(torch.as_tensor(
        rot6d, dtype=torch.float64)).cpu().numpy()
    parents = np.asarray(parents, np.int64)
    if global_R.ndim != 4 or global_R.shape[1] != 22:
        raise ValueError(f"SMPL source rotations must be [T,22,3,3], got {global_R.shape}")
    local_R = np.empty_like(global_R)
    for joint, parent in enumerate(parents):
        local_R[:, joint] = (global_R[:, joint] if parent < 0 else
                             np.swapaxes(global_R[:, parent], -1, -2)
                             @ global_R[:, joint])
    aa = Rotation.from_matrix(local_R.reshape(-1, 3, 3)).as_rotvec().reshape(
        len(global_R), 22, 3)
    root = np.zeros((len(global_R), 3), np.float32)
    if root_cm is not None:
        value = np.asarray(root_cm, np.float32)
        if value.shape != root.shape:
            raise ValueError(f"root trajectory must be {root.shape}, got {value.shape}")
        root = (value - value[:1]) / 100.0

    with _MODEL_LOCK, torch.inference_mode():
        body = _model(device)
        dtype = body.shapedirs.dtype
        kwargs = {
            "global_orient": torch.as_tensor(
                aa[:, 0], device=device, dtype=dtype),
            "body_pose": torch.as_tensor(
                aa[:, 1:].reshape(len(aa), -1), device=device, dtype=dtype),
            "transl": torch.as_tensor(root, device=device, dtype=dtype),
            "betas": torch.zeros((len(aa), 10), device=device, dtype=dtype),
            "left_hand_pose": torch.zeros(
                (len(aa), body.num_pca_comps), device=device, dtype=dtype),
            "right_hand_pose": torch.zeros(
                (len(aa), body.num_pca_comps), device=device, dtype=dtype),
            "jaw_pose": torch.zeros((len(aa), 3), device=device, dtype=dtype),
            "leye_pose": torch.zeros((len(aa), 3), device=device, dtype=dtype),
            "reye_pose": torch.zeros((len(aa), 3), device=device, dtype=dtype),
            "expression": torch.zeros(
                (len(aa), body.num_expression_coeffs),
                device=device, dtype=dtype),
        }
        vertices = body(**kwargs).vertices.detach().cpu().numpy()
        faces = np.asarray(body.faces, np.uint32)
    vertices = (vertices @ YUP_TO_ZUP).astype(np.float32, copy=False)
    vertices[..., 2] -= float(vertices[0, :, 2].min())
    return vertices, faces


def animated_preview_webp(vertices: np.ndarray, faces: np.ndarray,
                          fps: float = 30.0, max_frames: int = 30) -> bytes:
    """Render a fixed-camera, full-motion animated WebP from a skinned mesh."""
    vertices = np.asarray(vertices, np.float32)
    sample = np.linspace(0, len(vertices) - 1,
                         min(max_frames, len(vertices))).round().astype(int)
    shown = vertices[sample]
    lo, hi = shown.min(axis=(0, 1)), shown.max(axis=(0, 1))
    center = (lo + hi) * 0.5
    radius = max(float(np.linalg.norm(hi - lo)), 1e-3)
    camera = center + np.array([1.7, -2.2, 1.25], np.float32) * radius
    forward = center - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rel = shown - center
    px = rel @ right
    py = rel @ up
    span = max(float(np.ptp(px)), float(np.ptp(py)), 1e-3)
    scale = 0.84 * min(222, 148) / span
    width, height = 222, 148
    light = np.array([0.4, -0.5, 0.75], np.float32)
    light /= np.linalg.norm(light)
    frames = []
    for verts in shown:
        rel_v = verts - center
        x = width * 0.5 + (rel_v @ right) * scale
        y = height * 0.52 - (rel_v @ up) * scale
        depth = rel_v @ forward
        triangles = verts[faces]
        normal = np.cross(triangles[:, 1] - triangles[:, 0],
                          triangles[:, 2] - triangles[:, 0])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8
        shade = np.clip(np.abs(normal @ light), 0.0, 1.0)
        order = np.argsort(depth[faces].mean(axis=1))[::-1]
        image = Image.new("RGB", (width, height), (8, 9, 11))
        draw = ImageDraw.Draw(image)
        for fi in order:
            value = int(125 + 105 * shade[fi])
            color = (value, int(value * 0.88), int(value * 0.72))
            draw.polygon([(float(x[v]), float(y[v])) for v in faces[fi]],
                         fill=color)
        frames.append(image)
    # Sampling preserves the original duration. The extra duplicate last
    # frame creates the requested half-second pause before WebP loops.
    frame_ms = max(16, round(1000 * len(vertices) / max(fps, 1) / len(frames)))
    durations = [frame_ms] * len(frames) + [500]
    frames.append(frames[-1].copy())
    out = io.BytesIO()
    frames[0].save(out, format="WEBP", save_all=True,
                   append_images=frames[1:], duration=durations,
                   loop=0, quality=72, method=4)
    return out.getvalue()


def webp_data_url(data: bytes) -> str:
    return "data:image/webp;base64," + base64.b64encode(data).decode("ascii")
