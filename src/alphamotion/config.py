"""Runtime configuration. Environment prefix: ALPHAMOTION_*."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field


def default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def setup_gl_backend() -> None:
    """Platform-conditional GL selection for MuJoCo offscreen rendering.

    Linux headless -> EGL; Windows -> leave unset (WGL default); macOS -> cgl.
    Must be called BEFORE importing mujoco. The API gateway calls this during
    module initialization because its persistent viewer and MP4 exporter both
    use MuJoCo in the warm process.
    """
    if "MUJOCO_GL" in os.environ:
        return
    if sys.platform.startswith("linux"):
        os.environ["MUJOCO_GL"] = "egl"
    elif sys.platform == "darwin":
        os.environ["MUJOCO_GL"] = "cgl"
    # win32: WGL default


@dataclass
class AMConfig:
    device: str = field(default_factory=default_device)
    host: str = os.environ.get("ALPHAMOTION_HOST", "127.0.0.1")
    port: int = int(os.environ.get("ALPHAMOTION_PORT", "7860"))
    # target motion, body audit, and independent SMPL-X source comparison.
    viewer_ports: tuple[int, int, int] = (7871, 7876, 7877)
    hf_repo: str = os.environ.get("ALPHAMOTION_HF_REPO", "lloydlei/Greenwich")
    # perception (optional): python executable of the env that runs GENMO
    genmo_python: str = os.environ.get("ALPHAMOTION_GENMO_PYTHON", "")
    genmo_repo: str = os.environ.get("ALPHAMOTION_GENMO_REPO", "")


CONFIG = AMConfig()
