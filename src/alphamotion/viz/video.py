"""Headless trace -> mp4. Cross-platform: GL backend picked per platform
inside this process, ffmpeg from imageio-ffmpeg's bundled binary (no system
dependency). viser is the interactive surface; this is the export surface.

Rendering needs the robot's MJCF with meshes. Bundled bodies ship without
meshes (vendor assets are not redistributable); attach an xml once via
registry meta or ALPHAMOTION_ROBOT_ASSETS, after which rendering works.
User-ingested URDFs render out of the box (their meshes are local).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import setup_gl_backend
from ..engine.trace import MotionTrace

AX = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float64)  # Y-up -> Z-up


def render_trace(trace: MotionTrace, xml: str, body: str,
                 width: int = 640, height: int = 560) -> np.ndarray:
    setup_gl_backend()
    import mujoco as mj
    from scipy.spatial.transform import Rotation

    from ..engine.descriptor import build_from_mjcf
    model = mj.MjModel.from_xml_path(str(xml))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mj.MjData(model)
    spec, dof, rest, qnames, _ = build_from_mjcf(xml, body)
    tab = -np.ones((spec.J, 3), np.int64)
    for j, names in enumerate(qnames):
        for k, name in enumerate(names):
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                tab[j, k] = int(model.jnt_qposadr[jid])
    roots = [int(model.jnt_qposadr[j]) for j in range(model.njnt)
             if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE]
    root_adr = roots[0] if roots else -1

    mj.mj_forward(model, data)
    h0 = float(data.geom_xpos[:, 2].max() - data.geom_xpos[:, 2].min()) or 1.0
    cam = mj.MjvCamera()
    cam.lookat[:] = [0, 0, h0 * 0.55]
    cam.distance, cam.elevation, cam.azimuth = 3.1 * h0, -10, 160

    rend = mj.Renderer(model, height=height, width=width)
    frames = np.zeros((trace.frames, height, width, 3), np.uint8)
    for t in range(trace.frames):
        data.qpos[:] = model.qpos0
        if root_adr >= 0:
            pm = trace.gp[t] @ AX / 100.0
            data.qpos[root_adr:root_adr + 3] = [0, 0, -float(pm[:, 2].min())]
            data.qpos[root_adr + 3:root_adr + 7] = Rotation.from_matrix(
                AX.T @ trace.rootR[t] @ AX).as_quat(scalar_first=True)
        for j in range(spec.J):
            for k in range(3):
                if tab[j, k] >= 0:
                    data.qpos[tab[j, k]] = trace.q[t, j, k]
        mj.mj_forward(model, data)
        rend.update_scene(data, cam)
        frames[t] = rend.render()
    del rend
    return frames


def write_mp4(path: str | Path, frames: np.ndarray, fps: float = 30.0) -> Path:
    import imageio.v2 as imageio
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), format="FFMPEG", fps=fps,
                            codec="libx264", quality=7,
                            pixelformat="yuv420p") as w:
        for f in frames:
            w.append_data(f)
    return path


def trace_to_mp4(trace_path: str | Path, xml: str, body: str,
                 out: str | Path, fps: float | None = None) -> Path:
    tr = MotionTrace.load(trace_path)
    frames = render_trace(tr, xml, body)
    return write_mp4(out, frames, fps or tr.fps)
