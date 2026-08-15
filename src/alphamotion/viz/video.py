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


def render_trace(tr: MotionTrace, xml: str, body: str,
                 width: int = 640, height: int = 560,
                 follow: bool = True) -> np.ndarray:
    setup_gl_backend()
    import mujoco as mj
    from scipy.spatial.transform import Rotation

    from ..engine.descriptor import build_from_mjcf
    # vendor MJCFs ship the robot alone — no floor, no light — so offscreen
    # exports came out as a dim robot on black. Stage it: ground plane +
    # overhead light, and a bright headlight.
    try:
        spec = mj.MjSpec.from_file(str(xml))
        spec.worldbody.add_geom(
            name="_am_floor", type=mj.mjtGeom.mjGEOM_PLANE,
            size=[20, 20, 0.1], rgba=[0.92, 0.92, 0.94, 1.0])
        spec.worldbody.add_light(
            name="_am_sun", pos=[0, -1, 3], dir=[0, 0.25, -1],
            diffuse=[0.55, 0.55, 0.55], specular=[0.1, 0.1, 0.1],
            castshadow=True)
        model = spec.compile()
    except Exception:  # noqa: BLE001 — stage is cosmetic, never fatal
        model = mj.MjModel.from_xml_path(str(xml))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    model.vis.headlight.ambient[:] = [0.42, 0.42, 0.42]
    model.vis.headlight.diffuse[:] = [0.65, 0.65, 0.65]
    model.vis.headlight.specular[:] = [0.2, 0.2, 0.2]
    data = mj.MjData(model)
    spec, dof, rest, qnames, _ = build_from_mjcf(xml, body)
    tab = -np.ones((spec.J, 3), np.int64)
    for j, names in enumerate(qnames):
        for k, name in enumerate(names):
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                tab[j, k] = int(model.jnt_qposadr[jid])
    # the attached mesh MJCF may carry more/other joints than the descriptor
    # the trace was built with (e.g. h1 with hands vs the 20-joint cache spec);
    # map the trace's q columns onto the mesh's slots BY JOINT NAME
    src_of = list(range(spec.J))
    if getattr(tr, "joint_names", None):
        lut = {n: i for i, n in enumerate(tr.joint_names)}
        src_of = [lut.get(n, -1) for n in spec.joint_names]
    roots = [int(model.jnt_qposadr[j]) for j in range(model.njnt)
             if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE]
    root_adr = roots[0] if roots else -1

    mj.mj_forward(model, data)
    h0 = float(data.geom_xpos[:, 2].max() - data.geom_xpos[:, 2].min()) or 1.0
    cam = mj.MjvCamera()
    cam.lookat[:] = [0, 0, h0 * 0.55]
    cam.distance, cam.elevation, cam.azimuth = 3.1 * h0, -10, 160

    def pose_at(t, off_xy):
        data.qpos[:] = model.qpos0
        if root_adr >= 0:
            pm = tr.gp[t] @ AX / 100.0
            data.qpos[root_adr:root_adr + 3] = \
                [off_xy[0], off_xy[1], -float(pm[:, 2].min())]
            data.qpos[root_adr + 3:root_adr + 7] = Rotation.from_matrix(
                AX.T @ tr.rootR[t] @ AX).as_quat(scalar_first=True)
        for j in range(spec.J):
            sj = src_of[j]
            if sj < 0 or sj >= tr.q.shape[1]:
                continue
            for k in range(3):
                if tab[j, k] >= 0:
                    data.qpos[tab[j, k]] = tr.q[t, sj, k]
        mj.mj_forward(model, data)

    # pass 1 — FK only, feet in the FINAL world frame. Stride odometry must
    # run HERE, not in the cache frame: the Y-up->Z-up conjugation does not
    # commute with the root rotation (measured 15-85 deg direction error when
    # integrated upstream; 0.20 cm/frame stance slide when integrated here).
    from ..engine.odometry import foot_bodies, stance_offsets
    off = np.zeros((tr.frames, 2))
    if root_adr >= 0:
        fb = foot_bodies(model)
        fw = np.zeros((tr.frames, len(fb), 3))
        for t in range(tr.frames):
            pose_at(t, (0.0, 0.0))
            for i, b in enumerate(fb):
                fw[t, i] = data.xpos[b]
        off = stance_offsets(fw)

    # pass 2 — render with the walk
    rend = mj.Renderer(model, height=height, width=width)
    frames = np.zeros((tr.frames, height, width, 3), np.uint8)
    for t in range(tr.frames):
        pose_at(t, off[t])
        if follow:
            cam.lookat[0], cam.lookat[1] = off[t]
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
