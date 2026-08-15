"""Headless trace -> mp4. Cross-platform: GL backend picked per platform
inside this process, ffmpeg from imageio-ffmpeg's bundled binary (no system
dependency). viser is the interactive surface; this is the export surface.

Rendering needs the robot's MJCF with meshes. Verified local assets are used
for bundled bodies when available; user-ingested URDFs keep their local mesh
references and render after registration.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import setup_gl_backend
from ..engine.trace import MotionTrace


def render_trace(tr: MotionTrace, xml: str, body: str,
                 width: int = 640, height: int = 560,
                 follow: bool = True) -> np.ndarray:
    setup_gl_backend()
    import mujoco as mj
    from ..engine.descriptor import build_from_mjcf
    from .kinematics import (apply_ground_safe_pose, first_frame_ground_height,
                             free_root_address, joint_qpos_map,
                             root_world_offsets, source_joint_map)
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
    tab = joint_qpos_map(model, qnames)
    # the attached mesh MJCF may carry more/other joints than the descriptor
    # the trace was built with (e.g. h1 with hands vs the 20-joint cache spec);
    # map the trace's q columns onto the mesh's slots BY JOINT NAME
    src_of = source_joint_map(spec, tr)
    root_adr = free_root_address(model)
    ground_z = first_frame_ground_height(
        model, data, tr, spec, tab, src_of, root_adr)

    mj.mj_forward(model, data)
    h0 = float(data.geom_xpos[:, 2].max() - data.geom_xpos[:, 2].min()) or 1.0
    cam = mj.MjvCamera()
    cam.lookat[:] = [0, 0, ground_z + h0 * 0.55]
    cam.distance, cam.elevation, cam.azimuth = 3.1 * h0, -10, 160

    def pose_at(t, offset):
        xyz = np.asarray(offset, np.float64).copy()
        xyz[2] += ground_z
        apply_ground_safe_pose(model, data, tr, spec, tab, src_of, root_adr,
                               t, xyz)

    # world translation. Preferred: the trace's DATA root trajectory (owner
    # design — first frame = origin; trajectory continuity beats foot
    # contact, foot skate/clip accepted). World-vector map measured on three
    # GT windows: (x,y,z) Y-up -> (z,x,y) render Z-up (constant permutation;
    # NOT @AX, NOT the rotation conjugation — those landed 80-100 deg off).
    # Fallback: contact-derived stride odometry on the posed mesh.
    from ..engine.odometry import foot_bodies, stance_offsets
    off = np.zeros((tr.frames, 3))
    if getattr(tr, "root_t", None) is not None:
        off = root_world_offsets(tr.root_t, tr.frames)
    elif root_adr >= 0:
        fb = foot_bodies(model)
        fw = np.zeros((tr.frames, len(fb), 3))
        for t in range(tr.frames):
            pose_at(t, (0.0, 0.0, 0.0))
            for i, b in enumerate(fb):
                fw[t, i] = data.xpos[b]
        off[:, :2] = stance_offsets(fw)

    # camera: when the motion travels, look at the path's midpoint from the
    # SIDE (azimuth perpendicular to the net displacement) so the walk crosses
    # the frame laterally — head-on, a 60 cm walk reads as treadmill.
    span = off[:, :2].max(0) - off[:, :2].min(0)
    travel = float(np.linalg.norm(off[-1, :2]))
    if travel > 0.25 and not follow:
        mid = (off[:, :2].max(0) + off[:, :2].min(0)) / 2.0
        cam.lookat[0], cam.lookat[1] = mid
        heading = np.degrees(np.arctan2(off[-1, 1] - off[0, 1],
                                        off[-1, 0] - off[0, 0]))
        cam.azimuth = heading + 90.0
        cam.distance = max(3.1 * h0, 1.9 * float(np.linalg.norm(span)))

    # pass 2 — render with the walk
    rend = mj.Renderer(model, height=height, width=width)
    frames = np.zeros((tr.frames, height, width, 3), np.uint8)
    for t in range(tr.frames):
        pose_at(t, off[t])
        if follow:
            cam.lookat[0], cam.lookat[1] = off[t, :2]
            # Follow locomotion in the ground plane, but keep the vertical
            # datum fixed. Following root Z made jumps look stationary and
            # hid exactly the airborne motion the export is meant to show.
            cam.lookat[2] = ground_z + h0 * 0.55
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
