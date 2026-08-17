"""Interactive viser viewers.

motion_viewer: play a MotionTrace on a robot with meshes.
body_viewer:   rest-pose inspection with SEMANTIC coloring — each joint's mesh
               group tinted by its labeled part (semi-transparent), the
               product surface for auditing URDF ingest labels.
Run via `python -m alphamotion.viz.viewer --trace x.npz --xml r.xml --body n`.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from ..config import setup_gl_backend

PART_COLORS = {
    "head": (255, 158, 64), "torso": (200, 200, 210),
    "left arm": (255, 128, 0), "right arm": (255, 200, 120),
    "left leg": (96, 150, 247), "right leg": (140, 190, 255),
}


def motion_viewer(trace_path: str, xml: str, body: str, port: int = 7871):
    setup_gl_backend()
    import mujoco as mj
    import viser
    from scipy.spatial.transform import Rotation
    from ..engine.descriptor import build_from_mjcf
    from ..engine.trace import MotionTrace
    from .kinematics import (apply_ground_safe_pose,
                             contact_stabilized_root_offsets,
                             first_frame_ground_height, free_root_address,
                             joint_qpos_map, source_joint_map,
                             visual_mesh_geom_ids)
    tr = MotionTrace.load(trace_path)
    model = mj.MjModel.from_xml_path(xml)
    data = mj.MjData(model)
    spec, dof, rest, qnames, _ = build_from_mjcf(xml, body)
    tab = joint_qpos_map(model, qnames)
    # the attached mesh MJCF may carry more/other joints than the descriptor
    # the trace was built with (e.g. h1 with hands vs the 20-joint cache spec);
    # map the trace's q columns onto the mesh's slots BY JOINT NAME
    src_of = source_joint_map(spec, tr)
    root_adr = free_root_address(model)
    gids = visual_mesh_geom_ids(model)
    ground_z = first_frame_ground_height(
        model, data, tr, spec, tab, src_of, root_adr)
    offsets, _contact_report = contact_stabilized_root_offsets(
        model, data, tr, spec, tab, src_of, root_adr, ground_z)

    T = tr.frames
    pos = np.zeros((T, len(gids), 3), np.float32)
    wxyz = np.zeros((T, len(gids), 4), np.float32)
    for t in range(T):
        xyz = offsets[t].copy(); xyz[2] += ground_z
        apply_ground_safe_pose(model, data, tr, spec, tab, src_of, root_adr,
                               t, xyz)
        for i, gid in enumerate(gids):
            pos[t, i] = data.geom_xpos[gid]
            wxyz[t, i] = Rotation.from_matrix(
                data.geom_xmat[gid].reshape(3, 3)).as_quat(scalar_first=True)

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/floor", width=6, height=6, plane="xy")
    server.gui.configure_theme(
        control_layout="collapsible", control_width="small",
        dark_mode=True, show_logo=False, show_share_button=False,
        brand_color=(255, 126, 0))
    handles = []
    for i, gid in enumerate(gids):
        mid = model.geom_dataid[gid]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        # use the robot's own material colors; a flat single-color blob reads
        # terribly. Untextured default gray gets a papaya tint instead.
        rgba = model.geom_rgba[gid]
        if abs(float(rgba[0]) - 0.5) < 0.01 and abs(float(rgba[1]) - 0.5) < 0.01:
            color = (232, 148, 60)
        elif float(np.max(rgba[:3])) < 0.18:
            color = (173, 181, 191)
        else:
            color = tuple(int(255 * c) for c in rgba[:3])
        handles.append(server.scene.add_mesh_simple(
            f"/robot/m{i}", vertices=model.mesh_vert[va:va + vn],
            faces=model.mesh_face[fa:fa + fn], color=color,
            flat_shading=False))
    frame = server.gui.add_slider("frame", min=0, max=T - 1, step=1,
                                  initial_value=0)
    play = server.gui.add_checkbox("play", initial_value=True)

    def show(t):
        for i, h in enumerate(handles):
            h.position = pos[t, i]
            h.wxyz = wxyz[t, i]
    frame.on_update(lambda _: show(int(frame.value)))
    show(0)

    flat = pos.reshape(-1, 3)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    center = (lo + hi) * 0.5
    radius = float(np.clip(np.linalg.norm(hi - lo) * 0.58, 0.9, 3.0))

    @server.on_client_connect
    def configure_camera(client):
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.position = tuple(
            center + np.asarray([1.7, -2.2, 1.25]) * radius)
        client.camera.look_at = tuple(center)
        client.camera.fov = 0.8

    print(f"VISER READY frames={T}", flush=True)
    while True:
        if play.value:
            t = (int(frame.value) + 1) % T
            frame.value = t
            show(t)
        time.sleep(1.0 / max(tr.fps, 1))


def body_viewer(xml: str, body: str, labels: dict | None = None,
                port: int = 7872, alpha: float = 0.55):
    """Rest pose, semi-transparent, mesh tinted by semantic part labels."""
    setup_gl_backend()
    import mujoco as mj
    import viser
    from scipy.spatial.transform import Rotation
    model = mj.MjModel.from_xml_path(xml)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/floor", width=4, height=4, plane="xy")
    labels = labels or {}
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mj.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[gid]
        bid = model.geom_bodyid[gid]
        bname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) or ""
        part = None
        for jn, p in labels.items():
            if jn.lower() in bname.lower() or bname.lower() in jn.lower():
                part = p
                break
        color = PART_COLORS.get(part, (160, 165, 175))
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        h = server.scene.add_mesh_simple(
            f"/body/g{gid}", vertices=model.mesh_vert[va:va + vn],
            faces=model.mesh_face[fa:fa + fn], color=color,
            opacity=alpha)
        h.position = data.geom_xpos[gid]
        h.wxyz = Rotation.from_matrix(
            data.geom_xmat[gid].reshape(3, 3)).as_quat(scalar_first=True)
    print("BODY VIEWER READY", flush=True)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace")
    ap.add_argument("--xml", required=True)
    ap.add_argument("--body", required=True)
    ap.add_argument("--port", type=int, default=7871)
    a = ap.parse_args()
    if a.trace:
        motion_viewer(a.trace, a.xml, a.body, a.port)
    else:
        body_viewer(a.xml, a.body, port=a.port)
