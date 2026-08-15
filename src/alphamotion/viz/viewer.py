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
    tr = MotionTrace.load(trace_path)
    model = mj.MjModel.from_xml_path(xml)
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
    gids = [i for i in range(model.ngeom)
            if model.geom_type[i] == mj.mjtGeom.mjGEOM_MESH
            and model.geom_group[i] == 1] or \
        [i for i in range(model.ngeom)
         if model.geom_type[i] == mj.mjtGeom.mjGEOM_MESH]
    AX = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float64)

    T = tr.frames
    pos = np.zeros((T, len(gids), 3), np.float32)
    wxyz = np.zeros((T, len(gids), 4), np.float32)
    for t in range(T):
        data.qpos[:] = model.qpos0
        if root_adr >= 0:
            pm = tr.gp[t] @ AX / 100.0
            data.qpos[root_adr:root_adr + 3] = [0, 0, -float(pm[:, 2].min())]
            data.qpos[root_adr + 3:root_adr + 7] = Rotation.from_matrix(
                AX.T @ tr.rootR[t] @ AX).as_quat(scalar_first=True)
        for j in range(spec.J):
            for k in range(3):
                if tab[j, k] >= 0:
                    data.qpos[tab[j, k]] = tr.q[t, j, k]
        mj.mj_forward(model, data)
        for i, gid in enumerate(gids):
            pos[t, i] = data.geom_xpos[gid]
            wxyz[t, i] = Rotation.from_matrix(
                data.geom_xmat[gid].reshape(3, 3)).as_quat(scalar_first=True)

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/floor", width=6, height=6, plane="xy")
    handles = []
    for i, gid in enumerate(gids):
        mid = model.geom_dataid[gid]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        handles.append(server.scene.add_mesh_simple(
            f"/robot/m{i}", vertices=model.mesh_vert[va:va + vn],
            faces=model.mesh_face[fa:fa + fn], color=(255, 128, 0)))
    frame = server.gui.add_slider("frame", min=0, max=T - 1, step=1,
                                  initial_value=0)
    play = server.gui.add_checkbox("play", initial_value=True)

    def show(t):
        for i, h in enumerate(handles):
            h.position = pos[t, i]
            h.wxyz = wxyz[t, i]
    frame.on_update(lambda _: show(int(frame.value)))
    show(0)
    print(f"VISER READY frames={T}", flush=True)
    while True:
        if play.value:
            frame.value = (int(frame.value) + 1) % T
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
