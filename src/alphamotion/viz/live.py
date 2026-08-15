"""LiveViewer — ONE persistent in-process viser server (editor-style).

The per-job subprocess viewers were fragile (port churn, dead iframes). This
is the kimodo/ardy pattern instead: a single 3D canvas that lives as long as
the service, embedded permanently in the frontend; each generation swaps the
scene content in place. Default viser appearance — meshes keep their own MJCF
material colors, untinted.
"""
from __future__ import annotations

import threading
import time

import numpy as np

AX = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float64)


class LiveViewer:
    def __init__(self, port: int):
        import viser
        self.port = port
        self.server = viser.ViserServer(port=port, verbose=False)
        self.server.scene.set_up_direction("+z")
        self.server.scene.add_grid("/floor", width=8, height=8, plane="xy")
        self._handles: list = []
        self._pos = None
        self._wxyz = None
        self._fps = 30.0
        # RLock: gui .value setters fire on_update callbacks SYNCHRONOUSLY
        # in the same thread; _show inside those callbacks re-enters the
        # lock (a plain Lock deadlocked the whole GPU worker here)
        self._lock = threading.RLock()
        self._title = self.server.gui.add_markdown("*waiting for a motion…*")
        self._frame = self.server.gui.add_slider(
            "frame", min=0, max=1, step=1, initial_value=0)
        self._play = self.server.gui.add_checkbox("play", initial_value=True)
        self._speed = self.server.gui.add_slider(
            "speed", min=0.25, max=2.0, step=0.25, initial_value=1.0)
        self._frame.on_update(lambda _: self._show(int(self._frame.value)))
        threading.Thread(target=self._loop, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    # ------------------------------------------------------------- content --
    def set_trace(self, trace, xml: str, body: str) -> None:
        """Precompute mesh poses for the trace and swap the scene."""
        import mujoco as mj
        from scipy.spatial.transform import Rotation

        from ..engine.descriptor import build_from_mjcf
        model = mj.MjModel.from_xml_path(str(xml))
        data = mj.MjData(model)
        spec, dof, rest, qnames, _ = build_from_mjcf(xml, body)
        tab = -np.ones((spec.J, 3), np.int64)
        for j, names in enumerate(qnames):
            for k, name in enumerate(names):
                jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
                if jid >= 0:
                    tab[j, k] = int(model.jnt_qposadr[jid])
        src_of = list(range(spec.J))
        if getattr(trace, "joint_names", None):
            lut = {n: i for i, n in enumerate(trace.joint_names)}
            src_of = [lut.get(n, -1) for n in spec.joint_names]
        roots = [int(model.jnt_qposadr[j]) for j in range(model.njnt)
                 if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE]
        root_adr = roots[0] if roots else -1
        gids = [i for i in range(model.ngeom)
                if model.geom_type[i] == mj.mjtGeom.mjGEOM_MESH
                and model.geom_group[i] == 1] or \
            [i for i in range(model.ngeom)
             if model.geom_type[i] == mj.mjtGeom.mjGEOM_MESH]

        T = trace.frames
        pos = np.zeros((T, len(gids), 3), np.float32)
        wxyz = np.zeros((T, len(gids), 4), np.float32)
        from ..engine.odometry import foot_bodies, stance_offsets
        fb = foot_bodies(model) if root_adr >= 0 else []
        fw = np.zeros((T, len(fb), 3))
        for t in range(T):
            data.qpos[:] = model.qpos0
            if root_adr >= 0:
                pm = trace.gp[t] @ AX / 100.0
                data.qpos[root_adr:root_adr + 3] = \
                    [0, 0, -float(pm[:, 2].min())]
                data.qpos[root_adr + 3:root_adr + 7] = Rotation.from_matrix(
                    AX.T @ trace.rootR[t] @ AX).as_quat(scalar_first=True)
            for j in range(spec.J):
                sj = src_of[j]
                if sj < 0 or sj >= trace.q.shape[1]:
                    continue
                for k in range(3):
                    if tab[j, k] >= 0:
                        data.qpos[tab[j, k]] = trace.q[t, sj, k]
            mj.mj_forward(model, data)
            for i, b in enumerate(fb):
                fw[t, i] = data.xpos[b]
            for i, gid in enumerate(gids):
                pos[t, i] = data.geom_xpos[gid]
                wxyz[t, i] = Rotation.from_matrix(
                    data.geom_xmat[gid].reshape(3, 3)).as_quat(
                        scalar_first=True)
        # stride odometry in the FINAL frame (cache-frame integration lands
        # 15-85 deg off after axis conjugation — 0814 audit): shift every geom
        # by the stance-pinning offset. Poses are rigid, so translation is
        # exact here.
        if len(fb):
            off = stance_offsets(fw)
            pos[:, :, 0] += off[:, 0:1].astype(np.float32)
            pos[:, :, 1] += off[:, 1:2].astype(np.float32)

        with self._lock:
            for h in self._handles:
                h.remove()
            self._handles = []
            for i, gid in enumerate(gids):
                mid = model.geom_dataid[gid]
                va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
                rgba = model.geom_rgba[gid]
                self._handles.append(self.server.scene.add_mesh_simple(
                    f"/robot/m{i}", vertices=model.mesh_vert[va:va + vn],
                    faces=model.mesh_face[fa:fa + fn],
                    color=tuple(int(255 * c) for c in rgba[:3]),
                    flat_shading=False))
            self._pos, self._wxyz = pos, wxyz
            self._fps = float(trace.fps) or 30.0
            self._title.content = f"**{trace.title[:60]}** · {body} · {T}f"
        # gui mutations OUTSIDE the lock: their callbacks call _show
        self._frame.max = T - 1
        self._frame.value = 0
        self._show(0)

    def _show(self, t: int) -> None:
        with self._lock:
            if self._pos is None:
                return
            t = min(t, len(self._pos) - 1)
            for i, h in enumerate(self._handles):
                h.position = self._pos[t, i]
                h.wxyz = self._wxyz[t, i]

    def _loop(self) -> None:
        while True:
            try:
                if self._play.value and self._pos is not None:
                    t = (int(self._frame.value) + 1) % len(self._pos)
                    self._frame.value = t     # server-side assignment does NOT
                    self._show(t)             # fire on_update — drive directly
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0 / max(self._fps * self._speed.value, 1.0))
