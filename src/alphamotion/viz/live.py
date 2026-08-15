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


class LiveViewer:
    def __init__(self, port: int):
        import viser
        self.port = port
        self.server = viser.ViserServer(port=port, verbose=False)
        self.server.scene.set_up_direction("+z")
        self.server.scene.add_grid("/floor", width=8, height=8, plane="xy")
        self.server.gui.configure_theme(
            control_layout="collapsible", control_width="small",
            dark_mode=True, show_logo=False, show_share_button=False,
            brand_color=(255, 126, 0))
        self._handles: list = []
        self._annotations: list = []
        self._pos = None
        self._wxyz = None
        self._fps = 30.0
        self._camera_center = np.array([0.0, 0.0, 0.7], np.float64)
        self._camera_radius = 1.25
        self._frame_center = None
        self._client_follow_target: dict[int, np.ndarray] = {}
        self.server.on_client_connect(self._configure_camera)
        # RLock: gui .value setters fire on_update callbacks SYNCHRONOUSLY
        # in the same thread; _show inside those callbacks re-enters the
        # lock (a plain Lock deadlocked the whole GPU worker here)
        self._lock = threading.RLock()
        self._title = self.server.gui.add_markdown("*waiting for a motion…*")
        self._frame = self.server.gui.add_slider(
            "frame", min=0, max=1, step=1, initial_value=0)
        self._play = self.server.gui.add_checkbox("play", initial_value=True)
        self._follow = self.server.gui.add_checkbox(
            "follow camera", initial_value=True)
        self._speed = self.server.gui.add_slider(
            "speed", min=0.25, max=2.0, step=0.25, initial_value=1.0)
        self._frame.on_update(lambda _: self._show(int(self._frame.value)))
        threading.Thread(target=self._loop, daemon=True).start()

    def _configure_camera(self, client) -> None:
        """Frame the current trajectory for both existing and new clients."""
        center = self._camera_center
        radius = self._camera_radius
        # Viser updates camera orientation when position changes. Setting
        # look_at first therefore leaves some clients aimed at the old origin
        # after the subsequent translation (a valid scene, but black canvas).
        # Establish the camera frame first and set its target last.
        # Position and look-at must arrive in one websocket batch. Sending
        # them separately exposes an intermediate camera pose to the browser,
        # which appears as a full-canvas flash during playback.
        with client.atomic():
            client.camera.up_direction = (0.0, 0.0, 1.0)
            client.camera.position = tuple(
                center + np.asarray([1.7, -2.2, 1.25]) * radius)
            client.camera.look_at = tuple(center)
            client.camera.fov = 0.8
        self._client_follow_target[id(client)] = np.asarray(
            center, np.float64).copy()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    # ------------------------------------------------------------- content --
    def _clear_content(self) -> None:
        for handle in (*self._handles, *self._annotations):
            handle.remove()
        self._handles = []
        self._annotations = []

    def set_trace(self, trace, xml: str, body: str) -> None:
        """Precompute mesh poses for the trace and swap the scene."""
        import mujoco as mj
        from scipy.spatial.transform import Rotation
        from ..engine.descriptor import build_from_mjcf
        from .kinematics import (apply_ground_safe_pose,
                                 first_frame_ground_height,
                                 free_root_address, joint_qpos_map,
                                 root_world_offsets, source_joint_map,
                                 smooth_camera_path, visual_mesh_geom_ids)
        model = mj.MjModel.from_xml_path(str(xml))
        data = mj.MjData(model)
        spec, dof, rest, qnames, _ = build_from_mjcf(xml, body)
        tab = joint_qpos_map(model, qnames)
        src_of = source_joint_map(spec, trace)
        root_adr = free_root_address(model)
        root_body = next((int(model.jnt_bodyid[j])
                          for j in range(model.njnt)
                          if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE), -1)
        gids = visual_mesh_geom_ids(model)
        ground_z = first_frame_ground_height(
            model, data, trace, spec, tab, src_of, root_adr)

        T = trace.frames
        pos = np.zeros((T, len(gids), 3), np.float32)
        wxyz = np.zeros((T, len(gids), 4), np.float32)
        focus = np.zeros((T, 3), np.float64)
        from ..engine.odometry import foot_bodies, stance_offsets
        fb = foot_bodies(model) if root_adr >= 0 else []
        fw = np.zeros((T, len(fb), 3))
        root_off = root_world_offsets(
            trace.root_t, T) if getattr(trace, "root_t", None) is not None \
            else np.zeros((T, 3))
        for t in range(T):
            xyz = root_off[t].copy(); xyz[2] += ground_z
            apply_ground_safe_pose(model, data, trace, spec, tab, src_of,
                                   root_adr, t, xyz)
            if root_body >= 0:
                focus[t] = data.xpos[root_body]
            else:
                focus[t] = np.mean(data.geom_xpos[gids], axis=0)
            for i, b in enumerate(fb):
                fw[t, i] = data.xpos[b]
            for i, gid in enumerate(gids):
                pos[t, i] = data.geom_xpos[gid]
                wxyz[t, i] = Rotation.from_matrix(
                    data.geom_xmat[gid].reshape(3, 3)).as_quat(
                        scalar_first=True)
        # world translation: DATA root trajectory when the trace carries one
        # (owner design — first frame = origin; continuity beats contact;
        # world-vector map (x,y,z) Y-up -> (z,x,y) Z-up, measured on three GT
        # windows). Contact-derived stride odometry only as fallback.
        if getattr(trace, "root_t", None) is None and len(fb):
            off = stance_offsets(fw)
            pos[:, :, 0] += off[:, 0:1].astype(np.float32)
            pos[:, :, 1] += off[:, 1:2].astype(np.float32)
            focus[:, :2] += off[:, :2]

        with self._lock:
            self._clear_content()
            for i, gid in enumerate(gids):
                mid = model.geom_dataid[gid]
                va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
                rgba = model.geom_rgba[gid].copy()
                # Vendor H1 meshes are nearly black. On the product's dark
                # canvas that made a correctly framed robot literally
                # invisible; lift only near-black materials to neutral metal.
                if float(np.max(rgba[:3])) < 0.18:
                    rgba[:3] = [0.68, 0.71, 0.75]
                self._handles.append(self.server.scene.add_mesh_simple(
                    f"/robot/m{i}", vertices=model.mesh_vert[va:va + vn],
                    faces=model.mesh_face[fa:fa + fn],
                    color=tuple(int(np.clip(255 * c, 0, 255))
                                for c in rgba[:3]),
                    flat_shading=False))
            self._pos, self._wxyz = pos, wxyz
            self._fps = float(trace.fps) or 30.0
            frame_lo, frame_hi = pos.min(axis=1), pos.max(axis=1)
            self._frame_center = smooth_camera_path(
                focus, self._fps, window_s=0.30)
            self._camera_center = self._frame_center[0].astype(np.float64)
            body_span = np.linalg.norm(frame_hi - frame_lo, axis=1)
            self._camera_radius = float(np.clip(
                np.percentile(body_span, 90) * 0.58, 0.75, 1.5))
            self._title.content = f"**{trace.title[:60]}** · {body} · {T}f"
        # gui mutations OUTSIDE the lock: their callbacks call _show
        self._frame.max = T - 1
        self._frame.value = 0
        self._show(0)
        for client in self.server.get_clients().values():
            self._configure_camera(client)

    def set_body_preview(self, embodiment, semantics: dict | None = None) -> None:
        """Show one embodiment as a translucent mesh + labeled topology.

        Descriptor-only bundled bodies still get a faithful canonical
        skeleton.  When an MJCF/URDF is attached, its real visual meshes are
        rendered underneath the same semantic overlay.
        """
        from scipy.spatial.transform import Rotation

        from ..engine.spatial import rest_positions
        from .kinematics import (YUP_TO_ZUP, free_root_address,
                                 lowest_visual_z, visual_mesh_geom_ids)

        spec = embodiment.spec
        labels = (semantics or {}).get("per_joint", semantics or {})
        canonical = rest_positions(spec) @ YUP_TO_ZUP / 100.0
        canonical[:, 2] -= float(canonical[:, 2].min())
        palette = {
            "head": (245, 245, 245), "torso": (255, 128, 0),
            "left arm": (62, 165, 255), "right arm": (59, 214, 127),
            "left leg": (185, 116, 255), "right leg": (255, 200, 66),
        }
        colors = np.asarray([
            palette.get(labels.get(str(name), "torso"), (160, 165, 174))
            for name in spec.joint_names
        ], np.uint8)
        edges = np.asarray([
            [canonical[int(parent)], canonical[j]]
            for j, parent in enumerate(spec.parents) if int(parent) >= 0
        ], np.float32)

        with self._lock:
            self._clear_content()
            self._pos = self._wxyz = self._frame_center = None
            with self.server.atomic():
                if embodiment.xml:
                    import mujoco as mj
                    model = mj.MjModel.from_xml_path(str(embodiment.xml))
                    data = mj.MjData(model)
                    mj.mj_forward(model, data)
                    root_adr = free_root_address(model)
                    low = lowest_visual_z(model, data)
                    if root_adr >= 0 and np.isfinite(low):
                        data.qpos[root_adr + 2] -= low
                        mj.mj_forward(model, data)
                    for i, gid in enumerate(visual_mesh_geom_ids(model)):
                        mid = int(model.geom_dataid[gid])
                        if mid < 0:
                            continue
                        va, vn = int(model.mesh_vertadr[mid]), int(
                            model.mesh_vertnum[mid])
                        fa, fn = int(model.mesh_faceadr[mid]), int(
                            model.mesh_facenum[mid])
                        rgba = np.asarray(model.geom_rgba[gid]).copy()
                        if float(np.max(rgba[:3])) < 0.18:
                            rgba[:3] = [0.68, 0.71, 0.75]
                        self._handles.append(
                            self.server.scene.add_mesh_simple(
                                f"/body/mesh{i}",
                                vertices=model.mesh_vert[va:va + vn],
                                faces=model.mesh_face[fa:fa + fn],
                                color=tuple(int(np.clip(255 * c, 0, 255))
                                            for c in rgba[:3]),
                                opacity=0.48, side="double",
                                cast_shadow=False, receive_shadow=False,
                                flat_shading=False,
                                position=data.geom_xpos[gid],
                                wxyz=Rotation.from_matrix(
                                    data.geom_xmat[gid].reshape(3, 3)
                                ).as_quat(scalar_first=True)))
                if len(edges):
                    self._annotations.append(
                        self.server.scene.add_line_segments(
                            "/body/topology", edges,
                            colors=(255, 128, 0), line_width=2.5))
                self._annotations.append(self.server.scene.add_point_cloud(
                    "/body/joints", canonical.astype(np.float32), colors,
                    point_size=0.025, point_shape="circle"))
                for j, (name, point) in enumerate(
                        zip(spec.joint_names, canonical)):
                    part = labels.get(str(name), "unlabeled")
                    self._annotations.append(self.server.scene.add_label(
                        f"/body/labels/{j}", f"{name} · {part}",
                        position=point, font_screen_scale=0.65,
                        depth_test=False, anchor="bottom-left"))
            lo, hi = canonical.min(0), canonical.max(0)
            self._camera_center = (lo + hi) / 2
            self._camera_radius = float(np.clip(
                np.linalg.norm(hi - lo) * 0.72, 0.65, 1.5))
            self._title.content = (
                f"**{embodiment.name}** · {spec.J} joints · semantic topology")
        for client in self.server.get_clients().values():
            self._configure_camera(client)

    def _show(self, t: int) -> None:
        with self._lock:
            if self._pos is None:
                return
            t = min(t, len(self._pos) - 1)
            # A robot is composed of many mesh handles. Without an atomic
            # batch, the browser briefly renders a mixture of frame t and
            # frame t-1, perceived as limb flicker or an exploding mesh.
            with self.server.atomic():
                for i, h in enumerate(self._handles):
                    h.position = self._pos[t, i]
                    h.wxyz = self._wxyz[t, i]
            if self._follow.value and self._frame_center is not None:
                target = self._frame_center[t].astype(np.float64)
                self._camera_center = target
                for client in self.server.get_clients().values():
                    key = id(client)
                    old_target = self._client_follow_target.get(key, target)
                    old_position = np.asarray(client.camera.position,
                                              np.float64)
                    if np.isfinite(old_target).all() and np.isfinite(
                            old_position).all():
                        with client.atomic():
                            client.camera.position = tuple(
                                old_position + target - old_target)
                            client.camera.look_at = tuple(target)
                        self._client_follow_target[key] = target.copy()

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
