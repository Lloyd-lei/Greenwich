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
    def __init__(self, port: int, *, loop_playback: bool = True,
                 autoplay: bool = True, on_playback_end=None):
        import viser
        self.port = port
        self.server = viser.ViserServer(port=port, verbose=False)
        self.server.scene.set_up_direction("+z")
        # Follow mode moves this scene root instead of writing the browser's
        # camera every frame. Camera writes race Viser's orbit controls and
        # make manual rotations snap backwards; a world transform is visually
        # equivalent to camera translation and never takes camera ownership.
        self._world = self.server.scene.add_frame("/world", show_axes=False)
        self.server.scene.add_grid(
            "/world/floor", width=8, height=8, plane="xy")
        self.server.gui.configure_theme(
            control_layout="collapsible", control_width="small",
            dark_mode=True, show_logo=False, show_share_button=False,
            brand_color=(255, 126, 0))
        self._handles: list = []
        self._annotations: list = []
        self._pos = None
        self._wxyz = None
        self._skin_vertices = None
        self._fps = 30.0
        self._camera_center = np.array([0.0, 0.0, 0.7], np.float64)
        self._camera_radius = 1.25
        self._frame_center = None
        self._follow_origin = np.zeros(3, np.float64)
        self._loop_playback = bool(loop_playback)
        self._autoplay = bool(autoplay)
        self._on_playback_end = on_playback_end
        self._camera_peer = None
        self._camera_linked = False
        self._camera_ignore_until = 0.0
        self._camera_hooks: set[int] = set()
        # RLock: gui .value setters fire on_update callbacks SYNCHRONOUSLY
        # in the same thread; _show inside those callbacks re-enters the
        # lock (a plain Lock deadlocked the whole GPU worker here)
        self._lock = threading.RLock()
        self.server.on_client_connect(self._configure_camera)
        self._title = self.server.gui.add_markdown("*waiting for a motion…*")
        self._frame = self.server.gui.add_slider(
            "frame", min=0, max=1, step=1, initial_value=0)
        self._play = self.server.gui.add_checkbox(
            "play", initial_value=self._autoplay)
        self._follow = self.server.gui.add_checkbox(
            "follow camera", initial_value=True)
        self._speed = self.server.gui.add_slider(
            "speed", min=0.25, max=2.0, step=0.25, initial_value=1.0)
        self._frame.on_update(lambda _: self._show(int(self._frame.value)))
        self._follow.on_update(lambda _: self._sync_follow_targets())
        threading.Thread(target=self._loop, daemon=True).start()

    def _sync_follow_targets(self) -> None:
        """Start/stop scene-root follow at the current displayed position."""
        with self._lock:
            if self._frame_center is not None:
                index = min(int(self._frame.value), len(self._frame_center) - 1)
                target = self._frame_center[index].astype(np.float64)
            else:
                target = self._camera_center.copy()
            self._follow_origin = target
            self._world.position = (0.0, 0.0, 0.0)
            self._camera_center = target

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
            position = center + np.asarray([1.7, -2.2, 1.25]) * radius
            client.camera.position = tuple(position)
            client.camera.look_at = tuple(center)
            client.camera.fov = 0.8
        key = id(client)
        if key not in self._camera_hooks:
            self._camera_hooks.add(key)
            client.camera.on_update(self._mirror_camera)

    def link_camera_peer(self, peer) -> None:
        """Pair two viewers so orbit/pan/zoom can be mirrored on demand."""
        self._camera_peer = peer

    def set_camera_linked(self, enabled: bool, *, reset: bool = True) -> None:
        self._camera_linked = bool(enabled)
        if reset:
            self.reset_camera()

    def reset_camera(self) -> None:
        # Camera setters produce browser update events too. Ignore those
        # programmatic echoes so a reset cannot bounce through the peer.
        self._camera_ignore_until = time.monotonic() + 0.20
        for client in self.server.get_clients().values():
            self._configure_camera(client)

    def _mirror_camera(self, camera) -> None:
        peer = self._camera_peer
        if (time.monotonic() < self._camera_ignore_until
                or not self._camera_linked or peer is None
                or not peer._camera_linked):
            return
        source_radius = max(float(self._camera_radius), 1e-6)
        target_radius = max(float(peer._camera_radius), 1e-6)
        relative_position = ((np.asarray(camera.position)
                              - self._camera_center) / source_radius)
        relative_look = ((np.asarray(camera.look_at)
                          - self._camera_center) / source_radius)
        position = peer._camera_center + relative_position * target_radius
        look_at = peer._camera_center + relative_look * target_radius
        # Assigning position/look_at/fov on the peer emits camera updates.
        # Suppress that echo long enough for the websocket batch to arrive;
        # otherwise the two viewers repeatedly overwrite an in-progress orbit
        # and the camera appears to rotate forward then snap halfway back.
        peer._camera_ignore_until = time.monotonic() + 0.20
        for client in peer.server.get_clients().values():
            with client.atomic():
                client.camera.up_direction = tuple(camera.up_direction)
                client.camera.position = tuple(position)
                client.camera.look_at = tuple(look_at)
                client.camera.fov = float(camera.fov)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def transport_state(self) -> dict:
        """Current playback state for the editor's external transport bar."""
        with self._lock:
            frames = 0 if self._pos is None else len(self._pos)
            frame = min(int(self._frame.value), max(frames - 1, 0))
            return {"frame": frame, "frames": frames,
                    "play": bool(self._play.value),
                    "follow": bool(self._follow.value),
                    "speed": float(self._speed.value),
                    "fps": float(self._fps)}

    def set_transport(self, *, frame: int | None = None,
                      play: bool | None = None,
                      follow: bool | None = None,
                      speed: float | None = None) -> dict:
        """Apply external editor transport controls to the Viser player."""
        if play is not None:
            self._play.value = bool(play)
        if follow is not None:
            self._follow.value = bool(follow)
            self._sync_follow_targets()
        if speed is not None:
            self._speed.value = float(np.clip(speed, 0.25, 2.0))
        if frame is not None and self._pos is not None:
            value = int(np.clip(frame, 0, len(self._pos) - 1))
            self._frame.value = value
            self._show(value)
        return self.transport_state()

    def clear_motion(self) -> dict:
        """Remove the active motion and every rendered actor from the canvas."""
        self._play.value = False
        with self._lock:
            self._clear_content()
            self._pos = None
            self._wxyz = None
            self._skin_vertices = None
            self._frame_center = None
            self._world.position = (0.0, 0.0, 0.0)
            self._title.content = "*timeline is empty*"
        # GUI mutations stay outside the lock because setting the slider can
        # synchronously invoke its on_update callback.
        self._frame.max = 1
        self._frame.value = 0
        return self.transport_state()

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
                                 contact_stabilized_root_offsets,
                                 first_frame_ground_height,
                                 free_root_address, joint_qpos_map,
                                 source_joint_map,
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
        root_off, _contact_report = contact_stabilized_root_offsets(
            model, data, trace, spec, tab, src_of, root_adr, ground_z)
        for t in range(T):
            xyz = root_off[t].copy(); xyz[2] += ground_z
            apply_ground_safe_pose(model, data, trace, spec, tab, src_of,
                                   root_adr, t, xyz)
            if root_body >= 0:
                focus[t] = data.xpos[root_body]
            else:
                focus[t] = np.mean(data.geom_xpos[gids], axis=0)
            for i, gid in enumerate(gids):
                pos[t, i] = data.geom_xpos[gid]
                wxyz[t, i] = Rotation.from_matrix(
                    data.geom_xmat[gid].reshape(3, 3)).as_quat(
                        scalar_first=True)
        with self._lock:
            self._clear_content()
            self._skin_vertices = None
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
                    f"/world/robot/m{i}",
                    vertices=model.mesh_vert[va:va + vn],
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
            self._follow_origin = self._camera_center.copy()
            self._world.position = (0.0, 0.0, 0.0)
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

    def set_smplx_skin(self, vertices: np.ndarray, faces: np.ndarray,
                       title: str, fps: float = 30.0) -> None:
        """Display an independently controlled, fully skinned SMPL-X motion."""
        # Long imported clips stay compact in float16; only the visible frame
        # is expanded to float32 when it is sent to the browser.
        vertices = np.asarray(vertices)
        faces = np.asarray(faces, np.uint32)
        if vertices.ndim != 3 or vertices.shape[-1] != 3 or not len(vertices):
            raise ValueError("SMPL-X vertices must have shape [T,V,3]")
        with self._lock:
            self._clear_content()
            self._skin_vertices = vertices
            self._handles = [self.server.scene.add_mesh_simple(
                "/world/source/smplx",
                vertices=vertices[0].astype(np.float32), faces=faces,
                color=(221, 170, 116), side="double", flat_shading=False,
                cast_shadow=True, receive_shadow=True)]
            self._pos = np.zeros((len(vertices), 1, 3), np.float32)
            self._wxyz = None
            self._fps = float(fps) or 30.0
            lo = vertices.min(axis=(0, 1)).astype(np.float32)
            hi = vertices.max(axis=(0, 1)).astype(np.float32)
            center = (lo + hi) * 0.5
            # Keep the default camera framed around the complete source clip,
            # but retain a moving per-frame centre for Follow mode.
            frame_lo = vertices.min(axis=1).astype(np.float32)
            frame_hi = vertices.max(axis=1).astype(np.float32)
            from .kinematics import smooth_camera_path
            self._frame_center = smooth_camera_path(
                (frame_lo + frame_hi) * 0.5, self._fps, window_s=0.30)
            self._camera_center = center.astype(np.float64)
            self._follow_origin = self._camera_center.copy()
            self._world.position = (0.0, 0.0, 0.0)
            self._camera_radius = float(np.clip(
                np.linalg.norm(hi - lo) * 0.58, 0.75, 3.0))
            self._follow.value = False
            self._play.value = False
            self._title.content = f"**{title[:60]}** · SMPL-X source · {len(vertices)}f"
        self._frame.max = len(vertices) - 1
        self._frame.value = 0
        self._show(0)
        self.reset_camera()

    def set_body_preview(self, embodiment, semantics: dict | None = None) -> None:
        """Show one embodiment as a translucent mesh + labeled topology.

        Descriptor-only bundled bodies still get a faithful canonical
        skeleton.  When an MJCF/URDF is attached, its real visual meshes are
        rendered underneath the same semantic overlay.
        """
        from scipy.spatial.transform import Rotation

        from ..engine.spatial import rest_positions
        from .kinematics import (free_root_address, lowest_visual_z,
                                 preview_joint_positions,
                                 visual_mesh_geom_ids)

        spec = embodiment.spec
        labels = (semantics or {}).get("per_joint", semantics or {})
        canonical = preview_joint_positions(
            rest_positions(spec), align_to_mjcf=bool(embodiment.xml))
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
            self._skin_vertices = None
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
                                f"/world/body/mesh{i}",
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
                            "/world/body/topology", edges,
                            colors=(255, 128, 0), line_width=2.5))
                self._annotations.append(self.server.scene.add_point_cloud(
                    "/world/body/joints", canonical.astype(np.float32), colors,
                    point_size=0.025, point_shape="circle"))
                for j, (name, point) in enumerate(
                        zip(spec.joint_names, canonical)):
                    part = labels.get(str(name), "unlabeled")
                    self._annotations.append(self.server.scene.add_label(
                        f"/world/body/labels/{j}", f"{name} · {part}",
                        position=point, font_screen_scale=0.65,
                        depth_test=False, anchor="bottom-left"))
            lo, hi = canonical.min(0), canonical.max(0)
            self._camera_center = (lo + hi) / 2
            self._follow_origin = self._camera_center.copy()
            self._world.position = (0.0, 0.0, 0.0)
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
                skin_vertices = getattr(self, "_skin_vertices", None)
                if skin_vertices is not None:
                    self._handles[0].vertices = skin_vertices[t].astype(
                        np.float32)
                else:
                    for i, h in enumerate(self._handles):
                        h.position = self._pos[t, i]
                        h.wxyz = self._wxyz[t, i]
                if self._follow.value and self._frame_center is not None:
                    target = self._frame_center[t].astype(np.float64)
                    offset = self._follow_origin - target
                    self._world.position = tuple(offset)
                    self._camera_center = target + offset
                elif self._frame_center is not None:
                    self._world.position = (0.0, 0.0, 0.0)
                    self._camera_center = self._frame_center[t].astype(
                        np.float64)

    def _loop(self) -> None:
        while True:
            try:
                if self._play.value and self._pos is not None:
                    current = int(self._frame.value)
                    if not self._loop_playback and current >= len(self._pos) - 1:
                        self._play.value = False
                        if self._on_playback_end is not None:
                            self._on_playback_end()
                        continue
                    t = (current + 1) % len(self._pos)
                    self._frame.value = t     # server-side assignment does NOT
                    self._show(t)             # fire on_update — drive directly
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0 / max(self._fps * self._speed.value, 1.0))
