from contextlib import nullcontext
import threading

import numpy as np

from alphamotion.viz.live import LiveViewer


class _Value:
    def __init__(self, value):
        self.value = value


class _Camera:
    def __init__(self, position, look_at):
        self._position = np.asarray(position, np.float64)
        self.look_at = np.asarray(look_at, np.float64)
        self.up_direction = np.asarray([0, 0, 1], np.float64)
        self.fov = 0.8
        self.client = None

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        value = np.asarray(value, np.float64)
        delta = value - self._position
        self._position = value
        # This is Viser CameraHandle.position's documented behavior.
        self.look_at = self.look_at + delta


class _Client:
    def __init__(self, camera):
        self.camera = camera
        camera.client = self

    def atomic(self):
        return nullcontext()


class _Server:
    def __init__(self, client):
        self.client = client

    def atomic(self):
        return nullcontext()

    def get_clients(self):
        return {1: self.client}


class _World:
    def __init__(self):
        self.position = np.zeros(3, np.float64)


def _viewer(client):
    viewer = LiveViewer.__new__(LiveViewer)
    viewer.server = _Server(client)
    viewer._lock = threading.RLock()
    viewer._handles = []
    viewer._annotations = []
    viewer._pos = np.zeros((3, 0, 3), np.float32)
    viewer._wxyz = np.zeros((3, 0, 4), np.float32)
    viewer._skin_vertices = None
    viewer._frame = _Value(1)
    viewer._frame.max = 2
    viewer._play = _Value(True)
    viewer._follow = _Value(True)
    viewer._speed = _Value(1.0)
    viewer._fps = 30.0
    viewer._title = _Value("motion")
    viewer._world = _World()
    viewer._frame_center = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                                      np.float64)
    viewer._camera_center = viewer._frame_center[0].copy()
    viewer._camera_radius = 1.0
    viewer._camera_peer = None
    viewer._camera_linked = False
    viewer._camera_ignore_until = 0.0
    viewer._follow_origin = viewer._camera_center.copy()
    return viewer


def test_follow_moves_world_without_touching_user_camera():
    camera = _Camera([1, -2, 1], [0, 0.5, 0])
    client = _Client(camera)
    viewer = _viewer(client)
    position_before = camera.position.copy()
    look_at_before = camera.look_at.copy()

    viewer._show(1)

    np.testing.assert_allclose(camera.position, position_before)
    np.testing.assert_allclose(camera.look_at, look_at_before)
    np.testing.assert_allclose(viewer._world.position, [-1, 0, 0])


def test_follow_does_not_overwrite_orbit_received_during_playback():
    camera = _Camera([1, -2, 1], [0, 0, 0])
    client = _Client(camera)
    viewer = _viewer(client)

    # Simulate an orbit update received from the browser.
    camera._position = np.asarray([2, -1, 1], np.float64)
    position_before = camera.position.copy()
    look_at_before = camera.look_at.copy()

    viewer._show(1)
    np.testing.assert_allclose(camera.position, position_before)
    np.testing.assert_allclose(camera.look_at, look_at_before)
    np.testing.assert_allclose(viewer._world.position, [-1, 0, 0])


def test_clear_motion_removes_actor_and_resets_transport():
    camera = _Camera([1, -2, 1], [0, 0, 0])
    viewer = _viewer(_Client(camera))

    state = viewer.clear_motion()

    assert viewer._pos is None
    assert viewer._wxyz is None
    assert viewer._frame_center is None
    assert viewer._play.value is False
    assert viewer._frame.value == 0
    assert state["frames"] == 0
    assert state["play"] is False


def test_linked_camera_update_does_not_echo_back_and_snap():
    source_camera = _Camera([2, -2, 1], [0, 0, 0])
    target_camera = _Camera([1, -2, 1], [0, 0, 0])
    source = _viewer(_Client(source_camera))
    target = _viewer(_Client(target_camera))
    source._camera_peer = target
    target._camera_peer = source
    source._camera_linked = target._camera_linked = True

    source._mirror_camera(source_camera)
    source_after_forward_sync = source_camera.position.copy()
    target._mirror_camera(target_camera)

    np.testing.assert_allclose(target_camera.position, source_camera.position)
    np.testing.assert_allclose(source_camera.position, source_after_forward_sync)
