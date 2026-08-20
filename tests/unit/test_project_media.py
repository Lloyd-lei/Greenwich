from pathlib import Path

import numpy as np
import pytest

from alphamotion.project_media import (inspect_smpl_npz,
                                       missing_urdf_resources)


def test_inspect_smpl_npz_accepts_axis_angle_and_library_rot6d(tmp_path):
    axis_angle = tmp_path / "axis-angle.npz"
    np.savez(axis_angle, poses=np.zeros((12, 156), np.float32),
             mocap_framerate=np.array(60.0))
    assert inspect_smpl_npz(axis_angle) == {
        "frames": 12, "fps": 60.0, "pose_dimensions": 156,
        "representation": "axis_angle",
    }

    normalized = tmp_path / "normalized.npz"
    np.savez(normalized, local_rot6d=np.zeros((8, 22, 6), np.float32),
             root_cm=np.zeros((8, 3), np.float32))
    assert inspect_smpl_npz(normalized) == {
        "frames": 8, "fps": 30.0, "pose_dimensions": 132,
        "representation": "local_rot6d",
    }


def test_inspect_smpl_npz_rejects_non_motion_npz(tmp_path):
    path = tmp_path / "features.npz"
    np.savez(path, features=np.zeros((4, 32), np.float32))
    with pytest.raises(ValueError, match="neither poses nor local_rot6d"):
        inspect_smpl_npz(path)


def test_missing_urdf_resources_reports_unpacked_meshes(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("""<robot name="x"><link name="base"><visual>
        <geometry><mesh filename="./meshes/base.stl"/></geometry>
        </visual></link></robot>""")
    assert missing_urdf_resources(urdf) == ["./meshes/base.stl"]

    meshes = tmp_path / "meshes"
    meshes.mkdir()
    (meshes / "base.stl").write_bytes(b"solid base\nendsolid base\n")
    assert missing_urdf_resources(urdf) == []


def test_missing_urdf_resources_resolves_package_uri_in_zip_root(tmp_path):
    package = tmp_path / "robot_package"
    urdf_dir = package / "description" / "urdf"
    mesh_dir = package / "robot_description" / "meshes"
    urdf_dir.mkdir(parents=True)
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "base.stl").write_bytes(b"solid base\nendsolid base\n")
    urdf = urdf_dir / "robot.urdf"
    urdf.write_text("""<robot name="x"><link name="base"><visual>
        <geometry><mesh filename="package://robot_description/meshes/base.stl"/>
        </geometry></visual></link></robot>""")
    assert missing_urdf_resources(urdf, package) == []
