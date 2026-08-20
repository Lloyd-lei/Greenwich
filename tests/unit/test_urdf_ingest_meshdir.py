from pathlib import Path
from types import SimpleNamespace

from alphamotion.embodiment.urdf_ingest import anchor_urdf_meshdir


def test_anchor_urdf_meshdir_does_not_duplicate_relative_prefix(tmp_path):
    spec = SimpleNamespace(
        meshdir="meshes",
        meshes=[
            SimpleNamespace(file="./meshes/pelvis.stl"),
            SimpleNamespace(file="./meshes/head_pitch_link.stl"),
        ],
    )

    anchor_urdf_meshdir(spec, tmp_path / "X2-Ultra.urdf")

    assert spec.meshdir == str((tmp_path / "meshes").resolve())
    assert [mesh.file for mesh in spec.meshes] == [
        "pelvis.stl",
        "head_pitch_link.stl",
    ]


def test_anchor_urdf_meshdir_keeps_assets_outside_the_declared_prefix(tmp_path):
    spec = SimpleNamespace(
        meshdir="meshes",
        meshes=[SimpleNamespace(file="vendor/shared/pelvis.stl")],
    )

    anchor_urdf_meshdir(spec, tmp_path / "robot.urdf")

    assert spec.meshes[0].file == "vendor/shared/pelvis.stl"
