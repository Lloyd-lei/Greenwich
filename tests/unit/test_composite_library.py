import json

import numpy as np

from alphamotion.atlas.library import CompositeLibrary, Library


def _shard(path, name, dataset, source, token):
    path.mkdir()
    np.savez(
        path / "library.npz",
        tokens=np.full((1, 32), token, np.int32),
        bounds=np.zeros((1, 4, 256, 20), np.int8),
    )
    np.save(path / "library_codes.npy",
            np.zeros((1, 60, 256, 10), np.uint8))
    np.save(path / "library_root.npy",
            np.zeros((1, 1, 60, 3), np.float16))
    (path / "library_meta.json").write_text(json.dumps({
        "clips": [name], "window": 60, "datasets": [dataset],
        "sources": [source],
    }))
    (path / "library_root_meta.json").write_text(json.dumps({
        "bodies": ["human_smpl"],
    }))
    return Library(path / "library.npz")


def test_composite_library_keeps_stable_ids_and_dataset_filter(tmp_path):
    current = _shard(tmp_path / "current", "walk_current", "current",
                     "Curated 4096", 1)
    imported = _shard(tmp_path / "imported", "walk_imported",
                      "imported_smpl", "CMU", 2)
    library = CompositeLibrary([current, imported])

    assert len(library) == 2
    assert library.entry(0)[2] == "walk_current"
    assert library.entry(1)[2] == "walk_imported"
    result = library.search(dataset="imported_smpl")
    assert result["total"] == 1
    assert result["items"][0]["id"] == 1
    assert result["items"][0]["source"] == "CMU"
    assert library.dataset_summary() == [
        {"id": "current", "label": "Current curated", "count": 1},
        {"id": "imported_smpl", "label": "Data Studio", "count": 1},
    ]


def test_library_filters_data_studio_facets(tmp_path):
    path = tmp_path / "catalog"
    library = _shard(path, "walk", "imported_smpl", "CMU", 2)
    library.data_roles[0] = "augmented"
    library.augmentations[0] = "time"
    library.augmentation_values[0] = 1.15
    library.labels[0] = ["foot_contact"]

    item = library.search(data_role="augmented", augmentation="time",
                          label="foot_contact")["items"][0]

    assert item["data_role"] == "augmented"
    assert item["augmentation"] == "time"
    assert item["augmentation_value"] == 1.15


def test_composite_library_exposes_exact_source_rotation_sidecar(tmp_path):
    current = _shard(tmp_path / "current", "old", "current",
                     "Curated 4096", 1)
    imported_path = tmp_path / "imported"
    _shard(imported_path, "new", "imported_smpl", "BMLmovi", 2)
    exact = np.arange(60 * 22 * 6, dtype=np.float16).reshape(60, 22, 6)
    np.save(imported_path / "library_source_rot6d.npy", exact[None])
    imported = Library(imported_path / "library.npz")
    library = CompositeLibrary([current, imported])

    assert library.source_local_rot6d(0) is None
    np.testing.assert_array_equal(library.source_local_rot6d(1), exact)


def test_library_loads_full_source_sidecar_and_true_frame_count(tmp_path):
    path = tmp_path / "full"
    _shard(path, "capture", "imported_smpl", "BMLmovi", 3)
    meta = json.loads((path / "library_meta.json").read_text())
    meta.update({"source_frames": [119], "source_models": ["smplh"],
                 "source_genders": ["male"],
                 "vertical_ranges_cm": [12.5], "path_lengths_m": [1.25]})
    (path / "library_meta.json").write_text(json.dumps(meta))
    source = path / "source_clips"
    source.mkdir()
    np.savez_compressed(
        source / "000000.npz",
        local_rot6d=np.zeros((119, 22, 6), np.float16),
        root_cm=np.zeros((119, 3), np.float16),
        hand_pose=np.full((119, 90), 0.25, np.float16),
        betas=np.arange(10, dtype=np.float32), gender=np.asarray(1, np.uint8),
        model_family=np.asarray(1, np.uint8))

    library = Library(path / "library.npz")
    motion = library.source_motion(0)

    assert library.frames(0) == 119
    assert motion["gender"] == "male"
    assert motion["model_family"] == "smplh"
    assert motion["hand_pose"].shape == (119, 90)
    assert library.motion_metrics(0) == {
        "vertical_range_cm": 12.5, "path_m": 1.25}
