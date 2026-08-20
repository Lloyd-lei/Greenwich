import pytest

from alphamotion.projects import ProjectStore


def test_remove_motion_cleans_project_reference_and_bins(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("editor")
    project = store.add_media(
        project["id"],
        motions=[
            {"asset_id": "motion-a", "library_id": 7, "name": "walk"},
            {"asset_id": "motion-b", "library_id": 8, "name": "kick"},
        ],
        bin_name="Imported",
    )

    project, removed = store.remove_media(
        project["id"], kind="motion", library_id=7
    )

    assert removed == 1
    assert [item["asset_id"] for item in project["assets"]["motions"]] == [
        "motion-b"
    ]
    assert project["assets"]["bins"][0]["asset_ids"] == ["motion-b"]


def test_remove_robot_and_reject_ambiguous_request(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("editor")
    project = store.add_media(
        project["id"], bodies=[{"name": "unitree_h1"}, {"name": "booster_t1"}]
    )

    project, removed = store.remove_media(
        project["id"], kind="robot", name="unitree_h1"
    )

    assert removed == 1
    assert [item["name"] for item in project["assets"]["bodies"]] == [
        "booster_t1"
    ]
    with pytest.raises(ValueError, match="selector"):
        store.remove_media(project["id"], kind="motion")
