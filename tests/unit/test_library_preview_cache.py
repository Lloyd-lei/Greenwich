from types import SimpleNamespace

from alphamotion.service.app import _library_preview_key


def _library(name="kick", asset_id="asset-kick"):
    return SimpleNamespace(
        asset_ids=[asset_id], origin_ids=[asset_id], names=[name],
        datasets=["imported_smpl"], sources=["BMLmovi"],
        source_models=["smplh"], source_genders=["female"],
        augmentations=[""], augmentation_values=[None],
        frames=lambda _index: 126,
    )


def test_hover_preview_cache_key_tracks_asset_identity():
    kick = _library()
    throw = _library(name="throw", asset_id="asset-throw")

    assert _library_preview_key(kick, 0) == _library_preview_key(kick, 0)
    assert _library_preview_key(kick, 0) != _library_preview_key(throw, 0)


def test_hover_preview_cache_key_supports_composite_metadata():
    composite = _library()
    del composite.source_models
    del composite.source_genders

    assert _library_preview_key(composite, 0)
