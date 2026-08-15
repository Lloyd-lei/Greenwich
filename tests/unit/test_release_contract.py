import pytest

from alphamotion.engine.equator import A_RELEASE, B_RELEASE, _validate_config
from alphamotion.engine.greenwich import (RELEASE_CONTRACT,
                                          _validate_release_config)
from alphamotion.weights import ARTIFACTS


def test_library_release_is_native_playable():
    files = set(ARTIFACTS["library"][1])
    assert {"library_codes.npy", "library_root.npy",
            "library_root_meta.json"} <= files


def test_locked_model_contracts_reject_wrong_architecture():
    _validate_release_config(dict(RELEASE_CONTRACT))
    _validate_config(dict(A_RELEASE), A_RELEASE, "Equator A")
    _validate_config(dict(B_RELEASE), B_RELEASE, "Equator B")
    bad = dict(RELEASE_CONTRACT, fsq_stages=2)
    with pytest.raises(ValueError, match="release contract"):
        _validate_release_config(bad)
