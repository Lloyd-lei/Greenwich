from alphamotion.atlas.families import FAMILIES, family_id, family_of


def test_classify():
    assert family_of("ACCAD_walk_B10") == "walk"
    assert family_of("MartialArtsPunches") == "punch"
    assert family_of("xyz") == "other"
    assert FAMILIES[family_id("salsa_dance")] == "dance"
