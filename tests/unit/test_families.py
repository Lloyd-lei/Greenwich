from alphamotion.atlas.families import FAMILIES, family_id, family_of


def test_classify():
    assert family_of("ACCAD_walk_B10") == "walk"
    assert family_of("MartialArtsPunches") == "punch"
    assert family_of("xyz") == "other"
    assert family_of("knife_chop_1") == "other"
    assert family_of("light_hopping_stiff") == "jump"
    assert family_of("understanding_pose") == "other"
    assert FAMILIES[family_id("salsa_dance")] == "dance"
