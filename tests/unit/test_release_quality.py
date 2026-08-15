from alphamotion.service.quality import release_passed


def test_release_requires_all_three_product_gates():
    good = {"continuity": {"passed": True},
            "limb_synergy": {"passed": True}}
    assert release_passed(True, good)
    assert not release_passed(False, good)
    assert not release_passed(True, {
        **good, "limb_synergy": {"passed": False}})
    assert not release_passed(True, {
        **good, "continuity": {"passed": False}})


def test_release_reads_historical_nested_report_and_explicit_decision():
    nested = {"motion": {"continuity": {"passed": True},
                          "limb_synergy": {"passed": True}}}
    assert release_passed(True, nested)
    nested["motion"]["release_passed"] = False
    assert not release_passed(True, nested)
