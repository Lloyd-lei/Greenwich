import torch

from alphamotion.refiner.synergy import GATE_THRESHOLD, SynergyReport


def test_report_boundary():
    r = SynergyReport(ratio=0.70, nll_original=1.0, nll_refined=1.36,
                      per_token_ratio_min=0.1, passed=0.70 >= GATE_THRESHOLD)
    assert r.passed
    d = r.as_dict()
    assert d["threshold"] == 0.70
