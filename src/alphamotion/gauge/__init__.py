"""AlphaMotion data gauge: the single convention every motion source is
repaired INTO before it may enter the stack (caches, library, Greenwich,
Equator).  Diagnose -> repair -> validate -> stamp; only what cannot be
repaired is rejected.

    contract.py   the gauge itself: versioned constants (frame, units, root,
                  canonical human rest table, fps, window)
    rest.py       per-joint rest-frame repair (any rest convention -> canonical)
    validate.py   gates + stamp
"""
from .contract import GAUGE, GAUGE_VERSION
from .rest import (RestRepair, calibrate_neutral, find_neutral_frame,
                   reframe_local, reframe_pose)
from .validate import GaugeReport, fk_invariance_cm, validate_rest

__all__ = ["GAUGE", "GAUGE_VERSION", "RestRepair", "calibrate_neutral",
           "find_neutral_frame", "reframe_local", "reframe_pose",
           "GaugeReport", "fk_invariance_cm", "validate_rest"]
