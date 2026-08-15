"""Single release-eligibility contract shared by persistence and the UI."""
from __future__ import annotations

from collections.abc import Mapping


def release_passed(gate_passed: bool | None,
                   qc: Mapping | None) -> bool:
    """Return whether a motion may enter the public Atlas.

    Historical rows store the motion report under ``qc.motion`` while the
    live pipeline holds that report directly. Supporting both shapes keeps
    restarts deterministic without a destructive database migration.
    """
    if not gate_passed or not isinstance(qc, Mapping):
        return False
    motion = qc.get("motion", qc)
    if not isinstance(motion, Mapping):
        return False
    if "release_passed" in motion:
        return bool(motion["release_passed"])
    continuity = motion.get("continuity", {})
    limb = motion.get("limb_synergy", {})
    return bool(isinstance(continuity, Mapping)
                and isinstance(limb, Mapping)
                and continuity.get("passed")
                and limb.get("passed"))
