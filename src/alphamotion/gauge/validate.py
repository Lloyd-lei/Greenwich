"""Gauge gates.  Each check returns numbers; the report decides pass/fail
against the contract's tolerances and records what was repaired."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..engine.nets.rotations import SkeletonSpec, fk_global
from .contract import GAUGE


@dataclass
class GaugeReport:
    gauge_version: str = GAUGE.version
    checks: dict = field(default_factory=dict)
    repairs: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def add(self, name, value, ok, note=""):
        self.checks[name] = {"value": value, "ok": bool(ok), "note": note}
        if not ok:
            self.failures.append(name)


def _find(spec, *keys):
    names = [n.lower() for n in spec.joint_names]
    for k in keys:
        for i, n in enumerate(names):
            if k in n:
                return i
    return None


def rest_facing_yaw_deg(spec: SkeletonSpec) -> float | None:
    """Yaw of the rest pose's facing (+Z when 0), read from the hip lateral
    axis: forward = lateral(L-R) x up.  None if hips are not identifiable."""
    iL, iR = _find(spec, "l_hip", "lefthip", "left_hip", "leftupleg"), \
             _find(spec, "r_hip", "righthip", "right_hip", "rightupleg")
    if iL is None or iR is None:
        return None
    lat = spec.rest_offsets[iL] - spec.rest_offsets[iR]
    lat[1] = 0.0
    if np.linalg.norm(lat) < 1e-6:
        return None
    lat = lat / np.linalg.norm(lat)
    fwd = np.cross(lat, [0.0, 1.0, 0.0])
    return float(np.degrees(np.arctan2(fwd[0], fwd[2])))


def rest_upright(spec: SkeletonSpec) -> dict:
    """Signs the neutral pose must have in the gauge: spine up, legs down,
    arms not up (T-pose or hanging both pass)."""
    _, p = fk_global(torch.zeros(1, spec.J, 6).index_fill_(
        -1, torch.tensor([0, 4]), 1.0), spec)          # identity rot6d
    p = p[0].numpy()
    out = {}
    iH, iA = _find(spec, "head", "neck"), _find(spec, "l_ankle", "leftfoot", "left_ankle")
    iW = _find(spec, "l_wrist", "lefthand", "left_wrist")
    if iH is not None: out["head_above_root"] = bool(p[iH, 1] > 0)
    if iA is not None: out["ankle_below_root"] = bool(p[iA, 1] < 0)
    if iW is not None and iH is not None: out["wrist_not_above_head"] = bool(p[iW, 1] < p[iH, 1] + 1e-3)
    return out


def fk_invariance_cm(local_a, spec_a, local_b, spec_b) -> float:
    """Max joint displacement between two (local6d, spec) expressions of the
    same motion — must be ~0 after any re-framing."""
    _, pa = fk_global(local_a.float(), spec_a)
    _, pb = fk_global(local_b.float(), spec_b)
    return float((pa - pb).norm(dim=-1).max())


def validate_rest(spec: SkeletonSpec, report: GaugeReport | None = None) -> GaugeReport:
    rep = report or GaugeReport()
    yaw = rest_facing_yaw_deg(spec)
    rep.add("rest_facing_yaw_deg", yaw,
            yaw is None or abs(yaw) <= GAUGE.rest_yaw_tol_deg,
            "neutral pose must face +Z (yaw baked into calibration is the #1 killer)")
    up = rest_upright(spec)
    rep.add("rest_upright", up, all(up.values()),
            "identity pose must be a neutral standing/T pose")
    return rep
