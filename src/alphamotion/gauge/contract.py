"""The AlphaMotion gauge — data, not prose.  Everything below is the single
source of truth; validators read it, adapters target it, caches record
GAUGE_VERSION and readers refuse anything else.

Bump GAUGE_VERSION whenever any constant here changes: every cache built under
the old gauge becomes unreadable on purpose.
"""
from __future__ import annotations

from dataclasses import dataclass, field

GAUGE_VERSION = "1.0"


@dataclass(frozen=True)
class WorldFrame:
    handedness: str = "right"
    up: str = "+Y"
    forward: str = "+Z"        # SMPL T-pose faces +Z (toes / chest lean +Z)
    units: str = "cm"
    floor: str = "y=0 (feet-on-floor not enforced; root trajectory carries height)"


@dataclass(frozen=True)
class RootConvention:
    node: str = "Pelvis"       # never a floor / reference node
    pose_frame: str = "root-relative positions, world orientation (root at origin)"
    trajectory: str = ("delta_t = R0^T (p_t - p_0), first frame = origin; "
                       "yaw0 stored alongside so segments re-anchor with rotation")


@dataclass(frozen=True)
class RotationConvention:
    stored: str = "local rot6d [T,J,6] (root slot = global root rotation)"
    fed: str = "global rot6d + root-relative position / reach  ([T,J,9])"
    quaternion_order: str = "wxyz (source adapters convert; never stored)"
    rest_identity: str = ("identity global rotation <=> a NEUTRAL pose (SMPL "
                          "T-pose for SMPL-defined skeletons, else rig bind pose "
                          "or a detected standing frame), subject FACING +Z there; "
                          "rest_off = world bone vectors at that pose")


@dataclass(frozen=True)
class TimeConvention:
    fps: int = 30
    resample: str = "index round-to-nearest (no interpolation of rotations)"
    window: int = 60
    stride: int = 30
    segment_table: str = "sequences.json {name,start,len}; windows never cross segments"


# Canonical human topology (SMPL-22 as used by every human cache the release
# codec was trained on) and its T-pose bone DIRECTIONS (reference/diagnostic
# only: unit vector parent->joint at SMPL T-pose).  NOT a repair target — other
# rigs define joints differently (measured 0818: forcing SOMA onto this table
# made bones->G1 worse, 22 -> 32 deg).
HUMAN_JOINTS = ["Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee",
                "Spine", "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe",
                "Neck", "L_Thorax", "R_Thorax", "Head", "L_Shoulder",
                "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"]
HUMAN_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                 16, 17, 18, 19]
HUMAN_TPOSE_DIRS = {
    "Pelvis":     (0.000, 1.000, 0.000),
    "L_Hip":      (0.604, -0.795, -0.059),
    "R_Hip":      (-0.598, -0.800, -0.038),
    "Torso":      (-0.023, 0.971, -0.238),
    "L_Knee":     (0.091, -0.996, -0.012),
    "R_Knee":     (-0.100, -0.995, -0.023),
    "Spine":      (0.041, 0.999, 0.008),
    "L_Ankle":    (-0.034, -0.993, -0.109),
    "R_Ankle":    (0.039, -0.994, -0.106),
    "Chest":      (0.025, 0.901, 0.433),
    "L_Toe":      (0.196, -0.415, 0.888),
    "R_Toe":      (-0.188, -0.357, 0.915),
    "Neck":       (-0.013, 0.980, -0.196),
    "L_Thorax":   (0.529, 0.817, -0.229),
    "R_Thorax":   (-0.548, 0.796, -0.259),
    "Head":       (0.062, 0.783, 0.619),
    "L_Shoulder": (0.944, 0.316, -0.092),
    "R_Shoulder": (-0.943, 0.320, -0.090),
    "L_Elbow":    (0.993, -0.049, -0.105),
    "R_Elbow":    (-0.995, -0.052, -0.084),
    "L_Wrist":    (0.999, 0.036, -0.005),
    "R_Wrist":    (-0.999, 0.030, -0.022),
}


@dataclass(frozen=True)
class Gauge:
    version: str = GAUGE_VERSION
    world: WorldFrame = field(default_factory=WorldFrame)
    root: RootConvention = field(default_factory=RootConvention)
    rotation: RotationConvention = field(default_factory=RotationConvention)
    time: TimeConvention = field(default_factory=TimeConvention)
    human_joints: tuple = tuple(HUMAN_JOINTS)
    human_parents: tuple = tuple(HUMAN_PARENTS)
    # gates (validate.py reads these)
    fk_roundtrip_cm: float = 0.1
    rest_yaw_tol_deg: float = 5.0      # facing of the neutral pose vs +Z


GAUGE = Gauge()
