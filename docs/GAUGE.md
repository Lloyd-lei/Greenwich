# AlphaMotion data gauge (v1.0)

Every motion source is **diagnosed → repaired → validated → stamped** before it
may enter the stack (caches, library, Greenwich, Equator). The gauge is code:
`alphamotion/gauge/contract.py` is the single source of truth; validators read
it, adapters target it, caches record `GAUGE_VERSION`.

| Axis | Rule |
|---|---|
| World frame | right-handed, **Y-up**, forward **+Z**, **cm** |
| Root | node = pelvis (never a floor node); pose = root-relative positions in world orientation; trajectory = `Δp_t = R0ᵀ(p_t − p_0)`, first frame origin, `yaw0` stored |
| Rotations | stored local rot6d `[T,J,6]` (root slot = global root); fed to the codec as global rot6d + root-relative position / reach |
| **Rest / identity** | identity ⇔ a **neutral pose** (SMPL T-pose for SMPL-defined skeletons; otherwise the rig's bind pose or a detected standing frame), subject **facing +Z** at that pose; `rest_off` = world bone vectors at that pose |
| Time | 30 fps, index resample, 60-frame windows / stride 30, windows never cross `sequences.json` segments |

## Why the rest rule looks like this (measured 0818, release codec, AMASS → unitree_g1_29dof vs GMR GT)

| convention of the SAME motion | rot / pos error |
|---|---|
| native SMPL T-pose (training majority) | 23.2° / 6.9 cm |
| standing-frame calibration, **de-yawed** | 28.5° / 7.1 cm (accepted) |
| standing-frame calibration with the frame's yaw baked in | 43–46° / 9–10 cm |
| calibration at an arbitrary (non-neutral) frame | 85–90° |
| forcing another rig (SOMA) onto SMPL's T-pose *direction table* | worse than its neutral frame (bones→G1 22° → 32°) |

Take-aways: (1) the codec is **not** invariant to the per-joint rest convention
(codes change ~70%); (2) yaw baked into the calibration is the #1 killer; (3) a
neutral frame is what matters — T-pose vs standing costs ~5°; (4) twist is not
observable from bone directions, so a *direction table* is diagnostics only,
never a repair target.

The bones seed cache (0808 standing calibration at `body_check_001` frame 0,
facing +Z within 3°) already satisfies the rule: bones → G1 = 22.3° / 4.8 cm
against the seed's own G1 retarget.

## API

```python
from alphamotion.gauge import (calibrate_neutral, find_neutral_frame,
                               reframe_local, reframe_pose, validate_rest)
t   = find_neutral_frame(local6d, spec)                 # standing-score search
rep = calibrate_neutral(spec, global_rot_at_t, deyaw=True)
local6d_g = reframe_local(local6d, rep)                 # FK-invariant re-framing
assert validate_rest(rep.spec).ok                       # facing +Z, upright
```

Gates today: `rest_facing_yaw_deg` (|yaw| ≤ 5°), `rest_upright` (head above
root, ankle below, wrist not above head at identity), `fk_invariance_cm`.
Scope: human sources. Robot caches keep the GMR pipeline gates (FK round-trip,
data-vs-XML probe, uprightness).

Not yet built: world-frame / unit / fps / topology adapters and the stamped
store; see the plan in the session notes.
