# AlphaMotion SDK & Architecture (product team)

## The three towers

| tower | module | checkpoint | role |
|---|---|---|---|
| **Greenwich** | `engine/greenwich.py` | `greenwich/` (18.1M params) | spatial codec: pose ↔ 256×20 discrete codes, per frame, embodiment-conditioned |
| **Equator A3** | `engine/equator.py` | `equator_a/` (40M) | 60-frame code window ↔ 32 tokens (15,625-way), endpoint conditioner, retiming decoder |
| **Equator B3** | `engine/equator.py` | `equator_b/` (12.9M) | masked token prior P(interior \| start, goal, n): bridging, pins, AR likelihood |

Data flow of every generation:

```
segments (library tokens / bridges / perception) 
  → codes [T,256,20] → Greenwich.decode(target) → Refiner (conditional)
  → Synergy gate (≥0.70 likelihood ratio) → four-limb + continuity QC
  → MotionTrace (.npz) → DB row (family/duration/source)
  → [release-approved only: Atlas registration + edges] → [mp4]
```

## SDK quickstart

```python
from alphamotion.engine.greenwich import Greenwich
from alphamotion.engine.equator import Equator
from alphamotion.atlas.library import load_default as load_library
from alphamotion.atlas.search import load_default as load_atlas
from alphamotion.embodiment import registry

gw, eq = Greenwich.load(), Equator.load()
lib, atlas = load_library(), load_atlas()

# play a library clip on a robot, at its native duration
tok, bounds, name, family = lib.entry(3)
codes = torch.from_numpy(lib.raw_codes(3)).to(eq.device)
emb   = registry.load("unitree_h1")
rot6d = gw.decode(codes, emb.spec, emb.dof)          # [60, J, 6] global

# Retiming preserves the raw rotation stream by lattice interpolation; the
# service's timeline editor performs this complete operation automatically.
from alphamotion.engine.timeline import interpolate_lattice
ep = eq.endpoints_from_codes(torch.from_numpy(bounds))
rot_codes = interpolate_lattice(codes[:, 128:], 120)
codes_2x = eq.detokenize(torch.from_numpy(tok).to(eq.device), ep, 120,
                        boundary_codes=torch.stack([codes[0], codes[-1]]),
                        rot_codes=rot_codes)

# bridge two clips (the editor's gap)
codes_b = eq.detokenize(eq.sample_tokens(ep_gap, 45), ep_gap, 45)

# atlas:搜索接口 (owner-required utils surface)
atlas.portals(tokens, slot=16, k=8)   # windows sharing this rainbow code
atlas.knn(tokens, k=8)                # whole-motion neighbours
atlas.walk(seed_window, steps=6)      # graph wander
```

`utils/metrics.py` carries every eval metric (follow, amplitude, jerk, limit,
arm QC, re-encode fidelity); `utils/eval_gate.py` is the release gate.

## Service

`alphamotion serve` → FastAPI on :7860, frontend at `/`.

- Warm pool (`service/pool.py`): Greenwich+A3+B3+atlas resident from startup,
  one serialized GPU thread, async gateway — request path never cold-loads.
- Jobs persist in SQLite (`service/db.py`, WAL). Tables: `skeletons`,
  `motions` (family / duration_s / **source** — the axis that accumulates user
  data for future training), `assets`, `atlas_edges`, `jobs`.
- Key endpoints: `POST /api/jobs/play`, `POST /api/jobs/timeline`
  (segments: `library|gap|prompt|video`, per-segment `n`, `pins`, plus `se3`
  constraints), `POST /api/bodies/ingest` (URDF upload), `GET
  /api/atlas/portals/{motion_id}`, `POST /api/assets/video`, `POST
  /api/bodies/{name}/preview`, `GET /api/results/{file}`.

## URDF ingest contract

`embodiment/urdf_ingest.ingest(path)`:
1. `MjSpec.from_file` + injected freejoint (URDF has no floating base), meshdir
   anchored absolute; 2. descriptor via the exact bundled-robot pipeline; 3.
   limit census (unlimited / zero-span / non-hinge — reported, not hidden); 4.
   Qwen3 semantic labels (+ complete topology/name fallback) — **the encoder
   choice is baked into the codec checkpoint; do not substitute it for model
   conditioning**; 5. refiner config.
Registered bodies decode zero-shot.

## Refiner + synergy gate

`refiner/refine.py` is deliberately conditional — it measures before touching
(unconditional projection halved likelihood on a feasible body; blanket wrist
smoothing destroyed tracked motion). `refiner/synergy.py` defines the 70 %
gate: likelihood ratio with BOTH sides through the same re-encode channel (the
codec's intrinsic round-trip cost is a control, not a charge).

## Perception (optional)

`perception/genmo.py` shells into an external GENMO env:
`ALPHAMOTION_GENMO_PYTHON`, `ALPHAMOTION_GENMO_REPO`; text mode additionally
needs one cached reference video (`~/.cache/alphamotion/genmo_reference.mp4`)
because GENMO requires a video for camera intrinsics. Its weights come from
the official source on first run. Unconfigured = actionable error, never a
silent zero.

## Cross-platform notes

GL backend chosen before MuJoCo import (EGL on Linux,
WGL default on Windows); mp4 via imageio-ffmpeg's bundled binary; HF cache
symlinks disabled for Windows; subprocesses always `sys.executable`-resolved;
`tests/unit/test_paths_portability.py` blocks any machine-local literal from
entering the package.

## Release procedure

1. `pytest tests/unit` all green on Linux and Windows CI.
2. `alphamotion eval` all green → refresh `docs/BENCHMARK.md`.
3. `python scripts/export_weights.py && python scripts/build_atlas.py &&
   python scripts/build_library.py` (research machine).
4. `python scripts/upload_hf.py` then `git push origin main`.
