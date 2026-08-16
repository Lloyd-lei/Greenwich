# Local deployment

This checkout is pinned to Greenwich commit `f4024a7`.

```bash
./deploy/doctor.sh
./deploy/run.sh
```

The UI is served at <http://127.0.0.1:7860>.

Large environments, caches, models, and robot assets live under
`/media/sajio/New Volume/CodexDeployments/Greenwich`; the source checkout stays
on the system disk. `deploy/env.sh` contains the runtime configuration.

GENMO/GEM is installed in its own Python 3.10 environment with GEM-SMPL,
T5-3B, HMR2, ViTPose, SMPL-X and the GVHMR body-model support files. The
reference video is the official GVHMR single-person tennis example.

The local GENMO checkout contains compatibility fixes found during end-to-end
testing: canonical T5 cache routing, missing GVHMR support assets, and proper
video-only handling when no text segment is present.
