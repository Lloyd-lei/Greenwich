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

Motion perception runs in a separate Python environment. Both Add a Motion
(text) and Upload Video to Generate use GENMO and return the same global
SMPL-22 rotations and root-trajectory contract consumed by the Greenwich
retargeter.

The deployment expects an official GENMO checkout at `sources/GENMO` and its
environment at `envs/genmo`. Configure the exact paths through
`ALPHAMOTION_GENMO_REPO` and `ALPHAMOTION_GENMO_PYTHON`. Text generation also
requires a short camera/reference clip at the AlphaMotion cache path
`genmo_reference.mp4`. AlphaMotion does not bundle or re-host GENMO weights.
