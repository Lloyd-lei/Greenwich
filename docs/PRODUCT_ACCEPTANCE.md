# Product Acceptance

AlphaMotion ships with black-box gates that exercise the HTTP product instead
of importing pipeline internals. Run them against a warm local service:

```bash
PYTHONPATH=src python -m uvicorn --factory \
  alphamotion.service.app:create_app --host 0.0.0.0 --port 7860

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/unit
python scripts/product_acceptance.py \
  --base-url http://127.0.0.1:7860 --body unitree_h1 --render
python scripts/product_acceptance.py \
  --base-url http://127.0.0.1:7860 --body unitree_h1 --perception
python scripts/browser_acceptance.py \
  --url http://127.0.0.1:7860 --width 1440 --height 1000
python scripts/browser_acceptance.py \
  --url http://127.0.0.1:7860 --width 390 --height 844
```

## Covered Contracts

- native library playback and trace shape/finite-value contracts;
- temporal clip retiming, sampled gaps, seed, temperature and token pins;
- root and non-root SE(3) constraints with task-space residual checks;
- generated-frame hold and projection-branch repair;
- optional GENMO text-to-SMPL integration, including retained world-root
  translation and QC-failed sample isolation from Atlas Memory;
- Atlas portal lookup and executable bridge generation;
- synergy and kinematic-continuity gates;
- Viser camera framing, atomic mesh/camera playback, dynamic flash detection,
  semantic body preview and responsive mobile layout;
- H.264 MP4 MIME type, byte length and direct download;
- invalid body, malformed timeline and upload-boundary rejection;
- warm-process restart recovery for completed motions and stale jobs.

## Release Thresholds

- no NaN or Inf in `q`, `rootR` or global joint positions;
- maximum global joint rotation step below 90 degrees per frame;
- generated root-glide fraction at most 10 percent;
- root speed below 8 m/s;
- SE(3) end-effector mean position error below 0.2 cm in the acceptance case;
- browser document overflow at most 1 px and Viser pixel standard deviation
  above 5, which rejects blank or misframed canvases;
- over a 12-frame playback burst (including a loop boundary), adjacent mean
  luminance delta below 28 and black-area delta below 18 percent;
- no uncaught browser runtime exception.

## Scope

These gates certify product plumbing, temporal generation/editing, mechanical
projection, kinematic continuity and visual presentation. Viser and MP4 are
kinematic previews. They do not certify contact dynamics, torque feasibility,
closed-loop recovery or sim-to-real stability; those require a physics-on WBC
rollout and must be reported separately.
