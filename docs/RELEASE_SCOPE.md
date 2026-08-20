# AlphaMotion Release Scope

This file is the product contract for the locked Greenwich + Equator release.
It separates shipped behavior from optional integrations and from claims that
require a downstream physics controller.

## Shipped in the core package

| Surface | Contract | Acceptance evidence |
|---|---|---|
| Models | Greenwich c1 + Equator A/B config-locked and strict-loaded | startup fails on any architecture mismatch |
| Assets | 4,096 exact native clips, root trajectories, 65k Atlas windows | `library_native=true`; fresh-download manifest includes raw codes |
| Cross embodiment | bundled descriptors, 12 attached meshes, user URDF/ZIP ingest | parse, free root, limits, 100% semantic coverage, zero-shot decode |
| Refiner | global feasible projection plus temporal branch/hold repair | global likelihood gate >= 0.70 and regional four-limb retention gate |
| Temporal editor | clips, insertion, reorder, bridges, pins, seed, temperature, draggable `n` | strict black-box timeline acceptance |
| Task-space editing | root and articulated-joint SE(3) position/rotation constraints | residual and continuity thresholds in product acceptance |
| Atlas Memory | real motion tokens, portal graph, executable jumps, graph walk | only release-approved generations enter the shared graph |
| Persistence | SQLite WAL: skeletons, motions, assets, Atlas edges, jobs | generated/uploaded assets survive process restart |
| Runtime | warm resident models, serialized GPU scheduler, persistent viewers | no core model cold load on request path |
| Delivery | editable Conda/pip package, Linux and Windows CI definition | unit suite plus browser/product acceptance commands |

## Optional, configuration-gated

- GENMO text-to-motion and video-to-SMPL run in a separately licensed external
  environment. AlphaMotion provides a
  bounded upload channel and common SMPL-22 adapter, but does not re-host or
  silently emulate third-party weights.
- New joint names use the checkpoint's Qwen3 semantic tower when installed.
  Without it, deterministic name + topology labeling still covers every joint;
  this fallback is for inspection and registration, not a replacement model
  conditioning space.
- Vendor visual meshes are not universally redistributable. Descriptor-only
  bodies remain generatable and receive a semantic skeleton preview.

## Explicit non-claims

- Viser and MP4 are kinematic previews. They do not prove contact, torque,
  balance recovery, or sim-to-real stability.
- The current runtime is a single-GPU scheduler, not a multi-node inference
  load balancer. Horizontal deployment belongs at the process/orchestrator
  layer and should not duplicate CUDA contexts inside one worker.
- Windows CI is defined in the repository; it is not considered verified until
  the remote workflow has completed on the published commit.
- GitHub and Hugging Face publication are release actions. Run them only after
  the final dirty-worktree review and credential check.

## Release commands

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/unit -q
python scripts/product_acceptance.py --base-url http://127.0.0.1:7860
python scripts/browser_acceptance.py --url http://127.0.0.1:7860
python scripts/browser_acceptance.py --url http://127.0.0.1:7860 \
  --width 390 --height 844 --layout-only
```
