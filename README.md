# AlphaMotion

**One motion, every robot. A motion space you can navigate, not just sample.**

AlphaMotion is a natively cross-embodiment motion engine: a single latent
motion space shared by a human skeleton and 18 humanoid robot topologies, with
a constraint-native
temporal layer and a searchable index over everything it has ever seen or
generated. This is a pre-release build under active internal benchmarking.

```bash
pip install -e .            # Linux / Windows 11, Python >= 3.10
alphamotion download        # ~1 GB: weights + Atlas + lossless motion assets
alphamotion serve           # open http://127.0.0.1:7860
```

## What makes it different

Most motion-generation models are a text box in front of a black box: you
sample, you get a human skeleton clip, and everything else — retargeting,
editing, physical feasibility — is someone else's problem. AlphaMotion is
built the other way around.

**1 · Natively cross-embodiment.** Motion is encoded once into a shared
discrete code space and decoded onto *any* registered body — bundled robots or
one you ingest yourself. Drop a URDF and the pipeline parses it, injects a
floating base, audits its joint limits, labels every joint semantically
(the same frozen text tower the codec was trained with), and configures a
per-robot refiner — automatically, with an honest report of anything wrong.
An uploaded robot is handled zero-shot: no robot-specific codec retraining is
performed during registration.

**2 · The Atlas Map.** Every motion — corpus or generated — reduces to 32
discrete *rainbow codes*. The Atlas is a fixed-capacity index (65,536 windows,
a few MB) over these codes: any band of any motion's "DNA" is a **portal** into
every other motion that passes through the same code. Motion space becomes a
graph you can search (`portals`, `knn`, `walk` in `alphamotion.atlas.search`),
wander, and jump through — including into release-approved motions the model
just generated. QC-flagged traces remain auditable but never enter the shared
graph.

**3 · A constraint-native editor.** The temporal layer is a bridge prior
`P(interior | start, goal, n)`: endpoints are inputs, not accidents. The
timeline editor composes library clips, Equator-generated gaps, inserted
motions and SE(3) task-space constraints — and the **time budget `n` is a
first-class dial**: the same 32 tokens render at any duration (retiming is a
model property, verified, not a resampling trick).

**4 · A compact editable library, not a lossy demo reel.** The bundled library
contains 4,096 clips spanning 11 labeled motion families. Its 32-token
summaries and boundary
codes drive Atlas search, bridges and retiming, while a ~600 MB nibble-packed
dual-stream code store preserves native playback exactly. The package does not
pretend that Equator can reconstruct a rotation stream it was never trained to
generate.

**5 · It grades its own homework.** Every generation passes a refiner
(conditional: it measures before it touches) and a **synergy gate** — the
refined motion, re-encoded through the codec, must retain ≥ 70 % AR-likelihood
of the original coordination pattern. Failures are flagged, not hidden. The
whole benchmark is GT-free and reproducible on your install:
`alphamotion eval`.

## Current benchmark (v0.1)

See [docs/BENCHMARK.md](docs/BENCHMARK.md). Headline numbers, all reproducible
from packaged artifacts:

| | |
|---|---|
| codec round-trip fidelity | 0.64 (chance-corrected) |
| cross-body follow score | 0.42 (pipeline floor 0.00) |
| atlas portal precision@8 | 7.1× random |
| retiming self-consistency | 0.87 |
| synergy gate pass rate | 61 % over 3 bodies × 12 clips |

## Optional extras

- `alphamotion[labeling]` — live Qwen3 joint-name embeddings for ingesting
  URDFs with joint names outside the bundled cache.
- Perception uses GENMO for both text → motion and video → world-grounded SMPL
  motion; see [docs/SDK.md](docs/SDK.md). It runs in a separate environment,
  and its third-party weights retain their upstream license.

## Honest limitations

- Twelve bundled bodies currently have verified local visual assets; the
  remaining registered topologies are analysis-only until their vendor meshes
  are attached. Uploaded URDF or zipped URDF packages are validated before
  registration.
- The synergy gate genuinely fails some body × clip combinations (that is what
  it is for); per-body pass rates are in the benchmark.
- Bridging novel endpoint pairs costs ~1.0 nat over re-sampling known ones —
  a measured extrapolation cost, tracked in the gate, on the roadmap to shrink.
- Source world-root trajectories are preserved and generated gaps receive a
  continuous boundary-velocity bridge. The spatial position head remains
  root-relative; it is not presented as a learned world-trajectory head.
- Viser and MP4 outputs are kinematic product previews, not physics or WBC
  rollouts. Physical stability requires a downstream controller evaluation.

Release acceptance and its exact commands are documented in
[docs/PRODUCT_ACCEPTANCE.md](docs/PRODUCT_ACCEPTANCE.md).
The implementation boundary and optional dependencies are recorded in
[docs/RELEASE_SCOPE.md](docs/RELEASE_SCOPE.md).
The current control-by-control usability audit and required smoke gate are in
[docs/FUNCTIONAL_AUDIT.md](docs/FUNCTIONAL_AUDIT.md).

## License

Apache-2.0. Third-party components retain their own licenses; see
ATTRIBUTIONS.md.
