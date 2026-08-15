# AlphaMotion

**One motion, every robot. A motion space you can navigate, not just sample.**

AlphaMotion is a natively cross-embodiment motion engine: a single latent
motion space shared by humans and 20+ humanoid robots, with a constraint-native
temporal layer and a searchable index over everything it has ever seen or
generated. This is a pre-release build under active internal benchmarking.

```bash
pip install -e .            # Linux / Windows 11, Python >= 3.10
alphamotion download        # ~300 MB of weights + atlas from the hub
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
Zero-shot: the engine has never trained on your robot.

**2 · The Atlas Map.** Every motion — corpus or generated — reduces to 32
discrete *rainbow codes*. The Atlas is a fixed-capacity index (65,536 windows,
a few MB) over these codes: any band of any motion's "DNA" is a **portal** into
every other motion that passes through the same code. Motion space becomes a
graph you can search (`portals`, `knn`, `walk` in `alphamotion.atlas.search`),
wander, and jump through — including into motions the model just generated,
which register into the Atlas on completion. No other motion stack exposes its
latent space as a navigable, queryable structure.

**3 · A constraint-native editor.** The temporal layer is a bridge prior
`P(interior | start, goal, n)`: endpoints are inputs, not accidents. The
timeline editor composes library clips, Equator-generated gaps, inserted
motions and SE(3) task-space constraints — and the **time budget `n` is a
first-class dial**: the same 32 tokens render at any duration (retiming is a
model property, verified, not a resampling trick).

**4 · A generative library, not a motion zoo.** The bundled library is 4,096
family-balanced clips in 23 MB — because each entry is stored as tokens plus
four boundary frames and is *regenerated* on demand, at any length, on any
body.

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
| atlas portal precision@8 | 6.1× random |
| retiming self-consistency | 0.87 |
| synergy gate pass rate | 61 % over 3 bodies × 12 clips |

## Optional extras

- `alphamotion[labeling]` — live Qwen3 joint-name embeddings for ingesting
  URDFs with joint names outside the bundled cache.
- Perception (video → motion, text → motion) rides an external GENMO
  installation; see [docs/SDK.md](docs/SDK.md). Third-party model weights are
  downloaded from their original sources, never re-hosted here.

## Honest limitations

- Rendering (mp4/viser) needs robot meshes; bundled bodies ship *descriptors
  only* (vendor meshes are not redistributable) — attach an MJCF once, or
  ingest your own URDF, and rendering lights up.
- The synergy gate genuinely fails some body × clip combinations (that is what
  it is for); per-body pass rates are in the benchmark.
- Bridging novel endpoint pairs costs ~1.3 nats over re-sampling known ones —
  a measured extrapolation cost, tracked in the gate, on the roadmap to shrink.
- World translation is not yet part of the representation (root trajectories
  come from stride integration on the roadmap).

## License

Apache-2.0. Third-party components retain their own licenses; see
ATTRIBUTIONS.md.
