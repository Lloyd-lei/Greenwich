# AlphaMotion functional audit and implementation ledger

Last audited: 2026-08-18

This document is the product-level source of truth for controls that are
working, conditionally available, incomplete, or intentionally deferred. A
feature is not considered usable merely because its button renders.

## Status vocabulary

- **Working**: connected to a real code path and covered by a lightweight
  check or a deterministic manual workflow.
- **Conditional**: implemented, but unavailable unless an external model,
  service, mesh, or binary is configured.
- **Partial**: the visible workflow exists but is missing a production
  behavior users are likely to expect.
- **Not implemented**: a label or familiar application convention would imply
  behavior that the product does not provide.

## Runtime model ownership

Text-to-motion and video-to-motion both use the optional, separately installed
**GENMO** adapter in `src/alphamotion/perception/genmo.py`. The AlphaMotion
process does not import GENMO; it launches the configured GENMO Python process
and consumes its SMPL artifact. This audit deliberately does not run GENMO
inference on the local 12 GB GPU.

Required configuration:

- `ALPHAMOTION_GENMO_REPO`: official GENMO checkout containing
  `scripts/demo/demo_smpl.py`.
- `ALPHAMOTION_GENMO_PYTHON`: Python executable in the GENMO environment.
- Text generation additionally requires the cached
  `genmo_reference.mp4`. Video generation does not require that reference.

`GET /api/health` exposes separate `perception_text` and `perception_video`
flags. The corresponding controls must remain disabled when a capability is
false. The dormant MoMask/GVHMR adapter files are not imported by the service
and should either be removed in a dedicated cleanup or maintained as an
explicit alternative backend; they must not silently become the default.

## Application surface audit

| Surface | Status | What works | Limitations / required follow-up |
| --- | --- | --- | --- |
| File > New Sequence | Working | Clears the current timeline through the same action as the timeline Clear control. | **AUD-001:** no unsaved-change confirmation and no persisted project document. |
| File > Upload Video for Motion | Conditional | Opens the video picker only after checking `perception_video`; uploaded duration is converted to timeline frames at the selected FPS. | Requires GENMO. There is no resumable upload or multi-person selection. |
| File > Generate / Export | Conditional | Submits the real timeline job and optionally renders MP4. | Requires a valid timeline, a usable target body, all referenced model assets, and FFmpeg for MP4. The name is not a general file export command. |
| File menu as a desktop-style project menu | Not implemented | — | **AUD-002:** Open Project, Save, Save As, project autosave, media relink, and recent projects do not exist. Do not add labels for them until storage semantics exist. |
| Edit menu | Working | Undo, redo, and delete selected share the timeline history and keyboard handlers. | History is session-only and does not include every server-side generation side effect. |
| Clip menu / tool rail | Working | Selection, one-shot razor, bridge insertion, position and rotation endpoint editing are wired. Tooltips include shortcuts. | **AUD-003:** transform edits are kinematic endpoint edits, not a physics guarantee. Bridge quality is model-dependent. |
| Sequence menu and transport | Working | Start, play/pause, bridge insertion, playhead scrubbing, and timeline-driven preview are connected. | Timeline is a single motion track; there is no audio, multilayer compositing, ripple/roll/slip editing, or snapping preference panel. |
| Timeline panel menu | Working | Clear Sequence, Reset Timeline Zoom, and Generate / Export are connected. | Before this audit the three-dot button was inert; this control must stay in the smoke test. |
| Window menu | Working | Switches among Data Studio, Motion Studio, Atlas Map, and Bodies. | Workspaces cannot yet be saved, named, or reset independently. |
| Help > Keyboard Shortcuts | Partial | Shows the active shortcut summary in the sequence status line. | **AUD-004:** replace with a searchable shortcut reference/dialog and conflict reporting. |
| Data Studio tab | Conditional | Same-origin proxy and catalog synchronization are implemented. | Requires the separately running BodyDataStudio service. A missing service returns 503; there is no offline catalog editor. |
| Project / Library | Working | Search, role/augmentation/label/source facets, batch transfer, drag to timeline/source monitor, asset detail, and cached hover preview are connected. | Dataset-count chips are informational, not fake filters. Large catalogs still need virtualization and durable bins. |
| Saved/generated motion cards | Working | Click inserts at the timeline and drag chooses the insertion point. Recent motions also insert into the timeline. | Server-side generated assets have no rename/delete/version UI. |
| Source Monitor | Working | Displays the original SMPL-X motion with independent transport and optional synchronized camera/playback. | Depends on source parameters and SMPL-X model assets being readable. |
| Program Monitor | Working | Timeline/playhead owns the displayed robot motion; empty timeline clears the viewer. | Viser connection failure produces an empty monitor; there is no reconnect button or WebGL diagnostics panel. |
| Inspector / Temporal Segment | Working | Duration, endpoint transform summary, compile settings, constraints, result/QC, and recent motions are connected. | Inspector contents are not yet extensible per tool and several advanced values are raw engineering fields. |
| Body selection | Conditional | Skeleton semantics, limits, ingestion, and preview work for registered bodies. | **AUD-005:** `descriptor only` / `no mesh` means the skeleton can participate in semantics but cannot render. Compile/render must not imply visual readiness for these bodies. |
| MP4 export | Conditional | Result links are emitted when render succeeds. | Requires a renderable body and FFmpeg/imageio-ffmpeg. Capability reporting should eventually probe the actual encoder rather than report a constant. |
| Atlas Map | Conditional | Released motions can query portals, random walk, and jump jobs. | Only QC-released motions populate the useful graph. Index-only portals cannot bridge. |
| Bodies page | Working / Conditional | Bundled body inspection and user robot ingestion are real paths. | Imported URDF/MJCF quality, mesh paths, semantic inference dependencies, and joint topology determine the result. |

## Known product risks

1. **GENMO is capability-gated, not locally validated in this audit.** Static
   contracts and failure behavior are checked; output quality, runtime, and
   VRAM use require a suitable machine.
2. **Bridge generation is not a dynamics planner.** Continuity/QC checks do not
   prove that a robot can execute the result without collision, slipping, or
   torque-limit violations.
3. **Self-collision and environment collision are not a complete release
   gate.** Existing limit and quality metrics must not be presented as a
   physics certificate.
4. **Data Studio is an external-service dependency.** AlphaMotion can proxy it,
   but cannot restore its previews or mutations if that service or its media
   roots are unavailable.
5. **Project persistence is absent.** Browser refresh loses the current edit
   sequence even though generated motions remain in the database.
6. **Legacy alternative perception code is dormant.** Keeping unused MoMask
   and GVHMR files increases maintenance ambiguity until a backend-selection
   contract or cleanup is completed.

## 2026-08-18 verification record

- Deployment environment: `65 passed` in the non-GPU pytest suite.
- Frontend: the extracted inline JavaScript passes `node --check`.
- Service: `/api/health`, `/api/library-datasets`, `/api/library-facets`,
  `/api/bodies`, and `/api/motions` returned HTTP 200 with valid payloads.
- Browser smoke: all six application menus opened with their expected real
  commands; the timeline three-dot menu opened Clear Sequence, Reset Timeline
  Zoom, and Generate / Export; no application errors were logged. The only
  browser warnings were upstream Three.js deprecation warnings for
  `THREE.Clock`.
- GENMO: capability discovery reports text and video ready for the configured
  external environment. Inference was intentionally not run on this machine,
  so output quality, runtime, and VRAM usage remain unverified here.

## Required smoke gate for future implementation

Every UI or workflow change should satisfy all applicable checks before it is
called implemented:

1. Frontend JavaScript parses with `node --check` after extracting the inline
   script.
2. Python modules compile and the non-GPU, non-slow pytest suite passes.
3. The service starts without importing GENMO or another optional heavy model.
4. `GET /api/health`, `/api/library-datasets`, `/api/library-facets`,
   `/api/bodies`, and `/api/motions` return valid payloads.
5. Empty timeline means an empty Program Monitor; adding, deleting, splitting,
   and reordering a clip keeps monitor, playhead, and timeline frame aligned.
6. Every visible menu item and icon button either performs a real action,
   opens an implemented panel, or is visibly disabled with a capability
   explanation. Inert controls are release blockers.
7. GENMO-disabled smoke: text and video controls are disabled and the rest of
   Motion Studio remains usable.
8. GENMO-enabled smoke on capable hardware: one short prompt and one short
   video produce finite `[T, 22, 6]` global rotations plus `[T, 3]` root
   translation and persist a motion asset.
9. Data-Studio-disabled smoke: the tab reports a clear 503/unavailable state
   without breaking Motion Studio.
10. Resize all panel dividers at common desktop sizes and verify source/program
    monitors do not overlap or expose the embedded Viser control sidebar.

## UI design rules

The interface follows a dense professional-editor vocabulary: flat resizable
panels, compact contextual controls, subtle neutral separators, a restrained
orange selection/accent color, and pale purple reserved for AI generation.
Inactive controls should not be orange-outlined. Hover, focus, selected, and
disabled states must remain visually distinct.

Reference behavior is based on Adobe's official documentation for
[panel resizing and docking](https://helpx.adobe.com/ca/premiere/desktop/get-started/tour-the-workspace/customize-panels.html),
[tool panels and shortcut tooltips](https://helpx.adobe.com/ca/premiere/desktop/get-started/tour-the-workspace/tools-panel-and-options-panel.html),
and [contextual panel menus](https://helpx.adobe.com/premiere/desktop/get-started/tour-the-workspace/display-panel-options-and-menu.html).
The implementation should borrow interaction principles, not Adobe trademarks
or copyrighted visual assets.
