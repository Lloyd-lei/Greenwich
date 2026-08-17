"""AlphaMotion service gateway.

Warm pool + async job runner + SQLite persistence. Every generated motion goes
through: assemble codes -> decode on target -> refine -> synergy gate -> QC ->
trace asset -> DB row -> atlas registration + edges. Jobs survive restarts in
the DB; results are files under data_dir()/results.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..atlas.families import FAMILIES, family_id, family_of
from ..config import setup_gl_backend
from ..paths import data_dir, results_dir
from .db import Asset, AtlasEdge, Job, Motion, Skeleton, session
from .quality import release_passed

# The warm pool and embodiment registry may import MuJoCo long before an MP4
# is requested. Select the platform backend before either module can do so;
# setting MUJOCO_GL inside the renderer is already too late in a warm service.
setup_gl_backend()

from .pool import POOL
from .schemas import JumpRequest, PlayRequest, Segment, TimelineRequest

FRONTEND = Path(__file__).parent.parent / "assets" / "frontend"
MODEL_FRAME_BATCH = 256

def create_app() -> FastAPI:
    app = FastAPI(title="AlphaMotion", version="0.1.0")
    state: dict = {"library": None, "warm": None,
                   "library_previews": OrderedDict(),
                   "timeline_previews": OrderedDict(),
                   "edit_previews": OrderedDict(),
                   "source_skins": OrderedDict(),
                   "source_codes": OrderedDict(),
                   "sync_comparison": False,
                   "viewer_revision": 0}

    def _decode_full_batched(codes, spec, dof):
        """Decode long clips without materialising huge model activations."""
        rot, pos = [], []
        for chunk in codes.split(MODEL_FRAME_BATCH):
            chunk_rot, chunk_pos = POOL.greenwich.decode_full(
                chunk, spec, dof)
            rot.append(chunk_rot.detach())
            pos.append(chunk_pos.detach())
        return torch.cat(rot), torch.cat(pos)

    def _project_batched(global_rotations, spec, dof_t, rest_t,
                         method="global", lm_iters=20):
        """Project independent frames in bounded GPU batches."""
        from ..engine import constraints as MP
        feasible, positions, angles = [], [], []
        for chunk in global_rotations.split(MODEL_FRAME_BATCH):
            f, p, q = MP.project(
                chunk, spec, dof_t, rest=rest_t,
                method=method, lm_iters=lm_iters)
            feasible.append(f.detach())
            positions.append(p.detach())
            angles.append(q.detach())
        return torch.cat(feasible), torch.cat(positions), torch.cat(angles)

    @app.on_event("startup")
    async def _startup():
        from ..atlas.library import load_default as load_library
        from ..config import CONFIG
        from ..viz.live import LiveViewer
        state["warm"] = POOL.warm()
        state["library"] = load_library()
        # An empty editor must also have an idle player. Motions are started
        # explicitly by the Studio transport after a timeline is populated.
        state["live"] = LiveViewer(CONFIG.viewer_ports[0], autoplay=False)
        state["body_live"] = LiveViewer(CONFIG.viewer_ports[1])
        def source_finished():
            if state.get("sync_comparison"):
                state["live"].set_transport(play=False)
        state["source_live"] = LiveViewer(
            CONFIG.viewer_ports[2], loop_playback=False, autoplay=False,
            on_playback_end=source_finished)
        state["live"].link_camera_peer(state["source_live"])
        state["source_live"].link_camera_peer(state["live"])
        state["restored_motion"] = None

        # Async jobs cannot resume across a process restart. Leaving their
        # durable rows in "running" made the product poll forever, so close
        # them explicitly and preserve a truthful reason for the operator.
        atlas = POOL.atlas
        dynamic_capacity = max(atlas.capacity - atlas.frozen, 0)
        with session() as s:
            stale = s.query(Job).filter(Job.status.in_(("queued", "running")))
            for job in stale:
                job.status = "failed"
                job.error = "service restarted before this job completed"
                job.finished_at = dt.datetime.utcnow()
            candidates = s.query(Motion).order_by(Motion.id.desc()).limit(
                max(dynamic_capacity * 4, dynamic_capacity)).all()
            motions = [m for m in candidates
                       if release_passed(m.gate_passed, m.qc)]\
                [:dynamic_capacity][::-1]
            s.commit()

        # Rehydrate generated entries into the in-memory Atlas after a fresh
        # process restart. The frozen corpus comes from weights; generated
        # motions are durable in SQLite and must remain searchable too.
        if len(atlas.tokens) == atlas.frozen:
            for motion in motions:
                tokens = np.asarray(motion.tokens or [], np.int32)
                if tokens.shape == (32,):
                    atlas.add(tokens, motion.title, family_id(motion.family))

    # ------------------------------------------------------------- meta -----
    @app.get("/api/health")
    def health():
        from ..embodiment import registry
        from ..perception.genmo import status as perception_status
        warm = {**state["warm"], "atlas_windows": len(POOL.atlas.tokens)}
        perception = perception_status()
        return {"ok": POOL.greenwich is not None, "warm": warm,
                "library": len(state["library"]) if state["library"] else 0,
                "library_native": bool(state["library"] and
                                       state["library"].has_raw),
                "viewer": state["live"].url if state.get("live") else None,
                "body_viewer": state["body_live"].url
                if state.get("body_live") else None,
                "source_viewer": state["source_live"].url
                if state.get("source_live") else None,
                "restored_motion": state.get("restored_motion"),
                "capabilities": {
                    "perception": perception["ready"],
                    "perception_detail": perception,
                    "temporal": True, "token_pins": True, "se3": True,
                    "mp4": True,
                    "render_bodies": sorted(registry.mesh_map())}}

    @app.get("/api/bodies")
    def bodies():
        from ..embodiment import registry
        out = []
        renderable = registry.mesh_map()
        for n in registry.bundled_names():
            out.append({"name": n, "source": "bundled",
                        "renderable": n in renderable})
        for n in registry.user_names():
            out.append({"name": n, "source": "user",
                        "renderable": bool(registry.load(n).xml)})
        return {"bodies": out}

    @app.get("/api/bodies/{name}")
    def body_detail(name: str):
        from ..embodiment import registry
        from ..engine.spatial import rest_positions
        try:
            emb = registry.load(name)
        except KeyError as e:
            raise HTTPException(404, str(e))
        meta = {}
        with session() as s:
            row = s.query(Skeleton).filter_by(name=name).first()
            if row:
                meta = {"sem_labels": row.sem_labels,
                        "limit_report": row.limit_report}
        if not (meta.get("sem_labels") or {}).get("per_joint"):
            from ..embodiment.semantics_map import deterministic_part_labels
            from ..engine.spatial import key_joints
            labels = deterministic_part_labels(emb.spec)
            idx, names, tags = key_joints(emb.spec)
            meta["sem_labels"] = {
                "per_joint": labels,
                "key_joints": dict(zip(tags, names)),
                "method": "topology+name",
                "coverage": len(labels) / max(emb.spec.J, 1),
            }
        rest_pos = rest_positions(emb.spec)
        # ``SkeletonSpec.height`` is the sum of all bone lengths, not body
        # stature. Canonical skeletons are Y-up; their vertical rest extent is
        # the physically meaningful value to show in the product.
        height_cm = float(np.ptp(rest_pos[:, 1]))
        return {"name": name, "joints": emb.spec.J,
                "joint_names": list(emb.spec.joint_names),
                "height_cm": round(height_cm, 1),
                "source": emb.source, "renderable": bool(emb.xml), **meta}

    @app.post("/api/bodies/{name}/preview")
    def preview_body(name: str):
        from ..embodiment import registry
        try:
            emb = registry.load(name)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        detail = body_detail(name)
        state["body_live"].set_body_preview(emb, detail.get("sem_labels"))
        return {"ok": True, "body": name, "joints": emb.spec.J,
                "renderable": bool(emb.xml)}

    @app.get("/api/library")
    def library(q: str = Query(default="", max_length=200),
                family: str = Query(default="", max_length=32),
                dataset: str = Query(default="", max_length=64),
                offset: int = Query(default=0, ge=0),
                limit: int = Query(default=24, ge=1, le=100)):
        return state["library"].search(
            q=q, family=family, dataset=dataset, offset=offset, limit=limit)

    @app.get("/api/library-datasets")
    def library_datasets():
        return {"datasets": state["library"].dataset_summary()}

    @app.get("/api/library/{library_id}/preview")
    async def library_preview(library_id: int):
        """Metadata for the neutral SMPL-X hover preview."""
        lib = state["library"]
        if library_id < 0 or library_id >= len(lib):
            raise HTTPException(404, "no such library clip")
        return {"id": int(library_id), "name": lib.names[library_id],
                "family": lib.families[library_id],
                "dataset": lib.datasets[library_id],
                "source": lib.sources[library_id],
                "frames": lib.frames(library_id), "fps": 30.0,
                "skin": ("neutral SMPL-X · exact source parameters"
                         if lib.source_motion(library_id) is not None else
                         "neutral SMPL-X · decoded 60-frame window"),
                "url": f"/api/library/{library_id}/preview.webp"}

    def _source_skin(library_id: int):
        lib = state["library"]
        cache = state["source_skins"]
        if library_id in cache:
            cache.move_to_end(library_id)
            return cache[library_id]
        from ..viz.smplx_skin import skin_global_rot6d
        spec, dof, _rest = POOL.human
        motion = lib.source_motion(library_id)
        if motion is None:
            codes = torch.from_numpy(
                lib.raw_codes(library_id).astype(np.int64)).to(POOL.device)
            rot = POOL.greenwich.decode(codes, spec, dof)
            root = lib.root_delta(library_id, "human_smpl")
            hands = betas = None
            source_kind = "decoded-window"
        else:
            from ..engine.spatial import build_global
            rot, _pos, _reach = build_global(
                torch.from_numpy(motion["local_rot6d"]), spec, "cpu")
            root = motion["root_cm"]
            hands, betas = motion["hand_pose"], motion["betas"]
            source_kind = "exact-source"
        vertices, faces = skin_global_rot6d(
            rot.cpu().numpy(), spec.parents, root_cm=root,
            hand_pose=hands, betas=betas, device=POOL.device)
        value = (vertices, faces, lib.names[library_id], 30.0, source_kind)
        cache[library_id] = value
        cache.move_to_end(library_id)
        # Long-form clips can occupy hundreds of MB even in half precision.
        # Keep only the active source instead of accumulating stale previews.
        while len(cache) > 1:
            cache.popitem(last=False)
        return value

    @app.get("/api/library/{library_id}/preview.webp")
    async def library_skin_preview(library_id: int):
        """Fixed-camera SMPL-X skin animation at 1x with a 0.5s loop hold."""
        lib = state["library"]
        if library_id < 0 or library_id >= len(lib):
            raise HTTPException(404, "no such library clip")
        from ..paths import cache_dir
        # v3 includes full source duration, source betas and hand articulation.
        path = cache_dir() / "smplx_previews" / f"v3-{library_id}.webp"
        if not path.is_file():
            def work():
                from ..viz.smplx_skin import animated_preview_webp
                vertices, faces, _name, fps, _kind = _source_skin(library_id)
                data = animated_preview_webp(vertices, faces, fps=fps)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            await POOL.run(work)
        return FileResponse(path, media_type="image/webp",
                            headers={"cache-control": "public, max-age=31536000"})

    @app.get("/api/families")
    def families():
        return {"families": FAMILIES}

    @app.get("/api/viewer/transport")
    def viewer_transport():
        return state["live"].transport_state()

    @app.post("/api/viewer/transport")
    def set_viewer_transport(
            frame: int | None = Query(default=None, ge=0),
            play: bool | None = Query(default=None),
            follow: bool | None = Query(default=None),
            speed: float | None = Query(default=None, ge=0.25, le=2.0)):
        result = state["live"].set_transport(
            frame=frame, play=play, follow=follow, speed=speed)
        # Follow is a camera behaviour rather than a shared clock.  Mirror it
        # to the source viewport only while comparison synchronization is on.
        if follow is not None and state["sync_comparison"]:
            state["source_live"].set_transport(follow=follow)
        return result

    @app.post("/api/viewer/clear")
    def clear_viewer():
        # Invalidate previews that may still be decoding in the worker. They
        # must not repopulate the canvas after the final timeline block was
        # removed.
        state["viewer_revision"] += 1
        return state["live"].clear_motion()

    @app.post("/api/source/preview/library")
    async def preview_source_library(library_id: int = Query(ge=0)):
        lib = state["library"]
        _require_library_id(lib, library_id)
        def work():
            vertices, faces, name, fps, kind = _source_skin(library_id)
            state["source_live"].set_smplx_skin(
                vertices, faces, name, fps=fps)
            if state["sync_comparison"]:
                state["source_live"].set_transport(
                    follow=state["live"].transport_state()["follow"])
            return {**state["source_live"].transport_state(),
                    "name": name, "skin": "SMPL-X neutral",
                    "source_kind": kind}
        return await POOL.run(work)

    @app.get("/api/source/transport")
    def source_transport():
        return state["source_live"].transport_state()

    @app.post("/api/source/transport")
    def set_source_transport(
            frame: int | None = Query(default=None, ge=0),
            play: bool | None = Query(default=None),
            speed: float | None = Query(default=None, ge=0.25, le=2.0),
            sync: bool = Query(default=False)):
        state["sync_comparison"] = bool(sync)
        result = state["source_live"].set_transport(
            frame=frame, play=play, speed=speed)
        # Synchronized playback starts/pauses both clocks where they already
        # are. Deliberately do not seek either side.
        if play is not None and sync:
            state["live"].set_transport(play=play)
        return result

    @app.post("/api/comparison/sync")
    def set_comparison_sync(enabled: bool = Query(default=False)):
        state["sync_comparison"] = bool(enabled)
        target, source = state["live"], state["source_live"]
        target.set_camera_linked(enabled, reset=False)
        source.set_camera_linked(enabled, reset=False)
        # Enabling synchronization adopts the main viewport's current Follow
        # setting without changing either camera pose.  Once unlinked, leave
        # the main setting alone and return the source to its fixed camera.
        source.set_transport(
            follow=target.transport_state()["follow"] if enabled else False)
        return {"enabled": bool(enabled)}

    @app.post("/api/comparison/reset-camera")
    def reset_comparison_camera():
        """Reset the main view and, while synchronized, the source view."""
        state["live"].reset_camera()
        both = bool(state["sync_comparison"])
        if both:
            state["source_live"].reset_camera()
        return {"reset": "both" if both else "target"}

    # ------------------------------------------------------------- jobs -----
    def _submit(kind: str, request: dict, runner) -> str:
        job_id = uuid.uuid4().hex[:12]
        with session() as s:
            s.add(Job(id=job_id, kind=kind, request=request))
            s.commit()
        asyncio.create_task(_run_job(job_id, runner))
        return job_id

    async def _run_job(job_id: str, runner):
        def upd(**kw):
            with session() as s:
                j = s.get(Job, job_id)
                for k, v in kw.items():
                    setattr(j, k, v)
                s.commit()
        upd(status="running")
        try:
            result = await runner(job_id)
            upd(status="done", result=result,
                motion_id=result.get("motion_id"),
                finished_at=dt.datetime.utcnow())
        except Exception as exc:  # noqa: BLE001
            upd(status="failed", error=str(exc)[:2000],
                finished_at=dt.datetime.utcnow())

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        with session() as s:
            j = s.get(Job, job_id)
            if not j:
                raise HTTPException(404, "no such job")
            return {"id": j.id, "kind": j.kind, "status": j.status,
                    "result": j.result, "error": j.error,
                    "motion_id": j.motion_id}

    @app.get("/api/results/{name}")
    def result_file(name: str):
        p = (results_dir() / name).resolve()
        if not p.is_relative_to(results_dir().resolve()) or not p.is_file():
            raise HTTPException(404, "no such result")
        media = "video/mp4" if p.suffix.lower() == ".mp4" else None
        return FileResponse(p, media_type=media, filename=p.name)

    @app.post("/api/assets/video", status_code=201)
    async def upload_video(file: UploadFile = File(...)):
        """Store a perception input under a server-controlled path.

        Timeline requests carry only the returned opaque filename. They can
        never ask the GENMO subprocess to read an arbitrary host file.
        """
        suffix = Path(file.filename or "clip.mp4").suffix.lower()
        allowed = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
        if suffix not in allowed:
            raise HTTPException(422, "video must be MP4, MOV, MKV, WEBM, or AVI")
        token = f"{uuid.uuid4().hex}{suffix}"
        root = data_dir() / "uploads" / "videos"
        root.mkdir(parents=True, exist_ok=True)
        path = root / token
        total = 0
        try:
            with path.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > 250 * 1024 * 1024:
                        raise HTTPException(413,
                                            "video upload exceeds 250 MiB")
                    handle.write(chunk)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        if total == 0:
            path.unlink(missing_ok=True)
            raise HTTPException(422, "video upload is empty")
        with session() as s:
            s.add(Asset(kind="upload", path=str(path), bytes=total))
            s.commit()
        return {"asset": token, "name": Path(file.filename or token).name,
                "bytes": total}

    # ---------------------------------------------------------- pipeline ----
    def _library_playback_codes(library_id: int):
        """Exact full imported source, or the native curated 60f window."""
        lib = state["library"]
        motion = lib.source_motion(library_id)
        if motion is None:
            return torch.from_numpy(
                lib.raw_codes(library_id).astype(np.int64)).to(POOL.device)
        cache = state["source_codes"]
        if library_id in cache:
            cache.move_to_end(library_id)
            return cache[library_id]
        spec, dof, _rest = POOL.human
        local = torch.from_numpy(motion["local_rot6d"]).to(POOL.device)
        pose9, _ = POOL.greenwich.pose9(local, spec, is_global=False)
        codes = torch.cat([
            POOL.greenwich.encode(chunk, spec, dof).detach()
            for chunk in pose9.split(MODEL_FRAME_BATCH)
        ])
        cache[library_id] = codes
        cache.move_to_end(library_id)
        while len(cache) > 3:
            cache.popitem(last=False)
        return codes

    def _segment_codes(seg: Segment, eq, lib, target_body: str | None = None):
        """One library segment -> codes, name, family, and 3-D root path.

        Native playback is bit-faithful. Temporal edits regenerate the pose
        stream through A3 and continuously interpolate the ordered FSQ
        rotation stream; no nearest-neighbour frame duplication is allowed.
        The root trajectory is DATA passthrough (owner design: first frame =
        origin), never inferred."""
        if seg.kind == "library":
            tok, bounds, name, fam = lib.entry(seg.library_id)
            raw = _library_playback_codes(seg.library_id)
            base_n = seg.source_frames or seg.n
            root = None
            if target_body:
                from ..embodiment import registry
                from ..engine.spatial import rest_positions
                target = registry.load(target_body)
                hspec, _hdof, _hrest = POOL.human
                body_reach = float(np.linalg.norm(
                    rest_positions(target.spec), axis=-1).max())
                human_reach = float(np.linalg.norm(
                    rest_positions(hspec), axis=-1).max())
                root = lib.playback_root_delta(
                    seg.library_id, target_body, body_reach, human_reach)
            if root is not None and base_n != len(root):
                from ..engine.timeline import resample_continuous
                root = resample_continuous(root, base_n)
            if base_n == raw.shape[0] and not seg.pins:
                codes = raw.to(eq.device)
            else:
                from ..engine.timeline import interpolate_lattice
                tokens = torch.from_numpy(tok).long().to(eq.device)
                for slot, val in (seg.pins or {}).items():
                    tokens[int(slot)] = int(val)
                ep = eq.endpoints_from_codes(torch.from_numpy(bounds))
                rot_codes = interpolate_lattice(raw[:, 128:], base_n)
                boundary = torch.stack([raw[0], raw[-1]])
                codes = eq.detokenize(tokens, ep, base_n,
                                      boundary_codes=boundary,
                                      rot_codes=rot_codes)

            # Non-destructive timeline split. Generate the parent material once,
            # then take the exact requested interval. If the user subsequently
            # changes this half's time budget, tokenize and retime only the cut
            # material instead of replaying the complete Library asset.
            if seg.source_frames is not None:
                start = seg.source_start
                end = seg.source_end or seg.source_frames
                codes = codes[start:end]
                if root is not None:
                    root = np.asarray(root)[start:end]
            if seg.n != len(codes):
                from ..engine.timeline import (interpolate_lattice,
                                               resample_continuous)
                source = codes
                tokens, ep = eq.tokenize(source)
                rot_codes = interpolate_lattice(source[:, 128:], seg.n)
                boundary = torch.stack([source[0], source[-1]])
                codes = eq.detokenize(tokens, ep, seg.n,
                                      boundary_codes=boundary,
                                      rot_codes=rot_codes)
                if root is not None:
                    root = resample_continuous(root, seg.n)
            return codes, name, fam, root
        raise ValueError(f"segment kind '{seg.kind}' not handled here")

    def _bridge_codes(eq, prev_codes, next_codes, n, seed, temperature,
                      pins=None):
        """Equator bridge with `n` new interior frames.

        Pose stream: B3-sampled tokens through A3 (the validated bridge).
        Rotation stream: A3 cannot produce it — linear interpolation between
        the two boundary frames' raw rotation codes, snapped to the lattice."""
        codes4 = torch.cat([prev_codes[-2:], next_codes[:2]], 0)
        ep = eq.endpoints_from_codes(codes4.cpu())
        total = n + 2
        tok = eq.sample_tokens(ep, total, temperature=temperature, seed=seed,
                               pins=pins)
        a = prev_codes[-1, 128:].float()
        b = next_codes[0, 128:].float()
        w = torch.linspace(0, 1, total, device=a.device)[:, None, None]
        rot_interp = torch.round((1 - w) * a + w * b).long()
        generated = eq.detokenize(
            tok, ep, total, rot_codes=rot_interp,
            boundary_codes=torch.stack([prev_codes[-1], next_codes[0]]))
        return generated[1:-1]

    def _preview_trace(codes, target_body: str, title: str,
                       root_t=None, fps: float = 30.0):
        """Decode a source clip for the live viewer without creating a Motion.

        Timeline scrubbing is an editor operation, not a generation request;
        previews therefore stay in a small in-memory LRU instead of polluting
        SQLite, Atlas memory, result files, or quality-control history.
        """
        from ..embodiment import registry
        from ..engine import constraints as MP
        from ..engine.nets.rotations import rot6d_to_matrix
        from ..engine.trace import MotionTrace
        from ..viz.kinematics import balanced_root_rotations

        emb = registry.load(target_body)
        rot, _ = _decode_full_batched(codes, emb.spec, emb.dof)
        dof_t = torch.as_tensor(emb.dof, device=POOL.device,
                                dtype=torch.float64)
        rest_t = torch.as_tensor(emb.rest, device=POOL.device,
                                 dtype=torch.float64)
        feasible, gp, q = _project_batched(
            rot6d_to_matrix(rot).double(), emb.spec, dof_t, rest_t,
            method="global", lm_iters=20)
        root_index = int(np.where(np.asarray(emb.spec.parents) < 0)[0][0])
        rootR = rot6d_to_matrix(
            feasible[:, root_index:root_index + 1]).cpu().numpy()[:, 0]
        rootR = balanced_root_rotations(rootR)
        trace = MotionTrace(
            q=q.detach().cpu().numpy(), rootR=rootR,
            gp=gp.detach().cpu().numpy(),
            stage=np.zeros(len(codes), np.int32), fps=fps, title=title,
            target=target_body, joint_names=list(emb.spec.joint_names),
            root_t=None if root_t is None else np.asarray(root_t, np.float64))
        return trace, emb

    def _finalize(codes, target_body, title, source, prompt=None,
                  parent=None, render=True, fps=30.0, se3=(), root_t=None,
                  family=None, stage=None):
        """codes -> decode -> refine -> gate -> QC -> trace -> DB -> atlas."""
        from ..embodiment import registry
        from ..engine import constraints as MP
        from ..engine.nets.rotations import rot6d_to_matrix
        from ..engine.trace import MotionTrace
        from ..utils import metrics
        gw, eq, atlas = POOL.greenwich, POOL.equator, POOL.atlas
        hspec, hdof, _hrest = POOL.human
        emb = registry.load(target_body)
        if stage is None:
            stage = np.zeros(len(codes), np.int32) if source == "library" \
                else np.ones(len(codes), np.int32)
        else:
            stage = np.asarray(stage, np.int32)
            if stage.shape != (len(codes),):
                raise ValueError("stage must contain one value per frame")
        if root_t is not None:
            root_t = np.asarray(root_t, np.float64).copy()
            if root_t.shape != (len(codes), 3):
                raise ValueError("root_t must have shape [frames, 3]")
        rot, _pos_n = _decode_full_batched(codes, emb.spec, emb.dof)
        rot_h, _ = _decode_full_batched(codes, hspec, hdof)
        # Whole-chain global LM maps decoded rotations onto the target body's
        # feasible joint manifold. The spatial position head is root-relative;
        # it remains available for diagnostics but is not a world-root head.
        Rg = rot6d_to_matrix(rot).double()
        dof_t = torch.as_tensor(emb.dof, device=POOL.device,
                                dtype=torch.float64)
        rest_t = torch.as_tensor(emb.rest, device=POOL.device,
                                 dtype=torch.float64)
        feasible, gp, q = _project_batched(
            Rg, emb.spec, dof_t, rest_t, method="global", lm_iters=20)
        q = q.detach()
        gp = gp.detach()
        refined = feasible.float().detach()
        from ..engine.timeline import (repair_generated_holds,
                                       repair_generated_joint_jumps,
                                       repair_generated_joint_holds,
                                       repair_projection_branch_flips)
        root_index = int(np.where(np.asarray(emb.spec.parents) < 0)[0][0])
        root_rotation = Rg[:, root_index]
        q, branch_report = repair_projection_branch_flips(q)
        if branch_report["repaired_values"]:
            gR, gp = MP.fk_from_angles(q, emb.spec, dof_t, rest=rest_t,
                                       root_R=root_rotation)
            refined = MP.matrix_to_rot6d(gR).float().detach()
            gp = gp.detach()
        repaired, hold_report = repair_generated_holds(refined, stage)
        if hold_report["repaired_frames"]:
            feasible, gp, q = _project_batched(
                rot6d_to_matrix(repaired).double(), emb.spec, dof_t, rest_t,
                method="global", lm_iters=20)
            q = q.detach()
            gp = gp.detach()
            refined = feasible.float().detach()
            q, branch_report_2 = repair_projection_branch_flips(q)
            if branch_report_2["repaired_values"]:
                gR, gp = MP.fk_from_angles(
                    q, emb.spec, dof_t, rest=rest_t,
                    root_R=rot6d_to_matrix(refined)[:, root_index])
                refined = MP.matrix_to_rot6d(gR).float().detach()
                gp = gp.detach()
            branch_report = {
                "repaired_values": (branch_report["repaired_values"]
                                    + branch_report_2["repaired_values"]),
                "large_jumps_before": branch_report["large_jumps_before"],
                "large_jumps_after": branch_report_2["large_jumps_after"],
            }
        # Projection can collapse nearby interpolated rotations back onto the
        # same joint-limit branch. Repair that second-order failure directly
        # on the already feasible joint trajectory, then recover FK outputs.
        projected_root_R = rot6d_to_matrix(refined)[:, root_index]
        projected_gR, _ = MP.fk_from_angles(
            q, emb.spec, dof_t, rest=rest_t, root_R=projected_root_R)
        q, joint_hold_report = repair_generated_joint_holds(
            q, projected_gR, stage)
        if joint_hold_report["repaired_frames"]:
            projected_gR, gp = MP.fk_from_angles(
                q, emb.spec, dof_t, rest=rest_t, root_R=projected_root_R)
            refined = MP.matrix_to_rot6d(projected_gR).float().detach()
            gp = gp.detach()
        # SE3 constrained re-projection on requested spans
        se3_report = []
        effective_stage = stage.copy()
        for c in se3:
            if c.joint >= emb.spec.J:
                raise ValueError(f"SE3 joint {c.joint} outside body with "
                                 f"{emb.spec.J} joints")
            end = min(c.frame_end, len(refined))
            if c.frame_start >= end:
                raise ValueError("SE3 frame range does not overlap the motion")
            Rg = rot6d_to_matrix(refined).double()
            sl = slice(c.frame_start, end)
            delta_p = torch.as_tensor(c.delta_m, device=POOL.device,
                                      dtype=torch.float64) * 100.0
            target_pos = None
            root_translation = False
            if bool((delta_p.abs() > 0).any()):
                if c.joint == root_index:
                    if root_t is None:
                        root_t = np.zeros((len(refined), 3), np.float64)
                    root_t[sl] += delta_p.detach().cpu().numpy()[None]
                    root_translation = True
                else:
                    target_pos = (gp[sl, c.joint:c.joint + 1]
                                  + delta_p[None, None])
            target_rot = None
            if any(abs(v) > 0 for v in c.delta_rot_deg):
                from scipy.spatial.transform import Rotation
                dR = torch.as_tensor(
                    Rotation.from_euler("xyz", c.delta_rot_deg,
                                        degrees=True).as_matrix(),
                    device=POOL.device, dtype=torch.float64)
                target_rot = dR[None, None] @ Rg[sl, c.joint:c.joint + 1]
            if target_pos is None and target_rot is None:
                se3_report.append({"joint": c.joint,
                                   "joint_name": emb.spec.joint_names[c.joint],
                                   "frames": [c.frame_start, end],
                                   "position_error_cm": 0.0,
                                   "rotation_error_deg": 0.0,
                                   "root_translation": root_translation})
                effective_stage[c.frame_start:end] = 2
                continue
            r6, p2, q2, _q0 = MP.project_constrained(
                Rg[sl], emb.spec,
                dof_t,
                joints=(c.joint,),
                target_pos=target_pos, target_rot=target_rot, rest=rest_t)
            refined[sl] = r6.float()
            q[sl] = q2
            gp[sl] = p2
            pos_err = 0.0 if target_pos is None else float(torch.linalg.vector_norm(
                p2[:, c.joint] - target_pos[:, 0], dim=-1).mean())
            rot_err = 0.0
            if target_rot is not None:
                from ..engine.nets.rotations import geodesic_deg
                rot_err = float(geodesic_deg(
                    rot6d_to_matrix(r6[:, c.joint]), target_rot[:, 0]).mean())
            se3_report.append({"joint": c.joint,
                               "joint_name": emb.spec.joint_names[c.joint],
                               "frames": [c.frame_start, end],
                               "position_error_cm": round(pos_err, 4),
                               "rotation_error_deg": round(rot_err, 4),
                               "root_translation": root_translation})
            effective_stage[c.frame_start:end] = 2

        # A task-space solve is allowed to alter its marked span, but it must
        # not reintroduce root-gliding holds in the remaining generated frames.
        final_root_R = rot6d_to_matrix(refined)[:, root_index]
        final_gR, _ = MP.fk_from_angles(
            q, emb.spec, dof_t, rest=rest_t, root_R=final_root_R)
        q, jump_report = repair_generated_joint_jumps(
            q, final_gR, effective_stage)
        if jump_report["repaired_frames"]:
            final_gR, gp = MP.fk_from_angles(
                q, emb.spec, dof_t, rest=rest_t, root_R=final_root_R)
            refined = MP.matrix_to_rot6d(final_gR).float().detach()
            gp = gp.detach()
        q, final_hold_report = repair_generated_joint_holds(
            q, final_gR, effective_stage)
        if final_hold_report["repaired_frames"]:
            final_gR, gp = MP.fk_from_angles(
                q, emb.spec, dof_t, rest=rest_t, root_R=final_root_R)
            refined = MP.matrix_to_rot6d(final_gR).float().detach()
            gp = gp.detach()
        stage = effective_stage
        combined_hold_report = {
            "repaired_frames": int(hold_report["repaired_frames"]
                                   + joint_hold_report["repaired_frames"]
                                   + final_hold_report["repaired_frames"]),
            "holds_before": int(hold_report["holds_before"]),
            "holds_after": int(final_hold_report["holds_after"]),
            "rotation_pass": hold_report,
            "projection_pass": joint_hold_report,
            "post_constraint_pass": final_hold_report,
        }
        rrep = {"refiner": "global projection + temporal continuity repair",
                "hold_repair": combined_hold_report,
                "branch_repair": branch_report,
                "jump_repair": jump_report}
        # synergy gate vs the pre-refine decode's own tokens
        from ..refiner.synergy import synergy_gate
        p9_src, _ = gw.pose9(rot_h.cpu(), hspec, is_global=True)
        gate = synergy_gate(gw, eq, p9_src.to(POOL.device), hspec, hdof,
                            refined, emb.spec, emb.dof)
        source_rot = rot_h[..., :6].cpu().numpy()
        regional_before = metrics.regional_synergy_qc(
            source_rot, hspec, rot.cpu().numpy(), emb.spec)
        regional_after = metrics.regional_synergy_qc(
            source_rot, hspec, refined.cpu().numpy(), emb.spec)
        regional_qc = {"before_refiner": regional_before,
                       "after_refiner": regional_after}
        if regional_before.get("available") and regional_after.get("available"):
            retained = (regional_after["mean_score"]
                        / max(regional_before["mean_score"], 1e-9))
            regional_qc["retained_ratio"] = round(float(retained), 4)
            regional_qc["passed"] = bool(retained >= 0.70
                                          and regional_after["passed"])
        arm_qc = metrics.arm_qc(source_rot, hspec,
                                refined.cpu().numpy(), emb.spec)
        temporal_qc = metrics.continuity_qc(
            refined.cpu().numpy(), root_t, stage, fps)
        qc = {"arm": arm_qc, "limb_synergy": regional_qc,
              "continuity": temporal_qc}
        release_ok = bool(gate.passed and temporal_qc.get("passed")
                          and regional_qc.get("passed"))
        qc["release_passed"] = release_ok
        tok_final, _ep = eq.tokenize(codes)
        # trace + assets
        rootR = rot6d_to_matrix(
            refined[:, root_index:root_index + 1, :6]).cpu().numpy()[:, 0]
        # Robot roots need a balance guard: the human pelvis stream can carry
        # a persistent retargeting lean that is not a valid robot base pose.
        from ..viz.kinematics import balanced_root_rotations
        rootR = balanced_root_rotations(rootR)
        mid_title = title or f"{source}-{int(time.time())}"
        trace = MotionTrace(q=q.detach().cpu().numpy(), rootR=rootR,
                            gp=gp.detach().cpu().numpy(),
                            stage=stage, fps=fps, title=mid_title,
                            target=target_body,
                            tokens=tok_final.cpu().numpy(),
                            joint_names=list(emb.spec.joint_names),
                            root_t=None if root_t is None
                            else np.asarray(root_t, np.float64))
        tp = results_dir() / f"{uuid.uuid4().hex[:10]}_trace.npz"
        trace.save(tp)
        fam = family or family_of(prompt or mid_title)
        with session() as s:
            m = Motion(title=mid_title, family=fam,
                       duration_s=len(refined) / fps, fps=fps,
                       n_frames=len(refined), source=source, prompt=prompt,
                       parent_motion_id=parent,
                       tokens=[int(t) for t in tok_final.cpu()],
                       trace_path=str(tp), gate_ratio=gate.ratio,
                       gate_passed=gate.passed,
                       qc={"motion": qc, "refiner": rrep,
                           "se3": se3_report})
            s.add(m)
            s.commit()
            motion_id = m.id
            s.add(Asset(motion_id=motion_id, kind="trace", path=str(tp),
                        bytes=tp.stat().st_size))
            # Flagged traces remain durable for audit/refining, but they must
            # never become destinations that poison future Atlas searches.
            if release_ok:
                w = atlas.add(tok_final.cpu().numpy(), mid_title,
                              family_id(fam))
                for slot in (4, 12, 20, 28):
                    for pdct in atlas.portals(tok_final.cpu().numpy(), slot,
                                              k=2,
                                              exclude_clip=int(atlas.clip[w])):
                        s.add(AtlasEdge(src_motion_id=motion_id,
                                        src_slot=slot,
                                        dst_window=pdct["window"],
                                        dst_clip=pdct["clip"],
                                        dst_family=pdct["family"],
                                        score=pdct["score"]))
            s.commit()
        out = {"motion_id": motion_id, "frames": len(refined),
               "trace": tp.name, "gate": gate.as_dict(), "qc": qc,
               "refiner": rrep, "family": fam, "se3": se3_report,
               "release_passed": release_ok,
               "tokens": [int(t) for t in tok_final.cpu()]}
        if render:
            mp4 = _try_render(tp, target_body)
            if mp4:
                out["mp4"] = mp4
                with session() as s:
                    s.add(Asset(motion_id=motion_id, kind="mp4",
                                path=str(results_dir() / mp4),
                                bytes=(results_dir() / mp4).stat().st_size))
                    s.commit()
        if emb.xml and Path(emb.xml).exists() and state.get("live"):
            try:
                state["live"].set_trace(trace, emb.xml, target_body)
                out["viewer"] = state["live"].url
            except Exception as exc:  # noqa: BLE001 — viewer is a bonus
                out["viewer_note"] = f"viewer update failed: {exc}"[:200]
        else:
            out["viewer_note"] = ("no mesh attached for this body; add it to "
                                  "robot_meshes.json to light up rendering")
        return out

    def _try_render(trace_path: Path, body: str) -> str | None:
        from ..embodiment import registry
        emb = registry.load(body)
        if not emb.xml or not Path(emb.xml).exists():
            return None                     # no meshes attached: viser-only
        from ..viz.video import trace_to_mp4
        out = trace_path.with_name(trace_path.stem + ".mp4")
        trace_to_mp4(trace_path, emb.xml, body, out)
        return out.name

    # -------------------------------------------------------- endpoints -----
    def _require_body(name: str):
        from ..embodiment import registry
        try:
            return registry.load(name)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc

    def _require_library_id(lib, index: int):
        if index < 0 or index >= len(lib):
            raise HTTPException(422, f"library_id must be in [0, {len(lib)-1}]")

    @app.post("/api/viewer/preview/library")
    async def preview_library_on_timeline(
            library_id: int = Query(ge=0),
            n: int = Query(default=60, ge=1, le=10_000),
            source_frames: int | None = Query(default=None, ge=1, le=10_000),
            source_start: int = Query(default=0, ge=0),
            source_end: int | None = Query(default=None, ge=1, le=10_000),
            target_body: str = Query(default="unitree_h1", min_length=1,
                                     max_length=128)):
        """Put one target-retargeted Library clip on the live canvas."""
        lib = state["library"]
        emb = _require_body(target_body)
        _require_library_id(lib, library_id)
        if not emb.xml or not Path(emb.xml).is_file():
            raise HTTPException(422, "selected body has no renderable mesh")
        key = (int(library_id), int(n), source_frames, int(source_start),
               source_end, target_body)
        revision = state["viewer_revision"]

        def work():
            cache = state["timeline_previews"]
            if key in cache:
                trace = cache[key]
                cache.move_to_end(key)
            else:
                seg = Segment(kind="library", library_id=library_id, n=n,
                              source_frames=source_frames,
                              source_start=source_start,
                              source_end=source_end)
                codes, name, _fam, root = _segment_codes(
                    seg, POOL.equator, lib, target_body)
                trace, _ = _preview_trace(codes, target_body, name,
                                          root_t=root)
                cache[key] = trace
                cache.move_to_end(key)
                while len(cache) > 20:
                    cache.popitem(last=False)
            if revision != state["viewer_revision"]:
                return {**state["live"].transport_state(), "stale": True}
            state["live"].set_trace(trace, emb.xml, target_body)
            return {**state["live"].transport_state(),
                    "title": trace.title, "preview": True}

        return await POOL.run(work)

    @app.post("/api/viewer/preview/timeline")
    async def preview_uncompiled_timeline(req: TimelineRequest):
        """Build one frame-accurate editor preview without generating gaps.

        Library material is decoded normally. An unconstructed bridge contains
        no motion yet, so its entire frame budget repeats the preceding pose and
        root position. The live viewer can consequently run on the exact same
        global frame clock as the timeline instead of looping individual clips.
        """
        lib = state["library"]
        emb = _require_body(req.target_body)
        if not emb.xml or not Path(emb.xml).is_file():
            raise HTTPException(422, "selected body has no renderable mesh")
        if any(seg.kind not in ("library", "gap") for seg in req.segments):
            raise HTTPException(
                422, "prompt and video segments must be generated before preview")
        for seg in req.segments:
            if seg.kind == "library":
                _require_library_id(lib, seg.library_id)
        if req.segments[0].kind == "gap":
            raise HTTPException(422, "a bridge needs a clip on its left")
        cache_key = req.model_dump_json(exclude={"render", "se3", "title"})
        revision = state["viewer_revision"]

        def work():
            cache = state["edit_previews"]
            if cache_key in cache:
                trace = cache[cache_key]
                cache.move_to_end(cache_key)
            else:
                chunks, roots = [], []
                anchor = np.zeros(3, np.float64)
                have_roots = True
                for seg in req.segments:
                    if seg.kind == "gap":
                        if not chunks:
                            raise ValueError("a bridge needs a clip on its left")
                        chunks.append(chunks[-1][-1:].repeat(seg.n, 1, 1))
                        if have_roots:
                            roots.append(np.repeat(
                                anchor[None], seg.n, axis=0))
                        continue
                    codes, _name, _fam, root = _segment_codes(
                        seg, POOL.equator, lib, req.target_body)
                    chunks.append(codes)
                    if root is None:
                        have_roots = False
                        roots.clear()
                    elif have_roots:
                        absolute = anchor + np.asarray(root, np.float64)
                        roots.append(absolute)
                        anchor = absolute[-1].copy()
                codes = torch.cat(chunks, 0)
                root_t = np.concatenate(roots, 0) if have_roots else None
                trace, _ = _preview_trace(
                    codes, req.target_body, req.title or "Timeline preview",
                    root_t=root_t, fps=req.fps)
                cache[cache_key] = trace
                cache.move_to_end(cache_key)
                while len(cache) > 8:
                    cache.popitem(last=False)
            if revision != state["viewer_revision"]:
                return {**state["live"].transport_state(), "stale": True}
            state["live"].set_trace(trace, emb.xml, req.target_body)
            state["live"].set_transport(frame=0, play=False)
            return {**state["live"].transport_state(),
                    "title": trace.title, "preview": True,
                    "frozen_bridges": True}

        return await POOL.run(work)

    @app.post("/api/viewer/preview/motion/{motion_id}")
    async def preview_motion_on_timeline(motion_id: int):
        """Recall a generated Bridge (or any durable motion) for scrubbing."""
        from ..embodiment import registry
        from ..engine.trace import MotionTrace
        with session() as s:
            motion = s.get(Motion, motion_id)
            if not motion:
                raise HTTPException(404, "no such motion")
            trace_path = Path(motion.trace_path)
        if not trace_path.is_file():
            raise HTTPException(404, "motion trace is missing")

        revision = state["viewer_revision"]

        def work():
            trace = MotionTrace.load(trace_path)
            emb = registry.load(trace.target)
            if not emb.xml or not Path(emb.xml).is_file():
                raise ValueError("motion body has no renderable mesh")
            if revision != state["viewer_revision"]:
                return {**state["live"].transport_state(), "stale": True}
            state["live"].set_trace(trace, emb.xml, trace.target)
            return {**state["live"].transport_state(),
                    "title": trace.title, "preview": True}

        return await POOL.run(work)

    @app.post("/api/jobs/play", status_code=202)
    async def play(req: PlayRequest):
        lib = state["library"]
        _require_body(req.target_body)
        _require_library_id(lib, req.library_id)

        async def run(_id):
            def work():
                eq = POOL.equator
                seg = Segment(kind="library", library_id=req.library_id,
                              n=req.n or lib.frames(req.library_id))
                codes, name, fam, root = _segment_codes(seg, eq, lib,
                                                        req.target_body)
                return _finalize(codes, req.target_body, name, "library",
                                 render=req.render, root_t=root, family=fam)
            return await POOL.run(work)
        return {"job_id": _submit("play", req.model_dump(), run)}

    @app.post("/api/jobs/timeline", status_code=202)
    async def timeline(req: TimelineRequest):
        lib = state["library"]
        _require_body(req.target_body)
        for seg in req.segments:
            if seg.kind == "library":
                _require_library_id(lib, seg.library_id)
        if req.segments[0].kind == "gap" or req.segments[-1].kind == "gap":
            raise HTTPException(422, "a bridge needs clips on both sides")
        if any(a.kind == b.kind == "gap"
               for a, b in zip(req.segments, req.segments[1:])):
            raise HTTPException(422, "consecutive bridge gaps are invalid")

        async def run(_id):
            def work():
                eq = POOL.equator
                chunks, names, families, prompts = [], [], [], []
                for seg in req.segments:
                    if seg.kind == "gap":
                        chunks.append({"kind": "gap", "segment": seg})
                        continue
                    if seg.kind in ("prompt", "video"):
                        codes, root = _perception_codes(seg)
                        name = seg.text or seg.video_asset or "clip"
                        fam = family_of(name)
                        chunks.append({"kind": "codes", "codes": codes,
                                       "root": root, "family": fam,
                                       "generated": True})
                        names.append(name); families.append(fam)
                        if seg.text:
                            prompts.append(seg.text)
                        continue
                    codes, name, fam, root = _segment_codes(
                        seg, eq, lib, req.target_body)
                    chunks.append({"kind": "codes", "codes": codes,
                                   "root": root, "family": fam,
                                   "generated": bool(seg.pins)})
                    names.append(name); families.append(fam)
                # Resolve gaps between code chunks. Each source root path is
                # first-frame anchored; bridge_root carries boundary velocity
                # through generated interiors before re-anchoring the next.
                from collections import Counter
                from ..engine.timeline import bridge_root
                out, roots, stages = [], [], []
                anchor = np.zeros(3)
                have_root = all(c.get("root") is not None
                                for c in chunks if c["kind"] == "codes")
                for i, chunk in enumerate(chunks):
                    if chunk["kind"] == "codes":
                        val, root = chunk["codes"], chunk["root"]
                        out.append(val)
                        stages.append(np.full(len(val),
                                              1 if chunk["generated"] else 0,
                                              np.int32))
                        if have_root:
                            absolute = anchor + root
                            roots.append(absolute)
                            anchor = absolute[-1]
                        continue
                    prev = out[-1] if out else None
                    nxt_chunk = next((v for v in chunks[i + 1:]
                                      if v["kind"] == "codes"), None)
                    if prev is None or nxt_chunk is None:
                        raise ValueError("gap segment needs neighbours on "
                                         "both sides")
                    seg = chunk["segment"]
                    out.append(_bridge_codes(eq, prev, nxt_chunk["codes"],
                                             seg.n, seg.seed, seg.temperature,
                                             seg.pins))
                    stages.append(np.ones(seg.n, np.int32))
                    if have_root:
                        gap_root, anchor = bridge_root(
                            roots[-1], nxt_chunk["root"], seg.n)
                        roots.append(gap_root)
                codes = torch.cat(out, 0)
                root_t = np.concatenate(roots, 0) if have_root and roots \
                    else None
                title = req.title or " + ".join(names[:3])[:240]
                family = Counter(families).most_common(1)[0][0] \
                    if families else "other"
                kinds = {seg.kind for seg in req.segments}
                source = "text_prompt" if kinds == {"prompt"} else \
                    "video" if kinds == {"video"} else "edit"
                return _finalize(codes, req.target_body, title, source,
                                 render=req.render, fps=req.fps, se3=req.se3,
                                 root_t=root_t, family=family,
                                 prompt="; ".join(prompts) or None,
                                 stage=np.concatenate(stages))
            return await POOL.run(work)
        return {"job_id": _submit("timeline", req.model_dump(), run)}

    @app.post("/api/jobs/jump", status_code=202)
    async def jump(req: JumpRequest):
        """Portal jump: current motion -> bridge -> destination library clip.
        The atlas differentiator made playable."""
        lib = state["library"]
        _require_body(req.target_body)
        _require_library_id(lib, req.dest_library_id)
        with session() as s:
            source_motion = s.get(Motion, req.motion_id)
            if source_motion is None:
                raise HTTPException(404, "no such motion")
            if not release_passed(source_motion.gate_passed,
                                  source_motion.qc):
                raise HTTPException(
                    422, "source motion is QC-flagged and cannot seed an "
                    "Atlas jump")

        async def run(_id):
            def work():
                eq = POOL.equator
                with session() as s:
                    src = s.get(Motion, req.motion_id)
                if not src:
                    raise ValueError("no such motion")
                from ..embodiment import registry
                from ..engine import constraints as MP
                from ..engine.nets.rotations import matrix_to_rot6d
                from ..engine.timeline import bridge_root
                from ..engine.trace import MotionTrace
                tr = MotionTrace.load(src.trace_path)
                src_body = registry.load(tr.target)
                gR, _gp = MP.fk_from_angles(
                    torch.as_tensor(tr.q, device=POOL.device,
                                    dtype=torch.float64),
                    src_body.spec,
                    torch.as_tensor(src_body.dof, device=POOL.device,
                                    dtype=torch.float64),
                    rest=torch.as_tensor(src_body.rest, device=POOL.device,
                                         dtype=torch.float64),
                    root_R=torch.as_tensor(tr.rootR, device=POOL.device,
                                           dtype=torch.float64))
                p9, _ = POOL.greenwich.pose9(
                    matrix_to_rot6d(gR).float().cpu(), src_body.spec,
                    is_global=True)
                src_codes = POOL.greenwich.encode(
                    p9, src_body.spec, src_body.dof)
                cut = int(round((req.at_slot + 1) / 32 * len(src_codes)))
                cut = max(2, min(len(src_codes), cut))
                prefix = src_codes[:cut]
                dest_seg = Segment(kind="library",
                                   library_id=req.dest_library_id,
                                   n=lib.window)
                dest, dest_name, dest_family, dest_root = _segment_codes(
                    dest_seg, eq, lib, req.target_body)
                bridge = _bridge_codes(eq, prefix, dest, req.bridge_n, 0, 0.9)
                root_t = None
                if tr.root_t is not None and dest_root is not None:
                    src_root = np.asarray(tr.root_t[:cut], np.float64)
                    gap_root, anchor = bridge_root(
                        src_root, dest_root, req.bridge_n)
                    root_t = np.concatenate(
                        [src_root, gap_root, anchor + dest_root], axis=0)
                codes = torch.cat([prefix, bridge, dest], dim=0)
                stage = np.concatenate([
                    np.asarray(tr.stage[:cut], np.int32),
                    np.ones(req.bridge_n, np.int32),
                    np.zeros(len(dest), np.int32)])
                return _finalize(
                    codes, req.target_body,
                    f"{src.title} -> {dest_name}"[:240], "atlas_jump",
                    parent=src.id, render=req.render, root_t=root_t,
                    family=dest_family, stage=stage)
            return await POOL.run(work)
        return {"job_id": _submit("jump", req.model_dump(), run)}

    def _perception_codes(seg: Segment):
        from ..perception.genmo import motion_from_prompt, motion_from_video
        gw = POOL.greenwich
        hspec, hdof, _ = POOL.human
        if seg.kind == "prompt":
            rot6d, root_t = motion_from_prompt(seg.text, seg.n / 30.0)
        else:
            root = (data_dir() / "uploads" / "videos").resolve()
            video = (root / Path(seg.video_asset).name).resolve()
            if not video.is_relative_to(root) or not video.is_file():
                raise ValueError("video asset is not a registered upload")
            rot6d, root_t = motion_from_video(str(video))
        p9, _ = gw.pose9(rot6d, hspec, is_global=True)
        return gw.encode(p9.to(POOL.device), hspec, hdof), root_t

    # ----------------------------------------------------------- ingest -----
    @app.post("/api/bodies/ingest", status_code=202)
    async def ingest_urdf(file: UploadFile = File(...), name: str = ""):
        import shutil
        import stat
        import zipfile

        safe_name = Path(file.filename or "body.urdf").name
        suffix = Path(safe_name).suffix.lower()
        if suffix not in (".urdf", ".zip"):
            raise HTTPException(422, "body upload must be URDF or a ZIP package")
        package = data_dir() / "uploads" / "bodies" / uuid.uuid4().hex
        package.mkdir(parents=True, exist_ok=False)
        uploaded = package / safe_name
        total = 0
        try:
            with uploaded.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > 100 * 1024 * 1024:
                        raise HTTPException(413,
                                            "body upload exceeds 100 MiB")
                    handle.write(chunk)
            if not total:
                raise HTTPException(422, "body upload is empty")
            up = uploaded
            if suffix == ".zip":
                extracted = package / "extracted"
                extracted.mkdir()
                with zipfile.ZipFile(uploaded) as archive:
                    members = archive.infolist()
                    if sum(m.file_size for m in members) > 500 * 1024 * 1024:
                        raise HTTPException(
                            413, "expanded robot package exceeds 500 MiB")
                    for member in members:
                        target = (extracted / member.filename).resolve()
                        if not target.is_relative_to(extracted.resolve()):
                            raise HTTPException(422,
                                                "ZIP contains an unsafe path")
                        mode = member.external_attr >> 16
                        if stat.S_ISLNK(mode):
                            raise HTTPException(422,
                                                "ZIP symlinks are not allowed")
                    archive.extractall(extracted)
                urdfs = sorted(extracted.rglob("*.urdf"))
                if len(urdfs) != 1:
                    raise HTTPException(
                        422, "ZIP package must contain exactly one URDF")
                up = urdfs[0]
        except Exception:
            shutil.rmtree(package, ignore_errors=True)
            raise

        async def run(_id):
            def work():
                from ..embodiment.urdf_ingest import ingest, safe_body_name
                requested = safe_body_name(name) if name else None
                rep = ingest(up, requested, device=POOL.device)
                with session() as s:
                    row = s.query(Skeleton).filter_by(name=rep["name"]).first()
                    if row is None:
                        row = Skeleton(name=rep["name"], kind="user_urdf")
                        s.add(row)
                    row.joints = rep["joints"]
                    row.height_cm = rep["height_cm"]
                    row.xml_path = rep["mjcf"]
                    row.sem_labels = rep["semantics"]
                    row.limit_report = rep["limits"]
                    s.add(Asset(kind="urdf", path=str(up), bytes=total))
                    s.commit()
                return rep
            return await POOL.run(work)
        return {"job_id": _submit("ingest", {"file": safe_name}, run)}

    # ------------------------------------------------------------ atlas -----
    @app.get("/api/atlas/portals/{motion_id}")
    def portals(motion_id: int,
                slot: int = Query(default=16, ge=0, lt=32),
                k: int = Query(default=8, ge=1, le=32)):
        with session() as s:
            m = s.get(Motion, motion_id)
        if not m or not m.tokens:
            raise HTTPException(404, "motion has no tokens")
        ps = POOL.atlas.portals(np.asarray(m.tokens), slot, k=k + 4)
        ps = [p for p in ps if p["clip"] != m.title][:k]   # no self-portals
        # The frozen Atlas is larger than the curated raw-code library. Only
        # expose a play button when the hit resolves to a real raw-code row.
        # ``resolve_portal`` also repairs fixed-width labels from old indices
        # and generated rows that retained the exact source token sequence.
        lib = state["library"]
        for p in ps:
            p["library_id"] = lib.resolve_portal(
                p["clip"], POOL.atlas.tokens[p["window"]])
        return {"portals": ps}

    @app.get("/api/atlas/window/{window}")
    def atlas_window(window: int):
        a = POOL.atlas
        if window < 0 or window >= len(a.tokens):
            raise HTTPException(404, "no such window")
        return {"window": window,
                "tokens": [int(t) for t in a.tokens[window]],
                "clip": a.clips[int(a.clip[window])],
                "family": FAMILIES[int(a.family[window])]}

    @app.get("/api/atlas/walk/{window}")
    def atlas_walk(window: int,
                   steps: int = Query(default=6, ge=1, le=64),
                   seed: int = Query(default=0, ge=0, le=2**32 - 1)):
        a = POOL.atlas
        if window < 0 or window >= len(a.tokens):
            raise HTTPException(404, "no such window")
        path = a.walk(window, steps, seed)
        return {"path": [{"window": int(w),
                          "clip": a.clips[int(a.clip[w])],
                          "family": FAMILIES[int(a.family[w])]}
                         for w in path]}

    @app.get("/api/motions")
    def motions(limit: int = Query(default=50, ge=1, le=200)):
        with session() as s:
            rows = s.query(Motion).order_by(Motion.id.desc()).limit(limit)
            return {"motions": [
                {"id": m.id, "title": m.title, "family": m.family,
                 "frames": m.n_frames, "source": m.source,
                 "gate_ratio": m.gate_ratio, "gate_passed": m.gate_passed,
                 "release_passed": release_passed(m.gate_passed, m.qc),
                 "trace": Path(m.trace_path).name,
                 "tokens": m.tokens}
                for m in rows]}

    # ------------------------------------------------- viser proxy ----------
    # The viewport must survive ANY single-port tunnel (the browser may only
    # forward the gateway's port). All viser traffic — static client + the
    # msgpack websocket — is therefore proxied through the gateway itself:
    #   /viewer/<assets>  -> http://127.0.0.1:<viser>/<assets>
    #   /viser-ws         -> ws://127.0.0.1:<viser>/
    # and the iframe loads /viewer/?websocket=ws(s)://<host>/viser-ws.
    # NOTE: WebSocket must be importable from THIS MODULE's globals — with
    # `from __future__ import annotations` FastAPI resolves the string
    # annotation against module scope; a function-local import made it fall
    # back to "query parameter" and 403 every handshake.
    import httpx

    async def _viewer_proxy(port: int, path: str):
        from fastapi.responses import Response
        url = f"http://127.0.0.1:{port}/{path or ''}"
        async with httpx.AsyncClient() as client:
            r = await client.get(url)
        return Response(content=r.content,
                        status_code=r.status_code,
                        media_type=r.headers.get("content-type"),
                        headers={"cache-control": r.headers["cache-control"]}
                        if "cache-control" in r.headers else None)

    @app.get("/viewer/{path:path}")
    async def viewer_proxy(path: str):
        from ..config import CONFIG
        return await _viewer_proxy(CONFIG.viewer_ports[0], path)

    @app.get("/body-viewer/{path:path}")
    async def body_viewer_proxy(path: str):
        from ..config import CONFIG
        return await _viewer_proxy(CONFIG.viewer_ports[1], path)

    @app.get("/source-viewer/{path:path}")
    async def source_viewer_proxy(path: str):
        from ..config import CONFIG
        return await _viewer_proxy(CONFIG.viewer_ports[2], path)

    async def _viser_ws_proxy(ws: WebSocket, port: int):
        import websockets

        # viser carries its client version in the websocket SUBPROTOCOL and
        # rejects 'unknown' — forward the client's offer upstream, then accept
        # the browser with whatever viser negotiated
        offered = ws.scope.get("subprotocols") or []
        up = await websockets.connect(
            f"ws://127.0.0.1:{port}/",
            subprotocols=offered or None, max_size=None)
        await ws.accept(subprotocol=up.subprotocol)
        try:
            async def pump_up():
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        return
                    if msg.get("bytes") is not None:
                        await up.send(msg["bytes"])
                    elif msg.get("text") is not None:
                        await up.send(msg["text"])

            async def pump_down():
                async for m in up:
                    if isinstance(m, bytes):
                        await ws.send_bytes(m)
                    else:
                        await ws.send_text(m)

            tasks = (asyncio.create_task(pump_up()),
                     asyncio.create_task(pump_down()))
            _done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            # Always consume both task results. Without this, closing a mobile
            # browser leaves noisy "Task exception was never retrieved"
            # errors and eventually obscures real service failures.
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:  # noqa: BLE001 — disconnect is normal lifecycle
            pass
        finally:
            try:
                await up.close()
            except Exception:  # noqa: BLE001
                pass

    @app.websocket("/viser-ws")
    async def viser_ws(ws: WebSocket):
        from ..config import CONFIG
        await _viser_ws_proxy(ws, CONFIG.viewer_ports[0])

    @app.websocket("/body-viser-ws")
    async def body_viser_ws(ws: WebSocket):
        from ..config import CONFIG
        await _viser_ws_proxy(ws, CONFIG.viewer_ports[1])

    @app.websocket("/source-viser-ws")
    async def source_viser_ws(ws: WebSocket):
        from ..config import CONFIG
        await _viser_ws_proxy(ws, CONFIG.viewer_ports[2])

    if FRONTEND.exists():
        app.mount("/", StaticFiles(directory=FRONTEND, html=True),
                  name="frontend")

        # pure-ASGI no-cache shim: BaseHTTPMiddleware (@app.middleware) was
        # 403-ing every WebSocket upgrade — the classic starlette footgun
        class _NoCacheIndex:
            def __init__(self, inner):
                self.inner = inner

            async def __call__(self, scope, receive, send):
                if scope["type"] == "http" and scope.get("path") in ("/", "/index.html"):
                    async def send2(msg):
                        if msg["type"] == "http.response.start":
                            headers = [(k, v) for k, v in msg.get("headers", [])
                                       if k.lower() != b"cache-control"]
                            headers.append((b"cache-control", b"no-cache"))
                            msg = {**msg, "headers": headers}
                        await send(msg)
                    return await self.inner(scope, receive, send2)
                return await self.inner(scope, receive, send)
        app.add_middleware(_NoCacheIndex)
    return app
