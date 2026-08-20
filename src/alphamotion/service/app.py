"""AlphaMotion service gateway.

Warm pool + async job runner + SQLite persistence. Every generated motion goes
through: assemble codes -> decode on target -> refine -> synergy gate -> QC ->
trace asset -> DB row -> atlas registration + edges. Jobs survive restarts in
the DB; results are files under data_dir()/results.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import httpx
import numpy as np
import torch
from fastapi import (FastAPI, File, Form, HTTPException, Query, Request, UploadFile,
                     WebSocket)
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from ..atlas.families import FAMILIES, family_id, family_of
from ..config import setup_gl_backend
from ..paths import data_dir, results_dir
from ..projects import ProjectStore
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
BODY_THUMBNAIL_LOCK = threading.Lock()


def _library_preview_key(lib, library_id: int) -> str:
    """Content identity for hover previews, independent of list ordering."""
    i = int(library_id)
    # CompositeLibrary intentionally exposes only the metadata shared by all
    # shards.  Source model/gender are useful salt for a standalone Library,
    # but the asset ID remains the canonical identity when those optional
    # arrays are unavailable.
    def optional(name: str, default=None):
        values = getattr(lib, name, None)
        return values[i] if values is not None else default

    identity = {
        "renderer": "neutral-smplx-hover-v4",
        "asset_id": lib.asset_ids[i],
        "origin_id": lib.origin_ids[i],
        "name": lib.names[i],
        "dataset": lib.datasets[i],
        "source": lib.sources[i],
        "frames": lib.frames(i),
        "model": optional("source_models"),
        "gender": optional("source_genders", "neutral"),
        "augmentation": lib.augmentations[i],
        "augmentation_value": lib.augmentation_values[i],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"),
                         default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]

def create_app() -> FastAPI:
    app = FastAPI(title="AlphaMotion", version="0.1.0")
    state: dict = {"library": None, "warm": None,
                   "library_previews": OrderedDict(),
                   "timeline_previews": OrderedDict(),
                   "edit_previews": OrderedDict(),
                   "source_skins": OrderedDict(),
                   "source_codes": OrderedDict(),
                   "motion_codes": OrderedDict(),
                   "sync_comparison": False,
                   "editor_gizmo": {"seq": 0, "active": False},
                   "viewer_revision": 0, "data_studio_sync": None,
                   "data_studio": None, "data_studio_status": {},
                   "projects": ProjectStore(), "smpl_generations": {}}

    async def _sync_data_studio_library() -> None:
        """Publish saved derived clips, then atomically reload the ID space."""
        current = state.get("data_studio_sync")
        if isinstance(current, dict) and current.get("status") == "running":
            return
        state["data_studio_sync"] = {"status": "running", "started": time.time()}
        project = Path(__file__).resolve().parents[3]
        script = project / "scripts" / "sync_data_studio_library.py"
        env = os.environ.copy()
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(script), cwd=str(project), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await process.communicate()
        if process.returncode:
            state["data_studio_sync"] = {
                "status": "failed", "finished": time.time(),
                "error": stderr.decode(errors="replace")[-1200:]}
            return
        try:
            report = json.loads(stdout.decode(errors="replace"))
        except (ValueError, TypeError):
            report = {"output": stdout.decode(errors="replace")[-1200:]}
        from ..atlas.library import load_default as load_library
        state["library"] = load_library()
        state["library_previews"].clear()
        state["source_skins"].clear()
        state["data_studio_sync"] = {
            "status": "ready", "finished": time.time(), **report}

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
        from ..data_studio import DataStudioManager
        state["data_studio"] = DataStudioManager()
        state["data_studio_status"] = state["data_studio"].start()
        state["library"] = load_library()
        if not state["projects"].list():
            state["projects"].create("Untitled Motion Project")
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

    @app.on_event("shutdown")
    async def _shutdown():
        if state.get("data_studio") is not None:
            state["data_studio"].stop()

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
                "data_studio": state.get("data_studio_status", {}),
                "capabilities": {
                    "perception": perception["ready"],
                    "perception_text": perception["text"],
                    "perception_video": perception["video"],
                    "perception_detail": perception,
                    "temporal": True, "token_pins": True, "se3": True,
                    "mp4": True,
                    "render_bodies": sorted(registry.mesh_map())}}

    def _all_bodies():
        from ..embodiment import registry
        out = []
        renderable = registry.mesh_map()
        for n in registry.bundled_names():
            out.append({"name": n, "source": "bundled",
                        "renderable": n in renderable})
        for n in registry.user_names():
            out.append({"name": n, "source": "user",
                        "renderable": bool(registry.load(n).xml)})
        return out

    @app.get("/api/bodies")
    def bodies(project_id: str = Query(default="", max_length=32)):
        out = _all_bodies()
        if project_id:
            try:
                allowed = state["projects"].body_scope(project_id)
            except KeyError as exc:
                raise HTTPException(404, "project not found") from exc
            out = [item for item in out if item["name"] in allowed]
        return {"bodies": out}

    @app.get("/api/starter-bodies")
    def starter_bodies():
        """Shared robot library offered from Data Studio, never Motion Studio."""
        return {"bodies": _all_bodies()}

    @app.get("/api/bodies/{name}/thumbnail.webp")
    def body_thumbnail(name: str):
        """Cached mesh thumbnail in the model-authored reference pose."""
        from ..embodiment import registry
        from ..paths import cache_dir
        try:
            embodiment = registry.load(name)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        if not embodiment.xml:
            raise HTTPException(404, "body does not include a renderable mesh")
        xml = Path(embodiment.xml)
        try:
            stat = xml.stat()
            identity = f"reference-v3|{name}|{xml.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        except OSError:
            identity = f"reference-v3|{name}|{xml}"
        digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
        path = cache_dir() / "body_previews" / f"{digest}.webp"
        if not path.is_file():
            with BODY_THUMBNAIL_LOCK:
                if not path.is_file():
                    try:
                        from PIL import Image
                        from ..viz.video import render_body_reference
                        pixels = render_body_reference(str(xml))
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = path.with_suffix(".tmp.webp")
                        Image.fromarray(pixels).save(
                            temporary, "WEBP", quality=88, method=4)
                        os.replace(temporary, path)
                    except Exception as exc:  # noqa: BLE001
                        raise HTTPException(
                            422, f"could not render body thumbnail: {exc}") from exc
        return FileResponse(path, media_type="image/webp", headers={
            "cache-control": "public, max-age=31536000, immutable"})

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
                data_role: str = Query(default="", max_length=32),
                augmentation: str = Query(default="", max_length=64),
                label: str = Query(default="", max_length=64),
                source: str = Query(default="", max_length=100),
                project_id: str = Query(default="", max_length=32),
                offset: int = Query(default=0, ge=0),
                limit: int = Query(default=24, ge=1, le=100)):
        allowed_indices = allowed_assets = None
        if project_id:
            try:
                allowed_indices, allowed_assets = state["projects"].motion_scope(
                    project_id)
            except KeyError as exc:
                raise HTTPException(404, "project not found") from exc
        return state["library"].search(
            q=q, family=family, dataset=dataset, data_role=data_role,
            augmentation=augmentation, label=label, source=source,
            offset=offset, limit=limit, allowed_indices=allowed_indices,
            allowed_asset_ids=allowed_assets)

    @app.get("/api/starter-library")
    def starter_library(q: str = Query(default="", max_length=200),
                        family: str = Query(default="", max_length=32),
                        dataset: str = Query(default="", max_length=64),
                        data_role: str = Query(default="", max_length=32),
                        augmentation: str = Query(default="", max_length=64),
                        label: str = Query(default="", max_length=64),
                        source: str = Query(default="", max_length=100),
                        offset: int = Query(default=0, ge=0),
                        limit: int = Query(default=24, ge=1, le=100)):
        """Shared motion catalog shown only in Data Studio's import surface."""
        return state["library"].search(
            q=q, family=family, dataset=dataset, data_role=data_role,
            augmentation=augmentation, label=label, source=source,
            offset=offset, limit=limit)

    # ---------------------------------------------------------- projects ---
    @app.get("/api/projects")
    def projects():
        return {"projects": state["projects"].list()}

    @app.post("/api/projects", status_code=201)
    def create_project(payload: dict):
        try:
            return state["projects"].create(payload.get("name") or
                                             "Untitled Motion Project")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str):
        try:
            return state["projects"].get(project_id)
        except KeyError as exc:
            raise HTTPException(404, "project not found") from exc

    @app.put("/api/projects/{project_id}")
    def save_project(project_id: str, payload: dict):
        try:
            return state["projects"].save(project_id, payload)
        except KeyError as exc:
            raise HTTPException(404, "project not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/projects/{project_id}/media")
    def project_media(project_id: str):
        try:
            project = state["projects"].get(project_id)
        except KeyError as exc:
            raise HTTPException(404, "project not found") from exc
        assets = project.get("assets") or {}
        indices, asset_ids = state["projects"].motion_scope(project_id)
        ready = state["library"].search(
            offset=0, limit=100, allowed_indices=indices,
            allowed_asset_ids=asset_ids)
        ready_ids = {item.get("asset_id") for item in ready["items"]}
        ready_indices = {item["id"] for item in ready["items"]}
        from ..config import CONFIG
        from ..data_studio import catalog_lookup
        data_studio_lookup = catalog_lookup(CONFIG.data_studio_db)

        def data_studio_asset_id(item: dict) -> str:
            existing = str(item.get("data_studio_asset_id") or "")
            if existing:
                return existing
            library_id = item.get("library_id")
            if library_id is None:
                return str(item.get("asset_id") or "")
            index = int(library_id)
            title = str(state["library"].names[index]).rsplit("__", 1)[-1]
            source = str(state["library"].sources[index])
            record = data_studio_lookup.get(
                (source.casefold(), title.casefold()))
            return str((record or {}).get("asset_id") or
                       item.get("asset_id") or "")

        motions = []
        for item in assets.get("motions") or []:
            value = dict(item)
            library_id = value.get("library_id")
            if library_id is not None:
                try:
                    index = int(library_id)
                    if 0 <= index < len(state["library"]):
                        lib = state["library"]
                        value.update({
                            "library_id": index,
                            "name": lib.names[index],
                            "family": lib.families[index],
                            "dataset": lib.datasets[index],
                            "source": lib.sources[index],
                            "data_role": lib.data_roles[index],
                            "augmentation": lib.augmentations[index],
                            "augmentation_value":
                                lib.augmentation_values[index],
                            "labels": list(lib.labels[index]),
                            "variant_count": lib.variant_counts[index],
                            "frames": lib.frames(index),
                        })
                except (TypeError, ValueError):
                    pass
            value["data_studio_asset_id"] = data_studio_asset_id(value)
            value["motion_ready"] = (value.get("library_id") in ready_indices or
                                     value.get("asset_id") in ready_ids)
            motions.append(value)
        return {"project": {"id": project["id"], "name": project["name"]},
                "motions": motions, "bodies": assets.get("bodies") or [],
                "ready_motion_count": ready["total"]}

    @app.delete("/api/projects/{project_id}/media")
    def delete_project_media(project_id: str, kind: str,
                             asset_id: str = "",
                             library_id: int | None = None,
                             name: str = ""):
        try:
            project, removed = state["projects"].remove_media(
                project_id, kind=kind, asset_id=asset_id,
                library_id=library_id, name=name)
        except KeyError as exc:
            raise HTTPException(404, "project not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"ok": True, "removed": removed, "project": project}

    @app.post("/api/projects/{project_id}/import-generated")
    def import_generated_media(project_id: str, payload: dict):
        """Add durable AI-generated motions to a project's local Media bin.

        Generated traces live in the AlphaMotion store and do not need to be
        re-indexed into the shared SMPL catalog before a user can review them.
        The project keeps a reference only; deleting it never deletes the
        generated trace itself.
        """
        ids = []
        for raw in payload.get("motion_ids") or []:
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value > 0 and value not in ids:
                ids.append(value)
        if not ids:
            raise HTTPException(422, "motion_ids must contain at least one id")
        with session() as s:
            rows = s.query(Motion).filter(Motion.id.in_(ids)).all()
            by_id = {int(row.id): row for row in rows}
            missing = [value for value in ids if value not in by_id]
            if missing:
                raise HTTPException(404, f"generated motion not found: {missing}")
            motions = [{
                "asset_id": f"generated:{row.id}",
                "motion_id": int(row.id),
                "name": row.title,
                "origin": "ai_generated",
                "source": row.source,
                "family": row.family,
                "frames": row.n_frames,
                "fps": row.fps,
                "state": "ready",
                "generation_method": (row.qc or {}).get("generation", {}).get(
                    "method", row.source),
            } for row in (by_id[value] for value in ids)]
        try:
            project = state["projects"].add_media(
                project_id, motions=motions,
                bin_name=str(payload.get("bin_name") or "Generated motions"))
        except KeyError as exc:
            raise HTTPException(404, "project not found") from exc
        return {"ok": True, "imported": len(motions), "project": project,
                "motions": motions}

    @app.post("/api/projects/{project_id}/import-data-studio-results")
    def import_data_studio_results(project_id: str, payload: dict):
        """Attach durable Data Studio process outputs to Project Media."""
        try:
            state["projects"].get(project_id)
        except KeyError as exc:
            raise HTTPException(404, "project not found") from exc
        asset_ids = list(dict.fromkeys(
            str(value).strip() for value in payload.get("asset_ids") or []
            if str(value).strip()))
        if not asset_ids:
            raise HTTPException(422, "processed asset_ids are required")
        labeled = {str(value).strip() for value in
                   payload.get("labeled_asset_ids") or []}
        from ..config import CONFIG
        import sqlite3
        db = sqlite3.connect(CONFIG.data_studio_db)
        db.row_factory = sqlite3.Row
        try:
            placeholders = ",".join("?" for _ in asset_ids)
            rows = db.execute(
                f"SELECT * FROM assets WHERE id IN ({placeholders}) "
                "AND kind='smplh_motion' AND status='ready'", asset_ids
            ).fetchall()
        finally:
            db.close()
        by_id = {str(row["id"]): row for row in rows}
        missing = [value for value in asset_ids if value not in by_id]
        if missing:
            raise HTTPException(404, f"processed assets are not ready: {missing}")
        motions = []
        for asset_id in asset_ids:
            row = by_id[asset_id]
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except (TypeError, ValueError):
                metadata = {}
            frames = int(metadata.get("frames") or 0)
            fps = float(metadata.get("fps") or metadata.get("mocap_framerate")
                        or 30.0)
            path = Path(str(row["container"] or ""))
            if path.is_file() and (not frames or not metadata.get("fps")):
                try:
                    with np.load(path, allow_pickle=False) as archive:
                        sequence = (archive["poses"] if "poses" in archive else
                                    archive["local_rot6d"] if "local_rot6d" in archive
                                    else None)
                        if sequence is not None:
                            frames = int(len(sequence))
                        for key in ("mocap_framerate", "fps"):
                            if key in archive:
                                fps = float(np.asarray(archive[key]).reshape(-1)[0])
                                break
                except (OSError, ValueError, KeyError):
                    pass
            motions.append({
                "asset_id": asset_id,
                "data_studio_asset_id": asset_id,
                "name": str(row["title"]),
                "origin": "data_studio_processed",
                "source": str(row["source"]),
                "folder": str(row["folder"]),
                "local_path": str(row["container"]),
                "frames": frames,
                "fps": fps,
                "state": "ready",
                "labels": ["foot_contact"] if asset_id in labeled else [],
                "process_run_id": str(payload.get("run_id") or ""),
            })
        project = state["projects"].add_media(
            project_id, motions=motions, bin_name="Processed in Data Studio")
        return {"ok": True, "imported": len(motions), "motions": motions,
                "project": project}

    @app.post("/api/motions/{motion_id}/share")
    def share_generated_motion(motion_id: int):
        """Mark a generated motion for the shared Data Studio catalog.

        The catalog synchronizer can publish the trace on its next pass.  The
        explicit marker makes sharing idempotent and visible immediately in
        the Generated motions panel.
        """
        with session() as s:
            row = s.get(Motion, int(motion_id))
            if row is None:
                raise HTTPException(404, "generated motion not found")
            qc = dict(row.qc or {})
            generation = dict(qc.get("generation") or {})
            generation["shared_to_library"] = True
            qc["generation"] = generation
            row.qc = qc
            s.commit()
            return {"ok": True, "motion_id": row.id, "shared": True,
                    "title": row.title}

    @app.post("/api/projects/{project_id}/import")
    def import_project_media(project_id: str, payload: dict):
        lib = state["library"]
        from ..config import CONFIG
        from ..data_studio import catalog_lookup
        data_studio_lookup = catalog_lookup(CONFIG.data_studio_db)
        motions = []
        for raw in payload.get("library_ids") or []:
            try:
                index = int(raw)
                if index < 0 or index >= len(lib):
                    raise ValueError
            except (TypeError, ValueError):
                raise HTTPException(422, f"invalid library id: {raw}")
            title = str(lib.names[index]).rsplit("__", 1)[-1]
            record = data_studio_lookup.get(
                (str(lib.sources[index]).casefold(), title.casefold()))
            motions.append({
                "library_id": index,
                "asset_id": lib.asset_ids[index] or f"library:{index}",
                "data_studio_asset_id": str(
                    (record or {}).get("asset_id") or
                    lib.asset_ids[index] or f"library:{index}"),
                "name": lib.names[index],
                "origin": "starter",
                "family": lib.families[index],
                "dataset": lib.datasets[index],
                "source": lib.sources[index],
                "data_role": lib.data_roles[index],
                "augmentation": lib.augmentations[index],
                "augmentation_value": lib.augmentation_values[index],
                "labels": list(lib.labels[index]),
                "variant_count": lib.variant_counts[index],
                "frames": lib.frames(index),
                "state": "ready",
            })
        available = {item["name"]: item for item in _all_bodies()}
        bodies_to_add = []
        for name in payload.get("body_names") or []:
            if name not in available:
                raise HTTPException(422, f"unknown body: {name}")
            bodies_to_add.append({**available[name], "origin": "starter"})
        try:
            return state["projects"].add_media(
                project_id, motions=motions, bodies=bodies_to_add,
                bin_name=str(payload.get("bin_name") or "Imported from library"))
        except KeyError as exc:
            raise HTTPException(404, "project not found") from exc

    @app.post("/api/projects/{project_id}/upload-smpl", status_code=201)
    async def upload_project_smpl(project_id: str,
                                  file: UploadFile = File(...)):
        """Store and register a project-local SMPL-family NPZ for processing."""
        try:
            state["projects"].get(project_id)
        except KeyError as exc:
            raise HTTPException(404, "project not found") from exc
        safe_name = Path(file.filename or "motion.npz").name
        if Path(safe_name).suffix.lower() != ".npz":
            raise HTTPException(422, "SMPL upload must be an NPZ file")
        target = state["projects"].media_dir(project_id, "motions") / safe_name
        if target.exists():
            target = target.with_name(f"{target.stem}-{uuid.uuid4().hex[:8]}.npz")
        total = 0
        with target.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > 1024 * 1024 * 1024:
                    target.unlink(missing_ok=True)
                    raise HTTPException(413, "SMPL upload exceeds 1 GiB")
                handle.write(chunk)
        try:
            with np.load(target, allow_pickle=True) as data:
                if "poses" not in data.files:
                    raise ValueError("NPZ has no poses array")
                poses = np.asarray(data["poses"])
                if poses.ndim != 2 or poses.shape[0] < 1 or poses.shape[1] < 72:
                    raise ValueError(f"expected poses [frames, >=72], got {poses.shape}")
                fps = next((float(np.asarray(data[key]).reshape(()))
                            for key in ("mocap_framerate", "mocap_frame_rate", "fps")
                            if key in data.files), 30.0)
        except Exception as exc:
            target.unlink(missing_ok=True)
            raise HTTPException(422, f"invalid SMPL-family NPZ: {exc}") from exc
        source = f"Project / {project_id}"
        locator = f"direct|{target}|"
        asset_id = hashlib.sha1(f"{source}|{locator}".encode()).hexdigest()[:24]
        from ..config import CONFIG
        import sqlite3
        db = sqlite3.connect(CONFIG.data_studio_db)
        try:
            db.execute(
                """INSERT INTO assets
                (id,source,title,folder,kind,format,locator_type,container,
                 inner_path,aux_json,animated,status,size,metadata_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET title=excluded.title,
                container=excluded.container,size=excluded.size,
                metadata_json=excluded.metadata_json""",
                (asset_id, source, target.stem, "Uploads", "smplh_motion",
                 "npz", "direct", str(target), "", "{}", 1, "ready", total,
                 json.dumps({"frames": int(poses.shape[0]), "fps": fps,
                             "pose_dimensions": int(poses.shape[1]),
                             "project_id": project_id})))
            db.commit()
        finally:
            db.close()
        motion = {"asset_id": asset_id, "library_id": None,
                  "name": target.stem, "origin": "upload",
                  "source": source, "local_path": str(target),
                  "frames": int(poses.shape[0]), "fps": fps,
                  "state": "ready_for_data_studio"}
        state["projects"].add_media(project_id, motions=[motion])
        return motion

    @app.post("/api/projects/{project_id}/generate-smpl", status_code=202)
    async def generate_project_smpl(project_id: str, payload: dict):
        """Run GENMO into a project-local SMPL asset, never a robot trace."""
        try:
            state["projects"].get(project_id)
        except KeyError as exc:
            raise HTTPException(404, "project not found") from exc
        mode = str(payload.get("mode") or "prompt")
        text = str(payload.get("text") or "").strip()
        video_asset = str(payload.get("video_asset") or "").strip()
        frames = max(15, min(10_000, int(payload.get("frames") or 120)))
        if mode == "prompt" and not text:
            raise HTTPException(422, "prompt text is required")
        if mode not in ("prompt", "video"):
            raise HTTPException(422, "mode must be prompt or video")

        async def run(_id):
            def work():
                from ..perception.genmo import (global_to_local_rot6d,
                                                motion_from_prompt,
                                                motion_from_video)
                if mode == "prompt":
                    global6d, root_cm = motion_from_prompt(text, frames / 30.0)
                    title = text[:120]
                else:
                    root = (data_dir() / "uploads" / "videos").resolve()
                    video = (root / Path(video_asset).name).resolve()
                    if not video.is_relative_to(root) or not video.is_file():
                        raise ValueError("video asset is not a registered upload")
                    global6d, root_cm = motion_from_video(str(video), frames)
                    title = Path(video_asset).stem[:120]
                local6d = global_to_local_rot6d(global6d).cpu().numpy()
                generation_id = uuid.uuid4().hex[:16]
                target_root = data_dir() / "generated_smpl" / project_id
                target_root.mkdir(parents=True, exist_ok=True)
                target = target_root / f"{generation_id}.npz"
                np.savez_compressed(target, local_rot6d=local6d.astype(np.float16),
                                    root_cm=np.asarray(root_cm, np.float32),
                                    hand_pose=np.zeros((len(local6d), 0), np.float16),
                                    betas=np.zeros(10, np.float32), gender=np.asarray(0, np.uint8),
                                    model_family=np.asarray(0, np.uint8), fps=np.asarray(30.0))
                asset_id = hashlib.sha1(str(target).encode()).hexdigest()[:24]
                motion = {"asset_id": asset_id, "name": f"GENMO · {title}",
                          "origin": "genmo", "source": "GENMO", "local_path": str(target),
                          "data_studio_asset_id": asset_id,
                          "frames": len(local6d), "fps": 30.0, "state": "ready",
                          "generation_id": generation_id, "project_id": project_id,
                          "prompt": text if mode == "prompt" else "",
                          "imported": False, "shared": False,
                          "generation_method": "text-to-smpl" if mode == "prompt" else "video-to-smpl"}
                state["smpl_generations"][generation_id] = motion
                target.with_suffix(".json").write_text(
                    json.dumps(motion, ensure_ascii=False, indent=2), encoding="utf-8")
                return motion
            return await POOL.run(work)
        return {"job_id": _submit("generate-smpl", {"project_id": project_id, **payload}, run)}

    def _project_smpl_generations(project_id: str) -> list[dict]:
        root = data_dir() / "generated_smpl" / project_id
        result = []
        if root.is_dir():
            for path in root.glob("*.json"):
                try:
                    item = json.loads(path.read_text(encoding="utf-8"))
                    if Path(item.get("local_path", "")).is_file():
                        state["smpl_generations"][item["generation_id"]] = item
                        result.append(item)
                except (OSError, ValueError, KeyError, TypeError):
                    continue
        return sorted(result, key=lambda item: item["generation_id"], reverse=True)

    @app.get("/api/projects/{project_id}/smpl-generations")
    def list_project_smpl_generations(project_id: str):
        try:
            state["projects"].get(project_id)
        except KeyError as exc:
            raise HTTPException(404, "project not found") from exc
        return {"generations": _project_smpl_generations(project_id)}

    def _smpl_generation(generation_id: str, project_id: str = "") -> dict:
        item = state["smpl_generations"].get(generation_id)
        if item is None and project_id:
            _project_smpl_generations(project_id)
            item = state["smpl_generations"].get(generation_id)
        if item is None or (project_id and item.get("project_id") != project_id):
            raise HTTPException(404, "generated SMPL motion not found")
        return item

    def _preview_smpl_file(path: Path, name: str, fps: float = 30.0):
        def work():
            with np.load(path, allow_pickle=False) as raw:
                if "local_rot6d" in raw:
                    local = np.asarray(raw["local_rot6d"], np.float32)
                    root = np.asarray(raw["root_cm"], np.float32)
                    hands = np.asarray(raw["hand_pose"], np.float32) \
                        if "hand_pose" in raw else np.empty((len(local), 0))
                    betas = np.asarray(raw["betas"], np.float32) \
                        if "betas" in raw else np.zeros(10, np.float32)
                else:
                    from scipy.spatial.transform import Rotation
                    poses = np.asarray(raw["poses"], np.float32)
                    trans = np.asarray(raw["trans"], np.float32)
                    aa = poses[:, :66].reshape(-1, 22, 3)
                    matrices = Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix(
                    ).reshape(len(aa), 22, 3, 3)
                    basis = Rotation.from_euler("x", -90, degrees=True).as_matrix()
                    matrices[:, 0] = basis @ matrices[:, 0]
                    local = matrices[..., :, :2].transpose(0, 1, 3, 2).reshape(
                        len(aa), 22, 6).astype(np.float32)
                    root = ((trans @ basis.T) - (trans @ basis.T)[:1]) * 100.0
                    hands = poses[:, 66:156] if poses.shape[1] >= 156 else \
                        np.empty((len(poses), 0))
                    betas = np.asarray(raw["betas"], np.float32)[:10] \
                        if "betas" in raw else np.zeros(10, np.float32)
            from ..engine.spatial import build_global
            from ..viz.smplx_skin import skin_global_rot6d
            spec, _dof, _rest = POOL.human
            rotations, _pos, _reach = build_global(
                torch.from_numpy(local), spec, "cpu")
            vertices, faces = skin_global_rot6d(
                rotations.numpy(), spec.parents, root_cm=root,
                hand_pose=hands if hands.size else None,
                betas=betas, device=POOL.device)
            state["source_live"].set_smplx_skin(
                vertices, faces, name, fps=float(fps))
            return {**state["source_live"].transport_state(), "name": name,
                    "skin": "neutral SMPL-X", "source_kind": "project-native"}
        return work

    @app.post("/api/smpl-generations/{generation_id}/preview")
    async def preview_smpl_generation(generation_id: str):
        item = _smpl_generation(generation_id)
        result = await POOL.run(_preview_smpl_file(
            Path(item["local_path"]), item["name"], float(item.get("fps", 30.0))))
        return {**result, **item}

    @app.post("/api/projects/{project_id}/media/preview")
    async def preview_project_media(project_id: str, asset_id: str):
        try:
            project = state["projects"].get(project_id)
        except KeyError as exc:
            raise HTTPException(404, "project not found") from exc
        item = next((motion for motion in
                     (project.get("assets") or {}).get("motions", [])
                     if str(motion.get("asset_id")) == asset_id), None)
        if item is None or not item.get("local_path"):
            raise HTTPException(404, "project motion has no local SMPL source")
        path = Path(item["local_path"])
        if not path.is_file():
            raise HTTPException(404, "project motion source file is missing")
        return await POOL.run(_preview_smpl_file(
            path, item.get("name") or path.stem, float(item.get("fps", 30.0))))

    @app.post("/api/projects/{project_id}/smpl-generations/{generation_id}/commit")
    async def commit_smpl_generation(project_id: str, generation_id: str,
                                     payload: dict):
        item = _smpl_generation(generation_id, project_id)
        import_media = bool(payload.get("import_to_media"))
        share_library = bool(payload.get("share_to_library"))
        if not import_media and not share_library:
            raise HTTPException(422, "select Import to Media, Share to Library, or both")
        if import_media and not item.get("imported"):
            state["projects"].add_media(
                project_id, motions=[item], bin_name="GENMO SMPL")
            item["imported"] = True
        if share_library and not item.get("shared"):
            from ..config import CONFIG
            import sqlite3
            metadata = {"frames": item["frames"], "fps": item["fps"],
                        "project_id": project_id, "augmentation_type": "generated",
                        "augmentation_saved": True, "augmentation_draft": False,
                        "generation_method": item["generation_method"]}
            db = sqlite3.connect(CONFIG.data_studio_db)
            try:
                db.execute("""INSERT INTO assets
                  (id,source,title,folder,kind,format,locator_type,container,
                   inner_path,aux_json,animated,status,size,metadata_json)
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                  ON CONFLICT(id) DO UPDATE SET status='ready',
                  metadata_json=excluded.metadata_json""",
                  (item["asset_id"], "GENMO Shared", item["name"], "Generated",
                   "smplh_motion", "npz", "direct", item["local_path"], "", "{}",
                   1, "ready", Path(item["local_path"]).stat().st_size,
                   json.dumps(metadata)))
                db.commit()
            finally:
                db.close()
            item["shared"] = True
            asyncio.create_task(_sync_data_studio_library())
        state["smpl_generations"][generation_id] = item
        Path(item["local_path"]).with_suffix(".json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": True, "generation": item}

    @app.get("/api/library-datasets")
    def library_datasets():
        return {"datasets": state["library"].dataset_summary()}

    @app.get("/api/library-facets")
    def library_facets():
        from ..data_studio import catalog_summary
        return catalog_summary()

    @app.post("/api/library/resolve-assets")
    def library_resolve_assets(payload: dict):
        """Resolve Data Studio asset ids into Motion Studio library clips.

        Data Studio owns the durable asset identifiers while Motion Studio
        uses a dense integer clip index.  Keep that translation server-side so
        transferred bins survive filtering and never depend on display names.
        """
        requested = [str(value) for value in payload.get("asset_ids", [])
                     if value]
        lib = state["library"]
        exact = {asset_id: index for index, asset_id in enumerate(
            lib.asset_ids) if asset_id}

        # The Linux Data Studio index and the original imported Greenwich
        # shards can describe the same source clip with different database
        # ids.  Resolve those originals by their stable source/title identity
        # so sending an item does not require rebuilding 8k model shards.
        records: dict[str, tuple[str, str]] = {}
        unresolved = [value for value in requested if value not in exact]
        if unresolved:
            from ..config import CONFIG
            import sqlite3
            primary = Path(CONFIG.data_studio_db)
            if primary.is_file():
                db = sqlite3.connect(f"file:{primary}?mode=ro", uri=True)
                try:
                    placeholders = ",".join("?" for _ in unresolved)
                    for row in db.execute(
                            f"SELECT id,source,title FROM assets WHERE id IN ({placeholders})",
                            unresolved):
                        records[str(row[0])] = (str(row[1]), str(row[2]))
                finally:
                    db.close()

        def semantic_index(identity: str) -> int | None:
            record = records.get(identity)
            if record is None:
                return None
            source, title = record
            source_key = source.rsplit("/", 1)[-1].replace("AMASS ", "").strip()
            for index, name in enumerate(lib.names):
                clip_title = str(name).rsplit("__", 1)[-1]
                if clip_title == title and str(lib.sources[index]).strip() == source_key:
                    return index
            return None

        resolved: list[tuple[str, int]] = []
        for identity in requested:
            index = exact.get(identity)
            if index is None:
                index = semantic_index(identity)
            if index is not None:
                resolved.append((identity, index))

        items = [{
            "id": int(index), "name": lib.names[index],
            "family": lib.families[index], "dataset": lib.datasets[index],
            "source": lib.sources[index],
            "data_role": lib.data_roles[index],
            "asset_id": identity,
            "library_asset_id": lib.asset_ids[index],
            "origin_id": lib.origin_ids[index],
            "augmentation": lib.augmentations[index],
            "augmentation_value": lib.augmentation_values[index],
            "labels": lib.labels[index],
            "variant_count": lib.variant_counts[index],
            "frames": lib.frames(index), **lib.motion_metrics(index),
        } for identity, index in resolved]
        found = {identity for identity, _index in resolved}
        return {"ok": True, "items": items,
                "missing": [value for value in requested
                            if value not in found]}

    @app.get("/api/data-studio/status")
    def data_studio_status():
        from ..data_studio import catalog_summary
        manager = state.get("data_studio")
        runtime = manager.status() if manager is not None else {}
        state["data_studio_status"] = runtime
        return {**runtime,
                "catalog": catalog_summary(),
                "sync": state.get("data_studio_sync")}

    @app.post("/api/data-studio/sync")
    async def data_studio_sync():
        current = state.get("data_studio_sync")
        if not isinstance(current, dict) or current.get("status") != "running":
            asyncio.create_task(_sync_data_studio_library())
        return {"ok": True, "status": "syncing"}

    @app.get("/api/library/{library_id}/detail")
    def library_detail(library_id: int):
        """Lineage and review metadata for Motion Studio's asset inspector."""
        lib = state["library"]
        if library_id < 0 or library_id >= len(lib):
            raise HTTPException(404, "no such library clip")
        from ..data_studio import catalog_by_id
        catalog = catalog_by_id()
        asset_id = lib.asset_ids[library_id]
        record = catalog.get(asset_id) if asset_id else None
        origin = catalog.get(record["origin_id"]) if record else None
        variants = [catalog[value] for value in (record or {}).get(
            "variants", []) if value in catalog]
        if record and record["role"] == "augmented" and not variants:
            variants = [record]
        return {
            "id": library_id, "name": lib.names[library_id],
            "family": lib.families[library_id],
            "source": lib.sources[library_id],
            "frames": lib.frames(library_id),
            "data_role": lib.data_roles[library_id],
            "asset_id": asset_id, "origin_id": lib.origin_ids[library_id],
            "augmentation": lib.augmentations[library_id],
            "augmentation_value": lib.augmentation_values[library_id],
            "labels": lib.labels[library_id],
            "record": record, "origin": origin, "variants": variants,
        }

    def data_studio_target_id(library_id: int) -> str:
        detail = library_detail(library_id)
        record = detail.get("record") or {}
        if not record:
            return ""
        from ..config import CONFIG
        primary = Path(CONFIG.data_studio_db)
        if not primary.is_file():
            return ""
        import sqlite3
        db = None
        try:
            db = sqlite3.connect(f"file:{primary}?mode=ro", uri=True)
            row = db.execute(
                "SELECT id FROM assets WHERE title=? AND kind='smplh_motion' "
                "AND animated=1 AND status='ready' ORDER BY CASE WHEN "
                "source=? THEN 0 ELSE 1 END LIMIT 1",
                (record.get("title", ""), record.get("source", "")),
            ).fetchone()
            return str(row[0]) if row else ""
        except sqlite3.Error:
            return ""
        finally:
            if db is not None:
                db.close()

    async def _data_studio_contact_labels(target_id: str) -> dict:
        """Read one frame-aligned four-effector contact sidecar."""
        if not target_id:
            raise HTTPException(409, "contact labels are not indexed yet")
        from ..config import CONFIG
        base = f"http://127.0.0.1:{CONFIG.data_studio_port}"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(
                    base + "/api/contact-labels", params={"id": target_id})
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(503, "Data Studio labels unavailable") from exc
        if response.status_code >= 400:
            raise HTTPException(response.status_code,
                                payload.get("error", "label lookup failed"))
        if payload.get("status") != "ready" or not payload.get("labels"):
            raise HTTPException(404, "clip has no foot-contact labels")
        return payload["labels"]

    @app.get("/api/library/{library_id}/foot-contact")
    async def library_foot_contact(library_id: int):
        """Read the frame-aligned four-effector contact sidecar."""
        labels = await _data_studio_contact_labels(
            data_studio_target_id(library_id))
        return {"ok": True, "library_id": library_id,
                "labels": labels}

    @app.post("/api/library/{library_id}/label-foot-contact")
    async def library_label_foot_contact(library_id: int):
        """Run BodyDataStudio labeling, then expose its sidecar in Motion."""
        detail = library_detail(library_id)
        record = detail.get("record") or {}
        if not record:
            raise HTTPException(409, "clip has no Data Studio record")
        from ..config import CONFIG
        target_id = data_studio_target_id(library_id)
        if not target_id:
            raise HTTPException(
                409, "clip is still being indexed by Data Studio; retry shortly")
        base = f"http://127.0.0.1:{CONFIG.data_studio_port}"
        async with httpx.AsyncClient(timeout=120) as client:
            started = await client.post(
                base + "/api/contact-labels",
                json={"asset_ids": [target_id]})
            payload = started.json()
            if started.status_code >= 400 or not payload.get("job_id"):
                raise HTTPException(started.status_code, payload.get(
                    "error", "labeling could not start"))
            job = payload["job_id"]
            result = None
            for _ in range(600):
                await asyncio.sleep(.25)
                response = await client.get(
                    base + "/api/contact-labels", params={"job": job})
                current = response.json()
                if current.get("status") == "ready":
                    result = current.get("result") or {}
                    break
                if current.get("status") == "failed":
                    raise HTTPException(500, current.get(
                        "error", "contact labeling failed"))
            if result is None:
                raise HTTPException(504, "contact labeling timed out")
        await _sync_data_studio_library()
        return {"ok": True, "label": "foot_contact", "result": result}

    @app.get("/api/library/{library_id}/preview")
    async def library_preview(library_id: int):
        """Metadata for the neutral SMPL-X hover preview."""
        lib = state["library"]
        if library_id < 0 or library_id >= len(lib):
            raise HTTPException(404, "no such library clip")
        revision = _library_preview_key(lib, library_id)
        return {"id": int(library_id), "name": lib.names[library_id],
                "family": lib.families[library_id],
                "dataset": lib.datasets[library_id],
                "source": lib.sources[library_id],
                "frames": lib.frames(library_id), "fps": 30.0,
                "skin": ("neutral SMPL-X · exact source parameters"
                         if lib.source_motion(library_id) is not None else
                         "neutral SMPL-X · decoded 60-frame window"),
                "url": (f"/api/library/{library_id}/preview.webp"
                        f"?revision={revision}")}

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
        # The library is rebuilt and reordered as Data Studio publishes data.
        # A numeric library ID is therefore not a durable cache identity.
        revision = _library_preview_key(lib, library_id)
        path = cache_dir() / "smplx_previews" / f"v4-{revision}.webp"
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
        state["live"].clear_editor_gizmo()
        state["editor_gizmo"] = {"seq": state["editor_gizmo"].get("seq", 0) + 1,
                                 "active": False}
        return state["live"].clear_motion()

    @app.get("/api/viewer/gizmo")
    def editor_gizmo():
        return state["editor_gizmo"]

    @app.post("/api/viewer/gizmo")
    async def set_editor_gizmo(request: Request):
        payload = await request.json()
        mode = str(payload.get("mode", "position"))
        if mode not in {"position", "rotation"}:
            raise HTTPException(422, "gizmo mode must be position or rotation")
        position, wxyz = payload.get("position"), payload.get("wxyz")
        if position is not None and (len(position) != 3 or
                                     not np.isfinite(position).all()):
            raise HTTPException(422, "position must be three finite values")
        if wxyz is not None and (len(wxyz) != 4 or
                                 not np.isfinite(wxyz).all()):
            raise HTTPException(422, "wxyz must be four finite values")
        uid = str(payload.get("uid") or "")
        endpoint = str(payload.get("endpoint") or "start")
        if endpoint not in {"start", "end"}:
            raise HTTPException(422, "gizmo endpoint must be start or end")
        seq = int(state["editor_gizmo"].get("seq", 0)) + 1

        def changed(event):
            current = state["editor_gizmo"]
            pose = state["live"].editor_gizmo_state() or {}
            current.update({
                "seq": int(current.get("seq", 0)) + 1,
                "phase": event.phase,
                **pose,
            })

        pose = state["live"].set_editor_gizmo(
            mode, frame=max(0, int(payload.get("frame", 0))),
            position=position, wxyz=wxyz, on_update=changed)
        state["editor_gizmo"] = {"seq": seq, "active": True, "mode": mode,
                                 "uid": uid, "endpoint": endpoint,
                                 "phase": "ready", **pose}
        return state["editor_gizmo"]

    @app.delete("/api/viewer/gizmo")
    def clear_editor_gizmo():
        state["live"].clear_editor_gizmo()
        state["editor_gizmo"] = {"seq": int(state["editor_gizmo"].get("seq", 0)) + 1,
                                 "active": False}
        return state["editor_gizmo"]

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

    @app.post("/api/source/contact-overlay")
    async def source_contact_overlay(
            library_id: int | None = Query(default=None, ge=0),
            asset_id: str = Query(default="")):
        """Show dynamic heel/forefoot labels on the active source preview."""
        target_id = (data_studio_target_id(library_id)
                     if library_id is not None else str(asset_id).strip())
        labels = await _data_studio_contact_labels(target_id)
        if not labels.get("contact"):
            raise HTTPException(409, "foot-contact sidecar has no contact state")
        try:
            from ..viz.smplx_skin import sole_patch_indices
            state["source_live"].set_skin_contact_overlay(
                np.asarray(labels["contact"], np.uint8),
                sole_patch_indices(POOL.device),
                np.asarray(labels.get("contact_type"), np.uint8)
                if labels.get("contact_type") is not None else None,
                label_fps=float(labels.get("fps") or 0.0))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {**state["source_live"].transport_state(),
                "label": "foot_contact", "effectors": [
                    "left_heel", "left_forefoot",
                    "right_heel", "right_forefoot"]}

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
        never ask an external perception subprocess to read an arbitrary host
        file.
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
        try:
            from imageio_ffmpeg import count_frames_and_secs
            source_frames, duration = count_frames_and_secs(str(path))
            source_frames = int(source_frames)
            duration = float(duration)
            if source_frames < 1 or duration <= 0:
                raise ValueError("no decodable video frames")
            source_fps = source_frames / duration
        except Exception as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(
                422, f"video metadata could not be decoded: {exc}") from exc
        with session() as s:
            s.add(Asset(kind="upload", path=str(path), bytes=total))
            s.commit()
        return {"asset": token, "name": Path(file.filename or token).name,
                "bytes": total, "source_frames": source_frames,
                "source_fps": source_fps, "duration_seconds": duration}

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

    def _motion_playback_codes(motion_id: int):
        """Encode a durable robot trace back into the shared motion space."""
        cache = state["motion_codes"]
        if motion_id in cache:
            cache.move_to_end(motion_id)
            return cache[motion_id]
        with session() as s:
            motion = s.get(Motion, motion_id)
            if motion is None:
                raise ValueError(f"no such motion: {motion_id}")
            title, family, trace_path = (
                motion.title, motion.family, Path(motion.trace_path))
        if not trace_path.is_file():
            raise ValueError(f"motion trace is missing: {trace_path.name}")
        from ..embodiment import registry
        from ..engine import constraints as MP
        from ..engine.nets.rotations import matrix_to_rot6d
        from ..engine.trace import MotionTrace
        trace = MotionTrace.load(trace_path)
        source_body = registry.load(trace.target)
        gR, _gp = MP.fk_from_angles(
            torch.as_tensor(trace.q, device=POOL.device, dtype=torch.float64),
            source_body.spec,
            torch.as_tensor(source_body.dof, device=POOL.device,
                            dtype=torch.float64),
            rest=torch.as_tensor(source_body.rest, device=POOL.device,
                                 dtype=torch.float64),
            root_R=torch.as_tensor(trace.rootR, device=POOL.device,
                                   dtype=torch.float64))
        pose9, _ = POOL.greenwich.pose9(
            matrix_to_rot6d(gR).float().cpu(), source_body.spec,
            is_global=True)
        codes = torch.cat([
            POOL.greenwich.encode(chunk, source_body.spec, source_body.dof)
            .detach()
            for chunk in pose9.split(MODEL_FRAME_BATCH)
        ])
        root = None
        if trace.root_t is not None:
            root = np.asarray(trace.root_t, np.float64)
            root = root - root[:1]
        value = (codes, title, family, root)
        cache[motion_id] = value
        cache.move_to_end(motion_id)
        while len(cache) > 3:
            cache.popitem(last=False)
        return value

    def _rotation_matrix(wxyz) -> np.ndarray:
        if wxyz is None:
            return np.eye(3, dtype=np.float64)
        from scipy.spatial.transform import Rotation
        w, x, y, z = wxyz
        return Rotation.from_quat([x, y, z, w]).as_matrix()

    def _segment_rotation_path(seg: Segment, frames: int) -> np.ndarray:
        """Interpolate the selected clip's start/end world orientation."""
        start = _rotation_matrix(seg.world_rotation_wxyz)
        end_value = seg.world_end_rotation_wxyz
        if end_value is None or frames <= 1:
            return np.repeat(start[None], max(1, frames), axis=0)
        from scipy.spatial.transform import Rotation, Slerp
        end = _rotation_matrix(end_value)
        key = Rotation.from_matrix(np.stack([start, end]))
        return Slerp([0.0, 1.0], key)(
            np.linspace(0.0, 1.0, frames)).as_matrix()

    def _segment_rotation(seg: Segment) -> np.ndarray:
        """Start orientation retained for bridge-boundary compatibility."""
        return _rotation_matrix(seg.world_rotation_wxyz)

    def _segment_world_path(seg: Segment, root, frames: int,
                            anchor_world: np.ndarray):
        """Return a clip root path and orientation in Viser's Z-up world."""
        from ..viz.kinematics import root_world_offsets
        local = root_world_offsets(root, frames)
        rotations = _segment_rotation_path(seg, frames)
        local = np.einsum("tij,tj->ti", rotations, local)
        start = anchor_world if seg.world_position_m is None else \
            np.asarray(seg.world_position_m, np.float64)
        absolute = start[None] + local
        if seg.world_end_position_m is not None:
            target = np.asarray(seg.world_end_position_m, np.float64)
            u = np.linspace(0.0, 1.0, frames)[:, None]
            smooth = u * u * (3.0 - 2.0 * u)
            absolute += smooth * (target - absolute[-1])
        return absolute, rotations

    def _world_path_to_root_cm(world: np.ndarray) -> np.ndarray:
        world = np.asarray(world, np.float64)
        return np.stack([world[:, 1], world[:, 2], world[:, 0]], axis=1) * 100.0

    def _trace_root_from_world(world: np.ndarray):
        from ..viz.kinematics import world_offsets_to_root_cm
        world = np.asarray(world, np.float64)
        return world_offsets_to_root_cm(world), world[0].copy()

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
        if seg.kind == "motion":
            codes, name, fam, root = _motion_playback_codes(seg.motion_id)
            base_n = seg.source_frames or len(codes)
            if base_n != len(codes):
                from ..engine.timeline import (interpolate_lattice,
                                               resample_continuous)
                source = codes
                tokens, ep = eq.tokenize(source)
                rot_codes = interpolate_lattice(source[:, 128:], base_n)
                boundary = torch.stack([source[0], source[-1]])
                codes = eq.detokenize(tokens, ep, base_n,
                                      boundary_codes=boundary,
                                      rot_codes=rot_codes)
                if root is not None:
                    root = resample_continuous(root, base_n)
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
                       root_t=None, fps: float = 30.0, root_origin_m=None,
                       root_rotation_delta=None, root_path_locked=False):
        """Decode a source clip for the live viewer without creating a Motion.

        Timeline scrubbing is an editor operation, not a generation request;
        previews therefore stay in a small in-memory LRU instead of polluting
        SQLite, Atlas memory, result files, or quality-control history.
        """
        from ..embodiment import registry
        from ..engine import constraints as MP
        from ..engine.nets.rotations import rot6d_to_matrix
        from ..engine.trace import MotionTrace
        from ..viz.kinematics import (
            balanced_root_rotations, stabilize_trace_root_translation,
            zup_world_rotations_to_yup)

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
        if root_rotation_delta is not None:
            delta = np.asarray(root_rotation_delta, np.float64)
            if delta.shape != rootR.shape:
                raise ValueError("root rotation edits must have shape [T,3,3]")
            rootR = zup_world_rotations_to_yup(delta) @ rootR
        trace = MotionTrace(
            q=q.detach().cpu().numpy(), rootR=rootR,
            gp=gp.detach().cpu().numpy(),
            stage=np.zeros(len(codes), np.int32), fps=fps, title=title,
            target=target_body, joint_names=list(emb.spec.joint_names),
            root_t=None if root_t is None else np.asarray(root_t, np.float64),
            root_origin_m=None if root_origin_m is None else
            np.asarray(root_origin_m, np.float64),
            root_path_locked=bool(root_path_locked))
        if (emb.xml and Path(emb.xml).is_file()
                and not trace.root_path_locked):
            trace.root_t, report = stabilize_trace_root_translation(
                trace, emb.xml, target_body)
            trace.contact_stabilized = bool(report.get("available", False))
        return trace, emb

    def _finalize(codes, target_body, title, source, prompt=None,
                  parent=None, render=True, fps=30.0, se3=(), root_t=None,
                  family=None, stage=None, root_origin_m=None,
                  root_rotation_delta=None, root_path_locked=False):
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
        if root_rotation_delta is not None:
            delta = np.asarray(root_rotation_delta, np.float64)
            if delta.shape != rootR.shape:
                raise ValueError("root rotation edits must have shape [T,3,3]")
            from ..viz.kinematics import zup_world_rotations_to_yup
            rootR = zup_world_rotations_to_yup(delta) @ rootR
        mid_title = title or f"{source}-{int(time.time())}"
        trace = MotionTrace(q=q.detach().cpu().numpy(), rootR=rootR,
                            gp=gp.detach().cpu().numpy(),
                            stage=stage, fps=fps, title=mid_title,
                            target=target_body,
                            tokens=tok_final.cpu().numpy(),
                            joint_names=list(emb.spec.joint_names),
                            root_t=None if root_t is None
                            else np.asarray(root_t, np.float64),
                            root_origin_m=None if root_origin_m is None else
                            np.asarray(root_origin_m, np.float64),
                            root_path_locked=bool(root_path_locked))
        # Bake contact correction into new traces so downloaded assets, the
        # live viewer, and MP4 export share the same target-specific world
        # trajectory. Older traces receive the same correction at render time.
        contact_report = {"available": False, "reason": "no target mesh"}
        if emb.xml and not trace.root_path_locked:
            try:
                from ..viz.kinematics import stabilize_trace_root_translation
                trace.root_t, contact_report = stabilize_trace_root_translation(
                    trace, emb.xml, target_body)
                trace.contact_stabilized = bool(
                    contact_report.get("available", False))
            except Exception as exc:  # rendering remains usable without it
                contact_report = {"available": False,
                                  "reason": f"contact solve failed: {exc}"}
        qc["contact_stability"] = contact_report
        ai_method = {"text_prompt": "text-to-motion",
                     "video": "video-to-motion",
                     "ai_mixed": "multimodal-motion"}.get(source)
        if ai_method:
            # This metadata makes generated motions durable, queryable assets
            # instead of anonymous entries that only happen to appear in Recent.
            qc["generation"] = {"ai_generated": True,
                                "method": ai_method,
                                "auto_saved": True}
        rrep["contact_stabilization"] = contact_report
        if contact_report.get("available"):
            release_ok = bool(release_ok and contact_report.get("passed"))
            qc["release_passed"] = release_ok
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
        if ai_method:
            out["ai_asset"] = {"saved": True, "method": ai_method,
                               "collection": "AI Generations"}
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

    def _require_motion_id(index: int):
        with session() as s:
            if s.get(Motion, index) is None:
                raise HTTPException(404, f"no such motion: {index}")

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
        if any(seg.kind not in ("library", "motion", "gap")
               for seg in req.segments):
            raise HTTPException(
                422, "prompt and video segments must be generated before preview")
        for seg in req.segments:
            if seg.kind == "library":
                _require_library_id(lib, seg.library_id)
            elif seg.kind == "motion":
                _require_motion_id(seg.motion_id)
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
                chunks, world_paths, rotations = [], [], []
                anchor_world = np.zeros(3, np.float64)
                have_roots = True
                root_path_locked = any(
                    seg.world_position_m is not None or
                    seg.world_end_position_m is not None
                    for seg in req.segments
                    if seg.kind in ("library", "motion"))
                for seg in req.segments:
                    if seg.kind == "gap":
                        if not chunks:
                            raise ValueError("a bridge needs a clip on its left")
                        chunks.append(chunks[-1][-1:].repeat(seg.n, 1, 1))
                        if have_roots:
                            world_paths.append(np.repeat(
                                anchor_world[None], seg.n, axis=0))
                        rotations.append(np.repeat(
                            rotations[-1][-1: ], seg.n, axis=0)
                            if rotations else
                            np.repeat(np.eye(3)[None], seg.n, axis=0))
                        continue
                    codes, _name, _fam, root = _segment_codes(
                        seg, POOL.equator, lib, req.target_body)
                    chunks.append(codes)
                    if root is None:
                        have_roots = False
                        world_paths.clear()
                    elif have_roots:
                        absolute, _rotation_path = _segment_world_path(
                            seg, root, len(codes), anchor_world)
                        world_paths.append(absolute)
                        anchor_world = absolute[-1].copy()
                    rotations.append(_segment_rotation_path(seg, len(codes)))
                codes = torch.cat(chunks, 0)
                root_t, root_origin_m = (None, None)
                if have_roots and world_paths:
                    root_t, root_origin_m = _trace_root_from_world(
                        np.concatenate(world_paths, 0))
                trace, _ = _preview_trace(
                    codes, req.target_body, req.title or "Timeline preview",
                    root_t=root_t, fps=req.fps, root_origin_m=root_origin_m,
                    root_rotation_delta=np.concatenate(rotations, 0),
                    root_path_locked=root_path_locked)
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
            elif seg.kind == "motion":
                _require_motion_id(seg.motion_id)
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
                                       "generated": True, "segment": seg})
                        names.append(name); families.append(fam)
                        if seg.text:
                            prompts.append(seg.text)
                        continue
                    codes, name, fam, root = _segment_codes(
                        seg, eq, lib, req.target_body)
                    chunks.append({"kind": "codes", "codes": codes,
                                   "root": root, "family": fam,
                                   "generated": bool(seg.pins),
                                   "segment": seg})
                    names.append(name); families.append(fam)
                # Resolve gaps between code chunks. Each source root path is
                # first-frame anchored; bridge_root carries boundary velocity
                # through generated interiors before re-anchoring the next.
                from collections import Counter
                from ..engine.timeline import bridge_root
                out, roots, stages, rotation_edits = [], [], [], []
                anchor = np.zeros(3)
                have_root = all(c.get("root") is not None
                                for c in chunks if c["kind"] == "codes")
                has_world_edits = any(
                    c["segment"].world_position_m is not None or
                    c["segment"].world_rotation_wxyz is not None or
                    c["segment"].world_end_position_m is not None or
                    c["segment"].world_end_rotation_wxyz is not None
                    for c in chunks if c["kind"] == "codes")
                root_path_locked = any(
                    c["segment"].world_position_m is not None or
                    c["segment"].world_end_position_m is not None
                    for c in chunks if c["kind"] == "codes")
                anchor_world = np.zeros(3, np.float64)
                for i, chunk in enumerate(chunks):
                    if chunk["kind"] == "codes":
                        val, root = chunk["codes"], chunk["root"]
                        out.append(val)
                        stages.append(np.full(len(val),
                                              1 if chunk["generated"] else 0,
                                              np.int32))
                        if have_root:
                            if has_world_edits:
                                absolute, _rotation_path = _segment_world_path(
                                    chunk["segment"], root, len(val),
                                    anchor_world)
                                roots.append(absolute)
                                anchor_world = absolute[-1]
                            else:
                                absolute = anchor + root
                                roots.append(absolute)
                                anchor = absolute[-1]
                        rotation_edits.append(_segment_rotation_path(
                            chunk["segment"], len(val)))
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
                        if has_world_edits:
                            prev_cm = _world_path_to_root_cm(roots[-1])
                            next_local, _ = _segment_world_path(
                                nxt_chunk["segment"], nxt_chunk["root"],
                                len(nxt_chunk["codes"]), np.zeros(3))
                            next_cm = _world_path_to_root_cm(
                                next_local - next_local[0])
                            gap_cm, next_anchor_cm = bridge_root(
                                prev_cm, next_cm, seg.n)
                            gap_world = np.stack(
                                [gap_cm[:, 2], gap_cm[:, 0], gap_cm[:, 1]],
                                axis=1) / 100.0
                            explicit = nxt_chunk["segment"].world_position_m
                            if explicit is not None:
                                target = np.asarray(explicit, np.float64)
                                u = np.arange(1, seg.n + 1)[:, None] / (seg.n + 1)
                                smooth = u * u * (3.0 - 2.0 * u)
                                gap_world = (1.0 - smooth) * anchor_world + smooth * target
                                anchor_world = target
                            else:
                                anchor_world = np.asarray(
                                    [next_anchor_cm[2], next_anchor_cm[0],
                                     next_anchor_cm[1]]) / 100.0
                            roots.append(gap_world)
                        else:
                            gap_root, anchor = bridge_root(
                                roots[-1], nxt_chunk["root"], seg.n)
                            roots.append(gap_root)
                    rotation_edits.append(np.repeat(
                        rotation_edits[-1][-1: ], seg.n, axis=0))
                codes = torch.cat(out, 0)
                root_origin_m = None
                if have_root and roots:
                    if has_world_edits:
                        root_t, root_origin_m = _trace_root_from_world(
                            np.concatenate(roots, 0))
                    else:
                        root_t = np.concatenate(roots, 0)
                else:
                    root_t = None
                title = req.title or " + ".join(names[:3])[:240]
                family = Counter(families).most_common(1)[0][0] \
                    if families else "other"
                kinds = {seg.kind for seg in req.segments}
                source = "text_prompt" if kinds == {"prompt"} else \
                    "video" if kinds == {"video"} else \
                    "ai_mixed" if kinds & {"prompt", "video"} else "edit"
                return _finalize(codes, req.target_body, title, source,
                                 render=req.render, fps=req.fps, se3=req.se3,
                                 root_t=root_t, family=family,
                                 root_origin_m=root_origin_m,
                                 root_rotation_delta=np.concatenate(
                                     rotation_edits, 0),
                                 root_path_locked=root_path_locked,
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
            rot6d, root_t = motion_from_video(str(video), seg.n)
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
    def motions(limit: int = Query(default=50, ge=1, le=200),
                ai_only: bool = Query(default=False)):
        with session() as s:
            query = s.query(Motion)
            if ai_only:
                query = query.filter(Motion.source.in_(
                    ("text_prompt", "video", "ai_mixed")))
            rows = query.order_by(Motion.id.desc()).limit(limit)
            return {"motions": [
                {"id": m.id, "title": m.title, "family": m.family,
                 "frames": m.n_frames, "source": m.source,
                 "gate_ratio": m.gate_ratio, "gate_passed": m.gate_passed,
                 "release_passed": release_passed(m.gate_passed, m.qc),
                 "trace": Path(m.trace_path).name,
                 "prompt": m.prompt,
                 "created_at": m.created_at.isoformat(),
                 "ai_generated": bool((m.qc or {}).get(
                     "generation", {}).get("ai_generated")) or
                     m.source in ("text_prompt", "video", "ai_mixed"),
                 "generation_method": (m.qc or {}).get(
                     "generation", {}).get("method"),
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

    # -------------------------------------------- BodyDataStudio proxy -----
    # Keep the mature data worker isolated while presenting one same-origin
    # AlphaMotion application. Rewriting its absolute asset/API paths lets the
    # complete existing UI run below /data-studio-app/ without colliding with
    # AlphaMotion's own /api namespace.
    @app.get("/data-studio-app")
    async def data_studio_slash():
        return RedirectResponse("/data-studio-app/")

    @app.api_route(
        "/data-studio-app/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def data_studio_proxy(request: Request, path: str):
        from ..config import CONFIG
        url = f"http://127.0.0.1:{CONFIG.data_studio_port}/{path}"
        if request.url.query:
            url += "?" + request.url.query
        headers = {}
        for key in ("content-type", "accept", "cookie"):
            if key in request.headers:
                headers[key] = request.headers[key]
        body = await request.body()
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                upstream = await client.request(
                    request.method, url, content=body,
                    headers=headers)
        except httpx.HTTPError as exc:
            return Response(
                content=f"Data Studio unavailable: {exc}", status_code=503,
                media_type="text/plain")
        content = upstream.content
        content_type = upstream.headers.get("content-type", "")
        # Rewrite executable/static documents only. JSON contains real source
        # paths such as /media/...; changing those strings corrupts asset
        # locators and makes the embedded library appear empty.
        if any(kind in content_type for kind in (
                "text/html", "text/css", "javascript")):
            text = content.decode(upstream.encoding or "utf-8", errors="replace")
            for prefix in ("api", "media", "assets", "vendor"):
                text = text.replace(
                    f"'/{prefix}/", f"'/data-studio-app/{prefix}/")
                text = text.replace(
                    f'"/{prefix}/', f'"/data-studio-app/{prefix}/')
                text = text.replace(
                    f'`/{prefix}/', f'`/data-studio-app/{prefix}/')
            # BodyDataStudio keeps its two entry assets at the web root while
            # its remaining module imports are relative to app.js.
            text = text.replace('href="/style.css',
                                'href="/data-studio-app/style.css')
            text = text.replace('src="/app.js',
                                'src="/data-studio-app/app.js')
            text = text.replace("location.replace('/')",
                                "location.replace('/data-studio-app/')")
            content = text.encode("utf-8")
        response_headers = {"cache-control": "no-cache"}
        if "set-cookie" in upstream.headers:
            response_headers["set-cookie"] = upstream.headers["set-cookie"]
        should_sync = path == "api/augmentation-save"
        if path == "api/process" and body:
            try:
                should_sync = json.loads(body).get("action") == "save"
            except (ValueError, TypeError, AttributeError):
                pass
        if should_sync and 200 <= upstream.status_code < 300:
            asyncio.create_task(_sync_data_studio_library())
        return Response(content=content, status_code=upstream.status_code,
                        media_type=content_type or None,
                        headers=response_headers)

    # Data Studio API responses intentionally keep canonical `/media/...`
    # URLs.  When the application is embedded those URLs land on AlphaMotion,
    # so proxy the media namespace as well.  Without this route generated GLB
    # previews return 404 even though the files exist in BodyDataStudio.
    @app.api_route("/media/{path:path}", methods=["GET", "HEAD"])
    async def data_studio_media_proxy(request: Request, path: str):
        return await data_studio_proxy(request, f"media/{path}")

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
