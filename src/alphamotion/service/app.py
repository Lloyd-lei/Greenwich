"""AlphaMotion service gateway.

Warm pool + async job runner + SQLite persistence. Every generated motion goes
through: assemble codes -> decode on target -> refine -> synergy gate -> QC ->
trace asset -> DB row -> atlas registration + edges. Jobs survive restarts in
the DB; results are files under data_dir()/results.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from fastapi import (FastAPI, File, HTTPException, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..atlas.families import FAMILIES, family_id, family_of
from ..paths import data_dir, results_dir
from .db import Asset, AtlasEdge, Job, Motion, Skeleton, session
from .pool import POOL
from .schemas import JumpRequest, PlayRequest, Segment, TimelineRequest

FRONTEND = Path(__file__).parent.parent / "assets" / "frontend"

_viewers: dict[int, object] = {}          # port -> subprocess.Popen


def _launch_viewer(trace_path: Path, xml: str, body: str) -> str | None:
    """Per-result viser viewer from the configured port pool."""
    import subprocess
    import sys

    from ..config import CONFIG
    lo, hi = CONFIG.viewer_ports
    for port, proc in list(_viewers.items()):
        if proc.poll() is not None:
            _viewers.pop(port)
    free = [p for p in range(lo, hi) if p not in _viewers]
    if not free:
        oldest = next(iter(_viewers))
        _viewers.pop(oldest).terminate()
        free = [oldest]
    port = free[0]
    proc = subprocess.Popen(
        [sys.executable, "-m", "alphamotion.viz.viewer", "--trace",
         str(trace_path), "--xml", xml, "--body", body, "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    _viewers[port] = proc
    return f"http://127.0.0.1:{port}/"


def create_app() -> FastAPI:
    app = FastAPI(title="AlphaMotion", version="0.1.0")
    state: dict = {"library": None, "warm": None}

    @app.on_event("startup")
    async def _startup():
        from ..atlas.library import load_default as load_library
        from ..config import CONFIG
        from ..viz.live import LiveViewer
        state["warm"] = POOL.warm()
        state["library"] = load_library()
        state["live"] = LiveViewer(CONFIG.viewer_ports[0])

    # ------------------------------------------------------------- meta -----
    @app.get("/api/health")
    def health():
        return {"ok": POOL.greenwich is not None, "warm": state["warm"],
                "library": len(state["library"]) if state["library"] else 0,
                "viewer": state["live"].url if state.get("live") else None}

    @app.get("/api/bodies")
    def bodies():
        from ..embodiment import registry
        out = []
        for n in registry.bundled_names():
            out.append({"name": n, "source": "bundled"})
        for n in registry.user_names():
            out.append({"name": n, "source": "user"})
        return {"bodies": out}

    @app.get("/api/bodies/{name}")
    def body_detail(name: str):
        from ..embodiment import registry
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
        return {"name": name, "joints": emb.spec.J,
                "joint_names": list(emb.spec.joint_names),
                "height_cm": round(emb.spec.height, 1),
                "source": emb.source, **meta}

    @app.get("/api/library")
    def library(q: str = "", family: str = "", offset: int = 0,
                limit: int = 24):
        return state["library"].search(q, family, offset, limit)

    @app.get("/api/families")
    def families():
        return {"families": FAMILIES}

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
                motion_id=result.get("motion_id"))
        except Exception as exc:  # noqa: BLE001
            upd(status="failed", error=str(exc)[:2000])

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
        if not str(p).startswith(str(results_dir().resolve())) \
                or not p.is_file():
            raise HTTPException(404, "no such result")
        return FileResponse(p)

    # ---------------------------------------------------------- pipeline ----
    def _segment_codes(seg: Segment, eq, lib):
        """One timeline segment -> (codes [n,256,20] on device)."""
        if seg.kind == "library":
            tok, bounds, name, fam = lib.entry(seg.library_id)
            ep = eq.endpoints_from_codes(torch.from_numpy(bounds))
            tokens = torch.from_numpy(tok).to(eq.device)
            if seg.pins:
                for slot, val in seg.pins.items():
                    tokens[int(slot)] = int(val)
            return eq.detokenize(tokens, ep, seg.n), name
        raise ValueError(f"segment kind '{seg.kind}' not handled here")

    def _bridge_codes(eq, prev_codes, next_codes, n, seed, temperature):
        """Equator bridge between two code chunks (their boundary frames)."""
        codes4 = torch.cat([prev_codes[-2:], next_codes[:2]], 0)
        ep = eq.endpoints_from_codes(codes4.cpu())
        tok = eq.sample_tokens(ep, n, temperature=temperature, seed=seed)
        return eq.detokenize(tok, ep, n,
                             boundary_codes=torch.stack(
                                 [prev_codes[-1], next_codes[0]]))

    def _finalize(codes, target_body, title, source, prompt=None,
                  parent=None, render=True, fps=30.0, se3=()):
        """codes -> decode -> refine -> gate -> QC -> trace -> DB -> atlas."""
        from ..embodiment import registry
        from ..engine import constraints as MP
        from ..engine.nets.rotations import rot6d_to_matrix
        from ..engine.spatial import fk_pos
        from ..engine.trace import MotionTrace
        from ..refiner.refine import Refiner
        from ..utils import metrics
        gw, eq, atlas = POOL.greenwich, POOL.equator, POOL.atlas
        hspec, hdof, _hrest = POOL.human
        emb = registry.load(target_body)
        rot = gw.decode(codes, emb.spec, emb.dof)
        rot_h = gw.decode(codes, hspec, hdof)
        # RENDER PATH = the raw spatial-token decode, nothing else (owner
        # call 0814: no refinement of any kind). fit_angles(clamp=False,
        # greedy) is a pure kinematic CONVERSION of the decoded global
        # rotations into the mesh's hinge coordinates — no optimisation, no
        # limit clamping, no smoothing.
        Rg = rot6d_to_matrix(rot).double()
        dof_t = torch.as_tensor(emb.dof, device=POOL.device,
                                dtype=torch.float64)
        rest_t = torch.as_tensor(emb.rest, device=POOL.device,
                                 dtype=torch.float64)
        q, _ = MP.fit_angles(Rg, emb.spec, dof_t, rest=rest_t,
                             clamp=False, method="greedy")
        refined = rot                       # ship the decode itself
        rrep = {"refiner": "none (raw spatial-token decode)"}
        # SE3 constrained re-projection on requested spans
        for c in se3:
            Rg = rot6d_to_matrix(refined).double()
            sl = slice(c.frame_start, min(c.frame_end, len(refined)))
            r6, _p, q2 = MP.project_constrained(
                Rg[sl], emb.spec,
                torch.as_tensor(emb.dof, device=POOL.device,
                                dtype=torch.float64),
                joints=(c.joint,),
                target_pos=None, rest=torch.as_tensor(
                    emb.rest, device=POOL.device, dtype=torch.float64))
            refined[sl] = r6.float()
            q[sl] = q2
        # synergy gate vs the pre-refine decode's own tokens
        from ..refiner.synergy import synergy_gate
        p9_src, _ = gw.pose9(rot_h.cpu(), hspec, is_global=True)
        gate = synergy_gate(gw, eq, p9_src.to(POOL.device), hspec, hdof,
                            refined, emb.spec, emb.dof)
        qc = metrics.arm_qc(rot_h[..., :6].cpu().numpy(), hspec,
                            refined.cpu().numpy(), emb.spec)
        tok_final, _ep = eq.tokenize(codes)
        # trace + assets
        gp = fk_pos(refined.cpu().numpy(), emb.spec)
        rootR = rot6d_to_matrix(refined[:, 0:1, :6]).cpu().numpy()[:, 0]
        stage = np.ones(len(refined), np.int32)
        mid_title = title or f"{source}-{int(time.time())}"
        trace = MotionTrace(q=q.cpu().numpy(), rootR=rootR, gp=gp,
                            stage=stage, fps=fps, title=mid_title,
                            target=target_body,
                            tokens=tok_final.cpu().numpy(),
                            joint_names=list(emb.spec.joint_names))
        tp = results_dir() / f"{uuid.uuid4().hex[:10]}_trace.npz"
        trace.save(tp)
        fam = family_of(prompt or mid_title)
        with session() as s:
            m = Motion(title=mid_title, family=fam,
                       duration_s=len(refined) / fps, fps=fps,
                       n_frames=len(refined), source=source, prompt=prompt,
                       parent_motion_id=parent,
                       tokens=[int(t) for t in tok_final.cpu()],
                       trace_path=str(tp), gate_ratio=gate.ratio,
                       gate_passed=gate.passed,
                       qc={"arm": qc, "refiner": rrep})
            s.add(m)
            s.commit()
            motion_id = m.id
            s.add(Asset(motion_id=motion_id, kind="trace", path=str(tp),
                        bytes=tp.stat().st_size))
            # atlas registration + materialized edges
            w = atlas.add(tok_final.cpu().numpy(), mid_title, family_id(fam))
            for slot in (4, 12, 20, 28):
                for pdct in atlas.portals(tok_final.cpu().numpy(), slot, k=2,
                                          exclude_clip=int(atlas.clip[w])):
                    s.add(AtlasEdge(src_motion_id=motion_id, src_slot=slot,
                                    dst_window=pdct["window"],
                                    dst_clip=pdct["clip"],
                                    dst_family=pdct["family"],
                                    score=pdct["score"]))
            s.commit()
        out = {"motion_id": motion_id, "frames": len(refined),
               "trace": tp.name, "gate": gate.as_dict(), "qc": qc,
               "refiner": rrep, "family": fam,
               "tokens": [int(t) for t in tok_final.cpu()]}
        if render:
            mp4 = _try_render(tp, target_body)
            if mp4:
                out["mp4"] = mp4
                with session() as s:
                    s.add(Asset(motion_id=motion_id, kind="mp4",
                                path=str(results_dir() / mp4)))
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
    @app.post("/api/jobs/play", status_code=202)
    async def play(req: PlayRequest):
        lib = state["library"]

        async def run(_id):
            def work():
                eq = POOL.equator
                seg = Segment(kind="library", library_id=req.library_id,
                              n=req.n or lib.window)
                codes, name = _segment_codes(seg, eq, lib)
                return _finalize(codes, req.target_body, name, "library",
                                 render=req.render)
            return await POOL.run(work)
        return {"job_id": _submit("play", req.model_dump(), run)}

    @app.post("/api/jobs/timeline", status_code=202)
    async def timeline(req: TimelineRequest):
        lib = state["library"]
        if not req.segments:
            raise HTTPException(422, "empty timeline")

        async def run(_id):
            def work():
                eq = POOL.equator
                chunks, names = [], []
                for seg in req.segments:
                    if seg.kind == "gap":
                        chunks.append(("gap", seg))
                        continue
                    if seg.kind in ("prompt", "video"):
                        codes = _perception_codes(seg)
                        chunks.append(("codes", codes))
                        names.append(seg.text or seg.video_asset or "clip")
                        continue
                    codes, name = _segment_codes(seg, eq, lib)
                    chunks.append(("codes", codes))
                    names.append(name)
                # resolve gaps between neighbouring code chunks
                out = []
                for i, (kind, val) in enumerate(chunks):
                    if kind == "codes":
                        out.append(val)
                        continue
                    prev = out[-1] if out else None
                    nxt = next((v for k, v in chunks[i + 1:] if k == "codes"),
                               None)
                    if prev is None or nxt is None:
                        raise ValueError("gap segment needs neighbours on "
                                         "both sides")
                    out.append(_bridge_codes(eq, prev, nxt, val.n, val.seed,
                                             val.temperature))
                codes = torch.cat(out, 0)
                title = req.title or " + ".join(n[:24] for n in names[:3])
                return _finalize(codes, req.target_body, title, "edit",
                                 render=req.render, fps=req.fps, se3=req.se3)
            return await POOL.run(work)
        return {"job_id": _submit("timeline", req.model_dump(), run)}

    @app.post("/api/jobs/jump", status_code=202)
    async def jump(req: JumpRequest):
        """Portal jump: current motion -> bridge -> destination library clip.
        The atlas differentiator made playable."""
        lib = state["library"]

        async def run(_id):
            def work():
                eq = POOL.equator
                with session() as s:
                    src = s.get(Motion, req.motion_id)
                if not src:
                    raise ValueError("no such motion")
                src_trace = MotionTraceLoader(src.trace_path)
                # re-derive source codes from its tokens via its own trace?
                # source codes: regenerate from stored tokens + its endpoints
                # (the trace stores tokens; endpoints from its boundary frames
                # are not stored, so bridge from the SOURCE's last frames by
                # re-encoding its tail on the human body is the honest path)
                raise NotImplementedError
            return await POOL.run(work)
        # jump v1: implemented client-side as timeline [src_lib, gap, dst_lib]
        raise HTTPException(501, "use /api/jobs/timeline with a gap segment; "
                                 "native jump lands in v0.2")

    def _perception_codes(seg: Segment):
        from ..perception.genmo import motion_from_prompt, motion_from_video
        gw = POOL.greenwich
        hspec, hdof, _ = POOL.human
        if seg.kind == "prompt":
            rot6d = motion_from_prompt(seg.text, seg.n / 30.0)
        else:
            rot6d = motion_from_video(seg.video_asset)
        p9, _ = gw.pose9(rot6d, hspec, is_global=True)
        return gw.encode(p9.to(POOL.device), hspec, hdof)

    # ----------------------------------------------------------- ingest -----
    @app.post("/api/bodies/ingest", status_code=202)
    async def ingest_urdf(file: UploadFile = File(...), name: str = ""):
        raw = await file.read()
        up = data_dir() / "uploads" / f"{int(time.time())}_{file.filename}"
        up.parent.mkdir(parents=True, exist_ok=True)
        up.write_bytes(raw)

        async def run(_id):
            def work():
                from ..embodiment.urdf_ingest import ingest
                rep = ingest(up, name or None, device=POOL.device)
                with session() as s:
                    s.add(Skeleton(name=rep["name"], kind="user_urdf",
                                   joints=rep["joints"],
                                   height_cm=rep["height_cm"],
                                   xml_path=rep["mjcf"],
                                   sem_labels=rep["semantics"],
                                   limit_report=rep["limits"]))
                    s.commit()
                return rep
            return await POOL.run(work)
        return {"job_id": _submit("ingest", {"file": file.filename}, run)}

    # ------------------------------------------------------------ atlas -----
    @app.get("/api/atlas/portals/{motion_id}")
    def portals(motion_id: int, slot: int = 16, k: int = 8):
        with session() as s:
            m = s.get(Motion, motion_id)
        if not m or not m.tokens:
            raise HTTPException(404, "motion has no tokens")
        ps = POOL.atlas.portals(np.asarray(m.tokens), slot, k=k + 4)
        ps = [p for p in ps if p["clip"] != m.title][:k]   # no self-portals
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
    def atlas_walk(window: int, steps: int = 6, seed: int = 0):
        a = POOL.atlas
        path = a.walk(window, steps, seed)
        return {"path": [{"window": int(w),
                          "clip": a.clips[int(a.clip[w])],
                          "family": FAMILIES[int(a.family[w])]}
                         for w in path]}

    @app.get("/api/motions")
    def motions(limit: int = 50):
        with session() as s:
            rows = s.query(Motion).order_by(Motion.id.desc()).limit(limit)
            return {"motions": [
                {"id": m.id, "title": m.title, "family": m.family,
                 "frames": m.n_frames, "source": m.source,
                 "gate_ratio": m.gate_ratio, "gate_passed": m.gate_passed,
                 "trace": Path(m.trace_path).name}
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

    @app.get("/viewer/{path:path}")
    async def viewer_proxy(path: str):
        from fastapi.responses import Response
        from ..config import CONFIG
        url = f"http://127.0.0.1:{CONFIG.viewer_ports[0]}/{path or ''}"
        async with httpx.AsyncClient() as client:
            r = await client.get(url)
        return Response(content=r.content,
                        media_type=r.headers.get("content-type"))

    @app.websocket("/viser-ws")
    async def viser_ws(ws: WebSocket):
        import websockets

        from ..config import CONFIG
        # viser carries its client version in the websocket SUBPROTOCOL and
        # rejects 'unknown' — forward the client's offer upstream, then accept
        # the browser with whatever viser negotiated
        offered = ws.scope.get("subprotocols") or []
        up = await websockets.connect(
            f"ws://127.0.0.1:{CONFIG.viewer_ports[0]}/",
            subprotocols=offered or None, max_size=None)
        await ws.accept(subprotocol=up.subprotocol)
        try:
            if True:
                async def pump_up():
                    while True:
                        msg = await ws.receive()
                        if msg.get("type") == "websocket.disconnect":
                            break
                        if "bytes" in msg and msg["bytes"] is not None:
                            await up.send(msg["bytes"])
                        elif "text" in msg and msg["text"] is not None:
                            await up.send(msg["text"])

                async def pump_down():
                    async for m in up:
                        if isinstance(m, bytes):
                            await ws.send_bytes(m)
                        else:
                            await ws.send_text(m)
                t1 = asyncio.create_task(pump_up())
                t2 = asyncio.create_task(pump_down())
                done, pending = await asyncio.wait(
                    (t1, t2), return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
        except (WebSocketDisconnect, Exception):  # noqa: BLE001
            pass
        finally:
            try:
                await up.close()
            except Exception:  # noqa: BLE001
                pass

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


class MotionTraceLoader:  # placeholder referenced by jump v1 stub
    def __init__(self, path):
        from ..engine.trace import MotionTrace
        self.trace = MotionTrace.load(path)
