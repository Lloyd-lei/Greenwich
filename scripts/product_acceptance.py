#!/usr/bin/env python3
"""Black-box acceptance gate for the shipped AlphaMotion service.

This intentionally calls HTTP endpoints instead of importing pipeline
internals.  It covers native playback, temporal retiming, bridge sampling,
token pins, SE(3) editing, Atlas jump generation, trace contracts, and the
download surface used by the browser product.
"""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import httpx
import numpy as np


def request(base: str, path: str, payload: dict | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        base.rstrip("/") + path, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def submit(base: str, path: str, payload: dict, timeout: float = 180.0):
    job_id = request(base, path, payload)["job_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = request(base, f"/api/jobs/{job_id}")
        if job["status"] == "done":
            return job["result"]
        if job["status"] == "failed":
            raise RuntimeError(f"{path} failed: {job['error']}")
        time.sleep(0.5)
    raise TimeoutError(f"{path} timed out: {job_id}")


def trace(base: str, name: str):
    with urllib.request.urlopen(
            base.rstrip("/") + f"/api/results/{name}", timeout=30) as r:
        return np.load(io.BytesIO(r.read()), allow_pickle=True)


def assert_trace(data, frames: int, body: str):
    assert data["q"].shape[0] == frames
    assert data["rootR"].shape == (frames, 3, 3)
    assert data["gp"].shape[0] == frames
    assert data["stage"].shape == (frames,)
    assert str(data["target"]) == body
    for key in ("q", "rootR", "gp"):
        assert np.isfinite(data[key]).all(), f"{key} contains non-finite values"


def expect_http(base: str, path: str, payload: dict | None, status: int):
    try:
        request(base, path, payload)
    except urllib.error.HTTPError as exc:
        assert exc.code == status, (path, exc.code, status)
        return
    raise AssertionError(f"{path} unexpectedly accepted invalid input")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--body", default="unitree_h1")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--perception", action="store_true",
                        help="also run the configured GENMO text adapter")
    parser.add_argument("--out", default="artifacts/product_acceptance.json")
    args = parser.parse_args()
    base, body = args.base_url, args.body
    report: dict[str, object] = {"base_url": base, "body": body,
                                "render": args.render, "checks": {}}

    health = request(base, "/api/health")
    required = {"temporal", "token_pins", "se3", "mp4"}
    assert health["ok"] and all(health["capabilities"].get(k)
                                for k in required)
    bodies = request(base, "/api/bodies")["bodies"]
    assert any(b["name"] == body and b["renderable"] for b in bodies)
    detail = request(base, f"/api/bodies/{body}")
    names = detail["joint_names"]
    elbow = next((i for i, n in enumerate(names)
                  if "right_elbow" in n.lower()), len(names) - 1)
    root = 0
    report["checks"]["health"] = {"height_cm": detail["height_cm"],
                                     "joints": detail["joints"]}

    expect_http(base, "/api/jobs/play",
                {"library_id": 2, "target_body": "not_a_robot",
                 "render": False}, 422)
    expect_http(base, "/api/jobs/timeline",
                {"segments": [{"kind": "gap", "n": 8}],
                 "target_body": body, "render": False}, 422)
    report["checks"]["invalid_requests"] = "PASS"

    native = submit(base, "/api/jobs/play",
                    {"library_id": 0, "target_body": body,
                     "render": args.render})
    assert native["frames"] == 60 and native["qc"]["continuity"]["passed"]
    assert native["gate"]["passed"], native["gate"]
    assert native["qc"]["limb_synergy"]["passed"], native["qc"]
    assert native["release_passed"], native
    native_trace = trace(base, native["trace"])
    assert_trace(native_trace, 60, body)
    report["checks"]["native"] = native
    expect_http(base, f"/api/atlas/portals/{native['motion_id']}?slot=32",
                None, 422)
    expect_http(base, "/api/atlas/walk/999999999", None, 404)
    bad_upload = httpx.post(
        base.rstrip("/") + "/api/bodies/ingest",
        files={"file": ("not-a-body.txt", b"x", "text/plain")}, timeout=30)
    assert bad_upload.status_code == 422, bad_upload.text
    report["checks"]["boundary_validation"] = "PASS"

    segments = [
        {"kind": "library", "library_id": 0, "n": 60},
        {"kind": "gap", "n": 18, "seed": 7, "temperature": 0.8,
         "pins": {"4": 120}},
        {"kind": "library", "library_id": 1, "n": 60},
    ]
    controls = [
        {"joint": root, "frame_start": 10, "frame_end": 20,
         "delta_m": [0.02, 0.0, 0.0], "delta_rot_deg": [0.0, 5.0, 0.0]},
        {"joint": elbow, "frame_start": 65, "frame_end": 75,
         "delta_m": [0.02, 0.0, 0.0], "delta_rot_deg": [0.0, 5.0, 0.0]},
    ]
    edited = submit(base, "/api/jobs/timeline",
                    {"segments": segments, "target_body": body,
                     "title": "product acceptance temporal edit",
                     "se3": controls, "render": args.render, "fps": 30})
    assert edited["frames"] == 138
    assert edited["qc"]["continuity"]["passed"], edited["qc"]
    assert edited["gate"]["passed"], edited["gate"]
    assert edited["qc"]["limb_synergy"]["passed"], edited["qc"]
    assert edited["release_passed"], edited
    assert edited["refiner"]["hold_repair"]["holds_after"] == 0
    assert edited["se3"][0]["root_translation"] is True
    assert edited["se3"][0]["rotation_error_deg"] < 5.0
    assert edited["se3"][1]["position_error_cm"] < 0.2
    edited_trace = trace(base, edited["trace"])
    assert_trace(edited_trace, 138, body)
    assert set(np.unique(edited_trace["stage"])) >= {0, 1, 2}
    np.testing.assert_allclose(
        edited_trace["root_t"][10:20] - native_trace["root_t"][10:20],
        np.tile(np.array([2.0, 0.0, 0.0]), (10, 1)), atol=1e-4)
    report["checks"]["temporal_editor"] = edited

    jumped = submit(base, "/api/jobs/jump",
                    {"motion_id": native["motion_id"], "at_slot": 16,
                     "dest_library_id": 50, "bridge_n": 30,
                     "target_body": body, "render": args.render})
    assert jumped["frames"] > 60
    assert jumped["qc"]["continuity"]["passed"], jumped["qc"]
    assert jumped["gate"]["passed"], jumped["gate"]
    assert jumped["qc"]["limb_synergy"]["passed"], jumped["qc"]
    assert jumped["release_passed"], jumped
    jump_trace = trace(base, jumped["trace"])
    assert_trace(jump_trace, jumped["frames"], body)
    report["checks"]["atlas_jump"] = jumped

    if args.perception:
        assert health["capabilities"].get("perception"), \
            health["capabilities"].get("perception_detail")
        atlas_before = request(base, "/api/health")["warm"]["atlas_windows"]
        perceived = submit(
            base, "/api/jobs/timeline",
            {"segments": [{"kind": "prompt",
                           "text": "A person walks forward naturally.",
                           "n": 30}],
             "target_body": body,
             "title": "perception acceptance",
             "render": False, "fps": 30},
            timeout=600.0)
        perceived_trace = trace(base, perceived["trace"])
        assert_trace(perceived_trace, 30, body)
        assert "root_t" in perceived_trace.files
        root_path = float(np.linalg.norm(
            np.diff(perceived_trace["root_t"], axis=0), axis=1).sum())
        assert root_path > 0.1, "perception dropped the SMPL world trajectory"
        expected_release = bool(
            perceived["gate"]["passed"]
            and perceived["qc"]["continuity"]["passed"]
            and perceived["qc"]["limb_synergy"]["passed"])
        assert perceived["release_passed"] is expected_release
        atlas_after = request(base, "/api/health")["warm"]["atlas_windows"]
        assert atlas_after - atlas_before == int(expected_release), {
            "before": atlas_before, "after": atlas_after,
            "release": expected_release}
        report["checks"]["perception"] = {
            "motion": perceived, "root_path_cm": round(root_path, 3),
            "atlas_isolation": "PASS"}

    if args.render:
        for key, result in (("native", native), ("temporal", edited),
                            ("jump", jumped)):
            assert result.get("mp4"), f"{key} did not produce MP4"
            with urllib.request.urlopen(
                    base.rstrip("/") + f"/api/results/{result['mp4']}") as r:
                assert r.headers.get_content_type() == "video/mp4"
                assert int(r.headers["Content-Length"]) > 10000
        report["checks"]["downloads"] = "PASS"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: {out}")


if __name__ == "__main__":
    main()
