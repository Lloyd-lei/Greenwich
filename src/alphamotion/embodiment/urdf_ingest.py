"""User URDF ingest: parse -> check -> label -> refiner config, one report.

Every step reports honestly; nothing is silently "fixed". The output is an
ingest report dict plus (on success) a registered embodiment the engine can
decode onto zero-shot.

Steps:
  1. parse/compile: MjSpec.from_file(urdf); URDF has no floating base, so a
     freejoint is injected at the first body (verified route);
  2. descriptor: the exact MJCF pipeline the bundled robots went through
     (build_merge/build_spec/mjcf_dof) — same code path, no special-casing;
  3. limit check: missing limits, zero-span joints, non-hinge joints,
     DOF census;
  4. semantic labeling: Qwen3 joint-name embeddings (the encoder the released
     Greenwich was trained with) + geometric fallback for the five key parts;
  5. refiner config: wrist chains + limits, persisted with the skeleton.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ..paths import data_dir
from .registry import user_dir


def safe_body_name(value: str) -> str:
    """Stable filesystem/database identifier for an uploaded embodiment."""
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip()).strip("._-")
    if not name:
        raise ValueError("body name contains no usable characters")
    return name[:80].lower()


def anchor_urdf_meshdir(spec, urdf_path: str | Path) -> None:
    """Make converted MJCF mesh paths absolute without duplicating prefixes.

    MuJoCo may report ``meshdir='meshes'`` while retaining URDF asset names
    such as ``./meshes/pelvis.stl``.  Anchoring that meshdir directly would
    resolve the asset as ``meshes/meshes/pelvis.stl``.  Remove the already
    represented prefix from each asset before making meshdir absolute.
    """
    urdf_path = Path(urdf_path).resolve()
    meshdir = Path(getattr(spec, "meshdir", "") or "")
    if meshdir.is_absolute():
        return
    prefix = tuple(part for part in meshdir.parts if part not in ("", "."))
    if prefix:
        for mesh in spec.meshes:
            value = Path(str(getattr(mesh, "file", "") or ""))
            parts = tuple(part for part in value.parts if part not in ("", "."))
            if parts[:len(prefix)] == prefix and len(parts) > len(prefix):
                mesh.file = str(Path(*parts[len(prefix):]))
    spec.meshdir = str((urdf_path.parent / meshdir).resolve())


def urdf_to_mjcf(urdf_path: str | Path, name: str) -> Path:
    """URDF -> compilable MJCF with an injected floating base."""
    import mujoco
    urdf_path = Path(urdf_path).resolve()
    spec = mujoco.MjSpec.from_file(str(urdf_path))
    # URDF mesh references are relative to the URDF's own directory; the MJCF
    # we persist lives elsewhere, so anchor meshdir absolutely or the compile
    # of the saved file dies with "Error opening file 'meshes/...'"
    anchor_urdf_meshdir(spec, urdf_path)
    bodies = spec.bodies
    if len(bodies) < 2:
        raise ValueError("URDF has no articulated bodies")
    root = bodies[1]
    has_free = any(j.type == mujoco.mjtJoint.mjJNT_FREE
                   for b in bodies for j in b.joints)
    if not has_free:
        root.add_freejoint()
    model = spec.compile()          # raises with a real error message if broken
    out = user_dir() / f"{name}.xml"
    xml = spec.to_xml()
    out.write_text(xml)
    return out


def limit_check(model) -> dict:
    """Joint-limit census on the compiled model."""
    import mujoco
    issues, hinges, unlimited, zero_span = [], 0, [], []
    for j in range(model.njnt):
        jt = model.jnt_type[j]
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
        if jt == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if jt != mujoco.mjtJoint.mjJNT_HINGE:
            issues.append(f"non-hinge joint '{nm}' "
                          f"(type {int(jt)}) — treated as fixed by the codec")
            continue
        hinges += 1
        if not model.jnt_limited[j]:
            unlimited.append(nm)
        else:
            lo, hi = model.jnt_range[j]
            if hi - lo < 1e-6:
                zero_span.append(nm)
    return {"hinge_dofs": hinges,
            "unlimited_joints": unlimited,
            "zero_span_joints": zero_span,
            "other_issues": issues,
            "ok": not zero_span}


def semantic_labels(spec, device: str = "cuda") -> dict:
    """Per-joint part labels: Qwen3 name-embedding nearest-anchor vote with a
    geometric fallback (key_joints) when names carry no signal."""
    import torch

    from ..engine.spatial import key_joints
    from .semantics_map import (build_name_embeddings,
                                deterministic_part_labels, embed_matrix)
    anchors = {
        "head": "the head joint of a body",
        "left arm": "the left arm joint of a body",
        "right arm": "the right arm joint of a body",
        "left leg": "the left leg joint of a body",
        "right leg": "the right leg joint of a body",
        "torso": "the torso joint of a body",
    }
    labels = deterministic_part_labels(spec)
    method = "topology+name"
    try:
        # These are names, not already-expanded prompts. build_name_embeddings
        # applies the exact training prompt template once.
        lut = build_name_embeddings(list(spec.joint_names)
                                    + list(anchors), device=device)
        J = embed_matrix(spec.joint_names, lut)              # [J,1024]
        A = embed_matrix(list(anchors), lut)                 # [6,1024]
        sim = J @ A.T
        for i, n in enumerate(spec.joint_names):
            labels[n] = list(anchors)[int(sim[i].argmax())]
        method = "qwen3+topology"
    except Exception:  # optional labeling dependency/cache may be unavailable
        pass
    # geometric key joints always annotated on top (they anchor the QC metrics)
    kj, knames, tags = key_joints(spec)
    keymap = dict(zip(("root", "head", "l_wrist", "r_wrist",
                       "l_ankle", "r_ankle"), knames))
    # Preserve canonical anchors regardless of name-embedding ambiguity.
    overrides = {"root": "torso", "head": "head",
                 "l_wrist": "left arm", "r_wrist": "right arm",
                 "l_ankle": "left leg", "r_ankle": "right leg"}
    for tag, name in keymap.items():
        labels[name] = overrides[tag]
    return {"per_joint": labels, "key_joints": keymap, "method": method,
            "coverage": len(labels) / max(spec.J, 1)}


def ingest(urdf_path: str | Path, name: str | None = None,
           device: str = "cuda") -> dict:
    import mujoco

    from ..engine.descriptor import build_from_mjcf
    urdf_path = Path(urdf_path)
    name = safe_body_name(name or urdf_path.stem)
    report: dict = {"name": name, "source": str(urdf_path)}

    # 1-2: parse + descriptor
    xml = urdf_to_mjcf(urdf_path, name)
    report["mjcf"] = str(xml)
    spec, dof, rest, qn, _ = build_from_mjcf(xml, name)
    report["joints"] = spec.J
    report["height_cm"] = round(spec.height, 1)

    # 3: limits
    model = mujoco.MjModel.from_xml_path(str(xml))
    report["limits"] = limit_check(model)

    # 4: labels
    report["semantics"] = semantic_labels(spec, device)

    # 5: refiner config + persist
    wrists = [n for n in spec.joint_names
              if "wrist" in n.lower() or n.lower().endswith("hand")]
    refiner_cfg = {"wrist_joints": wrists, "wrist_alpha": 0.65, "lm_iters": 20}
    report["refiner"] = refiner_cfg

    u = user_dir()
    np.savez(u / f"{name}_spec.npz", name=spec.name, parents=spec.parents,
             rest_offsets=spec.rest_offsets, joint_names=spec.joint_names,
             norm_bonelen=spec.norm_bonelen, height=spec.height)
    np.savez(u / f"{name}_dof.npz", dof=dof, rest=rest)
    (u / f"{name}_meta.json").write_text(json.dumps(
        {"source": "user_urdf", "xml": str(xml),
         "qnames": [",".join(x) for x in (qn or [])],
         "semantics": report["semantics"], "refiner": refiner_cfg,
         "limits": report["limits"]}, indent=1))
    report["registered"] = True
    return report
