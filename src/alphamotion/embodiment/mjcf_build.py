"""MJCF -> SkeletonSpec builders (vendored from prep_gmr_cache, corpus code stripped)."""
from __future__ import annotations

import numpy as np
import mujoco

from ..engine.nets.rotations import SkeletonSpec

CM = 100.0                                    # MuJoCo metres -> cache centimetres
# Z-up MuJoCo world -> Y-up cache convention (prep_gmr_cache v2 fix)
AX = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]], np.float64)
FK_GATE_CM = 0.1

def build_merge(model, eps=1e-3):
    """Decide which co-located bodies to merge and which frame carries each kept joint.

    Returns (keep, frame) — keep: kept body ids ascending (topological in MJCF);
    frame[i]: body id whose xmat is the kept joint's global rotation.
    """
    import mujoco
    nb = model.nbody
    parent = model.body_parentid
    pos = model.body_pos
    drop = {i for i in range(2, nb) if np.linalg.norm(pos[i]) < eps}

    def compute(drop):
        keep = [i for i in range(1, nb) if i not in drop]
        kset = set(keep)

        def nka(b):                                   # nearest kept ancestor
            p = int(parent[b])
            while p >= 1 and p not in kset:
                p = int(parent[p])
            return p

        attach = {b: set() for b in keep}             # frames kept children hang from
        for c in keep:
            a = nka(c)
            if a >= 1:
                attach[a].add(int(parent[c]))
        conflicts = {b: s for b, s in attach.items() if len(s) > 1}
        return keep, attach, conflicts

    keep, attach, conflicts = compute(drop)
    while conflicts:                                  # un-drop conflicting attach points
        for b, s in conflicts.items():
            for x in s:
                drop.discard(x)
        keep, attach, conflicts = compute(drop)

    dropped_children = {}
    for d in sorted(drop):
        dropped_children.setdefault(int(parent[d]), []).append(d)

    def chain_end(b):                                 # deepest single co-located chain
        cur = b
        while len(dropped_children.get(cur, [])) == 1:
            cur = dropped_children[cur][0]
        return cur

    frame = {}
    for b in keep:
        s = attach[b]
        frame[b] = next(iter(s)) if len(s) == 1 else chain_end(b)
    return keep, frame


def standing_height(model):
    """True sole-to-pelvis height (m) at qpos0: root body z minus the lowest point of
    any collision/visual geom, mesh vertices included.

    NOT the MJCF spawn z (`body_pos[1][2]`). The spawn z is wherever the author
    parked the free joint in the world: booster_t1 spawns at 1.68 m but stands with
    its pelvis at 0.76 m, several robots spawn at 0 (which used to fall back to a
    hard-coded 0.8 m). Since this value is the spec's ROOT BONE it enters
    `spec.height` and therefore normalises every `norm_bonelen` the model is fed."""
    import mujoco
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    d.qpos[:] = model.qpos0
    mujoco.mj_forward(model, d)
    lo = np.inf
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0:                 # the MJCF's own floor plane
            continue
        p, R = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(model.geom_dataid[g])
            a, nv = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            v = model.mesh_vert[a:a + nv].reshape(-1, 3).astype(np.float64)
            lo = min(lo, float((v @ R.T)[:, 2].min() + p[2]))
        else:
            lo = min(lo, float(p[2] - model.geom_rbound[g]))
    h = float(d.xpos[1][2] - lo)
    if not np.isfinite(h) or h <= 1e-3:               # no geoms at all -> old behaviour
        h = float(model.body_pos[1][2] or 0.8)
    return h


def build_spec(model, name, keep):
    """keep -> SkeletonSpec (cm), same conventions as mjcf_skeleton.spec_from_mjcf
    (offset accumulation through dropped bodies, root bone = standing height)."""
    import mujoco
    parent = model.body_parentid
    pos = model.body_pos
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"
             for i in range(model.nbody)]
    kset = set(keep)
    remap = {b: k for k, b in enumerate(keep)}
    parents, offsets, jnames = [], [], []
    for b in keep:
        p = int(parent[b])
        while p >= 1 and p not in kset:
            p = int(parent[p])
        parents.append(remap[p] if p in kset else -1)
        acc = pos[b].copy()
        cur = int(parent[b])
        while cur >= 1 and cur not in kset:
            acc = acc + pos[cur]
            cur = int(parent[cur])
        offsets.append(acc * CM)
        jnames.append(names[b])
    offsets = np.asarray(offsets, np.float32)
    # root bone = the robot's REAL standing (sole -> pelvis) height, measured from
    # the mesh geometry at qpos0.  Was `model.body_pos[1][2] or 0.8` = the MJCF spawn
    # z with a hard-coded fallback: wrong by +91.5 cm on booster_t1, -22.9 cm on
    # fourier_gr3, +20.5 cm on booster_k1, ... (see LIMITS_AUDIT.md 5b).
    offsets[0] = np.array([0.0, 0.0, standing_height(model) * CM], np.float32)
    offsets = (offsets @ AX.T).astype(np.float32)   # Z-up -> Y-up (root -> (0,h,0))
    spec = SkeletonSpec(name, np.asarray(parents, np.int64), offsets, jnames)
    spec.order_root_first()
    return spec


