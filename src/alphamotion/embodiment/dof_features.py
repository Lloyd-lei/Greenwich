"""DOF descriptor extraction (vendored from prep_dof_features, corpus code stripped)."""
from __future__ import annotations

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as Rot

from . import mjcf_build as PG

AX = PG.AX
CM = PG.CM

def qmat(q):
    return Rot.from_quat(np.asarray(q, float), scalar_first=True).as_matrix()


def pack(slots, dof_count):
    """slots: list of (axis3, limited, half, center) -> 19 floats."""
    v = np.zeros(19, np.float32)
    for k, (a, lim, half, cen) in enumerate(slots[:3]):
        v[6 * k:6 * k + 3] = a
        v[6 * k + 3] = lim
        v[6 * k + 4] = half
        v[6 * k + 5] = cen
    v[18] = dof_count / 3.0
    return v


def ball_dof(J):
    """3-DOF ball joints (SMPL / BVH): orthonormal basis, unlimited.
    R_rest = I: the stored local rotation IS the joint rotation, and
    rot(x,q0) rot(y,q1) rot(z,q2) is an intrinsic-XYZ Euler chart covering SO(3)."""
    D = np.zeros((J, 19), np.float32)
    for k, a in enumerate(np.eye(3)):
        D[:, 6 * k:6 * k + 3] = a
    D[:, 18] = 1.0
    return D, np.tile(np.eye(3, dtype=np.float32), (J, 1, 1))


def rest6d(Rst):
    """[J,3,3] -> [J,6] (first two columns), the rot6d convention used everywhere."""
    return np.ascontiguousarray(np.asarray(Rst, np.float32)[:, :, :2].transpose(0, 2, 1)
                                ).reshape(-1, 6)


# --------------------------------------------------------------- MJCF path ---

def chain_from(model, b_top_excl, b_bot_incl):
    """body ids strictly below b_top_excl down to b_bot_incl, parent-most first."""
    out = []
    c = int(b_bot_incl)
    while c != b_top_excl:
        out.append(c)
        c = int(model.body_parentid[c])
        assert c >= 0, "not an ancestor"
    return out[::-1]


def body_hinges(model, b):
    a, n = int(model.body_jntadr[b]), int(model.body_jntnum[b])
    return [j for j in range(a, a + n)] if a >= 0 else []


def mjcf_dof(emb, xml, spec, ranges=None):
    """`ranges`: optional {mjcf_joint_name: (lo, hi)} in radians that OVERRIDES the
    MJCF's own jnt_range when encoding the limit features (axes, rest rotations and
    the tree still come from the MJCF).  Used for robots whose kinematic MJCF is a
    third party's retargeting model with hand-narrowed ranges — G1, where GMR's
    g1_mocap_29dof.xml clips 15 of 29 joints; see prep_xwbc_cache.LIMIT_URDF."""
    model = mujoco.MjModel.from_xml_path(xml)
    keep, frame = PG.build_merge(model)
    assert len(keep) == spec_J(spec), (emb, len(keep), spec_J(spec))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    assert [names[b] for b in keep] == list(spec["joint_names"]), emb
    kset = set(keep)

    D = np.zeros((len(keep), 19), np.float32)
    slot_names = []
    nonhinge = []
    rest = {}
    axes_raw = {}
    for k, b in enumerate(keep):
        if k == 0:
            chain = chain_from(model, 0, frame[b])        # world -> frame[root]
        else:
            p = int(model.body_parentid[b])
            while p >= 1 and p not in kset:
                p = int(model.body_parentid[p])
            chain = chain_from(model, frame[p], frame[b])
        B = np.eye(3)
        slots, snames = [], []
        for c in chain:
            B = B @ qmat(model.body_quat[c])
            for j in body_hinges(model, c):
                jt = model.jnt_type[j]
                jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"jnt{j}"
                if jt != mujoco.mjtJoint.mjJNT_HINGE:
                    nonhinge.append((jn, int(jt)))
                    continue
                a = AX @ (B @ model.jnt_axis[j])
                a = a / np.linalg.norm(a)
                lim = float(model.jnt_limited[j])
                lo, hi = model.jnt_range[j]
                if ranges and jn in ranges:
                    lo, hi = ranges[jn]
                    lim = 1.0
                half = (hi - lo) / 2 / np.pi if lim else 0.0
                cen = (hi + lo) / 2 / np.pi if lim else 0.0
                slots.append((a, lim, half, cen))
                snames.append(jn)
        rest[k] = B
        axes_raw[k] = [AX.T @ s[0] for s in slots]
        assert len(slots) <= 3, (emb, spec["joint_names"][k], len(slots))
        D[k] = pack(slots, len(slots))
        slot_names.append(",".join(snames))
    # Y-up rest rotations, same conjugation as the cache's globals (R' = AX R AX^T).
    # Root: identity by convention (its rotation is the free joint's, from data).
    Rst = np.stack([np.eye(3) if k == 0 else AX @ rest[k] @ AX.T
                    for k in range(len(keep))]).astype(np.float32)
    return D, slot_names, model, keep, frame, rest, axes_raw, nonhinge, Rst


def spec_J(spec):
    return len(spec["parents"])

