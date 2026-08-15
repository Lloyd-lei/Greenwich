"""SE3 precision ladder, layer 3: project decoded rotations onto what the target
robot can ACTUALLY do, optionally pinning constrained joints exactly.

Why this exists (measured, not assumed):
  * the decoder emits full 3-DOF rotations per joint, but robot joints are 1-DOF
    hinges -> predictions sit off the mechanical manifold (6-10 deg residual)
  * joint limits are fed as INPUT features but nothing enforces them; predictions
    violate G1 leg limits on 1-5% of frames
  * soft conditioning alone satisfies an SE3 constraint to only ~7 cm

Method (per frame, per skeleton):
  1. read each joint's DOF descriptor (axes + limits, from {emb}_dof.npz)
  2. fit hinge angles q that best reproduce the predicted local rotations
     (closed form: twist of the local rotation about the joint axis) and clamp to
     [lo, hi] * soft_margin  (margin < 1 keeps a safety band, applied at EXECUTION
     only — the input features stay truthful).  method="greedy"
  3. refine q by ONE least-squares problem over the whole kinematic chain,
     min_q sum_j |FK(q)_j - R_pred_j|_F^2  s.t. lo <= q <= hi, by projected
     Levenberg-Marquardt with analytic Jacobians, warm-started from step 2.
     method="global" (the default for project()).  Step 2 alone is per-joint and
     therefore compounds a hip residual into knee/ankle/toe: measured on real
     predictions it RAISES pose error 10.08 -> 12.69 deg, while step 3 LOWERS it
     to 8.86 deg. Both give 0% limit violations.
  4. optional constraint pass (project_constrained): same LM with specified joints
     as hard task-space goals (weight ~1e4) and the step-3 solution as the
     null-space reference, so the rest of the body keeps the model's intent.
     Measured: constraint satisfaction 90.4 mm -> 0.001 mm (G1, root + one hand).
  5. rebuild rotations by FK from q -> guaranteed feasible output

Temporal consistency is NOT assumed: temporal_report() reports frame-to-frame
angle jumps before/after so branch-flipping shows up instead of hiding.

CONVENTION (see pose_translator_design/DOF_FEATURES.md):
    R_local(q) = rot(A_0,q_0) rot(A_1,q_1) rot(A_2,q_2) @ R_rest
R_rest is the joint's STATIC rest rotation (`rest` key of {emb}_dof.npz). Dropping
it makes the angle -> rotation direction wrong for every robot whose MJCF has a
non-identity <body quat> (i.e. most of them). load_dof() reads it; the identity
fallback keeps pre-fix caches loadable (they are simply wrong, not crashing).
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from .nets.rotations import rot6d_to_matrix, matrix_to_rot6d


def load_dof(cache, emb, device="cpu", dtype=torch.float32):
    """-> (dof [J,19], rest [J,3,3]). Missing `rest` (old caches) -> identity."""
    d = np.load(f"{cache}/{emb}_dof.npz", allow_pickle=True)
    dof = torch.as_tensor(np.asarray(d["dof"]), dtype=dtype, device=device)
    if "rest" in d.files:
        rest = torch.as_tensor(np.asarray(d["rest"]), dtype=dtype, device=device)
    else:
        rest = torch.eye(3, dtype=dtype, device=device).expand(dof.shape[0], 3, 3).clone()
    return dof, rest


def _best_angle_about(X, axis):
    """argmax_q tr(rot(axis,q)^T X) — the exact least-squares hinge angle for a
    general X, and the exact angle when X = rot(axis,q). [...,3,3],[...,3] -> [...]."""
    A = (X - X.transpose(-1, -2)) / 2
    v = torch.stack([A[..., 2, 1], A[..., 0, 2], A[..., 1, 0]], -1)
    s = 2 * (v * axis).sum(-1)
    aXa = (axis.unsqueeze(-2) @ X @ axis.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    c = X.diagonal(dim1=-2, dim2=-1).sum(-1) - aXa
    return torch.atan2(s, c)


def _twist_about(R, axis):
    """Backwards-compatible alias (exact for a pure rotation about `axis`)."""
    return _best_angle_about(R, axis)


def _rot_axis_angle(axis, ang):
    """Rodrigues. axis [...,3] unit, ang [...] -> [...,3,3]."""
    z = torch.zeros_like(axis[..., 0])
    K = torch.stack([
        torch.stack([z, -axis[..., 2], axis[..., 1]], -1),
        torch.stack([axis[..., 2], z, -axis[..., 0]], -1),
        torch.stack([-axis[..., 1], axis[..., 0], z], -1)], -2)
    I = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(K.shape)
    s = torch.sin(ang)[..., None, None]; c = torch.cos(ang)[..., None, None]
    return I + s * K + (1 - c) * (K @ K)


def _slot_axes(dof):
    """dof [J,19] -> unit axes [J,3,3] (zero where the slot is absent) and the
    active mask [J,3]."""
    J = dof.shape[0]
    A = dof[:, :18].reshape(J, 3, 6)[:, :, :3].clone()       # [J,3,3]
    n = A.norm(dim=-1, keepdim=True)
    A = A / n.clamp_min(1e-12)
    n_dof = (dof[:, 18] * 3).round().long()
    slot = torch.arange(3, device=dof.device)[None, :]
    active = (n[..., 0] > 1e-6) & (slot < n_dof[:, None])
    return A * active[..., None], active


def _limits(dof, soft_margin):
    """-> (lo [J,3], hi [J,3], limited [J,3] bool)."""
    v = dof[:, :18].reshape(-1, 3, 6)
    limited = v[:, :, 3] > 0.5
    half = v[:, :, 4] * np.pi * soft_margin
    ctr = v[:, :, 5] * np.pi
    return ctr - half, ctr + half, limited


def _wrap_to_box(ang, lo, hi):
    """Pick the 2*pi-equivalent representative of `ang` with the least box overshoot.

    rot(a,q) is 2*pi-periodic, so this NEVER changes the rotation — but 9 of the 18
    cached embodiments have limit boxes that cross +-180 deg (e.g. tienkung
    shoulder_roll_l [-15, 194.8] deg, unitree_h1 shoulder_yaw [-74.5, 255]). The
    fitters produce PRINCIPAL angles (atan2 -> (-pi, pi]), so a perfectly legal
    185 deg comes back as -175, reads as out-of-box, and clamping snaps it to the
    wrong end of the range: measured 164.9 deg rotation error on tienkung's own
    cached motion. Judge and clamp the right representative instead. Only call this
    where `limited` is true — unlimited slots have a meaningless (0,0) box."""
    two_pi = 2 * np.pi
    best = ang
    best_over = (lo - ang).clamp_min(0) + (ang - hi).clamp_min(0)
    for s in (two_pi, -two_pi):
        cand = ang + s
        over = (lo - cand).clamp_min(0) + (cand - hi).clamp_min(0)
        take = over < best_over
        best = torch.where(take, cand, best)
        best_over = torch.where(take, over, best_over)
    return best


def _solve3(M, A, lo=None, hi=None, limited=None):
    """Closed-form generalized-Euler (Davenport) solve of
        M = rot(a0,q0) rot(a1,q1) rot(a2,q2)
    for arbitrary (a0,a1,a2). Needed as the SEED for 3-DOF joints: plain coordinate
    descent from q=0 lands in local minima there (SMPL ball joints hit 21 deg).
    M [B,J,3,3], A [J,3,3] -> q [B,J,3]. Degenerate axes fall back to q1 = 0.

    BRANCH CHOICE. The two acos branches (q1 = phi +- d) are the two arms of the
    double cover: BOTH reproduce M exactly, but they are different poses, and only
    one of them is generally inside the joint's limit box. Picking by residual alone
    is a coin flip between them, and on Apollo's / kuavo's 3-DOF merged joints it
    picked the out-of-box arm on ~13% of strictly-feasible samples (worst 168 deg
    outside), which `_fit_angles_greedy` then clamped into a badly wrong LM warm
    start. So: when both branches reproduce M equally well, prefer the FEASIBLE one;
    when neither is feasible (or the limits are unknown), fall back to the smaller
    residual exactly as before. lo/hi/limited [J,3] as returned by `_limits`."""
    a0, a1, a2 = A[:, 0], A[:, 1], A[:, 2]                      # [J,3]
    Mw = (M @ a2[None, :, :, None]).squeeze(-1)                 # [B,J,3]
    k1 = (a0 * a1).sum(-1) * (a1 * a2).sum(-1)                  # [J]
    Ac = (a0 * a2).sum(-1) - k1
    Bc = (a0 * torch.cross(a1, a2, dim=-1)).sum(-1)
    r = torch.sqrt(Ac ** 2 + Bc ** 2)
    phi = torch.atan2(Bc, Ac)
    rhs = ((a0[None] * Mw).sum(-1) - k1[None]) / r.clamp_min(1e-12)[None]
    d = torch.arccos(rhs.clamp(-1.0, 1.0))
    d = torch.where((r > 1e-9)[None], d, torch.zeros_like(d))
    box = None
    if lo is not None and hi is not None and limited is not None:
        box = (lo[None], hi[None], limited[None])
    cand = []
    for sgn in (1.0, -1.0):
        q1 = torch.where((r > 1e-9)[None], phi[None] + sgn * d, torch.zeros_like(d))
        R1 = _rot_axis_angle(a1[None].expand(q1.shape + (3,)), q1)
        v = (R1 @ a2[None, :, :, None]).squeeze(-1)             # [B,J,3]
        vpar = (a0[None] * v).sum(-1, keepdim=True) * a0[None]
        q0 = torch.atan2((torch.cross(a0[None].expand_as(v), v, dim=-1) * Mw).sum(-1),
                         ((v - vpar) * Mw).sum(-1))
        R0 = _rot_axis_angle(a0[None].expand(q0.shape + (3,)), q0)
        X = R1.transpose(-1, -2) @ R0.transpose(-1, -2) @ M
        q2 = _best_angle_about(X, a2[None].expand(X.shape[:-1]))
        R2 = _rot_axis_angle(a2[None].expand(q2.shape + (3,)), q2)
        err = (R0 @ R1 @ R2 - M).flatten(-2).norm(dim=-1)       # [B,J]
        # q1 = phi +- d lives in (-2pi, 2pi]; q0/q2 come from atan2 and are already
        # principal. rot(a, q) is 2pi-periodic, so wrapping changes nothing about the
        # rotation, but an unwrapped -265.6 deg reads as "far outside [-20, 120]"
        # when the equivalent +94.4 deg is comfortably inside. Wrap before judging —
        # and judge the representative NEAREST THE BOX, not the principal one: boxes
        # that cross +-180 deg (see _wrap_to_box) contain angles whose principal
        # representative reads as infeasible.
        q1 = torch.atan2(torch.sin(q1), torch.cos(q1))
        q = torch.stack([q0, q1, q2], -1)
        if box is None:
            feas = torch.zeros_like(err, dtype=torch.bool)
        else:
            b_lo, b_hi, b_lim = box
            q = torch.where(b_lim.expand_as(q), _wrap_to_box(q, b_lo, b_hi), q)
            out = ((b_lo - q).clamp_min(0) + (q - b_hi).clamp_min(0)) * b_lim
            feas = out.amax(-1) <= 1e-9                         # [B,J]
        cand.append((q, err, feas))
    (qa, ea, fa), (qb, eb, fb) = cand
    # "equally good" = reproduces M to within numerical noise of the better branch
    tol = 10.0 * float(torch.finfo(M.dtype).eps) ** 0.5
    best = torch.minimum(ea, eb)
    pa, pb = fa & (ea <= best + tol), fb & (eb <= best + tol)
    take_b = torch.where(pa ^ pb, pb, eb < ea)                  # else: smaller residual
    return torch.where(take_b[..., None], qb, qa)


def _fit_angles_greedy(global_rot6d, spec, dof, soft_margin=1.0, rest=None, clamp=True,
                       iters=32):
    """GLOBAL rotations -> per-DOF hinge angles (optionally clamped to limits).

    dof:  [J,19] = 3 slots x [axis(3), limited, half_range/pi, center/pi] + count/3
    rest: [J,3,3] static rest rotation; None -> identity (WRONG for real robots,
          kept only so pre-fix caches load).

    GREEDY / per-joint: each joint is fitted to ITS OWN local rotation, independently
    of the rest of the chain. Solves  rot(a0,q0) rot(a1,q1) rot(a2,q2) = R_local @
    R_rest^T  by block coordinate descent with the exact per-axis optimum (closed
    form, atan2 only, so no domain errors on off-manifold input). 1-DOF joints
    converge in one sweep; 2/3-DOF joints converge to machine precision on feasible
    input.

    Exact on feasible input (GT round trip 1e-14 deg), but on OFF-manifold input it
    is not the least-squares answer for the pose: a hip residual is inherited by
    knee/ankle/toe instead of being traded off against them. Use method="global"
    (whole-chain LM, below) when the input is a model prediction.

    Returns q [B,J,3] (unused slots zero) and the active-slot mask [J,3].
    """
    Rg = rot6d_to_matrix(global_rot6d) if global_rot6d.shape[-1] == 6 else global_rot6d
    B, J = Rg.shape[0], Rg.shape[1]
    dev, dt = Rg.device, Rg.dtype
    parents = torch.as_tensor(np.asarray(spec.parents), device=dev)
    pidx = parents.clamp_min(0)
    Rl = Rg[:, pidx].transpose(-1, -2) @ Rg                          # [B,J,3,3]
    Rl = torch.where((parents < 0)[None, :, None, None], Rg, Rl)     # root: local = global
    if rest is None:
        rest = torch.eye(3, device=dev, dtype=dt).expand(J, 3, 3)
    M = Rl @ rest.to(dev, dt).transpose(-1, -2)[None]                # target rotation product

    A, active = _slot_axes(dof.to(dev, dt))
    lo, hi, limited = _limits(dof.to(dev, dt), soft_margin)
    q = torch.zeros(B, J, 3, device=dev, dtype=dt)
    Ax = A[None].expand(B, J, 3, 3)                                  # [B,J,slot,3]
    Rs = [torch.eye(3, device=dev, dtype=dt).expand(B, J, 3, 3).contiguous()
          for _ in range(3)]
    if not active.any():
        return q, active
    n_dof = (dof[:, 18] * 3).round().long()
    if bool((n_dof == 3).any()):                    # closed-form seed for ball / triples
        seed = _solve3(M, A, lo, hi, limited & active)
        m3 = (n_dof == 3)[None, :, None]
        if clamp:
            seed = torch.where(limited[None], seed.maximum(lo[None]).minimum(hi[None]), seed)
        q = torch.where(m3, seed, q)
        for s in range(3):
            Rs[s] = _rot_axis_angle(Ax[:, :, s], q[:, :, s])
    n_it = 1 if int(n_dof.max()) <= 1 else iters
    for _ in range(n_it):
        for s in range(3):
            if not bool(active[:, s].any()):
                continue
            pre = Rs[0] if s == 1 else (Rs[0] @ Rs[1] if s == 2 else None)
            post = (Rs[1] @ Rs[2]) if s == 0 else (Rs[2] if s == 1 else None)
            X = M
            if pre is not None:
                X = pre.transpose(-1, -2) @ X
            if post is not None:
                X = X @ post.transpose(-1, -2)
            ang = _best_angle_about(X, Ax[:, :, s])
            ang = torch.where(limited[None, :, s],
                              _wrap_to_box(ang, lo[None, :, s], hi[None, :, s]),
                              ang)
            if clamp:
                ang = torch.where(limited[None, :, s],
                                  ang.maximum(lo[None, :, s]).minimum(hi[None, :, s]),
                                  ang)
            ang = torch.where(active[None, :, s], ang, torch.zeros_like(ang))
            q[:, :, s] = ang
            Rs[s] = _rot_axis_angle(Ax[:, :, s], ang)
    return q, active


def fk_from_angles(q, spec, dof, rest=None, root_R=None):
    """Hinge angles -> feasible GLOBAL rotations (and root-relative positions).

    root_R [B,3,3]: the root's global orientation. The root is a FREE joint (or, for
    SMPL/BVH, a ball joint) — its rotation is data, not a hinge angle — so it is
    passed through. None -> identity (root-relative output)."""
    B, J = q.shape[0], q.shape[1]
    dev, dt = q.device, q.dtype
    parents = spec.parents
    off = torch.as_tensor(np.asarray(spec.rest_offsets), dtype=dt, device=dev)
    A, active = _slot_axes(dof.to(dev, dt))
    if rest is None:
        rest = torch.eye(3, device=dev, dtype=dt).expand(J, 3, 3)
    rest = rest.to(dev, dt)
    Ax = A[None].expand(B, J, 3, 3)
    L = _rot_axis_angle(Ax[:, :, 0], q[:, :, 0]) \
        @ _rot_axis_angle(Ax[:, :, 1], q[:, :, 1]) \
        @ _rot_axis_angle(Ax[:, :, 2], q[:, :, 2]) @ rest[None]      # [B,J,3,3]
    gR = [None] * J; gp = [None] * J
    for j in range(J):
        p = int(parents[j])
        if p < 0:
            gR[j] = L[:, j] if root_R is None else root_R.to(dev, dt)
            gp[j] = torch.zeros(B, 3, device=dev, dtype=dt)
        else:
            gR[j] = gR[p] @ L[:, j]
            gp[j] = gp[p] + (gR[p] @ off[j])
    return torch.stack(gR, 1), torch.stack(gp, 1)


# --------------------------------------------------------------------------
# GLOBAL fit: one least-squares problem over the WHOLE kinematic chain
# --------------------------------------------------------------------------
# The greedy fit above is per-joint. Off the manifold that is the wrong objective:
# it minimizes each joint's LOCAL residual, but the thing we care about is the
# GLOBAL pose, where a parent's residual is inherited by every descendant. The
# global fit minimizes
#       sum_j w_rot_j |FK(q)_j - T_j|_F^2  +  sum_j w_pos_j |p(q)_j - t_j|^2
#                                          +  w_ref |q - q_ref|^2
# subject to  lo <= q <= hi,  by damped Gauss-Newton (Levenberg-Marquardt) with
# ANALYTIC Jacobians, batched over frames, warm-started from the greedy solution.
#
# Jacobian: rotating slot s of joint k by dq pre-multiplies every descendant's
# global rotation by exp(dq [w]_x) about the WORLD axis
#       w_{k,s} = G_parent(k) rot(a_0,q_0)...rot(a_{s-1},q_{s-1}) a_s
# so   d G_j / d q_{k,s} = [w_{k,s}]_x G_j            (k ancestor-or-self of j)
#      d p_j / d q_{k,s} = w_{k,s} x (p_j - p_k)      (same support)
# Exact, no autograd, no finite differences.


def _ancestor_mask(parents):
    """anc[j,k] = True iff k is an ancestor of j or j itself. [J,J] bool."""
    parents = np.asarray(parents)
    J = len(parents)
    anc = np.zeros((J, J), dtype=bool)
    for j in range(J):
        k = j
        while k >= 0:
            anc[j, k] = True
            k = int(parents[k])
    return anc


def _limits_open(dof, soft_margin):
    """Like _limits but unlimited slots get +-inf (the raw helper returns 0,0 there,
    which would pin every free joint to zero if used as a box)."""
    lo, hi, limited = _limits(dof, soft_margin)
    inf = torch.full_like(lo, float("inf"))
    return torch.where(limited, lo, -inf), torch.where(limited, hi, inf), limited


def _fk_axes(q, A, rest, parents, off, root_R):
    """FK + world-frame slot axes. -> gR [B,J,3,3], gp [B,J,3], W [B,J,3,3]
    (W[:,j,s] is the world axis of slot s of joint j)."""
    B, J = q.shape[0], q.shape[1]
    dev, dt = q.device, q.dtype
    Ax = A[None].expand(B, J, 3, 3)
    R0 = _rot_axis_angle(Ax[:, :, 0], q[:, :, 0])
    R1 = _rot_axis_angle(Ax[:, :, 1], q[:, :, 1])
    R2 = _rot_axis_angle(Ax[:, :, 2], q[:, :, 2])
    P1 = R0 @ R1
    L = P1 @ R2 @ rest[None]
    I = torch.eye(3, device=dev, dtype=dt).expand(B, 3, 3)
    gR = [None] * J; gp = [None] * J; W = [None] * J
    zero = torch.zeros(B, 3, device=dev, dtype=dt)
    for j in range(J):
        p = int(parents[j])
        Gp = I if p < 0 else gR[p]
        if p < 0:
            gR[j] = root_R
            gp[j] = zero
        else:
            gR[j] = gR[p] @ L[:, j]
            gp[j] = gp[p] + (gR[p] @ off[j])
        W[j] = torch.stack([
            (Gp @ Ax[:, j, 0, :, None]).squeeze(-1),
            (Gp @ R0[:, j] @ Ax[:, j, 1, :, None]).squeeze(-1),
            (Gp @ P1[:, j] @ Ax[:, j, 2, :, None]).squeeze(-1)], 1)
    return torch.stack(gR, 1), torch.stack(gp, 1), torch.stack(W, 1)


def _skew(v):
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], -1),
        torch.stack([v[..., 2], z, -v[..., 0]], -1),
        torch.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def lm_fit(target_R, spec, dof, q_init, rest=None, root_R=None, target_p=None,
           w_rot=None, w_pos=None, q_ref=None, w_ref=0.0, clamp=True,
           soft_margin=1.0, iters=25, lam0=1e-3, tol=1e-12, return_info=False):
    """Batched projected Levenberg-Marquardt over the whole chain.

    target_R [B,J,3,3] / target_p [B,J,3]: per-joint goals (any weight may be 0).
    w_rot / w_pos [J] or scalar: per-joint weights. w_ref: pull toward q_ref (the
    null-space reference — used by the constrained pass to stay near the model's
    intent). Box constraints are enforced by clamping inside the loop, with
    per-frame accept/reject on the true cost, so a rejected step only raises that
    frame's damping. Returns q [B,J,3] (+ info dict if return_info)."""
    B, J = q_init.shape[0], q_init.shape[1]
    dev, dt = q_init.device, q_init.dtype
    parents = np.asarray(spec.parents)
    off = torch.as_tensor(np.asarray(spec.rest_offsets), dtype=dt, device=dev)
    A, active = _slot_axes(dof.to(dev, dt))
    if rest is None:
        rest = torch.eye(3, device=q_init.device, dtype=dt).expand(J, 3, 3)
    rest = rest.to(q_init.device, dt)
    root = int(np.where(parents < 0)[0][0])
    if root_R is None:
        root_R = target_R[:, root]
    root_R = root_R.to(q_init.device, dt)

    # ---- parameter set: every active slot except the root's (the root's global
    # rotation is passed through, so its angles do not move the pose at all).
    par = active.clone(); par[root] = False
    pj, ps = par.nonzero(as_tuple=True)                       # [P]
    P = int(pj.numel())
    q = q_init.clone()
    if P == 0:
        return (q, {"iters": 0, "cost": 0.0}) if return_info else q

    lo, hi, limited = _limits_open(dof.to(q_init.device, dt), soft_margin)
    lo_p, hi_p = lo[pj, ps], hi[pj, ps]
    if not clamp:
        lo_p = torch.full_like(lo_p, -float("inf")); hi_p = torch.full_like(hi_p, float("inf"))

    anc = torch.as_tensor(_ancestor_mask(parents), device=q_init.device)   # [J,J]
    sup = anc[:, pj]                                          # [J,P] support of each param

    ones = torch.ones(J, device=q_init.device, dtype=dt)
    w_rot = ones if w_rot is None else (ones * w_rot if np.isscalar(w_rot)
                                        else w_rot.to(q_init.device, dt))
    w_pos = torch.zeros(J, device=q_init.device, dtype=dt) if w_pos is None else (
        ones * w_pos if np.isscalar(w_pos) else w_pos.to(q_init.device, dt))
    use_pos = bool((w_pos > 0).any()) and target_p is not None
    sr = w_rot.sqrt()[None, :, None, None]                    # [1,J,1,1]
    srj = w_rot.sqrt().view(1, J, 1, 1, 1)                    # for the [B,J,P,3,3] block
    sp = w_pos.sqrt()[None, :, None] if use_pos else None
    use_ref = w_ref > 0 and q_ref is not None
    if use_ref:
        qref_p = q_ref[:, pj, ps]
        sref = float(np.sqrt(w_ref))

    def cost_of(qq):
        gR, gp, _ = _fk_axes(qq, A, rest, parents, off, root_R)
        c = ((sr * (gR - target_R)) ** 2).flatten(1).sum(-1)
        if use_pos:
            c = c + ((sp * (gp - target_p)) ** 2).flatten(1).sum(-1)
        if use_ref:
            c = c + w_ref * ((qq[:, pj, ps] - qref_p) ** 2).sum(-1)
        return c, gR, gp

    lam = torch.full((B,), float(lam0), device=q_init.device, dtype=dt)
    cost, _, _ = cost_of(q)
    eye_p = torch.eye(P, device=q_init.device, dtype=dt)[None]
    hist = [float(cost.mean())]
    for _ in range(int(iters)):
        gR, gp, W = _fk_axes(q, A, rest, parents, off, root_R)
        w = W[:, pj, ps]                                      # [B,P,3] world axes
        S = _skew(w)                                          # [B,P,3,3]
        # d gR_j / d q_p  = [w_p]_x gR_j   on the support
        dR = torch.einsum("bpik,bjkl->bjpil", S, gR)          # [B,J,P,3,3]
        dR = dR * sup[None, :, :, None, None] * srj
        Jm = dR.reshape(B, J, P, 9).permute(0, 1, 3, 2).reshape(B, J * 9, P)
        r = (sr * (gR - target_R)).reshape(B, J * 9)
        if use_pos:
            rel = gp[:, :, None, :] - gp[:, pj][:, None, :, :]      # [B,J,P,3]
            dp = torch.cross(w[:, None, :, :].expand_as(rel), rel, dim=-1)
            dp = dp * sup[None, :, :, None] * sp[..., None]
            Jm = torch.cat([Jm, dp.permute(0, 1, 3, 2).reshape(B, J * 3, P)], 1)
            r = torch.cat([r, (sp * (gp - target_p)).reshape(B, J * 3)], 1)
        if use_ref:
            Jm = torch.cat([Jm, sref * eye_p.expand(B, P, P)], 1)
            r = torch.cat([r, sref * (q[:, pj, ps] - qref_p)], 1)
        JtJ = Jm.transpose(1, 2) @ Jm
        Jtr = (Jm.transpose(1, 2) @ r[..., None]).squeeze(-1)
        d = JtJ.diagonal(dim1=-2, dim2=-1)
        for _try in range(6):
            H = JtJ + (lam[:, None] * d.clamp_min(1e-9))[:, :, None] * eye_p + 1e-12 * eye_p
            try:
                step = torch.linalg.solve(H, -Jtr[..., None]).squeeze(-1)
            except Exception:
                step = torch.linalg.lstsq(H, -Jtr[..., None]).solution.squeeze(-1)
            qn = q.clone()
            qn[:, pj, ps] = (q[:, pj, ps] + step).clamp(lo_p[None], hi_p[None])
            cn, _, _ = cost_of(qn)
            ok = cn < cost
            if bool(ok.any()):
                q = torch.where(ok[:, None, None], qn, q)
                cost = torch.where(ok, cn, cost)
                lam = torch.where(ok, (lam / 3).clamp_min(1e-9), lam * 4)
                break
            lam = lam * 4
        hist.append(float(cost.mean()))
        if len(hist) > 2 and abs(hist[-2] - hist[-1]) < tol * max(hist[-1], 1e-9):
            break
    if return_info:
        return q, {"iters": len(hist) - 1, "cost": hist, "n_par": P}
    return q


def fit_angles(global_rot6d, spec, dof, soft_margin=1.0, rest=None, clamp=True,
               iters=32, method="greedy", lm_iters=25, w_pos=0.0, root_R=None,
               return_info=False):
    """GLOBAL rotations -> per-DOF hinge angles.  Returns (q [B,J,3], active [J,3]).

    method="greedy" (default, unchanged behaviour): exact per-joint local fit,
        root-down. Cheap, exact on feasible input, but compounds error off-manifold.
    method="global": whole-chain damped Gauss-Newton on the SAME box, warm-started
        from the greedy solution. Minimizes the global pose residual
        sum_j |FK(q)_j - R_pred_j|_F^2 (+ w_pos * position residual).
    """
    Rg = rot6d_to_matrix(global_rot6d) if global_rot6d.shape[-1] == 6 else global_rot6d
    q, active = _fit_angles_greedy(Rg, spec, dof, soft_margin, rest=rest,
                                   clamp=clamp, iters=iters)
    if method == "greedy":
        return (q, active, {"iters": 0}) if return_info else (q, active)
    if method != "global":
        raise ValueError(f"unknown method {method!r} (greedy|global)")
    root = int(np.where(np.asarray(spec.parents) < 0)[0][0])
    if root_R is None:
        root_R = Rg[:, root]
    tp = None
    if w_pos > 0:
        off = torch.as_tensor(np.asarray(spec.rest_offsets), dtype=Rg.dtype, device=Rg.device)
        gp = [None] * Rg.shape[1]
        for j, p in enumerate(np.asarray(spec.parents)):
            gp[j] = torch.zeros(Rg.shape[0], 3, dtype=Rg.dtype, device=Rg.device) \
                if int(p) < 0 else gp[int(p)] + (Rg[:, int(p)] @ off[j])
        tp = torch.stack(gp, 1)
    out = lm_fit(Rg, spec, dof, q, rest=rest, root_R=root_R, target_p=tp,
                 w_rot=1.0, w_pos=w_pos, clamp=clamp, soft_margin=soft_margin,
                 iters=lm_iters, return_info=return_info)
    if return_info:
        return out[0], active, out[1]
    return out, active


def project(global_rot6d, spec, dof, soft_margin=1.0, rest=None, clamp=True,
            method="global", lm_iters=25, w_pos=0.0):
    """One-shot feasibility projection. Returns (rot6d, pos, q).
    The root's global rotation is preserved exactly (it has no hinge to project).

    method="global" (DEFAULT) = whole-chain LM; method="greedy" = the old per-joint
    fit, kept for reproducing the comparison. Measured on E19_K128_0805 predictions
    (human_smpl -> unitree_g1_29dof, 600 frames): pre-projection 10.084 deg,
    greedy 12.687 (projection HURT), global 8.860 (projection HELPS), both 0% limit
    violations. w_pos > 0 adds a joint-position term (in spec offset units^2); it
    did not help there, so it defaults off."""
    Rg = rot6d_to_matrix(global_rot6d) if global_rot6d.shape[-1] == 6 else global_rot6d
    root = int(np.where(np.asarray(spec.parents) < 0)[0][0])
    q, _ = fit_angles(Rg, spec, dof, soft_margin, rest=rest, clamp=clamp,
                      method=method, lm_iters=lm_iters, w_pos=w_pos)
    gR, gp = fk_from_angles(q, spec, dof, rest=rest, root_R=Rg[:, root])
    return matrix_to_rot6d(gR), gp, q


def project_constrained(global_rot6d, spec, dof, joints=(), target_pos=None,
                        target_rot=None, rest=None, soft_margin=1.0,
                        w_task_pos=1e4, w_task_rot=1e4, w_ref=1e-3,
                        method="global", lm_iters=60, clamp=True,
                        vel_joints=None, vel_frames=None, vel_pos_target=None,
                        vel_rot_target=None, dt=1.0, w_vel_pos=None, w_vel_rot=None):
    """Feasibility projection with a few joints PINNED as hard task-space targets.

    joints [K] int: the constrained joints. target_pos [B,K,3] (root-relative, spec
    offset units) / target_rot [B,K,3,3]: their goals; either may be None.
    Stage 1 projects onto the mechanical manifold (`method`), stage 2 runs the same
    LM with the task joints weighted ~1e4 and a small pull (`w_ref`) toward the
    stage-1 angles as the null-space reference, so the unconstrained DOFs keep the
    model's pose. If the ROOT is constrained its target rotation is applied exactly
    (it is passed through, not solved for). Returns (rot6d, pos, q, q_stage1).

    ---- se(3) VELOCITY constraints (optional) --------------------------------
    "Contact must not slam": at the moment a cup meets the table the end effector's
    velocity must be ~0, not just its position pinned. These arguments constrain the
    finite-difference velocity between CONSECUTIVE frames of the batch (the batch is
    then interpreted as a time-ordered window at 1/dt fps):

      vel_joints  [Kv] int : joints whose velocity is constrained.
      vel_frames  [F] int  : frame indices t (each >= 1) where the constraint binds
                             the pair (t-1, t). Processed in ascending order.
      vel_pos_target [F,Kv,3] : target LINEAR velocity of each joint origin,
                             root-relative spec units / second. None -> zeros
                             ("come to rest"). NOTE positions here are root-relative;
                             a world-frame target must have the root's own velocity
                             subtracted by the caller.
      vel_rot_target [F,Kv,3] : optional target ANGULAR velocity omega (rad/s),
                             defined by R_t = R_{t-1} @ exp(dt [omega]_x) (body
                             frame of the joint at t-1). None -> angular part free.
      dt : seconds per frame.  w_vel_pos / w_vel_rot : task weights, default to
      w_task_pos / w_task_rot.

    FORM: sequential, not block-joint. Given frame t-1's SOLVED pose, the velocity
    constraint is exactly a position target  p_t = p_(t-1) + dt v  (rotation target
    R_t = R_(t-1) exp(dt [omega]_x)) — i.e. the machinery above, unchanged. Frames
    without velocity constraints are solved batched as before; constrained frames
    are then solved one at a time, each seeing its predecessor's solution. Chosen
    over solving frame pairs jointly because (a) it reuses the validated LM task
    path instead of a new 2P-variable block solver (LM is O(P^3): ~8x per pair),
    (b) it is causal, matching streaming execution, and (c) the predecessor frame
    SHOULD stay put — it is either the model's intent or an already-executed frame.
    Static targets on a vel joint are overridden by the velocity target on vel
    frames (two position targets for one joint cannot both bind)."""
    Rg = rot6d_to_matrix(global_rot6d) if global_rot6d.shape[-1] == 6 else global_rot6d
    B, J = Rg.shape[0], Rg.shape[1]
    dev, dtp = Rg.device, Rg.dtype
    root = int(np.where(np.asarray(spec.parents) < 0)[0][0])
    joints = list(int(j) for j in joints)
    root_R = Rg[:, root]
    if target_rot is not None and root in joints:
        root_R = target_rot[:, joints.index(root)].to(dev, dtp)
    q0, _ = fit_angles(Rg, spec, dof, soft_margin, rest=rest, clamp=clamp,
                       method=method, lm_iters=lm_iters, root_R=root_R)
    # stage 2: task targets on the constrained joints only
    TR = Rg.clone(); TP = torch.zeros(B, J, 3, device=dev, dtype=dtp)
    wr = torch.zeros(J, device=dev, dtype=dtp); wp = torch.zeros(J, device=dev, dtype=dtp)
    for k, j in enumerate(joints):
        if target_rot is not None:
            TR[:, j] = target_rot[:, k].to(dev, dtp); wr[j] = w_task_rot
        if target_pos is not None:
            TP[:, j] = target_pos[:, k].to(dev, dtp); wp[j] = w_task_pos

    vf = sorted(int(t) for t in (vel_frames if vel_frames is not None else []))
    if vel_joints is None or len(vf) == 0:
        q = lm_fit(TR, spec, dof, q0, rest=rest, root_R=root_R, target_p=TP,
                   w_rot=wr, w_pos=wp, q_ref=q0, w_ref=w_ref, clamp=clamp,
                   soft_margin=soft_margin, iters=lm_iters)
        gR, gp = fk_from_angles(q, spec, dof, rest=rest, root_R=root_R)
        return matrix_to_rot6d(gR), gp, q, q0

    if any(t < 1 or t >= B for t in vf):
        raise ValueError(f"vel_frames must be in [1, {B - 1}] (each t binds (t-1, t))")
    vel_joints = list(int(j) for j in vel_joints)
    w_vp = w_task_pos if w_vel_pos is None else w_vel_pos
    w_vr = w_task_rot if w_vel_rot is None else w_vel_rot
    # 1) frames WITHOUT velocity constraints: batched, exactly as before
    free = [t for t in range(B) if t not in set(vf)]
    q = q0.clone()
    if free:
        q[free] = lm_fit(TR[free], spec, dof, q0[free], rest=rest,
                         root_R=root_R[free], target_p=TP[free], w_rot=wr, w_pos=wp,
                         q_ref=q0[free], w_ref=w_ref, clamp=clamp,
                         soft_margin=soft_margin, iters=lm_iters)
    # 2) velocity frames: ascending, each anchored to its predecessor's SOLUTION
    for i, t in enumerate(vf):
        gRp, gpp = fk_from_angles(q[t - 1:t], spec, dof, rest=rest,
                                  root_R=root_R[t - 1:t])
        TRt, TPt = TR[t:t + 1].clone(), TP[t:t + 1].clone()
        wrt, wpt = wr.clone(), wp.clone()
        for k, j in enumerate(vel_joints):
            v = (torch.zeros(3, device=dev, dtype=dtp) if vel_pos_target is None
                 else torch.as_tensor(vel_pos_target[i, k], device=dev, dtype=dtp))
            TPt[0, j] = gpp[0, j] + dt * v
            wpt[j] = w_vp
            if vel_rot_target is not None:
                om = torch.as_tensor(vel_rot_target[i, k], device=dev, dtype=dtp)
                ang = om.norm().clamp_min(1e-12)
                TRt[0, j] = gRp[0, j] @ _rot_axis_angle(om / ang, ang * dt)
                wrt[j] = w_vr
        q[t:t + 1] = lm_fit(TRt, spec, dof, q[t:t + 1], rest=rest,
                            root_R=root_R[t:t + 1], target_p=TPt, w_rot=wrt,
                            w_pos=wpt, q_ref=q0[t:t + 1], w_ref=w_ref, clamp=clamp,
                            soft_margin=soft_margin, iters=lm_iters)
    gR, gp = fk_from_angles(q, spec, dof, rest=rest, root_R=root_R)
    return matrix_to_rot6d(gR), gp, q, q0


def temporal_report(q_before, q_after):
    """Frame-to-frame angle jumps before/after projection — branch flips show up
    as a fat tail, so report p99 and max, not just the mean."""
    def stats(q):
        d = (q[1:] - q[:-1]).abs()
        d = torch.rad2deg(d[d > 0])
        if d.numel() == 0:
            return dict(mean=0.0, p99=0.0, max=0.0)
        return dict(mean=round(float(d.mean()), 3),
                    p99=round(float(d.quantile(0.99)), 3),
                    max=round(float(d.max()), 3))
    return {"before": stats(q_before), "after": stats(q_after)}
