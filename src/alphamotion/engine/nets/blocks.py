"""Topology-agnostic single-frame SE3 autoencoder.

One shared network for every skeleton. A skeleton enters ONLY as conditioning
(per-joint geometry + graph structure). No per-embodiment parameters, no
hand-assigned part semantics -- pure geometry + graph + scale (bitter lesson).

Ablation toggle `structured`: if True, add a hand-crafted swing/twist output
head as the "prior" arm of the A/B test. Default False = plain.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rotations import rot6d_to_matrix, fk_global, matrix_to_rot6d


def hop_distance(parents: np.ndarray) -> np.ndarray:
    """All-pairs hop distance on the (undirected) kinematic tree. [J,J]."""
    J = len(parents)
    INF = 10**6
    D = np.full((J, J), INF, np.int64)
    adj = [[] for _ in range(J)]
    for j, p in enumerate(parents):
        if p >= 0:
            adj[j].append(int(p)); adj[int(p)].append(j)
    for s in range(J):
        D[s, s] = 0
        stack = [s]
        seen = {s}
        # BFS
        from collections import deque
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); D[s, v] = D[s, u] + 1; q.append(v)
    return D


class FSQ(nn.Module):
    """Finite Scalar Quantization (Mentzer et al. 2023). No codebook, no collapse.
    Each latent token is projected to `len(levels)` dims, each bounded and rounded
    to its number of levels with a straight-through gradient. This is the discrete
    bottleneck whose codes become the vocabulary for the downstream AR generator.
    """
    def __init__(self, d_model, levels=(8, 8, 8, 5, 5, 5)):
        super().__init__()
        self.levels = list(levels)
        self.f = len(self.levels)
        self.register_buffer("_levels", torch.tensor(self.levels, dtype=torch.float32))
        self.proj_in = nn.Linear(d_model, self.f)
        self.proj_out = nn.Linear(self.f, d_model)
        self.bits = float(sum(np.log2(l) for l in self.levels))

    def quantize(self, z):
        # z: [...,f] unbounded -> bounded to [-1,1]-ish then to level grid
        half = (self._levels - 1) / 2
        # bound with tanh scaled so range covers the levels
        zt = torch.tanh(z) * half
        zr = torch.round(zt)
        # straight-through
        zq = zt + (zr - zt).detach()
        # normalize to [-1,1] for the decoder
        return zq / half

    def forward(self, z_tokens):
        # z_tokens: [B, K, d]
        f = self.proj_in(z_tokens)
        q = self.quantize(f)
        return self.proj_out(q), q

    # --- discrete-interface helpers (AR-LM vocabulary; values consistent with forward) ---
    def features(self, z_tokens):
        """Pre-quantization continuous features [B,K,f] (the diffusion compute space)."""
        return self.proj_in(z_tokens)

    def ints(self, f):
        """[...,f] features -> integer levels [1,...,f]. Leading dim = stage (1).
        Offset = ceil(half): half is half-integer for even level counts, so adding
        `half` then truncating with .long() would break the bijection."""
        half = (self._levels - 1) / 2
        return (torch.round(torch.tanh(f) * half) + torch.ceil(half)).long()[None]

    def from_ints(self, ints):
        """[1,...,f] integer levels -> decoder-ready z [...,d] (same values as forward's output)."""
        half = (self._levels - 1) / 2
        q = (ints[0].float() - torch.ceil(half)) / half
        return self.proj_out(q)


class ResidualFSQ(nn.Module):
    """Residual FSQ (RVQ-style, codebook-free). Stage k quantizes the residual left
    by stages 1..k-1, so precision improves with depth at a controlled bit cost.
    This is the fix for single-FSQ quantization error found in R1 (robot 3.4->19deg).
    Total bits/token = n_stages * sum(log2(levels)).
    """
    def __init__(self, d_model, n_stages=2, levels=(8, 8, 8, 5, 5, 5)):
        super().__init__()
        self.n_stages = n_stages
        self.f = len(levels)
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.float32))
        self.proj_in = nn.Linear(d_model, self.f)
        self.proj_out = nn.Linear(self.f, d_model)
        # each stage gets its own learned scale so later stages resolve finer residuals
        self.stage_scale = nn.Parameter(torch.ones(n_stages))
        self.bits = float(sum(np.log2(l) for l in levels)) * n_stages

    def _q(self, x, scale):
        half = (self._levels - 1) / 2
        xt = torch.tanh(x * scale) * half
        xr = torch.round(xt)
        xq = xt + (xr - xt).detach()
        return xq / half / scale.clamp(min=1e-3)

    def forward(self, z_tokens):
        f = self.proj_in(z_tokens)
        acc = torch.zeros_like(f)
        residual = f
        codes = []
        for k in range(self.n_stages):
            q = self._q(residual, self.stage_scale[k].abs() + 1.0)
            acc = acc + q
            residual = f - acc
            codes.append(q)
        return self.proj_out(acc), torch.stack(codes, 0)

    # --- discrete-interface helpers (values consistent with forward's straight-through path) ---
    def features(self, z_tokens):
        """Pre-quantization continuous features [B,K,f] (the diffusion compute space)."""
        return self.proj_in(z_tokens)

    def _stage_ints(self, x, scale):
        # offset = ceil(half): half-integer `half` (even level counts) + .long() truncation
        # would break the int<->grid bijection
        half = (self._levels - 1) / 2
        xr = torch.round(torch.tanh(x * scale) * half)
        return (xr + torch.ceil(half)).long(), xr / half / scale.clamp(min=1e-3)

    def ints(self, f):
        """[...,f] features -> integer levels per stage [S,...,f] in [0, levels_d)."""
        acc = torch.zeros_like(f); residual = f; out = []
        for k in range(self.n_stages):
            scale = self.stage_scale[k].abs() + 1.0
            ii, q = self._stage_ints(residual, scale)
            out.append(ii)
            acc = acc + q; residual = f - acc
        return torch.stack(out, 0)

    def from_ints(self, ints):
        """[S,...,f] integer levels -> decoder-ready z [...,d]."""
        half = (self._levels - 1) / 2
        acc = None
        for k in range(self.n_stages):
            scale = self.stage_scale[k].abs() + 1.0
            q = (ints[k].float() - torch.ceil(half)) / half / scale.clamp(min=1e-3)
            acc = q if acc is None else acc + q
        return self.proj_out(acc)


class JointFeature(nn.Module):
    """Per-joint conditioning feature (geometry + graph). Topology-agnostic.

    Optionally adds a learned index positional embedding derived from the
    skeleton's own canonical joint order. This is NOT hand-assigned semantics --
    every skeleton generates its own indices from its own traversal -- it just
    lets the decoder disambiguate otherwise-geometrically-similar joints
    (e.g. left vs right limb). Toggle via `use_index`.
    """
    def __init__(self, d_model, use_index=False, max_joints=128, use_sem=False, sem_dim=768,
                 use_dof=False, dof_dim=19):
        super().__init__()
        # input: rest_offset_full(3) + rest_offset_dir(3) + norm_bonelen(1) + depth sinusoid(8) = 15
        # use_dof adds the joint's MECHANICAL spec: 3 DOF slots x [axis(3), limited, half_range/pi,
        # center/pi] + dof_count/3 = 19. Diagnosed as the missing conditioning that made stacked
        # hinges (hip pitch/roll/yaw) indistinguishable — SigLIP name embeddings cannot separate
        # them (cos 0.955 pitch-vs-roll, vs 0.907 knee-vs-hip).
        self.use_dof = use_dof
        self.proj = nn.Linear(15 + (dof_dim if use_dof else 0), d_model)
        self.use_index = use_index
        if use_index:
            self.idx_emb = nn.Embedding(max_joints, d_model)
        self.use_sem = use_sem
        if use_sem:  # SigLIP joint-name semantics: the cross-topology alignment bridge
            self.sem_proj = nn.Sequential(nn.Linear(sem_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, rest_off, bonelen, depth, sem=None, dof=None):
        # rest_off [B,J,3], bonelen [B,J], depth [B,J], sem [B,J,768] or None, dof [B,J,19] or None
        dirn = F.normalize(rest_off, dim=-1)
        d = depth.unsqueeze(-1).float()
        freqs = torch.arange(4, device=d.device).float()
        pe = torch.cat([torch.sin(d / (10 ** freqs)), torch.cos(d / (10 ** freqs))], -1)  # [B,J,8]
        feat = torch.cat([rest_off, dirn, bonelen.unsqueeze(-1), pe], -1)  # [B,J,15]
        if self.use_dof:
            assert dof is not None, "use_dof=True but no dof feature supplied"
            feat = torch.cat([feat, dof], -1)
        out = self.proj(feat)
        if self.use_sem and sem is not None:
            out = out + self.sem_proj(sem)
        if self.use_index:
            J = rest_off.shape[1]
            idx = torch.arange(J, device=rest_off.device).clamp(max=self.idx_emb.num_embeddings - 1)
            out = out + self.idx_emb(idx).unsqueeze(0)
        return out


class GraphBiasAttention(nn.Module):
    """MHA with additive per-head bias from clamped hop distance."""
    def __init__(self, d_model, n_heads, max_hop=16):
        super().__init__()
        self.h = n_heads; self.dk = d_model // n_heads
        self.q = nn.Linear(d_model, d_model); self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model); self.o = nn.Linear(d_model, d_model)
        self.hop_emb = nn.Embedding(max_hop + 2, n_heads)  # +1 pad/unreachable bucket
        self.max_hop = max_hop

    def forward(self, x, hop, key_padding_mask=None):
        B, J, D = x.shape
        q = self.q(x).view(B, J, self.h, self.dk).transpose(1, 2)
        k = self.k(x).view(B, J, self.h, self.dk).transpose(1, 2)
        v = self.v(x).view(B, J, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / (self.dk ** 0.5)  # [B,h,J,J]
        hb = hop.clamp(max=self.max_hop + 1)               # [B,J,J]
        bias = self.hop_emb(hb).permute(0, 3, 1, 2)         # [B,h,J,J]
        att = att + bias
        if key_padding_mask is not None:
            att = att.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        att = att.softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(B, J, D)
        return self.o(out)


class EncBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.a = GraphBiasAttention(d, h); self.n1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.n2 = nn.LayerNorm(d)

    def forward(self, x, hop, kpm):
        x = x + self.a(self.n1(x), hop, kpm)
        x = x + self.ff(self.n2(x))
        return x


class CrossBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, h, batch_first=True)
        self.n1 = nn.LayerNorm(d); self.nq = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.n2 = nn.LayerNorm(d)

    def forward(self, q, kv, key_padding_mask=None):
        a, _ = self.mha(self.nq(q), self.n1(kv), self.n1(kv), key_padding_mask=key_padding_mask)
        q = q + a
        q = q + self.ff(self.n2(q))
        return q


class TopoAE(nn.Module):
    def __init__(self, d_model=256, n_heads=8, enc_layers=4, dec_layers=4,
                 n_latent=8, structured=False, use_index=False, target="local",
                 use_fsq=False, fsq_levels=(8, 8, 8, 5, 5, 5), fsq_stages=1,
                 pool_mode="perceiver"):
        super().__init__()
        self.d = d_model; self.n_latent = n_latent; self.structured = structured
        self.pool_mode = pool_mode                     # "perceiver" (K tokens) | "per_joint"
        self.target = target                          # local | global | global_st (swing-twist)
        self.use_fsq = use_fsq
        if use_fsq:
            self.fsq = (ResidualFSQ(d_model, fsq_stages, fsq_levels) if fsq_stages > 1
                        else FSQ(d_model, fsq_levels))
        else:
            self.fsq = None
        self.jfeat = JointFeature(d_model, use_index=use_index)
        in_dim = 6 if target == "local" else 9        # global modes also feed root-rel position
        self.in_rot = nn.Linear(in_dim, d_model)      # encoder also sees the input pose
        self.enc = nn.ModuleList([EncBlock(d_model, n_heads) for _ in range(enc_layers)])
        self.latent_q = nn.Parameter(torch.randn(n_latent, d_model) * 0.02)
        self.pool = CrossBlock(d_model, n_heads)
        # decoder: per-joint query (target skeleton feature) attends to latent
        self.dec = nn.ModuleList([CrossBlock(d_model, n_heads) for _ in range(dec_layers)])
        self.dec_self = nn.ModuleList([EncBlock(d_model, n_heads) for _ in range(dec_layers)])
        out_dim = {"local": 6, "global": 9, "global_st": 8, "glob2loc": 6}[target]  # glob2loc: global in, local-rot out
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_dim))
        if structured:
            # hand-crafted: predict twist scalar (cos,sin) only; swing comes from geometry
            self.twist_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))

    def encode(self, pose_in, rest_off, bonelen, depth, hop, kpm):
        x = self.in_rot(pose_in) + self.jfeat(rest_off, bonelen, depth)
        for blk in self.enc:
            x = blk(x, hop, kpm)
        B = x.shape[0]
        if self.pool_mode == "per_joint":
            z = x                                     # [B, J, d] keep per-joint detail
        else:
            q = self.latent_q.unsqueeze(0).expand(B, -1, -1)
            z = self.pool(q, x, key_padding_mask=kpm)  # [B, n_latent, d]
        if self.use_fsq:
            z, _ = self.fsq(z)                        # discrete bottleneck (FSQ)
        return z

    def decode(self, z, rest_off, bonelen, depth, hop, kpm):
        qj = self.jfeat(rest_off, bonelen, depth)      # [B,J,d] target skeleton queries
        for cb, sb in zip(self.dec, self.dec_self):
            qj = cb(qj, z)                             # attend to latent
            qj = sb(qj, hop, kpm)                      # self-attn on target graph
        return self.head(qj)                          # [B,J,6]

    def forward(self, batch):
        pose_in = batch.get("pose_in", batch.get("local6d"))
        z = self.encode(pose_in, batch["rest_off"], batch["bonelen"],
                        batch["depth"], batch["hop"], batch["kpm"])
        pred = self.decode(z, batch["rest_off"], batch["bonelen"],
                           batch["depth"], batch["hop"], batch["kpm"])
        return pred, z


def parse_fsq_levels(s):
    """'8,8,8,5,5,5' -> (8,8,8,5,5,5)"""
    return tuple(int(x) for x in str(s).split(","))
