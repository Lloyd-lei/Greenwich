"""Dual-stream topology-agnostic translator: SEPARATE FSQ codes for position and
rotation, so rotation (esp. twist) gets protected bits instead of being starved by
position in a shared bottleneck. Confirmed to unfreeze rotation learning.

encode(any source) -> (z_pos, z_rot) two FSQ code streams -> decode(any target).
"""
from __future__ import annotations
import torch, torch.nn as nn
from .blocks import EncBlock, CrossBlock, FSQ, ResidualFSQ, JointFeature
from .rotations import rot6d_to_matrix, matrix_to_rot6d


class JudgeHead(nn.Module):
    """Per-embodiment sigmoid critics reading the quantized codes ("像不像我负责的具身").
    Sits between codes and decoder; whether its gradients reach the backbone is the
    caller's choice (detach or not) — that is the E14 experimental variable."""
    def __init__(self, d_model, n_emb, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * d_model, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_emb))

    def forward(self, zp, zr):                          # [B,K,d] x2 -> logits [B,n_emb]
        h = torch.cat([zp.mean(1), zp.amax(1), zr.mean(1), zr.amax(1)], -1)
        return self.net(h)


class DualTranslator(nn.Module):
    def __init__(self, d_model=256, n_heads=8, enc_layers=4, dec_layers=4,
                 n_latent=32, fsq_stages=2, fsq_levels=(8, 8, 8, 5, 5, 5), use_sem=True,
                 single_codebook=False, n_judges=0, use_dof=False, rot_head="free",
                 sem_dim=768, vis_residual=False, rot_only=False):
        super().__init__()
        self.rot_head = rot_head
        # 0812: with rot_only the position branch is gone and ALL n_latent
        # slots feed the rotation stream — the ablation showed the position
        # codes cannot reach the render at all, so this spends the same code
        # budget on the half that can.
        self.rot_only = rot_only
        # E28 vis_residual: joints whose pose the ENCODER actually saw are decoded as a
        # RESIDUAL on the observed pose (final = observed (+) delta) instead of being
        # re-invented from the bottleneck — attacks the measured E19 failure where the
        # model's own encode-decode of FULLY VISIBLE input cost 11.88 cm at the wrist.
        # Default False = byte-exact reproduction of every prior run (no extra modules,
        # so old checkpoints strict-load either way).
        self.vis_residual = vis_residual
        if vis_residual:
            # zero-init weights; rot bias = identity in 6D ([1,0,0,0,1,0]) so at init the
            # residual path is exactly "copy the observed pose".
            self.hrot_res = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 6))
            self.hpos_res = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 3))
            nn.init.zeros_(self.hrot_res[1].weight)
            nn.init.zeros_(self.hpos_res[1].weight)
            nn.init.zeros_(self.hpos_res[1].bias)
            with torch.no_grad():
                self.hrot_res[1].bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))
        self.judge = JudgeHead(d_model, n_judges) if n_judges else None
        # use_sem=True: frozen joint-name text semantics as cross-topology bridge; drop harmful
        # index emb. sem_dim = the text encoder's width (768 SigLIP / 1024 Qwen3-Embedding).
        self.jf = JointFeature(d_model, use_index=not use_sem, use_sem=use_sem, use_dof=use_dof,
                               sem_dim=sem_dim)
        self.inp = nn.Linear(9, d_model)                      # global 6D rot + root-rel pos
        # E0b masked pretraining: replaces a hidden joint's POSE embedding; geometry stays visible
        self.mask_tok = nn.Parameter(torch.randn(d_model) * 0.02)
        self.enc = nn.ModuleList([EncBlock(d_model, n_heads) for _ in range(enc_layers)])
        self.single_codebook = single_codebook
        if single_codebook:
            # E0c: ONE shared query bank (2K tokens) + ONE FSQ, same total bit budget as dual.
            # encode returns (z, z) so decoders/training code are unchanged.
            self.qp = nn.Parameter(torch.randn(2 * n_latent, d_model) * 0.02)
            self.poolp = CrossBlock(d_model, n_heads)
            self.fsqp = ResidualFSQ(d_model, fsq_stages, fsq_levels) if fsq_stages > 1 else FSQ(d_model, fsq_levels)
            self.bits = self.fsqp.bits * 2 * n_latent
        else:
            self.qp = nn.Parameter(torch.randn(n_latent, d_model) * 0.02)
            self.qr = nn.Parameter(torch.randn(n_latent, d_model) * 0.02)
            self.poolp = CrossBlock(d_model, n_heads)
            self.poolr = CrossBlock(d_model, n_heads)
            self.fsqp = ResidualFSQ(d_model, fsq_stages, fsq_levels) if fsq_stages > 1 else FSQ(d_model, fsq_levels)
            self.fsqr = ResidualFSQ(d_model, fsq_stages, fsq_levels) if fsq_stages > 1 else FSQ(d_model, fsq_levels)
            self.bits = (self.fsqp.bits + self.fsqr.bits) * n_latent
        self.decp = nn.ModuleList([CrossBlock(d_model, n_heads) for _ in range(dec_layers)])
        self.decr = nn.ModuleList([CrossBlock(d_model, n_heads) for _ in range(dec_layers)])
        self.selfp = nn.ModuleList([EncBlock(d_model, n_heads) for _ in range(dec_layers)])
        self.selfr = nn.ModuleList([EncBlock(d_model, n_heads) for _ in range(dec_layers)])
        self.hpos = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 3))
        # rot_head="free"   : 6D GLOBAL rotation per joint (all runs up to E24).
        # rot_head="angles" : E25 arm A. 6D + 3 hinge angles. The 6D branch is used only
        #   for the root (a free joint) and for genuine 3-DOF joints; every hinge joint
        #   gets its LOCAL rotation from the angles, FK'd to global in decode(). The
        #   whole architectural delta is 3 extra output units = 3*d_model+3 params.
        self.hrot = nn.Sequential(nn.LayerNorm(d_model),
                                  nn.Linear(d_model, 9 if rot_head == "angles" else 6))

    def encode(self, x, g, joint_mask=None, return_codes=False):
        """joint_mask [B,J] bool (True = pose hidden -> mask token). return_codes: also
        return the quantized grid values (probe input: leakage / partial-obs stability)."""
        h = self._trunk(x, g, joint_mask)
        B = h.shape[0]
        if self.single_codebook:
            z, cp = self.fsqp(self.poolp(self.qp[None].expand(B, -1, -1), h))
            zr, cr = z, cp
            zp = z
        else:
            if self.rot_only:
                zr, cr = self.fsqr(self.poolr(self.qr[None].expand(B, -1, -1), h))
                zp, cp = zr, cr           # alias: no separate position code
            else:
                zp, cp = self.fsqp(self.poolp(self.qp[None].expand(B, -1, -1), h))
                zr, cr = self.fsqr(self.poolr(self.qr[None].expand(B, -1, -1), h))
        if return_codes:
            return zp, zr, (cp, cr)
        return zp, zr

    def _trunk(self, x, g, joint_mask=None):
        h = self.inp(x)
        if joint_mask is not None:
            h = torch.where(joint_mask[..., None], self.mask_tok.expand_as(h), h)
        h = h + self.jf(g["rest_off"], g["bonelen"], g["depth"], g.get("sem"), g.get("dof"))
        for b in self.enc:
            h = b(h, g["hop"], None)
        return h

    def encode_features(self, x, g, joint_mask=None):
        """Pre-quantization continuous features per stream: (fp, fr) each [B,K,f].
        This is the diffusion compute space; FSQ rounding of these = the token interface."""
        h = self._trunk(x, g, joint_mask)
        B = h.shape[0]
        fp = self.fsqp.features(self.poolp(self.qp[None].expand(B, -1, -1), h))
        if self.single_codebook:
            return fp, fp
        fr = self.fsqr.features(self.poolr(self.qr[None].expand(B, -1, -1), h))
        return fp, fr

    def ints_from_features(self, fp, fr):
        """(fp, fr) -> integer level tensors ([S,B,K,f], [S,B,K,f])."""
        return self.fsqp.ints(fp), (self.fsqp if self.single_codebook else self.fsqr).ints(fr)

    def decode_from_ints(self, ip, ir, g):
        """Integer levels -> full-body (rot6d, pos) on target skeleton g."""
        zp = self.fsqp.from_ints(ip)
        zr = (self.fsqp if self.single_codebook else self.fsqr).from_ints(ir)
        return self.decode(zp, zr, g)

    def decode(self, zp, zr, g, obs=None, vis=None):
        """obs [B,J,9] / vis [B,J] (True = the ENCODER saw this joint's pose): E28
        vis_residual inputs, SAME-SKELETON decode ONLY. For visible joints the output
        becomes observed (+) delta (rotation: R_obs @ R_delta with R_delta init'd at
        identity; position: p_obs + dp). Masked joints decode from the codes as always.

        CROSS-BODY decode (different target skeleton) must pass obs=None: there is no
        observed target pose, the visibility mask is source-side, and the codes must
        keep carrying the full motion for translation — the residual path is therefore
        structurally unable to become an identity shortcut for cross-body decode.
        Callers (dual_common.run_dir, eval harnesses) enforce src == tgt before
        passing obs."""
        qp = self.jf(g["rest_off"], g["bonelen"], g["depth"], g.get("sem"), g.get("dof"))
        qr = qp.clone()
        if not self.rot_only:
            for cb, sb in zip(self.decp, self.selfp):
                qp = cb(qp, zp); qp = sb(qp, g["hop"], None)
        for cb, sb in zip(self.decr, self.selfr):
            qr = cb(qr, zr); qr = sb(qr, g["hop"], None)
        h = self.hrot(qr)
        if self.rot_head == "angles":               # E25 arm A: hinge angles -> FK -> global
            raise RuntimeError("rot_head='angles' is a research-only "
                               "branch not shipped in alphamotion")
            head_to_global6d = None  # unreachable
            assert g.get("kin") is not None, "rot_head='angles' needs g['kin'] (DOF descriptor)"
            h = head_to_global6d(h[..., :6], h[..., 6:9], g["kin"])
        if self.rot_only:
            # 0812: the position head is gone. Returning zeros here was a silent
            # trap — evaluate() computes |pos - gt| and would report |gt| as if
            # it were an error (it printed 21-48 cm of pure garbage in the smoke
            # run). Return the FK positions instead: the same quantity every
            # reported number already comes through, so any caller that wants
            # positions gets the correct ones.
            raise RuntimeError("rot_only checkpoints are research-only; "
                               "the release codec has both streams")
        else:
            pos = self.hpos(qp)
        if self.vis_residual and obs is not None:
            assert vis is not None, "vis_residual decode with obs needs the visibility mask"
            R_obs = rot6d_to_matrix(obs[..., :6])
            R_delta = rot6d_to_matrix(self.hrot_res(qr))
            h_res = matrix_to_rot6d(R_obs @ R_delta)
            p_res = obs[..., 6:9] + self.hpos_res(qp)
            m = vis[..., None]
            h = torch.where(m, h_res, h)
            pos = torch.where(m, p_res, pos)
        return h, pos                               # (global rot6d, pos)
