"""Greenwich — the cross-embodiment spatial codec (release surface).

One latent code space shared by every body. encode() a motion observed on any
registered embodiment; decode() it onto any other. The release checkpoint is
the repair-audited decoder (round-trip fidelity 61% vs the 11% of the
pre-repair decoder, with bit-identical codes) — the property the Atlas and the
synergy gate stand on.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .descriptor import geom_batch, target_geom
from .nets.codec import DualTranslator
from .nets.rotations import SkeletonSpec
from .spatial import build_global, fk_pos

SEM_DIM = {"siglip": 768, "qwen3": 1024}


def build_model(cfg: dict, device: str) -> DualTranslator:
    """Construct the codec from a run config (dual_common.build_model port)."""
    from .nets.blocks import parse_fsq_levels
    model = DualTranslator(
        d_model=cfg.get("d_model", 256),
        n_heads=cfg.get("n_heads", 8),
        enc_layers=cfg.get("enc_layers", 4),
        dec_layers=cfg.get("dec_layers", 4),
        n_latent=cfg.get("n_latent", 32),
        fsq_stages=cfg.get("fsq_stages", 2),
        fsq_levels=parse_fsq_levels(cfg.get("fsq_levels", "8,8,8,5,5,5")),
        single_codebook=cfg.get("single_codebook", False),
        n_judges=cfg.get("n_judges", 0),
        use_dof=cfg.get("use_dof", False),
        rot_head=cfg.get("rot_head", "free"),
        sem_dim=SEM_DIM[cfg.get("sem_encoder", "siglip")],
        vis_residual=cfg.get("vis_residual", False),
        rot_only=cfg.get("rot_only", False),
    ).to(device)
    return model


class Greenwich:
    def __init__(self, model: DualTranslator, cfg: dict, device: str):
        self.model = model
        self.cfg = cfg
        self.device = device
        self._geoms: dict[str, dict] = {}

    # ------------------------------------------------------------- loading --
    @classmethod
    def load(cls, run_dir: str | Path | None = None,
             device: str | None = None) -> "Greenwich":
        from ..config import CONFIG
        from ..weights import resolve
        device = device or CONFIG.device
        run = Path(run_dir) if run_dir else resolve("greenwich")
        cfg = json.load(open(run / "config.json"))
        model = build_model(cfg, device)
        sd = _load_state(run)
        model.load_state_dict(sd, strict=False)
        model.eval()
        return cls(model, cfg, device)

    # ---------------------------------------------------------- descriptors --
    def geom(self, spec: SkeletonSpec, dof) -> dict:
        key = spec.name
        if key not in self._geoms:
            self._geoms[key] = target_geom(spec, dof, self.device)
        return self._geoms[key]

    def pose9(self, local_or_global6d: torch.Tensor, spec: SkeletonSpec,
              is_global: bool = False, reach: float | None = None):
        """[T,J,6] rotations -> the [T,J,9] observation the encoder eats.

        Cached corpora store LOCAL rotations (compose first — feeding local
        rot6d straight to FK keeps bone lengths plausible while displacing
        joints ~12 cm, the classic silent bug); adapters that already composed
        global rotations pass is_global=True.
        """
        if is_global:
            gr6 = local_or_global6d.float()
            gp = torch.from_numpy(fk_pos(gr6.numpy(), spec)).float()
            r = reach or float(gp.norm(dim=-1).max())
        else:
            gr6, gp, r = build_global(local_or_global6d, spec, "cpu")
            gr6, gp = gr6.cpu(), gp.cpu()
            r = reach or r
        return torch.cat([gr6, gp / max(r, 1e-6)], -1).to(self.device), r

    # ------------------------------------------------------------- codes ----
    @torch.no_grad()
    def encode(self, pose9: torch.Tensor, spec: SkeletonSpec, dof):
        """[T,J,9] -> integer codes [T,256,20] (128 pose slots + 128 root)."""
        g = self.geom(spec, dof)
        fp, fr = self.model.encode_features(pose9, geom_batch(g, pose9.shape[0]))
        ipS, irS = self.model.ints_from_features(fp, fr)
        return torch.cat([ipS[0], irS[0]], 1).long()

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, spec: SkeletonSpec, dof):
        """[T,256,20] int codes -> global rot6d [T,J,6] on the target body."""
        return self.decode_full(codes, spec, dof)[0]

    @torch.no_grad()
    def decode_full(self, codes: torch.Tensor, spec: SkeletonSpec, dof):
        """Both decoder heads: (rot6d [T,J,6], pos [T,J,3] root-relative,
        reach-normalised). The position head carries placement information the
        rotation stream alone cannot express on a mismatched skeleton (owner
        call 0814: pose the mesh from position + rot together)."""
        g = self.geom(spec, dof)
        q = (codes.float() - 4.0) / 3.5
        zp = self.model.fsqp.proj_out(q[:, :128])
        zr = self.model.fsqr.proj_out(q[:, 128:])
        rot, pos = self.model.decode(zp, zr, geom_batch(g, codes.shape[0]))
        return rot, pos

    @torch.no_grad()
    def translate(self, pose9: torch.Tensor, src_spec, src_dof,
                  tgt_spec, tgt_dof):
        """Source observation -> target-body global rot6d, one hop."""
        g_s = self.geom(src_spec, src_dof)
        zp, zr = self.model.encode(pose9, geom_batch(g_s, pose9.shape[0]))
        g_t = self.geom(tgt_spec, tgt_dof)
        rot, _ = self.model.decode(zp, zr, geom_batch(g_t, pose9.shape[0]))
        return rot


def _load_state(run: Path) -> dict:
    """model.safetensors preferred; model.pt (pure state_dict) accepted."""
    st = run / "model.safetensors"
    if st.exists():
        from safetensors.torch import load_file
        return load_file(st)
    return torch.load(run / "model.pt", map_location="cpu")
