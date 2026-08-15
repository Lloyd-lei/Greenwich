"""Equator — the temporal layer: tokens, bridges, retiming, likelihood.

A3 compresses a motion window's codes into 32 discrete tokens (15,625-way each)
with an endpoint conditioner; B3 is the masked prior P(interior | start, goal,
n). Between them the product gets: endpoint-conditioned generation, sparse
pins, any-duration retiming of the same tokens, and an AR likelihood for any
given token sequence — the number the refiner's 70% synergy gate reads.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .nets.bridge import V5, BridgeA, BridgeB, tokens_of


class Equator:
    def __init__(self, a3: BridgeA, b3: BridgeB, device: str):
        self.a3 = a3
        self.b3 = b3
        self.device = device
        self._W = 5 ** torch.arange(6, device=device)

    @classmethod
    def load(cls, a_dir=None, b_dir=None, device: str | None = None):
        from ..config import CONFIG
        from ..weights import resolve
        device = device or CONFIG.device
        a_run = Path(a_dir) if a_dir else resolve("equator_a")
        b_run = Path(b_dir) if b_dir else resolve("equator_b")
        acfg = json.load(open(a_run / "config.json"))
        a3 = BridgeA(d=acfg.get("d_model", 256),
                     enc_layers=acfg.get("enc_layers", 4),
                     dec_layers=acfg.get("dec_layers", 4),
                     n_content=acfg.get("n_content", 32),
                     time_local=acfg.get("time_local", False),
                     use_root=acfg.get("use_root", False)).to(device)
        a3.load_state_dict(_state(a_run))
        a3.eval()
        for p in a3.parameters():
            p.requires_grad_(False)
        b3 = BridgeB(ep_dim=a3.d).to(device)
        b3.load_state_dict(_state(b_run))
        b3.eval()
        return cls(a3, b3, device)

    # ------------------------------------------------------------- tokens ---
    @torch.no_grad()
    def tokenize(self, codes: torch.Tensor):
        """[T,256,20] int codes -> (tokens [32], endpoints).

        Uses the B3-era token contract (tokens_of, unmasked) — the checkpoints
        were trained on it; the masked variant is NOT interchangeable
        (0.15% agreement, +15.2 nats)."""
        tok, ep = tokens_of(self.a3, codes[None].to(self.device))
        return tok[0], ep

    @torch.no_grad()
    def detokenize(self, tokens: torch.Tensor, endpoints, n_frames: int,
                   boundary_codes: torch.Tensor | None = None,
                   rot_codes: torch.Tensor | None = None):
        """32 tokens (+ endpoint features) -> [n,256,20] int codes.

        Same tokens, ANY n = retiming (validated: 0.5x/1x/2x on 300-frame
        clips). boundary_codes [2,256] pins the first/last frame's pose slots
        (A3's endpoint pass-through).

        rot_codes [n,128,20]: the ROTATION-stream codes to carry. A3 only ever
        learned to reconstruct the pose stream — its argmax on slots 128:256
        decodes ~24 m off (measured 0814); the research stack always fed that
        stream from ground-truth rows (bridge_demo.unpack_root). Pass the raw
        stream whenever you have it; the argmax fallback exists only for
        self-consistency metrics (eval_gate retime), never for playback."""
        z = self.a3.fsq.from_ints(
            ((tokens[:, None] // self._W) % 5)[None].float())[None]
        logits = self.a3.decode(z, endpoints, n_frames)
        codes = logits.argmax(-1)[0]
        if rot_codes is not None:
            codes[:, 128:] = rot_codes.long().to(self.device)
        if boundary_codes is not None:
            codes[0, :128] = boundary_codes[0, :128].to(self.device)
            codes[-1, :128] = boundary_codes[1, :128].to(self.device)
        return codes

    @torch.no_grad()
    def endpoints_from_codes(self, codes4: torch.Tensor):
        """[4,256,20] boundary codes (start[0:2] + goal[-2:]) -> endpoint
        features. This is what makes a library entry fully generative: 32
        tokens + 4 boundary frames regenerate the motion at any duration on
        any body."""
        return self.a3.endpoints(codes4[None].long().to(self.device))

    # ------------------------------------------------------------- prior ----
    @torch.no_grad()
    def sample_tokens(self, endpoints, n_frames: int, temperature: float = 0.9,
                      seed: int = 0, pins: dict[int, int] | None = None,
                      goal_drop: bool = False):
        """Confidence-ordered exact sampling (one slot at a time — never
        argmax-all-at-once). pins = {slot: token} hard constraints."""
        gen = torch.Generator(device=self.device).manual_seed(seed)
        tok = torch.zeros(1, 32, dtype=torch.long, device=self.device)
        hidden = torch.ones(1, 32, dtype=torch.bool, device=self.device)
        for s, v in (pins or {}).items():
            tok[0, s] = int(v)
            hidden[0, s] = False
        gd = torch.tensor([goal_drop], device=self.device)
        nf = torch.tensor([float(n_frames)], device=self.device)
        while bool(hidden.any()):
            logits = self.b3(tok, hidden, endpoints, nf, goal_drop=gd)
            probs = F.softmax(logits[0] / temperature, -1)
            conf = probs.max(-1).values.masked_fill(~hidden[0], -1)
            slot = int(conf.argmax())
            tok[0, slot] = torch.multinomial(probs[slot], 1, generator=gen)
            hidden[0, slot] = False
        return tok[0]

    @torch.no_grad()
    def token_nll(self, tokens: torch.Tensor, endpoints, n_frames: int):
        """Per-token NLL of a GIVEN sequence under B3 (full mask) — the
        synergy-gate ruler."""
        tok = tokens[None].long().to(self.device)
        mask = torch.ones_like(tok, dtype=torch.bool)
        nf = torch.tensor([float(n_frames)], device=self.device)
        logits = self.b3(tok, mask, endpoints, nf)
        return F.cross_entropy(logits[0], tok[0], reduction="none")


def _state(run: Path) -> dict:
    st = run / "model.safetensors"
    if st.exists():
        from safetensors.torch import load_file
        return load_file(st)
    return torch.load(run / "model.pt", map_location="cpu")
