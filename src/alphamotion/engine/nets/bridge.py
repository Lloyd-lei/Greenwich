"""Equator temporal nets, vendored verbatim from bridge_a.py / bridge_b.py.

BridgeA: motion codes <-> 32 discrete tokens (15625-way, endpoint conditioner).
BridgeB: masked token prior P(interior | start, goal, n).
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import FSQ

V5 = 5 ** 6

def fourier_time(u, n_freq=8):
    """u [B,T] in [0,1] -> [B,T,2*n_freq]."""
    f = (2.0 ** torch.arange(n_freq, device=u.device, dtype=u.dtype)) * torch.pi
    a = u[..., None] * f
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class BridgeA(nn.Module):
    def __init__(self, slots=128, dims=20, levels=9, d=256, n_content=32,
                 enc_layers=5, dec_layers=3, n_heads=8, n_freq=8,
                 time_local=False, use_root=False):
        super().__init__()
        self.slots, self.dims, self.levels, self.d = slots, dims, levels, d
        self.n_content = n_content
        self.time_local, self.use_root = time_local, use_root
        self.streams = 2 if use_root else 1
        # frame embedding: normalized levels -> linear (the "learned embedding of ints")
        self.femb = nn.Sequential(nn.Linear(slots * dims * self.streams, d), nn.LayerNorm(d))
        self.q = nn.Parameter(torch.randn(n_content, d) * 0.02)
        el = nn.TransformerEncoderLayer(d, n_heads, 4 * d, batch_first=True,
                                        norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(el, enc_layers)
        self.fsq = FSQ(d, (5, 5, 5, 5, 5, 5))
        self.tok_id = nn.Parameter(torch.randn(n_content, d) * 0.02)
        self.ep_id = nn.Parameter(torch.randn(2, d) * 0.02)
        self.vel_in = nn.Linear(d, d)
        self.time_in = nn.Linear(2 * n_freq + 1, d)
        dl = nn.TransformerDecoderLayer(d, n_heads, 4 * d, batch_first=True,
                                        norm_first=True, dropout=0.0)
        self.dec = nn.TransformerDecoder(dl, dec_layers)
        self.head = nn.Linear(d, slots * dims * levels * self.streams)

    def embed_frames(self, c):
        """c long [B,T,slots,dims] -> [B,T,d]. C1's FSQ emits ints 0..8 (the 9th is
        the tanh-saturation grid point 4/3.5=1.143 - a REAL level in the decode
        path, model.py edge case found 0810). Map ints to their true grid values."""
        x = (c.float() - 4.0) / 3.5
        return self.femb(x.reshape(*c.shape[:2], -1))

    def encode(self, c):
        """full window codes -> straight-through z tokens [B,32,d].
        time_local (v2, 完全体): token i may only attend frames of clock segment i
        (plus all tokens) — the code becomes TIME-LOCAL, which is what gives
        Stage B's forward/backward masks their meaning."""
        h = self.embed_frames(c)
        B, T = h.shape[0], h.shape[1]
        seq = torch.cat([self.q[None].expand(B, -1, -1), h], 1)
        mask = None
        if self.time_local:
            K = self.n_content
            L = K + T
            mask = torch.zeros(L, L, dtype=torch.bool, device=h.device)
            b = np.round(np.linspace(0, T, K + 1)).astype(int)
            for i in range(K):
                mask[i, K:] = True
                mask[i, K + b[i]:K + max(b[i + 1], b[i] + 1)] = False
                mask[i, :K] = False
        out = self.enc(seq, mask=mask)[:, :self.n_content]
        z, _ = self.fsq(out)
        return z

    def endpoints(self, c):
        """[B,T,slots,dims] -> endpoint kv [B,2,d] with (q,v)-style features."""
        e = self.embed_frames(torch.stack([c[:, 0], c[:, 1], c[:, -2], c[:, -1]], 1))
        q0, v0 = e[:, 0], e[:, 1] - e[:, 0]
        q1, v1 = e[:, 3], e[:, 3] - e[:, 2]
        ep = torch.stack([q0 + self.vel_in(v0), q1 + self.vel_in(v1)], 1)
        return ep + self.ep_id[None]

    def decode(self, z, ep, T_out, pad_mask=None):
        """-> interior logits [B,T_out,slots,dims,levels] (t=0/T-1 rows are junk;
        caller never trains or renders them)."""
        B = z.shape[0]
        u = torch.linspace(0, 1, T_out, device=z.device)[None].expand(B, -1)
        tf = torch.cat([fourier_time(u), torch.full_like(u[..., None], 0.0)
                        + np.log(T_out) / 5.0], -1)
        qs = self.time_in(tf)
        kv = torch.cat([z + self.tok_id[None], ep], 1)
        h = self.dec(qs, kv)
        return self.head(h).reshape(B, T_out, self.streams * self.slots, self.dims, -1)

    def forward(self, c, pad_mask=None):
        z = self.encode(c)
        return self.decode(z, self.endpoints(c), c.shape[1]), z



def load_bridge_a(run, dev):
    """Build BridgeA FROM ITS CONFIG (d_model/layers/flags) and load weights.
    0810 lesson: every consumer hardcoding d=256 died the moment A3 went d384."""
    import json as _j
    cfg = _j.load(open(f"{run}/config.json"))
    m = BridgeA(dims=20,
                d=int(cfg.get("d_model", 256)),
                enc_layers=int(cfg.get("enc_layers", 5)),
                dec_layers=int(cfg.get("dec_layers", 3)),
                time_local=bool(cfg.get("time_local", False)),
                use_root=bool(cfg.get("use_root", False)))
    m.load_state_dict(torch.load(f"{run}/model.pt", map_location=dev))
    return m.to(dev).eval(), cfg




class BridgeB(nn.Module):
    def __init__(self, d=256, n_tok=32, layers=6, n_heads=8, n_freq=8,
                 ep_dim=None):
        super().__init__()
        self.n_tok = n_tok
        self.tok_emb = nn.Embedding(V5 + 1, d)          # +1 = the <mask> id
        self.pos_id = nn.Parameter(torch.randn(n_tok, d) * 0.02)
        self.cond_in = nn.Linear(ep_dim or d, d)        # start/goal features (A's d!)
        self.cond_id = nn.Parameter(torch.randn(3, d) * 0.02)   # start/goal/n
        self.n_in = nn.Linear(2 * n_freq, d)
        el = nn.TransformerEncoderLayer(d, n_heads, 4 * d, batch_first=True,
                                        norm_first=True, dropout=0.0)
        self.trunk = nn.TransformerEncoder(el, layers)
        self.head = nn.Linear(d, V5)

    def forward(self, tok, mask, ep, n_frames, goal_drop=None):
        """tok [B,32] long (0..15624), mask [B,32] bool (True=hidden),
        ep [B,2,d] start/goal features, n_frames [B] float."""
        B = tok.shape[0]
        t = torch.where(mask, torch.full_like(tok, V5), tok)
        h = self.tok_emb(t) + self.pos_id[None]
        nf = fourier_time((n_frames[:, None] / 120.0).clamp(0, 1))[:, 0]
        cond = torch.stack([self.cond_in(ep[:, 0]),
                            self.cond_in(ep[:, 1]),
                            self.n_in(nf)], 1) + self.cond_id[None]
        if goal_drop is not None:
            cond = cond.clone()
            cond[goal_drop, 1] = 0.0
        h = self.trunk(torch.cat([cond, h], 1))[:, 3:]
        return self.head(h)


def tokens_of(ma, c, masked=False):
    """frozen Stage-A encode -> joint token ids [B,32] (0..15624).

    CONTRACT (audited 0812, runs/CONTRACT_AUDIT_0812.txt). bridge_a.encode()
    applies a time-local attention mask; this function historically did not.
    The two streams agree on 0.15% of tokens and B3 masked-CE differs by 15.2
    nats between them, so they are NOT interchangeable.
      masked=False  the pre-0812 path. B3/B6/B7/B8 and every eval that compares
                    against them MUST use this, or the numbers are nonsense.
      masked=True   bridge_a.encode's own contract, for contract-correct
                    retraining (B9 onwards).
    The default stays False so that existing checkpoints keep reproducing; flip
    it only together with the checkpoints that were trained on it."""
    with torch.no_grad():
        h = ma.embed_frames(c)
        B, T = h.shape[0], h.shape[1]
        seq = torch.cat([ma.q[None].expand(B, -1, -1), h], 1)
        am = None
        if masked and getattr(ma, "time_local", False):
            K, L = ma.n_content, ma.n_content + T
            am = torch.zeros(L, L, dtype=torch.bool, device=h.device)
            b = np.round(np.linspace(0, T, K + 1)).astype(int)
            for i in range(K):
                am[i, K:] = True
                am[i, K + b[i]:K + max(b[i + 1], b[i] + 1)] = False
                am[i, :K] = False
        out = ma.enc(seq, mask=am)[:, :ma.n_content]
        ints = ma.fsq.ints(ma.fsq.features(out))[0]      # [B,32,6] 0..4
        w = (5 ** torch.arange(6, device=ints.device))
        return (ints * w).sum(-1), ma.endpoints(c)


