"""Resident model pool — warm from startup, no cold loads on the request path.

Greenwich + Equator (+ atlas index + human descriptor) live in-process
(~200 MB GPU) behind ONE GPU worker thread: codec/bridge calls are 10-100 ms,
so a single serialized lane beats CUDA context contention. The gateway stays
async; every GPU call goes through run().

Heavy GENMO perception never enters this process — it runs as an on-demand
subprocess lane (perception.genmo) in its separately configured environment.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..config import CONFIG


class ModelPool:
    def __init__(self, device: str | None = None):
        self.device = device or CONFIG.device
        self._gpu = ThreadPoolExecutor(max_workers=1,
                                       thread_name_prefix="am-gpu")
        self.greenwich = None
        self.equator = None
        self.atlas = None
        self.human = None                 # (spec, dof) of the SMPL source body

    # -------------------------------------------------------------- startup --
    def warm(self) -> dict:
        """Blocking load + one dummy round to trigger CUDA kernels."""
        import torch

        from ..atlas.search import load_default
        from ..engine.descriptor import build_from_cache, bundled_cache
        from ..engine.equator import Equator
        from ..engine.greenwich import Greenwich
        self.greenwich = Greenwich.load(device=self.device)
        self.equator = Equator.load(device=self.device)
        self.atlas = load_default()
        spec, dof, rest, _, _ = build_from_cache(bundled_cache(), "human_smpl")
        self.human = (spec, dof, rest)
        # warmup round: encode/decode/tokenize a zero pose
        T = 4
        p9 = torch.zeros(T, spec.J, 9, device=self.device)
        p9[..., 0] = 1.0
        p9[..., 4] = 1.0                   # identity-ish rot6d
        codes = self.greenwich.encode(p9, spec, dof)
        _ = self.greenwich.decode(codes, spec, dof)
        tok, ep = self.equator.tokenize(codes)
        _ = self.equator.token_nll(tok, ep, T)
        return {"device": self.device,
                "atlas_windows": len(self.atlas.tokens),
                "human_joints": spec.J}

    # -------------------------------------------------------------- calling --
    async def run(self, fn, *args, **kwargs):
        """Run a GPU-touching callable on the single GPU thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._gpu,
                                          lambda: fn(*args, **kwargs))


POOL = ModelPool()
