"""Joint-name semantic embeddings — the cross-topology alignment signal.

The released Greenwich was trained with sem_encoder = "qwen3"
(Qwen3-Embedding-0.6B, 1024-d, last-token pooling). The encoder choice is baked
into the checkpoint: descriptors built with any other text tower live in a
different space and the codec's input projection will reject them, so this
module ships qwen3 only.

The ~20 bundled embodiments come with a precomputed embedding cache
(assets/embodiments/name_embeddings.pt) so the core install never needs the
1.2 GB Qwen model. It is downloaded from the official HF repo only when a user
ingests a NEW URDF whose joint names are not in the cache.
"""
from __future__ import annotations

import re

import torch

from ..paths import assets_root, cache_dir

QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
SEM_DIM = 1024
_BUNDLED = assets_root() / "embodiments" / "name_embeddings.pt"


def joint_prompt(n: str) -> str:
    """The one prompt template — must match training exactly."""
    return f"the {normalize_joint_name(n)} joint of a body"


def normalize_joint_name(n: str) -> str:
    n = re.sub(r"_skel$|_joint$|_dof$", "", n)
    n = re.sub(r"([a-z])([A-Z])", r"\1 \2", n)
    n = n.replace("_", " ").strip().lower()
    return n or "root"


@torch.no_grad()
def _encode_qwen3(prompts: list[str], device: str, batch: int = 16) -> torch.Tensor:
    """Qwen3-Embedding-0.6B: causal-LM encoder, official recipe pools the LAST
    token — left padding, read position -1. Identical to training."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN_MODEL, padding_side="left")
    model = AutoModel.from_pretrained(QWEN_MODEL, dtype=torch.float16) \
        .to(device).eval()
    out = []
    for i in range(0, len(prompts), batch):
        inp = tok(prompts[i:i + batch], padding=True, truncation=True,
                  max_length=64, return_tensors="pt").to(device)
        assert int(inp["attention_mask"][:, -1].min()) == 1, \
            "left padding broken: last token is a pad"
        h = model(**inp).last_hidden_state
        out.append(h[:, -1].float().cpu())
    feats = torch.cat(out, 0)
    assert feats.shape == (len(prompts), SEM_DIM)
    return feats


@torch.no_grad()
def build_name_embeddings(names: list[str], device: str = "cuda") -> dict:
    """{joint_name: [1024] unit tensor}. Bundled cache first, user cache second,
    live Qwen encode (requires the `labeling` extra) only for unseen names."""
    uniq = list(dict.fromkeys(names))
    lut: dict = {}
    for src in (_BUNDLED, cache_dir() / "name_embeddings_user.pt"):
        if src.exists():
            lut.update(torch.load(src, map_location="cpu", weights_only=False))
    missing = [n for n in uniq if n not in lut]
    if missing:
        try:
            feats = _encode_qwen3([joint_prompt(n) for n in missing], device)
        except ImportError as exc:  # transformers not installed
            raise RuntimeError(
                f"{len(missing)} joint names need fresh Qwen3 embeddings "
                f"(e.g. {missing[:3]}). Install the labeling extra: "
                f"pip install alphamotion[labeling]") from exc
        feats = torch.nn.functional.normalize(feats, dim=-1)
        for i, n in enumerate(missing):
            lut[n] = feats[i]
        user = cache_dir() / "name_embeddings_user.pt"
        prev = torch.load(user, map_location="cpu", weights_only=False) \
            if user.exists() else {}
        prev.update({n: lut[n] for n in missing})
        torch.save(prev, user)
    return lut


def embed_matrix(names: list[str], lookup: dict) -> torch.Tensor:
    return torch.stack([lookup[n] for n in names], 0)
