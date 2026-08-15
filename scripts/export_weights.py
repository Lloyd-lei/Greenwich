"""One-time (research machine only): research checkpoints -> HF-ready staging.

Produces weights_staging/ mirroring the HF repo layout:
    greenwich/{model.safetensors, config.json}
    equator_a/... equator_b/...
    embodiments/name_embeddings.pt      (Qwen3 embeds for every bundled body)

Run inside the research conda env. The staged tree is uploaded by
upload_hf.py and downloaded at user install time by alphamotion.weights.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

RUNS = Path("/var/tmp/alphamotion/topo_exp/runs")
HERE = Path(__file__).parent
STAGE = HERE.parent / "weights_staging"
PKG = HERE.parent / "src" / "alphamotion"

EXPORTS = {
    "greenwich": RUNS / "PT_c1dec_reenc_0812",
    "equator_a": RUNS / "BRIDGE_A3_prod_0810",
    "equator_b": RUNS / "BRIDGE_B3_prod_0810",
}


def export_models() -> None:
    for name, run in EXPORTS.items():
        out = STAGE / name
        out.mkdir(parents=True, exist_ok=True)
        sd = torch.load(run / "model.pt", map_location="cpu")
        assert all(torch.is_tensor(v) for v in sd.values()), f"{name}: not a pure state_dict"
        sd = {k: v.contiguous() for k, v in sd.items()}
        save_file(sd, str(out / "model.safetensors"))
        cfg = json.load(open(run / "config.json"))
        # scrub any research paths out of the shipped config
        cfg = {k: v for k, v in cfg.items()
               if not (isinstance(v, str) and ("/var/tmp" in v or "be water" in v))}
        json.dump(cfg, open(out / "config.json", "w"), indent=1)
        n = sum(v.numel() for v in sd.values())
        print(f"{name}: {n/1e6:.1f}M params -> {out/'model.safetensors'}")


def export_name_embeddings() -> None:
    """Qwen3 embeddings for every joint name of every bundled embodiment."""
    sys.path.insert(0, str(PKG.parent))
    import numpy as np

    from alphamotion.embodiment.semantics_map import (_encode_qwen3,
                                                      joint_prompt)
    root = PKG / "assets" / "embodiments"
    names: list[str] = []
    for f in sorted(root.glob("*_spec.npz")):
        d = np.load(f, allow_pickle=True)
        names += [str(x) for x in d["joint_names"]]
    uniq = list(dict.fromkeys(names))
    print(f"bundled joint names: {len(uniq)} unique")
    feats = _encode_qwen3([joint_prompt(n) for n in uniq], "cuda")
    feats = torch.nn.functional.normalize(feats, dim=-1)
    lut = {n: feats[i].clone() for i, n in enumerate(uniq)}
    out = STAGE / "embodiments"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(lut, out / "name_embeddings.pt")
    # also bundle into the package so the core install works offline
    torch.save(lut, root / "name_embeddings.pt")
    print(f"-> {out/'name_embeddings.pt'} ({len(lut)} names)")


if __name__ == "__main__":
    export_models()
    export_name_embeddings()
    print("STAGED", STAGE)
