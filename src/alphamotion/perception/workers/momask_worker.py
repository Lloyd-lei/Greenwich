"""Small process boundary around the official MoMask text-to-motion model.

This file intentionally has no AlphaMotion imports: it runs in the external
perception environment with the MoMask checkout inserted into ``sys.path``.
Unlike MoMask's demo, it saves the native HumanML3D 263-D representation and
does not spend time fitting BVH joints or rendering a Matplotlib movie.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--frames", required=True, type=int,
                   help="Requested MoMask frames at its native 20 FPS")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=10107)
    p.add_argument("--steps", type=int, default=18)
    return p


def main() -> None:
    args = _parser().parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))

    from models.mask_transformer.transformer import (MaskTransformer,
                                                       ResidualTransformer)
    from models.vq.model import RVQVAE
    from utils.fixseed import fixseed
    from utils.get_opt import get_opt

    # These are the official released HumanML3D model names.
    ckpt = repo / "checkpoints" / "t2m"
    trans_name = "t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns"
    res_name = "tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fixseed(args.seed)
    started = time.perf_counter()

    trans_opt = get_opt(str(ckpt / trans_name / "opt.txt"), device=device)
    vq_opt = get_opt(str(ckpt / trans_opt.vq_name / "opt.txt"),
                     device=device)
    res_opt = get_opt(str(ckpt / res_name / "opt.txt"), device=device)
    vq_opt.dim_pose = 263

    vq = RVQVAE(vq_opt, vq_opt.dim_pose, vq_opt.nb_code,
                vq_opt.code_dim, vq_opt.output_emb_width, vq_opt.down_t,
                vq_opt.stride_t, vq_opt.width, vq_opt.depth,
                vq_opt.dilation_growth_rate, vq_opt.vq_act, vq_opt.vq_norm)
    vq_state = torch.load(
        ckpt / trans_opt.vq_name / "model" / "net_best_fid.tar",
        map_location="cpu")
    vq.load_state_dict(vq_state.get("vq_model", vq_state.get("net")))

    trans_opt.num_tokens = vq_opt.nb_code
    trans_opt.num_quantizers = vq_opt.num_quantizers
    trans_opt.code_dim = vq_opt.code_dim
    transformer = MaskTransformer(
        code_dim=trans_opt.code_dim, cond_mode="text",
        latent_dim=trans_opt.latent_dim, ff_size=trans_opt.ff_size,
        num_layers=trans_opt.n_layers, num_heads=trans_opt.n_heads,
        dropout=trans_opt.dropout, clip_dim=512,
        cond_drop_prob=trans_opt.cond_drop_prob, clip_version="ViT-B/32",
        opt=trans_opt)
    trans_state = torch.load(
        ckpt / trans_name / "model" / "latest.tar", map_location="cpu")
    transformer.load_state_dict(
        trans_state.get("t2m_transformer", trans_state.get("trans")),
        strict=False)

    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    residual = ResidualTransformer(
        code_dim=vq_opt.code_dim, cond_mode="text",
        latent_dim=res_opt.latent_dim, ff_size=res_opt.ff_size,
        num_layers=res_opt.n_layers, num_heads=res_opt.n_heads,
        dropout=res_opt.dropout, clip_dim=512,
        shared_codebook=vq_opt.shared_codebook,
        cond_drop_prob=res_opt.cond_drop_prob,
        share_weight=res_opt.share_weight, clip_version="ViT-B/32",
        opt=res_opt)
    res_state = torch.load(
        ckpt / res_name / "model" / "net_best_fid.tar",
        map_location="cpu")
    residual.load_state_dict(res_state["res_transformer"], strict=False)

    models = (vq, transformer, residual)
    for model in models:
        model.eval().to(device)

    # MoMask's temporal tokenizer requires multiples of four and the released
    # HumanML3D model supports at most 196 native frames (9.8 seconds).
    native_frames = max(4, min(196, int(round(args.frames / 4.0)) * 4))
    token_lens = torch.tensor([native_frames // 4], device=device).long()
    with torch.inference_mode():
        mids = transformer.generate(
            [args.text], token_lens, timesteps=args.steps, cond_scale=4,
            temperature=1.0, topk_filter_thres=0.9, gsample=False)
        mids = residual.generate(mids, [args.text], token_lens,
                                 temperature=1.0, cond_scale=5)
        normalized = vq.forward_decoder(mids).detach().cpu().numpy()[0]

    mean = np.load(ckpt / trans_opt.vq_name / "meta" / "mean.npy")
    std = np.load(ckpt / trans_opt.vq_name / "meta" / "std.npy")
    humanml = (normalized[:native_frames] * std + mean).astype(np.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, humanml=humanml, fps=np.float32(20.0))
    peak = (torch.cuda.max_memory_allocated() / 1048576.0
            if device.type == "cuda" else 0.0)
    print(json.dumps({"ok": True, "frames": len(humanml),
                      "seconds": time.perf_counter() - started,
                      "peak_vram_mb": peak, "device": str(device)}))


if __name__ == "__main__":
    main()
