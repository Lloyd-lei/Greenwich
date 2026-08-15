"""One-time (research machine): 17 GB code cache -> fixed-capacity atlas.npz.

Windows of 60 frames, stride 30, over every corpus clip; capacity-capped by
uniform sampling ACROSS clips (coverage first, so no single long clip floods
the table). Each window -> Equator A3 32-token sequence via the B3-era token
contract (tokens_of) — the exact tokens the product's Equator.tokenize emits,
so runtime lookups hit the same space.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

AM = Path(__file__).parent.parent
sys.path.insert(0, str(AM / "src"))

RUNS = Path("/var/tmp/alphamotion/topo_exp/runs")
CODES = Path("/var/tmp/alphamotion/topo_exp/bridge_codes/PROD_c1_91k_dim20_tail_0810")
CACHE = Path("/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
K_CAP = 65536
WIN, STRIDE = 60, 30


def main() -> None:
    from alphamotion.atlas.families import family_id
    from alphamotion.engine.equator import Equator
    from alphamotion.engine.nets.bridge import tokens_of

    dev = "cuda"
    eq = Equator.load(RUNS / "BRIDGE_A3_prod_0810", RUNS / "BRIDGE_B3_prod_0810",
                      device=dev)
    cp = np.load(CODES / "codes_pose.npy", mmap_mode="r")
    cr = np.load(CODES / "codes_root_packed.npy", mmap_mode="r")
    seqs = json.load(open(CACHE / "sequences.json"))
    clips = [s["name"] for s in seqs]

    # enumerate candidate windows, then cap by round-robin across clips
    per_clip: list[list[int]] = []
    for s in seqs:
        starts = list(range(s["start"], s["start"] + s["len"] - WIN + 1, STRIDE))
        per_clip.append(starts)
    order: list[tuple[int, int]] = []
    rng = np.random.default_rng(0)
    for lst in per_clip:
        rng.shuffle(lst)
    depth = 0
    while len(order) < K_CAP and any(len(l) > depth for l in per_clip):
        for ci, lst in enumerate(per_clip):
            if depth < len(lst):
                order.append((ci, lst[depth]))
                if len(order) >= K_CAP:
                    break
        depth += 1
    print(f"windows selected: {len(order)} (cap {K_CAP}) from {len(clips)} clips")

    tokens = np.zeros((len(order), 32), np.int32)
    clip_arr = np.zeros(len(order), np.int32)
    start_arr = np.zeros(len(order), np.int32)
    fam_arr = np.zeros(len(order), np.int16)
    B = 96
    for b0 in range(0, len(order), B):
        chunk = order[b0:b0 + B]
        batch = torch.zeros(len(chunk), WIN, 256, 20, dtype=torch.long)
        for i, (ci, st) in enumerate(chunk):
            pose = torch.from_numpy(np.ascontiguousarray(cp[st:st + WIN])).long()
            pk = torch.from_numpy(np.ascontiguousarray(cr[st:st + WIN])).long()
            root = torch.empty_like(pose)
            root[..., 0::2] = pk & 0x0F
            root[..., 1::2] = (pk >> 4) & 0x0F
            batch[i] = torch.cat([pose, root], 1)
        with torch.no_grad():
            tok, _ep = tokens_of(eq.a3, batch.to(dev))
        tokens[b0:b0 + len(chunk)] = tok.cpu().numpy()
        for i, (ci, st) in enumerate(chunk):
            clip_arr[b0 + i] = ci
            start_arr[b0 + i] = st - seqs[ci]["start"]
            fam_arr[b0 + i] = family_id(clips[ci])
        if b0 % (B * 40) == 0:
            print(f"  {b0}/{len(order)}", flush=True)

    from alphamotion.atlas.search import AtlasIndex
    out = AM / "weights_staging" / "atlas"
    out.mkdir(parents=True, exist_ok=True)
    idx = AtlasIndex(tokens, clip_arr, start_arr, fam_arr, clips, WIN, STRIDE)
    idx.save(out / "atlas.npz")
    uniq = len({tuple(t) for t in tokens.tolist()})
    from collections import Counter
    fams = Counter(int(f) for f in fam_arr)
    print(f"-> {out/'atlas.npz'}  windows {len(tokens)}  distinct seqs {uniq}")
    print("family counts:", dict(fams))


if __name__ == "__main__":
    main()
