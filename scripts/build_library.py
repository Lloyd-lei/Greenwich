"""One-time (research machine): curated GENERATIVE clip library.

The atlas (65k windows) is an index — tokens only, 3 MB. To PLAY a window on
any robot we additionally need its 4 boundary-frame codes (A3's endpoint
conditioner input): 20 KB/window, too heavy for 65k but fine for a curated set.

This picks 4096 windows, family-balanced round-robin, and stores
    tokens [K,32] int32, bounds [K,4,256,20] int8, clip/start/family
so every library clip is fully regenerable from ~40 MB: endpoints from bounds,
detokenize(tokens, endpoints, any n), decode to any embodiment. The library IS
generative assets, not stored motion.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

AM = Path(__file__).parent.parent
sys.path.insert(0, str(AM / "src"))

CODES = Path("/var/tmp/alphamotion/topo_exp/bridge_codes/PROD_c1_91k_dim20_tail_0810")
K_LIB = 4096


def main() -> None:
    from alphamotion.atlas.families import FAMILIES
    from alphamotion.atlas.search import AtlasIndex
    idx = AtlasIndex.load(AM / "weights_staging" / "atlas" / "atlas.npz")
    cp = np.load(CODES / "codes_pose.npy", mmap_mode="r")
    cr = np.load(CODES / "codes_root_packed.npy", mmap_mode="r")
    seqs = json.load(open(
        "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full/sequences.json"))

    # family-balanced round robin over atlas windows
    byfam: dict[int, list[int]] = defaultdict(list)
    for w in range(len(idx.tokens)):
        byfam[int(idx.family[w])].append(w)
    rng = np.random.default_rng(0)
    for lst in byfam.values():
        rng.shuffle(lst)
    picked: list[int] = []
    depth = 0
    fams = sorted(byfam)
    while len(picked) < K_LIB and any(depth < len(byfam[f]) for f in fams):
        for f in fams:
            if depth < len(byfam[f]) and len(picked) < K_LIB:
                picked.append(byfam[f][depth])
        depth += 1
    print(f"library windows: {len(picked)}")

    bounds = np.zeros((len(picked), 4, 256, 20), np.int8)
    for i, w in enumerate(picked):
        ci, st_rel = int(idx.clip[w]), int(idx.start[w])
        st = seqs[ci]["start"] + st_rel
        rows = [st, st + 1, st + idx.window - 2, st + idx.window - 1]
        pose = torch.from_numpy(np.ascontiguousarray(cp[rows])).long()
        pk = torch.from_numpy(np.ascontiguousarray(cr[rows])).long()
        root = torch.empty_like(pose)
        root[..., 0::2] = pk & 0x0F
        root[..., 1::2] = (pk >> 4) & 0x0F
        bounds[i] = torch.cat([pose, root], 1).numpy().astype(np.int8)

    out = AM / "weights_staging" / "library"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "library.npz",
                        tokens=idx.tokens[picked].astype(np.int32),
                        bounds=bounds,
                        clip=idx.clip[picked], start=idx.start[picked],
                        family=idx.family[picked])
    names = [idx.clips[int(idx.clip[w])] for w in picked]
    (out / "library_meta.json").write_text(json.dumps(
        {"clips": names, "window": idx.window,
         "families": [FAMILIES[int(idx.family[w])] for w in picked]}))
    from collections import Counter
    print("family balance:",
          dict(Counter(FAMILIES[int(idx.family[w])] for w in picked)))
    print(f"-> {out/'library.npz'}")


if __name__ == "__main__":
    main()
