"""One-time (research machine): curated GENERATIVE clip library.

The atlas (65k windows) is an index — tokens only, a few MB. To PLAY a window
without inventing its rotation stream we keep lossless nibble-packed raw codes
for a curated set, plus boundary codes for temporal editing.

This picks 4096 windows, family-balanced round-robin, and stores
    tokens [K,32] int32, bounds [K,4,256,20] int8, clip/start/family
The compressed metadata is small; the exact dual-stream playback store is
roughly 600 MB for 4,096 x 60-frame windows. Equator retimes the pose stream;
the raw rotation stream is carried/interpolated explicitly.
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
    # RAW dual-stream codes, packed 2 dims/byte: playback must be the corpus
    # codes themselves — A3 cannot regenerate the rotation stream (0814 audit:
    # argmax on slots 128:256 decodes ~24 m off; raw stream decodes exact).
    packed = np.zeros((len(picked), idx.window, 256, 10), np.uint8)
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
        win = np.arange(st, st + idx.window)
        pw = np.ascontiguousarray(cp[win]).astype(np.uint8)      # [w,128,20]
        packed[i, :, :128] = pw[..., 0::2] | (pw[..., 1::2] << 4)
        packed[i, :, 128:] = np.ascontiguousarray(cr[win])       # already packed

    # root trajectory passthrough (owner design: first frame = world origin,
    # deltas carried alongside the codes; the codec itself is root-relative).
    # Stored for every embodiment that has GMR ground truth + human.
    RW_BODIES = ["human_smpl", "unitree_g1_29dof", "unitree_h1",
                 "booster_t1", "fourier_gr3", "tienkung"]
    rws = {}
    for b in RW_BODIES:
        p = Path(f"/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full/"
                 f"{b}_root_world.pt")
        if p.exists():
            import torch as _t
            rws[b] = _t.load(p, mmap=True)
    root_delta = np.zeros((len(picked), len(RW_BODIES), idx.window, 3),
                          np.float16)
    for i, w in enumerate(picked):
        ci, st_rel = int(idx.clip[w]), int(idx.start[w])
        st = seqs[ci]["start"] + st_rel
        for bi, b in enumerate(RW_BODIES):
            if b in rws:
                p = rws[b][st:st + idx.window].numpy().astype(np.float64)
                root_delta[i, bi] = (p - p[0]).astype(np.float16)  # cm, Y-up

    out = AM / "weights_staging" / "library"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "library_codes.npy", packed)   # .npy: mmap-able at load
    np.save(out / "library_root.npy", root_delta)
    (out / "library_root_meta.json").write_text(json.dumps(
        {"bodies": RW_BODIES, "units": "cm", "basis": "Y-up",
         "anchor": "first frame of window = origin"}))
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
