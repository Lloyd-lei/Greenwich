import numpy as np

from alphamotion.atlas.library import Library
from alphamotion.atlas.search import AtlasIndex


def make():
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 100, (50, 32))
    toks[10] = toks[5]                  # planted twin
    return AtlasIndex(toks, np.arange(50), np.zeros(50, int),
                      np.zeros(50, int), [f"c{i}" for i in range(50)],
                      capacity=52)


def test_portals_find_twin():
    idx = make()
    ps = idx.portals(idx.tokens[5], 7, k=5, exclude_clip=5)
    assert any(p["window"] == 10 for p in ps)


def test_ring_eviction():
    idx = make()
    idx._hists()                    # materialize cache before replacement
    for i in range(5):
        idx.add(np.full(32, 3), f"gen{i}", 0)
    assert len(idx.tokens) <= 52
    assert idx.frozen == 50             # corpus prefix never evicted
    dynamic = idx._ring[-1]
    assert idx.start[dynamic] == 0
    assert idx._hists()[dynamic, 3] == 1.0


def test_save_load_roundtrip(tmp_path):
    idx = make()
    idx.add(np.full(32, 7), "generated", 0)
    idx.save(tmp_path / "a.npz")
    idx2 = AtlasIndex.load(tmp_path / "a.npz")
    assert (idx2.tokens == idx.tokens).all()
    assert idx2.frozen == 50 and idx2._ring == [50]
    for i in range(5):
        idx2.add(np.full(32, i), f"next{i}", 0)
    assert len(idx2.tokens) <= idx2.capacity == 52


def test_library_resolves_truncated_and_generated_portals():
    lib = Library.__new__(Library)
    lib.names = ["SOMA__subject__walk_003_stageii", "other"]
    lib.tokens = np.stack([np.arange(32), np.arange(32) + 100])

    assert lib.resolve_portal("SOMA__subject__walk", np.zeros(32)) == 0
    assert lib.resolve_portal("user-facing edit", np.arange(32)) == 0
    assert lib.resolve_portal("unrelated", np.full(32, -1)) is None
