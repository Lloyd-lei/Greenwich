import numpy as np

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
    for i in range(5):
        idx.add(np.full(32, 3), f"gen{i}", 0)
    assert len(idx.tokens) <= 52
    assert idx.frozen == 50             # corpus prefix never evicted


def test_save_load_roundtrip(tmp_path):
    idx = make()
    idx.save(tmp_path / "a.npz")
    idx2 = AtlasIndex.load(tmp_path / "a.npz")
    assert (idx2.tokens == idx.tokens).all()
