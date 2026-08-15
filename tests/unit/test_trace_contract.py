import numpy as np

from alphamotion.engine.trace import MotionTrace


def test_roundtrip(tmp_path):
    T, J = 10, 5
    tr = MotionTrace(q=np.zeros((T, J, 3)), rootR=np.tile(np.eye(3), (T, 1, 1)),
                     gp=np.zeros((T, J, 3)), stage=np.ones(T, np.int32),
                     fps=30.0, title="t", target="b",
                     tokens=np.arange(32, dtype=np.int32))
    p = tr.save(tmp_path / "x.npz")
    tr2 = MotionTrace.load(p)
    assert tr2.frames == T and tr2.title == "t"
    assert (tr2.tokens == tr.tokens).all()
