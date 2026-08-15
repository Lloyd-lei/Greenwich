import numpy as np

from alphamotion.engine.trace import MotionTrace


def test_roundtrip(tmp_path):
    T, J = 10, 5
    tr = MotionTrace(q=np.zeros((T, J, 3)), rootR=np.tile(np.eye(3), (T, 1, 1)),
                     gp=np.zeros((T, J, 3)), stage=np.ones(T, np.int32),
                     fps=30.0, title="t", target="b",
                     tokens=np.arange(32, dtype=np.int32),
                     root_t=np.arange(T * 3, dtype=np.float32).reshape(T, 3))
    p = tr.save(tmp_path / "x.npz")
    tr2 = MotionTrace.load(p)
    assert tr2.frames == T and tr2.title == "t"
    assert (tr2.tokens == tr.tokens).all()
    assert (tr2.root_t == tr.root_t).all()


def test_rejects_bad_root_shape():
    T, J = 2, 3
    with np.testing.assert_raises(ValueError):
        MotionTrace(q=np.zeros((T, J, 3)),
                    rootR=np.tile(np.eye(3), (T, 1, 1)),
                    gp=np.zeros((T, J, 3)), stage=np.ones(T),
                    root_t=np.zeros((T, 2)))


def test_rejects_invalid_trace_contracts():
    T, J = 2, 3
    root = np.tile(np.eye(3), (T, 1, 1))
    with np.testing.assert_raises(ValueError):
        MotionTrace(q=np.zeros((T, J)), rootR=root,
                    gp=np.zeros((T, J, 3)), stage=np.ones(T))
    with np.testing.assert_raises(ValueError):
        MotionTrace(q=np.zeros((T, J, 3)), rootR=root,
                    gp=np.zeros((T, J, 3)), stage=np.array([0, 7]))
    with np.testing.assert_raises(ValueError):
        MotionTrace(q=np.zeros((T, J, 3)), rootR=root,
                    gp=np.zeros((T, J, 3)), stage=np.ones(T), fps=0)
