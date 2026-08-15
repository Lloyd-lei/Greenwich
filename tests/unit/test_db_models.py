import os


def test_crud(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHAMOTION_DATA", str(tmp_path))
    import importlib

    import alphamotion.paths
    importlib.reload(alphamotion.paths)
    import alphamotion.service.db as db
    importlib.reload(db)
    with db.session() as s:
        m = db.Motion(title="x", family="walk", duration_s=2.0, fps=30,
                      n_frames=60, source="edit", tokens=[1, 2, 3],
                      trace_path="t.npz")
        s.add(m)
        s.commit()
        s.add(db.Job(id="j1", kind="play", request={"a": 1}))
        s.commit()
        assert s.get(db.Job, "j1").status == "queued"
        assert s.query(db.Motion).filter_by(family="walk").count() == 1
