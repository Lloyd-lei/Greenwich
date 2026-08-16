from pathlib import Path


def test_empty_studio_clears_viewer_before_transport_sync():
    frontend = (
        Path(__file__).parents[2]
        / "src"
        / "alphamotion"
        / "assets"
        / "frontend"
        / "index.html"
    ).read_text()
    boot = frontend.rsplit("(async()=>", 1)[1]

    assert boot.index("await clearTargetViewer()") < boot.index("syncTransport()")
