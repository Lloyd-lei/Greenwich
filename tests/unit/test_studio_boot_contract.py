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


def test_endpoint_controls_belong_to_the_selected_clip():
    frontend = (
        Path(__file__).parents[2]
        / "src"
        / "alphamotion"
        / "assets"
        / "frontend"
        / "index.html"
    ).read_text()

    endpoint_renderer = frontend.split("function renderEndpointMarkers()", 1)[1].split(
        "function layoutTimeline()", 1
    )[0]
    assert "block.append(button)" in endpoint_renderer
    assert "C.append(button)" not in endpoint_renderer
    assert "endpoint-editing" in endpoint_renderer


def test_motion_workspace_exposes_persistent_panel_dividers():
    frontend = (
        Path(__file__).parents[2]
        / "src"
        / "alphamotion"
        / "assets"
        / "frontend"
        / "index.html"
    ).read_text()

    for divider in (
        "leftPanelDivider",
        "rowPanelDivider",
        "inspectorPanelDivider",
    ):
        assert f'id="{divider}"' in frontend
    assert "setupPanelResizers()" in frontend
    assert "alphamotion-panel-layout" in frontend
