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


def test_genmo_workflows_are_separate_between_data_and_motion_studios():
    frontend = (
        Path(__file__).parents[2]
        / "src/alphamotion/assets/frontend/index.html"
    ).read_text()

    assert 'id="generationReview"' in frontend
    assert 'id="genImportMedia"' in frontend
    assert 'id="genShareLibrary"' in frontend
    assert 'id="addAiSegment"' in frontend
    assert "openMotionGenerator('studio')" in frontend
    assert "Place placeholder on timeline" in frontend
    assert "/smpl-generations/${encodeURIComponent(item.generation_id)}/commit" in frontend


def test_data_studio_assets_preview_on_click_until_select_mode():
    frontend = (
        Path(__file__).parents[2]
        / "src/alphamotion/assets/frontend/index.html"
    ).read_text()

    assert "if(!dsSharedSelectMode){openDataStudioPreview(item);return}" in frontend
    assert "if(!dsLocalSelectMode||kind!=='motion'){kind==='motion'?openProjectMotionPreview(item):openRobotAsset(item.name);return}" in frontend
    assert "if(!dsRobotSelectMode){openRobotAsset(item.name);return}" in frontend
    assert "card.ondblclick=()=>{openMotionGenerator('dataStudio')" in frontend


def test_data_studio_library_uses_explicit_selection_mode():
    frontend = (
        Path(__file__).parents[2]
        / "src/alphamotion/assets/frontend/index.html"
    ).read_text()

    assert 'id="dsSharedSelect"' in frontend
    assert 'id="dsSelectAllShared"' not in frontend
    assert 'id="dsSelectAllRobots"' not in frontend
    assert "$('#dsImport').hidden=!dsSharedSelectMode" in frontend
    assert "dsSharedSelectMode=false" in frontend
    assert "dsRobotSelectMode=false" in frontend
    assert "<button class=\"asset-check\" title=\"Select for import\" ${dsSharedSelectMode?'':'hidden'}>" in frontend


def test_imported_project_motions_keep_library_preview_and_tags():
    frontend = (
        Path(__file__).parents[2]
        / "src/alphamotion/assets/frontend/index.html"
    ).read_text()

    assert "observeAssetPreview(img,item.library_id)" in frontend
    assert "${kind==='motion'?localTags(item):''}" in frontend
    assert "item.data_role||(['starter','upload','data_studio_processed'].includes(item.origin)?'original':'')" in frontend


def test_motion_studio_labeling_stays_in_source_preview():
    frontend = (
        Path(__file__).parents[2]
        / "src/alphamotion/assets/frontend/index.html"
    ).read_text()

    assert "hideAssetDetail();openSourcePreview(it,true)" in frontend
    assert "hideAssetDetail();openDataStudioPreview(it,true)" not in frontend
    assert "async function openSourcePreview(item,showLabels=false)" in frontend
    assert "activateView('studio');sourceOpen=true" in frontend


def test_project_motion_preview_uses_bodydata_results_workspace():
    frontend = (
        Path(__file__).parents[2]
        / "src/alphamotion/assets/frontend/index.html"
    ).read_text()

    assert "function openProjectMotionPreview(item)" in frontend
    assert "type:'alphamotion:open-project-preview'" in frontend
    assert "show_contacts:(item.labels||[]).includes('foot_contact')" in frontend


def test_project_media_delete_actions_are_exposed_in_both_studios():
    frontend = (
        Path(__file__).parents[2]
        / "src/alphamotion/assets/frontend/index.html"
    ).read_text()

    assert 'id="dsDeleteLocal"' in frontend
    assert 'id="deleteProjectMedia"' in frontend
    assert "function deleteProjectMotionItems(items)" in frontend
    assert "Shared Library source assets will not be deleted." in frontend
    assert "method:'DELETE'" in frontend
