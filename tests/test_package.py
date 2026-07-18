from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_comfy_entrypoint_and_web_assets_exist():
    assert (ROOT / "__init__.py").is_file()
    assert "comfy_entrypoint" in (ROOT / "__init__.py").read_text()
    assert (ROOT / "web" / "k2_region_studio.js").is_file()
    assert (ROOT / "web" / "k2_region_studio.css").is_file()


def test_extension_does_not_pin_or_install_torch():
    requirements = (ROOT / "requirements.txt").read_text().lower()
    assert "torch" not in requirements
    assert "cuda" not in requirements
    assert "rocm" not in requirements


def test_repository_metadata_points_to_comfyui_only_repository():
    metadata = (ROOT / "pyproject.toml").read_text()
    assert "soomrenald/krea_regional_prompt_comfyui" in metadata
    assert "soomrenald/krea_region_project\"" not in metadata


def test_studio_and_bare_node_ids_are_registered():
    node_sources = "\n".join(
        [
            (ROOT / "k2_region_comfy" / "nodes.py").read_text(),
            (ROOT / "k2_region_comfy" / "bare_nodes.py").read_text(),
        ]
    )
    expected = {
        "K2KreaLoader",
        "K2RegionStudio",
        "K2RegionalSampler",
        "K2FaceDetail",
        "K2PostUpscale",
        "K2BBoxToRegionalMask",
        "K2RegionalCharacterLoRA",
        "K2RegionalLoRAStack3",
        "K2RegionalLayerLoRAApply",
        "K2RegionalAttentionLoRASampler",
        "K2RegionalDecodeComposite",
    }
    assert all(f'node_id="{node_id}"' in node_sources for node_id in expected)


def test_sidebar_preserves_the_active_editor_pane_across_renders():
    source = (ROOT / "web" / "k2_region_studio.js").read_text()
    assert 'this.activePane = "Regions"' in source
    assert "this.activePane = name" in source
    assert "(activeButton || tabs.firstChild)?.click()" in source
