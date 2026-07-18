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
