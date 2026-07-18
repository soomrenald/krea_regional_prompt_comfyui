from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_every_v3_node_input_has_a_hover_tooltip():
    missing = []
    for relative in ("k2_region_comfy/nodes.py", "k2_region_comfy/bare_nodes.py"):
        source = (ROOT / relative).read_text()
        tree = ast.parse(source, filename=relative)
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if not isinstance(call.func, ast.Attribute) or call.func.attr != "Input":
                continue
            if not any(keyword.arg == "tooltip" for keyword in call.keywords):
                missing.append(f"{relative}:{call.lineno}")
    assert missing == []


def test_sidebar_controls_have_hover_help_and_complete_sections():
    source = (ROOT / "web" / "k2_region_studio.js").read_text()
    required_help = (
        "global_prompt",
        "region_enabled",
        "priority",
        "lora_routing",
        "emphasis_occurrence",
        '"spatial.lora_delta_adaptation_gain"',
        '"projector.identity_protection"',
        '"face_detail.detector_threshold"',
        "json_apply",
    )
    assert all(key in source for key in required_help)
    assert "title: HELP" in source


def test_readme_contains_control_reference_for_every_node_style():
    readme = (ROOT / "README.md").read_text()
    expected_sections = (
        "## Complete control reference",
        "### Sidebar: Regions",
        "### Sidebar: LoRAs",
        "### Sidebar: Emphasis",
        "### Sidebar: Spatial attention tuning",
        "### Sidebar: Projector control",
        "### Sidebar: Face detail tuning",
        "### K2 Regional Sampler",
        "### K2 Regional Face Detail",
        "### K2 Post Upscale",
        "### Bare: K2 BBox To Regional Mask",
        "### Bare: K2 Regional Character LoRA",
        "### Bare: K2 Regional LoRA Stack 3",
        "### Bare: K2 Regional Layer LoRA Apply",
        "### Bare: K2 Regional Attention LoRA Sampler",
        "### Bare: K2 Regional Decode Composite",
    )
    assert all(section in readme for section in expected_sections)
