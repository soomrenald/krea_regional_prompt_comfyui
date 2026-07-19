from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows" / "k2_bare_kj_ideogram_starter.json"


def _load_workflow() -> dict:
    return json.loads(WORKFLOW.read_text(encoding="utf-8"))


def test_bare_kj_starter_uses_single_pass_layer_injection() -> None:
    workflow = _load_workflow()
    nodes = {node["id"]: node for node in workflow["nodes"]}
    node_types = [node["type"] for node in nodes.values()]

    assert "Ideogram4PromptBuilderKJ" in node_types
    assert node_types.count("K2BBoxToRegionalMask") == 2
    assert node_types.count("K2RegionalCharacterLoRA") == 2
    assert "K2RegionalLoRAStack3" in node_types
    assert "K2RegionalLayerLoRAApply" in node_types
    assert node_types.count("KSampler") == 1
    assert "K2RegionalAttentionLoRASampler" not in node_types
    assert "K2RegionalDecodeComposite" not in node_types

    layer_apply = next(node for node in nodes.values() if node["type"] == "K2RegionalLayerLoRAApply")
    assert layer_apply["widgets_values"][:3] == ["attn_out_mlp", 0.0, 0.0]

    sampler = next(node for node in nodes.values() if node["type"] == "KSampler")
    assert sampler["widgets_values"][2:7] == [8, 1.0, "euler", "simple", 1.0]


def test_bare_kj_starter_seeds_two_labeled_boxes_and_indexes() -> None:
    workflow = _load_workflow()
    nodes = workflow["nodes"]
    builder = next(node for node in nodes if node["type"] == "Ideogram4PromptBuilderKJ")
    bbox_nodes = [node for node in nodes if node["type"] == "K2BBoxToRegionalMask"]

    assert len(builder["ideo"]["boxes"]) == 2
    assert all(box["desc"].strip() for box in builder["ideo"]["boxes"])
    assert sorted(node["widgets_values"][3] for node in bbox_nodes) == [0, 1]

    links = {link[0]: link for link in workflow["links"]}
    bbox_output_links = builder["outputs"][2]["links"]
    assert len(bbox_output_links) == 2
    assert {links[link_id][4] for link_id in bbox_output_links} == {0}
    assert {links[link_id][5] for link_id in bbox_output_links} == {"BOUNDING_BOX"}


def test_bare_kj_starter_links_are_internally_consistent() -> None:
    workflow = _load_workflow()
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = workflow["links"]
    link_ids = [link[0] for link in links]
    assert len(link_ids) == len(set(link_ids))

    incoming: dict[tuple[int, int], int] = {}
    for link_id, source_id, source_slot, target_id, target_slot, link_type in links:
        assert source_id in nodes
        assert target_id in nodes
        source = nodes[source_id]["outputs"][source_slot]
        target = nodes[target_id]["inputs"][target_slot]
        assert source["type"] == link_type
        assert target["type"] == link_type
        assert link_id in (source.get("links") or [])
        assert target.get("link") == link_id
        key = (target_id, target_slot)
        assert key not in incoming
        incoming[key] = link_id
