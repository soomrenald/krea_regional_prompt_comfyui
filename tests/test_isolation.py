import pytest

from krea_regional_prompt_comfyui.k2_region_core.regional_lora import (
    compile_lora_delta_routes,
    route_allows_adapter_target,
)
from krea_regional_prompt_comfyui.k2_region_core.regional_prompting import (
    compile_regional_prompt_plan,
)
from krea_regional_prompt_comfyui.k2_region_core.regions import PixelBox, RegionDefinition
from krea_regional_prompt_comfyui.k2_region_core.spatial_attention import (
    KreaSpatialAttentionOverride,
    image_region_ownership,
    text_region_ownership,
)


def _two_subject_plans():
    regions = (
        RegionDefinition(
            "left",
            "Left subject",
            PixelBox(0, 0, 16, 16),
            "a naked woman",
            priority=2,
            spatial_role="subject",
        ),
        RegionDefinition(
            "right",
            "Right subject",
            PixelBox(16, 0, 32, 16),
            "a clothed woman",
            priority=1,
            spatial_role="subject",
        ),
    )
    plan = compile_regional_prompt_plan(32, 16, "portrait", regions)
    bound = plan.bind_tokens(len, conditioning_text_token_count=len(plan.prompt))
    return plan, bound


def test_standard_route_excludes_broadcast_key_value_targets():
    plan, bound = _two_subject_plans()
    route = compile_lora_delta_routes(
        [
            {
                "id": "style",
                "name": "Style",
                "global": False,
                "region_ids": ["left"],
            }
        ],
        width=32,
        height=16,
        text_token_count=bound.text_token_count,
        regional_plan=plan,
        bound_plan=bound,
    )[0]

    assert not route_allows_adapter_target(
        route, "diffusion_model.blocks.0.attn.wk.weight"
    )
    assert not route_allows_adapter_target(
        route, "diffusion_model.blocks.0.attn.wv.weight"
    )
    assert route_allows_adapter_target(
        route, "diffusion_model.blocks.0.attn.wq.weight"
    )
    assert route_allows_adapter_target(
        route, "diffusion_model.blocks.0.attn.wo.weight"
    )
    assert route_allows_adapter_target(
        route, "diffusion_model.txtfusion.refiner_blocks.0.attn.wk.weight"
    )


def test_subject_prompt_attention_is_hard_partitioned():
    torch = pytest.importorskip("torch")
    _plan, bound = _two_subject_plans()
    override = KreaSpatialAttentionOverride(bound, strict_isolation=True)
    text_owners = torch.tensor(text_region_ownership(bound), dtype=torch.int16)
    image_owners = torch.tensor(image_region_ownership(bound), dtype=torch.int16)
    total = bound.text_token_count + bound.image_token_count
    scores = torch.zeros((1, 1, total, total), dtype=torch.float32)

    override._partition_regional_stream(scores, 0, total, text_owners, image_owners)

    left, right = bound.spans
    left_image = bound.text_token_count
    right_image = bound.text_token_count + 1
    assert torch.isneginf(scores[0, 0, right_image, left.start])
    assert scores[0, 0, right_image, right.start] == 0
    assert torch.isneginf(scores[0, 0, left_image, right.start])
    assert scores[0, 0, left_image, left.start] == 0
    assert torch.all(scores[0, 0, left_image:, left_image:] == 0)


def test_text_refiner_cannot_mix_subject_owned_clauses():
    torch = pytest.importorskip("torch")
    _plan, bound = _two_subject_plans()
    override = KreaSpatialAttentionOverride(bound, strict_isolation=True)
    owners = torch.tensor(text_region_ownership(bound), dtype=torch.int16)
    scores = torch.zeros(
        (1, 1, bound.text_token_count, bound.text_token_count),
        dtype=torch.float32,
    )

    override._partition_regional_text(scores, 0, bound.text_token_count, owners)

    left, right = bound.spans
    assert torch.isneginf(scores[0, 0, left.start, right.start])
    assert torch.isneginf(scores[0, 0, right.start, left.start])
    assert scores[0, 0, left.start, left.start] == 0
