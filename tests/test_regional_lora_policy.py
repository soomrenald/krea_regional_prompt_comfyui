from __future__ import annotations

from types import SimpleNamespace

import pytest

from krea_regional_prompt_comfyui.k2_region_comfy import backend
from krea_regional_prompt_comfyui.k2_region_core.lora import (
    CHARACTER_IDENTITY_LORA_ROUTING,
)
from krea_regional_prompt_comfyui.k2_region_core.regional_lora import (
    compile_lora_delta_routes,
    route_allows_adapter_target,
)
from krea_regional_prompt_comfyui.k2_region_core.regional_prompting import (
    compile_regional_prompt_plan,
)
from krea_regional_prompt_comfyui.k2_region_core.spatial_attention import (
    KreaSpatialAttentionOverride,
)
from krea_regional_prompt_comfyui.k2_region_core.regions import (
    PixelBox,
    RegionDefinition,
)


def plans(*, character_trigger: str | None = None):
    regions = (
        RegionDefinition("left", "Left", PixelBox(0, 0, 16, 16), "red coat"),
        RegionDefinition("right", "Right", PixelBox(16, 0, 32, 16), "blue coat"),
    )
    triggers = {"right": (character_trigger,)} if character_trigger else None
    plan = compile_regional_prompt_plan(
        32,
        16,
        "portrait",
        regions,
        character_identity_triggers=triggers,
    )
    bound = plan.bind_tokens(len, conditioning_text_token_count=len(plan.prompt))
    return plan, bound


def test_standard_regional_lora_gates_text_and_skips_main_broadcast_targets():
    plan, bound = plans()
    route = compile_lora_delta_routes(
        [{"id": "style", "global": False, "region_ids": ["right"]}],
        width=32,
        height=16,
        text_token_count=bound.text_token_count,
        regional_plan=plan,
        bound_plan=bound,
    )[0]

    right = next(span for span in bound.spans if span.region_id == "right")
    assert {index for index, value in enumerate(route.text_token_mask) if value} == set(
        range(right.start, right.end)
    )
    assert route.image_token_mask == (0.0, 1.0)
    assert route_allows_adapter_target(
        route, "diffusion_model.txtfusion.refiner_blocks.0.attn.wq.weight"
    )
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
        route, "diffusion_model.blocks.0.mlp.down.weight"
    )


def test_cross_modal_partition_preserves_image_to_image_attention():
    torch = pytest.importorskip("torch")
    _plan, bound = plans()
    override = KreaSpatialAttentionOverride(bound)
    reference = torch.zeros((1, 1, bound.text_token_count + 2, 1))
    _fields, _emphases, text_owners, image_owners = override._pair_fields(reference)
    scores = torch.zeros(
        (1, 1, bound.text_token_count + 2, bound.text_token_count + 2)
    )

    override._partition_regional_stream(
        scores,
        0,
        bound.text_token_count + 2,
        text_owners,
        image_owners,
    )

    left, _right = bound.spans
    left_image = bound.text_token_count
    right_image = left_image + 1
    assert torch.isneginf(scores[0, 0, left.start, right_image])
    assert torch.isneginf(scores[0, 0, right_image, left.start])
    assert float(scores[0, 0, left_image, right_image]) == 0.0
    assert float(scores[0, 0, right_image, left_image]) == 0.0
    assert torch.isneginf(scores[0, 0, 0, left_image])


def test_character_route_preserves_anchored_text_and_existing_targets():
    plan, bound = plans(character_trigger="lface")
    route = compile_lora_delta_routes(
        [
            {
                "id": "face",
                "global": False,
                "region_ids": ["right"],
                "routing_mode": CHARACTER_IDENTITY_LORA_ROUTING,
                "trigger_phrase": "lface",
            }
        ],
        width=32,
        height=16,
        text_token_count=bound.text_token_count,
        regional_plan=plan,
        bound_plan=bound,
    )[0]

    assert sum(route.text_token_mask) > 0
    assert route.image_token_mask == (0.0, 1.0)
    assert route_allows_adapter_target(
        route, "diffusion_model.blocks.0.attn.wv.weight"
    )


def test_backend_installs_only_spatially_local_standard_targets(monkeypatch):
    plan, bound = plans()
    specification = {
        "id": "style",
        "name": "Style",
        "path": "/unused/style.safetensors",
        "strength": 1.0,
        "global": False,
        "region_ids": ["right"],
        "routing_mode": "standard",
        "trigger_phrase": "",
    }
    patches = {
        "diffusion_model.txtfusion.refiner_blocks.0.attn.wq.weight": object(),
        "diffusion_model.blocks.0.attn.wv.weight": object(),
        "diffusion_model.blocks.0.attn.wo.weight": object(),
    }
    installed = {}

    monkeypatch.setattr(backend, "normalize_lora_specs", lambda config: [specification])
    monkeypatch.setattr(
        backend,
        "_load_lora_patches",
        lambda model, supplied: (
            patches,
            None,
            {
                "id": supplied["id"],
                "display_name": supplied["name"],
                "adapter_count": len(patches),
                "matched_model_targets": len(patches),
                "compatible": True,
            },
        ),
    )

    def fake_install(model, target_entries, statistics):
        del statistics
        installed.update(target_entries)
        return model

    monkeypatch.setattr(backend, "_install_routed_loras", fake_install)
    config = SimpleNamespace(
        width=32,
        height=16,
        regional_plan=plan,
    )

    _model, reports, _statistics = backend.apply_loras(object(), config, bound)

    assert set(installed) == {
        "diffusion_model.txtfusion.refiner_blocks.0.attn.wq.weight",
        "diffusion_model.blocks.0.attn.wo.weight",
    }
    assert reports[0]["applied_model_targets"] == 2
    assert reports[0]["locality_skipped_targets"] == 1
    assert reports[0]["application_mode"] == "unfused_region_text_image_delta_gate_v3"


def test_backend_rejects_regional_lora_when_every_target_would_broadcast(monkeypatch):
    plan, bound = plans()
    specification = {
        "id": "broadcast-only",
        "name": "Broadcast only",
        "path": "/unused/broadcast.safetensors",
        "strength": 1.0,
        "global": False,
        "region_ids": ["right"],
        "routing_mode": "standard",
        "trigger_phrase": "",
    }
    monkeypatch.setattr(backend, "normalize_lora_specs", lambda config: [specification])
    monkeypatch.setattr(
        backend,
        "_load_lora_patches",
        lambda model, supplied: (
            {"diffusion_model.blocks.0.attn.wv.weight": object()},
            None,
            {
                "id": supplied["id"],
                "display_name": supplied["name"],
                "adapter_count": 1,
                "matched_model_targets": 1,
                "compatible": True,
            },
        ),
    )
    config = SimpleNamespace(width=32, height=16, regional_plan=plan)

    with pytest.raises(ValueError, match="no targets that can be routed locally"):
        backend.apply_loras(object(), config, bound)


def test_backend_requires_spatial_router_for_a_regional_lora():
    _plan, bound = plans()
    config = SimpleNamespace(spatial={"enabled": False})

    with pytest.raises(ValueError, match="requires Spatial attention Enabled"):
        backend.attach_spatial_attention(
            object(),
            config,
            bound,
            backend.LoraDeltaStatistics(()),
            [{"status": "applied_regional"}],
            {},
        )
