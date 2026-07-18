from __future__ import annotations

from types import SimpleNamespace

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


def test_standard_regional_lora_is_image_only_and_skips_broadcast_targets():
    plan, bound = plans()
    route = compile_lora_delta_routes(
        [{"id": "style", "global": False, "region_ids": ["right"]}],
        width=32,
        height=16,
        text_token_count=bound.text_token_count,
        regional_plan=plan,
        bound_plan=bound,
    )[0]

    assert route.text_token_mask == (0.0,) * bound.text_token_count
    assert route.image_token_mask == (0.0, 1.0)
    assert not route_allows_adapter_target(
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

    assert set(installed) == {"diffusion_model.blocks.0.attn.wo.weight"}
    assert reports[0]["applied_model_targets"] == 1
    assert reports[0]["locality_skipped_targets"] == 2
    assert reports[0]["application_mode"] == "unfused_image_token_local_delta_gate"
