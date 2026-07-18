from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from krea_regional_prompt_comfyui.k2_region_bare.engine import (  # noqa: E402
    run_regional_velocity_sampler,
)
from krea_regional_prompt_comfyui.k2_region_bare.lora import (  # noqa: E402
    filter_lora_state_dict,
    is_krea_attention_out_lora_key,
    is_krea_mlp_lora_key,
    is_krea_writeback_lora_key,
)
from krea_regional_prompt_comfyui.k2_region_bare.masks import region_from_bbox  # noqa: E402
from krea_regional_prompt_comfyui.k2_region_bare.types import (  # noqa: E402
    K2RegionalLora,
    K2RegionalLoraStack,
)


def make_region(bbox, *, batch_size=1, bbox_format="xywh"):
    return region_from_bbox(
        [bbox],
        width=64,
        height=64,
        bbox_format=bbox_format,
        feather_px=0,
        snap_to_krea_token_grid=False,
        batch_size=batch_size,
        batch_mode="per_batch" if batch_size > 1 else "repeat",
    )


def make_lora(region, name, *, enabled=True):
    return K2RegionalLora(
        region=region,
        positive=[f"{name} positive"],
        negative=["negative"],
        lora_name=name,
        start_percent=0.0,
        end_percent=1.0,
        enabled=enabled,
    )


def run(stack, constants, *, batch_size=1):
    initial = torch.zeros((batch_size, 4, 8, 8), dtype=torch.float32)

    def predict(branch_name, x, sigma, cond, uncond):
        if branch_name == "base":
            return torch.zeros_like(x)
        return torch.full_like(x, constants[branch_name])

    return run_regional_velocity_sampler(
        initial=initial,
        stack=stack,
        base_positive=["base positive"],
        base_negative=["base negative"],
        cfg=4.0,
        schedule=[1.0, 0.0],
        predict=predict,
        pin_outside_regions=True,
    )


def test_bbox_conversion_builds_pixel_latent_and_token_masks():
    region = make_region((16, 16, 16, 16))
    assert region.pixel_bbox == (16, 16, 32, 32)
    assert region.pixel_mask.shape == (1, 64, 64)
    assert region.latent_mask.shape == (1, 1, 8, 8)
    assert region.token_mask.shape == (1, 16, 1)
    assert torch.count_nonzero(region.latent_mask) == 4


def test_regional_delta_changes_only_the_selected_region():
    region = make_region((16, 16, 16, 16))
    stack = K2RegionalLoraStack((make_lora(region, "character"),))
    samples, base, union, debug = run(stack, {"character": 1.0})
    outside = union == 0
    inside = union == 1
    assert torch.equal(samples[outside.expand_as(samples)], base[outside.expand_as(base)])
    assert torch.all(samples[inside.expand_as(samples)] == -1)
    assert debug.outside_equal_after_step == [True]


def test_three_regions_keep_their_lora_deltas_isolated():
    regions = [
        make_region((0, 0, 16, 16)),
        make_region((16, 16, 16, 16)),
        make_region((32, 32, 16, 16)),
    ]
    stack = K2RegionalLoraStack(
        tuple(make_lora(region, name) for region, name in zip(regions, ("one", "two", "three")))
    )
    samples, _base, _union, _debug = run(stack, {"one": 1.0, "two": 2.0, "three": 3.0})
    assert torch.all(samples[:, :, 0:2, 0:2] == -1)
    assert torch.all(samples[:, :, 2:4, 2:4] == -2)
    assert torch.all(samples[:, :, 4:6, 4:6] == -3)
    assert torch.all(samples[:, :, 6:8, 6:8] == 0)


def test_lora_key_filters_keep_krea_writeback_targets():
    state = {
        "clip.encoder.layers.0.self_attn.q_proj.lora_up.weight": 1,
        "diffusion_model.blocks.0.attn.wq.lora_up.weight": 2,
        "diffusion_model.blocks.0.attn.wo.lora_up.weight": 3,
    }
    filtered = filter_lora_state_dict(
        state, attention_only_filter=True, ignore_text_encoder_lora=True
    )
    assert set(filtered.values()) == {2, 3}
    assert not is_krea_attention_out_lora_key("diffusion_model.blocks.0.attn.wq.lora_up.weight")
    assert is_krea_attention_out_lora_key("diffusion_model.blocks.0.attn.wo.lora_up.weight")
    assert is_krea_mlp_lora_key("diffusion_model.blocks.0.mlp.fc1.lora_up.weight")
    assert is_krea_writeback_lora_key("diffusion_model.blocks.0.mlp.fc2.lora_up.weight")
