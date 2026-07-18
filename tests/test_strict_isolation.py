from __future__ import annotations

import json

import pytest


torch = pytest.importorskip("torch")

from krea_regional_prompt_comfyui.k2_region_comfy.backend import (  # noqa: E402
    LoraDeltaStatistics,
    RuntimeState,
    pin_regional_latent_to_baseline,
)
from krea_regional_prompt_comfyui.k2_region_comfy.config import (  # noqa: E402
    default_config_json,
    parse_studio_config,
)
from krea_regional_prompt_comfyui.k2_region_core.regional_lora import (  # noqa: E402
    LoraDeltaRoute,
)


def make_runtime() -> RuntimeState:
    payload = json.loads(default_config_json())
    payload["regions"] = [
        {
            "id": "left",
            "name": "Left subject",
            "box": {"x0": 0, "y0": 0, "x1": 128, "y1": 256},
            "prompt": "first subject",
            "enabled": True,
            "priority": 100,
            "spatial_role": "subject",
        },
        {
            "id": "right",
            "name": "Right subject",
            "box": {"x0": 128, "y0": 0, "x1": 256, "y1": 256},
            "prompt": "second subject",
            "enabled": True,
            "priority": 99,
            "spatial_role": "subject",
        },
    ]
    config = parse_studio_config(json.dumps(payload), 256, 256)
    route = LoraDeltaRoute(
        lora_id="left-only-lora",
        display_name="Left only LoRA",
        strength=1.0,
        global_scope=False,
        region_ids=("left",),
        region_names=("Left subject",),
        text_token_mask=(1.0,),
        image_token_mask=(1.0,),
    )
    return RuntimeState(
        config=config,
        bound_plan=None,
        attention_override=None,
        lora_statistics=LoraDeltaStatistics((route,)),
        lora_reports=[],
        projector_report={},
        report={},
    )


def test_two_subject_strict_isolation_restores_unassigned_subject_to_baseline():
    runtime = make_runtime()
    baseline = torch.zeros((1, 4, 32, 32), dtype=torch.float32)
    regional = torch.ones_like(baseline)

    pinned, mask = pin_regional_latent_to_baseline(regional, baseline, runtime)

    assert runtime.strict_lora_isolation is True
    assert torch.all(mask[:, :, :, :16] == 1)
    assert torch.all(mask[:, :, :, 16:] == 0)
    assert torch.all(pinned[:, :, :, :16] == 1)
    assert torch.equal(pinned[:, :, :, 16:], baseline[:, :, :, 16:])


def test_global_lora_does_not_trigger_regional_baseline_pin():
    runtime = make_runtime()
    regional = runtime.lora_statistics.routes[0]
    global_route = LoraDeltaRoute(
        lora_id=regional.lora_id,
        display_name=regional.display_name,
        strength=regional.strength,
        global_scope=True,
        region_ids=(),
        region_names=(),
        text_token_mask=regional.text_token_mask,
        image_token_mask=regional.image_token_mask,
    )
    runtime.lora_statistics = LoraDeltaStatistics((global_route,))
    assert runtime.strict_lora_isolation is False
