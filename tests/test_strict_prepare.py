from __future__ import annotations

import json

from krea_regional_prompt_comfyui.k2_region_comfy import backend
from krea_regional_prompt_comfyui.k2_region_comfy.backend import (
    LoraDeltaStatistics,
    RuntimeState,
)
from krea_regional_prompt_comfyui.k2_region_comfy.config import (
    default_config_json,
    parse_studio_config,
)
from krea_regional_prompt_comfyui.k2_region_core.regional_lora import LoraDeltaRoute


def route(identifier: str, *, global_scope: bool) -> LoraDeltaRoute:
    return LoraDeltaRoute(
        lora_id=identifier,
        display_name=identifier,
        strength=1.0,
        global_scope=global_scope,
        region_ids=() if global_scope else ("left",),
        region_names=() if global_scope else ("Left",),
        text_token_mask=(1.0,),
        image_token_mask=(1.0,),
    )


def test_prepare_studio_builds_baseline_with_global_but_not_regional_loras(monkeypatch):
    payload = json.loads(default_config_json())
    payload["regions"] = [
        {
            "id": "left",
            "name": "Left",
            "box": {"x0": 0, "y0": 0, "x1": 128, "y1": 256},
            "prompt": "left subject",
            "enabled": True,
            "priority": 100,
            "spatial_role": "subject",
        }
    ]
    payload["loras"] = [
        {"id": "global", "name": "global.safetensors", "global": True},
        {
            "id": "regional",
            "name": "regional.safetensors",
            "global": False,
            "region_ids": ["left"],
        },
    ]
    config = parse_studio_config(json.dumps(payload), 256, 256)
    lora_calls = []

    monkeypatch.setattr(
        backend,
        "encode_studio_conditioning",
        lambda clip, supplied: ("positive", "negative", "bound", "prompt"),
    )
    monkeypatch.setattr(
        backend, "apply_projector", lambda model, supplied, bound: ("projected", {})
    )

    def fake_apply_loras(model, supplied, bound):
        del bound
        identifiers = tuple(str(item["id"]) for item in supplied.loras)
        lora_calls.append(identifiers)
        routes = tuple(
            route(identifier, global_scope=identifier == "global")
            for identifier in identifiers
        )
        return f"{model}:{','.join(identifiers)}", [], LoraDeltaStatistics(routes)

    def fake_attach(model, supplied, bound, statistics, reports, projector):
        del bound, reports
        runtime = RuntimeState(
            config=supplied,
            bound_plan=None,
            attention_override=None,
            lora_statistics=statistics,
            lora_reports=[],
            projector_report=projector,
            report={},
        )
        return f"attached:{model}", runtime

    monkeypatch.setattr(backend, "apply_loras", fake_apply_loras)
    monkeypatch.setattr(backend, "attach_spatial_attention", fake_attach)
    monkeypatch.setattr(backend, "make_empty_latent", lambda *args: {"samples": "latent"})
    monkeypatch.setattr(backend, "region_union_mask", lambda supplied: "mask")

    prepared = backend.prepare_studio("model", "clip", config)

    assert lora_calls == [("global", "regional"), ("global",)]
    assert prepared["plan"].strict_lora_isolation is True
    assert prepared["plan"].baseline_model == "attached:projected:global"
    assert prepared["plan"].report["strict_lora_isolation"]["extra_sampling_passes"] == 1
