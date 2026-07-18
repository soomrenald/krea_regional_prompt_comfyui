from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..k2_region_core.regional_lora import character_identity_triggers
from ..k2_region_core.regional_prompting import (
    GLOBAL_EMPHASIS_SCOPE,
    PromptEmphasis,
    RegionalPromptPlan,
    compile_regional_prompt_plan,
    prompt_emphases_from_payload,
    region_definitions_from_payload,
)
from ..k2_region_core.regions import RegionDefinition


CONFIG_VERSION = 1


DEFAULT_CONFIG: dict[str, Any] = {
    "version": CONFIG_VERSION,
    "global_prompt": "",
    "global_negative": "",
    "regions": [],
    "loras": [],
    "emphases": [],
    "spatial": {
        "enabled": True,
        "strength": 1.0,
        "outside_penalty": 1.0,
        "falloff_pixels": 128.0,
        "subject_competition": True,
        "subject_fill": True,
        "late_step_scale": 0.35,
        "lora_delta_adaptation": False,
        "lora_delta_adaptation_gain": 0.35,
        "strict_lora_isolation": True,
    },
    "projector": {
        "enabled": False,
        "preset": "filter_bypass2",
        "values": [0.0] * 12,
        "multiplier": 1.0,
        "identity_protection": 1.0,
    },
    "face_detail": {
        "enabled": False,
        "steps": 8,
        "denoise": 0.15,
        "crop_size": 512,
        "padding": 2.0,
        "feather": 0.12,
        "blend": 0.5,
        "lora_scale": 0.5,
        "detector_threshold": 0.4,
    },
}


@dataclass(frozen=True, slots=True)
class StudioConfig:
    raw: dict[str, Any]
    width: int
    height: int
    global_prompt: str
    global_negative: str
    regions: tuple[RegionDefinition, ...]
    loras: tuple[dict[str, Any], ...]
    emphases: tuple[PromptEmphasis, ...]
    regional_plan: RegionalPromptPlan

    @property
    def spatial(self) -> dict[str, Any]:
        return self.raw["spatial"]

    @property
    def projector(self) -> dict[str, Any]:
        return self.raw["projector"]

    @property
    def face_detail(self) -> dict[str, Any]:
        return self.raw["face_detail"]

    def summary(self) -> dict[str, Any]:
        return {
            "version": CONFIG_VERSION,
            "width": self.width,
            "height": self.height,
            "global_prompt": self.global_prompt,
            "global_negative": self.global_negative,
            "region_count": len(self.regions),
            "lora_count": len(self.loras),
            "emphasis_count": len(self.emphases),
            "regional_prompting": self.regional_plan.summary(),
            "projector": dict(self.projector),
            "face_detail": dict(self.face_detail),
        }


def default_config_json() -> str:
    return json.dumps(DEFAULT_CONFIG, indent=2)


def _merged_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    defaults = DEFAULT_CONFIG[name]
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return {**defaults, **value}


def parse_studio_config(encoded: str, width: int, height: int) -> StudioConfig:
    try:
        supplied = json.loads(encoded or "{}")
    except json.JSONDecodeError as error:
        raise ValueError(f"Region Studio configuration is not valid JSON: {error}") from error
    if not isinstance(supplied, dict):
        raise ValueError("Region Studio configuration must be a JSON object")
    version = int(supplied.get("version", CONFIG_VERSION))
    if version > CONFIG_VERSION:
        raise ValueError(
            f"Region Studio configuration version {version} is newer than supported version "
            f"{CONFIG_VERSION}"
        )
    aligned_width = max(256, min(16384, int(width)))
    aligned_height = max(256, min(16384, int(height)))
    raw = {
        **DEFAULT_CONFIG,
        **supplied,
        "version": CONFIG_VERSION,
        "spatial": _merged_section(supplied, "spatial"),
        "projector": _merged_section(supplied, "projector"),
        "face_detail": _merged_section(supplied, "face_detail"),
    }
    region_items = raw.get("regions", [])
    if not isinstance(region_items, list):
        raise ValueError("regions must be a JSON array")
    regions = region_definitions_from_payload(region_items)
    if len({region.region_id for region in regions}) != len(regions):
        raise ValueError("region IDs must be unique")
    lora_items = raw.get("loras", [])
    if not isinstance(lora_items, list):
        raise ValueError("loras must be a JSON array")
    loras = tuple(dict(item) for item in lora_items)
    emphases_payload = raw.get("emphases", [])
    if not isinstance(emphases_payload, list):
        raise ValueError("emphases must be a JSON array")
    emphases = prompt_emphases_from_payload(emphases_payload)
    spatial = raw["spatial"]
    plan = compile_regional_prompt_plan(
        aligned_width,
        aligned_height,
        str(raw.get("global_prompt", "")),
        regions,
        strength=float(spatial["strength"]),
        outside_penalty=float(spatial["outside_penalty"]),
        falloff_pixels=float(spatial["falloff_pixels"]),
        subject_competition=bool(spatial["subject_competition"]),
        subject_fill=bool(spatial["subject_fill"]),
        late_step_scale=float(spatial["late_step_scale"]),
        emphases=emphases,
        character_identity_triggers=character_identity_triggers(list(loras)),
    )
    return StudioConfig(
        raw=raw,
        width=plan.width,
        height=plan.height,
        global_prompt=str(raw.get("global_prompt", "")),
        global_negative=str(raw.get("global_negative", "")),
        regions=regions,
        loras=loras,
        emphases=emphases,
        regional_plan=plan,
    )


def empty_region(index: int = 0) -> dict[str, Any]:
    offset = 64 + index * 24
    return {
        "id": f"region-{index + 1}",
        "name": f"Region {index + 1}",
        "box": {"x0": offset, "y0": offset, "x1": offset + 384, "y1": offset + 512},
        "prompt": "",
        "negative_prompt": "",
        "face_identity_prompt": "",
        "enabled": True,
        "priority": max(0, 100 - index),
        "spatial_role": "auto",
    }


__all__ = [
    "CONFIG_VERSION",
    "DEFAULT_CONFIG",
    "GLOBAL_EMPHASIS_SCOPE",
    "StudioConfig",
    "default_config_json",
    "empty_region",
    "parse_studio_config",
]
