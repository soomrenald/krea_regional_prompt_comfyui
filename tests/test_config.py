import json

import pytest

from krea_regional_prompt_comfyui.k2_region_comfy.config import (
    default_config_json,
    parse_studio_config,
)


def test_default_configuration_compiles_without_regions():
    config = parse_studio_config(default_config_json(), 1024, 768)
    assert config.width == 1024
    assert config.height == 768
    assert config.regions == ()
    assert config.spatial["enabled"] is True
    assert config.spatial["strict_lora_isolation"] is True


def test_region_configuration_compiles_pixel_box_and_prompt():
    payload = json.loads(default_config_json())
    payload["global_prompt"] = "a cinematic interior"
    payload["regions"] = [
        {
            "id": "left-person",
            "name": "Left person",
            "box": {"x0": 0, "y0": 64, "x1": 480, "y1": 960},
            "prompt": "a woman in a green coat",
            "negative_prompt": "",
            "face_identity_prompt": "",
            "enabled": True,
            "priority": 100,
            "spatial_role": "subject",
        }
    ]
    config = parse_studio_config(json.dumps(payload), 1024, 1024)
    assert len(config.regions) == 1
    assert config.regional_plan.regions[0].region_id == "left-person"
    assert "green coat" in config.regional_plan.prompt


def test_newer_configuration_version_is_rejected():
    with pytest.raises(ValueError, match="newer than supported"):
        parse_studio_config('{"version": 999}', 512, 512)


def test_legacy_spatial_section_enables_strict_isolation_by_default():
    config = parse_studio_config(
        json.dumps({"version": 1, "spatial": {"enabled": True, "strength": 1.0}}),
        512,
        512,
    )
    assert config.spatial["strict_lora_isolation"] is True


def test_strict_regional_lora_rejects_disabled_spatial_router():
    payload = json.loads(default_config_json())
    payload["spatial"]["enabled"] = False
    payload["regions"] = [
        {
            "id": "person",
            "name": "Person",
            "box": {"x0": 0, "y0": 0, "x1": 256, "y1": 512},
            "prompt": "a person",
            "enabled": True,
            "priority": 1,
            "spatial_role": "subject",
        }
    ]
    payload["loras"] = [
        {
            "id": "regional",
            "name": "regional.safetensors",
            "global": False,
            "region_ids": ["person"],
        }
    ]
    with pytest.raises(ValueError, match="requires spatial attention"):
        parse_studio_config(json.dumps(payload), 512, 512)
