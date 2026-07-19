from __future__ import annotations

from types import SimpleNamespace

from krea_regional_prompt_comfyui.k2_region_comfy.backend import (
    LoraDeltaStatistics,
    RuntimeState,
)


def test_lora_statistics_release_drops_all_per_run_tensor_references() -> None:
    statistics = LoraDeltaStatistics((SimpleNamespace(lora_id="subject"),))
    state = statistics.values["subject"]
    sentinel = object()
    for key in (
        "text_energy",
        "image_energy",
        "step_text_energy",
        "step_image_energy",
        "delta_reference",
    ):
        state[key] = sentinel
    for key in (
        "text_count",
        "image_count",
        "step_text_count",
        "step_image_count",
        "calls",
    ):
        state[key] = 7

    statistics.release_device_state()

    assert all(value is None for key, value in state.items() if "energy" in key)
    assert state["delta_reference"] is None
    assert state["text_count"] == 0
    assert state["image_count"] == 0
    assert state["step_text_count"] == 0
    assert state["step_image_count"] == 0
    assert state["calls"] == 0


def test_runtime_release_clears_attention_and_routed_lora_device_caches() -> None:
    calls: list[str] = []
    attention = SimpleNamespace(clear=lambda: calls.append("attention"))
    statistics = SimpleNamespace(release_device_state=lambda: calls.append("statistics"))
    runtime = RuntimeState(
        config=None,
        bound_plan=None,
        attention_override=attention,
        lora_statistics=statistics,
        lora_reports=[],
        projector_report={},
        report={},
        device_release_callbacks=(lambda: calls.append("lora"),),
    )

    runtime.release_device_state()

    assert calls == ["attention", "lora", "statistics"]
