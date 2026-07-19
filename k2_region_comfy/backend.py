from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..k2_region_core.lora import (
    adapter_prefixes,
    align_krea_lora_state_dict,
    inspect_lora_header,
)
from ..k2_region_core.projector import (
    PROJECTOR_VECTOR_COUNT,
    effective_projector_values,
    projector_preset_values,
    projector_token_delta_mask,
    validate_projector_values,
)
from ..k2_region_core.regional_lora import (
    LoraDeltaRoute,
    compile_lora_delta_routes,
    route_allows_adapter_target,
)
from ..k2_region_core.regional_prompting import (
    BoundRegionalPromptPlan,
    krea_prompt_token_count,
)
from ..k2_region_core.spatial_attention import KreaSpatialAttentionOverride

from .config import StudioConfig


ATTACHMENT_KEY = "k2_region_studio_runtime"


@dataclass(slots=True)
class LoraDeltaStatistics:
    routes: tuple[LoraDeltaRoute, ...]
    values: dict[str, dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        self.values = {
            route.lora_id: {
                "text_energy": None,
                "text_count": 0,
                "image_energy": None,
                "image_count": 0,
                "step_text_energy": None,
                "step_text_count": 0,
                "step_image_energy": None,
                "step_image_count": 0,
                "delta_reference": None,
                "calls": 0,
            }
            for route in self.routes
        }

    @staticmethod
    def _add(previous, value):
        return value if previous is None else previous + value

    @staticmethod
    def _rms(energy, count: int) -> float:
        if energy is None or count == 0:
            return 0.0
        return float((energy / count).sqrt().item())

    def observe(self, route: LoraDeltaRoute, token_norms, *, route_kind: str) -> None:
        state = self.values[route.lora_id]
        state["calls"] += 1
        batch = int(token_norms.shape[0])
        text_count = len(route.text_token_mask)
        enabled_text = sum(value > 0.0 for value in route.text_token_mask)
        if route_kind == "text_layerwise":
            text_norms = token_norms
            image_norms = None
            text_observations = (batch // text_count) * enabled_text * int(token_norms.shape[1])
        elif route_kind == "text_projector":
            text_norms = token_norms
            image_norms = None
            text_observations = batch * enabled_text * int(token_norms.shape[2])
        elif route_kind == "text_refiner":
            text_norms = token_norms
            image_norms = None
            text_observations = batch * enabled_text
        else:
            text_norms = token_norms[:, :text_count]
            image_norms = token_norms[:, text_count:]
            text_observations = batch * enabled_text
        if enabled_text:
            energy = text_norms.square().sum()
            state["text_energy"] = self._add(state["text_energy"], energy)
            state["text_count"] += text_observations
            state["step_text_energy"] = self._add(state["step_text_energy"], energy)
            state["step_text_count"] += text_observations
        enabled_image = sum(value > 0.0 for value in route.image_token_mask)
        if image_norms is not None and enabled_image:
            energy = image_norms.square().sum()
            state["image_energy"] = self._add(state["image_energy"], energy)
            state["image_count"] += batch * enabled_image
            state["step_image_energy"] = self._add(state["step_image_energy"], energy)
            state["step_image_count"] += batch * enabled_image

    def regional_attention_scales(self, gain: float) -> dict[str, float]:
        region_values: dict[str, list[float]] = {}
        route_map = {route.lora_id: route for route in self.routes}
        for lora_id, route in route_map.items():
            if route.global_scope or not route.region_ids:
                continue
            state = self.values[lora_id]
            values = [
                self._rms(state["step_text_energy"], state["step_text_count"]),
                self._rms(state["step_image_energy"], state["step_image_count"]),
            ]
            values = [value for value in values if value > 0.0]
            if not values:
                continue
            observed = sum(values) / len(values)
            reference = state["delta_reference"] or observed
            ratio = observed / max(float(reference), 1e-12)
            scale = min(1.5, max(0.5, 1.0 + gain * (ratio - 1.0)))
            state["delta_reference"] = 0.85 * float(reference) + 0.15 * observed
            for region_id in route.region_ids:
                region_values.setdefault(region_id, []).append(scale)
        return {
            region_id: sum(values) / len(values)
            for region_id, values in region_values.items()
        }

    def reset_step_measurements(self) -> None:
        for state in self.values.values():
            state["step_text_energy"] = None
            state["step_text_count"] = 0
            state["step_image_energy"] = None
            state["step_image_count"] = 0

    def release_device_state(self) -> None:
        """Drop per-run tensor accumulators so cached runtimes retain no GPU tensors."""
        for state in self.values.values():
            state["text_energy"] = None
            state["text_count"] = 0
            state["image_energy"] = None
            state["image_count"] = 0
            state["step_text_energy"] = None
            state["step_text_count"] = 0
            state["step_image_energy"] = None
            state["step_image_count"] = 0
            state["delta_reference"] = None
            state["calls"] = 0

    def summary(self) -> dict[str, Any]:
        return {
            lora_id: {
                "calls": state["calls"],
                "text_delta_rms": self._rms(state["text_energy"], state["text_count"]),
                "image_delta_rms": self._rms(state["image_energy"], state["image_count"]),
                "outside_gate_delta_rms": 0.0,
            }
            for lora_id, state in self.values.items()
        }


@dataclass(slots=True)
class RuntimeState:
    config: StudioConfig
    bound_plan: BoundRegionalPromptPlan
    attention_override: KreaSpatialAttentionOverride | None
    lora_statistics: LoraDeltaStatistics
    lora_reports: list[dict[str, Any]]
    projector_report: dict[str, Any]
    report: dict[str, Any]
    device_release_callbacks: tuple[Callable[[], None], ...] = ()

    def update_step(self, completed: int, total: int) -> None:
        if self.attention_override is None:
            return
        self.attention_override.set_denoising_progress(completed, total)
        spatial = self.config.spatial
        if bool(spatial.get("lora_delta_adaptation", False)):
            self.attention_override.set_lora_delta_scales(
                self.lora_statistics.regional_attention_scales(
                    float(spatial.get("lora_delta_adaptation_gain", 0.35))
                )
            )
            self.lora_statistics.reset_step_measurements()

    def final_report(self) -> dict[str, Any]:
        result = dict(self.report)
        if self.attention_override is not None:
            if self.attention_override.matched_calls == 0:
                raise RuntimeError(
                    "Krea main-stream attention was not reached by the spatial override"
                )
            if (
                self.attention_override.strict_isolation
                and self.attention_override.text_refiner_calls == 0
            ):
                raise RuntimeError(
                    "Krea text-refiner attention was not reached by the regional "
                    "text partition"
                )
        result["attention_calls"] = (
            self.attention_override.matched_calls if self.attention_override else 0
        )
        result["regional_attention"] = (
            self.attention_override.summary()
            if self.attention_override
            else {
                "strict_lora_isolation": bool(
                    self.config.spatial.get("strict_lora_isolation", True)
                ),
                "status": "disabled",
            }
        )
        result["lora_delta_statistics"] = self.lora_statistics.summary()
        return result

    def release_device_state(self) -> None:
        """Release GPU-only data owned by a cached Studio model after sampling."""
        if self.attention_override is not None:
            self.attention_override.clear()
        for release in self.device_release_callbacks:
            release()
        self.lora_statistics.release_device_state()


def normalize_lora_specs(config: StudioConfig) -> list[dict[str, Any]]:
    import folder_paths

    normalized: list[dict[str, Any]] = []
    for index, supplied in enumerate(config.loras):
        item = dict(supplied)
        name = str(item.get("name") or item.get("lora_name") or "").strip()
        if not name or name == "None":
            continue
        path = folder_paths.get_full_path_or_raise("loras", name)
        item.update(
            {
                "id": str(item.get("id", f"lora-{index + 1}")),
                "name": str(item.get("display_name", Path(name).stem)),
                "path": path,
                "strength": float(item.get("strength", 1.0)),
                "global": bool(item.get("global", True)),
                "region_ids": list(map(str, item.get("region_ids", []))),
                "routing_mode": str(item.get("routing_mode", "standard")),
                "trigger_phrase": str(item.get("trigger_phrase", "")),
            }
        )
        normalized.append(item)
    return normalized


def encode_studio_conditioning(clip, config: StudioConfig):
    plan = config.regional_plan
    prompt = plan.prompt if (plan.regions or plan.emphases) else config.global_prompt
    tokens = clip.tokenize(prompt)
    positive = clip.encode_from_tokens_scheduled(tokens)
    negative_text = config.global_negative.strip()
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_text))
    if not positive:
        raise RuntimeError("Krea text encoder returned no conditioning")
    token_counts = {int(condition[0].shape[1]) for condition in positive}
    if len(token_counts) != 1:
        raise RuntimeError("Krea conditioning must contain one text sequence length")
    bound = plan.bind_tokens(
        lambda prefix: krea_prompt_token_count(clip.tokenize(prefix)),
        conditioning_text_token_count=token_counts.pop(),
    )
    return positive, negative, bound, prompt


def make_empty_latent(model, width: int, height: int, batch_size: int = 1):
    import torch
    import comfy.model_management
    import comfy.sample

    latent = torch.zeros(
        [batch_size, 4, height // 8, width // 8],
        device=comfy.model_management.intermediate_device(),
        dtype=comfy.model_management.intermediate_dtype(),
    )
    latent = comfy.sample.fix_empty_latent_channels(
        model, latent, downscale_ratio_spacial=8
    )
    return {"samples": latent}


def region_union_mask(config: StudioConfig):
    import torch

    geometry = config.regional_plan
    values = [0.0] * (geometry.image_token_width * geometry.image_token_height)
    for region in geometry.regions:
        values = [
            max(current, candidate)
            for current, candidate in zip(values, region.image_token_field, strict=True)
        ]
    token_mask = torch.tensor(values, dtype=torch.float32).reshape(
        1, geometry.image_token_height, geometry.image_token_width
    )
    return torch.nn.functional.interpolate(
        token_mask.unsqueeze(1),
        size=(config.height, config.width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def _load_lora_patches(model, specification: dict[str, Any]):
    import comfy.lora
    import comfy.lora_convert
    import comfy.utils

    path = Path(str(specification["path"])).expanduser().resolve()
    state, metadata = comfy.utils.load_torch_file(
        str(path), safe_load=True, return_metadata=True
    )
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    aligned = align_krea_lora_state_dict(state, key_map)
    converted = comfy.lora_convert.convert_lora(aligned)
    patches = comfy.lora.load_lora(converted, key_map, log_missing=False)
    header = inspect_lora_header(path)
    prefixes = adapter_prefixes(converted)
    unmatched = [prefix for prefix in prefixes if prefix not in key_map]
    adapter_count = int(header["adapter_count"])
    report = {
        **header,
        "id": specification["id"],
        "display_name": specification["name"],
        "strength": specification["strength"],
        "global": specification["global"],
        "region_ids": specification["region_ids"],
        "routing_mode": specification["routing_mode"],
        "trigger_phrase": specification["trigger_phrase"],
        "matched_model_targets": len(patches),
        "unmatched_adapter_targets": max(0, adapter_count - len(patches)),
        "unmatched_prefix_examples": unmatched[:8],
        "compatible": bool(patches)
        and len(patches) == adapter_count
        and int(header["complete_adapter_pairs"]) == adapter_count,
        "model_only": True,
    }
    return patches, metadata, report


def _install_routed_loras(model, target_entries, statistics: LoraDeltaStatistics):
    import torch
    import comfy.weight_adapter

    base_adapter_type = comfy.weight_adapter.WeightAdapterBase

    class RoutedCompositeAdapter(base_adapter_type):
        name = "k2_region_studio_routed_composite"

        def __init__(self, entries, route_kind: str) -> None:
            self.entries = entries
            self.route_kind = route_kind
            self.weights = []
            self.loaded_keys = set()
            self._prepared: set[int] = set()
            self._mask_cache = {}

        def _prepare(self, adapter, route, x) -> None:
            adapter.multiplier = route.strength
            for name in (
                "is_conv",
                "conv_dim",
                "kernel_size",
                "in_channels",
                "out_channels",
                "kw_dict",
            ):
                setattr(adapter, name, getattr(self, name))
            identity = id(adapter)
            if identity in self._prepared:
                return
            weights = getattr(adapter, "weights", None)
            if isinstance(weights, (tuple, list)):
                moved = []
                for weight in weights:
                    if isinstance(weight, torch.Tensor):
                        dtype = x.dtype if weight.is_floating_point() else weight.dtype
                        moved.append(weight.to(device=x.device, dtype=dtype))
                    else:
                        moved.append(weight)
                adapter.weights = type(weights)(moved)
            self._prepared.add(identity)

        def _mask(self, route, x):
            key = (
                route.lora_id,
                self.route_kind,
                tuple(x.shape),
                x.device,
                x.dtype,
            )
            cached = self._mask_cache.get(key)
            if cached is not None:
                return cached
            if self.route_kind == "text_layerwise":
                values = route.layerwise_text_batch_mask(int(x.shape[0]))
                mask = torch.tensor(values, device=x.device, dtype=x.dtype).view(-1, 1, 1)
            elif self.route_kind == "text_projector":
                values = route.sequence_mask(int(x.shape[1]), text_fusion=True)
                mask = torch.tensor(values, device=x.device, dtype=x.dtype).view(
                    1, -1, 1, 1
                )
            else:
                values = route.sequence_mask(
                    int(x.shape[-2]), text_fusion=self.route_kind == "text_refiner"
                )
                mask = torch.tensor(values, device=x.device, dtype=x.dtype).view(1, -1, 1)
            self._mask_cache[key] = mask
            return mask

        def h(self, x, base_out):
            total = torch.zeros_like(base_out)
            for adapter, route in self.entries:
                self._prepare(adapter, route, x)
                applied = adapter.h(x, base_out) * self._mask(route, x)
                statistics.observe(
                    route,
                    torch.linalg.vector_norm(applied.detach(), dim=-1, dtype=torch.float32),
                    route_kind=self.route_kind,
                )
                total = total + applied
            return total

        def release_device_state(self) -> None:
            """Keep reusable adapter weights in RAM, not in the device allocator."""
            for adapter, _route in self.entries:
                weights = getattr(adapter, "weights", None)
                if not isinstance(weights, (tuple, list)):
                    continue
                moved = [
                    weight.detach().to(device="cpu")
                    if isinstance(weight, torch.Tensor)
                    else weight
                    for weight in weights
                ]
                adapter.weights = type(weights)(moved)
            self._prepared.clear()
            self._mask_cache.clear()

    manager = comfy.weight_adapter.BypassInjectionManager()
    routed_adapters = []
    for key, entries in target_entries.items():
        if not all(isinstance(adapter, base_adapter_type) for adapter, _route in entries):
            raise ValueError(f"unsupported non-adapter regional LoRA patch: {key}")
        route_kind = (
            "text_layerwise"
            if ".txtfusion.layerwise_blocks." in str(key)
            else "text_projector"
            if ".txtfusion.projector." in str(key)
            else "text_refiner"
            if ".txtfusion." in str(key) or ".txtmlp." in str(key)
            else "combined"
        )
        routed_adapter = RoutedCompositeAdapter(entries, route_kind=route_kind)
        routed_adapters.append(routed_adapter)
        manager.add_adapter(key, routed_adapter, strength=1.0)
    patched = model.clone()
    injections = manager.create_injections(patched.model)
    patched.set_injections("k2_region_studio_loras", injections)
    if manager.get_hook_count() != len(target_entries):
        raise RuntimeError(
            f"installed {manager.get_hook_count()}/{len(target_entries)} routed LoRA hooks"
        )
    return patched, tuple(adapter.release_device_state for adapter in routed_adapters)


def apply_loras(
    model,
    config: StudioConfig,
    bound_plan: BoundRegionalPromptPlan,
):
    specifications = normalize_lora_specs(config)
    routes = compile_lora_delta_routes(
        specifications,
        width=config.width,
        height=config.height,
        text_token_count=bound_plan.text_token_count,
        regional_plan=config.regional_plan,
        bound_plan=bound_plan,
    )
    route_map = {route.lora_id: route for route in routes}
    strict_isolation = bool(
        getattr(config, "spatial", {}).get("strict_lora_isolation", True)
    )
    target_entries: dict[str, list[tuple[Any, LoraDeltaRoute]]] = {}
    skipped_targets: dict[str, list[str]] = {}
    reports: list[dict[str, Any]] = []
    metadata_items = []
    for specification in specifications:
        if specification["strength"] == 0.0:
            reports.append({**specification, "status": "disabled"})
            continue
        patches, metadata, report = _load_lora_patches(model, specification)
        if not report["compatible"]:
            raise ValueError(
                f"LoRA {report['display_name']} matched "
                f"{report['matched_model_targets']}/{report['adapter_count']} Krea targets"
            )
        route = route_map[specification["id"]]
        for key, adapter in patches.items():
            if not strict_isolation or route_allows_adapter_target(route, str(key)):
                target_entries.setdefault(key, []).append((adapter, route))
            else:
                skipped_targets.setdefault(route.lora_id, []).append(str(key))
        report["status"] = "applied_global" if route.global_scope else "applied_regional"
        report["application_mode"] = (
            "unfused_token_delta_gate"
            if route.global_scope
            else "unfused_region_text_image_delta_gate_v3"
        )
        skipped = skipped_targets.get(route.lora_id, [])
        report["applied_model_targets"] = len(patches) - len(skipped)
        report["locality_skipped_targets"] = len(skipped)
        report["locality_skipped_target_examples"] = skipped[:8]
        report["route"] = route.summary()
        if report["applied_model_targets"] == 0:
            raise ValueError(
                f"Regional LoRA {report['display_name']!r} has no targets that can "
                "be routed locally; no LoRA was applied"
            )
        reports.append(report)
        if metadata:
            metadata_items.append({"id": route.lora_id, "metadata": metadata})
    statistics = LoraDeltaStatistics(routes)
    if target_entries:
        patched, device_release_callbacks = _install_routed_loras(
            model, target_entries, statistics
        )
    else:
        patched, device_release_callbacks = model, ()
    if metadata_items:
        patched.set_attachments("lora_metadata", metadata_items)
    return patched, reports, statistics, device_release_callbacks


def apply_projector(model, config: StudioConfig, bound: BoundRegionalPromptPlan):
    projector = config.projector
    enabled = bool(projector.get("enabled", False))
    preset = str(projector.get("preset", "filter_bypass2"))
    supplied_values = projector.get("values")
    values = (
        validate_projector_values(supplied_values)
        if preset == "custom" and supplied_values
        else projector_preset_values(preset)
    )
    effective = effective_projector_values(values, float(projector.get("multiplier", 1.0)))
    report = {
        "enabled": enabled,
        "preset": preset,
        "values": list(values),
        "effective_values": list(effective),
        "identity_protection": float(projector.get("identity_protection", 1.0)),
        "target": "diffusion_model.txtfusion.projector.weight",
    }
    if not enabled or not any(effective):
        report["status"] = "disabled" if not enabled else "zero_effect"
        return model, report
    import torch

    target = report["target"]
    state = model.model.state_dict()
    if target not in state:
        raise RuntimeError(f"Krea projector target is missing: {target}")
    if tuple(state[target].shape) != (1, PROJECTOR_VECTOR_COUNT):
        raise RuntimeError(f"unexpected Krea projector shape: {tuple(state[target].shape)}")
    delta = torch.tensor((effective,), dtype=torch.float32)
    protection = float(projector.get("identity_protection", 1.0))
    protected = tuple((identity.start, identity.end) for identity in bound.face_identities)
    if protected and protection > 0.0:
        import torch.nn.functional as functional
        import comfy.weight_adapter

        mask_values = projector_token_delta_mask(
            bound.text_token_count, protected, protection
        )
        base_adapter_type = comfy.weight_adapter.WeightAdapterBase

        class TokenSelectiveProjectorDelta(base_adapter_type):
            name = "k2_region_studio_projector"

            def __init__(self) -> None:
                self.weights = (delta,)
                self.loaded_keys = set()
                self._cache = {}

            def h(self, x, base_out):
                del base_out
                if x.ndim != 4 or x.shape[1] != len(mask_values):
                    raise RuntimeError("Krea projector received an unexpected token shape")
                key = (x.device, x.dtype)
                weight, mask = self._cache.get(key, (None, None))
                if weight is None:
                    weight = self.weights[0].to(device=x.device, dtype=x.dtype)
                    mask = torch.tensor(mask_values, device=x.device, dtype=x.dtype).view(
                        1, -1, 1, 1
                    )
                    self._cache[key] = (weight, mask)
                return functional.linear(x, weight) * mask

        manager = comfy.weight_adapter.BypassInjectionManager()
        manager.add_adapter(target, TokenSelectiveProjectorDelta(), strength=1.0)
        patched = model.clone()
        patched.set_injections(
            "k2_region_studio_projector", manager.create_injections(patched.model)
        )
        report["status"] = "applied_token_selective_diff"
        report["protected_token_spans"] = [list(span) for span in protected]
    else:
        patched = model.clone()
        if target not in patched.add_patches({target: ("diff", (delta,))}):
            raise RuntimeError("could not apply the Krea projector patch")
        report["status"] = "applied_global_diff"
    patched.set_attachments("projector_settings", report)
    return patched, report


def attach_spatial_attention(
    model,
    config: StudioConfig,
    bound: BoundRegionalPromptPlan,
    statistics: LoraDeltaStatistics,
    lora_reports: list[dict[str, Any]],
    projector_report: dict[str, Any],
):
    spatial = config.spatial
    enabled = bool(spatial.get("enabled", True)) and bool(bound.spans or bound.emphases)
    if not enabled and any(
        report.get("status") == "applied_regional" for report in lora_reports
    ):
        raise ValueError(
            "Regional LoRA isolation requires Spatial attention Enabled so its "
            "regional text cannot become shared scene conditioning"
        )
    override = None
    patched = model.clone()
    if enabled:
        override = KreaSpatialAttentionOverride(
            bound,
            lora_delta_adaptation=bool(spatial.get("lora_delta_adaptation", False)),
            lora_delta_adaptation_gain=float(
                spatial.get("lora_delta_adaptation_gain", 0.35)
            ),
            strict_isolation=bool(spatial.get("strict_lora_isolation", True)),
        )
        transformer_options = patched.model_options.setdefault("transformer_options", {})
        if "optimized_attention_override" in transformer_options:
            raise RuntimeError(
                "K2 Region Studio cannot be combined with another optimized-attention "
                "override on the same MODEL branch"
            )
        transformer_options["optimized_attention_override"] = override
    report = {
        **config.summary(),
        "lora_reports": lora_reports,
        "projector_report": projector_report,
        "standard_comfy_types": [
            "MODEL",
            "CLIP",
            "CONDITIONING",
            "LATENT",
            "MASK",
            "IMAGE",
        ],
    }
    runtime = RuntimeState(
        config=config,
        bound_plan=bound,
        attention_override=override,
        lora_statistics=statistics,
        lora_reports=lora_reports,
        projector_report=projector_report,
        report=report,
    )
    patched.set_attachments(ATTACHMENT_KEY, runtime)
    return patched, runtime


def prepare_studio(model, clip, config: StudioConfig, batch_size: int = 1):
    positive, negative, bound, prompt = encode_studio_conditioning(clip, config)
    patched, projector_report = apply_projector(model, config, bound)
    patched, lora_reports, statistics, device_release_callbacks = apply_loras(
        patched, config, bound
    )
    patched, runtime = attach_spatial_attention(
        patched, config, bound, statistics, lora_reports, projector_report
    )
    runtime.device_release_callbacks = device_release_callbacks
    latent = make_empty_latent(patched, config.width, config.height, batch_size)
    mask = region_union_mask(config)
    return {
        "model": patched,
        "clip": clip,
        "positive": positive,
        "negative": negative,
        "latent": latent,
        "mask": mask,
        "plan": runtime,
        "prompt": prompt,
        "report": json.dumps(runtime.report, indent=2, default=str),
    }


__all__ = [
    "ATTACHMENT_KEY",
    "RuntimeState",
    "normalize_lora_specs",
    "prepare_studio",
]
