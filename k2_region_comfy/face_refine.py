from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..k2_region_core.face_detail import (
    FaceDetailSettings,
    OnnxNanoFaceDetector,
    assign_faces_to_regional_loras,
    composite_face_crop,
    discover_face_detector,
    expanded_square_crop,
)

from .backend import RuntimeState, normalize_lora_specs


def _to_pil(image) -> Image.Image:
    array = (
        image.detach().to(device="cpu", dtype=__import__("torch").float32)
        .clamp(0, 1).numpy() * 255.0
    ).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _to_tensor(image: Image.Image):
    import torch

    return torch.from_numpy(
        np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    )


def _face_model(base_model, specifications: list[dict[str, Any]], scale: float):
    import comfy.lora
    import comfy.lora_convert
    import comfy.utils

    from ..k2_region_core.lora import align_krea_lora_state_dict

    model = base_model
    reports = []
    for specification in specifications:
        state = comfy.utils.load_torch_file(specification["path"], safe_load=True)
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        aligned = align_krea_lora_state_dict(state, key_map)
        patches = comfy.lora.load_lora(
            comfy.lora_convert.convert_lora(aligned), key_map, log_missing=False
        )
        model = model.clone()
        applied = model.add_patches(
            patches, strength_patch=float(specification["strength"]) * scale
        )
        reports.append(
            {
                "id": specification["id"],
                "name": specification["name"],
                "matched_targets": len(applied),
            }
        )
    return model, reports


def refine_faces(
    image,
    model,
    clip,
    vae,
    runtime: RuntimeState,
    *,
    seed: int,
    sampler_name: str,
    scheduler: str,
    detector_path: str = "",
):
    import folder_paths
    import torch
    import comfy.sample

    settings = FaceDetailSettings(**runtime.config.face_detail)
    if not settings.enabled:
        return image, json.dumps({"status": "disabled", "faces_refined": 0}, indent=2)
    path = Path(detector_path).expanduser() if detector_path.strip() else discover_face_detector(
        Path(folder_paths.base_path)
    )
    if path is None or not path.is_file():
        raise FileNotFoundError(
            "Face refinement needs a NanoDet face_det.onnx model. Set detector_path "
            "on K2 Face Detail or install ComfyUI-WanVideoWrapper/FantasyPortrait."
        )
    detector = OnnxNanoFaceDetector(path, threshold=settings.detector_threshold)
    loras = normalize_lora_specs(runtime.config)
    all_outputs = []
    batch_report = []
    for batch_index, tensor_image in enumerate(image):
        canvas = _to_pil(tensor_image)
        detections = detector.detect(canvas)
        targets = assign_faces_to_regional_loras(
            detections, runtime.config.regions, loras
        )
        target_reports = []
        for target_index, target in enumerate(targets):
            crop_box = expanded_square_crop(
                target.face.box, canvas.width, canvas.height, settings.padding
            )
            source = canvas.crop(crop_box).resize(
                (settings.crop_size, settings.crop_size), Image.Resampling.LANCZOS
            )
            pixels = _to_tensor(source).unsqueeze(0)
            with torch.no_grad():
                latent = vae.encode(pixels)
            latent = comfy.sample.fix_empty_latent_channels(
                model, latent, downscale_ratio_spacial=8
            )
            positive = clip.encode_from_tokens_scheduled(clip.tokenize(target.prompt))
            negative = clip.encode_from_tokens_scheduled(
                clip.tokenize(runtime.config.global_negative)
            )
            generation_model, lora_reports = _face_model(
                model, list(target.loras), settings.lora_scale
            )
            face_seed = int(seed) + batch_index * 10_000 + target_index
            noise = comfy.sample.prepare_noise(latent, face_seed)
            samples = comfy.sample.sample(
                generation_model,
                noise,
                settings.steps,
                1.0,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent,
                denoise=settings.denoise,
                disable_pbar=True,
                seed=face_seed,
            )
            with torch.no_grad():
                decoded = vae.decode(samples)
            refined = _to_pil(decoded[0])
            canvas = composite_face_crop(
                canvas, refined, crop_box, settings.feather, settings.blend
            )
            target_reports.append(
                {
                    "region_id": target.region_id,
                    "region_name": target.region_name,
                    "crop_box": list(crop_box),
                    "seed": face_seed,
                    "loras": lora_reports,
                }
            )
        all_outputs.append(_to_tensor(canvas))
        batch_report.append(
            {
                "batch_index": batch_index,
                "detections": len(detections),
                "faces_refined": len(targets),
                "targets": target_reports,
            }
        )
    return torch.stack(all_outputs), json.dumps(
        {"status": "complete", "batches": batch_report}, indent=2
    )


__all__ = ["refine_faces"]
