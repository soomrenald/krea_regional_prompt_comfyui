from __future__ import annotations

import json

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .backend import RuntimeState, prepare_studio
from .config import default_config_json, parse_studio_config
from .face_refine import refine_faces
from .bare_nodes import BARE_NODE_CLASSES


K2Plan = io.Custom("K2_REGION_PLAN")
CATEGORY = "K2 Region Studio"


class K2KreaLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import folder_paths
        import nodes

        return io.Schema(
            node_id="K2KreaLoader",
            display_name="K2 Load Krea 2",
            category=f"{CATEGORY}/loaders",
            description="Convenience loader for native ComfyUI Krea 2 model components.",
            inputs=[
                io.Combo.Input(
                    "diffusion_model",
                    options=folder_paths.get_filename_list("diffusion_models"),
                    tooltip="Krea 2 diffusion-model file from ComfyUI/models/diffusion_models.",
                ),
                io.Combo.Input(
                    "text_encoder",
                    options=folder_paths.get_filename_list("text_encoders"),
                    tooltip="Krea-compatible Qwen text encoder from ComfyUI/models/text_encoders.",
                ),
                io.Combo.Input(
                    "vae",
                    options=nodes.VAELoader.vae_list(nodes.VAELoader),
                    tooltip="VAE used to encode and decode Krea latents.",
                ),
                io.Combo.Input(
                    "weight_dtype",
                    options=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                    default="default",
                    tooltip="Model weight precision. Default follows the file/runtime; FP8 reduces memory when supported.",
                ),
                io.Combo.Input(
                    "text_encoder_device",
                    options=["default", "cpu"],
                    default="default",
                    tooltip="Keep default ComfyUI placement or force the Qwen text encoder to CPU to save VRAM.",
                ),
            ],
            outputs=[io.Model.Output(), io.Clip.Output(), io.Vae.Output()],
        )

    @classmethod
    def execute(
        cls, diffusion_model, text_encoder, vae, weight_dtype, text_encoder_device
    ):
        import torch
        import comfy.sd
        import folder_paths
        import nodes

        options = {}
        if weight_dtype == "fp8_e4m3fn":
            options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            options.update(dtype=torch.float8_e4m3fn, fp8_optimizations=True)
        elif weight_dtype == "fp8_e5m2":
            options["dtype"] = torch.float8_e5m2
        model = comfy.sd.load_diffusion_model(
            folder_paths.get_full_path_or_raise("diffusion_models", diffusion_model),
            model_options=options,
        )
        clip_options = {}
        if text_encoder_device == "cpu":
            cpu = torch.device("cpu")
            clip_options.update(load_device=cpu, offload_device=cpu)
        clip = comfy.sd.load_clip(
            ckpt_paths=[
                folder_paths.get_full_path_or_raise("text_encoders", text_encoder)
            ],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.KREA2,
            model_options=clip_options,
        )
        loaded_vae = nodes.VAELoader().load_vae(vae)[0]
        return io.NodeOutput(model, clip, loaded_vae)


class K2RegionStudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="K2RegionStudio",
            display_name="K2 Region Studio",
            category=CATEGORY,
            description=(
                "Compiles the sidebar region canvas into Krea spatial attention, "
                "regional LoRA routing, conditioning, latent, and masks."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="Native Krea 2 MODEL from K2 Load Krea 2 or a compatible ComfyUI loader.",
                ),
                io.Clip.Input(
                    "clip",
                    tooltip="Krea/Qwen CLIP used to tokenize the compiled global and regional prompts.",
                ),
                io.String.Input(
                    "region_config",
                    multiline=True,
                    default=default_config_json(),
                    tooltip="Portable JSON written by the K2 Regions sidebar; normally edited through the sidebar rather than here.",
                ),
                io.Int.Input(
                    "width",
                    default=1024,
                    min=256,
                    max=16384,
                    step=8,
                    tooltip="Output canvas width in pixels; also defines the sidebar region coordinate system.",
                ),
                io.Int.Input(
                    "height",
                    default=1024,
                    min=256,
                    max=16384,
                    step=8,
                    tooltip="Output canvas height in pixels; also defines the sidebar region coordinate system.",
                ),
                io.Int.Input(
                    "batch_size",
                    default=1,
                    min=1,
                    max=64,
                    tooltip="Number of latent images generated in one batch.",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="patched_model"),
                io.Clip.Output(),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(),
                io.Mask.Output(display_name="region_union_mask"),
                K2Plan.Output(display_name="region_plan"),
                io.String.Output(display_name="compiled_prompt"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, region_config, width, height, batch_size):
        config = parse_studio_config(region_config, width, height)
        prepared = prepare_studio(model, clip, config, batch_size)
        return io.NodeOutput(
            prepared["model"], prepared["clip"], prepared["positive"],
            prepared["negative"], prepared["latent"], prepared["mask"],
            prepared["plan"], prepared["prompt"], prepared["report"],
        )


class K2RegionalSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import comfy.samplers

        return io.Schema(
            node_id="K2RegionalSampler",
            display_name="K2 Regional Sampler",
            category=f"{CATEGORY}/sampling",
            description=(
                "KSampler-compatible sampler that also updates late-step relaxation "
                "and optional LoRA-delta adaptation."
            ),
            inputs=[
                io.Model.Input("model", tooltip="Patched MODEL output from K2 Region Studio."),
                io.Conditioning.Input(
                    "positive", tooltip="Compiled positive conditioning from K2 Region Studio."
                ),
                io.Conditioning.Input(
                    "negative", tooltip="Global negative conditioning from K2 Region Studio."
                ),
                io.Latent.Input(
                    "latent", tooltip="Starting latent, normally the LATENT output from K2 Region Studio."
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip="Random-noise seed. The same seed and settings reproduce the same starting noise.",
                ),
                io.Int.Input(
                    "steps",
                    default=20,
                    min=1,
                    max=10000,
                    tooltip="Number of denoising iterations. Krea Turbo commonly uses a small step count.",
                ),
                io.Float.Input(
                    "cfg",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    tooltip="Classifier-free guidance scale. Krea Turbo is designed around CFG 1.0.",
                ),
                io.Combo.Input(
                    "sampler_name",
                    options=comfy.samplers.KSampler.SAMPLERS,
                    default="euler",
                    tooltip="Denoising algorithm; Euler is the recommended default for the Krea Turbo workflow.",
                ),
                io.Combo.Input(
                    "scheduler",
                    options=comfy.samplers.KSampler.SCHEDULERS,
                    default="simple",
                    tooltip="Sigma/noise schedule; Simple is the recommended Krea Turbo default.",
                ),
                io.Float.Input(
                    "denoise",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Fraction of the schedule to run. Use 1.0 for text-to-image; lower values preserve an input latent.",
                ),
                K2Plan.Input(
                    "region_plan",
                    optional=True,
                    tooltip="Optional runtime plan that updates late-step relaxation and LoRA-delta adaptation during sampling.",
                ),
            ],
            outputs=[io.Latent.Output(), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(
        cls, model, positive, negative, latent, seed, steps, cfg,
        sampler_name, scheduler, denoise, region_plan=None,
    ):
        import comfy.model_management
        import comfy.sample
        import comfy.utils
        import latent_preview

        samples = comfy.sample.fix_empty_latent_channels(
            model,
            latent["samples"],
            latent.get("downscale_ratio_spacial"),
            latent.get("downscale_ratio_temporal"),
        )
        noise = comfy.sample.prepare_noise(samples, seed, latent.get("batch_index"))
        preview = latent_preview.prepare_callback(model, steps)

        def callback(step, denoised, current, total):
            if isinstance(region_plan, RuntimeState):
                region_plan.update_step(step + 1, total)
            preview(step, denoised, current, total)

        runtime = region_plan if isinstance(region_plan, RuntimeState) else None
        try:
            result = comfy.sample.sample(
                model, noise, steps, cfg, sampler_name, scheduler, positive, negative,
                samples, denoise=denoise, noise_mask=latent.get("noise_mask"),
                callback=callback, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed,
            )
            output = latent.copy()
            output.pop("downscale_ratio_spacial", None)
            output.pop("downscale_ratio_temporal", None)
            output["samples"] = result
            report = (
                runtime.final_report() if runtime is not None
                else {"status": "sampled", "regional_progress_updates": False}
            )
            return io.NodeOutput(output, json.dumps(report, indent=2, default=str))
        finally:
            if runtime is not None:
                runtime.release_device_state()


class K2FaceDetail(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import comfy.samplers

        return io.Schema(
            node_id="K2FaceDetail",
            display_name="K2 Regional Face Detail",
            category=f"{CATEGORY}/postprocessing",
            description="Detects faces, matches them to regions, and refines each with its routed character LoRAs.",
            inputs=[
                io.Image.Input("image", tooltip="Decoded image whose detected faces will be refined."),
                io.Model.Input("model", tooltip="Base Krea MODEL used for each face crop pass."),
                io.Clip.Input("clip", tooltip="Krea/Qwen CLIP used to encode each assigned face prompt."),
                io.Vae.Input("vae", tooltip="VAE used to encode face crops and decode their refined latents."),
                K2Plan.Input(
                    "region_plan",
                    tooltip="Runtime plan from K2 Region Studio containing region, LoRA, prompt, and face-detail settings.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip="Base face-refinement seed; each detected face receives a deterministic offset.",
                ),
                io.Combo.Input(
                    "sampler_name",
                    options=comfy.samplers.KSampler.SAMPLERS,
                    default="euler",
                    tooltip="Sampler used for every face crop refinement pass.",
                ),
                io.Combo.Input(
                    "scheduler",
                    options=comfy.samplers.KSampler.SCHEDULERS,
                    default="simple",
                    tooltip="Noise schedule used for every face crop refinement pass.",
                ),
                io.String.Input(
                    "detector_path",
                    default="",
                    tooltip="Optional path to face_det.onnx. Leave blank to auto-discover FantasyPortrait's detector.",
                ),
            ],
            outputs=[io.Image.Output(), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, image, model, clip, vae, region_plan, seed, sampler_name, scheduler, detector_path):
        output, report = refine_faces(
            image, model, clip, vae, region_plan, seed=seed,
            sampler_name=sampler_name, scheduler=scheduler, detector_path=detector_path,
        )
        return io.NodeOutput(output, report)


class K2PostUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="K2PostUpscale",
            display_name="K2 Post Upscale",
            category=f"{CATEGORY}/postprocessing",
            inputs=[
                io.Image.Input("image", tooltip="Image to resize or neural-upscale."),
                io.Float.Input(
                    "scale",
                    default=2.0,
                    min=1.0,
                    max=8.0,
                    step=0.25,
                    tooltip="Final width and height multiplier relative to the input image.",
                ),
                io.Combo.Input(
                    "method",
                    options=["lanczos", "upscale_model"],
                    default="lanczos",
                    tooltip="Lanczos performs exact resizing; upscale_model first runs the connected neural model, then resizes to the requested scale.",
                ),
                io.UpscaleModel.Input(
                    "upscale_model",
                    optional=True,
                    tooltip="Optional native UPSCALE_MODEL; required only when method is upscale_model.",
                ),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, scale, method, upscale_model=None):
        import comfy.utils

        if method == "upscale_model":
            if upscale_model is None:
                raise ValueError("method=upscale_model requires an UPSCALE_MODEL connection")
            from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

            result = ImageUpscaleWithModel.execute(upscale_model, image)[0]
        else:
            result = image
        height, width = image.shape[1:3]
        target_w, target_h = round(width * scale), round(height * scale)
        resized = comfy.utils.common_upscale(
            result.movedim(-1, 1), target_w, target_h, "lanczos", "disabled"
        ).movedim(1, -1)
        return io.NodeOutput(resized)


class K2RegionExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            K2KreaLoader,
            K2RegionStudio,
            K2RegionalSampler,
            K2FaceDetail,
            K2PostUpscale,
            *BARE_NODE_CLASSES,
        ]


__all__ = ["K2RegionExtension"]
