from __future__ import annotations

from comfy_api.latest import io

from ..k2_region_bare import legacy


CATEGORY = "K2 Region Studio/bare regional LoRA"
K2Region = io.Custom("K2REGION")
K2RegionalLora = io.Custom("K2REGIONAL_LORA")
K2RegionalLoraStack = io.Custom("K2REGIONAL_LORA_STACK")
BoundingBox = io.Custom("BOUNDING_BOX")
KJBoundingBox = io.Custom("BBOX")


class K2BBoxToRegionalMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="K2BBoxToRegionalMask",
            display_name="K2 BBox To Regional Mask",
            category=CATEGORY,
            description="Converts detector/KJ bounding boxes to Krea pixel, latent, and token masks.",
            inputs=[
                io.Int.Input(
                    "width",
                    default=1024,
                    min=16,
                    max=16384,
                    step=8,
                    tooltip="Image width used when no latent supplies the dimensions.",
                ),
                io.Int.Input(
                    "height",
                    default=1024,
                    min=16,
                    max=16384,
                    step=8,
                    tooltip="Image height used when no latent supplies the dimensions.",
                ),
                io.Combo.Input(
                    "bbox_format",
                    options=["xywh", "xyxy"],
                    default="xywh",
                    tooltip="Interpret boxes as x/y/width/height or as x0/y0/x1/y1 coordinates.",
                ),
                io.Int.Input(
                    "bbox_index",
                    default=0,
                    min=0,
                    max=4096,
                    tooltip="Zero-based box to select when the detector returns multiple boxes.",
                ),
                io.Int.Input(
                    "grow_px",
                    default=0,
                    min=-4096,
                    max=4096,
                    tooltip="Expand the selected box by this many pixels on every side; negative values shrink it.",
                ),
                io.Int.Input(
                    "feather_px",
                    default=32,
                    min=0,
                    max=2048,
                    tooltip="Width of the soft transition inside the box edge for pixel and latent blending.",
                ),
                io.Boolean.Input(
                    "snap_to_krea_token_grid",
                    default=True,
                    tooltip="Align box edges to Krea's 16-pixel image-token grid for stable token routing.",
                ),
                io.Combo.Input(
                    "batch_mode",
                    options=["single", "repeat", "per_batch"],
                    default="repeat",
                    tooltip="single keeps one mask, repeat broadcasts it, and per_batch creates a mask batch matching the latent.",
                ),
                BoundingBox.Input(
                    "bboxes",
                    optional=True,
                    tooltip="Optional standard BOUNDING_BOX input; preferred when connected.",
                ),
                KJBoundingBox.Input(
                    "kj_bboxes",
                    optional=True,
                    tooltip="Optional KJNodes-style BBOX input used when bboxes is not connected.",
                ),
                io.Latent.Input(
                    "latent",
                    optional=True,
                    tooltip="Optional latent used to infer image dimensions and batch size.",
                ),
            ],
            outputs=[
                io.Mask.Output(display_name="region_mask"),
                K2Region.Output(display_name="region"),
                io.Image.Output(display_name="debug_bbox_image"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*legacy.K2BBoxToRegionalMask().build(**kwargs))


class K2RegionalCharacterLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="K2RegionalCharacterLoRA",
            display_name="K2 Regional Character LoRA",
            category=CATEGORY,
            inputs=[
                K2Region.Input("region", tooltip="Region object produced by K2 BBox To Regional Mask."),
                io.Conditioning.Input(
                    "positive", tooltip="Positive conditioning used when evaluating this regional LoRA branch."
                ),
                io.Conditioning.Input(
                    "negative", tooltip="Negative conditioning used when evaluating this regional LoRA branch."
                ),
                io.Combo.Input(
                    "lora_name",
                    options=legacy._lora_names(),
                    tooltip="LoRA file from ComfyUI/models/loras to bind to this region.",
                ),
                io.Float.Input(
                    "lora_strength",
                    default=1.0,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                    tooltip="Strength used when loading the LoRA branch; negative values invert its learned delta.",
                ),
                io.Float.Input(
                    "delta_strength",
                    default=1.0,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                    tooltip="Additional multiplier applied to the regional prediction delta after comparing it with the base branch.",
                ),
                io.Float.Input(
                    "start_percent",
                    default=0.10,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Earliest fraction of the denoising schedule at which this LoRA is active.",
                ),
                io.Float.Input(
                    "end_percent",
                    default=0.95,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Latest fraction of the denoising schedule at which this LoRA is active.",
                ),
                io.Boolean.Input(
                    "enabled", default=True, tooltip="Disable to keep the binding in the graph without applying it."
                ),
                io.Boolean.Input(
                    "attention_only_filter",
                    default=True,
                    tooltip="Load only Krea attention LoRA keys in strict-adapter mode.",
                ),
                io.Boolean.Input(
                    "ignore_text_encoder_lora",
                    default=True,
                    tooltip="Discard text-encoder LoRA keys so regional routing affects only the diffusion model.",
                ),
            ],
            outputs=[K2RegionalLora.Output(display_name="regional_lora")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*legacy.K2RegionalCharacterLoRA().bind(**kwargs))


class K2RegionalLoRAStack3(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="K2RegionalLoRAStack3",
            display_name="K2 Regional LoRA Stack 3",
            category=CATEGORY,
            inputs=[
                K2RegionalLora.Input(
                    "regional_lora_1", tooltip="First required region-bound LoRA; first priority in priority_1 mode."
                ),
                io.Combo.Input(
                    "overlap_mode",
                    options=["normalize", "priority_1", "priority_3", "add_clamped"],
                    default="normalize",
                    tooltip="How overlapping deltas combine: average, first-wins, third/last-wins, or summed then clamped to [-1,1].",
                ),
                K2RegionalLora.Input(
                    "regional_lora_2", optional=True, tooltip="Optional second region-bound LoRA."
                ),
                K2RegionalLora.Input(
                    "regional_lora_3",
                    optional=True,
                    tooltip="Optional third region-bound LoRA; first priority in priority_3 mode.",
                ),
            ],
            outputs=[K2RegionalLoraStack.Output(display_name="regional_lora_stack")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*legacy.K2RegionalLoRAStack3().stack(**kwargs))


class K2RegionalLayerLoRAApply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="K2RegionalLayerLoRAApply",
            display_name="K2 Regional Layer LoRA Apply",
            category=CATEGORY,
            inputs=[
                io.Model.Input("model", tooltip="Base Krea MODEL to clone and patch with regional layer hooks."),
                K2RegionalLoraStack.Input(
                    "regional_lora_stack", tooltip="Stack of region-bound LoRAs to inject into the model."
                ),
                io.Combo.Input(
                    "layer_injection_targets",
                    options=["attn_out_mlp", "attention_only", "all_matched_linears"],
                    default="attn_out_mlp",
                    tooltip="Layer policy: output/MLP writeback only, attention projections only, or every matched linear layer.",
                ),
                io.Float.Input(
                    "outside_strength",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Fraction of each regional LoRA permitted outside its image region; zero is strict isolation.",
                ),
                io.Float.Input(
                    "text_token_strength",
                    default=0.0,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="LoRA mask strength on text tokens when a layer contains text and image token lanes.",
                ),
                io.Boolean.Input(
                    "debug_logging",
                    default=False,
                    tooltip="Include verbose matched/skipped layer details in the report and server log.",
                ),
            ],
            outputs=[io.Model.Output(), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*legacy.K2RegionalLayerLoRAApply().apply(**kwargs))


class K2RegionalAttentionLoRASampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="K2RegionalAttentionLoRASampler",
            display_name="K2 Regional Attention LoRA Sampler",
            category=CATEGORY,
            inputs=[
                io.Model.Input("model", tooltip="Base Krea MODEL used for the base and regional branches."),
                io.Conditioning.Input("positive", tooltip="Base positive conditioning."),
                io.Conditioning.Input("negative", tooltip="Base negative conditioning."),
                io.Latent.Input("latent_image", tooltip="Starting latent to denoise."),
                K2RegionalLoraStack.Input(
                    "regional_lora_stack", tooltip="Region-bound LoRAs and their overlap policy."
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip="Random-noise seed shared by the base and regional passes.",
                ),
                io.Int.Input(
                    "steps", default=20, min=1, max=10000, tooltip="Number of denoising iterations."
                ),
                io.Float.Input(
                    "cfg",
                    default=4.0,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    tooltip="Classifier-free guidance scale used by all sampling branches.",
                ),
                io.Combo.Input(
                    "sampler_name",
                    options=legacy._sampler_names(),
                    default="euler",
                    tooltip="Denoising algorithm used by the base and fallback regional passes.",
                ),
                io.Combo.Input(
                    "scheduler",
                    options=legacy._scheduler_names(),
                    default="simple",
                    tooltip="Sigma/noise schedule used by the base and fallback regional passes.",
                ),
                io.Float.Input(
                    "denoise",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Fraction of the denoising schedule to run; lower values preserve more of the input latent.",
                ),
                io.Combo.Input(
                    "execution_mode",
                    options=["auto", "strict_adapter", "layer_injection"],
                    default="auto",
                    tooltip="auto uses a model adapter when available then falls back; strict_adapter forbids fallback; layer_injection forces hooks.",
                ),
                io.Combo.Input(
                    "layer_injection_targets",
                    options=["attn_out_mlp", "attention_only", "all_matched_linears"],
                    default="attn_out_mlp",
                    tooltip="Layer policy used only by the layer-injection execution path.",
                ),
                io.Float.Input(
                    "layer_outside_strength",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Layer-injection LoRA strength allowed outside assigned regions; zero keeps strict isolation.",
                ),
                io.Float.Input(
                    "layer_text_token_strength",
                    default=0.0,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="Layer-injection mask strength applied to text-token lanes.",
                ),
                io.Boolean.Input(
                    "pin_outside_regions",
                    default=True,
                    tooltip="In adapter mode, restore the exact base trajectory outside the union of active regions after every step.",
                ),
                io.Boolean.Input(
                    "final_latent_pin",
                    default=True,
                    tooltip="In layer-injection mode, replace the final latent outside all regions with the base latent.",
                ),
                io.Boolean.Input(
                    "post_decode_safe_mode",
                    default=True,
                    tooltip="Compatibility flag retained for workflows; use K2 Regional Decode Composite for pixel-safe compositing.",
                ),
                io.Boolean.Input(
                    "debug_return_base_latent",
                    default=True,
                    tooltip="Return the separately sampled base latent for inspection and safe decode compositing.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="samples"),
                io.Latent.Output(display_name="base_samples"),
                io.Mask.Output(display_name="union_mask"),
                io.String.Output(display_name="debug_info"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*legacy.K2RegionalAttentionLoRASampler().sample(**kwargs))


class K2RegionalDecodeComposite(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="K2RegionalDecodeComposite",
            display_name="K2 Regional Decode Composite",
            category=CATEGORY,
            inputs=[
                io.Vae.Input("vae", tooltip="VAE used to decode both latent branches."),
                io.Latent.Input(
                    "regional_samples", tooltip="Regional latent output from the bare regional sampler."
                ),
                io.Latent.Input(
                    "base_samples", tooltip="Base latent output from the same bare regional sampler."
                ),
                io.Mask.Input(
                    "union_mask", tooltip="Pixel-space union mask identifying all regional edit areas."
                ),
                io.Int.Input(
                    "feather_px",
                    default=32,
                    min=0,
                    max=2048,
                    tooltip="Pixel radius used to soften the composite boundary between regional and base decodes.",
                ),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*legacy.K2RegionalDecodeComposite().composite(**kwargs))


BARE_NODE_CLASSES = [
    K2BBoxToRegionalMask,
    K2RegionalCharacterLoRA,
    K2RegionalLoRAStack3,
    K2RegionalLayerLoRAApply,
    K2RegionalAttentionLoRASampler,
    K2RegionalDecodeComposite,
]


__all__ = ["BARE_NODE_CLASSES"]
