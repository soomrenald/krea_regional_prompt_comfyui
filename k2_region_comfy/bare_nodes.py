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
                io.Int.Input("width", default=1024, min=16, max=16384, step=8),
                io.Int.Input("height", default=1024, min=16, max=16384, step=8),
                io.Combo.Input("bbox_format", options=["xywh", "xyxy"], default="xywh"),
                io.Int.Input("bbox_index", default=0, min=0, max=4096),
                io.Int.Input("grow_px", default=0, min=-4096, max=4096),
                io.Int.Input("feather_px", default=32, min=0, max=2048),
                io.Boolean.Input("snap_to_krea_token_grid", default=True),
                io.Combo.Input(
                    "batch_mode", options=["single", "repeat", "per_batch"], default="repeat"
                ),
                BoundingBox.Input("bboxes", optional=True),
                KJBoundingBox.Input("kj_bboxes", optional=True),
                io.Latent.Input("latent", optional=True),
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
                K2Region.Input("region"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Combo.Input("lora_name", options=legacy._lora_names()),
                io.Float.Input("lora_strength", default=1.0, min=-100.0, max=100.0, step=0.01),
                io.Float.Input("delta_strength", default=1.0, min=-100.0, max=100.0, step=0.01),
                io.Float.Input("start_percent", default=0.10, min=0.0, max=1.0, step=0.01),
                io.Float.Input("end_percent", default=0.95, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("enabled", default=True),
                io.Boolean.Input("attention_only_filter", default=True),
                io.Boolean.Input("ignore_text_encoder_lora", default=True),
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
                K2RegionalLora.Input("regional_lora_1"),
                io.Combo.Input(
                    "overlap_mode",
                    options=["normalize", "priority_1", "priority_3", "add_clamped"],
                    default="normalize",
                ),
                K2RegionalLora.Input("regional_lora_2", optional=True),
                K2RegionalLora.Input("regional_lora_3", optional=True),
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
                io.Model.Input("model"),
                K2RegionalLoraStack.Input("regional_lora_stack"),
                io.Combo.Input(
                    "layer_injection_targets",
                    options=["attn_out_mlp", "attention_only", "all_matched_linears"],
                    default="attn_out_mlp",
                ),
                io.Float.Input("outside_strength", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("text_token_strength", default=0.0, min=0.0, max=2.0, step=0.01),
                io.Boolean.Input("debug_logging", default=False),
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
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent_image"),
                K2RegionalLoraStack.Input("regional_lora_stack"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("cfg", default=4.0, min=0.0, max=100.0, step=0.1),
                io.Combo.Input("sampler_name", options=legacy._sampler_names()),
                io.Combo.Input("scheduler", options=legacy._scheduler_names()),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Combo.Input(
                    "execution_mode",
                    options=["auto", "strict_adapter", "layer_injection"],
                    default="auto",
                ),
                io.Combo.Input(
                    "layer_injection_targets",
                    options=["attn_out_mlp", "attention_only", "all_matched_linears"],
                    default="attn_out_mlp",
                ),
                io.Float.Input("layer_outside_strength", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input(
                    "layer_text_token_strength", default=0.0, min=0.0, max=2.0, step=0.01
                ),
                io.Boolean.Input("pin_outside_regions", default=True),
                io.Boolean.Input("final_latent_pin", default=True),
                io.Boolean.Input("post_decode_safe_mode", default=True),
                io.Boolean.Input("debug_return_base_latent", default=True),
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
                io.Vae.Input("vae"),
                io.Latent.Input("regional_samples"),
                io.Latent.Input("base_samples"),
                io.Mask.Input("union_mask"),
                io.Int.Input("feather_px", default=32, min=0, max=2048),
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
