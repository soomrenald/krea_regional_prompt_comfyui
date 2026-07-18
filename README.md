# K2 Region Studio for ComfyUI

K2 Region Studio is a standalone ComfyUI custom-node package that brings the K2 Region Lab composition engine into the normal ComfyUI graph. It keeps region prompting, strict regional Krea 2 LoRA routing, spatial attention, token emphasis, projector control, character face refinement, and post-upscaling while using ordinary ComfyUI models, workflows, queues, previews, masks, decoders, and output nodes.

The package is backend-neutral. It does not install or pin PyTorch, CUDA, or ROCm; it uses the accelerator stack and memory management of the ComfyUI installation that loads it.

## Install

Copy or clone this folder into the active ComfyUI `custom_nodes` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/soomrenald/krea_regional_prompt_comfyui.git
cd krea_regional_prompt_comfyui
/path/to/ComfyUI/python -m pip install -r requirements.txt
```

For a portable or desktop ComfyUI build, run its embedded Python executable instead. Restart ComfyUI after installation. The node menu will contain **K2 Region Studio**, and the left sidebar will contain **K2 Regions**.

This package targets the current ComfyUI V3 custom-node API (`comfy_api.latest`) and current sidebar API. Krea 2 itself must be supported by the installed ComfyUI version.

## Model layout

Put weights in ComfyUI’s standard directories. Subfolders work.

```text
ComfyUI/models/diffusion_models/krea2_turbo_fp8_scaled.safetensors
ComfyUI/models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors
ComfyUI/models/vae/qwen_image_vae.safetensors
ComfyUI/models/loras/your_krea_lora.safetensors
ComfyUI/models/upscale_models/your_upscaler.pth
```

`K2 Load Krea 2` is a convenience loader for those three Krea components. You can instead connect ComfyUI’s native `UNETLoader`, `CLIPLoader` with type `krea2`, and `VAELoader`. This is useful when other loader, quantization, caching, or device nodes are part of the workflow.

## Basic workflow

1. Add `K2 Load Krea 2`, or use the native ComfyUI model/CLIP/VAE loaders.
2. Add `K2 Region Studio` and connect `MODEL` and `CLIP`.
3. Select that node, open the **K2 Regions** sidebar, and add or drag regions.
4. Connect the patched model, positive/negative conditioning, latent, and region plan to `K2 Regional Sampler`.
5. Decode its `LATENT` with any normal `VAE Decode`, then use any preview/save/postprocessing nodes.

The `K2 Region Studio` outputs remain ordinary ComfyUI types. Its patched `MODEL` can feed ControlNet, guider, sampler, FreeU, compile, or other compatible nodes. Its conditioning, latent, and mask can be branched into normal graph operations. A standard `KSampler` also works, but use `K2 Regional Sampler` when late-step relaxation or LoRA-delta adaptation is enabled because it reports denoising progress to the regional engine.

The optional `K2 Regional Face Detail` node accepts a normal `IMAGE`, `MODEL`, `CLIP`, and `VAE`. The optional `K2 Post Upscale` accepts a normal `UPSCALE_MODEL`, including models loaded with ComfyUI’s `Load Upscale Model` node.

Two graph styles are included. The Studio nodes provide the integrated app-like editor and the complete high-level workflow. The **bare regional LoRA** nodes provide explicit, wireable region masks and LoRA stacks for users who want every routing stage visible in the graph. Both styles use native ComfyUI `MODEL`, `CLIP`, `VAE`, `CONDITIONING`, `LATENT`, `IMAGE`, `MASK`, and `UPSCALE_MODEL` connections wherever ComfyUI already defines them.

## Sidebar editor

The sidebar configuration is serialized into the node’s `region_config` widget, and therefore into the normal ComfyUI workflow JSON and generated-image workflow metadata. No separate project format is required.

- **Regions** edits the global prompt, negative prompt, draggable/resizable pixel boxes, region prompts, face identity prompts, role, priority, and depth order.
- **LoRAs** reads ComfyUI’s LoRA inventory and assigns each instance globally or to any union of named regions, with independent strength and character trigger.
- **Tuning** exposes inside/outside spatial bias, falloff, subject competition/fill, late-step relaxation, LoRA-delta adaptation, Krea projector settings, and face-detail settings.
- **JSON** provides lossless import/export and direct access to token emphasis arrays, custom projector vectors, and future fields.

Regional LoRAs use unfused forward adapters and gate each LoRA’s prediction delta by the compiled text and image token lanes. The base FP8 weights are not rewritten. Prompt attention may feather outside a box for scene coherence; regional LoRA deltas remain strictly zero outside their assigned lanes.

## Any GPU size: tuning and memory

There is no hard-coded GPU size or backend. Resolution, batch size, steps, weight dtype, text-encoder device, VAE placement, model offload, and upscale strategy remain graph or ComfyUI launch settings.

Useful starting points:

| GPU memory | Starting setup |
| --- | --- |
| 8 GB or less | `--lowvram`, FP8 model if supported, 512–768 px canvas, batch 1, CPU text encoder, tiled VAE decode/upscale |
| 10–16 GB | `--lowvram` or `--normalvram`, 768–1024 px canvas, batch 1, tiled decode if needed |
| 20–24 GB | `--normalvram`, 1024 px or larger, GPU VAE where practical |
| 32 GB+ | Increase canvas/batch gradually; keep a reserve for attention and decoded images |

Examples:

```bash
python main.py --lowvram --reserve-vram 1.5
python main.py --normalvram --reserve-vram 3
python main.py --cpu-vae --lowvram
```

The right values depend on model precision, backend kernels, image dimensions, batch size, installed graph nodes, and concurrent applications. Reduce canvas area before reducing steps when an attention allocation does not fit. ComfyUI’s tiled VAE and tiled upscale nodes are preferable on small cards. Do not install a second Torch build into this custom node: install the correct CUDA, ROCm, Intel, Apple, or CPU build as part of ComfyUI itself.

## Face detail

Face detail is disabled by default. It needs `onnxruntime` and a compatible NanoDet `face_det.onnx`. By default, the node discovers the copy shipped by FantasyPortrait at:

```text
ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/fantasyportrait/models/face_det.onnx
```

You can supply another path in the `detector_path` widget. The pass detects faces on CPU, assigns each detection to a nearby subject region, VAE-encodes a padded crop, samples with only that region’s character LoRAs, and composites a feathered result.

## Nodes

### Studio and app nodes

- `K2 Load Krea 2`: native Krea 2 `MODEL`, `CLIP`, and `VAE` loader.
- `K2 Region Studio`: compiles the region project and returns standard graph objects plus a `K2_REGION_PLAN` control object and reports.
- `K2 Regional Sampler`: standard KSampler controls plus regional denoising progress updates.
- `K2 Regional Face Detail`: optional assigned-face crop refinement.
- `K2 Post Upscale`: exact Lanczos resizing or a normal ComfyUI neural upscale model.

### Bare graph nodes

- `K2 BBox To Regional Mask`: accepts detector/KJ bounding boxes and produces a normal `MASK`, a reusable region object, and a debug image.
- `K2 Regional Character LoRA`: binds a native positive/negative conditioning pair and a ComfyUI LoRA filename to one region, with strength and denoising-range controls.
- `K2 Regional LoRA Stack 3`: combines up to three region-bound LoRAs with explicit overlap behavior.
- `K2 Regional Layer LoRA Apply`: applies a regional stack to a native `MODEL` branch with selectable injection targets.
- `K2 Regional Attention LoRA Sampler`: exposes the full sampler, regional execution, outside-pinning, and debug controls.
- `K2 Regional Decode Composite`: decodes and safely composites regional/base latent results through a native `VAE`.

The bare workflow is intentionally composable with native loaders, text encoders, detectors, samplers, VAE nodes, preview/save nodes, and third-party nodes that use standard ComfyUI types. The sidebar-driven `K2 Region Studio` is the app version: it provides the drawable labeled-region canvas and the same prompts, prompt emphases, regional LoRA assignments, spatial controls, projector controls, face-detail settings, and JSON import/export stored inside the workflow.

## Compatibility notes

- The spatial override occupies ComfyUI’s `optimized_attention_override` hook. Do not put another node that claims the same hook on the same model branch; branch before applying either override.
- Krea regional LoRAs are validated against the active Krea model namespace. A LoRA for another architecture is rejected instead of being silently applied to zero layers.
- Global negative conditioning is output normally. Region-local negative text is preserved in workflow configuration, but the current Krea Turbo CFG-free path has no separate regional negative branch.
- Cancel, queue, history, workflow embedding, output naming, model unloading, previews, and resource monitoring are provided by ComfyUI.

An editable starter graph is in [`workflows/k2_region_starter.json`](workflows/k2_region_starter.json). Select your locally installed model filenames after loading it.

## Development checks

```bash
python -m pytest -q
node --check web/k2_region_studio.js
python -m py_compile __init__.py k2_region_comfy/*.py k2_region_core/**/*.py
```
