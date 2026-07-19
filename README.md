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
- **Tuning** exposes inside/outside spatial bias, falloff, subject competition/fill, strict regional LoRA isolation, late-step relaxation, LoRA-delta adaptation, Krea projector settings, and face-detail settings.
- **JSON** provides lossless import/export and direct access to token emphasis arrays, custom projector vectors, and future fields.

Regional LoRAs use unfused forward adapters and gate each LoRA’s prediction delta by the compiled text and image token lanes. The base FP8 weights are not rewritten. With **Strict regional LoRA isolation** enabled (the default), this uses the same v6/v3 mechanism as the PySide application: subject-owned text keys are private to their subject in the text-refiner and main streams; image tokens can consume a subject clause only inside that subject’s hard box; and standard regional LoRAs omit main-stream key/value adapter targets whose modified values could otherwise be consumed by queries outside the region. Image-to-image attention remains intact, so this is one denoising pass and not crop compositing or a second sampler pass. Soft inside/outside bias still controls placement among permitted pairs, while the hard ownership partition prevents another subject from consuming LoRA-modified prompt tokens.

Disabling strict isolation restores soft-bias-only prompt routing and permits all compatible standard-LoRA targets. That mode is retained only for comparison and can allow subject attributes or LoRA capabilities to cross regions. The sampler report records `krea-unified-spatial-attention-v6`, `krea-regional-lora-delta-gating-v3`, skipped non-local key/value targets, and whether strict isolation was active. A trigger phrase is inserted automatically only for **Character identity** routing; a trigger saved on a **Standard regional** LoRA is metadata and does not itself provide isolation.

## Any GPU size: tuning and memory

There is no hard-coded GPU size or backend. Resolution, batch size, steps, weight dtype, text-encoder device, VAE placement, model offload, and upscale strategy remain graph or ComfyUI launch settings.

After each sidebar/Studio sampling run, routed LoRA weights are moved back to system RAM and the per-run attention, mask, and statistics caches are cleared. Cleanup also runs when sampling is interrupted or raises an OOM. This prevents ComfyUI's cached patched-model outputs from retaining a separate GPU-resident LoRA copy for every edited `region_config`. The next run re-uploads the active LoRA weights, which adds a small transfer cost but avoids progressive VRAM growth across sidebar edits.

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

## Complete control reference

ComfyUI shows a short version of these descriptions when you hover over a node input. The K2 Regions sidebar also shows hover notes on its tabs, labels, canvas, and buttons.

### K2 Load Krea 2

| Control | Default | Description |
| --- | --- | --- |
| `diffusion_model` | First available file | Krea 2 diffusion model from `models/diffusion_models`. The loader does not download a model. |
| `text_encoder` | First available file | Krea-compatible Qwen encoder from `models/text_encoders`. A generic non-Krea encoder will not produce the expected conditioning layout. |
| `vae` | First available VAE | VAE used for latent encoding and image decoding. |
| `weight_dtype` | `default` | `default` follows the model file and ComfyUI runtime. FP8 modes reduce model memory when the installed device and Torch build support them; `fp8_e4m3fn_fast` also enables ComfyUI FP8 optimizations. |
| `text_encoder_device` | `default` | `default` lets ComfyUI place/offload Qwen. `cpu` keeps its load and offload devices on CPU, saving VRAM at the cost of slower prompt encoding. |

Native `UNETLoader`, `CLIPLoader` with type `krea2`, and `VAELoader` may replace this convenience node.

### K2 Region Studio graph node

| Control | Default | Description |
| --- | --- | --- |
| `model` | Connected input | Native Krea 2 `MODEL`. The node clones and patches it; the incoming branch is not mutated. |
| `clip` | Connected input | Krea/Qwen `CLIP` used to tokenize the compiled prompt and produce positive and negative conditioning. |
| `region_config` | Empty Studio project | Portable JSON written by the sidebar and embedded in workflow metadata. The frontend hides this widget because the sidebar and JSON pane are safer editors. |
| `width` | `1024` | Output width and horizontal coordinate system for all pixel-space region boxes. Values are aligned by the backend as required. |
| `height` | `1024` | Output height and vertical coordinate system for all pixel-space region boxes. |
| `batch_size` | `1` | Number of empty latent samples created together. Region and LoRA routing is broadcast across the batch. |

The outputs are the patched model, original clip, compiled positive/negative conditioning, empty latent, normal union `MASK`, runtime `K2_REGION_PLAN`, compiled prompt text, and a JSON report.

### Sidebar: Regions

| Control | Description |
| --- | --- |
| `Global prompt` | Scene-wide positive description. Enabled regional clauses and spatial relationship instructions are compiled after it. |
| `Global negative` | Scene-wide negative text used for the main conditioning and face-detail passes. Region-local negatives are stored but Krea Turbo currently has no separate regional negative branch. |
| `+ Region` | Adds a labeled box with a unique internal ID and a descending default priority (`100`, `99`, and so on). |
| Drawing canvas | Drag inside a box to move it. Drag the square at its lower-right corner to resize it. Coordinates are clamped to the node width and height. |
| Up/down arrows | Change sidebar/list order. Equal-priority regions retain this order during compilation; explicit priorities otherwise determine compiled order. |
| `Name` | Human-readable label used in generated spatial instructions, relationship text, face assignments, and reports. Names do not replace stable internal region IDs. |
| `Enabled` | Includes or excludes the region from prompt compilation, spatial attention, and face assignment without deleting it. Remove the region from non-global LoRA assignments before disabling it; dangling regional assignments are rejected. |
| `Prompt` | Description assigned to the box. Empty regional prompts are excluded from the active regional plan. |
| `Identity prompt` | Face-specific description appended to the regional clause, protected by projector identity protection, and preferred by Face Detail. |
| `Negative` | Region-local negative text retained in workflow JSON for compatibility and future regional-negative paths. It is not a separate Krea Turbo conditioning branch today. |
| `Priority` | Higher numbers compile first. Higher-priority regions also get the first chance to claim an ambiguous detected face. Priority does not increase prompt/LoRA strength and is not an image-compositing z-index. Equal values preserve list order. |
| `Role` | `subject` enables subject competition/fill and a full outside penalty. `background` uses a softer outside penalty. `auto` treats boxes covering at least 70% of canvas width as background and smaller boxes as subjects. |

### Sidebar: LoRAs

| Control | Default | Description |
| --- | --- | --- |
| `+ LoRA` | — | Adds an assignment using ComfyUI's current `models/loras` inventory. |
| `Model` | First available LoRA | LoRA file to validate against and apply to the active Krea model. Incompatible namespaces are reported instead of silently doing nothing. |
| `Strength` | `1.0` | Model-delta multiplier from `-4` to `4`. Zero disables the assignment; negative values invert its learned delta. |
| `Routing` | `standard` | With Global scope off, `standard` gates text-fusion deltas to assigned clauses. Off-box image tokens cannot attend those clauses, and global or other-region text cannot read image keys in the selected boxes. Image-to-image attention remains continuous to avoid box seams. Compatible main-stream deltas use the selected boxes; main-stream attention key/value targets remain omitted. `character_identity` requires a selected region and trigger phrase and adds an identity anchor on top of the same isolation. |
| `Global scope` | Enabled | Applies the LoRA to all text and image lanes. Disable it to expose the region checklist and use strict regional routing. |
| `Regions` | None | Union of named regions receiving a non-global LoRA. The regional clause and LoRA text delta can condition only image tokens intersecting these boxes; compatible main-stream deltas use the same strict boxes. |
| `Trigger phrase` | Empty | Character activation phrase used by `character_identity` routing. It must be non-empty in that mode and should match the phrase learned during LoRA training. |

### Sidebar: Emphasis

| Control | Default | Description |
| --- | --- | --- |
| `Scope` | Global prompt | Limits exact-phrase matching to the global prompt or one enabled region's prompt. |
| `Exact phrase` | Empty | Case-sensitive text to locate in the selected scope. Compilation fails with a useful error if the phrase is absent. |
| `Strength` | `0.5` | Additional text-to-image spatial attention bias from `0` to `2`. It does not rewrite Qwen token weights or LoRA strength. |
| `Occurrence` | `0` | Zero-based matching index when the phrase occurs more than once: `0` is first, `1` second, `2` third, and so forth. |

### Sidebar: Spatial attention tuning

| Control | Default | Description |
| --- | --- | --- |
| `Enabled` | On | Installs the Krea text partition and spatial-attention router. It may be disabled only when no regional LoRA is active; otherwise compilation fails rather than allowing a text-fusion LoRA to leak through shared conditioning. |
| `Inside strength` | `1.0` | Positive attention-logit bias inside each region. Larger values bind regional text more strongly to its image-token field. |
| `Outside penalty` | `1.0` | Increases center-to-edge contrast inside a subject box; subject text is hard-blocked outside its box. Background regions use one quarter of this penalty and can still feather outside their boxes. |
| `Edge falloff (px)` | `128` | Distance beyond a background box over which its soft attention field fades. Subject prompt and text-fusion LoRA conditioning remains hard-confined to image tokens intersecting the subject box. |
| `Late-step scale` | `0.35` | Fraction of spatial strength retained at the final denoising step. Relaxation begins after 55% progress and interpolates toward this value. Requires `region_plan` on K2 Regional Sampler. |
| `Subject competition` | On | In overlaps between two or more subject regions, assigns soft ownership proportional to squared regional field strength so all subjects do not fully claim the same token. |
| `Fill unclaimed subject space` | On | Keeps a stronger field toward subject-box edges, reducing weak/unclaimed space inside a subject box. |
| `LoRA-delta adaptation` | Off | Uses observed regional LoRA-delta energy to adjust each region's spatial scale during sampling. Requires K2 Regional Sampler progress updates. |
| `LoRA adaptation gain` | `0.35` | Maximum correction gain used by LoRA-delta adaptation. Zero effectively disables the correction while leaving measurement enabled; larger values rebalance more aggressively. |

### Sidebar: Projector control

| Control | Default | Description |
| --- | --- | --- |
| `Enabled` | Off | Applies a delta to Krea's 12-value `txtfusion.projector.weight`. No base checkpoint weights are overwritten. |
| `Preset` | `filter_bypass2` | Selects a built-in projector vector. `custom` reads the 12 numbers in `projector.values`, edited in the JSON pane. |
| `Multiplier` | `1.0` | Signed scale applied to all preset values. Zero produces no projector effect; negative values reverse the vector. |
| `Identity protection` | `1.0` | Reduces projector changes on face-identity token spans. `0` applies the projector normally; `1` completely preserves the baseline projector behavior for those tokens. |

### Sidebar: Face detail tuning

These settings are stored in the region plan and used by the separate `K2 Regional Face Detail` node.

| Control | Default | Description |
| --- | --- | --- |
| `Enabled` | Off | Enables detector-driven face refinement. If off, the Face Detail node passes the image through unchanged. |
| `Steps` | `8` | Denoising iterations for each face crop. |
| `Denoise` | `0.15` | Crop denoise fraction. Low values preserve facial structure; higher values permit larger changes. |
| `Crop size` | `512` | Square working resolution used to encode, sample, and decode every face crop. |
| `Padding` | `2.0` | Multiplier expanding the detected face before constructing a square crop. More padding includes hair and surrounding context. |
| `Feather` | `0.12` | Fractional border width used to soften the crop mask during compositing. |
| `Blend` | `0.5` | Opacity of the refined crop over the source: `0` keeps the original and `1` uses the full refinement. |
| `LoRA scale` | `0.5` | Additional multiplier applied to each region-assigned LoRA during its face crop pass. |
| `Detector threshold` | `0.4` | Minimum NanoDet confidence from `0` to `1`. Higher values reject uncertain detections. |

### Sidebar: JSON

`Copy` places the complete portable configuration on the clipboard. `Apply JSON` parses and stores edited JSON in the selected Studio node. This pane also exposes advanced fields: the 12-number `projector.values` vector and future versioned fields. Invalid JSON or a configuration version newer than the package is rejected rather than partially applied.

### K2 Regional Sampler

| Control | Default | Description |
| --- | --- | --- |
| `model` | Connected input | Patched `MODEL` from K2 Region Studio. |
| `positive` / `negative` | Connected inputs | Compiled positive and global negative conditioning from Studio. |
| `latent` | Connected input | Starting latent. Studio supplies an empty latent, but any compatible native latent may be used. |
| `seed` | `0` | Random-noise seed. The same seed and settings reproduce the same starting noise. |
| `steps` | `20` | Denoising iterations. Krea Turbo commonly uses a much smaller count, such as the starter workflow's `8`. |
| `cfg` | `1.0` | Classifier-free guidance. Krea Turbo is designed around `1.0`; increasing CFG is not automatically an improvement. |
| `sampler_name` | `euler` | ComfyUI sampling algorithm. Euler is the package default for Krea Turbo. |
| `scheduler` | `simple` | ComfyUI sigma schedule. Simple is the package default for Krea Turbo. |
| `denoise` | `1.0` | Fraction of the schedule used. `1.0` is normal text-to-image; lower values preserve more of a supplied latent. |
| `region_plan` | Optional | Enables per-step late relaxation and LoRA-delta adaptation updates and produces a final regional report. Without it, the node behaves like a normal compatible sampler. |

### K2 Regional Face Detail

| Control | Description |
| --- | --- |
| `image` | Decoded image batch to inspect and refine. |
| `model`, `clip`, `vae` | Native Krea components used to sample each detected crop. |
| `region_plan` | Studio runtime plan containing regions, priorities, LoRA assignments, prompts, and the face-detail tuning above. |
| `seed` | Base seed. Batch and face indices receive deterministic offsets so separate faces do not share identical noise. |
| `sampler_name` / `scheduler` | Sampler and noise schedule for every crop pass; defaults are `euler` and `simple`. |
| `detector_path` | Optional path to a compatible NanoDet `face_det.onnx`. Blank enables the documented FantasyPortrait auto-discovery path. |

Only non-background regions with an active prompt, assigned non-global LoRA, and a nearby detected face become refinement targets. Higher-priority regions claim ambiguous detections first.

### K2 Post Upscale

| Control | Default | Description |
| --- | --- | --- |
| `image` | Connected input | Image batch to enlarge. |
| `scale` | `2.0` | Final width and height multiplier, from `1` to `8`. |
| `method` | `lanczos` | `lanczos` performs a deterministic high-quality resize. `upscale_model` first runs the connected neural model and then resizes to the exact requested dimensions. |
| `upscale_model` | Optional | Native ComfyUI `UPSCALE_MODEL`; required only for `upscale_model` mode. Use ComfyUI's Load Upscale Model node as its source. |

### Bare: K2 BBox To Regional Mask

| Control | Default | Description |
| --- | --- | --- |
| `width` / `height` | `1024` | Image dimensions used when no latent is connected. A connected latent overrides both using the Krea VAE scale. |
| `bbox_format` | `xywh` | `xywh` means x/y/width/height. `xyxy` means left/top/right/bottom. Normalized `0..1` coordinates are also accepted. |
| `bbox_index` | `0` | Zero-based box selected from a detector result. Out-of-range indices clamp to the available list. |
| `grow_px` | `0` | Expands every side by this many pixels; negative values shrink the box. |
| `feather_px` | `32` | Softens the pixel mask inward from its edge before latent and token masks are derived. |
| `snap_to_krea_token_grid` | On | Expands box edges to Krea's 16-pixel image-token grid. |
| `batch_mode` | `repeat` | `single` keeps a one-mask object, `repeat` broadcasts that mask to consumers, and `per_batch` creates masks matching latent batch size. |
| `bboxes` | Optional | Standard `BOUNDING_BOX` input. It takes precedence over `kj_bboxes`. |
| `kj_bboxes` | Optional | KJNodes-style `BBOX` fallback input. |
| `latent` | Optional | Supplies dimensions and batch size. |

### Bare: K2 Regional Character LoRA

| Control | Default | Description |
| --- | --- | --- |
| `region` | Connected input | Reusable region/mask object from the BBox node. |
| `positive` / `negative` | Connected inputs | Conditionings used by this LoRA's regional prediction branch. |
| `lora_name` | First available LoRA | LoRA file bound to this region. |
| `lora_strength` | `1.0` | Strength used when loading the branch. Negative values invert it. |
| `delta_strength` | `1.0` | Additional multiplier on `(LoRA prediction - base prediction)` before masking and overlap resolution. |
| `start_percent` / `end_percent` | `0.10` / `0.95` | Inclusive active interval in adapter-mode denoising progress. |
| `enabled` | On | Keeps the binding in the graph while allowing it to be bypassed. |
| `attention_only_filter` | On | In strict-adapter mode, removes non-Krea-attention LoRA keys. |
| `ignore_text_encoder_lora` | On | Removes CLIP/Qwen LoRA keys so the regional branch patches only the diffusion model. |

### Bare: K2 Regional LoRA Stack 3

| Control | Description |
| --- | --- |
| `regional_lora_1..3` | One required and two optional region-bound LoRAs. Their socket order is used by priority overlap modes. |
| `overlap_mode=normalize` | Averages active masked deltas in overlaps. |
| `overlap_mode=priority_1` | First connected LoRA claims overlapping locations before later inputs. |
| `overlap_mode=priority_3` | Reverses socket order, so the third/last connected LoRA claims overlaps first. |
| `overlap_mode=add_clamped` | Sums all masked deltas and clamps the combined delta to `[-1, 1]`. |

### Bare: K2 Regional Layer LoRA Apply

| Control | Default | Description |
| --- | --- | --- |
| `model` | Connected input | Base Krea model cloned for regional layer hooks. |
| `regional_lora_stack` | Connected input | LoRAs and masks to inject. |
| `layer_injection_targets` | `attn_out_mlp` | `attn_out_mlp` targets attention output and MLP writeback layers; `attention_only` targets Krea attention projections; `all_matched_linears` uses every compatible matched linear. Wider policies can increase effect and compatibility risk. |
| `outside_strength` | `0` | Fraction of a regional LoRA allowed outside its image mask. Zero is strict; one makes image-lane application global. |
| `text_token_strength` | `0` | Mask strength on text-token lanes in mixed text/image sequences. |
| `debug_logging` | Off | Adds verbose matching and skipped-layer information to the report/server log. |

### Bare: K2 Regional Attention LoRA Sampler

The standard `model`, conditioning, latent, seed, steps, CFG, sampler, scheduler, and denoise controls behave like ComfyUI sampling controls. Bare CFG defaults to `4.0`; sampler and scheduler default to `euler` and `simple`.

| Control | Default | Description |
| --- | --- | --- |
| `regional_lora_stack` | Connected input | Regional branches, masks, schedules, and overlap policy. |
| `execution_mode` | `auto` | `auto` uses `k2_regional_velocity_predictor` when the model exposes it and otherwise uses layer injection. `strict_adapter` returns base samples rather than falling back. `layer_injection` always uses cloned layer hooks. |
| `layer_injection_targets` | `attn_out_mlp` | Target policy used only by layer-injection execution. |
| `layer_outside_strength` | `0` | LoRA fraction permitted outside region masks in layer-injection mode. |
| `layer_text_token_strength` | `0` | LoRA mask strength on text tokens in layer-injection mode. |
| `pin_outside_regions` | On | Adapter mode restores the exact base trajectory outside the regional union after every denoising step. |
| `final_latent_pin` | On | Layer-injection mode replaces the final latent outside the union with the base latent. |
| `post_decode_safe_mode` | On | Workflow-compatibility flag. Pixel-safe isolation is performed by the separate Decode Composite node. |
| `debug_return_base_latent` | On | Returns a copied base latent payload for debugging/compositing. The base output socket remains populated for compatibility. |

Outputs are regional samples, base samples, the union mask, and a text debug report.

### Bare: K2 Regional Decode Composite

| Control | Default | Description |
| --- | --- | --- |
| `vae` | Connected input | Decodes both latent branches using the same VAE. |
| `regional_samples` | Connected input | Regional latent from the bare sampler. |
| `base_samples` | Connected input | Base latent from the same sampler. |
| `union_mask` | Connected input | Pixel union identifying where regional decoding is allowed to replace base decoding. |
| `feather_px` | `32` | Average-pool radius used to soften the pixel composite boundary. Zero produces a hard mask. |

## Compatibility notes

- The spatial override occupies ComfyUI’s `optimized_attention_override` hook. Do not put another node that claims the same hook on the same model branch; branch before applying either override.
- Krea regional LoRAs are validated against the active Krea model namespace. A LoRA for another architecture is rejected instead of being silently applied to zero layers.
- Standard regional routes deliberately omit text-fusion and attention key/value adapter targets; the report lists how many compatible targets were skipped for single-pass spatial locality. Global and character-identity routes retain their documented target behavior.
- Global negative conditioning is output normally. Region-local negative text is preserved in workflow configuration, but the current Krea Turbo CFG-free path has no separate regional negative branch.
- Cancel, queue, history, workflow embedding, output naming, model unloading, previews, and resource monitoring are provided by ComfyUI.

Two editable starter graphs are included:

- [`workflows/k2_region_starter.json`](workflows/k2_region_starter.json) is the compact sidebar-driven Studio workflow.
- [`workflows/k2_bare_kj_ideogram_starter.json`](workflows/k2_bare_kj_ideogram_starter.json) exposes the regional LoRA route as ordinary graph nodes and uses KJNodes' **Ideogram 4 Prompt Builder KJ** as the visual, labeled bounding-box and structured-prompt input.

### Bare KJ/Ideogram starter workflow

The bare KJ starter is deliberately a **single denoising pass**:

```text
Ideogram 4 Prompt Builder KJ
  ├─ prompt → native CLIP Text Encode
  └─ bboxes → K2 BBox To Regional Mask (one node per bbox index)
                  → K2 Regional Character LoRA
                  → K2 Regional LoRA Stack 3
native MODEL ─────→ K2 Regional Layer LoRA Apply
                  → native KSampler → VAE Decode → K2 Post Upscale → Preview / Save
```

It does not use `K2 Regional Attention LoRA Sampler` or `K2 Regional Decode Composite`. The layer-apply node installs the region masks on the selected Krea layers, then ComfyUI's native `KSampler` performs one generation. With `outside_strength=0` and `text_token_strength=0`, regional LoRA model deltas are restricted to the selected image-token regions instead of being produced by a second regional sampling pass.

Before queuing the graph:

1. Install or update [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) to a revision containing the `Ideogram4PromptBuilderKJ` node.
2. Select the locally installed Krea 2 diffusion model, Krea/Qwen text encoder (`type=krea2`), and VAE in the three native loader nodes.
3. Draw, resize, label, and reorder objects in the KJ builder. Its first object is bbox index `0`, its second is index `1`, and so on. Keep each `K2 BBox To Regional Mask` node's `bbox_index` aligned with that list order.
4. Select the LoRA file for each region. Set `outside_strength=0` for strict spatial isolation. `overlap_mode=priority_1` makes the first LoRA socket win where boxes overlap; use `normalize` to blend overlapping routes instead.
5. Keep Krea Turbo's native sampler defaults at CFG `1.0`, sampler `euler`, scheduler `simple`, and start with 8 steps. The negative encoder is intentionally blank because the current Krea Turbo path does not provide useful regional-negative behavior.
6. The included upscaler uses deterministic Lanczos at 2x. Connect a native **Load Upscale Model** output and change the method to `upscale_model` to use a neural upscaler.

The KJ node's labeled elements are compiled into its structured JSON prompt, while its pixel-space `BOUNDING_BOX` output drives the K2 masks. The graph starts with two subject boxes and two independent regional LoRA bindings; duplicate a bbox/mask/LoRA branch for more subjects. `K2 Regional LoRA Stack 3` accepts three bindings per stack.

## Development checks

```bash
python -m pytest -q
node --check web/k2_region_studio.js
python -m py_compile __init__.py k2_region_comfy/*.py k2_region_core/**/*.py
```
