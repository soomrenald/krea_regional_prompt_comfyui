import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const style = document.createElement("link");
style.rel = "stylesheet";
style.href = new URL("./k2_region_studio.css", import.meta.url).href;
document.head.append(style);

const NODE_TYPE = "K2RegionStudio";
const SAMPLER_NODE_TYPE = "K2RegionalSampler";
const palette = ["#f97316", "#22c55e", "#38bdf8", "#c084fc", "#f43f5e", "#facc15"];
const HELP = {
  "tab.Regions": "Draw labeled regions and edit the global and per-region prompts.",
  "tab.LoRAs": "Assign LoRA files globally or to selected regions.",
  "tab.Emphasis": "Boost an exact phrase occurrence in the global prompt or one region.",
  "tab.Tuning": "Tune spatial attention, projector behavior, and face refinement.",
  "tab.JSON": "Inspect, copy, or replace the complete portable Studio configuration.",
  global_prompt: "Scene-wide positive description. Regional clauses are compiled after this text.",
  global_negative: "Scene-wide negative prompt used by the main and face-detail conditioning.",
  add_region: "Add a draggable region with a unique ID and descending default priority.",
  canvas: "Drag inside a box to move it; drag its lower-right handle to resize it.",
  region_name: "Human-readable label used in the compiled spatial instructions and reports.",
  region_enabled: "Include this region in compilation and face assignment. Remove it from regional LoRA assignments before disabling it.",
  region_prompt: "Description that should appear inside this region.",
  identity_prompt: "Face-specific identity description protected from projector changes and reused by Face Detail.",
  region_negative: "Region-local negative text stored in the project; Krea Turbo currently uses the global negative branch.",
  priority: "Higher values compile first and get first chance to claim an ambiguous detected face; this is not a strength or z-index.",
  role: "auto treats wide boxes as background; subject enables competition/fill; background uses softer outside penalties.",
  lora_add: "Add a LoRA assignment using a file from ComfyUI/models/loras.",
  lora_model: "LoRA file applied by this assignment.",
  lora_strength: "Model-delta multiplier from -4 to 4; zero disables the assignment and negative values invert it.",
  lora_global: "Apply across all text and image tokens instead of restricting the LoRA to selected regions.",
  lora_routing: "With Global scope off, Standard gates text-fusion deltas to assigned clauses. Off-box image tokens cannot attend those clauses, and other text cannot read image keys in the box. Image-to-image attention remains continuous to avoid rectangular seams. Main-stream attention key/value targets remain omitted. Character identity adds its anchor on top of the same isolation.",
  lora_regions: "Pixel boxes whose regional clauses and compatible LoRA deltas are allowed to condition intersecting Krea image tokens.",
  lora_trigger: "Character trigger phrase appended to an assigned regional clause when character-identity routing is selected in JSON.",
  emphasis_add: "Add an exact-phrase emphasis rule.",
  emphasis_scope: "Prompt section searched for the exact phrase: global prompt or one named region.",
  emphasis_phrase: "Case-sensitive exact text to emphasize; it must exist in the selected scope.",
  emphasis_strength: "Additional spatial attention boost for the phrase, from 0 (none) to 2 (strong).",
  emphasis_occurrence: "Zero-based match index: 0 is the first occurrence, 1 the second, and so on.",
  "spatial.enabled": "Enable the regional text partition, box router, and phrase emphasis. Required whenever a non-global LoRA is active.",
  "spatial.strength": "Positive attention-logit bias inside each region; higher values bind its text more strongly to the box.",
  "spatial.outside_penalty": "Increases center-to-edge contrast inside subject boxes; subject text is hard-blocked outside. Background bands use one quarter of this value.",
  "spatial.falloff_pixels": "Distance over which background-band guidance fades outside its box. Subject text remains confined to intersecting image tokens.",
  "spatial.late_step_scale": "Fraction of spatial strength retained at the final denoising step after relaxation begins.",
  "spatial.subject_competition": "Divide ownership where subject regions overlap so they compete instead of fully stacking.",
  "spatial.subject_fill": "Use stronger coverage toward subject-box edges to reduce unclaimed space inside a subject region.",
  "spatial.lora_delta_adaptation": "Adjust per-region spatial strength from observed LoRA-delta energy while sampling.",
  "spatial.lora_delta_adaptation_gain": "Maximum correction gain used when LoRA-delta adaptation balances regional spatial strength.",
  "spatial.strict_lora_isolation": "Use the PySide v6/v3 hard subject-text partition and omit standard-LoRA key/value targets that can leak across regions. Keep enabled for regional LoRAs.",
  "projector.enabled": "Apply a selected 12-value delta to Krea's text-fusion projector.",
  "projector.preset": "Named projector-vector preset; custom values can be edited in the JSON pane.",
  "projector.multiplier": "Signed scale applied to every projector preset value; zero disables its effect.",
  "projector.identity_protection": "Reduce projector changes on face-identity prompt tokens: 0 applies fully, 1 protects completely.",
  "face_detail.enabled": "Enable the separate detector-driven face crop refinement node when it receives this region plan.",
  "face_detail.steps": "Denoising iterations for each assigned face crop.",
  "face_detail.denoise": "Face-crop denoise fraction; lower values preserve the original face structure.",
  "face_detail.crop_size": "Square pixel resolution used while refining each face crop.",
  "face_detail.padding": "Multiplier expanding the detected face box before it is made square and cropped.",
  "face_detail.feather": "Fractional edge width used to soften the refined crop mask during compositing.",
  "face_detail.blend": "Opacity of the refined crop over the source image; 0 keeps the source and 1 uses the refinement.",
  "face_detail.lora_scale": "Extra multiplier applied to region-assigned LoRAs during face refinement.",
  "face_detail.detector_threshold": "Minimum NanoDet face confidence accepted for refinement; higher values reject uncertain detections.",
  json: "Complete workflow-stored configuration, including advanced fields not shown as sidebar controls.",
  json_copy: "Copy the current configuration JSON to the clipboard.",
  json_apply: "Validate and apply the edited JSON to the selected K2 Region Studio node.",
};

const blankConfig = () => ({
  version: 1,
  global_prompt: "",
  global_negative: "",
  regions: [],
  loras: [],
  emphases: [],
  spatial: {
    enabled: true, strength: 1, outside_penalty: 1, falloff_pixels: 128,
    subject_competition: true, subject_fill: true, late_step_scale: 0.35,
    lora_delta_adaptation: false, lora_delta_adaptation_gain: 0.35,
    strict_lora_isolation: true,
  },
  projector: {
    enabled: false, preset: "filter_bypass2", values: Array(12).fill(0),
    multiplier: 1, identity_protection: 1,
  },
  face_detail: {
    enabled: false, steps: 8, denoise: 0.15, crop_size: 512, padding: 2,
    feather: 0.12, blend: 0.5, lora_scale: 0.5, detector_threshold: 0.15,
  },
});

const mergedConfig = (supplied = {}) => {
  const defaults = blankConfig();
  return {
    ...defaults,
    ...supplied,
    spatial: { ...defaults.spatial, ...(supplied.spatial || {}) },
    projector: { ...defaults.projector, ...(supplied.projector || {}) },
    face_detail: { ...defaults.face_detail, ...(supplied.face_detail || {}) },
  };
};

const h = (tag, attrs = {}, ...children) => {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") element.className = value;
    else if (key.startsWith("on")) element.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key === "checked") element.checked = value;
    else if (key === "selected") element.selected = value;
    else if (key === "value") element.value = value;
    else element.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child !== null && child !== undefined) element.append(child.nodeType ? child : String(child));
  }
  return element;
};

const widget = (node, name) => node?.widgets?.find((item) => item.name === name);

const repairShiftedSamplerDefaults = (node) => {
  const cfg = widget(node, "cfg");
  const sampler = widget(node, "sampler_name");
  const scheduler = widget(node, "scheduler");
  const denoise = widget(node, "denoise");
  if (typeof cfg?.value !== "string" || typeof sampler?.value !== "string" || typeof scheduler?.value !== "number") return;
  const samplerValue = cfg.value;
  const schedulerValue = sampler.value;
  const denoiseValue = scheduler.value;
  cfg.value = 1;
  sampler.value = samplerValue;
  scheduler.value = schedulerValue;
  if (denoise) denoise.value = denoiseValue;
};

class RegionStudio {
  constructor(root) {
    this.root = root;
    this.node = null;
    this.config = blankConfig();
    this.selected = 0;
    this.activePane = "Regions";
    this.drag = null;
    this.loraNames = [];
    this.renderShell();
    this.loadLoras();
    this.watch = window.setInterval(() => this.followSelection(), 500);
  }

  renderShell() {
    this.root.className = "k2-studio";
    this.root.replaceChildren(
      h("header", { class: "k2-header" },
        h("div", {}, h("strong", {}, "K2 Region Studio"), h("small", {}, "Krea 2 spatial composition")),
        h("button", { onClick: () => this.createNode(), title: "Create a K2 Region Studio node" }, "+ Node"),
      ),
      this.status = h("div", { class: "k2-status" }, "Select a K2 Region Studio node."),
      this.body = h("main", { class: "k2-body" }),
    );
    this.render();
  }

  async loadLoras() {
    try {
      const response = await api.fetchApi("/object_info/LoraLoader");
      const data = await response.json();
      this.loraNames = data?.LoraLoader?.input?.required?.lora_name?.[0] || [];
      this.render();
    } catch (_) { /* A free-text fallback remains available. */ }
  }

  followSelection() {
    const selected = Object.values(app.canvas?.selected_nodes || {}).at(-1);
    if (selected?.type === NODE_TYPE && selected !== this.node) this.bind(selected);
    if (this.node && !app.graph?._nodes?.includes(this.node)) this.bind(null);
  }

  createNode() {
    const node = LiteGraph.createNode(NODE_TYPE);
    if (!node) return;
    node.pos = [app.canvas.graph_mouse[0], app.canvas.graph_mouse[1]];
    app.graph.add(node);
    app.canvas.selectNode(node);
    this.bind(node);
  }

  bind(node) {
    this.node = node;
    if (node) {
      try { this.config = mergedConfig(JSON.parse(widget(node, "region_config")?.value || "{}")); }
      catch (_) { this.config = blankConfig(); }
      this.status.textContent = `Editing node ${node.id}`;
      this.status.classList.add("connected");
    } else {
      this.status.textContent = "Select a K2 Region Studio node.";
      this.status.classList.remove("connected");
    }
    this.selected = Math.min(this.selected, Math.max(0, this.config.regions.length - 1));
    this.render();
  }

  commit() {
    if (!this.node) return;
    const target = widget(this.node, "region_config");
    if (target) {
      target.value = JSON.stringify(this.config);
      target.callback?.(target.value, app.canvas, this.node, target.pos);
    }
    app.graph.setDirtyCanvas(true, true);
    this.renderCanvas();
  }

  width() { return Number(widget(this.node, "width")?.value || 1024); }
  height() { return Number(widget(this.node, "height")?.value || 1024); }

  render() {
    this.body.replaceChildren();
    if (!this.node) {
      this.body.append(h("div", { class: "k2-empty" }, "Add or select a K2 Region Studio node to edit its portable workflow configuration."));
      return;
    }
    const tabs = h("nav", { class: "k2-tabs" });
    const pages = h("div", {});
    let activeButton = null;
    for (const [name, renderer] of [
      ["Regions", () => this.regionsPage()],
      ["LoRAs", () => this.lorasPage()],
      ["Emphasis", () => this.emphasisPage()],
      ["Tuning", () => this.tuningPage()],
      ["JSON", () => this.jsonPage()],
    ]) {
      const button = h("button", { title: HELP[`tab.${name}`] }, name);
      button.onclick = () => {
        this.activePane = name;
        tabs.querySelectorAll("button").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        pages.replaceChildren(renderer());
      };
      if (name === this.activePane) activeButton = button;
      tabs.append(button);
    }
    this.body.append(tabs, pages);
    (activeButton || tabs.firstChild)?.click();
  }

  regionsPage() {
    const page = h("section", { class: "k2-page" });
    page.append(
      h("label", { title: HELP.global_prompt }, "Global prompt", h("textarea", {
        value: this.config.global_prompt,
        onInput: (event) => { this.config.global_prompt = event.target.value; this.commit(); },
      })),
      h("label", { title: HELP.global_negative }, "Global negative", h("textarea", {
        value: this.config.global_negative,
        onInput: (event) => { this.config.global_negative = event.target.value; this.commit(); },
      })),
      h("div", { class: "k2-canvas-toolbar" },
        h("span", {}, `${this.width()} × ${this.height()}`),
        h("button", { onClick: () => this.addRegion(), title: HELP.add_region }, "+ Region"),
      ),
      this.canvasWrap = h("div", { class: "k2-canvas-wrap", title: HELP.canvas },
        this.canvas = h("canvas", { class: "k2-canvas", title: HELP.canvas }),
      ),
      this.regionCards = h("div", { class: "k2-region-cards" }),
    );
    this.installCanvasEvents();
    this.renderCanvas();
    this.renderRegionCards();
    return page;
  }

  addRegion() {
    const index = this.config.regions.length;
    const inset = 64 + index * 24;
    this.config.regions.push({
      id: crypto.randomUUID(), name: `Region ${index + 1}`,
      box: { x0: inset, y0: inset, x1: Math.min(this.width(), inset + 384), y1: Math.min(this.height(), inset + 512) },
      prompt: "", negative_prompt: "", face_identity_prompt: "", enabled: true,
      priority: Math.max(0, 100 - index), spatial_role: "auto",
    });
    this.selected = index;
    this.commit(); this.render();
  }

  renderRegionCards() {
    this.regionCards.replaceChildren();
    this.config.regions.forEach((region, index) => {
      const card = h("article", { class: `k2-region-card ${index === this.selected ? "selected" : ""}` });
      const field = (label, key, type = "text", help = HELP[key]) => h("label", { title: help }, label, h("input", {
        type, value: region[key] ?? "",
        onInput: (event) => { region[key] = type === "number" ? Number(event.target.value) : event.target.value; this.commit(); },
      }));
      card.append(
        h("div", { class: "k2-card-head", onClick: () => { this.selected = index; this.renderCanvas(); this.renderRegionCards(); } },
          h("i", { style: `background:${palette[index % palette.length]}` }),
          h("strong", {}, region.name || `Region ${index + 1}`),
          h("button", { title: "Move forward", onClick: (event) => { event.stopPropagation(); if (index) { [this.config.regions[index - 1], this.config.regions[index]] = [this.config.regions[index], this.config.regions[index - 1]]; this.selected = index - 1; this.commit(); this.render(); } } }, "↑"),
          h("button", { title: "Move backward", onClick: (event) => { event.stopPropagation(); if (index < this.config.regions.length - 1) { [this.config.regions[index + 1], this.config.regions[index]] = [this.config.regions[index], this.config.regions[index + 1]]; this.selected = index + 1; this.commit(); this.render(); } } }, "↓"),
          h("button", { title: "Delete this region.", onClick: (event) => { event.stopPropagation(); this.config.regions.splice(index, 1); this.commit(); this.render(); } }, "×"),
        ),
        field("Name", "name", "text", HELP.region_name),
        h("label", { title: HELP.region_enabled }, h("input", { type: "checkbox", checked: region.enabled !== false, onChange: (e) => { region.enabled = e.target.checked; this.commit(); } }), " Enabled"),
        h("label", { title: HELP.region_prompt }, "Prompt", h("textarea", { value: region.prompt || "", onInput: (e) => { region.prompt = e.target.value; this.commit(); } })),
        h("label", { title: HELP.identity_prompt }, "Identity prompt", h("textarea", { value: region.face_identity_prompt || "", onInput: (e) => { region.face_identity_prompt = e.target.value; this.commit(); } })),
        h("label", { title: HELP.region_negative }, "Negative", h("textarea", { value: region.negative_prompt || "", onInput: (e) => { region.negative_prompt = e.target.value; this.commit(); } })),
        h("div", { class: "k2-grid" }, field("Priority", "priority", "number", HELP.priority),
          h("label", { title: HELP.role }, "Role", h("select", { onChange: (e) => { region.spatial_role = e.target.value; this.commit(); } },
            ...["auto", "subject", "background"].map((value) => h("option", { value, selected: value === region.spatial_role }, value)))),
        ),
      );
      this.regionCards.append(card);
    });
  }

  installCanvasEvents() {
    const point = (event) => {
      const rect = this.canvas.getBoundingClientRect();
      return [(event.clientX - rect.left) * this.width() / rect.width, (event.clientY - rect.top) * this.height() / rect.height];
    };
    this.canvas.onpointerdown = (event) => {
      const [x, y] = point(event);
      for (let index = this.config.regions.length - 1; index >= 0; index--) {
        const box = this.config.regions[index].box;
        if (x >= box.x0 && x <= box.x1 && y >= box.y0 && y <= box.y1) {
          this.selected = index;
          const resize = Math.abs(x - box.x1) < 28 && Math.abs(y - box.y1) < 28;
          this.drag = { x, y, box: { ...box }, resize };
          this.canvas.setPointerCapture(event.pointerId);
          this.renderRegionCards(); this.renderCanvas(); return;
        }
      }
    };
    this.canvas.onpointermove = (event) => {
      if (!this.drag) return;
      const [x, y] = point(event); const box = this.config.regions[this.selected].box;
      if (this.drag.resize) {
        box.x1 = Math.max(box.x0 + 16, Math.min(this.width(), this.drag.box.x1 + x - this.drag.x));
        box.y1 = Math.max(box.y0 + 16, Math.min(this.height(), this.drag.box.y1 + y - this.drag.y));
      } else {
        const dx = Math.max(-this.drag.box.x0, Math.min(this.width() - this.drag.box.x1, x - this.drag.x));
        const dy = Math.max(-this.drag.box.y0, Math.min(this.height() - this.drag.box.y1, y - this.drag.y));
        Object.assign(box, { x0: this.drag.box.x0 + dx, x1: this.drag.box.x1 + dx, y0: this.drag.box.y0 + dy, y1: this.drag.box.y1 + dy });
      }
      this.renderCanvas();
    };
    this.canvas.onpointerup = () => { if (this.drag) this.commit(); this.drag = null; };
  }

  renderCanvas() {
    if (!this.canvas) return;
    const width = this.width(), height = this.height();
    this.canvas.width = Math.min(900, width); this.canvas.height = this.canvas.width * height / width;
    const ctx = this.canvas.getContext("2d");
    ctx.fillStyle = "#111827"; ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    const sx = this.canvas.width / width, sy = this.canvas.height / height;
    this.config.regions.forEach((region, index) => {
      const box = region.box; ctx.fillStyle = `${palette[index % palette.length]}33`;
      ctx.strokeStyle = palette[index % palette.length]; ctx.lineWidth = index === this.selected ? 4 : 2;
      ctx.fillRect(box.x0 * sx, box.y0 * sy, (box.x1 - box.x0) * sx, (box.y1 - box.y0) * sy);
      ctx.strokeRect(box.x0 * sx, box.y0 * sy, (box.x1 - box.x0) * sx, (box.y1 - box.y0) * sy);
      ctx.fillStyle = "white"; ctx.font = "bold 13px sans-serif"; ctx.fillText(region.name || `Region ${index + 1}`, box.x0 * sx + 7, box.y0 * sy + 18);
      ctx.fillStyle = palette[index % palette.length]; ctx.fillRect(box.x1 * sx - 10, box.y1 * sy - 10, 10, 10);
    });
  }

  lorasPage() {
    const page = h("section", { class: "k2-page" },
      h("p", { class: "k2-hint" }, "Choose any LoRA from ComfyUI’s models/loras folder. Global entries affect the full image; regional entries are token-delta gated to selected boxes."),
      h("button", { title: HELP.lora_add, onClick: () => { this.config.loras.push({ id: crypto.randomUUID(), name: this.loraNames[0] || "", strength: 1, global: true, region_ids: [], routing_mode: "standard", trigger_phrase: "" }); this.commit(); this.render(); } }, "+ LoRA"),
    );
    this.config.loras.forEach((lora, index) => {
      const nameInput = this.loraNames.length
        ? h("select", { onChange: (e) => { lora.name = e.target.value; this.commit(); } }, ...this.loraNames.map((name) => h("option", { value: name, selected: name === lora.name }, name)))
        : h("input", { value: lora.name, onInput: (e) => { lora.name = e.target.value; this.commit(); } });
      page.append(h("article", { class: "k2-lora" },
        h("div", { class: "k2-card-head" }, h("strong", {}, `LoRA ${index + 1}`), h("button", { title: "Delete this LoRA assignment.", onClick: () => { this.config.loras.splice(index, 1); this.commit(); this.render(); } }, "×")),
        h("label", { title: HELP.lora_model }, "Model", nameInput),
        h("label", { title: HELP.lora_strength }, "Strength", h("input", { type: "number", min: -4, max: 4, step: .05, value: lora.strength, onInput: (e) => { lora.strength = Number(e.target.value); this.commit(); } })),
        h("label", { title: HELP.lora_routing }, "Routing", h("select", { onChange: (e) => { lora.routing_mode = e.target.value; this.commit(); this.render(); } },
          ...["standard", "character_identity"].map((value) => h("option", { value, selected: value === (lora.routing_mode || "standard") }, value)))),
        h("label", { title: HELP.lora_global }, h("input", { type: "checkbox", checked: lora.global, onChange: (e) => { lora.global = e.target.checked; this.commit(); this.render(); } }), " Global scope"),
        !lora.global && h("fieldset", { title: HELP.lora_regions }, h("legend", {}, "Regions"), ...this.config.regions.map((region) => h("label", { title: `Apply this LoRA to ${region.name}.` }, h("input", {
          type: "checkbox", checked: (lora.region_ids || []).includes(region.id),
          onChange: (e) => { const ids = new Set(lora.region_ids || []); e.target.checked ? ids.add(region.id) : ids.delete(region.id); lora.region_ids = [...ids]; this.commit(); },
        }), region.name))),
        lora.routing_mode === "character_identity" && h("label", { title: HELP.lora_trigger }, "Trigger phrase", h("input", { value: lora.trigger_phrase || "", onInput: (e) => { lora.trigger_phrase = e.target.value; this.commit(); } })),
      ));
    });
    return page;
  }

  emphasisPage() {
    const page = h("section", { class: "k2-page" },
      h("p", { class: "k2-hint" }, "Boost an exact phrase in the global prompt or one region. Phrase matching is resolved against the compiled Qwen token sequence."),
      h("button", { title: HELP.emphasis_add, onClick: () => { this.config.emphases.push({ scope_id: "__global__", phrase: "", strength: .5, occurrence: 0 }); this.render(); } }, "+ Emphasis"),
    );
    this.config.emphases.forEach((emphasis, index) => page.append(
      h("article", { class: "k2-lora" },
        h("div", { class: "k2-card-head" }, h("strong", {}, `Emphasis ${index + 1}`), h("button", { title: "Delete this emphasis rule.", onClick: () => { this.config.emphases.splice(index, 1); this.commit(); this.render(); } }, "×")),
        h("label", { title: HELP.emphasis_scope }, "Scope", h("select", { onChange: (e) => { emphasis.scope_id = e.target.value; this.commit(); } },
          h("option", { value: "__global__", selected: emphasis.scope_id === "__global__" }, "Global prompt"),
          ...this.config.regions.map((region) => h("option", { value: region.id, selected: emphasis.scope_id === region.id }, region.name)))),
        h("label", { title: HELP.emphasis_phrase }, "Exact phrase", h("input", { value: emphasis.phrase, onInput: (e) => { emphasis.phrase = e.target.value; } , onChange: () => this.commit() })),
        h("label", { title: HELP.emphasis_strength }, "Strength", h("input", { type: "number", min: 0, max: 2, step: .05, value: emphasis.strength, onInput: (e) => { emphasis.strength = Number(e.target.value); this.commit(); } })),
        h("label", { title: HELP.emphasis_occurrence }, "Occurrence (zero-based)", h("input", { type: "number", min: 0, step: 1, value: emphasis.occurrence || 0, onInput: (e) => { emphasis.occurrence = Number(e.target.value); this.commit(); } })),
      )
    ));
    return page;
  }

  tuningPage() {
    const page = h("section", { class: "k2-page" });
    const number = (section, key, label, min, max, step) => h("label", { title: HELP[`${section}.${key}`] }, label, h("input", {
      type: "number", min, max, step, value: this.config[section][key],
      onInput: (e) => { this.config[section][key] = Number(e.target.value); this.commit(); },
    }));
    const check = (section, key, label) => h("label", { title: HELP[`${section}.${key}`] }, h("input", { type: "checkbox", checked: this.config[section][key], onChange: (e) => { this.config[section][key] = e.target.checked; this.commit(); } }), label);
    page.append(
      h("h3", {}, "Spatial attention"), check("spatial", "enabled", " Enabled"),
      number("spatial", "strength", "Inside strength", 0, 8, .05),
      number("spatial", "outside_penalty", "Outside penalty", 0, 8, .05),
      number("spatial", "falloff_pixels", "Edge falloff (px)", 0, 2048, 8),
      number("spatial", "late_step_scale", "Late-step scale", 0, 2, .05),
      check("spatial", "subject_competition", " Subject competition"),
      check("spatial", "subject_fill", " Fill unclaimed subject space"),
      check("spatial", "lora_delta_adaptation", " LoRA-delta adaptation"),
      number("spatial", "lora_delta_adaptation_gain", "LoRA adaptation gain", 0, 1, .05),
      check("spatial", "strict_lora_isolation", " Strict regional LoRA isolation"),
      h("h3", {}, "Projector control"), check("projector", "enabled", " Enabled"),
      h("label", { title: HELP["projector.preset"] }, "Preset", h("select", { onChange: (e) => { this.config.projector.preset = e.target.value; this.commit(); } }, ...["filter_bypass2", "filter_bypass3", "skc3vo", "z0jglf", "custom"].map((value) => h("option", { value, selected: value === this.config.projector.preset }, value)))),
      number("projector", "multiplier", "Multiplier", -8, 8, .05),
      number("projector", "identity_protection", "Identity protection", 0, 1, .05),
      h("h3", {}, "Face detail"), check("face_detail", "enabled", " Enabled"),
      number("face_detail", "steps", "Steps", 1, 100, 1), number("face_detail", "denoise", "Denoise", .01, 1, .01),
      number("face_detail", "crop_size", "Crop size", 256, 1024, 256), number("face_detail", "padding", "Padding", 1, 4, .1),
      number("face_detail", "feather", "Feather", 0, .5, .01), number("face_detail", "blend", "Blend", 0, 1, .01),
      number("face_detail", "lora_scale", "LoRA scale", 0, 4, .05),
      number("face_detail", "detector_threshold", "Detector threshold", 0, 1, .01),
    );
    return page;
  }

  jsonPage() {
    const area = h("textarea", { class: "k2-json", title: HELP.json, value: JSON.stringify(this.config, null, 2) });
    const message = h("span", { class: "k2-json-message" });
    return h("section", { class: "k2-page" },
      h("p", { class: "k2-hint" }, "Lossless import/export and access to advanced emphasis vectors or future fields."),
      area,
      h("div", { class: "k2-json-actions" },
        h("button", { title: HELP.json_copy, onClick: () => navigator.clipboard.writeText(area.value) }, "Copy"),
        h("button", { title: HELP.json_apply, onClick: () => {
          try { this.config = mergedConfig(JSON.parse(area.value)); this.commit(); message.textContent = "Applied"; }
          catch (error) { message.textContent = error.message; }
        } }, "Apply JSON"), message,
      ),
    );
  }
}

app.registerExtension({
  name: "k2.region.studio",
  setup() {
    app.extensionManager.registerSidebarTab({
      id: "k2-region-studio", icon: "pi pi-th-large", title: "K2 Regions",
      tooltip: "K2 Region Studio", type: "custom",
      render: (container) => new RegionStudio(container),
    });
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === SAMPLER_NODE_TYPE) {
      const created = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        created?.apply(this, arguments);
        repairShiftedSamplerDefaults(this);
      };
      const configured = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        configured?.apply(this, arguments);
        repairShiftedSamplerDefaults(this);
      };
      return;
    }
    if (nodeData.name !== NODE_TYPE) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      created?.apply(this, arguments);
      const configWidget = widget(this, "region_config");
      if (configWidget) {
        configWidget.type = "hidden";
        configWidget.computeSize = () => [0, -4];
      }
      this.setSize([Math.max(this.size[0], 330), this.size[1]]);
    };
  },
});
