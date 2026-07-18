import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const style = document.createElement("link");
style.rel = "stylesheet";
style.href = new URL("./k2_region_studio.css", import.meta.url).href;
document.head.append(style);

const NODE_TYPE = "K2RegionStudio";
const palette = ["#f97316", "#22c55e", "#38bdf8", "#c084fc", "#f43f5e", "#facc15"];

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
  },
  projector: {
    enabled: false, preset: "filter_bypass2", values: Array(12).fill(0),
    multiplier: 1, identity_protection: 1,
  },
  face_detail: {
    enabled: false, steps: 8, denoise: 0.15, crop_size: 512, padding: 2,
    feather: 0.12, blend: 0.5, lora_scale: 0.5, detector_threshold: 0.4,
  },
});

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

class RegionStudio {
  constructor(root) {
    this.root = root;
    this.node = null;
    this.config = blankConfig();
    this.selected = 0;
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
      try { this.config = { ...blankConfig(), ...JSON.parse(widget(node, "region_config")?.value || "{}") }; }
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
    for (const [name, renderer] of [
      ["Regions", () => this.regionsPage()],
      ["LoRAs", () => this.lorasPage()],
      ["Emphasis", () => this.emphasisPage()],
      ["Tuning", () => this.tuningPage()],
      ["JSON", () => this.jsonPage()],
    ]) {
      const button = h("button", {}, name);
      button.onclick = () => {
        tabs.querySelectorAll("button").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        pages.replaceChildren(renderer());
      };
      tabs.append(button);
    }
    this.body.append(tabs, pages);
    tabs.firstChild.click();
  }

  regionsPage() {
    const page = h("section", { class: "k2-page" });
    page.append(
      h("label", {}, "Global prompt", h("textarea", {
        value: this.config.global_prompt,
        onInput: (event) => { this.config.global_prompt = event.target.value; this.commit(); },
      })),
      h("label", {}, "Global negative", h("textarea", {
        value: this.config.global_negative,
        onInput: (event) => { this.config.global_negative = event.target.value; this.commit(); },
      })),
      h("div", { class: "k2-canvas-toolbar" },
        h("span", {}, `${this.width()} × ${this.height()}`),
        h("button", { onClick: () => this.addRegion() }, "+ Region"),
      ),
      this.canvasWrap = h("div", { class: "k2-canvas-wrap" },
        this.canvas = h("canvas", { class: "k2-canvas" }),
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
      const field = (label, key, type = "text") => h("label", {}, label, h("input", {
        type, value: region[key] ?? "",
        onInput: (event) => { region[key] = type === "number" ? Number(event.target.value) : event.target.value; this.commit(); },
      }));
      card.append(
        h("div", { class: "k2-card-head", onClick: () => { this.selected = index; this.renderCanvas(); this.renderRegionCards(); } },
          h("i", { style: `background:${palette[index % palette.length]}` }),
          h("strong", {}, region.name || `Region ${index + 1}`),
          h("button", { title: "Move forward", onClick: (event) => { event.stopPropagation(); if (index) { [this.config.regions[index - 1], this.config.regions[index]] = [this.config.regions[index], this.config.regions[index - 1]]; this.selected = index - 1; this.commit(); this.render(); } } }, "↑"),
          h("button", { title: "Move backward", onClick: (event) => { event.stopPropagation(); if (index < this.config.regions.length - 1) { [this.config.regions[index + 1], this.config.regions[index]] = [this.config.regions[index], this.config.regions[index + 1]]; this.selected = index + 1; this.commit(); this.render(); } } }, "↓"),
          h("button", { onClick: (event) => { event.stopPropagation(); this.config.regions.splice(index, 1); this.commit(); this.render(); } }, "×"),
        ),
        field("Name", "name"),
        h("label", {}, "Prompt", h("textarea", { value: region.prompt || "", onInput: (e) => { region.prompt = e.target.value; this.commit(); } })),
        h("label", {}, "Identity prompt", h("textarea", { value: region.face_identity_prompt || "", onInput: (e) => { region.face_identity_prompt = e.target.value; this.commit(); } })),
        h("label", {}, "Negative", h("textarea", { value: region.negative_prompt || "", onInput: (e) => { region.negative_prompt = e.target.value; this.commit(); } })),
        h("div", { class: "k2-grid" }, field("Priority", "priority", "number"),
          h("label", {}, "Role", h("select", { onChange: (e) => { region.spatial_role = e.target.value; this.commit(); } },
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
      h("button", { onClick: () => { this.config.loras.push({ id: crypto.randomUUID(), name: this.loraNames[0] || "", strength: 1, global: true, region_ids: [], routing_mode: "standard", trigger_phrase: "" }); this.commit(); this.render(); } }, "+ LoRA"),
    );
    this.config.loras.forEach((lora, index) => {
      const nameInput = this.loraNames.length
        ? h("select", { onChange: (e) => { lora.name = e.target.value; this.commit(); } }, ...this.loraNames.map((name) => h("option", { value: name, selected: name === lora.name }, name)))
        : h("input", { value: lora.name, onInput: (e) => { lora.name = e.target.value; this.commit(); } });
      page.append(h("article", { class: "k2-lora" },
        h("div", { class: "k2-card-head" }, h("strong", {}, `LoRA ${index + 1}`), h("button", { onClick: () => { this.config.loras.splice(index, 1); this.commit(); this.render(); } }, "×")),
        h("label", {}, "Model", nameInput),
        h("label", {}, "Strength", h("input", { type: "number", min: -4, max: 4, step: .05, value: lora.strength, onInput: (e) => { lora.strength = Number(e.target.value); this.commit(); } })),
        h("label", {}, h("input", { type: "checkbox", checked: lora.global, onChange: (e) => { lora.global = e.target.checked; this.commit(); this.render(); } }), " Global scope"),
        !lora.global && h("fieldset", {}, h("legend", {}, "Regions"), ...this.config.regions.map((region) => h("label", {}, h("input", {
          type: "checkbox", checked: (lora.region_ids || []).includes(region.id),
          onChange: (e) => { const ids = new Set(lora.region_ids || []); e.target.checked ? ids.add(region.id) : ids.delete(region.id); lora.region_ids = [...ids]; this.commit(); },
        }), region.name))),
        h("label", {}, "Trigger phrase", h("input", { value: lora.trigger_phrase || "", onInput: (e) => { lora.trigger_phrase = e.target.value; this.commit(); } })),
      ));
    });
    return page;
  }

  emphasisPage() {
    const page = h("section", { class: "k2-page" },
      h("p", { class: "k2-hint" }, "Boost an exact phrase in the global prompt or one region. Phrase matching is resolved against the compiled Qwen token sequence."),
      h("button", { onClick: () => { this.config.emphases.push({ scope_id: "__global__", phrase: "", strength: .5, occurrence: 0 }); this.render(); } }, "+ Emphasis"),
    );
    this.config.emphases.forEach((emphasis, index) => page.append(
      h("article", { class: "k2-lora" },
        h("div", { class: "k2-card-head" }, h("strong", {}, `Emphasis ${index + 1}`), h("button", { onClick: () => { this.config.emphases.splice(index, 1); this.commit(); this.render(); } }, "×")),
        h("label", {}, "Scope", h("select", { onChange: (e) => { emphasis.scope_id = e.target.value; this.commit(); } },
          h("option", { value: "__global__", selected: emphasis.scope_id === "__global__" }, "Global prompt"),
          ...this.config.regions.map((region) => h("option", { value: region.id, selected: emphasis.scope_id === region.id }, region.name)))),
        h("label", {}, "Exact phrase", h("input", { value: emphasis.phrase, onInput: (e) => { emphasis.phrase = e.target.value; } , onChange: () => this.commit() })),
        h("label", {}, "Strength", h("input", { type: "number", min: 0, max: 2, step: .05, value: emphasis.strength, onInput: (e) => { emphasis.strength = Number(e.target.value); this.commit(); } })),
        h("label", {}, "Occurrence (zero-based)", h("input", { type: "number", min: 0, step: 1, value: emphasis.occurrence || 0, onInput: (e) => { emphasis.occurrence = Number(e.target.value); this.commit(); } })),
      )
    ));
    return page;
  }

  tuningPage() {
    const page = h("section", { class: "k2-page" });
    const number = (section, key, label, min, max, step) => h("label", {}, label, h("input", {
      type: "number", min, max, step, value: this.config[section][key],
      onInput: (e) => { this.config[section][key] = Number(e.target.value); this.commit(); },
    }));
    const check = (section, key, label) => h("label", {}, h("input", { type: "checkbox", checked: this.config[section][key], onChange: (e) => { this.config[section][key] = e.target.checked; this.commit(); } }), label);
    page.append(
      h("h3", {}, "Spatial attention"), check("spatial", "enabled", " Enabled"),
      number("spatial", "strength", "Inside strength", 0, 8, .05),
      number("spatial", "outside_penalty", "Outside penalty", 0, 8, .05),
      number("spatial", "falloff_pixels", "Edge falloff (px)", 0, 2048, 8),
      number("spatial", "late_step_scale", "Late-step scale", 0, 2, .05),
      check("spatial", "subject_competition", " Subject competition"),
      check("spatial", "subject_fill", " Fill unclaimed subject space"),
      check("spatial", "lora_delta_adaptation", " LoRA-delta adaptation"),
      h("h3", {}, "Projector control"), check("projector", "enabled", " Enabled"),
      h("label", {}, "Preset", h("select", { onChange: (e) => { this.config.projector.preset = e.target.value; this.commit(); } }, ...["filter_bypass2", "filter_bypass3", "skc3vo", "z0jglf", "custom"].map((value) => h("option", { value, selected: value === this.config.projector.preset }, value)))),
      number("projector", "multiplier", "Multiplier", -8, 8, .05),
      number("projector", "identity_protection", "Identity protection", 0, 1, .05),
      h("h3", {}, "Face detail"), check("face_detail", "enabled", " Enabled"),
      number("face_detail", "steps", "Steps", 1, 100, 1), number("face_detail", "denoise", "Denoise", .01, 1, .01),
      number("face_detail", "crop_size", "Crop size", 256, 1024, 256), number("face_detail", "padding", "Padding", 1, 4, .1),
      number("face_detail", "feather", "Feather", 0, .5, .01), number("face_detail", "blend", "Blend", 0, 1, .01),
      number("face_detail", "lora_scale", "LoRA scale", 0, 4, .05),
    );
    return page;
  }

  jsonPage() {
    const area = h("textarea", { class: "k2-json", value: JSON.stringify(this.config, null, 2) });
    const message = h("span", { class: "k2-json-message" });
    return h("section", { class: "k2-page" },
      h("p", { class: "k2-hint" }, "Lossless import/export and access to advanced emphasis vectors or future fields."),
      area,
      h("div", { class: "k2-json-actions" },
        h("button", { onClick: () => navigator.clipboard.writeText(area.value) }, "Copy"),
        h("button", { onClick: () => {
          try { this.config = { ...blankConfig(), ...JSON.parse(area.value) }; this.commit(); message.textContent = "Applied"; }
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
