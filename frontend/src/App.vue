<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from "vue";
import {
  BoxSelect,
  DatabaseZap,
  KeyRound,
  MousePointer2,
  Play,
  RefreshCw,
  Trash2,
  Upload
} from "lucide-vue-next";
import PromptCanvas from "./components/PromptCanvas.vue";
import ResultPane from "./components/ResultPane.vue";
import type { ModelCatalog, ModelInfo, PromptBox, PromptPoint, Task, VisionControls } from "./api";
import { fetchModelCatalog } from "./api";
import type { PanelResult } from "./compareRunner";
import { runComparison } from "./compareRunner";
import type { NaturalSize } from "./promptGeometry";

// ── state ────────────────────────────────────────────────────────────────────

const apiKey = ref(sessionStorage.getItem("vision-api-key") || "");
const catalog = ref<ModelCatalog | null>(null);
const catalogError = ref("");
const loadingModels = ref(false);
const task = ref<Task>("segment");
const modelA = ref("");
const modelB = ref("");
const imageFile = ref<File | null>(null);
const imageUrl = ref("");
const naturalSize = ref<NaturalSize>({ width: 0, height: 0 });
const promptMode = ref<"point" | "box">("point");
const pointLabel = ref<0 | 1>(1);
const running = ref(false);
const runError = ref("");
const resultA = ref<PanelResult | null>(null);
const resultB = ref<PanelResult | null>(null);

// shared controls: same for both models
const shared = reactive({
  conf: 0.25,
  overlay: "both",
  outputFormats: ["mask_png"] as string[],
  points: [] as PromptPoint[],
  boxes: [] as PromptBox[]
});

// per-model controls: independent text inputs
const perA = reactive({ prompt: "", classes: "" });
const perB = reactive({ prompt: "", classes: "" });

// ── model catalog helpers ────────────────────────────────────────────────────

const activeModels = computed<ModelInfo[]>(() =>
  (catalog.value?.models ?? []).filter((m) => m.task === task.value && m.active)
);

const modelAInfo = computed<ModelInfo | undefined>(() =>
  activeModels.value.find((m) => m.name === modelA.value)
);
const modelBInfo = computed<ModelInfo | undefined>(() =>
  activeModels.value.find((m) => m.name === modelB.value)
);

function modesOf(info: ModelInfo | undefined): Set<string> {
  return new Set(info?.input_modes ?? []);
}

const modesA = computed(() => modesOf(modelAInfo.value));
const modesB = computed(() => modesOf(modelBInfo.value));

// union — drives shared UI elements (canvas, toolbar)
const anyModes = computed<Set<string>>(() => new Set([...modesA.value, ...modesB.value]));

const supportsVisualPrompt = computed(
  () => anyModes.value.has("points") || anyModes.value.has("boxes")
);
const supportsPoints = computed(() => anyModes.value.has("points"));
const supportsBoxes  = computed(() => anyModes.value.has("boxes"));

// ── overlay options ──────────────────────────────────────────────────────────

const overlayOptions = computed(() =>
  task.value === "detect"
    ? [
        { value: "none", label: "None" },
        { value: "bbox", label: "BBox" }
      ]
    : [
        { value: "none", label: "None" },
        { value: "bbox", label: "BBox" },
        { value: "mask", label: "Mask" },
        { value: "both", label: "Both" }
      ]
);

const canRun = computed(() =>
  Boolean(apiKey.value && imageFile.value && modelA.value && modelB.value && !running.value)
);

// ── watchers ─────────────────────────────────────────────────────────────────

watch(apiKey, (value) => {
  if (value) {
    sessionStorage.setItem("vision-api-key", value);
    scheduleLoadModels();
  } else {
    sessionStorage.removeItem("vision-api-key");
    catalog.value = null;
    modelA.value = "";
    modelB.value = "";
  }
});

watch(task, () => {
  shared.overlay = task.value === "detect" ? "bbox" : "both";
  syncSelectedModels();
  clearResults();
});

// clear unsupported visual prompts when model changes
watch([modelA, modelB], () => {
  if (!supportsPoints.value) {
    shared.points.splice(0);
    if (promptMode.value === "point") promptMode.value = "box";
  }
  if (!supportsBoxes.value) {
    shared.boxes.splice(0);
    if (promptMode.value === "box") promptMode.value = "point";
  }
});

onMounted(() => {
  if (apiKey.value) loadModels();
});

// ── model loading ─────────────────────────────────────────────────────────────

let modelLoadTimer: number | undefined;

function scheduleLoadModels() {
  window.clearTimeout(modelLoadTimer);
  modelLoadTimer = window.setTimeout(loadModels, 350);
}

async function loadModels() {
  if (!apiKey.value) {
    catalogError.value = "API key required";
    return;
  }
  catalogError.value = "";
  loadingModels.value = true;
  try {
    catalog.value = await fetchModelCatalog(apiKey.value);
    syncSelectedModels();
  } catch (error) {
    catalog.value = null;
    catalogError.value =
      error && typeof error === "object" && "message" in error
        ? String(error.message)
        : "Unable to load models";
  } finally {
    loadingModels.value = false;
  }
}

function syncSelectedModels() {
  const models = activeModels.value;
  if (models.length === 0) {
    modelA.value = "";
    modelB.value = "";
    return;
  }
  const defaultName = catalog.value?.defaults[task.value];
  modelA.value = models.find((m) => m.name === defaultName)?.name ?? models[0].name;
  modelB.value = models.find((m) => m.name !== modelA.value)?.name ?? modelA.value;
  shared.conf = models.find((m) => m.name === modelA.value)?.default_conf ?? shared.conf;
}

// ── image upload ──────────────────────────────────────────────────────────────

function onFileChange(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;
  if (imageUrl.value) URL.revokeObjectURL(imageUrl.value);
  imageFile.value = file;
  imageUrl.value = URL.createObjectURL(file);
  clearResults();
}

function onPreviewImageLoad(event: Event) {
  const img = event.target as HTMLImageElement;
  naturalSize.value = { width: img.naturalWidth, height: img.naturalHeight };
}

// ── run ───────────────────────────────────────────────────────────────────────

function buildControls(per: { prompt: string; classes: string }): VisionControls {
  return {
    conf: shared.conf,
    overlay: shared.overlay,
    outputFormats: shared.outputFormats,
    points: shared.points,
    boxes: shared.boxes,
    prompt: per.prompt,
    classes: per.classes
  };
}

async function runAB() {
  if (!canRun.value || !imageFile.value) return;
  running.value = true;
  runError.value = "";
  resultA.value = null;
  resultB.value = null;
  try {
    const response = await runComparison(
      {
        apiKey: apiKey.value,
        left: {
          task: task.value,
          model: modelA.value,
          inputModes: modelAInfo.value?.input_modes ?? [],
          image: imageFile.value,
          filename: imageFile.value.name,
          controls: buildControls(perA)
        },
        right: {
          task: task.value,
          model: modelB.value,
          inputModes: modelBInfo.value?.input_modes ?? [],
          image: imageFile.value,
          filename: imageFile.value.name,
          controls: buildControls(perB)
        }
      },
      { pollIntervalMs: 1000 }
    );
    resultA.value = response.left;
    resultB.value = response.right;
  } catch (error) {
    runError.value = error instanceof Error ? error.message : "Comparison failed";
  } finally {
    running.value = false;
  }
}

// ── canvas helpers ────────────────────────────────────────────────────────────

function addPoint(point: PromptPoint) { shared.points.push(point); }
function addBox(box: PromptBox)       { shared.boxes.push(box); }
function removePoint(i: number)       { shared.points.splice(i, 1); }
function removeBox(i: number)         { shared.boxes.splice(i, 1); }

function clearPrompts() {
  shared.points.splice(0);
  shared.boxes.splice(0);
}

function clearResults() {
  resultA.value = null;
  resultB.value = null;
  runError.value = "";
}
</script>

<template>
  <main class="app-shell">
    <aside class="control-panel">
      <header class="brand-block">
        <DatabaseZap :size="22" />
        <div>
          <p class="eyebrow">Vision API</p>
          <h1>Playground</h1>
        </div>
      </header>

      <!-- API key -->
      <section class="field-group">
        <label>
          <span><KeyRound :size="16" /> API Key</span>
          <div class="inline-control">
            <input v-model="apiKey" type="password" autocomplete="off" />
            <button class="icon-button" type="button" title="Refresh models" @click="loadModels">
              <RefreshCw :size="17" />
            </button>
          </div>
        </label>
        <p v-if="catalogError" class="inline-error">{{ catalogError }}</p>
      </section>

      <!-- upload -->
      <section class="field-group">
        <label class="upload-box">
          <Upload :size="20" />
          <span>{{ imageFile?.name || "Upload image" }}</span>
          <input type="file" accept="image/png,image/jpeg,image/webp" @change="onFileChange" />
        </label>
      </section>

      <!-- task -->
      <section class="field-group">
        <span class="field-label">Task</span>
        <div class="segmented">
          <button :class="{ active: task === 'segment' }" type="button" @click="task = 'segment'">Segment</button>
          <button :class="{ active: task === 'detect' }"  type="button" @click="task = 'detect'">Detect</button>
        </div>
      </section>

      <!-- model selectors + per-model inputs -->
      <section class="field-group two-col">
        <div class="model-col">
          <label>
            <span>Model A</span>
            <select v-model="modelA" :disabled="activeModels.length === 0">
              <option v-if="activeModels.length === 0" value="">
                {{ loadingModels ? "Loading models" : "No active models" }}
              </option>
              <option v-for="m in activeModels" :key="m.name" :value="m.name">
                {{ m.name }}{{ m.ready ? "" : " (not ready)" }}
              </option>
            </select>
          </label>
          <label v-if="modesA.has('prompt')">
            <span>Prompt A</span>
            <input v-model="perA.prompt" type="text" placeholder="a person" />
          </label>
          <label v-if="modesA.has('classes')">
            <span>Classes A</span>
            <input v-model="perA.classes" type="text" placeholder="person, car" />
          </label>
        </div>

        <div class="model-col">
          <label>
            <span>Model B</span>
            <select v-model="modelB" :disabled="activeModels.length === 0">
              <option v-if="activeModels.length === 0" value="">
                {{ loadingModels ? "Loading models" : "No active models" }}
              </option>
              <option v-for="m in activeModels" :key="m.name" :value="m.name">
                {{ m.name }}{{ m.ready ? "" : " (not ready)" }}
              </option>
            </select>
          </label>
          <label v-if="modesB.has('prompt')">
            <span>Prompt B</span>
            <input v-model="perB.prompt" type="text" placeholder="a person" />
          </label>
          <label v-if="modesB.has('classes')">
            <span>Classes B</span>
            <input v-model="perB.classes" type="text" placeholder="person, car" />
          </label>
        </div>
      </section>

      <!-- shared controls -->
      <section class="field-group">
        <label>
          <span>Confidence {{ shared.conf.toFixed(2) }}</span>
          <input v-model.number="shared.conf" type="range" min="0" max="1" step="0.01" />
        </label>
        <label>
          <span>Overlay</span>
          <select v-model="shared.overlay">
            <option v-for="opt in overlayOptions" :key="opt.value" :value="opt.value">
              {{ opt.label }}
            </option>
          </select>
        </label>
      </section>

      <section v-if="task === 'segment'" class="field-group">
        <span class="field-label">Output formats</span>
        <div class="checks">
          <label v-for="fmt in ['mask_png', 'rle', 'polygon', 'alpha_matte']" :key="fmt">
            <input v-model="shared.outputFormats" type="checkbox" :value="fmt" />
            <span>{{ fmt }}</span>
          </label>
        </div>
      </section>

      <button class="run-button" type="button" :disabled="!canRun" @click="runAB">
        <Play :size="18" />
        <span>{{ running ? "Running" : "Run A/B" }}</span>
      </button>
      <p v-if="runError" class="inline-error">{{ runError }}</p>
    </aside>

    <section class="workspace">
      <div v-if="imageUrl" class="preview-band">
        <PromptCanvas
          v-if="supportsVisualPrompt"
          :src="imageUrl"
          :natural-size="naturalSize"
          :points="shared.points"
          :boxes="shared.boxes"
          :mode="promptMode"
          :point-label="pointLabel"
          @image-size="(s) => (naturalSize = s)"
          @add-point="addPoint"
          @add-box="addBox"
        />
        <div v-else class="plain-preview">
          <img :src="imageUrl" alt="Uploaded preview" @load="onPreviewImageLoad" />
        </div>

        <div v-if="supportsVisualPrompt" class="prompt-toolbar">
          <div class="segmented compact">
            <button
              v-if="supportsPoints"
              :class="{ active: promptMode === 'point' }"
              type="button"
              title="Point prompt"
              @click="promptMode = 'point'"
            >
              <MousePointer2 :size="16" />
            </button>
            <button
              v-if="supportsBoxes"
              :class="{ active: promptMode === 'box' }"
              type="button"
              title="Box prompt"
              @click="promptMode = 'box'"
            >
              <BoxSelect :size="16" />
            </button>
          </div>
          <div v-if="supportsPoints" class="segmented compact">
            <button :class="{ active: pointLabel === 1 }" type="button" @click="pointLabel = 1">FG</button>
            <button :class="{ active: pointLabel === 0 }" type="button" @click="pointLabel = 0">BG</button>
          </div>
          <button class="icon-text" type="button" @click="clearPrompts">
            <Trash2 :size="16" />
            <span>Clear</span>
          </button>
          <div class="prompt-counts">
            <span v-if="supportsPoints">{{ shared.points.length }} pts</span>
            <span v-if="supportsBoxes">{{ shared.boxes.length }} boxes</span>
          </div>
        </div>

        <div
          v-if="supportsVisualPrompt && (shared.points.length || shared.boxes.length)"
          class="prompt-lists"
        >
          <button
            v-for="(pt, i) in shared.points"
            :key="`pt-${i}`"
            type="button"
            @click="removePoint(i)"
          >
            P{{ i + 1 }} {{ pt.label ? "FG" : "BG" }} {{ pt.x.toFixed(0) }},{{ pt.y.toFixed(0) }}
          </button>
          <button
            v-for="(bx, i) in shared.boxes"
            :key="`bx-${i}`"
            type="button"
            @click="removeBox(i)"
          >
            B{{ i + 1 }} {{ bx.x1.toFixed(0) }},{{ bx.y1.toFixed(0) }}
          </button>
        </div>
      </div>

      <div v-else class="empty-workspace">
        <Upload :size="28" />
        <span>Upload an image to begin</span>
      </div>

      <section class="results-grid">
        <ResultPane title="Model A" :model="modelA" :result="resultA" :fallback-image-url="imageUrl" />
        <ResultPane title="Model B" :model="modelB" :result="resultB" :fallback-image-url="imageUrl" />
      </section>
    </section>
  </main>
</template>
