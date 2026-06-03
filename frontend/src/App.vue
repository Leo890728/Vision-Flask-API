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

const controls = reactive<VisionControls>({
  conf: 0.25,
  classes: "",
  overlay: "both",
  outputFormats: ["mask_png"],
  prompt: "",
  points: [],
  boxes: []
});

const activeModels = computed<ModelInfo[]>(() =>
  (catalog.value?.models || []).filter((model) => model.task === task.value && model.active)
);

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

const usesSam3 = computed(() => task.value === "segment" && (modelA.value === "sam3" || modelB.value === "sam3"));
const canRun = computed(() => Boolean(apiKey.value && imageFile.value && modelA.value && modelB.value && !running.value));

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
  controls.overlay = task.value === "detect" ? "bbox" : "both";
  syncSelectedModels();
  clearResults();
});

onMounted(() => {
  if (apiKey.value) {
    loadModels();
  }
});

let modelLoadTimer: number | undefined;

function scheduleLoadModels() {
  window.clearTimeout(modelLoadTimer);
  modelLoadTimer = window.setTimeout(() => {
    loadModels();
  }, 350);
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
  modelA.value = models.find((model) => model.name === defaultName)?.name || models[0].name;
  modelB.value = models.find((model) => model.name !== modelA.value)?.name || modelA.value;
  controls.conf = models.find((model) => model.name === modelA.value)?.default_conf || controls.conf;
}

function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) {
    return;
  }
  if (imageUrl.value) {
    URL.revokeObjectURL(imageUrl.value);
  }
  imageFile.value = file;
  imageUrl.value = URL.createObjectURL(file);
  clearResults();
}

function onPreviewImageLoad(event: Event) {
  const image = event.target as HTMLImageElement;
  setNaturalSize({ width: image.naturalWidth, height: image.naturalHeight });
}

function setNaturalSize(size: NaturalSize) {
  naturalSize.value = size;
}

async function runAB() {
  if (!canRun.value || !imageFile.value) {
    return;
  }
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
          image: imageFile.value,
          filename: imageFile.value.name,
          controls
        },
        right: {
          task: task.value,
          model: modelB.value,
          image: imageFile.value,
          filename: imageFile.value.name,
          controls
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

function addPoint(point: PromptPoint) {
  controls.points.push(point);
}

function addBox(box: PromptBox) {
  controls.boxes.push(box);
}

function removePoint(index: number) {
  controls.points.splice(index, 1);
}

function removeBox(index: number) {
  controls.boxes.splice(index, 1);
}

function clearPrompts() {
  controls.points.splice(0);
  controls.boxes.splice(0);
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

      <section class="field-group">
        <label class="upload-box">
          <Upload :size="20" />
          <span>{{ imageFile?.name || "Upload image" }}</span>
          <input type="file" accept="image/png,image/jpeg,image/webp" @change="onFileChange" />
        </label>
      </section>

      <section class="field-group">
        <span class="field-label">Task</span>
        <div class="segmented">
          <button :class="{ active: task === 'segment' }" type="button" @click="task = 'segment'">Segment</button>
          <button :class="{ active: task === 'detect' }" type="button" @click="task = 'detect'">Detect</button>
        </div>
      </section>

      <section class="field-group two-col">
        <label>
          <span>Model A</span>
          <select v-model="modelA" :disabled="activeModels.length === 0">
            <option v-if="activeModels.length === 0" value="">
              {{ loadingModels ? "Loading models" : "No active models" }}
            </option>
            <option v-for="model in activeModels" :key="model.name" :value="model.name">
              {{ model.name }}{{ model.ready ? "" : " (not ready)" }}
            </option>
          </select>
        </label>
        <label>
          <span>Model B</span>
          <select v-model="modelB" :disabled="activeModels.length === 0">
            <option v-if="activeModels.length === 0" value="">
              {{ loadingModels ? "Loading models" : "No active models" }}
            </option>
            <option v-for="model in activeModels" :key="model.name" :value="model.name">
              {{ model.name }}{{ model.ready ? "" : " (not ready)" }}
            </option>
          </select>
        </label>
      </section>

      <section class="field-group">
        <label>
          <span>Confidence {{ controls.conf.toFixed(2) }}</span>
          <input v-model.number="controls.conf" type="range" min="0" max="1" step="0.01" />
        </label>
        <label>
          <span>Classes</span>
          <input v-model="controls.classes" type="text" placeholder="person, car or [0, 2]" />
        </label>
        <label>
          <span>Overlay</span>
          <select v-model="controls.overlay">
            <option v-for="option in overlayOptions" :key="option.value" :value="option.value">
              {{ option.label }}
            </option>
          </select>
        </label>
      </section>

      <section v-if="task === 'segment'" class="field-group">
        <label>
          <span>Prompt</span>
          <input v-model="controls.prompt" type="text" placeholder="a person" />
        </label>
        <div class="checks">
          <label v-for="format in ['mask_png', 'rle', 'polygon', 'alpha_matte']" :key="format">
            <input v-model="controls.outputFormats" type="checkbox" :value="format" />
            <span>{{ format }}</span>
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
          v-if="task === 'segment' && usesSam3"
          :src="imageUrl"
          :natural-size="naturalSize"
          :points="controls.points"
          :boxes="controls.boxes"
          :mode="promptMode"
          :point-label="pointLabel"
          @image-size="setNaturalSize"
          @add-point="addPoint"
          @add-box="addBox"
        />
        <div v-else class="plain-preview">
          <img :src="imageUrl" alt="Uploaded preview" @load="onPreviewImageLoad" />
        </div>

        <div v-if="task === 'segment' && usesSam3" class="prompt-toolbar">
          <div class="segmented compact">
            <button :class="{ active: promptMode === 'point' }" type="button" title="Point prompt" @click="promptMode = 'point'">
              <MousePointer2 :size="16" />
            </button>
            <button :class="{ active: promptMode === 'box' }" type="button" title="Box prompt" @click="promptMode = 'box'">
              <BoxSelect :size="16" />
            </button>
          </div>
          <div class="segmented compact">
            <button :class="{ active: pointLabel === 1 }" type="button" @click="pointLabel = 1">FG</button>
            <button :class="{ active: pointLabel === 0 }" type="button" @click="pointLabel = 0">BG</button>
          </div>
          <button class="icon-text" type="button" @click="clearPrompts">
            <Trash2 :size="16" />
            <span>Clear</span>
          </button>
          <div class="prompt-counts">
            <span>{{ controls.points.length }} pts</span>
            <span>{{ controls.boxes.length }} boxes</span>
          </div>
        </div>

        <div v-if="task === 'segment' && usesSam3 && (controls.points.length || controls.boxes.length)" class="prompt-lists">
          <button v-for="(point, index) in controls.points" :key="`point-${index}`" type="button" @click="removePoint(index)">
            P{{ index + 1 }} {{ point.label ? "FG" : "BG" }} {{ point.x.toFixed(0) }},{{ point.y.toFixed(0) }}
          </button>
          <button v-for="(box, index) in controls.boxes" :key="`box-${index}`" type="button" @click="removeBox(index)">
            B{{ index + 1 }} {{ box.x1.toFixed(0) }},{{ box.y1.toFixed(0) }}
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
