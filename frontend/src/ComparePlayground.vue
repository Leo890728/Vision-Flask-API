<script setup lang="ts">
import { computed, inject, reactive, ref, watch } from "vue";
import { DatabaseZap, KeyRound, Play, RefreshCw, Upload } from "lucide-vue-next";
import ModeTabs from "./components/ModeTabs.vue";
import PreviewBand from "./components/PreviewBand.vue";
import ResultPane from "./components/ResultPane.vue";
import type { ModelInfo } from "./api";
import type { PanelResult } from "./compareRunner";
import { runComparison } from "./compareRunner";
import { playgroundKey, type PlaygroundView } from "./usePlayground";

defineProps<{ view: PlaygroundView }>();
const emit = defineEmits<{ change: [view: PlaygroundView] }>();

const pg = inject(playgroundKey)!;
const {
  apiKey,
  catalogError,
  loadingModels,
  task,
  imageFile,
  imageUrl,
  naturalSize,
  promptMode,
  pointLabel,
  shared,
  catalog,
  activeModels,
  overlayOptions,
  loadModels,
  onFileChange,
  addPoint,
  addBox,
  removePoint,
  removeBox,
  clearPrompts,
  modesOf,
  buildControls
} = pg;

// ── per-side selection & results ───────────────────────────────────────────
const modelA = ref("");
const modelB = ref("");
const running = ref(false);
const runError = ref("");
const resultA = ref<PanelResult | null>(null);
const resultB = ref<PanelResult | null>(null);

// per-model text inputs (prompt / classes)
const perA = reactive({ prompt: "", classes: "" });
const perB = reactive({ prompt: "", classes: "" });

const modelAInfo = computed<ModelInfo | undefined>(() =>
  activeModels.value.find((m) => m.name === modelA.value)
);
const modelBInfo = computed<ModelInfo | undefined>(() =>
  activeModels.value.find((m) => m.name === modelB.value)
);

const modesA = computed(() => modesOf(modelAInfo.value));
const modesB = computed(() => modesOf(modelBInfo.value));

// union — drives shared canvas / toolbar
const anyModes = computed<Set<string>>(() => new Set([...modesA.value, ...modesB.value]));
const supportsVisualPrompt = computed(() => anyModes.value.has("points") || anyModes.value.has("boxes"));
const supportsPoints = computed(() => anyModes.value.has("points"));
const supportsBoxes = computed(() => anyModes.value.has("boxes"));

const canRun = computed(() =>
  Boolean(apiKey.value && imageFile.value && modelA.value && modelB.value && !running.value)
);

function clearResults() {
  resultA.value = null;
  resultB.value = null;
  runError.value = "";
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

watch(activeModels, syncSelectedModels, { immediate: true });
watch([task, imageFile], clearResults);

// drop visual prompts neither model supports
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

      <ModeTabs :view="view" @change="emit('change', $event)" />

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
          <button :class="{ active: task === 'detect' }" type="button" @click="task = 'detect'">Detect</button>
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
      <PreviewBand
        :image-url="imageUrl"
        v-model:natural-size="naturalSize"
        :points="shared.points"
        :boxes="shared.boxes"
        v-model:prompt-mode="promptMode"
        v-model:point-label="pointLabel"
        :supports-visual-prompt="supportsVisualPrompt"
        :supports-points="supportsPoints"
        :supports-boxes="supportsBoxes"
        @add-point="addPoint"
        @add-box="addBox"
        @remove-point="removePoint"
        @remove-box="removeBox"
        @clear-prompts="clearPrompts"
      />

      <section class="results-grid">
        <ResultPane title="Model A" :model="modelA" :result="resultA" :fallback-image-url="imageUrl" />
        <ResultPane title="Model B" :model="modelB" :result="resultB" :fallback-image-url="imageUrl" />
      </section>
    </section>
  </main>
</template>
