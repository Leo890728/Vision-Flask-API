<script setup lang="ts">
import { computed, inject, reactive, ref, watch } from "vue";
import { DatabaseZap, KeyRound, Play, RefreshCw, Upload } from "lucide-vue-next";
import ClassSelect from "./components/ClassSelect.vue";
import ModeTabs from "./components/ModeTabs.vue";
import PreviewBand from "./components/PreviewBand.vue";
import ResultPane from "./components/ResultPane.vue";
import type { ModelInfo } from "./api";
import type { PanelResult } from "./compareRunner";
import { runPanel } from "./compareRunner";
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
  buildControls,
  ensureClassNames
} = pg;

// ── selection & result ─────────────────────────────────────────────────────
const model = ref("");
const running = ref(false);
const runError = ref("");
const result = ref<PanelResult | null>(null);

// text inputs (prompt / classes) for the selected model
const per = reactive({ prompt: "", classes: [] as string[] });

const modelInfo = computed<ModelInfo | undefined>(() =>
  activeModels.value.find((m) => m.name === model.value)
);

const modes = computed(() => modesOf(modelInfo.value));
const supportsVisualPrompt = computed(() => modes.value.has("points") || modes.value.has("boxes"));
const supportsPoints = computed(() => modes.value.has("points"));
const supportsBoxes = computed(() => modes.value.has("boxes"));

const canRun = computed(() =>
  Boolean(apiKey.value && imageFile.value && model.value && !running.value)
);

function clearResult() {
  result.value = null;
  runError.value = "";
}

function syncSelectedModel() {
  const models = activeModels.value;
  if (models.length === 0) {
    model.value = "";
    return;
  }
  const defaultName = catalog.value?.defaults[task.value];
  model.value = models.find((m) => m.name === defaultName)?.name ?? models[0].name;
  shared.conf = models.find((m) => m.name === model.value)?.default_conf ?? shared.conf;
}

watch(activeModels, syncSelectedModel, { immediate: true });
watch([task, imageFile], clearResult);
watch(model, (name) => { if (name) ensureClassNames(name); });

// drop visual prompts the selected model does not support
watch(model, () => {
  if (!supportsPoints.value) {
    shared.points.splice(0);
    if (promptMode.value === "point") promptMode.value = "box";
  }
  if (!supportsBoxes.value) {
    shared.boxes.splice(0);
    if (promptMode.value === "box") promptMode.value = "point";
  }
});

async function run() {
  if (!canRun.value || !imageFile.value) return;
  running.value = true;
  runError.value = "";
  result.value = null;
  try {
    result.value = await runPanel(
      {
        side: "A",
        task: task.value,
        model: model.value,
        inputModes: modelInfo.value?.input_modes ?? [],
        image: imageFile.value,
        filename: imageFile.value.name,
        controls: buildControls(per)
      },
      apiKey.value,
      { pollIntervalMs: 1000 }
    );
  } catch (error) {
    runError.value = error instanceof Error ? error.message : "Run failed";
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

      <!-- model selector + inputs -->
      <section class="field-group">
        <label>
          <span>Model</span>
          <select v-model="model" :disabled="activeModels.length === 0">
            <option v-if="activeModels.length === 0" value="">
              {{ loadingModels ? "Loading models" : "No active models" }}
            </option>
            <option v-for="m in activeModels" :key="m.name" :value="m.name">
              {{ m.name }}{{ m.ready ? "" : " (not ready)" }}
            </option>
          </select>
        </label>
        <label v-if="modes.has('prompt')">
          <span>Prompt</span>
          <input v-model="per.prompt" type="text" placeholder="a person" />
        </label>
        <ClassSelect
          v-if="modes.has('classes')"
          label="Classes"
          v-model="per.classes"
          :class-names="modelInfo?.class_names ?? null"
        />
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

      <button class="run-button" type="button" :disabled="!canRun" @click="run">
        <Play :size="18" />
        <span>{{ running ? "Running" : "Run" }}</span>
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

      <section class="results-grid single">
        <ResultPane title="Model" :model="model" :result="result" :fallback-image-url="imageUrl" />
      </section>
    </section>
  </main>
</template>
