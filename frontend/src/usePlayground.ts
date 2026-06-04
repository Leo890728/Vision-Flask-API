import { computed, onMounted, reactive, ref, watch, type InjectionKey } from "vue";
import type {
  ModelCatalog,
  ModelInfo,
  PromptBox,
  PromptPoint,
  Task,
  VisionControls
} from "./api";
import { fetchModelCatalog } from "./api";
import type { NaturalSize } from "./promptGeometry";

export type PlaygroundView = "compare" | "single";

export interface PerModelInput {
  prompt: string;
  classes: string;
}

/**
 * Shared playground state: API key, model catalog, image upload, prompt canvas
 * inputs and the controls common to every model. Instantiated once at the app
 * root and provided to each page so that the uploaded image, prompts and key
 * survive switching between the A/B and single-model views.
 */
export function usePlayground() {
  const apiKey = ref(sessionStorage.getItem("vision-api-key") || "");
  const catalog = ref<ModelCatalog | null>(null);
  const catalogError = ref("");
  const loadingModels = ref(false);
  const task = ref<Task>("segment");

  const imageFile = ref<File | null>(null);
  const imageUrl = ref("");
  const naturalSize = ref<NaturalSize>({ width: 0, height: 0 });

  const promptMode = ref<"point" | "box">("point");
  const pointLabel = ref<0 | 1>(1);

  // controls applied to every model in the current run
  const shared = reactive({
    conf: 0.25,
    overlay: "both",
    outputFormats: ["mask_png"] as string[],
    points: [] as PromptPoint[],
    boxes: [] as PromptBox[]
  });

  const activeModels = computed<ModelInfo[]>(() =>
    (catalog.value?.models ?? []).filter((m) => m.task === task.value && m.active)
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

  // ── model catalog loading ────────────────────────────────────────────────
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

  watch(apiKey, (value) => {
    if (value) {
      sessionStorage.setItem("vision-api-key", value);
      scheduleLoadModels();
    } else {
      sessionStorage.removeItem("vision-api-key");
      catalog.value = null;
    }
  });

  watch(task, () => {
    shared.overlay = task.value === "detect" ? "bbox" : "both";
  });

  onMounted(() => {
    if (apiKey.value) loadModels();
  });

  // ── image upload ───────────────────────────────────────────────────────────
  function onFileChange(event: Event) {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (!file) return;
    if (imageUrl.value) URL.revokeObjectURL(imageUrl.value);
    imageFile.value = file;
    imageUrl.value = URL.createObjectURL(file);
  }

  // ── prompt canvas helpers ────────────────────────────────────────────────
  function addPoint(point: PromptPoint) { shared.points.push(point); }
  function addBox(box: PromptBox)       { shared.boxes.push(box); }
  function removePoint(i: number)       { shared.points.splice(i, 1); }
  function removeBox(i: number)         { shared.boxes.splice(i, 1); }
  function clearPrompts() {
    shared.points.splice(0);
    shared.boxes.splice(0);
  }

  function modesOf(info: ModelInfo | undefined): Set<string> {
    return new Set(info?.input_modes ?? []);
  }

  function buildControls(per: PerModelInput): VisionControls {
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

  return {
    apiKey,
    catalog,
    catalogError,
    loadingModels,
    task,
    imageFile,
    imageUrl,
    naturalSize,
    promptMode,
    pointLabel,
    shared,
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
  };
}

export type Playground = ReturnType<typeof usePlayground>;

export const playgroundKey: InjectionKey<Playground> = Symbol("playground");
