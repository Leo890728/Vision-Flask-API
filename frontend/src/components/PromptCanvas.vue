<script setup lang="ts">
import { computed, ref } from "vue";
import type { NaturalSize } from "../promptGeometry";
import type { PromptBox, PromptPoint } from "../api";
import {
  displayBoxToNatural,
  displayToNaturalPoint,
  naturalBoxToDisplay,
  naturalPointToDisplay
} from "../promptGeometry";

const props = defineProps<{
  src: string;
  naturalSize: NaturalSize;
  points: PromptPoint[];
  boxes: PromptBox[];
  mode: "point" | "box";
  pointLabel: 0 | 1;
  disabled?: boolean;
}>();

const emit = defineEmits<{
  "add-point": [point: PromptPoint];
  "add-box": [box: PromptBox];
  "image-size": [size: NaturalSize];
}>();

const imageEl = ref<HTMLImageElement | null>(null);
const dragStart = ref<{ x: number; y: number } | null>(null);
const dragEnd = ref<{ x: number; y: number } | null>(null);

const pointViews = computed(() =>
  props.points.map((point) => ({
    ...naturalPointToDisplay(point, props.naturalSize),
    label: point.label
  }))
);

const boxViews = computed(() => props.boxes.map((box) => naturalBoxToDisplay(box, props.naturalSize)));

const previewBox = computed(() => {
  if (!dragStart.value || !dragEnd.value || !imageEl.value) {
    return null;
  }
  const rect = imageEl.value.getBoundingClientRect();
  const natural = displayBoxToNatural(
    dragStart.value.x,
    dragStart.value.y,
    dragEnd.value.x,
    dragEnd.value.y,
    rect,
    props.naturalSize
  );
  return naturalBoxToDisplay(natural, props.naturalSize);
});

function onImageLoad() {
  if (!imageEl.value) {
    return;
  }
  emit("image-size", {
    width: imageEl.value.naturalWidth,
    height: imageEl.value.naturalHeight
  });
}

function onClick(event: MouseEvent) {
  if (props.disabled || props.mode !== "point" || !imageEl.value) {
    return;
  }
  const rect = imageEl.value.getBoundingClientRect();
  emit("add-point", displayToNaturalPoint(event.clientX, event.clientY, rect, props.naturalSize, props.pointLabel));
}

function onPointerDown(event: PointerEvent) {
  if (props.disabled || props.mode !== "box") {
    return;
  }
  dragStart.value = { x: event.clientX, y: event.clientY };
  dragEnd.value = { x: event.clientX, y: event.clientY };
}

function onPointerMove(event: PointerEvent) {
  if (!dragStart.value || props.mode !== "box") {
    return;
  }
  dragEnd.value = { x: event.clientX, y: event.clientY };
}

function onPointerUp(event: PointerEvent) {
  if (props.disabled || props.mode !== "box" || !dragStart.value || !imageEl.value) {
    resetDrag();
    return;
  }
  const rect = imageEl.value.getBoundingClientRect();
  const box = displayBoxToNatural(
    dragStart.value.x,
    dragStart.value.y,
    event.clientX,
    event.clientY,
    rect,
    props.naturalSize
  );
  resetDrag();
  if (Math.abs(box.x2 - box.x1) >= 2 && Math.abs(box.y2 - box.y1) >= 2) {
    emit("add-box", box);
  }
}

function resetDrag() {
  dragStart.value = null;
  dragEnd.value = null;
}
</script>

<template>
  <div class="prompt-canvas" :class="{ 'is-disabled': disabled }">
    <img ref="imageEl" :src="src" alt="Uploaded preview" @load="onImageLoad" />
    <svg
      class="prompt-overlay"
      @click="onClick"
      @pointerdown="onPointerDown"
      @pointermove="onPointerMove"
      @pointerup="onPointerUp"
      @pointerleave="resetDrag"
    >
      <rect
        v-for="(box, index) in boxViews"
        :key="`box-${index}`"
        class="prompt-box"
        :x="`${box.left}%`"
        :y="`${box.top}%`"
        :width="`${box.width}%`"
        :height="`${box.height}%`"
      />
      <rect
        v-if="previewBox"
        class="prompt-box preview"
        :x="`${previewBox.left}%`"
        :y="`${previewBox.top}%`"
        :width="`${previewBox.width}%`"
        :height="`${previewBox.height}%`"
      />
      <circle
        v-for="(point, index) in pointViews"
        :key="`point-${index}`"
        class="prompt-point"
        :class="{ negative: point.label === 0 }"
        :cx="`${point.x}%`"
        :cy="`${point.y}%`"
        r="7"
      />
    </svg>
  </div>
</template>
