<script setup lang="ts">
import { BoxSelect, MousePointer2, Trash2, Upload } from "lucide-vue-next";
import PromptCanvas from "./PromptCanvas.vue";
import type { PromptBox, PromptPoint } from "../api";
import type { NaturalSize } from "../promptGeometry";

defineProps<{
  imageUrl: string;
  naturalSize: NaturalSize;
  points: PromptPoint[];
  boxes: PromptBox[];
  promptMode: "point" | "box";
  pointLabel: 0 | 1;
  supportsVisualPrompt: boolean;
  supportsPoints: boolean;
  supportsBoxes: boolean;
}>();

const emit = defineEmits<{
  "update:naturalSize": [size: NaturalSize];
  "update:promptMode": [mode: "point" | "box"];
  "update:pointLabel": [label: 0 | 1];
  "add-point": [point: PromptPoint];
  "add-box": [box: PromptBox];
  "remove-point": [index: number];
  "remove-box": [index: number];
  "clear-prompts": [];
}>();

function onPreviewImageLoad(event: Event) {
  const img = event.target as HTMLImageElement;
  emit("update:naturalSize", { width: img.naturalWidth, height: img.naturalHeight });
}
</script>

<template>
  <div v-if="imageUrl" class="preview-band">
    <PromptCanvas
      v-if="supportsVisualPrompt"
      :src="imageUrl"
      :natural-size="naturalSize"
      :points="points"
      :boxes="boxes"
      :mode="promptMode"
      :point-label="pointLabel"
      @image-size="(s) => emit('update:naturalSize', s)"
      @add-point="(p) => emit('add-point', p)"
      @add-box="(b) => emit('add-box', b)"
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
          @click="emit('update:promptMode', 'point')"
        >
          <MousePointer2 :size="16" />
        </button>
        <button
          v-if="supportsBoxes"
          :class="{ active: promptMode === 'box' }"
          type="button"
          title="Box prompt"
          @click="emit('update:promptMode', 'box')"
        >
          <BoxSelect :size="16" />
        </button>
      </div>
      <div v-if="supportsPoints" class="segmented compact">
        <button :class="{ active: pointLabel === 1 }" type="button" @click="emit('update:pointLabel', 1)">FG</button>
        <button :class="{ active: pointLabel === 0 }" type="button" @click="emit('update:pointLabel', 0)">BG</button>
      </div>
      <button class="icon-text" type="button" @click="emit('clear-prompts')">
        <Trash2 :size="16" />
        <span>Clear</span>
      </button>
      <div class="prompt-counts">
        <span v-if="supportsPoints">{{ points.length }} pts</span>
        <span v-if="supportsBoxes">{{ boxes.length }} boxes</span>
      </div>
    </div>

    <div v-if="supportsVisualPrompt && (points.length || boxes.length)" class="prompt-lists">
      <button v-for="(pt, i) in points" :key="`pt-${i}`" type="button" @click="emit('remove-point', i)">
        P{{ i + 1 }} {{ pt.label ? "FG" : "BG" }} {{ pt.x.toFixed(0) }},{{ pt.y.toFixed(0) }}
      </button>
      <button v-for="(bx, i) in boxes" :key="`bx-${i}`" type="button" @click="emit('remove-box', i)">
        B{{ i + 1 }} {{ bx.x1.toFixed(0) }},{{ bx.y1.toFixed(0) }}
      </button>
    </div>
  </div>

  <div v-else class="empty-workspace">
    <Upload :size="28" />
    <span>Upload an image to begin</span>
  </div>
</template>
