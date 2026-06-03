<script setup lang="ts">
import { computed } from "vue";
import type { PanelResult } from "../compareRunner";
import type { DetectionItem, VisionPayload } from "../api";

const props = defineProps<{
  title: string;
  model: string;
  result: PanelResult | null;
  fallbackImageUrl: string;
}>();

const payload = computed<VisionPayload | null>(() => {
  if (!props.result || props.result.state !== "done") {
    return null;
  }
  return props.result.payload;
});

const detections = computed<DetectionItem[]>(() => payload.value?.detections || []);
const imageUrl = computed(() => payload.value?.overlay_url || props.fallbackImageUrl);

const statusText = computed(() => {
  if (!props.result) {
    return "Idle";
  }
  if (props.result.state === "done") {
    return payload.value?.cached ? "Cached" : "Done";
  }
  if (props.result.state === "queued") {
    return "Queued";
  }
  return "Error";
});
</script>

<template>
  <section class="result-pane">
    <header class="result-head">
      <div>
        <p class="eyebrow">{{ title }}</p>
        <h2>{{ model || "No model" }}</h2>
      </div>
      <span class="status-pill" :class="result?.state || 'idle'">{{ statusText }}</span>
    </header>

    <div class="result-image">
      <img v-if="imageUrl" :src="imageUrl" alt="Result preview" />
      <div v-else class="empty-image">No image</div>
    </div>

    <div v-if="result?.state === 'error'" class="error-box">
      <strong>{{ result.error.code }}</strong>
      <span>{{ result.error.message }}</span>
    </div>

    <div v-else-if="result?.state === 'queued'" class="queued-box">
      <strong>{{ result.jobId || result.payload.job_id }}</strong>
      <span>{{ result.payload.status || "queued" }}</span>
    </div>

    <div v-else-if="payload" class="result-meta">
      <div class="metric-row">
        <span>Detections</span>
        <strong>{{ detections.length }}</strong>
      </div>
      <div class="metric-row">
        <span>Request</span>
        <strong>{{ payload.request_id || result?.jobId || "-" }}</strong>
      </div>
      <div v-if="payload.timing_ms" class="timing-grid">
        <div v-for="(value, key) in payload.timing_ms" :key="key">
          <span>{{ key }}</span>
          <strong>{{ value }} ms</strong>
        </div>
      </div>
      <div class="detections-list">
        <article v-for="item in detections" :key="item.id" class="detection-row">
          <div>
            <strong>{{ item.class_name || `Mask ${item.id}` }}</strong>
            <span>{{ item.score.toFixed(3) }}</span>
          </div>
          <code>[{{ item.bbox.map((value) => value.toFixed(1)).join(", ") }}]</code>
        </article>
      </div>
    </div>
  </section>
</template>
