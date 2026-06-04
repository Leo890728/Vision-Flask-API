<script setup lang="ts">
import type { PlaygroundView } from "../usePlayground";

defineProps<{ view: PlaygroundView }>();
const emit = defineEmits<{ change: [view: PlaygroundView] }>();
</script>

<template>
  <div class="segmented mode-tabs">
    <button type="button" :class="{ active: view === 'compare' }" @click="emit('change', 'compare')">
      A/B Compare
    </button>
    <button type="button" :class="{ active: view === 'single' }" @click="emit('change', 'single')">
      Single Model
    </button>
  </div>
</template>
