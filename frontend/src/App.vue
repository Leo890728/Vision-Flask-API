<script setup lang="ts">
import { provide, ref } from "vue";
import ComparePlayground from "./ComparePlayground.vue";
import SinglePlayground from "./SinglePlayground.vue";
import { playgroundKey, usePlayground, type PlaygroundView } from "./usePlayground";

// One shared playground instance keeps the API key, uploaded image and prompts
// alive while switching between the A/B and single-model views.
provide(playgroundKey, usePlayground());

const view = ref<PlaygroundView>("compare");
</script>

<template>
  <ComparePlayground v-if="view === 'compare'" :view="view" @change="view = $event" />
  <SinglePlayground v-else :view="view" @change="view = $event" />
</template>
