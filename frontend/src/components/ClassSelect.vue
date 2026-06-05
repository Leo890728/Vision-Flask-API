<script setup lang="ts">
import { computed, ref } from "vue";

const props = defineProps<{
  label: string;
  modelValue: string[];
  classNames: string[] | null;
}>();

const emit = defineEmits<{ "update:modelValue": [value: string[]] }>();

// ── checkbox mode (classNames provided) ────────────────────────────────────
const search = ref("");

const filtered = computed(() => {
  if (!props.classNames) return [];
  const q = search.value.trim().toLowerCase();
  return q ? props.classNames.filter((n) => n.toLowerCase().includes(q)) : props.classNames;
});

function toggle(name: string) {
  const next = props.modelValue.includes(name)
    ? props.modelValue.filter((n) => n !== name)
    : [...props.modelValue, name];
  emit("update:modelValue", next);
}

function selectAll() {
  emit("update:modelValue", props.classNames ? [...props.classNames] : []);
}

function clearAll() {
  emit("update:modelValue", []);
}

// ── text input fallback ────────────────────────────────────────────────────
function onTextInput(e: Event) {
  const raw = (e.target as HTMLInputElement).value;
  emit("update:modelValue", raw ? raw.split(",").map((s) => s.trim()).filter(Boolean) : []);
}

const textValue = computed(() => props.modelValue.join(", "));
</script>

<template>
  <div class="class-select">
    <span class="field-label">{{ label }}</span>

    <!-- text fallback when API hasn't returned class names yet -->
    <template v-if="!classNames">
      <input type="text" :value="textValue" placeholder="person, car" @input="onTextInput" />
    </template>

    <!-- checkbox list when class names are known -->
    <template v-else>
      <div class="class-select-header">
        <input
          v-model="search"
          class="class-search"
          type="text"
          placeholder="Search…"
        />
        <div class="class-select-actions">
          <button type="button" @click="selectAll">All</button>
          <button type="button" @click="clearAll">Clear</button>
        </div>
      </div>
      <div class="class-list">
        <label
          v-for="name in filtered"
          :key="name"
          class="class-item"
          :class="{ selected: modelValue.includes(name) }"
        >
          <input
            type="checkbox"
            :value="name"
            :checked="modelValue.includes(name)"
            @change="toggle(name)"
          />
          <span>{{ name }}</span>
        </label>
        <p v-if="filtered.length === 0" class="class-empty">No match</p>
      </div>
      <p class="class-count">{{ modelValue.length }} / {{ classNames.length }} selected</p>
    </template>
  </div>
</template>
