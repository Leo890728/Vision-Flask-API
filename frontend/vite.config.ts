import { defineConfig } from "vitest/config";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  base: "/playground/",
  plugins: [vue()],
  test: {
    include: ["src/**/*.test.ts"]
  }
});
