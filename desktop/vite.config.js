import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiTarget = process.env.WEBUI_API_PROXY_TARGET || "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  build: {
    outDir: "../agent/webui/dist",
    emptyOutDir: true
  },
  server: {
    port: 1420,
    strictPort: true,
    proxy: {
      "/health": apiTarget,
      "/models": apiTarget,
      "/chat": apiTarget,
      "/config": apiTarget,
      "/defaults": apiTarget,
      "/llm": apiTarget,
      "/providers": apiTarget,
      "/telegram": apiTarget,
      "/permissions": apiTarget,
      "/audit": apiTarget,
      "/modelops": apiTarget
    }
  }
});
