import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Vite dev server proxies HTTP + WebSocket to the Flask backend on :5000.
// Build output goes to ../web/static/dist so Flask can serve it in production.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "src") },
  },
  base: "/",
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:5000", changeOrigin: true },
      "/static": { target: "http://127.0.0.1:5000", changeOrigin: true },
      "/socket.io": {
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
        ws: true,
      },
    },
  },
  build: {
    outDir: "../web/static/dist",
    emptyOutDir: true,
    sourcemap: true,
  },
});
