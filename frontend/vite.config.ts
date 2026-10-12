import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies the gateway so the browser talks to one origin.
const GATEWAY = process.env.VITE_GATEWAY ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      "/api": GATEWAY,
      "/ws": { target: GATEWAY.replace(/^http/, "ws"), ws: true },
    },
  },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 1200 },
});
