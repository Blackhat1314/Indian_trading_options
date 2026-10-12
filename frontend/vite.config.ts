import { resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies the gateway so the browser talks to one origin.
const GATEWAY = process.env.VITE_GATEWAY ?? "http://127.0.0.1:8000";

/** Three pages: the terminal (index.html), /login and /admin — like the gateway serves them in production. */
const pages = (): Plugin => ({
  name: "td-pages",
  configureServer(server) {
    server.middlewares.use((req, _res, next) => {
      const path = (req.url ?? "").split("?")[0];
      const first = path.split("/")[1];
      if (first === "login" || first === "admin") req.url = `/${first}.html${req.url!.slice(path.length)}`;
      next();
    });
  },
});

export default defineConfig({
  plugins: [react(), pages()],
  server: {
    port: 5173,
    host: true,
    // xfwd: the gateway must see LAN visitors of the dev server as remote (login throttling, first-admin setup)
    proxy: {
      "/api": { target: GATEWAY, xfwd: true },
      "/ws": { target: GATEWAY.replace(/^http/, "ws"), ws: true, xfwd: true },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    chunkSizeWarningLimit: 1200,
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        login: resolve(__dirname, "login.html"),
        admin: resolve(__dirname, "admin.html"),
      },
    },
  },
});
