import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";
import fs from "node:fs";
import path from "node:path";

const projectRoot = fileURLToPath(new URL("..", import.meta.url));

// Pin root to this file's directory so it works regardless of the launch cwd.
export default defineConfig({
  root: fileURLToPath(new URL(".", import.meta.url)),
  plugins: [
    react(),
    {
      name: "save-snapshot-middleware",
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url === "/api/save-snapshot" && req.method === "POST") {
            let body = "";
            req.on("data", chunk => { body += chunk; });
            req.on("end", () => {
              try {
                const data = JSON.parse(body);
                const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
                const filename = `snapshot_${timestamp}.json`;
                const filepath = path.join(projectRoot, filename);
                fs.writeFileSync(filepath, JSON.stringify(data, null, 2), "utf8");
                res.writeHead(200, { "Content-Type": "application/json" });
                res.end(JSON.stringify({ success: true, filename, filepath }));
              } catch (e) {
                res.writeHead(500, { "Content-Type": "application/json" });
                res.end(JSON.stringify({ success: false, error: e.message }));
              }
            });
          } else {
            next();
          }
        });
      }
    }
  ],
  server: { port: 5188, strictPort: true, host: "127.0.0.1" },
});
