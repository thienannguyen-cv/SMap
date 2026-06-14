// Self-healing Vite launcher for the SMap preview.
// 1) Frees port 5188 if a stale/orphaned dev server is still holding it.
// 2) Runs Vite IN THIS process (not as a child) so the preview manager owns it
//    directly — stopping the server can't leave an orphaned child behind.
import { execSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const PORT = "5188";
const NETSTAT = "C:\\Windows\\System32\\netstat.exe";
const TASKKILL = "C:\\Windows\\System32\\taskkill.exe";

try {
  const out = execSync(`"${NETSTAT}" -ano`, { encoding: "utf8" });
  const re = new RegExp(":" + PORT + "\\b");
  const pids = [
    ...new Set(
      out
        .split("\n")
        .filter((l) => re.test(l) && /LISTENING/i.test(l))
        .map((l) => l.trim().split(/\s+/).pop())
    ),
  ].filter((p) => /^\d+$/.test(p) && p !== String(process.pid));
  for (const p of pids) {
    try {
      execSync(`"${TASKKILL}" /F /PID ${p}`);
      console.log(`[start] freed port ${PORT} (killed stale pid ${p})`);
    } catch {
      /* ignore */
    }
  }
} catch {
  /* netstat unavailable — fall through and let Vite report the bind error */
}

// Hand off to Vite's CLI in-process: set argv so its arg parser sees --config.
const viteBin = join(here, "node_modules", "vite", "bin", "vite.js");
const viteConfig = join(here, "vite.config.js");
process.argv = [process.argv[0], viteBin, "--config", viteConfig];
await import(pathToFileURL(viteBin).href);
