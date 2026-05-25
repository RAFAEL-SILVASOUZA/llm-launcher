/**
 * Electron main process.
 *
 *   • Spawneia o backend Python (api/server.py)
 *   • Captura a porta efetiva via stdout (linha "PORT=N") OU via fallback
 *     no arquivo api_port.tmp escrito pelo server.
 *   • Em dev (ELECTRON_START_URL setado por `npm run dev:electron`), carrega
 *     a URL do Vite. Em prod (build), carrega dist/index.html.
 *   • Mata o subprocesso quando todas as janelas fecham.
 */
const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const fs   = require("fs");
const { spawn } = require("child_process");

const DEV_URL    = process.env.ELECTRON_START_URL;
const PROJECT_DIR = path.join(__dirname, "..");
const PORT_FILE   = path.join(PROJECT_DIR, "api_port.tmp");

let apiProcess = null;
let apiPort    = null;
const portWaiters = [];

function notifyPort(port) {
  apiPort = port;
  while (portWaiters.length) {
    const resolve = portWaiters.shift();
    resolve(port);
  }
}

function waitForPort() {
  return new Promise((resolve) => {
    if (apiPort) return resolve(apiPort);
    portWaiters.push(resolve);
  });
}

function startApiServer() {
  // Em dev, executar `python api/server.py` direto. Em prod (empacotado),
  // o mesmo caminho relativo continua válido — o electron-builder inclui
  // a pasta `api/` no resources, e PROJECT_DIR aponta pra ela.
  const apiPath = path.join(PROJECT_DIR, "api", "server.py");
  const pythonCmd = process.env.PYTHON || "python";

  // Limpa porta antiga pra evitar pegar valor stale
  try { fs.unlinkSync(PORT_FILE); } catch { /* ignore */ }

  apiProcess = spawn(pythonCmd, [apiPath], {
    cwd: PROJECT_DIR,
    env: { ...process.env, PYTHONIOENCODING: "utf-8" },
    stdio: ["pipe", "pipe", "pipe"],
  });

  let stdoutBuf = "";
  apiProcess.stdout.on("data", (data) => {
    const text = data.toString("utf-8");
    stdoutBuf += text;
    process.stdout.write(`[api] ${text}`);
    if (apiPort == null) {
      const m = stdoutBuf.match(/PORT=(\d+)/);
      if (m) {
        notifyPort(parseInt(m[1], 10));
      }
    }
  });
  apiProcess.stderr.on("data", (data) => {
    process.stderr.write(`[api:err] ${data.toString("utf-8")}`);
  });
  apiProcess.on("error", (err) => {
    console.error("[api] spawn error:", err);
  });
  apiProcess.on("exit", (code, signal) => {
    console.log(`[api] exited code=${code} signal=${signal}`);
    apiProcess = null;
  });

  // Fallback: o file watcher pega a porta se o stdout for buffered/perdido.
  const startedAt = Date.now();
  const tryReadFile = () => {
    if (apiPort) return;
    if (Date.now() - startedAt > 30_000) return;  // timeout 30s
    try {
      if (fs.existsSync(PORT_FILE)) {
        const txt = fs.readFileSync(PORT_FILE, "utf-8").trim();
        const port = parseInt(txt, 10);
        if (Number.isFinite(port) && port > 0) {
          notifyPort(port);
          return;
        }
      }
    } catch { /* ignore */ }
    setTimeout(tryReadFile, 200);
  };
  setTimeout(tryReadFile, 200);
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    backgroundColor: "#0f1116",
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, "preload.js"),
    },
  });

  if (DEV_URL) {
    win.loadURL(DEV_URL);
    win.webContents.openDevTools({ mode: "detach" });
  } else {
    win.loadFile(path.join(PROJECT_DIR, "dist", "index.html"));
  }
  return win;
}

// IPC handler que o renderer chama via preload pra descobrir a porta da API.
ipcMain.handle("api:base-url", async () => {
  const port = await waitForPort();
  return `http://127.0.0.1:${port}`;
});

app.whenReady().then(() => {
  startApiServer();
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (apiProcess) {
    try { apiProcess.kill(); } catch { /* ignore */ }
  }
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  if (apiProcess) {
    try { apiProcess.kill(); } catch { /* ignore */ }
  }
});
