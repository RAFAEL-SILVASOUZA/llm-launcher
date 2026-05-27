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
const { app, BrowserWindow, ipcMain, dialog, Menu } = require("electron");
const path = require("path");
const fs   = require("fs");
const { spawn } = require("child_process");

const DEV_URL = process.env.ELECTRON_START_URL;
// Em packaged builds (asar: false), __dirname fica em
// <resources>/app/electron e PROJECT_DIR aponta pra <resources>/app — onde
// vivem api/, dist/, etc. No dev, é o repo direto.
const PROJECT_DIR = path.join(__dirname, "..");
const PORT_FILE   = path.join(PROJECT_DIR, "api_port.tmp");

// Log do backend Python vai pra um arquivo dentro do userData pra debug em
// produção (em dev a console aceita stdout). Sem isso, quando o spawn falha
// no .exe empacotado, a janela fica vazia sem nenhuma pista pro usuário.
const PYTHON_STARTUP_TIMEOUT_MS = 30_000;
let pythonLogPath = null;
let pythonLogStream = null;

function setupPythonLog() {
  try {
    const dir = app.getPath("userData");
    fs.mkdirSync(dir, { recursive: true });
    pythonLogPath = path.join(dir, "python.log");
    pythonLogStream = fs.createWriteStream(pythonLogPath, { flags: "w" });
    pythonLogStream.write(`=== ${new Date().toISOString()} starting ===\n`);
  } catch (err) {
    console.error("Could not open python log:", err);
  }
}

function logPython(prefix, text) {
  if (pythonLogStream) {
    try { pythonLogStream.write(`[${prefix}] ${text}`); } catch { /* ignore */ }
  }
  process.stdout.write(`[${prefix}] ${text}`);
}

let apiProcess = null;
let apiPort    = null;
let pythonStartupError = null;
const portWaiters = [];

function notifyPort(port) {
  apiPort = port;
  while (portWaiters.length) {
    const { resolve } = portWaiters.shift();
    resolve(port);
  }
}

function failPort(reason) {
  pythonStartupError = reason;
  while (portWaiters.length) {
    const { reject } = portWaiters.shift();
    reject(new Error(reason));
  }
}

function waitForPort() {
  return new Promise((resolve, reject) => {
    if (apiPort) return resolve(apiPort);
    if (pythonStartupError) return reject(new Error(pythonStartupError));
    portWaiters.push({ resolve, reject });
  });
}

function startApiServer() {
  setupPythonLog();
  const apiPath = path.join(PROJECT_DIR, "api", "server.py");
  const pythonCmd = process.env.PYTHON || "python";

  if (!fs.existsSync(apiPath)) {
    failPort(`api/server.py não encontrado em ${apiPath}`);
    return;
  }

  try { fs.unlinkSync(PORT_FILE); } catch { /* ignore */ }

  try {
    apiProcess = spawn(pythonCmd, [apiPath], {
      cwd: PROJECT_DIR,
      env: { ...process.env, PYTHONIOENCODING: "utf-8" },
      stdio: ["pipe", "pipe", "pipe"],
    });
  } catch (err) {
    failPort(`Falha ao iniciar Python: ${err.message}`);
    return;
  }

  let stdoutBuf = "";
  apiProcess.stdout.on("data", (data) => {
    const text = data.toString("utf-8");
    stdoutBuf += text;
    logPython("api", text);
    if (apiPort == null) {
      const m = stdoutBuf.match(/PORT=(\d+)/);
      if (m) notifyPort(parseInt(m[1], 10));
    }
  });
  apiProcess.stderr.on("data", (data) => {
    logPython("api:err", data.toString("utf-8"));
  });
  apiProcess.on("error", (err) => {
    logPython("spawn-error", `${err.message}\n`);
    if (apiPort == null) failPort(`Python não encontrado no PATH: ${err.message}`);
  });
  apiProcess.on("exit", (code, signal) => {
    logPython("api", `exited code=${code} signal=${signal}\n`);
    if (apiPort == null) {
      failPort(`Python encerrou cedo (code=${code}). Veja ${pythonLogPath} pra detalhes.`);
    }
    apiProcess = null;
  });

  // Fallback: lê api_port.tmp se o stdout não entregar a linha PORT=N.
  const startedAt = Date.now();
  const tryReadFile = () => {
    if (apiPort) return;
    if (Date.now() - startedAt > PYTHON_STARTUP_TIMEOUT_MS) {
      failPort(
        `Backend Python não respondeu em ${PYTHON_STARTUP_TIMEOUT_MS / 1000}s. ` +
        `Log em ${pythonLogPath}. Confira se Python e FastAPI/uvicorn estão instalados.`,
      );
      return;
    }
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
    if (process.env.DEVTOOLS === "1") {
      win.webContents.openDevTools({ mode: "detach" });
    }
  } else {
    win.loadFile(path.join(PROJECT_DIR, "dist", "index.html"));
  }
  return win;
}

ipcMain.handle("api:base-url", async () => {
  const port = await waitForPort();
  return `http://127.0.0.1:${port}`;
});

// Handler usado pelo renderer pra mostrar a localização do log do Python
// quando o backend não sobe. Visível só via UI de erro.
ipcMain.handle("app:python-log-path", () => pythonLogPath);

app.whenReady().then(() => {
  // Sem barra de menu nativa (File/Edit/View/Window/Help) — a app não usa
  // nenhuma das ações default e elas só poluem a UI.
  Menu.setApplicationMenu(null);
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
  if (pythonLogStream) {
    try { pythonLogStream.end(); } catch { /* ignore */ }
  }
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  if (apiProcess) {
    try { apiProcess.kill(); } catch { /* ignore */ }
  }
});
