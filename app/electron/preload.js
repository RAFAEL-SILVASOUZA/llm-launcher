/**
 * Preload — única superfície exposta ao renderer (contextIsolation: true).
 *
 * O renderer descobre a base URL do FastAPI via ipcRenderer; daí em diante
 * é fetch puro. Não exponha file-system, spawn, ou qualquer API node aqui —
 * mantém a renderer process isolada.
 */
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  getApiBaseUrl: () => ipcRenderer.invoke("api:base-url"),
});
