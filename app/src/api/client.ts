// Cliente HTTP do backend Python — wrappers tipados sobre fetch.
//
// O Electron preload injeta `window.electronAPI.getApiBaseUrl()`. Em dev fora
// do Electron (visualizar com `vite` no browser, p.ex.), cai pra
// `localhost:5174` como heurística — útil pra testes manuais quando o
// FastAPI é rodado standalone.

import type {
  ActiveLaunch,
  AppSettings,
  BackendStatus,
  DeletePlan,
  DeleteResult,
  Estimate,
  GgufMeta,
  HfDownloadPlan,
  HfRepoListing,
  HfSearchResult,
  LaunchConfig,
  LmsLoadResult,
  LmsStatus,
  McpServer,
  McpStatus,
  ModelDefaults,
  ModelInfo,
  ModelSampling,
  ModelUpdate,
  NvidiaStatus,
  Options,
  RouterLaunchConfig,
  SystemInfo,
} from "./types";

declare global {
  interface Window {
    electronAPI?: {
      getApiBaseUrl: () => Promise<string>;
      getPythonLogPath?: () => Promise<string | null>;
    };
  }
}

let _baseUrl: string | null = null;

export async function apiBase(): Promise<string> {
  if (_baseUrl) return _baseUrl;
  if (window.electronAPI?.getApiBaseUrl) {
    _baseUrl = await window.electronAPI.getApiBaseUrl();
    return _baseUrl;
  }
  // Fallback dev — o usuário precisa rodar `python api/server.py` manualmente
  // e setar VITE_API_BASE no .env.local, ou usar o servidor em :5174.
  const env = (import.meta as unknown as { env?: Record<string, string> }).env;
  _baseUrl = env?.VITE_API_BASE ?? "http://127.0.0.1:5174";
  return _baseUrl;
}

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const base = await apiBase();
  const r = await fetch(`${base}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      const body = await r.json();
      detail = body?.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(`API ${path}: ${detail}`);
  }
  return r.json() as Promise<T>;
}

export const api = {
  options:    () => jsonFetch<Options>("/api/options"),
  backends:   () => jsonFetch<BackendStatus[]>("/api/backends"),
  system:     () => jsonFetch<SystemInfo>("/api/system"),
  models:     () => jsonFetch<ModelInfo[]>("/api/models"),
  modelMeta:  (model: string) =>
    jsonFetch<GgufMeta>("/api/models/meta", {
      method: "POST",
      body: JSON.stringify({ model }),
    }),
  modelContextOptions: (model: string) =>
    jsonFetch<{ options: { value: number; label: string }[] }>("/api/models/context-options", {
      method: "POST",
      body: JSON.stringify({ model }),
    }),
  // Samplers que o modelo receberia agora + a procedência deles.
  modelSampling: (model: string, config?: LaunchConfig) =>
    jsonFetch<ModelSampling>("/api/models/sampling", {
      method: "POST",
      body: JSON.stringify({ model, config: config ?? null }),
    }),
  // Config inteira derivada do modelo + VRAM da máquina.
  modelDefaults: (model: string, backend: string, mode = "server") =>
    jsonFetch<ModelDefaults>("/api/models/defaults", {
      method: "POST",
      body: JSON.stringify({ model, backend, mode }),
    }),

  // Checagem de atualização de GGUFs. Sem deep: barato (tamanho + oid
  // registrado/cacheado), pra rodar na inicialização. Deep: hasheia o arquivo
  // pra confirmar — botão manual por modelo.
  modelsUpdates: (deep = false) =>
    jsonFetch<{ results: ModelUpdate[] }>(`/api/models/updates?deep=${deep ? "true" : "false"}`),
  checkModelUpdate: (model: string, deep = true) =>
    jsonFetch<ModelUpdate>("/api/models/check-update", {
      method: "POST",
      body: JSON.stringify({ model, deep }),
    }),

  listConfigs: () => jsonFetch<LaunchConfig[]>("/api/configs"),
  saveConfig:  (cfg: LaunchConfig) =>
    jsonFetch<{ ok: boolean; config: LaunchConfig }>("/api/configs", {
      method: "POST",
      body: JSON.stringify(cfg),
    }),
  deleteConfig: (id: string) =>
    jsonFetch<{ ok: boolean; removed: number }>("/api/configs", {
      method: "DELETE",
      body: JSON.stringify({ id }),
    }),

  estimateMany: (items: {
    model: string;
    backend: string;
    context_window: number;
    kv_cache: string;
    parallel_slots: number;
    gpu_layers: number;
    n_cpu_moe: number;
    mmproj: string | null;
    cache_ram: number;
    mode: "server" | "cli";
  }[]) =>
    jsonFetch<{ ok: boolean; estimate?: Estimate; error?: string; model: string; backend: string }[]>(
      "/api/estimate-many",
      { method: "POST", body: JSON.stringify({ items }) },
    ),
  estimate: (req: {
    model: string;
    backend: string;
    context_window: number;
    kv_cache: string;
    parallel_slots: number;
    gpu_layers: number;
    n_cpu_moe: number;
    mmproj: string | null;
    cache_ram: number;
    mode: "server" | "cli";
  }) =>
    jsonFetch<Estimate>("/api/estimate", {
      method: "POST",
      body: JSON.stringify(req),
    }),
  buildCommand: (cfg: LaunchConfig) =>
    jsonFetch<{ command: string; mode: string }>("/api/build-command", {
      method: "POST",
      body: JSON.stringify(cfg),
    }),
  suggestNCpuMoe: (req: {
    model: string;
    backend: string;
    context_window: number;
    kv_cache: string;
    parallel_slots: number;
    gpu_layers: number;
    mmproj: string | null;
    cache_ram: number;
    mode: "server" | "cli";
  }) =>
    jsonFetch<{ n_cpu_moe: number }>("/api/suggest/n-cpu-moe", {
      method: "POST",
      body: JSON.stringify(req),
    }),
  suggestNGpuLayers: (req: {
    model: string;
    backend: string;
    context_window: number;
    kv_cache: string;
    parallel_slots: number;
    gpu_layers: number;
    mmproj: string | null;
    cache_ram: number;
    mode: "server" | "cli";
  }) =>
    jsonFetch<{ n_gpu_layers: number }>("/api/suggest/n-gpu-layers", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  launch: (cfg: LaunchConfig) =>
    jsonFetch<{ launch_id: string }>("/api/launch", {
      method: "POST",
      body: JSON.stringify(cfg),
    }),
  // Modo router: sobe N configs de uma vez (llama-server --models-preset).
  launchRouter: (ids: string[]) =>
    jsonFetch<{ launch_id: string; config: RouterLaunchConfig }>("/api/launch-router", {
      method: "POST",
      body: JSON.stringify({ ids }),
    }),
  cancelLaunch: (id: string) =>
    jsonFetch<{ ok: boolean }>(`/api/launch/${id}/cancel`, { method: "POST" }),
  restartLaunch: (id: string) =>
    jsonFetch<{ ok: boolean }>(`/api/launch/${id}/restart`, { method: "POST" }),
  listLaunches: () =>
    jsonFetch<ActiveLaunch[]>("/api/launches"),

  // ─── HuggingFace ─────────────────────────────────────────────────────
  hfResolve: (url: string) =>
    jsonFetch<{ repo_id: string; filename: string | null; branch: string }>(
      "/api/hf/resolve",
      { method: "POST", body: JSON.stringify({ url }) },
    ),
  hfList: (repo_id: string) =>
    jsonFetch<HfRepoListing>("/api/hf/list", {
      method: "POST",
      body: JSON.stringify({ repo_id }),
    }),
  hfSearch: (query: string, limit = 25) =>
    jsonFetch<{ results: HfSearchResult[] }>("/api/hf/search", {
      method: "POST",
      body: JSON.stringify({ query, limit }),
    }),
  hfDownload: (req: {
    repo_id: string;
    branch?: string;
    rel_paths: string[];
    subdir?: string | null;
    base_dir: string;
  }) =>
    jsonFetch<{ download_id: string; plan: HfDownloadPlan }>("/api/hf/download", {
      method: "POST",
      body: JSON.stringify({ branch: "main", subdir: null, ...req }),
    }),
  cancelDownload: (id: string) =>
    jsonFetch<{ ok: boolean }>(`/api/hf/download/${id}/cancel`, { method: "POST" }),

  // ─── delete model ────────────────────────────────────────────────────
  planDeleteModel: (model: string) =>
    jsonFetch<DeletePlan>("/api/models/plan-delete", {
      method: "POST",
      body: JSON.stringify({ model, confirm: false }),
    }),
  deleteModel: (model: string) =>
    jsonFetch<DeleteResult>("/api/models", {
      method: "DELETE",
      body: JSON.stringify({ model, confirm: true }),
    }),

  // ─── LM Studio ───────────────────────────────────────────────────────
  lmsStatus: () => jsonFetch<LmsStatus>("/api/lms/status"),
  lmsLoad: (req: { model: string; context_window: number; parallel_slots: number }) =>
    jsonFetch<LmsLoadResult>("/api/lms/load", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  // ─── nvidia ──────────────────────────────────────────────────────────
  nvidia: () => jsonFetch<NvidiaStatus>("/api/nvidia"),

  // ─── MCP servers ─────────────────────────────────────────────────────
  mcpList: () => jsonFetch<McpServer[]>("/api/mcp"),
  mcpSave: (s: { id?: string; name: string; cwd: string; command: string; enabled?: boolean }) =>
    jsonFetch<{ ok: boolean; server: McpServer; status: McpStatus }>("/api/mcp", {
      method: "POST",
      body: JSON.stringify(s),
    }),
  mcpDelete: (id: string) =>
    jsonFetch<{ ok: boolean }>(`/api/mcp/${id}`, { method: "DELETE" }),
  mcpStart: (id: string) =>
    jsonFetch<McpStatus>(`/api/mcp/${id}/start`, { method: "POST" }),
  mcpStop: (id: string) =>
    jsonFetch<McpStatus>(`/api/mcp/${id}/stop`, { method: "POST" }),
  mcpLogs: (id: string, limit = 500) =>
    jsonFetch<{ logs: { ts: number; kind: string; text: string }[] }>(
      `/api/mcp/${id}/logs?limit=${limit}`,
    ),

  // ─── app settings ────────────────────────────────────────────────────
  getSettings: () => jsonFetch<AppSettings>("/api/settings"),
  // Só envia os campos editáveis — backend_paths_defaults é read-only.
  saveSettings: (settings: {
    model_paths: string[];
    backend_paths: Partial<Record<"turbo" | "vanilla", string>>;
  }) =>
    jsonFetch<AppSettings>("/api/settings", {
      method: "POST",
      body: JSON.stringify(settings),
    }),
};

export async function eventsStream(launchId: string): Promise<EventSource> {
  const base = await apiBase();
  return new EventSource(`${base}/api/launch/${launchId}/events`);
}

export async function downloadEventsStream(downloadId: string): Promise<EventSource> {
  const base = await apiBase();
  return new EventSource(`${base}/api/hf/download/${downloadId}/events`);
}

export async function mcpEventsStream(serverId: string): Promise<EventSource> {
  const base = await apiBase();
  return new EventSource(`${base}/api/mcp/${serverId}/events`);
}
