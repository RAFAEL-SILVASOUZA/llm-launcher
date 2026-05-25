// Cliente HTTP do backend Python — wrappers tipados sobre fetch.
//
// O Electron preload injeta `window.electronAPI.getApiBaseUrl()`. Em dev fora
// do Electron (visualizar com `vite` no browser, p.ex.), cai pra
// `localhost:5174` como heurística — útil pra testes manuais quando o
// FastAPI é rodado standalone.

import type {
  ActiveLaunch,
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
  ModelInfo,
  NvidiaStatus,
  Options,
  SystemInfo,
} from "./types";

declare global {
  interface Window {
    electronAPI?: {
      getApiBaseUrl: () => Promise<string>;
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

  listConfigs: () => jsonFetch<LaunchConfig[]>("/api/configs"),
  saveConfig:  (cfg: LaunchConfig) =>
    jsonFetch<{ ok: boolean; config: LaunchConfig }>("/api/configs", {
      method: "POST",
      body: JSON.stringify(cfg),
    }),
  deleteConfig: (model: string, backend?: string) =>
    jsonFetch<{ ok: boolean; removed: number }>("/api/configs", {
      method: "DELETE",
      body: JSON.stringify({ model, backend: backend ?? null }),
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
  cancelLaunch: (id: string) =>
    jsonFetch<{ ok: boolean }>(`/api/launch/${id}/cancel`, { method: "POST" }),
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
};

export async function eventsStream(launchId: string): Promise<EventSource> {
  const base = await apiBase();
  return new EventSource(`${base}/api/launch/${launchId}/events`);
}

export async function downloadEventsStream(downloadId: string): Promise<EventSource> {
  const base = await apiBase();
  return new EventSource(`${base}/api/hf/download/${downloadId}/events`);
}
