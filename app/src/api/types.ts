// Espelho dos tipos retornados pelo FastAPI. Mantém em sync com api/server.py.

export type Option<T = number | string> = {
  value: T;
  label: string;
  description?: string;
};

export type Options = {
  kv_cache: Option<string>[];
  context_window: Option<number>[];
  reasoning_budget: Option<number>[];
  parallel_slots: Option<number>[];
  max_tokens: Option<number>[];
  batch_size: Option<number>[];
  ubatch_size: Option<number>[];
  cache_ram: Option<number>[];
  ctx_checkpoints: Option<number>[];
  spec_draft_n_max: Option<number>[];
  cpu_threads_gen: number;
  cpu_threads_batch: number;
  sampler_presets: Record<"code" | "reasoning", Sampler>;
};

// ─── sampling ──────────────────────────────────────────────────────────────
// De onde vieram os samplers, em ordem de confiança. A UI mostra isto porque
// "default" e "template" são derivação nossa, não recomendação do autor.
export type SamplerSource = "generation_config" | "template" | "default" | "manual";

export type Sampler = {
  temp: number;
  top_p: number;
  top_k: number;
  min_p: number;
  repeat_penalty: number;
};

export type ModelSampling = Sampler & {
  source: SamplerSource;
  thinking: boolean;
  from_repo?: string | null;
};

// Backends de llama.cpp. `mtp` não é build próprio (é o vanilla + --spec-type
// draft-mtp); `custom` é o build/fork que um modelo específico exige — sem
// identidade fixa, o caminho vem do Settings.
export type BackendName = "turbo" | "vanilla" | "mtp" | "custom";
// Os que têm diretório próprio configurável (mtp segue o vanilla).
export type ConfigurableBackend = "turbo" | "vanilla" | "custom";

export type BackendStatus = {
  name: BackendName;
  label: string;
  description: string;
  server_path: string;
  cli_path: string;
  server_available: boolean;
  cli_available: boolean;
  supports_spec_mtp: boolean;
  kv_types_server: string[];
  kv_types_cli: string[];
  default_dir: string;
  configured_dir: string;
};

export type SystemInfo = {
  vram_total_mib: number | null;
  vram_free_mib: number | null;
  ram_total_mib: number | null;
  ram_avail_mib: number | null;
  gpu_count?: number;
};

export type ModelInfo = {
  path: string;
  alias: string;
  relative_path: string;
  filename: string;
  mmproj: string | null;
  size_mib: number;
  is_thinking: boolean;
  is_mtp: boolean;
  /** gguf da head MTP quando vem em pasta separada (`mtp-<quant>/`) — vira -md. */
  mtp_draft: string | null;
  sampling: ModelSampling;
};

export type GgufMeta = {
  arch: string;
  n_layer: number;
  n_embd: number;
  n_head: number;
  n_head_kv: number | number[];
  head_dim_k: number;
  head_dim_v: number;
  swa_window: number | null;
  swa_pattern: boolean[] | null;
  head_dim_k_swa: number | null;
  head_dim_v_swa: number | null;
  is_moe: boolean;
  expert_count: number;
  expert_used_count: number;
  moe_layers_count: number;
  moe_per_layer_mib: number;
  token_embd_mib: number;
  output_mib: number;
  blocks_mib: number;
  tensor_total_mib: number;
  attn_layers: number[];
  ssm_state_per_layer_mib: number;
  supports_preserve_thinking: boolean;
  // Vem do chat template, não do nome do arquivo — é a detecção confiável.
  has_thinking_template: boolean;
};

export type LaunchConfig = {
  id?: string;   // identidade estável; o backend gera no 1º save se ausente
  model: string;
  backend: BackendName;
  context_window: number;
  kv_cache: string;
  flash_attn: boolean;
  gpu_layers: number;
  n_cpu_moe: number;
  parallel_slots: number;
  reasoning_budget: number | null;
  preserve_thinking: boolean;
  mlock: boolean;
  max_tokens: number;
  batch_size: number;
  ubatch_size: number;
  threads_gen: number;
  threads_batch: number;
  cache_ram: number;
  ctx_checkpoints: number;
  spec_draft_n_max: number;
  mmproj: string | null;
  verbose: boolean;
  // "llama.cpp decide": comando mínimo (-m/--mmproj/--alias + --jinja/--metrics/
  // --host/--port). O resto da config continua salvo, só não vira flag.
  llama_auto?: boolean;
  // Samplers. null = resolvido por modelo no backend. sampler_source "manual"
  // trava os valores: o usuário editou e nada os sobrescreve.
  temp?: number | null;
  top_p?: number | null;
  top_k?: number | null;
  min_p?: number | null;
  repeat_penalty?: number | null;
  sampler_source?: SamplerSource | null;
  // Sem controle na UI, só via JSON da config:
  //   cors_origins     — undefined = "localhost" (o server reflete o Origin só
  //                      quando é localhost); "*" volta ao default do llama.cpp,
  //                      que é o que dispara o aviso de CORS aberto no boot.
  //   image_min_tokens — undefined = decide pelo projector (1024 em Qwen-VL,
  //                      nada no resto); 0 desliga a flag.
  cors_origins?: string | null;
  image_min_tokens?: number | null;
  // Multi-GPU (opcionais; null/undefined = default do llama.cpp = layer split)
  split_mode?: "none" | "layer" | "row" | null;
  tensor_split?: string | null;
  main_gpu?: number | null;
  mode?: "server" | "cli";
};

// Config recomendada pro modelo (botão "defaults do modelo"). `notes` explica cada
// escolha não-óbvia — o usuário precisa poder discordar do palpite com base.
export type ModelDefaults = {
  config: LaunchConfig;
  notes: string[];
  sampling: ModelSampling;
};

export type Estimate = {
  meta_ok: boolean;
  weights_mib: number;
  kv_total_mib: number;
  mmproj_mib: number;
  vram_weights: number;
  vram_kv: number;
  vram_compute: number;
  vram_ssm: number;
  vram_mmproj: number;
  vram_mmproj_weights: number;
  vram_mmproj_compute: number;
  vram_mtp_kv: number;
  vram_mtp_compute: number;
  vram_mtp_weights: number;
  ram_weights: number;
  ram_kv: number;
  ram_ssm: number;
  cache_ram: number;
  moe_offload_mib: number;
  vram_total: number;
  ram_total: number;
  vram_avail: number | null;
  vram_total_phys: number | null;
  ram_avail: number | null;
};

// Launch múltiplo (modo router do llama-server): config sintética que o
// backend devolve no lugar de uma LaunchConfig quando N modelos sobem juntos.
export type RouterLaunchConfig = {
  router: true;
  backend: string;
  config_ids: string[];   // ids das configs selecionadas na grid
  model_ids: string[];    // nomes das seções do preset = valor do campo "model" nos requests
  models: string[];       // paths dos .gguf
};

export type ActiveLaunchConfig = LaunchConfig | RouterLaunchConfig;

export const isRouterConfig = (c: ActiveLaunchConfig): c is RouterLaunchConfig =>
  "router" in c && c.router === true;

export type ActiveLaunch = {
  launch_id: string;
  attached: boolean;
  pid: number | null;
  config: ActiveLaunchConfig;
};

export type LaunchEvent =
  | { type: "start"; attempt: number; cmd: string; config: ActiveLaunchConfig }
  | { type: "stdout"; line: string }
  | { type: "load_ok"; attempt: number }
  | { type: "exit"; attempt: number; rc: number; load_ts: number | null }
  | { type: "failure"; category: string; excerpt: string; attempt: number }
  | { type: "degrade"; description: string; config: LaunchConfig }
  | { type: "restart"; attempt: number; backoff: number }
  | { type: "manual_restart"; attempt: number }
  | { type: "giveup"; reason: string; failure?: string; excerpt?: string; hint?: string }
  | { type: "done"; attempt: number };

// ─── HuggingFace download ──────────────────────────────────────────────────

export type HfFile = {
  path: string;
  size: number;
};

export type HfRepoListing = {
  repo_id: string;
  files: HfFile[];
  mmprojs: HfFile[];
  quants: Record<string, HfFile[]>;
};

export type HfSearchResult = {
  repo_id: string;
  downloads: number;
  likes: number;
};

export type HfDownloadPlan = {
  repo_id: string;
  branch: string;
  base_dir: string;
  items: {
    rel: string;
    dest: string;
    exists: boolean;
    size_disk: number;
    url: string;
  }[];
};

export type HfDownloadEvent =
  | { type: "file_start"; rel: string; dest: string; index: number; total: number }
  | { type: "progress"; rel: string; downloaded: number; total: number; speed: number }
  | { type: "file_done"; rel: string; dest: string }
  | { type: "file_skip"; rel: string; dest: string; reason: string; size: number }
  | {
      type: "sampling";
      found: boolean;
      source?: SamplerSource;
      from_repo?: string | null;
      values?: Partial<Sampler>;
      error?: string;
    }
  | { type: "error"; rel: string; message: string }
  | { type: "cancelled" }
  | { type: "done" };

// ─── atualização de GGUF ───────────────────────────────────────────────────
// Espelha api/core/updates.py. status: 'update_available' = o arquivo no HF
// difere do local; 'up_to_date' = igual (verified=false quando só o tamanho
// bateu e o sha256 não foi conferido); 'unknown' = sem repo de origem ou rede
// indisponível.
export type UpdateStatus = "up_to_date" | "update_available" | "unknown";

export type ModelUpdateFile = {
  name: string;
  rel: string | null;
  status: UpdateStatus;
  verified: boolean;
  reason: string;
};

export type ModelUpdate = {
  path: string;
  repo_id: string | null;
  branch: string;
  status: UpdateStatus;
  verified: boolean;
  has_origin: boolean;
  // Alvo do re-download: raiz cadastrada + subdir dentro de owner/repo. É o que
  // o endpoint de download espera; base_dir é só a pasta final, pra exibição.
  root: string | null;
  subdir: string | null;
  base_dir: string;
  rel_paths: string[];
  files: ModelUpdateFile[];
  error: string | null;
};

// ─── NVIDIA detalhado ──────────────────────────────────────────────────────

export type NvidiaStatus =
  | { available: false;
  error: string }
  | { available: true;
  gpus: Record<string, string>[];
  host_temp_c?: number | null };

// ─── LM Studio ──────────────────────────────────────────────────────────────

export type LmsStatus = {
  available: boolean;
  running: boolean;
  path: string;
  raw?: string;
  error?: string;
};

export type AppSettings = {
  model_paths: string[];
  // Override do diretório dos binários por backend. Chave ausente / valor
  // vazio = usa o default em backend_paths_defaults. mtp não entra: reusa o
  // build do vanilla (MTP nativo no upstream), então segue o path do vanilla.
  backend_paths: Partial<Record<ConfigurableBackend, string>>;
  backend_paths_defaults: Record<ConfigurableBackend, string>;
};

export type LmsLoadResult = {
  ok: boolean;
  key?: string;
  alias?: string;
  start_output?: string;
  load_output?: string;
  rc?: number;
  error?: string;
};

// ─── MCP servers ───────────────────────────────────────────────────────────

export type McpStatus = {
  id: string;
  running: boolean;
  pid: number | null;
  started_at: number | null;
  stopped_at: number | null;
  last_exit_code: number | null;
  last_error: string | null;
  auto_stopped: boolean;
};

export type McpServer = {
  id: string;
  name: string;
  cwd: string;
  command: string;
  enabled: boolean;
  created_at: number;
  status: McpStatus;
};

export type McpLogEvent = {
  ts: number;
  kind: "info" | "stdout" | "error" | "_eof";
  text: string;
};

// ─── delete model ──────────────────────────────────────────────────────────

export type DeletePlan = {
  files: string[];
  total_bytes: number;
  count: number;
};

export type DeleteResult = {
  removed: string[];
  errors: { path: string; error: string }[];
  total_bytes: number;
};
