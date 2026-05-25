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
};

export type BackendStatus = {
  name: "turbo" | "vanilla" | "mtp";
  label: string;
  description: string;
  server_path: string;
  cli_path: string;
  server_available: boolean;
  cli_available: boolean;
  supports_spec_mtp: boolean;
  kv_types_server: string[];
  kv_types_cli: string[];
};

export type SystemInfo = {
  vram_total_mib: number | null;
  vram_free_mib: number | null;
  ram_total_mib: number | null;
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
};

export type LaunchConfig = {
  model: string;
  backend: "turbo" | "vanilla" | "mtp";
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
  mode?: "server" | "cli";
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

export type LaunchEvent =
  | { type: "start"; attempt: number; cmd: string; config: LaunchConfig }
  | { type: "stdout"; line: string }
  | { type: "load_ok"; attempt: number }
  | { type: "exit"; attempt: number; rc: number; load_ts: number | null }
  | { type: "failure"; category: string; excerpt: string; attempt: number }
  | { type: "degrade"; description: string; config: LaunchConfig }
  | { type: "restart"; attempt: number; backoff: number }
  | { type: "giveup"; reason: string; failure?: string; excerpt?: string; hint?: string }
  | { type: "done"; attempt: number };
