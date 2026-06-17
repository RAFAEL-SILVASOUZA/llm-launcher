import { Rocket } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type {
  BackendStatus, Estimate, GgufMeta, LaunchConfig, ModelInfo, Options,
} from "../api/types";
import { modelAliasFromPath } from "../util/format";
import { EstimatePanel } from "./EstimatePanel";
import { Modal } from "./Modal";

type Props = {
  open: boolean;
  initial: Partial<LaunchConfig> | null;
  models: ModelInfo[];
  backends: BackendStatus[];
  options: Options | null;
  onClose: () => void;
  onSave: (cfg: LaunchConfig) => Promise<void>;
  onLaunch: (cfg: LaunchConfig) => Promise<void>;
};

const DEFAULT_CFG: LaunchConfig = {
  model: "",
  backend: "turbo",
  context_window: 65_536,
  kv_cache: "turbo4",
  flash_attn: true,
  gpu_layers: 99,
  n_cpu_moe: 0,
  parallel_slots: 1,
  reasoning_budget: null,
  preserve_thinking: false,
  mlock: false,
  max_tokens: -1,
  batch_size: 2_048,
  ubatch_size: 512,
  threads_gen: 14,
  threads_batch: 20,
  cache_ram: 24_576,
  ctx_checkpoints: 8,
  spec_draft_n_max: 2,
  mmproj: null,
  verbose: false,
  split_mode: null,
  tensor_split: null,
  main_gpu: null,
  mode: "server",
};

export function ConfigEditor({
  open, initial, models, backends, options, onClose, onSave, onLaunch,
}: Props) {
  const [cfg, setCfg] = useState<LaunchConfig>(DEFAULT_CFG);
  const [meta, setMeta] = useState<GgufMeta | null>(null);
  const [estimate, setEstimate] = useState<Estimate | null>(null);
  const [estimating, setEstimating] = useState(false);
  const [command, setCommand] = useState<string>("");
  const [saving, setSaving] = useState(false);
  const [gpuCount, setGpuCount] = useState(0);

  // Quantas GPUs há — decide se mostramos os controles de split multi-GPU.
  useEffect(() => {
    if (!open) return;
    api.system()
      .then((s) => setGpuCount(s.gpu_count ?? 0))
      .catch(() => setGpuCount(0));
  }, [open]);

  // (Re)inicializa quando abrir com um initial diferente
  useEffect(() => {
    if (!open) return;
    setCfg({
      ...DEFAULT_CFG,
      threads_gen: options?.cpu_threads_gen ?? DEFAULT_CFG.threads_gen,
      threads_batch: options?.cpu_threads_batch ?? DEFAULT_CFG.threads_batch,
      ...(initial ?? {}),
    });
  }, [open, initial, options]);

  // Carrega meta do modelo selecionado
  useEffect(() => {
    if (!cfg.model) {
      setMeta(null);
      return;
    }
    let cancelled = false;
    api.modelMeta(cfg.model)
      .then((m) => !cancelled && setMeta(m))
      .catch(() => !cancelled && setMeta(null));
    return () => { cancelled = true; };
  }, [cfg.model]);

  // Estimativa debounced — recalcula em qualquer mudança relevante
  const estimateRef = useRef<number | null>(null);
  useEffect(() => {
    if (!cfg.model) return;
    setEstimating(true);
    if (estimateRef.current) window.clearTimeout(estimateRef.current);
    estimateRef.current = window.setTimeout(async () => {
      try {
        const e = await api.estimate({
          model: cfg.model,
          backend: cfg.backend,
          context_window: cfg.context_window,
          kv_cache: cfg.kv_cache,
          parallel_slots: cfg.parallel_slots,
          gpu_layers: cfg.gpu_layers,
          n_cpu_moe: cfg.n_cpu_moe,
          mmproj: cfg.mmproj,
          cache_ram: cfg.cache_ram,
          mode: cfg.mode ?? "server",
        });
        setEstimate(e);
      } catch (err) {
        console.error(err);
      } finally {
        setEstimating(false);
      }
    }, 200);
    return () => {
      if (estimateRef.current) window.clearTimeout(estimateRef.current);
    };
  }, [
    cfg.model, cfg.backend, cfg.context_window, cfg.kv_cache,
    cfg.parallel_slots, cfg.gpu_layers, cfg.n_cpu_moe, cfg.mmproj,
    cfg.cache_ram, cfg.mode,
  ]);

  // Preview do comando
  const cmdRef = useRef<number | null>(null);
  useEffect(() => {
    if (!cfg.model) { setCommand(""); return; }
    if (cmdRef.current) window.clearTimeout(cmdRef.current);
    cmdRef.current = window.setTimeout(async () => {
      try {
        const r = await api.buildCommand(cfg);
        setCommand(r.command);
      } catch (err) {
        setCommand(`# erro ao construir comando: ${(err as Error).message}`);
      }
    }, 250);
    return () => {
      if (cmdRef.current) window.clearTimeout(cmdRef.current);
    };
  }, [cfg]);

  const isMoE      = meta?.is_moe ?? false;
  const isHybrid   = !!(meta && meta.ssm_state_per_layer_mib);
  const moeLayers  = meta?.moe_layers_count ?? 0;
  const isMtpModel = useMemo(
    () => !!cfg.model && /(?:^|[-_.])mtp(?:[-_.]|$)/i.test(cfg.model.split(/[\\/]/).join("/")),
    [cfg.model],
  );
  const backend = backends.find((b) => b.name === cfg.backend);
  const isThinkingModel = useMemo(() => {
    const lower = cfg.model.toLowerCase();
    return ["qwen3", "qwq", "deepseek-r1", "thinking", "reasoning", "gemma-4", "gemma4"].some((k) => lower.includes(k));
  }, [cfg.model]);

  const modelInfo = models.find((m) => m.path === cfg.model);
  const hasMmproj = !!modelInfo?.mmproj;

  const handleSave = async (then?: "launch" | "close") => {
    if (!cfg.model) return;
    setSaving(true);
    try {
      if (then === "launch") {
        await onLaunch(cfg);
      } else {
        await onSave(cfg);
        onClose();
      }
    } finally {
      setSaving(false);
    }
  };

  if (!options) return null;

  return (
    <Modal
      open={open}
      onClose={onClose}
      width="max-w-6xl"
      // Esta modal carrega bastante input — não deve fechar com Esc nem clique
      // no backdrop pra evitar perder edição por engano. Só fecha via X ou
      // botão Cancelar.
      closeOnEsc={false}
      closeOnBackdrop={false}
      title={initial?.model ? "Editar configuração" : "Nova configuração"}
      footer={
        <>
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-sm rounded bg-ink-800 hover:bg-ink-700"
          >
            Cancelar
          </button>
          <button
            onClick={() => handleSave("close")}
            disabled={!cfg.model || saving}
            className="px-3 py-1.5 text-sm rounded bg-ink-700 hover:bg-ink-600 disabled:opacity-50"
          >
            Salvar
          </button>
          <button
            onClick={() => handleSave("launch")}
            disabled={!cfg.model || saving || !backend?.server_available}
            className="px-3 py-1.5 text-sm rounded bg-accent hover:bg-accent/80 text-white disabled:opacity-50 flex items-center gap-1"
            title={!backend?.server_available ? "binário do backend não encontrado" : ""}
          >
            <Rocket className="w-3 h-3" />Salvar e launch
          </button>
        </>
      }
    >
      <div className="grid grid-cols-[1fr_320px] gap-0 min-h-[60vh]">
        {/* ── coluna esquerda: form ─────────────────────────────────────── */}
        <div className="p-5 space-y-5 overflow-y-auto">

          <Section title="Modelo & backend">
            <Field label="Modelo (.gguf)">
              <select
                value={cfg.model}
                onChange={(e) => {
                  const m = models.find((x) => x.path === e.target.value);
                  setCfg((c) => ({
                    ...c,
                    model: e.target.value,
                    mmproj: m?.mmproj ?? null,
                  }));
                }}
                className={selectCls}
              >
                <option value="">— selecionar —</option>
                {models.map((m) => (
                  <option key={m.path} value={m.path}>
                    {m.alias} {m.mmproj ? "🖼" : ""} {m.is_mtp ? "⚡MTP" : ""} ({m.relative_path})
                  </option>
                ))}
              </select>
            </Field>

            <Field label="Backend">
              <div className="flex gap-2">
                {backends.map((b) => (
                  <label
                    key={b.name}
                    className={`flex-1 border rounded p-2 cursor-pointer text-sm ${
                      cfg.backend === b.name
                        ? "border-accent bg-accent/10"
                        : "border-ink-700 hover:border-ink-500"
                    } ${!b.server_available ? "opacity-50" : ""}`}
                  >
                    <input
                      type="radio"
                      className="hidden"
                      checked={cfg.backend === b.name}
                      onChange={() => setCfg((c) => ({ ...c, backend: b.name }))}
                      disabled={!b.server_available}
                    />
                    <div className="font-medium mono">
                      {b.label}{b.supports_spec_mtp && " ⚡"}
                    </div>
                    <div className="text-xs text-ink-500 mt-0.5">{b.description}</div>
                    {!b.server_available && (
                      <div className="text-[11px] text-red-400 mt-1">binário não compilado</div>
                    )}
                  </label>
                ))}
              </div>
            </Field>

            {hasMmproj && (
              <Field label="mmproj (visão)">
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={!!cfg.mmproj}
                    onChange={(e) =>
                      setCfg((c) => ({
                        ...c,
                        mmproj: e.target.checked ? (modelInfo?.mmproj ?? null) : null,
                      }))
                    }
                  />
                  habilitar imagem ({modelInfo?.mmproj?.split(/[\\/]/).pop()})
                </label>
              </Field>
            )}
          </Section>

          <Section title="Contexto & KV">
            <div className="grid grid-cols-2 gap-3">
              <Field label="Context window (por slot)">
                <SelectFromOptions
                  options={options.context_window}
                  value={cfg.context_window}
                  onChange={(v) => setCfg((c) => ({ ...c, context_window: v as number }))}
                  allowCustom
                />
              </Field>
              <Field label="Slots paralelos (-np)">
                <SelectFromOptions
                  options={options.parallel_slots}
                  value={cfg.parallel_slots}
                  onChange={(v) => setCfg((c) => ({ ...c, parallel_slots: v as number }))}
                />
              </Field>
              <Field label="KV cache (--cache-type-k/v)">
                <select
                  value={cfg.kv_cache}
                  onChange={(e) => setCfg((c) => ({ ...c, kv_cache: e.target.value }))}
                  className={selectCls}
                >
                  {options.kv_cache.map((o) => {
                    const supported = backend?.kv_types_server.includes(o.value) ?? true;
                    return (
                      <option key={o.value} value={o.value} disabled={!supported}>
                        {o.label} — {o.description}{!supported ? "  (não suportado em " + cfg.backend + ")" : ""}
                      </option>
                    );
                  })}
                </select>
              </Field>
              <Field label="Flash Attention (-fa)">
                <Toggle
                  checked={cfg.flash_attn}
                  onChange={(v) => setCfg((c) => ({ ...c, flash_attn: v }))}
                  label={cfg.flash_attn ? "ON" : "OFF"}
                />
              </Field>
            </div>
          </Section>

          <Section title="GPU / offload">
            <div className="grid grid-cols-2 gap-3">
              <Field label={`Camadas na GPU (-ngl)${meta ? `  · ${meta.n_layer} totais` : ""}`}>
                <input
                  type="number"
                  value={cfg.gpu_layers}
                  onChange={(e) => setCfg((c) => ({ ...c, gpu_layers: Number(e.target.value) }))}
                  className={inputCls}
                  min={0}
                  max={meta ? meta.n_layer + 2 : 99}
                />
              </Field>
              {isMoE && (
                <Field label={`-ncmoe (camadas MoE na CPU, máx ${moeLayers})`}>
                  <input
                    type="number"
                    value={cfg.n_cpu_moe}
                    onChange={(e) => setCfg((c) => ({ ...c, n_cpu_moe: Number(e.target.value) }))}
                    className={inputCls}
                    min={0}
                    max={moeLayers}
                  />
                </Field>
              )}
            </div>
            <div className="flex gap-2 mt-2">
              <button
                onClick={async () => {
                  if (!cfg.model) return;
                  const r = await api.suggestNGpuLayers({
                    model: cfg.model, backend: cfg.backend,
                    context_window: cfg.context_window, kv_cache: cfg.kv_cache,
                    parallel_slots: cfg.parallel_slots, gpu_layers: cfg.gpu_layers,
                    mmproj: cfg.mmproj, cache_ram: cfg.cache_ram, mode: "server",
                  });
                  setCfg((c) => ({ ...c, gpu_layers: r.n_gpu_layers }));
                }}
                className="text-xs px-2 py-1 rounded bg-ink-800 hover:bg-ink-700"
              >
                sugerir -ngl
              </button>
              {isMoE && (
                <button
                  onClick={async () => {
                    if (!cfg.model) return;
                    const r = await api.suggestNCpuMoe({
                      model: cfg.model, backend: cfg.backend,
                      context_window: cfg.context_window, kv_cache: cfg.kv_cache,
                      parallel_slots: cfg.parallel_slots, gpu_layers: cfg.gpu_layers,
                      mmproj: cfg.mmproj, cache_ram: cfg.cache_ram, mode: "server",
                    });
                    setCfg((c) => ({ ...c, n_cpu_moe: r.n_cpu_moe }));
                  }}
                  className="text-xs px-2 py-1 rounded bg-ink-800 hover:bg-ink-700"
                >
                  sugerir -ncmoe
                </button>
              )}
            </div>
            {isHybrid && (
              <p className="text-xs text-amber-400 mt-2">
                ℹ Arch híbrida SSM detectada — reasoning agentic costuma travar
                com budget&gt;0. Considere --reasoning off.
              </p>
            )}

            {gpuCount >= 2 && (
              <div className="mt-3 border-t border-ink-800 pt-3 space-y-3">
                <p className="text-xs text-ink-400">
                  🖥️ {gpuCount} GPUs detectadas — por padrão (layer) o llama.cpp
                  reparte as camadas entre todas, proporcional à VRAM.
                </p>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Split entre GPUs (-sm)">
                    <select
                      value={cfg.split_mode ?? "layer"}
                      onChange={(e) => {
                        const v = e.target.value;
                        setCfg((c) => ({
                          ...c,
                          // "layer" é o default → guarda null pra não poluir o comando
                          split_mode: v === "layer" ? null : (v as "none" | "row"),
                          // -mg só faz sentido fora do modo "none"? Na verdade vale
                          // pra todos, mas só exibimos no none; limpa ao sair dele.
                          main_gpu: v === "none" ? (c.main_gpu ?? 0) : null,
                        }));
                      }}
                      className={selectCls}
                    >
                      <option value="layer">layer — repartir camadas (default)</option>
                      <option value="none">none — tudo numa GPU só</option>
                      <option value="row">row — split por linha (raro)</option>
                    </select>
                  </Field>
                  {cfg.split_mode === "none" && (
                    <Field label={`GPU principal (-mg, 0..${gpuCount - 1})`}>
                      <input
                        type="number"
                        value={cfg.main_gpu ?? 0}
                        onChange={(e) =>
                          setCfg((c) => ({ ...c, main_gpu: Number(e.target.value) }))
                        }
                        className={inputCls}
                        min={0}
                        max={gpuCount - 1}
                      />
                    </Field>
                  )}
                  {cfg.split_mode !== "none" && (
                    <Field label="Proporção -ts (vazio = auto por VRAM)">
                      <input
                        type="text"
                        value={cfg.tensor_split ?? ""}
                        onChange={(e) =>
                          setCfg((c) => ({
                            ...c,
                            tensor_split: e.target.value.trim() || null,
                          }))
                        }
                        placeholder={Array.from({ length: gpuCount }, () => "1").join(",")}
                        className={inputCls}
                      />
                    </Field>
                  )}
                </div>
              </div>
            )}
          </Section>

          <Section title="Server (KV total, prompt cache, checkpoints)">
            <div className="grid grid-cols-2 gap-3">
              <Field label="cache-ram (MiB)">
                <SelectFromOptions
                  options={options.cache_ram}
                  value={cfg.cache_ram}
                  onChange={(v) => setCfg((c) => ({ ...c, cache_ram: v as number }))}
                  allowCustom
                />
              </Field>
              <Field label="ctx-checkpoints">
                <SelectFromOptions
                  options={options.ctx_checkpoints}
                  value={cfg.ctx_checkpoints}
                  onChange={(v) => setCfg((c) => ({ ...c, ctx_checkpoints: v as number }))}
                />
              </Field>
            </div>
          </Section>

          {(isThinkingModel || cfg.reasoning_budget !== null) && (
            <Section title="Reasoning (modelos thinking)">
              <Field label="reasoning-budget">
                <SelectFromOptions
                  options={options.reasoning_budget}
                  value={cfg.reasoning_budget ?? 4_096}
                  onChange={(v) => setCfg((c) => ({ ...c, reasoning_budget: v as number }))}
                />
              </Field>
              {meta?.supports_preserve_thinking && cfg.reasoning_budget && cfg.reasoning_budget !== 0 && (
                <label className="flex items-center gap-2 text-sm mt-2">
                  <input
                    type="checkbox"
                    checked={cfg.preserve_thinking}
                    onChange={(e) => setCfg((c) => ({ ...c, preserve_thinking: e.target.checked }))}
                  />
                  preserve_thinking (manter &lt;think&gt; de turnos anteriores)
                </label>
              )}
            </Section>
          )}

          {backend?.supports_spec_mtp && isMtpModel && (
            <Section title="MTP — speculative decoding">
              <Field label="spec-draft-n-max">
                <SelectFromOptions
                  options={options.spec_draft_n_max}
                  value={cfg.spec_draft_n_max}
                  onChange={(v) => setCfg((c) => ({ ...c, spec_draft_n_max: v as number }))}
                />
              </Field>
            </Section>
          )}

          <Section title="Generation & misc">
            <div className="grid grid-cols-2 gap-3">
              <Field label="max-tokens (-n)">
                <SelectFromOptions
                  options={options.max_tokens}
                  value={cfg.max_tokens}
                  onChange={(v) => setCfg((c) => ({ ...c, max_tokens: v as number }))}
                />
              </Field>
              <Field label="batch (-b)">
                <SelectFromOptions
                  options={options.batch_size}
                  value={cfg.batch_size}
                  onChange={(v) => setCfg((c) => ({ ...c, batch_size: v as number }))}
                />
              </Field>
              <Field label="ubatch (-ub)">
                <SelectFromOptions
                  options={options.ubatch_size}
                  value={cfg.ubatch_size}
                  onChange={(v) => setCfg((c) => ({ ...c, ubatch_size: v as number }))}
                />
              </Field>
              <Field label="threads (-t / -tb)">
                <div className="flex gap-2">
                  <input
                    type="number"
                    value={cfg.threads_gen}
                    onChange={(e) => setCfg((c) => ({ ...c, threads_gen: Number(e.target.value) }))}
                    className={inputCls}
                  />
                  <input
                    type="number"
                    value={cfg.threads_batch}
                    onChange={(e) => setCfg((c) => ({ ...c, threads_batch: Number(e.target.value) }))}
                    className={inputCls}
                  />
                </div>
              </Field>
            </div>
            <div className="flex gap-4 mt-2">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={cfg.mlock}
                  onChange={(e) => setCfg((c) => ({ ...c, mlock: e.target.checked }))}
                />
                --mlock
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={cfg.verbose}
                  onChange={(e) => setCfg((c) => ({ ...c, verbose: e.target.checked }))}
                />
                --verbose
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="radio"
                  name="mode"
                  checked={(cfg.mode ?? "server") === "server"}
                  onChange={() => setCfg((c) => ({ ...c, mode: "server" }))}
                />
                server (HTTP)
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="radio"
                  name="mode"
                  checked={cfg.mode === "cli"}
                  onChange={() => setCfg((c) => ({ ...c, mode: "cli" }))}
                />
                cli
              </label>
            </div>
          </Section>

        </div>

        {/* ── coluna direita: estimate + comando ────────────────────────── */}
        <div className="border-l border-ink-800 p-5 space-y-4 bg-ink-950 overflow-y-auto sticky top-0">
          <h3 className="text-sm font-medium text-ink-200">Estimativa</h3>
          <EstimatePanel estimate={estimate} loading={estimating} />

          {meta && (
            <div className="text-xs text-ink-400 border-t border-ink-800 pt-3">
              <div className="font-medium text-ink-300 mb-1">{meta.arch}</div>
              <div className="mono space-y-0.5">
                <div>{meta.n_layer} layers · n_embd {meta.n_embd}</div>
                <div>{meta.n_head} heads · KV {Array.isArray(meta.n_head_kv) ? "per-layer" : meta.n_head_kv}</div>
                {meta.is_moe && <div>MoE: {meta.expert_count} experts ({meta.expert_used_count} ativos)</div>}
                {meta.ssm_state_per_layer_mib > 0 && <div>híbrido SSM</div>}
                {meta.swa_window && <div>SWA window {meta.swa_window}</div>}
              </div>
            </div>
          )}

          <div className="border-t border-ink-800 pt-3 space-y-1">
            <h3 className="text-sm font-medium text-ink-200">Comando</h3>
            <textarea
              value={command}
              readOnly
              className="w-full h-32 bg-ink-900 border border-ink-700 rounded p-2 text-[11px] mono text-ink-300 resize-y"
            />
          </div>
        </div>
      </div>
    </Modal>
  );
}

// ─── primitivos auxiliares ─────────────────────────────────────────────────

const inputCls  = "bg-ink-950 border border-ink-700 rounded px-2 py-1.5 text-sm w-full focus:outline-none focus:border-accent";
const selectCls = "bg-ink-950 border border-ink-700 rounded px-2 py-1.5 text-sm w-full focus:outline-none focus:border-accent";

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h3 className="text-xs font-medium uppercase text-ink-400 mb-2 tracking-wider">{title}</h3>
      <div className="space-y-3">{children}</div>
    </section>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs text-ink-400 mb-1 block">{label}</span>
      {children}
    </label>
  );
}

function Toggle({ checked, onChange, label }: {
  checked: boolean; onChange: (v: boolean) => void; label?: string;
}) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className={`px-3 py-1.5 rounded text-sm w-full ${
        checked
          ? "bg-emerald-900/40 text-emerald-300 border border-emerald-700"
          : "bg-ink-950 text-ink-400 border border-ink-700"
      }`}
    >
      {label ?? (checked ? "ON" : "OFF")}
    </button>
  );
}

function SelectFromOptions<T extends number | string>({
  options, value, onChange, allowCustom,
}: {
  options: { value: T; label: string; description?: string }[];
  value: T;
  onChange: (v: T) => void;
  allowCustom?: boolean;
}) {
  const known = options.some((o) => o.value === value);
  return (
    <div className="flex gap-2">
      <select
        value={known ? String(value) : "__custom"}
        onChange={(e) => {
          if (e.target.value === "__custom") return;
          const found = options.find((o) => String(o.value) === e.target.value);
          if (found) onChange(found.value);
        }}
        className={selectCls}
      >
        {options.map((o) => (
          <option key={String(o.value)} value={String(o.value)}>
            {o.label}
          </option>
        ))}
        {allowCustom && <option value="__custom">— customizado —</option>}
      </select>
      {allowCustom && (
        <input
          type="number"
          value={typeof value === "number" ? value : ""}
          onChange={(e) => onChange(Number(e.target.value) as T)}
          className={`${inputCls} w-28`}
        />
      )}
    </div>
  );
}
