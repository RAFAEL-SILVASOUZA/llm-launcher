import { useEffect, useMemo, useState } from "react";
import {
  Square, Pencil, Rocket, Trash2, ScrollText, Plus, Loader2,
  Copy, ChevronRight, ChevronDown, Brain, Image as ImageIcon, ClipboardCopy, Check,
} from "lucide-react";
import { api } from "../api/client";
import type { Estimate, LaunchConfig, ModelInfo } from "../api/types";
import { estimateStatus, fmtCtx, fmtMiB, modelAliasFromPath, shortPath, statusBg, statusText } from "../util/format";

type SortKey =
  | "alias" | "backend" | "context_window" | "kv_cache"
  | "gpu_layers" | "n_cpu_moe" | "parallel_slots" | "reasoning_budget";

type Props = {
  configs: LaunchConfig[];
  models: ModelInfo[];
  estimates: Record<string, Estimate>;
  onEdit:        (cfg: LaunchConfig) => void;
  onDuplicate:   (cfg: LaunchConfig) => void;
  onDelete:      (cfg: LaunchConfig) => void;
  onLaunch:      (cfg: LaunchConfig) => void;
  onNew:         () => void;
  activeLaunchCfg: LaunchConfig | null;
  onStopLaunch:    () => void;
  onOpenLaunchLogs:() => void;
};

const cfgKey = (c: LaunchConfig) => `${c.model}|${c.backend}`;

export function ConfigGrid({
  configs, models, estimates,
  onEdit, onDuplicate, onDelete, onLaunch, onNew,
  activeLaunchCfg, onStopLaunch, onOpenLaunchLogs,
}: Props) {
  const [query, setQuery] = useState("");
  const [backendFilter, setBackendFilter] = useState<string>("");
  const [sortKey, setSortKey] = useState<SortKey>("alias");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  const [expanded, setExpanded] = useState<string | null>(null);

  const modelByPath = useMemo(
    () => new Map(models.map((m) => [m.path, m])),
    [models],
  );

  const rows = useMemo(() => {
    const filtered = configs.filter((c) => {
      if (backendFilter && (c.backend ?? "turbo") !== backendFilter) return false;
      if (!query.trim()) return true;
      const q = query.toLowerCase();
      const alias = modelAliasFromPath(c.model).toLowerCase();
      return (
        alias.includes(q) ||
        c.model.toLowerCase().includes(q) ||
        (c.backend ?? "turbo").toLowerCase().includes(q) ||
        c.kv_cache.toLowerCase().includes(q)
      );
    });
    const sorted = [...filtered].sort((a, b) => {
      const va: number | string =
        sortKey === "alias" ? modelAliasFromPath(a.model)
        : sortKey === "backend" ? (a.backend ?? "turbo")
        : sortKey === "kv_cache" ? a.kv_cache
        : (a[sortKey] as number ?? 0);
      const vb: number | string =
        sortKey === "alias" ? modelAliasFromPath(b.model)
        : sortKey === "backend" ? (b.backend ?? "turbo")
        : sortKey === "kv_cache" ? b.kv_cache
        : (b[sortKey] as number ?? 0);
      const cmp = typeof va === "number" && typeof vb === "number"
        ? va - vb : String(va).localeCompare(String(vb));
      return sortDir === "asc" ? cmp : -cmp;
    });
    return sorted;
  }, [configs, query, backendFilter, sortKey, sortDir]);

  const setSort = (k: SortKey) => {
    if (k === sortKey) setSortDir(sortDir === "asc" ? "desc" : "asc");
    else { setSortKey(k); setSortDir("asc"); }
  };
  const sortIcon = (k: SortKey) =>
    sortKey === k ? (sortDir === "asc" ? "▲" : "▼") : "";

  const backendChips: { value: string; label: string; color: string }[] = [
    { value: "",        label: "todos",   color: "bg-ink-800 text-ink-200 border-ink-700" },
    { value: "turbo",   label: "turbo",   color: "bg-purple-900/40 text-purple-300 border-purple-800/60" },
    { value: "vanilla", label: "vanilla", color: "bg-blue-900/40   text-blue-300   border-blue-800/60" },
    { value: "mtp",     label: "mtp",     color: "bg-amber-900/40  text-amber-300  border-amber-800/60" },
  ];

  return (
    <section className="bg-ink-900 border border-ink-800 rounded-lg overflow-hidden">
      <div className="px-4 py-3 border-b border-ink-800 flex items-center gap-3 flex-wrap">
        <h2 className="font-medium text-ink-100">Configurações salvas</h2>
        <span className="text-xs text-ink-500">
          {rows.length} de {configs.length}
        </span>
        <div className="flex-1" />
        <input
          type="text"
          placeholder="filtrar por alias, backend, kv…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="bg-ink-950 border border-ink-700 rounded px-3 py-1.5 text-sm w-64 focus:outline-none focus:border-accent"
        />
        <div className="flex items-center gap-1">
          {backendChips.map((b) => (
            <button
              key={b.value || "all"}
              onClick={() => setBackendFilter(b.value)}
              className={`text-xs px-2 py-1 rounded border mono transition-colors ${
                backendFilter === b.value
                  ? `${b.color} ring-1 ring-accent/50`
                  : "bg-ink-950 text-ink-500 border-ink-800 hover:text-ink-300"
              }`}
            >
              {b.label}
            </button>
          ))}
        </div>
        <button
          onClick={onNew}
          className="bg-accent hover:bg-accent/80 text-white px-3 py-1.5 rounded text-sm font-medium flex items-center gap-1"
        >
          <Plus className="w-3 h-3" />nova
        </button>
      </div>

      <div className="overflow-auto max-h-[calc(100vh-280px)]">
        <table className="w-full text-sm">
          <thead className="bg-ink-950 sticky top-0 text-xs uppercase text-ink-400 z-[1]">
            <tr>
              <Th />
              <Th title="estimativa de VRAM da config: cabe / aperta / estoura">●</Th>
              <Th onClick={() => setSort("alias")}          icon={sortIcon("alias")}>Modelo</Th>
              <Th onClick={() => setSort("backend")}        icon={sortIcon("backend")}>Backend</Th>
              <Th onClick={() => setSort("context_window")} icon={sortIcon("context_window")} align="right">Ctx</Th>
              <Th onClick={() => setSort("kv_cache")}       icon={sortIcon("kv_cache")}>KV</Th>
              <Th onClick={() => setSort("gpu_layers")}     icon={sortIcon("gpu_layers")}     align="right">-ngl</Th>
              <Th onClick={() => setSort("n_cpu_moe")}      icon={sortIcon("n_cpu_moe")}      align="right">-ncmoe</Th>
              <Th onClick={() => setSort("parallel_slots")} icon={sortIcon("parallel_slots")} align="right">-np</Th>
              <Th onClick={() => setSort("reasoning_budget")} icon={sortIcon("reasoning_budget")} align="right">think</Th>
              <Th title="multimodal mmproj"><ImageIcon className="w-3 h-3 inline" /></Th>
              <Th align="right">Ações</Th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td colSpan={12} className="px-4 py-8 text-center text-ink-500">
                  {configs.length === 0
                    ? "Nenhuma config salva ainda. Clique em + nova pra começar."
                    : "Nenhuma config bate com o filtro."}
                </td>
              </tr>
            )}
            {rows.map((c, i) => {
              const key = cfgKey(c);
              const activeKey = activeLaunchCfg ? cfgKey(activeLaunchCfg) : null;
              const isActive = key === activeKey;
              const m = modelByPath.get(c.model);
              const alias = m?.alias ?? modelAliasFromPath(c.model);
              const backend = c.backend ?? "turbo";
              const reasoning = c.reasoning_budget;
              const est = estimates[key] ?? null;
              const status = est ? estimateStatus(est.vram_total, est.vram_avail) : "unknown";
              const isExpanded = expanded === key;

              return (
                <FragmentRow
                  key={`${c.model}|${backend}|${i}`}
                  cfg={c}
                  alias={alias}
                  relPath={m?.relative_path}
                  backend={backend}
                  reasoning={reasoning}
                  status={status}
                  est={est}
                  isActive={isActive}
                  isExpanded={isExpanded}
                  onToggleExpand={() => setExpanded(isExpanded ? null : key)}
                  onEdit={onEdit}
                  onDuplicate={onDuplicate}
                  onDelete={onDelete}
                  onLaunch={onLaunch}
                  onStopLaunch={onStopLaunch}
                  onOpenLaunchLogs={onOpenLaunchLogs}
                />
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// ─── linha + painel expandido (memoizado por config para evitar refetch) ──

type RowProps = {
  cfg: LaunchConfig;
  alias: string;
  relPath: string | undefined;
  backend: string;
  reasoning: number | null | undefined;
  status: "ok" | "tight" | "overflow" | "unknown";
  est: Estimate | null;
  isActive: boolean;
  isExpanded: boolean;
  onToggleExpand: () => void;
  onEdit:      (cfg: LaunchConfig) => void;
  onDuplicate: (cfg: LaunchConfig) => void;
  onDelete:    (cfg: LaunchConfig) => void;
  onLaunch:    (cfg: LaunchConfig) => void;
  onStopLaunch:    () => void;
  onOpenLaunchLogs:() => void;
};

function FragmentRow(p: RowProps) {
  const c = p.cfg;
  const backend = p.backend;
  const reasoning = p.reasoning;

  return (
    <>
      <tr
        className={`border-t border-ink-800 hover:bg-ink-800/50 group ${
          p.isActive ? "bg-amber-950/30 border-amber-700/40" : ""
        } ${p.isExpanded ? "bg-ink-800/30" : ""}`}
      >
        <td className="px-2 py-2 align-top w-6">
          <button
            onClick={p.onToggleExpand}
            className="text-ink-500 hover:text-ink-100 p-1 rounded hover:bg-ink-800"
            title={p.isExpanded ? "Recolher" : "Ver comando e breakdown"}
          >
            {p.isExpanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
          </button>
        </td>
        <td className="px-2 py-2 align-top">
          <span
            title={
              p.status === "ok" ? "cabe folgado em VRAM" :
              p.status === "tight" ? "VRAM apertada" :
              p.status === "overflow" ? "ESTOURA a VRAM disponível" :
              "sem estimativa"
            }
            className={`inline-block w-2.5 h-2.5 rounded-full ${statusBg[p.status]}`}
          />
        </td>
        <td className="px-3 py-2 align-top">
          <div className="font-medium text-ink-100 truncate max-w-[28ch] flex items-center gap-2" title={c.model}>
            {p.alias}
            {p.isActive && (
              <span className="text-[11px] px-1.5 py-0.5 rounded bg-amber-900/50 text-amber-300 mono shrink-0 flex items-center gap-1">
                <Loader2 className="w-3 h-3 animate-spin" />running
              </span>
            )}
          </div>
          <div className="text-xs text-ink-500 mono truncate max-w-[40ch]" title={c.model}>
            {shortPath(p.relPath ?? c.model, 60)}
          </div>
        </td>
        <td className="px-3 py-2 align-top">
          <span className={`mono text-xs px-1.5 py-0.5 rounded ${
            backend === "turbo"   ? "bg-purple-900/40 text-purple-300" :
            backend === "vanilla" ? "bg-blue-900/40   text-blue-300" :
                                    "bg-amber-900/40  text-amber-300"
          }`}>{backend}</span>
        </td>
        <td className="px-3 py-2 mono tabular-nums text-right align-top">{fmtCtx(c.context_window)}</td>
        <td className="px-3 py-2 mono align-top">{c.kv_cache}</td>
        <td className="px-3 py-2 mono tabular-nums text-right align-top">{c.gpu_layers}</td>
        <td className="px-3 py-2 mono tabular-nums text-right align-top">{c.n_cpu_moe || "—"}</td>
        <td className="px-3 py-2 mono tabular-nums text-right align-top">{c.parallel_slots}</td>
        <td className="px-3 py-2 align-top text-right">
          <ReasoningCell value={reasoning} />
        </td>
        <td className="px-3 py-2 align-top">
          {c.mmproj && <ImageIcon className="w-3.5 h-3.5 text-amber-400" />}
        </td>
        <td className="px-3 py-2 text-right align-top">
          <ActionButtons
            cfg={c}
            isActive={p.isActive}
            onEdit={p.onEdit}
            onDuplicate={p.onDuplicate}
            onDelete={p.onDelete}
            onLaunch={p.onLaunch}
            onStopLaunch={p.onStopLaunch}
            onOpenLaunchLogs={p.onOpenLaunchLogs}
          />
        </td>
      </tr>
      {p.isExpanded && (
        <tr className="bg-ink-950/60 border-t border-ink-800">
          <td colSpan={12} className="px-6 py-4">
            <ExpandedPanel cfg={c} est={p.est} />
          </td>
        </tr>
      )}
    </>
  );
}

function ActionButtons({
  cfg, isActive,
  onEdit, onDuplicate, onDelete, onLaunch, onStopLaunch, onOpenLaunchLogs,
}: {
  cfg: LaunchConfig; isActive: boolean;
  onEdit:      (cfg: LaunchConfig) => void;
  onDuplicate: (cfg: LaunchConfig) => void;
  onDelete:    (cfg: LaunchConfig) => void;
  onLaunch:    (cfg: LaunchConfig) => void;
  onStopLaunch:    () => void;
  onOpenLaunchLogs:() => void;
}) {
  const btnBase = "w-7 h-7 rounded flex items-center justify-center transition-colors";
  return (
    <div className="inline-flex gap-1 opacity-70 group-hover:opacity-100 items-center">
      <button
        onClick={() => onEdit(cfg)}
        className={`${btnBase} bg-ink-800 hover:bg-ink-700 text-ink-300`}
        title="Editar"
      ><Pencil className="w-3.5 h-3.5" /></button>

      <button
        onClick={() => onDuplicate(cfg)}
        className={`${btnBase} bg-ink-800 hover:bg-ink-700 text-ink-300`}
        title="Duplicar config (copia tudo, escolha outro backend)"
      ><Copy className="w-3.5 h-3.5" /></button>

      {isActive ? (
        <>
          <button
            onClick={onStopLaunch}
            className={`${btnBase} bg-red-900/60 hover:bg-red-800 text-red-100`}
            title="Interromper launch"
          ><Square className="w-3.5 h-3.5" /></button>
          <button
            onClick={onOpenLaunchLogs}
            className={`${btnBase} bg-ink-800 hover:bg-ink-700 text-ink-300`}
            title="Ver logs do launch"
          ><ScrollText className="w-3.5 h-3.5" /></button>
        </>
      ) : (
        <button
          onClick={() => onLaunch(cfg)}
          className={`${btnBase} bg-accent hover:bg-accent/80 text-white`}
          title="Subir o llama-server com esta config"
        ><Rocket className="w-3.5 h-3.5" /></button>
      )}

      <button
        onClick={() => onDelete(cfg)}
        className={`${btnBase} bg-ink-800 hover:bg-red-900/60 hover:text-red-200 text-ink-300`}
        title="Remover esta config"
      ><Trash2 className="w-3.5 h-3.5" /></button>
    </div>
  );
}

function ReasoningCell({ value }: { value: number | null | undefined }) {
  if (value === null || value === undefined) {
    return <span className="text-ink-500" title="reasoning não setado (default do modelo)">—</span>;
  }
  if (value === 0) {
    return (
      <span className="inline-flex items-center gap-1 text-ink-400 text-xs" title="--reasoning off">
        <Brain className="w-3 h-3 opacity-50" />off
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 text-cyan-300 text-xs mono tabular-nums" title={`reasoning-budget ${value}`}>
      <Brain className="w-3 h-3 text-cyan-400" />
      {value === -1 ? "∞" : value.toLocaleString()}
    </span>
  );
}

// ─── painel expandido: comando + breakdown VRAM/RAM ──────────────────────

function ExpandedPanel({ cfg, est }: { cfg: LaunchConfig; est: Estimate | null }) {
  const [command, setCommand] = useState<string | null>(null);
  const [error, setError]     = useState<string | null>(null);
  const [copied, setCopied]   = useState(false);

  useEffect(() => {
    setCommand(null); setError(null);
    let cancelled = false;
    api.buildCommand(cfg)
      .then((r) => !cancelled && setCommand(r.command))
      .catch((e) => !cancelled && setError((e as Error).message));
    return () => { cancelled = true; };
  }, [cfg]);

  const handleCopy = async () => {
    if (!command) return;
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* ignore */ }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-[1fr_280px] gap-4">
      {/* comando */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <h4 className="text-xs uppercase tracking-wider text-ink-400">Comando</h4>
          <button
            disabled={!command}
            onClick={handleCopy}
            className="text-xs px-2 py-0.5 rounded bg-ink-800 hover:bg-ink-700 text-ink-300 flex items-center gap-1 disabled:opacity-50"
            title="Copiar"
          >
            {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <ClipboardCopy className="w-3 h-3" />}
            {copied ? "copiado" : "copiar"}
          </button>
        </div>
        {error && <div className="text-xs text-red-400 mono">{error}</div>}
        {!command && !error && (
          <div className="text-xs text-ink-500 flex items-center gap-1">
            <Loader2 className="w-3 h-3 animate-spin" /> construindo…
          </div>
        )}
        {command && (
          <pre className="bg-ink-900 border border-ink-800 rounded p-2 text-[11px] mono text-ink-200 whitespace-pre-wrap break-all max-h-48 overflow-auto">
            {command}
          </pre>
        )}
      </div>

      {/* breakdown */}
      <div>
        <h4 className="text-xs uppercase tracking-wider text-ink-400 mb-1">Estimativa</h4>
        {!est ? (
          <div className="text-xs text-ink-500">sem dados</div>
        ) : (
          <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs mono">
            <DT label="VRAM total"     v={est.vram_total} hi />
            <DT label="VRAM livre"     v={est.vram_avail ?? 0} hi />
            <DT label="  ↳ pesos"      v={est.vram_weights} />
            <DT label="  ↳ kv"         v={est.vram_kv} />
            <DT label="  ↳ compute"    v={est.vram_compute} />
            {est.vram_ssm > 0 && <DT label="  ↳ ssm" v={est.vram_ssm} />}
            {est.vram_mmproj > 0 && <DT label="  ↳ mmproj" v={est.vram_mmproj} />}
            {(est.vram_mtp_kv + est.vram_mtp_compute) > 0 && (
              <DT label="  ↳ mtp" v={est.vram_mtp_kv + est.vram_mtp_compute} />
            )}
            <DT label="RAM total"      v={est.ram_total} hi />
            {est.moe_offload_mib > 0 && <DT label="MoE offload" v={est.moe_offload_mib} />}
          </dl>
        )}
      </div>
    </div>
  );
}

function DT({ label, v, hi }: { label: string; v: number; hi?: boolean }) {
  return (
    <>
      <dt className={hi ? "text-ink-200" : "text-ink-500"}>{label}</dt>
      <dd className={`text-right tabular-nums ${hi ? "text-ink-100" : "text-ink-300"}`}>{fmtMiB(v)}</dd>
    </>
  );
}

function Th({
  children, onClick, icon, align, title,
}: {
  children?: React.ReactNode;
  onClick?: () => void;
  icon?: string;
  align?: "right";
  title?: string;
}) {
  return (
    <th
      onClick={onClick}
      title={title}
      className={`px-3 py-2 font-medium ${align === "right" ? "text-right" : "text-left"} ${
        onClick ? "cursor-pointer hover:text-ink-100 select-none" : ""
      }`}
    >
      {children}{icon && <span className="ml-1 text-accent-fg">{icon}</span>}
    </th>
  );
}
