import { useMemo, useState } from "react";
import type { LaunchConfig, ModelInfo } from "../api/types";
import { fmtCtx, modelAliasFromPath, shortPath } from "../util/format";

type SortKey =
  | "alias" | "backend" | "context_window" | "kv_cache"
  | "gpu_layers" | "n_cpu_moe" | "parallel_slots" | "reasoning_budget";

type Props = {
  configs: LaunchConfig[];
  models: ModelInfo[];
  onEdit:    (cfg: LaunchConfig) => void;
  onDelete:  (cfg: LaunchConfig) => void;
  onLaunch:  (cfg: LaunchConfig) => void;
  onNew:     () => void;
};

export function ConfigGrid({ configs, models, onEdit, onDelete, onLaunch, onNew }: Props) {
  const [query, setQuery] = useState("");
  const [backendFilter, setBackendFilter] = useState<string>("");
  const [sortKey, setSortKey] = useState<SortKey>("alias");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");

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
        <select
          value={backendFilter}
          onChange={(e) => setBackendFilter(e.target.value)}
          className="bg-ink-950 border border-ink-700 rounded px-2 py-1.5 text-sm focus:outline-none focus:border-accent"
        >
          <option value="">todos backends</option>
          <option value="turbo">turbo</option>
          <option value="vanilla">vanilla</option>
          <option value="mtp">mtp</option>
        </select>
        <button
          onClick={onNew}
          className="bg-accent hover:bg-accent/80 text-white px-3 py-1.5 rounded text-sm font-medium"
        >
          + nova
        </button>
      </div>

      <div className="overflow-auto max-h-[calc(100vh-280px)]">
        <table className="w-full text-sm">
          <thead className="bg-ink-950 sticky top-0 text-xs uppercase text-ink-400 z-[1]">
            <tr>
              <Th onClick={() => setSort("alias")}          icon={sortIcon("alias")}>          Modelo</Th>
              <Th onClick={() => setSort("backend")}        icon={sortIcon("backend")}>        Backend</Th>
              <Th onClick={() => setSort("context_window")} icon={sortIcon("context_window")} align="right">Ctx</Th>
              <Th onClick={() => setSort("kv_cache")}       icon={sortIcon("kv_cache")}>      KV</Th>
              <Th onClick={() => setSort("gpu_layers")}     icon={sortIcon("gpu_layers")}     align="right">-ngl</Th>
              <Th onClick={() => setSort("n_cpu_moe")}      icon={sortIcon("n_cpu_moe")}      align="right">-ncmoe</Th>
              <Th onClick={() => setSort("parallel_slots")} icon={sortIcon("parallel_slots")} align="right">-np</Th>
              <Th onClick={() => setSort("reasoning_budget")} icon={sortIcon("reasoning_budget")} align="right">think</Th>
              <Th>mm</Th>
              <Th align="right">Ações</Th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td colSpan={10} className="px-4 py-8 text-center text-ink-500">
                  {configs.length === 0
                    ? "Nenhuma config salva ainda. Clique em + nova pra começar."
                    : "Nenhuma config bate com o filtro."}
                </td>
              </tr>
            )}
            {rows.map((c, i) => {
              const m = modelByPath.get(c.model);
              const alias = m?.alias ?? modelAliasFromPath(c.model);
              const backend = c.backend ?? "turbo";
              const reasoningStr =
                c.reasoning_budget === null || c.reasoning_budget === undefined ? "—"
                : c.reasoning_budget === 0 ? "off"
                : c.reasoning_budget === -1 ? "∞"
                : String(c.reasoning_budget);
              return (
                <tr
                  key={`${c.model}|${backend}|${i}`}
                  className="border-t border-ink-800 hover:bg-ink-800/50 group"
                >
                  <td className="px-3 py-2">
                    <div className="font-medium text-ink-100 truncate max-w-[28ch]" title={c.model}>
                      {alias}
                    </div>
                    <div className="text-xs text-ink-500 mono truncate max-w-[40ch]" title={c.model}>
                      {shortPath(m?.relative_path ?? c.model, 60)}
                    </div>
                  </td>
                  <td className="px-3 py-2">
                    <span className={`mono text-xs px-1.5 py-0.5 rounded ${
                      backend === "turbo"   ? "bg-purple-900/40 text-purple-300" :
                      backend === "vanilla" ? "bg-blue-900/40   text-blue-300" :
                                              "bg-amber-900/40  text-amber-300"
                    }`}>{backend}</span>
                  </td>
                  <td className="px-3 py-2 mono tabular-nums text-right">{fmtCtx(c.context_window)}</td>
                  <td className="px-3 py-2 mono">{c.kv_cache}</td>
                  <td className="px-3 py-2 mono tabular-nums text-right">{c.gpu_layers}</td>
                  <td className="px-3 py-2 mono tabular-nums text-right">{c.n_cpu_moe || "—"}</td>
                  <td className="px-3 py-2 mono tabular-nums text-right">{c.parallel_slots}</td>
                  <td className="px-3 py-2 mono tabular-nums text-right">{reasoningStr}</td>
                  <td className="px-3 py-2">{c.mmproj ? "🖼" : ""}</td>
                  <td className="px-3 py-2 text-right">
                    <div className="inline-flex gap-1 opacity-60 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={() => onEdit(c)}
                        className="text-xs px-2 py-1 rounded bg-ink-800 hover:bg-ink-700"
                        title="Editar"
                      >Edit</button>
                      <button
                        onClick={() => onLaunch(c)}
                        className="text-xs px-2 py-1 rounded bg-accent hover:bg-accent/80 text-white"
                        title="Subir o llama-server com esta config"
                      >▶ Launch</button>
                      <button
                        onClick={() => onDelete(c)}
                        className="text-xs px-2 py-1 rounded bg-ink-800 hover:bg-red-900/60 hover:text-red-200"
                        title="Remover esta config"
                      >🗑</button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Th({
  children, onClick, icon, align,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  icon?: string;
  align?: "right";
}) {
  return (
    <th
      onClick={onClick}
      className={`px-3 py-2 font-medium ${align === "right" ? "text-right" : "text-left"} ${
        onClick ? "cursor-pointer hover:text-ink-100 select-none" : ""
      }`}
    >
      {children}{icon && <span className="ml-1 text-accent-fg">{icon}</span>}
    </th>
  );
}
