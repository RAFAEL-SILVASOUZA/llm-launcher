import { RotateCcw, Zap } from "lucide-react";
import type { BackendStatus, SystemInfo } from "../api/types";
import { MemoryBar } from "./MemoryBar";

type Props = {
  system: SystemInfo | null;
  backends: BackendStatus[];
  onRefresh: () => void;
};

export function Header({ system, backends, onRefresh }: Props) {
  return (
    <header className="border-b border-ink-800 bg-ink-900/90 backdrop-blur-sm sticky top-0 z-10">
      <div className="px-6 py-3 flex items-center gap-6">
        <div className="flex items-baseline gap-2 mr-4">
          <h1 className="text-lg font-semibold tracking-tight">
            <span className="text-accent-fg">LLM</span> Launcher
          </h1>
          <span className="text-xs text-ink-500 mono">v0.1</span>
        </div>

        <div className="flex-1 grid grid-cols-2 gap-4 max-w-xl">
          <MemoryBar
            compact
            label="VRAM"
            used={
              system?.vram_total_mib != null && system?.vram_free_mib != null
                ? system.vram_total_mib - system.vram_free_mib
                : 0
            }
            total={system?.vram_total_mib ?? null}
          />
          <MemoryBar
            compact
            label="RAM"
            used={
              system?.ram_total_mib != null && system?.ram_avail_mib != null
                ? system.ram_total_mib - system.ram_avail_mib
                : 0
            }
            total={system?.ram_total_mib ?? null}
          />
        </div>

        <div className="flex items-center gap-1">
          {backends.map((b) => (
            <span
              key={b.name}
           title={`server: ${b.server_available ? "ok" : "not found"} ${b.server_path}\ncli: ${
              b.cli_available ? "ok" : "not found"
            }`}
              className={`text-xs px-2 py-1 rounded mono ${
                b.server_available
                  ? "bg-emerald-900/40 text-emerald-300 border border-emerald-700/50"
                  : "bg-ink-800 text-ink-500 border border-ink-700"
              }`}
            >
              {b.label}
              {b.supports_spec_mtp && <Zap className="w-3 h-3 inline ml-0.5 text-amber-400" />}
            </span>
          ))}
        </div>

        <button
          onClick={onRefresh}
          className="text-xs text-ink-400 hover:text-ink-100 px-2 py-1 rounded hover:bg-ink-800 flex items-center gap-1"
          title="Recarregar sistema/backends/configs"
        >
          <RotateCcw className="w-3 h-3" />refresh
        </button>
      </div>
    </header>
  );
}
