import { useEffect, useState } from "react";
import { Cpu, RotateCcw, Loader2, AlertTriangle, Thermometer, Activity, Zap, MemoryStick } from "lucide-react";
import { api } from "../api/client";
import type { NvidiaStatus } from "../api/types";

// Dashboard de GPU NVIDIA: roda nvidia-smi a cada 2 s, mostra cada GPU
// detectada com sensores em cards.
export function NvidiaPage() {
  const [status, setStatus] = useState<NvidiaStatus | null>(null);
  const [error, setError]   = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function tick() {
      try {
        const s = await api.nvidia();
        if (!cancelled) {
          setStatus(s);
          setError(null);
          setLoading(false);
        }
      } catch (e) {
        if (!cancelled) {
          setError((e as Error).message);
          setLoading(false);
        }
      }
    }
    tick();
    if (!autoRefresh) return;
    const id = setInterval(tick, 2_000);
    return () => { cancelled = true; clearInterval(id); };
  }, [autoRefresh]);

  return (
    <section className="bg-ink-900 border border-ink-800 rounded-lg overflow-hidden">
      <div className="px-4 py-3 border-b border-ink-800 flex items-center gap-3">
        <h2 className="font-medium text-ink-100 flex items-center gap-2">
          <Cpu className="w-4 h-4" /> NVIDIA status
        </h2>
        <div className="flex-1" />
        <label className="text-xs text-ink-400 flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={autoRefresh}
            onChange={(e) => setAutoRefresh(e.target.checked)}
          />
          auto-refresh 2 s
        </label>
        <button
          onClick={() => api.nvidia().then(setStatus).catch((e) => setError((e as Error).message))}
          className="text-xs px-2 py-1 rounded bg-ink-800 hover:bg-ink-700 flex items-center gap-1"
        >
          <RotateCcw className="w-3 h-3" /> refresh
        </button>
      </div>

      <div className="p-4">
        {loading && (
          <div className="text-ink-400 text-sm flex items-center gap-1">
            <Loader2 className="w-3 h-3 animate-spin" /> consultando nvidia-smi…
          </div>
        )}

        {error && (
          <div className="bg-red-950/60 border border-red-800 text-red-200 rounded p-3 text-sm mono">
            {error}
          </div>
        )}

        {status && !status.available && (
          <div className="bg-amber-950/30 border border-amber-800 text-amber-200 rounded p-3 text-sm flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
            <div>
              <div className="font-medium">nvidia-smi indisponível</div>
              <div className="text-xs text-amber-300/70 mt-1">{status.error}</div>
            </div>
          </div>
        )}

        {status && status.available && (
          <div className="grid gap-4 md:grid-cols-2">
            {status.gpus.map((gpu, i) => <GpuCard key={i} gpu={gpu} />)}
          </div>
        )}
      </div>
    </section>
  );
}

function GpuCard({ gpu }: { gpu: Record<string, string> }) {
  const name = gpu["name"] ?? "GPU";
  const driver = gpu["driver_version"] ?? "?";
  const memTotal = parseFloat(gpu["memory.total"] ?? "0");
  const memUsed  = parseFloat(gpu["memory.used"] ?? "0");
  const memFree  = parseFloat(gpu["memory.free"] ?? "0");
  const utilGpu  = gpu["utilization.gpu"] ?? "—";
  const utilMem  = gpu["utilization.memory"] ?? "—";
  const temp     = gpu["temperature.gpu"] ?? "—";
  const fan      = gpu["fan.speed"] ?? "—";
  const power    = gpu["power.draw"] ?? "—";
  const powerLim = gpu["power.limit"] ?? "—";
  const clkSm    = gpu["clocks.sm"] ?? "—";
  const clkMem   = gpu["clocks.mem"] ?? "—";
  const pstate   = gpu["pstate"] ?? "—";

  const memPct = memTotal > 0 ? (memUsed / memTotal) * 100 : 0;

  return (
    <div className="bg-ink-950 border border-ink-800 rounded p-4 space-y-3">
      <div>
        <h3 className="font-medium text-ink-100">{name}</h3>
        <div className="text-xs text-ink-500 mono">driver {driver} · P-state {pstate}</div>
      </div>

      <div>
        <div className="flex items-center justify-between text-xs mb-1">
          <span className="text-ink-300 flex items-center gap-1">
            <MemoryStick className="w-3 h-3" /> VRAM
          </span>
          <span className="mono tabular-nums">
            {(memUsed / 1024).toFixed(2)} / {(memTotal / 1024).toFixed(2)} GB
            <span className="text-ink-500"> · livre {(memFree / 1024).toFixed(2)} GB</span>
          </span>
        </div>
        <div className="h-2 bg-ink-800 rounded overflow-hidden">
          <div
            className={`h-full ${memPct >= 95 ? "bg-red-500" : memPct >= 80 ? "bg-amber-500" : "bg-emerald-500"}`}
            style={{ width: `${memPct}%` }}
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <Stat icon={<Activity className="w-3 h-3 text-cyan-400" />} label="Uso GPU"    value={utilGpu} />
        <Stat icon={<Activity className="w-3 h-3 text-cyan-400" />} label="Uso MEM"    value={utilMem} />
        <Stat icon={<Thermometer className="w-3 h-3 text-red-400" />} label="Temp"     value={`${temp} °C`} />
        <Stat icon={<Activity className="w-3 h-3 text-ink-400" />} label="Fan"          value={fan} />
        <Stat icon={<Zap className="w-3 h-3 text-amber-400" />} label="Power"           value={`${power} / ${powerLim}`} />
        <Stat icon={<Activity className="w-3 h-3 text-purple-400" />} label="Clock SM" value={clkSm} />
        <Stat icon={<Activity className="w-3 h-3 text-purple-400" />} label="Clock Mem" value={clkMem} />
      </div>
    </div>
  );
}

function Stat({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="bg-ink-900 border border-ink-800 rounded px-2 py-1.5 flex items-center justify-between">
      <span className="text-ink-500 flex items-center gap-1">{icon} {label}</span>
      <span className="mono tabular-nums text-ink-200">{value}</span>
    </div>
  );
}
