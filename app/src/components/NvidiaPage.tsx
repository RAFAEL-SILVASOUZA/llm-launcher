import { useEffect, useRef, useState } from "react";
import { Cpu, RotateCcw, Loader2, AlertTriangle, Thermometer, Activity, Zap, MemoryStick, LineChart, HardDrive, Lock, Unlock } from "lucide-react";
import { api } from "../api/client";
import type { NvidiaStatus, SystemInfo } from "../api/types";

// Janela rolante de histórico: 60 amostras × 2 s = 2 min visíveis nos gráficos.
const HISTORY_LEN = 60;

type Sample = { vramPct: number; gpuPct: number; ramPct: number };

// Dashboard de GPU NVIDIA: roda nvidia-smi a cada 2 s, mostra cada GPU
// detectada com sensores em cards + mini-gráficos de VRAM/GPU à direita.
export function NvidiaPage() {
  const [status, setStatus] = useState<NvidiaStatus | null>(null);
  const [error, setError]   = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [loading, setLoading] = useState(true);
  // Histórico por índice de GPU; mantido em ref pra não disparar re-render
  // duplicado (escrevemos junto com status no mesmo tick).
  const [history, setHistory] = useState<Record<number, Sample[]>>({});
  const historyRef = useRef(history);
  historyRef.current = history;

  // Lock de clock da VRAM (P2 fixo @ 10251 MHz). `supported` vem do backend
  // (tarefas agendadas registradas?); o estado ligado/desligado é derivado do
  // clocks.mem que o tick de 2 s já traz — sem polling extra.
  const [lockSupported, setLockSupported] = useState(false);
  const [lockMhz, setLockMhz] = useState(10251);
  const [lockBusy, setLockBusy] = useState(false);
  const [lockError, setLockError] = useState<string | null>(null);

  useEffect(() => {
    api.vramLock()
      .then((s) => { setLockSupported(s.supported); setLockMhz(s.lock_mhz); })
      .catch(() => setLockSupported(false));
  }, []);

  const memClock = status?.available ? parseInt(status.gpus[0]?.["clocks.mem"] ?? "", 10) : NaN;
  const locked: boolean | null = Number.isFinite(memClock) ? memClock >= lockMhz : null;

  async function toggleLock() {
    if (locked === null || lockBusy) return;
    setLockBusy(true);
    setLockError(null);
    try {
      await api.vramLockSet(!locked);
      // Re-consulta na hora pro botão refletir o novo clock sem esperar o tick.
      const s = await api.nvidia();
      setStatus(s);
    } catch (e) {
      setLockError((e as Error).message);
    } finally {
      setLockBusy(false);
    }
  }

  useEffect(() => {
    let cancelled = false;
    async function tick() {
      try {
        // Pega nvidia-smi e RAM no mesmo tick — se um falhar, o outro ainda
        // aparece. system() é barato no backend (cache de 2 s).
        const [s, sys] = await Promise.all([
          api.nvidia(),
          api.system().catch(() => null as SystemInfo | null),
        ]);
        if (cancelled) return;
        setStatus(s);
        setError(null);
        setLoading(false);
        if (s.available) {
          const ramTotal = sys?.ram_total_mib ?? 0;
          const ramAvail = sys?.ram_avail_mib ?? 0;
          const ramPct   = ramTotal > 0 ? ((ramTotal - ramAvail) / ramTotal) * 100 : 0;
          const next: Record<number, Sample[]> = { ...historyRef.current };
          s.gpus.forEach((g, i) => {
            const memTotal = parseFloat(g["memory.total"] ?? "0");
            const memUsed  = parseFloat(g["memory.used"]  ?? "0");
            const vramPct  = memTotal > 0 ? (memUsed / memTotal) * 100 : 0;
            const gpuPct   = parseFloat(g["utilization.gpu"] ?? "0") || 0;
            const prev = next[i] ?? [];
            const updated = prev.concat({ vramPct, gpuPct, ramPct });
            if (updated.length > HISTORY_LEN) updated.splice(0, updated.length - HISTORY_LEN);
            next[i] = updated;
          });
          setHistory(next);
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
        {lockSupported && (
          <button
            onClick={toggleLock}
            disabled={lockBusy || locked === null}
            title={
              locked
                ? `Clock da VRAM travado em ${lockMhz} MHz (evita queda P2→P3 na inferência, +35% t/s; idle ~90 W). Clique para destravar.`
                : `Clock da VRAM livre (driver pode rebaixar para 5001 MHz na inferência e derrubar o t/s). Clique para travar em ${lockMhz} MHz. O lock religa sozinho no boot.`
            }
            className={`text-xs px-2 py-1 rounded flex items-center gap-1 disabled:opacity-50 ${
              locked
                ? "bg-emerald-900/60 border border-emerald-700 text-emerald-200 hover:bg-emerald-900"
                : "bg-ink-800 border border-ink-700 text-ink-300 hover:bg-ink-700"
            }`}
          >
            {lockBusy
              ? <Loader2 className="w-3 h-3 animate-spin" />
              : locked
                ? <Lock className="w-3 h-3" />
                : <Unlock className="w-3 h-3" />}
            VRAM lock {locked === null ? "—" : locked ? "ON" : "OFF"}
          </button>
        )}
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

        {lockError && (
          <div className="bg-red-950/60 border border-red-800 text-red-200 rounded p-3 text-sm mono mb-4">
            VRAM lock: {lockError}
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
          <div className="space-y-4">
            {status.gpus.map((gpu, i) => (
              <div key={i} className="grid gap-4 lg:grid-cols-2">
                <GpuCard gpu={gpu} />
                <GpuCharts history={history[i] ?? []} />
              </div>
            ))}
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

function GpuCharts({ history }: { history: Sample[] }) {
  const vram = history.map((s) => s.vramPct);
  const gpu  = history.map((s) => s.gpuPct);
  const ram  = history.map((s) => s.ramPct);
  const lastVram = vram.length ? vram[vram.length - 1] : 0;
  const lastGpu  = gpu.length  ? gpu[gpu.length - 1]  : 0;
  const lastRam  = ram.length  ? ram[ram.length - 1]  : 0;
  // 2 s por amostra; horizonte é histórico real, não a janela cheia.
  const seconds  = history.length * 2;

  return (
    <div className="bg-ink-950 border border-ink-800 rounded p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-medium text-ink-100 flex items-center gap-2 text-sm">
          <LineChart className="w-4 h-4 text-accent-fg" /> Histórico
        </h3>
        <span className="text-[11px] text-ink-500 mono">
          {history.length === 0 ? "aguardando dados…" : `últimos ${seconds}s · ${history.length} amostras`}
        </span>
      </div>

      <ChartRow
        label="VRAM"
        icon={<MemoryStick className="w-3 h-3" />}
        values={vram}
        last={lastVram}
        color="#10b981"
        fill="#10b98133"
      />
      <ChartRow
        label="GPU"
        icon={<Activity className="w-3 h-3" />}
        values={gpu}
        last={lastGpu}
        color="#06b6d4"
        fill="#06b6d433"
      />
      <ChartRow
        label="RAM"
        icon={<HardDrive className="w-3 h-3" />}
        values={ram}
        last={lastRam}
        color="#a78bfa"
        fill="#a78bfa33"
      />
    </div>
  );
}

function ChartRow({
  label, icon, values, last, color, fill,
}: {
  label: string;
  icon: React.ReactNode;
  values: number[];
  last: number;
  color: string;
  fill: string;
}) {
  return (
    <div>
      <div className="flex items-center justify-between text-xs mb-1">
        <span className="text-ink-300 flex items-center gap-1">{icon} {label}</span>
        <span className="mono tabular-nums text-ink-200">{last.toFixed(0)}%</span>
      </div>
      <Sparkline values={values} color={color} fill={fill} />
    </div>
  );
}

// SVG sparkline 0–100%. Usa viewBox pra largura responsiva; altura fixa em
// 56 px. Quando a série tá vazia, devolve um quadro vazio com gridlines pra
// não pular de altura quando começarem a chegar amostras.
function Sparkline({ values, color, fill }: { values: number[]; color: string; fill: string }) {
  const w = 320;
  const h = 56;
  const pad = 4;
  const max = 100;

  const innerW = w - 2 * pad;
  const innerH = h - 2 * pad;

  const N = Math.max(values.length, 2);
  const dx = innerW / (N - 1);

  const points = values.length === 0
    ? []
    : values.map((v, i) => {
        const x = pad + i * dx;
        const y = pad + innerH * (1 - Math.max(0, Math.min(max, v)) / max);
        return [x, y] as const;
      });

  const linePoints = points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const areaPoints = points.length > 0
    ? `${pad},${h - pad} ${linePoints} ${(pad + (points.length - 1) * dx).toFixed(1)},${h - pad}`
    : "";

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="none"
      className="w-full h-14 bg-ink-900/40 border border-ink-800 rounded"
    >
      {/* gridlines 25/50/75% */}
      {[0.25, 0.5, 0.75].map((g) => (
        <line
          key={g}
          x1={pad} x2={w - pad}
          y1={pad + innerH * (1 - g)} y2={pad + innerH * (1 - g)}
          stroke="#1f2937" strokeWidth="0.5" strokeDasharray="2,3"
        />
      ))}
      {points.length > 0 && (
        <>
          <polygon points={areaPoints} fill={fill} />
          <polyline points={linePoints} fill="none" stroke={color} strokeWidth="1.5" />
        </>
      )}
    </svg>
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
