import type { Estimate } from "../api/types";
import { fmtMiB } from "../util/format";
import { MemoryBar } from "./MemoryBar";

type Props = {
  estimate: Estimate | null;
  loading?: boolean;
};

// Painel direito do editor: barras de VRAM/RAM com segmentos por componente
// (pesos, kv, compute, ssm, mmproj, mtp draft). Mantém paridade visual com
// `show_memory_estimate` do models.py.
export function EstimatePanel({ estimate, loading }: Props) {
  if (!estimate) {
    return (
      <div className="text-xs text-ink-500">
        {loading ? "calculando…" : "selecione um modelo pra ver a estimativa"}
      </div>
    );
  }
  const e = estimate;
  const vramSegments = [
    { label: "pesos",   value: e.vram_weights,        color: "bg-purple-500" },
    { label: "kv",      value: e.vram_kv,             color: "bg-cyan-500" },
    { label: "compute", value: e.vram_compute,        color: "bg-emerald-500" },
    { label: "ssm",     value: e.vram_ssm,            color: "bg-teal-500" },
    { label: "mmproj",  value: e.vram_mmproj,         color: "bg-amber-500" },
    { label: "mtp kv",  value: e.vram_mtp_kv,         color: "bg-pink-500" },
    { label: "mtp cmp", value: e.vram_mtp_compute,    color: "bg-pink-700" },
  ];
  const ramSegments = [
    { label: "pesos", value: e.ram_weights, color: "bg-purple-500" },
    { label: "kv",    value: e.ram_kv,      color: "bg-cyan-500" },
    { label: "ssm",   value: e.ram_ssm,     color: "bg-teal-500" },
  ];

  return (
    <div className="space-y-4">
      <MemoryBar
        label="VRAM"
        used={e.vram_total}
        total={e.vram_avail}
        segments={vramSegments}
      />
      <MemoryBar
        label="RAM"
        used={e.ram_total}
        total={e.ram_avail}
        segments={ramSegments}
      />

      {e.vram_total_phys && e.vram_avail && e.vram_total_phys - e.vram_avail >= 256 && (
        <p className="text-[11px] text-ink-500">
          VRAM livre considerada: {fmtMiB(e.vram_avail)} de {fmtMiB(e.vram_total_phys)} —
          {" "}{fmtMiB(e.vram_total_phys - e.vram_avail)} já em uso pelo SO/driver/apps.
        </p>
      )}

      <details className="text-xs text-ink-300">
        <summary className="cursor-pointer text-ink-400 hover:text-ink-100 select-none">
          breakdown
        </summary>
        <dl className="grid grid-cols-2 gap-x-3 gap-y-1 mt-2 mono">
          <DT label="VRAM pesos"          v={e.vram_weights} />
          <DT label="VRAM kv"             v={e.vram_kv} />
          <DT label="VRAM compute"        v={e.vram_compute} />
          {e.vram_ssm > 0 && <DT label="VRAM ssm" v={e.vram_ssm} />}
          {e.vram_mmproj > 0 && <DT label="VRAM mmproj" v={e.vram_mmproj} />}
          {(e.vram_mtp_kv + e.vram_mtp_compute) > 0 && (
            <DT label="VRAM mtp-draft" v={e.vram_mtp_kv + e.vram_mtp_compute} />
          )}
          {e.moe_offload_mib > 0 && (
            <DT label="MoE offload" v={e.moe_offload_mib} />
          )}
          <DT label="RAM pesos" v={e.ram_weights} />
          <DT label="RAM kv"    v={e.ram_kv} />
          {e.cache_ram > 0 && (
            <DT label="cache-ram (teto)" v={e.cache_ram} />
          )}
        </dl>
      </details>

      {!e.meta_ok && (
        <p className="text-xs text-amber-400">
          ⚠ Metadados do GGUF não lidos — KV cache não estimado.
        </p>
      )}
    </div>
  );
}

function DT({ label, v }: { label: string; v: number }) {
  return (
    <>
      <dt className="text-ink-500">{label}</dt>
      <dd className="text-right text-ink-200 tabular-nums">{fmtMiB(v)}</dd>
    </>
  );
}
