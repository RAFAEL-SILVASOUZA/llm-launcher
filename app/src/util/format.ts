// Helpers de formatação compartilhados pela UI.

export function fmtMiB(mib: number | null | undefined): string {
  if (mib == null || mib === 0) return "—";
  if (mib >= 1024) return `${(mib / 1024).toFixed(2)} GB`;
  return `${mib} MiB`;
}

// Mostra o valor cru do context window com separador de milhar. Notação "k"
// (base 1024) confunde porque 131072 = 128k exato — usuário que digitou
// "131072" enxergava "128k" e parecia arredondamento.
export function fmtCtx(ctx: number): string {
  return ctx.toLocaleString();
}

export function fmtPct(num: number, denom: number | null | undefined): string {
  if (!denom || denom <= 0) return "—";
  return `${((num / denom) * 100).toFixed(1)}%`;
}

export function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

export function modelAliasFromPath(p: string): string {
  const name = p.split(/[\\/]/).pop() ?? p;
  return name.replace(/\.gguf$/i, "");
}

export function shortPath(p: string, maxLen = 80): string {
  if (p.length <= maxLen) return p;
  const head = p.slice(0, 24);
  const tail = p.slice(-(maxLen - 28));
  return `${head}…${tail}`;
}

export type EstimateStatus = "ok" | "tight" | "overflow" | "unknown";

export function estimateStatus(total: number, avail: number | null): EstimateStatus {
  if (!avail || avail <= 0) return "unknown";
  const pct = total / avail;
  if (pct >= 0.95) return "overflow";
  if (pct >= 0.80) return "tight";
  return "ok";
}

export const statusBg: Record<EstimateStatus, string> = {
  ok:       "bg-emerald-500",
  tight:    "bg-amber-500",
  overflow: "bg-red-500",
  unknown:  "bg-ink-500",
};

export const statusText: Record<EstimateStatus, string> = {
  ok:       "text-emerald-400",
  tight:    "text-amber-400",
  overflow: "text-red-400",
  unknown:  "text-ink-400",
};
