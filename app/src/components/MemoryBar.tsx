import { estimateStatus, fmtMiB, statusBg, statusText } from "../util/format";

type Props = {
  label: string;
  used: number;
  total: number | null;
  segments?: { label: string; value: number; color: string }[];
  compact?: boolean;
};

// Barra de VRAM/RAM. Quando `segments` é fornecido, mostra os componentes
// empilhados (weights+kv+compute+...) — espelha o `_bar` do models.py.
export function MemoryBar({ label, used, total, segments, compact }: Props) {
  const status = estimateStatus(used, total);
  const pct = total && total > 0 ? Math.min(used / total, 1.0) : 0;
  const overflow = total && used > total ? (used - total) : 0;

  return (
    <div className={compact ? "space-y-1" : "space-y-2"}>
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium text-ink-200">{label}</span>
        <span className={`mono tabular-nums ${statusText[status]}`}>
          {fmtMiB(used)} / {fmtMiB(total ?? 0)}
          {overflow > 0 && (
            <span className="ml-2 text-red-400">+{fmtMiB(overflow)}</span>
          )}
        </span>
      </div>
      <div className={`h-2 ${compact ? "h-1.5" : "h-3"} rounded bg-ink-800 overflow-hidden flex`}>
        {segments && total ? (
          segments.map((s, i) =>
            s.value > 0 ? (
              <div
                key={i}
                className={s.color}
                style={{ width: `${Math.min((s.value / total) * 100, 100)}%` }}
                title={`${s.label}: ${fmtMiB(s.value)}`}
              />
            ) : null,
          )
        ) : (
          <div
            className={statusBg[status]}
            style={{ width: `${pct * 100}%` }}
          />
        )}
      </div>
    </div>
  );
}
