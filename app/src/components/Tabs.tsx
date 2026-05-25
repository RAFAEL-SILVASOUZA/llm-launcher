import { ReactNode } from "react";

export type Tab<T extends string> = {
  id: T;
  label: string;
  icon?: ReactNode;
  badge?: number | string;
};

type Props<T extends string> = {
  tabs: Tab<T>[];
  active: T;
  onChange: (id: T) => void;
};

export function Tabs<T extends string>({ tabs, active, onChange }: Props<T>) {
  return (
    <nav className="border-b border-ink-800 bg-ink-900/60 px-4">
      <ul className="flex gap-1">
        {tabs.map((t) => {
          const isActive = t.id === active;
          return (
            <li key={t.id}>
              <button
                onClick={() => onChange(t.id)}
                className={`flex items-center gap-2 px-4 py-2.5 text-sm border-b-2 -mb-px transition-colors ${
                  isActive
                    ? "border-accent text-ink-100"
                    : "border-transparent text-ink-400 hover:text-ink-200 hover:bg-ink-800/40"
                }`}
              >
                {t.icon}
                <span>{t.label}</span>
                {t.badge != null && (
                  <span className="text-[11px] mono px-1.5 py-0.5 rounded bg-ink-800 text-ink-400">
                    {t.badge}
                  </span>
                )}
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
