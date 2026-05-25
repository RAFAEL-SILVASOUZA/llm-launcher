import { useMemo, useState } from "react";
import { Trash2, Image as ImageIcon, Zap, Brain, FileText, AlertTriangle, Loader2 } from "lucide-react";
import { api } from "../api/client";
import type { DeletePlan, ModelInfo } from "../api/types";
import { fmtMiB, shortPath } from "../util/format";
import { Modal } from "./Modal";

type Props = {
  models: ModelInfo[];
  onAfterDelete: () => void;
};

export function ModelsPage({ models, onAfterDelete }: Props) {
  const [query, setQuery] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<ModelInfo | null>(null);
  const [plan, setPlan] = useState<DeletePlan | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [planLoading, setPlanLoading] = useState(false);

  const filtered = useMemo(() => {
    if (!query.trim()) return models;
    const q = query.toLowerCase();
    return models.filter(
      (m) =>
        m.alias.toLowerCase().includes(q) ||
        m.relative_path.toLowerCase().includes(q),
    );
  }, [models, query]);

  async function openDelete(m: ModelInfo) {
    setDeleteTarget(m);
    setPlan(null);
    setPlanLoading(true);
    try {
      const p = await api.planDeleteModel(m.path);
      setPlan(p);
    } catch (e) {
      alert(`Falha ao planejar delete: ${(e as Error).message}`);
      setDeleteTarget(null);
    } finally {
      setPlanLoading(false);
    }
  }

  async function confirmDelete() {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      await api.deleteModel(deleteTarget.path);
      setDeleteTarget(null);
      setPlan(null);
      onAfterDelete();
    } catch (e) {
      alert(`Falha ao apagar: ${(e as Error).message}`);
    } finally {
      setDeleting(false);
    }
  }

  return (
    <section className="bg-ink-900 border border-ink-800 rounded-lg overflow-hidden">
      <div className="px-4 py-3 border-b border-ink-800 flex items-center gap-3">
        <h2 className="font-medium text-ink-100">Modelos no disco</h2>
        <span className="text-xs text-ink-500">{filtered.length} de {models.length}</span>
        <div className="flex-1" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="filtrar…"
          className="bg-ink-950 border border-ink-700 rounded px-3 py-1.5 text-sm w-64 focus:outline-none focus:border-accent"
        />
      </div>

      <div className="overflow-auto max-h-[calc(100vh-220px)]">
        <table className="w-full text-sm">
          <thead className="bg-ink-950 sticky top-0 text-xs uppercase text-ink-400 z-[1]">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Alias / Caminho</th>
              <th className="px-3 py-2 text-right font-medium">Tamanho</th>
              <th className="px-3 py-2 font-medium">Flags</th>
              <th className="px-3 py-2 text-right font-medium">Ações</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 && (
              <tr>
                <td colSpan={4} className="px-4 py-8 text-center text-ink-500">
                  Nenhum modelo. Use a aba Download pra baixar do HuggingFace.
                </td>
              </tr>
            )}
            {filtered.map((m) => (
              <tr key={m.path} className="border-t border-ink-800 hover:bg-ink-800/50 group">
                <td className="px-3 py-2">
                  <div className="font-medium text-ink-100 truncate max-w-[40ch]" title={m.path}>
                    {m.alias}
                  </div>
                  <div className="text-xs text-ink-500 mono truncate max-w-[60ch]" title={m.path}>
                    {shortPath(m.relative_path, 80)}
                  </div>
                </td>
                <td className="px-3 py-2 mono tabular-nums text-right">{fmtMiB(m.size_mib)}</td>
                <td className="px-3 py-2">
                  <div className="flex items-center gap-1.5 text-xs">
                    {m.mmproj && <span title="tem mmproj"><ImageIcon className="w-3.5 h-3.5 text-amber-400" /></span>}
                    {m.is_mtp && <span title="modelo MTP"><Zap className="w-3.5 h-3.5 text-pink-400" /></span>}
                    {m.is_thinking && <span title="modelo thinking"><Brain className="w-3.5 h-3.5 text-cyan-400" /></span>}
                  </div>
                </td>
                <td className="px-3 py-2 text-right">
                  <button
                    onClick={() => openDelete(m)}
                    className="text-xs px-2 py-1 rounded bg-ink-800 hover:bg-red-900/60 hover:text-red-200 opacity-60 group-hover:opacity-100 flex items-center gap-1 ml-auto"
                    title="apagar do disco"
                  >
                    <Trash2 className="w-3 h-3" />delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* ── confirm modal ──────────────────────────────────────────── */}
      <Modal
        open={!!deleteTarget}
        onClose={() => { if (!deleting) { setDeleteTarget(null); setPlan(null); } }}
        title={`Apagar ${deleteTarget?.alias ?? ""}?`}
        width="max-w-2xl"
        footer={
          <>
            <button
              onClick={() => { setDeleteTarget(null); setPlan(null); }}
              disabled={deleting}
              className="px-3 py-1.5 text-sm rounded bg-ink-800 hover:bg-ink-700"
            >
              Cancelar
            </button>
            <button
              onClick={confirmDelete}
              disabled={deleting || !plan}
              className="px-3 py-1.5 text-sm rounded bg-red-900/80 hover:bg-red-800 text-red-100 flex items-center gap-1 disabled:opacity-50"
            >
              {deleting ? <Loader2 className="w-3 h-3 animate-spin" /> : <Trash2 className="w-3 h-3" />}
              apagar definitivamente
            </button>
          </>
        }
      >
        <div className="p-4 space-y-3">
          <div className="flex items-start gap-2 text-amber-400 text-sm">
            <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
            <span>
              Esta ação não pode ser desfeita. Os arquivos abaixo serão removidos
              do disco e a config salva do modelo também.
            </span>
          </div>

          {planLoading && (
            <div className="text-sm text-ink-400 flex items-center gap-1">
              <Loader2 className="w-3 h-3 animate-spin" /> listando arquivos…
            </div>
          )}

          {plan && (
            <>
              <div className="text-xs text-ink-400">
                {plan.count} arquivo(s) — total {(plan.total_bytes / (1024 ** 3)).toFixed(2)} GB
              </div>
              <ul className="bg-ink-950 border border-ink-800 rounded p-2 max-h-64 overflow-auto text-xs mono space-y-0.5">
                {plan.files.map((f) => (
                  <li key={f} className="flex items-center gap-1.5 text-ink-300">
                    <FileText className="w-3 h-3 text-ink-500 shrink-0" />
                    <span className="truncate" title={f}>{f}</span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      </Modal>
    </section>
  );
}
