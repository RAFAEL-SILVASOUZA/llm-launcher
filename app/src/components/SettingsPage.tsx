import { useEffect, useState } from "react";
import { FolderPlus, Trash2, Save, Loader2, FolderOpen, AlertTriangle } from "lucide-react";
import { api } from "../api/client";

type Props = {
  onSaved: () => void;
};

export function SettingsPage({ onSaved }: Props) {
  const [paths, setPaths] = useState<string[]>([]);
  const [original, setOriginal] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedFlash, setSavedFlash] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const s = await api.getSettings();
        const list = s.model_paths.length > 0 ? s.model_paths : [""];
        setPaths(list);
        setOriginal(s.model_paths);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const dirty =
    paths.length !== original.length ||
    paths.some((p, i) => p !== original[i]);

  const updatePath = (i: number, v: string) => {
    setPaths((prev) => prev.map((p, idx) => (idx === i ? v : p)));
  };

  const addPath = () => setPaths((prev) => [...prev, ""]);

  const removePath = (i: number) => {
    setPaths((prev) => prev.filter((_, idx) => idx !== i));
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const cleaned = paths.map((p) => p.trim()).filter((p) => p.length > 0);
      const saved = await api.saveSettings({ model_paths: cleaned });
      setOriginal(saved.model_paths);
      setPaths(saved.model_paths.length > 0 ? saved.model_paths : [""]);
      setSavedFlash(true);
      setTimeout(() => setSavedFlash(false), 1500);
      onSaved();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="text-ink-400 text-sm flex items-center gap-2">
        <Loader2 className="w-4 h-4 animate-spin" />
        carregando configurações…
      </div>
    );
  }

  return (
    <section className="bg-ink-900 border border-ink-800 rounded-lg overflow-hidden max-w-3xl">
      <div className="px-4 py-3 border-b border-ink-800 flex items-center gap-3">
        <h2 className="font-medium text-ink-100">Configurações</h2>
        <div className="flex-1" />
        {savedFlash && (
          <span className="text-xs text-emerald-400">salvo</span>
        )}
      </div>

      <div className="p-4 space-y-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <FolderOpen className="w-4 h-4 text-ink-400" />
            <h3 className="text-sm font-medium text-ink-100">Pastas de modelos</h3>
          </div>
          <p className="text-xs text-ink-500">
            Caminhos varridos para encontrar .gguf. A lista da aba Models recarrega quando você salva.
          </p>
        </div>

        <ul className="space-y-2">
          {paths.map((p, i) => (
            <li key={i} className="flex items-center gap-2">
              <input
                type="text"
                value={p}
                onChange={(e) => updatePath(i, e.target.value)}
                placeholder="C:\caminho\para\modelos"
                className="flex-1 bg-ink-950 border border-ink-700 rounded px-3 py-1.5 text-sm mono focus:outline-none focus:border-accent"
              />
              <button
                onClick={() => removePath(i)}
                title="remover este caminho"
                className="text-xs px-2 py-1.5 rounded bg-ink-800 hover:bg-red-900/60 hover:text-red-200 flex items-center gap-1"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            </li>
          ))}
          {paths.length === 0 && (
            <li className="text-xs text-ink-500 italic">
              Nenhum caminho. Sem caminhos configurados, o default do LM Studio é usado.
            </li>
          )}
        </ul>

        <div className="flex items-center gap-2">
          <button
            onClick={addPath}
            className="text-xs px-2.5 py-1.5 rounded bg-ink-800 hover:bg-ink-700 text-ink-200 flex items-center gap-1"
          >
            <FolderPlus className="w-3.5 h-3.5" />
            adicionar caminho
          </button>
          <div className="flex-1" />
          <button
            onClick={handleSave}
            disabled={saving || !dirty}
            className="text-xs px-3 py-1.5 rounded bg-accent/90 hover:bg-accent text-ink-950 font-medium flex items-center gap-1 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {saving ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5" />}
            salvar
          </button>
        </div>

        {error && (
          <div className="bg-red-950/60 border border-red-800 text-red-200 rounded p-3 text-sm flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>
    </section>
  );
}
