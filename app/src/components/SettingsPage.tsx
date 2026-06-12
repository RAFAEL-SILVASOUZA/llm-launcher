import { useEffect, useState } from "react";
import { FolderPlus, Trash2, Save, Loader2, FolderOpen, AlertTriangle, Cpu, RotateCcw } from "lucide-react";
import { api } from "../api/client";

type Props = {
  onSaved: () => void;
};

// mtp não aparece aqui: reusa o build do vanilla (MTP nativo no upstream), então
// segue o diretório configurado pro vanilla — sem path próprio pra configurar.
type BackendKey = "turbo" | "vanilla";
const BACKEND_KEYS: BackendKey[] = ["turbo", "vanilla"];

const BACKEND_LABEL: Record<BackendKey, string> = {
  turbo:   "turbo (turboquant)",
  vanilla: "vanilla (upstream + MTP nativo)",
};

export function SettingsPage({ onSaved }: Props) {
  const [paths, setPaths] = useState<string[]>([]);
  const [original, setOriginal] = useState<string[]>([]);

  // Diretórios por backend (turbo/vanilla). mtp segue o vanilla. Valor
  // vazio = usa o default exibido como placeholder.
  const [backendDirs, setBackendDirs] = useState<Record<BackendKey, string>>({
    turbo: "", vanilla: "",
  });
  const [originalBackendDirs, setOriginalBackendDirs] = useState<Record<BackendKey, string>>({
    turbo: "", vanilla: "",
  });
  const [backendDefaults, setBackendDefaults] = useState<Record<BackendKey, string>>({
    turbo: "", vanilla: "",
  });

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

        const cur: Record<BackendKey, string> = { turbo: "", vanilla: "" };
        for (const k of BACKEND_KEYS) cur[k] = s.backend_paths[k] ?? "";
        setBackendDirs(cur);
        setOriginalBackendDirs(cur);
        setBackendDefaults(s.backend_paths_defaults);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const pathsDirty =
    paths.length !== original.length ||
    paths.some((p, i) => p !== original[i]);

  const backendsDirty = BACKEND_KEYS.some(
    (k) => backendDirs[k].trim() !== originalBackendDirs[k].trim(),
  );

  const dirty = pathsDirty || backendsDirty;

  const updatePath = (i: number, v: string) => {
    setPaths((prev) => prev.map((p, idx) => (idx === i ? v : p)));
  };

  const addPath = () => setPaths((prev) => [...prev, ""]);

  const removePath = (i: number) => {
    setPaths((prev) => prev.filter((_, idx) => idx !== i));
  };

  const updateBackendDir = (k: BackendKey, v: string) => {
    setBackendDirs((prev) => ({ ...prev, [k]: v }));
  };

  const resetBackendDir = (k: BackendKey) => {
    setBackendDirs((prev) => ({ ...prev, [k]: "" }));
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const cleanedPaths = paths.map((p) => p.trim()).filter((p) => p.length > 0);
      const cleanedBackends: Partial<Record<BackendKey, string>> = {};
      for (const k of BACKEND_KEYS) {
        const v = backendDirs[k].trim();
        if (v) cleanedBackends[k] = v;
      }
      const saved = await api.saveSettings({
        model_paths:   cleanedPaths,
        backend_paths: cleanedBackends,
      });
      setOriginal(saved.model_paths);
      setPaths(saved.model_paths.length > 0 ? saved.model_paths : [""]);

      const nextBackends: Record<BackendKey, string> = { turbo: "", vanilla: "" };
      for (const k of BACKEND_KEYS) nextBackends[k] = saved.backend_paths[k] ?? "";
      setBackendDirs(nextBackends);
      setOriginalBackendDirs(nextBackends);
      setBackendDefaults(saved.backend_paths_defaults);

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

      <div className="p-4 space-y-6">
        {/* ── Pastas de modelos ────────────────────────────────────────── */}
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

        <div>
          <button
            onClick={addPath}
            className="text-xs px-2.5 py-1.5 rounded bg-ink-800 hover:bg-ink-700 text-ink-200 flex items-center gap-1"
          >
            <FolderPlus className="w-3.5 h-3.5" />
            adicionar caminho
          </button>
        </div>

        {/* ── Pastas dos binários llama.cpp ───────────────────────────── */}
        <div className="pt-2 border-t border-ink-800">
          <div className="flex items-center gap-2 mb-1 mt-3">
            <Cpu className="w-4 h-4 text-ink-400" />
            <h3 className="text-sm font-medium text-ink-100">Diretórios dos binários llama.cpp</h3>
          </div>
          <p className="text-xs text-ink-500">
            Pasta que contém <span className="mono">llama-server.exe</span> e <span className="mono">llama-cli.exe</span> de cada backend.
            Deixe vazio para usar o default.
          </p>
        </div>

        <ul className="space-y-3">
          {BACKEND_KEYS.map((k) => {
            const def = backendDefaults[k] ?? "";
            const cur = backendDirs[k];
            const usingDefault = cur.trim().length === 0;
            return (
              <li key={k} className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-ink-300 w-44 shrink-0">{BACKEND_LABEL[k]}</span>
                  <input
                    type="text"
                    value={cur}
                    onChange={(e) => updateBackendDir(k, e.target.value)}
                    placeholder={def}
                    className="flex-1 bg-ink-950 border border-ink-700 rounded px-3 py-1.5 text-sm mono focus:outline-none focus:border-accent"
                  />
                  <button
                    onClick={() => resetBackendDir(k)}
                    disabled={usingDefault}
                    title="usar default"
                    className="text-xs px-2 py-1.5 rounded bg-ink-800 hover:bg-ink-700 text-ink-300 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1"
                  >
                    <RotateCcw className="w-3 h-3" />
                  </button>
                </div>
                <div className="text-[11px] text-ink-500 pl-46 mono truncate" title={def}>
                  {usingDefault ? <>usando default: {def}</> : <>default: {def}</>}
                </div>
              </li>
            );
          })}
        </ul>

        {/* ── Save / erros ────────────────────────────────────────────── */}
        <div className="flex items-center gap-2 pt-2 border-t border-ink-800">
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
