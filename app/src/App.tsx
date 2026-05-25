import { useCallback, useEffect, useState } from "react";
import { api } from "./api/client";
import type {
  BackendStatus, LaunchConfig, ModelInfo, Options, SystemInfo,
} from "./api/types";
import { ConfigEditor } from "./components/ConfigEditor";
import { ConfigGrid } from "./components/ConfigGrid";
import { Header } from "./components/Header";
import { LaunchModal } from "./components/LaunchModal";

export default function App() {
  const [system, setSystem]       = useState<SystemInfo | null>(null);
  const [backends, setBackends]   = useState<BackendStatus[]>([]);
  const [options, setOptions]     = useState<Options | null>(null);
  const [models, setModels]       = useState<ModelInfo[]>([]);
  const [configs, setConfigs]     = useState<LaunchConfig[]>([]);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState<string | null>(null);

  const [editorOpen, setEditorOpen] = useState(false);
  const [editorInitial, setEditorInitial] = useState<Partial<LaunchConfig> | null>(null);

  const [launchId, setLaunchId] = useState<string | null>(null);
  const [launchCfg, setLaunchCfg] = useState<LaunchConfig | null>(null);

  const refresh = useCallback(async () => {
    try {
      setError(null);
      const [sys, bks, opts, mdls, cfgs] = await Promise.all([
        api.system(),
        api.backends(),
        api.options(),
        api.models(),
        api.listConfigs(),
      ]);
      setSystem(sys);
      setBackends(bks);
      setOptions(opts);
      setModels(mdls);
      setConfigs(cfgs);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // Poll periódico do system info pra barras se atualizarem enquanto edita.
  useEffect(() => {
    const id = setInterval(() => api.system().then(setSystem).catch(() => {}), 5_000);
    return () => clearInterval(id);
  }, []);

  const handleEdit = (cfg: LaunchConfig) => {
    setEditorInitial(cfg);
    setEditorOpen(true);
  };

  const handleNew = () => {
    setEditorInitial(null);
    setEditorOpen(true);
  };

  const handleDelete = async (cfg: LaunchConfig) => {
    const ok = window.confirm(
      `Remover a config de "${cfg.model}" (${cfg.backend})?\nNão dá pra desfazer.`,
    );
    if (!ok) return;
    await api.deleteConfig(cfg.model, cfg.backend);
    await refresh();
  };

  const handleSave = async (cfg: LaunchConfig) => {
    await api.saveConfig(cfg);
    await refresh();
  };

  const handleLaunch = async (cfg: LaunchConfig) => {
    await api.saveConfig(cfg);
    const r = await api.launch(cfg);
    setLaunchCfg(cfg);
    setLaunchId(r.launch_id);
    setEditorOpen(false);
    await refresh();
  };

  return (
    <div className="min-h-screen bg-ink-950 text-ink-100">
      <Header system={system} backends={backends} onRefresh={refresh} />

      <main className="px-6 py-5">
        {loading && (
          <div className="text-ink-400 text-sm">conectando ao backend Python…</div>
        )}
        {error && (
          <div className="bg-red-950/60 border border-red-800 text-red-200 rounded p-3 text-sm mono">
            ❌ {error}
            <div className="mt-1 text-red-300/70 text-xs">
              Verifique se o servidor Python (api/server.py) está rodando.
            </div>
          </div>
        )}

        {!loading && !error && (
          <ConfigGrid
            configs={configs}
            models={models}
            onEdit={handleEdit}
            onDelete={handleDelete}
            onLaunch={handleLaunch}
            onNew={handleNew}
          />
        )}
      </main>

      <ConfigEditor
        open={editorOpen}
        initial={editorInitial}
        models={models}
        backends={backends}
        options={options}
        onClose={() => setEditorOpen(false)}
        onSave={handleSave}
        onLaunch={handleLaunch}
      />

      <LaunchModal
        launchId={launchId}
        config={launchCfg}
        onClose={() => setLaunchId(null)}
      />
    </div>
  );
}
