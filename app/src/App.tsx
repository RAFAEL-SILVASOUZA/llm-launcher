import { useCallback, useEffect, useState } from "react";
import { Settings, HardDrive, Download, Cpu, Loader2, SlidersHorizontal, Plug } from "lucide-react";
import { api, downloadEventsStream } from "./api/client";
import type {
  ActiveLaunchConfig, BackendStatus, Estimate, HfDownloadEvent, HfDownloadPlan,
  LaunchConfig, ModelInfo, Options, SystemInfo,
} from "./api/types";

export type DownloadProgress = { dl: number; total: number; speed: number; done?: boolean };
import { ConfigEditor } from "./components/ConfigEditor";
import { ConfigGrid } from "./components/ConfigGrid";
import { DownloadPage } from "./components/DownloadPage";
import { Header } from "./components/Header";
import { LaunchModal } from "./components/LaunchModal";
import { MCPPage } from "./components/MCPPage";
import { ModelsPage } from "./components/ModelsPage";
import { NvidiaPage } from "./components/NvidiaPage";
import { SettingsPage } from "./components/SettingsPage";
import { Tabs } from "./components/Tabs";

type TabId = "configs" | "models" | "download" | "mcp" | "nvidia" | "settings";

export default function App() {
  const [system, setSystem]       = useState<SystemInfo | null>(null);
  const [backends, setBackends]   = useState<BackendStatus[]>([]);
  const [options, setOptions]     = useState<Options | null>(null);
  const [models, setModels]       = useState<ModelInfo[]>([]);
  const [configs, setConfigs]     = useState<LaunchConfig[]>([]);
  const [estimates, setEstimates] = useState<Record<string, Estimate>>({});
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState<string | null>(null);
  const [pythonLogPath, setPythonLogPath] = useState<string | null>(null);

  const [tab, setTab] = useState<TabId>("configs");

  const [editorOpen, setEditorOpen] = useState(false);
  const [editorInitial, setEditorInitial] = useState<Partial<LaunchConfig> | null>(null);

  const [launchId, setLaunchId] = useState<string | null>(null);
  const [launchCfg, setLaunchCfg] = useState<ActiveLaunchConfig | null>(null);
  const [launchModalOpen, setLaunchModalOpen] = useState(false);

  // ─── Download state: vive aqui (não na DownloadPage) pra sobreviver à
  // troca de aba — sem isso, navegar pra outra aba unmounta a página e
  // mata o EventSource, perdendo o progresso (download segue no backend).
  const [downloadId,     setDownloadId]     = useState<string | null>(null);
  const [downloadPlan,   setDownloadPlan]   = useState<HfDownloadPlan | null>(null);
  const [downloadEvents, setDownloadEvents] = useState<HfDownloadEvent[]>([]);
  const [downloadProgress, setDownloadProgress] = useState<Record<string, DownloadProgress>>({});
  const [downloadDone,   setDownloadDone]   = useState(false);
  const [downloadError,  setDownloadError]  = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setError(null);
      const [sys, bks, opts, mdls, cfgs, active] = await Promise.all([
        api.system(),
        api.backends(),
        api.options(),
        api.models(),
        api.listConfigs(),
        api.listLaunches(),
      ]);
      setSystem(sys);
      setBackends(bks);
      setOptions(opts);
      setModels(mdls);
      setConfigs(cfgs);

      // Reconcilia o estado de launch ativo com a verdade do servidor.
      // Cobre principalmente o caso: usuário fechou a janela com servidor
      // rodando, Python morreu, llama-server sobreviveu como órfão; ao
      // reabrir, o startup hook do server.py re-anexa o PID e expõe aqui.
      // Sem isto a UI não tinha botão de Stop pro órfão.
      setLaunchId((curId) => {
        if (active.length === 0) {
          if (curId) setLaunchCfg(null);
          return null;
        }
        const mine = curId && active.find((a) => a.launch_id === curId);
        if (mine) return curId;
        setLaunchCfg(active[0].config);
        return active[0].launch_id;
      });
      // Estimativa em batch pra alimentar a coluna de status do grid.
      // Roda em segundo plano — não bloqueia o render inicial.
      if (cfgs.length > 0) {
        api.estimateMany(cfgs.map((c) => ({
          model: c.model, backend: c.backend,
          context_window: c.context_window, kv_cache: c.kv_cache,
          parallel_slots: c.parallel_slots, gpu_layers: c.gpu_layers,
          n_cpu_moe: c.n_cpu_moe, mmproj: c.mmproj,
          cache_ram: c.cache_ram, mode: "server",
        }))).then((results) => {
          const map: Record<string, Estimate> = {};
          // Resposta vem na MESMA ordem dos items enviados — chaveia pelo id
          // da config (mesma chave que o grid usa via cfgKey), não por
          // model|backend, que colide com configs duplicadas e não bate com
          // o lookup do grid quando a config tem id.
          results.forEach((r, i) => {
            if (r.ok && r.estimate) {
              const c = cfgs[i];
              map[c.id ?? `${c.model}|${c.backend}`] = r.estimate;
            }
          });
          setEstimates(map);
        }).catch(() => { /* silencioso — só apaga as bolinhas */ });
      } else {
        setEstimates({});
      }
    } catch (err) {
      setError((err as Error).message);
      // Em prod (Electron empacotado), o backend Python pode não subir —
      // path do log ajuda o usuário a diagnosticar.
      try {
        const p = await window.electronAPI?.getPythonLogPath?.();
        if (p) setPythonLogPath(p);
      } catch { /* ignore */ }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // Poll periódico do system info pra barras se atualizarem enquanto edita.
  // 2 s casa com o TTL do cache de nvidia-smi no backend — sem polling extra
  // desperdiçado e sem ficar parado.
  useEffect(() => {
    const id = setInterval(() => api.system().then(setSystem).catch(() => {}), 2_000);
    return () => clearInterval(id);
  }, []);

  // SSE de download mora aqui (App) pra não morrer ao trocar de aba.
  useEffect(() => {
    if (!downloadId) return;
    let cancelled = false;
    let es: EventSource | null = null;
    (async () => {
      es = await downloadEventsStream(downloadId);
      es.onmessage = (msg) => {
        if (cancelled) return;
        try {
          const ev = JSON.parse(msg.data) as HfDownloadEvent;
          // Progress NÃO entra no array de eventos: chegam ~2.5/s durante o
          // download inteiro, o array crescia sem limite e cada tick
          // re-renderizava o App todo — a UI "travava" enquanto baixava.
          // A barra de progresso usa só o map downloadProgress.
          if (ev.type !== "progress") {
            setDownloadEvents((prev) => [...prev, ev]);
          }
          if (ev.type === "progress") {
            setDownloadProgress((p) => ({
              ...p,
              [ev.rel]: { dl: ev.downloaded, total: ev.total, speed: ev.speed },
            }));
          } else if (ev.type === "file_done") {
            setDownloadProgress((p) => ({
              ...p,
              [ev.rel]: { ...(p[ev.rel] ?? { dl: 0, total: 0, speed: 0 }), done: true },
            }));
          } else if (ev.type === "done") {
            setDownloadDone(true);
            es?.close();
            refresh();  // recarrega lista de modelos
          } else if (ev.type === "cancelled") {
            setDownloadDone(true);
            es?.close();
          } else if (ev.type === "error") {
            setDownloadError(ev.message);
          }
        } catch { /* ignore */ }
      };
      es.onerror = () => {
        if (cancelled) return;
        // Se já está done, é fim natural do stream; caso contrário, sinaliza.
        setDownloadDone(true);
        es?.close();
      };
    })();
    return () => { cancelled = true; es?.close(); };
  }, [downloadId, refresh]);

  const handleEdit = (cfg: LaunchConfig) => {
    // Mantém o id no initial → o save atualiza esta mesma entrada.
    setEditorInitial(cfg);
    setEditorOpen(true);
  };

  const handleNew = () => {
    setEditorInitial(null);
    setEditorOpen(true);
  };

  // Duplicar = abrir editor com uma cópia SEM id → o save gera um id novo e
  // cria outra entrada, deixando a original intacta.
  const handleDuplicate = (cfg: LaunchConfig) => {
    const { id: _id, ...rest } = cfg;
    setEditorInitial(rest);
    setEditorOpen(true);
  };

  const handleDeleteConfig = async (cfg: LaunchConfig) => {
    if (!cfg.id) return;
    const ok = window.confirm(
      `Remover a config de "${cfg.model}" (${cfg.backend})?\nNão dá pra desfazer.`,
    );
    if (!ok) return;
    await api.deleteConfig(cfg.id);
    await refresh();
  };

  // Identidade é o `id`: editar (mesmo trocando model/backend) atualiza a mesma
  // entrada, e salvar sem id cria uma nova — sem nunca sobrescrever outra config.
  const handleSave = async (cfg: LaunchConfig) => {
    await api.saveConfig(cfg);
    await refresh();
  };

  const handleLaunch = async (cfg: LaunchConfig) => {
    // Salva primeiro pra obter o id atribuído pelo backend, e lança ESSA config
    // (com id) — assim o launch não cria uma 2ª entrada e o auto-degrade
    // atualiza a config certa.
    try {
      const { config } = await api.saveConfig(cfg);
      const r = await api.launch(config);
      setLaunchCfg(config);
      setLaunchId(r.launch_id);
      setLaunchModalOpen(true);
      setEditorOpen(false);
    } catch (e) {
      // Ex.: 409 — já existe um launch ativo (um por vez).
      setError((e as Error).message);
    }
    await refresh();
  };

  // Launch múltiplo: modo router do llama-server — N configs sobem juntas,
  // cada uma num processo filho; o client escolhe pelo campo "model".
  const handleLaunchRouter = async (ids: string[]) => {
    try {
      const r = await api.launchRouter(ids);
      setLaunchCfg(r.config);
      setLaunchId(r.launch_id);
      setLaunchModalOpen(true);
    } catch (e) {
      setError((e as Error).message);
    }
    await refresh();
  };

  const handleStopLaunch = async () => {
    if (!launchId) return;
    await api.cancelLaunch(launchId);
    setLaunchId(null);
    setLaunchCfg(null);
    setLaunchModalOpen(false);
    await refresh();
  };

  // Restart 'soft': mantém a sessão (mesmo launchId/porta), só pede ao backend
  // pra matar e ressubir o llama-server com a mesma config. O cliente que
  // consome a porta fixa reconecta sozinho. Não fecha o modal nem limpa estado.
  const handleRestartLaunch = async () => {
    if (!launchId) return;
    try {
      await api.restartLaunch(launchId);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const handleOpenLaunchLogs = () => {
    if (launchId) setLaunchModalOpen(true);
  };

  const handleStartDownload = async (req: {
    repo_id: string; rel_paths: string[]; subdir?: string | null; base_dir: string;
  }) => {
    setDownloadEvents([]);
    setDownloadProgress({});
    setDownloadDone(false);
    setDownloadError(null);
    setDownloadPlan(null);
    setDownloadId(null);
    try {
      const r = await api.hfDownload(req);
      setDownloadPlan(r.plan);
      setDownloadId(r.download_id);
    } catch (e) {
      setDownloadError((e as Error).message);
      setDownloadDone(true);
    }
  };

  const handleCancelDownload = async () => {
    if (!downloadId) return;
    try { await api.cancelDownload(downloadId); } catch { /* ignore */ }
  };

  const handleResetDownload = () => {
    setDownloadId(null);
    setDownloadPlan(null);
    setDownloadEvents([]);
    setDownloadProgress({});
    setDownloadDone(false);
    setDownloadError(null);
  };

  // Spinner na aba Download quando há download em curso — sinaliza que dá
  // pra ficar em outra aba sem perder o progresso.
  const downloadActive = downloadId !== null && !downloadDone;
  const tabs = [
    { id: "configs" as TabId,  label: "Configs",  icon: <Settings className="w-4 h-4" />, badge: configs.length },
    { id: "models"  as TabId,  label: "Models",   icon: <HardDrive className="w-4 h-4" />, badge: models.length },
    {
      id: "download" as TabId,
      label: "Download",
      icon: downloadActive
        ? <Loader2 className="w-4 h-4 animate-spin text-accent-fg" />
        : <Download className="w-4 h-4" />,
    },
    { id: "mcp"     as TabId,  label: "MCP",      icon: <Plug className="w-4 h-4" /> },
    { id: "nvidia"  as TabId,  label: "NVIDIA",   icon: <Cpu className="w-4 h-4" /> },
    { id: "settings" as TabId, label: "Settings", icon: <SlidersHorizontal className="w-4 h-4" /> },
  ];

  return (
    <div className="min-h-screen bg-ink-950 text-ink-100">
      <Header system={system} backends={backends} onRefresh={refresh} />
      <Tabs tabs={tabs} active={tab} onChange={setTab} />

      <main className="px-6 py-5">
        {loading && (
          <div className="text-ink-400 text-sm">conectando ao backend Python…</div>
        )}
        {error && (
          <div className="bg-red-950/60 border border-red-800 text-red-200 rounded p-3 text-sm mono">
            ❌ {error}
            <div className="mt-1 text-red-300/70 text-xs">
              Verifique se Python + FastAPI/uvicorn estão instalados e no PATH.
            </div>
            {pythonLogPath && (
              <div className="mt-1 text-red-300/70 text-xs">
                Log do backend: <span className="text-red-200">{pythonLogPath}</span>
              </div>
            )}
          </div>
        )}

        {!loading && !error && (
          <>
            {tab === "configs" && (
              <ConfigGrid
                configs={configs}
                models={models}
                estimates={estimates}
                onEdit={handleEdit}
                onDuplicate={handleDuplicate}
                onDelete={handleDeleteConfig}
                onLaunch={handleLaunch}
                onLaunchRouter={handleLaunchRouter}
                onNew={handleNew}
                activeLaunchCfg={launchCfg}
                launchActive={launchId !== null}
                onStopLaunch={handleStopLaunch}
                onRestartLaunch={handleRestartLaunch}
                onOpenLaunchLogs={handleOpenLaunchLogs}
              />
            )}
            {tab === "models"  && <ModelsPage  models={models} onAfterDelete={refresh} />}
            {tab === "download" && (
              <DownloadPage
                downloadId={downloadId}
                plan={downloadPlan}
                events={downloadEvents}
                progress={downloadProgress}
                done={downloadDone}
                error={downloadError}
                models={models}
                onStart={handleStartDownload}
                onCancel={handleCancelDownload}
                onReset={handleResetDownload}
              />
            )}
            {tab === "mcp"     && <MCPPage />}
            {tab === "nvidia"  && <NvidiaPage />}
            {tab === "settings" && <SettingsPage onSaved={refresh} />}
          </>
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
        open={launchModalOpen}
        launchId={launchId}
        config={launchCfg}
        onClose={() => setLaunchModalOpen(false)}
      />
    </div>
  );
}
