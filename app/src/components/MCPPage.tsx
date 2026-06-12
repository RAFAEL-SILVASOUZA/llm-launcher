import { useCallback, useEffect, useRef, useState } from "react";
import {
  Play, Square, Trash2, Plus, Save, Loader2, AlertTriangle,
  CheckCircle, XCircle, FolderOpen, Terminal, Edit3, Eye,
} from "lucide-react";
import { api, mcpEventsStream } from "../api/client";
import type { McpLogEvent, McpServer } from "../api/types";
import { Modal } from "./Modal";

type EditorState = {
  open: boolean;
  initial: Partial<McpServer> | null;
};

export function MCPPage() {
  const [servers, setServers] = useState<McpServer[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editor, setEditor] = useState<EditorState>({ open: false, initial: null });
  const [logsFor, setLogsFor] = useState<McpServer | null>(null);

  const refresh = useCallback(async () => {
    try {
      setError(null);
      const list = await api.mcpList();
      setServers(list);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // Poll leve pra status (running/auto_stopped) refletir mortes externas.
  useEffect(() => {
    const id = setInterval(() => { refresh(); }, 3_000);
    return () => clearInterval(id);
  }, [refresh]);

  const handleStart = async (s: McpServer) => {
    try { await api.mcpStart(s.id); } catch (e) { alert((e as Error).message); }
    await refresh();
  };

  const handleStop = async (s: McpServer) => {
    try { await api.mcpStop(s.id); } catch (e) { alert((e as Error).message); }
    await refresh();
  };

  const handleDelete = async (s: McpServer) => {
    if (!window.confirm(`Remover servidor MCP "${s.name}"?`)) return;
    try { await api.mcpDelete(s.id); } catch (e) { alert((e as Error).message); }
    await refresh();
  };

  const handleSave = async (payload: { id?: string; name: string; cwd: string; command: string }) => {
    await api.mcpSave(payload);
    setEditor({ open: false, initial: null });
    await refresh();
  };

  if (loading) {
    return (
      <div className="text-ink-400 text-sm flex items-center gap-2">
        <Loader2 className="w-4 h-4 animate-spin" />
        carregando servidores MCP…
      </div>
    );
  }

  return (
    <section className="space-y-4 max-w-5xl">
      <div className="flex items-center gap-3">
        <h2 className="text-base font-medium text-ink-100">Servidores MCP</h2>
        <span className="text-xs text-ink-500">
          Cadastre servidores MCP locais (http stream). Liga/desliga manual; se cair, desliga automaticamente.
        </span>
        <div className="flex-1" />
        <button
          onClick={() => setEditor({ open: true, initial: null })}
          className="text-xs px-3 py-1.5 rounded bg-accent/90 hover:bg-accent text-ink-950 font-medium flex items-center gap-1"
        >
          <Plus className="w-3.5 h-3.5" />
          novo
        </button>
      </div>

      {error && (
        <div className="bg-red-950/60 border border-red-800 text-red-200 rounded p-3 text-sm flex items-start gap-2">
          <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {servers.length === 0 && (
        <div className="text-ink-500 text-sm italic border border-dashed border-ink-800 rounded p-6 text-center">
          Nenhum servidor cadastrado ainda. Clique em <b>novo</b> para adicionar o primeiro.
        </div>
      )}

      <ul className="space-y-2">
        {servers.map((s) => (
          <ServerRow
            key={s.id}
            server={s}
            onStart={() => handleStart(s)}
            onStop={() => handleStop(s)}
            onDelete={() => handleDelete(s)}
            onEdit={() => setEditor({ open: true, initial: s })}
            onLogs={() => setLogsFor(s)}
          />
        ))}
      </ul>

      <ServerEditor
        open={editor.open}
        initial={editor.initial}
        onClose={() => setEditor({ open: false, initial: null })}
        onSave={handleSave}
      />

      <LogsModal
        server={logsFor}
        onClose={() => setLogsFor(null)}
      />
    </section>
  );
}

type RowProps = {
  server: McpServer;
  onStart: () => void;
  onStop: () => void;
  onDelete: () => void;
  onEdit: () => void;
  onLogs: () => void;
};

function ServerRow({ server, onStart, onStop, onDelete, onEdit, onLogs }: RowProps) {
  const { status } = server;
  const running = status.running;
  return (
    <li className="bg-ink-900 border border-ink-800 rounded-lg px-4 py-3 flex items-center gap-3">
      <div className="flex flex-col items-center w-10 shrink-0">
        {running ? (
          <span title="rodando" className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
        ) : status.auto_stopped ? (
          <span title="auto-desligado por erro" className="w-2.5 h-2.5 rounded-full bg-red-500" />
        ) : (
          <span title="parado" className="w-2.5 h-2.5 rounded-full bg-ink-600" />
        )}
        {status.pid && (
          <span className="text-[10px] mono text-ink-500 mt-1">{status.pid}</span>
        )}
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium text-ink-100 truncate">{server.name}</span>
          {status.auto_stopped && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-950/60 text-red-300 border border-red-900">
              auto-stop (rc={status.last_exit_code})
            </span>
          )}
        </div>
        <div className="text-xs text-ink-400 mono truncate" title={server.cwd}>
          <FolderOpen className="w-3 h-3 inline mr-1 text-ink-500" />
          {server.cwd}
        </div>
        <div className="text-xs text-ink-400 mono truncate" title={server.command}>
          <Terminal className="w-3 h-3 inline mr-1 text-ink-500" />
          {server.command}
        </div>
        {status.last_error && !running && (
          <div className="text-[11px] text-red-300 truncate mt-0.5" title={status.last_error}>
            {status.last_error}
          </div>
        )}
      </div>

      <div className="flex items-center gap-1.5 shrink-0">
        {running ? (
          <button
            onClick={onStop}
            className="text-xs px-2.5 py-1.5 rounded bg-red-900/60 hover:bg-red-800 text-red-100 flex items-center gap-1"
          >
            <Square className="w-3.5 h-3.5" />
            desligar
          </button>
        ) : (
          <button
            onClick={onStart}
            className="text-xs px-2.5 py-1.5 rounded bg-emerald-900/60 hover:bg-emerald-800 text-emerald-100 flex items-center gap-1"
          >
            <Play className="w-3.5 h-3.5" />
            ligar
          </button>
        )}
        <button
          onClick={onLogs}
          className="text-xs px-2 py-1.5 rounded bg-ink-800 hover:bg-ink-700 text-ink-200 flex items-center gap-1"
          title="ver logs"
        >
          <Eye className="w-3.5 h-3.5" />
        </button>
        <button
          onClick={onEdit}
          disabled={running}
          className="text-xs px-2 py-1.5 rounded bg-ink-800 hover:bg-ink-700 text-ink-200 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1"
          title={running ? "desligue antes de editar" : "editar"}
        >
          <Edit3 className="w-3.5 h-3.5" />
        </button>
        <button
          onClick={onDelete}
          disabled={running}
          className="text-xs px-2 py-1.5 rounded bg-ink-800 hover:bg-red-900/60 hover:text-red-200 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1"
          title={running ? "desligue antes de remover" : "remover"}
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>
    </li>
  );
}

type EditorProps = {
  open: boolean;
  initial: Partial<McpServer> | null;
  onClose: () => void;
  onSave: (payload: { id?: string; name: string; cwd: string; command: string }) => Promise<void>;
};

function ServerEditor({ open, initial, onClose, onSave }: EditorProps) {
  const [name, setName] = useState("");
  const [cwd, setCwd] = useState("");
  const [command, setCommand] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setName(initial?.name ?? "");
    setCwd(initial?.cwd ?? "");
    setCommand(initial?.command ?? "");
    setError(null);
  }, [open, initial]);

  const isEdit = !!initial?.id;
  const canSave = name.trim().length > 0 && cwd.trim().length > 0 && command.trim().length > 0;

  const handleSave = async () => {
    if (!canSave) return;
    setSaving(true);
    setError(null);
    try {
      await onSave({
        id: initial?.id,
        name: name.trim(),
        cwd: cwd.trim(),
        command: command.trim(),
      });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={isEdit ? `Editar MCP — ${initial?.name}` : "Novo servidor MCP"}
      width="max-w-xl"
      footer={
        <>
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-sm rounded bg-ink-800 hover:bg-ink-700"
          >
            cancelar
          </button>
          <button
            onClick={handleSave}
            disabled={!canSave || saving}
            className="px-3 py-1.5 text-sm rounded bg-accent/90 hover:bg-accent text-ink-950 font-medium flex items-center gap-1 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {saving ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5" />}
            salvar
          </button>
        </>
      }
    >
      <div className="p-5 space-y-4">
        <Field label="Nome">
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="mcp-http-filesystem"
            className="w-full bg-ink-950 border border-ink-700 rounded px-3 py-1.5 text-sm focus:outline-none focus:border-accent"
          />
        </Field>
        <Field label="Diretório (cwd)">
          <input
            type="text"
            value={cwd}
            onChange={(e) => setCwd(e.target.value)}
            placeholder="C:\Projetos\mcp-http-filesystem-basic"
            className="w-full bg-ink-950 border border-ink-700 rounded px-3 py-1.5 text-sm mono focus:outline-none focus:border-accent"
          />
        </Field>
        <Field label="Comando">
          <input
            type="text"
            value={command}
            onChange={(e) => setCommand(e.target.value)}
            placeholder="npm run dev"
            className="w-full bg-ink-950 border border-ink-700 rounded px-3 py-1.5 text-sm mono focus:outline-none focus:border-accent"
          />
          <p className="text-[11px] text-ink-500 mt-1">
            Executado via shell no diretório acima. Stdout/stderr são capturados.
          </p>
        </Field>

        {error && (
          <div className="bg-red-950/60 border border-red-800 text-red-200 rounded p-3 text-sm flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>
    </Modal>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs text-ink-400 mb-1 block">{label}</span>
      {children}
    </label>
  );
}

function LogsModal({ server, onClose }: { server: McpServer | null; onClose: () => void }) {
  const [lines, setLines] = useState<McpLogEvent[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!server) {
      setLines([]);
      return;
    }
    let cancelled = false;
    let es: EventSource | null = null;

    // Se está rodando, pega SSE; senão pega snapshot dos logs.
    (async () => {
      if (server.status.running) {
        try {
          const stream = await mcpEventsStream(server.id);
          if (cancelled) {
            stream.close();
            return;
          }
          es = stream;
          es.onmessage = (e) => {
            if (cancelled) return;
            try {
              const ev = JSON.parse(e.data) as McpLogEvent;
              setLines((prev) => [...prev, ev]);
            } catch { /* ignore */ }
          };
          es.onerror = () => {
            if (cancelled) return;
            es?.close();
          };
        } catch {
          // fallback: snapshot
          const r = await api.mcpLogs(server.id);
          setLines(r.logs as McpLogEvent[]);
        }
      } else {
        const r = await api.mcpLogs(server.id);
        setLines(r.logs as McpLogEvent[]);
      }
    })();

    return () => {
      cancelled = true;
      es?.close();
    };
  }, [server]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [lines]);

  if (!server) return null;
  return (
    <Modal
      open={!!server}
      onClose={onClose}
      title={`Logs — ${server.name}`}
      width="max-w-4xl"
      footer={
        <button
          onClick={onClose}
          className="px-3 py-1.5 text-sm rounded bg-ink-800 hover:bg-ink-700"
        >
          fechar
        </button>
      }
    >
      <div className="p-4">
        <div
          ref={scrollRef}
          className="bg-ink-950 border border-ink-800 rounded mono text-xs h-[60vh] overflow-y-auto p-3 leading-relaxed"
        >
          {lines.length === 0 && (
            <div className="text-ink-500 italic">sem logs no buffer.</div>
          )}
          {lines.map((l, i) => (
            <div
              key={i}
              className={`flex items-start gap-1.5 ${
                l.kind === "stdout" ? "text-ink-300" :
                l.kind === "error"  ? "text-red-400" :
                                      "text-ink-400"
              }`}
            >
              {l.kind === "error" && <XCircle className="w-3 h-3 mt-0.5 shrink-0" />}
              {l.kind === "info" && <CheckCircle className="w-3 h-3 mt-0.5 shrink-0" />}
              <span className="whitespace-pre-wrap break-all">{l.text}</span>
            </div>
          ))}
        </div>
      </div>
    </Modal>
  );
}
