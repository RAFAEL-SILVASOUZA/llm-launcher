import { useEffect, useRef, useState } from "react";
import { api, eventsStream } from "../api/client";
import type { LaunchConfig, LaunchEvent } from "../api/types";
import { Modal } from "./Modal";

type Props = {
  launchId: string | null;
  config: LaunchConfig | null;
  onClose: () => void;
};

// Modal de execução: assina o SSE `/api/launch/{id}/events`, renderiza stdout
// scrollado + faixa de eventos de degrade/restart/giveup destacados.
export function LaunchModal({ launchId, config, onClose }: Props) {
  const [lines, setLines] = useState<{ kind: string; text: string }[]>([]);
  const [done, setDone] = useState(false);
  const [currentCfg, setCurrentCfg] = useState<LaunchConfig | null>(config);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setLines([]);
    setDone(false);
    setCurrentCfg(config);
  }, [launchId, config]);

  useEffect(() => {
    if (!launchId) return;
    let es: EventSource | null = null;
    let cancelled = false;
    (async () => {
      es = await eventsStream(launchId);
      es.onmessage = (e) => {
        if (cancelled) return;
        try {
          const ev = JSON.parse(e.data) as LaunchEvent;
          handleEvent(ev);
        } catch {
          /* ignore */
        }
      };
      es.onerror = () => {
        if (cancelled) return;
        setDone(true);
      };
    })();
    return () => {
      cancelled = true;
      es?.close();
    };
  }, [launchId]);

  function handleEvent(ev: LaunchEvent) {
    setLines((prev) => {
      const next = [...prev];
      switch (ev.type) {
        case "start":
          next.push({ kind: "info", text: `▶ tentativa #${ev.attempt}` });
          next.push({ kind: "cmd",  text: ev.cmd });
          setCurrentCfg(ev.config);
          break;
        case "stdout":
          next.push({ kind: "stdout", text: ev.line });
          break;
        case "load_ok":
          next.push({ kind: "ok", text: `✓ servidor carregou (tentativa #${ev.attempt}) — config salva` });
          break;
        case "exit":
          next.push({ kind: "info", text: `process exit rc=${ev.rc} (tentativa #${ev.attempt})` });
          break;
        case "failure":
          next.push({ kind: "fail", text: `✗ ${ev.category}: ${ev.excerpt}` });
          break;
        case "degrade":
          next.push({ kind: "degrade", text: `⤓ degrade: ${ev.description}` });
          setCurrentCfg(ev.config);
          break;
        case "restart":
          next.push({ kind: "info", text: `↻ reiniciando (#${ev.attempt}) em ${ev.backoff}s…` });
          break;
        case "giveup":
          next.push({ kind: "fail", text: `🛑 desistiu: ${ev.reason}${ev.failure ? ` (${ev.failure})` : ""}` });
          setDone(true);
          break;
        case "done":
          next.push({ kind: "ok", text: `✅ encerrou normalmente (tentativa #${ev.attempt})` });
          setDone(true);
          break;
      }
      return next;
    });
    // autoscroll
    setTimeout(() => {
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
    }, 0);
  }

  const handleCancel = async () => {
    if (!launchId) return;
    await api.cancelLaunch(launchId);
  };

  return (
    <Modal
      open={!!launchId}
      onClose={onClose}
      width="max-w-5xl"
      title={`Launch ${launchId ? `#${launchId.slice(0, 6)}` : ""}`}
      footer={
        <>
          {!done && (
            <button
              onClick={handleCancel}
              className="px-3 py-1.5 text-sm rounded bg-red-900/60 hover:bg-red-800 text-red-100"
            >
              Cancelar
            </button>
          )}
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-sm rounded bg-ink-800 hover:bg-ink-700"
          >
            {done ? "Fechar" : "Esconder"}
          </button>
        </>
      }
    >
      <div className="p-4 space-y-3">
        {currentCfg && (
          <div className="text-xs text-ink-400 mono">
            <span className="text-accent-fg">{currentCfg.backend}</span> ·
            ctx <span className="text-ink-200">{currentCfg.context_window.toLocaleString()}</span> ·
            kv <span className="text-ink-200">{currentCfg.kv_cache}</span> ·
            ngl <span className="text-ink-200">{currentCfg.gpu_layers}</span> ·
            np <span className="text-ink-200">{currentCfg.parallel_slots}</span>
            {currentCfg.n_cpu_moe > 0 && <> · ncmoe <span className="text-ink-200">{currentCfg.n_cpu_moe}</span></>}
          </div>
        )}

        <div
          ref={scrollRef}
          className="bg-ink-950 border border-ink-800 rounded mono text-xs h-[60vh] overflow-y-auto p-3 leading-relaxed"
        >
          {lines.length === 0 && (
            <div className="text-ink-500 italic">aguardando eventos…</div>
          )}
          {lines.map((l, i) => (
            <div
              key={i}
              className={
                l.kind === "stdout"  ? "text-ink-300" :
                l.kind === "ok"      ? "text-emerald-400" :
                l.kind === "fail"    ? "text-red-400" :
                l.kind === "degrade" ? "text-amber-400" :
                l.kind === "cmd"     ? "text-purple-300 break-all" :
                                       "text-ink-400"
              }
            >
              {l.text}
            </div>
          ))}
        </div>
      </div>
    </Modal>
  );
}
