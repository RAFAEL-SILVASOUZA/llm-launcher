import React from "react";
import { Square, XCircle, CheckCircle, AlertTriangle, ArrowDownUp, RotateCw, Play } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, eventsStream } from "../api/client";
import type { ActiveLaunchConfig, LaunchEvent } from "../api/types";
import { isRouterConfig } from "../api/types";
import { Modal } from "./Modal";

type Props = {
  open: boolean;
  launchId: string | null;
  config: ActiveLaunchConfig | null;
  onClose: () => void;
};

// Modal de execução: assina o SSE `/api/launch/{id}/events`, renderiza stdout
// scrollado + faixa de eventos de degrade/restart/giveup destacados. Quando
// o runner termina com hint=lms_fallback_available (mmproj quebrado),
// oferece botão pra tentar via LM Studio.
export function LaunchModal({ open, launchId, config, onClose }: Props) {
  const [lines, setLines] = useState<{ kind: string; text: string; icon?: React.ReactNode }[]>([]);
  const [done, setDone] = useState(false);
  const [currentCfg, setCurrentCfg] = useState<ActiveLaunchConfig | null>(config);
  const [lmsHint, setLmsHint] = useState(false);
  const [lmsLoading, setLmsLoading] = useState(false);
  const [lmsResult, setLmsResult] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setLines([]);
    setDone(false);
    setCurrentCfg(config);
    setLmsHint(false);
    setLmsResult(null);
  }, [launchId, config]);

  useEffect(() => {
    if (!open || !launchId) return;
    let es: EventSource | null = null;
    let cancelled = false;
    (async () => {
      const stream = await eventsStream(launchId);
      if (cancelled) {
        // Cleanup já rodou antes do await resolver — fecha pra não vazar
        // o SSE no servidor (o request fica drenando a queue à toa).
        stream.close();
        return;
      }
      es = stream;
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
  }, [launchId, open]);

  function handleEvent(ev: LaunchEvent) {
    setLines((prev) => {
      const next = [...prev];
      switch (ev.type) {
        case "start":
          next.push({ kind: "info", text: `tentativa #${ev.attempt}`, icon: <Play className="w-3 h-3" /> });
          next.push({ kind: "cmd",  text: ev.cmd });
          setCurrentCfg(ev.config);
          break;
        case "stdout":
          next.push({ kind: "stdout", text: ev.line });
          break;
        case "load_ok":
          next.push({ kind: "ok", text: `servidor carregou (tentativa #${ev.attempt}) — config salva`, icon: <CheckCircle className="w-3 h-3" /> });
          break;
        case "exit":
          next.push({ kind: "info", text: `process exit rc=${ev.rc} (tentativa #${ev.attempt})` });
          break;
        case "failure":
          next.push({ kind: "fail", text: `${ev.category}: ${ev.excerpt}`, icon: <XCircle className="w-3 h-3" /> });
          break;
        case "degrade":
          next.push({ kind: "degrade", text: `degrade: ${ev.description}`, icon: <ArrowDownUp className="w-3 h-3" /> });
          setCurrentCfg(ev.config);
          break;
        case "restart":
          next.push({ kind: "info", text: `reiniciando (#${ev.attempt}) em ${ev.backoff}s…`, icon: <RotateCw className="w-3 h-3" /> });
          break;
        case "manual_restart":
          next.push({ kind: "degrade", text: `restart manual — subindo de novo (#${ev.attempt}), mesma config`, icon: <RotateCw className="w-3 h-3" /> });
          break;
        case "giveup":
          next.push({ kind: "fail", text: `desistiu: ${ev.reason}${ev.failure ? ` (${ev.failure})` : ""}`, icon: <Square className="w-3 h-3" /> });
          setDone(true);
          if (ev.hint === "lms_fallback_available") setLmsHint(true);
          break;
        case "done":
          next.push({ kind: "ok", text: `encerrou normalmente (tentativa #${ev.attempt})`, icon: <CheckCircle className="w-3 h-3" /> });
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
    setDone(true);
    setLines((prev) => [...prev, { kind: "fail", text: "cancelado pelo usuário", icon: <Square className="w-3 h-3" /> }]);
    await api.cancelLaunch(launchId);
    onClose();
  };

  const handleRestart = async () => {
    if (!launchId) return;
    setLines((prev) => [...prev, { kind: "info", text: "restart manual solicitado…", icon: <RotateCw className="w-3 h-3" /> }]);
    try {
      await api.restartLaunch(launchId);
    } catch (e) {
      setLines((prev) => [...prev, { kind: "fail", text: `falha ao reiniciar: ${(e as Error).message}`, icon: <XCircle className="w-3 h-3" /> }]);
    }
  };

  const handleLmsFallback = async () => {
    if (!currentCfg || isRouterConfig(currentCfg)) return;
    setLmsLoading(true);
    setLines((prev) => [...prev, { kind: "info", text: `tentando carregar via LM Studio…`, icon: <Play className="w-3 h-3" /> }]);
    try {
      const r = await api.lmsLoad({
        model: currentCfg.model,
        context_window: currentCfg.context_window,
        parallel_slots: currentCfg.parallel_slots,
      });
      if (r.ok) {
        setLines((prev) => [
          ...prev,
          { kind: "ok", text: `LM Studio carregou: ${r.alias} (key: ${r.key})`, icon: <CheckCircle className="w-3 h-3" /> },
        ]);
        setLmsResult("ok");
      } else {
        setLines((prev) => [
          ...prev,
          { kind: "fail", text: `LM Studio falhou: ${r.error ?? r.load_output ?? "erro desconhecido"}`, icon: <XCircle className="w-3 h-3" /> },
        ]);
        setLmsResult("fail");
      }
    } catch (e) {
      setLines((prev) => [
        ...prev,
        { kind: "fail", text: `erro chamando lms: ${(e as Error).message}`, icon: <XCircle className="w-3 h-3" /> },
      ]);
      setLmsResult("fail");
    } finally {
      setLmsLoading(false);
    }
  };

  return (
    <Modal
      open={open}
      onClose={onClose}
      closable={done}
      width="max-w-5xl"
      title={`Launch ${launchId ? `#${launchId.slice(0, 6)}` : ""}`}
      footer={
        <>
          {!done && (
            <>
              <button
                onClick={handleRestart}
                className="px-3 py-1.5 text-sm rounded bg-amber-900/60 hover:bg-amber-800 text-amber-100 flex items-center gap-1"
                title="Mata e sobe o llama-server de novo com a mesma config (mesma porta). O cliente reconecta."
              >
                <RotateCw className="w-3.5 h-3.5" />Reiniciar
              </button>
              <button
                onClick={handleCancel}
                className="px-3 py-1.5 text-sm rounded bg-red-900/60 hover:bg-red-800 text-red-100 flex items-center gap-1"
              >
                <XCircle className="w-3.5 h-3.5" />Cancelar
              </button>
            </>
          )}
          {lmsHint && !lmsResult && (
            <button
              onClick={handleLmsFallback}
              disabled={lmsLoading}
              className="px-3 py-1.5 text-sm rounded bg-amber-900/60 hover:bg-amber-800 text-amber-100 flex items-center gap-1 disabled:opacity-50"
              title="mmproj falhou no llama-server — tentar carregar via LM Studio"
            >
              <Play className="w-3 h-3" /> tentar via LM Studio
            </button>
          )}
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-sm rounded bg-ink-800 hover:bg-ink-700 flex items-center gap-1"
          >
            {done ? "Fechar" : "Esconder"}
          </button>
        </>
      }
    >
      <div className="p-4 space-y-3">
        {currentCfg && (isRouterConfig(currentCfg) ? (
          <div className="text-xs text-ink-400 mono">
            <span className="text-emerald-400">router</span> ·
            <span className="text-accent-fg"> {currentCfg.backend}</span> ·
            <span className="text-ink-200"> {currentCfg.model_ids.length}</span> modelos:{" "}
            <span className="text-ink-200">{currentCfg.model_ids.join(", ")}</span>
            <span className="text-ink-500"> — o client escolhe pelo campo "model" do request</span>
          </div>
        ) : (
          <div className="text-xs text-ink-400 mono">
            <span className="text-accent-fg">{currentCfg.backend}</span> ·
            ctx <span className="text-ink-200">{currentCfg.context_window.toLocaleString()}</span> ·
            kv <span className="text-ink-200">{currentCfg.kv_cache}</span> ·
            ngl <span className="text-ink-200">{currentCfg.gpu_layers}</span> ·
            np <span className="text-ink-200">{currentCfg.parallel_slots}</span>
            {currentCfg.n_cpu_moe > 0 && <> · ncmoe <span className="text-ink-200">{currentCfg.n_cpu_moe}</span></>}
          </div>
        ))}
        {lines.some((l) => l.text === "(reanexado)") && (
          <div className="bg-amber-950/30 border border-amber-800/60 text-amber-200 text-xs rounded p-2 flex items-start gap-2">
            <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
            <span>
              Este servidor foi <b>reanexado</b> de uma sessão anterior do app —
              os logs ao vivo só começam num próximo launch. O botão de Cancelar
              continua funcionando.
            </span>
          </div>
        )}

        <div
          ref={scrollRef}
          className="bg-ink-950 border border-ink-800 rounded mono text-xs h-[60vh] overflow-y-auto p-3 leading-relaxed"
        >
          {lines.length === 0 && (
            <div className="text-ink-500 italic flex items-center gap-1.5">
              <Play className="w-3 h-3" />aguardando eventos…
            </div>
          )}
          {lines.map((l, i) => (
            <div
              key={i}
              className={`flex items-start gap-1.5 ${
                l.kind === "stdout"  ? "text-ink-300" :
                l.kind === "ok"      ? "text-emerald-400" :
                l.kind === "fail"    ? "text-red-400" :
                l.kind === "degrade" ? "text-amber-400" :
                l.kind === "cmd"     ? "text-purple-300 break-all" :
                                       "text-ink-400"
              }`}
            >
              {l.icon && <span className="mt-0.5 shrink-0">{l.icon}</span>}
              <span>{l.text}</span>
            </div>
          ))}
        </div>
      </div>
    </Modal>
  );
}
