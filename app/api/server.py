"""FastAPI app — backend HTTP do launcher Electron.

Sobe em porta dinâmica (OS-assigned). Escreve a porta efetiva em duas vias:
  1. stdout (linha "PORT=N"), pro Electron capturar via spawn stdout.
  2. arquivo api_port.tmp ao lado do server.py (fallback se o stdout bufferizar).

Endpoints (prefixo /api):
  GET    /options            — listas de choices (kv, ctx, parallel, etc.)
  GET    /backends           — status dos backends (binários, KV types)
  GET    /system             — VRAM total/livre, RAM total
  GET    /nvidia             — status detalhado da GPU

  GET    /models             — lista de modelos .gguf disponíveis
  POST   /models/meta        — GGUF meta de um path

  GET    /configs            — todas as configs salvas
  POST   /configs            — salva uma config
  DELETE /configs            — remove uma config (body: model, backend)

  POST   /estimate           — estima VRAM/RAM pra uma config
  POST   /build-command      — devolve a string do comando (server|cli)
  POST   /suggest/n-cpu-moe  — sugestão de --n-cpu-moe
  POST   /suggest/n-gpu-layers — sugestão de -ngl

  POST   /launch             — inicia o launch resiliente; devolve {launch_id}
  GET    /launch/{id}/events — SSE de eventos
  POST   /launch/{id}/cancel — cancela o launch
"""
from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import uuid
from pathlib import Path
from queue import Queue, Empty
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Permitir `python api/server.py` direto (sem -m) — adiciona o pai do api/
# no sys.path pra que `from api.core import ...` resolva.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from api.core import (  # noqa: E402
    backends as backends_mod,
    builder as builder_mod,
    config_store,
    constants,
    estimator,
    gguf,
    models_repo,
    nvidia,
    runner,
)

app = FastAPI(title="llm-launcher API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── modelos Pydantic ─────────────────────────────────────────────────────────

class LaunchConfig(BaseModel):
    """Subset persistido em last_config.json — espelha o schema de save_config."""
    model: str
    backend: str = "turbo"
    context_window: int = 65_536
    kv_cache: str = "turbo4"
    flash_attn: bool = True
    gpu_layers: int = 99
    n_cpu_moe: int = 0
    parallel_slots: int = 1
    reasoning_budget: int | None = None
    preserve_thinking: bool = False
    mlock: bool = False
    max_tokens: int = -1
    batch_size: int = 2_048
    ubatch_size: int = 512
    threads_gen: int = constants.CPU_THREADS_GEN
    threads_batch: int = constants.CPU_THREADS_BATCH
    cache_ram: int = 8_192
    ctx_checkpoints: int = 8
    spec_draft_n_max: int = 2
    mmproj: str | None = None
    verbose: bool = False
    mode: str = "server"   # "server" | "cli" — só usado por build/estimate, não persiste


class EstimateRequest(BaseModel):
    model: str
    backend: str = "turbo"
    context_window: int
    kv_cache: str
    parallel_slots: int = 1
    gpu_layers: int = 99
    n_cpu_moe: int = 0
    mmproj: str | None = None
    cache_ram: int = 8_192
    mode: str = "server"


class SuggestRequest(BaseModel):
    model: str
    backend: str = "turbo"
    context_window: int
    kv_cache: str
    parallel_slots: int = 1
    gpu_layers: int = 99
    mmproj: str | None = None
    cache_ram: int = 8_192
    mode: str = "server"


class DeleteConfigRequest(BaseModel):
    model: str
    backend: str | None = None


# ─── routes: opções e status ──────────────────────────────────────────────────

@app.get("/api/options")
def get_options():
    return {
        "kv_cache":         constants.KV_CACHE_OPTIONS,
        "context_window":   constants.CONTEXT_OPTIONS,
        "reasoning_budget": constants.REASONING_BUDGET_OPTIONS,
        "parallel_slots":   constants.PARALLEL_OPTIONS,
        "max_tokens":       constants.MAX_TOKENS_OPTIONS,
        "batch_size":       constants.BATCH_OPTIONS,
        "ubatch_size":      constants.UBATCH_OPTIONS,
        "cache_ram":        constants.CACHE_RAM_OPTIONS,
        "ctx_checkpoints":  constants.CTX_CHECKPOINTS_OPTIONS,
        "spec_draft_n_max": constants.SPEC_DRAFT_N_MAX_OPTIONS,
        "cpu_threads_gen":   constants.CPU_THREADS_GEN,
        "cpu_threads_batch": constants.CPU_THREADS_BATCH,
    }


@app.get("/api/backends")
def get_backends():
    return backends_mod.backend_status()


@app.get("/api/system")
def get_system():
    return {
        "vram_total_mib": nvidia.gpu_total_mib(),
        "vram_free_mib":  nvidia.gpu_free_mib(),
        "ram_total_mib":  nvidia.ram_total_mib(),
    }


@app.get("/api/nvidia")
def get_nvidia():
    return nvidia.nvidia_status()


# ─── routes: modelos ──────────────────────────────────────────────────────────

@app.get("/api/models")
def list_models():
    return [models_repo.describe_model(m) for m in models_repo.collect_models()]


class MetaRequest(BaseModel):
    model: str


@app.post("/api/models/meta")
def model_meta(req: MetaRequest):
    meta = gguf.read_meta(Path(req.model))
    if meta is None:
        raise HTTPException(404, "GGUF metadata não pôde ser lido")
    return meta


# ─── routes: configs ──────────────────────────────────────────────────────────

@app.get("/api/configs")
def list_configs():
    return config_store.read_all_configs()


@app.post("/api/configs")
def save_cfg(cfg: LaunchConfig):
    data = cfg.model_dump(exclude={"mode"})
    config_store.save_config(data)
    return {"ok": True, "config": data}


@app.delete("/api/configs")
def delete_cfg(req: DeleteConfigRequest):
    removed = config_store.delete_config(req.model, req.backend)
    return {"ok": True, "removed": removed}


# ─── routes: estimate / build / suggest ───────────────────────────────────────

@app.post("/api/estimate")
def estimate(req: EstimateRequest):
    mmproj = Path(req.mmproj) if req.mmproj else None
    return estimator.estimate_memory(
        Path(req.model), req.backend, req.context_window, req.parallel_slots,
        req.kv_cache, req.gpu_layers, mmproj, req.cache_ram, req.mode,
        n_cpu_moe=req.n_cpu_moe,
    )


@app.post("/api/build-command")
def build_command(cfg: LaunchConfig):
    try:
        data = cfg.model_dump()
        mode = data.pop("mode", "server")
        cmd = builder_mod.build_command_from_cfg(data, mode=mode)
        return {"command": cmd, "mode": mode}
    except builder_mod.BackendBinaryMissing as e:
        raise HTTPException(400, str(e))


@app.post("/api/suggest/n-cpu-moe")
def suggest_ncmoe(req: SuggestRequest):
    mmproj = Path(req.mmproj) if req.mmproj else None
    return {
        "n_cpu_moe": estimator.suggest_n_cpu_moe(
            Path(req.model), req.backend, req.context_window, req.parallel_slots,
            req.kv_cache, req.gpu_layers, mmproj, req.cache_ram, req.mode,
        )
    }


@app.post("/api/suggest/n-gpu-layers")
def suggest_ngl(req: SuggestRequest):
    mmproj = Path(req.mmproj) if req.mmproj else None
    return {
        "n_gpu_layers": estimator.suggest_n_gpu_layers(
            Path(req.model), req.backend, req.context_window, req.parallel_slots,
            req.kv_cache, mmproj, req.cache_ram, req.mode,
        )
    }


# ─── launches: fila de eventos por id, runner em thread, SSE drain ────────────

class LaunchSession:
    def __init__(self, cfg: dict):
        self.id = uuid.uuid4().hex[:12]
        self.cfg = cfg
        self.handle = runner.LaunchHandle()
        self.queue: Queue[dict] = Queue()
        self.done = False
        self.thread: threading.Thread | None = None

    def on_event(self, ev: dict) -> None:
        self.queue.put(ev)
        if ev.get("type") in ("done", "giveup"):
            self.done = True

    def start(self) -> None:
        def _runner_target():
            try:
                runner.run_server_resiliently(self.cfg, self.on_event, self.handle)
            except Exception as e:
                self.on_event({"type": "giveup", "reason": f"runner crash: {e}"})
            finally:
                self.done = True
                self.queue.put({"type": "_eof"})

        self.thread = threading.Thread(target=_runner_target, daemon=True)
        self.thread.start()


_SESSIONS: dict[str, LaunchSession] = {}
_SESSIONS_LOCK = threading.Lock()


@app.post("/api/launch")
def launch(cfg: LaunchConfig):
    data = cfg.model_dump(exclude={"mode"})
    config_store.save_config(data)
    session = LaunchSession(data)
    with _SESSIONS_LOCK:
        _SESSIONS[session.id] = session
    session.start()
    return {"launch_id": session.id}


@app.get("/api/launch/{launch_id}/events")
async def launch_events(launch_id: str, request: Request):
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(launch_id)
    if session is None:
        raise HTTPException(404, "launch_id desconhecido")

    async def generator():
        while True:
            if await request.is_disconnected():
                return
            try:
                ev = session.queue.get(timeout=0.5)
            except Empty:
                # keepalive comment line (SSE)
                yield ": keepalive\n\n"
                continue
            if ev.get("type") == "_eof":
                return
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.post("/api/launch/{launch_id}/cancel")
def launch_cancel(launch_id: str):
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(launch_id)
    if session is None:
        raise HTTPException(404, "launch_id desconhecido")
    session.handle.cancel()
    return {"ok": True}


# ─── bootstrap: porta livre + arquivo + stdout ────────────────────────────────

def _pick_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    port = _pick_free_port()
    port_file = _HERE.parent / "api_port.tmp"
    try:
        port_file.write_text(str(port), encoding="utf-8")
    except Exception:
        pass
    # Linha lida pelo Electron via stdout do spawn.
    print(f"PORT={port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
