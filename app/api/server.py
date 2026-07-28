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
                               (409 se já houver launch ativo — um por vez)
  POST   /launch-router      — sobe N configs de uma vez (modo router do
                               llama-server, preset INI); body {ids: [...]}
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
from contextlib import asynccontextmanager
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
    defaults,
    estimator,
    gguf,
    hf,
    lms,
    mcp_runner,
    mcp_store,
    models_repo,
    nvidia,
    runner,
    running,
    sampling,
    settings_store,
    updates,
)

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Re-anexa llama-server órfãos da sessão anterior antes de aceitar requests.
    _attach_orphan_servers()
    yield


app = FastAPI(title="llm-launcher API", version="0.1.0", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── modelos Pydantic ─────────────────────────────────────────────────────────

class LaunchConfig(BaseModel):
    """Subset persistido em last_config.json — espelha o schema de save_config."""
    id: str | None = None   # identidade estável; backend gera no 1º save se ausente
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
    # "llama.cpp decide": comando mínimo (-m/--mmproj/--alias + HTTP). As demais
    # chaves continuam persistidas, só não viram flag enquanto isto estiver on.
    llama_auto: bool = False
    # Samplers. None = resolvido por modelo (generation_config do autor > preset do
    # chat template). sampler_source == "manual" trava os valores abaixo: é o usuário
    # mandando, e nada os sobrescreve.
    temp: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    sampler_source: str | None = None
    # Multi-GPU (opcionais; None = default do llama.cpp = layer split entre todas)
    split_mode: str | None = None      # "none" | "layer" | "row"
    tensor_split: str | None = None    # ex.: "1,1" (meio a meio entre 2 GPUs)
    main_gpu: int | None = None        # índice base-0 da GPU principal
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
    id: str


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
        "sampler_presets":   sampling.PRESETS,
    }


@app.get("/api/backends")
def get_backends():
    return backends_mod.backend_status()


@app.get("/api/system")
def get_system():
    total, avail = nvidia._ram_mib_pair()
    return {
        "vram_total_mib": nvidia.gpu_total_mib(),
        "vram_free_mib":  nvidia.gpu_free_mib(),
        "ram_total_mib":  total,
        "ram_avail_mib":  avail,
        "gpu_count":      nvidia.gpu_count(),
    }


@app.get("/api/nvidia")
def get_nvidia():
    return nvidia.nvidia_status()


# ─── routes: modelos ──────────────────────────────────────────────────────────

@app.get("/api/models")
def list_models():
    roots = settings_store.get_scan_paths()
    return [models_repo.describe_model(m, roots) for m in models_repo.collect_models(roots)]


@app.get("/api/models/updates")
def models_updates(deep: bool = False):
    """Confere cada modelo contra o HF. `deep=true` autoriza hashear os ainda
    não verificados (custo alto — reservado a botão manual). Sem deep, roda com
    tamanho + oid registrado/cacheado, barato o bastante pra chamada de startup.

    Um repo é consultado uma vez só (remote_cache), e falha de rede num modelo
    não derruba os outros — cada um vira status 'unknown' isolado.
    """
    roots = settings_store.get_scan_paths()
    remote_cache: dict[str, list[dict]] = {}
    results = [
        updates.check_model(m, roots, deep=deep, remote_cache=remote_cache)
        for m in models_repo.collect_models(roots)
    ]
    return {"results": results}


class CheckUpdateRequest(BaseModel):
    model: str
    deep: bool = True


@app.post("/api/models/check-update")
def model_check_update(req: CheckUpdateRequest):
    """Checagem de um modelo só — default deep=true (hasheia pra confirmar)."""
    roots = settings_store.get_scan_paths()
    return updates.check_model(Path(req.model), roots, deep=req.deep)


# ─── routes: app settings (paths das pastas de modelos, etc.) ─────────────────

class SettingsPayload(BaseModel):
    model_paths:   list[str]      = Field(default_factory=list)
    backend_paths: dict[str, str] = Field(default_factory=dict)


@app.get("/api/settings")
def get_settings():
    return settings_store.read_settings()


@app.post("/api/settings")
def update_settings(payload: SettingsPayload):
    saved = settings_store.save_settings(payload.model_dump())
    # Os caches de --help / kv_types do backends_mod são por (backend, mode);
    # se o usuário trocou o diretório do binário, os caches ficam stale.
    backends_mod.invalidate_caches()
    return saved


class MetaRequest(BaseModel):
    model: str


@app.post("/api/models/meta")
def model_meta(req: MetaRequest):
    meta = gguf.read_meta(Path(req.model))
    if meta is None:
        raise HTTPException(404, "GGUF metadata não pôde ser lido")
    return meta


class ContextOptionsRequest(BaseModel):
    model: str


@app.post("/api/models/context-options")
def model_context_options(req: ContextOptionsRequest):
    meta = gguf.read_meta(Path(req.model))
    if meta is None:
        raise HTTPException(404, "GGUF metadata não pôde ser lido")
    n_ctx_train = meta.get("n_ctx_train")
    if n_ctx_train is None:
        return {"options": constants.CONTEXT_OPTIONS}
    options = [{"value": n_ctx_train, "label": f"{n_ctx_train:,} tokens [modelo]"}]
    options.extend(constants.CONTEXT_OPTIONS)
    return {"options": options}


class DefaultsRequest(BaseModel):
    model: str
    backend: str = "vanilla"
    mode: str = "server"


@app.post("/api/models/defaults")
def model_defaults(req: DefaultsRequest):
    """Config recomendada pro modelo: o que o botão "defaults do modelo" preenche.

    Deriva tudo que dá pra derivar (GGUF + VRAM + samplers do autor) e devolve as
    notas explicando cada escolha — palpite sem justificativa não é configurável.
    """
    try:
        return defaults.suggest_config(Path(req.model), req.backend, req.mode)
    except Exception as e:
        raise HTTPException(400, f"não consegui derivar defaults: {e}")


class SamplingRequest(BaseModel):
    model: str
    config: dict | None = None


@app.post("/api/models/sampling")
def model_sampling(req: SamplingRequest):
    """Samplers que este modelo receberia agora, e de onde vieram.

    `source` é o ponto todo: "generation_config" = números do autor; "template" =
    preset derivado do chat template; "default" = chute; "manual" = o usuário fixou.
    A UI mostra isso pra um preset genérico nunca passar por recomendação oficial.
    """
    return sampling.resolve(Path(req.model), req.config)


# ─── routes: configs ──────────────────────────────────────────────────────────

@app.get("/api/configs")
def list_configs():
    return config_store.read_all_configs()


@app.post("/api/configs")
def save_cfg(cfg: LaunchConfig):
    data = cfg.model_dump(exclude={"mode"})
    saved = config_store.save_config(data)   # atribui id se ausente (mutação)
    return {"ok": True, "config": saved}


@app.delete("/api/configs")
def delete_cfg(req: DeleteConfigRequest):
    removed = config_store.delete_config_by_id(req.id)
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


class EstimateManyRequest(BaseModel):
    items: list[EstimateRequest]


@app.post("/api/estimate-many")
def estimate_many(req: EstimateManyRequest):
    """Estimativa em lote — alimenta a coluna de status do grid sem N round-trips."""
    out: list[dict] = []
    for it in req.items:
        try:
            mmproj = Path(it.mmproj) if it.mmproj else None
            est = estimator.estimate_memory(
                Path(it.model), it.backend, it.context_window, it.parallel_slots,
                it.kv_cache, it.gpu_layers, mmproj, it.cache_ram, it.mode,
                n_cpu_moe=it.n_cpu_moe,
            )
            out.append({"ok": True, "estimate": est, "model": it.model, "backend": it.backend})
        except Exception as e:
            out.append({"ok": False, "error": str(e), "model": it.model, "backend": it.backend})
    return out


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
    def __init__(self, cfg: dict, *, attached_pid: int | None = None, launch_id: str | None = None,
                 member_cfgs: list[dict] | None = None):
        self.id = launch_id or uuid.uuid4().hex[:12]
        self.cfg = cfg
        # Modo router: cfg é sintética ({"router": True, ...}) e member_cfgs
        # tem as configs reais que viram seções do preset INI.
        self.member_cfgs = member_cfgs
        self.handle = runner.LaunchHandle(attached_pid=attached_pid)
        self.queue: Queue[dict] = Queue()
        self.done = False
        self.thread: threading.Thread | None = None
        self.attached = attached_pid is not None
        if self.attached:
            # Sessão sintética: alimenta a UI com 2 eventos pra parecer um
            # launch normal já carregado. Stop ainda funciona porque o
            # LaunchHandle conhece o PID.
            self.queue.put({"type": "start", "attempt": 1, "cmd": "(reanexado)", "config": cfg})
            self.queue.put({"type": "load_ok", "attempt": 1})

    def on_event(self, ev: dict) -> None:
        self.queue.put(ev)
        if ev.get("type") in ("done", "giveup"):
            self.done = True

    def _register_pid(self, pid: int) -> None:
        running.register(self.id, pid, self.cfg)

    def start(self) -> None:
        if self.attached:
            # Nada pra rodar — só aguarda cancel/morte do PID.
            return

        def _runner_target():
            try:
                if self.cfg.get("router") and self.member_cfgs:
                    runner.run_router_server(
                        self.member_cfgs, self.cfg, self.on_event, self.handle,
                        on_proc_pid=self._register_pid,
                    )
                else:
                    runner.run_server_resiliently(
                        self.cfg, self.on_event, self.handle,
                        on_proc_pid=self._register_pid,
                    )
            except Exception as e:
                self.on_event({"type": "giveup", "reason": f"runner crash: {e}"})
            finally:
                self.done = True
                running.deregister(self.id)
                self.queue.put({"type": "_eof"})

        self.thread = threading.Thread(target=_runner_target, daemon=True)
        self.thread.start()


_SESSIONS: dict[str, LaunchSession] = {}
_SESSIONS_LOCK = threading.Lock()


def _active_launch_id() -> str | None:
    """id do launch vivo (single ou router), se houver. Um por vez: a porta
    1234 é única e dois llama-server disputando GPU só geram OOM confuso."""
    with _SESSIONS_LOCK:
        for s in _SESSIONS.values():
            if not s.done:
                return s.id
    return None


@app.post("/api/launch")
def launch(cfg: LaunchConfig):
    active = _active_launch_id()
    if active is not None:
        raise HTTPException(
            409,
            "Já existe um launch ativo — pare o servidor atual antes de subir "
            "outro modelo (ou use a seleção múltipla pra rodar vários juntos).",
        )
    data = cfg.model_dump(exclude={"mode"})
    config_store.save_config(data)
    session = LaunchSession(data)
    with _SESSIONS_LOCK:
        _SESSIONS[session.id] = session
    session.start()
    return {"launch_id": session.id}


class RouterLaunchRequest(BaseModel):
    ids: list[str]


@app.post("/api/launch-router")
def launch_router(req: RouterLaunchRequest):
    """Sobe VÁRIOS modelos de uma vez no modo router do llama-server: um
    preset INI com uma seção por config selecionada, todas com
    load-on-startup. O client aponta pra porta 1234 e escolhe o modelo pelo
    campo "model" do request."""
    active = _active_launch_id()
    if active is not None:
        raise HTTPException(
            409, "Já existe um launch ativo — pare-o antes de subir o modo router.")
    if len(req.ids) < 2:
        raise HTTPException(400, "Selecione pelo menos 2 configs pro modo router.")

    by_id = {c.get("id"): c for c in config_store.read_all_configs()}
    missing = [i for i in req.ids if i not in by_id]
    if missing:
        raise HTTPException(404, f"Configs não encontradas: {', '.join(missing)}")
    cfgs = [by_id[i] for i in req.ids]

    try:
        builder_mod.router_binary_for(cfgs)  # binário único + existente
    except builder_mod.BackendBinaryMissing as e:
        raise HTTPException(400, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))

    # O binário precisa ser recente o bastante pra ter o modo router.
    probe_backend = cfgs[0].get("backend", "turbo")
    help_text = backends_mod.backend_help_text(probe_backend, "server")
    if help_text and "--models-preset" not in help_text:
        raise HTTPException(
            400,
            f"O llama-server do backend '{probe_backend}' não suporta o modo "
            "router (--models-preset). Atualize/recompile o binário.",
        )

    _, model_ids = builder_mod.build_router_preset(cfgs)
    router_cfg = {
        "router":     True,
        "backend":    probe_backend,
        "config_ids": req.ids,
        "model_ids":  model_ids,
        "models":     [c.get("model", "") for c in cfgs],
    }
    session = LaunchSession(router_cfg, member_cfgs=cfgs)
    with _SESSIONS_LOCK:
        _SESSIONS[session.id] = session
    session.start()
    return {"launch_id": session.id, "config": router_cfg}


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
                # Queue.get bloqueante NUNCA pode rodar direto no event loop:
                # trava o uvicorn inteiro (todas as rotas) enquanto espera o
                # próximo evento — a UI congela junto. to_thread manda a
                # espera pro thread pool e o loop segue servindo requests.
                ev = await asyncio.to_thread(session.queue.get, True, 0.5)
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
    if session.attached:
        # Sessão re-anexada não tem thread observando — fecha a sessão na mão.
        session.done = True
        running.deregister(session.id)
        session.queue.put({"type": "giveup", "reason": "cancelled"})
        session.queue.put({"type": "_eof"})
    return {"ok": True}


@app.post("/api/launch/{launch_id}/restart")
def launch_restart(launch_id: str):
    """Restart 'soft': mata o llama-server atual e o loop resiliente sobe de
    novo com a MESMA config (porta fixa → cliente reconecta). Não desiste,
    não degrada. Serve pra destravar um server vivo-mas-preso."""
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(launch_id)
    if session is None:
        raise HTTPException(404, "launch_id desconhecido")
    if session.done:
        raise HTTPException(400, "launch já encerrado — use Launch pra subir de novo")
    if session.attached:
        # Órfão reanexado: não há thread/loop pra ressubir, só o PID.
        raise HTTPException(
            400, "sessão reanexada não pode reiniciar (sem config/loop ativo)")
    ok = session.handle.request_restart()
    if not ok:
        raise HTTPException(409, "não foi possível reiniciar (cancelado ou sem processo)")
    return {"ok": True}


# ─── routes: launches ativos (sobreviventes a restart do Python) ──────────────

@app.get("/api/launches")
def list_launches():
    """Lista launches ativos — usado no boot do frontend pra recuperar
    botão de Stop de servidores órfãos da sessão anterior."""
    out = []
    with _SESSIONS_LOCK:
        for s in _SESSIONS.values():
            if s.done:
                continue
            out.append({
                "launch_id": s.id,
                "attached":  s.attached,
                "pid":       s.handle.pid,
                "config":    s.cfg,
            })
    return out


def _attach_orphan_servers() -> None:
    """Boot-time: re-anexa llama-server.exe órfãos da sessão anterior.

    Chamado uma vez no startup. Reconcilia o api_running.json filtrando
    PIDs mortos / reciclados, depois cria LaunchSession sintética pra
    cada sobrevivente — assim a UI consegue dar Stop neles.
    """
    survivors = running.reconcile()
    for entry in survivors:
        sid = entry.get("launch_id")
        pid = int(entry.get("pid", 0) or 0)
        cfg = entry.get("cfg") or {}
        if not sid or not pid or not cfg:
            continue
        session = LaunchSession(cfg, attached_pid=pid, launch_id=sid)
        with _SESSIONS_LOCK:
            _SESSIONS[session.id] = session




# ─── routes: HuggingFace download ─────────────────────────────────────────────

class HfUrlRequest(BaseModel):
    url: str


@app.post("/api/hf/resolve")
def hf_resolve(req: HfUrlRequest):
    try:
        repo_id, filename, branch = hf.parse_hf_url(req.url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"repo_id": repo_id, "filename": filename, "branch": branch}


class HfRepoRequest(BaseModel):
    repo_id: str


@app.post("/api/hf/list")
def hf_list(req: HfRepoRequest):
    try:
        files = hf.hf_list_files(req.repo_id)
    except Exception as e:
        raise HTTPException(400, f"falha ao listar {req.repo_id}: {e}")
    grouped = hf.group_gguf_files(files)
    return {"repo_id": req.repo_id, "files": files, **grouped}


class HfSearchRequest(BaseModel):
    query: str
    limit: int = 25


@app.post("/api/hf/search")
def hf_search(req: HfSearchRequest):
    try:
        return {"results": hf.hf_search_gguf(req.query, req.limit)}
    except Exception as e:
        raise HTTPException(400, f"busca falhou: {e}")


class HfDownloadRequest(BaseModel):
    repo_id: str
    branch: str = "main"
    rel_paths: list[str]
    subdir: str | None = None
    base_dir: str
    # Re-baixa mesmo se o arquivo já existir com o tamanho certo. Usado pelo
    # botão "Atualizar", onde a divergência foi detectada por sha256.
    force: bool = False


class DownloadSession:
    def __init__(self, plan: dict):
        self.id = uuid.uuid4().hex[:12]
        self.plan = plan
        self.queue: Queue[dict] = Queue()
        self.thread: threading.Thread | None = None
        self.cancelled = False

    def start(self) -> None:
        def _target():
            try:
                for ev in hf.stream_download(self.plan):
                    if self.cancelled:
                        self.queue.put({"type": "cancelled"})
                        break
                    self.queue.put(ev)
            except Exception as e:
                self.queue.put({"type": "error", "rel": "", "message": str(e)})
            finally:
                self.queue.put({"type": "_eof"})

        self.thread = threading.Thread(target=_target, daemon=True)
        self.thread.start()


_DOWNLOADS: dict[str, DownloadSession] = {}
_DOWNLOADS_LOCK = threading.Lock()


@app.post("/api/hf/download")
def hf_download(req: HfDownloadRequest):
    configured = settings_store.get_configured_paths()
    if not configured:
        raise HTTPException(
            400,
            "Nenhuma pasta de modelos cadastrada. Vá em Settings e adicione "
            "pelo menos um caminho antes de baixar.",
        )
    chosen = Path(req.base_dir)
    # Compara strings normalizadas — Path.__eq__ no Windows é case-sensitive,
    # mas o filesystem não. Resolve nas duas pontas pra evitar mismatch por
    # trailing slash / case.
    def _norm(p: Path) -> str:
        try:
            return str(p.resolve()).lower()
        except Exception:
            return str(p).lower()
    allowed = {_norm(p) for p in configured}
    if _norm(chosen) not in allowed:
        raise HTTPException(
            400,
            f"Caminho '{req.base_dir}' não está cadastrado. Caminhos válidos: "
            + ", ".join(str(p) for p in configured),
        )
    plan = hf.plan_download(req.repo_id, req.branch, req.rel_paths, req.subdir,
                            chosen, force=req.force)
    session = DownloadSession(plan)
    with _DOWNLOADS_LOCK:
        _DOWNLOADS[session.id] = session
    session.start()
    return {"download_id": session.id, "plan": plan}


@app.get("/api/hf/download/{download_id}/events")
async def hf_download_events(download_id: str, request: Request):
    with _DOWNLOADS_LOCK:
        session = _DOWNLOADS.get(download_id)
    if session is None:
        raise HTTPException(404, "download_id desconhecido")

    async def generator():
        while True:
            if await request.is_disconnected():
                return
            try:
                # to_thread: ver comentário no SSE de launch — get bloqueante
                # no event loop congelava a API inteira durante o download.
                ev = await asyncio.to_thread(session.queue.get, True, 0.5)
            except Empty:
                yield ": keepalive\n\n"
                continue
            if ev.get("type") == "_eof":
                return
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.post("/api/hf/download/{download_id}/cancel")
def hf_download_cancel(download_id: str):
    with _DOWNLOADS_LOCK:
        session = _DOWNLOADS.get(download_id)
    if session is None:
        raise HTTPException(404, "download_id desconhecido")
    session.cancelled = True
    return {"ok": True}


# ─── routes: delete model ─────────────────────────────────────────────────────

class DeleteModelRequest(BaseModel):
    model: str
    confirm: bool = False


@app.post("/api/models/plan-delete")
def model_plan_delete(req: DeleteModelRequest):
    """Lista o que seria removido sem apagar nada."""
    files = models_repo.plan_delete_model(Path(req.model))
    total = 0
    for f in files:
        try:
            total += f.stat().st_size
        except Exception:
            pass
    return {
        "files": [str(f) for f in files],
        "total_bytes": total,
        "count": len(files),
    }


@app.delete("/api/models")
def model_delete(req: DeleteModelRequest):
    if not req.confirm:
        raise HTTPException(400, "confirm=true é obrigatório")
    result = models_repo.delete_model(Path(req.model))
    # Também remove as configs salvas do modelo (todos os backends)
    config_store.delete_config(req.model, backend=None)
    return result


# ─── routes: LM Studio fallback ───────────────────────────────────────────────

@app.get("/api/lms/status")
def lms_status_route():
    return lms.lms_status()


class LmsLoadRequest(BaseModel):
    model: str
    context_window: int = 65_536
    parallel_slots: int = 1


@app.post("/api/lms/load")
def lms_load(req: LmsLoadRequest):
    return lms.load_via_lms(Path(req.model), req.context_window, req.parallel_slots)


# ─── routes: MCP servers ──────────────────────────────────────────────────────

class McpServerPayload(BaseModel):
    id: str | None = None
    name: str = ""
    cwd: str = ""
    command: str = ""
    enabled: bool = False


@app.get("/api/mcp")
def mcp_list():
    """Lista servers + status vivo de cada um."""
    servers = mcp_store.list_servers()
    statuses = mcp_runner.list_status()
    out = []
    for s in servers:
        sid = s.get("id", "")
        out.append({**s, "status": statuses.get(sid, mcp_runner.status_for(sid))})
    return out


@app.post("/api/mcp")
def mcp_upsert(payload: McpServerPayload):
    data = payload.model_dump()
    entry = mcp_store.upsert_server(data)
    return {"ok": True, "server": entry, "status": mcp_runner.status_for(entry["id"])}


@app.delete("/api/mcp/{server_id}")
def mcp_delete(server_id: str):
    # Para o processo antes de remover do disco, pra não deixar órfão.
    try:
        mcp_runner.stop(server_id)
    except Exception:
        pass
    removed = mcp_store.delete_server(server_id)
    return {"ok": removed}


@app.post("/api/mcp/{server_id}/start")
def mcp_start(server_id: str):
    try:
        return mcp_runner.start(server_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/mcp/{server_id}/stop")
def mcp_stop(server_id: str):
    return mcp_runner.stop(server_id)


@app.get("/api/mcp/{server_id}/logs")
def mcp_logs(server_id: str, limit: int = 500):
    return {"logs": mcp_runner.get_logs(server_id, limit=limit)}


@app.get("/api/mcp/{server_id}/events")
async def mcp_events(server_id: str, request: Request):
    inst = mcp_runner.get_instance(server_id)
    if inst is None:
        raise HTTPException(404, "servidor MCP não está em execução")
    q = inst.subscribe()

    async def generator():
        try:
            # Backfill: manda o buffer atual antes de seguir streaming.
            for ev in list(inst.log):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            while True:
                if await request.is_disconnected():
                    return
                try:
                    # to_thread: mesmo motivo dos outros SSE — não bloquear o loop.
                    ev = await asyncio.to_thread(q.get, True, 0.5)
                except Empty:
                    yield ": keepalive\n\n"
                    continue
                if ev.get("kind") == "_eof":
                    return
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        finally:
            inst.unsubscribe(q)

    return StreamingResponse(generator(), media_type="text/event-stream")


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
