"""Runner de servidores MCP locais — spawn, captura logs, auto-stop on error.

Modelo:
  - Cada server tem 1 instância gerenciada por server_id.
  - Spawn via subprocess.Popen com cwd e shell=True (precisa expandir
    "npm run dev" etc. via cmd.exe no Windows).
  - Logs vão pra um buffer ring + um Queue por subscriber SSE.
  - Se o processo morre por qualquer motivo, marcamos enabled=False e
    guardamos last_exit_code/last_error pra UI mostrar.

Detalhe de Windows:
  - shell=True cria cmd.exe wrapper; pra matar a árvore inteira (npm → node)
  - usamos taskkill /T /F via running.kill_pid_tree.
"""
from __future__ import annotations

import collections
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Optional

from . import mcp_store, running

LOG_BUFFER_LINES = 2000


class McpInstance:
    """Estado vivo de um servidor MCP em execução."""

    def __init__(self, server_id: str, cfg: dict):
        self.id = server_id
        self.cfg = dict(cfg)
        self.proc: subprocess.Popen | None = None
        self.started_at: float | None = None
        self.stopped_at: float | None = None
        self.last_exit_code: int | None = None
        self.last_error: str | None = None
        self.log: collections.deque[dict] = collections.deque(maxlen=LOG_BUFFER_LINES)
        self.subscribers: list[Queue] = []
        self._lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        self._auto_stopped = False

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc is not None else None

    def to_status(self) -> dict:
        return {
            "id":              self.id,
            "running":         self.running,
            "pid":             self.pid,
            "started_at":      self.started_at,
            "stopped_at":      self.stopped_at,
            "last_exit_code":  self.last_exit_code,
            "last_error":      self.last_error,
            "auto_stopped":    self._auto_stopped,
        }

    def append_log(self, kind: str, text: str) -> None:
        ev = {"ts": time.time(), "kind": kind, "text": text}
        self.log.append(ev)
        with self._lock:
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.put_nowait(ev)
            except Exception:
                pass

    def subscribe(self) -> Queue:
        q: Queue = Queue()
        with self._lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: Queue) -> None:
        with self._lock:
            try:
                self.subscribers.remove(q)
            except ValueError:
                pass


_INSTANCES: dict[str, McpInstance] = {}
_INSTANCES_LOCK = threading.Lock()


def _get_or_create(server_id: str, cfg: dict) -> McpInstance:
    with _INSTANCES_LOCK:
        inst = _INSTANCES.get(server_id)
        if inst is None:
            inst = McpInstance(server_id, cfg)
            _INSTANCES[server_id] = inst
        else:
            inst.cfg = dict(cfg)
        return inst


def get_instance(server_id: str) -> McpInstance | None:
    with _INSTANCES_LOCK:
        return _INSTANCES.get(server_id)


def status_for(server_id: str) -> dict:
    inst = get_instance(server_id)
    if inst is None:
        return {
            "id":             server_id,
            "running":        False,
            "pid":            None,
            "started_at":     None,
            "stopped_at":     None,
            "last_exit_code": None,
            "last_error":     None,
            "auto_stopped":   False,
        }
    return inst.to_status()


def list_status() -> dict[str, dict]:
    with _INSTANCES_LOCK:
        return {sid: inst.to_status() for sid, inst in _INSTANCES.items()}


def start(server_id: str) -> dict:
    """Liga o server. Idempotente: se já está rodando, no-op."""
    cfg = mcp_store.get_server(server_id)
    if cfg is None:
        raise ValueError(f"servidor MCP '{server_id}' não cadastrado")

    cwd = (cfg.get("cwd") or "").strip()
    command = (cfg.get("command") or "").strip()
    if not cwd or not command:
        raise ValueError("cwd e command são obrigatórios")
    cwd_path = Path(cwd)
    if not cwd_path.exists() or not cwd_path.is_dir():
        raise ValueError(f"cwd não existe ou não é diretório: {cwd}")

    inst = _get_or_create(server_id, cfg)
    if inst.running:
        return inst.to_status()

    inst.last_exit_code = None
    inst.last_error = None
    inst.stopped_at = None
    inst._auto_stopped = False
    inst.log.clear()

    try:
        # shell=True pra resolver `npm run dev`, `node x.js`, etc. via cmd.exe.
        # creationflags ajuda a manter a árvore num grupo separado pra
        # taskkill /T conseguir derrubar o node filho.
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        proc = subprocess.Popen(
            command,
            cwd=str(cwd_path),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )
    except Exception as e:
        inst.last_error = f"falha ao iniciar: {e}"
        inst.append_log("error", inst.last_error)
        mcp_store.set_enabled(server_id, False)
        return inst.to_status()

    inst.proc = proc
    inst.started_at = time.time()
    inst.append_log("info", f"started pid={proc.pid} cmd={command} cwd={cwd}")
    mcp_store.set_enabled(server_id, True)

    inst._reader_thread = threading.Thread(
        target=_drain, args=(inst,), daemon=True,
    )
    inst._reader_thread.start()
    return inst.to_status()


def _drain(inst: McpInstance) -> None:
    """Lê stdout/stderr linha-a-linha e detecta morte do processo.

    Se o processo morre com rc != 0, marca enabled=False (auto-stop on error)
    e sinaliza pro front via `auto_stopped`.
    """
    proc = inst.proc
    if proc is None or proc.stdout is None:
        return
    try:
        for line in proc.stdout:
            inst.append_log("stdout", line.rstrip("\n"))
    except Exception as e:
        inst.append_log("error", f"erro lendo stdout: {e}")
    rc = proc.wait()
    inst.last_exit_code = rc
    inst.stopped_at = time.time()
    if rc != 0:
        inst._auto_stopped = True
        inst.last_error = f"processo encerrou com rc={rc}"
        inst.append_log("error", f"auto-stop: processo encerrou com rc={rc}")
        mcp_store.set_enabled(inst.id, False)
    else:
        inst.append_log("info", f"processo encerrou rc={rc}")
        mcp_store.set_enabled(inst.id, False)
    # Sinaliza EOF pra subscribers fecharem o SSE.
    with inst._lock:
        subs = list(inst.subscribers)
    for q in subs:
        try:
            q.put_nowait({"ts": time.time(), "kind": "_eof", "text": ""})
        except Exception:
            pass


def stop(server_id: str) -> dict:
    """Desliga o server. Idempotente."""
    inst = get_instance(server_id)
    if inst is None:
        mcp_store.set_enabled(server_id, False)
        return status_for(server_id)
    if not inst.running:
        mcp_store.set_enabled(server_id, False)
        return inst.to_status()

    pid = inst.pid
    inst.append_log("info", f"stop requested (pid={pid})")
    if pid:
        running.kill_pid_tree(pid)
    try:
        if inst.proc is not None:
            inst.proc.wait(timeout=5)
    except Exception:
        pass
    mcp_store.set_enabled(server_id, False)
    return inst.to_status()


def get_logs(server_id: str, limit: int = 500) -> list[dict]:
    inst = get_instance(server_id)
    if inst is None:
        return []
    snap = list(inst.log)
    if limit > 0:
        snap = snap[-limit:]
    return snap
