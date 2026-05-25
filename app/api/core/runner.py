"""Executor resiliente do llama-server com streaming de eventos.

Substitui run_server_resiliently de models.py. Em vez de imprimir tudo no
stdout, dispara callbacks tipados (`on_event`) — a camada HTTP plugada
encaminha cada evento como SSE pro frontend.

Eventos:
    {"type": "stdout",    "line": "..."}                       — linha bruta
    {"type": "load_ok",   "attempt": N}                        — load detectado
    {"type": "stabilized", "attempt": N}                       — sobreviveu STABILIZE_SECONDS
    {"type": "exit",      "attempt": N, "rc": int}             — processo encerrou
    {"type": "failure",   "category": "OOM_RUNTIME",
                          "excerpt": "...", "attempt": N}      — queda classificada
    {"type": "degrade",   "description": "...",  "config": {…}} — degrau aplicado
    {"type": "restart",   "attempt": N, "backoff": s}          — vai reiniciar
    {"type": "giveup",    "reason": "…"}                       — escada esgotada / loop
    {"type": "done"}                                           — terminou normalmente
"""
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from .builder import build_command_from_cfg
from .config_store import append_fail_history, save_config
from .constants import LOAD_OK_PATTERNS, STABILIZE_SECONDS
from .failure import Failure, classify_failure, next_degrade

EventCallback = Callable[[dict], None]

SERVER_MAX_RESTARTS  = 10
SERVER_RESTART_WINDOW = 120
SERVER_BACKOFF_START  = 2
SERVER_BACKOFF_MAX    = 30


class LaunchHandle:
    """Handle de um launch em curso — permite cancelar de fora (UI cancel)."""
    def __init__(self):
        self._cancelled = False
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass

    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def _set_proc(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._proc = proc


def _run_capturing(cmd: str, handle: LaunchHandle, on_line: Callable[[str], None],
                   on_load: Callable[[], None]) -> tuple[int, str, float | None]:
    proc = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
    )
    handle._set_proc(proc)
    chunks: list[str] = []
    load_ts: float | None = None
    assert proc.stdout is not None
    for line in proc.stdout:
        if handle.cancelled():
            try:
                proc.terminate()
            except Exception:
                pass
            break
        chunks.append(line)
        on_line(line.rstrip("\n"))
        if load_ts is None:
            low = line.lower()
            if any(p in low for p in LOAD_OK_PATTERNS):
                load_ts = time.time()
                try:
                    on_load()
                except Exception:
                    pass
    rc = proc.wait()
    return rc, "".join(chunks), load_ts


def stop_lms_server_if_running() -> None:
    """Para o servidor do LM Studio (libera a porta 1234)."""
    from .constants import LMS
    if not LMS.exists():
        return
    try:
        status = subprocess.run(
            [str(LMS), "server", "status"],
            capture_output=True, text=True, encoding="utf-8", timeout=10,
        )
        out = ((status.stdout or "") + (status.stderr or "")).lower()
        if "running" in out and "not running" not in out:
            subprocess.run([str(LMS), "server", "stop"], timeout=10)
    except Exception:
        pass


def run_server_resiliently(
    cfg: dict,
    on_event: EventCallback,
    handle: LaunchHandle | None = None,
    stop_lms: bool = True,
) -> None:
    """Loop: launch → detecta load → roda → se cair, classifica + degrada + reinicia.

    Promove a config a salva quando sobrevive STABILIZE_SECONDS pós-load.
    Encerra ao receber cancel via `handle`, ou rc==0, ou escada esgotada.
    """
    if handle is None:
        handle = LaunchHandle()
    if stop_lms:
        stop_lms_server_if_running()

    restarts: list[float] = []
    backoff = SERVER_BACKOFF_START
    attempt = 0

    while True:
        if handle.cancelled():
            on_event({"type": "giveup", "reason": "cancelled"})
            return

        attempt += 1
        try:
            cmd = build_command_from_cfg(cfg, mode="server")
        except Exception as e:
            on_event({"type": "giveup", "reason": f"build_command falhou: {e}"})
            return

        on_event({"type": "start", "attempt": attempt, "cmd": cmd, "config": dict(cfg)})

        attempt_cfg = dict(cfg)

        def on_load() -> None:
            save_config(attempt_cfg)
            on_event({"type": "load_ok", "attempt": attempt})

        def on_line(line: str) -> None:
            on_event({"type": "stdout", "line": line})

        rc, output, load_ts = _run_capturing(cmd, handle, on_line, on_load)
        on_event({"type": "exit", "attempt": attempt, "rc": rc, "load_ts": load_ts})

        if handle.cancelled():
            on_event({"type": "giveup", "reason": "cancelled"})
            return

        stabilized = load_ts is not None and (time.time() - load_ts) >= STABILIZE_SECONDS
        if rc == 0:
            if stabilized:
                save_config(attempt_cfg)
            on_event({"type": "done", "attempt": attempt})
            return

        failure, excerpt = classify_failure(output, rc, load_ts)
        on_event({"type": "failure", "category": failure, "excerpt": excerpt, "attempt": attempt})

        if failure == Failure.MMPROJ:
            append_fail_history(attempt_cfg, failure, excerpt, None, attempt)
            on_event({"type": "giveup", "reason": "mmproj_unsupported",
                      "hint": "lms_fallback_available"})
            return

        new_cfg, degrade_desc = next_degrade(cfg, failure, excerpt)
        append_fail_history(attempt_cfg, failure, excerpt, degrade_desc, attempt)

        if new_cfg is None:
            on_event({"type": "giveup", "reason": "ladder_exhausted",
                      "failure": failure, "excerpt": excerpt})
            return

        now = time.time()
        restarts = [t for t in restarts if now - t < SERVER_RESTART_WINDOW]
        restarts.append(now)
        if len(restarts) > SERVER_MAX_RESTARTS:
            on_event({"type": "giveup", "reason": "too_many_restarts",
                      "window_s": SERVER_RESTART_WINDOW,
                      "count": len(restarts)})
            return

        on_event({"type": "degrade", "description": degrade_desc, "config": dict(new_cfg)})
        on_event({"type": "restart", "attempt": attempt + 1, "backoff": backoff})

        slept = 0.0
        step = 0.25
        while slept < backoff:
            if handle.cancelled():
                on_event({"type": "giveup", "reason": "cancelled"})
                return
            time.sleep(step)
            slept += step

        cfg = new_cfg
        backoff = min(backoff * 2, SERVER_BACKOFF_MAX)
