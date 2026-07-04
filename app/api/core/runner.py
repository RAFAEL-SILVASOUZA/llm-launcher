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
    {"type": "restart",   "attempt": N, "backoff": s}          — vai reiniciar (degrade)
    {"type": "manual_restart", "attempt": N}                   — restart pedido pela UI (mesma config)
    {"type": "giveup",    "reason": "…"}                       — escada esgotada / loop
    {"type": "done"}                                           — terminou normalmente
"""
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from .builder import build_command_from_cfg, build_router_command, build_router_preset
from .config_store import append_fail_history, save_config
from .constants import LOAD_OK_PATTERNS, STABILIZE_SECONDS
from .failure import Failure, classify_failure, next_degrade
from .nvidia import cuda_env

EventCallback = Callable[[dict], None]

SERVER_MAX_RESTARTS  = 10
SERVER_RESTART_WINDOW = 120
SERVER_BACKOFF_START  = 2
SERVER_BACKOFF_MAX    = 30


class LaunchHandle:
    """Handle de um launch — permite cancelar de fora (UI cancel).

    Dois modos:
      - normal: spawneamos via Popen, `_proc` está setado.
      - attached: re-anexamos um llama-server órfão de uma sessão anterior do
        Python; só temos o PID. `_proc` é None, cancel usa kill_pid_tree.
    """
    def __init__(self, attached_pid: int | None = None):
        self._cancelled = False
        self._restart = False
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._attached_pid: int | None = attached_pid

    @property
    def pid(self) -> int | None:
        if self._proc is not None:
            return self._proc.pid
        return self._attached_pid

    @property
    def attached(self) -> bool:
        return self._proc is None and self._attached_pid is not None

    def cancel(self) -> None:
        # Import tardio pra evitar ciclo (running → runner não importa, OK).
        from . import running
        with self._lock:
            self._cancelled = True
            pid = self.pid
            if pid is None:
                return
            if self._proc is not None and self._proc.poll() is not None:
                return  # já encerrou sozinho
            running.kill_pid_tree(pid)
            if self._proc is not None:
                try:
                    self._proc.wait(timeout=5)
                except Exception:
                    pass

    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def request_restart(self) -> bool:
        """Mata o processo atual mas NÃO desiste: o loop resiliente sobe de
        novo com a MESMA config (sem degradar). Usado pra destravar um server
        vivo-mas-preso e deixar o cliente (porta fixa) reconectar.

        Retorna False se já foi cancelado ou se não há processo pra reiniciar.
        """
        from . import running
        with self._lock:
            if self._cancelled:
                return False
            pid = self.pid
            if pid is None:
                return False
            self._restart = True
            # Se já morreu sozinho, o loop já vai reentrar — só sinaliza.
            if self._proc is not None and self._proc.poll() is not None:
                return True
            running.kill_pid_tree(pid)
            if self._proc is not None:
                try:
                    self._proc.wait(timeout=5)
                except Exception:
                    pass
        return True

    def restart_requested(self) -> bool:
        with self._lock:
            return self._restart

    def consume_restart(self) -> bool:
        """Lê-e-limpa o flag de restart de forma atômica."""
        with self._lock:
            if self._restart:
                self._restart = False
                return True
            return False

    def _set_proc(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._proc = proc
            self._attached_pid = None


def _run_capturing(cmd: str, handle: LaunchHandle, on_line: Callable[[str], None],
                   on_load: Callable[[], None],
                   on_proc_pid: Callable[[int], None] | None = None,
                   ) -> tuple[int, str, float | None]:
    # Removido shell=True: no Windows, shell=True cria cmd.exe wrapper que
    # survive ao terminate(), deixando o llama-server rodando sozinho.
    proc = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
        env=cuda_env(),  # 3090 Ti como device CUDA 0 → main GPU (-mg 0)
    )
    handle._set_proc(proc)
    if on_proc_pid is not None:
        try:
            on_proc_pid(proc.pid)
        except Exception:
            pass
    chunks: list[str] = []
    load_ts: float | None = None
    assert proc.stdout is not None
    for line in proc.stdout:
        if handle.cancelled() or handle.restart_requested():
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


# Preset do modo router — regravado a cada launch múltiplo. Vive ao lado do
# api_running.json (pasta api/), fora das pastas de modelos do usuário.
ROUTER_PRESET_FILE = Path(__file__).resolve().parent.parent / "router_preset.ini"


def run_router_server(
    cfgs: list[dict],
    router_cfg: dict,
    on_event: EventCallback,
    handle: LaunchHandle | None = None,
    stop_lms: bool = True,
    on_proc_pid: Callable[[int], None] | None = None,
) -> None:
    """Sobe o llama-server em MODO ROUTER com N modelos (um por seção do
    preset INI, todos com load-on-startup). Sem escada de auto-degrade: quem
    cai é o processo filho de UM modelo e o próprio router o gerencia/recarrega;
    se o ROUTER inteiro morrer, reportamos e desistimos — degradar uma config
    específica não faz sentido aqui. Cancel e restart manual funcionam igual
    ao launch normal (kill na árvore inteira mata router + filhos).
    """
    if handle is None:
        handle = LaunchHandle()
    if stop_lms:
        stop_lms_server_if_running()

    try:
        preset_text, names = build_router_preset(cfgs)
        ROUTER_PRESET_FILE.write_text(preset_text, encoding="utf-8")
        cmd = build_router_command(cfgs, ROUTER_PRESET_FILE)
    except Exception as e:
        on_event({"type": "giveup", "reason": f"build do router falhou: {e}"})
        return

    attempt = 0
    while True:
        if handle.cancelled():
            on_event({"type": "giveup", "reason": "cancelled"})
            return

        attempt += 1
        on_event({"type": "start", "attempt": attempt, "cmd": cmd,
                  "config": {**router_cfg, "model_ids": names}})
        on_event({"type": "stdout", "line": f"── preset: {ROUTER_PRESET_FILE} ──"})
        for line in preset_text.splitlines():
            on_event({"type": "stdout", "line": f"    {line}"})

        def on_load() -> None:
            on_event({"type": "load_ok", "attempt": attempt})

        def on_line(line: str) -> None:
            on_event({"type": "stdout", "line": line})

        rc, output, load_ts = _run_capturing(
            cmd, handle, on_line, on_load, on_proc_pid=on_proc_pid,
        )
        on_event({"type": "exit", "attempt": attempt, "rc": rc, "load_ts": load_ts})

        if handle.cancelled():
            on_event({"type": "giveup", "reason": "cancelled"})
            return

        # Restart manual (botão da UI): mata router + filhos e sobe de novo
        # com o mesmo preset.
        if handle.consume_restart():
            on_event({"type": "manual_restart", "attempt": attempt + 1})
            slept = 0.0
            while slept < 1.0:
                if handle.cancelled():
                    on_event({"type": "giveup", "reason": "cancelled"})
                    return
                time.sleep(0.25)
                slept += 0.25
            continue

        if rc == 0:
            on_event({"type": "done", "attempt": attempt})
            return

        excerpt = "\n".join(output.splitlines()[-15:])
        on_event({"type": "failure", "category": "ROUTER_EXIT",
                  "excerpt": excerpt, "attempt": attempt})
        on_event({"type": "giveup", "reason": "router_exit", "excerpt": excerpt})
        return


def run_server_resiliently(
    cfg: dict,
    on_event: EventCallback,
    handle: LaunchHandle | None = None,
    stop_lms: bool = True,
    on_proc_pid: Callable[[int], None] | None = None,
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

        rc, output, load_ts = _run_capturing(
            cmd, handle, on_line, on_load, on_proc_pid=on_proc_pid,
        )
        on_event({"type": "exit", "attempt": attempt, "rc": rc, "load_ts": load_ts})

        if handle.cancelled():
            on_event({"type": "giveup", "reason": "cancelled"})
            return

        # Restart manual (botão da UI): sobe de novo com a MESMA config, sem
        # classificar como falha nem aplicar degrade, e sem contar no limite de
        # too_many_restarts. Apenas zera o backoff e reentra no loop.
        if handle.consume_restart():
            backoff = SERVER_BACKOFF_START
            on_event({"type": "manual_restart", "attempt": attempt + 1})
            slept = 0.0
            while slept < 1.0:
                if handle.cancelled():
                    on_event({"type": "giveup", "reason": "cancelled"})
                    return
                time.sleep(0.25)
                slept += 0.25
            continue

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

        # Arquivo corrompido / shard faltando: auto-degrade não consegue
        # consertar. Dá giveup direto com hint pro usuário rebaixar o quant.
        if failure == Failure.MODEL_CORRUPTED:
            append_fail_history(attempt_cfg, failure, excerpt, None, attempt)
            on_event({"type": "giveup", "reason": "model_corrupted",
                      "excerpt": excerpt, "hint": "redownload_model"})
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
