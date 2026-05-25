"""Registry de launches ativos — sobrevive a restart do Python.

O Electron mata o processo Python quando a janela fecha (window-all-closed),
mas o llama-server.exe spawneado por Popen sobrevive no Windows (não vai pra
job object). Ao reabrir o app, o Python é novo e perdeu o handle do servidor
órfão — o usuário não consegue mais dar Stop.

Solução: cada launch grava seu PID em api_running.json. No boot do server.py,
varremos o arquivo, verificamos se cada PID ainda existe E ainda é um
llama-server (não um PID reciclado), e re-anexamos como sessão "detached".
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
REGISTRY_FILE = _HERE / "api_running.json"

_LOCK = threading.Lock()


def _read_all() -> list[dict]:
    try:
        if REGISTRY_FILE.exists():
            data = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def _write_all(entries: list[dict]) -> None:
    try:
        REGISTRY_FILE.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def register(launch_id: str, pid: int, cfg: dict) -> None:
    """Adiciona/atualiza o launch_id no registry."""
    with _LOCK:
        entries = [e for e in _read_all() if e.get("launch_id") != launch_id]
        entries.append({
            "launch_id": launch_id,
            "pid": int(pid),
            "cfg": cfg,
            "started_at": time.time(),
        })
        _write_all(entries)


def deregister(launch_id: str) -> None:
    with _LOCK:
        entries = [e for e in _read_all() if e.get("launch_id") != launch_id]
        _write_all(entries)


def list_active() -> list[dict]:
    return _read_all()


# ─── PID liveness ─────────────────────────────────────────────────────────

def pid_alive(pid: int) -> bool:
    """True se o processo com `pid` ainda existe."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid,
            )
            if h == 0:
                return False
            # Confere se ainda está rodando (STILL_ACTIVE = 259)
            exit_code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(exit_code))
            ctypes.windll.kernel32.CloseHandle(h)
            return exit_code.value == 259
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except Exception:
        return False


def pid_is_llama_server(pid: int) -> bool:
    """True se o PID for de um processo llama-server.* — guard anti-PID-reuse."""
    if not pid_alive(pid):
        return False
    if sys.platform == "win32":
        try:
            r = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, encoding="utf-8", timeout=5,
            )
            return "llama-server" in (r.stdout or "").lower()
        except Exception:
            return False
    try:
        with open(f"/proc/{pid}/comm") as f:
            return "llama" in f.read().lower()
    except Exception:
        return False


def kill_pid_tree(pid: int) -> bool:
    """Mata o PID e todos os descendentes. Best-effort."""
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            r = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True, timeout=10,
            )
            return r.returncode == 0
        import signal
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            os.kill(pid, signal.SIGTERM)
        return True
    except Exception:
        return False


def reconcile() -> list[dict]:
    """Limpa entradas mortas e devolve as ativas confirmadas.

    Chamado no boot do server.py. Remove entradas cujo PID já não existe
    (crash, reboot, kill manual) ou cujo PID foi reciclado por outro
    processo qualquer.
    """
    with _LOCK:
        entries = _read_all()
        alive = [e for e in entries if pid_is_llama_server(int(e.get("pid", 0)))]
        if len(alive) != len(entries):
            _write_all(alive)
    return alive
