"""nvidia-smi: VRAM (total/livre/usada), status detalhado e RAM total do sistema."""
import subprocess
import sys
import time
from pathlib import Path

from .constants import NVIDIA_SMI

_MIB = 1024 * 1024
# (timestamp, (total_mib, free_mib)). TTL curto: dados de VRAM mudam o tempo
# todo, queremos a barra do header acompanhando, mas sem rodar nvidia-smi a
# cada chamada do estimador.
_VRAM_INFO_CACHE: tuple[float, tuple[int, int]] | None = None
_VRAM_TTL_SECS = 2.0

_RAM_TOTAL_CACHE: int | None = None

NVIDIA_QUERY_FIELDS = [
    "name", "driver_version",
    "temperature.gpu", "fan.speed",
    "utilization.gpu", "utilization.memory",
    "memory.total", "memory.used", "memory.free",
    "power.draw", "power.limit",
    "clocks.sm", "clocks.mem",
    "pstate",
]


def nvidia_smi_path() -> str:
    return str(NVIDIA_SMI) if NVIDIA_SMI.exists() else "nvidia-smi"


def gpu_mem_mib(force_refresh: bool = False) -> tuple[int | None, int | None]:
    """(total, livre) em MiB. (None, None) se indisponível.

    Cache com TTL curto (_VRAM_TTL_SECS) — múltiplas chamadas dentro de uma
    request HTTP reusam o resultado, mas o próximo poll do header pega
    valores atualizados. Antes era cache eterno, o que travava as barras.
    """
    global _VRAM_INFO_CACHE
    now = time.time()
    if (
        _VRAM_INFO_CACHE is not None
        and not force_refresh
        and (now - _VRAM_INFO_CACHE[0]) < _VRAM_TTL_SECS
    ):
        t, fr = _VRAM_INFO_CACHE[1]
        return (t or None, fr or None)
    smi = nvidia_smi_path()
    try:
        r = subprocess.run(
            [smi, "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            total_s, free_s = (x.strip() for x in r.stdout.strip().splitlines()[0].split(","))
            _VRAM_INFO_CACHE = (now, (int(total_s), int(free_s)))
            return _VRAM_INFO_CACHE[1]
    except Exception:
        pass
    _VRAM_INFO_CACHE = (now, (0, 0))
    return (None, None)


def gpu_total_mib() -> int | None:
    return gpu_mem_mib()[0]


def gpu_free_mib() -> int | None:
    return gpu_mem_mib()[1]


def _ram_mib_pair() -> tuple[int | None, int | None]:
    """Devolve (total, avail) em MiB. `total` é cacheado (não muda durante a
    sessão); `avail` é sempre fresco — usado por show_memory_estimate e pela
    barra do header pra refletir RAM realmente livre agora.
    """
    global _RAM_TOTAL_CACHE
    try:
        if sys.platform == "win32":
            import ctypes
            class _MEMSTAT(ctypes.Structure):
                _fields_ = [
                    ("dwLength",                ctypes.c_ulong),
                    ("dwMemoryLoad",            ctypes.c_ulong),
                    ("ullTotalPhys",            ctypes.c_ulonglong),
                    ("ullAvailPhys",            ctypes.c_ulonglong),
                    ("ullTotalPageFile",        ctypes.c_ulonglong),
                    ("ullAvailPageFile",        ctypes.c_ulonglong),
                    ("ullTotalVirtual",         ctypes.c_ulonglong),
                    ("ullAvailVirtual",         ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual",ctypes.c_ulonglong),
                ]
            stat = _MEMSTAT()
            stat.dwLength = ctypes.sizeof(_MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total = int(stat.ullTotalPhys // _MIB)
            avail = int(stat.ullAvailPhys // _MIB)
            _RAM_TOTAL_CACHE = total
            return (total or None, avail or None)
        # Linux fallback — lê MemTotal e MemAvailable do /proc/meminfo
        total: int | None = None
        avail: int | None = None
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) // 1024
                if total and avail:
                    break
        if total:
            _RAM_TOTAL_CACHE = total
        return (total, avail)
    except Exception:
        return (_RAM_TOTAL_CACHE or None, None)


def ram_total_mib() -> int | None:
    return _ram_mib_pair()[0]


def ram_avail_mib() -> int | None:
    return _ram_mib_pair()[1]


# Lock de clock da VRAM (3090 Ti rebaixa P2->P3 na inferência, -35% t/s).
# O lock/reset exige admin; o backend roda sem elevação, então o toggle
# dispara tarefas agendadas SYSTEM pré-registradas com SDDL que permite
# Authenticated Users iniciá-las. Setup: scripts/setup_vram_lock_tasks.ps1.
VRAM_LOCK_MHZ = 10251
_VRAM_LOCK_TASK = {True: "VRAMLockOn", False: "VRAMLockOff"}
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _vram_lock_tasks_available() -> bool:
    if sys.platform != "win32":
        return False
    try:
        r = subprocess.run(
            ["schtasks", "/Query", "/TN", _VRAM_LOCK_TASK[True]],
            capture_output=True, timeout=10, creationflags=_NO_WINDOW,
        )
        return r.returncode == 0
    except Exception:
        return False


def vram_lock_status() -> dict:
    """Estado do lock de VRAM. `locked` é heurístico: clocks.mem >= 10251.

    Sob carga pesada a GPU pode boostar a 10251 mesmo sem lock, mas em idle
    (caso típico ao olhar a página) sem lock o clock fica em 405/5001.
    """
    smi = nvidia_smi_path()
    mem_mhz: int | None = None
    pstate: str | None = None
    try:
        r = subprocess.run(
            [smi, "--query-gpu=clocks.mem,pstate", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", timeout=5,
            creationflags=_NO_WINDOW,
        )
        if r.returncode == 0 and r.stdout.strip():
            mem_s, pstate = (x.strip() for x in r.stdout.strip().splitlines()[0].split(","))
            mem_mhz = int(mem_s)
    except Exception:
        pass
    return {
        "supported": _vram_lock_tasks_available(),
        "locked": (mem_mhz >= VRAM_LOCK_MHZ) if mem_mhz is not None else None,
        "mem_clock_mhz": mem_mhz,
        "pstate": pstate,
        "lock_mhz": VRAM_LOCK_MHZ,
    }


def set_vram_lock(enabled: bool) -> dict:
    """Liga/desliga o lock disparando a tarefa agendada correspondente."""
    task = _VRAM_LOCK_TASK[enabled]
    try:
        r = subprocess.run(
            ["schtasks", "/Run", "/TN", task],
            capture_output=True, text=True, timeout=15, creationflags=_NO_WINDOW,
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        return {
            "ok": False,
            "error": f"schtasks /Run {task} falhou: {err}. "
                     "Rode scripts/setup_vram_lock_tasks.ps1 como admin.",
        }
    # schtasks /Run é assíncrono e o P-state demora a transicionar (unlock
    # leva ~5 s pra cair de 10251 → 405). Polla até o estado refletir o
    # pedido; se não refletir em 10 s devolve o que tiver (a tarefa rodou).
    status = vram_lock_status()
    deadline = time.time() + 10.0
    while status.get("locked") is not enabled and time.time() < deadline:
        time.sleep(1.0)
        status = vram_lock_status()
    return {"ok": True, **status}


def nvidia_status() -> dict:
    """Status detalhado da GPU (todos os sensores em NVIDIA_QUERY_FIELDS)."""
    smi = nvidia_smi_path()
    try:
        r = subprocess.run(
            [smi, f"--query-gpu={','.join(NVIDIA_QUERY_FIELDS)}", "--format=csv,noheader"],
            capture_output=True, text=True, encoding="utf-8", timeout=15,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return {"available": False, "error": f"nvidia-smi rc={r.returncode}"}
        gpus = []
        for line in r.stdout.strip().splitlines():
            vals = [v.strip() for v in line.split(",")]
            gpus.append(dict(zip(NVIDIA_QUERY_FIELDS, vals)))
        return {"available": True, "gpus": gpus}
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi não encontrado"}
    except Exception as e:
        return {"available": False, "error": str(e)}
