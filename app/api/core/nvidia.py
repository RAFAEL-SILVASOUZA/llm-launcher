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
