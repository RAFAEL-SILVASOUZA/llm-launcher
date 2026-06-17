"""nvidia-smi: VRAM (total/livre/usada), status detalhado e RAM total do sistema."""
import os
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
    """(total, livre) em MiB SOMADOS sobre todas as GPUs. (None, None) se indisp.

    Com 2+ placas e o split-mode default do llama.cpp (layer), os pesos e o KV
    são repartidos entre as GPUs, então o orçamento relevante é a soma das VRAMs.
    Antes lia só `splitlines()[0]` (a GPU 0) e subestimava pela metade numa rig
    de 2 placas. A soma é aproximação de 1ª ordem: presume split proporcional à
    capacidade (default `-ts`); uma placa pode encher antes da outra em splits
    manuais desbalanceados — aí o auto-degrade/OOM real é a rede de segurança.

    Cache com TTL curto (_VRAM_TTL_SECS) — múltiplas chamadas dentro de uma
    request HTTP reusam o resultado, mas o próximo poll do header pega
    valores atualizados.
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
            total = free = 0
            for line in r.stdout.strip().splitlines():
                total_s, free_s = (x.strip() for x in line.split(","))
                total += int(total_s)
                free  += int(free_s)
            _VRAM_INFO_CACHE = (now, (total, free))
            return _VRAM_INFO_CACHE[1]
    except Exception:
        pass
    _VRAM_INFO_CACHE = (now, (0, 0))
    return (None, None)


def gpu_count() -> int:
    """Número de GPUs NVIDIA visíveis (0 se nvidia-smi indisponível)."""
    smi = nvidia_smi_path()
    try:
        r = subprocess.run(
            [smi, "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, encoding="utf-8", timeout=5,
        )
        if r.returncode == 0:
            return sum(1 for ln in r.stdout.strip().splitlines() if ln.strip())
    except Exception:
        pass
    return 0


def gpu_order_ti_first() -> list[str]:
    """UUIDs das GPUs com a 3090 Ti na frente (ela vira o device CUDA 0).

    Sem isso o CUDA enumera por slot PCIe: a 3090 (bus 01) vira device 0 e a
    3090 Ti (bus 04) device 1. Ordenamos por nome ("Ti" primeiro) e devolvemos
    UUIDs — usar UUID em CUDA_VISIBLE_DEVICES torna a escolha independente de
    qualquer heurística do driver (FASTEST_FIRST etc.). Lista vazia se indisp.
    """
    smi = nvidia_smi_path()
    try:
        r = subprocess.run(
            [smi, "--query-gpu=name,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, encoding="utf-8", timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            gpus = []
            for ln in r.stdout.strip().splitlines():
                if "," not in ln:
                    continue
                name, uuid = (x.strip() for x in ln.split(",", 1))
                gpus.append((name, uuid))
            gpus.sort(key=lambda g: 0 if "ti" in g[0].lower() else 1)
            return [u for _, u in gpus]
    except Exception:
        pass
    return []


def cuda_env() -> dict:
    """os.environ com a 3090 Ti pinada como GPU principal (device CUDA 0).

    Para passar a Popen do llama-server (env=...). Com a Ti em device 0, o
    default -mg 0 do llama.cpp faz dela a main GPU (KV/escalares) e -sm none
    roda só nela. Não altera o env do processo atual — só a cópia retornada.
    """
    env = dict(os.environ)
    order = gpu_order_ti_first()
    if order:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(order)
    return env


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
