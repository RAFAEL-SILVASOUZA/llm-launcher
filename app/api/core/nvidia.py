"""nvidia-smi: VRAM (total/livre/usada), status detalhado e RAM total do sistema."""
import ctypes
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
    "name", "driver_version", "uuid", "pci.bus_id",
    "temperature.gpu", "temperature.memory", "temperature.gpu.tlimit", "fan.speed",
    "utilization.gpu", "utilization.memory",
    "memory.total", "memory.used", "memory.free",
    "power.draw", "power.limit",
    "clocks.sm", "clocks.mem",
    "pstate",
]

# uuid -> {gpu_max, slowdown, shutdown} em °C. As GeForce de consumidor não
# expõem `temperature.gpu.tlimit` via nvidia-smi (vem "[N/A]"), mas o NVML
# entrega os thresholds térmicos por nvmlDeviceGetTemperatureThreshold. São
# fixos da placa, então cacheamos pra vida toda do processo.
_THRESHOLD_CACHE: dict[str, dict[str, int]] | None = None

# NVML temperature threshold ids (nvml.h: nvmlTemperatureThresholds_t)
_NVML_THR_SHUTDOWN = 0
_NVML_THR_SLOWDOWN = 1
_NVML_THR_GPU_MAX = 3


def _nvml_thresholds() -> dict[str, dict[str, int]]:
    """Lê os limites térmicos por GPU via NVML (ctypes), indexados por UUID.

    Devolve {} se a nvml.dll não carregar ou a placa não suportar — o caller
    cai no N/A original. MEM_MAX é NOT_SUPPORTED em GeForce, então só voltam
    gpu_max/slowdown/shutdown.
    """
    global _THRESHOLD_CACHE
    if _THRESHOLD_CACHE is not None:
        return _THRESHOLD_CACHE
    out: dict[str, dict[str, int]] = {}
    try:
        import ctypes
        nvml = ctypes.CDLL("nvml.dll")
        if nvml.nvmlInit_v2() != 0:
            _THRESHOLD_CACHE = out
            return out
        try:
            count = ctypes.c_uint()
            nvml.nvmlDeviceGetCount_v2(ctypes.byref(count))
            for i in range(count.value):
                h = ctypes.c_void_p()
                if nvml.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) != 0:
                    continue
                uuid_buf = ctypes.create_string_buffer(96)
                if nvml.nvmlDeviceGetUUID(h, uuid_buf, 96) != 0:
                    continue
                thr: dict[str, int] = {}
                for tid, key in (
                    (_NVML_THR_GPU_MAX, "gpu_max"),
                    (_NVML_THR_SLOWDOWN, "slowdown"),
                    (_NVML_THR_SHUTDOWN, "shutdown"),
                ):
                    val = ctypes.c_uint()
                    if nvml.nvmlDeviceGetTemperatureThreshold(h, tid, ctypes.byref(val)) == 0:
                        thr[key] = val.value
                if thr:
                    out[uuid_buf.value.decode()] = thr
        finally:
            nvml.nvmlShutdown()
    except Exception:
        pass
    _THRESHOLD_CACHE = out
    return out


# --- NVAPI: Hot Spot + Memory Junction (GDDR6X) -----------------------------
# nvidia-smi e NVML não expõem a temperatura de junção da memória nem o hot
# spot nas GeForce de consumidor (vêm N/A / NOT_SUPPORTED). Esses sensores só
# saem por uma função NÃO-DOCUMENTADA do NVAPI — a mesma que HWiNFO/GPU-Z/
# LibreHardwareMonitor usam. Replicamos o método do LibreHardwareMonitor:
# QueryInterface(0x65FE3AAD) -> NvAPI_GPU_GetThermalSensors, lê Temperatures[]
# (fixed-point /256). Índices validados contra o HWMonitor nesta rig (Ampere):
# Hot Spot = [1], Memory Junction = [9]. É frágil por natureza (offsets podem
# mudar em drivers futuros) — qualquer erro cai silenciosamente em N/A.
_NVAPI_THERMAL_VER = 168 | (2 << 16)  # sizeof(NvThermalSensors)=168, version 2
_NVAPI: dict | bool | None = None     # None=não tentado, False=indisponível
_NVAPI_MASK_CACHE: dict[int, int] = {}


class _NvThermalSensors(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("Version", ctypes.c_uint),
        ("Mask", ctypes.c_uint),
        ("Reserved", ctypes.c_int * 8),
        ("Temperatures", ctypes.c_int * 32),
    ]


def _nvapi_load() -> dict | bool:
    """Carrega nvapi64.dll e resolve as funções undocumented uma vez. Devolve
    dict de callables ou False se indisponível (não-Windows, dll ausente etc.)."""
    global _NVAPI
    if _NVAPI is not None:
        return _NVAPI
    if sys.platform != "win32":
        _NVAPI = False
        return _NVAPI
    try:
        nvapi = ctypes.WinDLL("nvapi64.dll")
        query = nvapi.nvapi_QueryInterface
        query.restype = ctypes.c_void_p
        query.argtypes = [ctypes.c_uint]

        def fn(off: int, restype, *argtypes):
            p = query(off)
            if not p:
                raise RuntimeError(f"QueryInterface({off:#x}) NULL")
            return ctypes.CFUNCTYPE(restype, *argtypes)(p)

        init = fn(0x0150E828, ctypes.c_int)  # NvAPI_Initialize
        if init() != 0:
            _NVAPI = False
            return _NVAPI
        _NVAPI = {
            "enum": fn(0xE5AC921F, ctypes.c_int,           # EnumPhysicalGPUs
                       ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int)),
            "therm": fn(0x65FE3AAD, ctypes.c_int,          # GetThermalSensors
                        ctypes.c_void_p, ctypes.POINTER(_NvThermalSensors)),
            "bus": fn(0x1BE0B8E5, ctypes.c_int,            # GetBusId
                      ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)),
        }
    except Exception:
        _NVAPI = False
    return _NVAPI


def _nvapi_thermals() -> dict[int, list[float]]:
    """busId(int) -> Temperatures[0..11] em °C (/256), lidos ao vivo via NVAPI.

    {} se indisponível. A máscara de sensores suportados é fixa da placa, então
    cacheamos por busId; só as temperaturas são relidas a cada chamada.
    """
    api = _nvapi_load()
    if not api:
        return {}
    out: dict[int, list[float]] = {}
    try:
        handles = (ctypes.c_void_p * 64)()
        count = ctypes.c_int()
        if api["enum"](handles, ctypes.byref(count)) != 0:
            return {}
        for i in range(count.value):
            h = handles[i]
            bus = ctypes.c_uint()
            if api["bus"](h, ctypes.byref(bus)) != 0:
                continue
            bid = bus.value
            mask = _NVAPI_MASK_CACHE.get(bid)
            if mask is None:
                mask = 0
                for bit in range(32):
                    ts = _NvThermalSensors(Version=_NVAPI_THERMAL_VER, Mask=(1 << bit))
                    if api["therm"](h, ctypes.byref(ts)) == 0:
                        mask = (1 << (bit + 1)) - 1
                    else:
                        break
                _NVAPI_MASK_CACHE[bid] = mask
            if not mask:
                continue
            ts = _NvThermalSensors(Version=_NVAPI_THERMAL_VER, Mask=mask)
            if api["therm"](h, ctypes.byref(ts)) == 0:
                out[bid] = [ts.Temperatures[j] / 256.0 for j in range(12)]
    except Exception:
        return out
    return out


def _nvapi_map(name: str, temps: list[float]) -> tuple[float | None, float | None]:
    """(hot_spot, memory_junction) a partir do array de temperaturas, escolhendo
    os índices por arquitetura — mesmo switch do LibreHardwareMonitor."""
    n = name.upper()
    try:
        if "RTX 50" in n:
            return (None, temps[2])          # Blackwell: hotspot não exposto
        if "RTX 40" in n:
            return (temps[1], temps[7])       # Ada
        return (temps[1], temps[9])           # Ampere e anteriores
    except IndexError:
        return (None, None)


def _pci_bus_num(bus_id: str) -> int | None:
    """'00000000:01:00.0' (nvidia-smi) -> 1 (busId do NVAPI). None se não parsear."""
    parts = bus_id.split(":")
    if len(parts) >= 2:
        try:
            return int(parts[1], 16)
        except ValueError:
            return None
    return None


def host_temperature_c() -> float | None:
    """Temperatura do host (CPU) via psutil, se disponível no Windows."""
    try:
        import psutil
        temps = psutil.sensors_temperatures()
        if not temps:
            return None
        # Tenta achar sensor da CPU primeiro; senão pega o primeiro valor
        for key in ("coretemp", "k10temp", "acpitz", "cpu_thermal"):
            if key in temps and temps[key]:
                return temps[key][0].current
        # Fallback: primeiro sensor disponível
        for entry in temps.values():
            if entry:
                return entry[0].current
    except Exception:
        pass
    return None


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
        thresholds = _nvml_thresholds()
        nvapi_t = _nvapi_thermals()
        gpus = []
        for line in r.stdout.strip().splitlines():
            vals = [v.strip() for v in line.split(",")]
            gpu = dict(zip(NVIDIA_QUERY_FIELDS, vals))
            # Substitui o tlimit "[N/A]" do nvidia-smi pelo limite térmico real
            # do NVML. Expomos a temperatura de throttle (gpu_max) — concreta e
            # intuitiva — junto da folga atual até ela.
            thr = thresholds.get(gpu.get("uuid", ""))
            if thr and "gpu_max" in thr:
                gpu["temperature.gpu.limit"] = str(thr["gpu_max"])
                try:
                    margin = thr["gpu_max"] - int(float(gpu.get("temperature.gpu", "")))
                    gpu["temperature.gpu.tlimit"] = str(margin)
                except (ValueError, TypeError):
                    pass
            # Hot Spot + Memory Junction via NVAPI (preenche o N/A do nvidia-smi).
            busid = _pci_bus_num(gpu.get("pci.bus_id", ""))
            temps = nvapi_t.get(busid) if busid is not None else None
            if temps:
                hot, mem = _nvapi_map(gpu.get("name", ""), temps)
                if mem and mem > 0:
                    gpu["temperature.memory"] = f"{mem:.0f}"
                if hot and hot > 0:
                    gpu["temperature.hotspot"] = f"{hot:.0f}"
            gpus.append(gpu)
        return {
            "available": True,
            "gpus": gpus,
            "host_temp_c": host_temperature_c(),
        }
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi não encontrado"}
    except Exception as e:
        return {"available": False, "error": str(e)}
