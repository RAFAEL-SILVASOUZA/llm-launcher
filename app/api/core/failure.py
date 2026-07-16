"""Classificação de falhas do llama-server e escada de auto-degrade.

Cada queda do servidor é classificada por categoria (OOM_LOAD, OOM_RUNTIME,
KV_TYPE_REJECTED, UNSUPPORTED_FLAG, MMPROJ, MODEL_CORRUPTED, UNKNOWN), e
next_degrade aplica o próximo passo na escada — preservando velocidade/
qualidade até onde der, sem mexer no context window (decisão do usuário).
"""
import re

from .backends import kv_unified_active, next_kv_tier, remap_kv_for_backend
from .constants import (
    KV_REJECTED_PATTERNS,
    MMPROJ_ERROR_PATTERNS,
    MODEL_CORRUPTED_PATTERNS,
    OOM_PATTERNS,
    UNSUPPORTED_FLAG_PATTERNS,
)


class Failure:
    OOM_LOAD         = "OOM_LOAD"
    OOM_RUNTIME      = "OOM_RUNTIME"
    UNSUPPORTED_FLAG = "UNSUPPORTED_FLAG"
    KV_TYPE_REJECTED = "KV_TYPE_REJECTED"
    MMPROJ           = "MMPROJ"
    MODEL_CORRUPTED  = "MODEL_CORRUPTED"
    UNKNOWN          = "UNKNOWN"


def has_mmproj_error(output: str) -> bool:
    low = output.lower()
    return any(p in low for p in MMPROJ_ERROR_PATTERNS)


def classify_failure(output: str, rc: int, load_ts: float | None) -> tuple[str, str]:
    """Devolve (categoria, trecho_do_erro). O trecho serve de evidência."""
    low = output.lower()
    lines = output.splitlines()

    def first_match(patterns: tuple[str, ...]) -> str:
        for ln in lines[-200:]:
            l = ln.lower()
            for p in patterns:
                if p in l:
                    return ln.strip()
        return ""

    # MODEL_CORRUPTED antes de tudo: se o arquivo tá quebrado, mmproj/kv/oom
    # podem aparecer no mesmo log como sintomas secundários — não confundir.
    if any(p in low for p in MODEL_CORRUPTED_PATTERNS):
        return Failure.MODEL_CORRUPTED, first_match(MODEL_CORRUPTED_PATTERNS)
    if any(p in low for p in MMPROJ_ERROR_PATTERNS):
        return Failure.MMPROJ, first_match(MMPROJ_ERROR_PATTERNS)
    if any(p in low for p in KV_REJECTED_PATTERNS):
        return Failure.KV_TYPE_REJECTED, first_match(KV_REJECTED_PATTERNS)
    if any(p in low for p in UNSUPPORTED_FLAG_PATTERNS):
        return Failure.UNSUPPORTED_FLAG, first_match(UNSUPPORTED_FLAG_PATTERNS)
    if any(p in low for p in OOM_PATTERNS):
        cat = Failure.OOM_RUNTIME if load_ts else Failure.OOM_LOAD
        return cat, first_match(OOM_PATTERNS)
    if load_ts is not None:
        return Failure.OOM_RUNTIME, f"crash silencioso após load (rc={rc})"
    return Failure.UNKNOWN, f"sem assinatura conhecida (rc={rc})"


def _extract_unknown_flag(error_excerpt: str) -> str | None:
    m = re.search(r"(?:unknown|unrecognized|invalid)\s+argument:\s*(\S+)", error_excerpt, re.IGNORECASE)
    return m.group(1) if m else None


def _parallel_reason(cfg: dict, np_: int, new_np: int) -> str:
    """Cortar slots só derruba KV quando cada slot tem o seu contexto."""
    if kv_unified_active(cfg.get("backend", "turbo"), cfg.get("kv_unified", True)):
        return f"parallel {np_} → {new_np} (libera estado SSM + compute; KV é pool único, não muda)"
    return f"parallel {np_} → {new_np} (KV escala linear com slots)"


def next_degrade(cfg: dict, failure: str, error_excerpt: str) -> tuple[dict | None, str | None]:
    """Aplica UM degrau na config. Devolve (nova_cfg, descrição) ou (None, None).

    Nunca mexe em context_window — quem chama precisa lidar com isso à parte.
    """
    cfg = dict(cfg)
    backend = cfg.get("backend", "turbo")

    if failure == Failure.UNSUPPORTED_FLAG:
        flag = _extract_unknown_flag(error_excerpt)
        if flag in ("--reasoning-budget", "--reasoning") and cfg.get("reasoning_budget") is not None:
            cfg["reasoning_budget"] = None
            return cfg, "removido flag de reasoning (build não suporta)"
        if flag == "--n-cpu-moe" and cfg.get("n_cpu_moe", 0):
            cfg["n_cpu_moe"] = 0
            return cfg, "removido --n-cpu-moe (build não suporta)"
        if flag == "--ctx-checkpoints" and cfg.get("ctx_checkpoints", 0):
            cfg["ctx_checkpoints"] = 0
            return cfg, "removido --ctx-checkpoints (build não suporta)"
        if flag == "--chat-template-kwargs" and cfg.get("preserve_thinking"):
            cfg["preserve_thinking"] = False
            return cfg, "removido --chat-template-kwargs (build não suporta)"
        # Sem -kvu o builder volta a multiplicar o -c pelos slots, senão o
        # llama.cpp repartiria o contexto e cortaria a janela em silêncio. O
        # preço é KV × slots — se não couber, a escada continua daqui.
        if flag in ("--kv-unified", "-kvu") and cfg.get("kv_unified", True):
            cfg["kv_unified"] = False
            return cfg, "removido --kv-unified (build não suporta) — KV volta a ser × slots"

    if failure == Failure.KV_TYPE_REJECTED:
        new_kv, _ = remap_kv_for_backend(cfg.get("kv_cache", "f16"), backend, "server")
        if new_kv != cfg.get("kv_cache"):
            old = cfg["kv_cache"]
            cfg["kv_cache"] = new_kv
            return cfg, f"KV {old} → {new_kv} (não aceito por '{backend}')"

    if failure == Failure.OOM_RUNTIME:
        ub = cfg.get("ubatch_size", 512)
        if ub > 128:
            new_ub = max(128, ub // 2)
            cfg["ubatch_size"] = new_ub
            return cfg, f"ubatch {ub} → {new_ub} (corta pico de ativação)"
        np_ = cfg.get("parallel_slots", 1)
        if np_ > 1:
            new_np = max(1, np_ // 2)
            cfg["parallel_slots"] = new_np
            return cfg, _parallel_reason(cfg, np_, new_np)

    if not cfg.get("flash_attn", True):
        cfg["flash_attn"] = True
        return cfg, "flash-attention OFF → ON (reduz KV, custo zero em velocidade)"

    np_ = cfg.get("parallel_slots", 1)
    if np_ > 1:
        new_np = max(1, np_ // 2)
        cfg["parallel_slots"] = new_np
        return cfg, _parallel_reason(cfg, np_, new_np)

    cur_kv = cfg.get("kv_cache", "f16")
    nxt_kv = next_kv_tier(cur_kv, backend)
    if nxt_kv is not None:
        cfg["kv_cache"] = nxt_kv
        return cfg, f"KV {cur_kv} → {nxt_kv} (compressão maior)"

    cr = cfg.get("cache_ram", 8_192)
    if cr > 2_048:
        new_cr = max(2_048, cr // 2)
        cfg["cache_ram"] = new_cr
        return cfg, f"cache-ram {cr} → {new_cr} MiB"

    ub = cfg.get("ubatch_size", 512)
    if ub > 128:
        new_ub = max(128, ub // 2)
        cfg["ubatch_size"] = new_ub
        return cfg, f"ubatch {ub} → {new_ub}"

    b = cfg.get("batch_size", 2_048)
    if b > 512:
        new_b = max(512, b // 2)
        cfg["batch_size"] = new_b
        return cfg, f"batch {b} → {new_b}"

    ngl = cfg.get("gpu_layers", 99)
    if ngl > 0:
        step = max(1, ngl // 4) if ngl > 4 else 1
        new_ngl = max(0, ngl - step)
        cfg["gpu_layers"] = new_ngl
        return cfg, f"GPU layers {ngl} → {new_ngl} (ÚLTIMO RECURSO — perde velocidade)"

    return None, None
