"""Estimador de VRAM/RAM: portado direto de estimate_memory em models.py.

Centro nervoso do launcher — combina metadados do GGUF, tipo de KV cache,
política de offload do llama.cpp e particularidades de cada arch (SSM,
MoE, SWA, MTP) pra prever quanto vai pra GPU e quanto pra CPU.
"""
from pathlib import Path

from .backends import kv_unified_active, remap_kv_for_backend
from .gguf import read_meta
from .models_repo import find_mtp_draft, is_mtp_model, model_disk_size_mib
from .nvidia import gpu_free_mib, gpu_total_mib, ram_total_mib

_MIB = 1024 * 1024

KV_BYTES_PER_ELEM = {
    "f32":    4.0,
    "f16":    2.0,
    "bf16":   2.0,
    "q8_0":   1.0625,
    "q5_1":   0.75,
    "q5_0":   0.6875,
    "q4_1":   0.625,
    "iq4_nl": 0.5625,
    "q4_0":   0.5625,
    "turbo2": 0.30,
    "turbo3": 0.40,
    "turbo4": 0.55,
}

_GPU_COMPUTE_BASE_MIB     = 450
_GPU_COMPUTE_KV_COEF      = 176
_GPU_COMPUTE_PARALLEL_MIB = 220
_GPU_COMPUTE_SSM_BASE_MIB = 50
_GPU_COMPUTE_SSM_PER_32K  = 240
_MTP_DRAFT_COMPUTE_MIB    = 800
_MMPROJ_COMPUTE_MIB       = 250
_WEIGHT_OVERHEAD          = 1.02


def _kv_cache_mib(meta: dict, ctx: int, streams: int,
                  bytes_pe_k: float, bytes_pe_v: float) -> int:
    """MiB de KV. `streams` = cópias do contexto: 1 com pool unificado (-kvu),
    senão uma por slot -np (ver kv_unified_active)."""
    n_layer = meta["n_layer"]
    nk      = meta["n_head_kv"]
    n_kv_per_layer: list[int] = nk if isinstance(nk, list) else [int(nk)] * n_layer
    if len(n_kv_per_layer) < n_layer:
        n_kv_per_layer = n_kv_per_layer + [n_kv_per_layer[-1]] * (n_layer - len(n_kv_per_layer))

    head_dim_k     = meta["head_dim_k"]
    head_dim_v     = meta["head_dim_v"]
    head_dim_k_swa = meta.get("head_dim_k_swa") or head_dim_k
    head_dim_v_swa = meta.get("head_dim_v_swa") or head_dim_v
    swa_window     = meta.get("swa_window")
    swa_pattern    = meta.get("swa_pattern") or []

    attn_layers = meta.get("attn_layers")
    attn_set: set[int] | None = set(attn_layers) if attn_layers else None

    total_bytes = 0.0
    for i in range(n_layer):
        if attn_set is not None and i not in attn_set:
            continue
        is_swa = bool(swa_pattern[i]) if (swa_window and i < len(swa_pattern)) else False
        layer_ctx = min(swa_window, ctx) if is_swa else ctx
        h_k       = head_dim_k_swa if is_swa else head_dim_k
        h_v       = head_dim_v_swa if is_swa else head_dim_v
        n_kv_i    = int(n_kv_per_layer[i])
        per_token = n_kv_i * (h_k * bytes_pe_k + h_v * bytes_pe_v)
        total_bytes += per_token * layer_ctx * streams
    return int(total_bytes / _MIB)


def _mtp_draft_kv_mib(meta: dict, ctx: int,
                      bytes_pe_k: float, bytes_pe_v: float) -> int:
    """KV do contexto MTP de rascunho: 1 layer full-attention sobre ctx inteiro."""
    nk = meta["n_head_kv"]
    n_kv = max(nk) if isinstance(nk, list) else int(nk)
    h_k  = meta["head_dim_k"]
    h_v  = meta["head_dim_v"]
    return int(n_kv * (h_k * bytes_pe_k + h_v * bytes_pe_v) * ctx / _MIB)


def _mtp_draft_weights_mib(model_path: Path) -> int:
    """Pesos da head MTP quando ela vem em gguf SEPARADO (-md).

    Head embutida já está contabilizada no arquivo do modelo — aqui é só o caso
    do drafter externo, que o llama.cpp carrega como um segundo modelo inteiro
    (token_embd + output + o bloco MTP; ~3 GB num 27B q8_0). Ignorar isso
    subestimava a VRAM e fazia o fit escolher um contexto que dava OOM.
    """
    draft = find_mtp_draft(model_path)
    if draft is None:
        return 0
    meta = read_meta(draft)
    if meta and meta.get("tensor_total_mib"):
        return int(meta["tensor_total_mib"])
    try:
        return int(draft.stat().st_size / _MIB)
    except OSError:
        return 0


def _split_weights_by_ngl(
    meta:           dict | None,
    weights_total:  int,
    gpu_layers:     int,
) -> tuple[int, int]:
    """Decide quantos MiB de pesos ficam em VRAM vs CPU, dado -ngl K.

    Política do llama.cpp:
      K == 0          → 0 na GPU
      1..n_layer      → output + (K-1) blocos na GPU; resto + token_embd na CPU
      n_layer + 1     → output + todos os blocos na GPU; token_embd na CPU
      K >= n_layer+2  → tudo na GPU
    """
    if not meta or not meta.get("n_layer"):
        return weights_total, 0
    n_layer = meta["n_layer"]
    if gpu_layers <= 0:
        return 0, weights_total

    token_embd = meta.get("token_embd_mib", 0) or 0
    output_w   = meta.get("output_mib",      0) or 0
    blocks_t   = meta.get("blocks_mib",      0) or 0

    if token_embd <= 0 or blocks_t <= 0:
        frac = 1.0 if gpu_layers >= n_layer else gpu_layers / n_layer
        gpu = int(weights_total * frac)
        return gpu, weights_total - gpu

    tied = output_w <= 0

    tensor_total = max(1, token_embd + output_w + blocks_t)
    scale = weights_total / tensor_total
    per_block_blocks = (blocks_t * scale) / n_layer
    output_eff = output_w  * scale
    token_eff  = token_embd * scale

    gpu_blocks_count = (
        n_layer if gpu_layers >= n_layer + 1
        else max(0, gpu_layers - 1)
    )
    gpu_blocks = per_block_blocks * gpu_blocks_count
    if tied:
        gpu_output = 0.0
        gpu_token  = token_eff if gpu_layers >= 1 else 0.0
    else:
        gpu_output = output_eff if gpu_layers >= 1            else 0.0
        gpu_token  = token_eff  if gpu_layers >= n_layer + 2  else 0.0

    vram = int(gpu_blocks + gpu_output + gpu_token)
    vram = min(vram, weights_total)
    return vram, weights_total - vram


def estimate_memory(
    model_path:     Path,
    backend:        str,
    context_window: int,
    parallel_slots: int,
    kv_cache:       str,
    gpu_layers:     int,
    mmproj:         Path | None,
    cache_ram:      int,
    mode:           str,
    n_cpu_moe:      int = 0,
    kv_unified:     bool = True,
) -> dict:
    """Estima componentes de VRAM/RAM (em MiB) para a config corrente."""
    meta = read_meta(model_path)
    if meta and meta.get("tensor_total_mib"):
        weights_mib = int(meta["tensor_total_mib"] * _WEIGHT_OVERHEAD)
    else:
        weights_mib = int(model_disk_size_mib(model_path) * _WEIGHT_OVERHEAD)

    mmproj_weights_mib = int((mmproj.stat().st_size // _MIB) * _WEIGHT_OVERHEAD) if mmproj else 0
    mmproj_compute_mib = _MMPROJ_COMPUTE_MIB if mmproj else 0

    kv_eff, _    = remap_kv_for_backend(kv_cache, backend, mode)
    bytes_pe     = KV_BYTES_PER_ELEM.get(kv_eff, 2.0)
    kv_eff_k = kv_eff
    if backend == "turbo" and kv_eff in ("turbo2", "turbo3", "turbo4") and meta:
        nk = meta["n_head_kv"]
        n_kv_typ = max(nk) if isinstance(nk, list) else int(nk)
        if n_kv_typ and meta.get("n_head", 0) > n_kv_typ:
            kv_eff_k = "q8_0"
    bytes_pe_k   = KV_BYTES_PER_ELEM.get(kv_eff_k, bytes_pe)
    # Pool unificado (-kvu) = uma cópia do contexto pros -np slots; sem ele, uma
    # por slot. Tem que casar com o -c que o builder emite, senão a estimativa
    # erra por np×.
    kv_streams   = 1 if kv_unified_active(backend, kv_unified, mode) else max(1, parallel_slots)
    kv_total_mib = (
        _kv_cache_mib(meta, context_window, kv_streams, bytes_pe_k, bytes_pe)
        if meta else 0
    )

    # Estado recorrente do híbrido SSM continua sendo POR SEQUÊNCIA mesmo com
    # -kvu (llama-model.cpp passa recurrent_rs_size = n_seq_max independente de
    # unified) — o pool compartilhado é só do KV de atenção.
    ssm_state_mib = 0
    if meta:
        per_ssm = meta.get("ssm_state_per_layer_mib", 0.0)
        attn_n  = len(meta.get("attn_layers") or [])
        if per_ssm and attn_n:
            n_ssm = max(0, meta["n_layer"] - attn_n)
            ssm_state_mib = int(per_ssm * n_ssm * max(1, parallel_slots))

    n_layer  = meta["n_layer"] if meta else 0
    vram_weights, ram_weights = _split_weights_by_ngl(meta, weights_mib, gpu_layers)
    gpu_active = gpu_layers > 0
    if not n_layer or gpu_layers >= n_layer:
        gpu_frac_kv = 1.0
    else:
        gpu_frac_kv = max(0, gpu_layers) / n_layer

    vram_kv  = int(kv_total_mib * gpu_frac_kv)
    vram_ssm = int(ssm_state_mib * gpu_frac_kv)

    is_ssm_hybrid = bool(meta and meta.get("ssm_state_per_layer_mib", 0))
    if not gpu_active:
        vram_compute = 0
    elif is_ssm_hybrid:
        vram_compute = int(
            _GPU_COMPUTE_SSM_BASE_MIB
            + _GPU_COMPUTE_SSM_PER_32K * (context_window / 32768)
            + _GPU_COMPUTE_PARALLEL_MIB * max(0, parallel_slots - 1)
        )
    else:
        vram_compute = int(
            _GPU_COMPUTE_BASE_MIB
            + _GPU_COMPUTE_KV_COEF * bytes_pe
            + _GPU_COMPUTE_PARALLEL_MIB * max(0, parallel_slots - 1)
        )

    vram_mmproj_weights = mmproj_weights_mib if gpu_active else 0
    vram_mmproj_compute = mmproj_compute_mib if gpu_active else 0
    vram_mmproj         = vram_mmproj_weights + vram_mmproj_compute

    vram_mtp_kv      = 0
    vram_mtp_compute = 0
    vram_mtp_weights = 0
    if (meta and gpu_active and backend == "mtp"
            and is_mtp_model(model_path)):
        vram_mtp_kv      = _mtp_draft_kv_mib(meta, context_window, bytes_pe_k, bytes_pe)
        vram_mtp_compute = _MTP_DRAFT_COMPUTE_MIB
        vram_mtp_weights = _mtp_draft_weights_mib(model_path)

    ram_kv  = kv_total_mib - vram_kv
    ram_ssm = ssm_state_mib - vram_ssm
    cache_ram_mib = cache_ram if mode == "server" else 0

    moe_offload_mib = 0
    if meta and meta.get("is_moe") and n_cpu_moe > 0:
        moe_layers = meta.get("moe_layers_count", 0)
        per_layer  = meta.get("moe_per_layer_mib", 0)
        n_off      = max(0, min(int(n_cpu_moe), moe_layers))
        moe_offload_mib = int(per_layer * n_off * _WEIGHT_OVERHEAD)
        moe_offload_mib = min(moe_offload_mib, vram_weights)
        vram_weights -= moe_offload_mib
        ram_weights  += moe_offload_mib

    return {
        "meta_ok":          meta is not None,
        "weights_mib":      weights_mib,
        "kv_total_mib":     kv_total_mib,
        "mmproj_mib":       mmproj_weights_mib + mmproj_compute_mib,
        "vram_weights":     vram_weights,
        "vram_kv":          vram_kv,
        "vram_compute":     vram_compute,
        "vram_ssm":         vram_ssm,
        "vram_mmproj":          vram_mmproj,
        "vram_mmproj_weights":  vram_mmproj_weights,
        "vram_mmproj_compute":  vram_mmproj_compute,
        "vram_mtp_kv":      vram_mtp_kv,
        "vram_mtp_compute": vram_mtp_compute,
        "vram_mtp_weights": vram_mtp_weights,
        "ram_weights":      ram_weights,
        "ram_kv":           ram_kv,
        "ram_ssm":          ram_ssm,
        "cache_ram":        cache_ram_mib,
        "moe_offload_mib":  moe_offload_mib,
        "vram_total":       (vram_weights + vram_kv + vram_compute + vram_ssm
                             + vram_mmproj + vram_mtp_kv + vram_mtp_compute
                             + vram_mtp_weights),
        "ram_total":        ram_weights + ram_kv + ram_ssm,
        "vram_avail":       gpu_free_mib(),
        "vram_total_phys":  gpu_total_mib(),
        "ram_avail":        ram_total_mib(),
    }


def estimate_status(est: dict) -> str:
    """'ok' / 'tight' / 'overflow' baseado no pior eixo (VRAM ou RAM)."""
    pcts = []
    for total, avail in (
        (est["vram_total"], est["vram_avail"]),
        (est["ram_total"],  est["ram_avail"]),
    ):
        if avail and avail > 0:
            pcts.append(total / avail)
    if not pcts:
        return "unknown"
    worst = max(pcts)
    if worst >= 0.95: return "overflow"
    if worst >= 0.80: return "tight"
    return "ok"


def suggest_n_cpu_moe(
    model_path:     Path,
    backend:        str,
    context_window: int,
    parallel_slots: int,
    kv_cache:       str,
    gpu_layers:     int,
    mmproj:         Path | None,
    cache_ram:      int,
    mode:           str,
    target_vram_pct: float = 0.88,
) -> int:
    meta = read_meta(model_path)
    if not meta or not meta.get("is_moe"):
        return 0
    moe_layers = meta.get("moe_layers_count", 0)
    if moe_layers <= 0:
        return 0
    # VRAM total × reserva, não a livre (vide suggest_n_gpu_layers): evita
    # mandar peso MoE pra CPU por causa de VRAM presa transitoriamente.
    vram_avail = gpu_total_mib()
    if not vram_avail:
        return 0
    budget = int(vram_avail * target_vram_pct)
    for n in range(0, moe_layers + 1):
        est = estimate_memory(
            model_path, backend, context_window, parallel_slots,
            kv_cache, gpu_layers, mmproj, cache_ram, mode,
            n_cpu_moe=n,
        )
        if est["vram_total"] <= budget:
            return n
    return moe_layers


def suggest_n_gpu_layers(
    model_path:      Path,
    backend:         str,
    context_window:  int,
    parallel_slots:  int,
    kv_cache:        str,
    mmproj:          Path | None,
    cache_ram:       int,
    mode:            str,
    target_vram_pct: float = 0.92,
) -> int:
    meta = read_meta(model_path)
    if not meta:
        return 99
    n_layer = meta["n_layer"]
    # Orça contra a VRAM TOTAL (× reserva), não a livre: o modelo vai ocupar a
    # GPU só depois do server subir, e a app descarrega o anterior antes. Usar a
    # livre no instante (desktop/browser/outro server) encolhe o budget e empurra
    # camadas pra CPU silenciosamente — lentidão sem aviso. OOM real → auto-degrade.
    vram_avail = gpu_total_mib()
    if not vram_avail:
        return 99
    budget = int(vram_avail * target_vram_pct)
    for ngl in range(n_layer + 2, -1, -1):
        est = estimate_memory(
            model_path, backend, context_window, parallel_slots,
            kv_cache, ngl, mmproj, cache_ram, mode,
        )
        if est["vram_total"] <= budget:
            return ngl
    return 0
