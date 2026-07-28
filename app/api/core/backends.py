"""Helpers de backend (turbo/vanilla/mtp): paths, probe de --help, KV types.

Mantém duas tabelas paralelas (KV_TYPES_FALLBACK estático + cache em memória
populado via --help) pra detectar quais tipos de KV cache cada binário aceita,
e um remapeador que escolhe o fallback mais próximo quando o usuário pede um
tipo que o backend atual não suporta (ex: turbo* em vanilla).
"""
import re
import subprocess
from pathlib import Path

from . import settings_store
from .constants import BACKEND_BINARY_ALIAS, BACKENDS

_BACKEND_HELP_CACHE: dict[tuple[str, str], str] = {}
_KV_TYPES_CACHE: dict[tuple[str, str], set[str]] = {}


def invalidate_caches() -> None:
    """Chame quando os diretórios de backend mudarem em settings — o probe de
    --help e os KV types vêm do binário, então caches por (backend, mode)
    ficam stale se o usuário troca o build."""
    _BACKEND_HELP_CACHE.clear()
    _KV_TYPES_CACHE.clear()

KV_TYPES_FALLBACK: dict[str, set[str]] = {
    "turbo":   {"f32","f16","bf16","q8_0","q4_0","q4_1","iq4_nl","q5_0","q5_1","turbo2","turbo3","turbo4"},
    "vanilla": {"f32","f16","bf16","q8_0","q4_0","q4_1","iq4_nl","q5_0","q5_1"},
    "mtp":     {"f32","f16","bf16","q8_0","q4_0","q4_1","iq4_nl","q5_0","q5_1"},
    # custom: build desconhecido — o fallback é o conjunto do upstream, mas o
    # probe do --help do binário informado sobrescreve (é o caso normal).
    "custom":  {"f32","f16","bf16","q8_0","q4_0","q4_1","iq4_nl","q5_0","q5_1"},
}

KV_REMAP_FALLBACK: dict[str, list[str]] = {
    "turbo2": ["q4_0", "iq4_nl", "q4_1"],
    "turbo3": ["q4_0", "iq4_nl", "q4_1"],
    "turbo4": ["q5_0", "q5_1", "q8_0"],
    "bf16":   ["f16", "f32"],
    "f32":    ["f16", "bf16"],
    "iq4_nl": ["q4_0", "q4_1"],
    "q4_1":   ["q4_0", "iq4_nl"],
    "q5_1":   ["q5_0", "q8_0"],
}

# Ordem global do KV cache, do menor pro maior consumo de VRAM — usada pelo
# next_degrade pra escolher o próximo tier de compressão.
KV_TIER_ORDER = [
    "f32", "bf16", "f16", "q8_0", "q5_1", "q5_0",
    "iq4_nl", "q4_1", "q4_0", "turbo4", "turbo3", "turbo2",
]


def backend_binary(backend: str, mode: str) -> Path:
    """Caminho do binário pro par (backend, mode).

    Resolve a partir de settings: se o usuário configurou um diretório custom
    pro backend, usa esse + o nome do binário hardcoded (preserva
    `llama-server.exe`/`llama-cli.exe`). Caso contrário, default do constants.
    Backends-alias (mtp→vanilla) resolvem pelo binário do alvo.
    """
    target = BACKEND_BINARY_ALIAS.get(backend, backend)
    default = BACKENDS[target]["server" if mode == "server" else "cli"]
    custom_dir = settings_store.get_backend_dir(target)
    if custom_dir is not None:
        return custom_dir / default.name
    return default


def backend_help_text(backend: str, mode: str) -> str:
    """Saída de `--help`, cacheada em memória. String vazia se falhou."""
    ck = (backend, mode)
    if ck in _BACKEND_HELP_CACHE:
        return _BACKEND_HELP_CACHE[ck]

    text = ""
    bin_path = backend_binary(backend, mode)
    if bin_path.exists():
        try:
            r = subprocess.run(
                [str(bin_path), "--help"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=10,
            )
            text = (r.stdout or "") + (r.stderr or "")
        except Exception:
            text = ""
    _BACKEND_HELP_CACHE[ck] = text
    return text


def backend_supports_chat_template_kwargs(backend: str, mode: str = "server") -> bool:
    help_text = backend_help_text(backend, mode)
    if not help_text:
        return True  # sem probe, assume suporte — auto-degrade remove se quebrar
    return "--chat-template-kwargs" in help_text


def backend_supports_flag(flag: str, backend: str, mode: str = "server") -> bool:
    """Se o binário aceita `flag` (parse do --help, cacheado).

    Sem probe assume que sim: o auto-degrade remove a flag se o server recusar.
    Builds antigos do turboquant não têm --reasoning-budget-message, por exemplo.
    """
    help_text = backend_help_text(backend, mode)
    if not help_text:
        return True
    return flag in help_text


def kv_unified_active(backend: str, kv_unified: bool = True, mode: str = "server") -> bool:
    """Se o KV vai ser um pool ÚNICO compartilhado pelos slots -np.

    llama-context.cpp decide o contexto por sequência a partir disso:
        -kvu   → n_ctx_seq = n_ctx           (os -np slots dividem um pool só)
        sem    → n_ctx_seq = n_ctx / n_seq_max  (cada slot tem o seu, KV × np)

    builder e estimator PRECISAM concordar aqui: um decide se multiplica o -c
    pelos slots, o outro conta o KV. Chamar esta função é o que mantém os dois
    em sincronia — não checar a flag na mão.
    """
    return bool(kv_unified) and backend_supports_flag("--kv-unified", backend, mode)


def probe_kv_types(backend: str, mode: str = "server") -> set[str]:
    """Lê os tipos de --cache-type-k aceitos via parse do --help. Cacheado."""
    cache_key = (backend, mode)
    if cache_key in _KV_TYPES_CACHE:
        return _KV_TYPES_CACHE[cache_key]

    fallback = KV_TYPES_FALLBACK.get(backend, set(KV_TYPES_FALLBACK["vanilla"]))
    bin_path = backend_binary(backend, mode)
    if not bin_path.exists():
        _KV_TYPES_CACHE[cache_key] = fallback
        return fallback

    text = backend_help_text(backend, mode)
    if not text:
        _KV_TYPES_CACHE[cache_key] = fallback
        return fallback

    lines = text.splitlines()
    types: set[str] = set()
    in_block = False
    grabbing = False
    buf: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if not in_block:
            if "--cache-type-k" in line:
                in_block = True
            continue
        if grabbing and (stripped.startswith("(default:") or stripped.startswith("-")):
            break
        idx = line.find("allowed values:")
        if idx >= 0:
            grabbing = True
            buf.append(line[idx + len("allowed values:"):])
            continue
        if grabbing:
            if not stripped or stripped.startswith("(default:") or stripped.startswith("-"):
                break
            buf.append(line)
    if buf:
        joined = " ".join(buf)
        for tok in re.split(r"[,\s]+", joined):
            t = tok.strip().lower()
            if t and re.fullmatch(r"[a-z0-9_]+", t):
                types.add(t)

    result = types or fallback
    _KV_TYPES_CACHE[cache_key] = result
    return result


def remap_kv_for_backend(kv_cache: str, backend: str, mode: str = "server") -> tuple[str, str | None]:
    """Retorna (kv_efetivo, aviso). Se o backend não aceita o tipo, escolhe o
    fallback mais próximo em KV_REMAP_FALLBACK."""
    supported = probe_kv_types(backend, mode)
    if kv_cache in supported:
        return kv_cache, None
    for candidate in KV_REMAP_FALLBACK.get(kv_cache, []):
        if candidate in supported:
            return candidate, (
                f"Backend '{backend}' não aceita KV '{kv_cache}' — usando '{candidate}'."
            )
    return "f16", f"Backend '{backend}' não aceita KV '{kv_cache}' nem fallback — usando 'f16'."


def backend_status() -> list[dict]:
    """Estado de cada backend pro frontend: binários presentes, kv types."""
    out: list[dict] = []
    for name, b in BACKENDS.items():
        server_path = backend_binary(name, "server")
        cli_path    = backend_binary(name, "cli")
        default_dir = settings_store.default_backend_dir(name)
        custom_dir  = settings_store.get_backend_dir(name)
        out.append({
            "name":              name,
            "label":             b["label"],
            "description":       b["description"],
            "server_path":       str(server_path),
            "cli_path":          str(cli_path),
            "server_available":  server_path.exists(),
            "cli_available":     cli_path.exists(),
            "supports_spec_mtp": b["supports_spec_mtp"],
            "kv_types_server":   sorted(probe_kv_types(name, "server")) if server_path.exists() else [],
            "kv_types_cli":      sorted(probe_kv_types(name, "cli")) if cli_path.exists() else [],
            "default_dir":       str(default_dir),
            "configured_dir":    str(custom_dir) if custom_dir is not None else "",
        })
    return out


def next_kv_tier(current: str, backend: str) -> str | None:
    """Próximo KV mais econômico que o atual, filtrado pelo backend."""
    supported = probe_kv_types(backend, "server")
    try:
        idx = KV_TIER_ORDER.index(current)
    except ValueError:
        idx = -1
    for cand in KV_TIER_ORDER[idx + 1:]:
        if cand in supported:
            return cand
    return None
