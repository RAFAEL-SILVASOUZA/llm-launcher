import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import questionary

# Samplers vêm de app/api/core/sampling.py — fonte ÚNICA, compartilhada com o app.
# Duplicar a resolução aqui é exatamente como o preset de código acabou sendo
# servido pra modelo de raciocínio: duas cópias, uma só foi corrigida.
sys.path.insert(0, str(Path(__file__).parent / "app"))
from api.core import sampling as sampling_core  # noqa: E402

MODELS_DIR  = Path(r"D:\models")

# ─── backends de llama.cpp disponíveis ────────────────────────────────────────
# Cada backend tem seu binário próprio, capacidades distintas e features que só
# funcionam ali. Usuário escolhe no início e o setup se adapta.
LLAMA_SERVER_TURBO   = Path(r"C:\Projetos\llama-cpp-turboquant\build\bin\Release\llama-server.exe")
LLAMA_CLI_TURBO      = Path(r"C:\Projetos\llama-cpp-turboquant\build\bin\Release\llama-cli.exe")
LLAMA_SERVER_VANILLA = Path(r"C:\Projetos\llama-cpp\build\bin\Release\llama-server.exe")
LLAMA_CLI_VANILLA    = Path(r"C:\Projetos\llama-cpp\build\bin\Release\llama-cli.exe")
# MTP agora é nativo no upstream (PR #22673, via --spec-type draft-mtp), inclusive
# o fix qwen35 post-norm hidden state (#24025). O fork am17an/llama.cpp:mtp-clean
# ficou obsoleto — apontamos pro mesmo build vanilla, que já carrega a head MTP.
LLAMA_SERVER_MTP     = LLAMA_SERVER_VANILLA
LLAMA_CLI_MTP        = LLAMA_CLI_VANILLA
# Fork spiritbuun/buun-llama-cpp (a.k.a. Anbeeld/beellama.cpp) — DFlash: drafter
# GGUF SEPARADO que cross-attende aos hidden states do alvo (--spec-type dflash).
# Caminho na convenção dos demais; ajustar quando o build existir.
LLAMA_SERVER_DFLASH  = Path(r"C:\Projetos\llama-cpp-dflash\build\bin\Release\llama-server.exe")
LLAMA_CLI_DFLASH     = Path(r"C:\Projetos\llama-cpp-dflash\build\bin\Release\llama-cli.exe")

BACKENDS: dict[str, dict] = {
    "turbo": {
        "label":            "turboquant  — KV turbo2/3/4 (compressão agressiva, ctx longo)",
        "server":           LLAMA_SERVER_TURBO,
        "cli":              LLAMA_CLI_TURBO,
        "supports_spec_mtp": False,
        "supports_spec_dflash": False,
    },
    "vanilla": {
        "label":            "vanilla     — upstream llama.cpp (parsers de chat/tool mais novos)",
        "server":           LLAMA_SERVER_VANILLA,
        "cli":              LLAMA_CLI_VANILLA,
        "supports_spec_mtp": False,
        "supports_spec_dflash": False,
    },
    "mtp": {
        "label":            "mtp         — vanilla upstream + speculative MTP nativo (--spec-type draft-mtp)",
        "server":           LLAMA_SERVER_MTP,
        "cli":              LLAMA_CLI_MTP,
        "supports_spec_mtp": True,
        "supports_spec_dflash": False,
    },
    "dflash": {
        "label":            "dflash      — fork buun/beellama, spec decoding via drafter GGUF separado",
        "server":           LLAMA_SERVER_DFLASH,
        "cli":              LLAMA_CLI_DFLASH,
        "supports_spec_mtp": False,
        "supports_spec_dflash": True,
    },
}
LMS          = Path(r"C:\Users\PC\.lmstudio\bin\lms.exe")
CONFIG_FILE  = Path(r"C:\Projetos\llama-cpp-turboquant\last_config.json")
# Compartilhado com a app Electron (app/api/core/settings_store.py). Mesmo
# arquivo, mesmo schema — editar num lado reflete no outro.
SETTINGS_FILE = CONFIG_FILE.with_name("app_settings.json")


def _read_settings_file() -> dict:
    try:
        if SETTINGS_FILE.exists():
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def get_configured_model_dirs() -> list[Path]:
    """Apenas o que está explicitamente no settings (sem fallback).

    Usado para validar destinos de download — precisa estar cadastrado.
    """
    raw = _read_settings_file().get("model_paths")
    if isinstance(raw, list):
        return [Path(str(p)) for p in raw if str(p).strip()]
    return []


def get_model_dirs() -> list[Path]:
    """Pastas onde varrer GGUF. Default = MODELS_DIR quando settings vazio."""
    paths = get_configured_model_dirs()
    return paths if paths else [MODELS_DIR]


def save_model_dirs(paths: list[Path]) -> None:
    """Sobrescreve `model_paths` no settings, preservando outras chaves."""
    current = _read_settings_file()
    seen: set[str] = set()
    cleaned: list[str] = []
    for p in paths:
        s = str(p).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
    current["model_paths"] = cleaned
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(
            json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8",
        )
    except Exception as e:
        print(f"⚠️  Falha ao salvar settings: {e}")


def primary_model_dir() -> Path:
    """Primeiro caminho configurado — destino de downloads."""
    dirs = get_model_dirs()
    return dirs[0] if dirs else MODELS_DIR


def rel_to_any_root(model_path: Path, roots: list[Path] | None = None) -> str:
    """Caminho relativo à primeira raiz ancestral; cai pro absoluto se não houver."""
    for root in (roots if roots is not None else get_model_dirs()):
        try:
            return model_path.relative_to(root).as_posix()
        except ValueError:
            continue
    return str(model_path)


def containing_root(model_path: Path, roots: list[Path] | None = None) -> Path | None:
    """Qual raiz configurada contém este modelo — None se nenhuma."""
    for root in (roots if roots is not None else get_model_dirs()):
        try:
            model_path.relative_to(root)
            return root
        except ValueError:
            continue
    return None

MMPROJ_ERROR_PATTERNS = (
    "unsupported decoder rope type",
    "failed to load multimodal model",
    "mtmd_init_from_file: error",
)

# Padrões pra classificar a causa-raiz de uma queda do llama-server. A ordem
# importa: classify_failure tenta cada categoria em cima pra baixo, então
# assinaturas mais específicas vêm antes (ex.: KV antes de OOM_LOAD genérico).
OOM_PATTERNS = (
    "cudamalloc failed",
    "cuda error: out of memory",
    "out of memory",
    "failed to allocate",
    "ggml_backend_cuda_buffer_type_alloc_buffer",
    "ggml_cuda_host_malloc",
    "cublas_status_alloc_failed",
)
KV_REJECTED_PATTERNS = (
    "kv cache type not supported",
    "unsupported cache type",
    "unknown cache type",
)
UNSUPPORTED_FLAG_PATTERNS = (
    "unknown argument:",
    "unrecognized argument:",
    "invalid argument:",
)
# Marcadores no stdout que indicam "subiu e está aceitando requests". Usado pra
# decidir se a tentativa pode ser considerada estável (e portanto salva).
LOAD_OK_PATTERNS = (
    "listening on",
    "http server listening",
    "main: server is listening",
)
FAIL_HISTORY_FILE = Path(r"C:\Projetos\llama-cpp-turboquant\fail_history.jsonl")
STABILIZE_SECONDS = 30  # tempo sem crash pós-load pra promover a config a "boa"

# i5-14600KF: 6P + 8E = 14 núcleos físicos, 20 threads lógicos
CPU_THREADS_GEN   = 14
CPU_THREADS_BATCH = 20

THINKING_KEYWORDS = {"qwen3", "qwq", "deepseek-r1", "thinking", "reasoning", "gemma-4", "gemma4"}

# ─── choices ──────────────────────────────────────────────────────────────────

MODE_CHOICES = [
    questionary.Choice("llama-server  — API HTTP (OpenAI-compatible, Claude Code, etc.)",  value="server"),
    questionary.Choice("llama-cli     — Chat interativo no terminal",                      value="cli"),
    questionary.Choice("download      — Baixar modelo do HuggingFace para a pasta do LM Studio", value="download"),
    questionary.Choice("delete        — Apagar um modelo (.gguf) e o mmproj associado do disco", value="delete"),
    questionary.Choice("settings      — Configurar pastas onde procurar modelos (.gguf)",  value="settings"),
    questionary.Choice("nvidia-status — Status completo da GPU NVIDIA (memória, temperatura, uso, clocks, power…)", value="nvidia-status"),
]

KV_CACHE_CHOICES = [
    questionary.Choice("f32    — 32-bit, sem compressão  (só para debug — usa o dobro)",  value="f32"),
    questionary.Choice("f16    — padrão, sem compressão",                                  value="f16"),
    questionary.Choice("bf16   — bfloat16, mesma faixa do treino (sem perda)",             value="bf16"),
    questionary.Choice("turbo2 — ~2-bit KV, compressão extrema  (qualidade pode cair)",   value="turbo2"),
    questionary.Choice("turbo3 — ~3-bit KV, alta compressão  (ótimo para ctx longo)",     value="turbo3"),
    questionary.Choice("turbo4 — ~4-bit KV, mais qualidade que turbo3  [recomendado]",    value="turbo4"),
    questionary.Choice("q8_0   — leve, quase sem perda de qualidade",                     value="q8_0"),
    questionary.Choice("q5_1   — compressão moderada (mais qualidade que q5_0)",          value="q5_1"),
    questionary.Choice("q5_0   — compressão moderada",                                    value="q5_0"),
    questionary.Choice("q4_1   — compressão alta (mais qualidade que q4_0)",              value="q4_1"),
    questionary.Choice("iq4_nl — 4-bit non-linear  (mais qualidade que q4_0, mesmo VRAM)",value="iq4_nl"),
    questionary.Choice("q4_0   — compressão alta, economiza mais VRAM",                   value="q4_0"),
]

CONTEXT_CHOICES = [
    questionary.Choice("  4 096  tokens",                 value=4_096),
    questionary.Choice("  8 192  tokens",                 value=8_192),
    questionary.Choice(" 16 384  tokens",                 value=16_384),
    questionary.Choice(" 32 768  tokens",                 value=32_768),
    questionary.Choice(" 65 536  tokens  [recomendado]",  value=65_536),
    questionary.Choice("131 072  tokens",                 value=131_072),
    questionary.Choice("262 144  tokens  (256k)",         value=262_144),
    questionary.Choice("Digitar valor customizado...",     value=0),
]

REASONING_BUDGET_CHOICES = [
    questionary.Choice("    0  — Desabilitar thinking  (--reasoning off)", value=0),
    questionary.Choice(" 1024  — Rápido",                            value=1_024),
    questionary.Choice(" 2048",                                       value=2_048),
    questionary.Choice(" 4096  [recomendado — agentes de código]",   value=4_096),
    questionary.Choice(" 8192  — Problemas complexos",               value=8_192),
    questionary.Choice("16384  — Máxima qualidade de raciocínio",    value=16_384),
    questionary.Choice("   -1  — Sem limite",                        value=-1),
]

PARALLEL_CHOICES = [
    questionary.Choice("1  [recomendado para ctx longo]",  value=1),
    questionary.Choice("2  — 2 requisições simultâneas",   value=2),
    questionary.Choice("4  — 4 requisições simultâneas",   value=4),
]

MAX_TOKENS_CHOICES = [
    questionary.Choice(" 8192  [default — limita runaway/loop]", value=8_192),
    questionary.Choice("16384",                                  value=16_384),
    questionary.Choice(" 4096",                                  value=4_096),
    questionary.Choice(" 2048",                                  value=2_048),
    questionary.Choice("   -1  — Ilimitado (sem proteção)",      value=-1),
]

BATCH_CHOICES = [
    questionary.Choice(" 512",                                   value=512),
    questionary.Choice("1024",                                   value=1_024),
    questionary.Choice("2048  [default]",                        value=2_048),
    questionary.Choice("4096  — máximo throughput de prefill",   value=4_096),
]

UBATCH_CHOICES = [
    questionary.Choice("128  — menor pico de VRAM no prefill",  value=128),
    questionary.Choice("256",                                    value=256),
    questionary.Choice("512  [default]",                         value=512),
]

# --cache-ram vive em RAM do sistema (não VRAM). Define o teto pro prompt cache
# que guarda KV snapshots inteiros pra evitar reprocessamento em cache hit.
# Cada prompt salvo é ~base + (n_checkpoints × ~150 MiB) — com ctx 131k pode
# passar de 7 GB por prompt. Limite apertado = cache miss em toda request com
# baixa LCP similarity → reprefill longo → cliente faz timeout/cancel.
CACHE_RAM_CHOICES = [
    questionary.Choice(" 2048  —  2 GB  (default antigo — causa cancelamento em prompts > 30k)", value=2_048),
    questionary.Choice(" 8192  —  8 GB  (default do llama-server — cabe 1 prompt grande)",       value=8_192),
    questionary.Choice("16384  — 16 GB  (cabe 2-3 prompts grandes)",                              value=16_384),
    questionary.Choice("24576  — 24 GB  [recomendado — máquinas com 48 GB+ RAM]",                 value=24_576),
    questionary.Choice("32768  — 32 GB  (máximo seguro c/ 48 GB RAM)",                            value=32_768),
    questionary.Choice("65536  — 64 GB  (servidores dedicados c/ 96 GB+ RAM)",                    value=65_536),
]

# --ctx-checkpoints: snapshots intermediários do KV cache dentro de UMA
# conversa, criados a cada ~8k tokens processados. Servem pra restauração
# rápida quando o cliente edita uma mensagem (reusa do checkpoint mais
# próximo) ou pra cache reuse parcial. Cada checkpoint ocupa o tamanho do
# KV até aquela posição (~150 MiB em ctx 131k q8_0). Conta no --cache-ram.
CTX_CHECKPOINTS_CHOICES = [
    questionary.Choice(" 0  — desabilitar (sem restauração de edição)",              value=0),
    questionary.Choice(" 8  [recomendado — agentes / conversas que só crescem]",     value=8),
    questionary.Choice("16  — melhor reuse parcial em conversas com edição",         value=16),
    questionary.Choice("32  — default do llama-server (overkill p/ agentes)",        value=32),
]

# --spec-draft-n-max: teto de tokens que a head MTP rascunha por passo de
# decode antes da verificação pelo modelo base. Só tem efeito no backend
# 'mtp' (vanilla upstream, --spec-type draft-mtp) com um modelo que carrega head
# MTP. 2 é o recomendado pela model card da Unsloth; valores altos rascunham mais
# tokens de uma vez, mas sobem a taxa de
# rejeição — o ganho cai quando o rascunho erra e é descartado.
SPEC_DRAFT_N_MAX_CHOICES = [
    questionary.Choice("1  — rascunho mínimo (1 token especulado)",     value=1),
    questionary.Choice("2  — recomendado p/ MTP (Unsloth)  [recomendado]", value=2),
    questionary.Choice("3  — rascunho maior",                            value=3),
    questionary.Choice("4  — rascunho agressivo (mais rejeição)",        value=4),
]

# DFlash (fork buun) não expõe knob de cross-ctx por CLI: o preset
# --spec-dflash-default já liga spec-type dflash + tuning (n_max=7, p_min=0) e o
# binário auto-seta -cd 256. O único parâmetro que escolhemos é o drafter (-md).

# ─── helpers de backend ───────────────────────────────────────────────────────

def backend_choices_for_mode(mode: str) -> list[questionary.Choice]:
    """Lista os backends cujo binário existe pro modo escolhido (server|cli)."""
    bin_key = "server" if mode == "server" else "cli"
    out: list[questionary.Choice] = []
    for name, b in BACKENDS.items():
        path: Path = b[bin_key]
        if path.exists():
            out.append(questionary.Choice(b["label"], value=name))
        else:
            out.append(questionary.Choice(
                f"{b['label']}   (NÃO COMPILADO em {path})",
                value=name,
                disabled="binário não encontrado",
            ))
    return out


def backend_binary(backend: str, mode: str) -> Path:
    """Caminho do binário pro par (backend, mode)."""
    return BACKENDS[backend]["server" if mode == "server" else "cli"]


_BACKEND_HELP_CACHE: dict[tuple[str, str], str] = {}


def _backend_help_text(backend: str, mode: str) -> str:
    """Texto de `--help` do binário do (backend, mode), cacheado em memória.

    Primeira invocação ~80-200 ms; depois zero. String vazia se o binário
    não existe ou o `--help` falhou — quem chama decide o fallback.
    """
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


def _backend_supports_chat_template_kwargs(backend: str, mode: str) -> bool:
    """True se o binário lista --chat-template-kwargs no --help.

    Se o --help não pôde ser lido (binário ausente/timeout), assume True:
    no server o auto-degrade remove a flag caso ela seja rejeitada de fato.
    """
    help_text = _backend_help_text(backend, mode)
    if not help_text:
        return True
    return "--chat-template-kwargs" in help_text


# Fallback estático caso o probe --help falhe (binário ausente, timeout, regex
# não casa). Cobre o que sabemos que os 3 backends compilados oferecem hoje.
_KV_TYPES_FALLBACK: dict[str, set[str]] = {
    "turbo":   {"f32","f16","bf16","q8_0","q4_0","q4_1","iq4_nl","q5_0","q5_1","turbo2","turbo3","turbo4"},
    "vanilla": {"f32","f16","bf16","q8_0","q4_0","q4_1","iq4_nl","q5_0","q5_1"},
    "mtp":     {"f32","f16","bf16","q8_0","q4_0","q4_1","iq4_nl","q5_0","q5_1"},
    "dflash":  {"f32","f16","bf16","q8_0","q4_0","q4_1","iq4_nl","q5_0","q5_1"},
}

# Ordem de preferência pro remap quando o tipo escolhido não é suportado pelo
# backend atual. Vai do mais "próximo" pro mais distante. Usado em
# remap_kv_for_backend pra preservar a intenção do usuário ao trocar de backend.
_KV_REMAP_FALLBACK: dict[str, list[str]] = {
    "turbo2": ["q4_0", "iq4_nl", "q4_1"],
    "turbo3": ["q4_0", "iq4_nl", "q4_1"],
    "turbo4": ["q5_0", "q5_1", "q8_0"],
    "bf16":   ["f16", "f32"],
    "f32":    ["f16", "bf16"],
    "iq4_nl": ["q4_0", "q4_1"],
    "q4_1":   ["q4_0", "iq4_nl"],
    "q5_1":   ["q5_0", "q8_0"],
}

_KV_TYPES_CACHE: dict[tuple[str, str], set[str]] = {}


def _probe_kv_types(backend: str, mode: str) -> set[str]:
    """Lê os tipos aceitos em --cache-type-k chamando `<binário> --help`.

    Parseia o bloco 'allowed values: ...' que pode quebrar em múltiplas linhas
    (ex: turboquant: `q5_1,\\n    turbo2, turbo3, turbo4`). Cacheia o resultado
    em memória — primeira invocação ~80-200 ms, depois zero.
    """
    cache_key = (backend, mode)
    if cache_key in _KV_TYPES_CACHE:
        return _KV_TYPES_CACHE[cache_key]

    fallback = _KV_TYPES_FALLBACK.get(backend, set(_KV_TYPES_FALLBACK["vanilla"]))
    bin_path = backend_binary(backend, mode)
    if not bin_path.exists():
        _KV_TYPES_CACHE[cache_key] = fallback
        return fallback

    text = _backend_help_text(backend, mode)
    if not text:
        _KV_TYPES_CACHE[cache_key] = fallback
        return fallback

    # Localiza a opção --cache-type-k e captura o bloco "allowed values:" até
    # a próxima opção (linha começando com '-') ou "(default:".
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


def kv_cache_choices_for(backend: str, mode: str = "server") -> list[questionary.Choice]:
    """Lista os tipos de KV aceitos pelo binário do (backend, mode) escolhido.

    Detecta via `--help` na primeira invocação, cacheia em memória. Itens em
    KV_CACHE_CHOICES sem suporte ficam desabilitados; tipos suportados que não
    têm label conhecido entram como opção genérica.
    """
    supported = _probe_kv_types(backend, mode)
    known = {str(c.value) for c in KV_CACHE_CHOICES}

    out: list[questionary.Choice] = []
    for c in KV_CACHE_CHOICES:
        if c.value in supported:
            out.append(c)
        else:
            out.append(questionary.Choice(
                f"{c.title}   (não suportado em {backend})",
                value=c.value,
                disabled="não suportado pelo binário",
            ))
    for extra in sorted(supported - known):
        out.append(questionary.Choice(f"{extra:6} — (detectado no binário)", value=extra))
    return out


def remap_kv_for_backend(kv_cache: str, backend: str, mode: str = "server") -> tuple[str, str | None]:
    """Retorna (kv_efetivo, aviso). Se o backend atual não aceita o tipo,
    escolhe o primeiro fallback em _KV_REMAP_FALLBACK que ele aceita."""
    supported = _probe_kv_types(backend, mode)
    if kv_cache in supported:
        return kv_cache, None
    for candidate in _KV_REMAP_FALLBACK.get(kv_cache, []):
        if candidate in supported:
            return candidate, (
                f"Backend '{backend}' não aceita KV '{kv_cache}' — usando '{candidate}'."
            )
    return "f16", f"Backend '{backend}' não aceita KV '{kv_cache}' nem fallback — usando 'f16'."


# ─── helpers ──────────────────────────────────────────────────────────────────

def _read_all_configs() -> list[dict]:
	"""Lê o array de configurações do JSON (um por modelo)."""
	try:
		if CONFIG_FILE.exists():
			data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
			if isinstance(data, list):
				return data
	except Exception:
		pass
	return []


def save_config(cfg: dict) -> None:
	"""Salva a config no array. Chave composta: (model, backend) — cada combinação
	tem sua própria entrada salva, então mudar de turbo p/ vanilla não sobrescreve."""
	configs = _read_all_configs()
	model_key   = cfg.get("model", "")
	backend_key = cfg.get("backend", "turbo")

	configs = [
		c for c in configs
		if not (c.get("model") == model_key and c.get("backend", "turbo") == backend_key)
	]
	configs.append(cfg)

	try:
		CONFIG_FILE.write_text(json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8")
	except Exception:
		pass


def load_config(model_path: Path, backend: str) -> dict | None:
	"""Retorna a config salva para (model_path, backend), ou None.
	Configs legadas sem 'backend' são tratadas como 'turbo'."""
	configs = _read_all_configs()
	for cfg in configs:
		if cfg.get("model") == str(model_path) and cfg.get("backend", "turbo") == backend:
			return cfg
	return None


def find_mmproj(folder: Path, *, include_parent: bool = True) -> Path | None:
    for f in folder.iterdir():
        if f.is_file() and 'mmproj' in f.name.lower() and f.suffix == '.gguf':
            return f
    if include_parent:
        parent = folder.parent
        if parent != folder and parent.exists():
            try:
                for f in parent.iterdir():
                    if f.is_file() and 'mmproj' in f.name.lower() and f.suffix == '.gguf':
                        return f
            except OSError:
                pass
    return None


def make_alias(model_path: Path) -> str:
    m = _SPLIT_RE.match(model_path.name)
    if m:
        return m.group("base")
    return model_path.stem


def collect_models() -> list[Path]:
    roots = get_model_dirs()
    missing = [r for r in roots if not r.exists()]
    for r in missing:
        print(f"⚠️  Pasta configurada não existe: {r}")
    existing = [r for r in roots if r.exists()]
    if not existing:
        print("Nenhuma pasta de modelos válida. Configure via menu 'settings'.")
        sys.exit(1)

    def _is_primary(f: Path) -> bool:
        low = f.name.lower()
        if 'mmproj' in low:
            return False
        # Drafter do DFlash (-md) é arquivo auxiliar, não um alvo selecionável.
        if _DFLASH_DRAFTER_RE.search(low):
            return False
        m = _SPLIT_RE.match(f.name)
        return not m or m.group("part") == "00001"

    found: dict[Path, Path] = {}
    for root in existing:
        for f in root.rglob("*.gguf"):
            if not _is_primary(f):
                continue
            try:
                key = f.resolve()
            except OSError:
                key = f
            found.setdefault(key, f)
    models = sorted(found.values())

    if not models:
        print("Nenhum modelo .gguf encontrado.")
        sys.exit(1)

    return models


def is_thinking_model(model_path: Path) -> bool:
    """Chat template do GGUF; keywords no nome são só fallback.

    O nome do arquivo não diz se o modelo pensa (ornith-1.0-35b não tem nenhuma
    palavra reveladora e é reasoning). O template diz: quem pensa abre com <think>.
    """
    meta = _gguf_meta(model_path)
    if meta is not None:
        return bool(meta.get("has_thinking_template"))
    name = model_path.name.lower()
    return any(kw in name for kw in THINKING_KEYWORDS)


def is_mtp_model(model_path: Path) -> bool:
    """Detecta o tag MTP no nome do arquivo OU em qualquer pasta ancestral.

    O tag costuma viver na pasta do repo HF (ex: 'Qwen3.6-35B-A3B-MTP-GGUF'),
    que fica DOIS níveis acima do .gguf — o download grava em
    owner/repo/<quant>/arquivo.gguf, então nem o nome nem o parent imediato
    carregam 'MTP'. Varrer todos os parents cobre esse layout.
    """
    pattern = re.compile(r"(?:^|[-_.])mtp(?:[-_.]|$)", re.IGNORECASE)
    parts = [model_path.name, *(p.name for p in model_path.parents)]
    return any(pattern.search(part) for part in parts)


# Drafter do DFlash: GGUF separado, nomeado tipo 'dflash-draft-3.6-q8_0.gguf'.
_DFLASH_DRAFTER_RE = re.compile(r"dflash.*draft", re.IGNORECASE)
_DFLASH_DRAFTERS_CACHE: dict[str, list[Path]] = {}


def find_dflash_drafters(model_path: Path) -> list[Path]:
    """Lista os GGUFs drafter do DFlash candidatos a serem o -md do alvo.

    DFlash precisa de um GGUF de rascunho SEPARADO (o drafter cross-attende aos
    hidden states do alvo). Procura primeiro na pasta do modelo e ancestrais
    próximos (o drafter costuma vir no mesmo repo HF do alvo), depois varre todas
    as pastas de modelos. Dedup por caminho resolvido, ordenado do MAIOR pro
    menor — o estimador usa o primeiro (q8_0 > q4) pra ser conservador na VRAM.

    Cacheado por modelo: o estimador roda em loop e o rglob das pastas é caro.
    """
    ck = str(model_path)
    if ck in _DFLASH_DRAFTERS_CACHE:
        return _DFLASH_DRAFTERS_CACHE[ck]

    found: dict[str, Path] = {}

    def _scan(d: Path) -> None:
        try:
            for f in d.glob("*.gguf"):
                if _DFLASH_DRAFTER_RE.search(f.name):
                    try:    key = str(f.resolve()).lower()
                    except OSError: key = str(f).lower()
                    found.setdefault(key, f)
        except Exception:
            pass

    for d in [model_path.parent, *list(model_path.parents)[:3]]:
        _scan(d)
    for root in get_model_dirs():
        if not root.exists():
            continue
        try:
            for f in root.rglob("*.gguf"):
                if _DFLASH_DRAFTER_RE.search(f.name):
                    try:    key = str(f.resolve()).lower()
                    except OSError: key = str(f).lower()
                    found.setdefault(key, f)
        except Exception:
            pass

    def _size(p: Path) -> int:
        try:    return p.stat().st_size
        except Exception: return 0

    result = sorted(found.values(), key=_size, reverse=True)
    _DFLASH_DRAFTERS_CACHE[ck] = result
    return result


def select_dflash_drafter(model_path: Path) -> Path | None:
    """Escolhe o drafter (-md) pro DFlash: auto se houver só um, pergunta se
    houver vários, None (com aviso) se não achar nenhum."""
    drafters = find_dflash_drafters(model_path)
    if not drafters:
        print("⚠️  Nenhum drafter DFlash (dflash-draft-*.gguf) encontrado nas pastas de modelos.")
        print("   O servidor vai subir SEM speculative decoding — baixe o drafter do alvo p/ ativar.")
        return None
    if len(drafters) == 1:
        print(f"🔮 DFlash drafter detectado: {drafters[0].name}")
        return drafters[0]
    choice = questionary.select(
        "🔮 DFlash — escolha o drafter (-md):",
        choices=[questionary.Choice(f"{d.name}  ({d.stat().st_size / 1024**3:.2f} GB)", value=str(d))
                 for d in drafters],
    ).ask()
    return Path(choice) if choice else None


def ask_int(prompt: str, default: int) -> int:
    val = questionary.text(f"{prompt} (default {default}):").ask().strip()
    return int(val) if val else default


def stop_lms_server_if_running() -> None:
    """Para o servidor do LM Studio se estiver rodando, para liberar a porta 1234."""
    if not LMS.exists():
        return
    try:
        status = subprocess.run(
            [str(LMS), "server", "status"],
            capture_output=True, text=True, encoding="utf-8", timeout=10,
        )
        out = ((status.stdout or "") + (status.stderr or "")).lower()
        if "running" in out and "not running" not in out:
            print("ℹ️  LM Studio server estava ativo — parando para liberar a porta 1234…")
            subprocess.run([str(LMS), "server", "stop"], timeout=10)
    except Exception:
        pass


def run_capturing(cmd: str, on_load: "callable | None" = None) -> tuple[int, str, float | None]:
    """Roda o comando ecoando saída em tempo real e capturando para análise.

    Retorna (returncode, output, load_ts). load_ts é o timestamp em que algum
    padrão de LOAD_OK_PATTERNS foi detectado no stdout (server subiu); None se
    nunca subiu antes de cair. on_load é chamado uma única vez quando o load
    é detectado — usado pra disparar persistência da config que efetivamente
    chegou até esse ponto.
    """
    proc = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
        env=_cuda_env(),  # 3090 Ti como device CUDA 0 → main GPU (-mg 0)
    )
    chunks: list[str] = []
    load_ts: float | None = None
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            chunks.append(line)
            if load_ts is None:
                low = line.lower()
                if any(p in low for p in LOAD_OK_PATTERNS):
                    load_ts = time.time()
                    if on_load is not None:
                        try:
                            on_load()
                        except Exception:
                            pass
    except KeyboardInterrupt:
        proc.terminate()
        raise
    return proc.wait(), "".join(chunks), load_ts


def has_mmproj_error(output: str) -> bool:
    low = output.lower()
    return any(pat in low for pat in MMPROJ_ERROR_PATTERNS)


# Classificador de causa-raiz da queda. As categorias guiam a escada de
# degradação (next_degrade) — cada categoria escolhe primeiro o knob certo.
class Failure:
    OOM_LOAD         = "OOM_LOAD"          # estourou VRAM antes de subir
    OOM_RUNTIME      = "OOM_RUNTIME"       # subiu e morreu sob carga (ativação)
    UNSUPPORTED_FLAG = "UNSUPPORTED_FLAG"  # flag inválida pro build atual
    KV_TYPE_REJECTED = "KV_TYPE_REJECTED"  # tipo de KV não aceito
    MMPROJ           = "MMPROJ"            # falhou só o projector multimodal
    UNKNOWN          = "UNKNOWN"           # sem assinatura clara


def classify_failure(output: str, rc: int, load_ts: float | None) -> tuple[str, str]:
    """Devolve (categoria, trecho_do_erro). O trecho é uma linha curta que
    serve de evidência no log e na mensagem ao usuário."""
    low = output.lower()
    lines = output.splitlines()

    def first_match(patterns: tuple[str, ...]) -> str:
        for ln in lines[-200:]:  # tail — o erro real costuma estar no fim
            l = ln.lower()
            for p in patterns:
                if p in l:
                    return ln.strip()
        return ""

    if any(p in low for p in MMPROJ_ERROR_PATTERNS):
        return Failure.MMPROJ, first_match(MMPROJ_ERROR_PATTERNS)
    if any(p in low for p in KV_REJECTED_PATTERNS):
        return Failure.KV_TYPE_REJECTED, first_match(KV_REJECTED_PATTERNS)
    if any(p in low for p in UNSUPPORTED_FLAG_PATTERNS):
        return Failure.UNSUPPORTED_FLAG, first_match(UNSUPPORTED_FLAG_PATTERNS)
    if any(p in low for p in OOM_PATTERNS):
        cat = Failure.OOM_RUNTIME if load_ts else Failure.OOM_LOAD
        return cat, first_match(OOM_PATTERNS)
    # Crash silencioso depois de subir → trata como ativação sob carga.
    if load_ts is not None:
        return Failure.OOM_RUNTIME, f"crash silencioso após load (rc={rc})"
    return Failure.UNKNOWN, f"sem assinatura conhecida (rc={rc})"


# Ordem global de preferência de KV cache, do menor pro maior consumo de VRAM.
# next_degrade percorre essa lista filtrada pelos tipos suportados no backend.
KV_TIER_ORDER = [
    "f32", "bf16", "f16", "q8_0", "q5_1", "q5_0",
    "iq4_nl", "q4_1", "q4_0", "turbo4", "turbo3", "turbo2",
]


def _next_kv_tier(current: str, backend: str) -> str | None:
    """Próximo KV mais econômico que o atual, filtrado pelo backend."""
    supported = _probe_kv_types(backend, "server")
    try:
        idx = KV_TIER_ORDER.index(current)
    except ValueError:
        idx = -1
    for cand in KV_TIER_ORDER[idx + 1:]:
        if cand in supported:
            return cand
    return None


def _extract_unknown_flag(error_excerpt: str) -> str | None:
    """Extrai o nome da flag de uma mensagem 'unknown argument: --foo'."""
    m = re.search(r"(?:unknown|unrecognized|invalid)\s+argument:\s*(\S+)", error_excerpt, re.IGNORECASE)
    return m.group(1) if m else None


def next_degrade(cfg: dict, failure: str, error_excerpt: str) -> tuple[dict | None, str | None]:
    """Aplica UM degrau na config preservando velocidade/qualidade até onde dá.
    Devolve (nova_cfg, descricao) ou (None, None) se a escada acabou.

    Nunca mexe em context_window — quem chama precisa lidar com isso à parte.
    A ordem dos passos é: tratar erros endereçáveis 1:1 primeiro (flag,
    KV rejeitado), depois escada genérica de VRAM. Em OOM_RUNTIME, ubatch e
    parallel sobem na fila porque atacam ativação sob carga.
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
        # Sem mapeamento direto: cai pra escada genérica abaixo.

    if failure == Failure.KV_TYPE_REJECTED:
        new_kv, warning = remap_kv_for_backend(cfg.get("kv_cache", "f16"), backend, "server")
        if new_kv != cfg.get("kv_cache"):
            old = cfg["kv_cache"]
            cfg["kv_cache"] = new_kv
            return cfg, f"KV {old} → {new_kv} (não aceito por '{backend}')"

    # OOM_RUNTIME ataca ativação primeiro: ubatch e parallel cortam o pico
    # de memória sob carga sem desligar layers da GPU.
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
            return cfg, f"parallel {np_} → {new_np} (KV escala linear com slots)"

    # Escada genérica de VRAM (OOM_LOAD, UNKNOWN, ou continuação de OOM_RUNTIME
    # depois que os atalhos acima esgotaram).
    if not cfg.get("flash_attn", True):
        cfg["flash_attn"] = True
        return cfg, "flash-attention OFF → ON (reduz KV, custo zero em velocidade)"

    np_ = cfg.get("parallel_slots", 1)
    if np_ > 1:
        new_np = max(1, np_ // 2)
        cfg["parallel_slots"] = new_np
        return cfg, f"parallel {np_} → {new_np} (KV escala linear com slots)"

    cur_kv = cfg.get("kv_cache", "f16")
    nxt_kv = _next_kv_tier(cur_kv, backend)
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
        # Último recurso — cada layer no CPU mata a velocidade.
        step = max(1, ngl // 4) if ngl > 4 else 1
        new_ngl = max(0, ngl - step)
        cfg["gpu_layers"] = new_ngl
        return cfg, f"GPU layers {ngl} → {new_ngl} (ÚLTIMO RECURSO — perde velocidade)"

    return None, None


def _build_cmd_from_cfg(cfg: dict) -> str:
    """Reconstrói a cmd do llama-server a partir de uma config (possivelmente
    degradada) — usado entre tentativas sem voltar pra UI."""
    mmproj = cfg.get("mmproj")
    return build_command(
        model_path=Path(cfg["model"]),
        backend=cfg.get("backend", "turbo"),
        context_window=cfg["context_window"],
        kv_cache=cfg["kv_cache"],
        flash_attn=cfg.get("flash_attn", True),
        gpu_layers=cfg.get("gpu_layers", 99),
        parallel_slots=cfg.get("parallel_slots", 1),
        batch_size=cfg.get("batch_size", 2_048),
        ubatch_size=cfg.get("ubatch_size", 512),
        threads_gen=cfg.get("threads_gen", CPU_THREADS_GEN),
        threads_batch=cfg.get("threads_batch", CPU_THREADS_BATCH),
        reasoning_budget=cfg.get("reasoning_budget"),
        mlock=cfg.get("mlock", True),
        max_tokens=cfg.get("max_tokens", 8_192),
        cache_ram=cfg.get("cache_ram", 8_192),
        ctx_checkpoints=cfg.get("ctx_checkpoints", 8),
        n_cpu_moe=cfg.get("n_cpu_moe", 0),
        spec_draft_n_max=cfg.get("spec_draft_n_max", 2),
        spec_dflash_draft=cfg.get("spec_dflash_draft"),
        preserve_thinking=cfg.get("preserve_thinking", False),
        mmproj=Path(mmproj) if mmproj else None,
        verbose=cfg.get("verbose", False),
        split_mode=cfg.get("split_mode"),
        tensor_split=cfg.get("tensor_split"),
        main_gpu=cfg.get("main_gpu"),
    )


def _append_fail_history(cfg: dict, failure: str, error_excerpt: str,
                         degrade_applied: str | None, attempt: int) -> None:
    """Loga uma tentativa fracassada em jsonl. Best-effort, não levanta."""
    try:
        FAIL_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "attempt": attempt,
            "model": cfg.get("model"),
            "backend": cfg.get("backend"),
            "failure": failure,
            "degrade_applied": degrade_applied,
            "error_excerpt": error_excerpt[:400] if error_excerpt else "",
            "config": cfg,
        }
        with FAIL_HISTORY_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


# Política de reinício automático do servidor:
#   - até MAX_RESTARTS quedas dentro de uma janela de RESTART_WINDOW segundos
#   - backoff exponencial entre tentativas, com teto em BACKOFF_MAX
#   - se exceder o limite, desiste (provavelmente é falha de configuração)
SERVER_MAX_RESTARTS  = 10
SERVER_RESTART_WINDOW = 120
SERVER_BACKOFF_START  = 2
SERVER_BACKOFF_MAX    = 30


def run_server_resiliently(
    cfg: dict,
    mode_label: str,
    use_mmproj: Path | None,
) -> None:
    """Executa o servidor em loop, degradando a config a cada queda.

    Comportamento:
    - rc == 0: encerra limpamente.
    - Ctrl+C: encerra limpamente.
    - MMPROJ: oferece fallback LM Studio (degradar não resolve).
    - Outras falhas: classifica → aplica próximo degrau via next_degrade →
      reconstrói cmd → reinicia. Cada falha grava linha no fail_history.jsonl.
    - Quando o servidor sobe (detectado por LOAD_OK_PATTERNS) e sobrevive
      STABILIZE_SECONDS sem cair, a config atual é promovida em last_config.json
      automaticamente, sem prompt.
    - Limite anti-loop: SERVER_MAX_RESTARTS quedas em SERVER_RESTART_WINDOW
      ou escada esgotada → desiste.
    """
    restarts: list[float] = []
    backoff = SERVER_BACKOFF_START
    attempt = 0
    model_path = Path(cfg["model"])
    context_window = cfg["context_window"]
    parallel_slots = cfg.get("parallel_slots", 1)

    while True:
        attempt += 1
        cmd = _build_cmd_from_cfg(cfg)
        if attempt > 1:
            print(f"\n🔁 Reiniciando {mode_label} (tentativa #{attempt})…")
            print(f"   {cmd}\n")

        # Estado pra esta tentativa: se subiu, marcar; se sobreviveu, promover.
        stable_marked = {"done": False}
        attempt_cfg = dict(cfg)

        def on_load() -> None:
            # Carregou — salva como "tentativa estável" imediatamente. Se
            # sobreviver STABILIZE_SECONDS sem cair, será promovida de novo no
            # post-loop (mesma operação, idempotente).
            save_config(attempt_cfg)
            print(f"   💾 Config salva (carregou — aguardando {STABILIZE_SECONDS}s pra estabilizar).")

        try:
            rc, output, load_ts = run_capturing(cmd, on_load=on_load)
        except KeyboardInterrupt:
            print("\n\n⛔ Interrompido pelo usuário.")
            return

        # Estabilizou se subiu e ficou STABILIZE_SECONDS sem morrer.
        stabilized = (
            load_ts is not None and (time.time() - load_ts) >= STABILIZE_SECONDS
        )

        if rc == 0:
            if stabilized:
                save_config(attempt_cfg)
            print(f"\n✅ {mode_label} encerrou normalmente.")
            return

        failure, excerpt = classify_failure(output, rc, load_ts)

        # MMPROJ: degradar não resolve — cai pro LM Studio.
        if failure == Failure.MMPROJ and use_mmproj is not None:
            _append_fail_history(attempt_cfg, failure, excerpt, None, attempt)
            print()
            print("⚠️  Falha ao carregar o projector multimodal (mmproj) no llama-server.")
            print("   Esse rope type/decoder ainda não é suportado por este build.")
            if questionary.confirm("🔁 Tentar carregar o modelo via LM Studio (lms)?", default=True).ask():
                run_with_lms(model_path, context_window, parallel_slots)
            return

        # Tenta um degrau. Se a escada acabou, ou se o servidor ficou estável
        # antes de cair (provavelmente crash legítimo, não config), desiste do
        # auto-degrade pra não destruir a config funcional.
        new_cfg, degrade_desc = next_degrade(cfg, failure, excerpt)
        _append_fail_history(attempt_cfg, failure, excerpt, degrade_desc, attempt)

        if new_cfg is None:
            print(f"\n❌ {mode_label} caiu ({failure}: {excerpt[:120]})")
            print("   Escada de degradação esgotada — sem mais knobs a baixar sem mexer no contexto.")
            return

        # Janela deslizante anti-loop — mesmo com degradação, se está caindo
        # rápido demais provavelmente o classificador está errado.
        now = time.time()
        restarts = [t for t in restarts if now - t < SERVER_RESTART_WINDOW]
        restarts.append(now)
        if len(restarts) > SERVER_MAX_RESTARTS:
            print(
                f"\n❌ {mode_label} caiu {len(restarts)} vezes em "
                f"{SERVER_RESTART_WINDOW}s — desistindo do auto-degrade."
            )
            return

        print(f"\n💥 Falha classificada: {failure}")
        if excerpt:
            print(f"   evidência: {excerpt[:160]}")
        print(f"   degrau aplicado: {degrade_desc}")
        print(f"   aguardando {backoff}s antes de reiniciar (Ctrl+C pra cancelar)…")
        try:
            time.sleep(backoff)
        except KeyboardInterrupt:
            print("\n\n⛔ Auto-degrade cancelado pelo usuário.")
            return

        cfg = new_cfg
        backoff = min(backoff * 2, SERVER_BACKOFF_MAX)


def lms_model_key(model_path: Path) -> str | None:
    """Resolve a modelKey do lms (ex: 'nvidia/nemotron-3-nano-omni@q4_k_m') a partir
    do caminho do .gguf, consultando 'lms ls --json --variants'."""
    if not LMS.exists():
        return None
    try:
        result = subprocess.run(
            [str(LMS), "ls", "--json", "--variants"],
            capture_output=True, text=True, encoding="utf-8", timeout=20,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout or "[]")
    except Exception:
        return None

    try:
        rel = model_path.relative_to(MODELS_DIR).as_posix().lower()
    except ValueError:
        return None

    for entry in data:
        base_key = (entry.get("model") or {}).get("modelKey")
        for variant in entry.get("variants", []):
            ident = variant.get("indexedModelIdentifier", "")
            if "@" not in ident:
                continue
            _, fs_path = ident.split("@", 1)
            if fs_path.lower() == rel:
                return base_key
    return None


def run_with_lms(model_path: Path, context_window: int, parallel_slots: int) -> int:
    """Fallback: sobe o servidor do LM Studio e carrega o modelo via lms."""
    if not LMS.exists():
        print(f"❌ lms não encontrado em: {LMS}")
        return 1

    alias = make_alias(model_path)
    key   = lms_model_key(model_path)

    if key is None:
        print(f"❌ Não consegui resolver a chave do lms para: {model_path.name}")
        print("   Rode 'lms ls' e carregue manualmente.")
        return 1

    print()
    print(f"🔁 Subindo LM Studio server e carregando '{key}'…")

    start_cmd = f'"{LMS}" server start --port 1234 --bind 0.0.0.0 --cors'
    print(f"   {start_cmd}")
    subprocess.run(start_cmd, shell=True)

    # --context-length é o KV total no LM Studio (llama.cpp por baixo), igual ao
    # -c do llama-server: multiplicar por parallel_slots dá context_window por slot.
    load_cmd = (
        f'"{LMS}" load "{key}"'
        f' --gpu max --context-length {context_window * parallel_slots}'
        f' --parallel {parallel_slots} --identifier "{alias}" -y'
    )
    print(f"   {load_cmd}")
    return subprocess.run(load_cmd, shell=True).returncode


def ask_context_window() -> int:
    selected = questionary.select(
        "📏 Janela de contexto (por slot — cada slot -np recebe esta janela inteira):",
        choices=CONTEXT_CHOICES,
    ).ask()
    if selected == 0:
        return ask_int("   Valor customizado", 65_536)
    return selected

# ─── download (HuggingFace) ───────────────────────────────────────────────────

def _human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


_SPLIT_RE = re.compile(r"^(?P<base>.+)-(?P<part>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)


def parse_hf_url(url: str) -> tuple[str, str | None, str]:
    """Extrai (repo_id, filename, branch) de uma URL do HuggingFace.

    `filename` é None quando a URL aponta para o repositório inteiro
    (ex: https://huggingface.co/google/gemma-3-27b-it).

    Suporta:
      - https://huggingface.co/{owner}/{repo}
      - https://huggingface.co/{owner}/{repo}?show_file_info={file}
      - https://huggingface.co/{owner}/{repo}/(resolve|blob|tree)/{branch}[/{caminho/arquivo.gguf}]
    """
    from urllib.parse import urlparse, parse_qs, unquote

    url = url.strip().strip('"').strip("'")
    parsed = urlparse(url)
    if parsed.netloc not in ("huggingface.co", "www.huggingface.co"):
        raise ValueError("A URL precisa ser do huggingface.co")

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError("Não consegui identificar o repositório na URL")

    owner, repo = parts[0], parts[1]
    repo_id = f"{owner}/{repo}"
    branch = "main"
    filename: str | None = None

    if len(parts) >= 4 and parts[2] in ("resolve", "blob", "tree"):
        branch = parts[3]
        rest = "/".join(parts[4:])
        filename = rest or None
    else:
        qs = parse_qs(parsed.query)
        for key in ("show_file_info", "show-file-info", "filename"):
            if qs.get(key):
                filename = qs[key][0]
                break

    return repo_id, (unquote(filename) if filename else None), branch


def _hf_headers() -> dict[str, str]:
    headers = {"User-Agent": "models.py-downloader/1.0"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def hf_list_files(repo_id: str) -> list[str]:
    """Lista os arquivos do repositório (branch principal) via API do HuggingFace."""
    import urllib.request

    req = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo_id}", headers=_hf_headers()
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [s["rfilename"] for s in data.get("siblings", []) if s.get("rfilename")]


def _is_mmproj(name: str) -> bool:
    return "mmproj" in Path(name).name.lower()


def _gguf_group_key(name: str) -> str:
    """Conjunto a que o arquivo pertence: o quant, sem o sufixo -00001-of-00005."""
    base = Path(name).name
    m = _SPLIT_RE.match(base)
    if m:
        return m.group("base")
    return base[:-5] if base.lower().endswith(".gguf") else base


def download_file(url: str, dest: Path) -> None:
    """Baixa `url` para `dest`, com barra de progresso e suporte a retomada (.part)."""
    import time
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")

    resume_pos = tmp.stat().st_size if tmp.exists() else 0

    headers = _hf_headers()
    if resume_pos:
        headers["Range"] = f"bytes={resume_pos}-"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        clen = resp.headers.get("Content-Length")
        partial = getattr(resp, "status", 200) == 206
        if resume_pos and partial:
            file_mode = "ab"
            total = (int(clen) + resume_pos) if clen else 0
        else:
            resume_pos = 0
            file_mode = "wb"
            total = int(clen) if clen else 0

        downloaded = resume_pos
        start = time.time()
        last = 0.0
        with open(tmp, file_mode) as f:
            while True:
                chunk = resp.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last >= 0.4:
                    last = now
                    speed = (downloaded - resume_pos) / max(now - start, 1e-6)
                    if total:
                        pct = downloaded / total * 100
                        filled = int(30 * downloaded / total)
                        bar = "#" * filled + "-" * (30 - filled)
                        sys.stdout.write(
                            f"\r   [{bar}] {pct:5.1f}%  "
                            f"{_human_size(downloaded)}/{_human_size(total)}  "
                            f"{_human_size(speed)}/s    "
                        )
                    else:
                        sys.stdout.write(
                            f"\r   {_human_size(downloaded)}  {_human_size(speed)}/s    "
                        )
                    sys.stdout.flush()
    sys.stdout.write("\r" + " " * 70 + "\r")
    sys.stdout.flush()
    tmp.replace(dest)


def pick_download_dir() -> Path | None:
    """Escolhe uma das pastas cadastradas para receber o download.

    Retorna None se o usuário cancelar ou não houver pasta cadastrada (com
    aviso pra ele ir configurar primeiro). Pula a pergunta quando só tem uma.
    """
    dirs = get_configured_model_dirs()
    if not dirs:
        print("❌ Nenhuma pasta de modelos cadastrada.")
        print("   Cadastre pelo menos um caminho no menu 'settings' antes de baixar.")
        return None
    if len(dirs) == 1:
        return dirs[0]
    choices = [questionary.Choice(str(p), value=p) for p in dirs]
    picked = questionary.select(
        "📁 Em qual pasta salvar?",
        choices=choices,
    ).ask()
    return picked


def _download_many(
    repo_id: str,
    branch: str,
    rel_paths: list[str],
    subdir: str | None = None,
    base_root: Path | None = None,
) -> None:
    if base_root is None:
        base_root = pick_download_dir()
        if base_root is None:
            return
    owner, repo = repo_id.split("/", 1)
    base_dir = base_root / owner / repo
    if subdir:
        base_dir = base_dir / subdir

    print()
    print(f"📦 Repositório : {repo_id}  (branch: {branch})")
    print(f"📁 Destino     : {base_dir}")
    print(f"📄 Arquivo(s)  :")
    for p in rel_paths:
        print(f"     • {p}")
    print()
    if not questionary.confirm(f"⬇️  Baixar {len(rel_paths)} arquivo(s)?", default=True).ask():
        return

    try:
        for i, rel in enumerate(rel_paths, 1):
            dest = base_dir / Path(rel).name
            url = f"https://huggingface.co/{repo_id}/resolve/{branch}/{rel}?download=true"
            print(f"\n[{i}/{len(rel_paths)}] {Path(rel).name}")
            if dest.exists():
                print(f"   já existe ({_human_size(dest.stat().st_size)}) — pulando "
                      f"(apague o arquivo para baixar de novo).")
                continue
            download_file(url, dest)
            print(f"   ✅ {dest}")
        print("\n✅ Concluído. Os modelos já aparecem na lista do LM Studio e deste launcher.")
    except KeyboardInterrupt:
        print("\n⛔ Download interrompido. O(s) arquivo(s) parcial(is) (.part) foram mantidos — "
              "rode de novo para retomar de onde parou.")
    except Exception as e:
        print(f"\n❌ Falha no download: {e}")
        print("   Se o repositório for restrito (gated), aceite os termos no site do HuggingFace")
        print("   e defina a variável de ambiente HF_TOKEN com seu token de acesso.")


def _pick_quant(groups: dict[str, list[str]]) -> str | None:
    if len(groups) == 1:
        return next(iter(groups))
    choices = []
    for q in sorted(groups):
        n = len(groups[q])
        choices.append(questionary.Choice(f"{q}" + (f"   ({n} partes)" if n > 1 else ""), value=q))
    return questionary.select("🎚️  Escolha o quant para baixar:", choices=choices).ask()


def hf_search_gguf(query: str, limit: int = 25) -> list[dict]:
    """Procura no HuggingFace repositórios com arquivos GGUF batendo com `query`."""
    import urllib.parse
    import urllib.request

    qs = urllib.parse.urlencode({
        "search": query, "filter": "gguf",
        "sort": "downloads", "direction": -1, "limit": limit,
    })
    req = urllib.request.Request(
        f"https://huggingface.co/api/models?{qs}", headers=_hf_headers()
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _offer_gguf_search(repo_id: str) -> None:
    """O repo não tem GGUF — procura mirrors GGUF do mesmo modelo e deixa escolher."""
    name_q = repo_id.split("/")[-1]
    print(f"ℹ️  {repo_id} não tem arquivos .gguf (parece o repo original em safetensors).")
    print(f"   Procurando versões GGUF de '{name_q}' no HuggingFace…")
    try:
        results = hf_search_gguf(name_q)
    except Exception as e:
        print(f"❌ A busca falhou: {e}")
        print("   Procure manualmente no site e cole o link de um repo '-GGUF'.")
        return

    choices: list = []
    for r in results:
        rid = r.get("id") or r.get("modelId") or ""
        if "/" not in rid:
            continue
        dl, lk = r.get("downloads", 0) or 0, r.get("likes", 0) or 0
        choices.append(questionary.Choice(f"{rid}    ↓{dl:,}  ♥{lk}", value=rid))

    if not choices:
        print("❌ Não encontrei nenhuma versão GGUF. Cole o link de um repo '-GGUF' manualmente.")
        return

    choices.append(questionary.Choice("⟵ cancelar", value=""))
    picked = questionary.select(
        f"📦 Encontrei {len(choices) - 1} repositório(s) GGUF — escolha:",
        choices=choices,
    ).ask()
    if not picked:
        return
    _download_repo_gguf(picked, "main")


def _download_repo_gguf(repo_id: str, branch: str) -> None:
    """Lista os .gguf do repo, deixa escolher o quant (+ mmproj) e baixa."""
    try:
        repo_files = hf_list_files(repo_id)
    except Exception as e:
        print(f"⚠️  Não consegui abrir {repo_id} ({e}).")
        _offer_gguf_search(repo_id)
        return

    gguf_files = [f for f in repo_files if f.lower().endswith(".gguf")]
    if not gguf_files:
        _offer_gguf_search(repo_id)
        return

    mmprojs     = [f for f in gguf_files if _is_mmproj(f)]
    model_files = [f for f in gguf_files if not _is_mmproj(f)]

    groups: dict[str, list[str]] = {}
    for f in model_files:
        groups.setdefault(_gguf_group_key(f), []).append(f)

    if not groups:                       # repo só com mmproj (raro)
        _download_many(repo_id, branch, sorted(mmprojs))
        return

    chosen = _pick_quant(groups)
    if not chosen:
        return

    to_get = sorted(groups[chosen])
    if mmprojs:
        names = ", ".join(Path(x).name for x in mmprojs)
        if questionary.confirm(f"🖼️  Incluir o(s) arquivo(s) de visão (mmproj)? ({names})", default=True).ask():
            to_get += mmprojs

    _download_many(repo_id, branch, to_get, subdir=chosen)


def download_flow() -> None:
    url = questionary.text(
        "🔗 Cole o link do HuggingFace (repositório, página do arquivo, ou link de download):"
    ).ask()
    if not url or not url.strip():
        print("Nenhum link informado.")
        return

    try:
        repo_id, filename, branch = parse_hf_url(url)
    except ValueError as e:
        print(f"❌ {e}")
        return

    # ── link do repositório inteiro: escolhe quant / oferece busca por mirror GGUF ──
    if filename is None:
        _download_repo_gguf(repo_id, branch)
        return

    # ── link de um arquivo específico ──────────────────────────────────────────
    repo_files: list[str] = []
    try:
        repo_files = hf_list_files(repo_id)
    except Exception:
        pass  # sem a listagem dá pra baixar só o arquivo do link

    gguf_files = [f for f in repo_files if f.lower().endswith(".gguf")]
    mmprojs    = [f for f in gguf_files if _is_mmproj(f)]

    to_get = [filename]
    m = _SPLIT_RE.match(Path(filename).name)
    if m:
        parts = sorted(f for f in gguf_files
                       if not _is_mmproj(f) and _gguf_group_key(f) == m.group("base"))
        if len(parts) > 1:
            print(f"ℹ️  Esse modelo está dividido em {len(parts)} partes — todas serão baixadas.")
            to_get = parts
    if mmprojs and not _is_mmproj(filename):
        names = ", ".join(Path(x).name for x in mmprojs)
        if questionary.confirm(f"🖼️  Incluir o arquivo de visão (mmproj) do repo? ({names})", default=True).ask():
            to_get += mmprojs

    subdir = None if _is_mmproj(filename) else _gguf_group_key(filename)
    _download_many(repo_id, branch, to_get, subdir=subdir)


# ─── delete ───────────────────────────────────────────────────────────────────

def delete_config(model_path: Path) -> None:
    """Remove a configuração salva do modelo, se existir."""
    configs = _read_all_configs()
    kept = [c for c in configs if c.get("model") != str(model_path)]
    if len(kept) != len(configs):
        try:
            CONFIG_FILE.write_text(json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass


def _rmdir_if_empty(path: Path, stop_at: Path) -> None:
    """Sobe removendo diretórios vazios até (sem incluir) `stop_at`."""
    try:
        while path != stop_at and path.is_dir() and not any(path.iterdir()):
            path.rmdir()
            path = path.parent
    except Exception:
        pass


def delete_flow() -> None:
    models = collect_models()
    roots = get_model_dirs()

    labels = []
    for m in models:
        mmproj = find_mmproj(m.parent)
        tag    = " | 🖼️ mmproj" if mmproj else ""
        rel    = rel_to_any_root(m, roots)
        labels.append(f"{make_alias(m)}{tag}  ({rel})")

    roots_str = ", ".join(str(r) for r in roots)
    print(f"\n📋 {len(models)} modelo(s) em {roots_str}\n")

    sel = questionary.select(
        "🗑️  Selecione o modelo para DELETAR:",
        choices=labels,
    ).ask()
    if not sel:
        return

    model_path = next((models[i] for i, l in enumerate(labels) if l == sel), None)
    if model_path is None:
        print("Modelo não encontrado.")
        return

    folder = model_path.parent

    # arquivo(s) do modelo: se for um conjunto dividido, pega todas as partes
    to_remove: list[Path] = []
    m = _SPLIT_RE.match(model_path.name)
    if m:
        key = m.group("base")
        for f in sorted(folder.glob("*.gguf")):
            if not _is_mmproj(f.name) and _gguf_group_key(f.name) == key:
                to_remove.append(f)
    else:
        to_remove.append(model_path)

    mmproj = find_mmproj(folder, include_parent=False)
    if mmproj:
        to_remove.append(mmproj)

    # restos de download interrompido
    for f in list(to_remove):
        part = f.with_name(f.name + ".part")
        if part.exists():
            to_remove.append(part)

    # dedup preservando ordem, só o que existe de fato
    seen: set[Path] = set()
    to_remove = [f for f in to_remove if f.exists() and not (f in seen or seen.add(f))]

    if not to_remove:
        print("Nada para remover.")
        return

    total = 0
    for f in to_remove:
        try:
            total += f.stat().st_size
        except Exception:
            pass

    print()
    print(f"Serão apagados {len(to_remove)} arquivo(s)  ({_human_size(total)}):")
    for f in to_remove:
        print(f"   • {f}")
    print()
    if not questionary.confirm("⚠️  Confirmar exclusão? Não dá pra desfazer.", default=False).ask():
        print("Cancelado.")
        return

    errors = 0
    for f in to_remove:
        try:
            f.unlink()
        except Exception as e:
            errors += 1
            print(f"   ❌ {f}: {e}")

    delete_config(model_path)
    _rmdir_if_empty(folder, containing_root(model_path) or MODELS_DIR)

    if errors:
        print(f"\n⚠️  Concluído com {errors} erro(s).")
    else:
        print("\n✅ Modelo removido.")
        print("   (se o LM Studio estiver aberto, ele atualiza a lista sozinho)")


# ─── settings ─────────────────────────────────────────────────────────────────

def settings_flow() -> None:
    """Edita a lista de pastas onde varremos GGUF. Compartilhado com a app."""
    while True:
        paths = get_model_dirs()
        print(f"\n📁 Pastas configuradas ({SETTINGS_FILE}):")
        if not paths:
            print("   (nenhuma — usando default)")
        else:
            for i, p in enumerate(paths, 1):
                marker = "" if p.exists() else "  ⚠️ não existe"
                print(f"   {i}. {p}{marker}")
        print()

        action = questionary.select(
            "O que você quer fazer?",
            choices=[
                questionary.Choice("➕ Adicionar caminho",                value="add"),
                questionary.Choice("✏️  Editar caminho",                  value="edit"),
                questionary.Choice("🗑️  Remover caminho",                value="remove"),
                questionary.Choice("↩️  Resetar para o default do LM Studio", value="reset"),
                questionary.Choice("✅ Sair",                              value="quit"),
            ],
        ).ask()

        if not action or action == "quit":
            return

        if action == "add":
            new = questionary.text("Caminho da pasta de modelos:").ask()
            if not new:
                continue
            new_path = Path(new.strip().strip('"'))
            if not new_path.exists():
                if not questionary.confirm(
                    f"⚠️  {new_path} não existe. Salvar mesmo assim?", default=False,
                ).ask():
                    continue
            save_model_dirs(paths + [new_path])
            print("✅ Adicionado e salvo.")

        elif action == "edit":
            if not paths:
                print("Nada para editar.")
                continue
            choices = [questionary.Choice(str(p), value=i) for i, p in enumerate(paths)]
            idx = questionary.select("Qual caminho editar?", choices=choices).ask()
            if idx is None:
                continue
            new = questionary.text("Novo valor:", default=str(paths[idx])).ask()
            if new is None:
                continue
            new_path = Path(new.strip().strip('"'))
            updated = list(paths)
            updated[idx] = new_path
            save_model_dirs(updated)
            print("✅ Atualizado e salvo.")

        elif action == "remove":
            if not paths:
                print("Nada para remover.")
                continue
            choices = [questionary.Choice(str(p), value=i) for i, p in enumerate(paths)]
            idx = questionary.select("Qual caminho remover?", choices=choices).ask()
            if idx is None:
                continue
            updated = [p for i, p in enumerate(paths) if i != idx]
            save_model_dirs(updated)
            print("✅ Removido e salvo.")

        elif action == "reset":
            if questionary.confirm(
                f"Resetar para apenas {MODELS_DIR}?", default=False,
            ).ask():
                save_model_dirs([MODELS_DIR])
                print("✅ Resetado.")


# ─── nvidia-status ────────────────────────────────────────────────────────────

NVIDIA_SMI = Path(r"C:\Windows\System32\nvidia-smi.exe")

_NVIDIA_QUERY_FIELDS = [
    "name", "driver_version",
    "temperature.gpu", "fan.speed",
    "utilization.gpu", "utilization.memory",
    "memory.total", "memory.used", "memory.free",
    "power.draw", "power.limit",
    "clocks.sm", "clocks.mem",
    "pstate",
]


def _nvidia_smi_path() -> str:
    """Caminho do nvidia-smi: o do System32 (se existir) ou só o nome no PATH."""
    return str(NVIDIA_SMI) if NVIDIA_SMI.exists() else "nvidia-smi"


def _cuda_env() -> dict:
    """os.environ com a 3090 Ti pinada como GPU principal (device CUDA 0).

    Sem isso o CUDA enumera por slot PCIe: a 3090 (bus 01) vira device 0 e a
    3090 Ti (bus 04) device 1. Ordenamos as GPUs por nome ("Ti" primeiro) e
    pinamos via UUID em CUDA_VISIBLE_DEVICES — UUID torna a escolha imune a
    qualquer heurística do driver. Com a Ti em device 0, o default -mg 0 do
    llama.cpp faz dela a main GPU e -sm none roda só nela. Devolve cópia do
    env (não altera o processo atual); env intacto se nvidia-smi indisponível.
    """
    env = dict(os.environ)
    smi = _nvidia_smi_path()
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
            if gpus:
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                env["CUDA_VISIBLE_DEVICES"] = ",".join(u for _, u in gpus)
    except Exception:
        pass
    return env


def nvidia_status_flow() -> None:
    """Mostra o status completo da GPU NVIDIA via nvidia-smi."""
    smi = _nvidia_smi_path()

    # 1) visão geral (tabela padrão: temp, uso, VRAM, power, processos)
    print()
    print("═" * 70)
    print("  nvidia-smi — visão geral")
    print("═" * 70)
    try:
        rc = subprocess.run([smi], shell=False).returncode
    except FileNotFoundError:
        print("❌ nvidia-smi não encontrado. Driver NVIDIA instalado?")
        return
    except Exception as e:
        print(f"❌ Erro ao executar nvidia-smi: {e}")
        return
    if rc != 0:
        print(f"⚠️  nvidia-smi encerrou com código {rc}.")

    # 2) resumo legível dos principais sensores
    print()
    print("═" * 70)
    print("  Resumo (memória / temperatura / uso / clocks / power)")
    print("═" * 70)
    try:
        result = subprocess.run(
            [smi, f"--query-gpu={','.join(_NVIDIA_QUERY_FIELDS)}",
             "--format=csv,noheader"],
            capture_output=True, text=True, encoding="utf-8", timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                vals = [v.strip() for v in line.split(",")]
                d = dict(zip(_NVIDIA_QUERY_FIELDS, vals))
                print(f"   GPU            : {d.get('name', '?')}  (driver {d.get('driver_version', '?')})")
                print(f"   Temperatura    : {d.get('temperature.gpu', '?')} °C   |  Fan: {d.get('fan.speed', '?')}")
                print(f"   Uso GPU        : {d.get('utilization.gpu', '?')}   |  Uso mem: {d.get('utilization.memory', '?')}")
                print(f"   VRAM           : {d.get('memory.used', '?')} usada / {d.get('memory.total', '?')} total  ({d.get('memory.free', '?')} livre)")
                print(f"   Clocks         : SM {d.get('clocks.sm', '?')}  |  Mem {d.get('clocks.mem', '?')}")
                print(f"   Power          : {d.get('power.draw', '?')} / {d.get('power.limit', '?')}  |  P-State: {d.get('pstate', '?')}")
        else:
            print(f"⚠️  Não consegui obter o resumo (rc={result.returncode}).")
    except Exception as e:
        print(f"⚠️  Falha ao obter o resumo: {e}")

    print()
    if questionary.confirm("🔁 Monitorar continuamente? (nvidia-smi -l 2 — Ctrl+C para sair)", default=False).ask():
        print()
        try:
            subprocess.run([smi, "-l", "2"], shell=False)
        except KeyboardInterrupt:
            print("\n⛔ Monitoramento encerrado.")
        except Exception as e:
            print(f"❌ Erro: {e}")


# ─── estimador de mem (VRAM/RAM) ──────────────────────────────────────────────

# Bytes por elemento do KV cache, por tipo.
# Tipos quantizados usam blocos de 32 elementos: q8_0=34B, q5_0=22B, q4_0=18B.
# turbo2/turbo3/turbo4 do fork turboquant comprimem em ~2 / ~3 / ~4 bits + headers.
_KV_BYTES_PER_ELEM = {
    "f32":    4.0,
    "f16":    2.0,
    "bf16":   2.0,
    "q8_0":   1.0625,   # 34/32
    "q5_1":   0.75,     # 24/32
    "q5_0":   0.6875,   # 22/32
    "q4_1":   0.625,    # 20/32
    "iq4_nl": 0.5625,   # 18/32 (mesmo footprint do q4_0)
    "q4_0":   0.5625,   # 18/32
    "turbo2": 0.30,     # ~2.4 bits efetivos
    "turbo3": 0.40,     # ~3.2 bits efetivos — calibrado contra Gemma 4 31B
    "turbo4": 0.55,     # ~4.4 bits efetivos
}

# Compute buffer da GPU pra modelos NÃO-SSM (transformers densos / MoE clássicas).
# Parte fixa do grafo + parcela que escala com o dtype do KV cache (o grafo de
# atenção carrega tensores no tipo do KV). Calibrado contra Qwen3-Next 80B
# @ ctx 200k, ub 512: KV bf16 → ~800 MiB, f32 → ~1150.
_GPU_COMPUTE_BASE_MIB = 450
_GPU_COMPUTE_KV_COEF  = 176          # MiB por byte-por-elemento do KV cache
_GPU_COMPUTE_PARALLEL_MIB = 220      # MiB extra por slot -np além do primeiro

# Em híbridos SSM com fused Gated Delta Net (arch qwen35/qwen3next) o scratch
# da DeltaNet domina o compute e escala LINEAR com ctx — quase não depende do
# dtype do KV. Substitui a fórmula base+KV_coef. Calibrado contra Qwen3.6 27B
# qwen35:
#   ctx  65k q8_0 → real 495 MiB
#   ctx 262k bf16 → real 1915 MiB
# Fit linear: 50 + 240 × (ctx/32k). Pequena folga conservadora vs ajuste exato
# (21 + 237 × ctx/32k) pra absorver variação por ub/np.
_GPU_COMPUTE_SSM_BASE_MIB  = 50
_GPU_COMPUTE_SSM_PER_32K   = 240

# Contexto MTP de rascunho (head MTP) cria um SEGUNDO llama_context com KV de
# uma camada full-attention + compute buffer próprio. KV escala com ctx;
# compute medido contra Qwen3.6 27B @ ctx 262k (824 MiB no log de OOM).
_MTP_DRAFT_COMPUTE_MIB = 800

# DFlash: o drafter é um GGUF SEPARADO carregado junto do alvo (-md), sempre na
# GPU (-ngld all). O custo dominante são os PESOS do drafter (~1.75 GB no
# drafter 3.6 q8_0) — somados via tamanho em disco. Além deles há um segundo
# llama_context pequeno (ctx do rascunho auto-setado pra 256 pelo preset → KV
# desprezível) e o buffer de cross-attention. Esses dois cabem num compute fixo.
_DFLASH_DRAFT_COMPUTE_MIB = 450

# mmproj (visão): pesos vão pra VRAM no load; o COMPUTE buffer é alocado em
# warmup (imagem máxima do modelo). Em Qwen3-VL 27B BF16 @ 1472² o warmup pede
# 248 MiB e pode ser rejeitado silenciosamente quando a VRAM tá apertada — o
# servidor sobe mas qualquer requisição multimodal estoura OOM. Manter no total
# pra a barra refletir "imagem funciona ou não".
_MMPROJ_COMPUTE_MIB  = 250

# Overhead sobre tensor_total (alinhamento + buffers internos do llama.cpp).
# Validado contra Qwen3.6 27B qwen35 Q6_K: tensor sum 21347 MiB, GPU+CPU
# loaded 21347 MiB → overhead efetivo ~0%. Mantemos 2% só pra absorver
# variação entre builds — o antigo 1.05 (5%) inflava ~1 GB e fazia ESTOURA
# falso. Aplicado quando NÃO temos tensor_total_mib do GGUF; com tensor_total
# usamos os valores exatos.
_WEIGHT_OVERHEAD     = 1.02
_MIB                 = 1024 * 1024

# Bytes por elemento dos tensores GGUF, por ggml_type. Quants usam blocos com
# (block_size_bytes / n_elems_per_block). Usado pra estimar o peso dos
# tensores MoE (ffn_*_exps) e decidir o default de --n-cpu-moe.
_GGML_BYTES_PER_ELEM = {
    0:  4.0,         # F32
    1:  2.0,         # F16
    2:  18/32,       # Q4_0
    3:  20/32,       # Q4_1
    6:  22/32,       # Q5_0
    7:  24/32,       # Q5_1
    8:  34/32,       # Q8_0
    9:  40/32,       # Q8_1
    10: 84/256,      # Q2_K
    11: 110/256,     # Q3_K
    12: 144/256,     # Q4_K
    13: 176/256,     # Q5_K
    14: 210/256,     # Q6_K
    15: 292/256,     # Q8_K
    16: 102/256,     # IQ2_XXS
    17: 110/256,     # IQ2_XS
    18: 174/256,     # IQ3_XXS
    19: 56/256,      # IQ1_S
    20: 144/256,     # IQ4_NL
    21: 132/256,     # IQ3_S
    22: 132/256,     # IQ2_S
    23: 144/256,     # IQ4_XS
    24: 1.0,         # I8
    25: 2.0,         # I16
    26: 4.0,         # I32
    27: 8.0,         # I64
    28: 8.0,         # F64
    29: 64/256,      # IQ1_M
    30: 2.0,         # BF16
    # 31-33 e 36-38 foram removidos do GGUF (Q4_0_*, IQ4_NL_*).
    34: 54/256,      # TQ1_0   (ternary, 54B/256 elems)
    35: 66/256,      # TQ2_0   (ternary, 66B/256 elems)
    39: 17/32,       # MXFP4   (4-bit + E8M0 scale por bloco de 32)
    40: 18/32,       # NVFP4   (4-bit + E4M3 scale; estimativa)
    41: 12/256,      # Q1_0    (~1-bit; estimativa conservadora)
}

_MOE_TENSOR_RE = re.compile(r"blk\.(\d+)\.[^/]*?(?:_exps|\.exps)", re.IGNORECASE)
# Tensor da projeção K da atenção — presente só nas camadas de atenção plena.
# Em híbridos SSM (qwen3next etc.) as camadas DeltaNet/SSM não têm esse tensor.
_ATTN_K_TENSOR_RE = re.compile(r"blk\.(\d+)\.attn_k", re.IGNORECASE)
# Tensores especiais que NÃO ficam dentro de `blk.N.*` — afetam a política de
# offload do llama.cpp (output sai pra GPU primeiro quando -ngl>=1; token_embd
# só vai pra GPU quando -ngl > n_layer+1). Validado contra Qwen3.6 27B qwen35
# Q6_K @ -ngl 64 (CPU buffer 1294 MiB = token_embd 994 + 1 block 298).
_TOKEN_EMBD_RE   = re.compile(r"^token_embd\.", re.IGNORECASE)
_OUTPUT_RE       = re.compile(r"^output\.weight$", re.IGNORECASE)
_BLOCK_RE        = re.compile(r"^blk\.(\d+)\.", re.IGNORECASE)

_GGUF_META_CACHE: dict[str, dict] = {}


def _gguf_meta(model_path: Path) -> dict | None:
    """Lê metadados do GGUF (arch, n_layer, n_head_kv, head_dim). Cache por path.

    Sempre percorre a seção de tensor_info pra coletar:
      - peso médio dos tensores `ffn_*_exps` por camada (MoE) — alimenta a
        sugestão de `--n-cpu-moe`.
      - quais camadas têm atenção plena (híbridos SSM tipo qwen3next/qwen35
        deixam só um subconjunto, o resto é DeltaNet/SSM sem KV ctx-scaling).
      - tamanho exato em bytes de `token_embd.weight`, `output.weight` e do
        somatório de todos os `blk.N.*` — alimenta o split de pesos GPU/CPU
        no estimador (llama.cpp offloada output primeiro, deixa token_embd
        e o(s) bloco(s) excedentes na CPU quando `-ngl <= n_layer`).
    """
    import struct
    key = str(model_path)
    if key in _GGUF_META_CACHE:
        return _GGUF_META_CACHE[key] or None

    raw: dict = {}
    n_tensors = 0
    moe_per_layer_bytes: dict[int, float] = {}
    attn_layer_set: set[int] = set()
    token_embd_bytes  = 0.0
    output_bytes      = 0.0
    blocks_bytes      = 0.0
    other_bytes       = 0.0
    try:
        with open(model_path, "rb") as f:
            if f.read(4) != b"GGUF":
                _GGUF_META_CACHE[key] = {}
                return None
            f.read(4)                                              # version
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            kv_count  = struct.unpack("<Q", f.read(8))[0]

            def read_value(vtype: int):
                if vtype == 0:  return f.read(1)[0]
                if vtype == 1:  return struct.unpack("<b", f.read(1))[0]
                if vtype == 2:  return struct.unpack("<H", f.read(2))[0]
                if vtype == 3:  return struct.unpack("<h", f.read(2))[0]
                if vtype == 4:  return struct.unpack("<I", f.read(4))[0]
                if vtype == 5:  return struct.unpack("<i", f.read(4))[0]
                if vtype == 6:  return struct.unpack("<f", f.read(4))[0]
                if vtype == 7:  return bool(f.read(1)[0])
                if vtype == 10: return struct.unpack("<Q", f.read(8))[0]
                if vtype == 11: return struct.unpack("<q", f.read(8))[0]
                if vtype == 12: return struct.unpack("<d", f.read(8))[0]
                if vtype == 8:
                    sl = struct.unpack("<Q", f.read(8))[0]
                    return f.read(sl).decode("utf-8", errors="replace")
                if vtype == 9:                                    # array
                    et = struct.unpack("<I", f.read(4))[0]
                    n  = struct.unpack("<Q", f.read(8))[0]
                    # Coleta arrays curtos de escalares (ex: head_count_kv per-layer,
                    # sliding_window_pattern). Vocabs/tokens são gigantes — só consome.
                    if n <= 4096 and et in (0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12):
                        return [read_value(et) for _ in range(n)]
                    for _ in range(n):
                        read_value(et)
                    return None
                raise ValueError(f"unknown GGUF type {vtype}")

            for _ in range(kv_count):
                key_len = struct.unpack("<Q", f.read(8))[0]
                k_str = f.read(key_len).decode("utf-8", errors="replace")
                vtype = struct.unpack("<I", f.read(4))[0]
                raw[k_str] = read_value(vtype)

            # Seção de tensor_info. Sempre percorrer — alimenta múltiplos campos:
            #   - moe_per_layer_bytes  (pesos `ffn_*_exps`, pra sugestão de ncmoe)
            #   - attn_layer_set       (quais camadas têm KV ctx-scaling em híbridos)
            #   - token_embd / output / blocks bytes (split exato GPU/CPU)
            for _ in range(n_tensors):
                nl = struct.unpack("<Q", f.read(8))[0]
                tname = f.read(nl).decode("utf-8", errors="replace")
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
                ttype = struct.unpack("<I", f.read(4))[0]
                f.read(8)                                      # offset (ignorado)

                elems = 1
                for d in dims:
                    elems *= d
                bpe = _GGML_BYTES_PER_ELEM.get(ttype, 1.0)
                t_bytes = elems * bpe

                am = _ATTN_K_TENSOR_RE.search(tname)
                if am:
                    attn_layer_set.add(int(am.group(1)))

                if _TOKEN_EMBD_RE.match(tname):
                    token_embd_bytes += t_bytes
                elif _OUTPUT_RE.match(tname):
                    output_bytes += t_bytes
                elif _BLOCK_RE.match(tname):
                    blocks_bytes += t_bytes
                    m = _MOE_TENSOR_RE.search(tname)
                    if m:
                        li = int(m.group(1))
                        moe_per_layer_bytes[li] = moe_per_layer_bytes.get(li, 0.0) + t_bytes
                else:
                    other_bytes += t_bytes
    except Exception:
        _GGUF_META_CACHE[key] = {}
        return None

    arch = raw.get("general.architecture")
    if not arch:
        _GGUF_META_CACHE[key] = {}
        return None

    def g(name, default=None):
        # Trata None (ex: chave guardada como array → read_value devolve None)
        # como ausente, pra cair no default.
        v = raw.get(f"{arch}.{name}")
        return default if v is None else v

    n_layer = g("block_count")
    n_embd  = g("embedding_length")
    n_head  = g("attention.head_count")
    n_kv    = g("attention.head_count_kv", n_head)
    key_len = g("attention.key_length")
    val_len = g("attention.value_length")

    # Sliding Window Attention (Gemma 2/3/4, Mistral 7B, etc.). Quando ativo,
    # algumas camadas usam só `sliding_window` cells de KV — não o ctx inteiro.
    swa_window     = g("attention.sliding_window")
    swa_pattern    = g("attention.sliding_window_pattern")
    key_len_swa    = g("attention.key_length_swa")
    val_len_swa    = g("attention.value_length_swa")

    # Híbridos SSM (qwen3next etc.): cada camada DeltaNet/SSM mantém um estado
    # recorrente de tamanho FIXO (não escala com ctx) — buffer de conv + estado.
    # Tamanho por camada = (conv + state) elems × 4 B (f32). Escala com n_seqs.
    ssm_inner = g("ssm.inner_size")
    ssm_state = g("ssm.state_size")
    ssm_conv  = g("ssm.conv_kernel")
    ssm_group = g("ssm.group_count")
    ssm_state_per_layer_mib = 0.0
    if ssm_inner and ssm_state and ssm_conv:
        conv_elems  = (int(ssm_inner) + 2 * int(ssm_group or 0) * int(ssm_state)) * (int(ssm_conv) - 1)
        state_elems = int(ssm_inner) * int(ssm_state)
        ssm_state_per_layer_mib = (conv_elems + state_elems) * 4 / _MIB

    if not (n_layer and n_embd and n_head):
        _GGUF_META_CACHE[key] = {}
        return None

    head_dim_k = key_len or (n_embd // n_head)
    head_dim_v = val_len or head_dim_k

    # n_head_kv pode vir como int (uniforme) ou list (per-layer) — mantém como está
    if isinstance(n_kv, list):
        n_kv_norm: int | list = [int(x) for x in n_kv]
    else:
        n_kv_norm = int(n_kv)

    # MoE: expert_count > 0 marca o modelo. moe_per_layer_mib é o peso médio
    # dos tensores `ffn_*_exps` em cada camada com experts — não inclui
    # gates/normas/atenção, só o que o llama.cpp realmente offloada com -ncmoe.
    expert_count        = int(g("expert_count", 0) or 0)
    expert_used_count   = int(g("expert_used_count", 0) or 0)
    moe_layers_count    = len(moe_per_layer_bytes)
    moe_per_layer_mib   = 0
    if moe_layers_count:
        avg_bytes = sum(moe_per_layer_bytes.values()) / moe_layers_count
        moe_per_layer_mib = int(avg_bytes / _MIB)

    # Template de chat embutido (Jinja). Alguns expõem a variável
    # `preserve_thinking` — quando o launcher passa --chat-template-kwargs
    # com ela, o template mantém os blocos <think> de turnos anteriores no
    # histórico em vez de descartá-los. Ver build_command.
    chat_template = raw.get("tokenizer.chat_template") or ""
    # E é o mesmo template que revela se o modelo PENSA: quem raciocina abre a
    # resposta com <think> (ou expõe enable_thinking / reasoning_content).
    has_thinking = any(
        kw in chat_template.lower()
        for kw in ("<think>", "enable_thinking", "reasoning_content")
    )

    # Split exato de pesos GPU/CPU: o llama.cpp trata `output.weight` e
    # `token_embd.weight` como "layers" separadas das `blk.N.*`. Ordem real
    # de offload com -ngl K (validado contra Qwen3.6 27B qwen35 ngl=64):
    #   K == 0          → nada na GPU
    #   1..n_layer      → output + (K-1) blocks na GPU; resto + token_embd na CPU
    #   n_layer+1       → output + todos os blocks na GPU; token_embd na CPU
    #   >= n_layer+2    → tudo na GPU (output + blocks + token_embd)
    # Quando o tensor_info não pôde ser lido (n_tensors == 0), os _mib vão a 0
    # e o estimador cai pro fallback gpu_frac antigo.
    token_embd_mib  = int(token_embd_bytes  / _MIB)
    output_mib      = int(output_bytes      / _MIB)
    blocks_mib      = int(blocks_bytes      / _MIB)
    tensor_total_mib = token_embd_mib + output_mib + blocks_mib + int(other_bytes / _MIB)

    meta = {
        "arch":              arch,
        "n_layer":           int(n_layer),
        "n_embd":            int(n_embd),
        "n_head":            int(n_head),
        "n_head_kv":         n_kv_norm,
        "head_dim_k":        int(head_dim_k),
        "head_dim_v":        int(head_dim_v),
        "swa_window":        int(swa_window) if swa_window else None,
        "swa_pattern":       [bool(x) for x in swa_pattern] if isinstance(swa_pattern, list) else None,
        "head_dim_k_swa":    int(key_len_swa) if key_len_swa else None,
        "head_dim_v_swa":    int(val_len_swa) if val_len_swa else None,
        "is_moe":            expert_count > 0,
        "expert_count":      expert_count,
        "expert_used_count": expert_used_count,
        "moe_layers_count":  moe_layers_count,
        "moe_per_layer_mib": moe_per_layer_mib,
        # Tensores não-block (afetam o split GPU/CPU) e somatório de blocos.
        "token_embd_mib":    token_embd_mib,
        "output_mib":        output_mib,
        "blocks_mib":        blocks_mib,
        "tensor_total_mib":  tensor_total_mib,
        # Camadas com atenção plena (KV escala com ctx). Vazio = não foi
        # possível inspecionar tensores → trata todas as camadas como atenção.
        "attn_layers":       sorted(attn_layer_set),
        # Estado recorrente por camada SSM (MiB, f32). 0 = não é híbrido SSM.
        "ssm_state_per_layer_mib": ssm_state_per_layer_mib,
        # Template de chat referencia a variável `preserve_thinking`?
        "supports_preserve_thinking": "preserve_thinking" in chat_template,
        "has_thinking_template":      has_thinking,
    }
    _GGUF_META_CACHE[key] = meta
    return meta


def _kv_cache_mib(meta: dict, ctx: int, parallel: int,
                  bytes_pe_k: float, bytes_pe_v: float) -> int:
    """Calcula o KV cache total em MiB, respeitando per-layer n_head_kv e SWA.

    K e V têm bytes-por-elemento separados: o build turboquant promove o K de
    turbo2/3/4 pra q8_0 em modelos GQA (auto-asymmetric). Quem decide os tipos
    efetivos de K e V é estimate_memory.
    """
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

    # Híbridos SSM (qwen3next etc.): só as camadas em attn_layers têm KV que
    # escala com o ctx; as DeltaNet/SSM guardam estado recorrente de tamanho
    # fixo (~poucas dezenas de MiB, já absorvido pelo compute buffer).
    # attn_layers vazio = modelo não inspecionado → conta todas as camadas.
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
        total_bytes += per_token * layer_ctx * parallel
    return int(total_bytes / _MIB)


def _mtp_draft_kv_mib(meta: dict, ctx: int,
                      bytes_pe_k: float, bytes_pe_v: float) -> int:
    """KV cache do contexto MTP de rascunho: 1 layer full-attention sobre ctx inteiro.

    O fork mtp do llama.cpp cria um segundo llama_context pra head MTP com
    n_ctx = ctx do main e n_layer = 1. Sem SWA (a head usa atenção plena).
    Usa o n_head_kv máximo do modelo pra ser conservador em arquiteturas
    com per-layer GQA variável.
    """
    nk = meta["n_head_kv"]
    n_kv = max(nk) if isinstance(nk, list) else int(nk)
    h_k  = meta["head_dim_k"]
    h_v  = meta["head_dim_v"]
    bytes_total = n_kv * (h_k * bytes_pe_k + h_v * bytes_pe_v) * ctx
    return int(bytes_total / _MIB)


def _model_disk_size_mib(model_path: Path) -> int:
    """Tamanho total do .gguf em MiB, somando partes split."""
    total = 0
    m = _SPLIT_RE.match(model_path.name)
    if m:
        key = m.group("base")
        for f in sorted(model_path.parent.glob("*.gguf")):
            if not _is_mmproj(f.name) and _gguf_group_key(f.name) == key:
                try: total += f.stat().st_size
                except Exception: pass
    else:
        try: total = model_path.stat().st_size
        except Exception: pass
    return total // _MIB


_VRAM_INFO_CACHE: tuple[int, int] | None = None   # (total, free) em MiB
_RAM_TOTAL_CACHE:  int | None = None


def _gpu_mem_mib() -> tuple[int | None, int | None]:
    """(total, livre) de VRAM em MiB SOMADOS sobre todas as GPUs. (None, None)
    se indisponível.

    A estimativa precisa do espaço *livre*, não do total: o SO/driver/desktop
    seguram ~1-1.5 GB que o llama.cpp nunca vê. Comparar contra o total fazia
    o estimador subestimar a ocupação e não flagrar OOM real.

    Com 2+ placas e split-mode layer (default), pesos e KV se repartem entre as
    GPUs — o orçamento é a SOMA. Antes lia só a GPU 0 e cortava o budget pela
    metade numa rig de 2 placas.
    """
    global _VRAM_INFO_CACHE
    if _VRAM_INFO_CACHE is not None:
        t, fr = _VRAM_INFO_CACHE
        return (t or None, fr or None)
    smi = _nvidia_smi_path()
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
            _VRAM_INFO_CACHE = (total, free)
            return _VRAM_INFO_CACHE
    except Exception:
        pass
    _VRAM_INFO_CACHE = (0, 0)
    return (None, None)


def _gpu_total_mib() -> int | None:
    """VRAM física total (memory.total)."""
    return _gpu_mem_mib()[0]


def _gpu_free_mib() -> int | None:
    """VRAM livre agora (memory.free) — o que o llama.cpp realmente terá."""
    return _gpu_mem_mib()[1]


def _gpu_list() -> list[tuple[int, str, int]]:
    """[(index, name, total_mib), ...] por GPU. [] se nvidia-smi indisponível."""
    smi = _nvidia_smi_path()
    try:
        r = subprocess.run(
            [smi, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            out = []
            for line in r.stdout.strip().splitlines():
                idx_s, name, tot_s = (x.strip() for x in line.split(","))
                out.append((int(idx_s), name, int(tot_s)))
            return out
    except Exception:
        pass
    return []


def _ram_total_mib() -> int | None:
    global _RAM_TOTAL_CACHE
    if _RAM_TOTAL_CACHE is not None:
        return _RAM_TOTAL_CACHE or None
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
            _RAM_TOTAL_CACHE = int(stat.ullTotalPhys // _MIB)
            return _RAM_TOTAL_CACHE
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    _RAM_TOTAL_CACHE = int(line.split()[1]) // 1024
                    return _RAM_TOTAL_CACHE
    except Exception:
        pass
    _RAM_TOTAL_CACHE = 0
    return None


def _split_weights_by_ngl(
    meta:           dict | None,
    weights_total:  int,
    gpu_layers:     int,
) -> tuple[int, int]:
    """Decide quantos MiB de pesos ficam em VRAM vs CPU, dado -ngl K.

    Política real do llama.cpp (validado contra Qwen3.6 27B qwen35 Q6_K @ ngl=64,
    com CPU buffer = 1294 MiB ≈ token_embd 994 + 1 bloco SSM 298):

        K == 0          → 0 na GPU
        1..n_layer      → output + (K-1) blocos na GPU; resto + token_embd na CPU
        n_layer + 1     → output + todos os blocos na GPU; token_embd na CPU
        K >= n_layer+2  → tudo na GPU

    Quando `output_mib == 0` mas `token_embd_mib > 0`, o modelo usa tied
    embeddings (Gemma 3/4, alguns Phi) — o token_embd faz o papel da output.
    Nesse caso ele segue a regra da output (vai pra GPU quando K >= 1) e o
    patamar "+2" colapsa pro "+1".

    Quando o tensor_info não pôde ser lido (token_embd_mib == 0) caímos pro
    fallback proporcional (comportamento antigo).
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
        # Fallback: split proporcional (comportamento antigo). Conservador.
        frac = 1.0 if gpu_layers >= n_layer else gpu_layers / n_layer
        gpu = int(weights_total * frac)
        return gpu, weights_total - gpu

    # Tied embeddings: token_embd serve como output. Trata como se fosse a
    # output layer (vai pra GPU quando K >= 1, sem patamar extra).
    tied = output_w <= 0

    # `weights_total` já tem o overhead aplicado em cima do tensor_total. Como
    # o overhead se distribui uniforme entre os tensores, escalamos cada
    # componente pela mesma razão pra preservar a soma total.
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
        # token_embd faz dupla função → vai pra GPU junto da output (K >= 1).
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
) -> dict:
    """Estima componentes de VRAM/RAM (em MiB) para a config corrente."""
    meta        = _gguf_meta(model_path)
    # Quando temos tensor_total_mib do GGUF parsing, usar ele direto (sum exato
    # dos tensores carregados). Senão usar o disk_size inflado pelo overhead.
    if meta and meta.get("tensor_total_mib"):
        weights_mib = int(meta["tensor_total_mib"] * _WEIGHT_OVERHEAD)
    else:
        weights_mib = int(_model_disk_size_mib(model_path) * _WEIGHT_OVERHEAD)

    # mmproj: pesos vão pra VRAM no load; compute buffer é alocado em warmup.
    # Em VRAM apertada o warmup do compute pode falhar silenciosamente — o
    # servidor sobe pra texto mas estoura na 1ª imagem. Manter ambos no total.
    mmproj_weights_mib = int((mmproj.stat().st_size // _MIB) * _WEIGHT_OVERHEAD) if mmproj else 0
    mmproj_compute_mib = _MMPROJ_COMPUTE_MIB if mmproj else 0

    kv_eff, _    = remap_kv_for_backend(kv_cache, backend, mode)
    bytes_pe     = _KV_BYTES_PER_ELEM.get(kv_eff, 2.0)
    # Auto-asymmetric do turboquant: em modelos GQA (n_head_kv < n_head) o K em
    # turbo2/3/4 é promovido pra q8_0 (a V mantém o tipo escolhido), senão a
    # qualidade cai porque cada K é compartilhado por vários query heads. Sem
    # modelar isso o estimador subconta o KV em ~2-3× a parcela do K.
    kv_eff_k = kv_eff
    if backend == "turbo" and kv_eff in ("turbo2", "turbo3", "turbo4") and meta:
        nk = meta["n_head_kv"]
        n_kv_typ = max(nk) if isinstance(nk, list) else int(nk)
        if n_kv_typ and meta.get("n_head", 0) > n_kv_typ:
            kv_eff_k = "q8_0"
    bytes_pe_k   = _KV_BYTES_PER_ELEM.get(kv_eff_k, bytes_pe)
    kv_total_mib = (
        _kv_cache_mib(meta, context_window, parallel_slots, bytes_pe_k, bytes_pe)
        if meta else 0
    )

    # Estado recorrente das camadas SSM (híbridos): tamanho fixo, escala só com
    # n_seqs. Camadas SSM = todas menos as de atenção plena detectadas.
    ssm_state_mib = 0
    if meta:
        per_ssm = meta.get("ssm_state_per_layer_mib", 0.0)
        attn_n  = len(meta.get("attn_layers") or [])
        if per_ssm and attn_n:
            n_ssm = max(0, meta["n_layer"] - attn_n)
            ssm_state_mib = int(per_ssm * n_ssm * max(1, parallel_slots))

    n_layer  = meta["n_layer"] if meta else 0
    # Split exato GPU/CPU baseado nos tensores reais (ver _split_weights_by_ngl).
    vram_weights, ram_weights = _split_weights_by_ngl(meta, weights_mib, gpu_layers)
    # Fração ativa pra KV/SSM (que ainda escalam proporcional ao -ngl). Quando
    # ngl == n_layer (todos os blocos repetitivos na GPU, output na GPU) usamos
    # 1.0 porque o KV é compartilhado entre todos os blocos offloaded.
    gpu_active = gpu_layers > 0
    if not n_layer or gpu_layers >= n_layer:
        gpu_frac_kv = 1.0
    else:
        gpu_frac_kv = max(0, gpu_layers) / n_layer

    vram_kv      = int(kv_total_mib * gpu_frac_kv)
    vram_ssm     = int(ssm_state_mib * gpu_frac_kv)

    # Compute buffer. Híbridos SSM (qwen35/qwen3next) usam fórmula linear-em-ctx
    # dominada pelo scratch da DeltaNet; demais arquiteturas usam base+KV_coef.
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

    # Contexto MTP de rascunho: segundo llama_context (1 layer + compute), só
    # criado quando o backend mtp encontra um modelo MTP. Vive sempre na GPU.
    vram_mtp_kv      = 0
    vram_mtp_compute = 0
    if (meta and gpu_active and backend == "mtp"
            and is_mtp_model(model_path)):
        vram_mtp_kv      = _mtp_draft_kv_mib(meta, context_window, bytes_pe_k, bytes_pe)
        vram_mtp_compute = _MTP_DRAFT_COMPUTE_MIB

    # DFlash: drafter GGUF separado (-md) + segundo contexto pequeno, ambos na
    # GPU. Pesos dominam; KV do rascunho (256 tokens) e cross-attention cabem no
    # compute fixo. Auto-detecta o maior drafter (q8_0 > q4) p/ não subestimar.
    vram_dflash_weights = 0
    vram_dflash_compute = 0
    if gpu_active and backend == "dflash":
        drafters = find_dflash_drafters(model_path)
        if drafters:
            vram_dflash_weights = int(_model_disk_size_mib(drafters[0]) * _WEIGHT_OVERHEAD)
            vram_dflash_compute = _DFLASH_DRAFT_COMPUTE_MIB

    ram_kv       = kv_total_mib - vram_kv
    ram_ssm      = ssm_state_mib - vram_ssm
    # --cache-ram é um TETO do prompt cache, não uma alocação: vale 0 no launch
    # e o llama.cpp despeja entradas pra ficar sob o limite. Some-lo a ram_total
    # produzia ESTOURA falso. Mantido só como informação, fora do total.
    cache_ram_mib = cache_ram if mode == "server" else 0

    # --n-cpu-moe N força os tensores MoE das N primeiras layers pra CPU.
    # Movemos só o peso dos `ffn_*_exps` (medido na meta) de VRAM pra RAM —
    # gates, normas e atenção ficam onde -ngl manda.
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
        "vram_dflash_weights": vram_dflash_weights,
        "vram_dflash_compute": vram_dflash_compute,
        "ram_weights":      ram_weights,
        "ram_kv":           ram_kv,
        "ram_ssm":          ram_ssm,
        "cache_ram":        cache_ram_mib,
        "moe_offload_mib":  moe_offload_mib,
        "vram_total":       (vram_weights + vram_kv + vram_compute + vram_ssm
                             + vram_mmproj + vram_mtp_kv + vram_mtp_compute
                             + vram_dflash_weights + vram_dflash_compute),
        "ram_total":        ram_weights + ram_kv + ram_ssm,
        "vram_avail":       _gpu_free_mib(),
        "vram_total_phys":  _gpu_total_mib(),
        "ram_avail":        _ram_total_mib(),
    }


_C_GREEN  = "\x1b[32m"
_C_YELLOW = "\x1b[33m"
_C_RED    = "\x1b[31m"
_C_DIM    = "\x1b[2m"
_C_RESET  = "\x1b[0m"
_VT_ENABLED = False


def _enable_vt_mode() -> None:
    """Habilita VT processing no console do Windows pra ANSI funcionar no conhost."""
    global _VT_ENABLED
    if _VT_ENABLED or sys.platform != "win32":
        _VT_ENABLED = True
        return
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if k.GetConsoleMode(h, ctypes.byref(mode)):
            k.SetConsoleMode(h, mode.value | 0x0004)
    except Exception:
        pass
    _VT_ENABLED = True


def _bar(label: str, used: int, total: int | None, width: int = 32) -> str:
    if not total or total <= 0:
        return f"   {label}  {_human_size(used * _MIB):>10}  (total desconhecido)"
    pct      = used / total
    filled   = int(width * min(pct, 1.0))
    bar_str  = "█" * filled + "░" * (width - filled)
    if   pct >= 1.00: color, tag = _C_RED,    "ESTOURA"
    elif pct >= 0.95: color, tag = _C_RED,    "CRÍTICO"
    elif pct >= 0.80: color, tag = _C_YELLOW, "APERTADO"
    else:             color, tag = _C_GREEN,  "OK"
    return (
        f"   {label}  {color}[{bar_str}]{_C_RESET}  "
        f"{used/1024:5.2f} / {total/1024:5.2f} GB  "
        f"({pct*100:5.1f}%)  {color}{tag}{_C_RESET}"
    )


def show_memory_estimate(
    model_path:     Path,
    backend:        str,
    context_window: int,
    parallel_slots: int,
    kv_cache:       str,
    gpu_layers:     int,
    mmproj:         Path | None,
    cache_ram:      int,
    mode:           str,
    verbose:        bool = False,
    header:         str | None = None,
    n_cpu_moe:      int = 0,
) -> dict:
    """Imprime barras de VRAM e RAM pra config atual. Retorna o dict de estimativa."""
    _enable_vt_mode()
    est = estimate_memory(
        model_path, backend, context_window, parallel_slots,
        kv_cache, gpu_layers, mmproj, cache_ram, mode,
        n_cpu_moe=n_cpu_moe,
    )
    print()
    if header:
        print(f"   {_C_DIM}↳ {header}{_C_RESET}")
    print(_bar("VRAM", est["vram_total"], est["vram_avail"]))
    phys, avail = est.get("vram_total_phys"), est.get("vram_avail")
    if phys and avail and phys - avail >= 256:
        print(
            f"   {_C_DIM}↳ VRAM livre considerada: {avail/1024:.2f} GB de {phys/1024:.2f} GB "
            f"({(phys-avail)/1024:.2f} GB já em uso pelo SO/driver/apps){_C_RESET}"
        )
    print(_bar(" RAM", est["ram_total"],  est["ram_avail"]))
    if verbose:
        moe_note = f" (ncmoe offload: {est['moe_offload_mib']} MiB)" if est.get("moe_offload_mib") else ""
        ssm_note = f" + ssm {est['vram_ssm']}" if est.get("vram_ssm") else ""
        mtp_note = (
            f" + mtp-draft {est['vram_mtp_kv'] + est['vram_mtp_compute']}"
            if est.get("vram_mtp_kv") or est.get("vram_mtp_compute") else ""
        )
        dflash_note = (
            f" + dflash-draft {est['vram_dflash_weights'] + est['vram_dflash_compute']}"
            if est.get("vram_dflash_weights") or est.get("vram_dflash_compute") else ""
        )
        # mmproj: mostra weights+compute separados pra deixar claro qual parte
        # pode ser rejeitada silenciosamente no warmup (compute) e qual sempre
        # fica em VRAM (weights).
        mw, mc = est.get("vram_mmproj_weights", 0), est.get("vram_mmproj_compute", 0)
        if mw or mc:
            mmproj_note = f" + mmproj {mw}w+{mc}c"
        else:
            mmproj_note = ""
        print(
            f"   {_C_DIM}vram = pesos {est['vram_weights']} + kv {est['vram_kv']} "
            f"+ compute {est['vram_compute']}{ssm_note}{mmproj_note}{mtp_note}{dflash_note} MiB{moe_note}{_C_RESET}"
        )
        print(
            f"   {_C_DIM}ram  = pesos {est['ram_weights']} + kv {est['ram_kv']} MiB{_C_RESET}"
        )
        if est.get("cache_ram"):
            print(
                f"   {_C_DIM}prompt cache: teto {est['cache_ram']} MiB "
                f"(--cache-ram — preenchido sob demanda, fora do total){_C_RESET}"
            )
    if not est["meta_ok"]:
        print(f"   {_C_YELLOW}⚠️  Metadados do GGUF não lidos — KV cache não estimado.{_C_RESET}")
    print()
    return est


def _estimate_status(est: dict) -> str:
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
    """Menor `n_cpu_moe` que faz a VRAM caber em `target_vram_pct` do disponível.

    Só faz sentido pra modelos MoE com metadata lida e VRAM total conhecida.
    Retorna 0 quando o modelo não é MoE ou já cabe sem offload.
    """
    meta = _gguf_meta(model_path)
    if not meta or not meta.get("is_moe"):
        return 0
    moe_layers = meta.get("moe_layers_count", 0)
    if moe_layers <= 0:
        return 0

    vram_avail = _gpu_free_mib()
    if not vram_avail:
        return 0
    budget = int(vram_avail * target_vram_pct)

    # Linear (40 layers no Qwen35moe, ~10ms total — não precisa binary search).
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
    """Maior `-ngl` cujo total de VRAM cabe em `target_vram_pct` da VRAM livre.

    Pra modelos densos, tirar camadas inteiras da GPU é o único jeito de
    aliviar VRAM. O teto é `n_layer + 2` (tudo: blocos + output + token_embd);
    `n_layer + 1` adiciona o último bloco e `n_layer` deixa output+embed na
    CPU. Retorna 99 quando não dá pra estimar (sem metadata/VRAM)."""
    meta = _gguf_meta(model_path)
    if not meta:
        return 99
    n_layer = meta["n_layer"]
    vram_avail = _gpu_free_mib()
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


def _ngl_menu_choices(
    model_path:     Path,
    backend:        str,
    context_window: int,
    kv_cache:       str,
    mmproj:         Path | None,
    cache_ram:      int,
    mode:           str,
    n_layer:        int,
    suggested:      int,
) -> list[questionary.Choice]:
    """Menu de -ngl com a projeção de VRAM de cada valor (cabe/estoura + RAM).

    Mostra a sugestão, um passo mais folgado, e os patamares de offload do
    llama.cpp: K==n_layer deixa output+token_embd na CPU; K==n_layer+1 leva
    output pra GPU; K==n_layer+2 leva também o token_embd. A sugestão leva
    a marca [recomendado]."""
    # Inclui os patamares-chave (n_layer, +1, +2) pra usuário ver o trade-off
    # de mover output / token_embd pra GPU. O cap superior é n_layer+2.
    cands: set[int] = {n_layer, n_layer + 1, n_layer + 2, suggested}
    if suggested > 0:
        cands.add(max(0, suggested - max(2, n_layer // 8)))
    ordered = sorted((c for c in cands if 0 <= c <= n_layer + 2), reverse=True)

    icons = {"ok": "🟢", "tight": "🟡", "overflow": "🔴", "unknown": "⚪"}
    out: list[questionary.Choice] = []
    for v in ordered:
        est    = estimate_memory(
            model_path, backend, context_window, 1,
            kv_cache, v, mmproj, cache_ram, mode,
        )
        status = _estimate_status(est)
        avail  = est["vram_avail"] or 0

        # Descritor de offload: o que fica na CPU em cada patamar.
        if v >= n_layer + 2:
            desc = "tudo na GPU (+ output + token_embd)"
        elif v == n_layer + 1:
            desc = "blocos + output (token_embd p/ CPU)"
        elif v == n_layer:
            desc = "blocos (output + embed p/ CPU)"
        elif v <= 0:
            desc = "tudo na CPU"
        else:
            on_ram = max(0, n_layer - v + 1)            # +1 bloco saindo
            unit = "camada" if on_ram == 1 else "camadas"
            desc = f"{on_ram} {unit} (~{est['ram_total']/1024:.1f} GB) p/ RAM"

        if status == "overflow":
            over = est["vram_total"] - (avail or est["vram_total"])
            tail = f"{icons['overflow']} ESTOURA +{over/1024:.1f} GB"
        else:
            tail = f"{icons[status]} cabe  {est['vram_total']/1024:.1f}/{avail/1024:.1f} GB"

        mark = "  [recomendado]" if v == suggested else ""
        out.append(questionary.Choice(f"{v:>3}  —  {desc:<38}  {tail}{mark}", value=v))

    out.append(questionary.Choice("  ✏  Digitar valor customizado...", value=-1))
    return out


def prompt_step_action(label: str, status: str) -> str:
    """Pergunta o que fazer depois de cada escolha. Sempre permite reescolher,
    com default inteligente: 'retry' quando estoura, 'continue' caso contrário.
    Retorna 'retry', 'continue' ou 'abort'."""
    if status == "overflow":
        msg = f"⚠️  Com {label} a memória deve ESTOURAR — o que fazer?"
        default_value = "retry"
    elif status == "tight":
        msg = f"⚠️  Com {label} a memória fica APERTADA — o que fazer?"
        default_value = "continue"
    elif status == "unknown":
        msg = f"❓ Sem como estimar VRAM/RAM pra {label} — o que fazer?"
        default_value = "continue"
    else:
        msg = f"✅ {label} cabe folgado — continuar ou ajustar?"
        default_value = "continue"

    choice = questionary.select(
        msg,
        choices=[
            questionary.Choice("✅ Continuar",                       value="continue"),
            questionary.Choice("🔁 Reescolher esse parâmetro",      value="retry"),
            questionary.Choice("⛔ Cancelar (sair sem subir nada)", value="abort"),
        ],
        default=default_value,
    ).ask()
    return choice or "abort"


# ─── command builder ──────────────────────────────────────────────────────────

def _multi_gpu_flags(
    split_mode:   str | None,
    tensor_split: str | None,
    main_gpu:     int | None,
) -> str:
    """Flags de distribuição entre GPUs (-sm/-ts/-mg). Vazio = default do
    llama.cpp (split layer entre todas as placas, proporcional à VRAM)."""
    out = ""
    if split_mode in ("none", "layer", "row"):
        out += f" -sm {split_mode}"
    if tensor_split:
        ts = tensor_split.strip().replace(" ", "")
        if ts:
            out += f" -ts {ts}"
    if main_gpu is not None and main_gpu >= 0:
        out += f" -mg {main_gpu}"
    return out


def build_command(
    model_path:       Path,
    backend:          str,
    context_window:   int,
    kv_cache:         str,
    flash_attn:       bool,
    gpu_layers:       int,
    parallel_slots:   int,
    batch_size:       int,
    ubatch_size:      int,
    threads_gen:      int,
    threads_batch:    int,
    reasoning_budget: int | None,
    mlock:            bool,
    max_tokens:       int,
    cache_ram:        int,
    ctx_checkpoints:  int,
    n_cpu_moe:        int = 0,
    spec_draft_n_max: int = 2,
    spec_dflash_draft: str | None = None,
    preserve_thinking: bool = False,
    mmproj:           Path | None = None,
    verbose:          bool = False,
    split_mode:       str | None = None,
    tensor_split:     str | None = None,
    main_gpu:         int | None = None,
) -> str:
    alias = make_alias(model_path)
    server_bin = backend_binary(backend, "server")

    if not server_bin.exists():
        print(f"❌ llama-server não encontrado em: {server_bin}")
        print(f"   Compile o backend '{backend}' ou ajuste o caminho em BACKENDS.")
        sys.exit(1)

    # Aviso: modelo MTP rodando em backend que não suporta speculative MTP.
    if is_mtp_model(model_path) and not BACKENDS[backend]["supports_spec_mtp"]:
        print(f"ℹ️  Modelo MTP detectado, mas backend '{backend}' não tem speculative MTP.")
        print("   A head MTP será ignorada pelo loader.")

    cmd = f'"{server_bin}" -m "{model_path}"'

    if mmproj:
        cmd += f' --mmproj "{mmproj}"'

    cmd += f' --alias "{alias}"'
    # -c no llama-server é o KV cache TOTAL, repartido igualmente entre os -np
    # slots. Multiplicar por parallel_slots faz cada slot receber context_window
    # inteiro (semântica "janela por slot" — alinha com o estimador e o aviso).
    cmd += f' -c {context_window * parallel_slots}'
    cmd += f' -ngl {gpu_layers}'
    cmd += _multi_gpu_flags(split_mode, tensor_split, main_gpu)
    if n_cpu_moe and n_cpu_moe > 0:
        cmd += f' --n-cpu-moe {n_cpu_moe}'
    cmd += f' --cache-type-k {kv_cache} --cache-type-v {kv_cache}'
    cmd += ' -fa on' if flash_attn else ' -fa off'
    cmd += f' -b {batch_size} -ub {ubatch_size}'
    cmd += f' -t {threads_gen} -tb {threads_batch}'
    cmd += f' -np {parallel_slots}'
    cmd += f' -n {max_tokens}'

    # ── Samplers (DEFAULTS do servidor; o request do client sobrescreve) ───
    # Resolvidos POR MODELO em app/api/core/sampling.py: preset de código pra
    # não-thinking (0.3/0.8/1.05 — o validado em A/B), preset de raciocínio pra
    # quem tem <think> no template, ou os números do próprio autor quando o
    # generation_config.json veio no download.
    # DRY continua fora: com --dry-allowed-length 4 ele pune a repetição LEGÍTIMA de
    # código (`this.getUser('1'), this.getUser('2')`), forçando o modelo a corromper o
    # texto — dropar caractere, trocar dígito/aspas — e quebrando applyPatch/edição
    # exata (comprovado 2026-06-17). Presence-penalty idem: taxa todo token já visto,
    # sem janela, e induz EOS precoce em sessão longa (A/B multi-turno 2026-06-11).
    cmd += sampling_core.cli_flags(sampling_core.resolve(model_path))
    # Quando o contexto enche, descartar tokens antigos em silêncio apaga o
    # system prompt + schema das tools — é aí que "tools que sempre funcionam"
    # passam a falhar. --no-context-shift devolve erro limpo: o agente compacta
    # e retenta, em vez de degradar sem aviso.
    cmd += ' --no-context-shift'

    if reasoning_budget is not None:
        if reasoning_budget == 0:
            cmd += ' --reasoning off'
        else:
            # --reasoning on liga enable_thinking no template. Obrigatório para
            # modelos cujo template tem thinking OFF por padrão (ex.: Gemma 4);
            # nos que já pensam por padrão (Qwen3) é redundante mas inofensivo.
            cmd += ' --reasoning on'
            cmd += f' --reasoning-budget {reasoning_budget}'

    # preserve_thinking: mantém os blocos <think> de turnos ANTERIORES no
    # histórico renderizado. Sem isso, o template do Qwen3.x só preserva o
    # raciocínio da rodada atual (assistant após a última msg do usuário) e
    # descarta o resto. Só faz sentido com reasoning ligado — quem chama já
    # garante isso. Aspas: cmd.exe repassa a linha; o CRT do llama-server faz
    # o split de argv, então \" vira aspa literal e o JSON chega inteiro.
    if preserve_thinking:
        cmd += ' --chat-template-kwargs "{\\"preserve_thinking\\": true}"'

    if mlock:
        cmd += ' --mlock'

    if BACKENDS[backend]["supports_spec_mtp"] and is_mtp_model(model_path):
        cmd += f' --spec-type draft-mtp --spec-draft-n-max {spec_draft_n_max}'

    # DFlash: drafter GGUF separado na GPU (-md/-ngld). --spec-dflash-default é o
    # preset do fork buun (liga spec-type dflash + tuning n_max=7/p_min=0; exige
    # -md). NÃO passamos -cd: o llama-server deste fork não registra essa opção
    # (rejeita 'invalid argument: -cd') e o preset já auto-seta o ctx do rascunho
    # pra 256. Só entra no backend dflash com drafter.
    if BACKENDS[backend].get("supports_spec_dflash") and spec_dflash_draft and Path(spec_dflash_draft).exists():
        cmd += f' -md "{spec_dflash_draft}" --spec-dflash-default -ngld 99'

    # Prompt cache em RAM do sistema (não VRAM). Valor baixo causa cancelamento
    # do cliente em prompts grandes: cache miss → reprefill de ~100k tokens →
    # >90s de prefill → harness desiste e fecha o SSE. Cada prompt salvo pode
    # ocupar 6-8 GB em ctx 131k.
    cmd += f' --cache-ram {cache_ram}'
    cmd += f' --ctx-checkpoints {ctx_checkpoints}'

    cmd += ' --jinja --metrics --host 0.0.0.0 --port 1234'

    if verbose:
        cmd += ' --verbose'

    return cmd

def build_cli_command(
    model_path:       Path,
    backend:          str,
    context_window:   int,
    kv_cache:         str,
    flash_attn:       bool,
    gpu_layers:       int,
    batch_size:       int,
    ubatch_size:      int,
    threads_gen:      int,
    threads_batch:    int,
    reasoning_budget: int | None,
    mlock:            bool,
    max_tokens:       int,
    n_cpu_moe:        int = 0,
    spec_draft_n_max: int = 2,
    spec_dflash_draft: str | None = None,
    mmproj:           Path | None = None,
    verbose:          bool = False,
    split_mode:       str | None = None,
    tensor_split:     str | None = None,
    main_gpu:         int | None = None,
) -> str:
    cli_bin = backend_binary(backend, "cli")

    if not cli_bin.exists():
        print(f"❌ llama-cli não encontrado em: {cli_bin}")
        print(f"   Compile o backend '{backend}' ou ajuste o caminho em BACKENDS.")
        sys.exit(1)

    if is_mtp_model(model_path) and not BACKENDS[backend]["supports_spec_mtp"]:
        print(f"ℹ️  Modelo MTP detectado, mas backend '{backend}' não tem speculative MTP.")
        print("   A head MTP será ignorada pelo loader.")

    cmd = f'"{cli_bin}" -m "{model_path}"'

    if mmproj:
        cmd += f' --mmproj "{mmproj}"'

    cmd += f' -c {context_window}'
    cmd += f' -ngl {gpu_layers}'
    cmd += _multi_gpu_flags(split_mode, tensor_split, main_gpu)
    if n_cpu_moe and n_cpu_moe > 0:
        cmd += f' --n-cpu-moe {n_cpu_moe}'
    cmd += f' --cache-type-k {kv_cache} --cache-type-v {kv_cache}'
    cmd += ' -fa on' if flash_attn else ' -fa off'
    cmd += f' -b {batch_size} -ub {ubatch_size}'
    cmd += f' -t {threads_gen} -tb {threads_batch}'
    cmd += f' -n {max_tokens}'

    # Samplers por modelo (ver build_command). No CLI interativo não passamos
    # --no-context-shift: numa conversa de terminal é melhor o shift do que a
    # sessão morrer com erro ao encher o contexto.
    cmd += sampling_core.cli_flags(sampling_core.resolve(model_path))

    if reasoning_budget is not None:
        if reasoning_budget == 0:
            cmd += ' --reasoning off'
        else:
            # --reasoning on liga enable_thinking no template. Obrigatório para
            # modelos cujo template tem thinking OFF por padrão (ex.: Gemma 4);
            # nos que já pensam por padrão (Qwen3) é redundante mas inofensivo.
            cmd += ' --reasoning on'
            cmd += f' --reasoning-budget {reasoning_budget}'

    if mlock:
        cmd += ' --mlock'

    if BACKENDS[backend]["supports_spec_mtp"] and is_mtp_model(model_path):
        cmd += f' --spec-type draft-mtp --spec-draft-n-max {spec_draft_n_max}'

    if BACKENDS[backend].get("supports_spec_dflash") and spec_dflash_draft and Path(spec_dflash_draft).exists():
        cmd += f' -md "{spec_dflash_draft}" --spec-dflash-default -ngld 99'

    cmd += ' --conversation'

    if verbose:
        cmd += ' --verbose'

    return cmd

# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    mode: str = questionary.select(
        "⚙️  Modo de execução:",
        choices=MODE_CHOICES,
    ).ask()

    if not mode:
        return

    if mode == "download":
        download_flow()
        return

    if mode == "delete":
        delete_flow()
        return

    if mode == "settings":
        settings_flow()
        return

    if mode == "nvidia-status":
        nvidia_status_flow()
        return

    # ── escolha do backend de llama.cpp (server | cli) ───────────────────────
    backend_choices = backend_choices_for_mode(mode)
    if not any(not c.disabled for c in backend_choices):
        print(f"❌ Nenhum binário do llama.cpp encontrado pro modo '{mode}'.")
        print("   Caminhos esperados:")
        for name, b in BACKENDS.items():
            print(f"   • {name:8s}: {b['server' if mode == 'server' else 'cli']}")
        sys.exit(1)

    backend: str = questionary.select(
        "🧰 Backend (build) de llama.cpp:",
        choices=backend_choices,
    ).ask()
    if not backend:
        return

    models = collect_models()
    roots = get_model_dirs()

    labels = []
    for m in models:
        alias  = make_alias(m)
        mmproj = find_mmproj(m.parent)
        tag    = " | 🖼️ Image" if mmproj else ""
        rel    = rel_to_any_root(m, roots)
        labels.append(f"{alias}{tag}  ({rel})")

    roots_str = ", ".join(str(r) for r in roots)
    print(f"\n📋 {len(models)} modelo(s) encontrado(s) em {roots_str}\n")

    selected_label = questionary.select(
        "➡️  Selecione um modelo:",
        choices=labels,
    ).ask()

    if not selected_label:
        print("Nenhum modelo selecionado.")
        return

    model_path = next(
        (models[i] for i, label in enumerate(labels) if label == selected_label),
        None,
    )

    if model_path is None:
        print("Modelo não encontrado.")
        return

    print()

    # ── config anterior deste (modelo, backend)? ─────────────────────────────
    prev_cfg = load_config(model_path, backend)
    use_prev = False

    if prev_cfg:
        prev_alias = make_alias(Path(prev_cfg.get("model", "")))
        ctx = prev_cfg.get('context_window', '?')
        ctx_str = f"{ctx:,}" if isinstance(ctx, int) else ctx
        rb = prev_cfg.get('reasoning_budget')
        rb_str = str(rb) if rb is not None else "—"
        pt_str = "on" if prev_cfg.get('preserve_thinking') else "off"

        cr = prev_cfg.get('cache_ram')
        cr_str = f"{cr} MiB" if cr is not None else "—"
        cc = prev_cfg.get('ctx_checkpoints')
        cc_str = str(cc) if cc is not None else "—"

        print(f"┌─ Configuração salva para [{backend}] {prev_alias} ────────────")
        ncmoe = prev_cfg.get('n_cpu_moe')
        ncmoe_str = str(ncmoe) if ncmoe else "—"

        print(f"│  Backend         : {backend}")
        print(f"│  Contexto        : {ctx_str}  (por slot)")
        print(f"│  KV Cache        : {prev_cfg.get('kv_cache', '?')}")
        print(f"│  Flash Attention : {prev_cfg.get('flash_attn', '?')}")
        print(f"│  GPU Layers      : {prev_cfg.get('gpu_layers', '?')}")
        print(f"│  CPU MoE Layers  : {ncmoe_str}")
        print(f"│  Slots paralelos : {prev_cfg.get('parallel_slots', '?')}")
        print(f"│  MLock           : {prev_cfg.get('mlock', '?')}")
        print(f"│  Max Tokens      : {prev_cfg.get('max_tokens', '?')}")
        print(f"│  Reasoning Budget: {rb_str}")
        print(f"│  Preserve Think. : {pt_str}")
        print(f"│  Cache RAM       : {cr_str}")
        print(f"│  Ctx Checkpoints : {cc_str}")
        if BACKENDS[backend]["supports_spec_mtp"] and is_mtp_model(model_path):
            print(f"│  Spec MTP n-max  : {prev_cfg.get('spec_draft_n_max', 2)}")
        if BACKENDS[backend].get("supports_spec_dflash") and prev_cfg.get("spec_dflash_draft"):
            print(f"│  DFlash drafter  : {Path(prev_cfg['spec_dflash_draft']).name}")
        print(f"│  Verbose         : {prev_cfg.get('verbose', False)}")
        print("└───────────────────────────────────────────────────────────")
        print()
        if questionary.confirm("   Usar a configuração salva?", default=True).ask():
            use_prev = True
            context_window   = prev_cfg.get("context_window", 65_536)
            kv_cache         = prev_cfg.get("kv_cache", "turbo4")
            kv_cache, warn   = remap_kv_for_backend(kv_cache, backend, mode)
            if warn:
                print(f"   ⚠️  {warn}")
            flash_attn       = prev_cfg.get("flash_attn", True)
            gpu_layers       = prev_cfg.get("gpu_layers", 99)
            n_cpu_moe        = int(prev_cfg.get("n_cpu_moe", 0) or 0)
            parallel_slots   = prev_cfg.get("parallel_slots", 1)
            reasoning_budget = prev_cfg.get("reasoning_budget")
            mlock            = prev_cfg.get("mlock", True)
            max_tokens       = prev_cfg.get("max_tokens", 8_192)
            batch_size       = prev_cfg.get("batch_size", 2_048)
            ubatch_size      = prev_cfg.get("ubatch_size", 512)
            threads_gen      = prev_cfg.get("threads_gen", CPU_THREADS_GEN)
            threads_batch    = prev_cfg.get("threads_batch", CPU_THREADS_BATCH)
            cache_ram        = prev_cfg.get("cache_ram", 24_576)
            ctx_checkpoints  = prev_cfg.get("ctx_checkpoints", 8)
            spec_draft_n_max = int(prev_cfg.get("spec_draft_n_max", 2) or 2)
            # Revalida o drafter salvo; se sumiu, re-detecta (auto se houver um só).
            spec_dflash_draft = prev_cfg.get("spec_dflash_draft")
            if BACKENDS[backend].get("supports_spec_dflash"):
                if not (spec_dflash_draft and Path(spec_dflash_draft).exists()):
                    d = select_dflash_drafter(model_path)
                    spec_dflash_draft = str(d) if d else None
            else:
                spec_dflash_draft = None
            verbose          = prev_cfg.get("verbose", False)
            split_mode       = prev_cfg.get("split_mode")
            tensor_split     = prev_cfg.get("tensor_split")
            main_gpu         = prev_cfg.get("main_gpu")
            use_mmproj: Path | None = None
            if prev_cfg.get("mmproj"):
                p = Path(prev_cfg["mmproj"])
                use_mmproj = p if p.exists() else None

            show_memory_estimate(
                model_path, backend, context_window, parallel_slots,
                kv_cache, gpu_layers, use_mmproj, cache_ram, mode,
                verbose=True, header="estimativa da configuração salva",
                n_cpu_moe=n_cpu_moe,
            )

    if not use_prev:
        # ── mmproj ───────────────────────────────────────────────────────────
        mmproj     = find_mmproj(model_path.parent)
        use_mmproj: Path | None = None
        if mmproj:
            if questionary.confirm("🖼️  Habilitar suporte a imagem (--mmproj)?", default=True).ask():
                use_mmproj = mmproj

        # ── seção 1: contexto e memória ──────────────────────────────────────
        while True:
            context_window = ask_context_window()
            est = show_memory_estimate(
                model_path, backend, context_window, parallel_slots=1,
                kv_cache="q8_0", gpu_layers=99, mmproj=use_mmproj,
                cache_ram=8_192, mode=mode,
                header=f"preliminar — ctx={context_window:,} (kv=q8_0, slots=1)",
            )
            action = prompt_step_action(f"ctx={context_window:,}", _estimate_status(est))
            if action == "retry":    continue
            if action == "abort":    return
            break

        while True:
            kv_cache = questionary.select(
                "🗜️  Cache KV (K e V):",
                choices=kv_cache_choices_for(backend, mode),
            ).ask()
            if kv_cache is None:
                return
            est = show_memory_estimate(
                model_path, backend, context_window, parallel_slots=1,
                kv_cache=kv_cache, gpu_layers=99, mmproj=use_mmproj,
                cache_ram=8_192, mode=mode,
                header=f"com kv={kv_cache}",
            )
            action = prompt_step_action(f"kv={kv_cache}", _estimate_status(est))
            if action == "retry":    continue
            if action == "abort":    return
            break

        flash_attn = questionary.confirm("⚡ Flash Attention (-fa)?", default=True).ask()

        # ── ngl: camadas na GPU ──────────────────────────────────────────────
        # Denso → menu com a projeção de VRAM de cada valor; default = o máximo
        # que cabe (tirar camada inteira é o único alívio de VRAM num denso).
        # MoE  → fica livre (default 99); quem ajusta VRAM é o -ncmoe adiante.
        _ngl_meta    = _gguf_meta(model_path)
        _ngl_is_moe  = bool(_ngl_meta and _ngl_meta.get("is_moe"))
        _ngl_n_layer = _ngl_meta["n_layer"] if _ngl_meta else 0
        if _ngl_is_moe:
            print("   ℹ️  Modelo MoE — deixe -ngl em 99; o alívio de VRAM é o -ncmoe (próximo passo).")

        while True:
            if _ngl_is_moe or not _ngl_n_layer:
                gpu_layers = ask_int("🎮 Camadas na GPU (-ngl)", 99)
            else:
                # Teto = n_layer + 2 (todos os blocos + output + token_embd).
                # Acima disso o llama.cpp não offloada mais nada — então clampa
                # a sugestão pra mostrar "tudo na GPU" como max no menu.
                suggested_ngl = min(
                    suggest_n_gpu_layers(
                        model_path, backend, context_window, 1,
                        kv_cache, use_mmproj, 8_192, mode,
                    ),
                    _ngl_n_layer + 2,
                )
                picked = questionary.select(
                    f"🎮 Camadas na GPU (-ngl):  denso · {_ngl_n_layer} camadas",
                    choices=_ngl_menu_choices(
                        model_path, backend, context_window, kv_cache,
                        use_mmproj, 8_192, mode, _ngl_n_layer, suggested_ngl,
                    ),
                    default=suggested_ngl,
                ).ask()
                if picked is None:
                    return
                gpu_layers = (
                    ask_int("   Valor customizado", suggested_ngl)
                    if picked == -1 else picked
                )

            est = show_memory_estimate(
                model_path, backend, context_window, parallel_slots=1,
                kv_cache=kv_cache, gpu_layers=gpu_layers, mmproj=use_mmproj,
                cache_ram=8_192, mode=mode,
                header=f"com ngl={gpu_layers} (split vram/ram conforme camadas)",
            )
            action = prompt_step_action(f"ngl={gpu_layers}", _estimate_status(est))
            if action == "retry":    continue
            if action == "abort":    return
            break

        # ── Multi-GPU: split entre placas (-sm/-ts/-mg) ──────────────────────
        # Só pergunta quando há 2+ GPUs. None em tudo = default do llama.cpp
        # (split layer proporcional à VRAM), que já usa todas as placas.
        split_mode:   str | None = None
        tensor_split: str | None = None
        main_gpu:     int | None = None
        _gpus = _gpu_list()
        if len(_gpus) >= 2:
            print()
            print("🖥️  Múltiplas GPUs detectadas:")
            for idx, name, tot in _gpus:
                print(f"     [{idx}] {name}  ({tot/1024:.0f} GB)")
            split_choice = questionary.select(
                "🔀 Como repartir o modelo entre as GPUs?",
                choices=[
                    questionary.Choice("layer — repartir camadas entre todas (default, recomendado)", value="layer"),
                    questionary.Choice("none — carregar tudo numa GPU só (escolho qual)",            value="none"),
                    questionary.Choice("row — split por linha (raro; pode ser melhor em NVLink)",    value="row"),
                    questionary.Choice("proporção custom (-ts, ex.: 1,1)",                            value="custom"),
                ],
                default="layer",
            ).ask()
            if split_choice is None:
                return
            if split_choice == "none":
                split_mode = "none"
                main_gpu = ask_int(f"   GPU principal (-mg, 0..{len(_gpus)-1})", _gpus[0][0])
                main_gpu = max(0, min(main_gpu, len(_gpus) - 1))
            elif split_choice == "row":
                split_mode = "row"
            elif split_choice == "custom":
                split_mode = "layer"
                tensor_split = questionary.text(
                    f"   Proporção -ts ({len(_gpus)} valores separados por vírgula)",
                    default=",".join("1" for _ in _gpus),
                ).ask() or None
            # "layer" puro deixa tudo None (é o default do llama.cpp).

        # ── MoE: --n-cpu-moe N (só pra modelos com experts) ──────────────────
        n_cpu_moe: int = 0
        _meta = _gguf_meta(model_path)
        if _meta and _meta.get("is_moe"):
            moe_layers   = _meta.get("moe_layers_count", 0)
            per_layer    = _meta.get("moe_per_layer_mib", 0)
            exp_count    = _meta.get("expert_count", 0)
            exp_used     = _meta.get("expert_used_count", 0)
            suggested    = suggest_n_cpu_moe(
                model_path, backend, context_window, 1,
                kv_cache, gpu_layers, use_mmproj, 8_192, mode,
            )
            print()
            print(f"🧠 Modelo MoE detectado — {exp_count} experts ({exp_used} ativos/token), "
                  f"{moe_layers} camadas × ~{per_layer} MiB de pesos MoE.")
            print(f"   Sugestão pra caber em VRAM: --n-cpu-moe {suggested}  "
                  f"(0 = tudo em GPU, {moe_layers} = todos os experts em CPU).")
            while True:
                n_cpu_moe = ask_int(
                    f"   Camadas MoE pra CPU (-ncmoe, 0..{moe_layers})",
                    suggested,
                )
                n_cpu_moe = max(0, min(n_cpu_moe, moe_layers))
                est = show_memory_estimate(
                    model_path, backend, context_window, parallel_slots=1,
                    kv_cache=kv_cache, gpu_layers=gpu_layers, mmproj=use_mmproj,
                    cache_ram=8_192, mode=mode,
                    header=f"com ncmoe={n_cpu_moe} ({n_cpu_moe}×~{per_layer} MiB pra CPU)",
                    n_cpu_moe=n_cpu_moe,
                )
                action = prompt_step_action(f"ncmoe={n_cpu_moe}", _estimate_status(est))
                if action == "retry":    continue
                if action == "abort":    return
                break

        print()

        # ── seção 2: paralelismo e slots (apenas server) ─────────────────────
        parallel_slots: int = 1
        cache_ram: int = 8_192          # default llama-server (só usado em server mode)
        ctx_checkpoints: int = 8        # default razoável (só usado em server mode)
        if mode == "server":
            while True:
                parallel_slots = questionary.select(
                    "🔀 Slots paralelos (-np):",
                    choices=PARALLEL_CHOICES,
                ).ask()
                if parallel_slots is None:
                    return

                kv_total = context_window * parallel_slots
                if parallel_slots > 1:
                    print(f"   ℹ️  {context_window:,} tokens/slot × {parallel_slots} slots → -c {kv_total:,} (KV cache total).")
                if kv_total > 131_072:
                    print(f"   ⚠️  {kv_total:,} tokens no KV cache — verifique a VRAM disponível.")

                est = show_memory_estimate(
                    model_path, backend, context_window, parallel_slots,
                    kv_cache, gpu_layers, use_mmproj, cache_ram=8_192, mode=mode,
                    header=f"com {parallel_slots} slot(s) — KV × {parallel_slots}",
                    n_cpu_moe=n_cpu_moe,
                )
                action = prompt_step_action(f"{parallel_slots} slot(s) paralelos", _estimate_status(est))
                if action == "retry":    continue
                if action == "abort":    return
                break

            print()
            cache_ram = questionary.select(
                "💾 Prompt cache em RAM (--cache-ram, em MiB):",
                choices=CACHE_RAM_CHOICES,
            ).ask()
            show_memory_estimate(
                model_path, backend, context_window, parallel_slots,
                kv_cache, gpu_layers, use_mmproj, cache_ram, mode,
                verbose=True,
                header=f"com cache-ram={cache_ram} MiB — estimativa final",
                n_cpu_moe=n_cpu_moe,
            )

            ctx_checkpoints = questionary.select(
                "📸 KV checkpoints por slot (--ctx-checkpoints):",
                choices=CTX_CHECKPOINTS_CHOICES,
            ).ask()

        # ── seção 3: reasoning (apenas modelos thinking) ─────────────────────
        reasoning_budget: int | None = None
        if is_thinking_model(model_path):
            print()
            reasoning_budget = questionary.select(
                "🧠 Reasoning / thinking  (0 = --reasoning off, N = --reasoning-budget N):",
                choices=REASONING_BUDGET_CHOICES,
            ).ask()

            if mode == "server" and reasoning_budget is not None and 0 < reasoning_budget <= 1_024:
                print(
                    f"   ⚠️  Budget {reasoning_budget} é baixo p/ agentes (Claude Code / OpenCode / Qwen Code).\n"
                    f"      Sintoma: modelo diz 'Agora vou fazer X...' e para — budget esgota durante\n"
                    f"      o <think>, llama.cpp força </think>, sobra pouco espaço pra emitir o tool call.\n"
                    f"      Recomendado: 4096 (agentes de código) ou -1 (sem limite)."
                )

        print()

        # ── speculative decoding MTP (só backend mtp + modelo com head MTP) ──
        spec_draft_n_max = 2
        if BACKENDS[backend]["supports_spec_mtp"] and is_mtp_model(model_path):
            spec_draft_n_max = questionary.select(
                "🔮 MTP — tokens de rascunho por passo (--spec-draft-n-max):",
                choices=SPEC_DRAFT_N_MAX_CHOICES,
                default=2,
            ).ask()
            if spec_draft_n_max is None:
                return
            print()

        # ── speculative decoding DFlash (só backend dflash + drafter GGUF) ──
        spec_dflash_draft: str | None = None
        if BACKENDS[backend].get("supports_spec_dflash"):
            d = select_dflash_drafter(model_path)
            spec_dflash_draft = str(d) if d else None
            print()

        # ── seção 4: otimizações de geração ──────────────────────────────────
        mlock = questionary.confirm(
            "🔒 Travar modelo na RAM? (--mlock)",
            default=False,
        ).ask()

        max_tokens: int = questionary.select(
            "📤 Máx tokens por resposta (-n):",
            choices=MAX_TOKENS_CHOICES,
        ).ask()

        print()

        # ── seção 5: avançado (batch / threads) ──────────────────────────────
        if questionary.confirm("⚙️  Configurar batch/threads avançado?", default=False).ask():
            print()
            batch_size: int = questionary.select(
                "   Batch size físico (-b):",
                choices=BATCH_CHOICES,
            ).ask()
            ubatch_size: int = questionary.select(
                "   Micro-batch size (-ub):",
                choices=UBATCH_CHOICES,
            ).ask()
            threads_gen   = ask_int("   Threads geração (-t)", CPU_THREADS_GEN)
            threads_batch = ask_int("   Threads batch   (-tb)", CPU_THREADS_BATCH)
        else:
            batch_size    = 2_048
            ubatch_size   = 512
            threads_gen   = CPU_THREADS_GEN
            threads_batch = CPU_THREADS_BATCH

        verbose = questionary.confirm(
            "🪵 Verbose (--verbose)? Loga prompt renderizado e mais detalhes",
            default=False,
        ).ask()

    # ── remap KV se o backend não suporta o tipo escolhido (ex: turbo* em vanilla) ──
    new_kv, warning = remap_kv_for_backend(kv_cache, backend, mode)
    if warning:
        print(f"⚠️  {warning}")
        kv_cache = new_kv

    # ── preserve_thinking ────────────────────────────────────────────────────
    # Pergunta se deve ligar --chat-template-kwargs '{"preserve_thinking": true}'
    # — mas só quando faz sentido, isto é, quando as três condições batem:
    #   1. modo server (só ali entra --jinja, que o kwarg exige);
    #   2. reasoning ligado (budget != None e != 0 — sem <think>, não há o que
    #      preservar);
    #   3. o template do modelo realmente expõe a variável `preserve_thinking`.
    # Fora desses casos não há o que decidir e fica OFF. O default do prompt é
    # ON (a decisão automática antiga). O binário ainda é checado via --help; se
    # faltar suporte, o auto-degrade remove a flag na primeira queda.
    preserve_thinking = False
    if mode == "server" and reasoning_budget not in (None, 0):
        _pt_meta = _gguf_meta(model_path)
        if (_pt_meta and _pt_meta.get("supports_preserve_thinking")
                and _backend_supports_chat_template_kwargs(backend, "server")):
            preserve_thinking = bool(questionary.confirm(
                "🧠 preserve_thinking? (mantém os blocos <think> de turnos "
                "anteriores no histórico, via --chat-template-kwargs)",
                default=True,
            ).ask())
            print(f"   🧠 preserve_thinking: {'ON' if preserve_thinking else 'OFF'}")

    # ── salva configuração ───────────────────────────────────────────────────
    cfg = {
        "model":          str(model_path),
        "backend":        backend,
        "context_window": context_window,
        "kv_cache":       kv_cache,
        "flash_attn":     flash_attn,
        "gpu_layers":     gpu_layers,
        "n_cpu_moe":      n_cpu_moe,
        "parallel_slots": parallel_slots,
        "reasoning_budget": reasoning_budget,
        "preserve_thinking": preserve_thinking,
        "mlock":          mlock,
        "max_tokens":     max_tokens,
        "batch_size":     batch_size,
        "ubatch_size":    ubatch_size,
        "threads_gen":    threads_gen,
        "threads_batch":  threads_batch,
        "cache_ram":      cache_ram,
        "ctx_checkpoints": ctx_checkpoints,
        "spec_draft_n_max": spec_draft_n_max,
        "spec_dflash_draft": spec_dflash_draft,
        "split_mode":     split_mode,
        "tensor_split":   tensor_split,
        "main_gpu":       main_gpu,
        "mmproj":         str(use_mmproj) if use_mmproj else None,
        "verbose":        verbose,
    }
    save_config(cfg)

    print()

    # ── build e execução ─────────────────────────────────────────────────────
    if mode == "server":
        cmd = build_command(
            model_path=model_path,
            backend=backend,
            context_window=context_window,
            kv_cache=kv_cache,
            flash_attn=flash_attn,
            gpu_layers=gpu_layers,
            parallel_slots=parallel_slots,
            batch_size=batch_size,
            ubatch_size=ubatch_size,
            threads_gen=threads_gen,
            threads_batch=threads_batch,
            reasoning_budget=reasoning_budget,
            mlock=mlock,
            max_tokens=max_tokens,
            cache_ram=cache_ram,
            ctx_checkpoints=ctx_checkpoints,
            n_cpu_moe=n_cpu_moe,
            spec_draft_n_max=spec_draft_n_max,
            spec_dflash_draft=spec_dflash_draft,
            preserve_thinking=preserve_thinking,
            mmproj=use_mmproj,
            verbose=verbose,
            split_mode=split_mode,
            tensor_split=tensor_split,
            main_gpu=main_gpu,
        )
    else:
        cmd = build_cli_command(
            model_path=model_path,
            backend=backend,
            context_window=context_window,
            kv_cache=kv_cache,
            flash_attn=flash_attn,
            gpu_layers=gpu_layers,
            batch_size=batch_size,
            ubatch_size=ubatch_size,
            threads_gen=threads_gen,
            threads_batch=threads_batch,
            reasoning_budget=reasoning_budget,
            mlock=mlock,
            max_tokens=max_tokens,
            n_cpu_moe=n_cpu_moe,
            spec_draft_n_max=spec_draft_n_max,
            spec_dflash_draft=spec_dflash_draft,
            mmproj=use_mmproj,
            verbose=verbose,
            split_mode=split_mode,
            tensor_split=tensor_split,
            main_gpu=main_gpu,
        )
    alias      = make_alias(model_path)
    mode_label = "llama-server" if mode == "server" else "llama-cli"

    img_status     = f"mmproj: {use_mmproj.name}" if use_mmproj else "mmproj: off"
    budget_status  = f"budget: {reasoning_budget}" if reasoning_budget is not None else None
    verbose_status = f"verbose: {'on' if verbose else 'off'}"

    ncmoe_tail = f"  |  ncmoe: {n_cpu_moe}" if n_cpu_moe else ""
    print(f"🚀 Iniciando [{mode_label}] backend={backend}: {alias}")
    if mode == "server" and parallel_slots > 1:
        print(f"   ctx: {context_window:,}/slot  (-c {context_window * parallel_slots:,} total)  |  kv: {kv_cache}  |  fa: {flash_attn}  |  ngl: {gpu_layers}{ncmoe_tail}")
    else:
        print(f"   ctx: {context_window:,}  |  kv: {kv_cache}  |  fa: {flash_attn}  |  ngl: {gpu_layers}{ncmoe_tail}")
    if mode == "server":
        print(f"   np: {parallel_slots}  |  b: {batch_size}  |  ub: {ubatch_size}  |  t: {threads_gen}/{threads_batch}  |  mlock: {mlock}")
        print(f"   cache-ram: {cache_ram} MiB  |  ctx-checkpoints: {ctx_checkpoints}")
    else:
        print(f"   b: {batch_size}  |  ub: {ubatch_size}  |  t: {threads_gen}/{threads_batch}  |  mlock: {mlock}")
    if budget_status:
        print(f"   {budget_status}  |  n: {max_tokens}  |  {img_status}  |  {verbose_status}")
    else:
        print(f"   n: {max_tokens}  |  {img_status}  |  {verbose_status}")
    if BACKENDS[backend]["supports_spec_mtp"] and is_mtp_model(model_path):
        print(f"   spec-mtp: draft-mtp  |  spec-draft-n-max: {spec_draft_n_max}")
    if BACKENDS[backend].get("supports_spec_dflash") and spec_dflash_draft:
        print(f"   spec-dflash: {Path(spec_dflash_draft).name}  (--spec-dflash-default)")
    if preserve_thinking:
        print('   chat-template-kwargs: {"preserve_thinking": true}')
    print()
    print(f"   {cmd}\n")

    show_memory_estimate(
        model_path, backend, context_window, parallel_slots,
        kv_cache, gpu_layers, use_mmproj, cache_ram, mode,
        verbose=True, header="estimativa antes do launch",
        n_cpu_moe=n_cpu_moe,
    )

    if mode == "server":
        stop_lms_server_if_running()

    try:
        if mode == "server":
            run_server_resiliently(
                cfg=cfg,
                mode_label=mode_label,
                use_mmproj=use_mmproj,
            )
        else:
            rc, _, _ = run_capturing(cmd)
            if rc != 0:
                print(f"❌ {mode_label} encerrou com código {rc}.")
    except KeyboardInterrupt:
        print("\n\n⛔ Interrompido pelo usuário.")
    except FileNotFoundError:
        print(f"❌ {mode_label} não encontrado.")
    except Exception as e:
        print(f"❌ Erro ao executar: {e}")


if __name__ == "__main__":
    main()
