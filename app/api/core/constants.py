"""Constantes globais: paths, backends, listas de opções, padrões de erro.

Espelha o topo do models.py original. As listas de "choices" são exportadas
como dicts JSON-serializáveis pro frontend mostrar as mesmas opções da CLI.
"""
from pathlib import Path

MODELS_DIR  = Path(r"C:\Users\PC\.lmstudio\models")
LMS         = Path(r"C:\Users\PC\.lmstudio\bin\lms.exe")
CONFIG_FILE = Path(r"C:\Projetos\llama-cpp-turboquant\last_config.json")
FAIL_HISTORY_FILE = Path(r"C:\Projetos\llama-cpp-turboquant\fail_history.jsonl")
NVIDIA_SMI  = Path(r"C:\Windows\System32\nvidia-smi.exe")

# i5-14600KF: 6P + 8E = 14 núcleos físicos, 20 threads lógicos
CPU_THREADS_GEN   = 14
CPU_THREADS_BATCH = 20

STABILIZE_SECONDS = 30

LLAMA_SERVER_TURBO   = Path(r"C:\Projetos\llama-cpp-turboquant\build\bin\Release\llama-server.exe")
LLAMA_CLI_TURBO      = Path(r"C:\Projetos\llama-cpp-turboquant\build\bin\Release\llama-cli.exe")
LLAMA_SERVER_VANILLA = Path(r"C:\Projetos\llama-cpp\build\bin\Release\llama-server.exe")
LLAMA_CLI_VANILLA    = Path(r"C:\Projetos\llama-cpp\build\bin\Release\llama-cli.exe")
# MTP nativo no upstream (PR #22673) — fork am17an obsoleto, reusa o build vanilla.
LLAMA_SERVER_MTP     = LLAMA_SERVER_VANILLA
LLAMA_CLI_MTP        = LLAMA_CLI_VANILLA

BACKENDS: dict[str, dict] = {
    "turbo": {
        "label":             "turboquant",
        "description":       "KV turbo2/3/4 (compressão agressiva, ctx longo)",
        "server":            LLAMA_SERVER_TURBO,
        "cli":               LLAMA_CLI_TURBO,
        "supports_spec_mtp": False,
    },
    "vanilla": {
        "label":             "vanilla",
        "description":       "upstream llama.cpp (parsers de chat/tool mais novos)",
        "server":            LLAMA_SERVER_VANILLA,
        "cli":               LLAMA_CLI_VANILLA,
        "supports_spec_mtp": False,
    },
    "mtp": {
        "label":             "mtp",
        "description":       "vanilla upstream + speculative MTP nativo (--spec-type draft-mtp)",
        "server":            LLAMA_SERVER_MTP,
        "cli":               LLAMA_CLI_MTP,
        "supports_spec_mtp": True,
    },
}

# Backends que NÃO têm build próprio — reusam o binário de outro, mudando só o
# perfil de launch. mtp = vanilla upstream com --spec-type draft-mtp (MTP virou
# nativo no PR #22673). Esses não aparecem como diretório configurável no
# Settings e seguem o path (default ou override) do backend-alvo.
BACKEND_BINARY_ALIAS = {"mtp": "vanilla"}

THINKING_KEYWORDS = {"qwen3", "qwq", "deepseek-r1", "thinking", "reasoning", "gemma-4", "gemma4"}

MMPROJ_ERROR_PATTERNS = (
    "unsupported decoder rope type",
    "failed to load multimodal model",
    "mtmd_init_from_file: error",
)
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
# GGUF corrompido / shard faltando / tensor fora dos bounds — caso típico:
# download interrompido de modelo split (`-NNNNN-of-NNNNN.gguf`). Auto-degrade
# não conserta isso, só atrapalha — classificar separado pra dar giveup direto.
MODEL_CORRUPTED_PATTERNS = (
    "data is not within the file bounds",
    "model is corrupted or incomplete",
    "tensor data is corrupted",
    "gguf_init_from_file: invalid magic number",
)
UNSUPPORTED_FLAG_PATTERNS = (
    "unknown argument:",
    "unrecognized argument:",
    "invalid argument:",
)
LOAD_OK_PATTERNS = (
    "listening on",
    "http server listening",
    "main: server is listening",
)

KV_CACHE_OPTIONS = [
    {"value": "f32",    "label": "f32",    "description": "32-bit, sem compressão (só para debug — usa o dobro)"},
    {"value": "f16",    "label": "f16",    "description": "padrão, sem compressão"},
    {"value": "bf16",   "label": "bf16",   "description": "bfloat16, mesma faixa do treino (sem perda)"},
    {"value": "turbo2", "label": "turbo2", "description": "~2-bit KV, compressão extrema (qualidade pode cair)"},
    {"value": "turbo3", "label": "turbo3", "description": "~3-bit KV, alta compressão (ótimo para ctx longo)"},
    {"value": "turbo4", "label": "turbo4", "description": "~4-bit KV, mais qualidade que turbo3 [recomendado]"},
    {"value": "q8_0",   "label": "q8_0",   "description": "leve, quase sem perda de qualidade"},
    {"value": "q5_1",   "label": "q5_1",   "description": "compressão moderada (mais qualidade que q5_0)"},
    {"value": "q5_0",   "label": "q5_0",   "description": "compressão moderada"},
    {"value": "q4_1",   "label": "q4_1",   "description": "compressão alta (mais qualidade que q4_0)"},
    {"value": "iq4_nl", "label": "iq4_nl", "description": "4-bit non-linear (mais qualidade que q4_0, mesmo VRAM)"},
    {"value": "q4_0",   "label": "q4_0",   "description": "compressão alta, economiza mais VRAM"},
]

CONTEXT_OPTIONS = [
    {"value": 4_096,   "label": "4 096 tokens"},
    {"value": 8_192,   "label": "8 192 tokens"},
    {"value": 16_384,  "label": "16 384 tokens"},
    {"value": 32_768,  "label": "32 768 tokens"},
    {"value": 65_536,  "label": "65 536 tokens [recomendado]"},
    {"value": 131_072, "label": "131 072 tokens"},
    {"value": 262_144, "label": "262 144 tokens (256k)"},
]

# Amostra pequena (Qwen3.6-27B, 2026-07-14, UMA execução por cenário — indicativo, não
# distribuição): turno agêntico gastou 37 e 119 tokens de thinking; matemática e
# arquitetura pesadas, 4106 e 3994. Duas leituras: (a) num agente o teto praticamente
# não é alcançado; (b) 4096 fica em cima da faixa das tarefas difíceis.
# A proteção contra o loop infinito vem de o budget ser FINITO. Apertar mais reduz o
# CUSTO de um runaway (metade dos tokens até o corte), não o risco dele — por isso 4096
# continua útil quando latência importa mais que raciocínio profundo.
REASONING_BUDGET_OPTIONS = [
    {"value": 0,       "label": "0 — Não abre <think> (o app emite --reasoning off) — último recurso p/ stall"},
    {"value": 1_024,   "label": "1 024 — Rápido"},
    {"value": 2_048,   "label": "2 048"},
    {"value": 4_096,   "label": "4 096 — corta runaway mais cedo; pode truncar raciocínio pesado"},
    {"value": 8_192,   "label": "8 192 [recomendado — ~2x o maior thinking observado na amostra]"},
    {"value": 16_384,  "label": "16 384 — exige max-tokens maior (o thinking come o espaço da resposta)"},
    {"value": -1,      "label": "-1 — Sem limite (já TRAVOU em loop agêntico em arch qwen35/SSM)"},
]

PARALLEL_OPTIONS = [
    {"value": 1, "label": "1 [recomendado para ctx longo]"},
    {"value": 2, "label": "2 — 2 requisições simultâneas"},
    {"value": 4, "label": "4 — 4 requisições simultâneas"},
]

MAX_TOKENS_OPTIONS = [
    {"value": 8_192,  "label": "8 192 [recomendado — limita runaway/loop]"},
    {"value": 16_384, "label": "16 384"},
    {"value": 4_096,  "label": "4 096"},
    {"value": 2_048,  "label": "2 048"},
    {"value": -1,     "label": "-1 — Ilimitado (sem proteção)"},
]

BATCH_OPTIONS = [
    {"value": 512,   "label": "512"},
    {"value": 1_024, "label": "1 024"},
    {"value": 2_048, "label": "2 048 [default]"},
    {"value": 4_096, "label": "4 096 — máximo throughput de prefill"},
]

UBATCH_OPTIONS = [
    {"value": 128, "label": "128 — menor pico de VRAM no prefill"},
    {"value": 256, "label": "256"},
    {"value": 512, "label": "512 [default]"},
]

CACHE_RAM_OPTIONS = [
    {"value": 2_048,  "label": "2 048 — 2 GB (default antigo)"},
    {"value": 8_192,  "label": "8 192 — 8 GB (default llama-server)"},
    {"value": 16_384, "label": "16 384 — 16 GB (cabe 2-3 prompts grandes)"},
    {"value": 24_576, "label": "24 576 — 24 GB [recomendado — máquinas com 48 GB+]"},
    {"value": 32_768, "label": "32 768 — 32 GB (máximo c/ 48 GB RAM)"},
    {"value": 65_536, "label": "65 536 — 64 GB (servidores dedicados)"},
]

CTX_CHECKPOINTS_OPTIONS = [
    {"value": 0,  "label": "0 — desabilitar (sem restauração de edição)"},
    {"value": 8,  "label": "8 [recomendado — agentes / conversas que só crescem]"},
    {"value": 16, "label": "16 — melhor reuse parcial em conversas com edição"},
    {"value": 32, "label": "32 — default do llama-server (overkill p/ agentes)"},
]

SPEC_DRAFT_N_MAX_OPTIONS = [
    {"value": 1, "label": "1 — rascunho mínimo"},
    {"value": 2, "label": "2 — recomendado p/ MTP (Unsloth) [recomendado]"},
    {"value": 3, "label": "3 — rascunho maior"},
    {"value": 4, "label": "4 — rascunho agressivo (mais rejeição)"},
]
