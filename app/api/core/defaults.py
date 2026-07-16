"""Config recomendada para um modelo — o que o botão "defaults do modelo" preenche.

Junta num lugar só tudo que dá pra DERIVAR do modelo em vez de o usuário adivinhar:
metadados do GGUF (thinking, híbrido SSM, MoE, contexto de treino), VRAM real da
máquina (estimador) e os samplers do autor. Cada escolha vem com uma nota — o
usuário tem que poder discordar do palpite, e pra isso precisa saber o porquê.
"""
from pathlib import Path

from . import estimator, gguf, nvidia, sampling
from .constants import CONTEXT_OPTIONS
from .models_repo import find_mmproj

# Teto de saída. NUNCA -1: sem cap, um modelo que degenera gera pra sempre — foi
# assim que o loop agêntico do híbrido SSM virou "travou e não solta" em vez de
# "respondeu truncado". O cap é o freio de mão, não enfeite.
DEFAULT_MAX_TOKENS = 16_384

# Reasoning: budget FINITO. O default do llama.cpp é -1 (pensa sem fim) e arch
# qwen35/híbrido SSM trava em loop agêntico assim (2026-05-17, e de novo com Ornith
# em 2026-07-14).
#
# 8192 e não 4096: o que impede o <think> infinito é o budget ser finito. Apertar mais
# reduz o custo de um runaway (metade dos tokens até o corte), não a chance dele — e
# custa raciocínio legítimo. Amostra pequena (Qwen3.6-27B, 2026-07-14, uma execução por
# cenário — indicativo, não distribuição): turno agêntico gastou 37 e 119 tokens de
# thinking; matemática e arquitetura pesadas, 4106 e 3994. Logo 4096 fica em cima da
# faixa das tarefas difíceis e 8192 dá ~2x de folga sobre o maior valor observado, com
# a mesma proteção. Quem precisa de latência baixa e contenção agressiva de runaway
# ainda pode escolher 4096 na UI.
#
# Não subir pra 16384 sem subir DEFAULT_MAX_TOKENS junto: budget e resposta dividem o
# mesmo orçamento de geração, e 16384/16384 deixaria o thinking comer a resposta.
DEFAULT_REASONING_BUDGET = 8_192
REASONING_BUDGET_MESSAGE = "Time to stop thinking and answer or call a tool."


def _fits(model_path: Path, backend: str, ctx: int, kv: str, mmproj: Path | None,
          cache_ram: int, vram_budget: int) -> bool:
    try:
        est = estimator.estimate_memory(
            model_path, backend, ctx, 1, kv, 99, mmproj, cache_ram, "server",
        )
    except Exception:
        return False
    return est["vram_total"] <= vram_budget


def suggest_config(model_path: Path, backend: str = "vanilla", mode: str = "server") -> dict:
    """Config completa + notas explicando cada escolha não-óbvia."""
    meta     = gguf.read_meta(model_path)
    notes:   list[str] = []

    thinking   = bool(meta and meta.get("has_thinking_template"))
    hybrid_ssm = bool(meta and meta.get("ssm_state_per_layer_mib"))
    is_moe     = bool(meta and meta.get("is_moe"))
    ctx_train  = (meta or {}).get("n_ctx_train")

    mmproj = find_mmproj(model_path.parent)

    # KV: turbo só existe no build turboquant; nos demais q8_0 é o teto de
    # qualidade que ainda economiza metade da VRAM do f16.
    kv_cache = "turbo4" if backend == "turbo" else "q8_0"

    cache_ram       = 24_576
    ctx_checkpoints = 32 if hybrid_ssm else 8
    if hybrid_ssm:
        notes.append(
            "híbrido SSM: ctx-checkpoints 32 + cache-ram grande habilitam reuso de KV "
            "entre turnos (sem isso o prefill é refeito a cada rodada)."
        )

    # Contexto: o maior da lista que (a) não passa do treino e (b) cabe na VRAM
    # física com o modelo inteiro na GPU. Sem isso a sugestão de -ngl depois vira
    # um jogo de empurra.
    vram_phys = nvidia.gpu_total_mib() or 0
    budget    = int(vram_phys * 0.92) if vram_phys else 0
    candidates = sorted((o["value"] for o in CONTEXT_OPTIONS), reverse=True)
    if ctx_train:
        candidates = [c for c in candidates if c <= ctx_train]
    context_window = candidates[-1] if candidates else 8_192
    for c in candidates:
        if not budget or _fits(model_path, backend, c, kv_cache, mmproj, cache_ram, budget):
            context_window = c
            break
    if ctx_train and context_window >= ctx_train:
        notes.append(f"contexto no teto de treino do modelo ({ctx_train:,} tokens).")

    # -ngl e -ncmoe: o estimador já sabe fazer isso; aqui só encadeamos na ordem
    # certa (ngl com o ctx escolhido, depois ncmoe com o ngl escolhido).
    try:
        gpu_layers = estimator.suggest_n_gpu_layers(
            model_path, backend, context_window, 1, kv_cache, mmproj, cache_ram, mode,
        )
    except Exception:
        gpu_layers = 99

    n_cpu_moe = 0
    if is_moe:
        try:
            n_cpu_moe = estimator.suggest_n_cpu_moe(
                model_path, backend, context_window, 1, kv_cache, gpu_layers,
                mmproj, cache_ram, mode,
            )
        except Exception:
            n_cpu_moe = 0
        if n_cpu_moe:
            notes.append(
                f"MoE não cabe inteiro na VRAM: {n_cpu_moe} camadas de experts vão pra CPU "
                "(-ncmoe). Só os experts — a atenção fica toda na GPU."
            )

    reasoning_budget = DEFAULT_REASONING_BUDGET if thinking else None
    if thinking:
        notes.append(
            f"modelo de raciocínio (<think> no chat template): reasoning-budget "
            f"{DEFAULT_REASONING_BUDGET:,} em vez de ilimitado — budget aberto já travou em "
            "loop agêntico neste tipo de arch. Numa amostra pequena, turno agêntico gastou "
            "~100 tokens de thinking e tarefa difícil ~4k; o teto só morde no segundo caso."
        )

    # Multi-GPU: layer split. row rende ~metade no 3090+3090 Ti (PCIe gen3 x4, sem
    # NVLink) — validado 2026-06-15.
    gpu_count  = nvidia.gpu_count() or 1
    split_mode = "layer" if gpu_count > 1 else None
    if gpu_count > 1:
        notes.append("2+ GPUs: split por camada (-sm layer); row é mais lento sem NVLink.")

    sampler = sampling.resolve(model_path)
    notes.append({
        "generation_config": "samplers do autor (generation_config.json do repo).",
        "template":          f"samplers: preset {'reasoning' if thinking else 'código'} (derivado do chat template).",
        "default":           "samplers: preset conservador — não deu pra ler o GGUF.",
        "manual":            "samplers fixados por você.",
    }[sampler["source"]])

    cfg = {
        "model":            str(model_path),
        "backend":          backend,
        "context_window":   context_window,
        "kv_cache":         kv_cache,
        "flash_attn":       True,
        "gpu_layers":       gpu_layers,
        "n_cpu_moe":        n_cpu_moe,
        "parallel_slots":   1,
        # Pool de KV único pros slots -np (o -kvu do llama.cpp, ligado por
        # padrão no LM Studio também). Não é opção de UI: só o auto-degrade
        # desliga, quando o build não tem a flag.
        "kv_unified":       True,
        "reasoning_budget": reasoning_budget,
        "preserve_thinking": False,
        "mlock":            False,
        "max_tokens":       DEFAULT_MAX_TOKENS,
        "batch_size":       2_048,
        "ubatch_size":      512,
        "cache_ram":        cache_ram,
        "ctx_checkpoints":  ctx_checkpoints,
        "spec_draft_n_max": 2,
        "mmproj":           str(mmproj) if mmproj else None,
        "verbose":          False,
        # Samplers ficam em auto (null): quem resolve é o sampling.py, no launch.
        "temp":             None,
        "top_p":            None,
        "top_k":            None,
        "min_p":            None,
        "repeat_penalty":   None,
        "sampler_source":   None,
        "split_mode":       split_mode,
        "tensor_split":     None,
        "main_gpu":         None,
    }
    return {"config": cfg, "notes": notes, "sampling": sampler}
