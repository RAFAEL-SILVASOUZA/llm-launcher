"""Builder das linhas de comando llama-server / llama-cli a partir de uma config."""
import sys
from pathlib import Path

from .backends import backend_binary
from .constants import BACKENDS, CPU_THREADS_BATCH, CPU_THREADS_GEN
from .models_repo import is_mtp_model, make_alias


class BackendBinaryMissing(Exception):
    """Disparado quando o .exe do backend escolhido não existe."""


def build_server_command(
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
    preserve_thinking: bool = False,
    mmproj:           Path | None = None,
    verbose:          bool = False,
    host:             str = "0.0.0.0",
    port:             int = 1234,
) -> str:
    alias = make_alias(model_path)
    server_bin = backend_binary(backend, "server")
    if not server_bin.exists():
        raise BackendBinaryMissing(f"llama-server não encontrado em: {server_bin}")

    cmd = f'"{server_bin}" -m "{model_path}"'
    if mmproj:
        cmd += f' --mmproj "{mmproj}"'

    cmd += f' --alias "{alias}"'
    cmd += f' -c {context_window * parallel_slots}'
    cmd += f' -ngl {gpu_layers}'
    if n_cpu_moe and n_cpu_moe > 0:
        cmd += f' --n-cpu-moe {n_cpu_moe}'
    cmd += f' --cache-type-k {kv_cache} --cache-type-v {kv_cache}'
    cmd += ' -fa on' if flash_attn else ' -fa off'
    cmd += f' -b {batch_size} -ub {ubatch_size}'
    cmd += f' -t {threads_gen} -tb {threads_batch}'
    cmd += f' -np {parallel_slots}'
    cmd += f' -n {max_tokens}'

    # Rede de segurança anti-loop (DEFAULTS do servidor; request pode sobrescrever).
    # Sem isto, cliente com temperature=0 e sem penalidade entra em loop até
    # estourar o contexto. DRY pune só repetição literal, mas precisa de janela
    # limitada: sobre o contexto inteiro (default -1) passa a punir o próprio
    # estilo do modelo em sessão longa e induz EOS precoce a partir de ~8k tokens
    # de saída acumulada (A/B multi-turno 2026-06-11). Pelo mesmo motivo não usar
    # presence-penalty como default — taxa todo token já visto, sem janela.
    cmd += ' --dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 4'
    cmd += ' --dry-penalty-last-n 2048'
    # repeat-penalty mexe no logit ANTES da escolha, então corta loop até em
    # temperature=0 (greedy), onde DRY não vence a margem do argmax. 1.1 é leve.
    cmd += ' --repeat-penalty 1.1 --repeat-last-n 256'
    # Não descarta tokens antigos em silêncio (perde system prompt + tools) quando
    # o contexto enche — devolve erro limpo pro agente compactar e retentar.
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

    if preserve_thinking:
        cmd += ' --chat-template-kwargs "{\\"preserve_thinking\\": true}"'

    if mlock:
        cmd += ' --mlock'

    if BACKENDS[backend]["supports_spec_mtp"] and is_mtp_model(model_path):
        cmd += f' --spec-type draft-mtp --spec-draft-n-max {spec_draft_n_max}'

    cmd += f' --cache-ram {cache_ram}'
    cmd += f' --ctx-checkpoints {ctx_checkpoints}'
    cmd += f' --jinja --metrics --host {host} --port {port}'

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
    mmproj:           Path | None = None,
    verbose:          bool = False,
) -> str:
    cli_bin = backend_binary(backend, "cli")
    if not cli_bin.exists():
        raise BackendBinaryMissing(f"llama-cli não encontrado em: {cli_bin}")

    cmd = f'"{cli_bin}" -m "{model_path}"'
    if mmproj:
        cmd += f' --mmproj "{mmproj}"'

    cmd += f' -c {context_window}'
    cmd += f' -ngl {gpu_layers}'
    if n_cpu_moe and n_cpu_moe > 0:
        cmd += f' --n-cpu-moe {n_cpu_moe}'
    cmd += f' --cache-type-k {kv_cache} --cache-type-v {kv_cache}'
    cmd += ' -fa on' if flash_attn else ' -fa off'
    cmd += f' -b {batch_size} -ub {ubatch_size}'
    cmd += f' -t {threads_gen} -tb {threads_batch}'
    cmd += f' -n {max_tokens}'

    # Rede de segurança anti-loop (ver build_server_command). Sem --no-context-shift
    # no CLI: numa conversa de terminal o shift é melhor que morrer com erro.
    cmd += ' --dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 4'
    cmd += ' --dry-penalty-last-n 2048'
    # repeat-penalty mexe no logit ANTES da escolha, então corta loop até em
    # temperature=0 (greedy), onde DRY não vence a margem do argmax. 1.1 é leve.
    cmd += ' --repeat-penalty 1.1 --repeat-last-n 256'

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

    cmd += ' --conversation'
    if verbose:
        cmd += ' --verbose'
    return cmd


def build_command_from_cfg(cfg: dict, mode: str = "server") -> str:
    """Reconstrói o comando a partir de uma config salva."""
    mmproj_str = cfg.get("mmproj")
    mmproj = Path(mmproj_str) if mmproj_str else None
    model_path = Path(cfg["model"])
    if mode == "server":
        return build_server_command(
            model_path=model_path,
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
            preserve_thinking=cfg.get("preserve_thinking", False),
            mmproj=mmproj,
            verbose=cfg.get("verbose", False),
        )
    return build_cli_command(
        model_path=model_path,
        backend=cfg.get("backend", "turbo"),
        context_window=cfg["context_window"],
        kv_cache=cfg["kv_cache"],
        flash_attn=cfg.get("flash_attn", True),
        gpu_layers=cfg.get("gpu_layers", 99),
        batch_size=cfg.get("batch_size", 2_048),
        ubatch_size=cfg.get("ubatch_size", 512),
        threads_gen=cfg.get("threads_gen", CPU_THREADS_GEN),
        threads_batch=cfg.get("threads_batch", CPU_THREADS_BATCH),
        reasoning_budget=cfg.get("reasoning_budget"),
        mlock=cfg.get("mlock", True),
        max_tokens=cfg.get("max_tokens", 8_192),
        n_cpu_moe=cfg.get("n_cpu_moe", 0),
        spec_draft_n_max=cfg.get("spec_draft_n_max", 2),
        mmproj=mmproj,
        verbose=cfg.get("verbose", False),
    )
