"""Builder das linhas de comando llama-server / llama-cli a partir de uma config."""
import sys
from pathlib import Path

from .backends import backend_binary
from .constants import BACKENDS, CPU_THREADS_BATCH, CPU_THREADS_GEN
from .models_repo import is_mtp_model, make_alias


class BackendBinaryMissing(Exception):
    """Disparado quando o .exe do backend escolhido não existe."""


def _multi_gpu_flags(
    split_mode:   str | None,
    tensor_split: str | None,
    main_gpu:     int | None,
) -> str:
    """Flags de distribuição entre GPUs (-sm/-ts/-mg).

    Vazio quando nada foi setado → llama.cpp usa o default (split-mode layer,
    repartindo as camadas entre todas as GPUs proporcional à VRAM). Só emite
    o que o usuário escolheu explicitamente.
      -sm none|layer|row  como repartir (none = tudo na main-gpu)
      -ts a,b[,...]       proporção por GPU (ex.: "1,1" = meio a meio)
      -mg i               GPU principal (recebe KV/escalares; índice base-0)
    """
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
    split_mode:       str | None = None,
    tensor_split:     str | None = None,
    main_gpu:         int | None = None,
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
    cmd += _multi_gpu_flags(split_mode, tensor_split, main_gpu)
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
    # DRY REMOVIDO: a premissa "pune só repetição literal, não machuca código" é FALSA.
    # Com --dry-allowed-length 4 ele pune a repetição LEGÍTIMA de código (ex.:
    # `this.getUser('1'), this.getUser('2')`), forçando o modelo a desviar e corromper o
    # texto — dropar caractere ("thisgetUser"), trocar dígito ('2'->'3'), trocar aspas.
    # Quebra applyPatch/edição exata; comprovado empiricamente (2026-06-17): mesmo prompt
    # acerta de primeira sem DRY. Anti-loop agora via sampling saudável + repeat-penalty
    # LEVE (1.05), que não corrompe reprodução exata.
    cmd += ' --temp 0.3 --top-p 0.8 --top-k 20'
    cmd += ' --repeat-penalty 1.05'
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
    split_mode:       str | None = None,
    tensor_split:     str | None = None,
    main_gpu:         int | None = None,
) -> str:
    cli_bin = backend_binary(backend, "cli")
    if not cli_bin.exists():
        raise BackendBinaryMissing(f"llama-cli não encontrado em: {cli_bin}")

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

    # Rede de segurança anti-loop (ver build_server_command). Sem --no-context-shift
    # no CLI: numa conversa de terminal o shift é melhor que morrer com erro.
    # DRY REMOVIDO (ver build_server_command): com --dry-allowed-length 4 ele pune a
    # repetição LEGÍTIMA de código e corrompe a reprodução exata (dropa caractere, troca
    # dígito/aspas), quebrando applyPatch. Comprovado 2026-06-17. Anti-loop agora via
    # sampling saudável + repeat-penalty LEVE (1.05).
    cmd += ' --temp 0.3 --top-p 0.8 --top-k 20'
    cmd += ' --repeat-penalty 1.05'

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


# ─── router mode (multi-modelo) ────────────────────────────────────────────
# llama-server sem -m + --models-preset arquivo.ini sobe o "router": cada
# seção do INI vira um modelo servível, carregado num processo filho isolado.
# O client escolhe o modelo pelo campo "model" do request (POST) ou pelo
# query param ?model= (GET) e o router encaminha pro filho certo.

def _router_section_lines(cfg: dict) -> list[str]:
    """Chaves INI de uma seção do preset — espelho 1:1 das flags que
    build_server_command emite (nomes long-form sem os `--`).

    host/port/alias ficam de fora: o README do server avisa que o router
    controla e sobrescreve esses argumentos ao subir cada instância.
    """
    lines: list[str] = []

    def add(key: str, val) -> None:
        lines.append(f"{key} = {val}")

    model_path = Path(cfg["model"])
    add("model", str(model_path))
    mmproj = cfg.get("mmproj")
    if mmproj:
        add("mmproj", str(Path(mmproj)))

    parallel = cfg.get("parallel_slots", 1)
    add("ctx-size", cfg["context_window"] * parallel)
    add("n-gpu-layers", cfg.get("gpu_layers", 99))

    split_mode = cfg.get("split_mode")
    if split_mode in ("none", "layer", "row"):
        add("split-mode", split_mode)
    tensor_split = (cfg.get("tensor_split") or "").strip().replace(" ", "")
    if tensor_split:
        add("tensor-split", tensor_split)
    main_gpu = cfg.get("main_gpu")
    if main_gpu is not None and main_gpu >= 0:
        add("main-gpu", main_gpu)

    n_cpu_moe = cfg.get("n_cpu_moe", 0)
    if n_cpu_moe and n_cpu_moe > 0:
        add("n-cpu-moe", n_cpu_moe)

    kv = cfg["kv_cache"]
    add("cache-type-k", kv)
    add("cache-type-v", kv)
    add("flash-attn", "on" if cfg.get("flash_attn", True) else "off")
    add("batch-size", cfg.get("batch_size", 2_048))
    add("ubatch-size", cfg.get("ubatch_size", 512))
    add("threads", cfg.get("threads_gen", CPU_THREADS_GEN))
    add("threads-batch", cfg.get("threads_batch", CPU_THREADS_BATCH))
    add("parallel", parallel)
    add("n-predict", cfg.get("max_tokens", 8_192))

    # Rede de segurança anti-loop — mesma justificativa do build_server_command.
    add("temp", 0.3)
    add("top-p", 0.8)
    add("top-k", 20)
    add("repeat-penalty", 1.05)
    add("no-context-shift", "true")

    reasoning_budget = cfg.get("reasoning_budget")
    if reasoning_budget is not None:
        if reasoning_budget == 0:
            add("reasoning", "off")
        else:
            add("reasoning", "on")
            add("reasoning-budget", reasoning_budget)

    if cfg.get("preserve_thinking", False):
        # Valor INI é a linha crua — sem escaping de shell.
        add("chat-template-kwargs", '{"preserve_thinking": true}')

    if cfg.get("mlock", True):
        add("mlock", "true")

    backend = cfg.get("backend", "turbo")
    if BACKENDS[backend]["supports_spec_mtp"] and is_mtp_model(model_path):
        add("spec-type", "draft-mtp")
        add("spec-draft-n-max", cfg.get("spec_draft_n_max", 2))

    add("cache-ram", cfg.get("cache_ram", 8_192))
    add("ctx-checkpoints", cfg.get("ctx_checkpoints", 8))
    add("jinja", "true")
    add("metrics", "true")
    if cfg.get("verbose", False):
        add("verbose", "true")

    # Exclusivo de preset: sobe o modelo junto com o router, sem esperar o
    # primeiro request — é o "rodar todos de uma vez" da UI.
    add("load-on-startup", "true")
    return lines


def build_router_preset(cfgs: list[dict]) -> tuple[str, list[str]]:
    """Gera o texto do INI e a lista de model ids (nomes das seções).

    O nome da seção é o id que o client usa no campo "model" do request.
    Aliases duplicados (mesmo modelo em duas configs) ganham sufixo -2, -3…
    """
    used: set[str] = set()
    names: list[str] = []
    blocks: list[str] = []
    for cfg in cfgs:
        base = make_alias(Path(cfg["model"]))
        name = base
        n = 2
        while name in used:
            name = f"{base}-{n}"
            n += 1
        used.add(name)
        names.append(name)
        blocks.append(f"[{name}]\n" + "\n".join(_router_section_lines(cfg)))
    text = "version = 1\n\n" + "\n\n".join(blocks) + "\n"
    return text, names


def router_binary_for(cfgs: list[dict]) -> Path:
    """Binário único do router. O router spawna os filhos com o PRÓPRIO exe,
    então todas as configs selecionadas precisam resolver pro mesmo
    llama-server (turbo x vanilla não podem misturar; vanilla+mtp podem,
    porque mtp é alias do binário vanilla)."""
    bins = {backend_binary(cfg.get("backend", "turbo"), "server") for cfg in cfgs}
    if len(bins) > 1:
        raise ValueError(
            "Configs selecionadas usam binários diferentes ("
            + ", ".join(str(b) for b in sorted(bins))
            + ") — o modo router roda todas no mesmo llama-server. "
            "Selecione configs de um único backend (vanilla e mtp podem misturar)."
        )
    server_bin = bins.pop()
    if not server_bin.exists():
        raise BackendBinaryMissing(f"llama-server não encontrado em: {server_bin}")
    return server_bin


def build_router_command(
    cfgs: list[dict],
    preset_path: Path,
    host: str = "0.0.0.0",
    port: int = 1234,
) -> str:
    """Comando do router: sem -m (é isso que ativa o modo router). Flags de
    modelo ficam TODAS no preset — argumento de linha de comando do router tem
    precedência sobre o INI e vazaria pra todas as instâncias filhas."""
    server_bin = router_binary_for(cfgs)
    cmd = f'"{server_bin}"'
    cmd += f' --models-preset "{preset_path}"'
    cmd += f' --models-max {len(cfgs)}'
    cmd += f' --host {host} --port {port}'
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
            split_mode=cfg.get("split_mode"),
            tensor_split=cfg.get("tensor_split"),
            main_gpu=cfg.get("main_gpu"),
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
        split_mode=cfg.get("split_mode"),
        tensor_split=cfg.get("tensor_split"),
        main_gpu=cfg.get("main_gpu"),
    )
