"""Builder das linhas de comando llama-server / llama-cli a partir de uma config."""
import sys
from pathlib import Path

from . import sampling
from .backends import backend_binary, backend_supports_flag, kv_unified_active
from .constants import (
    BACKENDS,
    CPU_THREADS_BATCH,
    CPU_THREADS_GEN,
    DEFAULT_CORS_ORIGINS,
    IMAGE_MIN_TOKENS_QWEN_VL,
)
from .defaults import REASONING_BUDGET_MESSAGE
from .models_repo import find_mtp_draft, is_mtp_model, is_qwen_vl_mmproj, make_alias


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


def _spec_mtp_flags(model_path: Path, spec_draft_n_max: int) -> str:
    """Flags de speculative MTP. Duas embalagens da mesma feature:

      • head EMBUTIDA no gguf do modelo (`Qwen3.6-27B-MTP-GGUF`) → só --spec-type;
      • head em gguf SEPARADO, na pasta irmã `mtp-<quant>/` → precisa também do
        -md apontando pro arquivo, senão o loader não acha a head e o
        --spec-type draft-mtp sobe sem rascunhar nada.

    -ngld fica de fora de propósito: o default do upstream é 'auto', que
    acomoda o drafter na VRAM que sobrar em vez de forçar 99 e estourar.
    """
    out = f' --spec-type draft-mtp --spec-draft-n-max {spec_draft_n_max}'
    draft = find_mtp_draft(model_path)
    if draft is not None:
        out += f' -md "{draft}"'
    return out


def _image_min_tokens(
    mmproj:   Path | None,
    backend:  str,
    mode:     str = "server",
    override: int | None = None,
) -> int | None:
    """Valor de --image-min-tokens, ou None quando a flag não deve entrar.

    `override`: None = decide pelo projector (1024 só em Qwen-VL); 0 = desligado
    (é o degrau que o auto-degrade usa se o build recusar a flag); >0 = valor
    explícito da config.
    """
    if mmproj is None or override == 0:
        return None
    tokens = override
    if tokens is None:
        if not is_qwen_vl_mmproj(mmproj):
            return None
        tokens = IMAGE_MIN_TOKENS_QWEN_VL
    if not backend_supports_flag("--image-min-tokens", backend, mode):
        return None
    return tokens


def _cors_flags(cors_origins: str | None, backend: str) -> str:
    """--cors-origins. Sem isso o server sobe com "*" e avisa que qualquer
    origem pode chamar a API sem API key (ver DEFAULT_CORS_ORIGINS)."""
    origins = (cors_origins or "").strip()
    if not origins:
        return ""
    if not backend_supports_flag("--cors-origins", backend, "server"):
        return ""
    # Aspas: o valor pode ser "*" ou uma lista separada por vírgula.
    return f' --cors-origins "{origins}"'


def build_auto_server_command(
    model_path: Path,
    backend:    str,
    mmproj:     Path | None = None,
    verbose:    bool = False,
    host:       str = "0.0.0.0",
    port:       int = 1234,
    cors_origins: str | None = DEFAULT_CORS_ORIGINS,
) -> str:
    """Comando MÍNIMO — o modo "llama.cpp decide" (cfg["llama_auto"]).

    Só entra o que identifica o modelo (-m/--mmproj/--alias) e o que serve o
    HTTP (--jinja/--metrics/--host/--port/--cors-origins). Nenhuma flag de
    tuning: contexto, -ngl, KV, threads, batch, samplers, reasoning e
    speculative ficam nos defaults do próprio llama.cpp, que decide tudo a
    partir do GGUF e da máquina. É a saída de emergência pra quando o nosso
    palpite atrapalha. --cors-origins fica porque não é tuning: é a porta de
    entrada HTTP, e o default "*" do llama.cpp abre a API pra qualquer site.
    """
    server_bin = backend_binary(backend, "server")
    if not server_bin.exists():
        raise BackendBinaryMissing(f"llama-server não encontrado em: {server_bin}")

    cmd = f'"{server_bin}" -m "{model_path}"'
    if mmproj:
        cmd += f' --mmproj "{mmproj}"'
    cmd += f' --alias "{make_alias(model_path)}"'
    cmd += f' --jinja --metrics --host {host} --port {port}'
    cmd += _cors_flags(cors_origins, backend)
    if verbose:
        cmd += ' --verbose'
    return cmd


def build_auto_cli_command(
    model_path: Path,
    backend:    str,
    mmproj:     Path | None = None,
    verbose:    bool = False,
) -> str:
    """Equivalente do modo auto pro llama-cli: modelo + --conversation (é o que
    torna o CLI interativo, não é tuning). O resto é default do llama.cpp."""
    cli_bin = backend_binary(backend, "cli")
    if not cli_bin.exists():
        raise BackendBinaryMissing(f"llama-cli não encontrado em: {cli_bin}")

    cmd = f'"{cli_bin}" -m "{model_path}"'
    if mmproj:
        cmd += f' --mmproj "{mmproj}"'
    cmd += ' --conversation'
    if verbose:
        cmd += ' --verbose'
    return cmd


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
    kv_unified:       bool = True,
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
    sampler:          dict | None = None,
    cors_origins:     str | None = DEFAULT_CORS_ORIGINS,
    image_min_tokens: int | None = None,
) -> str:
    alias = make_alias(model_path)
    server_bin = backend_binary(backend, "server")
    if not server_bin.exists():
        raise BackendBinaryMissing(f"llama-server não encontrado em: {server_bin}")

    cmd = f'"{server_bin}" -m "{model_path}"'
    if mmproj:
        cmd += f' --mmproj "{mmproj}"'
        img_tokens = _image_min_tokens(mmproj, backend, "server", image_min_tokens)
        if img_tokens:
            cmd += f' --image-min-tokens {img_tokens}'

    # -c é o contexto TOTAL, e o llama.cpp reparte entre os slots -np. Com -kvu
    # ele não reparte (n_ctx_seq = n_ctx): os slots dividem um pool só, e
    # context_window é tanto o pool quanto o teto de UMA request — a semântica do
    # LM Studio. Sem -kvu, cada slot precisa da sua fatia, então o -c multiplica e
    # o KV custa np×. As duas pernas mantêm a mesma promessa: uma request sempre
    # endereça context_window tokens.
    unified = kv_unified_active(backend, kv_unified)
    cmd += f' --alias "{alias}"'
    cmd += f' -c {context_window if unified else context_window * parallel_slots}'
    cmd += f' -ngl {gpu_layers}'
    cmd += _multi_gpu_flags(split_mode, tensor_split, main_gpu)
    if n_cpu_moe and n_cpu_moe > 0:
        cmd += f' --n-cpu-moe {n_cpu_moe}'
    cmd += f' --cache-type-k {kv_cache} --cache-type-v {kv_cache}'
    cmd += ' -fa on' if flash_attn else ' -fa off'
    cmd += f' -b {batch_size} -ub {ubatch_size}'
    cmd += f' -t {threads_gen} -tb {threads_batch}'
    cmd += f' -np {parallel_slots}'
    if unified:
        cmd += ' -kvu'
    cmd += f' -n {max_tokens}'

    # Samplers: DEFAULTS do servidor (o request do client ainda sobrescreve).
    # Resolvidos POR MODELO em sampling.py — preset de código pra não-thinking,
    # preset do card pra reasoning, ou os números do próprio autor quando o
    # generation_config.json veio junto no download.
    # DRY continua fora: com --dry-allowed-length 4 ele pune a repetição LEGÍTIMA de
    # código (`this.getUser('1'), this.getUser('2')`), forçando o modelo a desviar e
    # corromper o texto — dropar caractere, trocar dígito/aspas — o que quebra
    # applyPatch/edição exata (comprovado 2026-06-17). Presence-penalty idem: taxa
    # todo token já visto, sem janela, e induz EOS precoce em sessão longa (2026-06-11).
    cmd += sampling.cli_flags(sampler or sampling.resolve(model_path))
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
            # Budget estourado sem esta mensagem = o <think> é cortado no meio e o
            # modelo pode não se recuperar. A mensagem é injetada antes do
            # </think> forçado e devolve o modelo pra "responde ou chama a tool" —
            # é o que quebra o loop agêntico do híbrido SSM.
            if reasoning_budget > 0 and backend_supports_flag(
                "--reasoning-budget-message", backend, "server"
            ):
                cmd += f' --reasoning-budget-message "{REASONING_BUDGET_MESSAGE}"'

    if preserve_thinking:
        cmd += ' --chat-template-kwargs "{\\"preserve_thinking\\": true}"'

    if mlock:
        cmd += ' --mlock'

    if BACKENDS[backend]["supports_spec_mtp"] and is_mtp_model(model_path):
        cmd += _spec_mtp_flags(model_path, spec_draft_n_max)

    cmd += f' --cache-ram {cache_ram}'
    cmd += f' --ctx-checkpoints {ctx_checkpoints}'
    cmd += f' --jinja --metrics --host {host} --port {port}'
    cmd += _cors_flags(cors_origins, backend)

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
    sampler:          dict | None = None,
    image_min_tokens: int | None = None,
) -> str:
    cli_bin = backend_binary(backend, "cli")
    if not cli_bin.exists():
        raise BackendBinaryMissing(f"llama-cli não encontrado em: {cli_bin}")

    cmd = f'"{cli_bin}" -m "{model_path}"'
    if mmproj:
        cmd += f' --mmproj "{mmproj}"'
        img_tokens = _image_min_tokens(mmproj, backend, "cli", image_min_tokens)
        if img_tokens:
            cmd += f' --image-min-tokens {img_tokens}'

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

    # Samplers por modelo (ver build_server_command). Sem --no-context-shift no CLI:
    # numa conversa de terminal o shift é melhor que morrer com erro.
    cmd += sampling.cli_flags(sampler or sampling.resolve(model_path))

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
        cmd += _spec_mtp_flags(model_path, spec_draft_n_max)

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

    backend = cfg.get("backend", "turbo")
    model_path = Path(cfg["model"])
    add("model", str(model_path))
    mmproj = cfg.get("mmproj")
    if mmproj:
        add("mmproj", str(Path(mmproj)))
        img_tokens = _image_min_tokens(
            Path(mmproj), backend, "server", cfg.get("image_min_tokens")
        )
        if img_tokens:
            add("image-min-tokens", img_tokens)

    # Modo auto (checkbox "llama.cpp decide"): a seção fica só com o modelo —
    # o filho sobe com os defaults do llama.cpp, igual ao launch único.
    if cfg.get("llama_auto", False):
        if cfg.get("verbose", False):
            add("verbose", "true")
        add("load-on-startup", "true")
        return lines

    # ctx-size/kv-unified: mesma regra do build_server_command — com pool
    # compartilhado o ctx não multiplica pelos slots.
    parallel = cfg.get("parallel_slots", 1)
    unified  = kv_unified_active(backend, cfg.get("kv_unified", True))
    add("ctx-size", cfg["context_window"] if unified else cfg["context_window"] * parallel)
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
    if unified:
        add("kv-unified", "true")
    add("n-predict", cfg.get("max_tokens", 8_192))

    # Samplers por modelo — mesma resolução do build_server_command.
    for key, val in sampling.ini_pairs(sampling.resolve(model_path, cfg)):
        add(key, val)
    add("no-context-shift", "true")

    reasoning_budget = cfg.get("reasoning_budget")
    if reasoning_budget is not None:
        if reasoning_budget == 0:
            add("reasoning", "off")
        else:
            add("reasoning", "on")
            add("reasoning-budget", reasoning_budget)
            if reasoning_budget > 0 and backend_supports_flag(
                "--reasoning-budget-message", backend, "server"
            ):
                # Valor INI é a linha crua — sem aspas de shell.
                add("reasoning-budget-message", REASONING_BUDGET_MESSAGE)

    if cfg.get("preserve_thinking", False):
        # Valor INI é a linha crua — sem escaping de shell.
        add("chat-template-kwargs", '{"preserve_thinking": true}')

    if cfg.get("mlock", True):
        add("mlock", "true")

    if BACKENDS[backend]["supports_spec_mtp"] and is_mtp_model(model_path):
        add("spec-type", "draft-mtp")
        add("spec-draft-n-max", cfg.get("spec_draft_n_max", 2))
        draft = find_mtp_draft(model_path)
        if draft is not None:
            add("spec-draft-model", str(draft))

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
    cors_origins: str | None = DEFAULT_CORS_ORIGINS,
) -> str:
    """Comando do router: sem -m (é isso que ativa o modo router). Flags de
    modelo ficam TODAS no preset — argumento de linha de comando do router tem
    precedência sobre o INI e vazaria pra todas as instâncias filhas.

    --cors-origins é a exceção proposital: o vazamento é justamente o que se
    quer (o router é quem atende a porta pública, e os filhos herdam a mesma
    política de origem)."""
    server_bin = router_binary_for(cfgs)
    backend = cfgs[0].get("backend", "turbo") if cfgs else "turbo"
    # A política é do processo, não do modelo: a primeira config que definir
    # cors_origins manda (o router serve todas as seções na mesma porta).
    for cfg in cfgs:
        if cfg.get("cors_origins") is not None:
            cors_origins = cfg["cors_origins"]
            break
    cmd = f'"{server_bin}"'
    cmd += f' --models-preset "{preset_path}"'
    cmd += f' --models-max {len(cfgs)}'
    cmd += f' --host {host} --port {port}'
    cmd += _cors_flags(cors_origins, backend)
    return cmd


def build_command_from_cfg(cfg: dict, mode: str = "server") -> str:
    """Reconstrói o comando a partir de uma config salva."""
    mmproj_str = cfg.get("mmproj")
    mmproj = Path(mmproj_str) if mmproj_str else None
    model_path = Path(cfg["model"])

    # Nenhuma das duas tem controle na UI: cors_origins é a válvula pra quando um
    # front fora de localhost precisar chamar a API (ver DEFAULT_CORS_ORIGINS) e
    # image_min_tokens é o degrau do auto-degrade (0 = não emite a flag).
    cors_origins     = cfg.get("cors_origins", DEFAULT_CORS_ORIGINS)
    image_min_tokens = cfg.get("image_min_tokens")

    # Modo auto: nenhuma outra chave da config vira flag (elas continuam
    # salvas — desmarcar o checkbox devolve a config completa intacta).
    if cfg.get("llama_auto", False):
        backend = cfg.get("backend", "turbo")
        verbose = cfg.get("verbose", False)
        if mode == "server":
            return build_auto_server_command(
                model_path, backend, mmproj, verbose, cors_origins=cors_origins
            )
        return build_auto_cli_command(model_path, backend, mmproj, verbose)

    sampler = sampling.resolve(model_path, cfg)
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
            kv_unified=cfg.get("kv_unified", True),
            n_cpu_moe=cfg.get("n_cpu_moe", 0),
            spec_draft_n_max=cfg.get("spec_draft_n_max", 2),
            preserve_thinking=cfg.get("preserve_thinking", False),
            mmproj=mmproj,
            verbose=cfg.get("verbose", False),
            split_mode=cfg.get("split_mode"),
            tensor_split=cfg.get("tensor_split"),
            main_gpu=cfg.get("main_gpu"),
            sampler=sampler,
            cors_origins=cors_origins,
            image_min_tokens=image_min_tokens,
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
        sampler=sampler,
        image_min_tokens=image_min_tokens,
    )
