import json
import subprocess
import sys
from pathlib import Path

import questionary

MODELS_DIR  = Path(r"C:\Users\PC\.lmstudio\models")
LLAMA_SERVER = Path(r"C:\Projetos\llama-cpp-turboquant\build\bin\Release\llama-server.exe")
LLAMA_CLI    = Path(r"C:\Projetos\llama-cpp-turboquant\build\bin\Release\llama-cli.exe")
CONFIG_FILE  = Path(r"C:\Projetos\llama-cpp-turboquant\last_config.json")

# i5-14600KF: 6P + 8E = 14 núcleos físicos, 20 threads lógicos
CPU_THREADS_GEN   = 14
CPU_THREADS_BATCH = 20

THINKING_KEYWORDS = {"qwen3", "qwq", "deepseek-r1", "thinking"}

# ─── choices ──────────────────────────────────────────────────────────────────

MODE_CHOICES = [
    questionary.Choice("llama-server  — API HTTP (OpenAI-compatible, Claude Code, etc.)",  value="server"),
    questionary.Choice("llama-cli     — Chat interativo no terminal",                      value="cli"),
]

KV_CACHE_CHOICES = [
    questionary.Choice("f16    — padrão, sem compressão",                                 value="f16"),
    questionary.Choice("turbo3 — ~3-bit KV, alta compressão  (ótimo para ctx longo)",    value="turbo3"),
    questionary.Choice("turbo4 — ~4-bit KV, mais qualidade que turbo3  [recomendado]",   value="turbo4"),
    questionary.Choice("q8_0   — leve, quase sem perda de qualidade",                    value="q8_0"),
    questionary.Choice("q5_0   — compressão moderada",                                   value="q5_0"),
    questionary.Choice("q4_0   — compressão alta, economiza mais VRAM",                  value="q4_0"),
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
    questionary.Choice("    0  — Desabilitar thinking  (/no_think)", value=0),
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
    questionary.Choice("   -1  — Ilimitado  [default]", value=-1),
    questionary.Choice(" 2048",                          value=2_048),
    questionary.Choice(" 4096",                          value=4_096),
    questionary.Choice(" 8192",                          value=8_192),
    questionary.Choice("16384",                          value=16_384),
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
	"""Salva a config no array, substituindo a existente do mesmo modelo (se houver)."""
	configs = _read_all_configs()
	model_key = cfg.get("model", "")

	# Remove config antiga do mesmo modelo, se existir
	configs = [c for c in configs if c.get("model") != model_key]
	configs.append(cfg)

	try:
		CONFIG_FILE.write_text(json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8")
	except Exception:
		pass


def load_config(model_path: Path) -> dict | None:
	"""Retorna a config salva para o modelo especificado, ou None."""
	configs = _read_all_configs()
	for cfg in configs:
		if cfg.get("model") == str(model_path):
			return cfg
	return None


def find_mmproj(folder: Path) -> Path | None:
    for f in folder.iterdir():
        if f.is_file() and 'mmproj' in f.name.lower() and f.suffix == '.gguf':
            return f
    return None


def make_alias(model_path: Path) -> str:
    return model_path.stem


def collect_models() -> list[Path]:
    if not MODELS_DIR.exists():
        print(f"Pasta não encontrada: {MODELS_DIR}")
        sys.exit(1)

    models = sorted([
        f for f in MODELS_DIR.rglob("*.gguf")
        if 'mmproj' not in f.name.lower()
    ])

    if not models:
        print("Nenhum modelo .gguf encontrado.")
        sys.exit(1)

    return models


def is_thinking_model(model_path: Path) -> bool:
    name = model_path.name.lower()
    return any(kw in name for kw in THINKING_KEYWORDS)


def ask_int(prompt: str, default: int) -> int:
    val = questionary.text(f"{prompt} (default {default}):").ask().strip()
    return int(val) if val else default


def _read_gguf_template(model_path: Path) -> str | None:
    import struct
    try:
        with open(model_path, 'rb') as f:
            if f.read(4) != b'GGUF':
                return None
            f.read(4)   # version
            f.read(8)   # tensor_count
            kv_count = struct.unpack('<Q', f.read(8))[0]

            def skip_value(vtype: int) -> None:
                fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
                if vtype in fixed:
                    f.read(fixed[vtype])
                elif vtype == 8:
                    f.read(struct.unpack('<Q', f.read(8))[0])
                elif vtype == 9:
                    et = struct.unpack('<I', f.read(4))[0]
                    for _ in range(struct.unpack('<Q', f.read(8))[0]):
                        skip_value(et)
                else:
                    raise ValueError(f"unknown GGUF type {vtype}")

            for _ in range(kv_count):
                key_len = struct.unpack('<Q', f.read(8))[0]
                key = f.read(key_len).decode('utf-8')
                vtype = struct.unpack('<I', f.read(4))[0]
                if key == 'tokenizer.chat_template':
                    str_len = struct.unpack('<Q', f.read(8))[0]
                    return f.read(str_len).decode('utf-8')
                skip_value(vtype)
    except Exception:
        return None
    return None


_FALLBACK_TEMPLATE = (
    "{{% set default_system = {system} %}}\n"
    "{{% set ns = namespace(found_system=false) %}}\n"
    "{{% for message in messages %}}\n"
    "    {{% if message['role'] == 'system' %}}{{% set ns.found_system = true %}}{{% endif %}}\n"
    "{{% endfor %}}\n"
    "{{% if not ns.found_system %}}<|im_start|>system\n"
    "{{{{ default_system }}}}<|im_end|>\n"
    "{{% endif %}}\n"
    "{{% for message in messages %}}<|im_start|>{{{{ message['role'] }}}}\n"
    "{{{{ message['content'] }}}}<|im_end|>\n"
    "{{% endfor %}}\n"
    "{{% if add_generation_prompt %}}<|im_start|>assistant\n"
    "{{% endif %}}\n"
)

_KNOWN_DEFAULTS = [
    "You are Qwen, made by Alibaba Cloud. You are a helpful assistant.",
    "You are a helpful assistant.",
]


def generate_chat_template(system_file: Path, model_path: Path) -> Path:
    system_prompt = system_file.read_text(encoding="utf-8").strip()
    jinja_file = system_file.with_suffix(".jinja")

    native = _read_gguf_template(model_path)
    if native:
        template = native
        for default in _KNOWN_DEFAULTS:
            if default in native:
                template = native.replace(default, system_prompt, 1)
                break
    else:
        template = _FALLBACK_TEMPLATE.format(system=json.dumps(system_prompt, ensure_ascii=False))

    jinja_file.write_text(template, encoding="utf-8")
    return jinja_file


def ask_context_window() -> int:
    selected = questionary.select(
        "📏 Janela de contexto:",
        choices=CONTEXT_CHOICES,
    ).ask()
    if selected == 0:
        return ask_int("   Valor customizado", 65_536)
    return selected

# ─── command builder ──────────────────────────────────────────────────────────

def build_command(
    model_path:       Path,
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
    mmproj:           Path | None = None,
    system_file:      Path | None = None,
) -> str:
    alias = make_alias(model_path)

    if not LLAMA_SERVER.exists():
        print(f"❌ llama-server não encontrado em: {LLAMA_SERVER}")
        print("   Compile o turboquant ou ajuste o caminho em LLAMA_SERVER.")
        sys.exit(1)

    cmd = f'"{LLAMA_SERVER}" -m "{model_path}"'

    if mmproj:
        cmd += f' --mmproj "{mmproj}"'

    cmd += f' --alias "{alias}"'
    cmd += f' -c {context_window}'
    cmd += f' -ngl {gpu_layers}'
    cmd += f' --cache-type-k {kv_cache} --cache-type-v {kv_cache}'
    cmd += ' -fa on' if flash_attn else ' -fa off'
    cmd += f' -b {batch_size} -ub {ubatch_size}'
    cmd += f' -t {threads_gen} -tb {threads_batch}'
    cmd += f' -np {parallel_slots}'
    cmd += f' -n {max_tokens}'

    if reasoning_budget is not None:
        cmd += f' --reasoning-budget {reasoning_budget}'

    if mlock:
        cmd += ' --mlock'

    if system_file:
        jinja_path = generate_chat_template(system_file, model_path)
        cmd += f' --chat-template-file "{jinja_path}"'

    cmd += ' --defrag-thold 0.1'
    cmd += ' --jinja --metrics --host 0.0.0.0 --port 1234'

    return cmd

def build_cli_command(
    model_path:       Path,
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
    mmproj:           Path | None = None,
    system_file:      Path | None = None,
) -> str:
    if not LLAMA_CLI.exists():
        print(f"❌ llama-cli não encontrado em: {LLAMA_CLI}")
        print("   Compile o turboquant ou ajuste o caminho em LLAMA_CLI.")
        sys.exit(1)

    cmd = f'"{LLAMA_CLI}" -m "{model_path}"'

    if mmproj:
        cmd += f' --mmproj "{mmproj}"'

    cmd += f' -c {context_window}'
    cmd += f' -ngl {gpu_layers}'
    cmd += f' --cache-type-k {kv_cache} --cache-type-v {kv_cache}'
    cmd += ' -fa on' if flash_attn else ' -fa off'
    cmd += f' -b {batch_size} -ub {ubatch_size}'
    cmd += f' -t {threads_gen} -tb {threads_batch}'
    cmd += f' -n {max_tokens}'

    if reasoning_budget is not None:
        cmd += f' --reasoning-budget {reasoning_budget}'

    if mlock:
        cmd += ' --mlock'

    if system_file:
        jinja_path = generate_chat_template(system_file, model_path)
        cmd += f' --chat-template-file "{jinja_path}"'

    cmd += ' --conversation'

    return cmd

# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    models = collect_models()

    labels = []
    for m in models:
        alias  = make_alias(m)
        mmproj = find_mmproj(m.parent)
        tag    = " | 🖼️ Image" if mmproj else ""
        rel    = m.relative_to(MODELS_DIR).as_posix()
        labels.append(f"{alias}{tag}  ({rel})")

    print(f"\n📋 {len(models)} modelo(s) encontrado(s) em {MODELS_DIR}\n")

    mode: str = questionary.select(
        "⚙️  Modo de execução:",
        choices=MODE_CHOICES,
    ).ask()

    if not mode:
        return

    print()

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

    # ── config anterior deste modelo? ────────────────────────────────────────
    prev_cfg = load_config(model_path)
    use_prev = False

    if prev_cfg:
        prev_alias = make_alias(Path(prev_cfg.get("model", "")))
        ctx = prev_cfg.get('context_window', '?')
        ctx_str = f"{ctx:,}" if isinstance(ctx, int) else ctx
        rb = prev_cfg.get('reasoning_budget')
        rb_str = str(rb) if rb is not None else "—"

        print("┌─ Configuração salva para este modelo ─────────────────────")
        print(f"│  Modelo          : {prev_alias}")
        print(f"│  Contexto        : {ctx_str}")
        print(f"│  KV Cache        : {prev_cfg.get('kv_cache', '?')}")
        print(f"│  Flash Attention : {prev_cfg.get('flash_attn', '?')}")
        print(f"│  GPU Layers      : {prev_cfg.get('gpu_layers', '?')}")
        print(f"│  Slots paralelos : {prev_cfg.get('parallel_slots', '?')}")
        print(f"│  MLock           : {prev_cfg.get('mlock', '?')}")
        print(f"│  Max Tokens      : {prev_cfg.get('max_tokens', '?')}")
        print(f"│  Reasoning Budget: {rb_str}")
        print("└───────────────────────────────────────────────────────────")
        print()
        if questionary.confirm("   Usar a configuração salva?", default=True).ask():
            use_prev = True
            context_window   = prev_cfg.get("context_window", 65_536)
            kv_cache         = prev_cfg.get("kv_cache", "turbo4")
            flash_attn       = prev_cfg.get("flash_attn", True)
            gpu_layers       = prev_cfg.get("gpu_layers", 99)
            parallel_slots   = prev_cfg.get("parallel_slots", 1)
            reasoning_budget = prev_cfg.get("reasoning_budget")
            mlock            = prev_cfg.get("mlock", True)
            max_tokens       = prev_cfg.get("max_tokens", -1)
            batch_size       = prev_cfg.get("batch_size", 2_048)
            ubatch_size      = prev_cfg.get("ubatch_size", 512)
            threads_gen      = prev_cfg.get("threads_gen", CPU_THREADS_GEN)
            threads_batch    = prev_cfg.get("threads_batch", CPU_THREADS_BATCH)
            use_mmproj: Path | None = None
            if prev_cfg.get("mmproj"):
                p = Path(prev_cfg["mmproj"])
                use_mmproj = p if p.exists() else None

    if not use_prev:
        # ── mmproj ───────────────────────────────────────────────────────────
        mmproj     = find_mmproj(model_path.parent)
        use_mmproj: Path | None = None
        if mmproj:
            if questionary.confirm("🖼️  Habilitar suporte a imagem (--mmproj)?", default=True).ask():
                use_mmproj = mmproj

        # ── seção 1: contexto e memória ──────────────────────────────────────
        context_window = ask_context_window()

        kv_cache = questionary.select(
            "🗜️  Cache KV (K e V):",
            choices=KV_CACHE_CHOICES,
        ).ask()

        flash_attn = questionary.confirm("⚡ Flash Attention (-fa)?", default=True).ask()
        gpu_layers = ask_int("🎮 Camadas na GPU (-ngl)", 99)

        print()

        # ── seção 2: paralelismo e slots (apenas server) ─────────────────────
        parallel_slots: int = 1
        if mode == "server":
            parallel_slots = questionary.select(
                "🔀 Slots paralelos (-np):",
                choices=PARALLEL_CHOICES,
            ).ask()

            kv_total = context_window * parallel_slots
            if kv_total > 131_072:
                print(f"   ⚠️  {context_window:,} ctx × {parallel_slots} slots = {kv_total:,} tokens no KV cache — verifique VRAM disponível.")

        # ── seção 3: reasoning (apenas modelos thinking) ─────────────────────
        reasoning_budget: int | None = None
        if is_thinking_model(model_path):
            print()
            reasoning_budget = questionary.select(
                "🧠 Reasoning budget (--reasoning-budget):",
                choices=REASONING_BUDGET_CHOICES,
            ).ask()

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

    # ── salva configuração ───────────────────────────────────────────────────
    cfg = {
        "model":          str(model_path),
        "context_window": context_window,
        "kv_cache":       kv_cache,
        "flash_attn":     flash_attn,
        "gpu_layers":     gpu_layers,
        "parallel_slots": parallel_slots,
        "reasoning_budget": reasoning_budget,
        "mlock":          mlock,
        "max_tokens":     max_tokens,
        "batch_size":     batch_size,
        "ubatch_size":    ubatch_size,
        "threads_gen":    threads_gen,
        "threads_batch":  threads_batch,
        "mmproj":         str(use_mmproj) if use_mmproj else None,
    }
    save_config(cfg)

    print()

    # ── system prompt (arquivo .md) ──────────────────────────────────────────
    system_file: Path | None = None

    # ── build e execução ─────────────────────────────────────────────────────
    if mode == "server":
        cmd = build_command(
            model_path=model_path,
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
            mmproj=use_mmproj,
            system_file=system_file,
        )
    else:
        cmd = build_cli_command(
            model_path=model_path,
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
            mmproj=use_mmproj,
            system_file=system_file,
        )
    alias      = make_alias(model_path)
    mode_label = "llama-server" if mode == "server" else "llama-cli"

    img_status    = f"mmproj: {use_mmproj.name}" if use_mmproj else "mmproj: off"
    budget_status = f"budget: {reasoning_budget}" if reasoning_budget is not None else None
    sys_status    = f"system: {system_file.name}" if system_file else "system: off"

    print(f"🚀 Iniciando [{mode_label}]: {alias}")
    print(f"   ctx: {context_window:,}  |  kv: {kv_cache}  |  fa: {flash_attn}  |  ngl: {gpu_layers}")
    if mode == "server":
        print(f"   np: {parallel_slots}  |  b: {batch_size}  |  ub: {ubatch_size}  |  t: {threads_gen}/{threads_batch}  |  mlock: {mlock}")
    else:
        print(f"   b: {batch_size}  |  ub: {ubatch_size}  |  t: {threads_gen}/{threads_batch}  |  mlock: {mlock}")
    if budget_status:
        print(f"   {budget_status}  |  n: {max_tokens}  |  {img_status}  |  {sys_status}")
    else:
        print(f"   n: {max_tokens}  |  {img_status}  |  {sys_status}")
    print()
    print(f"   {cmd}\n")

    try:
        subprocess.run(cmd, shell=True, check=True)
    except KeyboardInterrupt:
        print("\n\n⛔ Interrompido pelo usuário.")
    except FileNotFoundError:
        print(f"❌ {mode_label} não encontrado.")
    except Exception as e:
        print(f"❌ Erro ao executar: {e}")


if __name__ == "__main__":
    main()
