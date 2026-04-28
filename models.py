import os
from pathlib import Path

MODELS_DIR = Path(r"C:\Users\PC\.lmstudio\models")

BASE_FLAGS = (
    '--jinja -ngl 99 -c 256000 -fa on '
    '--cache-type-k turbo3 --cache-type-v turbo3 '
    '-np 1 --metrics --host 0.0.0.0 --port 1234'
)


def find_mmproj(folder: Path) -> Path | None:
    """Retorna o primeiro mmproj encontrado na mesma pasta, ou None."""
    for f in folder.iterdir():
        if f.is_file() and 'mmproj' in f.name.lower() and f.suffix == '.gguf':
            return f
    return None


def make_alias(model_path: Path) -> str:
    """Gera alias a partir do nome do arquivo sem extensão."""
    return model_path.stem


def make_command(model_path: Path) -> str:
    alias = make_alias(model_path)
    mmproj = find_mmproj(model_path.parent)

    label = f'{alias} | Image' if mmproj else alias
    cmd = f'# {label}\n'
    cmd += f'llama-server -m "{model_path}"'

    if mmproj:
        cmd += f' --mmproj "{mmproj}"'

    cmd += f' --alias "{alias}" {BASE_FLAGS}'
    return cmd


def main():
    if not MODELS_DIR.exists():
        print(f"Pasta não encontrada: {MODELS_DIR}")
        return

    # Coleta todos os .gguf que NÃO são mmproj
    models = sorted([
        f for f in MODELS_DIR.rglob("*.gguf")
        if 'mmproj' not in f.name.lower()
    ])

    if not models:
        print("Nenhum modelo .gguf encontrado.")
        return

    print(f"# {len(models)} modelo(s) encontrado(s) em {MODELS_DIR}\n")

    commands = [make_command(m) for m in models]
    output = "\n\n\n".join(commands)  # 2 linhas em branco entre comandos

    print(output)

    # Salva também em arquivo .txt ao lado do script
    out_file = Path(__file__).parent / "comandos_llama.txt"
    out_file.write_text(output, encoding="utf-8")
    print(f"\n\n# Comandos salvos em: {out_file}")


if __name__ == "__main__":
    main()