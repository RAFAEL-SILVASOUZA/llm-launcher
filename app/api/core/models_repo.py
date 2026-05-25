"""Listagem e classificação de modelos GGUF na pasta do LM Studio.

Funções puras — não imprimem nada, não chamam exit, não interagem com TTY.
Foi extraído do bloco no topo de models.py (collect_models, find_mmproj,
make_alias, is_thinking_model, is_mtp_model, _SPLIT_RE).
"""
import re
from pathlib import Path

from .constants import MODELS_DIR, THINKING_KEYWORDS

SPLIT_RE = re.compile(
    r"^(?P<base>.+)-(?P<part>\d{5})-of-(?P<total>\d{5})\.gguf$",
    re.IGNORECASE,
)


def is_primary_part(f: Path) -> bool:
    """True se for o arquivo "principal" (sem split, ou parte 00001 do split)."""
    if "mmproj" in f.name.lower():
        return False
    m = SPLIT_RE.match(f.name)
    return not m or m.group("part") == "00001"


def gguf_group_key(name: str) -> str:
    """Conjunto a que o arquivo pertence: o quant, sem o sufixo -00001-of-00005."""
    base = Path(name).name
    m = SPLIT_RE.match(base)
    if m:
        return m.group("base")
    return base[:-5] if base.lower().endswith(".gguf") else base


def is_mmproj(name: str) -> bool:
    return "mmproj" in Path(name).name.lower()


def find_mmproj(folder: Path, *, include_parent: bool = True) -> Path | None:
    """Acha o mmproj associado ao modelo (mesma pasta ou pasta-pai)."""
    if not folder.exists():
        return None
    for f in folder.iterdir():
        if f.is_file() and "mmproj" in f.name.lower() and f.suffix == ".gguf":
            return f
    if include_parent:
        parent = folder.parent
        if parent != folder and parent.exists():
            try:
                for f in parent.iterdir():
                    if f.is_file() and "mmproj" in f.name.lower() and f.suffix == ".gguf":
                        return f
            except OSError:
                pass
    return None


def make_alias(model_path: Path) -> str:
    m = SPLIT_RE.match(model_path.name)
    if m:
        return m.group("base")
    return model_path.stem


def is_thinking_model(model_path: Path) -> bool:
    name = model_path.name.lower()
    return any(kw in name for kw in THINKING_KEYWORDS)


_MTP_PATTERN = re.compile(r"(?:^|[-_.])mtp(?:[-_.]|$)", re.IGNORECASE)


def is_mtp_model(model_path: Path) -> bool:
    """Detecta tag MTP no nome ou em qualquer pasta ancestral."""
    parts = [model_path.name, *(p.name for p in model_path.parents)]
    return any(_MTP_PATTERN.search(part) for part in parts)


def model_disk_size_mib(model_path: Path) -> int:
    """Tamanho total em MiB, somando partes split."""
    total = 0
    m = SPLIT_RE.match(model_path.name)
    if m:
        key = m.group("base")
        for f in sorted(model_path.parent.glob("*.gguf")):
            if not is_mmproj(f.name) and gguf_group_key(f.name) == key:
                try:
                    total += f.stat().st_size
                except Exception:
                    pass
    else:
        try:
            total = model_path.stat().st_size
        except Exception:
            pass
    return total // (1024 * 1024)


def collect_models(models_dir: Path = MODELS_DIR) -> list[Path]:
    """Lista todos os .gguf primários (sem mmproj, sem partes 00002+)."""
    if not models_dir.exists():
        return []
    return sorted(f for f in models_dir.rglob("*.gguf") if is_primary_part(f))


def describe_model(model_path: Path, models_dir: Path = MODELS_DIR) -> dict:
    """Dict serializável com infos básicas — alimenta a lista no frontend."""
    mmproj = find_mmproj(model_path.parent)
    try:
        rel = model_path.relative_to(models_dir).as_posix()
    except ValueError:
        rel = str(model_path)
    return {
        "path":           str(model_path),
        "alias":          make_alias(model_path),
        "relative_path":  rel,
        "filename":       model_path.name,
        "mmproj":         str(mmproj) if mmproj else None,
        "size_mib":       model_disk_size_mib(model_path),
        "is_thinking":    is_thinking_model(model_path),
        "is_mtp":         is_mtp_model(model_path),
    }
