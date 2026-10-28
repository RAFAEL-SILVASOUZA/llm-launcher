"""Listagem e classificação de modelos GGUF na pasta do LM Studio.

Funções puras — não imprimem nada, não chamam exit, não interagem com TTY.
Foi extraído do bloco no topo de models.py (collect_models, find_mmproj,
make_alias, is_thinking_model, is_mtp_model, _SPLIT_RE).
"""
import re
from pathlib import Path

from . import gguf, sampling
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


# Projectors da família Qwen-VL: "qwen2vl_merger", "qwen2.5vl_merger",
# "qwen3vl_merger" (clip-impl.h). Só eles pedem --image-min-tokens 1024.
_QWEN_VL_PROJECTOR_RE = re.compile(r"^qwen[\d.]*vl", re.IGNORECASE)


def is_qwen_vl_mmproj(mmproj: Path | None) -> bool:
    """Se o projector é Qwen-VL — a família que degrada em grounding com o
    default de image tokens.

    O clip.cpp limita a imagem a 8..4096 tokens por default e avisa no load que
    abaixo de 1024 tokens MÍNIMOS o modelo erra tarefas de localização ("onde
    está X na tela"). Quem responde é o projector, não o nome do modelo, então
    a decisão sai de `clip.projector_type` no gguf do mmproj. Fallback pelo
    nome da pasta só quando o gguf não pôde ser lido.
    """
    if mmproj is None:
        return False
    meta = gguf.read_mmproj_meta(mmproj)
    proj = (meta or {}).get("projector_type")
    if proj:
        return bool(_QWEN_VL_PROJECTOR_RE.match(proj))
    return "qwen" in mmproj.parent.name.lower()


def make_alias(model_path: Path) -> str:
    m = SPLIT_RE.match(model_path.name)
    if m:
        return m.group("base")
    return model_path.stem


def is_thinking_model(model_path: Path) -> bool:
    """Decide pelo chat template do GGUF; keywords no nome são só o fallback.

    A lista de keywords era a detecção principal e errava calado em todo modelo
    com nome fora do padrão (ornith, glm, minimax…): o modelo pensava, o app o
    tratava como não-thinking, escondia o reasoning-budget e servia o preset de
    sampling errado. Só cai no nome quando o GGUF não pôde ser lido.
    """
    meta = gguf.read_meta(model_path)
    if meta is not None:
        return bool(meta.get("has_thinking_template"))
    name = model_path.name.lower()
    return any(kw in name for kw in THINKING_KEYWORDS)


_MTP_PATTERN = re.compile(r"(?:^|[-_.])mtp(?:[-_.]|$)", re.IGNORECASE)
# Head MTP empacotada SEPARADA: o repo publica `<quant>/` e `mtp-<quant>/` lado a
# lado (ex.: Tess-4-27B-GGUF/{Tess-4-27B-Q8_0, mtp-Tess-4-27B-Q8_0}). O prefixo
# — e só o prefixo — marca o drafter; sufixo/meio (`Qwen3.6-27B-MTP-GGUF`) é
# modelo com a head embutida no próprio gguf, que não precisa de -md.
_MTP_PREFIX_PATTERN = re.compile(r"^mtp[-_. ]+", re.IGNORECASE)


def has_mtp_prefix(name: str) -> bool:
    return bool(_MTP_PREFIX_PATTERN.match(name))


def strip_mtp_prefix(name: str) -> str:
    return _MTP_PREFIX_PATTERN.sub("", name, count=1)


def _first_model_gguf(folder: Path, *, prefixed: bool | None = None) -> Path | None:
    """Primeiro .gguf servível da pasta (sem mmproj, sem partes 00002+).

    `prefixed`: None = tanto faz; True = só nomes `mtp-*`; False = só os outros.
    """
    if not folder.is_dir():
        return None
    try:
        files = sorted(folder.glob("*.gguf"))
    except OSError:
        return None
    for f in files:
        if not f.is_file() or not is_primary_part(f):
            continue
        if prefixed is not None and has_mtp_prefix(f.name) != prefixed:
            continue
        return f
    return None


def find_mtp_draft(model_path: Path) -> Path | None:
    """Acha a head MTP em gguf SEPARADO, pra entrar no comando como -md.

    Ordem de busca, do mais específico pro mais frouxo:
      1. `mtp-*.gguf` na MESMA pasta do modelo (repos que publicam tudo junto);
      2. pasta IRMÃ `mtp-<pasta do modelo>` — o layout do exemplo;
      3. única pasta irmã com prefixo `mtp-` (cobre quant divergente entre o
         alvo e a head, ex.: `Q8_0/` com `mtp-…-Q6_K/`).
    Um drafter nunca busca outro drafter — se o próprio arquivo/pasta já tem o
    prefixo, retorna None.
    """
    folder = model_path.parent
    if has_mtp_prefix(model_path.name) or has_mtp_prefix(folder.name):
        return None

    same_folder = _first_model_gguf(folder, prefixed=True)
    if same_folder is not None:
        return same_folder

    parent = folder.parent
    if parent == folder or not parent.exists():
        return None
    candidates: list[tuple[Path, Path]] = []
    try:
        for d in sorted(parent.iterdir()):
            if d.is_dir() and d != folder and has_mtp_prefix(d.name):
                g = _first_model_gguf(d, prefixed=None)
                if g is not None:
                    candidates.append((d, g))
    except OSError:
        return None

    target = folder.name.lower()
    for d, g in candidates:
        if strip_mtp_prefix(d.name).lower() == target:
            return g
    return candidates[0][1] if len(candidates) == 1 else None


def is_external_mtp_draft(path: Path) -> bool:
    """True se o .gguf é a head MTP de OUTRO modelo — drafter, não modelo.

    Serve pra tirar o drafter da lista: sozinho ele não roda (é uma camada só),
    e listá-lo dava um "modelo" de 3 GB que só falha ao subir.
    """
    folder = path.parent
    # Convivendo com o alvo na mesma pasta: o par é o gguf sem prefixo ao lado.
    if has_mtp_prefix(path.name) and _first_model_gguf(folder, prefixed=False) is not None:
        return True
    # Em pasta própria: o par é uma pasta irmã sem prefixo que tenha modelo.
    if has_mtp_prefix(folder.name):
        parent = folder.parent
        if parent == folder or not parent.exists():
            return False
        try:
            for d in sorted(parent.iterdir()):
                if (d.is_dir() and d != folder and not has_mtp_prefix(d.name)
                        and _first_model_gguf(d) is not None):
                    return True
        except OSError:
            return False
    return False


def is_mtp_model(model_path: Path) -> bool:
    """Detecta head MTP: tag no nome/pasta ancestral OU drafter em gguf separado."""
    parts = [model_path.name, *(p.name for p in model_path.parents)]
    if any(_MTP_PATTERN.search(part) for part in parts):
        return True
    return find_mtp_draft(model_path) is not None


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


def collect_models(models_dir: Path | list[Path] = MODELS_DIR) -> list[Path]:
    """Lista todos os .gguf primários (sem mmproj, sem partes 00002+).

    Aceita um único Path (compat) ou uma lista — varre todas as pastas, dedup
    por caminho resolvido e devolve em ordem alfabética estável.
    """
    roots = [models_dir] if isinstance(models_dir, Path) else list(models_dir)
    found: dict[Path, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for f in root.rglob("*.gguf"):
            if not is_primary_part(f):
                continue
            # Head MTP separada é insumo do modelo-alvo (entra via -md), não um
            # item da lista.
            if is_external_mtp_draft(f):
                continue
            try:
                key = f.resolve()
            except OSError:
                key = f
            found.setdefault(key, f)
    return sorted(found.values())


def plan_delete_model(model_path: Path) -> list[Path]:
    """Lista todos os arquivos que seriam removidos (não remove nada).

    Inclui o(s) .gguf do quant, mmproj associado e arquivos .part de download
    interrompido. Sempre dedup, preserva ordem, só lista o que existe.
    """
    folder = model_path.parent
    to_remove: list[Path] = []
    m = SPLIT_RE.match(model_path.name)
    if m:
        key = m.group("base")
        for f in sorted(folder.glob("*.gguf")):
            if not is_mmproj(f.name) and gguf_group_key(f.name) == key:
                to_remove.append(f)
    else:
        to_remove.append(model_path)

    mm = find_mmproj(folder, include_parent=False)
    if mm:
        to_remove.append(mm)

    # A head MTP não aparece na lista, então some junto com o alvo — senão fica
    # um gguf órfão de GB que ninguém consegue mais apagar pela UI.
    draft = find_mtp_draft(model_path)
    if draft is not None:
        to_remove.append(draft)
        draft_mm = find_mmproj(draft.parent, include_parent=False)
        if draft_mm:
            to_remove.append(draft_mm)

    for f in list(to_remove):
        part = f.with_name(f.name + ".part")
        if part.exists():
            to_remove.append(part)

    seen: set[Path] = set()
    return [f for f in to_remove if f.exists() and not (f in seen or seen.add(f))]


def rmdir_if_empty(path: Path, stop_at: Path) -> None:
    """Sobe removendo diretórios vazios até (sem incluir) stop_at."""
    try:
        while path != stop_at and path.is_dir() and not any(path.iterdir()):
            path.rmdir()
            path = path.parent
    except Exception:
        pass


def delete_model(model_path: Path, models_dir: Path = MODELS_DIR) -> dict:
    """Apaga os arquivos listados por plan_delete_model + diretórios vazios.

    Retorna {removed: [paths], errors: [{path, error}], total_bytes}.
    """
    targets = plan_delete_model(model_path)
    removed: list[str] = []
    errors: list[dict] = []
    total = 0
    for f in targets:
        try:
            total += f.stat().st_size
        except Exception:
            pass
        try:
            f.unlink()
            removed.append(str(f))
        except Exception as e:
            errors.append({"path": str(f), "error": str(e)})
    # Varre as pastas tocadas (a do modelo + a da head MTP, que é outra), da mais
    # funda pra mais rasa, pra não deixar diretório vazio pra trás.
    for folder in sorted({f.parent for f in targets}, key=lambda p: -len(p.parts)):
        rmdir_if_empty(folder, models_dir)
    return {"removed": removed, "errors": errors, "total_bytes": total}


def describe_model(model_path: Path, models_dir: Path | list[Path] = MODELS_DIR) -> dict:
    """Dict serializável com infos básicas — alimenta a lista no frontend.

    `models_dir` pode ser uma raiz só ou várias — usamos a primeira que contém
    o arquivo pra montar o caminho relativo. Cai pro path absoluto se nenhuma
    raiz for ancestral.
    """
    mmproj = find_mmproj(model_path.parent)
    mtp_draft = find_mtp_draft(model_path)
    roots = [models_dir] if isinstance(models_dir, Path) else list(models_dir)
    rel = str(model_path)
    for root in roots:
        try:
            rel = model_path.relative_to(root).as_posix()
            break
        except ValueError:
            continue
    return {
        "path":           str(model_path),
        "alias":          make_alias(model_path),
        "relative_path":  rel,
        "filename":       model_path.name,
        "mmproj":         str(mmproj) if mmproj else None,
        "size_mib":       model_disk_size_mib(model_path),
        "is_thinking":    is_thinking_model(model_path),
        "is_mtp":         is_mtp_model(model_path),
        "mtp_draft":      str(mtp_draft) if mtp_draft else None,
        "sampling":       sampling.resolve(model_path),
    }
