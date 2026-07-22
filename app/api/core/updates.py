"""Detecção de atualização de GGUFs baixados do HuggingFace.

Compara o que está no disco com o estado atual do repo no HF. O sinal primário é
o TAMANHO do arquivo (grátis — vem da árvore do repo); a confirmação é o sha256
(git-lfs oid), que pode vir de três fontes, em ordem de custo:

  1. `origin.json` gravado ao lado do modelo no download (o oid que baixamos);
  2. cache local de oids computados, chaveado por (path, mtime, size);
  3. hash sob demanda do arquivo em disco ("checagem a fundo").

A lógica de comparação (`compare_file`, `aggregate`) é pura e testável — não toca
rede nem disco. `check_model` orquestra rede + disco em volta dela.

Modelos baixados ANTES deste recurso não têm `origin.json`; o `repo_id` é então
inferido do layout de pastas (`<root>/<owner>/<repo>/.../arquivo.gguf`), que é
exatamente como o downloader organiza tudo.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from . import hf, models_repo
from .constants import CONFIG_FILE

UP_TO_DATE = "up_to_date"
UPDATE_AVAILABLE = "update_available"
UNKNOWN = "unknown"

ORIGIN_NAME = "origin.json"
_OID_CACHE_FILE = CONFIG_FILE.with_name("oid_cache.json")


@dataclass
class FileVerdict:
    status: str      # up_to_date | update_available | unknown
    verified: bool   # True se o veredito veio de sinal definitivo (tamanho ou oid)
    reason: str


# ─── lógica pura ──────────────────────────────────────────────────────────────

def normalize_oid(oid: str | None) -> str | None:
    """sha256 canônico: minúsculo, sem o prefixo `sha256:`. None se vazio."""
    if not oid:
        return None
    s = str(oid).strip()
    if s.lower().startswith("sha256:"):
        s = s[len("sha256:"):]
    s = s.strip().lower()
    return s or None


def compare_file(
    local_size: int,
    local_oid: str | None,
    remote_size: int | None,
    remote_oid: str | None,
) -> FileVerdict:
    """Veredito de um arquivo local contra sua contraparte no HF.

    - remoto ausente (remote_size None) → UNKNOWN (renomeado/removido lá em cima);
    - tamanho difere → UPDATE_AVAILABLE (sinal definitivo, sem precisar de oid);
    - tamanho igual + oids conhecidos e diferentes → UPDATE_AVAILABLE;
    - tamanho igual + oids conhecidos e iguais → UP_TO_DATE (verificado);
    - tamanho igual + oid local ou remoto desconhecido → UP_TO_DATE, mas
      `verified=False` (otimista: tamanho bate, mas não confirmamos o conteúdo).
    """
    if remote_size is None:
        return FileVerdict(UNKNOWN, False, "arquivo não encontrado no repositório")

    if int(local_size) != int(remote_size):
        return FileVerdict(
            UPDATE_AVAILABLE, True,
            f"tamanho difere (local {local_size} ≠ remoto {remote_size})",
        )

    lo = normalize_oid(local_oid)
    ro = normalize_oid(remote_oid)
    if lo and ro:
        if lo == ro:
            return FileVerdict(UP_TO_DATE, True, "sha256 confere")
        return FileVerdict(UPDATE_AVAILABLE, True, "mesmo tamanho, sha256 difere")

    # Tamanho bate mas não dá pra confirmar o conteúdo (faltou um dos oids).
    return FileVerdict(UP_TO_DATE, False, "tamanho confere (sha256 não verificado)")


def aggregate(verdicts: list[FileVerdict]) -> tuple[str, bool]:
    """Reduz os vereditos de arquivo ao status do modelo inteiro.

    Prioridade: qualquer UPDATE vence; senão qualquer UNKNOWN; senão UP_TO_DATE.
    `verified` só é True quando é UP_TO_DATE e todos os arquivos foram verificados.
    """
    if not verdicts:
        return UNKNOWN, False
    if any(v.status == UPDATE_AVAILABLE for v in verdicts):
        return UPDATE_AVAILABLE, True
    if any(v.status == UNKNOWN for v in verdicts):
        return UNKNOWN, False
    return UP_TO_DATE, all(v.verified for v in verdicts)


def infer_repo_id(model_path: Path, roots: list[Path]) -> str | None:
    """`owner/repo` a partir do layout `<root>/<owner>/<repo>/.../arquivo.gguf`.

    None se o modelo não está sob nenhuma raiz conhecida, ou está solto direto na
    raiz (sem os dois níveis owner/repo) — aí não há repo de origem pra consultar.
    """
    for root in roots:
        try:
            rel = model_path.relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) >= 3:  # owner / repo / … / arquivo.gguf
            return f"{parts[0]}/{parts[1]}"
        return None
    return None


def match_remote(name: str, remote_files: list[dict]) -> dict | None:
    """Acha no tree do HF o arquivo cujo basename bate com `name`.

    O downloader achata `subdir/arquivo.gguf` → `arquivo.gguf` em disco, então a
    comparação é sempre por basename. Em caso de colisão fica o de maior tamanho
    (o quant real, não um placeholder)."""
    name_l = name.lower()
    matches = [f for f in remote_files if Path(f["path"]).name.lower() == name_l]
    if not matches:
        return None
    return max(matches, key=lambda f: int(f.get("size") or 0))


# ─── origin.json (sidecar de procedência) ─────────────────────────────────────

def origin_path(base_dir: Path) -> Path:
    return base_dir / ORIGIN_NAME


def read_origin(model_path: Path) -> dict | None:
    """Lê o origin.json na pasta do modelo ou na pasta-pai (quant em subdir)."""
    for folder in (model_path.parent, model_path.parent.parent):
        f = folder / ORIGIN_NAME
        try:
            if f.is_file():
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("repo_id"):
                    return data
        except Exception:
            continue
    return None


def write_origin(base_dir: Path, repo_id: str, branch: str, files: list[dict]) -> Path | None:
    """Grava a procedência do download. `files`: [{rel, size, oid}]. Best-effort."""
    payload = {
        "repo_id": repo_id,
        "branch": branch,
        "recorded_at": int(time.time()),
        "files": {
            Path(f["rel"]).name: {
                "rel": f["rel"],
                "size": int(f.get("size") or 0),
                "oid": normalize_oid(f.get("oid")),
            }
            for f in files
        },
    }
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
        dest = base_dir / ORIGIN_NAME
        dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return dest
    except Exception:
        return None


# ─── cache de oid computado ───────────────────────────────────────────────────

def _read_oid_cache() -> dict:
    try:
        if _OID_CACHE_FILE.exists():
            data = json.loads(_OID_CACHE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _write_oid_cache(cache: dict) -> None:
    try:
        _OID_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _OID_CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _cache_key(path: Path) -> tuple[str, int, int] | None:
    try:
        st = path.stat()
        return (str(path), int(st.st_mtime), int(st.st_size))
    except OSError:
        return None


def cached_local_oid(path: Path) -> str | None:
    """oid do cache SÓ se path/mtime/size ainda batem. Nunca hasheia aqui."""
    key = _cache_key(path)
    if key is None:
        return None
    entry = _read_oid_cache().get(str(path))
    if isinstance(entry, dict) and entry.get("mtime") == key[1] and entry.get("size") == key[2]:
        return normalize_oid(entry.get("oid"))
    return None


def compute_local_oid(path: Path) -> str | None:
    """sha256 do arquivo em disco (git-lfs oid), com cache por (path, mtime, size).

    Custo O(tamanho) — reservado à "checagem a fundo", nunca no caminho crítico
    da inicialização."""
    key = _cache_key(path)
    if key is None:
        return None
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    oid = h.hexdigest()
    cache = _read_oid_cache()
    cache[str(path)] = {"mtime": key[1], "size": key[2], "oid": oid}
    _write_oid_cache(cache)
    return oid


# ─── orquestração ─────────────────────────────────────────────────────────────

def _local_files_for(model_path: Path) -> list[Path]:
    """.gguf(s) do modelo (quant + partes split) + mmproj associado."""
    folder = model_path.parent
    files: list[Path] = []
    m = models_repo.SPLIT_RE.match(model_path.name)
    if m:
        base = m.group("base")
        for f in sorted(folder.glob("*.gguf")):
            if not models_repo.is_mmproj(f.name) and models_repo.gguf_group_key(f.name) == base:
                files.append(f)
    else:
        files.append(model_path)
    mm = models_repo.find_mmproj(folder, include_parent=False)
    if mm and mm not in files:
        files.append(mm)
    return files


def _resolve_local_oid(path: Path, origin: dict | None, deep: bool) -> str | None:
    """oid local pela fonte mais barata disponível. Só hasheia se `deep`."""
    if origin:
        entry = (origin.get("files") or {}).get(path.name)
        if isinstance(entry, dict):
            # Só confia no oid registrado se o arquivo não mudou de tamanho desde
            # o download (proxy barato pra "é o mesmo arquivo").
            try:
                if int(entry.get("size") or -1) == path.stat().st_size:
                    oid = normalize_oid(entry.get("oid"))
                    if oid:
                        return oid
            except OSError:
                pass
    cached = cached_local_oid(path)
    if cached:
        return cached
    if deep:
        return compute_local_oid(path)
    return None


def check_model(
    model_path: Path,
    roots: list[Path],
    deep: bool = False,
    remote_cache: dict[str, list[dict]] | None = None,
) -> dict:
    """Confere um modelo contra o HF. Devolve dict serializável.

    `remote_cache` (repo_id → tree) evita re-consultar o mesmo repo pra vários
    quants. `deep=True` autoriza hashear arquivos ainda não verificados.

    Nunca levanta: rede caindo vira status UNKNOWN, não erro.
    """
    origin = read_origin(model_path)
    repo_id = (origin or {}).get("repo_id") or infer_repo_id(model_path, roots)
    branch = (origin or {}).get("branch") or "main"

    result = {
        "path": str(model_path),
        "repo_id": repo_id,
        "branch": branch,
        "status": UNKNOWN,
        "verified": False,
        "has_origin": origin is not None,
        # base_dir + rel_paths alimentam o botão "Atualizar", que reusa o mesmo
        # pipeline de download. rel_paths é o caminho REMOTO que casou (não o
        # basename local achatado), então re-baixar acerta até repo com subdir.
        "base_dir": str(model_path.parent),
        "rel_paths": [],
        "files": [],
        "error": None,
    }
    if not repo_id:
        result["error"] = "sem repositório de origem (modelo fora do layout owner/repo)"
        return result

    try:
        if remote_cache is not None and repo_id in remote_cache:
            remote_files = remote_cache[repo_id]
        else:
            remote_files = hf.hf_list_files(repo_id, branch)
            if remote_cache is not None:
                remote_cache[repo_id] = remote_files
    except Exception as e:
        result["error"] = f"falha ao consultar o HuggingFace: {e}"
        return result

    verdicts: list[FileVerdict] = []
    for lf in _local_files_for(model_path):
        try:
            local_size = lf.stat().st_size
        except OSError:
            continue
        remote = match_remote(lf.name, remote_files)
        local_oid = _resolve_local_oid(lf, origin, deep) if remote else None
        v = compare_file(
            local_size, local_oid,
            int(remote["size"]) if remote else None,
            (remote or {}).get("oid"),
        )
        verdicts.append(v)
        rel = remote["path"] if remote else None
        if rel:
            result["rel_paths"].append(rel)
        result["files"].append({
            "name": lf.name,
            "rel": rel,
            "status": v.status,
            "verified": v.verified,
            "reason": v.reason,
        })

    status, verified = aggregate(verdicts)
    result["status"] = status
    result["verified"] = verified
    return result
