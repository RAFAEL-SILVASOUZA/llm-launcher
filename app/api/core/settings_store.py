"""Persistência de configurações da app (não-launch).

Guarda hoje:
  - `model_paths`: pastas para varrer .gguf (fallback: MODELS_DIR).
  - `backend_paths`: override do diretório dos binários por backend
    (turbo/vanilla/mtp). Quando vazio/ausente, cai pro hardcoded default em
    constants.BACKENDS — pra que o app continue funcionando se o usuário
    nunca configurou.
"""
from __future__ import annotations

import json
from pathlib import Path

from .constants import BACKEND_BINARY_ALIAS, BACKENDS, CONFIG_FILE, MODELS_DIR

SETTINGS_FILE = CONFIG_FILE.with_name("app_settings.json")

# Backends cujo diretório o usuário pode customizar pela UI. Backends-alias
# (ex: mtp, que reusa o build do vanilla) ficam de fora: não têm binário
# próprio, então seguem o path do alvo e não viram linha no Settings.
CONFIGURABLE_BACKENDS = tuple(k for k in BACKENDS if k not in BACKEND_BINARY_ALIAS)


def _read_raw(path: Path = SETTINGS_FILE) -> dict:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def get_configured_paths(path: Path = SETTINGS_FILE) -> list[Path]:
    """Apenas o que está explicitamente no arquivo — sem fallback.

    Usado para validar destinos de download (precisa estar cadastrado).
    """
    raw = _read_raw(path).get("model_paths")
    if isinstance(raw, list):
        return [Path(str(p)) for p in raw if str(p).strip()]
    return []


def get_scan_paths(path: Path = SETTINGS_FILE) -> list[Path]:
    """Pastas para varrer GGUF — cai pro MODELS_DIR default se vazio."""
    paths = get_configured_paths(path)
    return paths if paths else [MODELS_DIR]


# Compat alias — algumas chamadas antigas continuam pedindo "model paths"
# pensando em scan; mantém o nome curto e aponta pro scan.
get_model_paths = get_scan_paths


def _read_backend_paths(path: Path = SETTINGS_FILE) -> dict[str, str]:
    """Dict {backend: dir} apenas com os explicitamente configurados."""
    raw = _read_raw(path).get("backend_paths")
    out: dict[str, str] = {}
    if isinstance(raw, dict):
        for name in CONFIGURABLE_BACKENDS:
            v = raw.get(name)
            if isinstance(v, str) and v.strip():
                out[name] = v.strip()
    return out


def get_backend_dir(backend: str, path: Path = SETTINGS_FILE) -> Path | None:
    """Diretório dos binários daquele backend, se configurado pelo usuário.
    Backends-alias (mtp→vanilla) herdam o override do alvo."""
    target = BACKEND_BINARY_ALIAS.get(backend, backend)
    raw = _read_backend_paths(path).get(target)
    return Path(raw) if raw else None


def default_backend_dir(backend: str) -> Path:
    """Diretório default (parent do binário em constants.BACKENDS).
    Assumimos server e cli no mesmo diretório — é o layout do CMake build.
    Backends-alias (mtp→vanilla) herdam o default do alvo."""
    target = BACKEND_BINARY_ALIAS.get(backend, backend)
    return BACKENDS[target]["server"].parent


def read_settings(path: Path = SETTINGS_FILE) -> dict:
    """Snapshot para o frontend — devolve só o que está realmente configurado,
    pra UI distinguir 'não configurado' de 'usando default'."""
    return {
        "model_paths":            [str(p) for p in get_configured_paths(path)],
        "backend_paths":          _read_backend_paths(path),
        "backend_paths_defaults": {
            name: str(default_backend_dir(name)) for name in CONFIGURABLE_BACKENDS
        },
    }


def save_settings(data: dict, path: Path = SETTINGS_FILE) -> dict:
    """Sobrescreve o arquivo com as chaves conhecidas. Ignora extras."""
    current = _read_raw(path)
    raw_paths = data.get("model_paths")
    if isinstance(raw_paths, list):
        cleaned: list[str] = []
        seen: set[str] = set()
        for p in raw_paths:
            s = str(p).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            cleaned.append(s)
        current["model_paths"] = cleaned

    raw_backends = data.get("backend_paths")
    if isinstance(raw_backends, dict):
        out: dict[str, str] = {}
        for name in CONFIGURABLE_BACKENDS:
            v = raw_backends.get(name)
            if isinstance(v, str) and v.strip():
                out[name] = v.strip()
        current["backend_paths"] = out

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return read_settings(path)
