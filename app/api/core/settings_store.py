"""Persistência de configurações da app (não-launch).

Hoje guarda só `model_paths` — lista de pastas onde varremos GGUF. O default
(quando o arquivo não existe ou está vazio) é a pasta padrão do LM Studio
em `MODELS_DIR`. Salvar pela UI sobrescreve.
"""
from __future__ import annotations

import json
from pathlib import Path

from .constants import CONFIG_FILE, MODELS_DIR

SETTINGS_FILE = CONFIG_FILE.with_name("app_settings.json")


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


def read_settings(path: Path = SETTINGS_FILE) -> dict:
    """Snapshot para o frontend — devolve só o que está realmente configurado,
    pra UI distinguir 'não configurado' de 'usando default'."""
    return {
        "model_paths": [str(p) for p in get_configured_paths(path)],
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
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return read_settings(path)
