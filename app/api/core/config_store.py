"""Persistência das configurações em last_config.json.

Cada entrada é chaveada por (model, backend) — mudar de turbo p/ vanilla não
sobrescreve a config anterior. Configs legadas sem `backend` são tratadas
como 'turbo' (era o único backend antes do split).
"""
import json
import time
from pathlib import Path

from .constants import CONFIG_FILE, FAIL_HISTORY_FILE


def read_all_configs(path: Path = CONFIG_FILE) -> list[dict]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def save_config(cfg: dict, path: Path = CONFIG_FILE) -> None:
    """Sobrescreve a entrada do (model, backend); cria se não existir."""
    configs = read_all_configs(path)
    model_key   = cfg.get("model", "")
    backend_key = cfg.get("backend", "turbo")
    configs = [
        c for c in configs
        if not (c.get("model") == model_key and c.get("backend", "turbo") == backend_key)
    ]
    configs.append(cfg)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def load_config(model_path: str | Path, backend: str, path: Path = CONFIG_FILE) -> dict | None:
    model_str = str(model_path)
    for cfg in read_all_configs(path):
        if cfg.get("model") == model_str and cfg.get("backend", "turbo") == backend:
            return cfg
    return None


def delete_config(model_path: str | Path, backend: str | None = None, path: Path = CONFIG_FILE) -> int:
    """Remove configs do modelo. Se backend=None, remove todas as variantes.

    Retorna quantas entradas foram removidas.
    """
    model_str = str(model_path)
    configs = read_all_configs(path)
    if backend is None:
        kept = [c for c in configs if c.get("model") != model_str]
    else:
        kept = [
            c for c in configs
            if not (c.get("model") == model_str and c.get("backend", "turbo") == backend)
        ]
    removed = len(configs) - len(kept)
    if removed:
        try:
            path.write_text(json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            return 0
    return removed


def append_fail_history(
    cfg: dict,
    failure: str,
    error_excerpt: str,
    degrade_applied: str | None,
    attempt: int,
    path: Path = FAIL_HISTORY_FILE,
) -> None:
    """Loga uma tentativa fracassada em jsonl. Best-effort."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts":              time.strftime("%Y-%m-%dT%H:%M:%S"),
            "attempt":         attempt,
            "model":           cfg.get("model"),
            "backend":         cfg.get("backend"),
            "failure":         failure,
            "degrade_applied": degrade_applied,
            "error_excerpt":   error_excerpt[:400] if error_excerpt else "",
            "config":          cfg,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
