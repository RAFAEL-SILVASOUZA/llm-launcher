"""Persistência das configurações em last_config.json.

Cada entrada tem um `id` único e estável — é por ele que se identifica a config.
Isso permite múltiplas configs do mesmo (model, backend) (ex.: o mesmo GGUF com
context windows diferentes) sem que uma sobrescreva a outra. Configs legadas sem
`id` recebem um uuid no primeiro load (migração best-effort). Configs legadas sem
`backend` são tratadas como 'turbo' (era o único backend antes do split).
"""
import json
import time
import uuid
from pathlib import Path

from .constants import CONFIG_FILE, FAIL_HISTORY_FILE


def _write(configs: list[dict], path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception:
        return False


def read_all_configs(path: Path = CONFIG_FILE) -> list[dict]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                # Migração: backfill de ids em configs legadas.
                changed = False
                for c in data:
                    if isinstance(c, dict) and not c.get("id"):
                        c["id"] = uuid.uuid4().hex
                        changed = True
                if changed:
                    _write(data, path)
                return data
    except Exception:
        pass
    return []


def save_config(cfg: dict, path: Path = CONFIG_FILE) -> dict:
    """Insere ou atualiza uma config identificada por `id`.

    Se `cfg` já tem `id`, a entrada com esse id é substituída (update in place).
    Se não tem, um id novo é gerado e a config é anexada (nunca sobrescreve
    outra). O `id` atribuído é gravado no próprio `cfg` (mutação) e a config
    resultante é retornada.
    """
    configs = read_all_configs(path)
    cfg_id = cfg.get("id")
    if cfg_id:
        configs = [c for c in configs if c.get("id") != cfg_id]
    else:
        cfg_id = uuid.uuid4().hex
        cfg["id"] = cfg_id
    cfg["last_used"] = time.time()
    configs.append(cfg)
    _write(configs, path)
    return cfg


def load_config(model_path: str | Path, backend: str, path: Path = CONFIG_FILE) -> dict | None:
    model_str = str(model_path)
    for cfg in read_all_configs(path):
        if cfg.get("model") == model_str and cfg.get("backend", "turbo") == backend:
            return cfg
    return None


def delete_config_by_id(config_id: str, path: Path = CONFIG_FILE) -> int:
    """Remove a config com o `id` dado. Retorna quantas foram removidas (0 ou 1)."""
    configs = read_all_configs(path)
    kept = [c for c in configs if c.get("id") != config_id]
    removed = len(configs) - len(kept)
    if removed:
        if not _write(kept, path):
            return 0
    return removed


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
        if not _write(kept, path):
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
