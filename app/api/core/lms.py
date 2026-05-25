"""Integração com LM Studio (lms CLI) — fallback pra modelos mmproj quebrados."""
import json
import subprocess
from pathlib import Path

from .constants import LMS, MODELS_DIR
from .models_repo import make_alias


def lms_available() -> bool:
    return LMS.exists()


def lms_status() -> dict:
    """Status do servidor LM Studio (running/stopped/unavailable)."""
    if not LMS.exists():
        return {"available": False, "running": False, "path": str(LMS)}
    try:
        r = subprocess.run(
            [str(LMS), "server", "status"],
            capture_output=True, text=True, encoding="utf-8", timeout=10,
        )
        out = ((r.stdout or "") + (r.stderr or "")).lower()
        running = "running" in out and "not running" not in out
        return {"available": True, "running": running, "path": str(LMS),
                "raw": (r.stdout or r.stderr or "").strip()[:300]}
    except Exception as e:
        return {"available": True, "running": False, "path": str(LMS),
                "error": str(e)}


def lms_model_key(model_path: Path) -> str | None:
    """Resolve modelKey (ex: 'nvidia/nemotron@q4_k_m') via 'lms ls --json --variants'."""
    if not LMS.exists():
        return None
    try:
        result = subprocess.run(
            [str(LMS), "ls", "--json", "--variants"],
            capture_output=True, text=True, encoding="utf-8", timeout=20,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout or "[]")
    except Exception:
        return None
    try:
        rel = model_path.relative_to(MODELS_DIR).as_posix().lower()
    except ValueError:
        return None
    for entry in data:
        base_key = (entry.get("model") or {}).get("modelKey")
        for variant in entry.get("variants", []):
            ident = variant.get("indexedModelIdentifier", "")
            if "@" not in ident:
                continue
            _, fs_path = ident.split("@", 1)
            if fs_path.lower() == rel:
                return base_key
    return None


def load_via_lms(model_path: Path, context_window: int, parallel_slots: int) -> dict:
    """Sobe o LM Studio server e carrega o modelo. Devolve {ok, output, key}."""
    if not LMS.exists():
        return {"ok": False, "error": f"lms.exe não encontrado em {LMS}"}

    alias = make_alias(model_path)
    key = lms_model_key(model_path)
    if key is None:
        return {"ok": False, "error": f"Não consegui resolver a key do lms para: {model_path.name}"}

    start = subprocess.run(
        [str(LMS), "server", "start", "--port", "1234", "--bind", "0.0.0.0", "--cors"],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    load = subprocess.run(
        [
            str(LMS), "load", key,
            "--gpu", "max",
            "--context-length", str(context_window * parallel_slots),
            "--parallel", str(parallel_slots),
            "--identifier", alias,
            "-y",
        ],
        capture_output=True, text=True, encoding="utf-8", timeout=300,
    )
    return {
        "ok": load.returncode == 0,
        "key": key,
        "alias": alias,
        "start_output": ((start.stdout or "") + (start.stderr or "")).strip()[:500],
        "load_output":  ((load.stdout or "") + (load.stderr or "")).strip()[:1500],
        "rc": load.returncode,
    }
