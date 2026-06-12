"""Persistência de servidores MCP cadastrados pelo usuário.

Cada entrada: {id, name, cwd, command, enabled, created_at}.
- id: hex curto, estável (usado em URLs e refs)
- name: rótulo amigável
- cwd: pasta de trabalho onde o comando é executado
- command: linha shell (ex: "npm run dev")
- enabled: hint persistido — "ligado da última vez". O runner em memória
  é a verdade pra estado vivo.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from .constants import CONFIG_FILE

MCP_FILE = CONFIG_FILE.with_name("mcp_servers.json")


def _read_raw() -> list[dict]:
    try:
        if MCP_FILE.exists():
            data = json.loads(MCP_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def _write_raw(entries: list[dict]) -> None:
    try:
        MCP_FILE.parent.mkdir(parents=True, exist_ok=True)
        MCP_FILE.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def list_servers() -> list[dict]:
    return _read_raw()


def get_server(server_id: str) -> dict | None:
    for s in _read_raw():
        if s.get("id") == server_id:
            return s
    return None


def upsert_server(payload: dict) -> dict:
    """Insere ou atualiza. Retorna a entrada final (com id garantido)."""
    entries = _read_raw()
    sid = (payload.get("id") or "").strip()
    if not sid:
        sid = uuid.uuid4().hex[:12]
    entry = {
        "id":         sid,
        "name":       str(payload.get("name", "")).strip() or sid,
        "cwd":        str(payload.get("cwd", "")).strip(),
        "command":    str(payload.get("command", "")).strip(),
        "enabled":    bool(payload.get("enabled", False)),
        "created_at": payload.get("created_at") or time.time(),
    }
    out: list[dict] = []
    replaced = False
    for e in entries:
        if e.get("id") == sid:
            out.append(entry)
            replaced = True
        else:
            out.append(e)
    if not replaced:
        out.append(entry)
    _write_raw(out)
    return entry


def delete_server(server_id: str) -> bool:
    entries = _read_raw()
    kept = [e for e in entries if e.get("id") != server_id]
    if len(kept) == len(entries):
        return False
    _write_raw(kept)
    return True


def set_enabled(server_id: str, enabled: bool) -> None:
    """Persiste apenas o flag enabled — usado quando o runner liga/desliga."""
    entries = _read_raw()
    changed = False
    for e in entries:
        if e.get("id") == server_id and e.get("enabled") != enabled:
            e["enabled"] = enabled
            changed = True
    if changed:
        _write_raw(entries)
