"""HuggingFace download utilities — portado de models.py.

Inclui parse de URLs, listagem de arquivos via API, busca por mirrors GGUF,
e download com progresso streamável (yield em vez de print).
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterator

from .constants import MODELS_DIR
from .models_repo import SPLIT_RE, gguf_group_key, is_mmproj


def hf_headers() -> dict[str, str]:
    headers = {"User-Agent": "llm-launcher/0.1"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def parse_hf_url(url: str) -> tuple[str, str | None, str]:
    """Extrai (repo_id, filename, branch). filename=None se for o repo inteiro.

    Suporta:
      - https://huggingface.co/{owner}/{repo}
      - https://huggingface.co/{owner}/{repo}?show_file_info={file}
      - https://huggingface.co/{owner}/{repo}/(resolve|blob|tree)/{branch}[/{file.gguf}]
    """
    url = url.strip().strip('"').strip("'")
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc not in ("huggingface.co", "www.huggingface.co"):
        raise ValueError("A URL precisa ser do huggingface.co")

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError("Não consegui identificar o repositório na URL")

    owner, repo = parts[0], parts[1]
    repo_id = f"{owner}/{repo}"
    branch = "main"
    filename: str | None = None

    if len(parts) >= 4 and parts[2] in ("resolve", "blob", "tree"):
        branch = parts[3]
        rest = "/".join(parts[4:])
        filename = rest or None
    else:
        qs = urllib.parse.parse_qs(parsed.query)
        for key in ("show_file_info", "show-file-info", "filename"):
            if qs.get(key):
                filename = qs[key][0]
                break

    return repo_id, (urllib.parse.unquote(filename) if filename else None), branch


def hf_list_files(repo_id: str, branch: str = "main") -> list[dict]:
    """Lista os arquivos do repo. Retorna dicts {path, size}.

    Usa o endpoint `tree?recursive=true` em vez do `siblings` do model card:
    o último retorna `size=0` pra arquivos LFS (limitação conhecida da API),
    e a app precisa de tamanho real pra validar integridade do download.
    """
    qs = urllib.parse.urlencode({"recursive": "true", "expand": "true"})
    req = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo_id}/tree/{branch}?{qs}",
        headers=hf_headers(),
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        entries = json.loads(resp.read().decode("utf-8"))
    out = []
    for e in entries:
        if e.get("type") != "file":
            continue
        path = e.get("path")
        if not path:
            continue
        # `lfs.size` é o tamanho do blob LFS; `size` é o tamanho do pointer
        # quando LFS. Pegamos o maior dos dois.
        lfs = e.get("lfs") or {}
        size = max(int(e.get("size") or 0), int(lfs.get("size") or 0))
        out.append({"path": path, "size": size})
    return out


def hf_search_gguf(query: str, limit: int = 25) -> list[dict]:
    """Busca repositórios com GGUF batendo `query`. Ordenado por downloads."""
    qs = urllib.parse.urlencode({
        "search": query, "filter": "gguf",
        "sort": "downloads", "direction": -1, "limit": limit,
    })
    req = urllib.request.Request(
        f"https://huggingface.co/api/models?{qs}", headers=hf_headers()
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        results = json.loads(resp.read().decode("utf-8"))
    out = []
    for r in results:
        rid = r.get("id") or r.get("modelId") or ""
        if "/" in rid:
            out.append({
                "repo_id":   rid,
                "downloads": r.get("downloads", 0) or 0,
                "likes":     r.get("likes", 0) or 0,
            })
    return out


def group_gguf_files(repo_files: list[dict]) -> dict:
    """Agrupa os GGUFs do repo em quants + mmprojs.

    Retorna {
      "mmprojs":  [{"path": ..., "size": ...}],
      "quants":   {"<base>": [{"path", "size"}, ...]},
    }
    """
    gguf = [f for f in repo_files if f["path"].lower().endswith(".gguf")]
    mmprojs = [f for f in gguf if is_mmproj(f["path"])]
    quants: dict[str, list[dict]] = {}
    for f in gguf:
        if is_mmproj(f["path"]):
            continue
        key = gguf_group_key(f["path"])
        quants.setdefault(key, []).append(f)
    return {"mmprojs": mmprojs, "quants": quants}


def _human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


class IncompleteDownload(Exception):
    """Levantada quando o arquivo termina menor que o esperado.

    Mantém o `.part` em disco pra próxima tentativa retomar — o caller
    decide se reentra ou reporta pro usuário.
    """


def download_file_stream(
    url: str,
    dest: Path,
    on_progress: "callable | None" = None,
    expected_size: int = 0,
) -> None:
    """Baixa `url` para `dest` com retomada via .part. `on_progress` recebe
    (downloaded_bytes, total_bytes, speed_bps) periodicamente.

    Se `expected_size > 0`, valida que o resultado tem exatamente esse tamanho
    antes de renomear `.part` → dest. Caso contrário levanta IncompleteDownload
    sem renomear, preservando os bytes baixados pra retomada.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")

    resume_pos = tmp.stat().st_size if tmp.exists() else 0
    headers = hf_headers()
    if resume_pos:
        headers["Range"] = f"bytes={resume_pos}-"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        clen = resp.headers.get("Content-Length")
        partial = getattr(resp, "status", 200) == 206
        if resume_pos and partial:
            file_mode = "ab"
            total_http = (int(clen) + resume_pos) if clen else 0
        else:
            resume_pos = 0
            file_mode = "wb"
            total_http = int(clen) if clen else 0

        # Prefere expected_size (vem da API do HF, mais confiável); cai pro
        # Content-Length quando o caller não passou tamanho.
        total = expected_size if expected_size > 0 else total_http

        downloaded = resume_pos
        start = time.time()
        last = 0.0
        with open(tmp, file_mode) as f:
            while True:
                chunk = resp.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if on_progress and now - last >= 0.4:
                    last = now
                    speed = (downloaded - resume_pos) / max(now - start, 1e-6)
                    on_progress(downloaded, total, speed)

    # Validação pós-download: se o servidor encerrou prematuramente (network
    # blip que vira EOF "limpo"), o loop sai sem exceção e antes desta checagem
    # o `.part` seria renomeado mesmo truncado. Não rename — deixa pra retomar.
    if expected_size > 0 and downloaded != expected_size:
        raise IncompleteDownload(
            f"download truncado: recebeu {downloaded} bytes, esperava {expected_size} "
            f"(faltam {expected_size - downloaded} bytes). Arquivo .part mantido pra retomada."
        )

    tmp.replace(dest)
    if on_progress:
        on_progress(downloaded, downloaded, 0.0)  # final tick


def plan_download(
    repo_id: str,
    branch: str,
    rel_paths: list[str],
    subdir: str | None = None,
    base_root: Path | None = None,
) -> dict:
    """Devolve o plano: arquivos esperados + caminhos de destino + tamanhos.

    base_root é a pasta raiz onde criar o subdir owner/repo. Quando None, cai
    pro MODELS_DIR — só pra manter compat de chamadas antigas; o caminho novo
    sempre passa base_root explícito.

    Resolve `expected_size` via hf_list_files (endpoint tree?recursive=true),
    que devolve o tamanho real do blob LFS — o downloader usa pra validar
    integridade no fim. Falha silenciosa de tamanho => não rename.
    """
    owner, repo = repo_id.split("/", 1)
    root = base_root or MODELS_DIR
    base_dir = root / owner / repo
    if subdir:
        base_dir = base_dir / subdir

    sizes: dict[str, int] = {}
    try:
        for f in hf_list_files(repo_id, branch):
            sizes[f["path"]] = int(f.get("size") or 0)
    except Exception:
        # Sem tamanho a UI ainda funciona — só perde a validação pós-download
        # pra esses arquivos específicos.
        pass

    items = []
    for rel in rel_paths:
        dest = base_dir / Path(rel).name
        items.append({
            "rel":           rel,
            "dest":          str(dest),
            "exists":        dest.exists(),
            "size_disk":     dest.stat().st_size if dest.exists() else 0,
            "expected_size": sizes.get(rel, 0),
            "url":           f"https://huggingface.co/{repo_id}/resolve/{branch}/{rel}?download=true",
        })
    return {
        "repo_id":  repo_id,
        "branch":   branch,
        "base_dir": str(base_dir),
        "items":    items,
    }


def stream_download(plan: dict) -> Iterator[dict]:
    """Generator de eventos do download:
        {"type": "file_start", "rel", "dest", "index", "total"}
        {"type": "progress",   "rel", "downloaded", "total", "speed"}
        {"type": "file_done",  "rel", "dest"}
        {"type": "file_skip",  "rel", "dest", "reason"}
        {"type": "error",      "rel", "message"}
        {"type": "done"}
    """
    items = plan["items"]
    total = len(items)
    for idx, it in enumerate(items, 1):
        rel  = it["rel"]
        dest = Path(it["dest"])
        expected = int(it.get("expected_size") or 0)
        if dest.exists():
            actual = dest.stat().st_size
            # Skip seguro: só pula se o tamanho bate (ou se a API não devolveu
            # tamanho — aí confia no que está em disco). Se está em disco mas
            # com tamanho errado, deleta e re-baixa — esse era o caso silencioso
            # que deixava arquivo truncado pro llama-server engasgar depois.
            if expected == 0 or actual == expected:
                yield {"type": "file_skip", "rel": rel, "dest": str(dest),
                       "reason": "already_exists", "size": actual}
                continue
            try:
                dest.unlink()
            except Exception:
                pass
            yield {"type": "file_start", "rel": rel, "dest": str(dest),
                   "index": idx, "total": total,
                   "note": f"arquivo em disco tem tamanho errado ({actual} != {expected}) — re-baixando"}
        else:
            yield {"type": "file_start", "rel": rel, "dest": str(dest),
                   "index": idx, "total": total}
        try:
            def emit(d: int, t: int, s: float, _rel=rel):
                # closure local pra capturar o rel atual
                _emit_buf.append({"type": "progress", "rel": _rel,
                                  "downloaded": d, "total": t, "speed": s})

            _emit_buf: list[dict] = []
            # Stream em chunks através do callback. Como o urllib.request é
            # síncrono, usamos buffer pra preservar a ordem: o callback enche
            # _emit_buf; após cada chunk principal flushamos. Simples e
            # suficiente pra UI sentir progresso.
            from threading import Thread
            import queue
            q: queue.Queue = queue.Queue()

            def runner(_expected=expected):
                try:
                    download_file_stream(
                        it["url"], dest,
                        on_progress=lambda d, t, s: q.put(("p", d, t, s)),
                        expected_size=_expected,
                    )
                    q.put(("done", None, None, None))
                except Exception as e:
                    q.put(("err", str(e), None, None))

            t = Thread(target=runner, daemon=True)
            t.start()
            while True:
                msg = q.get()
                tag = msg[0]
                if tag == "p":
                    _, d, tot, sp = msg
                    yield {"type": "progress", "rel": rel,
                           "downloaded": int(d), "total": int(tot or 0),
                           "speed": float(sp or 0.0)}
                elif tag == "err":
                    yield {"type": "error", "rel": rel, "message": msg[1]}
                    break
                else:
                    yield {"type": "file_done", "rel": rel, "dest": str(dest)}
                    break
        except Exception as e:
            yield {"type": "error", "rel": rel, "message": str(e)}
    yield {"type": "done"}
