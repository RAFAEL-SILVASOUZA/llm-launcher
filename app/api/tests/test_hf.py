"""Testes da lógica pura do downloader do HuggingFace."""
import sys
from pathlib import Path

_API_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_API_ROOT))

from api.core import hf  # noqa: E402


# ─── skip_existing ────────────────────────────────────────────────────────────

def test_skip_when_size_matches():
    assert hf.skip_existing(actual=100, expected=100) is True


def test_no_skip_when_size_differs():
    # Truncado ou desatualizado: re-baixa.
    assert hf.skip_existing(actual=50, expected=100) is False


def test_skip_when_remote_size_unknown():
    # API não devolveu tamanho — confia no que está em disco.
    assert hf.skip_existing(actual=100, expected=0) is True


def test_force_beats_matching_size():
    # Caso do botão "Atualizar": mesmo tamanho, sha256 diferente. Sem isto o
    # update terminava "concluído" sem trocar byte nenhum.
    assert hf.skip_existing(actual=100, expected=100, force=True) is False


def test_force_beats_unknown_size():
    assert hf.skip_existing(actual=100, expected=0, force=True) is False
