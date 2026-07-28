"""Testes da lógica pura de detecção de atualização de GGUF."""
import sys
from pathlib import Path

_API_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_API_ROOT))

from api.core import updates  # noqa: E402
from api.core.updates import UP_TO_DATE, UPDATE_AVAILABLE, UNKNOWN  # noqa: E402


# ─── normalize_oid ────────────────────────────────────────────────────────────

def test_normalize_oid_strips_prefix_and_lowercases():
    assert updates.normalize_oid("sha256:ABC123") == "abc123"
    assert updates.normalize_oid("ABC123") == "abc123"


def test_normalize_oid_none_and_empty():
    assert updates.normalize_oid(None) is None
    assert updates.normalize_oid("") is None


# ─── compare_file ─────────────────────────────────────────────────────────────

def test_compare_file_size_mismatch_is_update():
    v = updates.compare_file(local_size=100, local_oid=None,
                             remote_size=200, remote_oid=None)
    assert v.status == UPDATE_AVAILABLE
    assert v.verified is True  # tamanho é sinal definitivo


def test_compare_file_missing_remote_is_unknown():
    v = updates.compare_file(local_size=100, local_oid="aa",
                             remote_size=None, remote_oid=None)
    assert v.status == UNKNOWN


def test_compare_file_same_size_matching_oid_up_to_date():
    v = updates.compare_file(local_size=100, local_oid="aa",
                             remote_size=100, remote_oid="AA")
    assert v.status == UP_TO_DATE
    assert v.verified is True


def test_compare_file_same_size_differing_oid_is_update():
    v = updates.compare_file(local_size=100, local_oid="aa",
                             remote_size=100, remote_oid="bb")
    assert v.status == UPDATE_AVAILABLE
    assert v.verified is True


def test_compare_file_same_size_unknown_local_oid_is_optimistic_unverified():
    # Não hasheamos ainda; tamanho bate → tratamos como em dia, mas não verificado.
    v = updates.compare_file(local_size=100, local_oid=None,
                             remote_size=100, remote_oid="bb")
    assert v.status == UP_TO_DATE
    assert v.verified is False


def test_compare_file_same_size_no_remote_oid_unverified():
    # Arquivo não-LFS no remoto (sem oid): só dá pra confiar no tamanho.
    v = updates.compare_file(local_size=100, local_oid="aa",
                             remote_size=100, remote_oid=None)
    assert v.status == UP_TO_DATE
    assert v.verified is False


# ─── aggregate ────────────────────────────────────────────────────────────────

def _v(status, verified=True):
    return updates.FileVerdict(status=status, verified=verified, reason="")


def test_aggregate_any_update_wins():
    status, verified = updates.aggregate([
        _v(UP_TO_DATE), _v(UPDATE_AVAILABLE), _v(UNKNOWN),
    ])
    assert status == UPDATE_AVAILABLE


def test_aggregate_unknown_when_no_update_and_some_unknown():
    status, verified = updates.aggregate([_v(UP_TO_DATE), _v(UNKNOWN)])
    assert status == UNKNOWN


def test_aggregate_up_to_date_when_all_up_to_date():
    status, verified = updates.aggregate([_v(UP_TO_DATE), _v(UP_TO_DATE)])
    assert status == UP_TO_DATE
    assert verified is True


def test_aggregate_up_to_date_unverified_propagates():
    status, verified = updates.aggregate([
        _v(UP_TO_DATE, verified=True), _v(UP_TO_DATE, verified=False),
    ])
    assert status == UP_TO_DATE
    assert verified is False


def test_aggregate_empty_is_unknown():
    status, verified = updates.aggregate([])
    assert status == UNKNOWN


# ─── infer_repo_id ────────────────────────────────────────────────────────────

def test_infer_repo_id_from_owner_repo_layout():
    root = Path(r"D:\models")
    model = root / "unsloth" / "Qwen3-Coder" / "Q4_K_M" / "model.gguf"
    assert updates.infer_repo_id(model, [root]) == "unsloth/Qwen3-Coder"


def test_infer_repo_id_directly_under_root_is_none():
    root = Path(r"D:\models")
    model = root / "loose-model.gguf"
    assert updates.infer_repo_id(model, [root]) is None


def test_infer_repo_id_not_under_any_root_is_none():
    root = Path(r"D:\models")
    model = Path(r"E:\other") / "owner" / "repo" / "m.gguf"
    assert updates.infer_repo_id(model, [root]) is None


# ─── download_target ──────────────────────────────────────────────────────────

def test_download_target_root_and_subdir():
    # O endpoint remonta <root>/<owner>/<repo>/<subdir> — tem que dar na mesma
    # pasta em que o modelo já está.
    root = Path(r"D:\models")
    model = root / "unsloth" / "Laguna-S-2.1-GGUF" / "Laguna-S-2.1-UD-Q2_K_XL" / "m.gguf"
    assert updates.download_target(model, [root]) == (str(root), "Laguna-S-2.1-UD-Q2_K_XL")


def test_download_target_without_subdir():
    root = Path(r"D:\models")
    model = root / "unsloth" / "repo" / "m.gguf"
    assert updates.download_target(model, [root]) == (str(root), None)


def test_download_target_nested_subdirs_joined():
    root = Path(r"D:\models")
    model = root / "owner" / "repo" / "a" / "b" / "m.gguf"
    assert updates.download_target(model, [root]) == (str(root), "a/b")


def test_download_target_outside_roots_is_none():
    root = Path(r"D:\models")
    model = Path(r"E:\other") / "owner" / "repo" / "m.gguf"
    assert updates.download_target(model, [root]) == (None, None)


def test_download_target_loose_in_root_is_none():
    root = Path(r"D:\models")
    assert updates.download_target(root / "loose.gguf", [root]) == (None, None)


# ─── match_remote ─────────────────────────────────────────────────────────────

def test_match_remote_by_basename():
    remote = [
        {"path": "Q4_K_M/model.gguf", "size": 10, "oid": "aa"},
        {"path": "README.md", "size": 1, "oid": None},
    ]
    m = updates.match_remote("model.gguf", remote)
    assert m is not None and m["oid"] == "aa"


def test_match_remote_no_match_is_none():
    remote = [{"path": "other.gguf", "size": 10, "oid": "aa"}]
    assert updates.match_remote("model.gguf", remote) is None
