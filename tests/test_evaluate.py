import sys
from pathlib import Path

import pytest
from conftest import OFFICIAL_BASELINE, requires_official

from possession.evaluate import score
from possession.paths import private_gt_path, public_gt_path, resolve_gt
from possession.vendor import VENDOR_DIR


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "evaluation" / "gt").mkdir(parents=True)
    monkeypatch.setenv("MTAH_REPO_DIR", str(tmp_path))
    return tmp_path / "evaluation" / "gt"


def test_public_alias_resolves_under_repo_dir(repo: Path) -> None:
    public = repo / "X_possession_gt_public.csv"
    public.write_text("frame,label\n")
    (repo / "X_possession_gt_public_meta.json").write_text("{}")
    assert resolve_gt("public") == public == public_gt_path()


def test_private_alias_matches_only_full_match_file(repo: Path) -> None:
    (repo / "X_possession_gt_public.csv").write_text("frame,label\n")
    private = repo / "X_possession_gt.csv"
    private.write_text("frame,label\n")
    assert resolve_gt("private") == private == private_gt_path()


def test_private_alias_missing_raises(repo: Path) -> None:
    (repo / "X_possession_gt_public.csv").write_text("frame,label\n")
    with pytest.raises(FileNotFoundError, match="possession_gt.csv"):
        resolve_gt("private")


def test_other_values_are_paths(repo: Path) -> None:
    assert resolve_gt("some/gt.csv") == Path("some/gt.csv")


def test_vendor_dir_not_added_to_sys_path() -> None:
    assert str(VENDOR_DIR) not in sys.path


@requires_official
def test_public_alias_scores_like_explicit_path() -> None:
    assert score(OFFICIAL_BASELINE, "public") == score(OFFICIAL_BASELINE, str(public_gt_path()))
