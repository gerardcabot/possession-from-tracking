import filecmp
from pathlib import Path

import numpy as np
import pytest
from conftest import OFFICIAL_BASELINE, requires_data, requires_official

from possession import baseline
from possession.data import load_tracking, n_video_frames
from possession.evaluate import score
from possession.submit import write_submission


@pytest.fixture(scope="module")
def submission(tmp_path_factory: pytest.TempPathFactory) -> Path:
    labels = baseline.predict(load_tracking(use_cache=False), 50)
    return write_submission(tmp_path_factory.mktemp("out") / "baseline_smooth50.csv", labels)


def test_smooth_majority_vote() -> None:
    codes = np.array([0, 0, 1, 0, 0, -1, -1, -1], dtype=np.int8)
    assert baseline.smooth(codes, 1).tolist() == ["H", "H", "H", "H", "H", "H", "D", "D"]


def test_raw_window_maps_codes() -> None:
    codes = np.array([0, 1, -1], dtype=np.int8)
    assert baseline.smooth(codes, 0).tolist() == ["H", "A", "D"]


@requires_data
def test_output_shape_and_labels(submission: Path) -> None:
    lines = submission.read_text().splitlines()
    assert lines[0] == "frame,label"
    assert len(lines) - 1 == n_video_frames()
    assert {line.split(",")[1] for line in lines[1:]} <= {"H", "A", "D"}


@requires_data
@requires_official
def test_matches_official_baseline(submission: Path) -> None:
    assert filecmp.cmp(submission, OFFICIAL_BASELINE, shallow=False)


@requires_data
@requires_official
def test_public_score_matches_official(submission: Path) -> None:
    assert score(submission) == score(OFFICIAL_BASELINE)
