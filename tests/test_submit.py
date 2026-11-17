import numpy as np
import pytest

from possession.data import graded_blocks, graded_mask, n_video_frames
from possession.submit import finalize_labels


def test_finalize_writes_x_outside_graded() -> None:
    labels = np.array(["H", "A", "D", "H", "A"])
    graded = np.array([False, True, True, True, False])
    assert finalize_labels(labels, graded).tolist() == ["X", "A", "D", "H", "X"]


def test_finalize_rejects_x_inside_graded() -> None:
    labels = np.array(["H", "X", "D"])
    with pytest.raises(ValueError, match="X"):
        finalize_labels(labels, np.array([True, True, True]))


def test_finalize_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="Q"):
        finalize_labels(np.array(["H", "Q"]), np.array([True, True]))


def test_finalize_allows_x_input_outside_graded() -> None:
    labels = np.array(["X", "H", "X"])
    assert finalize_labels(labels, np.array([False, True, False])).tolist() == ["X", "H", "X"]


def test_finalize_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        finalize_labels(np.array(["H", "A"]), np.array([True]))


def test_graded_mask_matches_published_blocks() -> None:
    mask = graded_mask(n_video_frames())
    assert graded_blocks() == [(1, 11185, 90651), (2, 110808, 190197)]
    assert int(mask.sum()) == 158857
    assert mask[11185] and mask[90651] and not mask[11184] and not mask[90652]
    assert mask[110808] and mask[190197] and not mask[110807] and not mask[190198]
