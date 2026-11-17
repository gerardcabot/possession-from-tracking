import numpy as np
import pytest
from scipy.special import expit

from possession.learned import framefeatures as ff
from possession.learned.adapt import (
    AdaptationReport,
    adapted_dead_logit,
    block_frames,
    bounded_logit,
    cross_folds,
    held_out_log_ratio,
    observation_columns,
    share_tolerance,
)
from possession.learned.classifier import FrameClassifier, TrainingRows, fit_classifier

DEAD = 2


def test_player_features_exclude_ball_and_object_counts() -> None:
    names = ["ball_present", "ball_speed", "nearest_own", "tracked_own", "dwell_back",
             "gap_length", "mean_speed", "team_speed_own", "team_speed_other", "shape",
             "spread_x_own", "centroid_vx_other", "mean_speed_past_200", "shape_future_600",
             "ball_present_past_200", "near_diff_future_600"]
    chosen = [names[index] for index in ff.dead_columns(names, "players")]
    assert chosen == ["mean_speed", "team_speed_own", "team_speed_other", "shape",
                      "spread_x_own", "centroid_vx_other", "mean_speed_past_200",
                      "shape_future_600"]
    assert ff.dead_columns(names, "all") == tuple(range(len(names)))
    with pytest.raises(ValueError, match="feature set"):
        ff.dead_columns(names, "ball")
    with pytest.raises(ValueError, match="no player features"):
        ff.dead_columns(["ball_present"], "players")


def two_view_data(rng: np.random.Generator, n_frames: int, ball_signal: float,
                  dead_rate: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    labels = np.where(rng.random(n_frames) < dead_rate, DEAD, rng.integers(0, 2, n_frames))
    dead = labels == DEAD
    players = rng.normal(size=n_frames) - 1.0 * dead
    ball = rng.normal(size=n_frames) + ball_signal * dead
    team = rng.normal(size=n_frames) + 2.0 * (labels == 1)
    return np.column_stack([players, ball, team]).astype(np.float32), labels


def classifier_on_players(rng: np.random.Generator) -> FrameClassifier:
    matrix, labels = two_view_data(rng, 4000, ball_signal=-3.0)
    rows = TrainingRows(matrix=matrix, codes=labels, timed=np.ones(len(labels), dtype=bool),
                        match=np.zeros(len(labels), dtype=np.int64))
    return fit_classifier(rows, ["mean_speed", "ball_speed", "team"], 0, (0,))


def test_dead_model_reads_only_its_columns() -> None:
    classifier = classifier_on_players(np.random.default_rng(0))
    matrix, _ = two_view_data(np.random.default_rng(1), 500, ball_signal=-3.0)
    shuffled = matrix.copy()
    shuffled[:, 1:] = np.random.default_rng(2).normal(size=(500, 2))
    np.testing.assert_array_equal(classifier.dead_logit(matrix), classifier.dead_logit(shuffled))
    assert observation_columns(classifier) == [1, 2]
    assert classifier.dead_feature_names == ("mean_speed",)


def test_adaptation_learns_the_target_observation_process() -> None:
    classifier = classifier_on_players(np.random.default_rng(0))
    matrix, labels = two_view_data(np.random.default_rng(3), 6000, ball_signal=3.0)
    blocks = [(1, 0, 2999), (2, 3000, 5999)]
    plain = classifier.dead_logit(matrix)
    adapted, report = adapted_dead_logit(classifier, matrix, blocks, 0, 20)
    dead = labels == DEAD

    def accuracy(logit: np.ndarray) -> float:
        return float(np.mean((expit(logit) > 0.5) == dead))

    assert accuracy(adapted) > accuracy(plain) + 0.05
    assert report.converged and 1 <= report.iterations <= 20
    assert len(report.shares) == report.iterations + 1


def test_adaptation_is_a_no_op_without_iterations_or_observations() -> None:
    classifier = classifier_on_players(np.random.default_rng(0))
    matrix, _ = two_view_data(np.random.default_rng(3), 600, ball_signal=3.0)
    blocks = [(1, 0, 299), (2, 300, 599)]
    unchanged, report = adapted_dead_logit(classifier, matrix, blocks, 0, 0)
    np.testing.assert_array_equal(unchanged, classifier.dead_logit(matrix))
    assert report.shares == () and report.iterations == 0


def test_adaptation_leaves_frames_outside_blocks_unchanged() -> None:
    classifier = classifier_on_players(np.random.default_rng(0))
    matrix, _ = two_view_data(np.random.default_rng(4), 3000, ball_signal=3.0)
    blocks = [(1, 100, 1399), (2, 1500, 2899)]
    adapted, _ = adapted_dead_logit(classifier, matrix, blocks, 0, 3)
    plain = classifier.dead_logit(matrix)
    outside = np.ones(3000, dtype=bool)
    outside[100:1400] = outside[1500:2900] = False
    np.testing.assert_array_equal(adapted[outside], plain[outside])
    assert not np.array_equal(adapted[~outside], plain[~outside])


def test_block_frames_and_tolerance() -> None:
    folds = block_frames(10, [(1, 0, 3), (2, 6, 20)])
    assert [fold.tolist() for fold in folds] == [[0, 1, 2, 3], [6, 7, 8, 9]]
    assert share_tolerance(np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.25)
    assert share_tolerance(np.array([0.0, 1.0])) == 0.0



def test_single_block_is_cross_fitted_between_two_halves() -> None:
    folds = cross_folds(block_frames(10, [(1, 0, 9)]))
    assert [fold.tolist() for fold in folds] == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    two = block_frames(10, [(1, 0, 3), (2, 4, 9)])
    assert cross_folds(two) == two
    assert cross_folds(block_frames(1, [(1, 0, 0)])) == [np.array([0])]
    classifier = classifier_on_players(np.random.default_rng(0))
    matrix, _ = two_view_data(np.random.default_rng(3), 400, ball_signal=3.0)
    unchanged, report = adapted_dead_logit(classifier, matrix[:1], [(1, 0, 0)], 0, 5)
    np.testing.assert_array_equal(unchanged, classifier.dead_logit(matrix[:1]))
    assert report.shares == ()
    adapted, report = adapted_dead_logit(classifier, matrix, [(1, 0, 399)], 0, 2)
    assert report.iterations >= 1
    assert not np.array_equal(adapted, classifier.dead_logit(matrix))


def test_bounded_logit_is_finite_at_the_extremes() -> None:
    assert bounded_logit(0.0, 99) == pytest.approx(-np.log(99))
    assert bounded_logit(1.0, 99) == pytest.approx(np.log(99))
    assert bounded_logit(0.25, 99) == pytest.approx(np.log(1 / 3))


def test_degenerate_inputs_give_finite_outputs() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(200, 2)).astype(np.float32)
    target = rng.normal(size=(50, 2)).astype(np.float32)
    for dead in (np.zeros(200), np.ones(200)):
        assert np.isfinite(held_out_log_ratio(train, dead, target, 0)).all()
    classifier = classifier_on_players(np.random.default_rng(0))
    constant = np.zeros((300, 3), dtype=np.float32)
    adapted, report = adapted_dead_logit(classifier, constant, [(1, 0, 299)], 0, 3)
    assert np.isfinite(adapted).all()
    assert all(np.isfinite(report.shares))


def test_unconverged_adaptation_is_reported() -> None:
    classifier = classifier_on_players(np.random.default_rng(0))
    matrix, _ = two_view_data(np.random.default_rng(3), 2000, ball_signal=3.0, dead_rate=0.8)
    blocks = [(1, 0, 999), (2, 1000, 1999)]
    _, report = adapted_dead_logit(classifier, matrix, blocks, 0, 1)
    assert report.iterations == 1 and not report.converged
    values = report.to_dict()
    assert values["converged"] is False and values["iterations"] == 1
    assert values["final_share"] == report.shares[-1]
    assert "converged False" in report.describe()
    assert AdaptationReport(shares=(), converged=False).describe() == "adaptation skipped"


def test_prior_correction_moves_the_emitted_share_towards_the_target_prior() -> None:
    classifier = classifier_on_players(np.random.default_rng(0))
    matrix, labels = two_view_data(np.random.default_rng(3), 6000, ball_signal=3.0, dead_rate=0.85)
    blocks = [(1, 0, 2999), (2, 3000, 5999)]
    truth = float(np.mean(labels == DEAD))
    corrected, _ = adapted_dead_logit(classifier, matrix, blocks, 0, 20)
    plain, _ = adapted_dead_logit(classifier, matrix, blocks, 0, 20, correct_prior=False)
    assert abs(expit(corrected).mean() - truth) < abs(expit(plain).mean() - truth)
