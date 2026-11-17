import numpy as np

from possession import features
from possession.intervals import (
    drop_short_runs,
    fill_short_gaps,
    run_has_length,
    true_runs,
)

NAN = np.nan


def test_true_runs_returns_half_open_bounds() -> None:
    starts, ends = true_runs(np.array([True, True, False, True, False, False, True]))
    assert starts.tolist() == [0, 3, 6]
    assert ends.tolist() == [2, 4, 7]


def test_drop_short_runs_and_fill_short_gaps() -> None:
    mask = np.array([True, False, True, True, True, False, False, True, True, True])
    assert drop_short_runs(mask, 3).tolist() == [False, False, True, True, True, False, False,
                                                 True, True, True]
    assert fill_short_gaps(mask, 2).tolist() == [True, True, True, True, True, False, False,
                                                 True, True, True]


def test_fill_short_gaps_keeps_edges_open() -> None:
    mask = np.array([False, True, True, False])
    assert fill_short_gaps(mask, 5).tolist() == [False, True, True, False]


def test_run_has_length_looks_strictly_backwards() -> None:
    mask = np.array([True, True, False, True, True, True])
    assert run_has_length(mask, 2).tolist() == [False, False, True, False, False, True]


def test_moving_nanmean_ignores_missing() -> None:
    values = np.array([1.0, NAN, 3.0, 5.0, NAN])
    np.testing.assert_allclose(features.moving_nanmean(values, 3), [1.0, 2.0, 4.0, 4.0, 5.0])


def test_moving_nanmean_all_missing_window_is_nan() -> None:
    assert np.isnan(features.moving_nanmean(np.array([NAN, NAN, 1.0]), 1)[:2]).all()


def test_central_velocity_uses_one_sided_difference_at_gap_edges() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0, NAN, NAN])
    positions = np.column_stack([x, np.zeros_like(x)])
    velocity = features.central_velocity(positions, fps=1.0, half_span=2)
    np.testing.assert_allclose(velocity[:4, 0], [1.0, 1.0, 1.0, 1.0])
    assert np.isnan(velocity[4:, 0]).all()


def test_team_nearest_returns_first_and_second_per_team() -> None:
    distance = np.array([[3.0, 1.0, np.inf, 2.0, 7.0]])
    slot_team = np.array([0, 0, 0, 1, 1], dtype=np.int8)
    first, second = features.team_nearest(distance, slot_team)
    assert first.tolist() == [[1.0, 2.0]]
    assert second.tolist() == [[3.0, 7.0]]


def test_player_speeds_are_per_slot_central_differences() -> None:
    track = np.column_stack([np.arange(6.0), np.zeros(6)])
    players = np.stack([track, 2 * track], axis=1)
    speed = features.player_speeds(players, fps=1.0, half_span=1)
    np.testing.assert_allclose(speed, np.tile([1.0, 2.0], (6, 1)))


def goalkeeper_layout(home_goal_x: float, n_frames: int = 10) -> np.ndarray:
    home = [[home_goal_x, 30.0], [50.0, 30.0]]
    away = [[100.0 - home_goal_x, 30.0], [55.0, 30.0]]
    return np.tile(np.array(home + away), (n_frames, 1, 1))


def test_attack_sign_is_derived_from_deepest_players_per_half() -> None:
    first = goalkeeper_layout(2.0)
    second = goalkeeper_layout(98.0)
    players = np.concatenate([first, second])
    slot_team = np.array([0, 0, 1, 1], dtype=np.int8)
    signs = features.attack_signs(players, slot_team, [(1, 0, 9), (2, 10, 19)])
    assert signs[:10].tolist() == [1] * 10
    assert signs[10:].tolist() == [-1] * 10


def test_halfway_counts_respect_attack_direction() -> None:
    players = np.array([[[40.0, 0.0], [60.0, 0.0], [45.0, 0.0], [70.0, 0.0]]])
    slot_team = np.array([0, 0, 1, 1], dtype=np.int8)
    tracked = features.tracked_counts(players, slot_team)
    beyond = features.beyond_halfway_counts(players, slot_team, np.array([1]), 100.0, 1.0)
    assert tracked.tolist() == [[2, 2]]
    assert beyond.tolist() == [[1, 1]]


def test_defensive_line_is_second_deepest_player() -> None:
    players = np.array([[[2.0, 0.0], [20.0, 0.0], [30.0, 0.0], [98.0, 0.0], [70.0, 0.0],
                         [60.0, 0.0]]])
    slot_team = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    lines = features.defensive_lines(players, slot_team, np.array([1]))
    assert lines.tolist() == [[20.0, 70.0]]
