from dataclasses import replace

import numpy as np
import pytest
from external_synthetic import FPS, PITCH, make_match

from possession.data import Tracking
from possession.external import bounds
from possession.external.match import ExternalMatch

NAN = np.nan


def line(n_frames: int, step: float) -> np.ndarray:
    return np.column_stack([10.0 + step * np.arange(n_frames), np.full(n_frames, 20.0)])


def track_of(ball: np.ndarray, alive: np.ndarray | None = None) -> bounds.AliveTrack:
    alive = np.ones(len(ball), dtype=bool) if alive is None else alive
    return bounds.AliveTrack(ball=ball, players=ball[:, None, :], slot_team=np.zeros(1),
                             alive=alive, blocks=[(1, 0, len(ball) - 1)], fps=FPS)


def test_consistent_speed_ignores_an_out_and_back_spike() -> None:
    ball = line(10, 0.4)
    ball[5] = [80.0, 20.0]
    speed = bounds.consistent_speed(ball, FPS)
    assert speed.max() == pytest.approx(0.4 * FPS)


def test_consistent_speed_keeps_sustained_fast_motion() -> None:
    speed = bounds.consistent_speed(line(10, 1.6), FPS)
    np.testing.assert_allclose(speed, 1.6 * FPS)


def test_max_alive_speed_ignores_dead_frames() -> None:
    ball = line(20, 0.4)
    ball[12:] += [30.0, 0.0]
    alive = np.ones(20, dtype=bool)
    assert bounds.max_alive_speed(ball, track_of(ball, alive)) > 0.4 * FPS
    alive[10:14] = False
    assert bounds.max_alive_speed(ball, track_of(ball, alive)) == pytest.approx(0.4 * FPS)


def test_max_alive_deviation_measures_a_spike_off_both_extrapolations() -> None:
    ball = line(20, 0.4)
    ball[10] += [0.0, 3.0]
    assert bounds.max_alive_deviation(ball, track_of(ball)) == pytest.approx(3.0)
    alive = np.ones(20, dtype=bool)
    alive[9] = False
    assert bounds.max_alive_deviation(ball, track_of(ball, alive)) < 3.0


def test_max_alive_deviation_does_not_extrapolate_across_ball_gaps() -> None:
    angle = np.linspace(0.0, 3.6, 60)
    ball = np.column_stack([30.0 + 25.0 * np.sin(angle), 20.0 + 25.0 * (1.0 - np.cos(angle))])
    full = bounds.max_alive_deviation(ball, track_of(ball))
    gapped = ball.copy()
    gapped[20:35] = NAN
    assert bounds.max_alive_deviation(gapped, track_of(gapped)) <= full + 1e-9
    spiked = line(30, 0.4)
    spiked[15] += [0.0, 3.0]
    assert bounds.max_alive_deviation(spiked, track_of(spiked)) == pytest.approx(3.0)
    spiked[13] = NAN
    assert bounds.max_alive_deviation(spiked, track_of(spiked)) == pytest.approx(0.0, abs=1e-9)


def test_max_alive_deviation_ignores_smooth_fast_motion() -> None:
    ball = line(20, 2.0)
    assert bounds.max_alive_deviation(ball, track_of(ball)) == pytest.approx(0.0)


def test_robust_extreme_is_the_median_of_finite_match_extremes() -> None:
    assert bounds.robust_extreme([9.0, 10.0, 11.0, 50.0, NAN]) == pytest.approx(10.5)
    with pytest.raises(ValueError, match="finite"):
        bounds.robust_extreme([NAN])


def test_short_gap_limit_splits_dropouts_from_occlusions() -> None:
    gaps = np.array([1, 1, 2, 2, 3, 5, 8, 90, 120, 200, 260, 400, 800])
    limit = bounds.short_gap_limit(gaps)
    assert 8 <= limit < 90
    assert bounds.short_gap_limit(np.array([5, 5])) == 0


def test_short_gap_limit_survives_a_long_component_claiming_small_lengths() -> None:
    gaps = np.array([4] * 50 + [5] * 40 + [6] * 30 + list(range(1, 400, 3)))
    assert bounds.short_gap_limit(gaps) >= 6


def test_velocity_half_span_widens_only_when_noise_dominates() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(2000) / FPS
    ball = np.column_stack([50 + 20 * np.sin(t), 30 + 10 * np.cos(t / 2)])
    tracks = [track_of(ball)]
    assert bounds.velocity_half_span(tracks, "ball", 0.0, FPS) == 1
    assert bounds.velocity_half_span(tracks, "ball", 0.5, FPS) > 1
    noisy = ball + rng.normal(scale=0.5, size=ball.shape)
    assert bounds.velocity_half_span([track_of(noisy)], "ball", 0.0, FPS) == 1


def test_velocity_half_span_terminates_on_constant_velocity() -> None:
    ball = line(200, 0.3)
    half_span = bounds.velocity_half_span([track_of(ball)], "ball", 0.5, FPS)
    assert 1 <= half_span <= 100


def labelled(match: ExternalMatch) -> ExternalMatch:
    labels = match.labels.copy()
    labels[:100] = "D"
    return replace(match, labels=labels)


def our_tracking(match: ExternalMatch) -> Tracking:
    ball = match.ball.copy()
    for start, length in ((100, 2), (300, 3), (700, 150), (1500, 300), (2500, 200)):
        ball[start:start + length] = NAN
    return Tracking(ball=ball, ball_speed=np.zeros(match.n_frames), players=match.players,
                    player_speed=np.zeros(match.players.shape[:2]), slot_team=match.slot_team)


def test_estimate_params_is_deterministic_and_data_derived() -> None:
    references = [labelled(make_match(n_per_period=1500, seed=seed)) for seed in range(3)]
    ours = make_match(n_per_period=1500, seed=9)
    tracking = our_tracking(ours)
    first = bounds.estimate_params(references, tracking, PITCH, FPS, ours.blocks)
    second = bounds.estimate_params(references, tracking, PITCH, FPS, ours.blocks)
    assert first == second
    extremes = [bounds.match_extremes(bounds.alive_track(match)) for match in references]
    assert first.ball_max_speed_mps == pytest.approx(
        np.median([item.ball_speed_mps for item in extremes]))
    assert first.ball_deviation_m == pytest.approx(
        np.median([item.ball_deviation_m for item in extremes]))
    assert first.player_deviation_m == pytest.approx(
        np.median([item.player_deviation_m for item in extremes]))
    assert first.duplicate_distance_m == pytest.approx(
        np.median([item.teammate_distance_m for item in extremes]))
    assert 3 <= first.max_gap_frames < 150
    assert first.ball_half_span >= 1 and first.player_half_span >= 1
