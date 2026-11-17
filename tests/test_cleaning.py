import numpy as np
import pytest

from possession import cleaning
from possession.data import Tracking

PITCH = (100.0, 50.0)
FPS = 25
STEP_M = 0.4
NAN = np.nan
PARAMS = cleaning.CleaningParams(ball_max_speed_mps=40.0, ball_deviation_m=1.6,
                                 player_max_speed_mps=12.0, player_deviation_m=0.3,
                                 duplicate_distance_m=0.5, max_gap_frames=8, ball_half_span=1,
                                 player_half_span=1)


def straight_track(n_frames: int, step: float = STEP_M) -> np.ndarray:
    x = 10.0 + step * np.arange(n_frames)
    return np.column_stack([x, np.full(n_frames, 20.0)])


def ball_jumps(ball: np.ndarray) -> np.ndarray:
    return cleaning.displaced_runs(ball, PARAMS.ball_deviation_m)


def eased_move(n_frames: int, start: int, length: int, distance: float) -> np.ndarray:
    progress = np.clip((np.arange(n_frames) - start) / length, 0.0, 1.0)
    x = 10.0 + distance * (1.0 - np.cos(np.pi * progress)) / 2.0
    return np.column_stack([x, np.full(n_frames, 20.0)])


def test_to_metres_scales_axes() -> None:
    normalised = np.array([[0.5, 0.5], [1.0, 0.0]])
    assert cleaning.to_metres(normalised, PITCH).tolist() == [[50.0, 25.0], [100.0, 0.0]]


def test_previous_and_next_valid_index() -> None:
    valid = np.array([False, True, False, False, True, False])
    assert cleaning.previous_valid_index(valid).tolist() == [-1, -1, 1, 1, 1, 4]
    assert cleaning.next_valid_index(valid).tolist() == [1, 4, 4, 4, 6, 6]


def test_isolated_ball_spike_is_flagged() -> None:
    ball = straight_track(10)
    ball[5] = [60.0, 20.0]
    assert np.flatnonzero(ball_jumps(ball)).tolist() == [5]


def test_fast_but_consistent_ball_is_kept() -> None:
    ball = straight_track(10, step=1.4)
    assert not ball_jumps(ball).any()


def test_spike_next_to_gaps_is_not_judged_from_far_neighbours() -> None:
    ball = straight_track(10)
    ball[[3, 4, 6, 7]] = NAN
    ball[5] = [80.0, 20.0]
    assert not ball_jumps(ball).any()


def test_real_detections_next_to_long_gaps_on_a_curved_path_are_kept() -> None:
    angle = np.linspace(0.0, 3.6, 90)
    ball = np.column_stack([30.0 + 25.0 * np.sin(angle), 20.0 + 25.0 * (1.0 - np.cos(angle))])
    ball[20:35] = NAN
    ball[50:68] = NAN
    assert np.linalg.norm(np.diff(ball, axis=0), axis=1)[:5].min() > 0.9
    assert not ball_jumps(ball).any()


@pytest.mark.parametrize("length", [3, 4, 6])
def test_plateau_of_displaced_samples_is_removed_edge_to_edge(length: int) -> None:
    ball = straight_track(30)
    ball[10:10 + length] += [0.0, 15.0]
    assert np.flatnonzero(ball_jumps(ball)).tolist() == list(range(10, 10 + length))


def test_edge_samples_are_not_flagged() -> None:
    ball = straight_track(5)
    ball[0] = [90.0, 40.0]
    assert not ball_jumps(ball).any()


def test_smooth_motion_faster_than_the_speed_bound_is_kept() -> None:
    ball = straight_track(8, step=2.0)
    assert not ball_jumps(ball).any()


def test_smooth_aerial_trajectory_is_kept() -> None:
    ball = eased_move(60, 10, 40, 60.0)
    peak = np.max(np.linalg.norm(np.diff(ball, axis=0), axis=1)) * FPS
    assert peak > PARAMS.ball_max_speed_mps
    assert not ball_jumps(ball).any()


def test_two_frame_spike_is_flagged() -> None:
    ball = straight_track(12)
    ball[5:7] += [0.0, 15.0]
    assert np.flatnonzero(ball_jumps(ball)).tolist() == [5, 6]


def test_step_to_a_new_position_keeps_both_sides() -> None:
    ball = straight_track(10)
    ball[5:] += [30.0, 0.0]
    assert not ball_jumps(ball).any()


def test_short_gap_is_interpolated_linearly() -> None:
    expected = straight_track(12)
    ball = expected.copy()
    ball[3:9] = NAN
    filled, interpolated = cleaning.interpolate_short_gaps(ball, FPS, PARAMS.max_gap_frames,
                                                           PARAMS.ball_max_speed_mps)
    assert np.flatnonzero(interpolated).tolist() == list(range(3, 9))
    np.testing.assert_allclose(filled, expected)


def test_long_gap_is_left_missing() -> None:
    ball = straight_track(20)
    ball[3:3 + PARAMS.max_gap_frames + 1] = NAN
    filled, interpolated = cleaning.interpolate_short_gaps(ball, FPS, PARAMS.max_gap_frames,
                                                           PARAMS.ball_max_speed_mps)
    assert not interpolated.any()
    assert np.isnan(filled[3:3 + PARAMS.max_gap_frames + 1]).all()


def test_gap_implying_implausible_speed_is_left_missing() -> None:
    ball = straight_track(6)
    ball[2:4] = NAN
    ball[4:] += [50.0, 0.0]
    _, interpolated = cleaning.interpolate_short_gaps(ball, FPS, PARAMS.max_gap_frames,
                                                      PARAMS.ball_max_speed_mps)
    assert not interpolated.any()


def test_leading_and_trailing_gaps_are_left_missing() -> None:
    ball = straight_track(6)
    ball[[0, 5]] = NAN
    _, interpolated = cleaning.interpolate_short_gaps(ball, FPS, PARAMS.max_gap_frames,
                                                      PARAMS.ball_max_speed_mps)
    assert not interpolated.any()


def test_player_jump_is_per_slot() -> None:
    track = straight_track(8, step=0.2)
    players = np.stack([track, track + [0.0, 10.0]], axis=1)
    players[4, 1] = [90.0, 45.0]
    jumps = cleaning.impossible_motion(players, PARAMS.player_max_speed_mps, FPS)
    assert np.argwhere(jumps).tolist() == [[4, 1]]


def player_jumps(players: np.ndarray) -> np.ndarray:
    return cleaning.jump_segments(players, PARAMS.player_max_speed_mps,
                                  PARAMS.player_deviation_m, FPS)


def smooth_track(n_frames: int, speed: float, acceleration: float) -> np.ndarray:
    seconds = np.arange(n_frames) / FPS
    x = 10.0 + speed * seconds + 0.5 * acceleration * seconds ** 2
    return np.column_stack([x, np.full(n_frames, 20.0)])


def test_single_frame_teleport_removes_exactly_that_frame() -> None:
    players = straight_track(40, step=0.2)[:, None, :].copy()
    players[10, 0] = [90.0, 45.0]
    assert np.flatnonzero(player_jumps(players)[:, 0]).tolist() == [10]


def test_single_spike_on_a_decelerating_player_removes_only_the_spike() -> None:
    track = smooth_track(50, speed=8.0, acceleration=-4.0)
    track[10, 1] += 30.0
    assert np.flatnonzero(player_jumps(track[:, None, :])[:, 0]).tolist() == [10]


def test_single_spike_on_an_accelerating_player_removes_only_the_spike() -> None:
    track = smooth_track(60, speed=0.0, acceleration=3.0)
    track[40, 1] += 30.0
    assert np.flatnonzero(player_jumps(track[:, None, :])[:, 0]).tolist() == [40]


def test_player_jump_removes_the_whole_excursion() -> None:
    n_frames, start, length = 120, 30, 40
    fake = eased_move(n_frames, start, length, 25.0)
    players = np.stack([fake, straight_track(n_frames, step=0.2)], axis=1)
    jumps = player_jumps(players)
    core = cleaning.impossible_motion(players, PARAMS.player_max_speed_mps, FPS)
    shift = 25.0 * (1.0 - np.cos(np.pi * np.arange(length + 1) / length)) / 2.0
    departed = (shift > PARAMS.player_deviation_m) & (shift[::-1] > PARAMS.player_deviation_m)
    expected = (start + np.flatnonzero(departed)).tolist()
    assert core[:, 0].sum() < len(expected)
    assert np.flatnonzero(jumps[:, 0]).tolist() == expected
    assert not jumps[:, 1].any()


def test_player_out_and_back_excursion_is_removed_and_the_rest_kept() -> None:
    n_frames = 120
    out = eased_move(n_frames, 30, 15, 20.0)
    back = eased_move(n_frames, 50, 15, -20.0) - [10.0, 0.0]
    track = out + back
    players = track[:, None, :]
    jumps = player_jumps(players)[:, 0]
    shift = 20.0 * (1.0 - np.cos(np.pi * np.arange(16) / 15)) / 2.0
    departed = (shift > PARAMS.player_deviation_m) & (shift[::-1] > PARAMS.player_deviation_m)
    expected = np.flatnonzero(departed)
    assert np.flatnonzero(jumps).tolist() == (30 + expected).tolist() + (50 + expected).tolist()


def test_genuine_fast_running_is_not_a_jump() -> None:
    track = straight_track(60, step=PARAMS.player_max_speed_mps / FPS * 0.9)
    assert not player_jumps(track[:, None, :]).any()


def test_same_team_duplicates_are_merged_into_the_earlier_slot() -> None:
    players = np.array([[[10.0, 10.0], [10.3, 10.0], [10.1, 10.0], [40.0, 10.0]]])
    slot_team = np.array([0, 0, 1, 1], dtype=np.int8)
    merged, duplicate = cleaning.merge_duplicates(players, slot_team, 0.5)
    assert duplicate.tolist() == [[False, True, False, False]]
    np.testing.assert_array_equal(merged[0, 0], players[0, 0])
    assert np.isnan(merged[0, 1]).all()
    np.testing.assert_array_equal(merged[0, 2:], players[0, 2:])


def test_merging_a_triple_keeps_one_object() -> None:
    players = np.array([[[10.0, 10.0], [10.2, 10.0], [10.1, 10.1]]])
    merged, duplicate = cleaning.merge_duplicates(players, np.zeros(3, dtype=np.int8), 0.5)
    assert duplicate.tolist() == [[False, True, True]]
    assert (~np.isnan(merged[0, :, 0])).sum() == 1


def test_duplicate_distance_is_to_the_partner_it_merges_into() -> None:
    players = np.array([[[10.0, 10.0], [10.3, 10.0], [10.1, 10.0], [40.0, 10.0]]])
    slot_team = np.array([0, 0, 0, 1], dtype=np.int8)
    distance = cleaning.duplicate_distances(players, slot_team, 0.5)
    np.testing.assert_allclose(distance[0], [np.inf, 0.3, 0.1, np.inf])


def test_chained_duplicates_are_all_flagged() -> None:
    players = np.array([[[10.0, 10.0], [10.4, 10.0], [10.8, 10.0]]])
    _, duplicate = cleaning.merge_duplicates(players, np.zeros(3, dtype=np.int8), 0.5)
    assert duplicate.tolist() == [[False, True, True]]


def test_duplicates_ignore_missing_slots_and_other_frames() -> None:
    players = np.array([[[10.0, 10.0], [NAN, NAN]], [[10.0, 10.0], [30.0, 10.0]]])
    merged, duplicate = cleaning.merge_duplicates(players, np.array([0, 0], dtype=np.int8), 0.5)
    assert not duplicate.any()
    np.testing.assert_array_equal(merged, players)


def test_clean_pipeline_on_synthetic_tracking() -> None:
    n_frames = 12
    ball = straight_track(n_frames) / PITCH
    ball[6] = [0.95, 0.9]
    ball[2] = NAN
    track = straight_track(n_frames, 0.2)
    players = np.stack([track, track + [0.2, 0.0]], axis=1) / PITCH
    tracking = Tracking(ball=ball, ball_speed=np.full(n_frames, NAN), players=players,
                        player_speed=np.zeros((n_frames, 2)),
                        slot_team=np.array([0, 0], dtype=np.int8))
    cleaned = cleaning.clean(tracking, PITCH, FPS, PARAMS)
    assert np.flatnonzero(cleaned.ball_outlier).tolist() == [6]
    assert np.flatnonzero(cleaned.ball_interpolated).tolist() == [2, 6]
    np.testing.assert_allclose(cleaned.ball, straight_track(n_frames))
    assert cleaned.player_duplicate[:, 1].all() and not cleaned.player_duplicate[:, 0].any()
    assert np.isnan(cleaned.players[:, 1]).all()
    np.testing.assert_allclose(cleaned.players[:, 0], track)
    assert cleaned.ball_observed.sum() == n_frames - 1


def test_clean_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    ball = rng.random((200, 2))
    ball[rng.random(200) < 0.3] = NAN
    players = rng.random((200, 4, 2))
    tracking = Tracking(ball=ball, ball_speed=np.full(200, NAN), players=players,
                        player_speed=np.zeros((200, 4)),
                        slot_team=np.array([0, 0, 1, 1], dtype=np.int8))
    first = cleaning.clean(tracking, PITCH, FPS, PARAMS)
    second = cleaning.clean(tracking, PITCH, FPS, PARAMS)
    for name in ("ball", "players", "ball_outlier", "ball_interpolated", "player_jump",
                 "player_duplicate"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
