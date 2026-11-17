import numpy as np
from synthetic import FPS, PITCH, make_features, still_then_kicked

from possession import rules
from possession.cleaning import CleanTracking
from possession.data import AWAY, HOME
from possession.features import build_features, threshold_free_features
from possession.params import (
    LEGACY_CLEANING,
    LEGACY_HALFWAY_MARGIN_M,
    LEGACY_SHAPE_WINDOW_S,
    RulesParams,
)

N_FRAMES = 3000
KICKOFF, TAKEOVER, OUT_KICK, PICK_UP, THROW_IN = 400, 1200, 1950, 2380, 2400
HOME_BASE = [(5.0, 30.0), (30.0, 10.0), (35.0, 20.0), (40.0, 25.0), (49.5, 30.0), (40.0, 40.0),
             (35.0, 45.0), (30.0, 50.0)]
AWAY_BASE = [(100.0 - x, 60.0 - y) for x, y in HOME_BASE[:4]] + [(62.0, 35.0)] + \
    [(100.0 - x, 60.0 - y) for x, y in HOME_BASE[5:]]
HOME_CARRIER, AWAY_CARRIER, THROWER = 4, 12, 13


def linear(start: np.ndarray, velocity: np.ndarray, n_frames: int) -> np.ndarray:
    return start + np.outer(np.arange(n_frames), velocity) / FPS


def ball_path() -> np.ndarray:
    ball = np.zeros((N_FRAMES, 2))
    ball[:KICKOFF] = [50.0, 30.0]
    ball[KICKOFF:425] = linear(np.array([50.0, 30.0]), np.array([10.0, 0.0]), 25)
    ball[425:TAKEOVER] = linear(np.array([60.0, 30.0]), np.array([0.8, 0.0]), TAKEOVER - 425)
    ball[TAKEOVER:OUT_KICK] = linear(ball[TAKEOVER - 1], np.array([-0.8, 0.0]),
                                     OUT_KICK - TAKEOVER)
    ball[OUT_KICK:PICK_UP] = linear(ball[OUT_KICK - 1], np.array([0.0, -15.0]),
                                    PICK_UP - OUT_KICK)
    ball[OUT_KICK:PICK_UP, 1] = np.maximum(ball[OUT_KICK:PICK_UP, 1], -2.0)
    ball[PICK_UP:THROW_IN] = [ball[OUT_KICK - 1, 0], -0.5]
    ball[THROW_IN:] = linear(ball[PICK_UP], np.array([0.0, 8.0]), N_FRAMES - THROW_IN)
    ball[THROW_IN:, 1] = np.minimum(ball[THROW_IN:, 1], 15.0)
    return ball


def player_paths(ball: np.ndarray) -> np.ndarray:
    players = np.tile(np.array(HOME_BASE + AWAY_BASE), (N_FRAMES, 1, 1))
    players[KICKOFF:, 1:8, 0] += 20.0
    players[KICKOFF:, HOME_CARRIER] = HOME_BASE[HOME_CARRIER]
    players[425:TAKEOVER, HOME_CARRIER] = ball[425:TAKEOVER] - [0.5, 0.0]
    players[TAKEOVER:OUT_KICK, AWAY_CARRIER] = ball[TAKEOVER:OUT_KICK] + [0.5, 0.0]
    players[OUT_KICK:, AWAY_CARRIER] = players[OUT_KICK - 1, AWAY_CARRIER]
    players[PICK_UP:, THROWER] = ball[PICK_UP:] + [0.5, 0.0]
    return players


def player_speed() -> np.ndarray:
    speed = np.full((N_FRAMES, 16), 4.0)
    speed[:KICKOFF] = 0.0
    speed[OUT_KICK + 50:THROW_IN] = 0.5
    return speed


def scenario_clean() -> CleanTracking:
    ball = ball_path()
    players = player_paths(ball)
    flags = np.zeros(N_FRAMES, dtype=bool)
    return CleanTracking(ball=ball, ball_observed=~flags, ball_outlier=flags,
                          ball_interpolated=flags, players=players,
                          player_observed=np.ones(players.shape[:2], dtype=bool),
                          player_jump=np.zeros(players.shape[:2], dtype=bool),
                          player_duplicate=np.zeros(players.shape[:2], dtype=bool),
                          slot_team=np.array([HOME] * 8 + [AWAY] * 8, dtype=np.int8))


def legacy_build(blocks: list[tuple[int, int, int]]) -> object:
    return build_features(scenario_clean(), player_speed(), PITCH, FPS, blocks,
                          LEGACY_CLEANING.ball_half_span, LEGACY_SHAPE_WINDOW_S,
                          LEGACY_HALFWAY_MARGIN_M)


def scenario_labels() -> np.ndarray:
    blocks = [(1, 0, N_FRAMES - 1)]
    return rules.label_match(legacy_build(blocks), [], blocks, RulesParams())


def label_runs(labels: np.ndarray) -> list[tuple[str, int, int]]:
    starts = np.flatnonzero(np.concatenate(([True], labels[1:] != labels[:-1])))
    ends = np.append(starts[1:], len(labels))
    return [(str(labels[start]), int(start), int(end)) for start, end in zip(starts, ends)]


def test_scenario_produces_expected_possessions() -> None:
    runs = label_runs(scenario_labels())
    assert [label for label, _, _ in runs] == ["D", "H", "A", "D", "A"]
    expected = [KICKOFF, TAKEOVER, OUT_KICK + 50, THROW_IN]
    for (_, start, _), frame in zip(runs[1:], expected):
        assert abs(start - frame) <= 5, (start, frame)


def test_label_match_is_deterministic() -> None:
    np.testing.assert_array_equal(scenario_labels(), scenario_labels())


def test_threshold_free_features_carry_no_legacy_field() -> None:
    blocks = [(1, 0, N_FRAMES - 1)]
    core = threshold_free_features(scenario_clean(), player_speed(), PITCH, FPS, blocks, 1, 1)
    legacy = legacy_build(blocks)
    assert core.beyond_halfway is None and legacy.beyond_halfway is not None
    for name in ("second_nearest", "centroid", "defensive_line", "team_speed"):
        assert getattr(core, name) is not None
    np.testing.assert_array_equal(core.nearest, legacy.nearest)


def kickoff_less_features(n_frames: int, kick: int | None) -> object:
    ball = np.full((n_frames, 2), np.nan) if kick is None else \
        still_then_kicked(n_frames, kick, (30.0, 10.0))
    return make_features(n_frames, ball=ball)


def test_missing_kickoff_forces_dead_until_first_detected_restart(capsys) -> None:
    features = kickoff_less_features(600, 300)
    onsets = rules.kick_onsets(features, RulesParams())
    halves = rules.half_starts(features, onsets, [], [(1, 0, 599)], RulesParams())
    assert halves[0].kickoff is not None and 300 < halves[0].kickoff.frame < 310
    mask = rules.half_start_mask(np.zeros(600, dtype=bool), halves)
    assert mask[:halves[0].kickoff.frame].all() and not mask[halves[0].kickoff.frame:].any()
    assert "no kickoff formation found" in capsys.readouterr().err


def test_missing_kickoff_and_restart_forces_first_frame_and_warns(capsys) -> None:
    features = kickoff_less_features(200, None)
    onsets = rules.kick_onsets(features, RulesParams())
    halves = rules.half_starts(features, onsets, [], [(2, 50, 199)], RulesParams())
    assert halves[0].kickoff is None
    mask = rules.half_start_mask(np.zeros(200, dtype=bool), halves)
    assert np.flatnonzero(mask).tolist() == [50]
    assert "half 2" in capsys.readouterr().err


def test_kickoff_at_the_block_start_is_not_dropped() -> None:
    restart = rules.Restart(frame=0, team=HOME, onset=0)
    halves = [rules.HalfStart(first=0, kickoff=restart)]
    mask = rules.half_start_mask(np.zeros(50, dtype=bool), halves)
    assert mask[0] and not mask[1:].any()
    assert rules.kickoff_in(halves, 0, 1) is restart


def test_detected_kickoff_is_found_in_a_half_with_formation(capsys) -> None:
    n_frames = 1000
    beyond = np.full((n_frames, 2), 5)
    beyond[100:400] = 0
    features = make_features(n_frames, ball=np.full((n_frames, 2), np.nan),
                             beyond_halfway=beyond)
    onsets = rules.kick_onsets(features, RulesParams())
    halves = rules.half_starts(features, onsets, [], [(1, 0, 999)], RulesParams())
    assert halves[0].kickoff is not None and halves[0].kickoff.frame == 400
    assert capsys.readouterr().err == ""
