import numpy as np
from synthetic import make_features, still_then_kicked

from possession import restart
from possession.data import AWAY, HOME, NOBODY
from possession.params import RulesParams
from possession.tagging import TagEvent

NAN = np.nan
PARAMS = RulesParams()


def near_team(n_frames: int, team: int, distance: float = 0.5) -> np.ndarray:
    nearest = np.full((n_frames, 2), 12.0)
    nearest[:, team] = distance
    return nearest


def test_kick_onset_requires_a_settled_ball() -> None:
    ball = still_then_kicked(100, 60, (30.0, 10.0))
    onsets = restart.kick_onsets(make_features(100, ball=ball), PARAMS)
    assert onsets.onset.tolist() and abs(int(onsets.onset[0]) - 60) <= 2
    np.testing.assert_allclose(onsets.anchor[0], [30.0, 10.0], atol=0.5)


def test_moving_ball_never_settles_so_no_onset() -> None:
    ball = still_then_kicked(100, 0, (10.0, 10.0))
    assert restart.kick_onsets(make_features(100, ball=ball), PARAMS).onset.size == 0


def test_restart_frame_is_when_ball_clearly_moves() -> None:
    ball = still_then_kicked(100, 60, (30.0, 10.0), speed_mps=10.0)
    features = make_features(100, ball=ball, nearest=near_team(100, AWAY))
    onsets = restart.kick_onsets(features, PARAMS)
    found = restart.find_restart(features, onsets, [], 0, 100, 99, PARAMS)
    assert found.team == AWAY
    assert 60 < found.frame <= 60 + 4


def test_restart_picks_last_kick_after_a_settled_ball() -> None:
    n_frames = 200
    ball = still_then_kicked(n_frames, 140, (30.0, 10.0))
    ball[40:60, 1] = 10.0 + np.arange(20) * 0.4
    ball[60:, 1] = ball[59, 1]
    features = make_features(n_frames, ball=ball, nearest=near_team(n_frames, HOME))
    onsets = restart.kick_onsets(features, PARAMS)
    found = restart.find_restart(features, onsets, [], 0, n_frames, n_frames - 1, PARAMS)
    assert 140 < found.frame <= 145


def test_restart_falls_back_to_tagged_team_when_no_taker_is_near() -> None:
    ball = still_then_kicked(100, 60, (30.0, 10.0))
    features = make_features(100, ball=ball)
    onsets = restart.kick_onsets(features, PARAMS)
    tags = [TagEvent(code="set pieces", start=50, end=90, team=HOME, kind="corner kick")]
    assert restart.find_restart(features, onsets, tags, 0, 100, 99, PARAMS).team == HOME


def test_restart_without_kick_uses_fallback_frame() -> None:
    features = make_features(50, ball=np.full((50, 2), NAN))
    onsets = restart.kick_onsets(features, PARAMS)
    found = restart.find_restart(features, onsets, [], 0, 50, 42, PARAMS)
    assert (found.frame, found.team) == (42, NOBODY)


def test_kickoff_is_end_of_long_formation() -> None:
    n_frames = 1000
    beyond = np.full((n_frames, 2), 5)
    beyond[100:150] = 0
    beyond[300:700] = 1
    centre = np.full((n_frames, 2), 9.5)
    centre[:, AWAY] = 0.3
    features = make_features(n_frames, ball=np.full((n_frames, 2), NAN), beyond_halfway=beyond,
                             centre_distance=centre)
    onsets = restart.kick_onsets(features, PARAMS)
    found = restart.find_kickoff(features, onsets, 0, n_frames, PARAMS)
    assert found is not None
    assert (found.frame, found.team) == (700, AWAY)


def test_kickoff_refined_by_ball_leaving_centre_spot() -> None:
    n_frames = 1000
    beyond = np.full((n_frames, 2), 5)
    beyond[300:700] = 0
    ball = still_then_kicked(n_frames, 680, (50.0, 30.0))
    features = make_features(n_frames, ball=ball, beyond_halfway=beyond)
    onsets = restart.kick_onsets(features, PARAMS)
    found = restart.find_kickoff(features, onsets, 0, n_frames, PARAMS)
    assert found is not None and 680 < found.frame <= 684


def test_no_kickoff_without_formation() -> None:
    features = make_features(500)
    onsets = restart.kick_onsets(features, PARAMS)
    assert restart.find_kickoff(features, onsets, 0, 500, PARAMS) is None


def formation_features(n_frames: int, formations: list[tuple[int, int]]) -> object:
    beyond = np.full((n_frames, 2), 5)
    for start, end in formations:
        beyond[start:end] = 0
    return make_features(n_frames, ball=np.full((n_frames, 2), NAN), beyond_halfway=beyond)


def test_kickoff_ignores_formations_after_the_start_window() -> None:
    n_frames = 20000
    window = int(PARAMS.kickoff_window_s * 25)
    features = formation_features(n_frames, [(window + 500, window + 1000)])
    onsets = restart.kick_onsets(features, PARAMS)
    assert restart.find_kickoff(features, onsets, 0, n_frames, PARAMS) is None


def test_kickoff_window_starts_at_the_half_block_start() -> None:
    n_frames = 30000
    low = 12000
    window = int(PARAMS.kickoff_window_s * 25)
    features = formation_features(n_frames, [(low - 600, low - 100), (low + 400, low + 900),
                                             (low + window + 500, low + window + 1000)])
    onsets = restart.kick_onsets(features, PARAMS)
    found = restart.find_kickoff(features, onsets, low, n_frames, PARAMS)
    assert found is not None and found.frame == low + 900


def test_kickoff_prefers_the_earliest_formation_in_the_window() -> None:
    features = formation_features(5000, [(200, 600), (1500, 2000)])
    onsets = restart.kick_onsets(features, PARAMS)
    found = restart.find_kickoff(features, onsets, 0, 5000, PARAMS)
    assert found is not None and found.frame == 600


def test_first_half_kickoff_at_the_very_start_of_the_block() -> None:
    features = formation_features(3000, [(0, 400)])
    onsets = restart.kick_onsets(features, PARAMS)
    found = restart.find_kickoff(features, onsets, 0, 3000, PARAMS)
    assert found is not None and found.frame == 400


def test_tagged_team_window_is_half_open() -> None:
    tags = [TagEvent(code="set pieces", start=10, end=20, team=HOME, kind="throw-in")]
    assert [restart.tagged_team(tags, frame) for frame in (9, 10, 19, 20)] == [NOBODY, HOME,
                                                                               HOME, NOBODY]


def test_first_restart_is_the_earliest_clear_kick() -> None:
    n_frames = 300
    ball = still_then_kicked(n_frames, 200, (30.0, 10.0))
    ball[60:80] = ball[60:80] + np.column_stack([np.arange(20) * 0.5, np.zeros(20)])
    ball[80:200, 0] = ball[79, 0]
    features = make_features(n_frames, ball=ball, nearest=near_team(n_frames, HOME))
    onsets = restart.kick_onsets(features, PARAMS)
    found = restart.first_restart(features, onsets, [], 0, n_frames, PARAMS)
    assert found is not None and found.frame < 100
    assert restart.find_restart(features, onsets, [], 0, n_frames, n_frames - 1,
                                PARAMS).frame >= 200
