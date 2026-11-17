import numpy as np
from synthetic import make_features

from possession import control
from possession.data import AWAY, HOME, NOBODY
from possession.params import RulesParams

NAN = np.nan
PARAMS = RulesParams()


def test_candidate_requires_closest_team_within_radius() -> None:
    nearest = np.array([[0.5, 3.0], [3.0, 0.8], [1.0, 1.0], [2.0, 4.0]])
    features = make_features(4, nearest=nearest)
    assert control.control_candidates(features, PARAMS).tolist() == [HOME, AWAY, NOBODY, NOBODY]


def test_candidate_ignores_fast_ball_and_missing_ball() -> None:
    ball = np.column_stack([np.arange(6) * 1.0, np.zeros(6)])
    ball[5] = NAN
    nearest = np.tile([0.5, 5.0], (6, 1))
    candidates = control.control_candidates(make_features(6, ball=ball, nearest=nearest), PARAMS)
    assert (candidates[:5] == NOBODY).all() and candidates[5] == NOBODY


def test_control_events_need_minimum_dwell() -> None:
    team = np.array([NOBODY, HOME, HOME, NOBODY, AWAY, AWAY, AWAY, AWAY, HOME], dtype=np.int8)
    events = control.control_events(team, 3)
    assert events == [control.ControlEvent(team=AWAY, start=4, end=8)]


def test_detect_control_uses_seconds() -> None:
    nearest = np.tile([5.0, 0.5], (40, 1))
    nearest[:10] = [0.5, 5.0]
    events = control.detect_control(make_features(40, nearest=nearest), PARAMS)
    assert [(event.team, event.start, event.end) for event in events] == [(AWAY, 10, 40)]
