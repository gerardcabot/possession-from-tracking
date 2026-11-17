import numpy as np
from synthetic import make_features

from possession import statemachine as sm
from possession.control import ControlEvent
from possession.data import AWAY, HOME, NOBODY

DEAD = sm.DEAD


def context(n_frames: int, events: list[ControlEvent], missing: np.ndarray | None = None,
            shape: np.ndarray | None = None, gap_frames: int = 10,
            nearest: np.ndarray | None = None) -> sm.LiveContext:
    return sm.LiveContext(
        events=events,
        event_starts=[event.start for event in events],
        missing=np.zeros(n_frames, dtype=bool) if missing is None else missing,
        shape=np.zeros(n_frames) if shape is None else shape,
        gap_frames=gap_frames,
        nearest=np.full((n_frames, 2), np.inf) if nearest is None else nearest,
        touch_frames=25,
        touch_radius_m=1.5,
    )


def runs(codes: np.ndarray) -> list[tuple[int, int, int]]:
    starts = np.flatnonzero(np.concatenate(([True], codes[1:] != codes[:-1])))
    ends = np.append(starts[1:], len(codes))
    return [(int(codes[start]), int(start), int(end)) for start, end in zip(starts, ends)]


def test_frames_before_first_restart_are_dead() -> None:
    stoppages = [sm.Stoppage(start=0, restart=30, team=HOME, by_exit=False)]
    codes = sm.build_codes(100, stoppages, context(100, []))
    assert runs(codes) == [(DEAD, 0, 30), (HOME, 30, 100)]


def test_opponent_control_switches_at_first_touch() -> None:
    events = [ControlEvent(team=HOME, start=35, end=45), ControlEvent(team=AWAY, start=60, end=70)]
    stoppages = [sm.Stoppage(start=0, restart=30, team=HOME, by_exit=False)]
    codes = sm.build_codes(100, stoppages, context(100, events))
    assert runs(codes) == [(DEAD, 0, 30), (HOME, 30, 60), (AWAY, 60, 100)]


def test_same_team_control_keeps_possession() -> None:
    events = [ControlEvent(team=HOME, start=40, end=50), ControlEvent(team=HOME, start=70, end=80)]
    stoppages = [sm.Stoppage(start=0, restart=30, team=HOME, by_exit=False)]
    assert runs(sm.build_codes(100, stoppages, context(100, events)))[-1] == (HOME, 30, 100)


def test_stoppage_interrupts_possession_and_restart_team_takes_over() -> None:
    stoppages = [sm.Stoppage(start=0, restart=10, team=HOME, by_exit=False),
                 sm.Stoppage(start=50, restart=80, team=AWAY, by_exit=True)]
    codes = sm.build_codes(120, stoppages, context(120, []))
    assert runs(codes) == [(DEAD, 0, 10), (HOME, 10, 50), (DEAD, 50, 80), (AWAY, 80, 120)]


def test_unknown_restart_team_after_exit_goes_to_opponent() -> None:
    stoppages = [sm.Stoppage(start=0, restart=10, team=AWAY, by_exit=False),
                 sm.Stoppage(start=50, restart=80, team=NOBODY, by_exit=True)]
    codes = sm.build_codes(120, stoppages, context(120, []))
    assert runs(codes)[-1] == (HOME, 80, 120)


def test_unknown_restart_team_after_whistle_stays() -> None:
    stoppages = [sm.Stoppage(start=0, restart=10, team=AWAY, by_exit=False),
                 sm.Stoppage(start=50, restart=80, team=NOBODY, by_exit=False)]
    assert runs(sm.build_codes(120, stoppages, context(120, [])))[-1] == (AWAY, 80, 120)


def test_switch_inside_ball_gap_follows_shape_evidence() -> None:
    n_frames = 200
    missing = np.zeros(n_frames, dtype=bool)
    missing[50:150] = True
    shape = np.zeros(n_frames)
    shape[50:90] = 1.0
    shape[90:150] = -1.0
    events = [ControlEvent(team=HOME, start=20, end=40), ControlEvent(team=AWAY, start=150,
                                                                     end=170)]
    stoppages = [sm.Stoppage(start=0, restart=10, team=HOME, by_exit=False)]
    codes = sm.build_codes(n_frames, stoppages, context(n_frames, events, missing, shape))
    assert runs(codes) == [(DEAD, 0, 10), (HOME, 10, 90), (AWAY, 90, n_frames)]


def test_switch_inside_gap_without_evidence_goes_to_gap_middle() -> None:
    n_frames = 200
    missing = np.zeros(n_frames, dtype=bool)
    missing[50:150] = True
    events = [ControlEvent(team=HOME, start=20, end=40), ControlEvent(team=AWAY, start=150,
                                                                     end=170)]
    stoppages = [sm.Stoppage(start=0, restart=10, team=HOME, by_exit=False)]
    codes = sm.build_codes(n_frames, stoppages, context(n_frames, events, missing))
    assert runs(codes)[1:] == [(HOME, 10, 100), (AWAY, 100, n_frames)]


def test_short_gap_does_not_move_switch() -> None:
    n_frames = 100
    missing = np.zeros(n_frames, dtype=bool)
    missing[50:55] = True
    events = [ControlEvent(team=AWAY, start=60, end=70)]
    stoppages = [sm.Stoppage(start=0, restart=10, team=HOME, by_exit=False)]
    codes = sm.build_codes(n_frames, stoppages, context(n_frames, events, missing))
    assert runs(codes)[1:] == [(HOME, 10, 60), (AWAY, 60, n_frames)]


def test_changepoint_picks_best_split() -> None:
    assert sm.changepoint(np.array([1.0, 1.0, -1.0, -1.0, -1.0])) == 2
    assert sm.changepoint(np.array([-1.0, -1.0])) == 0
    assert sm.changepoint(np.zeros(6)) == 3


def test_shape_evidence_points_to_attacking_team() -> None:
    velocity = np.zeros((4, 2, 2))
    velocity[:, :, 0] = [[2.0, 2.0], [-2.0, -2.0], [2.0, 2.0], [1.0, np.nan]]
    attack = np.array([1, 1, -1, 1], dtype=np.int8)
    features = make_features(4, centroid_velocity=velocity, attack_sign=attack)
    assert sm.shape_evidence(features).tolist() == [2.0, -2.0, -2.0, 1.0]


def test_unknown_restart_team_uses_nearest_player_at_restart() -> None:
    nearest = np.full((120, 2), 8.0)
    nearest[80:, AWAY] = 0.4
    stoppages = [sm.Stoppage(start=0, restart=10, team=HOME, by_exit=False),
                 sm.Stoppage(start=50, restart=80, team=NOBODY, by_exit=False)]
    codes = sm.build_codes(120, stoppages, context(120, [], nearest=nearest))
    assert runs(codes)[-1] == (AWAY, 80, 120)


def test_unknown_restart_team_uses_first_team_to_touch_after_restart() -> None:
    nearest = np.full((120, 2), 8.0)
    nearest[90:, HOME] = 0.4
    stoppages = [sm.Stoppage(start=0, restart=10, team=AWAY, by_exit=False),
                 sm.Stoppage(start=50, restart=80, team=NOBODY, by_exit=True)]
    codes = sm.build_codes(120, stoppages, context(120, [], nearest=nearest))
    assert runs(codes)[-1] == (HOME, 80, 120)


def test_first_touch_ignores_ties_and_missing_tracking() -> None:
    nearest = np.full((60, 2), np.inf)
    nearest[10:20] = 0.5
    ctx = context(60, [], nearest=nearest)
    assert sm.first_touch_team(ctx, 10) == NOBODY
    assert sm.first_touch_team(ctx, 30) == NOBODY


def test_without_any_evidence_first_control_decides_never_a_fixed_team() -> None:
    events = [ControlEvent(team=AWAY, start=20, end=40)]
    stoppages = [sm.Stoppage(start=0, restart=10, team=NOBODY, by_exit=False)]
    codes = sm.build_codes(100, stoppages, context(100, events))
    assert runs(codes) == [(DEAD, 0, 10), (AWAY, 10, 100)]


def test_without_any_evidence_or_control_the_segment_stays_dead() -> None:
    stoppages = [sm.Stoppage(start=0, restart=10, team=NOBODY, by_exit=False)]
    assert runs(sm.build_codes(100, stoppages, context(100, []))) == [(DEAD, 0, 100)]
