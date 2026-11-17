import numpy as np
from synthetic import FPS, PITCH, make_features

from possession import deadball
from possession.params import RulesParams

NAN = np.nan
PARAMS = RulesParams()


def test_outside_pitch_uses_ball_radius_margin() -> None:
    ball = np.array([[-0.05, 10.0], [-0.2, 10.0], [50.0, 60.3], [100.0, 30.0], [NAN, NAN]])
    assert deadball.outside_pitch(ball, PITCH, 0.11).tolist() == [False, True, True, False,
                                                                  False]


def test_observed_exit_is_first_frame_over_the_line() -> None:
    outside = np.array([False, False, True, True, False, True])
    assert deadball.observed_exits(outside).tolist() == [2, 5]


def test_gap_exit_extrapolates_crossing_inside_gap() -> None:
    n_frames = 40
    ball = np.full((n_frames, 2), NAN)
    ball[:10] = np.column_stack([95.0 + np.arange(10) * 0.4, np.full(10, 30.0)])
    exits = deadball.gap_exits(make_features(n_frames, ball=ball), PARAMS)
    assert exits.tolist() == [13]


def test_gap_exit_ignores_ball_moving_away_from_line() -> None:
    n_frames = 40
    ball = np.full((n_frames, 2), NAN)
    ball[:10] = np.column_stack([99.0 - np.arange(10) * 0.4, np.full(10, 30.0)])
    assert deadball.gap_exits(make_features(n_frames, ball=ball), PARAMS).size == 0


def test_speed_evidence_is_capped_and_missing_is_neutral() -> None:
    evidence = deadball.speed_evidence(np.array([0.0, 1.5, 9.0, NAN]), PARAMS)
    assert evidence.tolist() == [3.0, 0.0, -3.0, 0.0]


def test_untracked_frames_alone_do_not_make_a_stoppage() -> None:
    n_frames = 400
    mean_speed = np.full(n_frames, 4.0)
    mean_speed[100:300] = NAN
    result = deadball.dead_evidence(make_features(n_frames, mean_speed=mean_speed), PARAMS)
    assert not deadball.dead_mask(result.evidence, PARAMS, FPS)[150:250].any()


def test_dead_mask_needs_a_sustained_stoppage() -> None:
    evidence = np.full(1000, -2.0)
    evidence[200:230] = 3.0
    evidence[500:700] = 3.0
    dead = deadball.dead_mask(evidence, PARAMS, FPS)
    assert not dead[:400].any()
    assert dead[520:680].all()


def test_dead_evidence_marks_ball_out_of_play() -> None:
    n_frames = 300
    ball = np.tile([50.0, 30.0], (n_frames, 1))
    ball[150:, 1] = -2.0
    result = deadball.dead_evidence(make_features(n_frames, ball=ball, mean_speed=2.0), PARAMS)
    assert result.outside[150:].all() and not result.outside[:150].any()
    assert 150 in result.crossings.tolist()
    assert (result.evidence[160:] > result.evidence[:140].max()).all()


def test_refine_start_prefers_a_nearby_line_crossing() -> None:
    collective = np.linspace(4.0, 1.0, 400)
    begin, by_exit = deadball.refine_start(200, np.array([150, 190, 300]), collective, 0, 75)
    assert (begin, by_exit) == (150, True)


def test_refine_start_falls_back_to_deceleration_onset() -> None:
    collective = np.concatenate([np.full(100, 2.0), [4.0], np.linspace(3.5, 0.5, 99)])
    begin, by_exit = deadball.refine_start(150, np.array([], dtype=np.int64), collective, 0, 75)
    assert (begin, by_exit) == (100, False)


def test_refine_start_never_goes_before_previous_restart() -> None:
    collective = np.concatenate([[9.0], np.full(199, 1.0)])
    begin, _ = deadball.refine_start(60, np.array([0]), collective, 30, 75)
    assert begin >= 30
