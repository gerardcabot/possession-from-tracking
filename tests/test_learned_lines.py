from dataclasses import replace

import numpy as np
import pytest
from learned_synthetic import possession_match
from synthetic import FPS, PITCH, make_features

from possession.cleaning import CleaningParams
from possession.data import AWAY, HOME
from possession.external.match import ExternalMatch
from possession.external.predict import to_tracking
from possession.learned import framefeatures as ff
from possession.learned.external import training_example
from possession.learned.model import DeadBallSettings, DecoderSettings, fit_model, predict_codes

SETTINGS = DecoderSettings(emission_weight=0.5, stride=5)
PARAMS = CleaningParams(ball_max_speed_mps=40.0, ball_deviation_m=1.6,
                        player_max_speed_mps=12.0, player_deviation_m=0.3,
                        duplicate_distance_m=0.1, max_gap_frames=8, ball_half_span=1,
                        player_half_span=1)


def segments(n_frames: int) -> ff.Segments:
    return ff.block_segments(n_frames, [(1, 0, n_frames - 1)])


def test_player_beyond_a_line_is_measured_in_metres_outside_it() -> None:
    nan = np.nan
    players = np.array([
        [[50.0, 31.0], [50.0, 62.0], [nan, nan]],
        [[nan, nan], [nan, nan], [nan, nan]],
        [[103.0, 30.0], [50.0, 30.0], [50.0, 30.0]],
    ])
    speed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    columns = ff.lines_columns(players, speed, PITCH)
    np.testing.assert_allclose(columns["player_out_touch"], [2.0, nan, -30.0])
    np.testing.assert_allclose(columns["player_out_goal"], [-50.0, nan, 3.0])
    np.testing.assert_allclose(columns["line_player_speed"], [2.0, nan, 7.0])


def test_keeper_is_the_deepest_player_of_each_team() -> None:
    n_frames = 3
    slot_team = np.array([HOME, HOME, AWAY, AWAY], dtype=np.int8)
    players = np.empty((n_frames, 4, 2))
    players[:] = [[10.0, 30.0], [40.0, 30.0], [90.0, 30.0], [60.0, 30.0]]
    players[2, 0] = [14.0, 30.0]
    speed = np.tile([0.5, 5.0, 0.7, 6.0], (n_frames, 1))
    ball = np.full((n_frames, 2), np.nan)
    ball[0] = [10.0, 40.0]
    features = make_features(n_frames, ball=ball)
    columns = ff.keeper_columns(players, speed, slot_team, features, segments(n_frames))
    np.testing.assert_allclose(columns["keeper_depth"][0], [10.0, 10.0])
    np.testing.assert_allclose(columns["keeper_speed"][0], [0.5, 0.7])
    np.testing.assert_allclose(columns["keeper_depth"][2], [14.0, 10.0])
    np.testing.assert_allclose(columns["keeper_last_ball_distance"][0], [10.0, np.hypot(80, 10)])
    np.testing.assert_allclose(columns["keeper_last_ball_distance"][2],
                               [np.hypot(4, 10), np.hypot(80, 10)])


def test_keeper_features_are_undefined_for_a_missing_team_or_ball() -> None:
    slot_team = np.array([HOME, AWAY], dtype=np.int8)
    players = np.full((2, 2, 2), np.nan)
    players[:, 1] = [60.0, 30.0]
    features = make_features(2, ball=np.full((2, 2), np.nan))
    columns = ff.keeper_columns(players, np.ones((2, 2)), slot_team, features, segments(2))
    assert np.isnan(columns["keeper_depth"][:, HOME]).all()
    assert np.isnan(columns["keeper_speed"][:, HOME]).all()
    assert np.isfinite(columns["keeper_depth"][:, AWAY]).all()
    assert np.isnan(columns["keeper_last_ball_distance"]).all()


def test_keeper_features_are_undefined_for_a_team_without_slots() -> None:
    slot_team = np.array([AWAY, AWAY], dtype=np.int8)
    players = np.full((2, 2, 2), 60.0)
    features = make_features(2, ball=np.full((2, 2), 30.0))
    columns = ff.keeper_columns(players, np.ones((2, 2)), slot_team, features, segments(2))
    assert np.isnan(columns["keeper_depth"][:, HOME]).all()
    assert np.isnan(columns["keeper_last_ball_distance"][:, HOME]).all()
    assert np.isfinite(columns["keeper_depth"][:, AWAY]).all()


def test_ball_line_columns_time_the_crossing_and_the_gap_ends() -> None:
    nan = np.nan
    ball = np.array([[50.0, 20.0], [50.0, 25.0], [nan, nan], [nan, nan], [50.0, 62.0],
                     [50.0, 62.0], [50.0, 40.0]])
    features = make_features(len(ball), ball=ball)
    columns = ff.ball_line_columns(features, segments(len(ball)))
    np.testing.assert_allclose(columns["since_out"] * FPS, [nan, nan, nan, nan, 0, 0, 1])
    np.testing.assert_allclose(columns["until_out"] * FPS, [4, 3, 2, 1, 0, 0, nan])
    np.testing.assert_allclose(columns["entry_touch_distance"][2:4], 25.0)
    np.testing.assert_allclose(columns["exit_touch_distance"][2:4], -2.0)
    np.testing.assert_allclose(columns["entry_goal_distance"][2:4], 50.0)


def test_ball_line_columns_stop_at_block_boundaries() -> None:
    ball = np.array([[50.0, 62.0], [50.0, 30.0], [50.0, 30.0], [50.0, 30.0]])
    features = make_features(4, ball=ball)
    blocks = [(1, 0, 1), (2, 2, 3)]
    columns = ff.ball_line_columns(features, ff.block_segments(4, blocks))
    assert np.isnan(columns["since_out"][2:]).all()
    assert np.isnan(columns["until_out"][2:]).all()


def test_line_groups_are_validated_and_recovered_from_names() -> None:
    ff.check_line_groups(())
    ff.check_line_groups(ff.LINE_GROUPS)
    with pytest.raises(ValueError, match="line feature groups"):
        ff.check_line_groups(("lines", "goalposts"))
    for group, names in ff.LINE_GROUP_NAMES.items():
        assert ff.line_groups_in(["mean_speed", *sorted(names)]) == (group,)
    assert ff.line_groups_in(["mean_speed", "ball_present_past_25"]) == ()
    assert ff.line_groups_in(["since_out", "player_out_touch"]) == ("lines", "ball")


@pytest.fixture(scope="module")
def match() -> ExternalMatch:
    return replace(possession_match(seed=0), name="m0")


def test_default_features_do_not_contain_line_columns(match: ExternalMatch) -> None:
    base = training_example(match, PARAMS).base[0]
    every_name = set().union(*ff.LINE_GROUP_NAMES.values())
    assert not every_name & set(base.names)
    layout = ff.FeatureLayout(scales=(25,), touch=True)
    assert not any(name.startswith(tuple(ff.LINE_PLAYER_SIGNALS))
                   for name in ff.feature_names(base, layout))


def test_requested_groups_add_exactly_their_columns(match: ExternalMatch) -> None:
    plain = set(training_example(match, PARAMS).base[0].names)
    for group, names in ff.LINE_GROUP_NAMES.items():
        base = training_example(match, PARAMS, lines=(group,)).base[0]
        assert set(base.names) - plain == names


def test_perspectives_swap_keeper_columns_and_share_line_columns(match: ExternalMatch) -> None:
    example = training_example(match, PARAMS, lines=ff.LINE_GROUPS)
    home, away = example.base
    np.testing.assert_array_equal(home.column("keeper_depth_own"),
                                  away.column("keeper_depth_other"))
    np.testing.assert_array_equal(home.column("player_out_touch"),
                                  away.column("player_out_touch"))
    np.testing.assert_array_equal(home.column("since_out"), away.column("since_out"))


def test_position_only_line_features_feed_the_dead_ball_model(match: ExternalMatch) -> None:
    base = training_example(match, PARAMS, lines=ff.LINE_GROUPS).base[0]
    names = ff.feature_names(base, ff.FeatureLayout(scales=(25,)))
    chosen = {names[index] for index in ff.dead_columns(names, "players")}
    assert {"player_out_touch", "player_out_goal", "line_player_speed", "keeper_depth_own",
            "keeper_speed_other", "player_out_touch_past_25",
            "keeper_speed_own_future_25"} <= chosen
    for name in ("since_out", "until_out", "entry_touch_distance", "exit_goal_distance",
                 "keeper_last_ball_distance_own"):
        assert name in names and name not in chosen


def test_model_with_line_groups_predicts_from_raw_tracking(match: ExternalMatch) -> None:
    groups = ("lines", "keeper")
    examples = [training_example(replace(match, name=f"m{seed}"), PARAMS, lines=groups)
                for seed in range(2)]
    model = fit_model(examples, SETTINGS, 4, DeadBallSettings("players", 0), "touch",
                      switch_prior=False)
    assert ff.line_groups_in(model.classifier.feature_names) == groups
    assert {"player_out_touch", "keeper_depth_own"} <= set(model.classifier.dead_feature_names)
    codes = predict_codes(model, to_tracking(match), match.pitch, match.fps, match.blocks)
    assert len(codes) == match.n_frames
    assert set(np.unique(codes).tolist()) <= {0, 1, 2}


def test_model_without_line_groups_ignores_them_at_prediction(match: ExternalMatch) -> None:
    example = training_example(match, PARAMS)
    model = fit_model([example], SETTINGS, 4, DeadBallSettings(), "core", switch_prior=False)
    assert ff.line_groups_in(model.classifier.feature_names) == ()
    assert len(predict_codes(model, to_tracking(match), match.pitch, match.fps,
                             match.blocks)) == match.n_frames
