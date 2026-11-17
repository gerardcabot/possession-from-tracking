from dataclasses import replace

import numpy as np
import pytest
from learned_synthetic import possession_match
from synthetic import FPS, make_features, still_then_kicked

from possession.cleaning import CleaningParams
from possession.data import AWAY, HOME
from possession.intervals import DEAD
from possession.learned import framefeatures as ff
from possession.learned.adapt import observation_columns
from possession.learned.external import training_example
from possession.learned.hsmm import weighted_runs
from possession.learned.model import (
    NO_TOUCH_EVIDENCE_THRESHOLD,
    DeadBallSettings,
    DecoderSettings,
    TouchEvidence,
    fit_model,
    model_bytes,
    timed_sequences,
    touch_threshold,
)

SETTINGS = DecoderSettings(emission_weight=0.5, stride=5)
PARAMS = CleaningParams(ball_max_speed_mps=40.0, ball_deviation_m=1.6,
                        player_max_speed_mps=12.0, player_deviation_m=0.3,
                        duplicate_distance_m=0.1, max_gap_frames=8, ball_half_span=1,
                        player_half_span=1)
KICK = 40
N_FRAMES = 100


def kicked_inputs(blocks: ff.Blocks | None = None, kick: int = KICK,
                  ball: np.ndarray | None = None, **overrides: object) -> ff.MatchInputs:
    blocks = [(1, 0, N_FRAMES - 1)] if blocks is None else blocks
    ball = still_then_kicked(N_FRAMES, kick, (30.0, 30.0)) if ball is None else ball
    nearest = np.column_stack([np.full(N_FRAMES, 1.0), np.full(N_FRAMES, 6.0)])
    features = make_features(N_FRAMES, ball=ball, nearest=nearest, **overrides)
    segments = ff.block_segments(N_FRAMES, blocks)
    return ff.MatchInputs(features=features, spread=np.ones((N_FRAMES, 2, 2)),
                          segments=segments, params=PARAMS,
                          touch=ff.touch_matrix(features, segments, PARAMS.ball_half_span))


def touch_base(inputs: ff.MatchInputs) -> dict[str, np.ndarray]:
    base = ff.base_features(inputs, HOME)
    return {name: base.column(name) for name in ff.TOUCH_COLUMNS}


def events(inputs: ff.MatchInputs, first: int = HOME, threshold: float = 0.0
           ) -> dict[str, np.ndarray]:
    return ff.touch_events(ff.base_features(inputs, first), threshold)


def touches(columns: dict[str, np.ndarray], side: str) -> np.ndarray:
    return np.flatnonzero(np.nan_to_num(columns[f"touch_count_{side}"]) > 0)


def test_touch_half_span_is_the_metric_tolerance_in_frames_of_the_match() -> None:
    assert ff.touch_half_span(25.0) == 12
    assert ff.touch_half_span(50.0) == 24


def test_local_peaks_keep_one_positive_maximum_per_window() -> None:
    values = np.array([0.0, 1.0, 3.0, 2.0, 0.0, 0.0, 5.0, np.nan, 4.0, 0.0])
    assert np.flatnonzero(ff.local_peaks(values, 1)).tolist() == [2, 6, 8]
    assert np.flatnonzero(ff.local_peaks(values, 2)).tolist() == [2, 6]
    assert not ff.local_peaks(np.zeros(5), 1).any()
    assert ff.local_peaks(np.zeros(0), 1).shape == (0,)


@pytest.mark.parametrize("half_span", [1, 2, 6])
def test_local_peaks_keep_the_first_frame_of_a_plateau(half_span: int) -> None:
    values = np.array([0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 3.0, 3.0, 1.0])
    assert np.flatnonzero(ff.local_peaks(values, half_span)).tolist() == [1, 15]


def test_kick_is_the_strongest_peak_and_a_touch_of_the_nearest_team() -> None:
    inputs = kicked_inputs()
    base = touch_base(inputs)
    peaks = np.flatnonzero(base["touch_peak"] == 1)
    kick = int(peaks[np.argmax(base["ball_impulse"][peaks])])
    assert abs(kick - KICK) <= 2
    assert base["ball_speed_change"][kick] > 0
    columns = events(inputs, threshold=float(base["ball_impulse"][kick]))
    assert touches(columns, "own").tolist() == [kick]
    assert len(touches(columns, "other")) == 0
    assert columns["touch_own"][kick] == pytest.approx(base["ball_impulse"][kick])
    after = slice(kick, N_FRAMES)
    np.testing.assert_array_equal(columns["touch_back_team"][after], 1.0)
    np.testing.assert_allclose(columns["touch_since_own"][after] * FPS,
                               np.arange(N_FRAMES - kick), atol=1e-4)
    assert np.isnan(columns["touch_since_other"]).all()
    assert np.isnan(columns["touch_back_team"][:kick]).all()
    np.testing.assert_allclose(columns["touch_forward_time"][:kick + 1] * FPS,
                               np.arange(kick, -1, -1), atol=1e-4)
    np.testing.assert_allclose(columns["touch_back_closest"][after], 1.0)
    assert len(touches(events(inputs), "own")) >= 1


def test_touch_names_match_the_columns() -> None:
    inputs = kicked_inputs()
    columns = ff.touch_columns(inputs.features, inputs.segments, PARAMS.ball_half_span)
    assert set(columns) == set(ff.TOUCH_COLUMNS)
    assert set(events(inputs)) == set(ff.TOUCH_EVENTS)
    assert set(ff.TOUCH_SIGNALS) <= set(ff.TOUCH_EVENTS)


def test_away_perspective_swaps_touch_teams() -> None:
    inputs = kicked_inputs()
    home, away = events(inputs, HOME), events(inputs, AWAY)
    np.testing.assert_array_equal(home["touch_own"], away["touch_other"])
    np.testing.assert_array_equal(home["touch_since_own"], away["touch_since_other"])
    np.testing.assert_array_equal(home["touch_back_team"], -away["touch_back_team"])
    np.testing.assert_array_equal(home["touch_back_impulse"], away["touch_back_impulse"])


def test_touch_features_stop_at_block_boundaries() -> None:
    boundary = 50
    blocks = [(1, 0, boundary - 1), (2, boundary, N_FRAMES - 1)]
    inputs = kicked_inputs(blocks)
    base = touch_base(inputs)
    assert np.isnan(base["ball_impulse"][boundary - 1])
    assert np.isnan(base["ball_impulse"][N_FRAMES - 1])
    threshold = float(np.nanmax(base["ball_impulse"][:boundary]))
    columns = events(inputs, threshold=threshold)
    kick = int(touches(columns, "own")[0])
    assert kick < boundary
    assert np.isnan(columns["touch_back_time"][boundary:]).all()
    assert np.isnan(columns["touch_since_own"][boundary:]).all()
    assert np.isnan(columns["touch_forward_time"][kick + 1:boundary]).all()
    features = ff.base_features(inputs, HOME)
    layout = ff.FeatureLayout(scales=(80,), touch=True, touch_threshold=threshold)
    names = ff.feature_names(features, layout)
    matrix = ff.feature_matrix(features, layout)
    np.testing.assert_array_equal(matrix[boundary:, names.index("touch_count_own_past_80")], 0.0)
    assert matrix[boundary - 1, names.index("touch_count_own_past_80")] > 0


def test_missing_ball_gives_no_touches_and_undefined_strengths() -> None:
    inputs = kicked_inputs(ball=np.full((N_FRAMES, 2), np.nan))
    base = touch_base(inputs)
    assert np.isnan(base["ball_impulse"]).all() and np.isnan(base["touch_peak"]).all()
    columns = events(inputs)
    for name in ("touch_own", "touch_count_other", "touch_back_team", "touch_since_own"):
        assert np.isnan(columns[name]).all(), name


def test_peak_evidence_labels_changes_and_steady_spells() -> None:
    inputs = kicked_inputs()
    base = ff.base_features(inputs, HOME)
    peaks = np.flatnonzero(base.column("touch_peak") == 1)
    codes = np.full(N_FRAMES, HOME)
    codes[KICK:] = AWAY
    scores, changes = ff.peak_evidence(base, codes, [(1, 0, N_FRAMES - 1)])
    span = ff.touch_half_span(FPS)
    near = np.abs(peaks - KICK) <= span
    np.testing.assert_array_equal(changes, near)
    np.testing.assert_array_equal(scores, base.column("ball_impulse")[peaks])
    codes[:] = DEAD
    assert len(ff.peak_evidence(base, codes, [(1, 0, N_FRAMES - 1)])[0]) == 0


def test_youden_threshold_separates_weighted_scores() -> None:
    scores = np.array([0.1, 0.2, 0.3, 2.0, 3.0, 0.15])
    positive = np.array([False, False, False, True, True, True])
    assert ff.youden_threshold(scores, positive, np.ones(6)) == 2.0
    weights = np.array([1, 1, 1, 1, 1, 10.0])
    assert ff.youden_threshold(scores, positive, weights) == 0.15
    with pytest.raises(ValueError, match="both changes and steady"):
        ff.youden_threshold(scores, np.ones(6, dtype=bool), np.ones(6))
    with pytest.raises(ValueError, match="both changes and steady"):
        ff.youden_threshold(scores, np.zeros(6, dtype=bool), np.ones(6))


def test_short_scales_are_weighted_quartiles_of_possession_runs() -> None:
    lengths = np.array([10, 20, 30, 40, 500])
    states = np.array([HOME, AWAY, HOME, AWAY, DEAD])
    assert ff.short_scales(lengths, states, np.ones(5), (25, 500)) == (10, 20)
    assert ff.short_scales(lengths, states, np.ones(5), (20,)) == (10,)
    assert ff.short_scales(lengths, states, np.array([1, 1, 6, 1, 1]), ()) == (30,)
    assert ff.short_scales(lengths[4:], states[4:], np.ones(1), ()) == ()


def test_feature_layout_per_team_feature_set() -> None:
    lengths, states = np.array([10, 20, 30, 40, 500]), np.array([HOME, AWAY, HOME, AWAY, DEAD])
    weights = np.ones(5)
    core = ff.feature_layout("core", lengths, states, weights)
    short = ff.feature_layout("short", lengths, states, weights)
    touch = ff.feature_layout("touch", lengths, states, weights)
    assert core == ff.FeatureLayout(scales=(25, 500))
    assert short == ff.FeatureLayout(scales=(25, 500), short_scales=(10, 20))
    assert touch == replace(short, touch=True) and touch.core == core
    assert ff.feature_layout("touch", lengths, states, weights, 2.5).touch_threshold == 2.5
    assert ff.feature_layout("short", lengths, states, weights, 2.5).touch_threshold == 0.0
    with pytest.raises(ValueError, match="team feature set"):
        ff.feature_layout("ball", lengths, states, weights)


def test_core_features_are_a_prefix_of_every_layout() -> None:
    base = ff.base_features(kicked_inputs(), HOME)
    layout = ff.FeatureLayout(scales=(8, 30), short_scales=(3,), touch=True)
    core = ff.feature_names(base, layout.core)
    names = ff.feature_names(base, layout)
    assert names[:len(core)] == core
    assert not set(core) & {*ff.TOUCH_COLUMNS, *ff.TOUCH_EVENTS}
    np.testing.assert_array_equal(ff.feature_matrix(base, layout)[:, :len(core)],
                                  ff.feature_matrix(base, layout.core))


def examples(seeds: tuple[int, ...]) -> list:
    return [training_example(replace(possession_match(seed=seed), name=f"m{seed}"), PARAMS)
            for seed in seeds]


@pytest.mark.parametrize("dead_ball", [DeadBallSettings(),
                                       DeadBallSettings(features="players",
                                                        adaptation_iterations=2)])
def test_new_team_features_leave_the_dead_ball_model_unchanged(
        dead_ball: DeadBallSettings) -> None:
    train = examples((0, 1))
    core = fit_model(train, SETTINGS, 4, dead_ball, "core")
    touch = fit_model(train, SETTINGS, 4, dead_ball, "touch")
    assert touch.layout.touch and touch.layout.core == core.layout
    assert touch.classifier.dead_feature_names == core.classifier.dead_feature_names
    assert observation_columns(touch.classifier) == observation_columns(core.classifier)
    base = train[0].base[0]
    np.testing.assert_array_equal(
        touch.classifier.dead_logit(ff.feature_matrix(base, touch.layout)),
        core.classifier.dead_logit(ff.feature_matrix(base, core.layout)))
    assert len(touch.classifier.feature_names) > len(core.classifier.feature_names)


def test_touch_model_is_deterministic() -> None:
    train = examples((0, 1))
    assert model_bytes(fit_model(train, SETTINGS, 4)) == model_bytes(fit_model(train, SETTINGS, 4))


def test_layout_is_estimated_from_the_training_matches_only() -> None:
    train = examples((0, 1))
    held_out = examples((2,))[0]
    long_runs = held_out.codes.copy()
    long_runs[:] = np.where(np.arange(len(long_runs)) < len(long_runs) // 2, HOME, AWAY)
    held_out = replace(held_out, codes=long_runs)
    model = fit_model(train, SETTINGS, 4)
    expected = ff.feature_layout("touch", *weighted_runs(timed_sequences(train)),
                                 touch_threshold(train).threshold)
    assert model.layout == expected
    with_held_out = [*train, held_out]
    assert model.layout != ff.feature_layout(
        "touch", *weighted_runs(timed_sequences(with_held_out)),
        touch_threshold(with_held_out).threshold)
    assert model.training_matches == tuple(example.name for example in train)


def test_touch_velocity_does_not_cross_a_block_boundary() -> None:
    boundary = 50
    ball = np.full((N_FRAMES, 2), 30.0)
    ball[boundary:, 0] = 70.0
    inputs = kicked_inputs([(1, 0, boundary - 1), (2, boundary, N_FRAMES - 1)], ball=ball)
    assert np.nanmax(np.abs(inputs.features.ball_velocity[boundary - 1:boundary + 1])) > 0
    base = touch_base(inputs)
    np.testing.assert_array_equal(np.nan_to_num(base["ball_impulse"]), 0.0)
    assert not (base["touch_peak"] == 1).any()


def test_absent_ball_frames_are_masked_before_peak_search() -> None:
    hidden = np.arange(KICK - 1, KICK + 2)
    present = np.ones(N_FRAMES, dtype=bool)
    present[hidden] = False
    base = touch_base(kicked_inputs(ball_present=present))
    assert np.isnan(base["ball_impulse"][hidden]).all()
    assert np.isnan(base["touch_peak"][hidden]).all()
    peaks = np.flatnonzero(base["touch_peak"] == 1)
    assert present[peaks].all()
    assert (np.abs(peaks - KICK) <= ff.touch_half_span(FPS)).any()


def test_peak_evidence_clamps_blocks_to_the_match() -> None:
    base = ff.base_features(kicked_inputs(), HOME)
    codes = np.full(N_FRAMES, HOME)
    codes[KICK:] = AWAY
    inside = ff.peak_evidence(base, codes, [(1, 0, N_FRAMES - 1)])
    beyond = ff.peak_evidence(base, codes, [(1, -10, N_FRAMES + 10)])
    for got, expected in zip(beyond, inside):
        np.testing.assert_array_equal(got, expected)


def test_touch_threshold_uses_team_only_matches() -> None:
    train = examples((0, 1))
    team_only = [replace(example, dead_labels=False) for example in train]
    with_dead = touch_threshold(train)
    assert with_dead.peaks > 0
    assert touch_threshold(team_only) == with_dead


def test_touch_threshold_falls_back_only_without_evidence(
        capsys: pytest.CaptureFixture[str]) -> None:
    train = [replace(example, codes=np.full_like(example.codes, DEAD))
             for example in examples((0,))]
    evidence = touch_threshold(train)
    assert evidence == TouchEvidence(threshold=NO_TOUCH_EVIDENCE_THRESHOLD, peaks=0)
    assert "no labelled impulse peak" in capsys.readouterr().err


def test_touch_inputs_are_built_once_per_match_and_only_when_needed() -> None:
    match = replace(possession_match(seed=0), name="m0")
    with_touch = training_example(match, PARAMS)
    assert with_touch.base[0].touch is with_touch.base[1].touch
    without = training_example(match, PARAMS, touch=False)
    assert all(base.touch is None for base in without.base)
    core = fit_model([without], SETTINGS, 4, DeadBallSettings(), "core", switch_prior=False)
    assert not core.layout.touch
    assert core.changes is None
    with pytest.raises(ValueError, match="touch inputs"):
        fit_model([without], SETTINGS, 4, DeadBallSettings(), "touch", switch_prior=False)
    with pytest.raises(ValueError, match="touch inputs"):
        fit_model([without], SETTINGS, 4, DeadBallSettings(), "core")


def test_touch_events_are_computed_once_per_match_and_threshold(
        monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = ff.perspective_events

    def counted(base: ff.BaseFeatures, threshold: float) -> dict[str, np.ndarray]:
        calls.append(threshold)
        return original(base, threshold)

    monkeypatch.setattr(ff, "perspective_events", counted)
    monkeypatch.setattr(ff, "LAST_TOUCH_EVENTS", ff.TouchEventCache())
    inputs = kicked_inputs()
    home, away = ff.base_features(inputs, HOME), ff.base_features(inputs, AWAY)
    layout = ff.FeatureLayout(scales=(8,), short_scales=(3,), touch=True)
    first = ff.feature_matrix(home, layout)
    ff.feature_matrix(away, layout)
    np.testing.assert_array_equal(ff.feature_matrix(home, layout), first)
    assert calls == [0.0]
    ff.feature_matrix(home, replace(layout, touch_threshold=1.0))
    assert calls == [0.0, 1.0]
