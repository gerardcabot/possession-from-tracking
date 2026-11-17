import pickle
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
from learned_synthetic import possession_match

from possession.cleaning import CleaningParams
from possession.external.match import ExternalMatch
from possession.external.predict import to_tracking
from possession.intervals import DEAD
from possession.learned import deadcontext as dc
from possession.learned import framefeatures as ff
from possession.learned.adapt import adapted_dead_logit
from possession.learned.classifier import FrameClassifier, TrainingRows
from possession.learned.external import training_example
from possession.learned.model import (
    DEFAULT_FITTING,
    ClassifierSettings,
    DeadBallSettings,
    DecoderSettings,
    LearnedModel,
    TrainingExample,
    fit_model,
    load_model,
    log_emission,
    model_metadata,
    predict_codes,
    save_model,
    training_keep,
    training_rows,
    validation_groups,
)
from possession.learned.train import fitting_record

SETTINGS = DecoderSettings(emission_weight=0.5, stride=5)
PARAMS = CleaningParams(ball_max_speed_mps=40.0, ball_deviation_m=1.6,
                        player_max_speed_mps=12.0, player_deviation_m=0.3,
                        duplicate_distance_m=0.1, max_gap_frames=8, ball_half_span=1,
                        player_half_span=1)
CONTEXT = ClassifierSettings(dead_context=True)
FRAME_STEP = 4


def make_matches(seeds: tuple[int, ...] = (0, 1, 2)) -> list[ExternalMatch]:
    return [replace(possession_match(seed=seed), name=f"dfl/m{seed}") for seed in seeds]


def make_examples(seeds: tuple[int, ...] = (0, 1, 2)) -> list[TrainingExample]:
    return [training_example(match, PARAMS) for match in make_matches(seeds)]


@pytest.fixture(scope="module")
def train() -> list[TrainingExample]:
    return make_examples()


@pytest.fixture(scope="module")
def plain(train: list[TrainingExample]) -> LearnedModel:
    return fit_model(train, SETTINGS, FRAME_STEP, DeadBallSettings("players", 2), "touch",
                     switch_prior=False)


@pytest.fixture(scope="module")
def contextual(train: list[TrainingExample]) -> LearnedModel:
    return fit_model(train, SETTINGS, FRAME_STEP, DeadBallSettings("players", 2), "touch",
                     switch_prior=False, fitting=CONTEXT)


def segments(*bounds: int) -> ff.Segments:
    return tuple(zip(bounds[:-1], bounds[1:]))


def grouped_rows(train: list[TrainingExample], layout: ff.FeatureLayout) -> TrainingRows:
    return training_rows(train, layout, FRAME_STEP, validation_groups(train))


def dead_inputs(rows: TrainingRows, columns: tuple[int, ...]
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    index = np.flatnonzero((rows.codes >= 0) & rows.timed)
    return (rows.matrix[np.ix_(index, columns)], rows.codes[index] == DEAD, rows.match[index],
            rows.group[index])


def test_centred_mean_is_the_mean_of_the_window_around_each_frame() -> None:
    values = np.array([1.0, 2.0, np.nan, 4.0, 8.0, 16.0])
    expected = [np.nanmean(values[max(i - 1, 0):i + 2]) for i in range(len(values))]
    np.testing.assert_allclose(ff.centred_nanmean(values, 3), expected)
    np.testing.assert_allclose(ff.centred_nanmean(values, 4)[3], np.nanmean(values[1:5]))
    np.testing.assert_allclose(ff.centred_nanmean(values, 1), values)


def test_directional_means_are_unchanged_by_the_shared_window_helper() -> None:
    values = np.array([1.0, 2.0, np.nan, 4.0, 8.0, 16.0])
    np.testing.assert_allclose(ff.directional_nanmean(values, 3, forward=False),
                               [1.0, 1.5, 1.5, 3.0, 6.0, 28.0 / 3.0])
    np.testing.assert_allclose(ff.directional_nanmean(values, 3, forward=True),
                               [1.5, 3.0, 6.0, 9.333333333, 12.0, 16.0])


def test_context_columns_never_cross_a_segment_boundary() -> None:
    logit = np.concatenate([np.full(6, -2.0), np.full(6, 1000.0)])
    scales = (3, 5)
    columns = dc.context_columns(logit, segments(0, 6, 12), scales)
    names = dc.context_names(scales)
    assert columns.shape == (12, len(names))
    assert len(set(names)) == len(names)
    assert np.all(columns[:6] == np.float32(-2.0))
    assert np.all(columns[6:] == np.float32(1000.0))


def test_context_columns_look_back_forward_and_around() -> None:
    logit = np.arange(10, dtype=float)
    columns = dc.context_columns(logit, segments(0, 10), (3,))
    by_name = dict(zip(dc.context_names((3,)), columns.T))
    np.testing.assert_allclose(by_name["dead_logit_past_3"][5], np.mean(logit[3:6]))
    np.testing.assert_allclose(by_name["dead_logit_future_3"][5], np.mean(logit[5:8]))
    np.testing.assert_allclose(by_name["dead_logit_centred_3"][5], np.mean(logit[4:7]))


def test_out_of_fold_logits_never_see_the_labels_of_their_own_group(
        train: list[TrainingExample], plain: LearnedModel) -> None:
    layout, columns = plain.layout, plain.classifier.dead_columns
    rows = grouped_rows(train, layout)
    bases = [example.base for example in train]
    dead_matrix, labels, match, group = dead_inputs(rows, columns)
    reference = dc.out_of_fold_logits(dead_matrix, labels, match, group, layout, columns, bases,
                                      0, "default")
    held = 1
    changed = dc.out_of_fold_logits(dead_matrix, np.where(match == held, ~labels, labels), match,
                                    group, layout, columns, bases, 0, "default")
    assert set(reference) == {0, 1, 2}
    for position in range(len(train[held].base)):
        np.testing.assert_array_equal(reference[held][position], changed[held][position])
    for other in (0, 2):
        assert not np.array_equal(reference[other][0], changed[other][0])


def test_out_of_fold_logits_use_one_fit_per_validation_group_not_per_match(
        train: list[TrainingExample], plain: LearnedModel,
        monkeypatch: pytest.MonkeyPatch) -> None:
    layout, columns = plain.layout, plain.classifier.dead_columns
    rows = grouped_rows(train, layout)
    fits: list[int] = []
    original = dc.fit_dead_ball

    def counted(*args: object, **kwargs: object) -> object:
        fits.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(dc, "fit_dead_ball", counted)
    dead_matrix, labels, match, group = dead_inputs(rows, columns)
    dc.out_of_fold_logits(dead_matrix, labels, match, group, layout, columns,
                          [example.base for example in train], 0, "default")
    assert len(fits) == len(np.unique(group))


def test_dead_columns_only_match_the_columns_of_the_full_matrix(
        train: list[TrainingExample], plain: LearnedModel) -> None:
    base, columns = train[0].base[0], plain.classifier.dead_columns
    full = ff.feature_matrix(base, plain.layout)
    np.testing.assert_array_equal(ff.feature_matrix(base, plain.layout.core, columns=columns),
                                  full[:, list(columns)])
    keep = training_keep(train[0], FRAME_STEP)
    np.testing.assert_array_equal(ff.feature_matrix(base, plain.layout, keep, columns),
                                  full[keep][:, list(columns)])


def test_out_of_fold_logits_differ_from_the_in_sample_model(
        train: list[TrainingExample], plain: LearnedModel) -> None:
    layout, columns = plain.layout, plain.classifier.dead_columns
    rows = grouped_rows(train, layout)
    dead_matrix, labels, match, group = dead_inputs(rows, columns)
    oof = dc.out_of_fold_logits(dead_matrix, labels, match, group, layout, columns,
                                [example.base for example in train], 0, "default")
    in_sample = plain.classifier.dead.decision_function(
        ff.feature_matrix(train[0].base[0], layout.core, columns=columns))
    assert oof[0][0].shape == in_sample.shape
    assert not np.array_equal(oof[0][0], in_sample)


def test_a_split_without_both_dead_ball_classes_is_a_clear_error(
        train: list[TrainingExample], plain: LearnedModel) -> None:
    layout, columns = plain.layout, plain.classifier.dead_columns
    rows = grouped_rows(train, layout)
    dead_matrix, labels, match, group = dead_inputs(rows, columns)
    with pytest.raises(ValueError, match="do not contain both dead-ball classes"):
        dc.out_of_fold_logits(dead_matrix, np.zeros_like(labels), match, group, layout, columns,
                              [example.base for example in train], 0, "default")
    one_group = np.zeros_like(group)
    with pytest.raises(ValueError, match="do not contain both dead-ball classes"):
        dc.out_of_fold_logits(dead_matrix, labels, match, one_group, layout, columns,
                              [example.base for example in train], 0, "default")


def test_context_needs_validation_groups_of_at_least_two_matches() -> None:
    with pytest.raises(ValueError, match="at least two training matches"):
        fit_model(make_examples((0,)), SETTINGS, FRAME_STEP, DeadBallSettings("players", 0),
                  "touch", switch_prior=False, fitting=CONTEXT)


def test_context_is_built_only_for_dead_ball_rows_in_training_row_order(
        train: list[TrainingExample]) -> None:
    layout = ff.FeatureLayout(scales=(25, 50))
    rows = training_rows(train, layout, FRAME_STEP)
    keeps = [training_keep(example, FRAME_STEP) for example in train]
    bases = [example.base for example in train]
    timed = (rows.codes >= 0) & rows.timed & (rows.match != 1)
    logits = {match: [np.full(len(base.matrix), float(10 * match + position))
                      for position, base in enumerate(match_bases)]
              for match, match_bases in enumerate(bases) if match != 1}
    width = len(dc.context_names(layout.all_scales))
    stage_two = np.full((int(timed.sum()), 2 + width), np.nan, dtype=np.float32)
    dc.fill_context(stage_two, 2, timed, logits, bases, keeps, layout.all_scales)
    assert not np.isnan(stage_two[:, 2:]).any() and np.isnan(stage_two[:, :2]).all()
    per_perspective = {match: int(timed[rows.match == match].sum()) // len(ff.PERSPECTIVES)
                       for match in (0, 2)}
    expected = np.concatenate([np.full(per_perspective[match], 10 * match + position)
                               for match in (0, 2) for position in range(len(ff.PERSPECTIVES))])
    np.testing.assert_allclose(stage_two[:, 2], expected, atol=1e-4)


def test_flag_off_builds_the_plain_classifier_and_records_nothing(plain: LearnedModel) -> None:
    assert type(plain.classifier) is FrameClassifier
    assert dc.context_metadata(plain.classifier) == {}
    assert "dead_context" not in model_metadata(plain, b"", "run", {})
    assert fitting_record(DEFAULT_FITTING) == {}
    assert fitting_record(ClassifierSettings(every_peak=True)) == {
        "classifier": {"spell_weights": False, "every_peak": True, "stopping": "default"}}
    assert ClassifierSettings() == DEFAULT_FITTING


def test_flag_off_emission_is_the_unchanged_classifier_path(
        plain: LearnedModel, train: list[TrainingExample]) -> None:
    example = train[0]
    base = example.base[0]
    matrix = ff.feature_matrix(base, plain.layout)
    unadapted = replace(plain, dead_ball=DeadBallSettings("players", 0))
    emission, report = log_emission(unadapted, base, example.blocks)
    assert report is None
    np.testing.assert_array_equal(emission, plain.classifier.log_emission(matrix))
    adapted, report = log_emission(plain, base, example.blocks)
    assert report is not None and report.shares
    assert adapted.shape == emission.shape


def test_flag_on_records_the_settings_only_when_on() -> None:
    assert fitting_record(CONTEXT) == {"classifier": {
        "spell_weights": False, "every_peak": False, "stopping": "default", "dead_context": True}}
    assert fitting_record(replace(CONTEXT, every_peak=True))["classifier"]["dead_context"] is True


def test_stage_one_and_the_team_model_are_untouched_by_the_flag(
        plain: LearnedModel, contextual: LearnedModel) -> None:
    off, on = plain.classifier, contextual.classifier
    assert isinstance(on, dc.ContextClassifier)
    assert pickle.dumps(on.dead) == pickle.dumps(off.dead)
    assert pickle.dumps(on.side) == pickle.dumps(off.side)
    np.testing.assert_array_equal(on.log_prior, off.log_prior)
    assert on.dead_columns == off.dead_columns
    assert on.feature_names == off.feature_names
    assert on.observation_limit == off.observation_limit
    assert contextual.layout == plain.layout


def test_context_classifier_copies_every_field_of_the_classifier(
        plain: LearnedModel, contextual: LearnedModel) -> None:
    for item in fields(FrameClassifier):
        copied, original = (getattr(model.classifier, item.name) for model in (contextual, plain))
        assert pickle.dumps(copied) == pickle.dumps(original)


def test_model_stores_its_context_settings_explicitly(contextual: LearnedModel) -> None:
    assert isinstance(contextual.classifier, dc.ContextClassifier)
    context = contextual.classifier.context
    assert context.scales == contextual.layout.all_scales
    assert len(context.names) == 3 * len(context.scales)
    stored = dc.context_metadata(contextual.classifier)["dead_context"]
    assert stored["scales"] == list(context.scales)
    assert stored["columns"] == len(context.names)
    assert stored["iterations"] == int(context.estimator.n_iter_)
    metadata = model_metadata(contextual, b"", "run", {})
    assert metadata["dead_context"] == stored
    assert metadata["dead_ball_features"] == len(contextual.classifier.dead_columns)
    assert context.estimator.n_features_in_ == (len(contextual.classifier.dead_columns)
                                                + len(context.names))


def test_context_classifier_refuses_a_match_without_its_segments(
        contextual: LearnedModel, train: list[TrainingExample]) -> None:
    matrix = ff.feature_matrix(train[0].base[0], contextual.layout)
    with pytest.raises(ValueError, match="block segments"):
        contextual.classifier.dead_logit(matrix)
    with pytest.raises(ValueError, match="block segments"):
        contextual.classifier.log_posterior(matrix)
    with pytest.raises(ValueError, match="block segments"):
        adapted_dead_logit(contextual.classifier, matrix, train[0].blocks, 0, 2)
    segments_of_match = train[0].base[0].segments
    posterior = contextual.classifier.log_posterior(matrix, segments=segments_of_match)
    assert posterior.shape == (len(matrix), 3)


def test_stage_two_logit_depends_on_the_block_layout(
        contextual: LearnedModel, train: list[TrainingExample]) -> None:
    base = train[0].base[0]
    matrix = ff.feature_matrix(base, contextual.layout)
    whole = contextual.classifier.dead_logit(matrix, segments(0, len(matrix)))
    split = contextual.classifier.dead_logit(matrix, base.segments)
    assert whole.shape == split.shape
    assert not np.array_equal(whole, split)


def test_adaptation_starts_from_the_stage_two_logit(
        contextual: LearnedModel, train: list[TrainingExample]) -> None:
    example, classifier = train[0], contextual.classifier
    assert isinstance(classifier, dc.ContextClassifier)
    base = example.base[0]
    matrix = ff.feature_matrix(base, contextual.layout)
    stage_two = classifier.dead_logit(matrix, base.segments)
    assert not np.array_equal(stage_two, classifier.stage_one_logit(matrix))
    unchanged, report = adapted_dead_logit(classifier, matrix, example.blocks, 0, 0,
                                           segments=base.segments)
    np.testing.assert_array_equal(unchanged, stage_two)
    assert not report.shares
    adapted, report = adapted_dead_logit(classifier, matrix, example.blocks, 0, 2,
                                         segments=base.segments)
    assert report.shares and not np.array_equal(adapted, stage_two)
    stage_one = FrameClassifier(**{item.name: getattr(classifier, item.name)
                                   for item in fields(FrameClassifier)})
    from_stage_one, _ = adapted_dead_logit(stage_one, matrix, example.blocks, 0, 2)
    assert not np.array_equal(adapted, from_stage_one)
    emission, _ = log_emission(contextual, base, example.blocks)
    np.testing.assert_array_equal(emission, classifier.log_emission(
        matrix, adapted_dead_logit(classifier, matrix, example.blocks, 0,
                                   contextual.dead_ball.adaptation_iterations,
                                   segments=base.segments)[0]))


def test_model_with_context_predicts_from_raw_tracking(
        contextual: LearnedModel, tmp_path: Path) -> None:
    match = make_matches((0,))[0]
    codes = predict_codes(contextual, to_tracking(match), match.pitch, match.fps, match.blocks)
    assert len(codes) == match.n_frames
    assert set(np.unique(codes).tolist()) <= {0, 1, 2}
    path = tmp_path / "model.pkl"
    save_model(path, contextual, "run")
    reloaded = load_model(path)
    assert isinstance(reloaded.classifier, dc.ContextClassifier)
    np.testing.assert_array_equal(
        codes, predict_codes(reloaded, to_tracking(match), match.pitch, match.fps, match.blocks))


def test_context_fitting_is_deterministic(
        train: list[TrainingExample], contextual: LearnedModel) -> None:
    again = fit_model(train, SETTINGS, FRAME_STEP, DeadBallSettings("players", 2), "touch",
                      switch_prior=False, fitting=CONTEXT)
    assert pickle.dumps(again) == pickle.dumps(contextual)
