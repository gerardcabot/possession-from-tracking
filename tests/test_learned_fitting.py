import sys
from dataclasses import replace

import numpy as np
import pytest
from learned_synthetic import possession_match
from sklearn.ensemble import HistGradientBoostingClassifier

from possession.cleaning import CleaningParams
from possession.learned import classifier as cl
from possession.learned.external import training_example
from possession.learned.model import (
    EVERY_PEAK,
    ClassifierSettings,
    DecoderSettings,
    TrainingExample,
    fit_model,
    model_bytes,
    run_lengths,
    touch_threshold,
    training_rows,
    validation_groups,
)
from possession.learned.train import fitting_record

SETTINGS = DecoderSettings(emission_weight=0.5, stride=5)
PARAMS = CleaningParams(ball_max_speed_mps=40.0, ball_deviation_m=1.6,
                        player_max_speed_mps=12.0, player_deviation_m=0.3,
                        duplicate_distance_m=0.1, max_gap_frames=8, ball_half_span=1,
                        player_half_span=1)


def examples(seeds: tuple[int, ...] = (0, 1, 2)) -> list[TrainingExample]:
    return [training_example(replace(possession_match(seed=seed), name=f"m{seed}"), PARAMS)
            for seed in seeds]


def noisy_rows(n_matches: int = 4, seed: int = 0) -> cl.TrainingRows:
    rng = np.random.default_rng(seed)
    n_rows = 400 * n_matches
    matrix = rng.normal(size=(n_rows, 3)).astype(np.float32)
    codes = np.where(rng.random(n_rows) < 0.3, 2,
                     (matrix[:, 0] + rng.normal(scale=1.5, size=n_rows) > 0).astype(np.int64))
    match = np.repeat(np.arange(n_matches), 400)
    return cl.TrainingRows(matrix=matrix, codes=codes, timed=np.ones(n_rows, dtype=bool),
                           match=match, run_length=np.ones(n_rows),
                           group=match % 2)


def test_spell_weights_give_every_spell_of_a_match_the_same_total() -> None:
    match = np.array([0, 0, 0, 0, 1, 1])
    run_length = np.array([1, 3, 3, 3, 2, 2])
    weights = cl.spell_weights(match, run_length)
    assert weights.mean() == pytest.approx(1.0)
    np.testing.assert_allclose(np.bincount(match, weights=weights), [3.0, 3.0])
    assert weights[0] == pytest.approx(weights[1:4].sum())


def test_run_lengths_restart_at_every_block() -> None:
    codes = np.array([0, 0, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(run_lengths(codes, [(1, 0, 3), (2, 4, 6)]),
                                  [2, 2, 2, 2, 1, 2, 2])


def test_validation_groups_deal_each_source_round_robin_into_k_groups() -> None:
    names = ["dfl/b", "dfl/a", "dfl/c", "metrica/y", "metrica/x"]
    train = [replace(example, name=name) for example, name in zip(examples((0,) * 5), names)]
    groups = validation_groups(train)
    assert groups is not None
    assert groups.tolist() == [1, 0, 0, 1, 0]
    assert validation_groups(train[:1]) is None


def test_validation_groups_count_is_the_smallest_source() -> None:
    names = [f"dfl/{index}" for index in range(5)] + [f"metrica/{index}" for index in range(3)]
    train = [replace(example, name=name) for example, name in zip(examples((0,) * 8), names)]
    groups = validation_groups(train)
    assert groups is not None
    assert groups.tolist() == [0, 1, 2, 0, 1, 0, 1, 2]


def test_grouped_stopping_refits_at_the_mean_stop(capsys: pytest.CaptureFixture[str]) -> None:
    rows = noisy_rows()
    fitted = cl.fit_classifier(rows, ["a", "b", "c"], 0, stopping="team")
    side = rows.codes != 2
    labels, weights = rows.codes[side] == 1, cl.match_weights(rows.match[side])
    stops = [cl.stopped_iterations(rows.matrix[side], labels, weights,
                                   rows.group[side] == group, 0) for group in (0, 1)]
    assert fitted.iterations == (100, round(float(np.mean(stops))))
    assert "grouped early stopping" in capsys.readouterr().err
    both = cl.fit_classifier(rows, ["a", "b", "c"], 0, stopping="both")
    assert both.iterations[0] != 100 and both.iterations[1] == fitted.iterations[1]


def test_stopped_iterations_is_the_best_iteration_not_the_patience_tail() -> None:
    rows = noisy_rows()
    side = rows.codes != 2
    matrix, labels = rows.matrix[side], rows.codes[side] == 1
    weights, held = cl.match_weights(rows.match[side]), rows.group[side] == 0
    stop = cl.stopped_iterations(matrix, labels, weights, held, 0)
    estimator = HistGradientBoostingClassifier(random_state=0, early_stopping=True,
                                               max_iter=sys.maxsize)
    estimator.fit(matrix[~held], labels[~held], sample_weight=weights[~held], X_val=matrix[held],
                  y_val=labels[held], sample_weight_val=weights[held])
    assert stop == int(np.argmax(estimator.validation_score_))
    assert stop == estimator.n_iter_ - estimator.n_iter_no_change
    assert stop < estimator.n_iter_


def test_grouped_stopping_without_groups_falls_back(capsys: pytest.CaptureFixture[str]) -> None:
    rows = replace(noisy_rows(), group=None)
    fitted = cl.fit_classifier(rows, ["a", "b", "c"], 0, stopping="team")
    assert fitted.iterations == (100, 100)
    assert "library default" in capsys.readouterr().err


def test_spell_weights_change_only_the_team_model() -> None:
    rows = noisy_rows()
    rows = replace(rows, run_length=np.where(np.arange(len(rows.codes)) % 2 == 0, 1.0, 50.0))
    plain = cl.fit_classifier(rows, ["a", "b", "c"], 0)
    spell = cl.fit_classifier(rows, ["a", "b", "c"], 0, spell=True)
    np.testing.assert_array_equal(plain.dead.decision_function(rows.matrix),
                                  spell.dead.decision_function(rows.matrix))
    assert not np.array_equal(plain.side.decision_function(rows.matrix),
                              spell.side.decision_function(rows.matrix))
    with pytest.raises(ValueError, match="run length"):
        cl.fit_classifier(replace(rows, run_length=None), ["a", "b", "c"], 0, spell=True)


def test_classifier_settings_reject_unknown_stopping() -> None:
    with pytest.raises(ValueError, match="stopping"):
        ClassifierSettings(stopping="forever")


def test_stopping_is_recorded_per_model(capsys: pytest.CaptureFixture[str]) -> None:
    rows = noisy_rows()
    assert cl.fit_classifier(rows, ["a", "b", "c"], 0).stopping == ("default", "default")
    assert cl.fit_classifier(rows, ["a", "b", "c"], 0, stopping="team").stopping == (
        "default", "grouped")
    assert cl.fit_classifier(rows, ["a", "b", "c"], 0, stopping="both").stopping == (
        "grouped", "grouped")
    fallback = cl.fit_classifier(replace(rows, group=None), ["a", "b", "c"], 0, stopping="both")
    assert fallback.stopping == ("default", "default")
    capsys.readouterr()


def test_default_fitting_leaves_the_run_inputs_unchanged() -> None:
    assert fitting_record(ClassifierSettings()) == {}
    assert fitting_record(ClassifierSettings(every_peak=True)) == {
        "classifier": {"spell_weights": False, "every_peak": True, "stopping": "default"}}


def test_run_lengths_are_built_only_for_spell_weights() -> None:
    train = examples()
    layout = fit_model(train, SETTINGS, 4).layout
    assert training_rows(train, layout, 4).run_length is None
    spell = training_rows(train, layout, 4, spell=True)
    assert spell.run_length is not None and len(spell.run_length) == len(spell.codes)


def test_every_peak_moves_only_the_team_touch_threshold() -> None:
    train = examples()
    youden = fit_model(train, SETTINGS, 4)
    every = fit_model(train, SETTINGS, 4, fitting=ClassifierSettings(every_peak=True))
    assert every.layout.touch_threshold == EVERY_PEAK
    assert youden.layout.touch_threshold == touch_threshold(train).threshold
    assert every.changes is not None and youden.changes is not None
    assert every.changes.threshold == youden.changes.threshold


def test_fitting_settings_are_deterministic() -> None:
    train = examples()
    fitting = ClassifierSettings(spell_weights=True, every_peak=True, stopping="both")
    assert (model_bytes(fit_model(train, SETTINGS, 4, fitting=fitting))
            == model_bytes(fit_model(train, SETTINGS, 4, fitting=fitting)))
