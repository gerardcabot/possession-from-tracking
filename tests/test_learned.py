import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from learned_synthetic import possession_match
from synthetic import make_features

from possession import run
from possession.cleaning import CleaningParams
from possession.data import AWAY, HOME
from possession.external.predict import to_tracking
from possession.external.sources import MatchFiles
from possession.learned import framefeatures as ff
from possession.learned import hsmm, train
from possession.learned.adapt import SAFETY_BOUND
from possession.learned.classifier import TrainingRows, fit_classifier, match_weights
from possession.learned.external import (
    fold_path,
    labelled_codes,
    model_for,
    training_example,
)
from possession.learned.model import (
    UNLABELLED,
    DeadBallSettings,
    DecoderSettings,
    fit_model,
    load_model,
    model_bytes,
    predict_codes,
    read_metadata,
    save_model,
    swap_teams,
)
from possession.paths import DEFAULT_MODEL_PATH, PROJECT_ROOT, cli_path
from possession.run import RunConfig, produce

NAN = np.nan
SETTINGS = DecoderSettings(emission_weight=0.5, stride=5)
RUN_ID = "run"
PARAMS = CleaningParams(ball_max_speed_mps=40.0, ball_deviation_m=1.6,
                        player_max_speed_mps=12.0, player_deviation_m=0.3,
                        duplicate_distance_m=0.1, max_gap_frames=8, ball_half_span=1,
                        player_half_span=1)


def inputs_for(n_frames: int, blocks: ff.Blocks | None = None,
               **overrides: object) -> ff.MatchInputs:
    blocks = [(1, 0, n_frames - 1)] if blocks is None else blocks
    features = make_features(n_frames, **overrides)
    segments = ff.block_segments(n_frames, blocks)
    return ff.MatchInputs(features=features, spread=np.ones((n_frames, 2, 2)),
                          segments=segments, params=PARAMS,
                          touch=ff.touch_matrix(features, segments, PARAMS.ball_half_span))


def test_directional_nanmean_looks_only_one_way() -> None:
    values = np.array([1.0, NAN, 3.0, 5.0])
    np.testing.assert_allclose(ff.directional_nanmean(values, 2, forward=False),
                               [1.0, 1.0, 3.0, 4.0])
    np.testing.assert_allclose(ff.directional_nanmean(values, 2, forward=True),
                               [1.0, 3.0, 4.0, 5.0])


def test_run_lengths_count_back_and_forward() -> None:
    values = np.array([1, 1, 0, -1, -1, -1])
    assert ff.run_length_back(values).tolist() == [1, 2, 1, 1, 2, 3]
    assert ff.run_length_forward(values).tolist() == [2, 1, 1, 3, 2, 1]


def test_rolling_scales_are_mean_run_lengths_of_possession_and_dead_ball() -> None:
    lengths = np.array([10, 30, 50, 4, 8])
    states = np.array([HOME, AWAY, HOME, 2, 2])
    assert ff.rolling_scales(lengths, states, np.ones(5)) == (6, 30)
    assert ff.rolling_scales(lengths, states, np.array([1, 1, 2, 1, 1])) == (6, 35)
    assert ff.rolling_scales(np.array([4, 4]), np.array([HOME, AWAY]), np.ones(2)) == (4,)


def test_gap_columns_measure_time_around_missing_ball() -> None:
    ball = np.full((8, 2), 50.0)
    ball[2:5] = NAN
    nearest = np.column_stack([np.full(8, 1.0), np.full(8, 4.0)])
    nearest[5:] = nearest[5:, ::-1]
    base = ff.base_columns(inputs_for(8, ball=ball, nearest=nearest), HOME)
    fps = 25.0
    np.testing.assert_allclose(base["since_seen"] * fps, [0, 0, 1, 2, 3, 0, 0, 0])
    np.testing.assert_allclose(base["until_seen"] * fps, [0, 0, 3, 2, 1, 0, 0, 0])
    np.testing.assert_allclose(base["gap_length"][2:5] * fps, [4, 4, 4])
    assert base["entry_near_diff"][3] == pytest.approx(-3.0)
    assert base["exit_near_diff"][3] == pytest.approx(3.0)


def test_block_segments_partition_frames() -> None:
    assert ff.block_segments(10, [(1, 2, 4), (2, 6, 9)]) == ((0, 2), (2, 5), (5, 6), (6, 10))
    assert ff.block_segments(0, []) == ()


def test_gap_dwell_and_rolling_features_stop_at_block_boundaries() -> None:
    ball = np.full((8, 2), 50.0)
    ball[3:6] = NAN
    nearest = np.column_stack([np.full(8, 1.0), np.full(8, 4.0)])
    inputs = inputs_for(8, blocks=[(1, 0, 3), (2, 4, 7)], ball=ball, nearest=nearest)
    base = ff.base_columns(inputs, HOME)
    fps = 25.0
    np.testing.assert_allclose(base["since_seen"] * fps, [0, 0, 0, 1, NAN, NAN, 0, 0])
    np.testing.assert_allclose(base["until_seen"] * fps, [0, 0, 0, NAN, 2, 1, 0, 0])
    assert np.isnan(base["exit_near_diff"][3]) and np.isnan(base["entry_near_diff"][4])
    np.testing.assert_allclose(base["dwell_back"] * fps, [1, 2, 3, 4, 1, 2, 3, 4])
    np.testing.assert_allclose(base["dwell_forward"] * fps, [4, 3, 2, 1, 4, 3, 2, 1])
    features = ff.base_features(inputs, HOME)
    matrix = ff.feature_matrix(features, ff.FeatureLayout(scales=(8,)))
    past = ff.feature_names(features, ff.FeatureLayout(scales=(8,))).index("ball_present_past_8")
    np.testing.assert_allclose(matrix[:, past], [1, 1, 1, 0.75, 0, 0, 1 / 3, 0.5])


def test_dwell_is_signed_by_nearest_team() -> None:
    nearest = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 1.0], [np.inf, np.inf]])
    base = ff.base_columns(inputs_for(4, nearest=nearest), HOME)
    np.testing.assert_allclose(base["dwell_back"] * 25.0, [1, 2, -1, 0])
    np.testing.assert_allclose(base["dwell_forward"] * 25.0, [2, 1, -1, 0])


def test_away_perspective_mirrors_home_perspective() -> None:
    ball = np.column_stack([np.linspace(20, 80, 6), np.full(6, 30.0)])
    nearest = np.column_stack([np.full(6, 1.0), np.full(6, 3.0)])
    inputs = inputs_for(6, ball=ball, nearest=nearest)
    home, away = ff.base_columns(inputs, HOME), ff.base_columns(inputs, AWAY)
    assert set(home) == set(away)
    np.testing.assert_allclose(away["ball_x"], -home["ball_x"])
    np.testing.assert_allclose(away["near_diff"], -home["near_diff"])
    np.testing.assert_allclose(away["nearest_own"], home["nearest_other"])
    np.testing.assert_allclose(away["dwell_back"], -home["dwell_back"])


def test_feature_matrix_matches_names() -> None:
    inputs = inputs_for(30)
    base = ff.base_features(inputs, HOME)
    layout = ff.FeatureLayout(scales=(2, 5), short_scales=(3,), touch=True)
    matrix = ff.feature_matrix(base, layout)
    names = ff.feature_names(base, layout)
    assert matrix.shape == (30, len(names))
    assert matrix.dtype == np.float32
    assert len(set(names)) == len(names)
    assert set(base.names) <= set(names)
    columns = ff.base_columns(inputs, HOME)
    for name in base.names:
        np.testing.assert_array_equal(matrix[:, names.index(name)], columns[name])
    keep = np.arange(30) % 4 == 0
    np.testing.assert_array_equal(ff.feature_matrix(base, layout, keep), matrix[keep])


def test_durations_are_normalised_with_a_geometric_tail() -> None:
    lengths = np.array([90, 100, 100, 110, 120, 95, 105])
    durations = hsmm.fit_durations(lengths, np.ones(len(lengths)))
    assert len(durations.body) == 120
    assert durations.body.sum() + durations.tail_mass == pytest.approx(1.0)
    assert 90 <= np.argmax(durations.body) + 1 <= 110
    assert durations.tail_mass > 0 and 0 < durations.hazard < 1
    pmf = hsmm.frame_pmf(durations, 5000)
    assert pmf.sum() == pytest.approx(1.0)
    assert (pmf[120:] > 0).all() and (np.diff(pmf[120:]) < 0).all()
    point = hsmm.fit_durations(np.array([7, 7]), np.ones(2))
    assert hsmm.frame_pmf(point, 1000).sum() == pytest.approx(1.0)
    assert point.hazard == pytest.approx(1 / 7)


def test_segment_model_learns_transitions_and_initial_state() -> None:
    sequences = [np.array([0, 0, 2, 2, 2, 1, 1, 0]), np.array([2, 2, 0, 0, 1])]
    model = hsmm.fit_segment_model([sequences])
    transition = np.exp(model.log_transition)
    np.testing.assert_allclose(transition[0], [0.0, 0.5, 0.5])
    np.testing.assert_allclose(transition[2], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(np.exp(model.log_initial), [0.5, 0.0, 0.5])
    assert [len(durations.body) for durations in model.durations] == [2, 2, 3]


def test_transition_counts_tolerate_matches_without_sequences() -> None:
    counts = hsmm.transition_counts([[], [np.array([0, 1, 1])]])
    assert counts.sum() == pytest.approx(1.0)
    assert hsmm.transition_counts([[]]).sum() == 0.0


def extended(table: np.ndarray, log_ratio: float, n_steps: int) -> np.ndarray:
    beyond = table[-1] + log_ratio * np.arange(1, n_steps - len(table) + 1)
    return np.concatenate([table, beyond])


def test_step_durations_sum_frame_pmf_per_step() -> None:
    durations = hsmm.Durations(body=np.array([0.1, 0.2, 0.3]), tail_mass=0.4, hazard=0.5)
    steps = hsmm.step_durations(durations, 2)
    np.testing.assert_allclose(np.exp(steps.log_pmf), [0.1 + 0.2, 0.3 + 0.2, 0.1 + 0.05])
    pmf = np.exp(extended(steps.log_pmf, steps.log_ratio, 60))
    assert pmf.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(np.exp(extended(steps.log_survival, steps.log_ratio, 60)),
                               np.cumsum(pmf[::-1])[::-1], atol=1e-12)
    assert np.exp(extended(steps.log_first, steps.log_ratio, 60)).sum() == pytest.approx(1.0)


def known_sequence(rng: np.random.Generator, runs: int) -> np.ndarray:
    codes, state = [], 2
    for _ in range(runs):
        state = int(rng.choice([value for value in range(3) if value != state]))
        codes += [state] * int(rng.integers(80, 160))
    return np.array(codes)


def noisy_emission(codes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    emission = rng.normal(scale=1.0, size=(len(codes), 3))
    emission[np.arange(len(codes)), codes] += 1.0
    return emission


def test_decoder_recovers_known_sequence_from_noisy_emissions() -> None:
    rng = np.random.default_rng(1)
    model = hsmm.fit_segment_model([[known_sequence(rng, 60) for _ in range(3)]])
    truth = known_sequence(rng, 30)
    codes = hsmm.decode_block(noisy_emission(truth, rng), model, 5)
    assert (codes == truth).mean() > 0.97
    assert np.count_nonzero(np.diff(codes)) == np.count_nonzero(np.diff(truth))


def test_frame_refinement_places_clean_boundaries_exactly() -> None:
    truth = np.array([0] * 103 + [1] * 97 + [2] * 121)
    model = hsmm.fit_segment_model([[truth, truth[::-1]]])
    emission = np.full((len(truth), 3), -1.0)
    emission[np.arange(len(truth)), truth] = 1.0
    assert (hsmm.decode_block(emission, model, 10) == truth).all()


def brute_force_best(emission: np.ndarray, model: hsmm.SegmentModel, stride: int) -> float:
    n_steps, n_states = emission.shape
    steps = [hsmm.step_durations(durations, stride) for durations in model.durations]
    cumulative = np.vstack([np.zeros(n_states), np.cumsum(emission, axis=0)])
    entry = np.full((n_steps + 1, n_states), -np.inf)
    entry[0] = model.log_initial
    best = np.full(n_states, -np.inf)
    for t in range(1, n_steps + 1):
        for state, durations in enumerate(steps):
            table, first = hsmm.duration_tables(durations, t == n_steps)
            table, first = (extended(values, durations.log_ratio, n_steps)
                            for values in (table, first))
            best[state] = max(entry[start, state] + (first if start == 0 else table)[t - start - 1]
                              + cumulative[t, state] - cumulative[start, state]
                              for start in range(t))
        if t < n_steps:
            entry[t] = (best[:, None] + model.log_transition).max(axis=0)
    return float(best.max())


def test_forward_pass_matches_brute_force_over_unbounded_durations() -> None:
    rng = np.random.default_rng(3)
    model = hsmm.fit_segment_model([[np.repeat(rng.permutation(3), rng.integers(1, 9, 3))
                                    for _ in range(30)]])
    for stride, n_steps in ((1, 30), (2, 25), (3, 17)):
        emission = rng.normal(scale=2.0, size=(n_steps, 3))
        steps = [hsmm.step_durations(durations, stride) for durations in model.durations]
        best, _, _ = hsmm.forward_pass(emission, steps, model)
        assert best[n_steps].max() == pytest.approx(brute_force_best(emission, model, stride))


@pytest.mark.parametrize("stride", [1, 5])
def test_decoder_outputs_runs_longer_than_any_training_run(stride: int) -> None:
    rng = np.random.default_rng(0)
    model = hsmm.fit_segment_model([[known_sequence(rng, 30) for _ in range(2)]])
    longest = max(len(durations.body) for durations in model.durations)
    n_frames = 4 * longest + 3
    state = int(np.argmax(model.log_initial))
    emission = np.full((n_frames, 3), -2.0)
    emission[:, state] = 2.0
    codes = hsmm.decode_block(emission, model, stride)
    assert (codes == state).all()


def test_decoder_handles_a_final_one_frame_step() -> None:
    stride = 5
    truth = np.array([0] * 15 + [1] * 5 + [2] * 1)
    model = hsmm.fit_segment_model([[np.repeat([0, 1, 2, 0, 2, 1], [15, 5, 1, 12, 3, 7])]])
    emission = np.full((len(truth), 3), -2.0)
    emission[np.arange(len(truth)), truth] = 2.0
    assert len(truth) % stride == 1
    assert (hsmm.decode_block(emission, model, stride) == truth).all()


def test_refine_boundaries_never_uses_an_empty_window() -> None:
    emission = np.eye(3)[[0] * 20 + [1]]
    cumulative = np.vstack([np.zeros(3), np.cumsum(emission, axis=0)])
    assert hsmm.refine_boundaries([20, 21], [0, 1, 2], cumulative, 5) == [20, 21]


def separable_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = rng.integers(0, 3, size=600)
    matrix = rng.normal(size=(600, 2)) + np.eye(3)[labels][:, :2] * 4
    return matrix.astype(np.float32), labels


def rows_for(matrix: np.ndarray, labels: np.ndarray, timed: np.ndarray | None = None,
             match: np.ndarray | None = None) -> TrainingRows:
    return TrainingRows(matrix=matrix, codes=labels,
                        timed=np.ones(len(labels), dtype=bool) if timed is None else timed,
                        match=np.zeros(len(labels), dtype=np.int64) if match is None else match)


def test_classifier_log_posterior_is_normalised_and_accurate() -> None:
    matrix, labels = separable_data(np.random.default_rng(0))
    classifier = fit_classifier(rows_for(matrix, labels), ["a", "b"], seed=0)
    posterior = classifier.log_posterior(matrix)
    np.testing.assert_allclose(np.exp(posterior).sum(axis=1), 1.0, rtol=1e-6)
    assert (posterior.argmax(axis=1) == labels).mean() > 0.9
    np.testing.assert_allclose(classifier.log_emission(matrix), posterior - classifier.log_prior)
    np.testing.assert_allclose(np.exp(classifier.log_prior), np.bincount(labels) / len(labels))
    with pytest.raises(ValueError, match="features"):
        classifier.log_posterior(matrix[:, :1])


def test_classifier_requires_every_state() -> None:
    matrix, labels = separable_data(np.random.default_rng(0))
    with pytest.raises(ValueError, match="dead-ball"):
        fit_classifier(rows_for(matrix[labels < 2], labels[labels < 2]), ["a", "b"], seed=0)
    with pytest.raises(ValueError, match="team"):
        fit_classifier(rows_for(matrix[labels > 0], labels[labels > 0]), ["a", "b"], seed=0)
    with pytest.raises(ValueError, match="dead-ball labels"):
        fit_classifier(rows_for(matrix, labels, timed=np.zeros(len(labels), dtype=bool)),
                       ["a", "b"], seed=0)


def test_matches_without_dead_labels_only_train_the_team_model() -> None:
    matrix, labels = separable_data(np.random.default_rng(0))
    timed = rows_for(matrix, labels)
    extra, extra_labels = separable_data(np.random.default_rng(1))
    alive = extra_labels < 2
    mixed = rows_for(np.vstack([matrix, extra[alive]]),
                     np.concatenate([labels, extra_labels[alive]]),
                     timed=np.concatenate([np.ones(len(labels), bool), np.zeros(alive.sum(), bool)]),
                     match=np.concatenate([np.zeros(len(labels), np.int64),
                                           np.ones(alive.sum(), np.int64)]))
    alone, both = fit_classifier(timed, ["a", "b"], 0), fit_classifier(mixed, ["a", "b"], 0)
    np.testing.assert_array_equal(alone.dead.decision_function(extra),
                                  both.dead.decision_function(extra))
    assert not np.array_equal(alone.side.decision_function(extra),
                              both.side.decision_function(extra))
    assert alone.log_prior[2] == pytest.approx(both.log_prior[2])


def test_match_weights_give_every_match_the_same_total() -> None:
    match = np.array([0, 0, 0, 1, 2, 2])
    weights = match_weights(match)
    assert weights.mean() == pytest.approx(1.0)
    totals = np.bincount(match, weights=weights)
    np.testing.assert_allclose(totals, totals[0])


def test_segment_model_weights_every_match_equally() -> None:
    long_match = [np.array([0, 2] * 20)]
    short_match = [np.array([1, 0])]
    model = hsmm.fit_segment_model([long_match, short_match])
    transition = np.exp(model.log_transition)
    np.testing.assert_allclose(transition[0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(np.exp(model.log_initial), [0.5, 0.5, 0.0])
    pooled = hsmm.fit_segment_model([long_match + short_match])
    assert np.exp(pooled.log_initial)[0] == pytest.approx(0.5)
    lengths, states, weights = hsmm.weighted_runs([long_match, short_match])
    assert weights[states == 1].sum() == pytest.approx(0.5)


def synthetic_examples() -> list:
    return [training_example(possession_match(seed=seed), PARAMS) for seed in range(2)]


def test_skillcorner_frames_without_a_team_stay_unlabelled() -> None:
    match = replace(possession_match(n_per_period=300), source="skillcorner")
    labels = match.labels.copy()
    labels[labels == "D"] = "X"
    codes = labelled_codes(replace(match, labels=labels))
    assert (codes[labels == "X"] == UNLABELLED).all()
    assert set(np.unique(codes).tolist()) == {UNLABELLED, 0, 1}
    assert not training_example(replace(match, labels=labels), PARAMS).dead_labels


def test_labelled_codes_mark_frames_outside_periods() -> None:
    match = possession_match(n_per_period=300)
    codes = labelled_codes(match)
    assert (codes != UNLABELLED).all()
    assert set(np.unique(codes).tolist()) == {0, 1, 2}


def test_swap_teams_exchanges_home_and_away_only() -> None:
    assert swap_teams(np.array([0, 1, 2, -1])).tolist() == [1, 0, 2, -1]


def test_training_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    examples = synthetic_examples()
    first, second = fit_model(examples, SETTINGS), fit_model(examples, SETTINGS)
    assert model_bytes(first) == model_bytes(second)
    path = tmp_path / "model.pkl"
    digest = save_model(path, first, RUN_ID)
    assert json.loads(path.with_suffix(".json").read_text())["sha256"] == digest
    matrix = np.zeros((4, len(first.classifier.feature_names)), dtype=np.float32)
    assert read_metadata(path)["iterations"] == list(first.classifier.iterations)
    np.testing.assert_array_equal(load_model(path).classifier.log_posterior(matrix),
                                  first.classifier.log_posterior(matrix))


def test_model_carries_the_cleaning_parameters_it_was_trained_with(tmp_path: Path) -> None:
    examples = synthetic_examples()
    model = fit_model(examples, SETTINGS)
    assert model.params == PARAMS
    path = tmp_path / "model.pkl"
    save_model(path, model, RUN_ID)
    assert read_metadata(path)["cleaning"] == PARAMS.to_dict()
    assert load_model(path).params == PARAMS
    other = replace(examples[1], params=replace(PARAMS, max_gap_frames=3))
    with pytest.raises(ValueError, match="parameter sets"):
        fit_model([examples[0], other], SETTINGS)


def test_load_rejects_a_modified_model(tmp_path: Path) -> None:
    path = tmp_path / "model.pkl"
    save_model(path, fit_model(synthetic_examples(), SETTINGS), RUN_ID)
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="does not match"):
        load_model(path)


def test_learned_model_labels_held_out_synthetic_match() -> None:
    model = fit_model(synthetic_examples(), SETTINGS)
    match = possession_match(seed=5)
    codes = predict_codes(model, to_tracking(match), match.pitch, match.fps, match.blocks)
    assert (codes == labelled_codes(match)).mean() > 0.8


def test_adaptation_recovers_dead_ball_from_the_target_observations(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    examples = synthetic_examples()
    players = fit_model(examples, SETTINGS, dead_ball=DeadBallSettings(features="players"))
    adapted = replace(players, dead_ball=DeadBallSettings("players", 5))
    match = possession_match(seed=5)
    truth = labelled_codes(match)
    accuracy = {name: (predict_codes(model, to_tracking(match), match.pitch, match.fps,
                                     match.blocks) == truth).mean()
                for name, model in (("players", players), ("adapted", adapted))}
    assert accuracy["adapted"] > accuracy["players"]
    logged = capsys.readouterr().err
    assert "match: adaptation iterations" in logged and "converged" in logged
    path = tmp_path / "model.pkl"
    save_model(path, adapted, RUN_ID)
    assert read_metadata(path)["dead_ball"] == {"features": "players",
                                                "adaptation_iterations": 5}
    assert load_model(path).dead_ball == adapted.dead_ball


def test_run_learned_writes_valid_submission(tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "model.pkl"
    save_model(path, fit_model(synthetic_examples(), SETTINGS), RUN_ID)
    match = possession_match(seed=7)
    graded = np.zeros(match.n_frames, dtype=bool)
    graded[100:match.n_frames - 100] = True
    blocks = [(1, 100, 1499), (2, 1500, match.n_frames - 101)]
    monkeypatch.setattr(run, "load_tracking", lambda use_cache=True: to_tracking(match))
    monkeypatch.setattr(run, "graded_blocks", lambda: blocks)
    monkeypatch.setattr(run, "graded_mask", lambda n: graded)
    monkeypatch.setattr(run, "pitch_size", lambda: match.pitch)
    monkeypatch.setattr(run, "fps", lambda: match.fps)
    outputs = []
    for index in range(2):
        config = RunConfig(method="learned", finalize=True, model=str(path))
        outputs.append(produce(config, tmp_path / f"learned{index}.csv").read_text())
    assert outputs[0] == outputs[1]
    rows = [line.split(",") for line in outputs[0].splitlines()[1:]]
    assert [int(frame) for frame, _ in rows] == list(range(match.n_frames))
    labels = np.array([label for _, label in rows])
    assert set(labels[graded].tolist()) <= {"H", "A", "D"}
    assert (labels[~graded] == "X").all()


VARIANT = "dfl"


def scored(spell: float, low: float | None = None, high: float | None = None) -> train.Scored:
    raw = {"spell_f1_mean": spell, "macro_f1": spell, "n_pred_spells": 1.0, "n_gt_spells": 1.0}
    if low is not None:
        raw.update(spell_bootstrap_low=low, spell_bootstrap_high=high)
    return train.Scored(raw=raw, fair=None)


def test_folds_keep_their_own_nested_settings(tmp_path: Path) -> None:
    examples = synthetic_examples()
    grid = [SETTINGS, DecoderSettings(emission_weight=0.1, stride=10)]
    folds = [train.Fold(key=example.name, source="dfl", model=fit_model([other], grid[0]),
                        scores=[])
             for example, other in zip(examples, examples[::-1])]
    result = train.VariantResult(name=VARIANT, members=[], folds=folds, grid=grid,
                                 final_grid=grid, fold_grids=[grid, grid], choices=[1, 0],
                                 selected=0, model=folds[0].model, outside={})
    out = tmp_path / "model.pkl"
    train.save_folds([result], out, RUN_ID)
    assert load_model(fold_path(out, VARIANT, examples[0].name)).settings == grid[1]
    assert load_model(fold_path(out, VARIANT, examples[1].name)).settings == grid[0]
    assert read_metadata(fold_path(out, VARIANT, examples[0].name))["stride"] == 10
    assert read_metadata(fold_path(out, VARIANT, examples[0].name))["variant"] == VARIANT


def test_validate_refuses_folds_from_another_training_run(tmp_path: Path) -> None:
    examples = synthetic_examples()
    out = tmp_path / "model.pkl"
    final = fit_model(examples, SETTINGS)
    save_model(out, final, RUN_ID)
    match = possession_match(seed=0)
    with pytest.raises(ValueError, match="variant"):
        model_for(match, final, out)
    save_model(out, final, RUN_ID, {"variant": VARIANT})
    fold = fit_model(examples[1:], SETTINGS)
    path = fold_path(out, VARIANT, examples[0].name)
    save_model(path, fold, "older")
    with pytest.raises(ValueError, match="training run"):
        model_for(match, final, out)
    save_model(path, fold, RUN_ID)
    assert model_for(match, final, out).training_matches == (examples[1].name,)
    save_model(path, final, RUN_ID)
    with pytest.raises(ValueError, match="other matches"):
        model_for(match, final, out)
    other = replace(match, name="unseen")
    assert model_for(other, final, out) is final


def test_selection_averages_sources_before_matches() -> None:
    folds = [train.Fold(key=f"dfl/{index}", source="dfl", model=None,
                        scores=[scored(0.9), scored(0.1)]) for index in range(3)]
    folds.append(train.Fold(key="metrica/game1", source="metrica", model=None,
                            scores=[scored(0.0), scored(0.9)]))
    np.testing.assert_allclose(train.selection_scores(folds)[:, 0], [0.45, 0.5])
    grid = [SETTINGS, SETTINGS]
    assert train.best_setting(train.selection_scores(folds), grid) == 1


def summary(spell: float, low: float, high: float) -> dict[str, float]:
    return {"spell_f1_mean": spell, "spell_bootstrap_low": low, "spell_bootstrap_high": high}


def test_variant_selection_breaks_metrica_ties_with_dfl_then_skillcorner() -> None:
    summaries = {
        "a": {"metrica": summary(0.60, 0.58, 0.62), "dfl": summary(0.50, 0.48, 0.52),
              "skillcorner": summary(0.3, 0.2, 0.4)},
        "b": {"metrica": summary(0.59, 0.57, 0.61), "dfl": summary(0.70, 0.68, 0.72),
              "skillcorner": summary(0.3, 0.2, 0.4)},
        "c": {"metrica": summary(0.50, 0.48, 0.52), "dfl": summary(0.90, 0.88, 0.92),
              "skillcorner": summary(0.9, 0.8, 1.0)},
    }
    assert train.select_variant(summaries) == "b"
    summaries["b"]["dfl"] = summary(0.51, 0.49, 0.53)
    summaries["a"]["skillcorner"] = summary(0.4, 0.3, 0.5)
    assert train.select_variant(summaries) == "a"
    summaries["c"]["metrica"] = summary(0.70, 0.69, 0.71)
    assert train.select_variant(summaries) == "c"


def test_variant_selection_survives_matches_without_spells() -> None:
    nan = float("nan")
    rows = [{"source": "metrica", "raw": {**scored(0.6, 0.5, 0.7).raw}, "fair": None},
            {"source": "metrica", "raw": {**scored(nan, nan, nan).raw}, "fair": None}]
    aggregated = train.source_summary(rows, "metrica", "raw")
    assert aggregated["n_matches"] == 1
    assert aggregated["n_without_spells"] == 1
    assert aggregated["spell_bootstrap_low"] == pytest.approx(0.5)
    assert train.source_summary(rows[1:], "metrica", "raw") is None
    broken = {"a": {"metrica": summary(0.5, nan, nan), "dfl": None, "skillcorner": None}}
    assert train.tied_with_best(broken, ["a"], "metrica") == ["a"]
    assert train.select_variant(broken) == "a"


def test_variant_selection_uses_the_chosen_metrica_score() -> None:
    summaries = {
        "a": {"metrica": summary(0.6, 0.58, 0.62), "metrica_raw": summary(0.4, 0.38, 0.42),
              "dfl": None, "skillcorner": None},
        "b": {"metrica": summary(0.5, 0.48, 0.52), "metrica_raw": summary(0.5, 0.48, 0.52),
              "dfl": None, "skillcorner": None},
    }
    assert train.select_variant(summaries) == "a"
    assert train.select_variant(summaries, train.SELECTION_SCORES["raw"]) == "b"


def test_variants_parse_in_canonical_order() -> None:
    assert train.parse_variants("all") == list(train.VARIANTS)
    assert train.parse_variants("metrica,dfl") == ["dfl", "metrica"]
    with pytest.raises(ValueError, match="unknown variants"):
        train.parse_variants("dfl,typo")


def test_folds_are_reused_only_for_identical_inputs(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = [train.Loaded(example=example, match=train.labels_only(possession_match(seed=seed)),
                           data_key=f"data{seed}")
              for seed, example in enumerate(synthetic_examples())]
    context = train.Context(seed=0, frame_step=1, grid=[SETTINGS], loaded=loaded,
                            out=tmp_path / "model.pkl", fingerprint="code")
    held = [loaded[0]]
    (first,) = train.run_fold(context, VARIANT, loaded, held, loaded[0].key)
    monkeypatch.setattr(train, "fit_model", lambda *args: pytest.fail("refitted"))
    (again,) = train.run_fold(context, VARIANT, loaded, held, loaded[0].key)
    assert again.scores == first.scores
    matrix = np.zeros((4, len(first.model.classifier.feature_names)), dtype=np.float32)
    np.testing.assert_array_equal(again.model.classifier.log_posterior(matrix),
                                  first.model.classifier.log_posterior(matrix))
    changed = replace(context, fingerprint="other code")
    with pytest.raises(pytest.fail.Exception):
        train.run_fold(changed, VARIANT, loaded, held, loaded[0].key)
    players = replace(context, dead_ball=DeadBallSettings(features="players"))
    with pytest.raises(pytest.fail.Exception):
        train.run_fold(players, VARIANT, loaded, held, loaded[0].key)


def test_dead_ball_settings_are_checked() -> None:
    assert train.dead_ball_settings("players", True) == DeadBallSettings("players", SAFETY_BOUND)
    assert train.dead_ball_settings("all", False) == DeadBallSettings()
    with pytest.raises(ValueError, match="needs dead_features 'players'"):
        train.dead_ball_settings("all", True)
    with pytest.raises(ValueError, match=">= 0"):
        DeadBallSettings("players", -1)
    with pytest.raises(ValueError, match="needs dead_features 'players'"):
        DeadBallSettings("all", 1)
    with pytest.raises(ValueError, match="feature set"):
        DeadBallSettings("ball")


def test_training_run_id_depends_on_config_and_model() -> None:
    config = {"seed": 0, "matches": ["a", "b"]}
    assert train.training_run_id(config, "x") == train.training_run_id(dict(config), "x")
    assert train.training_run_id(config, "x") != train.training_run_id(config, "y")
    assert train.training_run_id(config, "x") != train.training_run_id({**config, "seed": 1}, "x")


def test_partial_training_needs_an_explicit_output(tmp_path: Path) -> None:
    assert train.output_path(None, subset=False) == DEFAULT_MODEL_PATH
    with pytest.raises(ValueError, match="explicit"):
        train.output_path(None, subset=True)
    with pytest.raises(ValueError, match="explicit"):
        train.output_path(DEFAULT_MODEL_PATH, subset=True)
    assert train.output_path(tmp_path / "m.pkl", subset=True) == tmp_path / "m.pkl"


def test_variants_are_checked_before_any_fitting() -> None:
    train.check_variants(["dfl", "dfl+metrica"], ["dfl", "dfl", "metrica", "metrica"])
    with pytest.raises(ValueError, match="metrica matches.*found 1"):
        train.check_variants(["dfl", "dfl+metrica", "metrica"], ["dfl", "dfl", "metrica"])


def test_variants_subset_of_the_default_is_partial() -> None:
    assert not train.partial_variants(train.parse_variants(train.ALL_VARIANTS))
    assert not train.partial_variants(list(train.DEFAULT_VARIANTS))
    assert train.partial_variants(["dfl"])
    assert "dfl+metrica+skillcorner_ha" not in train.DEFAULT_VARIANTS


def test_unknown_training_matches_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    available = {"dfl": [MatchFiles("dfl", name, ()) for name in ("J03WMX", "J03WN1")],
                 "metrica": [MatchFiles("metrica", "game1", ())]}
    monkeypatch.setattr(train, "discover", lambda source: available.get(source, []))
    assert [files.name for files in train.selected_files({"J03WN1"})] == ["J03WN1"]
    assert [files.name for files in train.selected_files({"metrica/game1"})] == ["game1"]
    assert len(train.selected_files(None)) == 3
    assert [files.name for files in train.selected_files(None, limit=1)] == ["J03WMX", "game1"]
    with pytest.raises(ValueError, match="unknown"):
        train.selected_files({"J03WN1", "typo"})


def test_ambiguous_bare_match_names_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    available = {"dfl": [MatchFiles("dfl", "game1", ())],
                 "metrica": [MatchFiles("metrica", "game1", ())]}
    monkeypatch.setattr(train, "discover", lambda source: available.get(source, []))
    with pytest.raises(ValueError, match="ambiguous"):
        train.selected_files({"game1"})
    assert [files.source for files in train.selected_files({"dfl/game1"})] == ["dfl"]


def test_cli_paths_resolve_against_the_working_directory(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli_path("models/m.pkl") == tmp_path / "models" / "m.pkl"
    with pytest.raises(argparse.ArgumentTypeError):
        cli_path("")
    assert run.model_path(RunConfig(model="models/learned/model.pkl")) == DEFAULT_MODEL_PATH
    assert run.model_path(RunConfig(model=str(tmp_path / "m.pkl"))) == tmp_path / "m.pkl"


def test_baseline_and_rules_do_not_import_the_learned_stack() -> None:
    code = ("import sys, possession.run, possession.external.validate; "
            "print(sorted(name for name in ('sklearn', 'scipy', 'possession.learned') "
            "if name in sys.modules))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            check=True, cwd=PROJECT_ROOT)
    assert result.stdout.strip() == "[]"
