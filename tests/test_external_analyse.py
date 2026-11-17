from dataclasses import replace

import numpy as np
import pytest
from external_synthetic import make_match

from possession.external import analyse, validate
from possession.external import match as external_match
from possession.external.match import ExternalMatch, label_codes
from possession.external.sources import DFL, METRICA, SKILLCORNER
from possession.vendor import possession_metrics as pm

H, A, D = pm.H, pm.A, pm.D


def skillcorner_match(**kwargs: int) -> ExternalMatch:
    return replace(make_match(**kwargs), source=SKILLCORNER)


def codes_of(*pieces: tuple[int, int]) -> np.ndarray:
    return np.concatenate([np.full(length, code, dtype=np.int8) for code, length in pieces])


def runs_for(codes: np.ndarray) -> list[tuple[int, int]]:
    return [(0, len(codes))]


def classify(truth: np.ndarray, predicted: np.ndarray) -> dict[str, int]:
    runs = runs_for(truth)
    return analyse.spell_analysis(truth, predicted, runs)[3]


def test_dead_as_fill_continues_the_last_team_and_keeps_other_frames() -> None:
    match = skillcorner_match(n_per_period=300, periods=1)
    codes = codes_of((H, 100), (D, 20), (H, 80), (A, 60), (D, 40))
    filled = external_match.dead_as_fill(match, codes)
    assert filled.tolist() == codes_of((H, 200), (A, 100)).tolist()


def test_fair_score_removes_the_penalty_for_dead_frames_on_sources_without_dead_labels() -> None:
    match = skillcorner_match(n_per_period=1500)
    labels = match.labels.copy()
    labels[100:140] = "D"
    raw = validate.score_labels(match, labels)
    fair = validate.score_fair(match, labels)
    assert fair["spell_f1_mean"] == pytest.approx(1.0)
    assert raw["spell_f1_mean"] < fair["spell_f1_mean"]
    assert fair["n_pred_spells"] == fair["n_gt_spells"]


def metrica_match() -> ExternalMatch:
    match = make_match(n_per_period=1500)
    labels = match.labels.copy()
    labels[600:640] = "D"
    return replace(match, source=METRICA, labels=labels)


def test_fair_mode_fills_heuristic_dead_labels_in_truth_and_prediction() -> None:
    match = metrica_match()
    truth = external_match.fair_truth(match)
    assert not (truth == D).any()
    assert truth[600:640].tolist() == [truth[599]] * 40
    perfect = validate.score_fair(match, match.labels)
    assert perfect["spell_f1_mean"] == pytest.approx(1.0)
    no_dead = match.labels.copy()
    no_dead[600:640] = no_dead[599]
    assert validate.score_fair(match, no_dead) == perfect
    assert validate.score_labels(match, no_dead)["spell_f1_mean"] < 1.0


def test_fair_truth_is_unchanged_for_sources_without_dead_labels() -> None:
    match = skillcorner_match(n_per_period=1500)
    np.testing.assert_array_equal(external_match.fair_truth(match),
                                  external_match.filled_codes(match))


def test_analyse_fair_mode_fills_heuristic_dead_labels_in_the_reference() -> None:
    match = metrica_match()
    no_dead = match.labels.copy()
    no_dead[600:640] = no_dead[599]
    fair = analyse.analyse_match(match, no_dead, True)
    raw = analyse.analyse_match(match, no_dead, False)
    assert fair.matched.all() and not fair.error.any()
    assert not raw.matched.all()
    assert fair.true_dead_runs == raw.true_dead_runs == 1
    assert analyse.fair_modes(match, True) == [False, True]


def test_fair_score_is_raw_score_when_the_source_has_dead_labels() -> None:
    match = make_match(n_per_period=1500)
    labels = match.labels.copy()
    labels[600:640] = "D"
    labelled = replace(match, source=DFL, labels=labels)
    predicted = labels.copy()
    assert external_match.has_dead_labels(labelled)
    assert validate.score_fair(labelled, predicted) == validate.score_labels(labelled, predicted)


def test_frame_confusion_counts_inside_runs_only() -> None:
    truth = codes_of((H, 10), (A, 10), (D, 10))
    predicted = codes_of((H, 5), (A, 5), (A, 10), (D, 10))
    matrix = analyse.frame_confusion(truth, predicted, [(0, 20)])
    assert matrix.tolist() == [[5, 5, 0], [0, 10, 0], [0, 0, 0]]


def test_merge_across_dead_is_detected() -> None:
    truth = codes_of((H, 100), (D, 20), (H, 100))
    predicted = codes_of((H, 220))
    assert classify(truth, predicted)[analyse.MERGED_DEAD] == 2


def test_merge_across_short_opposite_spell_is_detected() -> None:
    truth = codes_of((H, 100), (A, 10), (H, 100))
    predicted = codes_of((H, 210))
    causes = classify(truth, predicted)
    assert causes[analyse.MERGED_OPPOSITE] == 2
    assert causes[analyse.MERGED_DEAD] == 0


def test_wrong_team_dead_and_fragmented_misses() -> None:
    truth = codes_of((H, 100), (A, 100), (H, 100))
    predicted = codes_of((A, 100), (D, 100), (H, 30), (A, 40), (H, 30))
    causes = classify(truth, predicted)
    assert causes[analyse.WRONG_TEAM] == 1
    assert causes[analyse.PREDICTED_DEAD] == 1
    assert causes[analyse.FRAGMENTED] == 1


def test_matched_spells_are_not_counted_as_misses() -> None:
    truth = codes_of((H, 100), (A, 100))
    matched, n_pred, _, causes = analyse.spell_analysis(truth, truth, runs_for(truth))
    assert matched.tolist() == [True, True]
    assert n_pred == 2
    assert sum(causes.values()) == 0


def test_boundary_offsets_are_signed_and_unmatched_boundaries_are_nan() -> None:
    truth = codes_of((H, 100), (A, 100), (D, 100))
    predicted = codes_of((H, 95), (A, 110), (D, 95))
    gap = np.zeros(len(truth), dtype=np.int64)
    boundaries = analyse.boundary_offsets(truth, predicted, runs_for(truth), 50, gap)
    by_kind = {boundary.kind: boundary.offset for boundary in boundaries}
    assert by_kind == {"H>A": -5.0, "A>D": 5.0}
    far = analyse.boundary_offsets(truth, codes_of((H, 200), (D, 100)), runs_for(truth), 50, gap)
    assert np.isnan({boundary.kind: boundary.offset for boundary in far}["H>A"])


def test_boundary_in_gap_flag_uses_the_ball_gap_at_the_true_transition() -> None:
    truth = codes_of((H, 100), (A, 100))
    gap = np.zeros(200, dtype=np.int64)
    gap[90:110] = 20
    boundaries = analyse.boundary_offsets(truth, truth, runs_for(truth), 50, gap)
    assert [boundary.in_gap for boundary in boundaries] == [True]


def test_gap_lengths_cover_each_gap_frame_with_its_gap_length() -> None:
    visible = np.ones(30, dtype=bool)
    visible[5:8] = False
    visible[20:30] = False
    per_frame = analyse.gap_lengths_per_frame(visible, [(0, 15), (15, 30)])
    assert per_frame[5:8].tolist() == [3, 3, 3]
    assert per_frame[20:30].tolist() == [10] * 10
    assert per_frame[:5].sum() == 0


def test_analyse_match_of_perfect_labels_has_no_misses_and_zero_offsets() -> None:
    match = skillcorner_match(n_per_period=1500)
    result = analyse.analyse_match(match, match.labels, True)
    assert result.matched.all()
    assert all(boundary.offset == 0.0 for boundary in result.boundaries)
    assert not result.error.any()
    assert result.pred_dead_runs == 0
    assert result.confusion.sum() == 2 * 1500
    assert np.diag(result.confusion).sum() == result.confusion.sum()


def test_analyse_match_fair_option_ignores_dead_predictions_for_spells_only() -> None:
    match = skillcorner_match(n_per_period=1500)
    labels = match.labels.copy()
    labels[100:140] = "D"
    fair = analyse.analyse_match(match, labels, True)
    raw = analyse.analyse_match(match, labels, False)
    assert fair.matched.all()
    assert not raw.matched.all()
    assert fair.pred_dead_runs == raw.pred_dead_runs == 1
    assert fair.confusion[H, D] == raw.confusion[H, D] == 40


def test_report_has_every_section() -> None:
    match = skillcorner_match(n_per_period=1500)
    labels = match.labels.copy()
    labels[100:140] = "D"
    text = "\n".join(analyse.report("test", [analyse.analyse_match(match, labels, False)]))
    for fragment in ("frame confusion", "spells at IoU", "missed true spells by cause",
                     "boundary timing", "ball visibility", "recall by true spell duration",
                     "D share true"):
        assert fragment in text


def test_label_codes_roundtrip_used_by_analysis() -> None:
    assert label_codes(np.array(["H", "A", "D"])).tolist() == [H, A, D]


def test_frame_confusion_ignores_unknown_truth_frames() -> None:
    truth = codes_of((pm.X, 10), (H, 10))
    predicted = codes_of((H, 20))
    matrix = analyse.frame_confusion(truth, predicted, [(0, 20)])
    assert matrix.tolist() == [[10, 0, 0], [0, 0, 0], [0, 0, 0]]


def test_dead_boundaries_are_keyed_by_destination_team() -> None:
    truth = codes_of((H, 100), (D, 50), (H, 100))
    wrong_restart = codes_of((H, 100), (D, 50), (A, 100))
    gap = np.zeros(len(truth), dtype=np.int64)
    boundaries = analyse.boundary_offsets(truth, wrong_restart, runs_for(truth), 20, gap)
    by_kind = {boundary.kind: boundary.offset for boundary in boundaries}
    assert by_kind["H>D"] == 0.0
    assert np.isnan(by_kind["D>H"])


def test_boundaries_match_one_to_one() -> None:
    truth = codes_of((H, 100), (A, 5), (H, 5), (A, 100))
    predicted = codes_of((H, 102), (A, 108))
    gap = np.zeros(len(truth), dtype=np.int64)
    boundaries = analyse.boundary_offsets(truth, predicted, runs_for(truth), 10, gap)
    swaps = [boundary for boundary in boundaries if boundary.kind == "H>A"]
    assert len(swaps) == 2
    assert sum(not np.isnan(boundary.offset) for boundary in swaps) == 1


def test_overrun_without_absorbing_another_spell_is_not_a_merge() -> None:
    truth = codes_of((H, 50), (A, 100), (D, 200))
    predicted = codes_of((H, 350))
    causes = classify(truth, predicted)
    assert causes[analyse.OVERRUN] == 1
    assert causes[analyse.MERGED_OTHER] == 0
