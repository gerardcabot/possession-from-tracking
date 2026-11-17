from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from learned_synthetic import possession_match
from scipy.special import logsumexp

from possession.cleaning import CleaningParams
from possession.data import AWAY, HOME
from possession.intervals import DEAD
from possession.learned import changes, hsmm, model, train
from possession.learned.external import training_example
from possession.learned.model import (
    DecoderSettings,
    Entering,
    decode_emission,
    fit_model,
    for_inference,
)

PARAMS = CleaningParams(ball_max_speed_mps=40.0, ball_deviation_m=1.6,
                        player_max_speed_mps=12.0, player_deviation_m=0.3,
                        duplicate_distance_m=0.1, max_gap_frames=8, ball_half_span=1,
                        player_half_span=1)
SETTINGS = DecoderSettings(emission_weight=0.5, stride=5)


def segment_model() -> hsmm.SegmentModel:
    rng = np.random.default_rng(0)
    sequences = []
    for first in range(3):
        codes, state = [], first
        for _ in range(40):
            codes += [state] * int(rng.integers(20, 200))
            state = int(rng.choice([value for value in range(3) if value != state]))
        sequences.append(np.array(codes))
    return hsmm.fit_segment_model([sequences])


def boundaries(codes: np.ndarray) -> list[int]:
    return (np.flatnonzero(np.diff(codes)) + 1).tolist()


def ambiguous_handover() -> np.ndarray:
    emission = np.zeros((300, 3))
    emission[:100, HOME], emission[:100, AWAY] = 1.0, -1.0
    emission[200:, HOME], emission[200:, AWAY] = -1.0, 1.0
    emission[:, DEAD] = -3.0
    return emission


def test_zero_switch_evidence_reproduces_the_decoder_exactly() -> None:
    fitted = segment_model()
    rng = np.random.default_rng(1)
    emission = rng.normal(size=(997, 3))
    base = hsmm.decode_block(emission, fitted, 5)
    np.testing.assert_array_equal(hsmm.decode_block(emission, fitted, 5, np.zeros((997, 3))),
                                  base)
    blocks = [(1, 0, 499), (2, 500, 996)]
    entering = rng.normal(scale=5.0, size=(997, 3))
    without = decode_emission(emission, blocks, fitted, SETTINGS)
    np.testing.assert_array_equal(decode_emission(emission, blocks, fitted, SETTINGS,
                                                  Entering(entering)), without)
    with pytest.raises(ValueError, match="switch prior"):
        decode_emission(emission, blocks, fitted, replace(SETTINGS, switch_weight=1.0))


@pytest.mark.parametrize("touch", [120, 180])
def test_switch_evidence_moves_an_ambiguous_boundary_to_the_touch(touch: int) -> None:
    fitted = segment_model()
    emission = ambiguous_handover()
    (alone,) = boundaries(hsmm.decode_block(emission, fitted, 5))
    assert alone != touch
    switch = np.zeros_like(emission)
    switch[touch, AWAY] = 8.0
    assert boundaries(hsmm.decode_block(emission, fitted, 5, switch)) == [touch]


def test_switch_evidence_creates_a_short_spell_the_emissions_alone_miss() -> None:
    fitted = segment_model()
    emission = np.zeros((400, 3))
    emission[:, HOME], emission[:, AWAY], emission[:, DEAD] = 0.3, -0.3, -3.0
    emission[180:220, HOME], emission[180:220, AWAY] = -0.05, 0.05
    assert boundaries(hsmm.decode_block(emission, fitted, 5)) == []
    switch = np.zeros_like(emission)
    switch[180, AWAY] = switch[220, HOME] = 6.0
    codes = hsmm.decode_block(emission, fitted, 5, switch)
    assert boundaries(codes) == [180, 220]
    assert (codes[180:220] == AWAY).all()


def test_switch_evidence_never_touches_dead_ball_transitions() -> None:
    fitted = segment_model()
    emission = np.zeros((300, 3))
    emission[:150, HOME], emission[150:, DEAD] = 1.0, 1.0
    emission[:150, DEAD], emission[150:, HOME] = -1.0, -1.0
    emission[:, AWAY] = -2.0
    switch = np.zeros_like(emission)
    switch[:, DEAD] = -50.0
    np.testing.assert_array_equal(hsmm.decode_block(emission, fitted, 5, switch),
                                  hsmm.decode_block(emission, fitted, 5))


def test_step_switch_is_the_mean_bayes_factor_of_the_frames_nearest_each_boundary() -> None:
    rng = np.random.default_rng(2)
    switch = rng.normal(scale=3.0, size=(23, 3))
    steps = hsmm.step_switch(switch, 5)
    assert steps.shape == (6, 3)
    groups = {1: switch[1:8], 2: switch[8:13], 3: switch[13:18], 4: switch[18:23]}
    for boundary, frames in groups.items():
        np.testing.assert_allclose(steps[boundary],
                                   logsumexp(frames, axis=0) - np.log(len(frames)))
    assert (steps[[0, 5]] == 0).all()


def test_step_switch_scales_linearly_so_the_weight_multiplies_the_log_mean() -> None:
    rng = np.random.default_rng(3)
    switch = rng.normal(scale=3.0, size=(40, 3))
    weighted = hsmm.step_switch(switch, 10)
    assert not np.allclose(hsmm.step_switch(2.0 * switch, 10), 2.0 * weighted)
    fitted = segment_model()
    emission = ambiguous_handover()
    entering = np.zeros_like(emission)
    entering[3, AWAY] = 4.0
    cached = Entering(entering)
    settings = replace(SETTINGS, switch_weight=2.0)
    decode_emission(emission, [(1, 0, 299)], fitted, settings, cached)
    np.testing.assert_allclose(cached.steps[0, SETTINGS.stride][1],
                               hsmm.step_switch(entering, SETTINGS.stride)[1])
    assert cached.steps[0, SETTINGS.stride][1, AWAY] > 0


def test_near_entries_mark_the_tight_tolerance_around_changes_inside_blocks() -> None:
    codes = np.array([HOME, HOME, AWAY, AWAY, AWAY, DEAD, HOME, AWAY, HOME, HOME])
    blocks = [(1, 0, 6), (2, 7, 9)]
    assert changes.near_entries(codes, AWAY, blocks, 1).tolist() == [
        False, True, True, True, False, False, False, False, False, False]
    assert changes.near_entries(codes, HOME, blocks, 1).tolist() == [
        False] * 7 + [True, True, True]


def synthetic_examples() -> list[model.TrainingExample]:
    return [training_example(replace(possession_match(seed=seed), name=f"m{seed}"), PARAMS)
            for seed in range(3)]


def test_entry_evidence_is_team_symmetric() -> None:
    examples = synthetic_examples()
    fitted = fit_model(examples[:2], SETTINGS, 4)
    assert fitted.changes is not None
    home, away = examples[2].base
    np.testing.assert_allclose(fitted.changes.entry_evidence(home),
                               fitted.changes.entry_evidence(away), atol=1e-12)
    assert (fitted.changes.entry_evidence(home)[:, DEAD] == 0).all()


def test_switch_prior_is_fitted_on_the_training_fold_only(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    examples = synthetic_examples()
    loaded = [train.Loaded(example=example,
                           match=train.labels_only(replace(possession_match(seed=seed),
                                                           name=f"m{seed}")),
                           data_key=f"data{seed}") for seed, example in enumerate(examples)]
    fitted_on: list[list[str]] = []
    original = model.fit_changes

    def recorded(train_examples: list[model.TrainingExample], *args: object
                 ) -> changes.ChangeModel:
        fitted_on.append([example.name for example in train_examples])
        return original(train_examples, *args)

    monkeypatch.setattr(model, "fit_changes", recorded)
    grid = [SETTINGS, replace(SETTINGS, switch_weight=1.0)]
    context = train.Context(seed=0, frame_step=4, grid=grid, loaded=loaded,
                            out=tmp_path / "model.pkl", fingerprint="code")
    (fold,) = train.run_fold(context, "dfl", loaded, [loaded[0]], loaded[0].key)
    assert fitted_on == [[loaded[1].key, loaded[2].key]]
    assert len(fold.scores) == len(grid)


def test_exact_ties_prefer_the_smaller_switch_weight() -> None:
    grid = [replace(SETTINGS, switch_weight=1.0), replace(SETTINGS, stride=10)]
    assert train.best_setting(np.array([[0.5, 0.5], [0.5, 0.5]]), grid) == 1
    assert train.best_setting(np.array([[0.6, 0.5], [0.5, 0.5]]), grid) == 0


def test_no_change_in_the_rows_drops_the_switch_prior(capsys: pytest.CaptureFixture[str]) -> None:
    rows = np.zeros((4, len(changes.CHANGE_FEATURES)), dtype=np.float32)
    assert changes.fit_change_model([(rows, np.zeros(4, dtype=bool))], 1.0, 3, 0) is None
    assert "no switch prior" in capsys.readouterr().err
    fitted = fit_model(synthetic_examples()[:2], SETTINGS, 4, switch_prior=False)
    assert train.effective_settings(fitted, replace(SETTINGS, switch_weight=1.0)) == SETTINGS
    assert for_inference(fitted, replace(SETTINGS, switch_weight=1.0)).settings == SETTINGS


def test_inference_keeps_the_switch_prior_only_when_its_weight_is_used() -> None:
    fitted = fit_model(synthetic_examples()[:2], SETTINGS, 4)
    assert fitted.changes is not None
    assert for_inference(fitted, SETTINGS).changes is None
    used = for_inference(fitted, replace(SETTINGS, switch_weight=0.5))
    assert used.changes is fitted.changes and used.settings.switch_weight == 0.5


def test_switch_grid_includes_the_previous_decoder_and_the_ablation_drops_the_rest() -> None:
    full, off = train.settings_grid(), train.settings_grid(switch_prior=False)
    assert {settings.switch_weight for settings in full} == set(train.SWITCH_WEIGHTS)
    assert {settings.switch_weight for settings in off} == {0.0}
    assert [settings for settings in full if settings.switch_weight == 0.0] == off
    assert len(full) == len(off) * len(train.SWITCH_WEIGHTS)


def test_series_continues_the_one_two_five_rule_in_both_directions() -> None:
    value, followed = 0.1, []
    for _ in range(9):
        value = train.next_in_series(value)
        followed.append(value)
    assert followed == pytest.approx([0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    assert train.next_in_series(0.5) == pytest.approx(1.0)
    assert train.next_in_series(3.0) == pytest.approx(5.0)
    assert train.next_in_series(max(train.SWITCH_WEIGHTS)) == pytest.approx(5.0)


def test_stride_series_continues_in_whole_frames_of_the_series_in_seconds() -> None:
    followed, value = [], max(train.STRIDES)
    for _ in range(4):
        value = train.next_stride(value)
        followed.append(value)
    assert followed == [50, 125, 250, 500]
    assert train.next_stride(5) == 25


def test_extension_adds_only_the_settings_on_the_new_axis_values() -> None:
    grid = train.settings_grid()
    weights = train.grid_extension(grid, True, False)
    assert {settings.switch_weight for settings in weights} == {5.0}
    assert {settings.stride for settings in weights} == set(train.STRIDES)
    strides = train.grid_extension(grid, False, True)
    assert {settings.stride for settings in strides} == {50}
    assert {settings.switch_weight for settings in strides} == set(train.SWITCH_WEIGHTS)
    both = train.grid_extension(grid, True, True)
    rectangle = (len(train.STRIDES) + 1) * (len(train.SWITCH_WEIGHTS) + 1)
    assert len(both) == len(train.EMISSION_SPANS) * rectangle - len(grid)
    assert len({*grid, *both}) == len(grid) + len(both)
    axes = train.grid_axes([*grid, *both])
    assert axes["switch_weights"] == [0.0, 0.5, 1.0, 2.0, 5.0]
    assert axes["strides"] == [*sorted(train.STRIDES), 50]
    assert train.grid_extension(grid, False, False) == []


def test_only_a_choice_of_the_largest_value_of_an_axis_is_an_edge() -> None:
    grid = [SETTINGS, replace(SETTINGS, switch_weight=1.0), replace(SETTINGS, stride=10)]
    assert train.grid_edges(grid, [0, 1]) == (True, False)
    assert train.grid_edges(grid, [0, 2]) == (False, True)
    assert train.grid_edges(grid, [0]) == (False, False)
    off = train.settings_grid(switch_prior=False)
    assert train.grid_edges(off, [0, 1, 4]) == (False, False)
    assert train.grid_edges(off, [2]) == (False, True)


def raw(value: float) -> train.Scored:
    return train.Scored(raw={"spell_f1_mean": value, "macro_f1": 0.0}, fair=None)


class Scorer:
    def __init__(self, monkeypatch: pytest.MonkeyPatch,
                 spell: dict[str, Callable[[DecoderSettings], float]]) -> None:
        self.spell = spell
        self.scored: list[list[DecoderSettings]] = []
        self.emissions: list[str] = []
        monkeypatch.setattr(train, "match_emission", self.emission)
        monkeypatch.setattr(train, "score_grid", self.score)

    def emission(self, fold_model: object, item: SimpleNamespace) -> str:
        self.emissions.append(item.key)
        return item.key

    def score(self, fold_model: object, item: SimpleNamespace, evidence: str,
              grid: list[DecoderSettings]) -> list[train.Scored]:
        self.scored.append(grid)
        return [raw(self.spell[item.key](settings)) for settings in grid]

    def folds(self, grid: list[DecoderSettings]) -> tuple[list[train.Fold], list[SimpleNamespace]]:
        folds = [train.Fold(key=key, source="dfl", model=None,
                            scores=[raw(function(settings)) for settings in grid])
                 for key, function in self.spell.items()]
        return folds, [SimpleNamespace(key=fold.key) for fold in folds]


def peaked(peak_weight: float, peak_stride: int, width: float = 100.0
           ) -> Callable[[DecoderSettings], float]:
    def spell(settings: DecoderSettings) -> float:
        gap = (abs(settings.switch_weight - peak_weight) / width
               + abs(settings.stride - peak_stride) / 1000)
        return 0.0 if settings.switch_weight == 0.0 else 0.9 - gap

    return spell


def test_grid_extends_both_axes_until_no_choice_sits_on_an_edge(
        monkeypatch: pytest.MonkeyPatch) -> None:
    grid = train.settings_grid()
    scorer = Scorer(monkeypatch, {f"dfl/{index}": peaked(10.0, 125) for index in range(3)})
    folds, members = scorer.folds(grid)
    chosen = train.select_settings("dfl", grid, members, folds)
    axes = train.grid_axes(chosen.final_grid)
    assert axes["switch_weights"] == [*train.SWITCH_WEIGHTS, 5.0, 10.0, 20.0]
    assert axes["strides"] == [*sorted(train.STRIDES), 50, 125, 250]
    best = chosen.grid[chosen.selected]
    assert (best.switch_weight, best.stride) == (10.0, 125)
    assert {chosen.grid[choice] for choice in chosen.choices} == {best}
    assert all(len(fold.scores) == len(chosen.grid) for fold in chosen.folds)
    assert len(set(chosen.grid)) == len(chosen.grid)
    assert not any(settings in grid for added in scorer.scored for settings in added)
    scored = Counter(settings for added in scorer.scored for settings in added)
    assert set(scored) == set(chosen.grid) - set(grid)
    assert set(scored.values()) == {len(folds)}
    for searched, choice in zip(chosen.fold_grids, chosen.choices):
        assert train.grid_edges(searched, [searched.index(chosen.grid[choice])]) == (False,
                                                                                   False)
    assert sorted(scorer.emissions) == sorted(fold.key for fold in folds)


def test_only_a_strictly_better_new_setting_keeps_the_extension_going(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def saturating(settings: DecoderSettings) -> float:
        return 0.0 if settings.switch_weight == 0.0 else min(
            settings.switch_weight + settings.stride / 25, 3.0) / 10

    grid = train.settings_grid()
    scorer = Scorer(monkeypatch, {f"dfl/{index}": saturating for index in range(2)})
    folds, members = scorer.folds(grid)
    chosen = train.select_settings("dfl", grid, members, folds)
    best = chosen.grid[chosen.selected]
    assert (best.switch_weight, best.stride) == (2.0, 25)
    assert len(scorer.scored) == 2
    axes = train.grid_axes(chosen.final_grid)
    assert axes["switch_weights"] == [*train.SWITCH_WEIGHTS, 5.0]
    assert axes["strides"] == [*sorted(train.STRIDES), 50]
    assert {chosen.grid[choice] for choice in chosen.choices} == {best}


def test_a_nested_choice_never_depends_on_the_fold_it_leaves_out(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def outcome(peak_of_first: float) -> train.Selection:
        grid = [replace(SETTINGS, stride=stride, switch_weight=weight)
                for stride in (5, 10) for weight in train.SWITCH_WEIGHTS]
        scorer = Scorer(monkeypatch, {"dfl/0": peaked(peak_of_first, 5, 8.0),
                                      "dfl/1": peaked(1.0, 5, 20.0),
                                      "dfl/2": peaked(1.0, 5, 20.0)})
        folds, members = scorer.folds(grid)
        return train.select_settings("dfl", grid, members, folds)

    near, far = outcome(5.0), outcome(500.0)
    assert train.top_weight(near.fold_grids[0]) == 2.0
    assert near.grid[near.choices[0]].switch_weight == 1.0
    assert train.top_weight(near.fold_grids[1]) == 10.0
    assert near.grid[near.choices[1]].switch_weight == 5.0
    assert near.grid[near.selected].switch_weight == 5.0
    assert near.fold_grids[0] == far.fold_grids[0]
    assert near.grid[near.choices[0]] == far.grid[far.choices[0]]
    assert near.grid[near.choices[1]] != far.grid[far.choices[1]]


def test_without_the_switch_prior_only_the_stride_can_extend(
        monkeypatch: pytest.MonkeyPatch) -> None:
    grid = train.settings_grid(switch_prior=False)
    assert {settings.switch_weight for settings in grid} == {0.0}
    best = next(settings for settings in grid
                if settings.stride == 10 and settings.emission_weight == 1.0)
    scorer = Scorer(monkeypatch, {f"dfl/{index}": lambda settings: float(settings == best)
                                  for index in range(2)})
    folds, members = scorer.folds(grid)
    chosen = train.select_settings("dfl", grid, members, folds)
    assert chosen.grid == grid and scorer.scored == []
    assert chosen.grid[chosen.selected] == best
    assert chosen.choices == [chosen.selected] * 2


def test_a_variant_ends_with_a_grid_whose_edges_were_not_selected(tmp_path: Path) -> None:
    examples = synthetic_examples()
    loaded = [train.Loaded(example=example,
                           match=train.labels_only(replace(possession_match(seed=seed),
                                                           name=f"m{seed}")),
                           data_key=f"data{seed}") for seed, example in enumerate(examples)]
    grid = [SETTINGS, replace(SETTINGS, switch_weight=1.0)]
    context = train.Context(seed=0, frame_step=4, grid=grid, loaded=loaded,
                            out=tmp_path / "model.pkl", fingerprint="code",
                            outside_sources=())
    result = train.run_variant(context, "dfl")
    assert result.grid[:len(grid)] == grid
    assert all(len(fold.scores) == len(result.grid) for fold in result.folds)
    assert train.grid_edges(result.grid, [*result.choices, result.selected]) == (False, False)
    assert result.model.settings == train.effective_settings(result.model,
                                                             result.grid[result.selected])
    weights = [row["switch_weight"] for row in result.held_out()]
    assert set(weights) <= {settings.switch_weight for settings in result.grid}
