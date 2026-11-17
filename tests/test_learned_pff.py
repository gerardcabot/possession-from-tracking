from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from learned_synthetic import possession_match

from possession.cleaning import CleaningParams
from possession.external.sources import DFL, METRICA, PFF
from possession.intervals import DEAD
from possession.learned import train
from possession.learned.external import fold_path, model_for, spilled, training_example
from possession.learned.framefeatures import FeatureLayout
from possession.learned.model import (
    DecoderSettings,
    fit_model,
    read_metadata,
    save_model,
    training_keep,
    training_rows,
)

SETTINGS = DecoderSettings(emission_weight=0.5, stride=5)
PARAMS = CleaningParams(ball_max_speed_mps=40.0, ball_deviation_m=1.6,
                        player_max_speed_mps=12.0, player_deviation_m=0.3,
                        duplicate_distance_m=0.1, max_gap_frames=8, ball_half_span=1,
                        player_half_span=1)
RUN_ID = "run"
LAYOUT = FeatureLayout(scales=(10,), short_scales=(4,), touch=True)


def fake_loaded(source: str, name: str) -> train.Loaded:
    example = train.TrainingExample(name=f"{source}/{name}", base=(), codes=np.zeros(0),
                                    blocks=[], params=PARAMS, dead_labels=True)
    return train.Loaded(example=example, match=SimpleNamespace(source=source), data_key=name)


def loaded_match(source: str, seed: int) -> train.Loaded:
    match = replace(possession_match(seed=seed), source=source, name=f"m{seed}")
    return train.Loaded(example=training_example(match, PARAMS),
                        match=train.labels_only(match), data_key=f"data{seed}")


def test_grouped_source_is_held_out_in_as_many_contiguous_groups_as_the_largest_other() -> None:
    members = ([fake_loaded(DFL, str(index)) for index in range(3)]
               + [fake_loaded(METRICA, "g1")]
               + [fake_loaded(PFF, str(index)) for index in range(8)])
    held = train.holdouts(members)
    single = [key for key, items in held if len(items) == 1 and not key.startswith(PFF)]
    assert single == ["dfl/0", "dfl/1", "dfl/2", "metrica/g1"]
    groups = [(key, [item.key for item in items]) for key, items in held
              if key.startswith(PFF)]
    assert [key for key, _ in groups] == ["pff/group1of3", "pff/group2of3", "pff/group3of3"]
    assert [keys for _, keys in groups] == [["pff/0", "pff/1", "pff/2"],
                                            ["pff/3", "pff/4", "pff/5"], ["pff/6", "pff/7"]]


def test_grouped_source_alone_falls_back_to_leave_one_match_out() -> None:
    members = [fake_loaded(PFF, str(index)) for index in range(3)]
    assert [key for key, _ in train.holdouts(members)] == ["pff/0", "pff/1", "pff/2"]


def test_grouped_source_rows_are_subsampled_to_the_largest_other_source() -> None:
    items = ([fake_loaded(DFL, str(index)) for index in range(7)]
             + [fake_loaded(METRICA, str(index)) for index in range(3)]
             + [fake_loaded(PFF, str(index)) for index in range(64)])
    assert train.row_steps(items) == {DFL: 1, METRICA: 1, PFF: 10}
    steps = {example.name: example.row_step for example in train.fit_examples(items)}
    assert steps["pff/0"] == 10 and steps["dfl/0"] == 1
    assert train.row_steps([fake_loaded(PFF, "a")]) == {PFF: 1}


def test_row_step_keeps_every_nth_frame_of_that_match_only() -> None:
    match = possession_match(seed=0)
    example = training_example(match, PARAMS)
    full = training_rows([example], LAYOUT, 1)
    stepped = training_rows([replace(example, row_step=4)], LAYOUT, 1)
    assert len(stepped.codes) == pytest.approx(len(full.codes) / 4, abs=2)


def test_team_only_variant_keeps_grouped_matches_out_of_the_dead_ball_model() -> None:
    loaded = [fake_loaded(DFL, "a"), fake_loaded(METRICA, "b"), fake_loaded(PFF, "c")]
    members = train.members_of("dfl+metrica+pff_ha", loaded)
    assert {item.key: item.example.dead_labels for item in members} == {
        "dfl/a": True, "metrica/b": True, "pff/c": False}
    full = train.members_of("dfl+metrica+pff", loaded)
    assert all(item.example.dead_labels for item in full)


def test_pff_variants_are_only_run_when_listed_and_a_superset_of_the_default_is_complete() -> None:
    assert not {"dfl+metrica+pff", "dfl+metrica+pff_ha"} & set(train.DEFAULT_VARIANTS)
    listed = train.parse_variants("dfl,dfl+metrica,metrica,dfl+metrica+pff,dfl+metrica+pff_ha")
    assert not train.partial_variants(listed)
    assert train.partial_variants(["dfl+metrica+pff", "dfl+metrica+pff_ha"])
    assert train.partial_variants(["dfl", "dfl+metrica"])


def test_pff_is_loaded_only_when_a_variant_trains_on_it_or_scoring_it_is_opted_in() -> None:
    assert train.scored_sources(["dfl", "metrica"]) == (DFL, METRICA, "skillcorner")
    assert train.scored_sources(list(train.DEFAULT_VARIANTS)) == (DFL, METRICA, "skillcorner")
    assert train.scored_sources(["dfl+metrica+pff"]) == (DFL, METRICA, "skillcorner", PFF)
    assert train.scored_sources(["dfl"], (PFF,)) == (DFL, METRICA, "skillcorner", PFF)


def test_final_model_scores_pff_outside_only_when_opted_in(tmp_path: Path) -> None:
    loaded = [loaded_match(DFL, 0), loaded_match(DFL, 1), loaded_match(PFF, 2)]
    members = loaded[:2]
    context = train.Context(seed=0, frame_step=1, grid=[SETTINGS], loaded=loaded,
                            out=tmp_path / "model.pkl", fingerprint="code")
    _, scores, _ = train.run_final(context, "dfl", members, SETTINGS)
    assert scores == {}
    opted = replace(context, out=tmp_path / "opted.pkl",
                    outside_sources=(*train.ALWAYS_SCORED, PFF))
    _, scores, _ = train.run_final(opted, "dfl", members, SETTINGS)
    assert list(scores) == ["pff/m2"]


def test_decoder_selection_ignores_folds_whose_dead_labels_the_variant_does_not_trust() -> None:
    def folds_of(source: str, spells: list[float], trusted: bool) -> train.Fold:
        raw = [{"spell_f1_mean": value, "macro_f1": value, "n_pred_spells": 1.0,
                "n_gt_spells": 1.0} for value in spells]
        return train.Fold(key=f"{source}/a", source=source, model=None,
                          scores=[train.Scored(raw=values, fair=None) for values in raw],
                          dead_labels=trusted)

    folds = [folds_of(DFL, [0.9, 0.1], True), folds_of(PFF, [0.0, 1.0], False)]
    assert train.best_setting(train.selection_scores(folds), [SETTINGS, SETTINGS]) == 0
    folds[1] = replace(folds[1], dead_labels=True)
    assert train.best_setting(train.selection_scores(folds), [SETTINGS, SETTINGS]) == 1


def test_spilled_features_are_read_from_disk_and_train_the_same_model(tmp_path: Path) -> None:
    match = replace(possession_match(seed=0), source=PFF, name="m0")
    example = training_example(match, PARAMS)
    disk = spilled(example, tmp_path)
    assert all(isinstance(features.matrix, np.memmap) for features in disk.base)
    for kept, original in zip(disk.base, example.base):
        np.testing.assert_array_equal(kept.matrix, original.matrix)
    np.testing.assert_array_equal(training_rows([disk], LAYOUT, 1).matrix,
                                  training_rows([example], LAYOUT, 1).matrix)
    assert train.labels_only(match).ball.base is None


def test_team_only_examples_train_on_no_dead_rows() -> None:
    example = training_example(replace(possession_match(seed=0), source=PFF, name="m0"), PARAMS)
    team_only = replace(example, dead_labels=False)
    assert (example.codes[training_keep(example, 1)] == DEAD).any()
    assert not (team_only.codes[training_keep(team_only, 1)] == DEAD).any()
    assert len(training_rows([team_only], LAYOUT, 1).codes) < len(
        training_rows([example], LAYOUT, 1).codes)


def test_group_fold_scores_every_held_out_match_and_is_reused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = [loaded_match(DFL, 0), loaded_match(PFF, 1), loaded_match(PFF, 2)]
    context = train.Context(seed=0, frame_step=1, grid=[SETTINGS], loaded=loaded,
                            out=tmp_path / "model.pkl", fingerprint="code")
    held = loaded[1:]
    folds = train.run_fold(context, "dfl+metrica+pff", loaded, held, "pff/group1of1")
    assert [fold.key for fold in folds] == ["pff/m1", "pff/m2"]
    assert all(fold.group == ("pff/m1", "pff/m2") for fold in folds)
    assert folds[0].model.training_matches == ("dfl/m0",)
    monkeypatch.setattr(train, "fit_model", lambda *args: pytest.fail("refitted"))
    again = train.run_fold(context, "dfl+metrica+pff", loaded, held, "pff/group1of1")
    assert [fold.scores for fold in again] == [fold.scores for fold in folds]
    rows = train.VariantResult(name="v", members=[], folds=folds, grid=[SETTINGS],
                               final_grid=[SETTINGS], fold_grids=[[SETTINGS]] * 2,
                               choices=[0, 0], selected=0, model=folds[0].model,
                               outside={}).held_out()
    assert [row["held_out_by"] for row in rows] == [train.HELD_OUT_GROUP] * 2


def test_validation_reads_a_group_fold_through_one_shared_model_file(tmp_path: Path) -> None:
    loaded = [loaded_match(DFL, 0), loaded_match(PFF, 1), loaded_match(PFF, 2)]
    examples = [item.example for item in loaded]
    out = tmp_path / "model.pkl"
    final = fit_model(examples, SETTINGS)
    save_model(out, final, RUN_ID, {"variant": "v"})
    group_model = fit_model(examples[:1], SETTINGS)
    group = ("pff/m1", "pff/m2")
    folds = [train.Fold(key=key, source=PFF, model=group_model, scores=[], group=group,
                        fold_key="pff/group1of1") for key in group]
    other = DecoderSettings(emission_weight=0.1, stride=10)
    result = train.VariantResult(name="v", members=[], folds=folds, grid=[SETTINGS, other],
                                 final_grid=[SETTINGS, other],
                                 fold_grids=[[SETTINGS, other]] * 2, choices=[0, 1],
                                 selected=0, model=final, outside={})
    train.save_folds([result], out, RUN_ID)
    shared = fold_path(out, "v", "pff/group1of1")
    assert shared.is_file()
    assert not fold_path(out, "v", "pff/m1").is_file()
    assert not fold_path(out, "v", "pff/m2").is_file()
    assert read_metadata(fold_path(out, "v", "pff/m1"))["held_out_group"] == list(group)
    settings = []
    for item in loaded[1:]:
        match = replace(possession_match(seed=0), source=PFF, name=item.key.split("/")[1])
        fold = model_for(match, final, out)
        assert fold.training_matches == ("dfl/m0",)
        settings.append(fold.settings)
    assert settings == [SETTINGS, other]
    save_model(fold_path(out, "v", "pff/m2"), group_model, RUN_ID,
               {"held_out_group": ["pff/m2"]})
    match = replace(possession_match(seed=0), source=PFF, name="m2")
    with pytest.raises(ValueError, match="other matches"):
        model_for(match, final, out)
    save_model(shared, group_model, "older")
    match = replace(possession_match(seed=0), source=PFF, name="m1")
    with pytest.raises(ValueError, match="training run"):
        model_for(match, final, out)
    shared.with_suffix(".json").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        model_for(match, final, out)


def test_pff_is_summarised_fair_and_raw_but_never_selects() -> None:
    rows = [{"source": PFF, "raw": {"spell_f1_mean": 0.4, "spell_bootstrap_low": 0.3,
                                    "spell_bootstrap_high": 0.5},
             "fair": {"spell_f1_mean": 0.6, "spell_bootstrap_low": 0.5,
                      "spell_bootstrap_high": 0.7}}]
    summary = train.variant_summary(rows)
    assert summary[PFF]["spell_f1_mean"] == pytest.approx(0.6)
    assert summary[train.PFF_RAW]["spell_f1_mean"] == pytest.approx(0.4)
    best = {"spell_f1_mean": 0.9, "spell_bootstrap_low": 0.89, "spell_bootstrap_high": 0.91}
    low = {"spell_f1_mean": 0.1, "spell_bootstrap_low": 0.09, "spell_bootstrap_high": 0.11}
    summaries = {"a": {METRICA: best, "metrica_raw": best, DFL: best, "skillcorner": None,
                       PFF: low, train.PFF_RAW: low},
                 "b": {METRICA: low, "metrica_raw": low, DFL: low, "skillcorner": None,
                       PFF: best, train.PFF_RAW: best}}
    assert train.select_variant(summaries, train.SELECTION_SCORES["raw"]) == "a"
