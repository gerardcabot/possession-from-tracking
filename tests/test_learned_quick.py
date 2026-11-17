import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from learned_synthetic import possession_match

from possession.cleaning import CleaningParams
from possession.learned import quick, train
from possession.learned.external import fold_path, labels_only, training_example
from possession.learned.framefeatures import feature_matrix
from possession.learned.model import DecoderSettings

PARAMS = CleaningParams(ball_max_speed_mps=40.0, ball_deviation_m=1.6,
                        player_max_speed_mps=12.0, player_deviation_m=0.3,
                        duplicate_distance_m=0.1, max_gap_frames=8, ball_half_span=1,
                        player_half_span=1)
SETTINGS = DecoderSettings(emission_weight=0.5, stride=5)
GRID = [SETTINGS, DecoderSettings(emission_weight=0.1, stride=10)]
VARIANT = "dfl"


def cached(tmp_path: Path, seeds: tuple[int, ...] = (0, 1, 2)
           ) -> tuple[Path, list[train.Loaded]]:
    features = tmp_path / "features"
    loaded = []
    for seed in seeds:
        match = possession_match(seed=seed)
        example = training_example(match, PARAMS)
        quick.save_example(quick.example_dir(features, example.name), example, labels_only(match),
                           f"data{seed}", ())
        loaded.append(train.Loaded(example=example, match=labels_only(match), data_key=f"data{seed}"))
    return features, loaded


def data_keys(loaded: list[train.Loaded]) -> dict[str, str]:
    return {item.key: item.data_key for item in loaded}


def context_for(loaded: list[train.Loaded], out: Path,
                grid: list[DecoderSettings] | None = None) -> train.Context:
    return train.Context(seed=0, frame_step=1, grid=list(grid or [SETTINGS]), loaded=loaded,
                         out=out, fingerprint="code", outside_sources=())


def row(match: str, spell: float, fair: bool = False) -> dict[str, object]:
    scores = {"spell_f1_mean": spell, "macro_f1": spell, "n_pred_spells": 1.0, "n_gt_spells": 1.0}
    return {"match": match, "source": match.split("/")[0], "raw": {**scores, "spell_f1_mean": 0.0}
            if fair else scores, "fair": scores if fair else None}


def test_cached_examples_round_trip(tmp_path: Path) -> None:
    features, loaded = cached(tmp_path)
    again = quick.load_examples(features, None, (VARIANT,), (), data_keys(loaded))
    assert [item.key for item in again] == [item.key for item in loaded]
    for before, after in zip(loaded, again):
        assert after.data_key == before.data_key
        assert after.source == before.source
        assert after.example.blocks == before.example.blocks
        assert after.example.params == before.example.params
        assert after.example.dead_labels == before.example.dead_labels
        np.testing.assert_array_equal(after.example.codes, before.example.codes)
        np.testing.assert_array_equal(after.match.labels, before.match.labels)
        np.testing.assert_array_equal(after.match.periods, before.match.periods)
        for old, new in zip(before.example.base, after.example.base):
            assert new.names == old.names
            assert new.first == old.first
            assert new.segments == old.segments
            assert new.fps == old.fps
            np.testing.assert_array_equal(np.asarray(new.matrix), old.matrix)
            np.testing.assert_array_equal(np.asarray(new.touch), old.touch)
    assert [item.key for item in quick.load_examples(features, {"synthetic1"}, (VARIANT,), (),
                                                     data_keys(loaded))] == [loaded[1].key]
    with pytest.raises(ValueError, match="not in the feature cache"):
        quick.load_examples(features, {"dfl/missing"}, (VARIANT,), (), data_keys(loaded))
    with pytest.raises(ValueError, match="degraded data data0, current changed"):
        quick.load_examples(features, None, (VARIANT,), (),
                            {**data_keys(loaded), loaded[0].key: "changed"})
    with pytest.raises(ValueError, match="no current degraded data"):
        quick.load_examples(features, None, (VARIANT,), (), data_keys(loaded[1:]))
    with pytest.raises(ValueError, match="line groups"):
        quick.load_example(quick.example_dir(features, loaded[0].key), ("lines",))
    meta_path = quick.example_dir(features, loaded[0].key) / quick.META_NAME
    meta = json.loads(meta_path.read_text())
    meta["code"] = "other"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="other code"):
        quick.load_example(quick.example_dir(features, loaded[0].key))


def test_quick_fold_scores_like_train_and_stores_evidence(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    features, loaded = cached(tmp_path)
    out = tmp_path / "quick" / "model.pkl"
    context = context_for(loaded, out)
    held = [loaded[0]]
    (fold,) = quick.run_quick_fold(context, VARIANT, loaded, held, loaded[0].key)
    (expected,) = train.run_fold(context_for(loaded, tmp_path / "train" / "model.pkl"), VARIANT,
                                 loaded, held, loaded[0].key)
    assert fold.scores == expected.scores
    path = fold_path(out, VARIANT, loaded[0].key)
    components = quick.load_components(quick.evidence_path(path, loaded[0].key))
    evidence = components.evidence
    recomputed = train.match_emission(fold.model, loaded[0])
    np.testing.assert_array_equal(evidence.emission, recomputed.emission)
    assert (evidence.entering is None) == (recomputed.entering is None)
    if recomputed.entering is not None:
        np.testing.assert_array_equal(evidence.entering.frames, recomputed.entering.frames)
    assert evidence.adaptation == recomputed.adaptation
    matrix = feature_matrix(loaded[0].example.base[0], fold.model.layout)
    np.testing.assert_array_equal(fold.model.classifier.log_emission(matrix, components.dead),
                                  evidence.emission)
    np.testing.assert_array_equal(fold.model.classifier.side.decision_function(matrix),
                                  components.away)
    monkeypatch.setattr(quick, "fit_model", lambda *args: pytest.fail("refitted"))
    quick.evidence_path(path, loaded[0].key).unlink()
    (again,) = quick.run_quick_fold(context, VARIANT, loaded, held, loaded[0].key)
    assert again.scores == fold.scores
    assert quick.evidence_path(path, loaded[0].key).is_file()
    elsewhere = train.Context(seed=0, frame_step=1, grid=[SETTINGS], loaded=loaded,
                              out=tmp_path / "reuse" / "model.pkl", fingerprint="edited code",
                              outside_sources=())
    with pytest.raises(pytest.fail.Exception):
        quick.run_quick_fold(elsewhere, VARIANT, loaded, held, loaded[0].key)
    (reused,) = quick.run_quick_fold(elsewhere, VARIANT, loaded, held, loaded[0].key, fit_from=out)
    assert reused.scores == fold.scores
    assert reused.model.training_matches == fold.model.training_matches
    with pytest.raises(FileNotFoundError, match="no fold model"):
        quick.run_quick_fold(elsewhere, VARIANT, loaded, [loaded[1]], loaded[1].key,
                             fit_from=tmp_path / "nowhere" / "model.pkl")
    fewer = context_for(loaded[:2], tmp_path / "fewer" / "model.pkl")
    with pytest.raises(ValueError, match=f"fold {loaded[0].key} was trained on"):
        quick.run_quick_fold(fewer, VARIANT, loaded[:2], held, loaded[0].key, fit_from=out)
    assert not fold_path(fewer.out, VARIANT, loaded[0].key).exists()


def test_parallel_folds_select_like_train_and_report(tmp_path: Path) -> None:
    features, loaded = cached(tmp_path)
    out = tmp_path / "quick" / "model.pkl"
    context = context_for(loaded, out, GRID)
    members, folds = quick.run_folds(features, context, VARIANT, workers=2)
    assert [fold.key for fold in folds] == [item.key for item in loaded]
    evidence = quick.fold_evidence(out, VARIANT, folds)
    selection = quick.select_settings(VARIANT, GRID, members, folds, evidence)
    expected = train.select_settings(VARIANT, GRID, members, folds)
    assert selection.choices == expected.choices
    assert selection.selected == expected.selected
    assert selection.grid == expected.grid
    result = quick.report(VARIANT, selection, {"frame_step": 1})
    assert [item["match"] for item in result["held_out"]] == [item.key for item in loaded]
    assert result["summary"][VARIANT]["n_matches"] == len(loaded)
    assert result["selected"] == asdict(selection.grid[selection.selected])
    path = quick.write_report(out, result)
    assert path == out.parent / quick.REPORT_NAME
    assert json.loads(path.read_text())["frame_step"] == 1
    named = quick.write_report(out.parent / "decode.json", result)
    assert named == out.parent / "decode.json"
    assert json.loads(path.read_text())["frame_step"] == 1
    quick.print_report(result)


def test_decode_rescoring_reproduces_fold_scores(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    features, loaded = cached(tmp_path)
    out = tmp_path / "quick" / "model.pkl"
    with pytest.raises(ValueError, match="no fold models"):
        quick.fold_models(out, VARIANT)
    _, folds = quick.run_folds(features, context_for(loaded, out, GRID), VARIANT, workers=1)
    assert quick.fold_keys(out, VARIANT) == [(item.key.replace("/", "_"), item.key)
                                             for item in loaded]
    members, decoded = quick.decode_folds(features, out, VARIANT, GRID, workers=1, lines=())
    assert [item.key for item in members] == [fold.key for fold in decoded]
    assert {fold.key: fold.scores for fold in decoded} == {fold.key: fold.scores for fold in folds}
    assert all(fold.model_key == fold.key.replace("/", "_") for fold in decoded)
    assert all(fold.dead_labels for fold in decoded)
    _, other = quick.decode_folds(features, out, VARIANT, [DecoderSettings(0.2, 25)], workers=1,
                                  lines=())
    assert all(len(fold.scores) == 1 for fold in other)
    monkeypatch.setitem(train.TEAM_ONLY, VARIANT, frozenset({VARIANT}))
    members, team = quick.decode_folds(features, out, VARIANT, GRID, workers=1, lines=())
    assert not any(item.example.dead_labels for item in members)
    assert not any(fold.dead_labels for fold in team)


def test_compare_reports_counts_folds_won() -> None:
    base = {"held_out": [row("dfl/a", 0.5), row("dfl/b", 0.6), row("metrica/c", 0.7, fair=True)]}
    other = {"held_out": [row("dfl/a", 0.55), row("dfl/b", 0.6), row("metrica/c", 0.65, fair=True),
                          row("dfl/extra", 0.9)]}
    comparison = quick.compare_reports(base, other)
    assert [item["match"] for item in comparison["folds"]] == ["dfl/a", "dfl/b", "metrica/c"]
    dfl = comparison["sources"]["dfl"]
    assert (dfl["won"], dfl["lost"], dfl["folds"]) == (1, 0, 2)
    assert dfl["delta"] == pytest.approx(0.025)
    metrica = comparison["sources"]["metrica"]
    assert (metrica["won"], metrica["lost"], metrica["folds"]) == (0, 1, 1)
    assert metrica["delta"] == pytest.approx(-0.05)
    assert np.isnan(metrica["delta_se"])
    quick.print_comparison(comparison)
