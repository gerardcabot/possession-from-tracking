import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from external_synthetic import make_match

from possession.external import predict, prepare, sources, validate
from possession.external.match import ExternalMatch

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


@pytest.mark.parametrize("method", sorted(predict.METHODS))
def test_methods_label_every_frame_with_valid_labels(method: str) -> None:
    match = make_match(n_per_period=1500)
    labels = predict.METHODS[method](match)
    assert labels.shape == (match.n_frames,)
    assert set(np.unique(labels).tolist()) <= {"H", "A", "D"}


def test_score_match_reports_headline_metrics() -> None:
    match = make_match(n_per_period=1500)
    scores = validate.score_match(match, "baseline")
    assert set(scores) == {"spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells"}
    assert 0.0 <= scores["spell_f1_mean"] <= 1.0
    assert scores["n_gt_spells"] == 12


def test_to_tracking_leaves_speeds_unset() -> None:
    match: ExternalMatch = make_match(n_per_period=200, periods=1)
    tracking = predict.to_tracking(match)
    assert np.isnan(tracking.ball_speed).all()
    assert np.isnan(tracking.player_speed).all()


def test_legacy_tracking_speed_matches_displacement() -> None:
    match: ExternalMatch = make_match(n_per_period=200, periods=1)
    tracking = predict.legacy_tracking(match)
    assert tracking.player_speed.shape == match.players.shape[:2]
    assert np.isfinite(tracking.player_speed[5:-5]).all()


def test_discovery_finds_paired_files(tmp_path, monkeypatch) -> None:
    dfl = tmp_path / "dfl"
    dfl.mkdir()
    (dfl / "DFL_04_03_positions_raw_observed_DFL-COM-1_DFL-MAT-AAA.xml").write_text("")
    (dfl / "DFL_02_01_matchinformation_DFL-COM-1_DFL-MAT-AAA.xml").write_text("")
    (dfl / "DFL_04_03_positions_raw_observed_DFL-COM-1_DFL-MAT-BBB.xml").write_text("")
    folder = tmp_path / "skillcorner" / "matches" / "42"
    folder.mkdir(parents=True)
    (folder / "42_match.json").write_text("{}")
    (folder / "42_tracking_extrapolated.jsonl").write_text("")
    monkeypatch.setenv("MTAH_EXTERNAL_DIR", str(tmp_path))
    found = [(files.source, files.name) for files in sources.discover()]
    assert found == [("dfl", "AAA"), ("skillcorner", "42")]


@pytest.mark.parametrize("module", ["possession.run", "possession.external.prepare",
                                    "possession.external.validate"])
def test_kloppy_is_not_imported_by_core_or_validation_modules(module: str) -> None:
    code = f"import sys, {module}; sys.exit(int('kloppy' in sys.modules))"
    env = {**os.environ, "PYTHONPATH": str(SOURCE_ROOT)}
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0


def test_code_fingerprint_depends_on_every_listed_module() -> None:
    one = prepare.code_fingerprint("possession.cleaning")
    two = prepare.code_fingerprint("possession.cleaning", "possession.features")
    assert one != two
    assert two == prepare.code_fingerprint("possession.features", "possession.cleaning")


def test_dependency_lists_cover_the_modules_the_result_reads() -> None:
    for name in ("possession.geometry", "possession.intervals", "possession.external.match",
                 "possession.external.harmonise"):
        assert name in prepare.HARMONISE_MODULES
        assert name in prepare.DEGRADE_MODULES
        assert name in prepare.BOUNDS_MODULES
    for name in ("possession.cleaning", "possession.external.stats"):
        assert name in prepare.STATS_MODULES
        assert name in prepare.DEGRADE_MODULES
        assert name in prepare.BOUNDS_MODULES
    assert "possession.cleaning" not in prepare.HARMONISE_MODULES
    assert "possession.external.degrade" in prepare.DEGRADE_MODULES
    assert "possession.external.bounds" in prepare.BOUNDS_MODULES


def test_prune_removes_only_stale_cache_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MTAH_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("MTAH_EXTERNAL_DIR", str(tmp_path / "none"))
    cache = prepare.external_cache()
    cache.mkdir(parents=True)
    current = [cache / "reference_current.npz", cache / "params_current.npz"]
    monkeypatch.setattr(prepare, "current_reference_files", lambda: {path.name for path in current})
    stale = [cache / "harmonised_dfl_A_old.npz", cache / "degraded_dfl_A_old.npz",
             cache / "reference_old.npz", cache / "params_old.npz"]
    keep = [*current, cache / "digest_abc.json", cache / "other.npz"]
    for path in stale + keep:
        path.write_bytes(b"")
    removed = prepare.prune_stale(object(), 0)
    assert sorted(removed) == sorted(stale)
    assert all(path.exists() for path in keep)
    assert not any(path.exists() for path in stale)
