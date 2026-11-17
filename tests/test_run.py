import argparse
import filecmp
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import OFFICIAL_BASELINE, PROJECT_ROOT, requires_data, requires_official, run_module

from possession import run
from possession.data import graded_blocks, graded_mask
from possession.run import RunConfig, load_config, produce, resolve_config
from possession.submit import sha256_of

BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline.json"
VALIDATOR = PROJECT_ROOT / "src" / "possession" / "vendor" / "validate_submission.py"


def namespace(**values: object) -> argparse.Namespace:
    defaults = {"config": None, "method": None, "window": None, "model": None,
                "no_cache": False, "finalize": False}
    return argparse.Namespace(**{**defaults, **values})


def test_baseline_config_loads() -> None:
    assert load_config(BASELINE_CONFIG) == RunConfig(method="baseline", window=50, use_cache=True)


def test_cli_overrides_config() -> None:
    config = resolve_config(namespace(config=BASELINE_CONFIG, window=25, no_cache=True))
    assert config == RunConfig(method="baseline", window=25, use_cache=False)


def test_unknown_config_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"method": "baseline", "windw": 50}))
    with pytest.raises(ValueError, match="windw"):
        load_config(path)


def test_unknown_method_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"method": "oracle"}))
    with pytest.raises(ValueError, match="oracle"):
        resolve_config(namespace(config=path))


@pytest.mark.parametrize("values", [
    {"window": "50"},
    {"window": 0},
    {"window": -3},
    {"window": 2.5},
    {"window": True},
    {"method": 3},
    {"use_cache": "yes"},
    {"use_cache": 1},
    {"finalize": "false"},
])
def test_config_value_types_rejected(tmp_path: Path, values: dict) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match=next(iter(values))):
        load_config(path)


def test_config_must_be_object(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[1, 2]")
    with pytest.raises(ValueError, match="JSON object"):
        load_config(path)


def test_finalize_cli_flag_overrides_config() -> None:
    config = resolve_config(namespace(config=BASELINE_CONFIG, finalize=True))
    assert config.finalize is True
    assert load_config(BASELINE_CONFIG).finalize is False


def stub_pipeline(monkeypatch: pytest.MonkeyPatch, n_frames: int) -> None:
    labels = np.array(["H", "A", "D", "H", "A", "D"] * n_frames)[:n_frames]
    graded = np.zeros(n_frames, dtype=bool)
    graded[2:5] = True
    monkeypatch.setattr(run, "load_tracking", lambda use_cache=True: None)
    monkeypatch.setitem(run.METHODS, "baseline", lambda tracking, config: labels)
    monkeypatch.setattr(run, "graded_mask", lambda n: graded)


def test_produce_without_finalize_keeps_labels_everywhere(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_pipeline(monkeypatch, 6)
    out = produce(RunConfig(), tmp_path / "plain.csv")
    assert out.read_text().splitlines()[1:] == ["0,H", "1,A", "2,D", "3,H", "4,A", "5,D"]


def test_produce_with_finalize_abstains_outside_graded(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_pipeline(monkeypatch, 6)
    out = produce(RunConfig(finalize=True), tmp_path / "final.csv")
    assert out.read_text().splitlines()[1:] == ["0,X", "1,X", "2,D", "3,H", "4,A", "5,X"]


@requires_data
@requires_official
def test_run_is_deterministic_across_cold_and_warm_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    outputs = [tmp_path / f"run{index}.csv" for index in range(2)]
    for out in outputs:
        run_module("possession.run", "--config", str(BASELINE_CONFIG), "--out", str(out),
                   cache_dir=cache)
    assert len(list(cache.glob("tracking_*.npz"))) == 1
    assert sha256_of(outputs[0]) == sha256_of(outputs[1]) == sha256_of(OFFICIAL_BASELINE)
    assert filecmp.cmp(outputs[0], OFFICIAL_BASELINE, shallow=False)


@requires_official
def test_vendored_validator_accepts_official_baseline() -> None:
    result = subprocess.run([sys.executable, str(VALIDATOR), str(OFFICIAL_BASELINE)],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.startswith("OK")


RULES_CONFIG = PROJECT_ROOT / "configs" / "rules.json"
SECOND_HALF_KICKOFF = 113742
KICKOFF_TOLERANCE_FRAMES = 50


def test_rules_config_loads() -> None:
    config = load_config(RULES_CONFIG)
    assert config.method == "rules" and config.finalize is True
    assert run.rules_params(config.params) == run.rules_params({})


@pytest.mark.parametrize("params", [{"control_s": -1}, {"control_s": "0.5"}, {"bogus": 1.0},
                                    {"gap_s": True}])
def test_rules_params_rejected(tmp_path: Path, params: dict) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"method": "rules", "params": params}))
    with pytest.raises(ValueError, match="params"):
        load_config(path)


def test_zero_is_accepted_only_where_it_is_meaningful() -> None:
    assert run.rules_params({"active_evidence": 0, "out_margin_m": 0}).active_evidence == 0.0
    with pytest.raises(ValueError, match="control_s"):
        run.rules_params({"control_s": 0})


def test_learned_run_forces_dead_before_the_official_kickoffs(monkeypatch) -> None:
    from possession.learned import model as learned_model

    n_frames = 1000
    codes = np.tile(np.array([0, 1]), n_frames // 2)
    monkeypatch.setattr(learned_model, "load_model", lambda path: None)
    monkeypatch.setattr(learned_model, "predict_codes", lambda *args: codes)
    monkeypatch.setattr(run, "graded_blocks", lambda: [(1, 0, 399), (2, 500, 999)])
    monkeypatch.setattr(run, "kickoff_frames", lambda: {1: 10, 2: 700})
    monkeypatch.setattr(run, "pitch_size", lambda: (100.0, 60.0))
    monkeypatch.setattr(run, "fps", lambda: 25)
    forced = run.predict_learned(object(), RunConfig(method="learned"))
    assert (forced[:11] == "D").all() and set(forced[11:500].tolist()) == {"H", "A"}
    assert (forced[500:701] == "D").all() and set(forced[701:].tolist()) == {"H", "A"}
    free = run.predict_learned(object(), RunConfig(method="learned",
                                                   force_prekickoff_dead=False))
    assert set(free.tolist()) == {"H", "A"}


def test_force_prekickoff_dead_must_be_boolean(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"method": "learned", "force_prekickoff_dead": "yes"}))
    with pytest.raises(ValueError, match="force_prekickoff_dead"):
        load_config(path)


def test_params_must_be_object(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"method": "rules", "params": [1]}))
    with pytest.raises(ValueError, match="params"):
        load_config(path)


@requires_data
def test_rules_run_is_deterministic_and_well_formed(tmp_path: Path) -> None:
    outputs = [tmp_path / f"rules{index}.csv" for index in range(2)]
    for out in outputs:
        run_module("possession.run", "--config", str(RULES_CONFIG), "--out", str(out),
                   cache_dir=tmp_path / "cache")
    assert sha256_of(outputs[0]) == sha256_of(outputs[1])
    labels = np.array([line.split(",")[1] for line in outputs[0].read_text().splitlines()[1:]])
    graded = graded_mask(len(labels))
    assert set(np.unique(labels[graded]).tolist()) <= {"H", "A", "D"}
    assert set(np.unique(labels[~graded]).tolist()) == {"X"}
    second_half_start = graded_blocks()[1][1]
    first_live = second_half_start + int(np.flatnonzero(labels[second_half_start:] != "D")[0])
    assert abs(first_live - SECOND_HALF_KICKOFF) <= KICKOFF_TOLERANCE_FRAMES


@requires_data
def test_official_kickoffs_force_the_faq_pre_kickoff_block_dead() -> None:
    from possession.data import kickoff_frames, prekickoff_mask

    blocks = graded_blocks()
    mask = prekickoff_mask(blocks[-1][2] + 1, blocks, kickoff_frames())
    second_half_start = blocks[1][1]
    assert abs(kickoff_frames()[2] - SECOND_HALF_KICKOFF) <= 1
    assert mask[second_half_start:SECOND_HALF_KICKOFF].all()
    assert not mask[SECOND_HALF_KICKOFF + 1]
