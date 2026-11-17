from dataclasses import fields
from pathlib import Path

import pytest

from possession import cleaning_params
from possession.cleaning import CleaningParams
from possession.params import LEGACY_CLEANING

STORED = CleaningParams(ball_max_speed_mps=30.0, ball_deviation_m=1.6,
                        player_max_speed_mps=10.0, player_deviation_m=0.3,
                        duplicate_distance_m=0.4, max_gap_frames=6, ball_half_span=1,
                        player_half_span=1)


class StoredModel:
    params = STORED


def test_legacy_flag_returns_phase_one_values() -> None:
    assert cleaning_params.cleaning_params(True, None) == LEGACY_CLEANING


def test_explicit_model_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from possession.learned import model

    monkeypatch.setattr(model, "load_model", lambda path: StoredModel())
    assert cleaning_params.cleaning_params(False, tmp_path / "m.pkl") == STORED


def test_default_model_is_used_when_present(monkeypatch: pytest.MonkeyPatch,
                                            tmp_path: Path) -> None:
    from possession.learned import model

    default = tmp_path / "model.pkl"
    default.write_bytes(b"")
    monkeypatch.setattr(cleaning_params, "DEFAULT_MODEL_PATH", default)
    monkeypatch.setattr(model, "load_model", lambda path: StoredModel())
    assert cleaning_params.cleaning_params(False, None) == STORED


def test_cached_params_are_used_without_a_model(monkeypatch: pytest.MonkeyPatch,
                                                tmp_path: Path) -> None:
    monkeypatch.setattr(cleaning_params, "DEFAULT_MODEL_PATH", tmp_path / "missing.pkl")
    monkeypatch.setattr(cleaning_params, "cached_params", lambda: STORED)
    assert cleaning_params.cleaning_params(False, None) == STORED


def test_missing_params_give_a_clear_error(monkeypatch: pytest.MonkeyPatch,
                                           tmp_path: Path) -> None:
    monkeypatch.setattr(cleaning_params, "DEFAULT_MODEL_PATH", tmp_path / "missing.pkl")
    monkeypatch.setattr(cleaning_params, "cached_params", lambda: None)
    monkeypatch.setattr(cleaning_params, "recomputed_params", lambda: None)
    with pytest.raises(SystemExit, match="--legacy"):
        cleaning_params.cleaning_params(False, None)


def old_params() -> CleaningParams:
    params = object.__new__(CleaningParams)
    for field in fields(CleaningParams):
        if field.name not in ("ball_deviation_m", "player_deviation_m"):
            object.__setattr__(params, field.name, getattr(STORED, field.name))
    return params


def test_model_with_params_missing_fields_falls_back_with_a_warning(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from possession.learned import model

    class OldModel:
        params = old_params()

    monkeypatch.setattr(model, "load_model", lambda path: OldModel())
    monkeypatch.setattr(cleaning_params, "cached_params", lambda: STORED)
    assert cleaning_params.cleaning_params(False, tmp_path / "m.pkl") == STORED
    assert "older set of cleaning parameters" in capsys.readouterr().err


@pytest.mark.parametrize("error", [ValueError("model does not match model.json"),
                                   AttributeError("'CleaningParams' has no ball_deviation_m")])
def test_unreadable_model_falls_back_with_a_warning(error: Exception,
                                                    monkeypatch: pytest.MonkeyPatch,
                                                    tmp_path: Path,
                                                    capsys: pytest.CaptureFixture) -> None:
    from possession.learned import model

    def fail(path: Path) -> None:
        raise error

    monkeypatch.setattr(model, "load_model", fail)
    monkeypatch.setattr(cleaning_params, "cached_params", lambda: STORED)
    assert cleaning_params.cleaning_params(False, tmp_path / "m.pkl") == STORED
    assert "cannot be used" in capsys.readouterr().err


def test_fallback_recomputes_when_nothing_is_cached(monkeypatch: pytest.MonkeyPatch,
                                                    tmp_path: Path) -> None:
    from possession.learned import model

    class OldModel:
        params = old_params()

    monkeypatch.setattr(model, "load_model", lambda path: OldModel())
    monkeypatch.setattr(cleaning_params, "cached_params", lambda: None)
    monkeypatch.setattr(cleaning_params, "recomputed_params", lambda: STORED)
    assert cleaning_params.cleaning_params(False, tmp_path / "m.pkl") == STORED
