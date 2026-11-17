import pickle
import sys
from dataclasses import fields
from pathlib import Path

from possession.cleaning import CleaningParams
from possession.params import LEGACY_CLEANING
from possession.paths import DEFAULT_MODEL_PATH

MISSING_PARAMS = ("no cleaning parameters available: pass --legacy for the Phase 1 values, "
                  "--model PATH for a trained model, or run `python -m possession.external."
                  "prepare` to estimate them from the external data (docs/EXTERNAL.md)")
UNREADABLE_MODEL = (pickle.UnpicklingError, AttributeError, ImportError, TypeError, ValueError,
                    EOFError)


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def cached_params() -> CleaningParams | None:
    from possession.external.prepare import cached_reference_params

    try:
        return cached_reference_params()
    except FileNotFoundError:
        return None


def recomputed_params() -> CleaningParams | None:
    from possession.external.prepare import reference_params

    try:
        return reference_params()
    except FileNotFoundError:
        return None


def is_complete(params: object) -> bool:
    return isinstance(params, CleaningParams) and all(
        hasattr(params, field.name) for field in fields(CleaningParams))


def stored_params(path: Path) -> CleaningParams | None:
    from possession.learned.model import load_model

    try:
        params = load_model(path).params
    except UNREADABLE_MODEL as error:
        warn(f"{path} cannot be used ({type(error).__name__}: {error}); using the reference "
             "cleaning parameters estimated from the external data instead")
        return None
    if not is_complete(params):
        warn(f"{path} was trained with an older set of cleaning parameters; using the reference "
             "cleaning parameters estimated from the external data instead")
        return None
    return params


def reference_params() -> CleaningParams:
    params = cached_params() or recomputed_params()
    if params is None:
        raise SystemExit(MISSING_PARAMS)
    return params


def cleaning_params(legacy: bool, model: Path | None) -> CleaningParams:
    if legacy:
        return LEGACY_CLEANING
    stored = model if model is not None else DEFAULT_MODEL_PATH
    if model is not None or stored.is_file():
        params = stored_params(stored)
        if params is not None:
            return params
    return reference_params()
