import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from possession.cleaning import CleaningParams
from possession.external.match import ExternalMatch, filled_codes, has_dead_labels, label_codes
from possession.external.predict import to_tracking
from possession.intervals import LABELS
from possession.learned.framefeatures import PERSPECTIVES, base_features, match_inputs
from possession.learned.model import (
    UNLABELLED,
    DecoderSettings,
    LearnedModel,
    TrainingExample,
    in_blocks,
    load_model,
    metadata_path,
    predict_codes,
    read_metadata,
)
from possession.vendor import possession_metrics as pm

FOLD_DIR = "folds"
GROUP_MODEL = "group_model"


def match_key(match: ExternalMatch) -> str:
    return f"{match.source}/{match.name}"


def labelled_codes(match: ExternalMatch) -> np.ndarray:
    filled = filled_codes(match) if has_dead_labels(match) else label_codes(match.labels)
    codes = filled.astype(np.int64)
    codes[(codes == pm.X) | ~in_blocks(match.n_frames, match.blocks)] = UNLABELLED
    return codes


def training_example(match: ExternalMatch, params: CleaningParams, touch: bool = True,
                     lines: tuple[str, ...] = ()) -> TrainingExample:
    inputs = match_inputs(to_tracking(match), match.pitch, match.fps, match.blocks, params, touch,
                          lines)
    return TrainingExample(name=match_key(match),
                           base=tuple(base_features(inputs, first) for first in PERSPECTIVES),
                           codes=labelled_codes(match), blocks=match.blocks, params=params,
                           dead_labels=has_dead_labels(match))


def spilled_array(values: np.ndarray, path: Path) -> np.ndarray:
    np.save(path, values)
    return np.load(path, mmap_mode="r")


def spilled(example: TrainingExample, directory: Path) -> TrainingExample:
    stem = example.name.replace('/', '_')
    touch = example.base[0].touch if example.base else None
    shared = None if touch is None else spilled_array(touch, directory / f"{stem}_touch.npy")
    base = [replace(features, matrix=spilled_array(features.matrix,
                                                   directory / f"{stem}_{first}.npy"),
                    touch=shared)
            for first, features in zip(PERSPECTIVES, example.base)]
    return replace(example, base=tuple(base))


def labels_only(match: ExternalMatch) -> ExternalMatch:
    return replace(match, ball=match.ball[:0].copy(), players=match.players[:0].copy())


def predict_match(model: LearnedModel, match: ExternalMatch) -> np.ndarray:
    codes = predict_codes(model, to_tracking(match), match.pitch, match.fps, match.blocks,
                           match_key(match))
    return LABELS[codes]


def fold_dir(model_path: Path, variant: str) -> Path:
    return model_path.parent / FOLD_DIR / variant


def fold_path(model_path: Path, variant: str, key: str) -> Path:
    return fold_dir(model_path, variant) / f"{key.replace('/', '_')}.pkl"


def save_group_pointer(path: Path, group_path: Path, settings: DecoderSettings, run_id: str,
                       details: dict[str, object]) -> None:
    path.unlink(missing_ok=True)
    metadata = {**details, "run_id": run_id, GROUP_MODEL: group_path.name,
                "emission_weight": settings.emission_weight, "stride": settings.stride,
                "switch_weight": settings.switch_weight}
    metadata_path(path).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def load_fold(path: Path, expected_run: object) -> LearnedModel:
    metadata = read_metadata(path)
    pointer = metadata.get(GROUP_MODEL)
    if pointer is None:
        return load_model(path)
    group_path = path.with_name(str(pointer))
    if not metadata_path(group_path).is_file():
        raise FileNotFoundError(f"{path} points to {group_path}, which is missing; run "
                                "possession.learned.train first")
    found = read_metadata(group_path).get("run_id")
    if found != expected_run:
        raise ValueError(f"{group_path} comes from training run {found}, but {path} from run "
                         f"{expected_run}; retrain with possession.learned.train")
    settings = DecoderSettings(emission_weight=float(metadata["emission_weight"]),
                               stride=int(metadata["stride"]),
                               switch_weight=float(metadata["switch_weight"]))
    return replace(load_model(group_path), settings=settings)


def model_for(match: ExternalMatch, final: LearnedModel, model_path: Path) -> LearnedModel:
    key = match_key(match)
    if key not in final.training_matches:
        return final
    variant = read_metadata(model_path).get("variant")
    if not isinstance(variant, str):
        raise ValueError(f"{model_path.with_suffix('.json')} names no training variant; retrain "
                         "with possession.learned.train")
    path = fold_path(model_path, variant, key)
    if not metadata_path(path).is_file():
        raise FileNotFoundError(f"{key} was used to train {model_path}; its held-out fold model "
                                f"{path} is missing, run possession.learned.train first")
    expected, found = read_metadata(model_path).get("run_id"), read_metadata(path).get("run_id")
    if found != expected:
        raise ValueError(f"{path} comes from training run {found}, but {model_path} from run "
                         f"{expected}; retrain with possession.learned.train")
    group = set(read_metadata(path).get("held_out_group") or [key])
    fold = load_fold(path, expected)
    if key not in group or set(fold.training_matches) != set(final.training_matches) - group:
        raise ValueError(f"{path} was not trained on the other matches of {model_path}")
    return fold
