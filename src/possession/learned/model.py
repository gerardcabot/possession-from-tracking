import hashlib
import json
import pickle
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import sklearn

from possession.cleaning import CleaningParams
from possession.data import AWAY, HOME, Tracking
from possession.intervals import DEAD, value_runs
from possession.learned.adapt import AdaptationReport, adapted_dead_logit
from possession.learned.changes import ChangeModel, change_rows, fit_change_model
from possession.learned.classifier import (
    MIN_GROUPS,
    STOPPING_RULES,
    FrameClassifier,
    TrainingRows,
    fit_classifier,
)
from possession.learned.deadcontext import context_metadata, fit_dead_context
from possession.learned.framefeatures import (
    DEAD_FEATURE_SETS,
    DEFAULT_TEAM_FEATURES,
    PERSPECTIVES,
    BaseFeatures,
    Blocks,
    FeatureLayout,
    base_features,
    dead_columns,
    feature_layout,
    feature_matrix,
    feature_names,
    line_groups_in,
    match_inputs,
    peak_evidence,
    youden_threshold,
)
from possession.learned.hsmm import (
    SegmentModel,
    decode_block,
    fit_segment_model,
    step_switch,
    weighted_runs,
)

MODEL_FORMAT = "learned-v8"
MODEL_SEED = 0
UNLABELLED = -1
GIGABYTE = 1e9
EVERY_PEAK = 0.0
NO_TOUCH_EVIDENCE_THRESHOLD = EVERY_PEAK


@dataclass(frozen=True)
class DecoderSettings:
    emission_weight: float
    stride: int
    switch_weight: float = 0.0


@dataclass(frozen=True)
class DeadBallSettings:
    features: str = "all"
    adaptation_iterations: int = 0

    def __post_init__(self) -> None:
        if self.features not in DEAD_FEATURE_SETS:
            raise ValueError(f"unknown dead-ball feature set {self.features!r}; choose from "
                             f"{DEAD_FEATURE_SETS}")
        if self.adaptation_iterations < 0:
            raise ValueError(f"adaptation_iterations must be >= 0, got "
                             f"{self.adaptation_iterations}")
        if self.adaptation_iterations > 0 and self.features != "players":
            raise ValueError("adaptation needs dead_features 'players': with all features there "
                             "is no ball-observation evidence left to adapt")


@dataclass(frozen=True)
class ClassifierSettings:
    spell_weights: bool = False
    every_peak: bool = False
    stopping: str = STOPPING_RULES[0]
    dead_context: bool = False

    def __post_init__(self) -> None:
        if self.stopping not in STOPPING_RULES:
            raise ValueError(f"unknown stopping rule {self.stopping!r}; choose from "
                             f"{STOPPING_RULES}")


DEFAULT_FITTING = ClassifierSettings()


@dataclass(frozen=True)
class TrainingExample:
    name: str
    base: tuple[BaseFeatures, ...]
    codes: np.ndarray
    blocks: Blocks
    params: CleaningParams
    dead_labels: bool
    row_step: int = 1


@dataclass(frozen=True)
class LearnedModel:
    classifier: FrameClassifier
    segments: SegmentModel
    layout: FeatureLayout
    settings: DecoderSettings
    training_matches: tuple[str, ...]
    params: CleaningParams
    dead_ball: DeadBallSettings
    touch_peaks: int = 0
    changes: ChangeModel | None = None


@dataclass(frozen=True)
class TouchEvidence:
    threshold: float
    peaks: int


@dataclass(frozen=True)
class Entering:
    frames: np.ndarray
    steps: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)

    def block(self, first: int, last: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
        frames = self.frames[first:last + 1]
        if (first, stride) not in self.steps:
            self.steps[first, stride] = step_switch(frames, stride)
        return frames, self.steps[first, stride]


def swap_teams(codes: np.ndarray) -> np.ndarray:
    swapped = codes.copy()
    swapped[codes == HOME] = AWAY
    swapped[codes == AWAY] = HOME
    return swapped


def perspective_codes(codes: np.ndarray, first: int) -> np.ndarray:
    return codes if first == HOME else swap_teams(codes)


def block_codes(codes: np.ndarray, blocks: Blocks) -> list[np.ndarray]:
    return [codes[first:last + 1] for _, first, last in blocks]


def labelled_sequences(example: TrainingExample) -> list[np.ndarray]:
    sequences = []
    for sequence in block_codes(example.codes, example.blocks):
        if (sequence == UNLABELLED).any():
            raise ValueError(f"{example.name}: unlabelled frames inside a period")
        sequences += [perspective_codes(sequence, first) for first in PERSPECTIVES]
    return sequences


def timed_sequences(examples: list[TrainingExample]) -> list[list[np.ndarray]]:
    matches = [labelled_sequences(example) for example in examples if example.dead_labels]
    if not matches:
        raise ValueError("training needs at least one match with dead-ball labels for the "
                         "durations and transitions")
    return matches


def in_blocks(n_frames: int, blocks: Blocks) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        mask[first:last + 1] = True
    return mask


def training_keep(example: TrainingExample, frame_step: int) -> np.ndarray:
    keep = in_blocks(len(example.codes), example.blocks) & (example.codes != UNLABELLED)
    if not example.dead_labels:
        keep &= example.codes != DEAD
    keep[np.arange(len(keep)) % (frame_step * example.row_step) != 0] = False
    return keep


def run_lengths(codes: np.ndarray, blocks: Blocks) -> np.ndarray:
    lengths = np.zeros(len(codes), dtype=np.int64)
    for _, first, last in blocks:
        starts, ends, _ = value_runs(codes[first:last + 1])
        lengths[first:last + 1] = np.repeat(ends - starts, ends - starts)
    return lengths


def source_of(example: TrainingExample) -> str:
    return example.name.split("/")[0]


def validation_groups(examples: list[TrainingExample]) -> np.ndarray | None:
    if len(examples) < MIN_GROUPS:
        return None
    by_source: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        by_source.setdefault(source_of(example), []).append(index)
    count = max(MIN_GROUPS, min(len(indices) for indices in by_source.values()))
    groups = np.empty(len(examples), dtype=np.int64)
    for indices in by_source.values():
        ranked = sorted(indices, key=lambda index: examples[index].name)
        groups[ranked] = np.arange(len(ranked)) % count
    return groups


def training_rows(examples: list[TrainingExample], layout: FeatureLayout,
                  frame_step: int, groups: np.ndarray | None = None,
                  spell: bool = False) -> TrainingRows:
    keeps = [training_keep(example, frame_step) for example in examples]
    total = sum(int(keep.sum()) * len(example.base) for example, keep in zip(examples, keeps))
    matrix = np.empty((total, len(feature_names(examples[0].base[0], layout))), dtype=np.float32)
    labels, timed, match, lengths, group = [], [], [], [], []
    cursor = 0
    for index, (example, keep) in enumerate(zip(examples, keeps)):
        kept = int(keep.sum())
        runs = run_lengths(example.codes, example.blocks)[keep] if spell else None
        for first, base in zip(PERSPECTIVES, example.base):
            matrix[cursor:cursor + kept] = feature_matrix(base, layout, keep)
            cursor += kept
            labels.append(perspective_codes(example.codes, first)[keep])
            timed.append(np.full(kept, example.dead_labels))
            match.append(np.full(kept, index, dtype=np.int64))
            if runs is not None:
                lengths.append(runs)
            if groups is not None:
                group.append(np.full(kept, groups[index], dtype=np.int64))
    return TrainingRows(matrix=matrix, codes=np.concatenate(labels).astype(np.int64),
                        timed=np.concatenate(timed), match=np.concatenate(match),
                        run_length=np.concatenate(lengths) if lengths else None,
                        group=np.concatenate(group) if groups is not None else None)


def log_training_rows(rows: TrainingRows) -> None:
    print(f"classifier rows {len(rows.matrix)} x {rows.matrix.shape[1]} features "
          f"({rows.matrix.nbytes / GIGABYTE:.2f} GB)", file=sys.stderr, flush=True)


def shared_params(examples: list[TrainingExample]) -> CleaningParams:
    distinct = {example.params for example in examples}
    if len(distinct) != 1:
        raise ValueError(f"training examples were cleaned with {len(distinct)} different "
                         "parameter sets")
    return distinct.pop()


def touch_threshold(examples: list[TrainingExample]) -> TouchEvidence:
    evidence = [peak_evidence(example.base[0], example.codes, example.blocks)
                for example in examples]
    evidence = [(scores, changes) for scores, changes in evidence if len(scores)]
    if not evidence:
        print(f"touch threshold: no labelled impulse peak in {len(examples)} training matches; "
              f"falling back to {NO_TOUCH_EVIDENCE_THRESHOLD} (every peak is a touch)",
              file=sys.stderr, flush=True)
        return TouchEvidence(threshold=NO_TOUCH_EVIDENCE_THRESHOLD, peaks=0)
    weights = np.concatenate([np.full(len(scores), 1.0 / len(scores)) for scores, _ in evidence])
    threshold = youden_threshold(np.concatenate([scores for scores, _ in evidence]),
                                 np.concatenate([changes for _, changes in evidence]), weights)
    return TouchEvidence(threshold=threshold, peaks=len(weights))


def check_touch_inputs(examples: list[TrainingExample]) -> None:
    missing = [example.name for example in examples
               if any(base.touch is None for base in example.base)]
    if missing:
        raise ValueError(f"touch team features and the switch prior need touch inputs; missing "
                         f"for {missing}")


def fit_changes(examples: list[TrainingExample], frame_step: int,
                evidence: TouchEvidence) -> ChangeModel | None:
    parts = [change_rows(example.base[0], example.codes, example.blocks,
                         training_keep(example, frame_step), evidence.threshold)
             for example in examples if example.dead_labels]
    return fit_change_model(parts, evidence.threshold, evidence.peaks, MODEL_SEED)


def fit_model(examples: list[TrainingExample], settings: DecoderSettings,
              frame_step: int = 1, dead_ball: DeadBallSettings = DeadBallSettings(),
              team_features: str = DEFAULT_TEAM_FEATURES,
              switch_prior: bool = True,
              fitting: ClassifierSettings = DEFAULT_FITTING) -> LearnedModel:
    params = shared_params(examples)
    matches = timed_sequences(examples)
    evidence = TouchEvidence(threshold=0.0, peaks=0)
    if team_features == "touch" or switch_prior:
        check_touch_inputs(examples)
        evidence = touch_threshold(examples)
    changes = fit_changes(examples, frame_step, evidence) if switch_prior else None
    team_threshold = EVERY_PEAK if fitting.every_peak else evidence.threshold
    layout = feature_layout(team_features, *weighted_runs(matches), team_threshold)
    grouped = fitting.stopping != STOPPING_RULES[0]
    rows = training_rows(examples, layout, frame_step,
                         validation_groups(examples) if grouped or fitting.dead_context else None,
                         fitting.spell_weights)
    log_training_rows(rows)
    names = feature_names(examples[0].base[0], layout)
    core = feature_names(examples[0].base[0], layout.core)
    classifier = fit_classifier(rows, names, MODEL_SEED, dead_columns(core, dead_ball.features),
                                len(core), fitting.spell_weights, fitting.stopping)
    if fitting.dead_context:
        classifier = fit_dead_context(classifier, rows, layout,
                                      [example.base for example in examples],
                                      [training_keep(example, frame_step) for example in examples],
                                      MODEL_SEED, fitting.stopping)
    return LearnedModel(classifier=classifier, segments=fit_segment_model(matches), layout=layout,
                        settings=settings,
                        training_matches=tuple(example.name for example in examples),
                        params=params, dead_ball=dead_ball, touch_peaks=evidence.peaks,
                        changes=changes)


def log_emission(model: LearnedModel, base: BaseFeatures, blocks: Blocks
                 ) -> tuple[np.ndarray, AdaptationReport | None]:
    matrix = feature_matrix(base, model.layout)
    if model.dead_ball.adaptation_iterations <= 0:
        return model.classifier.log_emission(matrix, segments=base.segments), None
    dead, report = adapted_dead_logit(model.classifier, matrix, blocks, MODEL_SEED,
                                      model.dead_ball.adaptation_iterations, segments=base.segments)
    return model.classifier.log_emission(matrix, dead), report


def entry_evidence(model: LearnedModel, base: BaseFeatures) -> Entering | None:
    return None if model.changes is None else Entering(model.changes.entry_evidence(base))


def effective_settings(model: LearnedModel, settings: DecoderSettings) -> DecoderSettings:
    return settings if model.changes is not None else replace(settings, switch_weight=0.0)


def for_inference(model: LearnedModel, settings: DecoderSettings) -> LearnedModel:
    settings = effective_settings(model, settings)
    return replace(model, settings=settings,
                   changes=model.changes if settings.switch_weight else None)


def log_adaptation(name: str, report: AdaptationReport | None) -> None:
    if report is not None:
        print(f"{name}: {report.describe()}", file=sys.stderr, flush=True)


def decode_emission(emission: np.ndarray, blocks: Blocks, segments: SegmentModel,
                    settings: DecoderSettings, entering: Entering | None = None) -> np.ndarray:
    if settings.switch_weight and entering is None:
        raise ValueError(f"switch weight {settings.switch_weight} needs a model fitted with the "
                         "switch prior")
    codes = np.full(len(emission), DEAD, dtype=np.int8)
    for _, first, last in blocks:
        weighted = settings.emission_weight * emission[first:last + 1]
        if not settings.switch_weight or entering is None:
            codes[first:last + 1] = decode_block(weighted, segments, settings.stride)
            continue
        frames, steps = entering.block(first, last, settings.stride)
        codes[first:last + 1] = decode_block(weighted, segments, settings.stride,
                                             settings.switch_weight * frames,
                                             settings.switch_weight * steps)
    return codes


def predict_codes(model: LearnedModel, tracking: Tracking, pitch: tuple[float, float],
                  fps: float, blocks: Blocks, name: str = "match") -> np.ndarray:
    model = for_inference(model, model.settings)
    inputs = match_inputs(tracking, pitch, fps, blocks, model.params,
                          model.layout.touch or model.changes is not None,
                          line_groups_in(model.classifier.feature_names))
    base = base_features(inputs, HOME)
    emission, report = log_emission(model, base, blocks)
    log_adaptation(name, report)
    return decode_emission(emission, blocks, model.segments, model.settings,
                           entry_evidence(model, base))


def model_bytes(model: LearnedModel) -> bytes:
    return pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)


def metadata_path(path: Path) -> Path:
    return path.with_suffix(".json")


def model_digest(model: LearnedModel) -> str:
    return hashlib.sha256(model_bytes(model)).hexdigest()


def model_metadata(model: LearnedModel, payload: bytes, run_id: str,
                   details: dict[str, object]) -> dict[str, object]:
    return {
        **details,
        "format": MODEL_FORMAT,
        "run_id": run_id,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "sklearn": sklearn.__version__,
        "emission_weight": model.settings.emission_weight,
        "stride": model.settings.stride,
        "switch_weight": model.settings.switch_weight,
        "switch_prior": model.changes is not None,
        "switch_prior_rows": None if model.changes is None else model.changes.rows,
        "switch_prior_threshold": None if model.changes is None else model.changes.threshold,
        "switch_prior_peaks": None if model.changes is None else model.changes.peaks,
        "scales": list(model.layout.scales),
        "short_scales": list(model.layout.short_scales),
        "touch_features": model.layout.touch,
        "touch_threshold": model.layout.touch_threshold,
        "touch_evidence_peaks": model.touch_peaks,
        "features": len(model.classifier.feature_names),
        "iterations": list(model.classifier.iterations),
        "stopping": list(model.classifier.stopping),
        "training_matches": list(model.training_matches),
        "cleaning": model.params.to_dict(),
        "dead_ball": asdict(model.dead_ball),
        "dead_ball_features": len(model.classifier.dead_columns),
        **line_metadata(line_groups_in(model.classifier.feature_names)),
        **context_metadata(model.classifier),
    }


def line_metadata(groups: tuple[str, ...]) -> dict[str, object]:
    return {"line_features": list(groups)} if groups else {}


def save_model(path: Path, model: LearnedModel, run_id: str,
               details: dict[str, object] | None = None) -> str:
    payload = model_bytes(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    metadata = model_metadata(model, payload, run_id, details or {})
    metadata_path(path).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return str(metadata["sha256"])


def read_metadata(path: Path) -> dict[str, object]:
    return json.loads(metadata_path(path).read_text())


def load_model(path: Path) -> LearnedModel:
    payload = path.read_bytes()
    metadata = read_metadata(path)
    digest = hashlib.sha256(payload).hexdigest()
    if metadata.get("format") != MODEL_FORMAT or metadata.get("sha256") != digest:
        raise ValueError(f"{path}: model does not match {metadata_path(path).name}")
    if metadata.get("sklearn") != sklearn.__version__:
        raise ValueError(f"{path}: trained with scikit-learn {metadata.get('sklearn')}, "
                         f"running {sklearn.__version__}")
    return pickle.loads(payload)
