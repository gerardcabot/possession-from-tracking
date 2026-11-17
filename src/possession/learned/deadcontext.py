import sys
from dataclasses import dataclass, field, fields

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from possession.intervals import DEAD
from possession.learned.classifier import (
    THREADS,
    FrameClassifier,
    TrainingRows,
    fit_dead_ball,
    match_weights,
    stopping_of,
)
from possession.learned.framefeatures import (
    CENTRED,
    ROLLING_DIRECTIONS,
    BaseFeatures,
    FeatureLayout,
    Rolling,
    Segments,
    feature_matrix,
    rolling_column,
)

CONTEXT_KINDS = (CENTRED, *ROLLING_DIRECTIONS)
CONTEXT_SIGNAL = "dead_logit"


def context_specs(scales: tuple[int, ...]) -> list[Rolling]:
    return [Rolling(CONTEXT_SIGNAL, kind, scale) for kind in CONTEXT_KINDS for scale in scales]


def context_names(scales: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(spec.name for spec in context_specs(scales))


def context_columns(logit: np.ndarray, segments: Segments, scales: tuple[int, ...]) -> np.ndarray:
    return np.stack([rolling_column(segments, logit, spec) for spec in context_specs(scales)],
                    axis=1).astype(np.float32)


def stage_two_matrix(matrix: np.ndarray, columns: tuple[int, ...], stage_one: np.ndarray,
                     segments: Segments, scales: tuple[int, ...]) -> np.ndarray:
    stage_two = np.empty((len(matrix), len(columns) + len(context_names(scales))),
                         dtype=np.float32)
    stage_two[:, :len(columns)] = matrix[:, list(columns)]
    stage_two[:, len(columns):] = context_columns(stage_one, segments, scales)
    return stage_two


@dataclass(frozen=True)
class DeadContext:
    estimator: HistGradientBoostingClassifier
    scales: tuple[int, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return context_names(self.scales)


@dataclass(frozen=True)
class ContextClassifier(FrameClassifier):
    context: DeadContext = field(kw_only=True)

    def stage_one_logit(self, matrix: np.ndarray) -> np.ndarray:
        return super().dead_logit(matrix)

    def dead_logit(self, matrix: np.ndarray, segments: Segments | None = None) -> np.ndarray:
        if segments is None:
            raise ValueError("a dead-ball context classifier needs the block segments of the "
                             "match to build its context columns")
        stage_two = stage_two_matrix(matrix, self.dead_columns, self.stage_one_logit(matrix),
                                     segments, self.context.scales)
        with threadpool_limits(limits=THREADS):
            return self.context.estimator.decision_function(stage_two)


def with_dead_context(classifier: FrameClassifier, context: DeadContext) -> ContextClassifier:
    return ContextClassifier(**{item.name: getattr(classifier, item.name)
                                for item in fields(classifier)}, context=context)


def context_metadata(classifier: FrameClassifier) -> dict[str, object]:
    if not isinstance(classifier, ContextClassifier):
        return {}
    context = classifier.context
    return {"dead_context": {"scales": list(context.scales), "columns": len(context.names),
                             "iterations": int(context.estimator.n_iter_),
                             "stopping": stopping_of(context.estimator)}}


def check_split(labels: np.ndarray, group: int) -> None:
    if len(np.unique(labels)) != 2:
        raise ValueError(f"dead-ball context: the training matches outside validation group "
                         f"{group} do not contain both dead-ball classes; it needs at least two "
                         f"validation groups with dead-ball labels")


def out_of_fold_logits(dead_matrix: np.ndarray, labels: np.ndarray, match: np.ndarray,
                       group: np.ndarray, layout: FeatureLayout, columns: tuple[int, ...],
                       bases: list[tuple[BaseFeatures, ...]], seed: int, stopping: str
                       ) -> dict[int, list[np.ndarray]]:
    logits: dict[int, list[np.ndarray]] = {}
    for held in np.unique(group):
        fit = group != held
        check_split(labels[fit], int(held))
        estimator, _ = fit_dead_ball(dead_matrix[fit], labels[fit], match_weights(match[fit]),
                                     group[fit], seed, stopping,
                                     "dead-ball (leave one validation group out)")
        for example in np.unique(match[group == held]):
            inputs = [feature_matrix(base, layout.core, columns=columns)
                      for base in bases[int(example)]]
            with threadpool_limits(limits=THREADS):
                logits[int(example)] = [estimator.decision_function(values) for values in inputs]
    return logits


def fill_context(stage_two: np.ndarray, first: int, timed: np.ndarray,
                 logits: dict[int, list[np.ndarray]], bases: list[tuple[BaseFeatures, ...]],
                 keeps: list[np.ndarray], scales: tuple[int, ...]) -> None:
    row, written = 0, 0
    for example, (example_bases, keep) in enumerate(zip(bases, keeps)):
        kept = int(keep.sum())
        for position, base in enumerate(example_bases):
            used = timed[row:row + kept]
            count = int(used.sum())
            if count:
                context = context_columns(logits[example][position], base.segments, scales)
                stage_two[written:written + count, first:] = context[keep][used]
                written += count
            row += kept
    if row != len(timed) or written != len(stage_two):
        raise ValueError(f"context covers {written} of {len(stage_two)} dead-ball rows and "
                         f"{row} of {len(timed)} training rows")


def fit_dead_context(classifier: FrameClassifier, rows: TrainingRows, layout: FeatureLayout,
                     bases: list[tuple[BaseFeatures, ...]], keeps: list[np.ndarray], seed: int,
                     stopping: str) -> ContextClassifier:
    if rows.group is None:
        raise ValueError("dead-ball context needs validation groups of whole matches, which "
                         "need at least two training matches")
    columns, scales = classifier.dead_columns, layout.all_scales
    timed = (rows.codes >= 0) & rows.timed
    index = np.flatnonzero(timed)
    stage_two = np.empty((len(index), len(columns) + len(context_names(scales))),
                         dtype=np.float32)
    for column, source in enumerate(columns):
        stage_two[:, column] = rows.matrix[index, source]
    labels, match, group = rows.codes[index] == DEAD, rows.match[index], rows.group[index]
    dead_matrix = stage_two[:, :len(columns)]
    logits = out_of_fold_logits(dead_matrix, labels, match, group, layout, columns, bases, seed,
                                stopping)
    fill_context(stage_two, len(columns), timed, logits, bases, keeps, scales)
    estimator, _ = fit_dead_ball(stage_two, labels, match_weights(match), group, seed, stopping,
                                 "dead-ball-context")
    print(f"dead-ball context: {len(np.unique(group))} validation-group fits, {len(scales)} "
          f"scales, {stage_two.shape[1]} features, {int(estimator.n_iter_)} iterations",
          file=sys.stderr, flush=True)
    return with_dead_context(classifier, DeadContext(estimator=estimator, scales=scales))
