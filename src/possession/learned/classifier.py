import sys
from dataclasses import dataclass

import numpy as np
from scipy.special import log_expit
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from possession.data import AWAY, HOME
from possession.intervals import DEAD
from possession.learned.framefeatures import Segments
from possession.learned.hsmm import N_STATES, safe_log

THREADS = 1
BINARY = (False, True)
STOPPING_RULES = ("default", "team", "both")
GROUPED = "grouped"
MIN_GROUPS = 2


@dataclass(frozen=True)
class TrainingRows:
    matrix: np.ndarray
    codes: np.ndarray
    timed: np.ndarray
    match: np.ndarray
    run_length: np.ndarray | None = None
    group: np.ndarray | None = None


@dataclass(frozen=True)
class FrameClassifier:
    dead: HistGradientBoostingClassifier
    side: HistGradientBoostingClassifier
    feature_names: tuple[str, ...]
    log_prior: np.ndarray
    dead_columns: tuple[int, ...]
    core_features: int | None = None

    @property
    def observation_limit(self) -> int:
        return len(self.feature_names) if self.core_features is None else self.core_features

    @property
    def iterations(self) -> tuple[int, int]:
        return int(self.dead.n_iter_), int(self.side.n_iter_)

    @property
    def stopping(self) -> tuple[str, str]:
        return stopping_of(self.dead), stopping_of(self.side)

    @property
    def dead_feature_names(self) -> tuple[str, ...]:
        return tuple(self.feature_names[column] for column in self.dead_columns)

    def check(self, matrix: np.ndarray) -> None:
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {matrix.shape[1]}")

    def dead_logit(self, matrix: np.ndarray, segments: Segments | None = None) -> np.ndarray:
        self.check(matrix)
        with threadpool_limits(limits=THREADS):
            return self.dead.decision_function(matrix[:, list(self.dead_columns)])

    def log_posterior(self, matrix: np.ndarray, dead: np.ndarray | None = None,
                      segments: Segments | None = None) -> np.ndarray:
        self.check(matrix)
        dead = self.dead_logit(matrix, segments) if dead is None else dead
        with threadpool_limits(limits=THREADS):
            away = self.side.decision_function(matrix)
        posterior = np.empty((len(matrix), N_STATES))
        alive = log_expit(-dead)
        posterior[:, HOME] = alive + log_expit(-away)
        posterior[:, AWAY] = alive + log_expit(away)
        posterior[:, DEAD] = log_expit(dead)
        return posterior

    def log_emission(self, matrix: np.ndarray, dead: np.ndarray | None = None,
                     segments: Segments | None = None) -> np.ndarray:
        return self.log_posterior(matrix, dead, segments) - self.log_prior


def stopping_of(estimator: HistGradientBoostingClassifier) -> str:
    return GROUPED if estimator.early_stopping is False else STOPPING_RULES[0]


def match_weights(match: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(match, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights * len(match) / weights.sum()


def spell_weights(match: np.ndarray, run_length: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(match, return_inverse=True)
    raw = 1.0 / np.asarray(run_length, dtype=float)
    weights = raw / np.bincount(inverse, weights=raw)[inverse]
    return weights * len(match) / weights.sum()


def check_binary(labels: np.ndarray, task: str) -> None:
    present = np.unique(labels)
    if present.tolist() != list(BINARY):
        raise ValueError(f"{task} training labels must contain both classes, got "
                         f"{present.tolist()}")


def fit_binary(matrix: np.ndarray, labels: np.ndarray, weights: np.ndarray,
               seed: int, task: str) -> tuple[HistGradientBoostingClassifier, float]:
    check_binary(labels, task)
    estimator = HistGradientBoostingClassifier(random_state=seed)
    with threadpool_limits(limits=THREADS):
        estimator.fit(matrix, labels, sample_weight=weights)
    return estimator, float(np.average(labels, weights=weights))


def stopped_iterations(matrix: np.ndarray, labels: np.ndarray, weights: np.ndarray,
                       held: np.ndarray, seed: int) -> int:
    estimator = HistGradientBoostingClassifier(random_state=seed, early_stopping=True,
                                               max_iter=sys.maxsize)
    with threadpool_limits(limits=THREADS):
        estimator.fit(matrix[~held], labels[~held], sample_weight=weights[~held],
                      X_val=matrix[held], y_val=labels[held], sample_weight_val=weights[held])
    return int(np.argmax(estimator.validation_score_))


def usable_group(labels: np.ndarray, held: np.ndarray) -> bool:
    return all(len(np.unique(labels[mask])) == len(BINARY) for mask in (held, ~held))


def fit_grouped(matrix: np.ndarray, labels: np.ndarray, weights: np.ndarray,
                groups: np.ndarray | None, seed: int, task: str
                ) -> tuple[HistGradientBoostingClassifier, float]:
    held_out = [] if groups is None else [groups == group for group in np.unique(groups)]
    held_out = [held for held in held_out if usable_group(labels, held)]
    if len(held_out) < MIN_GROUPS:
        print(f"{task}: {len(held_out)} usable validation groups, grouped early stopping needs "
              f"{MIN_GROUPS}; fitting with the library default iterations", file=sys.stderr,
              flush=True)
        return fit_binary(matrix, labels, weights, seed, task)
    stops = [stopped_iterations(matrix, labels, weights, held, seed) for held in held_out]
    iterations = max(1, round(float(np.mean(stops))))
    print(f"{task}: grouped early stopping {stops} -> {iterations} iterations", file=sys.stderr,
          flush=True)
    estimator = HistGradientBoostingClassifier(random_state=seed, max_iter=iterations,
                                               early_stopping=False)
    with threadpool_limits(limits=THREADS):
        estimator.fit(matrix, labels, sample_weight=weights)
    return estimator, float(np.average(labels, weights=weights))


def selected(matrix: np.ndarray, rows: np.ndarray, columns: tuple[int, ...]) -> np.ndarray:
    index = np.flatnonzero(rows)
    if columns == tuple(range(matrix.shape[1])):
        return matrix[index]
    return matrix[np.ix_(index, columns)]


def row_groups(rows: TrainingRows, mask: np.ndarray) -> np.ndarray | None:
    return None if rows.group is None else rows.group[mask]


def fit_dead_ball(matrix: np.ndarray, labels: np.ndarray, weights: np.ndarray,
                  groups: np.ndarray | None, seed: int, stopping: str,
                  task: str = "dead-ball") -> tuple[HistGradientBoostingClassifier, float]:
    if stopping == "both":
        return fit_grouped(matrix, labels, weights, groups, seed, task)
    return fit_binary(matrix, labels, weights, seed, task)


def fit_classifier(rows: TrainingRows, names: list[str], seed: int,
                   dead_columns: tuple[int, ...] | None = None,
                   core_features: int | None = None, spell: bool = False,
                   stopping: str = STOPPING_RULES[0]) -> FrameClassifier:
    if spell and rows.run_length is None:
        raise ValueError("spell weights need the run length of every training row")
    limit = len(names) if core_features is None else core_features
    if not 0 < limit <= len(names):
        raise ValueError(f"core_features {core_features} must be in 1..{len(names)}")
    columns = tuple(range(limit)) if dead_columns is None else tuple(dead_columns)
    if not columns or not set(columns) <= set(range(limit)):
        raise ValueError(f"dead-ball columns {columns} do not index {limit} core features")
    labelled = rows.codes >= 0
    timed = labelled & rows.timed
    side = labelled & (rows.codes != DEAD)
    if not timed.any():
        raise ValueError(f"training needs frames with dead-ball labels to cover all "
                         f"{N_STATES} states")
    dead_matrix, dead_labels = selected(rows.matrix, timed, columns), rows.codes[timed] == DEAD
    dead_weights = match_weights(rows.match[timed])
    dead, dead_share = fit_dead_ball(dead_matrix, dead_labels, dead_weights,
                                     row_groups(rows, timed), seed, stopping)
    side_labels = rows.codes[side] == AWAY
    side_weights = (spell_weights(rows.match[side], rows.run_length[side])
                    if rows.run_length is not None and spell else match_weights(rows.match[side]))
    if stopping == "default":
        away, away_share = fit_binary(rows.matrix[side], side_labels, side_weights, seed, "team")
    else:
        away, away_share = fit_grouped(rows.matrix[side], side_labels, side_weights,
                                       row_groups(rows, side), seed, "team")
    prior = np.empty(N_STATES)
    prior[HOME] = (1.0 - dead_share) * (1.0 - away_share)
    prior[AWAY] = (1.0 - dead_share) * away_share
    prior[DEAD] = dead_share
    return FrameClassifier(dead=dead, side=away, feature_names=tuple(names),
                           log_prior=safe_log(prior), dead_columns=columns,
                           core_features=core_features)
