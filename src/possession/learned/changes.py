import sys
from dataclasses import dataclass

import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from possession.data import AWAY, HOME
from possession.learned.classifier import THREADS, fit_binary, match_weights
from possession.learned.framefeatures import (
    TOUCH_COLUMNS,
    TOUCH_EVENTS,
    BaseFeatures,
    Blocks,
    other_perspective,
    touch_events,
    touch_half_span,
)
from possession.learned.hsmm import N_STATES

CHANGE_FEATURES = ("ball_present", "nearest_own", "nearest_other", *TOUCH_COLUMNS, *TOUCH_EVENTS)
TEAMS = (HOME, AWAY)


@dataclass(frozen=True)
class ChangeModel:
    estimator: HistGradientBoostingClassifier
    log_prior_odds: float
    threshold: float
    peaks: int
    rows: int

    def log_odds_ratio(self, columns: dict[str, np.ndarray]) -> np.ndarray:
        with threadpool_limits(limits=THREADS):
            return self.estimator.decision_function(change_matrix(columns)) - self.log_prior_odds

    def entry_evidence(self, base: BaseFeatures) -> np.ndarray:
        evidence = np.zeros((len(base.matrix), N_STATES))
        for team, columns in team_columns(base, self.threshold).items():
            evidence[:, team] = self.log_odds_ratio(columns)
        return evidence


def opponent(team: int) -> int:
    return AWAY if team == HOME else HOME


def change_columns(base: BaseFeatures, threshold: float) -> dict[str, np.ndarray]:
    columns = touch_events(base, threshold)
    for name in ("ball_present", "nearest_own", "nearest_other", *TOUCH_COLUMNS):
        columns[name] = base.column(name).astype(float)
    return columns


def team_columns(base: BaseFeatures, threshold: float) -> dict[int, dict[str, np.ndarray]]:
    columns = change_columns(base, threshold)
    return {base.first: columns, opponent(base.first): other_perspective(columns)}


def change_matrix(columns: dict[str, np.ndarray], keep: np.ndarray | None = None) -> np.ndarray:
    rows = slice(None) if keep is None else keep
    return np.stack([np.asarray(columns[name], dtype=float)[rows] for name in CHANGE_FEATURES],
                    axis=1).astype(np.float32)


def near_entries(codes: np.ndarray, team: int, blocks: Blocks, span: int) -> np.ndarray:
    n_frames = len(codes)
    near = np.zeros(n_frames + 1, dtype=np.int64)
    for _, start, last in blocks:
        first, end = min(max(start, 0), n_frames), min(max(last + 1, 0), n_frames)
        sequence = codes[first:end]
        entries = np.flatnonzero((sequence[1:] == team) & (sequence[:-1] == opponent(team)))
        entries += first + 1
        np.add.at(near, np.maximum(entries - span, first), 1)
        np.add.at(near, np.minimum(entries + span + 1, end), -1)
    return np.cumsum(near[:-1]) > 0


def change_rows(base: BaseFeatures, codes: np.ndarray, blocks: Blocks, keep: np.ndarray,
                threshold: float) -> tuple[np.ndarray, np.ndarray]:
    span = touch_half_span(base.fps)
    alive = keep & np.isin(codes, TEAMS)
    matrices, labels = [], []
    for team, columns in team_columns(base, threshold).items():
        matrices.append(change_matrix(columns, alive))
        labels.append(near_entries(codes, team, blocks, span)[alive])
    return np.concatenate(matrices), np.concatenate(labels)


def fit_change_model(parts: list[tuple[np.ndarray, np.ndarray]], threshold: float, peaks: int,
                     seed: int) -> ChangeModel | None:
    labels = (np.concatenate([values for _, values in parts]) if parts
              else np.zeros(0, dtype=bool))
    if labels.all() or not labels.any():
        print(f"switch prior: {int(labels.sum())} of {len(labels)} training rows are near a "
              f"change; both classes are needed, so this fit has no switch prior (switch weight "
              f"0 only)", file=sys.stderr, flush=True)
        return None
    matrix = np.concatenate([rows for rows, _ in parts])
    match = np.concatenate([np.full(len(values), index) for index, (_, values) in
                            enumerate(parts)])
    estimator, share = fit_binary(matrix, labels, match_weights(match), seed, "switch prior")
    return ChangeModel(estimator=estimator, log_prior_odds=float(logit(share)),
                       threshold=threshold, peaks=peaks, rows=len(labels))
