from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from possession.intervals import DEAD
from possession.learned.classifier import THREADS, FrameClassifier
from possession.learned.framefeatures import Blocks, Segments

MATCH_HALVES = 2
SAFETY_BOUND = 20


@dataclass(frozen=True)
class AdaptationReport:
    shares: tuple[float, ...]
    converged: bool

    @property
    def iterations(self) -> int:
        return max(len(self.shares) - 1, 0)

    @property
    def final_share(self) -> float | None:
        return self.shares[-1] if self.shares else None

    def to_dict(self) -> dict[str, object]:
        return {"iterations": self.iterations, "converged": self.converged,
                "final_share": self.final_share, "shares": list(self.shares)}

    def describe(self) -> str:
        if not self.shares:
            return "adaptation skipped"
        return (f"adaptation iterations {self.iterations} converged {self.converged} "
                f"D share {self.shares[0]:.4f} -> {self.shares[-1]:.4f}")


def bounded_logit(share: float, n_frames: int) -> float:
    margin = 1.0 / (n_frames + 1)
    return float(logit(min(max(share, margin), 1.0 - margin)))


def observation_columns(classifier: FrameClassifier) -> list[int]:
    used = set(classifier.dead_columns)
    return [column for column in range(classifier.observation_limit) if column not in used]


def block_frames(n_frames: int, blocks: Blocks) -> list[np.ndarray]:
    frames = [np.arange(max(first, 0), min(last + 1, n_frames)) for _, first, last in blocks]
    return [indices for indices in frames if indices.size]


def cross_folds(frames: list[np.ndarray]) -> list[np.ndarray]:
    if len(frames) != 1:
        return frames
    return [half for half in np.array_split(frames[0], MATCH_HALVES) if half.size]


def held_out_log_ratio(train: np.ndarray, dead: np.ndarray, target: np.ndarray,
                       seed: int) -> np.ndarray:
    labels = np.concatenate([np.ones(len(train), dtype=bool), np.zeros(len(train), dtype=bool)])
    weights = np.concatenate([dead, 1.0 - dead])
    weights = weights * len(weights) / weights.sum()
    estimator = HistGradientBoostingClassifier(early_stopping=False, random_state=seed)
    share = bounded_logit(float(dead.mean()), len(train))
    with threadpool_limits(limits=THREADS):
        estimator.fit(np.vstack([train, train]), labels, sample_weight=weights)
        return estimator.decision_function(target) - share


def observation_log_ratio(observed: np.ndarray, dead: np.ndarray, folds: list[np.ndarray],
                          seed: int) -> np.ndarray:
    ratio = np.zeros(len(observed))
    for index, fold in enumerate(folds):
        train = np.concatenate([other for position, other in enumerate(folds)
                                if position != index])
        ratio[fold] = held_out_log_ratio(observed[train], dead[train], observed[fold], seed)
    return ratio


def share_tolerance(dead: np.ndarray) -> float:
    return float(np.sqrt(np.sum(dead * (1.0 - dead))) / len(dead))


def adapted_dead_logit(classifier: FrameClassifier, matrix: np.ndarray, blocks: Blocks,
                       seed: int, max_iterations: int, correct_prior: bool = True,
                       segments: Segments | None = None
                       ) -> tuple[np.ndarray, AdaptationReport]:
    players = classifier.dead_logit(matrix, segments)
    columns = observation_columns(classifier)
    folds = cross_folds(block_frames(len(matrix), blocks))
    if not columns or len(folds) < MATCH_HALVES or max_iterations <= 0:
        return players, AdaptationReport(shares=(), converged=False)
    inside = np.concatenate(folds)
    observed = matrix[:, columns]
    training_logit = bounded_logit(float(np.exp(classifier.log_prior[DEAD])), len(inside))

    def prior_shift(share: float) -> float:
        if not correct_prior:
            return 0.0
        return bounded_logit(share, len(inside)) - training_logit

    dead = expit(players)
    shares = [float(dead[inside].mean())]
    converged = False
    for _ in range(max_iterations):
        ratio = observation_log_ratio(observed, dead, folds, seed)
        dead = expit(players + prior_shift(shares[-1]) + ratio)
        shares.append(float(dead[inside].mean()))
        if abs(shares[-1] - shares[-2]) <= share_tolerance(dead[inside]):
            converged = True
            break
    ratio = observation_log_ratio(observed, dead, folds, seed)
    adapted = players.copy()
    adapted[inside] = players[inside] + prior_shift(shares[-1]) + ratio[inside]
    return adapted, AdaptationReport(shares=tuple(shares), converged=converged)
