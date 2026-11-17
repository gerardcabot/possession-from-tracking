import hashlib
from dataclasses import dataclass, fields
from statistics import NormalDist

import numpy as np

from possession.cleaning import CleaningParams, clean, duplicate_distances, without
from possession.data import Tracking
from possession.geometry import home_attacks_positive, present, to_metres
from possession.intervals import value_runs

MAD_TO_SIGMA = 1.0 / NormalDist().inv_cdf(0.75)
SECOND_DIFFERENCE_VARIANCE_FACTOR = 6.0

Blocks = list[tuple[int, int, int]]


@dataclass(frozen=True)
class OurStats:
    fps: float
    ball_observed_share: float
    gap_lengths: np.ndarray
    slot_counts: np.ndarray
    segment_lengths: np.ndarray
    objects_per_frame: float
    duplicate_share: float
    duplicate_lengths: np.ndarray
    duplicate_offsets_m: np.ndarray
    duplicate_distance_m: float
    player_sigma_m: float
    ball_sigma_m: float
    propensity_mean: np.ndarray
    propensity_scale: np.ndarray
    propensity_coef: np.ndarray
    home_attacks_positive_first: bool

    def digest(self) -> str:
        hasher = hashlib.sha256()
        for field in fields(self):
            array = np.ascontiguousarray(getattr(self, field.name))
            hasher.update(f"{field.name}|{array.dtype.str}|{array.shape}|".encode())
            hasher.update(array.tobytes())
            hasher.update(b"|")
        return hasher.hexdigest()

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {field.name: np.asarray(getattr(self, field.name)) for field in fields(self)}

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "OurStats":
        return cls(**{field.name: arrays[field.name] if field.type is np.ndarray
                      else field.type(arrays[field.name].item()) for field in fields(cls)})


def block_slices(blocks: Blocks) -> list[slice]:
    return [slice(first, last + 1) for _, first, last in sorted(blocks, key=lambda b: b[1])]


def block_mask(n_frames: int, blocks: Blocks) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for window in block_slices(blocks):
        mask[window] = True
    return mask


def run_lengths(flags: np.ndarray, blocks: Blocks, value: bool) -> np.ndarray:
    lengths = []
    for window in block_slices(blocks):
        starts, ends, values = value_runs(flags[window])
        lengths.append((ends - starts)[values == value])
    return np.concatenate(lengths) if lengths else np.array([], dtype=np.int64)


def white_noise_sigma(positions: np.ndarray, blocks: Blocks) -> float:
    residuals = []
    for window in block_slices(blocks):
        block = positions[window]
        if len(block) < 3:
            continue
        second = block[2:] - 2 * block[1:-1] + block[:-2]
        residuals.append(np.abs(second[np.isfinite(second)]))
    values = np.concatenate(residuals) if residuals else np.array([])
    if values.size == 0:
        return 0.0
    return float(MAD_TO_SIGMA * np.median(values) / np.sqrt(SECOND_DIFFERENCE_VARIANCE_FACTOR))


def fill_ball(ball: np.ndarray) -> np.ndarray:
    seen = present(ball)
    index = np.arange(len(ball))
    if not seen.any():
        return ball.copy()
    return np.column_stack([np.interp(index, index[seen], ball[seen, axis]) for axis in (0, 1)])


def nearest_and_edge(ball_estimate: np.ndarray, players: np.ndarray,
                     pitch: tuple[float, float]) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        distance = np.linalg.norm(players - ball_estimate[:, None, :], axis=-1)
    distance = np.where(np.isnan(distance), np.inf, distance).min(axis=1)
    finite = np.isfinite(distance)
    nearest = np.where(finite, distance, distance[finite].mean() if finite.any() else 0.0)
    edge = np.minimum(ball_estimate[:, 0], pitch[0] - ball_estimate[:, 0])
    return np.column_stack([nearest, edge]).astype(float)


def standardise(covariates: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (covariates - mean) / scale


def fit_logistic(design: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=np.inf).fit(design, outcome.astype(int)).coef_[0]


def fit_propensity(observed: np.ndarray, ball_m: np.ndarray, players_m: np.ndarray,
                   pitch: tuple[float, float],
                   blocks: Blocks) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    graded = block_mask(len(observed), blocks)
    covariates = nearest_and_edge(fill_ball(ball_m), players_m, pitch)[graded]
    mean = covariates.mean(axis=0)
    scale = np.where(covariates.std(axis=0) > 0, covariates.std(axis=0), 1.0)
    coef = fit_logistic(standardise(covariates, mean, scale), (~observed[graded]).astype(float))
    return mean, scale, coef


def duplicate_offsets(players: np.ndarray, slot_team: np.ndarray, min_distance: float,
                      scope: np.ndarray) -> np.ndarray:
    partner = duplicate_distances(players, slot_team, min_distance)
    return np.sort(partner[scope & np.isfinite(partner)])


def our_stats(tracking: Tracking, pitch: tuple[float, float], fps: float, blocks: Blocks,
              params: CleaningParams) -> OurStats:
    cleaned = clean(tracking, pitch, fps, params)
    graded = block_mask(len(tracking.ball), blocks)
    players = cleaned.players
    ball = without(to_metres(tracking.ball, pitch), cleaned.ball_outlier)
    seen = present(players)
    slot_runs = np.concatenate([run_lengths(seen[:, slot], blocks, True)
                                for slot in range(seen.shape[1])])
    duplicate = cleaned.player_duplicate
    objects = seen | duplicate
    duplicate_runs = np.concatenate([run_lengths(duplicate[:, slot], blocks, True)
                                     for slot in range(seen.shape[1])])
    without_jumps = without(to_metres(tracking.players, pitch), cleaned.player_jump)
    observed = cleaned.ball_observed
    mean, scale, coef = fit_propensity(observed, to_metres(tracking.ball, pitch), players,
                                       pitch, blocks)
    return OurStats(
        fps=float(fps),
        ball_observed_share=float(observed[graded].mean()),
        gap_lengths=run_lengths(observed, blocks, False),
        slot_counts=np.bincount(tracking.slot_team, minlength=2),
        segment_lengths=slot_runs,
        objects_per_frame=float(seen[graded].sum(axis=1).mean()),
        duplicate_share=float(duplicate[graded].sum() / max(int(objects[graded].sum()), 1)),
        duplicate_lengths=duplicate_runs,
        duplicate_offsets_m=duplicate_offsets(without_jumps, tracking.slot_team,
                                              params.duplicate_distance_m, graded[:, None]),
        duplicate_distance_m=float(params.duplicate_distance_m),
        player_sigma_m=white_noise_sigma(players, blocks),
        ball_sigma_m=white_noise_sigma(ball, blocks),
        propensity_mean=mean, propensity_scale=scale, propensity_coef=coef,
        home_attacks_positive_first=home_attacks_positive_first(tracking, pitch, blocks),
    )


def home_attacks_positive_first(tracking: Tracking, pitch: tuple[float, float],
                                blocks: Blocks) -> bool:
    return home_attacks_positive(to_metres(tracking.players, pitch), tracking.slot_team,
                                 block_slices(blocks)[0])
