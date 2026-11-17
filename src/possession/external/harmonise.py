from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from possession.data import AWAY, HOME
from possession.external.match import LABEL_UNKNOWN, ExternalMatch
from possession.geometry import home_attacks_positive, to_metres
from possession.intervals import DEAD, LABELS

OWNER_NONE = -1
EXACT_TOLERANCE = 1e-9

LabelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class NativeSeries:
    ball: np.ndarray
    players: np.ndarray
    slot_team: np.ndarray
    alive: np.ndarray
    owner: np.ndarray
    periods: list[tuple[int, int, int]]
    fps: float
    pitch: tuple[float, float]


def team_labels(owner: np.ndarray) -> np.ndarray:
    labels = np.full(len(owner), LABEL_UNKNOWN, dtype="<U1")
    labels[owner == HOME] = LABELS[HOME]
    labels[owner == AWAY] = LABELS[AWAY]
    return labels


def dfl_labels(alive: np.ndarray, owner: np.ndarray) -> np.ndarray:
    labels = team_labels(owner)
    labels[~alive] = LABELS[DEAD]
    return labels


def skillcorner_labels(alive: np.ndarray, owner: np.ndarray) -> np.ndarray:
    return team_labels(owner)


def target_length(n_source: int, source_fps: float, target_fps: float) -> int:
    return int(np.floor((n_source - 1) * target_fps / source_fps + EXACT_TOLERANCE)) + 1


def source_position(n_source: int, source_fps: float, target_fps: float) -> np.ndarray:
    count = target_length(n_source, source_fps, target_fps)
    return np.arange(count) * source_fps / target_fps


def resample_linear(values: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    position = source_position(len(values), source_fps, target_fps)
    lower = np.floor(position + EXACT_TOLERANCE).astype(np.int64).clip(0, len(values) - 1)
    upper = np.minimum(lower + 1, len(values) - 1)
    weight = (position - lower).reshape((-1,) + (1,) * (values.ndim - 1))
    with np.errstate(invalid="ignore"):
        blended = values[lower] * (1.0 - weight) + values[upper] * weight
    return np.where(weight <= EXACT_TOLERANCE, values[lower], blended)


def resample_nearest(values: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    position = source_position(len(values), source_fps, target_fps)
    index = np.floor(position + 0.5).astype(np.int64).clip(0, len(values) - 1)
    return values[index]


def flip_pitch(coordinates: np.ndarray) -> np.ndarray:
    return 1.0 - coordinates


def orient_like(match: ExternalMatch, home_attacks_positive_first: bool) -> ExternalMatch:
    _, first, last = match.blocks[0]
    metres = to_metres(match.players, match.pitch)
    attacks_positive = home_attacks_positive(metres, match.slot_team, slice(first, last + 1))
    if attacks_positive == home_attacks_positive_first:
        return match
    return ExternalMatch(
        name=match.name, source=match.source, fps=match.fps, pitch=match.pitch,
        ball=flip_pitch(match.ball), players=flip_pitch(match.players),
        slot_team=match.slot_team, labels=match.labels, periods=match.periods)


def harmonise(series: NativeSeries, name: str, source: str, label_fn: LabelFn,
              target_fps: float, home_attacks_positive_first: bool) -> ExternalMatch:
    balls, squads, labels, periods = [], [], [], []
    cursor = 0
    for period, first, last in series.periods:
        window = slice(first, last + 1)
        ball = resample_linear(series.ball[window], series.fps, target_fps)
        alive = resample_nearest(series.alive[window], series.fps, target_fps)
        owner = resample_nearest(series.owner[window], series.fps, target_fps)
        balls.append(ball)
        squads.append(resample_linear(series.players[window], series.fps, target_fps))
        labels.append(label_fn(alive, owner))
        periods.append((period, cursor, cursor + len(ball) - 1))
        cursor += len(ball)
    match = ExternalMatch(
        name=name, source=source, fps=target_fps, pitch=series.pitch,
        ball=np.concatenate(balls), players=np.concatenate(squads),
        slot_team=series.slot_team, labels=np.concatenate(labels),
        periods=np.array(periods, dtype=np.int64))
    return orient_like(match, home_attacks_positive_first)
