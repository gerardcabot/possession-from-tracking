import warnings

import numpy as np

from possession.data import AWAY, HOME

TEAMS = (HOME, AWAY)


def to_metres(normalised: np.ndarray, pitch: tuple[float, float]) -> np.ndarray:
    return normalised * np.asarray(pitch, dtype=float)


def present(positions: np.ndarray) -> np.ndarray:
    return ~np.isnan(positions[..., 0]) & ~np.isnan(positions[..., 1])


def home_attacks_positive(players: np.ndarray, slot_team: np.ndarray,
                          frames: slice) -> bool:
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        deepest = [np.nanmedian(np.nanmin(players[frames, slot_team == team, 0], axis=1))
                   for team in TEAMS]
    return bool(deepest[HOME] <= deepest[AWAY])


def same_team_pairs(slot_team: np.ndarray) -> list[tuple[int, int]]:
    first, second = np.triu_indices(len(slot_team), 1)
    keep = slot_team[first] == slot_team[second]
    return list(zip(first[keep].tolist(), second[keep].tolist()))


def pair_distance(players: np.ndarray, first: int, second: int) -> np.ndarray:
    distance = np.linalg.norm(players[:, second] - players[:, first], axis=-1)
    return np.where(np.isnan(distance), np.inf, distance)


def nearest_teammate_distance(players: np.ndarray, slot_team: np.ndarray) -> np.ndarray:
    nearest = np.full(players.shape[:2], np.inf)
    for first, second in same_team_pairs(slot_team):
        distance = pair_distance(players, first, second)
        nearest[:, first] = np.minimum(nearest[:, first], distance)
        nearest[:, second] = np.minimum(nearest[:, second], distance)
    return nearest
