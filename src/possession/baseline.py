import numpy as np

from possession.data import AWAY, HOME, NOBODY, Tracking, pitch_size
from possession.intervals import LABELS


def nearest_player_codes(tracking: Tracking) -> np.ndarray:
    length, width = pitch_size()
    scale = np.array([length, width])
    ball = tracking.ball * scale
    players = tracking.players * scale
    codes = np.full(len(ball), NOBODY, dtype=np.int8)

    located = ~np.isnan(ball[:, 0])
    delta = players[located] - ball[located][:, None, :]
    distance = delta[..., 0] ** 2 + delta[..., 1] ** 2
    has_player = ~np.all(np.isnan(distance), axis=1)
    nearest = np.nanargmin(np.where(np.isnan(distance), np.inf, distance), axis=1)

    team = tracking.slot_team[nearest]
    codes[np.flatnonzero(located)[has_player]] = team[has_player]
    return codes


def window_counts(flags: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.concatenate(([0], np.cumsum(flags)))
    index = np.arange(len(flags))
    low = np.maximum(0, index - window)
    high = np.minimum(len(flags), index + window + 1)
    return cumulative[high] - cumulative[low]


def smooth(codes: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))

    home = window_counts(codes == HOME, window)
    away = window_counts(codes == AWAY, window)
    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


def predict(tracking: Tracking, window: int) -> np.ndarray:
    return smooth(nearest_player_codes(tracking), window)
