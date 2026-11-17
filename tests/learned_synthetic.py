import numpy as np

from possession.data import AWAY, HOME
from possession.external.match import ExternalMatch
from possession.intervals import LABELS

PITCH = (100.0, 60.0)
FPS = 25.0
PLAYERS_PER_TEAM = 11
STATE_CYCLE = (0, 1, 2, 1, 0, 2)


def label_sequence(n_frames: int, rng: np.random.Generator) -> np.ndarray:
    codes, index = [], 0
    while len(codes) < n_frames:
        codes += [STATE_CYCLE[index % len(STATE_CYCLE)]] * int(rng.integers(60, 240))
        index += 1
    return np.array(codes[:n_frames], dtype=np.int64)


def possession_match(n_per_period: int = 1500, seed: int = 0) -> ExternalMatch:
    rng = np.random.default_rng(seed)
    n_frames = 2 * n_per_period
    team = np.array([HOME] * PLAYERS_PER_TEAM + [AWAY] * PLAYERS_PER_TEAM, dtype=np.int8)
    base = np.concatenate([np.linspace(0.1, 0.45, PLAYERS_PER_TEAM),
                           np.linspace(0.55, 0.9, PLAYERS_PER_TEAM)])
    anchor = np.stack([base, np.linspace(0.1, 0.9, 2 * PLAYERS_PER_TEAM)], axis=1)
    walk = np.cumsum(rng.normal(scale=0.002, size=(n_frames, len(team), 2)), axis=0)
    players = np.clip(anchor[None] + walk - walk.mean(axis=0), 0.02, 0.98)
    codes = label_sequence(n_frames, rng)
    carrier = np.where(codes == HOME, 0, PLAYERS_PER_TEAM) + rng.integers(0, PLAYERS_PER_TEAM)
    ball = players[np.arange(n_frames), carrier] + rng.normal(scale=0.002, size=(n_frames, 2))
    ball[codes == 2] = np.nan
    periods = np.array([(1, 0, n_per_period - 1), (2, n_per_period, n_frames - 1)],
                       dtype=np.int64)
    return ExternalMatch(name=f"synthetic{seed}", source="dfl", fps=FPS, pitch=PITCH,
                         ball=ball, players=players, slot_team=team,
                         labels=LABELS[codes].astype("<U1"), periods=periods)
