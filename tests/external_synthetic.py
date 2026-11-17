import numpy as np

from possession.data import AWAY, HOME
from possession.external.match import ExternalMatch

PITCH = (100.0, 60.0)
FPS = 25.0
PLAYERS_PER_TEAM = 11


def make_match(n_per_period: int = 3000, periods: int = 2, home_positive: bool = True,
               seed: int = 0) -> ExternalMatch:
    rng = np.random.default_rng(seed)
    n_frames = n_per_period * periods
    team = np.array([HOME] * PLAYERS_PER_TEAM + [AWAY] * PLAYERS_PER_TEAM, dtype=np.int8)
    base = np.concatenate([np.linspace(0.05, 0.45, PLAYERS_PER_TEAM),
                           np.linspace(0.55, 0.95, PLAYERS_PER_TEAM)])
    if not home_positive:
        base = 1.0 - base
    walk = np.cumsum(rng.normal(scale=0.0005, size=(n_frames, len(team), 2)), axis=0)
    walk -= walk.mean(axis=0)
    players = np.clip(np.stack([base, np.full(len(team), 0.5)], axis=1)[None] + walk, 0.0, 1.0)
    ball = np.clip(0.5 + np.cumsum(rng.normal(scale=0.0007, size=(n_frames, 2)), axis=0) % 0.4
                   - 0.2, 0.0, 1.0)
    labels = np.where(np.arange(n_frames) % 500 < 250, "H", "A").astype("<U1")
    bounds = [(index + 1, index * n_per_period, (index + 1) * n_per_period - 1)
              for index in range(periods)]
    return ExternalMatch(name="synthetic", source="test", fps=FPS, pitch=PITCH, ball=ball,
                         players=players, slot_team=team, labels=labels,
                         periods=np.array(bounds, dtype=np.int64))
