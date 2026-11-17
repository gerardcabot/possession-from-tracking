from dataclasses import dataclass

import numpy as np

from possession.data import AWAY, HOME, NOBODY
from possession.features import Features
from possession.intervals import value_runs
from possession.params import RulesParams, frames


@dataclass(frozen=True)
class ControlEvent:
    team: int
    start: int
    end: int


def control_candidates(features: Features, params: RulesParams) -> np.ndarray:
    home, away = features.nearest[:, HOME], features.nearest[:, AWAY]
    with np.errstate(invalid="ignore"):
        slow = ~(features.ball_speed > params.control_ball_mps)
    usable = features.ball_present & slow
    team = np.full(len(home), NOBODY, dtype=np.int8)
    team[usable & (home < params.control_radius_m) & (home < away)] = HOME
    team[usable & (away < params.control_radius_m) & (away < home)] = AWAY
    return team


def control_events(team: np.ndarray, min_frames: int) -> list[ControlEvent]:
    starts, ends, values = value_runs(team)
    keep = (values != NOBODY) & (ends - starts >= min_frames)
    return [ControlEvent(team=int(value), start=int(start), end=int(end))
            for value, start, end in zip(values[keep], starts[keep], ends[keep])]


def detect_control(features: Features, params: RulesParams) -> list[ControlEvent]:
    return control_events(control_candidates(features, params),
                          frames(params.control_s, features.fps))
