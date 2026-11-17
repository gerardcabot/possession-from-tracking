from collections.abc import Callable
from dataclasses import replace

import numpy as np

from possession import baseline, rules
from possession.data import Tracking, pitch_size
from possession.external.match import ExternalMatch
from possession.features import player_speeds
from possession.geometry import to_metres
from possession.params import LEGACY_CLEANING, rules_params
from possession.run import RunConfig


def to_tracking(match: ExternalMatch) -> Tracking:
    return Tracking(ball=match.ball, ball_speed=np.full(match.n_frames, np.nan),
                    players=match.players,
                    player_speed=np.full(match.players.shape[:2], np.nan),
                    slot_team=match.slot_team)


def legacy_tracking(match: ExternalMatch) -> Tracking:
    speed = player_speeds(to_metres(match.players, match.pitch), match.fps,
                          LEGACY_CLEANING.player_half_span)
    return replace(to_tracking(match), player_speed=speed)


def predict_rules(match: ExternalMatch) -> np.ndarray:
    features = rules.legacy_features(legacy_tracking(match), match.pitch, match.fps,
                                     match.blocks)
    return rules.label_match(features, [], match.blocks, rules_params({}))


def predict_baseline(match: ExternalMatch) -> np.ndarray:
    tracking = to_tracking(match)
    ratio = np.asarray(match.pitch, dtype=float) / np.asarray(pitch_size(), dtype=float)
    rescaled = replace(tracking, ball=tracking.ball * ratio, players=tracking.players * ratio)
    return baseline.predict(rescaled, RunConfig.window)


METHODS: dict[str, Callable[[ExternalMatch], np.ndarray]] = {
    "baseline": predict_baseline,
    "rules": predict_rules,
}
