from dataclasses import dataclass

import numpy as np

from possession.features import Features, moving_nanmean
from possession.intervals import drop_short_runs, fill_short_gaps, true_runs
from possession.params import RulesParams, frames


@dataclass(frozen=True)
class DeadEvidence:
    collective_speed: np.ndarray
    outside: np.ndarray
    crossings: np.ndarray
    evidence: np.ndarray


def outside_pitch(ball: np.ndarray, pitch: tuple[float, float], margin: float) -> np.ndarray:
    length, width = pitch
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        return (x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin)


def observed_exits(outside: np.ndarray) -> np.ndarray:
    starts, _ = true_runs(outside)
    return starts


def outward_distances(position: np.ndarray, velocity: np.ndarray,
                      pitch: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    length, width = pitch
    x, y = position[:, 0], position[:, 1]
    vx, vy = velocity[:, 0], velocity[:, 1]
    distance = np.stack([x, length - x, y, width - y], axis=1)
    speed = np.stack([-vx, vx, -vy, vy], axis=1)
    return distance, speed


def gap_exits(features: Features, params: RulesParams) -> np.ndarray:
    starts, ends = true_runs(~features.ball_present)
    keep = starts > 0
    starts, ends = starts[keep], ends[keep]
    last = starts - 1
    distance, speed = outward_distances(features.ball[last], features.ball_velocity[last],
                                        features.pitch)
    with np.errstate(invalid="ignore", divide="ignore"):
        reachable = (distance <= params.exit_reach_m) & (speed >= params.exit_min_speed_mps)
        delay = np.where(reachable, distance / speed * features.fps, np.inf).min(axis=1)
    crossing = last + np.ceil(np.nan_to_num(delay, nan=np.inf, posinf=1e12)).astype(np.int64)
    inside_gap = np.isfinite(delay) & (crossing < ends)
    return np.maximum(crossing[inside_gap], starts[inside_gap])


def exit_frames(features: Features, outside: np.ndarray, params: RulesParams) -> np.ndarray:
    return np.unique(np.concatenate([observed_exits(outside), gap_exits(features, params)]))


def speed_evidence(collective: np.ndarray, params: RulesParams) -> np.ndarray:
    scaled = (params.calm_speed_mps - collective) / params.speed_scale_mps
    capped = np.clip(scaled, -params.speed_evidence_cap, params.speed_evidence_cap)
    return np.where(np.isnan(collective), 0.0, capped)


def dead_evidence(features: Features, params: RulesParams) -> DeadEvidence:
    collective = moving_nanmean(features.mean_speed, frames(params.speed_window_s, features.fps))
    outside = outside_pitch(features.ball, features.pitch, params.out_margin_m)
    with np.errstate(invalid="ignore"):
        active = features.ball_present & (features.ball_speed > params.active_ball_mps)
    evidence = (speed_evidence(collective, params) - params.active_evidence * active
                + params.out_evidence * outside)
    return DeadEvidence(collective_speed=collective, outside=outside,
                        crossings=exit_frames(features, outside, params), evidence=evidence)


def dead_mask(evidence: np.ndarray, params: RulesParams, fps: float) -> np.ndarray:
    smoothed = moving_nanmean(evidence, frames(params.evidence_window_s, fps))
    dead = smoothed > 0
    dead = fill_short_gaps(dead, frames(params.min_live_s, fps))
    return drop_short_runs(dead, frames(params.min_dead_s, fps))


def deceleration_onset(collective: np.ndarray, start: int, low: int) -> int:
    window = collective[low:start + 1]
    if not np.isfinite(window).any():
        return start
    peak = np.nanmax(window)
    return low + int(np.flatnonzero(window == peak)[-1])


def refine_start(start: int, crossings: np.ndarray, collective: np.ndarray, low: int,
                 search: int) -> tuple[int, bool]:
    lo = max(low, start - search)
    nearby = crossings[(crossings >= lo) & (crossings <= start + search)]
    if nearby.size:
        return int(nearby[0]), True
    return deceleration_onset(collective, start, lo), False
