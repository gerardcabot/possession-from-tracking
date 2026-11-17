from dataclasses import dataclass

import numpy as np

from possession.data import NOBODY
from possession.features import Features
from possession.intervals import run_has_length, true_runs
from possession.params import RulesParams, frames
from possession.tagging import TagEvent

CLEAR_MOVE_HORIZON_S = 1.0


@dataclass(frozen=True)
class Restart:
    frame: int
    team: int
    onset: int


@dataclass(frozen=True)
class KickOnsets:
    onset: np.ndarray
    anchor: np.ndarray


def kick_onsets(features: Features, params: RulesParams) -> KickOnsets:
    present = features.ball_present
    with np.errstate(invalid="ignore"):
        moving = present & (features.ball_speed >= params.rest_speed_mps)
    settled_or_missing = ~moving
    settled = run_has_length(settled_or_missing, frames(params.rest_s, features.fps))
    onset = np.flatnonzero(present & moving & settled)
    previous = np.clip(onset - 1, 0, None)
    anchor_frame = np.where(present[previous], previous, onset)
    return KickOnsets(onset=onset, anchor=features.ball[anchor_frame])


def clear_move_frame(features: Features, onset: int, anchor: np.ndarray,
                     params: RulesParams) -> int | None:
    horizon = frames(CLEAR_MOVE_HORIZON_S, features.fps)
    path = features.ball[onset:onset + horizon]
    with np.errstate(invalid="ignore"):
        moved = np.linalg.norm(path - anchor, axis=1) > params.clear_move_m
    hits = np.flatnonzero(moved)
    return onset + int(hits[0]) if hits.size else None


def taker_team(features: Features, onset: int, params: RulesParams) -> int:
    low = max(0, onset - frames(params.rest_s, features.fps))
    nearest = features.nearest[low:onset + 1]
    if not np.isfinite(nearest).any():
        return NOBODY
    closest = np.min(nearest, axis=0)
    team = int(np.argmin(closest))
    return team if closest[team] <= params.taker_radius_m else NOBODY


def tagged_team(tags: list[TagEvent], frame: int) -> int:
    for event in tags:
        if event.start <= frame < event.end and event.team != NOBODY:
            return event.team
    return NOBODY


def restart_at(features: Features, onsets: KickOnsets, tags: list[TagEvent], index: int,
               params: RulesParams) -> Restart | None:
    onset = int(onsets.onset[index])
    frame = clear_move_frame(features, onset, onsets.anchor[index], params)
    if frame is None:
        return None
    team = taker_team(features, onset, params)
    if team == NOBODY:
        team = tagged_team(tags, frame)
    return Restart(frame=frame, team=team, onset=onset)


def onsets_between(onsets: KickOnsets, low: int, high: int) -> list[int]:
    return np.flatnonzero((onsets.onset >= low) & (onsets.onset < high)).tolist()


def find_restart(features: Features, onsets: KickOnsets, tags: list[TagEvent], low: int,
                 high: int, fallback: int, params: RulesParams) -> Restart:
    for index in reversed(onsets_between(onsets, low, high)):
        found = restart_at(features, onsets, tags, index, params)
        if found is not None:
            return found
    return Restart(frame=fallback, team=tagged_team(tags, fallback), onset=fallback)


def first_restart(features: Features, onsets: KickOnsets, tags: list[TagEvent], low: int,
                  high: int, params: RulesParams) -> Restart | None:
    for index in onsets_between(onsets, low, high):
        found = restart_at(features, onsets, tags, index, params)
        if found is not None:
            return found
    return None


def kickoff_formation(features: Features, params: RulesParams) -> np.ndarray:
    enough = (features.tracked >= params.kickoff_min_tracked).all(axis=1)
    onside = (features.beyond_halfway <= params.kickoff_offside_players).all(axis=1)
    return enough & onside


def kicking_team(features: Features, frame: int) -> int:
    distance = features.centre_distance[frame]
    return int(np.argmin(distance)) if np.isfinite(distance).any() else NOBODY


def centre_kick(features: Features, onsets: KickOnsets, low: int, high: int,
                params: RulesParams) -> int | None:
    centre = np.asarray(features.pitch, dtype=float) / 2
    near_centre = np.linalg.norm(onsets.anchor - centre, axis=1) <= params.kickoff_radius_m
    selected = np.flatnonzero(near_centre & (onsets.onset >= low) & (onsets.onset < high))
    for index in selected.tolist():
        frame = clear_move_frame(features, int(onsets.onset[index]), onsets.anchor[index],
                                 params)
        if frame is not None:
            return frame
    return None


def kickoff_window(features: Features, low: int, high: int, params: RulesParams) -> int:
    return min(high, low + frames(params.kickoff_window_s, features.fps))


def find_kickoff(features: Features, onsets: KickOnsets, low: int, high: int,
                 params: RulesParams) -> Restart | None:
    high = kickoff_window(features, low, high, params)
    starts, ends = true_runs(kickoff_formation(features, params))
    long_enough = ends - starts >= frames(params.kickoff_formation_s, features.fps)
    selected = np.flatnonzero(long_enough & (ends >= low) & (ends < high))
    if selected.size == 0:
        return None
    broken = int(ends[selected[0]])
    search = frames(params.boundary_search_s, features.fps)
    kick = centre_kick(features, onsets, max(low, broken - search), broken + 1, params)
    frame = broken if kick is None else kick
    return Restart(frame=frame, team=kicking_team(features, max(frame - 1, 0)), onset=frame)
