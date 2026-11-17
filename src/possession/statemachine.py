from bisect import bisect_left
from dataclasses import dataclass

import numpy as np

from possession.control import ControlEvent
from possession.data import AWAY, HOME, NOBODY
from possession.features import Features
from possession.intervals import DEAD, true_runs


@dataclass(frozen=True)
class Stoppage:
    start: int
    restart: int
    team: int
    by_exit: bool


@dataclass(frozen=True)
class LiveContext:
    events: list[ControlEvent]
    event_starts: list[int]
    missing: np.ndarray
    shape: np.ndarray
    gap_frames: int
    nearest: np.ndarray
    touch_frames: int
    touch_radius_m: float


def shape_evidence(features: Features) -> np.ndarray:
    along = features.centroid_velocity[:, :, 0]
    finite = np.isfinite(along)
    mean = np.where(finite, along, 0.0).sum(axis=1) / np.maximum(finite.sum(axis=1), 1)
    return features.attack_sign * mean


def for_team(shape: np.ndarray, team: int) -> np.ndarray:
    return shape if team == HOME else -shape


def changepoint(entry_evidence: np.ndarray) -> int:
    prefix = np.concatenate(([0.0], np.cumsum(entry_evidence)))
    candidates = np.flatnonzero(prefix >= prefix.max())
    middle = len(entry_evidence) / 2
    return int(candidates[np.argmin(np.abs(candidates - middle))])


def switch_frame(event_start: int, uncertain_from: int, entry_team: int,
                 context: LiveContext) -> int:
    if event_start <= uncertain_from:
        return event_start
    starts, ends = true_runs(context.missing[uncertain_from:event_start])
    long_gaps = starts[ends - starts >= context.gap_frames]
    if long_gaps.size == 0:
        return event_start
    first = uncertain_from + int(long_gaps[0])
    return first + changepoint(for_team(context.shape[first:event_start], entry_team))


def events_between(context: LiveContext, start: int, end: int) -> list[ControlEvent]:
    first = bisect_left(context.event_starts, start)
    last = bisect_left(context.event_starts, end)
    return context.events[first:last]


def opponent(team: int) -> int:
    return AWAY if team == HOME else HOME


def first_touch_team(context: LiveContext, frame: int) -> int:
    window = context.nearest[frame:frame + context.touch_frames + 1]
    closest = np.min(window, axis=1)
    decided = np.isfinite(closest) & (window[:, HOME] != window[:, AWAY])
    near = np.flatnonzero(decided & (closest <= context.touch_radius_m))
    if near.size:
        return int(np.argmin(window[near[0]]))
    usable = np.flatnonzero(decided)
    if usable.size == 0:
        return NOBODY
    return int(np.argmin(window[usable[np.argmin(closest[usable])]]))


def restart_team(stoppage: Stoppage, previous: int, segment: list[ControlEvent],
                 context: LiveContext) -> int:
    if stoppage.team != NOBODY:
        return stoppage.team
    touch = first_touch_team(context, stoppage.restart)
    if touch != NOBODY:
        return touch
    if previous != NOBODY:
        return opponent(previous) if stoppage.by_exit else previous
    if segment:
        return segment[0].team
    return NOBODY


def fill_live(codes: np.ndarray, start: int, end: int, team: int,
              context: LiveContext) -> None:
    current, cursor, anchor = team, start, start
    for event in events_between(context, start, end):
        if event.team == current:
            anchor = max(anchor, min(event.end, end))
            continue
        frame = min(max(switch_frame(event.start, anchor, current, context), cursor), end)
        codes[cursor:frame] = current
        current, cursor, anchor = event.team, frame, min(event.end, end)
    codes[cursor:end] = current


def build_codes(n_frames: int, stoppages: list[Stoppage], context: LiveContext) -> np.ndarray:
    codes = np.full(n_frames, DEAD, dtype=np.int8)
    previous = NOBODY
    for index, stoppage in enumerate(stoppages):
        end = stoppages[index + 1].start if index + 1 < len(stoppages) else n_frames
        if stoppage.restart >= end:
            continue
        segment = events_between(context, stoppage.restart, end)
        team = restart_team(stoppage, previous, segment, context)
        if team == NOBODY:
            continue
        fill_live(codes, stoppage.restart, end, team, context)
        previous = int(codes[end - 1])
    return codes
