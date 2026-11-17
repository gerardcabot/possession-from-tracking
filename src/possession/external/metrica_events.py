import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from possession.data import AWAY, HOME
from possession.external.harmonise import OWNER_NONE

SUBTYPE_SEPARATOR = "-"
CSV_TEAMS = {"Home": HOME, "Away": AWAY}
DEAD_STATE = -2


@dataclass(frozen=True)
class MetricaEvent:
    team: int
    type: str
    subtypes: frozenset[str]
    period: int
    start: int
    end: int


@dataclass(frozen=True)
class StopRule:
    type: str
    subtypes: frozenset[str]
    needs_restart: bool

    def matches(self, event: MetricaEvent) -> bool:
        if event.type != self.type:
            return False
        return not self.subtypes or bool(self.subtypes & event.subtypes)


RESTART_TYPE = "SET PIECE"
STOP_RULES = (
    StopRule("BALL OUT", frozenset(), needs_restart=False),
    StopRule("FAULT RECEIVED", frozenset(), needs_restart=False),
    StopRule("SHOT", frozenset({"GOAL", "OUT"}), needs_restart=False),
    StopRule("BALL LOST", frozenset({"END HALF"}), needs_restart=False),
    StopRule("SHOT", frozenset({"OFFSIDE"}), needs_restart=True),
    StopRule("PASS", frozenset({"OFFSIDE"}), needs_restart=True),
    StopRule("CHALLENGE", frozenset({"FAULT"}), needs_restart=True),
    StopRule("BALL LOST", frozenset({"FORCED", "VOLUNTARY", "OFFSIDE", "HAND BALL"}),
             needs_restart=True),
)
PLAY_TYPES = frozenset({"PASS", "CARRY", "SHOT", RESTART_TYPE, "RECOVERY"})


def csv_subtypes(text: str) -> frozenset[str]:
    return frozenset(part for part in text.split(SUBTYPE_SEPARATOR) if part)


def read_events_csv(path: Path) -> list[MetricaEvent]:
    with open(path, newline="") as handle:
        return [MetricaEvent(
            team=CSV_TEAMS[row["Team"]], type=row["Type"], subtypes=csv_subtypes(row["Subtype"]),
            period=int(row["Period"]), start=int(row["Start Frame"]), end=int(row["End Frame"]))
            for row in csv.DictReader(handle)]


def json_subtypes(value: dict | list | None) -> frozenset[str]:
    if value is None:
        return frozenset()
    items = value if isinstance(value, list) else [value]
    return frozenset(item["name"] for item in items)


def read_events_json(path: Path, team_codes: dict[str, int]) -> list[MetricaEvent]:
    with open(path) as handle:
        rows = json.load(handle)["data"]
    return [MetricaEvent(
        team=team_codes[row["team"]["id"]], type=row["type"]["name"],
        subtypes=json_subtypes(row["subtypes"]), period=int(row["period"]),
        start=int(row["start"]["frame"]), end=int(row["end"]["frame"])) for row in rows]


def matching_rules(event: MetricaEvent) -> list[StopRule]:
    return [rule for rule in STOP_RULES if rule.matches(event)]


def is_stop_candidate(event: MetricaEvent) -> bool:
    return bool(matching_rules(event))


def is_play(event: MetricaEvent) -> bool:
    return event.type in PLAY_TYPES


def frame_of(event: MetricaEvent) -> int:
    return max(event.start, event.end)


def restart_follows(events: list[MetricaEvent], index: int) -> bool:
    frame = frame_of(events[index])
    for later in events[index + 1:]:
        if is_play(later) and later.start >= frame:
            return later.type == RESTART_TYPE
    return False


def effective_stops(events: list[MetricaEvent]) -> list[bool]:
    flags = []
    for index, event in enumerate(events):
        rules = matching_rules(event)
        flags.append(bool(rules) and (not all(rule.needs_restart for rule in rules)
                                      or restart_follows(events, index)))
    return flags


def ordered(events: list[MetricaEvent]) -> list[MetricaEvent]:
    return sorted(events, key=lambda item: (item.start, not is_stop_candidate(item)))


def last_event_frame(events: list[MetricaEvent], before: int) -> int:
    return max((frame_of(item) for item in events if item.start < before), default=-1)


def period_transitions(events: list[MetricaEvent], first_frame: int) -> list[tuple[int, int]]:
    transitions = [(first_frame, DEAD_STATE)]
    dead_frame = first_frame - 1
    state = DEAD_STATE
    events = ordered(events)
    for event, stops in zip(events, effective_stops(events)):
        if state == DEAD_STATE:
            if stops:
                dead_frame = max(dead_frame, frame_of(event))
            elif is_play(event) and event.start > dead_frame:
                state = event.team
                transitions.append((event.start, state))
            continue
        if not (is_play(event) or stops):
            continue
        if event.type == RESTART_TYPE:
            lull = last_event_frame(events, event.start)
            if transitions[-1][0] <= lull < event.start:
                transitions.append((lull, DEAD_STATE))
        if event.team != state or transitions[-1][1] == DEAD_STATE:
            state = event.team
            transitions.append((event.start, state))
        if stops:
            dead_frame = frame_of(event)
            state = DEAD_STATE
            transitions.append((dead_frame, state))
    return transitions


def fill_period(frames: np.ndarray, transitions: list[tuple[int, int]]) -> np.ndarray:
    states = np.full(len(frames), DEAD_STATE, dtype=np.int8)
    positions = [int(np.searchsorted(frames, frame)) for frame, _ in transitions]
    positions.append(len(frames))
    for (_, state), first, stop in zip(transitions, positions, positions[1:]):
        states[first:stop] = state
    return states


def event_state(events: list[MetricaEvent], frame_ids: np.ndarray,
                period_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states = np.full(len(frame_ids), DEAD_STATE, dtype=np.int8)
    for period in np.unique(period_ids):
        rows = np.flatnonzero(period_ids == period)
        frames = frame_ids[rows]
        mine = [event for event in events if event.period == int(period)]
        states[rows] = fill_period(frames, period_transitions(mine, int(frames[0])))
    alive = states != DEAD_STATE
    owner = np.where(alive, states, OWNER_NONE).astype(np.int8)
    return alive, owner
