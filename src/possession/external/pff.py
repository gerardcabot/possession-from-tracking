import bz2
import json
import warnings
from array import array
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from possession.data import AWAY, HOME
from possession.external.harmonise import OWNER_NONE, NativeSeries

PLAY_PERIODS = frozenset({1, 2, 3, 4})
LIVE_EVENTS = frozenset({"FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF",
                         "OTB"})
DEAD_EVENTS = frozenset({"OUT", "END"})
PLAYER_FIELDS = (("homePlayersSmoothed", HOME), ("awayPlayersSmoothed", AWAY))
BALL_FIELD = "balls"
COMPRESSED_SUFFIX = ".bz2"
CENTRE = 0.5
UNNUMBERED_PREFIX = "unnumbered@"


@dataclass(frozen=True)
class PffMetadata:
    game_id: str
    fps: float
    pitch: tuple[float, float]


@dataclass(frozen=True)
class GameEvent:
    kind: str
    start_frame: int
    home_ball: bool | None


@dataclass(frozen=True)
class PffTracking:
    frames: np.ndarray
    periods: list[tuple[int, int, int]]
    ball: np.ndarray
    players: np.ndarray
    slot_team: np.ndarray
    events: list[GameEvent]
    repeated_jerseys: int
    unnumbered_players: int
    ambiguous_balls: int


def read_metadata(path: Path) -> PffMetadata:
    records = json.loads(path.read_text())
    if len(records) != 1:
        raise ValueError(f"{path}: expected one game record, found {len(records)}")
    record = records[0]
    pitches = record["stadium"]["pitches"]
    if len(pitches) != 1:
        raise ValueError(f"{path}: expected one pitch, found {len(pitches)}")
    return PffMetadata(game_id=str(record["id"]), fps=float(record["fps"]),
                       pitch=(float(pitches[0]["length"]), float(pitches[0]["width"])))


def read_records(path: Path) -> Iterator[dict[str, object]]:
    opener = bz2.open if path.suffix == COMPRESSED_SUFFIX else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def located_balls(record: dict[str, object]) -> list[dict[str, float]]:
    return [ball for ball in record.get(BALL_FIELD) or ()
            if ball.get("x") is not None and ball.get("y") is not None]


def frame_period(record: dict[str, object]) -> int | None:
    period = record.get("period")
    return None if period is None else int(period)


@dataclass(frozen=True)
class Samples:
    frames: np.ndarray
    periods: np.ndarray
    ball: np.ndarray
    rows: np.ndarray
    slots: np.ndarray
    positions: np.ndarray
    slot_team: np.ndarray
    events: list[GameEvent]
    repeated_jerseys: int
    unnumbered_players: int
    ambiguous_balls: int


def event_key(record: dict[str, object], event: dict[str, object]) -> object:
    identifier = record.get("game_event_id")
    if identifier is not None:
        return identifier
    return (event["start_frame"], event["game_event_type"], event.get("end_frame"),
            event.get("home_ball"))


def collect_samples(records: Iterable[dict[str, object]]) -> Samples:
    slot_of: dict[tuple[int, str], int] = {}
    frames, periods, rows, slots = array("q"), array("q"), array("q"), array("q")
    ball, positions = array("d"), array("d")
    events: dict[object, GameEvent] = {}
    last: int | None = None
    repeated = unnumbered = ambiguous = 0
    for record in records:
        event = record.get("game_event")
        if event and event_key(record, event) not in events:
            events[event_key(record, event)] = GameEvent(
                kind=str(event["game_event_type"]), start_frame=int(event["start_frame"]),
                home_ball=event.get("home_ball"))
        period = frame_period(record)
        if period not in PLAY_PERIODS:
            continue
        number = int(record["frameNum"])
        if last is not None and number <= last:
            if number == last:
                continue
            raise ValueError(f"frame {number} follows frame {last}: frames are out of order")
        last = number
        row = len(frames)
        frames.append(number)
        periods.append(period)
        balls = located_balls(record)
        ambiguous += len(balls) > 1
        ball.extend((float(balls[0]["x"]), float(balls[0]["y"])) if len(balls) == 1
                    else (np.nan, np.nan))
        for field, team in PLAYER_FIELDS:
            seen: set[str] = set()
            for position, player in enumerate(record.get(field) or ()):
                if player.get("x") is None or player.get("y") is None:
                    continue
                if player.get("jerseyNum") is None:
                    jersey = f"{UNNUMBERED_PREFIX}{position}"
                    unnumbered += 1
                else:
                    jersey = str(player["jerseyNum"])
                    if jersey in seen:
                        repeated += 1
                        continue
                seen.add(jersey)
                rows.append(row)
                slots.append(slot_of.setdefault((team, jersey), len(slot_of)))
                positions.extend((float(player["x"]), float(player["y"])))
    slot_team = np.array([team for team, _ in slot_of], dtype=np.int8)
    return Samples(
        frames=np.frombuffer(frames, dtype=np.int64), periods=np.frombuffer(periods, dtype=np.int64),
        ball=np.frombuffer(ball, dtype=float).reshape(-1, 2),
        rows=np.frombuffer(rows, dtype=np.int64), slots=np.frombuffer(slots, dtype=np.int64),
        positions=np.frombuffer(positions, dtype=float).reshape(-1, 2), slot_team=slot_team,
        events=list(events.values()), repeated_jerseys=repeated,
        unnumbered_players=unnumbered, ambiguous_balls=ambiguous)


def period_grid(frames: np.ndarray, periods: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    numbers, targets, blocks = [], np.empty(len(frames), dtype=np.int64), []
    cursor = 0
    for period in dict.fromkeys(periods.tolist()):
        mask = periods == period
        if not np.all(np.diff(np.flatnonzero(mask)) == 1):
            raise ValueError(f"period {period} is not contiguous in the tracking file")
        first, last = int(frames[mask][0]), int(frames[mask][-1])
        numbers.append(np.arange(first, last + 1))
        targets[mask] = cursor + frames[mask] - first
        blocks.append((period, cursor, cursor + last - first))
        cursor += last - first + 1
    grid = np.concatenate(numbers) if numbers else np.zeros(0, dtype=np.int64)
    return grid, targets, blocks


def parse_tracking(records: Iterable[dict[str, object]]) -> PffTracking:
    samples = collect_samples(records)
    grid, targets, blocks = period_grid(samples.frames, samples.periods)
    ball = np.full((len(grid), 2), np.nan)
    ball[targets] = samples.ball
    players = np.full((len(grid), len(samples.slot_team), 2), np.nan)
    players[targets[samples.rows], samples.slots] = samples.positions
    return PffTracking(frames=grid, periods=blocks, ball=ball, players=players,
                       slot_team=samples.slot_team, events=samples.events,
                       repeated_jerseys=samples.repeated_jerseys,
                       unnumbered_players=samples.unnumbered_players,
                       ambiguous_balls=samples.ambiguous_balls)


def event_state(events: list[GameEvent], frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    changes = sorted((event for event in events if event.kind in LIVE_EVENTS | DEAD_EVENTS),
                     key=lambda event: event.start_frame)
    alive_after = np.zeros(len(changes) + 1, dtype=bool)
    owner_after = np.full(len(changes) + 1, OWNER_NONE, dtype=np.int8)
    alive, owner = False, OWNER_NONE
    for index, event in enumerate(changes, start=1):
        restarted = event.kind in LIVE_EVENTS and not alive
        alive = event.kind in LIVE_EVENTS
        if alive and event.home_ball is not None:
            owner = HOME if event.home_ball else AWAY
        elif restarted:
            owner = OWNER_NONE
        alive_after[index], owner_after[index] = alive, owner
    starts = np.array([event.start_frame for event in changes], dtype=np.int64)
    state = np.searchsorted(starts, frames, side="right")
    return alive_after[state], owner_after[state]


def normalised(metres: np.ndarray, pitch: tuple[float, float]) -> np.ndarray:
    length, width = pitch
    result = np.empty_like(metres)
    result[..., 0] = metres[..., 0] / length + CENTRE
    result[..., 1] = CENTRE - metres[..., 1] / width
    return result


def native_series(tracking: PffTracking, metadata: PffMetadata) -> NativeSeries:
    alive, owner = event_state(tracking.events, tracking.frames)
    return NativeSeries(
        ball=normalised(tracking.ball, metadata.pitch),
        players=normalised(tracking.players, metadata.pitch), slot_team=tracking.slot_team,
        alive=alive, owner=owner, periods=tracking.periods, fps=metadata.fps,
        pitch=metadata.pitch)


def warn_dropped(tracking: PffTracking) -> None:
    if tracking.repeated_jerseys:
        warnings.warn(f"dropped {tracking.repeated_jerseys} player samples that repeat a jersey "
                      "number of their team in the same frame", stacklevel=2)
    if tracking.unnumbered_players:
        warnings.warn(f"{tracking.unnumbered_players} player samples have no jersey number; "
                      "each is kept as a separate player by its position in the frame's list",
                      stacklevel=2)
    if tracking.ambiguous_balls:
        warnings.warn(f"{tracking.ambiguous_balls} frames list more than one located ball; "
                      "their ball is left missing", stacklevel=2)


def load_series(metadata_path: Path, tracking_path: Path) -> NativeSeries:
    metadata = read_metadata(metadata_path)
    tracking = parse_tracking(read_records(tracking_path))
    warn_dropped(tracking)
    return native_series(tracking, metadata)
