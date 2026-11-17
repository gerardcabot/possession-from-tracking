import warnings
from dataclasses import dataclass, replace

import numpy as np
from kloppy import metrica, skillcorner, sportec
from kloppy.domain import BallState, Ground, TrackingDataset

from possession.data import AWAY, HOME
from possession.external.harmonise import OWNER_NONE, NativeSeries
from possession.external.metrica_events import (
    MetricaEvent,
    event_state,
    read_events_csv,
    read_events_json,
)
from possession.external.pff import load_series
from possession.external.sources import DFL, METRICA, PFF, SKILLCORNER, MatchFiles
from possession.intervals import value_runs

GROUND_CODE = {Ground.HOME: HOME, Ground.AWAY: AWAY}


@dataclass(frozen=True)
class Squad:
    slot_of: dict[str, int]
    slot_team: np.ndarray
    dropped: frozenset[str]


def coordinates_or_nan(point: object) -> tuple[float, float]:
    if point is None:
        return np.nan, np.nan
    return float(point.x), float(point.y)


def squad_slots(dataset: TrackingDataset) -> Squad:
    by_ground = {team.ground: team for team in dataset.metadata.teams}
    slot_of: dict[str, int] = {}
    teams: list[int] = []
    for ground in (Ground.HOME, Ground.AWAY):
        for player in by_ground[ground].players:
            slot_of[player.player_id] = len(teams)
            teams.append(GROUND_CODE[ground])
    dropped: set[str] = set()
    for frame in dataset.records:
        for player in frame.players_data:
            if player.player_id in slot_of or player.player_id in dropped:
                continue
            ground = player.team.ground if player.team is not None else None
            if ground in GROUND_CODE:
                slot_of[player.player_id] = len(teams)
                teams.append(GROUND_CODE[ground])
            else:
                dropped.add(player.player_id)
    return Squad(slot_of, np.array(teams, dtype=np.int8), frozenset(dropped))


def period_blocks(period_ids: np.ndarray) -> list[tuple[int, int, int]]:
    starts, ends, values = value_runs(period_ids)
    return [(int(value), int(start), int(end) - 1)
            for start, end, value in zip(starts, ends, values)]


def native_series(dataset: TrackingDataset) -> NativeSeries:
    squad = squad_slots(dataset)
    records = dataset.records
    ball = np.full((len(records), 2), np.nan)
    players = np.full((len(records), len(squad.slot_team), 2), np.nan)
    alive = np.zeros(len(records), dtype=bool)
    owner = np.full(len(records), OWNER_NONE, dtype=np.int8)
    period_ids = np.zeros(len(records), dtype=np.int64)
    dropped_samples = 0
    for index, frame in enumerate(records):
        ball[index] = coordinates_or_nan(frame.ball_coordinates)
        alive[index] = frame.ball_state == BallState.ALIVE
        if frame.ball_owning_team is not None:
            owner[index] = GROUND_CODE[frame.ball_owning_team.ground]
        period_ids[index] = frame.period.id
        for player, data in frame.players_data.items():
            if player.player_id in squad.dropped:
                dropped_samples += 1
                continue
            players[index, squad.slot_of[player.player_id]] = coordinates_or_nan(data.coordinates)
    if squad.dropped:
        warnings.warn(f"dropped {dropped_samples} position samples of {len(squad.dropped)} "
                      "players with no team in the metadata", stacklevel=2)
    seen = ~np.isnan(players[..., 0]).all(axis=0)
    dimensions = dataset.metadata.pitch_dimensions
    return NativeSeries(
        ball=ball, players=players[:, seen], slot_team=squad.slot_team[seen], alive=alive,
        owner=owner, periods=period_blocks(period_ids), fps=float(dataset.metadata.frame_rate),
        pitch=(float(dimensions.pitch_length), float(dimensions.pitch_width)))


def load_dfl(files: MatchFiles) -> NativeSeries:
    meta, raw = files.paths
    dataset = sportec.load_tracking(raw_data=raw, meta_data=meta, only_alive=False,
                                    coordinates="kloppy")
    return native_series(dataset)


def load_skillcorner(files: MatchFiles) -> NativeSeries:
    meta, raw = files.paths
    dataset = skillcorner.load(meta_data=meta, raw_data=raw, coordinates="kloppy",
                               include_empty_frames=True)
    return native_series(dataset)


def metrica_dataset(files: MatchFiles) -> tuple[TrackingDataset, list[MetricaEvent]]:
    first, second, events = files.paths
    if events.suffix == ".json":
        dataset = metrica.load_tracking_epts(meta_data=first, raw_data=second,
                                             coordinates="kloppy")
        team_codes = {team.team_id: GROUND_CODE[team.ground] for team in dataset.metadata.teams}
        return dataset, read_events_json(events, team_codes)
    dataset = metrica.load_tracking_csv(home_data=first, away_data=second, coordinates="kloppy")
    return dataset, read_events_csv(events)


def load_metrica(files: MatchFiles) -> NativeSeries:
    dataset, events = metrica_dataset(files)
    series = native_series(dataset)
    frame_ids = np.array([frame.frame_id for frame in dataset.records], dtype=np.int64)
    period_ids = np.empty(len(frame_ids), dtype=np.int64)
    for period, first, last in series.periods:
        period_ids[first:last + 1] = period
    alive, owner = event_state(events, frame_ids, period_ids)
    return replace(series, alive=alive, owner=owner)


def load_pff(files: MatchFiles) -> NativeSeries:
    meta, raw = files.paths
    return load_series(meta, raw)


LOADERS = {DFL: load_dfl, SKILLCORNER: load_skillcorner, METRICA: load_metrica, PFF: load_pff}
