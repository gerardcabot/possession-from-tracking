from dataclasses import dataclass
from pathlib import Path

import numpy as np

from possession.data import write_npz
from possession.external.sources import SOURCES
from possession.intervals import value_runs
from possession.vendor import possession_metrics as pm

LABEL_UNKNOWN = "X"
MATCH_FIELDS = ("ball", "players", "slot_team", "labels", "periods", "fps", "pitch", "name",
                "source")


@dataclass(frozen=True)
class ExternalMatch:
    name: str
    source: str
    fps: float
    pitch: tuple[float, float]
    ball: np.ndarray
    players: np.ndarray
    slot_team: np.ndarray
    labels: np.ndarray
    periods: np.ndarray

    @property
    def n_frames(self) -> int:
        return len(self.ball)

    @property
    def blocks(self) -> list[tuple[int, int, int]]:
        return [(int(period), int(first), int(last)) for period, first, last in self.periods]


def label_codes(labels: np.ndarray) -> np.ndarray:
    return pm.label_codes(labels)


def period_runs(match: ExternalMatch) -> list[tuple[int, int]]:
    return [(first, last + 1) for _, first, last in match.blocks]


def filled_codes(match: ExternalMatch) -> np.ndarray:
    return pm.fill_abstentions(label_codes(match.labels), period_runs(match))


def has_dead_labels(match: ExternalMatch) -> bool:
    return SOURCES[match.source].has_dead_labels


def dead_labels_trusted(match: ExternalMatch) -> bool:
    info = SOURCES[match.source]
    return info.has_dead_labels and not info.dead_labels_heuristic


def dead_as_fill(match: ExternalMatch, codes: np.ndarray) -> np.ndarray:
    abstained = np.where(codes == pm.D, pm.X, codes).astype(codes.dtype)
    return pm.fill_abstentions(abstained, period_runs(match))


def fair_codes(match: ExternalMatch, labels: np.ndarray) -> np.ndarray:
    codes = label_codes(labels)
    return codes if dead_labels_trusted(match) else dead_as_fill(match, codes)


def fair_truth(match: ExternalMatch) -> np.ndarray:
    truth = filled_codes(match)
    return truth if dead_labels_trusted(match) else dead_as_fill(match, truth)


def period_ids(match: ExternalMatch) -> np.ndarray:
    ids = np.full(match.n_frames, -1, dtype=np.int64)
    for period, first, last in match.blocks:
        ids[first:last + 1] = period
    return ids


def padded(values: np.ndarray, match: ExternalMatch, fill: int) -> np.ndarray:
    pieces = []
    for index, (first, stop) in enumerate(period_runs(match)):
        if index:
            pieces.append(np.array([fill], dtype=values.dtype))
        pieces.append(values[first:stop])
    return np.concatenate(pieces)


def block_share(flags: np.ndarray, match: ExternalMatch) -> float:
    covered = np.concatenate([flags[first:stop] for first, stop in period_runs(match)])
    return float(covered.mean()) if covered.size else float("nan")


def spell_lengths(codes: np.ndarray, match: ExternalMatch) -> np.ndarray:
    lengths = []
    for first, stop in period_runs(match):
        starts, ends, values = value_runs(codes[first:stop])
        lengths.append((ends - starts)[np.isin(values, pm.POSSESSION_LABELS)])
    return np.concatenate(lengths) if lengths else np.array([], dtype=np.int64)


def save_match(path: Path, match: ExternalMatch) -> Path:
    return write_npz(path, {
        "ball": match.ball,
        "players": match.players,
        "slot_team": match.slot_team,
        "labels": match.labels,
        "periods": match.periods,
        "fps": np.array([match.fps]),
        "pitch": np.array(match.pitch),
        "name": np.array([match.name]),
        "source": np.array([match.source]),
    })


def load_match(path: Path) -> ExternalMatch:
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in MATCH_FIELDS}
    return ExternalMatch(
        name=str(values["name"][0]), source=str(values["source"][0]),
        fps=float(values["fps"][0]),
        pitch=(float(values["pitch"][0]), float(values["pitch"][1])), ball=values["ball"],
        players=values["players"], slot_team=values["slot_team"], labels=values["labels"],
        periods=values["periods"])
