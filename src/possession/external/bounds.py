from dataclasses import dataclass

import numpy as np
from sklearn.mixture import GaussianMixture

from possession.cleaning import CleaningParams, neighbour_deviation
from possession.data import Tracking
from possession.external.match import ExternalMatch
from possession.external.stats import (
    Blocks,
    block_slices,
    run_lengths,
    white_noise_sigma,
)
from possession.features import central_velocity
from possession.geometry import nearest_teammate_distance, present, to_metres
from possession.intervals import DEAD, LABELS

MIXTURE_SEED = 0
SHORT_AND_LONG = 2


@dataclass(frozen=True)
class MatchExtremes:
    ball_speed_mps: float
    ball_deviation_m: float
    player_speed_mps: float
    player_deviation_m: float
    teammate_distance_m: float


@dataclass(frozen=True)
class AliveTrack:
    ball: np.ndarray
    players: np.ndarray
    slot_team: np.ndarray
    alive: np.ndarray
    blocks: Blocks
    fps: float


def alive_track(match: ExternalMatch) -> AliveTrack:
    return AliveTrack(ball=to_metres(match.ball, match.pitch),
                      players=to_metres(match.players, match.pitch), slot_team=match.slot_team,
                      alive=match.labels != LABELS[DEAD], blocks=match.blocks, fps=match.fps)


def consistent_speed(positions: np.ndarray, fps: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        speed_in = np.linalg.norm(positions[1:-1] - positions[:-2], axis=-1) * fps
        speed_out = np.linalg.norm(positions[2:] - positions[1:-1], axis=-1) * fps
        speed_skip = np.linalg.norm(positions[2:] - positions[:-2], axis=-1) * fps / 2
    return np.minimum(np.minimum(speed_in, speed_out), speed_skip)


def alive_triples(alive: np.ndarray) -> np.ndarray:
    return alive[:-2] & alive[1:-1] & alive[2:]


def max_alive_speed(positions: np.ndarray, track: AliveTrack) -> float:
    maxima = []
    for window in block_slices(track.blocks):
        speed = consistent_speed(positions[window], track.fps)
        keep = alive_triples(track.alive[window])
        values = speed[keep]
        values = values[np.isfinite(values)]
        if values.size:
            maxima.append(float(values.max()))
    return max(maxima) if maxima else float("nan")


def alive_neighbourhood(alive: np.ndarray) -> np.ndarray:
    return alive[:-4] & alive[1:-3] & alive[2:-2] & alive[3:-1] & alive[4:]


def max_alive_deviation(positions: np.ndarray, track: AliveTrack) -> float:
    maxima = []
    for window in block_slices(track.blocks):
        deviation = neighbour_deviation(positions[window])
        keep = alive_neighbourhood(track.alive[window])
        values = deviation[2:-2][keep]
        values = values[np.isfinite(values)]
        if values.size:
            maxima.append(float(values.max()))
    return max(maxima) if maxima else float("nan")


def min_alive_teammate_distance(track: AliveTrack) -> float:
    nearest = nearest_teammate_distance(track.players[track.alive], track.slot_team)
    finite = nearest[np.isfinite(nearest)]
    return float(finite.min()) if finite.size else float("nan")


def match_extremes(track: AliveTrack) -> MatchExtremes:
    return MatchExtremes(ball_speed_mps=max_alive_speed(track.ball, track),
                         ball_deviation_m=max_alive_deviation(track.ball, track),
                         player_speed_mps=max_alive_speed(track.players, track),
                         player_deviation_m=max_alive_deviation(track.players, track),
                         teammate_distance_m=min_alive_teammate_distance(track))


def robust_extreme(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        raise ValueError("no reference match gives a finite extreme")
    return float(np.median(finite))


def short_gap_limit(gap_lengths: np.ndarray) -> int:
    lengths = np.asarray(gap_lengths, dtype=float)
    if np.unique(lengths).size < SHORT_AND_LONG:
        return 0
    mixture = GaussianMixture(n_components=SHORT_AND_LONG, random_state=MIXTURE_SEED)
    mixture.fit(np.log(lengths)[:, None])
    short = int(np.argmin(mixture.means_[:, 0]))
    candidates = np.arange(1, int(lengths.max()) + 1)
    is_short = mixture.predict(np.log(candidates)[:, None]) == short
    if not is_short.any():
        return 0
    first = int(np.argmax(is_short))
    stops = np.flatnonzero(~is_short[first:])
    last = first + int(stops[0]) if stops.size else len(candidates)
    return int(candidates[last - 1])


def signal_series(positions: np.ndarray) -> list[np.ndarray]:
    if positions.ndim == 2:
        return [positions]
    return [positions[:, slot] for slot in range(positions.shape[1])]


def velocity_bias(tracks: list[AliveTrack], signal: str, half_span: int) -> float:
    total, count = 0.0, 0
    for track in tracks:
        for window in block_slices(track.blocks):
            alive = track.alive[window]
            for series in signal_series(getattr(track, signal)[window]):
                reference = central_velocity(series, track.fps, 1)
                wider = central_velocity(series, track.fps, half_span)
                error = np.sum((wider - reference) ** 2, axis=1)
                keep = alive & np.isfinite(error)
                total += float(error[keep].sum())
                count += int(keep.sum())
    return total / count if count else float("nan")


def velocity_noise(sigma: float, fps: float, half_span: int) -> float:
    return (sigma * fps / half_span) ** 2


def max_half_span(tracks: list[AliveTrack]) -> int:
    shortest = min(last - first + 1 for track in tracks for _, first, last in track.blocks)
    return max(shortest // 2, 1)


def velocity_half_span(tracks: list[AliveTrack], signal: str, sigma: float, fps: float) -> int:
    half_span = 1
    error = velocity_noise(sigma, fps, half_span)
    limit = max_half_span(tracks)
    while half_span < limit:
        wider = half_span + 1
        candidate = velocity_bias(tracks, signal, wider) + velocity_noise(sigma, fps, wider)
        if not candidate < error:
            return half_span
        half_span, error = wider, candidate
    return half_span


def estimate_params(references: list[ExternalMatch], tracking: Tracking,
                    pitch: tuple[float, float], fps: float, blocks: Blocks) -> CleaningParams:
    tracks = [alive_track(match) for match in references]
    extremes = [match_extremes(track) for track in tracks]
    ball = to_metres(tracking.ball, pitch)
    players = to_metres(tracking.players, pitch)
    return CleaningParams(
        ball_max_speed_mps=robust_extreme([item.ball_speed_mps for item in extremes]),
        ball_deviation_m=robust_extreme([item.ball_deviation_m for item in extremes]),
        player_max_speed_mps=robust_extreme([item.player_speed_mps for item in extremes]),
        player_deviation_m=robust_extreme([item.player_deviation_m for item in extremes]),
        duplicate_distance_m=robust_extreme([item.teammate_distance_m for item in extremes]),
        max_gap_frames=short_gap_limit(run_lengths(present(ball), blocks, False)),
        ball_half_span=velocity_half_span(tracks, "ball", white_noise_sigma(ball, blocks), fps),
        player_half_span=velocity_half_span(tracks, "players",
                                            white_noise_sigma(players, blocks), fps),
    )
