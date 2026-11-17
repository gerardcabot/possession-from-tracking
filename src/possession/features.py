import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np

from possession.cleaning import CleanTracking
from possession.data import AWAY, HOME
from possession.geometry import home_attacks_positive, present

TEAMS = (HOME, AWAY)


@dataclass(frozen=True)
class Features:
    ball: np.ndarray
    ball_present: np.ndarray
    ball_velocity: np.ndarray
    ball_speed: np.ndarray
    nearest: np.ndarray
    centroid_velocity: np.ndarray
    mean_speed: np.ndarray
    attack_sign: np.ndarray
    tracked: np.ndarray
    centre_distance: np.ndarray
    pitch: tuple[float, float]
    fps: float
    second_nearest: np.ndarray | None = None
    centroid: np.ndarray | None = None
    defensive_line: np.ndarray | None = None
    team_speed: np.ndarray | None = None
    beyond_halfway: np.ndarray | None = None


@contextmanager
def quiet() -> Iterator[None]:
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def moving_nanmean(values: np.ndarray, window: int) -> np.ndarray:
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    pad = [(0, 0)] * values.ndim
    pad[0] = (1, 0)
    sums = np.pad(np.cumsum(filled, axis=0), pad)
    counts = np.pad(np.cumsum(finite, axis=0), pad)
    half = window // 2
    index = np.arange(len(values))
    low = np.clip(index - half, 0, len(values))
    high = np.clip(index + half + 1, 0, len(values))
    total = sums[high] - sums[low]
    count = counts[high] - counts[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def shifted(values: np.ndarray, offset: int) -> np.ndarray:
    result = np.full_like(values, np.nan)
    if offset > 0:
        result[:-offset] = values[offset:]
    elif offset < 0:
        result[-offset:] = values[:offset]
    else:
        result[:] = values
    return result


def central_velocity(positions: np.ndarray, fps: float, half_span: int) -> np.ndarray:
    velocity = np.full_like(positions, np.nan)
    estimates = [(span, -span, 2 * span) for span in range(half_span, 0, -1)]
    estimates += [(0, -1, 1), (1, 0, 1)]
    for ahead, behind, frames_apart in estimates:
        estimate = (shifted(positions, ahead) - shifted(positions, behind)) * fps / frames_apart
        fill = np.isnan(velocity[:, 0]) & ~np.isnan(estimate[:, 0])
        velocity[fill] = estimate[fill]
    return velocity


def player_speeds(players: np.ndarray, fps: float, half_span: int) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.stack([np.linalg.norm(central_velocity(players[:, slot], fps, half_span), axis=-1)
                         for slot in range(players.shape[1])], axis=1)


def ball_distances(ball: np.ndarray, players: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(players - ball[:, None, :], axis=-1)
    return np.where(np.isnan(distance), np.inf, distance)


def two_smallest(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if distance.shape[1] < 2:
        padded = np.concatenate([distance, np.full((len(distance), 2), np.inf)], axis=1)
        return two_smallest(padded)
    smallest = np.sort(np.partition(distance, 1, axis=1)[:, :2], axis=1)
    return smallest[:, 0], smallest[:, 1]


def team_nearest(distance: np.ndarray, slot_team: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs = [two_smallest(distance[:, slot_team == team]) for team in TEAMS]
    first = np.stack([pair[0] for pair in pairs], axis=1)
    second = np.stack([pair[1] for pair in pairs], axis=1)
    return first, second


def team_centroids(players: np.ndarray, slot_team: np.ndarray) -> np.ndarray:
    with quiet():
        return np.stack([np.nanmean(players[:, slot_team == team], axis=1) for team in TEAMS],
                        axis=1)


def team_speeds(speed: np.ndarray, slot_team: np.ndarray) -> np.ndarray:
    with quiet():
        return np.stack([np.nanmean(speed[:, slot_team == team], axis=1) for team in TEAMS],
                        axis=1)


def block_starts(blocks: list[tuple[int, int, int]]) -> np.ndarray:
    return np.array(sorted(first for _, first, _ in blocks), dtype=np.int64)


def block_index(n_frames: int, blocks: list[tuple[int, int, int]]) -> np.ndarray:
    starts = block_starts(blocks)
    return np.clip(np.searchsorted(starts, np.arange(n_frames), side="right") - 1, 0,
                   len(starts) - 1)


def attack_signs(players: np.ndarray, slot_team: np.ndarray,
                 blocks: list[tuple[int, int, int]]) -> np.ndarray:
    ordered = sorted(blocks, key=lambda block: block[1])
    signs = np.array([1 if home_attacks_positive(players, slot_team, slice(first, last + 1))
                      else -1 for _, first, last in ordered], dtype=np.int8)
    return signs[block_index(len(players), ordered)]


def defensive_lines(players: np.ndarray, slot_team: np.ndarray,
                    attack_sign: np.ndarray) -> np.ndarray:
    lines = []
    for team in TEAMS:
        direction = team_direction(attack_sign, team)
        depth = players[:, slot_team == team, 0] * direction[:, None]
        ordered = np.sort(np.where(np.isnan(depth), np.inf, depth), axis=1)
        line = ordered[:, 1] if ordered.shape[1] > 1 else ordered[:, 0]
        lines.append(np.where(np.isfinite(line), line * direction, np.nan))
    return np.stack(lines, axis=1)


def team_direction(attack_sign: np.ndarray, team: int) -> np.ndarray:
    return attack_sign * (1 if team == HOME else -1)


def tracked_counts(players: np.ndarray, slot_team: np.ndarray) -> np.ndarray:
    return np.stack([present(players[:, slot_team == team]).sum(axis=1) for team in TEAMS],
                    axis=1)


def beyond_halfway_counts(players: np.ndarray, slot_team: np.ndarray, attack_sign: np.ndarray,
                          length: float, margin: float) -> np.ndarray:
    beyond = []
    for team in TEAMS:
        x = players[:, slot_team == team, 0]
        forward = (x - length / 2) * team_direction(attack_sign, team)[:, None]
        with np.errstate(invalid="ignore"):
            beyond.append((forward > margin).sum(axis=1))
    return np.stack(beyond, axis=1)


def centre_distances(players: np.ndarray, slot_team: np.ndarray,
                     pitch: tuple[float, float]) -> np.ndarray:
    centre = np.asarray(pitch, dtype=float) / 2
    distance = np.linalg.norm(players - centre, axis=-1)
    distance = np.where(np.isnan(distance), np.inf, distance)
    return np.stack([distance[:, slot_team == team].min(axis=1) for team in TEAMS], axis=1)


def threshold_free_features(clean: CleanTracking, player_speed: np.ndarray,
                            pitch: tuple[float, float], fps: float,
                            blocks: list[tuple[int, int, int]], ball_half_span: int,
                            player_half_span: int) -> Features:
    players = clean.players
    speed = np.where(present(players), player_speed, np.nan)
    velocity = central_velocity(clean.ball, fps, ball_half_span)
    distance = ball_distances(clean.ball, players)
    nearest, second = team_nearest(distance, clean.slot_team)
    centroid = team_centroids(players, clean.slot_team)
    attack_sign = attack_signs(players, clean.slot_team, blocks)
    flat_centroid = centroid.reshape(len(centroid), -1)
    with quiet():
        mean_speed = np.nanmean(speed, axis=1)
    return Features(
        ball=clean.ball,
        ball_present=present(clean.ball),
        ball_velocity=velocity,
        ball_speed=np.linalg.norm(velocity, axis=1),
        nearest=nearest,
        centroid_velocity=central_velocity(flat_centroid, fps,
                                           player_half_span).reshape(centroid.shape),
        mean_speed=mean_speed,
        attack_sign=attack_sign,
        tracked=tracked_counts(players, clean.slot_team),
        centre_distance=centre_distances(players, clean.slot_team, pitch),
        pitch=pitch,
        fps=fps,
        second_nearest=second,
        centroid=centroid,
        defensive_line=defensive_lines(players, clean.slot_team, attack_sign),
        team_speed=team_speeds(speed, clean.slot_team),
    )


def build_features(clean: CleanTracking, player_speed: np.ndarray, pitch: tuple[float, float],
                   fps: float, blocks: list[tuple[int, int, int]], half_span: int,
                   shape_window_s: float, halfway_margin_m: float) -> Features:
    core = threshold_free_features(clean, player_speed, pitch, fps, blocks, half_span, half_span)
    flat = core.centroid_velocity.reshape(len(core.centroid_velocity), -1)
    window = int(round(shape_window_s * fps))
    return replace(
        core,
        centroid_velocity=moving_nanmean(flat, window).reshape(core.centroid_velocity.shape),
        beyond_halfway=beyond_halfway_counts(clean.players, clean.slot_team, core.attack_sign,
                                             pitch[0], halfway_margin_m),
    )
