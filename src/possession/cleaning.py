from dataclasses import asdict, dataclass

import numpy as np

from possession.data import Tracking
from possession.geometry import pair_distance, present, same_team_pairs, to_metres
from possession.intervals import true_runs


@dataclass(frozen=True)
class CleaningParams:
    ball_max_speed_mps: float
    ball_deviation_m: float
    player_max_speed_mps: float
    player_deviation_m: float
    duplicate_distance_m: float
    max_gap_frames: int
    ball_half_span: int
    player_half_span: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class CleanTracking:
    ball: np.ndarray
    ball_observed: np.ndarray
    ball_outlier: np.ndarray
    ball_interpolated: np.ndarray
    players: np.ndarray
    player_observed: np.ndarray
    player_jump: np.ndarray
    player_duplicate: np.ndarray
    slot_team: np.ndarray


def frame_numbers(valid: np.ndarray) -> np.ndarray:
    return np.arange(len(valid)).reshape((len(valid),) + (1,) * (valid.ndim - 1))


def previous_valid_index(valid: np.ndarray) -> np.ndarray:
    running = np.maximum.accumulate(np.where(valid, frame_numbers(valid), -1), axis=0)
    previous = np.full_like(running, -1)
    previous[1:] = running[:-1]
    return previous


def next_valid_index(valid: np.ndarray) -> np.ndarray:
    n_frames = len(valid)
    reversed_previous = previous_valid_index(valid[::-1])
    return np.where(reversed_previous >= 0, n_frames - 1 - reversed_previous, n_frames)[::-1]


def gather(positions: np.ndarray, index: np.ndarray) -> np.ndarray:
    clipped = np.clip(index, 0, len(positions) - 1)
    if index.ndim == 1:
        return positions[clipped]
    return positions[clipped, np.arange(index.shape[1])]


def implied_speed(start: np.ndarray, end: np.ndarray, frames_apart: np.ndarray,
                  fps: float) -> np.ndarray:
    return np.linalg.norm(end - start, axis=-1) * fps / frames_apart


def impossible_motion(positions: np.ndarray, max_speed: float, fps: float) -> np.ndarray:
    valid = present(positions)
    previous = previous_valid_index(valid)
    following = next_valid_index(valid)
    frames = frame_numbers(valid)
    speed_in = implied_speed(gather(positions, previous), positions, frames - previous, fps)
    speed_out = implied_speed(positions, gather(positions, following), following - frames, fps)
    has_both = valid & (previous >= 0) & (following < len(positions))
    return has_both & (speed_in > max_speed) & (speed_out > max_speed)


def neighbour_deviation(positions: np.ndarray) -> np.ndarray:
    deviation = np.full(positions.shape[:-1], np.nan)
    centre = positions[2:-2]
    with np.errstate(invalid="ignore"):
        from_left = 2 * positions[1:-3] - positions[:-4]
        from_right = 2 * positions[3:-1] - positions[4:]
        left = np.linalg.norm(centre - from_left, axis=-1)
        right = np.linalg.norm(centre - from_right, axis=-1)
    deviation[2:-2] = np.minimum(left, right)
    return deviation


def left_entries(positions: np.ndarray, max_deviation: float) -> np.ndarray:
    entries = np.zeros(len(positions), dtype=bool)
    with np.errstate(invalid="ignore"):
        from_left = 2 * positions[1:-1] - positions[:-2]
        entries[2:] = np.linalg.norm(positions[2:] - from_left, axis=-1) > max_deviation
    return entries & present(positions)


def displaced_run_end(positions: np.ndarray, start: int, max_deviation: float) -> int | None:
    n_frames = len(positions)
    velocity = positions[start - 1] - positions[start - 2]
    window = 64
    while True:
        stop = min(start + window, n_frames)
        offsets = np.arange(start, stop) - (start - 1)
        with np.errstate(invalid="ignore"):
            predicted = positions[start - 1] + velocity * offsets[:, None]
            displaced = np.linalg.norm(positions[start:stop] - predicted, axis=-1) > max_deviation
        if not displaced.all():
            return start + int(np.argmin(displaced))
        if stop == n_frames:
            return None
        window *= 4


def bridges_back(positions: np.ndarray, start: int, end: int, max_deviation: float) -> bool:
    if end + 1 >= len(positions) or not present(positions[end:end + 2]).all():
        return False
    backward = positions[end] + (positions[end] - positions[end + 1]) * (end - (start - 1))
    return bool(np.linalg.norm(positions[start - 1] - backward) <= max_deviation)


def displaced_runs(positions: np.ndarray, max_deviation: float) -> np.ndarray:
    flagged = np.zeros(len(positions), dtype=bool)
    valid = present(positions)
    for start in np.flatnonzero(left_entries(positions, max_deviation) & valid).tolist():
        if flagged[start] or not valid[start - 2:start].all():
            continue
        end = displaced_run_end(positions, start, max_deviation)
        if end is not None and valid[end] and bridges_back(positions, start, end,
                                                           max_deviation):
            flagged[start:end] = True
    return flagged


def step_speeds(positions: np.ndarray, fps: float) -> np.ndarray:
    speeds = np.full(positions.shape[:-1], np.nan)
    speeds[1:] = np.linalg.norm(positions[1:] - positions[:-1], axis=-1) * fps
    return speeds


def valley_before(speeds: np.ndarray, step: int) -> int:
    while step > 1 and speeds[step - 1] < speeds[step]:
        step -= 1
    return step


def valley_after(speeds: np.ndarray, step: int) -> int:
    while step + 1 < len(speeds) and speeds[step + 1] < speeds[step]:
        step += 1
    return step


def departure_from_both_sides(track: np.ndarray, first: int, last: int,
                              max_deviation: float) -> np.ndarray:
    frames = np.arange(first + 1, last - 1)
    with np.errstate(invalid="ignore"):
        before = track[first] + (track[first] - track[first - 1]) * (frames - first)[:, None]
        after = track[last - 1] + (track[last - 1] - track[last]) * (last - 1 - frames)[:, None]
        off_before = np.linalg.norm(track[frames] - before, axis=-1) > max_deviation
        off_after = np.linalg.norm(track[frames] - after, axis=-1) > max_deviation
    departed = np.zeros(len(track), dtype=bool)
    departed[frames] = off_before & off_after
    return departed


def excursion(track: np.ndarray, core: np.ndarray, fps: float,
              max_deviation: float) -> np.ndarray:
    speeds = step_speeds(track, fps)
    removed = core.copy()
    starts, ends = true_runs(core)
    for start, end in zip(starts.tolist(), ends.tolist()):
        first = valley_before(speeds, start)
        last = valley_after(speeds, end)
        departed = departure_from_both_sides(track, first, last, max_deviation)
        frame = start - 1
        while frame > first and departed[frame]:
            removed[frame] = True
            frame -= 1
        frame = end
        while frame < last - 1 and departed[frame]:
            removed[frame] = True
            frame += 1
    return removed & present(track)


def jump_segments(positions: np.ndarray, max_speed: float, max_deviation: float,
                  fps: float) -> np.ndarray:
    core = impossible_motion(positions, max_speed, fps)
    return np.stack([excursion(positions[:, slot], core[:, slot], fps, max_deviation)
                     for slot in range(positions.shape[1])], axis=1)


def without(positions: np.ndarray, mask: np.ndarray) -> np.ndarray:
    cleaned = positions.copy()
    cleaned[mask] = np.nan
    return cleaned


def interpolate_short_gaps(positions: np.ndarray, fps: float, max_gap: int,
                           max_speed: float) -> tuple[np.ndarray, np.ndarray]:
    valid = present(positions)
    previous = previous_valid_index(valid)
    following = next_valid_index(valid)
    frames = frame_numbers(valid)
    span = following - previous
    start = gather(positions, previous)
    end = gather(positions, following)
    bounded = ~valid & (previous >= 0) & (following < len(positions)) & (span - 1 <= max_gap)
    fill = bounded & (implied_speed(start, end, span, fps) <= max_speed)
    weight = ((frames - previous) / span)[..., None]
    filled = positions.copy()
    filled[fill] = (start + (end - start) * weight)[fill]
    return filled, fill


def duplicate_distances(players: np.ndarray, slot_team: np.ndarray,
                        min_distance: float) -> np.ndarray:
    partner_distance = np.full(players.shape[:2], np.inf)
    for first, second in same_team_pairs(slot_team):
        distance = pair_distance(players, first, second)
        close = distance < min_distance
        partner_distance[close, second] = np.minimum(partner_distance[close, second],
                                                     distance[close])
    return partner_distance


def merge_duplicates(players: np.ndarray, slot_team: np.ndarray,
                     min_distance: float) -> tuple[np.ndarray, np.ndarray]:
    duplicate = np.isfinite(duplicate_distances(players, slot_team, min_distance))
    merged = players.copy()
    merged[duplicate] = np.nan
    return merged, duplicate


def clean_ball(ball: np.ndarray, fps: float, params: CleaningParams) -> dict[str, np.ndarray]:
    outlier = displaced_runs(ball, params.ball_deviation_m)
    filled, interpolated = interpolate_short_gaps(without(ball, outlier), fps,
                                                  params.max_gap_frames,
                                                  params.ball_max_speed_mps)
    return {
        "ball": filled,
        "ball_observed": present(ball),
        "ball_outlier": outlier,
        "ball_interpolated": interpolated,
    }


def clean_players(players: np.ndarray, slot_team: np.ndarray, fps: float,
                  params: CleaningParams) -> dict[str, np.ndarray]:
    jump = jump_segments(players, params.player_max_speed_mps, params.player_deviation_m,
                         fps)
    merged, duplicate = merge_duplicates(without(players, jump), slot_team,
                                         params.duplicate_distance_m)
    return {
        "players": merged,
        "player_observed": present(players),
        "player_jump": jump,
        "player_duplicate": duplicate,
    }


def clean(tracking: Tracking, pitch: tuple[float, float], fps: float,
          params: CleaningParams) -> CleanTracking:
    with np.errstate(invalid="ignore", divide="ignore"):
        ball = clean_ball(to_metres(tracking.ball, pitch), fps, params)
        players = clean_players(to_metres(tracking.players, pitch), tracking.slot_team, fps,
                                params)
    return CleanTracking(**ball, **players, slot_team=tracking.slot_team)
