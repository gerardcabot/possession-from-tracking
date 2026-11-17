from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import maximum_filter1d
from sklearn.metrics import roc_curve

from possession.cleaning import CleaningParams, CleanTracking, clean
from possession.data import AWAY, HOME, Tracking
from possession.features import (
    Features,
    central_velocity,
    player_speeds,
    quiet,
    team_direction,
    threshold_free_features,
)
from possession.intervals import DEAD
from possession.vendor import possession_metrics as pm

PERSPECTIVES = (HOME, AWAY)
ROLLING_SIGNALS = ("ball_present", "ball_speed", "ball_x", "near_diff", "second_diff",
                   "shape", "mean_speed")
ROLLING_DIRECTIONS = ("past", "future")
CENTRED = "centred"
PLAYER_PAIRS = ("centroid_x", "centroid_vx", "defensive_line", "spread_x", "spread_y",
                "team_speed", "centre_distance")
PLAYER_SIGNALS = ("shape", "mean_speed")
DEAD_FEATURE_SETS = ("all", "players")
LINE_PLAYER_SIGNALS = ("player_out_touch", "player_out_goal", "line_player_speed")
KEEPER_SPEED = "keeper_speed"
KEEPER_PLAYER_PAIRS = ("keeper_depth", KEEPER_SPEED)
KEEPER_BALL_PAIRS = ("keeper_last_ball_distance",)
BALL_LINE_COLUMNS = ("since_out", "until_out", "entry_touch_distance", "exit_touch_distance",
                     "entry_goal_distance", "exit_goal_distance")
PLAYER_FEATURE_NAMES = frozenset(
    {f"{name}_{side}" for name in PLAYER_PAIRS for side in ("own", "other")} | set(PLAYER_SIGNALS))
KEEPER_PLAYER_NAMES = frozenset(
    f"{name}_{side}" for name in KEEPER_PLAYER_PAIRS for side in ("own", "other"))
LINE_ROLLING_SIGNALS = (*LINE_PLAYER_SIGNALS,
                        *(f"{KEEPER_SPEED}_{side}" for side in ("own", "other")))
LINE_GROUP_NAMES = {
    "lines": frozenset(LINE_PLAYER_SIGNALS),
    "keeper": KEEPER_PLAYER_NAMES | {f"{name}_{side}" for name in KEEPER_BALL_PAIRS
                                     for side in ("own", "other")},
    "ball": frozenset(BALL_LINE_COLUMNS),
}
LINE_GROUPS = tuple(LINE_GROUP_NAMES)
LINE_PLAYER_NAMES = frozenset(LINE_PLAYER_SIGNALS) | KEEPER_PLAYER_NAMES
TEAM_FEATURE_SETS = ("core", "short", "touch")
DEFAULT_TEAM_FEATURES = "touch"
SHORT_QUANTILES = (0.25, 0.5)
TOUCH_SIGNALS = ("touch_own", "touch_other", "touch_count_own", "touch_count_other")
TOUCH_COLUMNS = ("ball_impulse", "ball_speed_change", "ball_turn", "touch_peak")
TOUCH_EVENT_VALUES = ("team", "impulse", "turn", "closest")
TOUCH_SIDES = ("own", "other")
TOUCH_TEAM_EVENTS = ("touch_back_team", "touch_forward_team")
TOUCH_EVENTS = tuple(sorted(
    {f"touch_{event}_{value}" for event in ("back", "forward")
     for value in (*TOUCH_EVENT_VALUES, "time")}
    | {f"touch_{when}_{side}" for when in ("since", "until") for side in TOUCH_SIDES}
    | set(TOUCH_SIGNALS)))
Blocks = list[tuple[int, int, int]]
Segments = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class MatchInputs:
    features: Features
    spread: np.ndarray
    segments: Segments
    params: CleaningParams
    touch: np.ndarray | None = None
    lines: dict[str, np.ndarray] | None = None


@dataclass(frozen=True)
class BaseFeatures:
    names: tuple[str, ...]
    matrix: np.ndarray
    segments: Segments
    fps: float
    first: int = HOME
    touch: np.ndarray | None = None

    def column(self, name: str) -> np.ndarray:
        if name in TOUCH_COLUMNS:
            return self.touch_matrix[:, TOUCH_COLUMNS.index(name)]
        return self.matrix[:, self.names.index(name)]

    @property
    def touch_matrix(self) -> np.ndarray:
        if self.touch is None:
            raise ValueError("touch features were not computed for this match; build its "
                             "inputs with touch=True")
        return self.touch


@dataclass(frozen=True)
class FeatureLayout:
    scales: tuple[int, ...]
    short_scales: tuple[int, ...] = ()
    touch: bool = False
    touch_threshold: float = 0.0

    @property
    def core(self) -> "FeatureLayout":
        return FeatureLayout(scales=self.scales)

    @property
    def all_scales(self) -> tuple[int, ...]:
        return tuple(sorted({*self.scales, *self.short_scales}))


@dataclass(frozen=True)
class Rolling:
    signal: str
    direction: str
    scale: int

    @property
    def name(self) -> str:
        return f"{self.signal}_{self.direction}_{self.scale}"


def block_segments(n_frames: int, blocks: Blocks) -> Segments:
    edges = {0, n_frames}
    for _, first, last in blocks:
        edges.update((min(max(first, 0), n_frames), min(max(last + 1, 0), n_frames)))
    ordered = sorted(edges)
    return tuple(zip(ordered[:-1], ordered[1:]))


def by_segment(segments: Segments, function: Callable[[np.ndarray], np.ndarray],
               values: np.ndarray) -> np.ndarray:
    parts = [function(values[start:end]) for start, end in segments]
    return np.concatenate(parts) if parts else function(values[:0])


def match_inputs(tracking: Tracking, pitch: tuple[float, float], fps: float,
                 blocks: Blocks, params: CleaningParams, touch: bool = True,
                 lines: tuple[str, ...] = ()) -> MatchInputs:
    check_line_groups(lines)
    cleaned = clean(tracking, pitch, fps, params)
    players = cleaned.players
    speed = player_speeds(players, fps, params.player_half_span)
    features = threshold_free_features(cleaned, speed, pitch, fps, blocks, params.ball_half_span,
                                       params.player_half_span)
    with quiet():
        spread = np.stack([np.nanstd(players[:, cleaned.slot_team == team], axis=1)
                           for team in PERSPECTIVES], axis=1)
    segments = block_segments(len(features.ball), blocks)
    return MatchInputs(features=features, spread=spread, segments=segments, params=params,
                       touch=touch_matrix(features, segments, params.ball_half_span)
                       if touch else None,
                       lines=line_inputs(cleaned, speed, features, segments, lines)
                       if lines else None)


def finite(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values), values, np.nan)


def last_index(flags: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(np.where(flags, np.arange(len(flags)), -1))


def next_index(flags: np.ndarray) -> np.ndarray:
    n_frames = len(flags)
    reversed_last = last_index(flags[::-1])
    return np.where(reversed_last >= 0, n_frames - 1 - reversed_last, n_frames)[::-1]


def gather_at(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    valid = (index >= 0) & (index < len(values))
    result = np.full(len(index), np.nan)
    result[valid] = values[index[valid]]
    return result


def run_length_back(values: np.ndarray) -> np.ndarray:
    n_frames = len(values)
    if n_frames == 0:
        return np.zeros(0, dtype=np.int64)
    change = np.concatenate(([True], values[1:] != values[:-1]))
    starts = last_index(change)
    return np.arange(n_frames) - starts + 1


def run_length_forward(values: np.ndarray) -> np.ndarray:
    return run_length_back(values[::-1])[::-1]


def window_nanmean(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    sums = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0))))
    counts = np.concatenate(([0], np.cumsum(valid)))
    count = counts[high] - counts[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (sums[high] - sums[low]) / np.maximum(count, 1), np.nan)


def directional_nanmean(values: np.ndarray, window: int, forward: bool) -> np.ndarray:
    source = values[::-1] if forward else values
    high = np.arange(1, len(source) + 1)
    mean = window_nanmean(source, np.maximum(high - window, 0), high)
    return mean[::-1] if forward else mean


def centred_nanmean(values: np.ndarray, window: int) -> np.ndarray:
    back = window // 2
    index = np.arange(len(values))
    return window_nanmean(values, np.maximum(index - back, 0),
                          np.minimum(index + window - back, len(values)))


def rolling_scales(run_lengths: np.ndarray, run_states: np.ndarray,
                   run_weights: np.ndarray) -> tuple[int, ...]:
    lengths = np.asarray(run_lengths, dtype=float)
    weights = np.asarray(run_weights, dtype=float)
    dead = np.asarray(run_states) == DEAD
    scales = [np.average(lengths[group], weights=weights[group]) for group in (~dead, dead)
              if group.any()]
    return tuple(sorted({max(1, round(scale)) for scale in scales}))


def short_scales(run_lengths: np.ndarray, run_states: np.ndarray, run_weights: np.ndarray,
                 scales: tuple[int, ...]) -> tuple[int, ...]:
    lengths = np.asarray(run_lengths, dtype=float)
    alive = np.asarray(run_states) != DEAD
    if not alive.any():
        return ()
    values = np.quantile(lengths[alive], SHORT_QUANTILES,
                         weights=np.asarray(run_weights, dtype=float)[alive],
                         method="inverted_cdf")
    return tuple(sorted({max(1, round(float(value))) for value in values} - set(scales)))


def feature_layout(team_features: str, run_lengths: np.ndarray, run_states: np.ndarray,
                   run_weights: np.ndarray, touch_threshold: float = 0.0) -> FeatureLayout:
    if team_features not in TEAM_FEATURE_SETS:
        raise ValueError(f"unknown team feature set {team_features!r}; choose from "
                         f"{TEAM_FEATURE_SETS}")
    scales = rolling_scales(run_lengths, run_states, run_weights)
    if team_features == "core":
        return FeatureLayout(scales=scales)
    return FeatureLayout(scales=scales,
                         short_scales=short_scales(run_lengths, run_states, run_weights, scales),
                         touch=team_features == "touch",
                         touch_threshold=touch_threshold if team_features == "touch" else 0.0)


def along(values: np.ndarray, sign: np.ndarray, centre: float) -> np.ndarray:
    return sign * (values - centre)


def nearest_team(nearest: np.ndarray, first: int) -> np.ndarray:
    own, other = nearest[:, first], nearest[:, 1 - first]
    decided = np.isfinite(np.minimum(own, other)) & (own != other)
    return np.where(decided, np.where(own < other, 1, -1), 0).astype(np.int8)


def dwell_columns(nearest: np.ndarray, first: int, fps: float,
                  segments: Segments) -> dict[str, np.ndarray]:
    team = nearest_team(nearest, first)
    return {
        "dwell_back": team * by_segment(segments, run_length_back, team) / fps,
        "dwell_forward": team * by_segment(segments, run_length_forward, team) / fps,
    }


def segment_indices(flags: np.ndarray, segments: Segments) -> tuple[np.ndarray, np.ndarray]:
    n_frames = len(flags)
    before = np.full(n_frames, -1, dtype=np.int64)
    after = np.full(n_frames, n_frames, dtype=np.int64)
    for start, end in segments:
        last, following = last_index(flags[start:end]), next_index(flags[start:end])
        before[start:end] = np.where(last >= 0, last + start, -1)
        after[start:end] = np.where(following < end - start, following + start, n_frames)
    return before, after


def since_until(flags: np.ndarray, fps: float, segments: Segments
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = np.arange(len(flags))
    before, after = segment_indices(flags, segments)
    since = np.where(before >= 0, frame - before, np.nan) / fps
    until = np.where(after < len(flags), after - frame, np.nan) / fps
    return before, after, since, until


def gap_columns(features: Features, near_diff: np.ndarray, ball_x: np.ndarray,
                segments: Segments) -> dict[str, np.ndarray]:
    seen = features.ball_present
    before, after, since, until = since_until(seen, features.fps, segments)
    return {
        "since_seen": since,
        "until_seen": until,
        "gap_length": np.where(seen, 0.0, since + until),
        "entry_near_diff": gather_at(near_diff, before),
        "exit_near_diff": gather_at(near_diff, after),
        "entry_ball_x": gather_at(ball_x, before),
        "exit_ball_x": gather_at(ball_x, after),
    }


def line_groups_in(names: Iterable[str]) -> tuple[str, ...]:
    present = set(names)
    return tuple(group for group, columns in LINE_GROUP_NAMES.items() if columns & present)


def check_line_groups(groups: tuple[str, ...]) -> None:
    unknown = sorted(set(groups) - set(LINE_GROUPS))
    if unknown:
        raise ValueError(f"unknown line feature groups {unknown}; choose from {LINE_GROUPS}")


def beyond_lines(positions: np.ndarray, pitch: tuple[float, float]
                 ) -> tuple[np.ndarray, np.ndarray]:
    length, width = pitch
    touch = np.abs(positions[..., 1] - width / 2) - width / 2
    goal = np.abs(positions[..., 0] - length / 2) - length / 2
    return touch, goal


def furthest_beyond(values: np.ndarray) -> np.ndarray:
    with quiet():
        return np.nanmax(values, axis=1)


def lines_columns(players: np.ndarray, speed: np.ndarray,
                  pitch: tuple[float, float]) -> dict[str, np.ndarray]:
    touch, goal = beyond_lines(players, pitch)
    edge = np.maximum(touch, goal)
    nearest = np.argmax(np.where(np.isnan(edge), -np.inf, edge), axis=1)
    rows = np.arange(len(edge))
    return {"player_out_touch": furthest_beyond(touch),
            "player_out_goal": furthest_beyond(goal),
            "line_player_speed": np.where(np.isnan(edge[rows, nearest]), np.nan,
                                          speed[rows, nearest])}


def keeper_columns(players: np.ndarray, speed: np.ndarray, slot_team: np.ndarray,
                   features: Features, segments: Segments) -> dict[str, np.ndarray]:
    length = features.pitch[0]
    before, _ = segment_indices(features.ball_present, segments)
    last_ball = np.where((before >= 0)[:, None], features.ball[np.maximum(before, 0)], np.nan)
    depth, still, distance = [], [], []
    for team in PERSPECTIVES:
        slots = np.flatnonzero(slot_team == team)
        if slots.size == 0:
            missing = np.full(len(players), np.nan)
            depth.append(missing)
            still.append(missing)
            distance.append(missing)
            continue
        forward = (players[:, slots, 0] - length / 2) * team_direction(features.attack_sign,
                                                                       team)[:, None]
        ordered = np.where(np.isnan(forward), np.inf, forward)
        deepest = np.argmin(ordered, axis=1)
        rows = np.arange(len(ordered))
        found = np.isfinite(ordered[rows, deepest])
        keeper = players[rows, slots[deepest]]
        depth.append(np.where(found, length / 2 + ordered[rows, deepest], np.nan))
        still.append(np.where(found, speed[rows, slots[deepest]], np.nan))
        distance.append(np.where(found, np.linalg.norm(keeper - last_ball, axis=1), np.nan))
    return {"keeper_depth": np.stack(depth, axis=1), KEEPER_SPEED: np.stack(still, axis=1),
            "keeper_last_ball_distance": np.stack(distance, axis=1)}


def ball_line_columns(features: Features, segments: Segments) -> dict[str, np.ndarray]:
    touch, goal = beyond_lines(features.ball, features.pitch)
    seen = features.ball_present
    with quiet():
        outside = seen & ((touch > 0) | (goal > 0))
    _, _, since, until = since_until(outside, features.fps, segments)
    before, after = segment_indices(seen, segments)
    return {"since_out": since, "until_out": until,
            "entry_touch_distance": gather_at(-touch, before),
            "exit_touch_distance": gather_at(-touch, after),
            "entry_goal_distance": gather_at(-goal, before),
            "exit_goal_distance": gather_at(-goal, after)}


def line_inputs(cleaned: CleanTracking, speed: np.ndarray, features: Features,
                segments: Segments, groups: tuple[str, ...]) -> dict[str, np.ndarray]:
    columns: dict[str, np.ndarray] = {}
    if "lines" in groups:
        columns.update(lines_columns(cleaned.players, speed, features.pitch))
    if "keeper" in groups:
        columns.update(keeper_columns(cleaned.players, speed, cleaned.slot_team, features,
                                      segments))
    if "ball" in groups:
        columns.update(ball_line_columns(features, segments))
    return columns


def line_base_columns(lines: dict[str, np.ndarray], first: int) -> dict[str, np.ndarray]:
    columns: dict[str, np.ndarray] = {}
    for name, values in lines.items():
        if values.ndim == 2:
            columns.update(pair_columns(name, values, first))
        else:
            columns[name] = values
    return columns


def ball_columns(features: Features, sign: np.ndarray, half_span: int) -> dict[str, np.ndarray]:
    length, width = features.pitch
    x, y = features.ball[:, 0], features.ball[:, 1]
    acceleration = central_velocity(features.ball_velocity, features.fps, half_span)
    return {
        "ball_present": features.ball_present.astype(float),
        "ball_x": along(x, sign, length / 2),
        "ball_y": np.abs(y - width / 2),
        "ball_speed": features.ball_speed,
        "ball_vx": sign * features.ball_velocity[:, 0],
        "ball_acceleration": np.linalg.norm(acceleration, axis=1),
        "goal_line_distance": length / 2 - np.abs(x - length / 2),
        "touch_line_distance": width / 2 - np.abs(y - width / 2),
    }


def pair_columns(name: str, values: np.ndarray, first: int) -> dict[str, np.ndarray]:
    return {f"{name}_own": values[:, first], f"{name}_other": values[:, 1 - first]}


def team_columns(inputs: MatchInputs, first: int, sign: np.ndarray) -> dict[str, np.ndarray]:
    features = inputs.features
    length = features.pitch[0]
    centroid_x = along(features.centroid[:, :, 0], sign[:, None], length / 2)
    centroid_vx = sign[:, None] * features.centroid_velocity[:, :, 0]
    line = along(features.defensive_line, sign[:, None], length / 2)
    columns = {}
    for name, values in (("nearest", finite(features.nearest)),
                         ("second_nearest", finite(features.second_nearest)),
                         ("centroid_x", centroid_x), ("centroid_vx", centroid_vx),
                         ("defensive_line", line), ("spread_x", inputs.spread[:, :, 0]),
                         ("spread_y", inputs.spread[:, :, 1]), ("team_speed", features.team_speed),
                         ("tracked", features.tracked.astype(float)),
                         ("centre_distance", finite(features.centre_distance))):
        columns.update(pair_columns(name, values, first))
    return columns


def touch_half_span(fps: float) -> int:
    return max(1, round(pm.TAUS[0] * fps / pm.FPS))


def velocity_around(velocity: np.ndarray, window: int) -> np.ndarray:
    before = np.stack([directional_nanmean(velocity[:, axis], window, forward=False)
                       for axis in range(velocity.shape[1])], axis=1)
    after = np.full_like(before, np.nan)
    after[:-1] = np.stack([directional_nanmean(velocity[1:, axis], window, forward=True)
                           for axis in range(velocity.shape[1])], axis=1)
    return np.concatenate([before, after], axis=1)


def local_peaks(values: np.ndarray, half_span: int) -> np.ndarray:
    if not len(values):
        return np.zeros(0, dtype=bool)
    filled = np.where(np.isfinite(values), values, -np.inf)
    window = maximum_filter1d(filled, size=2 * half_span + 1, mode="constant", cval=-np.inf)
    padded = np.concatenate((np.full(half_span, -np.inf), filled[:-1]))
    earlier = sliding_window_view(padded, half_span).max(axis=1)
    return (filled > 0) & (filled == window) & (filled > earlier)


def ball_impulses(velocity: np.ndarray, half_span: int,
                  segments: Segments) -> dict[str, np.ndarray]:
    around = by_segment(segments, partial(velocity_around, window=half_span), velocity)
    before, after = around[:, :2], around[:, 2:]
    speed_before = np.linalg.norm(before, axis=1)
    speed_after = np.linalg.norm(after, axis=1)
    cross = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
    dot = np.einsum("ij,ij->i", before, after)
    return {
        "ball_impulse": np.linalg.norm(after - before, axis=1),
        "ball_turn": np.abs(np.arctan2(cross, dot)),
        "ball_speed_change": speed_after - speed_before,
    }


def block_velocity(features: Features, segments: Segments, ball_half_span: int) -> np.ndarray:
    velocity = partial(central_velocity, fps=features.fps, half_span=ball_half_span)
    return by_segment(segments, velocity, features.ball)


def touch_columns(features: Features, segments: Segments,
                  ball_half_span: int) -> dict[str, np.ndarray]:
    half_span = touch_half_span(features.fps)
    velocity = block_velocity(features, segments, ball_half_span)
    present = features.ball_present.astype(bool)
    columns = {name: np.where(present, values, np.nan) for name, values in
               ball_impulses(velocity, half_span, segments).items()}
    impulse = columns["ball_impulse"]
    peak = by_segment(segments, partial(local_peaks, half_span=half_span), impulse)
    columns["touch_peak"] = np.where(peak, 1.0, np.where(np.isfinite(impulse), 0.0, np.nan))
    return columns


def touch_matrix(features: Features, segments: Segments, ball_half_span: int) -> np.ndarray:
    columns = touch_columns(features, segments, ball_half_span)
    return np.stack([columns[name] for name in TOUCH_COLUMNS], axis=1).astype(np.float32)


def touch_teams(base: BaseFeatures) -> np.ndarray:
    own = np.nan_to_num(base.column("nearest_own").astype(float), nan=np.inf)
    other = np.nan_to_num(base.column("nearest_other").astype(float), nan=np.inf)
    return np.where(own < other, 1.0, np.where(other < own, -1.0, 0.0))


def touch_frames(base: BaseFeatures, threshold: float) -> np.ndarray:
    impulse = base.column("ball_impulse").astype(float)
    return (base.column("touch_peak") == 1) & (np.nan_to_num(impulse, nan=-np.inf) >= threshold)


def swapped_side(name: str) -> str:
    for side, other in zip(TOUCH_SIDES, TOUCH_SIDES[::-1]):
        if name.endswith(f"_{side}"):
            return f"{name[:-len(side)]}{other}"
    return name


def other_perspective(columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    swapped = {swapped_side(name): values for name, values in columns.items()}
    for name in TOUCH_TEAM_EVENTS:
        swapped[name] = 0.0 - swapped[name]
    return swapped


class TouchEventCache:
    def __init__(self) -> None:
        self.touch: np.ndarray | None = None
        self.threshold = 0.0
        self.first = HOME
        self.columns: dict[str, np.ndarray] = {}

    def events(self, base: BaseFeatures, threshold: float) -> dict[str, np.ndarray]:
        if self.touch is not base.touch_matrix or self.threshold != threshold:
            self.columns = perspective_events(base, threshold)
            self.touch, self.threshold, self.first = base.touch_matrix, threshold, base.first
        if base.first == self.first:
            return dict(self.columns)
        return other_perspective(self.columns)


LAST_TOUCH_EVENTS = TouchEventCache()


def touch_events(base: BaseFeatures, threshold: float) -> dict[str, np.ndarray]:
    return LAST_TOUCH_EVENTS.events(base, threshold)


def perspective_events(base: BaseFeatures, threshold: float) -> dict[str, np.ndarray]:
    touch = touch_frames(base, threshold)
    team = touch_teams(base)
    impulse = base.column("ball_impulse").astype(float)
    with quiet():
        closest = np.nanmin(np.stack([base.column("nearest_own"), base.column("nearest_other")],
                                     axis=1).astype(float), axis=1)
    values = {"team": team, "impulse": impulse, "turn": base.column("ball_turn").astype(float),
              "closest": closest}
    before, after, since, until = since_until(touch, base.fps, base.segments)
    columns = {"touch_back_time": since, "touch_forward_time": until}
    for name, series in values.items():
        columns[f"touch_back_{name}"] = gather_at(series, before)
        columns[f"touch_forward_{name}"] = gather_at(series, after)
    empty = np.where(np.isfinite(impulse), 0.0, np.nan)
    for side, sign in zip(TOUCH_SIDES, (1.0, -1.0)):
        flags = touch & (team == sign)
        _, _, columns[f"touch_since_{side}"], columns[f"touch_until_{side}"] = since_until(
            flags, base.fps, base.segments)
        columns[f"touch_{side}"] = np.where(flags, impulse, empty)
        columns[f"touch_count_{side}"] = np.where(flags, 1.0, empty)
    return columns


def peak_evidence(base: BaseFeatures, codes: np.ndarray, blocks: Blocks
                  ) -> tuple[np.ndarray, np.ndarray]:
    span = touch_half_span(base.fps)
    impulse = base.column("ball_impulse").astype(float)
    peak = base.column("touch_peak") == 1
    scores, changes = [], []
    n_frames = len(peak)
    for _, start, last in blocks:
        first, end = min(max(start, 0), n_frames), min(max(last + 1, 0), n_frames)
        frames = np.flatnonzero(peak[first:end])
        sequence = codes[first:end]
        home = np.concatenate(([0], np.cumsum(sequence == HOME)))
        away = np.concatenate(([0], np.cumsum(sequence == AWAY)))
        low = np.maximum(frames - span, 0)
        high = np.minimum(frames + span + 1, len(sequence))
        n_home, n_away, length = home[high] - home[low], away[high] - away[low], high - low
        change = (n_home > 0) & (n_away > 0)
        used = change | (n_home == length) | (n_away == length)
        scores.append(impulse[frames[used] + first])
        changes.append(change[used])
    if not scores:
        return np.zeros(0), np.zeros(0, dtype=bool)
    return np.concatenate(scores), np.concatenate(changes)


def youden_threshold(scores: np.ndarray, positive: np.ndarray, weights: np.ndarray) -> float:
    if positive.all() or not positive.any():
        raise ValueError(f"touch threshold needs both changes and steady spells among the "
                         f"{len(positive)} labelled impulse peaks, got "
                         f"{int(positive.sum())} changes")
    false_rate, true_rate, thresholds = roc_curve(positive, scores, sample_weight=weights)
    best = 1 + int(np.argmax((true_rate - false_rate)[1:]))
    return float(thresholds[best])


def base_columns(inputs: MatchInputs, first: int) -> dict[str, np.ndarray]:
    features = inputs.features
    sign = team_direction(features.attack_sign, first).astype(float)
    columns = ball_columns(features, sign, inputs.params.ball_half_span)
    columns.update(team_columns(inputs, first, sign))
    columns["near_diff"] = columns["nearest_own"] - columns["nearest_other"]
    columns["second_diff"] = columns["second_nearest_own"] - columns["second_nearest_other"]
    columns["shape"] = columns["centroid_vx_own"] - columns["centroid_vx_other"]
    columns["mean_speed"] = features.mean_speed
    columns.update(dwell_columns(features.nearest, first, features.fps, inputs.segments))
    columns.update(gap_columns(features, columns["near_diff"], columns["ball_x"], inputs.segments))
    if inputs.lines:
        columns.update(line_base_columns(inputs.lines, first))
    return {name: np.asarray(values, dtype=np.float32) for name, values in columns.items()}


def base_features(inputs: MatchInputs, first: int) -> BaseFeatures:
    columns = base_columns(inputs, first)
    names = tuple(sorted(columns))
    return BaseFeatures(names=names, matrix=np.stack([columns[name] for name in names], axis=1),
                        segments=inputs.segments, fps=inputs.features.fps, first=first,
                        touch=inputs.touch)


Spec = str | Rolling


def rolling_specs(signals: tuple[str, ...], scales: tuple[int, ...]) -> list[Rolling]:
    return [Rolling(signal, direction, scale) for signal in signals
            for direction in ROLLING_DIRECTIONS for scale in scales]


def rolling_signals(base: BaseFeatures) -> tuple[str, ...]:
    return ROLLING_SIGNALS + tuple(name for name in LINE_ROLLING_SIGNALS if name in base.names)


def feature_specs(base: BaseFeatures, layout: FeatureLayout) -> list[Spec]:
    specs: list[Spec] = list(base.names)
    signals = rolling_signals(base)
    specs += rolling_specs(signals, layout.scales)
    specs += rolling_specs(signals, layout.short_scales)
    if layout.touch:
        specs += list(TOUCH_COLUMNS)
        specs += list(TOUCH_EVENTS)
        specs += rolling_specs(TOUCH_SIGNALS, layout.all_scales)
    return specs


def spec_name(spec: Spec) -> str:
    return spec if isinstance(spec, str) else spec.name


def feature_names(base: BaseFeatures, layout: FeatureLayout) -> list[str]:
    return [spec_name(spec) for spec in feature_specs(base, layout)]


def rolling_column(segments: Segments, values: np.ndarray, spec: Rolling) -> np.ndarray:
    if spec.direction == CENTRED:
        return by_segment(segments, partial(centred_nanmean, window=spec.scale), values)
    mean = partial(directional_nanmean, window=spec.scale, forward=spec.direction == "future")
    return by_segment(segments, mean, values)


def is_player_feature(name: str) -> bool:
    if name in PLAYER_FEATURE_NAMES or name in LINE_PLAYER_NAMES:
        return True
    player_signals = (*PLAYER_SIGNALS, *LINE_ROLLING_SIGNALS)
    return any(name.startswith(f"{signal}_{direction}_") for signal in player_signals
               for direction in ROLLING_DIRECTIONS)


def dead_columns(names: list[str], feature_set: str) -> tuple[int, ...]:
    if feature_set not in DEAD_FEATURE_SETS:
        raise ValueError(f"unknown dead-ball feature set {feature_set!r}; choose from "
                         f"{DEAD_FEATURE_SETS}")
    if feature_set == "all":
        return tuple(range(len(names)))
    columns = tuple(index for index, name in enumerate(names) if is_player_feature(name))
    if not columns:
        raise ValueError("no player features to train the dead-ball model on")
    return columns


def feature_matrix(base: BaseFeatures, layout: FeatureLayout,
                   keep: np.ndarray | None = None,
                   columns: tuple[int, ...] | None = None) -> np.ndarray:
    rows = slice(None) if keep is None else keep
    n_rows = len(base.matrix) if keep is None else int(keep.sum())
    specs = feature_specs(base, layout)
    chosen = range(len(specs)) if columns is None else columns
    matrix = np.empty((n_rows, len(chosen)), dtype=np.float32)
    base_rows = np.asarray(base.matrix[rows])
    touch_values = touch_events(base, layout.touch_threshold) if layout.touch else {}
    rolling_inputs: dict[str, np.ndarray] = {}
    for column, index in enumerate(chosen):
        spec = specs[index]
        if isinstance(spec, Rolling):
            if spec.signal not in rolling_inputs:
                rolling_inputs[spec.signal] = (touch_values[spec.signal]
                                               if spec.signal in touch_values
                                               else base.column(spec.signal).astype(float))
            matrix[:, column] = rolling_column(base.segments, rolling_inputs[spec.signal],
                                               spec)[rows]
        elif spec in touch_values:
            matrix[:, column] = touch_values[spec][rows]
        elif spec in TOUCH_COLUMNS:
            matrix[:, column] = base.column(spec)[rows]
        else:
            matrix[:, column] = base_rows[:, base.names.index(spec)]
    return matrix
