from dataclasses import dataclass

import numpy as np
from scipy.stats import gaussian_kde

from possession.intervals import DEAD, value_runs

N_STATES = 3
ALIVE = np.arange(N_STATES) != DEAD
SWITCHES = ALIVE[:, None] & ALIVE[None, :] & ~np.eye(N_STATES, dtype=bool)


@dataclass(frozen=True)
class Durations:
    body: np.ndarray
    tail_mass: float
    hazard: float


@dataclass(frozen=True)
class SegmentModel:
    durations: tuple[Durations, ...]
    log_transition: np.ndarray
    log_initial: np.ndarray


@dataclass(frozen=True)
class StepDurations:
    log_pmf: np.ndarray
    log_survival: np.ndarray
    log_first: np.ndarray
    log_first_survival: np.ndarray
    log_ratio: float


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    state: int


def safe_log(values: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(values)


def fit_durations(lengths: np.ndarray, weights: np.ndarray) -> Durations:
    lengths = np.asarray(lengths, dtype=float)
    if lengths.size == 0:
        raise ValueError("cannot fit a duration distribution without runs")
    if np.unique(lengths).size < 2:
        return Durations(body=np.zeros(0), tail_mass=1.0, hazard=1.0 / float(lengths.mean()))
    kde = gaussian_kde(np.log(lengths), weights=weights)
    edges = safe_log(np.arange(int(lengths.max()) + 1, dtype=float))
    cdf = np.array([kde.integrate_box_1d(-np.inf, edge) for edge in edges])
    body = np.diff(cdf)
    tail_mass = float(kde.integrate_box_1d(edges[-1], np.inf))
    hazard = float(body[-1] / (body[-1] + tail_mass))
    if not 0.0 < hazard < 1.0:
        raise ValueError(f"degenerate duration tail (hazard {hazard})")
    return Durations(body=body, tail_mass=tail_mass, hazard=hazard)


def frame_pmf(durations: Durations, n_frames: int) -> np.ndarray:
    beyond = np.arange(1, max(n_frames - len(durations.body), 0) + 1)
    tail = durations.tail_mass * durations.hazard * (1.0 - durations.hazard) ** (beyond - 1)
    return np.concatenate([durations.body, tail])[:n_frames]


def match_runs(sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    runs = [value_runs(codes) for codes in sequences]
    lengths = [ends - starts for starts, ends, _ in runs]
    values = [values for _, _, values in runs]
    return (np.concatenate(lengths).astype(np.int64) if lengths else np.zeros(0, np.int64),
            np.concatenate(values).astype(np.int64) if values else np.zeros(0, np.int64))


def per_match(count: int) -> np.ndarray:
    return np.full(count, 1.0 / count) if count else np.zeros(0)


def weighted_runs(matches: list[list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths, states, weights = [], [], []
    for sequences in matches:
        match_lengths, match_states = match_runs(sequences)
        lengths.append(match_lengths)
        states.append(match_states)
        weights.append(per_match(len(match_lengths)))
    return np.concatenate(lengths), np.concatenate(states), np.concatenate(weights)


def transition_counts(matches: list[list[np.ndarray]]) -> np.ndarray:
    counts = np.zeros((N_STATES, N_STATES))
    for sequences in matches:
        pairs = [value_runs(codes)[2] for codes in sequences]
        origins = (np.concatenate([values[:-1] for values in pairs]) if pairs
                   else np.zeros(0, np.int64))
        targets = (np.concatenate([values[1:] for values in pairs]) if pairs
                   else np.zeros(0, np.int64))
        np.add.at(counts, (origins, targets), per_match(len(origins)))
    return counts


def initial_counts(matches: list[list[np.ndarray]]) -> np.ndarray:
    counts = np.zeros(N_STATES)
    for sequences in matches:
        firsts = np.array([int(codes[0]) for codes in sequences if len(codes)], dtype=np.int64)
        np.add.at(counts, firsts, per_match(len(firsts)))
    return counts


def normalised_log(counts: np.ndarray) -> np.ndarray:
    total = counts.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return safe_log(np.where(total > 0, counts / np.where(total > 0, total, 1.0), 0.0))


def fit_segment_model(matches: list[list[np.ndarray]]) -> SegmentModel:
    lengths, states, weights = weighted_runs(matches)
    return SegmentModel(durations=tuple(fit_durations(lengths[states == state],
                                                      weights[states == state])
                                        for state in range(N_STATES)),
                        log_transition=normalised_log(transition_counts(matches)),
                        log_initial=normalised_log(initial_counts(matches)))


def reverse_cumsum(values: np.ndarray) -> np.ndarray:
    return np.cumsum(values[::-1])[::-1]


def step_durations(durations: Durations, stride: int) -> StepDurations:
    explicit = -(-len(durations.body) // stride) + 1
    steps = frame_pmf(durations, explicit * stride).reshape(explicit, stride).sum(axis=1)
    log_ratio = stride * float(np.log1p(-durations.hazard))
    beyond = np.exp(log_ratio) / -np.expm1(log_ratio)
    survival = reverse_cumsum(steps) + steps[-1] * beyond
    survival_beyond = survival[-1] * beyond
    mean = survival.sum() + survival_beyond
    return StepDurations(log_pmf=safe_log(steps), log_survival=safe_log(survival),
                         log_first=safe_log(survival / mean),
                         log_first_survival=safe_log((reverse_cumsum(survival) + survival_beyond)
                                                     / mean),
                         log_ratio=log_ratio)


def duration_tables(durations: StepDurations, final: bool) -> tuple[np.ndarray, np.ndarray]:
    if final:
        return durations.log_survival, durations.log_first_survival
    return durations.log_pmf, durations.log_first


def segment_choice(entry: np.ndarray, cumulative: np.ndarray, durations: StepDurations, t: int,
                   final: bool, tail: tuple[float, int]) -> tuple[float, int]:
    table, first = duration_tables(durations, final)
    explicit = len(table)
    low = max(0, t - explicit)
    duration = table[:t - low][::-1].copy()
    if low == 0:
        duration[0] = first[t - 1]
    scores = entry[low:t] + duration + cumulative[t] - cumulative[low:t]
    index = int(np.argmax(scores))
    choices = [(float(scores[index]), low + index)]
    if t > explicit:
        overflow = (t - explicit) * durations.log_ratio + cumulative[t]
        tail_value, tail_start = tail
        if tail_start > 0:
            choices.insert(0, (tail_value + table[-1] + overflow, tail_start))
        choices.insert(0, (float(entry[0] + first[-1] + overflow), 0))
    return max(choices, key=lambda choice: (choice[0], -choice[1]))


def forward_pass(emission: np.ndarray, durations: list[StepDurations], model: SegmentModel,
                 switch: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_steps, n_states = emission.shape
    cumulative = np.vstack([np.zeros(n_states), np.cumsum(emission, axis=0)])
    entry = np.full((n_steps + 1, n_states), -np.inf)
    entry[0] = model.log_initial
    came_from = np.zeros((n_steps + 1, n_states), dtype=np.int64)
    started = np.zeros((n_steps + 1, n_states), dtype=np.int64)
    best = np.full((n_steps + 1, n_states), -np.inf)
    tails = [(-np.inf, 0)] * n_states
    for t in range(1, n_steps + 1):
        for state in range(n_states):
            steps = durations[state]
            start = t - len(steps.log_pmf) - 1
            if start > 0:
                value = float(entry[start, state] - cumulative[start, state]
                              - start * steps.log_ratio)
                if value > tails[state][0]:
                    tails[state] = (value, start)
            best[t, state], started[t, state] = segment_choice(
                entry[:, state], cumulative[:, state], steps, t, t == n_steps, tails[state])
        if t < n_steps:
            candidates = best[t][:, None] + model.log_transition
            if switch is not None:
                candidates = candidates + np.where(SWITCHES, switch[t][None, :], 0.0)
            came_from[t] = np.argmax(candidates, axis=0)
            entry[t] = candidates[came_from[t], np.arange(n_states)]
    return best, started, came_from


def viterbi_segments(emission: np.ndarray, model: SegmentModel, stride: int,
                     switch: np.ndarray | None = None) -> list[Segment]:
    if len(emission) == 0:
        return []
    durations = [step_durations(fitted, stride) for fitted in model.durations]
    best, started, came_from = forward_pass(emission, durations, model, switch)
    end = len(emission)
    state = int(np.argmax(best[end]))
    if not np.isfinite(best[end, state]):
        raise ValueError("no feasible segmentation under the duration model")
    segments = []
    while end > 0:
        start = int(started[end, state])
        segments.append(Segment(start=start, end=end, state=state))
        state, end = int(came_from[start, state]), start
    return segments[::-1]


def step_sums(emission: np.ndarray, stride: int) -> np.ndarray:
    return np.add.reduceat(emission, np.arange(0, len(emission), stride), axis=0)


def step_switch(switch: np.ndarray, stride: int) -> np.ndarray:
    n_frames, n_states = switch.shape
    n_steps = -(-n_frames // stride)
    steps = np.zeros((n_steps + 1, n_states))
    if n_steps < 2:
        return steps
    starts = np.arange(1, n_steps) * stride - stride // 2
    starts[0] = 1
    counts = np.diff(np.append(starts, n_frames))
    peak = np.maximum.reduceat(switch, starts, axis=0)
    sums = np.add.reduceat(np.exp(switch[1:] - np.repeat(peak, counts, axis=0)), starts - 1,
                           axis=0)
    steps[1:n_steps] = peak + np.log(sums / counts[:, None])
    return steps


def refine_boundaries(boundaries: list[int], states: list[int], cumulative: np.ndarray,
                      radius: int, switch: np.ndarray | None = None) -> list[int]:
    n_frames = len(cumulative) - 1
    refined: list[int] = []
    for index, boundary in enumerate(boundaries):
        floor = (refined[-1] if refined else 0) + 1
        ceiling = (boundaries[index + 1] if index + 1 < len(boundaries) else n_frames) - 1
        low = max(boundary - radius, min(floor, boundary))
        high = min(boundary + radius, max(ceiling, boundary))
        left, right = states[index], states[index + 1]
        gain = cumulative[low:high + 1, left] - cumulative[low:high + 1, right]
        if switch is not None and SWITCHES[left, right]:
            gain = gain + switch[low:high + 1, right]
        refined.append(low + int(np.argmax(gain)))
    return refined


def decode_block(emission: np.ndarray, model: SegmentModel, stride: int,
                 switch: np.ndarray | None = None, steps: np.ndarray | None = None
                 ) -> np.ndarray:
    n_frames = len(emission)
    if switch is not None and steps is None:
        steps = step_switch(switch, stride)
    segments = viterbi_segments(step_sums(emission, stride), model, stride, steps)
    states = [segment.state for segment in segments]
    boundaries = [min(segment.end * stride, n_frames) for segment in segments[:-1]]
    cumulative = np.vstack([np.zeros(emission.shape[1]), np.cumsum(emission, axis=0)])
    edges = [0, *refine_boundaries(boundaries, states, cumulative, stride, switch), n_frames]
    codes = np.empty(n_frames, dtype=np.int8)
    for state, start, end in zip(states, edges[:-1], edges[1:]):
        codes[start:end] = state
    return codes
