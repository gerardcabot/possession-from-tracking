import zlib
from dataclasses import dataclass

import numpy as np

from possession.cleaning import duplicate_distances
from possession.data import AWAY, HOME
from possession.external.match import ExternalMatch, period_ids
from possession.external.stats import (
    Blocks,
    OurStats,
    block_mask,
    block_slices,
    fill_ball,
    nearest_and_edge,
    run_lengths,
    standardise,
    white_noise_sigma,
)
from possession.geometry import present, to_metres
from possession.intervals import value_runs


@dataclass(frozen=True)
class DegradationReport:
    ball_missing_target: float
    ball_missing_before: float
    ball_missing_after: float
    gap_median_target_s: float
    gap_median_after_s: float
    objects_target: float
    objects_before: float
    objects_after: float
    duplicate_target: float
    duplicate_before: float
    duplicate_after: float
    player_sigma_added_m: float
    ball_sigma_added_m: float
    segments: int


def seeded_rng(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


def added_sigma(target: float, observed: float) -> float:
    return float(np.sqrt(max(target ** 2 - observed ** 2, 0.0)))


def jitter(values: np.ndarray, sigma_m: float, pitch: tuple[float, float],
           rng: np.random.Generator) -> np.ndarray:
    scale = sigma_m / np.asarray(pitch, dtype=float)
    return values + rng.normal(size=values.shape) * scale


def draw_until(total: int, lengths: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    drawn = np.array([], dtype=np.int64)
    while drawn.sum() < total:
        count = max(1, int(np.ceil((total - drawn.sum()) / lengths.mean())))
        drawn = np.concatenate([drawn, rng.choice(lengths, size=count).astype(np.int64)])
    cumulative = np.cumsum(drawn)
    kept = drawn[:int(np.searchsorted(cumulative, total, side="left")) + 1].copy()
    kept[-1] -= int(kept.sum() - total)
    return kept


def cut_run(length: int, lengths: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(draw_until(length, lengths, rng))])


def presence_segments(players: np.ndarray, blocks: Blocks, lengths: np.ndarray,
                      rng: np.random.Generator) -> list[tuple[int, int, int]]:
    seen = present(players)
    segments = []
    for window in block_slices(blocks):
        for player in range(players.shape[1]):
            starts, ends, values = value_runs(seen[window, player])
            for start, end in zip(starts[values], ends[values]):
                edges = cut_run(int(end - start), lengths, rng)
                base = window.start + int(start)
                segments += [(base + int(a), base + int(b), player)
                             for a, b in zip(edges[:-1], edges[1:])]
    return sorted(segments)


def assign_slots(players: np.ndarray, slot_team: np.ndarray, segments: list[tuple[int, int, int]],
                 keep: np.ndarray, pool_team: np.ndarray) -> np.ndarray:
    squad = np.full((len(players), len(pool_team), 2), np.nan)
    free_at = np.full(len(pool_team), -1, dtype=np.int64)
    team_slots = {team: np.flatnonzero(pool_team == team) for team in (HOME, AWAY)}
    for (start, end, player), kept in zip(segments, keep):
        if not kept:
            continue
        slots = team_slots[int(slot_team[player])]
        free = slots[free_at[slots] <= start]
        if free.size == 0:
            continue
        slot = free[np.argmin(free_at[free])]
        squad[start:end, slot] = players[start:end, player]
        free_at[slot] = end
    return squad


def duplicate_target_samples(share: float, objects: int) -> int:
    return int(round(share / (1.0 - share) * objects)) if share < 1.0 else 0


def add_duplicates(squad: np.ndarray, pool_team: np.ndarray, share: float, lengths: np.ndarray,
                   offsets: np.ndarray, blocks: Blocks, pitch: tuple[float, float],
                   rng: np.random.Generator) -> np.ndarray:
    seen = present(squad)
    target = duplicate_target_samples(share, int(seen.sum()))
    if target == 0 or lengths.size == 0 or offsets.size == 0:
        return squad
    result = squad.copy()
    occupied = seen.copy()
    starts_pool = np.flatnonzero(block_mask(len(squad), blocks))
    block_end = np.zeros(len(squad), dtype=np.int64)
    for window in block_slices(blocks):
        block_end[window] = window.stop
    placed = 0
    attempts = 0
    budget = int(np.ceil(target / lengths.mean())) * len(pool_team)
    scale = 1.0 / np.asarray(pitch, dtype=float)
    while placed < target and attempts < budget:
        attempts += 1
        start = int(rng.choice(starts_pool))
        end = min(start + min(int(rng.choice(lengths)), target - placed), int(block_end[start]))
        team = int(rng.integers(2))
        slots = np.flatnonzero(pool_team == team)
        sources = slots[seen[start, slots]]
        idle = slots[~occupied[start:end, slots].any(axis=0)]
        if sources.size == 0 or idle.size == 0:
            continue
        source, target_slot = int(rng.choice(sources)), int(rng.choice(idle))
        angle = rng.uniform(0.0, 2 * np.pi)
        radius = float(rng.choice(offsets))
        offset = np.array([np.cos(angle), np.sin(angle)]) * radius * scale
        result[start:end, target_slot] = squad[start:end, source] + offset
        occupied[start:end, target_slot] = True
        placed += int(present(result[start:end, target_slot]).sum())
    return result


def missing_logit(match: ExternalMatch, stats: OurStats) -> np.ndarray:
    ball = to_metres(match.ball, match.pitch)
    players = to_metres(match.players, match.pitch)
    covariates = nearest_and_edge(fill_ball(ball), players, match.pitch)
    return standardise(covariates, stats.propensity_mean,
                       stats.propensity_scale) @ stats.propensity_coef


def free_starts(occupied: np.ndarray, length: int) -> np.ndarray:
    cumulative = np.concatenate([[0], np.cumsum(occupied)])
    starts = np.arange(len(occupied) - length + 1)
    low = np.maximum(starts - 1, 0)
    high = np.minimum(starts + length + 1, len(occupied))
    return (cumulative[high] - cumulative[low]) == 0


def window_scores(cumulative: np.ndarray, ids: np.ndarray, length: int) -> np.ndarray:
    starts = np.arange(len(ids) - length + 1)
    same_period = (ids[starts] == ids[starts + length - 1]) & (ids[starts] >= 0)
    return (cumulative[starts + length] - cumulative[starts]) * same_period


def window_is_free(missing: np.ndarray, start: int, length: int) -> bool:
    return not missing[max(start - 1, 0):min(start + length + 1, len(missing))].any()


def pick_start(base: np.ndarray, cdf: np.ndarray, missing: np.ndarray, length: int,
               rng: np.random.Generator) -> int | None:
    for _ in range(int(np.ceil(len(missing) / length))):
        candidate = int(np.searchsorted(cdf, rng.random() * cdf[-1], side="right"))
        if window_is_free(missing, candidate, length):
            return candidate
    score = base * free_starts(missing, length)
    if score.sum() <= 0:
        return None
    return int(rng.choice(len(score), p=score / score.sum()))


def place_gaps(observed_missing: np.ndarray, logit: np.ndarray, lengths: np.ndarray,
               ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    missing = observed_missing.copy()
    weight = np.exp(logit - logit.max())
    cumulative = np.concatenate([[0.0], np.cumsum(weight)])
    distinct, counts = np.unique(lengths, return_counts=True)
    for length, count in zip(distinct[::-1].tolist(), counts[::-1].tolist()):
        if length > len(missing):
            continue
        base = window_scores(cumulative, ids, length)
        cdf = np.cumsum(base)
        if cdf[-1] <= 0:
            continue
        for _ in range(count):
            start = pick_start(base, cdf, missing, length, rng)
            if start is None:
                break
            missing[start:start + length] = True
    return missing


def ball_gap_mask(match: ExternalMatch, stats: OurStats, rng: np.random.Generator) -> np.ndarray:
    already = ~present(match.ball)
    needed = int(round((1.0 - stats.ball_observed_share) * match.n_frames)) - int(already.sum())
    if needed <= 0:
        return already
    lengths = draw_until(needed, stats.gap_lengths.astype(np.int64), rng)
    return place_gaps(already, missing_logit(match, stats), lengths, period_ids(match), rng)


def mean_objects(squad: np.ndarray, blocks: Blocks) -> float:
    return float(present(squad)[block_mask(len(squad), blocks)].sum(axis=1).mean())


def duplicate_share(squad: np.ndarray, pool_team: np.ndarray, pitch: tuple[float, float],
                    blocks: Blocks, distance: float) -> float:
    metres = to_metres(squad, pitch)
    flagged = np.isfinite(duplicate_distances(metres, pool_team, distance))
    graded = block_mask(len(squad), blocks)
    return float(flagged[graded].sum() / max(int(present(metres)[graded].sum()), 1))


def median_gap_s(missing: np.ndarray, blocks: Blocks, fps: float) -> float:
    lengths = run_lengths(~missing, blocks, False)
    return float(np.median(lengths)) / fps if lengths.size else 0.0


def degrade(match: ExternalMatch, stats: OurStats,
            seed: int) -> tuple[ExternalMatch, DegradationReport]:
    rng = seeded_rng(seed, f"{match.source}:{match.name}")
    blocks = match.blocks
    pitch = match.pitch
    true_players = to_metres(match.players, pitch)
    player_sigma = added_sigma(stats.player_sigma_m, white_noise_sigma(true_players, blocks))
    ball_sigma = added_sigma(stats.ball_sigma_m,
                             white_noise_sigma(to_metres(match.ball, pitch), blocks))
    objects_before = mean_objects(match.players, blocks)
    players = jitter(match.players, player_sigma, pitch, rng)
    ball = jitter(match.ball, ball_sigma, pitch, rng)
    pool_team = np.repeat(np.array([HOME, AWAY], dtype=np.int8), stats.slot_counts)
    keep_probability = min(1.0, stats.objects_per_frame / objects_before)
    segments = presence_segments(players, blocks, stats.segment_lengths.astype(np.int64), rng)
    keep = rng.random(len(segments)) < keep_probability
    thinned = assign_slots(players, match.slot_team, segments, keep, pool_team)
    squad = add_duplicates(thinned, pool_team, stats.duplicate_share,
                           stats.duplicate_lengths.astype(np.int64), stats.duplicate_offsets_m,
                           blocks, pitch, rng)
    missing = ball_gap_mask(match, stats, rng)
    degraded_ball = np.where(missing[:, None], np.nan, ball)
    degraded = ExternalMatch(
        name=match.name, source=match.source, fps=match.fps, pitch=pitch, ball=degraded_ball,
        players=squad, slot_team=pool_team, labels=match.labels, periods=match.periods)
    graded = block_mask(match.n_frames, blocks)
    report = DegradationReport(
        ball_missing_target=1.0 - stats.ball_observed_share,
        ball_missing_before=float((~present(match.ball))[graded].mean()),
        ball_missing_after=float(missing[graded].mean()),
        gap_median_target_s=float(np.median(stats.gap_lengths)) / stats.fps,
        gap_median_after_s=median_gap_s(missing, blocks, match.fps),
        objects_target=stats.objects_per_frame,
        objects_before=objects_before,
        objects_after=mean_objects(thinned, blocks),
        duplicate_target=stats.duplicate_share,
        duplicate_before=duplicate_share(match.players, match.slot_team, pitch, blocks,
                                         stats.duplicate_distance_m),
        duplicate_after=duplicate_share(squad, pool_team, pitch, blocks,
                                        stats.duplicate_distance_m),
        player_sigma_added_m=player_sigma,
        ball_sigma_added_m=ball_sigma,
        segments=int(keep.sum()),
    )
    return degraded, report
