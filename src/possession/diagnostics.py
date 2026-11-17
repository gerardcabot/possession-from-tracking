import argparse
from pathlib import Path

import numpy as np

from possession.cleaning import CleanTracking, clean
from possession.cleaning_params import cleaning_params
from possession.data import (
    fps,
    graded_blocks,
    graded_mask,
    load_tracking,
    n_video_frames,
    pitch_size,
)
from possession.intervals import value_runs
from possession.paths import cli_path
from possession.tagging import TagEvent, load_tags, phase_team_evidence, set_pieces
from possession.vendor import possession_metrics as pm

GAP_BUCKETS = ((1, 1), (2, 8), (9, 25), (26, 75), (76, 250), (251, None))
MAX_PLAYERS_ON_PITCH = 22
OBJECT_PERCENTILES = (5, 50, 95)
SPELL_BUCKETS_S = ((0, 1), (1, 3), (3, 10), (10, 30), (30, 60), (60, None))


def block_slices() -> list[tuple[int, slice]]:
    return [(half, slice(first, last + 1)) for half, first, last in graded_blocks()]


def block_mask(n_frames: int, block: slice) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    mask[block] = True
    return mask


def missing_run_lengths(observed: np.ndarray) -> np.ndarray:
    lengths = []
    for _, block in block_slices():
        starts, ends, run_value = value_runs(observed[block])
        lengths.append((ends - starts)[~run_value])
    return np.concatenate(lengths)


def share(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if mask.size else float("nan")


def print_header(title: str) -> None:
    print(f"\n{title}")


def print_coverage(cleaned: CleanTracking, graded: np.ndarray) -> None:
    usable = ~np.isnan(cleaned.ball[:, 0])
    print_header("ball coverage (raw ATD | after cleaning)")
    scopes = [("all frames", np.ones_like(graded)), ("graded", graded)]
    scopes += [(f"half {half}", block_mask(len(graded), block)) for half, block in block_slices()]
    for name, scope in scopes:
        raw, after = share(cleaned.ball_observed[scope]), share(usable[scope])
        print(f"  {name:<12}{raw:7.1%} | {after:7.1%}")


def bucket_label(low: int, high: int | None) -> str:
    if high is None:
        return f"{low}+"
    return str(low) if low == high else f"{low}-{high}"


def print_gap_histogram(observed: np.ndarray) -> None:
    lengths = missing_run_lengths(observed)
    print_header("raw ball gaps inside graded blocks (frames)")
    print(f"  {'length':<10}{'gaps':>8}{'frames':>10}")
    for low, high in GAP_BUCKETS:
        selected = (lengths >= low) & (lengths <= (high if high is not None else np.inf))
        frames = int(lengths[selected].sum())
        print(f"  {bucket_label(low, high):<10}{int(selected.sum()):>8}{frames:>10}")
    print(f"  {'total':<10}{len(lengths):>8}{int(lengths.sum()):>10}")


def print_cleaning_counts(cleaned: CleanTracking, graded: np.ndarray) -> None:
    counts = {
        "ball outliers removed": cleaned.ball_outlier[graded].sum(),
        "ball frames interpolated": cleaned.ball_interpolated[graded].sum(),
        "player jump samples removed": cleaned.player_jump[graded].sum(),
        "player duplicate samples merged": cleaned.player_duplicate[graded].sum(),
    }
    print_header("cleaning counts on graded frames")
    for name, count in counts.items():
        print(f"  {name:<30}{int(count):>9}")


def object_stats(counts: np.ndarray) -> str:
    low, mid, high = np.percentile(counts, OBJECT_PERCENTILES)
    return (f"mean {counts.mean():5.2f}  p5 {low:4.0f}  median {mid:4.0f}  p95 {high:4.0f}  "
            f"max {counts.max():3d}  "
            f">{MAX_PLAYERS_ON_PITCH} {share(counts > MAX_PLAYERS_ON_PITCH):6.2%}")


def print_object_counts(cleaned: CleanTracking, graded: np.ndarray) -> None:
    raw = cleaned.player_observed[graded].sum(axis=1)
    kept = (~np.isnan(cleaned.players[graded][..., 0])).sum(axis=1)
    print_header("player objects per graded frame")
    print(f"  raw      {object_stats(raw)}")
    print(f"  cleaned  {object_stats(kept)}")


def spell_lengths(codes: np.ndarray) -> np.ndarray:
    lengths = []
    for _, block in block_slices():
        starts, ends, run_value = value_runs(codes[block])
        lengths.append((ends - starts)[np.isin(run_value, pm.POSSESSION_LABELS)])
    return np.concatenate(lengths)


def dead_runs(codes: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    for _, block in block_slices():
        starts, ends, values = value_runs(codes[block])
        dead = values == pm.D
        runs += [(block.start + int(start), block.start + int(end))
                 for start, end in zip(starts[dead], ends[dead])]
    return runs


def restart_frames(runs: list[tuple[int, int]]) -> np.ndarray:
    restarts = [end for _, end in runs]
    blocks = {block.stop for _, block in block_slices()}
    return np.array([frame for frame in restarts if frame not in blocks], dtype=np.int64)


def spell_histogram(lengths: np.ndarray, rate: float) -> str:
    seconds = lengths / rate
    parts = []
    for low, high in SPELL_BUCKETS_S:
        selected = (seconds >= low) & (seconds < (high if high is not None else np.inf))
        parts.append(f"{bucket_label(low, high)}s {int(selected.sum())}")
    return "  ".join(parts)


def restarts_in(restarts: np.ndarray, piece: TagEvent) -> np.ndarray:
    return restarts[(restarts >= piece.start) & (restarts < piece.end)]


def restart_hits(restarts: np.ndarray, pieces: list[TagEvent]) -> list[int]:
    hits = []
    for piece in pieces:
        inside = restarts_in(restarts, piece)
        if inside.size:
            hits.append(int(inside[0]))
    return hits


def restart_team_agreement(codes: np.ndarray, restarts: np.ndarray,
                           pieces: list[TagEvent]) -> str:
    agree = total = 0
    for piece in pieces:
        if piece.team not in (0, 1):
            continue
        inside = restarts_in(restarts, piece)
        if inside.size and codes[inside[0]] in pm.POSSESSION_LABELS:
            total += 1
            agree += int(codes[inside[0]] == piece.team)
    return f"{agree}/{total}"


def phase_agreement(codes: np.ndarray, tags: list[TagEvent], graded: np.ndarray) -> float:
    votes = phase_team_evidence(tags, len(codes))
    decided = graded & (votes[:, 0] != votes[:, 1]) & np.isin(codes, pm.POSSESSION_LABELS)
    voted = np.where(votes[:, 0] > votes[:, 1], pm.H, pm.A)
    return share(codes[decided] == voted[decided])


def label_summary(codes: np.ndarray, graded: np.ndarray, tags: list[TagEvent]) -> dict:
    graded_codes = codes[graded]
    lengths = spell_lengths(codes)
    rate = fps()
    pieces = set_pieces(tags)
    runs = dead_runs(codes)
    restarts = restart_frames(runs)
    return {
        "shares": {label: share(graded_codes == index) for index, label in enumerate(pm.LABELS)},
        "spells": len(lengths),
        "mean_spell_s": lengths.mean() / rate if len(lengths) else 0.0,
        "median_spell_s": float(np.median(lengths)) / rate if len(lengths) else 0.0,
        "histogram": spell_histogram(lengths, rate),
        "stoppages": len(runs),
        "restart_hits": len(restart_hits(restarts, pieces)),
        "set_pieces": len(pieces),
        "restart_team": restart_team_agreement(codes, restarts, pieces),
        "phase_agreement": phase_agreement(codes, tags, graded),
    }


def print_summary(summary: dict) -> None:
    shares = "  ".join(f"{label} {value:6.1%}" for label, value in summary["shares"].items())
    print(f"  graded label share   {shares}")
    print(f"  possession spells    {summary['spells']}")
    print(f"  mean spell duration  {summary['mean_spell_s']:.2f} s")
    print(f"  median spell         {summary['median_spell_s']:.2f} s")
    print(f"  spell histogram      {summary['histogram']}")
    print(f"  stoppages (D runs)   {summary['stoppages']}")
    print(f"  set-piece windows holding a restart  "
          f"{summary['restart_hits']}/{summary['set_pieces']}")
    print(f"  restart team = set-piece team  {summary['restart_team']}")
    print(f"  phase-team agreement {summary['phase_agreement']:6.1%}")


def print_submission(path: Path, graded: np.ndarray, tags: list[TagEvent]) -> None:
    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, n_video_frames())
    print_header(f"submission {path.name}")
    if reason:
        print(f"  invalid: {reason}")
        return
    print_summary(label_summary(pm.label_codes(table), graded, tags))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate tracking and submission diagnostics")
    parser.add_argument("--submission", type=Path, action="append", default=[])
    parser.add_argument("--no-cache", action="store_true")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--legacy", action="store_true",
                        help="clean with the hand-set Phase 1 values (rules method only)")
    source.add_argument("--model", type=cli_path,
                        help="clean with the parameters stored in a trained learned model "
                             "(default: the stored model, else the cached estimated parameters)")
    args = parser.parse_args()

    params = cleaning_params(args.legacy, args.model)
    print(f"cleaning params {params.to_dict()}")
    cleaned = clean(load_tracking(use_cache=not args.no_cache), pitch_size(), fps(), params)
    graded = graded_mask(len(cleaned.ball))
    print_coverage(cleaned, graded)
    print_gap_histogram(cleaned.ball_observed)
    print_cleaning_counts(cleaned, graded)
    print_object_counts(cleaned, graded)
    tags = load_tags() if args.submission else []
    for path in args.submission:
        print_submission(path, graded, tags)


if __name__ == "__main__":
    main()
