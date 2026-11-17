import argparse
import time
from dataclasses import dataclass

import numpy as np

from possession.external.match import (
    ExternalMatch,
    dead_labels_trusted,
    fair_codes,
    fair_truth,
    filled_codes,
    label_codes,
    period_runs,
)
from possession.external.prepare import DEFAULT_SEED, VARIANTS, variant_matches
from possession.external.sources import SOURCES
from possession.external.validate import METHOD_CHOICES, predictor
from possession.geometry import present
from possession.intervals import true_runs
from possession.paths import DEFAULT_MODEL_PATH, cli_path
from possession.vendor import possession_metrics as pm

Runs = list[tuple[int, int]]
TERCILES = (1 / 3, 2 / 3)
QUARTILES = (0.25, 0.5, 0.75)
MERGED_DEAD = "merged across D"
MERGED_OPPOSITE = "merged across opposite spell"
MERGED_OTHER = "merged other"
OVERRUN = "overrun/boundary"
WRONG_TEAM = "wrong team"
PREDICTED_DEAD = "predicted dead"
FRAGMENTED = "fragmented"
MISS_CAUSES = (MERGED_DEAD, MERGED_OPPOSITE, MERGED_OTHER, OVERRUN, WRONG_TEAM,
               PREDICTED_DEAD, FRAGMENTED)
SWAP_KINDS = ("H>A", "A>H")
DEAD_KINDS = ("H>D", "A>D", "D>H", "D>A")
BOUNDARY_KINDS = (*SWAP_KINDS, *DEAD_KINDS)
TIMING_BUCKETS = (f"<={pm.TAUS[0]}f", f"<={pm.TAUS[1]}f", "<=window", "not found")


@dataclass(frozen=True)
class Boundary:
    kind: str
    offset: float
    window: int
    in_gap: bool


@dataclass(frozen=True)
class MatchAnalysis:
    confusion: np.ndarray
    n_frames: int
    true_dead_runs: int
    pred_dead_runs: int
    n_true_spells: int
    n_pred_spells: int
    matched: np.ndarray
    durations_s: np.ndarray
    causes: dict[str, int]
    boundaries: list[Boundary]
    error: np.ndarray
    gap_seconds: np.ndarray


def within(values: np.ndarray, runs: Runs) -> np.ndarray:
    return np.concatenate([values[first:stop] for first, stop in runs])


def frame_confusion(truth: np.ndarray, predicted: np.ndarray, runs: Runs) -> np.ndarray:
    true_codes = within(truth, runs).astype(np.int64)
    scored = true_codes < pm.N_SCORED
    pair = true_codes[scored] * pm.N_LABELS + within(predicted, runs)[scored]
    matrix = np.bincount(pair, minlength=pm.N_LABELS ** 2).reshape(pm.N_LABELS, pm.N_LABELS)
    return matrix[:pm.N_SCORED, :pm.N_SCORED]


def count_runs(codes: np.ndarray, runs: Runs, code: int) -> int:
    return sum(1 for _, _, value in pm.segments(codes, runs) if value == code)


def classify_miss(spell: tuple[int, int, int], predicted: np.ndarray,
                  pred_spells: list[tuple[int, int, int]],
                  true_spells: list[tuple[int, int, int]],
                  true_segments: list[tuple[int, int, int]]) -> str:
    start, stop, team = spell
    length = stop - start
    best, best_overlap = None, 0
    for other in pred_spells:
        overlap = min(stop, other[1]) - max(start, other[0])
        if other[2] == team and overlap > best_overlap:
            best, best_overlap = other, overlap
    if best is not None and best_overlap * 2 >= length:
        absorbed = any(
            other != spell and other[2] == team
            and (min(best[1], other[1]) - max(best[0], other[0])) >= pm.SPELL_IOU_HEADLINE
            * (other[1] - other[0])
            for other in true_spells)
        if not absorbed:
            return OVERRUN
        crossed = {code for first, last, code in true_segments
                   if first < best[1] and best[0] < last and (first, last) != (start, stop)}
        if pm.D in crossed:
            return MERGED_DEAD
        if crossed:
            return MERGED_OPPOSITE
        return MERGED_OTHER
    counts = np.bincount(predicted[start:stop], minlength=pm.N_SCORED)
    majority = int(np.argmax(counts))
    if majority == pm.D:
        return PREDICTED_DEAD
    if majority != team:
        return WRONG_TEAM
    return FRAGMENTED


def spell_analysis(truth: np.ndarray, predicted: np.ndarray, runs: Runs
                   ) -> tuple[np.ndarray, int, list[tuple[int, int, int]], dict[str, int]]:
    true_spells = pm.possession_spells(truth, runs)
    pred_spells = pm.possession_spells(predicted, runs)
    overlaps = pm.spell_overlaps(true_spells, pred_spells)
    matched = pm.matched_gt_flags(overlaps, len(true_spells), pm.SPELL_IOU_HEADLINE)
    segments = pm.segments(truth, runs)
    causes = dict.fromkeys(MISS_CAUSES, 0)
    for spell, hit in zip(true_spells, matched):
        if not hit:
            causes[classify_miss(spell, predicted, pred_spells, true_spells, segments)] += 1
    return matched, len(pred_spells), true_spells, causes


def boundary_kind(origin: int, target: int) -> str:
    return f"{pm.LABELS[origin]}>{pm.LABELS[target]}"


def transitions_by_kind(codes: np.ndarray, runs: Runs) -> dict[str, list[tuple[int, int, int]]]:
    grouped: dict[str, list[tuple[int, int, int]]] = {kind: [] for kind in BOUNDARY_KINDS}
    for transition in pm.transitions(codes, runs):
        kind = boundary_kind(transition[1], transition[2])
        if kind in grouped:
            grouped[kind].append(transition)
    return grouped


def boundary_offsets(truth: np.ndarray, predicted: np.ndarray, runs: Runs, window: int,
                     gap_length: np.ndarray) -> list[Boundary]:
    true_by_kind = transitions_by_kind(truth, runs)
    predicted_by_kind = transitions_by_kind(predicted, runs)
    boundaries = []
    for kind in BOUNDARY_KINDS:
        pairs, _, _ = pm.match_transitions(true_by_kind[kind], predicted_by_kind[kind], window,
                                           typed="none")
        offsets = {true[0]: float(found[0] - true[0]) for true, found in pairs}
        boundaries += [Boundary(kind, offsets.get(frame, float("nan")), window,
                                bool(gap_length[frame] > 0))
                       for frame, _, _ in true_by_kind[kind]]
    return boundaries


def gap_lengths_per_frame(visible: np.ndarray, runs: Runs) -> np.ndarray:
    per_frame = np.zeros(len(visible), dtype=np.int64)
    for first, stop in runs:
        starts, ends = true_runs(~visible[first:stop])
        for start, end in zip(starts.tolist(), ends.tolist()):
            per_frame[first + start:first + end] = end - start
    return per_frame


def analyse_match(match: ExternalMatch, labels: np.ndarray, fair: bool) -> MatchAnalysis:
    runs = period_runs(match)
    truth = filled_codes(match)
    raw = label_codes(labels)
    scored = fair_codes(match, labels) if fair else raw
    reference = fair_truth(match) if fair else truth
    matched, n_pred, true_spells, causes = spell_analysis(reference, scored, runs)
    durations = np.array([stop - start for start, stop, _ in true_spells], dtype=np.int64)
    window = int(np.median(durations)) if durations.size else 0
    gap_length = gap_lengths_per_frame(present(match.ball), runs)
    return MatchAnalysis(
        confusion=frame_confusion(truth, raw, runs),
        n_frames=int((within(truth, runs) < pm.N_SCORED).sum()), true_dead_runs=count_runs(truth, runs, pm.D),
        pred_dead_runs=count_runs(raw, runs, pm.D),
        n_true_spells=len(true_spells), n_pred_spells=n_pred, matched=matched,
        durations_s=durations / match.fps, causes=causes,
        boundaries=boundary_offsets(reference, scored, runs, window, gap_length),
        error=within(reference != scored, runs),
        gap_seconds=within(gap_length, runs) / match.fps)


def percent(part: float, whole: float) -> str:
    return f"{part / whole:6.1%}" if whole else "   n/a"


def error_rate(error: np.ndarray, mask: np.ndarray) -> str:
    return percent(error[mask].sum(), mask.sum()).strip()


def confusion_lines(analyses: list[MatchAnalysis]) -> list[str]:
    total = sum(item.confusion for item in analyses)
    lines = ["frame confusion (true rows, predicted columns, row share)  "
             + " ".join(f"{label:>6}" for label in pm.SCORED)]
    for index, label in enumerate(pm.SCORED):
        row = total[index]
        lines.append(f"  true {label}  n {int(row.sum()):>8}  "
                     + " ".join(percent(value, row.sum()) for value in row))
    return lines


def spell_lines(analyses: list[MatchAnalysis]) -> list[str]:
    true = sum(item.n_true_spells for item in analyses)
    predicted = sum(item.n_pred_spells for item in analyses)
    hits = sum(int(item.matched.sum()) for item in analyses)
    missed = true - hits
    lines = [f"spells at IoU {pm.SPELL_IOU_HEADLINE}: true {true}  predicted {predicted}  "
             f"matched {hits} ({percent(hits, true).strip()})  missed {missed}  "
             f"spurious {predicted - hits} ({percent(predicted - hits, predicted).strip()} "
             f"of predicted)"]
    causes = {cause: sum(item.causes[cause] for item in analyses) for cause in MISS_CAUSES}
    lines.append("  missed true spells by cause: " + "  ".join(
        f"{cause} {percent(count, missed).strip()}" for cause, count in causes.items()))
    return lines


def timing_bucket(boundary: Boundary) -> int:
    magnitude = abs(boundary.offset)
    if np.isnan(boundary.offset):
        return len(TIMING_BUCKETS) - 1
    for index, limit in enumerate((pm.TAUS[0], pm.TAUS[1], boundary.window)):
        if magnitude <= limit:
            return index
    return len(TIMING_BUCKETS) - 1


def boundary_lines(analyses: list[MatchAnalysis]) -> list[str]:
    pooled = [boundary for item in analyses for boundary in item.boundaries]
    lines = ["boundary timing, predicted minus true frame, nearest same-type boundary within "
             "the match median true spell length",
             f"  {'kind':<8}{'n':>6}{'found':>8}{'median':>8}{'p25':>7}{'p75':>7}  "
             + " ".join(f"{name:>10}" for name in TIMING_BUCKETS)]
    for kind in BOUNDARY_KINDS:
        group = [boundary for boundary in pooled if boundary.kind == kind]
        if not group:
            continue
        offsets = np.array([boundary.offset for boundary in group])
        found = offsets[~np.isnan(offsets)]
        counts = np.bincount([timing_bucket(boundary) for boundary in group],
                             minlength=len(TIMING_BUCKETS))
        cuts = np.percentile(found, [25, 50, 75]) if found.size else [float("nan")] * 3
        lines.append(f"  {kind:<8}{len(group):>6}{percent(found.size, len(group)):>8}"
                     f"{cuts[1]:>8.1f}{cuts[0]:>7.1f}{cuts[2]:>7.1f}  "
                     + " ".join(f"{percent(count, len(group)):>10}" for count in counts))
    return lines


def visibility_lines(analyses: list[MatchAnalysis]) -> list[str]:
    error = np.concatenate([item.error for item in analyses])
    gap_seconds = np.concatenate([item.gap_seconds for item in analyses])
    inside = gap_seconds > 0
    lines = [f"ball visibility: error rate visible {error_rate(error, ~inside)}"
             f" ({int((~inside).sum())} frames)  in gaps {error_rate(error, inside)}"
             f" ({int(inside.sum())} frames)"]
    if inside.any():
        edges = np.quantile(gap_seconds[inside], TERCILES)
        tercile = np.digitize(gap_seconds, edges, right=True)
        names = (f"<={edges[0]:.1f}s", f"<={edges[1]:.1f}s", f">{edges[1]:.1f}s")
        lines.append("  by gap length (terciles of frames in gaps, equal frame mass): "
                     + "  ".join(f"{name} {error_rate(error, inside & (tercile == index))}"
                                 f" ({int((inside & (tercile == index)).sum())} frames)"
                                 for index, name in enumerate(names)))
    swaps = [boundary for item in analyses for boundary in item.boundaries
             if boundary.kind in SWAP_KINDS]
    missed = [boundary for boundary in swaps if np.isnan(boundary.offset)]
    lines.append(f"  true H<->A transitions: {len(swaps)}, in gaps "
                 f"{percent(sum(b.in_gap for b in swaps), len(swaps)).strip()}; missed "
                 f"{len(missed)}, of which in gaps "
                 f"{percent(sum(b.in_gap for b in missed), len(missed)).strip()} "
                 f"(share of frames in gaps {percent(inside.sum(), inside.size).strip()})")
    return lines


def duration_lines(analyses: list[MatchAnalysis]) -> list[str]:
    durations = np.concatenate([item.durations_s for item in analyses])
    matched = np.concatenate([item.matched for item in analyses])
    if not durations.size:
        return []
    edges = np.quantile(durations, QUARTILES)
    bucket = np.digitize(durations, edges, right=True)
    names = [f"<={edges[0]:.1f}s", f"<={edges[1]:.1f}s", f"<={edges[2]:.1f}s",
             f">{edges[2]:.1f}s"]
    return ["recall by true spell duration (quartiles): " + "  ".join(
        f"{name} {percent(matched[bucket == index].sum(), (bucket == index).sum()).strip()}"
        f" (n {int((bucket == index).sum())})" for index, name in enumerate(names))]


def dead_lines(analyses: list[MatchAnalysis]) -> list[str]:
    total = sum(item.confusion for item in analyses)
    frames = sum(item.n_frames for item in analyses)
    return [f"D share true {percent(total[pm.D].sum(), frames).strip()} predicted "
            f"{percent(total[:, pm.D].sum(), frames).strip()}  D runs true "
            f"{sum(item.true_dead_runs for item in analyses)} predicted "
            f"{sum(item.pred_dead_runs for item in analyses)}"]


def report(source: str, analyses: list[MatchAnalysis]) -> list[str]:
    return [f"== {source} ({len(analyses)} matches)", *confusion_lines(analyses),
            *spell_lines(analyses), *boundary_lines(analyses), *visibility_lines(analyses),
            *duration_lines(analyses), *dead_lines(analyses)]


def fair_modes(match: ExternalMatch, dead_as_fill: bool) -> list[bool]:
    if not dead_as_fill:
        return [False]
    if SOURCES[match.source].dead_labels_heuristic:
        return [False, True]
    return [False] if dead_labels_trusted(match) else [True]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Error decomposition of a method on degraded external matches (aggregates)")
    parser.add_argument("--method", choices=METHOD_CHOICES, required=True)
    parser.add_argument("--source", choices=sorted(SOURCES))
    parser.add_argument("--variant", choices=VARIANTS, default="degraded")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, help="matches per source, for a quick look")
    parser.add_argument("--dead-as-fill", action=argparse.BooleanOptionalAction, default=True,
                        help="for sources without trusted D labels, treat D as continuation in "
                             "both prediction and reference in every statistic except the frame "
                             "confusion and D counts")
    parser.add_argument("--model", type=cli_path, default=DEFAULT_MODEL_PATH,
                        help="trained model for --method learned")
    args = parser.parse_args()
    start = time.perf_counter()
    predict = predictor(args.method, args.model)
    by_source: dict[str, list[MatchAnalysis]] = {}
    for match in variant_matches(args.source, args.seed, args.variant, args.limit):
        labels = predict(match)
        modes = fair_modes(match, args.dead_as_fill)
        for fair in modes:
            title = match.source if len(modes) == 1 else f"{match.source} {'fair' if fair else 'raw'}"
            by_source.setdefault(title, []).append(analyse_match(match, labels, fair))
        print(f"analysed {match.source}/{match.name}", flush=True)
    print(f"method {args.method}  variant {args.variant}  dead-as-fill {args.dead_as_fill}")
    for source, analyses in by_source.items():
        print("\n".join(report(source, analyses)))
    print(f"runtime {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
