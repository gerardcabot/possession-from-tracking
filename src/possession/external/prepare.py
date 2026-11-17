import argparse
import hashlib
import importlib.metadata
import importlib.util
from collections.abc import Iterator
from dataclasses import dataclass, fields
from functools import cache
from pathlib import Path

import numpy as np

from possession.cleaning import CleaningParams
from possession.data import (
    cache_key,
    cached_file_sha256,
    fps,
    graded_blocks,
    load_tracking,
    n_video_frames,
    pitch_size,
    tactical_paths,
    write_npz,
)
from possession.external.degrade import DegradationReport, degrade
from possession.external.harmonise import dfl_labels, harmonise, skillcorner_labels
from possession.external.match import (
    ExternalMatch,
    block_share,
    filled_codes,
    load_match,
    save_match,
    spell_lengths,
)
from possession.external.sources import (
    DFL,
    METRICA,
    PFF,
    SKILLCORNER,
    SOURCES,
    MatchFiles,
    discover,
    limit_per_source,
)
from possession.external.stats import OurStats, home_attacks_positive_first, our_stats
from possession.geometry import present
from possession.paths import cache_dir
from possession.vendor import possession_metrics as pm

FORMAT_VERSION = "external-v3"
KEY_CHARS = 24
DEFAULT_SEED = 0
VARIANTS = ("degraded", "harmonised")
LABELLERS = {DFL: dfl_labels, SKILLCORNER: skillcorner_labels, METRICA: dfl_labels,
             PFF: dfl_labels}
BOUNDS_SOURCE = DFL
CACHE_PREFIXES = ("harmonised_", "degraded_", "reference_", "params_")
STATS_MODULES = ("possession.external.stats", "possession.cleaning", "possession.geometry",
                 "possession.intervals", "possession.data")
HARMONISE_MODULES = ("possession.external.harmonise", "possession.external.loaders",
                     "possession.external.metrica_events", "possession.external.pff",
                     "possession.external.sources", "possession.external.match",
                     "possession.geometry", "possession.intervals", "possession.data")
BOUNDS_MODULES = ("possession.external.bounds", "possession.features", *STATS_MODULES,
                  *HARMONISE_MODULES)
DEGRADE_MODULES = ("possession.external.degrade", *STATS_MODULES, *HARMONISE_MODULES)


@dataclass(frozen=True)
class Prepared:
    harmonised: ExternalMatch
    degraded: ExternalMatch
    report: DegradationReport | None


def external_cache() -> Path:
    return cache_dir() / "external"


def digest_of(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:KEY_CHARS]


def code_fingerprint(*module_names: str) -> str:
    hashes = []
    for name in sorted(set(module_names)):
        origin = importlib.util.find_spec(name).origin
        hashes.append(f"{name}={hashlib.sha256(Path(origin).read_bytes()).hexdigest()}")
    return digest_of(*hashes)


def tracking_key() -> str:
    meta_path, raw_path = tactical_paths()
    return cache_key([meta_path, raw_path], n_video_frames(), sidecar_dir=cache_dir())


def match_metadata() -> str:
    return repr((pitch_size(), fps(), graded_blocks()))


@cache
def reference_orientation() -> bool:
    return home_attacks_positive_first(load_tracking(), pitch_size(), graded_blocks())


def harmonised_key(files: MatchFiles, orientation: bool) -> str:
    sources = [cached_file_sha256(path, external_cache()) for path in files.paths]
    return digest_of(FORMAT_VERSION, str(orientation), code_fingerprint(*HARMONISE_MODULES),
                     importlib.metadata.version("kloppy"), *sources)


def harmonised_path(files: MatchFiles, orientation: bool) -> Path:
    return external_cache() / f"harmonised_{files.source}_{files.name}_" \
                              f"{harmonised_key(files, orientation)}.npz"


def bounds_files() -> list[MatchFiles]:
    found = discover(BOUNDS_SOURCE)
    if not found:
        raise FileNotFoundError(f"the cleaning parameters are estimated from {BOUNDS_SOURCE} "
                                f"tracking; no {BOUNDS_SOURCE} match found under the external "
                                "data directory (docs/EXTERNAL.md)")
    return found


def params_path() -> Path:
    orientation = reference_orientation()
    keys = [harmonised_key(files, orientation) for files in bounds_files()]
    key = digest_of(FORMAT_VERSION, code_fingerprint(*BOUNDS_MODULES), tracking_key(),
                    match_metadata(), *keys)
    return external_cache() / f"params_{key}.npz"


def params_arrays(params: CleaningParams) -> dict[str, np.ndarray]:
    return {name: np.array([value]) for name, value in params.to_dict().items()}


def params_from_arrays(arrays: dict[str, np.ndarray]) -> CleaningParams:
    return CleaningParams(**{field.name: field.type(arrays[field.name][0].item())
                             for field in fields(CleaningParams)})


def cached_reference_params() -> CleaningParams | None:
    path = params_path()
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        if not {field.name for field in fields(CleaningParams)} <= set(archive.files):
            return None
        return params_from_arrays({name: archive[name] for name in archive.files})


@cache
def reference_params() -> CleaningParams:
    cached = cached_reference_params()
    if cached is not None:
        return cached
    from possession.external.bounds import estimate_params

    orientation = reference_orientation()
    references = [harmonised_match(files, orientation, force=False) for files in bounds_files()]
    params = estimate_params(references, load_tracking(), pitch_size(), fps(), graded_blocks())
    write_npz(params_path(), params_arrays(params))
    return params


def reference_path(params: CleaningParams) -> Path:
    key = digest_of(FORMAT_VERSION, code_fingerprint(*STATS_MODULES), tracking_key(),
                    match_metadata(), repr(sorted(params.to_dict().items())))
    return external_cache() / f"reference_{key}.npz"


def reference_stats() -> OurStats:
    params = reference_params()
    path = reference_path(params)
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            return OurStats.from_arrays({name: archive[name] for name in archive.files})
    stats = our_stats(load_tracking(), pitch_size(), fps(), graded_blocks(), params)
    write_npz(path, stats.to_arrays())
    return stats


def current_reference_files() -> set[str]:
    return {params_path().name, reference_path(reference_params()).name}


def degraded_path(files: MatchFiles, stats: OurStats, seed: int) -> Path:
    key = digest_of(harmonised_key(files, stats.home_attacks_positive_first), stats.digest(),
                    str(seed), code_fingerprint(*DEGRADE_MODULES))
    return external_cache() / f"degraded_{files.source}_{files.name}_{key}.npz"


def build_harmonised(files: MatchFiles, orientation: bool) -> ExternalMatch:
    from possession.external.loaders import LOADERS

    series = LOADERS[files.source](files)
    return harmonise(series, files.name, files.source, LABELLERS[files.source], fps(),
                     orientation)


def harmonised_match(files: MatchFiles, orientation: bool, force: bool) -> ExternalMatch:
    path = harmonised_path(files, orientation)
    if path.is_file() and not force:
        return load_match(path)
    harmonised = build_harmonised(files, orientation)
    save_match(path, harmonised)
    return harmonised


def build_degraded(files: MatchFiles, stats: OurStats, seed: int,
                   harmonised: ExternalMatch) -> tuple[ExternalMatch, DegradationReport]:
    degraded, report = degrade(harmonised, stats, seed)
    save_match(degraded_path(files, stats, seed), degraded)
    return degraded, report


def prepare_match(files: MatchFiles, stats: OurStats, seed: int, force: bool) -> Prepared:
    harmonised = harmonised_match(files, stats.home_attacks_positive_first, force)
    noisy_path = degraded_path(files, stats, seed)
    if noisy_path.is_file() and not force:
        return Prepared(harmonised, load_match(noisy_path), None)
    degraded, report = build_degraded(files, stats, seed, harmonised)
    return Prepared(harmonised, degraded, report)


def variant_match(files: MatchFiles, stats: OurStats, seed: int, variant: str,
                  force: bool) -> ExternalMatch:
    orientation = stats.home_attacks_positive_first
    if variant == "harmonised":
        return harmonised_match(files, orientation, force)
    path = degraded_path(files, stats, seed)
    if path.is_file() and not force:
        return load_match(path)
    return build_degraded(files, stats, seed, harmonised_match(files, orientation, force))[0]


def variant_matches(source: str | None, seed: int, variant: str,
                    limit: int | None = None) -> Iterator[ExternalMatch]:
    stats = reference_stats()
    for files in limit_per_source(discover(source), limit):
        yield variant_match(files, stats, seed, variant, force=False)


def prune_stale(stats: OurStats, seed: int) -> list[Path]:
    current = current_reference_files()
    for files in discover(None):
        current.add(harmonised_path(files, stats.home_attacks_positive_first).name)
        current.add(degraded_path(files, stats, seed).name)
    stale = [path for path in sorted(external_cache().glob("*.npz"))
             if path.name.startswith(CACHE_PREFIXES) and path.name not in current]
    for path in stale:
        path.unlink()
    return stale


def ball_coverage(match: ExternalMatch) -> float:
    return block_share(present(match.ball), match)


def label_shares(match: ExternalMatch) -> dict[str, float]:
    codes = pm.label_codes(match.labels)
    return {label: block_share(codes == index, match) for index, label in enumerate(pm.LABELS)}


def summary_line(prepared: Prepared) -> str:
    match = prepared.degraded
    shares = " ".join(f"{label} {value:5.1%}" for label, value in label_shares(match).items())
    lengths = spell_lengths(filled_codes(match), match)
    median = float(np.median(lengths)) / match.fps if lengths.size else 0.0
    return (f"{match.source}/{match.name}  frames {match.n_frames}  fps {match.fps:g}  "
            f"ball {ball_coverage(prepared.harmonised):6.1%} -> {ball_coverage(match):6.1%}  "
            f"{shares}  spells {len(lengths)}  median {median:5.1f}s")


def report_line(prepared: Prepared) -> str | None:
    report = prepared.report
    if report is None:
        return None
    return (f"    ball missing target {report.ball_missing_target:.3f} "
            f"achieved {report.ball_missing_after:.3f}  "
            f"median gap target {report.gap_median_target_s:.1f}s "
            f"achieved {report.gap_median_after_s:.1f}s  "
            f"objects/frame target {report.objects_target:.2f} "
            f"before {report.objects_before:.2f} after {report.objects_after:.2f}  "
            f"dup share target {report.duplicate_target:.4f} "
            f"added {report.duplicate_after - report.duplicate_before:.4f}  "
            f"jitter added player {report.player_sigma_added_m:.4f}m "
            f"ball {report.ball_sigma_added_m:.4f}m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harmonise and degrade external tracking matches into the cache")
    parser.add_argument("--source", choices=sorted(SOURCES))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true", help="ignore existing caches")
    parser.add_argument("--prune", action="store_true",
                        help="delete cache files not matching the current keys for --seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = reference_stats()
    print(f"cleaning params (estimated): {reference_params().to_dict()}")
    print(f"reference: ball observed {stats.ball_observed_share:.1%}  "
          f"objects/frame {stats.objects_per_frame:.2f}  "
          f"duplicate share {stats.duplicate_share:.4f}  slots {stats.slot_counts.tolist()}")
    for files in discover(args.source):
        prepared = prepare_match(files, stats, args.seed, args.force)
        print(summary_line(prepared))
        line = report_line(prepared)
        if line:
            print(line)
    if args.prune:
        removed = prune_stale(stats, args.seed)
        print(f"pruned {len(removed)} stale cache files")


if __name__ == "__main__":
    main()
