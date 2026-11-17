import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from possession.paths import external_dir

DFL = "dfl"
SKILLCORNER = "skillcorner"
METRICA = "metrica"
PFF = "pff"
DFL_MATCH_INFO_PREFIX = "DFL_02_01_matchinformation_"
DFL_POSITIONS_PREFIX = "DFL_04_03_positions_raw_observed_"
SKILLCORNER_MATCHES = "matches"
METRICA_REPO = "repo/data"
METRICA_GAME_PREFIX = "Sample_Game_"
PFF_TRACKING_DIR = "Tracking Data"
PFF_METADATA_DIR = "Metadata"
PFF_TRACKING_SUFFIX = ".jsonl.bz2"
PFF_METADATA_SUFFIX = ".json"
DIGEST_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class SourceInfo:
    has_dead_labels: bool
    dead_labels_heuristic: bool = False
    grouped_folds: bool = False


SOURCES = {DFL: SourceInfo(has_dead_labels=True), SKILLCORNER: SourceInfo(has_dead_labels=False),
           METRICA: SourceInfo(has_dead_labels=True, dead_labels_heuristic=True),
           PFF: SourceInfo(has_dead_labels=True, dead_labels_heuristic=True, grouped_folds=True)}


@dataclass(frozen=True)
class MatchFiles:
    source: str
    name: str
    paths: tuple[Path, ...]


def dfl_matches(root: Path) -> list[MatchFiles]:
    matches = []
    for positions in sorted(root.glob(f"{DFL_POSITIONS_PREFIX}*.xml")):
        suffix = positions.name.removeprefix(DFL_POSITIONS_PREFIX)
        info = root / f"{DFL_MATCH_INFO_PREFIX}{suffix}"
        if info.is_file():
            name = suffix.removesuffix(".xml").rsplit("-", 1)[-1]
            matches.append(MatchFiles(DFL, name, (info, positions)))
    return matches


def skillcorner_matches(root: Path) -> list[MatchFiles]:
    matches = []
    for folder in sorted((root / SKILLCORNER_MATCHES).glob("*")):
        meta = folder / f"{folder.name}_match.json"
        tracking = folder / f"{folder.name}_tracking_extrapolated.jsonl"
        if meta.is_file() and tracking.is_file():
            matches.append(MatchFiles(SKILLCORNER, folder.name, (meta, tracking)))
    return matches


def metrica_matches(root: Path) -> list[MatchFiles]:
    matches = []
    for folder in sorted((root / METRICA_REPO).glob(f"{METRICA_GAME_PREFIX}*")):
        stem = folder.name
        name = stem.removeprefix(METRICA_GAME_PREFIX)
        csv_paths = (folder / f"{stem}_RawTrackingData_Home_Team.csv",
                     folder / f"{stem}_RawTrackingData_Away_Team.csv",
                     folder / f"{stem}_RawEventsData.csv")
        epts_paths = (folder / f"{stem}_metadata.xml", folder / f"{stem}_tracking.txt",
                      folder / f"{stem}_events.json")
        chosen = next((paths for paths in (epts_paths, csv_paths)
                       if all(path.is_file() for path in paths)), None)
        if chosen is not None:
            matches.append(MatchFiles(METRICA, f"game{name}", chosen))
    return matches


@lru_cache(maxsize=None)
def content_digest(path: Path, size: int, modified_ns: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DIGEST_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_content(first: Path, second: Path) -> bool:
    first_stat, second_stat = first.stat(), second.stat()
    if first_stat.st_size != second_stat.st_size:
        return False
    return (content_digest(first, first_stat.st_size, first_stat.st_mtime_ns)
            == content_digest(second, second_stat.st_size, second_stat.st_mtime_ns))


def unique_copies(paths: Iterable[Path], suffix: str) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(paths):
        name = path.name.removesuffix(suffix)
        if name not in found:
            found[name] = path
        elif not same_content(found[name], path):
            raise ValueError(f"conflicting copies of PFF file {path.name}: {found[name]} and "
                             f"{path}")
    return found


def pff_copies(root: Path, folder: str, suffix: str) -> dict[str, Path]:
    pattern = f"{folder}/*{suffix}"
    return unique_copies([*root.glob(pattern), *root.glob(f"*/{pattern}")], suffix)


def game_order(name: str) -> tuple[int, str]:
    return len(name), name


def pff_matches(root: Path) -> list[MatchFiles]:
    tracking = pff_copies(root, PFF_TRACKING_DIR, PFF_TRACKING_SUFFIX)
    metadata = pff_copies(root, PFF_METADATA_DIR, PFF_METADATA_SUFFIX)
    return [MatchFiles(PFF, name, (metadata[name], tracking[name]))
            for name in sorted(tracking, key=game_order) if name in metadata]


def limit_per_source(found: list[MatchFiles], limit: int | None) -> list[MatchFiles]:
    if limit is None:
        return found
    kept: list[MatchFiles] = []
    for files in found:
        if sum(1 for item in kept if item.source == files.source) < limit:
            kept.append(files)
    return kept


def discover(source: str | None = None) -> list[MatchFiles]:
    root = external_dir()
    found = []
    if source in (None, DFL):
        found += dfl_matches(root / DFL)
    if source in (None, SKILLCORNER):
        found += skillcorner_matches(root / SKILLCORNER)
    if source in (None, METRICA):
        found += metrica_matches(root / METRICA)
    if source in (None, PFF):
        found += pff_matches(root / PFF)
    return found
