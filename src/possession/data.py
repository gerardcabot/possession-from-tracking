import contextlib
import hashlib
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, fields
from functools import cache
from pathlib import Path

import numpy as np

from possession.paths import cache_dir, data_dir
from possession.vendor import games

FIELD_SEP = ":"
GROUP_SEP = ";"
VALUE_SEP = ","
GROUP_WIDTH = 3
HOME, AWAY, NOBODY = 0, 1, -1
CACHE_FORMAT_VERSION = "tracking-v1"
CACHE_KEY_CHARS = 24
HASH_CHUNK_BYTES = 1 << 20
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
PARSE_CHUNK_LINES = 20_000
DIGEST_SIDECAR_CHARS = 16


@dataclass(frozen=True)
class Tracking:
    ball: np.ndarray
    ball_speed: np.ndarray
    players: np.ndarray
    player_speed: np.ndarray
    slot_team: np.ndarray


TRACKING_FIELDS = tuple(field.name for field in fields(Tracking))


def n_video_frames() -> int:
    return games.source_frames("tactical")


def pitch_size() -> tuple[float, float]:
    return games.GAMES[games.GAME]["pitch"]


def fps() -> int:
    return games.GAMES[games.GAME]["fps"]


def graded_blocks() -> list[tuple[int, int, int]]:
    return games.half_video_blocks(games.GAME)


def kickoff_frames() -> dict[int, int]:
    game = games.GAMES[games.GAME]
    return {1: games.anchor_video_frame(game["kickoff"]),
            2: games.anchor_video_frame(game["kickoff_2nd"])}


def prekickoff_mask(n_frames: int, blocks: list[tuple[int, int, int]],
                    kickoffs: dict[int, int]) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for half, first, _ in blocks:
        if half in kickoffs:
            mask[first:kickoffs[half] + 1] = True
    return mask


def graded_mask(n_frames: int) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in graded_blocks():
        mask[first:last + 1] = True
    return mask


def tactical_paths() -> tuple[Path, Path]:
    meta, raw = games.require(["atd_meta", "atd_raw"], data_dir=str(data_dir()))
    return Path(meta), Path(raw)


def slot_teams(meta_path: Path) -> np.ndarray:
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, found {len(team_ids)}")
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError(f"{meta_path}: no Player entries")
    return np.array([HOME if team_id == team_ids[0] else AWAY for team_id in slots], dtype=np.int8)


def line_pattern(n_slots: int) -> re.Pattern[str]:
    group = r"[^:;,]*,[^:;,]*,[^:;,]*"
    return re.compile(rf"[^:;,]+:(?:{group};){{{n_slots - 1}}}{group}:{group}")


def read_rows(raw_path: Path, n_slots: int) -> np.ndarray:
    width = 1 + GROUP_WIDTH * (n_slots + 1)
    pattern = line_pattern(n_slots)
    chunks: list[np.ndarray] = []
    pending: list[str] = []

    def flush() -> None:
        values = np.fromstring(VALUE_SEP.join(pending), sep=VALUE_SEP)
        chunks.append(values.reshape(len(pending), width))
        pending.clear()

    with open(raw_path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.count(FIELD_SEP) != 2:
                continue
            text = line.strip()
            if not pattern.fullmatch(text):
                raise ValueError(
                    f"{raw_path}:{line_number}: expected frame id plus {n_slots + 1} "
                    f"x,y,speed groups ({width} values per frame line)")
            pending.append(text.replace(FIELD_SEP, VALUE_SEP).replace(GROUP_SEP, VALUE_SEP))
            if len(pending) >= PARSE_CHUNK_LINES:
                flush()
    if pending:
        flush()
    rows = np.concatenate(chunks) if chunks else np.empty((0, width))
    if rows.shape[0] > 1 and not (np.diff(rows[:, 0]) > 0).all():
        raise ValueError(f"{raw_path}: frame ids must be strictly increasing")
    return rows


def frame_indices(rows: np.ndarray, n_frames: int, raw_path: Path) -> np.ndarray:
    frames = rows[:, 0].astype(np.int64) - 1
    if frames.size and (frames.min() < 0 or frames.max() >= n_frames):
        raise ValueError(f"{raw_path}: frame ids outside 1..{n_frames}")
    return frames


def read_tracking(raw_path: Path, slot_team: np.ndarray, n_frames: int) -> Tracking:
    n_slots = len(slot_team)
    rows = read_rows(raw_path, n_slots)
    frames = frame_indices(rows, n_frames, raw_path)
    groups = rows[:, 1:].reshape(len(rows), n_slots + 1, GROUP_WIDTH)

    ball = np.full((n_frames, 2), np.nan)
    ball_speed = np.full(n_frames, np.nan)
    players = np.full((n_frames, n_slots, 2), np.nan)
    player_speed = np.full((n_frames, n_slots), np.nan)
    ball[frames] = groups[:, -1, :2]
    ball_speed[frames] = groups[:, -1, 2]
    players[frames] = groups[:, :-1, :2]
    player_speed[frames] = groups[:, :-1, 2]
    return Tracking(ball=ball, ball_speed=ball_speed, players=players,
                    player_speed=player_speed, slot_team=slot_team)


def parse_tracking(meta_path: Path, raw_path: Path, n_frames: int) -> Tracking:
    return read_tracking(raw_path, slot_teams(meta_path), n_frames)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


@cache
def parser_fingerprint() -> str:
    return file_sha256(Path(__file__))


@contextlib.contextmanager
def atomic_target(path: Path) -> Iterator[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".partial")
    os.close(descriptor)
    temp = Path(name)
    try:
        yield temp
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def digest_sidecar(path: Path, sidecar_dir: Path) -> Path:
    name = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:DIGEST_SIDECAR_CHARS]
    return sidecar_dir / f"digest_{name}.json"


def cached_file_sha256(path: Path, sidecar_dir: Path | None) -> str:
    if sidecar_dir is None:
        return file_sha256(path)
    stat = path.stat()
    signature = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    sidecar = digest_sidecar(path, sidecar_dir)
    if sidecar.is_file():
        try:
            stored = json.loads(sidecar.read_text())
        except ValueError:
            stored = {}
        if stored.get("signature") == signature and isinstance(stored.get("sha256"), str):
            return stored["sha256"]
    digest = file_sha256(path)
    with atomic_target(sidecar) as temp:
        temp.write_text(json.dumps({"signature": signature, "sha256": digest}))
    return digest


def cache_key(paths: list[Path], n_frames: int, sidecar_dir: Path | None = None) -> str:
    digest = hashlib.sha256(
        f"{CACHE_FORMAT_VERSION}:{parser_fingerprint()}:{n_frames}".encode())
    for path in paths:
        digest.update(cached_file_sha256(path, sidecar_dir).encode())
    return digest.hexdigest()[:CACHE_KEY_CHARS]


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    with atomic_target(path) as partial, \
            zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=FIXED_ZIP_TIME)
            with archive.open(info, "w", force_zip64=True) as handle:
                np.lib.format.write_array(handle, np.ascontiguousarray(arrays[name]),
                                          allow_pickle=False)
    return path


def save_tracking(path: Path, tracking: Tracking) -> Path:
    return write_npz(path, {name: getattr(tracking, name) for name in TRACKING_FIELDS})


def load_cached_tracking(path: Path) -> Tracking:
    with np.load(path, allow_pickle=False) as archive:
        return Tracking(**{name: archive[name] for name in TRACKING_FIELDS})


def load_tracking(use_cache: bool = True) -> Tracking:
    meta_path, raw_path = tactical_paths()
    n_frames = n_video_frames()
    if not use_cache:
        return parse_tracking(meta_path, raw_path, n_frames)

    directory = cache_dir()
    key = cache_key([meta_path, raw_path], n_frames, sidecar_dir=directory)
    path = directory / f"tracking_{key}.npz"
    if path.is_file():
        return load_cached_tracking(path)
    tracking = parse_tracking(meta_path, raw_path, n_frames)
    save_tracking(path, tracking)
    return tracking
