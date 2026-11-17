from pathlib import Path

import numpy as np
import pytest

from possession import data
from possession.data import Tracking

NAN = np.nan


def synthetic_tracking(n_frames: int = 5, n_slots: int = 3) -> Tracking:
    rng = np.random.default_rng(1)
    ball = rng.random((n_frames, 2))
    ball[1] = NAN
    players = rng.random((n_frames, n_slots, 2))
    players[2, 0] = NAN
    return Tracking(ball=ball, ball_speed=rng.random(n_frames), players=players,
                    player_speed=rng.random((n_frames, n_slots)),
                    slot_team=np.array([0, 1, 1], dtype=np.int8)[:n_slots])


def assert_same_tracking(left: Tracking, right: Tracking) -> None:
    for name in data.TRACKING_FIELDS:
        actual, expected = getattr(left, name), getattr(right, name)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


def test_read_tracking_parses_positions_and_speeds(tmp_path: Path) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text(
        "1:0.1,0.2,1.5;NaN,NaN,NaN:0.5,0.6,NaN\n"
        "3:0.3,0.4,NaN;0.7,0.8,2.25:NaN,NaN,9.5\n")
    tracking = data.read_tracking(raw, np.array([0, 1], dtype=np.int8), 4)
    np.testing.assert_array_equal(tracking.ball, [[0.5, 0.6], [NAN, NAN], [NAN, NAN], [NAN, NAN]])
    np.testing.assert_array_equal(tracking.ball_speed, [NAN, NAN, 9.5, NAN])
    np.testing.assert_array_equal(tracking.players[2], [[0.3, 0.4], [0.7, 0.8]])
    np.testing.assert_array_equal(tracking.player_speed[0], [1.5, NAN])
    np.testing.assert_array_equal(tracking.player_speed[2], [NAN, 2.25])
    assert np.isnan(tracking.players[1]).all() and np.isnan(tracking.players[3]).all()


def test_read_tracking_rejects_wrong_group_count(tmp_path: Path) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text("1:0.1,0.2,1.5:0.5,0.6,NaN\n")
    with pytest.raises(ValueError, match="values per frame"):
        data.read_tracking(raw, np.array([0, 1], dtype=np.int8), 2)


@pytest.mark.parametrize("line", [
    "1:0.1,0.2,1.5;0.3,0.4:0.5,0.6,NaN,0.7",
    "1:0.1,0.2,1.5,0.3;0.4,0.5:0.5,0.6,NaN",
    "1:0.1,0.2,1.5:0.3,0.4,0.5;0.5,0.6,NaN",
    "1:0.1,0.2;1.5,0.3,0.4:0.5,0.6,NaN",
])
def test_read_rows_rejects_misaligned_line_with_correct_total(tmp_path: Path, line: str) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text("1:0.1,0.2,1.5;0.3,0.4,0.5:0.5,0.6,NaN\n" + line + "\n")
    with pytest.raises(ValueError, match="raw.txt:2"):
        data.read_rows(raw, 2)


@pytest.mark.parametrize("frame_ids", [(2, 2), (3, 1)])
def test_read_rows_rejects_non_increasing_frame_ids(tmp_path: Path,
                                                    frame_ids: tuple[int, int]) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text("".join(f"{frame}:0.1,0.2,1.5:0.5,0.6,NaN\n" for frame in frame_ids))
    with pytest.raises(ValueError, match="strictly increasing"):
        data.read_rows(raw, 1)


def test_read_rows_spans_multiple_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "PARSE_CHUNK_LINES", 2)
    raw = tmp_path / "raw.txt"
    raw.write_text("".join(f"{frame}:0.1,0.2,1.5:0.5,0.6,NaN\n" for frame in range(1, 6)))
    rows = data.read_rows(raw, 1)
    assert rows[:, 0].tolist() == [1, 2, 3, 4, 5]


def test_read_tracking_rejects_out_of_range_frame(tmp_path: Path) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text("5:0.1,0.2,1.5:0.5,0.6,NaN\n")
    with pytest.raises(ValueError, match="frame ids"):
        data.read_tracking(raw, np.array([0], dtype=np.int8), 2)


def test_cache_roundtrip(tmp_path: Path) -> None:
    tracking = synthetic_tracking()
    path = data.save_tracking(tmp_path / "tracking.npz", tracking)
    assert_same_tracking(data.load_cached_tracking(path), tracking)


def test_cache_bytes_are_deterministic(tmp_path: Path) -> None:
    tracking = synthetic_tracking()
    first = data.save_tracking(tmp_path / "a.npz", tracking).read_bytes()
    second = data.save_tracking(tmp_path / "b.npz", tracking).read_bytes()
    assert first == second


def test_cache_key_tracks_file_content(tmp_path: Path) -> None:
    source = tmp_path / "raw.txt"
    source.write_text("1:0.1,0.2,NaN\n")
    key = data.cache_key([source], 10)
    assert data.cache_key([source], 10) == key
    assert data.cache_key([source], 11) != key
    source.write_text("1:0.1,0.3,NaN\n")
    assert data.cache_key([source], 10) != key


def test_cache_key_tracks_parser_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "raw.txt"
    source.write_text("1:0.1,0.2,NaN\n")
    key = data.cache_key([source], 10)
    monkeypatch.setattr(data, "parser_fingerprint", lambda: "other-parser")
    assert data.cache_key([source], 10) != key


def test_cache_key_fast_path_skips_rehash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "raw.txt"
    source.write_text("1:0.1,0.2,NaN\n")
    sidecars = tmp_path / "sidecars"
    key = data.cache_key([source], 10, sidecar_dir=sidecars)
    assert key == data.cache_key([source], 10)

    def forbidden(path: Path) -> str:
        raise AssertionError("rehashed an unchanged file")

    monkeypatch.setattr(data, "file_sha256", forbidden)
    assert data.cache_key([source], 10, sidecar_dir=sidecars) == key


def test_cache_key_fast_path_detects_changes(tmp_path: Path) -> None:
    source = tmp_path / "raw.txt"
    source.write_text("1:0.1,0.2,NaN\n")
    sidecars = tmp_path / "sidecars"
    key = data.cache_key([source], 10, sidecar_dir=sidecars)
    source.write_text("1:0.1,0.3,NaN,extra\n")
    assert data.cache_key([source], 10, sidecar_dir=sidecars) != key
    assert data.cache_key([source], 10, sidecar_dir=sidecars) == data.cache_key([source], 10)


def test_write_npz_uses_unique_temp_and_leaves_no_partials(tmp_path: Path) -> None:
    stale = tmp_path / "t.npz.partial"
    stale.write_bytes(b"stale")
    tracking = synthetic_tracking()
    path = data.save_tracking(tmp_path / "t.npz", tracking)
    assert_same_tracking(data.load_cached_tracking(path), tracking)
    assert stale.read_bytes() == b"stale"
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["t.npz", "t.npz.partial"]


def test_write_npz_cleans_temp_on_failure(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        data.write_npz(tmp_path / "t.npz", {"bad": np.array([object()], dtype=object)})
    assert list(tmp_path.iterdir()) == []


def test_load_tracking_uses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    meta, raw = tmp_path / "meta.xml", tmp_path / "raw.txt"
    meta.write_text("<main/>")
    raw.write_text("1:0.1,0.2,NaN\n")
    tracking = synthetic_tracking()
    calls: list[int] = []

    def fake_parse(meta_path: Path, raw_path: Path, n_frames: int) -> Tracking:
        calls.append(n_frames)
        return tracking

    monkeypatch.setenv("MTAH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(data, "tactical_paths", lambda: (meta, raw))
    monkeypatch.setattr(data, "parse_tracking", fake_parse)
    assert_same_tracking(data.load_tracking(), tracking)
    assert_same_tracking(data.load_tracking(), tracking)
    assert len(calls) == 1
    assert len(list((tmp_path / "cache").glob("tracking_*.npz"))) == 1
