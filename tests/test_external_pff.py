import bz2
import json
from pathlib import Path

import numpy as np
import pytest

from possession.data import AWAY, HOME
from possession.external import pff, sources
from possession.external.harmonise import OWNER_NONE, dfl_labels, harmonise
from possession.external.sources import PFF, MatchFiles

PITCH = (105.0, 68.0)
PFF_FPS = 29.97
TARGET_FPS = 25.0


def player(jersey: str, x: float, y: float) -> dict[str, object]:
    return {"jerseyNum": jersey, "confidence": "HIGH", "visibility": "VISIBLE", "x": x, "y": y}


def event(kind: str, start: int, home_ball: bool | None) -> dict[str, object]:
    return {"game_event_type": kind, "start_frame": start, "end_frame": start,
            "home_ball": home_ball}


def record(frame: int, period: int | None = 1, ball: tuple[float, float] | None = (0.0, 0.0),
           game_event: dict[str, object] | None = None, event_id: int | None = None,
           home: list[dict[str, object]] | None = None,
           away: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "frameNum": frame, "period": period,
        "balls": [] if ball is None else [{"visibility": "VISIBLE", "x": ball[0], "y": ball[1],
                                           "z": 0.1}],
        "homePlayersSmoothed": [player("7", -10.0, 5.0)] if home is None else home,
        "awayPlayersSmoothed": [player("9", 10.0, -5.0)] if away is None else away,
        "game_event_id": event_id, "game_event": game_event,
    }


def metadata_record(game_id: str = "123") -> list[dict[str, object]]:
    return [{"id": game_id, "fps": PFF_FPS,
             "stadium": {"pitches": [{"length": PITCH[0], "width": PITCH[1]}]},
             "homeTeam": {"id": "1", "name": "Home"}, "awayTeam": {"id": "2", "name": "Away"},
             "week": 1, "date": "2022-11-20T16:00:00"}]


def write_tracking(path: Path, records: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with bz2.open(path, "wt") as handle:
        handle.write("".join(json.dumps(item) + "\n" for item in records))
    return path


def write_metadata(path: Path, game_id: str = "123") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata_record(game_id)))
    return path


def test_consecutive_duplicate_frames_are_merged_and_gaps_left_missing() -> None:
    tracking = pff.parse_tracking([record(10), record(10, ball=(5.0, 5.0)), record(11),
                                   record(13)])
    assert tracking.frames.tolist() == [10, 11, 12, 13]
    assert tracking.ball[0].tolist() == [0.0, 0.0]
    assert np.isnan(tracking.ball[2]).all()
    assert np.isnan(tracking.players[2]).all()
    assert tracking.periods == [(1, 0, 3)]


def test_frames_out_of_order_are_rejected() -> None:
    with pytest.raises(ValueError, match="out of order"):
        pff.parse_tracking([record(10), record(12), record(11)])


def test_periods_get_their_own_blocks_and_non_play_frames_are_skipped() -> None:
    tracking = pff.parse_tracking([record(1), record(2), record(3, period=None), record(50, 2),
                                   record(51, 2), record(52, 5)])
    assert tracking.periods == [(1, 0, 1), (2, 2, 3)]
    assert tracking.frames.tolist() == [1, 2, 50, 51]


def test_players_get_one_slot_per_team_and_jersey() -> None:
    home = [player("7", -10.0, 5.0), player("7", -11.0, 5.0), player("4", -20.0, 0.0)]
    away = [player("7", 10.0, -5.0)]
    tracking = pff.parse_tracking([record(1, home=home, away=away),
                                   record(2, home=home[2:], away=away)])
    assert tracking.slot_team.tolist() == [HOME, HOME, AWAY]
    assert tracking.repeated_jerseys == 1
    assert tracking.players[0, 0].tolist() == [-10.0, 5.0]
    assert np.isnan(tracking.players[1, 0]).all()
    assert tracking.players[1, 1].tolist() == [-20.0, 0.0]


def test_ball_is_missing_when_absent_or_ambiguous() -> None:
    two = record(2)
    two["balls"] = two["balls"] * 2
    tracking = pff.parse_tracking([record(1, ball=None), two, record(3, ball=(1.0, 2.0))])
    assert np.isnan(tracking.ball[:2]).all()
    assert tracking.ambiguous_balls == 1
    assert tracking.ball[2].tolist() == [1.0, 2.0]


def test_coordinates_follow_the_kloppy_normalised_frame() -> None:
    metres = np.array([[0.0, 0.0], [52.5, 34.0], [-52.5, -34.0]])
    assert pff.normalised(metres, PITCH).tolist() == [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]]


def test_out_and_end_start_dead_ball_until_the_next_on_ball_event() -> None:
    events = [pff.GameEvent("FIRSTKICKOFF", 10, True), pff.GameEvent("OTB", 12, False),
              pff.GameEvent("SUB", 13, None), pff.GameEvent("OUT", 14, False),
              pff.GameEvent("G", 15, None), pff.GameEvent("OTB", 16, True),
              pff.GameEvent("END", 18, True)]
    alive, owner = pff.event_state(events, np.arange(9, 20))
    assert alive.tolist() == [False, True, True, True, True, False, False, True, True, False,
                              False]
    assert owner[:5].tolist() == [OWNER_NONE, HOME, HOME, AWAY, AWAY]
    assert owner[7:9].tolist() == [HOME, HOME]
    labels = dfl_labels(alive, owner)
    assert "".join(labels) == "DHHAADDHHDD"


def test_owner_is_kept_when_an_on_ball_event_has_no_team() -> None:
    events = [pff.GameEvent("OTB", 1, False), pff.GameEvent("OTB", 3, None)]
    alive, owner = pff.event_state(events, np.arange(1, 5))
    assert alive.all()
    assert owner.tolist() == [AWAY] * 4


def test_restart_without_a_team_leaves_the_owner_unknown_until_one_is_given() -> None:
    events = [pff.GameEvent("OTB", 1, True), pff.GameEvent("OUT", 3, True),
              pff.GameEvent("OTB", 5, None), pff.GameEvent("OTB", 7, False)]
    alive, owner = pff.event_state(events, np.arange(1, 9))
    assert alive.tolist() == [True, True, False, False, True, True, True, True]
    assert owner.tolist() == [HOME, HOME, HOME, HOME, OWNER_NONE, OWNER_NONE, AWAY, AWAY]


def test_events_without_ids_are_kept_and_deduplicated_by_their_fields() -> None:
    records = [record(1, game_event=event("FIRSTKICKOFF", 1, True)),
               record(2, game_event=event("FIRSTKICKOFF", 1, True)),
               record(3, game_event=event("OUT", 3, True)),
               record(4, game_event=event("OTB", 4, False)),
               record(5, game_event=event("OTB", 5, False))]
    tracking = pff.parse_tracking(records)
    assert [(item.kind, item.start_frame) for item in tracking.events] == [
        ("FIRSTKICKOFF", 1), ("OUT", 3), ("OTB", 4), ("OTB", 5)]


def test_players_without_a_jersey_stay_distinct_and_are_counted_apart() -> None:
    nameless = {"jerseyNum": None, "x": -5.0, "y": 1.0}
    other = {"jerseyNum": None, "x": -6.0, "y": 2.0}
    tracking = pff.parse_tracking([record(1, home=[nameless, other], away=[player("9", 1.0, 1.0)])])
    assert tracking.slot_team.tolist() == [HOME, HOME, AWAY]
    assert tracking.unnumbered_players == 2
    assert tracking.repeated_jerseys == 0
    assert tracking.players[0, 0].tolist() == [-5.0, 1.0]
    assert tracking.players[0, 1].tolist() == [-6.0, 2.0]
    with pytest.warns(UserWarning, match="no jersey number"):
        pff.warn_dropped(tracking)


def test_events_are_read_from_every_record_including_duplicates() -> None:
    records = [record(1, game_event=event("FIRSTKICKOFF", 1, True), event_id=100),
               record(2), record(2, game_event=event("OUT", 2, True), event_id=101),
               record(3, game_event=event("OUT", 2, True), event_id=101),
               record(4, game_event=event("OTB", 4, False), event_id=102)]
    tracking = pff.parse_tracking(records)
    assert [item.kind for item in tracking.events] == ["FIRSTKICKOFF", "OUT", "OTB"]
    series = pff.native_series(tracking, pff.PffMetadata("123", PFF_FPS, PITCH))
    assert "".join(dfl_labels(series.alive, series.owner)) == "HDDA"


def test_native_series_resamples_to_target_fps_with_nearest_labels(tmp_path: Path) -> None:
    count = 300
    records = [record(frame, ball=(frame * 0.01, 0.0),
                      game_event=event("FIRSTKICKOFF", 0, True) if frame == 0 else
                      event("OTB", 150, False) if frame == 150 else None,
                      event_id=frame) for frame in range(count)]
    meta = write_metadata(tmp_path / "Metadata" / "123.json")
    raw = write_tracking(tmp_path / "Tracking Data" / "123.jsonl.bz2", records)
    series = pff.load_series(meta, raw)
    assert series.fps == PFF_FPS
    assert series.pitch == PITCH
    match = harmonise(series, "123", PFF, dfl_labels, TARGET_FPS, True)
    expected = int(np.floor((count - 1) * TARGET_FPS / PFF_FPS)) + 1
    assert match.n_frames == expected
    switch = int(np.argmax(match.labels == "A"))
    assert abs(switch * PFF_FPS / TARGET_FPS - 150) <= 1
    assert set(match.labels[:switch]) == {"H"}
    np.testing.assert_allclose(match.ball[:, 0], 0.5 + np.arange(expected) * PFF_FPS
                               / TARGET_FPS * 0.01 / PITCH[0], atol=1e-9)


def test_metadata_needs_exactly_one_pitch(tmp_path: Path) -> None:
    path = write_metadata(tmp_path / "123.json")
    assert pff.read_metadata(path) == pff.PffMetadata("123", PFF_FPS, PITCH)
    broken = metadata_record()
    broken[0]["stadium"]["pitches"] *= 2
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="one pitch"):
        pff.read_metadata(path)


def test_discovery_merges_identical_copies_across_download_folders(tmp_path: Path) -> None:
    root = tmp_path / PFF
    records = [record(1)]
    write_tracking(root / "part 1" / "Tracking Data" / "3812.jsonl.bz2", records)
    write_tracking(root / "part 2" / "Tracking Data" / "10502.jsonl.bz2", records)
    (root / "part 3" / "Tracking Data").mkdir(parents=True)
    (root / "part 3" / "Tracking Data" / "3812.jsonl.bz2").write_bytes(
        (root / "part 1" / "Tracking Data" / "3812.jsonl.bz2").read_bytes())
    write_metadata(root / "part 2" / "Metadata" / "3812.json", "3812")
    write_metadata(root / "part 3" / "Metadata" / "10502.json", "10502")
    write_tracking(root / "part 3" / "Tracking Data" / "999.jsonl.bz2", records)
    found = sources.pff_matches(root)
    assert [item.name for item in found] == ["3812", "10502"]
    assert all(isinstance(item, MatchFiles) and item.source == PFF for item in found)
    assert found[0].paths[0].parent.parent.name == "part 2"


def test_discovery_rejects_conflicting_copies(tmp_path: Path) -> None:
    root = tmp_path / PFF
    write_tracking(root / "part 1" / "Tracking Data" / "3812.jsonl.bz2", [record(1)])
    write_tracking(root / "part 2" / "Tracking Data" / "3812.jsonl.bz2", [record(2)])
    write_metadata(root / "part 1" / "Metadata" / "3812.json", "3812")
    with pytest.raises(ValueError, match="conflicting copies"):
        sources.pff_matches(root)


def test_copies_are_hashed_only_when_sizes_match_and_each_file_once(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sources.content_digest.cache_clear()
    first, second, longer = (tmp_path / name for name in ("a", "b", "c"))
    first.write_bytes(b"same"), second.write_bytes(b"same"), longer.write_bytes(b"longer")
    reads = []
    original = sources.content_digest.__wrapped__
    monkeypatch.setattr(sources, "content_digest",
                        sources.lru_cache(maxsize=None)(lambda *key: reads.append(key[0])
                                                        or original(*key)))
    assert not sources.same_content(first, longer)
    assert reads == []
    assert sources.same_content(first, second) and sources.same_content(second, first)
    assert sorted(reads) == [first, second]


def test_parser_agrees_with_kloppy_on_positions(tmp_path: Path) -> None:
    kloppy_pff = pytest.importorskip("kloppy.pff")
    records = [record(frame, ball=(3.0 + frame, -7.0),
                      home=[player("7", -10.0 - frame, 5.0)],
                      away=[player("9", 10.0, -5.0 + frame)],
                      game_event=event("FIRSTKICKOFF", 0, True) if frame == 0 else None,
                      event_id=0 if frame == 0 else None) for frame in range(3)]
    for item in records:
        item["ballsSmoothed"] = {"x": item["balls"][0]["x"], "y": item["balls"][0]["y"],
                                 "z": 0.1}
        item["periodElapsedTime"] = item["frameNum"] / PFF_FPS
    meta = write_metadata(tmp_path / "Metadata" / "123.json")
    raw = write_tracking(tmp_path / "Tracking Data" / "123.jsonl.bz2", records)
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps([
        {"team": {"id": "1"}, "player": {"id": "h7", "nickname": "h7"}, "shirtNumber": "7",
         "positionGroupType": "CM"},
        {"team": {"id": "2"}, "player": {"id": "a9", "nickname": "a9"}, "shirtNumber": "9",
         "positionGroupType": "CF"}]))
    dataset = kloppy_pff.load_tracking(meta_data=meta, roster_meta_data=roster, raw_data=raw,
                                       coordinates="kloppy")
    series = pff.load_series(meta, raw)
    for index, frame in enumerate(dataset.records):
        assert series.ball[index] == pytest.approx([frame.ball_coordinates.x,
                                                    frame.ball_coordinates.y])
        by_team = {player_.team.ground.name: data.coordinates
                   for player_, data in frame.players_data.items()}
        assert series.players[index, 0] == pytest.approx([by_team["HOME"].x, by_team["HOME"].y])
        assert series.players[index, 1] == pytest.approx([by_team["AWAY"].x, by_team["AWAY"].y])
