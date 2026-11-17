import json
from pathlib import Path

import numpy as np

from possession.data import AWAY, HOME
from possession.external import metrica_events as me
from possession.external.harmonise import OWNER_NONE, dfl_labels
from possession.external.sources import METRICA, SOURCES, metrica_matches

FIRST_FRAME = 1
N_FRAMES = 100
CSV_HEADER = ("Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],"
              "From,To,Start X,Start Y,End X,End Y\n")


def event(team: int, kind: str, start: int, end: int | None = None, subtypes: str = "",
          period: int = 1) -> me.MetricaEvent:
    return me.MetricaEvent(team=team, type=kind, subtypes=me.csv_subtypes(subtypes),
                           period=period, start=start, end=start if end is None else end)


def labels_of(events: list[me.MetricaEvent], n_frames: int = N_FRAMES) -> str:
    frame_ids = np.arange(FIRST_FRAME, FIRST_FRAME + n_frames)
    alive, owner = me.event_state(events, frame_ids, np.ones(n_frames, dtype=np.int64))
    return "".join(dfl_labels(alive, owner))


def test_period_starts_dead_until_kick_off_then_follows_team() -> None:
    labels = labels_of([event(HOME, "SET PIECE", 11, subtypes="KICK OFF"),
                        event(AWAY, "RECOVERY", 51)])
    assert labels == "D" * 10 + "H" * 40 + "A" * 50


def test_kick_off_on_first_frame_leaves_no_dead_frame() -> None:
    labels = labels_of([event(AWAY, "SET PIECE", FIRST_FRAME, FIRST_FRAME - 1,
                              subtypes="KICK OFF")])
    assert labels == "A" * N_FRAMES


def test_ball_lost_keeps_team_until_opponent_controls() -> None:
    labels = labels_of([event(HOME, "SET PIECE", 1), event(HOME, "BALL LOST", 20, 30),
                        event(AWAY, "RECOVERY", 41)])
    assert labels == "H" * 40 + "A" * 60


def test_challenge_never_changes_team() -> None:
    labels = labels_of([event(HOME, "SET PIECE", 1),
                        event(AWAY, "CHALLENGE", 20, subtypes="GROUND-WON"),
                        event(HOME, "CHALLENGE", 30, subtypes="GROUND-LOST")])
    assert labels == "H" * N_FRAMES


def test_pass_start_attributes_team_during_flight() -> None:
    labels = labels_of([event(HOME, "SET PIECE", 1), event(AWAY, "PASS", 30, 60)])
    assert labels == "H" * 29 + "A" * 71


def test_ball_out_is_dead_from_its_end_frame_until_next_restart() -> None:
    labels = labels_of([event(HOME, "SET PIECE", 1), event(HOME, "BALL OUT", 20, 30),
                        event(AWAY, "SET PIECE", 61, subtypes="THROW IN")])
    assert labels == "H" * 29 + "D" * 31 + "A" * 40


def test_goal_kick_pass_ends_dead_ball_like_a_set_piece() -> None:
    labels = labels_of([event(HOME, "SET PIECE", 1), event(HOME, "BALL OUT", 10, 20),
                        event(HOME, "PASS", 41, 50, subtypes="GOAL KICK")])
    assert labels == "H" * 19 + "D" * 21 + "H" * 60


def test_dead_ball_recovery_restarts_play() -> None:
    labels = labels_of([event(HOME, "SET PIECE", 1), event(HOME, "FAULT RECEIVED", 20),
                        event(AWAY, "RECOVERY", 41)])
    assert labels == "H" * 19 + "D" * 21 + "A" * 60


def test_events_inside_dead_ball_before_restart_are_ignored() -> None:
    labels = labels_of([event(HOME, "SET PIECE", 1), event(HOME, "BALL OUT", 10, 30),
                        event(AWAY, "RECOVERY", 20), event(AWAY, "SET PIECE", 51)])
    assert labels == "H" * 29 + "D" * 21 + "A" * 50


def test_second_stop_while_dead_keeps_dead_ball() -> None:
    labels = labels_of([event(HOME, "SET PIECE", 1), event(HOME, "FAULT RECEIVED", 20),
                        event(HOME, "BALL LOST", 25, 30, subtypes="FORCED"),
                        event(HOME, "SET PIECE", 41)])
    assert labels == "H" * 19 + "D" * 21 + "H" * 60


def test_stop_starting_with_a_recovery_during_dead_does_not_revive() -> None:
    labels = labels_of([event(HOME, "SET PIECE", 1), event(AWAY, "FAULT RECEIVED", 20),
                        event(HOME, "RECOVERY", 25, subtypes="INTERCEPTION"),
                        event(HOME, "BALL LOST", 25, 35, subtypes="FORCED"),
                        event(AWAY, "SET PIECE", 51, subtypes="FREE KICK")])
    assert labels == "H" * 19 + "D" * 31 + "A" * 50


def test_recovery_after_stop_span_revives_dead_ball() -> None:
    labels = labels_of([event(HOME, "SET PIECE", 1), event(AWAY, "FAULT RECEIVED", 20),
                        event(HOME, "BALL LOST", 25, 35, subtypes="FORCED"),
                        event(HOME, "RECOVERY", 36)])
    assert labels == "H" * 19 + "D" * 16 + "H" * 65


def test_hand_ball_is_a_stop_when_a_free_kick_follows() -> None:
    labels = labels_of([event(HOME, "SET PIECE", 1),
                        event(HOME, "BALL LOST", 10, 20, subtypes="HAND BALL"),
                        event(AWAY, "SET PIECE", 51, subtypes="FREE KICK")])
    assert labels == "H" * 19 + "D" * 31 + "A" * 50


def test_offside_with_play_continuing_is_not_a_stop() -> None:
    labels = labels_of([event(HOME, "SET PIECE", 1), event(HOME, "PASS", 10, 20, subtypes="OFFSIDE"),
                        event(HOME, "CARRY", 30)])
    assert labels == "H" * N_FRAMES


def test_offside_followed_by_free_kick_is_a_stop() -> None:
    labels = labels_of([event(HOME, "SET PIECE", 1), event(HOME, "PASS", 10, 20, subtypes="OFFSIDE"),
                        event(AWAY, "SET PIECE", 41, subtypes="FREE KICK")])
    assert labels == "H" * 19 + "D" * 21 + "A" * 60


def test_set_piece_without_recognised_stop_starts_dead_at_last_event_end() -> None:
    labels = labels_of([event(HOME, "SET PIECE", 1), event(HOME, "PASS", 10, 20),
                        event(AWAY, "SET PIECE", 41, subtypes="THROW IN")])
    assert labels == "H" * 19 + "D" * 21 + "A" * 60


def test_set_piece_right_after_event_has_no_dead_frames() -> None:
    labels = labels_of([event(HOME, "SET PIECE", 1), event(HOME, "PASS", 10, 40),
                        event(AWAY, "SET PIECE", 31, subtypes="THROW IN")])
    assert labels == "H" * 30 + "A" * 70


def test_stop_by_non_owning_team_takes_possession_from_its_start() -> None:
    labels = labels_of([event(HOME, "SET PIECE", 1), event(AWAY, "BALL OUT", 10, 20,
                                                           subtypes="CLEARANCE"),
                        event(HOME, "SET PIECE", 41, subtypes="THROW IN")])
    assert labels == "H" * 9 + "A" * 10 + "D" * 21 + "H" * 60


def test_goal_and_shot_out_stop_play_but_saved_shot_does_not() -> None:
    kick_off = labels_of([event(HOME, "SET PIECE", 1),
                          event(HOME, "SHOT", 10, 20, subtypes="ON TARGET-GOAL"),
                          event(AWAY, "SET PIECE", 61, subtypes="KICK OFF")])
    assert kick_off == "H" * 19 + "D" * 41 + "A" * 40
    goal_kick = labels_of([event(HOME, "SET PIECE", 1),
                           event(HOME, "SHOT", 10, 20, subtypes="OFF TARGET-OUT"),
                           event(AWAY, "PASS", 61, subtypes="GOAL KICK")])
    assert goal_kick == "H" * 19 + "D" * 41 + "A" * 40
    saved = labels_of([event(HOME, "SET PIECE", 1),
                       event(HOME, "SHOT", 10, 20, subtypes="ON TARGET-SAVED"),
                       event(AWAY, "RECOVERY", 21, subtypes="SAVED")])
    assert saved == "H" * 20 + "A" * 80


def test_ball_lost_stops_only_for_referee_subtypes() -> None:
    forced = labels_of([event(HOME, "SET PIECE", 1),
                        event(HOME, "BALL LOST", 10, 20, subtypes="FORCED"),
                        event(AWAY, "SET PIECE", 51, subtypes="FREE KICK")])
    assert forced == "H" * 19 + "D" * 31 + "A" * 50
    plain = labels_of([event(HOME, "SET PIECE", 1),
                       event(HOME, "BALL LOST", 10, 20, subtypes="INTERCEPTION")])
    assert plain == "H" * N_FRAMES


def test_card_does_not_stop_play() -> None:
    assert labels_of([event(HOME, "SET PIECE", 1), event(AWAY, "CARD", 30, subtypes="YELLOW")]) \
        == "H" * N_FRAMES


def test_end_half_stays_dead_to_period_end() -> None:
    labels = labels_of([event(HOME, "SET PIECE", 1),
                        event(HOME, "BALL LOST", 70, 80, subtypes="END HALF")])
    assert labels == "H" * 79 + "D" * 21


def test_periods_are_labelled_independently() -> None:
    frame_ids = np.arange(1, 21)
    period_ids = np.array([1] * 10 + [2] * 10, dtype=np.int64)
    events = [event(HOME, "SET PIECE", 1), event(AWAY, "SET PIECE", 15, period=2)]
    alive, owner = me.event_state(events, frame_ids, period_ids)
    assert "".join(dfl_labels(alive, owner)) == "H" * 10 + "D" * 4 + "A" * 6


def test_subtype_tokens_split_on_hyphen_only() -> None:
    assert me.csv_subtypes("HEAD-OFF TARGET-OUT") == frozenset({"HEAD", "OFF TARGET", "OUT"})
    assert me.csv_subtypes("") == frozenset()


def test_json_subtypes_accept_none_object_and_list() -> None:
    assert me.json_subtypes(None) == frozenset()
    assert me.json_subtypes({"name": "GOAL", "id": 30}) == frozenset({"GOAL"})
    assert me.json_subtypes([{"name": "OFF TARGET"}, {"name": "OUT"}]) == frozenset(
        {"OFF TARGET", "OUT"})


def test_read_events_csv(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"
    path.write_text(CSV_HEADER
                    + "Away,SET PIECE,KICK OFF,1,1,0.04,0,0,Player19,,NaN,NaN,NaN,NaN\n"
                    + "Home,SHOT,HEAD-ON TARGET-GOAL,1,2289,91.56,2309,92.36,P1,,0.1,0.2,0.3,0.4\n")
    events = me.read_events_csv(path)
    assert [(item.team, item.type, item.start, item.end) for item in events] == [
        (AWAY, "SET PIECE", 1, 0), (HOME, "SHOT", 2289, 2309)]
    assert events[1].subtypes == frozenset({"HEAD", "ON TARGET", "GOAL"})


def test_read_events_json(tmp_path: Path) -> None:
    rows = [{"team": {"id": "TA"}, "type": {"name": "PASS"}, "subtypes": None, "period": 1,
             "start": {"frame": 5}, "end": {"frame": 9}},
            {"team": {"id": "TB"}, "type": {"name": "BALL OUT"}, "subtypes": {"name": "CLEARANCE"},
             "period": 2, "start": {"frame": 20}, "end": {"frame": 30}}]
    path = tmp_path / "events.json"
    path.write_text(json.dumps({"data": rows, "metadata": []}))
    events = me.read_events_json(path, {"TA": HOME, "TB": AWAY})
    assert [(item.team, item.type, item.period, item.start, item.end) for item in events] == [
        (HOME, "PASS", 1, 5, 9), (AWAY, "BALL OUT", 2, 20, 30)]
    assert events[1].subtypes == frozenset({"CLEARANCE"})


def test_owner_none_while_dead() -> None:
    frame_ids = np.arange(1, 6)
    _, owner = me.event_state([], frame_ids, np.ones(5, dtype=np.int64))
    assert owner.tolist() == [OWNER_NONE] * 5


def test_metrica_is_a_source_with_dead_labels() -> None:
    assert SOURCES[METRICA].has_dead_labels
    assert SOURCES[METRICA].dead_labels_heuristic


def test_metrica_matches_discovers_csv_and_epts_games(tmp_path: Path) -> None:
    data = tmp_path / "repo" / "data"
    for name in ("Sample_Game_1_RawTrackingData_Home_Team.csv",
                 "Sample_Game_1_RawTrackingData_Away_Team.csv", "Sample_Game_1_RawEventsData.csv"):
        (data / "Sample_Game_1").mkdir(parents=True, exist_ok=True)
        (data / "Sample_Game_1" / name).write_text("")
    for name in ("Sample_Game_3_metadata.xml", "Sample_Game_3_tracking.txt",
                 "Sample_Game_3_events.json"):
        (data / "Sample_Game_3").mkdir(parents=True, exist_ok=True)
        (data / "Sample_Game_3" / name).write_text("")
    (data / "Sample_Game_2").mkdir()
    found = metrica_matches(tmp_path)
    assert [item.name for item in found] == ["game1", "game3"]
    assert found[1].paths[2].suffix == ".json"


def test_metrica_matches_picks_one_format_per_game_preferring_epts(tmp_path: Path) -> None:
    folder = tmp_path / "repo" / "data" / "Sample_Game_1"
    folder.mkdir(parents=True)
    for name in ("RawTrackingData_Home_Team.csv", "RawTrackingData_Away_Team.csv",
                 "RawEventsData.csv", "metadata.xml", "tracking.txt", "events.json"):
        (folder / f"Sample_Game_1_{name}").write_text("")
    found = metrica_matches(tmp_path)
    assert [item.name for item in found] == ["game1"]
    assert found[0].paths[2].suffix == ".json"
