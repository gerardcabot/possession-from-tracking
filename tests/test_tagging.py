from pathlib import Path

import numpy as np
import pytest

from possession import tagging
from possession.data import AWAY, HOME, NOBODY

XML = """<?xml version="1.0" ?>
<file><ALL_INSTANCES>
<instance><ID>1</ID><start>2.0</start><end>4.0</end><code>SET PIECES</code>
<label><group>Team</group><text>N/A</text></label>
<label><group>Type</group><text>Throw-in</text></label></instance>
<instance><ID>2</ID><start>1.0</start><end>3.0</end><code>BUILD UP</code>
<label><group>Team</group><text>Home FC</text></label></instance>
<instance><ID>3</ID><start>2.0</start><end>3.0</end><code>DEFENDING IN MIDDLE THIRD</code>
<label><group>Team</group><text>away club</text></label></instance>
<instance><ID>4</ID><start>5.0</start><end>6.0</end><code>DEFENSIVE TRANSITION</code>
<label><group>Team</group><text>Home FC</text></label></instance>
<instance><ID>5</ID><start>7.0</start><end>8.0</end><code>ATTACKING TRANSITION</code>
<label><group>Team</group><text>Home FC</text></label></instance>
</ALL_INSTANCES></file>
"""
NAMES = ["home fc", "away club"]


def write_tags(tmp_path: Path) -> list[tagging.TagEvent]:
    path = tmp_path / "tags.xml"
    path.write_text(XML)
    return tagging.parse_tags(path, NAMES, 10.0)


def test_parse_tags_converts_seconds_and_teams(tmp_path: Path) -> None:
    events = write_tags(tmp_path)
    assert events[0] == tagging.TagEvent(code="set pieces", start=20, end=40, team=NOBODY,
                                         kind="throw-in")
    assert (events[1].team, events[2].team) == (HOME, AWAY)


def test_team_names_read_from_metadata(tmp_path: Path) -> None:
    meta = tmp_path / "meta.xml"
    meta.write_text("<Root><Teams><Team id='a'><Name>Home FC</Name></Team>"
                    "<Team id='b'><Name>Away Club</Name></Team></Teams></Root>")
    assert tagging.team_names(meta) == ["Home FC", "Away Club"]


def test_defending_phase_votes_for_the_other_team(tmp_path: Path) -> None:
    events = write_tags(tmp_path)
    assert tagging.possessing_team(events[2]) == HOME
    votes = tagging.phase_team_evidence(events, 50)
    assert votes[25].tolist() == [2, 0]
    assert votes[15].tolist() == [1, 0]
    assert votes[35].tolist() == [0, 0]


def test_defensive_transition_votes_for_the_other_team(tmp_path: Path) -> None:
    events = write_tags(tmp_path)
    assert tagging.possessing_team(events[3]) == AWAY
    assert tagging.possessing_team(events[4]) == HOME
    votes = tagging.phase_team_evidence(events, 90)
    assert votes[55].tolist() == [0, 1]
    assert votes[75].tolist() == [1, 0]


def test_set_pieces_filter(tmp_path: Path) -> None:
    assert [event.kind for event in tagging.set_pieces(write_tags(tmp_path))] == ["throw-in"]


def test_phase_evidence_clips_to_match(tmp_path: Path) -> None:
    votes = tagging.phase_team_evidence(write_tags(tmp_path), 22)
    assert votes.shape == (22, 2) and np.all(votes[20:, 0] == 2)


@pytest.mark.parametrize("label, expected", [
    ("FC Barcelona", HOME), ("Barcelona", HOME), ("barcelona cf", HOME),
    ("Club Atlético de Madrid", AWAY), ("Atletico Madrid", AWAY), ("N/A", NOBODY), ("", NOBODY),
    ("Madrid Barcelona", NOBODY),
])
def test_team_code_normalises_names(label: str, expected: int) -> None:
    assert tagging.team_code(label, ["FC Barcelona", "Club Atlético de Madrid"]) == expected


def test_tags_that_map_to_no_team_raise(tmp_path: Path) -> None:
    path = tmp_path / "tags.xml"
    path.write_text(XML)
    with pytest.raises(ValueError, match="no tag team maps"):
        tagging.parse_tags(path, ["Real Sociedad", "Girona"], 10.0)
