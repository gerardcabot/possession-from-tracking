import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from possession.data import AWAY, HOME, NOBODY, fps, tactical_paths
from possession.paths import data_dir
from possession.vendor import games

SET_PIECES = "set pieces"
DEFENDING_PREFIX = "defending in"
DEFENSIVE_TRANSITION = "defensive transition"
NO_TEAM = "n/a"
CLUB_WORDS = frozenset({"fc", "cf", "club"})


@dataclass(frozen=True)
class TagEvent:
    code: str
    start: int
    end: int
    team: int
    kind: str


def normalised(text: str | None) -> str:
    return (text or "").strip().lower()


def name_tokens(text: str | None) -> frozenset[str]:
    stripped = unicodedata.normalize("NFKD", text or "")
    plain = "".join(char for char in stripped if not unicodedata.combining(char))
    words = re.findall(r"[a-z0-9]+", plain.casefold())
    return frozenset(words) - CLUB_WORDS


def team_names(meta_path: Path) -> list[str]:
    root = ET.parse(meta_path).getroot()
    names = [team.findtext("Name") for team in root.iter("Team")
             if team.findtext("Name") is not None]
    if len(names) != 2:
        raise ValueError(f"{meta_path}: expected 2 named teams, found {len(names)}")
    return names


def team_code(text: str | None, names: list[str]) -> int:
    tokens = name_tokens(text)
    overlap = [len(tokens & name_tokens(name)) for name in names]
    if overlap[HOME] == overlap[AWAY]:
        return NOBODY
    return HOME if overlap[HOME] > overlap[AWAY] else AWAY


def parse_tags(path: Path, names: list[str], rate: float) -> list[TagEvent]:
    events = []
    for instance in ET.parse(path).iter("instance"):
        labels = {label.findtext("group"): label.findtext("text")
                  for label in instance.findall("label")}
        events.append(TagEvent(
            code=normalised(instance.findtext("code")),
            start=int(round(float(instance.findtext("start")) * rate)),
            end=int(round(float(instance.findtext("end")) * rate)),
            team=team_code(labels.get("Team"), names),
            kind=normalised(labels.get("Type")),
        ))
    if not any(event.team != NOBODY for event in events):
        labelled = sorted({label for label in team_labels(path) if name_tokens(label)})
        raise ValueError(f"{path}: no tag team maps to {names}; tag teams are {labelled}")
    return events


def team_labels(path: Path) -> list[str]:
    return [label.findtext("text") or "" for label in ET.parse(path).iter("label")
            if label.findtext("group") == "Team"]


def load_tags() -> list[TagEvent]:
    meta_path, _ = tactical_paths()
    (path,) = games.require(["smart_tagging"], data_dir=str(data_dir()))
    return parse_tags(Path(path), team_names(meta_path), fps())


def set_pieces(events: list[TagEvent]) -> list[TagEvent]:
    return [event for event in events if event.code == SET_PIECES]


def possessing_team(event: TagEvent) -> int:
    if event.team == NOBODY or event.code == SET_PIECES:
        return event.team
    if event.code.startswith(DEFENDING_PREFIX) or event.code == DEFENSIVE_TRANSITION:
        return 1 - event.team
    return event.team


def phase_team_evidence(events: list[TagEvent], n_frames: int) -> np.ndarray:
    votes = np.zeros((n_frames, 2), dtype=np.int32)
    for event in events:
        team = possessing_team(event)
        if event.code == SET_PIECES or team == NOBODY:
            continue
        votes[max(event.start, 0):min(event.end, n_frames), team] += 1
    return votes
