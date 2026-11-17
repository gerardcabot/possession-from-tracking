from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from possession.data import AWAY, HOME

domain = pytest.importorskip("kloppy.domain")
loaders = pytest.importorskip("possession.external.loaders")
Ground = domain.Ground
BallState = domain.BallState


@dataclass(frozen=True)
class FakeTeam:
    ground: object


@dataclass(frozen=True)
class FakePlayer:
    player_id: str
    team: FakeTeam | None


def fake_dataset(frames: list[dict[FakePlayer, tuple[float, float]]]) -> SimpleNamespace:
    known = [FakePlayer("h1", None), FakePlayer("a1", None)]
    teams = [SimpleNamespace(ground=Ground.HOME, players=[known[0]]),
             SimpleNamespace(ground=Ground.AWAY, players=[known[1]])]
    records = [SimpleNamespace(
        ball_coordinates=None, ball_state=BallState.ALIVE, ball_owning_team=None,
        period=SimpleNamespace(id=1),
        players_data={player: SimpleNamespace(coordinates=SimpleNamespace(x=x, y=y))
                      for player, (x, y) in frame.items()}) for frame in frames]
    metadata = SimpleNamespace(
        teams=teams, frame_rate=25, pitch_dimensions=SimpleNamespace(pitch_length=105.0,
                                                                     pitch_width=68.0))
    return SimpleNamespace(metadata=metadata, records=records)


def test_unknown_player_takes_team_from_its_player_record() -> None:
    stranger = FakePlayer("sub", FakeTeam(Ground.AWAY))
    frames = [{FakePlayer("h1", None): (0.1, 0.1), FakePlayer("a1", None): (0.9, 0.9),
               stranger: (0.8, 0.8)}] * 3
    series = loaders.native_series(fake_dataset(frames))
    assert series.players.shape == (3, 3, 2)
    assert series.slot_team.tolist() == [HOME, AWAY, AWAY]
    assert np.allclose(series.players[:, 2], 0.8)


def test_unknown_player_without_team_is_dropped_with_counted_warning() -> None:
    stranger = FakePlayer("ghost", None)
    frames = [{FakePlayer("h1", None): (0.1, 0.1), FakePlayer("a1", None): (0.9, 0.9),
               stranger: (0.5, 0.5)}] * 4
    with pytest.warns(UserWarning, match=r"dropped 4 position samples of 1 players"):
        series = loaders.native_series(fake_dataset(frames))
    assert series.players.shape == (4, 2, 2)
    assert series.slot_team.tolist() == [HOME, AWAY]
