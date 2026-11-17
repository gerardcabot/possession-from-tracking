import numpy as np

from possession import diagnostics
from possession.tagging import TagEvent

PIECE = TagEvent(code="set pieces", start=100, end=200, team=0, kind="throw-in")


def test_restart_hits_use_half_open_set_piece_windows() -> None:
    restarts = np.array([99, 100, 199, 200], dtype=np.int64)
    assert diagnostics.restart_hits(restarts, [PIECE]) == [100]
    assert diagnostics.restarts_in(np.array([200]), PIECE).size == 0
    assert diagnostics.restarts_in(np.array([199]), PIECE).tolist() == [199]
