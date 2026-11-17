from dataclasses import replace

import numpy as np
from external_synthetic import FPS, make_match

from possession.data import AWAY, HOME
from possession.external import harmonise
from possession.external.harmonise import NativeSeries
from possession.external.match import (
    filled_codes,
    label_codes,
    load_match,
    padded,
    save_match,
    spell_lengths,
)
from possession.vendor import possession_metrics as pm

NATIVE_FPS = 10.0


def native(n_frames: int = 51) -> NativeSeries:
    base = np.array([0.1, 0.2, 0.8, 0.9])
    ramp = np.linspace(0.0, 0.1, n_frames)[:, None]
    players = np.stack([base[None, :] + ramp, np.full((n_frames, 4), 0.5)], axis=-1)
    ball = np.stack([np.linspace(0.0, 1.0, n_frames), np.full(n_frames, 0.25)], axis=1)
    return NativeSeries(
        ball=ball, players=players, slot_team=np.array([HOME, HOME, AWAY, AWAY], dtype=np.int8),
        alive=np.ones(n_frames, dtype=bool), owner=np.full(n_frames, HOME, dtype=np.int8),
        periods=[(1, 0, n_frames - 1)], fps=NATIVE_FPS, pitch=(105.0, 68.0))


def test_dfl_labels_dead_ball_wins_over_possession() -> None:
    alive = np.array([True, True, False, False, True])
    owner = np.array([HOME, AWAY, HOME, AWAY, harmonise.OWNER_NONE], dtype=np.int8)
    assert harmonise.dfl_labels(alive, owner).tolist() == ["H", "A", "D", "D", "X"]


def test_skillcorner_labels_never_dead_and_unowned_is_unknown() -> None:
    alive = np.array([True, False, True])
    owner = np.array([AWAY, harmonise.OWNER_NONE, HOME], dtype=np.int8)
    labels = harmonise.skillcorner_labels(alive, owner)
    assert labels.tolist() == ["A", "X", "H"]
    assert "D" not in labels


def test_resample_length_and_endpoints() -> None:
    values = np.arange(11, dtype=float)
    out = harmonise.resample_linear(values, NATIVE_FPS, FPS)
    assert len(out) == 26
    assert out[0] == 0.0 and out[-1] == 10.0


def test_resample_linear_interpolates_positions() -> None:
    values = np.arange(11, dtype=float)
    out = harmonise.resample_linear(values, NATIVE_FPS, FPS)
    assert np.allclose(out, np.arange(26) * NATIVE_FPS / FPS)


def test_resample_keeps_nan_where_a_neighbour_is_missing() -> None:
    values = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
    out = harmonise.resample_linear(values, NATIVE_FPS, FPS)
    position = np.arange(len(out)) * NATIVE_FPS / FPS
    expected_nan = (position > 1.0) & (position < 3.0)
    assert np.array_equal(np.isnan(out), expected_nan)


def test_resample_nearest_labels_repeat_native_values() -> None:
    labels = np.array(["H", "A", "D"])
    out = harmonise.resample_nearest(labels, NATIVE_FPS, FPS)
    assert out.tolist() == ["H", "H", "A", "A", "D", "D"]


def test_same_rate_resampling_is_identity() -> None:
    values = np.random.default_rng(0).random((20, 3, 2))
    assert np.array_equal(harmonise.resample_linear(values, FPS, FPS), values)


def test_harmonise_resamples_to_target_rate_and_maps_labels() -> None:
    match = harmonise.harmonise(native(), "m", "test", harmonise.dfl_labels, FPS, True)
    assert match.fps == FPS
    assert match.n_frames == 126
    assert match.ball.shape == (126, 2)
    assert match.players.shape == (126, 4, 2)
    assert set(match.labels.tolist()) == {"H"}
    assert match.blocks == [(1, 0, 125)]


def test_harmonise_flips_to_reference_orientation() -> None:
    kept = harmonise.harmonise(native(), "m", "test", harmonise.dfl_labels, FPS, True)
    flipped = harmonise.harmonise(native(), "m", "test", harmonise.dfl_labels, FPS, False)
    assert np.allclose(flipped.players, 1.0 - kept.players)
    assert np.allclose(flipped.ball, 1.0 - kept.ball)
    series = native()
    mirrored = replace(series, ball=1.0 - series.ball, players=1.0 - series.players)
    again = harmonise.harmonise(mirrored, "m", "test", harmonise.dfl_labels, FPS, True)
    assert np.allclose(again.players, kept.players)
    assert np.allclose(again.ball, kept.ball)


def test_orient_like_returns_same_match_when_already_oriented() -> None:
    match = make_match()
    assert harmonise.orient_like(match, True) is match
    assert harmonise.orient_like(match, False) is not match


def test_harmonise_keeps_periods_separate() -> None:
    series = native(21)
    two = NativeSeries(
        ball=np.concatenate([series.ball, series.ball]),
        players=np.concatenate([series.players, series.players]), slot_team=series.slot_team,
        alive=np.ones(42, dtype=bool), owner=np.zeros(42, dtype=np.int8),
        periods=[(1, 0, 20), (2, 21, 41)], fps=NATIVE_FPS, pitch=series.pitch)
    match = harmonise.harmonise(two, "m", "test", harmonise.dfl_labels, FPS, True)
    assert match.blocks == [(1, 0, 50), (2, 51, 101)]


def test_filled_codes_forward_fill_unknown_inside_periods() -> None:
    match = make_match(n_per_period=10, periods=2)
    labels = np.array(list("HXXAXXXXXX") + list("XAXXHXXXXX"), dtype="<U1")
    codes = filled_codes(replace(match, labels=labels))
    assert codes[:10].tolist() == [pm.H, pm.H, pm.H, pm.A, pm.A, pm.A, pm.A, pm.A, pm.A, pm.A]
    assert codes[10:].tolist() == [pm.A, pm.A, pm.A, pm.A, pm.H, pm.H, pm.H, pm.H, pm.H, pm.H]


def test_padded_inserts_one_separator_between_periods() -> None:
    match = make_match(n_per_period=5, periods=2)
    out = padded(np.arange(10), match, -1)
    assert out.tolist() == [0, 1, 2, 3, 4, -1, 5, 6, 7, 8, 9]


def test_spell_lengths_count_possession_runs_per_period() -> None:
    match = make_match(n_per_period=6, periods=2)
    codes = label_codes(np.array(list("HHADDA") + list("AAHHHD"), dtype="<U1"))
    assert sorted(spell_lengths(codes, match).tolist()) == [1, 1, 2, 2, 3]


def test_match_roundtrip(tmp_path) -> None:
    match = make_match()
    loaded = load_match(save_match(tmp_path / "match.npz", match))
    assert (loaded.name, loaded.source, loaded.fps, loaded.pitch) == \
        (match.name, match.source, match.fps, match.pitch)
    for field in ("ball", "players", "slot_team", "labels", "periods"):
        assert np.array_equal(getattr(loaded, field), getattr(match, field))
