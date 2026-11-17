from dataclasses import replace

import numpy as np
from external_synthetic import FPS, PITCH, make_match

from possession.cleaning import CleaningParams
from possession.data import AWAY, HOME, Tracking
from possession.external import degrade, stats
from possession.external.match import ExternalMatch, period_ids
from possession.geometry import present

TARGET_MISSING = 0.4
SEED = 7
PARAMS = CleaningParams(ball_max_speed_mps=40.0, ball_deviation_m=1.6,
                        player_max_speed_mps=12.0, player_deviation_m=0.3,
                        duplicate_distance_m=0.5, max_gap_frames=8, ball_half_span=1,
                        player_half_span=1)


def make_stats(**overrides: object) -> stats.OurStats:
    values = {
        "fps": FPS,
        "ball_observed_share": 1.0 - TARGET_MISSING,
        "gap_lengths": np.array([20, 40, 80, 80, 160, 300]),
        "slot_counts": np.array([14, 13]),
        "segment_lengths": np.array([200, 500, 900, 1500]),
        "objects_per_frame": 19.0,
        "duplicate_share": 0.01,
        "duplicate_lengths": np.array([5, 10, 20]),
        "duplicate_offsets_m": np.array([0.05, 0.2, 0.4]),
        "duplicate_distance_m": 0.5,
        "player_sigma_m": 0.05,
        "ball_sigma_m": 0.08,
        "propensity_mean": np.array([1.0, 25.0]),
        "propensity_scale": np.array([1.0, 15.0]),
        "propensity_coef": np.array([0.3, -0.6]),
        "home_attacks_positive_first": True,
    }
    values.update(overrides)
    return stats.OurStats(**values)


def missing_share(match: ExternalMatch) -> float:
    return float((~present(match.ball)).mean())


def same(first: ExternalMatch, second: ExternalMatch) -> bool:
    return np.array_equal(first.ball, second.ball, equal_nan=True) and \
        np.array_equal(first.players, second.players, equal_nan=True)


def test_degradation_is_deterministic_for_a_seed() -> None:
    match, reference = make_match(), make_stats()
    first, _ = degrade.degrade(match, reference, SEED)
    second, _ = degrade.degrade(match, reference, SEED)
    assert same(first, second)


def test_different_seed_gives_different_degradation() -> None:
    match, reference = make_match(), make_stats()
    first, _ = degrade.degrade(match, reference, SEED)
    second, _ = degrade.degrade(match, reference, SEED + 1)
    assert not same(first, second)


def test_ball_missing_share_reaches_target() -> None:
    degraded, report = degrade.degrade(make_match(), make_stats(), SEED)
    assert abs(missing_share(degraded) - TARGET_MISSING) < 0.01
    assert abs(report.ball_missing_after - report.ball_missing_target) < 0.01


def test_existing_ball_gaps_count_towards_target() -> None:
    match = make_match()
    ball = match.ball.copy()
    ball[100:900] = np.nan
    degraded, report = degrade.degrade(replace(match, ball=ball), make_stats(), SEED)
    assert report.ball_missing_before > 0.1
    assert abs(missing_share(degraded) - TARGET_MISSING) < 0.01
    assert np.isnan(degraded.ball[100:900]).all()


def test_no_extra_gaps_when_ball_already_sparser_than_target() -> None:
    match = make_match()
    ball = match.ball.copy()
    ball[: int(0.6 * len(ball))] = np.nan
    degraded, _ = degrade.degrade(replace(match, ball=ball), make_stats(), SEED)
    assert np.array_equal(present(degraded.ball), present(ball))


def test_gaps_follow_the_empirical_length_set_and_stay_inside_periods() -> None:
    degraded, _ = degrade.degrade(make_match(), make_stats(), SEED)
    ids = period_ids(degraded)
    missing = ~present(degraded.ball)
    starts, ends = np.flatnonzero(missing & ~np.r_[False, missing[:-1]]), \
        np.flatnonzero(missing & ~np.r_[missing[1:], False]) + 1
    assert all(ids[start] == ids[end - 1] for start, end in zip(starts, ends))
    allowed = set(make_stats().gap_lengths.tolist())
    lengths = (ends - starts).tolist()
    assert sum(length in allowed for length in lengths) >= len(lengths) - 1


def test_gaps_prefer_high_propensity_frames() -> None:
    match = make_match(n_per_period=6000)
    reference = make_stats(propensity_coef=np.array([0.0, -3.0]))
    degraded, _ = degrade.degrade(match, reference, SEED)
    edge = np.minimum(match.ball[:, 0], 1.0 - match.ball[:, 0])
    missing = ~present(degraded.ball)
    assert edge[missing].mean() < edge[~missing].mean()


def test_slot_pool_matches_reference_counts_and_teams() -> None:
    degraded, _ = degrade.degrade(make_match(), make_stats(), SEED)
    assert degraded.players.shape[1] == 27
    assert (degraded.slot_team == HOME).sum() == 14
    assert (degraded.slot_team == AWAY).sum() == 13


def test_object_count_is_brought_down_to_reference() -> None:
    match = make_match()
    degraded, report = degrade.degrade(match, make_stats(duplicate_share=0.0), SEED)
    assert report.objects_before == 22.0
    assert abs(report.objects_after - report.objects_target) < 1.5


def test_slots_fragment_a_continuous_player() -> None:
    match = make_match()
    degraded, _ = degrade.degrade(match, make_stats(duplicate_share=0.0, objects_per_frame=22.0,
                                                    player_sigma_m=0.0), SEED)
    first_player = match.players[:, 0]
    used = [slot for slot in range(degraded.players.shape[1])
            if np.isin(degraded.players[:, slot, 0], first_player[:, 0]).any()]
    assert len(used) > 1


def test_duplicates_are_added_near_a_same_team_player() -> None:
    match = make_match()
    _, without = degrade.degrade(match, make_stats(duplicate_share=0.0), SEED)
    _, report = degrade.degrade(match, make_stats(), SEED)
    assert abs(report.duplicate_after - without.duplicate_after - report.duplicate_target) < 0.003


def duplicate_pool(match: ExternalMatch) -> tuple[np.ndarray, np.ndarray]:
    pool_team = np.repeat(np.array([HOME, AWAY], dtype=np.int8), [14, 13])
    squad = np.full((match.n_frames, 27, 2), np.nan)
    squad[:, :11] = match.players[:, :11]
    squad[:, 14:25] = match.players[:, 11:]
    return squad, pool_team


def test_duplicates_reach_target_without_overwriting_or_double_counting() -> None:
    match = make_match()
    squad, pool_team = duplicate_pool(match)
    share = 0.03
    result = degrade.add_duplicates(squad, pool_team, share, np.array([5, 10, 20]),
                                    np.array([0.05, 0.2, 0.4]), match.blocks, PITCH,
                                    np.random.default_rng(SEED))
    originals = present(squad)
    assert np.array_equal(result[originals], squad[originals])
    added = int(present(result).sum() - originals.sum())
    target = degrade.duplicate_target_samples(share, int(originals.sum()))
    assert abs(added - target) <= 20
    assert abs(added / present(result).sum() - share) < 0.002


def test_duplicate_target_makes_share_of_all_objects() -> None:
    assert degrade.duplicate_target_samples(0.5, 100) == 100
    assert degrade.duplicate_target_samples(0.0, 100) == 0


def test_thinned_objects_match_target_before_duplicates_are_added() -> None:
    _, report = degrade.degrade(make_match(), make_stats(), SEED)
    assert abs(report.objects_after - report.objects_target) < 1.5


def test_dense_gap_target_is_still_reached_without_overlap() -> None:
    reference = make_stats(ball_observed_share=0.15, gap_lengths=np.array([10, 20, 40, 80]))
    degraded, report = degrade.degrade(make_match(), reference, SEED)
    assert abs(report.ball_missing_after - report.ball_missing_target) < 0.02
    assert abs(missing_share(degraded) - report.ball_missing_target) < 0.02


def test_jitter_added_only_for_missing_variance() -> None:
    assert abs(degrade.added_sigma(0.05, 0.03) - 0.04) < 1e-12
    assert degrade.added_sigma(0.02, 0.05) == 0.0


def test_white_noise_sigma_recovers_noise_level() -> None:
    rng = np.random.default_rng(0)
    sigma = 0.07
    positions = rng.normal(scale=sigma, size=(20000, 5, 2)) + 10.0
    estimate = stats.white_noise_sigma(positions, [(1, 0, 19999)])
    assert abs(estimate - sigma) / sigma < 0.05


def test_draw_until_sums_exactly() -> None:
    drawn = degrade.draw_until(1234, np.array([10, 50, 400]), np.random.default_rng(0))
    assert drawn.sum() == 1234
    assert (drawn >= 1).all()


def test_logistic_fit_recovers_sign_of_effect() -> None:
    rng = np.random.default_rng(0)
    design = rng.normal(size=(20000, 2))
    probability = 1.0 / (1.0 + np.exp(-(0.5 * design[:, 0] - 1.0 * design[:, 1])))
    outcome = (rng.random(20000) < probability).astype(float)
    coef = stats.fit_logistic(design, outcome)
    assert coef[0] > 0.3 and coef[1] < -0.7


def test_logistic_fit_is_unpenalised() -> None:
    rng = np.random.default_rng(1)
    design = rng.normal(size=(400, 2))
    outcome = (rng.random(400) < 1.0 / (1.0 + np.exp(-(3.0 * design[:, 0])))).astype(float)
    coef = stats.fit_logistic(design, outcome)
    unpenalised = newton_logit(design, outcome)
    np.testing.assert_allclose(coef, unpenalised, atol=1e-2)


def newton_logit(design: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    beta = np.zeros(design.shape[1] + 1)
    x = np.column_stack([np.ones(len(design)), design])
    for _ in range(50):
        probability = 1.0 / (1.0 + np.exp(-x @ beta))
        hessian = x.T @ (x * (probability * (1.0 - probability))[:, None])
        beta = beta + np.linalg.solve(hessian, x.T @ (outcome - probability))
    return beta[1:]


def synthetic_tracking(n_frames: int = 600) -> Tracking:
    match = make_match(n_per_period=n_frames // 2)
    ball = match.ball.copy()
    ball[100:160] = np.nan
    ball[400:430] = np.nan
    return Tracking(ball=ball, ball_speed=np.zeros(n_frames), players=match.players,
                    player_speed=np.zeros(match.players.shape[:2]), slot_team=match.slot_team)


def test_our_stats_measures_ball_coverage_and_gaps() -> None:
    tracking = synthetic_tracking()
    blocks = [(1, 0, 299), (2, 300, 599)]
    result = stats.our_stats(tracking, PITCH, FPS, blocks, PARAMS)
    assert abs(result.ball_observed_share - (1 - 90 / 600)) < 1e-9
    assert sorted(result.gap_lengths.tolist()) == [30, 60]
    assert result.slot_counts.tolist() == [11, 11]
    assert result.home_attacks_positive_first


def test_stats_digest_changes_with_content() -> None:
    assert make_stats().digest() == make_stats().digest()
    assert make_stats().digest() != make_stats(ball_observed_share=0.5).digest()


def test_stats_digest_covers_names_dtype_shape_and_boundaries() -> None:
    swapped = make_stats(ball_sigma_m=0.05, player_sigma_m=0.08)
    assert make_stats().digest() != swapped.digest()
    ints = make_stats(gap_lengths=np.array([20, 40], dtype=np.int64))
    floats = make_stats(gap_lengths=np.array([20, 40], dtype=np.float64))
    assert ints.digest() != floats.digest()
    split = make_stats(gap_lengths=np.array([1, 2, 3]), segment_lengths=np.array([4]))
    moved = make_stats(gap_lengths=np.array([1, 2]), segment_lengths=np.array([3, 4]))
    assert split.digest() != moved.digest()


def test_stats_array_roundtrip() -> None:
    original = make_stats()
    restored = stats.OurStats.from_arrays(original.to_arrays())
    assert restored.digest() == original.digest()
    assert isinstance(restored.home_attacks_positive_first, bool)
    assert isinstance(restored.fps, float)
