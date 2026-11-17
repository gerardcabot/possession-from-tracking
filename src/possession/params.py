from dataclasses import dataclass, fields

from possession.cleaning import CleaningParams


@dataclass(frozen=True)
class RulesParams:
    speed_window_s: float = 1.0
    calm_speed_mps: float = 1.5
    speed_scale_mps: float = 0.5
    speed_evidence_cap: float = 3.0
    active_ball_mps: float = 5.0
    active_evidence: float = 1.0
    out_margin_m: float = 0.11
    out_evidence: float = 3.0
    evidence_window_s: float = 2.0
    min_dead_s: float = 3.0
    min_live_s: float = 1.0
    boundary_search_s: float = 3.0
    exit_reach_m: float = 5.0
    exit_min_speed_mps: float = 3.0
    rest_speed_mps: float = 2.0
    rest_s: float = 0.5
    clear_move_m: float = 1.0
    taker_radius_m: float = 2.0
    kickoff_offside_players: float = 2.0
    kickoff_min_tracked: float = 8.0
    kickoff_formation_s: float = 10.0
    kickoff_window_s: float = 300.0
    kickoff_radius_m: float = 2.0
    control_radius_m: float = 1.5
    control_s: float = 0.5
    control_ball_mps: float = 8.0
    gap_s: float = 1.0


LEGACY_CLEANING = CleaningParams(ball_max_speed_mps=40.0, ball_deviation_m=1.6,
                                 player_max_speed_mps=12.0, player_deviation_m=0.41,
                                 duplicate_distance_m=0.5, max_gap_frames=8, ball_half_span=2,
                                 player_half_span=2)
LEGACY_SHAPE_WINDOW_S = 1.0
LEGACY_HALFWAY_MARGIN_M = 1.0
PARAM_NAMES = frozenset(field.name for field in fields(RulesParams))
ZERO_ALLOWED = frozenset({"active_evidence", "out_evidence", "out_margin_m",
                          "kickoff_offside_players"})


def rules_params(values: dict[str, float]) -> RulesParams:
    unknown = sorted(set(values) - PARAM_NAMES)
    if unknown:
        raise ValueError(f"unknown rules params {unknown}")
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"rules param {name} must be a number, got {value!r}")
        if value < 0 or (value == 0 and name not in ZERO_ALLOWED):
            bound = ">= 0" if name in ZERO_ALLOWED else "> 0"
            raise ValueError(f"rules param {name} must be a number {bound}, got {value!r}")
    return RulesParams(**{name: float(value) for name, value in values.items()})


def frames(seconds: float, fps: float) -> int:
    return max(1, int(round(seconds * fps)))
