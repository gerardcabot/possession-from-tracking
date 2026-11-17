import sys
from dataclasses import dataclass

import numpy as np

from possession.cleaning import clean
from possession.control import detect_control
from possession.data import Tracking, fps, graded_blocks, pitch_size
from possession.deadball import DeadEvidence, dead_evidence, dead_mask, refine_start
from possession.features import Features, build_features
from possession.intervals import LABELS, true_runs
from possession.params import (
    LEGACY_CLEANING,
    LEGACY_HALFWAY_MARGIN_M,
    LEGACY_SHAPE_WINDOW_S,
    RulesParams,
    frames,
)
from possession.restart import (
    KickOnsets,
    Restart,
    find_kickoff,
    find_restart,
    first_restart,
    kick_onsets,
    kickoff_window,
)
from possession.statemachine import LiveContext, Stoppage, build_codes, shape_evidence
from possession.tagging import TagEvent, load_tags, set_pieces

RESTART_TOUCH_HORIZON_S = 1.0


def legacy_features(tracking: Tracking, pitch: tuple[float, float], rate: float,
                    blocks: list[tuple[int, int, int]]) -> Features:
    cleaned = clean(tracking, pitch, rate, LEGACY_CLEANING)
    return build_features(cleaned, tracking.player_speed, pitch, rate, blocks,
                          LEGACY_CLEANING.ball_half_span, LEGACY_SHAPE_WINDOW_S,
                          LEGACY_HALFWAY_MARGIN_M)


def match_features(tracking: Tracking, blocks: list[tuple[int, int, int]]) -> Features:
    return legacy_features(tracking, pitch_size(), fps(), blocks)


@dataclass(frozen=True)
class HalfStart:
    first: int
    kickoff: Restart | None


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def half_start(features: Features, onsets: KickOnsets, tags: list[TagEvent], half: int,
               first: int, last: int, params: RulesParams) -> HalfStart:
    kickoff = find_kickoff(features, onsets, first, last + 1, params)
    if kickoff is not None:
        return HalfStart(first=first, kickoff=kickoff)
    window = kickoff_window(features, first, last + 1, params)
    kickoff = first_restart(features, onsets, tags, first, window, params)
    if kickoff is None:
        warn(f"half {half}: no kickoff formation and no restart found near the start; "
             f"forcing only the first frame as dead ball")
    else:
        warn(f"half {half}: no kickoff formation found; forcing dead ball until the first "
             f"detected restart at frame {kickoff.frame}")
    return HalfStart(first=first, kickoff=kickoff)


def half_starts(features: Features, onsets: KickOnsets, tags: list[TagEvent],
                blocks: list[tuple[int, int, int]], params: RulesParams) -> list[HalfStart]:
    return [half_start(features, onsets, tags, half, first, last, params)
            for half, first, last in blocks]


def half_start_mask(dead: np.ndarray, starts: list[HalfStart]) -> np.ndarray:
    forced = dead.copy()
    for start in starts:
        if 0 <= start.first < len(forced):
            until = start.kickoff.frame if start.kickoff else start.first
            forced[start.first:max(until, start.first + 1)] = True
    return forced


def kickoff_in(starts: list[HalfStart], start: int, end: int) -> Restart | None:
    for half in starts:
        if half.kickoff is not None and start <= half.first < end:
            return half.kickoff
    return None


def stoppage_at(features: Features, evidence: DeadEvidence, onsets: KickOnsets,
                tags: list[TagEvent], start: int, end: int, low: int, next_start: int,
                kickoff: Restart | None, params: RulesParams) -> Stoppage:
    search = frames(params.boundary_search_s, features.fps)
    lookahead = frames(params.min_live_s, features.fps)
    begin, by_exit = refine_start(start, evidence.crossings, evidence.collective_speed, low,
                                  search)
    restart = kickoff or find_restart(features, onsets, tags, begin + 1,
                                      min(end + lookahead, next_start), end, params)
    frame = min(max(restart.frame, begin + 1), next_start)
    return Stoppage(start=begin, restart=frame, team=restart.team, by_exit=by_exit)


def detect_stoppages(features: Features, tags: list[TagEvent],
                     blocks: list[tuple[int, int, int]], params: RulesParams) -> list[Stoppage]:
    evidence = dead_evidence(features, params)
    onsets = kick_onsets(features, params)
    pieces = set_pieces(tags)
    halves = half_starts(features, onsets, pieces, blocks, params)
    dead = half_start_mask(dead_mask(evidence.evidence, params, features.fps), halves)
    starts, ends = true_runs(dead)
    stoppages: list[Stoppage] = []
    low = 0
    for index, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        next_start = int(starts[index + 1]) if index + 1 < len(starts) else len(dead)
        stoppage = stoppage_at(features, evidence, onsets, pieces, start, end, low, next_start,
                               kickoff_in(halves, start, end), params)
        stoppages.append(stoppage)
        low = stoppage.restart
    return stoppages


def live_context(features: Features, params: RulesParams) -> LiveContext:
    events = detect_control(features, params)
    return LiveContext(events=events, event_starts=[event.start for event in events],
                       missing=~features.ball_present, shape=shape_evidence(features),
                       gap_frames=frames(params.gap_s, features.fps), nearest=features.nearest,
                       touch_frames=frames(RESTART_TOUCH_HORIZON_S, features.fps),
                       touch_radius_m=params.control_radius_m)


def label_match(features: Features, tags: list[TagEvent], blocks: list[tuple[int, int, int]],
                params: RulesParams) -> np.ndarray:
    stoppages = detect_stoppages(features, tags, blocks, params)
    codes = build_codes(len(features.ball), stoppages, live_context(features, params))
    return LABELS[codes]


def predict(tracking: Tracking, params: RulesParams) -> np.ndarray:
    blocks = graded_blocks()
    return label_match(match_features(tracking, blocks), load_tags(), blocks, params)
