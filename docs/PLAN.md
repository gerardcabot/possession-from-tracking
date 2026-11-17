# Possession Challenge — Implementation Plan (Oct 25 → Nov 17 2026)

> **Status at submission.** This is the original plan, kept as written; its phases and checkboxes were not updated. The final method is the `learned` model in `models/learned/`: the earlier internal model v3 retrained with the pitch-line features, adopted on 2026-11-13 (README, "Reproducing the submission"; `docs/DEADBALL.md`; `docs/PARAMETERS.md`). The compliance checklist actually maintained is `docs/COMPLIANCE.md`. Phase 4 (own ball detection on video) was not done.

## 0. Key facts and gotchas

**Frames and alignment**
- Submission: exactly 199,319 rows, `frame` 0…199318 in order, labels H/A/D/X, header `frame,label` (SUBMISSION.md:18-26).
- Video frame = ATD `frame_id - 1` (DATA.md:242-247). Smart Tagging frame = `round(seconds*25)` (DATA.md:162-166). Homography `frame` = submission row.
- Per-half offsets in games.py:51 belong to the GT's internal counter; not needed.
- Graded frames: 11,185–90,651 and 110,808–190,197 (158,857) (DATA.md:254-263). Use `games.half_video_blocks("BAR-ATH")`.
- Pre-kickoff trap: 110,808–113,741 graded and all D; 2nd-half kickoff at 113,742 (FAQ.md:38-41).
- games.py:59-60 anchors decode to 11186/113741, off by one vs docs. 1 frame is free on spellF1. DATA.md values authoritative.
- H = Barcelona = first `<Team>` in ATD XML; A = Athletic.
- Ball present on 59.3% of graded frames, missing more in boxes/aerials/tackles (FAQ.md:45-47).
- 46 slots are tracklets, not people; ~20.5 objects per graded frame, sometimes >22 (duplicates).
- ATD groups are `x,y,speed`; `data.py` currently drops speed.
- Pitch 105.3×68.0 m (games.py:62), coords normalised 0–1.

**Label semantics**
- Three states only. Contested/loose ball keeps last team. Change needs controlled possession by opponent (CHALLENGE.md:55-96).
- After every stoppage there is a clear possessor at restart.
- GT = mechanical derivation from Metrica corrected tracking: D where ball flagged dead, else owning team, unsmoothed (CHALLENGE.md:97-103). Very short spells possible.
- Public GT is the reference for semantics (CHALLENGE.md:82-87).

**spellF1 (evaluation/possession_metrics.py)**
- Computed per half over GT non-X spans; predictions outside halves never seen.
- X forward-filled with last non-X in same half, back-filled at half start (274-293). X on graded frame = previous label on spellF1, miss on macroF1.
- **Never write X on graded frames.** Write D explicitly in pre-kickoff block.
- Spells = maximal H/A runs; D never a spell. Greedy one-to-one IoU matching, same team, mean F1 over IoU 0.3–0.7.
- Timing: 3/8/25 frames late → 0.98/0.89/0.65. Decode offline, non-causal, no lag.
- Noise: spellF1 ~0.04, macroF1 ~0.01 on full match.
- Ranking: spellF1 → macroF1 → transition F1 (tol 25) → latency.
- Floor: baseline_smooth50 0.2433 spellF1 full match.

**Smart Tagging**
- 638 automatic instances; 59 set pieces (26 throw-ins, 12 corners, 10 goal kicks, 5 FK, 6 kickoffs). Many restarts untagged.
- Team N/A on 36, incl. all throw-ins. Windows open ~3.5 s before restart. Phases, not possessions.
- "Defending in …" tagged with defending team → possessor is the other.

## 1. Definition we implement

- H/A/D on every graded frame; X only on ungraded frames.
- Ball alive → team T keeps possession until opponent gains control. Loose, contested, deflections, blocks, uncontrolled aerials keep previous team.
- Control event by T: ball within r_ctrl of a T player ≥ t_dwell; release along pass/shot trajectory after dwell; two consecutive T touches; ball reappears in T dwell after a gap.
- Switch frame: first frame of opponent's controlling touch (offline back-dating).
- D starts at: ball crossing pitch lines (back-dated to crossing), goal, whistle-type stoppage (collective deceleration, ball static/missing).
- D ends at restart touch (first ball movement out of set-piece configuration). New possession → restarting team (geometry, confirmed by Smart Tagging team when available).
- Half starts are D until kickoff touch: detected for `rules`; for `learned` the official games.py kickoff anchors are used to force D (team decision 2026-10-26, organiser question open).
- Conventions informed by a one-time, logged, qualitative study of public GT (when D starts/ends, switch at first touch vs control). Written as rules in words; no numeric fitting.

## 2. Data cleaning

Superseded for `learned` (Oct 25 2026): the numbers below were the Phase 0 plan and now live only in
the legacy `rules` method. The `learned` cleaning estimates every bound from data and merges
duplicates; see `docs/PARAMETERS.md`, "Cleaning parameters".

1. Parse speed channels; cache to gitignored `.npz` with deterministic hash.
2. Ball outliers: isolated jumps > ~40 m/s; implausible acceleration; positions > ~3 m off pitch flagged as off-pitch candidates.
3. Spare/false balls: continuity tracker rejecting detections far from predicted trajectory unless persistent.
4. Gaps: linear interpolation only ≤ ~8 frames with plausible speed; longer gaps keep features (length, entry/exit position, nearest team at entry/exit). Optional Kalman/RTS smoother.
5. Players: drop same-team duplicates < 0.5 m, jumps > 12 m/s; recompute speed; check slot team consistency; per-frame goalkeeper detection.
6. Ungraded frames → X.
7. Aggregate diagnostics: ball coverage per minute, gap-length histogram, gaps per zone, object count, outlier counts.

## 3. Features (per frame, metres, 25 Hz)

- Ball: position, velocity, speed, acceleration, direction change, on-pitch, distance to line, goal mouth, missing flag, gap length so far/remaining, reappearance position.
- Ball–player: nearest and second-nearest distance per team, nearest team, relative velocity, dwell counters at 1.0/1.5/2.5 m, players within 3/5 m per team.
- Touch events: ball velocity discontinuities with player within ~1.5 m → touch team, same-team touch runs, time since last touch per team.
- Team shape: centroid and velocity along attack axis, length, width, hull area, defensive-line depth, compactness, mean/p90 speed, attacking direction per half.
- Set-piece configuration: kickoff, throw-in, corner, goal kick, free kick (wall), penalty.
- Smart Tagging: set-piece windows, phase team evidence (attacking phases → T, defending phases → other), goals.
- Optional: audio whistle detection from tactical mp4.

## 4. Phases

### Phase 0 — infrastructure and compliance (Oct 25–27)
- Self-contained repo: vendor `games.py` and `possession_metrics.py` (MIT, attributed) or submodule.
- `uv.lock`; single command `python -m possession.run --config configs/final.yaml --out <team>.csv`.
- Determinism test (sha256 twice). Cleaning module, feature cache, aggregate-only diagnostics CLI.
- `docs/COMPLIANCE.md`.
- Acceptance: baseline byte-identical, tests pass, fresh clone same hash, validator OK.

### Phase 1 — rule-based state machine (Oct 28–Nov 3)
1. Dead-ball detector: ball exit, goal, stoppage (speed collapse + static/missing ball ≥ T), fused with Smart Tagging set pieces.
2. Restart detector: set-piece configuration + first ball movement; team by geometry, Smart Tagging when present.
3. Control detector (§1).
4. State machine: D until restart → restarting team; switch only on opponent control, back-dated; long gaps compare entry/exit control.
5. Parameters from literature and external data, never public GT.
- Acceptance: on degraded external games beats external nearest-player ±50 by > 0.04 spellF1. BAR-ATH diagnostics in priors: D share ~40–48%, stoppages ~90–130, spell count/duration within external range. ≥ 90% Smart Tagging set pieces contain a detected restart.

### Phase 2 — HMM/HSMM + Viterbi (Nov 3–9)
1. States: H_ctrl, H_loose, A_ctrl, A_loose, D_out, restart sub-states.
2. Emissions: logistic/naive-Bayes log-likelihoods on §3; missing ball = uninformative ball emission, shape continues.
3. Semi-Markov durations fitted on external data.
4. Exact offline Viterbi; optional posterior boundary refinement.
5. Forced segments (pre-kickoff D, detected restarts) as hard constraints.
- Acceptance: external leave-one-game-out beats Phase 1 with non-overlapping bootstrap CIs; ablations; ±20% prior sensitivity moves possession count < ~15%.

### Phase 3 — supervised emissions on other games (Nov 1–11, parallel)
1. External data (verify availability and licence): DFL/IDSSE open Bundesliga (`BallStatus`/`BallPossession` via kloppy `sportec`), Metrica Sample Games 1–3 (algorithmic labels from events), SkillCorner open data (optional).
2. Degradation simulator from BAR-ATH aggregate gap statistics, slot fragmentation, jitter, duplicates.
3. Deterministic seeded LightGBM for P(control by team), P(dead), P(restart) → Phase 2 emissions.
4. Commit weights or regenerate deterministically.
- Acceptance: LOGO beats Phase 2 by > 0.04 spellF1; every dataset documented.

### Phase 4 (optional, if Phase 3 done by Nov 10) — own ball detection on tactical video
- Public pretrained ball detector on ATD ball-gap frames, project with homography, associate with trajectory.
- Acceptance: > 90% agreement within 1 m where ATD has ball; > 10 pp graded coverage gain; reproducible on CPU or documented GPU.

## 5. Validation protocol

1. Primary: external games, degraded, scored with official `possession_metrics.evaluate` (+ bootstrap). All selection here.
2. Secondary, label-free on BAR-ATH: possession count/duration histogram, D share, stoppage count, restart match rate with Smart Tagging, phase-team agreement, perturbation stability.
3. Public GT only for: one-time qualitative semantics study, alignment/format checks, a single sanity score of the frozen final candidate. Never compare variants on it.
4. Forbidden: hand labels on BAR-ATH, any human data about BAR-ATH, tuning to public GT counts.

## 6. Compliance checklist

- [ ] All labels from code; no hand corrections; no frame-specific constants (published half/kickoff frames only as asserts or masks — confirm with organisers).
- [ ] No GT or human data about BAR-ATH beyond §5.3.
- [ ] Match data, caches, per-frame outputs, CSVs, mp4 clips out of git; check full history before publishing.
- [ ] No frame-level derived exports published (RULES.md:106-111).
- [ ] LLM data clause (FAQ.md:109-112) — decision recorded by team.
- [ ] Every external dataset/model public and documented (RULES.md:7-10, 61-63).
- [ ] One command → byte-identical CSV from clean clone + data; `uv.lock`, Python 3.12, fixed seeds, deterministic threads.
- [ ] Validate exact uploaded file; `<team>.csv`, `<team>.pdf`; one form response.
- [ ] Team of 3–5 members (RULES.md:95-96).

## 7. Tech doc outline

1. Approach and rationale.
2. Assets used and dropped.
3. Handling missing/noisy tracking.
4. Possession rules implemented.
5. Training data (external only, sources, licences).
6. Validation protocol.
7. Results with CIs, single public sanity score, ablations.
8. Limitations and future work.
9. Reproduction: command, hardware, runtime, hash.

## 8. Timeline

| Dates | Work |
|---|---|
| Oct 25–27 | Phase 0, organiser questions, GT semantics study |
| Oct 28–Nov 3 | Phase 1, first real candidate Nov 3 |
| Nov 1–5 | External data ingestion + degradation simulator |
| Nov 3–9 | Phase 2 |
| Nov 6–11 | Phase 3 (Phase 4 only if staffed) |
| Nov 12 | Freeze method on external validation |
| Nov 13 | One public sanity score; real bug fixes only |
| Nov 13–15 | Tech doc |
| Nov 15 | Clean-machine reproduction, publish repo |
| Nov 16 | Final validate and upload |
| Nov 17 | Deadline |

## 9. Open questions and risks

**Organiser questions:** cloud coding assistant use; hard-coding published kickoff/graded frames; qualitative public GT study.

**Risks**
- External possession definitions differ from Metrica's → learn low-level events, keep rules explicit.
- Degradation simulator unrealistic → stronger label-free BAR-ATH checks.
- Transitions inside ball gaps dominate error → shape features, back-dating.
- Throw-ins without Smart Tagging team → geometry.
- ATD XML may expose extra ball channels (status/owner) → check locally.
- ML reproducibility, parser runtime → deterministic cache.
- Team-size eligibility.
