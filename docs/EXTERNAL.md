# External validation data

Open tracking data about other games, used to train and validate the learned method. Nothing here is used at
prediction time. Team decision (Oct 25 2026): external open data about other games is allowed, as
`RULES.md` (lines 12-13, 61 and 84) permits, consistent with `docs/COMPLIANCE.md`.

## Sources and licences (checked against the primary sources)

| Source | What | Licence | Where |
|---|---|---|---|
| DFL / Sportec (Bassek et al. 2025, Scientific Data) | 7 Bundesliga / 2. Bundesliga matches, 25 fps, 22 players + ball, event data | CC BY 4.0 (figshare record metadata) | https://doi.org/10.6084/m9.figshare.28196177 |
| SkillCorner open data | 20 A-League 2024/25 matches, broadcast tracking at 10 fps | MIT (`LICENSE` in the repository) | https://github.com/SkillCorner/opendata |
| Metrica Sports sample data | 3 anonymised matches, optical tracking at 25 fps, 105 x 68 m, event data. Same provider as our match's corrected tracking | No licence file. README, section "Legal stuff": "Please be responsible with the use of this data." and "If you use it for anything public, please aknowledge the source." (sic) | https://github.com/metrica-sports/sample-data (checked at commit `e706dd5`) |
| PFF FC FIFA World Cup 2022 | 64 matches, broadcast tracking (Sportlogiq, smoothed by PFF) at 29.97 fps, 105 x 68 m, synchronised game events | Free access through PFF FC's request form; the team obtained access through it on 2026-10-26. The terms are not reproduced in this repository. PFF FC is cited. PFF data was evaluated only: it is not in the final model, and no committed model was trained on it | https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset |

DFL and SkillCorner require attribution if results are published; Metrica asks for the same in its README (no formal licence, so use is by the repository's stated terms only); PFF FC must be cited. Downloads live in `data/external/<source>/`
(ignored by the parent repository), with figshare md5 and Git LFS sha256 verified. DFL, SkillCorner
and Metrica are parsed only with `kloppy` (pinned, optional `external` dependency group:
`uv sync --frozen --group external`); PFF is parsed by `possession.external.pff` (JSON lines, read
as data, see "PFF" below). Core `possession.run` imports neither.

Layout expected by `possession.external.sources` under `data/external/` (or `MTAH_EXTERNAL_DIR`;
default `<MTAH_DATA_DIR>/external`):

- `dfl/`: the figshare files side by side, `DFL_02_01_matchinformation_<id>.xml` and
  `DFL_04_03_positions_raw_observed_<id>.xml` for each match (a match needs both).
- `skillcorner/matches/<match id>/`: `<match id>_match.json` and
  `<match id>_tracking_extrapolated.jsonl`.
- `metrica/repo/data/Sample_Game_<n>/`: the sample-data repository, with either the EPTS files
  (`Sample_Game_<n>_metadata.xml`, `_tracking.txt`, `_events.json`) or the CSV files
  (`Sample_Game_<n>_RawTrackingData_Home_Team.csv`, `_Away_Team.csv`, `_RawEventsData.csv`).
- `pff/`: see "PFF" below.

Then `possession.external.prepare --source <source>` builds the harmonised and degraded caches.

## Harmonised match

`possession.external.match.ExternalMatch`: `fps` (25), `ball` (n,2) and `players` (n,k,2) in
normalised 0-1 pitch coordinates (kloppy default frame), `slot_team` (k,) with HOME=0 and AWAY=1,
`labels` (n,) in H/A/D/X, `pitch` in metres, `periods` (first and last frame per half, frames
between halves dropped). Home is H. Each match is rotated 180 degrees (x and y flipped) when needed
so that Home attacks the same way in the first half as in our BAR-ATH data.

Labels:

- DFL: `BallStatus` 0 gives D, otherwise `BallPossession` 1 gives H and 2 gives A.
- SkillCorner: the `possession.group` (home team or away team) gives H or A. Frames with no group
  (ball in flight, loose or not in play) are X, never D: the source has no dead-ball flag. For
  scoring, X is forward-filled inside each half with the last known team (the official scorer does
  the same to abstentions), so these frames follow the "loose ball keeps the last team" semantics.
  These labels are SkillCorner's, not a possession annotation in the challenge's sense: the
  possession player is the player nearest to the ball on 97 % of the frames with a located ball
  (10,169 of 10,487 in match 1886347, the one match checked), i.e. a nearest-player rule, and there
  is no ball status. Runs without a group mark stoppages and untracked phases (99 % of those frames
  lie in runs of 5 s or more), not passes in flight. The SkillCorner score is therefore a weak
  outside check of the team split under a change of tracker, not an independent possession
  reference.

- Metrica: built from the event files by `possession.external.metrica_events` (rules below). The
  tracking has no ball state or possession flag, so the loader replaces both from the events.
- PFF: from the game events carried by the tracking frames (`possession.external.pff`, rules in
  "PFF" below): `OUT` or `END` gives D until the next kick-off or on-the-ball event, otherwise the
  `home_ball` flag of the last such event gives H or A.

Players that appear in the tracking but not in the match metadata take the team of their kloppy
player record when it has one, otherwise their samples are dropped and counted in a warning.

SkillCorner is resampled from 10 to 25 fps per half: positions linearly (NaN if either neighbour is
missing), labels by nearest source frame.

## Metrica labels from events

Data: Sample_Game_1 and _2 (CSV tracking, home and away files, CSV events) and Sample_Game_3 (EPTS
tracking and metadata, JSON events), all 25 fps, normalised coordinates (kloppy `metrica` loaders),
no resampling. When a game folder holds both formats, the EPTS files are used (one format per game,
never both): they carry the team ids of the events and the player metadata. Event frames are tracking
frame numbers (checked: at every BALL OUT the ball is outside the pitch at the event's end frame, and
kick-offs have the ball at the centre). Event team Home/Away (JSON: by team id from the EPTS metadata)
gives H and A. Subtypes are split on `-` (CSV) or read as lists (JSON). Rules, stored as data
(`STOP_RULES`, `PLAY_TYPES`):

- Play events: PASS, CARRY, SHOT, SET PIECE, RECOVERY. At a play event's start frame the possession
  team becomes the event's team. BALL LOST, CHALLENGE, BALL OUT, FAULT RECEIVED and CARD that are not
  effective stops never change the team: a lost ball stays with the loser until the opponent's next
  play event (loose and contested phases keep the last team, as in DFL), a challenge counts only
  through the control event that follows it.
- Events are processed sorted by (start frame, stop candidates first). A stop that starts on the same
  frame as a recovery therefore wins, and the recovery does not revive play.
- Stop rules are of two kinds. Unconditional (the ball is verified out of play or the referee has
  whistled): BALL OUT (any), FAULT RECEIVED (any), SHOT with GOAL or OUT, BALL LOST with END HALF.
  Conditional (referee judgement that can be played on, e.g. advantage or a wrongly flagged offside):
  PASS or SHOT with OFFSIDE, CHALLENGE with FAULT, BALL LOST with FORCED, VOLUNTARY, OFFSIDE or HAND
  BALL. A conditional stop is effective only if the next play event that starts at or after the stop's
  end frame is a SET PIECE (the free kick, penalty or restart that confirms the whistle). If play
  continues by any other play event, the stop is ignored.
- Dead starts at an effective stop's end frame (`max(start, end)`; for BALL OUT this is the frame the
  ball crosses the line). The team that performs an effective stop takes possession from the stop's
  start frame (a clearance out, a forced loss), so the frames between the touch and the dead start
  belong to it even when the opponent held the ball before.
- Generic restart rule: a SET PIECE reached while play is alive (no effective stop before it) means
  the ball was dead before it, so D starts at the end frame of the last event that started before the
  set piece, and the set piece restarts play. If that end frame is not before the set piece starts
  (a pass still in flight), there is no D. This covers throw-ins after missed BALL OUT events, free
  kicks after challenges and cards, retaken set pieces.
- Dead ends at the first play event whose start frame is later than the dead frame (a SET PIECE, a
  goal-kick PASS, or a RECOVERY that restarts from a dead ball). Events that do not restart play are
  ignored while dead. A further effective stop while dead keeps it dead and extends the dead frame to
  its own end frame; a play event that is itself a stop (SHOT GOAL or OUT, SHOT OFFSIDE with a free
  kick) never revives play.
- Each period starts dead and ends at its first play event (the kick-off SET PIECE); a half-ending
  stop stays dead to the end of the period.
- CARD is not a stop; stoppages with no event (goalkeeper holding, injury treatment without a stop
  event) are not dead, so D is under-represented compared with DFL. The D labels are a heuristic
  (`dead_labels_heuristic` in `SourceInfo`), so validation and analysis score Metrica both raw (D as
  ground truth) and fair (D treated as continuation of the last team in both the prediction and the
  reference).

All distinct (type, subtype) combinations in the three games (events within the 3 games; "next" is
the next play event):

| combination (count) | decision |
|---|---|
| BALL OUT, any subtype (about 163 events: none, CLEARANCE, HEAD, CROSS, DEEP BALL, GOAL KICK, GOAL-WOODWORK) | stop, unconditional (ball verified out) |
| FAULT RECEIVED (54) | stop, unconditional (next is a FREE KICK in 49 of 54) |
| SHOT GOAL (10), OUT (29), HEAD-OFF TARGET-OUT, OUT-WOODWORK | stop, unconditional (kick-off or goal-kick restart follows) |
| SHOT OFFSIDE | stop if a SET PIECE follows (no instance in these games) |
| SHOT SAVED, BLOCKED, WOODWORK, plain OFF TARGET | not a stop (opponent recovers or play continues) |
| PASS OFFSIDE (2) | conditional; one is followed by a SHOT by the same team, so ignored there |
| PASS GOAL KICK, HEAD, CROSS, DEEP BALL, THROUGH BALL, CLEARANCE, plain | not stops; GOAL KICK passes restart a dead ball |
| BALL LOST END HALF (4) | stop, unconditional |
| BALL LOST FORCED (17), OFFSIDE (4), HAND BALL (5), VOLUNTARY (0) | conditional (next is a FREE KICK, PENALTY or KICK OFF RETAKEN) |
| BALL LOST INTERCEPTION, THEFT, HEAD, CLEARANCE, CROSS, DEEP BALL, GOAL KICK, WOODWORK, REFEREE HIT, plain | not stops (opponent RECOVERY or same-team CARRY follows) |
| CHALLENGE with FAULT (AERIAL, GROUND, TACKLE; WON or LOST; 148) | conditional (next is a FREE KICK in nearly all) |
| CHALLENGE GROUND, TACKLE, AERIAL, DRIBBLE, ADVANTAGE (WON or LOST) | not stops |
| CARD YELLOW (11) | not a stop; the free kick that follows starts D by the generic restart rule |
| RECOVERY (plain, INTERCEPTION, THEFT, SAVED, BLOCKED, GOAL) | play events, never stops (a throw-in or corner after one is covered by the generic restart rule) |
| CARRY, PASS (plain) | play events |
| SET PIECE KICK OFF, FREE KICK (DIRECT, INDIRECT, RETAKEN), CORNER KICK, THROW IN, PENALTY | play events and restarts |

Label shares and spells over the graded frames (harmonised, no degradation of labels), against DFL
(D 41 to 52 %, 370 to 511 spells, median 3.0 to 4.7 s):

| match | H | A | D | spells | median |
|---|---|---|---|---|---|
| game1 | 30.1 % | 27.8 % | 42.1 % | 350 | 5.0 s |
| game2 | 32.8 % | 28.3 % | 38.9 % | 320 | 5.9 s |
| game3 | 34.7 % | 33.0 % | 32.3 % | 378 | 5.5 s |

Median dead phase 12.3 to 18.3 s, longest 160 s.

Metrica ball track. The ball of the sample data is event-interpolated, not tracked: it moves in
straight lines at constant speed between annotated touches and jumps onto the receiving player at
the event's start frame, the frame where the event rules above switch the team. On the harmonised
games 1 and 3 (game 2 not measured), 89 % of the alive frames with a ball have a ball second
difference below 1 cm, against 9 % on DFL J03WMX, and the ball is within 10 cm of the nearest
player on 27 % (game 1) and 16 % (game 3) of those frames, against 4 % on DFL J03WMX. The degradation adds
millimetres of jitter and does not remove this. Ball and touch features learned on Metrica are
therefore cleaner, and closer to the label switches, than any automatic tracker gives, and Metrica
held-out scores are optimistic for that reason too.

## PFF FC World Cup 2022

Read from "PFF FC Tracking Data Specification v2.2" (October 2023) and "PFF FC Event Data
Specification v2.5" (v2.4 identical for the fields used), shipped with the download.

- Layout. Google Drive split the download into `FIFA World Cup 2022`, `FIFA World Cup 2022 2` and
  `FIFA World Cup 2022 3` under `data/external/pff/`, each with `Tracking Data/<game>.jsonl.bz2`,
  `Metadata/<game>.json`, `Rosters/` and `Event Data/` (the zips the folders came from are ignored).
  `sources.pff_matches` takes `<game>.jsonl.bz2` and `<game>.json` from any of them (or from the
  root), merges byte-identical copies (sizes are compared first, then a SHA-256 cached per path,
  size and modification time, `sources.content_digest`) and raises on copies that differ. A game
  needs both files.
  Rosters and event files are not read. The download has 64 distinct games, no duplicate files.
- Metadata: `fps` 29.97 in all 64, pitch 105 x 68 m (`stadium.pitches`, exactly one required),
  `homeTeamStartLeft`, period start and end times in video seconds.
- Tracking: one JSON object per line and per video frame: `frameNum` (from the start of the video),
  `period` (1, 2; 3, 4 extra time, in 5 games; no shoot-out; frames outside a period are not in the
  file), `periodElapsedTime`, `periodGameClockTime`, raw and smoothed players and ball,
  `game_event_id`, `game_event`, `possession_event`. Frames with several possession events are
  repeated with the same `frameNum` (consecutive; the first is kept). Frames must otherwise
  increase; a missing frame number inside a period is a missing sample.
- Coordinates: metres, origin at the centre spot, x along the touchlines left to right and y along
  the goal lines bottom to top as seen by the main camera, z up. Converted to the kloppy frame as
  x / length + 0.5 and 0.5 - y / width (`tests/test_external_pff.py` checks this against
  `kloppy.pff`).
- Players: `homePlayersSmoothed` and `awayPlayersSmoothed` (PFF's Kalman-filtered tracks), each
  with `jerseyNum`, `confidence`, `visibility` (`VISIBLE` or `ESTIMATED`: off-screen players are
  estimated by Sportlogiq from context, about two thirds of the samples) and x, y. Kept as they
  are. One slot per (team, jersey number); a second sample with the same team and jersey in one
  frame is dropped and counted in a warning (900 to 1,300 of about 4 million samples per match in the matches checked).
  A sample with no jersey number is not merged with others: it gets its own slot per position in
  the frame's player list and is counted in a separate warning (`unnumbered_players`).
- Ball: `balls`, the broadcast-tracked ball (`visibility`, x, y, z), not `ballsSmoothed`. The spec
  says the smoothed ball is placed "at the location of the ball carrier with a linear
  interpolation between events", i.e. built from the same event annotations as the labels; using it
  would leak the labels into the ball features. The raw ball is missing outside main-camera
  segments (replays, close-ups): it is present on 65 to 67 % of the frames of the two smoke
  matches. Frames listing more than one located ball get no ball (counted in a warning; none in
  the smoke matches). z is not used (2D like the other sources).
- No per-frame ball state or possession flag. The game event of the frame (same fields in the
  tracking spec section 5 and the event spec section 2) carries `game_event_type` (`FIRSTKICKOFF`,
  `SECONDKICKOFF`, `THIRDKICKOFF`, `FOURTHKICKOFF`, `OTB` on the ball, `OUT` ball out of play,
  `END` end of half, `G` woodwork or flag and back in play, `SUB`, `ON`, `OFF`, `VID` video
  missing), `start_frame`, `end_frame`, `home_team` (team of the player on the ball) and
  `home_ball` (team of the possession sequence, "an uninterrupted possession by one team"). The
  event spec gives `outType` for `OUT`: home or away score, out of touch, or whistle, so whistle
  stoppages (fouls, offside, injuries) are `OUT` events too.

Labels (`pff.event_state`, the rule kloppy 3.19.1's PFF deserializer applies to its ball state and
owning team): events are ordered by `start_frame`. `OUT` and `END` make the ball dead from their start frame; a kick-off or `OTB` makes
it alive from its start frame, and its `home_ball` sets the team. When the restart after a dead spell
has a null `home_ball` the team is unknown (X) until the next event with a team; an on-the-ball event
with a null `home_ball` inside a live spell keeps the team. Events are read from every frame record,
once each: by `game_event_id`, or, when it is missing, by start frame, type, end frame and `home_ball`.
Other events change nothing. Frames before the first such event are dead. Labels are therefore H or A between
restarts, with loose balls and passes in flight keeping the last sequence team, and D from the ball
going out or the whistle to the restart touch, as in our ground truth (GT_SEMANTICS 1 to 3).
`home_ball` is used instead of `home_team` so that a touch by the defending team inside the
attacking team's sequence (a block, a deflection; 57 of 1,288 on-ball events in game 10502) does not
make a sub-second spell; the GT flips at the first touch of a new possession (GT_SEMANTICS 4), which
is where the sequence changes. Checked on game 3812 against the event file: 59 of the 72 out-of-touch
`OUT` events with a located ball have it outside the pitch within one second of the event frame.

The D labels come from annotated events and a rule, not from a per-frame provider flag, and their
start is set by an annotator on broadcast video, so `SourceInfo` marks them heuristic like
Metrica's: validation and analysis report PFF raw and fair.

Why not `kloppy.pff.load_tracking`: it reads the whole decompressed file (about 770 MB per match)
into memory and keeps every parsed frame of a period while building the periods, takes the
event-smoothed ball, and drops players whose jersey is not in the roster. `possession.external.pff`
streams the file (about 0.7 GB peak and 20 s per match) and is checked against kloppy's coordinates
on synthetic records.

Resampling from 29.97 to 25 fps per period as for SkillCorner (positions linearly, NaN if either
neighbour is missing; labels by nearest source frame).

Smoke check (2 matches, harmonised, all period frames; D phases are runs of D):

| match | H | A | D | spells | median spell | D phases | median D phase |
|---|---|---|---|---|---|---|---|
| 3812 | 26.3 % | 28.3 % | 45.4 % | 291 | 6.5 s | 111 | 18.7 s |
| 3814 | 25.7 % | 27.1 % | 47.2 % | 255 | 6.7 s | 92 | 22.2 s |

D share inside the DFL range (41 to 52 %), spells fewer and longer than DFL's (370 to 511, median 3.0
to 4.7 s) and close to Metrica's (320 to 378, median 5.0 to 5.9 s); dead phases a little longer
than Metrica's (median 12.3 to 18.3 s), longest 197 s. The PFF D convention may still differ from
DFL's, which is why `dfl+metrica+pff_ha` uses PFF for the team model only.

## Degradation

Only aggregates of our match (graded frames, from `cleaning.clean` with the estimated cleaning
parameters, see "Cleaning parameters" below) are used, and every parameter is derived from them in
`possession.external.stats.our_stats`:

- ball missing share, and the empirical gap-length set (gaps are drawn from it and placed longest
  first);
- gap placement odds: a logistic model (scikit-learn `LogisticRegression` defaults) of "ball
  missing" on the distance from the (interpolated) ball to the nearest player and its distance to
  the nearest goal line, fitted on our match and applied to the external match. Gaps already in the
  source (SkillCorner) count towards the target and are never filled;
- slot fragmentation: external players are cut into segments with lengths drawn from our slot
  active-run lengths, segments are kept with the probability that brings the mean count of active
  non-duplicate objects to ours, then assigned to a pool of slots with our per-team slot counts
  (least recently used first);
- duplicates: added after thinning, as same-team copies of a real player offset by a distance
  drawn from the duplicate offsets measured on our match (the same-team distances of the samples our
  cleaning merged), only in idle slot frames, with our run lengths, until duplicates are our share of
  all objects. Cleaning then merges them exactly as on our match;
- jitter: white noise with variance equal to our robust second-difference noise variance minus the
  external one, floored at zero (both measured with the same estimator).

Seeded per match from `--seed` and the match name, deterministic.

## Cleaning parameters

`possession.cleaning.clean` needs a `CleaningParams`. For the `learned` method every value is
estimated from data by `possession.external.bounds.estimate_params` (no hand-set number), cached as
`cache/external/params_<key>.npz` by `prepare`, and stored inside the trained model, so
`possession.run` uses exactly the values the model was trained with. The statistics, the values
they give on the current data and the reasoning are in `docs/PARAMETERS.md` ("Cleaning
parameters"). The estimation reads the harmonised DFL matches (raw optical tracking, never degraded,
alive frames only) and aggregates of our own match; the key covers both, so new DFL files or a code
change re-estimate them. The orientation of the harmonised matches now comes from the raw tracking
of our match (no cleaning involved), so harmonisation does not depend on cleaning.

## Commands

```bash
.venv/bin/python -m possession.external.prepare [--source dfl|skillcorner|metrica|pff] [--seed N] [--force] [--prune]
.venv/bin/python -m possession.external.validate --method rules [--variant degraded|harmonised]
.venv/bin/python -m possession.learned.train [--variants dfl,dfl+metrica,metrica|all|...] [--selection-score raw|fair] [--seed N] [--score-outside pff] [--dead-features all|players] [--adaptation] [--out models/learned/model.pkl]
.venv/bin/python -m possession.external.validate --method learned [--model models/learned/model.pkl]
```

`validate` also prints a `fair` row for sources whose labels contain no D (SkillCorner) or only
heuristic D (Metrica, PFF); DFL has D from the provider. In fair mode D is turned into X and forward-filled
with the last team inside each half in both the prediction and the reference, so a stoppage counts as
continuation instead of splitting a spell on either side. For SkillCorner the reference has no D, so
only the prediction changes (unchanged behaviour). For Metrica the heuristic D of the reference is
filled too; before this fix only the prediction was filled, which left every Metrica D frame and
every spell split by a heuristic D scored as an error. `--no-dead-as-fill` prints the raw rows only.
Raw and fair numbers are both reported; fair is the comparable one for SkillCorner and Metrica.
`analyse` likewise reports Metrica and PFF twice, as `<source> raw` and `<source> fair`.

```bash
.venv/bin/python -m possession.external.analyse --method learned [--source dfl|skillcorner|metrica|pff] [--limit N]
```

`analyse` prints aggregates per source over the degraded matches: frame confusion, spell matches at IoU
0.5 with the cause of each missed true spell (merged across D, merged across an opposite spell, merged other, overrun/boundary, wrong
team, predicted dead, fragmented), boundary timing for H>A, A>H, H>D, A>D, D>H and D>A (one-to-one
matching per kind through `pm.match_transitions`, window = median true spell length of the match,
buckets from `pm.TAUS` and that window), error rate with the ball
visible versus inside gaps (gap-length terciles over frames in gaps, equal frame mass, with bucket frame counts) and the share of missed transitions in
gaps, recall by true spell duration quartile, and D share and run counts. Everything except the frame
confusion and the D counts uses the same dead-as-fill conversion, on both sides, on sources without
trusted D labels.

The `learned` method is the only one trained on these matches (degraded). Its training set is one
of six variants (DFL; DFL and Metrica; Metrica by default; and, only when requested, DFL, Metrica
and PFF; DFL, Metrica and PFF for the team split only; DFL, Metrica and SkillCorner for the team
split only), selected by held-out scores on every DFL, Metrica and SkillCorner match (PFF is
scored and reported, not used to select). See `docs/PARAMETERS.md` ("Training-set variants and selection").

Variant selection override (2026-10-26). A five-variant run (`dfl`, `dfl+metrica`, `metrica`,
`dfl+metrica+pff`, `dfl+metrica+pff_ha`; report in `models/learned_variants/variant_report.json`) gave
Metrica raw spellF1 0.648, 0.686, 0.678, 0.614 and 0.651. The coded selection rule picked
`dfl+metrica+pff` (all variants tied on Metrica, tied on DFL, SkillCorner tie-break 0.313). The team overrode
it and chose `dfl+metrica`, because adding PFF lowered both the Metrica and the DFL means and Metrica is the
priority source. This is a manual decision against the coded rule, taken after seeing the held-out numbers.

Caches go to `cache/external/` (ignored; about 200 MB per match uncompressed, harmonised plus
degraded, so the 64 PFF matches need about 13 GB of disk). Keys cover the source files, the kloppy version, the
source of every module the result depends on, the estimated cleaning parameters and the reference
statistics; `--prune` deletes cache
files that do not match the current keys for `--seed`. `validate` loads only the variant it scores. Output is aggregates only.
