---
title: "Possession labelling for BAR-ATH: technical document"
subtitle: "Team Graciencs. More Than A Hack 2027. Final model: v4, `models/learned/` (pitch-line features added to v3); repository: tag `submission`"
---

# 1. Summary

The task is to label each of the 199,319 frames of the tactical video of FC Barcelona vs Athletic Club as `H`, `A`, `D` (dead ball) or `X`. The board is ranked on spellF1 (possessions matched by temporal IoU) and tie-broken on macroF1 (`EVALUATION.md`).

**Method in five lines.**

1. The provided automatic tracking (ATD) is cleaned with bounds estimated from data: DFL optical tracking gives the physical limits, label-free aggregates of our own match give the gap length and the noise level.
2. The models learn from other games only: 7 DFL and 3 Metrica matches with provider or event-derived labels, degraded to look like our tracking.
3. A factorised frame classifier gives P(D) from a gradient-boosting model on player kinematics only (including players beyond the pitch lines) with a context stage, and P(H | alive) from a team model on ball, touch and team-shape features.
4. A duration-explicit hidden semi-Markov model (HSMM) over H, A, D turns the frame posteriors into possessions: exact Viterbi, learned durations, a touch-based switch prior, boundary refinement.
5. At inference an unsupervised EM re-estimates the dead-ball share and the ball-observation evidence on our match. No label of the match is involved at any point.

**Headline held-out numbers** (final model, nested leave-one-match-out over external matches, mean over matches with the 5 to 95 bootstrap interval of `models/learned/variant_report.json`):

| Source (matches) | spellF1 [interval] | macroF1 |
|----------------------------------------|------------------------------|------------------------------|
| DFL, raw (7) | 0.587 [0.546, 0.627] | 0.881 |
| Metrica, raw (3) | 0.782 [0.744, 0.819] | 0.915 |
| Metrica, fair (3) | 0.771 [0.730, 0.811] | not comparable (see 6.1) |
| SkillCorner, fair (20, unseen) | 0.318 [0.282, 0.355] | not comparable |

**Public sanity score** (one look per earlier internal model, three in all for v1, v2 and v3, section 6.3; none was used to choose a model; the figures here are the v3 look of 2026-11-05, and no public score is reported for v4): macroF1 0.8842, spellF1 0.5943, 18 predicted possessions of mean 13.5 s against 17 true ones of mean 14.2 s. The public window holds 17 possessions and is easier than the match (the ball is visible there about 88 % of the time against 59 % on the graded match, `docs/GT_SEMANTICS.md`), so it is a pipeline check, not an estimate of the full-match score (`EVALUATION.md`). Nothing in the model was fitted, tuned or selected on it.

**Reproducibility.** One Docker command (section 10) regenerates the submission byte for byte: sha256 `234c1366efa2d4a95a23eb0e8c9944af5ee6e6520da5e289f6204a14e665297b`, identical on linux/arm64 Docker, emulated linux/amd64 Docker and native macOS arm64. The committed model is `models/learned/model.pkl`, sha256 `f249108eca1c98df04b1a70c88d3b5b70cb150dcc6e59a4d7783e508fbc4198f`. Inference needs only the two tracking files of the match and no network.

# 2. Problem and metrics

Every frame of the tactical video gets one label: `H` or `A` for the team in possession with the ball alive, `D` when the ball is not in play, `X` when the system abstains. 158,857 frames are graded, in two blocks (11,185 to 90,651 and 110,808 to 190,197); elsewhere `X` is free (`DATA.md`).

**spellF1 (ranking).** A possession is a maximal run of `H` or of `A`; dead ball is never a possession. Predicted and true possessions are matched one to one by temporal IoU and the F1 of the matching is averaged over IoU thresholds 0.3, 0.4, 0.5, 0.6 and 0.7. A constant guess scores exactly 0. Differences below about 0.04 are noise on the full match. **macroF1 (tie-break)** is the unweighted mean of the F1 of H, A and D over graded frames, with a noise band of about 0.01. Before matching, the scorer forward-fills `X` with the last label of the half, so an abstention means "repeat my previous answer".

**Why spell boundaries matter.** The metric counts possessions, not frames, so a possession that is split, merged or shifted is a miss at the stricter IoU thresholds. EVALUATION.md measures the cost of timing alone on a perfect labelling: 3 frames late gives 0.9818, 8 frames late 0.8933, one second late 0.6539. The baselines show the same thing: `baseline_raw` has macroF1 0.6532 and spellF1 0.0486, and the published floor `baseline_smooth50` has spellF1 0.2433 on the full match. A system must therefore get the number, the team and the boundaries of the possessions right, and dead ball must be detected because every stoppage ends a possession. The rest of this document is organised around these two requirements: where the dead ball is, and who owns the ball in each stretch of play.

**Terms.** A *spell* is a possession as the scorer counts it (a maximal run of `H` or of `A`); *D* is dead ball. *Raw* scoring takes a source's D labels as truth; *fair* scoring turns D into continuation of the last team in prediction and reference alike (6.1). *LOMO* is leave-one-match-out cross-validation on the external matches.

# 3. Data

## 3.1 The provided match

Only the tactical tracking (ATD, `BAR-ATH_tactical_atd.xml` and `.txt`) is read at prediction time: 199,319 records, 46 anonymous track slots, no player identities, the ball present on 56.9 % of all frames and on 94,214 of the 158,857 graded frames, so missing on about 41 % of them (`DATA.md`). Metadata facts are used as given: 25 fps, pitch 105.3 x 68.0 m, the graded blocks and the kickoff anchors of the official `games.py`.

- **Smart Tagging** is used only in the legacy rule-based method (not part of the submission) and in label-free diagnostics of our match (set-piece windows holding a predicted restart, agreement between restart team and tag team). No module on the path of the learned method imports it, and external matches have no equivalent. Experiments that fed tag-derived features to the team model lost and were dropped (section 7).
- **Video, audio, the homography and the broadcast tracking** are not inputs of any model and are not read by the submitted code (section 7 lists what was tried with them).
- **Labels of this match**: none are used. No hand label, correction or frame-specific constant exists.

## 3.2 External training and evaluation data

All external data is about other games, public or freely requestable, and documented in `docs/EXTERNAL.md`. No team member labelled any frame by hand. Labels are provider fields or are derived by our code from provider event files.

| Source | Amount | Labels, and who made them | Role | Licence and attribution |
|-----------------|-----------------|-------------------------------|--------------|---------------------|
| DFL / Sportec (Bassek et al. 2025, *Scientific Data*) | 7 Bundesliga and 2. Bundesliga matches, 25 fps, 22 players and ball, events; 669 min of period time (about 1.0 million frames), 3,061 true spells (370 to 511 per match) | the data provider: per-frame fields `BallStatus` 0 gives D, else `BallPossession` gives H or A | training and validation | CC BY 4.0 (Bassek et al. 2025; figshare 10.6084/m9.figshare.28196177), attribution required |
| Metrica Sports sample data | 3 anonymised matches, 25 fps, 105 x 68 m, events; 287 min (about 0.43 million frames), 1,048 true spells (320 to 378 per match). Same provider as the corrected tracking of our match | events annotated by Metrica Sports; frame labels derived from them by our `metrica_events.py` (rules below). D share 32.3 to 42.1 %; D is heuristic. The sample ball track is event-interpolated, not tracked (below) | training and validation (raw and fair scoring) | no licence file; README asks to be responsible and to acknowledge the source, which we do (commit `e706dd5`) |
| SkillCorner open data | 20 A-League 2024/25 matches, broadcast tracking at 10 fps, resampled to 25; 259 true spells per match on average (fair) | SkillCorner's `possession.group` gives H or A. Its possession player is the player nearest to the ball (97 % of located-ball frames in the one match checked), and there is no ball status, so frames without a group are `X` | evaluation only, never trained on in the final model: a weak outside check of the team split | MIT, attribution |
| PFF FC FIFA World Cup 2022 | 64 matches, broadcast tracking at 29.97 fps | events annotated by PFF FC; frame labels derived from them by our `pff.py` (`OUT`, `END`, kick-offs, `home_ball`) | evaluated, rejected, not in the final model | free access through PFF FC's request form, obtained on 2026-10-26; the terms are not reproduced in the repository; PFF FC is cited (section 11); no committed model was trained on it |

The Metrica sample ball moves in straight lines between annotated touches and jumps onto the receiving player at the event frame where our rules switch the team: on games 1 and 3, 89 % of alive frames with a ball have a ball second difference below 1 cm (DFL J03WMX: 9 %) and the ball is within 10 cm of the nearest player on 16 to 27 % of them (DFL J03WMX: 4 %). Ball and touch features are therefore cleaner on Metrica than on any automatic tracking, one more reason why Metrica cross-validation is optimistic (section 9).

**Possession rules carried by the labels.** The learned model has no hand-written possession rules; it inherits the conventions of the training labels:

| Situation | DFL (provider fields) | Metrica (our event rules) | Challenge definition and public window (`docs/GT_SEMANTICS.md`) |
|------------------|------------------|----------------------------|----------------------------|
| Change of possession | `BallPossession` flips at the new team's first touch (30 % of DFL spells are shorter than 2 s) | at the start frame of the new team's first play event (pass, carry, shot, set piece, recovery) | "controlled" possession; in the public window the label flips at the new team's first touch, and spells under 1 s occur |
| Loose or contested ball, deflection, challenge, card | keeps the last team | keeps the last team: a lost ball, a challenge or a card never changes the team; a challenge counts only through the control event that follows it | keeps the last team |
| Ball out of play | D from `BallStatus` 0 | D from the frame the ball crosses the line (end frame of the ball-out event) | D from the line crossing |
| Foul, offside, other whistle | D from `BallStatus` 0 | foul received: D from the stop; offside, forced loss, hand ball, challenge with foul: D only if the next play event is a set piece | D starts while the players decelerate |
| Goal, half end | D from `BallStatus` 0 | D (shot with goal or out, ball lost at the end of a half) | D |
| Restart | the provider's flags | D ends at the first play event after it (the restart touch), which belongs to the restarting team | D ends at the restart touch; the new spell belongs to the taker's team |
| Period start | the provider's flags | dead until the kick-off set piece | the second half is D until its kick-off (FAQ); on our match D is forced through the published kickoff anchors (4.7) |
| Stoppage with no event (goalkeeper holding, treatment) | the provider's flags | alive (no event marks it), which is why Metrica D is heuristic | not settled by the definition; our model may label such stretches D (section 9) |
| Dead-ball ownership | none: D | none: D | none: "when the ball is dead the answer is D" (CHALLENGE.md) |

The D convention (from line crossing or whistle to restart touch) matches `docs/GT_SEMANTICS.md`, written after the qualitative study of the public window.

External matches are rotated when needed so that Home attacks the same way in the first half as in our match, resampled to 25 fps where needed, and every training match is used twice, once from each team's perspective (features mirrored, H and A swapped), so the model has no home bias.

# 4. Method

```
 ATD tracking of our match                  external matches (other games)
 (positions, 46 slots, no identity)         DFL x7, Metrica x3: training + CV
          |                                 SkillCorner x20: evaluation only
          v                                           |
 cleaning (bounds estimated from data) <--------------+-- degradation to mimic our
          |                                               tracking (label-free stats)
          v
 frame features: 178, threshold-free, metres, attacking-direction frame
          |
          +--> dead-ball model P(D|x): 39 player-only features
          |        -> context stage (stage-1 logit over neighbouring frames)
          |        -> EM adaptation on the match (D share, ball evidence)
          |
          +--> team model P(H|x, alive): all 178 features incl. touch events
          |
          +--> switch prior: P(possession passes to team s near frame t | touches)
                    |
                    v
 posteriors P(s|x) = P(D|x), (1 - P(D|x)) P(H|x,alive), (1 - P(D|x)) P(A|x,alive)
                    |
                    v
 duration HSMM, exact Viterbi on 25-frame steps + boundary refinement
                    |
                    v
 H / A / D per graded frame, X elsewhere; D forced before the published kickoffs
```

## 4.1 Cleaning

`possession.cleaning.clean` works in metres. A ball sample is removed only when neither its past nor its future trajectory predicts it (an isolated spike); a fast but smooth flight, such as a high ball projected on the ground, is kept. A player sample is removed together with the whole speed excursion around any motion faster than the physical maximum. Ball gaps up to a maximum length are filled linearly when the straight line is physically possible. Same-team objects closer than any two real teammates come are merged. Identity switches are not repaired and no feature depends on a slot keeping its identity.

Every bound is estimated (values stored in `model.json`): ball maximum speed 35.719 m/s, ball deviation 1.068 m, player maximum speed 10.831 m/s, player deviation 0.410 m, duplicate distance 0.041 m, interpolation up to 24 frames, velocity half-span 1 frame for ball and players (derivations in `docs/PARAMETERS.md`, "Cleaning parameters"). On our match this interpolates 37 ball frames, removes 4,745 player samples (89 segments) and merges 39 duplicate samples; our ATD ball has no isolated spike.

## 4.2 Degradation of external matches

DFL is clean optical tracking with the ball almost always present; our ATD loses the ball on about 41 % of graded frames, fragments players into slots and duplicates some. Training on clean data would teach the model an observation process it never meets. `external/degrade.py` therefore corrupts each external match with statistics measured on the cleaned graded frames of our match, without labels (`external/stats.py`):

- the empirical set of ball-gap lengths (placed longest first) and the ball-missing share;
- gap placement odds, from a logistic regression (scikit-learn defaults) of "ball missing" on the distance to the nearest player and to the nearest goal line, fitted on our match;
- slot fragmentation with our slot active-run lengths and per-team slot counts, thinned to our mean object count;
- duplicate objects with offsets measured on the samples our cleaning merged, up to our duplicate share;
- jitter noise with the variance difference between our tracker and the source.

The degradation is seeded (seed 0) and deterministic. It places the ball gaps independently of the dead or alive state, which is its known weakness (section 9).

## 4.3 Frame features

178 features, computed by the same code on our match and on the external matches, in metres and in the attacking-direction frame of the team in the perspective (`docs/PARAMETERS.md`, "Features"): ball presence, position, speed, acceleration and signed distance to the goal and touch lines; ball-gap descriptors (time since and until the ball was seen, gap length, entry and exit geometry); nearest and second-nearest player distance per team; the signed run length of the team holding the nearest player; team shape (centroid and its velocity, defensive line, spreads, team speed, tracked players, closest player to the centre mark, mean player speed); past and future rolling means over the rolling scales; the pitch-line features (the furthest distance of a tracked player beyond the touch lines and beyond the goal lines, the speed of the player closest to or furthest beyond a line, and their past and future rolling means); and ball touch events. The rolling scales are the mean training run length of the possession states and of dead ball, 204 and 602 frames, plus two short scales of 45 and 101 frames for the team model. Touch candidates are local maxima of the ball impulse (the norm of the mean ball velocity over the 12 frames after a frame minus that over the 12 before it, 12 frames being the metric's tight tolerance), searched where the ball is present; each touch gets the team of the nearest player, and the team model sees the impulse, turn angle and nearest-player distance of the last and next touch, the time since and until each team's last touch, and per-team touch strength and count averaged over past and future windows at every rolling scale.

## 4.4 Dead-ball model

Dead-ball detection is where a model trained on other providers breaks on ours, so it is the most constrained part (`docs/DEADBALL.md`). The ball observation process during dead ball is provider-specific: DFL never loses the ball and freezes it while dead; Metrica drops it, so that "ball missing" is almost the label (P(missing | D) 0.92 to 0.99); on our match the ball is neither frozen nor always removed, and a model trained on all features read every visible moving ball as live play and predicted D on only 0.085 of the frames with the ball wholly over a line. Player kinematics are not shifted: player speeds on our match are reproduced by a mixture of the external dead and alive distributions with a residual of 0.01 to 0.03.

The dead-ball model is therefore a `HistGradientBoostingClassifier` on 39 player-only features (team speeds, centroid position and velocity, shape, spreads, defensive line, closest player to the centre mark, mean player speed, rolling means of shape and mean speed, and the pitch-line features of 4.3), with no ball and no object-count column. The pitch-line features were added in v4 because throw-ins are the dead balls the model missed most and the thrower stands outside the touch line in the tracking (6.4). A **context stage** (`--dead-context`) follows: a second dead-ball model on the same features plus the centred, past and future rolling means of the first-stage logit at the four rolling scales (12 columns). The first-stage logits used to train it are out of fold (groups of whole matches), so no context column is built by a model that saw the labels of its own frames.

**Unsupervised adaptation.** At prediction time the dead-ball logit is adapted to the match: it is the player logit, plus the prior correction of Saerens et al. (2002) with the target D share estimated as the mean posterior, plus the log likelihood ratio of the ball and object-count features estimated on the match itself by EM (a density-ratio classifier fitted on soft labels, cross-fitted between the two halves so no frame is scored by a model fitted on it). EM stops when the change of the D share is below its own standard error, with a run-time bound of 20 iterations. No label of the match is read.

## 4.5 Team model

`P(H | x, alive)` is a second `HistGradientBoostingClassifier` on all 178 features, trained on alive frames. Three choices are made on external cross-validation: **spell weights** (each row weighs 1 / the length of its labelled run, so every spell of a match counts once, as spellF1 counts it), **every-peak touches** (every impulse peak is a touch event, so the model weighs weak and strong touches itself instead of a cut-off) and **grouped early stopping** (the number of boosting iterations comes from early stopping on whole held-out training matches, never on random frames 40 ms apart; 124 iterations in the final model; the dead-ball model, its context stage and the switch prior keep the library default, 100 iterations with the library's random 10 % validation split; all three ran to the 100-iteration `max_iter` (`n_iter_` 100), so their early stopping never triggered). Every match has the same total weight in each model. The posteriors are P(D) = P(D | x) and P(H) = (1 - P(D | x)) P(H | x, alive), likewise for A, so they sum to one and a source without dead-ball labels can teach the team split without touching P(D).

## 4.6 Duration HSMM decoder

Frame posteriors flicker; possessions do not. An explicit-duration HSMM over H, A and D turns them into segments (`learned/hsmm.py`):

- **Emission** of state s at frame t is `w (log P(s | x_t) - log P(s))`, the scaled hybrid likelihood with the training frame shares as P(s). The weight is `w` = 0.1 (a span of 10 frames).
- **Durations** of each state are a Gaussian kernel density estimate of the log run length (Scott's rule) with a geometric tail, learned from the training labels; **transitions** (no self-transitions) and **initial states** are normalised counts. Each match weighs the same. The first segment of a block uses the residual-life distribution and the last one the survival function.
- **Switch prior**: a third gradient-boosting model estimates P(possession passes to team s within 12 frames of t | touch features), trained from the training fold's labels. Its log Bayes factor, times a weight of 1.0, is added to every direct H to A or A to H transition, never to a transition from or to D. Its touch threshold, 11.839, is the Youden-optimal cut on 16,209 labelled impulse peaks of the training matches.
- **Decoding** is exact Viterbi on steps of 25 frames, followed by a local search that moves each boundary to the frame, within one step, that maximises the emission evidence.

The emission weight, stride and switch weight (0.1, 25, 1.0) are chosen by nested leave-one-match-out over grids on 1-2-5 series (section 5).

## 4.7 Output

`possession.run` writes `X` outside the graded blocks. On our match it also writes D on every graded frame from each block start through the kickoff frame published in the official `games.py` (anchors 11,186 and 113,741; the FAQ states that the second half is graded from 110,808 but kicks off later, and all of those frames are D). The values come from official files, not from the public ground truth. We have not recorded an organiser answer on whether using the published kickoff anchors this way is accepted; `configs/learned.json` switches it (`force_prekickoff_dead`).

# 5. No hand-set constants

**Rule.** No hand-picked number may sit on the path that produces the submission or prepares its training data. Every number is (a) a metadata fact or part of the challenge definition, (b) a library default, (c) estimated from data, (d) chosen by nested cross-validation on external matches, or is a seed or a structural constant with no tuning freedom. No number is fitted, tuned or selected on labels of our match.

| Kind | Examples on the path of the final model |
|------------------|----------------------------------------------------------------------------------|
| (a) metadata | 25 fps; pitch 105.3 x 68.0 m; graded blocks; three states; touch window of 12 frames, the tight tolerance of the metric |
| (b) library default | scikit-learn 1.9.1 `HistGradientBoostingClassifier` settings (100 iterations for the dead-ball model), `LogisticRegression`, two-component Gaussian mixture, `gaussian_kde` with Scott's bandwidth |
| (c) from external data | cleaning bounds (median over the 7 DFL matches of the per-match extremes): 35.719 m/s, 1.068 m, 10.831 m/s, 0.410 m, 0.041 m; rolling scales 204 and 602 frames; durations, transitions and initial states; the switch-prior touch threshold 11.839 |
| (c) from label-free statistics of our match | interpolation limit 24 frames (boundary between the dropout and occlusion modes of a two-component mixture on our ball-gap lengths: 17 gaps of 1 to 25 frames, none of 26 to 75, 133 of 76 or more); velocity half-spans (noise of our tracker); all degradation statistics |
| (d) nested LOMO | emission weight 0.1, stride 25, switch weight 1.0; the 124 team-model iterations (grouped early stopping); the training set `dfl+metrica` (held-out evaluation, with a team override disclosed in section 8) |
| seeds, structure | seed 0, one thread (determinism); equal weight per match; EM stopping at the standard error of the D share; run-time guards |

Candidate grids (emission spans 1, 2, 5, 10, 25, 50, 100 frames; strides 5, 10, 25; switch weights 0, 0.5, 1, 2) are 1-2-5 series that extend upward while their top is selected; the choice inside them is by cross-validation, never by hand.

**Audit.** `docs/PARAMETERS.md` ends with a table of every numeric literal in the modules of the learned path (`cleaning.py`, `features.py`, `learned/`, `external/`), each classified. Twelve distinct hand-set constants (14 uses) were found on the path and were replaced by estimated values or library defaults, or removed: the ball and player speed bounds, the duplicate distance (in cleaning and in the degradation), the interpolation gap, the velocity half-span, the iteration cap of the gap-placement fit, the quartiles of the core rolling scale, and the 5 m crowd radius of the degradation; the off-pitch margin, the 5 m pressure count and the 90th speed percentile were unused and deleted; the 1 s shape smoothing and the 1 m halfway margin moved to the legacy method.

**One hand-set pair remains and is disclosed.** `SHORT_QUANTILES = (0.25, 0.5)`: the levels of the weighted quantiles of the training run lengths that define the two short rolling scales (45 and 101 frames) of the team model. The run lengths are learned, the levels are not, and they were not chosen by cross-validation. The dead-ball model does not use them.

The Phase 1 rule-based method (`method rules`) has hand-set values argued from football and physics. It is kept as a legacy reference, is not on the path of the submission, and its public sanity score was macroF1 0.8449, spellF1 0.5444, 19 predicted possessions.

# 6. Validation

## 6.1 Protocol

Leave-one-match-out over the 10 external matches with dead-ball labels (7 DFL and 3 Metrica): each fold model is trained on the other 9 matches (both classifiers, the switch prior, durations, transitions, scales, adaptation settings) and scores the degraded held-out match. The decoder setting of a fold is chosen nested, on the other folds only. The final model is trained on all 10. Scores use the vendored official scorer, with 1000 bootstrap draws and 5 to 95 percentile intervals; the interval of a source is the mean of the per-match bounds, which treats the matches as fully correlated and is the widest. Differences below about 0.04 should be read as noise (`EVALUATION.md`).

**Raw and fair.** DFL D is provider-labelled, so DFL is scored raw. Metrica D is heuristic and SkillCorner has none, so they are also scored fair: D is turned into `X` and forward-filled with the last team inside each half, in prediction and reference, so a stoppage no longer splits a spell and only the team assignment and the boundaries are scored. Fair mode has no D frames, so its macroF1 (0.586 on Metrica) is not comparable with raw macroF1 and is not used. **SkillCorner** (20 A-League matches, broadcast tracking, H/A only, possession boundaries about 60 frames away from DFL's) is scored by the final model, which never saw it: a stress test of provider shift. Its labels are a nearest-player rule (the possession player is the player nearest to the ball on 97 % of the frames with a located ball, in the one match checked) with no ball status, so the SkillCorner score is a weak outside check of the team split, not an independent possession reference.

## 6.2 Results of the final model

| Match | spellF1 raw [5-95] | macroF1 raw | Spells pred / true | spellF1 fair [5-95] |
|----------------|-------------------------|----------|--------------|-------------------------|
| dfl/J03WMX | 0.519 [0.481, 0.557] | 0.883 | 370 / 511 | |
| dfl/J03WN1 | 0.562 [0.520, 0.600] | 0.861 | 330 / 500 | |
| dfl/J03WOH | 0.600 [0.556, 0.642] | 0.877 | 283 / 402 | |
| dfl/J03WOY | 0.593 [0.555, 0.632] | 0.879 | 369 / 456 | |
| dfl/J03WPY | 0.569 [0.526, 0.612] | 0.881 | 298 / 421 | |
| dfl/J03WQQ | 0.617 [0.576, 0.654] | 0.887 | 321 / 401 | |
| dfl/J03WR9 | 0.651 [0.607, 0.695] | 0.902 | 276 / 370 | |
| metrica/game1 | 0.788 [0.746, 0.828] | 0.921 | 305 / 350 | 0.776 [0.730, 0.817] |
| metrica/game2 | 0.813 [0.778, 0.847] | 0.923 | 311 / 320 | 0.780 [0.743, 0.819] |
| metrica/game3 | 0.745 [0.708, 0.782] | 0.902 | 339 / 378 | 0.758 [0.717, 0.797] |

Source means are in section 1. SkillCorner (20 matches, final model): fair spellF1 0.318 (per-match range 0.252 to 0.383), raw 0.257, 318.6 predicted against 259.0 true spells on average. The model predicts fewer possessions than the references on DFL and Metrica (321.0 against 437.3 and 318.3 against 349.3 on average): it merges some short spells.

## 6.3 Against the earlier internal models

Mean held-out spellF1 over the same matches; v1 (touch features and switch prior), v2 (spell weights, every-peak touches, grouped early stopping) and v3 (dead-ball context) are the earlier internal models, v4 the final model (pitch-line features). "n/r" means not recorded.

| Held-out spellF1 | v1 | v2 | v3 | v4 (final) |
|----------------------------------|----------------------|----------------------|----------------------|----------------------|
| DFL raw (7) | 0.534 | 0.573 | 0.577 | 0.587 |
| Metrica raw (3) | 0.761 [0.721, 0.799] | 0.790 [0.754, 0.825] | 0.781 [0.745, 0.816] | 0.782 [0.744, 0.819] |
| Metrica fair (3) | 0.730 | 0.773 | 0.775 | 0.771 |
| SkillCorner fair (20) | 0.300 | 0.302 | 0.304 | 0.318 |
| SkillCorner raw (20) | n/r | 0.246 | 0.251 | 0.257 |
| Public sanity, macroF1 / spellF1 / predicted spells | 0.8959 / 0.5481 / 10 | 0.9027 / 0.5943 / 18 | 0.8842 / 0.5943 / 18 | n/r |

v1 to v2 is the larger step on DFL and Metrica (better on 10 of 10 nested folds) and about flat on SkillCorner (12 of 20 matches improve). v2 to v3 is small and mixed: better on 7 of 7 DFL matches and on 14 of 20 SkillCorner matches (fair); on Metrica the mean is up 0.002 (fair) and down 0.009 (raw), each better in 1 of 3 matches. All of it is inside the noise band. v3 was adopted under the rule of 2026-11-03 (external held-out cross-validation only) because DFL and SkillCorner improve on the large majority of their matches and the Metrica raw loss is inside the spread of 3 matches.

v3 to v4 adds the pitch-line features (`--line-features lines`) and nothing else. DFL is better on 6 of 7 matches (+0.010; the one loss is J03WOH, -0.015) and its macroF1 goes from 0.867 to 0.881; SkillCorner is better on 15 of 20 matches fair and 12 of 20 raw; Metrica fair is down 0.004 and better in 1 of 3 matches (game1 -0.012, game2 -0.0008, game3 +0.001), Metrica raw up 0.001 (2 of 3). The rule fixed before the full run (DFL better on at least 5 of 7 folds, Metrica fair not worse on at least 2 of 3) is met on DFL and missed narrowly on Metrica, game2 being within one spell of a tie. v4 was adopted because DFL, whose provider labels switch at the new team's first touch, and SkillCorner, an independent provider whose broadcast tracking often misses the ball, resemble the challenge labels and the ball visibility of our match more than the Metrica sample, whose labels come from our own event rules and whose ball track is event-interpolated (3.2, 9). All of it is inside the noise band. The selected decoder smooths more (emission weight 0.1, stride 25, switch weight 1.0, from 0.2, 10, 0.5): every DFL fold chose stride 25 and kept it when the stride grid extended itself to 50, and predicted DFL spells per match fell from 330.9 to 321.0 against 437.3 true.

## 6.4 Evidence for the dead-ball design

Quick experiments (every fifth training frame, earlier models) compared the dead-ball variants. Cross-source transfer, the only labelled measurement of robustness to a change of tracking provider, favoured player-only features with adaptation: DFL to Metrica raw spellF1 / D recall 0.644 / 0.888 (all features), 0.593 / 0.901 (players), 0.641 / 0.948 (players plus adaptation); Metrica to DFL 0.385 / 0.611, 0.452 / 0.843, 0.472 / 0.866. In the full run on our match (label-free), the share of the 3,832 graded frames with the ball wholly over a line that are predicted D rose from 0.085 (all features) to 0.695 (players plus adaptation), at a cost on DFL (0.571 to 0.512 spellF1) and a tie on Metrica raw (0.686 against 0.663, intervals overlapping). The team accepted the DFL cost (section 8).

**Error analysis.** An oracle test at the v1 model replaced the team split of the emission by the true team and raised held-out spellF1 by 0.16 (Metrica fair) and 0.28 (DFL), while refitted durations, other strides or sharper emissions gained 0.01 at most: the loss sits in the team model, mostly in true spells of 1 to 3 s, which motivated the team-model options of section 4.5. Dead-ball recall by stoppage type (v2, held-out, DFL / Metrica): throw-in 0.81 / 0.79, goal kick 0.90 / 0.94, corner 0.97 / 0.99, free kick 0.94 / 0.94, kick-off 0.97 / 1.00. Throw-ins are 53 % (DFL) and 57 % (Metrica) of the missed dead-ball seconds. The pitch-line features of v4 target them: in the quick run on top of v2 they raised throw-in recall from 0.82 to 0.89 (DFL) and from 0.79 to 0.88 (Metrica).

# 7. What did not work

Each item was compared with the then-current model on external held-out cross-validation and rejected; quick runs use every fifth frame (`docs/DEADBALL.md`, "Experiments since v2 that did not win" and earlier sections).

| Idea | Result | Outcome |
|-------------------------|--------------------------------------------------|-------------------------|
| Phase 1 hand-set rules | public sanity macroF1 0.8449, spellF1 0.5444; many hand-set constants | superseded by the learned method |
| Dead-ball model on all features (baseline) | cannot read dead ball on our match: 0.085 of over-the-line frames predicted D | replaced by players-only plus adaptation (6.4) |
| Viterbi-training EM for the adaptation | no fixed point in 10 iterations; D share drifted down (our match 0.47 to 0.45) | dropped |
| PFF data (`dfl+metrica+pff` and `pff_ha`) | Metrica raw spellF1 0.614 and 0.651 against 0.686 for `dfl+metrica` (first run); team-only retrain against v2, best common decoder setting: DFL 0.559 against 0.573, Metrica fair 0.749 against 0.779; lost on 10 of 10 folds at v2's settings | rejected; the team overrode the coded rule that had picked `dfl+metrica+pff` |
| Pitch-line features (player beyond a line), first full run, on top of v2 | DFL 0.576 against 0.573; Metrica fair 0.757 against 0.773, raw 0.774 against 0.790; SkillCorner fair 0.313 against 0.302 | Metrica lost, not adopted then (throw-in recall had risen from 0.82 to 0.89 on DFL in the quick run); adopted on top of v3 as v4 (6.3) |
| Goalkeeper-proxy and ball-line feature groups | no gain in the quick run (Metrica fair -0.015 for the keeper group) | dropped; the goalkeeper-holding pattern is not fixed |
| Smart Tagging features simulated on external matches | DFL 0.560 to 0.553 and 0.587 to 0.579, Metrica fair 0.769 to 0.765; tags lag the play by about 6 s | dropped |
| Per-player possessor model stacked into the team model | DFL 0.5605 to 0.5607, Metrica fair 0.7454 to 0.7485 | dropped |
| Team auto-context (second team model on out-of-fold logits) | DFL 0.5745 to 0.5775, Metrica fair 0.7626 to 0.7701, 3.2 times the training cost | only the dead-ball stage kept |
| Grouped early stopping for the dead-ball model too | Metrica fair -0.009, DFL -0.001 against team-only | not used |
| Gap-exit re-anchoring of decoded switches | Metrica fair -0.006 and -0.005, DFL -0.004 and -0.002 | not implemented |
| Decoder variants (no refinement, duration bandwidth x0.5 and x2, stride 1 or 2) | DFL 0.5701 to 0.5728 and Metrica fair 0.7705 to 0.7792 against 0.5725 and 0.7732 for v2; only stride 1 is clearly worse on Metrica fair (0.7560) | flat surface, no change |
| Ball imputation from the broadcast video (with the provided homography) | headroom from the external oracle scaled to our gaps: about +0.02 (alive gaps) to +0.04 (all gaps); costly in time and CPU | within the noise band, not pursued |
| Ball detection on the tactical video (feasibility only) | pretrained detectors found mostly stationary balls; heavy CPU run time inside Docker; licence of the candidate weights unclear | dropped |
| Whistle detection from audio | the tactical video's audio track is essentially silent; the broadcast audio has commentary and replays; no external audio to train on | not pursued |
| Realistic ball-loss simulation in the degradation | would need a hand-made model of a visible dead ball | not implemented |

# 8. Compliance and process

**No hand labels, external data only for training and tuning.** Every submitted label is written by `python -m possession.run`. No label, correction or constant of the provided game was made by hand. We made no hand labels on any game. External data is about other games only (section 3.2), is public or freely requestable, and nothing is fetched at run time; the committed model reproduces the submission with no external data. The match data, caches and per-frame outputs are excluded from git, and diagnostics print aggregates only.

The v1 to v2 public change (0.548 to 0.594, 17 spells) is within noise and was not used to choose v2.

**Team decisions, disclosed.**

- On 2026-10-26, after quick experiments put the players-plus-adaptation candidate below the DFL guardrail, the team amended its own decision rule to rank by held-out Metrica raw spellF1, the source closest to our provider, with ties broken by label-free diagnostics of our match, and accepted the DFL loss.
- On 2026-10-26 the team overrode the coded variant-selection rule, which had picked `dfl+metrica+pff`, and chose `dfl+metrica`: a manual decision against the code, taken after seeing held-out numbers.
- Until 2026-11-03 label-free diagnostics of our match (Smart Tagging set-piece windows, restart team, phase-team agreement) served as tie-breaks between candidates tied on external cross-validation. From 2026-11-03 they are an alarm only, never a tie-break.
- On 2026-11-13 the team adopted v4 although Metrica fair was not worse in only 1 of 3 folds, against the 2 of 3 required by the rule fixed before the full run (game2 within one spell of a tie). It gave more weight to DFL and SkillCorner, whose labels and ball visibility resemble our match more than the Metrica sample's (6.3). This weighs the sources differently from the 2026-10-26 rule, which made Metrica the priority source.

**Other.** One minor cross-validation leak is documented: the cleaning parameters are computed on all 7 DFL matches including the held-out one (label-free; it flatters DFL cross-validation only).

# 9. Limitations and expected behaviour on the full match

- **The ball is missing on about 41 % of graded frames, mostly during stoppages.** Under the rule-method proxy (no labels) P(ball missing | dead) is 0.71 against 0.16 when alive. On our match (label-free, recorded after v2) 3.4 % of graded frames are predicted alive with the ball missing and 3.5 % of the predicted H/A switches fall in a gap (degraded DFL: 18.6 % and 22 %; degraded Metrica: 4.3 % and 19 %). Scaling the external oracle gains by those shares, perfect labels in alive gaps would be worth about +0.02 spellF1 and in all gaps about +0.04, both inside the noise band.
- **Cross-validation on Metrica is optimistic.** Its labels and conventions are ours (derived by `metrica_events.py`), the gain of the switch prior there is partly circular on the timing of switches, the sample ball track is event-interpolated and lands on the receiving player at the switch frame (3.2), D is heuristic and under-represented, and fair mode ignores D altogether. DFL cross-validation is mildly flattered by the leak above. The degradation simulates ball loss independently of the game state, which real ATD does not do. In the nested run of v4 the adaptation hit its 20-iteration bound on none of the 10 held-out matches and on 1 of the 20 SkillCorner matches (v3: 2 of the 10, both DFL).
- **The public window is easier than the match** (ball visible about 88 % against 59 %), contains 17 possessions, and one missed possession moves spellF1 by about six points there. The three public scores of the earlier internal models (0.548, 0.594, 0.594) are sanity checks. We have no labelled estimate for the full match; the external means on the two sources with dead-ball labels span 0.587 (DFL) to 0.782 (Metrica raw), and SkillCorner shows 0.318 under a strong provider shift.
- **Remaining error types.** Throw-ins are the main missed dead balls (recall 0.81 and 0.79 at v2; the pitch-line features of v4 target them, 6.4). No tested feature distinguishes a goalkeeper holding the ball from a stoppage, so such stretches may be labelled D. Short contested spells of 1 to 3 s are merged (fewer predicted than true possessions on external matches). A team has more than 11 objects on 12.5 % of live frames (a player carried by the wrong team) and no data-driven rule can fix it without appearance data. Identities are not tracked.
- **Label-free diagnostics of our match** (alarm only, recorded for v3): 325 possession spells, 96 dead-ball runs, 49 of 59 Smart Tagging set-piece windows holding a predicted restart, restart team equal to the tagged team in 18 of 23, phase-team agreement 75.7 %.
- **What we would do with more time** (documented as not done): a dedicated goalkeeper-holding feature measured on the DFL keeper-catch events; team swaps of tracked objects in the degradation; controlled versus loose sub-states; EM refinement of the durations on the match itself.

# 10. Reproducibility

**Docker (canonical submission).** From a clean clone of tag `submission` (the head of the repository), with the match data downloaded as in `DATA.md`:

```
scripts/reproduce_docker.sh /path/to/data outputs
```

The script builds the image (`python:3.12-slim` and `uv` 0.12.0, both pinned by digest; dependencies from `uv.lock`: numpy 2.4.4, scipy 1.18.1, scikit-learn 1.9.1), mounts only `tactical/BAR-ATH_tactical_atd.xml` and `.txt` read-only, runs `python -m possession.run --config configs/learned.json`, writes `outputs/learned_final_docker.csv` and prints its sha256. The vendored official validator accepts the file.

| Item | Value |
|--------------------------------------|--------------------------------------------------------------|
| Submission sha256 | `234c1366efa2d4a95a23eb0e8c9944af5ee6e6520da5e289f6204a14e665297b` |
| Model `models/learned/model.pkl` sha256 | `f249108eca1c98df04b1a70c88d3b5b70cb150dcc6e59a4d7783e508fbc4198f` |
| Model run id | `98ffe80dc9f2c861b3249a9558b507abf6d0b75ed82e23fdf6ca0c2cf066cdd2` |
| linux/arm64 Docker (native under OrbStack, 2026-11-15; clean export of the published tree) | 194 s, same sha256 |
| linux/amd64 Docker (emulated under OrbStack, 2026-11-15) | 242 s, same sha256 |
| native macOS arm64 (`uv sync --frozen`, 2026-11-13, tracking cache already built) | 166 s, same sha256 |

Hardware of these runs and of the training run below: Apple M4 Max (14 CPU cores: 10 performance, 4 efficiency), 36 GB of memory, macOS; Docker under OrbStack. The uploaded `Graciencs.csv` is a byte copy of the linux/arm64 Docker file (same sha256), and the vendored validator accepts that exact file. Superseded history: the earlier internal model v3 gave Docker sha256 `265068c905fd97523cd7184cadf282c11056e2bd286346e4344e2512afceacaa` on linux/arm64 (719 s, cold cache) and on emulated linux/amd64 (914 s), and its native macOS arm64 run (559 s) matched it; that file is not the submission.

The earlier internal models v1 and v2 did not match natively on macOS (163 and 210 rows differed, at spell boundaries, cause inferred from the two BLAS builds and not isolated), so native equality is not assumed in general and only the Docker file is the submission. The model loader refuses a model whose sha256 or scikit-learn version differs from `model.json`. The tracking cache is keyed by the sha256 of the source files and written with fixed timestamps, so cold and warm runs give the same bytes.

**Without Docker:** `uv sync --frozen` (or, without `uv`, `python3.12 -m venv .venv`, `.venv/bin/pip install -r requirements.txt` and `.venv/bin/pip install -e .`) then `MTAH_DATA_DIR=/path/to/data .venv/bin/python -m possession.run --config configs/learned.json --out team.csv` (Python 3.12).

**Training** (not needed to reproduce the file; the model is committed). After downloading the external data and running `possession.external.prepare`, `scripts/phase_final.txt` holds the command of the final model, run through `scripts/run_logged.sh`:

```
python -m possession.learned.train --variants dfl+metrica --dead-features players \
  --adaptation --team-features touch --spell-weights --every-peak --stopping team \
  --dead-context --line-features lines --out models/learned_v4/model.pkl
```

Its output, copied to `models/learned/`, is the committed model (same sha256 as `models/learned/model.pkl`). The nested run took 20,139 s. Training is single-threaded and seeded: a rerun of the v3 training with the same inputs and wheels gave a model identical in every tree and array, but its pickle bytes, and so its sha256 and run id, differed. `models/learned/variant_report.json` holds the aggregates behind section 6; no match data is in the repository.

# 11. References

- Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data*, 12, 195. https://doi.org/10.1038/s41597-025-04505-y. Data © DFL Deutsche Fußball Liga, CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), https://doi.org/10.6084/m9.figshare.28196177. Not redistributed; we trained on a degraded copy (ball gaps, player thinning, duplicates, jitter) and publish only model weights and aggregate scores.
- Metrica Sports. Sample data (Sample_Game_1 to 3), https://github.com/metrica-sports/sample-data (checked at commit `e706dd5`). Used for training and validation and acknowledged as the source, as its README asks ("If you use it for anything public, please acknowledge the source"); no licence file is provided. Not redistributed.
- SkillCorner. Open data, © 2020 SkillCorner, MIT licence, https://github.com/SkillCorner/opendata. Evaluation only, never trained on; credited as its README asks.
- PFF FC. FIFA World Cup 2022 enhanced dataset, https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset. Access through PFF FC's request form, obtained on 2026-10-26; the terms are not reproduced in the repository. Evaluated only, not used in the final model. Not redistributed.
- kloppy 3.19.1, standardised reader of tracking and event data (DFL, SkillCorner, Metrica), BSD 3-Clause licence, https://github.com/PySport/kloppy. Used for external data preparation only.
- scikit-learn 1.9.1, `HistGradientBoostingClassifier` and `LogisticRegression`; SciPy 1.18.1, `gaussian_kde`.
- Saerens, M., Latinne, P., Decaestecker, C. (2002). Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. *Neural Computation*, 14(1), 21–41. https://doi.org/10.1162/089976602753284446
- Metrica Sports, Barça Innovation Hub (2026). More than a Hack 2027: Frame-level Ball Possession, version 1.0.0, https://github.com/metrica-sports/more-than-a-hack-2027, MIT licence (as its `CITATION.cff` asks). Challenge material used: `RULES.md`, `EVALUATION.md`, `FAQ.md`, `DATA.md`, `CHALLENGE.md`; `evaluation/possession_metrics.py`, the validator and `games.py` are vendored unmodified under `src/possession/vendor/`.
