# Parameters

Project rule: no hand-picked number anywhere in the code path that produces the `learned` submission
or prepares its training data. Every number there is a metadata fact, a library default, a value
learned or estimated from data (our match or external data), a value chosen by cross-validation on
external data, a seed, or a structural/mathematical constant. The audit at the end of this file lists
every numeric literal on that path.

The Phase 1 `rules` method keeps its hand-set values as a **legacy reference only**. They live in
`possession.params` (`RulesParams` and the `LEGACY_*` constants, including the legacy cleaning
values) and are read only by `possession.rules` and by `external.predict` for `--method rules`. They
never reach the shared cleaning code, the `learned` features, the degradation of external data or
the `learned` model.

# LEGACY: rules parameters (method `rules`, Phase 1)

Legacy reference. Hand-set values, kept so the Phase 1 method can still be run and compared. Not
used by `learned`.

All values live in `possession.params.RulesParams`. `configs/rules.json` uses the defaults
(`"params": {}`); any field can be overridden there. No value was fitted to, tuned on or chosen with
the public ground truth. Each one comes from football or physics common sense, stated below.
Choices between variants used only label-free diagnostics on the whole match (see the end of this
file).

## Dead ball

| Parameter | Value | Rationale |
|---|---|---|
| `speed_window_s` | 1.0 s | Smooths per-frame tracker jitter in the mean player speed. One second is shorter than any stoppage, so it barely moves boundaries. |
| `calm_speed_mps` | 1.5 m/s | Typical human walking pace (about 1.4 m/s). During a stoppage the 20-odd players walk or stand. During live play the mean is higher: players cover about 10–11 km in 90 min, about 2 m/s including stoppages. |
| `speed_scale_mps` | 0.5 m/s | How strongly speed counts as evidence: one walking-pace step (0.5 m/s) is one unit of evidence. |
| `speed_evidence_cap` | 3 | Stops one extreme frame (everyone standing, or one sprint) from outweighing several seconds of other evidence. Frames with no tracked players count as neutral evidence (zero), never as dead. |
| `active_ball_mps` | 5 m/s | Slowest typical pass or shot. A ball moving faster than this is being played, which is evidence of live play. |
| `active_evidence` | 1 | Weaker than speed and the line test, because balls also move during stoppages (thrown back to the taker, placed). |
| `out_margin_m` | 0.11 m | Ball radius. The ball is out when it is wholly over the line (Law 9), so its centre is one radius past the line. |
| `out_evidence` | 3 | A ball seen over a line is near-certain evidence of a dead ball, as strong as everyone standing still. |
| `evidence_window_s` | 2.0 s | Hysteresis: dead and alive are decided over about two seconds of evidence, the time a whistle takes to stop a team. |
| `min_dead_s` | 3.0 s | Even quick throw-ins and free kicks take about 3 s from ball dead to restart. Shorter dead runs are treated as noise. |
| `min_live_s` | 1.0 s | Live play shorter than a second between two stoppages is not plausible. It is merged into the stoppage. |
| `boundary_search_s` | 3.0 s | How far around a detected stoppage we look for the line crossing or deceleration onset. This covers smoothing lag plus the referee's reaction. |
| `exit_reach_m` | 5 m | A ball last seen within 5 m of a line before a gap can plausibly have crossed it during the gap. |
| `exit_min_speed_mps` | 3 m/s | Only a ball moving toward the line at least at jogging pace is extrapolated across it. |

## Restart and kickoff

| Parameter | Value | Rationale |
|---|---|---|
| `rest_speed_mps` | 2 m/s | A ball held, placed or carried by a walking player moves slower than this. Faster means it was kicked or thrown. |
| `rest_s` | 0.5 s | A restart is taken from a settled ball (set-piece rule). Half a second of rest (or missing ball) before the kick separates it from a pass in play. |
| `clear_move_m` | 1.0 m | "The ball is in play when it is kicked and clearly moves" (Laws 8, 13–17). One metre is clearly beyond tracker noise and ball size. |
| `taker_radius_m` | 2.0 m | The taker is next to the ball when it is kicked. 2 m allows for tracker offset; walls stand at 9.15 m. |
| `kickoff_offside_players` | 2 | At a kickoff every player except the kicker is in their own half. One extra allows for tracker noise or a duplicate. |
| `kickoff_min_tracked` | 8 | We need most of a team tracked before reading a formation. |
| `kickoff_formation_s` | 10 s | Before a half starts, teams stand in kickoff formation for well over ten seconds while waiting for the whistle. Shorter runs happen while players walk onto the pitch. A 2 s value was tried first and accepted a 4.5 s run of players milling about before the second half (seen on tracking only, against the published kickoff frame, which is an allowed assertion). At ±20% (8 s or 12 s) the detected kickoffs do not change. |
| `kickoff_window_s` | 300 s | A half's kickoff formation is searched only in the first five minutes of that half's video block. Teams line up within a few minutes of the block start, while formations later in the half (after a goal, or a stoppage with everyone standing) are restarts, not the half's kickoff. The earliest qualifying formation in the window is used. If none is found, a warning goes to stderr and dead ball is forced until the first detected restart in the window (or the first frame only if there is none). |
| `kickoff_radius_m` | 2.0 m | A ball leaving within 2 m of the centre mark at the end of the formation is the kickoff kick. It is used to sharpen the kickoff frame when the ball is tracked. |

## Control and switches

| Parameter | Value | Rationale |
|---|---|---|
| `control_radius_m` | 1.5 m | Playing reach (leg length about 1 m) plus ATD position noise (about 0.5 m). |
| `control_s` | 0.5 s | "Controlled" means more than a touch: a first touch and settling the ball takes about half a second. A deflection or a pass going past a player is shorter. |
| `control_ball_mps` | 8 m/s | A ball at a player's feet moves no faster than a sprinting player (about 8–9 m/s). Faster means a pass or shot going by. |
| `gap_s` | 1.0 s | A ball missing for more than a second may hide the change of team, so the switch is placed by team-shape evidence inside the gap rather than at the reappearance. |

Parameters accept 0 only where 0 is meaningful: `active_evidence`, `out_evidence`, `out_margin_m` and `kickoff_offside_players`. All others must be greater than 0.

When neither the taker, the Smart Tagging team nor the previous possession decides the team that restarts play, the team comes from the nearest player to the ball at the restart frame and the next second (first team within `control_radius_m`). With no evidence at all the first control after the restart decides, and if there is none the segment stays dead ball. There is no fixed home-team default.

Fixed legacy constants (`possession.params.LEGACY_*` and `rules.RESTART_TOUCH_HORIZON_S`): velocity
by central differences over ±2 frames, team-shape smoothing 1 s, halfway margin 1 m, a 1 s horizon
for the clear-move test, and the legacy cleaning `LEGACY_CLEANING` (ball 40 m/s, ball spike 1.6 m = one 25 fps frame
at 40 m/s, players 12 m/s, player deviation 0.41 m (the DFL-derived `player_deviation_m` copied so `--legacy` needs no external data), duplicates 0.5 m, interpolation of gaps of 8 frames or fewer). The 5 m pressure count and the 90th
speed percentile were removed (no method used them).

## Selection and stability (label-free, full match)

Priors from general football knowledge, not from this match: a stoppage share of 40–48% of the
graded time (effective playing time in top leagues is about 52–58 of roughly 95–100 minutes), about
90–130 stoppages (about 40–50 throw-ins, 15–20 goal kicks, about 10 corners, 25–35 free kicks, a
few kickoffs, plus injuries and substitutions), and roughly 200–320 possession spells (Opta-style
counts give about 90–120 possessions per team, and a mechanical label counts some short
contested spells too).

Only `control_s` and `control_radius_m` were compared as variants. The rest are single values from
the tables above.

| Variant | spells | median spell | phase-team agreement |
|---|---|---|---|
| `control_s` 0.2 s | 435 | 4.3 s | 75.2% |
| `control_s` 0.4 s | 333 | 6.2 s | 77.5% |
| **`control_s` 0.5 s (chosen)** | **302** | **7.3 s** | **78.9%** |
| `control_s` 0.6 s | 274 | 8.0 s | 78.2% |
| `control_s` 0.4 s, radius 1.0 m | 269 | 8.1 s | 78.5% |

The other rows were measured before the later bug fixes and were not recomputed; the chosen row was. 0.5 s is the common-sense value. It sits inside the spell prior and ties for the best agreement
with Smart Tagging phase teams. That agreement is weak evidence, since Smart Tagging is automatic and
counts phases, not possessions.

A ±20% perturbation of each of the 26 original parameters, one at a time, changes the spell count by 1.4% on
average (worst case 11%, from `control_radius_m` ×0.8). It changes 0.8% of graded frame labels on
average. The one fragile parameter is `calm_speed_mps`: ×0.8 gives a stoppage share of 35%, ×1.2
gives 54%, and about 10–14% of graded frames change. The 1.5 m/s value gives 45%, inside the prior,
but the stoppage share depends on it. This is the main known weakness of Phase 1.

# Learned parameters (method `learned`, Phase 2)

Rule for this method: every number is (a) a metadata fact, (b) a library default, (c) learned from
external labelled data, or (d) chosen by leave-one-match-out (LOMO) cross-validation on degraded
external matches. No number is fitted, tuned or selected on BAR-ATH labels. The v1 to v2 public change is within
noise and was not used to choose v2. The hand-set values above stay inside the `rules` method only.

## What is learned from what

- Training data: one of six training-set variants of the degraded external matches
  (`possession.external.prepare`, seed 0), chosen by held-out evaluation (see "Training-set
  variants and selection" below): `dfl` (7 DFL matches, full H/A/D labels from `BallStatus` and
  `BallPossession`), `dfl+metrica` (plus the 3 Metrica matches, H/A/D from events, D heuristic),
  `dfl+metrica+pff` (plus the 64 PFF World Cup 2022 matches, H/A/D from PFF's game events, D
  heuristic), `dfl+metrica+pff_ha` (the same matches, PFF for the team model only, in case its D
  convention differs), `dfl+metrica+skillcorner_ha` (plus the 20 SkillCorner matches, H/A only) and
  `metrica`.
- What each source teaches. Sources with dead-ball labels (DFL, Metrica: `has_dead_labels` in
  `SourceInfo`) train everything: the dead-ball model, the team model, the HSMM durations,
  transitions and initial states, and the rolling scales, except in a variant that lists the source
  as team-only (`train.TEAM_ONLY`: PFF in `dfl+metrica+pff_ha`), where its matches are handled
  like SkillCorner's labelled frames: team model only, D frames left out. SkillCorner trains the
  team model only.
  It has no dead-ball flag: its frames without a possession group mix dead ball, loose ball and
  balls in flight, so they are left out of the loss entirely (never filled with the last team), and
  its labelled frames never enter the dead-ball model, so it cannot teach P(D). It does not
  contribute run durations either: without D its H/A runs either span stoppages or stop at
  unlabelled frames, and its possession boundaries follow a different convention (about ±60
  frames from DFL's), so its run lengths would describe a different quantity than the H/A
  durations the decoder needs.
- Team symmetry: every training match is used twice, once from each team's perspective (features
  mirrored, H and A swapped). Durations, transitions and initial states are pooled the same way, so
  the model has no home-team bias.
- Classifier: two `HistGradientBoostingClassifier`s, factorised as P(D | x) (dead-ball model, frames
  of sources with dead-ball labels) and P(H | x, alive) (team model, alive labelled frames of every
  source in the variant). P(H | x) = (1 - P(D | x)) P(H | x, alive), and likewise for A, so the three
  posteriors sum to one. The factorisation is what lets a source without D labels teach the team
  split without touching P(D); it is used for every variant so that variants differ only in their
  data.
- Team-model features (`--team-features touch`, the default): on top of the core features, rolling
  means at the short scales and ball touch events. Per frame: ball impulse, turn angle and speed
  change over the touch half-span; the last and next touch (time, team, impulse, turn, nearest
  player distance); time since and until the last touch of each team; and the per-team touch
  strength and count, averaged over past and future windows at every rolling scale. Touches never
  cross a block boundary. The core features come first in the same order as before, and the
  dead-ball model and the adaptation use only them, so `--team-features core` builds the
  same feature matrix, and so the same models, as before.
- Weighting: every match has the same total weight in each model, whatever its length or its share
  of labelled frames: sample weight 1 / (rows of the match in that model), rescaled so the mean
  weight is 1 (the unweighted scale of the gradient-boosting Hessians, so library defaults keep
  their meaning). The HSMM uses the same rule: each run, transition and period start of a match has
  weight 1 / (number of them in the match), in the duration KDEs (`gaussian_kde` weights), the
  transition and initial counts, and the rolling scales (weighted mean run length). No weight is
  tuned. With matches of equal length this reduces to the unweighted fit.
  With `--spell-weights` the team model alone weighs each row by 1 / (length of the labelled run
  it belongs to), renormalised so every match keeps the same total and the mean weight stays 1:
  inside a match every spell then weighs the same, whatever its length, which is how spellF1
  counts them (one possession, one match or miss). The run is the maximal run of one label inside
  a period of the training codes (the same runs the HSMM durations use), so the rule adds no
  number. Runs end at every dead-ball stoppage only when the source labels dead ball: every
  training source of `dfl+metrica` does (DFL and Metrica, `has_dead_labels`; checked, dead share
  0.32 to 0.52 in every match), but a source without dead-ball labels (SkillCorner, in
  `dfl+metrica+skillcorner_ha`) has no D in its codes, so its spells merge across stoppages and
  its run lengths overstate the spell length; that variant is not used with `--spell-weights`
  without a decision on this. The dead-ball model and the switch prior keep the per-match
  weights.
- Decoder: an explicit-duration HSMM over H, A and D. The emission of state s at frame t is
  `w * (log P(s | x_t) - log P(s))`, the hybrid scaled likelihood, with P(s) being the training
  frame shares. Durations, transitions and the initial state come from the training labels. Exact
  Viterbi runs on steps of `stride` frames. Then each boundary moves to the frame, within one
  stride, that maximises the emission evidence (exact local search). The first segment of a block
  uses the residual-life distribution (left-censored), and the last segment uses the survival
  function (right-censored). Every duration from one frame to the whole block has positive
  probability (see the duration tail below), so every block has a feasible segmentation. Beyond
  the longest training run the duration log-probability is linear in the length, so Viterbi handles
  the tail exactly with a running maximum instead of a longer table.
- Switch prior (default on; `--no-switch-prior` for the ablation): a third
  `HistGradientBoostingClassifier` estimates P(the possession passes to team s within the scorer's
  tight tolerance of frame t | touch-event features at t), for s = H and A, from the training
  fold's labels. Its log Bayes factor, logit P - logit(training share), measures how much more
  likely a change into s is at t than at a typical alive frame. The decoder adds `w_switch` times
  that log Bayes factor to every direct H->A or A->H transition into s (never to a transition
  from or to D): at a step boundary, `w_switch` times the log of the mean unweighted Bayes factor
  over the frames nearest that boundary (a uniform prior on where the change falls; every frame
  of the block after the first counts for exactly one boundary Viterbi can place, the frames
  before the first boundary and after the last one join those boundaries), and in the boundary
  refinement, `w_switch` times the value at the candidate frame. `w_switch = 0` is exactly the
  decoder without the prior (the term is not added), and a model whose selected `w_switch` is 0
  is saved without the prior (the final model is then fitted without it, and fold models drop
  it), so prediction never computes it. When the sub-sampled training rows of a fit hold no
  change (or only changes), that fit has no switch prior: logged on stderr, its `w_switch`
  settings decode as 0, and `switch_prior` is `null` for that fold in `variant_report.json`. Labels: a frame is positive for s when a direct
  change into s (an alive frame of the other team followed by one of s) lies within the touch
  half-span; rows are the alive frames of the training matches with dead-ball labels (the
  sources that also give the HSMM its transitions), sub-sampled like the classifier rows, from
  both team perspectives (one row per team and frame, features mirrored), weight 1 / (rows of the
  match). Features: ball present, nearest player distance of each team, ball impulse, speed
  change, turn and peak flag, and the touch events (last and next touch time, team, impulse, turn
  and closest distance; time since and until each team's last touch; per-team touch strength
  and count), using the training fold's touch threshold. The H column at frame t is the A
  column of the mirrored features, so the prior has no home bias.

## Parameter origins

| Parameter | Value | Origin |
|---|---|---|
| fps, pitch size, graded blocks / external periods | 25, 105.3 x 68.0 m, `games.half_video_blocks` | (a) metadata |
| H, A, D states | 3 | (a) challenge definition |
| Classifier hyperparameters (`learning_rate`, `max_iter`, `max_leaf_nodes`, `l2_regularization`, `early_stopping='auto'`, `validation_fraction`, ...) | scikit-learn 1.9.1 defaults | (b) library default |
| Classifier `random_state` (`MODEL_SEED`) | 0 | seed (drives the random validation split of `early_stopping='auto'` and the row subsample the library draws for the bin edges of a fit with more than 200,000 rows) |
| Boosting iterations (`--stopping`) | `default`: library defaults, `max_iter` 100 with `early_stopping='auto'`: on a fit with more than 10,000 rows the library holds out a stratified random 10 % of the rows (`validation_fraction`) as validation data and fits on the other 90 %. `team` (team model) or `both` (team and dead-ball models): early stopping on whole held-out training matches. The training matches of the fold are dealt into K validation groups: per source, in name order, match i goes to group i mod K, with K = max(2, matches of the smallest source), so a source with n matches puts floor(n/K) or ceil(n/K) of them in every group (one each for the smallest source, more for larger ones). For each group a model is fitted on the other groups and stopped on that group's loss (`X_val`, `y_val`, `sample_weight_val` with the model's own sample weights; `n_iter_no_change` and `tol` library defaults) in one fit with `max_iter` unbounded (`sys.maxsize`; the library keeps no storage proportional to `max_iter`, so no cap is chosen). The stop of a group is its best iteration, the argmax of `validation_score_` (index 0 is the baseline before the first tree), not `n_iter_`, which also counts the `n_iter_no_change` trees fitted after the best score (scikit-learn does not roll them back). The final model is refitted on all the fold's rows at the mean of the K stops (rounded, at least 1, `early_stopping=False`), so no training match is lost; with fewer than two usable groups (one training match, or a group without both classes) the library default applies, logged on stderr and recorded per model as `stopping` (`default` or `grouped`) in `model.json` and in the variant report | (b) library defaults, structural (leave one group out, groups of whole matches); per fold in CV. The committed model uses `team`: its team model has `early_stopping=False` and 124 iterations, so the random 10 % split of `early_stopping='auto'` is not used there (frames 40 ms apart are near copies, so a random split validates on the spells it trains on; in a scratch test it ran to 885-1000 iterations, against 87-150 with whole held-out matches). The dead-ball model, the dead-ball context stage and the switch prior keep the library default, `early_stopping='auto'`: each is fitted on a random 90 % of its rows and runs to the 100-iteration cap (in the committed model all three library-default fits ran to `max_iter` 100, `n_iter_` 100, so their early stopping never triggered; the team model has `n_iter_` 124; `iterations: [100, 124]` and `dead_context.iterations: 100` in `model.json`). Grouped stopping for the dead-ball model (`both`) did not raise held-out spellF1 on the quick evaluation (`docs/DEADBALL.md`, "Team-model fit"), so the final run uses `team` |
| Threads for fit and predict | 1 | determinism (same bytes on every machine with the same wheels) |
| Class prior P(s) | weighted training shares: P(D) from the dead-ball model's rows, P(A \| alive) from the team model's rows | (c) learned |
| Sample weights | 1 / rows of the match in that model, rescaled to mean 1; HSMM statistics 1 / runs (transitions, period starts) of the match | structural (equal weight per match, no tuning) |
| Team-model spell weights (`--spell-weights`) | 1 / (length of the row's labelled run), renormalised to the same total per match and mean 1 (`classifier.spell_weights`) | structural (equal weight per spell inside a match, as spellF1 counts spells; no new number) |
| Rolling feature scales | mean training run length of the possession states (H and A pooled, by team symmetry) and of D, in frames: the characteristic duration of each state family | (c) learned; per fold in CV |
| Short rolling scales (team model only, `--team-features short` or `touch`) | weighted lower quartile and median (`SHORT_QUANTILES`, `numpy.quantile` `inverted_cdf`, per-match weights as above) of the training H/A run lengths, minus any value already among the rolling scales | The scales are learned from the training run lengths (per fold in CV), but the quantile levels 0.25 and 0.5 are hand-set (MAGIC): the standard five-number-summary levels below the mean, not tuned and not chosen by cross-validation |
| Touch window half-span (velocity before/after a frame, and peak suppression radius) | `possession_metrics.TAUS[0]` frames at `possession_metrics.FPS`, converted to the match frame rate (12 frames at 25 fps) | (a) metadata: the metric's boundary tolerance, the resolution at which a touch can matter to spellF1 |
| Touch candidates | local maxima of the ball impulse \|mean velocity after - mean velocity before\| within one half-span (one per plateau: its first frame), searched after masking frames without the ball, inside one block; the ball velocity behind it is re-estimated per block so no window crosses a block or period boundary; impulse, turn angle and speed change are continuous inputs | structural (no threshold) |
| Touch impulse threshold (`FeatureLayout.touch_threshold`) | Youden's J on the training fold's candidate peaks: positive when both H and A occur within one half-span of the peak, negative when the window is one alive state only (mixed with D or unlabelled: left out); sample weight 1 / (peaks of the match); every training match with H/A labels contributes, with or without dead-ball labels; `sklearn.metrics.roc_curve`. Peaks of one class only is an error. With no labelled peak at all there is nothing to learn from, so the threshold falls back to 0 (every peak is a touch), logged on stderr and visible as `touch_evidence_peaks: 0` in the model metadata and variant report | (c) learned from training-fold labels only; stored in the model |
| Team touch threshold with `--every-peak` | `EVERY_PEAK` = 0: every impulse peak is a touch event of the team features (the impulse is a norm, so 0 keeps every peak with a ball velocity; the same value as the fallback without labelled peaks). The impulse, turn and closest-player distance of each event stay continuous inputs, so the team model weighs weak and strong touches itself instead of a Youden cut removing the weak ones. The switch prior keeps its own Youden threshold: on the quick evaluation below, giving it every peak too lowered held-out spellF1 (`docs/DEADBALL.md`, "Team-model fit") | structural (no threshold); prior threshold (c) learned |
| Touch team | team of the nearest player at the touch frame (own +1, other -1, tie or none 0) | structural |
| Duration pmf per state, up to the longest training run L | Gaussian KDE of log run length; length d gets the KDE mass of (log(d-1), log d], i.e. a run is the ceiling of a continuous length | (c) learned |
| KDE bandwidth | Scott's rule | (b) `scipy.stats.gaussian_kde` default |
| Duration tail beyond L | geometric; its mass is the KDE mass above log L and its per-frame hazard is the fitted hazard at L, P(D = L) / P(D >= L) (constant-hazard extrapolation) | (c) learned |
| Durations of a state with a single distinct training length | geometric with that mean (maximum-entropy law on 1, 2, ... for a given mean) | (c) learned |
| Transition matrix (no self-transitions) | normalised run-to-run counts | (c) learned |
| Initial state | normalised first-label counts of the periods | (c) learned |
| Emission weight `w = 1 / span` | span grid {1, 2, 5, 10, 25, 50, 100} frames (1-2-5 series from one frame to four seconds) | (d) LOMO CV |
| Viterbi `stride` | starting grid {5, 10, 25} frames (0.2, 0.4, 1.0 s at the match frame rate). Extended upward by the rule under "Switch weight" below: the next member of the 1-2-5 series in seconds above the selected stride that is a whole number of frames (`train.next_stride`: 1 s -> 2 s -> 5 s -> 10 s, i.e. 25 -> 50 -> 125 -> 250 frames; 0.5 s would be 12.5 frames, so it is skipped), added for every span and switch weight. The frame rate is `possession.data.fps()` (the games metadata): every source is resampled to it before it is used (`external.prepare.build_harmonised` calls `harmonise(..., fps(), ...)`, which sets `ExternalMatch.fps`; the degraded variant keeps it), and it is the rate of our match, so a stride in frames means the same time for every source. The fit does not depend on the stride (it only enters the decoder), so only the new settings are scored on the fitted held-out fold models | (d) LOMO CV; the series is the one the span grid uses, in seconds, and 1.0 s is already a member (the previous grid selected 25 in 10 of 10 nested folds and in the final `dfl+metrica` model, the top of the grid). The grids used are recorded like the switch weight's |
| Switch prior model (`LearnedModel.changes`) | `HistGradientBoostingClassifier` defaults, `MODEL_SEED`, one thread; labels within the touch half-span (`possession_metrics.TAUS[0]` at the match frame rate) of a training change; training share as the prior odds | (b) library default, (a) metric tolerance, (c) learned from the training fold only; stored in the model |
| Switch weight `w_switch` (`DecoderSettings.switch_weight`, `train.SWITCH_WEIGHTS`) | starting grid {0, 0.5, 1, 2}: off (the previous decoder), the Bayes factor at face value (1) and one step of the 1-2-5 series either side of it (the series the span grid uses). The grid has no fixed top, and neither has the stride grid. A selection (the final one over all folds, and each fold's nested one over the other folds) runs its own loop: if its best setting has the largest `w_switch` or the largest stride of the grid it is searching, the next member of the 1-2-5 series above it (`train.next_in_series`: 2 -> 5 -> 10 -> 20 -> ...; strides: see "Viterbi `stride`") is added for every span and for the other axis, the new settings are scored on the held-out fold models (nothing is refitted: the classifier fit depends on neither axis), and the loop continues only while a new setting beats the best of the previous grid strictly (spellF1, then macroF1). A tie keeps the old best and stops, so decodings that no longer change cannot extend it forever. Fold i's nested loop never reads fold i's scores (it extends only because of the other folds). Each (fold, setting) is scored once and shared by all loops through `train.GridSearch`. Per variant; `--no-switch-prior` keeps `w_switch = 0` only, so only the stride extends | (d) nested LOMO CV, jointly with span and stride; on exact ties of spellF1 and macroF1 the smaller `w_switch` wins, then the earlier grid setting. Recorded in `variant_report.json`: top-level `selected_variant_grid` (every setting scored for the selected variant), per variant `grid` (every setting scored), `final_selection_grid` (axes of the grid the final selection searched), and per nested fold `searched_max_switch_weight` and `searched_max_stride`; in `model.json`: `grid` (span weights, strides and switch weights the final selection searched). First run, grid {0, 0.5, 1, 2}: 2.0 selected in the final `dfl+metrica` model and in every nested fold, the top of the grid (`docs/DEADBALL.md`, 2026-10-27) |
| Step aggregation of the switch evidence | log of the mean unweighted Bayes factor over the frames nearest each step boundary (from `stride // 2` frames before it to the next boundary's group), then times `w_switch` | structural (uniform location of the change, no new number) |
| Switch prior in the model metadata and report | `switch_prior_threshold` and `switch_prior_peaks` (the touch threshold the prior's features used and the number of labelled peaks it came from), `switch_prior_rows`; per fold in `variant_report.json` | record only |
| Boundary refinement radius | one stride | structural (undoes the step quantisation, no new number) |
| Settings of each held-out fold model | the nested choice: best grid setting on the other folds only | (d) nested LOMO CV |
| Selection rule for decoder settings | highest spellF1, then macroF1 (challenge ranking order), averaged per source first and then over the sources of the variant (equal weight per source, so 20 SkillCorner matches do not outvote 3 Metrica matches), using DFL raw scores and Metrica and SkillCorner fair scores | (a) challenge ranking order |
| Training-set variant | see "Training-set variants and selection" | (d) held-out evaluation |
| Bootstrap draws (`train.BOOTSTRAP_DRAWS`) | 1000 | (a) vendored `possession_metrics`: "1000 for a final leaderboard"; its seed `BOOTSTRAP_SEED` 0 and 5-95 percentiles are used unchanged |
| Degradation seed | 0 (`prepare.DEFAULT_SEED`) | seed |
| Cleaning parameters (`CleaningParams`, seven values) | see "Cleaning parameters" below | (c) estimated from DFL and our match; stored in the model |
| `--frame-step` | 1 (every frame) | smoke runs only, never in a submitted model |
| `--team-features` | `touch` (default), `short` or `core` (the feature set before touch events, for ablation) | design switch; the dead-ball model and the (c) adaptation always read only the core columns, so they are identical across the three |
| `--spell-weights`, `--every-peak`, `--stopping` | off, off, `default` (the v8 model); recorded as `classifier` in `model.json`, `variant_report.json` and the fit digest | design switches (see the rows above) |
| `--no-switch-prior` | off (default: the switch prior is fitted and `w_switch` is in the grid, which extends itself along the 1-2-5 series while its largest weight is selected, as does the stride grid); on fits no switch prior and keeps `w_switch = 0` only, the decoder as before | ablation switch; independent of `--team-features` (the prior needs the touch inputs, which are then computed for `core` and `short` too) |

The selected variant, `span` and `stride`, and every held-out score are written to
`models/learned/variant_report.json` and `models/learned/model.json` by `possession.learned.train`
(not committed until training is done), together with the decoder grid the selection finished on
(the starting grid plus any switch weights and strides it added).
Each fold model in `models/learned/folds/<variant>/` is saved with its own nested choice, so
`external.validate --method learned` reports truly held-out scores for the matches of the selected
variant (other matches are scored by the final model, which never saw them). A run id (sha256 of the
training configuration, the variant, the degradation and model seeds, the match list and the final
model sha256) is written to `model.json`, to every fold sidecar and to `variant_report.json`;
`external.validate` refuses fold models whose run id differs from the final model's.

## Training-set variants and selection

`python -m possession.learned.train` evaluates each variant of `--variants` the same way. The default
is `dfl,dfl+metrica,metrica`; the PFF variants (`dfl+metrica+pff`, `dfl+metrica+pff_ha`) and the
SkillCorner variant `dfl+metrica+skillcorner_ha` are excluded by default and trained with
`--variants all` or by name. A list that lacks any of the three default variants is a partial run
and needs an explicit `--out`; a list that contains all three (for example
`dfl,dfl+metrica,metrica,dfl+metrica+pff,dfl+metrica+pff_ha`) is a superset of the default and may
write the default model path. All selected variants are checked for at least two matches per source
before any fitting.


1. Leave-one-match-out over the matches of the variant's sources. Each fold model is trained on the
   other matches and scores the held-out match over the decoder grid; the setting reported for the
   fold is the nested choice (best on the other folds only). Exception, PFF (`grouped_folds` in
   `SourceInfo`): its 64 matches would make 64 folds, each trained on about 70 matches. They are
   held out in K contiguous groups (discovery order, i.e. by game id; `numpy.array_split`), K =
   the match count of the largest other source of the variant (DFL, 7), so PFF adds as many folds
   as DFL. A group fold is trained on all matches outside the group and scores each match of the
   group; every match still gets a held-out score and its own nested setting. The group's model is
   saved once (`<source>_group<i>of<K>.pkl`); each match has only a sidecar `<source>_<name>.json`
   with `group_model` (the shared file), its own `emission_weight` and `stride`, `held_out_group`
   and the run id (`external.validate` reads the shared model through the pointer, applies the
   match's settings and checks that the fold was trained on the final model's matches minus that
   group).
   Rows: in every fit, the classifier rows of a grouped source are every s-th frame of each of its
   matches, s = ceil(n / K) with n its matches in the fit and K as above (64 PFF matches with 7 DFL:
   s = 10). PFF then adds about as many rows as DFL (about 7 matches' worth) instead of 64 matches'
   worth. Memory: the full-resolution features of every loaded match are written to a temporary
   directory of the training process (`.npy`, memory-mapped, removed at the end; about 4 GB of disk
   for 74 matches) instead of being held in RAM; the classifier matrix is float32, built in one
   preallocated array from the kept rows only, and the dead-ball fit selects its rows with a single
   copy. A smoke fit (2 DFL, 2 Metrica, 2 PFF, `--frame-step 10`) peaked at 2.4 GB resident memory,
   against 5.3 GB before (the held-out `ExternalMatch` kept views of its full ball and player
   arrays). Estimated for the full run: about 2.5 GB for the process and one match being loaded,
   plus 5 to 6 GB while the classifier fits (1.3 GB float32 rows, one selected copy, and the
   float64 copy that `HistGradientBoostingClassifier` makes, measured at 3.3 times the float32
   input), so about 8 GB, up to about 10 GB if the operating system keeps all mapped pages;
   the previous layout peaked at about 15 GB. Adjacent
   frames at 25 fps are strongly correlated, every match keeps its equal weight (weights are per
   match, not per row), and the HSMM durations, transitions and scales still use every frame. Every
   fit prints the rows, the features and the size of the float32 matrix to stderr. Matches of a
   source that is not in the variant are scored by the final model only (step 2). Selecting the
   decoder setting uses only the folds whose dead-ball labels the variant trusts for the dead-ball
   model and the durations (`Fold.dead_labels`): DFL and Metrica always, PFF only in
   `dfl+metrica+pff`; in `dfl+metrica+pff_ha` and for SkillCorner the folds are scored and reported
   but do not choose the setting, because their D labels are not used to train the variant.
2. The variant's final model is trained on all its matches with the setting that is best over all
   its folds. Matches of sources outside the variant (for example Metrica for `dfl`) are scored by
   this final model, which never saw them. So every variant reports a held-out score on every DFL,
   Metrica and SkillCorner match. PFF matches are loaded only when a selected variant trains on
   them, and are scored by the variants that train on them (their group folds); the 64 matches are
   scored by the other variants' final models only with `--score-outside pff` (default off, since
   it costs a full-match emission per PFF match and variant and PFF never selects). A run without
   PFF variants is the previous run.
3. Scores per match: DFL raw (provider D), Metrica fair and raw, SkillCorner fair, PFF fair and raw
   (as `external.validate`). The comparable score is raw for DFL and fair for Metrica, SkillCorner
   and PFF (`docs/EXTERNAL.md`: Metrica's and PFF's D are heuristic). Each comparable score carries
   the vendored possession bootstrap interval (`pm.bootstrap_spell_f1`, 1000 draws, 5-95
   percentiles; also on the Metrica and PFF raw scores). Matches with no ground-truth spells have no spellF1 or interval and are
   left out of the summaries (counted in `n_without_spells`).
4. Per variant and source: mean spellF1 over the matches, and the interval [mean of the per-match
   lower bounds, mean of the per-match upper bounds]. Averaging the bounds treats the per-match
   errors as fully correlated, which gives the widest (most conservative) interval of the mean; it
   has about the width of the official full-match noise band (`SPELL_NOISE_BAND` 0.04, EVALUATION.md),
   which is what it is meant to reproduce on small sets.

Selection rule (`train.select_variant`):

1. Primary: mean held-out Metrica spellF1 (Metrica is the provider of our ground truth), fair or raw
   as chosen by `--selection-score` (default `raw`); both are always reported. The variants tied
   with the best are those whose interval upper bound (mean of the per-match bootstrap upper
   bounds) reaches the best variant's lower bound (mean of the per-match lower bounds); if none
   does, the best mean is kept.
2. Among those, the same rule on the DFL raw mean.
3. Among those, the highest SkillCorner fair mean (then Metrica, then DFL means).

PFF is reported (columns `pff fair` and `pff raw`) but takes no part in the selection, which is
unchanged: its D convention is not yet validated against DFL's, and the decision rule of
`docs/DEADBALL.md` was fixed before PFF was added. PFF enters the decoder-setting choice only in
`dfl+metrica+pff`, where its D labels are trained on, as one source among three (step 1, averaged per
source).

Fold and final models are cached in `models/learned/folds/<variant>/` with a `.scores.json` sidecar
holding the held-out scores and a digest of every input (code of the training and scoring modules,
the degraded cache files of the training and evaluated matches, cleaning parameters, seeds, frame step, training and evaluated matches,
settings). A rerun reuses a fit only when the digest is identical, so an interrupted run resumes and
variants can be trained in separate processes (`--variants dfl --out <path>`, ...) before a final default run that only fits what is missing.
Folds are cached per output path, so use the same `--out` directory.

## Dead-ball model options (domain shift, `docs/DEADBALL.md`)

Two `train` flags, stored in the model (`dead_ball` in `model.json`) and in every fold digest. The
defaults reproduce the previous method exactly.

| Parameter | Value | Origin |
|---|---|---|
| `--dead-features` | `all` (default) or `players` | structural choice between two candidates, decided by the rule in `docs/DEADBALL.md`. `players` keeps only the features computed from player positions and speeds: centroid x and velocity, defensive line, spread in x and y, team speed, closest player to the centre mark (both teams), shape, mean speed, and the rolling means of shape and mean speed. Everything derived from the ball (presence, position, speed, gaps, nearest distances, dwell) or from the object count (`tracked_*`) is excluded, because its relation to the dead-ball state depends on the tracking provider (DEADBALL.md, Investigation). The partition is by construction, not by a shift threshold |
| `--adaptation` | off (default: none); on runs the EM until converged | switch only. The number of iterations is not chosen: the EM stops on its convergence test (below). `adapt.SAFETY_BOUND` = 20 iterations is a run-time guard against a non-terminating loop (struct, like the attempt budgets of the degradation), not a statistical setting: it caps compute and only matters if the test is never met. Reasoning for the value: the earlier quick experiments (without prior correction, cap 6) converged in 2 to 5 iterations on most matches, and a synthetic target whose D share is 0.85 against 0.40 in training converges in 12 with the prior correction, so 20 leaves headroom. Hitting the bound is visible: the per-match report (stderr, `variant_report.json`) has `converged: false` |
| EM prior correction | emitted and iterated logit = player logit + logit(target share) - logit(training share) + log ratio | Saerens, Latinne and Decaestecker (2002), target share = mean of q, training share = the classifier's training prior; (c) learned. The log ratio is refitted on the final q so the emitted logit, the share and q agree |
| EM share bound | shares are clipped to [1/(n+1), n/(n+1)], n = number of frames the share is computed on, before the logit | (c) Laplace-style margin from the sample size: the smallest share a sample of n frames can distinguish from 0 or 1, no free constant; keeps logits finite |
| EM convergence | stop when the change of the target D share is at most sqrt(sum q(1 - q)) / n over the in-block frames (the standard error of a mean of independent Bernoulli(q) frames) | (c) from the target's own posterior, no tolerance chosen |
| EM folds | the target's own graded blocks (halves): the observation model of each half is fitted on the other half; a single block is split into 2 contiguous halves (the 2 halves of a match); fewer than 2 non-empty folds: no adaptation | structural cross-fitting, no frame is scored by a model fitted on it |
| EM observation model | `HistGradientBoostingClassifier` defaults (`max_iter` 100) except `early_stopping=False`, seed `MODEL_SEED`, rows duplicated as D with weight q and alive with weight 1 - q, weights rescaled to mean 1 | (b) library default; the weighted fit is the M-step of EM written as a density-ratio classifier. Soft labels need the two copies of each frame; with the library's early stopping the internal validation split would contain the copy of a training frame (leakage), so early stopping is switched off (a library option) and the library `max_iter` bounds the fit instead of a hand-picked count |
| EM prior update | target D share = mean posterior (Saerens, Latinne and Decaestecker 2002) | (c) learned on the target, no labels |

The adaptation runs at prediction time on each predicted match (our match, each held-out external
match): the dead-ball log-odds are the player model's log-odds plus the Saerens prior correction (target
share against training share) plus the log likelihood ratio of the
ball and object-count features estimated on that match by EM, assuming those features and the
player features are conditionally independent given the state. The team model is unchanged.

## Pitch-line feature groups (`train --line-features`, `docs/DEADBALL.md`)

Three optional groups of features, stored in the model by their column names (`line_features` in
`model.json` only when a group is on, no extra field in the pickled model) and in the run id only when requested, so the
defaults reproduce the previous method exactly (same run id, same pickle). They exist because the
held-out external matches show that throw-ins and goal kicks are the stoppages the dead-ball model
misses most (`docs/DEADBALL.md`, "Dead ball by stoppage type"), and because in all three providers the
person taking a throw-in or goal kick stands outside the lines in the tracking. All are computed on
the cleaned tracking in metres, are threshold-free (a signed distance, a speed, a duration) and are
built by `framefeatures.line_inputs`:

| Group | Columns | What it is |
|---|---|---|
| `lines` | `player_out_touch`, `player_out_goal` | furthest distance, over the tracked players, beyond the touch lines or the goal lines (negative when everybody is inside; the pitch size is metadata) |
| | `line_player_speed` | speed of the tracked player closest to (or furthest beyond) a line, whichever line |
| | rolling past and future means of the three | same scales as the other rolling features |
| `keeper` | `keeper_depth_own/other` | distance to the own goal line of the team's deepest tracked player, a proxy for the goalkeeper (the tracking has no identities) |
| | `keeper_speed_own/other` and their rolling means | speed of that player |
| | `keeper_last_ball_distance_own/other` | distance from that player to the last observed ball position of the same period or block (ball based) |
| `ball` | `since_out`, `until_out` | time since and until the ball is observed beyond a line (ball based) |
| | `entry/exit_touch_distance`, `entry/exit_goal_distance` | signed distance of the last and next observed ball to the lines at the ends of a ball gap (ball based) |

The position-only columns (`lines` and the player columns of `keeper`) are player features: with
`--dead-features players` they feed the dead-ball model. The ball-based columns are not player
features: like the existing ball columns they are left out of the dead-ball model and enter the
adaptation (`--adaptation`) and the team model. Nothing is hand-set: no distance or duration is
compared with a constant, the models learn where the signals matter. Quick evaluation (docs/DEADBALL.md,
"Dead ball by stoppage type and pitch-line features"): `lines` alone raised Metrica fair by 0.010, Metrica raw by
0.020 and DFL by 0.020 (3, 3 and 6 of 7 folds better) and throw-in D recall from 0.82 to 0.89 (DFL) and 0.79 to 0.88
(Metrica); `keeper` and `ball` did not improve on `lines`. A first full run of `--line-features lines` on top of the v2
flags (`scripts/phase_v3.txt`) lost on Metrica and was not adopted. A second one on top of the dead-ball context model
(`scripts/phase_final.txt`) gave the committed model: `lines` is on in `models/learned/` (adopted 2026-11-13,
`docs/DEADBALL.md`, "Pitch-line features adopted (2026-11-13)"). Screened on the quick harness on top of the dead-ball
context model it gave DFL +0.018 (7 of 7 folds better); in the full nested run DFL +0.010 (0.5771 to 0.5872, 6 of 7),
Metrica fair -0.004 (0.7755 to 0.7714, 1 of 3) and SkillCorner fair +0.015 (0.3035 to 0.3183, 15 of 20 matches). The
dead-ball model goes from 24 to 39 columns and the team model from 151 to 178. The flag still defaults to none, so a
`train` without it reproduces the method of the earlier internal model v3.

Already in the features before this change (not re-added): the signed distance of the ball to the goal
line and the touch line (`goal_line_distance`, `touch_line_distance`) and the ball x at the ends of a
ball gap. New: everything in the table.

## Dead-ball context (`train --dead-context`, `docs/DEADBALL.md`)

One optional second stage for the dead-ball model, stored in the model by its own fields (`ContextClassifier.context`:
the stage-2 estimator and the scales; `dead_context` in `model.json` only when on) and in the run id and fold digests
only when requested, so the defaults reproduce the previous method exactly (same run id, same pickle). Stage 1 is the
dead-ball model as before, P(D | x) from the player features of one frame. Stage 2 is a second dead-ball model on the same
player features plus context columns computed from the stage-1 logit of the neighbouring frames of the same match:

| Item | Value | Origin |
|---|---|---|
| Context signal | the stage-1 dead-ball logit (`decision_function`) of the frame, as a series over the match | structural |
| Context columns | centred, past and future rolling means of the signal at every rolling scale of the layout (`FeatureLayout.all_scales`: the long scales and, with `--team-features touch`, the two short ones), so 3 x scales columns (`dead_logit_centred_S`, `dead_logit_past_S`, `dead_logit_future_S`) | (c) learned: the scales are the mean training run lengths and quantiles of the existing layout; no new scale |
| Window of a centred mean | `scale // 2` frames back and `scale - 1 - scale // 2` forward (the window holds `scale` frames, as the past and future means do; `framefeatures.centred_nanmean`, the third direction of `rolling_column`, sharing the windowed mean with the past and future ones) | struct |
| Segments | past, future and centred means are computed inside each block or period segment (`framefeatures.block_segments`), never across a block or a period boundary | structural (as the existing rolling features) |
| Stage-1 logit for training rows | out of fold by the validation groups the team model's grouped stopping already uses (`model.validation_groups`: per source, matches dealt round-robin into K = max(2, matches of the smallest source) groups): for each group, a dead-ball model fitted on the matches of the other groups predicts every frame of the group's matches. K fits instead of one per match; only the dead-ball columns are computed for the held-out matches. With `dfl+metrica`, K is the number of Metrica training matches: K = 3 for the committed model (7 DFL, 3 Metrica) and for the DFL folds (6 DFL, 3 Metrica), K = 2 for the Metrica folds (7 DFL, 2 Metrica) | structural cross-fitting, so a training row's context never comes from a model that saw its labels; no new number |
| Boosting iterations of the inner fits and of stage 2 | the same rule as the stage-1 dead-ball fit: library default (`max_iter` 100, `early_stopping='auto'` with its random 10 % validation split; stage 2 of the committed model reached the cap, 100 iterations) unless `--stopping both`, then grouped early stopping (`classifier.fit_dead_ball`) | (b) library default / (c) learned, as stage 1 |
| Stage-2 rows and weights | the rows of the dead-ball fit (labelled frames of matches with dead-ball labels), weight 1 / rows of the match | structural |
| Preconditions | at least two validation groups (two training matches), and the matches outside each held-out group must contain both dead-ball classes; otherwise the fit stops with an error naming the group | struct |
| Prediction | stage 1 on the match, context columns from its logit, stage 2: that logit replaces the dead-ball logit | structural |
| EM adaptation | unchanged and runs on the stage-2 logit: `ContextClassifier.dead_logit(matrix, segments)` returns the stage-2 logit, which `adapted_dead_logit` uses as the player logit (it refuses a context classifier without the match segments rather than falling back to stage 1), the observation columns (ball and object-count features outside the dead-ball columns), the training prior and the folds are those of stage 1, so the Saerens prior correction and the density-ratio refit see the contextual logit | structural; the same choice as the quick evaluation |

No threshold is compared with a constant and no number is hand-set. The team model, the switch prior and the decoder do not
change. Flag off: `fit_model`, `log_emission` and the metadata take the same path as before (the stage-1 logit is the
`decision_function` of the dead-ball model, passed explicitly to the same functions).

Training members are never scored with a model that holds their labels in a context column: `train.run_final` scores only
matches outside the final model's members (`HELD_OUT_FINAL`), the nested folds score their held-out match with the fold's
model, and `possession.run` and `diagnostics` only read the model's cleaning parameters or predict our match.
`--dead-context` fits one extra dead-ball model per validation group and one stage-2 model in every fold and in the final
fit, so the training cost grows by about that much (`docs/DEADBALL.md`).

Quick evaluation: `docs/DEADBALL.md`, "Dead-ball context". The full nested run (`--dead-context` on top of the v2 flags)
is `scripts/phase_final.txt`. The flag is on in the canonical model `models/learned/` (adopted 2026-11-05 on that run,
`docs/DEADBALL.md`, "Dead-ball context adopted (2026-11-05)"); the flag itself stays off by default, so a `train` without it
reproduces the v2 method.

## Features

All features are computed by the same code (`possession.learned.framefeatures`) on our match and on
the degraded external matches. Coordinates are in metres, with x along the attacking direction of the
team in the perspective (attacking directions are detected per half from the deepest players). All
features are threshold-free:

- ball: present flag, x, |y| from the centre line, speed, velocity along x, acceleration, signed
  distance to the goal line and to the touch line;
- ball gaps: time since the ball was last seen and until it is next seen, total gap length, and the
  nearest-player distance difference and ball x at gap entry and exit;
- ball to players: nearest and second-nearest player distance per team, and their differences
  (pressure as k-nearest distances instead of a count within a hand-set radius);
- dwell: signed run length (back and forward in time) of the team holding the nearest player;
- team shape: centroid x and unsmoothed centroid velocity along x, defensive line (second-deepest
  player, Law 11), spread in x and y, mean team speed, tracked players, distance of the closest
  player to the centre mark, mean speed of all players;
- multi-scale context: past and future means of ball present, ball speed, ball x, nearest
  difference, second-nearest difference, shape (centroid velocity difference) and mean speed, over
  the rolling scales above;
- optional pitch-line groups (`train --line-features`, off by default, "Pitch-line feature groups"
  below): `lines`, `keeper` and `ball`; `lines` is on in the committed model.

Ball gaps, dwell and the multi-scale means are computed separately inside each period (external
matches) or graded block (our match), and on the frames between blocks, so they never look across a
period or block boundary.

Player speeds are computed from the cleaned positions (central differences with the estimated
half-span) on our match and on the external matches alike. The ATD speed column is no longer used by
`learned`, so both domains go through the same computation.

Not used as features: the legacy hand-set quantities (5 m pressure count, 1 m halfway margin, 1 s
shape smoothing, 90th speed percentile) and Smart Tagging, which external matches do not have.
Features are built by `features.threshold_free_features`; the legacy `features.build_features` adds
the smoothed shape and the halfway count for `rules` only.

## Cleaning parameters (estimated, shared by `learned` and its training data)

`possession.cleaning.clean(tracking, pitch, fps, params)` takes a `CleaningParams` and has no
default. For `learned` the values come from `possession.external.bounds.estimate_params` (run by
`possession.external.prepare`, cached in `cache/external/params_<key>.npz`, stored in the model
pickle and in `model.json` under `cleaning`). The same values clean our match and every degraded
external match. Values below are from the current data (7 DFL matches, BAR-ATH aggregates).

| Parameter | Statistic | Value | Origin |
|---|---|---|---|
| `ball_max_speed_mps` | per DFL match, the maximum over alive frames of the *consistent* ball speed min(speed in, speed out, speed across the frame); then the median over matches. Used only to refuse interpolating across a gap whose straight line would be faster | 35.7 m/s (match maxima 32.0 to 41.6) | (c) DFL raw optical tracking |
| `ball_deviation_m` | per DFL match, the maximum over alive frames (five-frame alive neighbourhood, the sample and the four neighbours all present: no extrapolation across a gap) of the *consistent* deviation of a ball sample from its neighbours: min(distance to the constant-velocity extrapolation of the two previous samples, distance to that of the two next samples); then the median over matches | 1.07 m (match maxima 0.89 to 1.62; unchanged when the sample and its four neighbours are required present: DFL has no ball gaps in alive play) | (c) DFL raw optical tracking |
| `player_max_speed_mps` | same statistic on every DFL player | 10.8 m/s (match maxima 9.1 to 21.4; the two highest are tracking glitches, two-frame position steps) | (c) DFL |
| `player_deviation_m` | same statistic as `ball_deviation_m` on every DFL player (four neighbours present, alive five-frame neighbourhood); median over matches. Bounds how far a player sample may sit from the extrapolation of the clean track on a side before it counts as displaced | 0.41 m (match maxima 0.13 to 0.97) | (c) DFL |
| `duplicate_distance_m` | per DFL match, the minimum distance between two distinct teammates over alive frames; median over matches | 0.041 m (match minima 0.010 to 0.345) | (c) DFL |
| `max_gap_frames` | two-component Gaussian mixture (scikit-learn defaults, seed 0) on the log lengths of the raw ball gaps of our graded frames; the largest length still assigned to the short component (dropouts) rather than the long one (occlusions) | 24 frames (our gaps: 17 of 1 to 25 frames, none of 26 to 75, 133 of 76 or more) | (c) our match |
| `ball_half_span`, `player_half_span` | smallest half-span h (from 1 up) after which the velocity error stops decreasing: error(h) = mean squared difference between h and 1 central differences on DFL alive frames (bias) plus (sigma * fps / h)^2 (noise of our tracker, sigma = robust second-difference noise of our raw ball or players) | 1 and 1 (our tracker is smooth: sigma 2.9 mm ball, 0.6 mm players) | (c) DFL + our match |

Held-out leak (minor, label-free, not changed). `reference_params` and `reference_stats`
(`possession.external.prepare`, lines 152-163 and 172-181; `possession.external.bounds.estimate_params`,
lines 169-185) are computed once on all 7 DFL matches (plus our match's label-free tracking), so in
the leave-one-match-out folds the cleaning parameters of a held-out DFL match have seen that match's
own tracking (never its labels). `reference_stats` is a statistic of our own match computed with
those parameters, so it inherits the same dependence. The effect is on the held-out DFL scores
only (a robust median over 7 per-match extremes, label-free); Metrica and SkillCorner folds are
unaffected. It is a mild optimism in the DFL external CV, not a use of BAR-ATH labels.

Why these statistics:

- Speeds: DFL is optical tracking of a much higher quality than our ATD, but its raw positions still
  contain glitches (teleports at dead-ball repositioning, two-frame steps). The *consistent* speed
  removes out-and-back spikes, the alive mask removes ball repositioning, and the median over
  matches removes the remaining per-match glitches (breakdown point 50 %). It is the top speed
  reached in a typical match, so it is a physical maximum rather than a tail quantile with an
  invented level.
- Ball glitches (`displaced_runs`): a run of ball samples is removed only when (1) the run starts
  at a sample more than `ball_deviation_m` off the constant-velocity extrapolation of the two
  samples before it, (2) every sample of the run stays more than `ball_deviation_m` off that same
  extrapolation, (3) the run ends at a sample back within `ball_deviation_m` of it, and (4) the
  clean samples after the run, extrapolated backwards, reach the clean sample before the run within
  `ball_deviation_m` (the two clean sides bridge). Spikes of any length (one frame, a plateau of
  three or four frames, ...) are removed edge to edge, since the run is delimited by where the
  trajectory resumes, not by a per-sample test. `ball_deviation_m` is the largest deviation a real
  ball shows in DFL (kicks, headers and deflections included: a kick is predicted by one side, so
  its consistent deviation stays small); a real kick or a pass that is received and stopped changes
  the trajectory and fails (4), a step to a new position that continues smoothly fails (3) or (4).
  A run is judged only from the two samples immediately before it and the two immediately after it,
  all present: next to a gap the test is skipped rather than extrapolated across the gap with an
  invented tolerance, because the bound was derived at one frame distance. Speed plays no role:
  the ATD ball is the ground projection of the image position, so a high ball moves smoothly at up
  to about 70 m/s on the ground plane while it tracks the real ball in the image.
- Player jumps (`jump_segments`): the core is a sample whose speed both into and out of it exceeds
  `player_max_speed_mps` (`impossible_motion`). Our tracker is smooth, so a jump of the marker is
  not always a one-frame spike but a speed excursion: the marker accelerates away from the player,
  flies, and decelerates onto a position several metres away. The speed minima on each side of the
  core (walking outwards while the step speed keeps falling, stopping at a missing sample) give the
  outermost extent, whose quiet frames `first` and `last` are kept (slice `first + 1 .. last - 2`).
  Inside it, going outwards from the core, a sample is removed while it is more than
  `player_deviation_m` away from BOTH the forward extrapolation of the clean track at `first` and
  the backward extrapolation of the clean track at `last`; the first sample consistent with either
  side stops the removal. A single teleport removes exactly that frame; a smooth decelerating or
  accelerating player no longer loses the good frames that the speed walk used to take along; the
  smooth ramps of a long excursion are removed except their first and last frames that stay within
  `player_deviation_m` of the pre/post trajectory. No length or acceleration threshold is used.
  Without a present anchor pair the excursion is just its core. Our match: 4949 core samples, 9197
  removed (233 segments) with the reference parameters, against 12216 with the former speed-valley
  rule.
- Duplicates: a duplicate is two same-team objects closer than distinct real teammates ever get,
  i.e. the minimum distance between teammates, with the same median-over-matches robustness.
- Interpolation: our raw gap lengths are bimodal (tracker dropouts of a few frames, occlusions of
  three seconds or more). The mixture boundary separates the two modes with the Bayes rule; no
  length is chosen by hand. A gap is also filled only if the straight line across it implies at most
  `ball_max_speed_mps`.
- Velocity span: the trade-off between differencing noise and smoothing bias, with the bias measured
  on external data and the noise on our match.

Effect on our match (graded frames; `python -m possession.diagnostics`, legacy values first):

| | legacy (rules) | estimated (learned) |
|---|---|---|
| ball samples removed (spikes) | 0 | 0 |
| ball frames interpolated | 7 | 37 |
| player samples removed (jump excursions) | 3,842 | 4,745 (89 segments, median 51 frames, longest 146) |
| duplicate samples merged | 5,826 | 39 |
| player objects per frame, raw then cleaned (mean) | 20.49 then 20.43 | 20.49 then 20.46 |
| frames with more than 22 objects | 2.36 % then 2.23 % | 2.36 % then 2.33 % |

Before this rule change the speed rule removed 269 ball samples and 1,852 player samples (estimated
values). All 269 ball samples (19 runs of 3 to 23 frames) were smooth fast flight: their largest
consistent deviation is 0.23 m against the 1.07 m bound, and the largest on all our graded frames is
0.30 m, so our ATD ball has no isolated spike at all. Twelve of the 19 runs were in the second-half
warm-up before kickoff or in other stoppages. The player rule now removes the frames around each
old core where the marker was still wrong (accelerating away or settling), 4,745 samples instead of
1,852. The legacy 0.5 m duplicate radius merged mostly genuine close teammates: our nearest-teammate distances
are higher than DFL's at every range from 0 to 2 m, without a separate near-zero mode, and in frames
where a team has more than 11 objects the extra object is not closer to a teammate than usual. The
"more than 22 objects" frames are therefore not same-position duplicates (team counts such as 16
against 7 suggest slots carrying a player of the other team), which a distance rule cannot fix and
which we do not attempt to fix (no re-identification, see below).

## Duplicates, identity switches and slot continuity

- Duplicates are merged, not flagged: for each same-team pair of slots closer than
  `duplicate_distance_m` the earlier slot takes the mean position and the later slot is emptied
  (`cleaning.merge_duplicates`). This happens before every feature (nearest distances, counts,
  centroids, spread, defensive line, speeds) on our match and on the degraded external matches
  alike; the degradation injects duplicates with offsets measured on our match, so training sees
  the same artefact and the same merge.
- Identity switches are not repaired: no re-identification, no feature uses the identity of a slot
  over time. Every feature is computed per frame over the set of objects of a team (nearest,
  second nearest, centroid, spread, defensive line, counts, closest to the centre mark) or per team
  over time (dwell of the nearest team). The only use of one slot across frames is the per-slot
  velocity over ±`player_half_span` frames for the speed features; a switch that implies more than
  `player_max_speed_mps` is removed by cleaning, so a switch affects at most the speed of that slot
  in the frames next to it. The degradation fragments external players into slots with our slot
  run lengths, so the model is trained with the same discontinuities.

## Team labels of tracked objects (known limitation)

Aggregate check on our match (no frame dumps), live play = graded frames that the `learned`
submission labels H or A, after the detected kickoff (rules' kickoff detection):

- A team has more than 11 objects on 12.5 % of live frames. Mostly 12 against 10 or 9 (11.2 % of
  live frames, total at most 22: a player of the other team carried by the wrong team), rarely
  extra objects such as referees (1.2 %, total above 22). Episodes last a median 3.2 s, half of the
  affected frames are in episodes longer than 10 s.
- Per-object position against the team's shape does not separate mislabels: a per-frame two-team
  Mahalanobis assignment disagrees with the label for 36 to 38 % of the objects on DFL (true labels)
  and 37 % on ours, so no excess is detectable this way during live play.
- The object nearest the ball is in a team with more than 11 objects on 5.9 % of live frames with a
  ball, and an over-full team is not more often nearest the ball than usual (team 0: 57.1 % against
  55.5 %; team 1: 27.9 % against 44.5 %). With one wrong object in twelve, the nearest object is a
  mislabelled one on about 0.5 % of live frames.

Conclusion: it happens, but it rarely touches the ball-nearest features that decide possession, and
no data-driven rule can tell which of the twelve objects is wrong without appearance data (a shape
rule would drop a third of correctly labelled DFL objects). No mitigation is applied. The
degradation does not simulate these team swaps; that is a possible future addition.

## Not done

- No EM refinement of the decoder (durations, transitions) on our match. Only the dead-ball
  evidence of the ball-observation features can be re-estimated on the predicted match
  (`--adaptation`, above).
- No sub-states (controlled or loose). Three states only.

# Audit of numeric literals on the `learned` path

Scope: every module that produces the `learned` submission or prepares its training data:
`cleaning.py`, `geometry.py`, `features.py` (`threshold_free_features` and its helpers),
`learned/*`, `external/{bounds,stats,degrade,harmonise,prepare,match}.py`. Index arithmetic (0, 1,
-1 as first/last/previous/next, `+ 1` for inclusive ends, axis numbers, `[:, 0]` for x) and output
formatting widths are not listed. Classes: **meta** (metadata or challenge definition), **default**
(library default), **learned** (estimated from data), **CV** (chosen by cross-validation on external
data), **seed**, **struct** (mathematical or structural fact, no tuning freedom), **MAGIC** (hand-set).

| Location | Literal | Class | Note |
|---|---|---|---|
| `cleaning.py` (was) `BALL_MAX_SPEED_MPS` | 40 m/s | MAGIC, replaced | now `ball_max_speed_mps`, learned (35.7 m/s) |
| `cleaning.py` (was) `PLAYER_MAX_SPEED_MPS` | 12 m/s | MAGIC, replaced | now `player_max_speed_mps`, learned (10.8 m/s) |
| `cleaning.py` (was) `DUPLICATE_DISTANCE_M` | 0.5 m | MAGIC, replaced | now `duplicate_distance_m`, learned (0.041 m) |
| `cleaning.py` (was) `MAX_INTERPOLATION_GAP_FRAMES` | 8 frames | MAGIC, replaced | now `max_gap_frames`, learned (24) |
| `cleaning.py` (was) `OFF_PITCH_MARGIN_M` | 3 m | MAGIC, removed | the off-pitch flag fed no feature |
| `features.py` (was) `VELOCITY_HALF_SPAN` | ±2 frames | MAGIC, replaced | now `ball_half_span` / `player_half_span`, learned (1, 1); legacy keeps 2 |
| `features.py` (was) `PRESSURE_RADIUS_M` | 5 m | MAGIC, removed | was computed for `learned` but unused; also drove the degradation propensity (below) |
| `features.py` (was) `SPEED_PERCENTILE` | 90 | MAGIC, removed | computed for `learned` but unused |
| `features.py` (was) `SHAPE_WINDOW_S`, `HALFWAY_MARGIN_M` | 1 s, 1 m | MAGIC, moved to legacy | `learned` uses unsmoothed centroid velocity and no halfway count |
| `external/stats.py` (was) `crowd_and_edge` | players within 5 m | MAGIC, replaced | covariate is now the distance to the nearest player (threshold-free) |
| `external/stats.py` (was) `IRLS_ITERATIONS` | 50 | MAGIC, replaced | `LogisticRegression()` defaults (`stats.py:119`) |
| `external/degrade.py` (was) `uniform(0, DUPLICATE_DISTANCE_M)` | 0.5 m | MAGIC, replaced | offsets drawn from the duplicates measured on our match (`degrade.py:137`) |
| `learned/framefeatures.py` (was) `QUARTILES` | 0.25, 0.5, 0.75 | MAGIC, replaced | mean run length per state family (`framefeatures.py:123`) |
| `learned/framefeatures.py:53` `SHORT_QUANTILES` | (0.25, 0.5) | MAGIC, hand-set levels (the run lengths they are applied to are learned) | levels of the weighted quantiles of the training run lengths (alive states) that add two short rolling scales in the `short` and `touch` team-feature sets (`short_scales`, `framefeatures.py:224`). Not chosen by cross-validation; the "quartiles replaced" row above holds for the `core` scale only |
| `cleaning.py:101` `window = 64`, `cleaning.py:112` `window *= 4` | 64 frames, factor 4 | struct (search stride), no tuning freedom | initial size and growth factor of the look-ahead window in `displaced_run_end`; the window grows until the displaced run ends or the block ends, so the result does not depend on either value (only the speed of the search does) |
| `cleaning.py:104` | `/ 2` | struct | mean of the two merged positions |
| `external/bounds.py:20` `SHORT_AND_LONG` | 2 components | struct | the two populations of gaps (dropout, occlusion) being separated |
| `external/bounds.py:19` `MIXTURE_SEED` | 0 | seed | |
| `external/bounds.py:86` | median | struct | robust aggregate of per-match extremes (50 % breakdown) |
| `cleaning.py` `neighbour_deviation`, `displaced_runs` | two samples per side | struct | fewest samples that define a velocity to extrapolate |
| `external/bounds.py` `alive_neighbourhood` | five frames | struct | the sample and the two neighbours on each side the deviation uses |
| `cleaning.py` `valley_before` / `valley_after` | speed minimum | struct | outermost extent of the jump excursion, refined by `player_deviation_m`; no length or acceleration threshold |
| `external/bounds.py:113`, `:127` | half-span 1 | struct | smallest symmetric stencil, the bias reference and the search start |
| `external/bounds.py:123` | `(sigma * fps / h) ** 2` | struct | variance of a central difference of white noise, two axes |
| `external/stats.py:17` `MAD_TO_SIGMA` | 1 / Phi^-1(0.75) | struct | MAD-to-sigma factor of the normal law |
| `external/stats.py:18` | 6 | struct | variance of a second difference of white noise is 6 sigma^2 |
| `external/stats.py:83` | 3 | struct | a second difference needs three samples |
| `external/stats.py:107` | mean imputation | struct | frames with no tracked player get the mean nearest distance (neutral after standardisation) |
| `external/stats.py:119` | `LogisticRegression()` | default | scikit-learn 1.9.1 defaults |
| `external/degrade.py:123`, `:173` | attempt budgets | struct | loop guards proportional to the work requested; they bound run time, the target is met long before |
| `external/degrade.py:136` | 2 pi | struct | uniform direction of a duplicate offset |
| `external/degrade.py:156`, `:168` | ±1 frame | struct | two added gaps never touch (they would merge into one longer gap) |
| `external/harmonise.py:12` `EXACT_TOLERANCE` | 1e-9 | struct | floating-point guard on exact resampling positions |
| `external/harmonise.py:67` | 0.5 | struct | nearest-sample rounding |
| `external/prepare.py:49` `KEY_CHARS` | 24 | struct | cache-key length |
| `external/prepare.py:50` `DEFAULT_SEED` | 0 | seed | degradation seed |
| `learned/hsmm.py:8` `N_STATES` | 3 | meta | H, A, D |
| `learned/classifier.py:10` `THREADS` | 1 | struct | determinism |
| `learned/framefeatures.py` `centred_nanmean` | window `scale // 2` back, `scale - 1 - scale // 2` forward | struct | a window of `scale` frames around the frame, as the past and future means |
| `learned/model.py:27` `MODEL_SEED` | 0 | seed | |
| `learned/train.py` `EMISSION_SPANS` | 1, 2, 5, 10, 25, 50, 100 frames | CV | candidate grid; the value is chosen by nested LOMO CV |
| `learned/train.py` `STRIDES` | 5, 10, 25 frames | CV | starting grid; extended upward by the 1-2-5 rule in seconds while the largest stride is selected ("Viterbi `stride`" above) |
| `learned/train.py` `SWITCH_WEIGHTS`, `SERIES_MANTISSAS` | 0, 0.5, 1, 2 start; 1, 2, 5 times a power of ten | CV, (a) documented rule | starting grid; extended upward by the 1-2-5 rule while the largest weight is selected ("Switch weight" above) |
| `learned/train.py` `VARIANTS` | 6 training sets (3 by default) | CV | candidates; one is chosen by held-out evaluation |
| `external/sources.py` `SourceInfo.grouped_folds` (PFF) | held out in K contiguous groups, K = matches of the largest other source of the variant; rows every ceil(n / K)-th frame | struct | compute bound: a 64-match source costs as much as the largest other source (see "Training-set variants and selection") |
| `external/sources.py` `DIGEST_CHUNK_BYTES` | 1 MiB (`1 << 20`) | struct | read size when two PFF copies of equal size are hashed; no effect on any result |
| `learned/train.py` `MIN_MATCHES_PER_SOURCE` | 2 | struct | leave-one-match-out needs a training match for each held-out one |
| `learned/train.py` `--selection-score` | raw | CV | Metrica score used as the primary variant criterion; both reported |
| `learned/train.py` `BOOTSTRAP_DRAWS` | 1000 | default | vendored metrics' recommended draws for a final ranking |
| `learned/classifier.py` `match_weights` | 1 / rows per match, mean 1 | struct | equal weight per match |
| `learned/classifier.py` `spell_weights` | 1 / run length, same total per match, mean 1 | struct | equal weight per spell inside a match (`--spell-weights`) |
| `learned/classifier.py` `STOPPING_RULES`, `MIN_GROUPS` | default, team, both; 2 | struct | which models stop on held-out matches; leave one group out needs two groups |
| `learned/model.py` `EVERY_PEAK` | 0 | struct | impulse threshold that keeps every peak (`--every-peak`, and the fallback without labelled peaks) |
| `learned/train.py` `--frame-step` | 1 | struct | every frame (larger values only for smoke runs) |
| `learned/train.py` `--dead-features` | all / players | struct | candidate feature sets of the dead-ball model, chosen by the DEADBALL.md rule |
| `learned/adapt.py` `SAFETY_BOUND` | 20 | struct | EM run-time guard, set by `--adaptation`; the EM stops on its own convergence test, `converged: false` flags a hit |
| `learned/adapt.py` `MATCH_HALVES` | 2 | struct | folds of a single block: the two halves of a match |
| `learned/adapt.py` `share_tolerance` | sqrt(sum q(1 - q)) / n | struct | standard error of the posterior D share |
| `learned/adapt.py` observation model | `HistGradientBoostingClassifier` defaults | default | scikit-learn 1.9.1 |
| `learned/framefeatures.py` `LINE_GROUPS` (`--line-features`) | lines, keeper, ball; none by default, `lines` in the committed model | struct | candidate feature groups; chosen by the DEADBALL.md rule. No constant inside: signed distances to the pitch lines (pitch metadata), speeds of the cleaned tracking, durations in frames, the deepest tracked player as a keeper proxy |
| classifier hyperparameters, KDE bandwidth, mixture settings | library defaults | default | scikit-learn 1.9.1, SciPy |
| fps, pitch, graded blocks, periods | 25, 105.3 x 68.0 m, `games` | meta | |

Summary: 12 distinct hand-set constants (14 uses) were on the `learned` path. Replaced by values
estimated from data or by library defaults: ball and player speed bounds, duplicate distance (in
cleaning and in the degradation offsets), interpolation gap, velocity half-span, the IRLS iteration
cap (now `LogisticRegression` defaults), the quartiles of the core rolling scale (now the mean run length per
state family; the two `SHORT_QUANTILES` of the `short` and `touch` sets remain hand-set) and the 5 m crowd radius of the degradation propensity (now the nearest-player
distance). Removed from the path: the off-pitch margin, the 5 m pressure count and the 90th speed
percentile (unused), the 1 s shape smoothing and the 1 m halfway margin (kept for `rules` only in
`possession.params`). The legacy `rules` values in the first part of this file are not on the
`learned` path.
