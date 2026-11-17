# Dead-ball detection under domain shift

Problem: the `learned` model (dfl+metrica, run eb8430b6) assigns teams well on our match but
under-detects dead ball. External held-out D is fine (DFL D to D 95 %, Metrica 94 %). This file
records the investigation, the decision rule fixed before any candidate was compared, the
candidates and the decision. No number in this file was fitted, tuned or selected on BAR-ATH
labels: every number below comes from external labelled matches or from label-free aggregates of
our match. The
public score changed from 0.548 (v1) to 0.594 (v2) on 17 spells; that difference is within noise
and the options were chosen on external held-out cross-validation only.

Until 2026-11-03 the label-free diagnostics of our match (Smart Tagging set-piece windows,
restart team, phase-team agreement) were used as tie-breaks between candidates that tied on
external cross-validation (the tables of "Full-run results and decision" and "Switch prior on top
of (c) + touch" below). From 2026-11-03 they are an alarm only, never a tie-break ("Classifier
fit adopted").

## Decision rule (written before any candidate was compared)

Candidates are compared with the baseline (current method: one dead-ball model on all features,
no adaptation) on exactly these criteria, in this order.

1. **External guardrail (labelled, held out).** For each source (DFL raw, Metrica raw and fair),
   the candidate's mean held-out spellF1 must reach the baseline's interval lower bound (mean of
   the per-match 5-95 bootstrap lower bounds, the rule already used to tie training-set variants
   in `train.select_variant`). A candidate that fails on any source is rejected.
2. **Cross-source transfer (labelled, held out).** Train on one source, score the other (DFL to
   Metrica and Metrica to DFL). This is the only labelled measurement of robustness to a change of
   tracking provider, the kind of shift our match has. Reported as spellF1 and dead-ball recall.
3. **Our match, label-free plausibility.** Each item is a pass or a fail against the baseline:
   1. Smart Tagging set-piece windows holding a predicted restart (higher is better);
   2. share of frames with the ball observed wholly over a line (centre beyond the line by the
      ball radius, Law 2: circumference 68 to 70 cm, radius 0.11 m) that are predicted D
      (higher is better; Law 9 makes those frames dead);
   3. predicted D share inside the range of D shares of the external matches (DFL and Metrica
      labels);
   4. predicted stoppages per graded minute inside the external range;
   5. source stability: agreement of our-match graded labels between the candidate trained on
      DFL only and the candidate trained on Metrica only (a method robust to the provider should
      give the same labels whatever the provider of its training data; higher is better).
4. **Choice.** Among candidates passing 1, the one passing most items of 3; ties go to the better
   cross-source spellF1 of 2 (mean of both directions), then to the simpler candidate (fewer
   moving parts).

### Team decision after the quick experiments (2026-10-26, before any full run or public-GT look)

The quick experiments put candidate (c) below the DFL guardrail of rule 1. The team amended the
rule for the full runs, and records why: the team's priority is the source closest to the target,
Metrica, and the DFL loss is attributed to DFL's frozen-ball cue, which our match does not have.

Rule for the full runs, replacing rules 1 and 4 above:

1. Both the current method and candidate (c) are trained in full (all frames, full LOMO).
2. The winner is the one with the higher mean held-out Metrica raw spellF1, unless the two are tied:
   tied means each one's mean reaches the other's interval lower bound (same bound as rule 1).
3. On a tie, the winner is the one passing more items of rule 3 (our match, label-free); a further
   tie goes to the better cross-source spellF1 of rule 2, then to the simpler candidate.
4. The public window is not consulted to choose.

Quick experiments run on subsampled training frames (every fifth frame) with the decoder setting of
the current model (span 2, stride 5). They are marked as such. The final numbers come from the full
`train` run, which applies the same rule with full LOMO.

## Investigation (aggregates only)

Data: our match, graded frames (158,857), cleaned with the estimated parameters; the 7 DFL and 3
Metrica matches degraded with the current degradation (seed 0). Features are the `learned` base
features from `framefeatures.base_features`. Our match has no labels, so its state is described
by label-free proxies only: the `rules` output (a weak proxy, not truth), Smart Tagging set-piece
windows, and frames where the ball is observed wholly over a line.

### 1. The ball observation process during dead ball is provider-specific

| | P(ball missing \| D) | P(ball missing \| alive) | ball observed over a line, share of D frames |
|---|---|---|---|
| DFL raw | 0.000 | 0.000 | 4 to 15 % of all frames, 98 to 99 % of them D |
| DFL degraded | 0.46 to 0.54 | 0.30 to 0.35 | 14 to 33 % |
| Metrica raw | 0.92 to 0.99 | 0.003 to 0.05 | under 0.3 % |
| Metrica degraded | 0.92 to 0.99 | 0.03 to 0.13 | under 0.3 % |
| ours, `rules` D as proxy | 0.71 | 0.16 | ball over a line on 2.4 % of graded frames |

- DFL never loses the ball, and while the ball is dead its position is frozen: ball speed on DFL
  dead frames has median 0.06 m/s and 90th percentile 0.11 m/s (alive: median 6.0 m/s). Ball
  speed separates alive from dead almost perfectly on DFL (KS 0.98), but only because of that
  convention. Our degradation removes ball samples with a propensity on nearest-player and goal-line
  distance, which does not depend on the state, so the frozen ball survives on about half the DFL
  dead frames.
- Metrica's tracking drops the ball when it is dead: "ball missing" is almost the label itself
  (KS 0.89 between alive and dead). Its few observed off-pitch balls are mostly labelled alive
  (P(D | over a line) 0.07 to 0.15), a coordinate or event-timing artefact.
- On our match the ball is neither frozen nor always removed. Where the `rules` proxy says dead
  and the ball is visible, it moves (median 2.0 m/s, upper quartile 4.5 m/s: thrown back, carried,
  rolling); the visible-ball speed distribution of our match looks like DFL *alive* (best mixture
  weight of DFL dead: 0.05). The current model therefore reads every visible, moving ball as alive
  play, and its D is essentially "ball missing": P(ball missing | learned D) = 0.96, and of the
  frames with the ball observed over a line, `learned` labels 5 % D (`rules`: 88 %).
- This explains the aggregate public-window check without looking at it again: the ball is visible
  on 88 % of that window against 59 % on the match, so a D detector that keys on ball loss misses
  most of its stoppages.

### 2. Player kinematics are not shifted

For each feature, our match's distribution was compared with the external alive and dead
distributions, and the dead-ball weight pi of the best mixture pi F_dead + (1 - pi) F_alive was
fitted (KS residual of the best fit reported).

| feature | KS alive vs dead (DFL / Metrica) | best pi (DFL / Metrica) | residual (DFL / Metrica) |
|---|---|---|---|
| mean player speed | 0.69 / 0.64 | 0.54 / 0.46 | 0.01 / 0.02 |
| team speed (each team) | 0.66 / 0.60 | 0.53 to 0.55 / 0.47 | 0.01 to 0.02 / 0.02 to 0.03 |
| centroid velocity, shape | 0.12 to 0.21 | 0.37 to 0.58 | 0.01 to 0.02 |
| ball speed | 0.98 / 0.66 | 0.05 / 0.13 | 0.05 / 0.07 |
| ball acceleration | 0.78 / 0.18 | 0.19 / 0.00 | 0.07 / 0.45 |
| ball present, gap features | 0.17 / 0.89 | 0.45 / 0.38 | 0.00 to 0.03 |
| tracked objects (each team) | 0.04 to 0.07 | 0 or 1 (no fit) | 0.14 to 0.24 |
| spread in x (each team) | 0.10 to 0.14 | 1.0 (no fit) | 0.06 to 0.16 |

- Player speeds are the strongest provider-independent dead-ball signal. Our match is reproduced
  by a mixture with a dead share of 0.46 to 0.54 with a residual of 0.01 to 0.03, and its quantiles
  under the `rules` proxy match the external ones (dead median 0.98 m/s against 1.02 DFL and 1.01
  Metrica; alive 2.10 against 2.27 and 2.06).
- Ball features fit a mixture only on the "ball present" margin, which the degradation matches by
  construction. Their conditional behaviour (above) differs by provider.
- Object counts are shifted in a way the state does not explain: our tracker loses players during
  stoppages (10th percentile 6 objects in `rules`-dead frames against 9 in DFL), which the
  degradation (thinning to our mean count, independent of state) does not simulate. They carry
  almost no dead-ball information on external data (KS 0.03 to 0.07). Team spread is computed
  from the remaining objects and inherits part of this shift.

### 3. Is the degradation simulator unrealistic for dead ball?

Yes, for the ball: missingness is drawn independently of the state, and DFL's frozen dead ball is
kept as it is. Making it realistic (candidate a) would require P(ball missing | D, alive) on our
match and a model of how a visible dead ball moves. The first can be estimated without labels
(candidate c below does exactly that, as a by-product); the second cannot be simulated from DFL,
whose dead-ball trajectories are frozen. Candidate (a) was therefore not implemented: it would
replace one provider's convention with a hand-made one.

## Candidates

- **Baseline**: one dead-ball model on all 69 features (`--dead-features all`, the default).
- **(b) Player features only** (`--dead-features players`): the dead-ball model sees only player
  kinematics and team shape (24 of the 69 columns with the current scales, list in `docs/PARAMETERS.md`). The team model is unchanged.
- **(c) Player features plus unsupervised adaptation** (`--dead-features players --adaptation`,
  `possession.learned.adapt`): on each predicted match, EM re-estimates the log likelihood ratio of
  the ball and object-count features for dead against alive, anchored on the player model and
  cross-fitted between the match's two halves (a single block is split in two contiguous halves, so
  no frame is scored by a model fitted on it). Initialised from the player model, run until the D
  share moves by less than its own standard error, with a documented safety bound on the number of
  iterations. Each iteration sets q = expit(player logit + [logit(target share) - logit(training
  share)] + log ratio), where the bracket is the prior correction of Saerens et al. (2002) with the
  target share estimated by the mean of the previous q. The emitted logit is built the same way
  from the final q: the log ratio is refitted on the final q and the correction uses its share, so
  the emission and the reported share describe the same posterior. Each observation model is fitted
  once per frame on soft labels (rows duplicated as D with weight q and alive with weight 1 - q)
  with early stopping disabled, because the internal validation split of early stopping would
  place the two copies of a frame on both sides. Shares are bounded to [1/(n+1), n/(n+1)] before
  the logit. Every adapted match logs its iterations, convergence and D share to stderr and the
  report is stored with the held-out rows of `variant_report.json`.
- **(c') Viterbi-training variant of (c)**: the E-step uses the HSMM-decoded labels instead of the
  frame posteriors. Tried and dropped: it never reached a fixed point within 10 iterations, its D
  share drifted downwards on every match (our match 0.47 to 0.45, Metrica game1 0.46 to 0.40) and
  it lowered the off-pitch D share on our match from 0.76 to 0.38 to 0.47: confirmation bias.
- **(a) Degradation realism**: not implemented (section 3 above).

## Quick-experiment results (QUICK: every fifth training frame, decoder span 2 stride 5)

The (c) rows were produced before the adaptation was changed to apply the prior correction to the
emitted logit, to cross-fit a single block, and to fit the observation model without early
stopping, and with a cap of 6 iterations; the full run replaces them.

Held-out pairs: the model is trained on the other 8 matches and scores one DFL and one Metrica
match; six pairs cover all 7 DFL matches but J03WOY (scored in an earlier pair: baseline 0.582, (c)
0.534) and every Metrica match twice. The decoder setting selected over all of them is span 2,
stride 5 for every candidate (the same as the current model), so the comparison does not depend on
it.

| QUICK, mean held-out spellF1 | DFL raw | Metrica raw | Metrica fair |
|---|---|---|---|
| baseline | 0.570 | 0.689 | 0.637 |
| baseline guardrail (mean of per-match 5 % bootstrap bounds, about mean - 0.040) | about 0.530 | about 0.649 | about 0.597 |
| (b) players | 0.498 | 0.611 | 0.567 |
| (c) players + adaptation | 0.513 | 0.652 | 0.602 |

Cross-source transfer (QUICK, all matches of the other source):

| | DFL to Metrica: raw spellF1 / D recall | Metrica to DFL: raw spellF1 / D recall |
|---|---|---|
| baseline | 0.644 / 0.888 | 0.385 / 0.611 |
| (b) players | 0.593 / 0.901 | 0.452 / 0.843 |
| (c) players + adaptation | 0.641 / 0.948 | 0.472 / 0.866 |

Our match, label-free (QUICK, model trained on all 10 matches; external ranges from the labels of
the 10 matches: D share 0.32 to 0.52, stoppages 0.94 to 1.19 per graded minute):

| | D share | stoppages (per min) | set-piece windows with a restart | ball over a line predicted D | source stability |
|---|---|---|---|---|---|
| baseline | 0.375 | 132 (1.25, outside) | 42/59 | 0.09 | 0.889 |
| (b) players | 0.473 | 112 (1.06) | 47/59 | 0.76 | 0.924 |
| (c) players + adaptation | 0.473 | 109 (1.03) | 47/59 | 0.66 | 0.937 |

The EM estimate on our match, P(D) about 0.48, is close to the mixture weight that player speeds
alone imply (0.46 to 0.54, section 2).

## Decision (rule applied as written)

1. Guardrail: (b) fails on DFL and on Metrica. (c) passes Metrica raw (0.652 against about 0.649)
   and fair (0.602 against about 0.597) but **fails on DFL** (0.513 against about 0.530). Both are
   rejected by item 1.
2. Cross-source and 3. plausibility favour (c) clearly (it passes all five plausibility items, the
   baseline fails item 4 and is worst on 1, 2 and 5), but item 1 comes first.
4. **Outcome of the QUICK comparison: keep the baseline as the default.** The candidates stay
   available behind `train` flags so that the full run can apply the same rule with full LOMO and
   the nested decoder choice; if the full run puts (c) inside the DFL guardrail, (c) is chosen by
   items 2 and 3.

What the rule cannot see: the DFL loss is where the baseline exploits DFL's frozen dead ball
(ball speed separates states with KS 0.98 on DFL only), and the investigation shows that this cue
does not exist on our match. Changing the rule after seeing these numbers would be selection on
the results, so the rule is not changed here; whether to accept a DFL loss of about 0.06 in
exchange for the gains on our match is a team decision, to be taken and logged before anyone looks
at the public window again.

## Full comparison (to be run by the team)

```bash
.venv/bin/python -m possession.external.prepare --prune
.venv/bin/python -m possession.learned.train
.venv/bin/python -m possession.learned.train --dead-features players --adaptation --out models/learned_players/model.pkl
.venv/bin/python -m possession.external.validate --method learned
.venv/bin/python -m possession.external.validate --method learned --model models/learned_players/model.pkl
.venv/bin/python -m possession.external.analyse --method learned
.venv/bin/python -m possession.external.analyse --method learned --model models/learned_players/model.pkl
.venv/bin/python -m possession.run --config configs/learned.json --out outputs/learned.csv
.venv/bin/python -m possession.run --config configs/learned.json --model models/learned_players/model.pkl --out outputs/learned_players.csv
.venv/bin/python src/possession/vendor/validate_submission.py outputs/learned.csv
.venv/bin/python src/possession/vendor/validate_submission.py outputs/learned_players.csv
.venv/bin/python -m possession.diagnostics --submission outputs/learned.csv --submission outputs/learned_players.csv
```

Apply the decision rule above to `models/learned/variant_report.json` and
`models/learned_players/variant_report.json` (item 1: `summary` of the selected variant, DFL raw
and Metrica raw and fair intervals) and to the diagnostics (item 3). Source stability (item 3.5)
needs the `dfl` and `metrica` variants' final models of each run, both in
`models/<dir>/folds/<variant>/final.pkl`.

## Full-run results and decision (2026-10-26)

Both methods were trained in full (all frames, full LOMO), variant `dfl+metrica`. W is the previous method
(`models/learned_w/`); (c) is `--dead-features players --adaptation` (now `models/learned_c/`). Held-out
numbers come from `train`'s LOMO. Validation of the final (c) model on external matches was stopped (EM takes
about 4-5 minutes per match), so no separate `validate` run exists for it. Train runtime: W 1171 s, (c) 9524 s.

| Held-out spellF1 | W | (c) |
|---|---|---|
| Metrica raw (mean [lower, upper]) | 0.686 [0.646, 0.725] | 0.663 [0.623, 0.704] |
| DFL | 0.571 | 0.512 |
| SkillCorner fair | 0.287 | 0.301 |
| Metrica fair | 0.638 | 0.613 |

Rule 2 gives a tie on Metrica raw (each mean reaches the other's lower bound), so rule 3 applies, on our match
(label-free):

| Item | W | (c) | Pass |
|---|---|---|---|
| Set-piece windows holding a restart | 43/59 | 46/59 | (c) |
| Ball wholly over a line predicted D (3832 graded frames, 2.41%) | 0.085 | 0.695 | (c) |
| D share (external range 0.32-0.52) | 0.374 | 0.478 | both inside |
| Stoppages per graded minute (105.9 min; range 0.94-1.19) | 1.36 (144) | 0.91 (96) | both outside |
| Source stability | not computed | not computed | cannot change the outcome |

(c) passes two items, W none, so the winner is (c). Source stability was not computed because it cannot change
the outcome. The public window was not consulted.

Other aggregates on our match: (c) 284 spells, mean 11.68 s, median 5.88 s, phase-team agreement 79.2%,
restart team equal to the set-piece team in 16/22; W 391 spells, phase-team agreement 75.1%, 18/20.
Adaptation of (c) on our match: 2 iterations, converged, D share 0.4773 to 0.4821.

`models/learned_c/` holds (c). Its our-match CSV from `configs/learned.json`: sha256
`41c1f74bba074dc4a729e0617aad0d3d45e8bfa7d6be36febbcee7b0e5016fa2`.

## Touch features on top of (c) (2026-10-26)

(c) was retrained with the touch-event team features (`--team-features touch`, `docs/PARAMETERS.md`), same
variant, dead-ball settings and LOMO. Train runtime 9954 s.

| Held-out spellF1 | (c) | (c) + touch |
|---|---|---|
| Metrica raw (mean [lower, upper]) | 0.663 [0.623, 0.704] | 0.738 [0.698, 0.777] |
| Metrica fair | 0.613 | 0.706 |
| DFL | 0.512 | 0.542 |
| SkillCorner fair | 0.301 | 0.312 |

The mean of (c) does not reach the lower bound of (c) + touch, so there is no tie and (c) + touch wins on rule 2.
Our match, label-free: 253 spells (mean 13.10 s, median 7.60 s; (c) 284), D share 0.478, 96 stoppages,
set-piece windows holding a restart 46/59, restart team equal to the set-piece team 17/22, phase-team agreement
79.2%. Fewer short spells than (c) on our match (0-3 s: 59 against 89), likely because touches need the ball,
which is missing on 41% of graded frames; not used to choose.

`models/learned/` then held (c) + touch (sha256 `9f922b56776c91d939feac0a876781ffe541612cca4ddd724a23acd65671cca5`).
Our-match CSV from `configs/learned.json`: sha256 `e704007fbaec63948e5e8a540b685c89d38136ba856af06594bc8ff4e6038072`.

## Switch prior on top of (c) + touch (2026-10-27)

(c) + touch was retrained with the touch-based switch prior (`docs/PARAMETERS.md`, "Switch prior"), same variant,
dead-ball settings and LOMO; model format `learned-v8`. The switch weight `w_switch` is chosen from {0, 0.5, 1, 2}
by the nested LOMO jointly with span and stride; 2.0 was selected in the final model and in every nested fold
(10 of 10; the 20 further SkillCorner rows of `variant_report.json` are scored by the final model, not nested folds),
which is the top of the grid. The stride 25 was also selected in the final model and in 10 of 10 folds, the top of the
stride grid; later runs extend both grids (`docs/PARAMETERS.md`, "Switch weight" and "Viterbi `stride`"). Its threshold is the touch impulse threshold learned from the training
matches (11.84 in the final model). Train runtime 11667 s (`outputs/logs/switch/01.log`; our-match diagnostics in
`outputs/logs/switch/04.log`, previous model in `outputs/logs/touch/`).

| Held-out spellF1 | (c) + touch (v7) | (c) + touch + switch prior (v8) |
|---|---|---|
| Metrica raw (mean [lower, upper]) | 0.738 [0.698, 0.777] | 0.761 [0.721, 0.799] (better on 3/3 matches) |
| Metrica fair | 0.706 | 0.730 (better on 3/3) |
| DFL | 0.542 | 0.534 (better on 1/7) |
| SkillCorner fair | 0.312 | 0.300 (better on 4/20) |

The intervals overlap (each mean reaches the other's lower bound), so rule 2 gives a tie and rule 3 applies, on our
match (label-free):

| Item | v7 | v8 |
|---|---|---|
| Set-piece windows holding a restart | 46/59 | 48/59 |
| Restart team equal to the set-piece team | 17/22 | 17/22 |
| Phase-team agreement | 79.2% | 79.1% |
| Spells (0-1 s spells) | 253 (8) | 278 (25) |

The prior wins one item, ties one and loses one by 0.1 point, and adds spells (mean 11.84 s, median 6.10 s;
stoppages 91, D share 0.482), mostly very short ones (0-1 s: 8 to 25). By the agreed rule the new model is adopted.

Caveats. The Metrica gain is measured on labels that our own `metrica_events.py` derives from the event file, so
it is partly circular on the timing of switches, which is what the prior targets. DFL (better on 1 of 7) and
SkillCorner fair (4 of 20) lose a little. The selected weight is at the upper edge of the
grid, so a larger weight was not tried.

`models/learned/` then held (c) + touch + switch prior (sha256
`9ef5207686ded6df81f9f1c32ebccbdfec3f767f4b0e3125a68824bb581cc894`). The previous v7 model is not kept: it no
longer loads with this code. Our-match CSV from `configs/learned.json`:
sha256 `b5bf2b4d632a3195b83167acc689945d57edb241a671cc8c2d09d59b78d5b1e9`.

That sha256 is the macOS arm64 output. From 2026-11-02 the canonical file is the Docker one (sha256 `5313ec250b6a5432875654a0527e2e19abf96d87ad9bbb37477b4e4acf9688fb`, `docs/COMPLIANCE.md` item 7).

## Grid extension check (2026-11-02)

Both `w_switch` 2.0 and stride 25 sat at the top of their grids, so the grids were made to extend along the
1-2-5 series (`docs/PARAMETERS.md`) and the same model was retrained (`scripts/phase_switch2.txt`, logs in
`outputs/logs/switch2/`, train runtime 13428 s). Every nested fold and the final selection searched up to
`w_switch` 5.0 and stride 50; none of the added settings was strictly better, so the selection stays span 5,
stride 25, `w_switch` 2.0, now inside the searched grid on both axes. The retrained model and our-match CSV are
byte-identical to the ones above (same sha256); `models/learned/model.json` and `variant_report.json` now record
the extended grid.

## Team-model fit (planned, 2026-11-02)

An oracle error analysis on the held-out fold emissions of the current model (no BAR-ATH labels) puts the
remaining loss in the team model, not in the decoder: replacing the team split of the emission by the true team
at the classifier's median confidence raises held-out spellF1 by 0.16 (Metrica fair) and 0.28 (DFL), while
durations refitted on the held-out match's own labels, other strides, switch weights or sharper emissions gain
0.01 at most. Most of the loss is true spells of 1 to 3 s that the team model never or barely favours.

Three options of `train` target it (`docs/PARAMETERS.md`): `--spell-weights` (team rows weighted
1 / run length, so every spell of a match weighs the same), `--every-peak` (team touch features from every
impulse peak; the switch prior keeps its Youden threshold) and `--stopping team` (team-model iterations by early
stopping on whole held-out training matches, then a refit on the whole fold).

QUICK evaluation (every fifth training frame, the 10 LOMO folds, `possession.learned.model.fit_model` as in `src/`; the
held-out match's adapted dead-ball logit and decoder settings are those of the current fold model, so only the
team model and the switch prior change). Mean held-out spellF1 change against the same QUICK fit without the
options, with the folds that improve:

| Option | Metrica fair | Metrica raw | DFL |
|---|---|---|---|
| `--every-peak` | +0.015 (3/3) | +0.016 (3/3) | +0.022 (7/7) |
| `--spell-weights` | +0.010 (2/3) | -0.001 (2/3) | +0.007 (5/7) |
| `--stopping team` | +0.001 (1/3) | +0.001 (1/3) | +0.000 (3/7) |
| all three | **+0.050 (3/3)** | +0.026 (3/3) | +0.021 (6/7) |
| all three, switch prior on every peak too | +0.028 (3/3) | +0.005 (2/3) | +0.017 (6/7) |

All three against the current model's held-out scores: Metrica fair 0.784 against 0.730 (3/3), raw 0.792 against
0.761 (3/3), DFL 0.569 against 0.534 (6/7). The selected team iterations were 77 to 140 (validation groups: the
matches of each source dealt round-robin into K groups, K = matches of the smallest source, so 2 groups when a Metrica
match is held out and 3 when a DFL match is). The stop of a group is its best iteration, not the iteration at which
the stopping rule fired (10 later, `n_iter_no_change`): the first version of this table used the latter and its
numbers were 10 iterations higher, with all three at Metrica fair +0.046, raw +0.021, DFL +0.022 and `--stopping team`
alone at +0.001 / +0.004 / -0.001. `--stopping both` (the dead-ball model also stops on held-out matches, 98 to 172
iterations) was compared without adaptation: Metrica fair -0.009, DFL -0.001 against `team`, so it is not used. A gap-exit
re-anchoring of the decoded switches (move a switch decoded just after a ball gap, with no touch in between, back
to the gap start or middle) lost on both sources (Metrica fair -0.006 / -0.005, DFL -0.004 / -0.002) and was not
implemented.

The full run (`scripts/phase_cls.txt`, model `models/learned_cls/`) applies the rule of "Team decision after the
quick experiments" against the current model; the public window is not consulted.

## Classifier fit adopted (2026-11-03)

The full run of the team-model options (`scripts/phase_cls.txt`: `train --variants dfl+metrica --dead-features players
--adaptation --team-features touch --spell-weights --every-peak --stopping team`; logs `outputs/logs/cls` and
`outputs/logs/cls2`; the first attempt stopped when the laptop went to sleep and the second reused its 5 cached folds)
replaces the previous final model. The new model (the earlier internal model v2) was `models/learned/` (run id
`e5cfc0734f985b2d9f9f48419f29507a6ef65d2d4e5023468b525422c85e78ac`, `model.pkl` sha256
`4075fdf9d1a5ee611115990b9022dacae8b4d85a7f7c091e15986417ff4b8b5a`); the previous one (v1) is not kept in the repository
(`9ef5207686ded6df81f9f1c32ebccbdfec3f767f4b0e3125a68824bb581cc894`, run id
`330905fee75f290ebd6f60694018adbac617c4b348745f974c196c7ea2f19156`).

Held-out spellF1 (mean over the held-out matches of each source, 5-95 bootstrap interval in brackets), old model to new:

| Source | Old | New |
|---|---|---|
| Metrica raw | 0.761 [0.721-0.799] | 0.790 [0.754-0.825] |
| Metrica fair | 0.730 | 0.773 |
| DFL | 0.534 | 0.573 |
| SkillCorner fair (12 of 20 matches improve) | 0.300 | 0.302 |

The new model is better on 10 of 10 nested folds. The selected decoder changed from emission weight 0.2, stride 25,
switch weight 2.0 to emission weight 0.2, stride 10, switch weight 0.5.

Decomposition (external folds only; no BAR-ATH data). Mean held-out spellF1 as Metrica raw / Metrica fair / DFL, and
the number of the 10 folds (3 Metrica scored on fair, 7 DFL on raw) in which the variant beats the old model:

| Variant | Metrica raw | Metrica fair | DFL | Folds beating the old model |
|---|---|---|---|---|
| Old model | 0.761 | 0.730 | 0.534 | |
| New classifier, old decoder | 0.780 | 0.771 | 0.555 | 8 of 10 |
| Old classifier, new decoder | 0.756 | 0.719 | 0.535 | 5 of 10 |
| Equal-weight log-emission average of both classifiers, decoder selected by nested CV | 0.772 | 0.743 | 0.547 | 9 of 10 (beats the new model in 1 of 10) |
| New model (adopted) | 0.790 | 0.773 | 0.573 | 10 of 10 |

Most of the gain comes from the classifier; the new decoder alone does not help and adds to the classifier gain.
Averaging the two classifiers does not beat the new one. Error breakdown against the ground truth of the external matches: the new model
hits more true spells (DFL 1542 against 1425 of 3061, Metrica fair 727 against 657 of 984) and places the switches
closer (mean absolute switch offset DFL 27.1 against 31.9 frames, Metrica 34.2 against 41.5 frames).

Decision-rule change (team decision, 2026-11-03). The model is chosen on external held-out cross-validation only.
The diagnostics on our match (restart and phase-tag agreement, `possession.diagnostics`) are an alarm, never a
tie-break; this supersedes item 3 of "Decision rule" above for this and any later comparison. They are recorded here
as noted, not used: the new model is slightly lower on them (phase-team agreement 74.8 % against 79.1 %, restart team
16 of 22 against 17 of 22, set-piece windows holding a predicted restart 48 of 59 for both, 312 against 278 spells).
Noted, not used.

The public window was not consulted for this decision.

Our-match CSV from `configs/learned.json`: canonical Docker file sha256
`8e3821a3691cefba2bf703f6303ae60586dfdf693b4c43ecedf17334aa212ffb`, native macOS arm64 sha256
`53d82066c9ca28110e87bfa3e7118d6d57d5ac659b82ae4eeb8c7f7095b54e01` (210 of 199,319 rows differ, as with the
previous model; `docs/COMPLIANCE.md`, item 6). The previous Docker file
(`5313ec250b6a5432875654a0527e2e19abf96d87ad9bbb37477b4e4acf9688fb`) is superseded.

## Dead ball by stoppage type and pitch-line features (2026-11-03)

Two suspected failure patterns: dead balls over the lines missed or detected late, throw-ins above all (A); a
goalkeeper holding the ball labelled D (B). The pitch-line features below (v3) lost on external CV and were not
adopted then; on top of the dead-ball context model they were adopted on 2026-11-13 ("Pitch-line features adopted
(2026-11-13)"). Everything below uses external held-out folds or label-free aggregates of our
match; no public ground truth, no hand labels.

### 1. D recall by stoppage type (final model, held-out folds, degraded external matches)

Each ground-truth D run (not the first of a period) is typed by the restart event that ends it: DFL events
(`ThrowIn`, `GoalKick`, `CornerKick`, `FreeKick`, `KickOff`, `Penalty`, `RefereeBall`, event time to frame by the
first tracking frame of each half; the matching window is the 90th percentile of |event - run end|, 2.5 to 3.8 s),
Metrica events (`SET PIECE` subtypes and `PASS GOAL KICK`, exact frames). Recall is the share of ground-truth D frames
predicted D; "runs found" is the share of runs predicted D on at least half of their frames.

| | DFL n | recall | runs found | Metrica n | recall | runs found |
|---|---|---|---|---|---|---|
| throw-in | 300 | 0.81 | 0.68 | 109 | 0.79 | 0.62 |
| goal kick | 110 | 0.90 | 0.85 | 28 | 0.94 | 0.82 |
| corner | 56 | 0.97 | 1.00 | 25 | 0.99 | 1.00 |
| free kick | 159 | 0.94 | 0.90 | 89 | 0.94 | 0.85 |
| kick-off | 25 | 0.97 | 0.84 | 12 | 1.00 | 1.00 |

Throw-ins are 53 % (DFL) and 57 % (Metrica) of all missed D seconds (free kicks 19 and 31 %, goal kicks 17 and 10 %).
Timing is not the problem: the median onset and end offsets of the predicted D run are 0.0 s and 0.0 s (DFL) and 0.0 s
and 0.4 s (Metrica) for throw-ins; the first and last thirds of a run are found a little less often than the middle
(DFL throw-in 0.63 / 0.68 / 0.66). The misses are whole throw-ins labelled alive.
Mechanism: during throw-in D frames a tracked player is beyond a touch line on 70 % (DFL) and 55 % (Metrica) of the frames
against 3 % and 5 % on alive frames (the thrower stands outside), still 61 % and 49 % on the missed D frames; goal-kick D
frames have a player beyond a goal line on 17 % and 13 % against 0.6 % and 1.8 %. No existing feature sees it (the ball
distances to the lines existed; no player-based line feature).

Our match, aggregates only (final submission, 59 Smart Tagging set-piece windows): windows with no D at all 5 (4 throw-ins
of 26, 1 goal kick of 10), windows with no dead run ending inside 11 (throw-in 5 of 26, goal kick 3 of 10, corner 1 of 12,
free kick 1 of 5, kick-off after goal 1 of 4). So throw-ins are 5 of the 11 but 26 of the 59 windows: by rate goal kicks
are not better. A tracked player is beyond a touch line on 5.2 % of predicted-alive frames and 21 % of throw-in window
frames, so the cue exists in our tracking too.

### 2. Candidate features and quick evaluation

Three optional groups (`train --line-features`, `docs/PARAMETERS.md`, "Pitch-line feature groups"): `lines`
(player beyond a touch or goal line, speed of the player nearest a line, rolling means), `keeper` (depth, speed and
distance to the last seen ball of each team's deepest player, a proxy for the goalkeeper because there are no
identities) and `ball` (time since and until the ball is beyond a line, ball distance to the lines at the ends of a gap).
The position-only columns are player features and feed the dead-ball model, the ball-based ones go to the adaptation and
the team model.

QUICK: every fifth training frame, 10 nested folds, real `fit_model` with the final configuration (`--dead-features
players --adaptation --team-features touch --spell-weights --every-peak --stopping team`) and the real EM adaptation,
decoder settings of each fold of the final model; same harness for all rows. Mean held-out spellF1, difference to
the baseline and folds better (Metrica 3, DFL 7):

| QUICK | Metrica fair | Metrica raw | DFL | throw-in recall DFL / Metrica | goal-kick recall DFL / Metrica | false D in gaps >= 251 frames, Metrica / DFL |
|---|---|---|---|---|---|---|
| baseline (final configuration) | 0.763 | 0.772 | 0.575 | 0.82 / 0.79 | 0.89 / 0.95 | 0.070 / 0.107 |
| `lines` | 0.773 (+0.010, 2/3) | 0.792 (+0.020, 3/3) | 0.594 (+0.020, 6/7) | 0.89 / 0.88 | 0.90 / 0.98 | 0.058 / 0.112 |
| `keeper` | 0.748 (-0.015, 1/3) | 0.765 (-0.007, 1/3) | 0.586 (+0.011, 7/7) | 0.83 / 0.76 | 0.91 / 0.94 | 0.063 / 0.122 |
| `ball` | 0.763 (+0.000, 2/3) | 0.785 (+0.013, 3/3) | 0.575 (+0.000, 4/7) | 0.80 / 0.90 | 0.87 / 0.91 | 0.069 / 0.094 |
| `lines` + `keeper` | 0.762 (-0.000, 1/3) | 0.773 (+0.001, 1/3) | 0.597 (+0.023, 7/7) | 0.90 / 0.85 | 0.91 / 0.94 | 0.064 / 0.107 |
| `lines` + `ball` | 0.765 (+0.002, 2/3) | 0.788 (+0.016, 3/3) | 0.592 (+0.017, 6/7) | 0.89 / 0.95 | 0.89 / 0.94 | 0.070 / 0.089 |

`lines` alone is the only candidate with Metrica fair up, DFL up and most folds better (Metrica fair: game1 +0.021, game3
+0.022, game2 -0.012). The goalkeeper group does not help on the external folds and does not reduce the false-D share in
long ball gaps (gaps >= 251 frames). The goalkeeper-holding pattern (B) is therefore not fixed by this change; the external DFL events
have keeper catches that could be used to measure a dedicated feature in a later step, none was found to help here.
`lines` is available behind the flag, off by default; the full run is `scripts/phase_v3.txt` (not adopted, "Experiments
since v2 that did not win"; adopted later on top of the dead-ball context model, "Pitch-line features adopted
(2026-11-13)"). The quick numbers
replace nothing: the decision to adopt is the team's, on the full nested run.

### 3. Smart Tagging time base (our match, aggregates)

Question: are the Smart Tagging set-piece windows shifted against the physical restarts in the tracking? No offset can be derived from the data,
and the mapping `round(seconds * 25)` shows no sign of error:

- Kick-offs have a physical anchor (the ball leaves the centre spot in the tracking; the first-half and second-half
  anchors of `games` agree with it within a frame). The six kick-off windows start 3.1 to 3.6 s before the ball leaves
  the centre (77 to 89 frames, 2 first/second-half anchors 89 and 81 frames, 4 after goals 77 to 87).
- Corners: the ball is last observed at the corner spot 7.0 s after the window start (IQR 6.3 to 7.1 s, 5 corners with the
  ball tracked there).
- Over all 46 non-kick-off windows with a restart within 20 s, our restart (end of the predicted D run) minus the window
  start has median +6.9 s (bootstrap 95 % CI 5.8 to 7.7), +7.1 s in the first half and +6.7 s in the second, slope
  -0.033 s per minute (CI -0.088 to 0.016, i.e. no drift over 90 minutes).

So a window starts a stable 3 to 7 s before the restart and ends about 28 s after: the windows contain the restart rather
than being time-shifted, and the lead differs by type (a design property of the windows). The impression of a corner "before the
window" is the lead, plus our restart (end of D) being a few seconds later than the kick. No change to
the frame mapping in `src`.

## Dead-ball context (2026-11-04)

Idea: whether a frame is dead ball depends on the frames around it, but the dead-ball model sees one frame (plus the
rolling means of its inputs). A second stage gives it the neighbourhood of its own first-stage answer. Stage 1 is the
dead-ball model of the final configuration. Stage 2 is a dead-ball model on the same player features plus the centred,
past and future rolling means of the stage-1 logit at the layout's rolling scales (3 x scales columns, inside each
block or period). For training, the stage-1 logit of a match comes from a model fitted on the other training matches
(leave one validation group of whole matches out, the groups of the grouped stopping; same iteration rule as the stage-1 fit), so no context is built from a model that saw the labels
of its own frames. At prediction time stage 1 runs on the match and stage 2 replaces its logit; the EM adaptation runs
unchanged on the stage-2 logit (`docs/PARAMETERS.md`, "Dead-ball context"). Behind `train --dead-context`, off by default.
Training cost on the quick harness: about +3 %.

QUICK: every fifth training frame, nested folds, real `fit_model` with the final configuration (`--dead-features players
--adaptation --team-features touch --spell-weights --every-peak --stopping team`) and the real EM adaptation, decoder
settings of each fold of the final model; same harness for both rows. Mean held-out spellF1 and folds better:

| QUICK | DFL (7 folds) | Metrica fair (3 folds) | Metrica raw (3 folds) |
|---|---|---|---|
| baseline (final configuration) | 0.5745 | 0.7626 | 0.7721 |
| `--dead-context` | 0.5840 (+0.010, 7/7) | 0.7670 (+0.004, 2/3) | 0.7750 (+0.003, 1/3) |

DFL is better on every fold, Metrica fair on two of three and Metrica raw on one of three. The quick numbers replace nothing: the decision to adopt is the team's, on the full nested run
(`scripts/phase_final.txt`).

## Dead-ball context adopted (2026-11-05)

The full nested run of `train --dead-context` on top of the v2 flags (`scripts/phase_final.txt`: `train --variants
dfl+metrica --dead-features players --adaptation --team-features touch --spell-weights --every-peak --stopping team
--dead-context`; logs `outputs/logs/final2`, train runtime 18,920 s) replaces the v2 model. The new model (the earlier internal model v3) was
`models/learned/` until 2026-11-13 (run id `cb1bfa98d9e0351b2b5d15b1be672f4ff83c951e079b5f4aa4f7217bdcd49235`, `model.pkl` sha256
`285fc46b31bda4ed97112ebcf63696fe4626bc135c840a39d7eee0cf234fe5cb`); the previous one (v2) is not kept in the repository
(`model.pkl` sha256 `4075fdf9d1a5ee611115990b9022dacae8b4d85a7f7c091e15986417ff4b8b5a`, run id
`e5cfc0734f985b2d9f9f48419f29507a6ef65d2d4e5023468b525422c85e78ac`). The selected decoder is unchanged: emission
weight 0.2, stride 10, switch weight 0.5.

Held-out spellF1 (mean over the held-out matches of each source), v2 model to the new model, and the number of
held-out matches in which the new model is better:

| Source | v2 | New | Matches better |
|---|---|---|---|
| DFL (raw) | 0.5725 | 0.5771 | 7 of 7 |
| Metrica fair | 0.7732 | 0.7755 | 1 of 3 |
| Metrica raw | 0.7899 | 0.7813 | 1 of 3 |
| SkillCorner fair | 0.3015 | 0.3035 | 14 of 20 |
| SkillCorner raw | 0.2456 | 0.2508 | 18 of 20 |

The gain is small and consistent on DFL and SkillCorner, mixed on Metrica (fair up by 0.002, raw down by 0.009, each in
1 of 3 matches). The decision is by external held-out cross-validation only, as set on 2026-11-03: DFL and SkillCorner
improve on the large majority of their matches and the Metrica raw loss is inside the spread of 3 matches
(5-95 bootstrap interval of the v2 model on Metrica raw: 0.754 to 0.825).

Our-match diagnostics (`possession.diagnostics`, label-free) are an alarm, never a tie-break. They are recorded as noted,
not used. New model against v2: possession spells 325 against 312, stoppages (D runs) 96 against 89, set-piece windows
holding a predicted restart 49 of 59 against 48 of 59, restart team equals the set-piece team 18 of 23 against 16 of
22, phase-team agreement 75.7 % against 75.6 %. Noted, not used.

The public window was not consulted for this decision.

Our-match CSV from `configs/learned.json`: canonical Docker file sha256
`265068c905fd97523cd7184cadf282c11056e2bd286346e4344e2512afceacaa` (linux/arm64 and emulated linux/amd64, byte-identical; 719 s and 914 s). The macOS arm64 run gives the
same sha256, so no row differs from the Docker file for this model (the two models before it differed in 163 and 210 of
199,319 rows; `docs/COMPLIANCE.md`, item 6). The v2 Docker file
(`8e3821a3691cefba2bf703f6303ae60586dfdf693b4c43ecedf17334aa212ffb`) is superseded.

### Experiments since v2 that did not win

Each was compared with the v2 model on external held-out cross-validation; none replaced it. Numbers are given only
where a run summary exists in `models/*/model.json` or the logs; a missing number was not recorded.

- v3 pitch-line features (`train --line-features lines`, `scripts/phase_v3.txt`, model `models/learned_v3`): held-out
  spellF1 DFL 0.576 against 0.5725, Metrica fair 0.7574 against 0.7732, Metrica raw 0.774 against 0.7899, SkillCorner
  fair 0.313 against 0.3015; Metrica lost, so it was not adopted. Quick evaluation: "Dead ball by stoppage type and
  pitch-line features". The same flag on top of the dead-ball context model was adopted on 2026-11-13 ("Pitch-line
  features adopted (2026-11-13)").
The quick evaluations below ran on every 5th frame with cached dead-ball emissions, so their base differs a little from
the full-run numbers above.

- Smart Tagging features (8 tag-derived team features, the feed simulated on external matches from their events with the
  lag, window and thinning measured label-free on our match): quick evaluation on 3 folds, DFL J03WN1 0.560 to 0.553,
  DFL J03WQQ 0.587 to 0.579, Metrica game2 fair 0.769 to 0.765. The tags lag the play by about 6 s, too coarse for
  spell boundaries.
- Possessor model (per-player "has the ball" classifier trained on event-derived possessor labels, aggregated per team
  and stacked out of fold into the team model): quick evaluation on 10 folds, DFL 0.5605 to 0.5607 (2/7 folds),
  Metrica fair 0.7454 to 0.7485 (2/3), raw 0.7769 to 0.7778.
- Team auto-context (out-of-fold stage-1 team and dead-ball logits as context features of a second team model): quick
  evaluation on 10 folds, DFL 0.5745 to 0.5775 (4/7), Metrica fair 0.7626 to 0.7701 (1/3), at about 3.2 times the
  training cost. With the dead-ball stage as well: DFL +0.013 (6/7), Metrica fair +0.013 (1/3). Only the dead-ball
  stage was kept (`--dead-context`, adopted above).
- PFF team-only retrain (`dfl+metrica+pff_ha`, `scripts/phase_pff.txt`, `scripts/phase_pff2.txt`): stopped (status 130
  after 29,034 s in `outputs/logs/pff2`) once all 10 DFL and Metrica folds were cached; the remaining folds held out PFF
  matches only. From the cached per-fold grid scores, with the best common decoder setting chosen over the 10 folds for
  each model (not the nested rule, equally optimistic for both): DFL 0.5725 against 0.5588, Metrica fair 0.7792 against
  0.7491; at the settings that are best for v2 the PFF variant lost on 10 of 10 folds. Same conclusion as the
  2026-10-26 run (`docs/EXTERNAL.md`).
- Decoder variants on the cached held-out emissions, nested selection as in `train.py` (v2: DFL 0.5725, Metrica fair
  0.7732, raw 0.7899): no boundary refinement 0.5728 / 0.7767 / 0.7905; duration bandwidth x0.5 0.5727 / 0.7705;
  x2 0.5720 / 0.7792 / 0.7951; stride 2 fixed 0.5718 / 0.7789 / 0.7947 (paired fold difference +0.0012, standard
  error 0.0035); stride 1 fixed 0.5701 / 0.7560 / 0.7812; finer switch-weight and span grid gave the same choices as v2.
  The F1 surface is flat around the chosen setting.
- Ball-gap headroom on our match (label-free): 3.4 % of graded frames are predicted alive with the ball missing (DFL
  degraded 18.6 %, Metrica 4.3 %), 3.5 % of predicted H/A switches fall in a gap (DFL 22 %, Metrica 19 %). Scaling the
  external oracle gains by that share gives about +0.02 spellF1 for perfect team labels in alive gaps and +0.04 for
  perfect labels in all gaps, both within the 0.04 noise band, so ball imputation from the broadcast video was not
  pursued.

## Pitch-line features adopted (2026-11-13)

In this section v3 is the earlier internal dead-ball context model and v4 the same training with `--line-features
lines`.

Screening on the quick harness (every fifth training frame, nested folds) on top of the dead-ball context model:
`--line-features lines` gave DFL 0.6006 against 0.5830 (+0.018, 7 of 7 folds better), Metrica fair 0.7759 against
0.7678 (+0.008, 2 of 3), Metrica raw 0.7896.

Rule fixed on 2026-11-12, before the full run: adopt if DFL is better on at least 5 of 7 folds and Metrica fair is not
worse on at least 2 of 3, against `models/learned_v3_rebuild/variant_report.json`, and read SkillCorner as well. That
baseline is a rerun of the v3 training (2026-11-11) whose held-out numbers and per-fold choices are identical to those of
the committed v3 model (every tree and array is the same; the pickle bytes, and so the sha256 and run id, differ).

Full run (through `scripts/run_logged.sh`, logs `outputs/logs/lines`; the commands are in `scripts/phase_final.txt`): smoke run on the 3 Metrica matches at frame step 20
(1,855 s), then `train --variants dfl+metrica --dead-features players --adaptation --team-features touch --spell-weights
--every-peak --stopping team --dead-context --line-features lines` (nested run 20,139 s), `possession.run` (166 s) and
the validator (valid). The model was written to `models/learned_v4/` and copied to `models/learned/`: run id
`98ffe80dc9f2c861b3249a9558b507abf6d0b75ed82e23fdf6ca0c2cf066cdd2`, `model.pkl` sha256
`f249108eca1c98df04b1a70c88d3b5b70cb150dcc6e59a4d7783e508fbc4198f`. The dead-ball model has 39 columns (24 before) and
the team model 178 (151 before); `model.json` records `line_features: ["lines"]`, `iterations: [100, 124]` and
`stopping: ["default", "grouped"]`. The v3 model is not kept in the repository.

Held-out spellF1 (mean over the held-out matches of each source), v3 to v4, and the number of held-out matches in which
v4 is better:

| Source | v3 | v4 | Matches better |
|---|---|---|---|
| DFL (raw) | 0.5771 | 0.5872 | 6 of 7 |
| Metrica fair | 0.7755 | 0.7714 | 1 of 3 |
| Metrica raw | 0.7813 | 0.7821 | 2 of 3 |
| SkillCorner fair | 0.3035 | 0.3183 | 15 of 20 |
| SkillCorner raw | 0.2508 | 0.2565 | 12 of 20 |

DFL macroF1 goes from 0.867 to 0.881. DFL: the one loss is J03WOH (-0.015); the largest gains are J03WQQ (+0.036),
J03WR9 (+0.018) and J03WMX (+0.015). Metrica fair: game1 -0.012, game2 -0.0008, game3 +0.001. SkillCorner is scored by
the final model, which never saw it. The quick screening (+0.018, 7 of 7) overstated the full-run DFL gain (+0.010, 6 of
7).

The selected decoder moved from emission weight 0.2, stride 10, switch weight 0.5 to emission weight 0.1, stride 25,
switch weight 1.0. Every DFL fold chose stride 25, the top of the starting grid; the stride grid then extended itself to
50 and the folds still chose 25. The Metrica folds chose (0.1, 25, 0.5) for game1, (0.1, 25, 1.0) for game2 and
(0.2, 5, 0) for game3. Predicted DFL spells per match went from 330.9 to 321.0 against 437.3 true: the smoothing is
heavier and the model still predicts fewer spells than the reference.

Decision. DFL meets the rule (6 of 7). Metrica fair does not meet it strictly: game3 is better, game2 is lower by
0.0008, within one spell of a tie, and game1 is lower by 0.012. The team adopted the model on 2026-11-13, for this
reason: the DFL labels (provider fields; the team switches at the new team's first touch) and the SkillCorner matches
(an independent provider; broadcast tracking in which the ball is often missing) are closer to the conventions of the
challenge labels (`docs/GT_SEMANTICS.md`) and to the ball visibility of our match than the Metrica sample, whose labels
come from our own event rules (`metrica_events.py`) and whose ball track is event-interpolated (`docs/EXTERNAL.md`).
DFL improves on 6 of 7 matches and SkillCorner on 15 of 20 (fair), while the Metrica fair loss is 0.004 over 3 matches.
This weighs the sources differently from the rule of 2026-10-26, which made Metrica, the provider of the corrected
tracking of our match, the priority source.

Our-match CSV from `configs/learned.json`, native macOS arm64 run: sha256
`234c1366efa2d4a95a23eb0e8c9944af5ee6e6520da5e289f6204a14e665297b`. The v3 Docker file
(`265068c905fd97523cd7184cadf282c11056e2bd286346e4344e2512afceacaa`) is superseded.
