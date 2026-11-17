---
title: "From dots to possessions"
subtitle: "Team Graciencs. More Than A Hack 2027, Initial Challenge. Short technical document. Repository: https://github.com/gerardcabot/possession-from-tracking"
---

# 1. Approach

The task is to label each of the 199,319 frames of the tactical video of FC Barcelona vs Athletic Club as `H`, `A`, `D` (dead ball) or `X`. The ranking metric, spellF1, scores whole possessions, so the system has to get the number, the team and the boundaries of the possessions right, and has to find the stoppages because each one ends a possession.

We built a learned pipeline that never sees a label of the provided match:

1. **Cleaning.** Ball spikes and impossible player jumps are removed, short ball gaps are filled and duplicate objects merged. Every bound is estimated from data.
2. **Training on other matches, made to look like ours.** External matches with possession labels are degraded with label-free statistics of our tracking: its ball gaps, its fragmented player slots, its duplicates and its jitter.
3. **Frame classifiers** (scikit-learn gradient boosting). A dead-ball model uses player movement only, with a second stage that adds temporal context. A team model says who has the ball while play is alive. A switch prior scores how likely a change of possession is near each frame.
4. **Unsupervised adaptation.** At prediction time an EM step re-estimates the dead-ball share and the ball evidence on the match itself. No label is read.
5. **Decoding.** A duration-explicit hidden semi-Markov model over H, A and D, with durations and transitions learned from the external labels, turns the frame probabilities into possessions by exact Viterbi decoding and a boundary refinement.

**Why this and not rules.** Hand-written rules need thresholds, and a threshold cannot be validated without labels of the match, which may not be used. A learned model can be trained and validated on other matches, and the decoder addresses the metric directly: frame probabilities flicker, possessions do not.

**Why the dead-ball model ignores the ball.** How the ball behaves during a stoppage depends on the provider: one freezes it, another drops it, ours does neither. A model trained with ball features on other providers could not read dead ball on our match. Player movement does not shift between providers, so the dead-ball model uses it alone, and the ball evidence is learned on the match itself by the adaptation step.

**No hand-set constants.** Every number on the path of the submission is a metadata fact, a library default, an estimate from data, or a choice made by nested cross-validation on external matches. One pair of quantile levels is the exception and is disclosed in the full report.

# 2. Assets used

| Asset | Use |
|---|---|
| Tactical tracking (ATD) and its metadata | **The only input** of the submitted method: positions of 46 anonymous slots and the ball, 25 fps, pitch size, graded blocks and kickoff anchors as published in the official `games.py` |
| Smart Tagging | Not an input of the submitted method. Tag-derived features were simulated on external matches and lost; the tags lag the play by about 6 s. The tags are read only by label-free diagnostics of our match, used as a tie-break between candidates tied on external cross-validation until 2026-11-03 and as an alarm afterwards |
| Tactical and broadcast video, audio | Not used. Ball detection from video and whistle detection from audio were considered as extra inputs and dropped (run-time cost, no external audio to train on) |
| Broadcast tracking, homography | Not used. Imputing the ball from the broadcast feed had an estimated headroom of 0.02 to 0.04 spellF1, within the noise band |
| Public five-minute ground truth | Pipeline check only. Nothing was fitted, tuned or selected on it |

Also tried and rejected on external cross-validation: PFF World Cup data as extra training matches, a per-player possessor model, a second context stage for the team model, goalkeeper and ball-line feature groups, and several decoder variants. Each comparison is recorded in `docs/DEADBALL.md`.

# 3. Training data

None of it comes from the provided match, and no team member labelled any frame by hand.

| Dataset | Amount | Labels and who made them | Role | Licence |
|---|---|---|---|---|
| DFL / Sportec (Bassek et al. 2025) | 7 Bundesliga and 2. Bundesliga matches, about 1.0 million frames, 3,061 possessions | the provider: per-frame ball status and ball possession | training and validation | CC BY 4.0 |
| Metrica Sports sample data | 3 matches, about 0.43 million frames, 1,048 possessions | events annotated by Metrica Sports; frame labels derived from them by our code | training and validation | no licence file; source acknowledged as its README asks |
| SkillCorner open data | 20 A-League matches | the provider's possession group; no dead-ball label | evaluation only, never trained on | MIT |

Every training match is used from both teams' perspectives, so the model has no home bias. The decoder settings are chosen by nested leave-one-match-out cross-validation on these matches, and the boosting rounds of the team model by early stopping on whole held-out training matches. The training set was chosen on held-out external scores; the team overrode the coded selection rule once, as the full report discloses.

# 4. Possession rules implemented

The model has no hand-written possession rules. It inherits the conventions of its training labels, which match the challenge definition:

- **Change of possession** at the new team's first touch. A deflection does not buy a possession: the decoder needs sustained evidence to pay for a switch.
- **Loose balls, challenges and passes in flight** stay with the last team until the other team takes the ball.
- **Dead ball** from the line crossing or the whistle until the restart touch. Dead ball belongs to nobody: the label is `D`.
- **Restarts** (throw-ins, free kicks, goal kicks, corners) are `D` until the ball is played, then belong to the team that takes them.
- **Before each kickoff**, graded frames are written as `D`, using the kickoff anchors published in the official `games.py`.
- **Outside the graded blocks** the label is `X`.

# 5. Results and limitations

Nested leave-one-match-out on external matches: each match is scored by a model trained without it, with decoder settings chosen on the other matches.

| Held-out data | Matches | spellF1 [5 to 95 %] | macroF1 |
|---|---:|---|---:|
| DFL | 7 | 0.587 [0.546, 0.627] | 0.881 |
| Metrica sample | 3 | 0.782 [0.744, 0.819] | 0.915 |
| SkillCorner, dead ball ignored | 20 | 0.318 [0.282, 0.355] | |

These are estimates on other providers' data, not a score on the provided match.

- **Short possessions are merged.** The model predicts fewer possessions than the references (321 against 437 per DFL match). Spells of 1 to 3 s are the main loss.
- **The ball is missing on about 41 % of graded frames**, mostly during stoppages, and our simulation places ball gaps independently of the game state.
- **Throw-ins are the dead balls missed most**, and a goalkeeper holding the ball may be read as a stoppage.
- **Metrica cross-validation is optimistic**: its sample ball track is interpolated between events and its dead-ball labels are derived by our own rules.
- **Provider shift is real**: 0.318 on SkillCorner, whose labels follow the player nearest to the ball.
- **With more time:** a goalkeeper-holding feature, team swaps of tracked objects in the simulation, and refining the durations on the match itself.

# 6. Reproduction

With Docker and the match data from the official `DATA.md`, from a clone of the repository:

```
scripts/reproduce_docker.sh /path/to/data outputs
```

It builds a pinned image, runs the committed model on the two tactical tracking files and prints the sha256 of the CSV, which must be `234c1366efa2d4a95a23eb0e8c9944af5ee6e6520da5e289f6204a14e665297b`. It takes about three to four minutes plus the image build. The same bytes come out on linux/arm64, emulated linux/amd64 and native macOS arm64.

Without Docker (Python 3.12): `uv sync --frozen`, then `MTAH_DATA_DIR=/path/to/data .venv/bin/python -m possession.run --config configs/learned.json --out team.csv`.

Training is not needed to reproduce the file, because the model is committed. The command that trained it and the full method, numbers and decisions are in the repository: `docs/technical_document.md` (full report), `docs/PARAMETERS.md` (origin of every parameter), `docs/DEADBALL.md` (every model decision) and `docs/EXTERNAL.md` (datasets and licences).

**References.** Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data* 12, 195. Data © DFL Deutsche Fußball Liga, CC BY 4.0. Metrica Sports sample data, https://github.com/metrica-sports/sample-data. SkillCorner open data, © 2020 SkillCorner, MIT, https://github.com/SkillCorner/opendata.
