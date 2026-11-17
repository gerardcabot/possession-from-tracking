# Usage reference

Everything the top-level `README.md` leaves out: environment setup, the full reproduction notes, every
command, and what each tool prints.

## Setup

```bash
uv sync --frozen
uv sync --frozen --group external
```

The first command installs the core dependencies, enough for `possession.run`. The second one adds
the `external` group (`kloppy`), only needed to build the external validation data (see
`docs/EXTERNAL.md`).

Python 3.12, dependencies pinned in `uv.lock`. `requirements.txt` is the core pin set and
`requirements-external.txt` is the core plus `external` set, both exported with
`uv export --format requirements-txt --no-hashes --no-emit-project --no-header --no-annotate`
(add `--group external` for the second).

Without `uv` (needs a Python 3.12 interpreter on `PATH`):

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

`requirements.txt` alone does not make `possession` importable: the last line installs the package
itself in editable mode, as `uv sync` does (it fetches the `hatchling` build backend from PyPI). Use
`requirements-external.txt` instead of `requirements.txt` to add `kloppy`. The commands below work
the same with either environment.

| Variable | Default | Used for |
|---|---|---|
| `MTAH_DATA_DIR` | `../data` | match data, expects `tactical/BAR-ATH_tactical_atd.{xml,txt}` |
| `MTAH_REPO_DIR` | `..` | official challenge checkout, for the public GT and official baseline files |
| `MTAH_CACHE_DIR` | `cache/` | parsed tracking cache (`.npz`, keyed by sha256 of the source files) |
| `MTAH_EXTERNAL_DIR` | `<MTAH_DATA_DIR>/external` | external datasets for training and validation, one folder per source (`dfl/`, `metrica/`, `skillcorner/`, `pff/`; layout in `docs/EXTERNAL.md`); not needed by `possession.run` |

The first run parses the tracking (about 1.5 s) and writes the cache. Later runs load it. Delete
`cache/` to rebuild it, or pass `--no-cache`.

## Reproducing the submission

The canonical submission is the Docker output. From a clean clone of tag `submission`, with the match data
downloaded as described in the official `DATA.md`:

```bash
scripts/reproduce_docker.sh /path/to/data outputs
```

Both arguments are optional: the data directory defaults to `MTAH_DATA_DIR`, else `../data` (the
repository sitting as `solution/` inside the official checkout), and the output directory to `OUT_DIR`,
else `outputs/`. The script builds the image (`Dockerfile`: `python:3.12-slim` and `uv` 0.12.0, both pinned
by digest, dependencies from `uv.lock`), runs it with the two tracking files mounted read-only, writes
`outputs/learned_final_docker.csv` (`OUT_NAME`) and prints its sha256, which must be

`234c1366efa2d4a95a23eb0e8c9944af5ee6e6520da5e289f6204a14e665297b`

The model it loads, `models/learned/model.pkl`, has sha256
`f249108eca1c98df04b1a70c88d3b5b70cb150dcc6e59a4d7783e508fbc4198f`. `DOCKER_PLATFORM` (for example
`linux/amd64`) selects the platform and `IMAGE_TAG` the image name. The run needs no network access once the
image is built. The linux/arm64 run and the emulated linux/amd64 run (OrbStack on a Mac, 2026-11-15; amd64 242 s
plus the build) both give the sha256 above, the same as the native macOS arm64 run (2026-11-13); a clean export of
the published tree (tag `submission`) run the same way on linux/arm64 (2026-11-15, 194 s plus the build) gives it too. For the superseded earlier internal model v3, the runs of 2026-11-05 under OrbStack
(cold cache) took 719 s on linux/arm64 and 914 s on emulated linux/amd64, plus the build; that model's file is
not the submission.

The uploaded `Graciencs.csv` is a byte copy of the linux/arm64 Docker output (same sha256);
`OUT_NAME=Graciencs.csv scripts/reproduce_docker.sh /path/to/data outputs` writes it under that name
directly. To check the file with the vendored official validator inside the same image, with no local
Python environment (add `--platform "$DOCKER_PLATFORM"` if the image was built for another platform):

```bash
docker run --rm -v "$PWD/outputs":/out possession-repro \
  python src/possession/vendor/validate_submission.py /out/learned_final_docker.csv
```

`possession.run` reads `tactical/BAR-ATH_tactical_atd.xml` and `tactical/BAR-ATH_tactical_atd.txt` under
the data directory; the other `DATA.md` files are not needed. They may be symlinks, the script resolves
them before mounting.

### Without Docker

```bash
uv sync --frozen
MTAH_DATA_DIR=/path/to/data .venv/bin/python -m possession.run --config configs/learned.json --out team.csv
```

Python 3.12. The default `MTAH_DATA_DIR` is `../data`, so nothing needs setting only when the repository
sits as `solution/` inside the official checkout. `MTAH_REPO_DIR` is only needed by `possession.evaluate`
and the other tools that read the public ground truth or the official results, not by `possession.run`. It
writes a cache of about 214 MB under `cache/` (or `MTAH_CACHE_DIR`). The run of the current model took 166 s with the
tracking cache already built; the earlier internal model v3 took 559 s on a cold cache.

The bytes depend on the platform's numerical libraries: on macOS arm64 (numpy and scipy use Accelerate)
the native run of the current model gives the same sha256 as the linux/arm64 and linux/amd64 Docker runs above. Of the earlier
internal models, v3 matched too, while v1 and v2 did not: 163 and 210 rows differed, at possession-spell boundaries,
and the cause was inferred from the two BLAS builds, not isolated, so equality is not assumed. Only the Docker file is the submission. Results of the
reproduction runs are recorded in `docs/COMPLIANCE.md`, item 6.

## Run

```bash
.venv/bin/python -m possession.run --config configs/baseline.json --out outputs/baseline_smooth50.csv
.venv/bin/python -m possession.run --method baseline --window 50 --out outputs/baseline_smooth50.csv
```

Flags given on the command line override the config file. The run prints the output's sha256.

### Method `rules` (Phase 1, LEGACY)

Legacy reference only: its parameters are hand-set (`possession.params`, including its own legacy
cleaning values) and it does not influence `learned` or the shared cleaning.

```bash
.venv/bin/python -m possession.run --config configs/rules.json --out outputs/rules.csv
```

A rule-based, offline labeller (about 7 s on the whole match with a warm cache). Steps:

1. `features` gives per-frame metres-based features: ball kinematics, nearest player per team, team
   centroids and their smoothed velocity, mean player speed, players beyond halfway. It also gives each half's attacking direction, taken
   from where each team's deepest player sits.
2. `deadball` turns collective speed, ball activity and ball-over-line into dead-ball evidence and
   applies hysteresis to get stoppages. A stoppage starts at the line crossing (observed, or
   extrapolated into a ball gap) or else at the onset of the collective deceleration.
3. `restart` puts the restart at the last kick from a settled ball that clearly moves it, and
   gives it to the taker's team (or the Smart Tagging set-piece team). Each half is dead until a
   kickoff detected from the kickoff formation and the ball leaving the centre mark.
4. `control` detects controlled possession: the ball slow, within reach of a player of one team,
   who is closer than any opponent, for a short dwell.
5. `statemachine` gives the restart team possession and switches only on opponent control,
   back-dated to its first frame. If the ball was missing in between, the switch goes at the best
   change point of the team-shape evidence inside the gap.

Parameters and their rationale: `docs/PARAMETERS.md`. Public-GT conventions:
`docs/GT_SEMANTICS.md`.

### Method `learned` (Phase 2)

```bash
.venv/bin/python -m possession.learned.train
.venv/bin/python -m possession.external.validate --method learned
.venv/bin/python -m possession.run --config configs/learned.json --out outputs/learned.csv
```

Four scikit-learn `HistGradientBoostingClassifier`s trained on degraded external matches: a
dead-ball model P(D) on player kinematics only (including how far the players stand beyond the pitch lines), a dead-ball context stage (a second dead-ball model
that also sees rolling means of the first one's logit, trained on out-of-fold logits), a team model
P(H | alive), and a switch prior (the probability that possession passes to a team near a frame,
from touch evidence). At inference, EM adapts the dead-ball evidence to our match without labels:
it re-estimates the dead-ball prior and refits the ball-observation evidence on the match. The
training set (DFL, DFL + Metrica, DFL + Metrica + PFF, DFL + Metrica + PFF for the team split only,
DFL + Metrica + SkillCorner for the team split only, or Metrica) is compared on held-out scores
(the team overrode the coded selection once, `docs/EXTERNAL.md`). The frame posteriors feed an
explicit-duration HSMM whose durations, transitions and initial states are learned from the same
external labels; the switch prior is added to every direct H to A or A to H transition. Decoding is
exact Viterbi on strided steps, followed by frame-level boundary refinement. On our match, `force_prekickoff_dead` (default true in `configs/learned.json`) then writes D on every graded frame from each block start through the kickoff frame published in the official `games.py` (anchors 11,186 and 113,741; see `docs/COMPLIANCE.md` item 1). The emission weight, the
Viterbi stride and the switch-prior weight (0.1, 25 frames and 1.0 in the committed model) are
chosen by nested leave-one-match-out cross-validation. Design and parameter origins:
`docs/PARAMETERS.md` (section "Learned parameters"); the submitted technical document is
`docs/Graciencs.md`, a short version of the full report `docs/technical_document.md` (both listed under
"What is in the repository" in `README.md`).

`possession.learned.train` needs the external caches (`docs/EXTERNAL.md`). The cleaning parameters
are estimated from external DFL tracking and our match aggregates by `possession.external.prepare`
and stored inside the model, so `possession.run` cleans our match exactly as the training data was
cleaned (see "Cleaning"). The committed model is the `dfl+metrica` variant trained with
`--dead-features players --adaptation --team-features touch --spell-weights --every-peak --stopping team --dead-context --line-features lines`
(`scripts/phase_final.txt`, step 2; `docs/DEADBALL.md`, "Dead-ball context adopted (2026-11-05)" and "Pitch-line features adopted (2026-11-13)"). By default it trains the
variants `dfl`, `dfl+metrica` and `metrica`; the PFF variants (`dfl+metrica+pff`,
`dfl+metrica+pff_ha`) and the SkillCorner variant (`dfl+metrica+skillcorner_ha`) are only trained
when asked for (`--variants all` or by name; a list that contains the three default variants plus
PFF variants may write the default model path). PFF's 64 matches are held out in 7 groups and
subsampled for the classifier rows (`docs/PARAMETERS.md`, "Training-set variants and selection"). It writes:

- `models/learned/model.pkl` and `model.json` (sha256, selected settings, scales, training matches,
  cleaning parameters);
- `models/learned/variant_report.json` (aggregates only: held-out scores of every variant on every
  DFL, Metrica, SkillCorner and PFF match, and the selection);
- `models/learned/folds/<variant>/` (one held-out model per match of each variant, each with its own
  nested choice of decoder settings, plus each variant's final model; a PFF group fold is one model
  file shared by the group, and each of its matches has a sidecar that points to it and holds its own
  settings; ignored by git, reused by a rerun with identical inputs).

`external.validate --method learned` scores each match of the selected variant with its held-out
fold model and any other match with the final model. It refuses fold models from a different training run (run id in
`model.json` and in each fold's sidecar).

`--dead-features players` trains the dead-ball model on player kinematics only and `--adaptation` re-estimates the ball-observation evidence on each predicted match by EM; both default to the previous method. Why, and the comparison rule: `docs/DEADBALL.md`.

`--matches`, `--limit`, `--frame-step` (smoke runs) and a `--variants` list that lacks one of the
default variants need an explicit `--out` other than the default model path. Unknown and ambiguous match names (use
`source/name`) are errors, and every selected variant is checked for at least two matches per
source before any model is fitted. `--selection-score raw|fair` (default `raw`) picks the Metrica
score that is the primary criterion of the variant selection; both are always reported. Paths given on the command line (`--out`, `--model`)
are relative to the current directory; the `model` key in a config file is relative to the project
root.

The model and its two JSON files are committed, so `possession.run` reproduces from a clean clone
plus the match data, with no external data. Other committed records: `models/learned_variants/variant_report.json`
(the five-variant run) `models/learned_w/` (method W, `dfl+metrica`) and `models/learned_c/` ((c) without touch features), kept for the record; they use model format v6 and no longer load.
To retrain: download the external
data, run `possession.external.prepare`, then `possession.learned.train`. Training is single-threaded
and seeded: a rerun of the training of the earlier internal model v3 with the same inputs and wheels (2026-11-11) gave a model
identical in every tree and array, but its pickle bytes, and so its sha256 and run id, differed. `possession.run`
refuses a model whose sha256 or scikit-learn version does not match `model.json`.

## Logged runs

`scripts/run_logged.sh <phase> <commands-file>` runs each non-empty, non-`#` line of the file as a
shell command, in order, and writes `outputs/logs/<phase>/NN.log` per step plus
`outputs/logs/<phase>/summary.txt` (start, end, status, seconds); it stops at the first failing step.
`scripts/phase_ab.txt` lists the two full comparison runs of `docs/DEADBALL.md` (the previous method,
then `--dead-features players --adaptation`, each trained, validated, analysed and run on our match).

## Validate and evaluate

`possession.evaluate` defaults to the public ground truth. Use it only for pipeline sanity checks,
never to compare or select variants.

```bash
.venv/bin/python src/possession/vendor/validate_submission.py outputs/baseline_smooth50.csv
.venv/bin/python -m possession.evaluate outputs/baseline_smooth50.csv
.venv/bin/python -m possession.evaluate outputs/x.csv --gt path/to/gt.csv
```

## Diagnostics

```bash
.venv/bin/python -m possession.diagnostics --submission outputs/baseline_smooth50.csv
.venv/bin/python -m possession.diagnostics --model models/learned/model.pkl
.venv/bin/python -m possession.diagnostics --legacy
```

Cleaning uses the parameters estimated from external data by default (this needs the external
caches), the ones stored in a trained model with `--model`, or the legacy `rules` values with
`--legacy`. Prints aggregates only: ball coverage (overall, graded, per half, raw and after cleaning), ball
gap-length buckets, cleaning counts, player objects per frame, and for each `--submission` the
graded label shares, possession spell count, mean and median spell duration, a spell-duration
histogram, the stoppage count, Smart Tagging checks and the D share. The Smart Tagging checks are
set-piece windows that hold a detected restart, agreement between restart team and set-piece team,
and agreement between possession and the phase team, where "Defending in …" phases count for the
other team. It never prints or writes per-frame data.

## Cleaning

`possession.cleaning.clean(tracking, pitch, fps, params)` converts to metres and returns a
`CleanTracking`: a ball sample is removed only when neither its past nor its future trajectory
predicts it (an isolated spike; fast but smooth flight, such as a high ball projected on the ground,
is kept), a player sample is removed with the whole speed excursion around any motion faster than
the physical maximum (the samples between the speed minima around it that deviate from both the
pre- and post-jump trajectory by more than the DFL-derived player deviation), short ball gaps are filled linearly (up to the dropout length and only when the
straight line is physically possible), and same-team objects closer than any two real teammates
come are merged into one. `params` has no default. For `learned` every value is estimated from data
(DFL optical tracking and our match aggregates, `possession.external.bounds`); the `rules` method
uses its legacy hand-set values (`possession.params.LEGACY_CLEANING`). Statistics, current values and
their effect on our match: `docs/PARAMETERS.md`, "Cleaning parameters". Identity switches are not
repaired and no feature depends on a slot keeping its identity (same file, "Duplicates, identity
switches and slot continuity").

## Test

```bash
.venv/bin/python -m pytest
```

Tests that need the match data or the official checkout skip when those are missing. With them,
the suite checks the baseline is byte-identical to the official `baseline_smooth50.csv`, scores the
same, and gives the same sha256 across two fresh processes (cold and warm cache).
