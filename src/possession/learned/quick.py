import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from possession.cleaning import CleaningParams
from possession.external.match import ExternalMatch
from possession.external.prepare import (
    DEFAULT_SEED,
    code_fingerprint,
    degraded_path,
    reference_params,
    reference_stats,
    variant_match,
)
from possession.external.sources import DFL, METRICA, SOURCES
from possession.learned import train
from possession.learned.adapt import adapted_dead_logit
from possession.learned.classifier import STOPPING_RULES, THREADS
from possession.learned.external import fold_dir, fold_path, labels_only, training_example
from possession.learned.framefeatures import (
    DEAD_FEATURE_SETS,
    DEFAULT_TEAM_FEATURES,
    LINE_GROUPS,
    PERSPECTIVES,
    TEAM_FEATURE_SETS,
    BaseFeatures,
    block_segments,
    feature_matrix,
)
from possession.learned.model import (
    MODEL_SEED,
    ClassifierSettings,
    DeadBallSettings,
    DecoderSettings,
    Entering,
    LearnedModel,
    TrainingExample,
    entry_evidence,
    fit_model,
    load_model,
    read_metadata,
)
from possession.paths import cache_dir, cli_path

FEATURE_FORMAT = "features-v1"
FEATURE_MODULES = ("possession.learned.framefeatures", "possession.learned.external",
                   "possession.external.match", "possession.external.predict",
                   "possession.features", "possession.cleaning", "possession.geometry",
                   "possession.intervals", "possession.data")
META_NAME = "meta.json"
EVIDENCE_SUFFIX = ".evidence.npz"
REPORT_NAME = "quick_report.json"
QUICK_SOURCES = (DFL, METRICA)
PERSPECTIVE_FILES = {first: f"base{first}.npy" for first in PERSPECTIVES}
TOUCH_FILE = "touch.npy"


@dataclass(frozen=True)
class Job:
    features: Path
    out: Path
    variant: str
    members: tuple[str, ...]
    held: tuple[str, ...]
    key: str
    seed: int
    frame_step: int
    grid: tuple[DecoderSettings, ...]
    fingerprint: str
    dead_ball: DeadBallSettings
    team_features: str
    switch_prior: bool
    fitting: ClassifierSettings
    lines: tuple[str, ...]
    fit_from: Path | None = None

    def context(self, loaded: list[train.Loaded]) -> train.Context:
        return train.Context(seed=self.seed, frame_step=self.frame_step, grid=list(self.grid),
                             loaded=loaded, out=self.out, fingerprint=self.fingerprint,
                             dead_ball=self.dead_ball, outside_sources=(),
                             team_features=self.team_features, switch_prior=self.switch_prior,
                             fitting=self.fitting, lines=self.lines)


def default_features() -> Path:
    return cache_dir() / "features"


def example_dir(features: Path, key: str) -> Path:
    return features / key.replace("/", "_")


def save_example(directory: Path, example: TrainingExample, match: ExternalMatch, data_key: str,
                 lines: tuple[str, ...]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for first, base in zip(PERSPECTIVES, example.base):
        np.save(directory / PERSPECTIVE_FILES[first], np.asarray(base.matrix))
    touch = example.base[0].touch
    if touch is not None:
        np.save(directory / TOUCH_FILE, np.asarray(touch))
    np.save(directory / "codes.npy", example.codes)
    np.save(directory / "labels.npy", match.labels)
    np.save(directory / "periods.npy", match.periods)
    np.save(directory / "slot_team.npy", match.slot_team)
    meta = {"format": FEATURE_FORMAT, "code": code_fingerprint(*FEATURE_MODULES),
            "name": match.name, "source": match.source, "fps": match.fps,
            "pitch": list(match.pitch), "data_key": data_key, "names": list(example.base[0].names),
            "touch": touch is not None, "lines": list(lines),
            "cleaning": example.params.to_dict(), "dead_labels": example.dead_labels}
    (directory / META_NAME).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return directory


def check_meta(directory: Path, meta: dict[str, object], lines: tuple[str, ...],
               data_key: str | None = None) -> None:
    if meta.get("format") != FEATURE_FORMAT or meta.get("code") != code_fingerprint(*FEATURE_MODULES):
        raise ValueError(f"{directory}: feature cache built with other code; rebuild it with "
                         "possession.learned.quick cache")
    if data_key is not None and meta.get("data_key") != data_key:
        raise ValueError(f"{directory}: feature cache built from degraded data "
                         f"{meta.get('data_key')}, current {data_key}; rebuild it with "
                         "possession.learned.quick cache")
    if tuple(meta["lines"]) != lines:
        raise ValueError(f"{directory}: feature cache has line groups {meta['lines']}, requested "
                         f"{list(lines)}; build a cache with --line-features {' '.join(lines)}")


def load_example(directory: Path, lines: tuple[str, ...] = (),
                 data_key: str | None = None) -> train.Loaded:
    meta = json.loads((directory / META_NAME).read_text())
    check_meta(directory, meta, lines, data_key)
    labels = np.load(directory / "labels.npy")
    periods = np.load(directory / "periods.npy")
    match = ExternalMatch(name=str(meta["name"]), source=str(meta["source"]), fps=float(meta["fps"]),
                          pitch=(float(meta["pitch"][0]), float(meta["pitch"][1])),
                          ball=np.zeros((0, 2)), players=np.zeros((0, 0, 2)),
                          slot_team=np.load(directory / "slot_team.npy"), labels=labels,
                          periods=periods)
    blocks = match.blocks
    segments = block_segments(len(labels), blocks)
    touch = np.load(directory / TOUCH_FILE, mmap_mode="r") if meta["touch"] else None
    base = tuple(BaseFeatures(names=tuple(meta["names"]),
                              matrix=np.load(directory / PERSPECTIVE_FILES[first], mmap_mode="r"),
                              segments=segments, fps=match.fps, first=first, touch=touch)
                 for first in PERSPECTIVES)
    params = CleaningParams(**meta["cleaning"])
    example = TrainingExample(name=f"{match.source}/{match.name}", base=base,
                              codes=np.load(directory / "codes.npy"), blocks=blocks, params=params,
                              dead_labels=bool(meta["dead_labels"]))
    return train.Loaded(example=example, match=match, data_key=str(meta["data_key"]))


def cached_keys(features: Path) -> list[str]:
    keys = []
    for directory in sorted(path for path in features.iterdir() if (path / META_NAME).is_file()):
        meta = json.loads((directory / META_NAME).read_text())
        keys.append(f"{meta['source']}/{meta['name']}")
    return keys


def ordered_keys(features: Path, sources: tuple[str, ...]) -> list[str]:
    cached = [key for key in cached_keys(features) if key.split("/")[0] in sources]
    canonical = [train.file_key(files) for files in train.selected_files(None, None, sources)]
    return ([key for key in canonical if key in cached]
            + [key for key in cached if key not in canonical])


def current_data_keys(seed: int, names: set[str] | None, sources: tuple[str, ...]
                      ) -> dict[str, str]:
    stats = reference_stats()
    return {train.file_key(item): degraded_path(item, stats, seed).name
            for item in train.selected_files(names, None, sources)}


def load_examples(features: Path, names: set[str] | None, sources: tuple[str, ...],
                  lines: tuple[str, ...], data_keys: dict[str, str]) -> list[train.Loaded]:
    keys = [key for key in ordered_keys(features, sources)
            if names is None or key in names or key.split("/")[1] in names]
    missing = sorted((names or set()) - set(keys) - {key.split("/")[1] for key in keys})
    if missing:
        raise ValueError(f"matches {missing} are not in the feature cache {features}; available "
                         f"{cached_keys(features)}")
    unverified = sorted(set(keys) - set(data_keys))
    if unverified:
        raise ValueError(f"matches {unverified} of the feature cache {features} have no current "
                         "degraded data; rebuild it with possession.learned.quick cache")
    return [load_example(example_dir(features, key), lines, data_keys[key]) for key in keys]


def build_cache(features: Path, seed: int, names: set[str] | None, sources: tuple[str, ...],
                lines: tuple[str, ...]) -> None:
    files = train.selected_files(names, None, sources)
    stats = reference_stats()
    params = reference_params()
    for item in files:
        start = time.perf_counter()
        match = variant_match(item, stats, seed, "degraded", force=False)
        example = training_example(match, params, True, lines)
        directory = save_example(example_dir(features, train.file_key(item)), example,
                                 labels_only(match), degraded_path(item, stats, seed).name, lines)
        print(f"cached {train.file_key(item)} -> {directory} in {time.perf_counter() - start:.0f}s",
              flush=True)


def evidence_path(path: Path, key: str) -> Path:
    return path.with_name(f"{key.replace('/', '_')}{EVIDENCE_SUFFIX}")


@dataclass(frozen=True)
class Components:
    evidence: train.Evidence
    dead: np.ndarray
    away: np.ndarray


def match_components(model: LearnedModel, item: train.Loaded) -> Components:
    base = item.example.base[0]
    matrix = feature_matrix(base, model.layout)
    if model.dead_ball.adaptation_iterations <= 0:
        dead, report = model.classifier.dead_logit(matrix, segments=base.segments), None
    else:
        dead, report = adapted_dead_logit(model.classifier, matrix, item.example.blocks, MODEL_SEED,
                                          model.dead_ball.adaptation_iterations,
                                          segments=base.segments)
    with threadpool_limits(limits=THREADS):
        away = model.classifier.side.decision_function(matrix)
    train.log_adaptation(item.key, report)
    evidence = train.Evidence(emission=model.classifier.log_emission(matrix, dead),
                              entering=entry_evidence(model, base),
                              adaptation=None if report is None else report.to_dict())
    return Components(evidence=evidence, dead=dead, away=away)


def save_evidence(path: Path, components: Components) -> None:
    evidence = components.evidence
    entering = np.zeros((0, 0)) if evidence.entering is None else evidence.entering.frames
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, emission=evidence.emission, entering=entering, dead=components.dead,
             away=components.away, adaptation=np.array([json.dumps(evidence.adaptation)]))


def load_evidence(path: Path) -> train.Evidence:
    with np.load(path, allow_pickle=False) as archive:
        emission, entering = archive["emission"], archive["entering"]
        adaptation = json.loads(str(archive["adaptation"][0]))
    return train.Evidence(emission=emission,
                          entering=None if entering.size == 0 else Entering(entering),
                          adaptation=adaptation)


def load_components(path: Path) -> Components:
    with np.load(path, allow_pickle=False) as archive:
        dead, away = archive["dead"], archive["away"]
    return Components(evidence=load_evidence(path), dead=dead, away=away)


def reused_fit(fit_from: Path | None, variant: str, key: str,
               examples: list[TrainingExample]) -> LearnedModel | None:
    if fit_from is None:
        return None
    path = fold_path(fit_from, variant, key)
    if not path.is_file():
        raise FileNotFoundError(f"--fit-from {fit_from} has no fold model {path}")
    model = load_model(path)
    expected = {example.name for example in examples}
    if set(model.training_matches) != expected:
        raise ValueError(f"--fit-from {fit_from}: fold {key} was trained on "
                         f"{sorted(model.training_matches)}, this fold trains on {sorted(expected)}")
    return model


def run_quick_fold(context: train.Context, variant: str, members: list[train.Loaded],
                   held: list[train.Loaded], key: str, fit_from: Path | None = None
                   ) -> list[train.Fold]:
    keys = {item.key for item in held}
    examples = train.fit_examples([item for item in members if item.key not in keys])
    path = fold_path(context.out, variant, key)
    digest = train.fit_digest(context, examples, [item.key for item in held], context.grid)
    cached = train.cached_fit(path, digest)
    if cached is not None:
        model, stored, adaptation = cached
        for item in held:
            if not evidence_path(path, item.key).is_file():
                save_evidence(evidence_path(path, item.key), match_components(model, item))
        print(f"{variant} fold {key}: reused {path.name}", flush=True)
        return train.held_out_folds(model, held, stored, adaptation, key)
    start = time.perf_counter()
    model = reused_fit(fit_from, variant, key, examples)
    source = "fit" if model is None else f"fit reused from {fit_from}, elapsed"
    if model is None:
        model = fit_model(examples, context.grid[0], context.frame_step, context.dead_ball,
                          context.team_features, context.switch_prior, context.fitting)
    fitted = time.perf_counter()
    stored, adaptation = {}, {}
    for item in held:
        components = match_components(model, item)
        save_evidence(evidence_path(path, item.key), components)
        stored[item.key] = [scored.to_dict() for scored in
                            train.score_grid(model, item, components.evidence, context.grid)]
        adaptation[item.key] = components.evidence.adaptation
    train.store_fit(path, model, digest, stored, adaptation)
    print(f"{variant} fold {key}: {source} {fitted - start:.0f}s  grid "
          f"{time.perf_counter() - fitted:.0f}s ({len(held)} held out)", flush=True)
    return train.held_out_folds(model, held, stored, adaptation, key)


def run_job(job: Job) -> list[train.Fold]:
    loaded = [load_example(example_dir(job.features, key), job.lines) for key in job.members]
    members = train.members_of(job.variant, loaded)
    held = [item for item in members if item.key in job.held]
    return run_quick_fold(job.context(members), job.variant, members, held, job.key, job.fit_from)


def fold_jobs(features: Path, context: train.Context, variant: str,
              members: list[train.Loaded], fit_from: Path | None = None) -> list[Job]:
    return [Job(features=features, out=context.out, variant=variant,
                members=tuple(item.key for item in members),
                held=tuple(item.key for item in held), key=key, seed=context.seed,
                frame_step=context.frame_step, grid=tuple(context.grid),
                fingerprint=context.fingerprint, dead_ball=context.dead_ball,
                team_features=context.team_features, switch_prior=context.switch_prior,
                fitting=context.fitting, lines=context.lines, fit_from=fit_from)
            for key, held in train.holdouts(members)]


def run_folds(features: Path, context: train.Context, variant: str, workers: int,
              fit_from: Path | None = None) -> tuple[list[train.Loaded], list[train.Fold]]:
    members = train.members_of(variant, context.loaded)
    jobs = fold_jobs(features, context, variant, members, fit_from)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run_job, jobs))
    else:
        results = [run_job(job) for job in jobs]
    return members, [fold for folds in results for fold in folds]


def fold_evidence(out: Path, variant: str, folds: list[train.Fold]) -> dict[str, train.Evidence]:
    return {fold.key: load_evidence(evidence_path(fold_path(out, variant, fold.model_key), fold.key))
            for fold in folds}


def select_settings(variant: str, grid: list[DecoderSettings], members: list[train.Loaded],
                    folds: list[train.Fold], evidence: dict[str, train.Evidence]
                    ) -> train.Selection:
    search = train.GridSearch(variant, members, folds, grid, evidence=dict(evidence))
    final_view, selected = search.select()
    nested = [search.select(index) for index in range(len(folds))]
    return train.Selection(folds=search.folds, grid=search.grid,
                           final_grid=[search.grid[index] for index in final_view],
                           selected=selected,
                           fold_grids=[[search.grid[index] for index in view] for view, _ in nested],
                           choices=[choice for _, choice in nested])


def held_out_rows(selection: train.Selection) -> list[dict[str, object]]:
    return [{"match": fold.key, "source": fold.source,
             **asdict(train.effective_settings(fold.model, selection.grid[choice])),
             "searched_max_switch_weight": train.top_weight(searched),
             "searched_max_stride": train.top_stride(searched),
             **fold.scores[choice].to_dict(), "adaptation": fold.adaptation}
            for fold, choice, searched in zip(selection.folds, selection.choices,
                                              selection.fold_grids)]


def report(variant: str, selection: train.Selection, config: dict[str, object]) -> dict[str, object]:
    rows = held_out_rows(selection)
    return {**config, "variant": variant, "grid": [asdict(value) for value in selection.grid],
            "final_selection_grid": train.grid_axes(selection.final_grid),
            "selected": asdict(selection.grid[selection.selected]),
            "summary": train.variant_summary(rows), "held_out": rows}


def comparable(row: dict[str, object]) -> dict[str, float]:
    return row["raw"] if row["fair"] is None else row["fair"]


def print_report(result: dict[str, object]) -> None:
    print(f"== {result['variant']}: {len(result['held_out'])} folds, selected "
          f"{train.describe(DecoderSettings(**result['selected']))}")
    print(f"{'held-out match (nested)':<24}{'spellF1':>10}{'macroF1':>10}{'spells pred':>12}"
          f"{'spells true':>12}{'setting':>28}")
    for row in result["held_out"]:
        fair = row["fair"] is not None
        setting = DecoderSettings(emission_weight=row["emission_weight"], stride=row["stride"],
                                  switch_weight=row["switch_weight"])
        print(train.format_row(f"{row['match']}{' fair' if fair else ''}", comparable(row))
              + f"{train.describe(setting):>28}")
    for source, summary in result["summary"].items():
        if summary is not None:
            print(f"mean {source:<12} spellF1 {train.interval(summary)}  macroF1 "
                  f"{summary['macro_f1']:.4f}  ({summary['n_matches']} matches)")


def write_report(out: Path, result: dict[str, object]) -> Path:
    path = out if out.suffix == ".json" else out.parent / REPORT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def compare_reports(base: dict[str, object], other: dict[str, object]) -> dict[str, object]:
    before = {row["match"]: comparable(row) for row in base["held_out"]}
    after = {row["match"]: comparable(row) for row in other["held_out"]}
    rows = []
    for match in before:
        if match not in after:
            continue
        rows.append({"match": match, "source": match.split("/")[0],
                     "base": before[match]["spell_f1_mean"], "other": after[match]["spell_f1_mean"],
                     "delta": after[match]["spell_f1_mean"] - before[match]["spell_f1_mean"]})
    sources = {}
    for source in dict.fromkeys(row["source"] for row in rows):
        deltas = np.array([row["delta"] for row in rows if row["source"] == source])
        sources[source] = {"base": float(np.mean([row["base"] for row in rows
                                                  if row["source"] == source])),
                           "other": float(np.mean([row["other"] for row in rows
                                                   if row["source"] == source])),
                           "delta": float(deltas.mean()), "won": int((deltas > 0).sum()),
                           "lost": int((deltas < 0).sum()), "folds": int(deltas.size),
                           "delta_se": float(deltas.std(ddof=1) / np.sqrt(deltas.size))
                           if deltas.size > 1 else float("nan")}
    return {"folds": rows, "sources": sources}


def print_comparison(comparison: dict[str, object]) -> None:
    print(f"{'match':<24}{'base':>10}{'other':>10}{'delta':>10}")
    for row in comparison["folds"]:
        print(f"{row['match']:<24}{row['base']:>10.4f}{row['other']:>10.4f}{row['delta']:>+10.4f}")
    for source, summary in comparison["sources"].items():
        print(f"{source:<12} mean {summary['base']:.4f} -> {summary['other']:.4f} "
              f"({summary['delta']:+.4f} +- {summary['delta_se']:.4f} s.e.)  won "
              f"{summary['won']}/{summary['folds']}  lost {summary['lost']}/{summary['folds']}")


def model_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dead-features", choices=DEAD_FEATURE_SETS, default=DeadBallSettings.features)
    parser.add_argument("--adaptation", action="store_true")
    parser.add_argument("--team-features", choices=TEAM_FEATURE_SETS, default=DEFAULT_TEAM_FEATURES)
    parser.add_argument("--no-switch-prior", action="store_true")
    parser.add_argument("--spell-weights", action="store_true")
    parser.add_argument("--every-peak", action="store_true")
    parser.add_argument("--stopping", choices=STOPPING_RULES, default=STOPPING_RULES[0])
    parser.add_argument("--line-features", nargs="*", choices=LINE_GROUPS, default=[])
    parser.add_argument("--dead-context", action="store_true")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)


def grid_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spans", type=float, nargs="*", default=list(train.EMISSION_SPANS),
                        help="emission spans (weight 1/span) of the initial decoder grid")
    parser.add_argument("--strides", type=int, nargs="*", default=list(train.STRIDES))
    parser.add_argument("--switch-weights", type=float, nargs="*", default=list(train.SWITCH_WEIGHTS))


def parse_grid(args: argparse.Namespace) -> list[DecoderSettings]:
    weights = args.switch_weights if not args.no_switch_prior else [0.0]
    return [DecoderSettings(emission_weight=1.0 / span, stride=stride, switch_weight=weight)
            for span in args.spans for stride in args.strides for weight in weights]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick nested leave-one-match-out evaluation on cached frame features: "
                    "folds in parallel processes, held-out emissions stored next to the fold "
                    "models, decoder-only studies and fold-by-fold comparisons")
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache", help="compute and store the frame features of the "
                                              "degraded external matches (needs the reference "
                                              "statistics of the target match)")
    cache.add_argument("--features", type=cli_path, default=default_features())
    cache.add_argument("--matches")
    cache.add_argument("--sources", default=",".join(QUICK_SOURCES))
    cache.add_argument("--line-features", nargs="*", choices=LINE_GROUPS, default=[])
    cache.add_argument("--seed", type=int, default=DEFAULT_SEED)
    folds = commands.add_parser("folds", help="fit every leave-one-match-out fold of a variant on "
                                              "the cached features and select the decoder "
                                              "setting per fold on the other folds")
    folds.add_argument("--features", type=cli_path, default=default_features())
    folds.add_argument("--out", type=cli_path, required=True)
    folds.add_argument("--variant", choices=sorted(train.VARIANTS), default="dfl+metrica")
    folds.add_argument("--matches")
    folds.add_argument("--workers", type=int, default=1)
    folds.add_argument("--fit-from", type=cli_path,
                       help="model path of another run whose fold models are reused instead of "
                            "fitting when this run's fold cache misses; only the emission, "
                            "adaptation and decoding code of the current tree is exercised")
    model_flags(folds)
    grid_flags(folds)
    decode = commands.add_parser("decode", help="re-score a decoder grid on the stored held-out "
                                                "emissions of a fold directory, nested selection")
    decode.add_argument("--features", type=cli_path, default=default_features())
    decode.add_argument("--folds", type=cli_path, required=True,
                        help="model path whose folds/<variant> directory holds the fold models")
    decode.add_argument("--variant", choices=sorted(train.VARIANTS), default="dfl+metrica")
    decode.add_argument("--out", type=cli_path, required=True,
                        help=f"report path ending in .json, otherwise {REPORT_NAME} next to it")
    decode.add_argument("--workers", type=int, default=1)
    decode.add_argument("--no-switch-prior", action="store_true")
    grid_flags(decode)
    compare = commands.add_parser("compare", help="fold-by-fold comparison of two quick reports")
    compare.add_argument("base", type=cli_path)
    compare.add_argument("other", type=cli_path)
    return parser.parse_args()


def names_of(text: str | None) -> set[str] | None:
    return set(text.split(",")) if text is not None else None


def fold_models(out: Path, variant: str) -> list[Path]:
    paths = [path for path in sorted(fold_dir(out, variant).glob("*.pkl"))
             if path.stem != train.FINAL_NAME]
    if not paths:
        raise ValueError(f"{fold_dir(out, variant)} holds no fold models")
    return paths


def fold_keys(out: Path, variant: str) -> list[tuple[str, str]]:
    keys = []
    for path in fold_models(out, variant):
        stored = json.loads(train.scores_path(path).read_text())
        keys += [(path.stem, str(key)) for key in stored["scores"]]
    return keys


@dataclass(frozen=True)
class DecodeJob:
    features: Path
    out: Path
    variant: str
    fold_key: str
    held: str
    grid: tuple[DecoderSettings, ...]
    lines: tuple[str, ...]
    dead_labels: bool


def run_decode(job: DecodeJob) -> train.Fold:
    path = fold_path(job.out, job.variant, job.fold_key)
    model = load_model(path)
    item = load_example(example_dir(job.features, job.held), job.lines)
    evidence = load_evidence(evidence_path(path, job.held))
    scores = train.score_grid(model, item, evidence, list(job.grid))
    return train.Fold(key=item.key, source=item.source, model=model, scores=scores,
                      adaptation=evidence.adaptation, dead_labels=job.dead_labels,
                      fold_key=job.fold_key)


def decode_folds(features: Path, out: Path, variant: str, grid: list[DecoderSettings],
                 workers: int, lines: tuple[str, ...]
                 ) -> tuple[list[train.Loaded], list[train.Fold]]:
    keys = fold_keys(out, variant)
    members = train.members_of(variant, [load_example(example_dir(features, held), lines)
                                         for _, held in keys])
    dead_labels = {item.key: item.example.dead_labels for item in members}
    jobs = [DecodeJob(features=features, out=out, variant=variant, fold_key=fold_key, held=held,
                      grid=tuple(grid), lines=lines, dead_labels=dead_labels[held])
            for fold_key, held in keys]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return members, list(pool.map(run_decode, jobs))
    return members, [run_decode(job) for job in jobs]


def model_config(args: argparse.Namespace, grid: list[DecoderSettings]) -> dict[str, object]:
    fitting = ClassifierSettings(spell_weights=args.spell_weights, every_peak=args.every_peak,
                                 stopping=args.stopping, dead_context=args.dead_context)
    lines = tuple(group for group in LINE_GROUPS if group in args.line_features)
    return {"seed": args.seed, "frame_step": args.frame_step,
            "dead_ball": asdict(train.dead_ball_settings(args.dead_features, args.adaptation)),
            "team_features": args.team_features, "switch_prior": not args.no_switch_prior,
            **train.fitting_record(fitting), **train.line_record(lines),
            "initial_grid": train.grid_axes(grid)}


def main_folds(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    grid = parse_grid(args)
    lines = tuple(group for group in LINE_GROUPS if group in args.line_features)
    fitting = ClassifierSettings(spell_weights=args.spell_weights, every_peak=args.every_peak,
                                 stopping=args.stopping, dead_context=args.dead_context)
    names = names_of(args.matches)
    sources = train.VARIANTS[args.variant]
    loaded = load_examples(args.features, names, sources, lines,
                           current_data_keys(args.seed, names, sources))
    train.check_variants([args.variant], [item.source for item in loaded])
    context = train.Context(seed=args.seed, frame_step=args.frame_step, grid=grid, loaded=loaded,
                            out=args.out, fingerprint=code_fingerprint(*train.FIT_MODULES),
                            dead_ball=train.dead_ball_settings(args.dead_features, args.adaptation),
                            outside_sources=(), team_features=args.team_features,
                            switch_prior=not args.no_switch_prior, fitting=fitting, lines=lines)
    print(f"{len(loaded)} cached matches; {len(grid)} initial settings; {args.workers} workers",
          flush=True)
    members, folds = run_folds(args.features, context, args.variant, args.workers, args.fit_from)
    evidence = fold_evidence(args.out, args.variant, folds)
    selection = select_settings(args.variant, grid, members, folds, evidence)
    config = {**model_config(args, grid),
              **({"fit_from": str(args.fit_from)} if args.fit_from is not None else {})}
    result = report(args.variant, selection, config)
    print_report(result)
    print(f"report {write_report(args.out, result)}  runtime {time.perf_counter() - start:.0f}s")


def main_decode(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    grid = parse_grid(args)
    metadata = read_metadata(fold_models(args.folds, args.variant)[0])
    lines = tuple(metadata.get("line_features") or ())
    members, folds = decode_folds(args.features, args.folds, args.variant, grid, args.workers,
                                  lines)
    evidence = fold_evidence(args.folds, args.variant, folds)
    selection = select_settings(args.variant, grid, members, folds, evidence)
    config = {"folds": str(fold_dir(args.folds, args.variant)), "initial_grid": train.grid_axes(grid)}
    result = report(args.variant, selection, config)
    print_report(result)
    print(f"report {write_report(args.out, result)}  runtime {time.perf_counter() - start:.0f}s")


def main() -> None:
    args = parse_args()
    if args.command == "cache":
        lines = tuple(group for group in LINE_GROUPS if group in args.line_features)
        sources = tuple(source for source in SOURCES if source in args.sources.split(","))
        build_cache(args.features, args.seed, names_of(args.matches), sources, lines)
    elif args.command == "folds":
        main_folds(args)
    elif args.command == "decode":
        main_decode(args)
    else:
        comparison = compare_reports(json.loads(args.base.read_text()),
                                     json.loads(args.other.read_text()))
        print_comparison(comparison)


if __name__ == "__main__":
    main()
