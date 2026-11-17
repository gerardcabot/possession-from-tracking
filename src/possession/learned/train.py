import argparse
import hashlib
import json
import math
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from possession.external.match import ExternalMatch, dead_labels_trusted
from possession.external.prepare import (
    DEFAULT_SEED,
    code_fingerprint,
    degraded_path,
    reference_params,
    reference_stats,
    variant_match,
)
from possession.external.sources import (
    DFL,
    METRICA,
    PFF,
    SKILLCORNER,
    SOURCES,
    MatchFiles,
    discover,
    limit_per_source,
)
from possession.external.validate import format_row, mean_scores, score_fair, score_labels
from possession.data import fps
from possession.intervals import LABELS
from possession.learned.adapt import SAFETY_BOUND
from possession.learned.classifier import STOPPING_RULES
from possession.learned.external import (
    fold_dir,
    fold_path,
    labels_only,
    save_group_pointer,
    spilled,
    training_example,
)
from possession.learned.framefeatures import (
    DEAD_FEATURE_SETS,
    DEFAULT_TEAM_FEATURES,
    LINE_GROUPS,
    TEAM_FEATURE_SETS,
)
from possession.learned.model import (
    DEFAULT_FITTING,
    MODEL_FORMAT,
    MODEL_SEED,
    ClassifierSettings,
    DeadBallSettings,
    DecoderSettings,
    Entering,
    LearnedModel,
    TrainingExample,
    decode_emission,
    effective_settings,
    entry_evidence,
    fit_model,
    for_inference,
    load_model,
    log_adaptation,
    log_emission,
    model_digest,
    save_model,
)
from possession.paths import DEFAULT_MODEL_PATH, cli_path

EMISSION_SPANS = (1, 2, 5, 10, 25, 50, 100)
STRIDES = (5, 10, 25)
SWITCH_WEIGHTS = (0.0, 0.5, 1.0, 2.0)
SERIES_MANTISSAS = (1, 2, 5)
SELECTION_METRICS = ("spell_f1_mean", "macro_f1")
VARIANTS: dict[str, tuple[str, ...]] = {
    "dfl": (DFL,),
    "dfl+metrica": (DFL, METRICA),
    "dfl+metrica+pff": (DFL, METRICA, PFF),
    "dfl+metrica+pff_ha": (DFL, METRICA, PFF),
    "dfl+metrica+skillcorner_ha": (DFL, METRICA, SKILLCORNER),
    "metrica": (METRICA,),
}
TEAM_ONLY: dict[str, frozenset[str]] = {"dfl+metrica+pff_ha": frozenset({PFF})}
DEFAULT_VARIANTS = ("dfl", "dfl+metrica", "metrica")
ALL_VARIANTS = "all"
MIN_MATCHES_PER_SOURCE = 2
METRICA_RAW = f"{METRICA}_raw"
PFF_RAW = f"{PFF}_raw"
SELECTION_SCORES = {"raw": METRICA_RAW, "fair": METRICA}
DEFAULT_SELECTION_SCORE = "raw"
SCORED_SOURCES = (DFL, METRICA, SKILLCORNER, PFF)
ALWAYS_SCORED = (DFL, METRICA, SKILLCORNER)
OPT_IN_OUTSIDE = (PFF,)
RAW_SUMMARIES = {METRICA_RAW: METRICA, PFF_RAW: PFF}
SUMMARY_ONLY_SOURCES = (SKILLCORNER, PFF)
SPELL_KEYS = ("spell_f1_mean", "spell_bootstrap_low", "spell_bootstrap_high")
TIE_SOURCE = DFL
LAST_SOURCE = SKILLCORNER
BOOTSTRAP_DRAWS = 1000
REPORT_NAME = "variant_report.json"
SPILL_PREFIX = "possession-features-"
FINAL_NAME = "final"
SCORES_SUFFIX = ".scores.json"
OMITTED_AT_DEFAULT = ("dead_context",)
HELD_OUT_FOLD = "fold"
HELD_OUT_GROUP = "group fold"
HELD_OUT_FINAL = "final model"
FIT_MODULES = ("possession.learned.train", "possession.learned.model",
               "possession.learned.adapt", "possession.learned.changes",
               "possession.learned.classifier", "possession.learned.deadcontext",
               "possession.learned.hsmm", "possession.learned.framefeatures",
               "possession.learned.external",
               "possession.external.validate", "possession.external.match",
               "possession.external.predict", "possession.external.sources",
               "possession.data", "possession.features", "possession.cleaning",
               "possession.geometry", "possession.intervals",
               "possession.vendor.possession_metrics")


@dataclass(frozen=True)
class Scored:
    raw: dict[str, float]
    fair: dict[str, float] | None

    @property
    def comparable(self) -> dict[str, float]:
        return self.raw if self.fair is None else self.fair

    def to_dict(self) -> dict[str, object]:
        return {"raw": self.raw, "fair": self.fair}

    @staticmethod
    def from_dict(values: dict[str, dict[str, float] | None]) -> "Scored":
        raw = values["raw"]
        if raw is None:
            raise ValueError("stored scores have no raw entry")
        return Scored(raw=raw, fair=values["fair"])


@dataclass(frozen=True)
class Loaded:
    example: TrainingExample
    match: ExternalMatch
    data_key: str

    @property
    def key(self) -> str:
        return self.example.name

    @property
    def source(self) -> str:
        return self.match.source


@dataclass(frozen=True)
class Fold:
    key: str
    source: str
    model: LearnedModel
    scores: list[Scored]
    adaptation: dict[str, object] | None = None
    group: tuple[str, ...] = ()
    dead_labels: bool = True
    fold_key: str | None = None

    @property
    def model_key(self) -> str:
        return self.key if self.fold_key is None else self.fold_key


def fitting_record(fitting: ClassifierSettings) -> dict[str, object]:
    if fitting == DEFAULT_FITTING:
        return {}
    defaults = asdict(DEFAULT_FITTING)
    return {"classifier": {name: value for name, value in asdict(fitting).items()
                           if name not in OMITTED_AT_DEFAULT or value != defaults[name]}}


def line_record(lines: tuple[str, ...]) -> dict[str, object]:
    return {"line_features": list(lines)} if lines else {}


@dataclass(frozen=True)
class Context:
    seed: int
    frame_step: int
    grid: list[DecoderSettings]
    loaded: list[Loaded]
    out: Path
    fingerprint: str
    dead_ball: DeadBallSettings = DeadBallSettings()
    outside_sources: tuple[str, ...] = ALWAYS_SCORED
    team_features: str = DEFAULT_TEAM_FEATURES
    switch_prior: bool = True
    fitting: ClassifierSettings = DEFAULT_FITTING
    lines: tuple[str, ...] = ()

    def fit_inputs(self) -> dict[str, object]:
        return {"format": MODEL_FORMAT, "seed": self.seed, "model_seed": MODEL_SEED,
                "frame_step": self.frame_step, "bootstrap_draws": BOOTSTRAP_DRAWS,
                "cleaning": self.loaded[0].example.params.to_dict(), "code": self.fingerprint,
                "dead_ball": asdict(self.dead_ball), "team_features": self.team_features,
                "switch_prior": self.switch_prior, **fitting_record(self.fitting),
                **line_record(self.lines)}

    def data_keys(self, keys: list[str]) -> dict[str, str]:
        by_key = {item.key: item.data_key for item in self.loaded}
        return {key: by_key[key] for key in keys}


@dataclass(frozen=True)
class VariantResult:
    name: str
    members: list[str]
    folds: list[Fold]
    grid: list[DecoderSettings]
    final_grid: list[DecoderSettings]
    fold_grids: list[list[DecoderSettings]]
    choices: list[int]
    selected: int
    model: LearnedModel
    outside: dict[str, Scored]
    outside_adaptation: dict[str, dict[str, object] | None] = field(default_factory=dict)

    def held_out(self) -> list[dict[str, object]]:
        rows = [{"match": fold.key, "source": fold.source,
                 "held_out_by": HELD_OUT_GROUP if fold.group else HELD_OUT_FOLD,
                 "group": list(fold.group),
                 **asdict(effective_settings(fold.model, self.grid[choice])),
                 "searched_max_switch_weight": top_weight(searched),
                 "searched_max_stride": top_stride(searched),
                 **fold.scores[choice].to_dict(),
                 "adaptation": fold.adaptation}
                for fold, choice, searched in zip(self.folds, self.choices, self.fold_grids)]
        rows += [{"match": key, "source": key.split("/")[0], "held_out_by": HELD_OUT_FINAL,
                  **asdict(self.model.settings), **scored.to_dict(),
                  "adaptation": self.outside_adaptation.get(key)}
                 for key, scored in self.outside.items()]
        return rows


def settings_grid(switch_prior: bool = True) -> list[DecoderSettings]:
    weights = SWITCH_WEIGHTS if switch_prior else SWITCH_WEIGHTS[:1]
    return [DecoderSettings(emission_weight=1.0 / span, stride=stride, switch_weight=weight)
            for span in EMISSION_SPANS for stride in STRIDES for weight in weights]


def next_in_series(value: float) -> float:
    exponent = math.floor(math.log10(value))
    candidates = (mantissa * 10.0 ** power for power in (exponent, exponent + 1)
                  for mantissa in SERIES_MANTISSAS)
    return next(candidate for candidate in candidates
                if candidate > value and not math.isclose(candidate, value))


def top_weight(grid: list[DecoderSettings]) -> float:
    return max(settings.switch_weight for settings in grid)


def top_stride(grid: list[DecoderSettings]) -> int:
    return max(settings.stride for settings in grid)


def next_stride(stride: int) -> int:
    seconds = stride / fps()
    while True:
        seconds = next_in_series(seconds)
        frames = seconds * fps()
        if math.isclose(frames, round(frames)):
            return round(frames)


def grid_edges(grid: list[DecoderSettings], chosen: list[int]) -> tuple[bool, bool]:
    weight, stride = top_weight(grid), top_stride(grid)
    return (weight > 0.0 and any(grid[index].switch_weight == weight for index in chosen),
            any(grid[index].stride == stride for index in chosen))


def grid_extension(grid: list[DecoderSettings], weight_edge: bool, stride_edge: bool
                   ) -> list[DecoderSettings]:
    spans = sorted({settings.emission_weight for settings in grid}, reverse=True)
    strides = sorted({settings.stride for settings in grid})
    weights = sorted({settings.switch_weight for settings in grid})
    if stride_edge:
        strides.append(next_stride(strides[-1]))
    if weight_edge:
        weights.append(next_in_series(weights[-1]))
    present = set(grid)
    wanted = [DecoderSettings(emission_weight=span, stride=stride, switch_weight=weight)
              for span in spans for stride in strides for weight in weights]
    return [settings for settings in wanted if settings not in present]


def grid_axes(grid: list[DecoderSettings]) -> dict[str, list[float]]:
    return {"emission_weights": sorted({settings.emission_weight for settings in grid}),
            "strides": sorted({settings.stride for settings in grid}),
            "switch_weights": sorted({settings.switch_weight for settings in grid})}


def describe(settings: DecoderSettings) -> str:
    return (f"span {round(1.0 / settings.emission_weight):>3} stride {settings.stride:>2} "
            f"switch {settings.switch_weight:g}")


def parse_variants(text: str) -> list[str]:
    names = list(VARIANTS) if text == ALL_VARIANTS else text.split(",")
    unknown = sorted(set(names) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choose from {sorted(VARIANTS)} or "
                         f"{ALL_VARIANTS}")
    return [name for name in VARIANTS if name in names]


def file_key(files: MatchFiles) -> str:
    return f"{files.source}/{files.name}"


def scored_sources(variants: list[str], opted_in: tuple[str, ...] = ()) -> tuple[str, ...]:
    used = {source for variant in variants for source in VARIANTS[variant]}
    return tuple(source for source in SCORED_SOURCES
                 if source in ALWAYS_SCORED or source in used or source in opted_in)


def selected_files(names: set[str] | None, limit: int | None = None,
                   sources: tuple[str, ...] = ALWAYS_SCORED) -> list[MatchFiles]:
    available = [files for source in sources for files in discover(source)]
    known = {file_key(files) for files in available} | {files.name for files in available}
    unknown = sorted((names or set()) - known)
    if unknown:
        raise ValueError(f"unknown matches {unknown}; available "
                         f"{sorted(file_key(files) for files in available)}")
    ambiguous = {name: sorted(file_key(files) for files in available if files.name == name)
                 for name in names or set() if "/" not in name}
    ambiguous = {name: keys for name, keys in ambiguous.items() if len(keys) > 1}
    if ambiguous:
        raise ValueError(f"ambiguous match names {ambiguous}; use source/name")
    chosen = [files for files in available
              if names is None or files.name in names or file_key(files) in names]
    return limit_per_source(chosen, limit)


def load_matches(seed: int, files: list[MatchFiles], spill_dir: Path,
                 touch: bool = True, lines: tuple[str, ...] = ()) -> list[Loaded]:
    stats = reference_stats()
    params = reference_params()
    loaded = []
    for item in files:
        start = time.perf_counter()
        match = variant_match(item, stats, seed, "degraded", force=False)
        example = training_example(match, params, touch, lines)
        loaded.append(Loaded(example=spilled(example, spill_dir), match=labels_only(match),
                             data_key=degraded_path(item, stats, seed).name))
        print(f"loaded {file_key(item)} in {time.perf_counter() - start:.0f}s", flush=True)
    return loaded


def check_variants(variants: list[str], sources: list[str]) -> None:
    problems = []
    for variant in variants:
        for source in VARIANTS[variant]:
            found = sources.count(source)
            if found < MIN_MATCHES_PER_SOURCE:
                problems.append(f"variant {variant} needs {MIN_MATCHES_PER_SOURCE} {source} "
                                f"matches for leave-one-match-out, found {found}")
    if problems:
        raise ValueError("; ".join(problems))


def team_only(item: Loaded) -> Loaded:
    return replace(item, example=replace(item.example, dead_labels=False))


def members_of(variant: str, loaded: list[Loaded]) -> list[Loaded]:
    excluded = TEAM_ONLY.get(variant, frozenset())
    members = [team_only(item) if item.source in excluded else item
               for item in loaded if item.source in VARIANTS[variant]]
    if len(members) < 2:
        raise ValueError(f"variant {variant} needs at least two matches for "
                         f"leave-one-match-out, found {len(members)}")
    return members


def reference_count(counts: Counter[str]) -> int | None:
    return max((count for source, count in counts.items() if not SOURCES[source].grouped_folds),
               default=None)


def holdouts(members: list[Loaded]) -> list[tuple[str, list[Loaded]]]:
    reference = reference_count(Counter(item.source for item in members))
    result: list[tuple[str, list[Loaded]]] = []
    for source in dict.fromkeys(item.source for item in members):
        items = [item for item in members if item.source == source]
        if not SOURCES[source].grouped_folds or reference is None:
            result += [(item.key, [item]) for item in items]
            continue
        chunks = np.array_split(np.arange(len(items)), min(reference, len(items)))
        result += [(f"{source}/group{number}of{len(chunks)}", [items[index] for index in chunk])
                   for number, chunk in enumerate(chunks, start=1)]
    return result


def row_steps(items: list[Loaded]) -> dict[str, int]:
    counts = Counter(item.source for item in items)
    reference = reference_count(counts)
    return {source: math.ceil(count / reference)
            if SOURCES[source].grouped_folds and reference else 1
            for source, count in counts.items()}


def fit_examples(items: list[Loaded]) -> list[TrainingExample]:
    steps = row_steps(items)
    return [replace(item.example, row_step=steps[item.source]) for item in items]


def score_prediction(match: ExternalMatch, labels: np.ndarray) -> Scored:
    trusted = dead_labels_trusted(match)
    raw = score_labels(match, labels, BOOTSTRAP_DRAWS)
    return Scored(raw=raw, fair=None if trusted else score_fair(match, labels, BOOTSTRAP_DRAWS))


@dataclass(frozen=True)
class Evidence:
    emission: np.ndarray
    entering: Entering | None
    adaptation: dict[str, object] | None


def predicted_labels(model: LearnedModel, evidence: Evidence, item: Loaded,
                     settings: DecoderSettings) -> np.ndarray:
    return LABELS[decode_emission(evidence.emission, item.example.blocks, model.segments,
                                  effective_settings(model, settings), evidence.entering)]


def match_emission(model: LearnedModel, item: Loaded) -> Evidence:
    base = item.example.base[0]
    emission, report = log_emission(model, base, item.example.blocks)
    log_adaptation(item.key, report)
    return Evidence(emission=emission, entering=entry_evidence(model, base),
                    adaptation=None if report is None else report.to_dict())


def score_grid(model: LearnedModel, item: Loaded, evidence: Evidence,
               grid: list[DecoderSettings]) -> list[Scored]:
    scored: dict[bytes, Scored] = {}
    scores = []
    for settings in grid:
        labels = predicted_labels(model, evidence, item, settings)
        key = labels.tobytes()
        if key not in scored:
            scored[key] = score_prediction(item.match, labels)
        scores.append(scored[key])
    return scores


def grid_scores(model: LearnedModel, item: Loaded, grid: list[DecoderSettings]
                ) -> tuple[list[Scored], dict[str, object] | None]:
    evidence = match_emission(model, item)
    return score_grid(model, item, evidence, grid), evidence.adaptation


def switch_summary(model: LearnedModel) -> dict[str, float | int] | None:
    if model.changes is None:
        return None
    return {"threshold": model.changes.threshold, "peaks": model.changes.peaks,
            "rows": model.changes.rows}


def describe_switch(model: LearnedModel) -> str:
    summary = switch_summary(model)
    if summary is None:
        return "switch prior none"
    return (f"switch prior threshold {summary['threshold']:.2f} from {summary['peaks']} peaks, "
            f"{summary['rows']} rows")


def fit_digest(context: Context, train: list[TrainingExample], evaluated: list[str],
               settings: list[DecoderSettings]) -> str:
    trained = [example.name for example in train]
    roles = [[example.name, example.dead_labels, example.row_step] for example in train]
    content = {**context.fit_inputs(), "train": trained, "roles": roles, "evaluated": evaluated,
               "data": context.data_keys([*trained, *evaluated]),
               "settings": [asdict(value) for value in settings]}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def scores_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.stem + SCORES_SUFFIX)


def cached_fit(model_path: Path, digest: str) -> tuple[LearnedModel, object, object] | None:
    path = scores_path(model_path)
    if not (path.is_file() and model_path.is_file()):
        return None
    stored = json.loads(path.read_text())
    if stored.get("digest") != digest:
        return None
    return load_model(model_path), stored["scores"], stored["adaptation"]


def store_fit(model_path: Path, model: LearnedModel, digest: str, scores: object,
              adaptation: object) -> None:
    save_model(model_path, model, digest)
    scores_path(model_path).write_text(json.dumps({"digest": digest, "scores": scores,
                                                   "adaptation": adaptation}, indent=2) + "\n")


def held_out_folds(model: LearnedModel, held: list[Loaded], stored: dict[str, object],
                   adaptation: dict[str, object], key: str) -> list[Fold]:
    group = tuple(item.key for item in held) if len(held) > 1 else ()
    return [Fold(key=item.key, source=item.source, model=model,
                 scores=[Scored.from_dict(values) for values in stored[item.key]],
                 adaptation=adaptation[item.key], group=group,
                 dead_labels=item.example.dead_labels, fold_key=key) for item in held]


def run_fold(context: Context, variant: str, members: list[Loaded], held: list[Loaded],
             key: str) -> list[Fold]:
    keys = {item.key for item in held}
    train = fit_examples([item for item in members if item.key not in keys])
    path = fold_path(context.out, variant, key)
    digest = fit_digest(context, train, [item.key for item in held], context.grid)
    cached = cached_fit(path, digest)
    if cached is not None:
        model, stored, adaptation = cached
        print(f"{variant} fold {key}: reused {path.name}", flush=True)
        return held_out_folds(model, held, stored, adaptation, key)
    start = time.perf_counter()
    model = fit_model(train, context.grid[0], context.frame_step, context.dead_ball,
                      context.team_features, context.switch_prior, context.fitting)
    fitted = time.perf_counter()
    stored, adaptation = {}, {}
    for item in held:
        scores, adaptation[item.key] = grid_scores(model, item, context.grid)
        stored[item.key] = [scored.to_dict() for scored in scores]
    store_fit(path, model, digest, stored, adaptation)
    print(f"{variant} fold {key}: scales {list(model.layout.all_scales)}  touch "
          f"{model.layout.touch} threshold {model.layout.touch_threshold:.2f} from "
          f"{model.touch_peaks} peaks  iterations "
          f"{list(model.classifier.iterations)} {list(model.classifier.stopping)}  "
          f"{describe_switch(model)}  "
          f"fit {fitted - start:.0f}s  grid {time.perf_counter() - fitted:.0f}s "
          f"({len(held)} held out)", flush=True)
    return held_out_folds(model, held, stored, adaptation, key)


def selection_scores(folds: list[Fold]) -> np.ndarray:
    by_source: dict[str, list[np.ndarray]] = {}
    for fold in folds:
        if not fold.dead_labels:
            continue
        values = [[scored.comparable[metric] for metric in SELECTION_METRICS]
                  for scored in fold.scores]
        by_source.setdefault(fold.source, []).append(np.array(values))
    return np.mean([np.mean(rows, axis=0) for rows in by_source.values()], axis=0)


def best_setting(mean: np.ndarray, grid: list[DecoderSettings]) -> int:
    weights = np.array([settings.switch_weight for settings in grid])
    return int(np.lexsort((weights, -mean[:, 1], -mean[:, 0]))[0])


def improves(row: np.ndarray, best: np.ndarray) -> bool:
    return tuple(row.tolist()) > tuple(best.tolist())


@dataclass(frozen=True)
class Selection:
    folds: list[Fold]
    grid: list[DecoderSettings]
    final_grid: list[DecoderSettings]
    selected: int
    fold_grids: list[list[DecoderSettings]]
    choices: list[int]


@dataclass
class GridSearch:
    variant: str
    members: list[Loaded]
    folds: list[Fold]
    grid: list[DecoderSettings]
    evidence: dict[str, Evidence] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.initial = list(range(len(self.grid)))

    def add(self, settings: list[DecoderSettings]) -> list[int]:
        known = set(self.grid)
        fresh = [value for value in dict.fromkeys(settings) if value not in known]
        if fresh:
            by_key = {item.key: item for item in self.members}
            for fold in self.folds:
                if fold.key not in self.evidence:
                    self.evidence[fold.key] = match_emission(fold.model, by_key[fold.key])
            self.folds = [replace(fold, scores=[*fold.scores,
                                                *score_grid(fold.model, by_key[fold.key],
                                                            self.evidence[fold.key], fresh)])
                          for fold in self.folds]
            self.grid = [*self.grid, *fresh]
        position = {value: index for index, value in enumerate(self.grid)}
        return [position[value] for value in settings]

    def best(self, view: list[int], held: int | None) -> tuple[int, np.ndarray]:
        scores = selection_scores([fold for index, fold in enumerate(self.folds)
                                   if index != held])
        return view[best_setting(scores[view], [self.grid[index] for index in view])], scores

    def select(self, held: int | None = None) -> tuple[list[int], int]:
        view = list(self.initial)
        while True:
            best, _ = self.best(view, held)
            sub = [self.grid[index] for index in view]
            weight_edge, stride_edge = grid_edges(sub, [view.index(best)])
            if not (weight_edge or stride_edge):
                return view, best
            added = grid_extension(sub, weight_edge, stride_edge)
            print(f"{self.variant}: {'all folds' if held is None else f'fold {held} left out'}: "
                  f"the largest {'switch weight' if weight_edge else ''}"
                  f"{' and ' if weight_edge and stride_edge else ''}"
                  f"{'stride' if stride_edge else ''} was chosen; trying {len(added)} larger "
                  f"settings (switch weights up to "
                  f"{max(value.switch_weight for value in added):g}, strides up to "
                  f"{max(value.stride for value in added)})", flush=True)
            extended = [*view, *[index for index in self.add(added) if index not in view]]
            candidate, scores = self.best(extended, held)
            if candidate in view or not improves(scores[candidate], scores[best]):
                return extended, best
            view = extended


def select_settings(variant: str, grid: list[DecoderSettings], members: list[Loaded],
                    folds: list[Fold]) -> Selection:
    search = GridSearch(variant, members, folds, grid)
    final_view, selected = search.select()
    nested = [search.select(index) for index in range(len(folds))]
    return Selection(folds=search.folds, grid=search.grid,
                     final_grid=[search.grid[index] for index in final_view], selected=selected,
                     fold_grids=[[search.grid[index] for index in view] for view, _ in nested],
                     choices=[choice for _, choice in nested])


def run_final(context: Context, variant: str, members: list[Loaded],
              settings: DecoderSettings
              ) -> tuple[LearnedModel, dict[str, Scored], dict[str, dict[str, object] | None]]:
    inside = {item.key for item in members}
    outside = [item for item in context.loaded
               if item.key not in inside and item.source in context.outside_sources]
    path = fold_dir(context.out, variant) / f"{FINAL_NAME}.pkl"
    examples = fit_examples(members)
    digest = fit_digest(context, examples, [item.key for item in outside], [settings])
    cached = cached_fit(path, digest)
    if cached is not None:
        model, stored, adaptation = cached
        print(f"{variant} final model: reused {path.name}", flush=True)
        return model, {key: Scored.from_dict(values) for key, values in stored.items()}, adaptation
    start = time.perf_counter()
    model = for_inference(fit_model(examples, settings, context.frame_step, context.dead_ball,
                                    context.team_features,
                                    context.switch_prior and bool(settings.switch_weight),
                                    context.fitting),
                          settings)
    fitted = time.perf_counter()
    scores = {}
    adaptation = {}
    for item in outside:
        evidence = match_emission(model, item)
        adaptation[item.key] = evidence.adaptation
        scores[item.key] = score_prediction(item.match, predicted_labels(model, evidence, item,
                                                                         model.settings))
    store_fit(path, model, digest, {key: scored.to_dict() for key, scored in scores.items()},
              adaptation)
    print(f"{variant} final model: fit {fitted - start:.0f}s  scored {len(outside)} other "
          f"matches in {time.perf_counter() - fitted:.0f}s", flush=True)
    return model, scores, adaptation


def run_variant(context: Context, variant: str) -> VariantResult:
    members = members_of(variant, context.loaded)
    folds = [fold for key, held in holdouts(members)
             for fold in run_fold(context, variant, members, held, key)]
    chosen = select_settings(variant, context.grid, members, folds)
    model, outside, adaptation = run_final(context, variant, members, chosen.grid[chosen.selected])
    return VariantResult(name=variant, members=[item.key for item in members],
                         folds=chosen.folds, grid=chosen.grid, final_grid=chosen.final_grid,
                         fold_grids=chosen.fold_grids, choices=chosen.choices,
                         selected=chosen.selected, model=model, outside=outside,
                         outside_adaptation=adaptation)


def has_spells(scores: dict[str, float]) -> bool:
    return not any(np.isnan(scores[key]) for key in SPELL_KEYS if key in scores)


def source_summary(rows: list[dict[str, object]], source: str, mode: str
                   ) -> dict[str, float] | None:
    candidates = [row[mode] for row in rows if row["source"] == source and row[mode] is not None]
    picked = [scores for scores in candidates if has_spells(scores)]
    if not picked:
        return None
    return {**mean_scores(picked), "n_matches": len(picked),
            "n_without_spells": len(candidates) - len(picked)}


def variant_summary(rows: list[dict[str, object]]) -> dict[str, dict[str, float] | None]:
    summary = {}
    for source in SCORED_SOURCES:
        comparable = "raw" if source == DFL else "fair"
        summary[source] = source_summary(rows, source, comparable)
    for name, source in RAW_SUMMARIES.items():
        summary[name] = source_summary(rows, source, "raw")
    return summary


def tied_with_best(summaries: dict[str, dict], candidates: list[str], source: str) -> list[str]:
    scored = [name for name in candidates if summaries[name][source] is not None]
    if not scored:
        return candidates
    best = max(scored, key=lambda name: summaries[name][source]["spell_f1_mean"])
    floor = summaries[best][source]["spell_bootstrap_low"]
    tied = [name for name in scored if summaries[name][source]["spell_bootstrap_high"] >= floor]
    return tied or [best]


def select_variant(summaries: dict[str, dict], primary: str = METRICA) -> str:
    tie_sources = (primary, TIE_SOURCE)
    candidates = list(summaries)
    for source in tie_sources:
        candidates = tied_with_best(summaries, candidates, source)

    def order(name: str) -> tuple[float, ...]:
        return tuple(summaries[name][source]["spell_f1_mean"]
                     if summaries[name][source] is not None else -np.inf
                     for source in (LAST_SOURCE, *tie_sources))

    return max(candidates, key=order)


def interval(summary: dict[str, float] | None) -> str:
    if summary is None:
        return "-"
    if "spell_bootstrap_low" not in summary:
        return f"{summary['spell_f1_mean']:.4f}"
    return (f"{summary['spell_f1_mean']:.4f} [{summary['spell_bootstrap_low']:.3f},"
            f"{summary['spell_bootstrap_high']:.3f}]")


def print_variant(result: VariantResult, rows: list[dict[str, object]]) -> None:
    print(f"== variant {result.name}: trained on {len(result.members)} matches, selected "
          f"{describe(result.grid[result.selected])}")
    print(f"{'held-out match (nested)':<24}{'spellF1':>10}{'macroF1':>10}{'spells pred':>12}"
          f"{'spells true':>12}")
    for row in rows:
        if row["source"] in SUMMARY_ONLY_SOURCES:
            continue
        fair = row["fair"] is not None
        print(format_row(f"{row['match']}{' fair' if fair else ''}",
                         row["fair" if fair else "raw"]))


def print_table(summaries: dict[str, dict], selected: str, selection_score: str) -> None:
    columns = (METRICA, DFL, SKILLCORNER, METRICA_RAW, PFF, PFF_RAW)
    titles = ("metrica fair", "dfl", "skillcorner fair", "metrica raw", "pff fair", "pff raw")
    print(f"{"variant (mean spellF1, 5-95)":<30}"
          + "".join(f"{title:>26}" for title in titles))
    for name, summary in summaries.items():
        mark = " *" if name == selected else ""
        print(f"{name + mark:<30}" + "".join(f"{interval(summary[column]):>26}"
                                              for column in columns))
    print(f"selected: {selected} (highest Metrica {selection_score} spellF1; variants whose "
          f"interval, the mean of the per-match bootstrap bounds, reaches the best one's lower "
          f"bound are tied and decided by DFL, then by SkillCorner fair; PFF is reported, not "
          f"used to select)")


def save_folds(results: list[VariantResult], out: Path, run_id: str) -> None:
    for result in results:
        grid = result.grid
        saved_groups: set[str] = set()
        for fold, choice in zip(result.folds, result.choices):
            path = fold_path(out, result.name, fold.key)
            details = {"variant": result.name, "held_out": fold.key,
                       "held_out_group": list(fold.group or (fold.key,))}
            if not fold.group:
                save_model(path, for_inference(fold.model, grid[choice]), run_id, details)
                continue
            group_path = fold_path(out, result.name, fold.model_key)
            if fold.model_key not in saved_groups:
                saved_groups.add(fold.model_key)
                save_model(group_path, fold.model, run_id,
                           {"variant": result.name, "held_out_group": list(fold.group)})
            save_group_pointer(path, group_path, grid[choice], run_id, details)


def training_run_id(config: dict[str, object], digest: str) -> str:
    content = json.dumps({**config, "model_sha256": digest}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def partial_variants(variants: list[str]) -> bool:
    return not set(DEFAULT_VARIANTS) <= set(variants)


def output_path(out: Path | None, subset: bool) -> Path:
    if subset and (out is None or out.resolve() == DEFAULT_MODEL_PATH.resolve()):
        raise ValueError(f"--matches, --limit, --frame-step or a --variants subset trains a "
                         f"partial model; pass an explicit --out other than "
                         f"{DEFAULT_MODEL_PATH}")
    return DEFAULT_MODEL_PATH if out is None else out


def dead_ball_settings(features: str, adaptation: bool) -> DeadBallSettings:
    return DeadBallSettings(features=features,
                            adaptation_iterations=SAFETY_BOUND if adaptation else 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the learned method on degraded external matches for each training-set "
                    "variant, score every DFL, Metrica, SkillCorner and PFF match held out, select "
                    "the variant and save its model")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS),
                        help=f"comma-separated subset of {', '.join(VARIANTS)}, or "
                             f"{ALL_VARIANTS} (default {','.join(DEFAULT_VARIANTS)}; PFF and "
                             f"SkillCorner variants only when listed; a list that lacks a "
                             f"default variant needs --out)")
    parser.add_argument("--score-outside", nargs="*", choices=OPT_IN_OUTSIDE, default=[],
                        help="also score the matches of these sources with every variant that "
                             "does not train on them (default none: DFL, Metrica and SkillCorner "
                             "are always scored; PFF is scored only by the variants trained on "
                             "it)")
    parser.add_argument("--selection-score", choices=sorted(SELECTION_SCORES),
                        default=DEFAULT_SELECTION_SCORE,
                        help="Metrica score that is the primary criterion of the variant "
                             "selection; both are always reported "
                             f"(default {DEFAULT_SELECTION_SCORE})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="degradation seed")
    parser.add_argument("--out", type=cli_path,
                        help=f"model path (default {DEFAULT_MODEL_PATH}; required with "
                             f"--matches, --limit, --frame-step or a --variants subset)")
    parser.add_argument("--matches",
                        help="comma-separated match keys (source/name) or names (default: all)")
    parser.add_argument("--dead-features", choices=DEAD_FEATURE_SETS,
                        default=DeadBallSettings.features,
                        help="features of the dead-ball model: all, or players only (no ball "
                             "or object-count features; docs/DEADBALL.md)")
    parser.add_argument("--adaptation", action="store_true",
                        help="adapt the ball-observation evidence to each predicted match by EM, "
                             "run until converged (needs --dead-features players)")
    parser.add_argument("--team-features", choices=TEAM_FEATURE_SETS,
                        default=DEFAULT_TEAM_FEATURES,
                        help="features of the team model: core (as before), short (plus rolling "
                             "means at the lower quartile and median of the training H/A run "
                             "lengths) or touch (plus ball touch events; docs/PARAMETERS.md); the "
                             "dead-ball model always uses the core features")
    parser.add_argument("--no-switch-prior", action="store_true",
                        help="ablation: no touch-based switch prior in the decoder (switch weight "
                             "0 only, the decoder as before). By default the switch weight is "
                             "chosen from 0, 0.5, 1, 2 and the grid extends itself along the "
                             "1-2-5 series (5, 10, 20, ...) while a selection, final or nested, "
                             "is its largest weight; docs/PARAMETERS.md")
    parser.add_argument("--spell-weights", action="store_true",
                        help="weight each team-model training frame by 1 / (length of its "
                             "labelled run), so every spell of a match weighs the same, as in "
                             "spellF1 (default: every frame of a match weighs the same); "
                             "docs/PARAMETERS.md")
    parser.add_argument("--every-peak", action="store_true",
                        help="team touch features from every impulse peak instead of the "
                             "peaks above the Youden threshold; docs/PARAMETERS.md")
    parser.add_argument("--stopping", choices=STOPPING_RULES, default=STOPPING_RULES[0],
                        help="iterations of the boosted models: default (library default), "
                             "team (team model by early stopping on whole held-out training "
                             "matches, leave one group out, then a refit at the mean stop) or "
                             "both (team and dead-ball models); docs/PARAMETERS.md")
    parser.add_argument("--line-features", nargs="*", choices=LINE_GROUPS, default=[],
                        help="extra feature groups about the pitch lines (default none: the "
                             "features as before): lines (furthest player beyond a touchline or "
                             "goal line and the speed of the player nearest a line), keeper "
                             "(depth, speed and distance to the last seen ball of each team's "
                             "deepest player) and ball (time since and until the ball is beyond a "
                             "line, ball distance to the lines at the ends of a ball gap); "
                             "docs/DEADBALL.md")
    parser.add_argument("--dead-context", action="store_true",
                        help="second dead-ball stage: the dead-ball model also sees centred, past "
                             "and future means of the first stage's dead-ball logit (out-of-fold "
                             "by leaving out one validation group at a time in training) at the "
                             "rolling scales; "
                             "docs/DEADBALL.md")
    parser.add_argument("--limit", type=int, help="matches per source; only for smoke runs")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="train on every n-th frame; only for smoke runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    variants = parse_variants(args.variants)
    names = set(args.matches.split(",")) if args.matches is not None else None
    out = output_path(args.out, names is not None or args.limit is not None
                      or args.frame_step != 1 or partial_variants(variants))
    if args.frame_step != 1 or args.limit is not None:
        print(f"SMOKE RUN: every {args.frame_step}th frame, {args.limit} matches per source")
    dead_ball = dead_ball_settings(args.dead_features, args.adaptation)
    switch_prior = not args.no_switch_prior
    fitting = ClassifierSettings(spell_weights=args.spell_weights, every_peak=args.every_peak,
                                 stopping=args.stopping, dead_context=args.dead_context)
    lines = tuple(group for group in LINE_GROUPS if group in args.line_features)
    grid = settings_grid(switch_prior)
    files = selected_files(names, args.limit, scored_sources(variants, tuple(args.score_outside)))
    check_variants(variants, [item.source for item in files])
    spill = tempfile.TemporaryDirectory(prefix=SPILL_PREFIX)
    loaded = load_matches(args.seed, files, Path(spill.name),
                          args.team_features == "touch" or switch_prior, lines)
    context = Context(seed=args.seed, frame_step=args.frame_step, grid=grid, loaded=loaded,
                      out=out, fingerprint=code_fingerprint(*FIT_MODULES), dead_ball=dead_ball,
                      outside_sources=(*ALWAYS_SCORED, *args.score_outside),
                      team_features=args.team_features, switch_prior=switch_prior,
                      fitting=fitting, lines=lines)
    print(f"loaded {len(loaded)} matches (degraded, seed {args.seed}) in "
          f"{time.perf_counter() - start:.0f}s; variants {variants}; initial grid {len(grid)} "
          f"settings",
          flush=True)
    results = [run_variant(context, variant) for variant in variants]
    held_out = {result.name: result.held_out() for result in results}
    summaries = {name: variant_summary(rows) for name, rows in held_out.items()}
    for result in results:
        print_variant(result, held_out[result.name])
    selected_name = select_variant(summaries, SELECTION_SCORES[args.selection_score])
    print_table(summaries, selected_name, args.selection_score)
    chosen = next(result for result in results if result.name == selected_name)
    config = {"format": MODEL_FORMAT, "seed": args.seed, "model_seed": MODEL_SEED,
              "frame_step": args.frame_step, "bootstrap_draws": BOOTSTRAP_DRAWS,
              "selected_variant_grid": [asdict(settings) for settings in chosen.grid],
              "variants": variants,
              "selection_score": args.selection_score, "dead_ball": asdict(dead_ball),
              "team_features": args.team_features, "switch_prior": switch_prior,
              **fitting_record(fitting), **line_record(lines), "variant": selected_name,
              "matches": chosen.members,
              "cleaning": chosen.model.params.to_dict()}
    run_id = training_run_id(config, model_digest(chosen.model))
    details = {"variant": selected_name, "variant_sources": list(VARIANTS[selected_name]),
               "seed": args.seed, "frame_step": args.frame_step, **fitting_record(fitting),
               "grid": grid_axes(chosen.final_grid), "held_out_summary": summaries[selected_name]}
    digest = save_model(out, chosen.model, run_id, details)
    save_folds(results, out, run_id)
    report = {**config, "run_id": run_id, "model_sha256": digest,
              "selection_rule": {"primary": f"{METRICA} {args.selection_score} spell_f1_mean",
                                 "tie": "interval of the mean (mean of the per-match bootstrap "
                                        "5-95 lower and upper bounds) reaches the best "
                                        "variant's lower bound",
                                 "order": [SELECTION_SCORES[args.selection_score], TIE_SOURCE,
                                           LAST_SOURCE]},
              "variant_results": {result.name: {
                  "sources": list(VARIANTS[result.name]), "matches": result.members,
                  "grid": [asdict(settings) for settings in result.grid],
                  "final_selection_grid": grid_axes(result.final_grid),
                  "selected": asdict(result.grid[result.selected]),
                  "scales": list(result.model.layout.scales),
                  "short_scales": list(result.model.layout.short_scales),
                  "touch_threshold": result.model.layout.touch_threshold,
                  "touch_evidence_peaks": result.model.touch_peaks,
                  "switch_prior": {fold.model_key: switch_summary(fold.model)
                                   for fold in result.folds},
                  "stopping": {fold.model_key: list(fold.model.classifier.stopping)
                               for fold in result.folds},
                  "summary": summaries[result.name],
                  "held_out": held_out[result.name]} for result in results}}
    (out.parent / REPORT_NAME).write_text(json.dumps(report, indent=2) + "\n")
    print(f"final model ({selected_name}, {len(chosen.members)} matches): {out} sha256 {digest} "
          f"run {run_id}")
    spill.cleanup()
    print(f"runtime {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
