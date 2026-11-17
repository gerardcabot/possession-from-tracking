import argparse
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from possession.external.match import (
    ExternalMatch,
    dead_labels_trusted,
    fair_codes,
    fair_truth,
    filled_codes,
    label_codes,
    padded,
)
from possession.external.predict import METHODS
from possession.external.prepare import DEFAULT_SEED, VARIANTS, variant_matches
from possession.external.sources import SOURCES
from possession.paths import DEFAULT_MODEL_PATH, cli_path
from possession.vendor import possession_metrics as pm

LEARNED = "learned"
METHOD_CHOICES = sorted([*METHODS, LEARNED])
Predictor = Callable[[ExternalMatch], np.ndarray]


SCORE_KEYS = ("spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells")
BOOTSTRAP_KEYS = ("spell_bootstrap_low", "spell_bootstrap_high")


def score_labels(match: ExternalMatch, labels: np.ndarray,
                 bootstrap_n: int = 0) -> dict[str, float]:
    return score_codes(match, label_codes(labels), bootstrap_n=bootstrap_n)


def score_fair(match: ExternalMatch, labels: np.ndarray, bootstrap_n: int = 0) -> dict[str, float]:
    return score_codes(match, fair_codes(match, labels), fair_truth(match), bootstrap_n)


def score_codes(match: ExternalMatch, predicted: np.ndarray, truth: np.ndarray | None = None,
                bootstrap_n: int = 0) -> dict[str, float]:
    reference = padded(filled_codes(match) if truth is None else truth, match, pm.X)
    result = pm.evaluate(reference, padded(predicted, match, pm.X), taus=pm.TAUS,
                         bootstrap_n=bootstrap_n)
    keys = SCORE_KEYS + (BOOTSTRAP_KEYS if bootstrap_n else ())
    return {key: float("nan") if result[key] is None else float(result[key]) for key in keys}


def score_match(match: ExternalMatch, method: str) -> dict[str, float]:
    return score_labels(match, METHODS[method](match))


def learned_predictor(model_path: Path) -> Predictor:
    from possession.learned.external import model_for, predict_match
    from possession.learned.model import load_model

    final = load_model(model_path)
    return lambda match: predict_match(model_for(match, final, model_path), match)


def predictor(method: str, model_path: Path) -> Predictor:
    if method == LEARNED:
        return learned_predictor(model_path)
    return METHODS[method]


def format_row(label: str, scores: dict[str, float]) -> str:
    return (f"{label:<24}{scores['spell_f1_mean']:>10.4f}{scores['macro_f1']:>10.4f}"
            f"{int(scores['n_pred_spells']):>12}{int(scores['n_gt_spells']):>12}")


def mean_scores(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a method on degraded external matches against their labels")
    parser.add_argument("--method", choices=METHOD_CHOICES, required=True)
    parser.add_argument("--source", choices=sorted(SOURCES))
    parser.add_argument("--variant", choices=VARIANTS, default="degraded")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dead-as-fill", action=argparse.BooleanOptionalAction, default=True,
                        help="for sources without trusted D labels (SkillCorner, Metrica), also "
                             "report scores with D treated as continuation of the last team in "
                             "both the prediction and the reference")
    parser.add_argument("--model", type=cli_path, default=DEFAULT_MODEL_PATH,
                        help="trained model for --method learned")
    args = parser.parse_args()
    start = time.perf_counter()
    predict = predictor(args.method, args.model)
    by_source: dict[str, list[dict[str, float]]] = {}
    fair_by_source: dict[str, list[dict[str, float]]] = {}
    print(f"method {args.method}  variant {args.variant}")
    print(f"{'match':<24}{'spellF1':>10}{'macroF1':>10}{'spells pred':>12}{'spells true':>12}")
    for match in variant_matches(args.source, args.seed, args.variant):
        labels = predict(match)
        scores = score_labels(match, labels)
        by_source.setdefault(match.source, []).append(scores)
        print(format_row(f"{match.source}/{match.name}", scores), flush=True)
        if args.dead_as_fill and not dead_labels_trusted(match):
            fair = score_fair(match, labels)
            fair_by_source.setdefault(match.source, []).append(fair)
            print(format_row(f"{match.source}/{match.name} fair", fair), flush=True)
    for source, rows in by_source.items():
        print(format_row(f"mean {source} ({len(rows)})", mean_scores(rows)))
        if source in fair_by_source:
            fair_rows = fair_by_source[source]
            print(format_row(f"mean {source} fair ({len(fair_rows)})", mean_scores(fair_rows)))
    print(f"runtime {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
