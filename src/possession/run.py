import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import numpy as np

from possession import baseline, rules
from possession.data import (
    Tracking,
    fps,
    graded_blocks,
    graded_mask,
    kickoff_frames,
    load_tracking,
    pitch_size,
    prekickoff_mask,
)
from possession.intervals import DEAD, LABELS
from possession.params import rules_params
from possession.paths import DEFAULT_MODEL_PATH, PROJECT_ROOT, cli_path
from possession.submit import finalize_labels, sha256_of, write_submission


@dataclass(frozen=True)
class RunConfig:
    method: str = "baseline"
    window: int = 50
    use_cache: bool = True
    finalize: bool = False
    force_prekickoff_dead: bool = True
    params: dict[str, float] = field(default_factory=dict)
    model: str = str(DEFAULT_MODEL_PATH.relative_to(PROJECT_ROOT))


def predict_baseline(tracking: Tracking, config: RunConfig) -> np.ndarray:
    return baseline.predict(tracking, config.window)


def predict_rules(tracking: Tracking, config: RunConfig) -> np.ndarray:
    return rules.predict(tracking, rules_params(config.params))


def model_path(config: RunConfig) -> Path:
    path = Path(config.model)
    return path if path.is_absolute() else PROJECT_ROOT / path


def predict_learned(tracking: Tracking, config: RunConfig) -> np.ndarray:
    from possession.learned.model import load_model, predict_codes

    model = load_model(model_path(config))
    blocks = graded_blocks()
    labels = LABELS[predict_codes(model, tracking, pitch_size(), fps(), blocks, "our match")]
    if config.force_prekickoff_dead:
        mask = prekickoff_mask(len(labels), blocks, kickoff_frames())
        labels = np.where(mask, LABELS[DEAD], labels)
    return labels


METHODS: dict[str, Callable[[Tracking, RunConfig], np.ndarray]] = {
    "baseline": predict_baseline,
    "rules": predict_rules,
    "learned": predict_learned,
}
CONFIG_KEYS = frozenset(field.name for field in fields(RunConfig))


def is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_config(path: Path, values: dict) -> None:
    unknown = sorted(set(values) - CONFIG_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown config keys {unknown}")
    method = values.get("method", RunConfig.method)
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(f"{path}: method must be one of {sorted(METHODS)}, got {method!r}")
    window = values.get("window", RunConfig.window)
    if not is_int(window) or window <= 0:
        raise ValueError(f"{path}: window must be an integer > 0, got {window!r}")
    for key in ("use_cache", "finalize", "force_prekickoff_dead"):
        value = values.get(key, getattr(RunConfig, key))
        if not isinstance(value, bool):
            raise ValueError(f"{path}: {key} must be true or false, got {value!r}")
    model = values.get("model", RunConfig.model)
    if not isinstance(model, str) or not model:
        raise ValueError(f"{path}: model must be a non-empty path string, got {model!r}")
    params = values.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"{path}: params must be a JSON object, got {params!r}")
    try:
        rules_params(params)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def load_config(path: Path | None) -> RunConfig:
    if path is None:
        return RunConfig()
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError(f"{path}: config must be a JSON object")
    validate_config(path, values)
    return RunConfig(**values)


def resolve_config(args: argparse.Namespace) -> RunConfig:
    config = load_config(args.config)
    overrides = {key: getattr(args, key) for key in ("method", "window")
                 if getattr(args, key) is not None}
    if args.model is not None:
        overrides["model"] = str(args.model)
    if args.no_cache:
        overrides["use_cache"] = False
    if args.finalize:
        overrides["finalize"] = True
    config = replace(config, **overrides)
    if config.method not in METHODS:
        raise ValueError(f"unknown method {config.method!r}; have {sorted(METHODS)}")
    return config


def produce(config: RunConfig, out: Path) -> Path:
    tracking = load_tracking(use_cache=config.use_cache)
    labels = METHODS[config.method](tracking, config)
    if config.finalize:
        labels = finalize_labels(labels, graded_mask(len(labels)))
    return write_submission(out, labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Produce a possession submission CSV")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--method", choices=sorted(METHODS))
    parser.add_argument("--window", type=int)
    parser.add_argument("--model", type=cli_path,
                        help="trained model for the learned method, relative to the current "
                             "directory (a config file's model is relative to the project root)")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--finalize", action="store_true",
                        help="write X outside graded blocks and validate graded labels")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    start = time.perf_counter()
    path = produce(config, args.out)
    print(f"config {json.dumps(asdict(config), sort_keys=True)}")
    print(f"wrote {path} in {time.perf_counter() - start:.1f}s")
    print(f"sha256 {sha256_of(path)}")


if __name__ == "__main__":
    main()
