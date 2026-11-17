import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO_DIR = PROJECT_ROOT.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "learned" / "model.pkl"
PUBLIC_GT_PATTERN = "*_possession_gt_public.csv"
PRIVATE_GT_PATTERN = "*_possession_gt.csv"


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name) or default)


def cli_path(value: str) -> Path:
    if not value:
        raise argparse.ArgumentTypeError("path must not be empty")
    return Path(value).absolute()


def repo_dir() -> Path:
    return env_path("MTAH_REPO_DIR", DEFAULT_REPO_DIR)


def data_dir() -> Path:
    return env_path("MTAH_DATA_DIR", DEFAULT_REPO_DIR / "data")


def cache_dir() -> Path:
    return env_path("MTAH_CACHE_DIR", DEFAULT_CACHE_DIR)


def external_dir() -> Path:
    return env_path("MTAH_EXTERNAL_DIR", data_dir() / "external")


def gt_dir() -> Path:
    return repo_dir() / "evaluation" / "gt"


def single_gt(pattern: str) -> Path:
    candidates = sorted(gt_dir().glob(pattern))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one {pattern} in {gt_dir()}, found {len(candidates)}; "
            "set MTAH_REPO_DIR to the official challenge repository")
    return candidates[0]


def public_gt_path() -> Path:
    return single_gt(PUBLIC_GT_PATTERN)


def private_gt_path() -> Path:
    return single_gt(PRIVATE_GT_PATTERN)


def resolve_gt(name: str) -> Path:
    if name == "public":
        return public_gt_path()
    if name == "private":
        return private_gt_path()
    return Path(name)


def official_result_path(name: str) -> Path:
    return repo_dir() / "evaluation" / "results" / name
