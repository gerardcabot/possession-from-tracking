import importlib.util
import sys
from pathlib import Path
from types import ModuleType

VENDOR_DIR = Path(__file__).resolve().parent


def load_vendored(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, VENDOR_DIR / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load vendored module {name} from {VENDOR_DIR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


games = load_vendored("games")
possession_metrics = load_vendored("possession_metrics")
run_evaluation = load_vendored("run_evaluation")

__all__ = ["VENDOR_DIR", "games", "possession_metrics", "run_evaluation"]
