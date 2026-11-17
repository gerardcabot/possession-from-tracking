import os
import subprocess
import sys
from pathlib import Path

import pytest

from possession.paths import PROJECT_ROOT, data_dir, official_result_path

SRC_DIR = PROJECT_ROOT / "src"
OFFICIAL_BASELINE = official_result_path("baseline_smooth50.csv")


def has_match_data() -> bool:
    tactical = data_dir() / "tactical"
    return (tactical / "BAR-ATH_tactical_atd.txt").is_file() and \
        (tactical / "BAR-ATH_tactical_atd.xml").is_file()


requires_data = pytest.mark.skipif(not has_match_data(), reason="match data not available")
requires_official = pytest.mark.skipif(
    not OFFICIAL_BASELINE.is_file(), reason="official challenge repository not available")


def run_module(module: str, *args: str, cache_dir: Path) -> subprocess.CompletedProcess:
    python_path = os.pathsep.join(filter(None, [str(SRC_DIR), os.environ.get("PYTHONPATH")]))
    env = {**os.environ, "MTAH_CACHE_DIR": str(cache_dir), "PYTHONPATH": python_path}
    return subprocess.run([sys.executable, "-m", module, *args], cwd=PROJECT_ROOT, env=env,
                          capture_output=True, text=True, check=True)
