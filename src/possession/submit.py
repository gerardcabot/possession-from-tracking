import hashlib
from pathlib import Path

import numpy as np

ABSTAIN = "X"
GRADED_LABELS = frozenset({"H", "A", "D"})


def write_submission(path: Path, labels: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{frame},{label}\n" for frame, label in enumerate(labels))
    return path


def finalize_labels(labels: np.ndarray, graded: np.ndarray) -> np.ndarray:
    if labels.shape != graded.shape:
        raise ValueError(f"labels shape {labels.shape} != graded mask shape {graded.shape}")
    invalid = sorted(set(np.unique(labels[graded]).tolist()) - GRADED_LABELS)
    if invalid:
        raise ValueError(f"graded frames must be H, A or D; found {invalid}")
    return np.where(graded, labels, ABSTAIN).astype("<U1")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
