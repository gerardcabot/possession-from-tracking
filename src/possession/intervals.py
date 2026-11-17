import numpy as np

DEAD = 2
LABELS = np.array(["H", "A", "D"])


def value_runs(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(values) == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, values[:0]
    starts = np.flatnonzero(np.concatenate(([True], values[1:] != values[:-1])))
    return starts, np.append(starts[1:], len(values)), values[starts]


def true_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    change = np.flatnonzero(padded[1:] != padded[:-1])
    return change[::2], change[1::2]


def drop_short_runs(mask: np.ndarray, min_length: int) -> np.ndarray:
    result = mask.astype(bool).copy()
    starts, ends = true_runs(result)
    for start, end in zip(starts.tolist(), ends.tolist()):
        if end - start < min_length:
            result[start:end] = False
    return result


def fill_short_gaps(mask: np.ndarray, min_length: int) -> np.ndarray:
    result = mask.astype(bool).copy()
    starts, ends = true_runs(~result)
    for start, end in zip(starts.tolist(), ends.tolist()):
        if start > 0 and end < len(result) and end - start < min_length:
            result[start:end] = True
    return result


def run_has_length(mask: np.ndarray, length: int) -> np.ndarray:
    counts = np.concatenate(([0], np.cumsum(mask.astype(np.int64))))
    index = np.arange(len(mask))
    low = index - length
    full = low >= 0
    result = np.zeros(len(mask), dtype=bool)
    result[full] = (counts[index[full]] - counts[low[full]]) == length
    return result
