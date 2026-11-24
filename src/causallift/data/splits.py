import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from causallift.data.contract import DatasetSpec

PARTITIONS = ("train", "validation", "test")
RATIOS = np.array([0.6, 0.2, 0.2])


def allocate(total: int, proportions: np.ndarray) -> np.ndarray:
    ideal = total * proportions
    counts = np.floor(ideal).astype(int)
    remainder = total - counts.sum()
    counts[np.argsort(-(ideal - counts), kind="stable")[:remainder]] += 1
    return counts


def split_data(frame: pd.DataFrame, spec: DatasetSpec, seed: int = 42) -> pd.Series:
    spec.validate(frame)
    treatment = frame[spec.treatment_column].to_numpy(dtype=int)
    outcome = frame[spec.outcome_column].to_numpy(dtype=int)
    strata = 2 * treatment + outcome
    groups = frame[spec.unit_id] if spec.unit_id else None
    rng = np.random.default_rng(seed)

    def independent(labels):
        result = np.empty(len(frame), dtype=np.int8)
        for label in np.unique(labels):
            positions = rng.permutation(np.flatnonzero(labels == label))
            counts = allocate(len(positions), RATIOS)
            if len(positions) >= 3 and counts.min() == 0:
                counts[counts.argmin()] += 1
                counts[counts.argmax()] -= 1
            result[positions] = np.repeat(np.arange(3), counts)
        return result

    def covered(result):
        return np.unique(2 * result + treatment).size == 6

    if groups is not None and groups.duplicated().any():
        if groups.nunique() < 5:
            raise ValueError("Group-safe 60/20/20 requires at least five distinct groups")
        best = None
        best_score = np.inf
        labels = strata if np.bincount(strata)[np.bincount(strata) > 0].min() >= 5 else treatment
        for attempt_seed in rng.integers(0, 2**31 - 1, size=8):
            cv = StratifiedGroupKFold(5, shuffle=True, random_state=int(attempt_seed))
            folds = np.empty(len(frame), dtype=np.int8)
            for fold, (_, positions) in enumerate(cv.split(treatment, labels, groups)):
                folds[positions] = fold
            for validation in range(5):
                for test in range(validation + 1, 5):
                    result = (folds == validation).astype(np.int8) + 2 * (folds == test)
                    if not covered(result):
                        continue
                    joint = np.bincount(4 * result + strata, minlength=12).reshape(3, 4)
                    expected = np.outer(RATIOS, np.bincount(strata, minlength=4))
                    score = np.sum((joint - expected) ** 2 / np.maximum(expected, 1))
                    if score < best_score:
                        best, best_score = result, score
        if best is None:
            raise ValueError("Could not construct group-safe splits with both arms in every partition")
        result = best
    else:
        result = independent(strata)
        if not covered(result):
            warnings.warn("Joint allocation lacked arm coverage; using treatment strata.", stacklevel=2)
            result = independent(treatment)
    if not covered(result):
        raise ValueError("Every partition must contain both treatment arms")
    if np.any(np.bincount(4 * result + strata, minlength=12).reshape(3, 4) == 0):
        warnings.warn("At least one partition lacks a treatment/outcome combination.", stacklevel=2)

    return pd.Series(np.asarray(PARTITIONS)[result], index=frame.index, name="split")
