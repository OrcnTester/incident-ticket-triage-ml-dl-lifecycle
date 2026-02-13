from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitResult:
    train_idx: List[int]
    test_idx: List[int]


def stratified_split(
    y: Sequence,
    *,
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitResult:
    """
    Returns indices for train/test split stratified by y.
    """
    y = np.asarray(list(y))
    idx = np.arange(len(y))

    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    return SplitResult(
        train_idx=sorted(train_idx.tolist()),
        test_idx=sorted(test_idx.tolist()),
    )
