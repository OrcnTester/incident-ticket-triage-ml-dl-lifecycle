from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
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


def time_aware_split(
    timestamps: Sequence,
    *,
    test_size: float = 0.2,
    gap_days: int = 0,
) -> Tuple[SplitResult, Dict[str, Any]]:
    """
    Time-aware split to reduce leakage:
      - sort by timestamp
      - take the last N% as test
      - optional gap (in days) between train and test

    Returns:
      (SplitResult, meta)
    where meta includes cutoffs to prove no future data leaked into training.

    NOTE: indices are positions (0..n-1) into the provided timestamps sequence.
    """
    ts = pd.Series(pd.to_datetime(list(timestamps), errors="coerce", utc=True))  # ensure Series for .iloc
    n_bad = int(ts.isna().sum())
    if n_bad:
        raise ValueError(
            f"time_aware_split: {n_bad} invalid timestamps (NaT). "
            "Drop/fix them before splitting."
        )

    n = len(ts)
    if n == 0:
        raise ValueError("time_aware_split: empty timestamps")

    idx = np.arange(n)
    order = np.argsort(ts.values)

    n_test = int(np.ceil(n * float(test_size)))
    n_test = max(1, min(n_test, n - 1))  # keep at least 1 train row

    test_idx = order[-n_test:].tolist()
    train_idx = order[:-n_test].tolist()

    cutoff_test_min = pd.Timestamp(ts.iloc[test_idx].min()).to_pydatetime()
    gap_cutoff = cutoff_test_min
    if gap_days and gap_days > 0:
        gap_cutoff = (pd.Timestamp(cutoff_test_min) - pd.Timedelta(days=int(gap_days))).to_pydatetime()
        train_idx = [i for i in train_idx if ts.iloc[i].to_pydatetime() <= gap_cutoff]

    # meta evidence
    train_max = pd.Timestamp(ts.iloc[train_idx].max()).to_pydatetime() if train_idx else None
    test_min = pd.Timestamp(ts.iloc[test_idx].min()).to_pydatetime()
    test_max = pd.Timestamp(ts.iloc[test_idx].max()).to_pydatetime()

    meta: Dict[str, Any] = {
        "strategy": "time",
        "test_size": float(test_size),
        "gap_days": int(gap_days),
        "cutoff_timestamp": test_min.isoformat(),
        "train_max_time": train_max.isoformat() if train_max else None,
        "test_min_time": test_min.isoformat(),
        "test_max_time": test_max.isoformat(),
        "n_total": int(n),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }

    return (
        SplitResult(
            train_idx=sorted(train_idx),
            test_idx=sorted(test_idx),
        ),
        meta,
    )
