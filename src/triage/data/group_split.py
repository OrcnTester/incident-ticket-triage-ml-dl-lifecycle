from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GroupSplitResult:
    train_idx: List[int]
    val_idx: List[int]
    test_idx: List[int]
    meta: Dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def group_holdout_split(
    df: pd.DataFrame,
    *,
    group_field: str,
    test_size: float = 0.2,
    seed: int = 42,
    val_size: float = 0.0,
) -> GroupSplitResult:
    """
    Leakage-safe split: split by GROUPS (e.g., region_id/tile_id/system/service).
    Ensures group values do not overlap across train/val/test.

    Strategy (deterministic):
      - collect unique groups, sort for stability
      - shuffle with seed
      - allocate groups to test (and val) by fraction
      - return row indices for each split
    """
    if group_field not in df.columns:
        raise ValueError(f"group_field '{group_field}' not found in df columns: {list(df.columns)}")

    groups = df[group_field].astype(str).fillna("UNKNOWN")
    unique = sorted(groups.unique().tolist())

    rng = np.random.RandomState(seed)
    perm = unique.copy()
    rng.shuffle(perm)

    n = len(perm)
    n_test = max(1, int(np.ceil(n * float(test_size))))
    n_val = int(np.ceil(n * float(val_size))) if val_size and val_size > 0 else 0

    test_groups = set(perm[:n_test])
    val_groups = set(perm[n_test : n_test + n_val]) if n_val else set()
    train_groups = set(perm[n_test + n_val :])

    # if too few groups, fall back: keep at least 1 train group
    if not train_groups:
        # move one group from test to train
        g = next(iter(test_groups))
        test_groups.remove(g)
        train_groups.add(g)

    idx = np.arange(len(df))
    train_idx = idx[groups.isin(train_groups)].tolist()
    val_idx = idx[groups.isin(val_groups)].tolist()
    test_idx = idx[groups.isin(test_groups)].tolist()

    meta: Dict[str, Any] = {
        "strategy": "group_holdout",
        "group_field": group_field,
        "seed": int(seed),
        "test_size": float(test_size),
        "val_size": float(val_size),
        "n_rows": int(len(df)),
        "n_groups": int(len(unique)),
        "train_groups": sorted(train_groups),
        "val_groups": sorted(val_groups),
        "test_groups": sorted(test_groups),
        "generated_at": utc_now_iso(),
    }

    return GroupSplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, meta=meta)


def explicit_group_split(
    df: pd.DataFrame,
    *,
    group_field: str,
    train_groups: List[str],
    val_groups: List[str],
    test_groups: List[str],
) -> GroupSplitResult:
    """
    Most audit-friendly split: you explicitly list group memberships.
    Guarantees disjointness if the lists are disjoint.
    """
    if group_field not in df.columns:
        raise ValueError(f"group_field '{group_field}' not found in df columns: {list(df.columns)}")

    g = df[group_field].astype(str).fillna("UNKNOWN")
    train_set = set([str(x) for x in train_groups])
    val_set = set([str(x) for x in val_groups])
    test_set = set([str(x) for x in test_groups])

    # disjointness check
    if (train_set & test_set) or (train_set & val_set) or (val_set & test_set):
        raise ValueError("explicit_group_split requires train/val/test groups to be disjoint")

    idx = np.arange(len(df))
    train_idx = idx[g.isin(train_set)].tolist()
    val_idx = idx[g.isin(val_set)].tolist()
    test_idx = idx[g.isin(test_set)].tolist()

    meta: Dict[str, Any] = {
        "strategy": "explicit_groups",
        "group_field": group_field,
        "train_groups": sorted(train_set),
        "val_groups": sorted(val_set),
        "test_groups": sorted(test_set),
        "n_rows": int(len(df)),
        "generated_at": utc_now_iso(),
    }

    return GroupSplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, meta=meta)
