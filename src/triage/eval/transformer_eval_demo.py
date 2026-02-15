"""
transformer_eval_demo.py

Generate a demo probability dump (NPZ) that *matches your test split size* so
src.triage.eval.transformer_eval can evaluate it without length mismatches.

This is model-agnostic: it just writes "probs" (n_samples x n_classes).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# Reuse the same split helpers if available (keeps behavior consistent)
from .transformer_eval import compute_time_split_indices, compute_stratified_split_indices


@dataclass
class SplitCfg:
    strategy: str
    test_size: float
    seed: int
    time_col: str
    gap_days: int


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def get_target_col(target: str) -> str:
    # Convention in this repo: targets are "category" and "priority"
    return target


def build_label_list(y: pd.Series) -> List[str]:
    # Keep same label ordering as transformer_eval (sorted unique)
    return sorted(pd.Series(y).astype(str).unique().tolist())


def sample_probs_like(y_true_idx: np.ndarray, n_classes: int, approx_acc: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(y_true_idx)
    probs = rng.random((n, n_classes))
    probs = probs / probs.sum(axis=1, keepdims=True)

    # Make it *roughly* achieve approx_acc by boosting the true class for a fraction of samples
    mask = rng.random(n) < approx_acc
    for i in range(n):
        if mask[i]:
            t = int(y_true_idx[i])
            # boost true class, then renormalize
            probs[i] = probs[i] * 0.2
            probs[i, t] = 0.8
            probs[i] = probs[i] / probs[i].sum()
        else:
            # force a wrong top-1
            t = int(y_true_idx[i])
            wrong = int(rng.integers(0, n_classes - 1))
            if wrong >= t:
                wrong += 1
            probs[i] = probs[i] * 0.2
            probs[i, wrong] = 0.8
            probs[i] = probs[i] / probs[i].sum()

    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument_toggle = ap.add_argument  # small convenience when copy/pasting patterns

    ap.add_argument("--target", required=True, choices=["priority", "category"])
    ap.add_argument("--data", required=False, default=None, help="tickets CSV; if given, demo matches split size")
    ap.add_argument("--out", default="artifacts/transformer_probs_demo.npz")
    ap.add_argument("--approx-acc", type=float, default=0.82)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--split", choices=["time", "stratified"], default="time")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--time-col", default="timestamp")
    ap.add_argument("--gap-days", type=int, default=1)

    ap.add_argument("--n", type=int, default=200, help="Only used if --data is not provided")
    ap.add_argument("--n-classes", type=int, default=None, help="Only used if --data is not provided")

    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.data:
        df = load_csv(args.data)
        target_col = get_target_col(args.target)

        if args.split == "time":
            train_idx, test_idx, _meta = compute_time_split_indices(
                df=df,
                time_col=args.time_col,
                test_size=args.test_size,
                seed=args.seed,
                gap_days=args.gap_days,
            )
        else:
            train_idx, test_idx, _meta = compute_stratified_split_indices(
                df=df,
                y_col=target_col,
                test_size=args.test_size,
                seed=args.seed,
            )

        y_test = df.iloc[test_idx][target_col].astype(str).reset_index(drop=True)
        label_names = build_label_list(y_test)
        label_to_idx = {lab: i for i, lab in enumerate(label_names)}
        y_true_idx = y_test.map(label_to_idx).to_numpy()

        probs = sample_probs_like(
            y_true_idx=y_true_idx,
            n_classes=len(label_names),
            approx_acc=float(args.approx_acc),
            seed=int(args.seed),
        )

        np.savez_compressed(out_path, probs=probs)

        print(f"[OK] Wrote demo NPZ (matches split): {out_path}")
        print(
            json.dumps(
                {
                    "target": args.target,
                    "n": int(len(y_test)),
                    "n_classes": int(len(label_names)),
                    "approx_acc": float(args.approx_acc),
                    "labels": label_names,
                    "split": args.split,
                    "test_size": float(args.test_size),
                },
                indent=2,
            )
        )
        return

    # no data: generic demo
    if args.n_classes is None:
        args.n_classes = 4 if args.target == "priority" else 6

    rng = np.random.default_rng(args.seed)
    probs = rng.random((int(args.n), int(args.n_classes)))
    probs = probs / probs.sum(axis=1, keepdims=True)

    np.savez_compressed(out_path, probs=probs)

    print(f"[OK] Wrote generic demo NPZ: {out_path}")
    print(json.dumps({"target": args.target, "n": int(args.n), "n_classes": int(args.n_classes)}, indent=2))


if __name__ == "__main__":
    main()
