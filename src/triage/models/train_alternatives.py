from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier


# We prefer using the repo's loader if it exists (consistent text column handling).
# Fallback to a simple CSV loader if needed.
try:
    from src.triage.data.load import load_tickets  # type: ignore
except Exception:  # pragma: no cover
    load_tickets = None  # type: ignore


@dataclass(frozen=True)
class SplitResult:
    train_idx: List[int]
    test_idx: List[int]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def stratified_split(y: List[str], *, test_size: float, seed: int) -> SplitResult:
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=np.asarray(y),
    )
    return SplitResult(train_idx=sorted(train_idx.tolist()), test_idx=sorted(test_idx.tolist()))


def time_aware_split(
    ts: pd.Series,
    *,
    test_size: float,
    gap_days: int = 0,
) -> Tuple[SplitResult, Dict[str, Any]]:
    """
    Sort by timestamp. Use the last fraction as test.
    Optionally enforce a gap (in days) between train and test to reduce leakage.

    Returns (split, meta).
    """
    ts = pd.to_datetime(ts, errors="coerce")
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise SystemExit(f"time split requires valid timestamps. Found {bad} NaT values in time column.")
    n = len(ts)
    if n < 10:
        raise SystemExit("Dataset too small for time split demo (need at least 10 rows).")

    order = np.argsort(ts.to_numpy())
    n_test = max(1, int(np.ceil(n * float(test_size))))
    test_idx = order[-n_test:]
    train_idx = order[:-n_test]

    test_min = pd.Timestamp(ts.iloc[test_idx].min())
    cutoff = test_min - pd.Timedelta(days=int(gap_days))
    train_idx_gap = [int(i) for i in train_idx if ts.iloc[int(i)] < cutoff]

    if len(train_idx_gap) < 5:
        raise SystemExit(
            f"After applying gap_days={gap_days}, train set too small ({len(train_idx_gap)}). "
            "Reduce gap_days or test_size."
        )

    meta = {
        "strategy": "time",
        "n_total": int(n),
        "n_test": int(len(test_idx)),
        "n_train_before_gap": int(len(train_idx)),
        "n_train_after_gap": int(len(train_idx_gap)),
        "test_min_timestamp": test_min.isoformat(),
        "gap_days": int(gap_days),
        "train_cutoff_timestamp": pd.Timestamp(cutoff).isoformat(),
    }
    return SplitResult(train_idx=sorted(train_idx_gap), test_idx=sorted([int(i) for i in test_idx])), meta


def build_pipeline(
    model: str,
    *,
    svd_dim: int,
    rf_estimators: int,
    rf_max_depth: Optional[int],
    seed: int,
) -> Pipeline:
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=200_000,
    )

    if model == "nb":
        clf = MultinomialNB(alpha=1.0)
        return Pipeline([("tfidf", vec), ("clf", clf)])

    if model == "svm":
        clf = LinearSVC(class_weight="balanced")
        return Pipeline([("tfidf", vec), ("clf", clf)])

    if model == "rf":
        # Trees don't love huge sparse TF‑IDF → reduce dimensionality first.
        svd = TruncatedSVD(n_components=int(svd_dim), random_state=int(seed))
        clf = RandomForestClassifier(
            n_estimators=int(rf_estimators),
            random_state=int(seed),
            n_jobs=-1,
            class_weight="balanced_subsample",
            max_depth=None if rf_max_depth in (None, 0) else int(rf_max_depth),
        )
        return Pipeline([("tfidf", vec), ("svd", svd), ("clf", clf)])

    raise ValueError(f"Unknown model: {model}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train alternative classical models for incident triage.")
    ap.add_argument("--data", default="data/tickets.csv")
    ap.add_argument("--target", choices=["category", "priority"], required=True)
    ap.add_argument("--model", choices=["nb", "svm", "rf"], required=True)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    # split strategy
    ap.add_argument("--split", choices=["stratified", "time"], default="stratified")
    ap.add_argument("--time-col", type=str, default="timestamp")
    ap.add_argument("--gap-days", type=int, default=0)

    # loader options (only used if load_tickets exists)
    ap.add_argument("--no-dedup", action="store_true", help="Do not drop duplicate texts")
    ap.add_argument("--text-cols", nargs="*", default=None, help="Explicit text columns (e.g., title description)")

    # RF knobs
    ap.add_argument("--svd-dim", type=int, default=256)
    ap.add_argument("--rf-estimators", type=int, default=300)
    ap.add_argument("--rf-max-depth", type=int, default=0, help="0 means None")

    args = ap.parse_args()

    if load_tickets is None:
        # fallback
        df = pd.read_csv(args.data)
        if "text" not in df.columns:
            raise SystemExit("Fallback loader expects a 'text' column in tickets.csv")
        text_col = "text"
        category_col = "category" if "category" in df.columns else None
        priority_col = "priority" if "priority" in df.columns else None
    else:
        loaded = load_tickets(
            args.data,
            text_cols=args.text_cols,
            drop_duplicates_on_text=(not args.no_dedup),
        )
        df = loaded.df
        text_col = loaded.text_col
        category_col = loaded.category_col
        priority_col = loaded.priority_col

    if args.target == "category":
        if not category_col:
            raise SystemExit("Category label column not found in dataset.")
        y_col = category_col
    else:
        if not priority_col:
            raise SystemExit("Priority label column not found in dataset.")
        y_col = priority_col

    df = df[df[y_col].notna()].copy().reset_index(drop=True)
    X = df[text_col].astype(str).tolist()
    y = df[y_col].astype(str).tolist()

    split_meta: Dict[str, Any] = {"strategy": args.split}
    if args.split == "stratified":
        split = stratified_split(y, test_size=float(args.test_size), seed=int(args.seed))
    else:
        if args.time_col not in df.columns:
            raise SystemExit(f"time split requested but '{args.time_col}' column not found in dataset.")
        split, extra = time_aware_split(df[args.time_col], test_size=float(args.test_size), gap_days=int(args.gap_days))
        split_meta.update(extra)

    X_train = [X[i] for i in split.train_idx]
    y_train = [y[i] for i in split.train_idx]
    X_test = [X[i] for i in split.test_idx]
    y_test = [y[i] for i in split.test_idx]

    pipe = build_pipeline(
        args.model,
        svd_dim=int(args.svd_dim),
        rf_estimators=int(args.rf_estimators),
        rf_max_depth=None if int(args.rf_max_depth) == 0 else int(args.rf_max_depth),
        seed=int(args.seed),
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics = {
        "target": args.target,
        "model": args.model,
        "split_strategy": args.split,
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "n_rows": int(len(df)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_test, y_pred)),
    }

    outdir = Path("artifacts") / f"alt_{args.target}_{args.model}"
    ensure_dir(outdir)

    joblib.dump(pipe, outdir / "model.joblib")
    (outdir / "split.json").write_text(json.dumps(asdict(split), indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    meta: Dict[str, Any] = {
        "data_path": str(args.data),
        "text_col": text_col,
        "target_col": y_col,
        "outdir": str(outdir),
        "rows_after_cleaning": int(len(df)),
        "columns": list(df.columns),
        "split_meta": split_meta,
        "note": "Alternative classical models for baseline comparison",
        "rf": {
            "svd_dim": int(args.svd_dim),
            "rf_estimators": int(args.rf_estimators),
            "rf_max_depth": None if int(args.rf_max_depth) == 0 else int(args.rf_max_depth),
        } if args.model == "rf" else None,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Saved -> {outdir / 'model.joblib'}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
