from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score

from src.triage.data.load import load_tickets
from src.triage.data.split import stratified_split, time_aware_split


def build_pipeline(model_type: str) -> Pipeline:
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=200_000,
    )

    if model_type == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=None,
        )
    elif model_type == "linear_svc":
        clf = LinearSVC(class_weight="balanced")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline([("tfidf", vec), ("clf", clf)])


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def infer_time_col(df: pd.DataFrame) -> Optional[str]:
    # keep it simple + explicit for portfolio
    candidates = ["timestamp", "created_at", "created", "time", "ts"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tickets.csv")
    ap.add_argument("--target", choices=["category", "priority"], required=True)
    ap.add_argument("--model", choices=["logreg", "linear_svc"], default="logreg")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    # leakage-aware split knobs
    ap.add_argument("--split", choices=["stratified", "time"], default="stratified")
    ap.add_argument("--time-col", default=None, help="Timestamp column name (default: infer)")
    ap.add_argument("--gap-days", type=int, default=0, help="Optional gap between train and test (time split only)")

    ap.add_argument("--no-dedup", action="store_true", help="Do not drop duplicate texts")
    ap.add_argument("--text-cols", nargs="*", default=None, help="Explicit text columns (e.g. title description)")
    args = ap.parse_args()

    loaded = load_tickets(
        args.data,
        text_cols=args.text_cols,
        drop_duplicates_on_text=(not args.no_dedup),
    )
    df = loaded.df

    if args.target == "category":
        if not loaded.category_col:
            raise SystemExit("Category label column not found. Ensure tickets.csv has a category/label column.")
        y_col = loaded.category_col
        outdir = Path("artifacts") / "baseline_category"
    else:
        if not loaded.priority_col:
            raise SystemExit("Priority label column not found. Ensure tickets.csv has a priority/p column.")
        y_col = loaded.priority_col
        outdir = Path("artifacts") / "baseline_priority"

    # Drop rows with missing target
    df = df[df[y_col].notna()].copy().reset_index(drop=True)

    # Build split (stratified default; time-aware optional)
    split_payload: Dict[str, Any]
    if args.split == "time":
        time_col = args.time_col or infer_time_col(df)
        if not time_col:
            raise SystemExit(
                "Time-aware split requested but no time column found. "
                "Provide --time-col or add a timestamp column to tickets.csv."
            )

        # parse time, drop bad rows (keep indices consistent for report.py)
        ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        n_bad = int(ts.isna().sum())
        if n_bad:
            df = df[ts.notna()].copy().reset_index(drop=True)
            ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)

        X = df[loaded.text_col].astype(str).tolist()
        y = df[y_col].astype(str).tolist()

        split, meta = time_aware_split(ts, test_size=args.test_size, gap_days=args.gap_days)
        split_payload = {
            **asdict(split),
            **meta,
            "seed": int(args.seed),  # keep for completeness (even if not used)
            "time_col": str(time_col),
            "dropped_bad_timestamps": int(n_bad),
        }
    else:
        X = df[loaded.text_col].astype(str).tolist()
        y = df[y_col].astype(str).tolist()

        split = stratified_split(y, test_size=args.test_size, seed=args.seed)
        split_payload = {
            **asdict(split),
            "strategy": "stratified",
            "test_size": float(args.test_size),
            "seed": int(args.seed),
        }

    X_train = [X[i] for i in split_payload["train_idx"]]
    y_train = [y[i] for i in split_payload["train_idx"]]
    X_test = [X[i] for i in split_payload["test_idx"]]
    y_test = [y[i] for i in split_payload["test_idx"]]

    pipe = build_pipeline(args.model)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    metrics = {
        "target": args.target,
        "model": args.model,
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "split_strategy": split_payload.get("strategy", "stratified"),
        "n_rows": int(len(df)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_test, y_pred)),
    }

    ensure_dir(outdir)
    joblib.dump(pipe, outdir / "model.joblib")

    (outdir / "split.json").write_text(
        json.dumps(split_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (outdir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    meta: Dict[str, Any] = {
        "data_path": str(args.data),
        "text_col": loaded.text_col,
        "target_col": y_col,
        "outdir": str(outdir),
        "rows_after_cleaning": int(len(df)),
        "columns": list(df.columns),
        "note": "Baseline training for Option A (two independent models) per docs/08_algorithm_formulation.md",
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Saved -> {outdir / 'model.joblib'}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
