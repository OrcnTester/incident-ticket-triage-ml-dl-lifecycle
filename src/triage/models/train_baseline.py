from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class TrainConfig:
    data_path: str
    out_dir: str
    target: str  # "priority" or "category"
    test_size: float
    random_state: int


def build_pipeline() -> Pipeline:
    # text + simple one-hot on small structured fields
    pre = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000), "text"),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=None,
        class_weight="balanced",  # helps imbalance in synthetic
        multi_class="auto",
    )

    return Pipeline([("pre", pre), ("clf", clf)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/tickets.csv")
    ap.add_argument("--out", type=str, default="artifacts/baseline_priority")
    ap.add_argument("--target", type=str, choices=["priority", "category"], default="priority")
    ap.add_argument("--test", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = TrainConfig(
        data_path=args.data,
        out_dir=args.out,
        target=args.target,
        test_size=args.test,
        random_state=args.seed,
    )

    df = pd.read_csv(cfg.data_path)
    if cfg.target not in df.columns:
        raise ValueError(f"Target '{cfg.target}' not found in dataset columns: {list(df.columns)}")

    X = df[["text"]].copy()
    y = df[cfg.target].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)

    print("=== Classification report (holdout) ===")
    print(classification_report(y_test, preds))

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, out_dir / "model.joblib")
    joblib.dump({"target": cfg.target}, out_dir / "meta.joblib")

    print(f"✅ Saved model to {out_dir / 'model.joblib'} (target={cfg.target})")


if __name__ == "__main__":
    main()
