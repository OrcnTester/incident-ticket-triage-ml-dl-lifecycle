from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.triage.models.pipeline_factory import build_pipeline, VectorizerConfig
from src.triage.models.model_io import build_default_meta, save_bundle, load_bundle


try:
    from src.triage.data.load import load_tickets  # type: ignore
except Exception:  # pragma: no cover
    load_tickets = None  # type: ignore


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline design demo: train → joblib save → reload → verify")
    ap.add_argument("--data", type=str, default="data/tickets.csv")
    ap.add_argument("--target", choices=["category", "priority"], required=True)
    ap.add_argument("--model", choices=["logreg", "nb", "svm", "rf"], required=True)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    # vectorizer knobs
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--max-features", type=int, default=200_000)

    # rf knobs
    ap.add_argument("--svd-dim", type=int, default=256)
    ap.add_argument("--rf-estimators", type=int, default=300)

    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument("--report", type=str, default="")
    args = ap.parse_args()

    if load_tickets is None:
        df = pd.read_csv(args.data)
        if "text" not in df.columns:
            raise SystemExit("Fallback loader expects tickets.csv to include a 'text' column.")
        text_col = "text"
        y_col = args.target
    else:
        loaded = load_tickets(args.data, drop_duplicates_on_text=True)
        df = loaded.df
        text_col = loaded.text_col
        y_col = loaded.category_col if args.target == "category" else loaded.priority_col
        if not y_col:
            raise SystemExit(f"Target column for {args.target} not found by loader.")

    df = df[df[y_col].notna()].copy().reset_index(drop=True)
    X = df[text_col].astype(str).tolist()
    y = df[y_col].astype(str).tolist()

    # very simple stratified split via group sampling
    idx = np.arange(len(y))
    df_idx = pd.DataFrame({"i": idx, "y": y})
    test_idx = []
    for _, g in df_idx.groupby("y"):
        g = g.sample(frac=float(args.test_size), random_state=int(args.seed))
        test_idx.extend(g["i"].tolist())
    test_idx = sorted(set(map(int, test_idx)))
    train_idx = sorted(set(map(int, idx)) - set(test_idx))

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    vec_cfg = VectorizerConfig(min_df=int(args.min_df), max_features=int(args.max_features))
    pipe = build_pipeline(
        args.model,
        vec_cfg=vec_cfg,
        seed=int(args.seed),
        svd_dim=int(args.svd_dim),
        rf_estimators=int(args.rf_estimators),
    )

    pipe.fit(X_train, y_train)
    pred1 = pipe.predict(X_test)

    metrics = {
        "target": args.target,
        "model": args.model,
        "n_rows": int(len(df)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "macro_f1": float(f1_score(y_test, pred1, average="macro")),
        "weighted_f1": float(f1_score(y_test, pred1, average="weighted")),
        "accuracy": float(accuracy_score(y_test, pred1)),
    }

    out_dir = Path(args.out_dir) if args.out_dir else Path("artifacts") / f"pipeline_demo_{args.target}_{args.model}"
    report_path = Path(args.report) if args.report else Path("reports") / f"pipeline_design_{args.target}_{args.model}.md"
    ensure_dir(out_dir)
    ensure_dir(report_path.parent)

    meta = {
        "env": build_default_meta(notes="pipeline design demo").__dict__,
        "data_path": str(args.data),
        "text_col": text_col,
        "target_col": y_col,
        "vectorizer": vec_cfg.__dict__,
        "model": args.model,
        "rf": {"svd_dim": int(args.svd_dim), "rf_estimators": int(args.rf_estimators)} if args.model == "rf" else None,
        "split": {"strategy": "stratified_group_sample", "test_size": float(args.test_size), "seed": int(args.seed)},
        "metrics": metrics,
    }

    save_bundle(out_dir, pipe, meta)

    # round-trip check
    pipe2, _, compat = load_bundle(out_dir)
    pred2 = pipe2.predict(X_test)
    same = bool(np.array_equal(np.asarray(pred1), np.asarray(pred2)))

    # report
    steps = list(pipe.named_steps.keys())
    vocab_size = None
    if "tfidf" in pipe.named_steps:
        tfidf = pipe.named_steps["tfidf"]
        vocab_size = len(getattr(tfidf, "vocabulary_", {}) or {})

    lines: List[str] = []
    lines.append(f"# Pipeline Design Demo — {args.target} / {args.model}")
    lines.append("")
    lines.append("## Pipeline structure")
    lines.append(f"- steps: `{steps}`")
    if vocab_size is not None:
        lines.append(f"- tfidf vocab_size: `{vocab_size}`")
    lines.append("")
    lines.append("## Serialization round-trip")
    lines.append(f"- saved to: `{out_dir.as_posix()}`")
    lines.append(f"- compatibility warnings: `{compat.get('warnings', [])}`")
    lines.append(f"- predictions identical after reload: **{same}**")
    lines.append("")
    lines.append("## Metrics (holdout)")
    lines.append("```json")
    lines.append(json.dumps(metrics, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Vectorizer + model in one Pipeline prevents train/inference mismatch.")
    lines.append("- meta.json stores python/sklearn versions to catch persistence issues early.")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Saved -> {out_dir / 'model.joblib'}")
    print(f"[OK] Wrote report -> {report_path}")
    print(json.dumps({"round_trip_same": same, **metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
