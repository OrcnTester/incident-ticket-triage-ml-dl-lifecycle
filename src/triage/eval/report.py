from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/tickets.csv")
    ap.add_argument("--model", type=str, default="artifacts/baseline_priority/model.joblib")
    ap.add_argument("--target", type=str, choices=["priority", "category"], default="priority")
    ap.add_argument("--test", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="artifacts/reports/report.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X = df[["text"]].copy()
    y = df[args.target].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test, random_state=args.seed, stratify=y
    )

    model = joblib.load(args.model)
    preds = model.predict(X_test)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"Target: {args.target}")
    lines.append("")

    if args.target == "category":
        macro_f1 = f1_score(y_test, preds, average="macro")
        lines.append(f"Macro F1: {macro_f1:.4f}")
    else:
        # priority: focus on P0/P1 recall and precision
        y_true_bin = y_test.isin(["P0", "P1"]).astype(int)
        y_pred_bin = pd.Series(preds).isin(["P0", "P1"]).astype(int)

        rec = recall_score(y_true_bin, y_pred_bin)
        prec = precision_score(y_true_bin, y_pred_bin)

        lines.append(f"P0/P1 Recall (binary): {rec:.4f}")
        lines.append(f"P0/P1 Precision (binary): {prec:.4f}")
        lines.append("")
        lines.append("Confusion matrix (P0/P1 vs others):")
        lines.append(str(confusion_matrix(y_true_bin, y_pred_bin)))

    lines.append("")
    lines.append("Classification report:")
    lines.append(classification_report(y_test, preds))

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Wrote report to {out_path}")
    print("\n".join(lines[:10]))


if __name__ == "__main__":
    main()
