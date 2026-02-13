from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from src.triage.data.load import load_tickets


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_split(path: Path) -> Optional[Dict[str, List[int]]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def severe_mistake_rate(y_true: List[str], y_pred: List[str]) -> float:
    """
    Severe mistake: |pred-true| >= 2 when mapping P0..P3 -> 0..3
    """
    mapping = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    pairs = []
    for t, p in zip(y_true, y_pred):
        if t not in mapping or p not in mapping:
            continue
        pairs.append((mapping[t], mapping[p]))
    if not pairs:
        return 0.0
    sev = [1 for t, p in pairs if abs(p - t) >= 2]
    return float(sum(sev) / len(pairs))


def evaluate_target(
    *,
    name: str,
    model_path: Path,
    split_path: Path,
    X: List[str],
    y: List[str],
) -> Dict[str, Any]:
    pipe = joblib.load(model_path)
    split = load_split(split_path)

    if split is None:
        raise RuntimeError(f"Missing split file: {split_path}. Train the model first.")

    test_idx = split["test_idx"]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    y_pred = pipe.predict(X_test).tolist()

    out: Dict[str, Any] = {
        "target": name,
        "n_test": len(y_test),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "labels": sorted(list(set(y_test))),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, digits=3),
    }

    if name == "priority":
        out["severe_mistake_rate"] = severe_mistake_rate(y_test, y_pred)

    return out


def main() -> None:
    # Load dataset
    loaded = load_tickets("data/tickets.csv", drop_duplicates_on_text=True)
    df = loaded.df

    # Build X
    X = df[loaded.text_col].astype(str).tolist()

    results: Dict[str, Any] = {}
    report_lines: List[str] = []

    # CATEGORY
    cat_dir = Path("artifacts") / "baseline_category"
    if loaded.category_col and (cat_dir / "model.joblib").exists():
        df_cat = df[df[loaded.category_col].notna()].copy().reset_index(drop=True)
        X_cat = df_cat[loaded.text_col].astype(str).tolist()
        y_cat = df_cat[loaded.category_col].astype(str).tolist()

        cat = evaluate_target(
            name="category",
            model_path=cat_dir / "model.joblib",
            split_path=cat_dir / "split.json",
            X=X_cat,
            y=y_cat,
        )
        results["category"] = cat
        report_lines.append("=== CATEGORY (baseline) ===")
        report_lines.append(cat["classification_report"])
        report_lines.append(f"macro_f1={cat['macro_f1']:.4f} weighted_f1={cat['weighted_f1']:.4f} acc={cat['accuracy']:.4f}")
        report_lines.append("")
    else:
        report_lines.append("=== CATEGORY (baseline) ===")
        report_lines.append("Model not found. Train it first:")
        report_lines.append("  python -m src.triage.models.train_baseline --target category")
        report_lines.append("")

    # PRIORITY
    pr_dir = Path("artifacts") / "baseline_priority"
    if loaded.priority_col and (pr_dir / "model.joblib").exists():
        df_pr = df[df[loaded.priority_col].notna()].copy().reset_index(drop=True)
        X_pr = df_pr[loaded.text_col].astype(str).tolist()
        y_pr = df_pr[loaded.priority_col].astype(str).tolist()

        pr = evaluate_target(
            name="priority",
            model_path=pr_dir / "model.joblib",
            split_path=pr_dir / "split.json",
            X=X_pr,
            y=y_pr,
        )
        results["priority"] = pr
        report_lines.append("=== PRIORITY (baseline) ===")
        report_lines.append(pr["classification_report"])
        report_lines.append(
            f"macro_f1={pr['macro_f1']:.4f} weighted_f1={pr['weighted_f1']:.4f} acc={pr['accuracy']:.4f} "
            f"severe_mistake_rate={pr.get('severe_mistake_rate', 0.0):.4f}"
        )
        report_lines.append("")
    else:
        report_lines.append("=== PRIORITY (baseline) ===")
        report_lines.append("Model not found. Train it first:")
        report_lines.append("  python -m src.triage.models.train_baseline --target priority")
        report_lines.append("")

    ensure_dir(Path("reports"))
    Path("reports/report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    Path("reports/metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] Wrote reports/report.txt and reports/metrics.json")


if __name__ == "__main__":
    main()
