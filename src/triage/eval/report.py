from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

from src.triage.data.load import load_tickets


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_split(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def label_distribution(labels: List[str]) -> Dict[str, Any]:
    """
    Returns counts + ratios for a label list.
    """
    n = len(labels)
    counts: Dict[str, int] = {}
    for y in labels:
        counts[y] = counts.get(y, 0) + 1

    ratios = {k: (v / n if n else 0.0) for k, v in counts.items()}
    ordered_keys = sorted(counts.keys())
    return {
        "n": n,
        "counts": {k: counts[k] for k in ordered_keys},
        "ratios": {k: ratios[k] for k in ordered_keys},
    }


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


def p0_recall(y_true: List[str], y_pred: List[str]) -> float:
    """
    Recall for class P0 treated as positive vs all others.
    """
    y_true_bin = [1 if y == "P0" else 0 for y in y_true]
    y_pred_bin = [1 if y == "P0" else 0 for y in y_pred]
    return float(recall_score(y_true_bin, y_pred_bin, zero_division=0))


def p0p1_binary_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Binary gate metrics for "high severity": positive = {P0, P1}.
    """
    y_true_bin = [1 if y in ("P0", "P1") else 0 for y in y_true]
    y_pred_bin = [1 if y in ("P0", "P1") else 0 for y in y_pred]

    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    cm = confusion_matrix(y_true_bin, y_pred_bin).tolist()
    return {"recall": float(rec), "precision": float(prec), "confusion_matrix": cm}


def _split_meta(split: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        "strategy",
        "test_size",
        "seed",
        "time_col",
        "gap_days",
        "cutoff_timestamp",
        "train_max_time",
        "test_min_time",
        "test_max_time",
        "dropped_bad_timestamps",
    ]
    return {k: split.get(k) for k in keep if k in split}


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

    train_idx = split.get("train_idx", [])
    test_idx = split.get("test_idx", [])

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
        # imbalance evidence
        "label_distribution_test": label_distribution(y_test),
        # split evidence (helps for leakage discussion)
        "split": _split_meta(split),
    }

    if train_idx:
        y_train = [y[i] for i in train_idx]
        out["label_distribution_train"] = label_distribution(y_train)
    else:
        out["label_distribution_train"] = None

    if name == "priority":
        out["severe_mistake_rate"] = severe_mistake_rate(y_test, y_pred)
        out["p0_recall"] = p0_recall(y_test, y_pred)
        out["p0p1_binary"] = p0p1_binary_metrics(y_test, y_pred)

    return out


def _fmt_dist(d: Optional[Dict[str, Any]]) -> str:
    if not d:
        return "n/a"
    counts = d.get("counts", {})
    ratios = d.get("ratios", {})
    parts = []
    for k in counts.keys():
        parts.append(f"{k}:{counts[k]} ({ratios.get(k, 0.0)*100:.1f}%)")
    return ", ".join(parts) if parts else "n/a"


def _fmt_split_meta(m: Dict[str, Any]) -> str:
    if not m:
        return "n/a"
    if m.get("strategy") != "time":
        return f"strategy={m.get('strategy')} seed={m.get('seed')} test_size={m.get('test_size')}"
    return (
        f"strategy=time time_col={m.get('time_col')} test_size={m.get('test_size')} gap_days={m.get('gap_days')} "
        f"train_max={m.get('train_max_time')} test_min={m.get('test_min_time')}"
    )


def main() -> None:
    loaded = load_tickets("data/tickets.csv", drop_duplicates_on_text=True)
    df = loaded.df

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
        report_lines.append(
            f"macro_f1={cat['macro_f1']:.4f} weighted_f1={cat['weighted_f1']:.4f} acc={cat['accuracy']:.4f}"
        )
        report_lines.append("Split: " + _fmt_split_meta(cat.get("split", {}) or {}))
        report_lines.append("Label distribution (train): " + _fmt_dist(cat.get("label_distribution_train")))
        report_lines.append("Label distribution (test):  " + _fmt_dist(cat.get("label_distribution_test")))
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

        p0r = pr.get("p0_recall", 0.0)
        p0p1 = pr.get("p0p1_binary", {}) or {}
        report_lines.append(
            f"p0_recall={p0r:.4f} "
            f"p0p1_recall={float(p0p1.get('recall', 0.0)):.4f} "
            f"p0p1_precision={float(p0p1.get('precision', 0.0)):.4f}"
        )
        report_lines.append("Split: " + _fmt_split_meta(pr.get("split", {}) or {}))
        report_lines.append("P0/P1 confusion matrix (binary): " + str(p0p1.get("confusion_matrix")))
        report_lines.append("Label distribution (train): " + _fmt_dist(pr.get("label_distribution_train")))
        report_lines.append("Label distribution (test):  " + _fmt_dist(pr.get("label_distribution_test")))
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
