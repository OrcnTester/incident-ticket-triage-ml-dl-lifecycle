from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from src.triage.data.load import load_tickets


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_split(path: Path) -> Dict[str, List[int]]:
    return json.loads(path.read_text(encoding="utf-8"))


def severe_mistake_rate(y_true: List[str], y_pred: List[str]) -> float:
    mapping = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    pairs = []
    for t, p in zip(y_true, y_pred):
        if t in mapping and p in mapping:
            pairs.append((mapping[t], mapping[p]))
    if not pairs:
        return 0.0
    return float(sum(1 for t, p in pairs if abs(p - t) >= 2) / len(pairs))


def eval_model(model_path: Path, split_path: Path, X: List[str], y: List[str], *, target: str) -> Dict[str, Any]:
    pipe = joblib.load(model_path)
    split = load_split(split_path)
    test_idx = split["test_idx"]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    y_pred = pipe.predict(X_test).tolist()

    out: Dict[str, Any] = {
        "n_test": int(len(y_test)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "labels": sorted(list(set(y_test))),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, digits=3),
    }

    if target == "priority":
        # P0/P1 binary view is more operationally meaningful
        y_true_bin = np.array([1 if t in ("P0", "P1") else 0 for t in y_test], dtype=int)
        y_pred_bin = np.array([1 if p in ("P0", "P1") else 0 for p in y_pred], dtype=int)
        out["p0p1_binary"] = {
            "recall": float(recall_score(y_true_bin, y_pred_bin)),
            "precision": float(precision_score(y_true_bin, y_pred_bin)),
            "confusion_matrix": confusion_matrix(y_true_bin, y_pred_bin).tolist(),
        }
        out["severe_mistake_rate"] = severe_mistake_rate(y_test, y_pred)

    return out


def discover_artifacts() -> List[Tuple[str, str, Path]]:
    """
    Returns list of (target, model, dir) for artifacts/alt_<target>_<model>
    """
    root = Path("artifacts")
    out = []
    for p in root.glob("alt_*_*"):
        if not p.is_dir():
            continue
        parts = p.name.split("_", 2)
        if len(parts) != 3:
            continue
        _alt, target, model = parts
        if (p / "model.joblib").exists() and (p / "split.json").exists():
            out.append((target, model, p))
    return sorted(out)


def main() -> None:
    loaded = load_tickets("data/tickets.csv", drop_duplicates_on_text=True)
    df = loaded.df
    X_all = df[loaded.text_col].astype(str).tolist()

    results: Dict[str, Any] = {"category": {}, "priority": {}}
    lines: List[str] = []
    lines.append("# Alternative Models Report")
    lines.append("")
    lines.append("This report evaluates models saved under `artifacts/alt_<target>_<model>/` using their persisted `split.json`.")
    lines.append("")

    found = discover_artifacts()
    if not found:
        lines.append("No alternative artifacts found. Train some models first:")
        lines.append("```bash")
        lines.append("python -m src.triage.models.train_alternatives --target category --model svm")
        lines.append("```")
        ensure_dir(Path("reports"))
        Path("reports/alt_models_report.md").write_text("\n".join(lines), encoding="utf-8")
        Path("reports/alt_models_metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print("[OK] Wrote reports/alt_models_report.md and reports/alt_models_metrics.json")
        return

    # Evaluate per target with correct filtered df (drop NaNs)
    for target in ("category", "priority"):
        y_col = loaded.category_col if target == "category" else loaded.priority_col
        if not y_col:
            continue
        dft = df[df[y_col].notna()].copy().reset_index(drop=True)
        Xt = dft[loaded.text_col].astype(str).tolist()
        yt = dft[y_col].astype(str).tolist()

        lines.append(f"## Target: {target}")
        lines.append("")
        lines.append("| model | macro_f1 | weighted_f1 | accuracy | ops_notes |")
        lines.append("|---|---:|---:|---:|---|")

        rows = []
        for t, model, d in found:
            if t != target:
                continue
            ev = eval_model(d / "model.joblib", d / "split.json", Xt, yt, target=target)
            results[target][model] = ev

            ops = ""
            if target == "priority":
                p = ev["p0p1_binary"]
                ops = f"P0/P1 recall={p['recall']:.3f}, precision={p['precision']:.3f}, severe={ev.get('severe_mistake_rate',0.0):.3f}"
            rows.append((model, ev["macro_f1"], ev["weighted_f1"], ev["accuracy"], ops))

        # sort: category by macro_f1; priority by P0/P1 recall then macro_f1
        if target == "priority":
            def key_fn(r):
                m = results[target][r[0]].get("p0p1_binary", {})
                return (m.get("recall", 0.0), r[1])
            rows.sort(key=key_fn, reverse=True)
        else:
            rows.sort(key=lambda r: r[1], reverse=True)

        for model, mf1, wf1, acc, ops in rows:
            lines.append(f"| {model} | {mf1:.4f} | {wf1:.4f} | {acc:.4f} | {ops} |")

        lines.append("")
        # include short classification report for top model
        if rows:
            top_model = rows[0][0]
            lines.append(f"### Top model: {top_model}")
            lines.append("```")
            lines.append(results[target][top_model]["classification_report"])
            lines.append("```")
            lines.append("")

    ensure_dir(Path("reports"))
    Path("reports/alt_models_report.md").write_text("\n".join(lines), encoding="utf-8")
    Path("reports/alt_models_metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[OK] Wrote reports/alt_models_report.md and reports/alt_models_metrics.json")


if __name__ == "__main__":
    main()
