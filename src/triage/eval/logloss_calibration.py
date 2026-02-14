from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split


# -----------------------------
# Helpers
# -----------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_excerpt(text: str, max_len: int = 180) -> str:
    s = (text or "").replace("\n", " ").strip()
    s = " ".join(s.split())
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _as_utc_series(ts: pd.Series) -> pd.Series:
    # Robust timestamp parsing; keep UTC and normalize to tz-aware.
    out = pd.to_datetime(ts, errors="coerce", utc=True)
    if out.isna().any():
        # If some timestamps are missing, we still allow split but warn in output meta.
        pass
    return out


def _time_split_indices(
    df: pd.DataFrame,
    time_col: str,
    test_size: float,
    gap_days: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    ts = _as_utc_series(df[time_col])
    # Order by time (NaT last by default; we push NaT to the beginning to avoid contaminating test)
    order = ts.sort_values(na_position="first").index.to_numpy()

    n = len(df)
    n_test = int(round(n * test_size))
    n_test = max(1, min(n_test, n - 1))

    test_idx = order[-n_test:]
    train_idx = order[:-n_test]

    # Apply gap: remove any training points that are too close to the first test timestamp
    meta: Dict[str, Any] = {}
    test_min = ts.loc[test_idx].min()
    test_max = ts.loc[test_idx].max()
    train_min = ts.loc[train_idx].min()
    train_max = ts.loc[train_idx].max()

    if gap_days and pd.notna(test_min):
        cutoff = test_min - pd.Timedelta(days=int(gap_days))
        keep = ts.loc[train_idx] <= cutoff
        train_idx = train_idx[keep.to_numpy()]

    meta.update(
        {
            "strategy": "time",
            "test_size": test_size,
            "gap_days": int(gap_days),
            "time_col": time_col,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_min_ts": None if pd.isna(train_min) else train_min.isoformat(),
            "train_max_ts": None if pd.isna(train_max) else train_max.isoformat(),
            "test_min_ts": None if pd.isna(test_min) else test_min.isoformat(),
            "test_max_ts": None if pd.isna(test_max) else test_max.isoformat(),
            "nat_timestamps": int(ts.isna().sum()),
        }
    )
    return train_idx, test_idx, meta


def _stratified_split_indices(
    y: pd.Series,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y
    )
    meta = {
        "strategy": "stratified",
        "test_size": test_size,
        "seed": int(seed),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }
    return train_idx, test_idx, meta


def _ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    # Expected Calibration Error (ECE) over max-confidence predictions
    # confidence: [0,1], correct: {0,1}
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidence)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidence >= lo) & (confidence < hi) if i < n_bins - 1 else (confidence >= lo) & (confidence <= hi)
        if not np.any(mask):
            continue
        bin_conf = confidence[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)


@dataclass(frozen=True)
class RunConfig:
    data: str
    model: str
    target: str
    text_col: str
    split: str
    test_size: float
    seed: int
    time_col: str
    gap_days: int
    calibrate: str
    calib_frac: float
    out_dir: str
    report_dir: str
    top_k: int


# -----------------------------
# Main logic
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/tickets.csv")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--target", type=str, choices=["priority", "category"], required=True)
    ap.add_argument("--text-col", type=str, default="text")

    ap.add_argument("--split", type=str, choices=["time", "stratified"], default="time")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-col", type=str, default="timestamp")
    ap.add_argument("--gap-days", type=int, default=1)

    ap.add_argument("--calibrate", type=str, choices=["none", "sigmoid", "isotonic"], default="none")
    ap.add_argument("--calib-frac", type=float, default=0.2, help="fraction of TRAIN used for calibration when calibrate != none")

    ap.add_argument("--out-dir", type=str, default="artifacts/calibration")
    ap.add_argument("--report-dir", type=str, default="reports")
    ap.add_argument("--top-k", type=int, default=10, help="top K confident-but-wrong examples to include")

    args = ap.parse_args()

    cfg = RunConfig(
        data=args.data,
        model=args.model,
        target=args.target,
        text_col=args.text_col,
        split=args.split,
        test_size=args.test_size,
        seed=args.seed,
        time_col=args.time_col,
        gap_days=args.gap_days,
        calibrate=args.calibrate,
        calib_frac=args.calib_frac,
        out_dir=args.out_dir,
        report_dir=args.report_dir,
        top_k=args.top_k,
    )

    df = pd.read_csv(cfg.data)
    if cfg.text_col not in df.columns:
        raise ValueError(f"Missing text column '{cfg.text_col}'. Available: {list(df.columns)}")
    if cfg.target not in df.columns:
        raise ValueError(f"Missing target column '{cfg.target}'. Available: {list(df.columns)}")

    y = df[cfg.target].astype(str)
    X = df[cfg.text_col].astype(str).tolist()

    # Split indices
    if cfg.split == "time":
        if cfg.time_col not in df.columns:
            raise ValueError(f"--split time requires --time-col '{cfg.time_col}', but column not found.")
        train_idx, test_idx, split_meta = _time_split_indices(
            df=df, time_col=cfg.time_col, test_size=cfg.test_size, gap_days=cfg.gap_days
        )
        split_meta["seed"] = int(cfg.seed)  # for consistency in metadata
    else:
        train_idx, test_idx, split_meta = _stratified_split_indices(y=y, test_size=cfg.test_size, seed=cfg.seed)

    if len(train_idx) < 10 or len(test_idx) < 10:
        raise ValueError(f"Split too small (train={len(train_idx)}, test={len(test_idx)}).")

    pipe = joblib.load(cfg.model)

    # Prepare calibration wrapper if requested
    model_used = pipe
    calib_meta: Dict[str, Any] = {"method": cfg.calibrate}

    if cfg.calibrate != "none":
        # Post-hoc probability calibration on a held-out subset of TRAIN.
        # Note: In newer scikit-learn versions, "prefit" calibration is not supported,
        # so CalibratedClassifierCV will refit the estimator internally (CV on the calibration set).
        X_train = [X[int(i)] for i in train_idx]
        y_train = y.iloc[train_idx].reset_index(drop=True)

        fit_idx, cal_idx = train_test_split(
            np.arange(len(X_train)),
            test_size=cfg.calib_frac,
            random_state=cfg.seed,
            stratify=y_train,
        )
        X_cal = [X_train[int(i)] for i in cal_idx]
        y_cal = y_train.iloc[cal_idx].reset_index(drop=True)

        try:
            cal = CalibratedClassifierCV(
                estimator=clone(pipe),
                method=cfg.calibrate,
                cv=3,
            )
            cal.fit(X_cal, y_cal)
            model_used = cal
            calib_meta.update({
                "calib_frac": float(cfg.calib_frac),
                "calib_n": int(len(cal_idx)),
                "cv": 3,
                "refit": True,
            })
        except Exception as e:
            raise RuntimeError(
                "Calibration failed. Estimator must support decision_function or predict_proba. "
                f"Original error: {type(e).__name__}: {e}"
            ) from e
    # Predict on test
    X_test = [X[int(i)] for i in test_idx]
    y_test = y.iloc[test_idx].reset_index(drop=True)

    if not hasattr(model_used, "predict_proba"):
        raise RuntimeError(
            "Model does not support predict_proba(), so log loss cannot be computed as probability loss. "
            "Use a probabilistic model (LogReg/NB/RF) or run with --calibrate sigmoid/isotonic if decision_function exists."
        )

    proba = model_used.predict_proba(X_test)
    pred = model_used.predict(X_test).astype(str)

    # Classes / labels alignment
    if hasattr(model_used, "classes_"):
        labels = [str(x) for x in model_used.classes_]
    elif hasattr(pipe, "classes_"):
        labels = [str(x) for x in pipe.classes_]
    else:
        labels = sorted(y.unique().tolist())

    # Ensure proba columns match labels order; sklearn guarantees predict_proba aligns with classes_.
    # Compute metrics
    acc = float(accuracy_score(y_test, pred))
    macro_f1 = float(f1_score(y_test, pred, average="macro"))

    ll = float(log_loss(y_test, proba, labels=labels))

    conf = proba.max(axis=1)
    correct = (pred == y_test.to_numpy()).astype(float)
    ece = _ece(confidence=conf, correct=correct, n_bins=10)

    metrics: Dict[str, Any] = {
        "target": cfg.target,
        "model_path": cfg.model,
        "split": split_meta,
        "calibration": calib_meta,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "log_loss": ll,
        "ece": ece,
        "labels": labels,
        "n_test": int(len(y_test)),
    }

    # Priority-specific: binary P0/P1 view
    if cfg.target == "priority" and ("P0" in labels or "P1" in labels):
        idx_p0 = labels.index("P0") if "P0" in labels else None
        idx_p1 = labels.index("P1") if "P1" in labels else None
        p01 = np.zeros(len(y_test), dtype=float)
        if idx_p0 is not None:
            p01 += proba[:, idx_p0]
        if idx_p1 is not None:
            p01 += proba[:, idx_p1]

        y_bin = y_test.isin(["P0", "P1"]).astype(int).to_numpy()
        # Binary log loss: need [p(0), p(1)]
        p01 = np.clip(p01, 1e-12, 1 - 1e-12)
        proba_bin = np.vstack([1 - p01, p01]).T
        ll_bin = float(log_loss(y_bin, proba_bin, labels=[0, 1]))
        conf_bin = p01
        pred_bin = (p01 >= 0.5).astype(int)
        ece_bin = _ece(confidence=conf_bin, correct=(pred_bin == y_bin).astype(float), n_bins=10)

        metrics["p0p1_binary"] = {"log_loss": ll_bin, "ece": ece_bin}

    # Confident but wrong
    wrong_mask = pred != y_test.to_numpy()
    wrong_idx = np.where(wrong_mask)[0]
    wrong_conf = conf[wrong_idx]
    order = np.argsort(-wrong_conf)  # high confidence first

    examples: List[Dict[str, Any]] = []
    for j in order[: cfg.top_k]:
        i = int(wrong_idx[j])
        examples.append(
            {
                "true": str(y_test.iloc[i]),
                "pred": str(pred[i]),
                "confidence": float(conf[i]),
                "text_excerpt": _safe_excerpt(str(X_test[i])),
            }
        )
    metrics["top_confident_wrong"] = examples

    # Write outputs
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_dir = Path(cfg.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"{cfg.target}.json"
    out_md = report_dir / f"logloss_calibration_{cfg.target}.md"

    out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown report
    lines: List[str] = []
    lines.append(f"# Log Loss & Calibration Report — {cfg.target}")
    lines.append("")
    lines.append(f"- Generated: `{_utc_now_iso()}`")
    lines.append(f"- Model: `{cfg.model}`")
    lines.append(f"- Split: `{split_meta.get('strategy')}` (test_size={cfg.test_size})")
    if split_meta.get("strategy") == "time":
        lines.append(f"  - time_col={split_meta.get('time_col')}, gap_days={split_meta.get('gap_days')}")
        lines.append(f"  - train_max_ts={split_meta.get('train_max_ts')}")
        lines.append(f"  - test_min_ts={split_meta.get('test_min_ts')}")
    lines.append(f"- Calibration: `{cfg.calibrate}`")
    lines.append("")
    lines.append("## Why log loss (cross entropy)?")
    lines.append("- Accuracy: **right/wrong**")
    lines.append("- Log loss: **probability quality** (penalizes **confident but wrong**)")
    lines.append("")
    lines.append("## Metrics")
    lines.append(f"- Accuracy: **{acc:.4f}**")
    lines.append(f"- Macro F1: **{macro_f1:.4f}**")
    lines.append(f"- Log loss: **{ll:.4f}**")
    lines.append(f"- ECE (10 bins): **{ece:.4f}**")
    if "p0p1_binary" in metrics:
        lines.append("")
        lines.append("### Priority safety view (P0/P1 vs others)")
        lines.append(f"- P0/P1 binary log loss: **{metrics['p0p1_binary']['log_loss']:.4f}**")
        lines.append(f"- P0/P1 binary ECE: **{metrics['p0p1_binary']['ece']:.4f}**")
    lines.append("")
    lines.append("## Top confident-but-wrong examples")
    if not examples:
        lines.append("- ✅ No misclassifications in the evaluation split.")
    else:
        for ex in examples:
            lines.append(f"- **{ex['true']} → {ex['pred']}** (conf={ex['confidence']:.3f}) — `{ex['text_excerpt']}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- If log loss is high while accuracy is decent, the model is likely **miscalibrated** (over/under-confident).")
    lines.append("- For triage, focus on reducing **high-confidence mistakes** on P0/P1 to avoid expensive escalations.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- JSON: `{out_json.as_posix()}`")
    lines.append(f"- This report: `{out_md.as_posix()}`")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Wrote {out_json} and {out_md}")
    print(json.dumps({k: metrics[k] for k in ['target','accuracy','macro_f1','log_loss','ece']}, indent=2))


if __name__ == "__main__":
    main()