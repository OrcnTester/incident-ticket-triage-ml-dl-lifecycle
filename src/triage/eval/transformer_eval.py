"""
Transformer Classification Evaluation | Beyond Accuracy

Produces:
- reports/transformer_eval_<target>.md
- artifacts/transformer_eval/<target>.json

Works for transformers (logits/probs) and classical models (joblib) if you can produce probs.

Examples:
  python -m src.triage.eval.transformer_eval --target priority --model artifacts/baseline_priority/model.joblib --data data/tickets.csv --split time --time-col timestamp --gap-days 1
  python -m src.triage.eval.transformer_eval --target priority --probs-npz artifacts/transformer_probs_priority.npz --data data/tickets.csv --split time --time-col timestamp --gap-days 1
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_excerpt(s: str, max_len: int = 140) -> str:
    s = (s or "").replace("\n", " ").strip()
    return (s[: max_len - 3] + "...") if len(s) > max_len else s


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = np.asarray(logits)
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exps = np.exp(logits)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def expected_calibration_error(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10
) -> Tuple[float, List[Dict[str, float]]]:
    probs = _as_numpy(probs)
    y_true = _as_numpy(y_true).astype(int)

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    summary: List[Dict[str, float]] = []

    for i in range(n_bins):
        lo, hi = float(bins[i]), float(bins[i + 1])
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        bin_conf = float(conf[mask].mean())
        bin_acc = float(correct[mask].mean())
        frac = float(mask.mean())
        ece += abs(bin_acc - bin_conf) * frac
        summary.append(
            {"bin_lo": lo, "bin_hi": hi, "count": int(mask.sum()), "mean_conf": bin_conf, "acc": bin_acc, "frac": frac}
        )

    return float(ece), summary


PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def severe_mistake_rate_priority(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> float:
    if set(labels) != {"P0", "P1", "P2", "P3"}:
        return float("nan")

    y_true = _as_numpy(y_true)
    y_pred = _as_numpy(y_pred)
    idx_to_label = {i: lab for i, lab in enumerate(labels)}
    true_levels = np.array([PRIORITY_ORDER.get(idx_to_label[int(i)], 999) for i in y_true], dtype=int)
    pred_levels = np.array([PRIORITY_ORDER.get(idx_to_label[int(i)], 999) for i in y_pred], dtype=int)
    dist = np.abs(true_levels - pred_levels)
    return float(np.mean(dist >= 2))


@dataclass
class SplitConfig:
    strategy: str = "stratified"
    test_size: float = 0.2
    seed: int = 42
    time_col: str = "timestamp"
    gap_days: int = 0


@dataclass
class DataConfig:
    text_col: str = "text"
    target_col: str = "category"


def _load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _coerce_timestamp(df: pd.DataFrame, col: str) -> pd.Series:
    ts = pd.to_datetime(df[col], errors="coerce", utc=True)
    if ts.isna().any():
        raise ValueError(f"Failed to parse {int(ts.isna().sum())} timestamps in '{col}'.")
    return ts


def time_aware_split_indices(ts: pd.Series, test_size: float, gap_days: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    ts = pd.to_datetime(ts, utc=True)
    order = np.argsort(ts.values)
    n = len(ts)
    n_test = max(1, int(math.floor(n * test_size)))
    test_idx = order[-n_test:]
    train_idx = order[:-n_test]

    meta: Dict[str, Any] = {}
    test_min = ts.iloc[test_idx].min()
    if gap_days and gap_days > 0:
        cutoff = test_min - pd.Timedelta(days=int(gap_days))
        keep = ts.iloc[train_idx] <= cutoff
        train_idx = train_idx[keep.values]

    meta.update(
        {
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_min_ts": ts.iloc[train_idx].min().isoformat(),
            "train_max_ts": ts.iloc[train_idx].max().isoformat(),
            "test_min_ts": ts.iloc[test_idx].min().isoformat(),
            "test_max_ts": ts.iloc[test_idx].max().isoformat(),
            "gap_days": int(gap_days),
        }
    )
    return train_idx, test_idx, meta


def stratified_split_indices(y: pd.Series, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(len(y))
    train_idx, test_idx = next(sss.split(idx, y.astype(str).values))
    return train_idx, test_idx, {"n_train": int(len(train_idx)), "n_test": int(len(test_idx))}


# Backward-compat wrappers (older demo scripts imported these names)
def compute_time_split_indices(
    df: pd.DataFrame,
    time_col: str,
    test_size: float,
    seed: int = 42,
    gap_days: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    _ = seed  # time split deterministic given timestamps
    ts = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    return time_aware_split_indices(ts, test_size=test_size, gap_days=gap_days)


def compute_stratified_split_indices(
    df: pd.DataFrame,
    y_col: str,
    test_size: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    y = df[y_col]
    return stratified_split_indices(y, test_size=test_size, seed=seed)


def load_joblib_model(path: str):
    import joblib

    return joblib.load(path)


def load_probs_npz(path: str) -> Dict[str, Any]:
    arr = np.load(path, allow_pickle=True)
    if "probs" not in arr:
        raise ValueError(f"NPZ missing required key 'probs': {path}")
    out: Dict[str, Any] = {"probs": np.asarray(arr["probs"])}
    if "labels" in arr:
        out["labels"] = [str(x) for x in arr["labels"].tolist()]
    if "ids" in arr:
        out["ids"] = arr["ids"].tolist()
    return out


def _align_probs_to_labels(probs: np.ndarray, src_labels: Sequence[str], dst_labels: Sequence[str]) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D (n, C). Got shape={probs.shape}")
    idx = {str(l): j for j, l in enumerate(src_labels)}
    n = probs.shape[0]
    out = np.zeros((n, len(dst_labels)), dtype=float)
    for k, lab in enumerate(dst_labels):
        j = idx.get(str(lab))
        if j is not None:
            out[:, k] = probs[:, j]

    row_sum = out.sum(axis=1, keepdims=True)
    bad = row_sum.squeeze() <= 0
    if np.any(bad):
        out[bad, :] = 1.0 / max(len(dst_labels), 1)
        row_sum = out.sum(axis=1, keepdims=True)
    return out / row_sum


def extract_probs_and_pred(
    pipe: Any,
    npz: Optional[Dict[str, Any]],
    X_test: List[str],
    labels: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    if npz is not None:
        probs = np.asarray(npz["probs"], dtype=float)
        if probs.shape[0] != len(X_test):
            raise ValueError(f"NPZ probs n={probs.shape[0]} does not match X_test n={len(X_test)}")
        src_labels = npz.get("labels", labels)
        probs_aligned = _align_probs_to_labels(probs, src_labels, labels)
        pred_idx = probs_aligned.argmax(axis=1).astype(int)
        return probs_aligned, pred_idx

    if hasattr(pipe, "predict_proba"):
        probs = np.asarray(pipe.predict_proba(X_test), dtype=float)
        src_labels = getattr(pipe, "classes_", None)
        if src_labels is None and hasattr(pipe, "named_steps"):
            last = list(pipe.named_steps.values())[-1]
            src_labels = getattr(last, "classes_", None)
        src_labels = [str(x) for x in (src_labels.tolist() if hasattr(src_labels, "tolist") else (src_labels or labels))]
        probs_aligned = _align_probs_to_labels(probs, src_labels, labels)
        pred_idx = probs_aligned.argmax(axis=1).astype(int)
        return probs_aligned, pred_idx

    if hasattr(pipe, "decision_function"):
        scores = np.asarray(pipe.decision_function(X_test), dtype=float)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        probs = softmax(scores, axis=1)
        probs_aligned = _align_probs_to_labels(probs, labels, labels)
        pred_idx = probs_aligned.argmax(axis=1).astype(int)
        return probs_aligned, pred_idx

    pred = [str(x) for x in np.asarray(pipe.predict(X_test)).tolist()]
    idx = {lab: i for i, lab in enumerate(labels)}
    pred_idx = np.array([idx.get(p, 0) for p in pred], dtype=int)
    probs = np.zeros((len(X_test), len(labels)), dtype=float)
    probs[np.arange(len(X_test)), pred_idx] = 1.0
    return probs, pred_idx


def build_error_buckets(
    df_test: pd.DataFrame,
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    probs: np.ndarray,
    labels: List[str],
    text_col: str,
    max_pairs: int = 5,
    max_examples_per_pair: int = 3,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(len(labels))))
    pairs: List[Tuple[int, int, int]] = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                continue
            c = int(cm[i, j])
            if c > 0:
                pairs.append((c, i, j))
    pairs.sort(reverse=True)
    top_confusions = [{"true": labels[i], "pred": labels[j], "count": c} for c, i, j in pairs[:max_pairs]]

    examples: Dict[str, Any] = {}
    if not pairs:
        return top_confusions, examples

    df_test = df_test.reset_index(drop=True)
    for c, i, j in pairs[:max_pairs]:
        key = f"{labels[i]} → {labels[j]}"
        mask = (y_true_idx == i) & (y_pred_idx == j)
        idxs = np.where(mask)[0][:max_examples_per_pair]
        rows = []
        for k in idxs:
            conf = float(np.max(probs[k])) if probs is not None and len(probs) > k else None
            rows.append(
                {
                    "true": labels[i],
                    "pred": labels[j],
                    "timestamp": str(df_test.iloc[k].get("timestamp", "")),
                    "system": str(df_test.iloc[k].get("system", "")),
                    "source": str(df_test.iloc[k].get("source", "")),
                    "error_code": str(df_test.iloc[k].get("error_code", "")),
                    "confidence": conf,
                    "text_excerpt": _safe_excerpt(str(df_test.iloc[k].get(text_col, ""))),
                }
            )
        examples[key] = rows

    return top_confusions, examples


def write_markdown_report(
    path: Path,
    payload: Dict[str, Any],
    cm: np.ndarray,
    labels: List[str],
    top_confusions: List[Dict[str, Any]],
    examples: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(x):
        return f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x)

    lines = []
    lines.append(f"# Transformer Classification Evaluation — {payload['target']}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Accuracy: **{fmt(payload['accuracy'])}**")
    lines.append(f"- Macro F1: **{fmt(payload['macro_f1'])}**")
    lines.append(f"- Weighted F1: **{fmt(payload['weighted_f1'])}**")
    if payload.get("log_loss") is not None:
        lines.append(f"- Log loss (cross-entropy): **{fmt(payload['log_loss'])}**")
    if payload.get("ece") is not None:
        lines.append(f"- ECE (calibration error): **{fmt(payload['ece'])}**")
    lines.append("")

    lines.append("## Confusion Matrix")
    lines.append("")
    header = "| true \\\\ pred | " + " | ".join(labels) + " |"
    sep = "|" + "---|" * (len(labels) + 1)
    lines.append(header)
    lines.append(sep)
    for i, lab in enumerate(labels):
        row = [str(int(cm[i, j])) for j in range(len(labels))]
        lines.append("| " + lab + " | " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Top Confusions")
    lines.append("")
    if not top_confusions:
        lines.append("- None")
    else:
        for item in top_confusions:
            lines.append(f"- **{item['true']} → {item['pred']}**: {item['count']}")
    lines.append("")

    if examples:
        lines.append("## Example Error Buckets")
        lines.append("")
        for k, rows in examples.items():
            lines.append(f"### {k}")
            for r in rows:
                conf = r.get("confidence", None)
                conf_txt = f" (conf={conf:.2f})" if isinstance(conf, (float, int)) else ""
                lines.append(f"- {r.get('text_excerpt','').strip()}{conf_txt}")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["category", "priority"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--probs-npz", default=None)
    ap.add_argument("--split", default="stratified", choices=["stratified", "time"])
    ap.add_argument("--time-col", default="timestamp")
    ap.add_argument("--gap-days", type=int, default=0)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--labels-json", default=None)
    ap.add_argument("--n-bins", type=int, default=10)
    args = ap.parse_args()

    df = _load_df(args.data)
    cfg = SplitConfig(strategy=args.split, test_size=args.test_size, seed=args.seed, time_col=args.time_col, gap_days=args.gap_days)
    data_cfg = DataConfig(text_col=args.text_col, target_col=args.target)

    if data_cfg.target_col not in df.columns:
        raise ValueError(f"Missing target column: {data_cfg.target_col}")

    if data_cfg.text_col not in df.columns:
        raise ValueError(f"Missing text column: {data_cfg.text_col}")

    if args.labels_json:
        labels = json.loads(Path(args.labels_json).read_text(encoding="utf-8"))
        labels = [str(x) for x in labels]
    else:
        labels = sorted(df[data_cfg.target_col].astype(str).unique().tolist())

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    y_all = df[data_cfg.target_col].astype(str).map(label_to_idx).astype(int)

    if cfg.strategy == "time":
        if cfg.time_col not in df.columns:
            raise ValueError(f"Missing time column '{cfg.time_col}' for time split.")
        ts = _coerce_timestamp(df, cfg.time_col)
        train_idx, test_idx, split_meta = time_aware_split_indices(ts, cfg.test_size, cfg.gap_days)
        split_meta.update({"strategy": "time", "test_size": cfg.test_size, "seed": cfg.seed, "time_col": cfg.time_col})
    else:
        train_idx, test_idx, split_meta = stratified_split_indices(y_all, cfg.test_size, cfg.seed)
        split_meta.update({"strategy": "stratified", "test_size": cfg.test_size, "seed": cfg.seed})

    df_test = df.iloc[test_idx].reset_index(drop=True)
    y_test = y_all.iloc[test_idx].reset_index(drop=True).values
    X_test = df_test[data_cfg.text_col].astype(str).tolist()

    pipe = load_joblib_model(args.model) if args.model else None
    npz = load_probs_npz(args.probs_npz) if args.probs_npz else None
    probs, pred_idx = extract_probs_and_pred(pipe, npz, X_test, labels)

    if len(pred_idx) != len(y_test):
        raise ValueError(f"Prediction length mismatch: y_test={len(y_test)} vs y_pred={len(pred_idx)}")

    acc = float(accuracy_score(y_test, pred_idx))
    macro = float(f1_score(y_test, pred_idx, average="macro"))
    weighted = float(f1_score(y_test, pred_idx, average="weighted"))
    cm = confusion_matrix(y_test, pred_idx, labels=list(range(len(labels))))
    cls_rep = classification_report(y_test, pred_idx, target_names=labels, digits=3)

    top_conf, examples = build_error_buckets(
        df_test=df_test,
        y_true_idx=y_test,
        y_pred_idx=pred_idx,
        probs=probs,
        labels=labels,
        text_col=data_cfg.text_col,
    )

    payload: Dict[str, Any] = {
        "target": args.target,
        "model_path": args.model,
        "split": split_meta,
        "accuracy": acc,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "top_confusions": top_conf,
        "examples": examples,
        "classification_report": cls_rep,
    }

    # logloss + ECE only if we have probs (we always do here)
    payload["log_loss"] = float(log_loss(y_test, probs, labels=list(range(len(labels)))))
    ece, bins = expected_calibration_error(probs, y_test, n_bins=int(args.n_bins))
    payload["ece"] = float(ece)
    payload["calibration_bins"] = bins

    if set(labels) == {"P0", "P1", "P2", "P3"}:
        payload["severe_mistake_rate"] = severe_mistake_rate_priority(y_test, pred_idx, labels)
        idx_p0 = labels.index("P0")
        support_p0 = int(cm[idx_p0, :].sum())
        tp_p0 = int(cm[idx_p0, idx_p0])
        payload["p0_recall"] = float(tp_p0 / support_p0) if support_p0 else float("nan")

    out_art = Path("artifacts") / "transformer_eval"
    out_rep = Path("reports")
    _ensure_dir(out_art)
    _ensure_dir(out_rep)

    json_path = out_art / f"{args.target}.json"
    md_path = out_rep / f"transformer_eval_{args.target}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    write_markdown_report(md_path, payload, cm, labels, top_conf, examples)

    print(f"[OK] Wrote {md_path} and {json_path}")
    print(json.dumps({k: payload[k] for k in ["target", "accuracy", "macro_f1", "weighted_f1", "log_loss", "ece"]}, indent=2))


if __name__ == "__main__":
    main()
