from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib

def safe_predict(pipe, X_df: pd.DataFrame):
    """Predict with either DataFrame-based pipelines (ColumnTransformer) or Series-based text pipelines.

    Some pipelines are trained with X=df['text'] (a 1D Series). If we pass a DataFrame,
    sklearn's vectorizers may iterate over column names and produce a single prediction.
    This helper tries DataFrame first, then falls back to Series when needed.
    """
    n = len(X_df)
    try:
        pred = pipe.predict(X_df)
        try:
            if len(pred) == n:
                return pred
        except TypeError:
            pass
    except Exception:
        pass
    # Fallback: 1D text input
    return pipe.predict(X_df['text'].astype(str))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

PRIORITY_ORDER = ["P0", "P1", "P2", "P3"]
PRIORITY_RANK = {p: i for i, p in enumerate(PRIORITY_ORDER)}


@dataclass
class SplitMeta:
    strategy: str
    test_size: float
    seed: int
    gap_days: int
    time_col: str
    n_train: int
    n_test: int
    train_min_ts: Optional[str] = None
    train_max_ts: Optional[str] = None
    test_min_ts: Optional[str] = None
    test_max_ts: Optional[str] = None


def load_meta_from_artifact(model_path: Path) -> Dict[str, Any]:
    # Best-effort: read meta.json or meta.joblib next to model.joblib.
    parent = model_path.parent
    meta_json = parent / "meta.json"
    meta_joblib = parent / "meta.joblib"

    if meta_json.exists():
        return json.loads(meta_json.read_text(encoding="utf-8"))
    if meta_joblib.exists():
        obj = joblib.load(meta_joblib)
        return obj if isinstance(obj, dict) else {"meta": obj}
    return {}


def stratified_split_indices(y: pd.Series, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y
    )
    return np.array(train_idx), np.array(test_idx)


def time_aware_split_indices(
    ts: pd.Series, test_size: float, gap_days: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Optional[str]]]:
    # Hold out latest test_size portion as test; exclude a time "gap" from training.
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"time_aware_split: {bad} rows have invalid timestamps")

    order = np.argsort(ts.to_numpy())
    n = len(ts)
    n_test = max(1, int(round(n * float(test_size))))
    test_idx = order[-n_test:]

    test_min = ts.iloc[test_idx].min()
    test_max = ts.iloc[test_idx].max()

    cutoff = test_min - pd.Timedelta(days=int(gap_days))
    train_mask = ts < cutoff
    train_idx = np.where(train_mask.to_numpy())[0]

    if len(train_idx) == 0:
        raise ValueError(
            "time_aware_split produced empty train set. "
            "Reduce gap_days or test_size, or check timestamp distribution."
        )

    meta = {
        "train_min_ts": ts.iloc[train_idx].min().isoformat() if len(train_idx) else None,
        "train_max_ts": ts.iloc[train_idx].max().isoformat() if len(train_idx) else None,
        "test_min_ts": test_min.isoformat(),
        "test_max_ts": test_max.isoformat(),
    }
    return np.array(train_idx), np.array(test_idx), meta


def safe_excerpt(text: str, n: int = 160) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= n else s[: n - 3] + "..."


def top_confusions(y_true: List[str], y_pred: List[str], top_k: int = 8) -> List[Dict[str, Any]]:
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    df = df[df["true"] != df["pred"]].copy()
    if df.empty:
        return []
    pairs = (
        df.value_counts(["true", "pred"])
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_k)
    )
    return pairs.to_dict(orient="records")


def priority_severe_flags(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    # Severe = true P0/P1 but predicted P2/P3.
    t = y_true.map(PRIORITY_RANK)
    p = y_pred.map(PRIORITY_RANK)
    return (t <= 1) & (p >= 2)


def write_markdown_report(
    *,
    out_path: Path,
    target: str,
    model_path: Path,
    split_meta: SplitMeta,
    labels: List[str],
    cm: np.ndarray,
    macro_f1: float,
    weighted_f1: float,
    acc: float,
    top_confusions_rows: List[Dict[str, Any]],
    examples: Dict[str, List[Dict[str, Any]]],
    extra: Dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def cm_table() -> str:
        header = "|true\\pred|" + "|".join(labels) + "|\n"
        sep = "|" + "|".join(["---"] * (len(labels) + 1)) + "|\n"
        rows = []
        for i, t in enumerate(labels):
            rows.append("|" + t + "|" + "|".join(str(int(x)) for x in cm[i]) + "|\n")
        return header + sep + "".join(rows)

    lines: List[str] = []
    lines.append(f"# Error Analysis — {target}\n")
    lines.append(f"- Model: `{model_path.as_posix()}`")
    lines.append(
        f"- Split: `{split_meta.strategy}` | test_size={split_meta.test_size} | seed={split_meta.seed} | gap_days={split_meta.gap_days}"
    )
    if split_meta.strategy == "time":
        lines.append(f"- Train window: {split_meta.train_min_ts} → {split_meta.train_max_ts}")
        lines.append(f"- Test window:  {split_meta.test_min_ts} → {split_meta.test_max_ts}")
    lines.append("")
    lines.append("## Metrics (holdout)")
    lines.append(f"- Macro F1: **{macro_f1:.4f}**")
    lines.append(f"- Weighted F1: **{weighted_f1:.4f}**")
    lines.append(f"- Accuracy: **{acc:.4f}**")
    for k, v in extra.items():
        lines.append(f"- {k}: **{v}**")
    lines.append("")
    lines.append("## Confusion matrix")
    lines.append(cm_table())
    lines.append("")
    lines.append("## Top confusions (true → predicted)")
    if not top_confusions_rows:
        lines.append("- No confusions (perfect on this split).")
    else:
        for row in top_confusions_rows:
            lines.append(f"- **{row['true']} → {row['pred']}** (count={row['count']})")
    lines.append("")

    for section, rows in examples.items():
        lines.append(f"## Examples — {section}")
        if not rows:
            lines.append("- (none)\n")
            continue
        for r in rows:
            lines.append(
                f"- **true={r['true']} pred={r['pred']}** | ts={r.get('timestamp')} | system={r.get('system')} | source={r.get('source')} | err={r.get('error_code')}"
            )
            lines.append(f"  - text: `{r['text_excerpt']}`")
        lines.append("")

    lines.append("## Notes / Next steps")
    lines.append("- If *Critical FN* exists, treat as a **blocking** issue for autonomous escalation decisions.")
    lines.append("- Use these examples to refine synthetic generation rules or add new features (error_code/system/source).")
    lines.append("- When real data exists: add human review tags and build a labeled “hard cases” set.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/tickets.csv")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--target", type=str, choices=["priority", "category"], required=True)

    ap.add_argument("--split", type=str, choices=["stratified", "time"], default=None)
    ap.add_argument("--test-size", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--time-col", type=str, default="timestamp")
    ap.add_argument("--gap-days", type=int, default=0)

    ap.add_argument("--top-n", type=int, default=6)
    ap.add_argument("--out-md", type=str, default=None)
    ap.add_argument("--out-json", type=str, default=None)
    args = ap.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model)

    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not in columns: {list(df.columns)}")

    meta = load_meta_from_artifact(model_path)

    split_strategy = args.split or meta.get("split_strategy") or "stratified"
    test_size = float(args.test_size if args.test_size is not None else meta.get("test_size", 0.2))
    seed = int(args.seed if args.seed is not None else meta.get("seed", 42))
    gap_days = int(args.gap_days)

    y = df[args.target].astype(str)

    if split_strategy == "time":
        if args.time_col not in df.columns:
            raise ValueError(f"time-col '{args.time_col}' not found in columns: {list(df.columns)}")
        train_idx, test_idx, time_meta = time_aware_split_indices(
            df[args.time_col], test_size=test_size, gap_days=gap_days
        )
    else:
        train_idx, test_idx = stratified_split_indices(y, test_size=test_size, seed=seed)
        time_meta = {}

    pipe = joblib.load(model_path)

    X_test = df.iloc[test_idx][["text"]].copy()
    y_test = y.iloc[test_idx].copy()
    y_pred_arr = safe_predict(pipe, X_test)
    y_pred = pd.Series(y_pred_arr, index=y_test.index).astype(str)

    labels = PRIORITY_ORDER if args.target == "priority" else sorted(y.unique().tolist())

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    acc = accuracy_score(y_test, y_pred)

    conf = top_confusions(y_test.tolist(), y_pred.tolist(), top_k=10)

    test_df = df.iloc[test_idx].copy()
    test_df["true"] = y_test.values
    test_df["pred"] = y_pred.values

    examples: Dict[str, List[Dict[str, Any]]] = {}

    # Top confusion examples
    for row in conf[:3]:
        t, p = row["true"], row["pred"]
        rows = test_df[(test_df["true"] == t) & (test_df["pred"] == p)].head(args.top_n)
        examples[f"{t} → {p}"] = [
            {
                "true": r["true"],
                "pred": r["pred"],
                "timestamp": r.get("timestamp"),
                "system": r.get("system"),
                "source": r.get("source"),
                "error_code": r.get("error_code"),
                "text_excerpt": safe_excerpt(r.get("text", "")),
            }
            for _, r in rows.iterrows()
        ]

    extra: Dict[str, Any] = {}
    if args.target == "priority":
        severe = priority_severe_flags(y_test, y_pred)
        severe_rate = float(severe.mean())
        extra["severe_mistake_rate(P0/P1→P2/P3)"] = f"{severe_rate:.4f}"

        severe_rows = test_df[severe].head(args.top_n)
        examples["Critical FN (true P0/P1 predicted P2/P3)"] = [
            {
                "true": r["true"],
                "pred": r["pred"],
                "timestamp": r.get("timestamp"),
                "system": r.get("system"),
                "source": r.get("source"),
                "error_code": r.get("error_code"),
                "text_excerpt": safe_excerpt(r.get("text", "")),
            }
            for _, r in severe_rows.iterrows()
        ]

    out_md = Path(args.out_md) if args.out_md else Path(f"reports/error_analysis_{args.target}.md")
    out_json = Path(args.out_json) if args.out_json else Path(f"artifacts/error_analysis/{args.target}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    split_meta = SplitMeta(
        strategy=split_strategy,
        test_size=test_size,
        seed=seed,
        gap_days=gap_days,
        time_col=args.time_col,
        n_train=int(len(train_idx)),
        n_test=int(len(test_idx)),
        train_min_ts=time_meta.get("train_min_ts"),
        train_max_ts=time_meta.get("train_max_ts"),
        test_min_ts=time_meta.get("test_min_ts"),
        test_max_ts=time_meta.get("test_max_ts"),
    )

    write_markdown_report(
        out_path=out_md,
        target=args.target,
        model_path=model_path,
        split_meta=split_meta,
        labels=labels,
        cm=cm,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        acc=acc,
        top_confusions_rows=conf,
        examples=examples,
        extra=extra,
    )

    payload = {
        "target": args.target,
        "model_path": str(model_path),
        "split": asdict(split_meta),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": acc,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "top_confusions": conf,
        "examples": examples,
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Wrote {out_md} and {out_json}")


if __name__ == "__main__":
    main()
