"""
Loss Curves & Diagnostics for NLP Training

Goal: turn training logs (loss/acc) into a quick, human-readable diagnosis
(healthy vs underfit vs overfit vs suspicious labels).

Supports:
- Keras-style history JSON (keys: loss, val_loss, accuracy/acc, val_accuracy/val_acc)
- Generic JSON or CSV with train_loss/val_loss/train_acc/val_acc columns.
- Synthetic demo generation.

Usage examples:
  python -m src.triage.eval.loss_diagnostics --history reports/history.json --n-classes 4
  python -m src.triage.eval.loss_diagnostics --history reports/history.csv --n-classes 4
  python -m src.triage.eval.loss_diagnostics --synthetic overfit --n-classes 4

Writes:
  reports/loss_diagnostics.md
  artifacts/loss_diagnostics/summary.json
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Parsing / normalization
# ----------------------------
def _to_list(x: Any) -> Optional[List[float]]:
    if x is None:
        return None
    if isinstance(x, list):
        return [float(v) for v in x]
    if isinstance(x, (tuple, np.ndarray, pd.Series)):
        return [float(v) for v in list(x)]
    return None


def _pick(d: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def load_history(path: Path) -> Dict[str, List[float]]:
    """
    Load history from JSON or CSV and normalize keys to:
      train_loss, val_loss, train_acc, val_acc
    """
    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        # allow flexible column names
        cols = {c.lower(): c for c in df.columns}
        def col(*names: str) -> Optional[str]:
            for n in names:
                if n.lower() in cols:
                    return cols[n.lower()]
            return None

        tl = col("train_loss", "loss")
        vl = col("val_loss")
        ta = col("train_acc", "train_accuracy", "acc", "accuracy")
        va = col("val_acc", "val_accuracy")

        out: Dict[str, List[float]] = {}
        if tl: out["train_loss"] = df[tl].astype(float).tolist()
        if vl: out["val_loss"] = df[vl].astype(float).tolist()
        if ta: out["train_acc"] = df[ta].astype(float).tolist()
        if va: out["val_acc"] = df[va].astype(float).tolist()
        return out

    # JSON
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Keras may store: {"loss":[...], "val_loss":[...], "accuracy":[...], "val_accuracy":[...]}
    # or {"history": {"loss":[...] ...}}
    if isinstance(raw, dict) and "history" in raw and isinstance(raw["history"], dict):
        raw = raw["history"]

    if not isinstance(raw, dict):
        raise ValueError("History JSON must be a dict (or have a top-level 'history' dict).")

    tl = _to_list(_pick(raw, "train_loss", "loss"))
    vl = _to_list(_pick(raw, "val_loss"))
    ta = _to_list(_pick(raw, "train_acc", "train_accuracy", "accuracy", "acc"))
    va = _to_list(_pick(raw, "val_acc", "val_accuracy"))

    out2: Dict[str, List[float]] = {}
    if tl is not None: out2["train_loss"] = tl
    if vl is not None: out2["val_loss"] = vl
    if ta is not None: out2["train_acc"] = ta
    if va is not None: out2["val_acc"] = va
    return out2


# ----------------------------
# Diagnostics
# ----------------------------
@dataclass
class CurveStats:
    n_epochs: int
    train_loss_start: Optional[float]
    train_loss_end: Optional[float]
    train_loss_drop_ratio: Optional[float]
    val_loss_min: Optional[float]
    val_loss_end: Optional[float]
    val_loss_rebound_ratio: Optional[float]
    train_acc_end: Optional[float]
    val_acc_end: Optional[float]
    val_acc_mean: Optional[float]


@dataclass
class Diagnosis:
    status: str  # healthy | underfit | overfit | suspicious
    flags: List[str]
    next_actions: List[str]


def _safe_ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if abs(a) < 1e-12:
        return None
    return float((a - b) / max(abs(a), 1e-12))


def compute_stats(h: Dict[str, List[float]]) -> CurveStats:
    n = max(len(h.get("train_loss", [])), len(h.get("val_loss", [])),
            len(h.get("train_acc", [])), len(h.get("val_acc", [])))

    tl = h.get("train_loss")
    vl = h.get("val_loss")
    ta = h.get("train_acc")
    va = h.get("val_acc")

    tl0 = tl[0] if tl else None
    tln = tl[-1] if tl else None
    drop = None
    if tl0 is not None and tln is not None and abs(tl0) > 1e-12:
        drop = float((tl0 - tln) / abs(tl0))

    vmin = min(vl) if vl else None
    vln = vl[-1] if vl else None
    rebound = None
    if vmin is not None and vln is not None and abs(vmin) > 1e-12:
        rebound = float((vln - vmin) / abs(vmin))

    ta_end = ta[-1] if ta else None
    va_end = va[-1] if va else None
    va_mean = float(np.mean(va)) if va else None

    return CurveStats(
        n_epochs=int(n),
        train_loss_start=tl0,
        train_loss_end=tln,
        train_loss_drop_ratio=drop,
        val_loss_min=vmin,
        val_loss_end=vln,
        val_loss_rebound_ratio=rebound,
        train_acc_end=ta_end,
        val_acc_end=va_end,
        val_acc_mean=va_mean,
    )


def diagnose(stats: CurveStats, n_classes: int) -> Diagnosis:
    flags: List[str] = []
    actions: List[str] = []

    chance = 1.0 / max(int(n_classes), 2)

    # Heuristics (simple, explainable)
    not_learning = (stats.train_loss_drop_ratio is not None and stats.train_loss_drop_ratio < 0.10)
    overfit = (stats.val_loss_rebound_ratio is not None and stats.val_loss_rebound_ratio > 0.15)
    random_like = (stats.val_acc_mean is not None and abs(stats.val_acc_mean - chance) < 0.05)

    # Status selection
    status = "healthy"

    if not_learning:
        status = "underfit"
        flags.append("train_loss did not decrease much (<10%) → model may not be learning / pipeline mismatch.")
        actions += [
            "Verify preprocessing/tokenization is identical for train/val.",
            "Check learning rate / optimizer settings.",
            "Increase model capacity or feature strength (TF-IDF vs tiny vocab).",
        ]

    if overfit:
        status = "overfit" if status == "healthy" else status
        flags.append("val_loss rebounded >15% after its minimum while training improved → overfitting signal.")
        actions += [
            "Use early stopping (pick epoch at min val_loss).",
            "Add regularization (dropout/L2), reduce epochs, or simplify model.",
            "Get more diverse training data / augment data.",
        ]

    if random_like:
        # suspicious tends to dominate
        status = "suspicious"
        flags.append(f"val accuracy stayed near chance (~{chance:.2f}) → possible label noise/leakage/preprocess bug.")
        actions += [
            "Run label sanity checks: class distribution, label encoding stability, duplicates.",
            "Validate split strategy (time/group) to prevent leakage.",
            "Inspect a handful of samples per class manually (are labels consistent?).",
        ]

    # If no acc available, still give guidance
    if stats.train_acc_end is None or stats.val_acc_end is None:
        flags.append("Accuracy not provided in history → rely on loss trends only.")
        actions.append("Log both loss and accuracy in training to improve diagnosis quality.")

    # De-duplicate actions
    actions = list(dict.fromkeys(actions))

    if not flags:
        flags.append("No major red flags detected based on heuristic thresholds.")

    return Diagnosis(status=status, flags=flags, next_actions=actions)


# ----------------------------
# Synthetic generator (demo)
# ----------------------------
def synthetic_history(kind: str, n_epochs: int = 12, n_classes: int = 4) -> Dict[str, List[float]]:
    rng = np.random.default_rng(42)
    chance = 1.0 / max(n_classes, 2)

    kind = kind.lower().strip()
    t = np.arange(n_epochs)

    if kind == "healthy":
        train_loss = (1.2 * np.exp(-t / 6.0) + 0.2) + rng.normal(0, 0.02, size=n_epochs)
        val_loss = (1.3 * np.exp(-t / 6.5) + 0.25) + rng.normal(0, 0.03, size=n_epochs)
        train_acc = (0.35 + 0.6 * (1 - np.exp(-t / 6.0))) + rng.normal(0, 0.01, size=n_epochs)
        val_acc = (0.30 + 0.55 * (1 - np.exp(-t / 6.5))) + rng.normal(0, 0.015, size=n_epochs)
    elif kind == "overfit":
        train_loss = (1.2 * np.exp(-t / 5.0) + 0.15) + rng.normal(0, 0.02, size=n_epochs)
        # val improves then degrades
        val_loss = (1.1 * np.exp(-t / 5.5) + 0.25) + 0.08 * np.maximum(t - 6, 0) + rng.normal(0, 0.03, size=n_epochs)
        train_acc = (0.35 + 0.62 * (1 - np.exp(-t / 5.0))) + rng.normal(0, 0.01, size=n_epochs)
        val_acc = (0.30 + 0.42 * (1 - np.exp(-t / 5.5))) - 0.01 * np.maximum(t - 6, 0) + rng.normal(0, 0.015, size=n_epochs)
    elif kind == "underfit":
        train_loss = (0.95 + 0.05 * np.cos(t / 2.0)) + rng.normal(0, 0.02, size=n_epochs)
        val_loss = (1.05 + 0.05 * np.cos(t / 2.0)) + rng.normal(0, 0.02, size=n_epochs)
        train_acc = (chance + 0.05) + rng.normal(0, 0.01, size=n_epochs)
        val_acc = (chance + 0.02) + rng.normal(0, 0.01, size=n_epochs)
    else:  # random / noisy labels
        train_loss = (1.05 + 0.06 * rng.normal(size=n_epochs)).clip(0.8, 1.3)
        val_loss = (1.10 + 0.08 * rng.normal(size=n_epochs)).clip(0.85, 1.4)
        train_acc = (chance + 0.08 * rng.normal(size=n_epochs)).clip(0.0, 1.0)
        val_acc = (chance + 0.05 * rng.normal(size=n_epochs)).clip(0.0, 1.0)

    return {
        "train_loss": [float(x) for x in train_loss],
        "val_loss": [float(x) for x in val_loss],
        "train_acc": [float(x) for x in train_acc],
        "val_acc": [float(x) for x in val_acc],
    }


# ----------------------------
# Reporting
# ----------------------------
def write_report_md(out_path: Path, stats: CurveStats, diag: Diagnosis, history: Dict[str, List[float]], n_classes: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def f(x: Optional[float]) -> str:
        return "n/a" if x is None else f"{x:.4f}"

    lines: List[str] = []
    lines.append(f"# Loss Diagnostics Report")
    lines.append("")
    lines.append(f"- **Status:** `{diag.status}`")
    lines.append(f"- **n_classes:** {n_classes}")
    lines.append(f"- **epochs:** {stats.n_epochs}")
    lines.append("")
    lines.append("## Curve stats")
    lines.append(f"- train_loss: start={f(stats.train_loss_start)} end={f(stats.train_loss_end)} drop_ratio={f(stats.train_loss_drop_ratio)}")
    lines.append(f"- val_loss:   min={f(stats.val_loss_min)} end={f(stats.val_loss_end)} rebound_ratio={f(stats.val_loss_rebound_ratio)}")
    lines.append(f"- train_acc_end={f(stats.train_acc_end)}  val_acc_end={f(stats.val_acc_end)}  val_acc_mean={f(stats.val_acc_mean)}")
    lines.append("")
    lines.append("## Flags")
    for fl in diag.flags:
        lines.append(f"- {fl}")
    lines.append("")
    lines.append("## Next actions")
    for a in diag.next_actions:
        lines.append(f"- {a}")
    lines.append("")
    lines.append("## Raw history (first 5 epochs)")
    lines.append("```json")
    preview = {k: v[:5] for k, v in history.items()}
    lines.append(json.dumps(preview, indent=2))
    lines.append("```")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=str, default=None, help="Path to training history JSON/CSV")
    ap.add_argument("--synthetic", type=str, default=None, help="Generate synthetic history: healthy|overfit|underfit|random")
    ap.add_argument("--n-classes", type=int, default=4, help="Number of classes (for chance baseline)")
    ap.add_argument("--epochs", type=int, default=12, help="Synthetic epochs")
    ap.add_argument("--out-md", type=str, default=None)
    ap.add_argument("--out-json", type=str, default=None)
    args = ap.parse_args()

    if not args.history and not args.synthetic:
        raise SystemExit("Provide --history <path> or --synthetic <kind>.")

    if args.synthetic:
        history = synthetic_history(args.synthetic, n_epochs=int(args.epochs), n_classes=int(args.n_classes))
        src = f"synthetic:{args.synthetic}"
    else:
        history = load_history(Path(args.history))
        src = str(args.history)

    stats = compute_stats(history)
    diag = diagnose(stats, n_classes=int(args.n_classes))

    out_md = Path(args.out_md) if args.out_md else Path("reports/loss_diagnostics.md")
    out_json = Path(args.out_json) if args.out_json else Path("artifacts/loss_diagnostics/summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    write_report_md(out_md, stats, diag, history, n_classes=int(args.n_classes))

    payload = {
        "source": src,
        "n_classes": int(args.n_classes),
        "stats": asdict(stats),
        "diagnosis": asdict(diag),
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Wrote {out_md} and {out_json}")
    print(json.dumps({"status": diag.status, "flags": diag.flags[:2]}, indent=2))


if __name__ == "__main__":
    main()
