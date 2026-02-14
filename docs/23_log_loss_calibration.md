# Log Loss (Cross Entropy) as a Metric + Calibration Note

In incident triage, **accuracy alone can be misleading**.

- Accuracy asks: *“Did we predict the right class?”*
- **Log loss (cross entropy)** asks: *“Did we assign **good probabilities**?”*

That second question matters when a wrong high‑confidence prediction (e.g., “this is **P0**”) can trigger **expensive** pages/escalations.

---

## Why log loss matters beyond accuracy (2–3 lines)

**Log loss measures probability quality.**  
If the model is **confident but wrong**, log loss gets **punished hard**.  
If the model is correct but **honestly uncertain**, log loss is more forgiving.

**Phrase to remember:**  
> **confident but wrong** → **high log loss**

---

## Triage risk connection (why you should care)

A classifier can have decent accuracy yet still be dangerous:

- It may **over‑predict P0/P1** with high confidence → false alarms, pager fatigue
- Or **under‑predict** critical incidents with low confidence → slow response

So we want:
- correct labels **and**
- probabilities that reflect reality (**calibration**)

---

## What “calibration” means

A calibrated model’s probability matches observed frequency.

Example:
- If the model outputs **0.80 confidence**, then ~**80%** of those cases should be correct.

If not:
- Overconfident model: outputs 0.90, reality 0.60 → risky
- Underconfident model: outputs 0.55, reality 0.85 → you can’t trust confidence for routing/automation

---

## What we implement in this card (repo artifacts)

We add an evaluation tool that:

1. Computes **multiclass log loss** (category/priority)
2. Highlights **top “confident but wrong”** examples (the scary ones)
3. Computes **ECE (Expected Calibration Error)** as a quick calibration summary
4. For priority, also computes **P0/P1 binary log loss** using:
   - `p(P0/P1) = p(P0) + p(P1)`

Outputs:
- `reports/logloss_calibration_<target>.md`
- `artifacts/calibration/<target>.json`

---

## Optional: post-hoc calibration (safe improvement)

If your model has probabilities but they’re poorly calibrated, you can apply:
- **Sigmoid (Platt scaling)**: stable default
- **Isotonic**: more flexible, needs more data

We support:
- `--calibrate sigmoid`
- `--calibrate isotonic`

This does **not** change the underlying classifier features; it only adjusts the probability mapping.

---

## Acceptance criteria (done checklist)

- ✅ Explain why log loss matters beyond accuracy (**probability quality**)
- ✅ Include phrase: **confident but wrong → high log loss**
- ✅ Connect to triage risk: wrong high‑confidence P0 prediction is costly
- ✅ Provide measurable outputs: log loss + ECE + top overconfident errors

---

## How to run

Example (time split, triage-safe default):

```bash
# macOS/Linux (bash/zsh) — multiline
python -m src.triage.eval.logloss_calibration \
  --target priority \
  --model artifacts/baseline_priority/model.joblib \
  --data data/tickets.csv \
  --split time --time-col timestamp --gap-days 1
```

```powershell
# Windows PowerShell — single line (recommended)
python -m src.triage.eval.logloss_calibration --target priority --model artifacts/baseline_priority/model.joblib --data data/tickets.csv --split time --time-col timestamp --gap-days 1

# Windows PowerShell — multiline (use backtick)
python -m src.triage.eval.logloss_calibration `
  --target priority `
  --model artifacts/baseline_priority/model.joblib `
  --data data/tickets.csv `
  --split time --time-col timestamp --gap-days 1
```

Optional calibration:

```bash
# macOS/Linux (bash/zsh) — multiline
python -m src.triage.eval.logloss_calibration \
  --target priority \
  --model artifacts/baseline_priority/model.joblib \
  --data data/tickets.csv \
  --split time --time-col timestamp --gap-days 1 \
  --calibrate sigmoid
```

```powershell
# Windows PowerShell — single line (recommended)
python -m src.triage.eval.logloss_calibration --target priority --model artifacts/baseline_priority/model.joblib --data data/tickets.csv --split time --time-col timestamp --gap-days 1 --calibrate sigmoid
```
## CLI notes (Windows / PowerShell)

- If you want multi-line commands in PowerShell, use **backtick** (`` ` ``) at end of line — not the Linux backslash (`\`).
- This script expects the **raw text column as a list/Series of strings**, not a 1-column DataFrame.
  - Reason: a `TfidfVectorizer` pipeline treats a DataFrame as an iterable of **column names**, which would look like “1 sample” and break metrics.

## Calibration note (scikit-learn >= 1.8)

Recent scikit-learn versions no longer support `cv="prefit"` for `CalibratedClassifierCV`.
This demo uses `cv=3` on a held-out calibration subset, which **refits** the estimator internally while learning calibration.
That’s acceptable for an evaluation-only demo; in production you’d persist a calibrated model artifact explicitly.
