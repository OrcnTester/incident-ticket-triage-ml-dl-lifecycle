# Transformer Classification Evaluation | Beyond Accuracy

This card is about **evaluation discipline**: if you only track `accuracy`, you will miss the failures that hurt ops most (e.g., **confident-but-wrong P0/P1**).  
So here we evaluate classification outputs the same way you would evaluate a production incident triage assistant: **per-class**, **confusion-aware**, and **calibration-aware**.

---

## Why “beyond accuracy” matters (triage framing)

Accuracy is a *single scalar*. It hides:
- **Class imbalance** (a model can be “accurate” by always predicting the majority class)
- **Asymmetric cost** (a wrong high-confidence **P0** prediction is far more expensive than a wrong P3)
- **Failure modes** (the same accuracy can come from very different confusion patterns)

So we add:
- **Macro F1** (treat classes equally; good for imbalance)
- **Per-class precision/recall/F1** (who is getting harmed?)
- **Confusion matrix + top confusions** (which mistakes happen most?)
- **Error buckets with examples** (what does it *look* like in text?)
- **Calibration metrics** (probability quality): **log loss** and **ECE**

---

## What’s implemented in this repo

### 1) `src/triage/eval/transformer_eval.py`
A model-agnostic evaluator that can work with:
- `--model <joblib pipeline>` (scikit pipeline that supports `predict_proba` or `decision_function`)
- `--probs-npz <npz>` (Transformer-like dump with `probs` or `logits` arrays)

**Outputs**
- `reports/transformer_eval_<target>.md`
- `artifacts/transformer_eval/<target>.json` (metrics, confusion matrix, top confusions, example errors)

✅ Includes calibration summary (**log_loss**, **ECE**) when probabilities are available.

> **Important bug fix**  
> The evaluator must feed **1D text input** into sklearn text pipelines.  
> If you pass a full DataFrame into `CountVectorizer/TfidfVectorizer`, sklearn iterates columns and you get tiny prediction lengths (like 8).  
> The updated script passes `df_test[text_col]` instead.

### 2) `src/triage/eval/transformer_eval_demo.py`
Creates a **demo NPZ that matches your test split size**, so you don’t get length mismatches.

---

## How to run

### A) Evaluate a joblib baseline model (recommended default)
**Priority (time split):**
```powershell
python -m src.triage.eval.transformer_eval `
  --target priority `
  --model artifacts/baseline_priority/model.joblib `
  --data data/tickets.csv `
  --split time --time-col timestamp --gap-days 1
```

**Category (time split):**
```powershell
python -m src.triage.eval.transformer_eval `
  --target category `
  --model artifacts/baseline_category/model.joblib `
  --data data/tickets.csv `
  --split time --time-col timestamp --gap-days 1
```

### B) Evaluate Transformer-style probabilities (NPZ)
**Step 1 — generate demo probs aligned to your split**
```powershell
python -m src.triage.eval.transformer_eval_demo `
  --target priority `
  --data data/tickets.csv `
  --split time --time-col timestamp --gap-days 1 `
  --out artifacts/transformer_probs_priority_demo.npz
```

**Step 2 — evaluate those probs**
```powershell
python -m src.triage.eval.transformer_eval `
  --target priority `
  --probs-npz artifacts/transformer_probs_priority_demo.npz `
  --data data/tickets.csv `
  --split time --time-col timestamp --gap-days 1
```

---

## Calibration note (confidence vs correctness)

Two models can have the same accuracy but very different probability quality:

- **Low log loss + low ECE** → confidence matches reality → safer for automation thresholds
- **High log loss / high ECE** → “confident but wrong” → dangerous for P0/P1

In incident triage, **wrong + confident** is what pages humans at 3am for no reason.

---

## Acceptance criteria (ops-aligned)

A practical acceptance checklist:
- Macro F1 ≥ your baseline, and **no class collapse** (per-class recall not ~0)
- Confusion matrix shows **no frequent P0↔P3** catastrophic swaps
- Calibration improves with sigmoid/isotonic (log loss down, ECE down), or you keep “unknown” fallback for low confidence
- Top confusion examples are understandable and suggest clear data/label fixes (not random noise)

---

## Next steps (optional, but clean)
- Add “cost matrix” (P0 mistakes weighted heavier)
- Add thresholding policy: auto-route only when confidence ≥ τ, else “unknown”
- Add drift hooks: compare confusion patterns over time windows
