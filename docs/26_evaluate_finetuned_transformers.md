# Evaluate Fine-tuned Transformers | Task Metrics + Failure Modes

## Objective
Production-grade evaluation protocol for fine-tuned transformer models.

Includes:
- Classification metrics
- Calibration analysis
- Severe error tracking
- Acceptance gate
- Reproducibility check

---

## Primary Metrics (Classification)

Mandatory:
- Accuracy
- Macro F1
- Per-class Recall
- Log Loss
- ECE (Expected Calibration Error)

---

## Risk-Aware Metrics

For incident priority tasks (P0–P3):

- P0 Recall ≥ threshold
- Severe error rate (P0→P3) ≤ threshold
- ECE ≤ threshold

---

## Failure Analysis

After each training run:

1. Confusion matrix
2. Top confusion pairs
3. Most confident wrong predictions
4. Low-confidence correct predictions

---

## Acceptance Gate Example

Accuracy ≥ 0.75  
Macro F1 ≥ 0.70  
P0 Recall ≥ 0.90  
Log Loss ≤ 0.90  
ECE ≤ 0.05  
Severe Error Rate ≤ 1%

All must pass for deployment.

---

## Reproducibility

Train with multiple seeds.  
Variance must remain ≤ ±2%.