# 10 — Class Imbalance Considerations (Incident Priority)

This document captures **why class imbalance is expected** in incident datasets and how this repository evaluates and mitigates it—especially for **priority prediction (P0–P3)**.

The goal is not to “win a Kaggle score.”  
The goal is to build an **ops-usable** assistant: prioritize **high-severity recall** while managing false alarms.

---

## 1) Why imbalance is normal in incident ops

In real incident operations:
- **P0/P1** incidents are intentionally rare
- **P2/P3** tickets dominate day-to-day volume
- Priority labels can be **noisy** (human judgment, inconsistent policy, missing context)

Therefore, a realistic dataset should reflect:
- long-tail class frequency
- label noise (especially on borderline severities)
- evolving policies (priority definitions can drift)

---

## 2) Implications for modeling

### Accuracy is misleading
A model can achieve “good” accuracy by mostly predicting the majority classes (P2/P3), while still failing at what matters:
- missing **P0** events
- under-triaging high-impact incidents

### We optimize for ops risk
We treat misclassification costs as asymmetric:
- **False negative on P0** is far more costly than false positive on P0
- Large severity gaps matter most (e.g., predicting P3 when true is P0)

---

## 3) Evaluation: metrics we must report

This repo reports both standard and ops-specific metrics.

### Standard classification metrics
- **Macro F1** (imbalance-aware; primary)
- Weighted F1
- Confusion matrix

### Ops-specific metrics (priority)
- **P0 recall**: how often true P0 incidents are caught
- **P0/P1 binary recall + precision**
  - Positive class: {P0, P1}
  - This stress-tests “high severity detection”
- **Severe mistake rate**
  - Map {P0,P1,P2,P3} → {0,1,2,3}
  - Severe if |pred − true| ≥ 2  
  - Example: predicting P3 when true is P0

### Why these metrics
- P0 recall protects against catastrophic misses
- P0/P1 binary metrics expose the “high-severity gate” quality
- Severe mistake rate captures ops pain better than average loss

---

## 4) Mitigation strategies (baseline → advanced)

### Baseline (implemented)
- Use **class_weight="balanced"** for priority classifier
- Stratified splits by label distribution
- Report per-class metrics and ops metrics above

### Next options (if needed)
- Thresholding / top-k assistive outputs (don’t force a single label)
- Cost-sensitive learning (explicit misclassification costs)
- Resampling strategies:
  - oversample minority (careful with leakage)
  - undersample majority (may lose signal)
- Incorporate metadata (service/system/source) if available
- Multi-output shared encoder (DL track) with careful loss weighting

---

## 5) Operational boundaries (risk-aware behavior)

Even with metrics, priority predictions remain probabilistic.
In ops settings:
- allow **“unknown”** when evidence is weak
- surface **confidence** and **evidence**
- never auto-assign or auto-escalate without human review

This is aligned with the repo’s “assistive, not authoritative” principle.

---

## 6) Where to find evidence in this repo

- Baseline training:
  - `src/triage/models/train_baseline.py` (priority uses class weights)
- Evaluation outputs:
  - `reports/report.txt`
  - `reports/metrics.json`
- Baseline artifacts:
  - `artifacts/baseline_priority/`

This doc should be updated whenever the evaluation harness changes or new mitigation strategies are added.
