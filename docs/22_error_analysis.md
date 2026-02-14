# Error Analysis (Confusion Matrix + Critical False Negatives)

This card is about **making model failures visible** (and debuggable) — not just reporting aggregate F1.

## Why this matters
A single “good F1” can hide operationally dangerous errors:

- **False negatives on high severity (P0/P1)** → incidents that should page on-call may be missed.
- **Systematic confusions** (e.g., *latency* vs *outage*) → routing noise, slower triage, wrong teams paged.

Error analysis turns “model score” into **actionable next steps**:
- what labels are confused,
- which fields drive the errors,
- what to fix in data or features.

## What we implement
We add a small CLI that:

1. Loads a trained model (`joblib` pipeline)
2. Re-creates the same **split strategy** used in training  
   - `stratified` split (train_test_split + stratify)  
   - or **time-aware split** (holdout latest tickets + optional gap days)
3. Produces:
   - confusion matrix
   - top misclassification pairs (true → predicted)
   - example tickets per error type
   - **critical FN list** for priority (true P0/P1 predicted as P2/P3)
4. Writes:
   - `reports/error_analysis_<target>.md`
   - `artifacts/error_analysis/<target>.json`

## How to run

### Category
```bash
python -m src.triage.eval.error_analysis \
  --target category \
  --model artifacts/baseline_category/model.joblib
```

### Priority with time-aware split (recommended)
```bash
python -m src.triage.eval.error_analysis \
  --target priority \
  --model artifacts/baseline_priority/model.joblib \
  --split time --time-col timestamp --gap-days 1
```

### Using alternative models
```bash
python -m src.triage.eval.error_analysis \
  --target priority \
  --model artifacts/alt_priority_svm/model.joblib \
  --split time --time-col timestamp --gap-days 1
```

## Acceptance criteria (Done)
- Confusion matrix is produced for both tasks.
- Top confusions are listed with example tickets.
- Priority report includes a **Critical FN** section (P0/P1 missed).
- Reports are written under `reports/` and JSON artifacts under `artifacts/`.

## Notes
- Because `tickets.csv` is synthetic, scores may look “too good.” Error analysis still matters because it exercises the **workflow and tooling** you'd use in a real system.
