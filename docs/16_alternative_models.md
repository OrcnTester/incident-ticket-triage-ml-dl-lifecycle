# 16 — Alternative Models (Classical ML)

This card adds **strong, fast alternative classical baselines** for incident ticket text classification.

We already have a Logistic Regression TF‑IDF baseline. Here we add three commonly-used alternatives:

- **Multinomial Naive Bayes (MNB)** — very fast, surprisingly competitive on bag-of-words; great “speed baseline”.
- **Linear SVM (LinearSVC)** — often top performer for TF‑IDF text classification; strong margin-based classifier.
- **Random Forest (RF)** — can capture non-linearities, but risks overfitting and usually needs dimensionality reduction.

---

## Why do this?

In real ops/enterprise ML, “one baseline” is not enough.  
You want:
- **a speed baseline** (MNB) for quick iteration,
- **a strong linear baseline** (LinearSVC) for accuracy,
- **a non-linear check** (RF) to test if there’s signal beyond linear separability.

If your “fancy model” can’t beat LinearSVC on the same split, it’s a red flag 🚩.

---

## Implementation design

All models share:
- **TF‑IDF** (1–2 ngrams) vectorizer
- consistent training CLI and artifact outputs
- saved `split.json` so evaluation is reproducible

RF uses:
- **TruncatedSVD** to reduce TF‑IDF to dense vectors before tree training
  (trees are not a natural fit for huge sparse TF‑IDF matrices)

---

## Commands (train)

### Category
```bash
python -m src.triage.models.train_alternatives --target category --model nb
python -m src.triage.models.train_alternatives --target category --model svm
python -m src.triage.models.train_alternatives --target category --model rf --svd-dim 256 --rf-estimators 300
```

### Priority
```bash
python -m src.triage.models.train_alternatives --target priority --model nb
python -m src.triage.models.train_alternatives --target priority --model svm
python -m src.triage.models.train_alternatives --target priority --model rf --svd-dim 256 --rf-estimators 300
```

### Optional: time-aware split
```bash
python -m src.triage.models.train_alternatives --target priority --model svm --split time --time-col timestamp --gap-days 1
```

---

## Commands (compare report)

```bash
python -m src.triage.eval.alt_models_report
```

Outputs:
- `reports/alt_models_report.md`
- `reports/alt_models_metrics.json`

---

## Artifacts layout

Each trained model is saved under:
- `artifacts/alt_<target>_<model>/model.joblib`
- `artifacts/alt_<target>_<model>/split.json`
- `artifacts/alt_<target>_<model>/metrics.json`
- `artifacts/alt_<target>_<model>/meta.json`

Example:
- `artifacts/alt_category_svm/model.joblib`

---

## Done checklist (paste)

- [ ] Add MNB baseline (speed-oriented)
- [ ] Add Linear SVM baseline (strong text performance)
- [ ] Add RF variant (non-linear check) using SVD to handle sparse TF‑IDF
- [ ] Persist split indices + metrics for reproducibility
- [ ] Add a comparison report (markdown + json)

