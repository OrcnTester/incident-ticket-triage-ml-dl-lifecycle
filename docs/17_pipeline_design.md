# 17 — Pipeline Design (scikit-learn)

This card formalizes the **classical ML pipeline pattern** used in this repo:

> **vectorizer → classifier** (single `sklearn.pipeline.Pipeline` object)

…and shows how we serialize/restore that exact pipeline safely using **joblib + metadata**.

---

## Why a Pipeline?

Without a pipeline, it’s easy to accidentally:
- fit a vectorizer on train, then reuse a different vectorizer at inference
- change preprocessing between experiments
- forget which hyperparameters produced a given artifact

A `Pipeline` makes the *entire* feature + model chain one object:
- `fit(X_train, y_train)`
- `predict(X_test)`
- `joblib.dump(pipeline, "model.joblib")`

That means inference becomes trivial:
```python
pipe = joblib.load("model.joblib")
pred = pipe.predict(["ticket text here"])
```

---

## Standard design in this repo

### Shared step names
- `tfidf` — text vectorizer
- `svd` — optional dimensionality reduction (only for tree models)
- `clf` — classifier

### Default vectorizer
- `TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=200_000)`

### Supported classical models
- `logreg` — LogisticRegression (balanced)
- `nb` — MultinomialNB (speed baseline)
- `svm` — LinearSVC (strong baseline)
- `rf` — RandomForestClassifier with `TruncatedSVD` before the forest

---

## Serialization (joblib) + “defense-grade” metadata

`joblib` stores fitted sklearn objects efficiently, **but you must track versions**.

So we save:
- `model.joblib` — the fitted pipeline
- `meta.json` — python + sklearn + platform versions and key hyperparams

At load time, we can warn if:
- sklearn version differs
- python version differs

This avoids “it works on my machine” surprises and helps audit artifacts later.

---

## Demo CLI

### Train + save + reload (round-trip check)
```bash
python -m src.triage.models.pipeline_demo --target category --model logreg
python -m src.triage.models.pipeline_demo --target priority --model svm
python -m src.triage.models.pipeline_demo --target category --model rf --svd-dim 256 --rf-estimators 300
```

Outputs:
- `artifacts/pipeline_demo_<target>_<model>/model.joblib`
- `artifacts/pipeline_demo_<target>_<model>/meta.json`
- `reports/pipeline_design_<target>_<model>.md`

---

## Done checklist (paste)

- [ ] Document the standard sklearn pipeline pattern: vectorizer → classifier
- [ ] Add shared pipeline builder to avoid duplicated definitions
- [ ] Add joblib save/load helpers with version metadata
- [ ] Add a runnable demo that trains, saves, reloads, and validates round-trip behavior
