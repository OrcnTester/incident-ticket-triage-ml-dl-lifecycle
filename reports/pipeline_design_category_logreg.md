# Pipeline Design Demo — category / logreg

## Pipeline structure
- steps: `['tfidf', 'clf']`
- tfidf vocab_size: `994`

## Serialization round-trip
- saved to: `artifacts/pipeline_demo_category_logreg`
- compatibility warnings: `[]`
- predictions identical after reload: **True**

## Metrics (holdout)
```json
{
  "target": "category",
  "model": "logreg",
  "n_rows": 1886,
  "n_train": 1509,
  "n_test": 377,
  "macro_f1": 0.9821248113273989,
  "weighted_f1": 0.9814314065302104,
  "accuracy": 0.9814323607427056
}
```

## Notes
- Vectorizer + model in one Pipeline prevents train/inference mismatch.
- meta.json stores python/sklearn versions to catch persistence issues early.