# Pipeline Design Demo — priority / svm

## Pipeline structure
- steps: `['tfidf', 'clf']`
- tfidf vocab_size: `993`

## Serialization round-trip
- saved to: `artifacts/pipeline_demo_priority_svm`
- compatibility warnings: `[]`
- predictions identical after reload: **True**

## Metrics (holdout)
```json
{
  "target": "priority",
  "model": "svm",
  "n_rows": 1886,
  "n_train": 1508,
  "n_test": 378,
  "macro_f1": 0.5872942510222394,
  "weighted_f1": 0.6140300117352518,
  "accuracy": 0.6243386243386243
}
```

## Notes
- Vectorizer + model in one Pipeline prevents train/inference mismatch.
- meta.json stores python/sklearn versions to catch persistence issues early.