# 18 — Feature Baselines: One‑Hot vs BoW vs Embeddings

This card contrasts **classic text feature representations** and their trade‑offs:

- **One‑hot** (binary presence)
- **Bag‑of‑Words (BoW)** (token counts)
- **TF‑IDF** (weighted BoW)
- **Dense “Embeddings”** via **SVD/LSA** (TF‑IDF → low‑dim dense)
- **EmbeddingBag‑style pooling** (average of hashed token embeddings → dense)

> Goal: make feature choice an **explicit engineering decision** with measurable costs:
> dimensionality, sparsity, memory footprint, speed, and model quality.

---

## Quick intuition

### One‑hot (binary)
- Feature per token: 0/1 (“appears or not”)
- Pros: simplest, surprisingly strong on some tasks
- Cons: **very sparse**, high‑dimensional; ignores frequency

### BoW (counts)
- Like one‑hot but keeps *how many times* tokens occur
- Pros: still fast; counts sometimes add signal
- Cons: still sparse; ignores order/context

### TF‑IDF
- BoW + weights: down‑weight common words, up‑weight informative terms
- Pros: strong default baseline for classification
- Cons: sparse and high‑dim; still orderless

### Dense embeddings (SVD/LSA)
- TF‑IDF compressed into a dense vector (e.g., 256 dims)
- Pros: smaller vectors, faster downstream; can capture latent topics
- Cons: compression can lose rare but important tokens

### EmbeddingBag pooling (hash‑embeddings → average)
- Map tokens to a fixed embedding table (hashed buckets), then average them
- Pros: dense, fixed‑size, cheap; good “DL‑ready” representation for quick experiments
- Cons: hashing collisions; no contextual understanding

---

## What we ship in code

A runnable demo that trains/evaluates multiple feature baselines and writes a **report**:

- `src/triage/text/feature_baselines_demo.py` (CLI runner)
- `src/triage/text/feature_vectorizers.py` (shared builders + EmbeddingBagVectorizer)

Artifacts + reports:
- `artifacts/feature_baselines/<target>/<mode>/model.joblib`
- `artifacts/feature_baselines/<target>/<mode>/meta.json`
- `reports/feature_baselines_<target>.md`
- `reports/feature_baselines_<target>.json`

---

## Repro (commands)

Run all feature modes for a target:

```bash
python -m src.triage.text.feature_baselines_demo --target category --mode all
python -m src.triage.text.feature_baselines_demo --target priority --mode all
```

Run one specific mode:

```bash
python -m src.triage.text.feature_baselines_demo --target category --mode onehot
python -m src.triage.text.feature_baselines_demo --target category --mode bow
python -m src.triage.text.feature_baselines_demo --target category --mode tfidf
python -m src.triage.text.feature_baselines_demo --target category --mode svd --svd-dim 256
python -m src.triage.text.feature_baselines_demo --target category --mode embbag --emb-dim 128 --buckets 50000
```

---

## What to look at in the report

For each feature mode we capture:
- **Quality**: macro F1 / weighted F1 / accuracy
- **Footprint**: feature dim, sparsity (if sparse), approx memory MB
- **Speed**: fit time, predict time

For **priority**, we also report ops‑relevant metrics:
- **P0/P1 binary precision/recall**
- “severe mistake rate” (optional, if you implement)

---

## Decision rules (practical)

- If you want a **fast, strong baseline**: start with **TF‑IDF + Linear model**
- If memory is a concern: try **SVD embeddings** (256/512 dims)
- If you want a “DL‑ready” dense feature quickly: **EmbeddingBag pooling**
- If TF‑IDF is already strong, embeddings often don’t help unless the dataset is noisy/semantic

---

## Done checklist (paste)

- [ ] Implement a demo runner comparing one‑hot vs BoW vs TF‑IDF vs dense embeddings vs EmbeddingBag pooling
- [ ] Emit metrics + footprint stats (dim, sparsity, memory MB, fit/predict time)
- [ ] Save joblib artifacts + meta.json for reproducibility
- [ ] Write a markdown report summarizing results per target (category/priority)
