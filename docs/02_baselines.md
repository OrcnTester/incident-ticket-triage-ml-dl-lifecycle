# Modeling (Classical ML Baselines)

## Baseline Model Selection

### Description
We start with a strong, interpretable baseline to set a reference point before deep learning.

**Baseline choice**
- **Vectorizer:** TF-IDF (word or character n-grams, tuned later)
- **Model:** Logistic Regression (multinomial for multi-class)

**Why this baseline**
- Fast iteration and clear decision boundaries
- Reasonable performance on sparse text features
- Easy to debug (feature weights, error buckets)

### Deliverable
- This doc section + linked project card

### Done checklist
- [ ] Baseline objective stated (reference performance, not “final model”)
- [ ] TF-IDF + Logistic Regression justified (interpretability + speed)
- [ ] Note added: what will be compared later (DL / transformer / RAG)
- [ ] At least 1 risk/trade-off captured below

### Risks / Notes
- **Risk:** Sparse features can miss semantics (synonyms/paraphrases).
- **Trade-off:** Interpretability vs representational power.
- **Assumption:** Baseline is sufficient to validate labeling and evaluation pipeline early.
