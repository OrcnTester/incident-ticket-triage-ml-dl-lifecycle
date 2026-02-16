# Chunking Quality Evaluation | Responsiveness vs Accuracy

This document defines how to compare chunking strategies using both
efficiency (speed / token cost) and quality (retrieval + groundedness).

---

## 1. Goal

Evaluate whether a chunking configuration improves:

- Retrieval relevance
- Answer groundedness
- Latency proxy
- Token usage efficiency

We compare multiple chunk configs (e.g., 100/20 vs 200/40).

---

## 2. Efficiency Metrics (Responsiveness)

| Metric | Meaning |
|--------|--------|
| avg_chunk_tokens | Average chunk length |
| tokens_per_prompt | Retrieved tokens per query |
| latency_proxy | tokens_per_prompt (approx compute cost) |

Lower is better.

---

## 3. Quality Metrics

| Metric | Meaning |
|--------|--------|
| recall@k | Gold doc retrieved? |
| groundedness_rate | Answer supported by retrieved text |
| citation_rate | Citation present in answer |

Higher is better.

---

## 4. Test Set

Small fixed evaluation set (5–15 queries):

- Each query has gold_doc
- Each query has retrieved_docs
- Each query has answer

Stored in:
artifacts/chunking_eval_input.json

---

## 5. Acceptance Criteria

Chosen config must:

- Not degrade recall@k
- Not degrade groundedness_rate
- Reduce tokens_per_prompt OR improve quality

If trade-off exists:
Prefer config with higher groundedness.

---

## 6. Decision Rule

Winner score =

quality_score - 0.001 * tokens_per_prompt

Where:

quality_score = 0.5 * recall@k + 0.5 * groundedness_rate
