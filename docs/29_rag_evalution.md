# RAG Evaluation | Relevance + Groundedness + Citation Checks

## Goal
Evaluate RAG beyond answer quality:
- Retrieval relevance
- Groundedness (is answer supported by retrieved context?)
- Citation presence & correctness
- Safe fallback when evidence is missing

---

## 1. Retrieval Metrics

### Core
- Recall@k
- MRR (optional)
- Simple hit-rate (gold doc in top-k)

Definition:
Recall@k = (# queries where gold_doc ∈ top_k) / total_queries

Acceptance:
- Recall@5 ≥ 0.7 (baseline target)

---

## 2. Generation Checks

### Groundedness
Answer statements must appear in retrieved context.

Metric:
Groundedness Rate =
(# answers fully supported by context) / total_answers

---

### Citation Presence
Answer must include citation marker (e.g. [doc_id]).

Metric:
Citation Rate =
(# answers containing citation) / total_answers

---

### Hallucination Rate
Answer contains claims not present in retrieved context.

Metric:
Hallucination Rate =
(# unsupported answers) / total_answers

Acceptance:
- Groundedness ≥ 0.8
- Citation Rate ≥ 0.9
- Hallucination Rate ≤ 0.2

---

## 3. Gold Set Plan

Create 10–30 queries with:
- Known correct document id(s)
- Expected source doc

Example entry:

{
  "query": "What to do when API 5xx spikes?",
  "gold_doc": "runbook_api_5xx"
}

---

## 4. Safe Fallback Check

If:
- Retrieval returns low similarity
- OR no relevant context

Model must respond with:
"I don’t have enough evidence to answer."

Metric:
Safe Fallback Rate =
(# correct refusals when no gold doc exists) / total_no_evidence_queries

Acceptance:
≥ 0.9
