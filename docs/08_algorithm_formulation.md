# 08 — Algorithm Formulation (Incident Ticket Triage)

This document defines **how we mathematically formulate** the incident ticket triage problem in this repository.
It is intentionally **board-first**: the goal is to make the modeling choices *auditable* and *comparable* across baselines (scikit-learn) and DL tracks (Keras/PyTorch).

---

## 1) Problem Type

### Inputs
Typical ticket signals:
- **Unstructured text**: title + description (primary signal)
- **Light structured metadata** (optional): service/system, source, timestamp, region, error codes, etc.

### Outputs (targets)
We model two supervised targets:

1) **Category**  
- Type: **single-label multiclass classification**
- Output: one category from a fixed taxonomy of *K* classes

2) **Priority (P0–P3)**  
- Type: **single-label multiclass classification**
- Output: one label from {P0, P1, P2, P3}

> Even though priority is ordinal, we start with multiclass classification to keep the pipeline simple and comparable across model families.

---

## 2) Candidate Formulations

We consider three valid formulations. The repository chooses a default for clarity and reproducibility, but keeps alternatives as explicit future experiments.

### Option A — Two independent models (recommended baseline path)
Train two separate predictors:

- Model A: \( f_{cat}(x) \rightarrow y_{cat} \in \{1..K\} \)
- Model B: \( f_{prio}(x) \rightarrow y_{prio} \in \{P0,P1,P2,P3\} \)

**Pros**
- Simple training + debugging (clean separation of concerns)
- Different imbalance/noise levels can be handled per task
- Easier error analysis, faster iteration cycles

**Cons**
- Does not explicitly model correlations (category ↔ priority)

---

### Option B — Multi-output model (shared encoder, 2 heads)
One shared representation \( h(x) \) and two output heads:

- Category head: \( \hat{y}_{cat} = \text{softmax}(W_{cat} h(x)) \)
- Priority head: \( \hat{y}_{prio} = \text{softmax}(W_{prio} h(x)) \)

Loss:
\[
\mathcal{L} = \lambda \cdot CE(y_{cat}, \hat{y}_{cat}) + (1-\lambda) \cdot CE(y_{prio}, \hat{y}_{prio})
\]

**Pros**
- Can exploit shared signal (ticket language often informs both)
- Efficient inference (one pass, two outputs)

**Cons**
- Needs careful loss weighting and per-head evaluation
- Can hide regressions (one head improves while the other degrades)
- More sensitive to label noise in one target contaminating shared features

---

### Option C — Hierarchical / conditional (advanced)
Predict category first, then priority conditioned on category:

- Step 1: \( \hat{y}_{cat} \)
- Step 2: \( \hat{y}_{prio} = g(x, \hat{y}_{cat}) \)

**Pros**
- Mirrors ops workflow (routing and severity often depend on incident type)
- Can simplify priority prediction if categories have stable severity patterns

**Cons**
- Error propagation (wrong category → wrong priority)
- Requires more careful evaluation design

---

## 3) Repository Decision

**Default approach for this repo: Option A (two independent models).**

Rationale:
- Maximizes clarity and comparability for portfolio + lifecycle discussions
- Produces clean, interpretable baseline results quickly
- Keeps the ML/DL tracks modular (each target can evolve independently)

**Planned experiment (later): Option B (multi-output shared encoder)**
- Implemented as a DL experiment once evaluation harness is stable and baseline numbers are established.

---

## 4) Loss Functions (Initial)

### Category (multiclass)
- Output: softmax probabilities over *K* classes
- Loss: **Cross-entropy**

\[
CE(y, \hat{y}) = -\sum_{k=1}^{K} y_k \log(\hat{y}_k)
\]

### Priority (P0–P3)
- Output: softmax over 4 classes
- Loss: **Cross-entropy**

> Cross-entropy is the default because it is aligned to classification likelihood and works consistently across classical and DL models.

---

## 5) Evaluation Metrics (Must Report)

### Category metrics (imbalance-aware)
- **Macro F1** (key metric)
- **Weighted F1**
- Confusion matrix + top confusions
- Top-K accuracy (optional; useful for assistive UX)

### Priority metrics
- **Macro F1**, **Weighted F1**
- **Severe mistake rate** (ops-relevant):
  - Example: predicting **P3** when true is **P0** (and similar large gaps)

Suggested severe-mistake definition (one example):
- Map {P0,P1,P2,P3} → {0,1,2,3}
- Severe if |pred - true| ≥ 2

### Slice / cohort metrics (optional but valuable)
If metadata exists, report metrics by slices:
- service/system, source, region, time-of-day, etc.

---

## 6) Data Splits & Leakage Notes

Minimum requirements:
- Stratified split per label (category and priority)
- Keep a fixed random seed for reproducibility
- Track label distribution in train/val/test

Leakage checks (common incident pitfalls):
- Duplicated tickets across splits
- Near-duplicate text (copy/paste incidents)
- IDs/timestamps that implicitly encode label

---

## 7) Known Risks & Mitigations

### Label noise
Priority is often noisy (human judgement / inconsistent policy).
Mitigations:
- Macro metrics + severe mistake tracking
- Error analysis with examples
- Consider confidence thresholds (assistive UX)

### Class imbalance
Some categories are rare.
Mitigations:
- Macro F1, class weights, or resampling strategies
- Report per-class recall for rare categories

### Overconfidence / hallucination (GenAI notes)
If GenAI is added later, outputs must remain assistive and auditable:
- evidence tagging
- allow “unknown”
- top-k suggestions (no auto-assign)

(See: `docs/07_genai_incident_ops.md`.)

---

## 8) Implementation Pointers (How this maps to the repo)

- Baselines (classical ML): `src/triage/models/`
- Evaluation report: `src/triage/eval/report.py`
- GenAI contracts (assistive outputs): `src/triage/genai/`

This doc is the **source of truth** for problem formulation and will be referenced by:
- baseline training scripts
- DL training tracks
- evaluation harness and reports
