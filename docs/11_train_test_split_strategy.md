# 11 — Train / Test Split Strategy (Leakage-Aware)

This document explains **how this repository splits data** for training and evaluation, and **why**.
In incident operations, leakage is easy to introduce (duplicates, near-duplicates, and “future” signals),
so we make split behavior **explicit, reproducible, and auditable**.

---

## TL;DR

We support two split strategies:

1) **Stratified split (default)**  
- Maintains class distribution (good for baselines)  
- Reproducible via `seed` and persisted indices

2) **Time-aware split (recommended for ops realism)**  
- Train on **older** tickets, test on **newer** tickets  
- Prevents “future data” leakage  
- Optional `gap_days` adds a buffer window to reduce near-duplicate leakage

All splits are persisted to disk as **indices** (`train_idx` / `test_idx`) so evaluation is deterministic.

---

## Why splits matter in incident triage

### Leakage patterns in incident data
- **Duplicate tickets** (same text copied across incidents)
- **Near-duplicates** (same incident paraphrased)
- **Temporal leakage** (training sees patterns from future incidents)
- **Policy drift** (priority definitions shift over time)

A model that looks “great” on a leaky split may fail in real operations.
So we validate with leakage-aware splitting and keep evidence in artifacts.

---

## Strategy A — Stratified split (baseline)

**When to use**
- Early-stage baselines
- Small/medium datasets
- When timestamp is missing or unreliable

**How it works**
- Random split with stratification by target label (category or priority)
- Persist indices to `artifacts/.../split.json`

**Pros**
- Stable class balance
- Simple and fast

**Cons**
- Not time-realistic
- Can still be optimistic if duplicates exist across splits

---

## Strategy B — Time-aware split (ops realism)

**When to use**
- You have a valid time column (e.g., `timestamp`)
- You want realistic “future” generalization
- You want stronger leakage protection

**How it works**
1. Parse and sort tickets by `timestamp` ascending  
2. Hold out the latest `test_size` fraction as test  
3. Optional `gap_days` removes the most recent part of training so the model is not trained “too close” to test

**Evidence recorded**
The following fields are stored in `split.json`:
- `strategy: "time"`
- `time_col`
- `cutoff_timestamp`
- `train_max_time`
- `test_min_time`
- `test_max_time`
- `gap_days`

This enables an auditable check:
> `train_max_time < test_min_time`  → no future leakage into training.

---

## Reproducibility & auditability

### Split identity
Each training run writes:
- `artifacts/baseline_{target}/split.json` (indices + evidence)
- `artifacts/baseline_{target}/metrics.json`
- `artifacts/baseline_{target}/meta.json`

### Evaluation behavior
Evaluation uses the persisted indices, so results are reproducible across runs.

---

## How to run

### Stratified (default)
```bash
python -m src.triage.models.train_baseline --target category
python -m src.triage.models.train_baseline --target priority
python -m src.triage.eval.report
```

### Time-aware (recommended)
```bash
python -m src.triage.models.train_baseline --target category --split time --time-col timestamp --gap-days 1
python -m src.triage.models.train_baseline --target priority --split time --time-col timestamp --gap-days 1
python -m src.triage.eval.report
```

---

## Practical guidelines (rules of thumb)

- If you have a timestamp: **prefer time-aware split** for realism.
- If you don’t have a timestamp: use stratified, but **deduplicate** aggressively.
- Use `gap_days` when incidents are likely to repeat over short windows (deployments, ongoing outages).
- Always store split indices. Never rely on “random split in memory” without persistence.

---

## Limitations / future improvements

- **Near-duplicate detection**: gap helps but does not fully remove semantic duplicates.
- **Group-aware splitting**: if you have incident IDs, you can group-split by incident.
- **Drift evaluation**: extend time split into multiple windows (rolling evaluation) for long-term realism.

---

## “Done” checklist (for the GitHub card)

- [ ] Document split strategies (stratified vs time-aware)
- [ ] Persist split indices + evidence in `split.json`
- [ ] Provide reproduction commands
- [ ] Include leakage rationale and auditability notes
